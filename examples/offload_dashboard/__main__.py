"""Offload dashboard: run always-on AI on the NPU and your CPU stays free for your actual work.

The same continuous AI workload (selfie-segmentation background blur + YOLOX-tiny object detection on a video stream,
paced to a target 30 FPS) is placed on CPU, GPU and NPU in turn while a fixed CPU-bound "user work" job (one zlib
compression worker per logical CPU) runs in the foreground. Phases run sequentially:

    baseline (foreground only, AI off) -> AI on CPU -> AI on GPU -> AI on NPU

Run:   .venv\\Scripts\\python.exe -m examples.offload_dashboard
       .venv\\Scripts\\python.exe -m examples.offload_dashboard --no-display --record --phase-seconds 15
       .venv\\Scripts\\python.exe -m examples.offload_dashboard --devices NPU CPU --phase-seconds 10

Writes benchmarks/results/offload_dashboard.json (or --output) and media/offload_dashboard.png; with --record also
media/offload_dashboard.gif (time-compressed, <= ~20 s, < 5 MB) and media/mp4/offload_dashboard.mp4 (real time).
A device that does not compile is reported as failed/skipped; nothing is ever substituted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from examples.common import MEDIA_DIR, MODELS_DIR, REPO_ROOT, RESULTS_DIR
from examples.offload_dashboard import render, telemetry, workers

MODELS = {
    "seg": (str(MODELS_DIR / "mediapipe_selfie_segmentation" / "onnx" / "model.onnx"), [1, 3, 256, 256]),
    "det": (str(MODELS_DIR / "yolox_tiny-int8-ov" / "yolox_tiny.xml"), [1, 3, 416, 416]),
}
SAMPLE_S = 0.5
CLIP_CANDIDATES = ("person", "street")


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def resolve_clip(arg: str | None):
    """--clip path | clip name via examples.common.clips | first available stock clip | None (synthetic)."""
    if arg and Path(arg).exists():
        return str(arg), "file"
    try:
        from examples.common import clips
    except ImportError:
        return None, "examples.common.clips not available"
    names = [arg] if arg else list(CLIP_CANDIDATES)
    if hasattr(clips, "CLIPS"):
        names += [n for n in getattr(clips, "CLIPS") if n not in names]
    for name in names:
        try:
            p = clips.get_clip(name)
            if p and Path(p).exists():
                return str(p), f"clips.get_clip({name!r})"
        except Exception:
            continue
    return None, "no stock clip found"


def cpu_name() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return " ".join(str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).split())
    except OSError:
        return platform.processor() or "CPU"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 2) if xs else None


class Run:
    def __init__(self, args):
        self.args = args
        self.ncpu = psutil.cpu_count() or os.cpu_count()
        self.stop = mp.Event()
        self.device_slot = mp.RawValue("i", -1)
        self.stats = mp.RawArray("d", workers.N_STATS)
        self.frame_buf = mp.RawArray("B", workers.FRAME_W * workers.FRAME_H * 3)
        self.counters = mp.RawArray("d", args.fg_workers)
        self.lat = mp.RawArray("d", 513)
        self.info_q = mp.Queue()
        self.samples = []
        self.jpegs = []  # recorded dashboard frames (JPEG in memory; encoded after the run so it doesn't load the CPU)

    # ---------------------------------------------------------------- process management
    def start_ai(self):
        self.clip, self.clip_note = resolve_clip(self.args.clip)
        self.ai = mp.Process(target=workers.ai_process, daemon=True, name="ai-pipeline",
                             args=(MODELS, self.args.devices, self.clip, self.args.target_fps, self.device_slot,
                                   self.stats, self.frame_buf, self.info_q, self.stop))
        self.ai.start()
        print(f"[offload] compiling models on {self.args.devices} ...", file=sys.stderr, flush=True)
        self.ai_info = self.info_q.get(timeout=600)
        for dev, info in self.ai_info["compile"].items():
            print(f"[offload]   {dev}: " + ", ".join(f"{k}={v['status']} {v.get('execution_devices', v.get('reason', v.get('error', '')))}"[:160]
                                          for k, v in info.items()), file=sys.stderr, flush=True)

    def start_foreground(self):
        self.fg = [mp.Process(target=workers.foreground_worker, args=(i, self.counters, self.stop), daemon=True,
                              name=f"fg-{i}") for i in range(self.args.fg_workers)]
        self.probe = mp.Process(target=workers.probe_process, args=(self.lat, self.stop), daemon=True, name="probe")
        for p in self.fg + [self.probe]:
            p.start()

    def shutdown(self):
        self.stop.set()
        for p in [self.ai] + getattr(self, "fg", []) + [getattr(self, "probe", None)]:
            if p is None:
                continue
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.battery.close()

    # ---------------------------------------------------------------- telemetry
    def _proc_cpu(self, procs):
        total = 0.0
        for p in procs:
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total

    def init_telemetry(self):
        self.gpu = telemetry.GpuEngines()
        self.battery = telemetry.BatteryPower()
        self.p_ai = psutil.Process(self.ai.pid)
        self.p_fg = [psutil.Process(p.pid) for p in self.fg] + [psutil.Process(self.probe.pid)]
        self.p_main = psutil.Process()
        psutil.cpu_percent(None)
        self.last = self._snap()

    def _snap(self):
        return {"t": time.perf_counter(), "fg_ops": sum(self.counters[:]), "frames": self.stats[workers.S_FRAMES],
                "ai_cpu_s": self._proc_cpu([self.p_ai]), "fg_cpu_s": self._proc_cpu(self.p_fg),
                "main_cpu_s": self._proc_cpu([self.p_main]), "lat_idx": int(self.lat[0]),
                "sum_infer": self.stats[workers.S_SUM_INFER], "sum_prepost": self.stats[workers.S_SUM_PREPOST]}

    def sample(self, t_run, phase_idx, t_in_phase):
        now = self._snap()
        d = now["t"] - self.last["t"]
        cap = d * self.ncpu
        k0, k1 = self.last["lat_idx"], now["lat_idx"]
        n = len(self.lat) - 1
        lats = [self.lat[1 + k % n] for k in range(max(k0, k1 - n), k1)]
        acc = self.gpu.sample(self.ai.pid) if self.gpu.ok else {}
        frames = now["frames"] - self.last["frames"]
        dev = self.phases[phase_idx]["device"]
        s = {"t": round(t_run, 3), "phase": phase_idx, "device": dev, "t_in_phase": round(t_in_phase, 3),
             "cpu": psutil.cpu_percent(None),
             "fg_ops_s": (now["fg_ops"] - self.last["fg_ops"]) / d,
             "ai_fps": frames / d if dev else None,
             "ai_proc_cpu": 100 * (now["ai_cpu_s"] - self.last["ai_cpu_s"]) / cap,
             "fg_proc_cpu": 100 * (now["fg_cpu_s"] - self.last["fg_cpu_s"]) / cap,
             "main_proc_cpu": 100 * (now["main_cpu_s"] - self.last["main_cpu_s"]) / cap,
             "ai_cpu_ms_per_frame": 1000 * (now["ai_cpu_s"] - self.last["ai_cpu_s"]) / frames if frames and dev else None,
             "infer_ms": (now["sum_infer"] - self.last["sum_infer"]) / frames if frames and dev else None,
             "prepost_ms": (now["sum_prepost"] - self.last["sum_prepost"]) / frames if frames and dev else None,
             "lag_p95": float(np.percentile(lats, 95)) if lats else None,
             "npu": acc.get("NPU"), "gpu": acc.get("GPU"), "gpu_engine": acc.get("GPU_engine"),
             "npu_proc": acc.get("NPU_proc"), "gpu_proc": acc.get("GPU_proc"),
             "battery_w": self.battery.watts}
        s["other_cpu"] = max(0.0, s["cpu"] - s["ai_proc_cpu"] - s["fg_proc_cpu"] - s["main_proc_cpu"])
        base = self.baseline_ops
        s["fg_pct"] = 100 * s["fg_ops_s"] / base if base else (100.0 if phase_idx == 0 else None)
        self.samples.append(s)
        self.last = now
        return s

    # ---------------------------------------------------------------- phase summary
    def summarize(self, idx):
        ph = self.phases[idx]
        rows = [s for s in self.samples if s["phase"] == idx and s["t_in_phase"] >= self.args.settle]
        if not rows:
            return {}
        r = {"samples": len(rows), "fg_ops_s": _mean([s["fg_ops_s"] for s in rows]),
             "cpu_pct": _mean([s["cpu"] for s in rows]),
             "ai_proc_cpu_pct": _mean([s["ai_proc_cpu"] for s in rows]),
             "fg_proc_cpu_pct": _mean([s["fg_proc_cpu"] for s in rows]),
             "dashboard_proc_cpu_pct": _mean([s["main_proc_cpu"] for s in rows]),
             "other_procs_cpu_pct": _mean([s["other_cpu"] for s in rows]),
             "ui_lag_p95_ms": _mean([s["lag_p95"] for s in rows]),
             "npu_util": _mean([s["npu"] for s in rows]), "gpu_util": _mean([s["gpu"] for s in rows]),
             "npu_util_ai_proc": _mean([s["npu_proc"] for s in rows]),
             "gpu_util_ai_proc": _mean([s["gpu_proc"] for s in rows]),
             "battery_w": _mean([s["battery_w"] for s in rows])}
        if ph["device"] is None:  # live % uses the most recent AI-off phase; finalize() uses both neighbours
            self.baseline_ops = r["fg_ops_s"]
        r["fg_pct"] = round(100 * r["fg_ops_s"] / self.baseline_ops, 1) if self.baseline_ops else None
        if ph["device"]:
            # frame-weighted per-frame costs over the measured window
            t_secs = len(rows) * SAMPLE_S
            fps = [s["ai_fps"] for s in rows]
            r["ai_fps"] = _mean(fps)
            frames = [f * SAMPLE_S for f in fps]
            tot = sum(frames) or 1
            for key, src in (("infer_ms", "infer_ms"), ("prepost_ms", "prepost_ms"),
                             ("ai_proc_cpu_ms_per_frame", "ai_cpu_ms_per_frame")):
                vals = [(s[src], f) for s, f in zip(rows, frames) if s[src] is not None]
                r[key] = round(sum(v * f for v, f in vals) / tot, 2) if vals else None
            r["measured_s"] = t_secs
        return r

    def finalize(self):
        """Normalise each AI phase to its *adjacent* AI-off phases (mean of the one before and the one after).

        Sustained all-core load slowly lowers clocks (heat / power limits), so a single start-of-run baseline drifts;
        bracketing every placement with AI-off phases cancels most of that drift.
        """
        is_base = [p["device"] is None and bool(p.get("result", {}).get("fg_ops_s")) for p in self.phases]
        bases = [p["result"]["fg_ops_s"] for p, b in zip(self.phases, is_base) if b]
        if not bases:
            return
        overall = float(np.mean(bases))
        self.baseline_info = {"fg_ops_s": [round(b, 1) for b in bases], "mean_ops_s": round(overall, 1),
                              "drift_first_to_last_pct": round(100 * (bases[-1] / bases[0] - 1), 1)
                              if len(bases) > 1 else None,
                              "spread_pct": round(100 * (max(bases) - min(bases)) / overall, 1),
                              "method": "each AI phase vs mean of the AI-off phases immediately before and after it"}
        # backward-compatible aliases (pre-interleave JSON schema)
        self.baseline_info["reference_ops_s"] = self.baseline_info["mean_ops_s"]
        self.baseline_info["drift_pct"] = self.baseline_info["drift_first_to_last_pct"]
        for i, p in enumerate(self.phases):
            r = p.get("result", {})
            if not r.get("fg_ops_s"):
                continue
            if p["device"] is None:
                r["fg_pct"] = round(100 * r["fg_ops_s"] / overall, 1)
                continue
            nb = [self.phases[j]["result"]["fg_ops_s"] for j in (i - 1, i + 1)
                  if 0 <= j < len(self.phases) and is_base[j]]
            ref = float(np.mean(nb)) if nb else overall
            r["baseline_ops_s"] = round(ref, 1)
            r["baseline_from"] = "adjacent" if nb else "mean of all AI-off phases"
            r["fg_pct"] = round(100 * r["fg_ops_s"] / ref, 1)
            r["fg_cost_pct"] = round(100 - r["fg_pct"], 1)

    # ---------------------------------------------------------------- dashboard state
    def state(self, idx, t_in_phase, t_run):
        rows = [s for s in self.samples if s["phase"] == idx][-4:]  # ~2 s rolling
        dev = self.phases[idx]["device"]
        kpi = {"fg_pct": _mean([s["fg_pct"] for s in rows]), "ai_fps": _mean([s["ai_fps"] for s in rows]),
               "cpu": _mean([s["cpu"] for s in rows]), "ai_cpu": _mean([s["ai_proc_cpu"] for s in rows]),
               "lag": _mean([s["lag_p95"] for s in rows])}
        if dev:
            infer = self.stats[workers.S_INFER_MS]
            pp = self.stats[workers.S_PREPOST_MS]
            fps = kpi["ai_fps"]
            cap = (f"blur + detect  {fps:.1f} FPS | infer {infer:.1f} ms | pre/post {pp:.1f} ms CPU | "
                   f"{int(self.stats[workers.S_DETS])} obj") if fps else "warming up..."
        else:
            cap = "raw stream - foreground work only"
        hist = {k: [s[k] for s in self.samples] for k in ("fg_pct", "cpu", "ai_fps", "npu", "gpu", "device")}
        hist["ai_cpu"] = [s["ai_proc_cpu"] if s["device"] else None for s in self.samples]
        return {"phases": self.phases, "phase_idx": idx, "phase_frac": min(1, t_in_phase / self.phases[idx]["dur"]),
                "device": dev, "video_caption": cap, "kpi": kpi, "target_fps": self.args.target_fps,
                "t": [s["t"] for s in self.samples], "hist": hist,
                "t_total": self.phases[-1]["t1"], "subtitle": self.subtitle,
                "accel_ok": self.gpu.ok, "placements": self.placements}

    def aggregate(self) -> list[dict]:
        """One entry per placement (AI off, then each device): mean of its phases' results across rounds.

        `rounds` lists the per-round values of the headline metrics so the spread stays visible.
        """
        out = []
        for dev in [None] + list(self.args.devices):
            rs = [p["result"] for p in self.phases if p["device"] == dev and p.get("result", {}).get("samples")]
            if not rs:
                if dev and any(p["device"] == dev and p.get("result", {}).get("status") for p in self.phases):
                    out.append({"label": f"AI on {dev}", "bar": f"AI on\n{dev}", "device": dev,
                                "result": {"status": "compile failed / not available"}})
                continue
            agg = {}
            for k in rs[0]:
                vals = [r.get(k) for r in rs if isinstance(r.get(k), (int, float))]
                if vals:
                    agg[k] = round(float(np.mean(vals)), 2)
            agg["n_phases"] = len(rs)
            agg["rounds"] = {k: [r.get(k) for r in rs] for k in ("fg_pct", "ai_fps", "ai_proc_cpu_ms_per_frame",
                                                                  "ai_proc_cpu_pct", "fg_ops_s")}
            out.append({"label": "AI off" if dev is None else f"AI on {dev}",
                        "bar": "AI off" if dev is None else f"AI on\n{dev}", "device": dev, "result": agg})
        return out

    # ---------------------------------------------------------------- main loop
    def run(self):
        a = self.args
        self.start_ai()
        ready = self.ai_info["ready_devices"]
        off = lambda: {"label": "AI off", "short": "off", "bar": "AI off", "device": None,  # noqa: E731
                       "dur": a.baseline_seconds}
        self.phases = [off()]
        for rnd in range(a.rounds):
            order = a.devices if rnd % 2 == 0 else list(reversed(a.devices))  # alternate order: cancels drift bias
            for d in order:
                self.phases.append({"label": f"AI on {d}", "short": d, "bar": f"AI on\n{d}", "device": d,
                                    "round": rnd + 1, "dur": a.phase_seconds,
                                    "status": "ok" if d in ready else "compile failed / not available"})
                if not a.no_interleave:
                    self.phases.append(off())
        if a.no_interleave:
            self.phases.append(off())
        self.placements = []
        t = 0.0
        for ph in self.phases:
            ph["t0"], ph["t1"] = t, t + ph["dur"]
            t += ph["dur"]
        self.baseline_ops = None
        self.start_foreground()
        self.init_telemetry()
        src = self.ai_info["source"]
        self.subtitle = (f"{cpu_name()} | {self.ncpu} logical CPUs | "
                         f"{a.fg_workers} zlib workers = 'your work' | video: {src} | power: {self.battery.note}")
        time.sleep(3.0)  # let the foreground workers spin up (and process-start scanning settle)
        self.last = self._snap()
        if not a.no_display:
            cv2.namedWindow("offload_dashboard", cv2.WINDOW_AUTOSIZE)
        video = np.frombuffer(self.frame_buf, dtype=np.uint8).reshape(workers.FRAME_H, workers.FRAME_W, 3)
        t_start = time.perf_counter()
        quit_ = False
        for idx, ph in enumerate(self.phases):
            if ph["device"] and ph["device"] not in ready:
                ph["result"] = {"status": ph["status"]}
                print(f"[offload] phase {ph['label']}: SKIPPED ({ph['status']})", file=sys.stderr, flush=True)
                continue
            self.device_slot.value = workers.DEVICES.index(ph["device"]) if ph["device"] else -1
            ph["started_s"] = round(time.perf_counter() - t_start, 3)
            t_phase = time.perf_counter()
            self.last = self._snap()
            next_sample, next_frame = t_phase + SAMPLE_S, t_phase
            while True:
                now = time.perf_counter()
                tin = now - t_phase
                if tin >= ph["dur"]:
                    break
                if now >= next_sample:
                    self.sample(ph["t0"] + tin, idx, tin)  # planned timeline position
                    next_sample += SAMPLE_S
                if now >= next_frame:
                    next_frame += 1 / a.ui_fps
                    frame = render.dashboard_frame(video.copy(), self.state(idx, tin, now - t_start))
                    self._emit(frame)
                    if a.snapshot_dir and ph["device"] == "NPU" and tin >= 0.75 * ph["dur"] \
                            and not getattr(self, "_snapped", False):
                        a.snapshot_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(a.snapshot_dir / "offload_dashboard_live.png"), frame)
                        self._snapped = True
                    if not a.no_display:
                        cv2.imshow("offload_dashboard", frame)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            quit_ = True
                            break
                time.sleep(max(0, min(next_sample, next_frame) - time.perf_counter()))
            ph["result"] = self.summarize(idx)
            self.finalize()  # provisional: the newest AI phase only has its preceding AI-off neighbour so far
            self.placements = self.aggregate()
            r = ph["result"]
            print(f"[offload] phase {ph['label']}: fg {r.get('fg_pct')}% ({r.get('fg_ops_s')} ops/s) "
                  f"AI {r.get('ai_fps')} FPS cpu {r.get('cpu_pct')}% ai_proc {r.get('ai_proc_cpu_pct')}% "
                  f"npu {r.get('npu_util')}% gpu {r.get('gpu_util')}% prepost {r.get('prepost_ms')} ms "
                  f"lag {r.get('ui_lag_p95_ms')} ms", file=sys.stderr, flush=True)
            if quit_:
                break
        self.device_slot.value = -1
        self.finalize()
        self.placements = self.aggregate()
        summary = render.summary_frame(self.placements, self.subtitle, a.target_fps)
        if a.snapshot_dir:
            a.snapshot_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(a.snapshot_dir / "offload_dashboard_summary.png"), summary)
        n_hold = int(a.ui_fps * 4)
        for _ in range(n_hold):
            self._emit(summary)
        if not a.no_display:
            cv2.imshow("offload_dashboard", summary)
            cv2.waitKey(4000)
            cv2.destroyAllWindows()
        self.shutdown()
        return summary

    def _emit(self, frame):
        if self.args.record:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.jpegs.append(buf)


def headline(placements):
    res = {p["device"]: p["result"] for p in placements if p["device"] and p.get("result", {}).get("ai_fps")}
    if "NPU" not in res:
        return "Offload dashboard: same always-on AI on each device vs. foreground work"
    n = res["NPU"]
    s = (f"AI on NPU: {n['ai_fps']:.0f} FPS for {n['ai_proc_cpu_ms_per_frame']:.0f} ms CPU/frame, "
         f"your work keeps {n['fg_pct']:.0f}%")
    if "CPU" in res:
        c = res["CPU"]
        s += (f"  |  AI on CPU: {c['ai_fps']:.0f} FPS for {c['ai_proc_cpu_ms_per_frame']:.0f} ms CPU/frame, "
              f"work keeps {c['fg_pct']:.0f}%")
    return s


def save_media(run: Run, stem: str) -> dict:
    from examples.common.media import write_gif, write_mp4

    out = {}
    if not run.jpegs:
        return out
    n_hold = int(run.args.ui_fps * 4)
    live = run.jpegs[:-n_hold]
    hold = run.jpegs[-1]
    dec = lambda b: cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # noqa: E731

    class _Lazy:  # decode on demand so the whole recording never sits decoded in RAM
        def __init__(self, bufs):
            self.bufs = bufs

        def __iter__(self):
            return (dec(b) for b in self.bufs)

    out["mp4"] = _rel(write_mp4(_Lazy(run.jpegs), MEDIA_DIR / "mp4" / f"{stem}.mp4", run.args.ui_fps))
    # GIF: time-compress the live part to <= 16 s at 8 fps, then hold the summary for 3.5 s
    gif_fps = 8
    n_live = min(len(live), 16 * gif_fps)
    pick = np.linspace(0, len(live) - 1, n_live).round().astype(int)
    frames = [cv2.resize(dec(live[i]), (720, 405), interpolation=cv2.INTER_AREA) for i in pick]
    s = cv2.resize(dec(hold), (720, 405), interpolation=cv2.INTER_AREA)
    frames += [s] * int(3.5 * gif_fps)
    p = write_gif(frames, MEDIA_DIR / f"{stem}.gif", gif_fps, 720, quantize="fastoctree")
    out["gif"] = _rel(p)
    out["gif_mb"] = round(p.stat().st_size / 1e6, 2)
    out["gif_seconds"] = round(len(frames) / gif_fps, 1)
    out["gif_speedup"] = round(len(live) / run.args.ui_fps / (n_live / gif_fps), 1) if n_live else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-seconds", type=float, default=15.0)
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"], choices=workers.DEVICES)
    ap.add_argument("--record", action="store_true", help="write media/offload_dashboard.gif + media/mp4/*.mp4")
    ap.add_argument("--no-display", action="store_true", help="headless: no OpenCV window")
    ap.add_argument("--output", type=Path, default=RESULTS_DIR / "offload_dashboard.json")
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--fg-workers", type=int, default=psutil.cpu_count() or os.cpu_count(),
                    help="foreground zlib worker processes (default: one per logical CPU, i.e. a machine-saturating job)")
    ap.add_argument("--baseline-seconds", type=float, default=None,
                    help="length of each AI-off phase (default: 2/3 of --phase-seconds)")
    ap.add_argument("--rounds", type=int, default=2,
                    help="repeat every placement this many times, alternating the order (default 2: CPU,GPU,NPU then "
                         "NPU,GPU,CPU); results are averaged per placement")
    ap.add_argument("--no-interleave", action="store_true",
                    help="only one AI-off phase at the start and one at the end, instead of between every placement")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds at the start of each phase excluded from stats")
    ap.add_argument("--ui-fps", type=float, default=10.0, help="dashboard render/record rate")
    ap.add_argument("--clip", default=None, help="video file or examples.common.clips name (default: first stock clip)")
    ap.add_argument("--png", type=Path, default=MEDIA_DIR / "offload_dashboard.png")
    ap.add_argument("--snapshot-dir", type=Path, default=None,
                    help="also save one live dashboard frame (late in the NPU phase) and the summary frame as PNGs")
    a = ap.parse_args()
    if a.baseline_seconds is None:
        a.baseline_seconds = round(a.phase_seconds * 2 / 3, 1)

    import openvino as ov

    run = Run(a)
    t0 = time.time()
    run.run()
    data = {
        "example": "offload_dashboard",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "system": {"cpu": cpu_name(), "logical_cpus": run.ncpu, "openvino": ov.__version__,
                   "python": platform.python_version()},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "workload": {
            "ai": "per frame (640x360 BGR): selfie segmentation 256x256 + YOLOX-tiny 416x416 (sync, sequential), "
                  "CPU pre-processing, background blur composite, NMS + box drawing; paced to target_fps",
            "foreground": f"{a.fg_workers} processes, each repeating zlib.compress(level 6) of a fixed 384 KB buffer",
            "probe": "1 process requesting 2 ms sleeps; lateness p95 per 0.5 s window = 'UI lag'",
            "models": {k: _rel(v[0]) for k, v in MODELS.items()},
        },
        "video_source": {"source": run.ai_info["source"], "how": run.clip_note, "frames": run.ai_info["n_frames"]},
        "compile": run.ai_info["compile"],
        "telemetry": {"cpu": "psutil.cpu_percent (system) + per-process cpu_times deltas",
                      "accelerators": run.gpu.describe(), "power": run.battery.note},
        "baseline": getattr(run, "baseline_info", None),
        "placements": run.placements,
        "phases": run.phases,
        "samples": run.samples,
        "headline": headline(run.placements),
    }
    data["footnote"] = (f"{data['system']['cpu'] or 'CPU'} ({run.ncpu} threads), OpenVINO {ov.__version__}. "
                        f"Foreground: {a.fg_workers} zlib workers (one per logical CPU). AI: seg + YOLOX-tiny, "
                        f"{a.target_fps:.0f} FPS target, pre/post on CPU in every placement. {a.rounds} rounds x "
                        f"{a.phase_seconds:.0f} s per placement, each bracketed by {a.baseline_seconds:.0f} s AI-off "
                        f"phases (foreground % vs those neighbours); first {a.settle:.0f} s of each phase excluded. "
                        f"Power: {run.battery.note}. Video: {run.ai_info['source']}.")
    data["media"] = {"png": _rel(render.static_chart(data, a.png))}
    if a.record:
        data["media"].update(save_media(run, "offload_dashboard"))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=1, default=str))
    print(json.dumps({"output": _rel(a.output), "media": data["media"], "headline": data["headline"],
                      "elapsed_s": round(time.time() - t0, 1),
                      "baseline": data["baseline"],
                      "placements": [{"placement": p["label"], **{k: v for k, v in p.get("result", {}).items()
                                                                  if k in ("ai_fps", "fg_pct", "cpu_pct",
                                                                           "ai_proc_cpu_pct", "ai_proc_cpu_ms_per_frame",
                                                                           "npu_util", "gpu_util", "prepost_ms",
                                                                           "ui_lag_p95_ms", "status", "rounds")}}
                                     for p in run.placements]}, indent=1))


if __name__ == "__main__":
    main()
