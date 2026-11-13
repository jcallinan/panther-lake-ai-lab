"""Live vision side-by-side: the same video frames through NPU, GPU and CPU, drawn next to each other.

Each device gets its own compiled models and InferRequests: YOLOX-Tiny INT8 detection and/or MediaPipe
selfie segmentation (background blur). Every panel shows what that device produced, plus its measured
inference latency (p50/p95 and a sparkline) and the EXECUTION_DEVICES OpenVINO reports. Nothing ever
falls back to another device. A device that fails to compile is shown as failed.

Timing modes:
  default       sequential: on every frame each device runs alone, one after another, on the same input
                (device order rotates each frame). This keeps devices from perturbing each other's timing.
  --concurrent  all devices start at once (one AsyncInferQueue per device and model) and contend for
                memory bandwidth and host CPU. Latency is start_async -> completion callback.
Timed regions cover inference only. Decode, pre/post-processing and drawing are excluded, and the
end-to-end pipeline FPS is reported separately.

Run:  .venv\\Scripts\\python.exe -m examples.vision_side_by_side
      .venv\\Scripts\\python.exe -m examples.vision_side_by_side --source person --task segment
      .venv\\Scripts\\python.exe -m examples.vision_side_by_side --no-display --record --duration 15
      .venv\\Scripts\\python.exe -m examples.vision_side_by_side --source 0          (webcam, never recorded
                                                                                    unless --record)
"""
import argparse
import collections
import datetime as dt
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from examples.common import MEDIA_DIR, REPO_ROOT, RESULTS_DIR, hud, ovutil
from examples.common.clips import CLIPS, resolve_source
from examples.vision_side_by_side import pipeline, render

NAME = "vision_side_by_side"
CELL_W = 640
LIVE_WINDOW = 120  # frames used for on-screen p50/p95
GIF_WIDTH, GIF_FPS, GIF_MAX_BYTES = 960, 6, 4_900_000  # README GIF must stay < 5 MB


def log(msg: str):
    print(f"[{NAME}] {msg}", file=sys.stderr, flush=True)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="street",
                   help=f"clip name ({', '.join(CLIPS)}), video file path, or webcam index (default street)")
    p.add_argument("--devices", nargs="+", default=["NPU", "GPU", "CPU"])
    p.add_argument("--task", choices=["detect", "segment", "both"], default="both")
    p.add_argument("--concurrent", action="store_true", help="run all devices at once (AsyncInferQueue per device)")
    p.add_argument("--record", action="store_true", help="write media/vision_side_by_side.gif + media/mp4/...mp4")
    p.add_argument("--duration", type=float, default=None, help="seconds to run (default 15 when recording)")
    p.add_argument("--no-display", action="store_true", help="headless: no window")
    p.add_argument("--conf", type=float, default=0.35, help="detection score threshold")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / f"{NAME}.json", help="results JSON")
    p.add_argument("--snapshot", type=Path, default=None, help="also save the last composite frame as PNG")
    a = p.parse_args()
    a.devices = [d.upper() for d in a.devices]
    if a.record and a.duration is None:
        a.duration = 15.0
    return a


def open_source(source: str, record: bool):
    target, clip, kind = resolve_source(source)
    if kind == "webcam":
        if record:
            log("WARNING: --record is set with a WEBCAM source. Everything the camera sees will be written to "
                "media/. Stop now (Ctrl+C) if that is not intended.")
        else:
            log("webcam input: live only, nothing is recorded (pass --record explicitly to record)")
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {source!r}")
    return cap, clip, kind


def device_names() -> dict:
    out = {}
    for d in ovutil.core().available_devices:
        try:
            out[d] = ovutil.core().get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            out[d] = "?"
    return out


def compose(panels: list[np.ndarray], stats: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    frames = panels + [stats]
    cols = 2 if len(frames) <= 4 else 3
    return hud.tile(frames, cols, cell_w, cell_h, gap=4)


def main():
    a = parse_args()
    tasks = ["detect", "segment"] if a.task == "both" else [a.task]
    cap, clip, kind = open_source(a.source, a.record)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("source produced no frames")
    fh, fw = frame.shape[:2]
    cell_h = int(round(CELL_W * fh / fw / 2) * 2)
    panel_scale = CELL_W / fw

    log(f"OpenVINO {ov.__version__} | devices {a.devices} | task {a.task} | "
        f"mode {'concurrent' if a.concurrent else 'sequential'} | source {a.source} ({fw}x{fh} @ {src_fps:.1f})")
    models = {t: pipeline.load_model(t) for t in tasks}
    runners = []
    for d in a.devices:
        r = pipeline.DeviceRunner(d, tasks, concurrent=a.concurrent)
        ok_dev = r.compile(models)
        for t in tasks:
            i = r.info[t]
            log(f"  {d:4s} {t:8s} {i['status']:9s} compile {i.get('compile_s', '-')} s  "
                f"EXECUTION_DEVICES={i.get('execution_devices', i.get('error', i.get('reason')))}")
        r.ok = ok_dev
        runners.append(r)
    live = [r for r in runners if r.ok]
    if not live:
        raise SystemExit("no device compiled; nothing to run")

    # warm-up on the first frame (not recorded)
    det_in, det_scale = pipeline.preprocess_detect(frame)
    inputs = {"detect": det_in, "segment": pipeline.preprocess_segment(frame)}
    for r in live:
        for _ in range(5):
            if a.concurrent:
                r.start_async(inputs)
                r.wait_async(record=False)
            else:
                r.run_sequential(inputs, record=False)

    rec = None
    if a.record:
        from examples.common.media import Recorder
        rec = Recorder(NAME, fps=src_fps)
    if not a.no_display:
        cv2.namedWindow(NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(NAME, 2 * CELL_W + 4, 2 * cell_h + 4)

    mode_line = ("sequential: each device timed alone on the same frame" if not a.concurrent
                 else "concurrent: all devices run at once (AsyncInferQueue)")
    exec_text = {r.device: "/".join(sorted({e for v in r.execution_devices().values() for e in v})) for r in runners}
    credit = (f"clip: {clip.name} - {clip.author[:38]} ({clip.license.split(' (')[0]}), Wikimedia Commons"
              if clip else f"source: {'webcam' if kind == 'webcam' else Path(a.source).name}")
    pipe_fps = ovutil.RollingFPS(alpha=0.05)
    frames_done, t_start, rotate = 0, time.perf_counter(), 0
    composite = None
    try:
        while True:
            if frames_done:
                ok, frame = cap.read()
                if not ok:
                    if kind == "webcam":
                        break
                    if a.no_display and a.duration is None:
                        break  # single pass when headless without a duration
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break
            # ---- pre-processing: once, shared by every device (untimed)
            inputs = {}
            if "detect" in tasks:
                inputs["detect"], det_scale = pipeline.preprocess_detect(frame)
            if "segment" in tasks:
                inputs["segment"] = pipeline.preprocess_segment(frame)
            # ---- inference (timed inside the runners)
            if a.concurrent:
                for r in live:
                    r.start_async(inputs)
                for r in live:
                    r.wait_async()
            else:
                order = live[rotate % len(live):] + live[:rotate % len(live)]
                rotate += 1
                for r in order:
                    r.run_sequential(inputs)
            # ---- post-processing + drawing (untimed)
            base = cv2.resize(frame, (CELL_W, cell_h), interpolation=cv2.INTER_AREA)
            bg_layer = render.background_layer(base) if "segment" in tasks else None
            vmax = max([max(list(r.times["frame"])[-LIVE_WINDOW:]) for r in live] + [1.0])
            panels, rows = [], []
            for r in runners:
                color = hud.device_color(r.device)
                if not r.ok:
                    img = (base * 0.35).astype(np.uint8)
                    err = next((r.info[t] for t in tasks if r.info[t]["status"] != "compiled"), {})
                    hud.device_badge(img, r.device, [f"{err.get('status', 'failed')}",
                                                     (err.get("error_type") or err.get("reason") or "")[:22]])
                    panels.append(img)
                    rows.append({"device": r.device, "status": err.get("status", "failed")})
                    continue
                img = base.copy()
                n_obj = None
                if "segment" in tasks:
                    mask = pipeline.postprocess_segment(r.outputs["segment"][0], (CELL_W, cell_h))
                    img = render.apply_segmentation(img, bg_layer, mask, color)
                if "detect" in tasks:
                    dets = pipeline.postprocess_detect(r.outputs["detect"][0][0], r.outputs["detect"][1][0],
                                                       det_scale, conf=a.conf)
                    render.draw_detections(img, dets, panel_scale, color)
                    n_obj = len(dets)
                recent = list(r.times["frame"])[-LIVE_WINDOW:]
                p50, p95 = float(np.median(recent)), float(np.percentile(recent, 95))
                lines = [f"p50 {p50:.1f}  p95 {p95:.1f} ms",
                         f"{1000 / p50:.0f} inf/s" + (f"   {n_obj} objects" if n_obj is not None else "")]
                render.badge(img, r.device, f"exec: {exec_text[r.device]}", lines, recent[-206:], vmax)
                panels.append(img)
                rows.append({"device": r.device, "p50": p50, "p95": p95, "fps": 1000 / p50})
            pipe_fps.tick()
            frames_done += 1
            header = ["Same frames, three engines",
                      f"{' + '.join(tasks)} | {mode_line}",
                      f"pipeline {pipe_fps.fps or 0:.1f} FPS end-to-end (decode, pre/post, draw, all devices)"]
            footer = ["YOLOX-Tiny INT8 (Apache-2.0)" * ("detect" in tasks) +
                      ("  +  " if len(tasks) == 2 else "") +
                      "MediaPipe selfie seg (Apache-2.0)" * ("segment" in tasks),
                      f"OpenVINO {ov.__version__} | latency excludes pre/post-processing",
                      credit]
            stats = render.stats_panel(CELL_W, cell_h, rows, header, footer)
            composite = compose(panels, stats, CELL_W, cell_h)
            if rec is not None:
                rec.add(composite, bgr=True)
            if not a.no_display:
                cv2.imshow(NAME, composite)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if a.duration is not None and time.perf_counter() - t_start >= a.duration:
                break
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        cap.release()
        if not a.no_display:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    pipeline_fps = frames_done / wall if wall > 0 else 0.0
    media = {}
    if rec is not None and rec.frames:
        rec.fps = max(1.0, min(src_fps, pipeline_fps))  # play back at the rate frames were actually produced
        media = {k: _rel(v) for k, v in rec.save(gif=False).items()}
        from examples.common.media import write_gif
        step = max(1, round(rec.fps / GIF_FPS))
        gif = write_gif(rec.frames[::step], MEDIA_DIR / f"{NAME}.gif", rec.fps / step, GIF_WIDTH,
                        max_bytes=GIF_MAX_BYTES, quantize="fastoctree")
        media["gif"] = _rel(gif)
        media["gif_mb"] = round(gif.stat().st_size / 1e6, 2)
    if a.snapshot and composite is not None:
        a.snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.snapshot), composite)

    result = {
        "tool": f"examples.{NAME}", "date": dt.datetime.now().isoformat(timespec="seconds"),
        "openvino_version": ov.__version__, "device_names": device_names(),
        "mode": "concurrent" if a.concurrent else "sequential",
        "mode_note": ("all devices started together per frame via one AsyncInferQueue(jobs=1) per device and "
                      "model; latency = start_async -> completion callback; frame = slowest task on that device"
                      if a.concurrent else
                      "per frame each device runs alone (InferRequest.infer), device order rotates each frame; "
                      "frame latency = sum of the device's task latencies"),
        "timing_excludes": "video decode, pre-processing, NMS / mask post-processing, drawing",
        "task": a.task, "models": {t: pipeline.MODEL_INFO[t] for t in tasks},
        "source": {"arg": a.source, "kind": kind, "resolution": [fw, fh], "fps": round(src_fps, 2),
                   **({"clip": clip.as_dict()} if clip else {})},
        "frames": frames_done, "wall_s": round(wall, 2), "pipeline_fps": round(pipeline_fps, 2),
        "detection_conf": a.conf,
        "devices": {r.device: {"status": "ok" if r.ok else "failed",
                               "compile": r.info,
                               "execution_devices": r.execution_devices(),
                               "latency_ms": r.stats() if r.ok else {}} for r in runners},
        "media": media,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    summary = {"output": _rel(a.output), "frames": frames_done, "pipeline_fps": result["pipeline_fps"],
               "media": media,
               "frame_latency_ms": {d: {k: v["latency_ms"].get("frame", {}).get(k) for k in ("median_ms", "p95_ms")}
                                    for d, v in result["devices"].items() if v["status"] == "ok"},
               "execution_devices": {d: v["execution_devices"] for d, v in result["devices"].items()}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
