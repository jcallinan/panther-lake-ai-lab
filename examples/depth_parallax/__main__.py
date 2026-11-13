"""Depth Parallax: NPU monocular depth turns flat video into an orbiting 2.5D view.

Per frame: Depth Anything V2 Small (static 1x3x364x364) runs on --device (default NPU) and the CPU
re-projects the frame to a slowly orbiting virtual camera using that depth. The next frame's inference
is started before the current frame is rendered, so device and host work overlap; model latency is
the device-reported InferRequest latency and is shown separately from host (pre/post/render) time.

Run:   .venv\\Scripts\\python.exe -m examples.depth_parallax                       (default clip, live window)
       .venv\\Scripts\\python.exe -m examples.depth_parallax --source my.mp4
       .venv\\Scripts\\python.exe -m examples.depth_parallax --image photo.jpg      (3D-photo Ken Burns)
       .venv\\Scripts\\python.exe -m examples.depth_parallax --record --no-display  (media/depth_parallax.gif)

Devices are never substituted: if --device cannot compile the model the demo exits with the error,
and the comparison inset marks devices that are missing or failed instead of hiding them.
"""
import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from examples.common import MODELS_DIR, REPO_ROOT, RESULTS_DIR, hud, ovutil
from examples.depth_parallax import render as R

MODEL_PATH = MODELS_DIR / "depth-anything-v2-small" / "onnx" / "model.onnx"
MODEL_NAME = "Depth Anything V2 Small"
MODEL_LICENSE = "Apache-2.0 (onnx-community/depth-anything-v2-small)"
CACHE_DIR = MODELS_DIR / "npucache"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_CLIP = "forest"         # stock clip (examples.common.clips) with strong receding depth

CANVAS_W, CANVAS_H = 1280, 720
TITLE_H = 44
HERO_W, HERO_H = 896, 504
SIDE_X = HERO_W + 6
SIDE_W = CANVAS_W - SIDE_X
SIDE_H = round(SIDE_W * 9 / 16)
BG = (18, 14, 12)


def log(msg: str):
    print(f"[depth_parallax] {msg}", file=sys.stderr, flush=True)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- sources
class StillSource:
    def __init__(self, path: Path):
        self.frame = cv2.imread(str(path))
        if self.frame is None:
            raise SystemExit(f"cannot read image {path}")
        self.fps = 24.0

    def read(self):
        return self.frame.copy()

    def release(self):
        pass


class VideoSource:
    """Video file or webcam; files loop so recordings can run for any duration."""

    def __init__(self, spec, loop: bool = True):
        self.cap = cv2.VideoCapture(spec)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open video source {spec!r}")
        self.loop = loop and not isinstance(spec, int)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and 1 <= fps <= 120 else 30.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


def _clips():
    try:
        from examples.common import clips
        return clips
    except ImportError:
        return None


def clip_license(name: str) -> dict:
    """License/attribution for a registered stock clip (examples.common.clips)."""
    clips = _clips()
    clip = getattr(clips, "CLIPS", {}).get(name) if clips else None
    info = clip.as_dict() if clip is not None and hasattr(clip, "as_dict") else {"name": name}
    info["attribution_file"] = "assets/CLIPS_ATTRIBUTION.md"
    return info


def open_source(args):
    """Returns (source, still, description dict)."""
    if args.image:
        return StillSource(args.image), True, {"kind": "image", "path": _rel(args.image),
                                                 "license": "user-supplied"}
    spec = args.source
    if spec is not None and spec.isdigit():
        return VideoSource(int(spec)), False, {"kind": "webcam", "index": int(spec)}
    if spec is not None and Path(spec).exists():
        return VideoSource(str(spec)), False, {"kind": "file", "path": _rel(spec), "license": "user-supplied"}
    clips = _clips()
    if clips is None:
        raise SystemExit("examples.common.clips is not available: pass --source <video file> or --image <path>")
    names = [c["name"] for c in clips.list_clips()]
    name = spec or DEFAULT_CLIP
    if name not in names:
        raise SystemExit(f"--source {name!r} is not a file, webcam index or stock clip ({', '.join(names)})")
    path = clips.get_clip(name)
    return VideoSource(str(path)), False, {"kind": "clip", "path": _rel(path), **clip_license(name)}


# ---------------------------------------------------------------- model
def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    x = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) * (1 / 255.0)
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def load_model(size: int) -> ov.Model:
    if not MODEL_PATH.exists():
        raise SystemExit(f"model not found: {MODEL_PATH} (see README for download instructions)")
    model = ovutil.core().read_model(str(MODEL_PATH))
    model.reshape({model.inputs[0]: ov.PartialShape([1, 3, size, size])})
    return model


def _cache_files() -> set:
    return {p.name for p in CACHE_DIR.glob("*")} if CACHE_DIR.exists() else set()


def compile_device(model, device: str, splash=None) -> tuple:
    """Compile with the blob cache; info records compile_s and whether the cache was cold or warm."""
    config = {}
    if device in ("NPU", "GPU"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(CACHE_DIR)
    if splash:
        note = " (first NPU compile takes 15-35 s; cached in models/npucache after)" if device == "NPU" else ""
        splash(f"Compiling depth model for {device}...{note}")
    log(f"compiling for {device} {'with CACHE_DIR' if config else ''}...")
    before = _cache_files()
    compiled, info = ovutil.compile_on(model, device, config)
    if config:
        info["cache_dir"] = _rel(CACHE_DIR)
        info["cache"] = "cold (blob written)" if _cache_files() - before else "warm (loaded from cache)"
    log(f"{device}: {info['status']} compile_s={info.get('compile_s')} {info.get('cache', '')} "
        f"exec={info.get('execution_devices')} {info.get('error', '')[:300]}")
    return compiled, info


def bench(compiled, x: np.ndarray, device: str) -> dict:
    req = compiled.create_infer_request()
    req.set_input_tensor(ov.Tensor(x))
    runs = 12 if device == "CPU" else 40
    return ovutil.time_request(req, runs=runs, warmup=3)


# ---------------------------------------------------------------- drawing
def text(frame, s: str, x: int, y: int, scale: float = 0.6, color=(255, 255, 255), thickness: int = 1):
    """Drop-shadow text. (hud.text's thicker outline pass renders wider glyphs with OpenCV 5's font,
    leaving ghost characters after the string; a same-thickness offset shadow avoids that.)"""
    cv2.putText(frame, s, (x + 1, y + 1), hud.FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), hud.FONT, scale, color, thickness, cv2.LINE_AA)


def device_badge(frame, device: str, lines: list, x: int = 10, y: int = 10, w: int = 250):
    """Same layout as hud.device_badge, drawn with the shadow text above."""
    color = hud.device_color(device)
    h = 34 + 24 * len(lines)
    hud.panel(frame, x, y, w, h)
    cv2.rectangle(frame, (x, y), (x + 6, y + h), color, -1)
    text(frame, device, x + 16, y + 26, 0.8, color, 2)
    for i, line in enumerate(lines):
        text(frame, line, x + 16, y + 52 + 24 * i, 0.55)


def compare_inset(canvas, x, y, w, h, results: dict, live_device: str):
    hud.panel(canvas, x, y, w, h, 0.0)
    canvas[y:y + h, x:x + w] = (26, 22, 20)
    text(canvas, "Same model, same input: median latency", x + 12, y + 24, 0.5, (230, 230, 230))
    ok = {d: r["median_ms"] for d, r in results.items() if r.get("median_ms")}
    vmax = max(ok.values()) if ok else 1.0
    cpu = ok.get("CPU")
    row_h = 46
    for i, dev in enumerate(("NPU", "GPU", "CPU")):
        r = results.get(dev, {"status": "not run"})
        ry = y + 42 + i * row_h
        col = hud.device_color(dev)
        text(canvas, dev, x + 12, ry + 20, 0.6, col, 1)
        bx, bw = x + 62, w - 74
        if r.get("median_ms"):
            ms = r["median_ms"]
            cv2.rectangle(canvas, (bx, ry + 4), (bx + bw, ry + 24), (40, 36, 34), -1)
            cv2.rectangle(canvas, (bx, ry + 4), (bx + max(3, int(bw * ms / vmax)), ry + 24), col, -1)
            label = f"{ms:.1f} ms  ({1000 / ms:.0f}/s)"
            if cpu and dev != "CPU":
                label += f"  {cpu / ms:.1f}x CPU"
            text(canvas, label, bx, ry + 42, 0.45, (220, 220, 220))
        else:
            text(canvas, str(r.get("status", "n/a")), bx, ry + 20, 0.5, (150, 150, 150))
        if dev == live_device:
            cv2.rectangle(canvas, (x + 4, ry), (x + w - 4, ry + row_h - 2), col, 1)


def orbit_indicator(canvas, cx, cy, r, yaw, pitch, dolly):
    cv2.circle(canvas, (cx, cy), r, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx - r, cy), (cx + r, cy), (60, 60, 60), 1)
    cv2.line(canvas, (cx, cy - r), (cx, cy + r), (60, 60, 60), 1)
    px, py = int(cx + yaw / 0.12 * r), int(cy - pitch / 0.06 * r)
    cv2.circle(canvas, (px, py), 4 + int(30 * dolly), (255, 255, 255), -1, cv2.LINE_AA)
    text(canvas, f"camera {math.degrees(yaw):+.0f} deg", cx - r - 14, cy + r + 18, 0.42, (200, 200, 200))


def build_canvas(src, depth_bgr, hero, title, sub, badge_dev, badge_lines, cmp_results, live_dev,
                 timing_line, cam, lat_hist):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
    text(canvas, title, 14, 30, 0.75, (255, 255, 255), 1)
    (tw, _), _ = cv2.getTextSize(sub, hud.FONT, 0.5, 1)
    text(canvas, sub, CANVAS_W - tw - 14, 29, 0.5, (190, 190, 190))
    canvas[TITLE_H:TITLE_H + HERO_H, :HERO_W] = hero
    y = TITLE_H
    canvas[y:y + SIDE_H, SIDE_X:] = cv2.resize(src, (SIDE_W, SIDE_H), interpolation=cv2.INTER_AREA)
    hud.panel(canvas, SIDE_X, y + SIDE_H - 26, SIDE_W, 26, 0.55)
    text(canvas, "input frame (flat 2D)", SIDE_X + 8, y + SIDE_H - 8, 0.48)
    y2 = y + SIDE_H + 6
    canvas[y2:y2 + SIDE_H, SIDE_X:] = cv2.resize(depth_bgr, (SIDE_W, SIDE_H), interpolation=cv2.INTER_AREA)
    hud.panel(canvas, SIDE_X, y2 + SIDE_H - 26, SIDE_W, 26, 0.55)
    text(canvas, f"{live_dev} depth: bright = near", SIDE_X + 8, y2 + SIDE_H - 8, 0.48,
             hud.device_color(live_dev))
    cy = y2 + SIDE_H + 6
    compare_inset(canvas, SIDE_X, cy, SIDE_W, CANVAS_H - cy, cmp_results, live_dev)
    # hero overlays
    device_badge(canvas, badge_dev, badge_lines, 10, TITLE_H + 10)
    orbit_indicator(canvas, HERO_W - 60, TITLE_H + 50, 30, *cam)
    hud.panel(canvas, 0, TITLE_H + HERO_H - 30, HERO_W, 30, 0.5)
    text(canvas, "2.5D re-projection to an orbiting virtual camera (pixels moved by NPU depth)",
             10, TITLE_H + HERO_H - 10, 0.5, (235, 235, 235))
    # bottom strip
    by = TITLE_H + HERO_H + 8
    text(canvas, timing_line, 14, by + 22, 0.55, (225, 225, 225))
    text(canvas, "model latency (ms), last 240 frames", 14, by + 52, 0.45, (160, 160, 160))
    if len(lat_hist) > 1:
        vmax = max(30.0, float(np.percentile(lat_hist, 98)) * 1.2)
        cv2.rectangle(canvas, (14, by + 60), (14 + 480, by + 150), (30, 26, 24), -1)
        hud.sparkline(canvas, lat_hist, 14, by + 62, 480, 86, hud.device_color(live_dev), vmax)
        text(canvas, f"{vmax:.0f}", 500, by + 72, 0.4, (150, 150, 150))
        text(canvas, "0", 500, by + 150, 0.4, (150, 150, 150))
    text(canvas, f"{MODEL_NAME}  |  {MODEL_LICENSE.split(' ')[0]}  |  OpenVINO {ov.__version__.split('-')[0]}",
             540, by + 150, 0.42, (150, 150, 150))
    return canvas


def splash_frame(msg: str) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
    text(canvas, "Depth Parallax", 40, 300, 1.2, (255, 255, 255), 2)
    text(canvas, msg, 40, 350, 0.65, hud.device_color("NPU"))
    return canvas


# ---------------------------------------------------------------- main
def camera(t: float, still: bool, scale: float = 1.0):
    """Camera path: (yaw, pitch, dolly). Slow horizontal orbit, a little vertical bob, gentle dolly in/out."""
    period = 6.0
    a = 2 * math.pi * t / period
    yaw_amp, pitch_amp, dolly_amp = (0.085, 0.03, 0.18) if still else (0.09, 0.035, 0.10)
    yaw_amp, pitch_amp, dolly_amp = yaw_amp * scale, pitch_amp * scale, dolly_amp * scale
    yaw = yaw_amp * math.sin(a)
    pitch = pitch_amp * math.sin(2 * a + 0.6)
    dolly = dolly_amp * 0.5 * (1 - math.cos(a / 2))
    return yaw, pitch, dolly


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="stock clip name (examples.common.clips), video file, or webcam index")
    p.add_argument("--image", type=Path, help="single still image: 3D-photo Ken Burns mode")
    p.add_argument("--device", default="NPU", help="device for the live depth model (default NPU)")
    p.add_argument("--size", type=int, default=364, help="model input side, multiple of 14 (default 364)")
    p.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True,
                   help="startup benchmark of the same model on NPU/GPU/CPU for the inset (default on)")
    p.add_argument("--record", action="store_true", help="record media/depth_parallax.gif + media/mp4/*.mp4")
    p.add_argument("--duration", type=float, default=None, help="seconds of output (default 12 when recording "
                   "or headless; unlimited in a window)")
    p.add_argument("--no-display", action="store_true", help="headless (no window)")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "depth_parallax.json", help="results JSON")
    p.add_argument("--orbit", type=float, default=1.0,
                   help="camera orbit amplitude multiplier (default 1.0 = +/-5 deg yaw)")
    p.add_argument("--snapshots", type=Path, help="also save every 30th composite frame as PNG into this folder")
    a = p.parse_args()

    if a.size % 14:
        raise SystemExit("--size must be a multiple of 14")
    device = a.device.upper()
    display = not a.no_display
    duration = a.duration if a.duration is not None else (12.0 if (a.record or not display) else None)

    def splash(msg):
        log(msg)
        if display:
            cv2.imshow("Depth Parallax", splash_frame(msg))
            cv2.waitKey(1)

    source, still, src_info = open_source(a)
    first = source.read()
    if first is None:
        raise SystemExit("source produced no frames")
    log(f"source: {src_info}")
    x0 = preprocess(R.center_crop(first), a.size)

    model = load_model(a.size)
    compiled, live_info = compile_device(model, device, splash)
    if compiled is None:
        raise SystemExit(f"{device} could not compile the model ({live_info.get('status')}): "
                         f"{live_info.get('reason') or live_info.get('error')}  -- not falling back to another device")
    exec_devs = live_info["execution_devices"]

    devices_out = {device: {**live_info}}
    cmp_results = {}
    if a.compare:
        for dev in ("NPU", "GPU", "CPU"):
            cm, info = (compiled, live_info) if dev == device else compile_device(model, dev, splash)
            if cm is None:
                cmp_results[dev] = {"status": info["status"]}
                devices_out[dev] = info
                continue
            splash(f"Benchmarking the same model on {dev}...")
            stats = bench(cm, x0, dev)
            cmp_results[dev] = {**stats, "status": "ok"}
            devices_out[dev] = {**info, "benchmark": stats}
            log(f"{dev} benchmark: median {stats['median_ms']} ms p95 {stats['p95_ms']} ms")
            if cm is not compiled:
                del cm
    else:
        cmp_results[device] = {"status": "--no-compare"}

    # ---- live loop: double-buffered requests so device inference overlaps host rendering
    reqs = [compiled.create_infer_request() for _ in range(2)]
    renderer = R.ParallaxRenderer(HERO_W, HERO_H)
    norm = R.DepthNormalizer()
    pivot = None
    fps_meter = ovutil.RollingFPS(alpha=0.08)
    rec = None
    out_fps = min(source.fps, 30.0)
    if a.record:
        from examples.common.media import Recorder
        rec = Recorder("depth_parallax", fps=out_fps, gif_width=720, gif_fps=10, max_seconds=duration)
    model_ms, host_ms, render_ms, lat_hist = [], [], [], []
    title = f"Depth Parallax  |  {MODEL_NAME} {a.size}x{a.size} on {device}"
    if src_info["kind"] == "clip":
        sub = f"clip '{src_info['name']}': {src_info.get('author', '')}, {src_info.get('license', '')}"
    else:
        sub = Path(str(src_info.get("path", f"webcam {src_info.get('index')}"))).name
    if src_info["kind"] == "image":
        sub += "  (still image, inference repeated each frame)"

    cur_frame = R.center_crop(first)
    reqs[0].set_input_tensor(ov.Tensor(x0))
    reqs[0].start_async()
    i, t_start = 0, time.perf_counter()
    try:
        while True:
            req = reqs[i % 2]
            req.wait()
            lat = float(req.latency)
            depth = req.get_output_tensor(0).data[0].copy()
            h0 = time.perf_counter()
            # kick off the next frame on the device before rendering this one
            nxt = source.read()
            if nxt is not None:
                nxt = R.center_crop(nxt)
                reqs[(i + 1) % 2].set_input_tensor(ov.Tensor(preprocess(nxt, a.size)))
                reqs[(i + 1) % 2].start_async()

            frame = cv2.resize(cur_frame, (HERO_W, HERO_H), interpolation=cv2.INTER_AREA)
            dn = norm(depth)                      # model resolution; renderer resamples as needed
            med = float(np.median(dn[::4, ::4]))
            pivot = med if pivot is None else pivot + 0.05 * (med - pivot)
            t = i / out_fps
            cam = camera(t, still, a.orbit)
            r0 = time.perf_counter()
            hero, dt_ = renderer.render(frame, dn, *cam, pivot=pivot)
            hero = R.depth_shade(hero, dt_)
            r_ms = (time.perf_counter() - r0) * 1000
            depth_bgr = R.colorize(cv2.resize(dn, (SIDE_W, SIDE_H), interpolation=cv2.INTER_LINEAR))

            model_ms.append(lat)
            render_ms.append(r_ms)
            lat_hist = (lat_hist + [lat])[-240:]
            fps_meter.tick(lat)
            p50 = float(np.median(model_ms[-120:]))
            p95 = float(np.percentile(model_ms[-120:], 95))
            h_avg = float(np.mean(host_ms[-60:])) if host_ms else r_ms
            badge = [f"depth p50 {p50:.1f} ms",
                     f"depth p95 {p95:.1f} ms",
                     f"model {1000 / p50:.0f} FPS | demo {fps_meter.fps or 0:.0f}",
                     f"exec: {','.join(exec_devs)}"]
            timing = (f"per frame: {device} depth {p50:.1f} ms (overlapped)  +  CPU pre/render/HUD {h_avg:.1f} ms"
                      f"  ->  {fps_meter.fps or 0:.1f} FPS end-to-end")
            canvas = build_canvas(cur_frame, depth_bgr, hero, title, sub, device, badge, cmp_results, device,
                                  timing, cam, lat_hist)
            host_ms.append((time.perf_counter() - h0) * 1000)

            if rec is not None:
                rec.add(canvas, bgr=True)
            if display:
                cv2.imshow("Depth Parallax", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if a.snapshots and i % 30 == 0:
                a.snapshots.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(a.snapshots / f"frame_{i:04d}.png"), canvas)
            i += 1
            if nxt is None:
                break
            cur_frame = nxt
            if rec is not None and rec.full:
                break
            if duration is not None and rec is None and time.perf_counter() - t_start >= duration:
                break
    finally:
        for r in reqs:
            try:
                r.wait()
            except Exception:
                pass
        source.release()
        if display:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    media = {}
    if rec is not None:
        log(f"writing media ({len(rec.frames)} frames)...")
        media = {k: _rel(v) for k, v in rec.save().items()}
        if "gif" in media:
            media["gif_mb"] = round((REPO_ROOT / media["gif"]).stat().st_size / 1e6, 2)

    live = {
        "device": device, "execution_devices": exec_devs, "frames": i, "wall_s": round(wall, 2),
        "end_to_end_fps": round(i / wall, 2) if wall > 0 else None,
        "model_latency_ms": ovutil.latency_stats(model_ms) if model_ms else None,
        "host_ms_per_frame": ovutil.latency_stats(host_ms) if host_ms else None,
        "parallax_render_ms": ovutil.latency_stats(render_ms) if render_ms else None,
        "latency_note": "model latency = InferRequest.latency (device-side, overlapped with host work); "
                        "host = preprocess next frame + depth normalise + parallax render + HUD; "
                        "end-to-end FPS includes recording overhead when --record",
    }
    result = {
        "meta": {"tool": "examples.depth_parallax", "date": dt.date.today().isoformat(),
                 "openvino_version": ov.__version__, "available_devices": list(ovutil.core().available_devices),
                 "model": MODEL_NAME, "model_path": _rel(MODEL_PATH), "model_license": MODEL_LICENSE,
                 "input_shape": [1, 3, a.size, a.size], "hero_size": [HERO_W, HERO_H],
                 "source": src_info, "recorded": bool(rec), "displayed": display},
        "devices": devices_out, "live": live, "media": media,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    summary = {"output": _rel(a.output), "frames": i, "end_to_end_fps": live["end_to_end_fps"],
               "live_model_p50_ms": (live["model_latency_ms"] or {}).get("median_ms"),
               "host_p50_ms": (live["host_ms_per_frame"] or {}).get("median_ms"),
               "compare_median_ms": {d: r.get("median_ms") for d, r in cmp_results.items()},
               "compile": {d: {k: v.get(k) for k in ("status", "compile_s", "cache", "execution_devices")}
                           for d, v in devices_out.items()}, "media": media}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
