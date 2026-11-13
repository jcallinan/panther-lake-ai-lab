"""Child processes for the offload dashboard.

* `ai_process`: the always-on AI pipeline (selfie-segmentation background blur + YOLOX-tiny detection) paced to a
  target FPS on whichever device the orchestrator selects. Pre/post-processing runs on the CPU in every placement and
  is timed separately (thread CPU time), so its cost is reported honestly.
* `foreground_worker`: the "user's actual work": a fixed zlib-compression job repeated forever; each completed job
  bumps a shared counter, so throughput is ops/s.
* `probe_process`: an "app responsiveness" probe: asks for 2 ms sleeps and records how late it wakes (scheduling
  jitter, the thing that makes a UI feel sluggish under load).

All shared state is lock-free `multiprocessing` raw arrays; the orchestrator only samples them.
"""
from __future__ import annotations

import os
import time
import zlib

import numpy as np

FRAME_W, FRAME_H = 640, 360
DEVICES = ["CPU", "GPU", "NPU"]  # index used in the shared "device" slot; -1 = AI off

# Layout of the shared AI stats array (doubles).
S_FRAMES, S_INFER_MS, S_PREPOST_MS, S_FRAME_SEQ, S_DETS, S_ACTIVE_DEV, S_SUM_INFER, S_SUM_PREPOST, \
    S_SUM_THREAD_CPU, S_READY, S_SUM_WALL, S_ERROR = range(12)
N_STATS = 12

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie "
        "suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()

YOLOX_MEAN = (np.array([0.485, 0.456, 0.406], np.float32)[::-1] * 255).reshape(1, 1, 3)
YOLOX_STD = (np.array([0.229, 0.224, 0.225], np.float32)[::-1] * 255).reshape(1, 1, 3)


# --------------------------------------------------------------------------------------------- frame source
def synthetic_frames(n: int = 120) -> list[np.ndarray]:
    """Looping synthetic scene (moving shapes over a gradient) used when no stock clip is available."""
    import cv2

    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    frames = []
    for i in range(n):
        t = i / n * 2 * np.pi
        base = np.stack([90 + 60 * np.sin(xx / 120 + t), 80 + 50 * np.cos(yy / 90 - t), 110 + 40 * np.sin((xx + yy) / 200)],
                        -1)
        img = np.clip(base, 0, 255).astype(np.uint8)
        cx = int(FRAME_W / 2 + 180 * np.sin(t))
        cv2.ellipse(img, (cx, 200), (60, 140), 0, 0, 360, (60, 70, 90), -1)
        cv2.circle(img, (cx, 80), 42, (120, 150, 190), -1)
        cv2.rectangle(img, (40 + i * 4 % 500, 260), (130 + i * 4 % 500, 330), (30, 30, 160), -1)
        frames.append(img)
    return frames


def load_frames(clip: str | None, max_frames: int = 300) -> tuple[list[np.ndarray], str]:
    """Decode a clip into memory at 640x360 (so decode cost is identical/zero across phases)."""
    import cv2

    if clip:
        cap = cv2.VideoCapture(clip)
        frames = []
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            s = max(FRAME_W / w, FRAME_H / h)  # cover-crop to 16:9
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            y0, x0 = (f.shape[0] - FRAME_H) // 2, (f.shape[1] - FRAME_W) // 2
            frames.append(np.ascontiguousarray(f[y0:y0 + FRAME_H, x0:x0 + FRAME_W]))
        cap.release()
        if frames:
            return frames, os.path.basename(clip)
    return synthetic_frames(), "synthetic"


# --------------------------------------------------------------------------------------------- AI pipeline
class Pipeline:
    """Pre-process -> segmentation + detection inference -> blur composite + NMS + drawing."""

    def __init__(self, seg_compiled, det_compiled):
        self.seg = seg_compiled.create_infer_request()
        self.det = det_compiled.create_infer_request()

    # Normalisation (u8 -> f32, BGR->RGB, mean/scale) is folded into the models with OpenVINO's PrePostProcessor
    # (see build_model), so it runs wherever the model runs. The CPU only resizes.
    @staticmethod
    def pre_seg(frame):
        import cv2
        return cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)[None]

    @staticmethod
    def pre_det(frame):
        import cv2
        return cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)[None]

    @staticmethod
    def post(frame, alpha, boxes, labels, conf=0.5):
        import cv2

        h, w = frame.shape[:2]
        mask = cv2.resize(alpha.reshape(256, 256), (w, h), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (w, h), interpolation=cv2.INTER_LINEAR)
        out = cv2.blendLinear(frame, blurred, mask, 1.0 - mask)
        keep = boxes[:, 4] > conf
        b, lab = boxes[keep], labels[keep]
        n = 0
        if len(b):
            sx, sy = w / 416, h / 416
            xywh = np.stack([b[:, 0] * sx, b[:, 1] * sy, (b[:, 2] - b[:, 0]) * sx, (b[:, 3] - b[:, 1]) * sy], 1)
            idx = cv2.dnn.NMSBoxes(xywh.tolist(), b[:, 4].tolist(), conf, 0.45)
            for i in np.array(idx).reshape(-1):
                x, y, bw, bh = (int(v) for v in xywh[i])
                cls = COCO[int(lab[i])] if int(lab[i]) < len(COCO) else str(int(lab[i]))
                cv2.rectangle(out, (x, y), (x + bw, y + bh), (255, 255, 255), 2, cv2.LINE_AA)
                if bh >= 80:  # label only sizeable boxes to keep the frame legible
                    cv2.putText(out, f"{cls} {b[i, 4]:.2f}", (x + 3, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)
                n += 1
        return out, n

    def run(self, frame):
        """Returns (output_frame, n_dets, infer_wall_ms, prepost_wall_ms).

        Wall time (perf_counter) is used for pre/post because Windows thread CPU time has ~15.6 ms tick
        granularity. Unloaded, pre/post wall ~= its CPU time; under contention it also includes waiting to run.
        """
        c0 = time.perf_counter()
        xs, xd = self.pre_seg(frame), self.pre_det(frame)
        c1 = time.perf_counter()
        self.seg.infer({0: xs})
        self.det.infer({0: xd})
        c2 = time.perf_counter()
        infer_ms = (c2 - c1) * 1000
        alpha = self.seg.get_output_tensor(0).data
        boxes = self.det.get_output_tensor(0).data[0]
        labels = self.det.get_output_tensor(1).data[0]
        out, n = self.post(frame, alpha, boxes, labels)
        c3 = time.perf_counter()
        return out, n, infer_ms, ((c1 - c0) + (c3 - c2)) * 1000


YOLOX_MEAN_L = [float(v) for v in YOLOX_MEAN.ravel()]
YOLOX_STD_L = [float(v) for v in YOLOX_STD.ravel()]


def build_model(key: str, path: str, shape: list):
    """Read + reshape, then fold u8 NHWC BGR input conversion into the graph (PrePostProcessor)."""
    import openvino as ov
    from openvino.preprocess import PrePostProcessor

    from examples.common import ovutil

    m = ovutil.core().read_model(path)
    m.reshape({m.inputs[0]: ov.PartialShape(shape)})
    ppp = PrePostProcessor(m)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    pre = ppp.input().preprocess().convert_element_type(ov.Type.f32)
    if key == "seg":  # RGB / 255
        pre.reverse_channels().scale(255.0)
    else:  # YOLOX: BGR, ImageNet mean/std x 255
        pre.mean(YOLOX_MEAN_L).scale(YOLOX_STD_L)
    return ppp.build()


def ai_process(models: dict, devices: list, clip, target_fps: float, device_slot, stats, frame_buf, info_q,
               stop_evt):
    """Compile both models on every requested device up front, then serve frames on the device in `device_slot`."""
    from examples.common import ovutil

    frames, source = load_frames(clip)
    pipes, compile_info = {}, {}
    for dev in devices:
        info = {}
        compiled = []
        for key, (path, shape) in models.items():
            m = build_model(key, path, shape)
            cm, ci = ovutil.compile_on(m, dev, {"PERFORMANCE_HINT": "LATENCY"})
            info[key] = ci
            compiled.append(cm)
        compile_info[dev] = info
        if all(c is not None for c in compiled):
            p = Pipeline(*compiled)
            for _ in range(3):  # warm-up
                p.run(frames[0])
            pipes[dev] = p
    info_q.put({"source": source, "n_frames": len(frames), "compile": compile_info,
                "ready_devices": list(pipes)})
    stats[S_READY] = 1

    out_view = np.frombuffer(frame_buf, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 3)
    period = 1.0 / target_fps
    i, deadline = 0, time.perf_counter()
    while not stop_evt.is_set():
        idx = int(device_slot.value)
        frame = frames[i % len(frames)]
        i += 1
        if idx < 0 or DEVICES[idx] not in pipes:
            stats[S_ACTIVE_DEV] = -1
            out_view[:] = frame  # AI off: show the raw stream (a memcpy, ~0.7 MB)
            stats[S_FRAME_SEQ] += 1
            time.sleep(0.1)
            deadline = time.perf_counter()
            continue
        t_wall = time.perf_counter()
        c0 = time.thread_time()
        try:
            out, n, infer_ms, prepost_ms = pipes[DEVICES[idx]].run(frame)
        except Exception:  # record, never substitute a device
            stats[S_ERROR] += 1
            time.sleep(0.05)
            continue
        out_view[:] = out
        stats[S_ACTIVE_DEV] = idx
        stats[S_INFER_MS] = infer_ms
        stats[S_PREPOST_MS] = prepost_ms
        stats[S_DETS] = n
        stats[S_SUM_INFER] += infer_ms
        stats[S_SUM_PREPOST] += prepost_ms
        stats[S_SUM_THREAD_CPU] += (time.thread_time() - c0) * 1000
        stats[S_SUM_WALL] += (time.perf_counter() - t_wall) * 1000
        stats[S_FRAMES] += 1
        stats[S_FRAME_SEQ] += 1
        deadline += period
        now = time.perf_counter()
        if deadline > now:
            time.sleep(deadline - now)
        else:
            deadline = now  # running late: do not try to catch up with a burst


# --------------------------------------------------------------------------------------------- foreground work
def _work_buffer(seed: int, size: int = 256 * 1024) -> bytes:
    """Semi-compressible data (text-like token stream), so zlib does real work."""
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 2000, size // 4).astype(np.uint16)
    return (words % 251).astype(np.uint8).tobytes() + rng.integers(0, 255, size // 2, dtype=np.uint8).tobytes()


def foreground_worker(slot: int, counters, stop_evt):
    """One CPU-bound 'user work' process: zlib level-6 compress a fixed 384 KB buffer, forever."""
    buf = _work_buffer(slot)
    while not stop_evt.is_set():
        zlib.compress(buf, 6)
        counters[slot] += 1


def probe_process(lat, stop_evt, sleep_ms: float = 2.0):
    """Responsiveness probe: records wake-up lateness (ms) of 2 ms sleeps into a 512-slot ring buffer.

    lat[0] = write index; lat[1:] = ring of lateness samples.
    """
    n = len(lat) - 1
    while not stop_evt.is_set():
        t0 = time.perf_counter()
        time.sleep(sleep_ms / 1000)
        late = (time.perf_counter() - t0) * 1000 - sleep_ms
        k = int(lat[0])
        lat[1 + k % n] = late
        lat[0] = k + 1
