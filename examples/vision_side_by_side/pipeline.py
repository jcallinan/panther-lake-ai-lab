"""Models, pre/post-processing and per-device runners for the live vision side-by-side demo.

Pre-processing runs once per frame and the same input arrays go to every device. Timed regions cover
inference only (`InferRequest.infer()` in sequential mode, start_async -> completion callback in concurrent
mode). Decode, pre/post-processing and drawing are excluded and reported separately as pipeline FPS.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import openvino as ov

from examples.common import MODELS_DIR, ovutil

YOLOX_XML = MODELS_DIR / "yolox_tiny-int8-ov" / "yolox_tiny.xml"
SELFIE_ONNX = MODELS_DIR / "mediapipe_selfie_segmentation" / "onnx" / "model.onnx"
DET_SIZE = 416
SEG_SIZE = 256
# ImageNet mean/std on 0-255 pixels. The IR's model_info says RGB order; BGR gives near-identical results.
_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_STD = np.array([58.395, 57.12, 57.375], np.float32)
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote "
        "keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
        "hair_drier toothbrush").split()

MODEL_INFO = {
    "detect": {"model": "OpenVINO/yolox_tiny-int8-ov (YOLOX-Tiny, INT8 PTQ, COCO-80)", "license": "Apache-2.0",
               "input": f"1x3x{DET_SIZE}x{DET_SIZE}, letterbox (pad 114), RGB, ImageNet mean/std on 0-255",
               "post": "score >= conf, class-aware NMS (cv2.dnn.NMSBoxes), done on the host"},
    "segment": {"model": "onnx-community/mediapipe_selfie_segmentation (MediaPipe selfie segmenter)",
                "license": "Apache-2.0", "input": f"1x3x{SEG_SIZE}x{SEG_SIZE}, stretch-resize, RGB / 255",
                "post": "alpha mask upsampled to frame, smoothstep feather"},
}


def load_model(task: str) -> ov.Model:
    path, size = (YOLOX_XML, DET_SIZE) if task == "detect" else (SELFIE_ONNX, SEG_SIZE)
    model = ovutil.core().read_model(path)
    model.reshape({model.inputs[0]: ov.PartialShape([1, 3, size, size])})  # static shape for the NPU
    return model


# ----------------------------------------------------------------------------------------- pre-processing
def preprocess_detect(frame_bgr: np.ndarray):
    """Letterbox to 416x416 (top-left aligned, pad 114) -> NCHW float32. Returns (tensor, scale)."""
    h, w = frame_bgr.shape[:2]
    s = min(DET_SIZE / h, DET_SIZE / w)
    nh, nw = round(h * s), round(w * s)
    canvas = np.full((DET_SIZE, DET_SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = (canvas[..., ::-1].astype(np.float32) - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), s


def preprocess_segment(frame_bgr: np.ndarray) -> np.ndarray:
    x = cv2.resize(frame_bgr, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)[..., ::-1]
    return np.ascontiguousarray((x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


# ---------------------------------------------------------------------------------------- post-processing
@dataclass
class Detection:
    box: tuple  # x1, y1, x2, y2 in frame pixels
    score: float
    label: str


def postprocess_detect(boxes: np.ndarray, labels: np.ndarray, scale: float, conf: float = 0.35,
                       iou: float = 0.5, max_det: int = 100) -> list[Detection]:
    """boxes [3549,5] (x1,y1,x2,y2,score) in letterbox pixels, labels [3549]. YOLOX IR applies no NMS."""
    keep = boxes[:, 4] >= conf
    if not keep.any():
        return []
    b, lab = boxes[keep], labels[keep].astype(np.int64)
    xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    # class-aware NMS in one call: shift each class into its own coordinate island
    shifted = xywh.copy()
    shifted[:, :2] += lab[:, None] * 4096.0
    idx = cv2.dnn.NMSBoxes(shifted.tolist(), b[:, 4].tolist(), conf, iou, top_k=max_det)
    out = []
    for i in np.asarray(idx).reshape(-1):
        x1, y1, x2, y2 = (b[i, :4] / scale).tolist()
        name = COCO[lab[i]] if 0 <= lab[i] < len(COCO) else str(lab[i])
        out.append(Detection((x1, y1, x2, y2), float(b[i, 4]), name.replace("_", " ")))
    return out


def postprocess_segment(alpha: np.ndarray, size_wh: tuple) -> np.ndarray:
    """[1,1,256,256] alpha -> float32 HxW mask in [0,1], feathered."""
    m = cv2.resize(alpha.reshape(SEG_SIZE, SEG_SIZE), size_wh, interpolation=cv2.INTER_LINEAR)
    m = np.clip((m - 0.25) / 0.5, 0.0, 1.0)
    return (m * m * (3 - 2 * m)).astype(np.float32)  # smoothstep


# ------------------------------------------------------------------------------------------ device runner
@dataclass
class DeviceRunner:
    """Compiled models + one InferRequest (sequential) or one AsyncInferQueue (concurrent) per task."""
    device: str
    tasks: list
    concurrent: bool = False
    compiled: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    requests: dict = field(default_factory=dict)
    queues: dict = field(default_factory=dict)
    times: dict = field(default_factory=dict)  # task -> [ms]; plus "frame" -> [ms]
    outputs: dict = field(default_factory=dict)
    _done: dict = field(default_factory=dict)

    def compile(self, models: dict) -> bool:
        ok = True
        for task in self.tasks:
            compiled, info = ovutil.compile_on(models[task], self.device, {"PERFORMANCE_HINT": "LATENCY"})
            self.info[task] = info
            if compiled is None:
                ok = False
                continue
            self.compiled[task] = compiled
            self.times[task] = []
            if self.concurrent:
                q = ov.AsyncInferQueue(compiled, 1)
                q.set_callback(self._make_callback(task))
                self.queues[task] = q
            else:
                self.requests[task] = compiled.create_infer_request()
        self.times["frame"] = []
        return ok

    def _make_callback(self, task):
        def cb(request, t0):
            self._done[task] = (time.perf_counter() - t0) * 1000
            self.outputs[task] = [request.get_output_tensor(i).data.copy()
                                  for i in range(len(request.model_outputs))]
        return cb

    def execution_devices(self) -> dict:
        return {t: self.info[t].get("execution_devices", []) for t in self.tasks if t in self.info}

    # sequential mode: each device runs alone, one task after another; times are infer() only
    def run_sequential(self, inputs: dict, record: bool = True) -> float:
        total = 0.0
        for task, req in self.requests.items():
            t0 = time.perf_counter()
            req.infer({0: inputs[task]})
            ms = (time.perf_counter() - t0) * 1000
            total += ms
            self.outputs[task] = [req.get_output_tensor(i).data for i in range(len(req.model_outputs))]
            if record:
                self.times[task].append(ms)
        if record:
            self.times["frame"].append(total)
        return total

    # concurrent mode: start everything, caller waits on all devices
    def start_async(self, inputs: dict):
        self._done.clear()
        self._t0 = time.perf_counter()
        for task, q in self.queues.items():
            q.start_async({0: inputs[task]}, self._t0)

    def wait_async(self, record: bool = True) -> float:
        for q in self.queues.values():
            q.wait_all()
        frame_ms = max(self._done.values()) if self._done else 0.0
        if record:
            for task, ms in self._done.items():
                self.times[task].append(ms)
            self.times["frame"].append(frame_ms)
        return frame_ms

    def stats(self) -> dict:
        return {k: ovutil.latency_stats(v) for k, v in self.times.items() if v}
