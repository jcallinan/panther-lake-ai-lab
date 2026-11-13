"""OpenVINO helpers: honest device compilation and latency statistics.

Never silently falls back to another device: compile failures are returned, not hidden.
"""
import statistics
import time

import numpy as np
import openvino as ov

_core = None


def core() -> ov.Core:
    global _core
    if _core is None:
        _core = ov.Core()
    return _core


def available(device: str) -> bool:
    return device in core().available_devices


def execution_devices(compiled) -> list:
    """Devices that actually execute the compiled model, as a normalised list."""
    try:
        value = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        return ["unverified"]
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def compile_on(model, device: str, config: dict | None = None):
    """Compile `model` (ov.Model or path) on `device`.

    Returns (compiled_or_None, info_dict). info has status, compile_s, execution_devices or error.
    """
    info = {"device": device, "status": "skipped"}
    if not available(device):
        info["reason"] = "device not enumerated"
        return None, info
    try:
        start = time.perf_counter()
        compiled = core().compile_model(model, device, config or {})
        info.update(status="compiled", compile_s=round(time.perf_counter() - start, 3),
                    execution_devices=execution_devices(compiled))
        return compiled, info
    except Exception as exc:
        info.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1200])
        return None, info


def latency_stats(times_ms) -> dict:
    """Median / p95 / min / max (ms, 4 dp) plus throughput from a list of per-call latencies."""
    arr = np.asarray(times_ms, dtype=np.float64)
    med = float(statistics.median(arr))
    return {
        "runs": int(arr.size),
        "median_ms": round(med, 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "per_s_at_median": round(1000.0 / med, 2) if med > 0 else None,
    }


def time_request(request, runs: int = 200, warmup: int = 10) -> dict:
    """Time a prepared InferRequest (inputs already set) so only device work + dispatch is measured."""
    for _ in range(warmup):
        request.infer()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        request.infer()
        times.append((time.perf_counter() - start) * 1000)
    return latency_stats(times)


class RollingFPS:
    """Exponential moving average of frames/sec and latency for live overlays."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.latency_ms = None
        self._last = None
        self.fps = None

    def tick(self, latency_ms: float | None = None):
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps is None else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        if latency_ms is not None:
            self.latency_ms = latency_ms if self.latency_ms is None else (
                (1 - self.alpha) * self.latency_ms + self.alpha * latency_ms)
