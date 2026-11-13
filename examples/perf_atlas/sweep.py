"""Synthetic model builders, numpy references and the timed sweep for the NPU Performance Atlas.

Methodology (see README.md):
* Models are built in-process with ov.opset13 from seeded random weights (rng 2026). No downloads.
* Three weight formats:
    FP32 - float32 constants. CPU/GPU compiled with INFERENCE_PRECISION_HINT=f32; the NPU has no f32
           compute path, so it is compiled with its default (f16) and reported as such.
    FP16 - float16 constants + Convert (the compress_to_fp16 pattern); every device asked for f16.
    INT8 - weight-only symmetric per-output-channel int8 (i8 const -> Convert -> Multiply scale, the same
           decompression pattern NNCF's INT8_SYM weight compression emits). Activations run in f16.
           This is NOT full int8 activation quantization (no FakeQuantize / calibration).
* Each request is prepared once (input tensor bound) and timed with ovutil.time_request, so the timed
  region is device work + dispatch only. Output is copied and checked against a numpy float32 reference
  computed with the *effective* weights (fp16-rounded / dequantized), outside the timed region.
"""
from __future__ import annotations

import ctypes
import gc
import sys
import time
from dataclasses import dataclass

import numpy as np
import openvino as ov
import openvino.opset13 as ops

from examples.common import ovutil

SEED = 2026
PRECISIONS = ("FP32", "FP16", "INT8")


# --------------------------------------------------------------------------- model specs
@dataclass(frozen=True)
class Spec:
    family: str  # "MLP" or "CONV"
    size: str  # "S", "M", ...
    dim: int  # MLP width D, or CONV channels C
    layers: int
    res: int = 0  # CONV input resolution (square)

    @property
    def name(self) -> str:
        return f"{self.family}-{self.size}"

    def input_shape(self, batch: int) -> list[int]:
        return [batch, self.dim] if self.family == "MLP" else [batch, 3, self.res, self.res]

    @property
    def gflop_per_sample(self) -> float:
        if self.family == "MLP":  # per block: D->4D and 4D->D matmuls, 2 FLOP per MAC
            return 16 * self.dim ** 2 * self.layers / 1e9
        hw = (self.res // 2) ** 2
        stem = 2 * hw * self.dim * 3 * 9
        return (stem + self.layers * 2 * hw * self.dim * self.dim * 9) / 1e9

    @property
    def params_m(self) -> float:
        if self.family == "MLP":
            return 8 * self.dim ** 2 * self.layers / 1e6
        return (self.dim * 27 + self.layers * self.dim * self.dim * 9) / 1e6


SPECS = {
    "MLP-S": Spec("MLP", "S", 512, 4),
    "MLP-M": Spec("MLP", "M", 1024, 4),
    "MLP-L": Spec("MLP", "L", 2048, 3),
    "CONV-XS": Spec("CONV", "XS", 32, 3, res=64),
    "CONV-S": Spec("CONV", "S", 32, 4, res=128),
    "CONV-M": Spec("CONV", "M", 64, 6, res=128),
}


def make_weights(spec: Spec) -> list[np.ndarray]:
    """Seeded float32 weights, He-scaled so activations stay O(1) through the stack."""
    rng = np.random.default_rng(SEED)
    ws = []
    if spec.family == "MLP":
        d = spec.dim
        for _ in range(spec.layers):
            ws.append((rng.standard_normal((4 * d, d)) * np.sqrt(2.0 / d)).astype(np.float32))
            ws.append((rng.standard_normal((d, 4 * d)) * 0.5 / np.sqrt(4 * d)).astype(np.float32))
    else:
        c = spec.dim
        ws.append((rng.standard_normal((c, 3, 3, 3)) * np.sqrt(2.0 / 27)).astype(np.float32))
        for _ in range(spec.layers):
            ws.append((rng.standard_normal((c, c, 3, 3)) * np.sqrt(2.0 / (9 * c))).astype(np.float32))
    return ws


def make_input(spec: Spec, batch: int) -> np.ndarray:
    rng = np.random.default_rng((SEED, 1))
    return rng.standard_normal(spec.input_shape(batch)).astype(np.float32)


def quantize_int8(w: np.ndarray):
    """Symmetric per-output-channel int8: returns (q int8, scale float32 broadcastable to w)."""
    axes = tuple(range(1, w.ndim))
    scale = np.abs(w).max(axis=axes, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-12).astype(np.float32)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def effective_weights(ws: list[np.ndarray], precision: str) -> list[np.ndarray]:
    if precision == "FP32":
        return ws
    if precision == "FP16":
        return [w.astype(np.float16).astype(np.float32) for w in ws]
    out = []
    for w in ws:
        q, s = quantize_int8(w)
        out.append(q.astype(np.float32) * s)
    return out


def _const(w: np.ndarray, precision: str):
    if precision == "FP32":
        return ops.constant(w)
    if precision == "FP16":
        return ops.convert(ops.constant(w.astype(np.float16)), ov.Type.f32)
    q, s = quantize_int8(w)
    return ops.multiply(ops.convert(ops.constant(q), ov.Type.f32), ops.constant(s))


def build_model(spec: Spec, ws: list[np.ndarray], batch: int, precision: str) -> ov.Model:
    x = ops.parameter(spec.input_shape(batch), np.float32, name="input")
    if spec.family == "MLP":
        h = x
        for i in range(spec.layers):
            up = ops.relu(ops.matmul(h, _const(ws[2 * i], precision), False, True))
            h = ops.add(h, ops.matmul(up, _const(ws[2 * i + 1], precision), False, True))
        y = h
    else:
        h = ops.relu(ops.convolution(x, _const(ws[0], precision), [2, 2], [1, 1], [1, 1], [1, 1]))
        for w in ws[1:]:
            h = ops.relu(ops.convolution(h, _const(w, precision), [1, 1], [1, 1], [1, 1], [1, 1]))
        y = ops.reduce_mean(h, ops.constant(np.array([2, 3], dtype=np.int64)), False)
    return ov.Model([y], [x], f"{spec.name}_b{batch}_{precision}")


def _conv_np(x: np.ndarray, w: np.ndarray, stride: int) -> np.ndarray:
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))[:, :, ::stride, ::stride]
    out = np.tensordot(win, w, axes=([1, 4, 5], [1, 2, 3]))  # [B, Ho, Wo, Cout]
    return np.ascontiguousarray(out.transpose(0, 3, 1, 2))


def reference(spec: Spec, ws: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """float32 numpy forward pass (float64 accumulation for the MLP to keep the reference clean)."""
    if spec.family == "MLP":
        h = x.astype(np.float64)
        for i in range(spec.layers):
            up = np.maximum(h @ ws[2 * i].T.astype(np.float64), 0)
            h = h + up @ ws[2 * i + 1].T.astype(np.float64)
        return h.astype(np.float32)
    h = np.maximum(_conv_np(x, ws[0], 2), 0)
    for w in ws[1:]:
        h = np.maximum(_conv_np(h, w, 1), 0)
    return h.mean(axis=(2, 3))


# --------------------------------------------------------------------------- helpers
def power_state() -> dict:
    """AC/battery state via GetSystemPowerStatus (Windows). Timings differ a lot on battery."""
    try:
        class SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = SPS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        ac = {0: "battery", 1: "ac"}.get(s.ACLineStatus, "unknown")
        return {"power_source": ac, "battery_percent": None if s.BatteryLifePercent == 255 else s.BatteryLifePercent}
    except Exception:
        return {"power_source": "unknown"}


def device_names() -> dict:
    names = {}
    for d in ovutil.core().available_devices:
        try:
            names[d] = ovutil.core().get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            names[d] = d
    return names


def compile_config(device: str, precision: str, hint: str = "LATENCY") -> dict:
    cfg = {"PERFORMANCE_HINT": hint}
    if precision == "FP32":
        if device != "NPU":  # NPU has no f32 compute; leave its default (f16) and report it
            cfg["INFERENCE_PRECISION_HINT"] = "f32"
    else:
        cfg["INFERENCE_PRECISION_HINT"] = "f16"
    return cfg


def reported_precision(compiled) -> str:
    try:
        return str(compiled.get_property("INFERENCE_PRECISION_HINT")).replace("<Type: '", "").replace("'>", "")
    except Exception:
        return "unreported"


def tolerance(exec_precision: str) -> float:
    """Allowed max|out-ref| / max|ref| given the precision the device actually computes in."""
    p = exec_precision.lower()
    if "32" in p:
        return 2e-3
    if "bf16" in p:
        return 4e-2
    return 1.5e-2  # f16 / unreported (NPU); observed floor ~1e-3 (GPU int8-weight path ~9e-3)


def rel_err(out: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(out.astype(np.float64) - ref)) / (np.max(np.abs(ref)) + 1e-12))


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- measurement
def measure_sync(model, device, precision, x, ref, ref_fp32, budget_s, max_runs, min_runs=10) -> dict:
    cfg = compile_config(device, precision)
    compiled, info = ovutil.compile_on(model, device, cfg)
    row = dict(info)
    row["config"] = cfg
    if compiled is None:
        return row
    row["inference_precision"] = reported_precision(compiled)
    try:
        req = compiled.create_infer_request()
        req.set_input_tensor(ov.Tensor(x))
        t0 = time.perf_counter()
        req.infer()
        row["first_infer_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        out = req.get_output_tensor(0).data.copy()  # outside any timed region
        err = rel_err(out, ref)
        tol = tolerance(row["inference_precision"])
        row.update(rel_err=round(err, 6), rel_err_vs_fp32=round(rel_err(out, ref_fp32), 6), tolerance=tol)
        if not np.isfinite(err) or err > tol:
            row["status"] = "failed_correctness"
            if out.shape[0] > 1:  # diagnose: does every batch element just repeat sample 0's result?
                to_first = rel_err(out[1:], np.broadcast_to(ref[:1], ref[1:].shape))
                to_own = rel_err(out[1:], ref[1:])
                row["batch_diagnosis"] = {"err_samples_1plus_vs_own_ref": round(to_own, 6),
                                          "err_samples_1plus_vs_sample0_ref": round(to_first, 6),
                                          "outputs_replicate_sample0": bool(to_first < tol < to_own)}
            return row
        est = []
        for _ in range(2):
            t0 = time.perf_counter()
            req.infer()
            est.append((time.perf_counter() - t0) * 1000)
        runs = int(np.clip(budget_s * 1000 / max(min(est), 1e-3), min_runs, max_runs))
        row.update(ovutil.time_request(req, runs=runs, warmup=3))
        row["status"] = "executed"
    except Exception as exc:
        row.update(status="failed", stage="infer", error_type=type(exc).__name__, error=str(exc)[:1200])
    finally:
        del compiled
    return row


def measure_async(model, device, precision, x, budget_s) -> dict:
    """THROUGHPUT-hint compile + AsyncInferQueue with the device's optimal number of requests."""
    cfg = compile_config(device, precision, hint="THROUGHPUT")
    compiled, info = ovutil.compile_on(model, device, cfg)
    row = dict(info)
    row["config"] = cfg
    if compiled is None:
        return row
    try:
        try:
            nireq = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
        except Exception:
            nireq = 4
        nireq = max(1, min(nireq, 16))
        queue = ov.AsyncInferQueue(compiled, nireq)
        tensor = ov.Tensor(x)
        for r in queue:
            r.set_input_tensor(tensor)
        for _ in range(nireq * 2):  # warm-up
            queue.start_async()
        queue.wait_all()
        count, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < budget_s or count < nireq * 3:
            queue.start_async()
            count += 1
        queue.wait_all()
        elapsed = time.perf_counter() - t0
        row.update(status="executed", nireq=nireq, requests=count, elapsed_s=round(elapsed, 3),
                   infer_per_s=round(count / elapsed, 2), samples_per_s=round(count * x.shape[0] / elapsed, 2),
                   inference_precision=reported_precision(compiled))
    except Exception as exc:
        row.update(status="failed", stage="infer", error_type=type(exc).__name__, error=str(exc)[:1200])
    finally:
        del compiled
    return row


def run_sweep(specs, batches, precisions, devices, budget_s, max_runs, async_cfgs, async_budget_s,
              on_row=None) -> tuple[list, list]:
    results, async_rows = [], []
    seq = 0
    t_start = time.perf_counter()
    max_b = max(batches)
    for spec in specs:
        ws = make_weights(spec)
        x_full = make_input(spec, max_b)
        ref32_full = reference(spec, ws, x_full)
        for precision in precisions:
            ref_full = ref32_full if precision == "FP32" else reference(spec, effective_weights(ws, precision), x_full)
            for batch in batches:
                model = build_model(spec, ws, batch, precision)
                x = np.ascontiguousarray(x_full[:batch])
                base = {"model": spec.name, "family": spec.family, "size": spec.size, "batch": batch,
                        "precision": precision, "gflop_per_sample": round(spec.gflop_per_sample, 5),
                        "gflop_per_infer": round(spec.gflop_per_sample * batch, 5),
                        "params_m": round(spec.params_m, 3)}
                for device in devices:
                    row = dict(base)
                    row.update(measure_sync(model, device, precision, x, ref_full[:batch], ref32_full[:batch],
                                            budget_s, max_runs))
                    if row.get("status") == "executed":
                        med = row["median_ms"]
                        row["infer_per_s"] = round(1000.0 / med, 2)
                        row["samples_per_s"] = round(batch * 1000.0 / med, 2)
                        row["gflops_effective"] = round(base["gflop_per_infer"] / (med / 1000.0), 2)
                    row["seq"] = seq
                    row["t_s"] = round(time.perf_counter() - t_start, 2)
                    seq += 1
                    results.append(row)
                    log(f"[{row['t_s']:7.1f}s] {spec.name:8s} b{batch:<3d} {precision} {device:4s} "
                        f"{row['status']:18s} "
                        + (f"p50 {row['median_ms']:9.3f} ms  p95 {row['p95_ms']:9.3f}  "
                           f"compile {row['compile_s']:.2f}s  {row['gflops_effective']:8.1f} GFLOP/s"
                           if row["status"] == "executed" else str(row.get("error", row.get("rel_err", "")))[:120]))
                    if on_row:
                        on_row(row)
                if (spec.name, batch, precision) in async_cfgs:
                    for device in devices:
                        arow = dict(base)
                        arow.update(measure_async(model, device, precision, x, async_budget_s))
                        async_rows.append(arow)
                        log(f"   async {spec.name} b{batch} {precision} {device}: {arow['status']} "
                            f"{arow.get('samples_per_s', arow.get('error', ''))!s:.100} samples/s "
                            f"(nireq {arow.get('nireq')})")
                del model
                gc.collect()
    return results, async_rows
