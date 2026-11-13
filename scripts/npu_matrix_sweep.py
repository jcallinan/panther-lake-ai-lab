"""Compare an identical synthetic OpenVINO matrix model across available devices.

This is a compatibility/latency microbenchmark, NOT an LLM or TOPS measurement.
Small widths are dominated by dispatch overhead; see examples/perf_atlas for a sweep
large enough to show where the NPU helps.
Run: python scripts/npu_matrix_sweep.py --devices CPU GPU NPU
"""
import argparse
import json
import statistics
import time
import numpy as np
import openvino as ov

def model_for(width, weights):
    x = ov.opset13.parameter([1, width], dtype=np.float32, name="features")
    y = ov.opset13.matmul(x, ov.opset13.constant(weights), False, False)
    return ov.Model([y], [x], f"matmul_{width}")

def exec_devices(compiled):
    try:
        value = compiled.get_property("EXECUTION_DEVICES")
        return [value] if isinstance(value, str) else [str(v) for v in value]
    except Exception:
        return ["unverified"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    p.add_argument("--widths", nargs="+", type=int, default=[32, 128, 512])
    p.add_argument("--runs", type=int, default=200)
    a = p.parse_args()
    if a.runs < 1 or any(w < 1 or w > 4096 for w in a.widths):
        p.error("runs must be >=1 and widths must be between 1 and 4096")
    rng = np.random.default_rng(2026)
    core = ov.Core()
    available = list(core.available_devices)
    results = []
    for width in a.widths:
        weights = rng.normal(0, 1 / np.sqrt(width), (width, width)).astype(np.float32)
        data = rng.normal(0, 1, (1, width)).astype(np.float32)
        reference = data @ weights
        model = model_for(width, weights)
        for device in a.devices:
            row = {"device": device, "width": width, "status": "skipped"}
            if device not in available:
                row["reason"] = "device not enumerated"
                results.append(row)
                continue
            try:
                start = time.perf_counter()
                compiled = core.compile_model(model, device)
                row.update(status="compiled", compile_s=round(time.perf_counter() - start, 3),
                           execution_devices=exec_devices(compiled))
                request = compiled.create_infer_request()
                request.set_input_tensor(ov.Tensor(data))
                request.infer()  # warm-up before the correctness check
                output = request.get_output_tensor().data.copy()
                row["max_abs_error"] = round(float(np.max(np.abs(output - reference))), 6)
                # NPU/GPU may compute in FP16, so allow FP16-level error.
                if not np.allclose(output, reference, rtol=1e-2, atol=1e-2):
                    row["status"] = "failed_correctness"
                else:
                    for _ in range(10):
                        request.infer()
                    times = []
                    for _ in range(a.runs):
                        start = time.perf_counter()
                        request.infer()
                        times.append((time.perf_counter() - start) * 1000)
                    row.update(status="executed", median_ms=round(statistics.median(times), 4),
                               p95_ms=round(float(np.percentile(times, 95)), 4),
                               min_ms=round(min(times), 4), max_ms=round(max(times), 4),
                               runs=a.runs)
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1200])
            results.append(row)
    print(json.dumps({"openvino_version": ov.__version__, "available_devices": available,
                      "results": results}, indent=2))

if __name__ == "__main__":
    main()
