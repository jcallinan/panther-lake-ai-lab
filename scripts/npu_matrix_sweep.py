"""Compare an identical synthetic OpenVINO matrix model across available devices.

This is a compatibility/latency microbenchmark, NOT an LLM or TOPS measurement.
Run: python scripts/npu_matrix_sweep.py --devices CPU GPU NPU
"""
import argparse
import json
import statistics
import time
import numpy as np
import openvino as ov

def model_for(width):
    x = ov.opset13.parameter([1, width], dtype=np.float32, name="features")
    w = ov.opset13.constant(np.full((width, width), 1.0 / width, dtype=np.float32))
    y = ov.opset13.matmul(x, w, False, False)
    return ov.Model([y], [x], f"matmul_{width}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    p.add_argument("--widths", nargs="+", type=int, default=[32, 128, 512])
    p.add_argument("--runs", type=int, default=25)
    a = p.parse_args()
    if a.runs < 1 or any(w < 1 or w > 4096 for w in a.widths):
        p.error("runs must be >=1 and widths must be between 1 and 4096")
    core = ov.Core()
    available = list(core.available_devices)
    results = []
    for width in a.widths:
        model = model_for(width)
        data = np.ones((1, width), dtype=np.float32)
        for device in a.devices:
            row = {"device": device, "width": width, "status": "skipped"}
            if device not in available:
                row["reason"] = "device not enumerated"
                results.append(row)
                continue
            try:
                compiled = core.compile_model(model, device)
                row["status"] = "compiled"
                try:
                    row["execution_devices"] = str(compiled.get_property("EXECUTION_DEVICES"))
                except Exception:
                    row["execution_devices"] = "unverified"
                reference = np.ones((1, width), dtype=np.float32)
                output = np.asarray(compiled([data])[compiled.output(0)])
                row["max_abs_error"] = float(np.max(np.abs(output - reference)))
                if not np.allclose(output, reference, rtol=1e-3, atol=1e-3):
                    row["status"] = "failed_correctness"
                else:
                    for _ in range(5):
                        compiled([data])
                    times = []
                    for _ in range(a.runs):
                        start = time.perf_counter()
                        compiled([data])
                        times.append((time.perf_counter() - start) * 1000)
                    row.update(status="executed", median_ms=round(statistics.median(times), 4),
                               min_ms=round(min(times), 4), max_ms=round(max(times), 4),
                               runs=a.runs)
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1200])
            results.append(row)
    print(json.dumps({"openvino_version": ov.__version__, "available_devices": available,
                      "results": results}, indent=2))

if __name__ == "__main__":
    main()
