"""Synthetic IT/industrial anomaly scoring using OpenVINO on CPU/GPU/NPU.

Illustrates inference compatibility only. Random weights are NOT a trained detector.
Run: python scripts/npu_anomaly_demo.py --devices CPU GPU NPU
"""
import argparse
import json
import statistics
import time
import numpy as np
import openvino as ov

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >=1")
    rng = np.random.default_rng(2026)
    x = ov.opset13.parameter([1, 8], dtype=np.float32, name="synthetic_sensor_features")
    w1 = ov.opset13.constant(rng.normal(0, 0.1, (8, 32)).astype(np.float32))
    w2 = ov.opset13.constant(rng.normal(0, 0.1, (32, 1)).astype(np.float32))
    hidden = ov.opset13.relu(ov.opset13.matmul(x, w1, False, False))
    score = ov.opset13.sigmoid(ov.opset13.matmul(hidden, w2, False, False))
    model = ov.Model([score], [x], "synthetic_anomaly_score")
    samples = rng.normal(0, 1, (100, 8)).astype(np.float32)
    w1_np = w1.get_data()
    w2_np = w2.get_data()
    reference = [float(1 / (1 + np.exp(-(np.maximum(samples[i % len(samples)] @ w1_np, 0) @ w2_np)[0])))
                 for i in range(args.runs)]
    core = ov.Core()
    results = []
    for device in args.devices:
        row = {"device": device, "status": "skipped"}
        if device not in core.available_devices:
            row["reason"] = "device not enumerated"
            results.append(row)
            continue
        try:
            start = time.perf_counter()
            compiled = core.compile_model(model, device)
            row["compile_s"] = round(time.perf_counter() - start, 3)
            try:
                value = compiled.get_property("EXECUTION_DEVICES")
                row["execution_devices"] = [value] if isinstance(value, str) else [str(v) for v in value]
            except Exception:
                row["execution_devices"] = ["unverified"]
            request = compiled.create_infer_request()
            for _ in range(5):
                request.infer([samples[0:1]])
            times = []
            outputs = []
            for i in range(args.runs):
                sample = samples[i % len(samples):i % len(samples) + 1]
                start = time.perf_counter()
                request.infer([sample])
                times.append((time.perf_counter() - start) * 1000)
                outputs.append(request.get_output_tensor().data.copy())
            values = [float(o.reshape(-1)[0]) for o in outputs]
            max_err = max(abs(v - r) for v, r in zip(values, reference))
            row.update(status="executed" if max_err < 1e-2 else "failed_correctness",
                       median_ms=round(statistics.median(times), 4),
                       p95_ms=round(float(np.percentile(times, 95)), 4),
                       runs=args.runs, example_score=round(values[0], 6),
                       max_abs_error_vs_numpy=round(max_err, 6))
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1200])
        results.append(row)
    print(json.dumps({"disclaimer": "Synthetic inputs and random weights; not a validated anomaly detector.",
                      "openvino_version": ov.__version__, "results": results}, indent=2))

if __name__ == "__main__":
    main()
