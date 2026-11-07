"""OpenVINO matrix-inference smoke test, not an LLM benchmark."""
import argparse
import json
import statistics
import time

import numpy as np
import openvino as ov

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    core = ov.Core()
    if args.device not in core.available_devices:
        raise SystemExit(f"Device {args.device} unavailable: {core.available_devices}")
    x = ov.opset13.parameter([1, 128], dtype=np.float32, name="x")
    weights = ov.opset13.constant(np.ones((128, 128), dtype=np.float32))
    model = ov.Model([ov.opset13.matmul(x, weights, False, False)], [x], "matmul")
    compiled = core.compile_model(model, args.device)
    data = np.ones((1, 128), dtype=np.float32)
    for _ in range(5):
        compiled([data])
    durations = []
    for _ in range(args.runs):
        start = time.perf_counter()
        compiled([data])
        durations.append((time.perf_counter() - start) * 1000)
    output = {"requested_device": args.device, "runs": args.runs,
              "median_ms": round(statistics.median(durations), 4),
              "min_ms": round(min(durations), 4)}
    try:
        output["execution_devices"] = str(compiled.get_property("EXECUTION_DEVICES"))
    except Exception:
        output["execution_devices"] = "not reported"
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
