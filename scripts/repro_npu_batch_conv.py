"""Minimal reproduction: batched 3x3 conv stacks on the Intel NPU return sample 0's result for every batch element.

Compares NPU output against the CPU plugin (FP32) for the same model, across batch sizes and stack depths.
Run: python scripts/repro_npu_batch_conv.py [--batches 1 2 4 8 12 16 32] [--layers 1 4] [--res 128]
Prints one JSON object (versions, driver, and a per-config verdict).
"""
import argparse
import json
import subprocess

import numpy as np
import openvino as ov
import openvino.opset13 as ops


def build(batch, layers, ch, res, rng):
    x = ops.parameter([batch, 3, res, res], np.float32, name="input")
    h, cin = x, 3
    for i in range(layers):
        w = rng.normal(0, np.sqrt(2 / (cin * 9)), (ch, cin, 3, 3)).astype(np.float32)
        stride = [2, 2] if i == 0 else [1, 1]
        h = ops.relu(ops.convolution(h, ops.constant(w), stride, [1, 1], [1, 1], [1, 1]))
        cin = ch
    y = ops.reduce_mean(h, ops.constant(np.array([2, 3], dtype=np.int64)), False)
    return ov.Model([y], [x], f"conv{layers}_b{batch}")


def npu_driver():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_PnPSignedDriver | Where-Object DeviceName -match "
                              "'^Intel\\(R\\) (NPU|AI Boost)$' | Select-Object -First 1).DriverVersion"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16, 32])
    p.add_argument("--layers", nargs="+", type=int, default=[1, 4])
    p.add_argument("--ch", type=int, default=32)
    p.add_argument("--res", type=int, default=128)
    a = p.parse_args()
    core = ov.Core()
    rows = []
    for layers in a.layers:
        for b in a.batches:
            rng = np.random.default_rng(2026)
            model = build(b, layers, a.ch, a.res, rng)
            x = rng.normal(0, 1, (b, 3, a.res, a.res)).astype(np.float32)
            ref = core.compile_model(model, "CPU")([x])[0]
            npu = core.compile_model(model, "NPU")
            out = np.asarray(npu([x])[0])
            scale = float(np.abs(ref).max()) or 1.0
            err = float(np.abs(out - ref).max()) / scale
            row = {"layers": layers, "batch": b, "max_rel_err_vs_cpu": round(err, 6), "ok": err < 1.5e-2,
                   "execution_devices": list(npu.get_property("EXECUTION_DEVICES"))}
            if b > 1:
                row["outputs_replicate_sample0"] = bool(np.allclose(out[1:], out[0:1], rtol=1e-3, atol=1e-4))
                row["max_rel_err_vs_sample0_ref"] = round(float(np.abs(out[1:] - ref[0:1]).max()) / scale, 6)
            rows.append(row)
    print(json.dumps({"openvino": ov.__version__, "npu_device": core.get_property("NPU", "FULL_DEVICE_NAME"),
                      "npu_driver": npu_driver(), "channels": a.ch, "input_res": a.res, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
