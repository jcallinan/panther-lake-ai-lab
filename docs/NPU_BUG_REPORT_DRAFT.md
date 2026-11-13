# DRAFT bug report: NPU returns sample 0's output for every batch element (conv stacks ≥ 3 layers, batch 2–8)

> **Status: draft, not filed.** Suggested destination: [openvinotoolkit/openvino issues](https://github.com/openvinotoolkit/openvino/issues), using the bug template with the NPU plugin component. The [Intel Community forum for Intel Distribution of OpenVINO Toolkit](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit) is an alternative. Before filing, search existing issues for "NPU batch" and check the latest release notes.

---

## Summary

On an Intel Core Ultra 5 322 (Panther Lake, NPU 5), a model made of **3 or more** 3×3 convolutions (stride 2, then stride 1, each followed by ReLU, then a spatial `ReduceMean`) produces wrong results on the NPU at **batch sizes 2, 4 and 8**. Every element of the batch gets **the result for sample 0**. The same model is correct on the NPU at batch 1 and at batch ≥ 12, and correct on the CPU and GPU at all batch sizes. Models with 1–2 conv layers are correct at all batch sizes.

`EXECUTION_DEVICES` reports `['NPU']`. No error or warning is raised, so the failure is silent.

## Environment

| | |
|---|---|
| CPU / NPU | Intel Core Ultra 5 322 (Panther Lake). NPU reported as "Intel(R) AI Boost" (NPU 5) |
| NPU driver | **32.0.100.5540** (also reproduced on 32.0.100.4724) |
| Graphics driver | 32.0.101.9175 |
| OS | Windows 11 (build 26300) |
| OpenVINO | 2026.4.1 (`2026.4.1-22982-e213a147257-releases/2026/4`), pip wheel, Python 3.12.10 |
| Laptop | Lenovo IdeaPad 5 2-in-1 15IPH11, on AC power |

## Reproduction

The script is standalone, needs only `openvino` and `numpy`, and has no model files: [`scripts/repro_npu_batch_conv.py`](../scripts/repro_npu_batch_conv.py).

```powershell
python scripts/repro_npu_batch_conv.py                      # layers 1 and 4, batches 1 2 4 8 12 16 32, res 128
python scripts/repro_npu_batch_conv.py --layers 2 3 --batches 2 8 12
```

The model is input `[B,3,R,R]` → Conv3×3 (32 channels, stride 2, pad 1) + ReLU → (L−1) × [Conv3×3 (32→32, stride 1, pad 1) + ReLU] → ReduceMean over H,W → `[B,32]`. FP32 weights; seeded random weights and inputs. The NPU output is compared with the CPU plugin output for the same model.

## Observed (driver 32.0.100.5540, 2026-11-12)

| Conv layers | Batch | Max rel. error vs CPU | Outputs all equal to sample 0? |
|---|---|---|---|
| 1 | 1, 2, 4, 8, 12, 16, 32 | ≤ 4.5e-4 ✅ | no |
| 2 | 2, 8, 12 | ≤ 3.9e-4 ✅ | no |
| 3 | 2 | 3.2e-2 ❌ | **yes** |
| 3 | 8 | 4.2e-2 ❌ | **yes** |
| 3 | 12 | 5.5e-4 ✅ | no |
| 4 | 1 | 4.2e-4 ✅ | n/a |
| 4 | 2, 4, 8 | 1.6e-2 – 2.7e-2 ❌ | **yes** (each element matches the CPU result for sample 0 within 4.2e-4) |
| 4 | 12, 16, 32 | ≤ 6.1e-4 ✅ | no |
| 4 (res 64) | 8 | ❌ | **yes** |

The error is only 2–4% relative here because the output is a spatial mean. Comparing the full feature map, before `ReduceMean`, earlier runs showed 78–92% error.

## Expected

Batch element *i* of the NPU output should match the CPU output for sample *i*, within FP16 tolerance (about 1e-3 relative), for every batch size, as it already does at batch 1 and batch ≥ 12.

## Additional observations (possibly related)

- On the same NPU, **MLP stacks at batch 8 run 6–7× slower than at batch 32** (MLP-L: 24.2 ms at b8 vs 3.9 ms at b32), although their results are correct. This suggests batch 2–8 takes a different, unoptimized compilation path.
- The issue reproduces whether the weights are FP32 constants, FP16 constants with a Convert, or INT8 constants with a Convert and Multiply (weight-only).
- The broader sweep where this was first found is in [`examples/perf_atlas`](../examples/perf_atlas/) (`benchmarks/results/perf_atlas.json`, `batch_diagnosis` field).

## Workarounds

- Use batch 1 (separate infer requests or `AsyncInferQueue`), or batch ≥ 12.
- Validate NPU outputs against a CPU reference before trusting batched results.
