# NPU Experiment Suite — Research Backlog

**Platform:** Lenovo IdeaPad 5 2-in-1 15IPH11 / Core Ultra 5 322. **Status:** Proposed; no on-device results yet.

## Research principles
- Compare identical model graph, weights, precision, input and shape across devices wherever supported.
- Record detected vs compiled vs executed vs benchmarked separately; do not assume NPU support or silently substitute CPU.
- Treat different precisions, conversions and unsupported operators as distinct comparisons.
- Report correctness against a CPU reference and latency distributions, not just advertised TOPS.
- Keep synthetic industrial/IT data clearly labeled. Never claim that synthetic anomaly performance validates a real safety or security system.
- Keep battery power measurements exploratory unless instrumentation is validated.

## Ten proposed experiments

| ID | Experiment | Hypothesis and measurable outcome | First implementation |
|---|---|---|---|
| NPU-01 | CPU/GPU/NPU common-model benchmark | Acceleration changes latency depending on model size and batch; median and p95 | `scripts/npu_matrix_sweep.py` |
| NPU-02 | Offline intent/embedding assistant | Small models may be viable offline; accuracy, latency, memory | Pin an openly licensed embedding model and prompts |
| NPU-03 | Vision inference | Sustained classification/detection reveals thermal/throughput tradeoffs | Select a fixed public model and image set |
| NPU-04 | Speech intelligence | Continuous audio classification may benefit from efficient inference | Fixed public clips and correctness baseline |
| NPU-05 | IT telemetry anomaly detection | Small networks can flag synthetic anomalous events with low latency | `scripts/npu_anomaly_demo.py` |
| NPU-06 | Industrial sensor analytics | Simulated process measurements enable reproducible edge tests | `scripts/npu_anomaly_demo.py` |
| NPU-07 | Battery-powered inference | NPU may reduce energy per inference for compatible models | Validated power measurement and repeated sessions |
| NPU-08 | Quantization and operator coverage | Precision/operator constraints determine practical support | FP32/FP16/INT8 model conversion matrix |
| NPU-09 | Concurrent workloads | Offloading may preserve CPU responsiveness | Controlled concurrent compile/web/AI runs |
| NPU-10 | NPU compatibility explorer | Failure taxonomy exposes operator and runtime gaps | `scripts/npu_matrix_sweep.py` and structured failure logs |

## First-night order
1. Run hardware inventory and `scripts/detect_devices.py`.
2. Run `scripts/npu_matrix_sweep.py --devices CPU GPU NPU`; unavailable devices are reported as skipped.
3. Run `scripts/npu_anomaly_demo.py --devices CPU GPU NPU` on **synthetic** data.
4. Save terminal output, runtime/driver versions, and errors. Do not commit raw hardware identifiers.

## Study design
For each experiment, define model hash, runtime version, device, precision, inputs, warmup, iterations, CPU-reference correctness tolerance, latency distribution, temperature/power source, and explicit failure category. Prefer multiple sessions with randomized device order. Publish negative results and repeatability limits.

## Conference paper opportunities
1. **Compatibility ladder:** quantify detected/compiled/executed/benchmarked rates by device, operator and precision.
2. **Latency-energy Pareto frontiers:** use calibrated energy telemetry; compare same model/input across engines.
3. **Heterogeneous contention:** study NPU inference while CPU handles compilation and GPU renders UI.
4. **Cross-platform replication:** recruit additional Panther Lake and previous-generation Intel machines.

These are hypotheses and planned experiments, not validated findings.
