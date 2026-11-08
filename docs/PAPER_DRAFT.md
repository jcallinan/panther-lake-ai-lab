# Evaluating Heterogeneous Edge AI Inference on Intel Panther Lake
## Performance, Energy Efficiency, Runtime Compatibility, and Practical Deployment Constraints

**Manuscript status:** Research proposal and initial academic draft — no experimental results yet  
**Target level:** Master's-level empirical computer systems paper  
**Repository:** https://github.com/jcallinan/panther-lake-ai-lab  
**Author:** To be confirmed  
**Date:** November 2026

> **Integrity statement:** This is a preregistration-style manuscript draft. No benchmark numbers, statistical significance, runtime compatibility, or experimental outcomes are claimed without measurements. The conclusions are hypotheses, not findings.

## Abstract (proposal-stage)

Modern client processors increasingly integrate general-purpose CPU cores, programmable graphics, and specialized neural processing units (NPUs), promising private and energy-efficient artificial intelligence inference at the edge. However, peak theoretical AI throughput alone does not establish application-level performance, and heterogeneous runtime support may introduce significant deployment barriers. This study proposes a reproducible experimental characterization of the Intel Core Ultra 5 322 Panther Lake platform in a consumer Lenovo IdeaPad 5 2-in-1 laptop. We will evaluate model compatibility, inference latency, throughput, memory consumption, energy usage, thermal stability, and backend selection across CPU, integrated GPU, and NPU execution pathways. The methodology separates device enumeration, successful compilation, actual execution, and sustained benchmark completion. We propose controlled workloads spanning transformer inference, embeddings, speech, and computer vision, with attention to model precision and operator coverage. We further investigate the effects of shared memory, memory capacity, power modes, and sustained thermal constraints. The intended contributions are an openly documented compatibility dataset, a validated benchmarking framework, and evidence-based guidance for local AI deployment on heterogeneous consumer systems.

**Keywords:** edge AI; heterogeneous computing; neural processing unit; Intel Panther Lake; OpenVINO; inference benchmarking; energy efficiency; reproducibility.

## 1. Introduction

Local AI inference has become attractive for privacy-sensitive processing, low-latency applications, offline operation, and reduced dependence on remote services. At the same time, client processors have evolved from CPU-centric designs into heterogeneous systems containing CPUs, integrated GPUs, and dedicated NPUs. This architectural diversity complicates the interpretation of vendor peak throughput specifications: a nominally capable accelerator may be unavailable to a given runtime, unable to compile a model's operators, constrained by model memory requirements, or slower than another device for small workloads due to execution overhead.

This paper investigates the gap between **theoretical hardware capability**, **software-visible compatibility**, and **measured application performance**. The primary test platform is an Intel Core Ultra 5 322-based Lenovo IdeaPad 5 2-in-1 15IPH11. Its integrated graphics and NPU permit investigation of several accelerator pathways on one affordable, portable system, subject to driver and runtime support.

### 1.1 Research problem

Public product specifications often report aggregate AI operations per second, but developers must choose among runtime providers, quantization formats, operator sets, and device-specific execution paths. There is a need for transparent, reproducible measurements that distinguish unsupported configurations from actual inference and that account for energy and thermal effects.

### 1.2 Research questions

- **RQ1 (compatibility):** Which model/runtime/device combinations can be detected, compiled, executed, and benchmarked successfully?
- **RQ2 (performance):** For compatible identical workloads, how do CPU, GPU, and NPU differ in latency, throughput, and memory usage?
- **RQ3 (energy):** How do performance-per-watt and energy-per-inference differ under plugged-in and battery-powered conditions?
- **RQ4 (stability):** How do performance and thermals evolve during sustained inference?
- **RQ5 (memory):** What changes with memory capacity and channel configuration, while controlling for other factors?
- **RQ6 (usability):** What installation, driver, conversion, and fallback barriers limit reproducibility for developers?

### 1.3 Hypotheses (to be tested)

- **H1:** Supported NPU workloads can improve energy per inference over CPU execution, but not necessarily latency.
- **H2:** Compatibility and operator coverage materially limit the practical advantage of heterogeneous acceleration.
- **H3:** Larger batch sizes change the relative ranking of CPU, GPU, and NPU.
- **H4:** Sustained performance rankings differ from short benchmark rankings because of thermal and power limits.
- **H5:** RAM configuration affects integrated-GPU and larger-model performance, though the magnitude depends on the workload.

## 2. Background and related work — research tasks

This section will synthesize peer-reviewed literature on:
1. Heterogeneous CPU/GPU/NPU scheduling and execution.
2. Edge inference benchmarking, including latency-versus-throughput tradeoffs.
3. Energy-aware inference and measurement validity.
4. Model quantization and operator compatibility.
5. Reproducibility and statistical methodology in computer systems.
6. Client-side AI privacy and offline inference.

**Literature review requirement:** Build a source table with DOI, venue, year, methods, hardware, metrics, findings, limitations, and relevance. Prefer primary peer-reviewed publications. Vendor documentation should establish implementation details, not substitute for scientific evidence.

## 3. Experimental platform

### 3.1 Baseline hardware

- Lenovo IdeaPad 5 2-in-1 15IPH11; machine type 83UL, configuration 83UL0000US.
- Intel Core Ultra 5 322 Panther Lake CPU.
- Integrated Intel graphics and dedicated Intel NPU.
- Factory 16 GB DDR5 RAM and 512 GB NVMe SSD.
- Windows 11 as the initial host OS.

**All actual device identifiers, driver versions, BIOS, RAM module count and speed, storage characteristics, firmware power settings, and thermal environment must be measured before testing.** Hardware specifications from the Lenovo family documentation are provisional until verified on this unit.

### 3.2 Runtimes

OpenVINO, ONNX Runtime with supported execution providers, Windows ML, llama.cpp, Ollama, and Intel AI Playground are investigation candidates. Do not assume that any particular runtime supports the NPU. A compatible model must compile and execute on the intended accelerator to qualify as supported.

### 3.3 Workloads

Use a staged model suite:
- **Microbenchmarks:** minimal supported operator graphs for device validation only.
- **Computer vision:** classification or detection with public, versioned models and fixed input images.
- **Embeddings:** public sentence-embedding models, fixed sequences, cosine-similarity correctness checks.
- **Speech:** fixed public audio clips with latency and word error rate where ground truth exists.
- **LLMs:** versioned, legally distributable small quantized models, fixed prompts and context lengths; report prefill and decode separately.
- **Optional image generation:** only when the runtime and memory budget permit.

Model families, licenses, hashes, input sets, and reference outputs must be preregistered before comparative measurement.

## 4. Experimental design

### 4.1 Compatibility ladder

For each runtime × model × precision × requested device:
1. Detect the device.
2. Load or convert the model.
3. Compile the graph.
4. Execute a correctness test.
5. Confirm actual execution placement and any fallback.
6. Run repeatable timed trials.
7. Capture failures and exact error messages.

Do not treat device detection, successful installation, or compilation as proof of successful accelerated inference.

### 4.2 Independent variables

- Compute device (CPU/GPU/NPU), where the same model and precision can run.
- Runtime and execution provider.
- Model architecture, size, precision and quantization.
- Input size, sequence length, context length, batch size.
- Power mode (fixed and recorded).
- Plugged-in versus battery.
- Memory configuration (16 GB baseline versus later 32 GB, if upgraded).

### 4.3 Dependent variables

- End-to-end latency and steady-state latency (milliseconds).
- Time to first token, prefill tokens/s, decode tokens/s for LLMs.
- Throughput (inferences/s) for non-generative models.
- Peak resident memory and, where available, accelerator memory usage.
- Energy per inference (joules), measured with validated telemetry.
- Power draw (watts), battery drain, temperatures, and throttling.
- Correctness and model quality changes introduced by quantization.
- Compatibility success rate, setup time and documented blockers.

### 4.4 Controls

Fix OS and driver versions within a comparison block; record BIOS, background load, power profile, ambient temperature, and cooling configuration. Use identical model weights, input sets, warmups, and measurement windows across devices when technically possible. When precision or graph transformations differ, report results as separate comparisons rather than implying strict equivalence. Randomize test order where practical and allow cooling recovery between sustained-load trials.

### 4.5 Repetitions and statistics

Perform an initial pilot to estimate variability, then set trial counts and duration based on observed variance and a prespecified precision target. Report median, p95, dispersion, and confidence intervals using appropriate bootstrap or paired methods. For repeated measurements on one physical machine, distinguish **within-device repeatability** from generalizability across different units. Avoid claiming population-level significance from one laptop.

### 4.6 Power measurement validity

Operating-system battery estimates are not a substitute for calibrated power measurements. Identify and document telemetry source, sampling interval, uncertainty, and whether it represents package, system, or wall power. Where reliable energy measurement is unavailable, report latency/throughput and label power findings exploratory. Battery discharge tests should use controlled charge ranges and background conditions.

### 4.7 Threats to validity

Single-device sampling; firmware changes; thermal variation; driver maturity; quantization differences; operator fallback; shared-memory contention; measurement overhead; runtime-specific optimizations; nonrepresentative prompts; and publication bias toward successful configurations.

## 5. Planned results presentation

**Table 1:** Hardware and software inventory with version hashes.  
**Table 2:** Runtime × model × device compatibility ladder.  
**Figure 1:** Median latency and 95% intervals for equivalent workloads.  
**Figure 2:** Throughput versus batch size.  
**Figure 3:** Energy per inference versus latency (Pareto frontier).  
**Figure 4:** Sustained throughput, temperatures, and power over time.  
**Figure 5:** RAM configuration comparison (only if upgrade controlled).  
**Appendix:** Reproduction commands, failure logs, raw result schema and complete configuration manifest.

No tables or plots should contain fabricated results or unverified compatibility claims.

## 6. Discussion — planned structure

Interpret whether accelerator benefits are primarily determined by arithmetic throughput, model fit, precision, operator support, runtime scheduling, memory bandwidth, or energy constraints. Explain negative results and software barriers. Separate observations supported by data from hypotheses and manufacturer claims.

## 7. Conclusion — placeholder

A defensible conclusion will be written only after experiments, statistical analysis, and independent reproduction. The expected deliverable is a practical, reproducible characterization of heterogeneous local AI inference on a consumer Panther Lake platform.

## 8. Reproducibility and artifact availability

The public repository will provide pinned dependencies, hardware manifests, scripts, fixed test datasets or acquisition instructions, model checksums, results in machine-readable form, analysis notebooks, and exact figure-generation commands. Sensitive device identifiers and licensed model weights will not be committed without authorization.

## 9. Preliminary technical references (not a substitute for peer-reviewed literature)

- Intel OpenVINO documentation: https://docs.openvino.ai/
- ONNX Runtime documentation: https://onnxruntime.ai/docs/
- Microsoft Windows AI documentation: https://learn.microsoft.com/windows/ai/
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Ollama: https://github.com/ollama/ollama
- Lenovo PSREF: https://psref.lenovo.com/product/ideapad_5_2_in_1_15iph11?tab=spec

**Next writing milestone:** Complete a verified literature review with 20–30 relevant scholarly sources, identify the closest prior studies, and sharpen the novel contribution before selecting a conference venue.
