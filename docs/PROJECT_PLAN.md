# Panther Lake AI Lab — Project Plan

**Repository:** [jcallinan/panther-lake-ai-lab](https://github.com/jcallinan/panther-lake-ai-lab)  
**Status:** Initial implementation underway  
**Initial target:** Lenovo IdeaPad 5 2-in-1 15IPH11, Intel Core Ultra 5 322  
**Purpose:** Public, reproducible research into practical CPU, integrated GPU, and NPU acceleration on Intel Core Ultra Series 3 (Panther Lake).

## 1. Project vision

Build a public hardware research and benchmark repository that tests **every practical local AI execution pathway** available on this machine, rather than testing only Ollama. Capture both successful and unsuccessful configurations, record the actual hardware execution backend, and publish comparable measurements and reproducible instructions so other Panther Lake owners can repeat the experiments.

The central research question is: **Which local AI workloads can actually use the Core Ultra 5 322 CPU, Xe3 integrated graphics, and NPU, and how do performance, power consumption, and usability compare?**

The project is exploratory. Compatibility is not guaranteed merely because an API, execution provider, or accelerator is advertised.

## 2. Initial test hardware

| Component | Baseline / planned verification |
|---|---|
| Manufacturer and model | Lenovo IdeaPad 5 2-in-1 15IPH11 |
| Machine type / configuration | 83UL / 83UL0000US |
| CPU | Intel Core Ultra 5 322, Panther Lake |
| CPU topology | 2 performance cores + 4 low-power efficiency cores; 6 threads |
| GPU | Integrated Intel Xe3-generation graphics; shared system RAM |
| NPU | Intel NPU, advertised up to 46 TOPS |
| GPU AI performance | Advertised up to 18 TOPS |
| Installed RAM | 16 GB DDR5; determine module count and channel configuration |
| Upgrade target | 32 GB DDR5 (verify compatibility) |
| SSD | Factory 512 GB NVMe |
| Expansion | Additional M.2 slot reported in platform documentation; verify installed configuration |
| OS | Windows 11 Home |
| Battery | 60 Wh |

**Important:** The advertised TOPS values are theoretical, workload-specific figures; they are not directly equivalent to LLM tokens/second or GPU TFLOPS. Verify precise hardware details, driver versions, BIOS, memory layout and storage model with the inventory script. Do not publish machine-specific identifiers or serial numbers.

## 3. Initial experiment board

The following ten workstreams were proposed in the original conversation. Each starts as **not tested**, not as an established capability.

| # | Workstream | Initial experiment / deliverable | Status |
|---|---|---|---|
| 1 | Hardware inventory | CPU, GPU, NPU, RAM, SSD, BIOS and drivers | Script committed; execution pending |
| 2 | OpenVINO | Compare CPU vs GPU vs NPU; discover devices and run identical models | Discovery and smoke-test scripts committed |
| 3 | Ollama | CPU inference and any available supported GPU backends | Not tested |
| 4 | llama.cpp | CPU, Vulkan, SYCL and other supported backends | Not tested |
| 5 | ONNX Runtime | Native CPU and OpenVINO execution-provider testing | Not tested |
| 6 | Windows ML | Enumerate and test available execution providers | Not tested |
| 7 | Intel AI Playground | Installation, compatibility, and supported AI workflows | Not tested |
| 8 | Generative AI | LLMs, embeddings, speech, image and vision models | Not tested |
| 9 | Efficiency | Latency, tokens/sec, watts, thermals and battery | Not tested |
| 10 | Upgrade testing | 16 GB vs 32 GB; single-channel vs dual-channel memory | Not tested |

### Kanban progression

- **Backlog:** Planned experiments without an environment or confirmed prerequisites.
- **Ready:** Test instructions, model, expected outputs, and prerequisites defined.
- **In progress:** Running experiments and collecting evidence.
- **Blocked:** Missing driver, runtime, hardware access, or compatible model.
- **Review:** Verify output, actual execution device, reproducibility and interpretation.
- **Done:** Commit instructions, versions, results, logs, and conclusions.

A device's presence in a list is **detected**, not **executed**. Track detection, model compilation, actual inference, and benchmark completion separately.

## 4. Original proposed repository structure

This is the **target structure from the planning conversation**, not a claim that all files already exist:

```text
panther-lake-ai-lab/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── hardware/
│   ├── lenovo-15iph11.md
│   └── inventory.ps1
├── benchmarks/
│   ├── cpu/
│   ├── gpu/
│   ├── npu/
│   └── results/
├── scripts/
│   ├── detect_devices.py
│   ├── benchmark_openvino.py
│   ├── benchmark_ollama.py
│   └── system_metrics.ps1
├── models/
│   └── test-matrix.md
├── docs/
│   ├── installation.md
│   ├── compatibility-matrix.md
│   └── troubleshooting.md
└── .github/
    └── workflows/
        └── validate.yml
```

### Implementation status and gaps

Already committed at project initialization: `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, `hardware/inventory.ps1`, `hardware/lenovo-15iph11.md`, `scripts/detect_devices.py`, `scripts/benchmark_openvino.py`, `docs/compatibility-matrix.md`, `benchmarks/results/.gitkeep`, and `.github/workflows/validate.yml`.

Planned but **not yet implemented**: `scripts/benchmark_ollama.py`, `scripts/system_metrics.ps1`, `models/test-matrix.md`, `docs/installation.md`, `docs/troubleshooting.md`, and the per-device benchmark directories. Git does not track empty directories.

## 5. Execution frameworks and test matrix

| Framework | CPU path | GPU path | NPU path | Key verification |
|---|---|---|---|---|
| OpenVINO | Candidate | Candidate | Candidate | Report available devices, compile model, confirm execution |
| Ollama | Candidate | Investigate supported backends | Do not assume supported | Confirm actual backend in runtime logs |
| llama.cpp | Candidate | Vulkan/SYCL if supported by installed build | Experimental / investigate | Record build flags and backend |
| ONNX Runtime | CPU EP | OpenVINO EP where available | OpenVINO EP where available | Inspect installed providers and execution assignment |
| Windows ML | Candidate | Provider-dependent | Provider-dependent | Enumerate providers and verify placement |
| Intel AI Playground | Application-dependent | Application-dependent | Application-dependent | Verify model and device requirements |

Do not mark a backend supported until an actual model runs successfully on that device. Fallback to CPU must be recorded explicitly.

## 6. Workload categories

1. **Language models:** Small quantized LLMs, prompt processing, token generation, long-context behavior.
2. **Embeddings:** Sentence embeddings, semantic similarity, vector search and retrieval.
3. **Speech:** Speech-to-text, real-time transcription, text-to-speech where supported.
4. **Computer vision:** Classification, detection, OCR and image embeddings.
5. **Generative imaging:** Diffusion and other supported image generation paths.
6. **Video/media:** Intel Quick Sync encode/decode, AV1 and other supported codecs; distinguish media acceleration from neural inference.
7. **Mixed workloads:** Concurrent inference with background desktop, browser, development, and Docker activity.
8. **Battery and thermals:** Sustained inference, throttling, energy per request, plugged-in versus battery.

## 7. Benchmark methodology

For every run, capture:

- Timestamp, OS build, power plan, plugged-in/battery state and thermal conditions.
- CPU, GPU and NPU device IDs; BIOS, driver, runtime and package versions.
- RAM capacity, speed, channel configuration, and background load.
- Model name, version, checksum, format, quantization and context length.
- Requested backend **and observed actual execution device**.
- Warm-up count, timed iterations, prompt/input size and output length.
- First-token latency, generated tokens/second, total latency and peak memory where applicable.
- Power consumption, battery drain, temperatures and throttling if measured by a reliable method.
- Error logs, unsupported operations and fallback behavior.

**Rules:** Compare the same model and inputs, separate cold start from warmed runs, report median and variability, and never compare a tiny matrix smoke test directly with LLM generation speed. Report unsupported and failed configurations without fabricating benchmark values.

### Suggested result schema

```json
{
  "run_id": "example-001",
  "hardware": "Lenovo 15IPH11 / Core Ultra 5 322",
  "runtime": "OpenVINO",
  "runtime_version": null,
  "model": null,
  "precision": null,
  "requested_device": "NPU",
  "observed_device": null,
  "status": "not_tested",
  "latency_ms_median": null,
  "tokens_per_second": null,
  "power_watts": null,
  "notes": ""
}
```

Null values indicate not measured or unavailable, not zero.

## 8. Development roadmap

### Phase 0 — Repository and documentation
- [x] Create public repository.
- [x] Commit initial README, hardware inventory, OpenVINO discovery and smoke benchmark.
- [x] Add full project plan with experiment board and proposed file tree.
- [ ] Define issue labels, contribution guidance, reproducibility requirements and benchmark result template.

### Phase 1 — Baseline device discovery
- [ ] Run `hardware/inventory.ps1` on the Lenovo.
- [ ] Record Intel GPU and NPU driver versions and Windows build.
- [ ] Install supported Python and OpenVINO packages.
- [ ] Run `scripts/detect_devices.py` and save sanitized output.
- [ ] Run the CPU smoke benchmark, then test GPU and NPU only if detected.

### Phase 2 — CPU and GPU local AI
- [ ] Implement Ollama benchmarking.
- [ ] Test llama.cpp CPU backend.
- [ ] Investigate supported GPU backends and record any fallback.
- [ ] Compare small quantized language models and embedding models.

### Phase 3 — NPU-specific research
- [ ] Run compatible OpenVINO models on NPU.
- [ ] Evaluate ONNX Runtime/OpenVINO provider paths.
- [ ] Test Windows ML provider discovery and model execution.
- [ ] Explore Intel AI Playground and supported NPU workflows.

### Phase 4 — Real-world workloads and efficiency
- [ ] Benchmark speech, vision, embeddings, generation and media acceleration.
- [ ] Add temperature, power and battery collection.
- [ ] Compare performance per watt and sustained performance.

### Phase 5 — Hardware upgrades and publication
- [ ] Document 16 GB baseline RAM configuration.
- [ ] Test dual-channel behavior and optional 32 GB upgrade.
- [ ] Compare results before/after hardware or driver changes.
- [ ] Publish a dashboard, compatibility tables, and contribution instructions.

## 9. Initial run commands

```powershell
git clone https://github.com/jcallinan/panther-lake-ai-lab.git
cd panther-lake-ai-lab
powershell -ExecutionPolicy Bypass -File .\hardware\inventory.ps1
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\detect_devices.py
.\.venv\Scripts\python.exe scripts\benchmark_openvino.py --device CPU
```

Run GPU/NPU smoke tests only after confirming their presence in device discovery. These are initial environment checks, not proof of complete framework support.

## 10. Success criteria

The project succeeds when another user with a compatible Panther Lake machine can follow the instructions, reproduce test results, identify which processor actually executed the workload, and compare realistic speed and energy measurements across available backends.

The long-term value is an evidence-based compatibility library rather than a collection of marketing TOPS claims.

## 11. Reference documentation

- Lenovo PSREF: https://psref.lenovo.com/product/ideapad_5_2_in_1_15iph11?tab=spec
- Intel OpenVINO: https://docs.openvino.ai/
- ONNX Runtime: https://onnxruntime.ai/
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Ollama: https://github.com/ollama/ollama
- Windows ML: https://learn.microsoft.com/windows/ai/
