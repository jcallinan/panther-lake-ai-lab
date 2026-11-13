# Panther Lake Hardware Reference: Core Ultra 5 322 in the IdeaPad 5 2-in-1 15IPH11

Last researched: 2026-11-11

This page collects sourced facts about the test laptop described in [hardware/lenovo-15iph11.md](../hardware/lenovo-15iph11.md). Each claim links to its source. **Vendor claim** means Intel, Lenovo or Microsoft said it. **Independent** means a third party measured it. **(unverified)** means no confirming source was found. **Derived** means arithmetic or inference from cited figures, not a published number.

Locally observed on the test unit (not from web sources): OpenVINO 2026.4.1 on Windows 11 reports the GPU as `Intel(R) Graphics (iGPU)` and the NPU as `Intel(R) AI Boost`, with NPU optimization capabilities `FP16, INT8`.

---

## At a glance

| Item | Core Ultra 5 322 value | Source |
|---|---|---|
| Codename / family | Panther Lake, Intel Core Ultra Series 3 | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Launch | Q1'26 | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Process | CPU tile: Intel 18A. GPU tile: Intel 3 | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Cores / threads | 6C / 6T: 2 P-cores + 0 E-cores + 4 LP E-cores (no Hyper-Threading) | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html), [Lenovo PSREF](https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_5_2_in_1_15IPH11/IdeaPad_5_2_in_1_15IPH11_Spec.pdf) |
| P-core clocks | 2.5 GHz base / 4.4 GHz max turbo | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| LP E-core clocks | 1.9 GHz base / 3.3 GHz max turbo | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Cache | 12 MB Intel Smart Cache | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Power | 25 W base, 55 W max turbo, 12 W minimum assured | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Memory (SoC) | Up to DDR5-6400 or LPDDR5X-7467, 2 channels, 128 GB max | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Memory (this laptop) | DDR5-5600, two SO-DIMM slots, dual-channel, up to 32 GB offered | [Lenovo PSREF](https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_5_2_in_1_15IPH11/IdeaPad_5_2_in_1_15IPH11_Spec.pdf) |
| iGPU branding | **"Intel Graphics", not Intel Arc** | [Intel Quick Reference Guide](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf), [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| iGPU config | **2 Xe3 cores**, 2.3 GHz max, ray tracing, DirectX 12 Ultimate | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| GPU peak AI | **18 TOPS (INT8)** | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html), [Intel QRG](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf) |
| NPU peak AI | **46 TOPS (INT8)**, sparsity support: yes | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| Total platform TOPS | **Not published for this SKU.** ARK has no field for it. NPU + GPU = 64 TOPS (derived, excludes CPU). The family's "up to 180" applies only to select H SKUs. | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html), [Intel Edge overview](https://cdrdv2-public.intel.com/855291/Intel%C2%AE%20Core%E2%84%A2%20Ultra%20Processors%20Series%203%20for%20Edge%20Overview_2.pdf) |
| IPU | Intel IPU 7.5, 16 MP, 6 MIPI lanes / 3 ports | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| I/O | 12 PCIe lanes (PCIe 5.0 and 4.0), Thunderbolt 4 | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) |
| vPro | Not eligible (the sibling Core Ultra 5 332 is) | [Intel QRG](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf) |

---

## 1. Core Ultra 5 322 SKU details

### Where it sits in the line-up
- The 322 is the lowest Core Ultra 5 tier in Intel's [Quick Reference Guide SKU table](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf). Its 6 cores, 12 MB LLC, 46 NPU TOPS, Intel Graphics with 2 Xe cores at 2.3 GHz / 18 TOPS and 55 W max turbo match the Core Ultra 5 332. The 332 adds vPro and Intel SIPP eligibility.
- In the same guide, the Core Ultra 5 325 and 335 step up to 8 cores, 4 Xe cores (40 GPU TOPS) and 47 NPU TOPS. The Core Ultra 7 365 reaches 49 NPU TOPS. Only the H-series X7/X9 parts (12 Xe, 50 NPU TOPS) and the Core Ultra 5 338H (10 Xe, 47 NPU TOPS) carry Arc B390/B370 branding.
- Lenovo's [PSREF](https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_5_2_in_1_15IPH11/IdeaPad_5_2_in_1_15IPH11_Spec.pdf) offers this chassis with the 322, 325 or Core Ultra 7 355. The 322 row reads "Intel Graphics, up to 18 TOPS" and NPU "Up to 46 TOPS".

### GPU branding: this SKU is not "Arc"
- Footnote 2 of the [Quick Reference Guide](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf) says Built-in Intel Arc GPU "only available on select ... systems that meet minimum Xe core requirement. Other system configurations feature Intel® Graphics."
- [ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html) lists the GPU name as "Intel® Graphics" with 2 Xe-cores. This matches the OpenVINO device string `Intel(R) Graphics (iGPU)`.
- Third-party listings disagree on this SKU's GPU and NPU. Some list 4 Xe cores or 50 NPU TOPS. Intel's own pages above are authoritative: 2 Xe cores, 46 NPU TOPS.

### Xe3 config implied by 2 Xe cores (derived)
- Per [Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/), each Xe3 core keeps 8 × 512-bit vector engines (XVEs) and 8 × 2048-bit XMX matrix engines, the same as Xe2. Xe3 grows L1/SLM to 256 KB and lets each XVE track 10 threads, up from 8.
- That implies 16 XVEs and 16 XMX engines on this iGPU (derived). It is the smallest Panther Lake GPU configuration. The smallest *published* GPU tile has 4 Xe3 cores on Intel 3 ([Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)). The 322 likely uses that tile with 2 cores fused off **(unverified, inference)**.
- The 2P + 4LP-E CPU layout is likewise below the smallest published compute-tile layout (4P + 4LP-E). This suggests a harvested die **(unverified, inference)**.

### Launch
- ARK gives the launch as Q1'26. Intel unveiled Core Ultra Series 3 at CES on 2026-01-05, with systems shipping globally from 2026-01-27 ([TechSpot](https://www.techspot.com/news/110816-intel-launches-core-ultra-series-3-panther-lake.html)).

---

## 2. Panther Lake architecture overview

### Process: Intel 18A
- Intel 18A combines **RibbonFET** gate-all-around transistors with **PowerVia** backside power delivery. Intel claims up to 18% more performance at iso power, 38% lower power at iso performance, and 30% better density vs Intel 3 ([Intel 18A page](https://www.intel.com/content/www/us/en/foundry/process/18a.html), vendor claim).
- Intel calls Panther Lake the first client platform on 18A, built at Fab 52 in Arizona ([Intel architecture release](https://www.intel.com/content/www/us/en/newsroom/news/client-computing/intel-unveils-panther-lake-architecture-first-ai-pc-platform-built-on-18a.html), vendor claim).

### Tiles (Foveros)
Source: [Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)
- **Compute tile (Intel 18A):** CPU cores, LP E-cores, NPU, ISP/IPU, media engines, 8 MB memory-side cache, 128-bit memory bus.
- **GPU tile:** 4 Xe3 cores on Intel 3, or 12 Xe3 cores on TSMC N3E.
- **Platform controller tile (TSMC N6):** PCIe, Thunderbolt, USB, Wi-Fi 7, Bluetooth 6.0.
- Published die configs are 4P+4LPE, 4P+8E+4LPE, and 4P+8E+4LPE with 12 Xe. The 322's 2P+4LPE does not match any of them (see above).

### CPU cores
- **Cougar Cove P-cores** evolve Lion Cove: 50% larger TLB and a bigger branch predictor. **Darkmont E-cores** evolve Skymont: better branch prediction, prefetch and nanocode. Chips and Cheese calls the cores "evolution rather than revolution" ([Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)).
- Intel claims about 10% faster single-thread than Lunar Lake and Arrow Lake, and about 50% faster multi-thread than Lunar Lake (vendor claim, via [Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)).

### Xe3 GPU
- Render slices now hold up to 6 Xe cores (Xe2: 4), and the 12-core part has 16 MB L2. Ray-tracing units add dynamic ray management. Xe3 parts are marketed as Arc **B**-series, not C-series ([Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)). On the 322 the iGPU carries no Arc brand at all (see section 1).

### NPU 5 vs NPU 4 (Lunar Lake) vs NPU 3 (Meteor Lake)

| | NPU 3 (Meteor Lake) | NPU 4 (Lunar Lake) | NPU 5 (Panther Lake) |
|---|---|---|---|
| Peak INT8 TOPS | ~11 to 11.5 ([The Register](https://www.theregister.com/2024/06/04/intel_details_lunar_lake_tops/)) | up to 48 ([TechPowerUp](https://techpowerup.com/review/intel-lunar-lake-technical-deep-dive/6.html)) | up to 50 in the family; **46 on the 322** ([ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html)) |
| Neural Compute Engines | 2 ([TechPowerUp](https://techpowerup.com/review/intel-lunar-lake-technical-deep-dive/6.html)) | 6 ([TechPowerUp](https://techpowerup.com/review/intel-lunar-lake-technical-deep-dive/6.html)) | 3, each with a roughly doubled MAC array ([Jon Peddie Research](https://www.jonpeddie.com/techwatch/intel-npu5-in-panther-lake/), [Overclock3D](https://overclock3d.net/reviews/cpu_mainboard/how-intel-supercharged-mobile-and-gaming-with-panther-lake/5/)) |
| New data types | n/a | n/a | Native FP8 ([Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)) |
| Area | n/a | baseline | ~40% smaller than the Lunar Lake NPU ([Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)). Intel frames it as ~40% better TOPS/area (vendor claim). |

- Reporting also says activation functions became programmable and channel padding was removed ([TechBang](https://www.techbang.com/posts/125968-intel-panther-lake-npu), translated, secondary). Exact per-clock MAC counts are reported only by secondary sources **(unverified against an Intel document)**.
- Practical note: OpenVINO on the test unit reports NPU capabilities `FP16, INT8` only. FP8 hardware support is not exposed as an OpenVINO NPU capability there (local observation).

### IPU and memory
- IPU 7.5 sits on the compute tile ([Chips and Cheese](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/), [ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html)).
- Family memory support reaches LPDDR5X-9600 (Arc H parts), DDR5-7200 (some H parts) and LPCAMM2. The 322 tops out at LPDDR5X-7467 / DDR5-6400 ([Intel QRG](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf)).
- **This laptop runs DDR5-5600 SO-DIMMs** ([PSREF](https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_5_2_in_1_15IPH11/IdeaPad_5_2_in_1_15IPH11_Spec.pdf)). That is below the SoC's DDR5-6400 ceiling, which matters for memory-bound GPU and NPU LLM decode.

---

## 3. Software stack (NPU / GPU)

### OpenVINO
- **Version in use:** 2026.4.1 was published on PyPI on 2026-10-01 ([PyPI](https://pypi.org/project/openvino/#history)). The official [release notes page](https://docs.openvino.ai/releasenotes) documents 2026.4 (2026-09-16) but had no separate 2026.4.1 entry when checked.
- **2026.4 known issue CVS-233822:** possible Windows crashes on Core Ultra Series 3 with certain NPU drivers. The notes recommend **NPU driver 32.0.100.5540 or later** ([release notes](https://docs.openvino.ai/releasenotes)).
- **2026.4 NPU plugin changes:** more MoE models (for example Qwen3-30B-A3B), Gemma4 TTFT improvements, and preview bounded dynamic shapes for vision models ([release notes](https://docs.openvino.ai/releasenotes)).
- **NPU plugin** ([NPU Device docs](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)):
  - Supports F32/F16 and U8 (INT8 or mixed FP16-INT8). "Computation precision for the HW is FP16."
  - Static shapes are the standard path. Bounded dynamic shapes are preview-only, NPU40XX+ only, and validated on vision models.
  - Compiler-in-Plugin is the default from 2026.1, with fallback to Compiler-in-Driver.
  - `ov::cache_dir` caching is supported.
- **LLMs on NPU (OpenVINO GenAI)** ([GenAI on NPU guide](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)):
  - Export with symmetric INT4 or NF4 weights: `--sym --weight-format int4 --group-size -1` (or `128` for models up to ~4-5B), with `--ratio 1.0`.
  - `MAX_PROMPT_LEN` defaults to 1024 and `MIN_RESPONSE_LEN` to 128. `PREFILL_HINT` defaults to `DYNAMIC` since 2025.3.
  - `CACHE_DIR` cuts first-load compile time.
  - NF4 requires Core Ultra Series 2 or later.
- **OpenVINO GenAI repo:** [openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai), "Run Generative AI models with simple C++/Python API", covering CPU, GPU and NPU.
- **Intel-listed frameworks per engine** ([ARK](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html)):
  - NPU: OpenVINO, WindowsML, WebNN, ONNX RT.
  - GPU: those plus DirectML and WebGPU.

### Drivers (Windows)
- **Intel NPU Driver for Windows:** latest 32.0.100.5540 (2026-08-28). It lists Core Ultra Series 3 (Panther Lake) and supports Windows 11 23H2/24H2/25H2 ([Intel download](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)).
- **Intel Arc and Intel Graphics driver:** latest 32.0.101.9175 WHQL (2026-11-04). Platform support covers Panther Lake and "Intel Graphics for Intel Core Ultra Series 3 processors" ([Intel download](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html)). OEM (Lenovo) drivers may lag these.

### Windows ML, DirectML, ONNX Runtime
- **Windows ML** includes the CPU and **DirectML (labeled "legacy")** providers. The Intel **OpenVINO EP** (`OpenVINOExecutionProvider`) downloads on demand on Windows 11 24H2+ ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)).
- Intel's [Windows ML support guide](https://www.intel.com/content/www/us/en/developer/articles/guide/windows-ml-support.html) lists NPU support from Core Ultra Series 1 "or newer". It names no specific driver minimum and recommends the latest drivers.
- **ONNX Runtime OpenVINO EP** supports CPU, GPU and NPU. Its compatibility table pairs ORT 1.24.1 with OpenVINO 2025.4.1, so check pairing before mixing with OpenVINO 2026.4.1 ([ORT docs](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)).

### Apps and runtimes
- **Intel AI Playground** (latest 3.2.0 beta) bundles llama.cpp (Vulkan, GGUF), OpenVINO and ComfyUI. "Studio" mode lists Core Ultra Series 3, and the README does not describe NPU use ([GitHub](https://github.com/intel/AI-Playground)).
- **Ollama** runs on Intel GPUs through Vulkan, enabled by default on Windows. Its GPU docs make no mention of NPUs ([Ollama GPU docs](https://docs.ollama.com/gpu)).
- **llama.cpp:**
  - It now has an upstream **OpenVINO backend** that can target CPU, GPU or NPU (`GGML_OPENVINO_DEVICE`).
  - On NPU it uses static graphs with a 256-token prefill chunk and stateless execution, and it needs an explicit `-c`.
  - It is validated on Core Ultra Series 1/2. Panther Lake is not listed ([OPENVINO.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md)).
  - SYCL and Vulkan backends also run on the Xe3 iGPU (see community data below).
- **Models:** the [Hugging Face OpenVINO org](https://huggingface.co/OpenVINO) has about 470 pre-converted models, including a "VLMs optimized for NPU" collection (`*-int4-cw-ov` channel-wise INT4).
- **Tutorials:** [openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks) is updated for 2026.4.
- **Open Model Zoo** is "in maintenance mode as a source of models" ([GitHub](https://github.com/openvinotoolkit/open_model_zoo)). OpenVINO recommends Hugging Face instead ([legacy features](https://docs.openvino.ai/2025/documentation/legacy-features.html)).

---

## 4. Lenovo IdeaPad 5 2-in-1 15IPH11

All rows below come from the [PSREF spec sheet](https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_5_2_in_1_15IPH11/IdeaPad_5_2_in_1_15IPH11_Spec.pdf) (dated October 28, 2026). The [PSREF product page](https://psref.lenovo.com/product/ideapad_5_2_in_1_15iph11?tab=spec) is the interactive version.

| Item | Spec |
|---|---|
| CPU options | Core Ultra 5 322, Core Ultra 5 325, Core Ultra 7 355 |
| AI PC category | Copilot+ PC (Lenovo's note: 40+ NPU TOPS, 16 GB RAM, 256 GB storage) |
| Memory | 2 × DDR5 SO-DIMM, DDR5-5600, dual-channel, up to 32 GB offered |
| Storage | 2 × M.2 PCIe 4.0 x4 (one 2242, one 2280). Shipped as a single 2242 SSD, 512 GB or 1 TB |
| Display | 15.3" WUXGA IPS touch 60 Hz, or 2.5K OLED touch 165 Hz |
| Battery / adapter | 60 or 84 Wh / 65 W USB-C |
| Ports | 2 × USB-C 5 Gbps (PD + DP 1.2), 2 × USB-A 5 Gbps, HDMI 1.4b, microSD, 3.5 mm |
| Wireless | Wi-Fi 6 or Wi-Fi 7 M.2 card, BT 5.4 |
| Graphics | "Intel Graphics", integrated, shared memory |

- **Machine type:** 83UL ([hardware/lenovo-15iph11.md](../hardware/lenovo-15iph11.md)). A retailer manual also lists the 15IPH11 under machine type 83UL ([Coolblue PDF](https://manuals.coolblue.be/dd/lenovo-ideapad-5-2-in-1-15iph11-83ul0046mh.pdf)).
- **Support / drivers page:** no direct pcsupport.lenovo.com URL for 83UL could be confirmed **(unverified)**. Search [pcsupport.lenovo.com](https://pcsupport.lenovo.com/) for "83UL" or use Lenovo Vantage.
- Notebookcheck covered the [15IPH11 announcement](https://www.notebookcheck.net/Lenovo-releases-new-15-inch-convertible-laptop-with-165-Hz-OLED-and-84-Wh-battery.1294895.0.html).
- The test unit's actual PL1/PL2 (sustained and boost power limits) under Lenovo firmware are **(unverified)**. Measure them locally; do not assume 25 W / 55 W.

---

## 5. Reviews and benchmarks

**Same SKU (Core Ultra 5 322), independent**
- [Notebookcheck: Panther Lake debut, Core Ultra 5 322](https://www.notebookcheck.net/Panther-Lake-debut-Core-Ultra-5-322-reviewed-for-the-first-time.1317785.0.html) tested it in the IdeaPad 5i 2-in-1 14 (14-inch sibling).
  - CPU results: CB R23 multi 7862, CB R23 single 1782. Roughly Core i7-1225U class, behind the Core Ultra 7 258V and Ryzen AI 5 435.
  - It calls the iGPU weak but praises low power draw, with over 15 h of Wi-Fi browsing. No NPU testing.
- [Notebookcheck: Core Ultra 5 322 spec/benchmark page](https://www.notebookcheck.net/Intel-Core-Ultra-5-322-Processor-Benchmarks-and-Specs.1196964.0.html) collects aggregated results.

**Panther Lake family, independent** (mostly 12-Xe H SKUs, so not representative of the 322's 2-Xe GPU)
- [Phoronix: Core Ultra X7 358H on Linux](https://www.phoronix.com/review/intel-core-ultra-x7-358h-linux) found strong CPU efficiency in balanced mode (about 15 W average), with the lead lost in the performance profile.
- [Phoronix: Arc B390 on Linux](https://www.phoronix.com/review/intel-arc-b390-panther-lake-linux/10) measured about 1.23× Radeon 890M performance at about 1.25× the power, so efficiency is roughly even.
- [Phoronix: Linux 7.2 improves Xe3](https://www.phoronix.com/review/linux-72-panther-lake/3) attributes B390 gains mainly to Xe kernel driver changes.
- [HotHardware: Core Ultra X9 388H review](https://hothardware.com/reviews/intel-core-ultra-x9-388h-panther-lake-review) calls it a strong all-rounder with improved efficiency.
- [Tom's Hardware: Asus Zenbook Duo (2026) review](https://www.tomshardware.com/laptops/gaming-laptops/asus-zenbook-duo-2026-review) is its first Panther Lake (X9 388H) laptop review and calls the iGPU impressive.
- [Club386: Core Ultra Series 3 review](https://www.club386.com/intel-core-ultra-series-3-panther-lake-review/) praises battery life and the iGPU.
- [Chips and Cheese: Panther Lake's reveal at ITT 2025](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/) is an architecture analysis of Intel's disclosures, not measurements.

**Community measurements** (not this SKU)
- [llama.cpp discussion #23313](https://github.com/ggml-org/llama.cpp/discussions/23313) reports Qwen2.5-7B Q4_0 on a 358H-class Xe3 iGPU:
  - OpenVINO backend: pp1024 ≈ 2989 t/s, tg128 ≈ 17.5 t/s.
  - Vulkan: ≈ 770 / 14.6 t/s.
  - SYCL: ≈ 305 / 17.6 t/s.
  - Expect far lower prompt processing on 2 Xe cores **(unverified for the 322)**.

**Vendor claims** (Intel, not independently verified)
- Up to 50% faster CPU and over 50% faster graphics vs the previous generation, and up to 180 platform TOPS ([Intel](https://www.intel.com/content/www/us/en/newsroom/news/client-computing/intel-unveils-panther-lake-architecture-first-ai-pc-platform-built-on-18a.html)). These apply to top SKUs.
- Up to 77% faster gaming and up to 27 h battery life (CES 2026, reported by [Engadget](https://www.engadget.com/computing/ces-2026-intel-hopes-its-core-ultra-series-3-chips-are-the-start-of-a-comeback-000155611.html) / [TechSpot](https://www.techspot.com/news/110816-intel-launches-core-ultra-series-3-panther-lake.html)).
- Xe3 gaming uplift claims are covered by [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/intel-doubles-down-on-gaming-with-panther-lake-claims-76-percent-faster-gaming-performance-new-x-series-chips-deliver-up-to-12-xe3-cores).

---

## 6. Linux support (brief)

- **NPU kernel driver:** NPU 5 support landed in the `ivpu` accel driver in **Linux 6.13** ([Phoronix](https://www.phoronix.com/news/Intel-PantherLake-NPU-Linux-613)). The firmware later landed in linux-firmware.git ([Phoronix](https://www.phoronix.com/news/Intel-Panther-Lake-NPU-Firmware)).
- **NPU user space:** [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver/releases) v1.38.0 (Sept 2026) verifies Panther Lake on Ubuntu 24.04 and 26.04 with kernel 7.0.0-31-generic. Panther Lake support first appeared in v1.28.0 ([Phoronix](https://www.phoronix.com/news/Intel-User-Space-NPU-PTL)).
- **Xe3 graphics:**
  - Kernel enablement began in the Linux 6.13 DRM cycle ([Phoronix](https://www.phoronix.com/news/Linux-6.13-DRM-Kernel-Graphics)).
  - Mesa Xe3 code (Iris/ANV) was merged for Mesa 24.3 ([Wccftech](https://wccftech.com/intel-initial-support-next-gen-xe3-celestial-gpus-panther-lake-enablement/)).
  - Newer kernels (7.2) and newer Mesa keep improving performance ([Phoronix](https://www.phoronix.com/review/linux-72-panther-lake/3)).
- **OpenVINO 2026.4** added Ubuntu 26.04 LTS NPU support ([release notes](https://docs.openvino.ai/releasenotes)).

---

## How this relates to this repo

- **Device discovery:** [../scripts/detect_devices.py](../scripts/detect_devices.py) should report `Intel(R) Graphics (iGPU)` (2 Xe3 cores, not Arc) and `Intel(R) AI Boost` (NPU 5, 46 TOPS INT8 peak). Record NPU and GPU driver versions against the 32.0.100.5540 and 32.0.101.9175 baselines above.
- **Benchmarks:** for [../scripts/benchmark_openvino.py](../scripts/benchmark_openvino.py), [../scripts/npu_matrix_sweep.py](../scripts/npu_matrix_sweep.py) and [../scripts/npu_anomaly_demo.py](../scripts/npu_anomaly_demo.py):
  - The NPU computes in FP16 and wants static shapes, so sweep fixed shapes.
  - Treat dynamic shapes as preview.
  - Use `CACHE_DIR` to separate compile time from inference latency.
- **Peak TOPS are vendor ceilings.** 46 NPU + 18 GPU TOPS are INT8 peaks under ideal conditions ([QRG footnote 1](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf)). Report measured throughput, not these numbers.
- **Memory bandwidth matters.** DDR5-5600 (below the SoC's 6400 ceiling) bounds LLM decode on both GPU and NPU, so record DIMM population and speed.
- **Examples:** the [../examples](../examples) directory (in progress) should follow the GenAI-on-NPU export rules: symmetric INT4, channel-wise, explicit `MAX_PROMPT_LEN`.
- See also [compatibility-matrix.md](compatibility-matrix.md), [NPU_EXPERIMENT_SUITE.md](NPU_EXPERIMENT_SUITE.md) and [FIRST_TEST_SESSION.md](FIRST_TEST_SESSION.md).

---

## Links

### Intel official
- [Core Ultra 5 322 – ARK specifications](https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html)
- [Core Ultra Series 3 Quick Reference Guide (PDF)](https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors%20-%20Quick%20Reference%20Guide%20v1.pdf)
- [Core Ultra Series 3 product brief](https://www.intel.com/content/www/us/en/content-details/871378/content-details.html)
- [Panther Lake architecture press release](https://www.intel.com/content/www/us/en/newsroom/news/client-computing/intel-unveils-panther-lake-architecture-first-ai-pc-platform-built-on-18a.html)
- [Intel 18A process](https://www.intel.com/content/www/us/en/foundry/process/18a.html)
- [Core Ultra Series 3 for Edge overview (PDF)](https://cdrdv2-public.intel.com/855291/Intel%C2%AE%20Core%E2%84%A2%20Ultra%20Processors%20Series%203%20for%20Edge%20Overview_2.pdf)
- [CES 2026 press deck (PDF, large)](https://download.intel.com/newsroom/2026/CES2026/Intel-CES2026-Core-Ultra-Series-3-Press-Deck.pdf)

### Software / docs
- [OpenVINO release notes](https://docs.openvino.ai/releasenotes) · [openvino on PyPI](https://pypi.org/project/openvino/#history)
- [OpenVINO NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [OpenVINO GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)
- [ONNX Runtime OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [Windows ML execution providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers) · [Intel Windows ML support](https://www.intel.com/content/www/us/en/developer/articles/guide/windows-ml-support.html)
- [llama.cpp OpenVINO backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md) · [Ollama GPU docs](https://docs.ollama.com/gpu)
- [OpenVINO legacy features (Open Model Zoo)](https://docs.openvino.ai/2025/documentation/legacy-features.html)

### Drivers
- [Intel NPU Driver – Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)
- [Intel Arc & Intel Graphics Driver – Windows](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html)
- [intel/linux-npu-driver releases](https://github.com/intel/linux-npu-driver/releases)

### Lenovo
- [PSREF: IdeaPad 5 2-in-1 15IPH11 (spec PDF)](https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_5_2_in_1_15IPH11/IdeaPad_5_2_in_1_15IPH11_Spec.pdf)
- [PSREF product page](https://psref.lenovo.com/product/ideapad_5_2_in_1_15iph11?tab=spec)
- [Lenovo PC Support](https://pcsupport.lenovo.com/) (search "83UL")

### Reviews
- [Notebookcheck – Core Ultra 5 322 first review](https://www.notebookcheck.net/Panther-Lake-debut-Core-Ultra-5-322-reviewed-for-the-first-time.1317785.0.html)
- [Notebookcheck – Core Ultra 5 322 specs/benchmarks](https://www.notebookcheck.net/Intel-Core-Ultra-5-322-Processor-Benchmarks-and-Specs.1196964.0.html)
- [Phoronix – X7 358H Linux CPU review](https://www.phoronix.com/review/intel-core-ultra-x7-358h-linux) · [Phoronix – Arc B390 on Linux](https://www.phoronix.com/review/intel-arc-b390-panther-lake-linux/10)
- [Chips and Cheese – Panther Lake at ITT 2025](https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/)
- [HotHardware – X9 388H review](https://hothardware.com/reviews/intel-core-ultra-x9-388h-panther-lake-review) · [Tom's Hardware – Zenbook Duo 2026](https://www.tomshardware.com/laptops/gaming-laptops/asus-zenbook-duo-2026-review) · [Club386 – Series 3 review](https://www.club386.com/intel-core-ultra-series-3-panther-lake-review/)
- [TechPowerUp – Lunar Lake NPU 4 deep dive](https://techpowerup.com/review/intel-lunar-lake-technical-deep-dive/6.html) · [Jon Peddie – NPU 5](https://www.jonpeddie.com/techwatch/intel-npu5-in-panther-lake/)

### Community / GitHub
- [openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai)
- [openvinotoolkit/openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
- [openvinotoolkit/open_model_zoo (maintenance mode)](https://github.com/openvinotoolkit/open_model_zoo)
- [Hugging Face – OpenVINO org](https://huggingface.co/OpenVINO)
- [intel/AI-Playground](https://github.com/intel/AI-Playground)
- [llama.cpp discussion #23313 (Intel GPU backends incl. Panther Lake)](https://github.com/ggml-org/llama.cpp/discussions/23313)
