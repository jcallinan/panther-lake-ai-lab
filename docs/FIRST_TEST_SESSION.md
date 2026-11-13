# First Laptop Test Session — Baseline Protocol

**Target:** Lenovo IdeaPad 5 2-in-1 15IPH11 / Intel Core Ultra 5 322  
**Session:** First on-device testing, planned November 8, 2026  
**Status:** Not yet executed. Do not treat checklist items as completed until measured.

## Tonight's objectives

1. Record Windows version, BIOS, CPU, GPU, RAM topology, SSD and installed driver versions **before changing software or hardware**.
2. Establish whether OpenVINO detects CPU, GPU and NPU.
3. Run the CPU OpenVINO smoke test; attempt GPU/NPU only when detected.
4. Save outputs locally and review for serial numbers or other sensitive identifiers before publishing.
5. Report any failures with exact messages, rather than substituting estimates.

## Preparation

- Plug in the original AC adapter and use a stable power mode.
- Record ambient conditions and whether the laptop is on a desk or cooling stand.
- Close games, heavy downloads, updates and unrelated background workloads.
- Note Windows Update status and whether a reboot is pending.
- Do not change BIOS, drivers, RAM or SSD until the initial inventory is captured.

## Commands (PowerShell)

Prerequisites: Git and a supported Python 3 installation with the `py` launcher.

```powershell
git clone https://github.com/jcallinan/panther-lake-ai-lab.git
cd panther-lake-ai-lab

# Hardware baseline; output is excluded from Git by default
powershell -ExecutionPolicy Bypass -File .\hardware\inventory.ps1

# Record Windows version
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber

# Install test environment
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Detect actual OpenVINO devices
.\.venv\Scripts\python.exe scripts\detect_devices.py

# Run CPU baseline
.\.venv\Scripts\python.exe scripts\benchmark_openvino.py --device CPU

# Only if GPU/NPU appear in device detection:
.\.venv\Scripts\python.exe scripts\benchmark_openvino.py --device GPU
.\.venv\Scripts\python.exe scripts\benchmark_openvino.py --device NPU
```

**Important:** The initial benchmark is a small matrix operation for compatibility/smoke testing. It is not an LLM throughput benchmark. The GPU or NPU may detect but fail compilation or execution. Capture this as a compatibility result.

## Session record template

| Field | Value |
|---|---|
| Test date/time | Pending |
| Windows build | Pending |
| BIOS version | Pending |
| CPU model and topology | Pending |
| GPU driver version | 32.0.101.8860 (2026-06-24) baseline; see [DRIVER_SETUP.md](DRIVER_SETUP.md) |
| NPU device and driver | Intel(R) AI Boost, 32.0.100.4724 (2026-03-18) baseline; 5540+ recommended |
| RAM modules and channel configuration | Pending |
| SSD model | Pending |
| OpenVINO version | Pending |
| Detected OpenVINO devices | Pending |
| CPU smoke result | Pending |
| GPU smoke result | Pending |
| NPU smoke result | Pending |
| Power mode / AC | Pending |
| Ambient conditions | Pending |
| Errors and notes | Pending |

## What to share back

Paste the sanitized JSON from `benchmarks/results/hardware.json`, output from `scripts/detect_devices.py`, and the benchmark output/error messages. Avoid sharing serial numbers, UUIDs, Windows product keys, private paths or other unique identifiers.

## First-session acceptance gate

- [ ] Hardware inventory saved and reviewed
- [ ] OS/BIOS and drivers recorded
- [ ] Python environment installed
- [ ] OpenVINO device detection captured
- [ ] CPU smoke benchmark completed or documented failure
- [ ] GPU and NPU attempts recorded where applicable
- [ ] Results added to an issue or committed in sanitized form

Related: [Issue #2](https://github.com/jcallinan/panther-lake-ai-lab/issues/2), [Issue #3](https://github.com/jcallinan/panther-lake-ai-lab/issues/3), [Paper](PAPER_DRAFT.md), [Conference roadmap](CONFERENCE_ROADMAP.md).
