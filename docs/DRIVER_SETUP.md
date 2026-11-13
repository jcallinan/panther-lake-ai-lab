# Driver setup (Windows)

The NPU and integrated GPU depend on Intel drivers. The OpenVINO version in `requirements.txt` needs current ones. Record the driver versions with every published result, because driver updates can change both performance and correctness.

## Recommended minimum versions (as of 2026-11-11)

| Component | Device Manager name | Minimum | Download |
|---|---|---|---|
| NPU (NPU 5, "Intel AI Boost") | Neural processors → Intel(R) AI Boost | **32.0.100.5540** | [Intel NPU Driver – Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) |
| Integrated GPU (2 Xe3 cores, "Intel Graphics") | Display adapters → Intel(R) Graphics | **32.0.101.9175** | [Intel Arc & Intel Graphics Driver – Windows](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html) |

Why the NPU minimum matters: the [OpenVINO 2026.4 release notes](https://docs.openvino.ai/releasenotes) list a known issue, CVS-233822: possible Windows crashes on Core Ultra Series 3 with older NPU drivers. They recommend NPU driver 32.0.100.5540 or later.

The graphics download page lists "Intel Graphics for Intel Core Ultra Series 3 processors". The Core Ultra 5 322's GPU is not branded Arc, but this driver still applies.

## Check what is installed

```powershell
Get-CimInstance Win32_PnPSignedDriver |
  Where-Object DeviceName -match '^Intel\(R\) (NPU|AI Boost|Graphics)$' |
  Select-Object DeviceName, DriverVersion, DriverDate
```

`hardware/inventory.ps1` also records these versions in `benchmarks/results/hardware.json` (field `AcceleratorDrivers`).

## Install / update

**Close anything using OpenVINO first.** Driver installs reset the device, and running NPU or GPU workloads will crash.

1. **NPU driver:** download the zip from the NPU link above and extract it. If the package includes an installer or a README with other steps, follow those. Otherwise:
   - open **Device Manager → Neural processors → Intel(R) AI Boost**
   - choose **Update driver → Browse my computer for drivers**
   - select the extracted folder.
2. **Graphics driver:** run the installer from the graphics link above.
3. **Reboot.**
4. Verify everything:
   ```powershell
   .\.venv\Scripts\python.exe scripts\detect_devices.py   # NPU and GPU must still be listed
   powershell -ExecutionPolicy Bypass -File .\hardware\inventory.ps1
   ```

### Alternatives

- **[Intel Driver & Support Assistant](https://www.intel.com/content/www/us/en/support/detect.html):** detects and installs both drivers automatically.
- **Lenovo OEM drivers:** via Lenovo Vantage, or [pcsupport.lenovo.com](https://pcsupport.lenovo.com/) (search for machine type "83UL"). These are validated by the OEM, but they usually lag behind Intel's releases. Note which source you used.

## Driver history for the test laptop

Add a row every time a driver changes. Re-run the benchmarks and examples after each change, and keep results from different drivers separate.

| Date | NPU driver | Graphics driver | Source | Notes |
|---|---|---|---|---|
| 2026-11-11 | 32.0.100.4724 (2026-03-18) | 32.0.101.8860 (2026-06-24) | Factory / Windows Update | Baseline. All three OpenVINO devices execute. The Perf Atlas found NPU conv stacks at **batch 2–8 return sample 0's output for every batch element**; batch 1 and ≥ 12 are correct. Qwen3-0.6B (INT4_ASYM) hangs while compiling for the NPU. |
| 2026-11-12 | 32.0.100.5540 (2026-08-19) | 32.0.101.9175 (2026-11-03) | Intel (graphics: driver + Graphics Software only; ICIP telemetry and Level Zero SDK not installed) | Compile caches from the old driver were moved aside (`models/*cache_drv4724`). All examples re-recorded sequentially with `scripts/record_all.ps1`. The old driver's results are archived in `benchmarks/results/archive/`. **The batch 2–8 conv bug persists**: the same 9 NPU configs give `failed_correctness`. NPU MLPs at batch 8 are still about 7× slower than at batch 32 (MLP-L 24.6 ms vs 3.5 ms). With an idle machine, CPU and GPU are much faster than in the preliminary run, which had other jobs running (for example CONV-M b32 on CPU: 415 → 138 ms). GPU LLM cold load dropped from 228 s to 9.6 s. |

See [PANTHER_LAKE_REFERENCE.md](PANTHER_LAKE_REFERENCE.md) for hardware details and more links.
