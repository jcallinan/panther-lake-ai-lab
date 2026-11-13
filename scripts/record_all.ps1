# Re-run every benchmark and re-record every showcase GIF/MP4, strictly one at a time.
#
# Run from anywhere, on AC power, with no other OpenVINO workloads running:
#   powershell -ExecutionPolicy Bypass -File .\scripts\record_all.ps1 [-Skip llm_chat,perf_atlas]
#
# Before running, existing results are archived to benchmarks/results/archive/<timestamp>_superseded-by-run-on-npu<driver>/
# (folder named for the driver of the run that SUPERSEDES them) so runs on different drivers stay separate (see docs/DRIVER_SETUP.md).
param([string[]]$Skip = @())
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$results = Join-Path $root 'benchmarks\results'

# --- Environment snapshot -------------------------------------------------------------------
$npu = (Get-CimInstance Win32_PnPSignedDriver | Where-Object DeviceName -match '^Intel\(R\) (NPU|AI Boost)$' | Select-Object -First 1).DriverVersion
$gpu = (Get-CimInstance Win32_PnPSignedDriver | Where-Object DeviceName -eq 'Intel(R) Graphics' | Select-Object -First 1).DriverVersion
$onAC = (Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue).BatteryStatus -eq 2
Write-Host "NPU driver $npu | Graphics driver $gpu | AC power: $onAC"
if ([version]$npu -lt [version]'32.0.100.5540') { Write-Warning 'NPU driver is older than 32.0.100.5540 (see docs/DRIVER_SETUP.md).' }
if (-not $onAC) { Write-Warning 'Not on AC power: results are not comparable with AC runs.' }
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -match 'openvino|examples\.|scripts\\'
if ($busy) { Write-Warning "Other Python/OpenVINO processes are running (PIDs $($busy.ProcessId -join ', ')); timings may be perturbed." }

# --- Archive previous results ---------------------------------------------------------------
$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$archive = Join-Path $results "archive\${stamp}_superseded-by-run-on-npu$npu"
$old = Get-ChildItem $results -File -Filter *.json -ErrorAction SilentlyContinue
if ($old) { New-Item -ItemType Directory -Force $archive | Out-Null; $old | Copy-Item -Destination $archive; Write-Host "Archived $($old.Count) result files to $archive" }

# --- Keep the machine awake for the duration (restored automatically when this process exits) --
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$steps = [ordered]@{
  inventory           = { powershell -ExecutionPolicy Bypass -File (Join-Path $root 'hardware\inventory.ps1') }
  devices             = { & $py scripts\detect_devices.py | Set-Content -Encoding UTF8 (Join-Path $results 'devices.json') }
  matrix_sweep        = { & $py scripts\npu_matrix_sweep.py --devices CPU GPU NPU | Set-Content -Encoding UTF8 (Join-Path $results 'npu_matrix_sweep.json') }
  anomaly             = { & $py scripts\npu_anomaly_demo.py --devices CPU GPU NPU --runs 200 | Set-Content -Encoding UTF8 (Join-Path $results 'npu_anomaly_demo.json') }
  perf_atlas          = { & $py -m examples.perf_atlas }
  vision_side_by_side = { & $py -m examples.vision_side_by_side --record --no-display --duration 15 }
  depth_parallax      = { & $py -m examples.depth_parallax --record --no-display }
  offload_dashboard   = { & $py -m examples.offload_dashboard --record --no-display }
  llm_chat            = { & $py -m examples.llm_chat --bench --devices NPU CPU GPU; if ($LASTEXITCODE -eq 0) { & $py -m examples.llm_chat --render } }
}

$log = @()
foreach ($name in $steps.Keys) {
  if ($Skip -contains $name) { Write-Host "-- skip $name"; continue }
  Write-Host "`n== $name ==" -ForegroundColor Cyan
  $t = [Diagnostics.Stopwatch]::StartNew()
  $global:LASTEXITCODE = 0
  try { & $steps[$name]; $ok = ($LASTEXITCODE -eq 0) } catch { Write-Warning $_; $ok = $false }
  $log += [pscustomobject]@{ step = $name; ok = $ok; seconds = [int]$t.Elapsed.TotalSeconds }
  Start-Sleep -Seconds 10   # let clocks/thermals settle between workloads
}
[Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null

$summary = [ordered]@{ finished = (Get-Date).ToString('o'); npu_driver = $npu; graphics_driver = $gpu; on_ac = $onAC; steps = $log }
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $results 'record_all_summary.json')
$log | Format-Table -AutoSize
if ($log | Where-Object { -not $_.ok }) { exit 1 }
