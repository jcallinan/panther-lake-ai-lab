$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$info = [ordered]@{
  Timestamp = (Get-Date).ToString('o')
  Windows = @(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber)
  CPU = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
  Board = @(Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product,Version)
  RAM = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer,PartNumber,Capacity,Speed,DeviceLocator)
  Disks = @(Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,InterfaceType)
  GPU = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
  # NPU and Intel graphics driver versions (see docs/DRIVER_SETUP.md for recommended minimums).
  AcceleratorDrivers = @(Get-CimInstance Win32_PnPSignedDriver |
    Where-Object { $_.DeviceName -match '^Intel\(R\) (NPU|AI Boost|Graphics)$' } |
    Select-Object DeviceName,DriverVersion,@{n='DriverDate';e={$_.DriverDate.ToString('yyyy-MM-dd')}})
  BIOS = @(Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion,ReleaseDate)
  PowerSource = if ((Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue).BatteryStatus -eq 2) { 'AC' } else { 'Battery or unknown' }
}
$out = Join-Path $repoRoot 'benchmarks/results'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$info | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $out 'hardware.json')
Write-Host "Review $out\hardware.json before sharing."
