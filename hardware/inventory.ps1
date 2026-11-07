$ErrorActionPreference = 'Stop'
$info = [ordered]@{
  Timestamp = (Get-Date).ToString('o')
  CPU = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
  Board = @(Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product,Version)
  RAM = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer,PartNumber,Capacity,Speed,DeviceLocator)
  Disks = @(Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,InterfaceType)
  GPU = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
  BIOS = @(Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion,ReleaseDate)
}
New-Item -ItemType Directory -Force -Path benchmarks/results | Out-Null
$info | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 benchmarks/results/hardware.json
Write-Host 'Review benchmarks/results/hardware.json before sharing.'
