# Regenerate everything derived from benchmarks/results/*.json: charts, GitHub Pages site, both slide decks.
# Optionally re-record the offload dashboard first (needs an otherwise idle machine on AC power).
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\refresh_outputs.ps1 [-RecordOffload]
param([switch]$RecordOffload)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'

function Step($name, [scriptblock]$body) {
  Write-Host "`n== $name ==" -ForegroundColor Cyan
  $global:LASTEXITCODE = 0
  & $body
  if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

if ($RecordOffload) {
  Step 'offload_dashboard (record)' {
    powershell -ExecutionPolicy Bypass -File (Join-Path $root 'scripts\record_all.ps1') `
      -Skip inventory,devices,matrix_sweep,anomaly,perf_atlas,vision_side_by_side,depth_parallax,llm_chat
  }
}
Step 'charts'       { & $py scripts\make_charts.py }
Step 'site'         { & $py site\build.py }
Step 'tech deck'    { & $py docs\presentation\build_tech_deck.py }
Step 'board deck'   { & $py docs\presentation\build_board_deck.py }
Write-Host "`nDone. Review media/charts/, site/_build/ and docs/presentation/*.pptx; commit site/data/results.json with the media."
