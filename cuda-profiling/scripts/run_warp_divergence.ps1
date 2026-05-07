$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\warp_divergence_benchmark.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "warp_divergence_$Timestamp.csv"

if (-not (Test-Path $Exe)) {
    throw "Warp divergence benchmark executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $Exe --size 4194304 --repeat 20 --block-size 256 --iters 128 | Tee-Object -FilePath $ResultFile

Write-Host ""
Write-Host "Saved results to $ResultFile"

