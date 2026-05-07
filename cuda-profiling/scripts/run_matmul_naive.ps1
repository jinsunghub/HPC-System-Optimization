$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\matmul_naive_benchmark.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "matmul_naive_$Timestamp.csv"

if (-not (Test-Path $Exe)) {
    throw "Naive matmul benchmark executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $Exe --repeat 2 --sizes 128,256,512,1024 --block-size 16 | Tee-Object -FilePath $ResultFile

Write-Host ""
Write-Host "Saved results to $ResultFile"

