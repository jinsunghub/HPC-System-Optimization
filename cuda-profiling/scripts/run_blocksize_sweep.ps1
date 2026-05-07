$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\vector_blocksize_sweep.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "vector_blocksize_sweep_$Timestamp.csv"

if (-not (Test-Path $Exe)) {
    throw "Vector block-size sweep executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $Exe --size 10000000 --repeat 30 --blocks 64,128,256,512,1024 | Tee-Object -FilePath $ResultFile

Write-Host ""
Write-Host "Saved results to $ResultFile"

