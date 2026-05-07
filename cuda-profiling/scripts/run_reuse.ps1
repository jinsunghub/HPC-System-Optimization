$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\vector_reuse_benchmark.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "vector_reuse_$Timestamp.csv"

if (-not (Test-Path $Exe)) {
    throw "Reuse benchmark executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $Exe --repeat 3 --sizes 100000,1000000,10000000 --ops 1,10,100 | Tee-Object -FilePath $ResultFile

Write-Host ""
Write-Host "Saved results to $ResultFile"

