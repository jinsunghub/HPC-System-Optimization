$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ResultsDir = Join-Path $ProjectRoot "results"
$Exe = Join-Path $ProjectRoot "build\matmul_tiled_benchmark.exe"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputBase = Join-Path $ResultsDir "nsys_matmul_tiled_$Timestamp"
$StatsFile = "$OutputBase.stats.txt"

if (-not (Test-Path $Exe)) {
    throw "Tiled matmul benchmark executable not found. Run scripts\build.ps1 first."
}

$nsys = Get-Command nsys -ErrorAction SilentlyContinue
if ($nsys) {
    $nsysPath = $nsys.Source
} else {
    throw "nsys was not found. Install Nsight Systems and make sure nsys is on PATH."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $nsysPath profile `
    --trace=cuda,nvtx,osrt `
    --sample=none `
    --cpuctxsw=none `
    --force-overwrite=true `
    --output=$OutputBase `
    $Exe --repeat 1 --sizes 1024

$ReportFile = "$OutputBase.nsys-rep"
if (-not (Test-Path $ReportFile)) {
    $candidate = Get-ChildItem -Path $ResultsDir -Filter "$(Split-Path $OutputBase -Leaf)*.nsys-rep" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($candidate) {
        $ReportFile = $candidate.FullName
    }
}

& $nsysPath stats --report cudaapisum,cudakernsum,cudamemtimesum "$ReportFile" | Tee-Object -FilePath $StatsFile

Write-Host ""
Write-Host "Saved Nsight report to $ReportFile"
Write-Host "Saved stats to $StatsFile"
