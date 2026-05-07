$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildDir = Join-Path $ProjectRoot "build"

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc was not found. Install the CUDA Toolkit and make sure nvcc is on PATH."
}

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $VsWhere) {
        $VsPath = & $VsWhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
        $VsDevCmd = Join-Path $VsPath "Common7\Tools\VsDevCmd.bat"
        if (Test-Path $VsDevCmd) {
            Write-Host "MSVC is installed but not active. Re-running build inside VsDevCmd..."
            $BuildScript = $MyInvocation.MyCommand.Path
            & cmd.exe /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 && powershell -ExecutionPolicy Bypass -File `"$BuildScript`""
            exit $LASTEXITCODE
        }
    }
    throw "MSVC cl.exe was not found. Run from 'x64 Native Tools Command Prompt for VS' or install Visual Studio Build Tools."
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$CudaSrcDir = Join-Path $ProjectRoot "src\cuda"

$Targets = @(
    "vector_add_benchmark",
    "vector_reuse_benchmark",
    "pinned_memory_benchmark",
    "matmul_naive_benchmark",
    "matmul_tiled_benchmark",
    "vector_blocksize_sweep",
    "matmul_tile_sweep",
    "memory_access_pattern_benchmark",
    "warp_divergence_benchmark",
    "reduction_benchmark",
    "stream_overlap_benchmark",
    "cuda_graphs_benchmark",
    "shared_bank_conflict_benchmark",
    "register_pressure_benchmark"
)

foreach ($Name in $Targets) {
    $Source = Join-Path $CudaSrcDir "$Name.cu"
    $Output = Join-Path $BuildDir "$Name.exe"

    nvcc `
        -std=c++17 `
        -O2 `
        -lineinfo `
        -Xcompiler "/W3 /wd4819" `
        -o $Output `
        $Source

    Write-Host "Built $Output"
}
