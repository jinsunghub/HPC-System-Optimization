$ErrorActionPreference = "Stop"

function Test-Command($Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "CUDA Profiling environment check"
Write-Host ""

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if (Test-Command "nvidia-smi") {
    Write-Host "[OK] nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} else {
    Write-Host "[MISSING] nvidia-smi not found. Install or update the NVIDIA driver."
}

Write-Host ""

if (Test-Command "nvcc") {
    Write-Host "[OK] nvcc found"
    nvcc --version
} else {
    Write-Host "[MISSING] nvcc not found. Install the CUDA Toolkit and make sure nvcc is on PATH."
}

Write-Host ""

if (Test-Command "cl") {
    Write-Host "[OK] MSVC cl.exe found"
    cl 2>&1 | Select-Object -First 1
} else {
    Write-Host "[MISSING] MSVC cl.exe not found in PATH."
    Write-Host "          On Windows, run this from 'x64 Native Tools Command Prompt for VS'"
    Write-Host "          or install Visual Studio Build Tools with the C++ workload."
}

Write-Host ""

if (Test-Command "nsys") {
    Write-Host "[OK] Nsight Systems CLI found"
    nsys --version
} else {
    Write-Host "[OPTIONAL] nsys not found. Install Nsight Systems if you want timeline profiling."
}
