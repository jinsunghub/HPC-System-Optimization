# Windows CUDA Setup

This note describes the recommended Windows setup for building the CUDA benchmarks in this directory.

## Requirements

- NVIDIA GPU driver
- CUDA Toolkit with `nvcc`
- Visual Studio Build Tools with the C++ workload
- Optional: NVIDIA Nsight Systems for timeline profiling

## Install Order

1. Install Visual Studio Build Tools.
   - Choose "Desktop development with C++".
   - Make sure MSVC and the Windows SDK are selected.

2. Install the CUDA Toolkit.
   - Choose a CUDA Toolkit version supported by your NVIDIA driver.
   - Reopen the terminal after installation so that PATH is updated.

3. Install Nsight Systems.
   - This is optional for compiling.
   - It is useful for CUDA timeline profiling.

## Open the Right Terminal

On Windows, `nvcc` needs MSVC in the environment. Use one of these:

- "x64 Native Tools Command Prompt for VS"
- Developer PowerShell for Visual Studio

Then move to the project directory:

```powershell
cd path\to\HPC-System-Optimization\cuda-profiling
```

Check the environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_env.ps1
```

Build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

## First Interpretation Target

After the first run, do not only ask whether the GPU is faster. Ask:

- How much time is host-to-device copy?
- How much time is kernel execution?
- How much time is device-to-host copy?
- At what input size does kernel speedup appear?
- At what input size does total GPU speedup appear?
- Why are those two sizes different?
