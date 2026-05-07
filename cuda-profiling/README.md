# CUDA Profiling

Small CUDA benchmarks for learning where GPU programs spend time.

This project focuses on profiling basic CUDA bottlenecks that appear in AI/HPC workloads: host-device data transfer, memory reuse, shared memory, memory access patterns, branch divergence, reductions, and stream overlap.

## Benchmarks

| Benchmark | File | What it studies |
|---|---|---|
| Vector add | `src/cuda/vector_add_benchmark.cu` | CPU vs GPU time, H2D/D2H transfer, kernel time |
| GPU reuse | `src/cuda/vector_reuse_benchmark.cu` | Keeping data on GPU vs repeated round trips |
| Pinned memory | `src/cuda/pinned_memory_benchmark.cu` | Pageable vs pinned host memory transfer |
| Naive matmul | `src/cuda/matmul_naive_benchmark.cu` | Basic CUDA GEMM baseline |
| Tiled matmul | `src/cuda/matmul_tiled_benchmark.cu` | Shared memory tiling |
| Block/tile sweep | `src/cuda/vector_blocksize_sweep.cu`, `src/cuda/matmul_tile_sweep.cu` | Launch configuration sensitivity |
| Memory access | `src/cuda/memory_access_pattern_benchmark.cu` | Coalesced vs strided global-memory access |
| Warp divergence | `src/cuda/warp_divergence_benchmark.cu` | SIMT branch divergence |
| Reduction | `src/cuda/reduction_benchmark.cu` | Atomic, shared-memory, and warp-level reduction |
| Stream overlap | `src/cuda/stream_overlap_benchmark.cu` | Pinned memory plus multi-stream copy/compute overlap |
| CUDA Graphs | `src/cuda/cuda_graphs_benchmark.cu` | Kernel launch overhead reduction |
| Shared bank conflict | `src/cuda/shared_bank_conflict_benchmark.cu` | Shared-memory bank conflicts |
| Register pressure | `src/cuda/register_pressure_benchmark.cu` | Register usage and occupancy pressure |

## How to Build

Requirements:

- NVIDIA GPU and driver
- CUDA Toolkit with `nvcc`
- Visual Studio Build Tools with MSVC `cl.exe` on Windows

Check the environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_env.ps1
```

Build all CUDA benchmarks:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

Run the first benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

## Representative Results

Selected reports:

- `docs/first_results.md`
- `docs/reuse_results.md`
- `docs/pinned_memory_results.md`
- `docs/matmul_tiled_results.md`
- `docs/memory_access_results.md`
- `docs/warp_divergence_results.md`
- `docs/reduction_results.md`
- `docs/stream_overlap_results.md`

Selected CSV outputs are stored under:

- `results/selected_csv/`

## What I Learned

The main lesson is that GPU performance is not only about writing a parallel kernel. End-to-end performance is shaped by data movement, memory hierarchy, synchronization, launch overhead, and whether independent work can be overlapped.
