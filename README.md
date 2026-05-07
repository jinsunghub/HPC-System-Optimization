# HPC-System-Optimization

Hands-on experiments for understanding CPU/GPU performance bottlenecks in AI and HPC workloads.

This repository focuses on small, explainable experiments rather than high-level library benchmarking. The main goal is to study how memory hierarchy, data movement, thread scheduling, and vectorization affect the performance of common AI/HPC kernels such as GEMM, convolution, softmax, and reductions.

## Project Overview

| File / Directory | Topic | Main Ideas |
|---|---|---|
| `GEMM_Convolution_Optimization.ipynb` | CUDA GEMM and convolution | Shared memory tiling, coalesced global memory access, constant memory |
| `Softmax_Regression_Dynamic_Scheduling.ipynb` | OpenMP scheduling | Static vs dynamic scheduling, load imbalance, reduction |
| `Matrix_Multiplication_AVX.ipynb` | CPU matrix multiplication | Cache locality, B-matrix transpose, AVX2/FMA vectorization |
| `BigInt_Multiplication_Multithread.ipynb` | CPU multithreading | Independent work partitioning and thread-level parallelism |
| `cuda-profiling/` | CUDA profiling experiments | Transfer overhead, pinned memory, memory coalescing, warp divergence, reduction, stream overlap |

## 1. CUDA GEMM and Convolution Optimization

`GEMM_Convolution_Optimization.ipynb` studies GPU memory hierarchy using GEMM and 2D convolution.

### GEMM

The naive GEMM kernel assigns one output element to each thread:

```text
C[row][col] = sum_k A[row][k] * B[k][col]
```

This causes repeated global-memory reads of the same A and B elements across many threads. To reduce global-memory traffic, the optimized version uses shared memory tiling: each thread block loads tiles of A and B into on-chip shared memory and reuses them across threads.

Key concepts:

- Global memory bottleneck
- Shared memory tiling
- Arithmetic intensity
- Coalesced global-memory access

### Convolution

The convolution experiment compares a naive global-memory implementation with an optimized version that uses shared memory and constant memory.

- Shared memory is used to load an input tile, including halo regions needed by the convolution radius.
- Constant memory is used for convolution filter weights because the weights are read-only and repeatedly accessed by many threads.
- CUDA events are used to measure kernel execution time.
- CPU reference code is used to verify correctness.

Example observation from the notebook:

- Larger filters benefit more from shared-memory reuse.
- Very small filters can be slower after optimization because shared-memory loading and synchronization overhead may outweigh reuse benefits.

## 2. OpenMP Scheduling for Softmax Regression

`Softmax_Regression_Dynamic_Scheduling.ipynb` compares OpenMP scheduling policies for an ML inference-style workload.

Static scheduling has low scheduling overhead, but it can suffer from load imbalance when loop iterations have uneven work. Dynamic scheduling introduces more scheduling overhead, but it can reduce idle time because threads request new chunks at runtime.

The notebook also uses reduction to safely compute max/sum values in parallel without race conditions.

Key concepts:

- Static vs dynamic scheduling
- Load imbalance
- Race condition
- Reduction

## 3. AVX Matrix Multiplication

`Matrix_Multiplication_AVX.ipynb` optimizes CPU matrix multiplication for a CPU fallback or edge inference scenario.

The optimization first transposes matrix B to make memory accesses more contiguous during dot products. Then it uses AVX2/FMA intrinsics to vectorize the multiply-add loop. A `__m256` register can process eight single-precision floating-point values at once.

Key concepts:

- Cache locality
- Data layout transformation
- AVX2 intrinsics
- FMA vectorization

## 4. CUDA Profiling

`cuda-profiling/` is a local CUDA benchmark suite that extends the notebook experiments with more systematic measurements.

Included benchmarks:

- CPU vs GPU vector addition
- GPU data reuse vs repeated host-device round trips
- Pinned memory transfer
- Naive and tiled matrix multiplication
- Block-size and tile-size sweeps
- Global-memory access pattern sweep
- Warp divergence
- Parallel reduction
- CUDA stream overlap
- CUDA Graphs
- Shared-memory bank conflict
- Register pressure

Representative reports are in `cuda-profiling/docs/`, and selected CSV/PNG outputs are in `cuda-profiling/results/`.

## Key Takeaway

These experiments show that performance is not determined only by FLOPs. In many AI/HPC workloads, the main bottlenecks come from memory hierarchy, data movement, synchronization, scheduling, and backend execution strategy.

In particular:

- GPU optimization requires reducing global-memory traffic and improving memory access patterns.
- CPU optimization requires cache-friendly data layout and SIMD vectorization.
- Parallel speedup can be limited by load imbalance, synchronization, and memory bandwidth.
- End-to-end performance should include data transfer and runtime overhead, not only kernel execution time.

## Environment

Experiments were mainly run in Google Colab GPU environments and a local Windows CUDA environment.

Representative hardware used:

- NVIDIA Tesla T4 in Colab
- NVIDIA GeForce MX450 locally
- Intel x86 CPU with AVX2 support

## Repository Status

This repository is an educational and experimental codebase. The focus is on making each optimization idea easy to inspect, reproduce, and explain.
