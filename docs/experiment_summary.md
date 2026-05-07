# Experiment Summary

This repository contains hands-on experiments for understanding CPU/GPU bottlenecks in AI and HPC workloads.

## Motivation

AI inference and training kernels are often limited not only by arithmetic throughput, but also by memory movement, memory bandwidth, launch overhead, synchronization, and scheduling. These experiments were designed to expose those bottlenecks with small, directly inspectable programs.

## GPU Experiments

### GEMM and Convolution

The notebook `GEMM_Convolution_Optimization.ipynb` studies how global-memory traffic affects GEMM and convolution.

- Naive GEMM repeatedly reads the same A/B elements from global memory.
- Shared memory tiling improves data reuse within a thread block.
- Convolution reuses overlapping input regions, so input tiles and halo regions can be loaded into shared memory.
- Filter weights are good candidates for constant memory because they are read-only and repeatedly accessed by many threads.

### CUDA Profiling

The `cuda-profiling/` directory extends the notebook experiments with individual profiling benchmarks.

Key experiments:

- Vector add: separates H2D copy, kernel time, D2H copy, and total GPU time.
- GPU reuse: shows the cost of repeated CPU-GPU round trips.
- Pinned memory: compares pageable and page-locked host memory transfer.
- Matrix multiplication: compares naive and shared-memory tiled kernels.
- Memory coalescing: shows how strided global-memory access reduces bandwidth.
- Warp divergence: shows how branch divergence hurts SIMT efficiency.
- Reduction: compares atomic, shared-memory, and warp-level reductions.
- Stream overlap: overlaps H2D copy, kernel execution, and D2H copy across CUDA streams.

Representative result from the stream overlap experiment:

| Method | Total time |
|---|---:|
| Pageable sequential | 53.76 ms |
| Pinned sequential | 49.12 ms |
| Pinned + 2 streams | 29.70 ms |
| Pinned + 4 streams | 27.64 ms |

This showed that pinned memory plus multi-stream execution can reduce end-to-end time by overlapping data transfer and computation.

## CPU Experiments

### OpenMP Scheduling

`Softmax_Regression_Dynamic_Scheduling.ipynb` studies static vs dynamic scheduling.

- Static scheduling has low overhead but can suffer from load imbalance.
- Dynamic scheduling adds scheduling overhead but can reduce idle time for uneven workloads.
- Reduction is needed for safe parallel max/sum operations.

### AVX Matrix Multiplication

`Matrix_Multiplication_AVX.ipynb` studies CPU-side matrix multiplication optimization.

- Transposing B improves cache locality during dot products.
- AVX2/FMA vectorizes multiply-add operations.
- A `__m256` register can process eight `float` values at once.

## Overall Takeaway

The main takeaway is that AI/HPC performance depends heavily on how data moves through the system. FLOP count alone is not enough to explain performance. Memory hierarchy, data layout, scheduling, synchronization, and hardware execution behavior must be considered together.
