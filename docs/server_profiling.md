# Server-Side CPU and Memory Profiling

This document summarizes server-side CPU and memory profiling conducted in the CAMe Lab environment. These experiments provide the architectural background for the CUDA, OpenMP, and AVX experiments in this repository.

## Environment

| Server / Environment | CPU / Platform | Used For |
|---|---|---|
| `came02` | T340 E-2288G | Memory latency measurement |
| `came04` | 2 * Intel Xeon Scalable Gold 6530 | NUMA-aware memory latency measurement |
| CAMe Lab server environment | AVX-512 capable server CPU environment | Matrix multiplication profiling and Intel VTune profiling |

## 1. Matrix Multiplication Profiling

The matrix multiplication experiment analyzed how loop order, cache blocking, loop unrolling, and SIMD affect GEMM performance on a server environment.

| Optimization | Main Idea | Result |
|---|---|---:|
| Loop reordering | Improve row-major memory access by changing loop order | About 52% improvement |
| Loop unrolling | Reduce loop-control overhead | About 1.8% improvement |
| Loop blocking | Improve cache reuse with block-wise computation | About 44.5% improvement |
| SIMD / AVX-512 | Process multiple values per instruction | About 6.9x speedup |
| Loop unrolling + loop tiling + SIMD | Combine cache-aware blocking and vectorized execution | About 13.2x speedup |

Key observation: matrix multiplication performance depended heavily on memory access patterns, cache reuse, and whether SIMD was applied together with cache-aware blocking.

## 2. Memory Latency Measurement

Memory latency was measured using Google Multichase. The experiment covered cache-level latency, DRAM latency, stride and buffer-size effects, and NUMA local/remote access behavior.

Measured targets:

- L1 latency
- L2 latency
- LLC/L3 latency
- DRAM latency
- Local memory latency
- Remote memory latency on NUMA systems

For `came02`, the lab note reports the following latency values:

| Level | Latency |
|---|---:|
| L1 | 0.865 |
| L2 | 2.563 |
| LLC | 8.974 |
| DRAM | 48.54 |

`came02` used a single NUMA configuration, so remote memory latency was not measured.

For `came04`, local and remote memory access were compared to observe NUMA effects. The experiment used `numactl` to bind computation and memory placement to specific NUMA nodes.

## 3. Intel VTune Profiling

Intel VTune was used to analyze sample matrix multiplication kernels. The profiling focused on microarchitecture-level bottlenecks rather than only elapsed time.

Main metrics and concepts:

- Retiring
- Front-End Bound
- Back-End Bound
- Memory Bound
- Core Bound
- L1/L2/L3 Bound
- DRAM Bound
- DTLB overhead

The VTune analysis showed that matrix multiplication performance is limited not only by arithmetic throughput, but also by cache locality, memory access pattern, DTLB behavior, and the interaction between vectorization and cache blocking.

## Relationship to Repository Experiments

The server-side profiling work provides the basis for the implementation-level experiments in this repository:

- Server profiling explains why memory hierarchy, cache locality, NUMA behavior, and DTLB overhead matter.
- CUDA experiments apply the same data-movement perspective to GPU global memory, shared memory, pinned memory, and stream overlap.
- OpenMP experiments focus on scheduling, load imbalance, and safe parallel reduction.
- AVX experiments apply cache-friendly data layout and SIMD vectorization at the CPU implementation level.
