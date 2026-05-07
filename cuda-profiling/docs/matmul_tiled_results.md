# Lab 5 Results: Tiled Matrix Multiplication

## Environment

- Date: 2026-05-04
- GPU: NVIDIA GeForce MX450
- GPU memory: 2048 MiB
- Driver: 595.97
- CUDA compiler: 13.2, V13.2.78
- Build: local CUDA redist + Visual Studio 2019 MSVC

## Experiment

This lab compares two CUDA matrix multiplication kernels:

- naive: each thread computes one `C[row, col]` and reads `A` and `B` directly from global memory
- tiled: each block loads `16 x 16` tiles of `A` and `B` into shared memory, then reuses those values inside the block

Both kernels use the same input matrices and the same `16 x 16` thread block
shape. The CPU naive result is included as a baseline.

## Command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_matmul_tiled.ps1
```

## Raw Results

```csv
n,bytes_per_matrix,cpu_ms,cpu_gflops,h2d_ms,naive_kernel_ms,naive_kernel_gflops,tiled_kernel_ms,tiled_kernel_gflops,naive_total_ms,tiled_total_ms,tiled_speedup_vs_naive_kernel,tiled_speedup_vs_naive_total,naive_speedup_vs_cpu_total,tiled_speedup_vs_cpu_total,naive_max_abs_error,tiled_max_abs_error
256,262144,17.900950,1.874450,0.291248,0.317776,105.591462,0.155136,216.290423,0.805312,0.593520,2.048370,1.356840,22.228589,30.160650,0.000019,0.000019
512,1048576,129.396850,2.074513,0.941232,1.491488,179.978290,0.901952,297.616112,3.387296,2.705648,1.653622,1.251935,38.200633,47.824717,0.000031,0.000031
1024,4194304,7765.451150,0.276543,7.396864,57.996529,37.027796,29.161440,73.641208,71.879857,42.531024,1.988809,1.690057,108.033760,182.583216,0.000107,0.000107
```

## Key Observations

The tiled kernel was faster than the naive kernel for every tested matrix size.

For `1024 x 1024`:

- naive kernel: 58.00 ms, 37.03 GFLOPS
- tiled kernel: 29.16 ms, 73.64 GFLOPS
- tiled kernel speedup: 1.99x

Including one H2D copy and one D2H copy:

- naive total: 71.88 ms
- tiled total: 42.53 ms
- tiled total speedup: 1.69x

The correctness error stayed small for both kernels:

- naive max abs error: 0.000107
- tiled max abs error: 0.000107

## Interpretation

Naive matmul repeatedly reads the same values from global memory. For example,
many threads in a block need nearby values from the same row of `A` and the same
column region of `B`.

The tiled version loads a small block of `A` and `B` into shared memory. Once the
tile is loaded, the threads inside the block reuse those values. This reduces
repeated global memory traffic.

The key CUDA lesson:

```text
Shared memory helps when many threads reuse the same data.
```

This is the first real GPU memory hierarchy optimization in the lab. The earlier
experiments focused on host-device transfer. This one focuses on memory traffic
inside the GPU.

## Caveat

This is still a learning kernel. It does not use register tiling, vectorized
loads, loop unrolling, Tensor Cores, or cuBLAS. A production GEMM implementation
would be much faster.

The important result is not that this kernel is optimal. The important result is
that shared-memory data reuse improved the naive CUDA kernel by about 2x on the
largest tested matrix.

## Next Experiments

- Profile naive vs tiled with Nsight Systems.
- Add `cudaMemcpyAsync` with pinned memory to overlap transfer and compute.
- Implement reduction to study parallel reduction patterns.
- Later, compare against cuBLAS.

