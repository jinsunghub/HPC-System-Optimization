# Lab 4 Results: Naive Matrix Multiplication

## Environment

- GPU: NVIDIA GeForce MX450
- GPU memory: 2048 MiB
- Driver: 595.97
- CUDA compiler: 13.2, V13.2.78
- Build: local CUDA redist + Visual Studio 2019 MSVC

## Experiment

This lab compares naive CPU matrix multiplication with a naive CUDA kernel:

```cpp
C[row, col] = sum(A[row, k] * B[k, col])
```

Each CUDA thread computes one output element of `C`.

This is not an optimized matrix multiplication implementation. It does not use
tiling, shared memory, vectorization, Tensor Cores, or BLAS libraries. The goal
is to compare the shape of a compute-heavy workload against earlier memory-copy
heavy vector experiments.

## Command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_matmul_naive.ps1
```

## Raw Results

```csv
n,bytes_per_matrix,cpu_ms,cpu_gflops,gpu_h2d_ms,gpu_kernel_ms,gpu_kernel_gflops,gpu_d2h_ms,gpu_total_ms,gpu_total_gflops,kernel_speedup_vs_cpu,total_gpu_speedup_vs_cpu,max_abs_error
128,65536,1.793400,2.338744,0.103744,0.134400,31.207620,0.112256,0.367856,11.402027,13.343750,4.875277,0.000008
256,262144,14.382750,2.332964,0.508880,0.322384,104.082188,0.496944,1.351904,24.820131,44.613722,10.638885,0.000019
512,1048576,138.880200,1.932856,1.049024,1.628608,164.825089,0.881184,3.575392,75.078608,85.275401,38.843349,0.000031
1024,4194304,3115.286950,0.689337,3.127488,11.836880,181.423120,2.166864,17.145344,125.251711,263.184811,181.698716,0.000107
```

## Key Observations

The result is very different from vector addition.

For `1024 x 1024`:

- CPU time: 3115.29 ms
- GPU H2D copy: 3.13 ms
- GPU kernel: 11.84 ms
- GPU D2H copy: 2.17 ms
- GPU total: 17.15 ms

The GPU kernel was about 263x faster than the naive CPU version. Even including
host-device copies, the GPU path was about 182x faster.

For `128 x 128`, GPU total time was already about 4.88x faster. This happened
because matrix multiplication has much more computation per byte than vector
addition.

## Interpretation

Vector addition has low arithmetic intensity. It performs one addition while
moving multiple floats. Data movement dominates.

Matrix multiplication has high arithmetic intensity. For an `N x N` matrix, the
input/output data is `O(N^2)`, but the computation is `O(N^3)`. As `N` grows,
the amount of work grows much faster than the transfer size.

That is why GPU acceleration looks much stronger here. The host-device copy cost
still exists, but the kernel does enough work to amortize it.

The key CUDA lesson:

```text
GPUs shine when there is enough parallel computation per byte moved.
```

