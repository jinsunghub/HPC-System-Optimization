# Lab 2 Results: GPU Data Reuse

## Environment

- GPU: NVIDIA GeForce MX450
- GPU memory: 2048 MiB
- Driver: 595.97
- CUDA compiler: 13.2, V13.2.78
- Build: local CUDA redist + Visual Studio 2019 MSVC

## Experiment

This lab compares two GPU execution styles for repeated vector accumulation:

```cpp
c[i] += a[i] + b[i]
```

The two GPU styles are:

- roundtrip: copy CPU to GPU, run one kernel, copy GPU to CPU on every operation
- reuse: copy CPU to GPU once, run many kernels on GPU, copy GPU to CPU once

The CPU baseline runs the same number of accumulation operations.

## Command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_reuse.ps1
```

## Raw Results

```csv
size,ops,bytes,cpu_ms,roundtrip_h2d_ms,roundtrip_kernel_ms,roundtrip_d2h_ms,roundtrip_total_ms,reuse_h2d_ms,reuse_kernel_ms,reuse_d2h_ms,reuse_total_ms,roundtrip_speedup_vs_cpu,reuse_speedup_vs_cpu,reuse_speedup_vs_roundtrip,max_abs_error
100000,1,400000,0.114233,0.816544,0.121813,0.237280,1.215840,0.555776,0.107157,0.314912,0.995339,0.093954,0.114768,1.221534,0.000000
100000,10,400000,0.802467,5.543659,0.636587,2.126144,8.729397,0.561739,0.391829,0.299296,1.266848,0.091927,0.633436,6.890642,0.000000
100000,100,400000,7.731667,56.130936,7.423179,22.462116,90.685410,0.597387,3.659403,0.353493,4.634571,0.085258,1.668260,19.567165,0.000000
1000000,1,4000000,1.186533,4.325014,0.390667,1.852736,6.627605,4.346048,0.405344,2.050613,6.817387,0.179029,0.174045,0.972162,0.000000
1000000,10,4000000,9.003033,43.450573,3.929333,18.387146,66.504173,4.470859,3.446528,1.867733,9.800598,0.135375,0.918621,6.785726,0.000000
1000000,100,4000000,87.648600,436.643616,40.661194,183.820312,667.781738,4.538198,33.069027,2.003307,39.629131,0.131253,2.211721,16.850779,0.000000
10000000,1,40000000,9.897133,40.236160,3.345696,13.510998,57.178017,40.590954,3.360800,14.351456,58.318230,0.173093,0.169709,0.980448,0.000000
10000000,10,40000000,90.721533,403.579590,33.503052,144.427261,582.281677,40.590103,32.762070,15.367637,88.738251,0.155804,1.022350,6.561789,0.000000
10000000,100,40000000,921.820467,4052.726074,334.922455,1441.840576,5837.357422,40.445663,326.630127,14.878006,381.968353,0.157917,2.413343,15.282307,0.000000
```

## Key Observations

For one operation, reuse does not help much because it still pays the initial
copy cost and the final copy cost.

For repeated operations, reuse changes the result dramatically:

- 1,000,000 elements, 100 ops: reuse is about 2.21x faster than CPU.
- 10,000,000 elements, 10 ops: reuse is roughly equal to CPU, about 1.02x faster.
- 10,000,000 elements, 100 ops: reuse is about 2.41x faster than CPU.

The roundtrip version is always bad because it repeats host-device copies every
operation. At 10,000,000 elements and 100 ops, roundtrip took about 5837 ms,
while reuse took about 382 ms.

That makes reuse about 15.28x faster than roundtrip for the largest case.

## Interpretation

Lab 1 showed that GPU kernel time can be fast while total GPU time is slow
because copies dominate.

Lab 2 shows the fix: keep data resident on the GPU when possible. Once the input
and output arrays stay on the GPU, copy cost is amortized over many kernels.

This is one reason deep learning workloads fit GPUs well. Training and serving
usually do not copy every tensor back to the CPU after every tiny operation.
They keep tensors on GPU and run many kernels over them.

The lesson is:

```text
GPU speed appears when computation per transfer is high enough.
```

This idea is often called arithmetic intensity or data movement amortization.

