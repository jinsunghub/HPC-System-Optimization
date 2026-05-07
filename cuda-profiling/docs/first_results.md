# First Results: Vector Add

## Environment

- GPU: NVIDIA GeForce MX450
- GPU memory: 2048 MiB
- Driver: 595.97
- CUDA compiler: 13.2, V13.2.78
- Build: local CUDA redist + Visual Studio 2019 MSVC

## Command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

## Result

```csv
size,bytes,cpu_ms,gpu_h2d_ms,gpu_kernel_ms,gpu_d2h_ms,gpu_total_ms,kernel_speedup_vs_cpu,total_gpu_speedup_vs_cpu,max_abs_error
1024,4096,0.002240,0.329923,0.221370,0.255770,0.932823,0.010119,0.002401,0.000000
10000,40000,0.015600,0.101898,0.073456,0.083366,0.337485,0.212372,0.046224,0.000000
100000,400000,0.263760,0.439184,0.095162,0.286333,0.857309,2.771706,0.307660,0.000000
1000000,4000000,2.418960,3.225123,0.367427,4.903187,8.553776,6.583508,0.282794,0.000000
10000000,40000000,22.500210,27.358297,2.631555,45.357742,75.407700,8.550157,0.298381,0.000000
```

## Interpretation

The CUDA kernel becomes faster than the CPU loop starting around 100,000
elements, but the total GPU path is slower for every tested input size.

The reason is memory movement. Vector addition does very little work per byte:
it reads two floats and writes one float for only one addition. The GPU kernel
can process the operation quickly, but the program still has to copy input
arrays from host to device and copy the result back to host.

For 10,000,000 elements, kernel time was about 2.63 ms while total GPU time was
about 75.41 ms. That means the GPU was not limited by arithmetic here. It was
limited by data movement and overhead.

This is a good first CUDA lesson: measuring only kernel time can make a GPU
program look fast, while end-to-end time can tell a very different story.

## Block Size Sweep

The block-size sweep tested 128, 256, and 512 threads per block. Kernel time
changed a little, but the overall conclusion did not change. The main bottleneck
is not block size. The main bottleneck is host-device data movement.

