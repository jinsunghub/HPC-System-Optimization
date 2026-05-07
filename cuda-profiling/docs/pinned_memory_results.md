# Lab 3 Results: Pinned Host Memory

## Environment

- GPU: NVIDIA GeForce MX450
- GPU memory: 2048 MiB
- Driver: 595.97
- CUDA compiler: 13.2, V13.2.78
- Build: local CUDA redist + Visual Studio 2019 MSVC

## Experiment

This lab measures CPU-GPU copy speed with two types of host memory:

- pageable memory: normal host memory from `std::vector`
- pinned memory: page-locked host memory from `cudaMallocHost`

The benchmark measures:

- H2D: host to device copy
- D2H: device to host copy
- GB/s throughput
- pinned speedup over pageable

## Command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_pinned.ps1
```

## Raw Results

```csv
size,bytes,pageable_h2d_ms,pageable_h2d_gbps,pageable_d2h_ms,pageable_d2h_gbps,pinned_h2d_ms,pinned_h2d_gbps,pinned_d2h_ms,pinned_d2h_gbps,h2d_speedup,d2h_speedup,max_abs_error
100000,400000,0.142710,2.802887,0.191160,2.092488,0.141780,2.821272,0.134080,2.983294,1.006559,1.425716,0.000000
1000000,4000000,1.368580,2.922737,1.681180,2.379281,1.318740,3.033198,1.258970,3.177200,1.037794,1.335361,0.000000
10000000,40000000,13.528020,2.956826,13.955960,2.866159,12.851810,3.112402,12.357020,3.237026,1.052616,1.129395,0.000000
30000000,120000000,39.811600,3.014197,42.300790,2.836826,38.682120,3.102208,36.788950,3.261849,1.029199,1.149823,0.000000
```

## Key Observations

Pinned memory was consistently faster, but not dramatically faster on this
machine.

For the largest transfer:

- pageable H2D: 39.81 ms, 3.01 GB/s
- pinned H2D: 38.68 ms, 3.10 GB/s
- H2D speedup: 1.03x

For D2H:

- pageable D2H: 42.30 ms, 2.84 GB/s
- pinned D2H: 36.79 ms, 3.26 GB/s
- D2H speedup: 1.15x

The D2H benefit was more visible than the H2D benefit.

## Interpretation

Pinned memory helps because the host pages are locked in physical memory. That
makes it easier for CUDA to transfer data with DMA. Normal pageable memory may
require extra staging through an internal pinned buffer.

In this experiment, pinned memory improved transfer throughput, but the gain was
modest. This is normal. The benefit depends on GPU, PCIe link, driver behavior,
transfer size, and whether asynchronous copies are used.

The practical lesson is:

```text
Pinned memory can reduce transfer overhead, but it does not remove transfer cost.
Avoiding unnecessary transfers is usually more important.
```

This connects to Lab 2. Data reuse gave a much larger improvement than pinned
memory because it reduced the number of transfers. Pinned memory only makes
remaining transfers somewhat faster.

## When To Use Pinned Memory

Use pinned memory when:

- large host buffers are transferred repeatedly
- the buffer can be reused
- you want to overlap transfer and compute with `cudaMemcpyAsync`

Avoid using too much pinned memory because it makes life harder for the OS memory
manager.

