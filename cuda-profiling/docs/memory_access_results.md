# Global memory access pattern results

## Run

- Date: 2026-05-04
- GPU: NVIDIA GeForce MX450
- Result: `results/memory_access_pattern_20260504_020135.csv`
- Input size: 16,777,216 floats
- Kernel repeat: 30
- Block size: 256

## What was tested

This benchmark keeps the amount of work almost identical, but changes how each thread reads from global memory.

The kernel writes to contiguous output addresses every time. The read address changes with a permutation stride:

```cpp
source_idx = (idx * stride) & (n - 1);
```

With `stride = 1`, neighboring threads in a warp read neighboring addresses. This is coalesced.

With larger odd strides such as `31` or `63`, neighboring threads read addresses that are far apart. This breaks memory coalescing and causes many more memory transactions.

## Results

| stride | kernel ms | effective bandwidth | theoretical occupancy |
|---:|---:|---:|---:|
| 1 | 2.799511 | 47.94 GB/s | 1.00 |
| 3 | 5.452924 | 24.61 GB/s | 1.00 |
| 5 | 8.221478 | 16.33 GB/s | 1.00 |
| 7 | 11.038803 | 12.16 GB/s | 1.00 |
| 9 | 12.441285 | 10.79 GB/s | 1.00 |
| 15 | 12.455462 | 10.78 GB/s | 1.00 |
| 31 | 12.573307 | 10.67 GB/s | 1.00 |
| 63 | 16.770672 | 8.00 GB/s | 1.00 |

## Conclusion

- `stride = 1` reached about 47.94 GB/s.
- `stride = 63` dropped to about 8.00 GB/s.
- That is about a 6.0x bandwidth drop with the same theoretical occupancy.
- The register count, static shared memory, and theoretical occupancy stayed stable.

This is the key lesson: occupancy can be high while performance is still poor. For memory-bound kernels, the access pattern itself can dominate performance.

## Nsight Compute metrics to inspect later

When Nsight Compute is available, this experiment should be profiled with metrics like:

- global memory load efficiency
- memory throughput
- L1/TEX cache hit rate
- L2 cache hit rate
- DRAM throughput
- warp stall reasons related to memory dependency

## Short explanation

I built a CUDA memory access benchmark that keeps the same number of loads and stores but changes whether adjacent threads read adjacent memory. The coalesced version reached about 48 GB/s, while a high-stride permutation dropped to about 8 GB/s. Occupancy remained 1.00 across the sweep, so the result shows that occupancy alone is not enough. Global memory coalescing and memory transaction efficiency can dominate kernel performance.

