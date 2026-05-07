# CUDA Graphs launch overhead results

## Run

- Date: 2026-05-04
- GPU: NVIDIA GeForce MX450
- Result CSV: `results/cuda_graphs_20260504_222016.csv`
- Plot: `results/cuda_graphs_20260504_222016.png`
- Sizes: 1,024, 4,096, 16,384, 65,536, 262,144, 1,048,576 floats
- Repeat: 1,000
- Block size: 256
- Kernel compute iterations: 1

## What was tested

Two workloads were measured:

| workload | description |
|---|---|
| `kernel_only` | copy input to GPU once, then repeatedly launch only the kernel |
| `copy_kernel_copy` | repeatedly enqueue H2D copy, kernel, and D2H copy |

Each workload compares normal CUDA launch against CUDA Graph replay. The key metrics are:

- `per_iter_wall_us`: end-to-end elapsed time per repeated iteration
- `per_iter_enqueue_us`: CPU-side enqueue time per repeated iteration
- `speedup_wall_vs_normal`: normal launch wall time divided by graph replay wall time
- `speedup_enqueue_vs_normal`: normal launch CPU enqueue time divided by graph replay CPU enqueue time

## Kernel-only results

| size | normal wall us | graph wall us | wall speedup | normal enqueue us | graph enqueue us | enqueue speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 16.502 | 10.124 | 1.63x | 16.369 | 9.945 | 1.65x |
| 4,096 | 18.911 | 13.731 | 1.38x | 18.843 | 13.674 | 1.38x |
| 16,384 | 31.283 | 30.707 | 1.02x | 31.157 | 30.642 | 1.02x |
| 65,536 | 26.239 | 22.064 | 1.19x | 26.083 | 21.878 | 1.19x |
| 262,144 | 49.902 | 49.546 | 1.01x | 42.429 | 48.083 | 0.88x |
| 1,048,576 | 179.979 | 179.390 | 1.00x | 25.806 | 25.374 | 1.02x |

## Copy-kernel-copy results

| size | normal wall us | graph wall us | wall speedup | graph effective GB/s |
|---:|---:|---:|---:|---:|
| 1,024 | 612.447 | 689.658 | 0.89x | 0.01 |
| 4,096 | 649.216 | 507.588 | 1.28x | 0.06 |
| 16,384 | 531.884 | 521.802 | 1.02x | 0.25 |
| 65,536 | 707.758 | 692.273 | 1.02x | 0.76 |
| 262,144 | 1578.340 | 1585.091 | 1.00x | 1.32 |
| 1,048,576 | 4095.906 | 4135.843 | 0.99x | 2.03 |

## Conclusion

- CUDA Graph replay reduced kernel-only launch overhead most clearly for tiny workloads.
- At 1,024 floats, graph replay improved per-iteration wall time from 16.50 us to 10.12 us, a 1.63x speedup.
- At 4,096 floats, graph replay improved per-iteration wall time from 18.91 us to 13.73 us, a 1.38x speedup.
- As the kernel workload grows, launch overhead becomes a smaller fraction of total time, so graph replay speedup approaches 1.0x.
- In the copy-kernel-copy workload, transfer cost and driver queue behavior dominate more strongly. CUDA Graphs did not consistently improve end-to-end time there.

The practical takeaway is that CUDA Graphs are most useful when the application repeatedly submits the same small GPU work graph. They reduce CPU launch overhead, but they do not automatically fix transfer-bound workloads.

## Short explanation

I implemented a CUDA Graphs benchmark that compares normal repeated kernel launches with graph capture and replay. For a tiny kernel-only workload of 1,024 floats repeated 1,000 times, CUDA Graph replay reduced per-iteration time from about 16.5 us to 10.1 us, about a 1.6x improvement. As the input size grew, the kernel execution time dominated and the graph advantage mostly disappeared. This shows that CUDA Graphs are mainly a launch-overhead optimization for repeated small GPU workflows, not a replacement for fixing memory transfer bottlenecks.

