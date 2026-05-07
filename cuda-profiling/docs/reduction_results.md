# Parallel reduction results

## Run

- Date: 2026-05-04
- GPU: NVIDIA GeForce MX450
- Result: `results/reduction_20260504_021529.csv`
- Input size: 4,194,304 floats
- Kernel repeat: 5
- Block size: 256

## What was tested

The input array was filled with `1.0f`, so the expected sum was exactly `4,194,304`.

Four methods were compared:

| method | description |
|---|---|
| `cpu_loop` | serial CPU sum |
| `global_atomic` | every GPU thread calls `atomicAdd` on one global scalar |
| `shared_memory_reduction` | each block reduces in shared memory, then launches additional reduction passes |
| `warp_shuffle_reduction` | each block uses warp shuffle instructions for most of the reduction |

## Results

| method | time ms | speedup vs CPU | speedup vs atomic | occupancy | error |
|---|---:|---:|---:|---:|---:|
| `cpu_loop` | 4.396960 | 1.00x | 2.15x | - | 0 |
| `global_atomic` | 9.448864 | 0.47x | 1.00x | 1.00 | 0 |
| `shared_memory_reduction` | 0.346093 | 12.70x | 27.30x | 1.00 | 0 |
| `warp_shuffle_reduction` | 0.344064 | 12.78x | 27.46x | 1.00 | 0 |

## Conclusion

- Global atomic reduction was slower than the CPU loop for this input size.
- Shared memory reduction was about 27.3x faster than global atomic.
- Warp shuffle reduction was slightly faster than shared memory reduction on this GPU.
- All GPU methods reported theoretical occupancy 1.00, but their performance differed dramatically.

This is the key lesson: reductions are not just "parallel loops." A good reduction reduces contention, uses block-local aggregation, and minimizes synchronization and global memory traffic.

## Short explanation

I implemented a parallel reduction benchmark with a global atomic baseline, a shared memory block reduction, and a warp shuffle reduction. The global atomic version serialized many updates to one memory location and took about 9.45 ms. The optimized reductions first aggregated values inside each block and then reduced partial sums, bringing the runtime down to about 0.34 ms. This was roughly 27x faster than the atomic baseline while preserving exact correctness for the test input.

