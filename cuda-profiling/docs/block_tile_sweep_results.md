# Block size and tile size sweep results

## Run

- GPU: NVIDIA GeForce MX450
- Vector result: `results/vector_blocksize_sweep_20260504_014855.csv`
- Matmul result: `results/matmul_tile_sweep_20260504_014855.csv`

## Vector add block size sweep

Input size was 10,000,000 floats and each block size was repeated 30 times.

| block size | kernel ms | effective bandwidth | theoretical occupancy |
|---:|---:|---:|---:|
| 64 | 2.464528 | 48.69 GB/s | 1.00 |
| 128 | 2.507974 | 47.85 GB/s | 1.00 |
| 256 | 2.504367 | 47.92 GB/s | 1.00 |
| 512 | 2.504569 | 47.91 GB/s | 1.00 |
| 1024 | 2.513455 | 47.74 GB/s | 1.00 |

Conclusion:

- Every tested block size reached theoretical occupancy 1.00.
- Kernel time stayed almost flat, around 2.46 to 2.51 ms.
- For this memory bandwidth bound vector add kernel, block size was not the main bottleneck once there were enough threads.
- The best measured block size was 64, but the margin was small enough that it should not be treated as a universal rule.

## Matrix multiplication tile size sweep

Tile sizes 8, 16, and 32 were compared. Each tile uses shared memory for one tile of A and one tile of B.

For `1024 x 1024`:

| tile | threads/block | shared memory/block | kernel ms | kernel GFLOP/s | total ms |
|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 512 B | 21.078768 | 101.88 | 26.124081 |
| 16 | 256 | 2048 B | 7.206464 | 297.99 | 12.093121 |
| 32 | 1024 | 8192 B | 6.777024 | 316.88 | 11.886048 |

Conclusion:

- Tile 16 was much faster than tile 8 because it reused more data from shared memory per block.
- Tile 32 was slightly faster than tile 16 on this GPU for this input, but the improvement was much smaller than the 8 to 16 jump.
- All three tile sizes reported theoretical occupancy 1.00, so occupancy alone did not explain performance.
- This is the important lesson: high occupancy is useful, but it is not the same thing as high performance. Memory reuse, arithmetic intensity, register pressure, shared memory usage, and scheduling all matter.

