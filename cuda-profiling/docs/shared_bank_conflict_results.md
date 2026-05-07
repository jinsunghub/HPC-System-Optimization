# Shared memory bank conflict results

## Run

- GPU: NVIDIA GeForce MX450
- Result CSV: `results/shared_bank_conflict_20260504_223521.csv`
- Plot: `results/shared_bank_conflict_20260504_223521.png`
- Blocks: 256
- Block size: 256
- Repeat: 20
- Inner shared-memory iterations per kernel: 2,048
- Requested strides: 1, 2, 4, 8, 16, 32

## What was tested

The kernel maps each warp lane to a shared-memory address:

```text
index = warp_base + lane * effective_stride
```

For 32-bank shared memory with 4-byte floats, `effective_stride=1` maps lanes to different banks. Power-of-two strides cause bank conflicts:

| effective stride | expected conflict degree |
|---:|---:|
| 1 | 1-way |
| 2 | 2-way |
| 4 | 4-way |
| 8 | 8-way |
| 16 | 16-way |
| 32 | 32-way |

The padded cases use `effective_stride = requested_stride + 1`, for example 32 becomes 33. That changes the bank mapping back to conflict-free because the stride is no longer a multiple of the bank count.

## Results

| method | requested stride | effective stride | conflict degree | total ms | shared Gaccess/s | slowdown vs stride 1 | speedup vs conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| `conflict_free` | 1 | 1 | 1 | 1.207824 | 222.25 | 1.00x | 1.00x |
| `conflict` | 2 | 2 | 2 | 1.796664 | 149.41 | 1.49x | 1.00x |
| `padded` | 2 | 3 | 1 | 1.195827 | 224.48 | 0.99x | 1.50x |
| `conflict` | 4 | 4 | 4 | 3.550888 | 75.60 | 2.94x | 1.00x |
| `padded` | 4 | 5 | 1 | 1.188045 | 225.95 | 0.98x | 2.99x |
| `conflict` | 8 | 8 | 8 | 5.593314 | 47.99 | 4.63x | 1.00x |
| `padded` | 8 | 9 | 1 | 0.912077 | 294.31 | 0.76x | 6.13x |
| `conflict` | 16 | 16 | 16 | 10.895192 | 24.64 | 9.02x | 1.00x |
| `padded` | 16 | 17 | 1 | 0.913101 | 293.98 | 0.76x | 11.93x |
| `conflict` | 32 | 32 | 32 | 21.793089 | 12.32 | 18.04x | 1.00x |
| `padded` | 32 | 33 | 1 | 0.911894 | 294.37 | 0.75x | 23.90x |

## Conclusion

- Shared memory is fast only when warp lanes access banks efficiently.
- Increasing the stride from 1 to 32 changed the measured time from 1.21 ms to 21.79 ms.
- The 32-way conflict case was about 18.0x slower than the stride-1 conflict-free baseline.
- Padding the access pattern recovered most of the lost performance. For requested stride 32, padding changed effective stride 32 to 33 and improved runtime from 21.79 ms to 0.91 ms.
- This is the same layout idea used in real tiled kernels: a small padding column can avoid bank conflicts when reading transposed or strided shared-memory data.

