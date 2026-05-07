# CUDA stream overlap results

## Run

- Date: 2026-05-04
- GPU: NVIDIA GeForce MX450
- Result: `results/stream_overlap_20260504_214528.csv`
- Input size: 16,777,216 floats
- Total input size: 64 MB
- Total host-device transfer per run: 128 MB
- Chunks: 8
- Chunk size: 2,097,152 floats
- Kernel compute iterations: 64
- Repeat: 5
- GPU async copy engines: 2
- Concurrent kernels: 1

## What was tested

This benchmark compares four execution styles:

| method | description |
|---|---|
| `pageable_sequential` | pageable host memory, chunk by chunk H2D -> kernel -> D2H |
| `pinned_sequential` | pinned host memory, same sequential chunk order |
| `pinned_streams_2` | pinned host memory, two CUDA streams |
| `pinned_streams_4` | pinned host memory, four CUDA streams |

The goal is to test whether H2D copy, kernel execution, and D2H copy can be pipelined across chunks.

## Results

| method | total ms | effective transfer GB/s | speedup vs pageable | speedup vs pinned |
|---|---:|---:|---:|---:|
| `pageable_sequential` | 53.763480 | 2.50 | 1.00x | 0.91x |
| `pinned_sequential` | 49.117900 | 2.73 | 1.09x | 1.00x |
| `pinned_streams_2` | 29.699440 | 4.52 | 1.81x | 1.65x |
| `pinned_streams_4` | 27.636960 | 4.86 | 1.95x | 1.78x |

## Conclusion

- Pinned memory alone improved sequential execution from 53.76 ms to 49.12 ms.
- Pinned memory plus two streams reduced runtime to 29.70 ms.
- Pinned memory plus four streams reduced runtime to 27.64 ms.
- Four streams were about 1.95x faster than pageable sequential execution.
- Four streams were about 1.78x faster than pinned sequential execution.

The GPU reported two async copy engines, so overlapping transfer and compute is possible. The result shows that stream pipelining can hide part of the host-device transfer cost.

The `effective_transfer_gb_s` value is not pure PCIe bandwidth. It is an end-to-end pipeline metric using total H2D+D2H bytes divided by total elapsed time, so it includes kernel work.

## Short explanation

I implemented a CUDA stream overlap benchmark that splits a vector workload into chunks and compares sequential execution against pinned-memory multi-stream execution. The sequential baseline performed H2D copy, kernel, and D2H copy one chunk at a time. With pinned memory and four CUDA streams, the same workload improved from about 53.76 ms to 27.64 ms, roughly a 1.95x speedup. This showed that optimizing a GPU program is not only about making kernels faster; overlapping data transfer and compute can also improve end-to-end throughput.

