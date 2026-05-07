# CUDA stream overlap sweep results

## Run

- Date: 2026-05-04
- GPU: NVIDIA GeForce MX450
- Result CSV: `results/stream_overlap_sweep_20260504_220159.csv`
- Plot: `results/stream_overlap_sweep_20260504_220159.png`
- Input size: 16,777,216 floats
- Total input size: 64 MB
- Total host-device transfer per run: 128 MB
- Repeat: 5
- Block size: 256
- Chunks swept: 2, 4, 8, 16
- Streams swept: 1, 2, 4, 8
- Kernel compute iterations swept: 16, 64, 256
- GPU async copy engines: 2
- Concurrent kernels: 1

## Validity note

The existing benchmark requires `streams <= chunks`, because each stream is assigned chunk work. The requested combinations where streams exceed chunks were kept in the CSV with `status=invalid_streams_gt_chunks`:

| chunks | invalid streams |
|---:|---|
| 2 | 4, 8 |
| 4 | 8 |

This produced 63 valid measured rows and 9 invalid marker rows.

## Best measured stream configurations

| iters | best method | chunks | streams | total ms | effective transfer GB/s | speedup vs pageable | speedup vs pinned |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | `pinned_streams_8` | 16 | 8 | 25.980220 | 5.17 | 2.21x | 1.98x |
| 64 | `pinned_streams_8` | 16 | 8 | 25.915060 | 5.18 | 2.20x | 1.97x |
| 256 | `pinned_streams_4` | 16 | 4 | 26.150640 | 5.13 | 2.35x | 2.14x |

## Top overall results

| rank | method | chunks | streams | iters | total ms | effective transfer GB/s | speedup vs pageable | speedup vs pinned |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `pinned_streams_8` | 16 | 8 | 64 | 25.915060 | 5.18 | 2.20x | 1.97x |
| 2 | `pinned_streams_4` | 16 | 4 | 64 | 25.942120 | 5.17 | 2.19x | 1.97x |
| 3 | `pinned_streams_8` | 16 | 8 | 16 | 25.980220 | 5.17 | 2.21x | 1.98x |
| 4 | `pinned_streams_4` | 16 | 4 | 16 | 26.036360 | 5.16 | 2.21x | 1.97x |
| 5 | `pinned_streams_4` | 16 | 4 | 256 | 26.150640 | 5.13 | 2.35x | 2.14x |
| 6 | `pinned_streams_8` | 16 | 8 | 256 | 26.213140 | 5.12 | 2.35x | 2.13x |

## Conclusion

- The best overall configuration was 16 chunks, 8 streams, 64 compute iterations at 25.92 ms.
- The fastest results consistently used 16 chunks and either 4 or 8 streams.
- Moving from sequential pinned execution to stream overlap gave roughly 1.9x to 2.1x speedup in the best cases.
- The GPU reports two async copy engines, so the improvement is consistent with overlapping H2D copy, kernel work, and D2H copy across chunks.
- 8 streams did not materially beat 4 streams on this GPU. The best 4-stream and 8-stream results were within about 0.1 ms, so either is reasonable and 4 streams is still the cleaner practical choice.
