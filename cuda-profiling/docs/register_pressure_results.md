# Register pressure and occupancy results

## Run

- GPU: NVIDIA GeForce MX450
- Result CSV: `results/register_pressure_20260504_231245.csv`
- Plot: `results/register_pressure_20260504_231245.png`
- Input size: 4,194,304 floats
- Repeat: 20
- Block size: 256
- Inner compute iterations: 256
- Pressure levels: 4, 8, 16, 32, 64, 96

## What was tested

Each template specialization keeps a different number of per-thread accumulator values live:

```text
pressure = number of live accumulator values per thread
```

This increases register pressure and lets CUDA report the resulting resource usage through `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

The benchmark records:

- `registers_per_thread`: actual register count after compilation
- `active_blocks_per_sm`: theoretical active blocks per SM
- `theoretical_occupancy`: active threads divided by max SM threads
- `kernel_ms`: average kernel time
- `gop_s`: normalized arithmetic throughput estimate
- `local_bytes_per_thread`: whether register spilling to local memory occurred

## Results

| pressure | registers/thread | active blocks/SM | theoretical occupancy | kernel ms | GOP/s | slowdown vs P4 | throughput vs P4 | local bytes/thread |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 11 | 4 | 1.00 | 7.044275 | 2438.84 | 1.00x | 1.00x | 0 |
| 8 | 15 | 4 | 1.00 | 11.394970 | 3015.34 | 1.62x | 1.24x | 0 |
| 16 | 23 | 4 | 1.00 | 21.105766 | 3255.96 | 3.00x | 1.34x | 0 |
| 32 | 64 | 4 | 1.00 | 42.098108 | 3264.73 | 5.98x | 1.34x | 0 |
| 64 | 72 | 3 | 0.75 | 84.276636 | 3261.61 | 11.96x | 1.34x | 0 |
| 96 | 128 | 2 | 0.50 | 126.750476 | 3252.98 | 17.99x | 1.33x | 0 |

## Analysis

- Register pressure increased as intended: actual register usage rose from 11 to 128 registers per thread.
- Occupancy stayed at 1.00 through pressure 32, then dropped to 0.75 at pressure 64 and 0.50 at pressure 96.
- There was no register spilling in this run: `local_bytes_per_thread` stayed at 0 for all cases.
- Kernel time increased with pressure because each pressure level also performs more accumulator work.
- The normalized throughput was more informative than raw time. Throughput improved from pressure 4 to pressure 16/32, then stayed nearly flat even as occupancy dropped from 1.00 to 0.50.

This is the important lesson: higher occupancy is not automatically higher performance. The pressure 96 case had half the theoretical occupancy of the baseline, but normalized throughput remained about 1.33x the pressure-4 baseline. In this arithmetic-heavy kernel, there was enough independent work per thread to keep the pipelines busy despite lower occupancy.

## Conclusion

- Occupancy is a diagnostic metric, not the final optimization target.
- Register pressure can reduce occupancy by limiting how many blocks fit on each SM.
- Lower occupancy becomes a problem when there is not enough parallelism to hide latency.
- More registers can also expose more instruction-level parallelism, so the best point is workload-dependent.
- In this experiment, occupancy dropped at high pressure, but throughput did not collapse because the kernel was compute-heavy and had no local-memory spills.

