# Warp divergence results

## Run

- Date: 2026-05-04
- GPU: NVIDIA GeForce MX450
- Result: `results/warp_divergence_20260504_021005.csv`
- Input size: 4,194,304 floats
- Kernel repeat: 20
- Block size: 256
- Inner compute iterations: 128

## What was tested

Three kernels were compared:

| pattern | branch behavior |
|---|---|
| `no_branch` | every thread executes path A |
| `warp_uniform_branch` | half the warps execute path A and half execute path B, but each warp stays uniform |
| `thread_divergent_branch` | neighboring threads in the same warp split between path A and path B |

The important comparison is `warp_uniform_branch` versus `thread_divergent_branch`.

Both execute path A for half of the threads and path B for half of the threads. The difference is whether a single warp can stay on one path or has to serialize both paths.

## Results

| pattern | kernel ms | relative to uniform | theoretical occupancy | registers/thread |
|---|---:|---:|---:|---:|
| `no_branch` | 1.287992 | 1.03x | 1.00 | 9 |
| `warp_uniform_branch` | 1.245061 | 1.00x | 1.00 | 9 |
| `thread_divergent_branch` | 2.418835 | 1.94x | 1.00 | 9 |

## Conclusion

- `thread_divergent_branch` was about 1.94x slower than `warp_uniform_branch`.
- Theoretical occupancy stayed at 1.00 for all three kernels.
- Register count and shared memory usage stayed the same.
- Therefore the slowdown came from branch divergence, not from lower occupancy.

This is the key lesson: in CUDA, a warp executes one instruction stream at a time. If lanes in the same warp take different branches, the warp serializes the branch paths and masks inactive lanes.

## Short explanation

I measured warp divergence by comparing a branch that is uniform within each warp against a branch that splits neighboring lanes inside the same warp. Both kernels executed the same amount of path A and path B work across the whole grid, and both had theoretical occupancy 1.00. The divergent version was about 1.94x slower, showing that occupancy alone does not capture control-flow efficiency.

