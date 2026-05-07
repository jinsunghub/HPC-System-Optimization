# CUDA + GPU Profiling Study Plan

## Goal

Build the habit of explaining GPU performance with measurements, not guesses.

## Week 1: CUDA Execution Model

- Learn host vs device code.
- Understand kernel launch syntax.
- Understand thread, block, grid.
- Run `vector_add_benchmark`.
- Change `--block-size` and compare kernel time.

Questions to answer:

- Why does each thread handle one vector element?
- What changes when block size is 128, 256, 512, or 1024?
- Why can tiny inputs be slower on GPU than CPU?

## Week 2: Memory Movement

- Separate host-to-device copy, kernel, and device-to-host copy.
- Compare kernel speedup with total GPU speedup.
- Look for the input size where GPU total time becomes competitive.

Questions to answer:

- Is the kernel the bottleneck?
- Are memory copies dominating total time?
- Why does GPU acceleration depend on data size?

## Week 3: Nsight Systems

- Capture a timeline with `nsys profile`.
- Find `cudaMemcpy` and `vectorAddKernel`.
- Compare the timeline with the CSV output.

Questions to answer:

- Does the GPU sit idle between operations?
- Which part occupies the longest region in the timeline?
- What would change if copies and kernels overlapped?

## Week 4: Next Kernel

Implement one of:

- reduction sum
- naive matrix multiplication
- tiled matrix multiplication with shared memory

The next milestone is to explain memory bandwidth and reuse.

