# HPC-System-Optimization
# High-Performance Computing & System Optimization 🚀

This repository contains optimization projects focusing on **Parallel Computing** and **Memory Hierarchy Management**.
The goal is to maximize hardware throughput (CPU/GPU) by implementing **SIMD (AVX)**, **Multi-threading (OpenMP)**, and **GPU Kernel Optimization (CUDA)**.

## 🛠️ Tech Stack
- **Languages:** C++, Python, CUDA C
- **Libraries:** OpenMP, AVX2, cuBLAS
- **Hardware Profiled:** NVIDIA T4/V100 GPU, Intel Xeon CPU, TPU v5e

---

## 📂 Project Summaries

### 1. GPU Kernel Optimization (CUDA) [Folder: 03-GPU-Optimization-CUDA]
**Topic:** Breaking the Memory Wall in Matrix Operations (GEMM, Convolution)
- **Problem:** Naive implementation of Matrix Multiplication and Convolution suffered from high latency due to frequent Global Memory access.
- **Optimization Strategy:**
  - **Shared Memory Tiling:** Reduced global memory traffic by loading data blocks into on-chip Shared Memory.
  - **Constant Memory:** Placed Convolution kernel weights into Constant Memory to utilize the read-only cache.
  - **Coalesced Access:** Aligned memory access patterns to minimize transactions.
- **Result:** Achieved **~1.4x speedup** in GEMM and optimized memory bandwidth efficiency.

### 2. CPU Parallelization & Load Balancing (OpenMP) [Folder: 02-Parallel-Computing-OpenMP]
**Topic:** Softmax Regression Inference & BigInt Multiplication
- **Problem:** Static scheduling caused **Load Imbalance** where some threads finished early while others were still processing heavy workloads.
- **Optimization Strategy:**
  - Applied **Dynamic Scheduling** (`schedule(dynamic)`) to distribute tasks evenly across threads at runtime.
  - Utilized `reduction` clauses to prevent Race Conditions in parallel sums.
- **Result:** Achieved **~1.78x speedup** in Softmax Regression and **7.56x speedup** in BigInt Multiplication (w/ 8 threads).

### 3. SIMD Vectorization (AVX) [Folder: 01-CPU-Optimization-AVX]
**Topic:** Accelerating Matrix Multiplication with AVX2
- **Optimization Strategy:**
  - Utilized **AVX2 intrinsics** to process 256-bit vectors (8 floats) simultaneously.
  - Implemented Loop Unrolling to reduce branch prediction overhead.
- **Result:** Demonstrated significant performance gain over scalar implementation.

---
