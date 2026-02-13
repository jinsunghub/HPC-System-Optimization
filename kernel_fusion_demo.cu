#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

__global__ void scale_kernel(const float* x, float* tmp, int n, float alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) tmp[idx] = alpha * x[idx];
}

__global__ void bias_kernel(const float* tmp, float* out, int n, float beta) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) out[idx] = tmp[idx] + beta;
}

__global__ void relu_kernel(float* out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(out[idx], 0.0f);
}

__global__ void fused_scale_bias_relu_kernel(const float* x, float* out, int n,
                                             float alpha, float beta) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float v = alpha * x[idx] + beta;
        out[idx] = fmaxf(v, 0.0f);
    }
}

float run_unfused(const float* d_x, float* d_tmp, float* d_out, int n,
                  float alpha, float beta, int iters) {
    const int block = 256;
    const int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        scale_kernel<<<grid, block>>>(d_x, d_tmp, n, alpha);
        bias_kernel<<<grid, block>>>(d_tmp, d_out, n, beta);
        relu_kernel<<<grid, block>>>(d_out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

float run_fused(const float* d_x, float* d_out, int n, float alpha, float beta,
                int iters) {
    const int block = 256;
    const int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        fused_scale_bias_relu_kernel<<<grid, block>>>(d_x, d_out, n, alpha, beta);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

bool verify(const std::vector<float>& x, const std::vector<float>& y,
            float alpha, float beta) {
    const float eps = 1e-5f;
    for (size_t i = 0; i < x.size(); ++i) {
        float expected = std::fmax(alpha * x[i] + beta, 0.0f);
        if (std::fabs(expected - y[i]) > eps) {
            std::cerr << "Mismatch at " << i << ": expected=" << expected
                      << " got=" << y[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int n = 1 << 24;
    int iters = 200;
    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);

    float alpha = 1.618f;
    float beta = -0.42f;

    std::vector<float> h_x(n), h_out(n);
    for (int i = 0; i < n; ++i) h_x[i] = (i % 1024 - 512) * 0.001f;

    float *d_x = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float unfused_ms = run_unfused(d_x, d_tmp, d_out, n, alpha, beta, iters);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_unfused = verify(h_x, h_out, alpha, beta);

    float fused_ms = run_fused(d_x, d_out, n, alpha, beta, iters);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_fused = verify(h_x, h_out, alpha, beta);

    double speedup = unfused_ms / fused_ms;

    std::cout << "N=" << n << ", iterations=" << iters << "\n";
    std::cout << "Unfused (3 kernels): " << unfused_ms << " ms\n";
    std::cout << "Fused   (1 kernel): " << fused_ms << " ms\n";
    std::cout << "Speedup (unfused/fused): " << speedup << "x\n";
    std::cout << "Correctness unfused: " << (ok_unfused ? "PASS" : "FAIL") << "\n";
    std::cout << "Correctness fused: " << (ok_fused ? "PASS" : "FAIL") << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFree(d_out));

    return (ok_unfused && ok_fused) ? 0 : 1;
}
