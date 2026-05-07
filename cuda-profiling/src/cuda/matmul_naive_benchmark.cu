#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t status = (call);                                       \
        if (status != cudaSuccess) {                                       \
            std::ostringstream oss;                                        \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                << " - " << cudaGetErrorString(status);                   \
            throw std::runtime_error(oss.str());                           \
        }                                                                  \
    } while (0)

__global__ void matmulNaiveKernel(const float* a, const float* b, float* c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

struct Options {
    std::vector<int> sizes{128, 256, 512, 1024};
    int repeat = 2;
    int block_size = 16;
};

struct GpuTiming {
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
};

static std::vector<int> parseSizes(const std::string& value) {
    std::vector<int> sizes;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        int parsed = std::stoi(item);
        if (parsed <= 0) {
            throw std::invalid_argument("Matrix sizes must be positive");
        }
        sizes.push_back(parsed);
    }

    if (sizes.empty()) {
        throw std::invalid_argument("--sizes must contain at least one positive integer");
    }

    return sizes;
}

static Options parseArgs(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--sizes") {
            options.sizes = parseSizes(requireValue("--sizes"));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(requireValue("--repeat"));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(requireValue("--block-size"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: matmul_naive_benchmark [--sizes 128,256,512] "
                << "[--repeat 2] [--block-size 16]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
    }
    if (options.block_size <= 0 || options.block_size > 32) {
        throw std::invalid_argument("--block-size must be in the range 1..32");
    }

    return options;
}

static void fillMatrix(std::vector<float>& matrix, int n, int salt) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            int value = (row * 17 + col * 31 + salt) % 97;
            matrix[row * n + col] = static_cast<float>(value) / 97.0f;
        }
    }
}

static double runCpuMatmul(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int n) {
    using clock = std::chrono::steady_clock;
    std::fill(c.begin(), c.end(), 0.0f);

    auto start = clock::now();
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    auto end = clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static GpuTiming runGpuMatmul(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int n,
    int block_size) {
    const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t bytes = elems * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cudaEvent_t total_start;
    cudaEvent_t total_stop;
    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;

    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    dim3 block(block_size, block_size);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventRecord(total_start));

    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_stop));

    CUDA_CHECK(cudaEventRecord(kernel_start));
    matmulNaiveKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_stop));

    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    GpuTiming timing;
    timing.h2d_ms = elapsedMs(h2d_start, h2d_stop);
    timing.kernel_ms = elapsedMs(kernel_start, kernel_stop);
    timing.d2h_ms = elapsedMs(d2h_start, d2h_stop);
    timing.total_ms = elapsedMs(total_start, total_stop);

    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_stop));
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return timing;
}

static double maxAbsError(const std::vector<float>& expected, const std::vector<float>& actual) {
    double max_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        max_error = std::max(max_error, static_cast<double>(std::abs(expected[i] - actual[i])));
    }
    return max_error;
}

static double matmulGflops(int n, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }

    double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    return flops / (ms * 1.0e6);
}

int main(int argc, char** argv) {
    try {
        Options options = parseArgs(argc, argv);

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            std::cerr << "No CUDA devices found.\n";
            return 1;
        }

        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        CUDA_CHECK(cudaSetDevice(0));

        std::cerr << "Using GPU: " << props.name << "\n";
        std::cerr << "Repeat: " << options.repeat << ", block_size: " << options.block_size << "x"
                  << options.block_size << "\n";

        std::cout
            << "n,bytes_per_matrix,cpu_ms,cpu_gflops,gpu_h2d_ms,gpu_kernel_ms,"
            << "gpu_kernel_gflops,gpu_d2h_ms,gpu_total_ms,gpu_total_gflops,"
            << "kernel_speedup_vs_cpu,total_gpu_speedup_vs_cpu,max_abs_error\n";

        for (int n : options.sizes) {
            const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
            const std::size_t bytes = elems * sizeof(float);

            std::vector<float> a(elems);
            std::vector<float> b(elems);
            std::vector<float> cpu_out(elems);
            std::vector<float> gpu_out(elems);

            fillMatrix(a, n, 3);
            fillMatrix(b, n, 11);

            double cpu_total = 0.0;
            GpuTiming gpu_total;
            double error = 0.0;

            for (int r = 0; r < options.repeat; ++r) {
                double cpu_ms = runCpuMatmul(a, b, cpu_out, n);
                GpuTiming gpu = runGpuMatmul(a, b, gpu_out, n, options.block_size);

                cpu_total += cpu_ms;
                gpu_total.h2d_ms += gpu.h2d_ms;
                gpu_total.kernel_ms += gpu.kernel_ms;
                gpu_total.d2h_ms += gpu.d2h_ms;
                gpu_total.total_ms += gpu.total_ms;
                error = std::max(error, maxAbsError(cpu_out, gpu_out));
            }

            double inv_repeat = 1.0 / static_cast<double>(options.repeat);
            double cpu_ms = cpu_total * inv_repeat;
            gpu_total.h2d_ms = static_cast<float>(gpu_total.h2d_ms * inv_repeat);
            gpu_total.kernel_ms = static_cast<float>(gpu_total.kernel_ms * inv_repeat);
            gpu_total.d2h_ms = static_cast<float>(gpu_total.d2h_ms * inv_repeat);
            gpu_total.total_ms = static_cast<float>(gpu_total.total_ms * inv_repeat);

            double kernel_speedup = gpu_total.kernel_ms > 0.0f ? cpu_ms / gpu_total.kernel_ms : 0.0;
            double total_speedup = gpu_total.total_ms > 0.0f ? cpu_ms / gpu_total.total_ms : 0.0;

            std::cout << n << ","
                      << bytes << ","
                      << std::fixed << std::setprecision(6)
                      << cpu_ms << ","
                      << matmulGflops(n, cpu_ms) << ","
                      << gpu_total.h2d_ms << ","
                      << gpu_total.kernel_ms << ","
                      << matmulGflops(n, gpu_total.kernel_ms) << ","
                      << gpu_total.d2h_ms << ","
                      << gpu_total.total_ms << ","
                      << matmulGflops(n, gpu_total.total_ms) << ","
                      << kernel_speedup << ","
                      << total_speedup << ","
                      << error << "\n";
        }

        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

