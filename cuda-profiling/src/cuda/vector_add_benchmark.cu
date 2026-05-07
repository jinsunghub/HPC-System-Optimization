#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

__global__ void vectorAddKernel(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

struct Options {
    std::vector<std::size_t> sizes{1024, 10000, 100000, 1000000, 10000000};
    int repeat = 10;
    int block_size = 256;
};

struct GpuTiming {
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
};

static std::vector<std::size_t> parseSizes(const std::string& value) {
    std::vector<std::size_t> sizes;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || parsed == 0) {
            throw std::invalid_argument("Invalid size: " + item);
        }
        sizes.push_back(static_cast<std::size_t>(parsed));
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
                << "Usage: vector_add_benchmark [--sizes 1024,10000] [--repeat 10] [--block-size 256]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
    }
    if (options.block_size <= 0 || options.block_size > 1024) {
        throw std::invalid_argument("--block-size must be in the range 1..1024");
    }

    return options;
}

static void fillInputs(std::vector<float>& a, std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>((i % 1024) * 0.25);
        b[i] = static_cast<float>((i % 2048) * 0.125);
    }
}

static double runCpuVectorAdd(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int repeat) {
    using clock = std::chrono::steady_clock;
    double total_ms = 0.0;

    for (int r = 0; r < repeat; ++r) {
        auto start = clock::now();
        for (std::size_t i = 0; i < c.size(); ++i) {
            c[i] = a[i] + b[i];
        }
        auto end = clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total_ms / repeat;
}

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static GpuTiming runGpuVectorAdd(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int repeat,
    int block_size) {
    const std::size_t n = a.size();
    const std::size_t bytes = n * sizeof(float);

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

    GpuTiming totals;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(total_start));

        CUDA_CHECK(cudaEventRecord(h2d_start));
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(h2d_stop));

        CUDA_CHECK(cudaEventRecord(kernel_start));
        vectorAddKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(d2h_start));
        CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(d2h_stop));

        CUDA_CHECK(cudaEventRecord(total_stop));
        CUDA_CHECK(cudaEventSynchronize(total_stop));

        totals.h2d_ms += elapsedMs(h2d_start, h2d_stop);
        totals.kernel_ms += elapsedMs(kernel_start, kernel_stop);
        totals.d2h_ms += elapsedMs(d2h_start, d2h_stop);
        totals.total_ms += elapsedMs(total_start, total_stop);
    }

    totals.h2d_ms /= repeat;
    totals.kernel_ms /= repeat;
    totals.d2h_ms /= repeat;
    totals.total_ms /= repeat;

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

    return totals;
}

static double maxAbsError(const std::vector<float>& expected, const std::vector<float>& actual) {
    double max_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        max_error = std::max(max_error, static_cast<double>(std::abs(expected[i] - actual[i])));
    }
    return max_error;
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
        std::cerr << "Repeat: " << options.repeat << ", block_size: " << options.block_size << "\n";

        std::cout
            << "size,bytes,cpu_ms,gpu_h2d_ms,gpu_kernel_ms,gpu_d2h_ms,gpu_total_ms,"
            << "kernel_speedup_vs_cpu,total_gpu_speedup_vs_cpu,max_abs_error\n";

        for (std::size_t n : options.sizes) {
            std::vector<float> a(n);
            std::vector<float> b(n);
            std::vector<float> cpu_out(n);
            std::vector<float> gpu_out(n);

            fillInputs(a, b);

            double cpu_ms = runCpuVectorAdd(a, b, cpu_out, options.repeat);
            GpuTiming gpu = runGpuVectorAdd(a, b, gpu_out, options.repeat, options.block_size);
            double error = maxAbsError(cpu_out, gpu_out);

            double kernel_speedup = gpu.kernel_ms > 0.0f ? cpu_ms / gpu.kernel_ms : 0.0;
            double total_speedup = gpu.total_ms > 0.0f ? cpu_ms / gpu.total_ms : 0.0;

            std::cout << n << ","
                      << n * sizeof(float) << ","
                      << std::fixed << std::setprecision(6)
                      << cpu_ms << ","
                      << gpu.h2d_ms << ","
                      << gpu.kernel_ms << ","
                      << gpu.d2h_ms << ","
                      << gpu.total_ms << ","
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

