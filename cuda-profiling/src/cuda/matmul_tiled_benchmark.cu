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

constexpr int TILE = 16;

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

__global__ void matmulTiledKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int tile_count = (n + TILE - 1) / TILE;

    for (int tile = 0; tile < tile_count; ++tile) {
        int a_col = tile * TILE + tx;
        int b_row = tile * TILE + ty;

        tile_a[ty][tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tile_a[ty][k] * tile_b[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

struct Options {
    std::vector<int> sizes{256, 512, 1024};
    int repeat = 2;
};

struct KernelTiming {
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
};

struct Result {
    double cpu_ms = 0.0;
    float h2d_ms = 0.0f;
    KernelTiming naive;
    KernelTiming tiled;
    double naive_error = 0.0;
    double tiled_error = 0.0;
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: matmul_tiled_benchmark [--sizes 256,512,1024] [--repeat 2]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
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

template <typename LaunchKernel>
static KernelTiming timeKernelAndCopyBack(
    LaunchKernel launch,
    float* d_c,
    std::vector<float>& out,
    std::size_t bytes) {
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;

    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    CUDA_CHECK(cudaEventRecord(kernel_start));
    launch();
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));

    KernelTiming timing;
    timing.kernel_ms = elapsedMs(kernel_start, kernel_stop);
    timing.d2h_ms = elapsedMs(d2h_start, d2h_stop);

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    return timing;
}

static Result runOnce(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& cpu_out,
    std::vector<float>& naive_out,
    std::vector<float>& tiled_out,
    int n) {
    const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t bytes = elems * sizeof(float);

    Result result;
    result.cpu_ms = runCpuMatmul(a, b, cpu_out, n);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));

    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));
    result.h2d_ms = elapsedMs(h2d_start, h2d_stop);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    result.naive = timeKernelAndCopyBack(
        [&]() { matmulNaiveKernel<<<grid, block>>>(d_a, d_b, d_c, n); },
        d_c,
        naive_out,
        bytes);

    result.tiled = timeKernelAndCopyBack(
        [&]() { matmulTiledKernel<<<grid, block>>>(d_a, d_b, d_c, n); },
        d_c,
        tiled_out,
        bytes);

    result.naive_error = maxAbsError(cpu_out, naive_out);
    result.tiled_error = maxAbsError(cpu_out, tiled_out);

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return result;
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
        std::cerr << "Repeat: " << options.repeat << ", tile: " << TILE << "x" << TILE << "\n";

        std::cout
            << "n,bytes_per_matrix,cpu_ms,cpu_gflops,h2d_ms,naive_kernel_ms,"
            << "naive_kernel_gflops,tiled_kernel_ms,tiled_kernel_gflops,"
            << "naive_total_ms,tiled_total_ms,tiled_speedup_vs_naive_kernel,"
            << "tiled_speedup_vs_naive_total,naive_speedup_vs_cpu_total,"
            << "tiled_speedup_vs_cpu_total,naive_max_abs_error,tiled_max_abs_error\n";

        for (int n : options.sizes) {
            const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
            const std::size_t bytes = elems * sizeof(float);

            std::vector<float> a(elems);
            std::vector<float> b(elems);
            std::vector<float> cpu_out(elems);
            std::vector<float> naive_out(elems);
            std::vector<float> tiled_out(elems);

            fillMatrix(a, n, 3);
            fillMatrix(b, n, 11);

            Result total;
            for (int r = 0; r < options.repeat; ++r) {
                Result one = runOnce(a, b, cpu_out, naive_out, tiled_out, n);
                total.cpu_ms += one.cpu_ms;
                total.h2d_ms += one.h2d_ms;
                total.naive.kernel_ms += one.naive.kernel_ms;
                total.naive.d2h_ms += one.naive.d2h_ms;
                total.tiled.kernel_ms += one.tiled.kernel_ms;
                total.tiled.d2h_ms += one.tiled.d2h_ms;
                total.naive_error = std::max(total.naive_error, one.naive_error);
                total.tiled_error = std::max(total.tiled_error, one.tiled_error);
            }

            double inv_repeat = 1.0 / static_cast<double>(options.repeat);
            double cpu_ms = total.cpu_ms * inv_repeat;
            double h2d_ms = total.h2d_ms * inv_repeat;
            double naive_kernel_ms = total.naive.kernel_ms * inv_repeat;
            double naive_d2h_ms = total.naive.d2h_ms * inv_repeat;
            double tiled_kernel_ms = total.tiled.kernel_ms * inv_repeat;
            double tiled_d2h_ms = total.tiled.d2h_ms * inv_repeat;
            double naive_total_ms = h2d_ms + naive_kernel_ms + naive_d2h_ms;
            double tiled_total_ms = h2d_ms + tiled_kernel_ms + tiled_d2h_ms;

            double tiled_vs_naive_kernel = tiled_kernel_ms > 0.0 ? naive_kernel_ms / tiled_kernel_ms : 0.0;
            double tiled_vs_naive_total = tiled_total_ms > 0.0 ? naive_total_ms / tiled_total_ms : 0.0;
            double naive_vs_cpu_total = naive_total_ms > 0.0 ? cpu_ms / naive_total_ms : 0.0;
            double tiled_vs_cpu_total = tiled_total_ms > 0.0 ? cpu_ms / tiled_total_ms : 0.0;

            std::cout << n << ","
                      << bytes << ","
                      << std::fixed << std::setprecision(6)
                      << cpu_ms << ","
                      << matmulGflops(n, cpu_ms) << ","
                      << h2d_ms << ","
                      << naive_kernel_ms << ","
                      << matmulGflops(n, naive_kernel_ms) << ","
                      << tiled_kernel_ms << ","
                      << matmulGflops(n, tiled_kernel_ms) << ","
                      << naive_total_ms << ","
                      << tiled_total_ms << ","
                      << tiled_vs_naive_kernel << ","
                      << tiled_vs_naive_total << ","
                      << naive_vs_cpu_total << ","
                      << tiled_vs_cpu_total << ","
                      << total.naive_error << ","
                      << total.tiled_error << "\n";
        }

        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

