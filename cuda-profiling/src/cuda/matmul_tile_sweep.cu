#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

template <int TILE>
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

struct TileResult {
    int tile = 0;
    int threads_per_block = 0;
    int active_blocks_per_sm = 0;
    double theoretical_occupancy = 0.0;
    int registers_per_thread = 0;
    std::size_t static_shared_bytes = 0;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    double max_abs_error = 0.0;
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
            std::cout << "Usage: matmul_tile_sweep [--sizes 256,512,1024] [--repeat 2]\n";
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

template <int TILE>
static TileResult runTile(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& cpu_out,
    std::vector<float>& gpu_out,
    int n,
    int repeat,
    const cudaDeviceProp& props) {
    const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t bytes = elems * sizeof(float);
    constexpr int threads_per_block = TILE * TILE;

    if (threads_per_block > props.maxThreadsPerBlock) {
        throw std::runtime_error("Tile uses more threads than this GPU allows per block");
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    matmulTiledKernel<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double kernel_total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(kernel_start));
        matmulTiledKernel<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));
        kernel_total_ms += elapsedMs(kernel_start, kernel_stop);
    }

    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));

    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        matmulTiledKernel<TILE>,
        threads_per_block,
        0));

    cudaFuncAttributes attrs{};
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, matmulTiledKernel<TILE>));

    TileResult result;
    result.tile = TILE;
    result.threads_per_block = threads_per_block;
    result.active_blocks_per_sm = active_blocks;
    result.theoretical_occupancy =
        static_cast<double>(active_blocks * threads_per_block) /
        static_cast<double>(props.maxThreadsPerMultiProcessor);
    result.registers_per_thread = attrs.numRegs;
    result.static_shared_bytes = attrs.sharedSizeBytes;
    result.h2d_ms = elapsedMs(h2d_start, h2d_stop);
    result.kernel_ms = static_cast<float>(kernel_total_ms / static_cast<double>(repeat));
    result.d2h_ms = elapsedMs(d2h_start, d2h_stop);
    result.max_abs_error = maxAbsError(cpu_out, gpu_out);

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));
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
        std::cerr << "Repeat: " << options.repeat << ", tiles: 8,16,32\n";

        std::cout
            << "n,tile,threads_per_block,active_blocks_per_sm,theoretical_occupancy,"
            << "registers_per_thread,static_shared_bytes,cpu_ms,h2d_ms,kernel_ms,"
            << "kernel_gflops,d2h_ms,total_ms,total_speedup_vs_cpu,max_abs_error\n";

        for (int n : options.sizes) {
            const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

            std::vector<float> a(elems);
            std::vector<float> b(elems);
            std::vector<float> cpu_out(elems);
            std::vector<float> gpu_out(elems);

            fillMatrix(a, n, 3);
            fillMatrix(b, n, 11);

            double cpu_ms = runCpuMatmul(a, b, cpu_out, n);
            std::vector<TileResult> results;
            results.push_back(runTile<8>(a, b, cpu_out, gpu_out, n, options.repeat, props));
            results.push_back(runTile<16>(a, b, cpu_out, gpu_out, n, options.repeat, props));
            results.push_back(runTile<32>(a, b, cpu_out, gpu_out, n, options.repeat, props));

            for (const TileResult& result : results) {
                double total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
                double speedup = total_ms > 0.0 ? cpu_ms / total_ms : 0.0;

                std::cout << n << ","
                          << result.tile << ","
                          << result.threads_per_block << ","
                          << result.active_blocks_per_sm << ","
                          << std::fixed << std::setprecision(6)
                          << result.theoretical_occupancy << ","
                          << result.registers_per_thread << ","
                          << result.static_shared_bytes << ","
                          << cpu_ms << ","
                          << result.h2d_ms << ","
                          << result.kernel_ms << ","
                          << matmulGflops(n, result.kernel_ms) << ","
                          << result.d2h_ms << ","
                          << total_ms << ","
                          << speedup << ","
                          << result.max_abs_error << "\n";
            }
        }

        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
