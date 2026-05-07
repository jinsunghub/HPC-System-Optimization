#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
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

__device__ __noinline__ float pathA(float x, int iters) {
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 0.000003f);
        x = fmaf(x, -0.000002f, x);
    }
    return x;
}

__device__ __noinline__ float pathB(float x, int iters) {
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 0.999999f, 0.000007f);
        x = fmaf(x, 0.000003f, x);
    }
    return x;
}

__global__ void noBranchKernel(const float* input, float* output, std::size_t n, int iters) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = pathA(input[idx], iters);
    }
}

__global__ void warpUniformBranchKernel(const float* input, float* output, std::size_t n, int iters) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    bool take_a = ((idx >> 5) & 1) == 0;
    if (take_a) {
        output[idx] = pathA(input[idx], iters);
    } else {
        output[idx] = pathB(input[idx], iters);
    }
}

__global__ void threadDivergentBranchKernel(const float* input, float* output, std::size_t n, int iters) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    bool take_a = (idx & 1) == 0;
    if (take_a) {
        output[idx] = pathA(input[idx], iters);
    } else {
        output[idx] = pathB(input[idx], iters);
    }
}

enum class Pattern {
    NoBranch,
    WarpUniform,
    ThreadDivergent
};

struct Options {
    std::size_t size = 4194304;
    int repeat = 20;
    int block_size = 256;
    int iters = 128;
};

struct KernelInfo {
    int active_blocks_per_sm = 0;
    double theoretical_occupancy = 0.0;
    int registers_per_thread = 0;
    std::size_t static_shared_bytes = 0;
};

struct Timing {
    float kernel_ms = 0.0f;
    double sample_max_abs_error = 0.0;
};

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

        if (arg == "--size") {
            unsigned long long parsed = std::strtoull(requireValue("--size").c_str(), nullptr, 10);
            if (parsed == 0) {
                throw std::invalid_argument("--size must be positive");
            }
            options.size = static_cast<std::size_t>(parsed);
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(requireValue("--repeat"));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(requireValue("--block-size"));
        } else if (arg == "--iters") {
            options.iters = std::stoi(requireValue("--iters"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: warp_divergence_benchmark [--size 4194304] [--repeat 20]\n"
                << "                                 [--block-size 256] [--iters 128]\n";
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
    if (options.iters <= 0) {
        throw std::invalid_argument("--iters must be positive");
    }

    return options;
}

static void fillInput(std::vector<float>& input) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i % 1024) + 1) * 0.001f;
    }
}

static float hostPathA(float x, int iters) {
    for (int i = 0; i < iters; ++i) {
        x = std::fma(x, 1.000001f, 0.000003f);
        x = std::fma(x, -0.000002f, x);
    }
    return x;
}

static float hostPathB(float x, int iters) {
    for (int i = 0; i < iters; ++i) {
        x = std::fma(x, 0.999999f, 0.000007f);
        x = std::fma(x, 0.000003f, x);
    }
    return x;
}

static const char* patternName(Pattern pattern) {
    switch (pattern) {
        case Pattern::NoBranch:
            return "no_branch";
        case Pattern::WarpUniform:
            return "warp_uniform_branch";
        case Pattern::ThreadDivergent:
            return "thread_divergent_branch";
        default:
            return "unknown";
    }
}

static float expectedValue(float input, std::size_t idx, Pattern pattern, int iters) {
    switch (pattern) {
        case Pattern::NoBranch:
            return hostPathA(input, iters);
        case Pattern::WarpUniform:
            return (((idx >> 5) & 1) == 0) ? hostPathA(input, iters) : hostPathB(input, iters);
        case Pattern::ThreadDivergent:
            return ((idx & 1) == 0) ? hostPathA(input, iters) : hostPathB(input, iters);
        default:
            return 0.0f;
    }
}

static double sampleMaxAbsError(
    const std::vector<float>& input,
    const std::vector<float>& output,
    Pattern pattern,
    int iters) {
    const std::size_t samples = std::min<std::size_t>(4096, input.size());
    double max_error = 0.0;

    for (std::size_t i = 0; i < samples; ++i) {
        std::size_t idx = samples == input.size()
            ? i
            : (i * (input.size() - 1)) / (samples - 1);
        double expected = static_cast<double>(expectedValue(input[idx], idx, pattern, iters));
        max_error = std::max(max_error, std::abs(expected - static_cast<double>(output[idx])));
    }

    return max_error;
}

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

template <typename Kernel>
static KernelInfo getKernelInfo(Kernel kernel, int block_size, const cudaDeviceProp& props) {
    KernelInfo info;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &info.active_blocks_per_sm,
        kernel,
        block_size,
        0));

    cudaFuncAttributes attrs{};
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
    info.theoretical_occupancy =
        static_cast<double>(info.active_blocks_per_sm * block_size) /
        static_cast<double>(props.maxThreadsPerMultiProcessor);
    info.registers_per_thread = attrs.numRegs;
    info.static_shared_bytes = attrs.sharedSizeBytes;
    return info;
}

template <typename LaunchKernel>
static Timing timeKernel(
    LaunchKernel launch,
    const std::vector<float>& input,
    std::vector<float>& output,
    Pattern pattern,
    int repeat,
    int iters,
    float* d_output,
    std::size_t bytes) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        total_ms += elapsedMs(start, stop);
    }

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    Timing timing;
    timing.kernel_ms = static_cast<float>(total_ms / static_cast<double>(repeat));
    timing.sample_max_abs_error = sampleMaxAbsError(input, output, pattern, iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return timing;
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
        std::cerr << "Size: " << options.size
                  << ", repeat: " << options.repeat
                  << ", block_size: " << options.block_size
                  << ", iters: " << options.iters << "\n";

        std::vector<float> input(options.size);
        std::vector<float> output(options.size);
        fillInput(input);

        const std::size_t bytes = options.size * sizeof(float);
        const int grid_size = static_cast<int>((options.size + options.block_size - 1) / options.block_size);

        float* d_input = nullptr;
        float* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

        std::cout
            << "pattern,size,iters,block_size,grid_size,repeat,kernel_ms,"
            << "relative_to_no_branch,relative_to_warp_uniform,active_blocks_per_sm,"
            << "theoretical_occupancy,registers_per_thread,static_shared_bytes,"
            << "sample_max_abs_error\n";

        KernelInfo no_branch_info = getKernelInfo(noBranchKernel, options.block_size, props);
        Timing no_branch = timeKernel(
            [&]() { noBranchKernel<<<grid_size, options.block_size>>>(d_input, d_output, options.size, options.iters); },
            input,
            output,
            Pattern::NoBranch,
            options.repeat,
            options.iters,
            d_output,
            bytes);

        KernelInfo uniform_info = getKernelInfo(warpUniformBranchKernel, options.block_size, props);
        Timing uniform = timeKernel(
            [&]() { warpUniformBranchKernel<<<grid_size, options.block_size>>>(d_input, d_output, options.size, options.iters); },
            input,
            output,
            Pattern::WarpUniform,
            options.repeat,
            options.iters,
            d_output,
            bytes);

        KernelInfo divergent_info = getKernelInfo(threadDivergentBranchKernel, options.block_size, props);
        Timing divergent = timeKernel(
            [&]() { threadDivergentBranchKernel<<<grid_size, options.block_size>>>(d_input, d_output, options.size, options.iters); },
            input,
            output,
            Pattern::ThreadDivergent,
            options.repeat,
            options.iters,
            d_output,
            bytes);

        auto printRow = [&](Pattern pattern, const Timing& timing, const KernelInfo& info) {
            double relative_to_no_branch = no_branch.kernel_ms > 0.0f
                ? timing.kernel_ms / no_branch.kernel_ms
                : 0.0;
            double relative_to_uniform = uniform.kernel_ms > 0.0f
                ? timing.kernel_ms / uniform.kernel_ms
                : 0.0;

            std::cout << patternName(pattern) << ","
                      << options.size << ","
                      << options.iters << ","
                      << options.block_size << ","
                      << grid_size << ","
                      << options.repeat << ","
                      << std::fixed << std::setprecision(6)
                      << timing.kernel_ms << ","
                      << relative_to_no_branch << ","
                      << relative_to_uniform << ","
                      << info.active_blocks_per_sm << ","
                      << info.theoretical_occupancy << ","
                      << info.registers_per_thread << ","
                      << info.static_shared_bytes << ","
                      << timing.sample_max_abs_error << "\n";
        };

        printRow(Pattern::NoBranch, no_branch, no_branch_info);
        printRow(Pattern::WarpUniform, uniform, uniform_info);
        printRow(Pattern::ThreadDivergent, divergent, divergent_info);

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

