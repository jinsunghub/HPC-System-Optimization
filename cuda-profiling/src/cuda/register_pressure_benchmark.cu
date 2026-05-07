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

template <int PRESSURE>
__global__ void registerPressureKernel(const float* input, float* output, std::size_t n, int iters) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float base = input[idx];
    float regs[PRESSURE];

#pragma unroll
    for (int i = 0; i < PRESSURE; ++i) {
        regs[i] = base + static_cast<float>(i + 1) * 0.0001f;
    }

#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
        for (int i = 0; i < PRESSURE; ++i) {
            regs[i] = fmaf(regs[i], 1.000001f, 0.000001f * static_cast<float>(i + 1));
            regs[i] = fmaf(regs[i], -0.000003f, regs[i]);
        }
    }

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < PRESSURE; ++i) {
        sum += regs[i];
    }

    output[idx] = sum;
}

struct Options {
    std::size_t size = 4194304;
    int repeat = 20;
    int block_size = 256;
    int iters = 256;
};

struct KernelInfo {
    int active_blocks_per_sm = 0;
    double theoretical_occupancy = 0.0;
    int registers_per_thread = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t local_bytes_per_thread = 0;
};

struct Result {
    int pressure = 0;
    int grid_size = 0;
    double kernel_ms = 0.0;
    double gop_s = 0.0;
    double slowdown_vs_p4 = 1.0;
    double throughput_vs_p4 = 1.0;
    KernelInfo info;
    double checksum = 0.0;
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
                << "Usage: register_pressure_benchmark [--size 4194304]\n"
                << "                                   [--repeat 20] [--block-size 256]\n"
                << "                                   [--iters 256]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
    }
    if (options.block_size <= 0 || options.block_size > 1024 || options.block_size % 32 != 0) {
        throw std::invalid_argument("--block-size must be a positive multiple of 32 up to 1024");
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

static double checksumOutput(const std::vector<float>& values) {
    double sum = 0.0;
    std::size_t step = std::max<std::size_t>(1, values.size() / 4096);
    for (std::size_t i = 0; i < values.size(); i += step) {
        sum += static_cast<double>(values[i]);
    }
    return sum;
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
    info.local_bytes_per_thread = attrs.localSizeBytes;
    return info;
}

template <int PRESSURE>
static Result runCase(const Options& options,
                      const cudaDeviceProp& props,
                      const float* d_input,
                      float* d_output,
                      std::vector<float>& h_output) {
    int grid_size = static_cast<int>(
        (options.size + static_cast<std::size_t>(options.block_size) - 1) /
        static_cast<std::size_t>(options.block_size));

    KernelInfo info = getKernelInfo(registerPressureKernel<PRESSURE>, options.block_size, props);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    registerPressureKernel<PRESSURE><<<grid_size, options.block_size>>>(
        d_input, d_output, options.size, options.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < options.repeat; ++r) {
        registerPressureKernel<PRESSURE><<<grid_size, options.block_size>>>(
            d_input, d_output, options.size, options.iters);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double kernel_ms = static_cast<double>(elapsed_ms) / static_cast<double>(options.repeat);
    double op_count = static_cast<double>(options.size) * options.iters * PRESSURE * 4.0;

    Result result;
    result.pressure = PRESSURE;
    result.grid_size = grid_size;
    result.kernel_ms = kernel_ms;
    result.gop_s = op_count / (kernel_ms * 1.0e6);
    result.info = info;
    result.checksum = checksumOutput(h_output);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return result;
}

int main(int argc, char** argv) {
    try {
        Options options = parseArgs(argc, argv);

        int device = 0;
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));

        std::cerr << "Using GPU: " << props.name << "\n";
        std::cerr << "size: " << options.size
                  << ", repeat: " << options.repeat
                  << ", block_size: " << options.block_size
                  << ", iters: " << options.iters << "\n";

        std::vector<float> h_input(options.size);
        std::vector<float> h_output(options.size);
        fillInput(h_input);

        float* d_input = nullptr;
        float* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, options.size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, options.size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), options.size * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<Result> results;
        results.push_back(runCase<4>(options, props, d_input, d_output, h_output));
        results.push_back(runCase<8>(options, props, d_input, d_output, h_output));
        results.push_back(runCase<16>(options, props, d_input, d_output, h_output));
        results.push_back(runCase<32>(options, props, d_input, d_output, h_output));
        results.push_back(runCase<64>(options, props, d_input, d_output, h_output));
        results.push_back(runCase<96>(options, props, d_input, d_output, h_output));

        double baseline_ms = results.front().kernel_ms;
        double baseline_gop_s = results.front().gop_s;
        for (Result& result : results) {
            result.slowdown_vs_p4 = result.kernel_ms / baseline_ms;
            result.throughput_vs_p4 = result.gop_s / baseline_gop_s;
        }

        std::cout
            << "pressure,size,iters,repeat,block_size,grid_size,kernel_ms,gop_s,"
            << "slowdown_vs_p4,throughput_vs_p4,active_blocks_per_sm,"
            << "theoretical_occupancy,registers_per_thread,static_shared_bytes,"
            << "local_bytes_per_thread,checksum\n";

        for (const Result& result : results) {
            std::cout << result.pressure << ","
                      << options.size << ","
                      << options.iters << ","
                      << options.repeat << ","
                      << options.block_size << ","
                      << result.grid_size << ","
                      << std::fixed << std::setprecision(6)
                      << result.kernel_ms << ","
                      << result.gop_s << ","
                      << result.slowdown_vs_p4 << ","
                      << result.throughput_vs_p4 << ","
                      << result.info.active_blocks_per_sm << ","
                      << result.info.theoretical_occupancy << ","
                      << result.info.registers_per_thread << ","
                      << result.info.static_shared_bytes << ","
                      << result.info.local_bytes_per_thread << ","
                      << result.checksum << "\n";
        }

        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_input));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
