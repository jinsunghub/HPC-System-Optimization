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

template <int STRIDE>
__global__ void gatherStrideKernel(const float* input, float* output, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    std::size_t source_idx = (idx * static_cast<std::size_t>(STRIDE)) & (n - 1);

    output[idx] = input[source_idx] + 1.0f;
}

struct Options {
    std::size_t size = 16777216;
    int repeat = 30;
    int block_size = 256;
    std::vector<int> strides{1, 3, 5, 7, 9, 15, 31, 63};
};

struct KernelInfo {
    int active_blocks_per_sm = 0;
    double theoretical_occupancy = 0.0;
    int registers_per_thread = 0;
    std::size_t static_shared_bytes = 0;
};

struct Timing {
    float kernel_ms = 0.0f;
    double max_abs_error = 0.0;
};

static std::vector<int> parseIntList(const std::string& value, const char* name) {
    std::vector<int> out;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        int parsed = std::stoi(item);
        if (parsed <= 0) {
            throw std::invalid_argument(std::string(name) + " must contain positive integers");
        }
        out.push_back(parsed);
    }

    if (out.empty()) {
        throw std::invalid_argument(std::string(name) + " must contain at least one value");
    }

    return out;
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
        } else if (arg == "--strides") {
            options.strides = parseIntList(requireValue("--strides"), "--strides");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: memory_access_pattern_benchmark [--size 16777216] [--repeat 30]\n"
                << "                                       [--block-size 256] [--strides 1,3,5,7,9,15,31,63]\n";
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
    if ((options.size & (options.size - 1)) != 0) {
        throw std::invalid_argument("--size must be a power of two for this permutation benchmark");
    }
    for (int stride : options.strides) {
        if (stride != 1 && stride != 3 && stride != 5 && stride != 7 &&
            stride != 9 && stride != 15 && stride != 31 && stride != 63) {
            throw std::invalid_argument("Supported strides are 1,3,5,7,9,15,31,63");
        }
    }

    return options;
}

static void fillInput(std::vector<float>& input) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 4096) * 0.5f;
    }
}

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static double maxAbsErrorForStride(
    const std::vector<float>& input,
    const std::vector<float>& output,
    int stride) {
    double max_error = 0.0;
    std::size_t n = input.size();
    for (std::size_t idx = 0; idx < n; ++idx) {
        std::size_t source_idx = (idx * static_cast<std::size_t>(stride)) & (n - 1);
        double expected = static_cast<double>(input[source_idx]) + 1.0;
        max_error = std::max(max_error, std::abs(expected - static_cast<double>(output[idx])));
    }
    return max_error;
}

template <int STRIDE>
static KernelInfo getKernelInfo(int block_size, const cudaDeviceProp& props) {
    KernelInfo info;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &info.active_blocks_per_sm,
        gatherStrideKernel<STRIDE>,
        block_size,
        0));

    cudaFuncAttributes attrs{};
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, gatherStrideKernel<STRIDE>));
    info.theoretical_occupancy =
        static_cast<double>(info.active_blocks_per_sm * block_size) /
        static_cast<double>(props.maxThreadsPerMultiProcessor);
    info.registers_per_thread = attrs.numRegs;
    info.static_shared_bytes = attrs.sharedSizeBytes;
    return info;
}

template <int STRIDE>
static Timing runStride(
    const std::vector<float>& input,
    std::vector<float>& output,
    int block_size,
    int repeat) {
    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    gatherStrideKernel<STRIDE><<<grid_size, block_size>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        gatherStrideKernel<STRIDE><<<grid_size, block_size>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        total_ms += elapsedMs(start, stop);
    }

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    Timing timing;
    timing.kernel_ms = static_cast<float>(total_ms / static_cast<double>(repeat));
    timing.max_abs_error = maxAbsErrorForStride(input, output, STRIDE);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return timing;
}

template <int STRIDE>
static void printStrideResult(
    const std::vector<float>& input,
    std::vector<float>& output,
    int block_size,
    int repeat,
    const cudaDeviceProp& props) {
    Timing timing = runStride<STRIDE>(input, output, block_size, repeat);
    KernelInfo info = getKernelInfo<STRIDE>(block_size, props);

    std::size_t n = input.size();
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    double bytes_touched = static_cast<double>(n) * sizeof(float) * 2.0;
    double bandwidth = timing.kernel_ms > 0.0f ? bytes_touched / (timing.kernel_ms * 1.0e6) : 0.0;

    std::cout << n << ","
              << STRIDE << ","
              << block_size << ","
              << grid_size << ","
              << repeat << ","
              << std::fixed << std::setprecision(6)
              << timing.kernel_ms << ","
              << bandwidth << ","
              << info.active_blocks_per_sm << ","
              << info.theoretical_occupancy << ","
              << info.registers_per_thread << ","
              << info.static_shared_bytes << ","
              << timing.max_abs_error << "\n";
}

static void dispatchStride(
    int stride,
    const std::vector<float>& input,
    std::vector<float>& output,
    int block_size,
    int repeat,
    const cudaDeviceProp& props) {
    switch (stride) {
        case 1:
            printStrideResult<1>(input, output, block_size, repeat, props);
            break;
        case 3:
            printStrideResult<3>(input, output, block_size, repeat, props);
            break;
        case 5:
            printStrideResult<5>(input, output, block_size, repeat, props);
            break;
        case 7:
            printStrideResult<7>(input, output, block_size, repeat, props);
            break;
        case 9:
            printStrideResult<9>(input, output, block_size, repeat, props);
            break;
        case 15:
            printStrideResult<15>(input, output, block_size, repeat, props);
            break;
        case 31:
            printStrideResult<31>(input, output, block_size, repeat, props);
            break;
        case 63:
            printStrideResult<63>(input, output, block_size, repeat, props);
            break;
        default:
            throw std::invalid_argument("Unsupported stride");
    }
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
                  << ", block_size: " << options.block_size << "\n";

        std::vector<float> input(options.size);
        std::vector<float> output(options.size);
        fillInput(input);

        std::cout
            << "size,stride,block_size,grid_size,repeat,kernel_ms,effective_bandwidth_gb_s,"
            << "active_blocks_per_sm,theoretical_occupancy,registers_per_thread,"
            << "static_shared_bytes,max_abs_error\n";

        for (int stride : options.strides) {
            dispatchStride(stride, input, output, options.block_size, options.repeat, props);
        }

        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
