#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

__global__ void atomicSumKernel(const float* input, float* output, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

__global__ void sharedReduceKernel(const float* input, float* partial, std::size_t n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + tid;

    float sum = 0.0f;
    if (idx < n) {
        sum += input[idx];
    }
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

__device__ __forceinline__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void shuffleReduceKernel(const float* input, float* partial, std::size_t n) {
    __shared__ float warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + tid;

    float sum = 0.0f;
    if (idx < n) {
        sum += input[idx];
    }
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }

    sum = warpReduceSum(sum);

    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    unsigned int warp_count = (blockDim.x + 31) >> 5;
    sum = tid < warp_count ? warp_sums[lane] : 0.0f;

    if (warp == 0) {
        sum = warpReduceSum(sum);
        if (lane == 0) {
            partial[blockIdx.x] = sum;
        }
    }
}

struct Options {
    std::size_t size = 4194304;
    int repeat = 5;
    int block_size = 256;
};

struct KernelInfo {
    int active_blocks_per_sm = 0;
    double theoretical_occupancy = 0.0;
    int registers_per_thread = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t dynamic_shared_bytes = 0;
};

struct Timing {
    double ms = 0.0;
    double result = 0.0;
    double abs_error = 0.0;
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: reduction_benchmark [--size 4194304] [--repeat 5] [--block-size 256]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
    }
    if (options.block_size <= 0 || options.block_size > 1024 || options.block_size % 32 != 0) {
        throw std::invalid_argument("--block-size must be a multiple of 32 in the range 32..1024");
    }

    return options;
}

static void fillInput(std::vector<float>& input) {
    std::fill(input.begin(), input.end(), 1.0f);
}

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static Timing runCpuSum(const std::vector<float>& input, int repeat, double expected) {
    using clock = std::chrono::steady_clock;

    double total_ms = 0.0;
    double result = 0.0;

    for (int r = 0; r < repeat; ++r) {
        double sum = 0.0;
        auto start = clock::now();
        for (float value : input) {
            sum += static_cast<double>(value);
        }
        auto end = clock::now();

        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        result = sum;
    }

    Timing timing;
    timing.ms = total_ms / static_cast<double>(repeat);
    timing.result = result;
    timing.abs_error = std::abs(result - expected);
    return timing;
}

template <typename Kernel>
static KernelInfo getKernelInfo(
    Kernel kernel,
    int block_size,
    std::size_t dynamic_shared_bytes,
    const cudaDeviceProp& props) {
    KernelInfo info;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &info.active_blocks_per_sm,
        kernel,
        block_size,
        dynamic_shared_bytes));

    cudaFuncAttributes attrs{};
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
    info.theoretical_occupancy =
        static_cast<double>(info.active_blocks_per_sm * block_size) /
        static_cast<double>(props.maxThreadsPerMultiProcessor);
    info.registers_per_thread = attrs.numRegs;
    info.static_shared_bytes = attrs.sharedSizeBytes;
    info.dynamic_shared_bytes = dynamic_shared_bytes;
    return info;
}

static Timing runAtomicSum(
    const float* d_input,
    float* d_output,
    std::size_t n,
    int block_size,
    int repeat,
    double expected) {
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
    atomicSumKernel<<<grid_size, block_size>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        atomicSumKernel<<<grid_size, block_size>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        total_ms += elapsedMs(start, stop);
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    Timing timing;
    timing.ms = total_ms / static_cast<double>(repeat);
    timing.result = result;
    timing.abs_error = std::abs(static_cast<double>(result) - expected);
    return timing;
}

template <typename Kernel>
static Timing runIterativeReduction(
    Kernel kernel,
    const float* d_input,
    float* d_partial_a,
    float* d_partial_b,
    std::size_t n,
    int block_size,
    std::size_t dynamic_shared_bytes,
    int repeat,
    double expected) {
    auto launchPass = [&](const float* input, float* output, std::size_t count) -> std::size_t {
        std::size_t blocks = (count + static_cast<std::size_t>(block_size) * 2 - 1) /
            (static_cast<std::size_t>(block_size) * 2);
        kernel<<<static_cast<int>(blocks), block_size, dynamic_shared_bytes>>>(input, output, count);
        CUDA_CHECK(cudaGetLastError());
        return blocks;
    };

    const float* current_input = d_input;
    float* current_output = d_partial_a;
    std::size_t current_count = n;
    while (current_count > 1) {
        current_count = launchPass(current_input, current_output, current_count);
        current_input = current_output;
        current_output = current_output == d_partial_a ? d_partial_b : d_partial_a;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_ms = 0.0;
    const float* final_ptr = nullptr;

    for (int r = 0; r < repeat; ++r) {
        current_input = d_input;
        current_output = d_partial_a;
        current_count = n;

        CUDA_CHECK(cudaEventRecord(start));
        while (current_count > 1) {
            current_count = launchPass(current_input, current_output, current_count);
            current_input = current_output;
            current_output = current_output == d_partial_a ? d_partial_b : d_partial_a;
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        total_ms += elapsedMs(start, stop);
        final_ptr = current_input;
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, final_ptr, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    Timing timing;
    timing.ms = total_ms / static_cast<double>(repeat);
    timing.result = result;
    timing.abs_error = std::abs(static_cast<double>(result) - expected);
    return timing;
}

static void printRow(
    const char* method,
    const Timing& timing,
    const Timing& cpu,
    const Timing& atomic,
    const KernelInfo& info,
    std::size_t size,
    int repeat,
    int block_size) {
    double speedup_vs_cpu = timing.ms > 0.0 ? cpu.ms / timing.ms : 0.0;
    double speedup_vs_atomic = timing.ms > 0.0 ? atomic.ms / timing.ms : 0.0;

    std::cout << method << ","
              << size << ","
              << repeat << ","
              << block_size << ","
              << std::fixed << std::setprecision(6)
              << timing.ms << ","
              << speedup_vs_cpu << ","
              << speedup_vs_atomic << ","
              << info.active_blocks_per_sm << ","
              << info.theoretical_occupancy << ","
              << info.registers_per_thread << ","
              << info.static_shared_bytes << ","
              << info.dynamic_shared_bytes << ","
              << timing.result << ","
              << timing.abs_error << "\n";
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
        fillInput(input);

        double expected = static_cast<double>(options.size);
        Timing cpu = runCpuSum(input, options.repeat, expected);

        const std::size_t bytes = options.size * sizeof(float);
        std::size_t max_blocks = (options.size + static_cast<std::size_t>(options.block_size) * 2 - 1) /
            (static_cast<std::size_t>(options.block_size) * 2);

        float* d_input = nullptr;
        float* d_atomic_output = nullptr;
        float* d_partial_a = nullptr;
        float* d_partial_b = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_atomic_output, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial_a, max_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial_b, max_blocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

        std::size_t shared_dynamic = static_cast<std::size_t>(options.block_size) * sizeof(float);

        Timing atomic = runAtomicSum(
            d_input,
            d_atomic_output,
            options.size,
            options.block_size,
            options.repeat,
            expected);
        Timing shared = runIterativeReduction(
            sharedReduceKernel,
            d_input,
            d_partial_a,
            d_partial_b,
            options.size,
            options.block_size,
            shared_dynamic,
            options.repeat,
            expected);
        Timing shuffle = runIterativeReduction(
            shuffleReduceKernel,
            d_input,
            d_partial_a,
            d_partial_b,
            options.size,
            options.block_size,
            0,
            options.repeat,
            expected);

        KernelInfo empty_info;
        KernelInfo atomic_info = getKernelInfo(atomicSumKernel, options.block_size, 0, props);
        KernelInfo shared_info = getKernelInfo(sharedReduceKernel, options.block_size, shared_dynamic, props);
        KernelInfo shuffle_info = getKernelInfo(shuffleReduceKernel, options.block_size, 0, props);

        std::cout
            << "method,size,repeat,block_size,time_ms,speedup_vs_cpu,speedup_vs_atomic,"
            << "active_blocks_per_sm,theoretical_occupancy,registers_per_thread,"
            << "static_shared_bytes,dynamic_shared_bytes,result,abs_error\n";

        printRow("cpu_loop", cpu, cpu, atomic, empty_info, options.size, options.repeat, 0);
        printRow("global_atomic", atomic, cpu, atomic, atomic_info, options.size, options.repeat, options.block_size);
        printRow("shared_memory_reduction", shared, cpu, atomic, shared_info, options.size, options.repeat, options.block_size);
        printRow("warp_shuffle_reduction", shuffle, cpu, atomic, shuffle_info, options.size, options.repeat, options.block_size);

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_atomic_output));
        CUDA_CHECK(cudaFree(d_partial_a));
        CUDA_CHECK(cudaFree(d_partial_b));
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

