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

__global__ void vectorAddKernel(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

struct Options {
    std::size_t size = 10000000;
    int repeat = 30;
    std::vector<int> blocks{64, 128, 256, 512, 1024};
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
        } else if (arg == "--blocks") {
            options.blocks = parseIntList(requireValue("--blocks"), "--blocks");
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vector_blocksize_sweep [--size 10000000] [--repeat 30] [--blocks 64,128,256,512,1024]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
    }
    for (int block : options.blocks) {
        if (block <= 0 || block > 1024) {
            throw std::invalid_argument("Each block size must be in the range 1..1024");
        }
    }

    return options;
}

static void fillInputs(std::vector<float>& a, std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>((i % 1024) * 0.25);
        b[i] = static_cast<float>((i % 2048) * 0.125);
    }
}

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static double maxAbsError(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& actual) {
    double max_error = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        double expected = static_cast<double>(a[i]) + static_cast<double>(b[i]);
        max_error = std::max(max_error, std::abs(expected - static_cast<double>(actual[i])));
    }
    return max_error;
}

static Timing timeBlockSize(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& out,
    int block_size,
    int repeat) {
    const std::size_t n = a.size();
    const std::size_t bytes = n * sizeof(float);
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    vectorAddKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        vectorAddKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        total_ms += elapsedMs(start, stop);
    }

    CUDA_CHECK(cudaMemcpy(out.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    Timing timing;
    timing.kernel_ms = static_cast<float>(total_ms / static_cast<double>(repeat));
    timing.max_abs_error = maxAbsError(a, b, out);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

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
        std::cerr << "Size: " << options.size << ", repeat: " << options.repeat << "\n";

        std::vector<float> a(options.size);
        std::vector<float> b(options.size);
        std::vector<float> out(options.size);
        fillInputs(a, b);

        cudaFuncAttributes attrs{};
        CUDA_CHECK(cudaFuncGetAttributes(&attrs, vectorAddKernel));

        std::cout
            << "size,block_size,grid_size,repeat,kernel_ms,effective_bandwidth_gb_s,"
            << "active_blocks_per_sm,theoretical_occupancy,registers_per_thread,"
            << "static_shared_bytes,max_abs_error\n";

        for (int block_size : options.blocks) {
            int active_blocks = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                vectorAddKernel,
                block_size,
                0));

            Timing timing = timeBlockSize(a, b, out, block_size, options.repeat);
            int grid_size = static_cast<int>((options.size + block_size - 1) / block_size);
            double occupancy =
                static_cast<double>(active_blocks * block_size) /
                static_cast<double>(props.maxThreadsPerMultiProcessor);
            double bytes_touched = static_cast<double>(options.size) * sizeof(float) * 3.0;
            double bandwidth = timing.kernel_ms > 0.0f ? bytes_touched / (timing.kernel_ms * 1.0e6) : 0.0;

            std::cout << options.size << ","
                      << block_size << ","
                      << grid_size << ","
                      << options.repeat << ","
                      << std::fixed << std::setprecision(6)
                      << timing.kernel_ms << ","
                      << bandwidth << ","
                      << active_blocks << ","
                      << occupancy << ","
                      << attrs.numRegs << ","
                      << attrs.sharedSizeBytes << ","
                      << timing.max_abs_error << "\n";
        }

        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

