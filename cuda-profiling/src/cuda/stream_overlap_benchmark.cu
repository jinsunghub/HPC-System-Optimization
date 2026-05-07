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

__global__ void transformKernel(const float* input, float* output, std::size_t n, int iters) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float x = input[idx];
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 0.000003f);
        x = fmaf(x, -0.000002f, x);
    }
    output[idx] = x;
}

struct Options {
    std::size_t size = 16777216;
    int chunks = 8;
    int repeat = 5;
    int block_size = 256;
    int iters = 64;
    std::vector<int> stream_counts{2, 4};
};

struct Result {
    std::string method;
    int streams = 1;
    double ms = 0.0;
    double transfer_gb_s = 0.0;
    double speedup_vs_pageable = 0.0;
    double speedup_vs_pinned = 0.0;
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
        } else if (arg == "--chunks") {
            options.chunks = std::stoi(requireValue("--chunks"));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(requireValue("--repeat"));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(requireValue("--block-size"));
        } else if (arg == "--iters") {
            options.iters = std::stoi(requireValue("--iters"));
        } else if (arg == "--stream-counts") {
            options.stream_counts = parseIntList(requireValue("--stream-counts"), "--stream-counts");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: stream_overlap_benchmark [--size 16777216] [--chunks 8]\n"
                << "                                [--repeat 5] [--block-size 256]\n"
                << "                                [--iters 64] [--stream-counts 2,4]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.chunks <= 0) {
        throw std::invalid_argument("--chunks must be positive");
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
    if (options.size % static_cast<std::size_t>(options.chunks) != 0) {
        throw std::invalid_argument("--size must be divisible by --chunks");
    }
    for (int count : options.stream_counts) {
        if (count <= 0 || count > options.chunks) {
            throw std::invalid_argument("Each stream count must be in the range 1..chunks");
        }
    }

    return options;
}

static void fillInput(float* input, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>((i % 1024) + 1) * 0.001f;
    }
}

static float hostTransform(float x, int iters) {
    for (int i = 0; i < iters; ++i) {
        x = std::fma(x, 1.000001f, 0.000003f);
        x = std::fma(x, -0.000002f, x);
    }
    return x;
}

static double sampleMaxAbsError(const float* input, const float* output, std::size_t n, int iters) {
    const std::size_t samples = std::min<std::size_t>(4096, n);
    double max_error = 0.0;

    for (std::size_t i = 0; i < samples; ++i) {
        std::size_t idx = samples == n ? i : (i * (n - 1)) / (samples - 1);
        double expected = static_cast<double>(hostTransform(input[idx], iters));
        max_error = std::max(max_error, std::abs(expected - static_cast<double>(output[idx])));
    }

    return max_error;
}

static double transferGbPerSec(std::size_t n, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    double transferred_bytes = static_cast<double>(n) * sizeof(float) * 2.0;
    return transferred_bytes / (ms * 1.0e6);
}

static double nowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void launchChunk(
    const float* d_input,
    float* d_output,
    std::size_t chunk_elems,
    int block_size,
    int iters,
    cudaStream_t stream) {
    int grid_size = static_cast<int>((chunk_elems + block_size - 1) / block_size);
    transformKernel<<<grid_size, block_size, 0, stream>>>(d_input, d_output, chunk_elems, iters);
    CUDA_CHECK(cudaGetLastError());
}

static Result runPageableSequential(
    const float* h_input,
    float* h_output,
    std::size_t n,
    int chunks,
    int repeat,
    int block_size,
    int iters) {
    std::size_t chunk_elems = n / static_cast<std::size_t>(chunks);
    std::size_t chunk_bytes = chunk_elems * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, chunk_bytes));

    auto runOnce = [&]() {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            std::size_t offset = static_cast<std::size_t>(chunk) * chunk_elems;
            CUDA_CHECK(cudaMemcpy(d_input, h_input + offset, chunk_bytes, cudaMemcpyHostToDevice));
            launchChunk(d_input, d_output, chunk_elems, block_size, iters, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_output + offset, d_output, chunk_bytes, cudaMemcpyDeviceToHost));
        }
    };

    runOnce();
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        double start = nowMs();
        runOnce();
        CUDA_CHECK(cudaDeviceSynchronize());
        double stop = nowMs();
        total_ms += stop - start;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    Result result;
    result.method = "pageable_sequential";
    result.streams = 1;
    result.ms = total_ms / static_cast<double>(repeat);
    result.transfer_gb_s = transferGbPerSec(n, result.ms);
    result.max_abs_error = sampleMaxAbsError(h_input, h_output, n, iters);
    return result;
}

static Result runPinnedSequential(
    const float* h_input,
    float* h_output,
    std::size_t n,
    int chunks,
    int repeat,
    int block_size,
    int iters) {
    std::size_t chunk_elems = n / static_cast<std::size_t>(chunks);
    std::size_t chunk_bytes = chunk_elems * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, chunk_bytes));
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto runOnce = [&]() {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            std::size_t offset = static_cast<std::size_t>(chunk) * chunk_elems;
            CUDA_CHECK(cudaMemcpyAsync(d_input, h_input + offset, chunk_bytes, cudaMemcpyHostToDevice, stream));
            launchChunk(d_input, d_output, chunk_elems, block_size, iters, stream);
            CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_output, chunk_bytes, cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    runOnce();

    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        double start = nowMs();
        runOnce();
        double stop = nowMs();
        total_ms += stop - start;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    Result result;
    result.method = "pinned_sequential";
    result.streams = 1;
    result.ms = total_ms / static_cast<double>(repeat);
    result.transfer_gb_s = transferGbPerSec(n, result.ms);
    result.max_abs_error = sampleMaxAbsError(h_input, h_output, n, iters);
    return result;
}

static Result runPinnedStreams(
    const float* h_input,
    float* h_output,
    std::size_t n,
    int chunks,
    int stream_count,
    int repeat,
    int block_size,
    int iters) {
    std::size_t chunk_elems = n / static_cast<std::size_t>(chunks);
    std::size_t chunk_bytes = chunk_elems * sizeof(float);

    std::vector<cudaStream_t> streams(stream_count);
    std::vector<float*> d_inputs(stream_count, nullptr);
    std::vector<float*> d_outputs(stream_count, nullptr);

    for (int i = 0; i < stream_count; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_inputs[i], chunk_bytes));
        CUDA_CHECK(cudaMalloc(&d_outputs[i], chunk_bytes));
    }

    auto runOnce = [&]() {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            int sid = chunk % stream_count;
            std::size_t offset = static_cast<std::size_t>(chunk) * chunk_elems;
            CUDA_CHECK(cudaMemcpyAsync(d_inputs[sid], h_input + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[sid]));
            launchChunk(d_inputs[sid], d_outputs[sid], chunk_elems, block_size, iters, streams[sid]);
            CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_outputs[sid], chunk_bytes, cudaMemcpyDeviceToHost, streams[sid]));
        }
        for (cudaStream_t stream : streams) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    runOnce();

    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        double start = nowMs();
        runOnce();
        double stop = nowMs();
        total_ms += stop - start;
    }

    for (int i = 0; i < stream_count; ++i) {
        CUDA_CHECK(cudaFree(d_inputs[i]));
        CUDA_CHECK(cudaFree(d_outputs[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    Result result;
    result.method = "pinned_streams_" + std::to_string(stream_count);
    result.streams = stream_count;
    result.ms = total_ms / static_cast<double>(repeat);
    result.transfer_gb_s = transferGbPerSec(n, result.ms);
    result.max_abs_error = sampleMaxAbsError(h_input, h_output, n, iters);
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
        std::cerr << "Size: " << options.size
                  << ", chunks: " << options.chunks
                  << ", repeat: " << options.repeat
                  << ", block_size: " << options.block_size
                  << ", iters: " << options.iters << "\n";
        std::cerr << "asyncEngineCount: " << props.asyncEngineCount
                  << ", concurrentKernels: " << props.concurrentKernels << "\n";

        std::vector<float> pageable_input(options.size);
        std::vector<float> pageable_output(options.size);
        fillInput(pageable_input.data(), options.size);

        float* pinned_input = nullptr;
        float* pinned_output = nullptr;
        std::size_t bytes = options.size * sizeof(float);
        CUDA_CHECK(cudaHostAlloc(&pinned_input, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&pinned_output, bytes, cudaHostAllocDefault));
        fillInput(pinned_input, options.size);

        std::vector<Result> results;
        results.push_back(runPageableSequential(
            pageable_input.data(),
            pageable_output.data(),
            options.size,
            options.chunks,
            options.repeat,
            options.block_size,
            options.iters));
        results.push_back(runPinnedSequential(
            pinned_input,
            pinned_output,
            options.size,
            options.chunks,
            options.repeat,
            options.block_size,
            options.iters));

        for (int stream_count : options.stream_counts) {
            results.push_back(runPinnedStreams(
                pinned_input,
                pinned_output,
                options.size,
                options.chunks,
                stream_count,
                options.repeat,
                options.block_size,
                options.iters));
        }

        double pageable_ms = results[0].ms;
        double pinned_ms = results[1].ms;
        for (Result& result : results) {
            result.speedup_vs_pageable = result.ms > 0.0 ? pageable_ms / result.ms : 0.0;
            result.speedup_vs_pinned = result.ms > 0.0 ? pinned_ms / result.ms : 0.0;
        }

        std::cout
            << "method,size,total_mb,chunks,streams,chunk_elems,iters,repeat,total_ms,"
            << "effective_transfer_gb_s,speedup_vs_pageable_seq,speedup_vs_pinned_seq,"
            << "max_abs_error,async_engine_count,concurrent_kernels\n";

        std::size_t chunk_elems = options.size / static_cast<std::size_t>(options.chunks);
        double total_mb = static_cast<double>(options.size) * sizeof(float) / (1024.0 * 1024.0);
        for (const Result& result : results) {
            std::cout << result.method << ","
                      << options.size << ","
                      << std::fixed << std::setprecision(6)
                      << total_mb << ","
                      << options.chunks << ","
                      << result.streams << ","
                      << chunk_elems << ","
                      << options.iters << ","
                      << options.repeat << ","
                      << result.ms << ","
                      << result.transfer_gb_s << ","
                      << result.speedup_vs_pageable << ","
                      << result.speedup_vs_pinned << ","
                      << result.max_abs_error << ","
                      << props.asyncEngineCount << ","
                      << props.concurrentKernels << "\n";
        }

        CUDA_CHECK(cudaFreeHost(pinned_input));
        CUDA_CHECK(cudaFreeHost(pinned_output));
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

