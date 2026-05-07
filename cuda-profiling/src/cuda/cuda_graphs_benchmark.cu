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
    std::vector<std::size_t> sizes{1024, 4096, 16384, 65536, 262144, 1048576};
    int repeat = 1000;
    int block_size = 256;
    int iters = 1;
};

struct Result {
    std::string workload;
    std::string method;
    std::size_t size = 0;
    double total_wall_ms = 0.0;
    double cpu_enqueue_ms = 0.0;
    double gpu_elapsed_ms = 0.0;
    double per_iter_wall_us = 0.0;
    double per_iter_enqueue_us = 0.0;
    double effective_transfer_gb_s = 0.0;
    double speedup_wall_vs_normal = 0.0;
    double speedup_enqueue_vs_normal = 0.0;
    double max_abs_error = 0.0;
};

static std::vector<std::size_t> parseSizeList(const std::string& value) {
    std::vector<std::size_t> out;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        unsigned long long parsed = std::strtoull(item.c_str(), nullptr, 10);
        if (parsed == 0) {
            throw std::invalid_argument("--sizes must contain positive integers");
        }
        out.push_back(static_cast<std::size_t>(parsed));
    }

    if (out.empty()) {
        throw std::invalid_argument("--sizes must contain at least one value");
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

        if (arg == "--sizes") {
            options.sizes = parseSizeList(requireValue("--sizes"));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(requireValue("--repeat"));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(requireValue("--block-size"));
        } else if (arg == "--iters") {
            options.iters = std::stoi(requireValue("--iters"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: cuda_graphs_benchmark [--sizes 1024,4096,16384,65536,262144,1048576]\n"
                << "                             [--repeat 1000] [--block-size 256] [--iters 1]\n";
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

static double effectiveTransferGbPerSec(std::size_t n, int repeat, double total_wall_ms, bool copy_each_iter) {
    if (total_wall_ms <= 0.0) {
        return 0.0;
    }
    if (!copy_each_iter) {
        return 0.0;
    }
    double transferred_bytes = static_cast<double>(n) * sizeof(float) * 2.0 * repeat;
    return transferred_bytes / (total_wall_ms * 1.0e6);
}

static Result runNormal(float* h_input,
                        float* h_output,
                        float* d_input,
                        float* d_output,
                        std::size_t n,
                        int repeat,
                        int block_size,
                        int iters,
                        cudaStream_t stream,
                        bool copy_each_iter) {
    int grid = static_cast<int>((n + static_cast<std::size_t>(block_size) - 1) /
                                static_cast<std::size_t>(block_size));
    std::size_t bytes = n * sizeof(float);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, bytes, stream));
    if (!copy_each_iter) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start, stream));

    auto enqueue_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        if (copy_each_iter) {
            CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream));
        }
        transformKernel<<<grid, block_size, 0, stream>>>(d_input, d_output, n, iters);
        CUDA_CHECK(cudaGetLastError());
        if (copy_each_iter) {
            CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream));
        }
    }
    if (!copy_each_iter) {
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream));
    }
    auto enqueue_stop = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    auto wall_stop = std::chrono::steady_clock::now();

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    Result result;
    result.workload = copy_each_iter ? "copy_kernel_copy" : "kernel_only";
    result.method = "normal_launch";
    result.size = n;
    result.total_wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    result.cpu_enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_stop - enqueue_start).count();
    result.gpu_elapsed_ms = static_cast<double>(gpu_ms);
    result.per_iter_wall_us = result.total_wall_ms * 1000.0 / repeat;
    result.per_iter_enqueue_us = result.cpu_enqueue_ms * 1000.0 / repeat;
    result.effective_transfer_gb_s = effectiveTransferGbPerSec(n, repeat, result.total_wall_ms, copy_each_iter);
    result.max_abs_error = sampleMaxAbsError(h_input, h_output, n, iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return result;
}

static Result runGraph(float* h_input,
                       float* h_output,
                       float* d_input,
                       float* d_output,
                       std::size_t n,
                   int repeat,
                   int block_size,
                   int iters,
                   cudaStream_t stream,
                   bool copy_each_iter) {
    int grid = static_cast<int>((n + static_cast<std::size_t>(block_size) - 1) /
                                static_cast<std::size_t>(block_size));
    std::size_t bytes = n * sizeof(float);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, bytes, stream));
    if (!copy_each_iter) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (copy_each_iter) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream));
    }
    transformKernel<<<grid, block_size, 0, stream>>>(d_input, d_output, n, iters);
    CUDA_CHECK(cudaGetLastError());
    if (copy_each_iter) {
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start, stream));

    auto enqueue_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    if (!copy_each_iter) {
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream));
    }
    auto enqueue_stop = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    auto wall_stop = std::chrono::steady_clock::now();

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    Result result;
    result.workload = copy_each_iter ? "copy_kernel_copy" : "kernel_only";
    result.method = "cuda_graph_replay";
    result.size = n;
    result.total_wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    result.cpu_enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_stop - enqueue_start).count();
    result.gpu_elapsed_ms = static_cast<double>(gpu_ms);
    result.per_iter_wall_us = result.total_wall_ms * 1000.0 / repeat;
    result.per_iter_enqueue_us = result.cpu_enqueue_ms * 1000.0 / repeat;
    result.effective_transfer_gb_s = effectiveTransferGbPerSec(n, repeat, result.total_wall_ms, copy_each_iter);
    result.max_abs_error = sampleMaxAbsError(h_input, h_output, n, iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));

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
        std::cerr << "Repeat: " << options.repeat
                  << ", block_size: " << options.block_size
                  << ", iters: " << options.iters << "\n";

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::cout
            << "workload,method,size,total_mb,iters,repeat,block_size,total_wall_ms,cpu_enqueue_ms,"
            << "gpu_elapsed_ms,per_iter_wall_us,per_iter_enqueue_us,effective_transfer_gb_s,"
            << "speedup_wall_vs_normal,speedup_enqueue_vs_normal,max_abs_error\n";

        for (std::size_t n : options.sizes) {
            std::size_t bytes = n * sizeof(float);
            float* h_input = nullptr;
            float* h_output = nullptr;
            float* d_input = nullptr;
            float* d_output = nullptr;

            CUDA_CHECK(cudaMallocHost(&h_input, bytes));
            CUDA_CHECK(cudaMallocHost(&h_output, bytes));
            CUDA_CHECK(cudaMalloc(&d_input, bytes));
            CUDA_CHECK(cudaMalloc(&d_output, bytes));

            fillInput(h_input, n);

            std::vector<Result> results;
            for (bool copy_each_iter : {false, true}) {
                Result normal = runNormal(h_input, h_output, d_input, d_output, n, options.repeat,
                                          options.block_size, options.iters, stream, copy_each_iter);
                Result graph = runGraph(h_input, h_output, d_input, d_output, n, options.repeat,
                                        options.block_size, options.iters, stream, copy_each_iter);

                normal.speedup_wall_vs_normal = 1.0;
                normal.speedup_enqueue_vs_normal = 1.0;
                graph.speedup_wall_vs_normal = normal.total_wall_ms / graph.total_wall_ms;
                graph.speedup_enqueue_vs_normal = normal.cpu_enqueue_ms / graph.cpu_enqueue_ms;
                results.push_back(normal);
                results.push_back(graph);
            }

            double total_mb = static_cast<double>(n) * sizeof(float) / (1024.0 * 1024.0);
            for (const Result& result : results) {
                std::cout << result.workload << ","
                          << result.method << ","
                          << result.size << ","
                          << std::fixed << std::setprecision(6)
                          << total_mb << ","
                          << options.iters << ","
                          << options.repeat << ","
                          << options.block_size << ","
                          << result.total_wall_ms << ","
                          << result.cpu_enqueue_ms << ","
                          << result.gpu_elapsed_ms << ","
                          << result.per_iter_wall_us << ","
                          << result.per_iter_enqueue_us << ","
                          << result.effective_transfer_gb_s << ","
                          << result.speedup_wall_vs_normal << ","
                          << result.speedup_enqueue_vs_normal << ","
                          << result.max_abs_error << "\n";
            }

            CUDA_CHECK(cudaFree(d_output));
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFreeHost(h_output));
            CUDA_CHECK(cudaFreeHost(h_input));
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
