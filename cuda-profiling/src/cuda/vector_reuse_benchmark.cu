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

__global__ void accumulateAddKernel(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] += a[idx] + b[idx];
    }
}

struct Options {
    std::vector<std::size_t> sizes{100000, 1000000, 10000000};
    std::vector<int> ops{1, 10, 100};
    int repeat = 3;
    int block_size = 256;
};

struct Timing {
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

static std::vector<int> parseOps(const std::string& value) {
    std::vector<int> ops;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        int parsed = std::stoi(item);
        if (parsed <= 0) {
            throw std::invalid_argument("Operation counts must be positive");
        }
        ops.push_back(parsed);
    }

    if (ops.empty()) {
        throw std::invalid_argument("--ops must contain at least one positive integer");
    }

    return ops;
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
        } else if (arg == "--ops") {
            options.ops = parseOps(requireValue("--ops"));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(requireValue("--repeat"));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(requireValue("--block-size"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: vector_reuse_benchmark [--sizes 100000,1000000] "
                << "[--ops 1,10,100] [--repeat 3] [--block-size 256]\n";
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

static void fillInputs(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>((i % 1024) * 0.25);
        b[i] = static_cast<float>((i % 2048) * 0.125);
        c[i] = 0.0f;
    }
}

static double runCpuAccum(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int ops) {
    using clock = std::chrono::steady_clock;
    std::fill(c.begin(), c.end(), 0.0f);

    auto start = clock::now();
    for (int op = 0; op < ops; ++op) {
        for (std::size_t i = 0; i < c.size(); ++i) {
            c[i] += a[i] + b[i];
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

struct Events {
    cudaEvent_t total_start;
    cudaEvent_t total_stop;
    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;

    Events() {
        CUDA_CHECK(cudaEventCreate(&total_start));
        CUDA_CHECK(cudaEventCreate(&total_stop));
        CUDA_CHECK(cudaEventCreate(&h2d_start));
        CUDA_CHECK(cudaEventCreate(&h2d_stop));
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));
        CUDA_CHECK(cudaEventCreate(&d2h_start));
        CUDA_CHECK(cudaEventCreate(&d2h_stop));
    }

    ~Events() {
        cudaEventDestroy(total_start);
        cudaEventDestroy(total_stop);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_stop);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_stop);
    }
};

static Timing runGpuRoundTrip(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int ops,
    int block_size) {
    const std::size_t n = a.size();
    const std::size_t bytes = n * sizeof(float);
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    Events events;
    Timing timing;
    std::fill(c.begin(), c.end(), 0.0f);

    CUDA_CHECK(cudaEventRecord(events.total_start));
    for (int op = 0; op < ops; ++op) {
        CUDA_CHECK(cudaEventRecord(events.h2d_start));
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c, c.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(events.h2d_stop));
        CUDA_CHECK(cudaEventSynchronize(events.h2d_stop));
        timing.h2d_ms += elapsedMs(events.h2d_start, events.h2d_stop);

        CUDA_CHECK(cudaEventRecord(events.kernel_start));
        accumulateAddKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(events.kernel_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(events.kernel_stop));
        timing.kernel_ms += elapsedMs(events.kernel_start, events.kernel_stop);

        CUDA_CHECK(cudaEventRecord(events.d2h_start));
        CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(events.d2h_stop));
        CUDA_CHECK(cudaEventSynchronize(events.d2h_stop));
        timing.d2h_ms += elapsedMs(events.d2h_start, events.d2h_stop);
    }
    CUDA_CHECK(cudaEventRecord(events.total_stop));
    CUDA_CHECK(cudaEventSynchronize(events.total_stop));
    timing.total_ms = elapsedMs(events.total_start, events.total_stop);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return timing;
}

static Timing runGpuReuse(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int ops,
    int block_size) {
    const std::size_t n = a.size();
    const std::size_t bytes = n * sizeof(float);
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    Events events;
    Timing timing;
    std::fill(c.begin(), c.end(), 0.0f);

    CUDA_CHECK(cudaEventRecord(events.total_start));

    CUDA_CHECK(cudaEventRecord(events.h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(events.h2d_stop));

    CUDA_CHECK(cudaEventRecord(events.kernel_start));
    for (int op = 0; op < ops; ++op) {
        accumulateAddKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaEventRecord(events.kernel_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(events.d2h_start));
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(events.d2h_stop));

    CUDA_CHECK(cudaEventRecord(events.total_stop));
    CUDA_CHECK(cudaEventSynchronize(events.total_stop));

    timing.h2d_ms = elapsedMs(events.h2d_start, events.h2d_stop);
    timing.kernel_ms = elapsedMs(events.kernel_start, events.kernel_stop);
    timing.d2h_ms = elapsedMs(events.d2h_start, events.d2h_stop);
    timing.total_ms = elapsedMs(events.total_start, events.total_stop);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return timing;
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
            << "size,ops,bytes,cpu_ms,roundtrip_h2d_ms,roundtrip_kernel_ms,"
            << "roundtrip_d2h_ms,roundtrip_total_ms,reuse_h2d_ms,reuse_kernel_ms,"
            << "reuse_d2h_ms,reuse_total_ms,roundtrip_speedup_vs_cpu,"
            << "reuse_speedup_vs_cpu,reuse_speedup_vs_roundtrip,max_abs_error\n";

        for (std::size_t n : options.sizes) {
            std::vector<float> a(n);
            std::vector<float> b(n);
            std::vector<float> cpu_out(n);
            std::vector<float> roundtrip_out(n);
            std::vector<float> reuse_out(n);
            fillInputs(a, b, cpu_out);

            for (int ops : options.ops) {
                double cpu_total = 0.0;
                Timing roundtrip_total;
                Timing reuse_total;
                double error = 0.0;

                for (int r = 0; r < options.repeat; ++r) {
                    double cpu_ms = runCpuAccum(a, b, cpu_out, ops);
                    Timing roundtrip = runGpuRoundTrip(a, b, roundtrip_out, ops, options.block_size);
                    Timing reuse = runGpuReuse(a, b, reuse_out, ops, options.block_size);

                    cpu_total += cpu_ms;
                    roundtrip_total.h2d_ms += roundtrip.h2d_ms;
                    roundtrip_total.kernel_ms += roundtrip.kernel_ms;
                    roundtrip_total.d2h_ms += roundtrip.d2h_ms;
                    roundtrip_total.total_ms += roundtrip.total_ms;
                    reuse_total.h2d_ms += reuse.h2d_ms;
                    reuse_total.kernel_ms += reuse.kernel_ms;
                    reuse_total.d2h_ms += reuse.d2h_ms;
                    reuse_total.total_ms += reuse.total_ms;
                    error = std::max(error, maxAbsError(cpu_out, reuse_out));
                    error = std::max(error, maxAbsError(cpu_out, roundtrip_out));
                }

                const float inv_repeat = 1.0f / static_cast<float>(options.repeat);
                double cpu_ms = cpu_total / options.repeat;
                roundtrip_total.h2d_ms *= inv_repeat;
                roundtrip_total.kernel_ms *= inv_repeat;
                roundtrip_total.d2h_ms *= inv_repeat;
                roundtrip_total.total_ms *= inv_repeat;
                reuse_total.h2d_ms *= inv_repeat;
                reuse_total.kernel_ms *= inv_repeat;
                reuse_total.d2h_ms *= inv_repeat;
                reuse_total.total_ms *= inv_repeat;

                double roundtrip_speedup = roundtrip_total.total_ms > 0.0f
                    ? cpu_ms / roundtrip_total.total_ms
                    : 0.0;
                double reuse_speedup = reuse_total.total_ms > 0.0f
                    ? cpu_ms / reuse_total.total_ms
                    : 0.0;
                double reuse_vs_roundtrip = reuse_total.total_ms > 0.0f
                    ? roundtrip_total.total_ms / reuse_total.total_ms
                    : 0.0;

                std::cout << n << ","
                          << ops << ","
                          << n * sizeof(float) << ","
                          << std::fixed << std::setprecision(6)
                          << cpu_ms << ","
                          << roundtrip_total.h2d_ms << ","
                          << roundtrip_total.kernel_ms << ","
                          << roundtrip_total.d2h_ms << ","
                          << roundtrip_total.total_ms << ","
                          << reuse_total.h2d_ms << ","
                          << reuse_total.kernel_ms << ","
                          << reuse_total.d2h_ms << ","
                          << reuse_total.total_ms << ","
                          << roundtrip_speedup << ","
                          << reuse_speedup << ","
                          << reuse_vs_roundtrip << ","
                          << error << "\n";
            }
        }

        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

