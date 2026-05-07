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

struct Options {
    std::vector<std::size_t> sizes{100000, 1000000, 10000000, 30000000};
    int repeat = 10;
};

struct CopyTiming {
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;
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
            std::cout << "Usage: pinned_memory_benchmark [--sizes 100000,1000000] [--repeat 10]\n";
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

static void fillInput(float* ptr, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = static_cast<float>((i % 4096) * 0.03125);
    }
}

static double timeCopy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind, int repeat) {
    using clock = std::chrono::steady_clock;
    double total_ms = 0.0;

    for (int r = 0; r < repeat; ++r) {
        auto start = clock::now();
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
        auto end = clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total_ms / repeat;
}

static double throughputGbps(std::size_t bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / (ms * 1.0e6);
}

static double maxAbsError(const float* expected, const float* actual, std::size_t n) {
    double max_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
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
        std::cerr << "Repeat: " << options.repeat << "\n";

        std::cout
            << "size,bytes,pageable_h2d_ms,pageable_h2d_gbps,pageable_d2h_ms,"
            << "pageable_d2h_gbps,pinned_h2d_ms,pinned_h2d_gbps,pinned_d2h_ms,"
            << "pinned_d2h_gbps,h2d_speedup,d2h_speedup,max_abs_error\n";

        for (std::size_t n : options.sizes) {
            const std::size_t bytes = n * sizeof(float);

            std::vector<float> pageable_src(n);
            std::vector<float> pageable_dst(n, 0.0f);
            fillInput(pageable_src.data(), n);

            float* pinned_src = nullptr;
            float* pinned_dst = nullptr;
            float* d_buffer = nullptr;

            CUDA_CHECK(cudaMallocHost(&pinned_src, bytes));
            CUDA_CHECK(cudaMallocHost(&pinned_dst, bytes));
            CUDA_CHECK(cudaMalloc(&d_buffer, bytes));

            fillInput(pinned_src, n);
            std::fill(pinned_dst, pinned_dst + n, 0.0f);

            CUDA_CHECK(cudaMemcpy(d_buffer, pageable_src.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(pageable_dst.data(), d_buffer, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_buffer, pinned_src, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(pinned_dst, d_buffer, bytes, cudaMemcpyDeviceToHost));

            double pageable_h2d_ms = timeCopy(d_buffer, pageable_src.data(), bytes, cudaMemcpyHostToDevice, options.repeat);
            double pageable_d2h_ms = timeCopy(pageable_dst.data(), d_buffer, bytes, cudaMemcpyDeviceToHost, options.repeat);

            double pinned_h2d_ms = timeCopy(d_buffer, pinned_src, bytes, cudaMemcpyHostToDevice, options.repeat);
            double pinned_d2h_ms = timeCopy(pinned_dst, d_buffer, bytes, cudaMemcpyDeviceToHost, options.repeat);

            double error = std::max(
                maxAbsError(pageable_src.data(), pageable_dst.data(), n),
                maxAbsError(pinned_src, pinned_dst, n));

            double h2d_speedup = pinned_h2d_ms > 0.0 ? pageable_h2d_ms / pinned_h2d_ms : 0.0;
            double d2h_speedup = pinned_d2h_ms > 0.0 ? pageable_d2h_ms / pinned_d2h_ms : 0.0;

            std::cout << n << ","
                      << bytes << ","
                      << std::fixed << std::setprecision(6)
                      << pageable_h2d_ms << ","
                      << throughputGbps(bytes, pageable_h2d_ms) << ","
                      << pageable_d2h_ms << ","
                      << throughputGbps(bytes, pageable_d2h_ms) << ","
                      << pinned_h2d_ms << ","
                      << throughputGbps(bytes, pinned_h2d_ms) << ","
                      << pinned_d2h_ms << ","
                      << throughputGbps(bytes, pinned_d2h_ms) << ","
                      << h2d_speedup << ","
                      << d2h_speedup << ","
                      << error << "\n";

            CUDA_CHECK(cudaFree(d_buffer));
            CUDA_CHECK(cudaFreeHost(pinned_src));
            CUDA_CHECK(cudaFreeHost(pinned_dst));
        }

        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}

