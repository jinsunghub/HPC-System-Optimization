#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
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

__global__ void sharedBankKernel(float* output, int effective_stride, int inner_iters) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int segment = 33 * 32;
    int index = warp * segment + lane * effective_stride;
    int global = blockIdx.x * blockDim.x + tid;

    volatile float* vsmem = smem;
    float value = static_cast<float>((global & 255) + 1) * 0.001f;
    vsmem[index] = value;
    __syncthreads();

    float acc = value;
#pragma unroll 1
    for (int i = 0; i < inner_iters; ++i) {
        float x = vsmem[index];
        acc = fmaf(x, 0.000001f, acc);
        vsmem[index] = x + 0.000001f;
    }

    output[global] = acc;
}

struct Options {
    std::vector<int> strides{1, 2, 4, 8, 16, 32};
    int blocks = 256;
    int block_size = 256;
    int repeat = 20;
    int inner_iters = 2048;
};

struct Result {
    std::string method;
    int requested_stride = 1;
    int effective_stride = 1;
    int conflict_degree = 1;
    double total_ms = 0.0;
    double shared_gaccess_s = 0.0;
    double slowdown_vs_stride1 = 1.0;
    double speedup_vs_conflict = 1.0;
    double checksum = 0.0;
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

        if (arg == "--strides") {
            options.strides = parseIntList(requireValue("--strides"), "--strides");
        } else if (arg == "--blocks") {
            options.blocks = std::stoi(requireValue("--blocks"));
        } else if (arg == "--block-size") {
            options.block_size = std::stoi(requireValue("--block-size"));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(requireValue("--repeat"));
        } else if (arg == "--inner-iters") {
            options.inner_iters = std::stoi(requireValue("--inner-iters"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: shared_bank_conflict_benchmark [--strides 1,2,4,8,16,32]\n"
                << "                                      [--blocks 256] [--block-size 256]\n"
                << "                                      [--repeat 20] [--inner-iters 2048]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.blocks <= 0) {
        throw std::invalid_argument("--blocks must be positive");
    }
    if (options.block_size <= 0 || options.block_size > 1024 || options.block_size % 32 != 0) {
        throw std::invalid_argument("--block-size must be a positive multiple of 32 up to 1024");
    }
    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
    }
    if (options.inner_iters <= 0) {
        throw std::invalid_argument("--inner-iters must be positive");
    }
    for (int stride : options.strides) {
        if (stride > 32) {
            throw std::invalid_argument("This benchmark expects strides in the range 1..32");
        }
    }

    return options;
}

static int conflictDegree(int effective_stride) {
    return std::gcd(effective_stride, 32);
}

static double checksumOutput(const std::vector<float>& values) {
    double sum = 0.0;
    std::size_t step = std::max<std::size_t>(1, values.size() / 4096);
    for (std::size_t i = 0; i < values.size(); i += step) {
        sum += static_cast<double>(values[i]);
    }
    return sum;
}

static Result runCase(const Options& options,
                      float* d_output,
                      std::vector<float>& h_output,
                      int requested_stride,
                      int effective_stride,
                      const std::string& method) {
    int warps_per_block = options.block_size / 32;
    std::size_t shared_bytes = static_cast<std::size_t>(warps_per_block) * 33 * 32 * sizeof(float);
    std::size_t output_bytes = h_output.size() * sizeof(float);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < options.repeat; ++i) {
        sharedBankKernel<<<options.blocks, options.block_size, shared_bytes>>>(
            d_output, effective_stride, options.inner_iters);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    double per_launch_ms = static_cast<double>(elapsed_ms) / options.repeat;
    double accesses = static_cast<double>(options.blocks) * options.block_size *
                      options.inner_iters * 2.0;

    Result result;
    result.method = method;
    result.requested_stride = requested_stride;
    result.effective_stride = effective_stride;
    result.conflict_degree = conflictDegree(effective_stride);
    result.total_ms = per_launch_ms;
    result.shared_gaccess_s = accesses / (per_launch_ms * 1.0e6);
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
        std::cerr << "blocks: " << options.blocks
                  << ", block_size: " << options.block_size
                  << ", repeat: " << options.repeat
                  << ", inner_iters: " << options.inner_iters << "\n";

        std::size_t output_count = static_cast<std::size_t>(options.blocks) * options.block_size;
        std::vector<float> h_output(output_count, 0.0f);
        float* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_output, output_count * sizeof(float)));

        std::vector<Result> results;
        for (int stride : options.strides) {
            if (stride == 1) {
                results.push_back(runCase(options, d_output, h_output, stride, 1, "conflict_free"));
                continue;
            }

            results.push_back(runCase(options, d_output, h_output, stride, stride, "conflict"));
            results.push_back(runCase(options, d_output, h_output, stride, stride + 1, "padded"));
        }

        auto baseline_it = std::find_if(results.begin(), results.end(), [](const Result& result) {
            return result.method == "conflict_free" && result.requested_stride == 1;
        });
        if (baseline_it == results.end()) {
            throw std::runtime_error("Missing stride=1 baseline");
        }
        double baseline_ms = baseline_it->total_ms;

        for (Result& result : results) {
            result.slowdown_vs_stride1 = result.total_ms / baseline_ms;
            if (result.method == "padded") {
                auto conflict_it = std::find_if(results.begin(), results.end(), [&](const Result& other) {
                    return other.method == "conflict" &&
                           other.requested_stride == result.requested_stride;
                });
                if (conflict_it != results.end()) {
                    result.speedup_vs_conflict = conflict_it->total_ms / result.total_ms;
                }
            }
        }

        std::cout
            << "method,requested_stride,effective_stride,conflict_degree,blocks,block_size,"
            << "inner_iters,repeat,total_ms,shared_gaccess_s,slowdown_vs_stride1,"
            << "speedup_vs_conflict,checksum\n";

        for (const Result& result : results) {
            std::cout << result.method << ","
                      << result.requested_stride << ","
                      << result.effective_stride << ","
                      << result.conflict_degree << ","
                      << options.blocks << ","
                      << options.block_size << ","
                      << options.inner_iters << ","
                      << options.repeat << ","
                      << std::fixed << std::setprecision(6)
                      << result.total_ms << ","
                      << result.shared_gaccess_s << ","
                      << result.slowdown_vs_stride1 << ","
                      << result.speedup_vs_conflict << ","
                      << result.checksum << "\n";
        }

        CUDA_CHECK(cudaFree(d_output));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
