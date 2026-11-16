// CUDA error checking, RAII device/pinned buffers, event timing and device queries.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include "vs/common.h"

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s -> %s\n", __FILE__, __LINE__,    \
                         #call, cudaGetErrorString(err_));                                 \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

// Call after every kernel launch. Build with -DVS_SYNC_CHECK to also synchronize, which
// attributes asynchronous execution errors to the launch that caused them.
#ifdef VS_SYNC_CHECK
#define CUDA_CHECK_LAUNCH()                                                                \
    do {                                                                                   \
        CUDA_CHECK(cudaGetLastError());                                                    \
        CUDA_CHECK(cudaDeviceSynchronize());                                               \
    } while (0)
#else
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#endif

namespace vs {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { resize(n); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            std::swap(ptr_, o.ptr_);
            std::swap(n_, o.n_);
        }
        return *this;
    }

    // Reallocates (contents are not preserved) unless the size already matches.
    void resize(size_t n) {
        if (n == n_) return;
        release();
        if (n) CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        n_ = n;
    }
    // Grow-only resize: keeps steady-state calls free of cudaMalloc/cudaFree.
    void ensure(size_t n) {
        if (n > n_) resize(n);
    }
    void release() {
        if (ptr_) CUDA_CHECK(cudaFree(ptr_));
        ptr_ = nullptr;
        n_ = 0;
    }
    void upload(const T* host, size_t n, cudaStream_t s = 0) {
        ensure(n);
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, s));
    }
    void download(T* host, size_t n, cudaStream_t s = 0) const {
        VS_REQUIRE(n <= n_, "download larger than buffer");
        CUDA_CHECK(cudaMemcpyAsync(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, s));
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return n_; }

private:
    T* ptr_ = nullptr;
    size_t n_ = 0;
};

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t n) { ensure(n); }
    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void ensure(size_t n) {
        if (n <= n_) return;
        release();
        CUDA_CHECK(cudaMallocHost(&ptr_, n * sizeof(T)));
        n_ = n;
    }
    void release() {
        if (ptr_) CUDA_CHECK(cudaFreeHost(ptr_));
        ptr_ = nullptr;
        n_ = 0;
    }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t size() const { return n_; }

private:
    T* ptr_ = nullptr;
    size_t n_ = 0;
};

// Times a region of a stream with CUDA events.
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
    // Records the stop event, waits for it and returns the elapsed time in milliseconds.
    float stop(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, s));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

struct DeviceInfo {
    int device = 0;
    std::string name;
    int cc_major = 0, cc_minor = 0;
    int sm_count = 0;
    int max_threads_per_sm = 0;
    int regs_per_sm = 0;
    size_t smem_per_sm = 0;
    size_t total_mem = 0;
    double sm_clock_mhz = 0;
    double mem_clock_mhz = 0;
    int bus_width_bits = 0;
    double peak_bw_gbs = 0;       // 2 (DDR) * memory clock * bus width
    double peak_fp32_gflops = 0;  // 2 (FMA) * cores * SM clock
    int runtime_version = 0, driver_version = 0;

    std::string cc() const { return std::to_string(cc_major) + "." + std::to_string(cc_minor); }
    std::string runtime_str() const { return version_str(runtime_version); }
    std::string driver_str() const { return version_str(driver_version); }
    static std::string version_str(int v) {
        return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
    }
};

DeviceInfo query_device_info(int device = 0);
void print_device_info(const DeviceInfo& info);

}  // namespace vs
