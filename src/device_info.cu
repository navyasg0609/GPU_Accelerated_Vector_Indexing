#include <cstdio>

#include "vs/cuda_utils.cuh"

namespace vs {

namespace {

int fp32_cores_per_sm(int major, int minor) {
    switch (major) {
        case 6: return minor == 0 ? 64 : 128;
        case 7: return 64;
        case 8: return minor == 0 ? 64 : 128;
        default: return 128;
    }
}

int attr(cudaDeviceAttr a, int device) {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, a, device));
    return v;
}

}  // namespace

DeviceInfo query_device_info(int device) {
    DeviceInfo info;
    info.device = device;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info.regs_per_sm = prop.regsPerMultiprocessor;
    info.smem_per_sm = prop.sharedMemPerMultiprocessor;
    info.total_mem = prop.totalGlobalMem;
    // Clock rates are read through attributes (kHz); the cudaDeviceProp fields are deprecated.
    info.sm_clock_mhz = attr(cudaDevAttrClockRate, device) / 1000.0;
    info.mem_clock_mhz = attr(cudaDevAttrMemoryClockRate, device) / 1000.0;
    info.bus_width_bits = attr(cudaDevAttrGlobalMemoryBusWidth, device);
    info.peak_bw_gbs = 2.0 * info.mem_clock_mhz * 1e6 * (info.bus_width_bits / 8.0) / 1e9;
    info.peak_fp32_gflops =
        2.0 * fp32_cores_per_sm(info.cc_major, info.cc_minor) * info.sm_count * info.sm_clock_mhz / 1e3;
    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    return info;
}

void print_device_info(const DeviceInfo& i) {
    std::printf("GPU %d: %s (compute capability %s, %d SMs, %.1f GiB)\n", i.device, i.name.c_str(),
                i.cc().c_str(), i.sm_count, i.total_mem / double(1 << 30));
    std::printf("  CUDA runtime %s, driver API %s\n", i.runtime_str().c_str(),
                i.driver_str().c_str());
    std::printf("  memory: %d-bit bus @ %.0f MHz -> theoretical peak %.1f GB/s\n",
                i.bus_width_bits, i.mem_clock_mhz, i.peak_bw_gbs);
    std::printf("  SM clock %.0f MHz -> theoretical FP32 peak %.0f GFLOP/s\n", i.sm_clock_mhz,
                i.peak_fp32_gflops);
}

}  // namespace vs
