// Small device-side helpers shared by the kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vs {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFullMask, v, o);
    return v;
}

__device__ __forceinline__ float dot4(const float4 a, const float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Maps a float to an unsigned key whose unsigned order matches the float order.
__device__ __forceinline__ uint32_t float_to_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float key_to_float(uint32_t k) {
    const uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    return __uint_as_float(u);
}

}  // namespace vs
