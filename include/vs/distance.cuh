// Batched inner-product kernels: S[q * ldS + v] = <Q_q, X_v>.
//
// All vectors are L2-normalized up front, so the inner product is the cosine similarity.
// Q (nq x d) and X (nv x d) are row-major device arrays, d % 4 == 0, 16-byte aligned rows.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace vs {

enum class DistVariant {
    Naive,     // one thread per (query, vector); strided, uncoalesced global loads
    Warp,      // one warp per vector, float4 coalesced loads, a group of queries cached in smem
    Tiled,     // 64x64 shared-memory tiled GEMM with 4x4 register blocking (fixed 256 threads)
    Cublas,    // cublasSgemm reference (FP32, no TF32)
    WarpFp16,  // Warp with X stored as fp16 (half the bytes), fp32 accumulation
};

DistVariant parse_dist_variant(const std::string& s);
const char* dist_variant_name(DistVariant v);
// Whether --tpb changes the launch (Tiled and Cublas have a fixed block shape).
bool variant_uses_tpb(DistVariant v);

void launch_inner_product(DistVariant variant, const float* dQ, int nq, const float* dX,
                          int64_t nv, int d, float* dS, int64_t ldS, int threads_per_block,
                          cudaStream_t stream);

// WarpFp16 path: X is fp16, Q stays fp32. Requires d % 8 == 0.
void launch_inner_product_fp16(const float* dQ, int nq, const __half* dX, int64_t nv, int d,
                               float* dS, int64_t ldS, int threads_per_block, cudaStream_t stream);

void convert_to_half(const float* d_in, __half* d_out, int64_t count, cudaStream_t stream);

struct KernelAttrs {
    int regs_per_thread = 0;
    size_t static_smem = 0;
    size_t dynamic_smem = 0;
    int threads_per_block = 0;
    int active_blocks_per_sm = 0;
    double occupancy = 0;  // active warps / max warps per SM (theoretical)
};
// Attributes of the kernel launch_inner_product would use for this configuration
// (not available for Cublas).
KernelAttrs inner_product_kernel_attrs(DistVariant variant, int nq, int d, int threads_per_block);

}  // namespace vs
