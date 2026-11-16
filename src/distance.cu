#include "vs/distance.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <map>
#include <tuple>

#include "vs/cuda_utils.cuh"
#include "vs/device_utils.cuh"

#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        const cublasStatus_t st_ = (call);                                                 \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n", (int)st_, __FILE__,     \
                         __LINE__, #call);                                                 \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

namespace vs {

namespace {

constexpr int kTileM = 64;   // queries per block
constexpr int kTileN = 64;   // vectors per block
constexpr int kTileK = 16;   // dimensions per smem stage
constexpr int kTiledThreads = 256;
constexpr int kMaxGridY = 65535;

// ---------------------------------------------------------------------------------------
// Variant 1: naive. Thread v reads X[v*d + j]: neighbouring threads are d*4 bytes apart, so
// each warp-wide load touches 32 different cache lines.
__global__ void ip_naive_kernel(const float* __restrict__ Q, int nq, const float* __restrict__ X,
                                int64_t nv, int d, float* __restrict__ S, int64_t ldS) {
    const int64_t v = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    const float* x = X + v * d;
    for (int q = blockIdx.y; q < nq; q += gridDim.y) {
        const float* qv = Q + int64_t(q) * d;
        float acc = 0.f;
        for (int j = 0; j < d; ++j) acc += qv[j] * x[j];
        S[int64_t(q) * ldS + v] = acc;
    }
}

// ---------------------------------------------------------------------------------------
// Variant 2: warp per vector. A warp reads one 1.5 KB row (d=384) as 96 float4 loads that are
// contiguous across lanes (fully coalesced, 128-bit). QG queries are staged in shared memory
// so each row loaded from DRAM is reused QG times.
template <int QG>
__global__ void ip_warp_kernel(const float* __restrict__ Q, int nq, const float* __restrict__ X,
                               int64_t nv, int d, float* __restrict__ S, int64_t ldS) {
    extern __shared__ float4 sQ[];  // QG * d/4
    const int d4 = d >> 2;
    const int q0 = blockIdx.y * QG;
    const int nqg = min(QG, nq - q0);
    for (int i = threadIdx.x; i < QG * d4; i += blockDim.x) {
        const int g = i / d4, j = i - g * d4;
        sQ[i] = g < nqg ? reinterpret_cast<const float4*>(Q + int64_t(q0 + g) * d)[j]
                        : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    for (int64_t v = int64_t(blockIdx.x) * wpb + warp; v < nv; v += int64_t(gridDim.x) * wpb) {
        const float4* x4 = reinterpret_cast<const float4*>(X + v * d);
        float acc[QG];
#pragma unroll
        for (int g = 0; g < QG; ++g) acc[g] = 0.f;
        for (int j = lane; j < d4; j += 32) {
            const float4 xv = __ldg(x4 + j);
#pragma unroll
            for (int g = 0; g < QG; ++g) acc[g] += dot4(xv, sQ[g * d4 + j]);
        }
#pragma unroll
        for (int g = 0; g < QG; ++g) acc[g] = warp_sum(acc[g]);
        if (lane == 0) {
#pragma unroll
            for (int g = 0; g < QG; ++g)
                if (g < nqg) S[int64_t(q0 + g) * ldS + v] = acc[g];
        }
    }
}

// Variant 5: same as ip_warp_kernel but X is fp16; one uint4 load brings 8 halves.
template <int QG>
__global__ void ip_warp_fp16_kernel(const float* __restrict__ Q, int nq,
                                    const __half* __restrict__ X, int64_t nv, int d,
                                    float* __restrict__ S, int64_t ldS) {
    extern __shared__ float4 sQ[];
    const int d4 = d >> 2, d8 = d >> 3;
    const int q0 = blockIdx.y * QG;
    const int nqg = min(QG, nq - q0);
    for (int i = threadIdx.x; i < QG * d4; i += blockDim.x) {
        const int g = i / d4, j = i - g * d4;
        sQ[i] = g < nqg ? reinterpret_cast<const float4*>(Q + int64_t(q0 + g) * d)[j]
                        : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    for (int64_t v = int64_t(blockIdx.x) * wpb + warp; v < nv; v += int64_t(gridDim.x) * wpb) {
        const uint4* x8 = reinterpret_cast<const uint4*>(X + v * d);
        float acc[QG];
#pragma unroll
        for (int g = 0; g < QG; ++g) acc[g] = 0.f;
        for (int j = lane; j < d8; j += 32) {
            const uint4 raw = __ldg(x8 + j);
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            const float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]);
            const float2 f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
            const float4 xa = make_float4(f0.x, f0.y, f1.x, f1.y);
            const float4 xb = make_float4(f2.x, f2.y, f3.x, f3.y);
#pragma unroll
            for (int g = 0; g < QG; ++g)
                acc[g] += dot4(xa, sQ[g * d4 + 2 * j]) + dot4(xb, sQ[g * d4 + 2 * j + 1]);
        }
#pragma unroll
        for (int g = 0; g < QG; ++g) acc[g] = warp_sum(acc[g]);
        if (lane == 0) {
#pragma unroll
            for (int g = 0; g < QG; ++g)
                if (g < nqg) S[int64_t(q0 + g) * ldS + v] = acc[g];
        }
    }
}

// ---------------------------------------------------------------------------------------
// Variant 3: shared-memory tiled GEMM, C = A * B^T with A = Q (M x K) and B = X (N x K).
// Each block computes a 64x64 tile of S; each thread a 4x4 sub-tile at stride 16 so that smem
// reads of B are conflict-free and reads of A are broadcasts. Tiles are stored k-major with one
// float of padding to limit bank conflicts on the transposing store.
__global__ __launch_bounds__(kTiledThreads) void ip_tiled_kernel(
    const float* __restrict__ A, int M, const float* __restrict__ B, int64_t N, int K,
    float* __restrict__ C, int64_t ldc) {
    __shared__ float As[kTileK][kTileM + 1];
    __shared__ float Bs[kTileK][kTileN + 1];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const int m0 = blockIdx.y * kTileM;
    const int64_t n0 = int64_t(blockIdx.x) * kTileN;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += kTileK) {
        // 64 rows x 16 dims per operand; consecutive threads read consecutive dims (64 B runs).
        for (int i = threadIdx.x; i < kTileM * kTileK; i += kTiledThreads) {
            const int r = i / kTileK, kk = i % kTileK;
            const int gk = k0 + kk;
            const int gm = m0 + r;
            const int64_t gn = n0 + r;
            As[kk][r] = (gm < M && gk < K) ? A[int64_t(gm) * K + gk] : 0.f;
            Bs[kk][r] = (gn < N && gk < K) ? B[gn * K + gk] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm = m0 + ty + 16 * i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t gn = n0 + tx + 16 * j;
            if (gn < N) C[int64_t(gm) * ldc + gn] = acc[i][j];
        }
    }
}

__global__ void to_half_kernel(const float* __restrict__ in, __half* __restrict__ out,
                               int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x)
        out[i] = __float2half_rn(in[i]);
}

// ---------------------------------------------------------------------------------------

cublasHandle_t cublas_handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t x;
        CUBLAS_CHECK(cublasCreate(&x));
        CUBLAS_CHECK(cublasSetMathMode(x, CUBLAS_DEFAULT_MATH));
        return x;
    }();
    return h;
}

int device_sm_count() {
    static int sms = [] {
        int dev = 0, v = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
        return v;
    }();
    return sms;
}

// Blocks per SM the kernel can keep resident, cached per (kernel, block size, smem).
int resident_blocks_per_sm(const void* kernel, int threads, size_t smem) {
    static std::map<std::tuple<const void*, int, size_t>, int> cache;
    const auto key = std::make_tuple(kernel, threads, smem);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, smem));
    cache[key] = std::max(1, per_sm);
    return cache[key];
}

void check_tpb(int tpb) {
    VS_REQUIRE(tpb >= 32 && tpb <= 1024 && tpb % 32 == 0,
               "threads per block must be a multiple of 32 in [32, 1024]");
}

template <int QG, typename Kernel, typename XT>
void launch_warp_family(Kernel kern, const float* dQ, int nq, const XT* dX, int64_t nv, int d,
                        float* dS, int64_t ldS, int tpb, cudaStream_t stream) {
    check_tpb(tpb);
    const int64_t groups = ceil_div(nq, QG);
    VS_REQUIRE(groups <= kMaxGridY, "too many queries for one launch of the warp kernel");
    const size_t smem = size_t(QG) * d * sizeof(float);
    VS_REQUIRE(smem <= 48 * 1024, "query group does not fit in 48 KB of shared memory");
    const int wpb = tpb / 32;
    // Enough blocks for ~4 waves in total; each block then loops over many vectors so the
    // query group staged in shared memory is amortised.
    const int64_t fill = int64_t(resident_blocks_per_sm((const void*)kern, tpb, smem)) *
                         device_sm_count() * 4;
    const int64_t want = ceil_div(nv, wpb);
    const int64_t gx = std::max<int64_t>(1, std::min(want, ceil_div(fill, groups)));
    kern<<<dim3((unsigned)gx, (unsigned)groups), tpb, smem, stream>>>(dQ, nq, dX, nv, d, dS, ldS);
    CUDA_CHECK_LAUNCH();
}

}  // namespace

DistVariant parse_dist_variant(const std::string& s) {
    if (s == "naive") return DistVariant::Naive;
    if (s == "warp") return DistVariant::Warp;
    if (s == "tiled") return DistVariant::Tiled;
    if (s == "cublas") return DistVariant::Cublas;
    if (s == "warp_fp16") return DistVariant::WarpFp16;
    std::fprintf(stderr, "Unknown kernel variant '%s' (naive|warp|tiled|cublas|warp_fp16)\n",
                 s.c_str());
    std::exit(EXIT_FAILURE);
}

const char* dist_variant_name(DistVariant v) {
    switch (v) {
        case DistVariant::Naive: return "naive";
        case DistVariant::Warp: return "warp";
        case DistVariant::Tiled: return "tiled";
        case DistVariant::Cublas: return "cublas";
        case DistVariant::WarpFp16: return "warp_fp16";
    }
    return "?";
}

bool variant_uses_tpb(DistVariant v) {
    return v == DistVariant::Naive || v == DistVariant::Warp || v == DistVariant::WarpFp16;
}

void launch_inner_product(DistVariant variant, const float* dQ, int nq, const float* dX,
                          int64_t nv, int d, float* dS, int64_t ldS, int tpb,
                          cudaStream_t stream) {
    VS_REQUIRE(d % 4 == 0, "dimension must be a multiple of 4");
    VS_REQUIRE(ldS >= nv, "ldS must be >= nv");
    if (nq <= 0 || nv <= 0) return;
    switch (variant) {
        case DistVariant::Naive: {
            check_tpb(tpb);
            const dim3 grid((unsigned)ceil_div(nv, tpb), (unsigned)std::min(nq, kMaxGridY));
            ip_naive_kernel<<<grid, tpb, 0, stream>>>(dQ, nq, dX, nv, d, dS, ldS);
            CUDA_CHECK_LAUNCH();
            break;
        }
        case DistVariant::Warp:
            if (nq == 1)
                launch_warp_family<1>(ip_warp_kernel<1>, dQ, nq, dX, nv, d, dS, ldS, tpb, stream);
            else if (nq <= 4)
                launch_warp_family<4>(ip_warp_kernel<4>, dQ, nq, dX, nv, d, dS, ldS, tpb, stream);
            else
                launch_warp_family<8>(ip_warp_kernel<8>, dQ, nq, dX, nv, d, dS, ldS, tpb, stream);
            break;
        case DistVariant::Tiled: {
            VS_REQUIRE(ceil_div(nq, kTileM) <= kMaxGridY, "too many queries for the tiled kernel");
            const dim3 grid((unsigned)ceil_div(nv, kTileN), (unsigned)ceil_div(nq, kTileM));
            ip_tiled_kernel<<<grid, kTiledThreads, 0, stream>>>(dQ, nq, dX, nv, d, dS, ldS);
            CUDA_CHECK_LAUNCH();
            break;
        }
        case DistVariant::Cublas: {
            // Row-major S (nq x nv) is column-major S^T (nv x nq) = X * Q^T.
            VS_REQUIRE(nv <= INT32_MAX && ldS <= INT32_MAX, "matrix too large for cuBLAS int API");
            const float alpha = 1.f, beta = 0.f;
            cublasHandle_t h = cublas_handle();
            CUBLAS_CHECK(cublasSetStream(h, stream));
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)nv, nq, d, &alpha, dX, d,
                                     dQ, d, &beta, dS, (int)ldS));
            break;
        }
        case DistVariant::WarpFp16:
            VS_REQUIRE(false, "use launch_inner_product_fp16 for the fp16 variant");
    }
}

void launch_inner_product_fp16(const float* dQ, int nq, const __half* dX, int64_t nv, int d,
                               float* dS, int64_t ldS, int tpb, cudaStream_t stream) {
    VS_REQUIRE(d % 8 == 0, "fp16 kernel needs d % 8 == 0");
    VS_REQUIRE(ldS >= nv, "ldS must be >= nv");
    if (nq <= 0 || nv <= 0) return;
    if (nq == 1)
        launch_warp_family<1>(ip_warp_fp16_kernel<1>, dQ, nq, dX, nv, d, dS, ldS, tpb, stream);
    else if (nq <= 4)
        launch_warp_family<4>(ip_warp_fp16_kernel<4>, dQ, nq, dX, nv, d, dS, ldS, tpb, stream);
    else
        launch_warp_family<8>(ip_warp_fp16_kernel<8>, dQ, nq, dX, nv, d, dS, ldS, tpb, stream);
}

void convert_to_half(const float* d_in, __half* d_out, int64_t count, cudaStream_t stream) {
    const int tpb = 256;
    const int blocks = (int)std::min<int64_t>(ceil_div(count, tpb), 65535);
    to_half_kernel<<<blocks, tpb, 0, stream>>>(d_in, d_out, count);
    CUDA_CHECK_LAUNCH();
}

KernelAttrs inner_product_kernel_attrs(DistVariant variant, int nq, int d, int tpb) {
    const void* kern = nullptr;
    size_t smem = 0;
    const int qg = nq == 1 ? 1 : (nq <= 4 ? 4 : 8);
    switch (variant) {
        case DistVariant::Naive: kern = (const void*)ip_naive_kernel; break;
        case DistVariant::Warp:
            kern = qg == 1 ? (const void*)ip_warp_kernel<1>
                           : (qg == 4 ? (const void*)ip_warp_kernel<4> : (const void*)ip_warp_kernel<8>);
            smem = size_t(qg) * d * sizeof(float);
            break;
        case DistVariant::WarpFp16:
            kern = qg == 1 ? (const void*)ip_warp_fp16_kernel<1>
                           : (qg == 4 ? (const void*)ip_warp_fp16_kernel<4>
                                      : (const void*)ip_warp_fp16_kernel<8>);
            smem = size_t(qg) * d * sizeof(float);
            break;
        case DistVariant::Tiled:
            kern = (const void*)ip_tiled_kernel;
            tpb = kTiledThreads;
            break;
        case DistVariant::Cublas: return KernelAttrs{};
    }
    cudaFuncAttributes fa{};
    CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
    KernelAttrs a;
    a.regs_per_thread = fa.numRegs;
    a.static_smem = fa.sharedSizeBytes;
    a.dynamic_smem = smem;
    a.threads_per_block = tpb;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a.active_blocks_per_sm, kern, tpb, smem));
    int dev = 0, max_threads = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    a.occupancy = double(a.active_blocks_per_sm) * tpb / max_threads;
    return a;
}

}  // namespace vs
