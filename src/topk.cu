#include "vs/topk.cuh"

#include <algorithm>
#include <cmath>

#include "vs/device_utils.cuh"

namespace vs {

namespace {

// Sorts n (a power of two) (value, id) pairs in shared memory by descending value.
__device__ void bitonic_sort_desc(float* v, int32_t* id, int n) {
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const float a = v[i], b = v[j];
                    if (desc ? (a < b) : (a > b)) {
                        v[i] = b;
                        v[j] = a;
                        const int32_t t = id[i];
                        id[i] = id[j];
                        id[j] = t;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ __launch_bounds__(kTopkThreads) void topk_tile_kernel(
    const float* __restrict__ vals, const int32_t* __restrict__ ids,
    const int64_t* __restrict__ offsets, int64_t row_stride, int64_t row_len, int k, int k_pad,
    int32_t id_base, float* __restrict__ out_vals, int32_t* __restrict__ out_ids,
    int64_t out_stride) {
    extern __shared__ uint32_t smem[];
    uint32_t* s_keys = smem;                                        // kTopkTile
    float* s_v = reinterpret_cast<float*>(s_keys + kTopkTile);      // k_pad
    int32_t* s_i = reinterpret_cast<int32_t*>(s_v + k_pad);         // k_pad
    __shared__ uint32_t s_hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need, s_cnt_gt, s_cnt_eq;

    const int r = blockIdx.y;
    const int t = blockIdx.x;
    int64_t rs, rl;
    if (offsets) {
        rs = offsets[r] - offsets[0];
        rl = offsets[r + 1] - offsets[r];
    } else {
        rs = int64_t(r) * row_stride;
        rl = row_len;
    }
    const int64_t ts = int64_t(t) * kTopkTile;
    const int n = (int)max(int64_t(0), min(int64_t(kTopkTile), rl - ts));
    const float* v_in = vals + rs + ts;
    const int32_t* id_in = ids ? ids + rs + ts : nullptr;
    float* v_out = out_vals + int64_t(r) * out_stride + int64_t(t) * k;
    int32_t* i_out = out_ids + int64_t(r) * out_stride + int64_t(t) * k;

    if (n <= k) {
        // Fewer values than k: keep all of them.
        for (int i = threadIdx.x; i < k_pad; i += blockDim.x) {
            if (i < n) {
                s_v[i] = v_in[i];
                s_i[i] = id_in ? id_in[i] : id_base + int32_t(ts + i);
            } else {
                s_v[i] = -INFINITY;
                s_i[i] = -1;
            }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) s_keys[i] = float_to_key(v_in[i]);
        if (threadIdx.x == 0) {
            s_prefix = 0;
            s_mask = 0;
            s_need = k;
            s_cnt_gt = 0;
            s_cnt_eq = 0;
        }
        __syncthreads();

        // Radix select, most significant byte first. After each pass the prefix fixes one more
        // byte of the k-th largest key; s_need counts how many keys equal to the current prefix
        // still have to be taken.
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix, mask = s_mask;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const uint32_t u = s_keys[i];
                if ((u & mask) == prefix) atomicAdd(&s_hist[(u >> shift) & 0xFFu], 1u);
            }
            __syncthreads();
            if (threadIdx.x < 32) {
                // Lane l owns bins 255-8l .. 248-8l; an inclusive scan over lanes counts keys
                // from the top, and the first lane whose count reaches s_need holds the bin.
                const int lane = threadIdx.x;
                const int base = 255 - 8 * lane;
                uint32_t c[8], sum = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    c[j] = s_hist[base - j];
                    sum += c[j];
                }
                uint32_t inc = sum;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const uint32_t y = __shfl_up_sync(kFullMask, inc, o);
                    if (lane >= o) inc += y;
                }
                const uint32_t need = (uint32_t)s_need;
                const unsigned ballot = __ballot_sync(kFullMask, inc >= need);
                const int owner = __ffs(ballot) - 1;
                if (lane == owner) {
                    uint32_t cum = inc - sum;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if (cum + c[j] >= need) {
                            s_prefix = prefix | (uint32_t(base - j) << shift);
                            s_mask = mask | (0xFFu << shift);
                            s_need = int(need - cum);
                            break;
                        }
                        cum += c[j];
                    }
                }
            }
            __syncthreads();
        }

        // Gather: all keys above the threshold, plus the first s_need keys equal to it.
        const uint32_t thr = s_prefix;
        const int need_eq = s_need;
        const int n_gt = k - need_eq;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t u = s_keys[i];
            int pos = -1;
            if (u > thr) {
                pos = atomicAdd(&s_cnt_gt, 1);
            } else if (u == thr) {
                const int e = atomicAdd(&s_cnt_eq, 1);
                if (e < need_eq) pos = n_gt + e;
            }
            if (pos >= 0) {
                s_v[pos] = key_to_float(u);
                s_i[pos] = id_in ? id_in[i] : id_base + int32_t(ts + i);
            }
        }
        for (int i = k + threadIdx.x; i < k_pad; i += blockDim.x) {
            s_v[i] = -INFINITY;
            s_i[i] = -1;
        }
    }
    __syncthreads();
    bitonic_sort_desc(s_v, s_i, k_pad);
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        v_out[i] = s_v[i];
        i_out[i] = s_i[i];
    }
}

int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

void launch_tile_pass(const float* vals, const int32_t* ids, const int64_t* offsets,
                      int64_t row_stride, int64_t row_len, int nrows, int tiles, int k,
                      int32_t id_base, float* out_vals, int32_t* out_ids, int64_t out_stride,
                      cudaStream_t stream) {
    const int k_pad = next_pow2(k);
    const size_t smem = kTopkTile * sizeof(uint32_t) + size_t(k_pad) * (sizeof(float) + sizeof(int32_t));
    const dim3 grid((unsigned)tiles, (unsigned)nrows);
    topk_tile_kernel<<<grid, kTopkThreads, smem, stream>>>(vals, ids, offsets, row_stride, row_len,
                                                           k, k_pad, id_base, out_vals, out_ids,
                                                           out_stride);
    CUDA_CHECK_LAUNCH();
}

}  // namespace

void topk_rows(const float* d_vals, const int32_t* d_ids, const int64_t* d_row_offsets,
               int64_t row_stride, int64_t row_len, int nrows, int k, int32_t id_base,
               float* d_out_vals, int32_t* d_out_ids, int64_t out_stride, TopKWorkspace& ws,
               cudaStream_t stream) {
    VS_REQUIRE(k >= 1 && k <= kTopkMaxK, "k must be in [1, 1024]");
    VS_REQUIRE(nrows <= 65535, "topk_rows handles at most 65535 rows per call");
    VS_REQUIRE(out_stride >= k, "out_stride must be >= k");
    if (nrows <= 0) return;

    const float* vals = d_vals;
    const int32_t* ids = d_ids;
    const int64_t* offsets = d_row_offsets;
    int64_t stride = row_stride, len = row_len;
    int32_t base = id_base;
    int buf = 0;
    while (true) {
        const int64_t tiles = std::max<int64_t>(1, ceil_div(len, kTopkTile));
        if (tiles == 1) {
            launch_tile_pass(vals, ids, offsets, stride, len, nrows, 1, k, base, d_out_vals,
                             d_out_ids, out_stride, stream);
            return;
        }
        // Every row gets `tiles` slots of k candidates; tiles past the end of a shorter CSR row
        // write (-inf, -1) padding.
        const int64_t next_len = tiles * k;
        ws.vals[buf].ensure(size_t(nrows) * next_len);
        ws.ids[buf].ensure(size_t(nrows) * next_len);
        launch_tile_pass(vals, ids, offsets, stride, len, nrows, (int)tiles, k, base,
                         ws.vals[buf].data(), ws.ids[buf].data(), next_len, stream);
        vals = ws.vals[buf].data();
        ids = ws.ids[buf].data();
        offsets = nullptr;
        stride = len = next_len;
        base = 0;
        buf ^= 1;
    }
}

}  // namespace vs
