#include "vs/kmeans.cuh"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <random>
#include <unordered_set>

#include "vs/cuda_utils.cuh"
#include "vs/device_utils.cuh"

namespace vs {

namespace {

constexpr int kSplits = 32;  // partial-sum blocks per cluster

struct ToDouble {
    __host__ __device__ double operator()(float x) const { return double(x); }
};

// Warp per row: out[i] = |x_i|^2.
__global__ void row_sqnorm_kernel(const float* __restrict__ X, int64_t n, int d,
                                  float* __restrict__ out) {
    const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= n) return;
    const float4* x4 = reinterpret_cast<const float4*>(X + row * d);
    float acc = 0.f;
    for (int j = lane; j < (d >> 2); j += 32) {
        const float4 v = x4[j];
        acc += dot4(v, v);
    }
    acc = warp_sum(acc);
    if (lane == 0) out[row] = acc;
}

// Warp per row: label = argmin_c |c|^2 - 2<x,c> (ties -> lowest c), mind = |x|^2 + that.
__global__ void assign_kernel(const float* __restrict__ S, int64_t m, int k,
                              const float* __restrict__ xnorm, const float* __restrict__ cnorm,
                              int32_t* __restrict__ labels, float* __restrict__ mind) {
    const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= m) return;
    float best = INFINITY;
    int bi = INT_MAX;
    for (int c = lane; c < k; c += 32) {
        const float dist = cnorm[c] - 2.f * S[row * k + c];
        if (dist < best) {
            best = dist;
            bi = c;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(kFullMask, best, o);
        const int oi = __shfl_xor_sync(kFullMask, bi, o);
        if (ob < best || (ob == best && oi < bi)) {
            best = ob;
            bi = oi;
        }
    }
    if (lane == 0) {
        labels[row] = bi;
        mind[row] = fmaxf(xnorm[row] + best, 0.f);
    }
}

// Per-dimension sum and sum of squares (for the data variance used by the tolerance).
__global__ void col_stats_kernel(const float* __restrict__ X, int64_t n, int d,
                                 double* __restrict__ sum, double* __restrict__ sumsq) {
    const int64_t per = (n + gridDim.x - 1) / gridDim.x;
    const int64_t r0 = int64_t(blockIdx.x) * per;
    const int64_t r1 = min(n, r0 + per);
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float s = 0.f, s2 = 0.f;
        for (int64_t r = r0; r < r1; ++r) {
            const float x = X[r * d + j];
            s += x;
            s2 += x * x;
        }
        atomicAdd(&sum[j], double(s));
        atomicAdd(&sumsq[j], double(s2));
    }
}

__global__ void count_kernel(const int32_t* __restrict__ labels, int64_t n, int k,
                             int* __restrict__ counts) {
    extern __shared__ int sh_counts[];
    for (int i = threadIdx.x; i < k; i += blockDim.x) sh_counts[i] = 0;
    __syncthreads();
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += int64_t(gridDim.x) * blockDim.x)
        atomicAdd(&sh_counts[labels[i]], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        if (sh_counts[i]) atomicAdd(&counts[i], sh_counts[i]);
}

// grid = (k, kSplits). Block (c, s) sums points perm[off_c + s], perm[off_c + s + kSplits], ...
// Threads cover dimensions, so each point row is read with coalesced loads.
__global__ void partial_sum_kernel(const float* __restrict__ X, int d,
                                   const int32_t* __restrict__ perm,
                                   const int64_t* __restrict__ offsets,
                                   float* __restrict__ partial) {
    const int c = blockIdx.x, s = blockIdx.y;
    const int64_t b = offsets[c], e = offsets[c + 1];
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float acc = 0.f;
        for (int64_t i = b + s; i < e; i += kSplits) acc += X[int64_t(perm[i]) * d + j];
        partial[(int64_t(c) * kSplits + s) * d + j] = acc;
    }
}

// Thread per (cluster, dim): reduce the splits in fixed order, divide, record the shift.
__global__ void finalize_kernel(const float* __restrict__ partial, const int* __restrict__ counts,
                                int k, int d, const float* __restrict__ old_c,
                                float* __restrict__ new_c, float* __restrict__ shift_sq) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= int64_t(k) * d) return;
    const int c = int(idx / d), j = int(idx % d);
    float sum = 0.f;
    for (int s = 0; s < kSplits; ++s) sum += partial[(int64_t(c) * kSplits + s) * d + j];
    const int cnt = counts[c];
    const float v = cnt > 0 ? sum / float(cnt) : old_c[idx];
    new_c[idx] = v;
    const float diff = v - old_c[idx];
    shift_sq[idx] = diff * diff;
}

// Warp per row: mind[i] = |x_i - c|^2 (first) or min(mind[i], |x_i - c|^2).
__global__ void dist_to_centroid_kernel(const float* __restrict__ X, int64_t n, int d,
                                        const float* __restrict__ c, bool first,
                                        float* __restrict__ mind) {
    const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= n) return;
    const float4* x4 = reinterpret_cast<const float4*>(X + row * d);
    const float4* c4 = reinterpret_cast<const float4*>(c);
    float acc = 0.f;
    for (int j = lane; j < (d >> 2); j += 32) {
        const float4 a = x4[j], b = __ldg(c4 + j);
        const float4 df = make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
        acc += dot4(df, df);
    }
    acc = warp_sum(acc);
    if (lane == 0) mind[row] = first ? acc : fminf(mind[row], acc);
}

// k-means++ sampling step: picks the first row whose cumulative D^2 exceeds u * total, and
// copies it into out_c.
__global__ void kpp_pick_kernel(const double* __restrict__ cum, int64_t n, double u,
                                const float* __restrict__ X, int d, float* __restrict__ out_c) {
    __shared__ int64_t s_idx;
    if (threadIdx.x == 0) {
        const double r = u * cum[n - 1];
        int64_t lo = 0, hi = n - 1;
        while (lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (cum[mid] > r) hi = mid;
            else lo = mid + 1;
        }
        s_idx = lo;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < d; j += blockDim.x) out_c[j] = X[s_idx * d + j];
}

unsigned warp_rows_grid(int64_t rows, int tpb) { return (unsigned)ceil_div(rows * 32, tpb); }

}  // namespace

KMeansResult kmeans_gpu(const float* h_data, int64_t n, int d, const KMeansParams& p) {
    VS_REQUIRE(d % 4 == 0, "dimension must be a multiple of 4");
    VS_REQUIRE(p.k >= 1 && p.k <= n, "k must be in [1, n]");
    VS_REQUIRE(n <= INT32_MAX, "n must fit in int32");
    const int k = p.k;
    KMeansResult res;
    cudaStream_t s = 0;
    auto policy = thrust::cuda::par.on(s);

    GpuTimer timer;
    DeviceBuffer<float> X(size_t(n) * d);
    timer.start(s);
    CUDA_CHECK(cudaMemcpyAsync(X.data(), h_data, size_t(n) * d * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    res.h2d_ms = timer.stop(s);

    const int64_t chunk = std::min<int64_t>(p.chunk_rows, n);
    DeviceBuffer<float> C(size_t(k) * d), C_new(size_t(k) * d);
    DeviceBuffer<float> xnorm(n), cnorm(k), mind(n), S(size_t(chunk) * k);
    DeviceBuffer<int32_t> labels(n), sort_keys(n), perm(n);
    DeviceBuffer<int> counts(k);
    DeviceBuffer<int64_t> offsets(k + 1);
    DeviceBuffer<float> partial(size_t(k) * kSplits * d), shift_sq(size_t(k) * d);
    DeviceBuffer<double> col_sum(d), col_sumsq(d), cum(n);
    std::vector<int> h_counts(k);
    std::vector<int64_t> h_offsets(k + 1);
    std::vector<float> h_shift(size_t(k) * d);
    const int tpb = 256;

    GpuTimer total_timer;
    total_timer.start(s);
    timer.start(s);

    // ---- data statistics ----
    row_sqnorm_kernel<<<warp_rows_grid(n, tpb), tpb, 0, s>>>(X.data(), n, d, xnorm.data());
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemsetAsync(col_sum.data(), 0, d * sizeof(double), s));
    CUDA_CHECK(cudaMemsetAsync(col_sumsq.data(), 0, d * sizeof(double), s));
    col_stats_kernel<<<256, std::min(d, 1024), 0, s>>>(X.data(), n, d, col_sum.data(),
                                                       col_sumsq.data());
    CUDA_CHECK_LAUNCH();
    std::vector<double> h_sum(d), h_sumsq(d);
    col_sum.download(h_sum.data(), d, s);
    col_sumsq.download(h_sumsq.data(), d, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    double mean_var = 0;
    for (int j = 0; j < d; ++j) {
        const double mu = h_sum[j] / n;
        mean_var += h_sumsq[j] / n - mu * mu;
    }
    mean_var /= d;
    const double tol_abs = p.tol * mean_var;

    // ---- seeding ----
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    if (p.init == "random") {
        std::unordered_set<int64_t> chosen;
        std::vector<int64_t> picks;
        while ((int)picks.size() < k) {
            const int64_t i = int64_t(rng() % uint64_t(n));
            if (chosen.insert(i).second) picks.push_back(i);
        }
        for (int c = 0; c < k; ++c)
            CUDA_CHECK(cudaMemcpyAsync(C.data() + size_t(c) * d, X.data() + picks[c] * d,
                                       d * sizeof(float), cudaMemcpyDeviceToDevice, s));
    } else {
        VS_REQUIRE(p.init == "kmeans++", "init must be kmeans++ or random");
        const int64_t first = int64_t(rng() % uint64_t(n));
        CUDA_CHECK(cudaMemcpyAsync(C.data(), X.data() + first * d, d * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
        dist_to_centroid_kernel<<<warp_rows_grid(n, tpb), tpb, 0, s>>>(X.data(), n, d, C.data(),
                                                                      true, mind.data());
        CUDA_CHECK_LAUNCH();
        for (int c = 1; c < k; ++c) {
            auto as_double = thrust::make_transform_iterator(mind.data(), ToDouble());
            thrust::inclusive_scan(policy, as_double, as_double + n, cum.data());
            kpp_pick_kernel<<<1, 256, 0, s>>>(cum.data(), n, unif(rng), X.data(), d,
                                              C.data() + size_t(c) * d);
            CUDA_CHECK_LAUNCH();
            dist_to_centroid_kernel<<<warp_rows_grid(n, tpb), tpb, 0, s>>>(
                X.data(), n, d, C.data() + size_t(c) * d, false, mind.data());
            CUDA_CHECK_LAUNCH();
        }
    }
    res.init_ms = timer.stop(s);
    timer.start(s);

    auto assign_all = [&](const float* cent) -> double {
        row_sqnorm_kernel<<<warp_rows_grid(k, tpb), tpb, 0, s>>>(cent, k, d, cnorm.data());
        CUDA_CHECK_LAUNCH();
        for (int64_t r0 = 0; r0 < n; r0 += chunk) {
            const int64_t m = std::min(chunk, n - r0);
            launch_inner_product(p.variant, X.data() + r0 * d, (int)m, cent, k, d, S.data(), k,
                                 p.threads_per_block, s);
            assign_kernel<<<warp_rows_grid(m, tpb), tpb, 0, s>>>(
                S.data(), m, k, xnorm.data() + r0, cnorm.data(), labels.data() + r0,
                mind.data() + r0);
            CUDA_CHECK_LAUNCH();
        }
        auto as_double = thrust::make_transform_iterator(mind.data(), ToDouble());
        return thrust::reduce(policy, as_double, as_double + n, 0.0, thrust::plus<double>());
    };

    for (int it = 0; it < p.max_iter; ++it) {
        const double inertia = assign_all(C.data());

        // update step
        CUDA_CHECK(cudaMemsetAsync(counts.data(), 0, k * sizeof(int), s));
        count_kernel<<<256, 256, k * sizeof(int), s>>>(labels.data(), n, k, counts.data());
        CUDA_CHECK_LAUNCH();
        counts.download(h_counts.data(), k, s);
        CUDA_CHECK(cudaMemcpyAsync(sort_keys.data(), labels.data(), n * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, s));
        thrust::sequence(policy, perm.data(), perm.data() + n);
        thrust::stable_sort_by_key(policy, sort_keys.data(), sort_keys.data() + n, perm.data());
        CUDA_CHECK(cudaStreamSynchronize(s));
        h_offsets[0] = 0;
        res.empty_clusters = 0;
        for (int c = 0; c < k; ++c) {
            h_offsets[c + 1] = h_offsets[c] + h_counts[c];
            res.empty_clusters += h_counts[c] == 0;
        }
        offsets.upload(h_offsets.data(), k + 1, s);
        partial_sum_kernel<<<dim3(k, kSplits), 128, 0, s>>>(X.data(), d, perm.data(),
                                                            offsets.data(), partial.data());
        CUDA_CHECK_LAUNCH();
        finalize_kernel<<<(unsigned)ceil_div(int64_t(k) * d, 256), 256, 0, s>>>(
            partial.data(), counts.data(), k, d, C.data(), C_new.data(), shift_sq.data());
        CUDA_CHECK_LAUNCH();
        shift_sq.download(h_shift.data(), size_t(k) * d, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        double shift = 0;
        for (float v : h_shift) shift += v;
        std::swap(C, C_new);

        res.inertia_history.push_back(inertia);
        res.shift_history.push_back(shift);
        res.iterations = it + 1;
        if (p.verbose)
            std::printf("  iter %3d  inertia %.6e  center shift %.3e (tol %.3e)  empty %d\n", it,
                        inertia, shift, tol_abs, res.empty_clusters);
        if (shift <= tol_abs) {
            res.converged = true;
            break;
        }
    }
    res.inertia = assign_all(C.data());
    res.iter_ms = timer.stop(s);
    res.total_ms = total_timer.stop(s);

    res.centroids.resize(size_t(k) * d);
    res.labels.resize(n);
    C.download(res.centroids.data(), size_t(k) * d, s);
    labels.download(res.labels.data(), n, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    return res;
}

}  // namespace vs
