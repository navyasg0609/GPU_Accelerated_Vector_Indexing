#include "vs/ivf_index.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "vs/device_utils.cuh"

namespace vs {

namespace {

constexpr char kMagic[4] = {'I', 'V', 'F', '1'};

__global__ void add_bias_kernel(float* __restrict__ S, int nq, int nlist,
                                const float* __restrict__ bias) {
    const int64_t total = int64_t(nq) * nlist;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += int64_t(gridDim.x) * blockDim.x)
        S[i] += bias[i % nlist];
}

// One thread per query: exclusive prefix of the probed list sizes, and the query's total.
__global__ void probe_prefix_kernel(const int32_t* __restrict__ probe_ids, int nq, int nprobe,
                                    const int64_t* __restrict__ list_offsets,
                                    int64_t* __restrict__ prefix, int64_t* __restrict__ totals) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int64_t tot = 0;
    for (int p = 0; p < nprobe; ++p) {
        const int c = probe_ids[int64_t(q) * nprobe + p];
        prefix[int64_t(q) * nprobe + p] = tot;
        if (c >= 0) tot += list_offsets[c + 1] - list_offsets[c];
    }
    totals[q] = tot;
}

// grid = (nprobe * splits, nsub). Block (p * splits + s, qi) scans every splits-th group of
// warps of list probe_ids[q][p] for query q = q_begin + qi. One warp per vector, float4 loads.
__global__ void ivf_scan_kernel(const float* __restrict__ Q, int q_begin, int d,
                                const int32_t* __restrict__ probe_ids,
                                const int64_t* __restrict__ probe_prefix, int nprobe, int splits,
                                const int64_t* __restrict__ list_offsets,
                                const float* __restrict__ list_vecs,
                                const int32_t* __restrict__ list_ids,
                                const int64_t* __restrict__ cand_offsets,
                                float* __restrict__ cand_vals, int32_t* __restrict__ cand_ids) {
    extern __shared__ float4 sq[];
    const int d4 = d >> 2;
    const int p = blockIdx.x / splits;
    const int s = blockIdx.x - p * splits;
    const int qi = blockIdx.y;
    const int64_t q = int64_t(q_begin) + qi;
    for (int i = threadIdx.x; i < d4; i += blockDim.x)
        sq[i] = reinterpret_cast<const float4*>(Q + q * d)[i];
    __syncthreads();

    const int c = probe_ids[q * nprobe + p];
    if (c < 0) return;
    const int64_t ls = list_offsets[c], le = list_offsets[c + 1];
    const int64_t out0 = (cand_offsets[qi] - cand_offsets[0]) + probe_prefix[q * nprobe + p];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    for (int64_t i = ls + int64_t(s) * wpb + warp; i < le; i += int64_t(splits) * wpb) {
        const float4* x4 = reinterpret_cast<const float4*>(list_vecs + i * d);
        float acc = 0.f;
        for (int j = lane; j < d4; j += 32) acc += dot4(__ldg(x4 + j), sq[j]);
        acc = warp_sum(acc);
        if (lane == 0) {
            cand_vals[out0 + (i - ls)] = acc;
            cand_ids[out0 + (i - ls)] = list_ids[i];
        }
    }
}

int sm_count() {
    int dev = 0, v = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
    return v;
}

}  // namespace

IvfLists build_ivf_lists(const std::vector<float>& centroids, int nlist, int d,
                         const std::vector<int32_t>& labels) {
    IvfLists L;
    L.nlist = nlist;
    L.d = d;
    L.n = int64_t(labels.size());
    L.centroids = centroids;
    L.list_offsets.assign(nlist + 1, 0);
    for (int32_t l : labels) {
        VS_REQUIRE(l >= 0 && l < nlist, "label out of range");
        ++L.list_offsets[l + 1];
    }
    for (int c = 0; c < nlist; ++c) L.list_offsets[c + 1] += L.list_offsets[c];
    std::vector<int64_t> cursor(L.list_offsets.begin(), L.list_offsets.end() - 1);
    L.list_ids.resize(labels.size());
    for (int64_t i = 0; i < L.n; ++i) L.list_ids[cursor[labels[i]]++] = int32_t(i);
    return L;
}

void save_ivf(const std::string& path, const IvfLists& L) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    FILE* f = std::fopen(path.c_str(), "wb");
    VS_REQUIRE(f != nullptr, "could not open index file for writing");
    std::fwrite(kMagic, 1, 4, f);
    const int32_t hdr[2] = {L.nlist, L.d};
    std::fwrite(hdr, sizeof(int32_t), 2, f);
    std::fwrite(&L.n, sizeof(int64_t), 1, f);
    std::fwrite(L.centroids.data(), sizeof(float), L.centroids.size(), f);
    std::fwrite(L.list_offsets.data(), sizeof(int64_t), L.list_offsets.size(), f);
    VS_REQUIRE(std::fwrite(L.list_ids.data(), sizeof(int32_t), L.list_ids.size(), f) ==
                   L.list_ids.size(),
               "index write failed");
    std::fclose(f);
}

IvfLists load_ivf(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "Could not open index %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    char magic[4];
    VS_REQUIRE(std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, kMagic, 4) == 0,
               "not an IVF1 index file");
    int32_t hdr[2];
    IvfLists L;
    VS_REQUIRE(std::fread(hdr, sizeof(int32_t), 2, f) == 2, "truncated index");
    VS_REQUIRE(std::fread(&L.n, sizeof(int64_t), 1, f) == 1, "truncated index");
    L.nlist = hdr[0];
    L.d = hdr[1];
    L.centroids.resize(size_t(L.nlist) * L.d);
    L.list_offsets.resize(L.nlist + 1);
    L.list_ids.resize(L.n);
    VS_REQUIRE(std::fread(L.centroids.data(), sizeof(float), L.centroids.size(), f) ==
                   L.centroids.size(),
               "truncated index");
    VS_REQUIRE(std::fread(L.list_offsets.data(), sizeof(int64_t), L.list_offsets.size(), f) ==
                   L.list_offsets.size(),
               "truncated index");
    VS_REQUIRE(std::fread(L.list_ids.data(), sizeof(int32_t), L.list_ids.size(), f) ==
                   L.list_ids.size(),
               "truncated index");
    std::fclose(f);
    return L;
}

IvfIndexGpu::IvfIndexGpu(const IvfLists& L, const float* h_base)
    : nlist_(L.nlist), d_(L.d), n_(L.n) {
    VS_REQUIRE(d_ % 4 == 0, "dimension must be a multiple of 4");
    VS_REQUIRE(n_ <= INT32_MAX, "ids must fit in int32");
    std::vector<float> bias(nlist_);
    h_list_sizes_.resize(nlist_);
    for (int c = 0; c < nlist_; ++c) {
        double s = 0;
        for (int j = 0; j < d_; ++j) s += double(L.centroids[size_t(c) * d_ + j]) * L.centroids[size_t(c) * d_ + j];
        bias[c] = float(-0.5 * s);
        h_list_sizes_[c] = L.list_offsets[c + 1] - L.list_offsets[c];
        max_list_ = std::max(max_list_, h_list_sizes_[c]);
    }
    centroids_.upload(L.centroids.data(), L.centroids.size());
    centroid_bias_.upload(bias.data(), bias.size());
    list_offsets_.upload(L.list_offsets.data(), L.list_offsets.size());
    list_ids_.upload(L.list_ids.data(), L.list_ids.size());

    // Copy vectors into list order in chunks to bound host memory.
    list_vectors_.resize(size_t(n_) * d_);
    const int64_t chunk = 65536;
    std::vector<float> staging(size_t(std::min(chunk, n_)) * d_);
    for (int64_t s = 0; s < n_; s += chunk) {
        const int64_t e = std::min(n_, s + chunk);
        for (int64_t i = s; i < e; ++i)
            std::memcpy(&staging[size_t(i - s) * d_], h_base + int64_t(L.list_ids[i]) * d_,
                        d_ * sizeof(float));
        CUDA_CHECK(cudaMemcpy(list_vectors_.data() + s * d_, staging.data(),
                              size_t(e - s) * d_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void IvfIndexGpu::search(const float* d_queries, int nq, int k, const IvfSearchParams& p,
                         float* d_vals, int32_t* d_ids, cudaStream_t stream) {
    VS_REQUIRE(nq > 0 && nq <= 65535, "nq must be in [1, 65535]");
    VS_REQUIRE(p.nprobe >= 1 && p.nprobe <= nlist_, "nprobe must be in [1, nlist]");
    VS_REQUIRE(k >= 1 && k <= kTopkMaxK, "k must be in [1, 1024]");
    VS_REQUIRE(p.scan_tpb >= 32 && p.scan_tpb <= 1024 && p.scan_tpb % 32 == 0,
               "scan threads per block must be a multiple of 32");
    const int nprobe = p.nprobe;
    const size_t np = size_t(nq) * nprobe;

    // 1. coarse search
    coarse_scores_.ensure(size_t(nq) * nlist_);
    probe_vals_.ensure(np);
    probe_ids_.ensure(np);
    probe_prefix_.ensure(np);
    query_totals_.ensure(nq);
    cand_offsets_.ensure(nq + 1);
    h_totals_.ensure(nq);
    h_offsets_.ensure(nq + 1);

    launch_inner_product(p.coarse_variant, d_queries, nq, centroids_.data(), nlist_, d_,
                         coarse_scores_.data(), nlist_, p.coarse_tpb, stream);
    {
        const int blocks = (int)std::min<int64_t>(ceil_div(int64_t(nq) * nlist_, 256), 4096);
        add_bias_kernel<<<blocks, 256, 0, stream>>>(coarse_scores_.data(), nq, nlist_,
                                                    centroid_bias_.data());
        CUDA_CHECK_LAUNCH();
    }
    topk_rows(coarse_scores_.data(), nullptr, nullptr, nlist_, nlist_, nq, nprobe, 0,
              probe_vals_.data(), probe_ids_.data(), nprobe, ws_, stream);

    // 2. candidate layout
    probe_prefix_kernel<<<(unsigned)ceil_div(nq, 128), 128, 0, stream>>>(
        probe_ids_.data(), nq, nprobe, list_offsets_.data(), probe_prefix_.data(),
        query_totals_.data());
    CUDA_CHECK_LAUNCH();
    query_totals_.download(h_totals_.data(), nq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    h_offsets_[0] = 0;
    for (int q = 0; q < nq; ++q) h_offsets_[q + 1] = h_offsets_[q] + h_totals_[q];
    CUDA_CHECK(cudaMemcpyAsync(cand_offsets_.data(), h_offsets_.data(), (nq + 1) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));

    // Split into sub-batches whose candidates fit the budget.
    struct Sub {
        int q0, q1;
        int64_t max_len;
    };
    std::vector<Sub> subs;
    int64_t biggest = 0;
    for (int q0 = 0; q0 < nq;) {
        int q1 = q0;
        int64_t max_len = 0;
        while (q1 < nq && (q1 == q0 || h_offsets_[q1 + 1] - h_offsets_[q0] <= p.max_candidates)) {
            max_len = std::max(max_len, h_totals_[q1]);
            ++q1;
        }
        subs.push_back({q0, q1, max_len});
        biggest = std::max(biggest, h_offsets_[q1] - h_offsets_[q0]);
        q0 = q1;
    }
    cand_vals_.ensure(std::max<int64_t>(1, biggest));
    cand_ids_.ensure(std::max<int64_t>(1, biggest));

    // 3 + 4. scan the probed lists and select top-k per query
    const int wpb = p.scan_tpb / 32;
    const size_t smem = size_t(d_) * sizeof(float);
    for (const Sub& sb : subs) {
        const int nsub = sb.q1 - sb.q0;
        const int64_t target_blocks = int64_t(sm_count()) * 8;
        const int64_t useful = std::max<int64_t>(1, ceil_div(max_list_, int64_t(wpb) * 8));
        const int splits = (int)std::max<int64_t>(
            1, std::min(useful, ceil_div(target_blocks, int64_t(nprobe) * nsub)));
        const dim3 grid((unsigned)(int64_t(nprobe) * splits), (unsigned)nsub);
        ivf_scan_kernel<<<grid, p.scan_tpb, smem, stream>>>(
            d_queries, sb.q0, d_, probe_ids_.data(), probe_prefix_.data(), nprobe, splits,
            list_offsets_.data(), list_vectors_.data(), list_ids_.data(),
            cand_offsets_.data() + sb.q0, cand_vals_.data(), cand_ids_.data());
        CUDA_CHECK_LAUNCH();
        topk_rows(cand_vals_.data(), cand_ids_.data(), cand_offsets_.data() + sb.q0, 0,
                  sb.max_len, nsub, k, 0, d_vals + int64_t(sb.q0) * k,
                  d_ids + int64_t(sb.q0) * k, k, ws_, stream);
    }
}

}  // namespace vs
