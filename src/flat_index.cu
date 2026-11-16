#include "vs/flat_index.cuh"

#include <algorithm>

namespace vs {

FlatIndexGpu::FlatIndexGpu(const float* h_base, int64_t n, int d) : n_(n), d_(d) {
    VS_REQUIRE(n > 0 && n <= INT32_MAX, "base size must fit in int32 ids");
    base_.resize(size_t(n) * d);
    CUDA_CHECK(cudaMemcpy(base_.data(), h_base, size_t(n) * d * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void FlatIndexGpu::search(const float* d_queries, int nq, int k, const FlatSearchParams& p,
                          float* d_vals, int32_t* d_ids, cudaStream_t stream) {
    VS_REQUIRE(nq > 0 && k >= 1 && k <= kTopkMaxK, "invalid nq or k");

    if (p.variant == DistVariant::WarpFp16 && base_half_.size() == 0) {
        base_half_.resize(size_t(n_) * d_);
        convert_to_half(base_.data(), base_half_.data(), n_ * d_, stream);
    }

    int64_t chunk = std::max<int64_t>(kTopkTile, p.max_score_elems / nq);
    if (chunk < n_) chunk = (chunk / kTopkTile) * kTopkTile;
    chunk = std::min(chunk, n_);
    const int64_t nchunks = ceil_div(n_, chunk);
    scores_.ensure(size_t(nq) * chunk);

    auto scores_for = [&](int64_t start, int64_t len) {
        if (p.variant == DistVariant::WarpFp16)
            launch_inner_product_fp16(d_queries, nq, base_half_.data() + start * d_, len, d_,
                                      scores_.data(), chunk, p.threads_per_block, stream);
        else
            launch_inner_product(p.variant, d_queries, nq, base_.data() + start * d_, len, d_,
                                 scores_.data(), chunk, p.threads_per_block, stream);
    };

    if (nchunks == 1) {
        scores_for(0, n_);
        topk_rows(scores_.data(), nullptr, nullptr, chunk, n_, nq, k, 0, d_vals, d_ids, k, ws_,
                  stream);
        return;
    }

    const int64_t cand = nchunks * k;
    chunk_vals_.ensure(size_t(nq) * cand);
    chunk_ids_.ensure(size_t(nq) * cand);
    for (int64_t c = 0; c < nchunks; ++c) {
        const int64_t start = c * chunk;
        const int64_t len = std::min(chunk, n_ - start);
        scores_for(start, len);
        topk_rows(scores_.data(), nullptr, nullptr, chunk, len, nq, k, int32_t(start),
                  chunk_vals_.data() + c * k, chunk_ids_.data() + c * k, cand, ws_, stream);
    }
    topk_rows(chunk_vals_.data(), chunk_ids_.data(), nullptr, cand, cand, nq, k, 0, d_vals, d_ids,
              k, ws_, stream);
}

}  // namespace vs
