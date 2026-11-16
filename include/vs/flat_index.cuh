// Exact brute-force search on the GPU: chunked inner products followed by top-k selection.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "vs/cuda_utils.cuh"
#include "vs/distance.cuh"
#include "vs/topk.cuh"

namespace vs {

struct FlatSearchParams {
    DistVariant variant = DistVariant::Warp;
    int threads_per_block = 256;
    // Upper bound on the nq x chunk score matrix (elements); the base is processed in chunks
    // that fit, each chunk reduced to k candidates per query, then merged.
    int64_t max_score_elems = int64_t(128) << 20;
};

class FlatIndexGpu {
public:
    // Copies n x d host vectors (already L2-normalized) to the device.
    FlatIndexGpu(const float* h_base, int64_t n, int d);

    // d_queries: nq x d on the device. Outputs nq x k (descending score) on the device.
    void search(const float* d_queries, int nq, int k, const FlatSearchParams& p, float* d_vals,
                int32_t* d_ids, cudaStream_t stream);

    int64_t size() const { return n_; }
    int dim() const { return d_; }
    const float* device_data() const { return base_.data(); }

private:
    int64_t n_;
    int d_;
    DeviceBuffer<float> base_;
    DeviceBuffer<__half> base_half_;  // created on first WarpFp16 search
    DeviceBuffer<float> scores_;
    DeviceBuffer<float> chunk_vals_;
    DeviceBuffer<int32_t> chunk_ids_;
    TopKWorkspace ws_;
};

}  // namespace vs
