// IVF-Flat index: vectors grouped by nearest K-means centroid ("inverted lists").
//
// Search for a batch of queries (all stages are batched kernel launches):
//   1. coarse: scores vs. all centroids (L2 ranking, via <q,c> - |c|^2/2), top-nprobe lists
//   2. sizes:  per (query, probe) offsets into a candidate buffer
//   3. scan:   one block group per (query, probe list): inner products of the query with every
//              vector of the list, written with their ids into the candidate buffer
//   4. select: top-k over each query's candidates (segmented radix select)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vs/cuda_utils.cuh"
#include "vs/distance.cuh"
#include "vs/topk.cuh"

namespace vs {

// Host-side description of a trained IVF partition.
struct IvfLists {
    int nlist = 0;
    int d = 0;
    int64_t n = 0;
    std::vector<float> centroids;       // nlist x d
    std::vector<int64_t> list_offsets;  // nlist + 1
    std::vector<int32_t> list_ids;      // n vector ids grouped by list
};

// Groups ids 0..n-1 by label (stable, so ids stay ascending within a list).
IvfLists build_ivf_lists(const std::vector<float>& centroids, int nlist, int d,
                         const std::vector<int32_t>& labels);
void save_ivf(const std::string& path, const IvfLists& lists);
IvfLists load_ivf(const std::string& path);

struct IvfSearchParams {
    int nprobe = 8;
    DistVariant coarse_variant = DistVariant::Warp;
    int coarse_tpb = 256;
    int scan_tpb = 256;
    // Candidate buffer budget (elements). Larger batches are split into sub-batches that fit;
    // a single query that needs more than this still runs on its own.
    int64_t max_candidates = int64_t(128) << 20;
};

class IvfIndexGpu {
public:
    // h_base: n x d host vectors (L2-normalized); row i has id i.
    IvfIndexGpu(const IvfLists& lists, const float* h_base);

    void search(const float* d_queries, int nq, int k, const IvfSearchParams& p, float* d_vals,
                int32_t* d_ids, cudaStream_t stream);

    int nlist() const { return nlist_; }
    int dim() const { return d_; }
    int64_t size() const { return n_; }
    int64_t max_list_size() const { return max_list_; }

private:
    int nlist_;
    int d_;
    int64_t n_;
    int64_t max_list_ = 0;
    std::vector<int64_t> h_list_sizes_;

    DeviceBuffer<float> centroids_;
    DeviceBuffer<float> centroid_bias_;  // -|c|^2 / 2
    DeviceBuffer<int64_t> list_offsets_;
    DeviceBuffer<float> list_vectors_;   // n x d, in list order
    DeviceBuffer<int32_t> list_ids_;

    // per-search workspace
    DeviceBuffer<float> coarse_scores_;
    DeviceBuffer<float> probe_vals_;
    DeviceBuffer<int32_t> probe_ids_;
    DeviceBuffer<int64_t> probe_prefix_;
    DeviceBuffer<int64_t> query_totals_;
    DeviceBuffer<int64_t> cand_offsets_;
    DeviceBuffer<float> cand_vals_;
    DeviceBuffer<int32_t> cand_ids_;
    PinnedBuffer<int64_t> h_totals_;
    PinnedBuffer<int64_t> h_offsets_;
    TopKWorkspace ws_;
};

}  // namespace vs
