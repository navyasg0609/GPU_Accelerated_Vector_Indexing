// Benchmark plumbing shared by the apps: run metadata columns and batched, event-timed search.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vs/common.h"
#include "vs/cuda_utils.cuh"

namespace vs {

// Columns written in front of every CSV row so each measurement carries its context.
struct RunMeta {
    DeviceInfo dev;
    std::string dataset;
    int64_t n = 0;
    int d = 0;
    int nq = 0;

    static std::vector<std::string> header() {
        return {"dataset", "gpu", "compute_capability", "cuda_runtime", "cuda_driver", "n", "d", "nq"};
    }
    std::vector<std::string> values() const {
        return {dataset, dev.name, dev.cc(), dev.runtime_str(), dev.driver_str(),
                std::to_string(n), std::to_string(d), std::to_string(nq)};
    }
};

// Shared schema for search benchmarks (C++ and python/faiss_bench.py write the same columns).
std::vector<std::string> search_csv_header();
std::vector<std::string> search_csv_values(const std::string& method, const std::string& variant,
                                           int tpb, int nlist, int nprobe, int k, int batch,
                                           int runs, const Stats& latency_ms, const Stats& qps,
                                           double r1, double r10, double r100);

using SearchFn = std::function<void(const float* d_queries, int nq, float* d_vals,
                                    int32_t* d_ids, cudaStream_t stream)>;

struct BatchTiming {
    Stats latency_ms;               // per batch, over all batches of all timed runs
    Stats qps;                      // per timed run
    std::vector<int32_t> ids;       // nq x k results of the first timed run
};

// Runs every query in batches of `batch` (wrapping around the query set so every batch is
// full), `warmup` untimed passes then `runs` timed passes. Each batch is timed with CUDA events
// from the upload of its queries (pinned memory) to the download of its results, so the time
// includes host<->device transfers and any host synchronisation inside the search.
BatchTiming time_batched(const float* h_queries, int nq, int d, int batch, int k, int warmup,
                         int runs, const SearchFn& search, cudaStream_t stream);

// recall@{1,10,100} of ids (nq x k) against ground truth (nq x gt_k); NaN when r > min(k, gt_k).
void recall_triplet(const std::vector<int32_t>& ids, int k, const int32_t* gt, int gt_k, int nq,
                    double& r1, double& r10, double& r100);

}  // namespace vs
