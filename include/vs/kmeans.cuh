// Lloyd's K-means (squared L2) on the GPU.
//
// Per iteration:
//   assign   X·C^T on the GPU (selectable distance kernel), then a warp-per-row argmin kernel
//            using |x|^2 - 2<x,c> + |c|^2; writes labels and each point's squared distance
//   update   stable sort of point ids by label, per-cluster partial sums over a fixed split of
//            each cluster's points (deterministic order), then a reduction + divide kernel
//   converge stop when sum |c_new - c_old|^2 <= tol * mean per-dimension variance of X
//            (the scikit-learn criterion), or at max_iter
// A final assignment step makes labels and inertia consistent with the returned centroids.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vs/distance.cuh"

namespace vs {

struct KMeansParams {
    int k = 128;
    int max_iter = 100;
    double tol = 1e-4;
    uint64_t seed = 42;
    std::string init = "kmeans++";  // "kmeans++" or "random"
    DistVariant variant = DistVariant::Tiled;
    int threads_per_block = 256;
    int64_t chunk_rows = 262144;    // rows per assignment GEMM
    bool verbose = true;
};

struct KMeansResult {
    std::vector<float> centroids;  // k x d
    std::vector<int32_t> labels;   // n
    double inertia = 0;            // sum of squared distances to the assigned centroid
    int iterations = 0;
    bool converged = false;
    int empty_clusters = 0;        // clusters left empty in the last update (centroid kept)
    double h2d_ms = 0;             // copying the data to the GPU
    double init_ms = 0;            // seeding
    double iter_ms = 0;            // all Lloyd iterations + final assignment
    double total_ms = 0;           // init + iterations (data already on the GPU)
    std::vector<double> inertia_history;
    std::vector<double> shift_history;
};

KMeansResult kmeans_gpu(const float* h_data, int64_t n, int d, const KMeansParams& p);

}  // namespace vs
