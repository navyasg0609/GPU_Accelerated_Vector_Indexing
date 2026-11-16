// Trains the IVF coarse quantizer with GPU K-means and writes the index (centroids + lists).
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "vs/bench.cuh"
#include "vs/io.h"
#include "vs/ivf_index.cuh"
#include "vs/kmeans.cuh"

using namespace vs;

int main(int argc, char** argv) {
    Args a(argc, argv, "Train IVF centroids with CUDA K-means and build the inverted lists.");
    const std::string base_path = a.str("base", "data/base.fbin", "base vectors (.fbin)");
    const int64_t n_lim = a.i64("n", -1, "use the first n base vectors (-1 = all)");
    KMeansParams kp;
    kp.k = a.i32("nlist", 128, "number of clusters (inverted lists)");
    kp.max_iter = a.i32("max_iter", 100, "maximum Lloyd iterations");
    kp.tol = a.f64("tol", 1e-4, "relative tolerance on the centroid shift (scikit-learn definition)");
    kp.seed = (uint64_t)a.i64("seed", 42, "random seed");
    kp.init = a.str("init", "kmeans++", "seeding: kmeans++ | random");
    kp.variant = parse_dist_variant(a.str("variant", "tiled", "distance kernel for the assignment GEMM"));
    kp.threads_per_block = a.i32("tpb", 256, "threads per block for the distance kernel");
    kp.chunk_rows = a.i64("chunk_rows", 262144, "rows per assignment GEMM");
    kp.verbose = a.flag("verbose", false, "print every iteration");
    const int repeats = a.i32("runs", 3, "train this many times (same seed) for timing statistics");
    const std::string out_path = a.str("out", "data/ivf.bin", "output index file");
    const std::string centroids_out = a.str("centroids_out", "", "also write centroids as .fbin");
    const std::string csv = a.str("csv", "results/kmeans_gpu.csv", "timing/quality CSV (appended)");
    const std::string dataset = a.str("dataset", "unnamed", "dataset label for the CSV");
    const bool normalize = a.flag("normalize", true, "L2-normalize vectors first");
    a.finish();

    const DeviceInfo dev = query_device_info(0);
    print_device_info(dev);
    FloatMatrix base = read_fbin(base_path, n_lim);
    if (normalize) l2_normalize_rows(base.data.data(), base.rows, base.cols);
    std::printf("dataset %s: n=%lld d=%d, k=%d, init=%s, variant=%s, max_iter=%d, tol=%g\n",
                dataset.c_str(), (long long)base.rows, base.cols, kp.k, kp.init.c_str(),
                dist_variant_name(kp.variant), kp.max_iter, kp.tol);

    KMeansResult res;
    std::vector<double> total_ms, iter_ms, per_iter_ms;
    for (int r = 0; r < std::max(1, repeats); ++r) {
        kp.verbose = kp.verbose && r == 0;
        res = kmeans_gpu(base.data.data(), base.rows, base.cols, kp);
        total_ms.push_back(res.total_ms);
        iter_ms.push_back(res.iter_ms);
        per_iter_ms.push_back(res.iter_ms / (res.iterations + 1));  // + final assignment
        std::printf("run %d: %d iterations (%s), inertia %.6e, init %.1f ms, iterations %.1f ms, total %.1f ms (H2D %.1f ms)\n",
                    r, res.iterations, res.converged ? "converged" : "max_iter", res.inertia,
                    res.init_ms, res.iter_ms, res.total_ms, res.h2d_ms);
    }

    IvfLists lists = build_ivf_lists(res.centroids, kp.k, base.cols, res.labels);
    int64_t mn = INT64_MAX, mx = 0;
    for (int c = 0; c < kp.k; ++c) {
        const int64_t sz = lists.list_offsets[c + 1] - lists.list_offsets[c];
        mn = std::min(mn, sz);
        mx = std::max(mx, sz);
    }
    save_ivf(out_path, lists);
    if (!centroids_out.empty()) write_fbin(centroids_out, res.centroids.data(), kp.k, base.cols);
    std::printf("wrote %s (list sizes min %lld, max %lld, mean %.1f)\n", out_path.c_str(),
                (long long)mn, (long long)mx, double(base.rows) / kp.k);

    const Stats ts = compute_stats(total_ms), is = compute_stats(per_iter_ms);
    RunMeta meta{dev, dataset, base.rows, base.cols, 0};
    CsvWriter cw(csv, concat(RunMeta::header(),
                             {"method", "init", "variant", "nlist", "max_iter", "tol", "seed", "runs",
                              "iterations", "converged", "inertia", "total_ms_mean", "total_ms_std",
                              "ms_per_iter_mean", "ms_per_iter_std", "init_ms", "h2d_ms",
                              "empty_clusters", "min_list", "max_list"}));
    cw.row(concat(meta.values(),
                  {"ours_cuda_kmeans", kp.init, dist_variant_name(kp.variant), std::to_string(kp.k),
                   std::to_string(kp.max_iter), fmt(kp.tol), std::to_string(kp.seed),
                   std::to_string(total_ms.size()), std::to_string(res.iterations),
                   res.converged ? "1" : "0", fmt(res.inertia, 10), fmt(ts.mean), fmt(ts.std),
                   fmt(is.mean), fmt(is.std), fmt(res.init_ms), fmt(res.h2d_ms),
                   std::to_string(res.empty_clusters), std::to_string(mn), std::to_string(mx)}));
    return 0;
}
