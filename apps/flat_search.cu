// Exact brute-force search on the GPU and CPU.
//   1. computes ground-truth top-k for all queries on the GPU and writes <gt_prefix>.ibin/.fbin
//   2. checks the GPU results against the exact CPU search on a subset of queries
//   3. benchmarks GPU flat search per batch size and the CPU baseline
#include <cmath>
#include <cstdio>

#include "vs/bench.cuh"
#include "vs/cpu_ref.h"
#include "vs/flat_index.cuh"
#include "vs/io.h"

using namespace vs;

int main(int argc, char** argv) {
    Args a(argc, argv, "Exact (brute-force) GPU and CPU search: ground truth + baselines.");
    const std::string base_path = a.str("base", "data/base.fbin", "base vectors (.fbin)");
    const std::string query_path = a.str("queries", "data/queries.fbin", "query vectors (.fbin)");
    const int64_t n_lim = a.i64("n", -1, "use the first n base vectors (-1 = all)");
    const int nq_lim = a.i32("nq", -1, "use the first nq queries (-1 = all)");
    const int k = a.i32("k", 100, "neighbours per query");
    const std::string gt_prefix = a.str("gt_out", "", "write ground truth to <prefix>.ibin/.fbin");
    const DistVariant variant = parse_dist_variant(a.str("variant", "warp", "distance kernel"));
    const int tpb = a.i32("tpb", 256, "threads per block for the distance kernel");
    const std::vector<int> batches = a.int_list("batches", "1,16,128,1024", "GPU batch sizes to benchmark");
    const int warmup = a.i32("warmup", 2, "untimed passes before timing");
    const int runs = a.i32("runs", 5, "timed passes");
    const int cpu_check = a.i32("cpu_check", 100, "queries to verify against the CPU (0 = skip)");
    const int cpu_runs = a.i32("cpu_runs", 1, "timed passes of the CPU baseline");
    const std::string csv = a.str("csv", "results/flat_search.csv", "benchmark CSV (appended)");
    const std::string check_csv = a.str("check_csv", "results/correctness_flat.csv", "GPU-vs-CPU check CSV");
    const std::string dataset = a.str("dataset", "unnamed", "dataset label for the CSV");
    const bool normalize = a.flag("normalize", true, "L2-normalize vectors (cosine similarity)");
    a.finish();

    const DeviceInfo dev = query_device_info(0);
    print_device_info(dev);

    FloatMatrix base = read_fbin(base_path, n_lim);
    FloatMatrix queries = read_fbin(query_path, nq_lim);
    VS_REQUIRE(base.cols == queries.cols, "base and query dimensions differ");
    if (normalize) {
        l2_normalize_rows(base.data.data(), base.rows, base.cols);
        l2_normalize_rows(queries.data.data(), queries.rows, queries.cols);
    }
    const int d = base.cols;
    const int nq = (int)queries.rows;
    std::printf("dataset %s: n=%lld d=%d nq=%d k=%d\n", dataset.c_str(), (long long)base.rows, d, nq, k);
    RunMeta meta{dev, dataset, base.rows, d, nq};

    FlatIndexGpu index(base.data.data(), base.rows, d);
    FlatSearchParams fp;
    fp.variant = variant;
    fp.threads_per_block = tpb;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    SearchFn search = [&](const float* dq, int b, float* dv, int32_t* di, cudaStream_t s) {
        index.search(dq, b, k, fp, dv, di, s);
    };

    // ---- 1. ground truth ----
    std::vector<int32_t> gt_ids(size_t(nq) * k);
    std::vector<float> gt_vals(size_t(nq) * k);
    {
        const int gb = std::min(nq, 1024);
        DeviceBuffer<float> dq(size_t(gb) * d), dv(size_t(gb) * k);
        DeviceBuffer<int32_t> di(size_t(gb) * k);
        HostTimer t;
        for (int q0 = 0; q0 < nq; q0 += gb) {
            const int b = std::min(gb, nq - q0);
            dq.upload(queries.row(q0), size_t(b) * d, stream);
            index.search(dq.data(), b, k, fp, dv.data(), di.data(), stream);
            dv.download(gt_vals.data() + size_t(q0) * k, size_t(b) * k, stream);
            di.download(gt_ids.data() + size_t(q0) * k, size_t(b) * k, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("ground truth for %d queries computed in %.1f ms (incl. first-call setup)\n", nq, t.ms());
        if (!gt_prefix.empty()) {
            write_ibin(gt_prefix + ".ibin", gt_ids.data(), nq, k);
            write_fbin(gt_prefix + ".fbin", gt_vals.data(), nq, k);
            std::printf("wrote %s.ibin / .fbin\n", gt_prefix.c_str());
        }
    }

    // ---- 2. GPU vs CPU exact check + CPU baseline timing ----
    if (cpu_check > 0) {
        const int nc = std::min(cpu_check, nq);
        std::vector<int32_t> cpu_ids(size_t(nc) * k);
        std::vector<float> cpu_vals(size_t(nc) * k);
        std::vector<double> cpu_ms;
        for (int r = 0; r < std::max(1, cpu_runs); ++r) {
            HostTimer t;
            cpu_flat_search(queries.data.data(), nc, base.data.data(), base.rows, d, k,
                            cpu_ids.data(), cpu_vals.data());
            cpu_ms.push_back(t.ms());
        }
        double max_diff = 0;
        for (int q = 0; q < nc; ++q)
            for (int r = 0; r < k; ++r)
                if (std::isfinite(cpu_vals[size_t(q) * k + r]))
                    max_diff = std::max(max_diff, (double)std::fabs(cpu_vals[size_t(q) * k + r] -
                                                                    gt_vals[size_t(q) * k + r]));
        const double agree = recall_at(gt_ids.data(), k, cpu_ids.data(), k, nc, k);
        const bool pass = max_diff <= 1e-4 && agree >= 0.999;
        std::printf("GPU vs CPU exact on %d queries: max |score diff| = %.3g, id agreement@%d = %.5f -> %s\n",
                    nc, max_diff, k, agree, pass ? "PASS" : "FAIL");
        CsvWriter cw(check_csv, concat(RunMeta::header(),
                                       {"check", "queries", "k", "max_abs_score_diff", "id_agreement", "pass"}));
        cw.row(concat(meta.values(), {"flat_gpu_vs_cpu", std::to_string(nc), std::to_string(k),
                                      fmt(max_diff), fmt(agree), pass ? "1" : "0"}));

        CsvWriter out(csv, search_csv_header());
        const Stats lat = compute_stats(cpu_ms);
        std::vector<double> qps;
        for (double ms : cpu_ms) qps.push_back(nc / (ms / 1e3));
        RunMeta m2 = meta;
        m2.nq = nc;
        out.row(concat(m2.values(), search_csv_values("ours_flat_cpu", "omp", 0, 0, 0, k, nc,
                                                      (int)cpu_ms.size(), lat, compute_stats(qps),
                                                      1.0, 1.0, k >= 100 ? 1.0 : NAN)));
        std::printf("CPU flat: %d queries in %.1f ms -> %.1f QPS\n", nc, lat.mean, compute_stats(qps).mean);
    }

    // ---- 3. GPU benchmark per batch size ----
    CsvWriter out(csv, search_csv_header());
    std::printf("\n%-8s %14s %14s %12s\n", "batch", "latency ms", "std", "QPS");
    for (int b : batches) {
        const BatchTiming bt = time_batched(queries.data.data(), nq, d, b, k, warmup, runs, search, stream);
        double r1, r10, r100;
        recall_triplet(bt.ids, k, gt_ids.data(), k, nq, r1, r10, r100);
        out.row(concat(meta.values(), search_csv_values("ours_flat_gpu", dist_variant_name(variant), tpb,
                                                        0, 0, k, b, runs, bt.latency_ms, bt.qps, r1, r10, r100)));
        std::printf("%-8d %14.4f %14.4f %12.1f\n", b, bt.latency_ms.mean, bt.latency_ms.std, bt.qps.mean);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
