// Distance-kernel comparison: time, GFLOP/s, achieved DRAM bandwidth vs. theoretical peak,
// occupancy and register use for each variant, batch size and threads-per-block setting.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "vs/bench.cuh"
#include "vs/distance.cuh"
#include "vs/io.h"

using namespace vs;

int main(int argc, char** argv) {
    Args a(argc, argv, "Compare inner-product kernel variants.");
    const std::string base_path = a.str("base", "", "optional base vectors (.fbin); random if empty");
    const int64_t n = a.i64("n", 1000000, "number of vectors");
    const int d = a.i32("d", 384, "dimension (random data only)");
    const std::vector<int> batches = a.int_list("batches", "1,16,128,1024", "query batch sizes");
    const std::vector<std::string> variants =
        a.str_list("variants", "naive,warp,tiled,cublas,warp_fp16", "kernel variants");
    const std::vector<int> tpbs = a.int_list("tpb", "32,64,128,256,512,1024", "threads per block sweep");
    const int warmup = a.i32("warmup", 3, "untimed launches");
    const int runs = a.i32("runs", 20, "timed launches (reduced automatically for slow configs)");
    const int64_t max_score_bytes = a.i64("max_score_mb", 1024, "cap on the score matrix size (MB)") << 20;
    const uint64_t seed = (uint64_t)a.i64("seed", 42, "random seed");
    const std::string csv = a.str("csv", "results/kernel_bench.csv", "output CSV (appended)");
    const std::string dataset = a.str("dataset", "random", "dataset label");
    a.finish();

    const DeviceInfo dev = query_device_info(0);
    print_device_info(dev);

    FloatMatrix X;
    if (!base_path.empty()) {
        X = read_fbin(base_path, n);
    } else {
        X.rows = n;
        X.cols = d;
        X.data.resize(size_t(n) * d);
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : X.data) v = nd(rng);
    }
    l2_normalize_rows(X.data.data(), X.rows, X.cols);
    const int dim = X.cols;
    const int max_b = *std::max_element(batches.begin(), batches.end());
    std::vector<float> Q(size_t(max_b) * dim);
    {
        std::mt19937_64 rng(seed + 1);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : Q) v = nd(rng);
        l2_normalize_rows(Q.data(), max_b, dim);
    }

    DeviceBuffer<float> dX, dQ, dS;
    dX.upload(X.data.data(), X.data.size());
    dQ.upload(Q.data(), Q.size());
    DeviceBuffer<__half> dXh(X.data.size());
    convert_to_half(dX.data(), dXh.data(), int64_t(X.data.size()), 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    RunMeta meta{dev, dataset, X.rows, dim, 0};
    CsvWriter out(csv, concat(RunMeta::header(),
                              {"variant", "batch", "nv", "tpb", "runs", "ms_mean", "ms_std", "gflops",
                               "gbps", "pct_peak_bw", "peak_bw_gbs", "pct_peak_flops", "regs",
                               "smem_bytes", "blocks_per_sm", "occupancy", "max_abs_err"}));
    GpuTimer timer;
    std::mt19937_64 pick(seed + 2);

    std::printf("\n%-10s %6s %8s %5s %10s %9s %9s %8s %8s %7s %10s\n", "variant", "batch", "nv", "tpb",
                "ms", "std", "GFLOP/s", "GB/s", "%peak", "occ", "max err");
    for (const std::string& vname : variants) {
        const DistVariant v = parse_dist_variant(vname);
        const bool fp16 = v == DistVariant::WarpFp16;
        for (int b : batches) {
            const int64_t nv = std::min<int64_t>(X.rows, std::max<int64_t>(1, max_score_bytes / (4LL * b)));
            dS.ensure(size_t(b) * nv);
            const std::vector<int> tpb_list = variant_uses_tpb(v) ? tpbs : std::vector<int>{256};
            for (int tpb : tpb_list) {
                auto launch = [&] {
                    if (fp16) launch_inner_product_fp16(dQ.data(), b, dXh.data(), nv, dim, dS.data(), nv, tpb, 0);
                    else launch_inner_product(v, dQ.data(), b, dX.data(), nv, dim, dS.data(), nv, tpb, 0);
                };
                for (int i = 0; i < warmup; ++i) launch();
                timer.start();
                launch();
                const double first_ms = timer.stop();
                const int eff_runs = std::max(3, std::min(runs, int(2000.0 / std::max(first_ms, 1e-3))));
                std::vector<double> ms;
                for (int i = 0; i < eff_runs; ++i) {
                    timer.start();
                    launch();
                    ms.push_back(timer.stop());
                }
                const Stats st = compute_stats(ms);

                // Spot-check 256 random entries against a double-precision CPU dot product.
                double max_err = 0;
                for (int t = 0; t < 256; ++t) {
                    const int q = int(pick() % b);
                    const int64_t i = int64_t(pick() % uint64_t(nv));
                    float got = 0;
                    CUDA_CHECK(cudaMemcpy(&got, dS.data() + size_t(q) * nv + i, sizeof(float), cudaMemcpyDeviceToHost));
                    double ref = 0;
                    for (int j = 0; j < dim; ++j) ref += double(Q[size_t(q) * dim + j]) * X.data[size_t(i) * dim + j];
                    max_err = std::max(max_err, std::fabs(ref - got));
                }

                // Compulsory DRAM traffic: read X once, read Q, write S.
                const double bytes = double(nv) * dim * (fp16 ? 2 : 4) + double(b) * dim * 4 + double(b) * nv * 4;
                const double flops = 2.0 * b * double(nv) * dim;
                const double gbps = bytes / (st.mean * 1e-3) / 1e9;
                const double gflops = flops / (st.mean * 1e-3) / 1e9;
                const KernelAttrs ka = inner_product_kernel_attrs(v, b, dim, tpb);
                const bool has_attrs = v != DistVariant::Cublas;
                out.row(concat(meta.values(),
                               {vname, std::to_string(b), std::to_string(nv),
                                has_attrs ? std::to_string(ka.threads_per_block) : "", std::to_string(eff_runs),
                                fmt(st.mean), fmt(st.std), fmt(gflops), fmt(gbps),
                                fmt(100.0 * gbps / dev.peak_bw_gbs), fmt(dev.peak_bw_gbs),
                                fmt(100.0 * gflops / dev.peak_fp32_gflops),
                                has_attrs ? std::to_string(ka.regs_per_thread) : "",
                                has_attrs ? std::to_string(ka.static_smem + ka.dynamic_smem) : "",
                                has_attrs ? std::to_string(ka.active_blocks_per_sm) : "",
                                has_attrs ? fmt(ka.occupancy) : "", fmt(max_err)}));
                std::printf("%-10s %6d %8lld %5s %10.4f %9.4f %9.1f %8.1f %8.1f %7s %10.2e\n", vname.c_str(), b,
                            (long long)nv, has_attrs ? std::to_string(ka.threads_per_block).c_str() : "-",
                            st.mean, st.std, gflops, gbps, 100.0 * gbps / dev.peak_bw_gbs,
                            has_attrs ? fmt(ka.occupancy, 3).c_str() : "-", max_err);
            }
        }
    }
    return 0;
}
