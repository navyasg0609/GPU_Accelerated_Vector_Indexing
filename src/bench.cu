#include "vs/bench.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "vs/cpu_ref.h"

namespace vs {

std::vector<std::string> search_csv_header() {
    return concat(RunMeta::header(),
                  {"method", "variant", "tpb", "nlist", "nprobe", "k", "batch", "runs",
                   "latency_ms_mean", "latency_ms_std", "qps_mean", "qps_std", "recall_at_1",
                   "recall_at_10", "recall_at_100"});
}

std::vector<std::string> search_csv_values(const std::string& method, const std::string& variant,
                                           int tpb, int nlist, int nprobe, int k, int batch,
                                           int runs, const Stats& lat, const Stats& qps, double r1,
                                           double r10, double r100) {
    auto opt = [](int v) { return v > 0 ? std::to_string(v) : std::string(); };
    return {method,         variant,       opt(tpb),       opt(nlist),   opt(nprobe),
            std::to_string(k), std::to_string(batch), std::to_string(runs), fmt(lat.mean),
            fmt(lat.std),   fmt(qps.mean), fmt(qps.std),   fmt(r1),      fmt(r10),
            fmt(r100)};
}

BatchTiming time_batched(const float* h_queries, int nq, int d, int batch, int k, int warmup,
                         int runs, const SearchFn& search, cudaStream_t stream) {
    VS_REQUIRE(batch >= 1 && nq >= 1, "batch and nq must be positive");
    const int nbatches = (int)ceil_div(nq, batch);
    PinnedBuffer<float> h_q(size_t(batch) * d);
    PinnedBuffer<float> h_vals(size_t(batch) * k);
    PinnedBuffer<int32_t> h_ids(size_t(batch) * k);
    DeviceBuffer<float> d_q(size_t(batch) * d), d_vals(size_t(batch) * k);
    DeviceBuffer<int32_t> d_ids(size_t(batch) * k);
    GpuTimer timer;

    BatchTiming out;
    out.ids.assign(size_t(nq) * k, -1);
    std::vector<double> lat, qps;
    for (int run = 0; run < warmup + runs; ++run) {
        const bool timed = run >= warmup;
        double run_ms = 0;
        for (int b = 0; b < nbatches; ++b) {
            for (int j = 0; j < batch; ++j) {
                const int q = (b * batch + j) % nq;
                std::memcpy(h_q.data() + size_t(j) * d, h_queries + size_t(q) * d, d * sizeof(float));
            }
            timer.start(stream);
            CUDA_CHECK(cudaMemcpyAsync(d_q.data(), h_q.data(), size_t(batch) * d * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            search(d_q.data(), batch, d_vals.data(), d_ids.data(), stream);
            CUDA_CHECK(cudaMemcpyAsync(h_vals.data(), d_vals.data(), size_t(batch) * k * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_ids.data(), d_ids.data(), size_t(batch) * k * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            const double ms = timer.stop(stream);
            if (!timed) continue;
            lat.push_back(ms);
            run_ms += ms;
            if (run == warmup) {
                for (int j = 0; j < batch; ++j) {
                    const int64_t q = int64_t(b) * batch + j;
                    if (q >= nq) break;
                    std::memcpy(out.ids.data() + q * k, h_ids.data() + size_t(j) * k,
                                k * sizeof(int32_t));
                }
            }
        }
        if (timed) qps.push_back(double(nbatches) * batch / (run_ms / 1e3));
    }
    out.latency_ms = compute_stats(lat);
    out.qps = compute_stats(qps);
    return out;
}

void recall_triplet(const std::vector<int32_t>& ids, int k, const int32_t* gt, int gt_k, int nq,
                    double& r1, double& r10, double& r100) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int lim = std::min(k, gt_k);
    r1 = lim >= 1 ? recall_at(ids.data(), k, gt, gt_k, nq, 1) : nan;
    r10 = lim >= 10 ? recall_at(ids.data(), k, gt, gt_k, nq, 10) : nan;
    r100 = lim >= 100 ? recall_at(ids.data(), k, gt, gt_k, nq, 100) : nan;
}

}  // namespace vs
