// Correctness tests on small synthetic data (fixed seed). Exit code 0 only if every check passes.
//   1. every distance kernel vs. a double-precision CPU reference
//   2. GPU top-k vs. CPU sort (single tile, multi-tile, CSR rows, ties, rows shorter than k)
//   3. GPU flat search vs. CPU exact search
//   4. GPU K-means: monotone inertia, labels are true nearest centroids, lists well formed
//   5. IVF search vs. a CPU emulation of the same probes, and IVF(nprobe = nlist) == flat
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>

#include "vs/cpu_ref.h"
#include "vs/distance.cuh"
#include "vs/flat_index.cuh"
#include "vs/io.h"
#include "vs/ivf_index.cuh"
#include "vs/kmeans.cuh"
#include "vs/topk.cuh"

using namespace vs;

namespace {

int g_failures = 0;

void report(const char* name, bool ok, const std::string& detail) {
    std::printf("[%s] %-48s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++g_failures;
}

std::vector<float> random_unit(int64_t n, int d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(size_t(n) * d);
    for (auto& x : v) x = nd(rng);
    l2_normalize_rows(v.data(), n, d);
    return v;
}

// Points around `centers` random unit centres (clustered, like real embeddings).
std::vector<float> clustered_unit(int64_t n, int d, int centers, float noise, uint64_t seed) {
    std::vector<float> c = random_unit(centers, d, seed);
    std::mt19937_64 rng(seed + 7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(size_t(n) * d);
    for (int64_t i = 0; i < n; ++i) {
        const int ci = int(rng() % centers);
        for (int j = 0; j < d; ++j) v[size_t(i) * d + j] = c[size_t(ci) * d + j] + noise * nd(rng) / std::sqrt(float(d));
    }
    l2_normalize_rows(v.data(), n, d);
    return v;
}

void test_distance_kernels() {
    const int d = 384, nq = 37;
    const int64_t nv = 3001;  // not a multiple of any tile size
    auto Q = random_unit(nq, d, 1), X = random_unit(nv, d, 2);
    std::vector<double> ref(size_t(nq) * nv);
    cpu_inner_products_ref(Q.data(), nq, X.data(), nv, d, ref.data());
    DeviceBuffer<float> dQ, dX, dS(size_t(nq) * nv);
    dQ.upload(Q.data(), Q.size());
    dX.upload(X.data(), X.size());
    DeviceBuffer<__half> dXh(X.size());
    convert_to_half(dX.data(), dXh.data(), int64_t(X.size()), 0);
    std::vector<float> S(size_t(nq) * nv);

    for (const char* name : {"naive", "warp", "tiled", "cublas", "warp_fp16"}) {
        const DistVariant v = parse_dist_variant(name);
        for (int qn : {1, 3, nq}) {  // exercises the 1-, 4- and 8-query warp kernels
            CUDA_CHECK(cudaMemset(dS.data(), 0, dS.size() * sizeof(float)));
            if (v == DistVariant::WarpFp16) launch_inner_product_fp16(dQ.data(), qn, dXh.data(), nv, d, dS.data(), nv, 128, 0);
            else launch_inner_product(v, dQ.data(), qn, dX.data(), nv, d, dS.data(), nv, 128, 0);
            dS.download(S.data(), size_t(qn) * nv);
            CUDA_CHECK(cudaDeviceSynchronize());
            double err = 0;
            for (size_t i = 0; i < size_t(qn) * nv; ++i) err = std::max(err, std::fabs(S[i] - ref[i]));
            const double tol = v == DistVariant::WarpFp16 ? 5e-3 : 1e-5;
            char buf[96];
            std::snprintf(buf, sizeof(buf), "max |err| = %.2e (tol %.0e)", err, tol);
            const std::string label = std::string("distance ") + name + " nq=" + std::to_string(qn);
            report(label.c_str(), err <= tol, buf);
        }
    }
}

// Checks one top-k result row against the CPU: values must match the sorted input exactly and
// every id must point at an element with that value (ties may pick different ids).
bool check_topk_row(const std::vector<float>& row, const std::vector<int32_t>* row_ids, int k,
                    const float* got_v, const int32_t* got_i, int32_t id_base) {
    std::vector<float> sorted = row;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    std::set<int32_t> seen;
    for (int r = 0; r < k; ++r) {
        if (r >= (int)row.size()) {
            if (got_i[r] != -1 || !(std::isinf(got_v[r]) && got_v[r] < 0)) return false;
            continue;
        }
        if (got_v[r] != sorted[r]) return false;
        if (!seen.insert(got_i[r]).second) return false;
        int64_t pos = -1;
        if (row_ids) {
            for (size_t i = 0; i < row_ids->size(); ++i)
                if ((*row_ids)[i] == got_i[r]) pos = int64_t(i);
        } else {
            pos = got_i[r] - id_base;
        }
        if (pos < 0 || pos >= (int64_t)row.size() || row[pos] != got_v[r]) return false;
    }
    return true;
}

void test_topk() {
    std::mt19937_64 rng(3);
    TopKWorkspace ws;
    // Uniform rows: one tile, several tiles, and multiple reduction passes; with ties.
    for (auto cfg : std::vector<std::pair<int64_t, int>>{{1000, 1}, {1000, 100}, {4096, 100}, {50000, 10},
                                                         {200000, 100}, {300000, 1024}, {77, 100}}) {
        const int64_t len = cfg.first;
        const int k = cfg.second;
        const int rows = 5;
        std::vector<float> v(size_t(rows) * len);
        std::uniform_int_distribution<int> coarse(0, 2000);  // heavy ties
        std::normal_distribution<float> nd(0.f, 1.f);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (i % 2) ? nd(rng) : coarse(rng) / 1000.f - 1.f;
        DeviceBuffer<float> dv, ov(size_t(rows) * k);
        DeviceBuffer<int32_t> oi(size_t(rows) * k);
        dv.upload(v.data(), v.size());
        topk_rows(dv.data(), nullptr, nullptr, len, len, rows, k, 1000, ov.data(), oi.data(), k, ws, 0);
        std::vector<float> gv(size_t(rows) * k);
        std::vector<int32_t> gi(size_t(rows) * k);
        ov.download(gv.data(), gv.size());
        oi.download(gi.data(), gi.size());
        CUDA_CHECK(cudaDeviceSynchronize());
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            std::vector<float> row(v.begin() + size_t(r) * len, v.begin() + size_t(r + 1) * len);
            ok = ok && check_topk_row(row, nullptr, k, gv.data() + size_t(r) * k, gi.data() + size_t(r) * k, 1000);
        }
        const std::string label = "topk uniform len=" + std::to_string(len) + " k=" + std::to_string(k);
        report(label.c_str(), ok, "");
    }
    // CSR rows with explicit ids, including empty and shorter-than-k rows.
    {
        const int k = 50;
        const std::vector<int64_t> lens = {0, 10, 50, 51, 5000, 12345, 3};
        std::vector<int64_t> off(lens.size() + 1, 0);
        for (size_t i = 0; i < lens.size(); ++i) off[i + 1] = off[i] + lens[i];
        std::vector<float> v(off.back());
        std::vector<int32_t> ids(off.back());
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : v) x = nd(rng);
        std::iota(ids.begin(), ids.end(), 500000);
        std::shuffle(ids.begin(), ids.end(), rng);
        DeviceBuffer<float> dv, ov(lens.size() * k);
        DeviceBuffer<int32_t> di, oi(lens.size() * k);
        DeviceBuffer<int64_t> doff;
        dv.upload(v.data(), v.size());
        di.upload(ids.data(), ids.size());
        doff.upload(off.data(), off.size());
        const int64_t maxlen = *std::max_element(lens.begin(), lens.end());
        topk_rows(dv.data(), di.data(), doff.data(), 0, maxlen, (int)lens.size(), k, 0, ov.data(), oi.data(), k, ws, 0);
        std::vector<float> gv(lens.size() * k);
        std::vector<int32_t> gi(lens.size() * k);
        ov.download(gv.data(), gv.size());
        oi.download(gi.data(), gi.size());
        CUDA_CHECK(cudaDeviceSynchronize());
        bool ok = true;
        for (size_t r = 0; r < lens.size(); ++r) {
            std::vector<float> row(v.begin() + off[r], v.begin() + off[r + 1]);
            std::vector<int32_t> rid(ids.begin() + off[r], ids.begin() + off[r + 1]);
            ok = ok && check_topk_row(row, &rid, k, gv.data() + r * k, gi.data() + r * k, 0);
        }
        report("topk CSR rows with ids (empty/short/multi-tile)", ok, "");
    }
}

void test_flat() {
    const int d = 384, nq = 64, k = 100;
    const int64_t n = 30000;
    auto X = clustered_unit(n, d, 50, 0.8f, 4), Q = clustered_unit(nq, d, 50, 0.8f, 5);
    std::vector<int32_t> cpu_i(size_t(nq) * k), gpu_i(size_t(nq) * k);
    std::vector<float> cpu_v(size_t(nq) * k), gpu_v(size_t(nq) * k);
    cpu_flat_search(Q.data(), nq, X.data(), n, d, k, cpu_i.data(), cpu_v.data());
    FlatIndexGpu idx(X.data(), n, d);
    DeviceBuffer<float> dq, dv(gpu_v.size());
    DeviceBuffer<int32_t> di(gpu_i.size());
    dq.upload(Q.data(), Q.size());
    for (const char* vname : {"warp", "tiled", "naive"}) {
        FlatSearchParams fp;
        fp.variant = parse_dist_variant(vname);
        fp.max_score_elems = int64_t(nq) * 8192;  // forces several chunks + a merge
        idx.search(dq.data(), nq, k, fp, dv.data(), di.data(), 0);
        dv.download(gpu_v.data(), gpu_v.size());
        di.download(gpu_i.data(), gpu_i.size());
        CUDA_CHECK(cudaDeviceSynchronize());
        double err = 0;
        for (size_t i = 0; i < gpu_v.size(); ++i) err = std::max(err, (double)std::fabs(gpu_v[i] - cpu_v[i]));
        const double agree = recall_at(gpu_i.data(), k, cpu_i.data(), k, nq, k);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "max |score diff| %.2e, id agreement %.5f", err, agree);
        const std::string label = std::string("flat GPU (") + vname + ") vs CPU exact, chunked";
        report(label.c_str(), err <= 1e-5 && agree >= 0.999, buf);
    }
}

void test_kmeans_and_ivf() {
    const int d = 128, nlist = 32, nq = 50, k = 20;
    const int64_t n = 40000;
    auto X = clustered_unit(n, d, 64, 0.6f, 8), Q = clustered_unit(nq, d, 64, 0.6f, 9);

    for (const char* init : {"kmeans++", "random"}) {
        KMeansParams kp;
        kp.k = nlist;
        kp.max_iter = 30;
        kp.init = init;
        kp.verbose = false;
        kp.chunk_rows = 7000;  // several assignment chunks
        const KMeansResult r = kmeans_gpu(X.data(), n, d, kp);
        bool mono = true;
        for (size_t i = 1; i < r.inertia_history.size(); ++i)
            mono = mono && r.inertia_history[i] <= r.inertia_history[i - 1] * (1 + 1e-5);
        // Labels are the nearest centroid (allowing float ties) and inertia matches.
        int64_t wrong = 0;
        double inertia = 0;
        for (int64_t i = 0; i < n; ++i) {
            double best = 1e30, mine = 0;
            for (int c = 0; c < nlist; ++c) {
                double s = 0;
                for (int j = 0; j < d; ++j) {
                    const double t = double(X[size_t(i) * d + j]) - r.centroids[size_t(c) * d + j];
                    s += t * t;
                }
                best = std::min(best, s);
                if (c == r.labels[i]) mine = s;
            }
            inertia += mine;
            if (mine > best + 1e-4) ++wrong;
        }
        const double rel = std::fabs(inertia - r.inertia) / inertia;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%d iters, inertia %.4f (CPU %.4f), monotone %d, wrong labels %lld",
                      r.iterations, r.inertia, inertia, (int)mono, (long long)wrong);
        const std::string label = std::string("kmeans ") + init;
        report(label.c_str(), mono && wrong == 0 && rel < 1e-4, buf);
    }

    KMeansParams kp;
    kp.k = nlist;
    kp.max_iter = 30;
    kp.verbose = false;
    const KMeansResult km = kmeans_gpu(X.data(), n, d, kp);
    IvfLists lists = build_ivf_lists(km.centroids, nlist, d, km.labels);
    {
        bool ok = lists.list_offsets.back() == n;
        std::vector<int> seen(n, 0);
        for (int32_t id : lists.list_ids) ok = ok && id >= 0 && id < n && seen[id]++ == 0;
        report("ivf lists are a partition of the ids", ok, "");
    }
    save_ivf("test_ivf.tmp", lists);
    const IvfLists loaded = load_ivf("test_ivf.tmp");
    std::remove("test_ivf.tmp");
    report("ivf save/load round trip",
           loaded.list_ids == lists.list_ids && loaded.list_offsets == lists.list_offsets &&
               loaded.centroids == lists.centroids,
           "");

    IvfIndexGpu ivf(lists, X.data());
    DeviceBuffer<float> dq, dv(size_t(nq) * k);
    DeviceBuffer<int32_t> di(size_t(nq) * k);
    dq.upload(Q.data(), Q.size());
    std::vector<float> gv(size_t(nq) * k);
    std::vector<int32_t> gi(size_t(nq) * k);

    // CPU emulation: rank centroids by L2, scan the top nprobe lists exactly.
    for (int nprobe : {1, 4, 13}) {
        IvfSearchParams sp;
        sp.nprobe = nprobe;
        sp.max_candidates = 20000;  // forces sub-batching
        ivf.search(dq.data(), nq, k, sp, dv.data(), di.data(), 0);
        dv.download(gv.data(), gv.size());
        di.download(gi.data(), gi.size());
        CUDA_CHECK(cudaDeviceSynchronize());
        double err = 0;
        double agree = 0;
        for (int q = 0; q < nq; ++q) {
            std::vector<std::pair<double, int>> cd;
            for (int c = 0; c < nlist; ++c) {
                double s = 0;
                for (int j = 0; j < d; ++j) {
                    const double t = double(Q[size_t(q) * d + j]) - km.centroids[size_t(c) * d + j];
                    s += t * t;
                }
                cd.push_back({s, c});
            }
            std::sort(cd.begin(), cd.end());
            std::vector<std::pair<float, int32_t>> cand;
            for (int p = 0; p < nprobe; ++p) {
                const int c = cd[p].second;
                for (int64_t i = lists.list_offsets[c]; i < lists.list_offsets[c + 1]; ++i) {
                    const int32_t id = lists.list_ids[i];
                    double s = 0;
                    for (int j = 0; j < d; ++j) s += double(Q[size_t(q) * d + j]) * X[size_t(id) * d + j];
                    cand.push_back({float(s), id});
                }
            }
            std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            std::set<int32_t> truth;
            for (int r = 0; r < k && r < (int)cand.size(); ++r) {
                truth.insert(cand[r].second);
                err = std::max(err, (double)std::fabs(cand[r].first - gv[size_t(q) * k + r]));
            }
            int hit = 0;
            for (int r = 0; r < k; ++r) hit += truth.count(gi[size_t(q) * k + r]) ? 1 : 0;
            agree += double(hit) / k;
        }
        agree /= nq;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "max |score diff| %.2e, id agreement %.5f", err, agree);
        const std::string label = "ivf nprobe=" + std::to_string(nprobe) + " vs CPU emulation";
        report(label.c_str(), err <= 1e-5 && agree >= 0.995, buf);
    }

    // nprobe = nlist scans everything, so it must equal exact search.
    {
        IvfSearchParams sp;
        sp.nprobe = nlist;
        ivf.search(dq.data(), nq, k, sp, dv.data(), di.data(), 0);
        di.download(gi.data(), gi.size());
        dv.download(gv.data(), gv.size());
        std::vector<int32_t> ci(size_t(nq) * k);
        std::vector<float> cv(size_t(nq) * k);
        CUDA_CHECK(cudaDeviceSynchronize());
        cpu_flat_search(Q.data(), nq, X.data(), n, d, k, ci.data(), cv.data());
        double err = 0;
        for (size_t i = 0; i < cv.size(); ++i) err = std::max(err, (double)std::fabs(cv[i] - gv[i]));
        const double agree = recall_at(gi.data(), k, ci.data(), k, nq, k);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "max |score diff| %.2e, recall@%d %.5f", err, k, agree);
        report("ivf nprobe=nlist equals exact search", err <= 1e-5 && agree >= 0.999, buf);
    }
}

}  // namespace

int main() {
    print_device_info(query_device_info(0));
    test_distance_kernels();
    test_topk();
    test_flat();
    test_kmeans_and_ivf();
    std::printf("\n%s (%d failure%s)\n", g_failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
