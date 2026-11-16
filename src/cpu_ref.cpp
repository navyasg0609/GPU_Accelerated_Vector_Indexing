#include "vs/cpu_ref.h"

#include <omp.h>

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>

#include "vs/common.h"

namespace vs {

namespace {

using Entry = std::pair<float, int32_t>;

// Orders entries by descending score, then ascending id.
inline bool better(const Entry& a, const Entry& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
}

// Bounded min-heap (worst entry at the front) holding the k best entries seen so far.
struct TopK {
    int k;
    std::vector<Entry> heap;
    explicit TopK(int k_) : k(k_) { heap.reserve(k_); }
    void push(float s, int32_t id) {
        const Entry e{s, id};
        if ((int)heap.size() < k) {
            heap.push_back(e);
            std::push_heap(heap.begin(), heap.end(), better);
        } else if (better(e, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), better);
            heap.back() = e;
            std::push_heap(heap.begin(), heap.end(), better);
        }
    }
};

inline float dot(const float* a, const float* b, int d) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int j = 0; j < d; ++j) s += a[j] * b[j];
    return s;
}

}  // namespace

void cpu_flat_search(const float* Q, int nq, const float* X, int64_t n, int d, int k,
                     int32_t* out_ids, float* out_vals) {
    constexpr int kQB = 16;      // queries per block: each base block is reused kQB times
    constexpr int64_t kXB = 512; // base rows per block (~0.75 MB at d=384, stays in L2)
    const int nqb = (int)ceil_div(nq, kQB);
    const int threads = omp_get_max_threads();
    const int parts = std::max(1, threads / std::max(1, nqb));
    const int64_t part_len = ceil_div(n, parts);

    // partial[(qb * parts + part) * kQB + j] holds the heap of query qb*kQB+j over one partition.
    std::vector<TopK> partial(static_cast<size_t>(nqb) * parts * kQB, TopK(k));

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int qb = 0; qb < nqb; ++qb) {
        for (int part = 0; part < parts; ++part) {
            const int q0 = qb * kQB, q1 = std::min(nq, q0 + kQB);
            const int64_t x0 = part * part_len, x1 = std::min(n, x0 + part_len);
            TopK* heaps = &partial[(static_cast<size_t>(qb) * parts + part) * kQB];
            for (int64_t xb = x0; xb < x1; xb += kXB) {
                const int64_t xe = std::min(x1, xb + kXB);
                for (int q = q0; q < q1; ++q) {
                    const float* qv = Q + static_cast<int64_t>(q) * d;
                    TopK& h = heaps[q - q0];
                    for (int64_t i = xb; i < xe; ++i) h.push(dot(qv, X + i * d, d), (int32_t)i);
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int q = 0; q < nq; ++q) {
        const int qb = q / kQB, j = q % kQB;
        std::vector<Entry> all;
        for (int part = 0; part < parts; ++part) {
            const auto& h = partial[(static_cast<size_t>(qb) * parts + part) * kQB + j].heap;
            all.insert(all.end(), h.begin(), h.end());
        }
        const int take = std::min<int>(k, (int)all.size());
        std::partial_sort(all.begin(), all.begin() + take, all.end(), better);
        for (int r = 0; r < k; ++r) {
            const bool ok = r < take;
            out_vals[static_cast<int64_t>(q) * k + r] =
                ok ? all[r].first : -std::numeric_limits<float>::infinity();
            out_ids[static_cast<int64_t>(q) * k + r] = ok ? all[r].second : -1;
        }
    }
}

void cpu_inner_products_ref(const float* Q, int nq, const float* X, int64_t nv, int d, double* S) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int q = 0; q < nq; ++q) {
        for (int64_t v = 0; v < nv; ++v) {
            double s = 0;
            for (int j = 0; j < d; ++j) s += double(Q[int64_t(q) * d + j]) * X[v * d + j];
            S[int64_t(q) * nv + v] = s;
        }
    }
}

double recall_at(const int32_t* res, int res_stride, const int32_t* gt, int gt_stride, int nq,
                 int r) {
    VS_REQUIRE(r <= res_stride && r <= gt_stride, "recall@r needs at least r results per query");
    double total = 0;
    for (int q = 0; q < nq; ++q) {
        std::unordered_set<int32_t> truth(gt + int64_t(q) * gt_stride, gt + int64_t(q) * gt_stride + r);
        int hit = 0;
        for (int i = 0; i < r; ++i) hit += truth.count(res[int64_t(q) * res_stride + i]) ? 1 : 0;
        total += double(hit) / r;
    }
    return nq ? total / nq : 0.0;
}

}  // namespace vs
