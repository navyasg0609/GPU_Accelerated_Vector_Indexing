// Exact CPU reference implementations (OpenMP) used as baselines and in the correctness tests.
#pragma once

#include <cstdint>
#include <vector>

namespace vs {

// Exact top-k by inner product of each query against every base vector. Results are sorted by
// descending score (ties by ascending id); rows with fewer than k vectors are padded with
// (-inf, -1). Parallel over query blocks and base partitions, so it also uses all cores when
// nq is small.
void cpu_flat_search(const float* Q, int nq, const float* X, int64_t n, int d, int k,
                     int32_t* out_ids, float* out_vals);

// S[q * nv + v] = <Q_q, X_v> accumulated in double (reference for kernel tests).
void cpu_inner_products_ref(const float* Q, int nq, const float* X, int64_t nv, int d, double* S);

// Mean over queries of |res_q[:r] ∩ gt_q[:r]| / r.
double recall_at(const int32_t* res, int res_stride, const int32_t* gt, int gt_stride, int nq, int r);

}  // namespace vs
