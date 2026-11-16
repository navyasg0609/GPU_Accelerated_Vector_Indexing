// Block-level top-k selection (largest values) with radix select, not a full sort.
//
// Each row is split into tiles of kTopkTile values. A block loads one tile into shared memory,
// finds the k-th largest key with a 4-pass, 8-bit radix select (256-bin smem histogram, bin
// choice by a warp-wide prefix scan), gathers the k winners and bitonic-sorts only those k.
// Rows longer than one tile produce tiles*k candidates that are reduced again by the same
// kernel until a single tile per row remains.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "vs/cuda_utils.cuh"

namespace vs {

constexpr int kTopkTile = 4096;
constexpr int kTopkThreads = 256;
constexpr int kTopkMaxK = 1024;

// Reusable intermediate storage; grows on demand and is then reused without reallocation.
struct TopKWorkspace {
    DeviceBuffer<float> vals[2];
    DeviceBuffer<int32_t> ids[2];
};

// Selects the k largest values of each of nrows rows.
//
// Row layout:
//   - d_row_offsets == nullptr: row r is d_vals[r * row_stride, r * row_stride + row_len).
//   - otherwise CSR: row r is [offsets[r] - offsets[0], offsets[r+1] - offsets[0]) and row_len
//     must be the maximum row length. Offsets are relative to offsets[0], so a sub-range of a
//     larger CSR array can be passed directly.
// Ids: d_ids[pos] when d_ids != nullptr, otherwise id_base + (position within the row).
// Output: row r is written to d_out_vals/d_out_ids + r * out_stride, sorted by descending value,
// padded with (-inf, -1) when the row has fewer than k values.
void topk_rows(const float* d_vals, const int32_t* d_ids, const int64_t* d_row_offsets,
               int64_t row_stride, int64_t row_len, int nrows, int k, int32_t id_base,
               float* d_out_vals, int32_t* d_out_ids, int64_t out_stride, TopKWorkspace& ws,
               cudaStream_t stream);

}  // namespace vs
