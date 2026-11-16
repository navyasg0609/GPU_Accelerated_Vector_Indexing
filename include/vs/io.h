// Binary dataset files.
//
// .fbin / .ibin layout (same as the big-ann-benchmarks format):
//   uint32 rows, uint32 cols, then rows*cols little-endian float32 (.fbin) or int32 (.ibin).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vs {

template <typename T>
struct Matrix {
    int64_t rows = 0;
    int cols = 0;
    std::vector<T> data;

    T* row(int64_t i) { return data.data() + i * cols; }
    const T* row(int64_t i) const { return data.data() + i * cols; }
};

using FloatMatrix = Matrix<float>;
using IntMatrix = Matrix<int32_t>;

// Reads at most max_rows rows (all rows when max_rows < 0).
FloatMatrix read_fbin(const std::string& path, int64_t max_rows = -1);
IntMatrix read_ibin(const std::string& path, int64_t max_rows = -1);
void write_fbin(const std::string& path, const float* data, int64_t rows, int cols);
void write_ibin(const std::string& path, const int32_t* data, int64_t rows, int cols);

// Scales each row to unit L2 norm so that inner product equals cosine similarity.
void l2_normalize_rows(float* data, int64_t rows, int cols);

}  // namespace vs
