#include "vs/io.h"

#include <cmath>
#include <cstdio>
#include <filesystem>

#include "vs/common.h"

namespace vs {

namespace {

template <typename T>
Matrix<T> read_bin(const std::string& path, int64_t max_rows) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "Could not open %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    uint32_t header[2];
    VS_REQUIRE(std::fread(header, sizeof(uint32_t), 2, f) == 2, "truncated file header");
    Matrix<T> m;
    m.rows = header[0];
    m.cols = static_cast<int>(header[1]);
    if (max_rows >= 0 && max_rows < m.rows) m.rows = max_rows;
    m.data.resize(static_cast<size_t>(m.rows) * m.cols);
    const size_t want = m.data.size();
    VS_REQUIRE(std::fread(m.data.data(), sizeof(T), want, f) == want, "file shorter than header says");
    std::fclose(f);
    return m;
}

template <typename T>
void write_bin(const std::string& path, const T* data, int64_t rows, int cols) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    FILE* f = std::fopen(path.c_str(), "wb");
    VS_REQUIRE(f != nullptr, "could not open output file");
    VS_REQUIRE(rows <= UINT32_MAX, "too many rows for the .bin format");
    const uint32_t header[2] = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
    VS_REQUIRE(std::fwrite(header, sizeof(uint32_t), 2, f) == 2, "write failed");
    const size_t count = static_cast<size_t>(rows) * cols;
    VS_REQUIRE(std::fwrite(data, sizeof(T), count, f) == count, "write failed");
    std::fclose(f);
}

}  // namespace

FloatMatrix read_fbin(const std::string& path, int64_t max_rows) {
    return read_bin<float>(path, max_rows);
}
IntMatrix read_ibin(const std::string& path, int64_t max_rows) {
    return read_bin<int32_t>(path, max_rows);
}
void write_fbin(const std::string& path, const float* data, int64_t rows, int cols) {
    write_bin(path, data, rows, cols);
}
void write_ibin(const std::string& path, const int32_t* data, int64_t rows, int cols) {
    write_bin(path, data, rows, cols);
}

void l2_normalize_rows(float* data, int64_t rows, int cols) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        float* r = data + i * cols;
        double s = 0;
        for (int j = 0; j < cols; ++j) s += double(r[j]) * r[j];
        const float inv = s > 0 ? float(1.0 / std::sqrt(s)) : 0.f;
        for (int j = 0; j < cols; ++j) r[j] *= inv;
    }
}

}  // namespace vs
