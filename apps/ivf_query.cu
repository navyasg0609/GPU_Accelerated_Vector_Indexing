// Searches an IVF index for the queries in a file and prints (or saves) the top-k ids.
// Used by python/query_demo.py to answer free-text queries.
#include <cstdio>

#include "vs/cuda_utils.cuh"
#include "vs/io.h"
#include "vs/ivf_index.cuh"

using namespace vs;

int main(int argc, char** argv) {
    Args a(argc, argv, "Query an IVF index.");
    const std::string base_path = a.str("base", "data/base.fbin", "base vectors (.fbin)");
    const int64_t n_lim = a.i64("n", -1, "use the first n base vectors (must match the index)");
    const std::string index_path = a.str("index", "data/ivf.bin", "index from build_ivf");
    const std::string query_path = a.str("queries", "", "query vectors (.fbin)");
    const int k = a.i32("k", 10, "neighbours per query");
    IvfSearchParams sp;
    sp.nprobe = a.i32("nprobe", 16, "lists to scan per query");
    const std::string out_prefix = a.str("out", "", "write <prefix>.ibin/.fbin instead of printing");
    a.finish();
    VS_REQUIRE(!query_path.empty(), "--queries is required");

    IvfLists lists = load_ivf(index_path);
    FloatMatrix q = read_fbin(query_path);
    l2_normalize_rows(q.data.data(), q.rows, q.cols);
    FloatMatrix base = read_fbin(base_path, n_lim);
    VS_REQUIRE(base.rows == lists.n, "index was built on a different number of vectors (check --n)");
    l2_normalize_rows(base.data.data(), base.rows, base.cols);
    IvfIndexGpu index(lists, base.data.data());

    const int nq = (int)q.rows;
    DeviceBuffer<float> dq, dv(size_t(nq) * k);
    DeviceBuffer<int32_t> di(size_t(nq) * k);
    dq.upload(q.data.data(), q.data.size());
    index.search(dq.data(), nq, k, sp, dv.data(), di.data(), 0);
    std::vector<float> vals(size_t(nq) * k);
    std::vector<int32_t> ids(size_t(nq) * k);
    dv.download(vals.data(), vals.size());
    di.download(ids.data(), ids.size());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!out_prefix.empty()) {
        write_ibin(out_prefix + ".ibin", ids.data(), nq, k);
        write_fbin(out_prefix + ".fbin", vals.data(), nq, k);
        return 0;
    }
    for (int i = 0; i < nq; ++i) {
        std::printf("query %d:\n", i);
        for (int r = 0; r < k; ++r)
            std::printf("  %2d  id %-9d  cos %.4f\n", r + 1, ids[size_t(i) * k + r], vals[size_t(i) * k + r]);
    }
    return 0;
}
