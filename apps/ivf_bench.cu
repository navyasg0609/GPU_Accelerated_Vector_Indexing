// IVF search benchmark: recall@{1,10,100} vs. exact ground truth, latency and QPS, swept over
// nprobe and batch size.
#include <cstdio>
#include <memory>

#include "vs/bench.cuh"
#include "vs/io.h"
#include "vs/ivf_index.cuh"

using namespace vs;

int main(int argc, char** argv) {
    Args a(argc, argv, "Benchmark batched IVF search: recall, latency and QPS.");
    const std::string base_path = a.str("base", "data/base.fbin", "base vectors (.fbin)");
    const int64_t n_lim = a.i64("n", -1, "use the first n base vectors (must match the index)");
    const std::string index_path = a.str("index", "data/ivf.bin", "index from build_ivf");
    const std::string query_path = a.str("queries", "data/queries.fbin", "query vectors (.fbin)");
    const int nq_lim = a.i32("nq", -1, "use the first nq queries (-1 = all)");
    const std::string gt_path = a.str("gt", "data/gt.ibin", "ground-truth ids (.ibin) from flat_search");
    const int k = a.i32("k", 100, "neighbours per query");
    const std::vector<int> nprobes = a.int_list("nprobe", "1,2,4,8,16,32,64,128", "nprobe values");
    const std::vector<int> batches = a.int_list("batches", "1,16,128,1024", "batch sizes");
    const int warmup = a.i32("warmup", 2, "untimed passes before timing");
    const int runs = a.i32("runs", 5, "timed passes");
    IvfSearchParams sp;
    sp.coarse_variant = parse_dist_variant(a.str("coarse_variant", "warp", "kernel for query x centroid scores"));
    sp.coarse_tpb = a.i32("coarse_tpb", 256, "threads per block, coarse kernel");
    sp.scan_tpb = a.i32("tpb", 256, "threads per block, list-scan kernel");
    sp.max_candidates = a.i64("max_candidates", int64_t(128) << 20, "candidate buffer budget (elements)");
    const std::string csv = a.str("csv", "results/ivf_search.csv", "output CSV (appended)");
    const std::string dataset = a.str("dataset", "unnamed", "dataset label for the CSV");
    const bool normalize = a.flag("normalize", true, "L2-normalize vectors");
    a.finish();

    const DeviceInfo dev = query_device_info(0);
    print_device_info(dev);

    IvfLists lists = load_ivf(index_path);
    FloatMatrix queries = read_fbin(query_path, nq_lim);
    IntMatrix gt = read_ibin(gt_path, nq_lim);
    VS_REQUIRE(gt.rows == queries.rows, "ground truth and queries have different row counts");
    if (normalize) l2_normalize_rows(queries.data.data(), queries.rows, queries.cols);
    const int nq = (int)queries.rows;
    const int d = queries.cols;

    std::unique_ptr<IvfIndexGpu> index;
    {
        FloatMatrix base = read_fbin(base_path, n_lim);
        VS_REQUIRE(base.rows == lists.n, "index was built on a different number of vectors (check --n)");
        VS_REQUIRE(base.cols == d && lists.d == d, "dimension mismatch");
        if (normalize) l2_normalize_rows(base.data.data(), base.rows, base.cols);
        index = std::make_unique<IvfIndexGpu>(lists, base.data.data());
    }  // host copy of the base released here
    std::printf("dataset %s: n=%lld d=%d nq=%d nlist=%d (largest list %lld), k=%d\n", dataset.c_str(),
                (long long)lists.n, d, nq, lists.nlist, (long long)index->max_list_size(), k);

    RunMeta meta{dev, dataset, lists.n, d, nq};
    CsvWriter out(csv, search_csv_header());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::printf("\n%6s %7s %9s %9s %9s %12s %10s %12s\n", "batch", "nprobe", "R@1", "R@10", "R@100",
                "latency ms", "std", "QPS");
    for (int b : batches) {
        for (int np : nprobes) {
            if (np > lists.nlist) continue;
            sp.nprobe = np;
            SearchFn search = [&](const float* dq, int nb, float* dv, int32_t* di, cudaStream_t s) {
                index->search(dq, nb, k, sp, dv, di, s);
            };
            const BatchTiming bt = time_batched(queries.data.data(), nq, d, b, k, warmup, runs, search, stream);
            double r1, r10, r100;
            recall_triplet(bt.ids, k, gt.data.data(), gt.cols, nq, r1, r10, r100);
            out.row(concat(meta.values(),
                           search_csv_values("ours_ivf_gpu", dist_variant_name(sp.coarse_variant),
                                             sp.scan_tpb, lists.nlist, np, k, b, runs, bt.latency_ms,
                                             bt.qps, r1, r10, r100)));
            std::printf("%6d %7d %9.4f %9.4f %9.4f %12.4f %10.4f %12.1f\n", b, np, r1, r10, r100,
                        bt.latency_ms.mean, bt.latency_ms.std, bt.qps.mean);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
