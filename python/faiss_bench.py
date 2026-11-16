"""FAISS baselines on the same vectors, queries, ground truth, k and nprobe values as ivf_bench.

Indexes
  faiss_flat_cpu / faiss_flat_gpu        IndexFlatIP (exact)
  faiss_ivf_cpu / faiss_ivf_gpu          IndexIVFFlat, IP metric, IndexFlatIP quantizer trained by
                                         FAISS (its defaults: niter=25 on a 256*nlist sample)
  faiss_ivf_cpu (variant our_kmeans)     IndexIVFFlat with OUR centroids in an IndexFlatL2
                                         quantizer: the same partition and the same probe order as
                                         ours, so recall should match ours and only speed differs

Timing matches the C++ protocol: warm-up passes, then timed passes over all queries in batches
(wrapping around so every batch is full); each batch is timed around index.search on host
arrays, which includes host<->device copies for GPU indexes.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import time

import numpy as np

from vsutil import SEARCH_COLUMNS, Timer, environment, normalized, read_fbin, read_ibin, recall_at, stats


def time_batched(index, q: np.ndarray, k: int, batch: int, warmup: int, runs: int):
    nq = q.shape[0]
    nb = -(-nq // batch)
    batches = [q[np.arange(b * batch, (b + 1) * batch) % nq] for b in range(nb)]
    lat, qps, first = [], [], None
    for run in range(warmup + runs):
        run_s, ids = 0.0, []
        for qb in batches:
            t0 = time.perf_counter()
            _, i = index.search(qb, k)
            dt = time.perf_counter() - t0
            run_s += dt
            ids.append(i)
            if run >= warmup:
                lat.append(dt * 1e3)
        if run >= warmup:
            qps.append(nb * batch / run_s)
            if first is None:
                first = np.concatenate(ids)[:nq]
    return lat, qps, first


def set_nprobe(index, nprobe: int, gpu: bool):
    import faiss
    ps = faiss.GpuParameterSpace() if gpu else faiss.ParameterSpace()
    ps.set_index_parameter(index, "nprobe", nprobe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="data/base.fbin")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--queries", default="data/queries.fbin")
    ap.add_argument("--gt", default="data/gt.ibin")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--nlist", type=int, default=128)
    ap.add_argument("--nprobe", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--batches", default="1,16,128,1024")
    ap.add_argument("--cpu_batches", default="1,128,1024", help="batch sizes for CPU indexes")
    ap.add_argument("--cpu_max_queries", type=int, default=200,
                    help="exact CPU search (faiss_flat_cpu) uses the first N queries because it is slow; "
                         "every other index uses all queries")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--indexes", default="flat_cpu,flat_gpu,ivf_cpu,ivf_gpu,ivf_cpu_ours")
    ap.add_argument("--our_centroids", default="", help="centroids .fbin from build_ivf (for ivf_cpu_ours)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/faiss_search.csv")
    ap.add_argument("--train_out", default="results/faiss_build.csv")
    ap.add_argument("--dataset", default="unnamed")
    args = ap.parse_args()

    import faiss

    x = normalized(read_fbin(args.base, args.n if args.n > 0 else None))
    q_all = normalized(read_fbin(args.queries))
    gt_all = np.array(read_ibin(args.gt))
    n, d = x.shape
    k = args.k
    nprobes = [int(v) for v in args.nprobe.split(",") if int(v) <= args.nlist]
    want = [s for s in args.indexes.split(",") if s]
    have_gpu = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    res = faiss.StandardGpuResources() if have_gpu else None
    print(f"faiss {faiss.__version__}, GPUs: {faiss.get_num_gpus() if have_gpu else 0}, n={n} d={d}")

    rows, build_rows = [], []

    def bench(name, variant, index, gpu, nlist=0, nprobe_list=(0,)):
        nq = min(args.cpu_max_queries, q_all.shape[0]) if name == "faiss_flat_cpu" else q_all.shape[0]
        q, gt = q_all[:nq], gt_all[:nq]
        meta = environment(args.dataset, n, d, nq)
        batches = [int(b) for b in (args.batches if gpu else args.cpu_batches).split(",")]
        for b in batches:
            for npb in nprobe_list:
                if npb:
                    set_nprobe(index, npb, gpu)
                lat, qps, ids = time_batched(index, q, k, b, args.warmup, args.runs)
                lm, ls = stats(lat)
                qm, qs = stats(qps)
                r = {**meta, "method": name, "variant": variant, "tpb": "", "nlist": nlist or "",
                     "nprobe": npb or "", "k": k, "batch": b, "runs": args.runs,
                     "latency_ms_mean": lm, "latency_ms_std": ls, "qps_mean": qm, "qps_std": qs,
                     "recall_at_1": recall_at(ids, gt, 1), "recall_at_10": recall_at(ids, gt, 10),
                     "recall_at_100": recall_at(ids, gt, 100)}
                rows.append(r)
                print(f"{name:16s} {variant:12s} batch={b:5d} nprobe={npb:4d} R@10={r['recall_at_10']:.4f} "
                      f"lat={lm:9.3f}ms qps={qm:11.1f}", flush=True)

    def build_ivf(quantizer, centroids=None):
        ivf = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.cp.seed = args.seed
        with Timer() as t_train:
            if centroids is None:
                ivf.train(x)
            else:
                quantizer.add(centroids)
                ivf.is_trained = True
        with Timer() as t_add:
            ivf.add(x)
        return ivf, t_train.s, t_add.s

    def record_build(name, variant, train_s, add_s):
        build_rows.append({**environment(args.dataset, n, d, 0), "method": name, "variant": variant,
                           "nlist": args.nlist, "train_s": train_s, "add_s": add_s})

    if "flat_cpu" in want or "flat_gpu" in want:
        flat = faiss.IndexFlatIP(d)
        flat.add(x)
        if "flat_cpu" in want:
            bench("faiss_flat_cpu", "IndexFlatIP", flat, False)
        if "flat_gpu" in want and have_gpu:
            g = faiss.index_cpu_to_gpu(res, 0, flat)
            bench("faiss_flat_gpu", "IndexFlatIP", g, True)
            del g
        del flat
        gc.collect()

    if "ivf_cpu" in want or "ivf_gpu" in want:
        ivf, tr, ad = build_ivf(faiss.IndexFlatIP(d))
        record_build("faiss_ivf", "faiss_kmeans", tr, ad)
        if "ivf_cpu" in want:
            bench("faiss_ivf_cpu", "faiss_kmeans", ivf, False, args.nlist, nprobes)
        if "ivf_gpu" in want and have_gpu:
            g = faiss.index_cpu_to_gpu(res, 0, ivf)
            bench("faiss_ivf_gpu", "faiss_kmeans", g, True, args.nlist, nprobes)
            del g
        del ivf
        gc.collect()

    if "ivf_cpu_ours" in want and args.our_centroids:
        c = np.ascontiguousarray(read_fbin(args.our_centroids), dtype=np.float32)
        ivf, tr, ad = build_ivf(faiss.IndexFlatL2(d), c)
        record_build("faiss_ivf", "our_kmeans", tr, ad)
        bench("faiss_ivf_cpu", "our_kmeans", ivf, False, args.nlist, nprobes)
        del ivf
        gc.collect()

    for path, cols, data in ((args.out, SEARCH_COLUMNS, rows),
                             (args.train_out, list(build_rows[0].keys()) if build_rows else [], build_rows)):
        if not data:
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if fresh:
                w.writeheader()
            w.writerows(data)


if __name__ == "__main__":
    main()
