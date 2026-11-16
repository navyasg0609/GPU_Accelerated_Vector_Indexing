"""Reference K-means runs (scikit-learn, FAISS CPU/GPU) on the same data as build_ivf, plus a
common inertia evaluation of every centroid set (including ours) so quality is compared with one
formula: sum over points of the squared L2 distance to the nearest centroid."""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

from vsutil import META_COLUMNS, Timer, environment, normalized, read_fbin, stats

COLUMNS = META_COLUMNS + ["method", "init", "nlist", "max_iter", "tol", "seed", "runs", "iterations",
                          "inertia_eval", "inertia_reported", "time_s_mean", "time_s_std", "s_per_iter",
                          "notes"]


def eval_inertia(x: np.ndarray, c: np.ndarray, chunk: int = 65536) -> float:
    try:
        import faiss
        index = faiss.IndexFlatL2(c.shape[1])
        index.add(np.ascontiguousarray(c, dtype=np.float32))
        total = 0.0
        for s in range(0, x.shape[0], chunk):
            dist, _ = index.search(np.ascontiguousarray(x[s:s + chunk]), 1)
            total += float(dist.astype(np.float64).sum())
        return total
    except ImportError:
        cn = (c.astype(np.float64) ** 2).sum(1)
        total = 0.0
        for s in range(0, x.shape[0], chunk):
            xb = x[s:s + chunk].astype(np.float64)
            d2 = (xb ** 2).sum(1, keepdims=True) - 2 * xb @ c.T.astype(np.float64) + cn
            total += float(np.maximum(d2.min(1), 0).sum())
        return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="data/base.fbin")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--nlist", type=int, default=128)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", default="sklearn,faiss_cpu,faiss_gpu")
    ap.add_argument("--runs", type=int, default=1, help="repeats per method (sklearn is slow)")
    ap.add_argument("--ours", nargs="*", default=[],
                    help="label=centroids.fbin pairs from build_ivf --centroids_out")
    ap.add_argument("--out", default="results/kmeans_reference.csv")
    ap.add_argument("--dataset", default="unnamed")
    args = ap.parse_args()

    x = normalized(read_fbin(args.base, args.n if args.n > 0 else None))
    n, d = x.shape
    meta = environment(args.dataset, n, d, 0)
    rows = []

    def add(method, init, runs, iters, inertia_rep, times, centroids, notes=""):
        mean, std = stats(times)
        rows.append({**meta, "method": method, "init": init, "nlist": args.nlist, "max_iter": args.max_iter,
                     "tol": args.tol, "seed": args.seed, "runs": runs, "iterations": iters,
                     "inertia_eval": eval_inertia(x, centroids), "inertia_reported": inertia_rep,
                     "time_s_mean": mean, "time_s_std": std,
                     "s_per_iter": mean / iters if iters else math.nan, "notes": notes})
        print(f"{method:24s} init={init:9s} iters={iters:4d} time={mean:9.2f}s "
              f"inertia={rows[-1]['inertia_eval']:.6e}", flush=True)

    methods = [m for m in args.methods.split(",") if m]
    if "sklearn" in methods:
        from sklearn.cluster import KMeans
        times, km = [], None
        for _ in range(args.runs):
            km = KMeans(n_clusters=args.nlist, init="k-means++", n_init=1, max_iter=args.max_iter,
                        tol=args.tol, random_state=args.seed, algorithm="lloyd")
            with Timer() as t:
                km.fit(x)
            times.append(t.s)
        add("sklearn_kmeans", "kmeans++", args.runs, int(km.n_iter_), float(km.inertia_), times,
            km.cluster_centers_.astype(np.float32), "greedy k-means++, CPU")

    for m in ("faiss_cpu", "faiss_gpu"):
        if m not in methods:
            continue
        import faiss
        gpu = m == "faiss_gpu"
        if gpu and faiss.get_num_gpus() == 0:
            print("faiss has no GPU support here; skipping faiss_gpu")
            continue
        times, km = [], None
        for _ in range(args.runs):
            # max_points_per_centroid is raised so FAISS trains on all points, like the others.
            km = faiss.Kmeans(d, args.nlist, niter=args.max_iter, seed=args.seed, gpu=gpu, verbose=False,
                              max_points_per_centroid=int(math.ceil(n / args.nlist)) + 1)
            with Timer() as t:
                km.train(x)
            times.append(t.s)
        add(f"{m}_kmeans", "random", args.runs, args.max_iter, float(km.obj[-1]), times, km.centroids,
            "no tolerance stop: runs all max_iter iterations")

    for spec in args.ours:
        label, path = spec.split("=", 1)
        c = np.array(read_fbin(path), dtype=np.float32)
        add(label, "", 0, 0, math.nan, [], c, "centroids from build_ivf; timing in kmeans_gpu.csv")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fresh = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if fresh:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
