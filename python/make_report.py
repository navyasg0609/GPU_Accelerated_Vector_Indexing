"""Turns the CSVs in a results directory into plots (<dir>/plots/*.png) and a markdown summary
(<dir>/summary.md). Every number in the output is read from the CSVs; nothing is typed in."""
from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical palette (light mode), assigned to entities in a fixed order so a method
# keeps its colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES = {
    "ours IVF (GPU)": PALETTE[0],
    "FAISS IVF (GPU)": PALETTE[1],
    "FAISS IVF (CPU)": PALETTE[2],
    "FAISS IVF (CPU, our partition)": PALETTE[3],
    "ours flat (GPU)": PALETTE[4],
    "FAISS flat (GPU)": PALETTE[5],
    "FAISS flat (CPU)": PALETTE[6],
    "ours flat (CPU)": PALETTE[7],
}
VARIANT_COLORS = {"naive": PALETTE[0], "warp": PALETTE[1], "tiled": PALETTE[2], "cublas": PALETTE[3],
                  "warp_fp16": PALETTE[4]}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 6, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def label(row) -> str:
    m, v = row["method"], str(row.get("variant", ""))
    names = {"ours_ivf_gpu": "ours IVF (GPU)", "faiss_ivf_gpu": "FAISS IVF (GPU)",
             "ours_flat_gpu": "ours flat (GPU)", "faiss_flat_gpu": "FAISS flat (GPU)",
             "faiss_flat_cpu": "FAISS flat (CPU)", "ours_flat_cpu": "ours flat (CPU)"}
    if m == "faiss_ivf_cpu":
        return "FAISS IVF (CPU, our partition)" if v == "our_kmeans" else "FAISS IVF (CPU)"
    return names.get(m, m)


def plain_log2_ticks(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(int(v) for v in values)))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()


def load(path):
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 0 else None


def env_caption(df) -> str:
    r = df.iloc[0]
    lo, hi = int(df["n"].min()), int(df["n"].max())
    n = f"{hi:,}" if lo == hi else f"{lo:,}–{hi:,}"
    return f"{r['gpu']} · CUDA {r['cuda_runtime']} · {r['dataset']} · n={n} · d={int(r['d'])}"


def finish(fig, path, caption):
    fig.text(0.01, 0.005, caption, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def search_table(results):
    frames = [f for f in (load(os.path.join(results, "flat_search.csv")),
                          load(os.path.join(results, "ivf_search.csv")),
                          load(os.path.join(results, "faiss_search.csv"))) if f is not None]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["series"] = df.apply(label, axis=1)
    return df


def plot_recall_tradeoff(df, out, n):
    d = df[(df["n"] == n) & df["nprobe"].notna()]
    if d.empty:
        return
    gpu_b = d[d["series"].str.contains("GPU")]["batch"].max()
    for kind, ycol, ylabel, fname in (
            ("latency", "latency_ms_mean", "latency per query (ms, batch = 1)", "recall_vs_latency.png"),
            ("qps", "qps_mean", "throughput (queries / s, largest batch)", "recall_vs_qps.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for series, g in d.groupby("series"):
            if kind == "latency":
                g = g[g["batch"] == 1]
            else:
                g = g[g["batch"] == (gpu_b if "GPU" in series else g["batch"].max())]
            g = g.sort_values("nprobe")
            if g.empty:
                continue
            ax.plot(g["recall_at_10"], g[ycol], marker="o", color=SERIES.get(series, MUTED), label=series)
            if series == "ours IVF (GPU)":
                for pt in (g.iloc[0], g.iloc[-1]):
                    ax.annotate(f"nprobe {int(pt['nprobe'])}", (pt["recall_at_10"], pt[ycol]),
                                textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED)
        ax.set_yscale("log")
        ax.set_xlabel("recall@10 (vs. exact search)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Speed vs. recall@10, nprobe swept 1 → nlist")
        ax.legend(loc="best", fontsize=8)
        finish(fig, os.path.join(out, fname), env_caption(d))


def plot_batch(df, out, n):
    d = df[(df["n"] == n) & df["series"].str.contains("GPU")]
    if d.empty:
        return
    np_pick = [p for p in (8, 32) if p in set(d["nprobe"].dropna().astype(int))]
    for ycol, ylabel, fname in (("qps_mean", "throughput (queries / s)", "qps_vs_batch.png"),
                                ("latency_ms_mean", "latency per batch (ms)", "latency_vs_batch.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for series, g in d.groupby("series"):
            variants = [(None, g[g["nprobe"].isna()])] if g["nprobe"].isna().all() else \
                [(p, g[g["nprobe"] == p]) for p in np_pick]
            for i, (p, gg) in enumerate(variants):
                gg = gg.sort_values("batch")
                if gg.empty:
                    continue
                name = series if p is None else f"{series}, nprobe {p}"
                ax.errorbar(gg["batch"], gg[ycol], yerr=gg[ycol.replace("mean", "std")], marker="o",
                            capsize=2, color=SERIES.get(series, MUTED), linestyle="-" if i == 0 else "--",
                            label=name)
        plain_log2_ticks(ax, d["batch"])
        ax.set_yscale("log")
        ax.set_xlabel("batch size (queries per launch)")
        ax.set_ylabel(ylabel)
        ax.set_title("Batching: " + ("throughput" if "qps" in ycol else "latency") + " vs. batch size")
        ax.legend(fontsize=8)
        finish(fig, os.path.join(out, fname), env_caption(d))


def plot_faiss(df, out, n):
    d = df[(df["n"] == n) & df["nprobe"].notna()]
    if d.empty or not d["method"].str.startswith("faiss").any():
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for series, g in d.groupby("series"):
        b = g["batch"].max()
        g = g[g["batch"] == b].sort_values("nprobe")
        c = SERIES.get(series, MUTED)
        axes[0].plot(g["nprobe"], g["recall_at_10"], marker="o", color=c, label=f"{series} (batch {int(b)})")
        axes[1].plot(g["nprobe"], g["qps_mean"], marker="o", color=c, label=f"{series} (batch {int(b)})")
    for ax in axes:
        plain_log2_ticks(ax, d["nprobe"])
        ax.set_xlabel("nprobe")
    axes[0].set_ylabel("recall@10")
    axes[0].set_title("Recall at the same nprobe and k")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("queries / s")
    axes[1].set_title("Throughput at the same nprobe and k")
    axes[1].legend(fontsize=8)
    finish(fig, os.path.join(out, "faiss_comparison.png"), env_caption(d))


def plot_scaling(df, kmeans, out):
    if df is None or df["n"].nunique() < 2:
        return
    fig, axes = plt.subplots(1, 3 if kmeans is not None else 2, figsize=(14, 4.2))
    picks = [("ours flat (GPU)", None), ("ours IVF (GPU)", 8), ("ours IVF (GPU)", 32)]
    for j, (title, bsel) in enumerate((("latency per query, batch 1 (ms)", "min"),
                                       ("queries / s, largest batch", "max"))):
        ax = axes[j]
        for i, (series, p) in enumerate(picks):
            g = df[df["series"] == series]
            g = g[g["nprobe"].isna()] if p is None else g[g["nprobe"] == p]
            if g.empty:
                continue
            b = g["batch"].min() if bsel == "min" else g["batch"].max()
            g = g[g["batch"] == b].sort_values("n")
            col = "latency_ms_mean" if bsel == "min" else "qps_mean"
            ax.plot(g["n"], g[col], marker="o", color=PALETTE[i],
                    label=series if p is None else f"{series}, nprobe {p}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("vectors in the index (n)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    if kmeans is not None:
        ax = axes[2]
        for i, (init, g) in enumerate(kmeans.sort_values("n").groupby("init", sort=False)):
            ax.errorbar(g["n"], g["total_ms_mean"] / 1e3, yerr=g["total_ms_std"] / 1e3, marker="o",
                        color=PALETTE[i], capsize=2, linestyle="-" if len(g) > 1 else "none",
                        label=f"ours CUDA K-means ({init} init)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("vectors (n)")
        ax.set_title("K-means training time (s)")
        ax.legend(fontsize=8)
    finish(fig, os.path.join(out, "scaling.png"), env_caption(df))


def plot_kernels(kb, out):
    if kb is None:
        return
    best = kb.loc[kb.groupby(["variant", "batch"])["ms_mean"].idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for v, g in best.groupby("variant"):
        g = g.sort_values("batch")
        axes[0].plot(g["batch"], g["gflops"], marker="o", color=VARIANT_COLORS.get(v, MUTED), label=v)
        axes[1].plot(g["batch"], g["pct_peak_bw"], marker="o", color=VARIANT_COLORS.get(v, MUTED), label=v)
    for ax in axes:
        plain_log2_ticks(ax, best["batch"])
        ax.set_xlabel("queries per launch")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("GFLOP/s (best threads-per-block)")
    axes[0].set_title("Distance kernel throughput")
    axes[1].set_ylabel("% of theoretical DRAM bandwidth")
    axes[1].set_title(f"Achieved bandwidth (peak {kb['peak_bw_gbs'].iloc[0]:.0f} GB/s)")
    axes[1].legend(fontsize=8)
    finish(fig, os.path.join(out, "kernel_variants.png"), env_caption(kb.assign(nq=0)))

    sweep = kb[kb["tpb"].notna() & kb["variant"].isin(["naive", "warp", "warp_fp16"])]
    if sweep.empty:
        return
    bsel = sorted(sweep["batch"].unique())
    fig, axes = plt.subplots(1, len(bsel), figsize=(4 * len(bsel), 3.8), squeeze=False)
    for ax, b in zip(axes[0], bsel):
        for v, g in sweep[sweep["batch"] == b].groupby("variant"):
            g = g.sort_values("tpb")
            ax.errorbar(g["tpb"], g["ms_mean"], yerr=g["ms_std"], marker="o", capsize=2,
                        color=VARIANT_COLORS.get(v, MUTED), label=v)
        plain_log2_ticks(ax, sweep["tpb"])
        ax.set_yscale("log")
        ax.set_xlabel("threads per block")
        ax.set_title(f"batch {int(b)}")
    axes[0][0].set_ylabel("kernel time (ms)")
    axes[0][-1].legend(fontsize=8)
    finish(fig, os.path.join(out, "kernel_tpb_sweep.png"), env_caption(kb.assign(nq=0)))


def plot_kmeans(ref, ours, out, n):
    if ref is None or ours is None:
        return
    r = ref[ref["n"] == n]
    o = ours[ours["n"] == n]
    if r.empty or o.empty:
        return
    names, times, errs, inert = [], [], [], []
    for _, row in o.iterrows():
        names.append(f"ours CUDA ({row['init']}, {row['variant']})")
        times.append(row["total_ms_mean"] / 1e3)
        errs.append(row["total_ms_std"] / 1e3)
        match = r[r["method"] == f"ours_{row['init']}_{row['variant']}"]
        inert.append(match["inertia_eval"].iloc[0] if not match.empty else row["inertia"])
    for _, row in r[r["time_s_mean"].notna()].iterrows():
        names.append(row["method"].replace("_kmeans", "").replace("_", " "))
        times.append(row["time_s_mean"])
        errs.append(row["time_s_std"] if pd.notna(row["time_s_std"]) else 0)
        inert.append(row["inertia_eval"])
    ref_inertia = r[r["method"] == "sklearn_kmeans"]["inertia_eval"]
    base = ref_inertia.iloc[0] if not ref_inertia.empty else min(inert)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.arange(len(names))
    axes[0].barh(y, times, xerr=errs, color=PALETTE[0], height=0.6)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("training time (s)")
    axes[0].set_title("K-means training time")
    axes[1].barh(y, [100 * (v / base - 1) for v in inert], color=PALETTE[0], height=0.6)
    axes[1].set_yticks(y, [""] * len(names))
    axes[1].axvline(0, color=MUTED, linewidth=0.8)
    axes[1].set_xlabel("inertia vs. scikit-learn (%)")
    axes[1].set_title("Clustering quality (lower is better)")
    for ax in axes:
        ax.invert_yaxis()
    finish(fig, os.path.join(out, "kmeans_comparison.png"), env_caption(r.assign(nq=0)))


INT_COLS = {"nprobe", "batch", "nq", "n", "tpb", "regs", "iterations", "converged", "runs"}
SCI_COLS = {"max_abs_err"}


def md_table(df, cols, headers=None) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("")
            elif c in INT_COLS and float(v).is_integer():
                cells.append(str(int(v)))
            elif c in SCI_COLS:
                cells.append(f"{v:.1e}")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary(results, df, kb, kmeans, kref, n):
    parts = [f"# Results summary: `{os.path.basename(os.path.abspath(results))}`", "",
             "Generated by `python/make_report.py` from the CSVs in this folder.", ""]
    src = next((x for x in (df, kb, kmeans, kref) if x is not None), None)
    if src is not None:
        parts += [f"Environment: {env_caption(src)}", ""]
    if df is not None:
        ivf = df[(df["n"] == n) & (df["method"] == "ours_ivf_gpu")]
        for b in sorted(ivf["batch"].unique()):
            t = ivf[ivf["batch"] == b].sort_values("nprobe")
            parts += [f"## Our IVF, batch {int(b)} (n = {n:,})", "",
                      md_table(t, ["nprobe", "recall_at_1", "recall_at_10", "recall_at_100", "latency_ms_mean",
                                   "latency_ms_std", "qps_mean", "qps_std"],
                               ["nprobe", "R@1", "R@10", "R@100", "latency ms", "± std", "QPS", "± std"]), ""]
        flat = df[(df["n"] == n) & df["method"].str.contains("flat")].sort_values(["series", "batch"])
        if not flat.empty:
            parts += ["## Exact (flat) search", "",
                      md_table(flat, ["series", "nq", "batch", "latency_ms_mean", "latency_ms_std", "qps_mean",
                                      "recall_at_10"],
                               ["method", "queries", "batch", "latency ms", "± std", "QPS", "R@10"]), ""]
        fa = df[(df["n"] == n) & df["method"].str.startswith("faiss_ivf")]
        if not fa.empty:
            ours = df[(df["n"] == n) & (df["method"] == "ours_ivf_gpu")]
            both = pd.concat([ours, fa])
            both = both[both["batch"].isin([1, both["batch"].max()])].sort_values(["batch", "nprobe", "series"])
            parts += ["## Ours vs. FAISS IVF (same nprobe, k)", "",
                      md_table(both, ["batch", "nprobe", "series", "recall_at_10", "latency_ms_mean", "qps_mean"],
                               ["batch", "nprobe", "method", "R@10", "latency ms", "QPS"]), ""]
        if df["n"].nunique() > 1:
            sc = df[(df["method"].isin(["ours_flat_gpu", "ours_ivf_gpu"])) &
                    (df["nprobe"].isna() | df["nprobe"].isin([8, 32]))]
            sc = sc[sc["batch"].isin([1, sc["batch"].max()])].sort_values(["series", "nprobe", "batch", "n"])
            parts += ["## Scaling", "",
                      md_table(sc, ["series", "nprobe", "batch", "n", "recall_at_10", "latency_ms_mean", "qps_mean"],
                               ["method", "nprobe", "batch", "n", "R@10", "latency ms", "QPS"]), ""]
    if kmeans is not None:
        parts += ["## CUDA K-means", "",
                  md_table(kmeans.sort_values("n"), ["n", "init", "variant", "iterations", "converged", "inertia",
                                                     "total_ms_mean", "total_ms_std", "ms_per_iter_mean"],
                           ["n", "init", "kernel", "iters", "converged", "inertia", "total ms", "± std",
                            "ms / iter"]), ""]
    if kref is not None:
        # Our rows only re-evaluate centroids from build_ivf; their iterations and timing are above.
        kref = kref.assign(iterations=kref["iterations"].where(~kref["method"].str.startswith("ours")))
        parts += ["## K-means references (inertia evaluated with one formula)", "",
                  md_table(kref.sort_values(["n", "method"]), ["n", "method", "iterations", "inertia_eval",
                                                               "time_s_mean", "time_s_std", "notes"],
                           ["n", "method", "iters", "inertia", "time s", "± std", "notes"]), ""]
    if kb is not None:
        best = kb.loc[kb.groupby(["variant", "batch"])["ms_mean"].idxmin()].sort_values(["batch", "variant"])
        parts += ["## Distance kernels (best threads-per-block per variant)", "",
                  md_table(best, ["batch", "variant", "tpb", "ms_mean", "ms_std", "gflops", "gbps", "pct_peak_bw",
                                  "occupancy", "regs", "max_abs_err"],
                           ["batch", "variant", "tpb", "ms", "± std", "GFLOP/s", "GB/s", "% peak BW", "occupancy",
                            "regs", "max err"]), ""]
    path = os.path.join(results, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="results directory, e.g. results/wikipedia_minilm")
    ap.add_argument("--kernels", default="results/kernels/kernel_bench.csv")
    args = ap.parse_args()
    out = os.path.join(args.results, "plots")
    os.makedirs(out, exist_ok=True)

    df = search_table(args.results)
    kmeans = load(os.path.join(args.results, "kmeans_gpu.csv"))
    kref = load(os.path.join(args.results, "kmeans_reference.csv"))
    kb = load(args.kernels)
    n = int(df["n"].max()) if df is not None else (int(kmeans["n"].max()) if kmeans is not None else 0)
    if df is not None:
        plot_recall_tradeoff(df, out, n)
        plot_batch(df, out, n)
        plot_faiss(df, out, n)
    plot_scaling(df, kmeans, out)
    plot_kernels(kb, out)
    plot_kmeans(kref, kmeans, out, n)
    summary(args.results, df, kb, kmeans, kref, n)


if __name__ == "__main__":
    main()
