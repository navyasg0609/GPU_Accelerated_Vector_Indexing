# Results: findings and trade-offs

> **Pending GPU run.** This file is written only from measured data. Run
> `notebooks/run_benchmarks.ipynb` on Kaggle or Colab, commit the resulting `results/` folder,
> and this summary is then written from `results/*/summary.md`.

What each experiment answers, and where its numbers will come from:

| question | experiment | file |
|---|---|---|
| How much recall does IVF give up, and how fast is it compared with exact search? | nprobe sweep 1→128, k=100, 1,000 queries | `results/<dataset>/ivf_search.csv`, `plots/recall_vs_latency.png`, `plots/recall_vs_qps.png` |
| What does batching buy? | batch 1, 16, 128, 1024 | `plots/qps_vs_batch.png`, `plots/latency_vs_batch.png` |
| Is the CUDA K-means as good as scikit-learn / FAISS, and how much faster? | inertia + training time on the same data | `kmeans_gpu.csv`, `kmeans_reference.csv`, `plots/kmeans_comparison.png` |
| How does it compare with FAISS? | same data, k, nprobe | `faiss_search.csv`, `plots/faiss_comparison.png` |
| Which kernel design matters, and why? | 5 variants × batch × threads per block | `results/kernels/kernel_bench.csv`, `plots/kernel_variants.png`, `plots/kernel_tpb_sweep.png` |
| How does it scale with dataset size? | 100k → 1M (real), up to 5M (synthetic) | `plots/scaling.png` |
