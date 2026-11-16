# Legacy implementation (v1)

This folder keeps the first version of the project for reference. The new code in the repository root replaces it.

What v1 did:

- `python_indexing_scripts/embedding.py` embeds Wikipedia articles with `all-MiniLM-L6-v2` (384-d).
- `python_indexing_scripts/clusters.py` runs scikit-learn K-means (128 clusters) and writes one `.npy` file per cluster.
- `IVF.cpp` + `cosine_similarity.cu` run a coarse search over the centroids and a fine search over the top `n_probe` clusters, with "Atomic" and "NonAtomic" cosine-similarity kernels and CPU/GPU switches for each stage.
- `run_multiple_configs.sh` + `experiment*_config.txt` time configurations by averaging wall-clock runs.

Known limitations of v1 (all addressed in the root code): one query at a time, top-k merged on the CPU, a fresh `cudaMalloc`/copy of the whole cluster on every call, no recall measurement, no exact baseline, Python K-means.

Build (needs `json.hpp`, included here):

```bash
nvcc -O3 IVF.cpp cosine_similarity.cu -o IVF
./IVF --n_probe=30 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=true --use_cuda_fine=true --threadsperBlock=128
```

`IVF.cpp` expects the clustered data in `./cluster_data` and the article JSON in `./data`.
