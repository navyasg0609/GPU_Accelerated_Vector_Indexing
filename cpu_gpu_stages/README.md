# CPU/GPU stage study

A single-query IVF pipeline where each stage (coarse search, fine search) can run on the CPU or the
GPU, used to time the stages and two cosine-similarity kernels against each other.

Contents:

- `python_indexing_scripts/embedding.py` embeds Wikipedia articles with `all-MiniLM-L6-v2` (384-d).
- `python_indexing_scripts/clusters.py` runs scikit-learn K-means (128 clusters) and writes one `.npy` file per cluster.
- `IVF.cpp` + `cosine_similarity.cu` run a coarse search over the centroids and a fine search over the top `n_probe` clusters, with "Atomic" and "NonAtomic" cosine-similarity kernels and CPU/GPU switches for each stage.
- `run_multiple_configs.sh` + `experiment*_config.txt` time configurations by averaging wall-clock runs.

Build (needs `json.hpp`, included here):

```bash
nvcc -O3 IVF.cpp cosine_similarity.cu -o IVF
./IVF --n_probe=30 --mode=NonAtomic --sequential_fine_search=false --use_cuda_coarse=true --use_cuda_fine=true --threadsperBlock=128
```

`IVF.cpp` expects the clustered data in `./cluster_data` and the article JSON in `./data`.
