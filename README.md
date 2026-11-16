# GPU-Accelerated IVF Vector Search (C++/CUDA)

Approximate nearest-neighbour search over Wikipedia passage embeddings with an IVF (inverted
file) index, written in C++/CUDA and measured against exact search and FAISS.

Passages are embedded with `all-MiniLM-L6-v2` (384-d) and L2-normalized, so inner product equals
cosine similarity. A CUDA K-means splits the vectors into `nlist` clusters (128 by default). A
query first ranks the cluster centroids (coarse search), then scans only the `nprobe` closest
clusters (fine search) and returns the top-k passages.

What this version adds over the first one (kept in [`legacy/`](legacy/)):

| | v1 (legacy) | this version |
|---|---|---|
| queries | one at a time | batches of any size, each stage one kernel launch |
| results | top-1, merged on the CPU | top-k (k ≤ 1024), GPU radix select |
| quality | not measured | recall@1/10/100 against exact ground truth, 1,000 held-out queries |
| baselines | none | exact GPU + CPU search, FAISS Flat/IVF-Flat on CPU and GPU |
| clustering | scikit-learn (Python) | CUDA K-means, checked against scikit-learn and FAISS |
| kernels | 2 variants | 5 variants, bandwidth / occupancy / threads-per-block study |
| memory | `cudaMalloc` + copy on every call | data resident on the GPU, grow-only workspaces |

## Architecture

```
python/prepare_data.py ── Wikipedia passages ─► MiniLM embeddings ─► base.fbin, queries.fbin
                                                                         │
            ┌────────────────────────────────────────────────────────────┤
            ▼                                                            ▼
apps/flat_search  (exact, GPU + CPU)                       apps/build_ivf (CUDA K-means)
   distance kernel ─► top-k per chunk ─► merge                 seeding: k-means++ (GPU sampling)
   └─► gt.ibin  (ground truth top-k)                           loop: assign (GEMM + argmin)
            │                                                        update (sort + segmented sum)
            │                                                        converge (centroid shift)
            │                                                  └─► ivf.bin (centroids + lists)
            ▼                                                            │
apps/ivf_bench ◄─────────────────────────────────────────────────────────┘
   per batch of queries, all on the GPU:
   1 coarse   Q·Cᵀ − |c|²/2  ─► top-nprobe lists         (distance kernel + top-k)
   2 layout   per-query prefix sums of probed list sizes  (1 small kernel, 1 small copy to host)
   3 scan     block per (query, list, split): warp per vector, float4 loads ─► candidates
   4 select   segmented top-k over each query's candidates
   └─► recall@1/10/100 vs gt, latency and QPS per (nprobe, batch)

python/faiss_bench.py, kmeans_reference.py ─► same data, same k / nprobe ─► CSVs
python/make_report.py ─► results/<dataset>/plots/*.png + summary.md
```

Source layout:

| path | contents |
|---|---|
| `include/vs/`, `src/` | the library: `distance.cu` (kernels), `topk.cu` (selection), `flat_index.cu`, `ivf_index.cu`, `kmeans.cu`, `cpu_ref.cpp` (exact CPU baseline), `bench.cu` (timing protocol), `io.cpp` |
| `apps/` | `flat_search`, `build_ivf`, `ivf_bench`, `kernel_bench`, `ivf_query` |
| `tests/test_correctness.cu` | GPU vs. CPU checks for every component |
| `python/` | data prep, FAISS and scikit-learn baselines, report/plots, free-text demo |
| `scripts/` | `build.sh`, `run_all.sh` (whole pipeline) |
| `notebooks/run_benchmarks.ipynb` | one-click Kaggle / Colab run |
| `results/` | CSVs, plots and summaries from real runs |

### Key kernels

* **Distance** (`src/distance.cu`), `S = Q·Xᵀ`, chosen with `--variant`:
  * `naive`: one thread per (query, vector). Neighbouring threads read addresses `d·4` bytes apart, so loads are uncoalesced.
  * `warp`: one warp per vector with 128-bit `float4` loads, contiguous across lanes. Up to 8 queries are staged in shared memory, so each vector read from DRAM is used 8 times.
  * `tiled`: 64×64 shared-memory tiled GEMM with 4×4 register blocking.
  * `cublas`: reference (`cublasSgemm`, FP32).
  * `warp_fp16`: `warp` with fp16 storage and fp32 accumulation.
* **Top-k** (`src/topk.cu`): each block loads a 4,096-value tile into shared memory. A 4-pass 8-bit radix select finds the k-th largest key (256-bin histogram, bin picked by a warp prefix scan + ballot). The block then gathers the k winners and bitonic-sorts only those k. Long rows are reduced tile-by-tile in repeated passes. Nothing ever sorts a full row.
* **K-means** (`src/kmeans.cu`):
  * assignment: distance GEMM plus a warp-per-row argmin over `|x|² − 2x·c + |c|²`
  * update: stable sort of point ids by label, then a fixed-order per-cluster partial sum and a divide kernel (deterministic)
  * convergence: total centroid shift ≤ `tol × mean variance`, the scikit-learn rule

## Build

Requirements: CUDA ≥ 11.8 (tested setup: Kaggle/Colab CUDA 12.x), CMake ≥ 3.18, g++ with OpenMP,
Python 3.10+ with `numpy pandas matplotlib`, and for the full pipeline
`sentence-transformers scikit-learn faiss-gpu-cu12` (or `faiss-cpu`).

```bash
scripts/build.sh                 # CMake if present, else plain nvcc (ARCH=sm_75 to override)
./build/test_correctness         # exit code 0 = all checks passed
```

Manual CMake: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`.
Add `-DVS_SYNC_CHECK=ON` to synchronize after every kernel launch while debugging.

## Run

Everything, on Kaggle or Colab: open `notebooks/run_benchmarks.ipynb` and run all cells. Or from a shell:

```bash
SOURCE=kaggle_wikipedia SCALES="100000 500000 1000000" scripts/run_all.sh
SOURCE=synthetic SCALES="100000 1000000 5000000" scripts/run_all.sh     # scaling beyond the real data
```

Individual tools (`--help` on any binary lists every option):

```bash
python python/prepare_data.py --source hf_wikipedia --n 1000000 --nq 1000 --out_dir data/wiki --save_text
./build/flat_search --base data/wiki/base.fbin --queries data/wiki/queries.fbin --k 100 --gt_out data/wiki/gt
./build/build_ivf   --base data/wiki/base.fbin --nlist 128 --out data/wiki/ivf.bin --centroids_out data/wiki/centroids.fbin
./build/ivf_bench   --base data/wiki/base.fbin --index data/wiki/ivf.bin --queries data/wiki/queries.fbin \
                    --gt data/wiki/gt.ibin --k 100 --nprobe 1,2,4,8,16,32,64,128 --batches 1,16,128,1024 --tpb 256
./build/kernel_bench --base data/wiki/base.fbin --variants naive,warp,tiled,cublas,warp_fp16 --tpb 32,64,128,256,512,1024
python python/query_demo.py --data data/wiki --index data/wiki/ivf.bin "What is learning rate in gradient descent?"
```

Common options:

| option | meaning |
|---|---|
| `--n` | dataset size: uses the first n base vectors (the base is pre-shuffled) |
| `--nq` | number of queries |
| `--k` | neighbours returned per query |
| `--nprobe` | lists scanned per query |
| `--batches` | batch sizes to run |
| `--tpb` | threads per block |
| `--variant` / `--coarse_variant` | distance kernel |
| `--runs`, `--warmup` | timed and warm-up passes |
| `--max_candidates` | GPU candidate-buffer budget |

### Measurement protocol

* Fixed seed (42) for data shuffling, query hold-out, K-means seeding and random kernel data.
* Every CSV row records the GPU model, compute capability, CUDA runtime and driver versions, the dataset and n / d / nq.
* `--warmup` untimed passes run first, then `--runs` timed passes. Results are reported as mean ± sample standard deviation.
* GPU timings use `cudaEvent`s. Search latency is per batch, measured from the query upload (pinned memory) to the result download, so it includes transfers and the one host sync inside IVF search.
* QPS = queries per timed pass ÷ that pass's total time.
* recall@r = |top-r returned ∩ top-r exact| / r, averaged over the 1,000 held-out queries. The exact top-r comes from `flat_search`, which is itself checked against the CPU.
* Kernel bandwidth = compulsory bytes (read X once, read Q, write S) ÷ kernel time, compared with the theoretical peak `2 × memory clock × bus width`.

## Results

Benchmarks are produced by `scripts/run_all.sh` on a Kaggle/Colab GPU. Measured numbers live in
[`results/`](results/) (`summary.md` and `plots/` per dataset). [`RESULTS.md`](RESULTS.md)
explains them in plain language. This README quotes no figures that are not in those files.

> **Status:** the code and pipeline are complete. Benchmark results will be added once
> `notebooks/run_benchmarks.ipynb` has been run on a GPU. Until then `results/` is empty on
> purpose: no number in this repository is estimated or invented.

## Limitations

* **IVF-Flat only:** no compression, so memory is 4·d bytes per vector (1.5 KB at d=384), and about 5M vectors is the limit on a 16 GB GPU. PQ, and fp16 storage inside the IVF lists, are not implemented. fp16 is only in the kernel study and flat search.
* **The fine scan works per (query, list) pair.** Queries in a batch that probe the same list each re-read it. Grouping queries by list (a small GEMM per list) would raise large-batch throughput.
* **One host round-trip per IVF batch:** the candidate layout is copied to the host to size the buffers. This costs a few microseconds of latency, and large batches are split to respect `--max_candidates`.
* **No transfer/compute overlap:** a single stream is used, with no multi-stream pipelining.
* **Batch and k limits:** at most 65,535 queries per launch and k ≤ 1024.
* **K-means:**
  * k-means++ seeding is the plain (non-greedy) variant. scikit-learn uses greedy k-means++, so seeding quality can differ slightly.
  * Empty clusters keep their previous centroid.
  * FAISS K-means has no tolerance stop, so it always runs `max_iter` iterations.
* **Baseline scope:** the two exact CPU baselines (ours and FAISS `IndexFlatIP`) run on 100 and 200 queries (`--cpu_check`, `--cpu_max_queries`), because exact CPU search is slow. Every other method uses all 1,000 queries.
* **Synthetic scaling data:** the scaling experiments beyond the real dataset use a Gaussian-mixture dataset. Recall on it is not comparable to recall on real embeddings.

## Background

The first version of this project is kept in [`legacy/`](legacy/). Dataset: [Plain Text Wikipedia 2020-11](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011)
on Kaggle, or `wikimedia/wikipedia` on Hugging Face.
