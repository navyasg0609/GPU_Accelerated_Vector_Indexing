#!/usr/bin/env bash
# End-to-end benchmark run: build, test, prepare data, then for every scale compute ground truth,
# train the IVF index with CUDA K-means, and benchmark ours + FAISS. Finally the kernel study and
# the report (plots + summary.md). Every setting can be overridden from the environment, e.g.
#
#   SOURCE=synthetic SCALES="100000 1000000 5000000" scripts/run_all.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE=${SOURCE:-kaggle_wikipedia}           # kaggle_wikipedia | hf_wikipedia | synthetic
WIKI_DIR=${WIKI_DIR:-/kaggle/input/plain-text-wikipedia-202011}
SCALES=${SCALES:-"100000 500000 1000000"}    # base sizes; data is generated once at the largest
NQ=${NQ:-1000}
K=${K:-100}
NLIST=${NLIST:-128}
NPROBES=${NPROBES:-1,2,4,8,16,32,64,128}
BATCHES=${BATCHES:-1,16,128,1024}
RUNS=${RUNS:-5}
WARMUP=${WARMUP:-2}
KMEANS_ITERS=${KMEANS_ITERS:-100}
KMEANS_RUNS=${KMEANS_RUNS:-3}
SEED=${SEED:-42}
SKLEARN_MAX_N=${SKLEARN_MAX_N:-1000000}      # scikit-learn K-means is skipped above this size
SKIP_FAISS=${SKIP_FAISS:-0}
SKIP_KERNELS=${SKIP_KERNELS:-0}
case "$SOURCE" in
  synthetic) DATASET=${DATASET:-synthetic_gmm} ;;
  *)         DATASET=${DATASET:-wikipedia_minilm} ;;
esac
DATA=${DATA:-data/$DATASET}
RES=${RES:-results/$DATASET}
BIN=build

log() { printf '\n==== %s ====\n' "$*"; }

log "build"
scripts/build.sh

mkdir -p "$RES" results/kernels
# Each run starts from empty CSVs so rows from earlier runs never mix in (KEEP_RESULTS=1 to append).
if [[ "${KEEP_RESULTS:-0}" != 1 ]]; then
  rm -f "$RES"/*.csv results/kernels/*.csv
  rm -rf "$RES/plots"
fi
log "correctness tests"
$BIN/test_correctness | tee "$RES/test_correctness.log" || { echo "correctness tests failed"; exit 1; }

nvidia-smi > "$RES/nvidia_smi.txt" || true

NMAX=$(echo $SCALES | tr ' ' '\n' | sort -n | tail -1)
have_n() {  # rows in an existing dataset (0 if none)
  [[ -f "$DATA/meta.json" && -f "$DATA/base.fbin" ]] || { echo 0; return; }
  python -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['n'] if m['source']==sys.argv[2] and m['nq']==int(sys.argv[3]) else 0)"     "$DATA/meta.json" "$SOURCE" "$NQ"
}
if (( $(have_n) < NMAX )); then
  rm -f "$DATA"/*
  log "prepare data ($SOURCE, n=$NMAX, nq=$NQ)"
  python python/prepare_data.py --source "$SOURCE" --wiki_dir "$WIKI_DIR" --n "$NMAX" --nq "$NQ" \
    --out_dir "$DATA" --seed "$SEED"
fi
cp "$DATA/meta.json" "$RES/dataset_meta.json"

for N in $SCALES; do
  log "n=$N: exact ground truth + flat baselines"
  $BIN/flat_search --base "$DATA/base.fbin" --queries "$DATA/queries.fbin" --n "$N" --k "$K" \
    --gt_out "$DATA/gt_$N" --batches "$BATCHES" --runs "$RUNS" --warmup "$WARMUP" \
    --csv "$RES/flat_search.csv" --check_csv "$RES/correctness_flat.csv" --dataset "$DATASET"

  log "n=$N: CUDA K-means (nlist=$NLIST)"
  $BIN/build_ivf --base "$DATA/base.fbin" --n "$N" --nlist "$NLIST" --max_iter "$KMEANS_ITERS" \
    --seed "$SEED" --init kmeans++ --variant tiled --runs "$KMEANS_RUNS" \
    --out "$DATA/ivf_$N.bin" --centroids_out "$DATA/centroids_$N.fbin" \
    --csv "$RES/kmeans_gpu.csv" --dataset "$DATASET"

  log "n=$N: IVF search sweep"
  $BIN/ivf_bench --base "$DATA/base.fbin" --n "$N" --index "$DATA/ivf_$N.bin" \
    --queries "$DATA/queries.fbin" --gt "$DATA/gt_$N.ibin" --k "$K" --nprobe "$NPROBES" \
    --batches "$BATCHES" --runs "$RUNS" --warmup "$WARMUP" --csv "$RES/ivf_search.csv" --dataset "$DATASET"

  if [[ "$SKIP_FAISS" != 1 ]]; then
    log "n=$N: FAISS baselines"
    python python/faiss_bench.py --base "$DATA/base.fbin" --n "$N" --queries "$DATA/queries.fbin" \
      --gt "$DATA/gt_$N.ibin" --k "$K" --nlist "$NLIST" --nprobe "$NPROBES" --batches "$BATCHES" \
      --runs "$RUNS" --warmup "$WARMUP" --our_centroids "$DATA/centroids_$N.fbin" --seed "$SEED" \
      --out "$RES/faiss_search.csv" --train_out "$RES/faiss_build.csv" --dataset "$DATASET"
  fi
done

log "K-means quality/time vs. scikit-learn and FAISS (n=$NMAX)"
# The same CUDA K-means with random seeding, for a like-for-like comparison with FAISS.
$BIN/build_ivf --base "$DATA/base.fbin" --n "$NMAX" --nlist "$NLIST" --max_iter "$KMEANS_ITERS" \
  --seed "$SEED" --init random --variant tiled --runs "$KMEANS_RUNS" \
  --out "$DATA/ivf_${NMAX}_random.bin" --centroids_out "$DATA/centroids_${NMAX}_random.fbin" \
  --csv "$RES/kmeans_gpu.csv" --dataset "$DATASET"
METHODS=""
if (( NMAX <= SKLEARN_MAX_N )); then METHODS="sklearn,"; fi
if [[ "$SKIP_FAISS" != 1 ]]; then METHODS="${METHODS}faiss_cpu,faiss_gpu"; fi
python python/kmeans_reference.py --base "$DATA/base.fbin" --n "$NMAX" --nlist "$NLIST" \
  --max_iter "$KMEANS_ITERS" --seed "$SEED" --methods "$METHODS" --dataset "$DATASET" \
  --ours "ours_kmeans++_tiled=$DATA/centroids_$NMAX.fbin" "ours_random_tiled=$DATA/centroids_${NMAX}_random.fbin" \
  --out "$RES/kmeans_reference.csv"

if [[ "$SKIP_KERNELS" != 1 ]]; then
  log "distance kernel study"
  $BIN/kernel_bench --base "$DATA/base.fbin" --n "$(( NMAX < 1000000 ? NMAX : 1000000 ))" \
    --batches "$BATCHES" --csv results/kernels/kernel_bench.csv --dataset "$DATASET"
fi

log "report"
python python/make_report.py "$RES" --kernels results/kernels/kernel_bench.csv
echo "done: see $RES/summary.md and $RES/plots/"
