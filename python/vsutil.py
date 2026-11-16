"""Shared helpers: .fbin/.ibin I/O, normalisation, environment metadata, timing stats."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

import numpy as np

# Columns that every CSV row starts with (same as RunMeta in include/vs/bench.cuh).
META_COLUMNS = ["dataset", "gpu", "compute_capability", "cuda_runtime", "cuda_driver", "n", "d", "nq"]

# Search benchmark schema shared with the C++ apps (search_csv_header in src/bench.cu).
SEARCH_COLUMNS = META_COLUMNS + [
    "method", "variant", "tpb", "nlist", "nprobe", "k", "batch", "runs",
    "latency_ms_mean", "latency_ms_std", "qps_mean", "qps_std",
    "recall_at_1", "recall_at_10", "recall_at_100",
]


def read_bin(path: str, dtype, max_rows: int | None = None) -> np.ndarray:
    """Memory-maps a .fbin/.ibin file (uint32 rows, uint32 cols, then data)."""
    header = np.fromfile(path, dtype=np.uint32, count=2)
    rows, cols = int(header[0]), int(header[1])
    if max_rows is not None and 0 <= max_rows < rows:
        rows = max_rows
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, cols))


def read_fbin(path: str, max_rows: int | None = None) -> np.ndarray:
    return read_bin(path, np.float32, max_rows)


def read_ibin(path: str, max_rows: int | None = None) -> np.ndarray:
    return read_bin(path, np.int32, max_rows)


def write_bin(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = np.ascontiguousarray(arr)
    with open(path, "wb") as f:
        np.array(arr.shape, dtype=np.uint32).tofile(f)
        arr.tofile(f)


def create_fbin(path: str, rows: int, cols: int) -> np.ndarray:
    """Creates an .fbin file of the given shape and returns a writable memmap of its data."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        np.array([rows, cols], dtype=np.uint32).tofile(f)
        f.truncate(8 + rows * cols * 4)
    return np.memmap(path, dtype=np.float32, mode="r+", offset=8, shape=(rows, cols))


def normalized(x: np.ndarray) -> np.ndarray:
    """Float32 copy with unit-norm rows (inner product == cosine similarity)."""
    x = np.array(x, dtype=np.float32, copy=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    x /= norms
    return x


def _run(cmd: list[str]) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def environment(dataset: str, n: int, d: int, nq: int) -> dict:
    """GPU model, compute capability, CUDA runtime/driver versions and dataset size."""
    gpu = cc = ""
    q = _run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]).strip()
    if q:
        parts = [p.strip() for p in q.splitlines()[0].split(",")]
        gpu = parts[0]
        cc = parts[1] if len(parts) > 1 else ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]))
    driver = m.group(1) if m else ""
    m = re.search(r"release ([\d.]+)", _run(["nvcc", "--version"]))
    runtime = m.group(1) if m else ""
    return {"dataset": dataset, "gpu": gpu or "none", "compute_capability": cc,
            "cuda_runtime": runtime, "cuda_driver": driver, "n": n, "d": d, "nq": nq}


def stats(xs: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation."""
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def recall_at(res: np.ndarray, gt: np.ndarray, r: int) -> float:
    """Mean over queries of |res[:r] ∩ gt[:r]| / r."""
    if res.shape[1] < r or gt.shape[1] < r:
        return float("nan")
    hits = [len(np.intersect1d(res[i, :r], gt[i, :r], assume_unique=False)) for i in range(res.shape[0])]
    return float(np.mean(hits) / r)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
