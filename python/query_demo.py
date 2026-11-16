"""Free-text search demo: embeds questions with the same model, searches the IVF index with the
CUDA binary and prints the matching passages. Needs data prepared with --save_text.

  python python/query_demo.py --data data/wikipedia_minilm --index data/wikipedia_minilm/ivf_1000000.bin \
      --n 1000000 "What is learning rate in gradient descent?"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

from vsutil import read_fbin, read_ibin, write_bin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--data", required=True, help="folder with base.fbin and passages.jsonl")
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--binary", default="build/ivf_query")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    q = SentenceTransformer(args.model).encode(args.questions, normalize_embeddings=True).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        qpath = os.path.join(tmp, "q.fbin")
        write_bin(qpath, q)
        subprocess.run([args.binary, "--base", os.path.join(args.data, "base.fbin"), "--n", str(args.n),
                        "--index", args.index, "--queries", qpath, "--k", str(args.k),
                        "--nprobe", str(args.nprobe), "--out", os.path.join(tmp, "res")], check=True)
        ids = np.array(read_ibin(os.path.join(tmp, "res.ibin")))
        scores = np.array(read_fbin(os.path.join(tmp, "res.fbin")))

    wanted = set(int(i) for i in ids.ravel() if i >= 0)
    text = {}
    with open(os.path.join(args.data, "passages.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in wanted:
                text[row["id"]] = row
    for qi, question in enumerate(args.questions):
        print(f"\nQ: {question}")
        for r in range(args.k):
            i = int(ids[qi, r])
            p = text.get(i, {"title": "?", "text": ""})
            print(f"  {r + 1}. [{scores[qi, r]:.3f}] {p['title']}: {p['text'][:160]}...")


if __name__ == "__main__":
    main()
