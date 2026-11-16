"""Builds the base/query vector files used by every benchmark.

Sources
  kaggle_wikipedia  the Kaggle "plain-text-wikipedia-202011" JSON dump (attach it to the notebook)
  hf_wikipedia      Hugging Face wikimedia/wikipedia, streamed (no full download)
  synthetic         Gaussian mixture on the unit sphere, for scaling runs beyond the real data

Wikipedia articles are split into ~100-word passages and embedded with all-MiniLM-L6-v2
(384-d, the model used by the original project). Passages are shuffled with a fixed seed; the
first --nq become held-out queries (not in the base), the rest the base set. Because the base is
shuffled, the first n rows of base.fbin are a uniform random subset, which is how the scaling
experiments pick 100k / 500k / ... vectors.

Outputs in --out_dir: base.fbin, queries.fbin, meta.json and, with --save_text, passages.jsonl
(base row i -> title/text) and queries.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import numpy as np

from vsutil import create_fbin, write_bin


def article_passages(text: str, words_per_passage: int, max_per_article: int, min_words: int = 20):
    words = text.split()
    out = []
    for i in range(0, len(words), words_per_passage):
        chunk = words[i:i + words_per_passage]
        if len(chunk) < min_words:
            break
        out.append(" ".join(chunk))
        if len(out) >= max_per_article:
            break
    return out


def kaggle_articles(root: str, seed: int):
    files = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"no .json files under {root}; attach the Kaggle Wikipedia dataset")
    random.Random(seed).shuffle(files)
    for path in files:
        with open(path, encoding="utf-8") as f:
            try:
                articles = json.load(f)
            except json.JSONDecodeError:
                continue
        for art in articles:
            yield art.get("title", ""), art.get("text", "")


def hf_articles(seed: int):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for row in ds.shuffle(seed=seed, buffer_size=10_000):
        yield row["title"], row["text"]


def collect_passages(articles, total: int, words: int, per_article: int):
    titles, texts = [], []
    t0 = time.time()
    for title, text in articles:
        for p in article_passages(text, words, per_article):
            titles.append(title)
            texts.append(p)
        if len(texts) >= total:
            break
        if len(texts) and len(texts) % 200_000 < per_article:
            print(f"  collected {len(texts):,} passages ({time.time() - t0:.0f}s)", flush=True)
    if len(texts) < total:
        raise SystemExit(f"only {len(texts):,} passages available, asked for {total:,}")
    return titles[:total], texts[:total]


def embed_into(model, texts, out: np.ndarray, batch_size: int, chunk: int = 50_000):
    for s in range(0, len(texts), chunk):
        e = min(len(texts), s + chunk)
        out[s:e] = model.encode(texts[s:e], batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        print(f"  embedded {e:,}/{len(texts):,}", flush=True)


def synthetic(args):
    """Mixture of `centers` random unit directions plus isotropic noise, re-normalised."""
    rng = np.random.default_rng(args.seed)
    d = args.d
    centers = rng.standard_normal((args.centers, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Cluster sizes vary (Zipf-like weights), as in real embedding collections.
    w = 1.0 / np.arange(1, args.centers + 1) ** 0.5
    w /= w.sum()

    def sample(m, rng):
        idx = rng.choice(args.centers, size=m, p=w)
        x = centers[idx] + (args.noise / np.sqrt(d)) * rng.standard_normal((m, d)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    base = create_fbin(os.path.join(args.out_dir, "base.fbin"), args.n, d)
    for s in range(0, args.n, 500_000):
        e = min(args.n, s + 500_000)
        base[s:e] = sample(e - s, rng)
    base.flush()
    write_bin(os.path.join(args.out_dir, "queries.fbin"), sample(args.nq, rng))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["kaggle_wikipedia", "hf_wikipedia", "synthetic"], required=True)
    ap.add_argument("--wiki_dir", default="/kaggle/input/plain-text-wikipedia-202011",
                    help="root of the Kaggle dataset (searched recursively for *.json)")
    ap.add_argument("--n", type=int, default=1_000_000, help="base vectors")
    ap.add_argument("--nq", type=int, default=1_000, help="held-out query vectors")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--passage_words", type=int, default=100)
    ap.add_argument("--max_passages_per_article", type=int, default=3,
                    help="cap per article so the sample spans many articles")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--save_text", action="store_true", help="write passages.jsonl / queries.jsonl")
    ap.add_argument("--d", type=int, default=384, help="synthetic: dimension")
    ap.add_argument("--centers", type=int, default=2_000, help="synthetic: mixture components")
    ap.add_argument("--noise", type=float, default=1.0, help="synthetic: noise scale relative to the centre")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    if args.source == "synthetic":
        d = synthetic(args)
        model_name = None
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        articles = kaggle_articles(args.wiki_dir, args.seed) if args.source == "kaggle_wikipedia" \
            else hf_articles(args.seed)
        total = args.n + args.nq
        print(f"collecting {total:,} passages from {args.source} ...", flush=True)
        titles, texts = collect_passages(articles, total, args.passage_words, args.max_passages_per_article)
        perm = np.random.default_rng(args.seed).permutation(total)
        q_idx, b_idx = perm[:args.nq], perm[args.nq:]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(args.model, device=device)
        if device == "cuda":
            model = model.half()
        d = model.get_sentence_embedding_dimension()
        model_name = args.model
        print(f"embedding with {args.model} on {device} (d={d}) ...", flush=True)

        q = np.zeros((args.nq, d), dtype=np.float32)
        embed_into(model, [texts[i] for i in q_idx], q, args.batch_size)
        write_bin(os.path.join(args.out_dir, "queries.fbin"), q)
        base = create_fbin(os.path.join(args.out_dir, "base.fbin"), args.n, d)
        embed_into(model, [texts[i] for i in b_idx], base, args.batch_size)
        base.flush()

        if args.save_text:
            for name, idx in (("passages.jsonl", b_idx), ("queries.jsonl", q_idx)):
                with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
                    for row, i in enumerate(idx):
                        f.write(json.dumps({"id": row, "title": titles[i], "text": texts[i]}) + "\n")

    meta = {"source": args.source, "model": model_name, "n": args.n, "nq": args.nq, "d": d,
            "seed": args.seed, "passage_words": args.passage_words,
            "max_passages_per_article": args.max_passages_per_article,
            "synthetic": {"centers": args.centers, "noise": args.noise} if args.source == "synthetic" else None,
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
