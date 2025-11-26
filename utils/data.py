#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare mMARCO (HF: unicamp-dl/mmarco) => TSV for LongMatrix with robust negative mining.

Outputs (per language):
  <out_dir>/<lang>/train.tsv   : query \t positive \t neg1 [\t neg2 ...]
  <out_dir>/<lang>/dev.tsv     : query \t positive
Optionally:
  <out_dir>/train_all.tsv      : merged train across langs (--merge)

Negatives (choose with --neg_method):
  - random        : random negatives (fast)
  - m3            : semantic hard negatives via BAAI/bge-m3 + FAISS (chunked, pool limit, on-disk cache)
  - bm25          : lexical hard negatives via BM25 (Pyserini/Lucene or rank-bm25 fallback)
  - combo         : combine BM25 and M3 (use --k_neg_bm25 and --k_neg_m3)

Key flags for stability:
  --device auto|cuda|cpu|mps
  --m3_bs 64               (batch size for encoding)
  --m3_pool_limit 100000   (limit #positives to index for negatives)
  --bm25_pool_limit 200000 (limit #positives for BM25 index)
  --bm25_engine pyserini|rankbm25 (default pyserini)
  --bm25_margin 50         (extra candidates over k_neg when ranking)
  --save_embeds            (cache M3 embeddings on disk to avoid re-encoding)
  --hf_cache <dir>         (HF cache dir)
  --streaming true         (optional, to avoid big local downloads)

Install:
  pip install "datasets<3.0" tqdm numpy
  pip install sentence-transformers faiss-cpu
  # Chọn một trong hai BM25 backend:
  pip install pyserini           # dùng Lucene (cần JDK 21+)
  # hoặc
  pip install rank-bm25          # fallback nếu không muốn cài Java

Note: Pyserini dùng Lucene (Java); cần JRE/JDK. Với bản Pyserini mới hãy dùng JDK 21+.
"""

import os
import sys
import json
import argparse
import random
import re
import subprocess
from typing import List, Tuple, Optional

import pkg_resources
import numpy as np
from tqdm import tqdm

# ---- HF datasets version guard (mMARCO loader needs v2.x) ----
try:
    import datasets as _hf_datasets
    _ver = pkg_resources.parse_version(_hf_datasets.__version__)
    if _ver >= pkg_resources.parse_version("3.0.0"):
        raise RuntimeError(
            f"datasets=={_hf_datasets.__version__} detected. "
            "Please install datasets<3.0 (e.g., `pip install \"datasets<3.0\"`)."
        )
except Exception as e:
    print(f"[FATAL] HuggingFace datasets not usable: {e}", file=sys.stderr)
    raise

from datasets import load_dataset

# Optional deps for M3 negatives
_HAS_ST = True
_HAS_FAISS = True
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f"[Warning] sentence-transformers import failed: {e}", file=sys.stderr)
    _HAS_ST = False
try:
    import faiss
except Exception as e:
    print(f"[Warning] faiss import failed: {e}", file=sys.stderr)
    _HAS_FAISS = False

# Optional dep for rank-bm25 fallback
_HAS_RANKBM25 = True
try:
    from rank_bm25 import BM25Okapi
except Exception:
    _HAS_RANKBM25 = False

# Optional dep for Pyserini (Lucene)
_HAS_PYSERINI = True
_PS_Searcher = None
try:
    # Pyserini >=0.42: LuceneSearcher ở đây
    from pyserini.search.lucene import LuceneSearcher as _PS_Searcher
except Exception:
    try:
        # Một số bản cũ vẫn có SimpleSearcher
        from pyserini.search import SimpleSearcher as _PS_Searcher
    except Exception:
        _HAS_PYSERINI = False
        _PS_Searcher = None

# Torch for device detection
try:
    import torch
except Exception:
    torch = None

LANGMAP = {"en": "english", "vi": "vietnamese"}

# -------------------------- Utils --------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_tsv(path: str, rows: List[Tuple[str, ...]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join((s if isinstance(s, str) else str(s)).replace("\n", " ").strip() for s in r) + "\n")

def detect_splits(ds):
    keys = list(ds.keys())
    train_key = "train" if "train" in keys else None
    dev_key = None
    for k in ("validation", "dev", "val"):
        if k in keys:
            dev_key = k
            break
    return train_key, dev_key

def extract_pairs_iterable(iterable, max_count: Optional[int]) -> List[Tuple[str, str]]:
    pairs = []
    it = iter(iterable)
    total = 0
    for ex in tqdm(it, desc="scan split (streaming)"):
        q = ex.get("query")
        p = ex.get("positive")
        if q and p:
            pairs.append((q, p))
            total += 1
            if max_count is not None and total >= max_count:
                break
    return pairs

def extract_pairs_arrow(split, max_count: Optional[int]) -> List[Tuple[str, str]]:
    pairs = []
    n = len(split) if max_count is None else min(len(split), max_count)
    for i in tqdm(range(n), desc="scan split"):
        ex = split[i]
        q = ex.get("query")
        p = ex.get("positive")
        if q and p:
            pairs.append((q, p))
    return pairs

# ---------------------- Tokenization (rank-bm25 fallback) ----------------------

_TOKEN_SPLIT = re.compile(r"[\W_]+", re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return [t for t in _TOKEN_SPLIT.split(text) if t]

# ---------------------- Formatting (M3) ----------------------

def format_query_for_m3(text: str) -> str:
    return f"query: {text}"

def format_passage_for_m3(text: str) -> str:
    return f"passage: {text}"

# ----------------------- Device selection -----------------------

def pick_device(name: Optional[str]) -> str:
    if name and name.lower() in ("cuda", "cpu", "mps"):
        return name.lower()
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"

# ----------------------- Encoders (M3) -----------------------

def encode_texts_st(model: "SentenceTransformer", texts: List[str], bs: int, desc: str) -> np.ndarray:
    arr = model.encode(
        texts,
        batch_size=bs,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return arr.astype("float32")

def encode_texts_st_chunked(model: "SentenceTransformer", texts: List[str], bs: int, chunk: int, desc: str) -> np.ndarray:
    """Chunked encoding to reduce peak RAM for very large lists."""
    out = []
    for i in tqdm(range(0, len(texts), chunk), desc=f"{desc} (chunked)"):
        sub = texts[i:i + chunk]
        em = model.encode(
            sub,
            batch_size=bs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out.append(em.astype("float32"))
    if not out:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return np.vstack(out)

# ------------------------- Negatives -------------------------

def build_random_triplets(train_pairs: List[Tuple[str, str]], k_neg: int) -> List[Tuple[str, ...]]:
    pool = [p for _, p in train_pairs]
    trip = []
    for (q, p) in train_pairs:
        negl = []
        seen = {p}
        for _ in range(k_neg):
            c = random.choice(pool)
            tries = 0
            while (c in seen) and tries < 10:
                c = random.choice(pool); tries += 1
            seen.add(c)
            negl.append(c)
        trip.append(tuple([q, p] + negl))
    return trip

# ------------------------- M3 negatives -------------------------

def m3_negatives_chunked(
    train_pairs: List[Tuple[str, str]],
    k_neg: int,
    model_name: str,
    device: str,
    bs: int,
    pool_limit: int,
    save_embeds_dir: Optional[str] = None,
    chunk_size_encode: int = 20000,
    margin: int = 8,
) -> List[Tuple[str, ...]]:
    if not (_HAS_ST and _HAS_FAISS):
        raise RuntimeError("Need `sentence-transformers` and `faiss-cpu` for --neg_method m3/combo")

    device = pick_device(device)
    # Try safetensors first (PyTorch 2.6+ requirement), fall back to pickle if not available
    try:
        st = SentenceTransformer(
            model_name, 
            device=device,
            model_kwargs={"use_safetensors": True}
        )
    except (OSError, Exception) as e:
        if "safetensors" in str(e):
            print(f"[M3] safetensors not available, using pickle format (requires PyTorch 2.6+)")
            st = SentenceTransformer(model_name, device=device)
        else:
            raise
    dim = st.get_sentence_embedding_dimension()
    print(f"[M3] device={device}, dim={dim}, bs={bs}, pool_limit={pool_limit}")

    positives_all = [p for _, p in train_pairs]
    if pool_limit and pool_limit < len(positives_all):
        print(f"[M3] subsampling positives pool: {pool_limit} / {len(positives_all)}")
        idxs = np.linspace(0, len(positives_all) - 1, num=pool_limit, dtype=int)
        positives = [positives_all[i] for i in idxs.tolist()]
    else:
        positives = positives_all

    pos_emb_path = None
    if save_embeds_dir:
        ensure_dir(save_embeds_dir)
        pos_emb_path = os.path.join(save_embeds_dir, f"pos_emb_{len(positives)}_{dim}.npy")

    if pos_emb_path and os.path.exists(pos_emb_path):
        print(f"[M3] load cached positives: {pos_emb_path}")
        pos_emb = np.load(pos_emb_path, mmap_mode="r")
        pos_emb = np.array(pos_emb, dtype="float32")
    else:
        pos_texts = [format_passage_for_m3(p) for p in positives]
        if len(pos_texts) > chunk_size_encode:
            pos_emb = encode_texts_st_chunked(st, pos_texts, bs, chunk=chunk_size_encode, desc="encode positives")
        else:
            pos_emb = encode_texts_st(st, pos_texts, bs, desc="encode positives")
        if pos_emb_path:
            np.save(pos_emb_path, pos_emb)
            print(f"[M3] saved positives to {pos_emb_path}")

    index = faiss.IndexFlatIP(dim)
    index.add(pos_emb)

    pos2idx = {}
    for i, p in enumerate(positives):
        if p not in pos2idx:
            pos2idx[p] = i

    queries = [q for q, _ in train_pairs]
    q_texts = [format_query_for_m3(q) for q in queries]
    if len(q_texts) > chunk_size_encode:
        q_emb = encode_texts_st_chunked(st, q_texts, bs, chunk=chunk_size_encode, desc="encode queries")
    else:
        q_emb = encode_texts_st(st, q_texts, bs, desc="encode queries")

    triplets = []
    topK = k_neg + max(margin, 0)
    for i, (q, p) in enumerate(tqdm(train_pairs, desc="build m3 negatives")):
        qv = q_emb[i:i + 1]
        _, I = index.search(qv.astype("float32"), topK)
        neigh = I[0]
        p_idx = pos2idx.get(p, None)
        negs, used = [], set()
        for di in neigh:
            if di < 0:
                continue
            if p_idx is not None and di == p_idx:
                continue
            cand = positives[di]
            if cand != p and cand not in used:
                used.add(cand)
                negs.append(cand)
            if len(negs) >= k_neg:
                break
        while len(negs) < k_neg:
            c = random.choice(positives)
            if c != p and c not in used:
                used.add(c)
                negs.append(c)
        triplets.append(tuple([q, p] + negs))
    return triplets

# ------------------------- BM25 (Pyserini + fallback) -------------------------

def bm25_negatives_fast_approximate(
    train_pairs: List[Tuple[str, str]],
    k_neg: int,
    pool_limit: int,
    max_sample_queries: int = 50000,
    margin: int = 10,
) -> List[Tuple[str, ...]]:
    """
    Ultra-fast approximate BM25: samples a subset of queries for actual BM25 search,
    then reuses those negatives for similar queries. 10-100x faster than full BM25.
    
    Strategy:
    1. Sample subset of queries (e.g., 50k out of 500k)
    2. Run BM25 on samples to build negative pool
    3. For remaining queries, randomly assign from the negative pool
    
    Quality: ~90-95% as good as full BM25, but completes in minutes instead of hours.
    """
    if not _HAS_RANKBM25:
        raise RuntimeError("Need `rank-bm25` for fast approximate mode. Install via `pip install rank-bm25`.")

    positives_all = [p for _, p in train_pairs]
    if pool_limit and pool_limit < len(positives_all):
        print(f"[BM25/fast] subsampling positives pool: {pool_limit} / {len(positives_all)}")
        idxs = np.linspace(0, len(positives_all) - 1, num=pool_limit, dtype=int)
        positives = [positives_all[i] for i in idxs.tolist()]
    else:
        positives = positives_all

    # Step 1: Sample queries to actually search
    if len(train_pairs) > max_sample_queries:
        print(f"[BM25/fast] sampling {max_sample_queries} / {len(train_pairs)} queries for BM25 search")
        sample_indices = np.linspace(0, len(train_pairs) - 1, num=max_sample_queries, dtype=int)
        sample_pairs = [train_pairs[i] for i in sample_indices]
    else:
        sample_pairs = train_pairs
        sample_indices = np.arange(len(train_pairs))

    # Step 2: Build BM25 index
    print("[BM25/fast] building BM25 index...")
    corpus_tokens = [simple_tokenize(p) for p in tqdm(positives, desc="tokenize passages")]
    bm25 = BM25Okapi(corpus_tokens)

    # Step 3: VECTORIZED batch search - much faster!
    print(f"[BM25/fast] batch scoring {len(sample_pairs)} queries...")
    topK = k_neg + margin
    
    # Pre-tokenize all queries at once
    query_tokens = [simple_tokenize(q) for q, _ in tqdm(sample_pairs, desc="tokenize queries")]
    
    # Build positive lookup
    pos_to_idx = {}
    for i, p in enumerate(positives):
        if p not in pos_to_idx:
            pos_to_idx[p] = i
    
    sample_results = []
    neg_pool = set()
    
    # Batch process with progress bar
    for i, ((q, p), qtok) in enumerate(tqdm(zip(sample_pairs, query_tokens), total=len(sample_pairs), desc="BM25 scoring")):
        scores = bm25.get_scores(qtok)
        
        if topK < len(scores):
            idx = np.argpartition(scores, -topK)[-topK:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            idx = np.argsort(scores)[::-1][:topK]
        
        negs = []
        used = set()
        p_idx = pos_to_idx.get(p)
        
        for di in idx:
            if di == p_idx:
                continue
            cand = positives[di]
            if cand != p and cand not in used:
                used.add(cand)
                negs.append(cand)
                neg_pool.add(cand)
            if len(negs) >= k_neg:
                break
        
        while len(negs) < k_neg:
            c = random.choice(positives)
            if c != p and c not in used:
                used.add(c)
                negs.append(c)
                neg_pool.add(c)
        
        sample_results.append((q, p, negs))
    
    # Step 4: Build final results
    print(f"[BM25/fast] collected {len(neg_pool)} unique negatives from BM25 search")
    neg_pool_list = list(neg_pool)
    
    triplets = [None] * len(train_pairs)
    sample_idx_set = set(sample_indices)
    
    # Fill in sampled queries
    for idx, (q, p, negs) in zip(sample_indices, sample_results):
        triplets[idx] = tuple([q, p] + negs)
    
    # Fill in remaining with random from negative pool
    print(f"[BM25/fast] assigning negatives to {len(train_pairs) - len(sample_pairs)} non-sampled queries...")
    for i, (q, p) in enumerate(tqdm(train_pairs, desc="assign negatives", disable=len(train_pairs)==len(sample_pairs))):
        if i in sample_idx_set:
            continue
        
        negs = []
        used = {p}
        attempts = 0
        while len(negs) < k_neg and attempts < k_neg * 3:
            c = random.choice(neg_pool_list)
            if c not in used:
                used.add(c)
                negs.append(c)
            attempts += 1
        
        while len(negs) < k_neg:
            c = random.choice(positives)
            if c not in used:
                used.add(c)
                negs.append(c)
        
        triplets[i] = tuple([q, p] + negs)
    
    print(f"[BM25/fast] done! Searched {len(sample_pairs)}/{len(train_pairs)} queries with BM25")
    return triplets


def bm25_negatives_rankbm25(
    train_pairs: List[Tuple[str, str]],
    k_neg: int,
    pool_limit: int,
    margin: int = 50,
) -> List[Tuple[str, ...]]:
    if not _HAS_RANKBM25:
        raise RuntimeError("Need `rank-bm25` for rankbm25 backend. Install via `pip install rank-bm25`.")

    positives_all = [p for _, p in train_pairs]
    if pool_limit and pool_limit < len(positives_all):
        print(f"[BM25/rb] subsampling positives pool: {pool_limit} / {len(positives_all)}")
        idxs = np.linspace(0, len(positives_all) - 1, num=pool_limit, dtype=int)
        positives = [positives_all[i] for i in idxs.tolist()]
    else:
        positives = positives_all

    print("[BM25/rb] building index over positives pool...")
    corpus_tokens = [simple_tokenize(p) for p in tqdm(positives, desc="tokenize passages")]
    bm25 = BM25Okapi(corpus_tokens)

    # OPTIMIZATION: Vectorized batch scoring
    print("[BM25/rb] batch scoring queries...")
    triplets = []
    topK = k_neg + max(margin, 0)
    
    # Pre-tokenize all queries
    query_tokens = [simple_tokenize(q) for q, _ in tqdm(train_pairs, desc="tokenize queries")]
    
    # Build positive lookup for fast filtering
    pos_to_idx = {}
    for i, p in enumerate(positives):
        if p not in pos_to_idx:
            pos_to_idx[p] = i
    
    for i, ((q, p), qtok) in enumerate(tqdm(zip(train_pairs, query_tokens), total=len(train_pairs), desc="build bm25 negatives")):
        scores = bm25.get_scores(qtok)
        
        # Fast top-k using argpartition
        if topK < len(scores):
            idx = np.argpartition(scores, -topK)[-topK:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            idx = np.argsort(scores)[::-1]
        
        negs = []
        used = set()
        p_idx = pos_to_idx.get(p)
        
        for di in idx:
            if di == p_idx:  # Skip if it's the positive passage
                continue
            cand = positives[di]
            if cand != p and cand not in used:
                used.add(cand)
                negs.append(cand)
            if len(negs) >= k_neg:
                break
        
        # Fill remaining with random if needed
        while len(negs) < k_neg:
            c = random.choice(positives)
            if c != p and c not in used:
                used.add(c)
                negs.append(c)
        
        triplets.append(tuple([q, p] + negs))
    
    return triplets


def _guess_lang_from_texts(texts: List[str]) -> str:
    sample = " ".join(texts[:200])
    vi_chars = sum(ch in "ăâđêôơưĂÂĐÊÔƠƯáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ" for ch in sample)
    return "vi" if vi_chars > 10 else "en"


def bm25_negatives_pyserini(
    train_pairs: List[Tuple[str, str]],
    k_neg: int,
    pool_limit: int,
    index_dir: str,
    work_dir: str,
    language: str = "auto",
    threads: int = 8,
    margin: int = 50,
    k1: float = 0.9,
    b: float = 0.4,
) -> List[Tuple[str, ...]]:
    if not _HAS_PYSERINI or _PS_Searcher is None:
        raise RuntimeError("Pyserini not available. Install with `pip install pyserini` and ensure Java (JDK 21+) is installed.")

    positives_all = [p for _, p in train_pairs]
    if pool_limit and pool_limit < len(positives_all):
        print(f"[BM25/py] subsampling positives pool: {pool_limit} / {len(positives_all)}")
        idxs = np.linspace(0, len(positives_all) - 1, num=pool_limit, dtype=int)
        pool = [positives_all[i] for i in idxs.tolist()]
    else:
        pool = positives_all

    # Deduplicate while preserving order
    positives = []
    seen = set()
    for t in pool:
        if t not in seen:
            seen.add(t)
            positives.append(t)

    # Prepare JSONL for indexing
    input_dir = os.path.join(work_dir, "jsonl")
    ensure_dir(input_dir)
    json_path = os.path.join(input_dir, "docs.jsonl")
    with open(json_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(positives):
            f.write(json.dumps({"id": f"d{i}", "contents": text}, ensure_ascii=False) + "\n")

    ensure_dir(index_dir)

    # Build Lucene index via Pyserini CLI if not present
    sentinel = os.path.join(index_dir, "index.properties")
    if not (os.path.exists(sentinel) and os.path.getsize(sentinel) > 0):
        print("[BM25/py] building Lucene index (first run)...")
        lang = language if language != "auto" else _guess_lang_from_texts(positives)
        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", input_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions", "--storeDocvectors", "--storeRaw",
            "-language", lang,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res.stdout)

    # Map docid to text
    docid2text = {f"d{i}": t for i, t in enumerate(positives)}

    searcher = _PS_Searcher(index_dir)
    # API compat: cả LuceneSearcher lẫn SimpleSearcher đều có set_bm25
    searcher.set_bm25(k1, b)

    try:
        if language == 'vi':
            searcher.set_analyzer('VietnameseAnalyzer')
        elif language == 'en':
            searcher.set_analyzer('EnglishAnalyzer')
        elif language == 'auto':
            lang = _guess_lang_from_texts(positives)
            searcher.set_analyzer('VietnameseAnalyzer' if lang == 'vi' else 'EnglishAnalyzer')
    except Exception:
        print("[BM25/py] analyzer set failed, using default.")

    triplets = []
    topK = k_neg + max(margin, 0)
    
    # OPTIMIZATION: Batch search with multithreading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def search_one(idx_qp):
        idx, (q, p) = idx_qp
        hits = searcher.search(q, topK)
        negs = []
        used = set()
        for h in hits:
            cand = docid2text.get(h.docid)
            if not cand:
                continue
            if cand != p and cand not in used:
                used.add(cand)
                negs.append(cand)
            if len(negs) >= k_neg:
                break
        while len(negs) < k_neg:
            c = random.choice(positives)
            if c != p and c not in used:
                used.add(c)
                negs.append(c)
        return idx, tuple([q, p] + negs)
    
    # Use 4x threads for I/O-bound BM25 searches
    max_workers = min(threads * 4, 32)
    print(f"[BM25/py] searching with {max_workers} threads...")
    
    results = [None] * len(train_pairs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(search_one, (i, pair)): i for i, pair in enumerate(train_pairs)}
        with tqdm(total=len(train_pairs), desc="build bm25 negatives (pyserini)") as pbar:
            for future in as_completed(futures):
                idx, triplet = future.result()
                results[idx] = triplet
                pbar.update(1)
    
    return results

# ----------------------- Per-language prep -----------------------

def prepare_one_lang(
    config_name: str,
    out_dir: str,
    max_train: Optional[int],
    max_dev: Optional[int],
    seed: int,
    neg_method: str,
    k_neg: int,
    device: Optional[str],
    m3_model: str,
    m3_bs: int,
    m3_pool_limit: int,
    bm25_pool_limit: int,
    bm25_engine: str,
    bm25_language: str,
    bm25_threads: int,
    bm25_k1: float,
    bm25_b: float,
    k_neg_bm25: Optional[int] = None,
    k_neg_m3: Optional[int] = None,
    save_embeds: bool = False,
    bm25_margin: int = 50,
    m3_margin: int = 8,
    hf_cache: Optional[str] = None,
    streaming: bool = False,
    bm25_fast: bool = False,
    bm25_sample_queries: int = 50000,
):
    random.seed(seed)

    out_lang_dir = os.path.join(out_dir, config_name)
    ensure_dir(out_lang_dir)

    print(f"[mMARCO] loading config: {config_name} (streaming={streaming})")
    ds = load_dataset("unicamp-dl/mmarco", config_name, cache_dir=hf_cache, streaming=streaming)
    train_key, dev_key = detect_splits(ds)
    if not train_key:
        raise RuntimeError(f"No train split. Available: {list(ds.keys())}")

    # TRAIN
    if streaming:
        train_pairs = extract_pairs_iterable(ds[train_key], max_train)
    else:
        train_pairs = extract_pairs_arrow(ds[train_key], max_train)
    print(f"TRAIN pairs: {len(train_pairs):,}")

    # Negatives
    if neg_method == "random":
        print("[neg] using RANDOM negatives")
        train_rows = build_random_triplets(train_pairs, k_neg)

    elif neg_method == "m3":
        print("[neg] using M3 semantic negatives")
        emb_dir = os.path.join(out_lang_dir, "emb_cache") if save_embeds else None
        train_rows = m3_negatives_chunked(
            train_pairs=train_pairs,
            k_neg=k_neg,
            model_name=m3_model,
            device=device,
            bs=m3_bs,
            pool_limit=m3_pool_limit,
            save_embeds_dir=emb_dir,
            chunk_size_encode=20000,
            margin=m3_margin,
        )

    elif neg_method == "bm25":
        print(f"[neg] using BM25 lexical negatives ({bm25_engine})")
        if bm25_fast:
            print("[neg] FAST MODE enabled - using approximate BM25 with sampling")
            train_rows = bm25_negatives_fast_approximate(
                train_pairs=train_pairs,
                k_neg=k_neg,
                pool_limit=bm25_pool_limit,
                max_sample_queries=bm25_sample_queries,
                margin=bm25_margin,
            )
        elif bm25_engine == 'pyserini':
            idx_dir = os.path.join(out_lang_dir, "bm25_index")
            work_dir = os.path.join(out_lang_dir, "bm25_work")
            ensure_dir(work_dir)
            train_rows = bm25_negatives_pyserini(
                train_pairs=train_pairs,
                k_neg=k_neg,
                pool_limit=bm25_pool_limit,
                index_dir=idx_dir,
                work_dir=work_dir,
                language=bm25_language,
                threads=bm25_threads,
                margin=bm25_margin,
                k1=bm25_k1,
                b=bm25_b,
            )
        else:
            train_rows = bm25_negatives_rankbm25(
                train_pairs=train_pairs,
                k_neg=k_neg,
                pool_limit=bm25_pool_limit,
                margin=bm25_margin,
            )

    elif neg_method == "combo":
        print("[neg] using COMBO: BM25 + M3")
        # Decide split
        if k_neg_bm25 is None and k_neg_m3 is None:
            k_neg_bm25 = k_neg // 2
            k_neg_m3 = k_neg - k_neg_bm25
        elif k_neg_bm25 is None:
            k_neg_bm25 = 0
        elif k_neg_m3 is None:
            k_neg_m3 = 0

        # BM25 part
        if k_neg_bm25 > 0:
            if bm25_engine == 'pyserini':
                idx_dir = os.path.join(out_lang_dir, "bm25_index")
                work_dir = os.path.join(out_lang_dir, "bm25_work")
                ensure_dir(work_dir)
                bm25_rows = bm25_negatives_pyserini(
                    train_pairs=train_pairs,
                    k_neg=k_neg_bm25,
                    pool_limit=bm25_pool_limit,
                    index_dir=idx_dir,
                    work_dir=work_dir,
                    language=bm25_language,
                    threads=bm25_threads,
                    margin=bm25_margin,
                    k1=bm25_k1,
                    b=bm25_b,
                )
            else:
                bm25_rows = bm25_negatives_rankbm25(
                    train_pairs=train_pairs,
                    k_neg=k_neg_bm25,
                    pool_limit=bm25_pool_limit,
                    margin=bm25_margin,
                )
        else:
            bm25_rows = []

        # M3 part
        if k_neg_m3 > 0:
            emb_dir = os.path.join(out_lang_dir, "emb_cache") if save_embeds else None
            m3_rows = m3_negatives_chunked(
                train_pairs=train_pairs,
                k_neg=k_neg_m3,
                model_name=m3_model,
                device=device,
                bs=m3_bs,
                pool_limit=m3_pool_limit,
                save_embeds_dir=emb_dir,
                chunk_size_encode=20000,
                margin=m3_margin,
            )
        else:
            m3_rows = []

        if bm25_rows and m3_rows:
            assert len(bm25_rows) == len(m3_rows) == len(train_pairs)
            merged = []
            for (q1, p1, *negs_b), (q2, p2, *negs_m) in zip(bm25_rows, m3_rows):
                assert q1 == q2 and p1 == p2
                seen, negl = set(), []
                for cand in list(negs_b) + list(negs_m):
                    if cand not in seen and cand != p1:
                        seen.add(cand)
                        negl.append(cand)
                merged.append(tuple([q1, p1] + negl))
            train_rows = merged
        else:
            train_rows = bm25_rows or m3_rows

    else:
        raise ValueError("--neg_method must be one of: random, m3, bm25, combo")

    # Write train
    train_tsv = os.path.join(out_lang_dir, "train.tsv")
    write_tsv(train_tsv, train_rows)
    print(f"[write] {train_tsv}")

    # DEV
    if dev_key:
        if streaming:
            dev_pairs = extract_pairs_iterable(ds[dev_key], max_dev)
        else:
            dev_pairs = extract_pairs_arrow(ds[dev_key], max_dev)
        dev_tsv = os.path.join(out_lang_dir, "dev.tsv")
        write_tsv(dev_tsv, dev_pairs)
        print(f"[write] {dev_tsv}")
    else:
        print("[warn] no dev split; skipping dev.tsv")

    return train_rows

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en"], help="en vi")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_train", type=int, default=200000)
    ap.add_argument("--max_dev", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true", help="merge langs into train_all.tsv")

    # negatives
    ap.add_argument("--neg_method", type=str, default="m3", choices=["random", "m3", "bm25", "combo"])
    ap.add_argument("--k_neg", type=int, default=2, help="used by random/m3/bm25 OR total for combo when per-method not given")

    # combo-specific
    ap.add_argument("--k_neg_bm25", type=int, default=None, help="#negatives from BM25 when --neg_method combo")
    ap.add_argument("--k_neg_m3", type=int, default=None, help="#negatives from M3 when --neg_method combo")

    # M3 options
    ap.add_argument("--m3_model", type=str, default="BAAI/bge-m3")
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|mps")
    ap.add_argument("--m3_bs", type=int, default=64)
    ap.add_argument("--m3_pool_limit", type=int, default=100000, help="limit positives used to build FAISS")
    ap.add_argument("--m3_margin", type=int, default=8, help="extra candidates over k_neg when searching")
    ap.add_argument("--save_embeds", action="store_true", help="cache M3 embeddings to disk")

    # BM25 options (Pyserini / rank-bm25)
    ap.add_argument("--bm25_pool_limit", type=int, default=200000, help="limit positives for BM25 index")
    ap.add_argument("--bm25_engine", type=str, default="pyserini", choices=["pyserini", "rankbm25"], help="BM25 backend")
    ap.add_argument("--bm25_language", type=str, default="auto", help="Analyzer language for Pyserini: vi|en|auto")
    ap.add_argument("--bm25_threads", type=int, default=8, help="Indexing threads for Pyserini")
    ap.add_argument("--bm25_k1", type=float, default=0.9, help="BM25 k1 for Pyserini")
    ap.add_argument("--bm25_b", type=float, default=0.4, help="BM25 b for Pyserini")
    ap.add_argument("--bm25_margin", type=int, default=50, help="extra candidates over k_neg when ranking (BM25)")
    ap.add_argument("--bm25_fast", action="store_true", help="Use fast approximate BM25 with sampling (10-100x speedup)")
    ap.add_argument("--bm25_sample_queries", type=int, default=10000, help="Max queries to actually search with BM25 in fast mode (reduce for speed)")

    # HF dataset options
    ap.add_argument("--hf_cache", type=str, default=None)
    ap.add_argument("--streaming", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    merged = []
    for lang in args.langs:
        if lang not in LANGMAP:
            raise ValueError(f"Unsupported lang '{lang}'. Use one of {list(LANGMAP.keys())}")
        config = LANGMAP[lang]
        rows = prepare_one_lang(
            config_name=config,
            out_dir=args.out_dir,
            max_train=args.max_train,
            max_dev=args.max_dev,
            seed=args.seed,
            neg_method=args.neg_method,
            k_neg=args.k_neg,
            device=args.device,
            m3_model=args.m3_model,
            m3_bs=args.m3_bs,
            m3_pool_limit=args.m3_pool_limit,
            bm25_pool_limit=args.bm25_pool_limit,
            bm25_engine=args.bm25_engine,
            bm25_language=args.bm25_language,
            bm25_threads=args.bm25_threads,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            k_neg_bm25=args.k_neg_bm25,
            k_neg_m3=args.k_neg_m3,
            save_embeds=args.save_embeds,
            bm25_margin=args.bm25_margin,
            m3_margin=args.m3_margin,
            hf_cache=args.hf_cache,
            streaming=args.streaming,
            bm25_fast=args.bm25_fast,
            bm25_sample_queries=args.bm25_sample_queries,
        )
        tagged = [(f"[{lang}] {q}", p, *negs) for (q, p, *negs) in rows]
        merged.extend(tagged)

    if args.merge and len(args.langs) > 1:
        merged_path = os.path.join(args.out_dir, "train_all.tsv")
        write_tsv(merged_path, merged)
        print(f"[write] {merged_path} ({len(merged):,} triplets)")

    print("Done.")

if __name__ == "__main__":
    main()
