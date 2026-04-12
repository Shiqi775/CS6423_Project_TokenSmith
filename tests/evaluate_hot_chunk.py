"""
evaluate_hot_chunk.py

Standalone evaluation script for the buffer pool-inspired hot-chunk caching
system.  Produces concrete experimental results for the final report without
requiring model files.

Run:
    python -m tests.evaluate_hot_chunk

Outputs:
  - Table 1: Retrieval latency — hot cache vs full FAISS-style search
  - Table 2: Cache hit rate over query rounds
  - Table 3: Chunk access distribution (Pareto analysis)
  - Table 4: Popularity boost ranking impact
"""

import sys
import os
import time
import tempfile
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.instrumentation.chunk_tracker import ChunkAccessTracker, HotChunkCache
from src.ranking.ranker import EnsembleRanker
from unittest.mock import MagicMock

# ── Constants ────────────────────────────────────────────────────────────────
N_CHUNKS    = 2000    # total corpus size
DIM         = 128     # embedding dimension
N_HOT       = 50      # hot-cache capacity
TOP_K       = 10      # chunks returned per query
N_QUERIES   = 200     # total simulated queries
ZIPF_ALPHA  = 0.85    # Zipf exponent (controls Pareto steepness)
BOOST_ALPHA = 0.10    # popularity boost magnitude
REPEAT      = 100     # repetitions for latency timing

# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_corpus(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    """Return Zipf-distributed probabilities over n items."""
    ranks = np.arange(1, n + 1, dtype=float)
    probs = 1.0 / ranks ** alpha
    return probs / probs.sum()


def _make_dummy_embedder(corpus: np.ndarray):
    """Mock embedder that returns rows from corpus for given texts."""
    embedder = MagicMock()
    # Encode by hashing text to a corpus index (deterministic)
    def encode(texts):
        indices = [abs(hash(t)) % len(corpus) for t in texts]
        return corpus[indices].copy()
    embedder.encode.side_effect = encode
    return embedder


def _full_search(corpus: np.ndarray, q_vec: np.ndarray, top_k: int) -> Dict[int, float]:
    """Brute-force cosine similarity search over full corpus (simulates FAISS)."""
    sims = corpus @ q_vec.flatten()
    top_ids = np.argpartition(-sims, kth=top_k - 1)[:top_k]
    return {int(i): float(sims[i]) for i in top_ids}


def _hot_search(hot_ids: List[int], hot_vecs: np.ndarray,
                q_vec: np.ndarray) -> Dict[int, float]:
    """Hot-cache cosine similarity over cached subset."""
    sims = hot_vecs @ q_vec.flatten()
    return {cid: float(s) for cid, s in zip(hot_ids, sims)}


# ── Experiment 1: Latency comparison ─────────────────────────────────────────

def exp1_latency(corpus: np.ndarray) -> None:
    print("\n" + "=" * 62)
    print("Experiment 1: Hot Cache vs Full-Search Latency")
    print("=" * 62)
    print(f"  {'N (corpus)':>12}  {'Full (ms)':>10}  {'Hot (ms)':>9}  {'Speedup':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*9}  {'-'*8}")

    rng = np.random.default_rng(7)
    hot_indices = list(range(N_HOT))
    results = []

    for n in [500, 1000, 2000]:
        sub = corpus[:n]
        hot_vecs = sub[hot_indices].copy()
        queries = [rng.standard_normal(DIM).astype("float32") for _ in range(REPEAT)]
        queries = [q / np.linalg.norm(q) for q in queries]

        # Full search
        t0 = time.perf_counter()
        for q in queries:
            _full_search(sub, q, TOP_K)
        t_full_ms = (time.perf_counter() - t0) * 1000

        # Hot cache search
        t0 = time.perf_counter()
        for q in queries:
            _hot_search(hot_indices, hot_vecs, q)
        t_hot_ms = (time.perf_counter() - t0) * 1000

        speedup = t_full_ms / max(t_hot_ms, 1e-9)
        full_per = t_full_ms / REPEAT
        hot_per  = t_hot_ms  / REPEAT
        results.append((n, full_per, hot_per, speedup))
        print(f"  {n:>12,}  {full_per:>10.4f}  {hot_per:>9.4f}  {speedup:>7.1f}x")

    return results


# ── Experiment 2: Cache hit rate over queries ─────────────────────────────────

def exp2_cache_hit_rate(corpus: np.ndarray) -> None:
    """
    Measure cumulative cache hit rate: after N warm-up queries have populated
    the tracker, what fraction of NEW queries have ≥1 hot chunk in top-k?
    """
    print("\n" + "=" * 62)
    print("Experiment 2: Cache Hit Rate Over Query Rounds")
    print(f"  (Zipf alpha={ZIPF_ALPHA}, N={N_CHUNKS}, n_hot={N_HOT}, top_k={TOP_K})")
    print("=" * 62)
    print(f"  {'Warmup Qs':>10}  {'Chunks Seen':>12}  {'Hit Rate':>9}  {'Avg Hits/q':>11}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*9}  {'-'*11}")

    probs   = _zipf_probs(N_CHUNKS, ZIPF_ALPHA)
    rng     = np.random.default_rng(99)
    db_path = os.path.join(tempfile.gettempdir(), "eval_hitrate.db")

    # Clean up from any previous run
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    tracker  = ChunkAccessTracker(db_path)
    embedder = _make_dummy_embedder(corpus)
    chunks_placeholder = [f"c{i}" for i in range(N_CHUNKS)]

    warmup_phases = [10, 25, 50, 100, 150, 200]
    results = []
    prev_warmup = 0

    for warmup in warmup_phases:
        # Run additional warm-up queries
        for _ in range(warmup - prev_warmup):
            retrieved = rng.choice(N_CHUNKS, size=TOP_K, replace=False, p=probs)
            scores    = [0.8] * TOP_K
            tracker.log_access(retrieved.tolist(), scores)
        prev_warmup = warmup

        # Build cache from current tracker state
        cache   = HotChunkCache(tracker, chunks_placeholder, embedder, n=N_HOT)
        hot_set = set(cache._hot_ids)
        n_seen  = tracker.get_stats()["total_chunks_tracked"]

        # Measure hit rate on 50 fresh queries
        hits_total = 0
        n_eval = 50
        for _ in range(n_eval):
            test_retrieved = rng.choice(N_CHUNKS, size=TOP_K, replace=False, p=probs)
            hits = len(hot_set & set(test_retrieved.tolist()))
            hits_total += hits
        hit_rate = hits_total / (n_eval * TOP_K)   # fraction of retrieved slots that were hot
        avg_hits = hits_total / n_eval

        results.append((warmup, n_seen, hit_rate, avg_hits))
        print(f"  {warmup:>10}  {n_seen:>12}  {hit_rate:>8.1%}  {avg_hits:>11.2f}")

    # Clean up
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    return results


# ── Experiment 3: Access distribution (Pareto analysis) ──────────────────────

def exp3_access_distribution(corpus: np.ndarray) -> None:
    print("\n" + "=" * 62)
    print("Experiment 3: Chunk Access Distribution")
    print(f"  ({N_QUERIES} queries, Zipf alpha={ZIPF_ALPHA}, N={N_CHUNKS})")
    print("=" * 62)

    probs = _zipf_probs(N_CHUNKS, ZIPF_ALPHA)
    rng   = np.random.default_rng(42)
    access_counts = np.zeros(N_CHUNKS, dtype=int)

    db_path = os.path.join(tempfile.gettempdir(), "eval_dist.db")
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    tracker = ChunkAccessTracker(db_path)
    for _ in range(N_QUERIES):
        retrieved = rng.choice(N_CHUNKS, size=TOP_K, replace=False, p=probs)
        tracker.log_access(retrieved.tolist(), [0.8] * TOP_K)
        access_counts[retrieved] += 1

    stats = tracker.get_stats()
    total_accesses = stats["total_accesses"]
    tracked        = stats["total_chunks_tracked"]

    # Pareto: what % of corpus accounts for 80% of accesses?
    sorted_counts = np.sort(access_counts)[::-1]
    cumsum = np.cumsum(sorted_counts)
    thresh_80 = 0.80 * total_accesses
    chunks_for_80 = int(np.searchsorted(cumsum, thresh_80)) + 1
    pct_chunks = 100.0 * chunks_for_80 / N_CHUNKS

    dist = stats["access_distribution"]
    print(f"  Total accesses     : {total_accesses:,}")
    print(f"  Chunks touched     : {tracked} / {N_CHUNKS} ({100*tracked/N_CHUNKS:.1f}%)")
    print(f"  Pareto: {chunks_for_80} chunks ({pct_chunks:.1f}%) account for 80%% of accesses")
    print(f"\n  Access count distribution:")
    print(f"  {'Bracket':>12}  {'Chunks':>8}  {'% of tracked':>13}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*13}")
    tr = max(tracked, 1)
    brackets = [
        ("1",     dist["1_access"]),
        ("2-5",   dist["2_5_accesses"]),
        ("6-20",  dist["6_20_accesses"]),
        ("21+",   dist["20plus"]),
    ]
    for label, cnt in brackets:
        print(f"  {label:>12}  {cnt:>8}  {100*cnt/tr:>12.1f}%")

    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    return chunks_for_80, pct_chunks, dist, tracked


# ── Experiment 4: Popularity boost ranking impact ────────────────────────────

def exp4_boost_impact() -> None:
    print("\n" + "=" * 62)
    print("Experiment 4: Popularity Boost Ranking Impact")
    print(f"  (boost_alpha={BOOST_ALPHA})")
    print("=" * 62)

    ranker = EnsembleRanker(ensemble_method="rrf", weights={"faiss": 1.0}, rrf_k=60)
    rng = np.random.default_rng(13)

    # Simulate: chunk 0..49 are "hot" (high access counts)
    db_path = os.path.join(tempfile.gettempdir(), "eval_boost.db")
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass
    if True:
        db = db_path
        tracker = ChunkAccessTracker(db)

        # Seed: top 50 chunks accessed frequently
        for _ in range(30):
            tracker.log_access(list(range(50)), [0.85] * 50)
        # Cold chunks: accessed 1-2 times
        tracker.log_access(list(range(50, 150)), [0.5] * 100)

        boost_factors = tracker.get_boost_factors(list(range(N_CHUNKS)))

        # Generate synthetic scores where cold chunks rank higher than hot chunks
        # (simulating a case where hot chunk was slightly outscored by a cold chunk)
        raw_scores = {"faiss": {}}
        hot_chunks  = list(range(50))
        cold_chunks = list(range(50, 150))

        for c in hot_chunks:
            raw_scores["faiss"][c] = 0.70 + rng.random() * 0.10   # 0.70–0.80
        for c in cold_chunks:
            raw_scores["faiss"][c] = 0.72 + rng.random() * 0.10   # 0.72–0.82

        ids_no_boost, _ = ranker.rank(raw_scores, boost_factors=None)
        ids_boosted,  _ = ranker.rank(raw_scores, boost_factors=boost_factors,
                                      boost_alpha=BOOST_ALPHA)

        top10_no = set(ids_no_boost[:10])
        top10_bo = set(ids_boosted[:10])

        hot_in_no = len(top10_no & set(hot_chunks))
        hot_in_bo = len(top10_bo & set(hot_chunks))

        print(f"  Hot chunks in top-10 without boost : {hot_in_no}")
        print(f"  Hot chunks in top-10 with boost    : {hot_in_bo}")
        print(f"  Improvement in hot-chunk retrieval : +{hot_in_bo - hot_in_no}")

        jaccard_no = len(top10_no & top10_bo) / len(top10_no | top10_bo)
        print(f"  Jaccard(top10_no_boost, top10_boost): {jaccard_no:.2f}")

    return hot_in_no, hot_in_bo


# ── Experiment 5: Per-query latency distribution ─────────────────────────────

def exp5_latency_distribution(corpus: np.ndarray) -> None:
    print("\n" + "=" * 62)
    print("Experiment 5: Per-Query Hot-Cache Latency Distribution")
    print(f"  (n_hot={N_HOT}, dim={DIM}, {REPEAT} queries)")
    print("=" * 62)

    rng = np.random.default_rng(5)
    hot_indices = list(range(N_HOT))
    hot_vecs = corpus[hot_indices].copy()

    latencies_ms = []
    for _ in range(REPEAT):
        q = rng.standard_normal(DIM).astype("float32")
        q /= np.linalg.norm(q)
        t0 = time.perf_counter()
        _hot_search(hot_indices, hot_vecs, q)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies_ms)
    print(f"  Min  : {lat.min():.4f} ms")
    print(f"  Mean : {lat.mean():.4f} ms")
    print(f"  P50  : {np.percentile(lat, 50):.4f} ms")
    print(f"  P95  : {np.percentile(lat, 95):.4f} ms")
    print(f"  P99  : {np.percentile(lat, 99):.4f} ms")
    print(f"  Max  : {lat.max():.4f} ms")

    return lat.mean(), np.percentile(lat, 95)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Hot-Chunk Buffer Pool Evaluation Suite")
    print(f"  Corpus={N_CHUNKS} chunks, dim={DIM}, n_hot={N_HOT}")
    print("=" * 62)

    corpus = _build_corpus(N_CHUNKS, DIM)

    lat_results = exp1_latency(corpus)
    exp2_cache_hit_rate(corpus)
    exp3_access_distribution(corpus)
    exp4_boost_impact()
    exp5_latency_distribution(corpus)

    print("\n" + "=" * 62)
    print("  Evaluation complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()
