"""
chunk_tracker.py

Buffer pool-inspired chunk access tracking using SQLite.

Database concept: Buffer Pool Management / LRU Replacement
  - Disk page            → Textbook chunk
  - Page access count    → Chunk retrieval frequency
  - Buffer pool (hot RAM)→ HotChunkCache (pre-loaded top-N chunks)
  - LFU eviction         → Least-accessed chunks evicted from cache
  - Page popularity boost→ Popularity score boost in EnsembleRanker
  - pg_stat_user_tables  → chunk_access SQLite table
"""

import argparse
import pathlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ChunkAccessTracker — SQLite backing store (analogous to pg_stat_user_tables)
# ---------------------------------------------------------------------------

class ChunkAccessTracker:
    """
    SQLite-backed tracker for chunk retrieval frequency.

    Schema mirrors a database statistics table:
      chunk_id      INTEGER PRIMARY KEY  — unique chunk identifier
      access_count  INTEGER              — total retrieval count
      last_accessed TEXT                 — ISO-8601 timestamp of last access
      total_score   REAL                 — cumulative retrieval score
      avg_score     REAL                 — running average score
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS chunk_access (
            chunk_id      INTEGER PRIMARY KEY,
            access_count  INTEGER DEFAULT 0,
            last_accessed TEXT,
            total_score   REAL    DEFAULT 0.0,
            avg_score     REAL    DEFAULT 0.0
        );
    """

    def __init__(self, db_path: str = "data/chunk_access.db"):
        self.db_path = db_path
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent-write safe
        conn.execute("PRAGMA synchronous=NORMAL") # balance durability/speed
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self._SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_access(self, chunk_ids: List[int], scores: List[float]) -> None:
        """
        Upsert one row per chunk_id: increment access_count, update
        total_score / avg_score, set last_accessed = NOW().
        Uses INSERT … ON CONFLICT DO UPDATE (SQLite ≥ 3.24).
        """
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            for chunk_id, score in zip(chunk_ids, scores):
                conn.execute(
                    """
                    INSERT INTO chunk_access
                        (chunk_id, access_count, last_accessed, total_score, avg_score)
                    VALUES (?, 1, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        access_count  = access_count + 1,
                        last_accessed = excluded.last_accessed,
                        total_score   = chunk_access.total_score + excluded.total_score,
                        avg_score     = (chunk_access.total_score + excluded.total_score)
                                        / (chunk_access.access_count + 1)
                    """,
                    (int(chunk_id), now, float(score), float(score)),
                )

    def get_hot_chunks(self, n: int = 50) -> List[Tuple[int, int, float]]:
        """
        Return top-N chunks by access_count (avg_score as tiebreaker).
        Returns list of (chunk_id, access_count, avg_score).
        Used at startup to seed HotChunkCache.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, access_count, avg_score
                FROM   chunk_access
                ORDER  BY access_count DESC, avg_score DESC
                LIMIT  ?
                """,
                (n,),
            ).fetchall()
        return [(int(r[0]), int(r[1]), float(r[2])) for r in rows]

    def get_boost_factors(self, chunk_ids: List[int]) -> Dict[int, float]:
        """
        Return normalized popularity boost per chunk_id:
            boost = access_count / global_max_access_count

        Unknown chunks receive 0.0.  Returns {} if no data yet (cold start).
        """
        if not chunk_ids:
            return {}
        with self._connect() as conn:
            max_row = conn.execute(
                "SELECT MAX(access_count) FROM chunk_access"
            ).fetchone()
            if not max_row or max_row[0] is None or max_row[0] == 0:
                return {cid: 0.0 for cid in chunk_ids}
            max_count = float(max_row[0])

            placeholders = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"SELECT chunk_id, access_count FROM chunk_access "
                f"WHERE chunk_id IN ({placeholders})",
                [int(c) for c in chunk_ids],
            ).fetchall()

        known = {int(r[0]): float(r[1]) / max_count for r in rows}
        return {cid: known.get(cid, 0.0) for cid in chunk_ids}

    def get_stats(self) -> Dict:
        """Summary statistics for the hot-chunk report."""
        with self._connect() as conn:
            total_chunks = conn.execute(
                "SELECT COUNT(*) FROM chunk_access"
            ).fetchone()[0]
            total_accesses = (
                conn.execute("SELECT SUM(access_count) FROM chunk_access").fetchone()[0]
                or 0
            )
            top10 = conn.execute(
                """
                SELECT chunk_id, access_count, avg_score, last_accessed
                FROM   chunk_access
                ORDER  BY access_count DESC
                LIMIT  10
                """
            ).fetchall()
            dist = {
                "1_access":      conn.execute("SELECT COUNT(*) FROM chunk_access WHERE access_count = 1").fetchone()[0],
                "2_5_accesses":  conn.execute("SELECT COUNT(*) FROM chunk_access WHERE access_count BETWEEN 2 AND 5").fetchone()[0],
                "6_20_accesses": conn.execute("SELECT COUNT(*) FROM chunk_access WHERE access_count BETWEEN 6 AND 20").fetchone()[0],
                "20plus":        conn.execute("SELECT COUNT(*) FROM chunk_access WHERE access_count > 20").fetchone()[0],
            }

        return {
            "total_chunks_tracked": int(total_chunks),
            "total_accesses":       int(total_accesses),
            "top_10_hottest": [
                {
                    "chunk_id":     int(r[0]),
                    "access_count": int(r[1]),
                    "avg_score":    round(float(r[2]), 4),
                    "last_accessed": r[3],
                }
                for r in top10
            ],
            "access_distribution": dist,
        }

    def reset(self) -> None:
        """Delete all tracking data (useful for benchmarking cold starts)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunk_access")


# ---------------------------------------------------------------------------
# HotChunkCache — in-memory buffer pool (analogous to database buffer pool)
# ---------------------------------------------------------------------------

class HotChunkCache:
    """
    Pre-loads embeddings for the top-N most frequently accessed chunks.

    Analogous to a database buffer pool keeping hot pages in RAM:
      - On startup: seed from ChunkAccessTracker (hot pages preloaded)
      - On query:   compute cosine similarity directly on cached embeddings
                    (bypasses FAISS for hot chunks)
      - Periodically: evict_and_reload() replaces cold entries (LRU-style)
    """

    def __init__(
        self,
        tracker: ChunkAccessTracker,
        chunks: List[str],
        embedder,           # CachedEmbedder — avoids circular import
        n: int = 50,
    ):
        self.tracker = tracker
        self.chunks = chunks
        self.embedder = embedder
        self.n = n
        self._hot_ids: List[int] = []
        self._hot_embeddings: Optional[np.ndarray] = None
        self.hits = 0
        self.total_queries = 0
        self._load_hot_chunks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hot_chunks(self) -> None:
        """Load embeddings for top-N hot chunks into memory."""
        hot = self.tracker.get_hot_chunks(self.n)
        if not hot:
            self._hot_ids = []
            self._hot_embeddings = None
            return

        valid = [row[0] for row in hot if 0 <= row[0] < len(self.chunks)]
        if not valid:
            self._hot_ids = []
            self._hot_embeddings = None
            return

        self._hot_ids = valid
        hot_texts = [self.chunks[cid] for cid in self._hot_ids]
        self._hot_embeddings = self.embedder.encode(hot_texts).astype("float32")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hot_scores(self, query_vec: np.ndarray) -> Dict[int, float]:
        """
        Compute cosine similarity between query_vec and all cached embeddings.
        Returns {chunk_id: cosine_similarity} for hot chunks only.

        Bypasses FAISS for these chunks — embeddings are already in memory.
        O(n_hot * d) vs FAISS O(N * d) where n_hot << N.
        """
        self.total_queries += 1
        if self._hot_embeddings is None or len(self._hot_ids) == 0:
            return {}

        self.hits += 1

        q = query_vec.flatten().astype("float32")
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return {}
        q = q / q_norm

        # Row-wise L2 normalise cached embeddings
        norms = np.linalg.norm(self._hot_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = self._hot_embeddings / norms

        sims = normalised @ q   # (n_hot,)
        return {cid: float(sim) for cid, sim in zip(self._hot_ids, sims)}

    def is_hot(self, chunk_id: int) -> bool:
        return chunk_id in self._hot_ids

    def evict_and_reload(self) -> None:
        """LRU-style refresh: reload top-N from tracker, evict stale entries."""
        self._load_hot_chunks()

    def cache_hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    @property
    def cache_size(self) -> int:
        return len(self._hot_ids)


# ---------------------------------------------------------------------------
# CLI report (make analyze-hot-chunks)
# ---------------------------------------------------------------------------

def _print_report(db_path: str) -> None:
    tracker = ChunkAccessTracker(db_path)
    stats = tracker.get_stats()

    print("\n=== HOT CHUNK REPORT ===")
    print(f"Total chunks tracked : {stats['total_chunks_tracked']}")
    print(f"Total accesses logged: {stats['total_accesses']}\n")

    print("Top 10 Hottest Chunks:")
    header = f"  {'Rank':>4}  {'chunk_id':>8}  {'access_count':>12}  {'avg_score':>9}  last_accessed"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, entry in enumerate(stats["top_10_hottest"], 1):
        print(
            f"  {rank:>4}  {entry['chunk_id']:>8}  {entry['access_count']:>12}  "
            f"{entry['avg_score']:>9.4f}  {entry['last_accessed']}"
        )

    d = stats["access_distribution"]
    total = stats["total_chunks_tracked"] or 1
    print("\nAccess distribution:")
    print(f"  1 access   : {d['1_access']:>5} chunks ({100*d['1_access']/total:.1f}%)")
    print(f"  2-5        : {d['2_5_accesses']:>5} chunks ({100*d['2_5_accesses']/total:.1f}%)")
    print(f"  6-20       : {d['6_20_accesses']:>5} chunks ({100*d['6_20_accesses']/total:.1f}%)")
    print(f"  20+        : {d['20plus']:>5} chunks ({100*d['20plus']/total:.1f}%)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hot Chunk Report")
    parser.add_argument("--report", action="store_true", help="Print hot chunk stats")
    parser.add_argument("--db", default="data/chunk_access.db", help="Path to SQLite DB")
    args = parser.parse_args()
    if args.report:
        _print_report(args.db)
