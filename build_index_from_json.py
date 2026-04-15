"""
build_index_from_json.py

Builds FAISS + BM25 index directly from data/extracted_sections.json,
bypassing the markdown extraction step.
"""
import sys, json, pickle, pathlib
sys.path.insert(0, '.')

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer
from src.config import RAGConfig
from src.retriever import preprocess_for_bm25

# ── Config ──────────────────────────────────────────────────────
cfg = RAGConfig()
artifacts_dir = cfg.get_artifacts_directory()   # index/sections/
index_prefix  = "textbook_index"

json_path = pathlib.Path("data/extracted_sections.json")
print(f"Loading sections from {json_path} ...")
with open(json_path, encoding="utf-8") as f:
    sections = json.load(f)

# ── Build chunk list ─────────────────────────────────────────────
all_chunks: list[str] = []
sources:    list[str] = []
metadata:   list[dict] = []

for i, sec in enumerate(sections):
    content = sec.get("content", "").strip()
    if not content:
        continue
    heading = sec.get("heading", "")
    chapter = sec.get("chapter", 0)
    text = f"{heading}\n{content}" if heading else content
    all_chunks.append(text)
    sources.append(json_path.name)
    metadata.append({
        "filename": json_path.name,
        "chapter":  chapter,
        "heading":  heading,
        "section_path": f"Chapter {chapter} {heading}",
        "page_numbers": [],
    })

print(f"Total chunks: {len(all_chunks)}")

# ── Embed ────────────────────────────────────────────────────────
print(f"Loading embedding model: {cfg.embed_model}  (this may take a few minutes)...")
embedder = SentenceTransformer(cfg.embed_model)

print("Embedding chunks (batch size 16)...")
vectors = embedder.encode(all_chunks, batch_size=16, show_progress_bar=True)
vectors = vectors.astype("float32")

# L2-normalise for cosine similarity via IndexFlatIP
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.where(norms == 0, 1e-12, norms)

dim = vectors.shape[1]
print(f"Embedding dim: {dim}")

# ── FAISS index ──────────────────────────────────────────────────
print("Building FAISS index...")
index = faiss.IndexFlatIP(dim)
index.add(vectors)

# ── BM25 index ───────────────────────────────────────────────────
print("Building BM25 index...")
tokenized = [preprocess_for_bm25(c) for c in all_chunks]
bm25 = BM25Okapi(tokenized)

# ── Save ────────────────────────────────────────────────────────
artifacts_dir = pathlib.Path(artifacts_dir)
artifacts_dir.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
pickle.dump(bm25,     open(artifacts_dir / f"{index_prefix}_bm25.pkl",    "wb"))
pickle.dump(all_chunks, open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb"))
pickle.dump(sources,  open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb"))
pickle.dump(metadata, open(artifacts_dir / f"{index_prefix}_meta.pkl",    "wb"))

print(f"\nIndex saved to {artifacts_dir}/")
print(f"  {index_prefix}.faiss          ({index.ntotal} vectors)")
print(f"  {index_prefix}_chunks.pkl     ({len(all_chunks)} chunks)")
print(f"  {index_prefix}_bm25.pkl")
print(f"  {index_prefix}_sources.pkl")
print(f"  {index_prefix}_meta.pkl")
print("\nDone. You can now run: python -m src.main chat")
