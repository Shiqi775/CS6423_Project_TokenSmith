import sys
sys.path.insert(0, '.')
from src.config import RAGConfig
from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever
from src.ranking.ranker import EnsembleRanker

print("Loading config...")
config = RAGConfig()

print("Loading artifacts (FAISS index + chunks)...")
chunks, embedder, faiss_index = load_artifacts(config)
print(f"Loaded {len(chunks)} chunks")

print("Building retrievers...")
faiss_retriever = FAISSRetriever(faiss_index, embedder, config)
bm25_retriever  = BM25Retriever(chunks, config)
ranker          = EnsembleRanker(
    ensemble_method=config.ensemble_method,
    weights=config.ranker_weights,
    rrf_k=config.rrf_k,
)
print("Retrievers ready.")

query = "What is a database buffer pool?"
print(f"\nQuery: {query}")

faiss_scores = faiss_retriever.retrieve(query)
bm25_scores  = bm25_retriever.retrieve(query)
all_scores   = {"faiss": faiss_scores, "bm25": bm25_scores}
ranked_ids, _ = ranker.rank(all_scores)

print(f"\nTop-3 retrieved chunks:")
for i, cid in enumerate(ranked_ids[:3]):
    print(f"  [{i+1}] chunk {cid}: {chunks[cid][:120].strip()}...")
