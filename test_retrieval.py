import sys
sys.path.insert(0, '.')
from src.config import RAGConfig
from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever
from src.ranking.ranker import EnsembleRanker

config = RAGConfig()
artifacts_dir = config.get_artifacts_directory()
faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(artifacts_dir, 'textbook_index')
print(f'Loaded {len(chunks)} chunks, FAISS dim={faiss_index.d}')

faiss_ret = FAISSRetriever(faiss_index, config.embed_model)
bm25_ret  = BM25Retriever(bm25_index)
ranker    = EnsembleRanker(
    ensemble_method=config.ensemble_method,
    weights=config.ranker_weights,
    rrf_k=config.rrf_k,
)

query = 'What is a database buffer pool?'
faiss_scores = faiss_ret.get_scores(query, config.num_candidates, chunks)
bm25_scores  = bm25_ret.get_scores(query, config.num_candidates, chunks)
ranked_ids, _ = ranker.rank({'faiss': faiss_scores, 'bm25': bm25_scores})

print(f'\nQuery: {query}')
print('Top 3 results:')
for i, cid in enumerate(ranked_ids[:3]):
    print(f'  [{i+1}] chunk {cid}: {chunks[cid][:200].strip()}...')
print('\nRetrieval test PASSED.')
