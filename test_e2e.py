"""Quick end-to-end test: retrieval + generation (non-interactive)."""
import sys
sys.path.insert(0, '.')
from src.config import RAGConfig
from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever, filter_retrieved_chunks
from src.ranking.ranker import EnsembleRanker
from src.generator import answer as gen_answer

config = RAGConfig()
artifacts_dir = config.get_artifacts_directory()
faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(artifacts_dir, 'textbook_index')
print(f'Loaded {len(chunks)} chunks')

faiss_ret = FAISSRetriever(faiss_index, config.embed_model)
bm25_ret  = BM25Retriever(bm25_index)
ranker    = EnsembleRanker(
    ensemble_method=config.ensemble_method,
    weights=config.ranker_weights,
    rrf_k=config.rrf_k,
)

query = 'What is a database buffer pool and why is it important?'
print(f'\nQuery: {query}')

faiss_scores = faiss_ret.get_scores(query, config.num_candidates, chunks)
bm25_scores  = bm25_ret.get_scores(query, config.num_candidates, chunks)
ranked_ids, _ = ranker.rank({'faiss': faiss_scores, 'bm25': bm25_scores})
top_ids = filter_retrieved_chunks(config, chunks, ranked_ids)

context_chunks = [chunks[i] for i in top_ids[:5]]  # limit to 5 chunks to stay within context window
print(f'Retrieved {len(context_chunks)} chunks for context')

print('\nLoading generator + generating answer (this may take a few minutes)...\n')
stream = gen_answer(
    query=query,
    chunks=context_chunks,
    model_path=config.gen_model,
    max_tokens=config.max_gen_tokens,
    system_prompt_mode=config.system_prompt_mode,
)
print('=== ANSWER (streaming) ===')
full_text = ""
for delta in stream:
    print(delta, end="", flush=True)
    full_text += delta
print('\n==========================')
print(f'(Total tokens generated: ~{len(full_text.split())} words)')
print('\nEnd-to-end test PASSED.')
