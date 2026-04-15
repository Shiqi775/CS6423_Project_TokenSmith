import sys
sys.path.insert(0, '.')
from src.embedder import SentenceTransformer
from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts
from src.ranking.ranker import EnsembleRanker
from src.instrumentation.chunk_tracker import ChunkAccessTracker, HotChunkCache
from src.config import RAGConfig
print('all imports ok')
c = RAGConfig()
print('embed_model:', c.embed_model)
print('gen_model:', c.gen_model)
import os
print('embed model exists:', os.path.exists(c.embed_model))
print('gen model exists:', os.path.exists(c.gen_model))
