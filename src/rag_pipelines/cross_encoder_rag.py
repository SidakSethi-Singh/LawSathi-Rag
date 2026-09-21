import os
import sys
import time
import json
import logging
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from sentence_transformers import CrossEncoder

from src.utils import config
from src.utils.helpers import save_jsonl
from src.rag_pipelines.hybrid_rag import HybridRAG

logger = logging.getLogger(__name__)

class CrossEncoderRAG(HybridRAG):
    """Two-Stage RAG pipeline combining Hybrid (BM25 + Dense) retrieval with a Cross-Encoder Reranker."""

    def __init__(
        self,
        model_name: str = None,
        embed_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = None,
        alpha: float = 0.7,
        top_n_initial: int = None
    ):
        """Initialize CrossEncoderRAG pipeline, loading Dense encoder, ChromaDB, and CrossEncoder reranker."""
        super().__init__(model_name=model_name, embed_model=embed_model, alpha=alpha)
        self.reranker_model_name = reranker_model or config.RERANKER_MODEL
        self.top_n_initial = top_n_initial or config.TOP_N_INITIAL
        try:
            logger.info(f"CrossEncoderRAG: Loading cross-encoder reranker model '{self.reranker_model_name}'...")
            self.reranker = CrossEncoder(self.reranker_model_name)
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model '{self.reranker_model_name}': {e}")
            sys.exit(1)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Perform two-stage retrieval: initial hybrid candidate retrieval followed by cross-encoder re-scoring."""
        if not self.chunks:
            return []
        try:
            # Stage 1: Retrieve top initial candidate chunks using Hybrid RAG (BM25 + Dense)
            candidate_pool_size = max(k, self.top_n_initial)
            initial_candidates = super().retrieve(query, k=candidate_pool_size)
            if not initial_candidates:
                return []

            # Stage 2: Cross-Encoder joint scoring over (query, passage) pairs
            pairs = [[query, doc] for doc in initial_candidates]
            scores = self.reranker.predict(pairs)

            # Sort candidate chunks by cross-encoder score descending
            scored_candidates = list(zip(initial_candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Return top k re-scored passages
            reranked_docs = [doc for doc, score in scored_candidates[:k]]
            return reranked_docs
        except Exception as e:
            logger.error(f"Error during Cross-Encoder retrieval reranking: {e}")
            return super().retrieve(query, k=k)


def run_main() -> None:
    """Validate CrossEncoderRAG pipeline on 3 test records."""
    test_path = project_root / "data" / "test.jsonl"
    if not test_path.exists():
        logger.error(f"Test split not found at {test_path}")
        sys.exit(1)
    if config.USE_LOCAL_MODEL:
        try:
            requests.get(config.OLLAMA_BASE_URL, timeout=5.0)
        except Exception:
            logger.error("Ollama server is offline. Please run it before execution.")
            sys.exit(1)
    records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            records.append(json.loads(line))
    chunks = list(set([c for r in records for c in r.get("context_chunks", [])]))
    rag = CrossEncoderRAG()
    rag.index_documents(chunks)
    results = [rag.answer(r["question"]) for r in records]
    out_path = project_root / "results" / "predictions" / "cross_encoder_rag.jsonl"
    save_jsonl(out_path, results)
    logger.info("Successfully executed verification run for CrossEncoderRAG.")


if __name__ == "__main__":
    run_main()
