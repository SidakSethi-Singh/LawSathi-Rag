import os
import sys
import time
import json
import logging
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import config
from src.utils.helpers import retry_with_backoff, save_jsonl

logger = logging.getLogger(__name__)

class NaiveRAG:
    """Naive RAG pipeline utilizing BM25 retrieval and OpenAI or Ollama generator."""

    def __init__(self, model_name: str = None):
        """Initialize NaiveRAG configuration, model selections, and components."""
        from src.utils.config import OPENAI_API_KEY, USE_LOCAL_MODEL, API_BASE_URL, MODEL_NAME
        self.openai_key = OPENAI_API_KEY
        self.use_local = USE_LOCAL_MODEL
        self.api_base_url = API_BASE_URL
        self.model_name = model_name or MODEL_NAME
        self.chunks: List[str] = []
        self.bm25: BM25Okapi = None

    def index_documents(self, chunks: List[str]) -> None:
        """Store documents and build the BM25 index."""
        self.chunks = chunks
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve the top k chunks closest to the query string using BM25."""
        if not self.bm25 or not self.chunks:
            logger.warning("Empty search index. Returning zero results.")
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        k = min(k, len(self.chunks))
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

    def generate(self, query: str, contexts: List[str]) -> str:
        """Generate answer using configured API (NVIDIA NIM, OpenAI, or Ollama)."""
        context_text = "\n\n".join(contexts)
        prompt = (
            "Answer the following legal question based only on the provided context. "
            "If the answer is not in the context, say 'I cannot answer from the provided context.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        if config.USE_LOCAL_MODEL:
            # Ollama local path
            import requests
            url = f"{config.API_BASE_URL}/api/generate"
            payload = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                raise
        else:
            # NVIDIA NIM or OpenAI path
            from openai import OpenAI
            client = OpenAI(
                base_url=config.API_BASE_URL,
                api_key=config.OPENAI_API_KEY
            )
            try:
                response = client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"API error: {e}")
                raise

    def answer(self, query: str, k: int = 5) -> Dict:
        """Perform retrieval and generation, tracking execution latency."""
        contexts = self.retrieve(query, k=k)
        start = time.perf_counter()
        predicted = self.generate(query, contexts)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "question": query,
            "predicted_answer": predicted,
            "retrieved_chunks": contexts,
            "latency_ms": round(latency_ms, 2),
            "model_used": self.model_name
        }

    async def a_generate(self, query: str, contexts: List[str]) -> str:
        """Asynchronously generate answer using configured API."""
        context_text = "\n\n".join(contexts)
        prompt = (
            "Answer the following legal question based only on the provided context. "
            "If the answer is not in the context, say 'I cannot answer from the provided context.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        if config.USE_LOCAL_MODEL:
            import aiohttp
            url = f"{config.API_BASE_URL}/api/generate"
            payload = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=120) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data.get("response", "")
            except Exception as e:
                logger.error(f"Ollama async error: {e}")
                raise
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=config.API_BASE_URL,
                api_key=config.OPENAI_API_KEY
            )
            try:
                response = await client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Async API error: {e}")
                raise

    async def a_answer(self, query: str, k: int = 5) -> Dict:
        """Perform retrieval (sync) and generation (async), tracking execution latency."""
        contexts = self.retrieve(query, k=k)
        start = time.perf_counter()
        predicted = await self.a_generate(query, contexts)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "question": query,
            "predicted_answer": predicted,
            "retrieved_chunks": contexts,
            "latency_ms": round(latency_ms, 2),
            "model_used": self.model_name
        }

def run_main() -> None:
    """Validate pipeline on 3 records from test file."""
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
    rag = NaiveRAG()
    rag.index_documents(chunks)
    results = [rag.answer(r["question"]) for r in records]
    out_path = project_root / "results" / "predictions" / "naive_rag.jsonl"
    save_jsonl(out_path, results)
    logger.info("Successfully executed verification run for NaiveRAG.")

if __name__ == "__main__":
    run_main()
