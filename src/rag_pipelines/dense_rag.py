import os
import sys
import time
import json
import logging
import requests
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.utils import config
from src.utils.helpers import save_jsonl
from src.rag_pipelines.naive_rag import NaiveRAG

logger = logging.getLogger(__name__)

class DenseRAG(NaiveRAG):
    """Dense RAG pipeline utilizing SentenceTransformers embeddings and ChromaDB vector search."""

    def __init__(self, model_name: str = "gpt-4o-mini", embed_model: str = "all-MiniLM-L6-v2"):
        """Initialize DenseRAG pipeline, loading embed model and ephemeral ChromaDB."""
        super().__init__(model_name=model_name)
        try:
            logger.info(f"Loading sentence-transformer: {embed_model} on CPU...")
            self.encoder = SentenceTransformer(embed_model)
            
            logger.info("Initializing ChromaDB Ephemeral client...")
            self.client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
            self.collection = self.client.get_or_create_collection("legal_chunks")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB or Encoder: {e}")
            sys.exit(1)

    def index_documents(self, chunks: List[str]) -> None:
        """Generate document embeddings and index them in ChromaDB collection."""
        self.chunks = chunks
        try:
            embeddings = self.encoder.encode(chunks, show_progress_bar=True)
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings.tolist()
            )
        except Exception as e:
            logger.error(f"Error during document indexing in ChromaDB: {e}")

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve closest context chunks from ChromaDB for the user query."""
        if not self.chunks:
            logger.warning("Empty dense index. Returning zero results.")
            return []
        try:
            n_results = min(k, len(self.chunks))
            q_emb = self.encoder.encode([query]).tolist()
            results = self.collection.query(query_embeddings=q_emb, n_results=n_results)
            if results and "documents" in results and results["documents"]:
                return results["documents"][0]
        except Exception as e:
            logger.error(f"Error querying ChromaDB collection: {e}")
        return []

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

def run_main() -> None:
    """Validate DenseRAG pipeline on 3 test records."""
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
    rag = DenseRAG()
    rag.index_documents(chunks)
    results = [rag.answer(r["question"]) for r in records]
    out_path = project_root / "results" / "predictions" / "dense_rag.jsonl"
    save_jsonl(out_path, results)
    logger.info("Successfully executed verification run for DenseRAG.")

if __name__ == "__main__":
    run_main()
