import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RerankerRAG:
    """
    Two-Stage Retrieval Pipeline:
    Stage 1: Fast candidate generation using a base retriever (DenseRAG or HybridRAG).
    Stage 2: Semantic reranking of candidates using a Cross-Encoder for high precision.
    """
    def __init__(self, base_retriever, reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Initializing Cross-Encoder Reranker with model: {reranker_model_name}")
        self.base_retriever = base_retriever
        # Initialize the cross-encoder model for strict semantic scoring
        self.reranker = CrossEncoder(reranker_model_name, max_length=512)

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 25):
        """
        Retrieves top_k documents by first fetching fetch_k candidates and reranking them.
        """
        # Stage 1: Broad retrieval using the base pipeline (BM25, Dense, or Hybrid)
        candidate_docs = self.base_retriever.retrieve(query, top_k=fetch_k)
        
        if not candidate_docs:
            return []

        # Handle document structure (assuming list of dicts with 'text' or 'content' key)
        text_key = 'text' if 'text' in candidate_docs[0] else 'content'

        # Stage 2: Prepare Query-Document pairs for the Cross-Encoder
        pairs = [[query, doc[text_key]] for doc in candidate_docs]
        
        # Predict semantic entailment scores
        scores = self.reranker.predict(pairs)
        
        # Inject the new scores into the document dictionaries
        for idx, doc in enumerate(candidate_docs):
            doc['reranker_score'] = float(scores[idx])
            
        # Sort documents strictly descending by the new cross-encoder score
        reranked_docs = sorted(candidate_docs, key=lambda x: x['reranker_score'], reverse=True)
        
        # Return the strictly refined top_K results
        return reranked_docs[:top_k]
