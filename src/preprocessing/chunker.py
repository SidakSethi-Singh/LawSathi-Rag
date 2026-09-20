import logging
from typing import List
import tiktoken
from tqdm import tqdm

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split legal judgment text into token-based chunks using tiktoken.
    
    Args:
        text (str): Input text to split.
        chunk_size (int): Max token size for each chunk.
        overlap (int): Number of overlapping tokens between adjacent chunks.
        
    Returns:
        List[str]: Cleaned list of text chunks.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= chunk_size:
            return [text.strip()]
        chunks, start = [], 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_str = enc.decode(tokens[start:end]).strip()
            if chunk_str:
                chunks.append(chunk_str)
            if end >= len(tokens):
                break
            start = end - overlap
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text.strip()] if text.strip() else []

def chunk_documents(documents: List[str], chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split a list of documents into a flattened list of token-based chunks with progress tracking.
    
    Args:
        documents (List[str]): List of document strings.
        chunk_size (int): Target chunk size in tokens.
        overlap (int): Token overlap.
        
    Returns:
        List[str]: Flattened list of non-empty chunks.
    """
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        try:
            chunks = chunk_text(doc, chunk_size, overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing document: {e}")
    return all_chunks
