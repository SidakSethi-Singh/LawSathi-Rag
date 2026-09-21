import re
import logging
from typing import List, Dict, Optional
import tiktoken

logger = logging.getLogger(__name__)

class StatutoryLegalChunker:
    """Structure and section-aware legal text chunker for Indian statutory codes and judicial documents.
    
    Parses statutory section headings (e.g. 'Section 302', 'Article 21'), legal act references
    (e.g., IPC, CrPC, CPC, Constitution), and judgment headers ('Held:', 'Judgment:'), prepending
    parent section context to sub-chunks when splitting long legal passages.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoder = None

        # Regex patterns for Indian legal structure elements
        self.section_pattern = re.compile(
            r'(?i)(?:\n|^)\s*(?:section|sec\.|article|art\.|chapter|clause)\s+([0-9]+[A-Za-z]?)(?:\s*\(([0-9a-zA-Z]+)\))?',
            re.IGNORECASE
        )
        self.header_pattern = re.compile(
            r'(?i)(?:\n|^)\s*(?:held|judgment|order|facts|statute|background|issue|reasoning|conclusion)\s*:',
            re.IGNORECASE
        )

    def _token_count(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text.split())

    def split_by_legal_sections(self, text: str) -> List[Dict[str, str]]:
        """Split raw legal text into discrete structural section blocks with context headers."""
        if not isinstance(text, str) or not text.strip():
            return []

        # Find all match positions for sections and major headers
        split_indices = []
        for m in self.section_pattern.finditer(text):
            split_indices.append((m.start(), m.group(0).strip()))
        for m in self.header_pattern.finditer(text):
            split_indices.append((m.start(), m.group(0).strip()))

        # Sort split positions ascending
        split_indices.sort(key=lambda x: x[0])

        if not split_indices:
            # Fallback if no explicit legal headers found
            return [{"header": "General Legal Passage", "text": text.strip()}]

        blocks = []
        # Header text before first section
        if split_indices[0][0] > 0:
            prefix_text = text[:split_indices[0][0]].strip()
            if prefix_text:
                blocks.append({"header": "Legal Preamble", "text": prefix_text})

        for i in range(len(split_indices)):
            start_idx, header_title = split_indices[i]
            end_idx = split_indices[i+1][0] if i + 1 < len(split_indices) else len(text)
            section_content = text[start_idx:end_idx].strip()
            if section_content:
                blocks.append({"header": header_title, "text": section_content})

        return blocks

    def chunk_legal_text(self, text: str) -> List[str]:
        """Produce section-aware legal chunks with propagated headers within token limits."""
        if not isinstance(text, str) or not text.strip():
            return []

        blocks = self.split_by_legal_sections(text)
        final_chunks = []

        for block in blocks:
            header = block["header"]
            content = block["text"]
            header_prefix = f"[{header}] " if header else ""
            full_text = f"{header_prefix}{content}".strip()

            tokens_len = self._token_count(full_text)
            if tokens_len <= self.chunk_size:
                final_chunks.append(full_text)
            else:
                # Sub-chunk long section while prepending context header
                if self.encoder:
                    content_tokens = self.encoder.encode(content)
                    prefix_tokens = self.encoder.encode(header_prefix)
                    effective_max = self.chunk_size - len(prefix_tokens)
                    if effective_max <= 50:
                        effective_max = self.chunk_size

                    start = 0
                    while start < len(content_tokens):
                        end = min(start + effective_max, len(content_tokens))
                        sub_str = self.encoder.decode(content_tokens[start:end]).strip()
                        if sub_str:
                            final_chunks.append(f"{header_prefix}{sub_str}")
                        if end >= len(content_tokens):
                            break
                        start = end - self.overlap
                else:
                    # Character/word fallback
                    words = content.split()
                    chunk_words = self.chunk_size // 2
                    start = 0
                    while start < len(words):
                        sub_str = " ".join(words[start:start+chunk_words])
                        final_chunks.append(f"{header_prefix}{sub_str}")
                        start += chunk_words - 10

        return final_chunks

def chunk_statutory_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Convenience function for statutory section-aware text chunking."""
    chunker = StatutoryLegalChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_legal_text(text)

def run_main() -> None:
    """Sanity test for StatutoryLegalChunker on sample Indian statutory text."""
    sample_text = """
    CONSTITUTION OF INDIA
    
    Article 21: Protection of life and personal liberty.
    No person shall be deprived of his life or personal liberty except according to procedure established by law.
    
    INDIAN PENAL CODE (IPC)
    
    Section 302: Punishment for murder.
    Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.
    
    Section 420: Cheating and dishonestly inducing delivery of property.
    Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.
    
    Held:
    The Supreme Court affirmed that liberty under Article 21 extends to procedural safeguards under Criminal Procedure Code.
    """
    chunker = StatutoryLegalChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk_legal_text(sample_text)
    print(f"Generated {len(chunks)} statutory section chunks:")
    for idx, c in enumerate(chunks, 1):
        print(f"\n--- Chunk {idx} ---")
        print(c)

if __name__ == "__main__":
    run_main()
