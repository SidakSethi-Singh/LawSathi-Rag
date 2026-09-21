import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import config

logger = logging.getLogger(__name__)

# Statutory & Legal Terminology Mapping Dictionary for Indian Law
LEGAL_TERMINOLOGY_MAP = {
    "murder": "Section 300 Section 302 Indian Penal Code IPC culpable homicide",
    "kill": "Section 300 Section 302 Indian Penal Code IPC culpable homicide",
    "cheat": "Section 415 Section 420 Indian Penal Code IPC cheating property fraud",
    "cheating": "Section 415 Section 420 Indian Penal Code IPC cheating property fraud",
    "fraud": "Section 420 Indian Penal Code IPC cheating dishonesty fraud",
    "scam": "Section 420 Indian Penal Code IPC cheating fraud financial crime",
    "bail": "Section 436 Section 437 Section 438 Section 439 Code of Criminal Procedure CrPC anticipatory bail",
    "arrest": "Section 41 Section 437 Code of Criminal Procedure CrPC police custody",
    "liberty": "Article 21 Article 19 Constitution of India fundamental rights personal liberty",
    "fundamental right": "Article 12 Article 14 Article 19 Article 21 Article 32 Constitution of India",
    "divorce": "Hindu Marriage Act Section 13 Special Marriage Act dissolution of marriage",
    "maintenance": "Section 125 Code of Criminal Procedure CrPC maintenance wife children parents",
    "theft": "Section 378 Section 379 Indian Penal Code IPC theft stolen property",
    "extortion": "Section 383 Section 384 Indian Penal Code IPC extortion threat",
    "robbery": "Section 390 Section 392 Indian Penal Code IPC robbery dacoity",
    "defamation": "Section 499 Section 500 Indian Penal Code IPC criminal defamation reputation",
}

class LegalQueryExpander:
    """Legal Query Expansion & HyDE (Hypothetical Document Embeddings) Module."""

    def __init__(self, use_local_llm: Optional[bool] = None):
        self.use_local = use_local_llm if use_local_llm is not None else config.USE_LOCAL_MODEL

    def expand_legal_terms(self, query: str) -> str:
        """Expand user query with formal Indian legal terms and statutory section numbers."""
        if not query:
            return ""

        words = re_words = [w.lower().strip("?,.!") for w in query.split()]
        expansions = []
        for word in words:
            if word in LEGAL_TERMINOLOGY_MAP:
                expansions.append(LEGAL_TERMINOLOGY_MAP[word])

        if expansions:
            expanded_suffix = " ".join(set(" ".join(expansions).split()))
            return f"{query} {expanded_suffix}".strip()
        return query.strip()

    def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical legal precedent/statute passage for the query via LLM or fallback template."""
        system_prompt = (
            "You are a Supreme Court of India legal research expert. "
            "Write a short, hypothetical statutory provision or court ruling paragraph that directly answers the user's question."
        )
        prompt = f"Write a hypothetical court judgment passage or statutory section snippet for: '{query}'"

        try:
            if self.use_local:
                import requests
                url = f"{config.API_BASE_URL}/api/generate"
                payload = {
                    "model": "llama3.1",
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False
                }
                res = requests.post(url, json=payload, timeout=10.0)
                if res.status_code == 200:
                    hyde_text = res.json().get("response", "").strip()
                    if hyde_text:
                        return hyde_text
            else:
                if config.OPENAI_API_KEY:
                    from openai import OpenAI
                    client = OpenAI(base_url=config.API_BASE_URL, api_key=config.OPENAI_API_KEY)
                    res = client.chat.completions.create(
                        model=config.MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=256,
                        timeout=10.0
                    )
                    hyde_text = res.choices[0].message.content.strip()
                    if hyde_text:
                        return hyde_text
        except Exception as e:
            logger.warning(f"HyDE LLM generation fallback triggered: {e}")

        # Rule-based synthetic legal passage fallback if LLM is offline or no API key
        expanded = self.expand_legal_terms(query)
        return (
            f"Statutory Provision and Judicial Finding: In accordance with Indian statutory principles "
            f"and legal precedents regarding {query}, the Supreme Court observed that legal provisions under "
            f"{expanded} apply directly to establish rights, liabilities, and procedural rules."
        )

    def expand_query(self, query: str, use_hyde: bool = True) -> Dict[str, str]:
        """Produce expanded query representations for BM25 (lexical) and Vector (dense HyDE) search."""
        bm25_expanded = self.expand_legal_terms(query)
        vector_expanded = self.generate_hyde_document(query) if use_hyde else bm25_expanded
        return {
            "raw_query": query,
            "bm25_query": bm25_expanded,
            "vector_query": vector_expanded
        }

def run_main() -> None:
    """Sanity test for LegalQueryExpander."""
    expander = LegalQueryExpander()
    test_queries = [
        "What is the punishment for cheating in real estate deals?",
        "How to apply for anticipatory bail during police arrest?",
        "Is personal liberty protected under constitutional law?"
    ]
    for q in test_queries:
        res = expander.expand_query(q, use_hyde=True)
        print(f"\n=========================================")
        print(f"Original Query : {res['raw_query']}")
        print(f"BM25 Query     : {res['bm25_query']}")
        print(f"HyDE Vector Doc: {res['vector_query'][:200]}...")

if __name__ == "__main__":
    run_main()
