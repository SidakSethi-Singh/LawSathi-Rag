import re
from typing import Set


_SECTION_CITATION = re.compile(
    r"(?<!\w)(?:sections?|secs?\.?|s\.?|§)\s*"
    r"(\d+[A-Za-z]?(?:\s*\([^)]*\))*)",
    re.IGNORECASE,
)
_SCC_CITATION = re.compile(
    r"(?<!\w)(?:\(|\[)?(\d{4})(?:\)|\])?\s+"
    r"(\d+)\s+SCC\s+(\d+)(?!\w)",
    re.IGNORECASE,
)
_AIR_CITATION = re.compile(
    r"(?<!\w)AIR\s+(\d{4})\s+([A-Za-z]+)\s+(\d+)(?!\w)",
    re.IGNORECASE,
)


def extract_exact_citation_signatures(text: str) -> Set[str]:
    """Extract canonical legal citation identifiers for exact retrieval matching."""
    if not text:
        return set()

    signatures: Set[str] = set()

    for match in _SECTION_CITATION.finditer(text):
        reference = re.sub(r"\s+", "", match.group(1)).lower()
        signatures.add(f"section:{reference}")

    for match in _SCC_CITATION.finditer(text):
        year, volume, page = match.groups()
        signatures.add(f"scc:{year}:{volume}:{page}")

    for match in _AIR_CITATION.finditer(text):
        year, court, page = match.groups()
        signatures.add(f"air:{year}:{court.lower()}:{page}")

    return signatures
