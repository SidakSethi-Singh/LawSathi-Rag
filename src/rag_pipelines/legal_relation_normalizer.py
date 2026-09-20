import re


def normalize_legal_relation_operators(text: str) -> str:
    """
    Canonicalize common Indian legal relation abbreviations in retrieval text.

    Supported examples:
    - u/s 302 -> under section 302
    - u / s 302 -> under section 302
    - r/w 34 -> read with section 34
    - r / w section 34 -> read with section 34
    - read along with section 34 -> read with section 34
    """
    if not text:
        return ""

    normalized = text

    normalized = re.sub(
        r"(?i)\bu\s*/\s*s\.?\s*(?:sections?|secs?\.?|s\.?|§)?\s*"
        r"(?=\d+)",
        "under section ",
        normalized,
    )
    normalized = re.sub(
        r"(?i)\br\s*/\s*w\.?\s*(?:sections?|secs?\.?|s\.?|§)?\s*"
        r"(?=\d+)",
        "read with section ",
        normalized,
    )
    normalized = re.sub(
        r"(?i)\bread\s+(?:and\s+)?along\s+with\s+"
        r"(?:sections?|secs?\.?|s\.?|§)?\s*"
        r"(?=\d+)",
        "read with section ",
        normalized,
    )

    normalized = re.sub(
        r"(?i)\bunder\s+(?:sections?|secs?\.?|s\.?|§)\s*",
        "under section ",
        normalized,
    )
    normalized = re.sub(
        r"(?i)\bread\s+with\s+(?:sections?|secs?\.?|s\.?|§)\s*",
        "read with section ",
        normalized,
    )

    return re.sub(r"[ \t]+", " ", normalized).strip()


def tokenize_legal_relation_text(text: str) -> list[str]:
    """Return stable BM25 tokens after legal relation normalization."""
    normalized = normalize_legal_relation_operators(text).casefold()
    normalized = re.sub(r"(?<=\d)[.,;:]+(?=\s|$)", "", normalized)
    return normalized.split()
