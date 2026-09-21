import re
from typing import List, Tuple


_SECTION_REFERENCE = re.compile(
    r"(?<!\w)(?:sections?|secs?\.?|s\.?|§)\s*"
    r"(\d+[A-Za-z]?(?:\s*\([^)]*\))*"
    r"(?:\s*(?:-|–|—|to)\s*\d+[A-Za-z]?)?)(?!\w)",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(r"\(([^)]*)\)")
_RANGE = re.compile(r"^(\d+)(?:[A-Za-z]?)\s*(?:-|–|—|to)\s*(\d+)(?:[A-Za-z]?)$", re.IGNORECASE)


def _reference_tokens(reference: str) -> List[str]:
    normalized = re.sub(r"\s+", "", reference)
    range_match = _RANGE.fullmatch(normalized)

    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        tokens = [f"legal_section_range_{start}_{end}"]
        if start <= end and end - start <= 20:
            tokens.extend(f"legal_section_{number}" for number in range(start, end + 1))
        return tokens

    base_match = re.match(r"^(\d+[A-Za-z]?)", normalized)
    if not base_match:
        return []

    section = base_match.group(1).lower()
    tokens = [f"legal_section_{section}"]

    parts = _PARENTHETICAL.findall(normalized)
    hierarchy = [section]
    for index, part in enumerate(parts):
        value = re.sub(r"\s+", "", part).lower()
        if not value:
            continue
        hierarchy.append(value)
        prefix = "legal_subsection" if index == 0 else "legal_clause"
        tokens.append(f"{prefix}_{'_'.join(hierarchy)}")

    return tokens


def normalize_section_references(text: str) -> str:
    """Append canonical section, subsection, clause, and range tokens for retrieval."""
    if not text:
        return ""

    matches = list(_SECTION_REFERENCE.finditer(text))
    if not matches:
        return text

    canonical_tokens: List[str] = []
    for match in matches:
        canonical_tokens.extend(_reference_tokens(match.group(1)))

    if not canonical_tokens:
        return text

    unique_tokens = list(dict.fromkeys(canonical_tokens))
    return f"{text} {' '.join(unique_tokens)}"


def tokenize_legal_reference_text(text: str) -> List[str]:
    """Normalize section references and return whitespace-separated retrieval tokens."""
    return normalize_section_references(text).casefold().split()
