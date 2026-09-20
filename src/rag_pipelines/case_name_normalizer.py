import re
import unicodedata
from typing import List

_CASE_SEPARATOR_RE = re.compile(r"\b(?:v|vs|versus)\.?\b", re.IGNORECASE)
_CASE_CUE_RE = re.compile(
    r"\b(?:the\s+)?(?:case|matter)\s+of\s+",
    re.IGNORECASE,
)
_PARTY_SUFFIX_ALIASES = (
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\bprivate\s+ltd\b", "pvt ltd"),
    (r"\bpvt\s+limited\b", "pvt ltd"),
    (r"\blimited\b", "ltd"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcompany\b", "co"),
)


def normalize_case_name(value: str) -> str:
    """Canonicalize common case-name separators, punctuation, and party suffixes."""
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("&", " and ")
    text = re.sub(r"\bm\s*/\s*s\.?\b", "ms", text, flags=re.IGNORECASE)
    text = re.sub(r"\bv(?:s|s\.|ersus)\b", " v ", text, flags=re.IGNORECASE)
    for pattern, replacement in _PARTY_SUFFIX_ALIASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = text.replace("@", " alias ")
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_case_name_candidates(text: str) -> List[str]:
    """Extract canonical case-name candidates surrounding v/vs/versus separators."""
    candidates: List[str] = []
    for match in _CASE_SEPARATOR_RE.finditer(text):
        prefix = text[:match.start()]
        suffix = text[match.end():]

        cue_matches = list(_CASE_CUE_RE.finditer(prefix))
        if cue_matches:
            left = prefix[cue_matches[-1].end():]
        else:
            left = re.split(r"[?!;:\n]", prefix)[-1]

        right = re.split(r"[?!;:\n]", suffix)[0]
        candidate = normalize_case_name(f"{left} v {right}")
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    return candidates


def case_name_tokens(text: str) -> List[str]:
    """Return canonical BM25 tokens for case-name variants in arbitrary text."""
    candidates = extract_case_name_candidates(text)
    return [f"__case_name_{candidate.replace(' ', '_')}__" for candidate in candidates]
