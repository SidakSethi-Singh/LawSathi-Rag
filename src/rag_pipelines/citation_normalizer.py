import re
from typing import Dict, List

_REPORTER_ALIASES = (
    "scc online",
    "bom cr",
    "all mr",
    "comp cas",
    "cri lj",
    "mh lj",
    "taxmann",
    "scc",
    "scr",
    "air",
    "scale",
    "klt",
    "klj",
    "dlt",
    "manu",
    "ilr",
)

_REPORTER_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(_REPORTER_ALIASES, key=len, reverse=True)
)

_YEAR_FIRST_RE = re.compile(
    rf"(?<![A-Za-z0-9])"
    rf"(?:\((?P<year_paren>18|19|20)\d{{2}}\)|(?P<year>18|19|20)\d{{2}})"
    rf"\s+"
    rf"(?:(?P<volume>\d+)\s+)?"
    rf"(?P<reporter>{_REPORTER_PATTERN})"
    rf"(?:\s+(?P<court>[A-Za-z]{{2,8}}))?"
    rf"\s+(?P<page>\d+)"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_AIR_PATTERN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<reporter>air)"
    r"\s+(?P<year>18|19|20)\d{2}"
    r"\s+(?P<court>[A-Za-z]{2,8})"
    r"\s+(?P<page>\d+)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_PARTIAL_RE = re.compile(
    rf"(?<![A-Za-z0-9])"
    rf"(?P<reporter>{_REPORTER_PATTERN})"
    rf"\s+(?P<page>\d+)"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _clean_component(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip().lower()
    return cleaned or None


def _build_citation(
    *,
    year: str | None,
    volume: str | None,
    reporter: str | None,
    court: str | None,
    page: str | None,
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in (
        ("year", year),
        ("volume", volume),
        ("reporter", reporter),
        ("court", court),
        ("page", page),
    ):
        cleaned = _clean_component(value)
        if cleaned:
            result[key] = cleaned

    return result


def extract_reporter_citations(text: str) -> List[Dict[str, str]]:
    """Extract structured reporter citation components from legal text."""
    citations: List[Dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    for pattern in (_YEAR_FIRST_RE, _AIR_PATTERN_RE):
        for match in pattern.finditer(text):
            groups = match.groupdict()
            year = groups.get("year") or groups.get("year_paren")
            citation = _build_citation(
                year=year,
                volume=groups.get("volume"),
                reporter=groups.get("reporter"),
                court=groups.get("court"),
                page=groups.get("page"),
            )
            if citation:
                key = tuple(sorted(citation.items()))
                if key not in seen:
                    seen.add(key)
                    citations.append(citation)

    return citations


def citation_tokens(text: str) -> List[str]:
    """Create canonical BM25 tokens for full and partial reporter citations."""
    tokens: set[str] = set()

    for citation in extract_reporter_citations(text):
        for key, value in citation.items():
            tokens.add(f"__citation_{key}_{value.replace(' ', '_')}__")

        ordered = [
            citation.get("year"),
            citation.get("volume"),
            citation.get("reporter"),
            citation.get("court"),
            citation.get("page"),
        ]
        full = "_".join(part.replace(" ", "_") for part in ordered if part)
        if full:
            tokens.add(f"__citation_full_{full}__")

    for match in _PARTIAL_RE.finditer(text):
        reporter = _clean_component(match.group("reporter"))
        page = _clean_component(match.group("page"))
        if reporter:
            tokens.add(f"__citation_reporter_{reporter.replace(' ', '_')}__")
        if page:
            tokens.add(f"__citation_page_{page}__")

    return sorted(tokens)
