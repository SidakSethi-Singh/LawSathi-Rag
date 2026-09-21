import re
import unicodedata

_CANONICAL_COURTS = {
    "supreme court": "supreme_court_of_india",
    "supreme court of india": "supreme_court_of_india",
    "sc": "supreme_court_of_india",
    "delhi high court": "delhi_high_court",
    "high court of delhi": "delhi_high_court",
    "high court at delhi": "delhi_high_court",
    "delhi hc": "delhi_high_court",
    "bombay high court": "bombay_high_court",
    "high court of bombay": "bombay_high_court",
    "mumbai high court": "bombay_high_court",
    "bombay hc": "bombay_high_court",
    "madras high court": "madras_high_court",
    "high court of madras": "madras_high_court",
    "madras hc": "madras_high_court",
    "calcutta high court": "calcutta_high_court",
    "high court at calcutta": "calcutta_high_court",
    "calcutta hc": "calcutta_high_court",
    "karnataka high court": "karnataka_high_court",
    "high court of karnataka": "karnataka_high_court",
    "karnataka hc": "karnataka_high_court",
    "kerala high court": "kerala_high_court",
    "high court of kerala": "kerala_high_court",
    "kerala hc": "kerala_high_court",
    "allahabad high court": "allahabad_high_court",
    "high court of judicature at allahabad": "allahabad_high_court",
    "high court at allahabad": "allahabad_high_court",
    "allahabad hc": "allahabad_high_court",
    "rajasthan high court": "rajasthan_high_court",
    "high court of rajasthan": "rajasthan_high_court",
    "rajasthan hc": "rajasthan_high_court",
    "gujarat high court": "gujarat_high_court",
    "high court of gujarat": "gujarat_high_court",
    "gujarat hc": "gujarat_high_court",
    "madhya pradesh high court": "madhya_pradesh_high_court",
    "high court of madhya pradesh": "madhya_pradesh_high_court",
    "madhya pradesh hc": "madhya_pradesh_high_court",
    "punjab and haryana high court": "punjab_and_haryana_high_court",
    "high court of punjab and haryana": "punjab_and_haryana_high_court",
    "punjab and haryana high court": "punjab_and_haryana_high_court",
    "punjab and haryana hc": "punjab_and_haryana_high_court",
    "patna high court": "patna_high_court",
    "high court of patna": "patna_high_court",
    "patna hc": "patna_high_court",
    "jharkhand high court": "jharkhand_high_court",
    "high court of jharkhand": "jharkhand_high_court",
    "jharkhand hc": "jharkhand_high_court",
    "odisha high court": "odisha_high_court",
    "orissa high court": "odisha_high_court",
    "high court of odisha": "odisha_high_court",
    "high court of orissa": "odisha_high_court",
    "orissa hc": "odisha_high_court",
    "chhattisgarh high court": "chhattisgarh_high_court",
    "high court of chhattisgarh": "chhattisgarh_high_court",
    "chhattisgarh hc": "chhattisgarh_high_court",
    "telangana high court": "telangana_high_court",
    "high court for the state of telangana": "telangana_high_court",
    "telangana hc": "telangana_high_court",
    "andhra pradesh high court": "andhra_pradesh_high_court",
    "high court of andhra pradesh": "andhra_pradesh_high_court",
    "andhra pradesh hc": "andhra_pradesh_high_court",
    "uttarakhand high court": "uttarakhand_high_court",
    "high court of uttarakhand": "uttarakhand_high_court",
    "uttarakhand hc": "uttarakhand_high_court",
    "himachal pradesh high court": "himachal_pradesh_high_court",
    "high court of himachal pradesh": "himachal_pradesh_high_court",
    "himachal pradesh hc": "himachal_pradesh_high_court",
}

def _clean(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("&", " and ")
    value = re.sub(r"[^A-Za-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def normalize_court_name(value: str) -> str:
    """Map common Indian court names and abbreviations to stable identifiers."""
    cleaned = _clean(value)
    if not cleaned:
        return ""

    direct = _CANONICAL_COURTS.get(cleaned)
    if direct:
        return direct

    if cleaned.startswith("high court of "):
        candidate = f"{cleaned[14:]} high court"
        direct = _CANONICAL_COURTS.get(candidate)
        if direct:
            return direct

    if cleaned.startswith("high court at "):
        candidate = f"{cleaned[14:]} high court"
        direct = _CANONICAL_COURTS.get(candidate)
        if direct:
            return direct

    if cleaned.endswith(" hc"):
        candidate = f"{cleaned[:-3].strip()} high court"
        direct = _CANONICAL_COURTS.get(candidate)
        if direct:
            return direct

    return cleaned.replace(" ", "_")


def extract_court_name(record: dict) -> str:
    """Read and canonicalize a court value from common record locations."""
    sources = [record]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        sources.append(metadata)

    for source in sources:
        for key in ("court", "court_name", "courtName"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return normalize_court_name(str(value))

    return ""
