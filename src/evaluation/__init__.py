import re
from collections import Counter
from typing import List


def _tokens(text: str) -> List[str]:
    """Return normalized alphanumeric tokens, stripping punctuation."""
    return re.findall(r"\w+", str(text or "").lower())


def compute_f1(pred: str, gt: str) -> float:
    """Compute token-level F1 between a predicted and ground-truth answer.

    Uses Counter intersection so duplicate tokens are counted correctly.
    """
    p_tokens = _tokens(pred)
    g_tokens = _tokens(gt)
    if not p_tokens or not g_tokens:
        return 1.0 if p_tokens == g_tokens else 0.0

    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2.0 * precision * recall / (precision + recall)
