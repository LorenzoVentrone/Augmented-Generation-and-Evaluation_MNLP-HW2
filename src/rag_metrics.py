"""
Used by evaluate.py (full test-set scoring) and reusable for the LLM-judge subset.
All metrics take a single generated string and the list[str] of gold short answers,
and score 1/True if ANY gold matches (the dataset stores multiple valid forms).

Normalization removes the spurious artefacts seen in the gold labels during the
quick tests so that EM/sub-EM measure real correctness rather than formatting noise.
"""

import re
import string

# --- METEOR (nltk). wordnet/omw must be available offline on CINECA. ---
try:
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor
    _METEOR_AVAILABLE = True
except Exception:  # nltk not installed
    _METEOR_AVAILABLE = False


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_FOOTNOTE = re.compile(r"\[\s*\d+\s*\]")          # [4], [12], ...
_BRACKET_NOTE = re.compile(r"\[[^\]]*\]")          # [citation needed], [note 1]
_WS = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """
    SQuAD-style normalization.

    Lowercase, drop bracketed footnotes/notes, remove punctuation and articles,
    and collapse whitespace.
    """
    text = text.lower()
    text = _FOOTNOTE.sub(" ", text)
    text = _BRACKET_NOTE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def exact_match(generated: str, golds: list[str]) -> bool:
    """1 if the normalized generation equals any normalized gold answer."""
    g = normalize_answer(generated)
    return any(g == normalize_answer(a) for a in golds)


def sub_match(generated: str, golds: list[str]) -> bool:
    """1 if any normalized gold answer is a substring of the normalized generation."""
    g = normalize_answer(generated)
    return any(normalize_answer(a) in g for a in golds if normalize_answer(a))


def meteor(generated: str, golds: list[str]) -> float:
    """
    Max METEOR over the gold answers. Tokenized by whitespace on the lightly
    lowercased text (no punctuation stripping — METEOR handles morphology via
    wordnet). Returns 0.0 if nltk/wordnet is unavailable (logged once by caller).
    """
    if not _METEOR_AVAILABLE:
        return 0.0
    hyp = generated.lower().split()
    refs = [a.lower().split() for a in golds if a.strip()]
    if not refs or not hyp:
        return 0.0
    return float(_nltk_meteor(refs, hyp))


def meteor_available() -> bool:
    return _METEOR_AVAILABLE


def score_all(generated: str, golds: list[str]) -> dict:
    return {
        "em": int(exact_match(generated, golds)),
        "sub_em": int(sub_match(generated, golds)),
        "meteor": meteor(generated, golds),
    }
