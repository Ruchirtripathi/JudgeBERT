"""Language-aware, legally-conservative text preprocessing.

The guiding constraint
----------------------
Legal meaning lives in exactly the tokens that aggressive NLP preprocessing
throws away: numbers, dates, negation, modal verbs, quantifiers, conditions
and exceptions.  Turning ``shall not`` into ``shall`` or dropping ``30``
would erase the signal the model is supposed to detect, so every destructive
transform here is off by default and must be enabled explicitly in
``configs/config.yaml``.

Consequently this module does far less than a typical text-cleaning pipeline.
That is a deliberate design decision, and it is defended in the paper.

Two views of the text are always retained:
  * ``raw``    — the original string, offsets valid, used for explanations;
  * ``clean``  — whitespace-normalised, used for matching and vectorisation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Sequence

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults (mirrors configs/config.yaml -> preprocessing)
# ---------------------------------------------------------------------------
DEFAULT_PREPROCESSING_CONFIG: dict = {
    "unicode_nfkc": True,
    "strip_whitespace": True,
    "collapse_whitespace": True,
    "lowercase": False,
    "remove_punctuation": False,
    "remove_digits": False,
    "strip_accents": False,
}

#: Characters that terminate a sentence.  Devanagari uses the danda (।) and
#: double danda (॥) where English uses a full stop.
SENTENCE_TERMINATORS = ".!?।॥"

#: Characters that may appear *inside* a word.
#:
#: ``\w`` alone is not enough, and assuming it was is an easy and costly
#: mistake.  Python's ``\w`` is Unicode-aware but matches only characters with
#: ``str.isalnum() == True``.  Devanagari *matras* and the virama are combining
#: marks (Unicode category Mn/Mc), for which ``isalnum()`` is **False** — so
#: ``\w+`` matched only the consonants of a Hindi word and stopped at every
#: vowel sign, shattering ``क्रेता`` into ``क`` ``्`` ``र`` ``े`` ``त`` ``ा``.
#:
#: Everything downstream inherited that: ``token_jaccard`` (used for
#: near-duplicate leakage detection) and ``length_ratio`` (used for the
#: omission/hallucination heuristics) were silently computed over *characters*
#: for Hindi and over *words* for English — a cross-lingual inconsistency in
#: exactly the comparison Experiment 2 sets out to make.
#:
#: The ranges below add the marks back.  U+0964 (danda ``।``) and U+0965
#: (double danda ``॥``) are deliberately **excluded**: they are sentence
#: terminators, and are handled as punctuation just like ``.`` in English.
#: ZWJ/ZWNJ are included because they legitimately occur inside Devanagari
#: words in some fonts.
_WORD_CHARS = "‌‍" + "".join(
    f"{chr(lo)}-{chr(hi)}"
    for lo, hi in (
        (0x0900, 0x0963),   # Devanagari letters, vowel signs, virama, anusvara
        (0x0966, 0x097F),   # Devanagari digits, letters, signs
    )
)

#: Token pattern.  Words first (letters plus any combining marks), then any
#: single remaining non-space character as its own punctuation token.  Keeping
#: digits as tokens is intentional: they carry legal weight.
_TOKEN_RE = re.compile(
    rf"[\w{_WORD_CHARS}]+(?:['’\-][\w{_WORD_CHARS}]+)*|[^\w{_WORD_CHARS}\s]",
    flags=re.UNICODE,
)

#: Words that must never be removed even if a caller enables aggressive
#: cleaning.  Sourced from the config's ``protected_patterns`` by default.
_FALLBACK_PROTECTED: frozenset[str] = frozenset(
    {
        "shall", "must", "may", "can", "will", "should", "not", "no", "never",
        "if", "unless", "except", "all", "any", "none", "only",
        "नहीं", "बिना", "यदि", "सिवाय", "सभी", "केवल",
    }
)


def _protected_tokens(preproc_cfg: Mapping | None) -> frozenset[str]:
    """Collect the protected vocabulary from the config, lower-cased."""
    if not preproc_cfg:
        return _FALLBACK_PROTECTED
    patterns = preproc_cfg.get("protected_patterns") or {}
    tokens: set[str] = set(_FALLBACK_PROTECTED)
    for group in patterns.values():
        for token in group or []:
            tokens.add(str(token).lower())
    return frozenset(tokens)


def _contains_whole_token(haystack: str, needle: str) -> bool:
    """Whole-token containment that works on Devanagari.

    ``\\b`` cannot be used here, for the same reason ``\\w`` cannot be used in
    :data:`_TOKEN_RE`.  Python defines a word boundary in terms of ``\\w``, and
    Devanagari matras are combining marks that ``\\w`` does not match — so
    ``\\bनहीं\\b`` fails to match a sentence-final ``नहीं`` precisely because it
    is sentence-final.  The boundary is therefore tested by hand with
    ``str.isalnum()``, which is what :func:`extract_entities` in
    ``legal_cues`` does for the same reason.
    """
    needle = needle.strip().lower()
    if not needle:
        return False
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            return False
        before_ok = index == 0 or not haystack[index - 1].isalnum()
        after = index + len(needle)
        after_ok = after >= len(haystack) or not haystack[after].isalnum()
        if before_ok and after_ok:
            return True
        start = index + 1


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_text(text: str, cfg: Mapping | None = None) -> str:
    """Apply the configured, non-destructive normalisation.

    Returns the *clean* view.  Never raises on non-string input; a missing
    value becomes an empty string so that downstream code produces a clear
    validation error rather than an obscure ``AttributeError``.
    """
    if not isinstance(text, str):
        return ""

    preproc = dict(DEFAULT_PREPROCESSING_CONFIG)
    if cfg:
        preproc.update({k: v for k, v in cfg.items() if k in DEFAULT_PREPROCESSING_CONFIG})

    out = text

    if preproc.get("unicode_nfkc", True):
        # NFKC (not NFC) because statutes are frequently copied from PDFs
        # where Devanagari matras are decomposed and Latin punctuation is
        # full-width.  NFKC folds both into a comparable form.
        out = unicodedata.normalize("NFKC", out)

    if preproc.get("strip_accents"):
        # Decompose, drop combining marks, recompose.  Off by default: this
        # would destroy Devanagari vowel signs entirely.
        out = "".join(
            ch for ch in unicodedata.normalize("NFD", out)
            if not unicodedata.combining(ch)
        )
        out = unicodedata.normalize("NFC", out)

    if preproc.get("lowercase"):
        out = out.lower()

    if preproc.get("remove_punctuation"):
        out = re.sub(r"[^\w\s]", " ", out, flags=re.UNICODE)

    if preproc.get("remove_digits"):
        out = re.sub(r"\d", " ", out)
        out = re.sub(r"[०-९]", " ", out)

    if preproc.get("strip_whitespace", True):
        out = out.strip()

    if preproc.get("collapse_whitespace", True):
        out = re.sub(r"\s+", " ", out)

    return out


def tokenize(text: str, language: str = "en") -> list[str]:
    """Tokenise into words and punctuation, Unicode-aware.

    Deliberately *not* a language-specific morphological analyser.  A Hindi
    stemmer would collapse ``करेगा`` (shall do) and ``कर सकता है`` (may do)
    toward the same root, which is precisely the distinction under study.

    Parameters
    ----------
    text
        Input string; the caller decides whether it is raw or clean.
    language
        ``"en"`` or ``"hi"``.  Currently both use the same Unicode-aware
        pattern; the parameter exists so a future language-specific tokeniser
        can be slotted in without changing call sites.
    """
    if not isinstance(text, str) or not text:
        return []
    return _TOKEN_RE.findall(text)


def content_tokens(text: str, language: str = "en") -> list[str]:
    """Tokens with punctuation removed — the bag used for lexical overlap."""
    punctuation = set(".,;:!?()[]{}\"'`«»„“”‘’—–-/\\|@#$%^&*_+=~<>।॥")
    return [t for t in tokenize(text, language) if t not in punctuation]


def sentence_split(text: str, language: str = "en") -> list[str]:
    """Split a passage into sentences at English and Devanagari terminators.

    Used to build the ``context`` field (neighbouring sub-clauses) and to
    segment source statutes into candidate sentences.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    terminators = re.escape(SENTENCE_TERMINATORS)
    # Split after a terminator followed by whitespace, or at a newline.
    parts = re.split(rf"(?<=[{terminators}])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# Similarity primitives (used by leakage checks, not as model features)
# ---------------------------------------------------------------------------
def token_jaccard(a: str, b: str, language: str = "en") -> float:
    """Jaccard similarity of the lower-cased token sets of two strings.

    Used by the split logic to detect near-duplicate source sentences that
    would otherwise leak across train and test.
    """
    set_a = {t.lower() for t in content_tokens(a, language)}
    set_b = {t.lower() for t in content_tokens(b, language)}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams over a whitespace-collapsed string."""
    collapsed = re.sub(r"\s+", " ", normalize_text(text)).lower()
    if len(collapsed) < n:
        return [collapsed] if collapsed else []
    return [collapsed[i : i + n] for i in range(len(collapsed) - n + 1)]


def length_ratio(original: str, simplified: str) -> float:
    """Token-count ratio simplified/original.

    A very low ratio suggests omission; a very high ratio suggests added
    material.  Used as a weak signal by the rule-based error detector, never
    as a standalone verdict.
    """
    original_len = len(content_tokens(original))
    simplified_len = len(content_tokens(simplified))
    if original_len == 0:
        return float("inf") if simplified_len else 1.0
    return simplified_len / original_len


# ---------------------------------------------------------------------------
# Frame-level application
# ---------------------------------------------------------------------------
def add_clean_columns(df, text_columns: Sequence[str] = ("original_text", "simplified_text"),
                      cfg: Mapping | None = None):
    """Add ``<col>_clean`` and ``<col>_tokens`` columns to a DataFrame.

    The raw columns are left untouched so that explanations always cite the
    original wording.  Imported lazily to keep pandas out of the module
    import path for callers that only need tokenisation.
    """
    df = df.copy()
    for column in text_columns:
        if column not in df.columns:
            logger.warning("Column %s not present; skipping clean-view creation", column)
            continue
        df[f"{column}_clean"] = df[column].map(lambda t: normalize_text(t, cfg))
    return df


def preprocess_pair(
    original: str,
    simplified: str,
    language: str = "en",
    cfg: Mapping | None = None,
) -> dict:
    """Preprocess one pair, returning both views plus lexical diagnostics.

    This is the function the inference service calls at request time, so its
    output shape is a de facto API contract.
    """
    original_clean = normalize_text(original, cfg)
    simplified_clean = normalize_text(simplified, cfg)
    return {
        "original_raw": original,
        "simplified_raw": simplified,
        "original_clean": original_clean,
        "simplified_clean": simplified_clean,
        "original_tokens": tokenize(original_clean, language),
        "simplified_tokens": tokenize(simplified_clean, language),
        "token_jaccard": token_jaccard(original_clean, simplified_clean, language),
        "length_ratio": length_ratio(original_clean, simplified_clean),
        "language": language,
    }


def assert_legal_content_preserved(
    original: str,
    processed: str,
    language: str = "en",
    cfg: Mapping | None = None,
) -> list[str]:
    """Check that preprocessing did not destroy legally salient content.

    Returns a list of warnings (empty when clean).  Called by the test suite
    so that enabling a destructive flag in the config fails loudly rather
    than silently degrading every experiment.

    Two independent checks are run, because they fail differently:

    1. **Cue count** — extracted modals, negations, conditions, exceptions and
       quantities must not decrease.  This catches a transform that mangles a
       cue into something the extractor no longer recognises.
    2. **Protected vocabulary** — every token listed under
       ``preprocessing.protected_patterns`` that was present in the original
       must still be present in the processed text.  This catches a transform
       that deletes a protected word outright, which the cue extractor might
       miss if the word is legally salient but not one of its patterns.

    Both are necessary.  ``protected_patterns`` was previously consulted by
    nothing at all, so the config advertised a guarantee that no code
    enforced; check 2 is what makes that setting load-bearing.
    """
    warnings: list[str] = []
    from src.data.legal_cues import extract_cues

    before = extract_cues(original, language)
    after = extract_cues(processed, language)

    for attribute in ("modals", "negations", "conditions", "exceptions"):
        n_before = len(getattr(before, attribute))
        n_after = len(getattr(after, attribute))
        if n_after < n_before:
            warnings.append(
                f"preprocessing removed {n_before - n_after} {attribute} "
                f"(from {n_before} to {n_after})"
            )

    if len(before.quantities) > len(after.quantities):
        warnings.append(
            f"preprocessing removed {len(before.quantities) - len(after.quantities)} quantities"
        )

    preproc_cfg = (cfg or {}).get("preprocessing") if cfg else None
    # Comparison is done on the NFKC-normalised, lower-cased form of both
    # sides: the normalisation is itself part of preprocessing, and a
    # lowercase transform must not be reported as a deleted token.
    haystack = normalize_text(processed).lower()
    for token in sorted(_protected_tokens(preproc_cfg)):
        if _contains_whole_token(original.lower(), token) and not _contains_whole_token(
            haystack, token
        ):
            warnings.append(f"preprocessing removed the protected term {token!r}")

    return warnings
