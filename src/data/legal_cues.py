"""Legally salient cue extraction for English and Hindi.

Generic sentence-similarity models are blind to the distinctions that decide
legal meaning: ``must`` vs ``may``, ``30 days`` vs ``60 days``, ``all`` vs
``some``, the presence or absence of ``unless``.  This module extracts those
cues explicitly so that

  * the rule-assisted error detector can name what changed,
  * the explainability layer can point at the exact span responsible,
  * Experiment 4 can construct controlled minimal pair perturbations.

Design notes
------------
* Extraction is **surface-based and auditable**.  Every cue records the
  character offsets it came from, so an explanation can highlight the source
  text rather than asserting a cause.
* Quantities are normalised to canonical units (``seven days`` == ``7 days``
  == ``1 week``), because a duration-preserving paraphrase is *not* a legal
  meaning change.  This deliberately avoids false-positive error flags.
* Lexicons are intentionally small and high-precision rather than large and
  fuzzy.  A missed cue costs recall in the error detector; a spurious cue
  corrupts a reported legal finding.  The second failure is worse.
* Quantity units are read **after** the number, which is the form this corpus
  uses throughout (``five hundred rupees``).  A currency prefix is therefore
  read as a bare count: ``Rs. 500`` yields 500 with unit ``count`` rather than
  ``currency``.  The amount is still correct, so a changed figure is still
  detected; only the unit label differs.  The prefix form does not occur in
  the corpus, and widening the pattern for it would loosen the boundaries that
  keep ``one of the parties`` from becoming a quantity.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Categories and strengths
# ---------------------------------------------------------------------------

CATEGORY_MODAL = "modal"
CATEGORY_NEGATION = "negation"
CATEGORY_CONDITION = "condition"
CATEGORY_EXCEPTION = "exception"
CATEGORY_QUANTIFIER = "quantifier"
CATEGORY_SCOPE = "scope"
CATEGORY_QUANTITY = "quantity"
CATEGORY_ENTITY = "entity"

#: Deontic strength.  Higher = more restrictive / more demanding.
#: A *decrease* from original to simplified means the reformulation weakened
#: the norm (obligation -> permission), which is the classic meaning change.
STRENGTH_PROHIBITION = 4
STRENGTH_OBLIGATION = 3
STRENGTH_RECOMMENDATION = 2
STRENGTH_PERMISSION = 1
STRENGTH_DISCRETION = 0

STRENGTH_LABELS: dict[int, str] = {
    STRENGTH_PROHIBITION: "Prohibition",
    STRENGTH_OBLIGATION: "Obligation",
    STRENGTH_RECOMMENDATION: "Recommendation",
    STRENGTH_PERMISSION: "Permission",
    STRENGTH_DISCRETION: "Discretion",
}

# Quantifier strength: universal -> existential.
QUANT_UNIVERSAL = 3
QUANT_EXISTENTIAL = 1


# ---------------------------------------------------------------------------
# Devanagari word boundaries
# ---------------------------------------------------------------------------
#: Every Devanagari codepoint, including the combining matras (category Mc/Mn)
#: and the virama.
#:
#: ``\b`` cannot be used to delimit a Hindi word.  It is defined in terms of
#: ``\w``, and ``\w`` does not match combining marks — ``'ा'.isalnum()`` is
#: ``False`` — so ``\b`` fires in the middle of a Devanagari word.  A pattern
#: as ordinary as ``न\b`` therefore matches the final letter of करना and the
#: first letter of निर्वहन, and ``मत\b`` matches the tail of सहमति and सहमत.
#: On this corpus that produced 2,066 negation cues where 121 real ones exist.
#:
#: The failure is not symmetric.  A phantom cue makes ``negation_flipped``
#: compare two counts that are both non-zero, so a simplification that
#: genuinely drops a नहीं is reported as leaving negation untouched — the
#: extractor goes blind to the exact change it exists to find.
_DEVA_CLASS = r"[ऀ-ॿ]"


def _deva_word(pattern: str) -> str:
    """Anchor ``pattern`` so it cannot match inside a Devanagari word.

    Use this for a Hindi cue whose surface form is also a substring of an
    unrelated word.  It must *not* be used for a stem whose inflections are
    the same lexeme and should keep matching (``भाग`` in ``भागों``), because
    the anchoring would then delete good cues rather than spurious ones.
    """
    return rf"(?<!{_DEVA_CLASS}){pattern}(?!{_DEVA_CLASS})"


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
# Each entry is (regex pattern, canonical label, strength).  Patterns are
# matched case-insensitively.  Order within a list matters: the first match
# wins, so more specific patterns (e.g. "shall not") must precede general
# ones (e.g. "shall").

_MODALS_EN: list[tuple[str, str, int]] = [
    (r"\bshall\s+not\b|\bmust\s+not\b|\bmay\s+not\b|\bwill\s+not\b", "shall not", STRENGTH_PROHIBITION),
    (r"\b(?:is|are|shall\s+be)\s+prohibited\b", "prohibited", STRENGTH_PROHIBITION),
    (r"\b(?:is|are)\s+forbidden\b|\bforbidden\b", "forbidden", STRENGTH_PROHIBITION),
    (r"\bnot\s+permitted\b|\bnot\s+allowed\b", "not permitted", STRENGTH_PROHIBITION),
    (r"\b(?:is|are)\s+not\s+entitled\b", "not entitled", STRENGTH_PROHIBITION),
    (r"\bshall\b", "shall", STRENGTH_OBLIGATION),
    (r"\bmust\b", "must", STRENGTH_OBLIGATION),
    (r"\b(?:is|are)\s+required\s+to\b|\brequired\s+to\b", "required to", STRENGTH_OBLIGATION),
    (r"\b(?:is|are)\s+liable\s+to\b|\bliability\s+to\b", "liable to", STRENGTH_OBLIGATION),
    (r"\b(?:is|are)\s+obligated\b|\bobliged\s+to\b|\bobligation\s+to\b", "obligated to", STRENGTH_OBLIGATION),
    (r"\b(?:is|are)\s+bound\s+to\b|\bbound\s+to\b", "bound to", STRENGTH_OBLIGATION),
    (r"\bhas\s+to\b|\bhave\s+to\b|\bhad\s+to\b", "has to", STRENGTH_OBLIGATION),
    (r"\bundertakes\s+to\b|\bagrees\s+to\b|\bcovenants\s+to\b", "undertakes to", STRENGTH_OBLIGATION),
    (r"\bshall\s+be\s+entitled\b", "shall be entitled", STRENGTH_PERMISSION),
    (r"\bshould\b", "should", STRENGTH_RECOMMENDATION),
    (r"\bought\s+to\b", "ought to", STRENGTH_RECOMMENDATION),
    (r"\b(?:is|are)\s+expected\s+to\b", "expected to", STRENGTH_RECOMMENDATION),
    (r"\bmay\b", "may", STRENGTH_PERMISSION),
    (r"\b(?:is|are)\s+permitted\s+to\b|\bpermitted\s+to\b", "permitted to", STRENGTH_PERMISSION),
    (r"\b(?:is|are)\s+entitled\s+to\b|\bentitled\s+to\b", "entitled to", STRENGTH_PERMISSION),
    (r"\b(?:is|are)\s+authori[sz]ed\s+to\b|\bauthori[sz]ed\s+to\b", "authorised to", STRENGTH_PERMISSION),
    (r"\b(?:is|are)\s+at\s+liberty\s+to\b", "at liberty to", STRENGTH_PERMISSION),
    (r"\bcan\b", "can", STRENGTH_PERMISSION),
    (r"\bmay\s+opt\s+to\b", "may opt to", STRENGTH_PERMISSION),
    (r"\bwill\b", "will", STRENGTH_DISCRETION),
    (r"\bwould\b|\bcould\b|\bmight\b", "would", STRENGTH_DISCRETION),
]

# Hindi expresses modality largely through verbal suffixes and compound
# verbs, so these patterns are surface-form based.  "सकता है" (can/may) is
# permission; "करना होगा" / "चाहिए" is obligation.  The order below puts
# prohibitions first, then obligations, then permissions.
_MODALS_HI: list[tuple[str, str, int]] = [
    (r"नहीं\s+कर\s+सकत[ाेी]|नहीं\s+किया\s+जा\s+सकता", "नहीं कर सकता", STRENGTH_PROHIBITION),
    (r"प्रतिबंधित|निषिद्ध|वर्जित|मना\s+है|अनुमति\s+नहीं", "प्रतिबंधित", STRENGTH_PROHIBITION),
    (r"बाध्य\s+होगा|बाध्य\s+है|अनिवार्य\s+है|अनिवार्य\s+रूप\s+से", "अनिवार्य", STRENGTH_OBLIGATION),
    (r"करना\s+होगा|करनी\s+होगी|करने\s+होगे|देना\s+होगा|देनी\s+होगी|भुगतान\s+करना\s+होगा", "करना होगा", STRENGTH_OBLIGATION),
    (r"चाहिए|चाहिये", "चाहिए", STRENGTH_OBLIGATION),
    (r"आवश्यक\s+है|अपेक्षित\s+है|उत्तरदायी\s+होगा|उत्तरदायी\s+है|दायित्व\s+है", "आवश्यक है", STRENGTH_OBLIGATION),
    (r"करना\s+आवश्यक|करने\s+के\s+लिए\s+बाध्य", "करना आवश्यक", STRENGTH_OBLIGATION),
    # करेगा / करेगी / करेंगे — simple future used prescriptively in statutes.
    (r"करेगा|करेगी|करेंगे|देगा|देगी|देंगे|होगा\b(?![^।]*नहीं)", "करेगा", STRENGTH_OBLIGATION),
    (r"सकता\s+है|सकती\s+है|सकते\s+हैं|सकेगा|सकेगी|सकेंगे", "सकता है", STRENGTH_PERMISSION),
    (r"अधिकार\s+होगा|अधिकार\s+है|हक\s+है|अनुमति\s+है|इजाजत\s+है", "अधिकार है", STRENGTH_PERMISSION),
    (r"कर\s+सकता|कर\s+सकती|कर\s+सकते", "कर सकता", STRENGTH_PERMISSION),
    (r"चाहिए\s+कि\s+नहीं|विवेकाधीन|वैकल्पिक\s+है", "वैकल्पिक", STRENGTH_DISCRETION),
]

_NEGATIONS_EN = r"\b(?:not|no|never|neither|nor|cannot|without|none|nothing|n't)\b"
# नहीं/बिना/कदापि/रहित/वंचित are unambiguous strings.  मत and न are not: they
# occur inside सहमति, सहमत, सम्मति, करना, पालन, निर्वहन and hundreds of other
# words, so both are anchored.  See _deva_word.
_NEGATIONS_HI = (
    r"नहीं|बिना|कदापि|रहित|वंचित"
    rf"|{_deva_word('मत')}"
    rf"|{_deva_word('न')}"
)

_CONDITIONS_EN = (
    r"\bif\b|\bunless\b|\bprovided\s+that\b|\bprovided\s+further\b|\bsubject\s+to\b|"
    r"\bin\s+the\s+event\s+(?:of|that)\b|\bon\s+condition\s+that\b|\bwhen\b|\bwhere\b|"
    r"\bin\s+case\b|\bso\s+long\s+as\b|\bconditional\s+upon\b|\bcontingent\s+upon\b"
)
_CONDITIONS_HI = (
    r"यदि|अगर|जब|जबकि|बशर्ते|शर्त|प्रदान\s+किया\s+गया\s+है\s+कि|"
    r"की\s+स्थिति\s+में|के\s+मामले\s+में|इस\s+शर्त\s+पर"
)

_EXCEPTIONS_EN = (
    r"\bexcept\b|\bexception\b|\bother\s+than\b|\bsave\s+as\s+otherwise\b|"
    r"\bnotwithstanding\b|\bexcluding\b|\bbut\s+not\b|\bsave\s+and\s+except\b|"
    r"\bapart\s+from\b|\bunless\s+otherwise\b"
)
_EXCEPTIONS_HI = r"सिवाय|के\s+अलावा|अपवाद|को\s+छोड़कर|भिन्न|रहित\s+होकर"

_QUANTIFIERS_EN: list[tuple[str, str, int]] = [
    (r"\b(?:all|every|each|any|whatsoever|whosoever|entire|whole)\b", "all", QUANT_UNIVERSAL),
    (r"\b(?:some|certain|a\s+few|several|various|part\s+of)\b", "some", QUANT_EXISTENTIAL),
    (r"\b(?:none|no|nothing|nobody)\b", "none", 0),
]
_QUANTIFIERS_HI: list[tuple[str, str, int]] = [
    # हर is anchored: बाहर ("outside") ends in the same two letters and was
    # being counted as a universal quantifier.  सब and भाग are deliberately
    # left unanchored — सबके and भागों are inflections of the same lexeme
    # rather than different words, and anchoring would drop them.
    (rf"सभी|सब|प्रत्येक|{_deva_word('हर')}|कोई\s+भी|समस्त|संपूर्ण|पूरा", "सभी", QUANT_UNIVERSAL),
    (r"कुछ|कतिपय|कई|आंशिक|भाग", "कुछ", QUANT_EXISTENTIAL),
    (r"कोई\s+नहीं|कुछ\s+भी\s+नहीं|न\s+कोई", "कोई नहीं", 0),
]

# मात्र ("merely") is anchored because मात्रा ("amount") and मात्रात्मक
# ("quantitative") are different words that happen to start with it; treating
# them as a scope marker asserts a restriction the sentence does not make.
_SCOPE_EN = r"\bonly\b|\bsolely\b|\bexclusively\b|\bmerely\b|\bsole\b|\bexclusive\b"
_SCOPE_HI = rf"केवल|{_deva_word('मात्र')}|सिर्फ|अकेले|अनन्य|विशेष\s+रूप\s+से"

# Curated legal role nouns.  Generic NER is a poor fit here: in statutory
# text the "entities" that matter are roles (tenant, employer, insurer), not
# named organisations.
_ENTITIES_EN = [
    "tenant", "landlord", "lessee", "lessor", "licensor", "licensee",
    "employer", "employee", "workman", "buyer", "seller", "purchaser",
    "vendor", "company", "contractor", "sub-contractor", "agent", "principal",
    "insurer", "insured", "insurer's", "consumer", "service provider",
    "bailee", "bailor", "guarantor", "surety", "minor", "promisor",
    "promisee", "party", "parties", "assignee", "assignor", "mortgagor",
    "mortgagee", "partner", "firm", "shareholder", "director", "officer",
]
_ENTITIES_HI = [
    "किरायेदार", "मकान मालिक", "पट्टेदार", "नियोक्ता", "कर्मचारी", "श्रमिक",
    "क्रेता", "विक्रेता", "खरीदार", "कंपनी", "ठेकेदार", "एजेंट", "प्रधान",
    "बीमाकर्ता", "बीमित", "उपभोक्ता", "सेवा प्रदाता", "उपनिहिती", "निहिती",
    "प्रतिभू", "जमानतदार", "नाबालिग", "वचनदाता", "वचनग्रहीता", "पक्षकार",
    "पक्ष", "हस्तांतरिती", "भागीदार", "फर्म", "अंशधारक", "निदेशक",
]

# --- quantities -------------------------------------------------------------
# Number words are split into two tables because they compose rather than add:
# "five hundred" is 5 x 100, not 5 + 100.  Keeping the multipliers separate is
# what lets _compose_number_words distinguish "five hundred" (500) from
# "twenty five" (25).
#
# Indian legal prose uses the lakh/crore scale, and the corpus does too: ten
# rows carry "lakh" and one carries "crore".  A vocabulary that stopped at
# "thousand" silently dropped every one of those penalties.

_EN_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "a": 1, "an": 1,
    # "one hundred and fifty" is ordinary legal drafting; a zero-valued
    # connector lets the run continue instead of splitting into 100 and 50.
    "and": 0,
}

#: Words that multiply the value accumulated so far.  Those >= 1000 also
#: close off a sub-total, which is how "two thousand five hundred" becomes
#: 2 x 1000 + 5 x 100 rather than 2 x 1000 x 5 x 100.
_EN_MULTIPLIERS: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000,
    "million": 1_000_000, "millions": 1_000_000,
    "billion": 1_000_000_000,
}

_HI_NUMBER_WORDS: dict[str, int] = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11,
    "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "तीस": 30,
    "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80,
    "नब्बे": 90,
}

_HI_MULTIPLIERS: dict[str, int] = {
    "सौ": 100,
    "हज़ार": 1_000, "हजार": 1_000,
    "लाख": 100_000,
    "करोड़": 10_000_000, "करोड": 10_000_000,
}

# unit -> (canonical unit, multiplier to canonical base)
_EN_UNITS: dict[str, tuple[str, float]] = {
    "day": ("day", 1), "days": ("day", 1),
    "week": ("day", 7), "weeks": ("day", 7), "fortnight": ("day", 14),
    "month": ("day", 30), "months": ("day", 30),
    "year": ("day", 365), "years": ("day", 365),
    "hour": ("hour", 1), "hours": ("hour", 1),
    "minute": ("minute", 1), "minutes": ("minute", 1),
    "rupee": ("currency", 1), "rupees": ("currency", 1), "rs": ("currency", 1),
    "percent": ("percent", 1), "per cent": ("percent", 1),
}
_HI_UNITS: dict[str, tuple[str, float]] = {
    "दिन": ("day", 1), "दिवस": ("day", 1),
    "सप्ताह": ("day", 7), "हफ्ते": ("day", 7), "हफ्ता": ("day", 7),
    "पखवाड़ा": ("day", 14), "पखवाड़े": ("day", 14),
    "माह": ("day", 30), "महीना": ("day", 30), "महीने": ("day", 30),
    "वर्ष": ("day", 365), "साल": ("day", 365),
    "घंटा": ("hour", 1), "घंटे": ("hour", 1),
    "मिनट": ("minute", 1),
    "रुपये": ("currency", 1), "रुपया": ("currency", 1), "रु": ("currency", 1),
    "प्रतिशत": ("percent", 1),
}

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LegalCue:
    """A single legally salient span found in a sentence."""

    category: str
    canonical: str
    surface: str
    start: int
    end: int
    strength: int | None = None

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "canonical": self.canonical,
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
            "strength": self.strength,
        }


@dataclass(frozen=True)
class Quantity:
    """A normalised numeric or temporal quantity."""

    value: float
    raw: str
    canonical_unit: str
    normalized: float  # value expressed in the canonical unit's base
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "raw": self.raw,
            "canonical_unit": self.canonical_unit,
            "normalized": self.normalized,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class CueProfile:
    """All cues extracted from one sentence."""

    text: str
    language: str
    cues: list[LegalCue] = field(default_factory=list)
    quantities: list[Quantity] = field(default_factory=list)

    # -- convenience accessors -------------------------------------------
    def by_category(self, category: str) -> list[LegalCue]:
        return [c for c in self.cues if c.category == category]

    @property
    def modals(self) -> list[LegalCue]:
        return self.by_category(CATEGORY_MODAL)

    @property
    def negations(self) -> list[LegalCue]:
        return self.by_category(CATEGORY_NEGATION)

    @property
    def conditions(self) -> list[LegalCue]:
        return self.by_category(CATEGORY_CONDITION)

    @property
    def exceptions(self) -> list[LegalCue]:
        return self.by_category(CATEGORY_EXCEPTION)

    @property
    def entities(self) -> list[LegalCue]:
        return self.by_category(CATEGORY_ENTITY)

    def max_modal_strength(self) -> int | None:
        """Strongest deontic modal present, or ``None`` if there is none."""
        strengths = [c.strength for c in self.modals if c.strength is not None]
        return max(strengths) if strengths else None

    def canonical_multiset(self, category: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cue in self.by_category(category):
            counts[cue.canonical] = counts.get(cue.canonical, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "cues": [c.to_dict() for c in self.cues],
            "quantities": [q.to_dict() for q in self.quantities],
            "max_modal_strength": self.max_modal_strength(),
        }


@dataclass
class CueDelta:
    """Difference between the original and simplified cue profiles.

    Each field answers a specific legal question:
      * ``dropped_*``    — information present in the original but absent from
                           the simplification (candidate omission).
      * ``added_*``      — information in the simplification with no basis in
                           the original (candidate hallucination).
      * ``modal_*``      — deontic strength movement on the
                           prohibition/obligation/permission axis.
      * ``quantity_*``   — numeric or temporal changes.
    """

    dropped_conditions: list[str] = field(default_factory=list)
    added_conditions: list[str] = field(default_factory=list)
    dropped_exceptions: list[str] = field(default_factory=list)
    added_exceptions: list[str] = field(default_factory=list)
    dropped_entities: list[str] = field(default_factory=list)
    added_entities: list[str] = field(default_factory=list)
    quantity_changes: list[dict] = field(default_factory=list)
    original_modal_strength: int | None = None
    simplified_modal_strength: int | None = None
    negation_count_original: int = 0
    negation_count_simplified: int = 0
    original_quantifier: str | None = None
    simplified_quantifier: str | None = None

    @property
    def modal_weakened(self) -> bool:
        o, s = self.original_modal_strength, self.simplified_modal_strength
        return o is not None and s is not None and s < o

    @property
    def modal_strengthened(self) -> bool:
        o, s = self.original_modal_strength, self.simplified_modal_strength
        return o is not None and s is not None and s > o

    @property
    def negation_flipped(self) -> bool:
        """True when exactly one side is negated, or parity differs."""
        return (self.negation_count_original > 0) != (self.negation_count_simplified > 0)

    @property
    def has_any_change(self) -> bool:
        return any(
            [
                self.dropped_conditions, self.added_conditions,
                self.dropped_exceptions, self.added_exceptions,
                self.dropped_entities, self.added_entities,
                self.quantity_changes,
                self.modal_weakened, self.modal_strengthened,
                self.negation_flipped,
            ]
        )

    def to_dict(self) -> dict:
        return {
            "dropped_conditions": self.dropped_conditions,
            "added_conditions": self.added_conditions,
            "dropped_exceptions": self.dropped_exceptions,
            "added_exceptions": self.added_exceptions,
            "dropped_entities": self.dropped_entities,
            "added_entities": self.added_entities,
            "quantity_changes": self.quantity_changes,
            "original_modal_strength": self.original_modal_strength,
            "simplified_modal_strength": self.simplified_modal_strength,
            "original_modal_label": STRENGTH_LABELS.get(self.original_modal_strength or -1),
            "simplified_modal_label": STRENGTH_LABELS.get(self.simplified_modal_strength or -1),
            "modal_weakened": self.modal_weakened,
            "modal_strengthened": self.modal_strengthened,
            "negation_flipped": self.negation_flipped,
            "original_quantifier": self.original_quantifier,
            "simplified_quantifier": self.simplified_quantifier,
        }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def normalize_for_matching(text: str) -> str:
    """NFKC-normalise and collapse whitespace, **preserving offsets loosely**.

    NFKC can change string length (Devanagari combining marks), so character
    offsets from the normalised string are only approximately valid on the
    original.  We therefore run cue matching on the *original* text and use
    this only for comparison keys, never for offsets.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _ascii_digits(text: str) -> str:
    """Convert Devanagari digits to ASCII so quantities compare numerically."""
    return text.translate(_DEVA_DIGITS)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _find_pattern_cues(
    text: str,
    patterns: Iterable[tuple[str, str, int]],
    category: str,
) -> list[LegalCue]:
    """Match a (pattern, canonical, strength) table against ``text``.

    Overlapping matches are resolved by keeping the earliest-starting,
    longest match, which is why prohibition patterns are listed before the
    general obligation patterns they contain.
    """
    found: list[LegalCue] = []
    for pattern, canonical, strength in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append(
                LegalCue(
                    category=category,
                    canonical=canonical,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    strength=strength,
                )
            )
    return _resolve_overlaps(found)


def _find_simple_cues(text: str, pattern: str, category: str, canonical: str) -> list[LegalCue]:
    return [
        LegalCue(
            category=category,
            canonical=canonical,
            surface=m.group(0),
            start=m.start(),
            end=m.end(),
        )
        for m in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def _resolve_overlaps(cues: list[LegalCue]) -> list[LegalCue]:
    """Greedily keep non-overlapping cues, preferring longer spans."""
    if not cues:
        return []
    ordered = sorted(cues, key=lambda c: (c.start, -(c.end - c.start)))
    kept: list[LegalCue] = []
    for cue in ordered:
        if any(cue.start < k.end and k.start < cue.end for k in kept):
            continue
        kept.append(cue)
    return sorted(kept, key=lambda c: c.start)


def _compose_number_words(
    words: Sequence[str],
    additive: dict[str, int],
    multipliers: dict[str, int],
) -> tuple[float | None, bool]:
    """Evaluate a run of number words, returning ``(value, saw_multiplier)``.

    Multipliers apply to the value accumulated since the last large multiplier,
    which is what distinguishes the two readings a naive sum cannot tell apart::

        two thousand five hundred   ->  2 x 1000  +  5 x 100   = 2500
        twenty five                 ->  20        +  5         = 25

    ``saw_multiplier`` is returned because it decides whether the phrase is
    interpretable without a trailing unit.  "one lakh" is unambiguously a
    quantity even with no unit word after it, whereas a bare "one" is a
    pronoun as often as it is a number ("one of the parties").
    """
    total = 0.0
    current = 0.0
    saw_multiplier = False
    saw_any = False

    for word in words:
        if word in multipliers:
            factor = multipliers[word]
            current = (current or 1.0) * factor
            saw_multiplier = True
            saw_any = True
            if factor >= 1000:
                # A thousand-or-larger multiplier closes the sub-total, so a
                # following small number starts a new one.
                total += current
                current = 0.0
        elif word in additive:
            current += additive[word]
            saw_any = True
        else:
            return None, False

    if not saw_any:
        return None, False
    return total + current, saw_multiplier


def extract_quantities(text: str, language: str) -> list[Quantity]:
    """Extract numeric and temporal quantities, normalised to canonical units.

    Handles Arabic and Devanagari digits and the number-word forms that
    statutes use in prose (``within seven days``, ``not exceeding five lakh
    rupees``, ``सात दिन के भीतर``).

    A bare number word needs either a trailing unit or a multiplier to count.
    Without that rule ``one of the parties`` becomes a quantity of one, which
    is the single largest source of false positives on this corpus.
    """
    if not isinstance(text, str) or not text:
        return []

    additive = _EN_NUMBER_WORDS if language == "en" else _HI_NUMBER_WORDS
    multipliers = _EN_MULTIPLIERS if language == "en" else _HI_MULTIPLIERS
    units = _EN_UNITS if language == "en" else _HI_UNITS

    vocabulary = sorted(set(additive) | set(multipliers), key=len, reverse=True)
    word_alt = "|".join(re.escape(w) for w in vocabulary)
    unit_alt = "|".join(sorted((re.escape(u) for u in units), key=len, reverse=True))

    results: list[Quantity] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < e and s < end for s, e in consumed)

    # Pass 1: percentages written with a symbol ("5%").  This must run before
    # the general pass, not after: the general pattern matches the bare "5" on
    # its own, and if it goes first it claims that span as a unitless count and
    # the percent branch below never sees it.  The original code had the two
    # passes in the other order, which made the percent branch dead code and
    # labelled every percentage as a plain count.
    for match in re.finditer(r"(?P<num>[0-9]+(?:\.[0-9]+)?)\s*%", text):
        results.append(
            Quantity(
                value=float(match.group("num")),
                raw=match.group(0),
                canonical_unit="percent",
                normalized=float(match.group("num")),
                start=match.start(),
                end=match.end(),
            )
        )
        consumed.append((match.start(), match.end()))

    # Pass 2: digits or a run of number words, with an optional trailing unit.
    #
    # The boundary assertions are not decoration.  Without them ``an`` matches
    # inside "th[an] forty-five days" and the run composes 1 + 45 = 46; ``a``
    # matches inside "dat[a] days" and reports a one-day deadline that does not
    # exist.  ``\b`` cannot be used because it is defined in terms of ``\w``,
    # and Devanagari matras are combining marks that ``\w`` does not match, so
    # the boundary class is spelled out — ``\w`` plus the whole Devanagari
    # block, which includes the matras and the virama.
    word_boundary = r"[\wऀ-ॿ]"
    pattern = (
        rf"(?<!{word_boundary})"
        rf"(?P<num>[0-9]+(?:[.,][0-9]+)*"
        rf"|(?:{word_alt})(?:[\s-]+(?:{word_alt}))*)"
        rf"(?:\s+(?P<unit>{unit_alt}))?"
        rf"(?!{word_boundary})"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        if overlaps(match.start(), match.end()):
            continue

        raw = match.group(0)
        num_text = _ascii_digits(match.group("num")).replace(",", "")
        unit_text = match.group("unit")
        unit_key = unit_text.lower() if unit_text else None

        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", num_text):
            value = float(num_text)
        else:
            words = re.split(r"[\s-]+", match.group("num").lower())
            value, saw_multiplier = _compose_number_words(words, additive, multipliers)
            if value is None:
                continue
            if unit_key is None and not saw_multiplier:
                # Bare small number word with neither unit nor multiplier: too
                # ambiguous to be a legal quantity ("one of the parties").
                continue

        if unit_key and unit_key in units:
            canonical_unit, multiplier = units[unit_key]
        else:
            canonical_unit, multiplier = "count", 1.0

        results.append(
            Quantity(
                value=value,
                raw=raw,
                canonical_unit=canonical_unit,
                normalized=value * multiplier,
                start=match.start(),
                end=match.end(),
            )
        )
        consumed.append((match.start(), match.end()))

    return sorted(results, key=lambda q: q.start)


def extract_entities(text: str, language: str) -> list[LegalCue]:
    """Find curated legal role nouns (tenant, employer, insurer, ...)."""
    lexicon = _ENTITIES_EN if language == "en" else _ENTITIES_HI
    cues: list[LegalCue] = []
    lowered = text.lower()
    for term in lexicon:
        needle = term.lower()
        start = 0
        while True:
            idx = lowered.find(needle, start)
            if idx == -1:
                break
            # Require a word boundary so "party" does not match "partly".
            before_ok = idx == 0 or not lowered[idx - 1].isalnum()
            after = idx + len(needle)
            after_ok = after >= len(lowered) or not lowered[after].isalnum()
            if before_ok and after_ok:
                cues.append(
                    LegalCue(
                        category=CATEGORY_ENTITY,
                        canonical=term,
                        surface=text[idx:after],
                        start=idx,
                        end=after,
                    )
                )
            start = idx + len(needle)
    return _resolve_overlaps(cues)


def extract_cues(text: str, language: str = "en") -> CueProfile:
    """Extract every legally salient cue from ``text``.

    Parameters
    ----------
    text
        Raw sentence.  Not normalised first — offsets are reported against
        this exact string.
    language
        ``"en"`` or ``"hi"``.  Unknown codes fall back to English lexicons
        but keep the supplied code in the profile for traceability.
    """
    if not isinstance(text, str) or not text.strip():
        return CueProfile(text=text or "", language=language)

    is_hindi = language == "hi"
    modal_table = _MODALS_HI if is_hindi else _MODALS_EN
    negation_pattern = _NEGATIONS_HI if is_hindi else _NEGATIONS_EN
    condition_pattern = _CONDITIONS_HI if is_hindi else _CONDITIONS_EN
    exception_pattern = _EXCEPTIONS_HI if is_hindi else _EXCEPTIONS_EN
    quantifier_table = _QUANTIFIERS_HI if is_hindi else _QUANTIFIERS_EN
    scope_pattern = _SCOPE_HI if is_hindi else _SCOPE_EN

    cues: list[LegalCue] = []
    cues += _find_pattern_cues(text, modal_table, CATEGORY_MODAL)
    cues += _find_simple_cues(text, negation_pattern, CATEGORY_NEGATION, "negation")
    cues += _find_simple_cues(text, condition_pattern, CATEGORY_CONDITION, "condition")
    cues += _find_simple_cues(text, exception_pattern, CATEGORY_EXCEPTION, "exception")
    cues += _find_pattern_cues(text, quantifier_table, CATEGORY_QUANTIFIER)
    cues += _find_simple_cues(text, scope_pattern, CATEGORY_SCOPE, "only")
    cues += extract_entities(text, language)

    return CueProfile(
        text=text,
        language=language,
        cues=sorted(cues, key=lambda c: c.start),
        quantities=extract_quantities(text, language),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_profiles(original: CueProfile, simplified: CueProfile) -> CueDelta:
    """Compute the legally meaningful differences between two profiles.

    Multiset comparison (not set comparison) is used for entities, so that a
    simplification which drops one of two references to "the tenant" is still
    visible as a change.
    """
    delta = CueDelta()

    def multiset(cues: Sequence[LegalCue]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cue in cues:
            counts[cue.canonical] = counts.get(cue.canonical, 0) + 1
        return counts

    def diff(orig: dict[str, int], simp: dict[str, int]) -> tuple[list[str], list[str]]:
        dropped = [k for k in orig if orig[k] > simp.get(k, 0)]
        added = [k for k in simp if simp[k] > orig.get(k, 0)]
        return sorted(dropped), sorted(added)

    delta.dropped_conditions, delta.added_conditions = diff(
        multiset(original.conditions), multiset(simplified.conditions)
    )
    delta.dropped_exceptions, delta.added_exceptions = diff(
        multiset(original.exceptions), multiset(simplified.exceptions)
    )
    delta.dropped_entities, delta.added_entities = diff(
        multiset(original.entities), multiset(simplified.entities)
    )

    delta.original_modal_strength = original.max_modal_strength()
    delta.simplified_modal_strength = simplified.max_modal_strength()
    delta.negation_count_original = len(original.negations)
    delta.negation_count_simplified = len(simplified.negations)

    oq = original.by_category(CATEGORY_QUANTIFIER)
    sq = simplified.by_category(CATEGORY_QUANTIFIER)
    delta.original_quantifier = oq[0].canonical if oq else None
    delta.simplified_quantifier = sq[0].canonical if sq else None

    # Quantity comparison: match by canonical unit and compare normalised
    # values.  "seven days" vs "one week" both normalise to 7 days and are
    # therefore correctly reported as unchanged.
    orig_by_unit: dict[str, list[Quantity]] = {}
    simp_by_unit: dict[str, list[Quantity]] = {}
    for q in original.quantities:
        orig_by_unit.setdefault(q.canonical_unit, []).append(q)
    for q in simplified.quantities:
        simp_by_unit.setdefault(q.canonical_unit, []).append(q)

    for unit in sorted(set(orig_by_unit) | set(simp_by_unit)):
        o_values = sorted(q.normalized for q in orig_by_unit.get(unit, []))
        s_values = sorted(q.normalized for q in simp_by_unit.get(unit, []))
        if o_values != s_values:
            delta.quantity_changes.append(
                {
                    "unit": unit,
                    "original_values": o_values,
                    "simplified_values": s_values,
                    "original_raw": [q.raw for q in orig_by_unit.get(unit, [])],
                    "simplified_raw": [q.raw for q in simp_by_unit.get(unit, [])],
                }
            )

    return delta


def compare_texts(original: str, simplified: str, language: str = "en") -> CueDelta:
    """Convenience wrapper: extract from both texts, then compare."""
    return compare_profiles(
        extract_cues(original, language), extract_cues(simplified, language)
    )


def change_summary(delta: CueDelta) -> list[str]:
    """Render a :class:`CueDelta` as short human-readable phrases.

    These phrases seed the explanation layer.  They state *what* differs; the
    explanation layer turns them into a sentence about why it matters
    legally.  Keeping the two separate stops the rule layer from overclaiming.
    """
    notes: list[str] = []

    if delta.modal_weakened:
        notes.append(
            f"deontic strength reduced from "
            f"{STRENGTH_LABELS.get(delta.original_modal_strength, '?')} to "
            f"{STRENGTH_LABELS.get(delta.simplified_modal_strength, '?')}"
        )
    elif delta.modal_strengthened:
        notes.append(
            f"deontic strength increased from "
            f"{STRENGTH_LABELS.get(delta.original_modal_strength, '?')} to "
            f"{STRENGTH_LABELS.get(delta.simplified_modal_strength, '?')}"
        )

    if delta.negation_flipped:
        notes.append("negation present on only one side")

    if delta.dropped_conditions:
        notes.append("condition dropped: " + ", ".join(delta.dropped_conditions))
    if delta.added_conditions:
        notes.append("condition added: " + ", ".join(delta.added_conditions))
    if delta.dropped_exceptions:
        notes.append("exception dropped: " + ", ".join(delta.dropped_exceptions))
    if delta.added_exceptions:
        notes.append("exception added: " + ", ".join(delta.added_exceptions))
    if delta.dropped_entities:
        notes.append("entity dropped: " + ", ".join(delta.dropped_entities))
    if delta.added_entities:
        notes.append("entity added: " + ", ".join(delta.added_entities))

    for change in delta.quantity_changes:
        notes.append(
            f"{change['unit']} value changed from "
            f"{change['original_raw'] or change['original_values']} to "
            f"{change['simplified_raw'] or change['simplified_values']}"
        )

    oq, sq = delta.original_quantifier, delta.simplified_quantifier
    if oq and sq and oq != sq:
        notes.append(f"quantifier narrowed from '{oq}' to '{sq}'")

    return notes
