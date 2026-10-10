"""Tests for the language-aware, legally-conservative preprocessor.

The module's whole purpose is to *not* destroy legal signal, so most of these
tests assert that things are preserved. The two regression tests at the bottom
pin bugs that were found in this code and must not come back.
"""

from __future__ import annotations

import pytest

from src.data import preprocess as pp


# ---------------------------------------------------------------------------
# Normalisation: defaults must be non-destructive
# ---------------------------------------------------------------------------
def test_default_config_changes_nothing_but_whitespace():
    text = "  The buyer  shall not  return 30 goods.  "
    assert pp.normalize_text(text) == "The buyer shall not return 30 goods."


def test_normalize_text_does_not_lowercase_by_default():
    assert pp.normalize_text("Shall") == "Shall"


def test_normalize_text_keeps_punctuation_and_digits_by_default():
    out = pp.normalize_text("S. 45(2): 30 days, 5% penalty.")
    for fragment in ("S.", "45(2):", "30", "5%"):
        assert fragment in out


def test_normalize_text_returns_empty_string_for_non_strings():
    for value in (None, 42, float("nan"), ["not", "text"]):
        assert pp.normalize_text(value) == ""


def test_optional_lowercase_applies_only_when_requested():
    assert pp.normalize_text("Shall", {"lowercase": True}) == "shall"


def test_strip_accents_would_destroy_devanagari_so_is_off_by_default():
    """The flag exists, but enabling it is documented as catastrophic for Hindi.

    This test records why the default is False rather than asserting the
    destructive behaviour is desirable.
    """
    hindi = "क्रेता"
    assert pp.normalize_text(hindi) == hindi
    assert pp.normalize_text(hindi, {"strip_accents": True}) != hindi


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def test_tokenize_english_words_and_punctuation():
    assert pp.tokenize("The buyer shall not return the goods.") == [
        "The", "buyer", "shall", "not", "return", "the", "goods", ".",
    ]


def test_tokenize_keeps_hyphenated_and_apostrophised_words_together():
    assert pp.tokenize("Don't re-sell the goods.") == [
        "Don't", "re-sell", "the", "goods", ".",
    ]


def test_tokenize_keeps_digits_as_tokens():
    assert pp.tokenize("within 30 days") == ["within", "30", "days"]


def test_tokenize_empty_and_non_string_input():
    assert pp.tokenize("") == []
    assert pp.tokenize(None) == []


def test_content_tokens_drops_punctuation_including_danda():
    assert pp.content_tokens("क्रेता माल लौटाएगा।") == ["क्रेता", "माल", "लौटाएगा"]
    assert pp.content_tokens("goods, please.") == ["goods", "please"]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("One. Two. Three.", ["One.", "Two.", "Three."]),
        ("पहला। दूसरा।", ["पहला।", "दूसरा।"]),
        ("पहला। दूसरा॥ तीसरा?", ["पहला।", "दूसरा॥", "तीसरा?"]),
        ("Line one\nLine two", ["Line one", "Line two"]),
    ],
)
def test_sentence_split_handles_english_and_devanagari_terminators(text, expected):
    assert pp.sentence_split(text) == expected


def test_sentence_split_ignores_blank_input():
    assert pp.sentence_split("   ") == []
    assert pp.sentence_split(None) == []


# ---------------------------------------------------------------------------
# Similarity primitives
# ---------------------------------------------------------------------------
def test_token_jaccard_bounds_and_identity():
    assert pp.token_jaccard("a b c", "a b c") == 1.0
    assert pp.token_jaccard("a b", "x y") == 0.0
    assert pp.token_jaccard("a b", "b c") == pytest.approx(1 / 3)


def test_token_jaccard_is_empty_safe():
    assert pp.token_jaccard("", "") == 1.0
    assert pp.token_jaccard("a", "") == 0.0


def test_token_jaccard_is_case_insensitive():
    assert pp.token_jaccard("The Buyer", "the buyer") == 1.0


def test_length_ratio_counts_words_not_characters():
    assert pp.length_ratio("one two three four", "one two") == 0.5
    # Hindi must be measured on the same unit as English, or the omission
    # heuristic silently changes meaning between the two halves of the corpus.
    assert pp.length_ratio("एक दो तीन चार", "एक दो") == 0.5


def test_length_ratio_handles_empty_original():
    assert pp.length_ratio("", "") == 1.0
    assert pp.length_ratio("", "some text") == float("inf")


def test_char_ngrams_are_normalised_and_lowercased():
    assert pp.char_ngrams("ABCD", n=2) == ["ab", "bc", "cd"]
    # Whitespace is collapsed before n-gramming, so a padded input does not
    # produce n-grams full of spaces.
    assert pp.char_ngrams("  AB  cd ", n=3) == ["ab ", "b c", " cd"]


def test_char_ngrams_returns_the_short_string_whole():
    """Shorter than n: the whole string is returned rather than nothing.

    Returning an empty list here would make an empty vocabulary
    indistinguishable from a very short one.
    """
    assert pp.char_ngrams("AB", n=3) == ["ab"]
    assert pp.char_ngrams("", n=3) == []


# ---------------------------------------------------------------------------
# Frame-level helpers
# ---------------------------------------------------------------------------
def test_add_clean_columns_leaves_raw_columns_untouched(master_frame):
    out = pp.add_clean_columns(master_frame, ("original_text",))
    assert "original_text_clean" in out.columns
    assert out["original_text"].tolist() == master_frame["original_text"].tolist()


def test_add_clean_columns_warns_and_skips_absent_column():
    import pandas as pd

    out = pp.add_clean_columns(pd.DataFrame({"other": ["x"]}), ("original_text",))
    assert "original_text_clean" not in out.columns


def test_preprocess_pair_output_shape_is_a_contract():
    """``preprocess_pair`` is called at inference time, so its keys are an API."""
    out = pp.preprocess_pair("The buyer shall not return.", "The buyer returns.", "en")
    assert set(out) == {
        "original_raw", "simplified_raw", "original_clean", "simplified_clean",
        "original_tokens", "simplified_tokens", "token_jaccard",
        "length_ratio", "language",
    }
    assert out["original_raw"] == "The buyer shall not return."
    assert out["language"] == "en"


def test_preprocess_pair_reports_low_overlap_for_a_meaning_changing_rewrite():
    original = "The buyer shall not return the goods within 30 days."
    simplified = "The buyer may return the goods."
    out = pp.preprocess_pair(original, simplified, "en")
    assert out["token_jaccard"] < 0.6
    assert out["length_ratio"] < 1.0


# ---------------------------------------------------------------------------
# The legal-content guard
# ---------------------------------------------------------------------------
def test_guard_is_silent_when_nothing_was_destroyed():
    text = "The buyer shall not return the goods within 30 days."
    assert pp.assert_legal_content_preserved(text, text) == []


def test_guard_reports_a_dropped_modal_and_negation():
    original = "The buyer shall not return the goods."
    processed = "The buyer return the goods."
    warnings = pp.assert_legal_content_preserved(original, processed)
    assert any("modals" in w for w in warnings)
    assert any("negations" in w for w in warnings)


def test_guard_reports_a_dropped_quantity():
    original = "within 30 days"
    processed = "within days"
    assert any("quantities" in w for w in pp.assert_legal_content_preserved(original, processed))


def test_guard_reports_a_protected_term_removed_by_the_config_vocabulary():
    """``preprocessing.protected_patterns`` must be enforced, not decorative."""
    cfg = {"preprocessing": {"protected_patterns": {"modals": ["shall"]}}}
    warnings = pp.assert_legal_content_preserved("It shall be done.", "It will be done.", cfg=cfg)
    assert any("shall" in w for w in warnings)


def test_guard_is_case_insensitive_for_protected_terms():
    cfg = {"preprocessing": {"protected_patterns": {"modals": ["shall"]}}}
    assert pp.assert_legal_content_preserved("It Shall be done.", "It shall be done.", cfg=cfg) == []


def test_guard_does_not_flag_a_protected_term_absent_from_the_original():
    """The check is 'was it there and is it gone', not 'is it in the end text'."""
    cfg = {"preprocessing": {"protected_patterns": {"modals": ["shall"]}}}
    assert pp.assert_legal_content_preserved("It must be done.", "It must be done.", cfg=cfg) == []


def test_contains_whole_token_does_not_match_inside_a_longer_word():
    assert not pp._contains_whole_token("partly", "party")
    assert pp._contains_whole_token("the party", "party")


# ===========================================================================
# REGRESSION TESTS
# ===========================================================================
def test_regression_devanagari_matras_stay_inside_the_word():
    """Devanagari words must not be shattered into characters.

    The bug: ``\\w+`` matches only characters with ``str.isalnum() == True``.
    Devanagari matras and the virama are Unicode category Mn/Mc, for which
    ``isalnum()`` is **False**, so a naive ``\\w+`` pattern split ``क्रेता``
    into ``क`` ``्`` ``र`` ``े`` ``त`` ``ा``.  Every word in the Hindi half of
    the corpus was affected, and ``token_jaccard`` and ``length_ratio`` were
    silently computed over characters for Hindi while being computed over
    words for English.
    """
    assert pp.tokenize("क्रेता माल नहीं लौटाएगा।", "hi") == [
        "क्रेता", "माल", "नहीं", "लौटाएगा", "।",
    ]


def test_regression_danda_is_still_its_own_punctuation_token():
    """The matra fix must not swallow the sentence terminators.

    U+0964 danda and U+0965 double danda sit in the Devanagari block, so a
    careless word-character range would absorb them into the preceding word.
    """
    assert pp.tokenize("पहला। दूसरा॥", "hi") == ["पहला", "।", "दूसरा", "॥"]


def test_regression_hindi_and_english_agree_on_the_unit_of_measurement():
    """The cross-lingual inconsistency the bug caused, stated directly.

    A four-word Hindi string shortened to two words is a length ratio of 0.5,
    exactly as it is in English.  Under the bug the Hindi ratio was computed
    over characters and bore no relation to the English one — in precisely the
    comparison Experiment 2 is designed to make.
    """
    assert pp.length_ratio("एक दो तीन चार", "एक दो") == pp.length_ratio(
        "one two three four", "one two"
    )
    assert pp.token_jaccard("क्रेता माल", "क्रेता माल") == pp.token_jaccard(
        "buyer goods", "buyer goods"
    )


def test_regression_word_boundary_check_survives_sentence_final_devanagari():
    """``\\b`` cannot detect a boundary after a matra, so the guard hand-rolls one.

    ``\\bनहीं\\b`` does not match a sentence-final ``नहीं`` because the character
    before the boundary is a combining mark, which ``\\w`` does not match.  If
    the protected-term check had used ``\\b`` it would have silently skipped
    every sentence-final Hindi protected term — the commonest position in a
    statute.
    """
    warnings = pp.assert_legal_content_preserved("वह माल नहीं लौटाएगा।", "वह माल लौटाएगा।")
    assert any("नहीं" in w for w in warnings)


def test_regression_fallback_vocabulary_applies_without_a_config():
    """The guard must work when called with no config at all.

    ``assert_legal_content_preserved`` is called from tests and from the
    inference path, neither of which necessarily has a config in hand.  If the
    protected list were empty without one, the check would silently pass on
    everything.
    """
    assert pp._protected_tokens(None) >= pp._FALLBACK_PROTECTED
    assert pp._FALLBACK_PROTECTED  # non-empty
    warnings = pp.assert_legal_content_preserved(
        "The tenant shall not sublet.", "The tenant sublet."
    )
    assert warnings
