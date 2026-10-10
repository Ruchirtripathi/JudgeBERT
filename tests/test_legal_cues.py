"""Tests for legally-salient cue extraction.

This module is the evidentiary basis for the explainability layer and for the
rule-assisted error detector. Its failure mode is asymmetric: a missed cue
costs recall, but a *spurious* cue makes the system report a legal defect that
is not there. The tests below therefore care more about precision than recall.
"""

from __future__ import annotations

import pytest

from src.data import legal_cues as lc


# ---------------------------------------------------------------------------
# Modals and deontic strength
# ---------------------------------------------------------------------------
def test_shall_is_obligation_and_may_is_permission():
    assert lc.extract_cues("The buyer shall pay.", "en").max_modal_strength() == lc.STRENGTH_OBLIGATION
    assert lc.extract_cues("The buyer may pay.", "en").max_modal_strength() == lc.STRENGTH_PERMISSION


def test_shall_not_outranks_shall_as_a_prohibition():
    """'shall not' contains 'shall'; the prohibition must win.

    The lexicon orders specific patterns before general ones and resolves
    overlaps by preferring the longer span. If that ordering broke, a
    prohibition would be reported as a mere obligation — a legal error
    introduced by the extractor itself.
    """
    cues = lc.extract_cues("The buyer shall not return the goods.", "en")
    assert cues.max_modal_strength() == lc.STRENGTH_PROHIBITION
    assert [c.canonical for c in cues.modals] == ["shall not"]


def test_strength_scale_is_ordered_as_documented():
    assert (
        lc.STRENGTH_PROHIBITION
        > lc.STRENGTH_OBLIGATION
        > lc.STRENGTH_RECOMMENDATION
        > lc.STRENGTH_PERMISSION
        > lc.STRENGTH_DISCRETION
    )


def test_a_sentence_with_no_modal_reports_none_not_zero():
    """``None`` means 'no modal', ``0`` means Discretion. They are different."""
    assert lc.extract_cues("The goods are red.", "en").max_modal_strength() is None


def test_hindi_modal_suffixes_are_recognised():
    permission = lc.extract_cues("क्रेता माल लौटा सकता है।", "hi")
    assert permission.max_modal_strength() == lc.STRENGTH_PERMISSION

    obligation = lc.extract_cues("क्रेता को भुगतान करना होगा।", "hi")
    assert obligation.max_modal_strength() == lc.STRENGTH_OBLIGATION


# ---------------------------------------------------------------------------
# Negation, conditions, exceptions, quantifiers, scope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "The buyer shall not return.",
        "No person may enter.",
        "The buyer never returned.",
        "Neither party is liable.",
        "The buyer cannot assign.",
        "A contract without consideration is void.",
    ],
)
def test_english_negations_are_detected(text):
    assert lc.extract_cues(text, "en").negations


@pytest.mark.parametrize(
    "text",
    [
        "यदि क्रेता भुगतान नहीं करता है।",
        "अगर माल खराब है।",
        "बशर्ते कि अनुबंध वैध है।",
    ],
)
def test_hindi_conditions_are_detected(text):
    assert lc.extract_cues(text, "hi").conditions


@pytest.mark.parametrize("text", ["except", "notwithstanding", "other than", "unless otherwise"])
def test_english_exception_markers_are_detected(text):
    assert lc.extract_cues(f"The rule applies, {text} in this case.", "en").exceptions


def test_hindi_exception_marker_is_detected():
    assert lc.extract_cues("सिवाय इसके कि अन्यथा प्रावधान हो।", "hi").exceptions


def test_quantifiers_distinguish_universal_from_existential():
    universal = lc.extract_cues("All parties must sign.", "en").by_category(lc.CATEGORY_QUANTIFIER)
    existential = lc.extract_cues("Some parties may sign.", "en").by_category(lc.CATEGORY_QUANTIFIER)
    assert universal[0].strength == lc.QUANT_UNIVERSAL
    assert existential[0].strength == lc.QUANT_EXISTENTIAL


def test_scope_marker_only_is_detected():
    cues = lc.extract_cues("Only the buyer may assign.", "en")
    assert cues.by_category(lc.CATEGORY_SCOPE)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
def test_curated_role_nouns_are_found():
    cues = lc.extract_cues("The tenant shall pay the landlord.", "en")
    assert {c.canonical for c in cues.entities} == {"tenant", "landlord"}


def test_entity_matching_respects_word_boundaries():
    """'party' must not be found inside 'partly'."""
    assert not lc.extract_cues("The goods were partly damaged.", "en").entities


def test_entities_use_multiset_semantics_in_the_delta():
    """Dropping one of two references to the tenant is a real change."""
    delta = lc.compare_texts(
        "The tenant shall inform the tenant's agent.", "The tenant shall inform the agent.", "en"
    )
    assert "tenant" in delta.dropped_entities


# ---------------------------------------------------------------------------
# Quantities — the normalisation is the point
# ---------------------------------------------------------------------------
def test_equal_durations_in_different_units_are_not_a_change():
    """'seven days' and 'one week' are the same legal deadline."""
    delta = lc.compare_texts("within seven days", "within one week", "en")
    assert delta.quantity_changes == []


def test_a_changed_number_is_reported_with_both_raw_forms():
    delta = lc.compare_texts("within 30 days", "within 60 days", "en")
    assert len(delta.quantity_changes) == 1
    change = delta.quantity_changes[0]
    assert change["unit"] == "day"
    assert change["original_raw"] == ["30 days"]
    assert change["simplified_raw"] == ["60 days"]


def test_devanagari_digits_are_read_numerically():
    quantities = lc.extract_quantities("तीस दिन", "hi")
    assert len(quantities) == 1
    assert quantities[0].value == 30
    assert quantities[0].canonical_unit == "day"


def test_hindi_number_words_require_a_unit():
    """'एक' on its own is a pronoun as often as it is the number one."""
    assert lc.extract_quantities("एक पक्षकार", "hi") == []
    assert len(lc.extract_quantities("एक दिन", "hi")) == 1


def test_bare_english_number_word_without_a_unit_is_ignored():
    assert lc.extract_quantities("one of the parties", "en") == []


def test_percent_written_with_a_symbol_is_captured():
    quantities = lc.extract_quantities("interest at 5% per annum", "en")
    assert any(q.canonical_unit == "percent" for q in quantities)


def test_rupee_amounts_are_normalised_to_the_currency_unit():
    quantities = lc.extract_quantities("five hundred rupees", "en")
    assert len(quantities) == 1
    assert quantities[0].canonical_unit == "currency"
    assert quantities[0].normalized == 500


# ===========================================================================
# REGRESSION TESTS — quantity extraction
# ===========================================================================
@pytest.mark.parametrize(
    "text,language,expected",
    [
        # The bug: only single number words were read, so the multiplier word
        # was taken as the whole value. "five hundred" became 100 (5x too
        # small) and "five thousand" became 1000 — which is the corpus's own
        # monetary threshold in ENG_CON_101 and ENG_CYB_004.
        ("five hundred rupees", "en", 500),
        ("five thousand rupees", "en", 5000),
        ("one thousand rupees", "en", 1000),
        ("two thousand five hundred rupees", "en", 2500),
        ("one hundred and fifty rupees", "en", 150),
        ("twenty five days", "en", 25),
        ("पाँच सौ रुपये", "hi", 500),
        ("दो हज़ार पाँच सौ रुपये", "hi", 2500),
    ],
)
def test_regression_multi_word_number_phrases_compose(text, language, expected):
    """Number words multiply and accumulate; they do not sum alone."""
    quantities = lc.extract_quantities(text, language)
    assert len(quantities) == 1
    assert quantities[0].normalized == expected


@pytest.mark.parametrize(
    "text,language",
    [
        ("ten lakh rupees", "en"),
        ("fifteen lakh", "en"),
        ("one lakh", "en"),
        ("ten crore rupees", "en"),
        ("दस लाख रुपये", "hi"),
    ],
)
def test_regression_lakh_and_crore_are_in_the_vocabulary(text, language):
    """The corpus uses the Indian scale, so the extractor must too.

    Twelve rows carry lakh or crore amounts; a vocabulary that stopped at
    "thousand" silently dropped every one of them, which is a missed legal
    fact rather than a rounding error.
    """
    quantities = lc.extract_quantities(text, language)
    assert len(quantities) == 1
    assert quantities[0].normalized >= 100_000


def test_regression_a_magnitude_word_is_a_quantity_without_a_trailing_unit():
    """"one lakh" is unambiguous; a bare "one" is not.

    Requiring a unit word made "extend to one lakh" extract nothing at all.
    A multiplier word is evidence enough on its own.
    """
    assert lc.extract_quantities("extend to one lakh", "en")[0].normalized == 100_000
    # The rule that motivated the unit requirement still holds.
    assert lc.extract_quantities("one of the parties", "en") == []


def test_regression_a_number_word_inside_a_longer_word_is_not_a_quantity():
    """Without a boundary, ``a`` matches inside "dat[a] days".

    "data days" was read as a one-day deadline — a legal fact fabricated
    entirely by the extractor. ``\\b`` cannot fix this because it is defined in
    terms of ``\\w``, and Devanagari matras are combining marks that ``\\w``
    does not match.
    """
    assert lc.extract_quantities("data days are irrelevant", "en") == []
    assert lc.extract_quantities("than a party may assign", "en") == []


def test_regression_the_word_than_does_not_contribute_a_number():
    """"not less than forty-five days" is 45 days, not 46.

    The ``an`` at the end of ``than`` was matched as the number word "an" and
    composed with ``forty-five``.
    """
    quantities = lc.extract_quantities("not less than forty-five days", "en")
    assert len(quantities) == 1
    assert quantities[0].normalized == 45


def test_regression_hyphenated_number_words_compose_as_one_value():
    """"forty-five" is 45, not 40 followed by an unrelated 5."""
    quantities = lc.extract_quantities("not less than forty-five days", "en")
    assert len(quantities) == 1
    assert quantities[0].normalized == 45
    assert quantities[0].raw == "forty-five days"


def test_regression_percent_symbol_survives_the_general_pass():
    """The percent branch must run first or it is dead code.

    The general pattern matches the bare "5" in "5%", so when it ran first it
    claimed that span as a unitless count; the percent pass then found the
    span already consumed and skipped it.  Every percentage in the corpus was
    consequently labelled a plain count.
    """
    quantities = lc.extract_quantities("interest at 5% per annum", "en")
    assert len(quantities) == 1
    assert quantities[0].canonical_unit == "percent"
    assert quantities[0].raw == "5%"


def test_regression_quantity_offsets_still_index_the_source():
    """Composing runs of words must not shift the reported span."""
    text = "a penalty of five lakh rupees may be imposed"
    quantity = lc.extract_quantities(text, "en")[0]
    assert text[quantity.start : quantity.end] == quantity.raw


# ---------------------------------------------------------------------------
# Offsets: explanations cite spans, so offsets must be usable
# ---------------------------------------------------------------------------
def test_cue_offsets_index_the_original_string():
    text = "The buyer shall not assign."
    cues = lc.extract_cues(text, "en")
    for cue in cues.cues:
        assert text[cue.start : cue.end] == cue.surface


def test_hindi_cue_offsets_index_the_original_string():
    """NFKC can change length, so matching must run on the raw text."""
    text = "क्रेता माल नहीं लौटाएगा।"
    cues = lc.extract_cues(text, "hi")
    assert cues.cues
    for cue in cues.cues:
        assert text[cue.start : cue.end] == cue.surface


def test_quantity_offsets_index_the_original_string():
    text = "within 30 days of the notice"
    for quantity in lc.extract_quantities(text, "en"):
        assert text[quantity.start : quantity.end] == quantity.raw


# ---------------------------------------------------------------------------
# CueDelta
# ---------------------------------------------------------------------------
def test_delta_detects_modal_weakening():
    delta = lc.compare_texts("The buyer shall pay.", "The buyer may pay.", "en")
    assert delta.modal_weakened
    assert not delta.modal_strengthened


def test_delta_detects_modal_strengthening():
    delta = lc.compare_texts("The buyer may pay.", "The buyer shall pay.", "en")
    assert delta.modal_strengthened


def test_delta_distinguishes_weakening_from_strengthening_when_one_side_lacks_a_modal():
    """Removing a modal entirely is neither: ``None`` is not a strength of 0."""
    delta = lc.compare_texts("The buyer shall pay.", "The buyer pays.", "en")
    assert not delta.modal_weakened
    assert not delta.modal_strengthened


def test_delta_detects_negation_flip_in_one_direction_only():
    delta = lc.compare_texts("The buyer shall return.", "The buyer shall not return.", "en")
    assert delta.negation_flipped
    assert delta.negation_count_original == 0
    assert delta.negation_count_simplified == 1


def test_delta_detects_a_dropped_condition():
    delta = lc.compare_texts(
        "The buyer shall pay if the goods conform.", "The buyer shall pay.", "en"
    )
    assert delta.dropped_conditions
    assert not delta.added_conditions


def test_delta_detects_an_added_exception():
    delta = lc.compare_texts(
        "The buyer shall pay.", "The buyer shall pay, except in the case of force majeure.", "en"
    )
    assert delta.added_exceptions


def test_identical_texts_produce_an_empty_delta():
    delta = lc.compare_texts(
        "The buyer shall not return the goods within 30 days.",
        "The buyer shall not return the goods within 30 days.",
        "en",
    )
    assert not delta.has_any_change
    assert lc.change_summary(delta) == []


def test_change_summary_names_the_phenomenon_in_plain_language():
    delta = lc.compare_texts("The buyer shall pay.", "The buyer may pay.", "en")
    summary = lc.change_summary(delta)
    assert any("Obligation" in line and "Permission" in line for line in summary)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def test_empty_and_non_string_input_yield_an_empty_profile():
    for value in ("", "   ", None):
        profile = lc.extract_cues(value, "en")
        assert profile.cues == []
        assert profile.quantities == []


def test_unknown_language_falls_back_to_english_but_keeps_the_code():
    """A wrong language code must not crash the annotation tool mid-session."""
    profile = lc.extract_cues("The buyer shall pay.", "fr")
    assert profile.language == "fr"
    assert profile.modals


def test_profiles_and_deltas_serialise_to_plain_dicts():
    import json

    profile = lc.extract_cues("The buyer shall not pay within 30 days.", "en")
    delta = lc.compare_texts("The buyer shall pay.", "The buyer may pay.", "en")
    # Must be JSON-serialisable: both end up in results files and in the API.
    json.dumps(profile.to_dict())
    json.dumps(delta.to_dict())


@pytest.mark.parametrize(
    "text,language",
    [
        ("The tenant shall not sublet the premises.", "en"),
        ("किरायेदार परिसर उपकिराया नहीं करेगा।", "hi"),
    ],
)
def test_cues_never_overlap_within_a_category(text, language):
    """Within one category, spans must be disjoint.

    Overlap is resolved inside each category by ``_resolve_overlaps``; a
    regression there would double-count a cue in the delta.  Across
    *categories* overlap is deliberate and must be preserved — see the test
    below.
    """
    cues = lc.extract_cues(text, language).cues
    by_category: dict[str, list] = {}
    for cue in cues:
        by_category.setdefault(cue.category, []).append(cue)
    for category, group in by_category.items():
        ordered = sorted(group, key=lambda c: c.start)
        for earlier, later in zip(ordered, ordered[1:]):
            assert earlier.end <= later.start, category


def test_a_negation_inside_a_modal_is_reported_in_both_categories():
    """'shall not' is a prohibition *and* a negation, and both must survive.

    This looks like redundant overlap worth de-duplicating, and it is not.
    ``compare_profiles`` counts negations to compute ``negation_flipped``, so
    if the negation cue were resolved away as a sub-span of the prohibition,
    the single most important meaning change in the whole project —
    obligation silently becoming permission, or a negation being dropped —
    would stop being detected in exactly the constructions that carry it.
    """
    cues = lc.extract_cues("The buyer shall not return the goods.", "en")
    assert [c.canonical for c in cues.modals] == ["shall not"]
    assert [c.surface for c in cues.negations] == ["not"]

    delta = lc.compare_texts("The buyer shall not return.", "The buyer shall return.", "en")
    assert delta.negation_flipped


# ===========================================================================
# REGRESSION TESTS — Devanagari word boundaries
# ===========================================================================
# ``\b`` is defined in terms of ``\w``, and ``\w`` does not match Devanagari
# combining marks, so ``\b`` fires in the middle of a Hindi word.  The lexicons
# used ``न\b`` and ``मत\b`` for the standalone particles, which made those
# patterns match the tail of करना, पालन, निर्वहन, सहमति and सहमत.  On the
# corpus it produced 2,066 negation cues where 121 real ones exist — and
# because ``negation_flipped`` compares whether each side has *any* negation,
# the noise hid 26 of the 30 pairs that genuinely add or drop one.
def test_regression_a_negation_particle_inside_a_word_is_not_a_negation():
    """करना and पालन contain no negation; each ends in the letter न."""
    assert lc.extract_cues("पक्षकारों को पालन करना होगा।", "hi").negations == []
    assert lc.extract_cues("उसके दायित्व का निर्वहन करने का वचन", "hi").negations == []


def test_regression_consent_words_are_not_negations():
    """सहमति ("consent") and सहमत ("agreed") end in the particle मत, and are not it."""
    for text in ("स्वतंत्र सहमति से किया गया", "वे एक ही बात पर सहमत हों।"):
        assert lc.extract_cues(text, "hi").negations == []


@pytest.mark.parametrize("text", ["शून्य घोषित न हों।", "अन्यथा उपबंधित न हो।", "वह मत जाए।"])
def test_regression_a_standalone_negation_particle_is_still_found(text):
    """Anchoring must not cost the real particle: ``न``/``मत`` as whole words."""
    assert lc.extract_cues(text, "hi").negations


def test_regression_a_dropped_hindi_negation_is_visible_to_the_delta():
    """The failure this guards: a real negation drop reported as no change.

    HIN_CON_017's simplification drops "no suit shall be brought ..." entirely.
    Before the boundary fix both sides carried phantom cues from करना/पालन, so
    ``negation_flipped`` was False and the omission was invisible.
    """
    delta = lc.compare_texts(
        "बाजी के रूप में किए गए करार शून्य हैं; और किसी बाजी पर जीतने का दावा "
        "किए गए किसी भी धन की वसूली के लिए कोई वाद नहीं लाया जाएगा।",
        "बाजी के रूप में किए गए करार शून्य हैं।",
        "hi",
    )
    assert delta.negation_flipped
    assert delta.negation_count_simplified == 0


def test_regression_bahar_is_not_a_universal_quantifier():
    """बाहर ("outside") ends in हर ("every")."""
    cues = lc.extract_cues("किसी कंपनी के नियंत्रण से बाहर", "hi")
    assert cues.by_category(lc.CATEGORY_QUANTIFIER) == []
    assert lc.extract_cues("आवश्यक हर विधिपूर्ण कार्य", "hi").by_category(
        lc.CATEGORY_QUANTIFIER
    )


def test_regression_maatra_is_not_a_scope_marker():
    """मात्रा ("amount") and मात्रात्मक ("quantitative") are not मात्र ("merely")."""
    assert lc.extract_cues("क्षतिपूर्ति की मात्रा का निर्णय", "hi").by_category(
        lc.CATEGORY_SCOPE
    ) == []
    assert lc.extract_cues("लेनदार की मात्र सहनशीलता", "hi").by_category(lc.CATEGORY_SCOPE)


def test_regression_inflected_hindi_quantifiers_survive_anchoring():
    """Anchoring is only for a substring that is a *different* word.

    भागों and सबके are inflections of भाग and सब, so anchoring them would
    delete good cues rather than spurious ones.
    """
    assert lc.extract_cues("अपने-अपने भागों के अनुपात में", "hi").by_category(
        lc.CATEGORY_QUANTIFIER
    )
    assert lc.extract_cues("जीवनकाल में उन सबके पास", "hi").by_category(
        lc.CATEGORY_QUANTIFIER
    )
