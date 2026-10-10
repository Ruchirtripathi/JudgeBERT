"""Streamlit annotation tool for Legal Meaning Preservation.

Run with::

    streamlit run src/annotation/annotation_app.py

Three tabs
----------
**Annotate** — score pairs on the 1-10 LMP scale.  Strictly blind: the
curator's draft score is never rendered by this tool, in any tab, before or
after a score is saved.  A draft is an automatically generated expected label,
and showing it — even post-hoc, when the annotator can still revise — anchors
the judgement the annotation exists to obtain independently (guidelines §5.1).
The drafts stay in the master corpus for the curator's own post-hoc analysis
(:mod:`src.annotation.curator_review`); they are not annotator material.

**Review** — your own progress and your own score distribution, plus the
corpus-wide agreement coefficients.  Per-pair scores from *other* annotators
are deliberately withheld here: an annotator who can see a colleague's score
mid-pass is no longer annotating independently, and agreement computed from
that is not agreement.  The per-pair view belongs to the adjudicator.

**Adjudicate** — pairs where two annotators differ by 2 or more points; record
a consensus score.  The adjudicator must be a different person from either
annotator on the pair, and the tool refuses the record if they are not.

Each annotator's scores are written to their own file under
``data/annotations/annotators/<annotator_id>.csv``.  The master corpus is not
touched by this tool — that happens in
:mod:`src.annotation.merge_annotations`, so a label in the master can always be
traced to a file this app wrote.

Testability
-----------
Nothing here runs Streamlit at import time; :func:`_page_setup` is called from
:func:`main`.  An import-time ``st.set_page_config`` would make the module
impossible to import outside a script run, and the persistence helpers —
the only code that may write a human label — would then be untestable.
"""

from __future__ import annotations

import hashlib
import numbers
import sys, os
sys.path.insert(0, os.path.expanduser('~/.pyenv/versions/3.12.6/lib/python3.12/site-packages'))
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs this file as a script, so the repository root is not on
# sys.path.  Add it before importing anything from src.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotation import agreement  # noqa: E402
from src.data import schema  # noqa: E402
from src.data.load_data import load_master  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.io import utc_now_iso  # noqa: E402

#: Longest accepted annotator id. Ids become filenames, so they are bounded.
MAX_ANNOTATOR_ID_LENGTH = 40


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
#: Devanagari needs a font stack that does not fall back to tofu on a machine
#: with no Devanagari font installed.
_PAGE_CSS = """
    <style>
      .lmp-sentence {
          font-size: 1.05rem;
          line-height: 1.65;
          padding: 0.9rem 1.1rem;
          border-radius: 0.5rem;
          border: 1px solid rgba(128,128,128,0.35);
          background: rgba(128,128,128,0.07);
          font-family: "Noto Sans", "Noto Sans Devanagari", "Nirmala UI",
                       "Mangal", system-ui, sans-serif;
      }
      .lmp-label {
          font-size: 0.78rem;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          opacity: 0.65;
          margin-bottom: 0.25rem;
      }
      .lmp-meta { font-size: 0.85rem; opacity: 0.75; }
    </style>
"""


def _page_setup() -> None:
    """Configure the page and inject the stylesheet.

    Called at the top of :func:`main` rather than at import time so that this
    module can be imported by the test suite.
    """
    st.set_page_config(
        page_title="LMP Annotation Tool",
        page_icon="⚖️",
        layout="wide",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached config / corpus
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _config_fingerprint() -> str:
    cfg = load_config()
    return str(cfg["project"]["version"])


@st.cache_data(show_spinner=False)
def _load_corpus() -> pd.DataFrame:
    """Read the master corpus.

    ``subset="all"`` on purpose: the annotation tool must be able to show rows
    that carry no human score yet, which is precisely the interesting set.
    """
    return load_master(subset="all")


@st.cache_data(show_spinner=False)
def _decision_thresholds() -> tuple[float, float]:
    """``(preserved_min, partial_min)`` from the config.

    Read from the same place :mod:`src.annotation.merge_annotations` reads it,
    so the decision the annotator sees while scoring is the decision the merge
    will write.
    """
    decision_cfg = load_config().get("decision", {})
    return (
        float(decision_cfg.get("preserved_min", 8.0)),
        float(decision_cfg.get("partial_min", 5.0)),
    )


def _shuffle_seed(annotator_id: str) -> int:
    """A stable per-annotator queue seed.

    ``hash()`` is salted per process, so it cannot be used here: the queue
    would reorder itself on every restart and an annotator could not resume in
    a consistent position.  A SHA-256 prefix is stable across runs and
    machines.

    Two annotators get different orders, which is the point.  If both work the
    same sequence they also meet the awkward pairs at the same point in their
    session, so position and fatigue stop being independent of each other.
    """
    digest = hashlib.sha256(annotator_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _queue_frame(
    df: pd.DataFrame,
    language: str,
    domain: str,
    split: str,
    order: str,
    seed: int,
) -> pd.DataFrame:
    """Apply the sidebar filters and ordering to build the annotation queue.

    There is deliberately no "sort by draft score" option.  Sorting by a hidden
    field leaks it: the position of a pair in the queue would tell the
    annotator what the draft says about it, which is exactly the anchor blind
    mode exists to remove.
    """
    out = df
    if language != "all":
        out = out[out["language"] == language]
    if domain != "all":
        out = out[out["legal_domain"] == domain]
    if split != "all":
        out = out[out["split"] == split]

    if order == "Pair id":
        out = out.sort_values("pair_id")
    else:  # per-annotator seeded shuffle
        out = out.sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def _next_cursor(index: int, n_queue: int, only_unscored: bool) -> int:
    """Index to display after the pair at ``index`` has been saved.

    The two modes move differently, and getting this wrong silently skips
    pairs — the failure is invisible, because a skipped pair simply never
    appears.

    * With ``only_unscored`` the saved pair drops out of the queue on the next
      run, so every later pair slides down one place.  Staying on the same
      index therefore lands on the *next* pair; advancing would step over it.
    * With the whole corpus shown the queue does not move, so the cursor
      advances.
    """
    if only_unscored:
        return index
    return min(index + 1, max(n_queue - 1, 0))


# ---------------------------------------------------------------------------
# Annotation persistence
# ---------------------------------------------------------------------------
def _annotations_root(cfg: dict | None = None) -> Path:
    """The directory holding every annotation artefact.

    The single seam through which the tool decides where annotation files
    live, so the test suite can point it at a temporary directory instead of
    the real corpus's annotation store.
    """
    return resolve_path(cfg or load_config(), "annotations_dir")


def _validate_annotator_id(value: str) -> str:
    """Return a safe annotator id, or raise :class:`ValueError`.

    The id becomes a filename, so it is constrained rather than trusted.
    ``A01`` is fine; ``../../processed/lmp_pairs`` is not, and neither is a
    hidden file.  Pseudonymity is a separate matter, handled by the UI help
    text — this only stops the id from escaping the annotation directory.
    """
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("An annotator id is required.")
    if len(cleaned) > MAX_ANNOTATOR_ID_LENGTH:
        raise ValueError(
            f"An annotator id may be at most {MAX_ANNOTATOR_ID_LENGTH} characters."
        )
    if cleaned.startswith(".") or any(
        character in cleaned for character in ("/", "\\", ":", "\x00")
    ):
        raise ValueError(
            "An annotator id may not contain slashes, colons or start with a dot."
        )
    if not any(character.isalnum() for character in cleaned):
        raise ValueError("An annotator id must contain at least one letter or digit.")
    return cleaned


def _annotator_path(annotator_id: str) -> Path:
    directory = _annotations_root() / "annotators"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_validate_annotator_id(annotator_id)}.csv"


def _clean_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Split an annotator file into usable rows and rejected ones.

    A score is usable only if it is an integer in 1-10.  Anything else — a
    hand-edit, a stray blank, a value typed into the CSV by another tool — is
    dropped rather than coerced, and returned so the caller can say so.  It
    must never reach the scale: quadratic kappa and the ordinal metric are
    both defined in terms of distance on the declared range, so an 11 or a 0
    yields a coefficient that looks like a real result and is not.
    """
    if frame.empty:
        return frame, []

    scores = pd.to_numeric(frame["human_lmp_score"], errors="coerce")
    integral = scores.notna() & (scores % 1 == 0)
    in_range = scores.between(schema.SCORE_MIN, schema.SCORE_MAX)
    valid = integral & in_range

    rejected = [
        {"pair_id": str(row.pair_id), "human_lmp_score": row.human_lmp_score}
        for row in frame.loc[~valid].itertuples(index=False)
    ]
    cleaned = frame.loc[valid].copy()
    cleaned["human_lmp_score"] = scores.loc[valid].astype(int)
    return cleaned, rejected


def _load_own_scores(annotator_id: str) -> pd.DataFrame:
    """This annotator's saved scores, with unusable rows dropped.

    Any rejected rows are attached as ``frame.attrs["rejected"]`` so the caller
    can surface them; see :func:`_rejected_scores`.
    """
    path = _annotator_path(annotator_id)
    if not path.exists():
        frame = pd.DataFrame(columns=agreement.ANNOTATION_COLUMNS)
        frame.attrs["rejected"] = []
        return frame
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    for column in agreement.ANNOTATION_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""
    frame, rejected = _clean_scores(raw[agreement.ANNOTATION_COLUMNS])
    frame.attrs["rejected"] = rejected
    return frame


def _rejected_scores(frame: pd.DataFrame) -> list[dict]:
    """Rows :func:`_load_own_scores` dropped, for display."""
    return list(frame.attrs.get("rejected") or [])


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Replace ``path`` atomically.

    ``to_csv(path)`` opens the destination for writing, which truncates it
    first: a crash, a full disk or a closed laptop part-way through would leave
    a half-written file where the annotations used to be.  Those rows are hours
    of hand-entered human judgement and cannot be regenerated, so the write
    goes to a temporary file beside the target and is moved into place with
    :func:`os.replace`, which is atomic on POSIX.  Either the previous file
    survives intact or the new one is complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _validated_score(score: object, *, context: str) -> int:
    """Return ``score`` as an integer in 1-10, or raise :class:`ValueError`.

    The radio widget already restricts the UI, but annotation files are plain
    CSVs on disk: a hand-edit, a merge accident or a future interface change
    can put anything in them.  Enforcing the range here means nothing this tool
    wrote can be out of range, whatever the caller passed.
    """
    if isinstance(score, bool) or not isinstance(score, numbers.Integral):
        raise ValueError(f"{context}: the score must be a whole number, got {score!r}.")
    value = int(score)
    if not schema.SCORE_MIN <= value <= schema.SCORE_MAX:
        raise ValueError(
            f"{context}: {value} is outside the {schema.SCORE_MIN}-"
            f"{schema.SCORE_MAX} LMP scale."
        )
    return value


def _save_score(
    annotator_id: str,
    pair_id: str,
    score: int,
    error_type: str,
    notes: str,
) -> Path:
    """Upsert one annotation into this annotator's file.

    Upsert rather than append: re-scoring a pair after re-reading it should
    replace the earlier judgement, not create a second row for the same pair
    that the merge would then have to disambiguate.
    """
    value = _validated_score(score, context=f"{pair_id}")
    frame = _load_own_scores(annotator_id)
    record = {
        "pair_id": pair_id,
        "human_lmp_score": value,
        "error_type": error_type,
        "notes": notes.strip(),
        "annotator_id": annotator_id,
        "annotated_at": utc_now_iso(),
    }
    frame = frame[frame["pair_id"] != pair_id]
    frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
    frame = frame[agreement.ANNOTATION_COLUMNS]
    path = _annotator_path(annotator_id)
    _atomic_write_csv(frame, path)
    return path


def _save_adjudication(
    pair_id: str, score: int, error_type: str, adjudicator_id: str, notes: str
) -> Path:
    value = _validated_score(score, context=f"adjudication of {pair_id}")
    path = _annotations_root() / "adjudication.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    else:
        frame = pd.DataFrame(columns=agreement.ADJUDICATION_COLUMNS)

    record = {
        "pair_id": pair_id,
        "consensus_score": value,
        "error_type": error_type,
        "adjudicator_id": adjudicator_id,
        "notes": notes.strip(),
        "adjudicated_at": utc_now_iso(),
    }
    frame = frame[frame["pair_id"] != pair_id]
    frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
    frame = frame[agreement.ADJUDICATION_COLUMNS]
    _atomic_write_csv(frame, path)
    return path


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _sentence(label: str, text: str) -> None:
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f'<div class="lmp-label">{label}</div><div class="lmp-sentence">{safe}</div>',
        unsafe_allow_html=True,
    )


def _score_index(current: int | None) -> int | None:
    """Which score option to preselect, or ``None`` to preselect nothing.

    Out-of-range values return ``None`` rather than raising. They cannot occur
    through the widget, but they can arrive from a hand-edited file, and
    ``list(range(1, 11)).index(11)`` raises ``ValueError`` — a single bad score
    used to take the whole session down. A file that is damaged should be
    reported, not fatal.
    """
    if current is None:
        return None
    value = int(current)
    if schema.SCORE_MIN <= value <= schema.SCORE_MAX:
        return value - schema.SCORE_MIN
    return None


def _score_widget(key: str, current: int | None) -> int | None:
    """1-10 selector with nothing preselected.

    ``index=None`` matters: a slider that starts at 5 invites the annotator to
    accept 5, which would put a spike in the middle of the distribution that
    reflects the interface rather than the corpus.
    """
    options = list(range(schema.SCORE_MIN, schema.SCORE_MAX + 1))
    return st.radio(
        "Legal Meaning Preservation score",
        options=options,
        index=_score_index(current),
        horizontal=True,
        key=key,
        help=(
            "10 = legally indistinguishable, 5 = a real operative element lost, "
            "1 = the rewrite contradicts the original. See guidelines.md §2."
        ),
    )


def _adjudicator_conflict(adjudicator_id: str, annotators_on_pair: set[str]) -> str | None:
    """Why ``adjudicator_id`` may not adjudicate this pair, or ``None`` if they may.

    Adjudication exists to supply a reading independent of the two annotators
    in dispute. An annotator resolving their own disagreement supplies nothing
    of the sort: the resulting ``human_consensus`` row would be one person's
    opinion wearing a consensus label, and it would be indistinguishable in the
    data from a genuine third reading.
    """
    if adjudicator_id and adjudicator_id in annotators_on_pair:
        return (
            f"`{adjudicator_id}` is one of the annotators in dispute for this "
            "pair. Adjudication requires a different person."
        )
    return None


def _decision_preview(score: int | None) -> None:
    """Show the three-way decision the current score maps to.

    The decision is *derived*, not annotated — see guidelines §3 and
    :func:`src.data.schema.score_to_decision`.  Showing it is feedback on the
    annotator's own input using the documented thresholds, so it cannot anchor
    a judgement the way a hidden draft score does; it just saves them reading
    the threshold table.
    """
    if score is None:
        st.caption("Decision: _pick a score_ — it follows from the score mechanically.")
        return
    preserved_min, partial_min = _decision_thresholds()
    decision = schema.score_to_decision(score, preserved_min, partial_min)
    st.caption(
        f"Decision: **{decision}** — derived from the score "
        f"(≥{preserved_min:g} Preserved · ≥{partial_min:g} Partially Preserved)."
    )


# ---------------------------------------------------------------------------
# Tab 1 — Annotate
# ---------------------------------------------------------------------------
def tab_annotate(corpus: pd.DataFrame, annotator_id: str) -> None:
    st.header("Annotate")
    st.caption(
        "Score how much of the **legal** meaning survives. Read "
        "`src/annotation/guidelines.md` before your first pass — §2 holds the "
        "scale anchors and §7 the common confusions."
    )

    with st.sidebar:
        st.subheader("Queue")
        language = st.selectbox("Language", ["all", "en", "hi"], key="q_lang")
        domain = st.selectbox("Domain", ["all", *sorted(corpus["legal_domain"].dropna().unique())], key="q_dom")
        split = st.selectbox("Split", ["all", *schema.SPLITS], key="q_split")
        order = st.radio("Order", ["Pair id", "Shuffled"], key="q_order")
        only_unscored = st.checkbox("Only pairs I have not scored", value=True, key="q_unscored")

    seed = _shuffle_seed(annotator_id)
    queue = _queue_frame(corpus, language, domain, split, order, seed)
    own = _load_own_scores(annotator_id)

    rejected = _rejected_scores(own)
    if rejected:
        st.warning(
            f"{len(rejected)} saved row(s) hold a score outside "
            f"{schema.SCORE_MIN}-{schema.SCORE_MAX} or not a whole number, and are "
            "ignored: "
            + ", ".join(f"`{row['pair_id']}`={row['human_lmp_score']!r}" for row in rejected[:5])
            + (". Fix or delete them in your annotation file." if len(rejected) <= 5 else " …")
        )

    scored_ids = set(own["pair_id"].astype(str))

    if only_unscored:
        queue = queue[~queue["pair_id"].astype(str).isin(scored_ids)].reset_index(drop=True)

    total_in_view = len(_queue_frame(corpus, language, domain, split, order, seed))
    n_scored_in_view = len(scored_ids & set(_queue_frame(corpus, language, domain, split, order, seed)["pair_id"].astype(str)))

    st.progress(
        min(n_scored_in_view / total_in_view, 1.0) if total_in_view else 0.0,
        text=f"{n_scored_in_view} / {total_in_view} pairs scored in this view",
    )
    if queue.empty:
        st.success("Nothing left in this queue. Change the filters, or move to Review.")
        # No items in the current filtered queue. Reset cursor for future runs.
        # (Filter change handling occurs later.)
        return

    # Detect sidebar filter changes and reset cursor if needed
    prev = st.session_state.get('prev_filters')
    curr = {
        'language': language,
        'domain': domain,
        'split': split,
        'order': order,
        'only_unscored': only_unscored,
    }
    if prev != curr:
        # If filters changed, start from first item in new queue
        st.session_state['cursor'] = 0
    st.session_state['prev_filters'] = curr

    cursor_key = "cursor"
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = 0
    st.session_state[cursor_key] = min(st.session_state[cursor_key], len(queue) - 1)
    idx = st.session_state[cursor_key]
    row = queue.iloc[idx]


    pair_id = str(row["pair_id"])
    already = own[own["pair_id"].astype(str) == pair_id]



    # Layout three columns for navigation controls
    nav_left, nav_mid, nav_right = st.columns([1, 4, 1])
    with nav_left:
        # Added explicit key to avoid widget conflict across re‑runs
        if st.button("← Previous", key="nav_prev", disabled=idx == 0, use_container_width=True):
            st.session_state[cursor_key] = idx - 1
            st.rerun()
    with nav_mid:
        st.markdown(
            f'<div class="lmp-meta">Pair <b>{idx + 1}</b> of <b>{len(queue)}</b> '
            f'&nbsp;·&nbsp; <code>{pair_id}</code> '
            f'&nbsp;·&nbsp; {schema.LANGUAGES.get(str(row["language"]), row["language"])} '
            f'&nbsp;·&nbsp; {schema.DOMAINS.get(str(row["legal_domain"]), row["legal_domain"])} '
            f'&nbsp;·&nbsp; {row["source"]}, s.{row["source_section"]}</div>',
            unsafe_allow_html=True,
        )
    with nav_right:
        # Added explicit key for consistency
        if st.button("Next →", key="nav_next", disabled=idx >= len(queue) - 1, use_container_width=True):
            st.session_state[cursor_key] = idx + 1
            st.rerun()

    st.divider()

    _sentence("Original — authoritative wording", row["original_text"])
    st.write("")
    _sentence("Simplified — the rewrite being judged", row["simplified_text"])

    context = row.get("context")
    if isinstance(context, str) and context.strip():
        with st.expander("Surrounding context"):
            st.write(context)

    st.divider()

    current_score = None
    if not already.empty and pd.notna(already.iloc[0]["human_lmp_score"]):
        current_score = int(already.iloc[0]["human_lmp_score"])

    score = _score_widget(f"score_{pair_id}_{idx}", current_score)
    _decision_preview(score)

    error_type = st.selectbox(
        "Primary legal error",
        options=schema.ERROR_LABELS,
        index=schema.ERROR_LABELS.index(str(already.iloc[0]["error_type"]))
        if not already.empty and str(already.iloc[0]["error_type"]) in schema.ERROR_LABELS
        else None,
        key=f"err_{pair_id}_{idx}",
        help=(
            "The single defect that most drives your score. Choose 'None' when "
            "the rewrite preserves everything operative. See guidelines.md §4."
        ),
    )

    notes = st.text_input(
        "One sentence on what drove your score (optional)",
        value=str(already.iloc[0]["notes"]) if not already.empty else "",
        key=f"notes_{pair_id}_{idx}",
        placeholder="e.g. Drops the proviso guaranteeing a reasonable opportunity to be heard.",
    )

    save = st.button("Save & next", type="primary")

    if save:
        problems = []
        if score is None:
            problems.append("Choose a score before saving.")
        if error_type is None:
            problems.append(
                "Choose a primary error type — 'None' is the answer when nothing "
                "operative was lost."
            )
        if error_type == "Other" and not notes.strip():
            problems.append("An 'Other' error needs a note saying what it is.")
        if not problems:
            try:
                path = _save_score(annotator_id, pair_id, score, error_type, notes)
            except ValueError as exc:
                # An out-of-range or non-integer score must never be written;
                # say so instead of taking the whole session down.
                st.error(f"Not saved: {exc}")
            else:
                st.toast(f"Saved {pair_id} → {score}/10 ({path.name})")
                st.session_state[cursor_key] = _next_cursor(idx, len(queue), only_unscored)
                st.rerun()
        for problem in problems:
            st.warning(problem)

    st.caption(
        "🔒 The curator's draft score is not shown anywhere in this tool. "
        "Scoring blind is what makes the label independent; see guidelines.md §5.1."
    )


# ---------------------------------------------------------------------------
# Tab 2 — Review
# ---------------------------------------------------------------------------
def tab_review(corpus: pd.DataFrame, annotator_id: str) -> None:
    st.header("Review")
    own = _load_own_scores(annotator_id)
    total = len(corpus)
    n_own = len(own)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pairs you have scored", f"{n_own}", f"{n_own / total:.0%} of corpus" if total else None)
    if n_own:
        col2.metric("Your mean score", f"{own['human_lmp_score'].mean():.2f}")
        col3.metric(
            "Below 5 (Not Preserved)",
            f"{int((own['human_lmp_score'] < 5).sum())}",
        )

    st.progress(min(n_own / total, 1.0) if total else 0.0)
    if n_own < total:
        st.info(
            f"**{total - n_own} pair(s) still unannotated** in your queue. "
            "Experiments cannot be run on partial annotation."
        )
    else:
        st.success("You have scored the whole corpus. Run the agreement check below.")

    st.divider()
    st.subheader("Inter-annotator agreement")

    long_df = agreement.load_annotations()
    result = agreement.compute_agreement(long_df)

    if result["status"] == agreement.RESULT_PENDING:
        st.warning(f"**{agreement.RESULT_PENDING}** — {result.get('reason')}")
    else:
        kappa = result["cohen_kappa_quadratic"]
        alpha = result["krippendorff_alpha_ordinal"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cohen's κ (quadratic)", f"{kappa:.3f}", result["cohen_kappa_band"])
        c2.metric("Krippendorff's α", f"{alpha:.3f}")
        c3.metric("Within ±1 point", f"{result['percent_within_1']:.1%}")
        st.caption(
            f"Computed over {result['n_pairs_overlapping']} pairs scored by "
            f"{result['n_annotators']} annotators. These are measured values; "
            "they are reported as computed. See guidelines.md §6."
        )
        st.caption(
            "Per-pair scores are withheld from this tab. Seeing another "
            "annotator's score for a pair before you have scored it yourself "
            "would make the two judgements dependent, and the coefficient above "
            "would then be measuring something other than agreement. Disagreements "
            "are listed in the **Adjudicate** tab, for the adjudicator."
        )

    st.divider()
    with st.expander("Your scores"):
        if own.empty:
            st.write("No scores saved yet.")
        else:
            preserved_min, partial_min = _decision_thresholds()
            table = own.copy()
            table["decision"] = [
                schema.score_to_decision(value, preserved_min, partial_min)
                for value in table["human_lmp_score"]
            ]
            st.dataframe(
                table[
                    ["pair_id", "human_lmp_score", "decision", "error_type", "notes", "annotated_at"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download my annotations (CSV)",
                data=own.to_csv(index=False).encode("utf-8"),
                file_name=f"{annotator_id}_annotations.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Tab 3 — Adjudicate
# ---------------------------------------------------------------------------
def tab_adjudicate(corpus: pd.DataFrame) -> None:
    st.header("Adjudicate")
    st.caption(
        "Pairs where two annotators differ by 2 or more points. A consensus "
        "score recorded here is what `merge_annotations` writes as "
        "`human_consensus`."
    )

    long_df = agreement.load_annotations()
    if long_df.empty:
        st.info("No annotations on disk yet.")
        return

    matrix = agreement.rating_matrix(long_df)
    disputes = agreement.disagreement_table(matrix)
    adjudications = agreement.load_adjudications()
    resolved = set(adjudications["pair_id"].astype(str)) if not adjudications.empty else set()
    disputes = disputes[~disputes["pair_id"].astype(str).isin(resolved)]

    if disputes.empty:
        st.success("Nothing awaiting adjudication.")
        return

    st.write(f"**{len(disputes)}** pair(s) in dispute.")
    pair_id = st.selectbox("Pair", disputes["pair_id"].astype(str).tolist(), key="adj_pair")
    detail = disputes[disputes["pair_id"].astype(str) == pair_id].iloc[0]

    st.markdown(f"**Annotators:** {detail['annotators']} — spread of {detail['spread']:g} points.")

    row = corpus[corpus["pair_id"].astype(str) == pair_id]
    if not row.empty:
        row = row.iloc[0]
        _sentence("Original — authoritative wording", row["original_text"])
        st.write("")
        _sentence("Simplified — the rewrite being judged", row["simplified_text"])

    st.divider()
    st.subheader("Each annotator's reasoning")
    for entry in long_df[long_df["pair_id"].astype(str) == pair_id].itertuples(index=False):
        st.markdown(
            f"- **{entry.annotator_id}** scored **{entry.human_lmp_score:g}/10**, "
            f"error type `{entry.error_type}` — {entry.notes or '_(no note)_'}"
        )

    st.divider()
    adjudicator_id = st.text_input(
        "Adjudicator id",
        key="adj_id",
        help="Must be a different person from both annotators above. Never your real name.",
    )
    consensus = _score_widget("adj_score", None)
    _decision_preview(consensus)
    error_type = st.selectbox("Consensus error type", schema.ERROR_LABELS, key="adj_err")
    rationale = st.text_area(
        "Why this score resolves the disagreement",
        key="adj_notes",
        placeholder="Explain the reading you settled on, so the paper can report it.",
    )

    if st.button("Record adjudication", type="primary"):
        problems = []
        if not adjudicator_id.strip():
            problems.append("An adjudicator id is required.")
        annotators_on_pair = set(
            long_df.loc[long_df["pair_id"].astype(str) == pair_id, "annotator_id"].astype(str)
        )
        conflict = _adjudicator_conflict(adjudicator_id.strip(), annotators_on_pair)
        if conflict:
            problems.append(conflict)
        if consensus is None:
            problems.append("Choose a consensus score.")
        if not rationale.strip():
            problems.append("Record the reasoning — an unexplained adjudication cannot be defended.")
        if problems:
            for problem in problems:
                st.warning(problem)
        else:
            try:
                path = _save_adjudication(
                    pair_id, consensus, error_type, adjudicator_id.strip(), rationale
                )
            except ValueError as exc:
                st.error(f"Not recorded: {exc}")
            else:
                st.success(f"Recorded consensus {consensus}/10 for {pair_id} → {path.name}")
                st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _page_setup()

    st.title("⚖️ Legal Meaning Preservation — Annotation Tool")

    corpus = _load_corpus()

    with st.sidebar:
        st.header("Annotator")
        raw_annotator_id = st.text_input(
            "Your annotator id",
            key="annotator_id",
            placeholder="e.g. A01",
            help=(
                "A pseudonymous id. Never your real name. Each id writes to its "
                "own file, so two people annotating on one machine do not "
                "overwrite each other."
            ),
        ).strip()

        if raw_annotator_id:
            try:
                annotator_id = _validate_annotator_id(raw_annotator_id)
            except ValueError as exc:
                st.error(str(exc))
                return
            own = _load_own_scores(annotator_id)
            st.caption(f"**{len(own)}** / {len(corpus)} pairs scored by `{annotator_id}`.")
        else:
            annotator_id = ""
        st.divider()
        st.caption(
            f"Corpus v{_config_fingerprint()} · {len(corpus)} pairs · "
            f"{corpus['language'].nunique()} languages · "
            f"{corpus['legal_domain'].nunique()} domains"
        )

    if not annotator_id:
        st.info("Enter an annotator id in the sidebar to begin.")
        st.markdown(
            """
            ### Before you start

            1. Read **[`src/annotation/guidelines.md`](src/annotation/guidelines.md)**
               in full. Section 2 holds the scale anchors; section 7 covers the
               confusions that cost the most agreement.
            2. Scoring is **blind**. The curator's draft score is never shown by
               this tool — not before you save, and not after. If you have seen a
               draft for a pair, that pair's label is compromised; say so.
            3. A one-sentence note is optional but is what the paper's error
               analysis is built from. Add one wherever there is something to say.
            4. Two annotators score independently, then a **different** person
               adjudicates any pair where you differ by 2 or more points.
            """
        )
        return

    annotate_tab, review_tab, adjudicate_tab = st.tabs(["✍️ Annotate", "📊 Review", "⚖️ Adjudicate"])
    with annotate_tab:
        tab_annotate(corpus, annotator_id)
    with review_tab:
        tab_review(corpus, annotator_id)
    with adjudicate_tab:
        tab_adjudicate(corpus)


if __name__ == "__main__":
    main()
