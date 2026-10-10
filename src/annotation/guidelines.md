# Annotation Guidelines — Legal Meaning Preservation (LMP)

**Project:** Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT
**Task:** score how much of the *legal* meaning of an original statutory sentence survives in a simplified rewrite
**Scale:** integer 1–10
**Languages:** English and Hindi

> These guidelines are the specification the annotation tool implements. Read
> them in full before scoring anything. If a case is not covered here, stop and
> raise it with the project lead rather than inventing a rule — an undocumented
> rule produces labels nobody can reproduce.

---

## 1. What you are judging

You are given a **pair**:

| Field | Meaning |
| --- | --- |
| `original_text` | an authentic sentence from an Indian statute |
| `simplified_text` | a rewrite intended to be easier to read |
| `source` | the statute the original comes from |
| `source_section` | the section number, so you can check the surrounding context |

You output **one integer, 1–10**, answering a single question:

> **If a reader acted on the simplified sentence instead of the original, would
> they be led to the same legal conclusion?**

That is the whole task. It is **not** a question about readability, style,
grammar, or how much you personally like the rewrite. A beautifully written
simplification that drops a proviso scores low. A clumsy one that preserves
every operative element scores high.

### The task is asymmetric

Adding something the original does not say is **at least as serious** as
omitting something. An omission leaves a reader uninformed; an addition
actively misinforms them. Do not give credit for a rewrite simply because it is
"more complete" or "clearer" than the original.

### This is not a similarity judgement

Two sentences can share almost all their words and mean opposite things
(*"the buyer is **not** bound to return the goods"* → *"the buyer is bound to
return the goods"*). Two sentences can share almost no words and mean the same
thing. Judge **legal effect**, not surface overlap.

---

## 2. The 1–10 scale

Score the **worst material defect** in the pair. Do not average defects away,
and do not reward a rewrite for the parts it got right.

| Score | Label | What it means |
| --- | --- | --- |
| **10** | Perfect | Legally indistinguishable. Every operative element present; any rewording is stylistic only. |
| **9** | Excellent | One trivial, non-operative detail changed or dropped (a redundant phrase, a stylistic connector). No reader could reach a different legal conclusion. |
| **8** | Good | Meaning preserved, but the rewrite is noticeably less precise or less formal. Nothing operative is lost. |
| **7** | Acceptable | A minor operative element is softened or blurred — a modal weakened (`shall` → `should`), a threshold's importance downplayed — without creating a wrong conclusion. **Borderline: Preserved.** |
| **6** | Marginal | A minor operative element is dropped or narrowed, or scope is quietly restricted. A careful reader would now be missing something that matters. **Borderline: Partially Preserved.** |
| **5** | Partially preserved | A real operative element is lost: a proviso, a condition, a limb of a list, a stated purpose. The main thrust survives but the rule is now incomplete. |
| **4** | Weakly preserved | A central element is lost, or a significant exception is removed. Acting on the rewrite would lead to a different answer in a realistic case. |
| **3** | Poorly preserved | The rewrite changes what the provision *does*: the trigger condition, the threshold, or the class of persons affected. Only the topic survives. |
| **2** | Barely preserved | The rewrite asserts something close to the opposite of the original, or inverts the direction of an obligation, right or condition. |
| **1** | Not preserved | The rewrite contradicts the original, or states a rule the original expressly denies. Following it would produce the opposite legal outcome. |

### Anchors from this corpus

Read these before your first pass. They calibrate the middle of the scale,
which is where annotators disagree most.

The examples below are drawn from this corpus, but they are deliberately **not
identified by `pair_id`**. Naming the pair would turn this table into a lookup:
an annotator could read a score off it for the very pair in front of them
instead of judging it. Judge every pair on its own terms; if you recognise a
rewrite as one of these anchors, apply the reasoning, not the number.

**10 — Perfect.** *"Where in this Act any reference is made to a reasonable
time, the question what is a reasonable time is a question of fact."*
→ *"Wherever this Act refers to a reasonable time, what counts as a reasonable
time is a question of fact."* Nothing operative changed.

**9 — Excellent.** The three attribution routes for an electronic record (sent
by the originator / by an authorised person / by an automated system) are all
preserved; only wording changed.

**8 — Good.** A compliance-and-certification duty is rephrased into the active
voice. Both certified matters survive; the rewrite is less formal but loses
nothing.

**7 — Acceptable.** *"...shall **immediately** after such suspension or
revocation, surrender the licence..."* → *"...**should** surrender the licence
**promptly**."* Two softeners at once. The duty survives but is no longer
mandatory or time-bound. This is the archetypal 7.

**6 — Marginal.** *"The Controller may, **in writing**, authorise..."* → the
words *in writing* are dropped. Delegation that the Act requires to be
documented may now be oral. Minor element, real consequence.

**5 — Partially preserved.** *"...penalty not exceeding fifteen lakh rupees
**for each such failure**."* → the per-failure multiplier is dropped. A serial
defaulter's exposure becomes unbounded below. The penalty survives; its
escalation does not.

**4 — Weakly preserved.** The proviso guaranteeing an applicant *"a reasonable
opportunity of presenting his case"* is dropped. The power survives, but the
procedural protection — the only one in the section — is gone.

**3 — Poorly preserved.** *"fifteen lakh rupees"* → *"five lakh rupees"*. A
single number, but it is the operative ceiling. Every other element is intact,
which is precisely why a number error is easy to miss.

**2 — Barely preserved.** The second limb of a section — *"but a delivery of
part of the goods, with an intention of severing it from the whole, **does
not** operate as a delivery of the remainder"* — is dropped. Because that limb
is a *negation*, dropping it makes the rewrite assert the proposition the
section expressly denies.

**1 — Not preserved.** *"...he is **not bound** to return them to the seller,
but it is sufficient if he intimates... that he refuses to accept them."* →
*"...he **is bound** to return them to the seller."* The negation is removed
and a positive duty substituted. The rewrite states the exact opposite.

---

## 3. The three-way decision

The `decision` column is derived from your score, not annotated separately. The
thresholds are configuration values that are calibrated on validation data, not
assumed; the defaults are:

| Score | Decision |
| --- | --- |
| 8–10 | **Preserved** |
| 5–7 | **Partially Preserved** |
| 1–4 | **Not Preserved** |

You never type the decision. The tool displays the one your current score maps
to as you pick it, so you can see the consequence without holding the table in
your head. Because the mapping is fixed, a score and its decision can never
disagree — if the decision looks wrong for a pair, the **score** is what you
have got wrong, so revisit Section 2 rather than the table.

---

## 4. Error taxonomy

Record the **single primary** error — the one that most drives your score.
When two are present, choose the one whose removal would raise the score most.

| Label | Use when the rewrite... | Typical home |
| --- | --- | --- |
| `None` | preserves everything operative | 8–10 |
| `Omission` | drops a limb, proviso, factor or phrase that carries meaning | 1–6 |
| `Hallucination` | adds or alters something the original does not say — including a changed number or date | 2–5 |
| `Contradiction` | asserts the opposite of the original | 1–2 |
| `Obligation/Permission` | changes the deontic force: `shall`↔`may`, `must`↔`should`, duty↔discretion | 2–7 |
| `Condition` | drops or alters a condition precedent, trigger, or `unless`/`if` clause | 2–5 |
| `Exception` | drops or alters a proviso, saving clause, or `save as`/`except` provision | 4–7 |
| `Scope` | narrows or widens who or what the provision reaches | 3–7 |
| `Date` | changes a period, deadline, or effective date | 3–5 |
| `Number` | changes a monetary amount, quantity, or threshold | 2–5 |
| `Entity` | changes or removes a named party, office, statute or instrument | 3–5 |
| `Other` | a real defect that none of the above describes — **add a note explaining it** | — |

### Deontic strength, in order

When scoring an `Obligation/Permission` error, use this ordering. Moving down
the list is a weakening; moving up is a strengthening. Both cost score, but a
weakening is usually the more serious defect because it makes a rule
unenforceable.

```
PROHIBITION  >  OBLIGATION  >  RECOMMENDATION  >  PERMISSION  >  DISCRETION
(must not)      (shall/must)   (should/ought)     (may)         (it is open to)
```

### Quantifier strength

```
UNIVERSAL (all, every, each, any)  >  ... >  EXISTENTIAL (some, certain)  >  NONE
```

*"Every Certifying Authority shall..."* → *"A Certifying Authority shall..."*
looks like a stylistic change and is a scope change: a universal duty has
become a particular one.

---

## 5. Procedure

### 5.1 Blind scoring (mandatory)

Score every pair **without seeing the curator's draft score, ever**. The
annotation tool never displays a draft — not before you save a score, and not
after. There is no reveal control and no draft-comparison panel in the tool.

That last part is deliberate and worth stating plainly. It would be easy to
show the draft once your score is committed, as a curiosity. The tool lets you
**revise** a score, so a draft seen after the first save would anchor the
revision exactly as a draft seen before it would anchor the original. A
partially-anchored label is not an independent label.

The drafts remain in the corpus so the curator can measure afterwards how well
they anticipated the annotators (`src/annotation/curator_review.py`). That
report is curator-only, for the reason above: it prints the draft and the human
score side by side.

This is not a formality. The whole point of human annotation is to obtain an
independent judgement. If you have seen a draft for a pair — in an exported
file, in a conversation, in a report — that pair's label is compromised. Say so
and it will be re-annotated by someone else.

### 5.2 Context

Legality frequently lives in the surrounding sub-clauses. Read `source_section`
and consult the full section in `data/raw/sections/` before scoring any pair
where:

* the original begins mid-thought (`Subject to...`, `Provided that...`,
  `Where...`, `Notwithstanding...`);
* the original refers to another section, or to "the foregoing provisions";
* the section numbering in the original is ambiguous.

Do **not** consult the *simplified* text's apparent intent to decide what the
*original* must have meant. Resolve the original's meaning first, then compare.

### 5.3 Two annotators, then adjudication

1. Two annotators score the corpus independently in blind mode.
2. Compute agreement (Section 6). Record the real numbers — never an estimate.
3. Where the two scores differ by **2 or more**, the pair goes to adjudication.
4. The adjudicator sees both scores, both error types, and both notes, and
   records a **consensus** score with a note explaining the resolution. The
   adjudicator must be a **different person** from both annotators; the tool
   refuses an adjudication signed by either of them. An annotator resolving
   their own disagreement removes the third reading the step exists to obtain.
5. Only after adjudication does a pair reach `human_consensus`.

Do not average two disagreeing scores to manufacture agreement. Disagreement is
a finding about the task, and burying it corrupts the evaluation.

Note also what the tool does *not* show you while you are scoring: another
annotator's score for a pair you have not yet judged. The Review tab reports
the corpus-wide agreement coefficients but withholds the per-pair table, and
the per-pair view appears only in the Adjudicate tab. Agreement measured
between annotators who could see each other's work is not agreement.

### 5.4 Record, always

For every pair record:

* `human_lmp_score` — your integer. **Required.**
* `error_type` — the primary defect. **Required**, and `None` is a complete
  answer: it is the label for "nothing operative was lost". Choosing it
  explicitly is what makes "no error" a judgement rather than a blank nobody
  looked at.
* `notes` — **optional**, but this is what the adjudicator and the paper's
  error analysis will read, so add one wherever there is something to say.
  "Drops the reasonable-opportunity proviso" is useful; "meaning changed" is
  not. A note is required when you choose the `Other` error type, because
  nothing else records what "Other" meant.
* `annotator_id` — your pseudonymous identifier (never your real name)
* `annotated_at` — timestamp, written automatically

### 5.5 Do not

* **Do not** let the draft score influence you (§5.1). This includes reading
  it anywhere other than the tool: `data/processed/curator_drafts.csv` pairs
  every `pair_id` with its draft score, and the curator's post-hoc comparison
  report (`reports/curator_draft_review.md`) prints the drafts beside the human
  scores. Both are curator material. Opening either one compromises every pair
  in it, and "I only looked at a few" is not a defensible boundary.
* **Do not** mark a pair `human_consensus` on your own. Consensus requires
  adjudication between annotators.
* **Do not** edit `data/processed/lmp_pairs.csv` by hand. Annotations are
  written through the tool, which stamps the status, the annotator and the
  time. A hand-edited row has no provenance and will be rejected by validation.
* **Do not** leave a pair blank and move on. An unscored pair is a valid state
  (it stays `unannotated`), but it is not a score.
* **Do not** reward a rewrite for being longer, or penalise it for being
  shorter, as such. Length is evidence, not the criterion.

---

## 6. Agreement

`src/annotation/agreement.py` computes, from the real annotation files:

| Statistic | Why |
| --- | --- |
| **Cohen's κ** (quadratic weights) | pairwise agreement on an ordinal scale |
| **Krippendorff's α** (ordinal) | the same, generalised to >2 annotators and to missing data |
| **ICC(2,1)** | absolute agreement on the continuous reading of the scale |
| **% agreement within ±1** | the practical tolerance: is the disagreement material? |

Interpreting κ (Landis & Koch, 1977):

| κ | Interpretation |
| --- | --- |
| < 0.20 | slight |
| 0.21–0.40 | fair |
| 0.41–0.60 | moderate |
| 0.61–0.80 | substantial |
| 0.81–1.00 | almost perfect |

Report the **actual computed value**, whatever it is, and discuss it honestly
in the paper's Limitations section. If agreement is moderate, that is a real
result about the difficulty of the task — it is not something to be improved by
re-scoring until the numbers look better.

The agreement module returns `RESULT_PENDING` rather than a number when there
are fewer than two annotators or too few overlapping pairs. A statistic
computed on three overlapping rows is noise, and reporting it as a result would
be worse than reporting nothing.

---

## 7. Common confusions

**"The simplified version is clearer, so it must be better."**
Clarity is not meaning preservation. A rewrite can be clearer *and* wrong.

**"It only dropped one word."**
Weigh the word, not the token count. *"not"*, *"unless"*, *"only"*,
*"before"*, *"other than"* are single words that carry the entire rule.
Compare the lowest anchor in §2 — where a **single negation** is dropped and
the rewrite ends up asserting the opposite — with the mid-scale anchor where a
dropped two-word phrase (*in writing*) is a real but lesser defect.

**"Both sentences are about the same topic, so meaning is preserved."**
Topic is not meaning. A section about a penalty and a rewrite about a penalty
can still disagree on who pays, how much, and when.

**"The rewrite is ambiguous, so I'll give it the benefit of the doubt."**
Ambiguity introduced by a rewrite is a defect, not a neutral change. If the
original was determinate and the rewrite admits two readings, score the
weaker reading.

**Translating between your language and the sentence's.**
Annotate in the language of the pair. For a Hindi pair, judge the Hindi
sentences directly; do not route through an English paraphrase you construct
yourself, because a defect introduced in your own paraphrase will be scored as
though it were in the corpus.

---

## 8. Provenance note for annotators

Hindi `original_text` in this corpus is **author-rendered**, not machine-read
from an official Hindi PDF: every official Hindi statute PDF obtained for this
project uses a legacy non-Unicode Devanagari font whose glyph mapping is lossy,
so extracted text is corrupt and unusable. Those rows carry
`verbatim_quote: "false"`. See `reports/source_provenance.md` for the full
account and for evidence that the problem is systemic rather than a single bad
file.

This does not change your task — you judge the Hindi sentence you are shown —
but you should know it when writing notes about wording, and it is a stated
limitation of the corpus in the paper.
