---
phase: 18-uncertain-scheduling-investigation-spike
reviewed: 2026-07-09T00:00:00Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - docs/design/design.rst
  - docs/design/uncertain_scheduling_spike.rst
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues_found
---

# Phase 18: Code Review Report

**Reviewed:** 2026-07-09T00:00:00Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

This is a documentation-only phase: `docs/design/design.rst` gains one toctree line
registering the new durable-summary doc, and `docs/design/uncertain_scheduling_spike.rst`
is a brand-new Sphinx design note. There is no application code in scope, so this review
focuses on Sphinx/RST correctness, internal consistency of the new doc against its own
stated purpose (SCHED-01, per `.planning/REQUIREMENTS.md:12`), consistency against the
full-detail companion `18-DECISION.md`, house-style consistency against the sibling spike
doc (`eso_feasibility_spike.rst`), and PII leakage.

The toctree edit in `design.rst` is correct — the entry name matches the new file's
basename exactly, the file exists, and the build's autosummary log confirms it is picked
up (`design/uncertain_scheduling_spike` reads successfully during a full `sphinx-build`
run). No issues found in `design.rst` itself.

`uncertain_scheduling_spike.rst` is largely faithful to `18-DECISION.md` and to
`.planning/REQUIREMENTS.md`'s SCHED-01 acceptance criterion — I cross-checked every claim
in the "Decisions" table against the decision doc and against the actual code
(`solsys_code/campaign_utils.py`, `solsys_code/solsys_code_observatory/utils.py`,
`solsys_code/models.py`) and found the factual claims about field names, regex helper
names, `Observatory.obscode` max length, and the `to_observatory()` `float(longitude)`
call all check out. However, I found one real RST markup defect (renders differently
from the rest of the document, silently) and one real content-completeness gap where
the durable summary omits half of the answer to a question it explicitly poses,
weakening its stated purpose as a self-contained reference for future milestones.

## Warnings

### WR-01: Inline-literal markup uses single backticks in the "Key finding" section, inconsistent with the rest of the document and with house style

**File:** `docs/design/uncertain_scheduling_spike.rst:44,46,50,52,53`
**Issue:** RST uses double backticks (` `` `) for inline literal/monospace text.
A single backtick pair (`` ` `` ... `` ` ``) is instead the default *interpreted text*
role, which (absent a `default_role` setting in `docs/conf.py` — confirmed there is
none) renders as a `title-reference` (typically italicized/`<cite>`), not as code font.
Every other occurrence of these same identifiers in this document uses correct
double-backtick markup — e.g. line 24 ``` ``TBD`` ```, line 67 ``` ``window_start``/``window_end`` ``DateField`` ```,
line 87 ``` ``rapidfuzz`` to ``pyproject.toml`` ```, line 101 ``` ``Site Code`` ``` — and
the sibling doc `eso_feasibility_spike.rst` uses double backticks exclusively throughout.
Only the "Key finding" paragraphs (lines 42-54) regress to single backticks:
```
`window_start`/`window_end` `DateField` pair            (line 44)
the day-unknown `TBD` marker                             (line 46)
`rapidfuzz` and stdlib `difflib`                          (line 50)
the real messy `Site Code` corpus; add `rapidfuzz` to     (line 52)
`pyproject.toml` explicitly                               (line 53)
```
This is a silent rendering defect (Sphinx build does not warn or error on it — verified
with a full `sphinx-build -b html docs ...` run), so it will not be caught by CI; the
built HTML page will show these five spans in a different style (italic/citation) than
every other code/identifier reference on the same page and than the matching spans two
paragraphs later in the "Decisions" table.
**Fix:** Change all five single-backtick spans in lines 44-53 to double backticks, e.g.:
```rst
**Window schema: confirmed as the already-locked nullable
``window_start``/``window_end`` ``DateField`` pair — no schema change needed.**
Every real cell shape in the sheet (single exact date, full-date range,
compact same-month range, and the day-unknown ``TBD`` marker) maps cleanly
onto this pair.

**Fuzzy-match library: difflib is the primary choice — no new dependency
justified by this live test.** ``rapidfuzz`` and stdlib ``difflib`` produced the
same matches (including the same two false positives) and the same clean
misses on the real messy ``Site Code`` corpus; add ``rapidfuzz`` to
``pyproject.toml`` explicitly only if a future, wider candidate pool
demonstrates a case difflib genuinely misses.
```

### WR-02: "Decisions" table omits the actual finding for half of the SCHED-01 criterion-5 question it poses

**File:** `docs/design/uncertain_scheduling_spike.rst:20-38,90-94`
**Issue:** The "Background" section's fifth bullet (lines 37-38) states the question this
spike had to settle as two parts: "Whether `resolve_site()` correctly resolves real
space-observatory MPC codes, **and** whether `Observatory.obscode` needs widening."
This wording is lifted directly from `REQUIREMENTS.md:12`'s SCHED-01 acceptance
criterion, which likewise poses it as two confirmations.

The corresponding "Decisions" table row (lines 90-94) answers only the second half
("No widening of `Observatory.obscode` ... needed") and is silent on the first half. The
actual finding recorded in `18-DECISION.md` (criterion 5, "Important unexpected real
finding for 250/274/289") is that `resolve_site()` **cannot currently resolve any of the
three standard space-observatory codes** — `MPCObscodeFetcher.to_observatory()` raises
an unguarded `TypeError` (`float() argument must be a string or a real number, not
'NoneType'`) at `solsys_code/solsys_code_observatory/utils.py:81`
(`elong = float(self.obs_data['longitude'])`) because the MPC API returns
`longitude: null` for `"observations_type": "satellite"` records — confirmed against
the live code and live API response, not merely asserted. This is arguably the single
most actionable/surprising finding of the whole criterion-5 investigation, since it is a
real, reproducible bug (not a hypothetical), yet the durable summary's "Decisions" table
never states it — the "Future scope" section (lines 96-106) only obliquely alludes to
"the unrelated `to_observatory()` `TypeError` on satellite-type MPC records" as one item
in a list of things the full decision doc covers, without saying what the practical
consequence is (that `resolve_site()` presently fails for `250`/`274`/`289`).

Because this document explicitly frames itself (lines 10-16) as the durable summary
"written for future milestones to reference **without digging into this findings
record** [`18-DECISION.md`]," a Phase 19/21 reader who trusts that framing and reads only
this table would reasonably (and incorrectly) conclude that `resolve_site()` already
works fine for these three codes, since "no widening needed" is the only verdict given.
**Fix:** Add a sentence to the criterion-5 "Decision" cell (or a short paragraph after
the table) stating the actual resolve_site() finding, e.g.:
```rst
   * - ``resolve_site()`` / obscode widening
     - No widening of ``Observatory.obscode`` (``max_length=4``) needed — confirmed
       against the live field definition; real space-observatory MPC codes (``250``,
       ``274``, ``289``) all fit within 3 characters. Separately, ``resolve_site()``
       cannot *currently* resolve any of the three: ``MPCObscodeFetcher.to_observatory()``
       raises an unguarded ``TypeError`` on the MPC API's ``null`` longitude for
       satellite-type records, an unrelated bug for Phase 19/21 to be aware of (see
       Future scope).
     - None (confirmed as-is)
```

## Info

### IN-01: "Key finding" section highlights only 2 of the 5 SCHED-01 criteria, omitting a pointer to the resolve_site() finding

**File:** `docs/design/uncertain_scheduling_spike.rst:42-54`
**Issue:** The "Key finding" section calls out the window-schema and fuzzy-match-library
verdicts but says nothing about the TBD natural-key collision, the CSV parsing-rule
extension, or the resolve_site()/obscode-widening verdict — all three are relegated to
the "Decisions" table only. This mirrors the sibling `eso_feasibility_spike.rst`'s
pattern of highlighting a single headline finding, so it is not wrong, but given WR-02
above, the resolve_site() bug is arguably the most surprising/actionable result of this
spike and would benefit from a one-line callout here rather than being buried in
"Future scope" prose.
**Fix:** Optional: add a third bolded paragraph noting the resolve_site() TypeError
finding, consistent with how the other two headline findings are presented.

---

_Reviewed: 2026-07-09T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
