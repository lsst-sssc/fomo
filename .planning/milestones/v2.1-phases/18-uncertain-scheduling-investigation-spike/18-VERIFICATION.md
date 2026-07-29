---
phase: 18-uncertain-scheduling-investigation-spike
verified: 2026-07-09T00:00:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 18: Uncertain-Scheduling Investigation Spike Verification Report

**Phase Goal:** Settle the uncertain-scheduling design decisions against the real 3I/ATLAS sheet rows before any implementation lands — this is the phase-time spike the milestone deliberately includes rather than defers, and it gates the schema everything downstream depends on.
**Verified:** 2026-07-09
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

This is an investigation-only phase (mirrors Phase 13's ESO feasibility spike): no application
code ships, so "goal achievement" is verified as documentation content grounded in real
evidence, cross-checked directly against the actual codebase (`solsys_code/campaign_utils.py`,
`solsys_code/solsys_code_observatory/utils.py`, `solsys_code/solsys_code_observatory/models.py`)
rather than trusting SUMMARY.md's narrative.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | (SC1) A decision doc records the final window schema (`window_start`/`window_end` nullable `DateField` pair) validated against real 3I sheet rows | VERIFIED | `18-DECISION.md` Criterion 1 (Recommendation) states the locked schema and maps every real criterion-3 cell shape (single date, ` to ` range, compact range, TBD marker, blank) onto it. `.rst` "Decisions" table row 1 restates identically. |
| 2 | (SC2) The replacement natural key for TBD rows is decided and documented, including how the SQLite/PostgreSQL NULL-uniqueness gap will be closed | VERIFIED | `18-DECISION.md` Criterion 2 Finding shows the real two-JWST-row collision (`Obs. Date = '2025-12-?'`, distinguished only by `Filter(s)/Bandpass`); Recommendation locks folding `contact_person` into the key via a partial/conditional `UniqueConstraint` scoped to `window_start IS NULL`. Exact constraint syntax is explicitly deferred to Phase 19 (appropriate — Phase 19 is the schema-migration phase), but the decision and closure mechanism are documented as required. |
| 3 | (SC3) CSV range/TBD text-parsing rules enumerated from actual cell shapes in the real sheet, one rule per real shape | VERIFIED | `18-DECISION.md` Criterion 3 Finding enumerates 5 distinct real `Obs. Date`/`UT Time Range` shapes (blank, ` to ` range, compact same-month range, `YYYY-MM-?` marker, D-04 range-in-UT-column, D-05 garbage artifact) each with a redacted verbatim example; Recommendation gives one rule per shape, named against the existing `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC` regex family — confirmed these three regex names actually exist in `solsys_code/campaign_utils.py`. |
| 4 | (SC4) Fuzzy-match library choice made with recorded rationale from match-quality testing against real messy site-name input | VERIFIED | `18-DECISION.md` Criterion 4 Finding is a live-scored table (rapidfuzz WRatio/token_sort/token_set vs. difflib) against the real D-09 corpus (`X09`, `N50`, `X07`, `C65`) and the live `Observatory` candidate pool; Recommendation reaches an explicit split verdict (difflib primary, rapidfuzz deferred) citing the specific recorded scores, not asserted independently. `.rst` states the identical verdict. |
| 5 | (SC5) `resolve_site()` confirmed against real space-observatory MPC codes (250/274/289), with a documented obscode-length verdict | VERIFIED (surprising result, clearly documented) | `18-DECISION.md` Criterion 5 confirms `Observatory.obscode.max_length == 4` (verified directly: `models.py:28`) and that 250/274/289 fit — "no widening needed" per the default-answer parenthetical. The investigation additionally uncovered a real, independently-confirmed bug: `resolve_site()` currently does **not** actually resolve any of the three via Tier 2, because `MPCObscodeFetcher.to_observatory()` (`solsys_code/solsys_code_observatory/utils.py:81`, `elong = float(self.obs_data['longitude'])`) raises an unguarded `TypeError` on the MPC API's `longitude: null` for satellite-type records — confirmed present in the actual code, matching the doc's claim exactly. Both `18-DECISION.md` and `docs/design/uncertain_scheduling_spike.rst` state this finding prominently (the `.rst` Decisions table row explicitly says "resolve_site() cannot currently resolve any of the three" — this was the completeness gap WR-02 flagged and the code-fixer closed in commit `ea6ce06`). The literal roadmap wording anticipated resolve_site() would work; the spike's job — settle the question with real evidence — was still fully achieved, and the surprising result is flagged out-of-scope for Phase 19/21 rather than fixed here (correct given this is an investigation-only phase). |
| 6 | Durable summary `docs/design/uncertain_scheduling_spike.rst` exists, states the same fuzzy-library verdict and window-schema decision up front, reachable without digging into `.planning/` | VERIFIED | File exists, follows `eso_feasibility_spike.rst`'s skeleton (`=` title, `-` sections, bold "Key finding" one-liners, `list-table`), wired into `docs/design/design.rst`'s toctree (confirmed: `grep` hit at line 46, and a live `sphinx-build -M html` run picked up `design/uncertain_scheduling_spike` and completed with only 4 pre-existing, unrelated warnings). |

**Score:** 6/6 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` | Findings (criteria 2-5) + Recommendation (all 5 criteria) + Durable-summary pointer | VERIFIED | All sections present and substantive; no `<!-- completed in Plan 02 -->` placeholder remains; no email-address pattern found. |
| `docs/design/uncertain_scheduling_spike.rst` | Durable summary, house `.rst` style | VERIFIED | Exists, correct RST markup (double-backtick fix confirmed applied at lines 44-53), states resolve_site() bug finding (WR-02 fix confirmed applied), wired into `design.rst` toctree, builds cleanly via Sphinx. |
| `fuzzy_match_probe.py` (repo root) | Throwaway, git-excluded, never committed | VERIFIED | File exists on disk (10846 bytes); `git status --short` shows nothing for it (properly excluded); confirmed present in `.git/info/exclude`; never staged in any commit. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| Live probe stdout | Redacted Findings blocks in `18-DECISION.md` | Copied verbatim evidence | VERIFIED | `18-DECISION.md`'s tables/examples (candidate pool list, per-code scores, JWST collision rows) read as genuine captured stdout, not paraphrase — matches the level of specificity (exact scores, exact candidate-pool contents) a live probe run would produce. |
| Contact Person / Email cells | Redacted before write to `18-DECISION.md` | D-01 redaction discipline | VERIFIED | Email-address regex grep against both `18-DECISION.md` and the `.rst` returns zero matches; every evidence block in `18-DECISION.md` carries a "redacted per D-01" note; the JWST-collision block explicitly withholds Contact Person / Email while showing the load-bearing `Filter(s)/Bandpass` distinction. |
| Every `resolve_site()` call in the probe | Rolled-back `transaction.atomic()` | No persisted Observatory rows | VERIFIED | Independently confirmed by reading `fuzzy_match_probe.py` directly (not just trusting SUMMARY.md): every call is wrapped `with transaction.atomic(): ... transaction.set_rollback(True)` and passes `create_placeholder=False` (lines 182-186). |
| Plan 01's D-09 scores | Fuzzy-library / obscode verdicts in Recommendation | Evidence-to-decision traceability | VERIFIED | Criterion 4 Recommendation explicitly cites the specific `X09`→`309` / `C65`→`F65` score values and the "both libraries agree" pattern from the Finding table before reaching its verdict — not asserted independently. |
| `18-DECISION.md` verdict | `.rst` Key-finding one-liner | No divergence between the two docs | VERIFIED | Window-schema and fuzzy-library wording in the `.rst`'s "Key finding" section matches `18-DECISION.md`'s Recommendation section verdicts word-for-word in substance. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|--------------|-------------|-------------|--------|----------|
| SCHED-01 | 18-01, 18-02 | Phase-time investigation spike settling window schema, TBD natural key, CSV parsing rules, fuzzy-match library, and resolve_site()/obscode verdict against real 3I sheet rows | SATISFIED | All 5 sub-criteria verified above (Observable Truths #1-5); `REQUIREMENTS.md:70` marks SCHED-01 "Complete" against Phase 18, and no other requirement ID is orphaned to Phase 18 (`grep -n "Phase 18" .planning/REQUIREMENTS.md` shows only the SCHED-01 status row and a pre-existing obscode-widening note). |

No orphaned requirements found for this phase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` debt markers found in `18-DECISION.md` or `docs/design/uncertain_scheduling_spike.rst` (occurrences of the literal string "TBD" are describing the real CSV data shape, not marking incomplete work) | — | none |

`ruff check .` / `ruff format --check .` findings that exist in the repo today are all in unrelated files (notebooks, `settings.py`) untouched by this phase's `files_modified` — not attributable to Phase 18.

### Code Review Status

`18-REVIEW.md` (deep review of the 2 phase-touched files) found 0 critical, 2 warning, 1 info
finding. `18-REVIEW-FIX.md` confirms both warnings (WR-01 RST markup, WR-02 completeness gap
re: the resolve_site() bug) were fixed in commits `b3d7c75` and `ea6ce06`. Independently
re-verified: the `.rst`'s "Key finding" section now uses double-backtick markup throughout
(lines 44-53), and the Decisions table's `resolve_site()`/obscode row now states the
`TypeError` finding explicitly. IN-01 (info-level, optional) was correctly left unfixed as
out of fix-scope.

### Human Verification Required

None. This phase's deliverables are fully text/documentation artifacts that could be
directly read, cross-referenced against each other, and cross-checked against the actual
source code (`campaign_utils.py`, `solsys_code_observatory/utils.py`, `models.py`) and a
live Sphinx build — no runtime behavior, visual UI, or external-service integration
requires human judgment beyond what was already performed here.

### Gaps Summary

None. All 5 SCHED-01 criteria (roadmap Success Criteria) plus the durable-summary
deliverable are verified against actual file contents and cross-checked against the real
source code, not just SUMMARY.md claims. The one notable nuance — criterion 5's literal
"confirmed to resolve ... correctly" wording turned out to be contradicted by a real,
independently-confirmed bug (`to_observatory()`'s unguarded `float(None)` on satellite-type
MPC records) — does not constitute a gap: the spike's actual job (settle the question with
real evidence) was fully achieved, the surprising result is prominently and consistently
documented in both `18-DECISION.md` and `docs/design/uncertain_scheduling_spike.rst`
(confirmed after the code-review fix), and it is correctly flagged as out-of-scope for
Phase 19/21 rather than silently fixed or silently omitted.

---

*Verified: 2026-07-09*
*Verifier: Claude (gsd-verifier)*
