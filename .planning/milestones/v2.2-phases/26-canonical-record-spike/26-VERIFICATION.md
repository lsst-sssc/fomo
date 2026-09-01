---
phase: 26-canonical-record-spike
verified: 2026-07-29T05:07:11Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "SPIKE-03: the doc states one canonical reconciler event-key scheme, stable across all four pipeline stages — now settled for BOTH classically-scheduled and queue-scheduled runs via two coexisting key families (RUN:{run_pk}:{date} for classical runs; a bare RUN:{run_pk} whole-window container for queue runs, coexisting with the existing ObservationRecord-derived per-observation events), measured to D-11's evidence bar and locked by an explicit human decision at plan 26-05's blocking checkpoint."
  gaps_remaining: []
  regressions: []
human_verification: []
---

# Phase 26: Canonical-Record Spike Verification Report

**Phase Goal:** Settle the identity, key-scheme, migration and attribution questions milestone questioning deliberately left open — against the real dev-DB rows (`CampaignRun` pk=1, its 11 LCO queue events, the 19 unprojected 3I/ATLAS runs), not against hypotheticals — so that every later phase executes a decision instead of making one.
**Verified:** 2026-07-29T05:07:11Z
**Status:** passed
**Re-verification:** Yes — after gap closure (plans 26-04 and 26-05)

## Phase Nature Note

This remains an investigation-only spike. Plans 26-04 and 26-05 (the gap-closure plans
that ran since the prior verification) changed **zero** `solsys_code/` files —
independently confirmed: `git diff --stat 77e16b5 HEAD -- solsys_code/` is empty, and
`git branch -a` shows no scratch branch. Plan 26-04 built and counted three candidate
queue-run calendar projections against three disposable scratch copies of the dev DB,
all torn down before commit; plan 26-05 is documentation-only (an amended decision doc,
an amended durable design page, and synced `ROADMAP.md`/`REQUIREMENTS.md`). That is
correct, not a gap — the deliverable under test is the settled decision, not shipped
code.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SPIKE-01: decision doc fixes the `source` vocabulary and demonstrates, by executable check, `CampaignRun` pk=1 + its 11 LCO companion rows coexisting with no `IntegrityError` and no constraint change | VERIFIED | Unchanged since prior verification; `solsys_code/` byte-identical to `77e16b5` (regression check: `git diff --stat 77e16b5 HEAD -- solsys_code/` empty) so the prior source cross-check against `models.py:120-160`'s constraint tuples still holds |
| 2 | SPIKE-02: doc maps each adapter's existing calendar-event identity key onto a run (classical, LCO, Gemini, campaign-projection) with file:line construction sites and confidence tags | VERIFIED | Unchanged since prior verification; same regression basis (no `solsys_code/` drift) |
| 3 | SPIKE-03: doc states one canonical reconciler event-key scheme, stable across all four pipeline stages, and explicitly answers the class-wide fan-out question — **for both classically-scheduled and queue-scheduled runs** | **VERIFIED (gap closed)** | `26-DECISION.md`'s `#### Queue-run projection — settled` subsection (line 963) states the locked verdict: a queue-scheduled run gets a bare `RUN:{run_pk}` whole-window container event (no date component, `KEY_SET_STABLE=True` measured under a real window-narrowing edit) coexisting with its real `ObservationRecord`-derived events. Independently re-verified against live source: `sync_lco_observation_calendar.py:186` is exactly `url = facility.get_observation_url(record.observation_id)`, and `_time_window()` (lines 92-121) narrows from `parameters['start']/['end']` to `scheduled_start`/`scheduled_end` — confirming the "narrows/refines as scheduled and observed" claim is accurate, not asserted. `docs/design/canonical_record_spike.rst` states the identical verdict (Key finding, Event key row, Class-wide runs row, Future scope all updated, cross-read against `26-DECISION.md` — no divergence found). D-05's 400-event figure verdict: does not survive (pk=29/30 are QUEUE type, so both take the settled 1-event container form) — stated as a number, not a suspicion. |
| 4 | SPIKE-04: doc states migration + rename checklist naming every integration point, `related_name`-unchanged decision recorded | VERIFIED | Unchanged since prior verification; same regression basis |
| 5 | Decisions are durable and readable outside `.planning/` via a `docs/design/` page | VERIFIED | `docs/design/canonical_record_spike.rst` mirrors the settled verdict; `sphinx-build -M html ./docs <out> -T -E -d <doctrees> -D exclude_patterns=notebooks/*,_build` re-run independently: exit 0, 9 warnings, none naming `canonical_record_spike` (all 9 pre-existing: autoapi import-resolution, two docutils formatting warnings, five nonexistent-notebook toctree warnings — none of which touch this file) |

**Score:** 5/5 truths fully verified. The one prior gap (SPIKE-03's queue-run scope) is closed.

### D-11 (write-strategy deferral) — still correctly not reopened

D-11's adopt-vs-gap-fill write-strategy question remains the single, explicitly-labelled
open item Phase 29 inherits — confirmed unchanged: the measured Adopt/Gap-fill table
(lines 1111-1114), the two-writer-churn code-level finding (`sync_lco_observation_
calendar.py:361` + `calendar_utils._update_or_unchanged()`), and the trigger condition
(v2.3's adapter rewiring) all survive verbatim in `26-DECISION.md`. Only the three
sentences that would otherwise have contradicted the newly-settled queue-projection
verdict were corrected (the `#### D-11` heading parenthetical, the "second question is
now open too" sentence, and the "Consuming phase" line) — verified: none of the
forbidden stale phrases ("is now an **open question**", "and now doubly open", "also
open, per the domain correction", "very likely the same category error one level up")
remain anywhere in `26-DECISION.md`; none of ("two questions are now open rather than
one", "**two** still-open questions", "is likely a symptom of this exact same mix-up")
remain in the `.rst` page.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/26-canonical-record-spike/26-DECISION.md` | New `#### Queue-run projection — settled` subsection; Criterion 3 heading naming both run types; Domain-correction consequences 3/4/5 rewritten from open to settled; D-11 deferral preserved | VERIFIED | All present at the exact headings the plan specified (`### Criterion 3 / SPIKE-03 — the canonical event key (locked for classical and queue runs)...`, `#### Queue-run projection — settled`, `#### D-11 — ...(the queue-run projection question is now settled separately, see above)`) |
| `docs/design/canonical_record_spike.rst` | Mirrors the same verdict; Future scope reduced to one open question | VERIFIED | Confirmed section-by-section against `26-DECISION.md` — no divergence on any point |
| `.planning/ROADMAP.md` | Phase 26 bullet states what was settled; plan count 5/5; Phase 29 success criterion 2 restated for both run types | VERIFIED | Line 109 (bullet), line 139 (`5/5 plans complete`), line 203 (criterion 2: "a queue-scheduled run (site-resolved or class-wide) shows a single whole-window `RUN:{run_pk}` container event...") — matches the settled verdict exactly |
| `.planning/REQUIREMENTS.md` | SPIKE-03 line + traceability row agree (`Complete`); RECON-02/RECON-03 reconciled; SPIKE-01/02/04 byte-unchanged | VERIFIED | Line 16 (SPIKE-03 `[x]`), line 98 (traceability `Complete`), lines 34-35 (RECON-02/03 amended to name run type). `git diff` for `REQUIREMENTS.md` across the two plan commits shows zero hunks touching SPIKE-01/02/04 lines. Coverage block unchanged: `24 total / 24 mapped / 0 unmapped` |
| `.planning/todos/pending/2026-07-27-correct-owned-nights-framing-in-upstream-planning-docs.md` | Follow-up todo filed | VERIFIED | Exists, correct YAML front matter, `## Problem`/`## Solution` sections, names `26-CONTEXT.md`, correctly notes `26-DECISION.md`/`.rst` are already corrected |
| Scratch artifacts (`tmp/`, `local_settings.py`, three scratch `.sqlite3` copies, two probe scripts) | Torn down | VERIFIED | `ls local_settings.py` and `ls tmp` both error (literal absence); `git log --all --oneline -- 'tmp/' 'local_settings.py'` returns nothing |
| `solsys_code/` | Unmodified vs. committed pre-phase state | VERIFIED | `git diff --stat 77e16b5 HEAD -- solsys_code/` empty; no `spike/*` branch |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `26-DECISION.md`'s "narrows/refines as scheduled and observed" claim | `solsys_code/management/commands/sync_lco_observation_calendar.py` | direct source comparison | VERIFIED | Line 186 (`url = facility.get_observation_url(record.observation_id)`) and `_time_window()` lines 92-121 (banner `parameters['start']/['end']` → placed `scheduled_start`/`scheduled_end`) match the citation exactly |
| `26-DECISION.md`'s `campaign_gap.claimed_dates()` over-claim finding | `solsys_code/campaign_gap.py:207-209` | direct source comparison | VERIFIED | Lines 207-209 are exactly the quoted `n_days = (run.window_end - run.window_start).days + 1` / `for i in range(n_days)` / `claimed.add(...)` ground-branch loop |
| `26-DECISION.md`'s Criterion 3 verdict | `docs/design/canonical_record_spike.rst` | restatement, not re-derivation | VERIFIED | Cross-read section-by-section; both state: bare `RUN:{run_pk}` container for queue runs, `RUN:{run_pk}:{date}` for classical runs, coexisting with `ObservationRecord`-derived events, D-05 does not survive, one open question remains (write-strategy) |
| `26-DECISION.md`'s Criterion 3 verdict | `.planning/ROADMAP.md` Phase 29 success criterion 2 / `.planning/REQUIREMENTS.md` SPIKE-03, RECON-02, RECON-03 | upstream wording sync | VERIFIED | All four locations use the literal phrase "queue-scheduled" and state the settled per-run-type semantics; no unqualified pre-correction universal wording remains |
| `docs/design/canonical_record_spike.rst` | `docs/design/design.rst` toctree | grep | VERIFIED | Entry present at line 47; `sphinx-build` resolves it with no orphan warning |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Sphinx build of the durable page succeeds under the exact pre-commit invocation | `sphinx-build -M html ./docs <out> -T -E -d <doctrees> -D exclude_patterns=notebooks/*,_build` | `build succeeded, 9 warnings`; none name `canonical_record_spike` | PASS |
| `sync_lco_observation_calendar.py`'s cited lines match the decision doc's claim | `grep -n` on lines 92-121, 186 | exact match | PASS |
| `campaign_gap.py`'s cited lines match the decision doc's claim | `grep -n` on lines 207-209 | exact match | PASS |
| `pytest` suite (1-test collection by design) passes from the pristine tree | `python -m pytest -q` | `1 passed` | PASS |
| `solsys_code/` has zero diff vs. pre-phase commit | `git diff --stat 77e16b5 HEAD -- solsys_code/` | empty | PASS |
| No scratch branch or scratch artifact survives | `git branch -a`, `ls local_settings.py`, `ls tmp`, `git log --all -- tmp/ local_settings.py` | none found | PASS |
| SPIKE-01/02/04 REQUIREMENTS.md lines byte-unchanged across the gap-closure commits | `git diff <before>..<after> -- REQUIREMENTS.md \| grep SPIKE-01\|02\|04` | no hunks | PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` convention exists in this repository and none is declared in any phase-26 plan.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|--------------|--------|----------|
| SPIKE-01 | 26-02-PLAN.md | `source` vocabulary + constraint coexistence | SATISFIED | Unchanged, regression-confirmed |
| SPIKE-02 | 26-01-PLAN.md | Per-adapter identity key mapping | SATISFIED | Unchanged, regression-confirmed |
| SPIKE-03 | 26-02-PLAN.md, 26-03-PLAN.md, 26-04-PLAN.md, 26-05-PLAN.md | Canonical event-key scheme (both run types) + fan-out answer | **SATISFIED** | Gap closed: measured evidence (plan 26-04) + locked human decision (plan 26-05), independently source-verified |
| SPIKE-04 | 26-01-PLAN.md, 26-03-PLAN.md | Migration + attribution strategy, rename checklist | SATISFIED | Unchanged, regression-confirmed |

No orphaned requirements — all four Phase-26 requirement IDs are claimed by at least one
plan and `REQUIREMENTS.md`'s traceability table names Phase 26, status `Complete`, for
all four, with SPIKE-03's checkbox and traceability row now in agreement (both
"Complete").

### Anti-Patterns Found

None. No unreferenced `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` debt markers in any
of the five files this gap-closure phase modified. The one `TBD` occurrence in the new
todo file's `## Solution` section ("TBD. Options to weigh:") is a deliberate, established
convention for filed-but-undecided follow-up todos in this repo (same pattern used in
`2026-07-27-correct-project-md-stale-phase-25-calendar-event-claim.md`), not a debt
marker in shipped or committed decision content. `ROADMAP.md`'s pre-existing `Plans: TBD`
entries for not-yet-planned phases 27-29 are an established roadmap convention, unrelated
to this phase. No email addresses and no occurrence of "upsert" in any of the five
modified files (grepped clean). Pre-existing repo-wide `ruff`/format drift remains
correctly identified as unrelated to this phase (confirmed: `solsys_code/` has zero diff
against `77e16b5`).

### Minor Process Notes (not gaps)

- `.planning/STATE.md` (`stopped_at`/`Resume file` fields) still reads "Completed
  26-04-PLAN.md" rather than 26-05 — a bookkeeping staleness in orchestrator state, not
  in the phase's deliverable. Does not affect goal achievement.
- `26-DECISION.md`'s header preamble ("Status: In progress. This document is built up
  across plans 26-01, 26-02 and 26-03.") was not updated to mention plans 26-04/26-05.
  Cosmetic; the actual Findings and Recommendation content is current and correct.
- `.planning/ROADMAP.md`'s Phase 26 checkbox was ticked to `[x]` and its Progress-table
  Status cell set to `Complete` (commit `928b1e1`) ahead of this re-verification running
  — plan 26-05's own task 3 explicitly scoped that tick as "re-verification's call, not
  this plan's." This is a process-ordering detail (the tick happened outside the plan's
  own scope, likely by the standard end-of-phase orchestrator step), not a content
  defect; since this verification's independent conclusion is also `passed`, the
  checkbox state is consistent with the actual verdict, not merely premature.

### Human Verification Required

None. All must-haves for this phase resolve to VERIFIED programmatically, including via
independent live-source citation checks that were not present in the prior verification
pass. Nothing here requires interactive/visual confirmation beyond what the phase itself
already completed and recorded (the manual `/calendar/` load from SPIKE-04 criterion
4(c), already verified in the prior pass and unaffected by this gap closure).

### Gaps Summary

None remaining. The one gap the prior verification found — SPIKE-03's "stable across all
four pipeline stages" claim not holding for queue-scheduled runs, including `CampaignRun`
pk=1, the phase goal's own flagship anchor — is closed. Plan 26-04 produced D-11-grade
measured evidence (a run-type inventory of all 31 `CampaignRun` rows, a three-way
span/none/per-night scenario comparison against pk=1's real 15-night window and its real
11 LCO events, idempotency and window-narrowing key-stability tokens, and a rendered-
calendar occurrence measurement) entirely against real dev-DB data on disposable scratch
copies, cleanly torn down. Plan 26-05 turned that evidence into an explicit human decision
at a blocking checkpoint — a bare `RUN:{run_pk}` whole-window container event coexisting
with the run's real `ObservationRecord`-derived events — and propagated the settled
verdict consistently into `26-DECISION.md`, the durable `docs/design/` page, `ROADMAP.md`,
and `REQUIREMENTS.md`, verifying the "already ships" half of the claim against live
`sync_lco_observation_calendar.py` source rather than assuming it. Independent spot-checks
in this verification pass confirmed both of the plan's central source citations
(`sync_lco_observation_calendar.py:186`/`92-121` and `campaign_gap.py:207-209`) match the
file exactly, and confirmed the durable page states the identical verdict to the decision
doc with no divergence. D-11's write-strategy deferral — already judged acceptable by the
prior verification — was correctly left untouched and is now the sole remaining
explicitly-open item Phase 29 inherits. `solsys_code/` remains byte-identical to the
pre-phase commit throughout; all scratch artifacts are torn down and absent from every
git ref; `python -m pytest` and the Sphinx build both pass clean.

Phase 26's goal — settling the identity, key-scheme, migration and attribution questions
against real dev-DB rows so later phases execute decisions instead of making them — is
now achieved in full, including for the run type (`CampaignRun` pk=1, LCO queue) the
phase goal itself names as its anchor evidence.

---

*Verified: 2026-07-29T05:07:11Z*
*Verifier: Claude (gsd-verifier)*
