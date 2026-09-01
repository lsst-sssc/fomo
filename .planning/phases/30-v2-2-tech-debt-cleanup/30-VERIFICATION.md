---
phase: 30-v2-2-tech-debt-cleanup
verified: 2026-09-01T10:03:24Z
status: passed
score: 12/12 must-haves verified (all D-01..D-12 requirement IDs accounted for)
behavior_unverified: 0
overrides_applied: 0
---

# Phase 30: v2.2 Tech-Debt Cleanup Verification Report

**Phase Goal:** Close out the v2.2 deferred items with an accurate record — the
`approval_status` gap in attribution eligibility, the `telescope_class` half of the CSV
re-import guard, the ruff phantom-drift root cause, the unreconciled Nyquist validation
files, and correcting the milestone audit itself.
**Verified:** 2026-09-01T10:03:24Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (mapped to ROADMAP Success Criteria 1-6)

| # | Truth (ROADMAP criterion) | Status | Evidence |
|---|---|---|---|
| 1 | A REJECTED run is never offered for an orphan CalendarEvent or ObservationRecord; APPROVED/PENDING_REVIEW both stay offered; enforced at both eligibility gates (criterion 1) | VERIFIED | `_ATTRIBUTION_INELIGIBLE_APPROVAL_STATUSES` defined once (`campaign_attribution.py:106`), applied via `.exclude(approval_status__in=...)` at both `_eligible_runs_for_event` (:507) and `_eligible_runs_for_record` (:542). Independently re-ran `python manage.py test solsys_code.tests.test_campaign_attribution -v2` — 39/39 OK, including `TestApprovalStatusGate` (8/8: rejected/approved/pending_review controls for both event and record paths, `is_offered_candidate` server-side refusal, confirmed-link survival) |
| 2 | A confirmed attribution survives its run later being rejected (criterion 1) | VERIFIED | `test_confirmed_attribution_survives_its_run_being_rejected` passes; code trace confirms `orphan_calendar_events()` only examines un-linked events, so a rejected-after-confirmation run cannot be unlinked |
| 3 | CSV re-import can no longer replace a non-blank `telescope_class` with a different derived value; every guard firing is named on stderr and counted, not silently `unchanged` (criterion 2) | VERIFIED | `preserve_telescope_class` computed at `import_campaign_csv.py:298`, gates the pop/counter/stderr block at :399-410. Independently re-ran `python manage.py test solsys_code.tests.test_import_campaign_csv -v1` — 77/77 OK, including the 4 new `TestReImportTelescopeClassPreservation` tests and the unedited `test_telescope_class_never_blanked_by_reimport` |
| 4 | Lint/format gates pass under the pinned version; documented command matches enforced version; no reformat (criterion 3) | VERIFIED | `pyproject.toml:42` pins `ruff==0.2.1`, exactly matching `.pre-commit-config.yaml`'s `rev: v0.2.1`. `CLAUDE.md` names `pre-commit run ruff --all-files` / `pre-commit run ruff-format --all-files` in both places it documents the gate (Commands block and Testing bullet); `grep -cE 'ruff (check|format) \.' CLAUDE.md` = 0. Independently re-ran both hooks: both Passed, no files modified |
| 5 | Phases 26/27/27.1/28/29 each have a reconciled VALIDATION.md with a real verdict (criterion 4) | VERIFIED | All 5 files exist; `status: validated` / `nyquist_compliant: true` on every one (independently re-grepped); 27.1's file did not exist before this phase and now does; 27's Per-Task Verification Map rows are all `✅ green`, no lingering `TBD` |
| 6 | `.planning/v2.2-MILESTONE-AUDIT.md` states the true disposition of every tech-debt item, citing where closed (criterion 5) | VERIFIED | Every code-review row (27 IN-02, 27 WR-01, 27.1 WR-09, 27.1 WR-10, 29 reconciler docstrings, 26-DECISION.md header) carries an explicit disposition naming the closing plan or the pre-existing evidence; the WR-09/WR-10 runbook citations were independently re-derived and resolve against the current file (`sed -n` confirms the quoted phrase is present at the cited lines); lint/format section states the `0.15.20` misdiagnosis instead of recommending a reformat; item-total line reconciles (13 items / 6 groupings: 5 closed by Phase 30, 2 already closed, 6 accepted/deferred) |
| 7 | Runbook tells staff a rejected run is never offered; notebook demonstrates it in real executed output (criterion 6) | VERIFIED | `docs/runbooks/telescope_runs_calendar.rst:210` carries the exact required lead sentence. Notebook has 2 code cells referencing `rejected_demo_run`, both with non-empty `outputs` (independently parsed with `json`) |
| 8 | D-11: the record is internally consistent — ROADMAP goal corrected 2026-08-31 and the audit agrees | VERIFIED | `grep -c '2026-08-31' .planning/v2.2-MILESTONE-AUDIT.md` finds the D-11 note; ROADMAP.md Phase 30 section already reflects the corrected six-item scope and matches the audit's account |
| 9 | D-10: `campaign_reconciler.py` docstrings no longer name deleted functions; `26-DECISION.md` header reports Phase 26 complete, naming all 5 plans | VERIFIED | `grep -cE '_project_calendar_event|_calendar_event_title' solsys_code/campaign_reconciler.py` = 0; the one deliberately-kept mention in `reconcile_campaign_runs.py` survives (count 1); `26-DECISION.md`'s header now reads "Status: Complete" and names 26-04/26-05; diff confined to one hunk |
| 10 | All D-01..D-12 requirement IDs are accounted for in a plan's SUMMARY.md | VERIFIED | Cross-referenced PLAN frontmatter `requirements:` against SUMMARY `requirements-completed:`: 30-01→[D-01,D-02,D-03,D-12], 30-02→[D-05,D-06,D-07,D-10], 30-03→[D-04], 30-04→[D-08,D-09,D-11]. Union = D-01 through D-12, no gaps, no duplicates across plans |

**Score:** 12/12 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/campaign_attribution.py` | REJECTED exclusion at both gates | ✓ VERIFIED | Constant + 2 `.exclude()` call sites, docstrings cite 27-REVIEW IN-02/D-01/D-02/D-03 |
| `solsys_code/tests/test_campaign_attribution.py` | `TestApprovalStatusGate` (8 tests) | ✓ VERIFIED | Re-ran: 8/8 pass |
| `docs/runbooks/telescope_runs_calendar.rst` | Attribution paragraph + widened re-import gotcha | ✓ VERIFIED | Both edits present, non-overlapping (sed-checked line ranges) |
| `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` | 5th rejected run, real executed output | ✓ VERIFIED | 2 code cells reference `rejected_demo_run`, both carry outputs |
| `solsys_code/management/commands/import_campaign_csv.py` | Widened `preserve_telescope_class` guard | ✓ VERIFIED | Decision boolean, pop, counter, stderr line, help string all present |
| `solsys_code/tests/test_import_campaign_csv.py` | 4 new tests, old invariant test unedited | ✓ VERIFIED | Re-ran: 77/77 pass |
| `pyproject.toml` | ruff pinned to `0.2.1` | ✓ VERIFIED | Matches `.pre-commit-config.yaml` rev exactly |
| `CLAUDE.md` | pre-commit-mediated lint/format gate, both mentions | ✓ VERIFIED | No bare `ruff check|format .` invocation remains |
| `solsys_code/campaign_reconciler.py` | Dead-symbol docstring repair | ✓ VERIFIED | 0 remaining dead references; 49/49 reconciler+command tests pass |
| `.planning/phases/26-canonical-record-spike/26-DECISION.md` | Header corrected | ✓ VERIFIED | "Complete", all 5 plans named |
| 5× `VALIDATION.md` (26, 27, 27.1, 28, 29) | Reconciled, real verdict | ✓ VERIFIED | All `status: validated` / `nyquist_compliant: true` |
| `.planning/v2.2-MILESTONE-AUDIT.md` | True disposition per item | ✓ VERIFIED | Dispositions present and citations resolve; heading count (12) unchanged |

### Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| `_eligible_runs_for_event`/`_eligible_runs_for_record` | `candidates_for_event`/`candidates_for_record` | shared constant | WIRED — confirmed by test and code trace |
| `is_offered_candidate('event', ...)` | eligibility gates | server-side re-derivation | WIRED — `test_is_offered_candidate_refuses_a_rejected_run` passes |
| `.pre-commit-config.yaml` rev | `pyproject.toml` dev pin | version string equality | WIRED — `0.2.1` == `0.2.1` |
| validate-phase verdict per phase | audit's Nyquist Coverage table | value-for-value match | WIRED — audit table matches the 5 files' front matter |
| `.planning/v2.2-MILESTONE-AUDIT.md` | `/gsd-complete-milestone` | file it reads | Not independently exercisable (no milestone-complete run performed) — content verified directly instead |

### Behavioral Spot-Checks / Regression

| Behavior | Command | Result | Status |
|---|---|---|---|
| Attribution module tests | `python manage.py test solsys_code.tests.test_campaign_attribution -v2` | 39/39 OK | ✓ PASS |
| `TestApprovalStatusGate` isolated | same, filtered to class | 8/8 OK | ✓ PASS |
| CSV import module tests | `python manage.py test solsys_code.tests.test_import_campaign_csv -v1` | 77/77 OK | ✓ PASS |
| Reconciler + retired-command tests | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs -v1` | 49/49 OK | ✓ PASS |
| 9-module downstream regression (attribution + CSV + admin + templates + approval + reconciler) | `python manage.py test solsys_code.tests.test_admin solsys_code.tests.test_attribution_template solsys_code.tests.test_calendar_template solsys_code.tests.test_campaign_approval solsys_code.tests.test_campaign_attribution solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_campaign_run_observation solsys_code.tests.test_import_campaign_csv -v1` | 434/434 OK | ✓ PASS |
| Both ruff hooks | `pre-commit run ruff --all-files && pre-commit run ruff-format --all-files` | both Passed, 0 files modified | ✓ PASS |
| Sphinx build | `python -m sphinx -b html docs docs/_build/html -q` | exit 0, no warning names `telescope_runs_calendar.rst` | ✓ PASS |
| Scope discipline | `git diff --stat` across all Phase 30 commits | only the 19 files named in the four plans' `files_modified` were touched, no repo-wide reformat | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|---|---|---|---|
| D-01 | 30-01 | ✓ SATISFIED | REJECTED excluded, APPROVED/PENDING_REVIEW controls pass |
| D-02 | 30-01 | ✓ SATISFIED | Both gates carry the filter (record gate, previously missed by the original audit, now covered) |
| D-03 | 30-01 | ✓ SATISFIED | One shared constant; `is_offered_candidate` re-derivation test passes |
| D-04 | 30-03 | ✓ SATISFIED | Widened guard verified as a strict superset; 4 new + 1 unedited pre-existing test all pass |
| D-05 | 30-02 | ✓ SATISFIED | No reformat run; both hooks pass clean under pinned version |
| D-06 | 30-02 | ✓ SATISFIED | `pyproject.toml` pin matches pre-commit rev exactly |
| D-07 | 30-02 | ✓ SATISFIED | CLAUDE.md documents pre-commit-mediated gate in both places |
| D-08 | 30-04 | ✓ SATISFIED | All 5 VALIDATION.md files reconciled and `validated`/`compliant: true` |
| D-09 | 30-04 | ✓ SATISFIED | Audit amended with true dispositions and resolving citations |
| D-10 | 30-02 | ✓ SATISFIED | Reconciler docstrings and 26-DECISION.md header both corrected |
| D-11 | 30-04 | ✓ SATISFIED | Roadmap-correction note on record, dated 2026-08-31 |
| D-12 | 30-01 | ✓ SATISFIED | Both paired artifacts (runbook, notebook) extended with real executed output |

No orphaned requirements: `.planning/v2.2-MILESTONE-AUDIT.md` sourced these items directly and Phase 30's `requirements: TBD` note in ROADMAP.md explicitly routes to the D-xx list as the source of truth, which is fully covered above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/management/commands/import_campaign_csv.py` | 309 (`needs_review = site_resolution_failed and not telescope_class`) | Correctness gap: `needs_review`/`site_needs_review` is computed from the **pre-preservation** local `telescope_class` variable, not the value that actually lands in the DB once `preserve_telescope_class` pops the field at line 399. In the tier-3-placeholder-site + preserved-non-blank-telescope_class combination, this writes `site_needs_review=True` onto a row that also gets a placeholder `Observatory` alongside its correctly-preserved `telescope_class` — the "contradictory triple" `models.py`'s D-06 comment says should never happen. Independently confirmed by code trace (matches 30-REVIEW.md WR-01 exactly); no test in the codebase exercises this combination. | ⚠️ Warning (non-blocking) | Does not violate any of this phase's stated must-haves or ROADMAP criterion 2 text (which is scoped to `telescope_class` preservation and reporting, both of which are correct). Confirmed pre-existing: the same local/final mismatch was already reachable via the old blanking-only guard, so this phase's widening exposed rather than introduced it. `27-REVIEW WR-01`'s original scope (per `30-CONTEXT.md` D-04) was explicitly "mirror `preserve_site`'s shape" for `telescope_class`, not a general correctness audit of `site_needs_review`. Recommend a follow-up quick task or todo entry citing `30-REVIEW.md WR-01` and the fix already sketched there (`resulting_telescope_class = existing.telescope_class if preserve_telescope_class else telescope_class`); do not treat as blocking Phase 30's completion. |
| `solsys_code/tests/test_campaign_attribution.py` | — | `is_offered_candidate('record', ...)` has no direct test for the REJECTED-exclusion path (only `'event'` is tested) | ℹ️ Info (non-blocking) | Code trace confirms the `'record'` branch routes through the same changed gate (`candidates_for_record` → `_eligible_runs_for_record`), so there is no evidence of an actual functional gap — only a test-coverage gap. Matches `30-REVIEW.md IN-01`. |

No debt markers (`TBD`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER`) were introduced by this phase; the `TBD` occurrences found in the modified files are the codebase's pre-existing domain term for an unresolved observing window, unrelated to code debt.

### Human Verification Required

None. Every truth above was independently confirmed by re-running the cited automated command or by direct code trace against the current tree — no visual, real-time, or external-service-dependent behavior is in scope for this phase.

### Gaps Summary

No gaps. All six ROADMAP success criteria hold, all twelve D-01..D-12 decisions are implemented and independently re-verified (not merely re-read from SUMMARY.md claims), and every SUMMARY.md self-check claim that was spot-checked (test counts, grep counts, file diffs, hook results) reproduced identically on independent re-run.

One code-review-sourced correctness bug (WR-01's sibling: `site_needs_review` computed from the pre-preservation `telescope_class`) is real, verified by code trace, and worth a follow-up — but it is a pre-existing issue outside this phase's stated must-haves and ROADMAP criterion 2 text, correctly triaged as a Warning (not Critical) by the phase's own code review, and does not block phase completion. Recorded above as a non-blocking Warning finding; a human may choose to open a follow-up quick task citing `30-REVIEW.md WR-01`.

---

_Verified: 2026-09-01T10:03:24Z_
_Verifier: Claude (gsd-verifier)_
