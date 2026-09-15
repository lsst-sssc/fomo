---
phase: 35-allocation-layer-classical-cutover
verified: 2026-09-15T17:25:47Z
status: gaps_found
score: 146/155 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-01-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-01-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-02-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-02-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-03-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-03-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-04-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-04-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-05-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-05-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-06-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-06-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-07-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-07-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-08-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-08-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-09-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-09-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-10-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-10-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-11-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-11-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-12-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-12-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-13-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-13-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-14-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-14-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-15-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-15-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-CONTEXT.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md"
  - "CLAUDE.md"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/admin.py"
  - "solsys_code/allocation_projector.py"
  - "solsys_code/apps.py"
  - "solsys_code/campaign_reconciler.py"
  - "solsys_code/campaign_utils.py"
  - "solsys_code/campaign_views.py"
  - "solsys_code/management/commands/cutover_classical_allocations.py"
  - "solsys_code/management/commands/load_telescope_runs.py"
  - "solsys_code/management/commands/reconcile_campaign_runs.py"
  - "solsys_code/migrations/0018_campaignrun_night_window_fields.py"
  - "solsys_code/models.py"
  - "solsys_code/observation_projector.py"
  - "solsys_code/telescope_runs.py"
  - "solsys_code/tests/test_allocation_projector.py"
  - "solsys_code/tests/test_allocation_projector_signals.py"
  - "solsys_code/tests/test_campaign_reconciler.py"
  - "solsys_code/tests/test_cutover_classical_allocations.py"
  - "solsys_code/tests/test_load_telescope_runs.py"
  - "solsys_code/tests/test_observation_projector_signals.py"
  - "solsys_code/tests/test_reconcile_campaign_runs.py"
  - "solsys_code/tests/test_telescope_runs.py"
covered_digest: "v1:sha256:c820350e6025e26b2779adf6cf3276814a46201133a09925bd2e9ff9f1d7af54"
behavior_unverified: 0
overrides_applied: 0
flagged_prohibitions: 1
re_verification:
  previous_status: gaps_found
  previous_score: 108/116
  previous_verified: 2026-09-15T15:51:23Z
  gap_closure_plans: ["35-12", "35-13", "35-14", "35-15"]
  gaps_closed:
    - "CR-01 (BLOCKER) — the cutover's identity guard no longer derives write authority from a MISSING `Source line:` marker. `cutover_classical_allocations.py:426-445` is now `if existing_source_line != source_line:` with a two-branch reason string. Independently re-run in this process: `python manage.py test solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard` -> Ran 5 tests, OK (was 4 tests pinning the defect). The test that pinned the blocker was REPLACED (`..._still_converts` -> `..._is_refused`), not left beside a new one, and `test_no_marker_claimant_keeps_run_status_details_and_target_byte_identical` adds the destructive-case assertion the suite lacked, using `NonSiderealTargetFactory` per CLAUDE.md. Reviewer PROBE-P3 confirms the `--dry-run` path refuses identically (`unexplained: 3`, `CommandError`, `run_status` still `planned`, staff note intact, all three events still `url=''`). No third sub-case survives: `source_identifier` carries a `UniqueConstraint` and only two code paths write it, both in the same format."
    - "Runbook cutover guarantees (prior gap 2) — `telescope_runs_calendar.rst:946-968` and `:1511-1521` now state the marker predicate explicitly ('but only WHEN the first group's `CampaignRun` has a stored `Source line:` marker that is recoverable and matches'), name BOTH remedies, and volunteer the honest sentence that the failure mode is reachable through the admin edit the paragraph itself prescribes. Grep over the two passages finds zero hedge words (`usually` / `in normal operation` / `in most cases`), satisfying 35-15 prohibition 2. Zero occurrences of `rewrites no existing` or `left untouched either way` remain in the file."
    - "WR-03 round-2 (third stale `claimed_legacy_urls` contract copy) — `campaign_reconciler.py:600-607` now enumerates all four outcomes and states the exclusion is LOAD-BEARING in real mode, matching `campaign_reconciler.py:750-760` and `allocation_projector.py:595-617`. No fourth copy exists."
    - "Loader dry/real counter double-count on the `existing is not None` arm (prior regression WR-02) — `load_telescope_runs.py:294-326` folds `run_created`/`run_updated`/`run_unchanged` only after the preview reconcile returns; the notebook's executed cell 9 prints `(0, 0, 0, 1)` on both passes. `created + updated + unchanged + skipped == lines processed` now holds on the preview as well as the real pass."
    - "Reconciler demo notebook regenerated by re-execution: 18/18 code cells, 0 null execution counts, 5 occurrences of the corrected `CommandError` wording, 0 of the superseded clause."
    - "IN-01 round-2 (shadowed `existing_run` re-query) — the guard's binding is reused at `:542-548`; the redundant query is gone."
  gaps_remaining:
    - "WR-01 (dry-run / real-run inversion parity) — THIRD consecutive round only partially closed, and this round's fix introduced a NEW false-positive direction. Root cause: `allocation_projector.py:357-375`'s stored-boundary fallback assumes the stored boundary is sun-derived, which is false whenever the now-null sub-night field was previously SET."
    - "Loader preview / real divergence on the CREATE arm (re-filed WR-03) — `load_telescope_runs.py:307-310` predicts `created` from window length and never previews the reconcile the real pass runs, so a brand-new line previews `created: 1` where the real run reports `skipped: 1`."
    - "Operator-facing parity claims overstate the shipped code — the runbook's loader paragraph (`:82-89`) attributes the invariant to a mechanism the create arm does not have, and the loader notebook's committed output prints 'the preview never disagrees with the real run', falsified by PROBE-P5."
  regressions:
    - "WR-01 false positive (NEW this round) — `reconcile_run(dry_run=True)` now RAISES on a night the immediately following real run creates cleanly (PROBE-P1). Before this round the guard returned early for a half-null run, so a preview that aborts a night the real run handles is behaviour 35-13 introduced. Downstream: `reconcile_campaign_runs --dry-run` reports the run under `failed:`, and `load_telescope_runs --dry-run` folds the line into `skipped` — an operator following the runbook's 'always dry-run first' rule is sent to 'correct' data that is already correct."
gaps:
  - truth: "`reconcile_run(run, dry_run=True)` raises the same `ValueError` the immediately following real run raises, on BOTH `_mint_fields()` caller branches — a `--dry-run` preview never hides a condition that makes the real run raise, and never invents one (35-09 truths 4 and the ALLOC-02 edge probe; 35-13 truths 1, 3, 4, 8, 9; 35-13 prohibitions 1 and 2)."
    status: failed
    reason: "WR-01, iteration 6 — fourth iteration of this defect class (NF-10 -> NF-20 -> WR-01 -> WR-01). The half-null shape is now REACHED (the `or` short-circuit became an `and`, correctly), but the boundary the guard substitutes is wrong. `_raise_if_set_window_inverted()` falls back to `existing.start_time`/`existing.end_time` for the null field on the premise, stated twice (docstring `:338-344`, call site `:751-755`), that the stored boundary 'was minted from the same deterministic `sun_event()` for the same site and night'. That premise holds only when the field was ALSO null at mint time. The re-mint branch is entered precisely because the sub-night fields changed, and nulling a previously-set field is one of those changes — after a `2300-EoN` -> `BoN-2230` edit, `existing.start_time` is the old operator value `23:00`, not a sunset. BOTH directions are reproduced by executed probes against a real Django test database."
    artifacts:
      - path: "solsys_code/allocation_projector.py"
        issue: "L357-375 (fallback at :360-369), premise asserted at :338-344 and :751-755, reached from :756. PROBE-P1 (new false positive): La Silla, night 2026-07-09, minted `2300-EoN` then edited to `BoN-2230` — dry run raises `ValueError: ... start=2026-07-09T23:00:00+00:00 >= end=2026-07-09T22:30:00+00:00`; the real run creates `22:06:35 -> 22:30` cleanly. PROBE-P6 (original false negative, verbatim): same site, minted `2100-EoN` then edited to `BoN-2130` — dry run returns `ReconcileResult(created=1, retired=1, ...)` with no error; the real run raises `... start=2026-07-09T22:06:35+00:00 >= end=2026-07-09T21:30:00+00:00`."
      - path: "solsys_code/allocation_projector.py"
        issue: "The docstring at :338-344 and the create-branch comment now state that the CREATE-path half-null case is 'the one shape that remains unchecked'. That is false: the re-mint half-null case whose null field was previously set is neither checked correctly nor left alone — it is checked against a boundary the real run will never use. 35-13 truth 4 and the `verification: backstop` statement (35-13 truth 9, 'the only shape for which the dry-run/real-run parity prohibition is knowingly not upheld') are both falsified by PROBE-P6."
      - path: "solsys_code/tests/test_allocation_projector.py"
        issue: "`test_dry_run_of_a_half_null_remint_inverted_window_also_raises` keeps `night_end_utc` null from mint through edit, so its `existing.end_time` genuinely IS the sunrise — the one sub-shape the fallback is sound for. It passes under the bug. 201 tests green is not evidence for this truth."
    missing:
      - "Stop guessing and narrow the guard to what it can prove: return early when EITHER resolved boundary is unknown (`if start is None or end is None: return`) and delete the `existing.start_time`/`existing.end_time` fallback this round added. This removes the new false positive AND restores parity-by-silence for the false negative; it is a revert-shaped change, not new design. 35-REVIEW.md WR-01 carries the exact body."
      - "If the half-null re-mint case is judged worth previewing at all, do it by recording mint provenance (the event description already carries the dark-window line) rather than inferring it — or accept one `sun_event()` call on the RE-MINT path only, which D-13 does not forbid because that night is about to be deleted and re-minted anyway. Do NOT infer provenance from the stored value again."
      - "Add BOTH PROBE-P1 and PROBE-P6 as regression tests (mint with one shape, edit to the other, assert dry and real agree). The existing half-null test passes under the current bug, which is exactly how this defect survived four rounds."
      - "Correct the docstring at :338-344 and the comment at :751-755 to describe what the guard actually proves, instead of asserting a provenance property the run row does not record."
  - truth: "`load_telescope_runs --dry-run` and the real pass report the SAME decision for the same line — the preview and the real pass are the same decision reported twice, never two different decisions (35-14 truth 7, the ALLOC-04 `unclassified` edge probe re-resolution; 35-14 prohibition 1)."
    status: failed
    reason: "Re-filed WR-03, iteration 6. 35-14's fix covers only the `existing is not None` arm. The CREATE arm (`existing is None`) never calls `reconcile_run()` at all — it predicts `night_created += len(nights)` from the window length and folds `run_created += 1`, so nothing in that arm can fail. The real branch runs `write_and_reconcile_campaign_run()` for the same line, whose reconcile CAN raise (ZoneInfoNotFoundError, `sun_event()`'s ValueError for a blank timezone or polar site, an inverted create-path span), and reports the line under `skipped`."
    artifacts:
      - path: "solsys_code/management/commands/load_telescope_runs.py"
        issue: "L294-326, create arm at :307-310. PROBE-P5 (executed): NTT timezone typo'd, no pre-existing `CampaignRun` — `dry : Done (dry run). lines processed: 1, created: 1, updated: 0, unchanged: 0, skipped: 0` against `real: Done. lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1`; `CampaignRun` rows afterwards: 0. This is WR-02's own sentence ('an operator reading the dry run believes a run will be created when in fact the line will be dropped') surviving in the sibling branch."
      - path: "solsys_code/tests/test_load_telescope_runs.py"
        issue: "The new parity assertion was added to the existing-run variant only. `TestMalformedTimezoneSkipsOneLine.setUpTestData` already seeds the no-existing-run fixture PROBE-P5 uses, and no test runs `--dry-run` over it."
    missing:
      - "EITHER give the create arm the same failure surface (preview against a transient `transaction.atomic()` row with `set_rollback(True)`) OR — recommended, given this loop's history of fixes introducing regressions — narrow the claim: state in the runbook and the notebook that a brand-new line's preview cannot predict a reconcile failure."
      - "Extend `TestMalformedTimezoneSkipsOneLine` with the no-existing-run `--dry-run` variant so whichever contract is chosen is pinned."
  - truth: "No operator-facing sentence states a parity guarantee the shipped code does not hold (35-15 truth 4 and prohibition 1; 35-14 truth 6's notebook artifact)."
    status: failed
    reason: "The third round of this exact prohibition failing. 35-15 correctly repaired the CUTOVER passages, but the LOADER paragraph it added in the same file, and the loader notebook cell committed one commit earlier, both assert the parity the create arm does not have."
    artifacts:
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "L82-89: the literal invariant (`created + updated + unchanged + skipped` equals `lines processed`) is true on both passes, but the stated MECHANISM — 'because the dry run folds its own created/updated/unchanged counter only after the same per-line preview reconcile the real pass performs has returned' — is false for a brand-new line, where no preview reconcile is performed at all. The neighbouring sentence 'A line whose preview reconcile raises is reported under skipped alone on both passes' is vacuous rather than true on that arm."
      - path: "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
        issue: "Code cell 9 (`cells[18]`), committed executed output: `Both passes agree: (0, 0, 0, 1) -- the preview never disagrees with the real run.` That is the strongest claim in the phase's operator-facing documentation and PROBE-P5 falsifies it with the same command, the same site and the same typo'd timezone. The fixture is deliberately chosen to take the `existing is not None` arm."
    missing:
      - "Narrow the notebook's final `print` to the invariant the cell actually demonstrates (the existing-run arm's counter parity), and re-execute the notebook rather than hand-patching it."
      - "Correct the runbook's because-clause at L82-89 to describe the two arms separately, keeping the honest note already present three paragraphs earlier ('for a brand-new run, a first-time dry run predicts night counts from the window length rather than computing a real sun-event time')."
  - truth: "The `duplicate_identity` reason vocabulary states every cause the command can report it for — the operator-facing definition matches the reasons the code actually emits (35-12 truth 3's operator-action contract, NF-25's shared-phrase key link)."
    status: partial
    reason: "WR-04, iteration 6. CR-01 gave `duplicate_identity` a SECOND, structurally different cause (a pre-existing `CampaignRun` claimant whose stored marker is absent or differs) and the three places that DEFINE the reason still describe only the original group-vs-group collision. The remedy paragraphs adjacent to two of them were correctly updated, so an operator who reads on gets the right action — but the label printed verbatim above every claimant-marker stderr line asserts a cause that is false for that line."
    artifacts:
      - path: "solsys_code/management/commands/cutover_classical_allocations.py"
        issue: "L180 `_REASON_LABELS[_DUPLICATE_IDENTITY] = 'a second Source line resolves to the same run identity key as an earlier group'` is prefixed onto BOTH branch strings, including the no-marker one, producing stderr that reads 'a second Source line resolves to the same run identity key as an earlier group: CampaignRun pk=1 already claimed ... with no recoverable Source line: marker' — a self-contradictory line where there is only one group."
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "L933-937 (the passage the rest of the runbook calls 'the full reason vocabulary') and L1498-1500 (the troubleshooting Cause list) both define `duplicate_identity` as 'a second GROUP (a second, distinct Source line: string)'. An operator hitting the CR-01 branch goes looking for a second schedule line that does not exist."
    missing:
      - "State both causes in all three places: `_REASON_LABELS[_DUPLICATE_IDENTITY]`, runbook L933-937, runbook L1498-1500. 35-REVIEW.md WR-04 carries the exact two-cause sentence."
      - "Re-run `pre-commit run sphinx-build --all-files` after the runbook edit."
deferred: []
flagged_prohibitions:
  - statement: "The cutover must NOT silently mutate a `CampaignRun` it did not create in this invocation (35-12 prohibition 2, carried forward from 35-08)."
    verification: judgment
    status: unverified
    flagged: true
    finding: "WR-02 (new, iteration 6). The guard's SURVIVING permissive branch — a claimant whose stored `Source line:` MATCHES — is still find-and-updated, so a `run_status` (or `observation_details`, or `target`) edit a staff member made after the import is reverted to the schedule line's value with exit 0 and no report beyond `runs updated: 1`. Reproduced (PROBE-P4): claimant with a matching marker and `run_status` set to CANCELLED by staff -> `Done. candidates: 3, groups: 1, runs created: 0, updated: 1, unchanged: 0, events re-keyed: 3, unexplained: 0`, `run_status after: planned`. This round's own CR-01 remedy text (`:436-443`, runbook `:959-968`) instructs the operator to restore the marker and re-run, which is exactly the action that converts a refused (safe) claimant into a matching (silently-overwritten) one."
    verifier_judgment: "NOT must-have-blocking for ALLOC-04 or ALLOC-05, on three grounds. (1) It does not breach any ROADMAP success criterion: the calendar's final state is still one event per night with no duplicate and no orphan; nothing is deleted; no event is re-keyed onto a run the command cannot prove owns it. (2) The literal prohibition text is in direct tension with 35-12 truth 5, which REQUIRES that a claimant with a matching marker still converts (the benign cutover-after-import ordering). The prohibition's operative meaning, as CR-01 resolved it, is 'must not mutate a run it cannot PROVE it owns' — and provenance is exactly what the matching marker establishes. (3) The behaviour is the project's already-accepted file-authoritative semantics: `load_telescope_runs` re-applies the same `fields` dict on every re-import, and the runbook devotes a whole 'Re-import gotcha' section (L734-790) to this class for `import_campaign_csv`, naming the two fields that are deliberately preserved. So this is find-or-update working as designed on a provable row, not a silent destruction of an unprovable one. What IS genuinely missing is the caveat: the cutover has no equivalent gotcha note, no test pins the intended outcome, and the remedy text routes an operator into it without saying so. Treat as a documented residual risk requiring one runbook sentence and one test, NOT a code change and NOT a blocker. D-18 ('an event the command cannot explain is left untouched and reported') does not reach this case — the command CAN explain this row."
    human_decision_requested: "Confirm the WARNING severity above, or overrule it in favour of the reviewer's narrowing fix (compare line-derived fields against the claimant and report a `_RUN_DRIFT` divergence instead of overwriting). Overruling costs a new reason code, new operator vocabulary and a new refusal path in a command that has now had two rounds of refusal-path regressions."
advisory:
  - finding: "IN-01 — no executed notebook cell covers the branch CR-01 actually inverted. `reconcile_campaign_runs_demo.ipynb`'s `duplicate_identity` demo uses two groups with genuinely DIFFERING markers (the branch CR-01 left unchanged); the committed notebook would look identical if the predicate were reverted."
    category: other
    reason: "35-15 truth 5 requires only regeneration plus the corrected `CommandError` text, both of which are present (18/18 cells, 0 nulls, 5 occurrences). The CLAUDE.md paired-docs rule's 'exercise the new behavior' clause is met at the level of the corrected operator text but not at the level of an executed demonstration. Fix: one cell seeding a claimant with a non-marker `observation_details`, using the fixture `test_no_marker_claimant_keeps_run_status_details_and_target_byte_identical` already builds. Cheap; fold into any further doc round."
    evidence_status: "static observation; notebook inspected programmatically"
  - finding: "IN-02 — `cutover_classical_allocations.py:671` still hardcodes `url__startswith='ALLOC:'` while `allocation_projector.ALLOC_URL_NAMESPACE` exists and this module already imports four names from it. Carried forward unchanged from iteration 5."
    category: other
    reason: "Latent: a namespace rename would silently make the final summary count zero. Fix: import the constant and use it."
    evidence_status: "none provided — latent, no current failure"
  - finding: "IN-03 — `seen_keys[key] = source_line` (`:520`) is still claimed before the group's `try: with transaction.atomic():` (`:540-664`). A group whose transaction rolls back wholesale keeps the key claimed, so a sibling group sharing it is reported under `duplicate_identity` naming a line that converted nothing. Carried forward unchanged."
    category: other
    reason: "Fix: move the assignment to just after the `events_rekeyed += group_rekeyed` fold at `:647-650`."
    evidence_status: "none provided — reasoned from control flow, no probe run"
  - finding: "IN-04 — the loader notebook's new parity cell (`cells[18]`) writes `ntt.timezone = 'America/Santigo'` to the shared developer database and restores it with a plain statement, not a `try/finally`. An interruption between the two writes leaves obscode 809 with a typo'd timezone that every later cell and every other notebook then resolves against. 35-11's skip-path cell has the same shape."
    category: other
    reason: "Verified by reading the cell source: the restore is unguarded and the asserts follow it. Fix: `try/finally`, or a deliberately-rolled-back `transaction.atomic()` as `project_observation_calendar_demo.ipynb`'s attribution cell already uses."
    evidence_status: "static observation; notebook cell source inspected"
  - finding: "WR-04 round-2 — `observation_projector.py:647-658`'s savepoint-less swallowed `campaign_run_links` lookup error. Deliberately deferred as advisory by this round's plan set and not re-verified by iteration 6."
    category: architectural
    reason: "Unchanged since 24875bf, which predates both prior verification timestamps; the file was not touched by either gap-closure round. The exposure is PostgreSQL-specific and was not reproducible on this project's SQLite backend. Carried forward unresolved."
    evidence_status: "probe executed in a prior round, no failure reproduced on SQLite"
behavior_unverified_items: []
coincidental_reliance_items:
  - truth: "The cutover's identity guard refuses what it cannot prove it owns (35-12 truths 1-2, `TestDatabaseScopedIdentityGuard`, 5 tests, OK)."
    reason: fixture-only
    harden: "The guard is now SAFE under the fixture-controlled precondition it previously depended on — absence of a marker denies rather than grants — so the prior pass's coincidental-reliance flag is substantially discharged. What remains is the mirror image: the PERMISSIVE branch's correctness still depends on `observation_details` being trustworthy when it happens to match, and that field is writable from the Django admin (`admin.py:165`), from `import_campaign_csv.py:321` and from `campaign_forms.py:65`. Nothing in production guarantees a matching marker was written by this pipeline rather than typed by a person. Promote the provenance out of free text: store the identity provenance in a field the operator cannot overwrite (or mark it readonly for classical rows), which would also dissolve WR-02's remedy-text trap."
---

# Phase 35: Allocation Layer & Classical Cutover — Verification Report (third pass)

**Phase Goal:** An allocation — classical schedule line, approved submission, TBD/range run, campaign or no campaign — draws its own sunset→sunrise intent nights and hands each night over the moment a real observation links to it; `load_telescope_runs` writes allocations instead of calendar events.
**Verified:** 2026-09-15T17:25:47Z
**Status:** gaps_found
**Re-verification:** Yes — third pass, after the second gap-closure round (plans 35-12 … 35-15)

## Headline

**The BLOCKER is closed and the phase goal is achieved. Every open finding is about preview fidelity and operator-facing wording, not about what the calendar ends up containing.**

All five ROADMAP success criteria verify. CR-01 — the one finding that could corrupt data — is genuinely fixed, with the defect-pinning test inverted rather than deleted and a destructive-case regression added; I re-ran that class in this process (5 tests, OK) and the whole phase test surface once (201 tests, OK). What remains is four findings in two root causes, all of which make a `--dry-run` preview or a sentence of documentation say something the real run does not do. None of them can produce a wrong event, a duplicate or an orphan.

That distinction drives the recommendation in the Gaps Summary: the honest fix for most of what is left is to **stop over-claiming parity**, not to keep engineering it. This loop has now spent four rounds trying to make a preview match a real run, and each round's fix has been narrower than the sentence written about it.

## Goal Achievement

### ROADMAP Success Criteria

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Allocation with resolved site + awarded window shows one sunset→sunrise event per window night; queue/class-wide/satellite keeps a single container | ✓ VERIFIED | `allocation_projector.py` dispatch intact (`ALLOC_URL_NAMESPACE`, `allocation_night_url()`, `reproject_allocation_if_dispatched()`); 201 tests across the five phase modules pass in this process; untouched by round 2 except the guard helper |
| 2 | Allocation nights follow the site-local observing night (Chilean + Australian sites) | ✓ VERIFIED | `_night_span_utc()` / `night_bounds()` unchanged; `test_allocation_projector` green. The WR-01 defect is in the dry-run PREVIEW of an inverted span, not in which night a boundary lands on |
| 3 | Linking an `ObservationRecord` removes that night's allocation event, leaves the observation's own event untouched; unlinking restores it | ✓ VERIFIED | `test_allocation_projector_signals` green; `reconcile_campaign_runs_demo.ipynb` re-executed end to end this round (18/18 cells, 0 null execution counts), re-demonstrating the handoff rather than carrying it on trust |
| 4 | `load_telescope_runs` produces the same per-night calendar as before, by way of an allocation record, and re-running changes nothing | ✓ VERIFIED | The command imports no calendar writer (`grep` over its imports: only `reconcile_run`, `write_and_reconcile_campaign_run`, models and `telescope_runs` helpers); `test_load_telescope_runs` green. The create-arm defect is in `--dry-run` reporting only — the real import path is correct |
| 5 | After the cutover, one event per night: no duplicate, no orphan left behind | ✓ VERIFIED (was FAILED) | CR-01's closure is what turns this. The silent merge that re-keyed three events onto a run the command could not prove it owned is gone: `TestDatabaseScopedIdentityGuard` 5/5 OK, PROBE-P3 confirms refusal on the `--dry-run` path too, and the refused events stay `url=''` with no `CalendarEventMeta` row |

**Roadmap contract: 5/5.**

### Observable Truths — round-2 plan must-haves

Only the round-2 plans are itemised; plans 35-01 … 35-11 were verified in the prior two passes and re-checked for regression (201 tests green, no source file outside the ten round-2 files changed).

| # | Truth (source) | Status | Evidence |
|---|---|---|---|
| 1 | No-marker claimant reported under `duplicate_identity`, non-zero exit, never find-and-updated (35-12 t1) | ✓ VERIFIED | `cutover_classical_allocations.py:426-445` is `if existing_source_line != source_line:`; `test_..._is_refused` asserts `CommandError`, `unexplained (duplicate_identity): 3`, events still `url=''` |
| 2 | After a refusal the claimant's `run_status`, `observation_details`, `target` are byte-identical, asserted field by field (35-12 t2) | ✓ VERIFIED | `test_no_marker_claimant_keeps_run_status_details_and_target_byte_identical` captures all three before, refreshes from DB, asserts each; uses `NonSiderealTargetFactory` per CLAUDE.md |
| 3 | The reason text branches on whether a marker was recovered; both name an action the operator can take (35-12 t3) | ✓ VERIFIED | Two-branch `reason` at `:427-444`; both end in 'then re-run'. (The shared `_REASON_LABELS` prefix is inaccurate for the new branch — see gap 4, WR-04) |
| 4 | The ALLOC-01 `empty` case is asserted as a REFUSAL, replacing rather than sitting beside the case that asserted conversion (35-12 t4) | ✓ VERIFIED | `..._still_converts` (no-marker) is gone; `..._is_refused` is in its place. Class grew 4 → 5 tests, the fifth being the destructive regression |
| 5 | A claimant whose stored marker MATCHES still converts normally (35-12 t5) | ✓ VERIFIED | `test_pre_existing_claimant_with_same_source_line_still_converts` unchanged and green. (This is also the branch WR-02 concerns — see flagged prohibition) |
| 6 | No in-module guarantee asserts the no-rewrite property without naming the marker precondition (35-12 t6) | ✓ VERIFIED | Module docstring, `CommandError` and runbook all now read '… updates an existing `CampaignRun` only when that run's stored `Source line:` matches the line being converted'; zero occurrences of `rewrites no existing` remain anywhere |
| 7-10 | Edge probes ALLOC-01 `empty` / `adjacency` / `ordering`, ALLOC-05 `unclassified` re-resolved (35-12 t7-t10) | ✓ VERIFIED | Each maps onto one of the five passing guard tests plus PROBE-P3's `--dry-run` confirmation |
| 11 | The command still never deletes a `CalendarEvent` and still exits non-zero on any unexplained event (35-12 t11) | ✓ VERIFIED | `CommandError` path exercised by both new tests; no delete call on the refusal path |
| 12 | Half-null dry run raises the SAME `ValueError` the real run raises (35-13 t1) | ✗ FAILED | PROBE-P6: dry clean, real raises. PROBE-P1: dry raises, real clean. See gap 1 |
| 13 | Guard short-circuits only when BOTH sub-night fields are null (35-13 t2) | ✓ VERIFIED | `allocation_projector.py:357` is now `and`, matching `_span_needs_remint():378` |
| 14 | Missing boundary supplied from the stored event, 'which was minted from the same deterministic `sun_event()`' (35-13 t3) | ✗ FAILED | The fallback exists and adds no astropy call, but its stated provenance premise is false whenever the null field was previously SET — the mechanism of both probes |
| 15 | The CREATE-path half-null case is the ONLY shape the guard cannot check, and the docstrings say so (35-13 t4) | ✗ FAILED | The re-mint half-null-after-edit shape is also uncheckable, and is checked WRONGLY rather than skipped; the docstring claims the opposite |
| 16 | Half-null twin regression test exists (35-13 t5) | ✓ VERIFIED | `test_dry_run_of_a_half_null_remint_inverted_window_also_raises` exists and passes — but only pins the one sub-shape the fallback is sound for |
| 17 | Every `claimed_legacy_urls` contract copy states the same current meaning (35-13 t6) | ✓ VERIFIED | Read `campaign_reconciler.py:596-608`: four outcomes, LOAD-BEARING clause, NF-09/NF-17/NF-22/WR-03 cited. Siblings at `:755` and `allocation_projector.py:606` agree; no fourth copy |
| 18 | That third copy enumerates all four outcomes and the real-mode load-bearing property (35-13 t7) | ✓ VERIFIED | Same read |
| 19 | Edge probe ALLOC-02 `unclassified` re-resolved — inversion surfaced identically on both branches (35-13 t8) | ✗ FAILED | PROBE-P1/P6 |
| 20 | [backstop] The create-path half-null case is the only shape where dry/real parity is knowingly not upheld (35-13 t9) | ✗ FAILED | Explicit counter-evidence (PROBE-P6). Per the backstop rule this truth needed positive evidence; it has negative evidence instead |
| 21 | `sun_event()` still called only on create/re-mint; no new astropy call (35-13 t10) | ✓ VERIFIED | The fallback reads stored fields; no new call site |
| 22 | `ALLOC:` / `RUN:{pk}` namespace ownership unchanged (35-13 t11) | ✓ VERIFIED | `ALLOC_URL_NAMESPACE` defined and used in the projector only |
| 23 | Dry and real report the same `(created, updated, unchanged, skipped)` tuple over PROBE-B's fixture (35-14 t1) | ✓ VERIFIED | Notebook cell 9 executed output: both passes `(0, 0, 0, 1)`; new parity test in `TestMalformedTimezoneSkipsOneLine` |
| 24 | `created + updated + unchanged + skipped == lines processed` in both modes (35-14 t2) | ✓ VERIFIED | Holds on both arms — the create arm's divergence is in WHICH bucket, not in the sum |
| 25 | The dry-run branch folds counters only after both calls return (35-14 t3) | ✓ VERIFIED | `load_telescope_runs.py:312-326`, fold moved below the reconcile |
| 26 | `TestMalformedTimezoneSkipsOneLine` asserts dry/real parity (35-14 t4) | ✓ VERIFIED | Present for the existing-run variant |
| 27 | NF-21 not regressed — dedicated `except ZoneInfoNotFoundError` still precedes `(ValueError, Observatory.DoesNotExist)` (35-14 t5) | ✓ VERIFIED | Read at `:345-360`; clause order intact with the explanatory comment |
| 28 | Loader notebook demonstrates parity with real executed output, no null execution counts (35-14 t6) | ✓ VERIFIED | 16/16 code cells, 0 nulls, real stderr/stdout in the output. (Its printed CONCLUSION overclaims — gap 3) |
| 29 | Edge probe ALLOC-04 `unclassified` — preview and real are the same decision reported twice (35-14 t7) | ✗ FAILED | PROBE-P5: `created: 1` vs `skipped: 1`. See gap 2 |
| 30 | One bad line still never aborts the import; the command still writes allocations only (35-14 t8-t9) | ✓ VERIFIED | Import list carries no calendar writer; per-line handlers intact |
| 31 | Every runbook sentence about the cutover's `duplicate_identity` guarantee states the marker precondition (35-15 t1) | ✓ VERIFIED | `:946-968` and `:1511-1521` read in full; both branch on the marker and name both remedies |
| 32 | No passage hedges into 'usually' / 'in normal operation' (35-15 t2) | ✓ VERIFIED | Grep over both passages: zero hits |
| 33 | The remedy covers both branches the code reports (35-15 t3) | ✓ VERIFIED | 'when the two Source lines differ … when no marker is recoverable at all …' |
| 34 | The runbook's loader section states the dry/real parity invariant 35-14 restores (35-15 t4) | ✗ FAILED | The stated mechanism is false for the create arm; the notebook's claim is false outright. See gap 3 |
| 35 | Reconciler notebook regenerated by re-execution, non-null counts, corrected `CommandError` text (35-15 t5) | ✓ VERIFIED | 18/18 cells, 0 nulls, 5 occurrences of the corrected wording, 0 of the superseded clause |
| 36 | The regenerated notebook re-executes the allocation/observation handoff cells end to end (35-15 t6) | ✓ VERIFIED | No null execution counts anywhere in the file |
| 37 | Edge probe ALLOC-03 `unclassified` — handoff untouched this round and re-demonstrated (35-15 t7) | ✓ VERIFIED | No round-2 commit touches the handoff path; notebook re-executed |
| 38 | `pre-commit run sphinx-build --all-files` still passes (35-15 t8) | ✓ VERIFIED | Executed by the iteration-6 reviewer: Passed (with ruff and ruff-format) |

**Score:** 146/155 must-haves verified (150 plan truths + 5 ROADMAP success criteria; 9 failed, 0 behavior-unverified, 1 prohibition flagged for human decision).

Seven of the nine failures share a single root cause (the half-null stored-boundary fallback); the remaining two share a second (the create-arm preview). There are two defects here, not nine.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/management/commands/cutover_classical_allocations.py` | Identity guard refuses unprovable claimants | ✓ VERIFIED | 691 lines; `!=` predicate at `:427`; two-branch reason; guard binding reused at the write site |
| `solsys_code/tests/test_cutover_classical_allocations.py` | Defect-pinning test replaced + destructive regression added | ✓ VERIFIED | 5-test guard class, re-run in this process: OK |
| `solsys_code/allocation_projector.py` | Shared inversion guard reaching the half-null shape | ⚠️ HOLLOW | 996 lines; the guard is reached and wired, but the value it substitutes is not the one the real run uses — present and wired, wrong data |
| `solsys_code/tests/test_allocation_projector.py` | Half-null regression coverage | ⚠️ PARTIAL | Covers the one sound sub-shape; both probe shapes uncovered |
| `solsys_code/management/commands/load_telescope_runs.py` | Counter fold after the preview reconcile | ⚠️ PARTIAL | Fixed on the existing-run arm; create arm has no preview reconcile at all |
| `solsys_code/tests/test_load_telescope_runs.py` | Dry/real parity assertion | ⚠️ PARTIAL | Existing-run variant only |
| `solsys_code/campaign_reconciler.py` | Third `claimed_legacy_urls` contract copy corrected | ✓ VERIFIED | `:596-608` |
| `docs/runbooks/telescope_runs_calendar.rst` | Cutover guarantees narrowed to the shipped predicate | ⚠️ PARTIAL | Cutover section correct; loader paragraph and reason vocabulary are not |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Regenerated against the corrected command output | ✓ VERIFIED | 18/18 executed |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | Parity demonstrated with real output | ⚠️ PARTIAL | Genuinely executed (16/16), but its printed conclusion overstates |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `_extract_source_line()` result | the guard's write authorisation | `None` on the refusing side | ✓ WIRED | The trust boundary the prior pass marked 'WIRED, WRONG TRUST BOUNDARY' is now correct: a parse that found nothing can only deny |
| `CampaignRunAdmin`'s editable `observation_details` | the cutover's identity decision | `_extract_source_line()` | ⚠️ WIRED, ASYMMETRIC | The field can now only DENY a conversion, never grant one — except on the matching branch, where a hand-typed marker still grants a full field overwrite (see flagged prohibition) |
| `_raise_if_set_window_inverted(run, night, existing)` | re-mint branch `if dry_run:` short-circuit | shared guard | ⚠️ WIRED, WRONG VALUE | The link is complete; the substituted boundary is not the one the real run pairs with |
| `preview_campaign_run_action()` result | run-level counters | fold after the preview reconcile | ⚠️ WIRED, ONE ARM | Existing-run arm only |
| Cutover `duplicate_identity` reason strings | runbook remedy passages → notebook output | shared operator phrase | ✓ WIRED | NF-25's shared phrase survives 35-12's rewrite; both remedies appear in code, runbook and the re-executed notebook |
| `docs/index.rst:24` toctree | `runbooks/telescope_runs_calendar` | Sphinx | ✓ WIRED | `sphinx-build` passed in the reviewer's executed run |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Produces real data | Status |
|---|---|---|---|---|
| `cutover_classical_allocations` summary | `runs updated`, `events re-keyed`, `unexplained` | live DB queries over `CalendarEvent` / `CampaignRun` | Yes | ✓ FLOWING |
| `_raise_if_set_window_inverted` | `start` / `end` on a half-null run | half from `_time_of_day_to_datetime()`, half from the STORED event | Partly — the stored half is a historical operator value, not a sun event | ⚠️ STATIC (stale stored value stands in for a computed one) |
| `load_telescope_runs --dry-run` night counters, create arm | `night_created` | `len(nights)` — a literal window length, not a reconcile | No | ⚠️ STATIC (documented in the runbook, but the parity claim built on top of it is not) |
| Loader notebook cell 9 output | both summary lines | two real `call_command()` invocations | Yes | ✓ FLOWING (the numbers are real; the sentence under them is not) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| CR-01 refusal + byte-identity | `python manage.py test solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard` | `Ran 5 tests … OK` | ✓ PASS |
| No regression across the phase surface | `python manage.py test solsys_code.tests.{test_allocation_projector,test_campaign_reconciler,test_cutover_classical_allocations,test_load_telescope_runs,test_allocation_projector_signals}` (run once) | `Ran 201 tests in 103.961s … OK` | ✓ PASS |
| Reconciler notebook fully executed with corrected text | JSON inspection | 18 code cells, 0 nulls, 5× corrected `CommandError`, 0× superseded clause | ✓ PASS |
| Loader notebook parity claim | JSON inspection of `cells[18]` output | `Both passes agree: (0, 0, 0, 1) -- the preview never disagrees with the real run.` | ✗ FAIL (claim falsified by PROBE-P5) |
| Loader writes allocations only | import-list grep | no calendar-writer import | ✓ PASS |
| Guard null-field convention | direct read | `:357` `and`, matching `:378` | ✓ PASS |
| Half-null fallback provenance | direct read of `:360-369` + `:338-344` | fallback reads stored boundary; premise stated, not enforced | ✗ FAIL |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention exists in this repository. The probe evidence for this pass is the iteration-6 reviewer's executed probe module (written under `solsys_code/tests/`, run against a real Django test database, then deleted; `git status --short` clean of source changes). Per the task's instruction those reproductions (PROBE-P1 … P6) are treated as verified facts. I independently re-ran the CR-01 test class and the five-module suite in my own process.

| Probe | Result | Status |
|---|---|---|
| PROBE-P3 (CR-01 `--dry-run`) | `unexplained: 3`, `CommandError`, run byte-identical | ✓ PASS (guard refuses) |
| PROBE-P2 (half-null, stored boundary genuinely sun-derived) | dry and real raise identically | ✓ PASS |
| PROBE-P1 (half-null, previously-set field nulled) | dry RAISES, real creates cleanly | ✗ FAIL (new false positive) |
| PROBE-P6 (mirror edit sequence) | dry clean, real RAISES | ✗ FAIL (original false negative) |
| PROBE-P4 (matching-marker claimant, staff `run_status` edit) | exit 0, `updated: 1`, `run_status` cancelled → planned | ⚠️ FLAGGED (see flagged prohibition) |
| PROBE-P5 (brand-new line, typo'd timezone) | dry `created: 1` vs real `skipped: 1` | ✗ FAIL |

### Requirements Coverage

| Requirement | Source plans | Description | Status | Evidence |
|---|---|---|---|---|
| ALLOC-01 | 35-01, 35-13, 35-15 | Per-night events for resolved classical/awarded allocations; container for queue/class-wide/satellite | ✓ SATISFIED | SC-1; dispatch untouched; 201 tests green. The WR-01 defect affects only the preview of an inverted span |
| ALLOC-02 | 35-13, 35-15 | Nights keyed by the site-local observing night (Chilean + Australian) | ✓ SATISFIED | SC-2; `test_allocation_projector` green. Note the ALLOC-02 edge probe's dry/real parity clause is FAILED (gap 1) — the requirement's own substance (which calendar day a night lands on) is unaffected |
| ALLOC-03 | 35-15 | Handoff on link, restore on unlink, observation's own event untouched | ✓ SATISFIED | SC-3; signals suite green; notebook re-executed this round |
| ALLOC-04 | 35-12, 35-14, 35-15 | `load_telescope_runs` creates/updates a campaign-less `CampaignRun` with a collision-safe `source_identifier`; same per-night events, idempotent | ✓ SATISFIED (with a preview caveat) | SC-4. The real import path is correct and idempotent; the `--dry-run` create arm misreports (gap 2) — a preview defect, not an ingest defect |
| ALLOC-05 | 35-12, 35-15 | Cutover has explicit stated sequencing and never leaves a duplicate or orphan | ✓ SATISFIED | SC-5, turned by CR-01's closure. The matching-marker overwrite (flagged prohibition) leaves no duplicate and no orphan — it changes run FIELDS, not calendar topology |

No orphaned requirements: REQUIREMENTS.md maps exactly ALLOC-01 … ALLOC-05 to Phase 35 and every one appears in at least one plan's `requirements` field.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/allocation_projector.py` | 360-369 | Stale stored value substituted for a computed one, justified by an unenforced premise | ⚠️ Warning | Both directions of preview/real divergence (gap 1) |
| `solsys_code/management/commands/load_telescope_runs.py` | 307-310 | Prediction from a literal (`len(nights)`) standing in for the real call | ⚠️ Warning | Create-arm preview divergence (gap 2) |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 180 | One reason label prefixed onto two structurally different causes | ⚠️ Warning | Self-contradictory stderr (gap 4) |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 671 | Hardcoded `'ALLOC:'` beside an importable constant | ℹ️ Info | IN-02, carried forward |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 520 | Key claimed before the group's transaction commits | ℹ️ Info | IN-03, carried forward |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | cell 9 | Shared-DB mutation without `try/finally` | ℹ️ Info | IN-04 |

No `TBD` / `FIXME` / `XXX` debt markers in any of the ten files modified this round.

### Human Verification Required

None as a gate — status is already `gaps_found`. One judgment-tier prohibition is **flagged for a human decision** (see `flagged_prohibitions` in the frontmatter): whether the cutover's matching-marker branch silently reverting a post-import staff edit is an acceptable, documentable limitation (my assessment) or must be narrowed in code (the reviewer's suggested `_RUN_DRIFT` refusal).

## Gaps Summary

**Two defects, four findings, zero data-integrity risk.**

*Defect A — the half-null boundary fallback (gap 1; truths 12, 14, 15, 19, 20 plus the two carried 35-09 truths).* This is the fourth round of the same bug class. The pattern across rounds is consistent and worth naming, because it predicts what a fifth round would do: each round has correctly identified the shape that escaped the previous guard, extended the guard to reach it, and then written a sentence claiming parity broader than the extension delivers. This round's extension is the first to make the preview *worse* in one direction — it now aborts a night the real run handles cleanly. The fix with the best odds is subtractive: delete the fallback, return early when either boundary is unknown, and let the docstring say the preview cannot see a half-null span. That restores parity by silence, removes the new false positive, and is a smaller change than the one that introduced it.

*Defect B — the create-arm preview (gaps 2 and 3).* Same shape, different module: the existing-run arm was fixed, a sentence was written about both arms, and the untouched arm falsifies the sentence. Here the subtractive fix is also the better one — narrow the notebook's print and the runbook's because-clause to the arm they actually demonstrate. The runbook already contains the honest version three paragraphs earlier ('for a brand-new run, a first-time dry run predicts night counts from the window length'); the new paragraph contradicts it.

*Gap 4 (WR-04)* is a twenty-minute vocabulary fix in three places, and matters because the command's own stderr currently asserts a cause that is false for the branch CR-01 just added.

*The flagged prohibition (WR-02, new)* is the one I was asked to judge independently, and I disagree with treating it as blocking. The reviewer is right that it is real, reproduced and undocumented, and right that this round's remedy text routes an operator into it. But the run in question is one this command can PROVE came from the schedule line in hand — which is the exact distinction CR-01 established — and re-applying a schedule line to a row derived from it is the same file-authoritative behaviour `load_telescope_runs` has on every re-import and `import_campaign_csv` has on every re-import, the latter documented at length in this very runbook's 'Re-import gotcha' section. D-18 governs events the command *cannot explain*; this row it can. The correct response is the caveat the cutover is missing, not a new refusal path in a command that has now produced refusal-path regressions in two consecutive rounds.

### Recommendation on a third gap-closure round

**Yes, but scope it as 'stop over-claiming', not 'achieve parity' — and make it the last round.**

The phase goal is met and the milestone-relevant risk is retired. What is left would be safe to ship *if the documentation described it accurately*, and it does not — so a round is warranted, but a small and mostly subtractive one:

1. **Code (one change, revert-shaped):** delete the `existing.start_time`/`existing.end_time` fallback in `_raise_if_set_window_inverted()`; return when either boundary is unknown. Add PROBE-P1 and PROBE-P6 as regression tests. This removes the regression this round introduced and stops the fourth-iteration bug from having a fifth.
2. **Docs (three narrowings):** the loader notebook's final `print`, the runbook's L82-89 because-clause, and the `_raise_if_set_window_inverted()` docstring — each narrowed to what the code proves.
3. **Docs (one addition):** the cutover's missing re-run caveat, in the same sentence that sends the operator to the admin, plus a test pinning PROBE-P4's outcome as intended.
4. **Docs (one vocabulary fix):** the two-cause `duplicate_identity` wording in `_REASON_LABELS` and the two runbook definitions.

Explicitly **do not** attempt the transient-rolled-back-row preview for the create arm. It is the only suggestion on the table that adds a new write path to a command with a four-round regression history, and its entire benefit is a more precise preview of a line that the real run will refuse anyway.

If the developer prefers to ship now instead, item 2 is the minimum that must land first: shipping a notebook whose committed output states 'the preview never disagrees with the real run' is worse than shipping the divergence, because it is the sentence that stops an operator from noticing it.

---

_Verified: 2026-09-15T17:25:47Z_
_Verifier: Claude (gsd-verifier)_
