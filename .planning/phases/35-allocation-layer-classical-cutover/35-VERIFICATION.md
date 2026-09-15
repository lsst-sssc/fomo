---
phase: 35-allocation-layer-classical-cutover
verified: 2026-09-15T19:11:38Z
status: gaps_found
score: 183/186 must-haves verified
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
  - ".planning/phases/35-allocation-layer-classical-cutover/35-16-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-16-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-17-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-17-SUMMARY.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-18-PLAN.md"
  - ".planning/phases/35-allocation-layer-classical-cutover/35-18-SUMMARY.md"
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
covered_digest: "v1:sha256:dc69c7a68a5773c68df7b3991820a8c83cf72b8c1cf7d4424d9e42cf197f65cb"
behavior_unverified: 0
overrides_applied: 0
flagged_prohibitions: 0
re_verification:
  previous_status: gaps_found
  previous_score: 146/155
  previous_verified: 2026-09-15T17:25:47Z
  gap_closure_plans: ["35-16", "35-17", "35-18"]
  gaps_closed:
    - "Prior gap 1 (Defect A — the half-null stored-boundary fallback, WR-01, four rounds) — CLOSED BY GENUINE REVERT, independently confirmed against the source, not the SUMMARY. `_raise_if_set_window_inverted()` is back to a two-parameter signature `(run, night)`; `grep -rn '_raise_if_set_window_inverted' solsys_code/` returns exactly one definition (`allocation_projector.py:321`) and two two-argument call sites (`:750`, `:780`). The `existing` parameter, both `existing.start_time`/`existing.end_time` fallback expressions and the round-2 premise comment are gone; lines 359-368 now read both-null early return -> resolve each boundary from the run's own fields -> `if start is None or end is None: return` -> `_raise_if_inverted()`. PROBE-P1's false positive (round 2's regression) and PROBE-P6's false negative are both now pinned as tests that assert the NARROWED contract: `test_dry_run_of_a_half_null_remint_after_nulling_a_set_start_agrees_with_the_real_run` (:1223) and `test_dry_run_cannot_see_a_half_null_remint_inversion_and_the_real_run_still_raises` (:1286). The test that passed under the defect is REPLACED, not supplemented — `test_dry_run_of_a_half_null_remint_inverted_window_also_raises` returns zero hits repo-wide; its slot is now `..._stays_silent_while_the_real_run_raises` (:1256) whose docstring states why it must not raise. Both set/set regressions survive unchanged (:1168 create path, :1195 re-mint path). No `sun_event()` call was added to the guard."
    - "Prior gap 2 (Defect B — the loader create-arm preview) — CLOSED BY NARROWING, per the prior pass's own recommendation, with no new write path. `load_telescope_runs.py:310-325` now carries an explicit create-arm comment stating that `reconcile_run()` is never called on this arm, that nothing on it can fail, that the real branch's reconcile CAN raise and report `skipped`, and that the speculative transient-row preview was deliberately rejected. The counter fold is still `night_created += len(nights)` — no preview write path was added (35-17 prohibition 1 upheld). `test_dry_run_of_a_brand_new_line_cannot_predict_a_reconcile_failure` (`test_load_telescope_runs.py:751`, inside `TestMalformedTimezoneSkipsOneLine`) pins PROBE-P5's exact five-tuples, `dry=(1,1,0,0,0)` against `real=(1,0,0,0,1)`, and asserts `CampaignRun.objects.count() == 0` afterwards. It passes in my own run."
    - "Prior gap 3 (operator-facing over-claims) — CLOSED. Runbook lines 76-97 now describe the two arms separately in the shipped code's own terms: the invariant holds on both passes; the existing-run arm folds after the preview reconcile so preview and real report the same decision; 'For a brand-new line there is no preview reconcile at all ... a line the real pass drops under `skipped` can still preview as `created`.' The loader notebook's committed output no longer contains the phrase 'never disagrees' (0 occurrences repo-wide); cell index 18, execution count 9, now prints `Both passes agree: (0, 0, 0, 1) -- the preview on the existing-run arm agrees with the real pass.` Both notebooks carry 0 null execution counts (16/16 and 18/18 code cells), and the 35-18 commits (a313392, 15000cc, aa6dbb0) touch only `docs/` — no file under `solsys_code/` was edited from the docs plan (35-18 prohibition 3 upheld). `src/fomo_db.sqlite3` is unmodified in `git status` (prohibition 2 upheld). Zero hedge words (`usually` / `in normal operation` / `in most cases`) anywhere in the runbook."
    - "Prior gap 4 (the `duplicate_identity` reason vocabulary, partial) — CLOSED in all three places. `_REASON_LABELS[_DUPLICATE_IDENTITY]` (`cutover_classical_allocations.py:194-197`) now reads '... or a CampaignRun already holds the derived identity key and cannot be proved to have come from this line', so the label prefixed onto the no-marker branch's stderr (`:455`) is no longer self-contradictory. Both runbook definitions state both causes (`:947` in the reason vocabulary, `:1535` in the troubleshooting Cause list). The reconciler notebook's EXECUTED output carries the two-cause phrase 12 times — it was regenerated against the new label, not left stale."
    - "Prior flagged prohibition (WR-02 iteration 6 — the matching-marker claimant's staff edits silently reverted) — RESOLVED AS DOCUMENTED, the disposition the prior pass recommended and 35-17 truth 8 adopted. The predicate is untouched (`existing_source_line != source_line`), no `_RUN_DRIFT` refusal path was added, and the behaviour is now stated as a 'Re-run gotcha' in both the cutover module docstring and the runbook (`:989-1000`, in the voice of the existing `import_campaign_csv` note). The no-marker remedy string itself (`:455-462`) closes the loop in its own sentence: '... then re-run -- note a run whose marker then matches will have its fields re-applied from the schedule line on that re-run'. `test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line` (`test_cutover_classical_allocations.py:1254`) pins the outcome as INTENDED. No prohibition is flagged for human decision this pass."
  gaps_remaining: []
  gaps_superseded:
    - "35-13 truths 1, 3, 4, 8 and 9 (the round-2 half-null parity claims the prior pass marked FAILED) are SUPERSEDED, not carried. Plan 35-16 was written and executed specifically to retract them: its truths 3, 4 and 7 replace 'the preview raises the same ValueError' with 'the preview stays silent for a half-null run on either branch', and that narrowed contract is pinned by two named tests whose docstrings say why a future round must not widen it again. The ROADMAP success criteria never required dry-run parity, so retracting a plan-level claim in favour of a truthful narrower one is closure, not an unmet must-have. Recorded here rather than dropped."
    - "35-14 truth 7 (the loader's ALLOC-04 edge probe, previously FAILED) is SUPERSEDED by 35-17 truths 1, 2, 3 and 9, which state the create arm's inability to predict a reconcile failure rather than asserting agreement it cannot deliver, and pin it with `test_dry_run_of_a_brand_new_line_cannot_predict_a_reconcile_failure`."
  regressions: []
gaps:
  - truth: "When a stored allocation night's `start_time` or `end_time` no longer matches what the run's current sub-night fields say it should be, the projector deletes and re-creates that night (D-13) rather than leaving the stored span in place (35-03 truth 5); and a null sub-night field means the computed sunset (start) or sunrise (end) for that night (35-03 truth 2). ROADMAP SC-1 depends on both: the event an allocation shows must be the sunset→sunrise span the run currently declares."
    status: failed
    reason: "CR-01, 35-REVIEW.md iteration 7 — a NEW blocker found while narrowing round 3's docstring, in the REAL re-mint path rather than the preview. `_span_needs_remint()` only ever compares a SET sub-night field against the stored boundary, so clearing a previously-SET field to null never marks the night for re-mint: the calendar keeps the operator's stale OLD boundary permanently, with no exception, no log line and no counter — `reconcile_run()` reports it as `unchanged`. The premise round 3's own new `_raise_if_set_window_inverted()` docstring states ('nulling a previously-set field leaves that field's old operator value sitting in the stored event, not a sunset or sunrise') is exactly what breaks this decision, and round 3 applied that premise only to the preview. Reachable through the Django admin: `CampaignRunAdmin` (`admin.py:132-166`) declares no `fields` and no `exclude`, and `readonly_fields` is `['approval_status']` alone (`get_readonly_fields()` narrows only `source`), so both sub-night fields are fully editable — and clearing one is the documented way an operator says 'this run uses the whole night'. NOT reachable via a `load_telescope_runs` re-import, because dropping the window token changes `_source_identifier()` and mints a different run; admin/API edits are the live path. INDEPENDENTLY REPRODUCED IN THIS VERIFICATION PROCESS (temporary `TestCase` against a real migrated Django test DB, La Silla obscode 809 / `America/Santiago`, night 2026-07-09, true sunset/sunrise `22:06:35` / `11:29:46` UTC; probe deleted afterwards, `git status --short` clean of source changes). All three shapes stale, all three reported `unchanged=1`: (A) set/set `23:00`-`05:00` -> both nulled, event stays `23:00 -> 05:00`; (B) half-null `23:00`/None -> both nulled, event stays `23:00 -> 11:29:46`; (C) set/set -> start nulled only, event stays `23:00 -> 05:00` (the nastiest branch — the other field still matches, so the second `if` is reached, evaluates False, and the function returns 'no re-mint needed' even though one declared boundary genuinely changed)."
    artifacts:
      - path: "solsys_code/allocation_projector.py"
        issue: "`_span_needs_remint()` at L393-404, consumed at L736. L393-394 returns False for both-null; L396 and L400 each guard on `... is not None`, so a null field is never compared against the stored boundary at all. Correct for D-13's drift case (a null field means 'use the sun event', which moves by fractions of a second between runs — do not rewrite for that), wrong for an operator CLEARING a previously-set field, which is a real semantic change whose stored boundary is the operator's own old value, not a sun event."
      - path: "solsys_code/admin.py"
        issue: "`CampaignRunAdmin` L132-166 makes `night_start_utc`/`night_end_utc` editable (no `fields`, no `exclude`, `readonly_fields = ['approval_status']`). This is 35-03 truth 4 working as designed — it is what makes the defect reachable, not a defect itself."
      - path: "solsys_code/tests/test_allocation_projector.py"
        issue: "No test covers any clear-to-null transition. The three round-3 half-null tests (L1223, L1256, L1286) all keep at least one field SET after the edit, so all three pass under this defect — the same way the prior four rounds of this bug class survived. 205 tests green is not evidence for this truth."
    missing:
      - "Make `_span_needs_remint()` fire when a sub-night field is null but the stored boundary is provably NOT sun-derived. 35-REVIEW.md CR-01 carries a concrete body: compare the null side against `sun_event(run.site, night, kind='sun')` with a one-minute tolerance, so astropy drift alone can never trigger a rewrite (D-13's actual concern) while a cleared operator value always does. This costs exactly one `sun_event()` call, only on a transition that is rare and is a genuine operator change — and on a night that is about to be deleted and re-minted anyway."
      - "If reintroducing `sun_event()` on the update path is judged to breach D-13 outright, take the provenance route instead: persist the resolved boundaries a night was minted from (a `CalendarEventMeta` column, or a second structured line in the event description alongside the existing `Dark window (-15 deg, UTC): ` line) and compare against that. This is the same provenance work the accepted WR-01 limitation would need, so one implementation settles both. Do NOT infer provenance from the stored value again — that is precisely what round 2's reverted fallback did."
      - "Add a regression test for ALL THREE probe shapes — set/set -> null/null, half-null -> null/null, and set/set -> half-null — since each reaches the defect through a different branch of `_span_needs_remint()` and shape C passes the first two `if` statements before returning False. Assert the re-minted event's boundaries equal the true sunset/sunrise, not merely that `result.updated` or `result.created` is non-zero."
      - "Decide whether a silent `unchanged` is acceptable for any run/calendar disagreement this projector can detect. Today the defect produces no counter, no log line and no exception, and there is no later pass that can ever discover it — that is the property that makes it a BLOCKER rather than the WARNING its preview-side sibling was."
deferred: []
user_deferred:
  - finding: "WR-01 (35-REVIEW.md iteration 7) — after the revert, `--dry-run` cannot detect ANY half-null inverted span, and the half-night classical line (`1130-EoN` / `BoN-0230` via `_window_token_to_time()`) is exactly that shape. `grep -in 'invert|half-null|half-night|1130-EoN|BoN-0230' docs/runbooks/telescope_runs_calendar.rst` returns NOTHING (confirmed in this pass), so the limitation is documented only in code comments and test docstrings while the runbook's own always-dry-run-first instruction implies a protection that does not exist for this shape."
    severity: warning
    decision: "EXPLICITLY DEFERRED BY THE USER to a later round. Recorded here so it is not silently dropped; deliberately NOT in the actionable `gaps` list, so a `/gsd-plan-phase 35 --gaps` run stays scoped to the single BLOCKER."
  - finding: "WR-02 (35-REVIEW.md iteration 7) — round 3's new 'Re-run gotcha' caveat under-enumerates the fields the cutover re-applies, and its two copies disagree with each other. `fields` at `cutover_classical_allocations.py:489-504` has 14 keys; the module docstring omits `site_needs_review` and `telescope_instrument`, and the runbook note (`:989-1000`) omits those two plus `source` and `approval_status`. `site_needs_review` is not cosmetic — `campaign_views.py:221` filters the staff 'Sites Needing Review' queue on it, so a run a staff member deliberately re-flagged silently leaves that queue on the next cutover run: the exact class of 'post-import staff edit does not survive' the caveat exists to warn about."
    severity: warning
    decision: "EXPLICITLY DEFERRED BY THE USER to a later round. Note this does NOT falsify 35-17 truth 5 or 35-18 truth 5 as those truths are worded — both enumerate the same subset the shipped prose does, and both are marked VERIFIED against their own wording. The finding is that the wording itself was scoped too narrowly."
  - finding: "WR-03 (35-REVIEW.md iteration 7) — `cutover_classical_allocations.py:649` discards `adopt_event_into_run()`'s refusal return value where its sibling call site (`allocation_projector.py:518`) checks it. A refused adoption would leave an event re-keyed into `ALLOC:` but unattributed and counted as a success, which is what D-18 exists to prevent. Unreachable today because the pre-filter at `:511-517` rejects any event whose `meta.run_id is not None`."
    severity: warning
    decision: "EXPLICITLY DEFERRED BY THE USER to a later round. Latent, not currently reachable; recorded, not actioned."
advisory:
  - finding: "IN-01 — the both-null early return in `_raise_if_set_window_inverted()` (`allocation_projector.py:359-360`) is now redundant: after the revert, any path reaching it with a null field also hits `if start is None or end is None: return` at L366. Two returns express one rule, and the docstring describes only the L366 check."
    category: other
    reason: "Behaviourally harmless (I re-read both returns and confirmed the subsumption). Keeping the `and` form was required by 35-16 prohibition 5 so it keeps agreeing with `_span_needs_remint()`. Fix: one docstring line noting the both-null return is a fast path, or collapse to the single `or`. New-scope, no deterministic failure evidence."
    evidence_status: "none provided — reasoned from control flow, no failing behaviour"
  - finding: "IN-02 — `cutover_classical_allocations.py:690` still hardcodes `url__startswith='ALLOC:'` while `allocation_projector.ALLOC_URL_NAMESPACE` exists and this module already imports four names from that module. Carried forward unchanged from iterations 5, 6 and 7."
    category: other
    reason: "Latent: a namespace rename would silently make the final summary count zero. Fix: import the constant and use it."
    evidence_status: "none provided — latent, no current failure"
  - finding: "IN-03 — `seen_keys[key] = source_line` (`:539`) is still claimed before the group's `with transaction.atomic():` (`:560`). A group whose transaction rolls back wholesale keeps the key claimed, so a sibling group sharing it is reported under `duplicate_identity` naming a line that converted nothing. Carried forward unchanged."
    category: other
    reason: "Fix: move the assignment to just after the `runs_created += group_created` fold at `:666`, alongside the other post-commit bookkeeping."
    evidence_status: "none provided — reasoned from control flow, no probe run"
  - finding: "IN-04 — no executed notebook cell exercises the `existing_source_line is None` refusal, the exact branch round 2's CR-01 inverted and round 3's remedy text extended. The reconciler notebook's executed outputs carry the mismatched-marker message but never the no-marker one; that branch appears only as markdown prose. Carried forward from iteration 6."
    category: other
    reason: "The paired-docs rule's 'exercise the new behavior' clause is met at the level of the corrected operator text (12 occurrences of the two-cause label in executed output) but not as an executed demonstration of the branch. Fix: one throwaway fixture cell with a claimant whose `observation_details` has no `Source line:` marker."
    evidence_status: "static observation; both notebooks inspected programmatically (16/16 and 18/18 code cells, 0 null execution counts)"
  - finding: "IN-05 — the loader notebook restores the mutated `Observatory.timezone` without `try/finally` (cells 16 and 18). If `call_command()` raises, the `America/Santigo` typo persists for every later cell in the same run. Iteration 6 framed this as mutating the shared dev database; iteration 7 corrected that (cell 1 confirms a per-run scratch copy under `/tmp/fomo-notebook-db-*`), and I confirmed `src/fomo_db.sqlite3` is unmodified in `git status`. The ordering fragility stands; the shared-DB risk does not."
    category: other
    reason: "Fix: wrap each in `try: ... finally: ntt.timezone = original; ntt.save(update_fields=['timezone'])`."
    evidence_status: "static observation; notebook cell source inspected"
  - finding: "IN-06 / IN-07 — `logger` is defined and never called in the cutover command (`:149`), and the reason breakdown is printed in first-seen order on stdout but sorted order in the `CommandError`, despite the constants being declared 'in report order'."
    category: other
    reason: "Both cosmetic. The second matters slightly more than it looks: a notebook caller driving the command through `call_command()` sees only the `CommandError` message. Fix: delete the dead logger; iterate both summaries in `_REASON_LABELS` declaration order."
    evidence_status: "none provided — cosmetic, no failing behaviour"
  - finding: "WR-04 round-2 — `observation_projector.py:647-658`'s savepoint-less swallowed `campaign_run_links` lookup error. Deliberately deferred as advisory by rounds 2 and 3 and not re-reviewed by iteration 7."
    category: architectural
    reason: "Unchanged since 24875bf, which predates all three prior verification timestamps; the file was not touched by any gap-closure round. The exposure is PostgreSQL-specific and was not reproducible on this project's SQLite backend. Carried forward unresolved."
    evidence_status: "probe executed in a prior round, no failure reproduced on SQLite"
behavior_unverified_items: []
coincidental_reliance_items:
  - truth: "The cutover's identity guard refuses what it cannot prove it owns (35-12 truths 1-2; `TestDatabaseScopedIdentityGuard`, 5 tests, green in this pass's 205-test run)."
    reason: undeclared-precondition
    harden: "The REFUSING branch is now sound — absence of a marker denies rather than grants. What remains is its mirror: the PERMISSIVE branch's correctness depends on `observation_details` being trustworthy when it happens to match, and that field is writable from the Django admin (`admin.py:165`), from `import_campaign_csv.py:321` and from `campaign_forms.py:65`. Nothing in production guarantees a matching marker was written by this pipeline rather than typed by a person. Round 3 documented the consequence (the re-run gotcha) rather than removing the dependency, which is the accepted disposition — but the precondition is still undeclared in code. Promote the provenance out of free text: store the identity provenance in a field the operator cannot overwrite, or mark it read-only for classical rows. Advisory only; does not affect the score or the status."
  - truth: "Allocation nights re-mint when the run's sub-night fields change (35-03 truth 5, the non-null half)."
    reason: undeclared-precondition
    harden: "Where this truth still holds, it holds because the stored boundary happens to be comparable without astropy — a precondition that exists only while at least one field is SET. Nothing in the code declares 'the stored boundary's provenance is knowable', and CR-01 is what happens when that unstated precondition lapses. The durable fix is the same provenance recording CR-01's `missing` list names; recording it would turn an incidental property into a declared one."
---

# Phase 35: Allocation Layer & Classical Cutover — Verification Report (fourth pass)

**Phase Goal:** An allocation — classical schedule line, approved submission, TBD/range run, campaign or no campaign — draws its own sunset→sunrise intent nights and hands each night over the moment a real observation links to it; `load_telescope_runs` writes allocations instead of calendar events.
**Verified:** 2026-09-15T19:11:38Z
**Status:** gaps_found
**Re-verification:** Yes — fourth pass, after the third gap-closure round (plans 35-16, 35-17, 35-18)

## Headline

**Round 3 did what it was scoped to do — all four items verified against the source, not the SUMMARY. A new BLOCKER was found in a code path round 3 came within one function of checking.**

The third round was the good round. Its central change was *subtractive*: it reverted round 2's stored-boundary fallback rather than patching it a fifth time, and in doing so it ended a bug class that had survived four rounds. I confirmed the revert is genuine (two-parameter signature, no `existing` parameter, both fallback expressions deleted, exactly two two-argument call sites), that the test which passed under the defect was replaced rather than supplemented, and that the three documentation over-claims and the reason-vocabulary gap are all closed in the shipped text and the re-executed notebook output.

But the docstring round 3 wrote to justify the revert states the falsifying premise in plain words — *"nulling a previously-set field leaves that field's old operator value sitting in the stored event, not a sunset or sunrise"* — and then applies it only to the **preview**. Nobody checked whether the same fact breaks the **real** re-mint decision fifty lines below. It does. `_span_needs_remint()` never compares a null sub-night field against anything, so an operator clearing a field to null — the documented way to say "this run uses the whole night" — leaves the calendar showing the old boundary forever, reported as `unchanged`.

That is materially different from every finding this phase has carried since round 1. The prior three passes could each say *"no finding can produce a wrong event, a duplicate or an orphan."* This one produces a wrong event: a run declaring a full sunset→sunrise night whose calendar entry says 23:00–05:00, silently, with no counter, no log line, no exception, and no later pass that can ever discover it.

I reproduced it in this process — all three shapes — against a real migrated Django test database.

## Goal Achievement

### ROADMAP Success Criteria

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Allocation with resolved site + awarded window shows one sunset→sunrise event per window night; queue/class-wide/satellite keeps a single container | ✗ FAILED | Holds for a freshly-minted allocation and for every SET sub-night edit — dispatch intact, 205 tests green. Broken by CR-01 on a reachable, documented operator action: after clearing a previously-set sub-night field to null the run declares sunset→sunrise while the event still shows the old boundary. Reproduced three ways in this process (see Probe Execution) |
| 2 | Allocation nights follow the site-local observing night (Chilean + Australian sites) | ✓ VERIFIED | `_night_span_utc()` / `night_bounds()` untouched by round 3 (`git show --stat` over 5eeaa48 and 3635573 shows only the guard and its tests); `test_allocation_projector` green. CR-01 is about WHICH boundary a night carries, not which calendar day it lands on |
| 3 | Linking an `ObservationRecord` removes that night's allocation event, leaves the observation's own event untouched; unlinking restores it | ✓ VERIFIED | `test_allocation_projector_signals` green within the 205-test run; `reconcile_campaign_runs_demo.ipynb` re-executed again this round (18/18 code cells, 0 null execution counts), so the handoff evidence is current rather than carried on trust |
| 4 | `load_telescope_runs` produces the same per-night calendar as before, by way of an allocation record, and re-running changes nothing | ✓ VERIFIED | The command imports no calendar writer; `test_load_telescope_runs` green. CR-01 is NOT reachable from this command: dropping a window token changes `_source_identifier()` and mints a different run, so a re-import cannot produce the null-clearing transition. The create-arm preview divergence is now stated and pinned rather than over-claimed |
| 5 | After the cutover, one event per night: no duplicate, no orphan left behind | ✓ VERIFIED | Unchanged since the prior pass turned it. `TestDatabaseScopedIdentityGuard` 5/5 green; the identity predicate is byte-identical to what round 2 left (only the `else:` branch's reason string grew). CR-01 changes an event's SPAN, not the calendar's topology |

**Roadmap contract: 4/5.**

### Observable Truths — round-3 plan must-haves

Plans 35-01 … 35-15 were verified across the three prior passes. Those whose truths round 3 explicitly retracted are recorded under `re_verification.gaps_superseded` rather than dropped; the rest were re-checked for regression (205 tests green, ruff + ruff-format + sphinx-build all Passed, no source file outside the round-3 set changed).

#### Plan 35-16 — revert the half-null fallback

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | The dry run never raises for a night the real run creates cleanly (PROBE-P1, round 2's regression) | ✓ VERIFIED | `test_dry_run_of_a_half_null_remint_after_nulling_a_set_start_agrees_with_the_real_run` (`test_allocation_projector.py:1223`) mints `23:00`/None, edits to None/`22:30`, runs the preview with no `assertRaises`, then the real run and asserts `start < end`. Green |
| 2 | The guard resolves a boundary only from the run's own sub-night fields; never substitutes a stored `CalendarEvent` boundary | ✓ VERIFIED | Read `allocation_projector.py:359-368` in full. Two-parameter signature `(run, night)`; `grep -rn '_raise_if_set_window_inverted' solsys_code/` -> one definition, two two-argument call sites (`:750`, `:780`). Zero occurrences of `existing.start_time` / `existing.end_time` in the function |
| 3 | A half-null run is not previewed for inversion on EITHER caller branch | ✓ VERIFIED | `if start is None or end is None: return` at L366 subsumes every half-null path on both branches |
| 4 | The narrowed contract is pinned by an explicit test (PROBE-P6 asserted as the KNOWN limitation) | ✓ VERIFIED | `test_dry_run_cannot_see_a_half_null_remint_inversion_and_the_real_run_still_raises` (`:1286`), whose docstring states that a future change making this preview raise must first solve provenance rather than re-infer it |
| 5 | `test_dry_run_of_a_half_null_remint_inverted_window_also_raises` no longer exists asserting that contract | ✓ VERIFIED | `grep -rn` over `solsys_code/` returns zero hits for that name. Its slot is now `..._stays_silent_while_the_real_run_raises` (`:1256`) — REPLACED, not supplemented, per the CR-01 precedent |
| 6 | Both set/set inversion regressions still pass unchanged | ✓ VERIFIED | `..._brand_new_inverted_window_also_raises` (`:1168`, NF-10) and `..._remint_inverted_window_also_raises` (`:1195`, NF-20) present; 205 tests OK |
| 7 | The docstring and call-site comment describe what the guard PROVES, not an unrecorded provenance property | ✓ VERIFIED | `:322-349` now states the revert, names both falsifying probes and says the guard "returns without raising whenever either resolved boundary is unknown — restoring parity by silence, not by a guessed answer". No claim that a stored boundary is sun-derived; no claim that the create path is the only unchecked shape. **This docstring is also where CR-01 hides: it states the correct premise and applies it only to the preview** |
| 8 | D-13 not breached; no astropy call added; the change is subtractive | ✓ VERIFIED | No `sun_event()` in the guard; 5eeaa48 is net −20 lines in `allocation_projector.py` |
| 9 | [edge probe ALLOC-02] For a half-null run the preview and real run agree or the preview is silent — never two different decisions | ✓ VERIFIED | Both directions asserted (`:1223` agreement, `:1286`/`:1256` silence-then-raise) |
| 10 | [already covered] `_night_span_utc()` / `night_bounds()` untouched | ✓ VERIFIED | `git show --stat` over both 35-16 commits: only the guard and its tests |

#### Plan 35-17 — narrow the loader's create-arm claim; vocabulary and gotcha

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | No in-code sentence claims a dry/real parity the loader does not have | ✓ VERIFIED | `load_telescope_runs.py:310-325`: the create-arm comment states `reconcile_run()` is never called on this arm, quotes PROBE-P5's two summary lines verbatim, and records that the transient-row alternative was deliberately rejected |
| 2 | The stated parity is scoped to the `existing is not None` arm; the create arm's inability is stated, not implied | ✓ VERIFIED | `--dry-run` help text at `:172` is arm-scoped ("For a line whose run already exists, the night-level preview comes from reconcile_run(dry_run=True)"); the WR-02 fold comment at `:327` unchanged. `write_and_reconcile_campaign_run()` remains the sole write path |
| 3 | `TestMalformedTimezoneSkipsOneLine` pins the create arm's divergence as the KNOWN contract | ✓ VERIFIED | `test_load_telescope_runs.py:751`, inside that class (`class TestMalformedTimezoneSkipsOneLine` at `:578`). Asserts `dry_tuple == (1,1,0,0,0)` and `real_tuple == (1,0,0,0,1)` and `CampaignRun.objects.count() == 0`. Green |
| 4 | The `duplicate_identity` label states BOTH causes the command reports it for | ✓ VERIFIED | `_REASON_LABELS[_DUPLICATE_IDENTITY]` at `:194-197` carries both clauses; it is prefixed onto both branch strings (`:448` mismatched, `:455` no-marker), so the no-marker stderr line is no longer self-contradictory |
| 5 | The cutover documents its own re-run gotcha | ✓ VERIFIED (against its own wording) | Module docstring `:69-83` states the find-and-update behaviour, the field list, "A post-import staff edit to any of them does not survive the next cutover run", and the `import_campaign_csv` precedent. **The enumeration is incomplete — WR-02 iteration 7, explicitly deferred by the user; see `user_deferred`** |
| 6 | The no-marker remedy names the consequence in the same sentence that sends the operator to the admin | ✓ VERIFIED | `:455-462` ends "... then re-run -- note a run whose marker then matches will have its fields re-applied from the schedule line on that re-run", and the docstring adds "restore the marker deliberately, not reflexively" |
| 7 | PROBE-P4's outcome is pinned by a test as INTENDED behaviour | ✓ VERIFIED | `test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line` (`test_cutover_classical_allocations.py:1254`), docstring explicitly distinguishes it from the refusal test |
| 8 | The matching-marker overwrite is NOT narrowed in code | ✓ VERIFIED | The predicate `existing_source_line != source_line` at `:445` is unchanged; no `_RUN_DRIFT` reason, no field-comparison guard. 73a2149 touches docstring, label and the `else:` reason string only |
| 9 | [edge probe ALLOC-04] Preview and real agree, or the inability is stated in code and pinned by a test | ✓ VERIFIED | Truths 1-3 above |
| 10 | [edge probe ALLOC-05] Every reason the cutover prints names a cause true for its branch | ✓ VERIFIED | Two-cause label + two branch-specific bodies |
| 11 | [already covered] The identity guard still refuses an absent/differing marker, exits non-zero, leaves the run byte-identical | ✓ VERIFIED | `TestDatabaseScopedIdentityGuard` green in the 205-test run |

#### Plan 35-18 — runbook and both notebooks

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | No operator-facing sentence states a parity guarantee the code does not hold | ✓ VERIFIED | Runbook `:76-97` and the loader notebook's cell-18 output both read arm-scoped now; `grep` for "never disagrees" returns 0 across both notebooks |
| 2 | The runbook's loader paragraph describes the two arms separately | ✓ VERIFIED | Read in full: invariant stated for both passes, then "The two arms reach that invariant differently, and only one of them lets the preview predict what the real pass will report", then the explicit create-arm sentence "a line the real pass drops under `skipped` can still preview as `created`" |
| 3 | The loader notebook's committed output is arm-scoped and regenerated by re-execution | ✓ VERIFIED | Cell index 18, execution count 9: `Both passes agree: (0, 0, 0, 1) -- the preview on the existing-run arm agrees with the real pass.` 16/16 code cells, 0 null execution counts |
| 4 | Both runbook `duplicate_identity` definitions state both causes | ✓ VERIFIED | Reason vocabulary at `:947` and troubleshooting Cause list at `:1535`, both carrying the shared phrase "already holds the derived identity key" that anchors the label, the runbook and the notebook |
| 5 | The cutover's re-run gotcha appears in the runbook, in the `import_campaign_csv` note's voice | ✓ VERIFIED (against its own wording) | `.. note:: **Re-run gotcha:**` at `:989-1000`, sharing the phrase "does not survive the next cutover run" with the module docstring. **Same incomplete enumeration as 35-17 truth 5 — WR-02, user-deferred** |
| 6 | The reconciler notebook is regenerated against the two-cause label | ✓ VERIFIED | The two-cause phrase "cannot be proved to have come" appears 12 times in the notebook's EXECUTED output (0 times in the loader notebook, which does not run the cutover) — the label change and the regeneration landed together |
| 7 | Both notebooks regenerated by re-execution; no hand-edited output | ✓ VERIFIED | 16/16 and 18/18 code cells with non-null execution counts; iteration 7 independently confirmed every `iopub` timestamp moved from `16:52` to `18:46` |
| 8 | [edge probe ALLOC-03] The handoff is re-demonstrated end to end by the re-execution | ✓ VERIFIED | No null execution counts anywhere in the reconciler notebook |
| 9 | [already covered] The 35-15 cutover corrections are not undone | ✓ VERIFIED | The marker precondition survives at `:947`; zero hedge words in the whole file |
| 10 | `pre-commit run sphinx-build --all-files` still passes | ✓ VERIFIED | Executed in this process: **Passed** |

**Score:** 183/186 must-haves verified (181 plan truths + 5 ROADMAP success criteria; 3 failed — 35-03 truth 2, 35-03 truth 5 and SC-1, all one root cause; 0 behavior-unverified; 0 prohibitions flagged; 0 overrides).

The three failures are one defect, found in a plan from wave 2 of the ORIGINAL phase (35-03), not in anything round 3 built. Every truth the third gap-closure round set out to establish is verified.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `solsys_code/allocation_projector.py` | Guard reverted to run-own-fields only; re-mint decision correct | ⚠️ HOLLOW | The guard (`:321-368`) is correct, wired and substantively reverted. `_span_needs_remint()` (`:393-404`) is present and wired at `:736` but returns the wrong answer for every clear-to-null transition — present + wired, wrong decision |
| `solsys_code/tests/test_allocation_projector.py` | PROBE-P1 agreement + PROBE-P6 limitation pinned; stale test replaced | ⚠️ PARTIAL | Both round-3 tests present and green; the replaced test name is gone repo-wide. No test covers any clear-to-null transition — all three half-null tests pass under CR-01 |
| `solsys_code/management/commands/load_telescope_runs.py` | Create-arm claim narrowed, no new write path | ✓ VERIFIED | `:310-325` states the limitation and cites PROBE-P5; fold still `len(nights)`; no transient row |
| `solsys_code/tests/test_load_telescope_runs.py` | Create-arm divergence pinned | ✓ VERIFIED | `:751` asserts both five-tuples and zero rows written |
| `solsys_code/management/commands/cutover_classical_allocations.py` | Two-cause label; re-run gotcha; predicate untouched | ✓ VERIFIED | `:194-197` label; `:69-83` gotcha; `:445` predicate byte-identical to round 2 |
| `solsys_code/tests/test_cutover_classical_allocations.py` | PROBE-P4 pinned as intended | ✓ VERIFIED | `:1254`; guard class still 5 tests, all green |
| `docs/runbooks/telescope_runs_calendar.rst` | Loader claim narrowed; both reason definitions; gotcha note | ✓ VERIFIED | `:76-97`, `:947`, `:1535`, `:989-1000`; 0 hedge words; sphinx-build Passed |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | Arm-scoped conclusion, regenerated | ✓ VERIFIED | 16/16 executed; "never disagrees" gone |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Regenerated against the two-cause label | ✓ VERIFIED | 18/18 executed; 12 occurrences of the new label in executed output |
| `solsys_code/admin.py` | Sub-night fields staff-editable with no admin change (35-03 t4) | ✓ VERIFIED | `:132-166`: no `fields`, no `exclude`, `readonly_fields = ['approval_status']`. Working as designed — and the reachability path for CR-01 |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| A staff admin edit of `night_start_utc` / `night_end_utc` | the next `reconcile_run()` re-mints exactly the affected nights (35-03 key_link 2) | `_span_needs_remint()` | ✗ NOT WIRED (for clear-to-null) | Wired for a SET -> SET edit and for a null -> SET edit; dead for SET -> null. This is the link CR-01 breaks |
| `_raise_if_set_window_inverted(run, night)` | both `_mint_fields()` caller branches' `if dry_run:` short-circuits | shared guard | ✓ WIRED | Two two-argument call sites (`:750` create, `:780` re-mint); the guard reads only the run's own fields |
| `_REASON_LABELS[_DUPLICATE_IDENTITY]` | both branch reason strings → runbook `:947` / `:1535` → reconciler notebook executed output | shared phrase "already holds the derived identity key" | ✓ WIRED | All four copies carry both causes; the notebook's output was regenerated against the label, so drift would show as a stale notebook rather than as an operator hunting a nonexistent second line |
| The cutover module docstring's gotcha | runbook `:989-1000` | shared phrase "does not survive the next cutover run" | ✓ WIRED | Present in both. Both enumerations are incomplete (WR-02, user-deferred) |
| `load_telescope_runs.py`'s create-arm comment | runbook `:76-97` → loader notebook cell 18 | three copies of one claim | ✓ WIRED | All three now arm-scoped and mutually consistent |
| `_extract_source_line()` result | the guard's write authorisation | `None` on the refusing side | ✓ WIRED | Unchanged from the prior pass; a parse that found nothing can only deny |
| `docs/index.rst:24` toctree | `runbooks/telescope_runs_calendar` | Sphinx | ✓ WIRED | `pre-commit run sphinx-build --all-files` Passed in this process |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Produces real data | Status |
|---|---|---|---|---|
| Allocation night `start_time` / `end_time`, fresh mint | both boundaries | `sun_event()` + `_time_of_day_to_datetime()` in `_mint_fields()` | Yes | ✓ FLOWING |
| Allocation night `start_time` / `end_time`, after a clear-to-null edit | the nulled boundary | **nothing** — the stored value from before the edit survives | No | ✗ DISCONNECTED (CR-01) |
| `_raise_if_set_window_inverted` | `start` / `end` | the run's own sub-night fields only | Yes, or `None` and the guard stays silent | ✓ FLOWING |
| `load_telescope_runs --dry-run`, create arm | `night_created` | `len(nights)` — a literal window length | No — and the code, runbook and notebook all now say so | ✓ FLOWING (as a documented, pinned limitation) |
| Loader notebook cell 18 | both summary lines + the conclusion under them | two real `call_command()` invocations | Yes — and the sentence beneath now matches them | ✓ FLOWING |
| Reconciler notebook `duplicate_identity` output | the two-cause label | a real command run against the current `_REASON_LABELS` | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| No regression across the phase surface | `python manage.py test solsys_code.tests.{test_allocation_projector,test_allocation_projector_signals,test_campaign_reconciler,test_cutover_classical_allocations,test_load_telescope_runs}` (run once) | `Ran 205 tests in 105.529s ... OK` | ✓ PASS |
| Guard revert is genuine, not narrated | `grep -rn '_raise_if_set_window_inverted' solsys_code/` + read of `:321-368` | 1 definition, 2 two-argument call sites, 0 `existing.` references | ✓ PASS |
| Stale half-null test replaced, not supplemented | `grep -rn 'test_dry_run_of_a_half_null_remint_inverted_window_also_raises' solsys_code/` | 0 hits | ✓ PASS |
| Loader notebook no longer over-claims | JSON inspection of executed outputs | 0 occurrences of "never disagrees"; cell 18 prints the arm-scoped conclusion | ✓ PASS |
| Reason label reached the notebook's executed output | JSON inspection | 12 occurrences of "cannot be proved to have come" | ✓ PASS |
| D-07 lint gate | `pre-commit run ruff --all-files`; `pre-commit run ruff-format --all-files` | Passed; Passed | ✓ PASS |
| Docs gate | `pre-commit run sphinx-build --all-files` | Passed | ✓ PASS |
| Sub-night fields editable in the admin (35-03 t4 / CR-01 reachability) | read of `admin.py:132-166` | no `fields`, no `exclude`, `readonly_fields = ['approval_status']` | ✓ PASS (design), ✗ enables CR-01 |
| **Clear-to-null re-mint** | temporary `TestCase`, real migrated test DB — see Probe Execution | all three shapes STALE, all reported `unchanged=1` | ✗ **FAIL** |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention exists in this repository. For CR-01 I did not rely on the reviewer's reproduction: I wrote my own temporary `django.test.TestCase` (`solsys_code/tests/test_zz_verifier_probe_cr01.py`), ran it with `python manage.py test`, and deleted it — `git status --short -- solsys_code/ docs/` is clean of source changes afterwards. Real `Observatory` (obscode 809, `America/Santiago`), real `sun_event()`, real `reconcile_run()` end to end. True sunset/sunrise for 2026-07-09: `22:06:35.918` / `11:29:46.816` UTC.

| Probe | Shape | Result | Status |
|---|---|---|---|
| A | set/set `23:00`–`05:00` → both nulled (operator reverts to "full night") | after mint `23:00 → 05:00`; `ReconcileResult(created=0, updated=0, unchanged=1, ...)`; after the null edit **`23:00 → 05:00`** | ✗ FAIL — STALE |
| B | half-null `23:00`/None → both nulled | after mint `23:00 → 11:29:46`; `unchanged=1`; after the null edit **`23:00 → 11:29:46`** | ✗ FAIL — STALE start |
| C | set/set → start nulled only, end left unchanged | after mint `23:00 → 05:00`; `unchanged=1`; after the null edit **`23:00 → 05:00`** | ✗ FAIL — STALE start; the branch that passes both `if` statements before returning False |

**Re-verification evidence gate (#3304):** CR-01 is new-scope rather than a carried-forward gap, so it needs deterministic evidence to block. It has two independent grounds. (a) The flagged file `solsys_code/allocation_projector.py` was git-modified after the prior `verified:` timestamp — commits 5eeaa48 (`2026-09-15T18:24:45Z`) and 3635573 (`2026-09-15T18:27:14Z`) both land after `2026-09-15T17:25:47Z` — which alone makes it block unconditionally. (b) I reproduced it in my own process with the executed probe above. It is a 🛑 BLOCKER, not an advisory.

Prior-round probes that this pass re-confirms as CLOSED rather than re-running: PROBE-P1 and PROBE-P6 are now the two named regression tests at `:1223` and `:1286` and both pass; PROBE-P5 is `test_dry_run_of_a_brand_new_line_cannot_predict_a_reconcile_failure` and passes; PROBE-P4 is `test_matching_marker_claimant_has_its_staff_edited_fields_reapplied_from_the_line` and passes; PROBE-P3's refusal is covered by `TestDatabaseScopedIdentityGuard`, 5/5 green. Every probe the prior three passes ran by hand is now a committed test.

### Requirements Coverage

| Requirement | Source plans | Description | Status | Evidence |
|---|---|---|---|---|
| ALLOC-01 | 35-01, 35-03, 35-13, 35-15, 35-16 | Per-night events for resolved classical/awarded allocations; container for queue/class-wide/satellite | ✗ BLOCKED | SC-1 FAILED. Dispatch, fan-out and the container rule are all correct and green; the requirement's own sentence — "projects one per-night sunset→sunrise event per window night" — stops being true for a run whose sub-night fields were cleared to null, and nothing reports the divergence. REQUIREMENTS.md line 113 currently marks this Complete; that mark is premature |
| ALLOC-02 | 35-13, 35-15, 35-16 | Nights keyed by the site-local observing night (Chilean + Australian) | ✓ SATISFIED | SC-2. `_night_span_utc()` / `night_bounds()` untouched this round; the ALLOC-02 edge probe's parity clause is now RESOLVED by 35-16 truth 9 rather than failed |
| ALLOC-03 | 35-15, 35-18 | Handoff on link, restore on unlink, observation's own event untouched | ✓ SATISFIED | SC-3; signals suite green; reconciler notebook re-executed again this round |
| ALLOC-04 | 35-12, 35-14, 35-15, 35-17, 35-18 | `load_telescope_runs` creates or updates a campaign-less `CampaignRun` with a collision-safe `source_identifier`; same per-night events, idempotent | ✓ SATISFIED | SC-4. The real import path is correct and idempotent; the preview's create-arm limitation is now stated in the code, the runbook and the notebook, and pinned by a test. CR-01 is not reachable from this command |
| ALLOC-05 | 35-12, 35-15, 35-17, 35-18 | Cutover has explicit stated sequencing and never leaves a duplicate or orphan | ✓ SATISFIED | SC-5. Identity guard green; re-run gotcha documented and pinned; the two-cause reason vocabulary is consistent across code, runbook and executed notebook output |

All five requirement IDs from the plan frontmatter are accounted for. REQUIREMENTS.md maps exactly ALLOC-01 … ALLOC-05 to Phase 35 and every one appears in at least one plan's `requirements` field — **no orphaned requirements**. Note that REQUIREMENTS.md lines 31 and 113 already mark ALLOC-01 `[x]` / Complete; that status should not stand while SC-1 is failed.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `solsys_code/allocation_projector.py` | 393-404 | A decision guarded on `is not None` for every comparison, so the null case silently returns "nothing to do" — absence of a value treated as absence of a change | 🛑 Blocker | CR-01: permanent stale boundary, reported as `unchanged` |
| `solsys_code/allocation_projector.py` | 359-360 | Two returns expressing one rule (the both-null fast path is subsumed by L366) | ℹ️ Info | IN-01; behaviourally harmless, required by 35-16 prohibition 5 |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 690 | Hardcoded `'ALLOC:'` beside an importable constant | ℹ️ Info | IN-02, carried forward |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 539 | Identity key claimed before the group's transaction commits | ℹ️ Info | IN-03, carried forward |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 149 | `logger` defined, zero call sites; ruff does not flag it | ℹ️ Info | IN-06 |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | cells 16, 18 | Mutate-then-restore without `try/finally` | ℹ️ Info | IN-05; scoped to a `/tmp` scratch DB, so ordering fragility only |

**Debt markers:** `grep -nE "TBD|FIXME|XXX"` over every round-3 file returns two hits, both domain vocabulary rather than debt — `'TBD window'` as a `skipped_reason` value (`test_allocation_projector.py:941`) and its runbook definition (`:1155`), meaning a run with no concrete `window_start`/`window_end`. No `FIXME`, no `XXX`, no unreferenced `TODO` anywhere in the round-3 set.

### Human Verification Required

None as a gate — status is already `gaps_found`, and no truth is left ⚠️ PRESENT_BEHAVIOR_UNVERIFIED: the one behaviour-dependent truth in question (the clear-to-null re-mint transition) was exercised directly by the probe above and FAILED rather than being left unverified. **No prohibition is flagged for human decision this pass** — the one the prior pass flagged (the matching-marker overwrite) was resolved by round 3 along the documented-not-narrowed route the prior pass itself recommended.

Three findings await the developer only in the sense that the developer has already ruled on them: WR-01, WR-02 and WR-03 from 35-REVIEW.md iteration 7 are recorded under `user_deferred` in the frontmatter by explicit user decision and are deliberately absent from the `gaps` list, so a `/gsd-plan-phase 35 --gaps` run stays scoped to the single BLOCKER.

## Gaps Summary

**One blocker. One root cause. Everything round 3 was asked to do, it did.**

The honest read of this pass is that round 3 succeeded and got unlucky in the most useful way possible. Its central move — reverting rather than patching — was the right call and it worked: the four-round half-null bug class is dead, the test that had passed under the defect for four rounds is gone, and three documentation over-claims plus a self-contradictory stderr label are all fixed in the shipped text and in re-executed notebook output rather than in prose about them. I checked each of those against the source and the executed cell outputs, not the SUMMARYs, and each one holds.

CR-01 is the cost of having finally written down the true premise. Round 3's new docstring says, correctly, that nulling a previously-set field leaves the operator's old value in the stored event. That sentence is the whole defect — it just got applied to `_raise_if_set_window_inverted()` and not to `_span_needs_remint()` fifty lines below, which makes the same null-versus-set distinction and gets it wrong. Every comparison in that function is gated on `is not None`, so "the field is now null" is read as "nothing to compare" rather than as "the boundary changed".

**Why this is a BLOCKER where the previous three rounds' findings were not.** The prior passes could each truthfully say no finding could produce a wrong event. This one does: the run row and the calendar it owns end up permanently disagreeing, on a path an operator is explicitly told to use, with no counter, no log line, no exception and no later sweep that can ever discover it. `reconcile_run()` returns `unchanged=1`. Shape C is the one I would emphasise to whoever fixes this — the other field is still set and still matches, so the function reaches the second `if`, evaluates it False, and returns "no re-mint needed" while one of the two boundaries the run declares has genuinely changed. A fix that only handles both-fields-nulled will pass two of the three probes.

**On scope.** WR-01, WR-02 and WR-03 are real, are recorded in `user_deferred` with full detail, and are deliberately not in the `gaps` list. That is the user's explicit call and I think it is the right one: WR-01 and WR-03 are latent or preview-only, and WR-02 is a documentation enumeration. Keeping them out means the next round is one function, one decision and one set of three regression tests — which is the smallest this phase's gap-closure scope has ever been, and after four rounds that is worth more than breadth.

**On the fix.** 35-REVIEW.md CR-01 gives a concrete body, and the review is right that the D-13 objection to calling `sun_event()` here is weaker than it looks: D-13 forbids rewriting a stored boundary *for astropy drift*, and a one-minute tolerance makes drift structurally incapable of triggering the rewrite. The night is about to be deleted and re-minted anyway, which is where `sun_event()` is already called. The alternative — recording mint provenance — is strictly better and would also dissolve the accepted WR-01 limitation, so if anyone has appetite for the larger change, that is the one that pays twice. What must not happen is a third attempt to infer provenance from the stored value; that is what round 2 did and what round 3 correctly deleted.

**Recommendation:** run a fourth gap-closure round, scoped to CR-01 alone. Do not re-open the docs, the loader or the cutover — all three are in a better state than at any prior pass, and reopening them is how the last two rounds acquired their regressions.

---

_Verified: 2026-09-15T19:11:38Z_
_Verifier: Claude (gsd-verifier)_
