---
phase: 35-allocation-layer-classical-cutover
verified: 2026-09-15T15:51:23Z
status: gaps_found
score: 108/116 must-haves verified
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
  - ".planning/phases/35-allocation-layer-classical-cutover/35-REVIEW.md"
  - "CLAUDE.md"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json"
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
covered_digest: "v1:sha256:7c530f3dc93173aa6d4f077812feafecab0b0a02e2d0680a3bdef9e4559610cc"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 74/76
  previous_verified: 2026-09-15T04:29:09Z
  gap_closure_plans: ["35-08", "35-09", "35-10", "35-11"]
  gaps_closed:
    - "NF-21 — load_telescope_runs.py:338-353 now has a dedicated `except ZoneInfoNotFoundError` clause AHEAD of `(ValueError, Observatory.DoesNotExist)`; one malformed site timezone skips its line and the following line still processes."
    - "NF-22 — allocation_projector.py:690-698 claims `legacy_urls_claimed` before the `_may_write()` check, symmetric with the retired branch; a blocked takeover legacy event is counted once."
    - "NF-23 — campaign_reconciler.py:678 is now `-> tuple[int, int, int, int]`, matching docstring, return statement and 4-way caller unpack."
    - "NF-24 — both stale paired notebooks regenerated with real executed output. `load_telescope_runs_demo.ipynb` 15/15 code cells, 0 null execution counts, new per-line skip-path section whose OUTPUT contains the live `invalid Observatory.timezone` stderr text. `project_observation_calendar_demo.ipynb` 13/13 code cells, 0 nulls, creating-save attribution cell with real `CalendarEventMeta`/`run_id` output."
    - "NF-25 — both `duplicate_identity` message sites (cutover_classical_allocations.py:396-398, :421-424) and all three runbook passages now name the Django-admin `Source line:` edit; the schedule-file remedy survives unchanged in the `load_telescope_runs` section, where it is correct."
    - "IN-01 — test_cutover_classical_allocations.py comment now cites identifiers, not absolute line numbers."
    - "IN-02 — `seen_keys[key] = source_line` moved to cutover_classical_allocations.py:501, after the campaign-mismatch, status-lookup and all-events-foreign checks."
    - "T-35-17 — `reconcile_campaign_runs_demo.ipynb` regenerated and committed: 18/18 code cells, 0 null execution counts. `git status --short` shows no dirty source or notebook file."
  gaps_remaining:
    - "NF-19 (BLOCKER) — only PARTIALLY closed. The guard is now database-scoped, which closes both harms the prior pass reproduced, but its `not in (None, source_line)` predicate makes the no-marker claimant permissive. Re-filed as CR-01 and reproduced end-to-end by 35-REVIEW.md iteration 5."
    - "NF-20 — only PARTIALLY closed. The shared `_raise_if_set_window_inverted()` helper exists and both `_mint_fields()` caller branches call it, but its `or` early-out skips the half-null shape `_span_needs_remint()`'s `and` short-circuit reaches. Re-filed as WR-01; third consecutive iteration of the same dry-run/real divergence (NF-10 -> NF-20 -> WR-01)."
    - "Runbook accuracy — one false operator-facing guarantee was replaced by a narrower one that is still false on the CR-01 path (telescope_runs_calendar.rst:939-949, :1497, :1503)."
  regressions:
    - "WR-02 — load_telescope_runs.py:294-316: the dry-run branch now increments run_created/run_updated/run_unchanged BEFORE the `reconcile_run(existing, dry_run=True)` call at :305 that can raise. The NF-21 handler added by this same gap-closure round then adds `run_skipped += 1` for the same line, so `created + updated + unchanged + skipped != lines processed` on the preview. Introduced/exposed by 35-09's own fix."
    - "WR-03 — campaign_reconciler.py:600-606: NF-23's batch corrected two copies of the `claimed_legacy_urls` contract and left the third (on `_stale_dated_events()`, the function that PERFORMS the exclusion at :615-616) carrying the pre-NF-09 meaning, now also missing the blocked-takeover outcome NF-22 added in this very batch."
gaps:
  - truth: "The cutover never silently mutates a `CampaignRun` it did not create in this invocation — NF-19 case 2: a database claimant of the derived identity key whose stored `Source line:` does not provably match the group's is reported under `duplicate_identity` with a non-zero exit, never a find-and-update (35-08 must_haves truth 2 and prohibition 1; 35-08 truth 12; ROADMAP SC-5's stated sequencing)."
    status: failed
    reason: "CR-01 (BLOCKER, 35-REVIEW.md iteration 5). The database-scoped guard resolves WHO the claimant is by re-parsing the claimant's own `observation_details` — a field `CampaignRunAdmin` leaves fully editable — and puts `None` on the PERMISSIVE side of the predicate. A claimant with no recoverable `Source line:` marker is exactly the row this one-time destructive migration CANNOT prove it owns, and the write it then performs is a full find-and-update of every dispatch-deciding field on an APPROVED run. Reproduced end-to-end by the reviewer against a real Django test database, with exit 0 and empty stderr."
    artifacts:
      - path: "solsys_code/management/commands/cutover_classical_allocations.py"
        issue: "L414-426, predicate at L417: `if existing_source_line not in (None, source_line):` — `None` is permissive. The comment at L411-413 states the rationale explicitly ('a database row with no recoverable Source line: marker has nothing to disagree with, so it is treated as the SAME line rather than rejected'), so this is a deliberate inverted default, not an oversight. REPRODUCTION (35-REVIEW.md, executed probe): a pre-existing classical `CampaignRun` holding the derived key whose `observation_details` a staff member replaced with an ops note ('Rescheduled per PI request; see ticket OPS-4412.'), plus the stranded blank-url events of the CANCELLED counterpart line -> `Done. candidates: 3, groups: 1, runs created: 0, updated: 1, events re-keyed: 3, unexplained: 0`, EXIT 0, `run_status` planned -> cancelled, `observation_details` overwritten to 'Status: cancelled\\nSource line: NTT EFOSC2 cancelled 9-12 July', `target` set to None, and `campaign`/`window_start`/`window_end`/`site`/`site_raw` all overwritten from a line that is not the line that created the run (fields dict at L451-466). The three events are then re-keyed onto it and retitled `[CANCELLED] NTT EFOSC2` by `allocation_night_title()` on the next sweep (step 4 of the documented four-step cutover sequence)."
      - path: "solsys_code/admin.py"
        issue: "L165 `readonly_fields = ['approval_status']` on `CampaignRunAdmin`, and `get_readonly_fields()` (L168+) narrows only `source`. `observation_details` is therefore freely editable in the Django admin for every `CampaignRun`, classical ones included — and this command's OWN remedy text (cutover_classical_allocations.py:396-398, :421-424) sends operators into the admin to edit exactly these rows. It is also written from an arbitrary CSV column by `import_campaign_csv.py:321` and from a `forms.CharField(widget=forms.Textarea)` at `campaign_forms.py:65`. Any of those clears the marker and silently disarms the guard."
      - path: "solsys_code/tests/test_cutover_classical_allocations.py"
        issue: "L1221 `test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` PINS THE WRONG OUTCOME as correct. Verified by running the class in this process: `python manage.py test solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard` -> Ran 4 tests, OK. The test builds the claimant with `observation_details=''` and an `allocation`-shaped line, so no field visibly changes and the destruction is invisible to the assertion. The suite therefore green-lights the defect; 215 green tests are not evidence here."
      - path: "solsys_code/management/commands/cutover_classical_allocations.py"
        issue: "Three documentation statements now assert the guarantee UNCONDITIONALLY and are false for this path: module docstring L48 'This guarantee holds on every invocation, not only the first, because the check reads the database'; `CommandError` L664 'and rewrites no existing CampaignRun (NF-19, 35-REVIEW.md)', which is printed verbatim into the reconciler notebook's committed output."
    missing:
      - "CR-01 fix (one-predicate inversion): change L417 to `if existing_source_line != source_line:` so the command REFUSES what it cannot prove it owns. Give the unprovable case its own actionable reason text rather than folding it into the existing `duplicate_identity` wording — see 35-REVIEW.md section CR-01 for the exact suggested `reason` string, which branches on `existing_source_line is not None` and tells the operator to restore or correct the run's `observation_details` 'Source line:' text in the Django admin."
      - "REPLACE `test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` (test_cutover_classical_allocations.py:1221) with its INVERSE — the no-marker claimant must now be reported, not converted."
      - "ADD the destructive-case regression the current suite is missing: a claimant with `run_status=PLANNED` and a non-marker `observation_details`, a group whose line is the `cancelled` counterpart, asserting a NON-ZERO exit AND that `run_status`, `observation_details` and `target` are byte-identical afterwards. This is the assertion shape that would have caught CR-01."
      - "Correct the three unconditional documentation statements to name the marker requirement: cutover_classical_allocations.py:48 (module docstring) and :664 (CommandError text)."
      - "Re-generate `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` afterwards — its committed output prints the L664 CommandError text verbatim."
  - truth: "The operator runbook's cutover section states only guarantees the shipped code holds (35-07 truth 1; 35-10 must_haves truth 1 and truth 8; 35-10 prohibition 2 'An operator-facing runbook sentence must NOT state a guarantee the shipped code does not hold')."
    status: failed
    reason: "35-10 corrected the PRIOR false sentence and replaced it with a narrower one that is still false on the CR-01 path. This is the same failure mode at one remove: the documentation was updated to match the intended fix rather than the shipped fix."
    artifacts:
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "L938-949: 'the SECOND group is never merged into the first group's run -- this holds on the first invocation and on every re-run, because the guard reads the database rather than only this process's own bookkeeping, so a claimant left behind by an earlier cutover pass or by a prior load_telescope_runs import is caught too' and 'the first group's run and events are converted and left untouched either way, across however many times the command is repeated'. Both halves are false when the claimant's `observation_details` carries no recoverable `Source line:` marker (CR-01), which is reachable through the very admin edit this same passage instructs the operator to perform."
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "L1497 '(the earlier group's run and events are converted and left untouched either way)' and L1503 'and a repeat pass rewrites no existing ``CampaignRun``' — the same unconditional claim, repeated."
    missing:
      - "After the CR-01 predicate inversion lands, narrow all three runbook passages (L938-949, L1497, L1503) to state the marker requirement: the guarantee holds when the claimant's stored `Source line:` is recoverable AND matches, and a claimant with no recoverable marker is REPORTED rather than converted."
      - "Do not soften this into 'usually' or 'in normal operation' — state the predicate, since the operator is the person who can restore the marker."
  - truth: "`reconcile_run(run, dry_run=True)` raises the same `ValueError` the immediately following real run raises, on BOTH `_mint_fields()` caller branches — a `--dry-run` preview never hides a condition that makes the real run raise (35-09 must_haves truth 4, 35-09 flagged assumption ALLOC-02 `unclassified`, 35-09 prohibition 2)."
    status: failed
    reason: "WR-01 (35-REVIEW.md iteration 5), third consecutive iteration of the same defect (NF-10 -> NF-20 -> WR-01). 35-09 built the correct shared helper and wired both branches to it, then gave the helper an early-out whose null-field convention does NOT match the one its own docstring claims to mirror."
    artifacts:
      - path: "solsys_code/allocation_projector.py"
        issue: "L348-349 `if run.night_start_utc is None or run.night_end_utc is None: return` — an OR. The helper's docstring at L332-338 justifies this as 'the same null-field convention `_span_needs_remint()` itself already uses'. It is not the same: `_span_needs_remint()` at L378 short-circuits on `if run.night_start_utc is None and run.night_end_utc is None` — an AND — so for a HALF-NULL run it checks the one set field and can return True, routing the night into the re-mint branch (L725-733), whose `if dry_run:` short-circuit at L729-733 then calls a guard that declines to look. A half-night classical line (`1130-EoN`, `BoN-0230`) produces exactly one set field and one null one via `_window_token_to_time()`, so this is not an exotic shape."
      - path: "solsys_code/allocation_projector.py"
        issue: "L750-754 comment now asserts the opposite without a null-field qualifier: '`_raise_if_set_window_inverted()` is the shared guard also called from the re-mint branch above, so the two passes cannot drift apart on this check.'"
      - path: "solsys_code/tests/test_allocation_projector.py"
        issue: "`test_dry_run_of_a_remint_inverted_window_also_raises` pins only the set/set re-mint case. No test covers the half-null shape, which is why 215 green tests do not catch WR-01."
    missing:
      - "WR-01 fix: change L348 to `if run.night_start_utc is None and run.night_end_utc is None: return`, resolve each boundary independently, and on the RE-MINT branch supply the missing boundary from `existing.start_time`/`existing.end_time` — it was minted from the same deterministic `sun_event()` for the same site and night, so this adds no astropy call and does not breach D-13. 35-REVIEW.md section WR-01 carries the full suggested helper body with an `existing: CalendarEvent | None = None` parameter."
      - "Call it as `_raise_if_set_window_inverted(run, night, existing)` from the re-mint branch (L732) and `_raise_if_set_window_inverted(run, night)` from the create branch (L756)."
      - "Add the half-null twin of `test_dry_run_of_a_remint_inverted_window_also_raises`. The reviewer's executed probe is the fixture: La Silla, one night, `night_start_utc=23:00`/`night_end_utc=None` minted first, then the start edited to `11:30` — after that night's 11:29:46 sunrise. Current behaviour: `reconcile_run(dry_run=True)` -> `ReconcileResult(created=1, retired=1, ...)` and NO error; `reconcile_run(run)` -> `ValueError: Computed an inverted allocation-night span for run pk=1 night=2026-07-09: start=2026-07-10T11:30:00+00:00 >= end=2026-07-10T11:29:46+00:00`."
      - "Narrow the L750-754 comment and the L332-338 helper docstring to say the CREATE-path half-null case is the one shape that remains unpreviewable (it has no stored counterpart and genuinely needs `sun_event()`, which D-13 forbids on a preview) — rather than claiming full parity."
  - truth: "`load_telescope_runs --dry-run` and the real pass report the same outcome for the same line — `created + updated + unchanged + skipped` equals `lines processed` on both."
    status: failed
    reason: "WR-02 (35-REVIEW.md iteration 5). 35-09's NF-21 fix added a per-line `except ZoneInfoNotFoundError` handler that increments `run_skipped`, but the dry-run branch had already folded the line into `run_unchanged` before the call that raises. The real branch does not have this shape, so one line produces two different outcomes."
    artifacts:
      - path: "solsys_code/management/commands/load_telescope_runs.py"
        issue: "L294-316: the dry-run branch increments `run_created`/`run_updated`/`run_unchanged` at L297-302 from `preview_campaign_run_action()`, BEFORE calling `reconcile_run(existing, dry_run=True)` at L305. If that call raises (NF-21's `ZoneInfoNotFoundError`, `sun_event()`'s own `ValueError` for a blank timezone or a polar site, or WR-01's inverted-span `ValueError`), the handlers at L338-353 / L354-357 then add `run_skipped += 1` for the same line. The real branch at L317-337 increments only after the `transaction.atomic()` block has returned, so a failure there yields `skipped` alone."
      - path: "solsys_code/tests/test_load_telescope_runs.py"
        issue: "`TestMalformedTimezoneSkipsOneLine` exercises only the REAL path, which is why this survived the NF-21 fix."
    missing:
      - "WR-02 fix: mirror the real branch — compute the action, run the preview reconcile, and fold `run_created`/`run_updated`/`run_unchanged` only once BOTH have succeeded. 35-REVIEW.md section WR-02 carries the suggested re-ordered block."
      - "Add a dry/real parity assertion to `TestMalformedTimezoneSkipsOneLine`: same fixture, both modes, identical `(created, updated, unchanged, skipped)` tuple. Reproduced current behaviour — preview: `Done (dry run). lines processed: 1, created: 0, updated: 0, unchanged: 1, skipped: 1`; real: `Done. lines processed: 1, created: 0, updated: 0, unchanged: 0, skipped: 1`."
  - truth: "Every copy of the `claimed_legacy_urls` contract in the codebase states the same, current meaning (the NF-17 contract 35-09 was closing)."
    status: partial
    reason: "WR-03 (35-REVIEW.md iteration 5). NF-17 corrected two of three copies. The third sits on `_stale_dated_events()` — the function whose `stale_dated.exclude(url__in=claimed_legacy_urls)` IS the exclusion — so a reader of the function that does the work gets the superseded contract while a reader of its caller gets the current one."
    artifacts:
      - path: "solsys_code/campaign_reconciler.py"
        issue: "L600-606 Arg docstring still says 'already decided the fate of THIS call (a takeover re-key or a retirement delete) -- excluded here so a ``dry_run`` preview never double-counts'. Both halves are stale: NF-09 widened the set to blocked and human-declined urls and NF-22 (closed in this same batch) widened it again to the blocked-takeover url, so the parenthetical enumeration is missing two of four outcomes; and the 'so a dry_run preview never double-counts' clause is the exact claim NF-17 was filed to correct — the two sibling copies at L746-756 and `allocation_projector.py:576-595` both now state that the exclusion is LOAD-BEARING IN REAL MODE for a blocked or declined url."
    missing:
      - "Replace campaign_reconciler.py:600-606 with the same wording the two corrected copies use, adding the blocked-takeover outcome NF-22 introduced. 35-REVIEW.md section WR-03 carries the exact replacement text."
deferred: []
advisory:
  - finding: "WR-04 — `observation_projector.py:647-658`'s swallowed `campaign_run_links` lookup error has no `transaction.atomic()` savepoint, while the module's own `project_record()` spends twelve lines (L339-352) explaining that the savepoint, not the `except`, is what makes such a catch safe. The comment's claim 'same guarantee project_record() gets above' is inaccurate. On PostgreSQL (the production target CLAUDE.md names) a failed statement poisons the surrounding transaction, so swallowing hands the caller an `InFailedSqlTransaction` on its next statement — the save is relocated, not protected."
    category: architectural
    reason: "The reviewer's executed probe showed NO failure on this project's current SQLite backend ({'count': 0}, the caller's next query succeeded), so the exposure is backend-dependent and not reproducible here today. The file was last modified by 24875bf, which PREDATES the prior verification's 2026-09-15T04:29:09Z timestamp — `git log --since` over solsys_code shows observation_projector.py unchanged by the 35-08..35-11 gap-closure round. Under the convergence evidence gate this is a new-scope finding with no deterministic evidence of failure, so it is recorded here rather than blocking. Resolution: wrap the lookup in `with transaction.atomic():` with the `except` OUTSIDE the `with`, correct the comment to name the savepoint as the mechanism, and either extend the same treatment to the three sibling catches (observation_projector.py:664-675, allocation_projector.py:871-880 and :952-961) or note in each why it is unnecessary there."
    evidence_status: "probe executed, no failure reproduced on SQLite; PostgreSQL exposure argued from backend semantics, not demonstrated"
  - finding: "IN-01 — cutover_classical_allocations.py:524 re-runs `CampaignRun.objects.filter(source_identifier=key).first()` and rebinds `existing_run`, a name already bound at L414 by the CR-01 guard. Redundant query plus a shadowed name that invites a reader to assume the guard's result is reused."
    category: other
    reason: "Cosmetic; no behavioural difference. Fix: delete L524 and use the outer binding, or rename (`claimant_run` for the guard, `existing_run` for the write) if a fresh read is wanted for race safety. Worth folding into the CR-01 fix since it touches the same predicate's surroundings."
    evidence_status: "none provided — static observation, no failing behaviour"
  - finding: "IN-02 — cutover_classical_allocations.py:648 hardcodes `url__startswith='ALLOC:'` while the module imports `allocation_night_url`/`allocation_night_title` from `solsys_code.allocation_projector`, which defines `ALLOC_URL_NAMESPACE = 'ALLOC:'` for exactly this purpose. A namespace rename would silently make the final summary count zero."
    category: other
    reason: "Fix: import `ALLOC_URL_NAMESPACE` alongside the existing names and use it at L648 and L656."
    evidence_status: "none provided — latent, no current failure"
  - finding: "IN-03 — cutover_classical_allocations.py:501 claims `seen_keys[key]` after the convertibility checks (IN-02's fix, correctly landed) but still BEFORE the group's `try: with transaction.atomic():` at L521. A group whose transaction rolls back entirely (the `except` at L628, which marks every event `_OTHER` and writes nothing) keeps the key claimed, so a sibling group sharing that key is reported under `duplicate_identity` naming a line that converted nothing."
    category: other
    reason: "Residual of the same operator-confusion IN-02 described. Fix: move `seen_keys[key] = source_line` to just after the `runs_created += group_created` fold at L624-627, i.e. only once the group's writes have committed."
    evidence_status: "none provided — reasoned from control flow, no probe run"
behavior_unverified_items: []
coincidental_reliance_items:
  - truth: "The cutover's identity guard rejects a colliding second group (35-08 truths 1-3, `TestDatabaseScopedIdentityGuard`, 4 tests, OK)."
    reason: fixture-only
    harden: "Every passing case in `TestDatabaseScopedIdentityGuard` constructs its claimant through `_make_pre_existing_claimant()`, which supplies an `observation_details` string the fixture itself controls. Production claimants come from `import_campaign_csv.py:321` (an arbitrary CSV column), `campaign_forms.py:65` (a free-text Textarea) and the Django admin (`admin.py:165` leaves the field editable) — none of which guarantees a `Source line:` marker. The guard's correctness silently depends on a precondition only the fixture establishes. Promote it into a declared precondition: either make `observation_details` non-editable for classical rows, or store the identity provenance in a field the operator cannot overwrite, or (the CR-01 fix) refuse whenever the precondition cannot be proven."
---

# Phase 35: Allocation Layer & Classical Cutover — Verification Report

**Phase Goal:** An allocation — classical schedule line, approved submission, TBD/range run, campaign or no campaign — draws its own sunset→sunrise intent nights and hands each night over the moment a real observation links to it; `load_telescope_runs` writes allocations instead of calendar events.

**Verified:** 2026-09-15T15:51:23Z
**Status:** gaps_found
**Re-verification:** Yes — second gap-closure round check. Prior pass `gaps_found` 74/76 at 2026-09-15T04:29:09Z; plans 35-08 through 35-11 executed since (commits `868caa6`, `9e555eb`, `0eceeb5`, `eb79263`, `0b7599f`, `45b38a7`, `33f0a61`, `11b14da`, `4cee1f1`, `5eb2718`).

---

## What changed since the prior pass

| Event | Outcome |
|-------|---------|
| 35-08 (`868caa6`, `9e555eb`) | NF-19 guard made database-scoped; NF-25 remedy made actionable; IN-02 claim ordering moved; IN-01 comment fixed. **Guard is database-scoped — but its permissive `None` branch re-opens the harm. See CR-01.** |
| 35-09 (`0eceeb5`, `eb79263`, `0b7599f`) | NF-21 **closed**; NF-22 **closed**; NF-23 **closed**; NF-20 shared helper built and both branches wired. **Helper's `or` early-out misses the half-null shape. See WR-01. New divergence introduced on the loader's dry-run counters. See WR-02.** |
| 35-10 (`45b38a7`, `33f0a61`, `11b14da`) | Runbook remedies corrected; reconciler notebook regenerated (18/18 cells, T-35-17 closed); CLAUDE.md now maps `allocation_projector.py`. **Runbook's replacement guarantee is still false on the CR-01 path.** |
| 35-11 (`4cee1f1`, `5eb2718`) | NF-24 **closed, both halves.** Loader demo 15/15 cells with live `invalid Observatory.timezone` stderr in real output; projector demo 13/13 cells with creating-save attribution output; SCHED-06 baseline rewritten. |
| 35-REVIEW.md iteration 5 (`894c982`) | **`status: issues_found` — 1 BLOCKER (CR-01) + 4 WARNINGs + 3 INFO.** 7 of 9 prior findings genuinely closed, 2 only partially closed with the surviving half re-filed under a new id. |

**Working tree is clean** of source and notebook modifications (`git status --short` shows only `.planning/` and untracked scratch files) — T-35-17's uncommitted hand-edit from the prior pass is gone.

### Convergence evidence gate (applied, is_re_verification = true)

Every file carrying a blocking finding this pass was **modified by the gap-closure round itself**, so each is in-contract under the gate and blocks unconditionally regardless of evidence:

| Finding | File | Modified since 2026-09-15T04:29:09Z? | Also a carried-forward gap? | Verdict |
|---|---|---|---|---|
| CR-01 | `cutover_classical_allocations.py` | Yes — `868caa6` 06:25, `9e555eb` 06:26 | Yes — prior gap 1's first `missing` item is the NF-19 guard | 🛑 Blocks (regression + carried-forward + executed reproduction) |
| WR-01 | `allocation_projector.py` | Yes — `eb79263` 07:23, `0b7599f` 07:42 | Yes — prior gap 3 named NF-20 | 🛑 In-contract (WARNING severity) |
| WR-02 | `load_telescope_runs.py` | Yes — `0eceeb5` 06:52 | Regression introduced by the NF-21 fix | 🛑 In-contract (WARNING severity) |
| WR-03 | `campaign_reconciler.py` | Yes — `0b7599f` 07:42 | Yes — prior gap 3 named NF-23 | 🛑 In-contract (WARNING severity) |
| WR-04 | `observation_projector.py` | **No** — last touched by `24875bf`, before the prior pass | No | 📋 Advisory — new-scope, probe reproduced no failure on SQLite |
| IN-01/02/03 | `cutover_classical_allocations.py` | Yes | No | 📋 Advisory — INFO severity, no failing behaviour |

---

## Goal Achievement

### ROADMAP Success Criteria (the contract)

| # | Success Criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Per-night events for a resolved-site awarded window; queue/class-wide/satellite keeps one whole-window entry | ✓ VERIFIED | `reconcile_run()` dispatches on `dispatches_per_night(run)`; container branch otherwise. 215 tests across the five phase modules pass (35-REVIEW.md verification method); `TestDatabaseScopedIdentityGuard` re-run in this process — 4 tests, OK. |
| 2 | Nights follow the site-local observing night, Chile and Australia | ✓ VERIFIED | `ZoneInfo(run.site.timezone)`; `_time_of_day_to_datetime()` resolves against the site's own UTC span with no hour threshold. Unchanged since the prior pass; `TestAllocationNightBoundary` passes both hemispheres. |
| 3 | Linking an `ObservationRecord` retires that night; unlinking restores it; the observation's own event is untouched | ✓ VERIFIED | Attribution writes route only through `adopt_event_into_run()`/`unlink_event_from_run()`. Now also shown in executed notebook output: `project_observation_calendar_demo.ipynb`'s creating-save cell prints `CalendarEventMeta.run_id: 74`, `allocation nights left: 2`. |
| 4 | `load_telescope_runs` produces the same per-night calendar, via an allocation record, idempotently | ✓ VERIFIED | The module imports no calendar writer; the only write path is `write_and_reconcile_campaign_run()`. NF-21 closed — the dedicated `ZoneInfoNotFoundError` clause at L338-353 restores the per-line skip-and-log invariant, verified by execution (a three-line file with a typo'd timezone on line 2 reports `skipped: 2`, processes line 3, leaves no `CampaignRun` behind). **Reporting defect WR-02 does not falsify this criterion** — the per-night calendar and idempotence are correct; the dry-run *summary* disagrees with the real run. |
| 5 | After the cutover, one event per night — no duplicate, no orphan | ✗ **FAILED** | The end-state COUNTS from the real-database run stand (241→233, `RUN:{pk}:{date}` 56→0, `ALLOC:` 0→57, human-confirmed in 35-UAT.md). But ALLOC-05's criterion is about the **stated sequencing**, and CR-01 shows an operator following that documented sequencing silently merging two distinct schedule lines into one `CampaignRun` — the allocation line's run keeps the cancelled line's identity, its three nights are re-keyed onto it, and the next sweep retitles them `[CANCELLED] NTT EFOSC2`. Exit 0, `unexplained: 0`, empty stderr. The prior pass recorded this as a caveat on a VERIFIED criterion; after a full gap-closure round failed to close it, it is recorded as a failure. |

**ROADMAP score: 4/5**

### Plan-level Must-Have Truths

| Plan | Truths | Status | Evidence |
|------|--------|--------|----------|
| 35-01 | 16 + 1 backstop | ✓ 17/17 VERIFIED | Unchanged since the prior pass; the 35-09 edits to `allocation_projector.py` were additive guard/counter changes, and the module's test file passes. |
| 35-02 | 6 + 1 backstop | ✓ 7/7 VERIFIED | Unchanged. |
| 35-03 | 7 | ✓ 7/7 VERIFIED | Unchanged. `0018_campaignrun_night_window_fields.py` remains two `AddField` ops, no `RunPython`. |
| 35-04 | 9 | ✓ 9/9 VERIFIED | Unchanged. |
| 35-05 | 10 | ✓ 10/10 VERIFIED | Unchanged. |
| 35-06 | 10 + 1 backstop | ✓ 11/11 VERIFIED (literal) | `TestSecondInvocationIsANoOp` passes, so truth 5's literal wording still holds. The CR-01 harm lives on the re-run path after an *unsuccessful* pass, which no 35-06 truth names — recorded as a blocker anti-pattern and against 35-08's truths, not here. |
| 35-07 | 10 | ✓ 9/10 | **Truth 5 now CLOSED** — `load_telescope_runs_demo.ipynb` regenerated at `4cee1f1` with the per-line skip-path section, 15/15 code cells, 0 null execution counts, real `invalid Observatory.timezone` stderr in output. **Truth 1 still FAILED** — the runbook's one false guarantee was replaced by a narrower one that is still false (L938-949, L1497, L1503). |
| 35-08 | 12 | ✗ 10/12 | Truths 1, 3-11 verified (`TestDatabaseScopedIdentityGuard` 4 tests OK; remedy text actionable; `seen_keys` moved to L501; `--dry-run` parity preserved; no absolute line numbers in comments). **Truth 2 FAILED** — 'never a silent find-and-update' holds only when a `Source line:` marker is recoverable. **Truth 12 FAILED** — 'a re-run after an unresolved `duplicate_identity` changes no `CampaignRun` field' is falsified by the executed reproduction. **Truth 11 is VERIFIED literally and is the defect's specification**: it states the permissive `None` predicate as the intended behaviour, flagged at plan time as an unresolved ALLOC-01 `empty` assumption. The code matches the plan; the plan encoded the wrong default. That is why 215 green tests and a literally-satisfied must-have coexist with a live BLOCKER. |
| 35-09 | 10 | ✗ 8/10 | Truths 1, 2, 3, 5, 6, 8, 9, 10 verified (NF-21 dedicated clause with correct order and provably-bound `site`; `transaction.atomic()` rollback intact; NF-22 claim before `_may_write()` at L690-698; NF-23 four-tuple at L678). **Truth 4 FAILED** — the shared guard skips the half-null shape. **Flagged assumption ALLOC-02 `unclassified` FAILED** — it asserts the helper matches `_span_needs_remint()`'s null-field convention; the two differ (`or` at L348 vs `and` at L378), proven by reproduction. |
| 35-10 | 9 | ✗ 7/9 | Truths 2-7 and 9 verified (all three remedy passages corrected; the `load_telescope_runs` schedule-file remedy survives; NF-21 bullet names `ZoneInfoNotFoundError` at L1399; notebook 18/18 cells, 0 nulls; `CLAUDE.md` now maps `allocation_projector.py`). **Truth 1 FAILED** — the runbook's `duplicate_identity` passage states a guarantee the shipped code does not hold. **Truth 8 FAILED** — 'the runbook's "safe to repeat" promise is now backed by 35-08's guard' is false on the no-marker path. |
| 35-11 | 9 | ✓ 9/9 VERIFIED | Both notebooks regenerated with real executed output and non-null execution counts on every code cell; creating-save attribution cell present and run inside a rolled-back `atomic()`; `NonSiderealTargetFactory` used per CLAUDE.md; `sched06-baseline.json` rewritten. |

**Plan-level score: 104/111**

**Overall score: 108/116 truths verified (0 present, behavior-unverified)**

### Advisory (New Scope, Unevidenced)

New-scope findings from Step 7 with no deterministic evidence — reported, not blocking, do not revert a completed must-have.

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | WR-04 — no savepoint around `observation_projector.py:647-658`'s swallowed lookup error | architectural | New-scope: the file was NOT modified since the prior `verified:` timestamp (`24875bf` predates it), and the reviewer's executed probe reproduced no failure on this project's SQLite backend. The PostgreSQL exposure is argued from backend semantics, not demonstrated. |
| 2 | IN-01 — `existing_run` re-queried and shadowed at `cutover_classical_allocations.py:524` | other | INFO severity, no behavioural difference. Fold into the CR-01 fix. |
| 3 | IN-02 — `'ALLOC:'` literal at `cutover_classical_allocations.py:648` instead of `ALLOC_URL_NAMESPACE` | other | Latent; no current failure. |
| 4 | IN-03 — `seen_keys[key]` claimed at L501, still before the group's `transaction.atomic()` at L521 | other | Reasoned from control flow; no probe run. |

### Deferred Items

None. `roadmap.analyze` shows Phases 36 (Unattended Operation — SCHED-08/09/10, DISCOVER-01) and 37 (Status Vocabulary, Public Tallies — STATUS/TALLY/UNUSED/GAPB) as the only later phases in this milestone, and neither goal nor any of their ten success criteria covers the cutover identity guard, the dry-run inversion guard, the loader's dry-run counters or the reconciler docstring. All five gaps are this phase's own work.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/allocation_projector.py` | Module owning the `ALLOC:` namespace | ⚠️ VERIFIED (defect) | 966 lines. Imported and used by `campaign_reconciler`, `apps.py`, `observation_projector`, `models.py`, `cutover_classical_allocations`. Wired, data flowing. NF-22 closed at L690-698. **WR-01 open at L348-349.** |
| `solsys_code/management/commands/cutover_classical_allocations.py` | One-time cutover command | ✗ **DEFECTIVE** | 666 lines; registered and runnable. Guard is now database-scoped (L414) — a real improvement — but **CR-01 (BLOCKER) open at L417**, plus IN-01/02/03. |
| `solsys_code/tests/test_cutover_classical_allocations.py` | Cutover coverage incl. the NF-19 hole | ⚠️ INCOMPLETE — **pins the defect** | `TestDatabaseScopedIdentityGuard` added, 4 tests, all pass (re-run in this process). The four cases (merge / differing line / same line / no marker) are the right partition — but the fourth case's assertion pins the PERMISSIVE outcome as correct, and its `observation_details=''` fixture makes the destruction invisible. |
| `solsys_code/management/commands/load_telescope_runs.py` | Writes allocations, not events | ⚠️ VERIFIED (defect) | 377 lines; no calendar import. **NF-21 closed** — dedicated `except ZoneInfoNotFoundError` at L338-353, correctly ordered ahead of `(ValueError, Observatory.DoesNotExist)` at L354. **WR-02 open at L294-316.** |
| `solsys_code/campaign_reconciler.py` | Migrated onto `ALLOC:`/D-10 | ⚠️ VERIFIED (doc defect) | 902 lines. **NF-23 closed** — L678 is `-> tuple[int, int, int, int]`, matching docstring, return and 4-way unpack. **WR-03 open at L600-606.** |
| `solsys_code/observation_projector.py` | Record-save trigger, D-11 re-project | ⚠️ VERIFIED (advisory) | 811 lines. F-34-1 guard present at L647-658; WR-04 recorded as advisory (see gate table). |
| `solsys_code/admin.py` | Staff surface for `CampaignRun` | ⚠️ **CR-01 PREMISE** | L165 `readonly_fields = ['approval_status']`; `get_readonly_fields()` withholds only `source`. `observation_details` is freely editable — this is what makes CR-01 reachable by an ordinary staff action. |
| `docs/runbooks/telescope_runs_calendar.rst` | Cutover + ingest + source sections | ✗ **INACCURATE** | 1565 lines, toctree-wired at `docs/index.rst:24`. NF-25 and NF-21 passages **corrected**. **L938-949, L1497, L1503 state a guarantee the code does not hold.** |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | Executed allocation-path demo | ✓ **VERIFIED — was STALE** | 15/15 code cells, 0 null execution counts, all with output. New "Per-line skip paths" section; output contains the live `invalid Observatory.timezone` stderr text. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | D-11 trigger demo | ✓ **VERIFIED — was STALE** | 13/13 code cells, 0 nulls, all with output. Creating-save attribution cell present with real `CalendarEventMeta`/`run_id` output, run inside a rolled-back `atomic()`. |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Executed cutover before/after diff | ✓ **VERIFIED — was DIRTY** | 18/18 code cells, 0 null execution counts. T-35-17 closed: the working tree is clean. **Caveat:** its committed output prints `cutover_classical_allocations.py:664`'s CommandError text verbatim, including the now-false "rewrites no existing CampaignRun" clause — it must be regenerated after the CR-01 fix. |
| `CLAUDE.md` | Paired-docs notebook map | ✓ VERIFIED | Now maps `solsys_code/allocation_projector.py` into `reconcile_campaign_runs_demo.ipynb` (`11b14da`), closing the hole that let this phase's central module ship with no mapped demo notebook. Breach-history entry added. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `campaign_reconciler.reconcile_run()` | `allocation_projector.project_allocation()` | single dispatch seam (D-09) | ✓ WIRED | Unchanged; guarded by `dispatches_per_night(run)`. |
| `allocation_projector` | `telescope_runs.observing_night()` / `sun_event()` | shared site-local night anchor (D-05) | ✓ WIRED | `ZoneInfo(run.site.timezone)`. |
| `allocation_projector` | `campaign_utils.adopt_event_into_run()` / `unlink_event_from_run()` | only attribution writers (D-08) | ✓ WIRED | No direct `meta.run =` in the bridge. |
| `_raise_if_set_window_inverted()` | both `if dry_run:` short-circuits in `project_allocation()` | one shared guard (35-09 key link) | ⚠️ **WIRED, INCOMPLETE** | Called from L732 (re-mint) and L756 (create) — the wiring 35-09 promised is real. But the helper's L348 `or` early-out declines to check the half-null shape the re-mint caller reaches, so the link carries no signal on exactly the operator edit it was built for (WR-01). |
| `cutover_classical_allocations` group loop | `CampaignRun.objects.filter(source_identifier=key)` | database-scoped identity guard (35-08 key link) | ⚠️ **WIRED, UNSOUND** | L414 — the lookup IS now against the database, the same place `insert_or_create_campaign_run()` matches. The defect is downstream at L417: the guard then derives its authority from `observation_details`, an admin-editable free-text field, and treats absence as consent (CR-01). |
| `_extract_source_line()` | `CampaignRun.observation_details` | reuse the existing parser (35-08 key link) | ⚠️ **WIRED, WRONG TRUST BOUNDARY** | The parser is reused with no new parsing code, as planned. The problem is what the *result* authorises: `None` (parse found nothing) is routed to the same branch as a positive match. |
| `load_telescope_runs.handle()` | `write_and_reconcile_campaign_run()` → `reconcile_run()` → `project_allocation()` | ALLOC-04 chain | ✓ WIRED | Inside `transaction.atomic()`; per-line handlers now cover both `ZoneInfoNotFoundError` and `(ValueError, Observatory.DoesNotExist)`. |
| `_DUPLICATE_IDENTITY` message | runbook's three remedy passages | one shared operator phrase (35-10 key link) | ✓ WIRED | "edit the affected events' description `Source line:` text … in the Django admin" appears at `cutover_classical_allocations.py:396-398`, `:421-424`, `telescope_runs_calendar.rst:939-949`, `:1454-1461`, `:1494-1503`, and in the notebook's committed output. |
| `docs/index.rst` toctree | `runbooks/telescope_runs_calendar` | operator-reachable page | ✓ WIRED | `docs/index.rst:24`. |
| `CLAUDE.md` notebook map | `allocation_projector.py` → `reconcile_campaign_runs_demo.ipynb` | paired-docs enforceability | ✓ WIRED | `CLAUDE.md:133-135`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `allocation_projector._mint_fields()` | `start`/`end` | `sun_event(run.site, night)` via `night_bounds()` | Yes — real astropy spans in the executed notebooks | ✓ FLOWING |
| `cutover_classical_allocations` | `existing_source_line` | `_extract_source_line(existing_run.observation_details)` — an **admin-editable** `TextField` also written by `import_campaign_csv.py:321` and `campaign_forms.py:65` | Yes, but the value is **operator-controlled and may legitimately be `None`**, and `None` is routed to the permissive branch. This is the CR-01 seam at the data-flow level. | ⚠️ **FLOWING FROM AN UNTRUSTED SOURCE** |
| `allocation_night_title()` | event title prefix | `run.run_status` → `_RUN_STATUS_CALENDAR_PREFIX` | Yes — **and this is what makes CR-01 calendar-visible**: the silently flipped `run_status` retitles every one of that run's nights `[CANCELLED] …` on the next sweep (step 4 of the documented cutover sequence) | ✓ FLOWING |
| `load_telescope_runs` dry-run summary | `run_unchanged` / `run_skipped` | `preview_campaign_run_action()` folded at L297-302, then the exception handler at L338-353 | Both increment for the same line — the preview total no longer sums to `lines processed` (WR-02) | ⚠️ **DOUBLE-COUNTED** |
| `campaign_views.py:750` | `result.skipped_nights` | nothing — no code path assigns it any more | No | ⚠️ STATIC (cosmetic, pre-disclosed in 35-02-SUMMARY.md) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| The NF-19 guard's four cases | `python manage.py test solsys_code.tests.test_cutover_classical_allocations.TestDatabaseScopedIdentityGuard` | **Ran 4 tests in 0.031s — OK** | ✓ PASS — **and this is the problem.** The fourth case, `test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` (L1221), asserts the CR-01 behaviour is correct. A green run here certifies the defect. |
| Full phase suite | `python manage.py test` over the five phase test modules (35-REVIEW.md verification method, executed by the reviewer) | **215 tests, all passing** | ✓ PASS (not re-run here — one full-suite run per verification, and the reviewer's run is the one) |
| CR-01 reproduction | Reviewer's executed probe against a real Django test DB (35-REVIEW.md, PROBE-A) | `Done. candidates: 3, groups: 1, runs created: 0, updated: 1, unchanged: 0, events re-keyed: 3, unexplained: 0`; **exit 0**; `run_status` planned → cancelled; `observation_details` `'Rescheduled per PI request; see ticket OPS-4412.'` → `'Status: cancelled\nSource line: NTT EFOSC2 cancelled 9-12 July'` | ✗ **FAIL — reproduced** |
| WR-01 reproduction | Reviewer's executed probe (35-REVIEW.md, PROBE-D), La Silla half-null run | sunset/sunrise `2026-07-09 22:06:35.918` / `2026-07-10 11:29:46.816`; `reconcile_run(dry_run=True)` → `ReconcileResult(created=1, retired=1, …)` **no error**; `reconcile_run(run)` → `ValueError: Computed an inverted allocation-night span for run pk=1 night=2026-07-09: start=2026-07-10T11:30:00+00:00 >= end=2026-07-10T11:29:46+00:00` | ✗ **FAIL — reproduced** |
| WR-02 reproduction | Reviewer's executed probe (35-REVIEW.md, PROBE-B) | dry run `unchanged: 1, skipped: 1` vs real run `unchanged: 0, skipped: 1` — one line, two outcomes | ✗ **FAIL — reproduced** |
| CR-01 predicate present in current code | `grep -n "not in (None, source_line)" cutover_classical_allocations.py` | `417:                if existing_source_line not in (None, source_line):` | ✗ FAIL (confirmed) |
| WR-01 convention mismatch present | read `allocation_projector.py:348` vs `:378` | L348 `if run.night_start_utc is None or run.night_end_utc is None:` vs L378 `if run.night_start_utc is None and run.night_end_utc is None:` | ✗ FAIL (confirmed) |
| `observation_details` admin-editable | read `admin.py:165` + `get_readonly_fields()` | `readonly_fields = ['approval_status']`; `get_readonly_fields()` withholds only `source` | ✗ FAIL — CR-01 premise confirmed |
| Other writers of `observation_details` | `grep -n observation_details import_campaign_csv.py campaign_forms.py` | `import_campaign_csv.py:321` (`row.get('Observation Details', '')`), `campaign_forms.py:65` (`forms.CharField(widget=forms.Textarea)`) | ✗ FAIL — three independent paths can clear the marker |
| NF-21 closure | read `load_telescope_runs.py:338-357` | Dedicated `except ZoneInfoNotFoundError` clause placed AHEAD of `(ValueError, Observatory.DoesNotExist)`; stderr names the obscode and offending timezone | ✓ PASS |
| NF-22 closure | read `allocation_projector.py:690-698` | `legacy_urls_claimed.add(legacy_url)` precedes the `_may_write()` check | ✓ PASS |
| NF-23 closure | `sed -n '676,679p' campaign_reconciler.py` | `-> tuple[int, int, int, int]` | ✓ PASS |
| WR-03 stale docstring | read `campaign_reconciler.py:600-606` | Still says "(a takeover re-key or a retirement delete)" and "so a ``dry_run`` preview never double-counts" | ✗ FAIL (confirmed) |
| WR-04 no savepoint | read `observation_projector.py:647-658` | Bare `try:` with no `with transaction.atomic():`; comment claims "same guarantee project_record() gets above" | ⚠️ confirmed present, no failure reproducible on SQLite |
| Notebook execution counts | JSON scan of all three pre-executed notebooks | loader 15/15 cells 0 nulls; projector 13/13 0 nulls; reconciler 18/18 0 nulls; every code cell has output | ✓ PASS |
| Notebook content currency | JSON scan for new-cell markers in source and output | loader: "Per-line skip" in source, `invalid Observatory.timezone` in OUTPUT; projector: "creating save" in source, `CalendarEventMeta`/`run_id` in OUTPUT | ✓ PASS |
| Working tree clean of source edits | `git status --short` | Only `.planning/config.json` (M) and untracked `.gsd/`, scratch JSON, phase-34 review files. No source or notebook file dirty. | ✓ PASS |
| Content fingerprint | `gsd_run query verification.fingerprint` | `7c530f3d…` vs the prior report's `b6a64521…` — changed, as expected for a gap-closure round | ℹ️ Recorded |

### Probe Execution

Not applicable — this project defines no `scripts/*/tests/probe-*.sh` probes and no plan declares one. The Django test runner and the code reviewer's executed reproduction probes (written under `solsys_code/tests/`, run, then deleted; `git status --short` confirms no source file was modified by that review) are the equivalent runnable evidence.

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | no probes declared or conventional in this repo | ? SKIP |

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `test_cutover_classical_allocations.py` | ALLOC-04, ALLOC-05 | yes (`TestDatabaseScopedIdentityGuard` 4/4 pass) | 0 | no | Value + behavioural (second-invocation) | ⚠️ **INSUFFICIENT for the requirement it claims.** The class is the first in this phase to test a SECOND invocation — genuine progress. But `test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` (L1221) asserts the wrong outcome, and its `observation_details=''` fixture is chosen such that no field visibly changes, so the destructive write it permits is invisible to every assertion in the case. No test asserts `run_status`/`observation_details`/`target` are byte-identical after a rejected pass. |
| `test_allocation_projector.py` | ALLOC-01, ALLOC-02 | yes | 0 | no | Value + exception-raising | ⚠️ **INCOMPLETE.** `test_dry_run_of_a_remint_inverted_window_also_raises` covers the set/set shape only; no half-null twin, which is the WR-01 hole. |
| `test_load_telescope_runs.py` | ALLOC-04 | yes | 0 | no | Value + behavioural (following line still processed) | ⚠️ **INCOMPLETE.** `TestMalformedTimezoneSkipsOneLine` correctly asserts the *following* line still processed — the right invariant. But it exercises only the real path, so WR-02's dry/real counter divergence is untested. |
| `test_observation_projector_signals.py` | ALLOC-03 | yes | 0 | no | Behavioural | ⚠️ Advisory. `test_linked_run_lookup_raising_does_not_abort_the_records_own_save_or_projection` patches `campaign_run_links` with a `SimpleNamespace` whose `select_related` raises a Python-constructed `OperationalError` that never reaches the database — it proves the `except` clause exists, not that the transaction survives (WR-04). |

**Disabled tests on requirements:** 0 — no `@skip`, `@unittest.skip`, `self.skipTest` or `@expectedFailure` in any phase test module.
**Circular patterns detected:** 0.
**Insufficient assertions:** 3 → ⚠️ WARNING, and one of them (`test_cutover_classical_allocations.py:1221`) actively pins a BLOCKER as correct behaviour, which is the single most important finding of this audit.

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|----------|
| ALLOC-01 | 35-01, 35-02, 35-03, 35-09 | Per-night events for resolved-site awarded windows; queue/class-wide/satellite keep one container | ✓ SATISFIED | SC-1; `dispatches_per_night()` dispatch; 215 tests green; NF-22 closed. |
| ALLOC-02 | 35-01, 35-09 | Nights keyed by site-local observing night, Chile and Australia | ✓ SATISFIED | SC-2; `_time_of_day_to_datetime()` span-nearest rule; boundary tests both hemispheres. WR-01 is a *preview-parity* defect on this surface, not a wrong-night defect — the real run computes the correct span and raises when the operator's edit is inverted. |
| ALLOC-03 | 35-01, 35-02, 35-04, 35-11 | Linked record ⇒ no allocation event; unlink restores; observation's own event untouched | ✓ SATISFIED | SC-3; signal and attribution-bridge tests; now also shown in executed notebook output (creating-save attribution). |
| ALLOC-04 | 35-03, 35-05, 35-07, 35-08, 35-09, 35-10, 35-11 | `load_telescope_runs` writes a campaign-less `CampaignRun` with a collision-safe `source_identifier`; same per-night events, idempotent | ✗ **BLOCKED** | SC-4's calendar and idempotence properties hold, and NF-21/NF-24 are closed. **But the collision-safety of the `source_identifier` contract is not upheld end to end:** a `CampaignRun` created by `load_telescope_runs` is exactly one of the claimant shapes CR-01 lets the cutover silently find-and-update — the guard's own comment at L405-406 names "from load_telescope_runs" as a case it exists to catch, and the no-marker branch lets it through. WR-02 additionally breaks the command's dry/real summary parity. |
| ALLOC-05 | 35-06, 35-07, 35-08, 35-10 | Cutover has explicit stated sequencing that never leaves a duplicate or orphan | ✗ **BLOCKED** | SC-5 failed. The stated sequencing — including the `CommandError`'s own "it is safe to repeat … rewrites no existing CampaignRun" at L664 and the runbook's L938-949 — drives an operator into the reproduced CR-01 silent merge, with exit 0 and empty stderr. Two schedule lines collapse into one run and the calendar retitles the allocated nights `[CANCELLED]`. |

**Orphaned requirements:** none. REQUIREMENTS.md maps exactly ALLOC-01..05 to Phase 35 (L113-117, all currently marked "Complete" — **these three rows should not be treated as authoritative while ALLOC-04 and ALLOC-05 are blocked**), all five appear in plan frontmatter, and no plan claims an ID not in REQUIREMENTS.md.

### Prohibitions

Judgment-tier. The seven carried forward from earlier passes were human-confirmed in 35-UAT.md test 3 (pass). The gap-closure plans added new ones, and **three are violated**:

| # | Plan | Prohibition | Verdict | Evidence |
|---|------|-------------|---------|----------|
| 1 | 35-08 | "The cutover must NOT silently mutate a `CampaignRun` it did not create in this invocation." (Explicitly widened from the Phase 35 original in response to the prior pass's own Prohibitions note.) | ✗ **VIOLATED** | CR-01's executed reproduction: `run_status` planned → cancelled, `observation_details` overwritten, `target` set to `None`, exit 0, `unexplained: 0`, empty stderr. The prohibition was widened correctly and the code still breaches it. |
| 2 | 35-08 | "A `--dry-run` must NOT exit 0 over a fixture the immediately following real run rejects." (cutover) | ✓ Upheld | `--dry-run`/real parity preserved on every cutover fixture. |
| 3 | 35-08 | "The cutover must NOT delete a `CalendarEvent` on any path." | ✓ Upheld | No `.delete()` call site in the command. |
| 4 | 35-08 | "An operator-facing remedy string must NOT name an artifact this command never reads." | ✓ Upheld | NF-25 closed — all sites name the Django admin edit. |
| 5 | 35-09 | "A `--dry-run` preview must NOT hide a condition that makes the immediately following real run raise." | ✗ **VIOLATED** | WR-01's executed reproduction: dry run returns `ReconcileResult(created=1, retired=1, …)` with no error; the real run raises `ValueError`. Third consecutive iteration of this exact prohibition failing. |
| 6 | 35-09 | "One bad schedule line must NOT abort the whole `load_telescope_runs` import." | ✓ Upheld | NF-21 closed; verified by execution (line 2 skipped, line 3 processed). |
| 7 | 35-09 | "A single calendar row must NOT be reported under a counter more than once." | ⚠️ **Partially violated** | Upheld for the calendar row (NF-22 closed). Violated for a schedule *line* in the loader's dry-run summary (WR-02: `unchanged: 1` + `skipped: 1` for one line). |
| 8 | 35-09 | "A function's signature annotation must NOT contradict its own docstring or its return statement." | ✓ Upheld | NF-23 closed at L678. |
| 9 | 35-10 | "An operator-facing runbook sentence must NOT state a guarantee the shipped code does not hold." | ✗ **VIOLATED** | `telescope_runs_calendar.rst:938-949`, `:1497`, `:1503` — the replacement guarantee is still false on the CR-01 path. |
| 10 | 35-10 / 35-11 | "A paired notebook must NOT be hand-patched to repair execution counts or output." | ✓ Upheld | All three notebooks show contiguous non-null execution counts and real output; T-35-17's hand-edit is gone from the working tree. |
| 11 | 35-11 | "A `Target` fixture must NOT use `SiderealTargetFactory`." | ✓ Upheld | `NonSiderealTargetFactory` in the projector demo's new cell, per CLAUDE.md. |

### Decision Coverage

Non-blocking gate. The phase's CONTEXT.md decisions (D-05, D-08, D-09, D-10, D-11, D-13, D-14, D-16, D-18, D-19) are each traceable to a shipped artifact and, for the gap-closure round, to a commit subject: D-13's "no `sun_event()` on a preview" is explicitly reasoned about in both branches of the WR-01 fix site (`allocation_projector.py:332-338`, `:739-754`), and D-18's six unexplained categories are intact in the cutover. **No decision vanished during execution.** Recorded for drift tracking only; no status impact.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/management/commands/cutover_classical_allocations.py` | 417 | Trust boundary inverted — an admin-editable free-text field's *absence* is treated as consent to a destructive find-and-update (CR-01) | 🛑 **Blocker** | Reproduced end-to-end. In-contract under the evidence gate: carried-forward gap AND the file was modified by this round. An operator following the documented cutover sequence silently loses a run's lifecycle state and staff note, with exit 0. |
| `solsys_code/tests/test_cutover_classical_allocations.py` | 1221 | A passing regression test that pins a BLOCKER as correct behaviour | 🛑 **Blocker** | Self-evidencing: the test name states the asserted outcome. This is why 215 green tests, a passing UAT and a clean security audit coexist with a live blocker — and why the fix must REPLACE this test, not just add one. |
| `solsys_code/allocation_projector.py` | 348-349 | `or` where the convention it claims to mirror uses `and` — guard declines to check the shape its caller reaches (WR-01) | ⚠️ Warning | Dry run previews clean; real run raises. Third consecutive iteration (NF-10 → NF-20 → WR-01). The L750-754 comment now asserts full parity, so the code and its own documentation disagree. |
| `solsys_code/management/commands/load_telescope_runs.py` | 294-316 | Counter folded before the call that can fail; the failure handler then counts the same line again (WR-02) | ⚠️ Warning | `created + updated + unchanged + skipped != lines processed` on the preview. An operator reading the dry run believes a run will be left untouched when the line will in fact be dropped. Regression introduced by this round's own NF-21 fix. |
| `solsys_code/campaign_reconciler.py` | 600-606 | Third copy of a contract corrected in two places — on the function that actually performs the exclusion (WR-03) | ⚠️ Warning | Superseded meaning, missing two of four outcomes including the one NF-22 added in this same batch. |
| `solsys_code/observation_projector.py` | 647-658 | Swallowed DB error with no savepoint, comment claims a guarantee it does not have (WR-04) | 📋 **Advisory** | New-scope (file unmodified since the prior pass) and no failure reproducible on SQLite. See Advisory table. |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 524 | `existing_run` re-queried and shadowed (IN-01) | 📋 Advisory | Redundant query; invites a reader to assume the guard's result is reused. |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 648 | `'ALLOC:'` literal instead of `ALLOC_URL_NAMESPACE` (IN-02) | 📋 Advisory | A namespace rename would silently zero the summary count. |
| `solsys_code/management/commands/cutover_classical_allocations.py` | 501 vs 521 | `seen_keys[key]` claimed before the group's transaction (IN-03) | 📋 Advisory | A rolled-back group keeps the key claimed; a sibling is reported naming a line that converted nothing. |
| `solsys_code/campaign_views.py` | 750 | Interpolates `result.skipped_nights`, which no code path assigns | ℹ️ Info | Cosmetic; pre-disclosed in 35-02-SUMMARY.md as a deliberate non-fix. Carried forward unchanged. |

**Debt-marker gate: clean.** Every `TBD` across this phase's modified files is domain vocabulary (`'TBD window'` — a run whose dates are To Be Determined: `campaign_reconciler.py:240`, `test_allocation_projector.py:941`, `telescope_runs_calendar.rst:1104`). No `FIXME`, `XXX`, `HACK`, `PLACEHOLDER`, or un-referenced `TODO` in any file this phase touched.

**Stub scan: clean.** No stub returns on any write path; every rendered value traces to a real `sun_event()` computation, a model field, or a DB query.

### Human Verification Required

None. Every failure this pass is code-evidenced: four by executed reproduction against a real Django test database (CR-01, WR-01, WR-02, WR-04's scope), the rest by direct reading of the cited lines and one named test run in this process. The three human items from the 2026-09-13 pass remain closed (35-UAT.md, 3/3 pass), and no truth is behavior-unverified.

---

## Gaps Summary

**The gap-closure round did substantial, genuine work — and the phase still does not pass, because the two findings it only half-closed are the two that carry the phase's data-integrity contract.**

What is genuinely fixed, verified against the code rather than the summaries: NF-21 (the loader's exception routing, with correct clause ORDER and a provably-bound `site`), NF-22 (blocked takeover counted once), NF-23 (the four-tuple annotation), NF-25 (actionable remedy text in all five places), IN-01, IN-02, T-35-17, and both halves of NF-24 — the two stale paired notebooks are regenerated with real executed output, and `CLAUDE.md` now maps `allocation_projector.py`, closing the enforcement hole that let this phase's central module ship unmapped. Three prior gaps are closed; the working tree is clean; 215 tests pass; both ruff gates pass.

**The BLOCKER is not closed, and its shape changed rather than shrank.**

NF-19's fix correctly moved the identity guard from an in-process `dict` to a database lookup (`cutover_classical_allocations.py:414`). That closes both harms the prior verification reproduced. It then resolves *who* the database claimant is by re-parsing the claimant's own `observation_details` and puts `None` on the permissive side of the predicate at L417 — with a comment (L411-413) stating the rationale outright: *"a database row with no recoverable `Source line:` marker has nothing to disagree with, so it is treated as the SAME line rather than rejected."*

That reasoning is inverted for a one-time destructive migration. A row with no marker is precisely the row this command **cannot prove it owns**, and the write it then performs is a full find-and-update of every dispatch-deciding field on an APPROVED run. `observation_details` is not an internal field: `CampaignRunAdmin` (`admin.py:165`) leaves it fully editable, `import_campaign_csv.py:321` writes it from an arbitrary CSV column, `campaign_forms.py:65` exposes it as a free-text Textarea — and this command's own remedy text sends operators into the admin to edit exactly these rows. Reproduced against a real test database: a staff-edited run flips `run_status` planned → cancelled, loses its ops note, has `target` nulled and `campaign`/`window_start`/`window_end`/`site`/`site_raw` overwritten from a line that did not create it, with three events re-keyed onto it — **exit 0, `unexplained: 0`, empty stderr.** `allocation_night_title()` then retitles those nights `[CANCELLED] NTT EFOSC2` on step 4 of the documented sequence.

**Why the green suite did not catch it:** `test_pre_existing_claimant_with_no_recoverable_source_line_still_converts` (`test_cutover_classical_allocations.py:1221`) *pins this outcome as correct*. I re-ran `TestDatabaseScopedIdentityGuard` in this process — 4 tests, OK. The four cases (merge / differing line / same line / no marker) are the right partition; the bug is that the fourth case asserts the wrong answer, and its `observation_details=''` fixture is chosen such that no field visibly changes, making the destruction invisible to every assertion. The fix must **replace** that test, not add beside it.

**Why this kept recurring, for both open findings.** 35-08's plan frontmatter carries the permissive predicate as a *flagged, unresolved* must-have (ALLOC-01 `empty`): *"is treated as the SAME line and allowed to proceed — the permissive predicate 35-REVIEW.md NF-19 prescribes."* 35-09's carries the parallel one for WR-01: *"matching `_span_needs_remint()`'s own null-field convention."* **Both flagged assumptions are false, and both were implemented faithfully.** The executor did what the plan said; the plan encoded an inverted default and a mis-read convention (`or` at `allocation_projector.py:348` versus `and` at `:378` — not the same convention at all). Three truths in this report are literally VERIFIED while specifying the defect. The closure plan should treat the flagged assumptions themselves as the artifacts to correct, not only the code.

WR-01 is now the third consecutive iteration of one sentence — NF-10, then NF-20, now WR-01: *a dry run previews clean and the immediately following real run raises.* 35-09 built the right shared helper and wired both callers to it; the helper then declines to look at the half-null shape (`1130-EoN`, `BoN-0230` — an ordinary half-night classical line), which is exactly the shape `_span_needs_remint()` routes into the branch that calls it. The good news is that the fix is now cheap and local: on the re-mint branch the missing boundary is already on `existing`, minted from the same deterministic `sun_event()` for the same site and night, so no astropy call and no D-13 breach.

Two further in-contract warnings, both self-inflicted by this round: **WR-02**, where the NF-21 handler this round added now double-counts a line the dry-run branch had already folded into `unchanged` (the preview total no longer sums to `lines processed`); and **WR-03**, where NF-23's batch corrected two copies of the `claimed_legacy_urls` contract and left the third — on `_stale_dated_events()`, the function that *performs* the exclusion — carrying the superseded meaning, now also missing the blocked-takeover outcome NF-22 added in this very batch.

**Requirement impact:** ALLOC-01, ALLOC-02 and ALLOC-03 are satisfied. **ALLOC-04 and ALLOC-05 are blocked** — a `load_telescope_runs`-created run is one of the claimant shapes CR-01 lets the cutover silently overwrite (the guard's own comment names it as a case it exists to catch), and ALLOC-05's "explicit, stated sequencing" is the surface the operator follows into the merge. REQUIREMENTS.md currently marks all five "Complete" at L113-117; those two rows should not be treated as authoritative until CR-01 is closed.

**What passes:** the allocation layer itself. SC-1 through SC-4 are solid, every key link is wired with real data flowing, the debt-marker and stub gates are clean, no test is disabled or circular, requirement traceability is complete with no orphans, all three paired notebooks are current and executed, the runbook is toctree-wired, and the real-database cutover numbers remain human-confirmed. The defects are concentrated in one predicate, one boolean operator, one counter ordering, and one docstring.

---

_Verified: 2026-09-15T15:51:23Z_
_Verifier: Claude (gsd-verifier)_
