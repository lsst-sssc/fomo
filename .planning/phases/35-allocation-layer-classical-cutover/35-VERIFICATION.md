---
phase: 35-allocation-layer-classical-cutover
verified: 2026-09-13T12:40:00Z
status: human_needed
score: 76/76 must-haves verified
covered_files:
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
  - ".planning/REQUIREMENTS.md"
  - "CLAUDE.md"
  - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb"
  - "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
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
  - "solsys_code/tests/test_campaign_approval.py"
  - "solsys_code/tests/test_campaign_reconciler.py"
  - "solsys_code/tests/test_cutover_classical_allocations.py"
  - "solsys_code/tests/test_load_telescope_runs.py"
  - "solsys_code/tests/test_observation_projector_signals.py"
  - "solsys_code/tests/test_reconcile_campaign_runs.py"
  - "solsys_code/tests/test_telescope_runs.py"
  - "solsys_code/tests/test_write_and_reconcile.py"
covered_digest: "v1:sha256:211d217fc834f8b06c0bfad06a8825f4037302a09bde4acb02fade4d3954996c"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Open docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb and read the executed cutover before/after table and the unexplained list."
    expected: "The before/after numbers are real figures from a copy of the developer database (241 -> 233 total, 56 -> 0 RUN:{pk}:{date}, 16 -> 16 bare RUN:{pk} containers, 10 -> 1 blank-url, 0 -> 57 ALLOC:, 159 -> 159 facility-url), the four end-state assertion cells executed without raising, and the single remaining blank-url row (pk=334, title 'tmp') is one you recognise as pre-existing junk rather than a real observing night the cutover failed to convert."
    why_human: "Whether the unexplained list contains only rows an operator recognises is a judgement about what the calendar MEANS on this specific database, not a property any test can assert. Harvested from 35-06-PLAN.md task 3 and 35-07-PLAN.md task 3 <human-check> blocks (deferred to end-of-phase)."
  - test: "Read the 35-06 real-database cutover record in 35-VALIDATION.md ('Manual-Only Verifications' table) and confirm the three-group reconciliation sums."
    expected: "48 rekeyed + 8 legacy_deleted + 0 retired-by-observation = 56, which is exactly the before-count of RUN:{pk}:{date} events; the bare RUN:{pk} container count is unchanged at 16; all 159 facility-url-keyed observation events are reported byte-identical. The 8 containers whose stale pre-Phase-33 title this first post-D-12 sweep corrected read as a pre-existing-staleness correction, not a Phase 35 regression."
    why_human: "The arithmetic is checkable but the judgement -- that the 8 corrected container titles are acceptable churn rather than an unwanted rewrite -- is an operator call about the real calendar."
  - test: "Review the seven judgment-tier prohibitions listed in the 'Prohibitions' section below and confirm each verdict."
    expected: "Each prohibition is upheld by the cited code and test evidence."
    why_human: "unverified-prohibition -- human review recommended. Autonomous verify records a NON-AUTHORITATIVE LLM-judge verdict for judgment-tier prohibitions; these are never silently passed."
---

# Phase 35: Allocation Layer & Classical Cutover — Verification Report

**Phase Goal:** An allocation — classical schedule line, approved submission, TBD/range run, campaign or no campaign — draws its own sunset→sunrise intent nights and hands each night over the moment a real observation links to it; `load_telescope_runs` writes allocations instead of calendar events.

**Verified:** 2026-09-13T12:40:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### ROADMAP Success Criteria (the contract)

| # | Success Criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Allocation with resolved site + awarded window shows one sunset→sunrise event per window night; queue/class-wide/satellite run keeps its single whole-window entry | ✓ VERIFIED | `campaign_reconciler.reconcile_run()` L611-641 dispatches on `telescope_class` → container, `site.observations_type == SATELLITE` → container, `source in {LCO,SOAR,GEMINI,ESO}_QUEUE` → container, else → `allocation_projector.project_allocation()`. Behavioral: `test_allocation_projector.TestEndToEndAllocationNight.test_campaign_less_run_projects_one_alloc_event_per_window_night` and `.test_queue_sourced_run_keeps_its_single_container_never_fanned_out` — PASS (91-test run). |
| 2 | Allocation nights follow the site-local observing night, verified for a Chilean and an Australian site | ✓ VERIFIED | `allocation_projector.retired_nights()` L316 calls `observing_night(window_start, site_zone)` with `ZoneInfo(run.site.timezone)`. Behavioral: `TestAllocationNightBoundary` — 8 tests spanning `America/Santiago` and `Australia/Sydney`, including exact-local-noon, one-second-before-noon, post-local-midnight and UTC-date-differs cases — all PASS. |
| 3 | Linking an `ObservationRecord` removes that night's allocation event and leaves the observation's own event untouched; unlinking restores it; neither transition edits the observation's event | ✓ VERIFIED | `project_allocation()` retire branch L459-467 deletes the `ALLOC:` event + legacy twin; `_sync_observation_attribution()` L306-372 writes ONLY `CalendarEventMeta.run` via `adopt_event_into_run()`/`unlink_event_from_run()` — never a `CalendarEvent` field. Behavioral: `TestObservationHandoff.test_linked_placed_record_retires_its_night`, `.test_unlinking_restores_the_retired_night_with_a_fresh_event`, `TestAttributionBridge.test_d08_round_trip_link_and_unlink_attribution`, `test_observation_projector_signals` (111-test run) — all PASS. |
| 4 | `load_telescope_runs` produces the same per-night calendar as before, by way of an allocation record not a direct event write, and re-running changes nothing | ✓ VERIFIED | `load_telescope_runs.py` imports no calendar writer at all (L1-11); the only write path is `write_and_reconcile_campaign_run()` L292. Behavioral: `test_load_telescope_runs.TestClassicalCalendarUnchangedByCutover.test_calendar_matches_pre_cutover_contract_field_by_field` pins counts/titles/telescope/instrument/spans field-by-field across 4 representative lines; `.test_idempotent_rerun_no_duplicates`, `.test_unchanged_rerun_does_not_update_existing_rows` — all PASS. Executed notebook shows the real second import as `created: 0, unchanged: 2` / `nights ... unchanged: 7`. |
| 5 | After the cutover step, the operator sees one event per night: no duplicate and no orphan from the old `load_telescope_runs` events or the reconciler's `RUN:{pk}:{date}` events | ✓ VERIFIED | Fixture proof: `test_cutover_classical_allocations.TestCutoverSequenceContract.test_cutover_then_sweep_reaches_the_pinned_end_state` — PASS. Real-data proof (executed notebook, scratch copy of the developer DB): total 241→233, `RUN:{pk}:{date}` 56→**0**, bare `RUN:{pk}` 16→16, blank-url 10→1, `ALLOC:` 0→57, facility-url 159→159 with **0** byte-level differences; the one remaining blank-url row is the pk=334 `tmp` row the command explicitly reported as unexplainable (D-18 leaves it byte-identical by design). Reconciliation: 48 rekeyed + 8 legacy_deleted + 0 retired = 56 = the exact before-count. Second sweep: `created: 0, updated: 0, retired: 0, rekeyed: 0, legacy_deleted: 0` — idempotent. |

**ROADMAP score: 5/5**

### Plan-level Must-Have Truths

| Plan | Truths | Status | Evidence |
|------|--------|--------|----------|
| 35-01 | 16 + 1 backstop | ✓ 17/17 VERIFIED | Dispatch, D-10 queue container, both-hemisphere night keying, retire/restore, D-08 both directions, human-confirmation guard (`confirmed_by__isnull=True` filter, L363), terminal-negative keeps retired, update writes only title/description/target_list, zero `sun_event()` on unchanged reconcile, D-14 orphan delete, D-16 in-place re-key preserving pk/start/end, delete-cascade via `writable_allocation_events()` (`models.py` L496-500), single-night/overlapping-window cases, TBD-window and inverted-window skips, order-stable idempotence. Every truth has a named passing test in `test_allocation_projector.py` (56 tests). Backstop closed by the full-suite run below. |
| 35-02 | 6 + 1 backstop | ✓ 7/7 VERIFIED | `TestQueueSourceDoesNotChangeShape` renamed to `TestQueueSourceDispatchesToContainer` with the D-10 premise inverted (`test_campaign_reconciler.py` L177-296 now assert `events.get().url == f'RUN:{run.pk}'`). No test constructs a `RUN:{pk}:{date}` url as an **expected write-path output** — every surviving occurrence is a deliberately pre-seeded legacy fixture for the takeover/delete paths (verified at `test_reconcile_campaign_runs.py` L378-430, L448, L481, L551). Classification table present in 35-02-SUMMARY.md L121-201: 81 classes surveyed, kept 55, migrated 20, retired 6 — **0 without a named destination or reason**. |
| 35-03 | 7 | ✓ 7/7 VERIFIED | `night_start_utc`/`night_end_utc` on `models.py` L282-283; additive migration `0018_campaignrun_night_window_fields.py` with **no `RunPython` step**. `_time_of_day_to_datetime()` L163-176 implements the before-12:00-UTC → next-morning rule. `_span_needs_remint()` L204-224 + `project_allocation()` L498-509 delete-and-re-create rather than rewriting a span. Admin editability spot-checked live: `CampaignRunAdmin.get_form()` exposes both fields (no `fields`/`fieldsets`/`exclude` declared). Behavioral: `TestSubNightWindow` (6 tests) + `TestNoSunEventRecompute` — PASS. |
| 35-04 | 9 | ✓ 9/9 VERIFIED | Both receivers connected in `apps.py` L64-75 with `weak=False` and unique dispatch_uids; **no `post_save` on `CampaignRun`** (only a `pre_delete` at `models.py` L456). `raw=True` early return L586. Cascade guard uses `kwargs['origin']` (L643) — a genuinely correct fix for Django's collector ordering, documented in the docstring. Never-raise contract logs `type(exc).__name__` only (L581, L650). Record-side re-project at `observation_projector.py` L621-630. Behavioral: `test_allocation_projector_signals.py` — 14 tests including `test_neither_receiver_reaches_a_facility`, `test_ready_connected_twice_does_not_double_project`, `test_deleting_the_run_cascades_the_link_without_raising_or_re_projecting`, and 4 never-raise tests — all PASS. |
| 35-05 | 10 | ✓ 10/10 VERIFIED | `load_telescope_runs.py` writes no `CalendarEvent` (no calendar import at all); `source=CLASSICAL_FILE` + `approval_status=APPROVED` hardcoded L252-253; `_CLASSICAL_RUN_STATUS` maps status → `run_status` only, with the status word preserved in `observation_details` L246; `_source_identifier()` includes telescope, instrument, both window dates, both sub-night tokens and the optional proposal L47-77; in-file collision reported on stderr and skipped, never merged L239-244; `--dry-run` previewed with zero writes. Behavioral: `test_load_telescope_runs.py` (25 tests) + `test_telescope_runs.py` — PASS. Executed notebook shows the two-proposal case producing two distinct rows and the token-less duplicate reported as `skipped_collision: 1`. |
| 35-06 | 10 + 1 backstop | ✓ 11/11 VERIFIED | Four-step sequence stated in the runbook L873-895 in order. No `RunPython` in migration 0018. `cutover_classical_allocations.py` re-parses `Source line:` from the event's own description (needs no schedule file), re-keys in place via `update_calendar_event_key_and_fields()` L304, and **contains no `.delete()` call at all** (grep-confirmed; only a docstring mention). D-18: 6 distinct unexplained categories, each left byte-identical and reported, `CommandError` raised → non-zero exit L325-332. Summary prints the final `ALLOC:` count L322. `--dry-run` read-only. Behavioral: `test_cutover_classical_allocations.py` (13 tests incl. `TestNeverDeletesACalendarEvent`, `TestSecondInvocationIsANoOp`, `TestDryRunMatchesRealRun`) — PASS. Backstop closed by the executed real-database notebook diff. |
| 35-07 | 10 | ✓ 10/10 VERIFIED | Runbook: classical-ingest section rewritten for the allocation path with the proposal token and `--dry-run` (L30-90); new cutover section with the four ordered steps (L858-935); `Can I correct a run's source?` carries both new consequences — queue source ⇒ whole-window (L647-654) and *"A `LEGACY` row stays per-night until a human relabels it"* (L656-661); cheat-sheet row for `cutover_classical_allocations` and `--dry-run` on `load_telescope_runs`; troubleshooting entries for the `source_identifier` collision (L1343-1363) and the unexplainable event (L1385). Both notebooks committed with executed output on **every** code cell (14/14 and 16/16, sequential execution counts 1..N). Reconciler notebook catches `CommandError` on both cutover cells so the four assertion cells run. `CLAUDE.md` L136 pairs `cutover_classical_allocations.py` → `reconcile_campaign_runs_demo.ipynb`. Sphinx build clean (gate run). |

**Plan-level score: 71/71**

**Overall score: 76/76 truths verified (0 present, behavior-unverified)**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/allocation_projector.py` | New module owning the `ALLOC:` namespace | ✓ VERIFIED | 662 lines. Imported by `campaign_reconciler.reconcile_run()`, `apps.py`, `observation_projector.py`, `models.py` pre_delete, `cutover_classical_allocations.py`. Wired + data flowing. |
| `solsys_code/tests/test_allocation_projector.py` | ALLOC-01/02/03 coverage | ✓ VERIFIED | 756 lines, 56 tests, all passing. |
| `solsys_code/migrations/0018_campaignrun_night_window_fields.py` | Additive, no data step | ✓ VERIFIED | 23 lines, two `AddField` ops only. |
| `solsys_code/tests/test_allocation_projector_signals.py` | Trigger contract | ✓ VERIFIED | 298 lines, 14 tests, all passing. |
| `solsys_code/management/commands/load_telescope_runs.py` | Writes allocations, not events | ✓ VERIFIED | 329 lines; no calendar-writer import. |
| `solsys_code/management/commands/cutover_classical_allocations.py` | One-time cutover command | ✓ VERIFIED | 332 lines; registered and runnable (exercised by the executed notebook against a scratch DB copy). |
| `solsys_code/tests/test_cutover_classical_allocations.py` | Cutover coverage | ✓ VERIFIED | 530 lines, 13 tests, all passing. |
| `solsys_code/tests/test_campaign_reconciler.py` | Migrated onto `ALLOC:`/D-10 | ✓ VERIFIED | Migrated; passing. |
| `solsys_code/tests/test_reconcile_campaign_runs.py` | Summary counters incl. retired/rekeyed/legacy_deleted | ✓ VERIFIED | Migrated; passing. |
| `docs/runbooks/telescope_runs_calendar.rst` | Cutover + ingest + source sections | ✓ VERIFIED | 1447 lines; wired into `docs/index.rst` toctree; Sphinx build clean. |
| `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` | Executed allocation-path demo | ✓ VERIFIED | 14/14 code cells with output, execution counts 1-14. |
| `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` | Executed cutover before/after diff | ✓ VERIFIED | 16/16 code cells with output, execution counts 1-16; four SC-5 assertions executed and passed. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `campaign_reconciler.reconcile_run()` | `allocation_projector.project_allocation()` | single dispatch seam (D-09) | ✓ WIRED | L638-640; the only non-container branch. |
| `allocation_projector` | `telescope_runs.observing_night()` / `sun_event()` | shared site-local night anchor (D-05) | ✓ WIRED | Imported L57; used at L316, L249-250. |
| `allocation_projector` | `campaign_utils.adopt_event_into_run()` / `unlink_event_from_run()` | only attribution writers (D-08) | ✓ WIRED | L352, L370; no direct `meta.run =` in the bridge. |
| `allocation_projector` | `calendar_utils.insert_or_create_calendar_event()` / `update_calendar_event_key_and_fields()` | only calendar writers | ✓ WIRED | L507, L526-528, L481. |
| `CampaignRun.night_start_utc/night_end_utc` | per-night span computation | `night_bounds()` | ✓ WIRED | L178-200; consumed in `_mint_fields()` L254. |
| `AttributionDecisionView` | `CampaignRunObservation` post_save/post_delete → `project_allocation()` | D-11 | ✓ WIRED | `apps.py` L64-75; proven by `test_confirm_and_undo_each_invoke_project_allocation_exactly_once`. |
| `ObservationRecord.save()` | linked-run re-project | `observation_projector.receiver_on_record_save()` | ✓ WIRED | L621-630. |
| `load_telescope_runs.handle()` | `write_and_reconcile_campaign_run()` → `reconcile_run()` → `project_allocation()` | ALLOC-04 chain | ✓ WIRED | L292; end-to-end proven by the executed notebook (ALLOC: events created from a schedule file). |
| `parse_run_line()` → `ParsedRun.proposal` | `_source_identifier()` | D-01 | ✓ WIRED | L76-77; notebook shows two distinct keys differing only by proposal token. |
| `unique_campaign_run_source_identifier` partial constraint | command's find-or-create lookup | — | ✓ WIRED | `{'source_identifier': key}` lookup L292 / L272. |
| `cutover_classical_allocations` | `telescope_runs.parse_run_line()` / `get_site()` | same parse as ingest | ✓ WIRED | L91 import; L197, L204. |
| `cutover_classical_allocations` | `calendar_utils.update_calendar_event_key_and_fields()` | in-place re-key | ✓ WIRED | L304. |
| `docs/index.rst` toctree | `docs/runbooks/telescope_runs_calendar` | operator-reachable page | ✓ WIRED | Sphinx build clean, no orphan warning. |
| `CLAUDE.md` notebook map | `cutover_classical_allocations.py` → `reconcile_campaign_runs_demo.ipynb` | paired-docs enforceability | ✓ WIRED | CLAUDE.md L136. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `allocation_projector._mint_fields()` | `start`/`end` | `sun_event(run.site, night)` (astropy) via `night_bounds()` | Yes — notebook shows `2026-07-09T22:06:35+00:00` .. `2026-07-10T11:29:46+00:00` for NTT | ✓ FLOWING |
| `allocation_projector` | `target_list` | `run.campaign` (FK) | Yes — notebook shows a named `TargetList` on each night event | ✓ FLOWING |
| `allocation_projector.retired_nights()` | `nights` | `run.observation_links` → `record.scheduled_start/end` | Yes — notebook handoff cell retires `ALLOC:77:2026-09-02` from a real linked record | ✓ FLOWING |
| `load_telescope_runs` | `night_start_utc`/`night_end_utc` | parsed `BoN`/`EoN`/`HHMM` tokens | Yes — notebook partial-night cell shows `06:26:00` end on 3 nights | ✓ FLOWING |
| `cutover_classical_allocations` | run fields | re-parsed `Source line:` in each event's own description | Yes — 3 runs created, 9 events re-keyed against the real DB copy | ✓ FLOWING |
| `reconcile_campaign_runs` summary | `legacy_deleted` | `_detach_stale_family_events()` return | Yes — real sweep reported `legacy_deleted: 8` | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Core allocation/signal/cutover/ingest behavior | `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_cutover_classical_allocations solsys_code.tests.test_load_telescope_runs` | Ran 91 tests in 312.5s — OK | ✓ PASS |
| Migrated reconciler suite (35-02, 35-03) | `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_models solsys_code.tests.test_observation_projector_signals` | Ran 111 tests in 89.7s — OK | ✓ PASS |
| Full label-list suite (closes 3 `verification: backstop` truths) | `python manage.py test $LABELS` (42 modules, `test_views.py` excluded) | **EXIT=0**, Ran 1215 tests in 691.6s — OK (skipped=1) | ✓ PASS |
| `test_views` backstop subset | `python manage.py test ...TestSplitNumberUnitRegex ...TestJPLSBDBQuery` | Ran 40 tests — OK | ✓ PASS |
| Sub-night fields editable in the Django admin with no `admin.py` change | `CampaignRunAdmin(...).get_form(None, obj=None).base_fields` | `night_start_utc: True`, `night_end_utc: True` | ✓ PASS |
| Lint gate (D-07) | `pre-commit run ruff --all-files` | Passed | ✓ PASS |
| Format gate (D-07) | `pre-commit run ruff-format --all-files` | Passed | ✓ PASS |
| Sphinx docs build | `pre-commit run sphinx-build --all-files` | Passed | ✓ PASS |
| Cutover never deletes a `CalendarEvent` | `grep -n "delete" cutover_classical_allocations.py` | Single hit, in a docstring ("Never deletes a...") — zero call sites | ✓ PASS |
| Classical loader has no direct calendar write path | import scan of `load_telescope_runs.py` | No `CalendarEvent` / `calendar_utils` import | ✓ PASS |

### Probe Execution

Not applicable — this project defines no `scripts/*/tests/probe-*.sh` probes and no plan declares one. Equivalent runnable evidence is the Django test suite and the pre-commit gates above.

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | no probes declared or conventional in this repo | ? SKIP |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|----------|
| ALLOC-01 | 35-01, 35-02, 35-03 | Per-night events for resolved-site awarded windows; queue/class-wide/satellite keep one container | ✓ SATISFIED | SC-1 evidence; `reconcile_run()` dispatch L611-641; `TestQueueSourceDispatchesToContainer`. |
| ALLOC-02 | 35-01 | Nights keyed by site-local observing night, verified for Chile and Australia | ✓ SATISFIED | SC-2 evidence; `TestAllocationNightBoundary` 8 tests, both hemispheres. |
| ALLOC-03 | 35-01, 35-02, 35-04 | Linked record ⇒ no allocation event; unlink restores; observation's own event untouched | ✓ SATISFIED | SC-3 evidence; `TestObservationHandoff`, `TestAttributionBridge`, `test_allocation_projector_signals`. |
| ALLOC-04 | 35-03, 35-05, 35-07 | `load_telescope_runs` writes a campaign-less `CampaignRun` with a collision-safe `source_identifier`; same per-night events, idempotent | ✓ SATISFIED | SC-4 evidence; `TestClassicalCalendarUnchangedByCutover`; `_source_identifier()` L47-77. |
| ALLOC-05 | 35-06, 35-07 | Cutover has explicit stated sequencing that never leaves a duplicate or orphan | ✓ SATISFIED | SC-5 evidence; runbook four-step section; `TestCutoverSequenceContract`; real-DB notebook diff. |

**Orphaned requirements:** none. REQUIREMENTS.md maps exactly ALLOC-01..05 to Phase 35, and all five appear in plan frontmatter.

### Prohibitions (judgment tier — NON-AUTHORITATIVE LLM-judge verdict)

> ⚠️ **unverified-prohibition — human review recommended.** These are `verification: judgment` items. The verdicts below are the verifier's own reading of the code, not a test result, and are never a silent pass.

| # | Plan | Prohibition (abbreviated) | Judge verdict | Code evidence |
|---|------|---------------------------|---------------|---------------|
| 1 | 35-01 | Must NOT clear/overwrite a staff-confirmed `CalendarEventMeta` attribution | Upheld | `_sync_observation_attribution()` unlink half filters `confirmed_by__isnull=True` (L363); link half uses `adopt_event_into_run()` which refuses a foreign run and is logged + counted under `blocked` (L351-359). Tests `test_confirmed_attribution_survives_automated_unlink`, `test_foreign_attribution_is_refused_and_counted` — PASS. |
| 2 | 35-01 | Must NOT infer a run's `source` from telescope name, site or event text | Upheld | `reconcile_run()` L619-622 reads the stored `run.source` enum only; no string matching on telescope or title anywhere in the dispatch. |
| 3 | 35-02 | A test covering real behaviour must NOT be deleted without a written destination or reason | Upheld | 35-02-SUMMARY.md classification table L121-201: 81 classes, 6 retired, **0 without a named destination or reason**; each retirement names its counterpart or why the scenario can no longer occur. |
| 4 | 35-04 | A calendar projection fault must NOT abort the staff action that triggered it | Upheld | Both receivers wrap `project_allocation()` in `except Exception` and return (L578-584, L647-653); `observation_projector` L627-630 does the same. Four never-raise tests PASS; one swallow was observed live in the test log (`receiver_on_run_observation_save failed for link pk=1 run pk=1: ValueError`). |
| 5 | 35-05 | Two distinct proposals must NOT be silently merged on a key collision | Upheld | `seen_keys` check L239-244 writes both line numbers to stderr and `continue`s; `skipped_collision` counter surfaced in the summary. Notebook shows the real report. |
| 6 | 35-05 | Classical ingest must NOT re-acquire a direct calendar write path | Upheld | `load_telescope_runs.py` imports no calendar module; sole write is `write_and_reconcile_campaign_run()`. |
| 7 | 35-06 | Cutover must NOT delete/re-key an event it cannot explain, and must NOT reclassify a `LEGACY` row | Upheld | Zero `.delete()` call sites; six unexplained categories each `continue` before any write; the foreign-attribution guard runs **before** the write (L258-264). The command only creates NEW `CLASSICAL:`-keyed runs from blank-url events that never had a run — it never rewrites an existing `LEGACY` row's `source`. Runbook L656-661 states the `LEGACY`-stays-per-night rule to the operator. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/campaign_views.py` | 750 | Success message interpolates `result.skipped_nights`, which no code path assigns any more (the per-night `RUN:` branch that set it was removed by 35-01) | ℹ️ Info | Cosmetic: the site-resolution "no new entries" message now always reads "0 night(s) are already covered…". Already disclosed as a deliberate non-fix in 35-02-SUMMARY.md key-decisions and tracked as a follow-up. Not a must-have, does not affect any success criterion, and no later milestone phase claims it. |

**Debt-marker gate:** clean. Every `TBD` hit across the phase's modified files is domain vocabulary (`'TBD window'` — a run whose dates are To Be Determined), and every `PLACEHOLDER` hit is the tier-3 `Observatory` site classification constant. No `FIXME`, no `XXX`, no un-referenced `TODO`, no `HACK` in any file this phase touched.

**Stub scan:** clean. No `return null`/empty-collection stub returns on any write path; all data flows trace to a real `sun_event()` computation, a model field, or a DB query (see Level-4 table).

### Human Verification Required

#### 1. The cutover's real-database before/after diff reads correctly

**Test:** Open `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` and read the executed cutover before/after table and the unexplained list.
**Expected:** 241 → 233 total, 56 → 0 `RUN:{pk}:{date}`, 16 → 16 bare `RUN:{pk}`, 10 → 1 blank-url, 0 → 57 `ALLOC:`, 159 → 159 facility-url with zero byte-level differences; the four end-state assertion cells executed without raising; the single remaining blank-url row (pk=334, title `'tmp'`) is recognisable as pre-existing junk, not a real observing night the cutover failed to convert.
**Why human:** Whether the unexplained list contains only rows an operator recognises is a judgement about what the calendar *means* on this specific database, not a property a test can assert. (Harvested from the deferred `<human-check>` blocks in 35-06-PLAN.md task 3 and 35-07-PLAN.md task 3.)

#### 2. The three-group reconciliation and the corrected container titles

**Test:** Read the 35-06 real-database cutover record in `35-VALIDATION.md` (Manual-Only Verifications table) and confirm the sums.
**Expected:** 48 rekeyed + 8 legacy_deleted + 0 retired-by-observation = 56 = the exact before-count of `RUN:{pk}:{date}` events; bare containers unchanged at 16; all 159 facility-url observation events byte-identical. The 8 containers whose stale pre-Phase-33 title this first post-D-12 sweep corrected should read as a pre-existing-staleness correction, not a Phase 35 regression.
**Why human:** The arithmetic is checkable, but whether those 8 title corrections are acceptable churn is an operator call about the real calendar.

#### 3. Judgment-tier prohibition review

**Test:** Review the seven prohibitions in the Prohibitions table above and confirm each verdict.
**Expected:** Each prohibition is upheld by the cited code and test evidence.
**Why human:** `unverified-prohibition — human review recommended`. Autonomous verification records a non-authoritative LLM-judge verdict for `verification: judgment` prohibitions; they are never silently absorbed into a passing verdict.

### Gaps Summary

**No gaps.** Every observable truth the phase committed to is backed by code that exists, is substantive, is wired into a real call path, and carries a named passing test — including the three `verification: backstop` truths, which this verification closed independently rather than accepting the orchestrator's claim: the full 42-module label-list suite was re-run under this verifier's own process and returned **EXIT=0, 1215 tests, OK**, plus the 40-test `test_views` backstop subset.

The phase's most falsifiable claim — Success Criterion 5, "no duplicate and no orphan left behind" — is not asserted in prose anywhere it matters: it is pinned by four executed `assert` statements inside a committed notebook that ran against a copy of the real developer database, and by a synthetic-fixture test (`TestCutoverSequenceContract`) that reaches the same end state without needing that database. The one blank-url row that survives (pk=334, `'tmp'`) is not an orphan the cutover missed; it is an event the command deliberately refused to explain, reported by pk on stderr, left byte-identical, and surfaced through a non-zero exit — exactly the D-18 behaviour the phase committed to.

Three attempts to falsify the narrative found nothing:
1. **Did the classical loader secretly keep a calendar write path?** No — the module imports no calendar writer at all, and `TestClassicalCalendarUnchangedByCutover` pins the resulting calendar field-by-field against the pre-cutover contract.
2. **Did the test migration quietly shrink coverage?** No — the classification table accounts for all 81 surveyed classes with 6 retirements, each naming a destination or a reason, and every surviving `RUN:{pk}:{date}` string in the test suite is a deliberately seeded legacy fixture rather than an expected write-path output.
3. **Did the cutover reclassify anything a human owns?** No — it only creates new `CLASSICAL:`-keyed runs from blank-url events that never had a run, and its foreign-attribution guard runs before any write.

The single non-conforming finding is informational: `campaign_views.py:750` still interpolates `result.skipped_nights`, a counter this phase's removal of the per-night `RUN:` branch left permanently 0, so that one site-resolution success message now always reads "0 night(s)". It is cosmetic, pre-disclosed in 35-02-SUMMARY.md as a deliberate non-fix, outside every must-have, and it affects no success criterion.

**Status is `human_needed` rather than `passed` solely because three human-judgement items exist** — two `<human-check>` blocks the planner deliberately deferred to end-of-phase, and the routine review flag on this phase's seven judgment-tier prohibitions. Every automated check in scope is green.

---

_Verified: 2026-09-13T12:40:00Z_
_Verifier: Claude (gsd-verifier)_
