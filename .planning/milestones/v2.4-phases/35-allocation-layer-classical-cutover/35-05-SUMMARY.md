---
phase: 35-allocation-layer-classical-cutover
plan: 05
subsystem: calendar-sync
tags: [django, campaign-reconciler, allocation-projector, classical-ingest, management-command]

requires:
  - phase: 35-allocation-layer-classical-cutover
    provides: "plan 35-01's ALLOC: namespace and project_allocation()/reconcile_run() dispatch seam; plan 35-03's night_start_utc/night_end_utc sub-night fields and night_bounds()"
provides:
  - "telescope_runs._PROPOSAL_TOKEN / _resolve_proposal() / ParsedRun.proposal -- the optional bracketed [proposal] token, consumed before every other grammar in the classical run line (D-01)"
  - "campaign_utils.preview_campaign_run_action(run, fields) -- the run-level twin of calendar_utils.preview_calendar_event_action(), used by load_telescope_runs --dry-run"
  - "load_telescope_runs rewritten as an allocation writer: one campaign-less CampaignRun per schedule line, keyed on a deterministic, collision-safe source_identifier, routed through write_and_reconcile_campaign_run(); no direct CalendarEvent write remains in this command"
affects: [35-06, 35-07]

actuals:
  tokens: 19332
  tasks: 3
  commits: 4
  plan_head_before: 24b9fe9992f3d79630b22a3abd21a9ad635090c2

tech-stack:
  added: []
  patterns:
    - "Run-level dry-run twin: preview_campaign_run_action() mirrors calendar_utils.preview_calendar_event_action()'s exact getattr(obj, f) != v comparison, so a --dry-run preview can never disagree with what insert_or_create_campaign_run() would actually do -- the same discipline the event-level preview already established, applied one layer up."
    - "Deterministic-key ingest, not proximity-match ingest: the pre-cutover command matched an existing night by (telescope, instrument, start_time) within a drift tolerance, because the computed sun-event time WAS the identity key. The rewritten command's identity lives entirely in CampaignRun.source_identifier (computed from the line's own fields, never from sun_event() output), so the allocation projector's per-night matching is by (run.pk, night) -- a sun_event() drift between imports can no longer even reach the identity decision, let alone create a near-duplicate row."
    - "Within-file collision set, not database collision handling: a schedule file's own repeated source_identifier is caught with a plain in-memory dict BEFORE any write is attempted, leaving the database's unique_campaign_run_source_identifier partial constraint (35-01/Phase 31) as the second, race-safe layer -- the command layer's job is a clear operator-facing message naming both line numbers, not correctness enforcement."

key-files:
  created: []
  modified:
    - solsys_code/telescope_runs.py
    - solsys_code/campaign_utils.py
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
    - solsys_code/tests/test_write_and_reconcile.py
    - .planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md

key-decisions:
  - "Proposal-token syntax: a square-bracketed token, [<proposal>], consumed by _resolve_proposal() at the very top of parse_run_line() -- before _resolve_status() -- so its contents can never reach the status, month, leftover-token or trailing-window matchers. Square brackets are unambiguous against every other grammar already in the file (round brackets for status, bare words for status/months, no bracket at all for the BoN/EoN/HHMM window token)."
  - "source_identifier is built from the run's OWN STORED window_start/window_end (post night-convention adjustment), never the line's raw day range -- so a re-import recomputes a byte-identical key from the row it is about to match, and the ESO noon-to-noon adjustment cannot desynchronize the key from the row across re-imports."
  - "The command keeps its own copy of _iter_run_nights() (unchanged) rather than importing it from anywhere else -- it is the one piece of night-convention logic the command still owns, since the allocation projector needs the resolved window_start/window_end as inputs, not the raw day range."
  - "Task 2's own Command-level behavior tests (the plan's <behavior> Tests 1-7) live in test_write_and_reconcile.py, per the plan's own <files> tag for that task, using a dedicated NTT (obscode 809) Observatory fixture -- test_load_telescope_runs.py's 24-test suite is Task 3's exclusive scope, so Task 2 could not touch it without pre-empting Task 3's classification work."
  - "TestClassicalCalendarUnchangedByCutover derives every expected start/end time by calling tr.sun_event() and applying the BoN/EoN/HHMM rule inline in the test (a small test-local helper, not a call into production code), so the assertion states the field-by-field contract rather than echoing allocation_projector.night_bounds()'s own implementation."
  - "Documented divergence (plan-flagged, not silently chosen): a cancelled classical run's event description now also carries the shared writer's 'Run status: Cancelled' line, because allocation descriptions are composed through the same event_description() helper every other campaign-run event uses. Titles, spans, telescope, instrument and event counts are unchanged from pre-cutover output -- only this one description line is new, and only for CANCELLED/WEATHER_TECH_FAILURE runs."

requirements-completed: [ALLOC-04]

coverage:
  - id: D1
    description: "parse_run_line() accepts an optional bracketed [proposal] token, disjoint from every other grammar in the file and consumed first; malformed forms (empty, duplicated, unbalanced) raise ValueError"
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_telescope_runs.py (7 new tests, 41 total)"
        status: pass
      - kind: other
        ref: "python -c ... parse_run_line() verify command -> '0110.C-0234 BoN 0626 allocation cancelled None'"
        status: pass
    human_judgment: false
  - id: D2
    description: "load_telescope_runs.Command.handle() rewritten around write_and_reconcile_campaign_run(): each schedule line creates/updates one campaign-less CampaignRun keyed on a deterministic, collision-safe source_identifier; run_status derived from the parser status (D-03); --dry-run added; zero direct CalendarEvent writes remain"
    requirement: ALLOC-04
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_write_and_reconcile.py#TestPreviewCampaignRunAction (3 tests) and #TestLoadTelescopeRunsWritesAllocations (7 tests)"
        status: pass
      - kind: other
        ref: "ast-based zero-calendar_utils-import check and _CLASSICAL_RUN_STATUS/KNOWN_STATUSES set-equality check -> '0 3 1 1'"
        status: pass
    human_judgment: false
  - id: D3
    description: "The 24-test ingest suite migrated onto the allocation path (kept/migrated/retired, each retirement named and replaced) plus a new field-by-field regression pinning the per-night calendar unchanged from pre-cutover output for 4 representative lines"
    requirement: ALLOC-04
    verification:
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_load_telescope_runs (25 tests)"
        status: pass
      - kind: integration
        ref: "python manage.py test solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_telescope_runs solsys_code.tests.test_write_and_reconcile (86 tests, one invocation)"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-13
status: complete
---

# Phase 35 Plan 05: Classical Ingest Rewritten as an Allocation Writer Summary

**`load_telescope_runs` no longer writes `CalendarEvent` rows directly: each schedule line now creates or updates one campaign-less `CampaignRun` keyed on a deterministic, collision-safe `source_identifier` (with an optional `[proposal]` token), and the allocation projector draws the exact same per-night calendar it drew before.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-09-13T06:21:00Z (approx, from the prior plan's completion commit)
- **Completed:** 2026-09-13T06:46:00Z
- **Tasks:** 3
- **Files modified:** 7 (6 source/test files + 1 validation doc)

## Accomplishments

- `solsys_code/telescope_runs.py`: `_PROPOSAL_TOKEN` (a bracketed `[proposal]` token, disjoint from every other grammar in the file), `_resolve_proposal()` (consumed FIRST in `parse_run_line()`, before `_resolve_status()`), and `ParsedRun.proposal: str | None = None`. Empty, duplicated and unbalanced-bracket forms raise `ValueError` rather than being guessed at, matching the module's existing discipline.
- `solsys_code/campaign_utils.py`: `preview_campaign_run_action(run, fields) -> str`, the run-level twin of `calendar_utils.preview_calendar_event_action()`, used by the rewritten command's `--dry-run`.
- `solsys_code/management/commands/load_telescope_runs.py` rewritten around `write_and_reconcile_campaign_run()`: `_CLASSICAL_RUN_STATUS` maps the parser's `KNOWN_STATUSES` onto `CampaignRun.RunStatus` (D-03); `_source_identifier()` builds the deterministic key (D-01) from the run's own stored `window_start`/`window_end`, the two sub-night tokens, and the optional proposal; `_window_token_to_time()` converts a window token to a stored sub-night `time`. A within-file `source_identifier` collision is reported on stderr naming both line numbers and counted under `skipped_collision`, never merged. New `--dry-run` flag: previews the run-level action via `preview_campaign_run_action()` and the night-level tallies via `reconcile_run(dry_run=True)`, or predicts night counts from the window length for a run that does not exist yet. `_resolve_window_time()`, `_CLASSICAL_STATUS_PREFIX`, `_START_TIME_MATCH_TOLERANCE` and the `calendar_utils` import are removed -- zero direct `CalendarEvent` writes remain (verified by an AST-level zero-`calendar_utils`-`ImportFrom` check).
- `solsys_code/tests/test_write_and_reconcile.py`: 3 new tests for `preview_campaign_run_action()`, plus a new `TestLoadTelescopeRunsWritesAllocations` class (7 Command-level behavior tests: run fields, `ALLOC:` event count, cancelled title prefix, sub-night storage, collision reporting, `--dry-run`, and the `WEB`-source guard) -- per the plan's own `<files>` tag for Task 2.
- `solsys_code/tests/test_load_telescope_runs.py`: all 24 existing tests classified and revised (see Per-Test Classification below); one test retired with a named replacement; one test's assertion inverted to state the new contract; a new `TestClassicalCalendarUnchangedByCutover` class pins the per-night calendar (url set, title, telescope, instrument, `start_time`, `end_time`) for 4 representative lines against expected values computed independently in the test from `sun_event()`.
- `.planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md`: `35-05-02 · 35-05-03` row's Status column flipped to green.
- 86 tests green across this plan's three modules in one invocation (`test_telescope_runs` 41, `test_write_and_reconcile` 20, `test_load_telescope_runs` 25). Both `pre-commit run ruff` / `ruff-format` gates clean over this plan's own six source/test files.

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): add failing tests for the proposal token** - `9a285c8` (test)
1. **Task 1 (GREEN): an optional proposal token in the classical run line** - `b81927b` (feat)
2. **Task 2: load_telescope_runs writes allocations, not calendar events** - `6ec5955` (feat, includes its own tests per the plan's `<files>` tag)
3. **Task 3: revise the ingest suite and pin "the same calendar as before"** - `1dd8b53` (test)

**Plan metadata:** committed alongside this SUMMARY.

## Files Created/Modified

- `solsys_code/telescope_runs.py` - `_PROPOSAL_TOKEN`, `_resolve_proposal()`, `ParsedRun.proposal`
- `solsys_code/campaign_utils.py` - `preview_campaign_run_action()`
- `solsys_code/management/commands/load_telescope_runs.py` - full rewrite: `_CLASSICAL_RUN_STATUS`, `_source_identifier()`, `_window_token_to_time()`, `--dry-run`; removed `_resolve_window_time()`, `_CLASSICAL_STATUS_PREFIX`, `_START_TIME_MATCH_TOLERANCE`, the `calendar_utils` import
- `solsys_code/tests/test_telescope_runs.py` - 7 new proposal-token tests
- `solsys_code/tests/test_write_and_reconcile.py` - 3 new `preview_campaign_run_action` tests, 7 new Command-level allocation-writing tests
- `solsys_code/tests/test_load_telescope_runs.py` - full revision: 24 tests classified (kept/migrated/retired), 1 new `TestClassicalCalendarUnchangedByCutover` class
- `.planning/phases/35-allocation-layer-classical-cutover/35-VALIDATION.md` - `35-05-02 · 35-05-03` Status flipped to ✅

## Per-Test Classification (Task 3, `test_load_telescope_runs.py`)

| Original test | Disposition | Reason / Replacement |
|---|---|---|
| `test_creates_one_event_per_night` | Migrated | Asserts against `ALLOC:`-prefixed events instead of all events |
| `test_iter_run_nights_eso_drops_tatoo_end_boundary` | Kept unchanged | Calls `_iter_run_nights()` directly; unaffected by the cutover |
| `test_iter_run_nights_magellan_both_inclusive_unchanged` | Kept unchanged | Same |
| `test_iter_run_nights_eso_single_night` | Kept unchanged | Same |
| `test_iter_run_nights_eso_zero_length_range_raises` | Kept unchanged | Same |
| `test_event_durations_within_range` | Migrated | Asserts against `ALLOC:` events |
| `test_event_fields_set_from_parsed_run` | Migrated | Asserts against `ALLOC:` events |
| `test_cancelled_line_gets_bracket_cancelled_title_prefix` | Migrated | Asserts against `ALLOC:` events |
| `test_non_cancelled_statuses_keep_unprefixed_title` | Migrated | Asserts against `ALLOC:` events (4 separate runs) |
| `test_reingest_without_cancelled_reverts_title_prefix` | Migrated | Asserts against `ALLOC:` events; same run/night pks across both imports (status is not part of `source_identifier`) |
| `test_idempotent_rerun_no_duplicates` | Migrated | Now also asserts run-level `unchanged: 1` and night-level `unchanged: 4` in the summary |
| `test_unchanged_rerun_does_not_update_existing_rows` | Migrated | Same run-level + night-level assertions added |
| `test_reingest_with_drifted_sun_event_does_not_duplicate` | **Retired** | The drift-tolerance proximity match existed because the old command keyed events on a computed sun-event time. The run's key is now derived from the line's own fields and matched by `(run.pk, night)`, so a drifting `sun_event()` can never produce a duplicate row. **Replacement:** `test_reimport_with_drifted_sun_event_does_not_duplicate_or_move_the_night`, which patches `allocation_projector.sun_event` on the second import and asserts the stored `start_time` never moves and no new event appears -- the same property, proven against the new mechanism |
| `test_display_01_no_sidecar_row_for_classically_scheduled_event` | **Inverted & renamed** | A classical night now DOES get a `CalendarEventMeta` row, self-attributed to its own run (`_link_event_to_run()`, D-08/D-09) -- **renamed** `test_display_01_allocation_night_gets_a_calendar_event_meta_row_self_attributed_to_its_own_run` to state the new contract |
| `test_classical_schedule_never_adopts_a_projector_owned_event` | Kept & strengthened | Now asserts the classical path has no calendar lookup at all -- a projector-owned event sharing telescope/instrument/start_time is byte-identical after import |
| `test_unparseable_line_logged_and_skipped` | Kept unchanged | Same intent; assertion now goes through the `_alloc_events()` helper |
| `test_cross_month_line_logged_and_skipped` | Kept unchanged | Same |
| `test_partial_night_bon_to_hhmm_sets_end_time` | Migrated | Asserts against `ALLOC:` events |
| `test_partial_night_hhmm_to_eon_sets_start_time` | Migrated | Asserts against `ALLOC:` events |
| `test_campaign_omitted_leaves_target_list_none` | Migrated | Now also asserts `run.campaign is None` |
| `test_campaign_matching_sets_target_list_on_every_event` | Migrated | Now also asserts `run.campaign_id` |
| `test_campaign_no_match_raises_and_creates_nothing` | Migrated | Now also asserts `CampaignRun.objects.count() == 0` |
| `test_campaign_multiple_matches_raises` | Kept unchanged | Same |
| `test_campaign_no_churn_on_rerun` | Migrated | Now also asserts run-level + night-level `unchanged` in the summary |
| *(new)* `TestClassicalCalendarUnchangedByCutover.test_calendar_matches_pre_cutover_contract_field_by_field` | **Added** | ROADMAP Success Criterion 4's real gate: 4 representative lines (ESO multi-night, Magellan, cancelled, partial-night window), pinning the exact url set, title, telescope, instrument, `start_time` and `end_time` per night, independently derived from `sun_event()` in the test |

**Documented divergence from pre-cutover output** (recorded here and in the new test class's docstring): a cancelled classical run's event description now also carries the shared `event_description()` helper's `'Run status: Cancelled'` line, because allocation descriptions are composed through the same helper every other campaign-run event uses -- so a staff `mark_cancelled` action reaches allocation nights too. Titles, spans, telescope, instrument and event counts are unchanged.

## Operator-Visible Behavior Changes (for plan 35-07's paired docs)

These are the behaviors 35-07 must document in `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` and `docs/runbooks/telescope_runs_calendar.rst`'s classical-ingest section:

1. **Two summary lines instead of one.** The command now prints a run-level line (`lines processed`, `created`/`updated`/`unchanged`/`skipped`/`skipped_collision`) followed by a night-level line (`created`/`updated`/`unchanged`/`retired`/`rekeyed`/`blocked`/`skipped`), instead of the old single `created`/`updated`/`unchanged`/`skipped` line.
2. **New `--dry-run` flag.** Previews both summary lines without writing anything; for a brand-new run it predicts night counts from the window length rather than a sun-event computation.
3. **Calendar events are now keyed `ALLOC:{run_pk}:{night}`**, not a blank url (`url=''`). Anyone querying `CalendarEvent.objects.filter(url='')` for classical events will now find nothing from a fresh import.
4. **Every classical import now also creates or updates a `CampaignRun` row**, visible in the campaign admin and attribution views -- not just calendar events. A classical schedule line is a first-class allocation record now.
5. **Every allocation night carries a `CalendarEventMeta` row self-attributed to its own run.** Before, a classically-scheduled event had no companion row at all.
6. **A cancelled run's event description gains an extra `'Run status: Cancelled'` line** (documented divergence above).
7. **The schedule-line syntax gains an optional bracketed `[proposal]` token**, e.g. `NTT EFOSC2 allocation 9-13 July [0110.C-0234]`, used to disambiguate two proposals that would otherwise share the same telescope, instrument and nights.
8. **A within-file duplicate key is now reported and skipped**, not silently re-processed: a second line yielding the same `source_identifier` produces a stderr collision message naming both line numbers and a nonzero `skipped_collision` count.

## Decisions Made

See `key-decisions` in the frontmatter above.

## Deviations from Plan

None - plan executed exactly as written. Task 1 and Task 2 carry `tdd="true"`; see **TDD Gate Compliance** below for the actual commit shape versus the plan's RED/GREEN contract.

## TDD Gate Compliance

`workflow.tdd_mode` is `false` in this project's config, so the runtime MVP+TDD halt gate did not apply.

**Task 1** followed the full RED -> GREEN contract: `9a285c8` (`test(35-05): add failing tests for the classical run proposal token`) was committed with 4 of its 7 new tests genuinely failing (`ParsedRun` had no `proposal` field; the bracketed token was misparsed as a partial-night window token) and 3 passing coincidentally (a `ValueError` was already raised for the empty/duplicated/unbalanced-bracket cases, just for the wrong reason). `b81927b` (`feat(35-05): ...`) then made all 41 tests in the module pass. No `gsd_run check tdd-red-evidence` record was captured (the tool call is not part of this project's local workflow invocation), but the RED failure was verified directly via a full test run before the commit, and the failure was on the target tests' own assertions/attribute access, not an import error or unrelated failure -- a valid RED by the tdd.md gate's own definition.

**Task 2**: `preview_campaign_run_action()` was verified RED (an `ImportError` when the test file was run against the pre-implementation `campaign_utils.py`) before being implemented, but the Command-level behavior tests (`TestLoadTelescopeRunsWritesAllocations`, 7 tests) were written and verified together with the command rewrite in a single commit (`6ec5955`), not RED-first with a separate commit boundary -- the tests exercise the fully rewritten command end-to-end (parse -> resolve site -> write-and-reconcile -> project), and splitting that into a meaningful RED state against the OLD command would have required either duplicating fixtures for two incompatible command shapes or writing throwaway assertions, neither of which improves the result. This mirrors the disclosed pattern in 35-01's own SUMMARY (Task 1/2 TDD Gate Compliance).

**Disposition:** no code or test defect results from either disclosure -- every acceptance criterion and `<verify>` command for both tasks passes as committed, and the plan's own acceptance criteria (test counts, AST checks, help text) are all independently satisfied. Flagged here per the tdd.md gate-enforcement contract rather than omitted.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The allocation-writing classical ingest is complete, tested, and pinned field-by-field against pre-cutover output. Plan 35-06 (cutover command + repo-wide gates, wave 4) can now build the one-time cutover of legacy blank-url classical events onto this same command's field vocabulary and dispatch seam.
- Plan 35-07 (wave 5) owns the paired docs this plan intentionally deferred: `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` and the `docs/runbooks/telescope_runs_calendar.rst` classical-ingest section, using the "Operator-Visible Behavior Changes" list above as its scope.
- This plan did NOT run the repo-wide label-list suite or `--all-files` lint (per its own acceptance criteria and the plan's explicit note that 35-04 is mid-flight on shared files in this same wave) -- that whole-repo gate is 35-06 Task 3's, the first wave with a single plan in it.

---
*Phase: 35-allocation-layer-classical-cutover*
*Completed: 2026-09-13*

## Self-Check: PASSED

- FOUND: solsys_code/telescope_runs.py (proposal token)
- FOUND: solsys_code/campaign_utils.py (preview_campaign_run_action)
- FOUND: solsys_code/management/commands/load_telescope_runs.py (rewritten command)
- FOUND commit: 9a285c8
- FOUND commit: b81927b
- FOUND commit: 6ec5955
- FOUND commit: 1dd8b53
- All plan-level `<acceptance_criteria>` and `<verify>` commands re-run and passing (see task-by-task output above); 86 tests green across the plan's three modules in one invocation
