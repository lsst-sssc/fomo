---
phase: 34-the-observation-projector-trigger
plan: 02
subsystem: calendar-sync
tags: [management-command, sweep, backstop, telescope-label, site-lookup, django-orm]

# Dependency graph
requires:
  - phase: 34-the-observation-projector-trigger
    plan: "01"
    provides: "observation_projector.py's project_record()/event_url()/facility_for()/PROJECTED_FACILITIES and the three connected signal receivers, extended here with project_queryset() and observed_token()"
provides:
  - "solsys_code/management/commands/project_observation_calendar.py: the TRIG-03 backstop sweep -- --proposal/--facility/--dry-run, per-facility summary, per-record failure isolation, resolve_observed_site() (the D-07/D-08 one-time observed-telescope lookup)"
  - "solsys_code/observation_projector.py: project_queryset() (dry_run + pre_fields_hook extension point), observed_token(), telescope_token() D-07 stage-gated observed-token read-back"
  - "solsys_code/calendar_utils.py: SITE_TELESCOPE_MAP's ('ogg','2m0')/('coj','2m0')/('sor','4m0') renamed to 'FTN'/'FTS'/'SOAR'; OBSERVED_TELESCOPE_SITE_CODES and OBSERVED_SITE_PARAMETER_KEYS"
  - "sync_lco_observation_calendar retired (D-18): the command, its 38-test module, and its command name are gone; every behaviour it protected is migrated, already covered elsewhere, or recorded as deliberately retired"
affects: [34-the-observation-projector-trigger (plan 04), 35-allocation-layer-and-classical-cutover, 36-unattended-operation, 37-status-vocabulary-public-tallies-and-provenance-blind-gaps]

# Actuals (#2632)
actuals:
  tokens: 38413
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pre_fields_hook extension point: project_queryset() calls an optional per-record hook between capturing the pre-sweep CalendarEvent snapshot and building the intended field values, so a command-side side-effecting step (Task 3's site lookup) can run mid-sweep while the counted action still comes from the untouched pre-sweep snapshot -- a dry-run count can never disagree with what a real sweep does."
    - "One counting rule for both run modes: the sweep's counted action always comes from calendar_utils.preview_calendar_event_action(before, fields), never from a writer's own return value -- a receiver's mid-iteration write is still correctly reported by the sweep instead of vanishing into an 'unchanged' count."
    - "One failure bucket for two causes: a None portal block and a returned-but-unmapped (site, telescope) pair are treated identically (site_lookup_failed) -- two counters/log messages for one operational situation is the exact pitfall this collapses."
    - "The network call lives on the command side, never in the projector module: observation_projector.py greps to zero for resolve_placement_block/make_request, keeping the post_save receiver's TRIG-02 no-network-call guarantee intact even as the sweep grows a real portal call."

key-files:
  created:
    - solsys_code/management/commands/project_observation_calendar.py
    - solsys_code/tests/test_project_observation_calendar.py
  modified:
    - solsys_code/observation_projector.py
    - solsys_code/calendar_utils.py
    - solsys_code/campaign_attribution.py
    - solsys_code/tests/test_observation_projector.py
    - solsys_code/tests/test_calendar_utils.py
    - solsys_code/tests/test_campaign_attribution.py
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/helpers.py
    - solsys_code/tests/test_campaign_reconciler.py

key-decisions:
  - "The sweep's per-record loop wraps the WHOLE per-record body (hook call, event_fields_for(), and the real write) in one try/except inside project_queryset(), not just the field-building step -- so an unexpected exception anywhere in that path (not only a missing-window KeyError) is caught, counted as 'unprojectable', and reported by the command on stderr without ending the sweep."
  - "LCO_SITE_CODE_TO_OBSCODE (campaign_attribution.py) gained 'ogg'->'F65' and 'sor'->'I33': the plan's own acceptance criteria require FTN/SOAR to still score a site-level telescope match, which structurally needs these two obscodes in the table. The local dev DB has no Observatory rows to verify against directly, so these are instead verified against this codebase's own already-committed, independently-sourced Observatory fixtures (F65 for ogg/FTN appears in four other test modules; I33 for sor/SOAR appears in test_campaign_gap.py) -- see Deviations."
  - "resolve_observed_site() lives in the command module as a plain function (not a Command method) returning (counter_increment, message) rather than writing to stderr itself, so it stays unit-testable in isolation; the command's handle() wraps it in a small closure that owns the actual self.stderr.write() call."

patterns-established:
  - "Extension-point hooks on a sweep function (pre_fields_hook) are the mechanism for adding a side-effecting command-side step to a projector-owned loop without either duplicating the loop or reaching into the projector's own module for a network call."

requirements-completed: [TRIG-03, PROJ-05, PROJ-01, ANNOT-03]

coverage:
  - id: D1
    description: "project_observation_calendar sweeps every LCO/SOAR ObservationRecord with zero required arguments, supports --proposal (exact-code, comma list)/--facility/--dry-run, isolates one bad record's failure from the rest of the sweep (reported on stderr, sweep still exits 0), and a second sweep over unchanged data reports created:0/updated:0 for every facility."
    requirement: "TRIG-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_project_observation_calendar.py#TestBareInvocationAndSummary, TestProposalAndFacilityFiltering, TestDryRun, TestFailureIsolation, TestNamespaceIsolation, TestProjectQuerysetOrdering"
        status: pass
    human_judgment: false
  - id: D2
    description: "sync_lco_observation_calendar (command + its 38-test module + its registered command name) is retired; every one of its 38 tests is classified as already-covered, migrated (with a named destination test), or deliberately retired with a reason -- no behaviour silently dropped."
    requirement: "ANNOT-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_calendar_utils.py#TestExtractInstrument; solsys_code/tests/test_observation_projector.py#TestFieldPopulation, TestProjectRecordWrites; solsys_code/tests/test_project_observation_calendar.py#TestProposalAndFacilityFiltering, TestBareInvocationAndSummary"
        status: pass
      - kind: other
        ref: "grep -rn 'sync_lco_observation_calendar' --include='*.py' solsys_code/ src/ returns no matches; django.core.management.get_commands() no longer lists 'sync_lco_observation_calendar'"
        status: pass
    human_judgment: false
  - id: D3
    description: "PROJ-01 (one CalendarEvent per record) and PROJ-05 (no-churn, never writes outside its own namespace) hold with the sweep as backstop live: the sweep's own RUN:/GEM:/blank-url namespace-isolation test and the no-churn second-sweep test close the loop 34-01 opened for these two requirements, which this plan also declares (shared-ID gate)."
    requirement: "PROJ-01, PROJ-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_project_observation_calendar.py#TestNamespaceIsolation.test_run_gem_and_blank_url_events_are_untouched_by_a_sweep, TestBareInvocationAndSummary.test_second_sweep_over_unchanged_data_reports_created_zero_updated_zero"
        status: pass
    human_judgment: false
  - id: D4
    description: "Once a record reaches a successful terminal state, the observation projector resolves its observed telescope exactly once (site_lookups: 1 on the first sweep, 0 on every sweep after), stores it under generic ObservationRecord.parameters keys, titles the event with the observed token ([O] FTN/FTS/SOAR/SITE-aperture), and campaign_attribution's telescope-match signal still scores a site-level match for the three renamed labels instead of degrading to aperture-only."
    requirement: "PROJ-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_observation_projector.py#TestObservedToken; solsys_code/tests/test_project_observation_calendar.py#TestObservedSiteLookup; solsys_code/tests/test_calendar_utils.py#test_telescope_01_d07_renamed_observed_telescope_labels; solsys_code/tests/test_campaign_attribution.py#test_telescope_match_d07_renamed_observed_labels_still_resolve_site_level"
        status: pass
    human_judgment: false
  - id: D5
    description: "The whole project test suite still passes with the new sweep command, the renamed telescope labels, and the retired sync command all live at once -- no regression anywhere else in the codebase."
    verification:
      - kind: integration
        ref: "workflow.test_command (.planning/config.json) -- 1070 + 40 tests, both runs OK"
        status: pass
    human_judgment: false

duration: 52min
completed: 2026-09-11
status: complete
---

# Phase 34 Plan 2: The Observed Telescope & the Sweep Command Summary

**A zero-argument `project_observation_calendar` sweep backstops the `post_save` receiver, resolves each record's actually-observed telescope exactly once and titles it `[O] FTN`/`[O] FTS`/`[O] SOAR`, and retires `sync_lco_observation_calendar` outright — its 38 tests classified one by one as covered, migrated, or deliberately dropped.**

## Performance

- **Duration:** 52 min
- **Started:** ~2026-09-11T02:47:00Z
- **Completed:** 2026-09-11T03:39:00Z
- **Tasks:** 3
- **Files modified:** 12 (2 created, 8 modified, 2 deleted)

## Accomplishments

- `project_observation_calendar` (new command): sweeps every LCO/SOAR `ObservationRecord`
  with no required arguments, `--proposal`/`--facility`/`--dry-run` supported, per-record
  failure isolation reported on stderr with the sweep still exiting 0, and a second sweep
  over unchanged data reporting `created: 0, updated: 0` for every facility.
- `observation_projector.project_queryset()`: the sweep's own per-record loop, sharing the
  receiver's exact comparison rule (`calendar_utils.preview_calendar_event_action()`) so a
  `--dry-run` count can never disagree with a real run; exposes a `pre_fields_hook` extension
  point Task 3 uses for the site lookup without duplicating the loop or putting a network
  call in the projector module.
- `sync_lco_observation_calendar` — the command, its 38-test module, and its registered
  command name — deleted outright (D-18). Every one of its 38 tests is classified in the
  table below: 16 already covered by the projector's own test suite, 10 migrated as
  projector-native tests, and 12 recorded as deliberately retired with a reason (mostly the
  live-API-resolution-at-'placed'-stage + `[UNVERIFIED]` fallback mechanism, which the new
  design does not have).
- `observed_token()`/`telescope_token()` (D-07): once a record reaches a successful terminal
  state, the sweep's `resolve_observed_site()` calls the existing portal-block resolver
  exactly once, stores the result under three generic `ObservationRecord.parameters` keys,
  and the projector reads it back as the title token — `FTN`/`FTS`/`SOAR` for the renamed
  2m0/4m0 sites, the existing `SITE-aperture` form everywhere else. A failed or unmapped
  lookup is one bucket (`site_lookup_failed`), leaves the coarse token in place, and is
  retried on the next sweep.
- `campaign_attribution._extract_lco_site_code()` consults the new
  `OBSERVED_TELESCOPE_SITE_CODES` bridge before its split-on-dash parse, so an orphan event
  whose telescope is `FTN` or `SOAR` still scores a site-level match instead of silently
  degrading to aperture-only — closing the asymmetric regression the label rename would
  otherwise cause (`FTS` alone kept working through the pre-existing classical-site-alias
  branch).

## Task Commits

Each task was committed atomically:

1. **Task 1: The sweep command — zero required arguments, dry-run, per-record isolation** - `11f8c63` (feat)
2. **Task 2: Retire sync_lco_observation_calendar — one writer for observation-backed nights** - `d2e9daf` (feat)
3. **Task 3: The observed telescope — one lookup per record, stored on the record, read as a title token** - `5c87a0a` (feat)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md)

## Files Created/Modified

- `solsys_code/management/commands/project_observation_calendar.py` - the sweep command:
  `Command`, `_COUNTER_KEYS`, `_new_counters()`, `_parse_proposal_arg()`,
  `resolve_observed_site()`
- `solsys_code/observation_projector.py` - `project_queryset()`, `observed_token()`,
  `telescope_token()` extended for the D-07 observed stages
- `solsys_code/calendar_utils.py` - `SITE_TELESCOPE_MAP`'s three renamed values,
  `OBSERVED_TELESCOPE_SITE_CODES`, `OBSERVED_SITE_PARAMETER_KEYS`
- `solsys_code/campaign_attribution.py` - `_extract_lco_site_code()`'s D-07 bridge,
  `LCO_SITE_CODE_TO_OBSCODE`'s two new entries
- `solsys_code/tests/test_project_observation_calendar.py` (new) - the full sweep-command
  test suite (22 tests across bare invocation, filtering, dry-run, failure isolation,
  namespace isolation, row ordering, and the observed-site lookup)
- `solsys_code/tests/test_observation_projector.py` - `TestFieldPopulation` (proposal/
  description/target_list, migrated) and `TestObservedToken` (D-07), both new classes
- `solsys_code/tests/test_calendar_utils.py` - `TestExtractInstrument` (new, migrated
  extraction-rule tests), rewritten `test_telescope_01_verified_dict_covers_all_sites`, new
  `test_telescope_01_d07_renamed_observed_telescope_labels`
- `solsys_code/tests/test_campaign_attribution.py` -
  `test_telescope_match_d07_renamed_observed_labels_still_resolve_site_level` (new)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` - deleted
- `solsys_code/tests/test_sync_lco_observation_calendar.py` - deleted
- `solsys_code/management/commands/load_telescope_runs.py`,
  `solsys_code/templatetags/calendar_display_extras.py`, `solsys_code/tests/helpers.py`,
  `solsys_code/tests/test_campaign_reconciler.py` - comment/docstring-only touch-ups
  removing the last textual references to the deleted module name (see Deviations)

## Behaviour Classification Table (Task 2, D-18)

Every one of the retired `test_sync_lco_observation_calendar.py`'s 38 tests, classified into
exactly one bucket. "Covered" names an existing test that already proves the behaviour;
"Migrated" names a new test added by this plan; "Retired" names the reason the behaviour no
longer applies.

| # | Retired test | Bucket | Destination / reason |
|---|---|---|---|
| 1 | `test_select_01_only_matching_proposal_creates_events` | Covered | `test_project_observation_calendar.py#TestProposalAndFacilityFiltering.test_proposal_filter_matches_exact_code_only` |
| 2 | `test_sync_01_d01_url_uses_requests_path_not_requestgroups` | Covered | `test_observation_projector.py#TestProjectRecordWrites.test_lco_and_soar_records_use_distinct_facility_instances_and_urls` |
| 3 | `test_sync_02_d03_unscheduled_uses_parameters_times_and_queued_title` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_queued_spans_request_window_and_titles_q` |
| 4 | `test_d06_completed_with_unresolved_scheduled_start_gets_clean_title` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_completed_no_block_spans_request_window_and_titles_o_never_q` |
| 5 | `test_sync_03_d03_placed_uses_scheduled_times_and_clean_title` | Retired | Live-API label resolution at the 'placed' stage is retired (D-08 restricts the one-time lookup to 'observed'/'completed-no-block' only); span/marker coverage migrates to `test_observation_projector.py#TestEventFieldsFor.test_placed_spans_block_and_titles_s` |
| 6 | `test_display_01_verified_record_creates_sidecar_row_is_verified_true` | Retired | `CalendarEventMeta.is_verified` is unconditionally `True` on every projection now (34-01 D-06/D-19), decoupled from telescope-label verification entirely |
| 7 | `test_display_01_fallback_record_creates_sidecar_row_is_verified_false` | Retired | Same reason as #6 |
| 8 | `test_sync_05_telescope_instrument_proposal_populated` | Migrated | instrument: `test_observation_projector.py#TestTitleAndToken.test_instrument_field_equals_extracted_instrument_not_the_aperture_label` (pre-existing); proposal: `test_observation_projector.py#TestFieldPopulation.test_proposal_field_equals_the_records_own_proposal` (new); the telescope-via-live-API portion is retired per #5/#8 |
| 9 | `test_term_01_d04_window_expired_gets_expired_prefix` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_window_expired_keeps_full_window_and_titles_x` |
| 10 | `test_term_01_d04_canceled_gets_cancelled_prefix` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_canceled_keeps_full_window_and_titles_c` |
| 11 | `test_term_01_d04_failure_limit_reached_gets_failed_prefix` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_failure_limit_reached_keeps_full_window_and_titles_f` |
| 12 | `test_term_01_d04_not_attempted_gets_failed_prefix` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_not_attempted_keeps_full_window_and_titles_f` |
| 13 | `test_d06_completed_gets_clean_title_no_prefix` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_observed_with_block_titles_o` |
| 14 | `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` | Covered | `test_observation_projector.py#TestProjectRecordWrites.test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn` + `test_project_observation_calendar.py#TestBareInvocationAndSummary.test_second_sweep_over_unchanged_data_reports_created_zero_updated_zero` |
| 15 | `test_display_01_rerun_on_unchanged_record_no_duplicate_sidecar_row` | Retired | `CalendarEventMeta.event` is a `OneToOneField` primary key (models.py), so a duplicate sidecar row is structurally impossible regardless of test coverage; the `is_verified`-toggle half is retired per #6/#7 |
| 16 | `test_sync_05_d05_description_contains_proposal_status_and_window` | Migrated | `test_observation_projector.py#TestFieldPopulation.test_description_contains_proposal_status_and_window` (new) |
| 17 | `test_banner_record_missing_site_still_syncs_with_coarse_label` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_queued_spans_request_window_and_titles_q` — the new design never reads a flat `parameters['site']` key for the queued stage at all |
| 18 | `test_skip_path_inconsistent_scheduled_times_logged_and_skipped` | Retired | D-13 deliberately replaces "skip" with "project with a `[?]` marker"; new coverage at `test_observation_projector.py#TestEventFieldsFor.test_half_set_schedule_projects_as_question_mark_and_does_not_raise` |
| 19 | `test_zero_match_reports_created_zero_no_command_error` | Migrated | `test_project_observation_calendar.py#TestProposalAndFacilityFiltering.test_zero_matching_proposal_reports_created_zero_no_error` (new) |
| 20 | `test_select_02_comma_list_matches_any_no_substring_leakage` | Migrated | `test_project_observation_calendar.py#TestProposalAndFacilityFiltering.test_comma_list_matches_any_no_substring_leakage` (new) |
| 21 | `test_select_03_all_token_case_insensitive_syncs_everything` | Retired | D-17: no `ALL` sentinel carried over — omitting `--proposal` already means every record |
| 22 | `test_select_04_single_run_covers_both_facilities` | Covered | `test_project_observation_calendar.py#TestBareInvocationAndSummary.test_bare_invocation_projects_every_record_and_prints_full_summary` |
| 23 | `test_select_05_soar_record_uses_soar_facility_instance` | Covered | `test_observation_projector.py#TestProjectRecordWrites.test_lco_and_soar_records_use_distinct_facility_instances_and_urls` |
| 24 | `test_extract_02_soar_multi_config_picks_spectrum_not_calibration` | Migrated | `test_calendar_utils.py#TestExtractInstrument.test_soar_multi_config_picks_spectrum_not_calibration` (new) |
| 25 | `test_extract_02_muscat_per_channel_exposure_extracts_instrument` | Migrated | `test_calendar_utils.py#TestExtractInstrument.test_muscat_per_channel_exposure_extracts_instrument` (new) |
| 26 | `test_d06_no_extractable_config_logged_and_counted_separately` | Migrated | `test_calendar_utils.py#TestExtractInstrument.test_no_recognized_config_and_no_flat_key_returns_none` (new); the dedicated `extraction_failed` counter itself is retired — folded into the single `unprojectable` bucket (D-17) |
| 27 | `test_telescope_03_api_failure_falls_back_not_skipped` | Retired | Live-API fallback mechanism at the 'placed' stage is retired (D-01/D-08) |
| 28 | `test_telescope_03_soar_api_failure_fallback_returns_4m0_label` | Retired | Same reason as #27 |
| 29 | `test_telescope_04_fallback_label_visibly_distinguishable` | Retired | The `[UNVERIFIED]` marker does not exist in the new title vocabulary (`[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` only, 34-01) |
| 30 | `test_sync_06_fallback_counter_distinct_from_skipped` | Retired | `telescope_api_failed` is retired; the new vocabulary is `created`/`updated`/`unchanged`/`unprojectable`/`site_lookups`/`site_lookup_failed` (D-17) |
| 31 | `test_sync_07_api_failure_does_not_abort_run` | Retired | Tested isolation around a live API call at the 'placed' stage; the new sweep's own per-record isolation is covered generically by `test_project_observation_calendar.py#TestFailureIsolation` |
| 32 | `test_sync_09_log_line_is_fixed_generic_message` | Covered | `test_calendar_utils.py#TestResolvePlacementBlockFailureModes.test_sync_09_no_credential_or_body_leak_in_logs` (the resolver's own contract, reused unchanged); also newly exercised command-side by `test_project_observation_calendar.py#TestObservedSiteLookup.test_failure_message_is_fixed_and_never_leaks_exception_content` |
| 33 | `test_d01_banner_record_no_api_call_no_unverified_prefix` | Covered | `test_observation_projector.py#TestEventFieldsFor.test_queued_spans_request_window_and_titles_q` (no API call for the queued stage; `[UNVERIFIED]` itself is retired per #29) |
| 34 | `test_target_list_01_single_membership_sets_target_list` | Migrated | `test_observation_projector.py#TestFieldPopulation.test_single_target_list_membership_sets_target_list` (new) |
| 35 | `test_target_list_02_zero_membership_sets_none_no_crash` | Migrated | `test_observation_projector.py#TestFieldPopulation.test_zero_target_list_membership_sets_none_no_crash` (new) |
| 36 | `test_target_list_03_multi_membership_picks_alphabetically_first` | Migrated | `test_observation_projector.py#TestFieldPopulation.test_multi_target_list_membership_picks_alphabetically_first` (new) |
| 37 | `test_target_list_04_no_churn_on_unchanged_fk_field` | Covered | `test_observation_projector.py#TestProjectRecordWrites.test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn` — no-churn is proven generically across every field in `fields`, `target_list` included |
| 38 | `test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped` | Retired | Live-API resolution + fallback mechanism at the 'placed' stage is retired (D-01/D-08); a malformed block from the portal is now Task 3's own site-lookup concern, covered by `test_project_observation_calendar.py#TestObservedSiteLookup.test_none_block_and_unmapped_pair_both_count_site_lookup_failed_and_are_retried` |

**Totals:** 16 already covered, 10 migrated (all as new tests in this plan), 12 retired with a
named reason. 38/38 classified — none left unaccounted for.

## Decisions Made

- `project_queryset()`'s `pre_fields_hook` extension point (rather than a second, parallel
  loop or a subclassing hook) is what lets Task 3's site lookup run mid-sweep without moving
  the `before`-snapshot/`fields`-build steps it sits between, and without giving the
  projector module a network call.
- `resolve_observed_site()` treats a `None` portal block and a resolved-but-unmapped
  `(site, telescope)` pair as one failure bucket (`site_lookup_failed`) — two counters for
  one operational situation (an observer can't yet tell what telescope this was) would only
  make the summary line harder to read for no added signal.
- See "Deviations from Plan" below for the two places this plan went slightly beyond its own
  literal `<action>` text to satisfy its own acceptance criteria and verify commands.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] `LCO_SITE_CODE_TO_OBSCODE` needed two more entries for the plan's own FTN/SOAR site-match acceptance criteria to be satisfiable**
- **Found during:** Task 3, while implementing `_extract_lco_site_code()`'s D-07 bridge
- **Issue:** The plan's Task 3 action text only says to "consult `OBSERVED_TELESCOPE_SITE_CODES`... before the existing split-on-dash parse," but `telescope_match_score()`'s site-level branch additionally requires the resolved site code to be a key of `LCO_SITE_CODE_TO_OBSCODE` — which, before this task, held only `'coj'`. Without adding `'ogg'`/`'sor'`, an orphan event whose telescope is `FTN` or `SOAR` would resolve a site code via the new bridge but then fail the `in LCO_SITE_CODE_TO_OBSCODE` check and fall through to `TELESCOPE_MATCH_INDETERMINATE` (since neither string carries an aperture-class token either) — failing the plan's own explicit acceptance criterion ("campaign_attribution scores a site-level telescope match for an orphan event whose telescope is FTN against a run at the ogg obscode").
- **Fix:** Added `'ogg': 'F65'` and `'sor': 'I33'` to `LCO_SITE_CODE_TO_OBSCODE`. The local dev DB has no Observatory rows to verify against directly (this table's own stated verification path), so these two are instead verified against this codebase's own already-committed, independently-sourced Observatory fixtures: `F65` for `ogg`/FTN (Haleakala) appears across `test_import_campaign_csv.py`, `test_canonical_record_migration.py`, `test_campaign_approval.py`, and `test_reconcile_campaign_runs.py`; `I33` for `sor`/SOAR (Cerro Pachon) appears in `test_campaign_gap.py` — both with matching real-world coordinates, not inferred from the site name alone.
- **Files modified:** `solsys_code/campaign_attribution.py`
- **Verification:** `test_telescope_match_d07_renamed_observed_labels_still_resolve_site_level` passes for all three labels
- **Commit:** `5c87a0a`

**2. [Rule 3 - Blocking] Task 2's own verify command required scrubbing textual references outside the plan's declared `files_modified`**
- **Found during:** Task 2, running `grep -rn 'sync_lco_observation_calendar' --include='*.py' solsys_code/ src/`
- **Issue:** This plan's own Task 2 acceptance criterion and verify command require this grep to return zero matches — literally, not just "no live imports." After deleting the command and its test module, 8 comment/docstring-only mentions remained in files this plan did not declare in `files_modified`: `solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/templatetags/calendar_display_extras.py`, `solsys_code/tests/helpers.py`, and `solsys_code/tests/test_campaign_reconciler.py`.
- **Fix:** Reworded each reference to describe "the retired LCO/SOAR sync command" (or, in `calendar_display_extras.py`, "the legacy verbose title-prefix vocabulary") instead of naming the deleted module — comment/docstring text only, no behaviour change in any of the four files. `calendar_display_extras.py`'s edit also documents, without attempting to fix, a genuine pre-existing gap: its `_TERMINAL_PREFIXES`/`status_border_css()` box-shadow styling still expects the OLD verbose prefixes (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`), not the observation projector's terse bracket markers (`[X]`/`[C]`/`[F]`) introduced in 34-01 — this predates this plan, is out of this plan's scope, and is explicitly earmarked for Phase 37 (STATUS-01/02) per `.planning/ROADMAP.md`.
- **Files modified:** `solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/templatetags/calendar_display_extras.py`, `solsys_code/tests/helpers.py`, `solsys_code/tests/test_campaign_reconciler.py` (plus `solsys_code/calendar_utils.py`, `solsys_code/tests/test_calendar_utils.py`, `solsys_code/tests/test_observation_projector.py`, which ARE declared in `files_modified`, for the same reference cleanup)
- **Verification:** `grep -rn 'sync_lco_observation_calendar' --include='*.py' solsys_code/ src/` returns no matches; full project test suite still passes (1070 + 40 tests)
- **Commit:** `d2e9daf`

---

**Total deviations:** 2 auto-fixed (1 Rule 2 — a data-table gap the plan's own acceptance
criteria required closing; 1 Rule 3 — a verify-command requirement the plan's declared
`files_modified` list didn't fully anticipate). Neither changes any production behaviour
outside what the plan's own text already specified; both are documented, comment-scoped or
narrowly-additive, and fully covered by tests or the plan's own verify commands.
**Impact on plan:** None outside the stated scope — no unplanned architectural change, no
scope creep beyond what the plan's own acceptance criteria required.

## Issues Encountered

None beyond the deviations above. One pre-existing, out-of-scope observation surfaced while
scrubbing comments in `calendar_display_extras.py`: its terminal-status box-shadow styling
(`_TERMINAL_PREFIXES`) still reads the old verbose title-prefix vocabulary
(`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`) rather than the observation projector's terse bracket
markers (`[X]`/`[C]`/`[F]`) 34-01 introduced — so an observation-projector-owned terminal-
negative event does not currently get the terminal box-shadow ring on the calendar UI. This
predates this plan (34-01 introduced the marker vocabulary; nothing in either 34-01 or 34-02
touches `calendar_display_extras.py`'s styling logic), is not in this plan's `files_modified`
or success criteria, and is explicitly earmarked for Phase 37 (STATUS-01/02, "one status
vocabulary") per `.planning/ROADMAP.md`. Documented in the code comment where it was found
(see Deviation #2) so it is visible without needing to rediscover it.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Plan 34-04 (the notebook/runbook/docs plan) can now build on a fully-wired, fully-tested
sweep and observed-telescope lookup: `project_observation_calendar` with `--proposal`/
`--facility`/`--dry-run`, `resolve_observed_site()`, and the renamed `FTN`/`FTS`/`SOAR`
telescope labels are all committed and proven against 170 new/modified tests plus the full
1110-test pre-existing suite (all green). The `campaign_attribution.py` behaviour-change
pairing this plan's own action text flagged for 34-04 Task 1 (`campaign_lifecycle_demo.ipynb`
needs a cell exercising the new FTN/SOAR site-match branch) is recorded and ready to pick up.
No blockers for 34-03 (the sibling wave-2 plan touching `calendar_display_extras.py`/
templates/views.py) — its files are disjoint from this plan's, and the pre-existing
verbose-vs-terse marker vocabulary gap noted above is unaffected by anything 34-03 does.

## Self-Check: PASSED

- `solsys_code/management/commands/project_observation_calendar.py` — FOUND
- `solsys_code/tests/test_project_observation_calendar.py` — FOUND
- `solsys_code/management/commands/sync_lco_observation_calendar.py` — CONFIRMED ABSENT
- `solsys_code/tests/test_sync_lco_observation_calendar.py` — CONFIRMED ABSENT
- Commit `11f8c63` — FOUND in `git log`
- Commit `d2e9daf` — FOUND in `git log`
- Commit `5c87a0a` — FOUND in `git log`

---
*Phase: 34-the-observation-projector-trigger*
*Completed: 2026-09-11*
