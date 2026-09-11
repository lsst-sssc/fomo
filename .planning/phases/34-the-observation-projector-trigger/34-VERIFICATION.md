---
phase: 34-the-observation-projector-trigger
verified: 2026-09-11T23:05:00Z
status: gaps_found
score: 11/14 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-01-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-01-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-02-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-02-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-03-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-03-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-04-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-04-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-05-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-05-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-06-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-06-SUMMARY.md"
  - "CLAUDE.md"
  - "docs/notebooks.rst"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json"
  - "docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/apps.py"
  - "solsys_code/calendar_utils.py"
  - "solsys_code/campaign_attribution.py"
  - "solsys_code/management/commands/project_observation_calendar.py"
  - "solsys_code/observation_projector.py"
  - "solsys_code/templatetags/calendar_display_extras.py"
  - "solsys_code/tests/test_calendar_display_extras.py"
  - "solsys_code/tests/test_calendar_template.py"
  - "solsys_code/tests/test_calendar_utils.py"
  - "solsys_code/tests/test_campaign_attribution.py"
  - "solsys_code/tests/test_observation_projector.py"
  - "solsys_code/tests/test_observation_projector_signals.py"
  - "solsys_code/tests/test_project_observation_calendar.py"
  - "solsys_code/views.py"
  - "src/templates/tom_calendar/partials/calendar.html"
  - "src/templates/tom_calendar/partials/event_form.html"
covered_digest: "v1:sha256:9778304b72881883f21fbffd5b230317c011ccf77ff3809a198f0ba2e5619182"
behavior_unverified: 1
overrides_applied: 0
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
re_verification:
  previous_status: human_needed
  previous_score: 11/14
  gaps_closed:
    - "G-34-2 (UAT Test 2 blocker): the post_save receiver no longer raises AttributeError on the real `updatestatus` path. `calendar_utils.coerce_schedule_datetime()` converts the portal's ISO strings to aware UTC datetimes; `record_time_window()` routes BOTH branches through it. Proven live: `tmp/34-06-updatestatus.txt` (a real portal-backed `updatestatus` run against a scratch copy) contains ZERO `unprojectable` lines, and the dry-run sweep that follows it (`tmp/34-06-dry-run.txt`) reports `LCO: created: 0, updated: 0, unchanged: 159` -- the receiver, not the sweep, did all the work."
    - "Truth 14 (interrupted sweep converges with no repair step) -- previously abstained `insufficient_spec`, now VERIFIED on directly observed operator behaviour recorded in 34-UAT.md Test 3 (result: pass) with concrete counts."
  gaps_remaining: []
  regressions:
    - "Truth 10 (paired docs): commits 73b465d and 46d8390 re-executed `project_observation_calendar_demo.ipynb` against an already-converged scratch copy. The committed executed output no longer demonstrates the one-time takeover (0 of 159 events re-titled, empty sample list) and its first and second sweep summary lines are now byte-identical (both all-zero), which is exactly what two of plan 34-04's must_haves say must NOT be the case. At the previous verification (a87f5f8) this demonstration was intact (156 of 156 re-titled; first sweep `created: 3, updated: 156, site_lookups: 59`)."
gaps:
  - truth: "The sweep's paired demo notebook shows the first sweep taking over the legacy URL-keyed events in place, and its first and second sweep summary lines are printed together and differ -- the first carrying non-zero created/updated and site_lookups, the second zeros (34-04 must_haves; TRIG-03, ANNOT-03, D-19, D-08, D-17; CLAUDE.md paired-docs rule)."
    status: failed
    reason: "The notebook was re-executed (73b465d, then again in 46d8390) against `tmp/fomo_g34_2_copy.sqlite3` -- a scratch copy that plan 34-06's own live-proof step had ALREADY swept to convergence at 14:29-14:30 before the notebook ran. The committed executed output therefore shows a takeover that takes over nothing and a convergence claim with no divergent first run to contrast against. The notebook's own prose now contradicts its own output, and the notebook has no assertion that would catch this."
    artifacts:
      - path: "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
        issue: "Cell 9 executed output: `0 of 159 pre-existing facility-url-keyed events were re-titled by the takeover.` followed by an EMPTY `Sample before -> after title pairs:` list. Cell 10 executed output: first sweep and second sweep summary lines are identical (`LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 1`), so the trailing prose 'the second sweep reporting zero created/updated/site_lookups for every facility is convergence, not silence' is now vacuous. Cell 5 markdown still states 'This particular run's first sweep is expected to report a non-zero `updated` count for LCO', which its own output falsifies. Cell 19's closing table still points PROJ-05 and TRIG-03 evidence at 'The one-time takeover' cells that now show nothing."
    missing:
      - "Re-execute the notebook against a scratch copy taken from an UN-swept state (clone `src/fomo_db.sqlite3` fresh, do NOT run `updatestatus` or the sweep against the clone first), so cell 9 shows real before -> after title pairs and cell 10's two summary lines differ."
      - "Add an assertion to cell 10 that the first sweep summary differs from the second (e.g. non-zero created+updated on the first run for at least one facility in scope), so this demonstration can never silently go empty again."
      - "Reconcile cell 5's 'expected to report a non-zero `updated` count' prose with whatever the re-execution actually produces."
      - "Do NOT satisfy this by running anything against `src/fomo_db.sqlite3` -- its 33 stale LCO events are UAT Test 4's SCHED-06 evidence and must stay untouched."
advisory: []
behavior_unverified_items:
  - truth: "Over real nights a pending KEY2026B-004 record's event narrows queued -> scheduled -> observed with nobody running anything (SCHED-06, ROADMAP criterion 4)."
    test: "From now, run ONLY `python manage.py updatestatus` over several real observing nights -- never `python manage.py project_observation_calendar`. Then re-execute `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` end to end, UN-routed (no FOMO_DATABASE_PATH), and `git diff` its SCHED-06 section against `project_observation_calendar_demo.sched06-baseline.json` (74 pending records at baseline: 56 queued, 18 placed, captured 2026-09-11T04:44:59.526430+00:00)."
    expected: "At least one KEY2026B-004 record has moved queued -> placed (or placed -> observed) and its calendar event span/title narrowed to match, with no sweep run in between. Record the outcome in the dated re-check table in `34-UAT.md` and flip Test 4's verdict from blocked/PARTIAL."
    why_human: "SCHED-06 is a verification-over-time requirement -- it depends on the real LCO scheduler placing and observing real requests on real nights. No test or grep can produce that evidence; only elapsed observing time can. Plan 34-04 deliberately established the baseline and left the re-check open."
    advisory: "The re-executed notebook runs two real sweeps (cells 8 and 10) BEFORE it re-captures the SCHED-06 baseline, so the re-captured snapshot alone cannot prove which writer narrowed the events. The discriminator is the re-execution's OWN first-sweep summary line: if it reports `created: 0, updated: 0` for the narrowed records, the post_save receiver had already narrowed them before the sweep ran. Read that line first, and note it in the 34-UAT.md row. G-34-2 is now fixed, so the 33 stale LCO events on `src/fomo_db.sqlite3` should be repaired by the receiver alone on the next real `updatestatus` -- that repair is itself part of the evidence."
insufficient_spec_items:
  - truth: "If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state (34-01 must_haves, verification: backstop)."
    test: "Drive two concurrent/interleaved saves of one LCO ObservationRecord (e.g. two `updatestatus` runs overlapping) and inspect the resulting CalendarEvent span and title against the record's final persisted fields."
    expected: "The surviving event matches the record's final persisted scheduled_start/scheduled_end/status -- no event left describing a superseded intermediate state, and no `unprojectable` line in the logs."
    why_human: "Declared non-inferable (`verification: backstop`). No held-out or property-based test exercises interleaved saves; the suite is single-threaded. 34-UAT.md Test 2 did run this, but its result was contaminated by G-34-2 (AttributeError on nearly every record) and was recorded as `issue`/blocker. G-34-2 is now fixed, so a clean re-run is cheap -- but no post-fix interleaved-save evidence exists yet."
human_verification:
  - test: "SCHED-06 / UAT Test 4 (already tracked in 34-UAT.md -- MERGE into the existing tracker, do not overwrite it). Run ONLY `python manage.py updatestatus` against `src/fomo_db.sqlite3` over several real observing nights, never the sweep. Then re-execute the demo notebook UN-routed and `git diff` the SCHED-06 baseline JSON. Read the re-execution's own FIRST sweep summary line before anything else."
    expected: "At least one KEY2026B-004 record narrowed queued -> placed (or placed -> observed) with its event following, and the first sweep reports `created: 0, updated: 0` for it -- proving the post_save receiver, not the sweep, did the narrowing. Fill in the dated row in 34-UAT.md's SCHED-06 re-check table and close SCHED-06."
    why_human: "Verification-over-time requirement; depends on real observing nights elapsing. This is the SINGLE remaining human item and it is already Test 4 in 34-UAT.md -- merge this outcome into that tracker rather than creating a new UAT file."
  - test: "Re-run 34-UAT.md Test 2 now that G-34-2 is fixed: drive two overlapping `updatestatus` runs against a COPY of the developer database (FOMO_DATABASE_PATH=<absolute scratch path>) and inspect the surviving CalendarEvents."
    expected: "Each event matches its record's final persisted field state, and the logs contain no `unprojectable ... AttributeError` lines (the failure that made the original Test 2 run a blocker)."
    why_human: "Declared `verification: backstop` (non-inferable); no test exercises concurrency. The original operator run is unusable as evidence because G-34-2 contaminated it."
deferred:
  - truth: "The two title-prefix vocabularies (the legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` prefixes still emitted by load_telescope_runs.py and campaign_views.py, and the projector's terse `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers) are unified into one vocabulary."
    addressed_in: "Phase 37"
    evidence: "ROADMAP Phase 37 'Status vocabulary, public tallies and provenance blind gaps' (STATUS-01/02) owns the final vocabulary; Phase 34's ROADMAP scope note states 'Title prefixes ship provisionally here; Phase 37 owns the final vocabulary.' Both vocabularies paint the correct ring today (calendar_display_extras._TERMINAL_PREFIXES carries all of them)."
---

# Phase 34: The Observation Projector & Trigger — Verification Report

**Phase Goal:** Every LCO/SOAR observation record draws and keeps current its own calendar event on every save, with a sweep as backstop; the old LCO sync command is retired in its favour.
**Verified:** 2026-09-11T23:05:00Z (HEAD `c1d6d39`)
**Status:** gaps_found
**Re-verification:** Yes — after gap-closure plans 34-05 / 34-06 and the 12-fix code-review round (`1d6ef6c..46d8390`). Supersedes the 2026-09-11T05:33:49Z report.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **(SC1)** Every LCO/SOAR `ObservationRecord` has exactly one calendar event keyed by its facility observation URL, spanning the request window while queued / the placed block once scheduled / the observed block once observed; a terminal-negative record keeps a visibly marked event on its window night. | ✓ VERIFIED | Code read of `observation_projector.stage_for()` / `event_fields_for()`: the span is derived from record fields alone (`record_time_window()` for the placed/observed block, `parameters['start'/'end']` for queued and for the `[?]` inconsistent case). Identity is `event_url()` = `facility.get_observation_url(record.observation_id)` and nothing else. Notebook cell 12 executed output over the real corpus: `159 LCO/SOAR records`, `159 facility-url-keyed events`, `records with no facility-url-keyed event: 0`, `159 - 0 = 159; matches event count: True`. Cell 13 per-marker tally: `[Q]:33 [S]:19 [O]:74 [X]:26 [C]:6 [F]:1 [?]:0`. Tests (all passing): `TestEventFieldsFor` — `test_window_expired_keeps_full_window_and_titles_x`, `test_canceled_...titles_c`, `test_failure_limit_reached_...titles_f`, `test_not_attempted_...titles_f`, `test_failure_marker_wins_over_stage_marker_when_block_is_placed`, `test_half_set_schedule_projects_as_question_mark_and_does_not_raise`, plus `test_two_records_whose_windows_exactly_abut_produce_two_separate_events`. |
| 2 | **(SC2)** Saving a record updates its event with no operator command — including the schedule-only placement save TOM's own hook misses and the `updatestatus` path — while a save that changes nothing writes nothing and a projector error is logged rather than aborting the record save. | ✓ VERIFIED | `apps.py:ready()` connects `post_save`(ObservationRecord), `m2m_changed`(ObservationGroup through-model) and `pre_delete`, each with `weak=False` + a `dispatch_uid`. **G-34-2 closed and behaviourally proven**: `test_updatestatus_narrows_the_event_with_no_command_run` drives the REAL `LCOFacility().update_observation_status()` with a portal payload of ISO-8601 strings inside `assertNoLogs(..., WARNING)` and asserts the event narrows to the block with an `[S] ` title; `test_updatestatus_event_span_matches_the_reloaded_record_no_churn` pins receiver-vs-sweep agreement; `test_updatestatus_with_unparseable_schedule_value_leaves_the_event_untouched` pins the TRIG-02 never-abort-the-save contract. Also passing: `test_schedule_only_save_narrows_the_same_event_row`, `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`, `test_raising_projector_does_not_block_a_save`, `test_record_saved_inside_a_rolled_back_transaction_leaves_no_event`, `test_make_request_is_never_called_during_a_record_save`, `test_raw_save_writes_no_calendar_event`. Live: `tmp/34-06-updatestatus.txt` — a real portal-backed `updatestatus` run against a scratch copy with **zero** `unprojectable` lines. |
| 3 | **(SC3)** One sweep command re-projects any set of records (`--dry-run` supported, per-record failures isolated) as the backstop for bulk-write paths; a second sweep reports everything unchanged and no event outside the projector's key namespace is created, modified or deleted. | ✓ VERIFIED | `python manage.py project_observation_calendar --help` runs with zero required args and offers `--proposal`, `--facility {LCO,SOAR}`, `--dry-run`. `project_queryset()` counts from `preview_calendar_event_action(before, fields)` against a pre-sweep snapshot, so a mid-iteration receiver write is still reported. Tests: `TestDryRun`, `TestFailureIsolation` (4), `TestNamespaceIsolation` (`test_run_gem_and_blank_url_events_are_untouched_by_a_sweep`), `TestProjectQuerysetOrdering`, `TestObservedSiteLookup` (8), `test_second_sweep_over_unchanged_data_reports_created_zero_updated_zero`, `test_first_sweep_resolves_site_once_and_counts_updated_and_site_lookups`, `test_second_sweep_issues_no_portal_call_and_reports_unchanged`. Live: `tmp/34-06-dry-run.txt` → `Done (dry run). failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0 \| SOAR: all zero`. The *notebook's* demonstration of the first-vs-second distinction has regressed — see truth 10; the behaviour itself is proven by the two named tests above. |
| 4 | **(SC4a)** Over real nights, a user watches a `KEY2026B-004` record's event narrow queued → scheduled → observed with nobody running anything (closes spike 004's PARTIAL verdict). | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Mechanism present, wired and unit-tested (truth 2), and G-34-2 — which previously made this impossible — is fixed. Baseline committed and intact: `project_observation_calendar_demo.sched06-baseline.json`, `captured_at 2026-09-11T04:44:59.526430+00:00`, 74 pending records (56 queued, 18 placed), byte-unchanged since `a87f5f8` (`git log` on that path shows only that commit). `src/fomo_db.sqlite3` untouched (mtime `1789157248`, clean `git status`), so its 33 stale LCO events remain as the re-check evidence. Routed to human verification — not counted as a gap (verification-over-time, per the ROADMAP scope note). |
| 5 | **(SC4b)** Every event title is short enough to read in a month cell (PROJ-06). | ✓ VERIFIED | `title_for()` produces `'[marker] <token> <target>'` capped at 200 chars; `test_marker_and_token_within_first_16_characters` asserts the marker plus telescope token fit inside the month cell's tightest `truncatechars:16` budget. Real committed titles from the notebook/baseline: `'[S] 1m0 220P'`, `'[X] 2m0 Didymos COJ 2026 Field #04'`, `'[O] FTS Didymos COJ 2026 Field #02'`. |
| 6 | **(SC5)** `sync_lco_observation_calendar` no longer exists; the same events come from the projector and sweep in the same key namespace, with runbook section, demo notebook and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-read-back caveat documented. | ✓ VERIFIED | `python manage.py help` lists `project_observation_calendar` and **no** `sync_lco_observation_calendar`. `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/tests/test_sync_lco_observation_calendar.py` and `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` are all absent from the tree. Remaining textual references are historical only (`docs/design/*`, `.planning/` archives). 34-02-SUMMARY carries the 38-test classification table (16 already-covered, 10 migrated with named destinations, 12 retired with reasons). Gemini command source unchanged; `sync_gemini_observation_calendar_demo.ipynb` cell 1 carries the "No observation-status or observation-URL read-back (Phase 34, D-21)" caveat, mirrored in the runbook (L389-399). |
| 7 | A calendar visitor can tell what each projector marker means from a legend on the calendar page itself, including `[?]`. | ✓ VERIFIED | `observation_status_legend()` returns the fixed 7-entry vocabulary; `calendar.html:329-333` renders it. `test_returns_seven_entries_covering_every_marker` and `test_calendar_page_renders_every_legend_marker_and_label` both pass. |
| 8 | Every projector marker paints the right status ring, and no existing ring is lost. | ✓ VERIFIED | `status_border_css()` matches `'[Q] '` beside the legacy `'[QUEUED] '`; `_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]', '[X] ', '[C] ', '[F] ', '[?] ')` — all four legacy verbose prefixes retained. `[S] `/`[O] ` deliberately return `''` (placed bucket, per D-09's reserved border treatment). `TestProjectorMarkerRings` + `TestTelescopeStripeContrast` pass. |
| 9 | Series identity ("night n of N") is rendered at request time from `CalendarEventMeta.observation_group`, never stored in the event, ordered by window start, with no per-event query fan-out. | ✓ VERIFIED | `observation_series_decoration()` is read-only (code read: no `save`/`create`/`update`/`get_or_create` anywhere in the body); returns `None` for a missing companion row, a missing group/record link, and groups with `< 2` members; sorts by `(_window_start_or_max(member), member.pk)` with a never-raise fallback. Rendered at `event_form.html:148` beside the campaign block. `views.fomo_render_calendar` prefetches `telescope_label_meta` with `select_related('run__campaign')`. Tests: `TestObservationSeriesDecoration`, `test_grouped_and_attributed_event_shows_both_decorations`, `test_modal_query_count_does_not_grow_with_group_size`. |
| 10 | The paired-docs rule is satisfied: the sweep has a pre-executed demo notebook whose executed output demonstrates the one-time takeover and a first-vs-second sweep that differ; the retired command's notebook is gone; the runbook describes the projector/sweep, legend, series block and Gemini caveat. | ✗ FAILED | **Regression introduced after the previous verification.** Registration and runbook halves pass (see below), but the notebook's executed output no longer demonstrates what two 34-04 must_haves require. Cell 9: `0 of 159 pre-existing facility-url-keyed events were re-titled by the takeover.` with an EMPTY sample list. Cell 10: first and second sweep summary lines are **identical** (`LCO: created: 0, updated: 0, unchanged: 159, ..., site_lookups: 0, site_lookup_failed: 1`). At `a87f5f8` (the state the previous verification saw) the same cells read `156 of 156 ... re-titled` with 8 real before→after pairs and `first sweep: created: 3, updated: 156, site_lookups: 59` vs `second: created: 0, updated: 0`. Cause: `73b465d` and `46d8390` re-executed the notebook against `tmp/fomo_g34_2_copy.sqlite3`, which plan 34-06's own live-proof step had already swept to convergence. Cell 5's prose ("this run's first sweep is expected to report a non-zero `updated` count for LCO") is falsified by its own output, and cell 19's closing table still cites those cells as the PROJ-05/TRIG-03 evidence. Passing halves: `docs/notebooks.rst:15` toctree entry present; `CLAUDE.md:129` notebook-map entry present; `sync_lco_observation_calendar_demo.ipynb` deleted; runbook covers projector/sweep (L54-197), legend (L99-101), Observation series (L108-118), one-time title change (L133-143), cheat-sheet, and `unprojectable`/`site_lookup_failed` troubleshooting. |
| 11 | Nothing group-derived is written into `CalendarEvent.title` or `.description`; series identity lives only in `CalendarEventMeta.observation_group` (PROJ-04 title-stem clause). | ✓ VERIFIED | `event_fields_for()` builds `description` from proposal/status/stage/window only; `title_for()` from marker/token/target only — neither calls `series_group_for()`. `write_event_meta()` writes exactly `is_verified`/`observation_record`/`observation_group`, never the campaign attribution link or its confirmation stamps, and first clears a stale one-to-one claim rather than raising IntegrityError. Tests: `test_grouped_record_sets_observation_group_and_omits_name_from_title_and_description`, `test_existing_campaign_attribution_survives_projection_and_is_verified_becomes_true`, `test_stale_companion_claim_on_a_different_event_is_cleared_not_integrity_error`, `test_record_in_two_groups_links_the_lowest_pk_group`. |
| 12 | The observed telescope is resolved once, stored on the record, read back as a network-free title token, and the D-07 label rename does not downgrade Phase 28's site-level attribution matching. | ✓ VERIFIED | `resolve_observed_site()` lives on the command side only (the projector makes no network call); it returns `(None, None)` for a non-terminal stage or an already-resolved record, and stores `observed_site`/`observed_telescope`/`observed_enclosure` with `update_fields=['parameters']`. `observed_token()` is a pure read through `derive_telescope()`. `SITE_TELESCOPE_MAP` carries `FTN`/`FTS`/`SOAR`; `campaign_attribution.OBSERVED_TELESCOPE_OBSCODES` bridges those three labels straight to obscodes (`F65`/`E10`/`I33`) ahead of `_extract_lco_site_code()`, with `calendar_utils.OBSERVED_TELESCOPE_SITE_CODES` as the classical site-code bridge. `campaign_lifecycle_demo.ipynb` cell 22 executed output: `FTN -> 'ogg' -> 'F65'`, `FTS -> 'coj' -> 'E10'`, `SOAR -> 'sor' -> 'I33'`, `FTN match level: 1.0`. |
| 13 | If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state. | ⚠️ insufficient_spec (abstained) | Declared `verification: backstop` in 34-01 must_haves. No held-out, property-based or concurrency test exists; the suite is single-threaded. 34-UAT.md Test 2 did exercise it, but the run was contaminated by G-34-2 (`AttributeError` on nearly every LCO record) and recorded as `issue`/blocker. G-34-2 is now fixed but no post-fix interleaved-save evidence exists. Routed to human verification. |
| 14 | A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step. | ✓ VERIFIED | Previously abstained; now closed on **directly observed operator behaviour** recorded in `34-UAT.md` Test 3 (`result: pass`), run against a copy (`FOMO_DATABASE_PATH=/tmp/fomo_uat_copy.sqlite3`) so the real DB stayed untouched: the interrupted run repaired 1 record, the re-run reported `created: 0, updated: 32, unchanged: 127, unprojectable: 0` (`tmp/project_observation_calendar_rerun.txt`), and a third dry run reported `updated: 0, unchanged: 159` — convergence with no repair or cleanup step. One documented `site_lookup_failed` (4276100) used the fallback label as designed. |

**Score:** 11/14 truths verified (1 present-but-behavior-unverified, 1 abstained as non-inferable, 1 failed)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Unification of the two title-prefix vocabularies (legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` vs. the projector's terse bracket-letter markers) | Phase 37 | ROADMAP Phase 34 scope note: "Title prefixes ship provisionally here; Phase 37 owns the final vocabulary." Phase 37 = "Status vocabulary, public tallies and provenance blind gaps" (STATUS-01/02). Both vocabularies already paint the correct ring, so nothing is broken today — only unreconciled. |

### Advisory (New Scope, Unevidenced)

New-scope findings from Step 7 with no deterministic evidence — reported, not blocking.

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None | — | Re-verification ran; every 🛑 finding this pass is either a carried-forward/regression item on a file modified since the prior `verified:` timestamp, or was already classified in the prior pass. Nothing was downgraded. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/observation_projector.py` | Projector + three signal receivers | ✓ VERIFIED | 737 lines. `project_record()` wraps every write in its own `transaction.atomic()` savepoint with the `except` OUTSIDE the `with` (so the savepoint actually rolls back); `unprojectable` log lines now carry the message, not just the class name (`logger.warning('unprojectable observation_id=%r: %s: %s', ...)`, WR-01 fix `1f6786b`). Imported by `apps.py`, the sweep command, and 3 test modules. |
| `solsys_code/apps.py` | `ready()` wires the three receivers | ✓ VERIFIED | `post_save` / `m2m_changed` / `pre_delete`, each `weak=False` with a distinct `dispatch_uid`, function-local imports per the app-loading convention. |
| `solsys_code/calendar_utils.py` | `coerce_schedule_datetime()` used by both `record_time_window()` branches | ✓ VERIFIED | `coerce_schedule_datetime()` handles `None`/`str`/`datetime`, attaches UTC to naive values, `astimezone(utc)` for non-UTC offsets (CR-01 `1d6ef6c`), raises on an unparseable string, a bare ISO date (WR-03 `38aeb0d`) and a non-str/non-datetime (WR-04 `bac6161`). `record_time_window()` routes the `parameters` branch through it too (CR-02 `d996b21`) and `cast()`s the both-populated branch (WR-05 `e5b6a13`). |
| `solsys_code/management/commands/project_observation_calendar.py` | Backstop sweep, zero required args | ✓ VERIFIED | 11.7 KB; registered (`manage.py help`); `--proposal` fails closed on an all-empty value; `--dry-run` passes `pre_fields_hook=None` so no closure over stderr is even built; per-record failures counted and written to stderr; exit 0. |
| `solsys_code/templatetags/calendar_display_extras.py` | Rings, legend, series decoration | ✓ VERIFIED | `status_border_css()`, `observation_status_legend()`, `observation_series_decoration()` all registered as template tags and all called from the two templates. |
| `src/templates/tom_calendar/partials/calendar.html` | Legend row + ring on month cells | ✓ VERIFIED | `{% observation_status_legend as status_legend %}` at L331; `{% status_border_css event.title as status_border %}` at L238 and L274. |
| `src/templates/tom_calendar/partials/event_form.html` | Series block beside the campaign block | ✓ VERIFIED | `{% observation_series_decoration event as series %}` at L148, rendered alongside (not replacing) the campaign decoration. |
| `solsys_code/campaign_attribution.py` | D-07 rename does not break Phase 28 matching | ✓ VERIFIED | `OBSERVED_TELESCOPE_OBSCODES = {'FTN': 'F65', 'FTS': 'E10', 'SOAR': 'I33'}` consulted ahead of `_extract_lco_site_code()`; that function documents why it returns `None` for the three label forms. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | Paired demo with executed output showing the takeover | ⚠️ HOLLOW | Exists, registered, 12/12 code cells carry committed output, scratch-copy and baseline guards are real (raise, not assert) — but the takeover/first-vs-second-sweep **demonstration is empty** (truth 10). |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json` | Byte-identical to its `a87f5f8` state | ✓ VERIFIED | `git log` on the path shows only `a87f5f8`; clean `git status`; content `captured_at 2026-09-11T04:44:59.526430+00:00`, `record_count 74`. |
| `docs/runbooks/telescope_runs_calendar.rst` | Projector/sweep section replacing the LCO sync section | ✓ VERIFIED | 1223 lines; projector narrative, marker legend note, Observation series block, one-time title change note, sweep flags + real summary line, dry-run caveats, `unprojectable`/`site_lookup_failed` troubleshooting, Gemini no-read-back caveat. No `[UNVERIFIED]`/`telescope_api_failed` residue. |
| `solsys_code/tests/*` (6 modules) | Behaviour pinned | ✓ VERIFIED | 49 + 23 + 25 projector/sweep tests plus the calendar_utils / display / template suites. Full run: **1127 tests, OK, exit 0**. No skipped/xfail markers in any phase-34 test module. |
| `src/fomo_db.sqlite3` | Unmodified (SCHED-06 evidence) | ✓ VERIFIED | mtime `1789157248` (the value the phase recorded); clean `git status`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ObservationRecord.save()` | `CalendarEvent` row | `post_save` → `receiver_on_record_save()` → `project_record()` → `insert_or_create_calendar_event({'url': …})` | ✓ WIRED | Whole chain read end-to-end; proven by `test_updatestatus_narrows_the_event_with_no_command_run` driving the real facility method. |
| `project_record()` | `CalendarEventMeta.observation_record/.observation_group` | `write_event_meta()` | ✓ WIRED | Writes exactly three fields; stale one-to-one claims cleared first. |
| `SolsysCodeConfig.ready()` | the three receivers | `post_save`/`m2m_changed`/`pre_delete` `.connect()` | ✓ WIRED | Without this nothing fires; all three present with `dispatch_uid`. |
| `ObservationGroup.observation_records.through` | `project_record()` | `m2m_changed` → `receiver_on_group_membership_changed()` | ✓ WIRED | `post_add`/`post_remove`/`post_clear` handled; `post_clear` re-derives members from the companion rows (WR-08) rather than a module global. |
| sweep per-record loop | `project_record()` / `preview_calendar_event_action()` | `project_queryset()` | ✓ WIRED | One projection path shared by receiver and sweep; dry-run counts derived from the same comparison helper. |
| `resolve_placement_block()` + `derive_telescope()` | title token | `record.parameters['observed_site'/'observed_telescope']` → `telescope_token()` | ✓ WIRED | Network-free read-back; `TestObservedToken` (6 tests) + `TestObservedSiteLookup` (8 tests). |
| `calendar_utils` telescope labels | Phase 28 attribution matching | `OBSERVED_TELESCOPE_OBSCODES` / `OBSERVED_TELESCOPE_SITE_CODES` | ✓ WIRED (route differs from the plan's literal wording) | The plan's key_link named `_extract_lco_site_code()`; the implementation instead resolves `FTN`/`FTS`/`SOAR` through a LABEL-keyed obscode table *before* that function, which is strictly more correct for a multi-telescope site like `ogg`. Intent satisfied; proven by `campaign_lifecycle_demo.ipynb` cell 22's executed output and `test_campaign_attribution.py`. |
| `CalendarEventMeta.observation_group` | `event_form.html` | `observation_series_decoration()` | ✓ WIRED | Display-time only; no write anywhere in the tag. |
| `FOMO_DATABASE_PATH` | notebook kernel | `src/fomo/settings.py:134` | ✓ WIRED | Notebook cell 2's guard resolves both paths and **raises** (not asserts) when an override resolves to the developer database — WR-08 fix `46d8390`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `observation_projector.event_fields_for()` | `start_time`/`end_time` | `record_time_window(record)` / `record.parameters` — never a stored event field | Yes (re-derived from record state on every projection) | ✓ FLOWING |
| `observation_projector.telescope_token()` | `token` | `record.parameters['observed_site'/'observed_telescope']` written once by the sweep's portal lookup, else `coarse_telescope_label(instrument, facility)` | Yes | ✓ FLOWING |
| `project_queryset()` counters | `action` | `preview_calendar_event_action(before, fields)` against a pre-sweep DB snapshot | Yes | ✓ FLOWING |
| `observation_series_decoration()` | `index`/`size` | `meta.observation_group.observation_records.all()` at request time | Yes | ✓ FLOWING |
| `observation_status_legend()` | legend entries | Module constant `_OBSERVATION_STATUS_LEGEND` | Intentionally fixed (documented: data-driven would drift from `status_border_css()`) | ✓ FLOWING (by design) |
| demo notebook cell 9/10 | `changed_titles`, `first_sweep_summary` | Live sweep against an **already-converged** scratch copy | **No** — the executed output is empty/degenerate | ⚠️ HOLLOW (truth 10) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite passes | `.planning/config.json` `workflow.test_command` (first invocation, run once, output saved and grepped) | `Ran 1127 tests in 132.782s` / `OK` / exit 0 | ✓ PASS |
| Sweep command registered with its flags | `python manage.py project_observation_calendar --help` | `[--proposal PROPOSAL] [--facility {LCO,SOAR}] [--dry-run]` | ✓ PASS |
| Retired command really gone | `python manage.py help` (`[solsys_code]` block) | `project_observation_calendar` listed; `sync_lco_observation_calendar` absent | ✓ PASS |
| Real `updatestatus` logs no `unprojectable` | `grep -c unprojectable tmp/34-06-updatestatus.txt` | `0` | ✓ PASS |
| Post-`updatestatus` dry-run sweep reports no work left | `tail tmp/34-06-dry-run.txt` | `Done (dry run). failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159, …` | ✓ PASS |
| Developer DB untouched | `stat -c %Y src/fomo_db.sqlite3` + `git status --porcelain` | `1789157248`, clean | ✓ PASS |
| SCHED-06 baseline untouched | `git log -- …sched06-baseline.json` | only `a87f5f8` | ✓ PASS |
| Notebook takeover demonstration non-empty | read cell 9 executed output | `0 of 159 … re-titled`, empty sample list | ✗ FAIL (truth 10) |
| Notebook first/second sweep lines differ | read cell 10 executed output | byte-identical all-zero lines | ✗ FAIL (truth 10) |
| Live narrowing over real nights | — | requires elapsed observing time | ? SKIP → human (truth 4) |
| Interleaved concurrent saves | — | no concurrency harness; single-threaded suite | ? SKIP → human (truth 13) |

**Note on DB safety:** no `updatestatus`, no `project_observation_calendar` (dry-run or otherwise) and no notebook execution was performed by this verification. Every live number above is read from committed executed output or from the gitignored evidence files plan 34-06 produced.

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN/SUMMARY declares a probe | N/A — skipped |

### Decision Coverage

`gsd_run query check.decision-coverage-verify` → `{ total: 21, honored: 21, not_honored: [] }`.
All trackable CONTEXT.md decisions are honored by shipped artifacts. **Non-blocking; no status impact.**

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROJ-01 | 34-01, 34-02 | Exactly one `CalendarEvent` per LCO/SOAR record, keyed by `facility.get_observation_url()` | ✓ SATISFIED | Truth 1; notebook cell 12 (`159 - 0 = 159`); `test_two_records_whose_windows_exactly_abut_produce_two_separate_events` |
| PROJ-02 | 34-01, 34-05, 34-06 | Span follows the stage: request window → placed block → observed block | ✓ SATISFIED | Truth 1 + truth 2; `TestEventFieldsFor`; `coerce_schedule_datetime()` closes the portal-string path |
| PROJ-03 | 34-01, 34-03 | Terminal-negative record keeps a visibly marked event | ✓ SATISFIED | Truths 1, 7, 8; `[X]`/`[C]`/`[F]` marker tests + ring + legend tests |
| PROJ-04 (title-stem clause) | **not declared in any plan's `requirements:`** | Series identity carried by real FKs, not text in the title | ✓ SATISFIED but ⚠️ **ORPHANED** | REQUIREMENTS.md maps the title-stem clause to Phase 34, yet no plan frontmatter claims `PROJ-04`. It is nonetheless delivered — truths 9 and 11 — so this is a traceability gap in the plans, not a delivery gap. |
| PROJ-05 | 34-01, 34-02, 34-03 | No-churn; never touches an event it does not own | ✓ SATISFIED | `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`; `TestNamespaceIsolation` in both projector and sweep suites; notebook cell 9 asserts `RUN:`/`GEM:`/blank-url families unchanged |
| PROJ-06 | 34-01, 34-03 | Compact titles that fit a month cell | ✓ SATISFIED | Truth 5; `test_marker_and_token_within_first_16_characters` |
| SCHED-06 | 34-04, 34-05, 34-06 | A user watches a record narrow over real nights with no command | ? NEEDS HUMAN | Truth 4 — baseline committed, mechanism fixed and wired, evidence requires elapsed observing time (UAT Test 4) |
| TRIG-01 | 34-01, 34-05, 34-06 | `post_save` receiver in `apps.ready()`, covering the schedule-only and `updatestatus` paths | ✓ SATISFIED | Truth 2; `apps.py:ready()`; `TestUpdateObservationStatusPath` (4 tests) |
| TRIG-02 | 34-01, 34-05, 34-06 | Single-record, idempotent, cheap, error-logged-never-aborts | ✓ SATISFIED | Truth 2; `test_make_request_is_never_called_during_a_record_save`, `test_raising_projector_does_not_block_a_save`, `test_record_saved_inside_a_rolled_back_transaction_leaves_no_event` |
| TRIG-03 | 34-02, 34-04 | Sweep command with `--dry-run`, failure isolation, and a paired pre-executed demo notebook | ✗ BLOCKED (partial) | Command and behaviour ✓ (truth 3). The **paired demo notebook clause** fails: its committed executed output no longer demonstrates the takeover or a first-vs-second sweep difference (truth 10). |
| ANNOT-03 | 34-02, 34-04 | Old LCO sync retired; runbook/notebook/tests migrated; Gemini caveat documented | ✗ BLOCKED (partial) | Retirement, runbook migration, test migration and Gemini caveat all ✓ (truth 6). The migrated notebook's takeover evidence — the thing that proves the retirement was a plain in-place takeover — is now empty (truth 10). |

**Orphaned requirements:** `PROJ-04` (title-stem clause) is mapped to Phase 34 in REQUIREMENTS.md but appears in no plan's `requirements:` field. Delivered anyway; flagged for traceability.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `test_observation_projector.py` | PROJ-01/02/03/05/06, PROJ-04 stem | 49 | 0 | No | Value + behavioral | ✓ Sufficient |
| `test_observation_projector_signals.py` | TRIG-01/02, PROJ-02 | 23 | 0 | No | Behavioral (drives the real `update_observation_status()`) | ✓ Sufficient |
| `test_project_observation_calendar.py` | TRIG-03, PROJ-05 | 25 | 0 | No | Value + behavioral | ✓ Sufficient |
| `test_calendar_utils.py` | PROJ-02 (G-34-2) | incl. `TestCoerceScheduleDatetime` (5) + `TestRecordTimeWindow` | 0 | No | Value | ✓ Sufficient |
| `test_calendar_display_extras.py` / `test_calendar_template.py` | PROJ-03/05/06, PROJ-04 stem | many | 0 | No | Value + rendered-markup | ✓ Sufficient |

**Disabled tests on requirements:** 0. **Circular patterns detected:** 0 (no fixture-generating script imports the system under test; the SCHED-06 baseline JSON is a *record of real DB state*, explicitly diffed by a human, not an assertion oracle). **Insufficient assertions:** 0.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/views.py` | 310 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | **Pre-existing** — `git blame` attributes it to `a8613bc8` (2025-07-23), 14 months before this phase. Not introduced here; classified identically by the previous verification. Debt-marker gate does not fire. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | cells 5, 9, 10, 19 | Executed output no longer supports the prose/claims around it | 🛑 Blocker | **Regression** on a file modified after the prior `verified:` timestamp (`73b465d` 14:28, `46d8390` 15:07 vs. prior verification 05:33Z) — blocks unconditionally under the convergence evidence gate. See truth 10 / `gaps`. |

Every `TBD` hit in the changed files is the `CampaignRun` domain term ("a TBD window" = window not yet resolved), pre-dating this phase. No `FIXME`, no `TODO`, no `HACK`, no `PLACEHOLDER` in any file this phase modified.

### Human Verification Required

#### 1. SCHED-06 live narrowing (UAT Test 4 — merge into the existing `34-UAT.md`, do NOT overwrite it)

**Test:** From now, run **only** `python manage.py updatestatus` against `src/fomo_db.sqlite3` over several real observing nights — never `project_observation_calendar`. Then re-execute `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` end to end **un-routed** (no `FOMO_DATABASE_PATH`) and `git diff` the SCHED-06 baseline JSON. Read the re-execution's own **first** sweep summary line before anything else.
**Expected:** At least one `KEY2026B-004` record narrowed queued → placed (or placed → observed) with its event following, and the first sweep reports `created: 0, updated: 0` for it — proving the `post_save` receiver, not the sweep, did the narrowing. Fill in the dated row in `34-UAT.md`'s SCHED-06 re-check table and flip Test 4's verdict from `blocked`.
**Why human:** Verification-over-time; depends on real observing nights elapsing. This is the single expected remaining human item, and it is already **Test 4** in `34-UAT.md` — the orchestrator should merge this outcome into that tracker rather than create a new UAT file.
**Now unblocked:** the original `blocked_by` reason (1) — G-34-2 — is fixed. The 33 stale LCO events on the developer DB should be repaired by the receiver alone on the next real `updatestatus`; that repair is itself part of the evidence.

#### 2. Interleaved-save re-run (UAT Test 2 — previously contaminated by G-34-2)

**Test:** Drive two overlapping `updatestatus` runs against a **copy** of the developer database (`FOMO_DATABASE_PATH=<absolute scratch path>`) and inspect the surviving `CalendarEvent`s.
**Expected:** Each event matches its record's final persisted field state, and the logs carry no `unprojectable … AttributeError` lines.
**Why human:** Declared `verification: backstop` (non-inferable); no test exercises concurrency. The original Test 2 run is unusable as evidence because G-34-2 dominated it.

### Gaps Summary

The phase's **code** delivers the goal. Every LCO/SOAR record draws and keeps current its own facility-URL-keyed calendar event on every save through three `apps.ready()`-wired receivers; the sweep is a real, flag-complete, failure-isolating backstop sharing one projection path with the receiver; and `sync_lco_observation_calendar` is genuinely gone — command, tests, notebook and command-name alike. The G-34-2 blocker that made the previous pass's UAT Test 2 fail is closed, and closed with the strongest evidence available: a behavioural test that drives the real `LCOFacility().update_observation_status()` inside `assertNoLogs(WARNING)`, plus a real portal-backed `updatestatus` run producing zero `unprojectable` lines and a follow-up dry-run sweep with nothing left to do. 1127 tests pass. The developer database and the SCHED-06 baseline are both provably untouched.

One gap blocks: **the paired demo notebook's executed output stopped demonstrating the one thing it exists to demonstrate.** Plan 34-06 and the review-fix round each re-executed the notebook against `tmp/fomo_g34_2_copy.sqlite3` — a scratch copy that 34-06's own live-proof step had already swept to convergence minutes earlier. The result is a "one-time takeover" section reporting `0 of 159 … re-titled` with an empty before→after sample list, and a convergence section whose first and second sweep summary lines are byte-identical all-zero. Two of plan 34-04's must_haves state the opposite explicitly, and the notebook's own markdown ("this run's first sweep is expected to report a non-zero `updated` count for LCO") is now contradicted by the cell beneath it. At the previous verification this section was intact (`156 of 156` re-titled, first sweep `created: 3, updated: 156, site_lookups: 59`), so this is a regression, not a pre-existing shortfall — and CLAUDE.md's paired-docs rule directs the verifier to treat a stale notebook as a must-have gap rather than a nice-to-have.

The fix is small and does not require touching the developer database: take a **fresh, un-swept** clone of `src/fomo_db.sqlite3`, re-execute the notebook against it, and add an assertion to cell 10 that the two sweep summary lines actually differ so the demonstration can never silently empty out again.

Two items remain for a human, both expected: UAT Test 4 (SCHED-06 live narrowing — needs real observing nights) and a clean re-run of UAT Test 2 (interleaved saves, whose original run was contaminated by the now-fixed G-34-2).

---

_Verified: 2026-09-11T23:05:00Z_
_Verifier: Claude (gsd-verifier)_
