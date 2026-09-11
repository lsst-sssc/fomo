---
phase: 34-the-observation-projector-trigger
verified: 2026-09-11T05:33:49Z
status: human_needed
score: 11/14 must-haves verified
behavior_unverified: 1
overrides_applied: 0
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
behavior_unverified_items:
  - truth: "Over real nights a pending KEY2026B-004 record's event narrows queued -> scheduled -> observed with nobody running anything (SCHED-06, ROADMAP criterion 4)."
    test: "From now, run ONLY `python manage.py updatestatus` over several real observing nights -- never `python manage.py project_observation_calendar`. Then re-execute `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` end to end and diff its SCHED-06 section against `project_observation_calendar_demo.sched06-baseline.json` (74 pending records at baseline: 56 queued, 18 placed)."
    expected: "At least one KEY2026B-004 record has moved queued -> placed (or placed -> observed) and its calendar event span/title narrowed to match, with no sweep run in between. Record the outcome in the dated re-check table in `34-UAT.md` and flip its verdict from PARTIAL."
    why_human: "SCHED-06 is a verification-over-time requirement -- it depends on the real LCO scheduler placing and observing real requests on real nights. No test or grep can produce that evidence; only elapsed observing time can. Plan 34-04 deliberately established the baseline and left the re-check open."
    advisory: "The re-executed notebook runs two real sweeps (cells 8 and 10) BEFORE it re-captures the SCHED-06 baseline, so the re-captured snapshot alone cannot prove which writer narrowed the events. The discriminator is the re-execution's OWN first-sweep summary line: if it reports `created: 0, updated: 0` for the narrowed records, the post_save receiver had already narrowed them before the sweep ran. Read that line first, and note it in the 34-UAT.md row."
insufficient_spec_items:
  - truth: "If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state (34-01 must_haves, verification: backstop)."
    test: "Drive two concurrent/interleaved saves of one LCO ObservationRecord (e.g. two `updatestatus` runs overlapping, or two request threads saving the same record) and inspect the resulting CalendarEvent span and title against the record's final persisted fields."
    expected: "The surviving event matches the record's final persisted scheduled_start/scheduled_end/status -- no event left describing a superseded intermediate state."
    why_human: "Declared non-inferable (`verification: backstop`). No held-out or property-based test exercises interleaved saves; the suite's tests are all single-threaded. Presence and wiring cannot establish a concurrency invariant."
  - truth: "A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step (34-02 must_haves, verification: backstop)."
    test: "Interrupt `python manage.py project_observation_calendar` partway (Ctrl-C mid-run) against the developer database, then re-run it to completion and inspect the summary line."
    expected: "Every record processed before the interrupt still carries a correct event; the re-run converges (created: 0, updated: 0 for already-processed records) with no repair or cleanup step."
    why_human: "Declared non-inferable (`verification: backstop`). No test simulates a partway interrupt. The observed full-convergence run (159 unchanged) is evidence of idempotency, not of interrupt-safety."
human_verification:
  - test: "From now, run ONLY `python manage.py updatestatus` over several real observing nights (never the sweep), then re-execute `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` and diff its SCHED-06 section against the committed baseline JSON. Read the re-execution's own FIRST sweep summary line before anything else."
    expected: "At least one KEY2026B-004 record narrowed queued -> placed (or placed -> observed) with its event following, and the first sweep reports created: 0, updated: 0 for it -- proving the post_save receiver, not the sweep, did the narrowing. Fill in the dated row in 34-UAT.md and close SCHED-06."
    why_human: "Verification-over-time requirement; depends on real observing nights elapsing. Closes spike 004's PARTIAL verdict."
  - test: "Open the calendar month view in a browser (`python manage.py runserver`, then the calendar page) on a month containing KEY2026B-004 entries. Check the marker legend row, the status rings on [Q]/[X]/[C]/[F] chips, the month-cell titles, and open an event modal for a record that belongs to an ObservationGroup."
    expected: "The legend lists [Q] Queued, [S] Scheduled, [O] Observed, [X] Window expired, [C] Cancelled, [F] Failed, [?] Inconsistent record. [Q] chips carry the dark queued ring and [X]/[C]/[F]/[?] the red terminal ring, while [S]/[O] carry none. Each month-cell title reads legibly within its truncation budget -- the marker and telescope token are both visible. The modal shows an 'Observation series' block with the group name, 'Night n of N', and working links, rendered beside (not overwriting) the campaign block."
    why_human: "Visual appearance, ring contrast against real chip colours, and month-cell title legibility are judgment calls no grep or template test can settle. 296 automated tests confirm the markup is produced; they cannot confirm it reads well."
  - test: "Drive two interleaved saves of the same LCO ObservationRecord and inspect the surviving CalendarEvent."
    expected: "The event matches the record's final persisted field state, not a superseded intermediate one."
    why_human: "Declared `verification: backstop` (non-inferable); no test exercises concurrency."
  - test: "Interrupt `project_observation_calendar` partway, then re-run it to completion."
    expected: "Already-processed records keep correct events; the re-run converges with no repair step."
    why_human: "Declared `verification: backstop` (non-inferable); no test simulates a partway interrupt."
deferred:
  - truth: "The two title-prefix vocabularies (the legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` prefixes still emitted by load_telescope_runs.py and campaign_views.py, and the projector's terse `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers) are unified into one vocabulary."
    addressed_in: "Phase 37"
    evidence: "ROADMAP Phase 37 'Status vocabulary, public tallies and provenance blind gaps' (STATUS-01/02) owns the final vocabulary; Phase 34's ROADMAP scope note states 'Title prefixes ship provisionally here; Phase 37 owns the final vocabulary.' Both vocabularies paint the correct ring today (calendar_display_extras._TERMINAL_PREFIXES carries all of them)."
---

# Phase 34: The Observation Projector & Trigger — Verification Report

**Phase Goal:** Every LCO/SOAR observation record draws its own calendar event and keeps it current on every save with no operator command, and the old LCO sync command is retired in its favour — one writer for observation-backed nights.
**Verified:** 2026-09-11T05:33:49Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **(SC1)** Every LCO/SOAR `ObservationRecord` has exactly one calendar event keyed by its facility observation URL, spanning the request window while queued / the placed block once scheduled / the observed block once observed, proven across all real `KEY2026B-004` records; a terminal-negative record keeps a visibly marked event on its window night. | ✓ VERIFIED | Live query against the developer DB (not the SUMMARY): 159 LCO/SOAR records, **146** of them `KEY2026B-004`; **0** KEY2026B-004 records lack a facility-URL-keyed event; **0** duplicate facility URLs. Spot-check of one record in each terminal-negative status: `WINDOW_EXPIRED` → `'[X] 1m0 10P'`, `CANCELED` → `'[C] 1m0 259P'`, `FAILURE_LIMIT_REACHED` → `'[F] 1m0 10P'`, each spanning exactly its submitted `parameters['start'/'end']` window. `observation_projector.stage_for()`/`event_fields_for()` derive the span from record fields alone (`record_time_window()` for the placed/observed block, `parameters` for queued/inconsistent). Tests: `test_observation_projector.py#TestEventFieldsFor` (11 span/marker tests, all stages). |
| 2 | **(SC2)** Saving a record updates its event with no operator command — including the schedule-only placement save TOM's own hook misses and the `updatestatus` path — while a save that changes nothing writes nothing and a projector error is logged rather than aborting the record save. | ✓ VERIFIED | `apps.py:ready()` connects `post_save`(ObservationRecord), `m2m_changed`(group through-model) and `pre_delete` with `weak=False` + `dispatch_uid`. Behavioral tests, all passing: `test_schedule_only_save_narrows_the_same_event_row`, `test_updatestatus_narrows_the_event_with_no_command_run`, `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`, `test_raising_projector_does_not_block_a_save`, `test_record_saved_inside_a_rolled_back_transaction_leaves_no_event`, `test_make_request_is_never_called_during_a_record_save`, `test_raw_save_writes_no_calendar_event`, `test_queryset_update_bypasses_the_receiver_and_writes_no_new_event`. Code read confirms `receiver_on_record_save` returns early on `raw=True` / non-LCO-SOAR facility and wraps `project_record()` in its own `try/except`. |
| 3 | **(SC3)** One sweep command re-projects any set of records (`--dry-run` supported, per-record failures isolated) as the backstop for bulk-write paths; a second sweep reports everything unchanged and no event outside the projector's key namespace is created, modified or deleted. | ✓ VERIFIED | `python manage.py project_observation_calendar --help` runs; zero required args, `--proposal`/`--facility {LCO,SOAR}`/`--dry-run`. **Live run performed by this verifier** against the real developer DB: `Done (dry run). failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159, ... \| SOAR: ... 0`. Notebook (executed output) shows first sweep `updated: 156, created: 3, site_lookups: 59` and second `unchanged: 159, created: 0, updated: 0`, with `RUN:` (72), `GEM:` (0) and blank-url (10) event sets asserted byte-identical across the takeover. Tests: `TestDryRun`, `TestFailureIsolation`, `TestNamespaceIsolation`, `TestProjectQuerysetOrdering`, `TestBareInvocationAndSummary` (22 tests). |
| 4 | **(SC4a)** Over real nights, a user watches a `KEY2026B-004` record's event narrow queued → scheduled → observed with nobody running anything (closes spike 004's PARTIAL verdict). | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | The mechanism is present, wired and unit-tested (truth 2), and the SCHED-06 baseline is committed: 74 pending records (56 queued, 18 placed) snapshotted at 2026-09-11T04:44:59Z into `project_observation_calendar_demo.sched06-baseline.json` + the notebook's SCHED-06 section, with an explicit "run only `updatestatus`, never the sweep" rule and a dated re-check table in `34-UAT.md` (verdict recorded as PARTIAL / open). Per the ROADMAP note this is a verification-over-time requirement — routed to human verification, **not** counted as a gap. |
| 5 | **(SC4b)** Every event title is short enough to read in a month cell (PROJ-06). | ✓ VERIFIED | `title_for()` produces `'[marker] <token> <target>'` capped at 200 chars; `test_marker_and_token_within_first_16_characters` asserts the marker plus telescope token fit inside the month cell's tightest `truncatechars:16` budget. `calendar.html` truncation budgets (`truncatechars:18` line 254, `truncatechars:16` line 282) are **unchanged** by this phase's diff. Real titles confirm: `'[X] 1m0 10P'`, `'[O] FTS Didymos COJ 2026 Field #02'`. Visual legibility folded into human verification item 2. |
| 6 | **(SC5)** `sync_lco_observation_calendar` no longer exists; the same events come from the projector and sweep in the same key namespace, with runbook section, demo notebook and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-read-back caveat documented. | ✓ VERIFIED | Command module, its 38-test module and its notebook all deleted (commits `d2e9daf`, `a87f5f8`); `git log --diff-filter=D` confirms. Key namespace preserved: the takeover updated 156 pre-existing `https://observe.lco.global/requests/...` events **in place** (no migration script, no new keys). 34-02-SUMMARY carries a 38-row classification table (16 already-covered, 10 migrated with named destination tests, 12 retired each with a stated reason) — spot-checked rows 5/6/7/15/18/21/27-31/38 all carry reasons. Gemini command untouched by the diff; its notebook gained a "No observation-status or observation-URL read-back (Phase 34, D-21)" markdown cell and the runbook lines 351-361 carry the operator-facing version. |
| 7 | A calendar visitor can tell what each projector marker means from a legend on the calendar page itself, including `[?]`. | ✓ VERIFIED | `calendar_display_extras.observation_status_legend()` returns the 7-entry `_OBSERVATION_STATUS_LEGEND` ([Q]/[S]/[O]/[X]/[C]/[F]/[?]); `calendar.html:317-324` renders it in the legend row. Covered by `test_calendar_display_extras.py` / `test_calendar_template.py` (129 tests, all passing). |
| 8 | Every projector marker paints the right status ring, and no existing ring is lost. | ✓ VERIFIED | `status_border_css()` adds a `[Q] ` branch beside the legacy `[QUEUED] ` branch; `_TERMINAL_PREFIXES` now carries `('[EXPIRED]', '[CANCELLED]', '[FAILED]', '[WEATHERED]', '[X] ', '[C] ', '[F] ', '[?] ')` — all four legacy verbose prefixes retained byte-identical, so reconciler/classical rings are unaffected. `[S] `/`[O] ` deliberately return `''` (placed bucket). |
| 9 | Series identity ("night n of N") is rendered at request time from `CalendarEventMeta.observation_group`, never stored in the event, ordered by window start, with no per-event query fan-out. | ✓ VERIFIED | `observation_series_decoration()` read only — code read confirms no `save`/`update`/`create`/`get_or_create` call anywhere in its body; returns `None` for missing companion row, missing group/record link, and groups with `< 2` members; sorts members by `(_window_start_or_max(member), member.pk)`. Rendered at `event_form.html:148-159` beside (not replacing) `campaign_decoration` at line 161. `views.fomo_render_calendar` extends the existing `Prefetch` with `select_related('run__campaign', 'observation_record__target', 'observation_group')`. |
| 10 | The paired-docs rule is satisfied: the sweep has a pre-executed demo notebook registered in `docs/notebooks.rst` and CLAUDE.md's map; the retired command's notebook is gone; the runbook describes the projector/sweep, the legend, the series block and the Gemini caveat. | ✓ VERIFIED | `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` — 20 cells, **12/12 code cells carry committed output**; git-tracked along with its `.sched06-baseline.json`. `docs/notebooks.rst:15` toctree entry present; `CLAUDE.md:128-129` notebook-map entry present. `sync_lco_observation_calendar_demo.ipynb` deleted. Runbook covers: marker legend (L95-101), Observation series block (L103-118), one-time title change (L119-123), projector/sweep section + flags + real summary line (L136-158), Gemini no-read-back caveat (L351-361), cheat-sheet entry for the sweep with no entry for the retired command (L1039), troubleshooting rewritten to `unprojectable`/`site_lookup_failed` with an explicit "there is no `[UNVERIFIED]` in this vocabulary" note (L1106-1113). `pre-commit run sphinx-build --all-files` → **Passed** (toctree and notebook both build). |
| 11 | Nothing group-derived is written into `CalendarEvent.title` or `.description`; series identity lives only in `CalendarEventMeta.observation_group` (PROJ-04 title-stem clause). | ✓ VERIFIED | `event_fields_for()` builds description from proposal/status/stage/window only; `title_for()` from marker/token/target only — neither reads `series_group_for()`. `write_event_meta()` writes exactly `is_verified`/`observation_record`/`observation_group` and never the attribution link or its stamps. Tests: `test_grouped_record_sets_observation_group_and_omits_name_from_title_and_description`, `test_existing_campaign_attribution_survives_projection_and_is_verified_becomes_true`, `test_record_in_two_groups_links_the_lowest_pk_group`. |
| 12 | The observed telescope is resolved once, stored on the record, read back as a network-free title token, and the D-07 label rename does not downgrade Phase 28's site-level attribution matching. | ✓ VERIFIED | `resolve_observed_site()` lives on the command side (never in the projector), skips non-terminal stages and already-resolved records, stores `observed_site`/`observed_telescope`/`observed_enclosure`. `SITE_TELESCOPE_MAP` renamed to `FTS`/`FTN`/`SOAR`; `OBSERVED_TELESCOPE_SITE_CODES` is the inverse bridge consulted **first** by `campaign_attribution._extract_lco_site_code()`. `campaign_lifecycle_demo.ipynb` cell 22 executed output: `FTN -> 'ogg' -> 'F65'`, `FTS -> 'coj' -> 'E10'`, `SOAR -> 'sor' -> 'I33'`, and `FTN match level: 1.0 (TELESCOPE_MATCH_SITE)`. |
| 13 | If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state. | ⚠️ insufficient_spec (abstained) | Declared `verification: backstop` in 34-01 must_haves. No held-out, property-based or concurrency test exercises interleaved saves — the whole suite is single-threaded. Presence + wiring cannot establish this. Routed to human verification. |
| 14 | A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step. | ⚠️ insufficient_spec (abstained) | Declared `verification: backstop` in 34-02 must_haves. The observed full-run convergence (159 unchanged on a live dry-run) proves idempotency, not interrupt-safety; no test simulates a partway interrupt. Routed to human verification. |

**Score:** 11/14 truths verified (1 present, behavior-unverified; 2 abstained as non-inferable)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Unification of the two title-prefix vocabularies (legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` vs. the projector's terse bracket-letter markers) | Phase 37 | ROADMAP Phase 34 scope note: "Title prefixes ship provisionally here; Phase 37 owns the final vocabulary." Phase 37 = "Status vocabulary, public tallies and provenance blind gaps" (STATUS-01/02). Both vocabularies already paint the correct ring, so nothing is broken today — only unreconciled. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/observation_projector.py` | Projector + 3 receivers | ✓ VERIFIED | 594 lines (new). 18 top-level functions; imported by `apps.py`, `project_observation_calendar.py`, `test_*` and the demo notebook. |
| `solsys_code/apps.py` | `ready()` signal wiring | ✓ VERIFIED | +41 lines; three `.connect()` calls with `weak=False` + `dispatch_uid`, function-local imports. |
| `solsys_code/management/commands/project_observation_calendar.py` | The sweep | ✓ VERIFIED | 205 lines (new); `--help` runs; live `--dry-run` executed successfully. |
| `solsys_code/calendar_utils.py` | Retained helpers + renamed map + new keys | ✓ VERIFIED | `SITE_TELESCOPE_MAP` (FTN/FTS/SOAR rename), `OBSERVED_TELESCOPE_SITE_CODES`, `OBSERVED_SITE_PARAMETER_KEYS`, `resolve_placement_block`, `derive_telescope`, `aperture_class_from_telescope_code`, `extract_instrument`, `record_time_window`, `preview_calendar_event_action`, `insert_or_create_calendar_event` all present. |
| `solsys_code/campaign_attribution.py` | D-07 bridge in `_extract_lco_site_code()` | ✓ VERIFIED | Bridge checked before the split-on-dash parse; `LCO_SITE_CODE_TO_OBSCODE` gained `ogg→F65`, `sor→I33`. |
| `solsys_code/templatetags/calendar_display_extras.py` | Rings, legend, series tag | ✓ VERIFIED | +175 lines; `observation_status_legend()`, `observation_series_decoration()`, extended `_TERMINAL_PREFIXES`, `[Q] ` branch. |
| `src/templates/tom_calendar/partials/calendar.html` | Legend row | ✓ VERIFIED | +8 lines at 317-324; truncation budgets unchanged. |
| `src/templates/tom_calendar/partials/event_form.html` | Series block | ✓ VERIFIED | +21 lines at 141-159; renders beside the campaign block. |
| `solsys_code/views.py` | Prefetch widened | ✓ VERIFIED | `select_related('run__campaign', 'observation_record__target', 'observation_group')`. |
| `solsys_code/tests/test_observation_projector.py` | Stage/marker/no-churn coverage | ✓ VERIFIED | 43 tests, 0 skipped. |
| `solsys_code/tests/test_observation_projector_signals.py` | Trigger coverage | ✓ VERIFIED | 19 tests, 0 skipped — includes rollback, no-HTTP, raw-save, queryset.update bypass. |
| `solsys_code/tests/test_project_observation_calendar.py` | Sweep coverage | ✓ VERIFIED | 22 tests, 0 skipped. |
| `solsys_code/tests/test_calendar_display_extras.py` | Ring/legend/series coverage | ✓ VERIFIED | 73 tests, 0 skipped. |
| `solsys_code/tests/test_calendar_template.py` | Template render coverage | ✓ VERIFIED | 56 tests, 0 skipped. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | Pre-executed sweep demo | ✓ VERIFIED | 20 cells, 12/12 code cells with committed output. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json` | SCHED-06 baseline | ✓ VERIFIED | Git-tracked; `captured_at`/`proposal`/`record_count` + 74 per-record entries. |
| `docs/notebooks.rst` | Toctree entry | ✓ VERIFIED | Line 15. |
| `docs/runbooks/telescope_runs_calendar.rst` | Projector/sweep/legend/series/Gemini | ✓ VERIFIED | +205/-... rewrite; all required sections present (see truth 10). |
| `CLAUDE.md` | Notebook map entry | ✓ VERIFIED | Lines 128-129. |
| `.planning/phases/34-.../34-UAT.md` | SCHED-06 tracker | ✓ VERIFIED | Baseline date, rule, dated re-check table, PARTIAL verdict. |
| `solsys_code/management/commands/sync_lco_observation_calendar.py` | **must NOT exist** | ✓ VERIFIED (absent) | Deleted in `d2e9daf`. |
| `solsys_code/tests/test_sync_lco_observation_calendar.py` | **must NOT exist** | ✓ VERIFIED (absent) | Deleted in `d2e9daf`. |
| `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb` | **must NOT exist** | ✓ VERIFIED (absent) | Deleted in `a87f5f8`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ObservationRecord.save()` | `CalendarEvent` | `post_save` → `receiver_on_record_save()` → `project_record()` → `insert_or_create_calendar_event({'url': facility URL}, fields)` | ✓ WIRED | Connected in `apps.ready()`; end-to-end proven by `test_create_projects_a_queued_event_over_the_request_window` and `test_updatestatus_narrows_the_event_with_no_command_run`. |
| `project_record()` | `CalendarEventMeta.observation_record/.observation_group` | `write_event_meta()` | ✓ WIRED | Phase 33 carrier fields populated; stale one-to-one claims cleared first (`test_stale_companion_claim_on_a_different_event_is_cleared_not_integrity_error`). |
| `SolsysCodeConfig.ready()` | the three receivers | `post_save`/`m2m_changed`/`pre_delete` `.connect()` | ✓ WIRED | Without this nothing fires; the live dry-run and 296 passing tests exercise it. |
| `ObservationGroup.observation_records.through` | `project_record()` | `m2m_changed` → `receiver_on_group_membership_changed()` | ✓ WIRED | `post_add`/`post_remove`/`pre_clear`+`post_clear`/reverse-direction all handled; 5 dedicated tests. |
| `telescope_token()` | `CalendarEvent.telescope` + title token → `campaign_attribution._extract_lco_site_code()` | `OBSERVED_TELESCOPE_SITE_CODES` | ✓ WIRED | `test_telescope_field_equals_title_token_for_every_stage`; attribution bridge proven in the lifecycle notebook (`FTN match level: 1.0`). |
| sweep per-record loop | `observation_projector.project_queryset()` → `project_record()` | shared `preview_calendar_event_action()` comparison | ✓ WIRED | One projection path for receiver and sweep; `test_dry_run_writes_nothing_and_matches_the_subsequent_real_run`. |
| `resolve_placement_block()` + `derive_telescope()` | `record.parameters['observed_site'/'observed_telescope']` → `telescope_token()` | `pre_fields_hook` | ✓ WIRED | `TestObservedSiteLookup` (8 tests); notebook first sweep `site_lookups: 59`, second `0`. |
| `CalendarEventMeta.observation_group` | `event_form.html` "Observation series" block | `observation_series_decoration()` at request time | ✓ WIRED | Tag registered, called at `event_form.html:148`; no write path. |
| `fomo_render_calendar` Prefetch | companion row's `observation_group`/`observation_record` | `Prefetch(... select_related(...))` | ✓ WIRED | Prevents the new modal tag becoming an N+1. |
| `docs/notebooks.rst` toctree | new notebook path | Sphinx toctree | ✓ WIRED | `sphinx-build` passes. |
| notebook baseline snapshot | post-nights re-execution → SCHED-06 verdict | `34-UAT.md` dated re-check table | ⚠️ PENDING | Baseline and instructions are in place; the closing observation has not happened yet (by design). |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `CalendarEvent.title` | `marker`/`token`/`target_name` | `record.status`, `facility.get_failed_observing_states()`, `record.parameters`, `record.target.name` | Yes — real titles observed in the dev DB (`'[X] 1m0 10P'`, `'[O] FTS Didymos COJ 2026 Field #02'`) | ✓ FLOWING |
| `CalendarEvent.start_time`/`end_time` | `record_time_window(record)` / `parameters['start'/'end']` | `ObservationRecord.scheduled_start/_end` and submitted window | Yes — placed records span ~16 min blocks, queued span ~1 day windows (baseline snapshot) | ✓ FLOWING |
| `CalendarEvent.telescope` | `telescope_token()` | `record.parameters['observed_site'/'observed_telescope']` → `derive_telescope()`, else `coarse_telescope_label()` | Yes — 59 real site lookups resolved in the takeover sweep | ✓ FLOWING |
| `CalendarEventMeta.observation_group` | `series_group_for(record)` | `ObservationGroup.objects.filter(observation_records=record)` | Yes — real ORM query, lowest-pk tiebreak | ✓ FLOWING |
| Month-cell status ring | `status_border_css(event.title)` | real `CalendarEvent.title` | Yes — 18 `[X]`, 6 `[C]`, 1 `[F]`, 56 `[Q]` real events in the dev DB | ✓ FLOWING |
| Modal "night n of N" | `observation_series_decoration(event)` | `CalendarEventMeta` links → group members sorted by window start | Yes — computed per request, never stored | ✓ FLOWING |
| Legend row | `observation_status_legend()` | fixed 7-entry vocabulary constant | Static by design (a legend is a vocabulary, not data) | ✓ FLOWING (intentionally fixed) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase test modules pass | `python manage.py test solsys_code.tests.test_observation_projector test_observation_projector_signals test_project_observation_calendar test_calendar_display_extras test_calendar_template test_calendar_utils test_campaign_attribution` | `Ran 296 tests in 24.155s — OK` | ✓ PASS |
| Sweep command is runnable with zero required args | `python manage.py project_observation_calendar --help` | usage line shows `[--proposal] [--facility {LCO,SOAR}] [--dry-run]`, nothing required | ✓ PASS |
| Second sweep converges on the real corpus | `python manage.py project_observation_calendar --dry-run` | `Done (dry run). failed: 0 \| LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 \| SOAR: all zero` | ✓ PASS |
| Every real `KEY2026B-004` record has exactly one facility-URL event | Django shell count over `ObservationRecord.objects.filter(facility__in=('LCO','SOAR'))` | 159 records / 146 KEY2026B-004 / 0 missing events / 0 duplicate URLs | ✓ PASS |
| Terminal-negative records keep a marked window-night event | Django shell spot-check per failure status | `[X]`/`[C]`/`[F]` titles, each spanning exactly its submitted window (`window_match=True` ×3) | ✓ PASS |
| Lint gate clean | `pre-commit run ruff --all-files` | Passed | ✓ PASS |
| Format gate clean | `pre-commit run ruff-format --all-files` | Passed | ✓ PASS |
| Docs build (toctree + notebook) | `pre-commit run sphinx-build --all-files` | Passed | ✓ PASS |
| Retired command unregistered | `ls solsys_code/management/commands/` | `sync_lco_observation_calendar.py` absent | ✓ PASS |
| Live narrowing over real nights | — | Requires elapsed observing time | ? SKIP → human verification |

### Probe Execution

N/A — this project has no `scripts/*/tests/probe-*.sh` convention and neither the PLANs nor the SUMMARYs declare probes. The Django test runner plus the live command runs above are the equivalent evidence.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `test_observation_projector.py` | PROJ-01/02/03/05/06 | 43 | 0 | No | Value + behavioral (exact titles, exact spans, `modified` timestamp comparison) | ✓ Sufficient |
| `test_observation_projector_signals.py` | TRIG-01/02 | 19 | 0 | No | Behavioral (save → event state, rollback, patched `make_request` never called) | ✓ Sufficient |
| `test_project_observation_calendar.py` | TRIG-03, PROJ-01/05, ANNOT-03 | 22 | 0 | No | Value + behavioral (summary-line counters, byte-identical namespace assertions) | ✓ Sufficient |
| `test_calendar_display_extras.py` | PROJ-03/05/06 | 73 | 0 | No | Value (exact CSS fragments, exact decoration dicts) | ✓ Sufficient |
| `test_calendar_template.py` | PROJ-03/05 | 56 | 0 | No | Behavioral (rendered HTML assertions) | ✓ Sufficient |

**Disabled tests on requirements:** 0
**Circular patterns detected:** 0 — no test module writes fixture files; expected titles/spans are hand-written literals, and the D-07 attribution evidence in `campaign_lifecycle_demo.ipynb` is scored through the independent `telescope_match_score()` path against real Observatory obscodes (F65/E10/I33), not against projector output.
**Insufficient assertions:** 0

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PROJ-01 | 34-01, 34-02 | One `CalendarEvent` per LCO/SOAR record, keyed by `facility.get_observation_url()` | ✓ SATISFIED | Truth 1 — 146/146 KEY2026B-004, 0 duplicates |
| PROJ-02 | 34-01 | Span follows the record's stage | ✓ SATISFIED | Truth 1; `TestEventFieldsFor` |
| PROJ-03 | 34-01, 34-03 | Terminal-negative record keeps a visibly marked event | ✓ SATISFIED | Truth 1 spot-check; rings + legend (truths 7, 8) |
| **PROJ-04** | **(none — see note)** | Series identity carried by `CalendarEventMeta` FKs; title-stem clause | ⚠️ ORPHANED (declaration) / ✓ SATISFIED in substance | REQUIREMENTS.md line 106 maps PROJ-04's title-stem clause to Phase 34, but **no plan's `requirements:` frontmatter lists PROJ-04**. Substantively delivered: truth 11 (nothing group-derived in title/description) and truth 9 (series identity read from the FK at request time). The plans reference "PROJ-04 title-stem clause" inside must-have text but never declare the ID. See Anti-Patterns / Gaps Summary. |
| PROJ-05 | 34-01, 34-02, 34-03 | No-churn; never writes outside its own namespace | ✓ SATISFIED | Truths 2, 3; `TestNamespaceIsolation`; notebook byte-identical assertion |
| PROJ-06 | 34-01, 34-03 | Compact titles that fit a month cell | ✓ SATISFIED | Truth 5 |
| SCHED-06 | 34-04 | User watches a record narrow over live nights | ? NEEDS HUMAN | Truth 4 — baseline + re-check protocol in place; closes only with elapsed observing time |
| TRIG-01 | 34-01 | `post_save` receiver in `apps.ready()`, incl. schedule-only + `updatestatus` | ✓ SATISFIED | Truth 2 |
| TRIG-02 | 34-01 | Single-record, idempotent, inline in the caller's transaction; errors logged not raised | ✓ SATISFIED | Truth 2; rollback + no-HTTP tests |
| TRIG-03 | 34-02, 34-04 | Sweep with `--dry-run`, per-record isolation, paired demo notebook | ✓ SATISFIED | Truths 3, 10 |
| ANNOT-03 | 34-02, 34-04 | Retire `sync_lco_observation_calendar`; Gemini caveat documented | ✓ SATISFIED | Truth 6 |

**Orphan check:** REQUIREMENTS.md maps 11 IDs to Phase 34 (PROJ-01..06, SCHED-06, TRIG-01..03, ANNOT-03). The plans declare 10 — **PROJ-04 is declared by no plan** despite REQUIREMENTS.md line 106 assigning its title-stem clause here and line 18 stating "Phase 34 must satisfy the title-stem clause alongside PROJ-06's compact-title requirement." Its substance shipped; only the traceability declaration is missing.

### Decision Coverage

`gsd_run query check.decision-coverage-verify` → **21/21 trackable CONTEXT.md decisions honored by shipped artifacts; none not honored.** (Non-blocking gate.)

### Prohibition Disposition (judgment tier — human sign-off at the end-of-phase checkpoint)

All 15 declared prohibitions are `verification: judgment`, `status: resolved`. None is test-tier, so the fail-closed test-tier rule does not apply. Verifier assessment below is a **non-authoritative LLM-judge verdict**; the items marked "advisory" warrant a human glance.

| # | Plan | Prohibition (abbreviated) | Verifier disposition | Evidence |
|---|------|---------------------------|----------------------|----------|
| 1 | 34-01 | No network call / sun-event / external dependency in the `post_save` path | Upheld | `test_make_request_is_never_called_during_a_record_save`; no `requests`/`resolve_placement_block` import in `observation_projector.py` |
| 2 | 34-01 | Never create/modify/delete an event outside its own key namespace | Upheld | `test_run_gem_and_blank_url_events_are_byte_identical_after_a_projection`; all writes keyed on `{'url': event_url(...)}` |
| 3 | 34-01 | A projector failure must not abort the triggering record save | Upheld | Three `try/except` wrapped receivers; `test_raising_projector_does_not_block_a_save/_membership_change/_delete` |
| 4 | 34-01 | Must not write the campaign attribution link or its confirmation stamps | Upheld | `write_event_meta()` defaults dict contains exactly 3 keys; `test_existing_campaign_attribution_survives_projection_...` |
| 5 | 34-02 | Never interpolate a caught portal exception into a log line or output | Upheld | `resolve_observed_site()` has no `except`; every projector/sweep log uses `type(exc).__name__`; `test_failure_message_is_fixed_and_never_leaks_exception_content` |
| 6 | 34-02 | `--dry-run` must write nothing | Upheld | Double-guarded (hook passed as `None` **and** `project_queryset` skips it); `test_dry_run_writes_nothing_and_matches_the_subsequent_real_run`, `test_dry_run_performs_no_lookup_and_writes_nothing_to_parameters`; live dry-run left the corpus at 159 unchanged |
| 7 | 34-02 | Must not re-query the portal for an already-resolved record | Upheld | Early return on `record.parameters.get(site_key)`; `test_second_sweep_issues_no_portal_call_and_reports_unchanged` |
| 8 | 34-02 | Retiring the old command must not silently drop a behaviour | Upheld | 38-row classification table in 34-02-SUMMARY; spot-checked 12 "Retired" rows all carry a stated reason |
| 9 | 34-03 | Series identity must never be written into a `CalendarEvent` field | Upheld | Full read of `observation_series_decoration()` — no write call of any kind |
| 10 | 34-03 | New markers must not remove or weaken an existing status ring | Upheld | All four legacy verbose prefixes retained byte-identical in `_TERMINAL_PREFIXES` |
| 11 | 34-03 | Modal series block must not expose contact/submitter/provenance | Upheld | Return dict has exactly `group_name`, `group_pk`, `index`, `size`, `group_list_url`, `record_url` |
| 12 | 34-04 | Notebook must not embed credentials, raw portal bodies, or contact details | Upheld | Credential-pattern scan of the new notebook → clean; the `password`/`@example.org` hits in the two pre-existing notebooks were **not** introduced by this phase's diff |
| 13 | 34-04 | The SCHED-06 section must not run the sweep between baseline and re-check | Upheld — **advisory** | The instruction is present and explicit in both the notebook and `34-UAT.md`. Advisory: the re-execution itself re-runs two real sweeps *before* re-capturing the baseline, so the re-captured snapshot alone cannot attribute the narrowing. Use the re-execution's own first-sweep summary line as the discriminator (folded into human verification item 1). |
| 14 | 34-04 | The retired command's notebook must not be left in the tree | Upheld | Deleted in `a87f5f8` |
| 15 | 34-04 | The Gemini caveat must not read as a defect report | Upheld | Both the notebook cell and runbook L351-361 state an operating limitation and explicitly say "not a defect in this command" |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/templates/tom_calendar/partials/event_form.html` | 109 | Django `{% comment %}` block still names `sync_lco_observation_calendar` as a live pipeline | ℹ️ Info | Non-rendered developer comment; 34-02-SUMMARY claimed it removed "the last textual references to the deleted module name", which is slightly overstated. No functional or operator-facing impact. |
| `docs/design/telescope_runs_calendar.rst` | 262 | "Stage 4 — observation-record sync. `sync_lco_observation_calendar` (management command, or a `post_save` signal…)" | ℹ️ Info | Historical **design proposal** doc, outside the CLAUDE.md paired-docs scope (`docs/runbooks/` only). It describes the option that was ultimately taken; it reads as a record of the original design, not stale operator guidance. |
| `solsys_code/views.py` | 310 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | **Pre-existing** — `git log -L` attributes it to `a8613bc` ("Add first version of ephemeris generation using layup"), long before this phase. Not introduced here; debt-marker gate does not fire. |
| `src/templates/tom_calendar/partials/event_form.html` | 223 | `Save the event to add TODOs` | ℹ️ Info | **Pre-existing** (`197cb64`, Phase 27) and not a debt marker — it is UI copy about the event's TODO list feature. |

**Debt-marker gate:** no `TBD`/`FIXME`/`XXX` marker was introduced by this phase. Every `TBD` hit in the changed files is the `CampaignRun` domain term ("a TBD window" = window not yet resolved), pre-dating this phase. → **No blocker.**

### Human Verification Required

#### 1. SCHED-06 — live narrowing over real observing nights

**Test:** From now on, run **only** `python manage.py updatestatus` over several real observing nights — never `python manage.py project_observation_calendar`. Then re-execute the demo notebook (`jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`) and diff its SCHED-06 section against the committed `project_observation_calendar_demo.sched06-baseline.json` (74 pending records: 56 queued, 18 placed, captured 2026-09-11T04:44:59Z).
**Expected:** At least one `KEY2026B-004` record has moved queued → placed (or placed → observed) and its calendar event span and title followed it, with no sweep run in between. Fill in a dated row in `34-UAT.md` and flip the verdict from PARTIAL.
**Why human:** SCHED-06 is a verification-over-time requirement — it needs the real LCO scheduler to place and observe real requests on real nights. No test can produce that evidence.
**Read this first:** the re-execution runs two real sweeps *before* it re-captures the baseline. The thing that proves the `post_save` receiver (not the sweep) did the narrowing is the re-execution's **own first sweep summary line**: if it reports `created: 0, updated: 0` for the narrowed records, they were already current before the sweep touched them. Note that line in the 34-UAT.md row.

#### 2. Calendar page — legend, rings, title legibility, series block

**Test:** `python manage.py runserver`, open the calendar month view on a month containing `KEY2026B-004` entries (August/September 2026 both have them). Check the legend row, the chips' status rings, the month-cell titles, and open the event modal for a record belonging to an `ObservationGroup`.
**Expected:** The legend lists `[Q] Queued`, `[S] Scheduled`, `[O] Observed`, `[X] Window expired`, `[C] Cancelled`, `[F] Failed`, `[?] Inconsistent record`. `[Q]` chips carry the dark queued ring; `[X]`/`[C]`/`[F]`/`[?]` the red terminal ring; `[S]`/`[O]` none. Each month-cell title reads legibly inside its truncation budget with the marker and telescope token both visible. The modal shows an "Observation series" block (group name, "Night n of N", working links) rendered beside — not overwriting — the campaign block.
**Why human:** Visual appearance, ring contrast against real chip colours, and month-cell legibility are judgment calls. 129 template/tag tests prove the markup is produced; they cannot prove it reads well.

#### 3. Backstop — interleaved saves of the same record

**Test:** Drive two overlapping saves of one LCO `ObservationRecord` (e.g. two `updatestatus` runs overlapping, or two request threads saving the same record).
**Expected:** The surviving `CalendarEvent` matches the record's **final persisted** field state — no event left describing a superseded intermediate state.
**Why human:** Declared `verification: backstop` (non-inferable). The whole test suite is single-threaded; presence and wiring cannot establish a concurrency invariant.

#### 4. Backstop — sweep interrupted partway

**Test:** Interrupt `python manage.py project_observation_calendar` mid-run (Ctrl-C), then re-run it to completion.
**Expected:** Every record processed before the interrupt still carries a correct event; the re-run converges (`created: 0, updated: 0` for them) with no repair step.
**Why human:** Declared `verification: backstop` (non-inferable). No test simulates a partway interrupt. The observed full-run convergence proves idempotency, not interrupt-safety.

### Gaps Summary

**No gaps block the phase goal.** Every falsifiable roadmap success criterion was checked against the codebase and the real developer database rather than against SUMMARY.md prose, and each held:

- The projector, its three receivers and the `apps.ready()` wiring are real, substantive and connected — not a stub. A live query found **146/146 real `KEY2026B-004` records** carrying exactly one facility-URL-keyed calendar event, zero duplicates, with terminal-negative records keeping `[X]`/`[C]`/`[F]`-marked events spanning their full submitted window.
- The sweep is runnable with zero arguments and **converged live** in this verifier's own process: `159 unchanged, 0 created, 0 updated`, writing nothing.
- The retired command, its 38-test module and its notebook are genuinely gone; the takeover updated 156 pre-existing events **in place** in the same URL namespace, with `RUN:`, `GEM:` and blank-url events asserted byte-identical.
- The paired-docs rule is fully met: the sweep's notebook is pre-executed with committed output, registered in both `docs/notebooks.rst` and CLAUDE.md's map, the runbook rewrite covers the projector, sweep, legend, series block, one-time title change and Gemini caveat, and `sphinx-build` passes.
- Quality gates are clean (`ruff`, `ruff-format`, `sphinx-build`) and 296 targeted tests pass with zero skips and no circular fixtures.

**What holds the phase at `human_needed` rather than `passed`:**

1. **SCHED-06 (truth 4)** cannot close without real observing nights. This is by design — the ROADMAP names it a verification-over-time requirement and plan 34-04 correctly delivered the baseline plus a dated re-check protocol instead of a fabricated verdict. Routed to human verification, **not** recorded as a gap.
2. **Two `verification: backstop` truths** (interleaved saves; interrupted sweep) are non-inferable and abstain for lack of held-out evidence. Neither is a code defect; both are honest blind spots the plans themselves flagged.
3. **Visual verification of the new display layer** (legend, rings, month-cell legibility, series block) is a judgment call the 129 template tests cannot settle.

**One traceability warning (not a blocker):** `PROJ-04` is mapped to Phase 34 in `REQUIREMENTS.md` (line 106, "shared title stem") but appears in **no plan's `requirements:` frontmatter`**. Its substance did ship — `write_event_meta()` populates the Phase 33 carrier fields, nothing group-derived reaches the title or description, and the display-time series decoration renders the identity — so the requirement is satisfied; only the declaration is missing. Worth adding PROJ-04 to the phase's traceability record so a milestone audit does not read it as unclaimed.

**Two stale textual references** to the deleted module survive — a non-rendered Django comment in `event_form.html:109` and a line in the historical design doc `docs/design/telescope_runs_calendar.rst:262`. Neither is operator-facing and neither is in the paired-docs scope, but they make 34-02-SUMMARY's "removing the last textual references to the deleted module name" claim slightly stronger than the tree supports.

---

_Verified: 2026-09-11T05:33:49Z_
_Verifier: Claude (gsd-verifier)_
