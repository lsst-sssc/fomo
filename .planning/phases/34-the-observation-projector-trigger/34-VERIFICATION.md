---
phase: 34-the-observation-projector-trigger
verified: 2026-09-14T22:39:20Z
status: human_needed
score: 14/14 must-haves verified
covered_files:

  - ".planning/REQUIREMENTS.md"
  - ".planning/debug/resolved/34-updatestatus-receiver-attributeerror.md"
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
  - ".planning/phases/34-the-observation-projector-trigger/34-07-PLAN.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-07-SUMMARY.md"
  - ".planning/phases/34-the-observation-projector-trigger/34-UAT.md"
  - "CLAUDE.md"
  - "docs/notebooks.rst"
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb"
  - "docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json"
  - "docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb"
  - "docs/runbooks/telescope_runs_calendar.rst"
  - "solsys_code/allocation_projector.py"
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
  - "solsys_code/tests/test_projector_demo_notebook.py"
  - "solsys_code/views.py"
  - "src/templates/tom_calendar/partials/calendar.html"
  - "src/templates/tom_calendar/partials/event_form.html"

covered_digest: "v1:sha256:1040102aedb77119b1c17909cbe4c52e78575d9be16030535ea3fe928880dc83"
behavior_unverified: 0
overrides_applied: 0
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
re_verification:
  previous_status: passed
  previous_score: 14/14
  trigger: "content-fingerprint staleness -- Phase 35 modified files this phase's prior VERIFICATION.md declared covered (solsys_code/observation_projector.py x4, solsys_code/apps.py, three phase-34 test modules, docs/runbooks/telescope_runs_calendar.rst, CLAUDE.md, .planning/REQUIREMENTS.md)"
  gaps_closed:
    - "Truth 4 (SCHED-06, was PRESENT_BEHAVIOR_UNVERIFIED): closed by directly observed live behaviour on 2026-09-12T22:10Z, independently corroborated from src/fomo_db.sqlite3 this pass -- records 4378332 and 4378046 each narrowed [Q] -> [O] with CalendarEvent.modified == ObservationRecord.modified to the second, through the post_save receiver alone, with no sweep ever having been run against that database."
    - "Truth 13 (interleaved saves, was insufficient_spec/abstained): closed by 34-UAT.md Test 1 (result: pass) -- two overlapping updatestatus runs against a scratch copy, 0 AttributeError, 0 unprojectable, 0 OperationalError."
  gaps_remaining: []
  regressions: []
gaps: []
deferred:

  - truth: "The two title-prefix vocabularies (the legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` prefixes and the projector's terse `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers) are unified into one vocabulary."
    addressed_in: "Phase 37"
    evidence: "ROADMAP Phase 34 scope note: 'Title prefixes ship provisionally here; Phase 37 owns the final vocabulary.' Phase 37 = 'Status vocabulary, public tallies and provenance blind gaps' (STATUS-01/02). Both vocabularies paint the correct ring today (calendar_display_extras._TERMINAL_PREFIXES carries all of them)."
advisory:

  - finding: "The post_save receiver no longer early-returns for a non-LCO/SOAR facility. 34-01's must_have wording ('The receiver returns immediately ... for any facility other than LCO or SOAR') is literally superseded by Phase 35's D-11/WR-01 linked-run re-project loop, which is deliberately facility-independent."
    category: architectural
    reason: "The behavioural intent behind the Phase 34 clause -- 'Gemini records stay with the submission-echo command', i.e. no facility-url CalendarEvent is ever written for a non-LCO/SOAR record -- is intact and still pinned (observation_projector.PROJECTED_FACILITIES still guards the base projection; test_gemini_record_added_to_group_writes_no_calendar_event and test_raw_save_writes_no_calendar_event both pass). What changed is that a Gemini record LINKED to a CampaignRun now re-projects that run's ALLOC: nights, which is Phase 35's own requirement ALLOC-03 and its own namespace's own writer. No override was recorded; flagged for the record rather than treated as a deviation."
    evidence_status: "test_gemini_record_save_still_reprojects_its_linked_run (Phase 35) passes; test_record_with_no_campaign_run_links_never_calls_project_allocation pins that an unlinked record never reaches the allocation writer"
  - finding: "34-01's PROJ-05 prohibition ('the projector never ... writes the reconciler's RUN:/allocation namespace') is now satisfied at module level but not at receiver level: observation_projector.receiver_on_record_save() delegates to allocation_projector.reproject_allocation_if_dispatched(), which does write ALLOC: events."
    category: architectural
    reason: "observation_projector itself still writes only facility-url events and only the three CalendarEventMeta fields (write_event_meta() verified unchanged). The ALLOC: writes are performed by that namespace's own owner through a documented, per-link try/except entry point, and Phase 35's own attribution bridge preserves the human guard the Phase 34 prohibition was protecting (adopt_event_into_run() refuses an event already attributed to a different run; the unlink half filters confirmed_by__isnull=True). Intent preserved; wording superseded by a downstream requirement."
    evidence_status: "code read of _sync_observation_attribution(); test_existing_campaign_attribution_survives_projection_and_is_verified_becomes_true passes"
human_verification:

  - test: "Reconcile `.planning/debug/resolved/34-updatestatus-receiver-attributeerror.md` (committed 2026-09-14 as cff5984) with the developer database. The doc's Resolution states: 'an overnight `updatestatus`-only run against the real DB narrowed all 33 previously-stale LCO events through the receiver alone'. Re-run the two read-only checks below against `src/fomo_db.sqlite3` and decide whether to correct the doc's wording or to produce the missing run evidence."
    expected: "The database says otherwise on both halves of the claim. (a) No `updatestatus` run occurred after 2026-09-12: `sqlite3 'file:src/fomo_db.sqlite3?mode=ro' \"SELECT MAX(modified) FROM tom_calendar_calendarevent;\"` returns `2026-09-12 22:10:48.773592`, and `stat -c '%Y' src/fomo_db.sqlite3` is 1789251048 = 2026-09-12T22:10:48Z. (b) Not all 33 narrowed: 14 are still stale -- COMPLETED records whose events still read `[Q]`/`[S]` with `CalendarEvent.modified` = 2026-09-11 04:44 (observation_ids 4378021-4378025, 4378038-4378045, 4378323, 4378331). A dry-run sweep against a COPY reports `LCO: created: 0, updated: 14, unchanged: 145, unprojectable: 0`. This is exactly finding F-34-1, which 34-UAT.md already recorded and which the prior VERIFICATION.md already corrected ('the 14 terminal ones never will without one sweep') -- the newer debug doc contradicts both. SCHED-06 itself is NOT in doubt: its real evidence is the 2026-09-12T22:10Z run, independently corroborated in this report."
    why_human: "This is a decision, not a measurement -- the measurement is already done and reported above. Either the doc's wording is corrected to match F-34-1's split, or an actual post-2026-09-12 `updatestatus`-only run is performed and its evidence recorded. A verifier must not edit a human-signed debug resolution."
  - test: "Decide whether to spend the SCHED-06 evidence now and run the backstop sweep once against the developer database: `python manage.py project_observation_calendar` (preceded by `--dry-run`). SCHED-06's live evidence has been collected and corroborated, so the 33 stale events no longer need preserving."
    expected: "The sweep reports `LCO: created: 0, updated: 14, unchanged: 145, unprojectable: 0` and the 14 legacy `[Q]`/`[S]`-on-COMPLETED events become `[O]` over their observed blocks. A second run reports `updated: 0`. This is F-34-1's designed remedy -- `tom_observations.facility.update_all_observation_statuses()` excludes terminal states (facility.py:573), so no `updatestatus` run will ever repair these 14; only the sweep this phase shipped can."
    why_human: "Spending the SCHED-06 baseline evidence is an operator decision with a one-way effect on the developer database, and CLAUDE.md's workflow rule keeps write commands out of a verifier's hands. Not a code gap: the sweep is the shipped, documented backstop for exactly this residue (TRIG-03)."
---

# Phase 34: The Observation Projector & Trigger — Verification Report (re-verification)

**Phase Goal:** Every LCO/SOAR observation record draws its own calendar event and keeps it current on every save with no operator command, and the old LCO sync command is retired in its favour — one writer for observation-backed nights.
**Verified:** 2026-09-14T22:39:20Z (HEAD `31ec92c`, branch `issue37-telescope-runs-calendar`)
**Status:** human_needed (14/14 truths verified; 0 gaps; 2 items need a human decision)
**Re-verification:** Yes — triggered by content-fingerprint staleness after Phase 35 modified shared files. Supersedes the 2026-09-12T00:20:00Z report (`passed`, 14/14).

## What this pass was asked to establish

Phase 35 changed `solsys_code/observation_projector.py` four times (NF-04, WR-01/WR-02, CR-01, feat 35-04), `solsys_code/apps.py`, three of this phase's test modules, the runbook, `CLAUDE.md` and `.planning/REQUIREMENTS.md`. The question is whether Phase 34's must-haves still hold against the **current** tree.

**Answer: they do. No regression was found.** Every Phase 34 truth was re-checked against today's code, today's test run, and today's developer database — not carried forward on the prior report's word. Two items nevertheless need a human, and one of them is new: a Phase 34 artifact committed *after* the prior verification makes a claim the developer database contradicts.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **(SC1)** Every LCO/SOAR `ObservationRecord` has exactly one calendar event keyed by its facility observation URL, spanning the request window while queued / the placed block once scheduled / the observed block once observed; a terminal-negative record keeps a visibly marked event on its window night. | ✓ VERIFIED | Re-proved **live against `src/fomo_db.sqlite3` this pass** (read-only SQL, no Django write path): `lco_soar_records = 159`, `facility_url_events = 159`, `events_no_record = 0`, and **zero duplicate non-blank urls** (`GROUP BY url HAVING COUNT(*)>1` returns only the blank-url family). Per-marker tally sums exactly to the record count: `[Q]:42 [S]:22 [O]:62 [X]:26 [C]:6 [F]:1` = 159. Identity is still `event_url()` = `facility.get_observation_url(...)` and nothing else; span is still re-derived from record fields every projection. A full dry-run sweep with **today's** projector over the whole real corpus reports `unprojectable: 0` across all 159. Stage/marker behaviour pinned by `TestEventFieldsFor` and `test_two_records_whose_windows_exactly_abut_produce_two_separate_events` in the 342-test run. |
| 2 | **(SC2)** Saving a record updates its event with no operator command — including the schedule-only placement save TOM's own hook misses and the `updatestatus` path — while a save that changes nothing writes nothing and a projector error is logged rather than aborting the record save. | ✓ VERIFIED | `apps.py:30-46` still connects `post_save`/`m2m_changed`/`pre_delete`, each `weak=False` with a distinct `dispatch_uid` (Phase 35 **added** two `CampaignRunObservation` receivers below them; it removed none). `coerce_schedule_datetime()` still routed through by both `record_time_window()` branches. Behavioural, re-run this pass: `test_updatestatus_narrows_the_event_with_no_command_run` (drives the real `LCOFacility().update_observation_status()` with portal ISO strings inside `assertNoLogs(WARNING)`), `test_schedule_only_save_narrows_the_same_event_row`, `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`, `test_raising_projector_does_not_block_a_save`, `test_record_saved_inside_a_rolled_back_transaction_leaves_no_event`, `test_make_request_is_never_called_during_a_record_save` — all pass. Live no-churn corroboration: the 2026-09-12 `updatestatus` run saved **52** LCO/SOAR records and wrote only **14** events. |
| 3 | **(SC3)** One sweep command re-projects any set of records (`--dry-run` supported, per-record failures isolated) as the backstop for bulk-write paths; a second sweep reports everything unchanged and no event outside the projector's key namespace is created, modified or deleted. | ✓ VERIFIED | Re-run: `project_observation_calendar --help` → `[--proposal PROPOSAL] [--facility {LCO,SOAR}] [--dry-run]`, zero required args. Executed `--dry-run` against a scratch **copy** of the real database this pass: `Done (dry run). failed: 0 \| LCO: created: 0, updated: 14, unchanged: 145, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 \| SOAR: all zero`. Namespace isolation re-proved live: the developer database still holds `RUN:` 72 and blank-url 10 events — the same counts the Phase 34 notebook recorded — after a receiver-driven narrowing pass touched 14 facility-url events. `TestDryRun`, `TestFailureIsolation`, `TestNamespaceIsolation`, `TestProjectQuerysetOrdering`, `TestObservedSiteLookup` all pass. |
| 4 | **(SC4a)** Over real nights, a user watches a `KEY2026B-004` record's event narrow queued → scheduled → observed with nobody running anything (closes spike 004's PARTIAL verdict). | ✓ VERIFIED (was ⚠️ PRESENT_BEHAVIOR_UNVERIFIED) | **Independently corroborated from the database, not taken from the UAT write-up.** After one real `updatestatus` at 2026-09-12T22:10Z — and no `project_observation_calendar` sweep has *ever* been run against `src/fomo_db.sqlite3` — record `4378332` (11P) is `COMPLETED, modified 2026-09-12 22:10:48` and its event is `[O] 1m0 11P` spanning `2026-09-12 01:06:46 → 01:26:04` with `CalendarEvent.modified = 2026-09-12 22:10:48` — **equal to the record's own `modified` to the microsecond-truncated second**, which is only possible if the `post_save` receiver wrote it inline. `4378046` (10P) shows the identical pattern at `22:10:21`. 34-UAT.md Test 3 `result: pass`. **Temporal caveat, stated plainly:** this live run predates Phase 35's first receiver change (`aad61e4`, 2026-09-13T05:56Z) by ~8 hours; the post-change receiver is re-proved at integration level by `test_updatestatus_narrows_the_event_with_no_command_run` and at corpus level by this pass's `unprojectable: 0` dry run over all 159 real records. |
| 5 | **(SC4b)** Every event title is short enough to read in a month cell (PROJ-06). | ✓ VERIFIED | `title_for()` still produces `'[marker] <token> <target>'` capped at 200 chars; `test_marker_and_token_within_first_16_characters` passes. Real live titles queried this pass: `'[O] 1m0 11P'`, `'[S] 1m0 220P'`, `'[Q] 1m0 10P'`. |
| 6 | **(SC5)** `sync_lco_observation_calendar` no longer exists; the same events come from the projector and sweep in the same key namespace, with runbook section, demo notebook and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-read-back caveat documented. | ✓ VERIFIED | Re-checked: `python manage.py help \| grep -c sync_lco_observation_calendar` → **0**. `management/commands/` listing contains `project_observation_calendar.py` and `sync_gemini_observation_calendar.py`, no retired module. `pre_executed/` contains `project_observation_calendar_demo.ipynb` and `sync_gemini_observation_calendar_demo.ipynb`, no retired notebook. `docs/runbooks/telescope_runs_calendar.rst` — **modified 6 times by Phase 35** — still carries **0** references to the retired command and still documents the projector, the legend (L152), the Observation series block (L160-164), the one-time title change (L185), the sweep's counters (L223-248) and the `unprojectable`/`site_lookup_failed` troubleshooting rows (L1397-1407). |
| 7 | A calendar visitor can tell what each projector marker means from a legend on the calendar page itself, including `[?]`. | ✓ VERIFIED | `{% observation_status_legend as status_legend %}` still at `calendar.html:331` (template files unchanged since the prior pass); `test_returns_seven_entries_covering_every_marker` and `test_calendar_page_renders_every_legend_marker_and_label` pass. |
| 8 | Every projector marker paints the right status ring, and no existing ring is lost. | ✓ VERIFIED | `{% status_border_css event.title as status_border %}` at `calendar.html:238` and `:274`; `calendar_display_extras.py` unchanged since the prior pass; `_TERMINAL_PREFIXES` still retains all four legacy verbose prefixes beside the bracket-letter markers. `TestProjectorMarkerRings` + `TestTelescopeStripeContrast` pass. |
| 9 | Series identity ("night n of N") is rendered at request time from `CalendarEventMeta.observation_group`, never stored in the event, ordered by window start, with no per-event query fan-out. | ✓ VERIFIED | `{% observation_series_decoration event as series %}` at `event_form.html:148`, beside (not replacing) the campaign block; tag body contains no write call. `TestObservationSeriesDecoration` and `test_modal_query_count_does_not_grow_with_group_size` pass. |
| 10 | The paired-docs rule is satisfied: the sweep has a pre-executed demo notebook whose executed output demonstrates the one-time takeover and a first-vs-second sweep that differ; the retired command's notebook is gone; the runbook describes the projector/sweep, legend, series block and Gemini caveat. | ✓ VERIFIED | Notebook and `sched06-baseline.json` both **byte-unchanged** since the prior pass (`git log` on both paths shows no commit since; `git status --porcelain` on `docs/` is clean apart from Phase 35's `reconcile_campaign_runs_demo.ipynb`). Baseline sha256 `453ae2ba…d353859c` still matches `git show a87f5f8:<path>`. The guard `test_projector_demo_notebook.py` passes (5 ok, 1 branch-correct skip) and was **re-mutation-tested this pass** — see spot-checks. |
| 11 | Nothing group-derived is written into `CalendarEvent.title` or `.description`; series identity lives only in `CalendarEventMeta.observation_group` (PROJ-04 title-stem clause). | ✓ VERIFIED | `write_event_meta()` (observation_projector.py:312-334) read this pass: still writes exactly `is_verified`/`observation_record`/`observation_group`, with the docstring's "never the campaign attribution link or its confirmation stamps" intact and **untouched by any Phase 35 commit**. `test_grouped_record_sets_observation_group_and_omits_name_from_title_and_description`, `test_existing_campaign_attribution_survives_projection_and_is_verified_becomes_true`, `test_stale_companion_claim_on_a_different_event_is_cleared_not_integrity_error`, `test_record_in_two_groups_links_the_lowest_pk_group` all pass. |
| 12 | The observed telescope is resolved once, stored on the record, read back as a network-free title token, and the D-07 label rename does not downgrade Phase 28's site-level attribution matching. | ✓ VERIFIED | `campaign_attribution.py` and `calendar_utils.py` both **unchanged** since the prior pass (0 commits). `test_campaign_attribution.py` passes. Live no-re-query proof with today's code: the full dry-run sweep over 159 records reported `site_lookups: 0` — every terminal record's site is already stored, so D-08's once-only contract holds on the real corpus. |
| 13 | If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state. | ✓ VERIFIED (was ⚠️ insufficient_spec, abstained) | Declared `verification: backstop`. Closed by directly observed operator behaviour: 34-UAT.md Test 1 `result: pass` — two overlapping `updatestatus` runs against a scratch copy, **0 `AttributeError`, 0 `unprojectable`, 0 `OperationalError`**, both runs `Update completed successfully`, every record either run saved projecting correctly. **Temporal caveat:** that run predates Phase 35's receiver change. The change is provably isolated from the write path this truth concerns — the linked-run loop runs strictly *after* the base projection and each link carries its own `try`/`except`; `test_linked_run_reproject_raising_does_not_abort_the_records_own_save_or_projection` asserts the record's own event survives a linked-run failure. |
| 14 | A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step. | ✓ VERIFIED | Directly observed operator behaviour in 34-UAT.md Test 3 (`result: pass`) against a copy: interrupted run repaired 1 record, re-run `created: 0, updated: 32, unchanged: 127, unprojectable: 0`, third dry run `updated: 0, unchanged: 159`. Corroborated by the notebook's two-sweep convergence (33 → 0) and, this pass, by the real corpus already sitting at `unchanged: 145` with no repair step ever having been run against it. |

**Score:** 14/14 truths verified (0 present-but-behavior-unverified, 0 abstained, 0 failed)

### Compatibility Review — did Phase 35 regress anything?

This is the question that triggered the re-verification. Each change was read, not assumed.

| Phase 35 change | File | Phase 34 property at risk | Verdict |
|---|---|---|---|
| `feat(35-04)` `aad61e4` + `WR-01/WR-02` `9909e82` + `NF-04` `30112a0` — append a D-11 linked-run re-project loop to `receiver_on_record_save()` | `observation_projector.py` | TRIG-01/TRIG-02 (receiver never raises, no network call), PROJ-05 (namespace isolation) | ✓ **Compatible.** Base projection still runs first and is still guarded by `PROJECTED_FACILITIES`; its `try` is unchanged. The new loop has a **per-link** `try`/`except` (never a bare loop-wide one), logs `run pk=…`, and re-raises nothing. `test_linked_run_reproject_makes_no_network_call`, `test_linked_run_reproject_raising_does_not_abort_the_records_own_save_or_projection`, `test_one_failing_linked_run_does_not_skip_a_later_one`, `test_record_with_no_campaign_run_links_never_calls_project_allocation` all pass. |
| `CR-01` `2a34a04` — route allocation triggers through a guarded dispatch entry point | `observation_projector.py` / `allocation_projector.py` | PROJ-05 no-foreign-writes | ✓ Compatible. `observation_projector` still writes only facility-url events; ALLOC: writes go through that namespace's own owner. See advisory 2. |
| Two new receivers on `CampaignRunObservation` | `apps.py` | TRIG-01 wiring | ✓ **Additive only.** All three Phase 34 connections survive verbatim at `apps.py` with their original `dispatch_uid`s; the docstring explicitly records them as "unchanged since Phase 34". |
| `RUN:` → `ALLOC:` in namespace-isolation fixtures | 3 phase-34 test modules | PROJ-05 test strength | ✓ **No weakening.** Diff is 3 url-literal renames plus explanatory comments (`git diff` on the two projector-test modules is 4 changed lines total; the signals module's 250 insertions are all *new* tests). **No Phase 34 test was deleted or had an assertion relaxed** — every one of the 18 named tests the prior report cited still exists and passes. The rename tracks the live key form; and the retired `RUN:` form is still guarded live — 72 `RUN:` events sit untouched on the developer database. |
| 6 runbook commits, `CLAUDE.md`, `REQUIREMENTS.md` | docs | ANNOT-03 | ✓ Compatible. Runbook still has 0 references to the retired command and retains all six Phase 34 sections. `REQUIREMENTS.md` diff since the prior pass touches only ALLOC-01..05 checkboxes; every PROJ/TRIG/SCHED/ANNOT line is byte-identical. |

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Unification of the two title-prefix vocabularies (legacy verbose vs. terse bracket-letter markers) | Phase 37 | ROADMAP Phase 34 scope note: "Title prefixes ship provisionally here; Phase 37 owns the final vocabulary." Both vocabularies already paint the correct ring. |

### Advisory (New Scope, Unevidenced-as-Blocker)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | The receiver no longer early-returns for a non-LCO/SOAR facility — 34-01's literal must_have wording is superseded by Phase 35's deliberately facility-independent D-11 loop | architectural | The behavioural intent (no facility-url event for a Gemini record) is intact and pinned by passing tests; the changed behaviour is a downstream requirement (ALLOC-03), not a defect. No must-have FAILED. |
| 2 | 34-01's PROJ-05 prohibition holds at module level but the receiver now delegates ALLOC: writes to `allocation_projector` | architectural | `observation_projector` still writes only facility-url events and only the three meta fields. The human guard the prohibition protected survives (`adopt_event_into_run()` refuses a foreign-attributed event; the unlink half filters `confirmed_by__isnull=True`). Intent preserved. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/observation_projector.py` | Projector + three signal receivers | ✓ VERIFIED | 770 lines. `project_record()`, `event_fields_for()`, `write_event_meta()`, `event_url()`, `project_queryset()` all unchanged by Phase 35 — the entire 97-line diff is confined to `receiver_on_record_save()`. Module docstring still states the three ownership rules. Imported by `apps.py`, the sweep command, `allocation_projector.py` and 4 test modules. |
| `solsys_code/apps.py` | `ready()` wires the three receivers | ✓ VERIFIED | The three Phase 34 connections intact with original `dispatch_uid`s; two Phase 35 receivers appended. |
| `solsys_code/calendar_utils.py` | `coerce_schedule_datetime()` used by both `record_time_window()` branches | ✓ VERIFIED | Unchanged since the prior pass (0 commits). `TestCoerceScheduleDatetime` + `TestRecordTimeWindow` pass. |
| `solsys_code/management/commands/project_observation_calendar.py` | Backstop sweep, zero required args | ✓ VERIFIED | Unchanged. `--help` clean; executed `--dry-run` against a real-corpus copy this pass. |
| `solsys_code/templatetags/calendar_display_extras.py` | Rings, legend, series decoration | ✓ VERIFIED | Unchanged; all three tags still called from the two (also unchanged) templates. |
| `solsys_code/campaign_attribution.py` | `OBSERVED_TELESCOPE_OBSCODES` bridge ahead of `_extract_lco_site_code()` | ✓ VERIFIED | Unchanged; its test module passes. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | Paired demo with a real takeover | ✓ VERIFIED | Byte-unchanged; guard passes; guard re-mutation-tested. |
| `…/project_observation_calendar_demo.sched06-baseline.json` | Byte-identical to committed state | ✓ VERIFIED | sha256 `453ae2ba…d353859c` matches `git show a87f5f8:<path>`; `git log` on the path still shows only `a87f5f8`. |
| `solsys_code/tests/test_projector_demo_notebook.py` | Repo-level guard over the committed notebook | ✓ VERIFIED | Unchanged; 5 pass + 1 branch-correct skip; mutation-proven again this pass. |
| `docs/runbooks/telescope_runs_calendar.rst` | Projector/sweep section, no stale sync content | ✓ VERIFIED | 6 Phase 35 commits; 0 references to the retired command; all six Phase 34 sections still present. |
| `src/fomo_db.sqlite3` | SCHED-06 evidence database | ⚠️ NOTED (not an artifact defect) | mtime moved 2026-09-11T20:07Z → **2026-09-12T22:10Z** — the UAT Test 3 `updatestatus` run that *produced* the SCHED-06 evidence. Size unchanged at 1232896. No sweep has ever run against it (`RUN:` 72 / blank-url 10 intact, 14 pre-fix events still stale). Untouched by this verification: the dry run was executed against a scratch copy and the stamp is identical before and after. |
| `solsys_code/tests/*` (8 phase modules) | Behaviour pinned | ✓ VERIFIED | `Ran 342 tests in 16.116s … OK (skipped=1)` — up from 292, with no Phase 34 test removed. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ObservationRecord.save()` | `CalendarEvent` row | `post_save` → `receiver_on_record_save()` → `project_record()` | ✓ WIRED | Re-proved by `test_updatestatus_narrows_the_event_with_no_command_run` **and** by the live database: 14 events whose `modified` equals their own record's `modified` to the second. |
| `SolsysCodeConfig.ready()` | the three receivers | `.connect(weak=False, dispatch_uid=…)` | ✓ WIRED | `apps.py` — all three survive Phase 35's additions. |
| sweep per-record loop | `project_record()` / `preview_calendar_event_action()` | `project_queryset()` | ✓ WIRED | One projection path shared by receiver and sweep; the dry run's `unchanged: 145` against receiver-written events is the proof the two agree. |
| `receiver_on_record_save()` | `allocation_projector.reproject_allocation_if_dispatched()` | `instance.campaign_run_links.select_related('run')`, per-link `try` | ✓ WIRED (new, Phase 35) | Additive. Never reached for an unlinked record (`test_record_with_no_campaign_run_links_never_calls_project_allocation`). The developer database holds exactly 1 `CampaignRunObservation` row and 0 `ALLOC:` events, so the live `updatestatus` path is effectively unchanged today. |
| `CalendarEventMeta.observation_group` | `event_form.html` | `observation_series_decoration()` | ✓ WIRED | Display-time only. |
| `calendar_utils` telescope labels | Phase 28 attribution matching | `OBSERVED_TELESCOPE_OBSCODES` | ✓ WIRED | Carried forward; both files unchanged. |
| committed notebook evidence | every `manage.py test` run | `test_projector_demo_notebook.py` | ✓ WIRED | Guard runs in the default suite and still fails on emptied evidence. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `event_fields_for()` | `start_time`/`end_time` | `record_time_window(record)` / `record.parameters` — never a stored event field | Yes — live: 4378332's event spans its own observed block | ✓ FLOWING |
| `telescope_token()` | `token` | `record.parameters['observed_site'/'observed_telescope']`, else `coarse_telescope_label()` | Yes — `site_lookups: 0` on a 159-record sweep proves the stored values are being read back | ✓ FLOWING |
| `project_queryset()` counters | `action` | `preview_calendar_event_action(before, fields)` vs a pre-sweep snapshot | Yes — `updated: 14 / unchanged: 145` discriminates, not a blanket count | ✓ FLOWING |
| `receiver_on_record_save()` linked-run loop | `link.run` | `instance.campaign_run_links` (real FK traversal) | Yes (tested); dormant on the live corpus (1 link, 0 ALLOC: events) | ✓ FLOWING |
| notebook cells `05528b38` / `556d2a9f` | `changed_titles`, sweep summaries | Live before/after snapshots around a real sweep on a fresh clone | Yes — 33 of 159; 33 then 0 | ✓ FLOWING |
| `observation_status_legend()` | legend entries | Module constant | Intentionally fixed (documented) | ✓ FLOWING (by design) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-34 test modules pass (8 modules, one run) | `python manage.py test solsys_code.tests.test_observation_projector …test_campaign_attribution` | `Ran 342 tests in 16.116s` / `OK (skipped=1)` | ✓ PASS |
| Notebook-evidence guard passes against the committed artifact | `python manage.py test solsys_code.tests.test_projector_demo_notebook -v2` | 6 tests: 5 ok, 1 skip (`not routed to a scratch copy` branch) | ✓ PASS |
| **Guard still fails against the pre-34-07 regressed notebook** (mutation, re-run this pass) | `git show 46d8390:…demo.ipynb` via `FOMO_DEMO_NOTEBOOK_PATH` | `FAILED (failures=1)` — `Cell 05528b38 does not report a non-zero re-titled count: '…0 of 159 pre-existing facility-url-keyed events were re-titled…'` | ✓ PASS |
| Sweep registered with its flags | `python manage.py project_observation_calendar --help` | `[--proposal PROPOSAL] [--facility {LCO,SOAR}] [--dry-run]` | ✓ PASS |
| **Full-corpus dry run with today's projector** | `FOMO_DATABASE_PATH=<scratch copy> python manage.py project_observation_calendar --dry-run` | `failed: 0 \| LCO: created: 0, updated: 14, unchanged: 145, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 \| SOAR: all zero` | ✓ PASS |
| Retired command really gone | `python manage.py help \| grep -c sync_lco_observation_calendar` | `0` | ✓ PASS |
| One event per record, no duplicates (live) | read-only SQL over `src/fomo_db.sqlite3` | `159 records / 159 facility-url events / 0 orphans / 0 duplicate non-blank urls` | ✓ PASS |
| Foreign namespaces intact (live) | read-only SQL | `RUN: 72`, blank-url `10` — unchanged from the Phase 34 notebook figures | ✓ PASS |
| SCHED-06 receiver-only narrowing (live) | read-only SQL on 4378332 / 4378046 | `[O]` titles over observed blocks, `CalendarEvent.modified == ObservationRecord.modified` | ✓ PASS |
| Baseline JSON byte-identical to `a87f5f8` | `sha256sum` vs `git show` | both `453ae2ba…d353859c` | ✓ PASS |
| Developer DB untouched by this verification | `stat -c '%Y %s'` before/after every command | `1789251048 1232896` both times | ✓ PASS |
| Debt markers in phase-34 files changed since prior pass | `grep -nE "TBD\|FIXME\|XXX\|HACK\|PLACEHOLDER"` over 7 files | 1 hit, non-marker (see anti-patterns) | ✓ PASS |

**Note on DB safety:** no `updatestatus` and no real sweep was run by this verification. All database inspection was read-only (`file:…?mode=ro`); the one sweep was `--dry-run` against a scratch copy under `FOMO_DATABASE_PATH`. The developer database's size and mtime are identical before and after this pass.

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN/SUMMARY declares a probe | N/A — skipped |

### Decision Coverage

`gsd_run query check.decision-coverage-verify` → `{ skipped: false, blocking: false, total: 21, honored: 21, not_honored: [] }` — "All trackable CONTEXT.md decisions are honored by shipped artifacts." **Non-blocking; no status impact.**

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROJ-01 | 34-01, 34-02 | Exactly one `CalendarEvent` per LCO/SOAR record, keyed by `facility.get_observation_url()` | ✓ SATISFIED | Truth 1 — live: 159/159, 0 duplicate non-blank urls |
| PROJ-02 | 34-01, 34-05, 34-06 | Span follows the stage: request window → placed block → observed block | ✓ SATISFIED | Truths 1-2, 4; `TestEventFieldsFor`; `coerce_schedule_datetime()` intact |
| PROJ-03 | 34-01, 34-03 | Terminal-negative record keeps a visibly marked event | ✓ SATISFIED | Truths 1, 7, 8; live `[X]:26 [C]:6 [F]:1` |
| PROJ-04 (title-stem clause) | **not declared in any plan's `requirements:`** | Series identity carried by real FKs, not text in the title | ✓ SATISFIED but ⚠️ **ORPHANED** | REQUIREMENTS.md L18/L106 maps the title-stem clause to Phase 34, yet no plan frontmatter claims `PROJ-04`, and the ROADMAP's own phase requirement list omits it. Delivered anyway (truths 9, 11). Traceability gap in the plans, not a delivery gap — carried forward unchanged. |
| PROJ-05 | 34-01, 34-02, 34-03, 34-07 | No-churn; never touches an event it does not own | ✓ SATISFIED | `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`; `TestNamespaceIsolation` (both modules); live `RUN: 72` / blank-url `10` untouched across a receiver-driven narrowing pass. See advisory 2 for the receiver-level nuance. |
| PROJ-06 | 34-01, 34-03 | Compact titles that fit a month cell | ✓ SATISFIED | Truth 5 |
| SCHED-06 | 34-04, 34-05, 34-06, 34-07 | A user watches a record narrow over real nights with no command | ✓ SATISFIED (was ? NEEDS HUMAN) | Truth 4 — live evidence independently corroborated from the database. **The claim in the 2026-09-14 debug doc that "all 33" narrowed is separately contradicted and routed to human item 1; SCHED-06 itself does not depend on it.** |
| TRIG-01 | 34-01, 34-05, 34-06 | `post_save` receiver in `apps.ready()`, covering schedule-only and `updatestatus` paths | ✓ SATISFIED | Truth 2; `apps.py`; `TestUpdateObservationStatusPath` |
| TRIG-02 | 34-01, 34-05, 34-06 | Single-record, idempotent, cheap, error-logged-never-aborts | ✓ SATISFIED | Truth 2; `test_make_request_is_never_called_during_a_record_save`, `test_raising_projector_does_not_block_a_save`; Phase 35's added loop carries its own per-link `try` |
| TRIG-03 | 34-02, 34-04, 34-07 | Sweep command with `--dry-run`, failure isolation, and a paired pre-executed demo notebook | ✓ SATISFIED | Truth 3; executed dry run this pass; notebook + guard intact |
| ANNOT-03 | 34-02, 34-04, 34-07 | Old LCO sync retired; runbook/notebook/tests migrated; Gemini caveat documented | ✓ SATISFIED | Truth 6 — re-checked against the 6-times-modified runbook |

**Orphaned requirements:** `PROJ-04` (title-stem clause) is mapped to Phase 34 in REQUIREMENTS.md but appears in no plan's `requirements:` field and not in the ROADMAP's phase requirement list. Delivered; flagged for traceability only.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `test_observation_projector.py` | PROJ-01/02/03/05/06, PROJ-04 stem | 49 | 0 | No | Value + behavioral | ✓ Sufficient |
| `test_observation_projector_signals.py` | TRIG-01/02, PROJ-02 (+ Phase 35 D-11) | 31 | 0 | No | Behavioral (drives the real `update_observation_status()`) | ✓ Sufficient — grew by 8, none removed |
| `test_project_observation_calendar.py` | TRIG-03, PROJ-05 | 25 | 0 | No | Value + behavioral | ✓ Sufficient |
| `test_calendar_utils.py` | PROJ-02 (G-34-2) | `TestCoerceScheduleDatetime` + `TestRecordTimeWindow` | 0 | No | Value | ✓ Sufficient |
| `test_calendar_display_extras.py` / `test_calendar_template.py` | PROJ-03/05/06, PROJ-04 stem | many | 0 | No | Value + rendered-markup | ✓ Sufficient |
| `test_projector_demo_notebook.py` | PROJ-05, TRIG-03, SCHED-06, ANNOT-03 | 5 | 1 (branch-correct) | No | Artifact-content, mutation-proven | ✓ Sufficient |

**Disabled tests on requirements:** 0. **Circular patterns:** 0. **Insufficient assertions:** 0 — the notebook guard was mutation-verified again this pass rather than trusted. **Assertions weakened by Phase 35:** 0 — the only edits to Phase 34 test modules are three url-literal renames (`RUN:` → `ALLOC:`) with explanatory comments; no test was deleted and no assertion relaxed.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/views.py` | 310 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | **Pre-existing** (`git blame` → `a8613bc8`, 2025-07-23). `views.py` has **0 commits** since the prior verification, so it is outside the regression window. Debt-marker gate does not fire — classified identically by all three passes. |
| `docs/runbooks/telescope_runs_calendar.rst` | 1099 | `` ``TBD window`` `` | ℹ️ Info | **Not a debt marker.** It is a documented skip-reason *value* the reconciler emits for a `CampaignRun` with no parsed `Obs. Date`, listed beside `not approved` / `unresolved site` / `window_end before window_start`. Introduced by Phase 35 (in the regression window) but carries no unfinished-work semantics. |
| `src/fomo_db.sqlite3` (data, not code) | — | 14 stale facility-url events: COMPLETED records still titled `[Q]`/`[S]`, `CalendarEvent.modified` = 2026-09-11 04:44 | ℹ️ Info (carried forward as finding **F-34-1**) | Pre-fix legacy residue, **unchanged since before Phase 35** and already adjudicated: `update_all_observation_statuses()` excludes terminal states (`facility.py:573`), so no `updatestatus` will ever re-save them; only the sweep this phase shipped can. Quantified exactly this pass: `LCO: updated: 14`. Not a code defect and not a regression — routed to human item 2 as an operator decision. |
| `.planning/debug/resolved/34-updatestatus-receiver-attributeerror.md` | Resolution / `signal_real_path` | Claims "an overnight `updatestatus`-only run against the real DB narrowed **all 33** previously-stale LCO events … with no intervening sweep" (committed 2026-09-14, **inside the regression window**) | ⚠️ Warning | **Contradicted by the database, with deterministic evidence.** (a) `MAX(CalendarEvent.modified)` = `2026-09-12 22:10:48.773592` and `stat -c '%Y'` = `1789251048` (= 2026-09-12T22:10:48Z) — no run occurred on 09-13 or 09-14. (b) 14 of the 33 are still stale; a dry-run sweep reports `LCO: updated: 14`. The prior VERIFICATION.md already corrected this exact premise ("the 14 terminal ones never will without one sweep"); the newer doc re-asserts it. **No must-have is FAILED** — SCHED-06's real evidence is the 2026-09-12 run, verified independently above — so this is a WARNING requiring a human decision, not a BLOCKER. Routed to human item 1. |

### Human Verification Required

#### 1. Reconcile the 2026-09-14 debug resolution with the developer database

**Test:** Open `.planning/debug/resolved/34-updatestatus-receiver-attributeerror.md` (committed as `cff5984`). Its Resolution and `signal_real_path` both state: *"an overnight `updatestatus`-only run against the real DB narrowed all 33 previously-stale LCO events through the receiver alone, with no intervening sweep."* Re-run these two read-only checks and decide what to do:

```console
$ stat -c '%Y %s' src/fomo_db.sqlite3
1789251048 1232896                 # = 2026-09-12T22:10:48Z

$ sqlite3 "file:src/fomo_db.sqlite3?mode=ro" "SELECT MAX(modified) FROM tom_calendar_calendarevent;"
2026-09-12 22:10:48.773592         # nothing written on 09-13 or 09-14

$ cp src/fomo_db.sqlite3 /tmp/check.sqlite3
$ FOMO_DATABASE_PATH=/tmp/check.sqlite3 python manage.py project_observation_calendar --dry-run
... LCO: created: 0, updated: 14, unchanged: 145, unprojectable: 0 ...
```

**Expected:** Both halves of the doc's claim fail. No `updatestatus` run happened after 2026-09-12, and 14 of the 33 never narrowed — they are observation_ids `4378021`-`4378025`, `4378038`-`4378045`, `4378323`, `4378331`, all `COMPLETED` with `record.modified` 2026-09-11 20:02-20:04 UTC and `event.modified` 2026-09-11 04:44. This is precisely finding **F-34-1**, which `34-UAT.md` recorded and which the prior VERIFICATION.md already corrected. Either amend the debug doc to match (19-ish repaired / 14 permanently un-repairable by `updatestatus`), or perform the claimed run and record its real evidence.
**Why human:** The measurement is already done and reported here; what remains is a decision about a human-signed debug resolution, which a verifier must not edit. **SCHED-06 is not at risk** — its evidence is the independently corroborated 2026-09-12T22:10Z run (truth 4), not this doc.

#### 2. Decide whether to spend the SCHED-06 evidence and clear F-34-1's 14 legacy events

**Test:** SCHED-06's live evidence has now been collected and corroborated, so the stale events no longer need preserving. Run the backstop sweep once against the developer database:

```console
$ python manage.py project_observation_calendar --dry-run   # expect: LCO updated: 14
$ python manage.py project_observation_calendar             # then re-run --dry-run: expect updated: 0
```

**Expected:** `LCO: created: 0, updated: 14, unchanged: 145, unprojectable: 0`, after which the 14 `[Q]`/`[S]`-on-COMPLETED events become `[O]` over their observed blocks and a second run converges to `updated: 0`.
**Why human:** A write against the developer database is an operator decision with a one-way effect, and CLAUDE.md's workflow rule keeps write commands out of a verifier's hands. **Not a code gap** — this is exactly the backstop role TRIG-03 shipped the sweep for, and the residue predates Phase 35 entirely.

### Gaps Summary

**No gaps. No regressions. 14/14 truths verified.**

Phase 35 touched `observation_projector.py` four times, but every one of those edits is confined to `receiver_on_record_save()`, and everything Phase 34's must-haves actually rest on — `project_record()`, `event_fields_for()`, `write_event_meta()`, `event_url()`, `project_queryset()`, `coerce_schedule_datetime()`, the sweep command, the template tags, the templates, `campaign_attribution.py` — is byte-identical to its state at the previous verification. The three receivers Phase 34 wired in `apps.ready()` survive verbatim with their original `dispatch_uid`s; Phase 35 appended two, removed none. The three edits inside Phase 34's own test modules are url-literal renames (`RUN:` → `ALLOC:`, tracking the live key form) with explanatory comments: no test deleted, no assertion relaxed. All 18 named tests the prior report cited still exist and still pass, inside a suite that grew from 292 to 342.

I did not take the prior report's word for the live properties. The single-writer-per-facility-URL property was re-proved by read-only SQL against the real developer database with today's code in place: 159 LCO/SOAR records against 159 facility-url-keyed events, zero orphans, zero duplicate non-blank urls, and a per-marker tally summing exactly to 159. The foreign namespaces the projector must never touch are still intact at their Phase 34 counts (`RUN:` 72, blank-url 10) even after a receiver-driven narrowing pass wrote 14 events. A full `--dry-run` sweep over all 159 real records — run against a scratch copy, never the developer database — reports `unprojectable: 0` and `site_lookups: 0`, which says today's projector handles every real record without a single failure and without re-querying a site it already resolved.

SCHED-06 and the interleaved-save backstop, the prior pass's two open human items, are both closed. The SCHED-06 evidence I verified myself rather than reading it out of the UAT: records `4378332` and `4378046` each carry an `[O]` event over their own observed block with `CalendarEvent.modified` equal to `ObservationRecord.modified` to the second — a signature only an inline `post_save` receiver can produce, on a database no sweep has ever been run against. Both live observations predate Phase 35's receiver change by hours; I have stated that caveat on both truths rather than hiding it, and the post-change path is re-proved at integration level by `test_updatestatus_narrows_the_event_with_no_command_run` and at corpus level by the clean dry run above.

Two items go to a human, and the first is the reason this report is `human_needed` rather than `passed`. A Phase 34 artifact committed after the last verification — the resolved debug session for G-34-2 — states that an overnight `updatestatus`-only run narrowed all 33 previously-stale LCO events. The database says no run occurred after 2026-09-12T22:10:48Z, and that 14 of those 33 are still stale today; the dry-run sweep puts the number at exactly `updated: 14`. This is finding F-34-1, which `34-UAT.md` recorded and the previous verification explicitly corrected — the newer doc re-asserts the premise that correction removed. Nothing in the code depends on it and no must-have fails because of it, so it is a warning, not a blocker: the audit trail for this phase's one verification-over-time requirement needs a human to either fix the wording or produce the missing run. The second item is the operator decision that follows from it — one sweep clears the 14 legacy events, which is the backstop role this phase shipped the sweep for.

The phase goal itself — one writer for observation-backed nights, drawn and kept current with no operator command — holds in the codebase today, with Phase 35's changes in place.

---

_Verified: 2026-09-14T22:39:20Z_
_Verifier: Claude (gsd-verifier)_
