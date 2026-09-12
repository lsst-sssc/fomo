---
phase: 34-the-observation-projector-trigger
verified: 2026-09-12T00:20:00Z
status: passed
score: 14/14 must-haves verified
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

covered_digest: "v1:sha256:534a506170e25db4e01775140eaa25cd7dd1b1ae3083cbd6db1bd280a83cb660"
behavior_unverified: 1
overrides_applied: 0
decision_coverage:
  honored: 21
  total: 21
  not_honored: []
re_verification:
  previous_status: gaps_found
  previous_score: 11/14
  gaps_closed:
    - "G-34-3 (the only open gap): the paired demo notebook's takeover demonstration is real again and is now self-guarding. Cell `05528b38` reports `33 of 159 pre-existing facility-url-keyed events were re-titled by the takeover` with 8 concrete before -> after pairs (e.g. `'[Q] 1m0 11P'` -> `'[O] TFN-1m0 11P'`); cell `556d2a9f`'s first and second sweep lines differ (`created: 0, updated: 33, unchanged: 126, site_lookups: 14` vs `created: 0, updated: 0, unchanged: 159, site_lookups: 0`) and it prints `First sweep work (created + updated, every facility): 33`. Cell `7022f987` names the database used: `/home/tlister/git/fomo_devel/tmp/34-07-fresh-clone.sqlite3 -- routed to a scratch copy`. The prose (cells `7e7bd66e`, `8eeddc83`, `6a9bd576`) and the closing table (`35debc54`, now f-strings over this run's own variables) all quote the run's real numbers."
  gaps_remaining: []
  regressions: []
gaps: []
advisory: []
behavior_unverified_items:

  - truth: "Over real nights a pending KEY2026B-004 record's event narrows queued -> placed -> observed with nobody running anything (SCHED-06, ROADMAP criterion 4)."
    test: "From now, run ONLY `python manage.py updatestatus` over several real observing nights -- never `python manage.py project_observation_calendar`. Then re-execute `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` end to end, UN-routed (no FOMO_DATABASE_PATH), and `git diff` its SCHED-06 section against `project_observation_calendar_demo.sched06-baseline.json` (74 pending records at baseline: 56 queued, 18 placed, captured 2026-09-11T04:44:59.526430+00:00)."
    expected: "At least one KEY2026B-004 record has moved queued -> placed (or placed -> observed) and its calendar event span/title narrowed to match, with no sweep run in between. Record the outcome in the dated re-check table in `34-UAT.md` and flip Test 4's verdict from blocked."
    why_human: "SCHED-06 is a verification-over-time requirement -- it depends on the real LCO scheduler placing and observing real requests on real nights. No test or grep can produce that evidence; only elapsed observing time can. Plan 34-04 deliberately established the baseline and left the re-check open; the ROADMAP scope note says the same."
    advisory: "Read the re-execution's OWN first sweep summary line first: if it reports `created: 0, updated: 0` for the narrowed records, the post_save receiver -- not the sweep -- did the narrowing. Note that the developer database currently holds 33 stale LCO events (the ones the broken pre-34-05 receiver failed to narrow; the 34-07 clone run re-titled exactly those 33). Their repair by the receiver alone on the next real `updatestatus` is itself part of the evidence. An un-routed re-execution overwrites the baseline JSON in place -- that is intended, and it is what `git diff` compares."
insufficient_spec_items:

  - truth: "If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state (34-01 must_haves, verification: backstop)."
    test: "Drive two concurrent/interleaved saves of one LCO ObservationRecord (e.g. two `updatestatus` runs overlapping) against a COPY of the developer database (`FOMO_DATABASE_PATH=<absolute scratch path>`) and inspect the resulting CalendarEvent span and title against the record's final persisted fields."
    expected: "The surviving event matches the record's final persisted scheduled_start/scheduled_end/status -- no event left describing a superseded intermediate state, and no `unprojectable ... AttributeError` line in the logs."
    why_human: "Declared non-inferable (`verification: backstop`). No held-out or property-based test exercises interleaved saves; the suite is single-threaded. 34-UAT.md Test 2 did run this, but the run was contaminated by G-34-2 (AttributeError on nearly every record) and recorded as `issue`/blocker. G-34-2 is now closed, so a clean re-run is cheap -- but no post-fix interleaved-save evidence exists yet."
human_verification:

  - test: "SCHED-06 / UAT Test 4 (already tracked in 34-UAT.md -- MERGE into the existing tracker, do not overwrite it). Run ONLY `python manage.py updatestatus` against `src/fomo_db.sqlite3` over several real observing nights, never the sweep. Then re-execute the demo notebook UN-routed and `git diff` the SCHED-06 baseline JSON. Read the re-execution's own FIRST sweep summary line before anything else."
    expected: "At least one KEY2026B-004 record narrowed queued -> placed (or placed -> observed) with its event following, and the first sweep reports `created: 0, updated: 0` for it -- proving the post_save receiver, not the sweep, did the narrowing. Fill in the dated row in 34-UAT.md's SCHED-06 re-check table and close SCHED-06."
    why_human: "Verification-over-time requirement; depends on real observing nights elapsing. This is Test 4 in 34-UAT.md -- merge this outcome into that tracker rather than creating a new UAT file."
  - test: "Re-run 34-UAT.md Test 2 now that G-34-2 is fixed: drive two overlapping `updatestatus` runs against a COPY of the developer database (FOMO_DATABASE_PATH=<absolute scratch path>) and inspect the surviving CalendarEvents."
    expected: "Each event matches its record's final persisted field state, and the logs contain no `unprojectable ... AttributeError` lines (the failure that made the original Test 2 run a blocker)."
    why_human: "Declared `verification: backstop` (non-inferable); no test exercises concurrency. The original operator run is unusable as evidence because G-34-2 contaminated it."
deferred:

  - truth: "The two title-prefix vocabularies (the legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` prefixes still emitted by load_telescope_runs.py and campaign_views.py, and the projector's terse `[Q]`/`[S]`/`[O]`/`[X]`/`[C]`/`[F]`/`[?]` markers) are unified into one vocabulary."
    addressed_in: "Phase 37"
    evidence: "ROADMAP Phase 37 'Status vocabulary, public tallies and provenance blind gaps' (STATUS-01/02) owns the final vocabulary; Phase 34's ROADMAP scope note states 'Title prefixes ship provisionally here; Phase 37 owns the final vocabulary.' Both vocabularies paint the correct ring today (calendar_display_extras._TERMINAL_PREFIXES carries all of them)."
---

# Phase 34: The Observation Projector & Trigger — Verification Report

**Phase Goal:** Every LCO/SOAR observation record draws its own calendar event and keeps it current on every save with no operator command, and the old LCO sync command is retired in its favour — one writer for observation-backed nights.
**Verified:** 2026-09-12T00:20:00Z (HEAD `025d741`, branch `issue37-telescope-runs-calendar`)
**Status:** passed
**Re-verification:** Yes — after gap-closure plan 34-07 (`gap_closure: true`, `gap_ids: [G-34-3]`). Supersedes the 2026-09-11T23:05:00Z report (`gaps_found`, 11/14).

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **(SC1)** Every LCO/SOAR `ObservationRecord` has exactly one calendar event keyed by its facility observation URL, spanning the request window while queued / the placed block once scheduled / the observed block once observed; a terminal-negative record keeps a visibly marked event on its window night. | ✓ VERIFIED | Regression-checked. Identity is `event_url()` = `facility.get_observation_url(record.observation_id)` and nothing else; span is re-derived from record fields on every projection (`record_time_window()` / `parameters['start'|'end']`). Notebook cell `5b5a036e` executed output over the real corpus: `159 LCO/SOAR records`, `159 facility-url-keyed events`, `records with no facility-url-keyed event: 0`, `159 - 0 = 159; matches event count: True`. Cell `055eff76` per-marker tally: `[Q]:33 [S]:19 [O]:74 [X]:26 [C]:6 [F]:1 [?]:0`. Behavioral: the 292-test phase suite (below) passes, including `TestEventFieldsFor`'s terminal-negative and `[?]` cases and `test_two_records_whose_windows_exactly_abut_produce_two_separate_events`. |
| 2 | **(SC2)** Saving a record updates its event with no operator command — including the schedule-only placement save TOM's own hook misses and the `updatestatus` path — while a save that changes nothing writes nothing and a projector error is logged rather than aborting the record save. | ✓ VERIFIED | Regression-checked. `apps.py:30-46` connects `post_save`, `m2m_changed` and `pre_delete`, each `weak=False` with a distinct `dispatch_uid`. `calendar_utils.coerce_schedule_datetime()` (L460) is routed through by **both** `record_time_window()` branches (L548-555), which is the G-34-2 fix. Behavioral evidence re-run this pass: `test_updatestatus_narrows_the_event_with_no_command_run` drives the real `LCOFacility().update_observation_status()` with a portal payload of ISO strings inside `assertNoLogs(WARNING)`; `test_schedule_only_save_narrows_the_same_event_row`, `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`, `test_raising_projector_does_not_block_a_save`, `test_record_saved_inside_a_rolled_back_transaction_leaves_no_event`, `test_make_request_is_never_called_during_a_record_save` all pass in the 292-test run. |
| 3 | **(SC3)** One sweep command re-projects any set of records (`--dry-run` supported, per-record failures isolated) as the backstop for bulk-write paths; a second sweep reports everything unchanged and no event outside the projector's key namespace is created, modified or deleted. | ✓ VERIFIED | Re-run this pass: `python manage.py project_observation_calendar --help` → `[--proposal PROPOSAL] [--facility {LCO,SOAR}] [--dry-run]`, zero required args. The notebook now carries the *live* demonstration again: first sweep `LCO: created: 0, updated: 33, unchanged: 126, unprojectable: 0, site_lookups: 14, site_lookup_failed: 1`, second sweep `created: 0, updated: 0, unchanged: 159, site_lookups: 0` — different lines, converging. Namespace isolation proven in the same run: cell `05528b38` reports `RUN: unchanged: True (72 events)`, `GEM: unchanged: True (0)`, `blank-url unchanged: True (10)` with a hard `assert` behind each. `TestDryRun`, `TestFailureIsolation`, `TestNamespaceIsolation`, `TestProjectQuerysetOrdering`, `TestObservedSiteLookup` all pass. |
| 4 | **(SC4a)** Over real nights, a user watches a `KEY2026B-004` record's event narrow queued → scheduled → observed with nobody running anything (closes spike 004's PARTIAL verdict). | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Mechanism present, wired and unit-tested (truth 2); G-34-2 fixed. Baseline still byte-identical to its `a87f5f8` state — `sha256 453ae2ba…d353859c` on both the working copy and `git show a87f5f8:…`, and `git log` on the path lists only that one commit. `src/fomo_db.sqlite3` untouched by this plan: `stat -c '%Y %s'` = `1789157248 1232896`, exactly the value the phase recorded, so its 33 stale LCO events remain as the re-check evidence. Routed to human verification (verification-over-time, per the ROADMAP scope note) — not a gap. |
| 5 | **(SC4b)** Every event title is short enough to read in a month cell (PROJ-06). | ✓ VERIFIED | Regression-checked. `title_for()` produces `'[marker] <token> <target>'` capped at 200 chars; `test_marker_and_token_within_first_16_characters` passes in the 292-test run. Real titles from this run's notebook output: `'[O] TFN-1m0 11P'`, `'[S] 1m0 11P'`, `'[Q] 1m0 11P'`. |
| 6 | **(SC5)** `sync_lco_observation_calendar` no longer exists; the same events come from the projector and sweep in the same key namespace, with runbook section, demo notebook and tests migrated rather than duplicated; `sync_gemini_observation_calendar` stays as submission-echo with its no-read-back caveat documented. | ✓ VERIFIED | Re-checked this pass: `python manage.py help` → **0** occurrences of `sync_lco_observation_calendar`; `project_observation_calendar` registered. Only `sync_gemini_observation_calendar.py` remains in `management/commands/`, only `sync_gemini_observation_calendar_demo.ipynb` in `pre_executed/`. `docs/runbooks/telescope_runs_calendar.rst` contains **0** references to the retired command and covers the projector (L55), the one-time title change (L133), the sweep and its flags (L147-196), and the `unprojectable`/`site_lookup_failed` counters. |
| 7 | A calendar visitor can tell what each projector marker means from a legend on the calendar page itself, including `[?]`. | ✓ VERIFIED | Regression-checked. `{% observation_status_legend as status_legend %}` at `calendar.html:331`; `test_returns_seven_entries_covering_every_marker` and `test_calendar_page_renders_every_legend_marker_and_label` pass in the 292-test run. |
| 8 | Every projector marker paints the right status ring, and no existing ring is lost. | ✓ VERIFIED | Regression-checked. `{% status_border_css event.title as status_border %}` at `calendar.html:238` and `:274`; `_TERMINAL_PREFIXES` retains all four legacy verbose prefixes beside the new bracket-letter markers. `TestProjectorMarkerRings` + `TestTelescopeStripeContrast` pass. |
| 9 | Series identity ("night n of N") is rendered at request time from `CalendarEventMeta.observation_group`, never stored in the event, ordered by window start, with no per-event query fan-out. | ✓ VERIFIED | Regression-checked. `{% observation_series_decoration event as series %}` at `event_form.html:148`, beside (not replacing) the campaign block; the tag body contains no write call. `TestObservationSeriesDecoration` and `test_modal_query_count_does_not_grow_with_group_size` pass. |
| 10 | The paired-docs rule is satisfied: the sweep has a pre-executed demo notebook whose executed output demonstrates the one-time takeover and a first-vs-second sweep that differ; the retired command's notebook is gone; the runbook describes the projector/sweep, legend, series block and Gemini caveat. | ✓ VERIFIED | **G-34-3 closed — and closed with a mechanism, not just a re-run.** Committed notebook (20 cells, 12 code, all 12 carrying output, nbformat 4.5 stable ids): cell `7022f987` prints `Resolved database: '…/tmp/34-07-fresh-clone.sqlite3' -- routed to a scratch copy`; cell `65e111fa` first sweep `LCO: created: 0, updated: 33, …, site_lookups: 14`; cell `05528b38` `33 of 159 pre-existing facility-url-keyed events were re-titled by the takeover` **plus** `33 of 159 … changed in at least one snapshot field` and 8 real before → after pairs; cell `556d2a9f` prints both summary lines (they differ) and `First sweep work (created + updated, every facility): 33`. Prose reconciled: cell `8eeddc83` now quotes *this run's* `LCO created: 0, updated: 33` and `33 of 159`; cell `7e7bd66e` states the run used a scratch copy; cell `6a9bd576` carries the un-swept-clone rule; cell `35debc54`'s PROJ-05/TRIG-03 rows are f-strings over `len(changed_titles)` and `first_sweep_work`, printing `33 of 159 …` and `first sweep created+updated total 33`. Registration halves unchanged: `docs/notebooks.rst:15` toctree entry, `CLAUDE.md:129` notebook-map entry, `sync_lco_observation_calendar_demo.ipynb` absent. See the mutation evidence under Behavioral Spot-Checks — the new guard fails on the *exact* pre-34-07 notebook. |
| 11 | Nothing group-derived is written into `CalendarEvent.title` or `.description`; series identity lives only in `CalendarEventMeta.observation_group` (PROJ-04 title-stem clause). | ✓ VERIFIED | Regression-checked. `write_event_meta()` writes exactly `is_verified`/`observation_record`/`observation_group`; `test_grouped_record_sets_observation_group_and_omits_name_from_title_and_description`, `test_existing_campaign_attribution_survives_projection_and_is_verified_becomes_true`, `test_stale_companion_claim_on_a_different_event_is_cleared_not_integrity_error`, `test_record_in_two_groups_links_the_lowest_pk_group` all pass. |
| 12 | The observed telescope is resolved once, stored on the record, read back as a network-free title token, and the D-07 label rename does not downgrade Phase 28's site-level attribution matching. | ✓ VERIFIED | Regression-checked. `campaign_attribution.OBSERVED_TELESCOPE_OBSCODES` bridges `FTN`/`FTS`/`SOAR` → `F65`/`E10`/`I33` ahead of `_extract_lco_site_code()`; `test_campaign_attribution.py` passes. This run's notebook shows the read-back token live in a real title: `'[O] TFN-1m0 11P'` (a site-qualified token, resolved during the sweep's 14 successful site lookups). |
| 13 | If two saves of the same record interleave, the calendar event left behind matches the record's final persisted field state. | ⚠️ insufficient_spec (abstained) | Declared `verification: backstop` in 34-01 must_haves. No held-out, property-based or concurrency test exists; the suite is single-threaded. 34-UAT.md Test 2 exercised it, but the run was contaminated by G-34-2 and recorded as `issue`/blocker. G-34-2 is now closed; no post-fix interleaved-save evidence exists. Routed to human verification. |
| 14 | A sweep interrupted partway leaves every already-processed record with a correct event, and a re-run converges with no repair step. | ✓ VERIFIED | Carried forward. Directly observed operator behaviour in `34-UAT.md` Test 3 (`result: pass`), run against a copy (`FOMO_DATABASE_PATH=/tmp/fomo_uat_copy.sqlite3`): interrupted run repaired 1 record, re-run reported `created: 0, updated: 32, unchanged: 127, unprojectable: 0`, third dry run `updated: 0, unchanged: 159`. Independently corroborated this pass by the notebook's own two-sweep convergence on a fresh clone (33 → 0). |

**Score:** 12/14 truths verified (1 present-but-behavior-unverified, 1 abstained as non-inferable, 0 failed)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Unification of the two title-prefix vocabularies (legacy verbose `[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/`[WEATHERED]` vs. the projector's terse bracket-letter markers) | Phase 37 | ROADMAP Phase 34 scope note: "Title prefixes ship provisionally here; Phase 37 owns the final vocabulary." Phase 37 = "Status vocabulary, public tallies and provenance blind gaps" (STATUS-01/02). Both vocabularies already paint the correct ring, so nothing is broken today — only unreconciled. |

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None | — | Re-verification ran; no new-scope 🛑 finding was raised this pass, so nothing was downgraded. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` | Paired demo whose executed output demonstrates a real takeover | ✓ VERIFIED (was ⚠️ HOLLOW) | 20 cells, 12 code, all 12 with committed output. Takeover non-empty (`33 of 159`, 8 sample pairs), sweep lines differ, prose reconciled, closing table computed from the run's own variables. |
| `solsys_code/tests/test_projector_demo_notebook.py` | Repo-level guard over the committed notebook evidence | ✓ VERIFIED (new) | 166-line `SimpleTestCase`, no DB. Addresses cells by nbformat id (`_CELL_ROUTING`/`_CELL_TAKEOVER_DIFF`/`_CELL_SECOND_SWEEP`/`_CELL_SCHED06_BASELINE`), raises in `setUpClass` if any id is missing. 6 tests: 5 pass, 1 skip (the un-routed branch, correctly inapplicable to this commit's scratch-routed run). **Mutation-proven** — see spot-checks. |
| In-notebook assertions, cells `05528b38` / `556d2a9f` | Raise on a vacuous scratch-routed re-execution | ✓ VERIFIED | Code read of both sources: each re-reads `FOMO_DATABASE_PATH` itself (the WR-09 self-contained pattern), and on the scratch-routed branch asserts `changed_keys`, `changed_titles`, `first_sweep_work > 0` and `first_sweep_summary != second_sweep_summary`, each with a message telling the reader to re-clone. On the un-routed branch it prints the legitimate already-converged note instead. `changed_keys` compares the **full snapshot tuple**, so a span-only or link-only change still counts — strictly stronger than the title-only diff the gap asked for. |
| `…/project_observation_calendar_demo.sched06-baseline.json` | Byte-identical to its committed state | ✓ VERIFIED | `sha256 453ae2bae6cfda579c459a4536c1c7fc5f14bd3aab510130724df0aed353859c` on both the working tree and `git show a87f5f8:<path>`; `git log` on the path shows only `a87f5f8`; `git status --porcelain` on the directory is empty. Cell `250b5d0b` printed `Routed to a scratch copy -- showing the COMMITTED baseline below, not this copy.` — the guard took the read-only branch, as designed. |
| `src/fomo_db.sqlite3` | Unmodified (SCHED-06 evidence) | ✓ VERIFIED | `stat -c '%Y %s'` = `1789157248 1232896`, identical to the value recorded before plan 34-07 and identical to the prior verification's reading. The notebook ran against `tmp/34-07-fresh-clone.sqlite3` per cell `7022f987`'s own printed output. `tmp/` is gitignored and `git ls-files tmp/` is empty — no scratch artifact leaked into the commit. |
| `solsys_code/observation_projector.py` | Projector + three signal receivers | ✓ VERIFIED | 737 lines, unchanged by 34-07. Imported by `apps.py`, the sweep command and 3 test modules. |
| `solsys_code/apps.py` | `ready()` wires the three receivers | ✓ VERIFIED | L30-46: `post_save` / `m2m_changed` / `pre_delete`, each `weak=False` + distinct `dispatch_uid`. |
| `solsys_code/calendar_utils.py` | `coerce_schedule_datetime()` used by both `record_time_window()` branches | ✓ VERIFIED | 698 lines; `coerce_schedule_datetime()` at L460, consumed at L548-549 (parameters branch) and L554-555 (scheduled_* branch, `cast()`-wrapped). |
| `solsys_code/management/commands/project_observation_calendar.py` | Backstop sweep, zero required args | ✓ VERIFIED | `--help` runs clean with `[--proposal] [--facility {LCO,SOAR}] [--dry-run]` and no required argument. |
| `solsys_code/templatetags/calendar_display_extras.py` | Rings, legend, series decoration | ✓ VERIFIED | 712 lines; all three tags called from the two templates (line numbers in truths 7-9). |
| `docs/runbooks/telescope_runs_calendar.rst` | Projector/sweep section, no stale sync content | ✓ VERIFIED | 0 references to `sync_lco_observation_calendar`; projector/sweep/counters/troubleshooting all present. 34-07's runbook determination (no edit needed — it changes no operator-visible behaviour) confirmed independently here. |
| `solsys_code/tests/*` (7 phase modules) | Behaviour pinned | ✓ VERIFIED | `Ran 292 tests … OK (skipped=1)` this pass. The single skip is the guard's un-routed branch, correctly inapplicable. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ObservationRecord.save()` | `CalendarEvent` row | `post_save` → `receiver_on_record_save()` → `project_record()` | ✓ WIRED | Carried forward; proven by `test_updatestatus_narrows_the_event_with_no_command_run` in this pass's 292-test run. |
| `SolsysCodeConfig.ready()` | the three receivers | `.connect(weak=False, dispatch_uid=…)` | ✓ WIRED | `apps.py:30-46`. |
| sweep per-record loop | `project_record()` / `preview_calendar_event_action()` | `project_queryset()` | ✓ WIRED | One projection path shared by receiver and sweep. |
| `FOMO_DATABASE_PATH` | notebook kernel + every management command | `src/fomo/settings.py:134` | ✓ WIRED | The one mechanism keeping 34-07's writes off the developer database — and it held: cell `7022f987` names the clone, and the developer DB's size/mtime are unchanged. |
| preflight dry-run over the fresh clone | the decision to execute at all | `project_observation_calendar --dry-run` before `nbconvert` | ✓ WIRED | 34-07-SUMMARY records `LCO: created: 0, updated: 33` on the clone before execution; the committed first sweep reports the same 33, so the preflight and the run agree. |
| in-notebook assertions (`05528b38`, `556d2a9f`) | `jupyter nbconvert --execute` exit status | Python `assert` inside executed cells | ✓ WIRED | An `AssertionError` in a cell aborts the conversion, so a vacuous re-execution can no longer reach a commit. Independently corroborated: 34-07-SUMMARY records the *pre-existing* per-segment convergence assert catching a real network flake on the first attempt and forcing a re-clone. |
| committed notebook evidence | every `manage.py test` run | `solsys_code/tests/test_projector_demo_notebook.py` | ✓ WIRED | The guard now runs in the default suite — this is what turns a one-off re-execution into a standing invariant. |
| `CalendarEventMeta.observation_group` | `event_form.html` | `observation_series_decoration()` | ✓ WIRED | Display-time only. |
| `calendar_utils` telescope labels | Phase 28 attribution matching | `OBSERVED_TELESCOPE_OBSCODES` / `OBSERVED_TELESCOPE_SITE_CODES` | ✓ WIRED (route differs from the plan's literal wording) | Carried forward from the prior pass: label-keyed obscode table consulted *before* `_extract_lco_site_code()`. Intent satisfied. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `event_fields_for()` | `start_time`/`end_time` | `record_time_window(record)` / `record.parameters` — never a stored event field | Yes | ✓ FLOWING |
| `telescope_token()` | `token` | `record.parameters['observed_site'/'observed_telescope']`, else `coarse_telescope_label()` | Yes (`'[O] TFN-1m0 11P'` in this run's output) | ✓ FLOWING |
| `project_queryset()` counters | `action` | `preview_calendar_event_action(before, fields)` vs a pre-sweep snapshot | Yes | ✓ FLOWING |
| notebook cell `05528b38` | `changed_titles`, `changed_keys` | Live before/after snapshots of `CalendarEvent` around a real sweep on a fresh clone | **Yes — 33 of 159, 8 sample pairs** | ✓ FLOWING (was ⚠️ HOLLOW) |
| notebook cell `556d2a9f` | `first_sweep_summary`, `second_sweep_summary`, `first_sweep_work` | Two real `call_command` sweeps, parsed with a regex over the summary lines | Yes — 33 then 0, lines differ | ✓ FLOWING (was ⚠️ HOLLOW) |
| notebook cell `250b5d0b` | committed baseline | Read from the committed JSON, never rewritten on the scratch-routed branch | Yes (read-only branch taken) | ✓ FLOWING |
| `observation_status_legend()` | legend entries | Module constant | Intentionally fixed (documented) | ✓ FLOWING (by design) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-34 test modules pass (7 modules, one run) | `python manage.py test solsys_code.tests.test_observation_projector …test_projector_demo_notebook` | `Ran 292 tests in 7.284s` / `OK (skipped=1)` | ✓ PASS |
| Notebook-evidence guard passes against the committed artifact | `python manage.py test solsys_code.tests.test_projector_demo_notebook -v2` | 6 tests: 5 ok, 1 skip (`not routed to a scratch copy` branch) | ✓ PASS |
| **Guard fails against emptied evidence** (mutation 1) | copy of the notebook with cell `05528b38`'s outputs removed + baseline sibling, via `FOMO_DEMO_NOTEBOOK_PATH` | `FAILED (failures=1)` — `AssertionError: unexpectedly None : Cell 05528b38 does not report a non-zero re-titled count: ''` | ✓ PASS |
| **Guard fails against the exact pre-34-07 regressed notebook** (mutation 2) | `git show 46d8390:…demo.ipynb` via `FOMO_DEMO_NOTEBOOK_PATH` | `FAILED (failures=1)` — `Cell 05528b38 does not report a non-zero re-titled count: '…0 of 159 pre-existing facility-url-keyed events were re-titled by the takeover.\nSample before -> after title pairs:\n'` | ✓ PASS |
| Sweep command registered with its flags | `python manage.py project_observation_calendar --help` | `[--proposal PROPOSAL] [--facility {LCO,SOAR}] [--dry-run]` | ✓ PASS |
| Retired command really gone | `python manage.py help \| grep -c sync_lco_observation_calendar` | `0` | ✓ PASS |
| Baseline JSON byte-identical to `a87f5f8` | `sha256sum` working copy vs `git show a87f5f8:<path> \| sha256sum` | both `453ae2ba…d353859c` | ✓ PASS |
| Developer DB untouched | `stat -c '%Y %s' src/fomo_db.sqlite3` | `1789157248 1232896` (unchanged) | ✓ PASS |
| No scratch artifact committed | `git ls-files tmp/`; `git status --porcelain docs/ solsys_code/ src/ CLAUDE.md` | both empty | ✓ PASS |
| Live narrowing over real nights | — | requires elapsed observing time | ? SKIP → human (truth 4) |
| Interleaved concurrent saves | — | no concurrency harness; single-threaded suite | ? SKIP → human (truth 13) |

**Mutation note.** The two mutation runs are the load-bearing evidence for truth 10. They show the guard is not a tautology over whatever the notebook happens to contain: pointed at the *actual* notebook state that produced gap G-34-3, it fails with exactly the assertion the gap describes. The regression that this phase shipped once can no longer ship silently.

**Note on DB safety:** no `updatestatus`, no `project_observation_calendar` sweep (dry-run or otherwise) and no notebook execution was performed by this verification. `--help` and the test suite (which uses a throwaway test database) were the only commands run against Django. Mutation copies were written to the session scratchpad, never into the repo.

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist in this repo and no PLAN/SUMMARY declares a probe | N/A — skipped |

### Decision Coverage

Carried forward from the prior pass: `{ total: 21, honored: 21, not_honored: [] }`. Plan 34-07 introduced no new CONTEXT decisions; it honours D-19 (takeover), D-08 (convergence), D-20 (baseline and developer-DB untouchability) explicitly, all three re-confirmed above. **Non-blocking; no status impact.**

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROJ-01 | 34-01, 34-02 | Exactly one `CalendarEvent` per LCO/SOAR record, keyed by `facility.get_observation_url()` | ✓ SATISFIED | Truth 1; notebook cell `5b5a036e` (`159 - 0 = 159`) |
| PROJ-02 | 34-01, 34-05, 34-06 | Span follows the stage: request window → placed block → observed block | ✓ SATISFIED | Truths 1-2; `TestEventFieldsFor`; `coerce_schedule_datetime()` closes the portal-string path |
| PROJ-03 | 34-01, 34-03 | Terminal-negative record keeps a visibly marked event | ✓ SATISFIED | Truths 1, 7, 8; `[X]`/`[C]`/`[F]` marker + ring + legend tests |
| PROJ-04 (title-stem clause) | **not declared in any plan's `requirements:`** | Series identity carried by real FKs, not text in the title | ✓ SATISFIED but ⚠️ **ORPHANED** | REQUIREMENTS.md L18/L106 maps the title-stem clause to Phase 34, yet no plan frontmatter claims `PROJ-04`. Delivered anyway (truths 9 and 11) — a traceability gap in the plans, not a delivery gap. Carried forward unchanged from the prior pass. |
| PROJ-05 | 34-01, 34-02, 34-03, **34-07** | No-churn; never touches an event it does not own | ✓ SATISFIED | `test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn`; `TestNamespaceIsolation`; notebook cell `05528b38`'s three `assert`ed namespace comparisons (`RUN:` 72, `GEM:` 0, blank-url 10, all unchanged) — now against a sweep that genuinely wrote 33 events, which is what makes the isolation claim meaningful |
| PROJ-06 | 34-01, 34-03 | Compact titles that fit a month cell | ✓ SATISFIED | Truth 5 |
| SCHED-06 | 34-04, 34-05, 34-06, **34-07** | A user watches a record narrow over real nights with no command | ? NEEDS HUMAN | Truth 4 — baseline committed and proven byte-identical, developer DB proven untouched, mechanism fixed and wired. Evidence requires elapsed observing time (UAT Test 4). Not failed: the ROADMAP scope note states SCHED-06 is verification-over-time, not new code. |
| TRIG-01 | 34-01, 34-05, 34-06 | `post_save` receiver in `apps.ready()`, covering schedule-only and `updatestatus` paths | ✓ SATISFIED | Truth 2; `apps.py:30-46`; `TestUpdateObservationStatusPath` |
| TRIG-02 | 34-01, 34-05, 34-06 | Single-record, idempotent, cheap, error-logged-never-aborts | ✓ SATISFIED | Truth 2; `test_make_request_is_never_called_during_a_record_save`, `test_raising_projector_does_not_block_a_save` |
| TRIG-03 | 34-02, 34-04, **34-07** | Sweep command with `--dry-run`, failure isolation, and a paired pre-executed demo notebook | ✓ SATISFIED (was ✗ BLOCKED) | Command and behaviour: truth 3. The paired-notebook clause is now met: the notebook's own output shows the first sweep doing 33 units of work and the second doing none, and a repo-level test keeps it that way (truth 10 + mutation evidence). |
| ANNOT-03 | 34-02, 34-04, **34-07** | Old LCO sync retired; runbook/notebook/tests migrated; Gemini caveat documented | ✓ SATISFIED (was ✗ BLOCKED) | Retirement, runbook migration, test migration and Gemini caveat: truth 6. The takeover evidence that proves the retirement was a plain in-place update is restored and reconciled with its own prose: truth 10. |

**Orphaned requirements:** `PROJ-04` (title-stem clause) is mapped to Phase 34 in REQUIREMENTS.md but appears in no plan's `requirements:` field. Delivered; flagged for traceability only.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| `test_observation_projector.py` | PROJ-01/02/03/05/06, PROJ-04 stem | 49 | 0 | No | Value + behavioral | ✓ Sufficient |
| `test_observation_projector_signals.py` | TRIG-01/02, PROJ-02 | 23 | 0 | No | Behavioral (drives the real `update_observation_status()`) | ✓ Sufficient |
| `test_project_observation_calendar.py` | TRIG-03, PROJ-05 | 25 | 0 | No | Value + behavioral | ✓ Sufficient |
| `test_calendar_utils.py` | PROJ-02 (G-34-2) | `TestCoerceScheduleDatetime` + `TestRecordTimeWindow` | 0 | No | Value | ✓ Sufficient |
| `test_calendar_display_extras.py` / `test_calendar_template.py` | PROJ-03/05/06, PROJ-04 stem | many | 0 | No | Value + rendered-markup | ✓ Sufficient |
| `test_projector_demo_notebook.py` **(new)** | PROJ-05, TRIG-03, SCHED-06, ANNOT-03 | 5 | 1 (branch-correct) | No | Artifact-content, mutation-proven | ✓ Sufficient |

**Disabled tests on requirements:** 0. The one skip is a documented branch guard (un-routed vs scratch-routed run), not a disabled assertion — and the complementary branch is asserted, so exactly one of the two always applies. **Circular patterns:** 0 — the guard parses a committed artifact produced by a real execution against a real database; it does not import the system under test nor regenerate its own fixture. **Insufficient assertions:** 0 — verified by mutation, not by reading.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/views.py` | 310 | `# XXX Could replace this by a creation of the missing Observatory` | ℹ️ Info | **Pre-existing** (`git blame` → `a8613bc8`, 2025-07-23, 14 months before this phase). Not introduced here; classified identically by both prior passes. Debt-marker gate does not fire. |
| `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`, `solsys_code/tests/test_projector_demo_notebook.py` | — | Debt-marker scan (`TBD\|FIXME\|XXX\|HACK\|PLACEHOLDER`) | — | **Zero hits** in both files 34-07 touched. |
| `docs/notebooks/…demo.ipynb` cell `250b5d0b` output | — | `the scratch copy currently holds 52 pending records, already mutated by the sweep cells above` | ℹ️ Info | The 74 → 52 drop in pending `KEY2026B-004` records is not caused by the sweep cells (a sweep writes events, not record statuses) — it reflects statuses that advanced on the developer database since the 04:44Z baseline, before the clone was taken. The sentence is accurate about *why the copy's snapshot is not shown* (its events were mutated), and the copy's snapshot is correctly withheld, so no claim is false. Worth tightening the wording on the next re-execution; not a gap. |

### Human Verification Required

#### 1. SCHED-06 live narrowing (UAT Test 4 — merge into the existing `34-UAT.md`, do NOT overwrite it)

**Test:** From now, run **only** `python manage.py updatestatus` against `src/fomo_db.sqlite3` over several real observing nights — never `project_observation_calendar`. Then re-execute `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` end to end **un-routed** (no `FOMO_DATABASE_PATH`) and `git diff` the SCHED-06 baseline JSON. Read the re-execution's own **first** sweep summary line before anything else.
**Expected:** At least one `KEY2026B-004` record narrowed queued → placed (or placed → observed) with its event following, and the first sweep reports `created: 0, updated: 0` for it — proving the `post_save` receiver, not the sweep, did the narrowing. Fill in the dated row in `34-UAT.md`'s SCHED-06 re-check table and flip Test 4's verdict from `blocked`.
**Why human:** Verification-over-time; depends on real observing nights elapsing. This is already **Test 4** in `34-UAT.md`.
**Standing evidence for the re-check:** the developer database still holds the 33 stale LCO events the broken pre-34-05 receiver left behind — the 34-07 clone run re-titled exactly those 33, so the same 33 repairing themselves under `updatestatus` alone is the cleanest possible demonstration. Note that an un-routed re-execution *does* rewrite the baseline JSON in place (by design; that is what `git diff` reads), whereas the current scratch-routed commit left it untouched.

#### 2. Interleaved-save re-run (UAT Test 2 — previously contaminated by G-34-2)

**Test:** Drive two overlapping `updatestatus` runs against a **copy** of the developer database (`FOMO_DATABASE_PATH=<absolute scratch path>`) and inspect the surviving `CalendarEvent`s.
**Expected:** Each event matches its record's final persisted field state, and the logs carry no `unprojectable … AttributeError` lines.
**Why human:** Declared `verification: backstop` (non-inferable); no test exercises concurrency. The original Test 2 run is unusable as evidence because G-34-2 dominated it.

#### Re-verification (2026-09-12T22:26:21Z) — both human items closed by UAT (`34-UAT.md`, commit 9792fb4)

**Item 1, SCHED-06 live narrowing (truth 4, UAT Test 3): PASS.** After one real
`python manage.py updatestatus` against `src/fomo_db.sqlite3` at 2026-09-12 22:10 UTC (no sweep has
ever run against that database), two baseline `KEY2026B-004` records narrowed queued → observed with
their events following, through the `post_save` receiver alone: `4378332` (11P) — baseline
`[Q] 1m0 11P`, no schedule → `[O] 1m0 11P` 2026-09-12 01:06:46–01:26:04, `CalendarEvent.modified` ==
`ObservationRecord.modified` == 22:10:48; `4378046` (10P) — baseline `[Q] 1m0 10P` → `[O] 1m0 10P`
2026-09-12 02:08:39–02:22:01. A real sweep on a throwaway copy changed only their site token
(`1m0` → `TFN-1m0` / `LSC-1m0`), which is sweep-only by design (34-02 D4). Evidence was gathered
read-only from scratch copies; the un-routed notebook re-execution remains a paired-docs follow-up
(recorded in `34-UAT.md` Deferred Follow-Ups).

**Item 2, interleaved-save re-run (truth 13, UAT Test 1): PASS.** Two overlapping `updatestatus`
runs against a scratch copy: 0 `AttributeError`, 0 `unprojectable`, 0 `OperationalError`, both
`Update completed successfully`; every record either run saved projects correctly. The dry-run
sweep afterwards reported `LCO: updated: 14`, initially logged as gap G-34-1 and withdrawn the same
day: the 14 are terminal (COMPLETED) records whose `modified` is 2026-09-11 20:02–20:04 UTC —
before the 34-05 fix — and `tom_observations.facility.update_all_observation_statuses()` excludes
terminal states (`facility.py:573`), so no `updatestatus` run ever re-saves them. They are pre-fix
legacy residue that only the sweep can repair (recorded as finding F-34-1 in `34-UAT.md`). This
also corrects the premise in 34-06/34-07 and in this report's original item 1 that "the next
`updatestatus` run repairs all 33 stale events": the 19 non-terminal ones did; the 14 terminal ones
never will without one sweep.

### Gaps Summary

**No gaps remain.** G-34-3 — the sole open item from the previous pass — is closed, and closed better than the gap asked for.

The gap asked for three things: re-execute the notebook against an un-swept clone, add an assertion so the demonstration can never silently empty out again, and reconcile the prose. All three are present and independently confirmed against the artifact rather than the SUMMARY. The takeover reports `33 of 159` events re-titled with eight real before → after pairs; the first and second sweep lines differ (`updated: 33` then `updated: 0`); the framing markdown and the closing table now quote the run's own numbers, the latter as f-strings over the very variables the cells computed, so they cannot drift from the output again.

Two things push this past the letter of the gap. First, the in-notebook guard compares the **full event snapshot tuple**, not just titles — a sweep that narrowed a span or re-linked a companion row without touching a title would now still count as work, where a title-only diff would have shown nothing. Second, the demonstration is no longer guarded only at execution time: `solsys_code/tests/test_projector_demo_notebook.py` checks the committed artifact on every `manage.py test` run. I mutation-tested that guard rather than taking its passing run on trust — pointed at a copy with the takeover output emptied, and separately at the *actual* pre-34-07 notebook from commit `46d8390`, it fails both times with precisely the assertion the gap describes (`Cell 05528b38 does not report a non-zero re-titled count`). The regression this phase shipped once cannot ship silently again.

The two hard constraints held. `project_observation_calendar_demo.sched06-baseline.json` is byte-identical to its `a87f5f8` state (matching sha256 against `git show`, single commit on the path), and `src/fomo_db.sqlite3` carries the same size and mtime the phase recorded before 34-07 began — the notebook ran against `tmp/34-07-fresh-clone.sqlite3`, as its own setup cell prints. Nothing from `tmp/` reached the commit.

Everything the previous pass verified was re-checked, not assumed: the retired command is still absent from `manage.py help`, the three receivers are still wired in `apps.ready()` with `dispatch_uid`s, the template tags are still called from both templates, and the seven phase test modules pass as one 292-test run.

Two items remain for a human, both expected and both already tracked in `34-UAT.md`: Test 4 (SCHED-06 live narrowing, which only elapsed observing nights can produce) and a clean re-run of Test 2 (interleaved saves, whose original run was contaminated by the now-closed G-34-2). Neither is a code gap; the phase goal itself — one writer for observation-backed nights, drawn and kept current with no operator command — is achieved in the codebase.

---

_Verified: 2026-09-12T00:20:00Z_
_Verifier: Claude (gsd-verifier)_
