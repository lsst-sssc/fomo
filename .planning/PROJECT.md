# Telescope Runs Calendar — Stages 1, 2 & 3

## What This Is

A helper module and management commands for FOMO that:

1. (`solsys_code/telescope_runs.py`) resolves a telescope name to its observing site (via the `Observatory` model, by MPC obscode) and computes dip-corrected UTC sunset, sunrise, and -15° dark-window crossing times for a given date — Stage 1.
2. (`solsys_code/management/commands/load_telescope_runs.py`) parses classical-schedule run lines and idempotently creates one `tom_calendar.CalendarEvent` per observing night, populated with sunset/sunrise times and the -15° dark window — Stage 2.
3. (`solsys_code/management/commands/sync_lco_observation_calendar.py`) syncs LCO queue ObservationRecords (FTS/MuSCAT4) to the calendar as unified CalendarEvents — starting as a scheduling-window banner and updating in place to the placed block once the LCO scheduler acts — Stage 3.

This is a Stages-1-through-3-complete implementation of the "telescope runs on the calendar" feature (issue #37). Stage 4 (full observation-record sync for all facilities) remains future work.

## Current State

**Shipped:**
- ✅ v1.0 "Site/Ephemeris Helper" — 2026-06-14 (Phase 1)
- ✅ v1.1 "Classical Run Ingest" — 2026-06-16 (Phases 2-3)
- ✅ v1.2 "LCO Queue Calendar Sync" — 2026-06-17 (Phase 4)

**Working code:**
- `solsys_code/telescope_runs.py`: `SITES`, `get_site()`, `horizon_dip()`, `sun_event()`, `ParsedRun`, `parse_run_line()`, `KNOWN_STATUSES`
- `solsys_code/management/commands/load_telescope_runs.py`: `load_telescope_runs` BaseCommand
- `solsys_code/management/commands/sync_lco_observation_calendar.py`: `sync_lco_observation_calendar` BaseCommand, `SITE_TELESCOPE_MAP`
- `solsys_code/tests/test_telescope_runs.py`: 26 tests (16 site/ephem + 10 parser)
- `solsys_code/tests/test_load_telescope_runs.py`: 6 tests (ingest + idempotency + error paths)
- `solsys_code/tests/test_sync_lco_observation_calendar.py`: 15 tests (selection, banner/placed sync, no-churn, terminal-state prefixes, error paths)
- `docs/notebooks/pre_executed/telescope_runs_demo.ipynb`: Stage 1 demo
- `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb`: Stage 2 demo
- `docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`: Stage 3 demo
- **All 110 `./manage.py test solsys_code` tests pass.**

## Core Value

Stage 1 (v1.0): Sun-event times accurate to within 2 minutes of the Las Campanas skycalc reference tool — the foundation that Stages 2-4 build on. Also: validated the GSD discuss→plan→execute→verify loop end-to-end on this codebase.

Stage 2 (v1.1): A `load_telescope_runs` management command turns classical-schedule run lines into accurate, idempotent `tom_calendar.CalendarEvent`s — one per observing night — using Stage 1's `SITES`/`get_site()`/`sun_event()` for sunset/sunrise times.

Stage 3 (v1.2): A `sync_lco_observation_calendar` management command syncs LCO queue ObservationRecords (FTS/MuSCAT4) to the calendar — one CalendarEvent per record, keyed on the LCO portal URL, transitioning from a scheduling-window banner (`parameters['start'`/`'end']`) to a placed block (`scheduled_start`/`scheduled_end`) as the scheduler acts, and updating in place if the block is rescheduled.

## Milestone Status

v1.2 "LCO Queue Calendar Sync" shipped 2026-06-18 (Phase 4). v1.3 "Full LCO Facility Sync" started 2026-06-18.

## Current Milestone: v1.3 Full LCO Facility Sync

**Goal:** Generalize `sync_lco_observation_calendar` to correctly sync all LCO-family facilities (LCO + SOAR) for all real site codes and any combination of proposals, fixing the parameter-shape bugs found in v1.2 against real data.

**Target features:**
- `--proposal` accepts a comma-separated list or `ALL` (sync every LCO-family record regardless of proposal)
- Facility scope: LCO + SOAR only; Gemini and ESO explicitly deferred
- Instrument-type extraction scans `c_1..c_5` configs for the one with a populated `exposure_time`, instead of assuming a flat `instrument_type` key
- Telescope label: try an LCO API call per record for the real site/enclosure/telescope, map through the verified fully-qualified-code dict (`siteid-enclid-telid` -> MPC code -> label); fall back to a coarse instrument-class label (`1m0`/`0m4`/`2m0`) if that call fails
- Site/telescope mapping is a verified static dict (not `Observatory`-model-tied), covering all 8 real LCO sites

## Requirements

### Validated

- ✓ `Observatory` model stores MPC-obscode-keyed site `lat`/`lon`/`altitude` with geodetic/geocentric conversion helpers — existing
- ✓ `tom_calendar.models.CalendarEvent` has the fields needed to represent a telescope run — existing
- ✓ `tom_observations.models.ObservationRecord` carries `scheduled_start`/`scheduled_end`/status for real observation blocks — existing, used by Stage 4
- ✓ `get_site()` resolves a telescope name to an `Observatory` record + `EarthLocation` + timezone — v1.0 (SITE-01, SITE-02)
- ✓ `sun_event(site, date, kind)` returns UTC sunset, sunrise, and -15° dark crossings with dip correction — v1.0 (EPHEM-01, EPHEM-02)
- ✓ Horizon-dip helper returns 1.44° ± 0.02° at 2402 m — v1.0 (EPHEM-03)
- ✓ Las Campanas sunset/sunrise for June 2026 agree with skycalc to within 2 minutes — v1.0 (EPHEM-04)
- ✓ Astronomical twilight (-18°) for Las Campanas on 10 June 2026 agrees with skycalc to within 2 minutes — v1.0 (EPHEM-05)
- ✓ `America/Santiago` / `Australia/Sydney` DST offsets correct — v1.0 (EPHEM-06)
- ✓ Observatory records exist for Magellan (Las Campanas), NTT (La Silla), FTS (Siding Spring) — v1.0 (SITE-03)
- ✓ **PARSE-01**: Parse classical run line into `ParsedRun(telescope, instrument, status, year, month, day1, day2)`, both date-range orderings — v1.1 (Phase 2)
- ✓ **PARSE-02**: Hyphenated instrument names parse as single token — v1.1 (Phase 2)
- ✓ **PARSE-03**: No-year defaults to current year; late-December rolls to next year — v1.1 (Phase 2)
- ✓ **INGEST-01**: `load_telescope_runs` expands `S..E` into `E - S + 1` nightly CalendarEvents (`start_time = sunset(d)`, `end_time = sunrise(d+1)`) — v1.1 (Phase 3)
- ✓ **INGEST-02**: Each event sets `telescope`/`instrument`/`title` and `description` with -15° dark window, status, and original run line text — v1.1 (Phase 3)
- ✓ **INGEST-03**: Running the command twice on the same file creates no duplicate CalendarEvents — v1.1 (Phase 3)
- ✓ **SELECT-01**: `sync_lco_observation_calendar --proposal <code>` syncs all `ObservationRecord(facility='LCO')` matching `parameters['proposal']` — v1.2 (Phase 4)
- ✓ **SYNC-01**: One `CalendarEvent` per matching record, keyed on `url` = `LCOFacility().get_observation_url(observation_id)` — v1.2 (Phase 4)
- ✓ **SYNC-02**: When `scheduled_start` is `None`, event times come from `parameters['start']`/`parameters['end']` (window banner); title is `[QUEUED]`-prefixed — v1.2 (Phase 4)
- ✓ **SYNC-03**: When `scheduled_start`/`scheduled_end` are populated, event times are set from those values (placed block replaces banner) — v1.2 (Phase 4)
- ✓ **SYNC-04**: Re-running after rescheduling updates the existing event in place, no duplicates, no `modified` churn on unchanged records — v1.2 (Phase 4)
- ✓ **SYNC-05**: `telescope`, `instrument`, `proposal` on `CalendarEvent` are populated from the record — v1.2 (Phase 4)
- ✓ **TERM-01**: Terminal-failure states (WINDOW_EXPIRED/CANCELED/FAILURE_LIMIT_REACHED/NOT_ATTEMPTED) get a `[EXPIRED]`/`[CANCELLED]`/`[FAILED]` title prefix; event is retained; `COMPLETED` gets a clean title — v1.2 (Phase 4)

### Active

- [ ] `--proposal` accepts a comma-separated list of proposal codes, or `ALL`
- [ ] Facility scope generalized to LCO + SOAR (both share `LCOFacility`'s OCS API/parameters shape)
- [ ] Instrument-type extraction scans `c_1..c_5_instrument_type` configs for the meaningful one (populated `exposure_time`), replacing the broken flat-key assumption
- [ ] Telescope label resolved via per-record LCO API call (`/api/requests/<id>/observations/`) mapped through the verified fully-qualified-code dict, with a coarse instrument-class fallback (`1m0`/`0m4`/`2m0`) if the call fails
- [ ] Verified static site/telescope mapping dict covering all 8 real LCO sites (replaces the 2-site `[ASSUMED]` `SITE_TELESCOPE_MAP`)

### Out of Scope

- Gemini facility support — different base class (`BaseRoboticObservationFacility`), stub `get_observation_url()` (no portal URL to key the idempotent sync on), different parameter keys and terminal-states vocabulary than LCO
- ESO/NTT facility support — classically scheduled, already handled by Stage 2 (`load_telescope_runs`); never goes through `ObservationRecord`/queue sync
- Status-aware `CalendarEvent` coloring (telescope/proposal-keyed hash, opacity by `[QUEUED]`/terminal-prefix status) — explicitly deferred during v1.2 close; see `.planning/todos/pending/2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`; requires a project-level `tom_calendar` template override since the upstream model's `color` property isn't separately overridable
- Any `tom_calendar` UI/template changes beyond the v1.2-shipped `[QUEUED]` de-emphasis style — not needed for Stages 1-3
- Reworking `tom_observations`' existing astroplan-based visibility/airmass plots — separate, not touched by this feature
- Distinguishing Magellan Baade vs Clay in `telescope` field — open item; bare `'Magellan'` is deliberately ambiguous (both at Las Campanas, same ephemeris)
- Replacing `SITES`'s hardcoded telescope-name → obscode mapping with a data-driven `Observatory.short_name` lookup — Stage 2+ consideration, not required for Stage 2 success criteria

## Context

- **Codebase**: FOMO (Django + TOM Toolkit), Solar System follow-up coordination.
- **Design doc**: `docs/design/telescope_runs_calendar.rst` — full feasibility study and 4-stage implementation plan for issue #37.
- **Experiment doc**: `docs/design/gsd_experiment.rst` — rationale for using this feature as a GSD trial.
- **Site coordinate sourcing**: coordinates come from `Observatory` records (by MPC obscode), not hardcoded constants.
- **Two-test-suite split**: `solsys_code/` Django app tests run via `./manage.py test solsys_code`; pure-Python helpers under `tests/` run via `python -m pytest`.
- **SPICE kernel side effect**: `telescope_runs.py` avoids importing `solsys_code.ephem_utils` (triggers ~1.6 GB SPICE kernel download).
- **Environment**: `tomtoolkit==3.0a9`/`tom_catalogs` mismatch (v1.0 blocker) resolved by PR #38 (merged 2026-06-11).
- **Current codebase state (as of v1.2 close)**: ~4,735 LOC across `solsys_code/`; 110 tests passing under `./manage.py test solsys_code` (26 site/ephem+parser, 6 ingest, 15 sync); `ruff check .` clean repo-wide; `ruff format --check .` clean for all GSD-touched files (two pre-existing files — `src/fomo/settings.py`, `load_telescope_runs_demo.ipynb` — are flagged but untouched by any phase, tracked in `.planning/phases/04-lco-queue-sync-command/deferred-items.md`).
- **Known technical debt**: status-aware `CalendarEvent` coloring not implemented (see Out of Scope); `[QUEUED]` de-emphasis style required a same-day contrast-regression follow-up fix (260618-lw4 → 260618-mck).
- **v1.2 correctness bug found against real data (drives v1.3)**: checked real `ObservationRecord` rows in this dev DB (pk=1 obs_id=3780553 PENDING, pk=2 obs_id=3781325 COMPLETED, both proposal `LTP2025A-004`). Neither has a `site` key in `parameters`, and neither has a flat `instrument_type` key — real LCO submissions use multi-configuration cadence requests (`c_1_instrument_type`..`c_5_instrument_type`, each with `c_N_ic_1..5_*` exposure settings); only the configuration(s) with a populated `exposure_time` are "meaningful" (in both records checked, only `c_1` was populated). `SITE_TELESCOPE_MAP`'s 2-entry `coj`/`ogg` dict was also `[ASSUMED]`/web-search-only, never confirmed against real data. v1.2's shipped command would silently `KeyError`/skip every real record in this database.
- **LCO site -> MPC code reference table** (from https://lco.global/observatory/sites/mpccodes/), basis for the v1.3 verified mapping dict: ogg/Haleakala (F65,T04,T03), elp/McDonald (V37,V39,V38,V45,V47), lsc/Cerro Tololo (W85,W86,W87,W89,W79), cpt/Sutherland (K91,K92,K93,L09), coj/Siding Spring (Q58,Q59,Q63,Q64,E10), tfn/Tenerife (Z31,Z24,Z21,Z17), tlv/Wise Observatory (097), sor/SOAR Cerro Pachon (I33). A bare 3-letter site code is 1-to-many against MPC codes; the fully-qualified `siteid-enclid-telid` code (e.g. `coj-clma-2m0a` -> `E10` -> "FTS") is 1-to-1.

## Constraints

- **Astronomy library**: `astropy` for sun-position calculations.
- **Timezones**: `zoneinfo` (stdlib, `tzdata` installed).
- **Data source**: Site coordinates from `Observatory` model records (MPC obscode lookup).
- **Precision**: Sunset/sunrise must match Las Campanas skycalc to ≤ 2 minutes; horizon dip at 2402 m must be 1.44° ± 0.02°.
- **DB precision**: `astropy Time.to_datetime()` produces microseconds — strip with `.replace(microsecond=0)` before any DB key lookup.
- **Testing**: DB-dependent tests in `solsys_code/tests/`, run with `./manage.py test solsys_code`. Quality gates: `ruff check .` and `ruff format --check .`.

## Demo Notebooks

Each phase ships a demo notebook under `docs/notebooks/pre_executed/`. Notebooks require manual execution to see outputs (pre-commit hook clears all `.ipynb` output cells — consistent project convention). Use the Django setup boilerplate from the Django setup boilerplate section below before importing any model.

### Django setup boilerplate for notebooks

```python
import os
import sys
from pathlib import Path

import django

repo_root = str(Path.cwd().resolve().parents[2])  # adjust depth to repo root
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')
os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

django.setup()
```

Without the `sys.path` fix, imports fail with `ModuleNotFoundError: No module named 'src'`. Without `DJANGO_ALLOW_ASYNC_UNSAFE`, ORM calls raise `SynchronousOnlyOperation`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Source `SITES` coordinates from `Observatory` model by MPC obscode | Avoids duplicating lat/lon/altitude already modeled in `solsys_code_observatory`; `tom_observations.facilities.lco` is incomplete/inconsistent for this purpose | Implemented in Phase 01 via `get_site()` and `Observatory.to_earth_location()` |
| Scope Stage 1 GSD run to a single self-contained unit | Per `gsd_experiment.rst` — trial the GSD workflow before committing to the full 4-stage feature | Phase 01 completed end-to-end; 9/9 requirements validated; GSD loop validated |
| DB-dependent tests go in `solsys_code/tests/` (Django suite) | Consistent with existing two-suite split; pure-Python `tests/` suite has no DB access | Implemented in Phase 01 (`test_telescope_runs.py`), Phase 03 (`test_load_telescope_runs.py`) |
| Telescope token resolved by prefix match against `SITES.keys()` | Exact match wins; 2+ matches raise `ValueError` listing candidates — no silent guessing (D-01) | Implemented in Phase 02; bare `'Magellan'` correctly raises `ValueError` naming both Clay/Baade |
| Three date-range regex patterns tried in order | month-after-range → cross-month → month-before-range; covers all 3 sample formats | Implemented in Phase 02; all 4 success criteria pass |
| CalendarEvent upsert keyed on `(telescope, instrument, start_time)` via `get_or_create` + conditional save | Idempotent re-run; no `modified`-timestamp churn on unchanged events (D-04) | Implemented in Phase 03; INGEST-03 validated by test and UAT |
| `astropy Time.to_datetime()` microsecond-strip | `to_datetime()` produces sub-second precision that breaks `get_or_create` key matching on re-run | Fixed in Phase 03 code review (commit `437aa53`); `.replace(microsecond=0)` before DB save |
| Per-line `(ValueError, Observatory.DoesNotExist)` handler (log+skip) | Both are data/setup issues for that line; abort would discard all subsequent valid lines | Implemented in Phase 03 (D-02); skipped lines reported with line number + original text |
| `CalendarEvent.url` keyed on `LCOFacility().get_observation_url(observation_id)`, not the literal `requestgroups/<id>/` string from the original ROADMAP wording | Real method returns `/requests/<id>` (no trailing slash); using the wrong literal would silently break find-or-create matching against real LCO data (D-01) | Implemented in Phase 04; confirmed live via `LCOFacility().get_observation_url('12345')`; `grep -c requestgroups` on source = 0 |
| Terminal-state title prefix trigger uses `get_failed_observing_states()` (4 states), not `get_terminal_observing_states()` (5 states = those 4 + `COMPLETED`) | Research correction (D-06): the wrong helper would wrongly prefix `COMPLETED` records, which should get a clean title | Implemented in Phase 04; verified live (4-vs-5 state sets) plus a dedicated COMPLETED-gets-clean-title test |
| No-churn create-or-update compares all 7 changeable fields before `.save()` | Avoids `modified`-timestamp churn on unchanged records, same pattern as Phase 03's `load_telescope_runs` | Implemented in Phase 04 (SYNC-04); verified by a test asserting bit-for-bit-identical `modified` on an unchanged re-run |
| Status-aware `CalendarEvent` coloring deferred rather than built alongside the narrower `[QUEUED]` de-emphasis fix | Visual-design decision (telescope/proposal-keyed hash + status opacity), not just engineering; user wanted to explore "striping" as an alternative before committing | Narrower de-emphasis fix shipped (260618-lw4/mck); fuller scheme captured as a pending todo for a future milestone |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-18 — v1.3 "Full LCO Facility Sync" milestone started*
