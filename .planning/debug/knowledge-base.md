# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## start-time-idempotency-key — load_telescope_runs silently duplicated CalendarEvents on re-ingest because its lookup key required exact equality on a drift-prone computed sunset time
- **Date:** 2026-07-08
- **Error patterns:** load_telescope_runs, insert_or_create_calendar_event, get_or_create, CalendarEvent, duplicate, idempotency, start_time, sun_event, IERS, Earth-orientation, UT1-UTC, AltAz, astropy, exact-match lookup key, telescope_runs, SYNC-04, silent duplicate creation
- **Root cause:** The create-or-update lookup key {telescope, instrument, start_time} used exact datetime equality on start_time, a value computed fresh each run by telescope_runs.sun_event(). sun_event() is deterministic within a process but its output is a direct function of astropy's IERS Earth-orientation data (UT1-UTC / polar motion) applied in the get_sun/AltAz transform, which astropy refreshes over time — so independent ingests of the identical (site, night) days apart produced start_time values ~2s apart, defeating get_or_create's exact match and creating near-duplicate rows.
- **Fix:** Added an optional `start_time_tolerance` kwarg to the shared insert_or_create_calendar_event() helper; when set (only load_telescope_runs passes it, at 5 minutes) it matches an existing event whose start_time is within a +/- window (start_time__range, a proximity window rather than minute-bucketing which would still split events across a bucket boundary) scoped by the other lookup keys, leaving the stored start_time pinned to avoid churn. Default None keeps exact get_or_create behaviour, so URL-keyed sync callers (sync_lco/sync_gemini, lookup {'url': ...}) are unaffected.
- **Files changed:** solsys_code/calendar_utils.py, solsys_code/management/commands/load_telescope_runs.py, solsys_code/tests/test_calendar_utils.py, solsys_code/tests/test_load_telescope_runs.py, docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
---
