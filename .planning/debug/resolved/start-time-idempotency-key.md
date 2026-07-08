---
status: resolved
trigger: "load_telescope_runs' create-or-update idempotency key includes an exact-match `start_time` computed by telescope_runs.sun_event(). That computed timestamp is not stable across separate ingests of the identical run line/night (observed ~2 second drift for the same (site, date) between an ingest on 2026-07-01 and a re-ingest on 2026-07-08), so re-running the loader against unchanged source data silently creates duplicate CalendarEvent rows instead of matching/updating the existing one."
created: 2026-07-08T00:00:00Z
updated: 2026-07-08T12:00:00Z
---

## Current Focus

reasoning_checkpoint:
  hypothesis: "load_telescope_runs' create-or-update lookup key includes an exact-match `start_time` computed by telescope_runs.sun_event(); that computed sunset/sunrise time drifts by a second or two between independent ingests of the identical (site, night) because astropy's IERS Earth-orientation data (UT1-UTC / polar motion) applied to the AltAz frame transform is refreshed between runs, so the exact-datetime lookup misses the existing row and get_or_create creates a near-duplicate CalendarEvent."
  confirming_evidence:
    - "Real dev DB: pk16 (07-17 22:10:56, created 07-01) vs pk48 (07-17 22:10:58, created 07-08) and pk17/pk49 (07-18 22:11:27 vs :29) -- identical source line, +2s systematic drift across a 7-day gap, two duplicate rows."
    - "Experiment (A): running the exact bisection sun-event math twice in one process gives bit-identical output (22:10:58.652, delta 0.000000 s) -- NO in-repo nondeterminism."
    - "Experiment (B): perturbing UT1-UTC (the IERS Earth-orientation quantity astropy applies in the AltAz transform) across its physical range shifts the computed sunset by up to ~0.9 s per component -- confirms Earth-orientation data drift is the mechanism; baseline 22:10:58 matches the real pk48 row created on 07-08."
    - "Same-day immediate re-ingest reported created:0 unchanged:9 -- drift is cross-session (IERS refresh), never within one process."
  falsification_test: "If sun_event() output had varied run-to-run within a single process (experiment A non-identical), or if perturbing UT1-UTC left the sunset time unchanged (experiment B flat), the IERS-mechanism hypothesis would be false. Both came out as predicted."
  fix_rationale: "The computed start_time is a legitimately drift-prone value, so it is the wrong thing to require exact equality on. Make load_telescope_runs' lookup match an existing CalendarEvent whose start_time is within a small tolerance window of the freshly computed value (proximity match, not exact) -- this addresses the root cause of the duplicate (the fragile exact key) directly. A tolerance WINDOW (not minute-truncation/rounding) is chosen deliberately: any bucketing scheme still splits an event that drifts across a bucket boundary, whereas a +/- window centred on the target never does."
  blind_spots: "end_time (computed sunrise for full-night events) and the dark-window times embedded in `description` drift by the same ~2s but are in `fields`, not the lookup key -- so cross-session re-ingest may report 'updated' (churn) rather than 'unchanged'. This is the lower-severity issue the debug file already flagged; the reported bug is the DUPLICATE, and 'updated' (no duplicate) is an acceptable outcome (per the requested regression test's own 'unchanged/updated rather than created' wording). Partial-night HHMM windows use fixed wall-clock times and do not drift."
next_action: "Implement the tolerance-window fix (add optional `start_time_tolerance` kwarg to insert_or_create_calendar_event; pass it only from load_telescope_runs), add regression tests, update load_telescope_runs_demo.ipynb, run quality gates."

## Symptoms

expected: Running `./manage.py load_telescope_runs <file>` repeatedly against an unchanged schedule file produces `created: 0` (or `updated` only for genuine field changes) on every run after the first -- CalendarEvent rows are found-and-reused via the lookup key, never duplicated.
actual: |
  Real dev-DB observation (2026-07-08 session): `Magellan-Baade IMACS 17-18 July` had 2 existing
  CalendarEvent rows (pk 16, 17; `created` timestamp 2026-07-01T08:52:5x) from an earlier ingest.
  Re-running `python manage.py load_telescope_runs Didymos_runs` today (2026-07-08) reported
  `created: 6` for that run (should have been at most 4, for the NTT rows freshly deleted moments
  earlier as part of unrelated verification) -- 2 of those 6 "created" rows were actually
  re-creations of the already-existing Magellan-Baade nights (pk 48, 49; `created` 2026-07-08),
  each ~2 seconds later in `start_time` than the corresponding pre-existing row:
  ```
  pk 16  2026-07-17 22:10:56+00:00  (created 2026-07-01)
  pk 48  2026-07-17 22:10:58+00:00  (created 2026-07-08)   <- duplicate of pk 16's night
  pk 17  2026-07-18 22:11:27+00:00  (created 2026-07-01)
  pk 49  2026-07-18 22:11:29+00:00  (created 2026-07-08)   <- duplicate of pk 17's night
  ```
  A same-session immediate re-run (`load_telescope_runs Didymos_runs` again, seconds later) DID
  correctly report `unchanged: 9` with no new duplicates -- so the drift is not moment-to-moment
  nondeterminism within a single process/day, it manifested across a ~7-day gap between ingests.
  The stale pk 16/17 rows have since been deleted (unrelated cleanup); pk 48/49 remain and are
  confirmed stable under immediate re-ingest.
errors: None raised -- silent duplicate creation, not an exception. No existing test catches this
  (solsys_code/tests/test_load_telescope_runs.py exercises create/update/unchanged paths with
  presumably-stable fixture times, not a real cross-session sun_event() drift scenario).
reproduction: |
  Hard to reproduce deterministically in a single session (see actual: same-day re-run was stable).
  Best current repro is historical: compare `CalendarEvent.objects.filter(telescope='Magellan-Baade')`
  `start_time` + `created` values from before vs. after 2026-07-08's re-ingest (see actual, pk
  16/17/48/49). A more reliable synthetic repro: call `telescope_runs.sun_event()` for a fixed
  future (site, date) now, then again after forcing astropy to refresh its cached IERS table
  (e.g. `astropy.utils.iers.IERS_Auto.open(); astropy.utils.iers.LeapSeconds.auto_open()` or
  clearing `~/.astropy` cache) and diff the two `Time` results.
started: Unknown -- first observed as a side effect of 2026-07-08 verification of the (separately
  fixed) ESO/NTT off-by-one night-count bug (debug session eso-run-noon-off-by-one.md). The
  Magellan-Baade duplicate predates that fix and is unrelated to it (Magellan's night-count
  formula was not touched by that fix).

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-07-08T00:00:00Z
  checked: solsys_code/calendar_utils.py:295-331 (insert_or_create_calendar_event)
  found: |
    `event, created = CalendarEvent.objects.get_or_create(**lookup, defaults=fields)`. For
    load_telescope_runs, `lookup = {'telescope': parsed.telescope, 'instrument': parsed.instrument,
    'start_time': start_time}` (see load_telescope_runs.py Command.handle). `start_time` is an
    exact `datetime` (microsecond=0 per `_resolve_window_time`'s `.replace(microsecond=0)`, so
    only sub-second jitter is stripped -- second-level drift is NOT tolerated).
  implication: The lookup key is exact-equality on a computed (not user-supplied) timestamp that
    can legitimately drift between independent computations of "the same" sunset/sunrise event.
    Any such drift, however small, defeats get_or_create's match and produces a duplicate row.

- timestamp: 2026-07-08T00:00:00Z
  checked: Live dev-DB CalendarEvent rows for telescope='Magellan-Baade', comparing `created`
    field (row-insert timestamp, not astronomical event time) against `start_time`.
  found: |
    pk 16 created 2026-07-01T08:52:52.945Z, start_time 2026-07-17T22:10:56Z
    pk 48 created 2026-07-08T08:52:48.999Z, start_time 2026-07-17T22:10:58Z  (+2s vs pk16)
    pk 17 created 2026-07-01T08:52:53.974Z, start_time 2026-07-18T22:11:27Z
    pk 49 created 2026-07-08T08:52:50.217Z, start_time 2026-07-18T22:11:29Z  (+2s vs pk17)
  implication: Two ingests of the literal same source line, 7 days apart, produced start_time
    values consistently ~2 seconds later on the second (later) run for both nights of that run --
    consistent with a systematic shift in the underlying sun-position/frame-transform computation
    (e.g. IERS Earth-orientation prediction data updated/refreshed between 07-01 and 07-08) rather
    than random jitter.

- timestamp: 2026-07-08T00:00:00Z
  checked: Immediate same-day re-ingest (`python manage.py load_telescope_runs Didymos_runs` run
    twice within the same session, seconds apart)
  found: Second run reported `created: 0, updated: 0, unchanged: 9` -- exactly matched all 9
    events (including the freshly-created pk 48/49 pair) with no further drift.
  implication: The computation is stable within a single process/day; the ~2s discrepancy only
    appears across a multi-day gap, supporting the IERS-data-refresh hypothesis over the possibility
    of the bisection root-finder itself being nondeterministic (that would jitter run-to-run
    within the same session too, and it did not).

- timestamp: 2026-07-08T00:00:00Z
  checked: solsys_code/telescope_runs.py:172-213 (_find_crossing) and :231-279 (sun_event)
  found: |
    `_find_crossing` uses a fixed-iteration (10-step) bisection with no external state or
    randomness -- purely a function of `(anchor, location, threshold_deg)`. `anchor` comes from
    `_local_noon_utc(date, site.timezone)`, purely a function of `(date, timezone)`. `location`
    comes from `site.to_earth_location()` (Observatory model fields, static). The only external,
    time-varying input to the whole chain is whatever `astropy.coordinates.get_sun` /
    `AltAz.transform_to` pulls from astropy's IERS/Earth-orientation tables under the hood for
    frame conversion -- which astropy caches locally and periodically refreshes/extends with newer
    predicted or final values, especially for near-future dates.
  implication: Confirms there is no code-level randomness in this repo's own logic; the drift
    source is external (astropy's IERS data), which is consistent with evidence entries above.
    User's proposed fix (bucket/round `start_time` to the minute for the lookup key) sidesteps
    the root cause rather than eliminating it, but is a pragmatic mitigation since IERS-driven
    drift for near-future dates is bounded to well under a minute in practice.

- timestamp: 2026-07-08T00:00:00Z
  checked: Standalone astropy experiment replicating sun_event()'s pure math for Magellan-Baade
    (lat -29.0146, lon -70.6926, alt 2402 m) on 2026-07-17 -- (A) run the 10-step bisection
    crossing-finder twice in one process; (B) recompute with a manually overridden UT1-UTC
    offset (the exact IERS Earth-orientation quantity astropy applies in the AltAz transform).
  found: |
    (A) run 1 = run 2 = 2026-07-17 22:10:58.652 UTC, delta 0.000000 s -- bit-identical.
    (B) baseline (current IERS) sunset = 22:10:58.652; UT1-UTC -500 ms -> +0.527 s,
        -100 ms -> +0.117 s, +100 ms -> -0.059 s, +500 ms -> -0.469 s, +900 ms -> -0.879 s.
        Baseline 22:10:58 matches the real pk48 row (created 07-08) to the second.
  implication: Directly confirms the root-cause mechanism. (A) proves the repo's own bisection
    is deterministic (no in-process jitter) -- eliminates in-repo nondeterminism. (B) proves the
    computed sunset time is a direct, monotonic function of the IERS UT1-UTC offset (and, by the
    same AltAz path, polar motion), which astropy refreshes between ingests -- so the ~2s
    cross-session drift observed in the dev DB is Earth-orientation-data drift, exactly as
    hypothesised. The exact-datetime lookup key is therefore fundamentally fragile against a
    value that is *designed* to be refined over time.

## Eliminated

- hypothesis: The bisection root-finder (_find_crossing) or something else in this repo's own
    code is nondeterministic, producing the run-to-run start_time drift.
  evidence: Experiment (A) -- the exact bisection math run twice in one process is bit-identical
    (22:10:58.652, delta 0 s); and the same-day immediate re-ingest reported unchanged:9 with no
    drift. In-process nondeterminism would have shown up in both and did not.
  timestamp: 2026-07-08T00:00:00Z

## Resolution

root_cause: |
  load_telescope_runs.Command.handle() builds its create-or-update lookup key as
  {'telescope', 'instrument', 'start_time'} where start_time is a fresh output of
  telescope_runs.sun_event() -> _find_crossing()'s bisection root-find. sun_event() is
  deterministic within a process (confirmed bit-identical, experiment A) but its result is a
  direct function of astropy's IERS Earth-orientation data (UT1-UTC / polar motion) applied in
  the get_sun/AltAz transform (confirmed: perturbing UT1-UTC shifts the sunset time, experiment
  B). astropy refreshes that IERS data over time, so independent ingests of the identical (site,
  night) days/weeks apart legitimately produce start_time values a second or two apart (observed
  +2s across a 7-day gap in the dev DB, pk16->48 and pk17->49). Because the shared helper
  insert_or_create_calendar_event() uses CalendarEvent.objects.get_or_create() with EXACT
  equality on that lookup key, the drifted start_time misses the existing row and a near-duplicate
  CalendarEvent is silently created -- breaking the SYNC-04 idempotency contract. The URL-keyed
  sync commands are unaffected because their lookup key ({'url': ...}) is a stable
  externally-supplied identifier, not a recomputed value.
fix: |
  Added an optional keyword `start_time_tolerance: timedelta | None = None` to the shared
  helper insert_or_create_calendar_event() (solsys_code/calendar_utils.py). When provided AND
  the lookup contains a `start_time`, the helper matches an existing CalendarEvent whose
  start_time is within +/- that tolerance of the lookup value (a proximity WINDOW via
  start_time__range, not exact equality and not minute-bucketing), scoped by the other lookup
  keys; if matched it updates-or-leaves-unchanged (the stored start_time is deliberately left
  pinned to the first-ingested value to avoid churn), otherwise it creates with the exact
  computed start_time. Default None preserves the exact get_or_create() behaviour, so the
  URL-keyed sync callers (sync_lco / sync_gemini, lookup {'url': ...}) are unchanged.
  load_telescope_runs.Command.handle() now passes start_time_tolerance=_START_TIME_MATCH_TOLERANCE
  (timedelta(minutes=5); ~150x the observed ~2s drift, ~3 orders of magnitude below the ~24h
  inter-night spacing so it can never merge distinct nights). A window rather than
  rounding/truncation was chosen because any fixed bucket still splits an event that drifts
  across a bucket boundary. Note: end_time / description dark-window times still drift within
  `fields`, so a cross-session re-ingest of a full-night event may report 'updated' (not
  'unchanged') -- acceptable, as the reported bug (silent DUPLICATE creation) is eliminated.
verification: |
  Root-cause mechanism confirmed by standalone astropy experiment (Evidence: determinism +
  UT1-UTC sensitivity). Fix verified by tests (python manage.py test
  solsys_code.tests.test_calendar_utils solsys_code.tests.test_load_telescope_runs -- 23 passed):
  new integration test test_reingest_with_drifted_sun_event_does_not_duplicate reproduces the
  exact dev-DB scenario (Magellan-Baade IMACS 17-18 July re-ingested with sun-event times shifted
  +2s) and asserts count stays 2, same pks, 'created: 0' -- i.e. no duplicate; new unit tests in
  test_calendar_utils cover within-tolerance unchanged/updated, minute-boundary straddle,
  start_time pinning, distinct-night (outside window) still creates, and other-lookup-key scoping;
  pre-existing idempotent/unchanged-rerun tests still pass (no regression). Paired demo notebook
  load_telescope_runs_demo.ipynb regenerated (jupyter nbconvert --execute) with a new cell showing
  a +2s-drifted re-ingest reporting `created: 0` (count 19 -> 19, nights matched and updated).
  Quality gates: `ruff check` + `ruff format --check` pass on all changed Python files and the
  notebook (repo-wide format failures in sync_lco notebook / settings.py are pre-existing, not
  touched by this fix). URL-keyed sync callers unaffected (verified by test_calendar_utils
  URL-lookup tests).
files_changed:
  - solsys_code/calendar_utils.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_load_telescope_runs.py
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
