---
phase: 01-site-ephemeris-helper
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - solsys_code/telescope_runs.py
  - solsys_code/solsys_code_observatory/migrations/0002_observatory_timezone_seed.py
  - solsys_code/solsys_code_observatory/models.py
  - solsys_code/tests/test_telescope_runs.py
findings:
  critical: 0
  warning: 4
  info: 2
  total: 6
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-06-12T00:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed the new `telescope_runs.py` helper module, the supporting `Observatory`
model changes (new `timezone` field, `to_earth_location()`), the
`0002_observatory_timezone_seed.py` data migration, and the test suite. The
core sun-event algorithm (coarse scan + bisection refine) is sound and the
test suite is thorough for the four seeded sites. No critical/security
issues were found.

The main concerns are robustness gaps: `sun_event()` assumes `_find_crossing`
always returns exactly the two crossings it expects in chronological
(set, then rise) order, with no validation or error handling for the case
where it doesn't (e.g. a missing or malformed `timezone` value, or an
altitude that makes `horizon_dip()` fail). These wouldn't be hit by the
4 seeded sites under normal conditions, but the lack of guardrails means a
future bad observatory record or a site near the search-window edge will
surface as an unhandled `IndexError` / `ValueError` rather than a clear
domain error. Also flagging a data-migration robustness concern and two
minor doc/comment gaps.

## Warnings

### WR-01: `sun_event()` assumes exactly 2 crossings without validation

**File:** `solsys_code/telescope_runs.py:159-160`
**Issue:** `_find_crossing()` returns a `list[Time]` whose length depends on
how many sign changes occur in the 24-hour coarse scan. `sun_event()`
unconditionally unpacks `crossings[0]` and `crossings[1]` as
`(setting, rising)`. If the scan finds 0, 1, or >2 crossings — e.g. due to
a malformed `Observatory.altitude`/`lat`/`lon`, or a site/date combination
near a search-window edge case — this either raises an unhandled
`IndexError` (too few) or silently returns the wrong pair of events (too
many), with no diagnostic message tying the failure back to `site`/`date`/`kind`.
**Fix:**
```python
crossings = _find_crossing(anchor, location, threshold, search_hours=24)
if len(crossings) != 2:
    raise ValueError(
        f'Expected 2 sun-event crossings for {site.short_name} on {date} '
        f"(kind={kind!r}), got {len(crossings)}: {crossings}"
    )
return crossings[0], crossings[1]
```

### WR-02: `horizon_dip()` will raise on negative altitude

**File:** `solsys_code/telescope_runs.py:54`
**Issue:** `horizon_dip()` computes `sqrt(altitude_m)`, which raises
`ValueError: math domain error` for negative `altitude_m`. The `Observatory`
model's `altitude` field (`solsys_code/solsys_code_observatory/models.py:54`)
is a nullable `FloatField` with no minimum-value validator, so a future
below-sea-level or unset (`None`) observatory record passed through
`sun_event(site, date, 'sun')` would crash with an unhelpful generic
`TypeError`/`ValueError` rather than a clear domain error pointing at the
bad `altitude` value. None of the 4 seeded sites trigger this today, but
there's no guard at the boundary.
**Fix:**
```python
def horizon_dip(altitude_m: float) -> float:
    if altitude_m is None or altitude_m < 0:
        raise ValueError(f'altitude_m must be a non-negative number, got {altitude_m!r}')
    dip_arcmin = 1.76 * sqrt(altitude_m)
    return dip_arcmin / 60.0
```

### WR-03: `_local_noon_utc()` will raise an opaque error if `site.timezone` is unset

**File:** `solsys_code/telescope_runs.py:125-127`
**Issue:** `Observatory.timezone` defaults to `''` (`blank=True, default=''`,
see `solsys_code_observatory/models.py:55`). If `get_site()` returns a record
where `timezone` was never populated (e.g. a future site added without
updating the seed data, or via the `CreateObservatory` form which doesn't
require this field), `ZoneInfo('')` raises
`zoneinfo.ZoneInfoNotFoundError` (or similar) deep inside `_local_noon_utc`,
with no indication that the root cause is a missing `Observatory.timezone`
value. `sun_event`'s docstring only documents a `ValueError` for bad `kind`.
**Fix:** Validate at the top of `sun_event` (or in `_local_noon_utc`):
```python
if not site.timezone:
    raise ValueError(f'Observatory {site.short_name!r} (obscode={site.obscode}) has no timezone set')
```

### WR-04: Seed migration's `update_or_create` can collide with `name`'s unique constraint

**File:** `solsys_code/solsys_code_observatory/migrations/0002_observatory_timezone_seed.py:46-48`
**Issue:** `Observatory.name` has `unique=True`
(`solsys_code_observatory/models.py:30-32`). The seed migration does
`Observatory.objects.update_or_create(obscode=obscode, defaults=rec)` where
`rec` includes `name=...`. If a database already has an `Observatory` row
with `obscode='268'` but a different `name` (e.g. manually entered before
this migration runs), `update_or_create` will try to set `name='Magellan
Clay Telescope'` on that row — and if a *different* row already has that
exact `name` value, the update raises `IntegrityError` and the migration
fails partway through (with 3 of 4 records potentially already
created/updated, since each `update_or_create` call commits independently
outside an explicit atomic block consideration). This is a low-probability
scenario on fresh dev/test DBs (which is why the test suite passes) but is
a real risk for any existing production database that already has rows for
these MPC obscodes under different names.
**Fix:** Either wrap in `try/except IntegrityError` with a `logger.warning`
and `continue`, or document in the migration docstring/commit message that
this migration is only safe to run against databases without pre-existing
records for obscodes 268/269/809/E10.

## Info

### IN-01: `_find_crossing`'s 24-hour search window is actually 1439 minutes wide

**File:** `solsys_code/telescope_runs.py:95-99`
**Issue:** `offsets = np.arange(0, search_hours * 60, coarse_step_min)` with
`search_hours=24, coarse_step_min=1.0` produces minute offsets `0..1439`
(1440 points), so the last coarse sample is at `anchor + 1439 min`, not
`anchor + 1440 min` (24h). The crossing-detection loop
(`for i in range(len(alt) - 1)`) therefore only checks pairs up to
`(1438, 1439)`, leaving a ~1-minute blind spot at the very end of the
nominal 24-hour window. The docstring's claim that "scanning forward
search_hours=24 guarantees both ... crossings fall within the window" is
true for the 4 seeded sites in June (comfortably inside the window), but
the off-by-one means the guarantee is technically for `[anchor, anchor +
1439 min)` rather than the full 24h stated.
**Fix:** Either document the actual window precisely, or extend the range
slightly, e.g. `np.arange(0, search_hours * 60 + coarse_step_min, coarse_step_min)`.

### IN-02: Magic number `0.833` lacks an inline comment

**File:** `solsys_code/telescope_runs.py:153`
**Issue:** `threshold = -(0.833 + dip)` uses the standard solar
semi-diameter + atmospheric refraction correction (≈34 arcmin) at the
horizon, but this value isn't explained inline. Per this project's comment
conventions ("Comment constants and their meaning"), other magic numbers in
the module (e.g. `1.76` in `horizon_dip`, `-15.0` dark threshold) are
explained via docstrings/formula comments, but `0.833` is bare.
**Fix:**
```python
if kind == 'sun':
    dip = horizon_dip(site.altitude)
    # 0.833 deg = standard solar semi-diameter (~16') + horizon refraction (~34')
    threshold = -(0.833 + dip)
```

---

_Reviewed: 2026-06-12T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
