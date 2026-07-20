---
status: resolved
trigger: "Resolving CampaignRun pk=32 (site_raw='Benedetto', obscode 434) via the Sites Needing Review row's Resolve button correctly replaces the placeholder Observatory with the real one, but the calendar projection then fails with a traceback in the runserver log: horizon_dip() at telescope_runs.py:154 raises ValueError('altitude_m must be a non-negative number, got -18.834226888866702') from sun_event() called by _project_calendar_event() at campaign_views.py:423, itself called from _resolve_site() at campaign_views.py:667."
created: 2026-07-16T16:00:00Z
updated: 2026-07-16T16:35:00Z
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: |
  Root cause confirmed (see Evidence). Three questions answered:
  Q1: The runserver traceback is EXPECTED — _resolve_site()'s projection
  try/except (campaign_views.py:666-675) catches `except Exception` (covers
  ValueError) and logs via logger.exception() (full traceback in log). Not a
  500; behaves as CR-01's graceful non-reverting path.
  Q2: -18.834226888866702 m is NOT a from_parallax_constants() bug. The erfa
  gc2gde conversion is exact; the negative value is an unavoidable artifact of
  MPC's 5-decimal parallax constants (1 LSB = 63.78 m altitude granularity) for
  the genuinely near-sea-level real site obscode 434 "S. Benedetto Po" (~19 m
  real elevation, lat 45.052 N, lon 10.92 E). Real altitudes of 0/10/19/24 m all
  round-trip to -18.83 m through the published constants.
  Q3: horizon_dip()'s non-negative guard IS too strict. The dip formula
  1.76'*sqrt(h) = sqrt(2h/R) physically applies only to an observer elevated
  above the horizon plane (h>0); at/below sea level the dip is 0. The guard was
  defensive input-validation (Stage 1 tests), not a physically-motivated
  rejection of below-sea-level sites.
test: |
  Fix horizon_dip() to clamp altitude<=0 to a 0 dip (keep the None guard).
  Verify obscode-434 site resolves end-to-end and add a regression test.
expecting: |
  With dip=0 for the -18.8 m site, sun_event(kind='sun') uses threshold -0.833
  deg, finds 2 crossings, projection succeeds, site_needs_review clears.
next_action: |
  Apply the horizon_dip() clamp fix, update the contradicting negative-altitude
  test, add a regression test for the below-sea-level path, and verify
  sun_event succeeds for obscode 434.

reasoning_checkpoint:
  hypothesis: |
    horizon_dip() raises ValueError for the real MPC site obscode 434 because
    its parallax-constant-derived geodetic altitude is a small negative number
    (-18.83 m) — physically normal for a near-sea-level site given MPC's ~64 m
    parallax-constant granularity — but horizon_dip()'s non-negative guard
    treats any negative altitude as invalid. This blocks calendar projection
    permanently, leaving the resolved run stuck in Sites Needing Review.
  confirming_evidence:
    - "DB Observatory pk=16 obscode 434 has altitude=-18.834226888866702, exactly the traceback value; timezone='Europe/Rome' is set, so altitude is the sole blocker."
    - "erfa gc2gde reproduces -18.83 m exactly from the live MPC constants (0.70765, 0.70419); resolves to San Benedetto Po (lat 45.052, lon 10.92)."
    - "Round-trip: real altitudes 0-24 m all yield the published 5-decimal constants and convert back to -18.83 m (1 LSB = 63.78 m), so the value is precision granularity, not a conversion bug."
    - "Design doc + formula show dip = sqrt(2h/R) applies only for elevated observers (h>0); at/below sea level the physical dip is 0."
  falsification_test: |
    If, after clamping altitude<=0 to a 0 dip, sun_event(site_434, window_start,
    kind='sun') still raised (or produced != 2 crossings), the hypothesis would
    be wrong — the failure would lie elsewhere (e.g. timezone, coordinates, or
    the crossing solver), not the altitude guard.
  fix_rationale: |
    Clamping altitude<=0 to a 0 dip addresses the ROOT CAUSE (the guard is too
    strict for physically-normal near/below-sea-level MPC sites) rather than the
    symptom. dip=0 is the physically correct value at/below sea level, and for a
    near-sea-level site the lost sub-arcminute dip correction is far inside the
    <=2 min accuracy tolerance. The None guard is retained because an unset
    altitude is a genuine data error, not a physical location.
  blind_spots: |
    Not tested: whether any downstream consumer of horizon_dip() relies on it
    raising for negatives (grep shows only sun_event calls it). Not tested:
    genuine deep-below-sea-level sites (e.g. Dead Sea ~-430 m) — but dip=0 is
    still physically correct there. The two Stage-1 tests encoding the old
    reject-negative contract must be updated to the new contract.

---

## Symptoms

expected: |
  Resolving a Sites Needing Review row's placeholder site to a real, genuine MPC observatory
  (obscode 434, 'Benedetto') via the Resolve button should either (a) succeed end-to-end,
  clearing site_needs_review and creating a CalendarEvent, or (b) fail gracefully per CR-01's
  designed non-reverting path (site replaced, site_needs_review stays True, row stays
  actionable, a clear warning message shown, no crash) -- exactly as it does for a
  Tier-2-resolved site missing other required data. Either outcome should never produce an
  unhandled server-side exception.
actual: |
  The site WAS replaced (the placeholder Observatory for obscode 434 is now the real
  Observatory record), and the row still shows a Resolve button (consistent with
  site_needs_review still True) -- so behaviorally this LOOKS like CR-01's graceful-failure
  path. But the runserver log shows a full Python traceback for a ValueError raised deep in
  horizon_dip(), which needs to be confirmed as caught-and-logged (expected) rather than an
  actual unhandled 500.
errors: |
  Traceback (most recent call last):
    File "/home/tlister/git/fomo_devel/solsys_code/campaign_views.py", line 667, in _resolve_site
      created = _project_calendar_event(run)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/tlister/git/fomo_devel/solsys_code/campaign_views.py", line 423, in _project_calendar_event
      sunset, sunrise = sun_event(run.site, run.window_start, kind='sun')
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/tlister/git/fomo_devel/solsys_code/telescope_runs.py", line 264, in sun_event
      dip = horizon_dip(site.altitude)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/tlister/git/fomo_devel/solsys_code/telescope_runs.py", line 154, in horizon_dip
      raise ValueError(f'altitude_m must be a non-negative number, got {altitude_m!r}')
  ValueError: altitude_m must be a non-negative number, got -18.834226888866702
timeline: |
  Found via manual browser testing immediately after quick task 260716-js7 (confirm-before-
  approve guard) was verified working for a NEW CampaignRun (pk=33). This is on pk=32,
  the SAME run used earlier this session to confirm the duplicate-suggestion fix ('Benedetto',
  obscode 434) -- so obscode 434's Observatory row may have been created moments ago via the
  Tier-2 resolve path during that earlier testing, and this is the first time its site has
  actually been RESOLVED (replacing the DCT-style placeholder) and gone through calendar
  projection. Quick task 260716-h8c (2026-07-16, earlier today) added timezone backfill via
  timezonefinder for Tier-2-resolved sites -- this is a DIFFERENT failure (altitude, not
  timezone), so h8c's fix is not directly implicated, though the SAME to_observatory()
  code path (parallax-constant-derived lat/lon/altitude) is where this new site's altitude
  was computed.
reproduction: |
  1. Go to the staff approval queue's Sites Needing Review section.
  2. Find the row for CampaignRun pk=32 (site_raw='Benedetto', obscode 434), whose site is
     (or was, before this test) a placeholder Observatory.
  3. Type/select the real 'Benedetto' MPC candidate (obscode 434) and click Resolve.
  4. Observe: site replaces to the real Observatory, but the runserver log shows the
     ValueError traceback above; the row still shows Resolve (implying site_needs_review
     stayed True).

## Evidence

- timestamp: 2026-07-16T16:20:00Z
  checked: |
    _resolve_site() at campaign_views.py:662-675 — the try/except wrapping the
    _project_calendar_event(run) call at line 667.
  found: |
    The except clause is `except Exception:` (line 668), which DOES catch
    ValueError. It calls logger.exception(...) (line 669) — that is what emits
    the full traceback into the runserver log — then messages.warning(...) and
    redirect(...). It does NOT re-raise.
  implication: |
    Q1 ANSWERED: the traceback is the EXPECTED caught-and-logged output of
    logger.exception(), not an unhandled 500. Behaviorally matches CR-01's
    non-reverting graceful-failure contract: site stays replaced,
    site_needs_review stays True (flag only clears at line 678 AFTER a
    non-raising projection), row stays actionable, warning shown. NOT a crash.
    BUT the site can never be successfully resolved while horizon_dip() rejects
    its altitude — it is permanently stuck in Sites Needing Review. That is the
    real (non-500) bug.

- timestamp: 2026-07-16T16:22:00Z
  checked: |
    Live MPC obscodes API record for obscode 434 (GET
    https://data.minorplanetcenter.net/api/obscodes).
  found: |
    obscode 434 = "S. Benedetto Po" (San Benedetto Po, Italy), a real optical
    MPC observatory. longitude=10.9206, rhocosphi=0.70765, rhosinphi=0.70419
    (both only 5 decimal places). observations_type=optical.
  implication: |
    Genuine near-sea-level real site (San Benedetto Po sits in the Po river
    valley at ~19 m real elevation). Not a garbage/test record.

- timestamp: 2026-07-16T16:24:00Z
  checked: |
    Reproduced Observatory.from_parallax_constants(10.9206, 0.70765, 0.70419)
    using the exact erfa.eform(1)/erfa.gc2gde() math from models.py.
  found: |
    Output: lat=45.05201 deg, lon=10.9206 deg, altitude=-18.834226888866702 m —
    byte-for-byte the value in the traceback. Coordinates are exactly San
    Benedetto Po. 1 LSB of a 5-decimal parallax constant = _r*1e-5 = 63.78 m of
    altitude. Round-trip check: real altitudes of 0/10/19/24 m ALL produce the
    published (0.70765, 0.70419) after rounding to 5 decimals, and all convert
    back to -18.83 m.
  implication: |
    Q2 ANSWERED: from_parallax_constants() is CORRECT — no sign error, no
    reference-surface mismatch. The negative altitude is an unavoidable artifact
    of MPC's ~64 m parallax-constant granularity for a near-sea-level site. The
    conversion cannot distinguish +19 m from -18.8 m here.

- timestamp: 2026-07-16T16:26:00Z
  checked: |
    horizon_dip() (telescope_runs.py:134-155), its tests
    (test_telescope_runs.py:115-124), and the design doc
    (docs/design/telescope_runs_calendar.rst:106-118).
  found: |
    Formula: dip = 1.76' * sqrt(h), derived from spherical geometry
    theta ~ sqrt(2h/R) with refraction folded in — valid only for an observer
    elevated above the horizon plane (h>0). Guard: raises ValueError if
    altitude_m is None or < 0. Tests test_horizon_dip_raises_on_negative_altitude
    (horizon_dip(-10)) and test_horizon_dip_raises_on_none_altitude encode that
    guard. Design doc discusses only positive elevations (dip depresses the
    visible horizon for an ELEVATED observer); says nothing about below-sea-level
    handling.
  implication: |
    Q3 ANSWERED: the non-negative guard is too strict for real MPC-derived
    sites. Physically the dip at/below sea level is 0 (the sqrt(2h/R) model has
    no elevated horizon to depress). The guard was defensive input validation,
    not a physical rejection of below-sea-level sites. Fix: clamp altitude<=0 to
    dip=0; keep the None guard (unset altitude is a genuine data error).

## Eliminated

- hypothesis: |
    The ValueError propagates as an unhandled 500 that crashes the resolve_site
    request.
  evidence: |
    _resolve_site()'s projection call is wrapped in try/except Exception
    (campaign_views.py:666-675) which catches ValueError, logs it via
    logger.exception (source of the traceback), warns the user, and redirects.
    No re-raise. The request returns a normal 302 redirect.
  timestamp: 2026-07-16T16:20:00Z

- hypothesis: |
    from_parallax_constants() has a sign error or reference-surface mismatch
    producing a nonsensical negative altitude for obscode 434.
  evidence: |
    The erfa gc2gde conversion reproduces -18.834226888866702 m exactly and
    resolves lat/lon precisely to San Benedetto Po. Round-tripping real
    near-sea-level altitudes (0-24 m) through 5-decimal-rounded parallax
    constants reproduces both the published constants and the -18.83 m output.
    The math is correct; the negative value is MPC precision granularity, not a
    bug.
  timestamp: 2026-07-16T16:24:00Z


## Resolution

root_cause: |
  horizon_dip() (telescope_runs.py) rejected any negative altitude with
  ValueError, but the parallax-constant-to-geodetic conversion in
  Observatory.from_parallax_constants() legitimately produces a small NEGATIVE
  geodetic height (-18.834226888866702 m) for the real, near-sea-level MPC site
  obscode 434 "S. Benedetto Po" (~19 m real elevation). This is not a conversion
  bug: MPC publishes parallax constants to only 5 decimal places (1 LSB = 63.78 m
  of altitude), so a genuine near-sea-level site cannot be distinguished from a
  slightly-below-ellipsoid one. When _resolve_site() replaced the placeholder
  Observatory with this real record and called _project_calendar_event() ->
  sun_event(kind='sun') -> horizon_dip(site.altitude), the guard raised. The
  raise was CORRECTLY caught by _resolve_site()'s non-reverting try/except
  Exception (CR-01) and logged via logger.exception (that is the traceback in
  the runserver log -- NOT an unhandled 500), leaving the row in Sites Needing
  Review. But because every retry hit the same guard, the site could never be
  successfully projected: permanently stuck.
fix: |
  Relaxed horizon_dip() to treat any altitude at or below sea level (<= 0) as a
  0 dip instead of raising. This is physically correct -- the Nautical Almanac
  dip model dip = 1.76'*sqrt(h) = sqrt(2h/R) only describes horizon depression
  for an observer ELEVATED above the reference surface; at/below sea level there
  is no elevated horizon to depress, so dip = 0. The None guard is retained (an
  unset altitude is a genuine data error, not a physical location). For a near-
  sea-level site the discarded sub-arcminute dip correction is far inside the
  <=2 min sun-event accuracy tolerance.
verification: |
  - Full telescope_runs suite: 33/33 pass, including new
    test_horizon_dip_zero_at_sea_level, test_horizon_dip_zero_for_below_sea_
    level_altitude, and end-to-end test_sun_event_succeeds_for_below_sea_level_
    site (creates an obscode-434-shaped record and asserts sun_event no longer
    raises).
  - Live check against the real stuck DB record (Observatory pk=16, obscode 434,
    altitude=-18.834226888866702, tz Europe/Rome): horizon_dip -> 0.0 arcmin;
    sun_event(kind='sun', 2026-06-10) -> sunset 19:02:27 UTC / sunrise 03:29:07
    UTC (21:02 / 05:29 local, correct for northern Italy in June). No raise.
  - Campaign approval suite (exercises resolve_site + projection): 94/94 pass.
  - ruff check + ruff format --check: clean on both changed files.
  - HUMAN CONFIRMED (2026-07-16): re-clicking Resolve on CampaignRun pk=32
    ("Benedetto"/434) in the browser now resolves correctly -- the row leaves
    Sites Needing Review and no traceback appears in the runserver log.
files_changed:
  - solsys_code/telescope_runs.py (horizon_dip: clamp altitude<=0 to 0 dip; keep None guard; docstring)
  - solsys_code/tests/test_telescope_runs.py (replace reject-negative test with sea-level/below-sea-level dip tests; add end-to-end sun_event regression for a below-sea-level site)
