---
phase: quick-260913-rmd
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/allocation_projector.py
  - solsys_code/models.py
  - solsys_code/tests/test_allocation_projector.py
  - docs/runbooks/telescope_runs_calendar.rst  # conditional -- only if Task 3's audit finds the old two-case rule stated there; planning-time grep found no such statement, so the expected outcome is "no change, audit recorded in SUMMARY"
autonomous: true
requirements:
  - ALLOC-02
  - NF-03

estimate:
  tokens: 55000
  raw_tokens: 55000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "Band 1 (site offset > +6, Australia/Sydney +10, night entirely inside its own UTC date): every stored sub-night time-of-day resolves onto the night's own date -- unchanged from today, proven by the two existing TestSubNightWindowSiteDirection Sydney tests staying green unmodified"
    - "Band 2 west (-6 < offset <= +6, America/Santiago -4, night straddles UTC midnight): an evening-side time resolves onto the night, a morning-side time onto night + 1 day -- unchanged from today, proven by the existing Chile tests staying green unmodified"
    - "Band 2 east (Africa/Johannesburg +2): a morning-side night_end_utc of 03:00 on night 2026-07-09 resolves to 2026-07-10T03:00:00+00:00, giving a non-inverted span against the site's own computed sunset start -- where today the same run raises ValueError on every single reconcile"
    - "Band 2 half-hour (Asia/Kolkata +5:30): a 00:00-02:00 window on night 2026-07-09 resolves to 2026-07-10T00:00:00+00:00 and 2026-07-10T02:00:00+00:00 -- where today both land on 2026-07-09, a full day early, with no error raised at all"
    - "Band 3 (offset <= -6, Pacific/Honolulu -10, night entirely inside the NEXT UTC date): a 13:00-15:00 window on night 2026-07-09 resolves to 2026-07-10T13:00:00+00:00 and 2026-07-10T15:00:00+00:00, and an evening-side 04:30 also resolves onto night + 1 -- where today the late-hour values land on 2026-07-09, a full day early, with no error raised"
    - "Each end is still resolved independently: a run naming only night_end_utc keeps its start at the site's own sun_event() sunset, and a run naming only night_start_utc keeps its end at sunrise"
    - "night_bounds()'s inverted-span ValueError guard survives byte-identical in message and behaviour -- the existing Sydney 19:00/08:00 guard test still raises and still writes no event"
    - "The resolution stays astropy-free: a second reconcile of a band-3 run with both sub-night fields set reports unchanged with no sun_event() call and the same event primary key, so _span_needs_remint() and the mint path agree and no night re-mints forever"
    - "The changed functions' docstrings and the models.py night_start_utc/night_end_utc field-contract comment both state the three-band taxonomy; neither asserts the superseded two-case rule as durable contract"
    - "python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler passes in full, and pre-commit's pinned ruff and ruff-format are clean on every changed Python file"
  artifacts:
    - solsys_code/allocation_projector.py
    - solsys_code/models.py
    - solsys_code/tests/test_allocation_projector.py
  key_links:
    - "The property the rule needs is WHERE the site's observing night sits relative to UTC midnight, not the SIGN of its UTC offset. A local night runs roughly local 18:00 -> local 06:00, i.e. (18 - offset) -> (30 - offset) in UTC: entirely inside its own UTC date only when offset > +6, straddling UTC midnight for -6 < offset <= +6, entirely inside the NEXT UTC date when offset <= -6. Three bands, not two"
    - "Both boundaries are computed from ONE per-night UTC span (the site's own local 18:00 and the local 06:00 twelve hours later, converted with zoneinfo), so start and end can never be resolved against two different spans"
    - "D-13 forbids any sun_event() call on _span_needs_remint()'s update path -- the span is a zoneinfo conversion only, never an astropy solar scan, which is exactly why the site's nominal local 18:00 is used rather than its true sunset"
    - "The 12:00 UTC hour threshold disappears entirely: a candidate date is chosen by asking which one lands inside the night's own UTC span, which is why the half-hour Asia/Kolkata offset needs no special case"
    - "night_bounds()'s ValueError guard is a correct backstop and stays: it is what keeps a genuinely inverted operator input (or a site this rule still mis-serves) out of the shared calendar. The fix is that it stops firing for band-2-east and band-3 sites, not that it is removed"
    - "Six Observatory records with affected offsets already exist in this project's database: K92/K93 Sutherland-LCO (+2), C65 Montsec (+2), N50 Hanle (+5:30), 500 Geocentric (0), 568 Maunakea (-10), F65 FTN (-10). The defect is latent only because migration 0018_campaignrun_night_window_fields is unapplied in the dev DB and the fields are set only by a classical schedule line naming a partial night"
---

<objective>
Close 35-REVIEW.md NF-03. `allocation_projector._site_runs_behind_utc()` keys the whole
sub-night date-offset rule on the SIGN of the site's UTC offset, and
`_time_of_day_to_datetime()` then picks a UTC calendar date with a hard-coded `t.hour < 12`
threshold. The sign of the offset is not the property the rule needs, and there are three
bands, not two -- so every site with an offset of `0..+6` (SAAO Sutherland, Montsec, Hanle,
Geocentric) and every site with an offset at or below `-6` (Maunakea, FTN) resolves at least
one boundary onto the wrong UTC date. Two failure modes follow: a LOUD one, where the two
ends land on opposite sides, `night_bounds()` raises `ValueError` on every reconcile and the
run is counted `failed` forever; and a SILENT one, where both ends land wrong together, no
error is raised, and a `CalendarEvent` is written a full day early onto the shared calendar.

Replace the boolean with a per-boundary date resolution against the site's own observing-night
UTC span, correct the docstrings and the `models.py` field-contract comment that currently
assert the false two-case taxonomy as durable contract, and pin all three bands with tests.

Purpose: `Observatory.timezone` is a free-form field populated from the MPC Observatory Codes
API, so an affected site is one admin action away with no code change -- and six such records
already exist here. The existing tests pass only because the two fixture sites (Chile,
Sydney) are precisely the two the broken rule happens to fit.

Output: a three-band resolution in `allocation_projector.py`, a corrected durable field
contract in `models.py`, and band coverage for +10, -4, +2, +5:30 and -10 sites.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@solsys_code/allocation_projector.py
@solsys_code/models.py
@solsys_code/tests/test_allocation_projector.py
</context>

<ground_truth>
The orchestrator diagnosed this finding and verified the arithmetic empirically; the planner
re-verified every resolution below with a standalone `zoneinfo` probe at planning time. These
values are ground truth -- do NOT re-derive them, and do NOT adopt the review's own "Fix:"
block, which proposes a `_night_crosses_utc_midnight()` predicate that is still TWO-way and
is strictly WORSE than the current code for band 3 (at Pacific/Honolulu both ends of the
night land on the same UTC date, so it returns False, and a loud `ValueError` becomes silent
day-early corruption).

The site's observing-night UTC span is its own local 18:00 through the local 06:00 twelve
hours later. Verified spans and resolutions (`night` is the site-local observing night, the
evening date):

| Site zone | Offset | Night | Span (UTC) | Stored time | Resolves to |
|-----------|--------|-------|-----------|-------------|-------------|
| Australia/Sydney | +10 | 2026-08-01 | 08:00 .. 20:00 same date | 09:30 | 2026-08-01T09:30Z |
| Australia/Sydney | +10 | 2026-08-01 | 08:00 .. 20:00 same date | 19:00 | 2026-08-01T19:00Z |
| Australia/Sydney | +10 | 2026-08-01 | 08:00 .. 20:00 same date | 08:00 | 2026-08-01T08:00Z |
| America/Santiago | -4 | 2026-07-09 | 22:00 .. next 10:00 | 23:30 | 2026-07-09T23:30Z |
| America/Santiago | -4 | 2026-07-09 | 22:00 .. next 10:00 | 06:26 | 2026-07-10T06:26Z |
| Africa/Johannesburg | +2 | 2026-07-09 | 16:00 .. next 04:00 | 03:00 | 2026-07-10T03:00Z |
| Asia/Kolkata | +5:30 | 2026-07-09 | 12:30 .. next 00:30 | 00:00 | 2026-07-10T00:00Z |
| Asia/Kolkata | +5:30 | 2026-07-09 | 12:30 .. next 00:30 | 02:00 | 2026-07-10T02:00Z |
| Pacific/Honolulu | -10 | 2026-07-09 | next 04:00 .. next 16:00 | 04:30 | 2026-07-10T04:30Z |
| Pacific/Honolulu | -10 | 2026-07-09 | next 04:00 .. next 16:00 | 13:00 | 2026-07-10T13:00Z |
| Pacific/Honolulu | -10 | 2026-07-09 | next 04:00 .. next 16:00 | 15:00 | 2026-07-10T15:00Z |

Note the last Asia/Kolkata row and the Sydney 08:00 row: 02:00 on the following date falls
OUTSIDE that span (which ends 00:30), and 08:00 sits exactly ON the Sydney span's start. The
resolution therefore needs a defined answer for a candidate that is not inside the span and
for a candidate exactly on a boundary -- see Task 1's rule.
</ground_truth>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Resolve each sub-night boundary against the site's own observing-night UTC span</name>
  <files>solsys_code/allocation_projector.py, solsys_code/models.py, solsys_code/tests/test_allocation_projector.py</files>
  <reversibility rating="reversible">Pure date arithmetic inside one module; no migration, no stored data rewritten. Reverting restores the prior behaviour exactly.</reversibility>
  <behavior>
    The end-to-end slice this task proves (one new test, `Africa/Johannesburg`, the band the
    review reproduced as a hard failure):
    - Create an `Observatory` for SAAO Sutherland (obscode `K92`, short_name `LSC-SAAO`,
      lat -32.3808, lon 20.8101, altitude 1804, timezone `Africa/Johannesburg`,
      observations_type `Observatory.OPTICAL_OBSTYPE`) in the shared test base alongside the
      two existing fixture sites.
    - A one-night `CampaignRun` on that site, `window_start = window_end = date(2026, 7, 9)`,
      `night_end_utc = time(3, 0)`, `night_start_utc` left null.
    - `reconcile_run(run)` must NOT raise. The resulting `ALLOC:{run.pk}:2026-07-09` event
      must have `end_time == datetime(2026, 7, 10, 3, 0, 0, tzinfo=dt_timezone.utc)` and
      `start_time` equal to the site's own `sun_event(site, night, kind='sun')` sunset
      rounded to seconds, asserted the same way the existing
      `test_set_end_and_null_start_computes_sunset_start_and_next_morning_end` asserts it,
      with `start_time < end_time`.
    - Today this exact fixture raises `ValueError` with an inverted span, so the test is a
      genuine RED before the module change and GREEN after it.
  </behavior>
  <action>
Rewrite the sub-night date rule in `solsys_code/allocation_projector.py` as a per-boundary
resolution against the site's own observing-night UTC span, replacing the offset-sign boolean
entirely.

1. Replace `_site_runs_behind_utc(run, night) -> bool` with a private helper
   `_night_span_utc(run, night) -> tuple[datetime, datetime]`. It builds
   `ZoneInfo(run.site.timezone)`, forms the site's nominal local evening as a `datetime` at
   hour 18 on `night` carrying that zone, and returns that instant converted to UTC together
   with the instant twelve hours later on the local wall clock (add `timedelta(hours=12)` to
   the zone-carrying local datetime BEFORE converting, so a DST shift inside the night is
   applied by the conversion). Both returned values are UTC-aware. This is a `zoneinfo`
   lookup only -- state in the docstring that it must never call `sun_event()`, because D-13
   forbids any astropy work on `_span_needs_remint()`'s update path, and that using the
   site's nominal local 18:00 rather than its true sunset is exactly what buys that.

2. Change `_time_of_day_to_datetime(t, night, ...)`'s third parameter from the boolean to the
   span tuple (name it `night_span`). Build the two candidate UTC datetimes for `t` -- one on
   `night`, one on `night + timedelta(days=1)`, each at `t.hour`/`t.minute`/`t.second` with
   `tzinfo=dt_timezone.utc` -- and pick by distance to the span: distance is zero when the
   candidate lies within the span inclusive of both endpoints, otherwise the smaller of its
   distances to the two span endpoints. Return the candidate with the smaller distance; on an
   exact tie return the one on `night`. Implement the choice so the tie rule is a property of
   the code rather than an accident (for example `min(candidates, key=...)` over a list
   ordered `night` first, which returns the first minimum). Delete the hour-based threshold
   expression; there is no hour comparison anywhere in the new body.

3. Update both call sites to compute the span once and pass it through: `night_bounds()`
   replaces its single boolean lookup with one `_night_span_utc(run, night)` call used for
   both ends, and `_span_needs_remint()` does the same. Keep both functions resolving their
   two ends independently -- the null-field branches, their sun-event fallbacks and the order
   of the checks are unchanged.

4. Keep `night_bounds()`'s inverted-span guard exactly as it is: same `logger.error(...)`
   call, same `ValueError` message text, same position after both ends are resolved. It is a
   correct backstop; the fix is that it stops firing for the two bands it was firing on
   wrongly, not that it is relaxed.

5. Rewrite the docstrings of `_night_span_utc()`, `_time_of_day_to_datetime()`,
   `night_bounds()` and `_span_needs_remint()` so they describe the three bands (night
   entirely inside its own UTC date for an offset above +6; night straddling UTC midnight for
   an offset above -6 and at or below +6, where an evening-side time belongs to the night and
   a morning-side time to the following date; night entirely inside the next UTC date for an
   offset at or below -6). Name a real site per band (Siding Spring +10; La Silla -4, SAAO
   Sutherland +2 and Hanle +5:30; Maunakea/FTN -10), cite `35-REVIEW.md NF-03`, and say
   plainly that the superseded rule keyed on the sign of the UTC offset and served only the
   two bands the fixture sites happened to occupy. No docstring may continue to present a
   two-case taxonomy as the durable rule.

6. In `solsys_code/models.py`, rewrite the `night_start_utc`/`night_end_utc` field-contract
   comment paragraph (the one that today points at the removed helper and asserts the
   two-case taxonomy, currently around lines 266-277) to state the same three bands, to name
   `_night_span_utc()`/`night_bounds()` as where the rule now lives, and to keep the
   surrounding paragraphs -- the null-means-use-the-computed-sun-event paragraph, the
   deliberately-no-null-together-constraint paragraph and the staff-edit paragraph -- byte
   identical. The comment must not name a 12:00 threshold as the rule.

7. Add the single band-2-east end-to-end test described in `<behavior>` to
   `solsys_code/tests/test_allocation_projector.py`: the new `Observatory` fixture goes in
   `AllocationProjectorTestBase.setUpTestData()` next to the two existing sites (follow their
   exact construction), and the test itself goes in the existing
   `TestSubNightWindowSiteDirection` class. Use `self._make_run(site=..., site_raw='K92', ...)`
   for the run, matching how the existing Sydney tests override the site.

Do not touch `project_allocation()`, `_mint_fields()`, `retired_nights()` or any receiver.
  </action>
  <verify>
    <automated>python manage.py test solsys_code.tests.test_allocation_projector 2>&1 | tail -20; grep -c 'def _night_span_utc' solsys_code/allocation_projector.py</automated>
  </verify>
  <done>`python manage.py test solsys_code.tests.test_allocation_projector` passes in full, including the four pre-existing `TestSubNightWindowSiteDirection` tests unmodified (Sydney early-hour, Sydney late-hour, Chile unchanged, Sydney inverted-guard raises) and the new SAAO Sutherland test asserting `end_time == 2026-07-10T03:00:00+00:00` with a sun-event start strictly before it. `grep -c 'def _night_span_utc' solsys_code/allocation_projector.py` reports 1. The `models.py` field-contract comment states three bands and points at the new helper.</done>
</task>

<task type="auto">
  <name>Task 2: Pin the remaining bands -- half-hour offset, the fully-next-date band, and re-mint agreement</name>
  <files>solsys_code/tests/test_allocation_projector.py</files>
  <action>
Extend `TestSubNightWindowSiteDirection` in `solsys_code/tests/test_allocation_projector.py`
with the remaining band coverage, and add the two `Observatory` fixtures it needs to
`AllocationProjectorTestBase.setUpTestData()` (same construction as the existing rows): IAO
Hanle -- obscode `N50`, short_name `HANLE`, lat 32.7794, lon 78.9642, altitude 4500, timezone
`Asia/Kolkata`; and Haleakala/FTN -- obscode `F65`, short_name `FTN`, lat 20.7069, lon
-156.2570, altitude 3055, timezone `Pacific/Honolulu`. Both with
`observations_type=Observatory.OPTICAL_OBSTYPE`.

Every test drives `reconcile_run(run)` and asserts the resolved UTC datetimes EXPLICITLY
against `CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')`. Asserting
"no exception raised" is not sufficient for any of these -- the failure they guard is silent.
Take every expected value from the plan's `<ground_truth>` table; do not recompute them.

Add these tests:

1. Half-hour offset, both ends silently a day early today (Hanle, `Asia/Kolkata`): night
   `date(2026, 7, 9)`, one-night window, `night_start_utc = time(0, 0)`,
   `night_end_utc = time(2, 0)`. Assert `start_time == datetime(2026, 7, 10, 0, 0, 0, tzinfo=dt_timezone.utc)`
   and `end_time == datetime(2026, 7, 10, 2, 0, 0, tzinfo=dt_timezone.utc)`. Name in the
   docstring that the end value falls outside the night's own span and is therefore resolved
   by the nearer-candidate rule, and that today both values land on 2026-07-09 with no error
   raised at all.

2. Offset at or below -6, morning-side pair silently a day early today (FTN,
   `Pacific/Honolulu`): night `date(2026, 7, 9)`, one-night window,
   `night_start_utc = time(13, 0)`, `night_end_utc = time(15, 0)`. Assert
   `start_time == datetime(2026, 7, 10, 13, 0, 0, tzinfo=dt_timezone.utc)` and
   `end_time == datetime(2026, 7, 10, 15, 0, 0, tzinfo=dt_timezone.utc)`. Docstring: this
   site's whole observing night lies inside the NEXT UTC date, so BOTH ends resolve onto
   `night + 1` -- the band the review's own suggested two-way predicate would have got wrong.

3. Offset at or below -6, evening-side end with a computed start (FTN): night
   `date(2026, 7, 9)`, one-night window, `night_end_utc = time(4, 30)` with
   `night_start_utc` left null. Assert `end_time == datetime(2026, 7, 10, 4, 30, 0, tzinfo=dt_timezone.utc)`
   and `start_time` equal to `sun_event(self.ftn_site, night, kind='sun')`'s sunset converted
   to UTC with `microsecond=0` -- proving the two ends are still resolved independently and
   that the evening-side value that was right by luck under the old rule is still right.

4. Re-mint agreement on the fully-next-date band (FTN): reconcile the test-2 run, capture the
   event primary key, then reconcile again inside
   `patch('solsys_code.allocation_projector.sun_event')` and assert the mock was not called,
   that the result reports `unchanged == 1` with `created == 0` and `retired == 0`, and that
   the event primary key is unchanged. This is what proves `_span_needs_remint()` resolves
   the same dates the mint path wrote, so a band-3 night does not re-mint on every sweep.

Follow the existing class's fixture and assertion style throughout; if a `Target` is ever
needed use `NonSiderealTargetFactory` (CLAUDE.md), never `SiderealTargetFactory`.
  </action>
  <verify>
    <automated>python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler 2>&1 | tail -20</automated>
  </verify>
  <done>All three test modules pass in full. `TestSubNightWindowSiteDirection` now covers +10, -4, +2, +5:30 and -10 sites with explicitly asserted UTC datetimes, plus the pre-existing inverted-span guard test, and a band-3 second reconcile makes no `sun_event()` call and reports `unchanged`.</done>
</task>

<task type="auto">
  <name>Task 3: Paired-docs audit and quality gates</name>
  <files>docs/runbooks/telescope_runs_calendar.rst</files>
  <action>
Run the CLAUDE.md paired-docs audit for a behaviour change in `allocation_projector.py`, and
record the OUTCOME of each step in the SUMMARY whether or not anything changed -- an audit
with no finding is a result, not a skipped step.

1. Runbook audit. Search `docs/runbooks/telescope_runs_calendar.rst` for any statement of the
   superseded two-case rule -- a 12:00 UTC threshold, a "clock runs behind/ahead of UTC"
   framing, or any claim that a site ahead of UTC maps its whole night onto a single UTC date.
   A planning-time search found no such statement (the runbook's sub-night material is about
   the schedule-line tokens and the identity key, not about date mapping), so the expected
   outcome is no edit. If the search does find one, correct it to the three-band rule in the
   same prose register as the surrounding sections and keep the edit minimal. Record in the
   SUMMARY which terms were searched and what was found.

2. Notebook audit. `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` and
   `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` are the notebooks whose
   subject matter touches this projector. Confirm, by inspecting their committed cells rather
   than by assumption, that every site they exercise is one of the two unaffected fixture
   sites (Chile / Sydney -- bands 1 and 2-west, whose resolutions this change leaves
   byte-identical) and that no committed cell records an output this change alters. If that
   holds, no re-execution is needed and the SUMMARY records that finding with the sites named.
   If any cell exercises an affected site or records an altered output, re-execute that
   notebook with `jupyter nbconvert --to notebook --execute --inplace` and commit it with
   output, per the repo convention.

3. Quality gates over every file actually changed by this task set. Run pre-commit's pinned
   ruff -- an unpinned `ruff` on PATH reports findings the enforced gate does not have (D-07).

Do not re-run the full unfiltered Django suite: `test_views.TestEphemeris` segfaults in native
ASSIST. Invoke the test runner as `python manage.py test`, never `./manage.py`.
  </action>
  <verify>
    <automated>pre-commit run ruff --files solsys_code/allocation_projector.py solsys_code/models.py solsys_code/tests/test_allocation_projector.py && pre-commit run ruff-format --files solsys_code/allocation_projector.py solsys_code/models.py solsys_code/tests/test_allocation_projector.py && python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler 2>&1 | tail -10</automated>
  </verify>
  <done>Both pre-commit hooks report clean on all changed Python files, and the three test modules pass. The SUMMARY records the runbook audit outcome (terms searched, what was found) and the notebook audit outcome (sites exercised, whether re-execution was needed) explicitly, either way.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| staff admin -> `CampaignRun.night_start_utc`/`night_end_utc` | A staff member may edit either `TimeField` directly; the next reconcile re-mints the affected nights |
| MPC Observatory Codes API -> `Observatory.timezone` | A free-form `CharField` populated by `MPCObscodeFetcher`; its value selects the `zoneinfo` zone this rule resolves against |
| classical schedule file -> `load_telescope_runs` -> sub-night fields | The only automated writer of the two fields |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-rmd-01 | Tampering | `allocation_projector.night_bounds()` | medium | mitigate | A boundary resolved onto the wrong UTC date writes a `CalendarEvent` a full day off onto the shared calendar with no error. Mitigated by Task 1's span-based resolution plus Task 2's explicit UTC-datetime assertions across all three bands -- the silent cases raise nothing, so assertion-free tests would not catch them |
| T-rmd-02 | Denial of Service | `campaign_reconciler.reconcile_run()` | medium | mitigate | The loud failure mode counts an affected run `failed` on every reconcile forever, skips its schedule line forever, and 500s a staff action on it. Mitigated by Task 1; the `ValueError` guard is retained deliberately so a genuinely inverted operator input still fails loudly rather than writing bad data |
| T-rmd-03 | Denial of Service | `_span_needs_remint()` astropy-free path | low | mitigate | A resolution mismatch between the mint path and the re-mint check would re-mint every night on every sweep, and re-introducing an astropy call here would put a solar scan on the update path against D-13. Mitigated by Task 2 test 4 (mocked `sun_event` asserted not called, `unchanged` reported, same primary key) |
| T-rmd-04 | Information Disclosure | new test fixtures | low | accept | The three new `Observatory` rows carry only public MPC site coordinates and IANA zone names, created in the test database only |
| T-rmd-SC | Tampering | npm/pip/cargo installs | n/a | accept | No package is installed by this task set; `zoneinfo` is stdlib and already imported by this module. The package-legitimacy gate does not apply |
</threat_model>

<verification>
1. `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_allocation_projector_signals solsys_code.tests.test_campaign_reconciler` passes in full (never `./manage.py`; never the full unfiltered suite -- `test_views.TestEphemeris` segfaults in native ASSIST).
2. The four pre-existing `TestSubNightWindowSiteDirection` tests and all six `TestSubNightWindow` tests pass unmodified -- bands 1 and 2-west are byte-identical regressions guards, not rewritten expectations.
3. Every new band test asserts resolved UTC datetimes explicitly against the `<ground_truth>` table.
4. `pre-commit run ruff --files ...` and `pre-commit run ruff-format --files ...` are clean on every changed Python file.
5. The `models.py` field-contract comment and the changed docstrings state three bands; neither presents the superseded two-case rule as the durable contract.
</verification>

<success_criteria>
- A `night_end_utc` of 03:00 on an `Africa/Johannesburg` site resolves to the following UTC date, producing a non-inverted span where the current code raises on every reconcile.
- An `Asia/Kolkata` 00:00-02:00 window resolves to the following UTC date rather than silently a day early on the evening date.
- A `Pacific/Honolulu` run resolves BOTH ends onto the following UTC date, including the late-hour end that lands a full day early today.
- `Australia/Sydney` and `America/Santiago` behaviour is unchanged.
- `night_bounds()`'s inverted-span `ValueError` guard is intact with its current message and still fires for a genuinely inverted input.
- The resolution makes no `sun_event()` call and uses no hard-coded hour threshold.
- The durable field contract in `models.py` and the changed docstrings describe three bands.
- Runbook and notebook audit outcomes are recorded in the SUMMARY either way.
</success_criteria>

<output>
Create `.planning/quick/260913-rmd-fix-35-review-md-nf-03-replace-the-sign-/260913-rmd-SUMMARY.md` when done.

Commit convention: `fix(quick-260913-rmd): ...` for code, `docs(quick-260913-rmd): ...` for
documentation. Every commit message ends with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MBQXMHen2DLP5owNpwRMKr
```
</output>
