---
status: resolved
trigger: "User disagrees with the behavior that range-window CampaignRun approvals never project a CalendarEvent, even for real, competed-for/awarded telescope time. Concrete case: the real GS-2026A-FT-115 Gemini South run (CampaignRun pk=34, 65803 Didymos, Gemini-South GMOS-S, 6.50 awarded hours, window 2026-07-13..2026-07-16, resolved to the real Gemini South Observatory via I11) was created, approved, and resolved -- CalendarEvent.objects.filter(url='CAMPAIGN:34').count() confirmed 0. User's stated reasoning: this was competed-for and awarded time which followers of the campaign would want to know about and which would (likely) have got observed if not for the storm -- per the Gemini South website (https://www.gemini.edu/news/science-operations-announcements/gemini-south-shutdown-advanced-one-week), the shutdown was brought forward because of the storm, so no observations from July 13 onwards. This is a design-goal investigation, not a functional-bug hunt: diagnose only, produce a Root Cause Report plus a clear before/after behavior spec for a future planning phase -- do not modify code or tests in this session."
created: 2026-07-17T13:00:00Z
updated: 2026-07-18T09:30:00Z
implemented_in: |
  Phase 25 (range-window-calendarevent-projection-allow-approved-site-re), plans 25-01 and
  25-02, following this file's before/after spec almost exactly:
  - 25-01 (efbd366, df676f1, 5187e4b, e2f703a, a6de37f): rewrote _project_calendar_event()'s
    guard to drop the window_start == window_end equality term (kept both-non-null), gave the
    ground branch real per-night dip-corrected date-math (one CalendarEvent per night, going
    further than the spec's Option A whole-day-span recommendation), kept the satellite
    branch's existing whole-day span, added the window-context title suffix via a shared
    title helper, and updated _set_run_status() so every night's event gets synced. All four
    tests this file identified as needing to flip (0 -> 1) were revised per the spec.
  - 25-02 (1fb56ae, 0ff4a70, 4a889d7, d2ef1dc): added `backfill_range_calendar_events`, a
    one-off management command that finds already-APPROVED, site-resolved range-window
    CampaignRuns with no existing CalendarEvent (exactly the disputed real case, pk=34
    GS-2026A-FT-115) and projects them by delegating to the rewritten
    _project_calendar_event() -- closing the gap left by projection only firing on the
    approve/resolve_site POST actions.
  Phase 25 verification passed (25-VERIFICATION.md, status: passed).
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

status: DIAGNOSED (diagnose-only; no code changed).

root_cause_confirmed: |
  The `run.window_start == run.window_end` sub-clause in the
  _project_calendar_event() guard (solsys_code/campaign_views.py:412) is the
  sole cause -- it returns False early for every range-window run before the
  projection branch runs. Verified verbatim against source. Provenance is Phase
  19 D-06 behavior-preservation (deferred, not decided-against). CalendarEvent
  supports multi-day spans (no model limitation). See Resolution + Evidence.

key_refinement: |
  Orchestrator's "just an unexercised code path" framing is only half-right:
  the SATELLITE branch's multi-day date-math is already correct, but the GROUND
  branch (which the real Gemini I11 OPTICAL site takes) computes both event
  timestamps from a single sun_event(window_start) call and never reads
  window_end -- so a bare guard flip would mislabel a 4-day allocation as its
  first night. A correct fix needs new ground-branch range date-math too.

next_action: |
  None -- investigation complete. Hand the ROOT CAUSE FOUND report + the
  before/after spec (Resolution.next_planning_phase) to a future planning
  phase. No fix applied in this session per goal: find_root_cause_only.

---

## Symptoms

expected: |
  A ground-based CampaignRun representing real, competed-for/awarded
  telescope time -- even when its window is a date range rather than a
  single concrete night (e.g. the real GS-2026A-FT-115 Gemini South run,
  6.50 GMOS-S hours awarded across 2026-07-13..2026-07-16) -- should be
  visible on the campaign calendar once approved and site-resolved, so
  campaign followers can see the allocation and understand when weather or
  other factors preempted it.
actual: |
  No CalendarEvent is ever created for a range-window run, regardless of
  significance. Verified directly against the real dev-DB row:
  CampaignRun pk=34 (GS-2026A-FT-115) was created, approved, and its site
  resolved (I11 -> Gemini South Observatory) -- `CalendarEvent.objects.filter(url='CAMPAIGN:34').count()`
  returns 0 and stays 0 through approval. This matches the encoded-as-correct
  behavior in solsys_code/tests/test_campaign_approval.py's
  TestCalendarProjection.test_approve_range_run_creates_no_calendar_event
  and this session's new TestGeminiFtScenario (23-03), both of which assert
  the count stays 0 for range-window runs through approve ->
  mark_weather_failure -> mark_cancelled.
errors: |
  None -- this is silent-by-design behavior (an early `return False` from
  `_project_calendar_event()`), not an exception or crash.
timeline: |
  Not a regression. Traces to Phase 19 (window-schema-migration)'s D-06
  decision, which replaced the old obs_date/ut_start/ut_end model with
  window_start/window_end and explicitly chose to reproduce the prior
  single-night-only gate ("Only projects when window_start == window_end...
  ranges and TBD runs still don't get a CalendarEvent, matching today's
  ut_end-missing gate behavior") rather than design range projection at that
  time. No later phase (20, 21, 22, 23) revisited this gate -- Phase 23's own
  new tests (TestGeminiFtScenario) explicitly re-encode it as correct
  behavior for the real Gemini scenario.
reproduction: |
  1. Create (or use the existing) real GS-2026A-FT-115 CampaignRun: Didymos
     2026 TargetList, telescope_instrument='Gemini-South GMOS-S',
     site_raw='I11', window_start=2026-07-13, window_end=2026-07-16,
     target=65803 Didymos (Target pk=2), run_status=REQUESTED.
  2. Approve it through the approval queue; let the site resolve (I11 ->
     Gemini South Observatory, ground-based, America/Santiago).
  3. Check `CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').count()`
     -- returns 0, though the run is a real, approved, site-resolved,
     awarded allocation.

## Evidence

- timestamp: 2026-07-17T14:00:00Z
  checked: |
    solsys_code/campaign_views.py:412 -- the guard in _project_calendar_event(),
    read in full (function spans 392-455).
  found: |
    Guard is exactly:
    `if not (run.telescope_instrument and run.site and run.window_start and
    run.window_start == run.window_end): return False`.
    The `run.window_start == run.window_end` sub-clause is the SOLE reason a
    range-window run (window_start != window_end, both non-null) returns False
    early -- before the ground-vs-satellite branch is ever selected. Orchestrator's
    preliminary line number (412) and logic confirmed verbatim.
  implication: |
    Immediate mechanism confirmed. Removing/relaxing this one sub-clause is
    what unblocks range projection. The truthiness checks
    (telescope_instrument, site, window_start) correctly keep excluding TBD and
    unresolved-site runs.

- timestamp: 2026-07-17T14:05:00Z
  checked: |
    solsys_code/campaign_views.py:420-429 (satellite branch) vs 442-453
    (ground branch) -- the two projection code paths AFTER the guard.
  found: |
    SATELLITE branch already builds a genuine multi-day span:
    start_time = combine(window_start, 00:00 UTC), end_time =
    combine(window_end, 23:59 UTC) -- range-correct as written.
    GROUND branch is SINGLE-NIGHT-ONLY: it calls
    `sun_event(run.site, run.window_start, kind='sun')` and assigns BOTH
    start_time (sunset) and end_time (sunrise) from that one window_start
    night. It never reads window_end at all.
  implication: |
    CRITICAL REFINEMENT to the orchestrator hypothesis. The disputed real case
    (Gemini South I11) is OPTICAL_OBSTYPE = GROUND, not satellite. Simply
    deleting the `== ` clause would make a ground range run project only the
    FIRST night's dark window, mislabelling a 4-day awarded allocation as a
    single night. The ground branch needs NEW range date-math (first-night
    sunset -> last-night sunrise, or a whole-day span mirroring the satellite
    branch). This is a genuinely unexercised path for the ground branch, not
    just a guard flip.

- timestamp: 2026-07-17T14:10:00Z
  checked: |
    tom_calendar/models.py (site-packages) -- CalendarEvent field definitions.
  found: |
    start_time = DateTimeField(); end_time = DateTimeField() -- both
    non-nullable, no default, and NO constraint requiring start_time==end_time
    or any relationship between them. url = URLField(blank), used as the
    CAMPAIGN:{pk} idempotency key.
  implication: |
    Confirms this is NOT a model limitation. Multi-day spans are fully
    supported (the satellite branch already stores one). The zero-event
    behavior is purely the view-layer guard, not the schema.

- timestamp: 2026-07-17T14:15:00Z
  checked: |
    .planning/phases/19-window-schema-migration/19-CONTEXT.md D-06 (lines
    69-89) and scope note (lines 23-25).
  found: |
    D-06 states the gate "Only projects when window_start == window_end (a
    single concrete night) -- ranges and TBD runs still don't get a
    CalendarEvent, matching today's ut_end-missing gate behavior." Out-of-scope
    note: "actually importing range/TBD CSV rows ... range rows only display,
    not project, in this phase". The ground-vs-space hybrid was explicitly a
    "narrow, early application" for Phase 19.
  implication: |
    Provenance confirmed. The equality gate is a deliberate
    BEHAVIOR-PRESERVATION choice reproducing the legacy ut_start/ut_end-missing
    single-night gate, with range projection consciously DEFERRED as
    out-of-scope -- NOT a considered decision that ranges should never be
    visible. This directly supports the user's position.

- timestamp: 2026-07-17T14:20:00Z
  checked: |
    solsys_code/campaign_views.py:708-763 (_set_run_status) and the
    _RUN_STATUS_CALENDAR_PREFIX map (379-382).
  found: |
    mark_cancelled/mark_weather_failure calendar sync is guarded ONLY by
    `if CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}').exists()`.
    It makes NO window-shape assumption -- it just updates title (prepending
    [CANCELLED] / [WEATHERED]) and description in place. It never calls
    _project_calendar_event() and never fabricates an event.
  implication: |
    Answers test-question (4): NO change is needed to Plan 02's
    existence-guarded sync. Once approve projects a multi-day event for a range
    run, the existence guard flips True and the [WEATHERED]/[CANCELLED] prefix
    flow (exactly the Gemini storm-cancellation scenario) Just Works with zero
    additional code.

- timestamp: 2026-07-17T14:25:00Z
  checked: |
    solsys_code/tests/test_campaign_approval.py -- every assertion encoding the
    disputed zero-event contract.
  found: |
    FOUR test methods assert count==0 for a RANGE run (window_start != window_end):
    (1) TestCalendarProjection.test_approve_range_run_creates_no_calendar_event
        (349-354): range 8/1..8/15, count==0 after approve.
    (2) TestRunStatusChange.test_mark_range_window_run_does_not_crash_and_creates_no_event
        (450-458): range 8/1..8/15, count==0 through mark_cancelled.
    (3) TestGeminiFtScenario.test_gemini_ft115_range_window_flows_through_same_mechanism_no_event_fabricated
        (2162-2208): the real I11 range 7/13..7/16, THREE count==0 assertions
        (approve L2183, mark_weather_failure L2193, mark_cancelled L2202).
    (4) TestResolveSite...test_resolve_range_tbd_run_clears_flag_with_no_calendar_event
        (997-1010): range 8/1..8/15 via resolve_site, count==0 AND message
        'Site resolved.' (NOT '... run added to the calendar.').
    Tests that must STAY count==0 (guard must keep excluding these):
    test_approve_tbd_run_creates_no_calendar_event (356, both windows null),
    test_approve_without_telescope_instrument_creates_no_calendar_event (363),
    test_sun_event_valueerror_skips_projection_without_reverting_approval (370,
    blank-timezone site).
  implication: |
    Exactly four assertions/blocks must flip from count 0 -> 1; test (4) also
    flips its success message; tests (2) and (3) additionally begin exercising
    the [CANCELLED]/[WEATHERED] in-place prefix update on the now-existing
    event. Three separate tests must remain unchanged, and they pin down the
    precise shape the new guard must keep (TBD, no-instrument,
    unresolved/blank-tz site all still project nothing).

## Eliminated

- hypothesis: |
    The zero-event behavior is a CalendarEvent model limitation (start_time/
    end_time can't express a multi-day span, or require equality).
  evidence: |
    tom_calendar/models.py: both fields are plain non-nullable DateTimeFields
    with no inter-field constraint, and the satellite branch already persists a
    multi-day (window_start 00:00 -> window_end 23:59) event. The block is
    entirely the view-layer guard.
  timestamp: 2026-07-17T14:10:00Z

- hypothesis: |
    Removing the `window_start == window_end` sub-clause alone is a complete
    fix (the orchestrator's "just an unexercised code path" framing).
  evidence: |
    True for the satellite branch, but FALSE for the ground branch (the actual
    Gemini I11 = OPTICAL case): the ground branch derives both start_time and
    end_time from a single sun_event(window_start) call and never reads
    window_end, so a bare guard flip would mislabel a 4-day awarded window as
    its first night only. New ground-range date-math is required.
  timestamp: 2026-07-17T14:05:00Z

## Resolution

root_cause: |
  CONFIRMED. Range-window CampaignRuns never project a CalendarEvent because of
  a single sub-clause in the projection guard at
  solsys_code/campaign_views.py:412:

    if not (run.telescope_instrument and run.site and run.window_start
            and run.window_start == run.window_end):
        return False

  The `run.window_start == run.window_end` term forces an early `return False`
  for every run whose window is a date range, so the ground-vs-satellite
  projection branch below is never reached. For the disputed real case
  (GS-2026A-FT-115, CampaignRun pk=34, window 2026-07-13..2026-07-16, resolved
  to Gemini South I11), the run is fully approved and site-resolved yet
  `CalendarEvent.objects.filter(url='CAMPAIGN:34').count()` stays 0.

  This is NOT a model limitation: CalendarEvent.start_time/end_time are plain
  non-nullable DateTimeFields with no inter-field constraint, and the SATELLITE
  branch (campaign_views.py:420-429) already persists a genuine multi-day span.

  PROVENANCE: the equality gate is a deliberate BEHAVIOR-PRESERVATION choice
  from Phase 19 D-06 (19-CONTEXT.md:82-84), reproducing the legacy
  ut_start/ut_end-missing single-night gate. Range projection was explicitly
  deferred as out-of-scope for Phase 19 ("range rows only display, not project,
  in this phase", 19-CONTEXT.md:23-25), never a considered decision that ranges
  should stay invisible. No later phase (20-23) revisited it; Phase 23's
  TestGeminiFtScenario merely re-encoded the deferred behavior as the current
  contract. The user's objection is therefore well-founded: the zero-event
  behavior is an unfinished-feature artifact, not an intentional product rule.

  KEY REFINEMENT (corrects the orchestrator's preliminary "unexercised code
  path" framing): only the SATELLITE branch's date-math is range-correct today.
  The GROUND branch (campaign_views.py:442-453 -- the branch the real I11
  OPTICAL site actually takes) computes both start_time and end_time from a
  single `sun_event(run.site, run.window_start)` call and never reads
  window_end. So relaxing the guard alone is INSUFFICIENT for ground runs: it
  would project only the FIRST night's dark window and mislabel a multi-day
  awarded allocation as a single night. A correct fix must also give the ground
  branch multi-day date-math.

next_planning_phase: |
  BEFORE / AFTER BEHAVIOR SPEC for a future planning phase (diagnose-only
  session -- no code changed here).

  Scope: solsys_code/campaign_views.py (_project_calendar_event) +
  solsys_code/tests/test_campaign_approval.py. campaign_views.py is NOT one of
  the four demo-notebook modules in CLAUDE.md, so no paired notebook is
  required.

  --- Guard: BEFORE ---
    if not (telescope_instrument and site and window_start
            and window_start == window_end): return False
  --- Guard: AFTER (recommended) ---
    if not (telescope_instrument and site and window_start and window_end):
        return False
  Rationale: require BOTH window ends non-null (concrete dates on both sides)
  but drop the equality term. Adding the `window_end` truthiness check is
  REQUIRED, not optional -- without it a half-open window (window_start set,
  window_end null) would reach datetime.combine(None, ...) / an under-specified
  span. This keeps every currently-excluded case excluded:
    (a) TBD runs (window_start is None)          -> still no event  [unchanged]
    (b) unresolved-site runs (site is None)      -> still no event  [unchanged]
    (c) missing telescope_instrument             -> still no event  [unchanged]
    (d) blank-timezone site (sun_event raises)   -> still handled by the
        existing try/except (approve swallows; resolve_site keeps retry surface)
  and newly ADMITS:
    (e) ground range-window run, resolved site   -> project multi-day banner
    (f) satellite range-window run, resolved site-> project multi-day banner
        (already works today; guard was the only blocker)

  --- Ground branch date-math: BEFORE ---
    sunset, sunrise = sun_event(site, window_start)
    start_time = sunset; end_time = sunrise      # first night only
  --- Ground branch date-math: AFTER (two options for the planner) ---
    Option A (simplest, mirror satellite whole-day span):
      start_time = combine(window_start, 00:00 UTC)
      end_time   = combine(window_end,   23:59 UTC)
      -- reuses the exact satellite-branch construction; no per-night sun math;
         a single unbroken multi-day allocation banner. Recommended for a first
         cut: it is unambiguous and matches how "awarded time" reads.
    Option B (dip-corrected endpoints):
      first_sunset, _ = sun_event(site, window_start)
      _, last_sunrise = sun_event(site, window_end)
      start_time = first_sunset; end_time = last_sunrise
      -- more physically precise endpoints, but the multi-day span makes the
         intermediate nights implicit anyway, so the extra accuracy is cosmetic.
    Single-night ground runs (window_start == window_end) MUST keep their
    current dip-corrected sun_event() behavior unchanged either way.

  --- Title / description spec ---
    Current title `f'{campaign.name}: {telescope_instrument}'` and description
    `run.observation_details` work for a range unchanged. RECOMMENDED
    enhancement so a multi-day awarded banner is distinguishable from a single
    night slot: append the window to the title or description, e.g.
    title `f'{campaign.name}: {telescope_instrument} (window {window_start}..{window_end})'`
    for range runs only (leave single-night titles as-is). This is a product
    choice for discuss-phase, not a hard requirement. The [CANCELLED] /
    [WEATHERED] prefix flow already prepends cleanly to whatever title is
    chosen.

  --- mark_weather_failure / mark_cancelled ---
    NO change needed. _set_run_status()'s calendar sync is guarded solely by
    `CalendarEvent.objects.filter(url='CAMPAIGN:{pk}').exists()` and makes no
    window-shape assumption. Once approve projects the range event, marking the
    run weathered/cancelled updates that event's title/description in place with
    the correct prefix -- exactly the Gemini storm-cancellation narrative the
    user wants visible.

  --- Tests whose assertions MUST flip (0 -> 1), all in test_campaign_approval.py ---
    1. TestCalendarProjection.test_approve_range_run_creates_no_calendar_event
       (349-354): assert count == 1 (+ assert the multi-day start/end span).
    2. TestRunStatusChange.test_mark_range_window_run_does_not_crash_and_creates_no_event
       (450-458): assert count == 1 and the event title now carries [CANCELLED].
       (Rename away from "creates_no_event".)
    3. TestGeminiFtScenario.test_gemini_ft115_range_window_flows_through_same_mechanism_no_event_fabricated
       (2162-2208): all three count==0 assertions -> count==1; mark_weather_failure
       -> [WEATHERED] title; mark_cancelled -> [CANCELLED] title; still exactly ONE
       event (in-place update, never a second). Update the docstring/name that
       currently says "range-window skip-by-design" / "no_event_fabricated".
    4. TestResolveSite...test_resolve_range_tbd_run_clears_flag_with_no_calendar_event
       (997-1010): a pure range (not TBD); assert count == 1 AND success message
       'Site resolved — run added to the calendar.' (not 'Site resolved.').
       (Misnamed -- it is a range case; split out a genuine TBD-resolve case that
       stays count==0 if TBD-resolve coverage is wanted.)
  --- Tests that MUST stay count == 0 (guard must keep excluding these) ---
    test_approve_tbd_run_creates_no_calendar_event (356),
    test_approve_without_telescope_instrument_creates_no_calendar_event (363),
    test_sun_event_valueerror_skips_projection_without_reverting_approval (370).
  --- New coverage to add ---
    A ground range-window approve test asserting the FULL multi-day span (start
    on window_start, end on window_end) -- guards against a regression to the
    first-night-only ground math.

files_changed: |
  None -- diagnose-only session (goal: find_root_cause_only). No production
  code or tests modified. Implementation belongs to the future planning phase
  scoped by next_planning_phase above.
