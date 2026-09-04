---
phase: 33-series-identity-reconciler-inversion
plan: 01
subsystem: campaign-coordination
tags: [reconciler, calendar-events, django-templatetags, attribution, campaign-run]

# Dependency graph
requires: []
provides:
  - "reconcile_run() skip-the-night rule: an attributed non-RUN: event is never adopted, re-keyed or written to"
  - "ReconcileResult.skipped_nights counter"
  - "calendar_display_extras.campaign_decoration() simple_tag -- read-only, request-time campaign attribution decoration"
  - "event_form.html modal block extended to render the decoration, with an anchored #run-{pk} campaign-table link"
  - "event_title() no longer embeds a campaign name (D-12) -- the decoration tag is the single campaign label"
affects: [34-the-observation-projector-and-trigger, 35-allocation-layer-and-classical-cutover, 37-status-vocabulary-public-tallies-and-provenance-blind-gaps]

# Actuals (#2632)
actuals:
  tokens: 12264
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Attribution-not-ownership: CalendarEventMeta.run is read as 'this event is attributed to that run', never as 'the reconciler owns this event' -- namespace identity (owned_events()/writable_events()) is the only real ownership concept the reconciler still has."
    - "One query before the per-night loop: _attributed_nights() computes the full set of already-attributed nights once, so a multi-week reconcile issues one extra query total, not one per night."
    - "Display-time decoration: campaign attribution renders from a link at request time (campaign_decoration() simple_tag), never written into CalendarEvent fields, so re-projecting title/description cannot erase it."

key-files:
  created: []
  modified:
    - solsys_code/campaign_reconciler.py
    - solsys_code/templatetags/calendar_display_extras.py
    - src/templates/tom_calendar/partials/event_form.html
    - solsys_code/tests/test_campaign_reconciler.py
    - solsys_code/tests/test_calendar_template.py
    - solsys_code/tests/test_null_campaign_guards.py
    - solsys_code/tests/test_write_and_reconcile.py

key-decisions:
  - "Retired _adopted_event_for_night() outright rather than leaving it dead code -- D-01 makes adopt/re-key permanently unreachable, and a retained-but-unused re-key path is exactly the landmine spike 002 named for Phase 34."
  - "_attributed_nights() carries no blank-url restriction (unlike the retired helper's event__url='' filter) so a facility-URL-keyed attributed event -- the Phase 34 observation-event shape -- is skipped too, not just a load_telescope_runs blank-url one."
  - "campaign_decoration() builds its campaign-table link with reverse() in Python, not {% url %} in the template, so a null campaign_id returns table_url=None instead of raising NoReverseMatch on the public calendar (RESEARCH.md Pitfall 1)."
  - "event_title() drops its campaign-name branch entirely (with-campaign and without-campaign alike) rather than keeping it for campaign-having runs -- the decoration tag is now the single campaign label everywhere, avoiding two sources of truth for the same fact."

patterns-established:
  - "Skip-the-night rule: 'a night with an attributed non-RUN: event has no reconciler event' -- the same sentence Phase 35's allocation handoff will reuse verbatim."

requirements-completed: [ANNOT-01, ANNOT-02]

coverage:
  - id: D1
    description: "reconcile_run() skips an already-attributed classical night entirely -- no adopt, no re-key, no field write, no duplicate event minted"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestAttributedNightSkip"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestAttributedEventsSurviveReconcile"
        status: pass
    human_judgment: false
  - id: D2
    description: "The skip rule matches on the site-local observing night and matches any url outside the RUN: namespace, blank or facility-URL-keyed alike"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestAttributedNightSkip::test_skip_matches_on_site_local_night_not_naive_utc_date"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestAttributedNightSkip::test_facility_url_keyed_attributed_event_skips_its_night"
        status: pass
    human_judgment: false
  - id: D3
    description: "A foreign attribution (RUN:{pk}:{date} event whose companion row points at a different run) stays blocked and is never reset to the reconciling run"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_reconciler.py#TestCrossRunOwnershipGuards::test_reconcile_reports_blocked_for_a_night_attributed_to_a_different_run"
        status: pass
    human_judgment: false
  - id: D4
    description: "The calendar event modal renders campaign name, telescope/instrument, window and run status for an attributed, publicly-visible run, with an anchored #run-{pk} campaign-table link, from CalendarEventMeta.run at request time"
    requirement: "ANNOT-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalCampaignRunLinkTest::test_approved_run_shows_attributed_label_and_anchored_campaign_link"
        status: pass
    human_judgment: false
  - id: D5
    description: "A run with no campaign, no companion row, an unset run link, or a not-publicly-visible run each render no decoration and never raise (including no NoReverseMatch on a null campaign pk)"
    requirement: "ANNOT-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalCampaignRunLinkTest::test_no_campaign_run_renders_200_with_telescope_instrument_and_no_campaign_link"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalCampaignRunLinkTest::test_no_companion_row_at_all_renders_200_with_no_exception"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#EventModalCampaignRunLinkTest::test_pending_run_shows_no_run_block_to_staff_visitor"
        status: pass
    human_judgment: false
  - id: D6
    description: "Reconciler-written RUN: event titles no longer carry the campaign-name prefix; the terminal-status prefix vocabulary (status ring) is unchanged"
    requirement: "ANNOT-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_null_campaign_guards.py#TestEventTitleGuard"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_write_and_reconcile.py#TestEventTitleNullCampaignGuard"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_calendar_display_extras.py#StatusBorderCssTest"
        status: pass
    human_judgment: false

duration: 73min
completed: 2026-09-04
status: complete
---

# Phase 33 Plan 1: Series Identity & Reconciler Inversion (annotator inversion + display-time decoration) Summary

**Inverted `campaign_reconciler.reconcile_run()` from an owner into an annotator (skip-the-night rule, `skipped_nights` counter) and added a request-time `campaign_decoration()` template tag so an attributed event's calendar modal shows its campaign without ever writing to the event.**

## Performance

- **Duration:** 73 min
- **Started:** 2026-09-04T14:10:58Z
- **Completed:** 2026-09-04T15:23:43Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- `_adopted_event_for_night()` (the adopt/re-key write path D-01 retires) is deleted outright; `_attributed_nights()` replaces it with a single-query, per-reconcile computation of every site-local night already covered by an attributed non-`RUN:` event, and `_reconcile_classical_nights()` skips those nights entirely -- no event created, modified, re-keyed or deleted for them.
- `ReconcileResult` gained a `skipped_nights` counter, seeded in the classical branch's `totals` dict literal so the first skipped night never raises `KeyError`.
- `calendar_display_extras.campaign_decoration()` is a new read-only `simple_tag`: it reads `CalendarEventMeta.run` at request time, gates on `is_publicly_visible`, and returns `None` (never raises) for a companion-row-less event, an unset link, or a non-public run. Its campaign-table link is built with `reverse()` in Python so a null-campaign run gets `table_url=None` instead of a `NoReverseMatch` crash.
- `event_form.html`'s "Campaign run" modal block became "Attributed campaign run", driven entirely by `campaign_decoration()`, with the campaign-table link anchored to `#run-{pk}`.
- `event_title()` no longer embeds a campaign-name prefix for any run, with or without a campaign -- the decoration tag is the single campaign label everywhere now; the terminal-status prefix (`[CANCELLED]`/`[WEATHERED]`) that `status_border_css()` matches on is unchanged.
- Test coverage: `TestAdoptAndRekey` rewritten as `TestAttributedNightSkip` (4 tests, including the facility-URL-keyed case); a new `TestAttributedEventsSurviveReconcile` class proves byte-identical survival and idempotency for both blank-url and facility-URL-keyed attributed events; `TestCrossRunOwnershipGuards` gained a foreign-attribution-blocked regression and an in-place-refresh regression; `TestRecordEventNonInterference` gained one new test proving the attribution link, not `CampaignRunObservation`, is what triggers the skip; `EventModalCampaignRunLinkTest` gained the anchored-link assertion, a null-campaign-run fixture, and a no-decoration-for-staff assertion on the pending-review case.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end "an attributed night survives a sweep and shows its campaign"** - `09a6612` (feat)
2. **Task 2: Prove the inversion — byte-identical attributed events, URL-keyed skip, foreign attribution still blocked** - `a59fa3c` (test)
3. **Task 3: Drop the campaign name from reconciler event titles and restate wording as attribution** - `29ee683` (refactor)

_Note: Task 2 carried `tdd="true"` but its `<files>` was test-only -- Task 1 already implemented the behavior it proves, so these tests were GREEN on first run rather than following a separate RED phase. See "TDD Gate Compliance" below._

## Files Created/Modified

- `solsys_code/campaign_reconciler.py` - `_adopted_event_for_night()` deleted; `_attributed_nights()` added; `ReconcileResult.skipped_nights`; skip-the-night rule wired into `_reconcile_classical_nights()`; `event_title()` no longer embeds a campaign name; D-17 attribution-not-ownership wording pass over the module docstring and `_link_event_to_run()`/`_detach_stale_family_events()`
- `solsys_code/templatetags/calendar_display_extras.py` - new `campaign_decoration()` simple_tag
- `src/templates/tom_calendar/partials/event_form.html` - "Attributed campaign run" decoration block driven by `campaign_decoration()`, anchored campaign-table link
- `solsys_code/tests/test_campaign_reconciler.py` - `TestAttributedNightSkip` (renamed/rewritten from `TestAdoptAndRekey`); `TestAttributedEventsSurviveReconcile` (new); regressions added to `TestCrossRunOwnershipGuards` and `TestRecordEventNonInterference`
- `solsys_code/tests/test_calendar_template.py` - `EventModalCampaignRunLinkTest` extended (anchored link, null-campaign fixture, staff no-decoration assertion)
- `solsys_code/tests/test_null_campaign_guards.py` - `TestEventTitleGuard` updated to the no-campaign-label title
- `solsys_code/tests/test_write_and_reconcile.py` - `TestEventTitleNullCampaignGuard` updated to the no-campaign-label title

## Decisions Made

- Retired `_adopted_event_for_night()` outright rather than leaving it as unused dead code -- D-01 makes the adopt/re-key path permanently unreachable, and a retained-but-unused re-key helper is exactly the kind of landmine spike 002 named for Phase 34's projector.
- `_attributed_nights()` deliberately carries no blank-url restriction (unlike the retired helper's `event__url=''` filter), so a facility-URL-keyed attributed event -- the shape Phase 34's observation events will have -- is skipped too, not only a `load_telescope_runs` blank-url event.
- `campaign_decoration()` builds its campaign-table link with `reverse()` in Python rather than `{% url %}` in the template, so a null `campaign_id` returns `table_url=None` instead of raising `NoReverseMatch` on the public, unauthenticated calendar page (RESEARCH.md Pitfall 1, threat T-33-01).
- `event_title()`'s campaign-name branch was dropped entirely rather than kept for campaign-having runs only -- the decoration tag is now the single source of the campaign label for every attributed event, avoiding two places asserting the same fact.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## TDD Gate Compliance

Task 2 carried `tdd="true"` in its frontmatter, but its `<files>` scope was test-only (`solsys_code/tests/test_campaign_reconciler.py`) -- the behavior it proves (the skip rule, `_attributed_nights()`, `skipped_nights`) was already implemented by Task 1's `feat` commit. There is no separate RED-phase commit for Task 2: its tests were GREEN against Task 1's already-committed implementation from the first run, by design (this is a coverage-completion task following a prior implementation task, not a from-scratch feature). Fail-fast RED-phase behavior ("investigate a test that passes before implementation exists") does not apply here because the implementation is intentionally pre-existing per the plan's own task split. No gate violation is being flagged -- this is the plan's designed structure, documented here for auditability.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `reconcile_run()` never adopts, re-keys or writes into a non-`RUN:` attributed event -- the ordering landmine Phase 34's observation projector needed cleared before it can run alongside the campaign layer.
- `CalendarEventMeta.run` is now uniformly read as attribution, not ownership, across the reconciler's own docstrings -- consistent with what Phase 34/35 will build on.
- Plan 33-02 (the `CalendarEventMeta` series-identity link fields) and 33-03/33-04/33-05 in this phase are unaffected by and do not block on this plan's scope.
- No blockers.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-04*

## Self-Check: PASSED

All 7 modified files verified present on disk; all 3 task commits (09a6612, a59fa3c, 29ee683) verified present in git log. Full acceptance-criteria and `<verify>` command re-run: `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_calendar_template solsys_code.tests.test_null_campaign_guards solsys_code.tests.test_write_and_reconcile` (114 tests, OK); the project full-suite command (985 tests total across both invocations, OK); `pre-commit run ruff --all-files` and `ruff-format --all-files` (both Passed); `! grep -q '_adopted_event_for_night' solsys_code/campaign_reconciler.py` (PASS).
