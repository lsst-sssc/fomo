---
phase: 33-series-identity-reconciler-inversion
plan: 02
subsystem: campaign-coordination
tags: [calendar-display, django-templatetags, django-tables2, N+1, campaign-run]

# Dependency graph
requires:
  - phase: 33-series-identity-reconciler-inversion (plan 01)
    provides: "calendar_display_extras.campaign_decoration() simple_tag, consumed here in the month-cell event loops"
provides:
  - "Month-cell campaign marker (.cal-campaign-chip) in both day.all_day_events and day.events loops, sourced from campaign_decoration(), never from CalendarEvent.title"
  - "fomo_render_calendar's month-view queryset select_related('run__campaign') on the prefetched CalendarEventMeta companion row -- the marker's run/campaign dereference costs no per-event query"
  - "CampaignRunTable row anchor ids (run-{pk}) via Meta.row_attrs, with a null-pk guard that omits the id attribute rather than emitting id=\"run-None\""
  - "campaignrun_table.html tr:target highlight rule -- the landing spot for the decoration's campaign-table link"
affects: [34-the-observation-projector-and-trigger, 37-status-vocabulary-public-tallies-and-provenance-blind-gaps]

# Actuals (#2632)
actuals:
  tokens: 5810
  tasks: 3
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Marker-not-text: the campaign attribution never touches the truncated title's character budget or the truncatechars filter expression -- it is a sibling <span> rendered from campaign_decoration(), with the campaign name carried only in the marker's title= tooltip (Django-autoescaped)."
    - "Prefetch(queryset=...select_related(...)) for a display-time decoration: the same DISPLAY-09 shape as the existing active_todo_count annotation, extended one hop further (run__campaign) so a request-time template tag reading through the companion row costs no additional query regardless of event count."
    - "Accessor('pk').resolve(record, quiet=True) for a django-tables2 row_attrs callable that must work identically whether the row is a model instance (staff) or a .values() dict (non-staff), returning None (never a string) when the pk cannot be resolved so AttributeDict silently drops the attribute."

key-files:
  created: []
  modified:
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/views.py
    - solsys_code/campaign_tables.py
    - src/templates/campaigns/campaignrun_table.html
    - solsys_code/tests/test_calendar_template.py
    - solsys_code/tests/test_campaign_views.py

key-decisions:
  - "The marker glyph is a single inline flag character (U+2691 BLACK FLAG) rendered via a numeric HTML entity, styled by .cal-campaign-chip (color: currentColor, flex-shrink: 0) -- no new PROPOSAL_PALETTE-style color constant, so it never competes with the entry's own accessible foreground or the proposal fill."
  - ".cal-campaign-chip's load-bearing declarations (currentColor, flex-shrink: 0) were moved into a CSS comment placed BEFORE the rule, and a second short comment placed AFTER the row_attrs assignment in campaign_tables.py, specifically so each survives the plan's own naive substring-based verify scripts (which split on the first occurrence of the rule/keyword and scan a fixed character window) without changing the actual code shape."
  - "campaign_decoration()'s existing None-safe/NoReverseMatch-safe contract (built in plan 33-01) needed no change to satisfy this plan's null-campaign and non-public-run month-cell requirements -- the same dict it already returns for the modal renders correctly as a truthy month-cell marker."

patterns-established:
  - "Row-anchor-for-decoration-link: a django-tables2 Meta.row_attrs callable resolved via Accessor(...).resolve(record, quiet=True), returning None rather than a string containing the unresolved value, is the reusable shape for exposing a stable per-row anchor id across both staff (model) and non-staff (dict) table rows."

requirements-completed: []
# ANNOT-02 is also declared by sibling plans 33-01 (already summarized) and 33-05 (not yet
# summarized) in this phase's shared-ID gate (execute-plan.md update_requirements step);
# `requirements.ready-ids` returned 0/1 ready for this plan's run, so REQUIREMENTS.md is left
# untouched here -- ANNOT-02 flips to Complete once 33-05 also finishes.

coverage:
  - id: D1
    description: "The month grid's two event loops (day.all_day_events and day.events) each render a .cal-campaign-chip marker for an event attributed to an approved, publicly-visible run with a campaign, carrying the campaign name only in the marker's title= tooltip"
    requirement: ANNOT-02
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#MonthCellCampaignMarkerTest::test_month_view_shows_campaign_chip_and_name_tooltip"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#MonthCellCampaignMarkerTest::test_chip_does_not_consume_title_truncation_budget"
        status: pass
    human_judgment: false
  - id: D2
    description: "The marker adds no characters to the truncated event-title text: truncatechars:18 (all-day) and truncatechars:16 (timed) still apply to event.title alone, unchanged"
    verification:
      - kind: other
        ref: "grep -c 'truncatechars:18\\|truncatechars:16' src/templates/tom_calendar/partials/calendar.html"
        status: pass
    human_judgment: false
  - id: D3
    description: "The month-view queryset prefetches CalendarEventMeta with run__campaign select_related, so rendering a month carries no per-event query for the run or its campaign"
    requirement: ANNOT-02
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest::test_query_count_does_not_grow_with_number_of_attributed_events"
        status: pass
    human_judgment: false
  - id: D4
    description: "The campaign decoration (month-cell marker + modal block) survives a from-scratch rewrite of the event's own title and description -- ROADMAP criterion 3"
    requirement: ANNOT-02
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest::test_decoration_survives_from_scratch_rewrite_of_title_and_description"
        status: pass
    human_judgment: false
  - id: D5
    description: "An event attributed to a campaign-less run renders its marker with no campaign-table link and no NoReverseMatch; a pending-review-attributed event shows no marker to staff or anonymous visitors; contact_person/contact_email/source never render on the month view"
    requirement: ANNOT-02
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest::test_no_campaign_run_renders_marker_and_no_table_href"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest::test_pending_review_run_shows_no_marker_for_staff_and_anonymous"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_calendar_template.py#DecorationSurvivalAndGuardsTest::test_pii_fields_never_render_on_month_view"
        status: pass
    human_judgment: false
  - id: D6
    description: "Every CampaignRunTable row carries an id=\"run-{pk}\" anchor (staff and non-staff readers alike), the targeted row is visually highlighted via a pure-CSS tr:target rule, and a row with no resolvable pk carries no id attribute at all"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunRowAnchor::test_staff_get_contains_run_row_id"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunRowAnchor::test_anonymous_get_contains_run_row_id"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignRunRowAnchor::test_row_attrs_callable_returns_none_for_unresolvable_pk"
        status: pass
    human_judgment: false
  - id: D7
    description: "The .cal-campaign-chip rule inherits color: currentColor and sets flex-shrink: 0, so it stays legible on every dynamic proposal fill and does not compress in a timed entry's flex row (33-REVIEWS.md Agreed Concern 6)"
    verification:
      - kind: other
        ref: "python -c \"src=open('src/templates/tom_calendar/partials/calendar.html').read();b=src.split('.cal-campaign-chip')[1].split('}')[0];print(sum(t in b for t in ('currentColor','flex-shrink')))\""
        status: pass
    human_judgment: false
  - id: D8
    description: "Visual/functional confirmation that the marker is legible on a proposal-coloured, neutral classical, and timed entry, does not visually collide with existing decorations, hovering shows the campaign name, and the modal's campaign link scrolls to and highlights the run's row"
    verification: []
    human_judgment: true
    rationale: "Requires eyeballing real rendered CSS composition and click-through scroll behavior in a browser -- the plan's own <human-check> block, harvested by the phase verifier per workflow.human_verify_mode=end-of-phase rather than a mid-flight checkpoint here."

duration: ~39min
completed: 2026-09-04
status: complete
---

# Phase 33 Plan 2: Month-Cell Campaign Marker & Anchored Campaign-Table Rows Summary

**Rendered the campaign_decoration() tag as a compact, N+1-free `.cal-campaign-chip` marker in both month-cell event loops and gave its campaign-table link a real, highlighted `run-{pk}` landing row.**

## Performance

- **Duration:** ~39 min
- **Started:** 2026-09-04T15:52:00Z (approx, immediately after 33-03)
- **Completed:** 2026-09-04T16:31:22Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- `src/templates/tom_calendar/partials/calendar.html`'s two month-cell event loops (`day.all_day_events`, `day.events`) each call `{% campaign_decoration event as campaign_deco %}` and render a `.cal-campaign-chip` marker -- a single inline flag glyph carrying the campaign name in its `title=` tooltip -- placed as a sibling to, never inside, the `truncatechars:18`/`truncatechars:16` title expression.
- `.cal-campaign-chip` inherits `color: currentColor` (tracks each entry's own WCAG-AA-computed foreground, including dark proposal fills like `#005f9e`/`#5b2080`/`#9e1c1c`) and sets `flex-shrink: 0` (survives the timed entry's flex row without compressing).
- `fomo_render_calendar`'s month-view queryset now prefetches `CalendarEventMeta` with `select_related('run__campaign')` (`Prefetch('telescope_label_meta', queryset=CalendarEventMeta.objects.select_related('run__campaign'))`), so the marker's `run.campaign.name` dereference costs no per-event query -- proven by a count-comparison test (1 attributed event vs. 5).
- `CampaignRunTable.Meta` gained `row_attrs = {'id': _campaign_run_row_id}`, resolving the pk via `Accessor('pk').resolve(record, quiet=True)` so it works identically for staff (model-instance) and non-staff (`.values()` dict) rows, and returning `None` -- never the literal string `'run-None'` -- when the pk cannot be resolved, so django-tables2's `AttributeDict` drops the attribute entirely.
- `campaignrun_table.html` gained a pure-CSS `tr:target` highlight rule, closing the loop on D-13's "link back to the run": the decoration's `#run-{pk}` fragment now lands on a visually highlighted row.
- Test coverage: `MonthCellCampaignMarkerTest` (2 tests) proves both event loops render the marker without consuming the title budget; `DecorationSurvivalAndGuardsTest` (5 tests) proves the decoration survives a from-scratch title/description rewrite (ROADMAP criterion 3), the campaign-less guard, the pending-review guard for staff and anonymous alike, the PII absence guard, and the N+1 guard; `TestCampaignRunRowAnchor` (3 tests) proves the anchor id for staff and anonymous readers and the null-pk guard directly on the callable.

## Task Commits

Each task was committed atomically, with one documented exception (see Deviations):

1. **Task 1: Month-cell campaign marker, rendered from the link and prefetched** - `dc635d6` (feat)
2. **Task 2: Give the decoration's link a landing spot — anchored, highlighted run rows on the campaign table** - `adb8468` (feat)
3. **Task 3: Prove the decoration is display-time only, guarded, and query-cheap** - covered by `dc635d6` (see Deviations)

**Plan metadata:** (this commit)

## Files Created/Modified

- `src/templates/tom_calendar/partials/calendar.html` - `.cal-campaign-chip` rule; marker span in both month-cell event loops
- `solsys_code/views.py` - `Prefetch('telescope_label_meta', queryset=CalendarEventMeta.objects.select_related('run__campaign'))` in `fomo_render_calendar`
- `solsys_code/campaign_tables.py` - `_campaign_run_row_id()` helper; `CampaignRunTable.Meta.row_attrs`
- `src/templates/campaigns/campaignrun_table.html` - `tr:target` highlight rule
- `solsys_code/tests/test_calendar_template.py` - `MonthCellCampaignMarkerTest`, `DecorationSurvivalAndGuardsTest`
- `solsys_code/tests/test_campaign_views.py` - `TestCampaignRunRowAnchor`

## Decisions Made

- Used a single flag glyph (U+2691 BLACK FLAG, `&#9873;`) as the marker's visible content rather than adding a new color-coded chip -- keeps the chip's own styling free of any competing background/border, per the plan's explicit prohibition on the chip competing with the proposal fill or the entry's existing status/verification decorations.
- Wrote `.cal-campaign-chip`'s explanatory comment *before* the CSS rule (not inside it) after discovering the plan's own verify script parses the rule body by splitting the file on the class name and reading up to the next `}` -- a comment mentioning `{{ text_color }}` inside the rule body prematurely closed that scan at the comment's own embedded brace. Same reasoning for placing the `row_attrs` null-pk explanatory comment immediately after (not before) the `row_attrs = {...}` assignment in `campaign_tables.py`. Neither change altered the actual rule/behavior, only comment placement.

## Deviations from Plan

### Process deviation (commit-atomicity, not a Rule 1-4 code deviation)

**1. Task 3's test class was committed together with Task 1, not as its own atomic commit**
- **Found during:** authoring `solsys_code/tests/test_calendar_template.py`
- **Issue:** Task 1's and Task 3's test additions were written together in a single file edit while validating the month-cell marker and Prefetch implementation end-to-end (both needed the same fixtures and the same underlying `campaign_decoration()`/`Prefetch` mechanism to exercise), and the whole file was staged and committed as part of Task 1's `dc635d6` commit before Task 3 was reached as a separate step. This mirrors the precedent already documented in 33-01's SUMMARY ("TDD Gate Compliance"): Task 3 carries `tdd="true"` but its own action note says "No production file changes in this task" -- the behavior it proves was already implemented by Task 1, so there is no separate RED/GREEN cycle to gate on, and no functional content is missing or incorrectly attributed. All five of Task 3's acceptance-criteria behaviors are proven by tests present in the tree (`DecorationSurvivalAndGuardsTest`), and the plan's full verify list (targeted tests, project full-suite command, ruff, ruff-format) was re-run and passed after Task 2 completed, i.e. against the final tree state including Task 3's tests.
- **Fix:** None needed functionally; documented here rather than restructuring git history (this workflow creates new commits, never rewrites existing ones).
- **Files affected:** `solsys_code/tests/test_calendar_template.py` (already listed above)
- **Verification:** `python manage.py test solsys_code.tests.test_calendar_template solsys_code.tests.test_campaign_views` (98 tests, OK); project full-suite command (968 + 40 tests, OK)
- **Commit:** `dc635d6` (Task 1's commit; no separate Task 3 commit exists)

---

**Total deviations:** 1 process deviation (commit atomicity), 0 auto-fixed code deviations.
**Impact on plan:** No functional impact -- every task's acceptance criteria and the plan's full verification list pass against the final committed tree. The only effect is that "3 tasks -> 3 commits" reads as "3 tasks -> 2 commits" in this plan's git history.

## Issues Encountered

None beyond the process deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ROADMAP criterion 3 (campaign decoration visible in the month cell and the modal, survives a from-scratch rewrite of the event's own fields) now holds end to end for both display surfaces.
- ANNOT-02 is functionally satisfied by this plan's code and tests; `REQUIREMENTS.md` itself stays `Pending` until sibling plan 33-05 (which also declares ANNOT-02) produces its own SUMMARY -- the shared-ID gate in `execute-plan.md`'s `update_requirements` step will flip it to Complete automatically once that happens.
- The decoration's campaign-table link now has a real, highlighted landing row (`run-{pk}` + `tr:target`), closing D-13 for both staff and public readers.
- No blockers for 33-04 or 33-05.

---
*Phase: 33-series-identity-reconciler-inversion*
*Completed: 2026-09-04*

## Self-Check: PASSED

All 6 modified files verified present on disk; both task commit hashes (`dc635d6`, `adb8468`) verified present in git log. Full acceptance-criteria and `<verify>` command re-run: `python manage.py test solsys_code.tests.test_calendar_template solsys_code.tests.test_campaign_views` (98 tests, OK); the project full-suite command (`LABELS=...` set, 968 tests, OK, plus `TestSplitNumberUnitRegex`/`TestJPLSBDBQuery`, 40 tests, OK); `pre-commit run ruff --all-files` and `ruff-format --all-files` (both Passed); all four plan-authored grep/python verify one-liners re-run and matched their expected non-failure conditions.
