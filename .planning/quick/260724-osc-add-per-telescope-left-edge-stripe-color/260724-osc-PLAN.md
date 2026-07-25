---
phase: quick-260724-osc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/test_calendar_display_extras.py
  - src/templates/tom_calendar/partials/calendar.html
  - solsys_code/tests/test_calendar_template.py
autonomous: true
requirements:
  - QUICK-260724-osc

must_haves:
  truths:
    - "Classical-schedule (empty-proposal) all-day events render a per-telescope colored left-edge stripe over an unchanged neutral-gray fill."
    - "Two different telescopes render two different stripe colors; the same telescope always renders the same color."
    - "Proposal-having events render NO telescope stripe (proposal fill already encodes identity)."
    - "A display-only legend section decodes the visible telescope stripe colors, without hooking into the click-to-filter JS."
  artifacts:
    - "telescope_color simple_tag in calendar_display_extras.py"
    - "visible_classical_telescopes simple_tag in calendar_display_extras.py"
    - "border-left stripe wiring in calendar.html all-day branch"
    - "telescope legend section in calendar.html"
  key_links:
    - "template compares bg_color to NEUTRAL_SLOT_COLOR to decide classical-vs-proposal, reusing the already-computed proposal_color result"
    - "telescope_color and proposal_color share PROPOSAL_PALETTE + sha256 normalization"
---

<objective>
Add per-telescope left-edge stripe coloring to classical-schedule calendar events so
telescopes are visually distinguishable in month view, without changing the existing
proposal-color encoding or the click-to-filter behavior.

Purpose: Every `load_telescope_runs` event currently collapses into one flat-gray
"Classical schedule" bucket, making telescopes indistinguishable. A colored left-edge
stripe keyed by telescope (gray fill unchanged underneath) restores per-telescope
identity while composing with the existing proposal-fill / status-ring visual language.
Output: Two new template tags, template stripe + legend wiring, and full unit +
integration test coverage.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@solsys_code/templatetags/calendar_display_extras.py
@src/templates/tom_calendar/partials/calendar.html
@solsys_code/tests/test_calendar_display_extras.py
@solsys_code/tests/test_calendar_template.py
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Add telescope_color + visible_classical_telescopes tags with unit tests</name>
  <files>solsys_code/templatetags/calendar_display_extras.py, solsys_code/tests/test_calendar_display_extras.py</files>
  <behavior>
    telescope_color(telescope):
    - Test: deterministic — same telescope string returns the same color.
    - Test: normalization — '.strip().upper()' applied, so casing/whitespace variants share a color.
    - Test: blank/None/whitespace-only telescope returns NEUTRAL_SLOT_COLOR (defensive fallback, mirrors proposal_color).
    - Test: non-empty telescope returns a member of PROPOSAL_PALETTE.
    visible_classical_telescopes(weeks):
    - Test: only classical-schedule events (empty proposal) contribute entries; proposal-having events are excluded even when their telescope field is set.
    - Test: entries are grouped by resulting color so hash-colliding telescopes share one legend entry (mirror VisibleProposalsTest collision handling — build expected mapping dynamically).
    - Test: telescopes absent from the visible weeks do not appear.
    - Test: supports both dict-based day objects and attribute-based (SimpleNamespace) stubs, mirroring visible_proposals.
  </behavior>
  <action>
Add a `telescope_color(telescope: str) -> str` @register.simple_tag reusing PROPOSAL_PALETTE
and the exact normalization/hashing approach of proposal_color: `(telescope or '').strip().upper()`,
return NEUTRAL_SLOT_COLOR when empty, else index PROPOSAL_PALETTE by
`int(hashlib.sha256(normalized.encode()).hexdigest(), 16) % len(PROPOSAL_PALETTE)`. Do NOT
invent a second palette. Cite the defensive-fallback rationale in the docstring (template tags
must not raise on unexpected data) and note it mirrors proposal_color's own pattern.

Add a `visible_classical_telescopes(weeks) -> list[dict]` @register.simple_tag mirroring
visible_proposals: same weeks iteration, same dict-vs-attribute dual support, same
group-by-color collision handling. Scope it to classical-schedule events only — include an
event only when its normalized proposal is empty (`(event.proposal or '').strip().upper() == ''`),
and key the grouping on `telescope_color(event.telescope)` with the label taken from the
normalized telescope string. Return the same dict shape visible_proposals returns
(keys 'color', 'codes', 'label'); no forced-last neutral entry is needed here since every
included event is classical. Add a new test class for each tag in the test file, mirroring the
structure of the existing ProposalColorTest and VisibleProposalsTest classes. Reuse the
existing `_make_weeks` helper pattern (extend or add a sibling helper that also sets a
`telescope` attribute and a blank/non-blank `proposal` on each SimpleNamespace event).
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo && ./manage.py test solsys_code.tests.test_calendar_display_extras 2>&1 | tail -20</automated>
  </verify>
  <done>New telescope_color and visible_classical_telescopes tags exist and all new + existing tests in test_calendar_display_extras.py pass. `ruff check solsys_code/templatetags/calendar_display_extras.py solsys_code/tests/test_calendar_display_extras.py` is clean.</done>
</task>

<task type="auto">
  <name>Task 2: Wire the stripe + legend into calendar.html with integration tests</name>
  <files>src/templates/tom_calendar/partials/calendar.html, solsys_code/tests/test_calendar_template.py</files>
  <action>
In `src/templates/tom_calendar/partials/calendar.html`, in the `{% for event in day.all_day_events %}`
loop (currently ~lines 177-198) ONLY (leave the `{% for event in day.events %}` timed loop
untouched — confirmed classical events are always all-day midnight-spanning banners, never timed):

1. Expose the neutral-slot constant to the template without a magic literal: add a tiny
   assignment simple_tag in calendar_display_extras.py (e.g. `neutral_slot_color`) that returns
   NEUTRAL_SLOT_COLOR, and in the template do `{% neutral_slot_color as neutral_color %}`.
   (Do NOT re-derive "is classical" a different way — compare the already-computed `bg_color`
   to `neutral_color`.) If you add this tag, add a one-line unit test for it in Task 1's test
   file before wiring — but since Task 1 is already committed, it is acceptable to add the tag
   here and its test alongside; keep ruff clean either way.
2. When `bg_color == neutral_color`, compute `{% telescope_color event.telescope as tel_color %}`
   and add `border-left: 4px solid {{ tel_color }};` INTO the existing `.cal-event-all-day` div's
   inline `style` attribute — in BOTH the `is_verified == False` (dashed) branch and the else
   branch — composed alongside the existing `background-color`/`color`/`{{ status_border }}`
   (and the dashed unverified border where present) so none clobbers another. When
   `bg_color != neutral_color` (real proposal), emit NO border-left telescope stripe.
   Use a `{% if bg_color == neutral_color %}...{% endif %}` guard around just the border-left
   fragment inside the style attribute, so the rest of the style string is unchanged for
   proposal-having events.

3. Render the telescope legend near the existing proposal legend (~lines 239-245): call
   `{% visible_classical_telescopes weeks as telescope_legend %}` and, only when non-empty,
   render a small additional legend section (its own row/grouping) reusing the same swatch
   visual (colored `▌` + `<small>` label) as the proposal legend. This section is DISPLAY-ONLY:
   do NOT add a `data-proposal` attribute and do NOT add `.cal-legend-swatch` (which the JS
   click handler keys on) — use a distinct, non-interactive class (e.g. `cal-legend-telescope`)
   so it cannot hook into the click-to-filter JS at the bottom of the file. Do NOT touch that
   `<script>` block.

In `solsys_code/tests/test_calendar_template.py`, add integration tests mirroring the existing
CalendarTemplateTest conventions (use `self.client.get(reverse('calendar:calendar'), ...)`,
decode `response.content`, import `telescope_color`):
- Create a classical-schedule all-day fixture (`proposal=''`, `telescope='NTT'`, midnight-spanning
  start/end in June 2026 on a free date) and assert the rendered `.cal-event-all-day` div includes
  `border-left: 4px solid {telescope_color('NTT')};`.
- Assert a proposal-having all-day event's rendered div does NOT include any
  `border-left: 4px solid` telescope-color declaration (confirms the conditional scoping).
- Assert the telescope legend section renders (e.g. the `cal-legend-telescope` class appears)
  when a classical event is visible.
Keep all existing dashed-border / status-ring COUNT assertions passing — pick fixture dates that
do not perturb `num_fallback_day_cell_occurrences` (no CalendarEventTelescopeLabel with
is_verified=False on the new fixtures unless you also update that count).
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo && ./manage.py test solsys_code.tests.test_calendar_template solsys_code.tests.test_calendar_display_extras 2>&1 | tail -25</automated>
  </verify>
  <done>Classical all-day events render a per-telescope `border-left: 4px solid <hex>;` stripe; proposal-having events do not; a display-only telescope legend renders when classical events are visible and is NOT wired to the click-to-filter JS. All new + existing tests in both test modules pass. `ruff check .` and `ruff format --check .` are clean.</done>
  </task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CalendarEvent.telescope → template style attribute | DB string flows into an inline CSS `style` value |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-osc-01 | Injection (Tampering) | telescope_color output in style attr | low | mitigate | telescope_color never echoes the raw telescope string — it returns only a fixed PROPOSAL_PALETTE hex or NEUTRAL_SLOT_COLOR selected by hash, so no untrusted content reaches the style attribute (same mitigation proposal_color already relies on). |
| T-osc-02 | Denial of Service | template tag on unexpected data | low | mitigate | Defensive blank/None fallback returns NEUTRAL_SLOT_COLOR so the tag cannot raise during render. |
</threat_model>

<verification>
- `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` passes (new + existing).
- `ruff check .` and `ruff format --check .` are clean.
- No changes to the click-to-filter `<script>` block, the `data-proposal` wiring, `visible_proposals()`, the timed-event loop, or any of the four demo-notebook-paired modules.
</verification>

<success_criteria>
- Classical-schedule all-day events show a deterministic per-telescope left-edge stripe over unchanged gray fill.
- Proposal-having events show no telescope stripe.
- A display-only telescope legend decodes the visible stripe colors and does not participate in click-to-filter.
- All new and existing tests pass; ruff clean.
</success_criteria>

<output>
Create `.planning/quick/260724-osc-add-per-telescope-left-edge-stripe-color/260724-osc-SUMMARY.md` when done.
</output>
