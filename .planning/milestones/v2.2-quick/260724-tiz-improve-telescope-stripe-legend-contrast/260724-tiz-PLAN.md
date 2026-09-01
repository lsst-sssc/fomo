---
phase: quick-260724-tiz
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
requirements: []
must_haves:
  truths:
    - "telescope_color() returns a member of the new brighter TELESCOPE_PALETTE, not PROPOSAL_PALETTE"
    - "Classical-schedule all-day events render a 6px per-telescope stripe with a light seam via a CSS pseudo-element (no inline border-left)"
    - "The stripe/seam does not collide with the status-ring box-shadow (separate DOM elements)"
    - "Proposal-having all-day events render no stripe (no cal-event-classical class, no --tel-color)"
    - "Both legends render bigger .cal-legend-chip swatches instead of the thin ▌ glyph"
    - "Proposal legend click-to-filter behavior (data-proposal / cal-legend-swatch) is unchanged"
  artifacts:
    - solsys_code/templatetags/calendar_display_extras.py
    - src/templates/tom_calendar/partials/calendar.html
  key_links:
    - "telescope_color() hashing → TELESCOPE_PALETTE index"
    - "template --tel-color custom property → .cal-event-classical::before var(--tel-color)"
---

<objective>
Follow-up contrast fix to the per-telescope stripe/legend shipped in quick-260724-osc. The
stripe reused the deliberately-dark PROPOSAL_PALETTE, giving 1.02–1.72:1 contrast against the
gray classical fill (effectively invisible). Switch telescope_color() to a pre-validated
brighter palette, re-implement the stripe (plus a light seam) via a CSS pseudo-element to avoid
a box-shadow property collision with the status ring, widen it to 6px, and enlarge both legend
swatches into filled rounded chips.

Purpose: make the per-telescope color actually distinguishable on the calendar without widening
the blast radius (the shared gray fill and proposal palette are untouched, per the user's
explicit rejection of the "lighten the gray fill" option).

Output: updated telescope_color()/palette in calendar_display_extras.py, updated stripe + legend
markup and CSS in calendar.html, and updated existing tests in both test modules.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@solsys_code/templatetags/calendar_display_extras.py
@src/templates/tom_calendar/partials/calendar.html
@solsys_code/tests/test_calendar_display_extras.py
@solsys_code/tests/test_calendar_template.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add TELESCOPE_PALETTE, point telescope_color() at it, update unit tests</name>
  <files>solsys_code/templatetags/calendar_display_extras.py, solsys_code/tests/test_calendar_display_extras.py</files>
  <read_first>
solsys_code/templatetags/calendar_display_extras.py lines 28-42 (PROPOSAL_PALETTE / NEUTRAL_SLOT_COLOR)
and 219-240 (telescope_color). solsys_code/tests/test_calendar_display_extras.py lines 12-23 (imports)
and 202-230 (TelescopeColorTest).
  </read_first>
  <action>
In calendar_display_extras.py, add a new module-level constant TELESCOPE_PALETTE immediately after
PROPOSAL_PALETTE (leave PROPOSAL_PALETTE untouched — it stays owned by proposal_color). Use exactly
these 8 hex values, in this order, verbatim (already validated via the dataviz skill's palette
validator against the #5a6268 gray fill — do NOT re-derive or substitute):
['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'].
Add a short comment noting these are the brighter dark-surface categorical set chosen for
against-gray-fill stripe contrast (measured ~1.3–2.0:1, a ~50-70% improvement over the reused dark
palette), legal as a sub-3:1 WARN because the telescope name always carries the identity in the
event title and legend label (never color-alone).

Update telescope_color() (currently ~lines 219-240): change the final return to index into
TELESCOPE_PALETTE instead of PROPOSAL_PALETTE. Keep the .strip().upper() normalization, the
NEUTRAL_SLOT_COLOR blank/None fallback, and the hashlib.sha256(...).hexdigest() hashing exactly as
they are — only the palette constant referenced changes. Fix the docstring: the two lines that
currently reference PROPOSAL_PALETTE (the summary line and the Returns line saying "A hex color
string from PROPOSAL_PALETTE") must name TELESCOPE_PALETTE instead.

In test_calendar_display_extras.py: add TELESCOPE_PALETTE to the import block (lines 12-23,
alphabetically near PROPOSAL_PALETTE). Update test_nonempty_telescope_returns_palette_member (~line
227-230) so it asserts telescope_color('NTT') is in TELESCOPE_PALETTE (not PROPOSAL_PALETTE). Leave
all proposal_color / PROPOSAL_PALETTE tests unchanged. The VisibleClassicalTelescopesTest cases
compute expected colors dynamically via telescope_color(), so they need no change — confirm they
still pass after the palette swap.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_calendar_display_extras 2>&1 | tail -5</automated>
  </verify>
  <done>
TELESCOPE_PALETTE exists with the 8 specified hexes; telescope_color() returns members of
TELESCOPE_PALETTE; docstring no longer references PROPOSAL_PALETTE; test module passes with the
membership assertion updated; PROPOSAL_PALETTE / proposal_color untouched.
  </done>
</task>

<task type="auto">
  <name>Task 2: Re-implement stripe+seam via pseudo-element and enlarge legend chips in calendar.html, update integration tests</name>
  <files>src/templates/tom_calendar/partials/calendar.html, solsys_code/tests/test_calendar_template.py</files>
  <read_first>
src/templates/tom_calendar/partials/calendar.html lines 1-130 (the &lt;style&gt; block), 184-207
(all-day event loop with the two inline border-left fragments), and 248-261 (both legend loops).
solsys_code/tests/test_calendar_template.py lines 308-360 (the quick-260724-osc test section).
  </read_first>
  <action>
CSS (in the &lt;style&gt; block near .cal-event-all-day, ~lines 79-97): add a .cal-event-classical
class that draws the stripe via a ::before pseudo-element so the light seam lives on its own DOM
element and cannot collide with the status-ring box-shadow that status_border_css emits on the same
inline style attribute. The parent rule sets position: relative and padding-left: 9px (overriding
the base .cal-event-all-day padding: 1px 5px so title text clears the wider stripe). The ::before
rule sets content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background-color:
var(--tel-color); box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.55); border-radius: 3px 0 0 3px;
— the inset box-shadow is the light seam between the stripe and the gray fill.

Also add a shared .cal-legend-chip class: display: inline-block; width: 12px; height: 12px;
border-radius: 3px; margin-right: 2px; vertical-align: middle. Do NOT modify .cal-legend-swatch or
.cal-legend-telescope (both still needed for the JS filter hook and the un-clickable telescope
legend respectively).

Template — all-day loop (~lines 195-200): both the is_verified == False (dashed) branch and the
plain branch currently carry an inline
{% if bg_color == neutral_color %} border-left: 4px solid {{ tel_color }};{% endif %} fragment.
Replace BOTH fragments: remove the inline border-left, and instead (a) conditionally add the
cal-event-classical class to the .cal-event-all-day div's class attribute when bg_color ==
neutral_color, and (b) conditionally add a --tel-color: {{ tel_color }}; declaration to that div's
inline style attribute when bg_color == neutral_color. A CSS custom property composes with the
existing background-color / color / status_border / dashed-border declarations (different property
name, no collision). When bg_color != neutral_color, add neither the class nor the custom property
(no stripe, exactly as before).

Template — legend loops (~lines 249-260): in BOTH the proposal legend loop and the telescope legend
loop, replace the inner &lt;span style="color: {{ entry.color }};"&gt;▌&lt;/span&gt; swatch glyph with
&lt;span class="cal-legend-chip" style="background-color: {{ entry.color }};"&gt;&lt;/span&gt;. Leave the
outer wrapper spans (.cal-legend-swatch with role/aria/data-proposal on the proposal legend, and
.cal-legend-telescope on the telescope legend) and the &lt;small&gt; labels exactly as they are — only
the innermost swatch markup changes.

Tests — test_calendar_template.py (quick-260724-osc section, ~lines 320-360):
- Update test_osc_classical_event_renders_telescope_stripe: instead of asserting
  'border-left: 4px solid {tel_hex};', assert the scoped event div now carries the cal-event-classical
  class AND a --tel-color: {tel_hex}; custom property (assert against the actual rendered inline style,
  not a guessed string — read the rendered output if unsure of exact spacing).
- Update test_osc_proposal_having_event_has_no_telescope_stripe: assert the proposal-having event divs
  do NOT contain 'cal-event-classical' and do NOT contain '--tel-color' (mirrors the old
  no-border-left scoping intent for the new markup).
- Add a test asserting both legend loops render .cal-legend-chip spans carrying the correct
  background-color (e.g. the telescope legend chip uses telescope_color('NTT'); a proposal legend chip
  uses a proposal_color of a visible fixture proposal).
- Leave test_osc_telescope_legend_renders_when_classical_event_visible and
  test_osc_telescope_legend_is_not_click_to_filter_wired passing (the wrapper spans are unchanged);
  fix them only if the chip change incidentally breaks an assertion.
Run the module and fix every failure — do not leave any test red or skipped.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_calendar_template 2>&1 | tail -5</automated>
  </verify>
  <done>
The gray-fill classical all-day event renders with the cal-event-classical class and a --tel-color
custom property (no inline border-left anywhere); proposal-having all-day events carry neither; the
::before stripe+seam and .cal-legend-chip CSS exist; both legends render .cal-legend-chip swatches;
proposal legend click-to-filter markup (cal-legend-swatch/data-proposal) is unchanged; the test
module passes with updated + added assertions.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CalendarEvent.telescope/.proposal → rendered HTML | model string fields flow into the calendar template |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-tiz-01 | Injection (Tampering) | telescope_color / --tel-color | low | mitigate | Color emitted into style is a fixed TELESCOPE_PALETTE constant (or NEUTRAL_SLOT_COLOR), never the raw telescope string; the raw string is used only as a hash input, same mitigation as quick-260724-osc T-osc-02. |
| T-tiz-02 | Tampering | dependency installs | low | accept | No new packages installed; CSS/Python-constant change only. |
</threat_model>

<verification>
Run both affected Django test modules together (avoids collecting ephem tests that trigger the
~1.6 GB SPICE download):
`./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template`
Then quality gates: `ruff check .` and `ruff format --check .` must stay clean.
</verification>

<success_criteria>
- telescope_color() draws from TELESCOPE_PALETTE (8 specified brighter hexes); PROPOSAL_PALETTE and proposal_color unchanged.
- Classical all-day events render a 6px pseudo-element stripe + light seam via cal-event-classical + --tel-color; no inline border-left remains.
- Stripe/seam and status-ring box-shadow do not collide (separate DOM elements).
- Proposal-having events render no stripe.
- Both legends render .cal-legend-chip swatches; proposal click-to-filter wiring unchanged.
- Both test modules pass; ruff clean.
</success_criteria>

<output>
Create `.planning/quick/260724-tiz-improve-telescope-stripe-legend-contrast/260724-tiz-SUMMARY.md` when done
</output>
