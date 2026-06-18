---
phase: quick-260618-mck
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/templates/tom_calendar/partials/calendar.html
autonomous: true
requirements: [QUEUED-CONTRAST]
subsystem: calendar-ui
tags: [django-templates, tom_calendar, ui, css]

must_haves:
  truths:
    - "The [QUEUED] all-day event box is clearly legible on a white in-month day cell."
    - "The [QUEUED] all-day event box is clearly legible on the light-gray (#f8f9fa) other-month / overflow day cell."
    - "The [QUEUED] event still looks visibly LESS prominent than a solid event.color confirmed/placed block."
    - "No var(--...) Bootstrap color token appears in the [QUEUED] style block."
  artifacts:
    - path: "src/templates/tom_calendar/partials/calendar.html"
      provides: "Updated [QUEUED] de-emphasis style with sufficient contrast against both white and #f8f9fa backgrounds"
      contains: "[QUEUED] "
  key_links:
    - from: "calendar.html [QUEUED] branch"
      to: ".cal-event-all-day color:#fff !important rule (same file)"
      via: "fill dark enough to carry the forced white text"
      pattern: "background-color: rgba\\(0, 0, 0"
---

<objective>
Fix the contrast/legibility of the `[QUEUED]` all-day calendar event de-emphasis style introduced by quick task 260618-lw4. The current style (`background-color: rgba(0, 0, 0, 0.06); border: 1px dashed rgba(0, 0, 0, 0.35);`) is too weak: `.cal-event-all-day` forces `color: #fff !important` on the title text, and a 0.06-alpha black fill over both white in-month cells and the light-gray other-month overflow cells leaves white text on a near-white background — nearly invisible on the overflow days.

Purpose: The `[QUEUED]` box must read clearly on BOTH day-cell backgrounds while remaining visibly de-emphasized relative to a solid `event.color` block.
Output: One modified Django template file, change confined to the same `[QUEUED]` conditional branch.

## Discovered facts (do NOT re-investigate)

- Other-month / overflow day cells: `.cal-day` -> `background-color: var(--light)`. `--light` is defined in `tom_common/static/tom_common/css/main.css` and `dark.css` as `#f8f9fa` (light gray, identical in both). This is the gray the user reported the box vanishing against.
- In-month day cells: `.cal-day.is-current-month` -> `background-color: transparent` (renders on white page background).
- `.cal-event-all-day` (and its `a`) force `color: #fff !important;` — the title text is ALWAYS white. This is the root cause: a near-transparent fill cannot support white text.
- The existing `[QUEUED]` inline style is the ONLY divergence from upstream; the else branch keeps `style="background-color: {{ event.color }};"` (solid saturated color) for confirmed/placed events.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260618-lw4-de-emphasize-queued-calendar-events-so-t/260618-lw4-SUMMARY.md
@src/templates/tom_calendar/partials/calendar.html
</context>

<tasks>

<task type="auto">
  <name>Task 1: Strengthen the [QUEUED] de-emphasis style for contrast on white AND #f8f9fa</name>
  <files>src/templates/tom_calendar/partials/calendar.html</files>
  <action>
Edit ONLY the inline `style` attribute on the `[QUEUED]` branch `cal-event-all-day` div (currently `style="background-color: rgba(0, 0, 0, 0.06); border: 1px dashed rgba(0, 0, 0, 0.35);"`, inside the `{% if event.title|slice:":9" == "[QUEUED] " %}` block).

Because `.cal-event-all-day` forces `color: #fff !important`, the fill must be dark enough that white text is legible on it, while sitting on both a transparent/white cell AND the `#f8f9fa` other-month cell. Replace the fill with a medium-dark muted neutral, and make the border solid (not dashed) at higher opacity so the box edge reads against the gray overflow cell. Keep it visibly MUTED/gray — never a solid saturated `event.color` — so it stays de-emphasized relative to confirmed/placed blocks.

Set the inline style to exactly: `style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);"`

Rationale to keep in mind: rgba(0,0,0,0.45) composites to ~#8c8c8c over white and ~#898a8b over #f8f9fa — a mid-gray giving the #fff title strong contrast on BOTH cell types, yet plainly a desaturated gray box, not a vivid event.color block, so the 260618-lw4 de-emphasis intent still holds. The solid 0.55-alpha border (replacing the dashed 0.35) makes the outline visible against the light-gray overflow cell.

Constraints (carried from 260618-lw4 — must still hold):
- Do NOT revert to solid `event.color` for `[QUEUED]`; it must remain clearly less prominent.
- Use ONLY literal hex/rgba values — NO `var(--purple)`, `var(--bs-purple)`, or any `var(--...)` token (forward-compat with tomtoolkit 3.0.0a9 -> 3.0.0a10 bootstrap4 -> bootstrap5 var rename).
- Touch ONLY this one inline `style` attribute. Do not modify the `<style>` block, the else branch, the timed-event block, any other part of the template, or any `.py` file. Do not expand into the telescope/proposal-keyed coloring scheme (separate deferred backlog item, pending todo 2026-06-18-status-aware-calendar-event-coloring).
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel && git --no-pager diff src/templates/tom_calendar/partials/calendar.html</automated>
    Inspect the diff: it MUST show exactly one changed line, the [QUEUED]-branch cal-event-all-day div, with the new `rgba(0, 0, 0, 0.45)` fill and `1px solid rgba(0, 0, 0, 0.55)` border. No other hunks.
    Guard against re-introduced tokens: `grep -n 'var(--' src/templates/tom_calendar/partials/calendar.html` must show only the pre-existing `.cal-day` `<style>` rules (none inside the [QUEUED] inline style).
  </verify>
  <done>The [QUEUED] branch div uses `background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);`; the diff is confined to that single line; no var(--) token appears in any inline event style; the else branch, timed-event block, and all other files are unchanged.</done>
</task>

</tasks>

<verification>
- `git --no-pager diff src/templates/tom_calendar/partials/calendar.html` shows a single-line change in the `[QUEUED]` branch only.
- New fill `rgba(0, 0, 0, 0.45)` and solid border `rgba(0, 0, 0, 0.55)` present; old `0.06` / dashed `0.35` values gone.
- No `var(--...)` token in any inline event style.
- Manual render check (note for the operator): open the calendar month view from the screenshot. Confirm the `[QUEUED] FTS 2M0-...` box is clearly legible (white title on a mid-gray box) on BOTH (a) a normal white in-month day cell, and (b) a light-gray (#f8f9fa) other-month overflow day cell at the end of the month — and that it still looks distinctly more muted than a solid colored confirmed/placed event block.
</verification>

<success_criteria>
- `[QUEUED]` all-day events are legible on white and on #f8f9fa overflow-day cells.
- They remain visibly de-emphasized vs solid `event.color` events.
- Change is one line, confined to the existing `[QUEUED]` style block; no `.py` or other template regions touched; no `var(--...)` color tokens.
</success_criteria>

<output>
Create `.planning/quick/260618-mck-fix-insufficient-contrast-in-queued-cale/260618-mck-SUMMARY.md` when done.
</output>
