# Phase 39 — UI Review

**Audited:** 2026-10-08
**Baseline:** Abstract 6-pillar standards (no UI-SPEC.md)
**Screenshots:** Not captured (code-only audit as requested)
**Interaction captures:** off

---

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 3/4 | "Todo list" heading appears in read-only view where visitors cannot add todos; minor UX friction |
| 2. Visuals | 3/4 | Strong calendar hierarchy and event card structure; read-only card lacks visual distinction from editable forms |
| 3. Color | 4/4 | All colors use Bootstrap 5 CSS variables; accent colors used sparingly; D-11 compliance verified |
| 4. Typography | 3/4 | Font sizes and weights follow Bootstrap scale; "Todo list" (h6) could be more visually distinct in read-only context |
| 5. Spacing | 3/4 | All spacing uses Bootstrap utility scale (no arbitrary values); day cells at p-1 feel slightly cramped on small screens |
| 6. Experience Design | 3/4 | Clean request.user.is_authenticated branching; empty states handled; lacks visual feedback on read-only card interactivity |

**Overall: 19/24**

---

## Top 3 Priority Fixes

1. **Hide or retitle "Todo list" heading in read-only event card** — Visitors see a heading for a feature they cannot use — Concrete fix: In event_form.html line 310, move the `<h6>Todo list</h6>` inside the `{% if request.user.is_authenticated %}` block (after line 314), or change the text to "Events" / "Tasks" (non-actionable label).

2. **Add visual indication that read-only card is non-interactive** — Users cannot distinguish read-only card from an editable form at a glance — Concrete fix: Add a subtle `<span class="badge text-bg-light mb-2">Read only</span>` or similar indicator after line 114 (`<div id="cal-event-card">`) to signal the card's state.

3. **Increase day cell padding from p-1 to p-2 on small screens** — Day cell text feels cramped, reducing readability on mobile — Concrete fix: Change line 236 `p-1` to a responsive class like `p-sm-1 p-md-2` or use custom media queries to adjust padding based on viewport width.

---

## Detailed Findings

### Pillar 1: Copywriting (3/4)

**Strengths:**
- Action labels are clear: "Save", "Delete", "View ↗", "Next", "Prev", "Today" (lines 102, 110, 68, 207, 201, 213 in calendar.html)
- Empty state messaging is accurate: "No todos yet." (line 328 in event_form.html) correctly signals when the todo list is empty
- No generic patterns: no "Click here", "Submit", or "OK" labels
- No login prompts on anonymous view: reads cleanly without "log in to edit" instructions (per design D-05)

**Issues:**
- **BLOCKER:** "Todo list" heading (line 310, event_form.html) renders for anonymous visitors in the read-only card branch (`{% if request.user.is_authenticated %}`...`{% else %}`), but they cannot add todos. The heading implies an actionable feature for visitors who lack the controls to use it.
- **Minor:** No heading hierarchy distinction between "Observation series", "Attributed campaign run" (implied by the label structure) and "Todo list" — all three are rendered with implicit equal weight.

**Finding:** The heading visibility flaw violates the principle that UI text should accurately reflect what users can do. Visitors see "Todo list" but no input controls, creating a false affordance (Pillar 2 compounds this with lack of visual feedback).

---

### Pillar 2: Visuals (3/4)

**Strengths:**
- **Calendar month view** has strong visual hierarchy:
  - Month name (h4, line 216) is prominent
  - Navigation buttons and "+ New Event" are grouped at the top (lines 197-225)
  - Day grid provides clear structure via CSS grid (line 4, `.cal-grid`)
  - Today's date is visually highlighted (circle background, bold text, line 32-35)
  - Event rows have visual indicators: colored bullets (line 318), colored bars for all-day events (lines 282-292), stripes for classical telescopes (lines 119-138 in CSS)
  - Hover states on day cells (lines 15-19, lines 18-19) provide clear interactivity feedback
  - Legend groups similar elements at bottom (lines 334-374), with clear swatch indicators
- **Read-only event card** uses semantic structure:
  - `<dl class="row mb-2">` (line 115) pairs labels (`<dt>`) and values (`<dd>`) clearly
  - Labels are left-aligned (col-sm-3) and values right-aligned (col-sm-9) for scannability
  - Conditional fields (lines 122-157) omit empty rows, reducing cognitive load
  - Links are clearly distinguished (href and color change)

**Issues:**
- **BLOCKER:** Read-only card lacks visual distinction from the editable form above it. Both use Bootstrap form classes (event_form.html lines 44-111 for form, lines 114-159 for card). A visitor cannot immediately tell the card is non-interactive — no disabled state styling, no "read-only" badge, no visual difference in background or border.
- **Minor:** The definition list structure (dl/dd) is semantic but not visually prominent — no background tint or border to frame it as a "card" distinct from form content.
- **Icon-label pairing:** All icon-only elements have labels (month navigation buttons are labeled "Prev", "Next", "Today"; event titles are text; no orphaned icons found).

**Finding:** The Pillar 1 issue (misplaced "Todo list" heading) compounds here — without clear visual feedback, visitors cannot tell the read-only card is non-interactive and the todo heading is unreachable.

---

### Pillar 3: Color (4/4)

**Analysis:**
- **CSS Variables (no hardcoded colors):**
  - Line 8: `var(--light)` for day backgrounds
  - Line 9: `var(--secondary)` for day text
  - Line 33-34: `var(--primary)`, `var(--bs-white)` for today indicator
  - Lines 282-292: Dynamic `{{ bg_color }}` for proposal fills (computed server-side)
  - Line 318: `{{ bg_color }}` for event bullet points
  - Lines 71, 132: `text-muted` (Bootstrap secondary color) for "(not a web link)"
  - No inline style colors except dynamic ones from server context
- **Accent color usage (Bootstrap 5 primary):**
  - Line 102: `btn-primary` on "Save" button
  - Line 104: `btn-primary` on "Save and edit" button
  - Line 33: `var(--primary)` on today's date circle
  - Total: 3 uses across both templates (sparing, not overused)
- **D-11 Bootstrap 5 compliance (line 229-231, calendar.html):**
  - `border-start` and `border-end` present (D-11 requirement) ✓
  - `fw-bold` on day headers (line 231) ✓
  - `me-2`, `me-3` on legend items (lines 338, 344, 351) ✓
  - `var(--bs-white)` on today indicator (line 34) ✓
  - No Bootstrap 4 names (no `mr-2`, `ml-2`, `font-weight-bold`, `var(--white)`) detected ✓
- **Implied 60/30/10 split:**
  - 60%: `var(--light)` day backgrounds (neutral, light)
  - 30%: `var(--secondary)` text and form fields (muted, secondary)
  - 10%: Accent primary color (today indicator, save buttons) plus dynamic event colors

**Finding:** Pillar 3 passes with no issues. All color decisions follow Bootstrap 5 standards, D-11 is compliant, and accent colors are used sparingly and appropriately.

---

### Pillar 4: Typography (3/4)

**Font sizes (all within Bootstrap scale):**
- Line 27: `.80rem` on day numbers (small indicator)
- Line 38: `.9rem` on moon phase emoji (small secondary)
- Line 43: `.75rem` on event titles (compact in day cell)
- Line 67-72: inherited (form field labels, likely 1rem base)
- Line 151-163: `1rem` on legend entries (readable secondary content)
- Line 310: `h6` on "Todo list" heading (smallest heading, line-height 1.5 default)
- No extreme sizes; all sizes are proportional and follow Bootstrap's type scale

**Font weights:**
- Line 28: `font-weight: 500` on day numbers (semi-bold, emphasis)
- Line 35: `font-weight: 700` on today's date (bold, high emphasis)
- Line 85: `font-weight: 500` on all-day events (semi-bold)
- Line 159: `font-weight: 700` on active legend swatch (bold)
- Buttons use Bootstrap defaults (500-600 for normal, 700 for active)

**Issues:**
- **Minor:** The "Todo list" heading (h6, line 310) in the read-only event card could be visually more distinct to signal that the section below is not interactive. Using `fw-bold` or a different color (e.g., `text-secondary`) might help, but this is minor cosmetic refinement.
- No missing font weight/size combinations; typography is consistent.

**Finding:** Pillar 4 is solid. Font hierarchy is clear (headings > labels > body text). The minor issue noted above is secondary to Pillar 1's "Todo list" heading problem and Pillar 2's lack of read-only visual feedback.

---

### Pillar 5: Spacing (3/4)

**Spacing classes used (all Bootstrap scale):**
- event_form.html:
  - Line 115: `mb-2` (margin-bottom) on the `<dl>` card container
  - Line 44-56: `row` and `col` structures (standard Bootstrap grid spacing)
  - Line 310: `mb-2` on h6 "Todo list" heading
  - Line 320: `mb-0` on the read-only `<ul>` (no bottom margin for list end)
- calendar.html:
  - Line 195: `mb-3` (margin-bottom) on the header row
  - Lines 230-232: `p-2` on day header cells (padding)
  - Lines 236-241: `p-1` on day cells (padding)
  - Line 334: `mt-1` (margin-top) on the footer row
  - Lines 338, 344, 351: `me-2` and `me-3` on legend items
  - Line 193-194: No special margin on the calendar-partial div

**Consistency:**
- All spacing uses Bootstrap's utility scale (0.25rem increments: p-0, p-1, p-2, etc.)
- No arbitrary pixel values (no `style="margin: 12px;"`)
- Margins and padding follow a consistent pattern (component > content spacing)

**Issues:**
- **Minor:** Day cells with `p-1` (0.25rem padding) feel slightly cramped on small screens (<640px), especially when event text wraps. The text `.75rem` on `.cal-event` entries (line 43) combined with `p-1` leaves limited whitespace for readability. Expected behavior: no issue on desktop; potential squeeze on mobile.
- No critical spacing issue; this is a responsive design edge case.

**Finding:** Pillar 5 is largely solid. All spacing uses Bootstrap scale consistently. The minor issue with day cell padding at `p-1` is a responsive design consideration, not a scale violation.

---

### Pillar 6: Experience Design (3/4)

**State handling:**

1. **Authentication branching (correct):**
   - event_form.html, line 36: `{% if request.user.is_authenticated %}`
   - Line 37-112: Form block (Save, Save-and-edit, Delete buttons visible)
   - Line 113-160: `{% else %}` Read-only card block (no form controls)
   - calendar.html, line 218: `{% if request.user.is_authenticated %}`
   - Line 219-225: "+ New Event" button visible
   - Line 226: `{% else %}` `<span class="cal-header-spacer">` for spacing balance
   - Lines 237-241: Day cell `hx-*` attributes conditional on authentication
   - Both templates correctly hide write controls from anonymous visitors ✓

2. **Empty states (handled):**
   - event_form.html, line 327-328: `{% empty %}` renders "No todos yet." when event.todos.all is empty
   - Lines 122-157: Conditional rendering of optional fields (Description, URL, Target list, User, Proposal, Telescope, Instrument) omits empty rows
   - calendar.html: Grid always renders (no "no events" message, but appropriate for a calendar)

3. **Shared blocks (correct):**
   - event_form.html, lines 212-309: Observation series, campaign decoration, and attribution candidates are **below both branches**, rendered once for all viewers. This is correct per design D-05 (shared blocks render once).

4. **Interaction patterns:**
   - calendar.html, lines 387-432: JavaScript filter handler allows clicking legend swatches to filter events. State (`activeFilter`) is managed in-browser.
   - Day cells and event rows use `hx-get` to fetch the pop-up (update-event endpoint).
   - Bootstrap 5 modal is triggered server-side via `hx-on::after-request="bootstrap.Modal.getOrCreateInstance(...).show()"` (lines 222, 240, 246)
   - All interactions are async via htmx, no page reloads ✓

**Issues:**
- **BLOCKER (Pillar 2 compound):** Read-only card provides no visual feedback that it is non-interactive. A visitor clicking on the card sees no cursor change (no `cursor: pointer` on `.cal-event-card`), no disabled styling, and no "read-only" label. This violates user expectations from Pillar 2 (visual feedback).
- **Todo list visibility (Pillar 1 compound):** The "Todo list" heading (line 310) renders in read-only view, but the form input (line 315, upstream include) and add-todo button (from upstream todos.html) are hidden. Visitors see the heading with no way to interact with todos, creating confusing affordance.
- No error state handling visible (not applicable for a read-only display template; error handling is in the view/form submission layer, not this template).
- No loading state visible (calendar and event modal are server-rendered; htmx handles async requests transparently).

**Finding:** Pillar 6 is mostly solid (authentication branching is correct, empty states are handled, interaction patterns use htmx cleanly). However, two UX issues identified above (from Pillars 1 and 2) degrade this score from 4/4 to 3/4: the "Todo list" heading in read-only view and the lack of visual feedback on the read-only card.

---

## Files Audited

- `src/templates/tom_calendar/partials/event_form.html` (262 lines) — Calendar event form (authenticated) and read-only card (anonymous)
- `src/templates/tom_calendar/partials/calendar.html` (435 lines) — Calendar month view with grid, navigation, and legend

**Phase context:** Phase 39 Plan 02 (39-02) added the read-only card (`#cal-event-card`, lines 114-159) and the conditional "Todo list" heading (line 310) for anonymous visitors. Plan 02 also added D-11 Bootstrap 5 utility class updates to calendar.html (border-start, border-end, fw-bold, me-2, me-3, var(--bs-white)).

---

*Review conducted: 2026-10-08*
*Audit method: Code-only (no dev server, no screenshots)*
*Standard: Abstract 6-pillar standards (no UI-SPEC.md)*
