---
phase: 09-proposal-color-status-visual-treatment
verified: 2026-06-25T00:00:00Z
status: human_needed
score: 4/5
behavior_unverified: 1
overrides_applied: 0
human_verification:
  - test: "Load the calendar page with at least two distinct proposals visible. Click a legend swatch for one proposal."
    expected: "That proposal's events remain at full opacity; all other events dim (opacity ~0.18, grayscale). The clicked swatch gets font-weight bold + underline (is-active). Clicking the same swatch again restores all events to full opacity and removes is-active."
    why_human: "The JS IIFE click-handler toggle (add/remove .cal-filtering, .cal-filter-match, .is-active) is a DOM state-machine. Grep and server-side test rendering cannot exercise click events or verify CSS class mutations."
  - test: "Navigate to Prev month then back to original month via the Prev/Next buttons. Click a legend swatch."
    expected: "The click-to-filter JS fires correctly after the htmx swap — activeProposal resets to null on each swap, and the filtering behavior works identically on the re-rendered fragment."
    why_human: "htmx replaces the outerHTML of #calendar-partial; only a browser test can confirm the inline <script> re-executes inside the swapped fragment (Pitfall 5 / D-03)."
  - test: "Verify the 8-colour PROPOSAL_PALETTE renders distinguishably for deuteranopia and protanopia users. Use a CVD simulator (e.g. Coblis) on the calendar with >= 5 proposals visible."
    expected: "All 8 palette colours remain mutually distinguishable under both deuteranopia and protanopia simulation (09-VALIDATION.md Manual A1 — colorblind-vetted claim)."
    why_human: "Colour-vision-deficiency simulation requires visual inspection; no automated assertion can substitute for human perceptual judgment."
behavior_unverified_items:
  - truth: "Clicking a legend swatch dims non-matching events and highlights matching ones client-side; clicking again clears it; behavior survives htmx month swaps (DISPLAY-07, D-03)"
    test: "Load calendar with proposals, click a .cal-legend-swatch element"
    expected: "container.classList contains 'cal-filtering'; matching .cal-event elements have 'cal-filter-match'; swatch has 'is-active'; click again removes all three class mutations"
    why_human: "JS DOM state-machine (toggle via click event delegation) cannot be exercised by Django test-client response rendering or grep. Presence and wiring of the IIFE, CSS classes, and data-proposal attributes are verified; the transition is not."
---

# Phase 9: Proposal Color & Status Visual Treatment — Verification Report

**Phase Goal:** A calendar viewer can identify which proposal an event belongs to by color alone (consistent across telescopes and re-renders) and can distinguish queued/placed/terminal-failure status visually, not just by reading title-prefix text.
**Verified:** 2026-06-25
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Same normalized proposal string renders the same color across restarts; empty/blank/None proposal gets dedicated neutral slot `#5a6268`, not a hash-of-empty-string color (SC1 / DISPLAY-04) | VERIFIED | `proposal_color` normalizes with `.strip().upper()` then sha256-hashes (never built-in hash). Tests `test_normalization_case_insensitive`, `test_normalization_trailing_space`, `test_empty_string_returns_neutral_slot`, `test_none_returns_neutral_slot` all green. `grep -c 'hash(' calendar_display_extras.py` = 0. |
| 2 | A `[QUEUED]` all-day event renders its proposal-keyed background-color; the old flat-grey `background-color: rgba(0,0,0,0.45)` override is gone (SC2 / DISPLAY-05) | VERIFIED | `grep -c 'background-color: rgba(0, 0, 0, 0.45)' calendar.html` = 0. Template uses `{% proposal_color event.proposal as bg_color %}` and `background-color: {{ bg_color }}` in both branches. Integration test `test_display05_old_queued_grey_background_color_is_gone` and `test_display05_queued_event_renders_proposal_background_color` both green. |
| 3 | Queued, placed, and terminal-failure events are visually distinguishable via box-shadow rings layered on top of proposal color (not replacing it); Phase 8 dashed border is composed alongside, not replaced; applies to both all-day and timed branches (SC3 / DISPLAY-06) | VERIFIED | `status_border_css` returns exact locked strings: queued `box-shadow: 0 0 0 2px rgba(0,0,0,0.45);`, terminal `box-shadow: 0 0 0 3px rgba(160,0,0,0.55);`, placed `''`. Template composes: `{{ status_border }} border: 2px dashed rgba(0,0,0,0.65);` for fallback-labeled events. Integration tests `test_display06_queued_box_shadow_present`, `test_display06_terminal_box_shadow_present`, `test_display06_pitfall3_composition_dashed_and_queued_coexist` all green. `grep -c 'dashed' calendar_display_extras.py` = 0 (D-09). |
| 4 | An on-page footer legend maps rendered colors to proposal codes; collision-grouped proposals share one swatch; empty-proposal events appear as `Classical schedule` entry ordered last (SC4 / DISPLAY-07 — legend) | VERIFIED | `{% visible_proposals weeks as proposal_legend %}` in footer. `.cal-legend-swatch` with `data-proposal="{{ entry.color }}"` (color-keyed, not raw string — enables strict === collision grouping). `grep -c 'cal-legend-swatch' calendar.html` = 6. Integration tests `test_display07_legend_swatch_markup_present` and `test_display07_classical_schedule_label_present_when_empty_proposal_events_visible` green. |
| 5 | Clicking a legend swatch dims non-matching events and highlights matching ones client-side; clicking again clears it; behavior survives htmx month swaps (SC5 / DISPLAY-07 — click-to-filter) | PRESENT_BEHAVIOR_UNVERIFIED | JS IIFE present inside `#calendar-partial` (line 253, before closing `</div>` at line 280). All five CSS rules present (`.cal-legend-swatch`, `.is-active`, `.cal-event` transition, `#calendar-partial.cal-filtering .cal-event`, `#calendar-partial.cal-filtering .cal-event.cal-filter-match`). `data-proposal="{{ bg_color }}"` on every `.cal-event` element; `data-proposal="{{ entry.color }}"` on each swatch. Code and wiring are present; DOM state-transition on click cannot be verified without browser — see Human Verification. |

**Score:** 4/5 truths verified (1 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/templatetags/__init__.py` | Empty package marker (0 bytes) | VERIFIED | Exists, `wc -c` = 0. Django discovers the `templatetags` package. |
| `solsys_code/templatetags/calendar_display_extras.py` | Three `simple_tag`s + constants | VERIFIED | All 6 public names exported: `PROPOSAL_PALETTE`, `NEUTRAL_SLOT_COLOR`, `CLASSICAL_SCHEDULE_LABEL`, `proposal_color`, `status_border_css`, `visible_proposals`. `register = template.Library()` present. |
| `solsys_code/tests/test_calendar_display_extras.py` | 3 TestCase classes, 23 tests | VERIFIED | `ProposalColorTest` (8), `StatusBorderCssTest` (10), `VisibleProposalsTest` (5). All 23 tests PASS. |
| `src/templates/tom_calendar/partials/calendar.html` | Rewritten with Phase 9 changes | VERIFIED | `{% load … calendar_display_extras %}` on line 123. Both branches rewritten. Footer legend present. CSS rules present. Inline `<script>` IIFE present. |
| `solsys_code/tests/test_calendar_template.py` | Extended with DISPLAY-04/05/06/07 tests | VERIFIED | 4 marker constants, 5 Phase 9 fixtures, 10 new test methods. `num_fallback_day_cell_occurrences` updated to 4. All 13 tests PASS. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `calendar.html` | `calendar_display_extras` | `{% load tz calendar_tags calendar_display_extras %}` | WIRED | Line 123 of calendar.html. Django template library auto-discovered via `register = template.Library()` in the module. |
| `calendar.html` events | `proposal_color` tag | `{% proposal_color event.proposal as bg_color %}` | WIRED | Lines 176, 197 (all-day and timed loops). |
| `calendar.html` events | `status_border_css` tag | `{% status_border_css event.title as status_border %}` | WIRED | Lines 177, 198 (all-day and timed loops). |
| `calendar.html` legend | `visible_proposals` tag | `{% visible_proposals weeks as proposal_legend %}` | WIRED | Line 236. |
| `.cal-event` elements | legend `.cal-legend-swatch` elements | `data-proposal="{{ bg_color }}"` on events; `data-proposal="{{ entry.color }}"` on swatches — both color-keyed | WIRED | Lines 179, 201, 207 (events); line 238 (swatches). Same hex key enables JS `===` collision grouping (D-04). |
| `calendar.html` inline `<script>` | `#calendar-partial` wrapper | IIFE placed inside the swapped fragment (before closing `</div>` at line 280) | WIRED | Lines 253-279. Re-executes on each htmx `outerHTML` swap (Pitfall 5 / D-03). |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `calendar.html` all-day branch | `bg_color` | `{% proposal_color event.proposal %}` → sha256 of `CalendarEvent.proposal` DB field | Yes — real DB field → deterministic palette hex | FLOWING |
| `calendar.html` all-day branch | `status_border` | `{% status_border_css event.title %}` → startswith check on `CalendarEvent.title` DB field | Yes — real DB field → fixed CSS literal | FLOWING |
| `calendar.html` footer legend | `proposal_legend` | `{% visible_proposals weeks %}` → iterates the `weeks` context already materialized by `render_calendar()` (no extra DB query, D-02) | Yes — real event objects already in context | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 23 unit tests for `calendar_display_extras` | `python manage.py test solsys_code.tests.test_calendar_display_extras -v 2` | 23/23 OK, 0.030s | PASS |
| 13 integration tests for calendar template | `python manage.py test solsys_code.tests.test_calendar_template -v 2` | 13/13 OK, 12.18s | PASS |
| Full `solsys_code` test suite (no regressions) | `python manage.py test solsys_code` | 171/171 OK, 121.85s | PASS |
| `dashed` not in `calendar_display_extras.py` (D-09) | `grep -c 'dashed' solsys_code/templatetags/calendar_display_extras.py` | 0 | PASS |
| Built-in `hash(` not used (determinism) | `grep -c 'hash(' solsys_code/templatetags/calendar_display_extras.py` | 0 | PASS |
| Old `[QUEUED]` grey override removed (DISPLAY-05) | `grep -c 'background-color: rgba(0, 0, 0, 0.45)' src/templates/tom_calendar/partials/calendar.html` | 0 | PASS |
| `calendar_display_extras` loaded in template | `grep -n 'calendar_display_extras' calendar.html` | line 123 | PASS |
| ruff lint clean on new files | `ruff check solsys_code/templatetags/ solsys_code/tests/test_calendar_display_extras.py` | All checks passed | PASS |
| ruff format clean on new files | `ruff format --check solsys_code/templatetags/ solsys_code/tests/test_calendar_display_extras.py` | 3 files already formatted | PASS |
| Click-to-filter JS toggle (DOM state machine) | (requires browser) | Cannot test without browser | SKIP — routed to human verification |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DISPLAY-04 | Plan 01, Plan 02 | CalendarEvent color hashed from normalized proposal into colorblind-vetted palette; same proposal same color; empty proposal neutral slot | SATISFIED | `proposal_color` tag with sha256 + `PROPOSAL_PALETTE[index % 8]`; template applies to both all-day and timed branches; 8 unit tests + 2 integration tests green |
| DISPLAY-05 | Plan 02 | `[QUEUED]` template override that discarded proposal color with flat grey removed | SATISFIED | `grep -c 'background-color: rgba(0,0,0,0.45)' calendar.html` = 0; integration test `test_display05_old_queued_grey_background_color_is_gone` green |
| DISPLAY-06 | Plan 01, Plan 02 | Status visual treatment (box-shadow) layered on top of (not replacing) proposal color; queued/placed/terminal distinguishable; D-09 no-dashed enforced; Phase 8 dashed border composed | SATISFIED | `status_border_css` emits exact locked box-shadow strings; template concatenates `{{ status_border }}` alongside existing border literal; Pitfall 3 composition test green |
| DISPLAY-07 | Plan 01, Plan 02 | On-page legend with proposal-to-color mapping; collision grouping; Classical schedule entry; click-to-filter client-side highlighting | PARTIALLY SATISFIED | Legend markup, visible_proposals aggregation, CSS dim/highlight rules, JS IIFE — all present and wired. Legend static rendering VERIFIED. Click-to-filter behavior requires human testing (see Human Verification). |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | No TBD/FIXME/XXX/TODO markers in modified files. No stubs, no hardcoded empty returns, no placeholder text. |

### Human Verification Required

#### 1. Click-to-Filter: highlight matching events on legend swatch click

**Test:** Load the calendar page with at least two distinct proposals visible in the current month. Click a legend swatch (`.cal-legend-swatch`) for one proposal.
**Expected:** That proposal's events remain at full opacity; all other `.cal-event` elements dim to ~0.18 opacity with grayscale. The clicked swatch gains bold text and underline (`.is-active`). Clicking the same swatch again removes `.cal-filtering` from `#calendar-partial`, removes `.cal-filter-match` from all events, and removes `.is-active` from all swatches.
**Why human:** The JS IIFE is a click-event delegation state machine (`activeProposal` variable, CSS class toggles). Django test-client rendering exercises HTML generation but cannot simulate click events or assert resulting DOM class mutations.

#### 2. Click-to-Filter: htmx month-swap survival

**Test:** With the calendar showing, navigate to Prev or Next month via the buttons. After the htmx `outerHTML` swap completes, click a legend swatch.
**Expected:** The inline `<script>` re-executes (it is inside the swapped `#calendar-partial` fragment, before its closing `</div>`). `activeProposal` resets to `null` on each swap. Click-to-filter works identically on the re-rendered month.
**Why human:** htmx fragment swap re-execution of inline scripts requires a browser; server-side tests render static HTML and cannot verify script re-initialization after swap.

#### 3. Colorblind accessibility of PROPOSAL_PALETTE

**Test:** With >= 5 proposals visible on the calendar, run the page through a CVD simulator (e.g. Coblis or Pilestone simulator) for deuteranopia and protanopia.
**Expected:** All 8 palette entries (`#005f9e`, `#a34000`, `#5b2080`, `#006b4e`, `#9e1c1c`, `#006b6b`, `#6b2060`, `#7a4500`) remain mutually distinguishable under both deficiency simulations — the colorblind-vetted claim in 09-VALIDATION.md Manual A1.
**Why human:** Perceptual colour distinction requires human visual assessment; no automated assertion substitutes for CVD simulation judgment.

### Gaps Summary

No blocking gaps found. All five must-haves from ROADMAP Success Criteria have their supporting code present, substantive, and wired. The one PRESENT_BEHAVIOR_UNVERIFIED truth (SC5 — click-to-filter DOM state machine) requires browser-level verification and is routed to human verification above.

---

_Verified: 2026-06-25_
_Verifier: Claude (gsd-verifier)_
