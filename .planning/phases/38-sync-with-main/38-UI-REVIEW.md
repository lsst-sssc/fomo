# Phase 38 — UI Review

**Audited:** 2026-10-07
**Baseline:** Abstract 6-pillar standards (no UI-SPEC.md for Phase 38)
**Screenshots:** Not captured (no dev server; code-only audit)
**Interaction captures:** Off (workflow.ui_interaction_capture is false)

---

## Scope Note

Phase 38 is an infrastructure sync with `origin/main`. No FOMO UI was authored in this phase. The only UI files in scope are three templates that arrived from `origin/main` through the merge and are byte-identical to main's copies:
- `src/templates/solsys_code/partials/navbar_list.html` (Rubin ToO navbar menu)
- `src/templates/solsys_code/scout_rubin_too_list.html` (Scout candidates list)
- `src/templates/solsys_code/scout_rubin_too_stats.html` (Scout filter statistics)

**Findings below are observations for `origin/main` and Phase 39+, not Phase 38 defects.** Phase 38 adopted these unchanged and makes no UI changes.

---

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 4/4 | All labels specific and contextual; empty states actionable; no generic patterns |
| 2. Visuals | 3/4 | Clear hierarchy via h1 titles and intro text; table layouts dense with no visual column grouping |
| 3. Color | 3/4 | Bootstrap tokens only, no hardcoded colors; conservative accent use (text-muted, table-primary) |
| 4. Typography | 3/4 | Consistent Bootstrap defaults; hierarchy via semantic HTML (h1 > p > td); no custom sizing |
| 5. Spacing | 3/4 | Bootstrap grid and spacing defaults throughout; tables lack whitespace between rows |
| 6. Experience Design | 3/4 | Empty states well-handled; navigation clear; no loading states visible in templates |

**Overall: 20/24**

---

## Top 3 Priority Fixes (for main/Phase 39+)

1. **Dense table layouts lack visual column grouping** — Scout list (14 columns) and stats tables are hard to scan; coordinates and dates should be visually grouped. Users eye-jump between scattered related columns. — Recommendation: Add `<colgroup>` or CSS column borders to group related data (e.g., RA/Dec together, observation dates together); consider a "metadata" column separator.

2. **No loading states shown in templates** — Scout list and stats pages appear static; if data is fetched async or slow, users get no feedback. Unclear when a table is loading vs. ready. — Recommendation: Add a `<div>` with spinner/skeleton behind `{% if scout_details or years %}` condition; or, if view is always synchronous, document that in template comments.

3. **Table header tooltips not keyboard-accessible** — Column titles use `title="..."` attributes (mouse-only). Keyboard-only users cannot discover filter explanations (e.g., "NEO digest score (>= 98)"). — Recommendation: Replace `title` with ARIA labels (`aria-label`) or add a collapsible legend; ensure tooltips are focusable for keyboard navigation.

---

## Detailed Findings

### Pillar 1: Copywriting (4/4)
**EXCELLENT**

All three templates use specific, contextual labels:

- **navbar_list.html:** Menu items "Current candidates", "First-pass stats" are specific to function, not generic.
- **scout_rubin_too_list.html:** Page title "Scout candidates passing the Rubin ToO filters" is descriptive. Intro text explains snapshot behavior. Empty state "No Scout candidates currently pass the filters." is specific, not "No data" or "Empty". Table headers: abbreviated technical labels (NEO, Geo, RMS, nObs, Arc, Vmag, Unc+1d, Rate) each paired with a title attribute explaining the metric.
- **scout_rubin_too_stats.html:** Page title "Rubin ToO filter — first-pass events per year" is specific. Empty state "No Scout history data yet — run `rundataquery <query_id>` then `updatescout` to populate." is actionable, telling users exactly what to do next.

**No generic patterns found:** No "Click Here", "Submit", "OK", "Cancel", "Error", or "Success" labels. All copy is domain-specific and user-actionable.

### Pillar 2: Visuals (3/4)
**GOOD, with one notable gap**

All three templates extend `tom_common/base.html` and use Bootstrap 5 classes for consistent styling:

- **Visual hierarchy:** Each page starts with h1 (focal point), followed by descriptive paragraph (supporting context), then the main content (table). Clear and effective.
- **navbar_list.html:** Dropdown structure is standard Bootstrap; active state managed via conditional CSS class (`active` when URL name matches).
- **scout_rubin_too_list.html:** Table with hover effect (`.table-hover`) provides feedback on row selection. 14 columns spread across full width, but no visual grouping of related data.
- **scout_rubin_too_stats.html:** Color-coded rows (`.table-primary` for "Total Targets" and "Combined" rows) create visual hierarchy; year columns are indexed clearly.

**Gap:** Dense table layouts without visual column grouping. Coordinates (RA, Dec) are separated by columns (Vmag, Unc+1d, Rate). Users must track which column is which across 14 columns in scout_rubin_too_list.html. No separators or background color zones.

### Pillar 3: Color (3/4)
**GOOD**

All color usage is via Bootstrap design tokens:

- **navbar_list.html:** Native Bootstrap nav styling; no explicit color classes.
- **scout_rubin_too_list.html:** `.text-muted` on intro paragraph (secondary text); `.table-hover` (default gray on hover). No hardcoded hex/rgb values.
- **scout_rubin_too_stats.html:** `.text-muted` on intro; `.table-primary` on key rows (light blue background, per Bootstrap theme); `.thead-light` on table header (subtle background).

**Accent usage:** Conservative. `.table-primary` is used only on "Total Targets" and "Combined" rows, the two rows that represent aggregate data. Appropriate use of accent to differentiate key rows from detail rows. No overuse (not more than 10 unique color-class combinations).

**Minor observation:** `.table-primary` contrast ratio with default text may need verification for WCAG AA compliance, but this is a Bootstrap-level concern, not template-specific.

### Pillar 4: Typography (3/4)
**GOOD**

All three templates use Bootstrap's default typography:

- **Font sizes:** Only Bootstrap defaults (no custom `text-*-size` classes). h1 for titles; p, td, th at default size.
- **Font weights:** Only Bootstrap defaults (no custom `font-*-weight` classes). `<strong>` for emphasis (e.g., counts in scout_rubin_too_stats.html).
- **Hierarchy:** Clear via semantic HTML structure: h1 (page title) > p (intro text) > table (data). No custom styling needed.

**Observations:**
- scout_rubin_too_list.html: Column abbreviations (NEO, Geo, RMS, nObs, Vmag, Unc+1d, Rate) are terse; readability depends on title tooltips.
- scout_rubin_too_stats.html: "Section 2.1" reference in intro text assumes reader familiarity with source doc; no definition provided.

Consistent and functional. No issues with readability at desktop resolution.

### Pillar 5: Spacing (3/4)
**GOOD**

All spacing uses Bootstrap's built-in grid and component defaults:

- **navbar_list.html:** Bootstrap `.nav-item`, `.nav-link`, `.dropdown-menu` handle all spacing internally; no custom margin/padding classes.
- **scout_rubin_too_list.html:** Bootstrap `.row` and `.col-md-12` for layout; `.table`, `.table-hover`, `.table-sm` for table spacing. No custom `p-*`, `m-*`, or `gap-*` classes outside Bootstrap defaults.
- **scout_rubin_too_stats.html:** `.row` and `.col-md-auto` for layout; same table defaults.

**Consistent and standard.** No arbitrary `<div class="p-10">` or hardcoded margin values.

**Minor observation:** `.table-sm` is applied to both tables, making them compact. Vertical spacing between rows is minimal. For large tables with many columns (scout_rubin_too_list.html: 14 columns), tighter row spacing reduces white space and may make scanning harder. Acceptable for dense data tables, but worth noting for readability.

### Pillar 6: Experience Design (3/4)
**GOOD, with gaps in loading/error states**

**State coverage:**

| State | Coverage | Notes |
|-------|----------|-------|
| Empty | YES | Both main templates handle empty state explicitly with `{% empty %}` or `{% if %}` blocks. Copy is helpful and actionable. |
| Loading | NO | No loading spinner, skeleton, or progress indicator visible in templates. If data is fetched async, users see no feedback during load. |
| Error | NO | No error state template. Error handling is view responsibility (not visible here). |
| Disabled | N/A | No form inputs or buttons; not applicable. |
| Active/Selected | YES | navbar_list.html: `.active` class applied to current page. scout_rubin_too_list.html: `.table-hover` provides row-level feedback. |

**Navigation and context:**

- **navbar_list.html:** Dropdown groups related views (Current candidates, First-pass stats). Clear.
- **scout_rubin_too_list.html:** Table headers have `title` tooltips explaining each column (e.g., "NEO digest score (>= 98)"). Provides context without cluttering the header.
- **scout_rubin_too_stats.html:** Link back to candidate list ("View currently passing candidates →") provides navigation. Intro text explains what "first-pass events" means.

**Gaps:**

1. **Loading states:** If Scout list or stats views fetch data async (e.g., from an API), users see no feedback while waiting. Table appears empty until data arrives, creating ambiguity: "Is the table loading or is there no data?"
2. **Error states:** No error template. If a view fails, users may see a blank page or a generic Django error. No recovery guidance.
3. **Accessibility of tooltips:** `title` attributes are mouse-only. Keyboard-only users cannot access column explanations.

---

## Files Audited

1. `src/templates/solsys_code/partials/navbar_list.html` (8 lines) — Navbar dropdown menu for Rubin ToO feature
2. `src/templates/solsys_code/scout_rubin_too_list.html` (73 lines) — Table view of Scout candidates passing Rubin ToO filters
3. `src/templates/solsys_code/scout_rubin_too_stats.html` (60 lines) — Statistics table for Scout filter first-pass events by year

---

## Context

These three templates were part of `origin/main` before Phase 38 began. Phase 38 merged `origin/main` into `issue37-telescope-runs-calendar` (merge commit e12158c, 2026-10-07), and these templates arrived unchanged. No edits were made to them during Phase 38 execution. They are therefore observations for `origin/main` and downstream phases (Phase 39+), not Phase 38-specific findings.

## UI REVIEW COMPLETE
