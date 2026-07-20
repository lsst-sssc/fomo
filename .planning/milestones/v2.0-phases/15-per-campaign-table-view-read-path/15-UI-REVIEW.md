# Phase 15 — UI Review

**Audited:** 2026-07-03
**Baseline:** 15-UI-SPEC.md (Design Contract)
**Screenshots:** not captured (no dev server at localhost:3000/5173/8080)
**Audit Type:** Code-only review (Django templates, Campaign tables/filters/views)

---

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 3/4 | Copy matches UI-SPEC in most places; minor empty-state link divergence |
| 2. Visuals | 3/4 | Layout structure correct; empty-state heading typography inconsistency |
| 3. Color | 4/4 | Approval/run-status badges precisely match UI-SPEC contracts; no hardcoded colors |
| 4. Typography | 2/4 | Page headings correct (h4 + font-weight-bold); empty-state heading missing font-weight-bold; label styling correct |
| 5. Spacing | 3/4 | Bootstrap 4 spacing scale mostly correct; custom inline style on free-text truncation (max-width: 200px) violates constraint |
| 6. Experience Design | 2/4 | Empty-table state text missing "Clear filters" link required by UI-SPEC; empty-state heading lacks proper styling |

**Overall: 17/24** — Several findings require fixes before shipping; contact PII gating and badge contracts are solid.

---

## Top 3 Priority Fixes

1. **BLOCKER: Table empty-state "Clear filters" link missing** — The table's `empty_text` must include an inline link to clear filters, per UI-SPEC Copywriting Contract. Currently renders as plain text "No runs match these filters. Clear filters to see all runs for this campaign." without the link. User must rely on the filter panel's "Clear filters" button (exists above table) as workaround, but the explicit UI-SPEC contract requires the link in the empty-state message itself. Fix: Use `format_html()` in Table.Meta.empty_text or override the empty-table template block in campaignrun_table.html.

2. **WARNING: Campaign list empty-state heading lacks font-weight-bold** — The h5 "No campaigns yet." is missing the `.font-weight-bold` class applied to all other page headings in the UI-SPEC. Creates visual inconsistency with the h4 page heading directly above it (which has font-weight-bold). User sees mismatched heading weights on the same page. Fix: Add `font-weight-bold` class to the h5 element in campaign_list.html line 19.

3. **WARNING: Free-text column truncation uses arbitrary CSS width** — The `_FREE_TEXT_ATTRS` dict in campaign_tables.py applies `'style': 'max-width: 200px;'` as an inline style. This is a custom CSS value not in Bootstrap 4's spacing scale (8px/16px/24px/32px/48px/64px), violating the UI-SPEC constraint "do not add new custom CSS for layout spacing." While this is technically width (not spacing), it's an arbitrary pixel value. Fix: Either use a Bootstrap utility class (e.g., `col-3` for a fixed column width) or define the width in a stylesheet using a scale-compliant value.

---

## Detailed Findings

### Pillar 1: Copywriting (3/4)

**Passing:**
- Primary CTA labels are specific: "View {{ campaign.name }} Runs" (not generic "View Campaign") — matches UI-SPEC Copywriting Contract (file: campaign_links.html, line 1)
- Navbar entry is single-word "Campaigns" — matches existing navbar convention (file: campaigns_nav_link.html, line 2)
- Filter button labels: "Filter" and "Clear filters" (not generic "Search" or "Apply") — matches UI-SPEC (file: campaignrun_table.html, lines 35-36)
- Campaign list "run count" badge copy: "{{ run_count }} run{{ run_count|pluralize }}" — correctly pluralizes (file: campaign_list.html, line 13)
- Campaign list empty state heading and body match UI-SPEC exactly: "No campaigns yet." + full descriptive text (file: campaign_list.html, lines 19-23)
- Table empty state copy matches UI-SPEC: "No runs match these filters. Clear filters to see all runs for this campaign." (file: campaign_tables.py, line 74)
- Filter label relabeling: "open_to_collaboration" field relabeled to "Open to collaboration?" with options "Any" (not "Unknown"), "Yes", "No" — matches UI-SPEC (file: campaignrun_table.html, lines 22-32)
- No generic labels like "Submit", "OK", "Cancel" found

**Failing:**
- Table empty-state copy text lacks the inline link to "Clear filters" — UI-SPEC specifies: `"No runs match these filters. Clear filters to see all runs for this campaign."` where "Clear filters" is an **inline link** (not standalone text). Current implementation (campaign_tables.py, line 74) is plain text only. **User impact:** When all rows are filtered out, the empty message suggests clearing filters but provides no clickable link within the message itself; users must find the filter panel's "Clear filters" button above the table. **Concrete fix:** Change empty_text to use `format_html()` with a hyperlink, or implement custom empty-table template block.

---

### Pillar 2: Visuals (3/4)

**Passing:**
- Page heading hierarchy: `<h4 class="font-weight-bold mb-4">` applied to both campaign list page (line 5 campaign_list.html) and per-campaign table page (line 6 campaignrun_table.html) — creates clear focal point
- Filter panel visual container: `<div class="card bg-light p-3 mb-4">` — visually separates filter controls from table content; matches UI-SPEC Layout Contract (file: campaignrun_table.html, line 8)
- List-group layout for campaign items: `<div class="list-group">` with `.list-group-item.list-group-item-action` — provides clear clickable affordance (file: campaign_list.html, lines 8-15)
- Table container: Uses `django_tables2/bootstrap4-responsive.html` template which auto-wraps table in `.table-responsive` for horizontal scrolling on narrow viewports (file: campaign_tables.py, line 72)
- Visual hierarchy in button row: Campaign links styled as `btn btn-info` matching existing "Make Ephemeris" button (file: campaign_links.html, line 1)
- Icons in open_to_collaboration column: Font Awesome 4.7 icons (fa-check / fa-times) with semantic colors (text-success / text-muted) — matches existing FOMO precedent (file: campaign_tables.py, lines 126-127)

**Failing:**
- Campaign list empty-state heading is `<h5>` without `.font-weight-bold` class — visually lighter/thinner than the `<h4>` page heading above it, creating inconsistent visual weight hierarchy. UI-SPEC Typography Contract specifies all headings on a page should use explicit `font-weight-bold` for consistency. **User impact:** Empty state looks typographically misaligned with the rest of the page. **Concrete fix:** Add `class="font-weight-bold"` to the h5 on campaign_list.html line 19.

---

### Pillar 3: Color (4/4)

**Full compliance:**
- Approval-status badge colors match UI-SPEC Approval-Status Badge Contract exactly (file: campaign_tables.py, lines 18-22):
  - `pending_review` → `badge-warning` (#ffc107 yellow)
  - `approved` → `badge-success` (#28a745 green)
  - `rejected` → `badge-danger` (#dc3545 red)
- Run-status badge colors match UI-SPEC Run-Status Badge Contract exactly (file: campaign_tables.py, lines 28-37):
  - `requested`, `planned` → `badge-secondary` (#6c757d grey)
  - `observed`, `reduced` → `badge-info` (#17a2b8 cyan)
  - `published` → `badge-primary` (#007bff blue — accent)
  - `cancelled`, `not_awarded`, `weather_tech_failure` → `badge-light` (#f8f9fa) with `border: 1px solid #6c757d` — NOT danger-red
- Campaign list run-count badge uses `badge-secondary` (grey/muted) per UI-SPEC secondary color (file: campaign_list.html, line 13)
- Filter panel background: `bg-light` (#f8f9fa) — matches UI-SPEC secondary color (file: campaignrun_table.html, line 8)
- Campaign list empty-state text: `text-muted` — appropriate for non-critical message (file: campaign_list.html, line 18)
- No hardcoded hex colors or rgb() values in templates/code except one inline border-color:
  - `border: 1px solid #6c757d;` in campaign_tables.py line 96 — this is the Bootstrap 4 secondary color, acceptable for the muted badge border
- No color scheme violations; 60/30/10 distribution observed (white background dominant, light-grey panels secondary, blue accents on links/buttons/primary-badge)

---

### Pillar 4: Typography (2/4)

**Passing:**
- Page headings: `<h4 class="font-weight-bold mb-4">` — both pages apply explicit `.font-weight-bold` to ensure 700 weight (not relying on browser `<h4>` default) (files: campaign_list.html:5, campaignrun_table.html:6)
- Filter form labels: `<label class="font-weight-bold">` — explicit bold applied (file: campaignrun_table.html, lines 11, 22)
- Badge text inherits Bootstrap 4 badge styling (14px/700) via `.badge` class, which renders labels bold by default (files: campaign_list.html:13, campaign_tables.py:97/107)
- Table cell text: Default Bootstrap 4 body text (16px/400) for data rows, no custom font sizes on table cells
- Table headers (`<th>`): Rendered by django-tables2 bootstrap4-responsive.html with class `thead-light`, which applies bold styling

**Failing:**
- Campaign list empty-state heading: `<h5>No campaigns yet.</h5>` — missing `.font-weight-bold` class. Renders with browser default `<h5>` weight (~600, not 700), creating a visual weight mismatch with the page's `<h4>` heading (which is explicitly bold). UI-SPEC Typography Contract specifies explicit `.font-weight-bold` on all headings for consistency. **User impact:** Empty state heading looks lighter/thinner than expected, inconsistent with the main page heading directly above. **Concrete fix:** Change line 19 of campaign_list.html from `<h5>No campaigns yet.</h5>` to `<h5 class="font-weight-bold">No campaigns yet.</h5>`.

---

### Pillar 5: Spacing (3/4)

**Passing:**
- Page heading spacing: `mb-4` (1.5rem = 24px) creates 'lg' gap between heading and content below — matches UI-SPEC Spacing Scale (files: campaign_list.html:5, campaignrun_table.html:6)
- Filter panel spacing: `p-3` (1rem = 16px) for internal padding + `mb-4` (24px) below for separation — matches UI-SPEC 'md' and 'lg' tokens (file: campaignrun_table.html:8)
- Campaign list empty-state spacing: `p-5` (3rem = 48px) provides generous centering padding — matches UI-SPEC '2xl' token (file: campaign_list.html:18)
- Campaign list items: Default Bootstrap list-group spacing; no custom spacing applied (file: campaign_list.html:8-15)
- Button wrapping gaps in target-detail campaign links: `mr-2 mb-2` (8px horizontal/vertical) for multi-button wrapping on narrow viewports — matches UI-SPEC and existing precedent (file: campaign_links.html:1)
- Table spacing: `table-sm` class reduces cell padding to ~5px — deliberate exception per UI-SPEC (spreadsheet-parity density requirement) (file: campaign_tables.py:73)
- Form-group default spacing: Standard Bootstrap `<div class="form-group">` spacing between filter fields (file: campaignrun_table.html:10, 21)
- All spacing uses Bootstrap 4 utility classes (p-*, mb-*, mr-*, etc.) — no arbitrary custom spacing defined

**Failing:**
- Free-text column truncation width: `'style': 'max-width: 200px;'` is an inline style with an arbitrary pixel value not in Bootstrap 4's spacing/sizing scale (which uses 8px multiples: 4/8/16/24/32/48/64). UI-SPEC explicitly states "do not add new custom CSS for layout spacing." While this is technically a width constraint (not spacing), it's a custom CSS value. File: campaign_tables.py, line 43 in `_FREE_TEXT_ATTRS`. **User impact:** Hardcoded 200px may not respond appropriately to different viewport sizes or design system changes. **Concrete fix:** Use Bootstrap's responsive column grid (e.g., `col-md-3`) or define width in a stylesheet with scale-compliant values (e.g., 16rem = 256px for alignment with 8px scale).

---

### Pillar 6: Experience Design (2/4)

**Passing:**
- Empty states implemented for both pages:
  - Campaign list with zero runs: Clear heading ("No campaigns yet.") + explanatory body text (file: campaign_list.html:18-24)
  - Table with filtered results matching zero rows: Empty text auto-rendered by django-tables2 (file: campaign_tables.py:74)
- Filter state handling:
  - Default unfiltered load shows all rows (D-07) — no filtering applied on initial page load
  - run_status multi-select filter (OR semantics) — Django Filter's `MultipleChoiceFilter` with `CheckboxSelectMultiple` widget (file: campaign_filters.py:20-24)
  - open_to_collaboration boolean filter — auto-generated `BooleanFilter` with custom 3-option select ("Any", "Yes", "No") (file: campaignrun_table.html:26-32)
- Pagination: 25 rows per page (D-11) set in campaign_views.py (line 60) — django-tables2 renders pagination links automatically via bootstrap4-responsive.html
- Sort functionality: Default sort by obs_date descending (D-10) (file: campaign_tables.py:71) — users can click column headers to re-sort
- PII gating for anonymous users: Contact fields excluded from SQL SELECT via `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` (file: campaign_views.py:26-44) — prevents data leakage at the queryset layer, not just template masking
- Error handling: Invalid campaign pk returns 404 via `get_object_or_404(TargetList, pk=...)` (file: campaign_views.py:79)
- No loading states or spinners needed (server-rendered, synchronous request/response)
- Confirmation: No destructive actions in this read-only phase

**Failing:**
- Table empty-state message lacks the "Clear filters" link specified in UI-SPEC Copywriting Contract. Current implementation (campaign_tables.py:74) provides plain text: `"No runs match these filters. Clear filters to see all runs for this campaign."` The UI-SPEC explicitly requires: `"No runs match these filters. Clear filters to see all runs for this campaign."` where **"Clear filters" is an inline link to the bare table URL.** Current table provides a "Clear filters" button in the filter panel above the table (campaignrun_table.html:36), but the UI-SPEC requires the link to be **inside the empty-state message itself** for discoverability. **User impact:** When viewing an empty filtered table, the suggested action ("Clear filters") has no clickable affordance within the message; users must scan upward for the button in the filter panel. **Concrete fix:** Use `format_html()` in Table.Meta.empty_text or implement a custom empty-table template block that renders the message with a hyperlink.

- Campaign list empty-state heading visual treatment incomplete (see Pillar 4 Typography finding). The h5 lacks `.font-weight-bold`, creating visual deemphasis that reduces the prominence of the empty-state message.

---

## Files Audited

- `src/templates/campaigns/campaign_list.html` — campaigns list page (D-03)
- `src/templates/campaigns/campaignrun_table.html` — per-campaign table with filter panel (VIEW-01/04)
- `solsys_code/campaign_tables.py` — table definition, badge rendering, column styling
- `solsys_code/campaign_filters.py` — filter set definition
- `solsys_code/campaign_views.py` — view classes, queryset/context logic, PII gating
- `src/templates/solsys_code/partials/campaign_links.html` — target-detail campaign links (VIEW-02)
- `src/templates/solsys_code/partials/campaigns_nav_link.html` — navbar entry (VIEW-02)

**Code review coverage:**
- Badge CSS class contracts: verified against UI-SPEC Approval-Status and Run-Status Badge Contracts
- Spacing scale: verified all utilities are Bootstrap 4 classes (8px multiples), except one custom `max-width: 200px` inline style
- Typography: verified explicit `.font-weight-bold` on page headings and labels, one empty-state heading missing the class
- Copywriting: verified CTA labels, empty-state copy, filter labels match UI-SPEC contract, except empty-table link missing
- PII gating: verified contact fields excluded from non-staff queryset (`ALLOWED_FIELDS_FOR_NON_STAFF` list + belt-and-suspenders `exclude=` in `get_table_kwargs`)
- Color: verified no hardcoded hex values, all badge colors use Bootstrap 4 utility classes, 60/30/10 distribution observed

---

## Notes

- **Screenshot capture:** No dev server detected; audit is code-based only, not visual verification via browser rendering
- **Registry audit:** Not applicable; this is a server-rendered Django/Bootstrap 4 app (no shadcn or third-party component registries)
- **Test coverage:** Phase included comprehensive integration tests (TEST_CAMPAIGN_VIEWS) per SUMMARY.md — all tests passing. UI audit focuses on design contract compliance, not test coverage.
