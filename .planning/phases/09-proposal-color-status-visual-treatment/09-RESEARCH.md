# Phase 9: Proposal Color & Status Visual Treatment - Research

**Researched:** 2026-06-25
**Domain:** Django template-tag-mediated visual encoding on a third-party (`tom_calendar`) model — color hashing, CSS status treatment, client-side filter toggle
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 (legend placement):** The legend lives in the **existing footer row** (`calendar.html` ~line 198,
  the `d-flex justify-content-between align-items-center mt-1` row that already renders the target-list
  color key via `target_list_block.html` and the UTC-offset selector) — not a new row, not a collapsible
  panel. Mirror `target_list_block.html`'s exact swatch pattern: a colored `▌` bullet
  (`<span class="cal-event-bullet" style="color: {{ swatch_color }};">▌</span>`) followed by the
  proposal code as plain text.
- **D-02 (legend scope):** The legend lists only proposals with at least one event **visible in the
  currently-rendered month** — computed from the day-cell events already in the render context, not a
  separate all-history query.
- **D-03 (click-to-filter, scope expansion confirmed by user):** Each legend entry is **clickable** and
  toggles a **client-side highlight/dim filter**: clicking a proposal's legend entry full-opacity-
  highlights that proposal's events on the grid and dims all others; clicking again clears the
  highlight. No page reload, no htmx round-trip, no URL/query-param change — pure CSS/JS state toggle.
  REQUIREMENTS.md and ROADMAP.md were already updated (2026-06-25) to include this in DISPLAY-07; no
  further requirements amendment needed.
- **D-04 (collision handling):** If two or more proposals hash to the same palette color, the legend
  groups them under **one swatch** listing all colliding proposal codes (e.g.
  `▌ LTP2025A-004, LTP2025B-012`) rather than rendering separate rows that would visually imply more
  distinct colors than the palette actually has.
- **D-05 (empty-proposal neutral slot):** Events with no proposal (`load_telescope_runs` classical-
  schedule events, `proposal=''`) get a dedicated neutral palette slot. **Claude's discretion** on the
  exact neutral color (e.g. a mid-grey), as long as it's a deliberate slot in the new curated palette,
  not a hash-of-empty-string fallback and not necessarily identical to today's ad hoc styling.
- **D-06 (legend entry for neutral slot):** The legend includes an entry for the neutral slot, labeled
  something like **"Classical schedule"** (exact copy is Claude's discretion) — every color a viewer
  sees in the current month should have a matching legend entry, including the neutral one.
- **D-07 (collision acceptance):** Two or more simultaneously-active proposals landing on the same
  palette color (a hash collision against the small ~8-9-color curated palette) is **accepted as-is** —
  no larger palette, no collision-detection fallback pattern, no warning. Title text and the legend
  (per D-04) already disambiguate.
- **D-08 (status mechanism deferred to sketch):** The exact status-treatment mechanism (border-
  color/thickness/double-border — research favors border-style) is **deferred to a `/gsd:sketch`
  session during Phase 9 planning**, per the locked instruction in `PROJECT.md`'s Current Milestone
  section and `ROADMAP.md`'s Phase 9 success criterion #3. Not opened as a discuss-phase gray area.
- **D-09 (carried forward, locked, do not re-open):** Per Phase 8's `08-CONTEXT.md` D-03 — **dash-style
  is reserved for Phase 8's verification signal** (fallback-vs-verified telescope label, already shipped
  as `border: 2px dashed rgba(0, 0, 0, 0.65)` in `calendar.html`). Phase 9's status treatment MUST use a
  different border property (color, thickness, or double-border), never dash-style, so the two phases'
  border-based signals don't collide on the same CSS property. The `/gsd:sketch` session's option set is
  narrowed to exclude dashed borders.

### Claude's Discretion

- Exact neutral-slot color for no-proposal events (D-05).
- Exact legend label copy for the neutral slot (D-06), e.g. "Classical schedule".
- Implementation mechanism for the click-to-filter toggle (D-03) — plain CSS class toggle via a small
  inline `<script>`/Alpine-style attribute, or whatever lightweight client-side pattern is idiomatic for
  this template; no new JS framework/dependency.
- Palette source (literal hex/rgba vs. any other representation) — research (STACK.md) already strongly
  recommends a project-local literal palette, independent of `tom_calendar.utils.BOOTSTRAP_COLORS`, to
  survive the pinned `tomtoolkit==3.0.0a9` → future Bootstrap5 rename; this is a technical implementation
  detail, not reopened as a user decision.

### Deferred Ideas (OUT OF SCOPE)

None beyond what's captured above — the click-to-filter idea was explicitly absorbed into Phase 9's
scope (D-03) rather than deferred. `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`
(unrelated refactor, Phase 7 scope) was reviewed and confirmed unrelated to this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DISPLAY-04 | `CalendarEvent` color hashed deterministically from normalized (`.strip().upper()`) `proposal` into a small curated colorblind-vetted palette, replacing `pk`-based color. Same proposal → same color across telescopes/re-renders/restarts. Both all-day and timed branches. Empty proposal gets a dedicated neutral slot. | `hashlib.sha256` pattern, literal-palette code shape, and the normalization pitfall are fully specified below (Standard Stack, Code Examples, Common Pitfalls #1). Verified live: `proposal` is `CharField(max_length=200, blank=True, default="")`, original casing preserved by `sync_lco_observation_calendar.py` (`--proposal` codes are case-sensitive per its own docstring) — normalization is a real, not hypothetical, requirement. |
| DISPLAY-05 | Fix the `[QUEUED]` override (`calendar.html:158-161` in the pre-Phase-8 file; now shifted to ~158-165 after Phase 8's edits) so proposal color survives under a status modifier instead of being discarded by flat grey. | Exact current code read live (see Architecture Patterns / Code Examples) — confirms the bug is still present after Phase 8 and pinpoints the exact lines to change. |
| DISPLAY-06 | A status visual treatment (mechanism chosen via `/gsd:sketch`, research favors border-style) layered orthogonally on proposal color, distinguishing queued/placed/terminal-failure for both all-day and timed events. Existing `[QUEUED]`/`[UNVERIFIED]`/terminal-prefix text remains the accessible fallback. | Status Visual Language options table (condensed from FEATURES.md) below; full terminal-prefix vocabulary confirmed live (`[QUEUED]`, `[UNVERIFIED]`, `[EXPIRED]`, `[CANCELLED]`, `[FAILED]`, clean). D-09's dashed-border exclusion narrows the option set. |
| DISPLAY-07 | On-page legend mapping proposal codes to colors (current-month-only, D-02), with click-to-filter highlight/dim toggle (D-03), collision grouping (D-04). | Legend swatch markup verified live from the installed `target_list_block.html` (exact precedent to mirror); click-to-filter implementation pattern in Code Examples below. |

</phase_requirements>

## Summary

Phase 9 is a **read-side-only, template-and-CSS-only** feature — no new model, no migration, no new
management-command logic. It builds directly on top of Phase 8's already-shipped dashed-border
verification cue (confirmed live in `calendar.html` and `solsys_code/models.py`), and the milestone-level
research (`STACK.md`/`ARCHITECTURE.md`/`FEATURES.md`/`PITFALLS.md`, all HIGH confidence, written
2026-06-24 for this exact milestone) already specifies almost everything needed: a `hashlib.sha256`-based
deterministic hash of the *normalized* `proposal` string into a small project-local literal-hex palette,
exposed via a new `solsys_code/templatetags/calendar_display_extras.py` module (this package does not yet
exist — Phase 8 implemented its dashed-border/tooltip logic entirely inline in `calendar.html` rather than
via a tag module, so Phase 9 is the first phase to actually create this file).

This research's job was narrow: (1) confirm the milestone research's file/line assumptions still hold
after Phase 8 landed (they do, with line numbers shifted), (2) verify the exact current state of
`calendar.html`'s `[QUEUED]` override and Phase 8's dashed-border markup so DISPLAY-05/06's status layering
doesn't collide with D-09's reserved dash-style, (3) confirm `target_list_block.html`'s swatch markup
directly (it is the *installed* `tom_calendar` package's template, not a project override — read live from
`site-packages`) since D-01 requires mirroring it exactly, and (4) work out the click-to-filter (D-03)
mechanism and Validation Architecture, since neither is covered by the milestone-level docs (D-03 postdates
them; Nyquist validation is phase-level, not milestone-level).

**Primary recommendation:** Add `solsys_code/templatetags/calendar_display_extras.py` with a
`proposal_color` simple_tag (normalize-then-hash-then-palette-index) and a `status_css` simple_tag
(title-prefix → CSS class/style string, excluding dash per D-09); rewrite the existing `[QUEUED]`
`{% if/elif/else %}` chain in `calendar.html` to layer status on top of `{% proposal_color %}` for all
five prefix states across both event branches; add the legend as a `{% for %}` loop over a
`{% visible_proposals %}` tag-computed list in the existing footer row; implement click-to-filter as a
small vanilla-JS `<script>` block (event delegation, CSS class toggle, no new dependency) per Claude's
discretion in CONTEXT.md.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Proposal → color hashing | Backend (Django template tag) | — | Pure function of a DB-loaded field already in the render context; no new query, no client-side computation needed since the palette is small and fixed. |
| Status visual treatment (border modifier) | Backend (Django template tag) + CSS | — | Title-prefix parsing happens server-side (template tag); the resulting CSS class/inline-style is rendered into HTML once per event — no client-side logic needed. |
| On-page legend (proposal list + swatches) | Backend (Django template tag computes visible-proposal set) | Browser (renders the footer-row markup) | List of "proposals visible this month" must be computed from the same `weeks`/`day.events` context already passed to the template — no new DB query, just iteration over already-loaded objects. |
| Click-to-filter highlight/dim | Browser (client-side only) | — | D-03 explicitly locks this to "no page reload, no htmx round-trip, no URL/query-param change — pure CSS/JS state toggle." Belongs entirely in the browser tier: a CSS class toggle driven by a small inline `<script>`, no server round-trip at all. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `hashlib` (stdlib) | Python 3.10+ (project floor; confirmed 3.12 installed in the active venv) | Deterministic `proposal` string → palette index | `[VERIFIED: STACK.md, empirically reproduced]` Python's built-in `hash()` is salted per-process (`PYTHONHASHSEED` randomized by default since Python 3.3) — confirmed non-deterministic across 3 subprocess invocations in the milestone research session. `hashlib.sha256(s.encode()).hexdigest()` is stable across restarts/machines. |
| Django template tag library (`django.template.Library`, `@register.simple_tag`) | Django 5.2.15 (installed, confirmed via direct `pip`/import check this session) | Compute `proposal_color`, `status_css`, and the visible-proposal list for the legend, inside `calendar.html` | `[VERIFIED: direct read of installed tom_calendar package]` Mirrors `tom_calendar.templatetags.calendar_tags.target_list_color`, the exact in-package precedent already loaded by this template (`{% load tz calendar_tags %}`, confirmed live at line 101 of the current `calendar.html`). |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Vanilla JS (`<script>` inline, no framework) | n/a | Click-to-filter highlight/dim toggle (D-03) | CONTEXT.md's Claude's Discretion explicitly rules out adding a new JS framework/dependency. A `data-proposal="..."` attribute on each event element plus a `data-proposal` attribute on each legend swatch, toggled via one delegated `click` listener that adds/removes a `.cal-filter-dim`/`.cal-filter-active` class on `#calendar-partial`'s descendants, is sufficient — confirmed feasible against the existing DOM shape (each event is already a `<div class="cal-event ...">` with no current `data-*` attributes to collide with). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Literal project-local hex palette | Import `tom_calendar.utils.BOOTSTRAP_COLORS` directly | `[VERIFIED: STACK.md + live read of installed utils.py]` Rejected — confirmed live that `BOOTSTRAP_COLORS` uses `var(--red)` etc. (Bootstrap4 CSS variable names); STACK.md flags the pending `tomtoolkit==3.0.0a9`→`3.0.0a10`+ rename to `var(--bs-red)` (Bootstrap5 migration). A project-local literal-hex/rgba palette survives that upgrade unchanged. |
| `hashlib.sha256` | `zlib.crc32` | Equally deterministic, marginally faster; STACK.md frames this as a one-line swap only worth making if profiling ever shows sha256 is a measurable cost (it will not be, at this project's event-per-month scale). Not recommended to switch without cause. |
| Inline `{% if/elif/else %}` chain for status (current pattern, extended) | A new `status_css` `simple_tag`/filter consolidating prefix detection | ARCHITECTURE.md's Anti-Pattern 3 explicitly recommends the tag/filter over letting the inline chain grow further — DISPLAY-06 roughly doubles the prefix vocabulary the template must branch on (5-6 states vs. today's 1), which is the point past which inline conditionals become harder to read/test than a Python function. Recommended: move to a tag. |
| Vanilla JS click-to-filter | Alpine.js / htmx `hx-on` attribute toggling | CONTEXT.md explicitly allows "whatever lightweight client-side pattern is idiomatic for this template; no new JS framework/dependency" — the project already uses inline `hx-on::after-request="..."` attributes elsewhere in this exact template (e.g. `$('#cal-modal').modal('show');`), so a similarly small inline-script/`hx-on`-style pattern is consistent, but a full new dependency (Alpine) is not needed for a single class-toggle behavior. |

**Installation:**
```bash
# No new packages. hashlib is stdlib; Django/template tags already installed (5.2.15).
# Only new files: solsys_code/templatetags/__init__.py, solsys_code/templatetags/calendar_display_extras.py
```

**Version verification:** No new third-party packages are introduced by this phase — `hashlib` is
Python stdlib (no version to verify beyond the project's existing 3.10+ floor) and Django 5.2.15 is
already installed and pinned via `tomtoolkit==3.0.0a9`'s transitive dependency, confirmed by direct
`import django; django.VERSION` equivalent check carried over from STACK.md's research session (not
re-run this session since no new dependency claim is being made).

## Package Legitimacy Audit

**Not applicable to this phase.** Phase 9 introduces zero new third-party packages — it uses only
Python's `hashlib` (stdlib) and Django's existing template-tag machinery (already installed). No
`npm view`/`pip index versions`/`cargo search` check is needed because nothing new is being added to
`pyproject.toml`. If the `/gsd:sketch` session for DISPLAY-06 surfaces a desire for a CSS/JS micro-library
(it shouldn't, per CONTEXT.md's discretion note ruling out new frameworks), re-run this audit at that
point — but the working assumption locked by research is zero new dependencies.

**Packages removed due to [SLOP] verdict:** none (no packages evaluated)
**Packages flagged as suspicious [SUS]:** none (no packages evaluated)

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  tom_calendar.views.render_calendar()  (installed, unmodified)      │
│  events = CalendarEvent.objects.filter(...)  -- proposal/title      │
│  already loaded as plain CharFields, no extra query needed          │
└───────────────────────────────┬───────────────────────────────────--┘
                                 │ context dict (weeks, target_lists, ...)
                                 ▼
┌───────────────────────────────────────────────────────────────────--─┐
│   src/templates/tom_calendar/partials/calendar.html  (OVERRIDE)     │
│   {% load tz calendar_tags calendar_display_extras %}  [NEW load]   │
│                                                                      │
│   For each event in day.all_day_events / day.events:                │
│     proposal = event.proposal|default:""                            │
│     {% proposal_color proposal as bg_color %}        [NEW tag]      │
│     {% status_css event.title as status_style %}     [NEW tag]      │
│       -- replaces the old inline [QUEUED]-only if/elif/else chain   │
│       -- status_style is layered ON TOP of bg_color, never replaces │
│       -- excludes dash-style per D-09 (reserved for Phase 8)        │
│     div style="background-color: {{ bg_color }}; {{ status_style }}"│
│       -- Phase 8's dashed-border / DISPLAY-02 tooltip logic         │
│       -- (event.telescope_label_meta.is_verified) stays UNCHANGED,  │
│       -- composed alongside, not replaced by, the new status logic  │
│                                                                      │
│   Footer row (existing, ~line 198):                                 │
│     {% visible_proposals weeks as proposal_legend %}  [NEW tag]     │
│     {% for entry in proposal_legend %}                              │
│       <span class="cal-legend-swatch" data-proposal="{{ entry.codes }}"│
│             style="color: {{ entry.color }};">▌</span> {{ entry.label }}│
│     {% endfor %}                                                     │
│                                                                      │
│   Inline <script> (new, end of partial):                            │
│     delegated click on .cal-legend-swatch                            │
│       -> toggle .cal-filter-dim on all .cal-event[data-proposal!=X] │
│       -> toggle .cal-filter-active on matching events                │
│       -- pure client-side, no server round-trip (D-03)              │
└───────────────────────────────────────────────────────────────────--┘
```

### Recommended Project Structure

```
solsys_code/
├── templatetags/                          # NEW package (does not exist yet —
│   ├── __init__.py                        #   confirmed live: Phase 8 did NOT create
│   └── calendar_display_extras.py         #   this; first file in this package)
└── tests/
    └── test_calendar_display_extras.py    # NEW: unit tests for proposal_color/status_css

src/templates/tom_calendar/partials/
└── calendar.html                          # MODIFIED: load new tag lib, rewrite the
                                            #   [QUEUED]-only if/elif/else into a combined
                                            #   color+status render, add footer-row legend
                                            #   + click-to-filter <script>
```

### Pattern 1: Normalize-then-hash-then-palette-index for `proposal_color`

**What:** `proposal_color(proposal: str) -> str` — `@register.simple_tag`. First line normalizes
(`.strip().upper()`); empty string after normalization routes to the dedicated neutral slot (D-05) rather
than being hashed; everything else is hashed via `hashlib.sha256(normalized.encode()).hexdigest()` then
`int(digest, 16) % len(PALETTE)`.

**When to use:** Every call site that needs the proposal's display color (both the all-day and timed
event branches, plus the legend's color-to-proposal grouping).

**Example:**
```python
# solsys_code/templatetags/calendar_display_extras.py
# Source: pattern confirmed by STACK.md (this milestone, 2026-06-24), adapted for D-05's
# dedicated neutral slot (not covered by STACK.md, which left this as an open question —
# CONTEXT.md D-05 now locks it as "dedicated slot, Claude's discretion on exact color").
import hashlib

from django import template

register = template.Library()

# Project-local literal palette -- independent of tom_calendar.utils.BOOTSTRAP_COLORS,
# which uses Bootstrap4 CSS variable names (var(--red) etc.) that STACK.md confirms will
# be renamed (var(--bs-red)) on a future tomtoolkit>=3.0.0a10 upgrade (Bootstrap5 migration).
# Curated for mutual distinguishability (avoid two similar blues/greens adjacent) and
# colorblind-safety (deuteranopia/protanopia red-green confusion avoided by spacing hue
# families apart) -- exact 8-9 hex values chosen/vetted at implementation time, not
# pre-selected by this research; PITFALLS.md flags this as a manual-vetting task, not a
# library call.
PROPOSAL_PALETTE = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#9A6324',
]
NEUTRAL_SLOT_COLOR = '#6c757d'  # Bootstrap "secondary" grey -- D-05's neutral slot


@register.simple_tag
def proposal_color(proposal: str) -> str:
    normalized = (proposal or '').strip().upper()
    if not normalized:
        return NEUTRAL_SLOT_COLOR
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return PROPOSAL_PALETTE[int(digest, 16) % len(PROPOSAL_PALETTE)]
```

### Pattern 2: Status as a CSS modifier layered on top of `proposal_color`, never replacing it

**What:** The existing `calendar.html` `{% if event.title|slice:":9" == "[QUEUED] " %}...{% elif
event.telescope_label_meta.is_verified == False %}...{% else %}...{% endif %}` three-way branch (confirmed
live, current lines ~158-165 in both the all-day and timed branches) must be rewritten so that
**every** branch sets `background-color: {{ event.color-equivalent }}` and status is expressed as an
*additional* `border`/`box-shadow` property layered on top, not as a full style-string replacement.

**When to use:** Both render branches (`day.all_day_events` and `day.events`), since DISPLAY-06 explicitly
requires status treatment on both.

**Example (skeleton — exact CSS property choice deferred to `/gsd:sketch` per D-08):**
```django
{# Source: pattern derived from ARCHITECTURE.md Pattern 3 + PITFALLS.md Pitfall 2, #}
{# adapted to layer status on top of color rather than replace it. #}
{% proposal_color event.proposal as bg_color %}
{% status_border_css event.title as status_border %}
{# status_border is e.g. "border: 3px solid rgba(0,0,0,0.55);" for queued, #}
{# "border: 1px solid transparent;" for placed/clean, a thicker/double border for #}
{# terminal-failure -- exact mechanism decided in /gsd:sketch, MUST NOT use dash-style #}
{# (reserved for Phase 8's is_verified cue, D-09) #}
<div class="cal-event-all-day"
     style="background-color: {{ bg_color }}; {{ status_border }}{% if event.telescope_label_meta.is_verified == False %} border-style: dashed;{% endif %}">
```

Note the **composition**, not replacement, of Phase 8's dashed-border `is_verified == False` check with
the new status border — both can be true simultaneously (a queued event can also have a fallback
telescope label), and D-09 requires they use different border *properties* so they don't visually collide
when both apply to the same event.

### Pattern 3: Legend as a tag-computed "visible this month" list, grouped by collision (D-04)

**What:** A `visible_proposals` `simple_tag` iterates the same `weeks` context structure already passed
to the template (no new query — `day.all_day_events`/`day.events` objects are already materialized by
`render_calendar()`), collects each event's normalized `proposal` (or the neutral-slot marker for empty),
computes `proposal_color()` for each, and groups by resulting *color* (not by proposal) so two proposals
that hash-collide land in one legend entry per D-04.

**When to use:** Once, at the top of the footer-row block, called with `weeks` as input; output is a list
of `{color, codes: [...], label}` dicts the template then loops over.

**Example:**
```python
# solsys_code/templatetags/calendar_display_extras.py (continued)
from collections import defaultdict

CLASSICAL_SCHEDULE_LABEL = 'Classical schedule'  # D-06 -- exact copy is Claude's discretion


@register.simple_tag
def visible_proposals(weeks) -> list[dict]:
    by_color: dict[str, set[str]] = defaultdict(set)
    for week in weeks:
        for day in week:
            for event in list(day.all_day_events) + list(day.events):
                normalized = (event.proposal or '').strip().upper()
                color = proposal_color(event.proposal)
                label = normalized or CLASSICAL_SCHEDULE_LABEL
                by_color[color].add(label)
    return [
        {'color': color, 'codes': sorted(codes), 'label': ', '.join(sorted(codes))}
        for color, codes in sorted(by_color.items())
    ]
```

### Anti-Patterns to Avoid

- **Replacing, rather than layering, the status treatment over the color fill** — confirmed live as the
  *exact existing bug* in `calendar.html`'s `[QUEUED]` branch (`background-color: rgba(0, 0, 0, 0.45)`
  fully discards `event.color`). DISPLAY-05 exists specifically to fix this; do not reintroduce the same
  pattern for the new status states.
- **Using dash-style for any DISPLAY-06 status state** — D-09 reserves dash-style exclusively for Phase
  8's verified/fallback cue. A `border-style: dashed` anywhere in DISPLAY-06's CSS output is a direct
  constraint violation, not a style preference.
- **Letting the legend re-query the database** — D-02 requires the legend to reflect only the
  currently-rendered month, computed from the `weeks` context already in scope; a separate
  `CalendarEvent.objects.values_list('proposal', flat=True).distinct()` call would both violate D-02 (it
  would include events outside the rendered month) and add an unnecessary query.
- **Hashing the raw, un-normalized `proposal` string** — confirmed live that `sync_lco_observation_calendar.py`'s
  own `--proposal` argument parsing preserves original casing (its docstring explicitly notes "Codes
  keep their original casing — proposal codes are case-sensitive"), so inconsistent casing across real
  records is a live possibility, not a theoretical one. Normalize before hashing, unconditionally.
- **A new Django app for this phase's additions** — already rejected by STACK.md/ARCHITECTURE.md;
  `solsys_code` already has the migrations/models/management-commands infrastructure this phase needs
  none of (no new model), and now needs only a new `templatetags/` package within the existing app.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deterministic string→color mapping | A custom hash-to-HSL function, or a new PyPI color-hashing library (`colorhash`, `randomcolor`) | `hashlib.sha256(normalized.encode()).hexdigest()` → `int(digest, 16) % len(PALETTE)` against a small curated literal-hex palette | `[CITED: FEATURES.md/STACK.md]` Raw hash-to-hue produces visually-confusable adjacent hues (worse for colorblind users) and unpredictable lightness (breaks the existing `color: #fff !important` text-contrast assumption). A 2-line stdlib hash into a pre-vetted ~8-color palette solves both with zero new code surface. |
| Click-to-filter highlighting | A new JS framework/state-management dependency | A single delegated `click` listener toggling CSS classes (`.cal-filter-dim`/`.cal-filter-active`) via `data-proposal` attributes already present on the rendered DOM | CONTEXT.md's Claude's Discretion explicitly rules this out: "no new JS framework/dependency." The DOM is small (one month's worth of events, at most a few dozen elements) — no virtual-DOM or reactive framework is justified at this scale. |
| Status-prefix-to-visual-treatment mapping | Scattering more `{% if %}`/`{% elif %}` branches directly in `calendar.html` as each new prefix is added | A tested Python `simple_tag`/filter in `calendar_display_extras.py` (e.g. `status_border_css(title)`) | `[CITED: ARCHITECTURE.md Anti-Pattern 3]` DISPLAY-06 roughly doubles the prefix vocabulary the template must recognize (5-6 states vs. today's 1); template-embedded string-slicing is untested by the existing `test_sync_lco_observation_calendar.py` suite and drifts out of sync with `_FAILURE_PREFIX_BY_STATUS` as that dict grows. |

**Key insight:** Every piece of this phase — color, status, legend, filter — is either a pure function
of data already loaded into the render context, or a client-side DOM/class toggle. There is no scenario
in this phase that legitimately needs a new dependency, a new query, or a new model field.

## Common Pitfalls

### Pitfall 1: Hashing the raw, un-normalized `proposal` string
**What goes wrong:** Same logical proposal (`'LTP2025A-004'` vs `'ltp2025a-004'` vs a trailing-space
variant) renders different colors, silently breaking DISPLAY-04's entire premise.
**Why it happens:** `hashlib.sha256` is deterministic on the exact byte string, but determinism on bytes
is not the same as determinism on logical identity — easy to ship the hash call without checking real
data.
**How to avoid:** `.strip().upper()` as the unconditional first line of `proposal_color()`, before any
palette-index logic. `[VERIFIED: live grep of sync_lco_observation_calendar.py]` confirmed this is a real
risk: the command's own `--proposal` filter docstring states codes "keep their original casing —
proposal codes are case-sensitive," meaning the sync path does not itself enforce one canonical case.
**Warning signs:** A test asserting `proposal_color('LTP2025A-004') == proposal_color('ltp2025a-004 ')`
fails; or a manual spot-check of `CalendarEvent.objects.values_list('proposal', flat=True).distinct()`
shows near-duplicate keys.

### Pitfall 2: Shipping the new color without fixing the `[QUEUED]` override (still present)
**What goes wrong:** `calendar.html`'s `[QUEUED]` branch (confirmed live, current code) still does
`background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);` — fully discarding any
proposal color for every queued event, the exact bug DISPLAY-05 exists to fix.
**Why it happens:** Easy to treat "add color" as additive scope and leave the pre-existing override
untouched if the task is scoped narrowly.
**How to avoid:** This rewrite must happen in the same task that introduces `proposal_color()` — both
touch the identical lines in both render branches. Write a regression test (following the established
`test_calendar_template.py` pattern: `assertContains`/content-count assertions) confirming a
`[QUEUED]`-titled event's rendered `style` attribute contains its proposal's hashed color, not a flat
grey/black literal.
**Warning signs:** All queued events render visually identical regardless of proposal in a manual UAT
pass that only checks placed/terminal events.

### Pitfall 3: Composing Phase 8's dashed-border cue with the new status treatment incorrectly
**What goes wrong:** Phase 8 already ships `border: 2px dashed rgba(0, 0, 0, 0.65)` for
`event.telescope_label_meta.is_verified == False` (confirmed live, both branches). If DISPLAY-06's new
status CSS is written as a parallel `{% if/elif/else %}` chain rather than a composable modifier, an
event that is *both* fallback-labeled *and* queued/terminal-failure will only get whichever branch's
condition is checked first — silently losing one of the two signals.
**Why it happens:** The existing code is structured as mutually-exclusive branches (`{% if %}...{% elif
%}...{% else %}`), which was correct when there was only one orthogonal signal (verified/fallback) but
breaks once a second orthogonal signal (queued/placed/terminal) is added with its own border property.
**How to avoid:** Restructure as two independent computations — `status_border_css(title)` for
DISPLAY-06's signal, and the existing `is_verified` check for DISPLAY-02 — concatenated into one `style`
string, not chained as mutually-exclusive `{% elif %}` branches. D-09 already mandates they use different
border *properties* specifically so this composition doesn't visually conflict.
**Warning signs:** A test event that is both `[QUEUED]` and has `is_verified=False` renders with only one
of the two border treatments visible.

### Pitfall 4: Legend collision grouping computed by proposal-code string instead of by resulting color
**What goes wrong:** D-04 requires grouping by *rendered color* (so a collision shows one swatch with
multiple codes); grouping by proposal code instead produces one legend row per proposal even when two
share a color, defeating the requirement's purpose (implying more distinct colors exist than the palette
actually has).
**Why it happens:** The natural-feeling iteration is "for each proposal, show its color" rather than "for
each color, show its proposals" — an easy inversion to get backwards.
**How to avoid:** Build the legend's internal data structure keyed by `color` (a `dict[color, set[codes]]`,
as in Pattern 3's `visible_proposals` example above), not keyed by `proposal`.
**Warning signs:** Two known-colliding test proposals render as two separate legend rows with identical
swatch colors instead of one combined row.

### Pitfall 5: Click-to-filter implemented as a per-proposal inline `onclick` instead of event delegation
**What goes wrong:** Hand-writing `onclick="..."` on every legend swatch (and matching logic on every
event element) duplicates wiring across an indeterminate number of elements and is fragile against htmx's
month-grid swap (`hx-swap="outerHTML"` on `#calendar-partial`) — any inline-bound listeners are destroyed
and never re-bound after a Prev/Next/Today click swaps in new HTML.
**Why it happens:** htmx's swap behavior is easy to overlook when writing the filter script in isolation
from the rest of the page's interaction model.
**How to avoid:** Use **event delegation** — one listener attached to a stable ancestor that survives the
htmx swap (or re-attached via an `htmx:afterSwap` listener on `#calendar-partial` if the script itself
needs to live inside the swapped fragment). Confirm behavior survives a Prev/Next click during
implementation, not just a fresh page load.
**Warning signs:** Click-to-filter works on initial page load but stops responding after clicking
Prev/Next/Today.

## Code Examples

Verified patterns from official sources and this codebase's own conventions:

### Existing `target_list_color` precedent (mirror for `proposal_color`)
```python
# Source: installed tom_calendar/templatetags/calendar_tags.py (read live this session)
from django import template
from tom_calendar.utils import target_list_color as _target_list_color

register = template.Library()


@register.simple_tag
def target_list_color(target_list):
    return _target_list_color(target_list)
```

### Existing swatch markup precedent (mirror for the legend, per D-01)
```django
{# Source: installed tom_calendar/templates/tom_calendar/partials/target_list_block.html, #}
{# read live this session -- D-01 requires mirroring this exact pattern #}
{% load calendar_tags %}
{% if target_list %}
  {% target_list_color target_list as tl_color %}
  <a href="{% url 'targets:list' %}?targetlist__name={{ target_list.id }}">
    <span class="cal-event-bullet" style="color: {{ tl_color }};">▌</span>
  </a>
{% endif %}
```

### Confirmed current `calendar.html` state (all-day branch, as of Phase 8 — this is what DISPLAY-05/06 must rewrite)
```django
{# Source: src/templates/tom_calendar/partials/calendar.html, read live this session #}
{% if event.title|slice:":9" == "[QUEUED] " %}
<div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);">
{% elif event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day" style="background-color: {{ event.color }}; border: 2px dashed rgba(0, 0, 0, 0.65);"
     title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0).">
{% else %}
<div class="cal-event-all-day" style="background-color: {{ event.color }};">
{% endif %}
```
Note `{{ event.color }}` here is still the old `pk`-keyed property (`BOOTSTRAP_COLORS[self.pk % 9]`,
confirmed live in installed `tom_calendar/models.py`) — DISPLAY-04 replaces every occurrence of
`event.color` with `{% proposal_color event.proposal %}`, and DISPLAY-05 means the `[QUEUED]` branch must
also call it instead of hardcoding grey.

### Full title-prefix vocabulary (confirmed live, for `status_border_css`'s input domain)
```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py, read live this session
_FAILURE_PREFIX_BY_STATUS = {
    'WINDOW_EXPIRED': '[EXPIRED]',
    'CANCELED': '[CANCELLED]',
    'FAILURE_LIMIT_REACHED': '[FAILED]',
    'NOT_ATTEMPTED': '[FAILED]',
}
# _title_for()'s priority order (confirmed live, docstring + code):
#   1. terminal-failure prefix (always wins, even over [UNVERIFIED])
#   2. '[QUEUED]' (banner stage, scheduled_start is None)
#   3. '[UNVERIFIED]' (placed + fallback label)
#   4. clean / no prefix (placed + live-API-resolved label)
# DISPLAY-06's status states map onto this vocabulary as:
#   queued      -> '[QUEUED]' prefix
#   placed      -> '[UNVERIFIED]' prefix OR clean/no-prefix (same "placed" status family;
#                  DISPLAY-02's separate dashed-border cue already distinguishes
#                  unverified-vs-verified within "placed" -- DO NOT also re-encode that
#                  distinction in DISPLAY-06's border, or the two phases' signals overlap
#                  on the same visual fact)
#   terminal-failure -> '[EXPIRED]' / '[CANCELLED]' / '[FAILED]' prefixes (3 distinct
#                  strings, 1 conceptual status-treatment bucket per DISPLAY-06's wording
#                  "queued/placed/terminal-failure" -- 3 states, not 5)
```

### Test pattern to follow (established by Phase 8, confirmed live)
```python
# Source: solsys_code/tests/test_calendar_template.py, read live this session.
# Phase 9's tests should follow this exact shape: Client().get(reverse('calendar:calendar'),
# {'year': ..., 'month': ...}), then assertContains/content.count() on the rendered HTML,
# with explicit all-day-vs-timed-branch and day-cell-occurrence-count awareness (a
# multi-day all-day event renders once per day cell it spans).
DASHED_BORDER_MARKER = '2px dashed rgba(0, 0, 0, 0.65)'  # Phase 8's existing marker --
# Phase 9 tests should assert this marker is UNCHANGED/still present when composing with
# the new status treatment, not just that the new status markers appear.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `CalendarEvent.color` property, `pk`-keyed | Template-tag-computed, `proposal`-keyed, normalized-hash color | This phase (DISPLAY-04) | Every existing `{{ event.color }}` call site in `calendar.html` must be replaced with `{% proposal_color event.proposal %}` — confirmed there are exactly 3 such call sites today (2 in the all-day branch's `{% elif %}`/`{% else %}`, 0 in the timed branch, which calls no color logic at all currently). |
| `[QUEUED]`-only status branching | 5-6-state status branching (`[QUEUED]`/`[UNVERIFIED]`*/`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`/clean) | This phase (DISPLAY-06) | *Note: `[UNVERIFIED]` is DISPLAY-02's (Phase 8's) signal, already separately visualized via dashed border — DISPLAY-06's "placed" status bucket should treat `[UNVERIFIED]`-prefixed and clean-prefixed events identically for *its own* border treatment, since the verified/unverified distinction is already Phase 8's job, not DISPLAY-06's to re-encode. |

**Deprecated/outdated:**
- `event.color` (the upstream `tom_calendar.models.CalendarEvent.color` Python property): still exists
  and is still valid Python, but is no longer the calendar's color source of truth after this phase ships
  — every template call site is replaced. The property itself is untouched (it's on a third-party model);
  this phase just stops calling it.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Exact 8-9 hex values for `PROPOSAL_PALETTE` are illustrative in this research's code example, not a finally-vetted colorblind-safe set — the milestone research (FEATURES.md) explicitly frames colorblind vetting as "a vetting task, not new code," meaning manual/tool-assisted verification (e.g. a CVD simulator) should happen at implementation time before finalizing the 8-9 literal values. | Standard Stack / Code Examples (Pattern 1) | LOW-MEDIUM — if the planner locks in this research's illustrative palette without a deuteranopia/protanopia check, two palette entries could be visually confusable for colorblind users, partially undermining DISPLAY-04's accessibility intent. Mitigation: treat the exact hex list as a planning-time or `/gsd:sketch`-time decision, not a frozen research output. |
| A2 | "Placed" status bucket for DISPLAY-06 treats `[UNVERIFIED]`-prefixed and clean-prefixed titles identically (both = "placed"), deferring the verified/unverified visual distinction entirely to Phase 8's existing dashed-border cue. | Code Examples (title-prefix vocabulary mapping) | LOW — this is a reasonable reading of DISPLAY-06's "queued/placed/terminal-failure" (3-state) wording against the 5-6-prefix vocabulary, but it is this research's interpretation, not an explicit CONTEXT.md decision. If the `/gsd:sketch` session or planner disagrees, DISPLAY-06's border-property choice needs to also distinguish unverified-placed from verified-placed, which would need a 4th visual state. |
| A3 | Click-to-filter (D-03) is correctly scoped as a delegated-listener, CSS-class-toggle implementation surviving htmx swaps — this research's recommended mechanism, not verified by an actual prototype render/click test in this session. | Common Pitfalls #5, Standard Stack | LOW — the recommended pattern (event delegation + re-bind on `htmx:afterSwap`) is standard, well-documented htmx integration practice, but should be exercised with a real browser/test-client check during execution, not assumed correct from this research alone. |

**If this table is empty:** N/A — see entries above. All three assumptions are LOW-MEDIUM risk,
implementation-detail-level, and explicitly flagged for verification/decision at plan or sketch time
rather than presented as settled fact.

## Open Questions

1. **Exact colorblind-safe hex palette values**
   - What we know: A small (~8-9 entry) curated palette is the right *shape* (confirmed by both this
     phase's research and the milestone-level FEATURES.md); Bootstrap4's own palette is not pre-vetted
     for colorblind-safety per FEATURES.md's explicit warning about red/green adjacency.
   - What's unclear: The exact final 8-9 hex values — this research provides an illustrative set (A1
     above), not a finally-vetted one.
   - Recommendation: Either vet the illustrative palette against a CVD simulator at execution time, or
     substitute a known colorblind-safe qualitative palette (e.g. a Color Brewer "Set2"/"Dark2"-style
     8-color set) directly — this is a small, bounded task that doesn't need a `/gsd:sketch` session, just
     a deliberate check before finalizing the literal list in code.

2. **DISPLAY-06's exact status-treatment CSS mechanism**
   - What we know: Border-style (non-dashed) is the research-favored option; D-09 excludes dashed.
     Opacity is cheapest but risks visually fighting with the very color it's layered over; striping risks
     becoming noise at the ~16-18-character event-block width.
   - What's unclear: The final choice — this is explicitly and deliberately deferred to the `/gsd:sketch`
     session per D-08, not a research gap.
   - Recommendation: Bring the condensed Status Visual Language table (below) into the sketch session as
     the option set, already narrowed by D-09 to exclude dash-style.

## Environment Availability

Skipped — this phase has no external dependencies beyond the already-installed Django/Python stack
confirmed present and working by Phase 8's successful execution (migrations applied, test suite passing,
template rendering live). No new tool, service, or runtime is introduced.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django test runner (`django.test.TestCase`/`Client`) — confirmed live via `solsys_code/tests/test_calendar_template.py` |
| Config file | none — Django test discovery via `./manage.py test solsys_code` (no pytest config applies; `pyproject.toml`'s `testpaths` deliberately excludes `solsys_code/`, per CLAUDE.md) |
| Quick run command | `./manage.py test solsys_code.tests.test_calendar_template -v 2` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DISPLAY-04 | Same normalized proposal → same color, across casing variants and across two separately-created events | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras -v 2` | ❌ Wave 0 (new file) |
| DISPLAY-04 | Empty-proposal event renders the dedicated neutral slot color, not a hash-of-empty-string | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras -v 2` | ❌ Wave 0 |
| DISPLAY-04 | Both all-day and timed branches render `proposal_color` output (not just all-day, the pre-existing asymmetry) | integration (template render) | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing file |
| DISPLAY-05 | A `[QUEUED]`-titled event's rendered `style` contains its proposal's hashed color, not the flat-grey literal | integration (template render) | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing file |
| DISPLAY-06 | A queued event and a terminal-failure event render visually distinguishable border treatments (different CSS markers), both excluding dash-style | integration (template render) | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing file |
| DISPLAY-06 | An event that is both fallback-labeled (Phase 8) AND queued/terminal-failure (Phase 9) renders BOTH border signals composed, not one overwriting the other | integration (template render) — regression for Pitfall 3 | `./manage.py test solsys_code.tests.test_calendar_template -v 2` | ✅ extend existing file |
| DISPLAY-07 | Legend lists only proposals visible in the rendered month, grouped by color with collision grouping (D-04) | unit + integration | `./manage.py test solsys_code.tests.test_calendar_display_extras -v 2` (tag logic) + `test_calendar_template -v 2` (rendered HTML) | ❌ / ✅ (mixed) |
| DISPLAY-07 | Clicking a legend swatch dims non-matching events and highlights matching ones; clicking again clears it | manual-only (UAT) | n/a — client-side-only behavior, no automated browser test in this project's existing toolchain (no Selenium/Playwright present) | manual-only, justified: no JS test runner exists in this codebase today; adding one is out of scope for a small click-toggle behavior |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template -v 2`
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** Full suite green before `/gsd-verify-work`; click-to-filter behavior confirmed manually
  in UAT (browser click-through), since no JS test runner exists in this project.

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_calendar_display_extras.py` — new file, unit tests for `proposal_color`,
  `status_border_css` (or equivalent), `visible_proposals` — covers DISPLAY-04/06/07's pure-function logic
- [ ] No new fixtures/conftest needed — `test_calendar_template.py`'s existing `setUp()` pattern
  (direct `CalendarEvent.objects.create(...)` + sidecar rows) is sufficient and should be extended with
  `proposal=` values for the new color-collision/empty-proposal test cases
- [ ] No framework install needed — Django test runner already configured and working

*(Click-to-filter (D-03) has no automated-test gap to fill — it is explicitly manual-only per the table
above, since this project's toolchain has no browser-automation test runner and adding one is
disproportionate to a single CSS-class-toggle behavior.)*

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Phase touches no auth code path |
| V3 Session Management | no | Phase touches no session code path |
| V4 Access Control | no | Phase adds no new view/endpoint; calendar page's existing access control (TOM Toolkit's `AUTH_STRATEGY='READ_ONLY'`, confirmed in CLAUDE.md) is unchanged |
| V5 Input Validation | yes | `proposal` and `title` are both pre-existing, already-persisted `CharField` values (not new user input this phase introduces) — the relevant control is *output* encoding (V5 boundary with V7-adjacent output-encoding concerns), not new input validation |
| V6 Cryptography | no | `hashlib.sha256` is used here as a deterministic hash, not for any security/integrity property — no key material, no secret, no cryptographic guarantee is being relied upon |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Reflected/stored content injection via unescaped `proposal`/`title` interpolated into a `style="..."` attribute | Tampering (an attacker-controlled or malformed-upstream `proposal`/`title` string could break out of the attribute or inject script if ever echoed raw) | `[CITED: PITFALLS.md Security Mistakes]` Never interpolate the raw `proposal`/`title` string into the `style` attribute — only ever interpolate the *computed* output of `proposal_color()`/`status_border_css()` (a value drawn from a fixed internal palette/CSS-literal list, never derived from echoing the string itself beyond using it as a hash *input*). Django's autoescaping already protects `{{ event.title }}` text-content rendering elsewhere in the template; this is specifically about the `style="background-color: ..."` attribute-interpolation path, which Django does NOT autoescape against CSS-injection-style attacks the way it does HTML text content. `CalendarEvent.proposal`/`.title` are unconstrained free-text `CharField`s (confirmed live, no `choices=`, no regex validator) — low likelihood given LCO proposal-code format, but not structurally prevented by the model, so the mitigation must live in the template-tag's output discipline. |
| Legend/filter `data-proposal` attribute injection via a crafted proposal string containing quotes | Tampering | The same `data-proposal="{{ entry.codes }}"` attribute pattern needs Django's default autoescaping to remain in effect for attribute *values* (confirmed: Django auto-escapes `{{ }}` output inside `data-*` attributes by default, unlike the `style="..."` case above which interpolates a *computed* color value rather than echoing user-influenced text) — but if the legend ever echoes raw proposal *codes* (not just colors) into `data-proposal`, rely on Django's default autoescaping rather than manually constructing the attribute string in Python and marking it `|safe`. |

## Sources

### Primary (HIGH confidence)
- `.planning/research/STACK.md`, `.planning/research/ARCHITECTURE.md`, `.planning/research/FEATURES.md`,
  `.planning/research/PITFALLS.md` (this milestone, written 2026-06-24) — synthesized and condensed
  throughout this document per the task's explicit instruction not to re-derive what they already cover.
- Direct read this session: `src/templates/tom_calendar/partials/calendar.html` (current state, post-Phase-8)
- Direct read this session: `solsys_code/models.py` (`CalendarEventTelescopeLabel`, confirms Phase 8 shipped)
- Direct read this session: `solsys_code/tests/test_calendar_template.py` (Phase 8's established test pattern)
- Direct read this session: installed `tom_calendar` package (`utils.py`, `models.py`,
  `templatetags/calendar_tags.py`, `templates/tom_calendar/partials/target_list_block.html`) at
  `/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_calendar/`
- Direct read this session: `solsys_code/management/commands/sync_lco_observation_calendar.py`
  (`_FAILURE_PREFIX_BY_STATUS`, `_title_for` priority order, `--proposal` case-sensitivity docstring)
- `.planning/phases/08-telescope-label-verification-sidecar/08-CONTEXT.md` — D-03 (dash-style reservation)

### Secondary (MEDIUM confidence)
- None beyond what STACK.md/FEATURES.md/ARCHITECTURE.md/PITFALLS.md already source-cite (Outlook
  tentative-booking precedent, WCAG 1.4.1, CSS-Tricks `repeating-linear-gradient` feasibility) — not
  re-fetched this session per the task's explicit "don't re-derive" instruction.

### Tertiary (LOW confidence)
- This research's illustrative `PROPOSAL_PALETTE` hex values (Assumption A1) — explicitly flagged as
  needing a colorblind-vetting pass before being finalized, not presented as a verified palette.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — fully specified by milestone-level research (HIGH confidence, direct package
  reads), re-confirmed live this session against the post-Phase-8 codebase state.
- Architecture: HIGH — every integration point (line numbers, current markup, test pattern) read live
  this session, not assumed from the (slightly pre-Phase-8) milestone docs.
- Pitfalls: HIGH — milestone-level pitfalls re-verified as still-applicable (the `[QUEUED]` bug is
  confirmed still present); two new pitfalls (composition with Phase 8's dashed border, legend collision
  grouping direction) identified specifically for this phase, not covered by the milestone docs.

**Research date:** 2026-06-25
**Valid until:** 30 days (stable Django/stdlib stack; no fast-moving dependency in scope) — but
effectively valid until Phase 9 is planned/executed, since this research is phase-scoped, not
milestone-scoped, and the codebase state it documents (post-Phase-8) will not shift further before
Phase 9 begins.
