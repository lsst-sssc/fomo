# Phase 9: Proposal Color & Status Visual Treatment - Pattern Map

**Mapped:** 2026-06-25
**Files analyzed:** 5 (3 new, 2 modified)
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `solsys_code/templatetags/__init__.py` | config (package init) | n/a | `solsys_code/solsys_code_observatory/__init__.py` | exact (empty init) |
| `solsys_code/templatetags/calendar_display_extras.py` | utility (template tag library) | transform | installed `tom_calendar/templatetags/calendar_tags.py` | exact (same role: Django template tag module with `register = template.Library()` and `@register.simple_tag`) |
| `src/templates/tom_calendar/partials/calendar.html` | template | request-response | itself (current state, read live) | self-modification |
| `solsys_code/tests/test_calendar_display_extras.py` | test (unit) | transform | `solsys_code/tests/test_calendar_template.py` | role-match |
| `solsys_code/tests/test_calendar_template.py` | test (integration) | request-response | itself (extend existing) | self-modification |

## Pattern Assignments

### `solsys_code/templatetags/__init__.py` (package init)

**Analog:** `solsys_code/solsys_code_observatory/__init__.py` (empty file, just marks the directory as a Python package)

**Core pattern:** Empty file. No imports, no content.

---

### `solsys_code/templatetags/calendar_display_extras.py` (template tag library, transform)

**Analog:** installed `tom_calendar/templatetags/calendar_tags.py`

**Imports pattern** (calendar_tags.py lines 1-7):
```python
from datetime import timedelta

from django import template

from tom_calendar.utils import target_list_color as _target_list_color

register = template.Library()
```

**Core tag pattern** (calendar_tags.py lines 10-12 and 15-17):
```python
@register.simple_tag
def target_list_color(target_list):
    return _target_list_color(target_list)


@register.filter
def offset_time(dt, utc_offset):
    return dt + timedelta(hours=int(utc_offset))
```

**Pattern for `proposal_color` tag** — the new tag follows the same `@register.simple_tag` shape; normalize-then-hash-then-palette-index (from RESEARCH.md Pattern 1):
```python
import hashlib
from collections import defaultdict

from django import template

register = template.Library()

# Project-local literal hex palette — independent of tom_calendar.utils.BOOTSTRAP_COLORS,
# which uses Bootstrap4 CSS variable names that will be renamed in a future Bootstrap5 upgrade.
# Curated for mutual distinguishability and colorblind-safety (vet against a CVD simulator
# before finalizing exact values — see RESEARCH.md Assumption A1).
PROPOSAL_PALETTE = [
    '#4363d8', '#f58231', '#3cb44b', '#911eb4',
    '#42d4f4', '#f032e6', '#9A6324', '#e6194B',
]
NEUTRAL_SLOT_COLOR = '#6c757d'   # Bootstrap "secondary" grey — D-05's dedicated neutral slot
CLASSICAL_SCHEDULE_LABEL = 'Classical schedule'   # D-06


@register.simple_tag
def proposal_color(proposal: str) -> str:
    """Return a deterministic hex color for a proposal code.

    Args:
        proposal: Raw proposal string from CalendarEvent.proposal (may be blank,
            mixed-case, or have surrounding whitespace).

    Returns:
        A hex color string from PROPOSAL_PALETTE, or NEUTRAL_SLOT_COLOR for
        blank/missing proposals.
    """
    normalized = (proposal or '').strip().upper()
    if not normalized:
        return NEUTRAL_SLOT_COLOR
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return PROPOSAL_PALETTE[int(digest, 16) % len(PROPOSAL_PALETTE)]
```

**`visible_proposals` tag pattern** (from RESEARCH.md Pattern 3 — groups by resulting color, not by proposal code, per D-04):
```python
@register.simple_tag
def visible_proposals(weeks) -> list[dict]:
    """Compute the set of proposals visible in the currently-rendered month.

    Iterates the weeks/day context already materialized by render_calendar() —
    no new database query (D-02).  Groups by resulting color so hash-colliding
    proposals share one legend entry (D-04).

    Args:
        weeks: The weeks context list passed to calendar.html.

    Returns:
        List of dicts with keys 'color', 'codes' (sorted list), 'label' (joined
        string), sorted by color for stable legend ordering.
    """
    by_color: dict[str, set[str]] = defaultdict(set)
    for week in weeks:
        for day in week:
            for event in list(day.all_day_events) + list(day.events):
                normalized = (event.proposal or '').strip().upper()
                color = proposal_color(event.proposal)
                label = normalized if normalized else CLASSICAL_SCHEDULE_LABEL
                by_color[color].add(label)
    return [
        {'color': color, 'codes': sorted(codes), 'label': ', '.join(sorted(codes))}
        for color, codes in sorted(by_color.items())
    ]
```

**`status_border_css` tag pattern** — maps title-prefix vocabulary to a CSS border/box-shadow string; the exact CSS property is determined by the `/gsd:sketch` session (D-08), but the tag interface and exclusion of `border-style: dashed` (D-09) are fixed:
```python
# Title-prefix vocabulary from sync_lco_observation_calendar.py (confirmed live):
#   '[QUEUED] '  -> queued (banner stage, no scheduled_start)
#   '[EXPIRED]', '[CANCELLED]', '[FAILED]' -> terminal-failure
#   '[UNVERIFIED]' or no prefix -> placed (treat identically for DISPLAY-06; Phase 8's
#       dashed border already handles the verified/fallback distinction — MUST NOT re-encode
#       that distinction here, D-09)
# MUST NOT emit border-style: dashed (reserved for Phase 8's is_verified cue, D-09).

_TERMINAL_PREFIXES = ('[EXPIRED]', '[CANCELLED]', '[FAILED]')

@register.simple_tag
def status_border_css(title: str) -> str:
    """Return a CSS property string encoding the observation status of an event.

    The exact CSS values (border color, thickness) are chosen during the /gsd:sketch
    session per D-08.  This stub shows the interface contract.

    Args:
        title: CalendarEvent.title — may start with a known prefix.

    Returns:
        A CSS fragment suitable for direct inclusion in a style attribute, e.g.
        'border: 3px solid rgba(0,0,0,0.55);'.  Never emits border-style: dashed
        (reserved for Phase 8's is_verified cue per D-09).
    """
    if title.startswith('[QUEUED] '):
        return ''   # fill in from /gsd:sketch: e.g. 'border: 3px solid rgba(0,0,0,0.55);'
    if any(title.startswith(p) for p in _TERMINAL_PREFIXES):
        return ''   # fill in from /gsd:sketch: e.g. 'border: 4px solid rgba(180,0,0,0.7);'
    return ''       # placed (verified or unverified) — minimal or no extra border
```

**Docstring style** — Google-style with `Args:` and `Returns:` sections, matching `solsys_code/ephem_utils.py` conventions.

**Formatting/lint notes:** single quotes, 120-col max, ruff-formatted.

---

### `src/templates/tom_calendar/partials/calendar.html` (template, request-response)

**Analog:** current state of this file (self-modification, lines 101–218 read live)

**Tag load line to update** (current line 101):
```django
{% load tz calendar_tags %}
```
Must become:
```django
{% load tz calendar_tags calendar_display_extras %}
```

**Current `[QUEUED]` override to rewrite** (current lines 158–165, all-day branch — this is the DISPLAY-05 bug):
```django
{% if event.title|slice:":9" == "[QUEUED] " %}
<div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.45); border: 1px solid rgba(0, 0, 0, 0.55);">
{% elif event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day" style="background-color: {{ event.color }}; border: 2px dashed rgba(0, 0, 0, 0.65);"
     title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0).">
{% else %}
<div class="cal-event-all-day" style="background-color: {{ event.color }};">
{% endif %}
```

**Replacement pattern** (status and color computed independently, then composed — D-09 composition rule):
```django
{% proposal_color event.proposal as bg_color %}
{% status_border_css event.title as status_border %}
{% if event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day cal-event--{{ event.id }}"
     data-proposal="{{ event.proposal|default:'' }}"
     style="background-color: {{ bg_color }}; {{ status_border }} border-style: dashed;"
     title="Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0).">
{% else %}
<div class="cal-event-all-day cal-event--{{ event.id }}"
     data-proposal="{{ event.proposal|default:'' }}"
     style="background-color: {{ bg_color }}; {{ status_border }}">
{% endif %}
```
Note: `status_border` is a CSS fragment that MUST NOT contain `border-style: dashed` (D-09); when composed with the Phase 8 `is_verified` branch the dashed style is appended separately, so both signals remain visible simultaneously when both apply (Pitfall 3 prevention).

**Timed branch** (current lines 173–182) analogously replaces `{% else %}` branch with `proposal_color`-keyed `background-color` and `status_border_css`-keyed border, while keeping Phase 8's dashed-border `is_verified == False` check orthogonal.

**Footer row** (current lines 198–206 — swatch pattern to mirror from `target_list_block.html`, per D-01):

Existing swatch markup in `target_list_block.html` (analog to mirror):
```django
{% target_list_color target_list as tl_color %}
<span class="cal-event-bullet" style="color: {{ tl_color }};">▌</span>
```

New proposal legend to add inside the `<div>` at line 199 (alongside the existing target-list loop):
```django
{% visible_proposals weeks as proposal_legend %}
{% for entry in proposal_legend %}
  <span class="cal-legend-swatch"
        data-proposal="{{ entry.codes|join:',' }}"
        style="color: {{ entry.color }}; cursor: pointer;">▌</span>
  <small class="text-secondary mr-2">{{ entry.label }}</small>
{% endfor %}
```

**Click-to-filter script** (new inline `<script>` at end of partial, before closing `</div>` — event delegation surviving htmx swaps, Pitfall 5 prevention):
```django
<script>
(function () {
  var CAL = document.getElementById('calendar-partial');
  if (!CAL) return;

  function applyFilter(activeProposal) {
    CAL.querySelectorAll('.cal-event[data-proposal]').forEach(function (el) {
      if (!activeProposal || el.dataset.proposal === activeProposal) {
        el.classList.remove('cal-filter-dim');
        el.classList.add(activeProposal ? 'cal-filter-active' : '');
      } else {
        el.classList.add('cal-filter-dim');
        el.classList.remove('cal-filter-active');
      }
    });
  }

  var active = null;
  CAL.addEventListener('click', function (e) {
    var swatch = e.target.closest('.cal-legend-swatch');
    if (!swatch) return;
    e.stopPropagation();
    var proposal = swatch.dataset.proposal;
    if (active === proposal) {
      active = null;
      applyFilter(null);
    } else {
      active = proposal;
      applyFilter(proposal);
    }
  });

  // Re-bind after htmx swaps replace #calendar-partial (Pitfall 5).
  document.body.addEventListener('htmx:afterSwap', function (e) {
    if (e.detail.target && e.detail.target.id === 'calendar-partial') {
      CAL = e.detail.target;
      active = null;
    }
  });
}());
</script>
```

**CSS for dim/active** (add to the `<style>` block at top of the partial, ~lines 1-99):
```css
.cal-filter-dim { opacity: 0.2; }
.cal-filter-active { outline: 2px solid rgba(0,0,0,0.35); }
```

---

### `solsys_code/tests/test_calendar_display_extras.py` (unit test, transform)

**Analog:** `solsys_code/tests/test_calendar_template.py` (established pattern)

**Imports pattern** (from test_calendar_template.py lines 1–17 — mirror this shape):
```python
from django.test import TestCase
```
For pure tag-function unit tests no HTTP client is needed; the tag functions are pure Python and testable directly without a database or request.

**Core unit-test pattern** (mirror the `django.test.TestCase` class structure, `setUp`, assertion style):
```python
from django.test import TestCase

from solsys_code.templatetags.calendar_display_extras import (
    NEUTRAL_SLOT_COLOR,
    PROPOSAL_PALETTE,
    proposal_color,
    visible_proposals,
    status_border_css,
)


class ProposalColorTest(TestCase):
    def test_same_proposal_same_color(self):
        self.assertEqual(proposal_color('LTP2025A-004'), proposal_color('LTP2025A-004'))

    def test_normalized_case_insensitive(self):
        # Normalization (.strip().upper()) must make these identical.
        self.assertEqual(proposal_color('LTP2025A-004'), proposal_color('ltp2025a-004'))
        self.assertEqual(proposal_color('LTP2025A-004'), proposal_color('LTP2025A-004 '))

    def test_empty_proposal_returns_neutral_slot(self):
        self.assertEqual(proposal_color(''), NEUTRAL_SLOT_COLOR)
        self.assertEqual(proposal_color('   '), NEUTRAL_SLOT_COLOR)
        self.assertEqual(proposal_color(None), NEUTRAL_SLOT_COLOR)

    def test_nonempty_proposal_returns_palette_color(self):
        color = proposal_color('LTP2025A-004')
        self.assertIn(color, PROPOSAL_PALETTE)
```

**Test structure notes:**
- Class per tag function (`ProposalColorTest`, `StatusBorderCssTest`, `VisibleProposalsTest`)
- No `setUp()` needed for the pure-function tag tests (`proposal_color`, `status_border_css`)
- `visible_proposals` tests need a minimal `weeks`-shaped list of day-stub objects (or simple `SimpleNamespace` fakes) to avoid DB dependency; keep them lightweight
- Per CLAUDE.md, test files are exempt from `D101`/`D102` docstring requirements

---

### `solsys_code/tests/test_calendar_template.py` (integration test, extend existing)

**Analog:** itself (current file, read live — 98 lines, Phase 8 established pattern)

**Existing fixture pattern to extend** (lines 22–73 — `CalendarEvent.objects.create(...)` + `CalendarEventTelescopeLabel.objects.create(...)` in `setUp`):
```python
self.all_day_fallback = CalendarEvent.objects.create(
    title='All-day fallback',
    start_time=datetime(2026, 6, 10, 22, 0, tzinfo=dt_timezone.utc),
    end_time=datetime(2026, 6, 11, 6, 0, tzinfo=dt_timezone.utc),
)
CalendarEventTelescopeLabel.objects.create(event=self.all_day_fallback, is_verified=False)
```

Phase 9 extends `setUp()` to add events with `proposal=` values:
```python
self.queued_event = CalendarEvent.objects.create(
    title='[QUEUED] LTP2025A run',
    proposal='LTP2025A-004',
    start_time=datetime(2026, 6, 20, 22, 0, tzinfo=dt_timezone.utc),
    end_time=datetime(2026, 6, 21, 6, 0, tzinfo=dt_timezone.utc),
)
self.terminal_event = CalendarEvent.objects.create(
    title='[FAILED] LTP2025B run',
    proposal='LTP2025B-012',
    start_time=datetime(2026, 6, 22, 22, 0, tzinfo=dt_timezone.utc),
    end_time=datetime(2026, 6, 23, 6, 0, tzinfo=dt_timezone.utc),
)
self.no_proposal_event = CalendarEvent.objects.create(
    title='Classical block',
    proposal='',
    start_time=datetime(2026, 6, 24, 22, 0, tzinfo=dt_timezone.utc),
    end_time=datetime(2026, 6, 25, 6, 0, tzinfo=dt_timezone.utc),
)
```

**Existing assertion pattern to extend** (lines 83–97 — `assertContains` + `content.count()` for precise occurrence counting):
```python
def test_fallback_events_get_dashed_border_and_tooltip(self):
    response = self._get_calendar()
    self.assertContains(response, DASHED_BORDER_MARKER)
    self.assertContains(response, TOOLTIP_SUBSTRING)

def test_dashed_border_count_matches_fallback_event_count_only(self):
    response = self._get_calendar()
    content = response.content.decode()
    self.assertEqual(content.count(DASHED_BORDER_MARKER), self.num_fallback_day_cell_occurrences)
```

Phase 9 adds assertions following this exact shape:
- `assertNotIn` flat-grey literal (`rgba(0, 0, 0, 0.45)`) in rendered content — DISPLAY-05 regression
- `assertIn` proposal's hashed color in the `[QUEUED]` event's rendered `style` — DISPLAY-05
- `assertIn` the DISPLAY-06 status border CSS marker for queued events — DISPLAY-06
- `assertIn` the DISPLAY-06 status border CSS marker for terminal-failure events — DISPLAY-06
- `assertNotEqual` the two markers (queued vs terminal-failure are visually distinct) — DISPLAY-06
- `assertIn` `DASHED_BORDER_MARKER` still present (Phase 8 regression guard) — Pitfall 3
- `assertIn` neutral-slot color for no-proposal event — DISPLAY-04
- `assertIn` legend swatch markup (`▌`) in footer row — DISPLAY-07

---

## Shared Patterns

### Template Tag Registration
**Source:** `tom_calendar/templatetags/calendar_tags.py` lines 1-7
**Apply to:** `solsys_code/templatetags/calendar_display_extras.py`
```python
from django import template

register = template.Library()
```
Every tag module starts with this exact boilerplate. No `__all__`, no app_name prefix needed — Django discovers tag libraries by module name, which must be unique across all `INSTALLED_APPS`.

### Single Quotes and 120-col Line Length
**Source:** `pyproject.toml` ruff config (project-wide)
**Apply to:** All new Python files
All string literals use single quotes. Lines max 120 characters. Run `ruff check . --fix && ruff format .` before committing.

### Google-Style Docstrings
**Source:** `solsys_code/ephem_utils.py` (project convention per CLAUDE.md)
**Apply to:** `calendar_display_extras.py` public functions
```python
def proposal_color(proposal: str) -> str:
    """One-line summary.

    Args:
        proposal: Description.

    Returns:
        Description.
    """
```

### Django Test Class Structure
**Source:** `solsys_code/tests/test_calendar_template.py` lines 22-97
**Apply to:** `test_calendar_display_extras.py`, extended `test_calendar_template.py`
- `class Foo(TestCase):` with `setUp(self) -> None:`
- `Client().get(reverse('calendar:calendar'), {'year': ..., 'month': ...})` for integration tests
- `assertContains(response, marker)` and `response.content.decode().count(marker)` for HTML assertion
- File-level constants for repeated marker strings (e.g. `DASHED_BORDER_MARKER = '2px dashed rgba(0, 0, 0, 0.65)'`)

### CSS Composition (not Replacement) of Border Signals
**Source:** `calendar.html` lines 160-165 (the bug) + RESEARCH.md Pitfall 3
**Apply to:** `calendar.html` rewrite and `status_border_css` tag design
Phase 8's `border-style: dashed` and Phase 9's status border must be two independent CSS fragments, concatenated into one `style` attribute — not chained as mutually-exclusive `{% elif %}` branches. The `status_border_css` tag MUST NOT emit `border-style: dashed` (D-09).

---

## No Analog Found

All files have analogs. No gaps.

| File | Role | Data Flow | Notes |
|------|------|-----------|-------|
| — | — | — | All 5 files have a direct analog or are self-modifications of existing files |

---

## Metadata

**Analog search scope:** `solsys_code/`, `src/templates/`, installed `tom_calendar` package at `/home/tlister/venvs/fomo312_venv/lib/python3.12/site-packages/tom_calendar/`
**Files scanned:** `calendar_tags.py`, `calendar.html` (lines 95-218), `test_calendar_template.py` (full, 98 lines)
**Pattern extraction date:** 2026-06-25
