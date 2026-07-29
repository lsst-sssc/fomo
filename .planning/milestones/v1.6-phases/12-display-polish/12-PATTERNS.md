# Phase 12: Display Polish - Pattern Map

**Mapped:** 2026-06-27
**Files analyzed:** 7
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `solsys_code/templatetags/calendar_display_extras.py` | template-tag library | transform | same file (add peer tag) | exact |
| `src/templates/tom_calendar/partials/calendar.html` | template | transform | same file (targeted edit) | exact |
| `src/fomo/urls.py` | config | request-response | `solsys_code/solsys_code_observatory/urls.py` | role-match |
| `solsys_code/calendar_urls.py` | config | request-response | `solsys_code/solsys_code_observatory/urls.py` | exact |
| `solsys_code/views.py` | view/service | request-response | same file (add function) | exact |
| `solsys_code/tests/test_calendar_display_extras.py` | test | — | same file (add test class) | exact |
| `solsys_code/tests/test_calendar_template.py` | test | — | same file (add test method) | exact |

---

## Pattern Assignments

### `solsys_code/templatetags/calendar_display_extras.py` (template-tag, transform)

**Analog:** Same file — `text_color_for_bg` is a peer of the existing `proposal_color` and `status_border_css` tags.

**Imports pattern** (lines 1-19 of `calendar_display_extras.py`):
```python
import hashlib
from collections import defaultdict

from django import template

register = template.Library()
```
No new imports are needed for `text_color_for_bg` — it uses only Python stdlib int/float math.

**Core simple_tag pattern** (lines 49-71 — `proposal_color`):
```python
@register.simple_tag
def proposal_color(proposal: str) -> str:
    """Return a deterministic hex color for a proposal code (DISPLAY-04).

    Args:
        proposal: Raw proposal string from CalendarEvent.proposal (may be
            blank, mixed-case, or have surrounding whitespace).

    Returns:
        A hex color string from PROPOSAL_PALETTE, or NEUTRAL_SLOT_COLOR for
        blank/missing proposals (D-05).
    """
    normalized = (proposal or '').strip().upper()
    if not normalized:
        return NEUTRAL_SLOT_COLOR
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return PROPOSAL_PALETTE[int(digest, 16) % len(PROPOSAL_PALETTE)]
```

**New tag to add — exact implementation** (DISPLAY-08, after line 98 / after `status_border_css`):
```python
def _relative_luminance(hex_color: str) -> float:
    """Return relative luminance (0.0-1.0) for a #rrggbb hex color per WCAG 2.1."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def linearize(c: int) -> float:
        L = c / 255
        return L / 12.92 if L <= 0.04045 else ((L + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


@register.simple_tag
def text_color_for_bg(hex_color: str) -> str:
    """Return '#fff' or '#000' — whichever achieves WCAG AA 4.5:1 contrast against hex_color (DISPLAY-08).

    Uses the WCAG 2.1 relative luminance formula. White text achieves 4.5:1 against
    any background with luminance <= 0.183; all PROPOSAL_PALETTE and NEUTRAL_SLOT_COLOR
    entries are dark, so '#fff' is returned for all current palette members.

    Args:
        hex_color: A '#rrggbb' hex color string (e.g. '#005f9e').

    Returns:
        '#fff' if white text achieves >= 4.5:1 contrast; '#000' otherwise.
    """
    lum = _relative_luminance(hex_color)
    white_contrast = 1.05 / (lum + 0.05)
    return '#fff' if white_contrast >= 4.5 else '#000'
```

**Convention notes:**
- Leading underscore on `_relative_luminance` marks it private (no `__all__` used — project convention)
- Google-style docstring with `Args:` / `Returns:` required (D103 enforced by ruff)
- Single quotes, 120-col line length

---

### `solsys_code/calendar_urls.py` (new config file, request-response)

**Analog:** `solsys_code/solsys_code_observatory/urls.py` (lines 1-12)

**Full file pattern to copy**:
```python
from django.urls import path

from solsys_code.solsys_code_observatory.views import CreateObservatory, ObservatoryDetailView, ObservatoryList

app_name = 'solsys_code.solsys_code_observatory'

urlpatterns = [
    path('create/', CreateObservatory.as_view(), name='create'),
    path('<int:pk>/', ObservatoryDetailView.as_view(), name='detail'),
    path('', ObservatoryList.as_view(), name='list'),
]
```

**New file contents** (DISPLAY-09 — shadow single calendar route):
```python
from django.urls import path

from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
]
```

**Key convention:** `app_name` must match the namespace used in the `include()` call in `urls.py` (Django raises `ImproperlyConfigured` if mismatched). Run `./manage.py check` after adding this file and the `urls.py` entry.

---

### `src/fomo/urls.py` (config, request-response)

**Analog:** Same file (lines 17-26) — existing `include()` entries are the pattern:
```python
from django.urls import include, path

from solsys_code.views import Ephemeris, MakeEphemerisView

urlpatterns = [
    path('observatory/', include('solsys_code.solsys_code_observatory.urls', namespace='solsys_code_observatory')),
    path('ephem/<int:pk>/', Ephemeris.as_view(), name='ephem'),
    path('targets/<int:pk>/makeephem/', MakeEphemerisView.as_view(), name='makeephem'),
    path('', include('tom_common.urls')),
]
```

**Change:** Insert the FOMO calendar path BEFORE `path('', include('tom_common.urls'))`:
```python
    path('calendar/', include('solsys_code.calendar_urls', namespace='calendar')),  # DISPLAY-09 — before tom_common
    path('', include('tom_common.urls')),
```

**Critical ordering:** The new path must appear before `path('', include('tom_common.urls'))` — tom_common re-exports `tom_calendar.urls` under `calendar/`; Django's first-match resolution means any path declared after tom_common is unreachable for `calendar/` requests.

---

### `solsys_code/views.py` (view function, request-response)

**Analog:** Same file — existing imports (lines 21-25) show the project's import style:
```python
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.generic import FormView, View
```

**New imports to add** (DISPLAY-09):
```python
from django.db.models import Count, Q
from tom_calendar.models import CalendarEvent
```

**New function to add** — `fomo_render_calendar` (DISPLAY-09). The upstream `render_calendar` in the venv at `tom_calendar/views.py` is the function being shadowed. The wrapper must:
1. Replicate the full `render_calendar` function body (including date/week logic and `render()` call)
2. Augment ONLY the queryset line with `.prefetch_related('telescope_label_meta').annotate(active_todo_count=Count('todos', filter=Q(todos__is_completed=False)))`

**Queryset pattern** (from CONTEXT.md D-06, verified against tom_calendar source):
```python
events = CalendarEvent.objects.filter(
    start_time__date__lte=weeks[-1][-1],
    end_time__date__gte=weeks[0][0],
).prefetch_related(
    'telescope_label_meta',
).annotate(
    active_todo_count=Count('todos', filter=Q(todos__is_completed=False))
)
events = list(events)
```

**Docstring pattern** (Google-style, matching project convention):
```python
def fomo_render_calendar(request, month=None):
    """Shadow tom_calendar.views.render_calendar to inject prefetch + Count annotation.

    Eliminates two N+1 query patterns per CalendarEvent:
    - telescope_label_meta OneToOneField reverse accessor (DISPLAY-09)
    - active_todos.count filtered aggregate (DISPLAY-09)

    Args:
        request: Django HttpRequest.
        month: Optional month string (same signature as upstream render_calendar).

    Returns:
        HttpResponse rendering calendar.html with prefetched event data.
    """
```

---

### `src/templates/tom_calendar/partials/calendar.html` (template, transform)

**Analog:** Same file. The CONTEXT.md and RESEARCH.md have verified the exact line numbers.

**DISPLAY-08 CSS change** — remove from the `<style>` block:
- Line 85: `color: #fff !important;` from `.cal-event-all-day { ... }` rule
- Line 94: `color: #fff !important;` from `.cal-event-all-day a { ... }` rule

Both must be removed together — leaving either one causes `!important` to override the inline `color:` style on child `<a>` elements.

**DISPLAY-08 template tag addition** — in the all-day event loop, after existing tags, before the `{% if event.telescope_label_meta.is_verified == False %}` branch:
```html+django
{% proposal_color event.proposal as bg_color %}
{% status_border_css event.title as status_border %}
{% text_color_for_bg bg_color as text_color %}
{% if event.telescope_label_meta.is_verified == False %}
<div class="cal-event-all-day" style="background-color: {{ bg_color }}; color: {{ text_color }}; {{ status_border }} border: 2px dashed rgba(0, 0, 0, 0.65);">
{% else %}
<div class="cal-event-all-day" style="background-color: {{ bg_color }}; color: {{ text_color }}; {{ status_border }}">
{% endif %}
```

**DISPLAY-09 template rename** — at both occurrences (lines ~193-195 all-day, lines ~219-221 timed):
```
event.active_todos.count  →  event.active_todo_count
```

---

### `solsys_code/tests/test_calendar_display_extras.py` (test, —)

**Analog:** Same file — existing `ProposalColorTest` and `StatusBorderCssTest` classes (lines 25-103) show the test pattern:

**Import pattern** (lines 1-23):
```python
from types import SimpleNamespace

from django.test import TestCase

from solsys_code.templatetags.calendar_display_extras import (
    CLASSICAL_SCHEDULE_LABEL,
    NEUTRAL_SLOT_COLOR,
    PROPOSAL_PALETTE,
    proposal_color,
    status_border_css,
    visible_proposals,
)
```

**New imports to add** (DISPLAY-08):
```python
from solsys_code.templatetags.calendar_display_extras import (
    ...
    text_color_for_bg,   # add to existing import block
)
```

**New test class pattern** — copy structure from `StatusBorderCssTest`:
```python
class TextColorForBgTest(TestCase):
    def test_all_palette_colors_return_white(self):
        """DISPLAY-08: all 8 PROPOSAL_PALETTE entries achieve WCAG AA with white text."""
        for hex_color in PROPOSAL_PALETTE:
            with self.subTest(hex_color=hex_color):
                self.assertEqual(text_color_for_bg(hex_color), '#fff')

    def test_neutral_slot_returns_white(self):
        """DISPLAY-08: NEUTRAL_SLOT_COLOR (#5a6268) achieves WCAG AA with white text."""
        self.assertEqual(text_color_for_bg(NEUTRAL_SLOT_COLOR), '#fff')

    def test_bright_background_returns_black(self):
        """DISPLAY-08: formula correctness — pure white background yields black text."""
        self.assertEqual(text_color_for_bg('#ffffff'), '#000')
```

**Convention notes:**
- `subTest` (not pytest `parametrize`) — project uses `django.test.TestCase`
- Short inline comments after `self.assert*` calls explain the DISPLAY-XX reference
- No docstrings required on test methods in this project (D101/D102 suppressed for tests)

---

### `solsys_code/tests/test_calendar_template.py` (test, —)

**Analog:** Same file — `CalendarTemplateTest` class with `setUp` and `_get_calendar` helper (lines 37-139 read above).

**Existing helper pattern** (lines 134-135):
```python
def _get_calendar(self):
    return self.client.get(reverse('calendar:calendar'), {'year': self.year, 'month': self.month})
```

**New test method to add** (DISPLAY-09 N+1 assertion):
```python
def test_display09_calendar_query_count_does_not_grow_with_events(self):
    """DISPLAY-09: query count is bounded regardless of number of CalendarEvents."""
    # Determine baseline with the events already created in setUp.
    # Use assertNumQueries with an upper bound rather than exact count — robust
    # to minor framework overhead changes. A bound of 10 is generous for
    # a calendar page that should need: session + auth + events + prefetch + annotate.
    with self.assertNumQueries(less_than=10):
        self._get_calendar()
```

Note: `assertNumQueries` does not accept `less_than`. Use the two-event vs one-event comparison approach instead to prove no O(N) growth:
```python
def test_display09_query_count_bounded(self):
    """DISPLAY-09: same query count with 1 event vs N events (no O(N) growth)."""
    # Count queries with current setUp fixtures (multiple events).
    from django.test.utils import CaptureQueriesContext
    from django.db import connection
    with CaptureQueriesContext(connection) as multi_ctx:
        self._get_calendar()
    multi_count = len(multi_ctx)

    # Add another batch of events and recount — count must not increase.
    CalendarEvent.objects.create(
        title='Extra event',
        start_time=datetime(2026, 6, 5, 22, 0, tzinfo=dt_timezone.utc),
        end_time=datetime(2026, 6, 6, 6, 0, tzinfo=dt_timezone.utc),
    )
    with CaptureQueriesContext(connection) as extra_ctx:
        self._get_calendar()

    self.assertEqual(len(extra_ctx), multi_count)
```

**Imports to add** (only if `CaptureQueriesContext` approach is used):
```python
from django.db import connection
from django.test.utils import CaptureQueriesContext
```

---

## Shared Patterns

### `@register.simple_tag` return pattern
**Source:** `solsys_code/templatetags/calendar_display_extras.py` lines 49-71 (`proposal_color`) and 74-98 (`status_border_css`)
**Apply to:** `text_color_for_bg` in the same file
- Decorator: `@register.simple_tag`
- Type-annotated signature: `def tag_name(arg: type) -> str:`
- Google-style docstring with `Args:` and `Returns:`
- Called in template as `{% tag_name value as variable_name %}`

### Django ORM annotation pattern
**Source:** CONTEXT.md D-07 / RESEARCH.md Pattern 2 (confirmed against tom_calendar source)
**Apply to:** `fomo_render_calendar` in `solsys_code/views.py`
```python
from django.db.models import Count, Q
events = queryset.prefetch_related('telescope_label_meta').annotate(
    active_todo_count=Count('todos', filter=Q(todos__is_completed=False))
)
events = list(events)   # execute query before downstream template iteration
```

### URL include with app_name pattern
**Source:** `solsys_code/solsys_code_observatory/urls.py` lines 1-12
**Apply to:** `solsys_code/calendar_urls.py`
- `app_name = 'calendar'` at module level (required when `namespace='calendar'` used in `include()`)
- Function-based views referenced by dotted import, not `as_view()`

### Django TestCase with subTest
**Source:** Existing test classes in `solsys_code/tests/test_calendar_display_extras.py` and `test_calendar_template.py`
**Apply to:** New `TextColorForBgTest` class
- `from django.test import TestCase`
- `self.subTest(param=value)` for parametrized assertions within one test method
- Short inline comment above each `assert*` call citing the requirement ID (e.g., `# DISPLAY-08:`)

---

## No Analog Found

All files have close analogs in the existing codebase. No files require falling back to RESEARCH.md patterns alone.

---

## Metadata

**Analog search scope:** `solsys_code/`, `solsys_code/templatetags/`, `solsys_code/tests/`, `src/fomo/`, `solsys_code/solsys_code_observatory/`
**Files scanned:** 6 source files read directly
**Pattern extraction date:** 2026-06-27
