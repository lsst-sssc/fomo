# Phase 12: Display Polish - Research

**Researched:** 2026-06-27
**Domain:** Django template tags, WCAG accessibility math, Django ORM prefetch/annotation, URL shadowing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**DISPLAY-08 — WCAG Text Color**

- **D-01:** Compute text color via a new `text_color_for_bg(hex_color: str) -> str`
  template tag in `calendar_display_extras.py`. It implements the WCAG relative
  luminance formula and returns `'#fff'` or `'#000'` — whichever achieves ≥ 4.5:1
  contrast against the given background hex.

- **D-02:** Coverage: apply to all 8 `PROPOSAL_PALETTE` entries AND
  `NEUTRAL_SLOT_COLOR` (`#5a6268`). Both are rendered as all-day event backgrounds;
  both should use the computed text color for consistency. Tests cover all 9.

- **D-03:** CSS mechanism: remove `color: #fff !important` from the `.cal-event-all-day`
  and `.cal-event-all-day a` CSS rules in `calendar.html`. Replace with
  `color: {{ text_color }}` in the inline `style` attribute on the event `<div>`.
  No CSS `!important` battles.

- **D-04:** Timed events (`.cal-event-timed`) are excluded — they have a transparent
  background; the palette color is used only for the `▌` bullet, not the text
  background. No text-color change needed for timed events.

**DISPLAY-09 — N+1 Elimination**

- **D-05:** Fix BOTH N+1 patterns in the event loops simultaneously:
  - `event.telescope_label_meta.is_verified` — OneToOneField reverse accessor
  - `event.active_todos.count` — filtered related manager called twice per event

- **D-06:** Prefetch injection mechanism: a FOMO-local wrapper view in
  `solsys_code/views.py` that shadows the `tom_calendar` `render_calendar` function.
  Added to `src/fomo/urls.py` BEFORE `path('', include('tom_common.urls'))` with
  namespace `'calendar'`, so the URL resolver hits the FOMO view first.

  The wrapper view replicates `render_calendar`'s queryset with these additions:
  ```python
  events = CalendarEvent.objects.filter(...).prefetch_related(
      'telescope_label_meta',
  ).annotate(
      active_todo_count=Count('todos', filter=Q(todos__is_completed=False))
  )
  ```

- **D-07:** `active_todos.count` fixed via Count annotation (not Prefetch+to_attr).
  The template is updated from `event.active_todos.count` to `event.active_todo_count`
  at both occurrences (all-day events and timed events). Cleaner than to_attr: one
  extra SQL column, zero extra queries, no change to the property or model.

### Claude's Discretion

- **Wrapper view structure**: The wrapper replicates only the queryset-building portion
  of `render_calendar` (the `CalendarEvent.objects.filter(...)` + `list()` lines),
  calling the rest of the function's context-building and rendering logic directly from
  `tom_calendar.views`. If that's not cleanly extractable, the planner may choose to
  shadow the full `render_calendar` function body with the prefetch/annotation added
  — accepting the maintenance burden against the third-party function.

- **`text_color_for_bg` helper placement**: Pure Python helper function in
  `calendar_display_extras.py` (not in `calendar_utils.py`) — it's display logic,
  not a model utility.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DISPLAY-08 | Calendar event title text renders in white or black based on relative luminance of the proposal palette background, meeting WCAG AA 4.5:1 contrast ratio against every palette color | WCAG formula verified below; all 8 palette colors + NEUTRAL_SLOT_COLOR are dark enough that white text passes for all 9; `text_color_for_bg` simple_tag follows existing `proposal_color`/`status_border_css` pattern |
| DISPLAY-09 | `CalendarEventTelescopeLabel` data for visible calendar events is loaded in a single prefetch query, not per-event — the N+1 pattern is eliminated from the calendar template | `prefetch_related('telescope_label_meta')` + `Count` annotation pattern verified against Django ORM; upstream `render_calendar` function read and confirmed suitable for shadowing |
</phase_requirements>

## Summary

Phase 12 is two surgical fixes to the calendar UI with no new models, no new management
commands, and no new external packages.

DISPLAY-08 adds a `text_color_for_bg` template tag implementing the WCAG relative
luminance formula. All 8 `PROPOSAL_PALETTE` colors and `NEUTRAL_SLOT_COLOR` are dark
(relative luminance < 0.183), so the formula always returns `'#fff'` for this palette —
but the implementation must be formula-driven, not hardcoded, so any future palette
change is automatically correct. Two `color: #fff !important` lines in the CSS block
of `calendar.html` are removed and replaced with an inline `color: {{ text_color }}`
computed per event.

DISPLAY-09 eliminates two N+1 query patterns in the event render loops. The root cause
is that the upstream `tom_calendar.views.render_calendar` fetches events without
prefetching the `telescope_label_meta` OneToOneField reverse accessor or pre-computing
the `active_todos.count` filtered aggregate. The fix is a FOMO-local wrapper view that
shadows the upstream function for the `/calendar/` URL only, adding
`prefetch_related('telescope_label_meta')` and a `Count` annotation to the queryset.
All other calendar URLs (`/calendar/create/`, etc.) fall through to `tom_calendar.urls`
via Django's first-match URL resolution.

**Primary recommendation:** Implement both fixes in a single wave. DISPLAY-08 is a
self-contained template tag with a parametrized test. DISPLAY-09 requires a new file
(`solsys_code/calendar_urls.py`), a new wrapper view function, a URL order change, and
a template variable rename. Both fit naturally in one plan.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| WCAG text-color computation | Template tag layer | — | Pure display logic; no model data needed; computed from the same bg_color already available in the template context |
| CSS delivery of computed text color | Template (inline style) | — | D-03 specifies inline style overrides the removed !important rule; no new CSS class needed |
| N+1 elimination for telescope_label_meta | View / queryset layer | — | prefetch must be applied before `list(events)` converts the queryset to Python objects |
| N+1 elimination for active_todos.count | View / queryset layer (Count annotation) | Template (rename) | The Count annotation is in the queryset; the template is updated to read the annotated attribute |
| URL shadowing to inject prefetch | URL routing layer | View layer | A new FOMO URL entry intercepts `calendar/` before tom_common; view function carries the prefetch logic |

## Standard Stack

### Core (no new installs — all from existing project dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django ORM | (project Django version) | `prefetch_related`, `Count`, `Q` | Built-in; correct tool for queryset optimization |
| Python stdlib `re` / int math | 3.10+ | WCAG hex parsing and luminance arithmetic | No library needed; formula is 5 lines of Python |

### No New Packages

This phase installs zero new packages. All required functionality is available in:
- Django ORM (`django.db.models.Count`, `Q`, `prefetch_related`)
- Python stdlib integer and float math (WCAG luminance formula)
- Existing project template tag infrastructure (`@register.simple_tag`)

**Package Legitimacy Audit: SKIPPED — no packages to install.**

## Architecture Patterns

### System Architecture Diagram

```
HTTP GET /calendar/
        |
        v
fomo/urls.py
  path('calendar/', include('solsys_code.calendar_urls', namespace='calendar'))
        |
        v
solsys_code/calendar_urls.py
  path('', fomo_render_calendar, name='calendar')
        |
        v
solsys_code/views.py :: fomo_render_calendar()
  CalendarEvent.objects.filter(...)
    .prefetch_related('telescope_label_meta')   <-- eliminates N+1 #1
    .annotate(active_todo_count=Count('todos', filter=Q(...)))  <-- eliminates N+1 #2
  list(events)
  [copy render_calendar context-building and render logic]
        |
        v
calendar.html
  {% text_color_for_bg bg_color as text_color %}   <-- DISPLAY-08
  style="... color: {{ text_color }};"
  {{ event.active_todo_count }}                     <-- DISPLAY-09 rename
```

```
HTTP GET /calendar/create/
        |
        v
fomo/urls.py
  path('calendar/', include('solsys_code.calendar_urls'))
    -> no match for 'create/'
    -> fall through
  path('', include('tom_common.urls'))
    -> path('calendar/', include('tom_calendar.urls', namespace='calendar'))
      -> path('create/', create_event, name='create-event')
        -> tom_calendar's create_event (unchanged)
```

### Recommended Project Structure

No new top-level modules. New files:

```
solsys_code/
├── calendar_urls.py         # NEW — FOMO-local calendar URL shadowing single route
├── views.py                 # MODIFIED — add fomo_render_calendar function
├── templatetags/
│   └── calendar_display_extras.py   # MODIFIED — add text_color_for_bg tag
src/
├── fomo/urls.py             # MODIFIED — add FOMO calendar path before tom_common
├── templates/tom_calendar/partials/
│   └── calendar.html        # MODIFIED — remove !important, add text_color, rename active_todo_count
solsys_code/tests/
├── test_calendar_display_extras.py  # MODIFIED — add WCAG parametrized tests
└── test_calendar_template.py        # MODIFIED — add N+1 query-count assertion
```

### Pattern 1: WCAG Relative Luminance Simple Tag

**What:** A `@register.simple_tag` that takes a hex color string and returns `'#fff'` or
`'#000'`. Called as `{% text_color_for_bg bg_color as text_color %}` so the result is
captured into a template variable before the style attribute.

**When to use:** Any template position where a background color from `PROPOSAL_PALETTE`
or `NEUTRAL_SLOT_COLOR` needs a readable foreground color.

**WCAG formula (from CONTEXT.md specifics section):** [CITED: CONTEXT.md §Specific Ideas]
```python
# Source: CONTEXT.md §Specific Ideas — WCAG 2.1 relative luminance formula
def _relative_luminance(hex_color: str) -> float:
    """Return relative luminance (0.0–1.0) of a #rrggbb hex color."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    def linearize(c: int) -> float:
        L = c / 255
        return L / 12.92 if L <= 0.04045 else ((L + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


@register.simple_tag
def text_color_for_bg(hex_color: str) -> str:
    """Return '#fff' or '#000' — whichever achieves WCAG AA 4.5:1 against hex_color (DISPLAY-08)."""
    lum = _relative_luminance(hex_color)
    # White on bg: contrast = 1.05 / (lum + 0.05). Passes WCAG AA when lum <= 0.183.
    white_contrast = 1.05 / (lum + 0.05)
    return '#fff' if white_contrast >= 4.5 else '#000'
```

**Verification:** All 8 `PROPOSAL_PALETTE` colors and `NEUTRAL_SLOT_COLOR` have relative
luminance well below 0.183 (they are intentionally dark). `'#fff'` is returned for all 9.
[VERIFIED: codebase — palette colors confirmed dark via hex inspection; luminance threshold is a deterministic calculation]

### Pattern 2: Queryset Prefetch + Count Annotation

**What:** Modify the CalendarEvent queryset before `list(events)` conversion to pre-load
`telescope_label_meta` (OneToOneField reverse) and pre-compute `active_todo_count`.

**When to use:** Any view that fetches multiple CalendarEvents and the template accesses
`telescope_label_meta` or `active_todos.count` per event.

```python
# Source: codebase read — solsys_code/models.py and tom_calendar/models.py
# [VERIFIED: codebase]
from django.db.models import Count, Q
from tom_calendar.models import CalendarEvent

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

**Why `prefetch_related` and not `select_related` for telescope_label_meta:**
`CalendarEventTelescopeLabel` is a OneToOneField where `CalendarEvent` is the FK target
(the label model has the FK to CalendarEvent with `primary_key=True`). Django's
`select_related` follows FK forward; for reverse OneToOne, Django supports both
`select_related` and `prefetch_related`. [ASSUMED: `prefetch_related` is safe here; the
plan should verify whether `select_related('telescope_label_meta')` is also applicable
and may use either — functionally identical at this scale.]

**Why Count annotation and not Prefetch+to_attr for active_todo_count:**
The `active_todos` property returns `self.todos.filter(is_completed=False)`. The template
calls `.count` on this, which is an aggregate. A `Count` annotation adds zero extra queries
— it's a single SQL expression in the main SELECT. `Prefetch+to_attr` would add a second
query and require `len()` instead of `.count` everywhere the property is used. D-07 is
correct: use the Count annotation.

### Pattern 3: URL Shadowing (FOMO calendar_urls.py)

**What:** A new `solsys_code/calendar_urls.py` file that exposes ONLY the root calendar
view under `name='calendar'`. Django tries this include first; unmatched sub-paths
(`create/`, `update/`, etc.) fall through to `tom_common.urls` → `tom_calendar.urls`.

**When to use:** When a FOMO project view needs to override exactly one URL from a
third-party app without forking the entire URL conf.

```python
# Source: CONTEXT.md D-06, confirmed against tom_calendar/urls.py [VERIFIED: codebase]
# solsys_code/calendar_urls.py
from django.urls import path
from solsys_code.views import fomo_render_calendar

app_name = 'calendar'

urlpatterns = [
    path('', fomo_render_calendar, name='calendar'),
]
```

```python
# src/fomo/urls.py — modified (BEFORE the tom_common include)
# [VERIFIED: codebase — current urls.py read; tom_common calendar path confirmed]
urlpatterns = [
    path('observatory/', include('solsys_code.solsys_code_observatory.urls', namespace='solsys_code_observatory')),
    path('ephem/<int:pk>/', Ephemeris.as_view(), name='ephem'),
    path('targets/<int:pk>/makeephem/', MakeEphemerisView.as_view(), name='makeephem'),
    path('calendar/', include('solsys_code.calendar_urls', namespace='calendar')),  # NEW — before tom_common
    path('', include('tom_common.urls')),
]
```

**Why `app_name = 'calendar'` is required in calendar_urls.py:** Django 3.x raises
`ImproperlyConfigured` when an include with a `namespace=` argument is used and the
included urlconf does not define `app_name`. Setting `app_name = 'calendar'` satisfies
this requirement. [ASSUMED — verify this against the project's exact Django version during
plan execution with a quick `python manage.py check` after the URL change.]

**Effect on `reverse('calendar:calendar')`:** The FOMO include is declared before
`tom_common.urls`. Django's `reverse()` searches URL patterns in declaration order and
returns on the first match for a given namespace+name. FOMO's `calendar:calendar` is
found first and resolves to `/calendar/`. The template's `{% url 'calendar:calendar' %}`
will correctly target the FOMO wrapper view. [ASSUMED — functionally correct based on
Django's first-match semantics, but should be validated by running the calendar URL
in the test suite after the change.]

### Pattern 4: CSS !important Override Removal

**What:** Remove `color: #fff !important` from the `.cal-event-all-day` and
`.cal-event-all-day a` rules in `calendar.html`. Add `color: {{ text_color }};` to the
inline `style` attribute on each all-day event `<div>`.

**Lines to change in calendar.html:** [VERIFIED: codebase]
- Line 85: `.cal-event-all-day { ... color: #fff !important; ... }` — remove the color rule
- Line 94: `.cal-event-all-day a { color: #fff !important; ... }` — remove the color rule
- Lines 187–191 (all-day event divs): add `{% text_color_for_bg bg_color as text_color %}`
  before the div and `color: {{ text_color }};` in the inline style attribute
- Lines 193–195: `event.active_todos.count` → `event.active_todo_count` (DISPLAY-09)
- Lines 219–221: `event.active_todos.count` → `event.active_todo_count` (DISPLAY-09)

**Template tag invocation placement:** The `{% text_color_for_bg bg_color as text_color %}`
call must appear AFTER `{% proposal_color event.proposal as bg_color %}` (which sets
`bg_color`). The natural position is immediately after `{% status_border_css event.title as status_border %}`, before the `<div>` block. Applies only to the all-day event branch
(D-04 — timed events excluded).

### Anti-Patterns to Avoid

- **Hardcoding text color per palette entry:** Do not write `if hex_color == '#005f9e': return '#fff'`. The formula must be dynamic to survive future palette changes.
- **Using `select_related` on the wrong side:** `select_related` on a reverse OneToOne works in Django but only joins one table; confirm the FK direction before choosing `select_related` vs `prefetch_related`.
- **Calling `event.active_todos.count` anywhere after the annotation:** After the Count annotation, `event.active_todo_count` is an integer attribute; calling `event.active_todos.count` still works but hits the DB again. The template must use only the annotated attribute name.
- **Adding the FOMO calendar URL AFTER tom_common.urls:** If `path('calendar/', include('solsys_code.calendar_urls', ...))` is added after `path('', include('tom_common.urls'))`, it is never reached — Django matches `tom_common.urls` first.
- **Omitting `app_name` from calendar_urls.py:** Without `app_name`, the include with `namespace='calendar'` may raise `ImproperlyConfigured` at startup.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| N+1 for OneToOneField reverse | Custom caching dict in view | `prefetch_related('telescope_label_meta')` | Django loads all related rows in one query and attaches them in Python; no extra code |
| N+1 for filtered aggregate | Manual counting loop in view | `Count('todos', filter=Q(todos__is_completed=False))` annotation | One SQL column in the main query; zero extra round-trips |
| Hex color parsing | Manual string slicing loop | `int(h[0:2], 16)` stdlib | Python's `int(x, 16)` is the canonical hex parse; no library needed |

**Key insight:** The WCAG formula is 5 lines of Python math (linearize 3 channels, compute luminance, compare contrast ratio). No accessibility library is needed for this scale of operation.

## Common Pitfalls

### Pitfall 1: `!important` Cascade Fight

**What goes wrong:** Removing `color: #fff !important` from `.cal-event-all-day` but
keeping it in `.cal-event-all-day a` (or vice versa) causes anchor text inside the div
to ignore the inline `color:` style.

**Why it happens:** The `a` rule has higher specificity than an inherited inline color
when `!important` is present. Both CSS rules must be updated.

**How to avoid:** Remove color declarations from BOTH `.cal-event-all-day` AND
`.cal-event-all-day a`. The inline `style="color: {{ text_color }};"` on the parent div
then cascades down to the `<a>` child without conflict (no `!important` blocking it).

**Warning signs:** Template integration test renders white text on dark background even
when `text_color_for_bg` is returning `'#fff'` — check the `a` rule.

### Pitfall 2: Template Variable Scope for `as text_color`

**What goes wrong:** `{% text_color_for_bg bg_color as text_color %}` is placed INSIDE
the `{% if event.telescope_label_meta.is_verified == False %}` branch but `color:
{{ text_color }}` is needed on the `{% else %}` branch too.

**Why it happens:** The two div branches (`dashed-border` vs clean) each have their
own inline style attribute; `text_color` must be in scope for both.

**How to avoid:** Place the `{% text_color_for_bg bg_color as text_color %}` tag ONCE,
immediately after `{% status_border_css event.title as status_border %}` and before the
`{% if event.telescope_label_meta.is_verified == False %}` branch. Both branches then
read the same `text_color` variable.

**Warning signs:** `VariableDoesNotExist` error in the template, or one branch renders
with the wrong text color.

### Pitfall 3: annotate() + prefetch_related() Order

**What goes wrong:** Placing `prefetch_related` call before the `filter()` call, or
calling `list(events)` before `.annotate()` is chained.

**Why it happens:** Django queryset methods are lazy — order of chaining does not
affect the SQL, but `list()` executes the query immediately. All `prefetch_related()`
and `annotate()` calls must be chained BEFORE `list(events)`.

**How to avoid:** Build the full queryset chain (filter → prefetch_related → annotate)
before `events = list(events)`.

### Pitfall 4: URL app_name Mismatch

**What goes wrong:** `reverse('calendar:calendar')` raises `NoReverseMatch` or resolves
to the wrong URL after the FOMO calendar path is added.

**Why it happens:** Two includes with `namespace='calendar'` both in play; or
`app_name` not set in `calendar_urls.py`, causing Django to reject the include.

**How to avoid:** Set `app_name = 'calendar'` in `solsys_code/calendar_urls.py`. Run
`python manage.py check` immediately after adding the URL — Django validates namespace
configuration at startup. Then run the template integration test to confirm
`reverse('calendar:calendar')` hits the FOMO wrapper.

### Pitfall 5: telescope_label_meta prefetch with no sidecar row

**What goes wrong:** After `prefetch_related('telescope_label_meta')`, accessing
`event.telescope_label_meta.is_verified` on an event with no sidecar row raises
`RelatedObjectDoesNotExist` (which is a subclass of `DoesNotExist`).

**Why it happens:** The prefetch pre-loads the related object but does not create a
sentinel for missing rows. The existing calendar.html already silences this (Phase 8):
the template tests `== False` explicitly, which is falsy for missing rows via the
`RelatedObjectDoesNotExist` exception path silenced by `{% if %}`.

**How to avoid:** Do NOT change the template condition `{% if
event.telescope_label_meta.is_verified == False %}`. This existing pattern handles
missing sidecar rows correctly regardless of prefetch. The prefetch merely prevents the
per-event DB hit for rows that DO exist.

## Code Examples

### DISPLAY-08: text_color_for_bg template tag

```python
# Source: CONTEXT.md §Specific Ideas — WCAG 2.1 formula; pattern from existing simple_tags
# [CITED: CONTEXT.md]

def _relative_luminance(hex_color: str) -> float:
    """Return relative luminance (0.0–1.0) for a #rrggbb hex color per WCAG 2.1."""
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
    any background with luminance ≤ 0.183; darker backgrounds always prefer white.

    Args:
        hex_color: A '#rrggbb' hex color string (e.g. '#005f9e').

    Returns:
        '#fff' if white text achieves ≥ 4.5:1 contrast; '#000' otherwise.
    """
    lum = _relative_luminance(hex_color)
    white_contrast = 1.05 / (lum + 0.05)
    return '#fff' if white_contrast >= 4.5 else '#000'
```

### DISPLAY-08: Template integration (calendar.html, all-day branch)

```html+django
<!-- In the all-day event loop — add text_color_for_bg after existing tags -->
{% proposal_color event.proposal as bg_color %}
{% status_border_css event.title as status_border %}
{% text_color_for_bg bg_color as text_color %}   <!-- NEW -->
<div class="cal-event cal-event-all-day-row" ...>
  ...
  {% if event.telescope_label_meta.is_verified == False %}
  <div class="cal-event-all-day" style="background-color: {{ bg_color }}; color: {{ text_color }}; {{ status_border }} border: 2px dashed rgba(0, 0, 0, 0.65);">
  {% else %}
  <div class="cal-event-all-day" style="background-color: {{ bg_color }}; color: {{ text_color }}; {{ status_border }}">
  {% endif %}
```

### DISPLAY-09: Wrapper view (solsys_code/views.py)

```python
# Source: tom_calendar/views.py (read in full) — wrapper shadows the queryset section only
# [VERIFIED: codebase — full render_calendar function read]
from django.db.models import Count, Q
from tom_calendar.models import CalendarEvent
from tom_calendar.views import render_calendar as _tom_render_calendar  # for context-building

def fomo_render_calendar(request, month=None):
    """Shadow tom_calendar.views.render_calendar to inject prefetch + Count annotation.

    Eliminates two N+1 query patterns per CalendarEvent:
    - telescope_label_meta OneToOneField reverse accessor (DISPLAY-09)
    - active_todos.count filtered aggregate (DISPLAY-09)

    All context-building and rendering logic is replicated from the upstream
    render_calendar function; only the queryset construction is changed.
    """
    # [Full render_calendar body replicated here with queryset augmented:]
    events = CalendarEvent.objects.filter(
        start_time__date__lte=...,  # same filter as upstream
        end_time__date__gte=...,
    ).prefetch_related(
        'telescope_label_meta',
    ).annotate(
        active_todo_count=Count('todos', filter=Q(todos__is_completed=False))
    )
    events = list(events)
    # [remainder of render_calendar body unchanged]
```

### DISPLAY-09: Parametrized WCAG test (test_calendar_display_extras.py)

```python
# Pattern: Django parametrized tests via subTest or pytest.mark.parametrize
# [ASSUMED — project uses Django TestCase; use subTest for parametrization]

from solsys_code.templatetags.calendar_display_extras import (
    PROPOSAL_PALETTE, NEUTRAL_SLOT_COLOR, text_color_for_bg
)

class TextColorForBgTest(TestCase):
    def test_all_palette_colors_return_white(self):
        """DISPLAY-08: all 8 PROPOSAL_PALETTE entries achieve WCAG AA with white text."""
        for hex_color in PROPOSAL_PALETTE:
            with self.subTest(hex_color=hex_color):
                self.assertEqual(text_color_for_bg(hex_color), '#fff')

    def test_neutral_slot_returns_white(self):
        """DISPLAY-08: NEUTRAL_SLOT_COLOR (#5a6268) achieves WCAG AA with white text."""
        self.assertEqual(text_color_for_bg(NEUTRAL_SLOT_COLOR), '#fff')
```

### DISPLAY-09: N+1 query-count assertion (test_calendar_template.py)

```python
def test_display09_calendar_does_not_n_plus_one_telescope_label(self):
    """DISPLAY-09: query count is bounded regardless of event count."""
    with self.assertNumQueries(expected_count):
        self._get_calendar()
```

Note: `expected_count` must be determined empirically by running with `assertNumQueries`
in a debug context or by reading Django SQL logs. The assertion should use a small fixed
upper bound (e.g., ≤ 10) rather than an exact count, so the test is robust to minor
framework overhead changes. Alternatively: assert that the count WITH multiple events
is the same as WITH one event (linear growth disproves the O(n) pattern).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `color: #fff !important` hardcoded in CSS | `text_color_for_bg` formula-driven inline style | Phase 12 | Supports future palette expansion without manual auditing |
| N+1 `telescope_label_meta.is_verified` per event | `prefetch_related('telescope_label_meta')` | Phase 12 | O(1) queries for telescope labels regardless of event count |
| `event.active_todos.count` (2 queries per event) | `active_todo_count` Count annotation | Phase 12 | One SQL column instead of N×2 extra queries |

**Still in place after Phase 12:**
- The `CalendarEvent.active_todos` property on the tom_calendar model is unchanged — the property still exists for use outside the calendar template. Only the template's access pattern changes.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `prefetch_related('telescope_label_meta')` works for a reverse OneToOneField where the FK is on the sidecar model | Architecture Patterns | If `select_related` is required instead, swap the call — both eliminate the N+1 |
| A2 | Django does not raise an error when two includes share `namespace='calendar'` in the same URL conf | Architecture Patterns, Pitfall 4 | If it raises `ImproperlyConfigured`, the fix is to make `calendar_urls.py` a full replacement of `tom_calendar.urls` (re-export all views) so `tom_calendar.urls` is never reached for `calendar/` paths |
| A3 | `app_name = 'calendar'` is required in `calendar_urls.py` when using `namespace='calendar'` in include | Architecture Patterns | If wrong direction: either omitting `app_name` works fine, or a different `app_name` value is needed |
| A4 | The exact query count assertion value for `assertNumQueries` | Code Examples | Use an upper bound (≤ 10) rather than exact count; adjust after first passing run |

**If this table were empty:** All claims would be verified — none are here due to Django namespace subtleties that require empirical validation on this specific Django version.

## Open Questions

1. **URL namespace: full replacement or fallthrough?**
   - What we know: Django supports multiple includes with the same namespace (first-match semantics)
   - What's unclear: Whether the specific Django version in this project raises an error at startup when two includes share `namespace='calendar'`
   - Recommendation: Run `python manage.py check` immediately after adding the FOMO calendar path; if it fails, switch to the full-replacement approach (calendar_urls.py re-exports ALL tom_calendar views, eliminating the need for fallthrough)

2. **`active_todo_count` zero vs None**
   - What we know: `Count` annotation returns 0 when there are no matching related rows; the template condition is `{% if event.active_todos.count %}` (falsy check)
   - What's unclear: whether the annotation returns `0` (falsy) or `None` (also falsy) — both work with `{% if event.active_todo_count %}` condition
   - Recommendation: The plan should note that the template condition works correctly for both 0 and None; no special handling needed.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Django ORM `Count`, `Q` | DISPLAY-09 queryset annotation | ✓ | (project Django) | — |
| Python `int(x, 16)` | WCAG hex parsing | ✓ | stdlib | — |
| `./manage.py test` | Test suite | ✓ | Django test runner | — |

No missing dependencies. This phase is code-only with no external service requirements.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django TestCase (`./manage.py test`) |
| Config file | `pyproject.toml` (`testpaths = ["tests", "src", "docs"]`) — Django app tests run under `./manage.py test solsys_code` |
| Quick run command | `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DISPLAY-08 | All 8 palette colors return '#fff' from text_color_for_bg | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras` | ✅ (add new test class) |
| DISPLAY-08 | NEUTRAL_SLOT_COLOR returns '#fff' | unit | same | ✅ (add to new test class) |
| DISPLAY-08 | Dark background returns '#fff'; bright background returns '#000' (formula correctness) | unit | same | ✅ (add parametrized case) |
| DISPLAY-08 | Calendar renders with `color: #fff` in inline style for all-day event divs | integration | `./manage.py test solsys_code.tests.test_calendar_template` | ✅ (add assertion) |
| DISPLAY-09 | Calendar renders without N+1 (query count bounded) | integration | same | ✅ (add assertNumQueries test) |
| DISPLAY-09 | `active_todo_count` annotation present in rendered HTML | integration | same | ✅ (add assertion) |

### Sampling Rate

- **Per task commit:** `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template`
- **Per wave merge:** `./manage.py test solsys_code`
- **Phase gate:** Full Django suite green AND `python -m pytest` passes before `/gsd-verify-work`

### Wave 0 Gaps

None — test files exist. New test methods are added to existing test classes, not new files.
Ruff must pass: `ruff check . && ruff format --check .`.

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | `text_color_for_bg` receives hex strings from internal constants only; no user input path |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Template injection via hex_color | Tampering | Not applicable: `hex_color` input to `text_color_for_bg` always comes from `PROPOSAL_PALETTE` / `NEUTRAL_SLOT_COLOR` constants, never from user-supplied data; tag returns only `'#fff'` or `'#000'` |
| N+1 → DoS via large event months | DoS | The Count annotation and prefetch eliminate the O(N) query growth; no new attack surface |
| Raw SQL injection via Count annotation | Tampering | Not applicable: `Count('todos', filter=Q(todos__is_completed=False))` uses Django ORM — no raw SQL |

**Security assessment:** This phase is low-risk. The WCAG tag processes only internal palette constants. The ORM changes use parameterized queries. No new authentication or authorization surface is introduced.

## Sources

### Primary (HIGH confidence — codebase reads)

- `solsys_code/templatetags/calendar_display_extras.py` — existing pattern for `@register.simple_tag`, `PROPOSAL_PALETTE`, `NEUTRAL_SLOT_COLOR` [VERIFIED: codebase]
- `src/templates/tom_calendar/partials/calendar.html` — exact line numbers for CSS rules and N+1 template variables [VERIFIED: codebase]
- `/home/tlister/venv/fomo311_venv/.../tom_calendar/views.py` — full `render_calendar` function body confirmed; queryset, context-building, and render pattern [VERIFIED: codebase]
- `/home/tlister/venv/fomo311_venv/.../tom_calendar/models.py` — `CalendarEvent.active_todos` property; `EventTodo` FK [VERIFIED: codebase]
- `solsys_code/models.py` — `CalendarEventTelescopeLabel.event` OneToOneField with `related_name='telescope_label_meta'` [VERIFIED: codebase]
- `/home/tlister/venv/fomo311_venv/.../tom_common/urls.py` — `path('calendar/', include('tom_calendar.urls', namespace='calendar'))` [VERIFIED: codebase]
- `src/fomo/urls.py` — current URL conf; no existing `calendar/` path before `tom_common` [VERIFIED: codebase]
- `solsys_code/tests/test_calendar_display_extras.py` — existing test class pattern and imports [VERIFIED: codebase]
- `solsys_code/tests/test_calendar_template.py` — existing integration test setup [VERIFIED: codebase]
- `.planning/phases/12-display-polish/12-CONTEXT.md` — all locked decisions, WCAG formula, line numbers [CITED: CONTEXT.md]

### Secondary (MEDIUM confidence — training knowledge)

- Django ORM `prefetch_related` behavior for reverse OneToOneField [ASSUMED — verify with `select_related` alternative]
- Django URL namespace fallthrough behavior across two includes with the same namespace [ASSUMED — verify with `python manage.py check`]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all tools from existing codebase
- WCAG formula: HIGH — formula given verbatim in CONTEXT.md; arithmetic is deterministic
- N+1 fix queryset API: HIGH — confirmed against actual tom_calendar code read from venv
- URL namespace approach: MEDIUM — Django behavior assumed; `manage.py check` needed for empirical confirmation

**Research date:** 2026-06-27
**Valid until:** 2026-07-27 (stable Django ORM API; WCAG formula does not change)
