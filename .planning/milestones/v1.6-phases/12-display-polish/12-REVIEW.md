---
phase: 12-display-polish
reviewed: 2026-06-28T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - solsys_code/calendar_urls.py
  - solsys_code/templatetags/calendar_display_extras.py
  - solsys_code/tests/test_calendar_display_extras.py
  - solsys_code/views.py
  - src/fomo/urls.py
  - src/templates/tom_calendar/partials/calendar.html
  - solsys_code/tests/test_calendar_template.py
findings:
  critical: 1
  warning: 4
  info: 2
  total: 7
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-06-28T00:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 12 adds `text_color_for_bg` (WCAG 2.1 contrast tag), the `fomo_render_calendar` wrapper view with prefetch + Count annotation, a FOMO-local `calendar_urls.py` that shadows `tom_calendar.urls`, and template edits. The WCAG formula is implemented correctly; the ORM annotation and prefetch strategy for `telescope_label_meta` are sound. Four issues require attention before ship.

The most serious is an unguarded `int()` conversion on three query parameters in `fomo_render_calendar` — any non-integer value from a crawler, HTMX client, or browser bookmark causes an unhandled `ValueError` and a 500 response. The dual URL-namespace registration is functionally correct today but is load-bearing on an ordering coincidence that will break silently if the upstream URL structure ever changes.

---

## Critical Issues

### CR-01: Unguarded `int()` on query parameters causes HTTP 500 for non-integer input

**File:** `solsys_code/views.py:70-78`
**Issue:** All three query parameters — `utc_offset`, `month`, and `year` — are fed directly to `int()` with no exception handling. Any non-integer value (typo in a bookmark, HTMX client bug, or a scanner sending `?year=abc`) raises `ValueError` which propagates unhandled to Django's error handler, producing an HTTP 500 response. `utc_offset` also has no range check: `?utc_offset=99999` produces a 4166-day `timedelta`, shifting `today` four thousand years into the future and rendering a senseless calendar page (year out of range → second unhandled `ValueError` from `date(year, month, 1)`).

```python
# Current — crashes on non-integer input
utc_offset = int(request.GET.get('utc_offset', 0))
month = int(request.GET.get('month', now_offset.month))
year = int(request.GET.get('year', now_offset.year))
```

**Fix:**
```python
try:
    utc_offset = int(request.GET.get('utc_offset', 0))
except ValueError:
    utc_offset = 0
utc_offset = max(-12, min(12, utc_offset))   # clamp to valid TZ range
offset = timedelta(hours=utc_offset)
now = dj_timezone.now()
now_offset = now + offset
today = now_offset.date()

if month is None:
    _month_raw = request.GET.get('month', now_offset.month)
else:
    _month_raw = month
try:
    month = int(_month_raw)
except ValueError:
    month = now_offset.month
month = max(1, min(12, month))

try:
    year = int(request.GET.get('year', now_offset.year))
except ValueError:
    year = now_offset.year
# clamp to Python datetime-safe range
year = max(1, min(9999, year))
```

---

## Warnings

### WR-01: Dual registration of instance namespace `calendar` — `reverse()` resolves through the wrong resolver

**File:** `solsys_code/calendar_urls.py:14` and `src/fomo/urls.py:25`
**Issue:** `calendar_urls.py` declares `app_name = 'calendar'` and the `include()` call uses `namespace='calendar'`, registering app-namespace **calendar** / instance-namespace **calendar**. `tom_common.urls` (pulled in immediately below via `path('', include('tom_common.urls'))`) registers `path('calendar/', include('tom_calendar.urls', namespace='calendar'))`, which has `app_name = 'tom_calendar'` but also instance-namespace **calendar**. In Django 5.x, `URLResolver._populate()` builds `namespace_dict` keyed by instance namespace; the second registration (tom_calendar's, because it is processed after FOMO's in `urlpatterns`) silently overwrites the first. `reverse('calendar:calendar')` therefore resolves via the `tom_calendar` resolver, not FOMO's. The system happens to work today because both registrations produce the identical URL path `/calendar/`, and URL *routing* is top-to-bottom (FOMO wins). If the upstream URL prefix in `tom_common.urls` ever changes, `reverse()` returns a stale path while routing still hits FOMO — a latent silent mismatch.

Additionally, `app_name = 'calendar'` in `calendar_urls.py` is redundant when `namespace='calendar'` is also passed in `include()`. The `namespace` kwarg to `include()` sets the instance namespace; the module-level `app_name` sets the application namespace. Keeping them identical is fine, but the application namespace is then `'calendar'` (FOMO's) while the reverse lookup silently uses the other instance (`tom_calendar`'s app-namespace `'tom_calendar'`). The design is fragile.

**Fix:** Establish a distinct FOMO instance namespace so `reverse()` is unambiguous, and remove the redundant `app_name`:

In `calendar_urls.py` — remove `app_name = 'calendar'` entirely (let the `namespace` in `include()` own the instance namespace). In `src/fomo/urls.py` — the existing `namespace='calendar'` is sufficient. Django will then use instance-namespace 'calendar' for FOMO exclusively, and tom_calendar's app_namespace 'tom_calendar' stays separate. Test: `resolve(reverse('calendar:calendar')).func` should be `fomo_render_calendar`.

---

### WR-02: Prev / Next / Today navigation buttons discard the UTC-offset selection

**File:** `src/templates/tom_calendar/partials/calendar.html:132-147`
**Issue:** The three HTMX navigation buttons hard-code `month` and `year` in the query string but do not include `utc_offset`. When a user has selected, say, UTC-3 and clicks "Prev", the browser sends the request without `utc_offset`, the view defaults to 0, and all event timestamps are re-rendered in UTC. The UTC-offset `<select>` correctly preserves the value (its own `name="utc_offset"` is included automatically by HTMX), but the buttons are standalone `<button>` elements, not form children.

```html
<!-- Current — loses utc_offset on navigation -->
hx-get="{% url 'calendar:calendar' %}?month={{ prev_month }}&year={{ prev_year }}"
```

**Fix:**
```html
hx-get="{% url 'calendar:calendar' %}?month={{ prev_month }}&year={{ prev_year }}&utc_offset={{ utc_offset }}"
```
Apply the same change to the Next and Today buttons. Today's button should use `utc_offset={{ utc_offset }}` (the Python variable), not `{% now %}`.

---

### WR-03: `status_border_css` crashes on `None` title

**File:** `solsys_code/templatetags/calendar_display_extras.py:94`
**Issue:** `title.startswith('[QUEUED] ')` raises `AttributeError` if `title` is `None`. In practice, `CalendarEvent.title` is a `CharField` without `null=True`, so `None` will not come from the ORM. However, the function's declared signature accepts `str` and callers can pass any Python value; there is no defensive guard. An upstream `tom_calendar` schema migration adding `null=True` to `title`, or a direct Python call with `None`, would crash the template render.

```python
# Current
def status_border_css(title: str) -> str:
    if title.startswith('[QUEUED] '):
```

**Fix:**
```python
def status_border_css(title: str) -> str:
    title = title or ''
    if title.startswith('[QUEUED] '):
```

---

### WR-04: `_relative_luminance` crashes on invalid `hex_color` input

**File:** `solsys_code/templatetags/calendar_display_extras.py:103-104`
**Issue:** `hex_color.lstrip('#')` raises `AttributeError` if `hex_color` is `None`. A short-form color like `#abc` produces `h = 'abc'`; `int(h[4:6], 16)` then calls `int('', 16)` which raises `ValueError`. The function is private and the only caller in production is the `text_color_for_bg` template tag, which only receives values from `proposal_color` (always a valid 7-character `#rrggbb` string). The risk is low today, but the function is a public utility candidate and any direct template usage with an unexpected value crashes the page render with a 500.

**Fix:**
```python
def _relative_luminance(hex_color: str) -> float:
    """Return relative luminance (0.0-1.0) for a #rrggbb hex color per WCAG 2.1."""
    if not hex_color or not isinstance(hex_color, str):
        return 0.0   # treat invalid input as black (worst case → white text returned)
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return 0.0
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    ...
```

---

## Info

### IN-01: N+1 test does not cover events that have a `target_list`

**File:** `solsys_code/tests/test_calendar_template.py:249-266`
**Issue:** `test_display09_query_count_bounded` creates all fixture events with no `target_list` (FK is NULL). The ORM returns `None` for a NULL FK without a DB hit, so the baseline query count does not include any `target_list` lookup. Adding a second event (also without a `target_list`) leaves the count unchanged — the test passes. However, `fomo_render_calendar` does not call `.prefetch_related('target_list')`, and the template accesses `event.target_list` for every event in the render loop (`{% include 'target_list_block.html' with target_list=event.target_list %}`). An event with a non-NULL `target_list` FK would trigger one additional query per event, which would cause the test to fail and reveal the missing prefetch. The test should add at least one fixture event with a `TargetList` to exercise this path.

---

### IN-02: WCAG formula comment slightly imprecise at boundary value

**File:** `solsys_code/templatetags/calendar_display_extras.py:118-119`
**Issue:** The docstring states "White text achieves 4.5:1 against any background with luminance <= 0.183". The exact threshold is lum ≤ 0.8250/4.5 = 0.18333…, which rounds to 0.183 when truncated. The `>=` comparison used in the code (`white_contrast >= 4.5`) is correct — at exactly lum = 0.18333, white_contrast = 4.5 and white is returned. The prose is harmlessly imprecise (it says 0.183, which is slightly conservative — a background at lum = 0.18334 would be called "> 0.183" by the comment but still returns white text, correctly). No code change required; the comment can be tightened to "luminance ≤ 0.1833 (1/12 × 11/15)" if precision matters to future readers.

---

_Reviewed: 2026-06-28T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
