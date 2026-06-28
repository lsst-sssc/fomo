---
phase: 12-display-polish
plan: "01"
subsystem: calendar-ui
tags: [wcag, accessibility, n+1-fix, template-tag, url-routing, django-orm]
dependency_graph:
  requires: []
  provides:
    - text_color_for_bg template tag (DISPLAY-08)
    - fomo_render_calendar view with prefetch+annotation (DISPLAY-09)
    - calendar namespace full-replacement URL conf
  affects:
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/views.py
    - src/fomo/urls.py
tech_stack:
  added: []
  patterns:
    - WCAG 2.1 relative luminance formula via stdlib int/float math
    - Django ORM prefetch_related + Count annotation to eliminate N+1
    - URL namespace full-replacement pattern (shadow single view, re-export all sub-paths)
key_files:
  created:
    - solsys_code/calendar_urls.py
  modified:
    - solsys_code/templatetags/calendar_display_extras.py
    - solsys_code/tests/test_calendar_display_extras.py
    - solsys_code/views.py
    - src/fomo/urls.py
    - src/templates/tom_calendar/partials/calendar.html
    - solsys_code/tests/test_calendar_template.py
decisions:
  - "calendar_urls.py is a full replacement of tom_calendar.urls (all 6 URL names), not a single-route shadow — needed so all calendar:* URL reversals resolve through our namespace"
  - "TDD RED/GREEN followed for Task 1: test commit d79a734 (RED), implementation commit cda8789 (GREEN)"
metrics:
  duration: "~45 minutes"
  completed: "2026-06-28"
  tasks_completed: 3
  files_changed: 7
---

# Phase 12 Plan 01: Calendar Display Polish (DISPLAY-08/09) Summary

**One-liner:** WCAG-formula-driven all-day event text color via `text_color_for_bg` template tag, plus N+1 elimination via a FOMO wrapper view with `prefetch_related` + `Count` annotation.

## What Was Built

### Task 1 — text_color_for_bg WCAG template tag (DISPLAY-08)

- Added `_relative_luminance(hex_color: str) -> float` private helper implementing the WCAG 2.1 sRGB transfer function linearization formula
- Added `@register.simple_tag text_color_for_bg(hex_color: str) -> str` returning `'#fff'` when `white_contrast >= 4.5`, else `'#000'` — formula-driven, not hardcoded per palette entry
- All 8 `PROPOSAL_PALETTE` entries and `NEUTRAL_SLOT_COLOR` return `'#fff'`; `#ffffff` returns `'#000'`; `#000000` returns `'#fff'`
- Added `TextColorForBgTest` class (4 test methods) covering palette sweep, neutral slot, pure-white, pure-black boundary cases

### Task 2 — FOMO wrapper view + URL routing (DISPLAY-09)

- Created `solsys_code/calendar_urls.py` as a full replacement of `tom_calendar.urls` — 6 URL patterns (root through all sub-paths) so that all `calendar:*` reversals resolve through our namespace
- Added `fomo_render_calendar` view to `views.py`: replicates `render_calendar` body, replacing the queryset with `.prefetch_related('telescope_label_meta').annotate(active_todo_count=Count('todos', filter=Q(todos__is_completed=False)))`
- Updated `src/fomo/urls.py` to insert `path('calendar/', include('solsys_code.calendar_urls', namespace='calendar'))` BEFORE `path('', include('tom_common.urls'))` — critical ordering so FOMO route wins for `/calendar/`
- `manage.py check` exits 0 (W005 warning for duplicate namespace is expected/harmless); `reverse('calendar:calendar')` → `/calendar/`

### Task 3 — calendar.html edits + integration tests (DISPLAY-08/09)

- Removed both `color: #fff !important` declarations from `.cal-event-all-day` and `.cal-event-all-day a` CSS rules; replaced `a` rule with `color: inherit`
- Added `{% text_color_for_bg bg_color as text_color %}` once in the all-day event loop (after `{% status_border_css %}`, before the `{% if telescope_label_meta.is_verified == False %}` branch)
- Added `color: {{ text_color }};` to inline `style` attribute on BOTH all-day div branches (dashed and clean)
- Renamed `event.active_todos.count` → `event.active_todo_count` at both occurrences (all-day loop lines 193-194, timed loop lines 219-220)
- Added 4 integration tests: DISPLAY-08 inline color present, DISPLAY-08 !important absent, DISPLAY-09 `CaptureQueriesContext` N+1 regression, DISPLAY-09 todo count behavioral parity

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] calendar_urls.py expanded to full replacement of tom_calendar.urls**

- **Found during:** Task 3 — tests failed with `NoReverseMatch: Reverse for 'create-event' not found`
- **Issue:** When our `calendar_urls.py` contained only `path('', fomo_render_calendar, name='calendar')`, all `calendar:create-event`, `calendar:update-event`, etc. reversals raised `NoReverseMatch`. Django's namespace resolution uses the first matching namespace from `urlpatterns` (ours), and ours only had one URL name.
- **Fix:** Added all 6 `tom_calendar` URL patterns to `calendar_urls.py` (importing the upstream view functions). The root still uses `fomo_render_calendar`; all sub-paths delegate to `tom_calendar.views` unchanged.
- **Research confirmation:** The 12-RESEARCH.md explicitly names this as the contingency (A2: "If it raises `ImproperlyConfigured`, the fix is to make `calendar_urls.py` a full replacement").
- **Files modified:** `solsys_code/calendar_urls.py`
- **Commit:** `4caa9d0` (included in Task 3 commit)

## TDD Gate Compliance

Task 1 followed TDD:
- RED gate: commit `d79a734` — `TextColorForBgTest` added with ImportError failure confirmed via `manage.py test`
- GREEN gate: commit `cda8789` — `text_color_for_bg` and `_relative_luminance` implemented; 27 tests pass

## Known Stubs

None — all template values are wired to real data (computed text color from palette constants, todo count from ORM annotation).

## Threat Flags

No new threat surface: `text_color_for_bg` input is always a palette constant (never user-supplied), and the Count annotation uses Django ORM parameterized queries.

## Self-Check: PASSED

- `solsys_code/templatetags/calendar_display_extras.py` — FOUND
- `solsys_code/calendar_urls.py` — FOUND
- `solsys_code/views.py` contains `fomo_render_calendar` — FOUND
- `src/fomo/urls.py` contains `solsys_code.calendar_urls` — FOUND
- `src/templates/tom_calendar/partials/calendar.html` contains `text_color_for_bg` — FOUND
- `solsys_code/tests/test_calendar_display_extras.py` contains `TextColorForBgTest` — FOUND
- `solsys_code/tests/test_calendar_template.py` contains `CaptureQueriesContext` — FOUND
- Commit d79a734 (RED test) — FOUND
- Commit cda8789 (GREEN impl) — FOUND
- Commit cff5233 (Task 2) — FOUND
- Commit 4caa9d0 (Task 3) — FOUND
- Full suite (194 tests) — PASS
