---
phase: 12-display-polish
fixed_at: 2026-06-28T00:00:00Z
review_path: .planning/phases/12-display-polish/12-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 4
skipped: 1
status: partial
---

# Phase 12: Code Review Fix Report

**Fixed at:** 2026-06-28T00:00:00Z
**Source review:** .planning/phases/12-display-polish/12-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (1 Critical + 4 Warnings)
- Fixed: 4
- Skipped: 1

## Fixed Issues

### CR-01: Unguarded `int()` on query parameters causes HTTP 500 for non-integer input

**Files modified:** `solsys_code/views.py`
**Commit:** c8dc4de
**Applied fix:** Wrapped each of the three `int()` calls (`utc_offset`, `month`, `year`) in `try/except ValueError` blocks with sensible defaults (fall back to 0 / current month / current year respectively). Added `max(-12, min(12, utc_offset))` clamp to prevent multi-year timedelta from an out-of-range offset, and `max(1, min(9999, year))` clamp to keep `year` within Python `datetime.date` safe range. The `month` guard was placed inside the existing `if month is None` branch to preserve the function's `month` positional-argument behavior.

---

### WR-02: Prev / Next / Today navigation buttons discard the UTC-offset selection

**Files modified:** `src/templates/tom_calendar/partials/calendar.html`
**Commit:** 0aa442b
**Applied fix:** Appended `&utc_offset={{ utc_offset }}` to the `hx-get` URL of all three nav buttons (Prev, Next, Today). The Today button uses `{{ utc_offset }}` (the Python template variable from the view context) rather than `{% now %}`, preserving the user's selected offset across all navigation actions.

---

### WR-03: `status_border_css` crashes on `None` title

**Files modified:** `solsys_code/templatetags/calendar_display_extras.py`
**Commit:** 936c1e6
**Applied fix:** Added `title = title or ''` as the first line of `status_border_css`, before the `startswith` check. This converts `None` (and any other falsy value) to an empty string, making `.startswith()` safe while preserving all existing behavior for non-None strings.

---

### WR-04: `_relative_luminance` crashes on invalid `hex_color` input

**Files modified:** `solsys_code/templatetags/calendar_display_extras.py`
**Commit:** 831214f
**Applied fix:** Added a two-part guard at the top of `_relative_luminance`: (1) `if not hex_color or not isinstance(hex_color, str): return 0.0` catches `None` and non-string input; (2) `if len(h) != 6: return 0.0` (after `.lstrip('#')`) catches 3-digit short-form hex like `#abc` and any other non-6-digit value, returning 0.0 (treated as black) in both cases.

---

## Skipped Issues

### WR-01: Dual registration of instance namespace `calendar`

**File:** `solsys_code/calendar_urls.py:14`, `src/fomo/urls.py:25`
**Reason:** The reviewer's suggested fix (remove `app_name = 'calendar'` from `calendar_urls.py`) is not safe in Django 5.2. Django 5.x raises `ImproperlyConfigured: Specifying a namespace in include() without providing an app_name is not supported` when `namespace` is given to `include()` but the included module has no `app_name`. This was verified directly against the runtime (Django 5.2.15 installed).

The alternative fix (remove `namespace='calendar'` from `src/fomo/urls.py`) does not actually resolve the underlying conflict: with only `app_name = 'calendar'` in the module and no `namespace` kwarg, Django derives the instance namespace from `app_name`, producing the same 'calendar' instance namespace — `tom_calendar`'s registration (loaded second via `tom_common.urls`) still overwrites FOMO's in `URLResolver._populate()`.

The only clean resolution would be giving FOMO a distinct instance namespace (e.g., `namespace='fomo_calendar'`) and updating every `{% url 'calendar:...' %}` reversal throughout the template tree — a more invasive change that is out of scope for a code-review fix pass. The risk is latent (today both resolvers produce the same `/calendar/` path, so routing and reverse are consistent), and the `urls.W005` system-check warning is already present to flag this for future attention.

**Original issue:** `reverse('calendar:calendar')` silently resolves through `tom_calendar`'s resolver (not FOMO's `fomo_render_calendar`) because `tom_common.urls` is loaded after FOMO's URL conf and overwrites the 'calendar' instance-namespace entry in Django's `namespace_dict`. Works today because both resolvers emit `/calendar/`; breaks silently if upstream URL prefix changes.

---

_Fixed: 2026-06-28T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
