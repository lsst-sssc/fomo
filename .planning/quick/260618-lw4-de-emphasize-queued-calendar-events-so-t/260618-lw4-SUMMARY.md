---
phase: quick-260618-lw4
plan: 01
status: complete
subsystem: calendar-ui
tags: [django-templates, tom_calendar, ui]
dependency-graph:
  requires: []
  provides:
    - "src/templates/tom_calendar/partials/calendar.html (project-level override of tom_calendar's bundled partial)"
  affects:
    - "Calendar month view rendering for all-day CalendarEvent entries"
tech-stack:
  added: []
  patterns:
    - "Django template DIRS-before-APP_DIRS override to patch a vendored TOM Toolkit plugin template without editing site-packages"
key-files:
  created:
    - src/templates/tom_calendar/partials/calendar.html
  modified: []
decisions:
  - "Used Django template slice comparison (`event.title|slice:\":9\" == \"[QUEUED] \"`) instead of a custom template filter, keeping the change template-only (no .py touched)"
  - "De-emphasis style uses literal rgba(0,0,0,0.06) fill + 1px dashed rgba(0,0,0,0.35) outline — explicitly avoids Bootstrap var(--purple)/var(--bs-purple) tokens for forward-compat with tomtoolkit 3.0.0a10's --bs- var rename"
metrics:
  duration: "~2 minutes"
  completed: 2026-06-18
---

# Quick Task 260618-lw4: De-emphasize [QUEUED] calendar events Summary

De-emphasized `[QUEUED]` all-day calendar events with a muted/dashed-outline treatment via a minimal project-level Django template override, so a tentative LCO observation can never visually outrank a confirmed/placed one.

## What Was Built

`tom_calendar.models.CalendarEvent.color` is assigned purely from `BOOTSTRAP_COLORS[self.pk % len(...)]` — a function of primary key with zero relationship to event status. This meant a `[QUEUED]` (unscheduled) event could coincidentally land a loud color (e.g. purple) while a confirmed/placed event got a quiet one, inverting the intended visual priority on the calendar.

Since `tom_calendar` ships bundled inside the `tomtoolkit` wheel and must not be edited in `site-packages`, the fix is a project-level Django template override:

- Created `src/templates/tom_calendar/partials/calendar.html` as a near-verbatim copy (209 lines) of the installed upstream `tom_calendar/partials/calendar.html`.
- The only behavioral change: the all-day event `<div class="cal-event-all-day" style="background-color: {{ event.color }};">` block (upstream line 158) is now wrapped in a Django template conditional that checks the event title for the `[QUEUED] ` prefix (confirmed verbatim from `sync_lco_observation_calendar.py` line 87: `return f'[QUEUED] {telescope} {instrument}'`).
  - `[QUEUED]` branch: `style="background-color: rgba(0, 0, 0, 0.06); border: 1px dashed rgba(0, 0, 0, 0.35);"` — a low-contrast translucent fill with a dashed outline.
  - Else branch: unchanged, original `style="background-color: {{ event.color }};"`.
- The timed-event block (which doesn't use `event.color` inline) was left untouched, as planned.

Django's `TEMPLATES` `DIRS` (`src/templates`) resolves before `APP_DIRS` (the bundled `tom_calendar` package templates), so this file transparently shadows the upstream one without modifying any installed package.

## Verification

**Diff against upstream** — confined to exactly the targeted region (4 added lines, no other hunks):

```diff
157a158,160
>                 {% if event.title|slice:":9" == "[QUEUED] " %}
>                 <div class="cal-event-all-day" style="background-color: rgba(0, 0, 0, 0.06); border: 1px dashed rgba(0, 0, 0, 0.35);">
>                 {% else %}
158a162
>                 {% endif %}
```

- `./manage.py check` (via `python manage.py check`, since `./manage.py` was not executable in this worktree) → `System check identified no issues (0 silenced).`
- Manual render verification: rendered the all-day-event snippet logic directly through Django's template engine with two fixture-style mock events — one titled `[QUEUED] LCO 1m0 Sinistro` with `color='purple'`, one titled `Confirmed Run NGC1566` with `color='purple'`. Confirmed the `[QUEUED]` event rendered with `background-color: rgba(0, 0, 0, 0.06); border: 1px dashed rgba(0, 0, 0, 0.35);` while the non-`[QUEUED]` event rendered with `background-color: purple;` (its original `event.color`), proving the conditional discriminates correctly and the non-`[QUEUED]` path is unaffected.
- Confirmed the project override (not the site-packages template) is the one Django actually resolves: `template.origin.name` pointed at `src/templates/tom_calendar/partials/calendar.html`.
- `grep` guard confirms no `var(--red|teal|orange|indigo|pink|green|cyan|purple|blue)` or `var(--bs-...)` Bootstrap color tokens appear anywhere in the new style block.
- `ruff check .` clean. No `.py` files were touched by this task, so `ruff format --check .` pre-existing format diffs in unrelated files (two demo notebooks, `src/fomo/settings.py`) are out of scope and untouched by this change — confirmed via `git diff --stat` showing zero uncommitted changes to those files in this worktree.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] `./manage.py check` could not run — missing setuptools_scm-generated `_version.py`**
- **Found during:** Task 2 verification
- **Issue:** This worktree's `src/fomo/_version.py` (gitignored, generated by setuptools_scm at build/install time) was absent, so `python manage.py check` failed with `ModuleNotFoundError: No module named 'src.fomo._version'` before reaching any template/settings checks. This is an environment artifact gap in the fresh worktree checkout, unrelated to the calendar template change — sibling checkouts (`/home/tlister/git/fomo`, `/home/tlister/git/fomo_fresh`) already have this file.
- **Fix:** Wrote an equivalent `src/fomo/_version.py` in this worktree only, matching the format/content pattern of the file in sibling checkouts (this file is `.gitignore`d — confirmed via `git check-ignore` — so it is not tracked or committed; it has zero footprint in git history).
- **Files modified:** `src/fomo/_version.py` (gitignored, not committed)
- **Commit:** N/A (gitignored generated file, not part of any commit)

None of the deviations affected the actual template change in `src/templates/tom_calendar/partials/calendar.html`, which matches the plan exactly.

## Known Stubs

None.

## Self-Check: PASSED

- `src/templates/tom_calendar/partials/calendar.html` — FOUND (created, committed at `517e8bc`)
- Commit `517e8bc` — FOUND in `git log --oneline`
