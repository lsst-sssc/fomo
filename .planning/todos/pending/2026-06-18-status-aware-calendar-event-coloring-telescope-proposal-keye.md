---
created: 2026-06-18T22:44:44.098Z
title: Status-aware calendar event coloring (telescope/proposal-keyed, alpha by confidence)
area: ui
files:
  - tom_calendar/models.py (CalendarEvent.color property, third-party/bundled in tomtoolkit)
  - tom_calendar/utils.py (BOOTSTRAP_COLORS)
  - tom_calendar/templates/tom_calendar/partials/calendar.html:158
  - solsys_code/management/commands/sync_lco_observation_calendar.py (title prefixes only, no color logic today)
---

## Problem

`CalendarEvent.color` (in `tom_calendar`, bundled inside the `tomtoolkit` package —
not separately overridable) is `BOOTSTRAP_COLORS[self.pk % 9]`: purely row-PK-based,
with zero awareness of event status or title prefix. This means a `[QUEUED]` event
(tentative, scheduling window only, might never happen) can coincidentally render in
a more visually prominent color (e.g. purple) than a confirmed/placed event with a
clean title — backwards from what a user would want at a glance. Observed directly
in the FOMO calendar UI (June vs July 2026 screenshots): two `[QUEUED]` events
rendered purple, while a placed/scheduled event rendered in an unremarkable default
color.

This was confirmed NOT a regression from Stage 3 (`sync_lco_observation_calendar.py`):
that command only ever sets/prefixes the `title` string (`_FAILURE_PREFIX_BY_STATUS`
/ `_failure_prefix()`), never touches color. Checked `.planning/PROJECT.md` and
`.planning/ROADMAP.md` for "color|purple|visual|banner|css|style" — banners/title
mechanics are referenced throughout, but visual/color treatment was never part of
any phase's defined scope.

**Forward-compatibility note:** confirmed by inspecting the `tomtoolkit==3.0.0a10`
wheel (vs. the currently-installed `3.0.0a9`) that the upstream Bootstrap4→Bootstrap5
migration renames `BOOTSTRAP_COLORS` entries from `var(--red)` etc. to
`var(--bs-red)` etc., alongside other Bootstrap5 template changes (`fw-bold`,
`border-end`, vanilla-JS `bootstrap.Modal`). FOMO is currently pinned to `3.0.0a9`
with `bootstrap4`/`crispy_bootstrap4`, so nothing is broken today — but any local
fix here must NOT hardcode either `var(--purple)` or `var(--bs-purple)`; use literal
hex/rgba values instead so it survives that eventual upgrade.

## Solution

Local Django template override at `src/templates/tom_calendar/partials/calendar.html`
(Django prefers a project-level template path over the third-party app's bundled
one — `tom_calendar` can't be edited in place since it ships inside `tomtoolkit`).

Two independent pieces, discussed with the user but deliberately deferred together
rather than implemented now:

1. **Color keyed to telescope or proposal, not row `pk`.** A small deterministic
   hash-into-fixed-palette helper (same idea as the existing `pk % len(...)`
   rotation, but keyed on `event.telescope` or `event.proposal` string) so the same
   telescope/proposal renders consistently across the calendar instead of randomly
   per-row.
2. **Status-driven opacity/border treatment**, derived from the `[QUEUED]` /
   `[EXPIRED]` / `[CANCELLED]` / `[FAILED]` title prefix (the only signal currently
   available — no dedicated status field on `CalendarEvent`):
   - solid background — placed/confirmed (no prefix)
   - translucent / outlined — `[QUEUED]` (tentative, scheduling window only)
   - muted + translucent — terminal failure prefixes (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`)

User explicitly suggested "striping" as an alternative/addition to opacity for
status — worth a quick visual exploration (possibly via `/gsd-sketch`) before
committing to one treatment, since this is a visual-design decision, not just an
engineering one.

A narrower, immediate fix (just de-prioritizing `[QUEUED]`'s prominence rather than
building the full telescope/proposal-keyed scheme) was scoped separately and done
first — see the corresponding quick task/commit. This todo is for the fuller
treatment.
