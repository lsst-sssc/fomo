# Open research questions

Questions surfaced during exploration that need deeper investigation before they can be
treated as settled. Each entry records where it came from and what depends on the answer.

## 2026-09-03 — from /gsd-explore: observation-first calendar layering

### Q: Does `tom_observations` core provide any ObservationGroup timeline or calendar view?

- **Status:** unresolved — the research pass abstained (targeted searches of
  `facility.py`, `models.py`, `updatestatus.py` found nothing, but `views.py` and
  templates were not exhaustively checked for Plotly/timeline code).
- **Why it matters:** if a group timeline convention already exists upstream, the
  base-layer projector's group identity (D1 in
  `.planning/notes/observation-first-calendar-layering.md`) should align with it rather
  than invent a second one.
- **How to answer:** grep the installed `tom_observations` (views, templates, templatetags)
  for `ObservationGroup` rendering, `plotly`, `timeline`, `calendar`.

### Q: Is `tom_calendar` maintained by the TOM Toolkit org, or is it a third-party/LCO package?

- **Status:** unresolved — not researched. What is known (admitted, research pass):
  it is installed in FOMO's site-packages and wired into `INSTALLED_APPS`
  (`src/fomo/settings.py:67`); its `CalendarEvent` model has no FK to `ObservationRecord`
  (`tom_calendar/models.py:1-58`).
- **Why it matters:** decides the contribution target for SEED-004 (contribute the
  projector to tom_calendar vs tom_observations vs a standalone `tom_*` plugin), and
  whether FOMO can rely on `tom_calendar`'s model staying stable.
- **How to answer:** check the package's metadata (`pip show tom-calendar` — Home-page /
  Author), the repository it points to, and whether it lives under github.com/TOMToolkit.
