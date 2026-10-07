# Phase 38: tom_calendar override comparison (input to Phase 39)

Written by plan 38-03, Task 3, on 2026-10-07. Phase 39 (Calendar Write Access: ACCESS-01/02, WARN-01) starts from this note.

## Method

- Downloaded the `tomtoolkit==3.0.1` and `tomtoolkit==3.1.0` wheels with `pip download --no-deps --only-binary :all:`, each into its own new directory under `$HOME/tmp/phase38-wheels/`, and unpacked each into its own new directory with `python -I -m zipfile -e`, run from the repo root. Nothing from the wheels was imported, installed or executed; they were only compared with `diff`/`cmp`.
- `diff -rq -x __pycache__` of the 3.1.0 wheel's `tom_calendar/` against the `tom_calendar/` installed in the dev venv (`/home/tlister/venv/devel_fomo311_venv`): no differences. The comparison below is therefore against the code that actually runs.
- `diff -rq -x __pycache__` of the 3.0.1 wheel's `tom_calendar/` against the 3.1.0 wheel's: no differences. The whole `tom_calendar` package (views, urls, templates, tags, models, migrations) is byte-identical between the two versions.
- Each FOMO override was compared with its 3.1.0 upstream file using `diff -u`.

## The three tom_calendar overrides

| FOMO file | Upstream 3.1.0 path (in the wheel) | Upstream changed 3.0.1 to 3.1.0? | How FOMO's copy differs from 3.1.0 |
|-----------|-----------------------------------|----------------------------------|------------------------------------|
| `solsys_code/calendar_urls.py` | `tom_calendar/urls.py` | no | Same six routes with the same names and arguments (`calendar`, `create-event`, `update-event`, `delete-event`, `create-todo`, `update-todo`). Differences: the root route `''` is served by `solsys_code.views.fomo_render_calendar` instead of upstream's `render_calendar`; the other five routes still call the upstream `tom_calendar.views` functions unchanged; `app_name = 'calendar'` (upstream: `'tom_calendar'`); FOMO adds a module docstring. |
| `src/templates/tom_calendar/partials/calendar.html` | `tom_calendar/templates/tom_calendar/partials/calendar.html` | no | Large additions on top of the upstream month grid: the campaign chip (`campaign_chip.html`), the unused-night marker and CSS (`cal-event-unused`), the classical-night stripe (`cal-event-classical`, telescope colour), proposal colour and status-border tags, the proposal legend and filter, `active_todo_count` (the view's annotation) in place of `active_todos.count`, the `utc_offset` query parameter carried through the previous, next, today and `data-url` links, and a separate dashed-border branch for events whose telescope label is unverified. Click targets are unchanged from upstream: a day cell opens `calendar:create-event` and an event opens `calendar:update-event`, both by `hx-get`, then the modal is opened with the Bootstrap 5 API. Observation (not a 3.1.0 change, see below): FOMO's copy still uses a few Bootstrap 4 utility names where upstream uses Bootstrap 5 ones (`border-left`/`border-right`, `mr-2`, `font-weight-bold`, `var(--white)` against upstream's `border-start`/`border-end`, `me-2`, `fw-bold`, `var(--bs-white)`). |
| `src/templates/tom_calendar/partials/event_form.html` | `tom_calendar/templates/tom_calendar/partials/event_form.html` | no | (1) A long header comment and `{% load django_bootstrap5 attribution_display_extras calendar_display_extras %}`. (2) The URL label shows the "View" link only for http(s) addresses (`is_web_url` filter, with `rel="noopener noreferrer"`), otherwise a "(not a web link)" note. (3) The Save, Save-and-edit and Delete buttons are plain `<button>` elements, where upstream uses `{% bootstrap_button %}` for the first two and a differently laid-out Delete button; the button label is "Save and edit" against upstream's "Save and Edit". (4) After `</form>` and before the "Todo list" heading, FOMO inserts the blocks that upstream does not have: the observation-series line (`observation_series_decoration`), the attributed campaign run block (`campaign_decoration`) and the staff-only candidate-attribution branch for an event that has no campaign link yet. |

What Phase 39 needs from this:

- WARN-01 edits `event_form.html`: the exact list of FOMO-only blocks is items (1) to (4) above; everything outside the inserted blocks is upstream's 3.1.0 text apart from the button markup in item (3).
- ACCESS-01/02 touch the URL wiring and the month view's click targets: the URL entries are the six routes above (only the root view is FOMO's; create, update, delete and the todo routes are upstream's views, so any access rule on those has to be added around them, not inside `calendar_urls.py`'s root view), and the click targets are the day-cell `hx-get` to `calendar:create-event` and the event `hx-get` to `calendar:update-event` in `calendar.html`.
- `src/fomo/urls.py` mounts `solsys_code.calendar_urls` as namespace `calendar` ahead of `tom_common.urls`. tomtoolkit 3.1.0's `tom_common/urls.py` line 62 mounts `tom_calendar.urls` under the same namespace `calendar`; that is the source of the one `urls.W005` warning the system check prints. FOMO's entry comes first, so it wins; this is unchanged from 3.0.1.

## Other FOMO overrides of tomtoolkit templates

Every file under `src/templates/` (26 files) was checked against every `templates/` directory in the 3.0.1 and 3.1.0 wheels, and against the installed packages in the dev venv. Four paths shadow an upstream file: the two calendar partials above and the two below. The other 22 files have no upstream counterpart.

| FOMO file | Upstream 3.1.0 path | Upstream changed 3.0.1 to 3.1.0? | How FOMO's copy differs |
|-----------|---------------------|----------------------------------|-------------------------|
| `src/templates/tom_common/index.html` | `tom_common/templates/tom_common/index.html` | no (byte-identical) | Home page rewritten for FOMO: welcome heading, Rubin and TOM logos, links to targets and observations, SSSC link. |
| `src/templates/tom_targets/partials/module_buttons.html` | `tom_targets/templates/tom_targets/partials/module_buttons.html` | no (byte-identical) | Adds a branch so the "Make Ephemeris" button renders only for non-sidereal targets; every other button still goes through `show_individual_app_partial`. |

## Conclusion

No upstream file that FOMO shadows changed between tomtoolkit 3.0.1 and 3.1.0: `tom_calendar` as a whole, `tom_common/index.html` and `tom_targets/partials/module_buttons.html` are all byte-identical across the two versions. So no upstream change is newly hidden by an override, and there is no SYNC-07 fix from this comparison. No FOMO file was changed for it.

Two observations for the developer, neither caused by 3.1.0 and neither fixed here:

1. `calendar.html` keeps some Bootstrap 4 utility class names (listed in the first table) that Bootstrap 5, which tomtoolkit 3.x loads, does not define. Upstream 3.0.1 and 3.1.0 are the same file, so this already existed before the sync. Phase 39 is editing the calendar templates and can decide whether to tidy it.
2. Upstream 3.1.0's own `partials/calendar.html` writes `data-bs-url=` while its `calendar_page.html` reads `cal.dataset.url`, which only matches a `data-url` attribute. FOMO's override writes `data-url`, which is what `calendar_page.html` reads (and `test_calendar_template.py` asserts), so FOMO's copy is the consistent one. If the override were ever deleted in favour of upstream's, that attribute would need checking.
