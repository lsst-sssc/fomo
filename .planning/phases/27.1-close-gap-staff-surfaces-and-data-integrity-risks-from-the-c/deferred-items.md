# Phase 27.1 — Deferred Items

Out-of-scope discoveries found during execution that were not fixed, per the executor's
scope-boundary rule (only auto-fix issues directly caused by the current task's changes).

## Pre-existing repo-wide ruff/format drift

Independently observed and logged by all three Wave 1 plans (27.1-01, 27.1-02, 27.1-03) at
their Task 3 quality gates. All three reached the same conclusion, so the reports are
consolidated here.

`ruff check .` and `ruff format --check .`, run against the full repository as part of each
plan's Task 3 verification, surface failures unrelated to those plans' own changes -- the
same class of pre-existing drift previously logged in
`.planning/phases/26-canonical-record-spike/deferred-items.md`:

- `ruff check .`: `D103` (missing docstring in public function) on `make_gem_record()` in
  `docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb` (cell 6).
- `ruff format --check .`: would reformat
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`,
  `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`,
  `src/fomo/settings.py`.

Confirmed pre-existing: `git diff 2872537392a31d392241162578ff724d576f0e65 -- <each path
above>` is empty for every one of these files -- none was touched by any 27.1 task.

**Likely root cause (from 27.1-03):** the notebook's own commit message (`400a176 fix(27):
IN-03 make the pre-executed notebooks stable under both pinned ruff versions`) suggests the
`D103` finding is ruff-version drift -- the executor's locally installed ruff was 0.15.20 --
rather than a real regression. Worth confirming against the pinned version before anyone
"fixes" the notebook.

Each plan's own files are individually clean under both `ruff check` and
`ruff format --check`:

- 27.1-01: `solsys_code/tests/test_calendar_template.py`,
  `src/templates/tom_calendar/partials/event_form.html`. (The `.html` file is not linted by
  ruff's Python-only default include patterns when discovered via directory walk, which is
  why `ruff check .` / `ruff format --check .` never flag it; explicitly naming it on the
  command line falsely triggers Python-syntax parsing -- confirmed not a real finding.)
- 27.1-02: `solsys_code/models.py`, `solsys_code/admin.py`,
  `solsys_code/tests/test_admin.py`.
- 27.1-03: `solsys_code/campaign_views.py`, `solsys_code/tests/test_campaign_views.py`,
  `src/templates/campaigns/campaign_list.html`, `docs/runbooks/telescope_runs_calendar.rst`.

Not fixed here -- out of scope for plans whose committed deliverables are the event modal
template, the admin/model label changes, and the approval-queue navigation. Whoever next runs
a repo-wide `ruff check . --fix && ruff format .` cleanup pass should pick these up.
