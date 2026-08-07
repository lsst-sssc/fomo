---
phase: quick-260806-ol7
plan: 01
subsystem: docs-notebooks
tags: [campaign-lifecycle, django-test-client, jupyter, sphinx, docs]

# Dependency graph
requires:
  - phase: 26-29 (v2.2 milestone -- the-canonical-run-record)
    provides: "CampaignRun model, campaigns:submit/decide/attribution/attribution_decide views, campaign_attribution scoring, campaign_reconciler dispatch"
provides:
  - "docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb -- a view-driven, pre-executed walkthrough of the full v2.2 campaign lifecycle"
  - "docs/notebooks.rst toctree entry and docs/runbooks/telescope_runs_calendar.rst cross-reference making the notebook discoverable"
  - "CLAUDE.md pairing-map entry recording the notebook as a cross-cutting (not 1:1-module) paired artifact"
affects: [docs, onboarding, future-campaign-workflow-changes]

# Tech tracking
tech-stack:
  added: []
  patterns: ["django.test.Client-driven demo notebook (vs direct-ORM seeding)", "demo-scoped reset cell keyed on TargetList name + legacy url prefix"]

key-files:
  created:
    - docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb
  modified:
    - docs/notebooks.rst
    - docs/runbooks/telescope_runs_calendar.rst
    - CLAUDE.md

key-decisions:
  - "New standalone notebook, not an extension of reconcile_campaign_runs_demo.ipynb -- that notebook stays scoped to the reconciler alone against direct-ORM pre-approved runs (D-02)."
  - "Every state transition (submission, approve, resolve_site, attribution confirm) goes through the real staff-facing view via django.test.Client -- never a direct-ORM CampaignRun.objects.create(approval_status=APPROVED) shortcut (D-01)."
  - "Four CampaignRun variations seeded: CLASSICAL_FILE, LCO_QUEUE, ESO_QUEUE (Paranal/VLT/FORS2-shaped, deliberately unresolvable site_raw), and a class-wide/site-agnostic run (D-03)."
  - "source/telescope_class are hand-stamped after creation (the public form exposes neither), with load-bearing prose explaining this is provenance-only, not a workflow shortcut."

requirements-completed: [OL7-01, OL7-02, OL7-03, OL7-04, OL7-05]

# Metrics
duration: ~2h
completed: 2026-08-06
---

# Quick Task 260806-ol7: Campaign Lifecycle Demo Notebook Summary

**New pre-executed demo notebook drives a campaign and four CampaignRuns end-to-end -- public submission through `campaigns:submit`, staff approval and site resolution through `campaigns:decide`, and operator-assisted attribution through `campaigns:attribution_decide` -- proving that `source` never decides an event's calendar shape, only `telescope_class`/`site` do.**

## Performance

- **Duration:** ~2h (includes environment setup for a fresh git worktree with no prior dev DB)
- **Tasks:** 3
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`: 28 cells (13 code, 15 markdown),
  executed twice end to end with no errors, proving the demo-scoped reset makes it genuinely
  re-runnable against a shared dev DB.
- Every write goes through a real view: `campaigns:submit` (anonymous), `campaigns:decide`
  with `action=approve` and `action=resolve_site` (staff), `campaigns:attribution_decide` with
  `action=confirm` (staff). No `CampaignRun.objects.create(..., approval_status=APPROVED)`
  shortcut anywhere in the notebook's code cells.
- Executed output makes the D-03 window-shape rule directly visible and asserted: the
  classical, LCO-queue and ESO-queue runs each render three `RUN:{pk}:{date}` per-night
  events; only the class-wide run (`telescope_class` set) renders a single bare `RUN:{pk}`
  container. Prose states plainly that `reconcile_run()` reads `source` NOWHERE in its
  dispatch (quick task `260805-tad`).
- One run (ESO queue) reaches the calendar only after a genuine site-review round trip: its
  submitted free-text site (`'Paranal Observatory (site TBC)'`) fails resolution at approve
  time, appears in the approval queue's Sites Needing Review section, and a `resolve_site`
  POST against the seeded `Y23` (Paranal-shaped) Observatory resolves it and retro-projects
  its events.
- One hand-entered orphan `CalendarEvent` (representing a pre-canon, `load_telescope_runs`-
  style entry) is scored via `campaign_attribution.candidates_for_event()` and confirmed
  through the attribution queue, with `CalendarEventMeta.run`/`confirmed_by`/`confirmed_at`
  visible in the executed output.
- `docs/notebooks.rst` toctree, `docs/runbooks/telescope_runs_calendar.rst`'s See also
  section, and CLAUDE.md's notebook pairing map all updated to make the new notebook
  discoverable and to record it as a cross-cutting (not 1:1-module) paired artifact.

## Task Commits

1. **Task 1: Notebook part A -- bootstrap, demo-scoped reset, seeding, four submissions** -
   `8cf4f9a` (feat)
2. **Task 2: Notebook part B -- staff surfaces, attribution, four-way calendar payoff** -
   `175f842` (feat)
3. **Task 3: Wire the notebook into the docs -- toctree, runbook cross-reference, CLAUDE.md
   pairing map** - `6f81adc` (docs)

## Files Created/Modified

- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` - New standalone, pre-executed
  demo notebook (28 cells) walking a campaign and four `CampaignRun`s from public submission
  to calendar events through the real staff-facing views.
- `docs/notebooks.rst` - Added a toctree entry ("The Full Campaign Lifecycle (submission to
  calendar)"), last in the Demonstration Notebooks list.
- `docs/runbooks/telescope_runs_calendar.rst` - Added a one-sentence `:doc:` cross-reference
  to the new notebook in the existing See also section.
- `CLAUDE.md` - Added a notebook-pairing-map entry recording that
  `campaign_lifecycle_demo.ipynb` covers the cross-cutting v2.2 campaign lifecycle
  (submission/approval/attribution/reconciler surfaces) rather than a single module.

## Decisions Made

- New standalone notebook rather than extending `reconcile_campaign_runs_demo.ipynb` (locked
  by CONTEXT.md/PLAN.md D-02): that notebook stays scoped to the reconciler alone.
- Every state transition is driven through the real view via `django.test.Client`, never a
  direct-ORM pre-approved-row shortcut (D-01); verified programmatically by the plan's own
  automated check (`'approval_status=CampaignRun.ApprovalStatus.APPROVED' not in src`).
- `source`/`telescope_class` are the only two fields written directly (not through a view),
  because the public submission form has no field for either -- documented as provenance-only
  stamping, not a shortcut for any state transition.
- Reused the plan's suggested `Y21`/`Y22`/`Y23` demo obscodes and the exact
  `'Paranal Observatory (site TBC)'` deliberately-unresolvable site string, since both were
  already vetted while planning against `selection_to_obscode()`'s parenthesis-stripping
  behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Worktree had no generated `src/fomo/_version.py`**
- **Found during:** Task 1, first notebook execution attempt
- **Issue:** `import django; django.setup()` failed with `ModuleNotFoundError: No module
  named 'src.fomo._version'`. `pyproject.toml`'s `[tool.setuptools_scm] write_to =
  "src/fomo/_version.py"` generates this file at install/build time; it's gitignored, so a
  fresh git worktree checkout has no copy (the main repo's working tree has one from an
  earlier install, but worktrees don't share untracked files).
- **Fix:** Ran `setuptools_scm.get_version(root='.', write_to='src/fomo/_version.py')`
  directly in the worktree to regenerate it. Matches the precedent already documented in
  quick task `260805-tad`'s summary for the same worktree-isolation gap.
- **Files modified:** none committed (the file is gitignored by design -- `.gitignore:29`).
- **Verification:** `django.setup()` succeeds; notebook executes.

**2. [Rule 3 - Blocking] Worktree's `src/fomo_db.sqlite3` was an empty, unmigrated file**
- **Found during:** Task 1, second notebook execution attempt
- **Issue:** `User.objects.get_or_create(...)` raised `OperationalError: no such table:
  auth_user` -- the worktree's dev DB (also gitignored, also not shared across worktrees) was
  a fresh 0-byte SQLite file auto-created by Django on first connection, never migrated.
- **Fix:** Ran `python manage.py migrate` once in the worktree to create the full schema.
  Also matches `260805-tad`'s documented precedent for this exact gap.
- **Files modified:** none committed (`src/fomo_db.sqlite3` is gitignored).
- **Verification:** Notebook executes end to end; `python manage.py test solsys_code`
  (targeted subset, see Verification below) passes 913/913.

---

**Total deviations:** 2 auto-fixed (both Rule 3 -- blocking, both pre-documented worktree-
isolation environment gaps, neither touching a committed file).
**Impact on plan:** Zero scope creep -- both fixes are one-time local environment setup steps
necessary to execute and verify the notebook at all; neither changes any deliverable.

## Issues Encountered

- **ruff formatter version skew:** this repo's `venv` has ruff 0.15.20 installed, while
  `.pre-commit-config.yaml` pins `v0.2.1` (the actual project gate, matching
  `pyproject.toml`'s `>=0.2.1` minimum). For one multi-line `assert ... , (f'...')` pattern,
  the two versions disagree on wrapping style, so re-running the newer venv's
  `ruff format --check .` against the already-committed, pre-commit-formatted notebook falsely
  reports "would reformat." Verified authoritatively against the actual pinned binary
  (`~/.cache/pre-commit/.../py_env-python3.11/bin/ruff`, v0.2.1): `ruff check .` and
  `ruff format --check .` both pass clean, matching what the pre-commit hook itself confirmed
  on both notebook commits. No repo change made; documented here so a future executor in this
  same environment doesn't chase a phantom formatting diff.
- **`python manage.py test solsys_code` runs `test_views.TestEphemeris`, documented to
  segfault in native ASSIST** (per project memory). Ran a targeted 33-module label list
  covering every other `solsys_code`/`solsys_code_observatory` test module (including
  `test_campaign_*`, `test_reconcile_campaign_runs`, `test_ephem_utils`, and
  `test_views.TestSplitNumberUnitRegex`/`TestJPLSBDBQuery`) plus everything else in the app,
  explicitly excluding only `test_views.TestEphemeris`. Result: 913 tests, all pass (`OK`,
  425.7s). This plan changed no application code, so this full-app pass confirms the
  notebook's writes (through the real views) exercised nothing that regressed shared app
  behavior.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The paired-docs convention (CLAUDE.md) is now current for the full v2.2 campaign surface;
  any future plan touching `campaign_views.py`/`campaign_forms.py`/`campaign_attribution.py`/
  `campaign_reconciler.py` should treat `campaign_lifecycle_demo.ipynb` as an in-scope paired
  artifact per the pairing map's trigger clause.
- No blockers. The two environment gaps found (missing `_version.py`, unmigrated dev DB) are
  one-time, per-worktree, gitignored-artifact issues -- already documented by a prior quick
  task and now reinforced here; a future executor in a fresh worktree should expect and
  quickly resolve the same two steps before running any Django command.

---
*Phase: quick-260806-ol7*
*Completed: 2026-08-06*

## Self-Check: PASSED

- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`: FOUND
- `docs/notebooks.rst`: FOUND
- `docs/runbooks/telescope_runs_calendar.rst`: FOUND
- `CLAUDE.md`: FOUND
- Commit `8cf4f9a` (Task 1): FOUND in `git log --oneline --all`
- Commit `175f842` (Task 2): FOUND in `git log --oneline --all`
- Commit `6f81adc` (Task 3): FOUND in `git log --oneline --all`
