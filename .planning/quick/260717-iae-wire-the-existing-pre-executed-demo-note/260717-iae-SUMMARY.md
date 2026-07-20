---
phase: 260717-iae-wire-the-existing-pre-executed-demo-note
plan: 01
subsystem: docs
tags: [sphinx, nbsphinx, docs, toctree]

# Dependency graph
requires: []
provides:
  - "docs/notebooks.rst toctree now wires all five committed pre-executed demo notebooks into the published docs"
affects: [docs, sphinx-build]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Section + nested .. toctree:: block pattern (matching docs/design/design.rst's 'Design Notes' section) reused for grouping related toctree entries under a page's existing top-level toctree."

key-files:
  created: []
  modified:
    - docs/notebooks.rst

key-decisions:
  - "Verified the change with the exact full-build command from the plan (sphinx-build -M html ... -T -E -n, no notebook exclusion), matching CI/ReadTheDocs behavior rather than the notebook-excluding pre-commit build."
  - "Local dev environment was missing the system pandoc binary (required by nbsphinx to render markdown cells in ANY notebook, confirmed pre-existing/unrelated to this change via a baseline build against the unmodified notebooks.rst). Downloaded the official pandoc 3.1.11 static binary from the upstream jgm/pandoc GitHub release into the session scratchpad and added it to PATH for verification only — not installed system-wide, not committed, not a fix to any repo file."

patterns-established: []

requirements-completed: [DOCS-NOTEBOOKS-TOCTREE]

coverage:
  - id: D1
    description: "The five committed, pre-executed demo notebooks (telescope_runs_demo, load_telescope_runs_demo, sync_lco_observation_calendar_demo, sync_gemini_observation_calendar_demo, import_campaign_csv_demo) are wired into docs/notebooks.rst's toctree under a new 'Demonstration Notebooks' section, and a full Sphinx HTML build renders each to HTML with no toctree warning for the new entries."
    requirement: "DOCS-NOTEBOOKS-TOCTREE"
    verification:
      - kind: other
        ref: "sphinx-build -M html ./docs ./docs/_build/notebook_wiring_verify -T -E -n (full build, no notebook exclusion) — exit code 0, no 'toctree contains reference to nonexisting/excluded document' warning mentioning pre_executed, and notebooks/pre_executed/<name>.html produced for all five notebooks"
        status: pass
    human_judgment: false

# Metrics
duration: ~20min
completed: 2026-07-17
status: complete
---

# Quick Task 260717-iae: Wire pre-executed demo notebooks into Sphinx toctree Summary

**Added a "Demonstration Notebooks" toctree section to `docs/notebooks.rst` so the five already-committed, pre-executed demo notebooks under `docs/notebooks/pre_executed/` are no longer orphaned and now render in the published docs site.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-17T12:20:11Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- `docs/notebooks.rst` now has a "Demonstration Notebooks" section (mirroring the "Design Notes" pattern in `docs/design/design.rst`) with a `.. toctree::` listing all five pre-executed demo notebooks by title.
- Full `sphinx-build -M html` (no notebook exclusion, mirroring CI/ReadTheDocs) confirmed exit code 0, no toctree warning for the new entries, and generated HTML output for each of the five notebooks.
- Existing `notebooks/intro_notebook` entry, all notebook content, and `docs/conf.py` were left untouched.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the five pre-executed demo notebooks to the notebooks.rst toctree** - `6b3c145` (docs)

**Plan metadata:** committed separately by the orchestrator.

## Files Created/Modified
- `docs/notebooks.rst` - Added a "Demonstration Notebooks" section with a toctree listing the five pre-executed demo notebooks, below the existing untouched intro-notebook toctree.

## Decisions Made
- Used the exact section-then-toctree pattern already established by `docs/design/design.rst`'s "Design Notes" section, rather than inventing a new structure, for consistency with the rest of the docs tree.
- Verified against the literal full-build command specified in the plan (not the notebook-excluding pre-commit variant), since that's the build mode that actually exercises the new toctree entries and mirrors what CI/ReadTheDocs run.

## Deviations from Plan

None in the code change itself — plan executed exactly as written (single-file toctree addition, no notebook or `conf.py` edits).

### Verification environment note (not a plan deviation, no code change)

The local dev machine's `pandoc` binary was missing, which nbsphinx needs to render *any* notebook containing markdown cells (all five new notebooks, plus the pre-existing `intro_notebook.ipynb` and `ESO_How_to_download_data.ipynb`, all have markdown cells). Confirmed via a baseline full build against the unmodified `docs/notebooks.rst` (before this task's edit was applied) that the exact same `PandocMissing` failure already occurred — proving this is a pre-existing, unrelated environment gap, not something caused by this change, and out of scope to "fix" in the repo per the Scope Boundary rule.

To still exercise the plan's literal full-build verification command, the official `pandoc` 3.1.11 static binary (from the upstream `jgm/pandoc` GitHub release) was downloaded into the session scratchpad and prepended to `PATH` for the verification run only. No system package was installed, no `pip`/`apt` install occurred, nothing was committed to the repo, and this has no bearing on the actual code change. With pandoc available, the full build succeeded exactly as specified in `<verify>`: exit 0, no toctree warning for `pre_executed` entries, and HTML produced for all five notebooks.

This does not require a `deferred-items.md` entry since it isn't a repo defect — CI/ReadTheDocs already install pandoc as part of their build environment (see `CLAUDE.md`'s tech-stack section, which lists pandoc as a documented platform requirement); it was only this local sandbox that lacked it.

## Issues Encountered

Missing local `pandoc` binary initially blocked the plan's full-build verification command (see above); worked around for verification purposes only via a session-local download, no repo or system changes.

## User Setup Required

None - no external service configuration required. (Note: contributors building docs locally with `sphinx-build` in non-excluded/full mode will need `pandoc` installed on their machine — this is a pre-existing project requirement already documented in `CLAUDE.md`'s Platform Requirements, not something introduced by this task.)

## Next Phase Readiness

The five demo notebooks are now discoverable in the published documentation site. No blockers for future work; this was a self-contained docs-wiring quick task.

---
*Quick task: 260717-iae-wire-the-existing-pre-executed-demo-note*
*Completed: 2026-07-17*

## Self-Check: PASSED

- FOUND: docs/notebooks.rst
- FOUND: 6b3c145 (task commit)
- docs/notebooks.rst contains 5 `pre_executed` toctree entries as expected
