# Deferred Items — Phase 27 Plan 07

## Pre-existing, out-of-scope ruff finding

**Found during:** Task 3's repo-wide `ruff check .` gate.

**Finding:**

```
D103 Missing docstring in public function
 --> docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb:cell 6:6:5
```

**Why deferred, not fixed:** `git diff <plan-07-worktree-base> -- docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb`
shows zero changes from this plan — the notebook was not touched by any
task in 27-07. Per the executor's SCOPE BOUNDARY rule, only issues
directly caused by the current task's changes are auto-fixed; pre-existing
findings in unrelated files are logged here instead. The notebook's most
recent commit (`400a176 fix(27): IN-03 make the pre-executed notebooks
stable under both pinned ruff versions`) suggests this specific finding may
be sensitive to which ruff version runs the check, which is consistent
with it not appearing in this plan's own scoped `ruff check` runs (limited
to the files this plan actually modified) and only surfacing on the full
repo-wide sweep.

**Recommendation:** a future plan/quick-task touching
`sync_gemini_observation_calendar_demo.ipynb` should add a docstring to
`make_gem_record()` in that notebook's cell 6, or confirm the D103 finding
is a ruff-version artifact and adjust the per-file ignore if so.

## Pre-existing, out-of-scope `ruff format --check .` findings

**Found during:** Task 3's repo-wide `ruff format --check .` gate.

**Finding:** three files would be reformatted — none touched by this plan
(confirmed via `git diff <plan-07-worktree-base>`, zero changes in all
three):

- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_nb.py`
- `.planning/quick/260619-f7u-phase-5-notebook-gap-update-sync-lco-obs/verify_project.py`
- `src/fomo/settings.py`

**Why deferred, not fixed:** same SCOPE BOUNDARY rationale as the ruff
lint finding above — these are pre-existing formatting drifts in files
this plan never touched, most likely from a ruff-formatter-version
difference between when they were last committed and the pinned/installed
version in this environment. Reformatting them here would mix an unrelated
formatting-only diff into a gap-closure plan's commit history.

**Recommendation:** a future formatting-hygiene pass (or a `ruff format .`
run under the exact pre-commit-pinned ruff version) should pick these up
separately.
