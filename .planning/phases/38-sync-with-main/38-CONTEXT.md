# Phase 38: Sync with main - Context

**Gathered:** 2026-10-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Bring `issue37-telescope-runs-calendar` onto `origin/main`'s tree with one merge commit (62 `main`
commits since merge base `756680f`; `main` head `a910c17` at discuss time), adopt exactly what `main`
already requires (tomtoolkit>=3.1.0, tom_jpl>=0.3.0, ruff 0.16.9, LINCC python-project-template v2.2.0,
the Django test runner with coverage in CI and pre-commit), remove what the branch still carries that
`main` has dropped (pytest config and extras, `tests/`, the `pytest-check` hook) plus `tom-registration`,
get the full suite green on tomtoolkit 3.1.0 without skipping anything, and refresh draft PR #43 — both
its head branch `issue37-code-only` and its description — to show v2.4.

Requirements: SYNC-01..08. Locked at milestone start (ROADMAP.md "Locked constraints"): merge, never
rebase; match `main`'s floors and nothing more (no Django/astropy/sorcha bumps); fix tests, never skip;
PR #43 stays a draft; `git branch --show-current` before any branch-implicit git command.

Not in this phase: the calendar write-access fix (Phase 39), notebook isolation (Phase 40), todo triage
(Phase 41), re-verification (Phase 42), merging PR #43.

</domain>

<decisions>
## Implementation Decisions

### The merge and the conflict policy
- **D-01:** The **executor performs `git merge origin/main` inside a plan task** and resolves the
  conflicts itself, then **pauses at a checkpoint before making the merge commit** so the developer can
  inspect `git diff --cached` (and `git status`) and approve. Worktree parallelism is off for this phase
  so the merge lands on the real branch, not in a disposable worktree.
  — **Reversibility:** costly — once pushed, the merge commit is a published parent of every later
  v2.5 commit; undoing it means `git revert -m 1` plus re-merging, so the checkpoint is the cheap moment
  to catch a wrong resolution.
- **D-02:** The **merge commit contains `origin/main` plus the minimum edits that resolve the conflicts,
  nothing else.** Dependency floors, the ruff 0.16.9 reformat and lint fixes, CI changes, CLAUDE.md
  rewrites, `tom-registration` removal and test fixes are each their own later commit, so
  `git show <merge>` reads as a pure sync and each follow-up can be reverted alone.
- **D-03:** Conflict resolution rule for the 9 conflicting files (`.gitignore`, `docs/conf.py`,
  `docs/design/design.rst`, `docs/notebooks.rst`, `pyproject.toml`, `solsys_code/admin.py`,
  `solsys_code/apps.py`, `solsys_code/tests/test_bootstrap5_rendering.py`, `src/fomo/urls.py`):
  **keep both sides** — `main`'s Scout / Rubin ToO wiring (views, filters, admin entries, URL patterns,
  navbar items, `@tag('functional')` on the Playwright tests) **and** the branch's calendar / campaign /
  observatory wiring. Nothing FOMO-specific from the branch is dropped by the merge (SYNC-04).
- **D-04:** `pyproject.toml` dependencies after the sync: **`tom-registration` is removed** (not needed;
  `main` never had it) — from `[project.dependencies]`, from `src/fomo/settings.py` (`INSTALLED_APPS`
  entry and `tom_registration.middleware.RedirectAuthenticatedUsersFromRegisterMiddleware`), from
  `docs/installation.rst` and from CLAUDE.md's Key Dependencies list. `settings.py` auto-merges, so
  this removal is a **follow-up commit, not a conflict resolution**. **`timezonefinder>=6.0` (runtime)
  and `graphifyy` (dev) stay**; `main`'s `tom_jpl>=0.3.0` is added; `tomtoolkit` floor becomes
  `>=3.1.0`. No other floor is raised beyond what `main` has.

### The test command in CI and pre-commit
- **D-05:** The CI unit-test matrix runs **`coverage run manage.py test --exclude-tag functional
  --exclude-tag ephemeris_segfault`** followed by `coverage xml` — `main`'s command plus the branch's
  exclusion of the one class (`TestEphemeris`, `@tag('ephemeris_segfault')` in
  `solsys_code/tests/test_views.py`) that crashes the interpreter in native ASSIST. Playwright tests
  carry `main`'s `@tag('functional')` and run in `main`'s separate `functional-tests` job with
  `--tag functional`, as `main` does. The pytest CI step is gone.
- **D-06:** **Adopt `main`'s `django-test` pre-commit hook** (it replaces the dead `pytest-check` hook
  one-for-one) with the same two exclusions in its entry:
  `bash -c "coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault && coverage html"`.
  Developers may `SKIP=django-test git commit` for work in progress; the hook's comment should say so.
- **D-07:** **Drop the branch's `sphinx-build` pre-commit hook** (`main` dropped it; docs are still built
  by the `build-documentation` CI workflow). **Keep `main`'s `pre-executed-nb-never-execute` hook but
  repoint** its `files:` pattern and `args:` at **`docs/notebooks/pre_executed/`** — `main`'s
  `docs/pre_executed/` path matches nothing on the branch, so taken verbatim the hook would be a no-op.
  Otherwise the hook list follows `main`'s file (template-version check at `lincc-frameworks/pre-commit-hooks`
  v0.2.2, `ruff-pre-commit` v0.16.9 for both `ruff` and `ruff-format`, etc.).

### CLAUDE.md and ruff after the merge
- **D-08:** The merged CLAUDE.md **keeps `pre-commit run ruff --all-files` /
  `pre-commit run ruff-format --all-files` as the documented lint/format commands** and keeps the
  D-07 note, updated from v0.2.1 to **v0.16.9**, with its warning that an unpinned `ruff` on PATH can
  report findings the enforced gate does not. `main`'s bare `ruff check . --fix` / `ruff format .` lines
  are **dropped** from the merged file so the two instructions never sit side by side. (Phase 30's
  D-05/D-06/D-07 history: the repo-wide "drift" was executors following a bare `ruff` invocation that
  resolved to whatever version the env held.)
- **D-09:** When ruff 0.16.9 raises a new finding, **fix the code**. The `[tool.ruff.lint]` ignore list
  grows only for a rule that contradicts the existing Rubin-DM `N8xx`-style allowances (astronomical
  names such as `H`, `G`, `RA_deg`) or that `main`'s own config already ignores. A lint fix that would
  change behavior is flagged for the developer, never applied silently — and if it changes the behavior
  of a module in CLAUDE.md's notebook map, the paired-docs rule applies to that plan.
- **D-10:** `pyproject.toml` tool blocks: **`[tool.ruff] exclude` is the union** — keep the branch's
  `.planning`, `.planning/**` and `docs/notebooks/ESO_How_to_download_data.ipynb` alongside the
  migrations exclude (so ruff never lints GSD artifacts); **`[tool.coverage.*]` takes `main`'s block**
  (`source = ["solsys_code", "src/fomo", "src/templatetags"]`, `omit` of `_version.py`, `*/migrations/*`,
  `*/tests/*`) so the CI coverage number means the same thing on both branches.

### PR #43
- **D-11:** **Refreshing `issue37-code-only` is in scope.** Recipe: check out `issue37-code-only`,
  `git merge origin/main` into it (its base is the old `main` commit `67fb479`), then **one snapshot
  commit** that sets the tree to the merged `issue37-telescope-runs-calendar` tree **minus `.planning/`**
  (`.claude/` is gitignored; `deploy/`, new in v2.4, is included this time), in the same style as the
  2026-09-01 "Sync code-only branch … through v2.2" commit. **Push normally, no force-push** — the PR's
  four existing commits stay, and its three-dot diff becomes branch-vs-current-`main`. Pushing the head
  branch is not merging the PR; **it stays a draft.**
  — **Reversibility:** costly — a pushed snapshot on a public PR branch can only be undone with a
  further commit or a force-push, which D-11 rules out.
- **D-12:** The rewritten PR body carries: **one section per v2.4 pillar** (observation projector,
  allocation layer, unattended operation, public tallies), a link to
  `docs/runbooks/telescope_runs_calendar.rst`, a **short "how to try it"** section (the few commands an
  evaluator runs, e.g. `migrate`, `load_telescope_runs`, a dry-run of `run_unattended`), and **one line
  saying it stays a draft until v2.5's Phases 39-42 land**. Not a full v1.0→v2.4 changelog.

### Claude's Discretion
- **Commit shape of the ruff bump:** one mechanical `style:` reformat commit under 0.16.9, separate
  from the lint-fix commit(s), so reviewers can skip the reformat.
- **CLAUDE.md Testing section:** rewritten in `main`'s terms (Django runner is the only runner; no
  pytest config, no `tests/`; `@tag('functional')` split) plus the branch's `--exclude-tag
  ephemeris_segfault` convention; the Conventions line that says pre-commit "runs the pytest suite"
  changes to describe the `django-test` hook. The `tom_registration` line in Key Dependencies goes.
- **Where the `tom_calendar` override comparison is recorded** (ROADMAP scope note: compare
  `solsys_code/calendar_urls.py`, `src/templates/tom_calendar/partials/calendar.html` and
  `event_form.html` against tomtoolkit 3.1.0's upstream copies): a short note in this phase directory
  that Phase 39 reads; an upstream change an override now hides is a SYNC-07 fix here.
- **Dev environment upgrade:** in place (`pip install -e .[dev]` after the floor change) is fine, but
  SYNC-02's "fresh install" check must be satisfied by `pip show tomtoolkit` reporting 3.1.0+ and
  `pip show tom_jpl` 0.3.0+; the planner decides whether a throwaway venv is needed to prove it.
- **Order of follow-up commits** after the merge (floors → install → reformat → lint → CI/hooks →
  CLAUDE.md → suite fixes → PR), as long as D-02 holds.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone scope and locked constraints
- `.planning/ROADMAP.md` §"v2.5 Main Sync & Consolidation" → "Locked constraints" and §"Phase 38: Sync
  with main" — the eight SYNC requirements, the orchestrator's 2026-10-06 checks, the scope note on
  ruff and the three `tom_calendar` overrides, and the five success criteria.
- `.planning/REQUIREMENTS.md` §"Sync with main (SYNC)" — SYNC-01..08 verbatim; §"Out of Scope"
  (no PR merge, no bumps beyond `main`'s floors).
- `.planning/PROJECT.md` §"Current Milestone: v2.5" and §"Key Decisions" (the Phase 30 ruff-pin
  decision that D-08 carries forward).
- `.planning/STATE.md` §"Blockers/Concerns" (the PR #43 head-branch question, now settled by D-11).

### What `main` actually has (read from git, not from memory)
- `origin/main:pyproject.toml` — the floors (`tomtoolkit>=3.1.0`, `tom_jpl>=0.3.0`, `ruff>=0.16`,
  `coverage`), the `[tool.ruff]` and `[tool.coverage.*]` blocks D-10 merges.
- `origin/main:.pre-commit-config.yaml` — the hook list D-06/D-07 follow (`django-test`,
  `pre-executed-nb-never-execute`, `ruff-pre-commit` v0.16.9, template check v0.2.2).
- `origin/main:.github/workflows/testing-and-coverage.yml` and `smoke-test.yml` — the Django runner
  with `coverage`, the `functional-tests` job, `uv` install, SPICE-kernel cache.
- `origin/main:CLAUDE.md` — `main`'s Testing section wording (reuse its terms; keep D-08's commands).
- `origin/main:solsys_code/tests/test_bootstrap5_rendering.py` — the `@tag('functional')` D-05 relies on.

### Repository conventions
- `CLAUDE.md` (branch) §"Commands", §"Testing", §"Conventions" — the D-07 note being updated; the
  branch-verification rule; the paired-docs rule that applies if a SYNC-07 fix changes module behavior.
- `docs/runbooks/telescope_runs_calendar.rst` — the runbook PR #43's body links (D-12).

### PR #43
- GitHub PR #43 "Telescope runs calendar (issue #37)", draft, head `issue37-code-only` (4 commits on
  `67fb479`; last `5a1f27e` 2026-09-01), base `main`. Read its current body before rewriting.

</canonical_refs>

<code_context>
## Existing Code Insights

### State of the two branches (scouted 2026-10-07)
- `git rev-list --left-right --count HEAD...origin/main` → 2331 branch commits, 62 `main` commits;
  merge base `756680f`. `git merge-tree --write-tree HEAD origin/main` reports the 9 conflicts in D-03;
  `.pre-commit-config.yaml`, `CLAUDE.md`, `src/fomo/settings.py`, `docs/index.rst`,
  `docs/installation.rst` and `solsys_code/solsys_code_observatory/views.py` auto-merge.
- `main` already deleted `tests/` and the pytest extras (issue #54, template v2.2.0), so the merge does
  most of SYNC-06; what remains on the branch afterwards is whatever the `pyproject.toml` conflict
  resolution keeps (`[tool.pytest.ini_options]`, `pytest`, `pytest-cov`) and the `pytest-check` hook.
- `main` added: `solsys_code/scout_views.py`, `rubin_too.py`, `filters.py`, `search.py`, `tables.py`,
  their tests, two Scout templates, a navbar entry, `docs/scout_rubin_too.rst`, several design docs, and
  a large `src/fomo/settings.py` rework — none of it overlaps the branch's calendar/campaign code except
  in the shared wiring files listed in D-03.
- Installed now: tomtoolkit 3.0.1, ruff 0.2.1 (env), Django 5.2.17. The Django bump is out of scope.

### Reusable Assets
- `solsys_code/tests/test_views.py:98` — `@tag('ephemeris_segfault')` with its explanatory comment; the
  tag name D-05/D-06 reuse.
- The 2026-09-01 snapshot commit `5a1f27e` on `issue37-code-only` — the recipe D-11 repeats.
- `main`'s `testing-and-coverage.yml` `functional-tests` job — Playwright, chromium cache; taken as-is.

### Established Patterns
- Phase 30 D-05..D-07: lint/format go through `pre-commit run` so the enforced version is the one that
  runs. D-08 keeps this with the new rev.
- The known-flaky Playwright test `test_observatory_create_form_submits_to_observatory_url`
  (STATE.md Deferred Items) now sits behind `@tag('functional')`, so it leaves the unit-test matrix and
  the pre-commit hook; a failure there is not a 3.1.0 regression.
- Baseline at the v2.4 close: 2095 tests OK under
  `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`.

### Integration Points
- `src/fomo/urls.py`, `solsys_code/apps.py`, `solsys_code/admin.py` — where `main`'s Scout wiring and
  the branch's calendar/campaign wiring meet (D-03 union).
- `src/fomo/settings.py` — auto-merges, but D-04's `tom_registration` removal edits it afterwards.
- `.github/workflows/testing-and-coverage.yml`, `smoke-test.yml`, `pre-commit-ci.yml` — CI (D-05).
- `.pre-commit-config.yaml` — hooks (D-06, D-07).
- `CLAUDE.md` — commands, D-07 note, Testing section, Key Dependencies (D-08, discretion).

</code_context>

<specifics>
## Specific Ideas

- The merge checkpoint should show the developer the resolved diff of the 9 files, not a summary.
- The `django-test` hook comment should mention `SKIP=django-test` for WIP commits.
- The PR body's "how to try it" is a handful of commands, not a tutorial; the runbook is the tutorial.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
- "Isolate the campaign table query-count test from the shared file cache" (2026-10-07) — a test fix
  for Phase 41's triage; fold into SYNC-07 only if it flakes on the merged tree.
- "load_telescope_runs: skip comment lines and warn on a bare proposal token" (2026-10-02) — a
  behavior change to a notebook-paired module; Phase 41 triage.
- "Run pre-executed demo notebooks against a scratch DB copy" (2026-10-02) — already WARN-05 in
  Phase 40.

</deferred>

---

*Phase: 38-Sync with main*
*Context gathered: 2026-10-07*
