# Phase 38: Sync with main - Research

**Researched:** 2026-10-07
**Domain:** git merge of a long-running Django/TOM Toolkit branch onto `main`; dependency floors (tomtoolkit 3.1.0, tom_jpl 0.3.0); ruff 0.2.1 -> 0.16.9; LINCC python-project-template v2.2.0 (CI + pre-commit); refreshing a draft GitHub PR's head branch
**Confidence:** HIGH (the merge, the conflict set, the post-merge lint/format counts and the full test run were all reproduced in a scratch clone this session)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.

Reviewed todos (not folded): "Isolate the campaign table query-count test from the shared file cache"
(fold into SYNC-07 only if it flakes on the merged tree); "load_telescope_runs: skip comment lines and
warn on a bare proposal token" (Phase 41); "Run pre-executed demo notebooks against a scratch DB copy"
(already WARN-05 in Phase 40). Also out of scope: the calendar write-access fix (Phase 39), notebook
isolation (Phase 40), todo triage (Phase 41), re-verification (Phase 42), merging PR #43.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SYNC-01 | `origin/main` merged with `git merge` (not rebase); merge parents = branch head + `origin/main` head; no main commit missing | "The Merge" section: exact conflict set, resolution table, preflight, checks. Dry run in a scratch clone produced a merge whose `git merge-base --is-ancestor origin/main HEAD` succeeded. |
| SYNC-02 | `tomtoolkit>=3.1.0` and `tom_jpl>=0.3.0`; dev env runs 3.1.0; no other floor raised | The merge itself delivers both floors (pyproject auto-merges them). A fresh `pip install -e .[dev]` gave tomtoolkit 3.1.0, tom_jpl 0.3.0. Env pitfalls (editable tom_jpl, tom-registration, full `/` disk) listed. |
| SYNC-03 | ruff 0.16.9 everywhere; both pre-commit ruff hooks clean on the merged tree | The merge already brings `rev: v0.16.9` and `ruff>=0.16`. Measured: 1 lint finding (SIM103) and 12 files to reformat. Remaining stale text: CLAUDE.md lines 36 and 267. |
| SYNC-04 | LINCC template v2.2.0 as `main` has it; no FOMO-specific file lost | The merge takes `.copier-answers.yml` (`_commit: v2.2.0`), PR template, workflows, `.readthedocs.yml`, `requirements.txt`. No branch-only file is touched by main's diff except the 9 conflicts. |
| SYNC-05 | CI runs the Django runner with coverage; pytest job gone | Workflows arrive from `main` via the merge; they need `--exclude-tag ephemeris_segfault` added (D-05). A trigger gap exists (see Open Questions 1). |
| SYNC-06 | Dead pytest config removed; CLAUDE.md Testing section no longer describes pytest | The merge already removes `[tool.pytest.ini_options]`, `[tool.black]`, `[tool.isort]`, the pytest extras (after the one pyproject conflict is resolved), `tests/`, and the `pytest-check` hook. CLAUDE.md Testing section auto-merges to `main`'s wording. Remaining stale lines listed. |
| SYNC-07 | Full suite passes on merged tree with tomtoolkit 3.1.0; failures fixed, not skipped | Dry run: **2178 tests OK, 0 skipped, zero code fixes needed** on tomtoolkit 3.1.0 / tom_jpl 0.3.0 / Django 5.2.18. `tom_calendar` is byte-identical between 3.0.1 and 3.1.0. |
| SYNC-08 | Draft PR #43's description rewritten for v2.4 + runbook link; stays a draft | PR #43 state read from GitHub. A discovery changes D-11's premise: local `issue37-code-only` already holds an unpushed v2.4 snapshot commit. Safe recipe (`git worktree`, `merge -s ours`) tested. |
</phase_requirements>

## Summary

This phase is mostly *git and tooling hygiene*, and the surprise is how much of it the merge does by
itself. I reproduced the merge in a scratch clone (`git merge --no-commit --no-ff origin/main`, 9 real
conflicts, resolved by hand), then installed the merged tree's dev extras into a **fresh venv** and ran
the whole suite. Result: **2178 tests, OK, none skipped, no FOMO code changes needed** on tomtoolkit 3.1.0,
tom_jpl 0.3.0, Django 5.2.18. The 83 extra tests over the v2.4 baseline of 2095 are `main`'s Scout / Rubin
ToO / search / packaging tests. tomtoolkit 3.1.0 is an *authentication* release (django-allauth, MFA, new
`tom_common` migrations, `tom_registration` deprecated); `tom_calendar`, `tom_targets`, `tom_observations`
and `tom_dataservices` are **byte-identical** between the 3.0.1 and 3.1.0 wheels.

Because `git merge` also carries every `main` change the branch never touched, several things CONTEXT.md
lists as "follow-up commits" are **already done by the merge commit** and will be empty or tiny follow-ups:
`ruff-pre-commit` rev v0.16.9 and `ruff>=0.16`; removal of the `pytest-check` and `sphinx-build` hooks;
the pytest/black/isort pyproject blocks; the legacy `tests/` directory; all `.github/workflows/*`;
`tom-registration` leaving `pyproject.toml`, `settings.py` (apps and middleware) and
`docs/installation.rst`; and CLAUDE.md's Testing section. What genuinely remains is small and concrete:
three hook/CI line edits (add `--exclude-tag ephemeris_segfault`, repoint the notebook hook), one ruff
reformat of 12 files, one SIM103 lint fix, a handful of CLAUDE.md lines, and the PR #43 refresh. Three
conflict resolutions need care because a literal "keep both sides" produces broken code (details below):
a duplicate `nav_items` in `apps.py`, a colliding `TargetAdmin` in `admin.py`, and a `_version.py`
autoapi rule in `docs/conf.py`.

**Primary recommendation:** Do the merge in one executor task with a human checkpoint, resolving the 9
conflicts exactly as the table in "The Merge" shows (not by blindly keeping both sides); then make the
follow-ups small, separate commits; do the PR #43 refresh last, in a separate `git worktree`, never by
switching branches in the primary checkout (it would delete the on-disk `.planning/` tree).

## Architectural Responsibility Map

This phase changes tooling, not application tiers. The map is by "who owns the behaviour".

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Branch/history integrity (merge, parents, no rewrite) | git (local repo) | GitHub (published branch) | `git merge` creates the merge commit; GitHub only sees it after a push |
| Dependency floors | `pyproject.toml` | installed env (`pip`) | Floors are declared in pyproject; the env must be re-installed to prove them |
| Lint/format gate | pre-commit (pinned ruff rev) | `[tool.ruff]` config in pyproject | The pre-commit rev is the enforced version (Phase 30 D-05..D-07), not the ruff on PATH |
| Test execution | Django test runner (`manage.py test`) | `coverage`, CI workflows, `django-test` pre-commit hook | One runner everywhere; tags (`functional`, `ephemeris_segfault`) select what runs where |
| TOM auth / registration | tomtoolkit 3.1.0 (`tom_common`, allauth) | FOMO `settings.py` (`TOMTOOLKIT_*` defaults) | 3.1.0 owns the auth stack; FOMO only adds apps on top of `TOMTOOLKIT_INSTALLED_APPS` |
| Calendar templates/URLs | FOMO overrides (`calendar_urls.py`, 2 templates) | upstream `tom_calendar` | Upstream unchanged in 3.1.0, so the overrides still shadow the same upstream files |
| PR #43 content | `issue37-code-only` branch (code snapshot) | GitHub PR body | The PR diff is the branch; the description is separate and edited with `gh pr edit` |

## Standard Stack

### Core (what the phase installs / pins)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tomtoolkit | 3.1.0 (`>=3.1.0`) | TOM framework | `main` floor; `pip index versions tomtoolkit` lists 3.1.0 as newest [VERIFIED: pip index versions tomtoolkit, run this session] |
| tom_jpl | 0.3.0 (`>=0.3.0`) | Scout data service, `ScoutDataService`, factories | `main` floor; newest on PyPI [VERIFIED: pip index versions tom_jpl] |
| ruff | 0.16.9 via `ruff-pre-commit` rev `v0.16.9`; dev extra `ruff>=0.16` | lint + format | `main`'s pins [VERIFIED: `git show origin/main:.pre-commit-config.yaml` -> `rev: v0.16.9`; `git show origin/main:pyproject.toml` -> `"ruff>=0.16"`] |
| coverage | 7.16.2 on a fresh install | CI/hook coverage | In main's dev extras (`"coverage", # Used to report total code coverage of the Django test suite`) [VERIFIED: pip list in scratch venv] |
| Django | 5.2.18 on a fresh install | web framework | tomtoolkit 3.1.0 requires `django (>=5.2.17,<6)` [VERIFIED: wheel METADATA]; do not bump beyond that (locked) |
| django-allauth[mfa] | 65.19.7 on a fresh install | new transitive dep of tomtoolkit 3.1.0 | `Requires-Dist: django-allauth[mfa] (>=65.19.4,<66)` [VERIFIED: wheel METADATA] |

### Supporting (kept from the branch)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| timezonefinder | `>=6.0` (runtime) | site timezone lookup | Stays (D-04). Not on `main`; it is the only runtime dep the branch adds. |
| graphifyy | unpinned (dev) | knowledge-graph tooling | Stays in `[dev]` (D-04) |
| playwright | unpinned (dev) | functional tests | Already in `main`'s dev extras; Chromium was present in the scratch run |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `git merge origin/main` into `issue37-code-only` with conflict resolution | `git merge -s ours origin/main` then the snapshot commit | Same history shape (merge commit + one snapshot commit) and the same final tree, but no pointless conflict resolution on a branch whose tree is about to be replaced wholesale. Recommended. |
| Switch branches in the primary checkout for the PR refresh | `git worktree add` a second checkout | The primary checkout holds the on-disk `.planning/`. Switching to `issue37-code-only` (which never tracks `.planning/`) deletes those tracked files from disk, and switching back then fails on "untracked files would be overwritten" (reproduced). Use a worktree. |

**Installation (proves SYNC-02 in a fresh env; use `TMPDIR` on `/home` — see Environment Availability):**
```bash
python3 -m venv /home/<user>/venv_fomo_check
TMPDIR=/home/<user>/tmp /home/<user>/venv_fomo_check/bin/pip install -e '.[dev]'
/home/<user>/venv_fomo_check/bin/pip show tomtoolkit tom_jpl   # expect 3.1.0 and 0.3.0
```

**Version verification:** `pip index versions tomtoolkit` -> `tomtoolkit (3.1.0)`; `pip index versions tom_jpl` -> `tom_jpl (0.3.0)` [VERIFIED: run this session]. A fresh `pip install -e '.[dev]'` of the merged tree resolved tomtoolkit 3.1.0, tom_jpl 0.3.0, Django 5.2.18, django-allauth 65.19.7, coverage 7.16.2 and ruff **0.16.10** (not 0.16.9, because the dev floor is `>=0.16`).

## Package Legitimacy Audit

No package is *new* to this project: `tomtoolkit`, `tom_jpl`, `coverage` and `ruff` are already declared
on `main` and (for the first two) installed here. The seam was run anyway:

`gsd_run query package-legitimacy check --ecosystem pypi tomtoolkit tom_jpl coverage ruff`

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| tomtoolkit | PyPI | many releases since 2018; latest 3.1.0 published 2026-09-24 | seam: unknown | seam: none found (project is github.com/TOMToolkit/tom_base) | SUS (seam reasons: too-new, unknown-downloads, no-repository) | Approved with note: heuristic keyed on the *latest release date*, not package age. Already a `main` floor and the project's framework. |
| tom_jpl | PyPI | 0.1.0, 0.2.0, 0.3.0 (0.3.0 published 2026-09-10) | seam: unknown | seam: none found | SUS (same three reasons) | Approved with note: first-party ecosystem (TOM Toolkit org); already a `main` floor. |
| coverage | PyPI | long-established | seam: unknown | — | SUS (heuristic, same family of reasons) | Approved: already in `main`'s dev extras. |
| ruff | PyPI | long-established | seam: unknown | — | not separately inspected; same registry heuristics | Approved: already `main`'s pin. |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** tomtoolkit, tom_jpl, coverage (all three: false positives from "latest release is recent" and missing download/repo metadata in the seam's PyPI signals). No install checkpoint is needed *beyond* D-01's merge checkpoint, which already puts the `pyproject.toml` diff in front of the developer; the planner should not add a separate `checkpoint:human-verify` for these. [ASSUMED: this disposition is my judgement of the heuristic, not a seam output. Confirm if the project wants a stricter reading.]

## Architecture Patterns

### Flow of the work (system diagram)

```
git fetch origin ──► origin/main head (a910c17 at research time; re-read at merge time)
        │
        ▼
 PREFLIGHT (branch check, clean/stash .planning/state.json, untracked files OK)
        │
        ▼
 git merge --no-commit --no-ff origin/main ──► 9 conflicts + ~40 auto-merged files
        │                                          (settings.py, pre-commit, CI, CLAUDE.md, tests/ removal …)
        ▼
 resolve 9 files by table ──► CHECKPOINT: developer reads `git diff --cached` ──► merge commit (D-02)
        │
        ├─► verify SC#1 (merge-base --is-ancestor, parents)
        ├─► env: pip install -e '.[dev]'  ──► pip show tomtoolkit/tom_jpl         (SYNC-02)
        ├─► small follow-ups (own commits):
        │     • hooks/CI: add --exclude-tag ephemeris_segfault; repoint nb hook   (SYNC-05/06, D-05..07)
        │     • style: ruff 0.16.9 reformat (12 files)  /  fix: SIM103 (1 file)    (SYNC-03)
        │     • CLAUDE.md: lines listed below                                     (SYNC-03/06)
        ├─► full suite (SYNC-07) + tom_calendar override note for Phase 39
        ▼
 git push origin issue37-telescope-runs-calendar     (CI does NOT run on this push — see Open Question 1)
        │
        ▼
 PR #43 refresh in a SEPARATE WORKTREE of issue37-code-only:
   merge -s ours origin/main ─► snapshot commit (merged tree minus .planning/) ─► push (no force)
        │                                                 └─► pull_request workflows run against base main
        ▼
 gh pr edit 43 --body-file …   (still draft)
```

### Recommended plan decomposition

1. **Merge + checkpoint** (SYNC-01, SYNC-04): preflight, merge, 9 resolutions, checkpoint, commit, SC#1 checks.
2. **Environment + floors proof** (SYNC-02): re-install, `pip show`, back up and migrate the dev DB (tomtoolkit 3.1.0 adds `tom_common` 0005/0006 and `tom_dataproducts` 0019/0020 migrations).
3. **ruff 0.16.9** (SYNC-03): one `style:` commit, one `fix:` commit for SIM103; CLAUDE.md ruff lines.
4. **Hooks and CI** (SYNC-05, SYNC-06): the small edits listed below; one-time notebook metadata commit.
5. **CLAUDE.md + docs** (SYNC-03/06): Testing section addendum, stale lines, `docs/installation.rst` `timezonefinder` bullet.
6. **Suite + override comparison** (SYNC-07): full run, `tom_calendar` note.
7. **PR #43** (SYNC-08): worktree, `merge -s ours`, snapshot, push, wait for CI, `gh pr edit`.

### The Merge — facts

- Merge base `756680f`; `git log --oneline 756680f..origin/main` = 62 commits; `origin/main` = `a910c178be2e6e8063f8a262b51934ca05cdbb01` at research time [VERIFIED: git rev-parse origin/main after `git fetch origin`]. `git diff --stat 756680f origin/main` = 50 files, +3044/-266.
- `git merge-tree --write-tree HEAD origin/main` conflicts [VERIFIED: run this session]: `.gitignore`, `docs/conf.py`, `docs/design/design.rst`, `docs/notebooks.rst`, `pyproject.toml`, `solsys_code/admin.py`, `solsys_code/apps.py`, `solsys_code/tests/test_bootstrap5_rendering.py`, `src/fomo/urls.py` — exactly CONTEXT's 9. Auto-merged cleanly: `.pre-commit-config.yaml`, `CLAUDE.md`, `docs/index.rst`, `docs/installation.rst`, `solsys_code/solsys_code_observatory/views.py`, `src/fomo/settings.py`.
- Working tree today is not clean: ` M .planning/state.json` plus untracked `.planning/agent-history.json`, `reqgroup_2682493.json`, three `src/fomo_db_*.sqlite3` backups. None is touched by the merge; `state.json` should be committed or stashed first so the checkpoint's `git status` is readable. The `fomo_db_20*.sqlite3` backups become ignored once `main`'s `.gitignore` lines arrive.
- Re-run `git fetch origin` immediately before merging; if `main` moved, re-read this table (conflict set could change).

#### Resolution table (what I did in the scratch clone; the resulting tree passed the full suite)

| File | Correct resolution | Why a literal "keep both" is wrong |
|------|--------------------|------------------------------------|
| `.gitignore` | Keep both blocks (branch's `.claude/`, `.planning/graphs/`, `graphify-out/`; main's `src/_static/`). | — |
| `docs/design/design.rst` | Union the toctree: branch's seven entries, then main's `target_origin_tracking`, `scout_element_history`, then the shared `fink_sso_support`, `target_search_limitations`. All 11 target `.rst` files exist. | — |
| `docs/notebooks.rst` | Keep both: branch's "Demonstration Notebooks" toctree, then main's `Scout candidate lifecycle <notebooks/scout_lifecycle_exploration>` entry. | — |
| `pyproject.toml` | One hunk only, in `[project.optional-dependencies] dev`: take `"ruff>=0.16", …` from main, keep `"graphifyy", …` from the branch, drop `pytest`, `pytest-cov`, `ruff==0.2.1`. Everything else (floors, `timezonefinder>=6.0`, ruff `exclude` union, coverage block, removal of `[tool.pytest.ini_options]`/`[tool.black]`/`[tool.isort]`) auto-merges correctly. | Keeping the branch's `ruff==0.2.1` line would pin ruff against the new hook rev. |
| `docs/conf.py` | `autoapi_ignore = ['*/__main__.py', '*/local_settings.py']` (drop `*/_version.py`); keep main's `_skip_version_module` + `setup(app)`; keep the branch's `nbsphinx_allow_errors = True` and the CR-03 comment. | Main's commit `7621227` ("Let autoapi resolve fomo.__version__ without documenting _version") removes `*/_version.py` from the ignore list on purpose; its comment says `_version` "must stay parsed (not in autoapi_ignore)". Keeping both rules reintroduces the bug main fixed. `*/local_settings.py` must stay: it is the CR-03 secret-leak guard. |
| `solsys_code/apps.py` | Keep the branch's `ready()` unchanged. Do **not** add main's second `def nav_items`. Instead append main's entry to the existing `nav_items` list: `{'partial': f'{self.name}/partials/navbar_list.html'}` after the `campaigns_nav_link` dict. | Two `def nav_items` in one class is legal Python: the later one silently wins, so either the Campaigns link or the Rubin ToO menu vanishes with no error. (Ruff F811 would flag it, but only after the commit.) |
| `solsys_code/admin.py` | Take branch's imports/classes plus main's `TargetNameInline` and `SolsysTargetAdmin`; imports become `from tom_targets.admin import TargetAdmin` and `from tom_targets.models import Target, TargetName`. **Delete the branch's own `class TargetAdmin(admin.ModelAdmin)` and its `admin.site.unregister(Target)` / `register(Target, TargetAdmin)`**; keep a single `admin.site.unregister(Target); admin.site.register(Target, SolsysTargetAdmin)` at the end. | The branch's local class named `TargetAdmin` collides with the imported `TargetAdmin` that `SolsysTargetAdmin` subclasses, and a second `unregister(Target)` raises `NotRegistered`. Main's commit `0896881` ("Reconcile the Target admin with the earlier issue37 attempt") documents that main's version is the intended winner (it adds alias search; the branch's version has `search_fields = ['name']` only). `test_admin.TargetAdminChangelistAndTypeFilterTests` and `test_search.TestTargetAdminOverride` pass against it. |
| `solsys_code/tests/test_bootstrap5_rendering.py` | Import line `from django.test import SimpleTestCase, tag` (union). | Main's side imports only `tag`, the branch's side only `SimpleTestCase`; the file uses both (`@tag('functional')` on `TestBootstrap5Rendering`, and `SimpleTestCase` at `TestTemplatesUseBootstrap5DataAttributes`). Taking either side alone is a `NameError`/`F821`. |
| `src/fomo/urls.py` | Import block: main's `scout_views` import plus `from solsys_code.views import Ephemeris, MakeEphemerisView, ProtectedUserDeleteView`. URL list: keep main's two `scout/rubin-too/` paths **and** the branch's `calendar/`, `campaigns/`, `alerts/`, `users/<int:pk>/delete/` entries. | Taking main's `views` import line alone drops `ProtectedUserDeleteView` and breaks the branch's `user-delete` route. |

#### What the merge already does (so these are *not* separate edits)

[VERIFIED: inspected the merged tree in the scratch clone]

- `.pre-commit-config.yaml`: both `ruff-pre-commit` entries at `rev: v0.16.9`; template hooks at `v0.2.2`; `sphinx-build` and `pytest-check` hooks gone; `django-test` hook present; the branch's `exclude: ^docs/notebooks/pre_executed` on `jupyter-nb-clear-output` survives. The only differences from main's file are that exclude line.
- `.github/workflows/*`: byte-for-byte main's (the branch never edited them). `testing-and-coverage.yml` runs `coverage run manage.py test --exclude-tag functional`, `coverage xml`, plus a `functional-tests` job.
- `pyproject.toml`: `tomtoolkit>=3.1.0`, `tom_jpl>=0.3.0`, no `tom-registration`, `timezonefinder>=6.0` kept, no pytest/black/isort tool blocks, ruff `exclude` is the union D-10 asks for, `[tool.coverage.run]` is main's block.
- `tests/` is deleted (`tests/fomo/conftest.py -> requirements.txt` rename, `tests/fomo/test_packaging.py` deleted; main added `solsys_code/tests/test_packaging.py`).
- `src/fomo/settings.py`: now `INSTALLED_APPS = TOMTOOLKIT_INSTALLED_APPS + [...]` and `MIDDLEWARE = (TOMTOOLKIT_MIDDLEWARE + [...])`; **`tom_registration` is already gone from `INSTALLED_APPS` and `MIDDLEWARE`** (the apps and middleware now come from `tom_common.default_settings`, which in 3.1.0 adds allauth). Only a commented-out `# TOM_REGISTRATION = {...}` block (main's) remains. So D-04's "edit settings.py in a follow-up commit" is a no-op: the executor should *verify* with `git grep -n tom_registration -- src/fomo/settings.py` and not invent an edit. `DATE_FORMAT`/`DATETIME_FORMAT`/`USE_L10N` lines disappear but `FORMAT_MODULE_PATH = ['tom_base.formats']` supplies the identical `'Y-m-d H:i:s'` / `'Y-m-d'` [VERIFIED: tom_base/formats/en/formats.py in the 3.1.0 wheel].
- `docs/installation.rst`: `tom-registration` bullet gone; `tomtoolkit>=3.1.0`, `tom_jpl>=0.3.0` listed. It does **not** list `timezonefinder>=6.0`; main's PR checklist says dependency changes must be mirrored there, so add that bullet.
- CLAUDE.md "Testing" section and the pre-commit Conventions bullet auto-merge to main's wording; main's new bullet about copier updates also arrives.

#### What genuinely remains after the merge

| Item | Where | Edit |
|------|-------|------|
| `django-test` hook entry | `.pre-commit-config.yaml` | change entry to `bash -c "coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault && coverage html"`; extend the hook's comment with the `SKIP=django-test git commit` note (D-06) |
| notebook hook repoint | `.pre-commit-config.yaml` | `files: ^docs/pre_executed/.*\.ipynb$` -> `files: ^docs/notebooks/pre_executed/.*\.ipynb$`, `["docs/pre_executed/"]` -> `["docs/notebooks/pre_executed/"]` (D-07). **First run rewrites the 8 pre-executed notebooks** (see Pitfall 5). |
| CI unit-test step | `.github/workflows/testing-and-coverage.yml` | add `--exclude-tag ephemeris_segfault` to the `coverage run` line (D-05) |
| CI smoke test | `.github/workflows/smoke-test.yml` | its step is `python manage.py test --exclude-tag functional`; add `--exclude-tag ephemeris_segfault` too, or the daily job hits the ASSIST segfault. (D-05 names only the unit-test matrix; flag to developer, it is the same one-token edit.) |
| CLAUDE.md line 36 | Commands comment | `(v0.2.1)` -> `(v0.16.9)`; keep the `pre-commit run ruff ...` lines; main's bare `ruff check . --fix`/`ruff format .` already did **not** survive the auto-merge (the branch's Phase 30 text won), so D-08's "drop main's lines" needs only a check |
| CLAUDE.md Commands, Tests | Commands block | auto-merged main text lacks the branch's `--exclude-tag=ephemeris_segfault` convention; add it to the test lines and to the Testing section (discretion item) |
| CLAUDE.md "Key Dependencies" | ~lines 263-267, 279 | remove `pytest` / `pytest-cov` bullets, `ruff 0.2.1+` -> `ruff 0.16.9`, remove the `tom_registration` bullet |
| CLAUDE.md "Configuration" | ~line 303 | `.pre-commit-config.yaml` description still says `(ruff, pytest, Sphinx, validation)` |
| `docs/conf.py` | comment above `suppress_warnings` | refers to the removed `sphinx-build` hook; harmless, optional tidy-up |
| ruff | 12 files | one `style:` commit; one `fix:` commit (below) |
| `.planning/codebase/INTEGRATIONS.md`, `STACK.md` | mention `tom-registration` | out of scope (generated mapping docs); leave |

### Pattern: checkpoint contents for D-01

At the checkpoint show (a) `git status --short`, (b) `git diff --cached --stat`, and (c) the **resolved
diff of the 9 files** (`git diff --cached -- <9 paths>`) — CONTEXT's "Specific Ideas" asks for the diff,
not a summary. Also show `git grep -n "def nav_items" solsys_code/apps.py` (must print exactly one hit) and
`python manage.py check` output (expect only the known benign `urls.W005` for namespace `calendar`).

### Anti-Patterns to Avoid

- **Literal keep-both on `apps.py`/`admin.py`/`docs/conf.py`:** produces duplicate methods, a name collision and a double `unregister` — see the table.
- **Resolving conflicts by running `ruff --fix` over the tree before the checkpoint:** violates D-02 (the merge commit would contain a reformat).
- **`git rebase`, `git push --force`, `git push origin <branch>` followed by branch-implicit git commands without `git branch --show-current`** (locked constraint + CLAUDE.md).
- **Switching the primary checkout to `issue37-code-only`:** deletes `.planning/` from disk.
- **Per-commit full test runs:** the `django-test` hook runs the whole suite (measured **613 s** for 2178 tests); every executor task commit needs `SKIP=django-test` or each commit takes ten minutes. D-06's note covers this; the plans must say so explicitly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Making the PR branch hold "merged tree minus `.planning/`" | A cherry-pick loop over 2331 commits, or hand-copied files | `git merge -s ours origin/main`, then `git read-tree -u --reset <merged-branch>` and `git rm -r --cached .planning` in a worktree, one commit | Tested: result is a merge commit (parents: old code-only tip, `origin/main`) plus one snapshot commit; `git diff origin/main...HEAD --shortstat` = 174 files, +88416/-60. (`/gsd-pr-branch` cherry-picks everything and does not refresh `issue37-code-only`, per the discussion log.) |
| Proving lint/format state | Running a bare `ruff` | `pre-commit run ruff --all-files`, `pre-commit run ruff-format --all-files` | The hook rev is the enforced version; a fresh install's ruff is 0.16.10, the hook's is 0.16.9 |
| PR body editing | Editing in the browser | `gh pr edit 43 --body-file <file>` | Reproducible, reviewable text; PR stays a draft (don't pass `--ready`) |
| Template/hook version check | Manual comparison | main's `check-lincc-frameworks-template-version` hook | Already in the merged config |

**Key insight:** the merge, not the executor, carries most of the "adopt main's tooling" work. Plans that
re-implement those edits by hand will produce empty commits or, worse, diverge from `main`'s files.

## Runtime State Inventory

This is a sync/migration-style phase, so each category is answered explicitly.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `src/fomo_db.sqlite3` (the dev database, ignored by git) needs tomtoolkit 3.1.0's new migrations: `tom_common` 0005 (`profile_password_changed_at_profile_phone_number`), 0006 (`termsofserviceacceptance`), `tom_dataproducts` 0019, 0020 [VERIFIED: wheel file listing]. Untracked backups `src/fomo_db_20260929.sqlite3`, `_20261002_pre_l04`, `_20261002_pre_step5` already exist. Tests use a throwaway DB, so the suite does not touch this. | Data migration (run `python manage.py migrate`) on the dev DB **after copying it to a dated backup**; not part of the merge. `makemigrations --check` reports "No changes detected" on the merged tree [VERIFIED]. |
| Live service config | GitHub PR #43 body and head branch (`issue37-code-only`) live on GitHub, not in git; GitHub Actions run config lives in workflows (arrives via merge). | Edit via `gh pr edit`, push via git (SYNC-08). |
| OS-registered state | The developer's pre-commit hooks installed into `.git/hooks` (`pre-commit install`) call hook ids from `.pre-commit-config.yaml`; the old `pytest-check` id is removed by the merge. Cron / systemd for `run_unattended` use `deploy/cron/fomo.crontab.example` templates only — nothing registered by this phase. | After the merge run `pre-commit install` again only if the hook script is missing; hook ids are read from the config at run time, so no re-registration is required. |
| Secrets / env vars | No secret names change. `local_settings.py` (the documented home of credentials) is untouched; `docs/conf.py`'s `*/local_settings.py` autoapi exclusion must survive the conflict (see table). | None, except do not lose that exclusion. |
| Build artifacts / installed packages | The dev venv has `tom-registration 2.0.1` installed and `tom_jpl 0.1.0.post6.dev0+818ef75` as an **editable install from `/home/tlister/git/tom_jpl`** [VERIFIED: pip list]; `tomtoolkit 3.0.1`, `ruff 0.2.1`, `pytest 9.1.1`, `pytest-cov 7.1.0`. | After the merge: `pip install -e '.[dev]'` will replace the editable `tom_jpl` with PyPI 0.3.0 (the local clone on disk is not modified); `pip uninstall tom-registration` (it is not pulled out automatically and conflicts conceptually with 3.1.0's allauth registration); `pip uninstall pytest pytest-cov` is optional. `.pytest_cache/`, `htmlcov/`, `.coverage` already gitignored. |

## Common Pitfalls

### Pitfall 1: Silent shadowing in `apps.py`
**What goes wrong:** Two `nav_items` methods in `SolsysCodeConfig`; the Rubin ToO menu or the Campaigns link disappears, tests may not notice.
**Why it happens:** D-03 says "keep both sides" for "navbar items"; main's hunk adds a *new method* at the position where the branch has `ready()`, not a new list entry.
**How to avoid:** One `nav_items` returning a two-element list (table above). Checkpoint item: `git grep -c "def nav_items" solsys_code/apps.py` must print 1.
**Warning signs:** ruff F811; navbar missing "Rubin ToO" or "Campaigns".

### Pitfall 2: Admin registration collision
**What goes wrong:** Import-time failure (`NotRegistered`) or the branch's weaker admin wins.
**How to avoid:** Single `SolsysTargetAdmin` registration (table).
**Warning signs:** `python manage.py check` raising on `admin.site.unregister(Target)`.

### Pitfall 3: A second runner of the segfaulting class
**What goes wrong:** CI/hook/smoke-test process dies with a segfault in native ASSIST; no report.
**Why it happens:** `main` never had `@tag('ephemeris_segfault')`; its workflows only exclude `functional`. The class is `@tag('ephemeris_segfault')` at `solsys_code/tests/test_views.py:98` [VERIFIED: Read this session: `@tag('ephemeris_segfault')  # Phase 37 Plan 07: the native ASSIST integrator crashes the whole`].
**How to avoid:** Three edits: unit-test matrix (D-05), `django-test` hook (D-06), **and the daily `smoke-test.yml`** (not named in the decisions).

### Pitfall 4: Fresh-install ruff is not the hook's ruff
**What goes wrong:** `pip install -e .[dev]` yields ruff 0.16.10 while the hook pins 0.16.9; a bare `ruff` can report different findings. This is exactly the Phase 30 drift problem.
**How to avoid:** Keep D-08: document and use `pre-commit run ruff ...` only. In the plan's verification use `pre-commit run`, not `ruff check`.

### Pitfall 5: The repointed notebook hook edits the notebooks
**What goes wrong:** The first `pre-commit run pre-executed-nb-never-execute --all-files` after repointing prints `Modified notebook to set nbsphinx.execute='never'` for **all 8** notebooks under `docs/notebooks/pre_executed/` and fails the commit; on the second run it passes [VERIFIED: ran in scratch clone; rc=0 on the second run]. The edit adds `"nbsphinx": {"execute": "never"}` to each notebook's metadata and drops the trailing newline. It also prints a non-fatal `Expecting value: line 1 column 1 (char 0)` (the hook walks the whole directory including the `.json` baseline and `fixtures/`).
**Why it matters:** the eight notebooks are in CLAUDE.md's paired-docs map. This is a *metadata-only* tooling change, not a behavior change, so no re-execution is required, but it must be committed deliberately as its own commit ("chore: mark pre-executed notebooks nbsphinx.execute=never"), not smuggled into another commit by a failing hook.

### Pitfall 6: `ruff-format` also rewrites notebooks
Pre-commit's `ruff-format` hook covers `jupyter`, so the 12-file reformat includes 3 pre-executed notebooks (`campaign_lifecycle_demo`, `project_observation_calendar_demo`, `reconcile_campaign_runs_demo`). Pure formatting of source cells; outputs are untouched. Not a behavior change, so the paired-docs rule is not triggered (scope note in the roadmap says so).

### Pitfall 7: Dirty primary checkout and PR branch work
See "Anti-Patterns". Use `git worktree add ../fomo_code_only issue37-code-only` for SYNC-08; remove it afterwards (`git worktree remove`).

### Pitfall 8: CLAUDE.md and `.planning/` interplay on the snapshot
The snapshot tree is the merged tree minus `.planning/`; it will contain `CLAUDE.md` (which has GSD-workflow text) and `.gitattributes`, exactly as the 2026-09-01 and 2026-10-06 snapshots did. Nothing extra to filter; `.claude/` is gitignored.

## Code Examples

### Preflight and merge (executor)

```bash
git branch --show-current            # must print issue37-telescope-runs-calendar
git fetch origin
git status --short                   # commit or stash .planning/state.json first
git merge --no-commit --no-ff origin/main     # expect 9 conflicts
git diff --name-only --diff-filter=U           # must list exactly the 9 files
# ... resolve per the table, then:
git diff --cached                    # CHECKPOINT for the developer
git commit -m "Merge origin/main into issue37-telescope-runs-calendar"
git merge-base --is-ancestor origin/main HEAD && echo ok
git rev-parse HEAD^1 HEAD^2          # branch head before merge, origin/main head
```

### Hook entries after D-06 / D-07 (values quoted from `origin/main:.pre-commit-config.yaml`, changed only where D-06/D-07 say)

```yaml
  - repo: https://github.com/lincc-frameworks/pre-commit-hooks
    rev: v0.2.2
    hooks:
      - id: pre-executed-nb-never-execute
        name: Check pre-executed notebooks
        files: ^docs/notebooks/pre_executed/.*\.ipynb$
        verbose: true
        args:
          ["docs/notebooks/pre_executed/"]
  - repo: local
    hooks:
      - id: django-test
        name: Run Django unit tests (excluding functional)
        description: Run the Django test suite with coverage, excluding tests tagged 'functional'.
        # Takes ~10 minutes. For work-in-progress commits: SKIP=django-test git commit ...
        entry: bash -c "coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault && coverage html"
        language: system
        pass_filenames: false
        always_run: true
```

### The one ruff lint finding (verify the code, then apply)

`ruff` 0.16.9 on `solsys_code/management/commands/backfill_lco_observations.py:176` reports `SIM103 Return the negated condition directly`. The function ends:

```python
    if created_before is not None and created > created_before:
        return False
    return True
```

The behavior-preserving rewrite is `return not (created_before is not None and created > created_before)`. ruff marks it a hidden (unsafe) fix, so apply by hand. The file is in CLAUDE.md's notebook map (`backfill_lco_observations_demo.ipynb`), but a boolean simplification is a pure refactor, not a behavior change, so the paired-notebook rule does not trigger. Flag it to the developer anyway per D-09.

### Refresh PR #43's head branch (separate worktree, after all branch work is pushed)

```bash
git branch --show-current                                   # primary checkout: still the v2.5 branch
git fetch origin
git worktree add ../fomo_code_only issue37-code-only        # local branch is at 372d02c (see Open Question 2)
cd ../fomo_code_only && git branch --show-current            # issue37-code-only
git merge -s ours origin/main -m "Merge origin/main into issue37-code-only (tree superseded by the snapshot that follows)"
git read-tree -u --reset issue37-telescope-runs-calendar     # whole merged tree
git rm -r -q --cached .planning ; rm -rf .planning           # snapshot excludes .planning/
git commit -m "Sync code-only branch with issue37-telescope-runs-calendar through v2.5 Phase 38"
git diff origin/main...HEAD --shortstat                      # tested: 174 files changed in the dry run
git push origin issue37-code-only                            # normal push, never --force
```

Then `gh pr edit 43 --body-file <file>`; confirm `gh pr view 43 --json isDraft,headRefName` shows `isDraft: true`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `tom_registration` app + `ModelBackend` | tomtoolkit 3.1.0 built-in auth on django-allauth (`allauth`, `allauth.account`, `allauth.mfa` in `TOMTOOLKIT_INSTALLED_APPS`; `ModelBackend` deliberately removed from `TOMTOOLKIT_AUTHENTICATION_BACKENDS`) | tomtoolkit 3.1.0, 2026-09-24 | FOMO settings now inherit these; tests that log in via `force_login` still pass (suite OK) |
| LINCC template v2.1.0 (pytest-based CI/hooks) | v2.2.0 customised by `main` for the Django runner | `main`, PR #57 | `.copier-answers.yml` `_commit: v2.2.0`, `custom_install: custom`; per main's CLAUDE.md, answer `custom`, never `retrofit` on later `copier update` |
| ruff 0.2.1 | ruff 0.16.9 (hook) | `main` | 1 lint finding, 12 files reformat on this tree (f-string quote style, assert wrapping) |

**Deprecated/outdated:** `tom_registration` (deprecated by 3.1.0 [CITED: github.com/TOMToolkit/tom_base/releases/tag/3.1.0]); `tom_alerts` broker settings `BROKERS`/`TOM_ALERT_CLASSES` (removed from main's settings; auto-merge removes them from the branch's too).

## tomtoolkit 3.0.1 -> 3.1.0 and the `tom_calendar` overrides

[VERIFIED: downloaded both wheels with `pip download --no-deps` into separate directories and ran `diff -r`]

- **Identical between 3.0.1 and 3.1.0:** `tom_calendar`, `tom_targets`, `tom_observations`, `tom_dataservices`.
- **Changed:** `tom_common` (allauth accounts, MFA, middleware, `urls.py`, `views.py`, `default_settings.py`, templates under `account/`, `allauth/`, `mfa/`; migrations 0005, 0006), `tom_dataproducts` (models + migrations 0019, 0020), `tom_base/settings.py`, `tom_setup` settings template.
- **Consequence for the three overrides** (`solsys_code/calendar_urls.py`, `src/templates/tom_calendar/partials/calendar.html`, `.../event_form.html`): their upstream counterparts did not change, so **no upstream change is newly hidden**; there is no SYNC-07 fix from this comparison. Phase 39 starts from the same upstream files as before (upstream has `tom_calendar/templates/tom_calendar/{calendar_page.html, partials/calendar.html, partials/event_form.html, partials/navbar_item.html, partials/target_list_block.html, partials/todos.html}`). Record this in a short `38-OVERRIDE-COMPARISON.md` in the phase directory.
- **Other FOMO overrides checked:** `src/templates/tom_common/index.html` (upstream `index.html` is not in the changed list) and `src/templates/tom_targets/partials/module_buttons.html` (`tom_targets` unchanged). `tom_common/templates/tom_common/base.html` and `navbar_content.html` did change upstream but FOMO does not override them.
- **FOMO `ProtectedUserDeleteView`** subclasses TOM's `UserDeleteView`; `tom_common/views.py` changed only in token-regeneration and password-change code, and the user-delete tests pass.
- **Observed warning to keep:** `urls.W005` namespace `calendar` not unique (the branch's `calendar/` include shadows tom_common's); it is expected and documented in `src/fomo/urls.py`.

## Forecast for SYNC-07 (measured, not guessed)

Scratch clone of the branch + merge resolved per the table; fresh venv, `pip install -e '.[dev]'`; `TMPDIR` on `/home`:

```
python manage.py test solsys_code --exclude-tag=ephemeris_segfault
Ran 2178 tests in 613.470s
OK
```

- Zero failures, zero errors, **no `(skipped=N)` suffix** — the run did not skip or tag out anything. It included the Playwright tests (Chromium installed). 2178 = 2095 (v2.4 baseline) + 83 from `main`'s new test modules.
- `python manage.py check` -> only the benign `urls.W005`; `makemigrations --check --dry-run` -> "No changes detected".
- Not measured: the CI form `--exclude-tag functional --exclude-tag ephemeris_segfault` (a subset of the above) and `--tag functional` as a separate job; Python 3.10/3.12 (CI matrix); the known-flaky `test_observatory_create_form_submits_to_observatory_url` happened to pass.
- The real run in the executor's environment may still differ (the executor's venv is the existing dev venv, upgraded in place). If anything fails there, first reproduce with the fresh-venv recipe to separate environment drift from a real regression.

## Common Operational Facts for the Plans

### PR #43 (state read from GitHub this session)

- Title "Telescope runs calendar (issue #37)", `isDraft: true`, base `main`, head `issue37-code-only`; 4 commits on GitHub (`3968d17`, `4410c32`, `252c2b0`, `5a1f27e`); `gh` is authenticated.
- Current body: the old generic PR-template checklist plus a paragraph about replacing closed PR #41 and a pointer to `docs/design/telescope_runs_calendar.rst`. `main` replaced the PR template (`cd10a2e`) with a FOMO-specific one (sections: Change Description, Solution Description, Code Quality, FOMO-Specific Checklist). The rewritten body should keep those headings so it matches the template (D-12 content goes under Change/Solution Description).
- Source material for the four v2.4 pillars is in `.planning/PROJECT.md` (the "v2.4" paragraph and the "Validated" block): observation projector (`observation_projector.py`, `project_observation_calendar`), allocation layer (`allocation_projector.py`, `cutover_classical_allocations`, `ProposalTimeAllocation`), unattended operation (`run_unattended` / `check_unattended`, `WatchedProposal`, `deploy/cron`, `deploy/logrotate`, email + heartbeat), public tallies (`campaign_tally.py`, `status_vocabulary.py`). Runbook: `docs/runbooks/telescope_runs_calendar.rst` (anchor `.. _unattended-operation:` at line 1845, section "How do I run everything unattended?").
- "How to try it" commands: `python manage.py migrate`, `python manage.py load_telescope_runs <file> --dry-run` (flags `filepath`, `--campaign`, `--dry-run` [VERIFIED: add_arguments in `load_telescope_runs.py`]), `python manage.py run_unattended --dry-run` (flags `--dry-run`, `--step` [VERIFIED: `run_unattended.py` add_arguments]).

### Reusable assets
`.setup_dev.sh` (editable install + `[dev]` + `docs/requirements.txt` + `pre-commit install`; it prompts if no venv is active), `solsys_code/tests/test_views.py:98` tag, `.github/workflows/testing-and-coverage.yml` from main (taken as-is plus the tag), the 2026-09-01 snapshot commit `5a1f27e` and the unpushed 2026-10-06 snapshot `372d02c` as the model for D-11. No Makefile.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The package-legitimacy SUS verdicts are heuristic false positives (all four packages are existing `main` dependencies) and need no separate checkpoint | Package Legitimacy Audit | Low; D-01's checkpoint already shows the pyproject diff |
| A2 | Ruff 0.16.10 (fresh-install) and 0.16.9 (hook) give the same findings on this tree; only 0.16.9 was measured | Pitfall 4 | Low; verification uses `pre-commit run`, which is 0.16.9 |
| A3 | `git merge -s ours` + snapshot is acceptable as the implementation of D-11's "git merge origin/main into it, then one snapshot commit" | Don't Hand-Roll | Medium; developer may have expected a real conflict-resolved merge. The end state (parents and tree) is the same; confirm at plan review |
| A4 | CI on GitHub will pass on the pushed `issue37-code-only` snapshot (not run; only local equivalents were) | Operational Facts | Medium; the Docs build and Python 3.10/3.12 jobs were not exercised locally |

## Open Questions (RESOLVED)

1. **How does success criterion 4 ("A push to the branch runs the Django test runner with coverage in CI") become true?**
   - What we know: every workflow in `main` (and so on the merged branch) triggers only on `push` to `main` and `pull_request` to `main`. `gh run list --branch issue37-telescope-runs-calendar` last shows runs on 2026-07-16 (when a PR from that branch existed). `issue37-telescope-runs-calendar` has no open PR; PR #43's head is `issue37-code-only`.
   - What's unclear: whether the criterion means "the workflow *files* run the Django runner" (true after the merge + D-05 edit) or literally "pushing this branch triggers CI" (false without a trigger change).
   - Recommendation: treat the PR #43 refresh push to `issue37-code-only` as the CI proof (it triggers `Unit test and code coverage`, `Run pre-commit hooks`, `Build documentation` via `pull_request`); do **not** edit `on:` triggers (that would diverge from main's template-managed files). Ask the developer to confirm this reading of SC#4 at plan review.
   - RESOLVED: 38-04 adopts the recommendation as its stated assumption OQ1 -- Task 3 proves SC#4's CI clause from the `pull_request` runs on the pushed `issue37-code-only` snapshot (the `Run Django unit tests with coverage` step, the `functional-tests` job, no pytest job), and the developer confirms that reading at 38-04's Task 2 blocking-human checkpoint before anything is pushed. 38-02's prohibition keeps every `on:` trigger exactly as main has it.

2. **D-11's premise about `issue37-code-only` is out of date.**
   - What we know: the remote branch is at `5a1f27e` (4 commits) as CONTEXT says, but the **local** `issue37-code-only` is at `372d02c` "Sync code-only branch with issue37-telescope-runs-calendar through v2.4" (2026-10-06 21:29, one unpushed commit; its tree is **identical** to the current branch HEAD outside `.planning/`: `git diff --quiet 372d02c HEAD -- . ':!.planning'` succeeds [VERIFIED]). So it is already a valid v2.4 snapshot, just not yet merged with `main` and not pushed.
   - What's unclear: whether to push 372d02c as part of the refresh (it is just a snapshot commit; the new one supersedes it) or reset it first.
   - Recommendation: keep it (no history rewrite; D-11 forbids force-push). The merge + new snapshot go on top, so a normal push publishes three new commits (`372d02c`, the merge, the new snapshot) onto the PR's four, not "four existing + two". Update the plan text accordingly and tell the developer.
   - RESOLVED: 38-04 adopts the recommendation as its stated assumption OQ2 -- Task 1 keeps `372d02c` and builds the merge and the new snapshot on top of it; Task 2's checkpoint shows the developer the three commits to be published, and its revise option offers rebuilding without `372d02c` (it is unpushed, so resetting the local branch to origin's tip rewrites nothing published).

3. **`use_worktrees: true` in `.planning/config.json` vs D-01 "worktree parallelism is off."**
   - What we know: `workflow.use_worktrees` is `true`; `parallelization` is `true`. Memory notes say worktrees degrade to sequential on feature branches.
   - Recommendation: the merge plan must run in the primary checkout in a non-worktree wave (single plan, wave 1, no parallel plans with it); state this in the plan's frontmatter/objective.
   - RESOLVED: 38-01's execution constraints require the primary checkout on `issue37-telescope-runs-calendar` (the developer sets `workflow.use_worktrees` to false for this phase before `/gsd-execute-phase 38`), 38-01 is alone in wave 1, and its Task 1 precondition halts the executor if `git rev-parse --git-dir` and `--git-common-dir` differ (a worktree). 38-02, 38-03 and 38-04 repeat the primary-checkout constraint; 38-04 does its code-only work in a separate `git worktree` only for `issue37-code-only`.

4. **Is the 10-minute `django-test` hook acceptable for executor commits?** Recommendation: every plan task commit runs with `SKIP=django-test`; one explicit suite run in the SYNC-07 plan instead.
   - RESOLVED: every commit in 38-01 through 38-04 uses `SKIP=django-test` (38-01's merge commit also skips `ruff` and `ruff-format`, per D-02). The suite runs explicitly instead: 38-01 Task 3 on the merge commit, and 38-03 Task 2 (SYNC-07) both as a full fresh-venv run and as `pre-commit run django-test --all-files`, the CI form.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| git | merge, worktree | yes | (present) | — |
| gh CLI | PR #43 view/edit | yes (authenticated; `gh pr view 43` worked) | — | browser |
| Python | everything | yes | 3.11.13 (venv `/home/tlister/venv/devel_fomo311_venv`) | — |
| pre-commit | SYNC-03/06 | yes | in dev venv; hooks for ruff v0.16.9, lincc hooks v0.2.2 cloned fine (network OK) | — |
| Node for gsd tooling | `gsd-tools.cjs` | default `node` is v14.21.3 | need 22 | `source ~/.nvm/nvm.sh && nvm use 22` (project memory; confirmed v22.23.2 works) |
| Playwright Chromium | functional tests | yes (Playwright tests ran and passed) | — | — |
| SPICE kernels (`~/.cache/sorcha`) | importing `ephem_utils` | yes (suite ran without a download) | — | would download ~1.6 GB |
| Network / PyPI | `pip install`, pre-commit repos, wheels | yes | — | — |
| Free disk on `/` | pip/pre-commit temp files | **only ~489 MB free (100% used)**; `/home` has ~519 GB free | — | `export TMPDIR=/home/<user>/tmp` before pip installs and long runs. A first venv in the scratchpad under `/tmp` failed with `OSError: [Errno 28] No space left on device`. |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** Node 22 (nvm), disk-space workaround (TMPDIR on `/home`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django test runner (`python manage.py test`), unittest-style `django.test.TestCase`; `coverage` for reports |
| Config file | none beyond `[tool.coverage.run]` in `pyproject.toml` (main's block); no pytest config |
| Quick run command | `python manage.py test solsys_code.tests.test_packaging solsys_code.tests.test_search solsys_code.tests.test_admin` |
| Full suite command | `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` (≈10 min; 2178 tests) |
| CI-form command | `coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault && coverage xml` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SYNC-01 | Merge ancestry and parents | git check | `git merge-base --is-ancestor origin/main HEAD && git rev-list --merges -1 HEAD --parents` | n/a (git) |
| SYNC-02 | Floors and installed versions | command | `pip show tomtoolkit tom_jpl` (3.1.0+/0.3.0+); `git diff origin/main -- pyproject.toml` shows only branch-additions (timezonefinder, graphifyy) | n/a |
| SYNC-03 | ruff 0.16.9 clean | hook | `pre-commit run ruff --all-files && pre-commit run ruff-format --all-files`; `git grep -n "0\.2\.1" -- CLAUDE.md pyproject.toml .pre-commit-config.yaml` returns nothing | n/a |
| SYNC-04 | Template files present, branch files kept | git check | `git diff --name-status <pre-merge-head> HEAD` shows no `D` for a branch-only file except `tests/fomo/*`; `grep _commit .copier-answers.yml` = `v2.2.0` | n/a |
| SYNC-05 | CI runs Django runner with coverage | grep | `git grep -n "pytest" -- .github` returns nothing; `git grep -n "exclude-tag ephemeris_segfault" -- .github .pre-commit-config.yaml` hits unit-test, smoke-test and hook | n/a |
| SYNC-06 | Dead pytest config gone | grep | `git grep -n -i "pytest" -- pyproject.toml .pre-commit-config.yaml .github CLAUDE.md` shows only the intentional "do not reintroduce" prose; `test ! -d tests` | n/a |
| SYNC-07 | Full suite green on 3.1.0 | full suite | command above; assert no `skipped=` in the summary | yes (existing) |
| SYNC-08 | PR still draft, body updated | gh | `gh pr view 43 --json isDraft,body` (`isDraft` true; body contains `docs/runbooks/telescope_runs_calendar.rst`) | n/a |

### Sampling Rate
- **Per task commit:** `SKIP=django-test git commit`; run `pre-commit run ruff --all-files` after any Python edit; quick tests above where code changed.
- **Per wave merge:** full suite once (after the merge commit, and again after the last code-touching follow-up).
- **Phase gate:** full suite green plus the SYNC-01..08 checks above before `/gsd-verify-work`.

### Wave 0 Gaps
None — existing test infrastructure covers all phase requirements. (No new test files are needed; if a follow-up changes `apps.py` navbar wiring, `solsys_code/tests/test_scout_views.py` and the navbar assertions already exist from main.)

## Security Domain

`security_enforcement` is on (ASVS level 1, block on high), so applicable items:

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes (inherited) | tomtoolkit 3.1.0 allauth (`TOMTOOLKIT_AUTHENTICATION_BACKENDS`); FOMO adds nothing. Do not re-add `ModelBackend` or `tom_registration`. |
| V3 Session Management | yes (inherited) | Django sessions via `TOMTOOLKIT_MIDDLEWARE`; unchanged by FOMO |
| V4 Access Control | yes | `ProtectedUserDeleteView` route must survive the `urls.py` merge; `AuthStrategyMiddleware` etc. come from TOM |
| V5 Input Validation | not new | no new input surface in this phase |
| V6 Cryptography | no | none |
| V14 Config / supply chain | yes | `docs/conf.py` `*/local_settings.py` exclusion (CR-03 secret-leak guard) must be kept; GitHub Actions use pinned major versions already in main's files; no secrets added to workflows |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Credentials leaking into generated API docs | Information disclosure | keep `autoapi_ignore` entry `*/local_settings.py` in the merged `docs/conf.py` |
| Loss of a custom delete-view guard through a merge | Elevation of privilege/DoS (500 on protected FK) | keep `ProtectedUserDeleteView` import and URL (resolution table) |
| Dropped user-registration backends changing who can sign up | Spoofing | `TOM_REGISTRATION_STRATEGY = 'open'` arrives from main's settings; note the behavior is now tomtoolkit's, flag to developer rather than change it |
| Pushing `.planning/` or local backups to a public PR branch | Information disclosure | snapshot excludes `.planning/`; `fomo_db_20*.sqlite3` are gitignored by main's `.gitignore` lines; check `git ls-files | grep -E "sqlite|local_settings"` on the snapshot before pushing |

## Project Constraints (from CLAUDE.md)

- Invoke Django as `python manage.py ...`, never `./manage.py`. Full-suite command: `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`. The Django runner is the only test runner; add no pytest tests; pytest/`tests/` are legacy (to be removed by this phase).
- Lint/format through `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` (hook-pinned version); single quotes, 120 columns.
- Fixtures for `Target` use `NonSiderealTargetFactory`, never `SiderealTargetFactory` (this phase should not need new fixtures).
- Verify the branch with `git branch --show-current` before any branch-implicit git command (`merge`, `reset`, `rebase`, `cherry-pick`, `commit --amend`), especially after pushing a different branch.
- Prefer merge over rebase (also locked at milestone start).
- Paired-docs rule: a behavior change in a mapped module requires its notebook/runbook in `files_modified`; a pure reformat or a boolean simplification does not. The `backfill_lco_observations.py` SIM103 fix and the reformat of `load_telescope_runs.py`, `allocation_projector.py`, `reconcile_campaign_runs.py` are in that category. The `pre-executed-nb-never-execute` metadata edit touches notebooks but only tooling metadata; keep it a separate commit.
- Planning-doc wording: plain English ("create or update", not "upsert"; no jargon like "shape").
- Pre-commit blocks commits to `main`; GSD workflow: `use_worktrees` is true in config, D-01 requires the merge not run in a disposable worktree.
- GSD tooling needs Node 22 (`nvm use 22`).

## Sources

### Primary (HIGH confidence)
- Local git, run this session: `git fetch origin`, `git log 756680f..origin/main`, `git diff --stat`, `git merge-tree --write-tree`, `git merge --no-commit --no-ff` in a scratch clone, `git show origin/main:<file>` for `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/*.yml`, `CLAUDE.md`, `solsys_code/tests/test_packaging.py`, `.github/pull_request_template.md`.
- tomtoolkit 3.0.1 and 3.1.0 wheels (`pip download --no-deps`), unzipped and diffed (`diff -r`); `METADATA` `Requires-Dist`; `tom_common/default_settings.py`, `urls.py`, `middleware.py`, `views.py`.
- Fresh-venv `pip install -e '.[dev]'` of the merged tree; `pip list`; `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` (2178 tests OK); `manage.py check`; `makemigrations --check`.
- `pre-commit run ruff | ruff-format | validate-pyproject | check-github-workflows | pre-executed-nb-never-execute --all-files` in the scratch clone (real hook revs).
- `gh pr view 43`, `gh run list` (read-only).
- `solsys_code/tests/test_views.py:98-103` (Read this session).

### Secondary (MEDIUM confidence)
- GitHub release notes for tomtoolkit 3.1.0 (WebFetch): authentication/registration overhaul, MFA, `tom_registration` deprecated, no `tom_calendar`/`tom_targets`/`tom_observations` changes noted [CITED: https://github.com/TOMToolkit/tom_base/releases/tag/3.1.0] — consistent with the wheel diff.
- `pip index versions tomtoolkit` / `tom_jpl` (PyPI listing).

### Tertiary (LOW confidence)
- Package-legitimacy seam verdicts (heuristic signals only; see audit).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions read from PyPI and a fresh install.
- Architecture / merge resolution: HIGH — reproduced end to end with the full suite green; the one judgement call is Open Question 1/2.
- Pitfalls: HIGH for 1-5 and 7 (each reproduced or read from code); MEDIUM for CI behavior on GitHub (not run remotely).

**Research date:** 2026-10-07
**Valid until:** 2026-10-14 (the merge facts depend on `origin/main` not moving; re-run `git fetch origin && git rev-parse origin/main` and compare with `a910c178be2e6e8063f8a262b51934ca05cdbb01` before relying on the conflict table)
