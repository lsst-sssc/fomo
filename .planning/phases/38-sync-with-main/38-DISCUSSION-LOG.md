# Phase 38: Sync with main - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-07
**Phase:** 38-Sync with main
**Areas discussed:** Who does the merge and the conflict policy, The test command in CI and pre-commit, CLAUDE.md and ruff after the merge, PR #43: refresh its branch or body only

---

## Todo cross-reference (before gray areas)

| Option | Description | Selected |
|--------|-------------|----------|
| None — leave all for Phases 40/41 | Scratch-DB notebook todo is already WARN-05; the other two are Phase 41 triage items | ✓ |
| Isolate campaign-table query-count test from the shared file cache | Fold only if it flakes during SYNC-07 | |
| load_telescope_runs: skip comment lines, warn on bare proposal token | Behavior change to a notebook-paired module | |
| Run pre-executed notebooks against a scratch DB copy | Already WARN-05 in Phase 40 | |

**User's choice:** None folded.

---

## Who does the merge, and the conflict policy

| Option | Description | Selected |
|--------|-------------|----------|
| Executor, in a plan, with a checkpoint before committing | Plan task runs the merge, resolves conflicts (keep both sides), pauses so the developer inspects `git diff --cached`; worktrees off | ✓ |
| You merge by hand first; GSD plans from the merged tree | Merge outside the audit trail; plans cover only follow-ups | |
| Executor does it unattended, no checkpoint | Fastest; a wrong resolution only caught by the suite | |

| Option | Description | Selected |
|--------|-------------|----------|
| Conflict resolutions only; everything else after | Merge commit = origin/main + minimum conflict edits; floors, ruff, CI, CLAUDE.md, test fixes as separate commits | ✓ |
| Merge + whatever it takes for the suite to pass | No red commit, but mixes a sync with code changes | |

| Option | Description | Selected |
|--------|-------------|----------|
| Keep all three (`tom-registration`, `timezonefinder`, `graphifyy`), add `tom_jpl>=0.3.0` | Union of both sides | |
| Keep the runtime two, drop `graphifyy` from `[dev]` | graphifyy is GSD tooling | |
| *Other:* "tom-registration is removed and not needed now, keep the other two" | | ✓ |

**User's choice:** Executor merges in-plan with a pre-commit checkpoint; merge commit holds conflict resolutions only; `tom-registration` removed (follow-up commit touching pyproject, settings.py app + middleware, docs/installation.rst, CLAUDE.md), `timezonefinder` and `graphifyy` kept, `tom_jpl` added.
**Notes:** `main` has no trace of `tom_registration`; on the branch `settings.py` auto-merges, so the removal is deliberate, not a conflict resolution.

---

## The test command in CI and pre-commit

| Option | Description | Selected |
|--------|-------------|----------|
| main's command plus `--exclude-tag ephemeris_segfault` | `coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault`; Playwright in the functional-tests job | ✓ |
| main's command exactly (`--exclude-tag functional` only) | TestEphemeris runs in CI; a crash gives no report | |
| The branch's own `test solsys_code --exclude-tag=ephemeris_segfault` with coverage | Drops main's functional split | |

| Option | Description | Selected |
|--------|-------------|----------|
| Adopt main's `django-test` hook, with the same ephemeris exclusion | Replaces pytest-check one-for-one; `SKIP=django-test` for WIP | ✓ |
| Adopt it as-is (main's exact entry) | Every local commit would hit the ASSIST segfault | |
| Drop it: no test run in pre-commit | Diverges from main's hook list | |

| Option | Description | Selected |
|--------|-------------|----------|
| Drop sphinx-build; keep nb hook, repoint at `docs/notebooks/pre_executed/` | Follows main's list; nb hook actually watches the branch's notebooks | ✓ |
| Keep both: sphinx-build stays, nb hook repointed | More per-commit time | |
| Take main's file verbatim | nb hook a no-op on the branch | |

**User's choice:** As marked.
**Notes:** None.

---

## CLAUDE.md and ruff after the merge

| Option | Description | Selected |
|--------|-------------|----------|
| Keep `pre-commit run ruff` routing; update D-07 to 0.16.9 | Drop main's bare-ruff lines from the merged file | ✓ |
| Adopt main's bare `ruff check . --fix` / `ruff format .` text | Drops the D-07 note | |
| Document both | Two commands for one job | |

| Option | Description | Selected |
|--------|-------------|----------|
| Fix the code; extend ignores only for rules Rubin DM style already waives | Behavior-changing fixes flagged, not silently applied | ✓ |
| Add every new rule to the ignore list | Zero churn, lint-clean by disabling | |
| Adopt main's `[tool.ruff]` verbatim and fix what it reports | Drops the `.planning` exclude | |

| Option | Description | Selected |
|--------|-------------|----------|
| Union: keep the branch's excludes, take main's coverage block | ruff keeps skipping `.planning`; coverage means the same on both branches | ✓ |
| Take main's blocks verbatim | ruff would lint `.planning/` | |

**User's choice:** As marked.
**Notes:** None.

---

## PR #43: refresh its branch or body only

| Option | Description | Selected |
|--------|-------------|----------|
| Yes: rebuild the PR branch, push, then rewrite the body | PR diff shows v2.4 code on current main; stays a draft | ✓ |
| Body only; leave `issue37-code-only` at 2026-09-01 | Description would outrun the diff | |
| Retarget PR #43 to `issue37-telescope-runs-calendar` | PR diff would include every `.planning/` commit | |

Correction raised during discussion: `/gsd-pr-branch` cherry-picks all 2331 commits into a new `-pr` branch and (with `pr_strict` unset) keeps STATE/ROADMAP/PROJECT; it does not refresh `issue37-code-only`, which is 4 snapshot commits on old-main `67fb479`.

| Option | Description | Selected |
|--------|-------------|----------|
| Extend it: merge origin/main into it, one snapshot commit, push normally | No force-push; PR's 4 commits stay; three-dot diff = branch vs current main | ✓ |
| Rebuild it from origin/main + one snapshot, force-push | Cleanest single commit, rewrites draft history | |
| `/gsd-pr-branch` with `pr_strict: true`, retarget the PR | Full filtered history, heavy, head branch changes | |

| Option | Description | Selected |
|--------|-------------|----------|
| Pillars + runbook link + short "how to try it" + draft-pending-v2.5 note | | ✓ |
| Pillars + runbook link only | Minimal | |
| Full changelog v1.0 → v2.4 | Long, goes stale | |

**User's choice:** As marked.
**Notes:** `.claude/` is gitignored; `deploy/` (added in v2.4) is absent from the current code-only branch and is included this time.

---

## Claude's Discretion

- One mechanical `style:` reformat commit separate from lint-fix commits.
- CLAUDE.md Testing section rewritten in main's terms plus the `ephemeris_segfault` exclusion; Conventions line about pre-commit running pytest updated; `tom_registration` dependency line removed.
- Where the `tom_calendar` override comparison for Phase 39 is recorded (a note in the phase directory).
- Whether a throwaway venv is needed to prove SYNC-02's "fresh install".
- Ordering of the follow-up commits after the merge.

## Deferred Ideas

None — discussion stayed within phase scope. Three todos reviewed and left for Phases 40/41 (see CONTEXT.md).
