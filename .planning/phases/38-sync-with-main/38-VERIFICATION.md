---
phase: 38-sync-with-main
verified: 2026-10-08T00:35:00Z
status: human_needed
score: 46/46 must-haves verified
covered_files:
  - ".copier-answers.yml"
  - ".github/workflows/smoke-test.yml"
  - ".github/workflows/testing-and-coverage.yml"
  - ".gitignore"
  - ".planning/phases/38-sync-with-main/38-01-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-01-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-02-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-02-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-03-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-03-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-04-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-04-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-05-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-05-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-05-red-evidence.json"
  - ".planning/phases/38-sync-with-main/38-06-PLAN.md"
  - ".planning/phases/38-sync-with-main/38-06-SUMMARY.md"
  - ".planning/phases/38-sync-with-main/38-OVERRIDE-COMPARISON.md"
  - ".planning/phases/38-sync-with-main/38-PATTERNS.md"
  - ".planning/phases/38-sync-with-main/38-PR43-BODY.md"
  - ".planning/phases/38-sync-with-main/38-RESEARCH.md"
  - ".pre-commit-config.yaml"
  - "CLAUDE.md"
  - "docs/conf.py"
  - "docs/design/design.rst"
  - "docs/installation.rst"
  - "docs/notebooks.rst"
  - "pyproject.toml"
  - "solsys_code/admin.py"
  - "solsys_code/apps.py"
  - "solsys_code/management/commands/backfill_lco_observations.py"
  - "solsys_code/tests/test_bootstrap5_rendering.py"
  - "solsys_code/tests/test_urls.py"
  - "solsys_code/tests/test_views.py"
  - "src/fomo/settings.py"
  - "src/fomo/urls.py"
covered_digest: "v3:sha256:fa170b06c12dbe089a70f40428700abf048d9e98c40f61e9cf33fe875b98c8a5"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 29/30
  gaps_closed:
    - "The branch carries everything main already has: main's tomtoolkit 3.1.0 adaptation in commit ada2000, which deletes the project-level 'alerts/' route because tom_alerts is no longer an installed app, is present on the merged tree (all four missing items: include deleted, regression test added, PR #43 head re-snapshotted, planning docs corrected)"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "WR-01 decision (carried unchanged; src/fomo/settings.py not modified since the previous verification): local_settings.py location. The merged settings import `fomo.local_settings` (branch commit c0f883d, deliberate per 36-REVIEW WR-32); main imports top-level `local_settings`. Neither docs/installation.rst nor the PR #43 body tells a host set up for main that the file must now live at src/fomo/local_settings.py, and a missing file falls back silently to dev defaults."
    expected: "Developer chooses: (a) document the location in docs/installation.rst and 38-PR43-BODY.md / PR #43, or (b) accept both locations during the transition. Either is a follow-up before PR #43 leaves draft."
    why_human: "Deployment-contract choice between two working designs; not a Phase 38 success criterion."
  - test: "WR-02 decision (carried unchanged; .github/workflows not modified since the previous verification): no CI job runs TestEphemeris (tagged ephemeris_segfault); main's unit-test matrix ran it. Locked decision D-05 set the CI form `--exclude-tag functional --exclude-tag ephemeris_segfault`, and the phase implemented it exactly (CI log on 846be34 shows that command in all three build jobs)."
    expected: "Developer decides whether to add a separate (possibly non-blocking) `python manage.py test --tag ephemeris_segfault` step, or accept the coverage loss as the cost of D-05."
    why_human: "Locked decision D-05 produced this outcome on purpose; changing it is the developer's call."
  - test: "Judgment-tier prohibitions (non-authoritative LLM verdicts, flagged unverified-prohibition -- human review recommended): (1) nothing installed before the developer confirmed the packages at 38-01 Task 2; (2) developer DB not migrated before a verified backup and never with the cron line active; (3) no live LCO/SOAR portal call, real email or heartbeat ping from an executor run; (4) downloaded tomtoolkit wheels only unzipped and diffed; (5) primary checkout never switched to issue37-code-only (38-04 and 38-06)."
    expected: "Developer confirms each from memory of the session. Verifier evidence: SUMMARYs record 'approve' and 'publish' verbatim; backup src/fomo_db_20261007_pre_phase38.sqlite3 exists (mtime 11:10 PDT, after the 10:19 merge); crontab has no PHASE38-PAUSED line; HEAD's reflog for 2026-10-07 holds only commit entries (no checkout, switch, reset, rebase or amend), including through the 38-06 publish window."
    why_human: "Ordering of approvals/cron state against installs and migrate cannot be reconstructed from repository state."
---

# Phase 38: Sync with main Verification Report

**Phase Goal:** The branch carries everything `main` already has — its dependency floors, ruff 0.16.9, LINCC python-project-template v2.2.0 and the Django test runner in CI — and the full suite passes on tomtoolkit 3.1.0, so every later v2.5 phase works on the merged tree.
**Verified:** 2026-10-08T00:35:00Z
**Status:** human_needed
**Re-verification:** Yes — after gap closure (plans 38-05 and 38-06)

## Goal Achievement

The one gap from the previous report (CR-01: the merge resolution had restored main's deleted `alerts/` include) is closed, on both the local branch and the published PR #43 head. I checked this against HEAD (7d5757f) and origin, not against the SUMMARYs:

- `git diff origin/main -- src/fomo/urls.py` now contains only additions: the `ProtectedUserDeleteView` import and the branch's `calendar/`, `campaigns/` and `users/<int:pk>/delete/` entries. Nothing that main has is removed or restored.
- At runtime, `/alerts/query/list/` and `/alerts/` raise Resolver404, and `alerts:list` and `tom_alerts:list` both raise NoReverseMatch.
- The new regression module passes. I also ran it against the pre-fix urlconf in scratch, and it failed there.

Since the previous verification, only two files outside `.planning/` have changed (`git diff --stat 6fd4d87 HEAD -- . ':(exclude).planning'`): `src/fomo/urls.py` (-4 lines) and the new `solsys_code/tests/test_urls.py` (+56 lines). So the 29 truths that passed before were checked for regressions, not re-derived. Nothing they depend on (pyproject, workflows, hooks, settings, CLAUDE.md, docs, admin/apps) changed.

Status is `human_needed`, not `passed`. The cause is the three human-decision items carried from the previous report (WR-01, WR-02, judgment-tier prohibitions), not any failed truth.

### Observable Truths

Truths 1-30 are the previous report's must-haves (re-verification). Truths 31-46 are the must_haves of the gap-closure plans 38-05 and 38-06.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC#1: one merge commit, parents = pre-merge head + origin/main head; origin/main ancestor; no branch commit rewritten | ✓ VERIFIED (regression check) | origin/main still a910c17; `merge-base --is-ancestor origin/main HEAD` ok; HEAD reflog since the previous verification has only `commit:` entries |
| 2 | SC#2: fresh install gives tomtoolkit ≥3.1.0, tom_jpl ≥0.3.0; no floor above main's; suite passes with nothing newly skipped | ✓ VERIFIED (regression check) | pyproject.toml unchanged since the previous verification; fresh-venv evidence stands. The corrected tree's dev-venv full run: `Ran 2182 tests in 461.288s`, exact `OK`, 0 FAIL/ERROR headers, no `skipped=` (log mtime 14:59:06 PDT, after fix commit a4d77f2 at 14:51:14, and the three alerts tests pass in it, which they cannot on the unfixed tree) |
| 3 | SC#3: both ruff hooks run 0.16.9 and are clean; 0.2.1 references gone | ✓ VERIFIED (regression check) | config/CLAUDE.md unchanged; `pre-commit run ruff` / `ruff-format` on the two changed files: Passed / Passed, porcelain empty |
| 4 | SC#4: CI runs Django runner with coverage, no pytest; pytest config gone; LINCC v2.2.0 present; no FOMO file lost | ✓ VERIFIED | Run 37700624248 on 846be34: build (3.10/3.11/3.12) step `Run Django unit tests with coverage` (`coverage run manage.py test --exclude-tag functional --exclude-tag ephemeris_segfault`) success, `Ran 2174 tests` in each; functional-tests success; pre-commit 37700624234 and docs 37700624246 success; no pytest job/step |
| 5 | SC#5: PR #43 still a draft; body covers the four v2.4 pillars and links the runbook | ✓ VERIFIED | `gh pr view 43`: isDraft true, OPEN, head issue37-code-only @ 846be34, base main; all D-12 sections, runbook path and draft line present |
| 6 | Goal: the branch carries everything main has, including main's ada2000 removal of the `alerts/` route | ✓ VERIFIED (was ✗ FAILED) | urls.py has no `tom_alerts`/`alerts/` text; diff vs origin/main is additions only; `apps.is_installed('tom_alerts')` False; `resolve('/alerts/query/list/')` and `resolve('/alerts/')` → Resolver404; `reverse('alerts:list')` and `reverse('tom_alerts:list')` → NoReverseMatch; no `alerts:`/`tom_alerts:` reversal left in src/ or solsys_code/ |
| 7 | 38-01 parent order / exactly two parents / no unmerged path | ✓ VERIFIED (regression check) | merge commit untouched; fix is an ordinary later commit |
| 8 | D-02: outside the nine paths the merge equals git's automatic merge | ✓ VERIFIED (regression check) | merge commit untouched |
| 9 | One `nav_items`, both menus | ✓ VERIFIED (regression check) | apps.py unchanged |
| 10 | Target registered once via SolsysTargetAdmin | ✓ VERIFIED (regression check) | admin.py unchanged |
| 11 | docs/conf.py guard + skip hook + `nbsphinx_allow_errors` | ✓ VERIFIED (regression check) | unchanged; docs build success on 846be34 |
| 12 | URL order: shadow routes before tom_common (truth as corrected by 38-05: no `alerts/` route) | ✓ VERIFIED | Route-set check re-run: routes = origin/main's + `calendar/`, `campaigns/`, `users/<int:pk>/delete/`, no duplicate, none after `include('tom_common.urls')` |
| 13 | Merge deletes exactly the two tests/fomo files | ✓ VERIFIED (regression check) | merge commit untouched |
| 14 | No commit landed while MERGE_HEAD existed | ✓ VERIFIED (regression check) | historical; unchanged |
| 15 | pyproject after merge (D-04, D-10) | ✓ VERIFIED (regression check) | unchanged |
| 16 | Dev venv on 3.1.0 / 0.3.0, tom-registration gone | ✓ VERIFIED (regression check) | no install since; test runs use the same venv |
| 17 | Merged tree boots: only urls.W005; no migration drift | ✓ VERIFIED | Verifier ran `manage.py check`: only `(urls.W005) URL namespace 'calendar' isn't unique` |
| 18 | SYNC-03 ordering: style commit before fix, pure reformat | ✓ VERIFIED (regression check) | historical commits untouched |
| 19 | D-09: SIM103 fixed in code | ✓ VERIFIED (regression check) | backfill_lco_observations.py unchanged |
| 20 | D-05/SYNC-05: CI files differ from main by one line each | ✓ VERIFIED (regression check) | .github unchanged since the previous verification |
| 21 | D-06/D-07 hook config | ✓ VERIFIED (regression check) | .pre-commit-config.yaml unchanged |
| 22 | `@tag('ephemeris_segfault')` on exactly one class | ✓ VERIFIED (regression check) | test_views.py unchanged; test_urls.py carries no tag |
| 23 | Eight pre-executed notebooks never-execute | ✓ VERIFIED (regression check) | notebooks unchanged |
| 24 | CLAUDE.md, installation.rst, settings in step | ✓ VERIFIED (regression check) | unchanged |
| 25 | Runbook system-check paragraph stays true | ✓ VERIFIED | `manage.py check` still reports only urls.W005 |
| 26 | SYNC-05 local proof: CI form passes | ✓ VERIFIED | CI form ran on GitHub for the corrected tree: 2174 tests OK × 3 Python versions (2170 + the 4 new tests) |
| 27 | Developer DB migrated behind a backup; crontab restored | ✓ VERIFIED (regression check) | backup present (mtime 11:10); no `PHASE38-PAUSED` line in crontab; 38-05 migrated nothing |
| 28 | 38-OVERRIDE-COMPARISON.md covers every override | ✓ VERIFIED (regression check) | no template changed |
| 29 | D-11 code-only refresh: snapshot equals v2.5 tree minus `.planning/`; no force-push; no leaks | ✓ VERIFIED | `git diff --quiet HEAD origin/issue37-code-only -- . ':(exclude).planning'` equal at HEAD 7d5757f / 846be34 |
| 30 | Worktree removed; primary checkout on the v2.5 branch | ✓ VERIFIED | `git worktree list`: only the primary checkout, on issue37-telescope-runs-calendar |
| 31 | 38-05 gap truth: urls.py no longer routes alerts/; Resolver404, NoReverseMatch, logged-in GET → 404 | ✓ VERIFIED | `python manage.py test -v 2 solsys_code.tests.test_urls`: Ran 4 tests, OK; the GET test logs `Not Found: /alerts/query/list/` |
| 32 | 38-05 Edge SYNC-04/ordering (corrected route list) | ✓ VERIFIED | Same as truth 12 |
| 33 | 38-05 Edge SYNC-04/empty: fix commit removes exactly four lines, adds none | ✓ VERIFIED | `git diff --numstat a4d77f2^ a4d77f2` → `0	4	src/fomo/urls.py` |
| 34 | Regression guard: two classes / four tests, committed before the fix, RED then GREEN | ✓ VERIFIED | 32dafa2 (test, 56 lines) is an ancestor of a4d77f2^. 38-05-red-evidence.json: exitCode 1, `FAILED (failures=3)`, `500 != 404`. Independent mutation check: test_urls run against `git show a4d77f2^:src/fomo/urls.py` via ROOT_URLCONF in scratch → 3 FAIL (path, namespace, GET), guard test passes, `FAILED (failures=3)` |
| 35 | Guard runs everywhere the suite runs (no tag/skip/expectedFailure) | ✓ VERIFIED | grep for `@tag|skip|expectedFailure` in test_urls.py: nothing. The CI build logs (which exclude functional and ephemeris_segfault) contain the alerts-probe test output in all three jobs (9 matching lines) |
| 36 | SYNC-07 on corrected tree: ≥2182 tests, exact OK, no FAILED/skipped, all four test_urls listed | ✓ VERIFIED | `$HOME/tmp/phase38-05-suite.log`: line 5952 `Ran 2182 tests`, 5954 `OK`; four test_urls lines at 5579-5581 and 5923 (log evidence, not re-run: no non-.planning file changed since) |
| 37 | `manage.py check` only urls.W005; ruff hooks clean on both files | ✓ VERIFIED | Verifier runs (above) |
| 38 | Planning docs no longer name alerts/ as a branch route | ✓ VERIFIED | None of the seven old phrases remains. "corrected by 38-05" appears 4× in 38-01, 1× in RESEARCH and 1× in PATTERNS. 38-01 frontmatter parses, and its SYNC-04/ordering truth now lists `calendar/`, `campaigns/`, `users/<int:pk>/delete/`. db3ae7c numstat: 4/4, 1/1, 1/1, plus the evidence JSON |
| 39 | No paired notebook or runbook change needed | ✓ VERIFIED | urls.py and solsys_code/tests/ are not in CLAUDE.md's notebook map; `grep -i 'alerts/\|tom_alerts'` in docs/runbooks/: nothing |
| 40 | 38-06 gap item 3: PR head has no alerts route; PR's urls.py diff adds no tom_alerts line; test_urls.py on PR head | ✓ VERIFIED | `git show origin/issue37-code-only:src/fomo/urls.py`: no alerts text; three-dot diff has no `+…alerts` line; the PR head tree equals HEAD outside .planning, so it includes test_urls.py |
| 41 | D-11 re-snapshot: one commit on 1a68a76, tree = v2.5 minus .planning, changes exactly two files | ✓ VERIFIED | `rev-list --parents -n1 846be34` → single parent 1a68a76; `diff --name-only 1a68a76 846be34` → test_urls.py, urls.py |
| 42 | No new merge needed: origin/main already an ancestor | ✓ VERIFIED | origin/main still a910c17, an ancestor of origin/issue37-code-only; 846be34 has one parent |
| 43 | Publish-time concurrency: nothing pushed before the developer's answer | ✓ VERIFIED | Remote-tracking reflogs: one `update by push` per branch since 38-04 (11:41), both at 16:09 PDT. The SUMMARY records the "publish" answer at 23:07Z (16:07 PDT) |
| 44 | No force-push; origin tips equal local | ✓ VERIFIED | 1a68a76 is an ancestor of origin/issue37-code-only (= local 846be34), and 62605b8 is an ancestor of origin/issue37-telescope-runs-calendar (49be149). The local branch has since gained only `.planning` docs commits, which are not pushed |
| 45 | PR #43 still a draft, body unchanged with D-12 sections; CI green on 846be34 with Django runner, no pytest | ✓ VERIFIED | Truths 4 and 5 |
| 46 | Nothing private published | ✓ VERIFIED | `ls-tree -r origin/issue37-code-only` has no `.planning/`, `*.sqlite3`, `local_settings.py` or `reqgroup_*.json` |

**Score:** 46/46 truths verified (0 present, behavior-unverified)

### Prohibitions (ADR-550)

| Prohibition | Tier | Disposition | Evidence |
|-------------|------|-------------|----------|
| (38-01..38-04 test-tier items) | test | ✓ held (regression) | Merge history untouched; no floor or workflow change since the previous verification |
| tom_alerts not re-added to INSTALLED_APPS/settings (38-05) | test | ✓ held | settings.py unchanged since 6fd4d87; `apps.is_installed('tom_alerts')` False; the only `tom_alerts` text in settings.py is the pre-existing line-312 comment (`tom_alertstreams` is a different app) |
| No other route removed/reordered/reworded (38-05) | test | ✓ held | numstat 0/4; route-set check; diff vs origin/main is additions only |
| Regression test not tagged/skipped/outside solsys_code/tests (38-05) | test | ✓ held | grep check; module at solsys_code/tests/test_urls.py; runs in CI |
| No executed-plan history rewritten beyond the alerts wording (38-05) | test | ✓ held | `git diff --name-only 99e4975 49be149`: only 38-01-PLAN, RESEARCH, PATTERNS, the evidence JSON, 38-05-SUMMARY and the STATE/ROADMAP/REQUIREMENTS bookkeeping. No earlier SUMMARY, MERGE-RESOLUTION.diff, VERIFICATION, REVIEW or PR43-BODY was edited |
| Nothing pushed in 38-05 (38-05) | test | ✓ held | The first push after 11:41 was at 16:09, after 38-05 completed (15:14) |
| No force-push / no rewrite on origin (38-06) | test | ✓ held | Ancestry of both pre-push tips; reflog `update by push` |
| PR #43 not readied, merged or edited (38-06) | test | ✓ held | isDraft true; body sections unchanged |
| Nothing private published (38-06) | test | ✓ held | Leak scan (truth 46) |
| Code-only gains one commit, no worktree-only change (38-06) | test | ✓ held | Truth 41; tree equality |
| Package gate / backup-before-migrate / no live portal calls / wheels only diffed (38-01, 38-03) | judgment | ⚠ flagged (non-authoritative: likely held) | Carried; see Human Verification |
| Primary checkout never switched to issue37-code-only (38-04, 38-06) | judgment | ⚠ flagged (non-authoritative: held) | HEAD reflog 2026-10-07 has only commit entries; worktree used and removed |

### Advisory (New Scope, Unevidenced)

None. The re-verification evidence gate had nothing to downgrade: Step 7 found no 🛑 Blocker on any file.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/fomo/urls.py` | main's routes + calendar/campaigns/user-delete, all before tom_common; no alerts include | ✓ VERIFIED | contains `ProtectedUserDeleteView.as_view(), name='user-delete'`; 36 lines; no `tom_alerts` |
| `solsys_code/tests/test_urls.py` | regression guard | ✓ VERIFIED | contains `class TestAlertsRouteRemoved(TestCase)`; 3+1 tests; passes; fails on mutated urlconf |
| `38-05-red-evidence.json` | RED evidence record | ✓ VERIFIED | targetTest `test_alerts_path_does_not_resolve`, exitCode 1, real output |
| `38-01-PLAN.md`, `38-RESEARCH.md`, `38-PATTERNS.md` | corrected wording | ✓ VERIFIED | `corrected by 38-05` markers; old phrases gone |
| (truths 1-30 artifacts) | as in the previous report | ✓ VERIFIED (regression) | files unchanged |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| solsys_code/tests/test_urls.py | src/fomo/urls.py | `resolve('/alerts/query/list/')`, `reverse()`, logged-in Client GET against ROOT_URLCONF | ✓ WIRED (mutation check proves it reads the live urlconf) |
| .github/workflows/testing-and-coverage.yml | solsys_code/tests/test_urls.py | untagged module under `--exclude-tag functional --exclude-tag ephemeris_segfault` | ✓ WIRED (alerts-probe output in all three CI build logs; 2174 = 2170 + 4) |
| src/fomo/urls.py | solsys_code/views.py | `ProtectedUserDeleteView.as_view(), name='user-delete'` before tom_common | ✓ WIRED (guard test asserts `view_class is ProtectedUserDeleteView`) |
| issue37-code-only snapshot | issue37-telescope-runs-calendar | read-tree snapshot minus .planning | ✓ WIRED (tree-equal at HEAD) |
| PR #43 | testing-and-coverage.yml | pull_request event | ✓ WIRED (runs on 846be34) |
| src/fomo/urls.py | tom_alerts.urls | (previous report: STALE) | ✓ REMOVED, as on main |
| (other previous links) | | | ✓ WIRED (regression, files unchanged) |

### Data-Flow Trace (Level 4)

Not applicable. The phase changes tooling, dependencies and URL wiring; it adds no dynamic-data rendering. The URL-resolution runtime checks stand in for it.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Regression module passes | `python manage.py test --noinput -v 2 solsys_code.tests.test_urls` | Ran 4 tests, OK | ✓ PASS |
| Regression module catches the defect | test_urls against `a4d77f2^:src/fomo/urls.py` loaded as ROOT_URLCONF from scratch (programmatic Django DiscoverRunner, not `manage.py test`; no repo file touched) | FAILED (failures=3): path, namespace, GET | ✓ PASS (guard is real) |
| alerts URL space gone at runtime | `manage.py shell -c` resolve/reverse probe | Resolver404 ×2, NoReverseMatch for `alerts:list` and `tom_alerts:list`; tom_alerts not installed | ✓ PASS |
| `targets/export/` shadow still effective (IN-05 context) | `python manage.py test solsys_code.tests.test_scout_views.ScoutTargetListFilterTest.test_export_honours_scout_filter` | OK | ✓ PASS |
| Django boots | `python manage.py check` | only urls.W005 | ✓ PASS |
| Enforced ruff gate on changed files | `pre-commit run ruff/ruff-format --files ...` | Passed / Passed | ✓ PASS |
| Full suite on corrected tree | log tail (not re-run: no non-.planning change since the log) | Ran 2182, OK | ✓ PASS (log evidence) |
| GitHub CI on PR head 846be34 | `gh run list/view`, `gh run view --log` | 3 workflows success; Django runner step in 3 build jobs; Ran 2174 ×3 | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. No probe scripts are declared in any plan, and none exist under `scripts/*/tests/probe-*.sh`.

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| SYNC-01 | 38-01 | ✓ SATISFIED | Truths 1, 7, 14. History-level merge; the content-level revert (CR-01) is now undone (truth 6) |
| SYNC-02 | 38-01, 38-03 | ✓ SATISFIED | Truths 2, 15, 16 |
| SYNC-03 | 38-02 | ✓ SATISFIED | Truths 3, 18, 19, 24, 37 |
| SYNC-04 | 38-01, 38-05, 38-06 | ✓ SATISFIED (was BLOCKED) | Truths 4, 6, 12, 31-33, 40. The repo now follows main for the tomtoolkit 3.1.0 URL wiring, on the branch and on the PR head |
| SYNC-05 | 38-02, 38-03, 38-04, 38-06 | ✓ SATISFIED | Truths 4, 20, 26, 45; CI form per locked D-05 (see WR-02) |
| SYNC-06 | 38-02 | ✓ SATISFIED | Truths 4, 21, 24 |
| SYNC-07 | 38-03, 38-05 | ✓ SATISFIED | Truths 2, 36: 2182 OK on the corrected tree, nothing skipped; the alerts/ space is now covered by a test |
| SYNC-08 | 38-04, 38-06 | ✓ SATISFIED | Truths 5, 45 |

No orphaned requirements: REQUIREMENTS.md maps exactly SYNC-01..08 to Phase 38, and every one is claimed by at least one plan. Bookkeeping note (not a gap): REQUIREMENTS.md still shows SYNC-01, -02, -03 and -06 unchecked with traceability status "Gaps Found". That reflects the earlier `e80c9ae` revert. The orchestrator should tick all eight when it records this verification.

### Code Review Findings (38-REVIEW.md re-review) — Classification

| ID | Classification | Reasoning |
|----|----------------|-----------|
| CR-01 | **Closed** | Confirmed independently (truths 6, 31-36, 40). |
| IN-05 (`TestProjectRoutesStillResolve` omits `targets/export/`; six checks in one method) | **Info / advisory. Not a must-have gap.** | 38-05's must_haves list the guard's routes exactly (`/targets/`, `/scout/rubin-too/`, `/scout/rubin-too/stats/`, `/calendar/`, `/campaigns/`, `/users/1/delete/`), and the test implements that list exactly. The ordering truth (12/32) requires `targets/export/` before tom_common, and it holds: I checked it with the route-set script, not with this test. The reorder the review describes is still caught by behavior in `ScoutTargetListFilterTest.test_export_honours_scout_filter`, which I ran and which passes. (The review names it `TestScoutTargetFilter`; the real class is `ScoutTargetListFilterTest`.) The docstring says "Routes registered before tom_common.urls still win", not "every such route", so it is not a false claim. Worth a small follow-up, with no effect on status. |
| IN-06 (namespace test checks `alerts:` but not `tom_alerts:`) | **Info / advisory. Not a must-have gap.** | Gap item 2 asked for a test that `/alerts/query/list/` 404s or that resolve() raises Resolver404. 38-05's truth specifies `reverse('alerts:list')`. Both are met. The uncaught case would be an include of tom_alerts.urls with no namespace under a prefix other than `alerts/`, and that is not a restoration of the removed `alerts/` route. Today `reverse('tom_alerts:list')` does raise NoReverseMatch (verifier probe). Adding it to the test is a cheap hardening follow-up. |
| WR-01, WR-02 | Human decision (unchanged) | src/fomo/settings.py and .github/workflows have not changed since the previous verification, so the classification stays the same. |
| IN-01..IN-04 | Info (unchanged) | Files outside the gap-closure scope are unchanged. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/fomo/urls.py, solsys_code/tests/test_urls.py | — | TBD/FIXME/XXX/TODO/HACK scan | — | none found |
| solsys_code/tests/test_urls.py | 38-56 | Six route assertions in one method (no `subTest`) | ℹ️ Info | First failure hides the rest (IN-05); diagnostic only |
| docs/_build/html/autoapi/fomo/local_settings/ (untracked, git-ignored, predates phase) | — | Stale local Sphinx build rendering local_settings values | ℹ️ Info | Carried from the previous report; not tracked or published. Deleting `docs/_build/` locally is advisable |

### Human Verification Required

1. **WR-01 local_settings location.** Decide whether to document it (installation.rst and the PR #43 body) or to accept both import locations before PR #43 leaves draft.
2. **WR-02 TestEphemeris CI coverage.** Decide whether to add a separate `--tag ephemeris_segfault` CI step, possibly non-blocking, or to accept the D-05 trade-off.
3. **Judgment-tier prohibitions.** Confirm five things:
   - the package gate ran before any install;
   - the backup came before the cron pause, which came before the migrate;
   - no live portal, email or heartbeat calls were made;
   - the wheels were only diffed;
   - the primary checkout never switched (38-04 and 38-06).

   Repository evidence is consistent with all five.

### Gaps Summary

There are no gaps. The previous gap (CR-01) is closed in all four of its named parts:

1. Main's side of the `alerts/` include is taken in `src/fomo/urls.py`: the fix deletes exactly four lines, and the file now differs from origin/main only by the branch's additions.
2. An untagged regression module guards it. It was proven red against the unfixed urlconf, both by the executor and by the verifier's independent mutation run, and it is green in the dev venv and in CI on 3.10, 3.11 and 3.12.
3. PR #43's head was re-snapshotted as 846be34, a plain fast-forward with one parent, two changed files and a tree equal to the branch minus `.planning/`. The PR is still a draft, its body is unchanged and its CI is green.
4. The planning docs no longer call `alerts/` a branch route.

The goal now holds: the branch and its PR carry everything main has, and the full suite passes on tomtoolkit 3.1.0. Status is `human_needed` only because of the three carried human-decision items. IN-05 and IN-06 are Info-level hardening suggestions, not must-have gaps.

---

_Verified: 2026-10-08T00:35:00Z_
_Verifier: Claude (gsd-verifier)_
