---
phase: 38-sync-with-main
verified: 2026-10-07T19:45:00Z
status: gaps_found
score: 29/30 must-haves verified
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
  - ".planning/phases/38-sync-with-main/38-OVERRIDE-COMPARISON.md"
  - ".planning/phases/38-sync-with-main/38-PR43-BODY.md"
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
  - "solsys_code/tests/test_views.py"
  - "src/fomo/settings.py"
  - "src/fomo/urls.py"
covered_digest: "v3:sha256:a05428f575629465fa0ecf65aa903dc9a4ffcede6aca9d8c41a33bc20ce0ce2d"
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "The branch carries everything main already has: main's tomtoolkit 3.1.0 adaptation in commit ada2000, which deletes the project-level 'alerts/' route because tom_alerts is no longer an installed app, is present on the merged tree (ROADMAP Phase 38 goal; SYNC-04 'as main does'; review CR-01)"
    status: failed
    reason: "The 38-01 conflict resolution for src/fomo/urls.py kept `path('alerts/', include('tom_alerts.urls', namespace='alerts'))` and its comment. That line was NOT branch work: it is shared merge-base content (756680f) that the branch never changed (`git diff 756680f 088f73b -- src/fomo/urls.py` leaves it as context) and that main deliberately deleted in ada2000 ('remove alerts/ url as tom_alerts is gone/going'). It sat inside the conflict hunk only because the branch's calendar/campaigns/user-delete additions surround it, and the resolution table (RESEARCH line 287, 38-01 truth 'Edge SYNC-04/ordering') wrongly listed it as a branch entry. Result, confirmed by the verifier in the dev venv: `apps.is_installed('tom_alerts')` is False (tomtoolkit 3.1.0's TOMTOOLKIT_INSTALLED_APPS has no tom_alerts), yet `/alerts/query/list/` resolves to `alerts:list` (tom_alerts.views), and its template lookup raises TemplateDoesNotExist (tom_alerts/brokerquery_list.html), so the alerts pages 500. The route's comment ('tom_alerts is still an installed app') is now false. The same tree is published on PR #43's head, so the PR diff silently re-adds what main removed. Not caught by the suite or `manage.py check`. Note: the review's headline 'every page view 500s' overstates it -- the full suite (2178 OK) renders the other pages; the breakage is confined to the alerts/ URL space, but that includes live redirecting endpoints (CreateTargetFromAlertView, SubmitAlertUpstreamView) served from a dropped app."
    artifacts:
      - path: "src/fomo/urls.py"
        issue: "lines 32-35: the alerts/ include and its three-line comment, which main's ada2000 removed, restored by the merge resolution"
    missing:
      - "Take main's side for that hunk: delete the three comment lines and `path('alerts/', include('tom_alerts.urls', namespace='alerts'))` from src/fomo/urls.py (keep the branch's calendar/, campaigns/ and users/<int:pk>/delete/ entries)"
      - "Add a regression test (solsys_code/tests/) asserting `/alerts/query/list/` returns 404 (or that resolve() raises Resolver404), so a later merge cannot quietly bring it back"
      - "Re-snapshot issue37-code-only after the fix so PR #43 stops re-adding the route (D-11 snapshot procedure, plain push)"
      - "Correct the 38-01 must_haves wording / RESEARCH resolution table entry that names 'alerts/' as a branch route, so re-verification does not re-assert it"
human_verification:
  - test: "WR-01 decision: local_settings.py location. The merged settings import `fomo.local_settings` (branch commit c0f883d, deliberate per 36-REVIEW WR-32); main imports top-level `local_settings`. The settings.py region auto-merged (main never touched it), so this is not a merge error, but neither docs/installation.rst nor the PR #43 body tells a host set up for main that the file must now live at src/fomo/local_settings.py, and a missing file falls back silently to dev defaults."
    expected: "Developer chooses: (a) document the location in docs/installation.rst and 38-PR43-BODY.md / PR #43, or (b) accept both locations during the transition as 38-REVIEW.md suggests. Either is a follow-up before PR #43 leaves draft."
    why_human: "Deployment-contract choice between two working designs; not a Phase 38 success criterion."
  - test: "WR-02 decision: no CI job runs TestEphemeris (tagged ephemeris_segfault) any more; main's unit-test matrix ran it. D-05 locked the CI form `--exclude-tag functional --exclude-tag ephemeris_segfault`, which the phase implemented exactly."
    expected: "Developer decides whether to add a separate (possibly non-blocking) `python manage.py test --tag ephemeris_segfault` step, or accept the coverage loss as the cost of D-05."
    why_human: "Locked decision D-05 produced this outcome on purpose; changing it is the developer's call."
  - test: "Judgment-tier prohibitions (non-authoritative LLM verdicts, flagged unverified-prohibition -- human review recommended): (1) nothing installed before the developer confirmed the packages at 38-01 Task 2; (2) developer DB not migrated before a verified backup and never with the cron line active; (3) no live LCO/SOAR portal call, real email or heartbeat ping from an executor run; (4) downloaded tomtoolkit wheels only unzipped and diffed; (5) primary checkout never switched to issue37-code-only."
    expected: "Developer confirms each from memory of the session. Verifier evidence: SUMMARY records 'approve' and 'publish' verbatim; backup src/fomo_db_20261007_pre_phase38.sqlite3 exists (mtime 11:10 PDT, after the 10:19 merge), is git-ignored, and `migrate --check` exits 0; crontab is now byte-identical to the saved backup; HEAD's reflog for 2026-10-07 holds only commit entries (no checkout, reset, rebase or amend)."
    why_human: "Ordering of approvals/cron state against installs and migrate cannot be reconstructed from repository state."
---

# Phase 38: Sync with main Verification Report

**Phase Goal:** The branch carries everything `main` already has — its dependency floors, ruff 0.16.9, LINCC python-project-template v2.2.0 and the Django test runner in CI — and the full suite passes on tomtoolkit 3.1.0, so every later v2.5 phase works on the merged tree.
**Verified:** 2026-10-07T19:45:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

All five ROADMAP success criteria hold on the current tree and on the published PR head. One goal-level truth fails: the merge resolution restored a route that main deliberately removed for tomtoolkit 3.1.0 (review CR-01). So the branch does not yet carry *everything* main has. This is a one-hunk fix, but it is a real regression: it is live on the published PR head, it breaks pages, and no test catches it.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC#1: one merge commit, parents = pre-merge head + origin/main head; origin/main ancestor; no branch commit rewritten | ✓ VERIFIED | `git rev-list --parents -n1 e12158c` → 088f73b a910c17; origin/main = a910c17 (fetched today, unmoved); `merge-base --is-ancestor` origin/main, fb07a66, 088f73b → HEAD all succeed; 75ad2be ancestor of origin branch; HEAD reflog 2026-10-07 has only commit entries (no amend/rebase/reset) |
| 2 | SC#2: fresh install gives tomtoolkit ≥3.1.0, tom_jpl ≥0.3.0; no floor above main's; suite passes there with nothing newly skipped | ✓ VERIFIED | `$HOME/venv/fomo_phase38_fresh`: tomtoolkit 3.1.0, tom_jpl 0.3.0, Django 5.2.18, pip check clean; `git diff origin/main -- pyproject.toml` adds only `timezonefinder>=6.0` as a versioned line; fresh-suite log: `Ran 2178 tests`, exact `OK`, 0 `ERROR:/FAIL:` lines, no `skipped=`; tree outside `.planning/` is unchanged since 29dfb30 (before both logs were written); added lines under solsys_code/ since 088f73b include no skip marker and only main's `@tag('functional')` |
| 3 | SC#3: both ruff hooks run 0.16.9 and are clean; 0.2.1 pin/rev/CLAUDE.md refs gone | ✓ VERIFIED | Verifier ran `pre-commit run ruff --all-files` and `ruff-format --all-files`: both Passed, porcelain unchanged; hook env binary reports `ruff 0.16.9`; both ruff-pre-commit revs `v0.16.9`; `grep 0.2.1` in CLAUDE.md, pyproject.toml, .pre-commit-config.yaml → nothing |
| 4 | SC#4: CI runs Django runner with coverage, no pytest job; pytest config/extras/tests/ gone; CLAUDE.md Testing describes only the Django runner; LINCC v2.2.0 files present; no FOMO file lost | ✓ VERIFIED | Run 37668777675 on snapshot 1a68a76: build (3.10/3.11/3.12) step `Run Django unit tests with coverage` success, functional-tests success; pre-commit and docs runs success; no pytest in .github/workflows; `[tool]` tables are coverage/ruff/setuptools_scm only; dev extra has no pytest; `git ls-files tests` empty; `_commit: v2.2.0`, pull_request_template.md present, `src/_static/` and `fomo_db_20*.sqlite3` ignored; the only deletions 088f73b→HEAD are tests/fomo/conftest.py and tests/fomo/test_packaging.py |
| 5 | SC#5: PR #43 still a draft; body covers the four v2.4 pillars and links the runbook | ✓ VERIFIED | `gh pr view 43`: isDraft true, head issue37-code-only, base main, OPEN; body has all four `###` pillar sections, the runbook link, `### How to try it` and the draft line |
| 6 | Goal: the branch carries everything main has, including main's tomtoolkit 3.1.0 removal of the `alerts/` route (ada2000) | ✗ FAILED | src/fomo/urls.py:32-35 restores it. Shared base content that main deleted, not branch work. `apps.is_installed('tom_alerts')` False; `/alerts/query/list/` resolves to `alerts:list`; template lookup → TemplateDoesNotExist. See Gaps. |
| 7 | 38-01 parent order / exactly two parents / no unmerged path | ✓ VERIFIED | e12158c^1 = 088f73b (ORIG_HEAD at merge), ^2 = a910c17, no ^3; tree parses, no markers (ruff and the suite run on it) |
| 8 | D-02: outside the nine paths the merge equals git's automatic merge; no `.planning/` in the merge | ✓ VERIFIED | `git merge-tree --write-tree e12158c^1 e12158c^2` → 9aa9cc8; `git diff --quiet` against e12158c with the nine excludes → equal; 0 `.planning/` paths in the merge |
| 9 | One `nav_items`, both menus; `ready()` unchanged | ✓ VERIFIED | 1 `def nav_items`; runtime `nav_items()` → `[campaigns_nav_link.html, navbar_list.html]` |
| 10 | Target registered once via SolsysTargetAdmin; branch admins kept | ✓ VERIFIED | One `unregister(Target)` + `register(Target, SolsysTargetAdmin)`; no local TargetAdmin class; the same five registrations as 088f73b; test_admin passes (verifier run, 219 tests OK across three labels) |
| 11 | docs/conf.py keeps CR-03 guard, main's `_skip_version_module`/`setup`, `nbsphinx_allow_errors` | ✓ VERIFIED | conf.py:69, 75, 78, 85 |
| 12 | URL order: shadow routes before tom_common; design.rst/notebooks.rst order | ✓ VERIFIED (as written) | All listed routes precede `include('tom_common.urls')`. This truth also names `'alerts/'`, which is the CR-01 defect (truth 6). |
| 13 | Merge deletes exactly the two tests/fomo files | ✓ VERIFIED | `--diff-filter=D` output |
| 14 | Concurrency: no commit landed while MERGE_HEAD existed | ✓ VERIFIED | HEAD reflog: 088f73b (09:43) → e12158c merge commit (10:19) with nothing between |
| 15 | pyproject after merge (D-04, D-10) | ✓ VERIFIED | deps carry tomtoolkit>=3.1.0, tom_jpl>=0.3.0, timezonefinder>=6.0, no tom-registration; dev has coverage, ruff>=0.16, graphifyy, no pytest/ruff pin; no pytest/black/isort tables; unchanged since e12158c |
| 16 | Dev venv on 3.1.0 / 0.3.0, tom-registration gone, pip check clean | ✓ VERIFIED | Verifier read importlib metadata: tomtoolkit 3.1.0, tom_jpl 0.3.0, tom-registration absent; pip check clean |
| 17 | Merged tree boots: only urls.W005; no migration drift | ✓ VERIFIED | Verifier ran `manage.py check` (only urls.W005) and `makemigrations --check --dry-run` ("No changes detected") |
| 18 | SYNC-03 ordering: style commit before the fix commit, pure reformat | ✓ VERIFIED | ff8dd3c precedes 398fa26; verifier re-ran the AST/notebook-output check: 12 files, all AST-identical / outputs identical |
| 19 | D-09: SIM103 fixed in code, behavior-preserving; `[tool.ruff.lint]` unchanged | ✓ VERIFIED | backfill_lco_observations.py:176; pyproject identical to e12158c; test_backfill_lco_observations passes (verifier run) |
| 20 | D-05/SYNC-05: CI files differ from main by one line each | ✓ VERIFIED | `git diff --numstat origin/main -- .github/workflows` → `1 1` smoke-test.yml, `1 1` testing-and-coverage.yml; functional job keeps `--tag functional` |
| 21 | D-06/D-07: django-test hook entry, SKIP note, notebook hook repointed, clear-output exclude kept, no pytest/sphinx hook | ✓ VERIFIED | .pre-commit-config.yaml:20, 68, 71, 80, 86; no `id: pytest`/`id: sphinx` |
| 22 | `@tag('ephemeris_segfault')` on exactly one class | ✓ VERIFIED | solsys_code/tests/test_views.py:98 only |
| 23 | Eight pre-executed notebooks `nbsphinx.execute == never`, metadata commit cell-identical | ✓ VERIFIED | Verifier check: 8/8; 1ac9a50 changed 8 notebooks, cells identical |
| 24 | CLAUDE.md, installation.rst, settings in step (D-04, D-08) | ✓ VERIFIED | CLAUDE.md:36 `enforces (v0.16.9)`, exact pre-commit lines, no bare-ruff line, 8 `ephemeris_segfault` mentions, Testing section describes only the Django runner, Key Dependencies lines; installation.rst:22 `timezonefinder>=6.0`, no registration; settings.py has no tom_registration app or middleware |
| 25 | Runbook system-check paragraph stays true | ✓ VERIFIED | `manage.py check` reports only urls.W005, which the runbook already describes |
| 26 | SYNC-05 local proof: the django-test hook (CI form) passes | ✓ VERIFIED | hook log: `Run Django unit tests (excluding functional and ephemeris_segfault)...Passed`, `Ran 2170 tests`, exact `OK`, no FAILED/skipped= |
| 27 | Developer DB migrated behind a backup; crontab restored | ✓ VERIFIED | backup src/fomo_db_20261007_pre_phase38.sqlite3 present and git-ignored; `migrate --check` exit 0 (verifier run); `crontab -l` byte-identical to `$HOME/tmp/phase38-crontab.bak`, no `#PHASE38-PAUSED` line |
| 28 | 38-OVERRIDE-COMPARISON.md covers the three tom_calendar overrides and every other override | ✓ VERIFIED | Verifier listed every src/templates file that shadows an installed tomtoolkit template: calendar.html, event_form.html, tom_common/index.html, tom_targets/partials/module_buttons.html. All four are in the note, plus calendar_urls.py; the conclusion is explicit |
| 29 | D-11: code-only refresh = merge (2nd parent origin/main) + snapshot equal to the v2.5 tree minus `.planning/`; no force-push; no leaks | ✓ VERIFIED | 8ef5445 parents 372d02c, a910c17; 1a68a76 single parent 8ef5445; `git diff --quiet HEAD origin/issue37-code-only -- . ':(exclude).planning'` equal; 5a1f27e ancestor of origin tip; remote-tracking reflog shows one plain update per branch on 2026-10-07; ls-tree has no `.planning/`, `*.sqlite3`, `local_settings.py` or `reqgroup_*`; three-dot shortstat 176 files |
| 30 | Worktree removed; primary checkout on the v2.5 branch | ✓ VERIFIED | `git worktree list` shows only the primary checkout, on issue37-telescope-runs-calendar |

**Score:** 29/30 truths verified (0 present, behavior-unverified)

### Prohibitions (ADR-550)

| Prohibition | Tier | Disposition | Evidence |
|-------------|------|-------------|----------|
| No branch commit rewritten (38-01) | test | ✓ held | Ancestry of fb07a66 / 088f73b / 75ad2be; HEAD reflog has no amend, rebase or reset |
| No floor raised beyond main (38-01) | test | ✓ held | Versioned-line diff check vs origin/main: only timezonefinder>=6.0 |
| Nothing pushed in 38-01 | test | ✓ held | origin/issue37-telescope-runs-calendar reflog: 75ad2be (10-06 21:15) → 62605b8 (10-07 11:41, Plan 04); no push near the 10:19 merge |
| No behavior-changing ruff fix without sign-off; ignore list not grown (38-02) | test | ✓ held | Single SIM103 truth-table rewrite; pyproject identical to e12158c |
| No workflow trigger/action/secret change (38-02) | test | ✓ held | numstat shows only two one-line changes |
| No notebook cell/output change in metadata commit (38-02) | test | ✓ held | Cell-identity check on 1ac9a50 |
| No test newly skipped/tagged/deleted (38-03) | test | ✓ held | Skip-marker diff check; deletions limited to tests/fomo/* (legacy pytest, SYNC-06) |
| No force-push; PR never readied/merged; nothing private published (38-04) | test | ✓ held | Ancestry + reflog; isDraft true; leak scan |
| Package-legitimacy gate before install (38-01) | judgment | ⚠ flagged (non-authoritative: likely held) | SUMMARY records "approve" verbatim; install order not reconstructible |
| DB backup before migrate, never with cron active (38-03) | judgment | ⚠ flagged (non-authoritative: likely held) | Backup mtime 11:10, after the merge; crontab restored afterwards |
| No live portal/email/heartbeat from an executor (38-03) | judgment | ⚠ flagged (non-authoritative: no contrary evidence) | No such command named in the SUMMARY |
| Wheels only unzipped/diffed (38-03) | judgment | ⚠ flagged (non-authoritative: no contrary evidence) | Described in the SUMMARY and the note |
| Primary checkout never switched (38-04) | judgment | ⚠ flagged (non-authoritative: held) | HEAD reflog has no checkout entries on 2026-10-07 |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/apps.py` | one nav_items with both menus | ✓ VERIFIED | contains `partials/navbar_list.html`; runtime checked |
| `solsys_code/admin.py` | SolsysTargetAdmin as the single Target admin | ✓ VERIFIED | line 624 |
| `src/fomo/urls.py` | main's Scout routes + branch routes before tom_common | ⚠ VERIFIED with defect | wiring present; also carries the stale `alerts/` include (CR-01) |
| `docs/conf.py` | CR-03 guard + `_version` skip hook | ✓ VERIFIED | |
| `solsys_code/tests/test_bootstrap5_rendering.py` | `from django.test import SimpleTestCase, tag` | ✓ VERIFIED | line 26 |
| `pyproject.toml` | `"ruff>=0.16"` and main's floors | ✓ VERIFIED | |
| `38-MERGE-RESOLUTION.diff` | both views | ✓ VERIFIED | present; shows the alerts lines re-added in view 1 |
| `.pre-commit-config.yaml` | D-06 entry | ✓ VERIFIED | line 86 |
| `.github/workflows/testing-and-coverage.yml` | coverage run with both exclusions | ✓ VERIFIED | line 41 |
| `.github/workflows/smoke-test.yml` | both exclusions | ✓ VERIFIED | line 43 |
| `backfill_lco_observations.py` | SIM103 rewrite | ✓ VERIFIED | line 176 |
| `CLAUDE.md` | `enforces (v0.16.9)` | ✓ VERIFIED | line 36 |
| `docs/installation.rst` | `* timezonefinder>=6.0` | ✓ VERIFIED | line 22 |
| `38-OVERRIDE-COMPARISON.md` | names event_form.html | ✓ VERIFIED | 42 lines; tables plus a conclusion |
| `38-PR43-BODY.md` | runbook path | ✓ VERIFIED | matches the published body |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| src/fomo/urls.py | solsys_code/views.py | `ProtectedUserDeleteView.as_view(), name='user-delete'` | ✓ WIRED (TestUserDeleteView passes) |
| src/fomo/urls.py | solsys_code/scout_views.py | `RubinTooScoutListView.as_view()` | ✓ WIRED |
| solsys_code/admin.py | tom_targets.admin.TargetAdmin | `class SolsysTargetAdmin(TargetAdmin)` | ✓ WIRED |
| src/fomo/settings.py | tom_common.default_settings | `TOMTOOLKIT_AUTHENTICATION_BACKENDS` | ✓ WIRED (settings import on 3.1.0; check passes) |
| .pre-commit-config.yaml | test_views.py | `--exclude-tag ephemeris_segfault` → one tagged class | ✓ WIRED |
| pre-commit-ci.yml | .pre-commit-config.yaml | repointed notebook hook passes in CI | ✓ WIRED (pre-commit run success on 1a68a76) |
| CLAUDE.md | .planning/codebase/STACK.md | generated stack block | ✓ WIRED (per 38-02; STACK.md not re-read) |
| 38-OVERRIDE-COMPARISON.md | solsys_code/calendar_urls.py | diff vs 3.1.0 urls.py | ✓ WIRED |
| django-test hook | testing-and-coverage.yml | same command | ✓ WIRED |
| issue37-code-only snapshot | issue37-telescope-runs-calendar | read-tree snapshot | ✓ WIRED (tree-equal outside .planning) |
| PR #43 | testing-and-coverage.yml | pull_request event | ✓ WIRED (runs on 1a68a76) |
| src/fomo/urls.py | tom_alerts.urls | `include('tom_alerts.urls', namespace='alerts')` | ✗ STALE: wired to an app that is no longer installed (CR-01) |

### Data-Flow Trace (Level 4)

Not applicable. The phase changes tooling, dependencies and merge wiring; it adds no new dynamic-data rendering. The URL-wiring runtime checks above stand in for it.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Enforced ruff gate clean | `pre-commit run ruff/ruff-format --all-files` | Passed / Passed, no file changed | ✓ PASS |
| Django boots on 3.1.0 | `manage.py check` | only urls.W005 | ✓ PASS |
| No migration drift | `makemigrations --check --dry-run` | No changes detected | ✓ PASS |
| Dev DB on 3.1.0 schema | `migrate --check` | exit 0 | ✓ PASS |
| Merge wiring + SIM103 module | `manage.py test solsys_code.tests.test_admin ...TestUserDeleteView solsys_code.tests.test_backfill_lco_observations` | Ran 219 tests, OK | ✓ PASS |
| nav_items union | `manage.py shell -c ...nav_items()` | both partials | ✓ PASS |
| alerts route removed as main has it | `resolve('/alerts/query/list/')`, `apps.is_installed('tom_alerts')` | resolves to alerts:list; app not installed; template missing | ✗ FAIL |
| Full suite (fresh venv) | log tail (not re-run, per instructions) | Ran 2178, OK | ✓ PASS (log evidence) |
| CI form (dev venv) | hook log tail | Passed, Ran 2170, OK | ✓ PASS (log evidence) |
| GitHub CI on PR push | `gh run list/view` | 3 workflows success; Django runner step in 3 build jobs | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. No probe scripts are declared, and none exist under `scripts/*/tests/probe-*.sh`.

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| SYNC-01 | 38-01 | ✓ SATISFIED | Truths 1, 7, 14. History-level: every main commit is on the branch. The content-level revert of ada2000's urls.py hunk is CR-01, tracked under SYNC-04 and the goal. |
| SYNC-02 | 38-01, 38-03 | ✓ SATISFIED | Truths 2, 15, 16 |
| SYNC-03 | 38-02 | ✓ SATISFIED | Truths 3, 18, 19, 24 |
| SYNC-04 | 38-01 | ✗ BLOCKED (partial) | Template v2.2.0 files are present and no FOMO file was lost (truth 4). But the repo does not follow main "as main does" for the tomtoolkit 3.1.0 URL wiring (truth 6, CR-01). |
| SYNC-05 | 38-02, 38-03, 38-04 | ✓ SATISFIED | Truths 4, 20, 26; CI form per locked D-05 (see WR-02) |
| SYNC-06 | 38-02 | ✓ SATISFIED | Truths 4, 21, 24 |
| SYNC-07 | 38-03 | ✓ SATISFIED | Truth 2 (fresh suite), 26; nothing skipped. The suite passing does not cover CR-01 because no test exercises alerts/. |
| SYNC-08 | 38-04 | ✓ SATISFIED | Truth 5 |

No orphaned requirements: REQUIREMENTS.md maps exactly SYNC-01..08 to Phase 38, and the plans claim all eight. (REQUIREMENTS.md already ticks SYNC-04 as complete; that should be revisited after the CR-01 fix.)

### Code Review Findings (38-REVIEW.md) — Classification

| ID | Classification | Reasoning |
|----|----------------|-----------|
| CR-01 | **Must-have gap (BLOCKER)**, truth 6 / SYNC-04 | The kept lines were not "the branch's side" in D-03's sense. The branch never touched them; they are base content main deleted as part of its tomtoolkit 3.1.0 update. Keeping them reverts a main change, which contradicts the goal's "carries everything main already has". The 38-01 truth that lists `'alerts/'` encodes a planning error, so passing that plan check is not evidence against the gap. The headline's "every page view 500s" is overstated; the impact is the alerts/ URL space. Not covered by any later phase (39-42), so it is not deferred. |
| WR-01 | Human decision (documentation follow-up before PR #43 leaves draft) | `fomo.local_settings` is deliberate pre-existing branch behavior (c0f883d, WR-32) that auto-merged because main never changed those lines. It is not a sync failure. The gap is the undocumented location for main-layout hosts (installation.rst, PR body). |
| WR-02 | Human decision | The outcome follows exactly from locked D-05. Restoring CI coverage of TestEphemeris (for example, a separate non-blocking step) is the developer's choice. |
| IN-01 | Info / out of scope | Stale-reason `suppress_warnings`. The comment was rewritten in 38-02, as planned. |
| IN-02 | Info | Stale CLAUDE.md lines (Django 2.1+, Bootstrap 4, tom_fink floor, black/isort mentions) outside the lines the plans targeted. Worth a doc follow-up because subagents read CLAUDE.md as instructions. |
| IN-03 | Info | ruff `exclude` without `force-exclude` came from the auto-merge union, and the gate passes anyway. |
| IN-04 | Info | setuptools floor vs PEP 639 license. Both sides already had it; builds with isolation work. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/fomo/urls.py | 32-35 | Route to an uninstalled app; comment states a false premise | 🛑 Blocker | alerts/ pages raise TemplateDoesNotExist; on a fresh DB the tom_alerts tables are never created; the PR diff re-adds what main removed |
| docs/_build/html/autoapi/fomo/local_settings/ (untracked, git-ignored, mtime 2026-09-17) | — | Stale local Sphinx build that rendered the real local_settings values, including an API key | ℹ️ Info | Not tracked, not in the snapshot, predates the phase (pre-CR-03 build). Deleting `docs/_build/` locally is advisable. |

No unreferenced TBD/FIXME/XXX markers were added in any non-.planning file since 088f73b.

### Human Verification Required

These are listed for the developer. They do not change the gaps_found status.

1. **WR-01 local_settings location.** Decide whether to document it (installation.rst + PR #43 body) or accept both import locations.
2. **WR-02 TestEphemeris CI coverage.** Decide whether to add a separate, possibly non-blocking, `--tag ephemeris_segfault` CI step.
3. **Judgment-tier prohibitions.** Confirm the package gate, the order of backup, cron pause and migrate, that no live portal calls were made, the wheel handling, and that the primary checkout never switched. Repository evidence is consistent with all five.

### Gaps Summary

There is one gap, and it has a single root cause. The 38-01 resolution table treated the `alerts/` include in `src/fomo/urls.py` as a branch route to keep. In fact it was shared base content that main removed in ada2000, when main moved to tomtoolkit 3.1.0 and dropped `tom_alerts` from the installed apps. As a result:

- The merged branch, and the published PR #43 head, route `/alerts/...` into an app that is no longer installed.
- Those views fail with TemplateDoesNotExist (and with missing tables on a fresh database).
- The branch does not carry main's tomtoolkit 3.1.0 URL change, against the phase goal and SYNC-04.

Everything else checked out:

- The merge shape (D-01/D-02) is correct.
- Floors, ruff 0.16.9, LINCC v2.2.0 and the CI/hook test form are in place.
- The full suite is green in a fresh venv and in the CI form.
- GitHub CI is green on Python 3.10, 3.11 and 3.12.
- PR #43 is still a draft with the D-12 body.

Fix: delete the four lines, add a regression test that `/alerts/query/list/` returns 404, and re-snapshot issue37-code-only with a plain push. Then re-verify.

---

_Verified: 2026-10-07T19:45:00Z_
_Verifier: Claude (gsd-verifier)_
