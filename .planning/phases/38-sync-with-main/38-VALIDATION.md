---
phase: "38"
slug: "sync-with-main"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-10-07"
---

# Phase 38 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`) |
| **Config file** | `manage.py` (sets `DJANGO_SETTINGS_MODULE=src.fomo.settings`); no pytest config after the merge |
| **Quick run command** | `python manage.py test solsys_code.tests.test_views --exclude-tag=ephemeris_segfault` |
| **Full suite command** | `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` |
| **Estimated runtime** | ~600 seconds (full suite, 2178 tests on the merged tree) |

---

## Sampling Rate

- **After every task commit:** Run `python manage.py test solsys_code.tests.test_views --exclude-tag=ephemeris_segfault`
- **After every plan wave:** Run `python manage.py test solsys_code --exclude-tag=ephemeris_segfault`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 600 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 38-01-01 | 01 | 1 | SYNC-01, SYNC-04 | T-38-01, T-38-03, T-38-07 | `autoapi_ignore` keeps `*/local_settings.py`; only the nine paths staged; cron line paused behind its flock | integration | 6 `<automated>` blocks: no unmerged path/marker; index equals `git merge-tree` outside the nine paths; both sides wired; AST/URL-order/toctree check; `validate-pyproject` + tomllib; cron paused | ✅ | ✅ green |
| 38-01-02 | 01 | 1 | SYNC-01 | T-38-03, T-38-06 | blocking-human review of the staged diff and the package install | manual (checkpoint:decision) | — (developer answered `approve`; recorded in 38-01-SUMMARY.md) | — | ✅ green |
| 38-01-03 | 01 | 1 | SYNC-01, SYNC-02, SYNC-04 | T-38-02, T-38-05, T-38-06 | `TestUserDeleteView` and admin/search tests run before the commit; `pip check` after install | integration | 7 `<automated>` blocks: versions + `pip check`; `manage.py check` (only urls.W005) + `makemigrations --check`; `nav_items()` order; 7 wiring test labels; parents/ancestry/pure-sync/deletions; template v2.2.0 + ignores + not pushed; full suite exact `OK` (2178) | ✅ | ✅ green |
| 38-02-01 | 02 | 2 | SYNC-03 | T-38-09 | reformat proven AST-identical; SIM103 rewrite behavior-preserving | unit + lint | 3 `<automated>` blocks: both ruff hooks pass twice; style-commit AST/notebook-output identity; SIM103 line + pyproject unchanged + 5 test modules (239 OK) | ✅ | ✅ green |
| 38-02-02 | 02 | 2 | SYNC-05, SYNC-06 | T-38-08, T-38-10, T-38-11 | CI differs from main by exactly two tokens; one `ephemeris_segfault` tag; notebook cells identical | config | 3 `<automated>` blocks: hook-list greps + yaml revs; `git diff --numstat origin/main -- .github/workflows`; notebook hook + `check-github-workflows` + cell-identity script | ✅ | ✅ green |
| 38-02-03 | 02 | 2 | SYNC-03, SYNC-06 | — | N/A | docs | 3 `<automated>` blocks: CLAUDE.md negative/positive greps; Testing section, maps, installation, settings; every `manage.py check` id in the runbook | ✅ | ✅ green |
| 38-03-01 | 03 | 3 | SYNC-02 | T-38-12, T-38-13 | fresh venv `pip check`; backup integrity before migrate; cron paused | integration | 3 `<automated>` blocks: fresh-venv versions + `pip check`; pyproject floor diff; backup ignored + integrity + `migrate --check` + showmigrations | ✅ | ✅ green |
| 38-03-02 | 03 | 3 | SYNC-05, SYNC-07 | T-38-14, T-38-16 | exact `OK` lines, no `skipped=`; no new skip/tag; cron restored only after green | integration | 4 `<automated>` blocks: fresh-venv log (2178 OK); django-test hook log (2170 OK, Passed); no-new-skip diff script; crontab byte-identical to backup | ✅ | ✅ green |
| 38-03-03 | 03 | 3 | SYNC-07 | T-38-15 | wheels unzipped with `python -I -m zipfile`, never imported | docs | 1 `<automated>` block: note content + installed tom_calendar == 3.1.0 wheel | ✅ | ✅ green |
| 38-04-01 | 04 | 4 | SYNC-08 | T-38-17, T-38-21 | leak grep over `git ls-files`; body rejects `/home/` and api_key text | integration | 2 `<automated>` blocks: snapshot tree/ancestry/fast-forward/leak check; PR body D-12 elements | ✅ | ✅ green |
| 38-04-02 | 04 | 4 | SYNC-08 | T-38-17, T-38-18 | blocking-human approval before any push | manual (checkpoint:decision) | — (developer answered `publish`; recorded in 38-04-SUMMARY.md) | — | ✅ green |
| 38-04-03 | 04 | 4 | SYNC-05, SYNC-08 | T-38-18, T-38-19, T-38-20 | fast-forward pushes only; `isDraft` re-checked; worktree removed | integration | 4 `<automated>` blocks: remote tips/ancestry/tree equality; `gh pr view` draft + body; three CI runs success with the coverage step and no pytest; worktree gone | ✅ | ✅ green |
| 38-05-01 | 05 | 5 | SYNC-04, SYNC-07 | T-38-22, T-38-23, T-38-24 | `alerts/` include deleted (main's ada2000); tom_alerts not re-installed; route set = main + calendar/campaigns/user-delete before tom_common | unit + integration (TDD: RED `FAILED (failures=3)` → `RED_EVIDENCE_OK` → GREEN) | 7 `<automated>` blocks: `manage.py test solsys_code.tests.test_urls` (Ran 4, OK); RED evidence classified; route-set/order/AST check; `manage.py check` only urls.W005; TestUserDeleteView + test_scout_views; both ruff hooks; test commit precedes fix, numstat `0 4`, nothing pushed | ✅ | ✅ green |
| 38-05-02 | 05 | 5 | SYNC-04, SYNC-07 | T-38-23, T-38-25 | planning-doc edits limited to six named lines, each marked "corrected by 38-05"; no SUMMARY/REVIEW/VERIFICATION edited | integration (full suite) + docs | 3 `<automated>` blocks: old `alerts/` wording absent + 38-01 frontmatter parses; docs commit numstat 4/4, 1/1, 1/1 in exactly the planned files; full suite log newer than the fix, `Ran 2182 tests`, exact `OK`, no skipped, all four test_urls tests present | ✅ | ✅ green |
| 38-06-01 | 06 | 6 | SYNC-04 | T-38-26, T-38-30 | snapshot tree = v2.5 tree minus `.planning/`; leak grep over `ls-files`; changed paths exactly the two 38-05 files; no edit in the worktree | integration (git) | 2 `<automated>` blocks: one snapshot commit on the PR head with parent = pre-push tip, tree equality, origin/main contained; leak check + no tom_alerts in the snapshot's urls.py + test_urls.py present | ✅ | ✅ green |
| 38-06-02 | 06 | 6 | SYNC-04, SYNC-08 | T-38-27 | blocking-human decision before anything leaves the machine; `ls-remote` still at the recorded tips while waiting | manual (checkpoint:decision) | — (developer answered `publish`; recorded in 38-06-SUMMARY.md) | — | ✅ green |
| 38-06-03 | 06 | 6 | SYNC-04, SYNC-05, SYNC-08 | T-38-27, T-38-28, T-38-29 | plain fast-forward pushes only; `isDraft` re-checked, no `gh pr edit`; branch guard before each push; worktree removed | integration | 4 `<automated>` blocks: both remotes equal local and descend from the recorded tips, PR head tree = v2.5 tree minus `.planning/`, no tom_alerts in the PR's urls.py diff; `gh pr view` draft + D-12 body; three CI runs success on 846be34 with the coverage step and no pytest; worktree gone | ✅ | ✅ green |
| 38-07-01 | 07 | 7 | SYNC-08 | T-38-31, T-38-32, T-38-35 | installation guide, FOMO_BASE_URL note, runbook step 2 and PR body file name `src/fomo/local_settings.py`; docs checked against the live `settings.py` import; no home path or key in added lines; nothing pushed | docs + integration (docutils, `manage.py shell`) | 5 `<automated>` blocks: placement/phrase assertions vs `from fomo.local_settings import *`; docutils parse of both rst files at warning level; `manage.py shell -c "import fomo.local_settings"` prints `.../src/fomo/local_settings.py`; PR-body Settings-line assertions + D-12 sections intact; docs-only diff in the three planned files, numstat 1/1 ×2, ≤120 cols, no `$HOME`/`api_key`, `ls-remote` tips unchanged | ✅ | ✅ green |
| 38-07-02 | 07 | 7 | SYNC-08 | T-38-33, T-38-34 | blocking-human decision before any `gh pr edit`; live body re-compared with the pre-plan file at every stop | manual (checkpoint:decision) | — (developer answered `revise` then `apply`; recorded verbatim in 38-07-SUMMARY.md) | — | ✅ green |
| 38-07-03 | 07 | 7 | SYNC-08 | T-38-33, T-38-34 | exactly one body-only `gh pr edit 43 --body-file`; `isDraft`, head and base re-read afterwards; `settings.py` unchanged; nothing pushed | integration (gh) | 2 `<automated>` blocks: `gh pr view 43` body equals 38-PR43-BODY.md, isDraft true, head `issue37-code-only`, base `main`, required elements present; `settings.py` unchanged vs BASE and `ls-remote` tips equal the recorded ones | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

All 59 automated verify commands across the seventeen executed tasks (36 in 38-01..38-04, 23 in the gap-closure plans 38-05, 38-06 and 38-07) carry a `<fails_when>` direction (`gsd_run check verify-failure-directions 38` → status ok, 0 non-ok) and every SUMMARY reports `Self-Check: PASSED`.

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI runs the Django test runner with coverage on a push | SYNC-05 | GitHub Actions runs remotely; the workflows trigger only on `push` to `main` and `pull_request` to `main` | Done in 38-04 Task 3 (automated via `gh run list/view` against snapshot `1a68a76`): "Unit test and code coverage" ran `build (3.10/3.11/3.12)` + `functional-tests` with the `Run Django unit tests with coverage` step, no pytest job; "Run pre-commit hooks" and "Build documentation" also succeeded. Repeated in 38-06 Task 3 against the gap-closure re-snapshot `846be34` (same three workflows, all success; first CI run of `test_urls.py` on Python 3.10/3.12). To re-check by hand: open the Actions tab for PR #43's head commit |
| Developer approval of the staged merge resolution and of publishing PR #43 | SYNC-01, SYNC-08 | blocking-human checkpoints (D-01, D-11) — a judgment, not a test | Answers recorded verbatim in 38-01-SUMMARY.md (`approve`), 38-04-SUMMARY.md (`publish`), 38-06-SUMMARY.md (`publish`, the gap-closure re-snapshot) and 38-07-SUMMARY.md (`revise`, then `apply`, for the PR #43 body edit) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (the three checkpoint tasks are human gates with recorded answers)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (none)
- [x] No watch-mode flags
- [x] Feedback latency < 600s (targeted runs ≤ 60 s; the full suite ran 457–550 s)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-10-07 by the execute-phase verify:post hook (State A audit; 0 gaps, 0 escalations); re-audited 2026-10-07 after the gap-closure run (38-05, 38-06 added to the map; 0 gaps, 0 escalations); re-audited 2026-10-08 after the G-38-1 gap-closure plan (38-07 added to the map, docs-only; 0 gaps, 0 escalations)

## Validation Audit 2026-10-07

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Automated commands | 36 |
| Executed tasks | 10 |
| Checkpoint tasks | 2 |

## Validation Audit 2026-10-07

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Automated commands | 52 |
| Executed tasks | 14 |
| Checkpoint tasks | 3 |

## Validation Audit 2026-10-08

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Automated commands | 59 |
| Executed tasks | 17 |
| Checkpoint tasks | 4 |
