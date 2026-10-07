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

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

All 36 automated verify commands across the ten executed tasks carry a `<fails_when>` direction (`gsd_run check verify-failure-directions 38` → status ok, 0 non-ok) and every SUMMARY reports `Self-Check: PASSED`.

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI runs the Django test runner with coverage on a push | SYNC-05 | GitHub Actions runs remotely; the workflows trigger only on `push` to `main` and `pull_request` to `main` | Done in 38-04 Task 3 (automated via `gh run list/view` against snapshot `1a68a76`): "Unit test and code coverage" ran `build (3.10/3.11/3.12)` + `functional-tests` with the `Run Django unit tests with coverage` step, no pytest job; "Run pre-commit hooks" and "Build documentation" also succeeded. To re-check by hand: open the Actions tab for PR #43's head commit |
| Developer approval of the staged merge resolution and of publishing PR #43 | SYNC-01, SYNC-08 | blocking-human checkpoints (D-01, D-11) — a judgment, not a test | Answers recorded verbatim in 38-01-SUMMARY.md (`approve`) and 38-04-SUMMARY.md (`publish`) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (the two checkpoint tasks are human gates with recorded answers)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (none)
- [x] No watch-mode flags
- [x] Feedback latency < 600s (targeted runs ≤ 60 s; the full suite ran 457–550 s)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-10-07 by the execute-phase verify:post hook (State A audit; 0 gaps, 0 escalations)

## Validation Audit 2026-10-07

| Metric | Count |
|---|---|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Automated commands | 36 |
| Executed tasks | 10 |
| Checkpoint tasks | 2 |
