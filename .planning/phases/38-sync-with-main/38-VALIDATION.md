---
phase: "38"
slug: "sync-with-main"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| 38-01-01 | 01 | 1 | SYNC-01 | — | N/A | integration | `git merge-base --is-ancestor origin/main HEAD` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI runs the Django test runner with coverage on a push | SYNC-05 | GitHub Actions runs remotely; the workflows trigger only on `push` to `main` and `pull_request` to `main` | Push the refreshed PR #43 branch and open the Actions tab; confirm the `testing-and-coverage` job runs `python manage.py test` and no pytest job exists |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 600s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
