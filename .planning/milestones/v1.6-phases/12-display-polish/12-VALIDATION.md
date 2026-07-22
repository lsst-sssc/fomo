---
phase: 12
slug: display-polish
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-27
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django TestCase (`./manage.py test`) |
| **Config file** | `pyproject.toml` (`testpaths = ["tests", "src", "docs"]`) — Django app tests via `./manage.py test solsys_code` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template`
- **After every plan wave:** Run `./manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | DISPLAY-08 | — | Returns only '#fff' or '#000'; input is palette constants only, no user data path | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras` | ✅ | ⬜ pending |
| 12-01-02 | 01 | 1 | DISPLAY-08 | — | All 8 PROPOSAL_PALETTE colors return '#fff' (WCAG AA luminance ≤ 0.183) | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras` | ✅ | ⬜ pending |
| 12-01-03 | 01 | 1 | DISPLAY-08 | — | NEUTRAL_SLOT_COLOR (#5a6268) returns '#fff' | unit | `./manage.py test solsys_code.tests.test_calendar_display_extras` | ✅ | ⬜ pending |
| 12-01-04 | 01 | 1 | DISPLAY-08 | — | All-day event divs in calendar.html render inline `color:` style (no `!important` in CSS rules) | integration | `./manage.py test solsys_code.tests.test_calendar_template` | ✅ | ⬜ pending |
| 12-01-05 | 01 | 1 | DISPLAY-09 | — | Query count is bounded ≤ constant regardless of event count (no N+1 for telescope_label_meta) | integration | `./manage.py test solsys_code.tests.test_calendar_template` | ✅ | ⬜ pending |
| 12-01-06 | 01 | 1 | DISPLAY-09 | — | `active_todo_count` Count annotation renders correctly; `event.active_todos.count` no longer in template | integration | `./manage.py test solsys_code.tests.test_calendar_template` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No new test files needed — new test methods are added to existing test classes only:

- `solsys_code/tests/test_calendar_display_extras.py` — add `TextColorForBgTest` class (parametrized palette + NEUTRAL_SLOT_COLOR tests)
- `solsys_code/tests/test_calendar_template.py` — add `assertNumQueries` N+1 regression test

Pre-existing quality gate: `ruff check . && ruff format --check .` must pass after every task commit.

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
