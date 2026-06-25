---
phase: 8
slug: telescope-label-verification-sidecar
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-25
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`), DB-dependent — per CLAUDE.md, this phase's tests belong in `solsys_code/tests/`, NOT the separate pytest suite (`tests/`, `src/`, `docs/`) |
| **Config file** | none — uses `manage.py test` against `DJANGO_SETTINGS_MODULE=src.fomo.settings` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~30-60 seconds (in-process Django test runner, no network I/O, no SPICE/ASSIST kernel loading) |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`
- **After every plan wave:** Run `./manage.py test solsys_code` + `ruff check .` + `ruff format --check .`
- **Before `/gsd-verify-work`:** Full suite (`./manage.py test solsys_code`) must be green, plus `ruff check .` / `ruff format --check .` clean, plus the paired demo notebook (`docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb`) regenerated via `jupyter nbconvert --to notebook --execute --inplace` and committed with output (CLAUDE.md demo-notebook-companion convention).
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

> Task ID / Plan / Wave columns are assigned by the planner once PLAN.md exists for this phase; rows below are keyed by requirement ID from RESEARCH.md's "Phase Requirements → Test Map" until then.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 0 | DISPLAY-01 | — | Sync writes a sidecar row matching `telescope_api_failed` outcome (verified/fallback) | unit/integration | new test method in `test_sync_lco_observation_calendar.py` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | DISPLAY-01 | — | No sidecar row created for `load_telescope_runs`-created events | unit/integration | new test method in `test_load_telescope_runs.py` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | DISPLAY-01 | — | Re-running sync on unchanged records does not create duplicate sidecar rows or churn `CalendarEvent.modified` | unit/integration (no-churn regression) | new test method mirroring `test_sync_04_rerun_updates_in_place_no_churn_on_unchanged` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | DISPLAY-02 | — | Fallback-labeled event renders with dashed-border CSS (all-day and timed branches); verified event renders with solid/default border | template-rendering / view-level integration | new test file/method asserting rendered HTML | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | DISPLAY-02 | — | A `load_telescope_runs` event (no sidecar row) renders as verified, not as a template error | template-rendering / view-level integration | same new test file, assert no exception + solid-border rendering | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | DISPLAY-03 | — | Hovering a fallback-labeled event shows a tooltip with the verification detail | template-rendering | same new test file, assert `title="..."` substring in rendered HTML | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] No template-rendering test exists yet for `src/templates/tom_calendar/partials/calendar.html` — this phase must establish that pattern (`django.test.Client` against a real `CalendarEvent`/`CalendarEventTelescopeLabel` fixture, consistent with `solsys_code/solsys_code_observatory/tests/test_views.py`'s existing `Client`-based precedent).
- [ ] First-ever migration for `solsys_code` — confirm `./manage.py makemigrations solsys_code` produces a clean migration and `./manage.py migrate` runs clean on a fresh DB as part of Wave 0/the first task, not assumed.

*`assertNumQueries` is NOT required for this phase — N+1 mitigation is deferred per CONTEXT.md's locked decision (accept-as-is, DISPLAY-09 deferred to v2).*

---

## Manual-Only Verifications

All phase behaviors have automated verification (template-rendering test covers the visual cue and tooltip; Django test suite covers the sidecar write/no-churn/migration behaviors).

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
