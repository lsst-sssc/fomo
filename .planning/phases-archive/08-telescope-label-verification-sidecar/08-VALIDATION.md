---
phase: 8
slug: telescope-label-verification-sidecar
status: final
nyquist_compliant: true
wave_0_complete: true
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

> Reconciled against the finalized 08-01-PLAN.md (Wave 1) and 08-02-PLAN.md (Wave 2).
> Each plan task carries an `<automated>` verify command; rows below map task → requirement → command.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| Task 1 | 01 | 1 | DISPLAY-01 | T-08-02 | Sidecar model + first migration exist; reverse accessor `telescope_label_meta` resolves; CASCADE delete prevents orphan rows | migration/model integration | `./manage.py makemigrations solsys_code --check --dry-run; ./manage.py migrate solsys_code; python -c "from solsys_code.models import CalendarEventTelescopeLabel ..."` | ✅ planned | ⬜ pending |
| Task 2 | 01 | 1 | DISPLAY-01 | T-08-01 | Sync writes a sidecar row matching `telescope_api_failed` outcome (verified→is_verified True, fallback→False); no row for `load_telescope_runs` events; re-run creates no duplicate / no churn | unit/integration | `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_load_telescope_runs -v2` | ✅ planned | ⬜ pending |
| Task 3 | 01 | 1 | DISPLAY-01 | T-08-03 | Demo notebook re-executed, references `CalendarEventTelescopeLabel`, shows verified + fallback + no-row cases, committed with output | notebook-execution / convention | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/sync_lco_observation_calendar_demo.ipynb && python -c "... assert 'CalendarEventTelescopeLabel' in src ..."` | ✅ planned | ⬜ pending |
| Task 1 | 02 | 2 | DISPLAY-02, DISPLAY-03 | T-08-04 | Fallback event renders 2px dashed border + `title=` tooltip on both all-day and timed branches; verified / no-row events unstyled; `== False` idiom only | template (static assertion) | `python -c "s=open('src/templates/tom_calendar/partials/calendar.html').read(); assert s.count('is_verified == False')==2 ... assert 'title=' in s and 'estimate' in s"` | ✅ planned | ⬜ pending |
| Task 2 | 02 | 2 | DISPLAY-02, DISPLAY-03 | T-08-06 | Calendar page renders 200; fallback events show dashed border + tooltip; verified/no-row events excluded; no-sidecar-row event does not 500 (A1) | template-rendering / view-level integration | `./manage.py test solsys_code.tests.test_calendar_template -v2` | ✅ planned | ⬜ pending |

*File Exists: ✅ planned = task defined in PLAN.md with concrete `<automated>` verify · ❌ = absent · Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

No dedicated Wave-0 scaffolding tasks exist — every task in the finalized plans carries a real,
runnable `<automated>` verify command (no `MISSING — Wave 0 must create ...` placeholders). The two
infrastructure prerequisites originally flagged here are each satisfied inline by a Wave-1 task rather
than a separate Wave-0 step, so `wave_0_complete` is vacuously true:

- [x] **First-ever migration for `solsys_code`** — handled by **Plan 01, Task 1 (Wave 1)**, whose verify
  runs `./manage.py makemigrations solsys_code --check --dry-run` and `./manage.py migrate solsys_code`,
  proving the generated migration applies clean. Not assumed.
- [x] **First `calendar.html` template-rendering test** — established by **Plan 02, Task 2 (Wave 2)**,
  which creates `solsys_code/tests/test_calendar_template.py` using `django.test.Client` against real
  `CalendarEvent`/`CalendarEventTelescopeLabel` fixtures, per the
  `solsys_code/solsys_code_observatory/tests/test_views.py` precedent.

*`assertNumQueries` is NOT required for this phase — N+1 mitigation is deferred per CONTEXT.md's locked decision (accept-as-is, DISPLAY-09 deferred to v2).*

---

## Manual-Only Verifications

All phase behaviors have automated verification (template-rendering test covers the visual cue and tooltip; Django test suite covers the sidecar write/no-churn/migration behaviors).

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — all 5 tasks across both plans carry a concrete `<automated>` command (no `MISSING` placeholders).
- [x] Sampling continuity: no 3 consecutive tasks without automated verify — every task has automated verify, so there is no run of unverified tasks at all.
- [x] Wave 0 covers all MISSING references — vacuously satisfied: there are no `MISSING — Wave 0 must create ...` references in either plan.
- [x] No watch-mode flags — no `--watch`/`-w`/watch-mode commands; all verifies are single-shot (`./manage.py test`, `makemigrations`/`migrate`, `python -c`, one-shot `jupyter nbconvert`).
- [x] Feedback latency < 60s — satisfied for all task-commit and wave commands. The Django test commands (`./manage.py test solsys_code`) run in ~30–60s with no network/SPICE I/O. **Exception (acknowledged, acceptable):** Plan 01 Task 3's `jupyter nbconvert --to notebook --execute` is the one command expected to exceed 60s (notebook kernel startup + cell execution, est. ~60–120s). This is a once-per-phase pre-`/gsd-verify-work` notebook regeneration, not a per-task-commit sampling command, so it does not break the inner feedback loop; the fast inner-loop command stays `./manage.py test solsys_code.tests.test_sync_lco_observation_calendar`.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved (reconciled against finalized 08-01-PLAN.md / 08-02-PLAN.md)
