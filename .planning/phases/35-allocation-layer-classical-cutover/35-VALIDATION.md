---
phase: "35"
slug: "allocation-layer-classical-cutover"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-12"
---

# Phase 35 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django built-in test runner (`django.test.TestCase` via `python manage.py test`) — the project's only functioning suite |
| **Config file** | none — Django test discovery via `manage.py test <labels>`; `pyproject.toml`'s pytest config is legacy/unused for this app |
| **Quick run command** | `python manage.py test solsys_code.tests.test_allocation_projector` (new file) or `python manage.py test solsys_code.tests.test_load_telescope_runs` for ingest changes |
| **Full suite command** | `LABELS=$(ls solsys_code/tests/test_*.py solsys_code/solsys_code_observatory/tests/test_*.py \| grep -v "tests/test_views\.py$" \| sed "s\|/\|.\|g; s\|\.py$\|\|" \| tr "\n" " "); python manage.py test $LABELS && python manage.py test solsys_code.tests.test_views.TestSplitNumberUnitRegex solsys_code.tests.test_views.TestJPLSBDBQuery` |
| **Estimated runtime** | ~60 seconds per module (SPICE kernel load dominates); full label-list suite several minutes |

---

## Sampling Rate

- **After every task commit:** Run the relevant single test module/class (e.g. `python manage.py test solsys_code.tests.test_allocation_projector`)
- **After every plan wave:** Run the full label-list suite command above
- **Before `/gsd-verify-work`:** Full suite must be green, plus `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files`
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 35-XX-XX | TBD | TBD | ALLOC-01 | T-35-XX / — | per-night events only for resolved-site awarded windows; queue/class-wide/satellite runs keep one container | unit | `python manage.py test solsys_code.tests.test_allocation_projector` | ❌ W0 | ⬜ pending |
| 35-XX-XX | TBD | TBD | ALLOC-02 | — | site-local observing-night keying for a Chilean and an Australian site | unit | `python manage.py test solsys_code.tests.test_allocation_projector` | ❌ W0 | ⬜ pending |
| 35-XX-XX | TBD | TBD | ALLOC-03 | T-35-XX / — | link retires the night, unlink restores it, observation event byte-unchanged | unit + integration | `python manage.py test solsys_code.tests.test_allocation_projector solsys_code.tests.test_observation_projector_signals` | ❌ W0 / ✅ | ⬜ pending |
| 35-XX-XX | TBD | TBD | ALLOC-04 | T-35-XX / — | `load_telescope_runs` writes a campaign-less `CampaignRun` with collision-safe `source_identifier`; re-import is a no-op | integration | `python manage.py test solsys_code.tests.test_load_telescope_runs` | ✅ | ⬜ pending |
| 35-XX-XX | TBD | TBD | ALLOC-05 | T-35-XX / — | cutover never leaves a duplicate or orphan; unexplained events left untouched and reported | integration + notebook | `python manage.py test solsys_code.tests.test_cutover_classical_allocations` (name TBD by planner) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*The planner fills real task IDs / plan / wave once PLAN.md files exist; the rows above are the requirement-level contract from RESEARCH.md `## Validation Architecture`.*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_allocation_projector.py` — new file, covers ALLOC-01/02/03; needs a campaign-less (`campaign=None`) `CampaignRun` fixture with a resolved Chilean site alongside the existing `CampaignReconcilerTestBase` pattern
- [ ] A Chilean `Observatory` test fixture (`timezone='America/Santiago'`) — the existing reconciler test base fixtures only an Australian site (`F65`); ALLOC-02 requires both hemispheres, and the test must create its own `Observatory` row rather than depend on dev-DB content
- [ ] Framework install: none — `django.test.TestCase` and `tom_targets.tests.factories.NonSiderealTargetFactory` are already available project-wide

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cutover against the real dev DB leaves one event per night (241 events / 56 `RUN:{pk}:{date}` / 0 `ALLOC:` baseline) | ALLOC-05 | Real-data before/after diff; the unit test proves the mechanism on fixtures, not the operator's actual calendar | Run the cutover command with `--dry-run`, then for real, and record before/after counts in `reconcile_campaign_runs_demo.ipynb` (paired-docs obligation) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
