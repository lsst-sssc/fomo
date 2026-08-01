---
phase: 28
slug: operator-assisted-attribution
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-01
---

# Phase 28 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase` / `TransactionTestCase`) |
| **Config file** | none dedicated — `pyproject.toml` `testpaths` deliberately excludes `solsys_code/`, so app tests run under the Django runner, not pytest (CLAUDE.md testing split) |
| **Quick run command** | `python manage.py test solsys_code.tests.test_campaign_attribution` |
| **Full suite command** | `python manage.py test solsys_code` (exclude `test_views.TestEphemeris` — segfaults in native ASSIST, unrelated to this phase) |
| **Estimated runtime** | ~10 s quick / ~2–4 min full |

**Invocation caveat:** use `python manage.py`, never `./manage.py`.

---

## Sampling Rate

- **After every task commit:** Run `python manage.py test solsys_code.tests.test_campaign_attribution` plus the specific new test file(s) that task touched
- **After every plan wave:** Run `python manage.py test solsys_code` (excluding `TestEphemeris`)
- **Before `/gsd-verify-work`:** Full suite green, plus `ruff check .` and `ruff format --check .` clean
- **Max feedback latency:** ~10 seconds for the quick command

---

## Per-Task Verification Map

Populated by the planner — each plan task must map to a row here. Requirement→test
coverage is fixed by the table below; task IDs are assigned during planning.

| Req ID | Behavior | Test Type | Automated Command | File Exists |
|--------|----------|-----------|-------------------|-------------|
| ATTRIB-01 | Worklist shows evidence columns (telescope, date overlap, campaign, instrument) per candidate — never a bare score | unit + view | `python manage.py test solsys_code.tests.test_campaign_attribution_views.TestEvidenceColumns` | ❌ W0 |
| ATTRIB-02 | Candidates confidence-scored and filterable by named band | unit | `python manage.py test solsys_code.tests.test_campaign_attribution.TestScoringAndBanding` | ❌ W0 |
| ATTRIB-03 | No association without explicit confirmation; no cross-campaign/target suggestion ever offered | unit + view | `python manage.py test solsys_code.tests.test_campaign_attribution.TestCampaignBoundaryGate` | ❌ W0 |
| ATTRIB-04 | Confirm and undo both attributable to a person and a time (event side *and* record side) | view + model | `python manage.py test solsys_code.tests.test_campaign_attribution_views.TestConfirmUndo` | ❌ W0 |
| ATTRIB-05 | The real criterion-5 case is surfaced despite mismatched instrument strings and the one-day span difference | integration | `python manage.py test solsys_code.tests.test_campaign_attribution.TestCriterion5RealCase` | ❌ W0 |
| ATTRIB-06 | Queue drains to zero and states the remaining count, before any reconcile sweep | view (end-to-end) | `python manage.py test solsys_code.tests.test_campaign_attribution_views.TestQueueDrainsToEmpty` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_campaign_attribution.py` — matcher unit tests: the weighted-sum
      scoring formula, band cut-points, the campaign/target boundary as the single hard gate,
      and instrument-string similarity. Must cover the measured real case directly:
      `difflib.SequenceMatcher` on `"FTS/MuSCAT4"` vs `"2M0-SCICAM-MUSCAT"` ratios **0.500**,
      below this codebase's own 0.6 fuzzy cutoff — proof that "instrument similarity must never
      disqualify" (D-11) is load-bearing, not precautionary.
- [ ] `solsys_code/tests/test_campaign_attribution_views.py` — view/POST integration tests:
      confirm on the event side (atomic conditional `.update()`), confirm on the record side
      (`get_or_create` + `IntegrityError`), dismiss, undo, double-submit no-op, the two-staff
      race on both link types, and `StaffRequiredMixin` gating.
- [ ] Criterion-5 acceptance test built as an equivalent fixture, **not** against live pks
      53/58 verbatim — research confirmed those two rows are already claimed, leaving 10 genuine
      orphans per side. Use `NonSiderealTargetFactory` (never `SiderealTargetFactory`) following
      the fixture style in `test_campaign_run_observation.py`.
- [ ] Extend `solsys_code/tests/test_admin.py`'s `CampaignRunAdminInlinesTests` with
      `test_save_formset_stamps_calendar_event_meta_on_run_transition` — the `CalendarEventMeta`
      branch needs a `run_id` None→not-None transition check, **not** the `pk is None` check the
      `CampaignRunObservation` branch uses.
- [ ] No framework install needed — `django.test.TestCase` / `TransactionTestCase` cover every
      case above. No new dependency is permitted (`rapidfuzz` rejected twice; `difflib` is the tool).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Evidence columns are legible side by side and the score reads as *additional* to the evidence, not a replacement | ATTRIB-01 | Visual judgement against UI-SPEC.md; no assertion can prove "staff can sanity-check the banding" | Load the attribution page as staff, confirm each candidate row renders matched telescope, date overlap, campaign and instrument similarity beside the numeric score and its band |
| The high-band cut-point is tight enough that multi-select is not silent guessing | ATTRIB-02, D-09 | The residual risk CONTEXT.md states explicitly — a correctness decision that only inspection of real candidates can settle | Filter to the High band on real data, inspect every checkboxable row, confirm each is one a human would confirm unhesitatingly |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s for the quick command
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
