---
phase: 28
slug: operator-assisted-attribution
status: validated
nyquist_compliant: true
wave_0_complete: true
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
| ATTRIB-01 | Worklist shows evidence columns (telescope, date overlap, campaign, instrument) per candidate — never a bare score | unit + view | `python manage.py test solsys_code.tests.test_attribution_template` (`TestEvidenceColumns`) | ✅ exists | ✅ green |
| ATTRIB-02 | Candidates confidence-scored and filterable by named band | unit | `python manage.py test solsys_code.tests.test_campaign_attribution` (`TestBandFilterAndBanner`, `TestSoleHighCandidateUnderBandFilter`) | ✅ exists | ✅ green |
| ATTRIB-03 | No association without explicit confirmation; no cross-campaign/target suggestion ever offered | unit + view | `python manage.py test solsys_code.tests.test_campaign_attribution_views` (`TestConcurrencyAndTampering`) | ✅ exists | ✅ green |
| ATTRIB-04 | Confirm and undo both attributable to a person and a time (event side *and* record side) | view + model | `python manage.py test solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_admin` (`TestConfirmUndo`, `TestUndoConfirmationOrdering`, `CalendarEventMetaStandaloneAdminAuditStampTests`) | ✅ exists | ✅ green |
| ATTRIB-05 | The real criterion-5 case is surfaced despite mismatched instrument strings and the one-day span difference | integration | `python manage.py test solsys_code.tests.test_campaign_attribution` (`TestCriterion5RealCase`) | ✅ exists | ✅ green |
| ATTRIB-06 | Queue drains to zero and states the remaining count, before any reconcile sweep | view (end-to-end) | `python manage.py test solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_attribution_dismissals` | ✅ exists | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Source:** `28-VERIFICATION.md` (2026-08-02 re-verification, 8/8 truths, 146 tests across 5
modules — `test_campaign_attribution`, `test_campaign_attribution_views`,
`test_attribution_dismissals`, `test_admin`, `test_attribution_template` — independently
re-run by the verifier). CR-01 (Confirm button gated behind Dismiss-only required field) and
CR-02 (standalone admin audit fields unprotected) were BLOCKER findings from the initial pass,
both closed and pinned by mutation-checked regression tests before this phase's final
verification.

---

- [x] `solsys_code/tests/test_campaign_attribution.py` — matcher unit tests present, including
      the measured real-case `difflib.SequenceMatcher` ratio proof.
- [x] `solsys_code/tests/test_campaign_attribution_views.py` — view/POST integration tests
      present (confirm event/record, dismiss, undo, double-submit, race, `StaffRequiredMixin`).
- [x] `TestCriterion5RealCase` present in `test_campaign_attribution.py`, built as an
      equivalent fixture using `NonSiderealTargetFactory`, unmodified since the gap-closure
      round per 28-06's explicit requirement.
- [x] `test_admin.py`'s `test_save_formset_stamps_calendar_event_meta_on_run_transition` and
      the standalone-admin-page audit-stamp tests (`CalendarEventMetaStandaloneAdminAuditStampTests`,
      7 tests, closing CR-02) both present.
- [x] `test_attribution_template.py` added mid-phase (gap closure, 28-05) — 6 structure-only
      tests that parse rendered HTML and never call `self.client.post()`, specifically designed
      to catch CR-01's blind spot (a `<button>` gated behind an unrelated required field), which
      124 previously-green tests had all missed.

**All Wave 0 items closed during execution; the phase's own re-verification (2026-08-02) found
2 BLOCKER regressions (CR-01, CR-02) in the first-pass implementation and confirmed both fixed
with mutation-checked tests before signing off 8/8.**

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Evidence columns are legible side by side and the score reads as *additional* to the evidence, not a replacement | ATTRIB-01 | Visual judgement against UI-SPEC.md; no assertion can prove "staff can sanity-check the banding" | Load the attribution page as staff, confirm each candidate row renders matched telescope, date overlap, campaign and instrument similarity beside the numeric score and its band |
| The high-band cut-point is tight enough that multi-select is not silent guessing | ATTRIB-02, D-09 | The residual risk CONTEXT.md states explicitly — a correctness decision that only inspection of real candidates can settle | Filter to the High band on real data, inspect every checkboxable row, confirm each is one a human would confirm unhesitatingly |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s for the quick command
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated (Phase 30 plan 30-04, D-08 reconciliation)

## Validation Audit 2026-08-31

Reconciled by Phase 30 plan 30-04 (D-08). This file was seeded by plan-phase with 6 Wave 0
gaps (❌) before execution. Cross-referenced against `28-VERIFICATION.md` (2026-08-02
re-verification, a second pass after the phase's own initial verification found 2 BLOCKER
regressions): all Wave 0 test modules were created during execution, both blockers were
closed with mutation-checked regression tests, and the full 146-test targeted suite was
independently re-run by the verifier. No new test was written by this reconciliation.

| Metric | Count |
|--------|-------|
| Gaps found | 6 (all pre-existing Wave 0 markers, already closed by execution) |
| Resolved | 6 (confirmed closed via 28-VERIFICATION.md cross-reference, no new work needed) |
| Escalated | 0 |
