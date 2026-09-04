---
phase: "32"
slug: "adapter-consolidation"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: true
wave_0_complete: false
created: "2026-09-02"
---

# Phase 32 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Populated from 32-RESEARCH.md `## Validation Architecture`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django `TestCase` (`django.test.TestCase`), run via `python manage.py test` |
| **Config file** | none — settings module is `src.fomo.settings`, set by `manage.py`. Invoke as `python manage.py …`, never `./manage.py`. |
| **Quick run command** | `python manage.py test solsys_code.tests.test_write_and_reconcile solsys_code.tests.test_null_campaign_guards solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_sync_gemini_observation_calendar solsys_code.tests.test_campaign_reconciler` |
| **Full suite command** | `.planning/config.json` `workflow.test_command` — every `solsys_code/tests/test_*.py` and `solsys_code/solsys_code_observatory/tests/test_*.py` label except `test_views.py` (which segfaults in native ASSIST), plus the two named safe `test_views` classes (`TestSplitNumberUnitRegex`, `TestJPLSBDBQuery`). Matches CLAUDE.md's documented gotcha exactly. |
| **Estimated runtime** | quick run ~120 s; full suite ~300 s (Django `TestCase` on SQLite plus per-night astropy `sun_event()` scans in the reconciler tests) |

Note: any task that only touches the modules of one plan may narrow the quick run to that
plan's own `<automated>` labels — every task below already names them.

---

## Sampling Rate

- **After every task commit:** run that task's own `<automated>` commands (the narrowest
  labels in the Per-Task Verification Map below).
- **After every plan wave:** run the quick run command above. Waves here are one plan each
  (32-01 → 32-02 → 32-03 → 32-04), so this is effectively per plan.
- **Before `/gsd-verify-work`:** the full suite command must be green, plus
  `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` clean (D-07).
- **Max feedback latency:** 300 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 32-01-01 | 01 | 1 | ADAPT-02, ADAPT-03, ADAPT-05 | T-32-01 / T-32-02 / T-32-03 | Helper never rewrites `source`/`approval_status` on an existing `source='web'` row; partial unique constraint makes a `source_identifier` collision a DB error, not a silent overwrite | unit + migration | `python manage.py makemigrations --check --dry-run solsys_code` · `python manage.py migrate solsys_code` · `python manage.py test solsys_code.tests.test_write_and_reconcile solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_campaign_models` | ❌ W0 — `test_write_and_reconcile.py` created by this task (`tdd="true"`) | ⬜ pending |
| 32-01-02 | 01 | 1 | ADAPT-05 | T-32-02 | All five `run.campaign.name` dereference sites render a null-campaign run without raising | unit | `python manage.py test solsys_code.tests.test_null_campaign_guards solsys_code.tests.test_campaign_attribution solsys_code.tests.test_attribution_dismissals solsys_code.tests.test_campaign_views` | ❌ W0 — `test_null_campaign_guards.py` created by this task (`tdd="true"`) | ⬜ pending |
| 32-01-CP | 01 | 1 | ADAPT-02, ADAPT-03 | — | N/A — `checkpoint:decision`, gates the one-way reconciler-dispatch change (a run with exactly one linked `CampaignRunObservation` takes the container branch) that 32-01-03 implements. Positioned after 32-01-02, immediately before the task it gates — not before Task 1, since the schema/helper tracer and the null guards do not depend on this decision | checkpoint | none (human gate) | N/A | ⬜ pending |
| 32-01-03 | 01 | 1 | ADAPT-02, ADAPT-03, ADAPT-05 | T-32-04 / T-32-19 | Adoption bridge refuses to steal an event whose companion row points at a different run (staff-confirmed attribution outranks an adapter guess); `_linked_observation_window()` never trusts `.first()` on an ambiguous (0- or 2+-linked) run, falling back to the whole-window default instead of guessing | unit + docs | `python manage.py test solsys_code.tests.test_write_and_reconcile solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs` · `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` · same for `campaign_lifecycle_demo.ipynb` · `pre-commit run ruff --all-files && pre-commit run ruff-format --all-files` | ✅ (created by 32-01-01) | ⬜ pending |
| 32-02-01 | 02 | 2 | ADAPT-01, ADAPT-05 | T-32-08 / T-32-09 | A written-but-unprojected night is reported on stderr and counted in `not_projected`, never as a success | unit | `python manage.py test solsys_code.tests.test_load_telescope_runs` · `grep -v '^ *#' … \| grep -c 'insert_or_create_calendar_event'` == 0 | ✅ existing | ⬜ pending |
| 32-02-02 | 02 | 2 | ADAPT-01, ADAPT-04, ADAPT-05 | T-32-07 | Cutover mints no duplicate event and leaves no orphan (`CalendarEvent.objects.count()` unchanged) | unit + integration | `python manage.py test solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_calendar_utils solsys_code.tests.test_reconcile_campaign_runs` | ✅ existing | ⬜ pending |
| 32-02-03 | 02 | 2 | ADAPT-01 | T-32-06 | D-04's accepted proposal-collision risk is recorded in the runbook rather than silently inherited | docs | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` · `grep -c 'CampaignRun' docs/runbooks/telescope_runs_calendar.rst` · notebook-has-output JSON check · `pre-commit run ruff/ruff-format --all-files` | ✅ existing | ⬜ pending |
| 32-03-01 | 03 | 3 | ADAPT-02, ADAPT-03, ADAPT-05 | T-32-13 / T-32-14 / T-32-19 | DISPLAY-01 `is_verified` signal survives cutover; the new `not_projected` stderr line interpolates only the observation id and the reconciler's fixed skip vocabulary, never a caught exception; this task's `observation_record=record` argument is the sole producer of the link 32-01's T-32-19 guard protects | unit | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` · `grep … 'insert_or_create_calendar_event'` == 0 · `grep … 'SOAR_QUEUE'` ≥ 1 · `python manage.py check` | ✅ existing | ⬜ pending |
| 32-03-02 | 03 | 3 | ADAPT-02, ADAPT-03, ADAPT-04, ADAPT-05 | T-32-11 / T-32-12 | No SOAR record written under `lco_queue`; cutover adopts rather than duplicates; every written run carries `approval_status='approved'`; an LCO and a SOAR run each own exactly ONE narrowing `CalendarEvent` (`test_soar_and_lco_both_own_exactly_one_narrowing_event`), and a SOAR record's event narrows to `scheduled_start`/`scheduled_end` exactly as LCO's does (`test_soar_event_narrows_with_scheduled_times`) — no SOAR-per-night regression | unit + integration | `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar solsys_code.tests.test_campaign_run_observation` · `python manage.py test solsys_code.tests.test_campaign_attribution solsys_code.tests.test_campaign_attribution_views solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_campaign_reconciler` | ✅ existing | ⬜ pending |
| 32-03-03 | 03 | 3 | ADAPT-03 | T-32-10 | Runbook states the adapter-created link is structural exact identity with `confirmed_by` unset, not a staff confirmation, and that both LCO and SOAR own one narrowing entry per record (not one-per-night for SOAR); contains no occurrence of the incorrect Gemini obscode `568` | docs | `jupyter nbconvert … sync_lco_observation_calendar_demo.ipynb` · `grep -c 'soar_queue' …rst` · `grep -c 'Phase 35' …rst` · `grep -c '568' …rst` == 0 · notebook-has-output JSON check · `pre-commit run ruff/ruff-format --all-files` | ✅ existing | ⬜ pending |
| 32-04-01 | 04 | 4 | ADAPT-06, ADAPT-05 | T-32-15 / T-32-17 / T-32-18 | GEM-SECURE-01 holds: no new stderr line interpolates `record.parameters` or a caught exception message; no `CampaignRunObservation` link is created | unit | `python manage.py test solsys_code.tests.test_sync_gemini_observation_calendar` · `grep … 'insert_or_create_calendar_event'` == 0 · `grep -ci 'outcome propagation' …py` ≥ 2 · `python manage.py check` | ✅ existing | ⬜ pending |
| 32-04-02 | 04 | 4 | ADAPT-06, ADAPT-04, ADAPT-05 | T-32-16 | Cutover adopts rather than duplicates; every existing skip path still writes no `CampaignRun` and leaks no parameter content | unit + integration + full suite | `python manage.py test solsys_code.tests.test_sync_gemini_observation_calendar` · the `.planning/config.json` full-suite command | ✅ existing | ⬜ pending |
| 32-04-03 | 04 | 4 | ADAPT-06 | T-32-17 | The D-02 outcome-propagation caveat is present in the runbook as its own subsection, so Phase 33 cannot be surprised by it; Gemini North's obscode is named as `T15`, never the incorrect generic Maunakea code `568` | docs | `jupyter nbconvert … sync_gemini_observation_calendar_demo.ipynb` · `grep -ci 'outcome propagation' …rst` · `grep -c 'gemini_queue' …rst` · `grep -c 'T15' …rst` ≥ 1 · `grep -c '568' …rst` == 0 · notebook-has-output JSON check · `cd docs && make html` · `pre-commit run ruff/ruff-format --all-files` | ✅ existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

The three gaps RESEARCH.md's Validation Architecture named. All three are closed inside plan
32-01 (wave 1) or by the first task that needs them, so no separate Wave 0 plan exists — but
each must land before the adapter cutover that depends on it.

- [ ] **Cutover-simulation integration test per adapter** (covers ADAPT-05, resolves Open
  Question 1 / Pitfall 4). No "simulated cutover" test exists today for any of the three
  adapters. Landed as `test_cutover_does_not_duplicate_existing_event` +
  `test_cutover_leaves_no_orphan` (32-02 Task 2), `test_cutover_does_not_duplicate_existing_event`
  (32-03 Task 2), and `test_cutover_does_not_duplicate_existing_event` (32-04 Task 2), all
  standing on the bridge tests in 32-01 Task 3.
- [ ] **`CampaignRunObservation` exact-identity-link assertions** in
  `solsys_code/tests/test_sync_lco_observation_calendar.py` (covers ADAPT-02/03's linking half,
  not just the run-write half). Landed as `test_observation_link_created_with_no_confirming_user`
  and `test_rerun_no_churn_campaignrun_and_link` (32-03 Task 2), with the negative case
  `test_no_observation_link_created` (32-04 Task 2).
- [ ] **Null-campaign-row regression tests for the five guarded read sites**
  (`models.CampaignRun.__str__`, `campaign_reconciler.event_title`, both
  `campaign_tables.py` `render_run` methods, `campaign_attribution._campaign_evidence`). None of
  these sites has a null-campaign case today because no code path could produce one before this
  phase. Landed as the new module `solsys_code/tests/test_null_campaign_guards.py` (32-01 Task 2),
  each site with both a null-campaign case and a with-campaign regression case.
- [x] **Framework install:** none needed — `django.test.TestCase` is already fully configured.

Two new test modules are created by the `tdd="true"` tasks that first need them
(`test_write_and_reconcile.py` in 32-01 Task 1, `test_null_campaign_guards.py` in 32-01 Task 2),
which is why no task carries a `MISSING —` verify marker.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Runbook prose is accurate and the three command sections do not contradict each other | ADAPT-06 (docs half), ADAPT-01/02/03 | Prose coherence across three sections written by three different plans cannot be grepped; the automated checks only prove the required tokens are present | Read `docs/runbooks/telescope_runs_calendar.rst` end to end after plan 32-04 Task 3. Confirm each of the three command sections says what its command now writes, that the per-branch event-count wording is consistent (LCO and SOAR, exact-identity-linked to an `ObservationRecord` → one narrowing event per record; classical and Gemini, no observation link → one event per night), and that no section still describes a direct `CalendarEvent` write. |
| Calendar continuity through the whole cutover sequence — one event per night, no duplicates, no orphans, at every point (ROADMAP Success Criterion 5) | ADAPT-05 | The per-adapter tests each prove one cutover in isolation; the cross-adapter sequence on a real dev DB with pre-existing rows is a UAT concern | After all four plans ship, on the dev DB: record `CalendarEvent.objects.count()`, run all three commands in D-05's order, re-run each once more, and confirm the count only grew by genuinely new nights and that every event has a `CalendarEventMeta.run` set. |
| Notebook narrative prose no longer describes any command as writing calendar events itself | ADAPT-01/02/03/06 | The automated checks prove cells re-executed with output; they cannot judge whether the surrounding markdown still tells the old story | Read the four `docs/notebooks/pre_executed/*_demo.ipynb` markdown cells after each plan's docs task. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — every task except the
  `checkpoint:decision` gate carries at least one `<automated>` command with a `<fails_when>`.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify — every one of the
  12 non-checkpoint tasks verifies.
- [x] Wave 0 covers all MISSING references — no task emits a `MISSING —` marker; the two new
  test modules are created by the `tdd="true"` tasks that need them.
- [x] No watch-mode flags — every command is a single-shot `python manage.py test`,
  `nbconvert --execute`, `grep`, or `pre-commit run`.
- [x] Feedback latency < 300 s.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
