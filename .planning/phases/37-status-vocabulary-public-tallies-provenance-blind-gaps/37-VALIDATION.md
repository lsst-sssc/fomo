---
phase: "37"
slug: "status-vocabulary-public-tallies-provenance-blind-gaps"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: true
wave_0_complete: false
created: "2026-09-18"
---

# Phase 37 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django's built-in `TestCase` runner (unittest-based) — the only functioning suite per CLAUDE.md; `python -m pytest` does not collect these tests |
| **Config file** | none — invoked directly via `manage.py test` |
| **Quick run command** | `python manage.py test solsys_code.tests.<module>` |
| **Full suite command** | `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` — 37-07 Task 3 tags `solsys_code.tests.test_views.TestEphemeris` with `ephemeris_segfault` so the project's known-issue caveat (it segfaults in native ASSIST) becomes part of the invocation rather than prose beside it. Waves 1-4 run before the tag exists, where the flag matches nothing and excludes nothing — those waves omit the class by hand instead, exactly as the project note describes. |
| **Estimated runtime** | ~5 s for a single module (measured: `test_campaign_gap`, 27 tests, 4.9 s wall including Django startup and test-database creation) |

---

## Sampling Rate

- **After every task commit:** Run the task's own `<automated>` commands — always at least one
  `python manage.py test solsys_code.tests.<module>`
- **After every plan wave:** Run the full suite in the form given in the table above (waves 1-4
  by hand-omitting the ephemeris class; wave 5 onward with `--exclude-tag=ephemeris_segfault`)
- **Before `/gsd-verify-work`:** Full suite green, plus `pre-commit run ruff --all-files` and
  `pre-commit run ruff-format --all-files` clean
- **Max feedback latency:** ~5 s per task-level check (a single module); the full suite is the
  wave-level gate, not the per-task one

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 37-01-01 | 01 | 1 | STATUS-01 | T-37-02 | Legend stays a fixed vocabulary with no database read, so it cannot leak a row's existence | unit (tracer) | `python manage.py test solsys_code.tests.test_status_vocabulary` | ❌ W0 (new file, created by this task) | ⬜ pending |
| 37-01-02 | 01 | 1 | STATUS-01 | T-37-03 | Marker matching keeps the trailing-space rule, so a user-authored title cannot claim a ring | unit | `python manage.py test solsys_code.tests.test_campaign_approval solsys_code.tests.test_status_vocabulary` | ✅ / ❌ W0 | ⬜ pending |
| 37-01-03 | 01 | 1 | STATUS-02 | T-37-01 | The state-literal swap leaves the never-log-the-portal-body except clause untouched | unit | `python manage.py test solsys_code.tests.test_status_vocabulary solsys_code.tests.test_calendar_utils` | ✅ / ❌ W0 | ⬜ pending |
| 37-02-01 | 02 | 2 | TALLY-01 | — | N/A (decision checkpoint, `gate="blocking"`) | manual | — (checkpoint) | — | ⬜ pending |
| 37-02-02 | 02 | 2 | TALLY-01 | T-37-07 | Runner-owned figures are read-only in the admin | unit | `python manage.py test solsys_code.tests.test_proposal_allocation solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_admin` | ❌ W0 (new file) | ⬜ pending |
| 37-02-03 | 02 | 2 | TALLY-01 | T-37-04, T-37-05, T-37-06 | No credential or portal body in any log line, step summary or failure email | unit | `python manage.py test solsys_code.tests.test_proposal_allocation solsys_code.tests.test_unattended` | ✅ / ❌ W0 | ⬜ pending |
| 37-03-01 | 03 | 2 | GAPB-01 | T-37-09 | The new observation query has its own field restriction and no `select_related` back to the run | unit | `python manage.py test solsys_code.tests.test_campaign_gap` | ✅ | ⬜ pending |
| 37-03-02 | 03 | 2 | GAPB-01 | T-37-10, T-37-11 | The page adds no query parameter and shows dates and counts only | unit | `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_gap` | ✅ | ⬜ pending |
| 37-04-01 | 04 | 3 | TALLY-01 | T-37-13, T-37-15 | Every query is pk-keyed and column-restricted; the cached dict holds integers and flags only | unit | `python manage.py test solsys_code.tests.test_campaign_tally` | ❌ W0 (new file) | ⬜ pending |
| 37-04-02 | 04 | 3 | TALLY-01, UNUSED-01 | — | N/A | unit | `python manage.py test solsys_code.tests.test_campaign_tally` | ✅ / ❌ W0 | ⬜ pending |
| 37-04-03 | 04 | 3 | TALLY-02, TALLY-03 | T-37-14, T-37-16 | Pending-review runs excluded at the queryset level; run status provably never written | unit | `python manage.py test solsys_code.tests.test_campaign_tally` | ✅ / ❌ W0 | ⬜ pending |
| 37-05-01 | 05 | 4 | TALLY-01 | T-37-17, T-37-19 | `get_queryset()` and the non-staff field list are untouched; query count does not grow per row | unit | `python manage.py test solsys_code.tests.test_campaign_views` | ✅ | ⬜ pending |
| 37-05-02 | 05 | 4 | TALLY-02 | T-37-18 | A roll-up sums only publicly visible runs | unit | `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_tally` | ✅ | ⬜ pending |
| 37-06-01 | 06 | 4 | TALLY-01 | T-37-21, T-37-22 | The tally tag applies the same public-visibility gate as the campaign decoration and renders counts only | unit | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` | ✅ | ⬜ pending |
| 37-06-02 | 06 | 4 | UNUSED-01 | T-37-23, T-37-24 | The unused token is render-time only; the stored title is unchanged by a render | unit | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` | ✅ | ⬜ pending |
| 37-07-01 | 07 | 5 | STATUS-01, TALLY-01, GAPB-01 | T-37-26 | The runbook names the credential setting key, never a value | doc build | `pre-commit run --all-files` (includes the Sphinx build hook) | ✅ | ⬜ pending |
| 37-07-02 | 07 | 5 | TALLY-01, TALLY-02, UNUSED-01 | T-37-25 | No regenerated notebook output carries a contact detail, an API key or a real portal group id | integration (executed notebook) | the executed-cell-count probe in 37-07 Task 2's `<verify>` | ✅ | ⬜ pending |
| 37-07-03 | 07 | 5 | STATUS-01 | T-37-27 | The retirement list is deleted only after the database holds no legacy-spelled title | unit + data query | `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_status_vocabulary.py` — new file; created by 37-01 Task 1 as part of
      the tracer, covering STATUS-01/02's classifier, marker tables, legend and ring buckets
- [ ] `solsys_code/tests/test_proposal_allocation.py` — new file; created by 37-02 Task 2,
      covering the model's create-or-update behaviour and the estimate arithmetic
- [ ] `solsys_code/tests/test_campaign_tally.py` — new file; created by 37-04 Task 1, and home of
      the TALLY-03 guard class added in 37-04 Task 3
- [ ] A live `GET /api/proposals/<code>/` against one real watched proposal, to confirm the
      response field names and which allocation types the estimate should sum — folded into
      37-02 Task 1's `checkpoint:decision` rather than left as a separate wave-0 errand
      (RESEARCH.md assumptions A1/A2)
- [ ] Decision on RESEARCH.md Open Question 1 (the proposal-code carrier) — 37-02 Task 1

*No framework install needed — Django's `TestCase` is already fully wired for this app. Every
other test module this phase touches already exists.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The unused night's muted chip reads as visibly different from a realised night at a glance, and the legend's unused entry filters the month view in one click | UNUSED-01 (D-13, D-15) | Perceptual — an automated test can assert the class, the attribute and the token are present, but not that the treatment reads as distinct to a human eye | 37-07 Task 3's `<human-check>`: open a month containing an elapsed allocation night, a cancelled run night and an observed record; confirm the muted style and the unused token, the cancelled marker and its ring, and that clicking the unused legend entry isolates then clears the filter |
| The pop-up tally line reads sensibly beside the campaign name | TALLY-01 (D-09) | Layout/readability judgement | Same `<human-check>`: open one attributed entry's pop-up |

`workflow.human_verify_mode` is `end-of-phase`, so these are `<verify><human-check>` items inside
37-07 Task 3 rather than mid-flight `checkpoint:human-verify` tasks. The one blocking checkpoint
in this phase (37-02 Task 1) is a `checkpoint:decision`, which is unaffected by that setting.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (the sole exception is the
      `checkpoint:decision`, which gates the work rather than verifying it)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (three new test modules, each created by the first
      task that needs it)
- [x] No watch-mode flags
- [x] Feedback latency < 37s (measured ~5 s for a single module)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
