---
phase: "37"
slug: "status-vocabulary-public-tallies-provenance-blind-gaps"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-18"
validated: "2026-09-21"
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
| **Full suite command** | `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` — 37-07 Task 3 tags `solsys_code.tests.test_views.TestEphemeris` with `ephemeris_segfault` so the project's known-issue caveat (it segfaults in native ASSIST) becomes part of the invocation rather than prose beside it. Waves 1-4 ran before the tag existed, where the flag matched nothing and excluded nothing — those waves omitted the class by hand instead, exactly as the project note describes. |
| **Measured runtime** | ~5 s for a single module; 21 s for the four core phase-37 modules (184 tests); 54 s for the five view/display/template/admin modules (391 tests); **338 s for the full suite (1776 tests, `OK (skipped=1)`)** |

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
| 37-01-01 | 01 | 1 | STATUS-01 | T-37-02 | Legend stays a fixed vocabulary with no database read, so it cannot leak a row's existence | unit (tracer) | `python manage.py test solsys_code.tests.test_status_vocabulary` | ✅ | ✅ green |
| 37-01-02 | 01 | 1 | STATUS-01 | T-37-03 | Marker matching keeps the trailing-space rule, so a user-authored title cannot claim a ring | unit | `python manage.py test solsys_code.tests.test_campaign_approval solsys_code.tests.test_status_vocabulary` | ✅ | ✅ green |
| 37-01-03 | 01 | 1 | STATUS-02 | T-37-01 | The state-literal swap leaves the never-log-the-portal-body except clause untouched | unit | `python manage.py test solsys_code.tests.test_status_vocabulary solsys_code.tests.test_calendar_utils` | ✅ | ✅ green |
| 37-02-01 | 02 | 2 | TALLY-01 | — | N/A (decision checkpoint, `gate="blocking"`) | manual | — (checkpoint) | — | ✅ decided |
| 37-02-02 | 02 | 2 | TALLY-01 | T-37-07 | Runner-owned figures are read-only in the admin | unit | `python manage.py test solsys_code.tests.test_proposal_allocation solsys_code.tests.test_load_telescope_runs solsys_code.tests.test_admin` | ✅ | ✅ green |
| 37-02-03 | 02 | 2 | TALLY-01 | T-37-04, T-37-05, T-37-06 | No credential or portal body in any log line, step summary or failure email | unit | `python manage.py test solsys_code.tests.test_proposal_allocation solsys_code.tests.test_unattended` | ✅ | ✅ green |
| 37-03-01 | 03 | 2 | GAPB-01 | T-37-09 | The new observation query has its own field restriction and no `select_related` back to the run | unit | `python manage.py test solsys_code.tests.test_campaign_gap` | ✅ | ✅ green |
| 37-03-02 | 03 | 2 | GAPB-01 | T-37-10, T-37-11 | The page adds no query parameter and shows dates and counts only | unit | `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_gap` | ✅ | ✅ green |
| 37-04-01 | 04 | 3 | TALLY-01 | T-37-13, T-37-15 | Every query is pk-keyed and column-restricted; the cached dict holds integers and flags only | unit | `python manage.py test solsys_code.tests.test_campaign_tally` | ✅ | ✅ green |
| 37-04-02 | 04 | 3 | TALLY-01, UNUSED-01 | — | N/A | unit | `python manage.py test solsys_code.tests.test_campaign_tally` | ✅ | ✅ green |
| 37-04-03 | 04 | 3 | TALLY-02, TALLY-03 | T-37-14, T-37-16 | Pending-review runs excluded at the queryset level; run status provably never written | unit | `python manage.py test solsys_code.tests.test_campaign_tally` | ✅ | ✅ green |
| 37-05-01 | 05 | 4 | TALLY-01 | T-37-17, T-37-19 | `get_queryset()` and the non-staff field list are untouched; query count does not grow per row | unit | `python manage.py test solsys_code.tests.test_campaign_views` | ✅ | ✅ green |
| 37-05-02 | 05 | 4 | TALLY-02 | T-37-18 | A roll-up sums only publicly visible runs | unit | `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_tally` | ✅ | ✅ green |
| 37-06-01 | 06 | 4 | TALLY-01 | T-37-21, T-37-22 | The tally tag applies the same public-visibility gate as the campaign decoration and renders counts only | unit | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` | ✅ | ✅ green |
| 37-06-02 | 06 | 4 | UNUSED-01 | T-37-23, T-37-24 | The unused token is render-time only; the stored title is unchanged by a render | unit | `python manage.py test solsys_code.tests.test_calendar_display_extras solsys_code.tests.test_calendar_template` | ✅ | ✅ green |
| 37-07-01 | 07 | 5 | STATUS-01, TALLY-01, GAPB-01 | T-37-26 | The runbook names the credential setting key, never a value | doc build | `pre-commit run --all-files` (includes the Sphinx build hook) | ✅ | ✅ green |
| 37-07-02 | 07 | 5 | TALLY-01, TALLY-02, UNUSED-01 | T-37-25 | No regenerated notebook output carries a contact detail, an API key or a real portal group id | integration (executed notebook) | the executed-cell-count probe in 37-07 Task 2's `<verify>` | ✅ | ✅ green |
| 37-07-03 | 07 | 5 | STATUS-01 | T-37-27 | The retirement list is deleted only after the database holds no legacy-spelled title | unit + data query | `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` | ✅ | ✅ green |
| 37-08-01 | 08 | 6 | TALLY-02 | T-37-41, T-37-43, T-37-44 | The campaign-level applier writes only the three existing `unused_*` keys and never `run_status`; a staff edit is visible on the next load | integration (tracer, TDD) | `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignRollup` | ✅ | ✅ green |
| 37-08-02 | 08 | 6 | TALLY-02 | T-37-42 | `estimated_unused_nights` keeps exactly two call sites; nothing computed is ever written to the cache | unit | `python manage.py test solsys_code.tests.test_campaign_tally.TestGetOrComputeRollupFreshness` | ✅ | ✅ green |
| 37-08-03 | 08 | 6 | TALLY-01 | T-37-40 | The anonymous campaign-list fan-out is re-pinned as a named constant and proven *constant* per campaign, not merely bounded | unit + doc build | `python manage.py test solsys_code.tests.test_campaign_views` + `pre-commit run --all-files` | ✅ | ✅ green |
| 37-09-01 | 09 | 7 | TALLY-01 | T-37-09-01, T-37-09-03 | The tally queryset feeds a counts dict assigned to `table.tallies`, never merged into `table.data`; `get_queryset()` untouched | integration (tracer, TDD) | `python manage.py test solsys_code.tests.test_campaign_views.TestProgressColumnCoversEveryRenderedRow` | ✅ | ✅ green |
| 37-09-02 | 09 | 7 | TALLY-01 | T-37-09-02, T-37-09-04 | `per_page` is capped at `MAX_TABLE_PER_PAGE=100` before `RequestConfig.configure()` reads it; `sort`/`page` resolve against declared columns with clamped pages tested, not assumed | unit | `python manage.py test solsys_code.tests.test_campaign_views.TestProgressColumnCoversEveryRenderedRow solsys_code.tests.test_campaign_views.TestProgressColumnOnDegenerateCampaigns` | ✅ | ✅ green |
| 37-09-03 | 09 | 7 | TALLY-01 | T-37-09-SC | No new dependency; the regenerated notebook carries no credential or real portal id | e2e (executed notebook) | `jupyter nbconvert --to notebook --execute --inplace campaign_lifecycle_demo.ipynb` | ✅ | ✅ green |
| 37-10-01 | 10 | 8 | TALLY-02 | T-37-10-01, T-37-10-02 | `unused_unknown_runs` is derived inside the already pending-review-excluded run set; the widened applier reads and writes nothing | integration (tracer, TDD) | `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignRollup` | ✅ | ✅ green |
| 37-10-02 | 10 | 8 | TALLY-02 | T-37-10-03, T-37-10-04 | The displayed total is self-accounting (known contributors + `unused_unknown_runs` == `rollup['runs']`); no new queryset, pass or query | unit | `python manage.py test solsys_code.tests.test_campaign_tally.TestRollupPartiallyKnownUnusedTotal` | ✅ | ✅ green |
| 37-10-03 | 10 | 8 | UNUSED-01 | T-37-10-SC | Runbook and notebook describe the partially-known strip without exposing a proposal's private detail; no new dependency | doc build + e2e notebook | `pre-commit run --all-files` + `jupyter nbconvert --to notebook --execute --inplace` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `solsys_code/tests/test_status_vocabulary.py` — created by 37-01 Task 1 as part of
      the tracer, covering STATUS-01/02's classifier, marker tables, legend and ring buckets
- [x] `solsys_code/tests/test_proposal_allocation.py` — created by 37-02 Task 2,
      covering the model's create-or-update behaviour and the estimate arithmetic
- [x] `solsys_code/tests/test_campaign_tally.py` — created by 37-04 Task 1; home of
      the TALLY-03 guard class (`TestTallyNeverWritesRunStatus`) added in 37-04 Task 3, and later of
      37-08's `TestGetOrComputeRollupFreshness` and 37-10's `TestRollupPartiallyKnownUnusedTotal`
- [x] A live `GET /api/proposals/<code>/` against one real watched proposal, to confirm the
      response field names and which allocation types the estimate should sum — folded into
      37-02 Task 1's `checkpoint:decision` rather than left as a separate wave-0 errand
      (RESEARCH.md assumptions A1/A2)
- [x] Decision on RESEARCH.md Open Question 1 (the proposal-code carrier) — 37-02 Task 1

*No framework install was needed — Django's `TestCase` was already fully wired for this app. Every
other test module this phase touches already existed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions | UAT Result |
|----------|-------------|------------|-------------------|------------|
| The unused night's muted chip reads as visibly different from a realised night at a glance | UNUSED-01 (D-13, D-15) | Perceptual — an automated test can assert the class, the attribute and the token are present, but not that the treatment reads as distinct to a human eye | 37-07 Task 3's `<human-check>`: open a month containing an elapsed allocation night, a cancelled run night and an observed record; confirm the muted style and the unused token, the cancelled marker and its ring | ✅ pass (37-UAT.md test 1) |
| The legend's unused entry filters the month view in one click, and clicking again clears it | UNUSED-01 (D-15) | Browser JavaScript with no JS test runner in this Django codebase. The markup preconditions *are* automated (`data-filter="unused"` presence, the `cal-event-unused` CSS rule, the generalized handler source) — only the click-then-toggle interaction needs a human or a future Playwright-style check | 37-07 Task 3's `<human-check>`: click the `[U]` legend entry, confirm it isolates the unused nights, click again to clear, and confirm the existing single-active-filter and proposal-filter behaviour is unchanged | ✅ pass (37-UAT.md test 2) |
| The pop-up tally line reads sensibly beside the campaign name | TALLY-01 (D-09) | Layout/readability judgement | Same `<human-check>`: open one attributed entry's pop-up; a not-yet-known unused figure must read as a word, never a bare `0` | ✅ pass (37-UAT.md test 3) |

`workflow.human_verify_mode` is `end-of-phase`, so these were `<verify><human-check>` items inside
37-07 Task 3 rather than mid-flight `checkpoint:human-verify` tasks. The one blocking checkpoint
in this phase (37-02 Task 1) is a `checkpoint:decision`, which is unaffected by that setting. A
fourth UAT item (test 4, the roll-up staleness product decision) was a decision rather than a
check; the developer chose option (b), which became plan 37-08.

---

## Validation Audit 2026-09-21

| Metric | Count |
|--------|-------|
| Requirements in scope | 7 (STATUS-01, STATUS-02, TALLY-01, TALLY-02, TALLY-03, GAPB-01, UNUSED-01) |
| Tasks mapped | 27 (26 automated + 1 decision checkpoint) |
| Gaps found | 0 |
| Resolved | 0 (none needed) |
| Escalated | 0 |

**What this audit did.** The seeded map covered plans 37-01 .. 37-07 only and carried every row at
`⬜ pending`; the three gap-closure plans added after verification (37-08, 37-09, 37-10, waves 6-8)
were absent from it entirely. Every named test class across all ten plans' `coverage:` blocks was
resolved against the working tree, then executed.

**Evidence.**

- All 32 test classes named in the ten SUMMARY `coverage:` blocks exist in `solsys_code/tests/`.
- Targeted runs: 184 tests green across `test_status_vocabulary`, `test_proposal_allocation`,
  `test_campaign_tally`, `test_campaign_gap` (21 s); 391 tests green across `test_campaign_views`,
  `test_calendar_display_extras`, `test_calendar_template`, `test_unattended`, `test_admin` (54 s).
- Full suite: `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` →
  **`Ran 1776 tests in 338.421s` / `OK (skipped=1)`**, exit 0.
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` both Passed.
- `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb`: 24 code cells, 0 with a null
  `execution_count` — the pre-executed pairing is real, not a stub.
- TALLY-03 (the requirement most at risk of being asserted rather than tested) is pinned by
  `test_campaign_tally.py#TestTallyNeverWritesRunStatus`, a behavioural before/after snapshot
  across every tally entry point.

**One stale reference, not a coverage gap.** 37-09-SUMMARY.md's D2 cites
`TestProgressColumnCoversEveryRenderedRow::test_huge_per_page_is_capped_and_still_fully_covered`,
which no longer exists under that name. The 37-REVIEW CR-01 fix split it into
`test_huge_per_page_is_capped_at_the_maximum`, `test_per_page_exactly_at_the_cap_is_honoured` and
`test_degenerate_per_page_falls_back_to_the_default_not_the_maximum`. The behaviour is covered by
all three; only the summary's reference is stale. Recorded here rather than rewritten into the
sealed SUMMARY.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (the sole exception is the
      `checkpoint:decision`, which gates the work rather than verifying it)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references — all three new test modules were created by the first
      task that needed them, and all three now exist and run green
- [x] No watch-mode flags
- [x] Feedback latency < 37 s (measured ~5 s for a single module, 21 s for the four core modules)
- [x] `nyquist_compliant: true` set in frontmatter
- [x] Every requirement traced to at least one existing, executing, green test
- [x] Manual-only items are genuinely manual (perceptual or browser-JS) and all three passed UAT

**Approval:** validated 2026-09-21 — Phase 37 is Nyquist-compliant.
