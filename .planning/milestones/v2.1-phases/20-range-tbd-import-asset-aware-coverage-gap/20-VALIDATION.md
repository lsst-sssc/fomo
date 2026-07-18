---
phase: 20
slug: range-tbd-import-asset-aware-coverage-gap
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-10
---

# Phase 20 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django's own test runner (`django.test.TestCase`) — not pytest; matches CLAUDE.md's testing split (DB-dependent tests live under `solsys_code/tests/`) |
| **Config file** | none — settings module `src.fomo.settings` via `manage.py` |
| **Quick run command** | `python manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_campaign_gap` |
| **Full suite command** | `python manage.py test solsys_code` |
| **Estimated runtime** | ~4s test execution / ~16s wall (quick command, 56 tests — baselined this session, all green pre-phase). Full suite pays the heavy SPICE-kernel import cost once (per CLAUDE.md's documented `ephem_utils` module-load side effect), not measured this session. |

Note: `./manage.py` is not executable in this environment (`Permission denied`) — use `python manage.py ...` for all commands below.

---

## Sampling Rate

- **After every task commit:** Run `python manage.py test solsys_code.tests.test_import_campaign_csv solsys_code.tests.test_campaign_gap`
- **After every plan wave:** Run `python manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green, plus `ruff check .` and `ruff format --check .` clean (CLAUDE.md quality gate)
- **Max feedback latency:** ~16s (quick command) — well under any reasonable threshold; neither target file imports the heavy `ephem_utils`/`views` chain (guarded by the existing `TestNoHeavyEphemerisImport` test)

---

## Per-Task Verification Map

Task ID / Plan / Wave are assigned once `/gsd-planner` runs (this doc is created before planning, per Nyquist gate ordering). Rows below map each phase requirement to its concrete test; the planner should carry these into per-task `<acceptance_criteria>` and this table's Task ID/Plan/Wave columns should be back-filled from the resulting PLAN.md files.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|--------------------|-------------|--------|
| TBD | TBD | TBD | IMPORT-01 | — | N/A | unit | `python manage.py test solsys_code.tests.test_import_campaign_csv.TestCampaignUtils` | ✅ extend existing class | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01 | — | N/A | unit | same — compact same-month range (`"2025-11-02 -25"`) parses correctly | 🔁 existing test currently asserts this string is malformed — must be edited, not just added to | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01 | — | N/A | unit | same — D-11 rollover, same year (e.g. Nov→Dec) | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01 | — | N/A | unit | same — D-11 rollover crossing Dec→Jan (year increments) | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01 | — | N/A | unit | same — compact range with invalid resulting day (e.g. day2=35) falls through to TBD, never raises | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-02 | — | N/A | unit | same — blank `Obs. Date` → TBD tuple (`window_start is None`, `window_needs_review=True`, `original_obs_date_raw=''`) | 🔁 existing test asserts old `ValueError` contract — must be rewritten | ⬜ pending |
| TBD | TBD | TBD | IMPORT-02 | — | N/A | unit | same — `"YYYY-MM-?"` marker → TBD tuple with `original_obs_date_raw` preserved | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-02 | — | N/A | unit | same — genuine garbage text (e.g. `"TBD pending Cycle 2"`) → TBD tuple, `window_needs_review=True` | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01/02 | — | N/A | integration | `python manage.py test solsys_code.tests.test_import_campaign_csv.TestImportCampaignCsv` — range-shaped row creates a `CampaignRun` with matching window (not skipped) | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01/02 | — | N/A | integration | same — TBD-shaped row creates a flagged `window_needs_review=True` row, counted in new summary counter, never `skipped_count` | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01/02 | — | N/A | integration | same — two TBD rows, same campaign+telescope, different `contact_person`, both import | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | IMPORT-01/02 | — | N/A | integration | same — two TBD rows, same campaign+telescope+`contact_person` in one batch, second collides and is skipped/logged | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | ASSET-01/02 | — | N/A | unit | `python manage.py test solsys_code.tests.test_campaign_gap.TestClaimedDates` — ground run, full window, every date claimed | ✅ existing `test_range_run_claims_every_date_in_window` — confirm unchanged | ⬜ pending |
| TBD | TBD | TBD | ASSET-01/02 | — | N/A | unit | same — space run, `window_start == window_end` (narrowed), that one date claimed | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | ASSET-01/02 | — | N/A | unit | same — space run, un-narrowed range, zero dates claimed, lands in `pending_narrowing_runs` (not `undated_runs`) | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | ASSET-01/02 | — | N/A | unit | same — space run, TBD (both null), zero dates claimed, lands in `undated_runs` (D-09 explicit distinction) | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | ASSET-02 | — | N/A | integration (view-level) | `python manage.py test solsys_code.tests.test_campaign_gap.TestGapAnalysisView` — new `pending_narrowing_runs` alert block renders with correct count | ❌ W0 — new test | ⬜ pending |
| TBD | TBD | TBD | D-08 | — | N/A | unit (table-render) | `python manage.py test solsys_code` — TBD badge `title` attribute shows `original_obs_date_raw` when set, absent when blank; exact target file (`test_campaign_tables.py` vs `test_campaign_views.py`) unconfirmed — no dedicated file found this session | ❌ W0 — new test; confirm target file during planning | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_import_campaign_csv.py::test_parse_obs_window_unparseable_date_raises` (~line 270-272) — rewrite to assert the new TBD-tuple contract (IMPORT-02) instead of `assertRaises(ValueError)`
- [ ] `solsys_code/tests/test_import_campaign_csv.py::test_natural_key_failure_skipped_and_logged` (~line 470-502) — replace its `'2025-11-02 -25'` fixture row (now a valid parseable range) with a genuine non-date natural-key failure (e.g. blank `Telescope / Instrument`)
- [ ] New unit tests in `test_import_campaign_csv.py` for `parse_obs_window()`'s D-11/D-12 branches (same-year rollover, year-crossing rollover, invalid-day fallback, en-dash/hyphen full-range variants)
- [ ] New unit tests in `test_campaign_gap.py` for `claimed_dates()`'s ground-vs-space branch and `pending_narrowing_runs` bucket — reuse `TestClaimedDates._make_run` helper and `test_campaign_approval.py`'s space-`Observatory` fixture (~line 200-206); no new fixture infrastructure needed
- [ ] New integration tests in `test_import_campaign_csv.py` for `import_campaign_csv`'s range/TBD import paths, including the TBD natural-key collision case — reuse existing `_WriteCsvMixin`/`_row()` helpers
- [ ] Locate (or create) the test file covering `CampaignRunTable.render_window_start()`'s new tooltip (D-08) — no dedicated `test_campaign_tables.py` found this session; confirm during planning whether it belongs in a new file or `test_campaign_views.py`

*Framework and factory conventions (`NonSiderealTargetFactory`, `django.test.TestCase`) are already established — Wave 0 is test-content work, not infrastructure setup.*

---

## Manual-Only Verifications

All phase behaviors have automated verification. D-08's tooltip is rendered via `format_html()` (same pattern as the existing `render_site()` `site_needs_review` tooltip) and is testable via string assertion — no manual/visual check required.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 16s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
