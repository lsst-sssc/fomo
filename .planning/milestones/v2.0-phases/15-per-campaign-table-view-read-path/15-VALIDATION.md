---
phase: 15
slug: per-campaign-table-view-read-path
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-03
updated: 2026-07-03
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django test runner (`django.test.TestCase`), per CLAUDE.md's "DB-dependent tests go in `solsys_code/tests/`" convention |
| **Config file** | none — Django test discovery via `./manage.py test solsys_code` |
| **Quick run command** | `./manage.py test solsys_code.tests.test_campaign_views` |
| **Full suite command** | `./manage.py test solsys_code` |
| **Estimated runtime** | ~30s quick / ~120s+ full (full suite collection imports `solsys_code.ephem_utils`, which pays a one-time ~1.6GB SPICE kernel download on a cold cache per CLAUDE.md; cached on subsequent runs) |

---

## Sampling Rate

- **After every task commit:** Run `./manage.py test solsys_code.tests.test_campaign_views`
- **After every plan wave:** Run `./manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green, plus `ruff check .` / `ruff format --check .` clean per CLAUDE.md
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 15-01/T1-T3 | 15-01 | 1 | VIEW-01 | — | Table lists all runs for a campaign, sortable/paginated (25/page, default `-obs_date`) | integration (Django `Client`) | `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignRunTableView` | ✅ | ✅ green (4/4) |
| 15-02/T1-T3 | 15-02 | 2 | VIEW-02 | — | Target-detail page shows one link per matching campaign; navbar shows "Campaigns" entry | integration (Django `Client`) | `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignDetailIntegration` | ✅ | ✅ green (3/3) |
| 15-01/T1-T3 | 15-01 | 1 | VIEW-03 | T-15-01 (see `15-SECURITY.md`) | Anonymous client never sees `contact_person`/`contact_email` (context AND content); staff client does | integration (anonymous `Client()` vs. `is_staff=True` `Client()`) | `python manage.py test solsys_code.tests.test_campaign_views.TestContactFieldGating` | ✅ | ✅ green (4/4) |
| 15-01/T1-T3 | 15-01 | 1 | VIEW-04 | T-15-02 (see `15-SECURITY.md`) | `run_status` multi-select filter narrows rows (OR semantics); `open_to_collaboration` filter narrows rows; default (no filter) shows everything | integration (Django `Client`, GET with query params) | `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignRunFilterSet` | ✅ | ✅ green (3/3) |
| 15-01/T3 | 15-01 | 1 | D-03 (campaigns list, not a numbered requirement) | — | `GET /campaigns/` lists only `TargetList`s with ≥1 `CampaignRun` | integration (Django `Client`) | `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignListView` | ✅ | ✅ green (2/2) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Audited post-execution (2026-07-03) via `/gsd-validate-phase`: re-ran `python manage.py test solsys_code.tests.test_campaign_views` fresh (not read from a log) — 16/16 tests pass, matching `15-VERIFICATION.md`'s independent re-run. Every VIEW-01..04 requirement is COVERED by a real, currently-green automated test; no MISSING or PARTIAL gaps found.

---

## Wave 0 Requirements

- [x] `solsys_code/tests/test_campaign_views.py` — new test module covering VIEW-01..04, created in 15-01/Task 1 (`d4c9f84`), extended in 15-02/Task 3 (`b205c09`) — 16 tests total, all green
- [x] Fixture data: `NonSiderealTargetFactory` (CLAUDE.md convention honored — no `SiderealTargetFactory` usage anywhere in the module) + `TargetList` + `CampaignRun.objects.create(...)` rows spanning multiple `run_status`/`approval_status` values, plus seeded `contact_person`/`contact_email` for the PII-gating tests
- [x] `User(is_staff=True)` test fixture (`staff_user = User.objects.create_user(..., is_staff=True)`, `test_campaign_views.py:54`) for the staff-vs-anonymous `Client` split in `TestContactFieldGating` — new fixture pattern, confirmed in use and passing

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| Table visually resembles the reference 3I/ATLAS spreadsheet layout and the `approval_status` badge (D-08) is visually distinct/legible | VIEW-01 | Spreadsheet-parity "feel" and badge contrast are subjective visual judgments; functional column presence/values are covered by automated tests above, but visual layout review belongs to UI-SPEC.md / `/gsd-ui-review`, not this functional test suite | Render the campaign table in a browser, compare column set/order against the reference 3I/ATLAS sheet (https://docs.google.com/spreadsheets/d/1INhxLWlHoa-JkW-uKRzmSyms06wI80wEXTqBJSR3YAI/), confirm the `approval_status` badge is visually distinct per D-08 |

---

## Validation Audit 2026-07-03

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 (none needed — all requirements were already COVERED by real, passing tests) |
| Escalated | 0 |

Both plans executed with TDD (15-01 Task 1 is `tdd="true"` in PLAN.md; RED commit `d4c9f84` precedes GREEN commits `73afcfe`/`5229a7c`). Cross-checked against `15-VERIFICATION.md` (independently re-ran the same suite, same 16/16 result) and `15-SECURITY.md` (T-15-01/T-15-02 threat mitigations map 1:1 onto `TestContactFieldGating`/`TestCampaignRunFilterSet`).

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (none remained — all three Wave 0 items were satisfied during execution)
- [x] No watch-mode flags
- [x] Feedback latency < 30s (16-test module runs in ~5.6s)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** verified 2026-07-03
