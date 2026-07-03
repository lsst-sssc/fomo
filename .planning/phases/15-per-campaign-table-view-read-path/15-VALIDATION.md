---
phase: 15
slug: per-campaign-table-view-read-path
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-03
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
| TBD (assigned by planner) | TBD | TBD | VIEW-01 | — | Table lists all runs for a campaign, sortable/paginated (25/page, D-11) | integration (Django `Client`) | `./manage.py test solsys_code.tests.test_campaign_views.TestCampaignRunTableView.test_lists_all_runs_paginated` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | VIEW-02 | — | Target-detail page shows one link per matching campaign; navbar shows "Campaigns" entry | integration (Django `Client`) | `./manage.py test solsys_code.tests.test_campaign_views.TestCampaignDetailIntegration` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | VIEW-03 | PII disclosure — see Security Domain below | Anonymous client never sees `contact_person`/`contact_email` (context AND content); staff client does | integration (anonymous `Client()` vs. `is_staff=True` `Client()`) | `./manage.py test solsys_code.tests.test_campaign_views.TestContactFieldGating` | ❌ W0 | ⬜ pending |
| TBD (assigned by planner) | TBD | TBD | VIEW-04 | Tampering (unvalidated GET params) — see Security Domain below | `run_status` multi-select filter narrows rows (OR semantics); `open_to_collaboration` filter narrows rows; default (no filter) shows everything (D-07) | integration (Django `Client`, GET with query params) | `./manage.py test solsys_code.tests.test_campaign_views.TestCampaignRunFilterSet` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Task ID/Plan/Wave columns are placeholders — the planner assigns concrete task IDs when creating PLAN.md files and must update this table (or the plan's own `must_haves`) accordingly.

---

## Wave 0 Requirements

- [ ] `solsys_code/tests/test_campaign_views.py` — new test module covering VIEW-01..04 (does not exist yet; no prior view tests for `CampaignRun` exist — `test_campaign_models.py` from Phase 14 only covers the model, not views)
- [ ] Fixture data: reuse `NonSiderealTargetFactory` (CLAUDE.md convention — never `SiderealTargetFactory`) + `TargetList` + `CampaignRun.objects.create(...)` rows spanning multiple `run_status`/`approval_status` values, and at least one row with `contact_person`/`contact_email` populated, following the pattern already established in `solsys_code/tests/test_campaign_models.py`
- [ ] A `User(is_staff=True)` test fixture/helper for the staff-vs-anonymous `Client` split in `TestContactFieldGating` — no existing precedent for staff-user test fixtures in this codebase (grep for `is_staff` across `solsys_code/`/`src/` returned no hits), so this is a genuinely new fixture pattern for the test suite

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|--------------------|
| Table visually resembles the reference 3I/ATLAS spreadsheet layout and the `approval_status` badge (D-08) is visually distinct/legible | VIEW-01 | Spreadsheet-parity "feel" and badge contrast are subjective visual judgments; functional column presence/values are covered by automated tests above, but visual layout review belongs to UI-SPEC.md / `/gsd-ui-review`, not this functional test suite | Render the campaign table in a browser, compare column set/order against the reference 3I/ATLAS sheet (https://docs.google.com/spreadsheets/d/1INhxLWlHoa-JkW-uKRzmSyms06wI80wEXTqBJSR3YAI/), confirm the `approval_status` badge is visually distinct per D-08 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
