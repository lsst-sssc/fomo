---
phase: 21
slug: site-disambiguation-submitter-contact-opt-in
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-11
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Django's own test runner (`django.test.TestCase`) — not pytest; matches CLAUDE.md's testing split (DB-dependent tests live under `solsys_code/tests/`) |
| **Config file** | none — settings module `src.fomo.settings` via `manage.py` |
| **Quick run command** | `python manage.py test solsys_code.tests.test_campaign_approval` (SITE-01/02/03 — site-resolution/clobber-guard changes) and `python manage.py test solsys_code.tests.test_campaign_views solsys_code.tests.test_campaign_forms solsys_code.tests.test_campaign_submission` (VIEW-05 — opt-in changes) |
| **Full suite command** | `python manage.py test solsys_code` plus `ruff check .` / `ruff format --check .` |
| **Estimated runtime** | ~25s test execution / ~38s wall (quick command combo, 75 tests — baselined this session, all green pre-phase) |

Note: `./manage.py` is not executable in this environment (`Permission denied`) — use `python manage.py ...` for all commands below.

---

## Sampling Rate

- **After every task commit:** Run the quick command for the module touched — `test_campaign_approval` for SITE-01/02/03 work, or the `test_campaign_views`/`test_campaign_forms`/`test_campaign_submission` trio for VIEW-05 work
- **After every plan wave:** Run `python manage.py test solsys_code`
- **Before `/gsd-verify-work`:** Full suite must be green, plus `ruff check .` and `ruff format --check .` clean (CLAUDE.md quality gate)
- **Max feedback latency:** ~38s (quick command combo) — well under any reasonable threshold; none of the touched modules import the heavy `ephem_utils`/`views` SPICE-kernel chain

---

## Per-Task Verification Map

Task ID / Plan / Wave are assigned once `/gsd-planner` runs (this doc is created before planning, per Nyquist gate ordering). Rows below map each phase requirement (and the Security Domain's threat patterns) to its concrete test; the planner should carry these into per-task `<acceptance_criteria>` and this table's Task ID/Plan/Wave columns should be back-filled from the resulting PLAN.md files.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|--------------------|-------------|--------|
| 21-01-T3 | 21-01 | 1 | SITE-01 | — | N/A | unit | `python manage.py test solsys_code.tests.test_campaign_approval.TestSiteFuzzyMatch` — candidate pool building against widened MPC list, `difflib.get_close_matches` invocation | ❌ W0 — new test class | ⬜ pending |
| 21-03-T3 | 21-03 | 2 | SITE-01 | — | N/A | unit | same — Site column renders a `<datalist>`/`<input list=...>` of fuzzy-matched candidates for an unresolved pending row | ❌ W0 — new test | ⬜ pending |
| 21-01-T2 | 21-01 | 1 | SITE-01 | — | N/A | unit | `MPCObscodeFetcher` bulk-fetch method, result cached via `django.core.cache.cache`; mock `requests.get` (never hit the live API in tests, mirror existing tier-2 mocking pattern) | ❌ W0 — new test | ⬜ pending |
| 21-04-T1 | 21-04 | 3 | SITE-02 | — | N/A | unit | `python manage.py test solsys_code.tests.test_campaign_approval.TestApprovalSiteResolution` (extend existing class) — staff-typed/selected code resolves to an existing `Observatory` | ✅ existing class, add cases | ⬜ pending |
| 21-04-T2 | 21-04 | 3 | SITE-02 | — | N/A | integration | same — explicit "create new" path round-trips through `CreateObservatory` with `?next=`/`?obscode=` support (new extension to that view — flagged gap) and returns to the approval queue | ❌ W0 — new test; view extension needed | ⬜ pending |
| 21-04-T2 | 21-04 | 3 | SITE-02 | — | N/A | unit | same — no placeholder `Observatory` is ever auto-fabricated (regression on `260705-l1v`'s existing invariant) | ✅ existing class, add cases | ⬜ pending |
| 21-04-T1 | 21-04 | 3 | SITE-03 | — | N/A | unit | `python manage.py test solsys_code.tests.test_campaign_approval.TestApproval` (extend) — approving a run with `run.site` already set does not call `resolve_site()` again | ✅ existing class, add cases | ⬜ pending |
| 21-04-T1 | 21-04 | 3 | SITE-03 | — | N/A | regression | same — Pitfall 3 exact reproduction: approve succeeds at site-resolution, fails at calendar projection, reverts to `PENDING_REVIEW`; approve again and assert `resolve_site`/`requests.get` is NOT called a second time | ❌ W0 — new test, no existing coverage | ⬜ pending |
| 21-02-T3 | 21-02 | 1 | VIEW-05 | T-21-02 | Per-row PII gated at the SQL `SELECT` level via `Case`/`When`, not template-only | unit | `python manage.py test solsys_code.tests.test_campaign_views` — opted-in run's `contact_person`/`contact_email` visible to anonymous visitors on the per-campaign table | ✅ existing file, add cases | ⬜ pending |
| 21-02-T3 | 21-02 | 1 | VIEW-05 | T-21-02 | Raw `.values()` dict (not just rendered HTML) never contains a non-empty contact field for an opted-out row | unit | same — opted-out run stays staff-only exactly as today (regression on 15-RESEARCH.md Pitfall 1's "restrict the queryset, not just the template" invariant) | ✅ existing file, add cases | ⬜ pending |
| 21-02-T2 | 21-02 | 1 | VIEW-05 | — | N/A | unit | `python manage.py test solsys_code.tests.test_campaign_submission solsys_code.tests.test_campaign_forms` — submission form's opt-in checkbox (default unchecked) persists onto the created `CampaignRun` | ✅ existing files, add cases | ⬜ pending |
| 21-03-T3 | 21-03 | 2 | — | T-21-01 | Untrusted text (`site_raw`, MPC `name_utf8`) rendered via `format_html`/`format_html_join`, never `mark_safe` or string interpolation | unit | `python manage.py test solsys_code.tests.test_campaign_approval` — stored-XSS guard on the new site-input/datalist rendering (mirrors existing `render_window_start()` tooltip pattern) | ❌ W0 — new test | ⬜ pending |
| 21-04-T1 | 21-04 | 3 | — | T-21-04 | Oversized/malformed `site_selection` free text is flagged, not processed | unit | same — existing `_MAX_OBSCODE_LEN` guard applies unchanged to `site_selection` routed through `resolve_site()` | ✅ existing guard, add case | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `TestSiteFuzzyMatch` class in `solsys_code/tests/test_campaign_approval.py` — covers SITE-01 (candidate pool building, `difflib` invocation, datalist rendering), mocking `MPCObscodeFetcher`'s bulk-fetch method (never hit the real live API in tests — mirror the existing `@patch('requests.get', ...)` pattern already used elsewhere in this test module for tier-2 mocking)
- [ ] A regression test for Pitfall 3's exact reproduction sequence (approve succeeds at site-resolution but fails at calendar projection, reverts to `PENDING_REVIEW`, approve again, assert `resolve_site`/underlying `requests.get` is NOT called a second time)
- [ ] Test fixtures for the bulk MPC response shape — a small hand-built dict mirroring the real `{obscode: {name_utf8, short_name, old_names, observations_type, longitude, ...}}` shape (5-10 representative entries suffice, including one `observations_type='satellite'` entry with `longitude: None`)
- [ ] `CreateObservatory`'s `get_success_url()` needs `?next=`/`?obscode=` support before a test can assert the return-to-approval-queue round-trip (flagged gap, not existing behavior)
- [ ] Framework install: none — Django `TestCase` already the established framework for this module

*Factory conventions (`NonSiderealTargetFactory`, `django.test.TestCase`) are already established — Wave 0 is test-content work plus the one small `CreateObservatory` view extension, not infrastructure setup.*

---

## Manual-Only Verifications

All phase behaviors have automated verification. The new `<datalist>`/`form=` attribute UI is server-rendered HTML (no JS interactivity), so both markup shape and behavior are assertable via the Django test client's response content — no manual/visual check required.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 40s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
