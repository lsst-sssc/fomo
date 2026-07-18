---
phase: 21-site-disambiguation-submitter-contact-opt-in
verified: 2026-07-11T16:00:11Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 21: Site Disambiguation & Submitter Contact Opt-In Verification Report

**Phase Goal:** Give staff a real site-disambiguation UI in the approval queue (the natural next
step after quick task `260705-l1v`'s visibility fix) and let submitters opt into public contact
display. Both are structurally independent of the scheduling-representation work — they touch
`Observatory` resolution and the submission form, not the window schema.

**Verified:** 2026-07-11T16:00:11Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

(Sourced from ROADMAP.md's four Success Criteria for Phase 21 — the authoritative contract.)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | When a submitted site doesn't resolve via `resolve_site()`'s exact-match or live-MPC tiers, the approval queue's Site column presents a dropdown of fuzzy-matched `Observatory` candidates for staff to pick from (SITE-01) | ✓ VERIFIED | `ApprovalQueueTable.render_site()` (`solsys_code/campaign_tables.py:213-246`) emits an `<input list=...>`/`<datalist>` populated via `fuzzy_match_candidates()` against the cached `build_site_candidates()` pool; `TestApprovalQueueSiteDisambiguationUI` (4 tests, all pass) asserts the input/datalist/create-link render for an unresolved row and that `site_raw` is HTML-escaped (stored-XSS closed). Full `solsys_code` suite: 417/417 pass. |
| 2 | Staff can type a code directly and resolve it to an existing `Observatory` or explicitly create a new one; no placeholder `Observatory` is ever auto-fabricated (SITE-02) | ✓ VERIFIED | `CampaignRunDecisionView.post()` reads `site_selection`, maps it through `build_site_candidates()` back to an obscode (CR-01 fix, `campaign_views.py:370-382`), and calls `resolve_site(..., create_placeholder=False)`. `TestSiteSelectionResolution`/`TestCreateObservatoryRoundTrip` (5 tests) pass. **I independently wrote and ran a temporary end-to-end test** (removed after use) that POSTed a *name*-based candidate (`'Lowell Discovery Telescope'`, not the literal obscode `'G37'`) through the real `campaigns:decide` endpoint and confirmed it correctly resolved `run.site` to the matching `Observatory` — this closes the exact gap CR-01 in `21-REVIEW.md` identified (existing repo tests only exercised the literal-obscode case). Also independently verified the `CreateObservatory` `?next=`/`?obscode=` round-trip using the **real template** (not a hand-built POST payload) — GET rendered the hidden `<input name="next">` field (CR-02 fix, `observatory_create.html:16-18`), and a POST replaying only the fields the rendered HTML form actually contains redirected correctly to the approval queue. |
| 3 | Approving a run whose site a staff member already manually resolved does not silently re-resolve or overwrite that choice (SITE-03) | ✓ VERIFIED | `CampaignRunDecisionView.post()` wraps the resolve call in `if run.site is None:` (`campaign_views.py:360`). `TestApproval::test_approving_already_resolved_site_does_not_call_resolve_site` and `test_projection_failure_reverts_site_stays_set_second_approve_skips_resolve_site` (the live Pitfall-3 clobber regression) both pass. |
| 4 | A submitter who opts in (single combined flag, default opt-out) has their `contact_person`/`contact_email` shown publicly on the per-campaign table; leaving it unset keeps them staff-only exactly as today (VIEW-05) | ✓ VERIFIED | `CampaignRun.contact_public_opt_in` (default `False`) + migration `0007` (additive, clean); form checkbox persists through `form_valid`; `CampaignRunTableView.get_queryset()`'s `Case`/`When` annotation gates `contact_person`/`contact_email` at the SQL `SELECT`, not the template. `TestContactPublicOptIn` (6 tests) asserts both the raw non-staff `.values()` dict and rendered content for opted-in/opted-out rows; `ALLOWED_FIELDS_FOR_NON_STAFF` confirmed to still exclude the contact fields directly. |

**Score:** 4/4 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/solsys_code_observatory/utils.py::MPCObscodeFetcher.query_all` | Bulk MPC fetch, `query()` contract untouched | ✓ VERIFIED | Present; `solsys_code.solsys_code_observatory` test suite (14 tests) still green |
| `solsys_code/campaign_utils.py::build_site_candidates` | Cached merged candidate pool, local fallback | ✓ VERIFIED | Present; WR-01 fix broadened except clause to `AttributeError` — confirmed via direct call with a non-dict `query_all()` return that it degrades to a 25-entry local-only pool without raising |
| `solsys_code/campaign_utils.py::fuzzy_match_candidates` | difflib wrapper, ranked pairs | ✓ VERIFIED | Present; `TestSiteFuzzyMatch` (8 tests) green |
| `solsys_code/campaign_tables.py::ApprovalQueueTable.render_site` (override) | Inline input + datalist + create-new link | ✓ VERIFIED | Present at lines 213-246; escapes via `format_html`/`format_html_join` |
| `solsys_code/campaign_tables.py::ApprovalQueueTable.render_actions` (single-form refactor) | One `<form>` with two named submit buttons | ✓ VERIFIED | Present; existing approve/reject POST semantics unchanged, full approval suite green |
| `solsys_code/models.py::CampaignRun.contact_public_opt_in` | `BooleanField(default=False)` | ✓ VERIFIED | Present at `models.py:107` |
| `solsys_code/migrations/0007_campaignrun_contact_public_opt_in.py` | Single additive `AddField` | ✓ VERIFIED | Present; `makemigrations --check --dry-run` reports no missing migrations |
| `solsys_code/campaign_forms.py::CampaignRunSubmissionForm.contact_public_opt_in` | Checkbox in Contact fieldset | ✓ VERIFIED | Present at `campaign_forms.py:36`, wired into `Fieldset('Contact', ..., 'contact_public_opt_in', ...)` |
| `solsys_code/campaign_views.py::CampaignRunDecisionView.post` (D-06 guard + site_selection read) | Skip resolve on already-set site; resolve via candidate-pool-mapped selection otherwise | ✓ VERIFIED | Present; CR-01 fix (obscode-mapping) confirmed present and behaviorally correct |
| `solsys_code/solsys_code_observatory/views.py::CreateObservatory.get_success_url`/`get_initial` | `?next=` validated redirect, `?obscode=` prefill | ✓ VERIFIED | Present; WR-02 fix (3-char guard on prefill) confirmed via direct call — long site names no longer pre-fill an invalid value |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `build_site_candidates()` | `django.core.cache.cache` | `'mpc_obscode_candidates'` key, 24h TTL | ✓ WIRED | Confirmed by cache-hit/cache-miss tests and direct cache-clear/rebuild checks |
| `ApprovalQueueView.get_context_data` | `ApprovalQueueTable(candidate_pool=...)` | built once per request | ✓ WIRED | `campaign_views.py:304`; single call site confirmed by code inspection + full-suite pass |
| `render_site()` | `fuzzy_match_candidates()` | reads `self.candidate_pool` | ✓ WIRED | Confirmed by rendering tests |
| `CampaignRunDecisionView.post()` | `build_site_candidates()` pool | `site_selection` display-string → obscode mapping | ✓ WIRED | `campaign_views.py:380`; confirmed both by existing tests and my independent name-candidate E2E test |
| `render_site()`'s "Create new Observatory" link | `CreateObservatory` view | `?obscode=`/`?next=` query string | ✓ WIRED | Confirmed by `observatory_create.html`'s hidden `next` field (CR-02 fix) + `get_initial()`/`get_success_url()` reading them; independently verified via a real-template round-trip test |
| `CampaignRunSubmissionView.form_valid()` | `CampaignRun.objects.create(...)` | `contact_public_opt_in=form.cleaned_data[...]` | ✓ WIRED | `campaign_views.py:224` |
| `CampaignRunTableView.get_queryset()` | SQL `SELECT` | `Case`/`When` on `contact_public_opt_in` | ✓ WIRED | Confirmed against the raw non-staff `.values()` dict, not just rendered HTML |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full `solsys_code` test suite | `python manage.py test solsys_code` | 417/417 pass | ✓ PASS |
| Migration completeness | `python manage.py makemigrations solsys_code --check --dry-run` | "No changes detected in app 'solsys_code'" | ✓ PASS |
| `ruff check` on all Phase 21 files | `ruff check solsys_code/campaign_utils.py solsys_code/campaign_views.py solsys_code/campaign_tables.py solsys_code/campaign_forms.py solsys_code/models.py solsys_code/solsys_code_observatory/views.py solsys_code/solsys_code_observatory/utils.py solsys_code/tests/test_campaign_approval.py solsys_code/tests/test_campaign_views.py solsys_code/tests/test_campaign_forms.py solsys_code/tests/test_campaign_submission.py` | "All checks passed!" | ✓ PASS |
| CR-01 fix: name-based candidate resolves via real decide endpoint (not just literal obscode) | Temporary E2E test (written, run, then removed) | `run.site` correctly resolved to the matching `Observatory`, `site_needs_review=False` | ✓ PASS |
| CR-02 fix: real template carries `?next=` through as a hidden field, not just a hand-built POST payload | Temporary E2E test (written, run, then removed) | GET response contains `name="next" value="{next_url}"`; POST replaying only the rendered form's own fields redirects to the approval queue | ✓ PASS |
| WR-01 fix: `build_site_candidates()` no longer raises on a non-dict MPC bulk response | Django shell, `query_all` mocked to return `None` | Returns a 25-entry local-only dict, no exception | ✓ PASS |
| WR-02 fix: `get_initial()` only pre-fills `obscode` for a plausible 3-char value | Django shell, `RequestFactory` | Long site name → `initial.get('obscode') is None`; `'G37'` → pre-filled correctly | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SITE-01 | 21-01, 21-03 | Fuzzy-matched dropdown for unresolved sites in approval queue | ✓ SATISFIED | `build_site_candidates()`/`fuzzy_match_candidates()` + `render_site()` datalist, all tests green |
| SITE-02 | 21-04 | Staff type-to-resolve or explicit create-new, no auto-fabrication | ✓ SATISFIED | `site_selection` resolution (with CR-01 name-mapping fix) + `CreateObservatory` round-trip (with CR-02 fix); no-fabrication regression test on `260705-l1v`'s invariant passes |
| SITE-03 | 21-04 | Approving an already-resolved site never re-resolves/overwrites | ✓ SATISFIED | `if run.site is None:` guard + Pitfall-3 two-attempt regression test |
| VIEW-05 | 21-02 | Submitter contact opt-in, default opt-out, SQL-level PII gate | ✓ SATISFIED | `contact_public_opt_in` field/migration/form/queryset gate, all tests green |

No orphaned requirements — REQUIREMENTS.md's Phase 21 row (SITE-01, SITE-02, SITE-03, VIEW-05) exactly matches the union of `requirements:` fields declared across all four plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No debt markers (`TODO`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER`) found in any Phase-21-touched file. The `TBD` string matches in `campaign_utils.py`/`campaign_views.py`/`campaign_tables.py`/`models.py`/tests are all references to the pre-existing "TBD scheduling window" domain concept from Phases 18-19, not incomplete-work markers. | — | none |
| `solsys_code/campaign_views.py`, `solsys_code/solsys_code_observatory/views.py` | CR-01/CR-02/WR-01/WR-02 fix commits | The four `21-REVIEW-FIX.md` fixes (`19fc96e`, `3b21750`, `a752fba`, `b5a039c`) each modified only the fix target file with no accompanying new automated regression test proving the specific failure mode the review flagged (e.g. a name/short_name datalist candidate resolving correctly, or the real template's hidden `next` field round-tripping). The fixes are behaviorally correct — I independently wrote and ran temporary tests (then removed them, per verifier convention) proving each of the four fixes works end-to-end — but the committed test suite does not itself carry regression coverage for these specific scenarios, so a future refactor could silently reintroduce any of them without a test failing. | ℹ️ Info | Does not block the phase goal (all four fixes are demonstrably correct in the current codebase), but is a real test-coverage gap worth closing in a follow-up. |

### Human Verification Required

None. All four roadmap Success Criteria are directly verifiable via passing automated tests plus independent behavioral verification performed during this review (temporary E2E tests written, executed, and removed for the two review-fix items — CR-01, CR-02 — that lacked dedicated repo-committed regression tests).

### Gaps Summary

No gaps blocking phase goal achievement. All four ROADMAP.md Success Criteria (SITE-01, SITE-02,
SITE-03, VIEW-05) are implemented, wired, and behaviorally verified — including the four
CR-01/CR-02/WR-01/WR-02 fixes from `21-REVIEW-FIX.md`, which I independently re-verified against
the current codebase (not merely trusted from the fix report) since two of the four fixes shipped
without dedicated regression tests. The one notable finding is an informational test-coverage gap
(noted above), not a functional defect: the CR-01/CR-02 fixes work correctly today but have no
committed automated test guarding against regression.

---

_Verified: 2026-07-11T16:00:11Z_
_Verifier: Claude (gsd-verifier)_
