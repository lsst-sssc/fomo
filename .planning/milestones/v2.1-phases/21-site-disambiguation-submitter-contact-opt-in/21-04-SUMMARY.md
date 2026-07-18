---
phase: 21-site-disambiguation-submitter-contact-opt-in
plan: 04
subsystem: api
tags: [django, open-redirect, mpc-obscodes, django-crossview-redirect]

# Dependency graph
requires:
  - phase: 21-site-disambiguation-submitter-contact-opt-in
    provides: "Plan 21-01's build_site_candidates()/fuzzy_match_candidates() helpers and Plan 21-03's inline site_selection input + 'Create new Observatory' ?obscode=/?next= link"
provides:
  - "CampaignRunDecisionView.post() D-06 clobber guard -- resolve_site() only runs when run.site is None"
  - "SITE-02 site_selection resolution wired into the approve path"
  - "CreateObservatory ?obscode= prefill + validated ?next= redirect round-trip back to the approval queue"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "django.utils.http.url_has_allowed_host_and_scheme(allowed_hosts={request.get_host()}, require_https=request.is_secure()) as the open-redirect guard for a query-string ?next= target"
    - "get_initial() override mirroring the existing get_context_data 'call super, return dict' style for GET-param form prefill"

key-files:
  created: []
  modified:
    - solsys_code/campaign_views.py
    - solsys_code/solsys_code_observatory/views.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "Kept the except Exception revert block byte-for-byte unchanged per PATTERNS.md -- the D-06 fix is purely the new `if run.site is None:` guard placed before the resolve_site() call, not a change to the failure-recovery contract"
  - "Split the SITE-02 create-new round-trip into three focused test methods (prefill, safe-next redirect, unsafe-next fallback) rather than one combined test, matching this test file's existing one-assertion-focus-per-method style"
  - "Mocked MPCObscodeFetcher.to_observatory() directly (side_effect creating+returning a real Observatory row) for the CreateObservatory round-trip tests, rather than mocking query()+letting to_observatory() run for real -- to_observatory() reads several MPC-response dict keys with no defaults, so a bare query() mock (which never has to_observatory()'s expected obs_data shape) would raise MissingDataException"

patterns-established:
  - "A satellite-type site_selection (250/274/289) still falls through to (None, True) via resolve_site()'s pre-existing to_observatory() TypeError path -- documented in code as expected, pre-existing behavior, not a Phase 21 regression (RESEARCH.md Pitfall 4)"

requirements-completed: [SITE-02, SITE-03]

coverage:
  - id: D1
    description: "Approving a run whose site is already set skips resolve_site() entirely and never overwrites the existing resolution"
    requirement: "SITE-03"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval.test_approving_already_resolved_site_does_not_call_resolve_site"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval.test_projection_failure_reverts_site_stays_set_second_approve_skips_resolve_site"
        status: pass
    human_judgment: false
  - id: D2
    description: "Approving a run with site=None reads the staff-submitted site_selection (falling back to site_raw) and resolves it via the existing 3-tier resolver, never fabricating a placeholder Observatory"
    requirement: "SITE-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteSelectionResolution.test_staff_typed_existing_obscode_resolves_via_site_selection_tier_1_hit"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteSelectionResolution.test_unresolvable_site_selection_leaves_observatory_count_unchanged"
        status: pass
    human_judgment: false
  - id: D3
    description: "An oversized/malformed site_selection is flagged for review via resolve_site's existing _MAX_OBSCODE_LEN guard, not processed or crashed, with no network call"
    requirement: "SITE-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestApproval.test_oversized_site_selection_is_flagged_with_no_network_call_or_fabrication"
        status: pass
    human_judgment: false
  - id: D4
    description: "Staff can create a new Observatory from an unresolved row and be returned to the approval queue afterward via ?obscode= prefill + a validated ?next= redirect, with an open-redirect fallback for an unsafe next"
    requirement: "SITE-02"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCreateObservatoryRoundTrip.test_get_with_obscode_and_next_prefills_form_initial"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCreateObservatoryRoundTrip.test_valid_create_with_safe_next_redirects_to_approval_queue"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestCreateObservatoryRoundTrip.test_unsafe_next_falls_back_to_detail_redirect"
        status: pass
    human_judgment: false

# Metrics
duration: 13min
completed: 2026-07-11
status: complete
---

# Phase 21 Plan 04: Decision-Time Site Resolution & CreateObservatory Round-Trip Summary

**A `if run.site is None:` guard closes the SITE-03 clobbering bug in `CampaignRunDecisionView.post()`, wires the staff-submitted `site_selection` field into approve-time resolution (SITE-02), and extends `CreateObservatory` with a `?obscode=` prefill + validated `?next=` redirect so the "Create new Observatory" link round-trips back to the approval queue.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-07-11T15:11:39Z
- **Completed:** 2026-07-11T15:25:01Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `CampaignRunDecisionView.post()` now wraps the `resolve_site()` call in `if run.site is None:` -- an already-resolved run (from CSV import, tier 1/2 auto-resolution, or a prior staff-UI resolution) is trusted and never re-resolved, closing the live-reachable Pitfall 3 clobbering bug where a projection-failure revert (back to `PENDING_REVIEW`, `run.site` left set) followed by a second approve unconditionally re-ran `resolve_site()`
- Unresolved runs (`run.site is None`) now read `request.POST.get('site_selection', '').strip()` (Plan 21-03's inline input), falling back to `run.site_raw` when blank, before calling `resolve_site(selection, create_placeholder=False)`
- `CreateObservatory.get_initial()` pre-fills the `obscode` form field from `?obscode=`; `CreateObservatory.get_success_url()` honors a validated `?next=` target (`url_has_allowed_host_and_scheme`), falling back to the existing Observatory detail redirect for a missing or unsafe (off-host/bad-scheme) `next` -- closing the T-21-06 open-redirect risk
- 9 new tests: 3 in `TestApproval` (SITE-03 clobber guard, Pitfall 3 two-attempt regression, T-21-04 oversized-`site_selection`), 2 in a new `TestSiteSelectionResolution` class (SITE-02 tier-1 typed-code resolution, no-fabrication regression on `260705-l1v`'s invariant), and 3 in a new `TestCreateObservatoryRoundTrip` class (obscode prefill, safe-`next` redirect, unsafe-`next` fallback)

## Task Commits

Each task was committed atomically:

1. **Task 1: D-06 clobber guard + site_selection read + SITE-03/T-21-04 tests** - `3d37f08` (fix)
2. **Task 2: CreateObservatory ?obscode= prefill + ?next= redirect + SITE-02 create-new tests** - `b9b0b64` (feat)

_Both tasks were `tdd="true"`; per this plan's own build-then-prove ordering, tests were written and verified alongside each task's implementation in the same commit (no separate standalone RED commit), matching the precedent already established in Plans 21-01/21-03._

## Files Created/Modified
- `solsys_code/campaign_views.py` - `CampaignRunDecisionView.post()`: `if run.site is None:` D-06 guard wrapping the `resolve_site()` call; reads `site_selection` POST field (falling back to `site_raw`) inside that branch
- `solsys_code/solsys_code_observatory/views.py` - `CreateObservatory.get_initial()` (new, `obscode` prefill); `CreateObservatory.get_success_url()` (rewritten to honor a validated `?next=`); new `url_has_allowed_host_and_scheme` import
- `solsys_code/tests/test_campaign_approval.py` - 3 new tests in `TestApproval`, new `TestSiteSelectionResolution` class (2 tests), new `TestCreateObservatoryRoundTrip` class (3 tests) + a module-level `_stub_to_observatory()` helper

## Decisions Made
- Kept the `except Exception:` revert block byte-for-byte unchanged per `21-PATTERNS.md` -- the D-06 fix is purely the new guard condition placed before the `resolve_site()` call, not a change to the failure-recovery contract
- Split the SITE-02 create-new round-trip into three focused test methods (prefill / safe-`next` redirect / unsafe-`next` fallback) rather than one combined test, matching this test file's existing one-assertion-focus-per-method convention
- Mocked `MPCObscodeFetcher.to_observatory()` directly (via a `side_effect` that creates and returns a real `Observatory` row) for the `CreateObservatory` round-trip tests, rather than only mocking `query()` -- `to_observatory()` reads several MPC-response dict keys with no `.get()` defaults, so a bare `query()` mock (which never populates the exact `self.obs_data` shape `to_observatory()` expects) would raise `MissingDataException` instead of exercising the success path being tested

## Deviations from Plan

None - plan executed exactly as written. The `except Exception:` revert path, `resolve_site()`, and `form_valid()`'s existing error handling were all left byte-for-byte unchanged, matching the plan's explicit "preserve unmodified" guidance.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. No new packages installed.

## Next Phase Readiness

- Phase 21's site-disambiguation UI (Plan 21-03) and decision-time resolution (this plan) are now fully wired end-to-end: the inline `site_selection` input, its fuzzy-matched `<datalist>`, and the "Create new Observatory" round-trip all function against the live approve path.
- All 3 Phase 21 requirements are shipped: SITE-01 (Plan 21-01), SITE-02/SITE-03 (this plan). VIEW-05 (Plan 21-02) was independent and already shipped.
- Full `python manage.py test solsys_code` suite (417 tests) passes; `ruff check .` / `ruff format --check .` clean on all Phase 21 files (pre-existing, unrelated warnings in `docs/notebooks/` and a debug script are out of this plan's scope, not touched).
- No paired demo notebook update required -- this plan doesn't touch any of the four CLAUDE.md-listed notebook-paired modules (`telescope_runs.py`, `load_telescope_runs.py`, `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`).
- No blockers. Phase 21 (site-disambiguation-submitter-contact-opt-in) is ready to close out -- all 4 plans complete.

---
*Phase: 21-site-disambiguation-submitter-contact-opt-in*
*Completed: 2026-07-11*

## Self-Check: PASSED

- FOUND: solsys_code/campaign_views.py
- FOUND: solsys_code/solsys_code_observatory/views.py
- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND commit: 3d37f08
- FOUND commit: b9b0b64
