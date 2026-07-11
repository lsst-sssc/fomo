---
phase: 21-site-disambiguation-submitter-contact-opt-in
plan: 01
subsystem: api
tags: [difflib, django-cache, mpc-obscodes, fuzzy-matching, django]

# Dependency graph
requires:
  - phase: 18-uncertain-scheduling-investigation-spike
    provides: locked difflib fuzzy-match library choice (18-DECISION.md Criterion 4)
provides:
  - MPCObscodeFetcher.query_all() bulk-fetch method (solsys_code_observatory/utils.py)
  - campaign_utils.build_site_candidates() cached merged candidate pool
  - campaign_utils.fuzzy_match_candidates() difflib wrapper
  - TestSiteFuzzyMatch test class + BULK_MPC_FIXTURE reusable fixture
affects: [21-03 (site-disambiguation UI wiring, consumes these helpers)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "django.core.cache.cache TTL-cache pattern (mirrors campaign_gap.py) reused for a global, non-parameterized cache key"
    - "difflib.get_close_matches(n=5, cutoff=0.6) fuzzy matching against a flattened {string: obscode} candidate pool"
    - "Wave-0 scaffold tests reference not-yet-existing helpers via module attribute access (campaign_utils.foo) rather than named import, so RED failures are localized AttributeErrors instead of a module-wide ImportError"

key-files:
  created: []
  modified:
    - solsys_code/solsys_code_observatory/utils.py
    - solsys_code/campaign_utils.py
    - solsys_code/tests/test_campaign_approval.py

key-decisions:
  - "old_names included as one whole string (not split) per RESEARCH.md Open Question 2 recommendation"
  - "Local Observatory candidates merge in after the MPC pool via dict.setdefault(), so an already-vetted local record's display string wins any first-seen collision over raw MPC bulk data"
  - "MPC bulk-fetch failure is caught narrowly (RequestException, ValueError, KeyError, TypeError) and logged at debug level, falling back to a local-only pool -- never raises into the caller"

patterns-established:
  - "MPCObscodeFetcher.query_all() is a distinct sibling method to query() -- json={} body triggers bulk mode; self.obs_data shape differs from query()'s and must never be passed to to_observatory()"

requirements-completed: [SITE-01]

coverage:
  - id: D1
    description: "MPCObscodeFetcher.query_all() bulk-fetches the full MPC obscode list (json={} body) and stores it on self.obs_data without disturbing query()'s single-code contract"
    requirement: "SITE-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_query_all_returns_fixture_dict_without_mutating_query_contract"
        status: pass
      - kind: unit
        ref: "solsys_code.solsys_code_observatory test suite (14 tests, query()/to_observatory() contract)"
        status: pass
    human_judgment: false
  - id: D2
    description: "build_site_candidates() merges local Observatory rows with a 24h-cached bulk MPC fetch into a flattened {string: obscode} pool, falling back to local-only on MPC failure"
    requirement: "SITE-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_build_site_candidates_flattens_obscode_name_and_short_name"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_build_site_candidates_caches_result_under_fixed_key"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_build_site_candidates_second_call_reuses_cache_not_query_all"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_build_site_candidates_cold_cache_mpc_failure_falls_back_to_local_pool"
        status: pass
    human_judgment: false
  - id: D3
    description: "fuzzy_match_candidates() wraps difflib.get_close_matches to return ranked (display_string, obscode) pairs, correctly returning [] for an acronym/nickname (e.g. 'DCT') that difflib cannot bridge"
    requirement: "SITE-01"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_fuzzy_match_candidates_exact_hit_includes_obscode"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_fuzzy_match_candidates_near_typo_scores_above_cutoff"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_approval.py#TestSiteFuzzyMatch.test_fuzzy_match_candidates_nickname_returns_no_matches"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-07-11
status: complete
---

# Phase 21 Plan 01: Site Candidate Pool & Fuzzy-Match Helpers Summary

**Bulk MPC obscode fetch (`MPCObscodeFetcher.query_all()`) feeding a 24h-cached, local+MPC merged candidate pool (`build_site_candidates()`) and a `difflib`-based fuzzy matcher (`fuzzy_match_candidates()`), plus the Wave-0 `TestSiteFuzzyMatch` scaffold with a reusable bulk-MPC fixture.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-11T11:36:33Z
- **Completed:** 2026-07-11T11:44:40Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- `MPCObscodeFetcher.query_all()` — a new sibling method to `query()` that sends `json={}` to trigger the MPC API's bulk-list mode, storing the full obscode-keyed dict on `self.obs_data` and returning it, with `query()`'s single-code contract left completely unchanged
- `campaign_utils.build_site_candidates()` — merges the local `Observatory` table with a 24h-cached bulk MPC fetch (mirroring `campaign_gap.py`'s `cache.get`/`cache.set` TTL pattern) into a flattened `{candidate_string: obscode}` pool; falls back to a local-only pool on any MPC network/parse failure, never raising into the caller
- `campaign_utils.fuzzy_match_candidates()` — wraps `difflib.get_close_matches(n=5, cutoff=0.6)` and resolves matches back to obscodes; correctly returns `[]` for a genuine acronym/nickname case (`'DCT'`), confirming Phase 21's RESEARCH.md Pitfall 2 finding in code
- `TestSiteFuzzyMatch` test class + `BULK_MPC_FIXTURE` — 8 new tests covering the bulk fetch, candidate-pool flattening/caching/fallback, and fuzzy-match exact-hit/near-typo/no-match cases, all mocked at the `requests.get`/`MPCObscodeFetcher.query_all` boundary (never hits the live MPC API)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wave-0 scaffold — TestSiteFuzzyMatch class + bulk-MPC fixture** - `f2b8418` (test)
2. **Task 2: MPCObscodeFetcher.query_all() bulk-fetch method** - `d571b18` (feat)
3. **Task 3: Cached candidate-pool + difflib fuzzy-match helpers** - `5c1ac30` (feat)

_Note: Tasks 2 and 3 were `tdd="true"` — the RED state was already established by Task 1's scaffold (AttributeError on the not-yet-existing methods), so each GREEN commit both implemented the method and turned the corresponding Task-1 test cases green in the same commit; no separate standalone RED commit was needed per task._

## Files Created/Modified
- `solsys_code/solsys_code_observatory/utils.py` - `MPCObscodeFetcher.query_all()` bulk-fetch method
- `solsys_code/campaign_utils.py` - `build_site_candidates()`, `fuzzy_match_candidates()`, `_flatten_mpc_candidates()`, `_local_observatory_candidates()` helpers, `MPC_CANDIDATE_CACHE_TTL_SECONDS` constant, module logger
- `solsys_code/tests/test_campaign_approval.py` - `BULK_MPC_FIXTURE`, `TestSiteFuzzyMatch` (8 tests)

## Decisions Made
- `old_names` included as one whole string (not split) per RESEARCH.md Open Question 2's recommendation — spot checks found it `None` for the vast majority of real records, so splitting has low practical impact
- Local `Observatory` candidates merge in after the MPC pool via `dict.setdefault()` so an already-vetted local record's display string wins any first-seen collision over raw MPC bulk data (the reverse of RESEARCH.md's example ordering, but functionally equivalent since collisions are rare and this ordering favors trusted local data)
- `build_site_candidates()`'s bulk-fetch exception handling mirrors `resolve_site()`'s narrow `except (RequestException, ValueError, KeyError, TypeError)` fallthrough discipline exactly, logging at `debug` level per this codebase's established convention

## Deviations from Plan

None - plan executed exactly as written. `resolve_site()` source is byte-for-byte unchanged (confirmed via `git diff` scoped to that function).

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required. No new packages installed (`difflib` is stdlib, `django.core.cache`/`requests` already project dependencies).

## Next Phase Readiness
- `build_site_candidates()`/`fuzzy_match_candidates()` are ready for Plan 21-03 to wire into `ApprovalQueueView.get_context_data()` and `ApprovalQueueTable.render_site()` per RESEARCH.md's Pattern 4/5 architecture — build the pool once per request (Pitfall 5), pass via `candidate_pool=` kwarg
- `MPCObscodeFetcher.query_all()` is ready for reuse; `to_observatory()` must never be called on a `query_all()` result (documented in its docstring)
- No blockers. `resolve_site()`, `CampaignRunDecisionView.post()`, and `CreateObservatory` are untouched by this plan and remain in scope for Plan 21-04 (SITE-02/SITE-03) as planned

---
*Phase: 21-site-disambiguation-submitter-contact-opt-in*
*Completed: 2026-07-11*

## Self-Check: PASSED

- FOUND: solsys_code/solsys_code_observatory/utils.py
- FOUND: solsys_code/campaign_utils.py
- FOUND: solsys_code/tests/test_campaign_approval.py
- FOUND commit: f2b8418 (test)
- FOUND commit: d571b18 (feat)
- FOUND commit: 5c1ac30 (feat)
