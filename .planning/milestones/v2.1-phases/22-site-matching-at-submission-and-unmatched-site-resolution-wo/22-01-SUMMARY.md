---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
plan: 01
subsystem: api
tags: [django, htmx, difflib, django-cache, campaigns]

requires:
  - phase: 21-site-disambiguation-submitter-contact-opt-in
    provides: build_site_candidates(), fuzzy_match_candidates(), resolve_site() in campaign_utils.py
provides:
  - substring_or_fuzzy_match_candidates() -- substring-first, difflib-fallback site matcher (D-04)
  - _check_and_increment_throttle() + SITE_SEARCH_THROTTLE_LIMIT/WINDOW_SECONDS -- zero-dependency per-IP throttle (D-02)
  - SiteSearchView -- anonymous, throttled, GET-only HTML-fragment live-search endpoint (D-01/D-03)
  - campaigns:site_search URL
  - campaigns/partials/site_search_results.html suggestion fragment (D-05, escaped per T-22-01/T-22-10)
affects: [22-site-matching-at-submission-and-unmatched-site-resolution-wo-plan-02, 22-site-matching-at-submission-and-unmatched-site-resolution-wo-plan-03]

tech-stack:
  added: []
  patterns:
    - "Substring-containment-first matching with difflib fallback only on zero hits (bridges short partial queries against long official strings that difflib's 0.6 cutoff can't reach)"
    - "Zero-dependency per-IP fixed-window throttle via django.core.cache.add()/incr(), with a staff-session exemption"
    - "Server-side allowlist regex replacing an untrusted GET param with a safe default BEFORE it reaches template context, in addition to template-level |escapejs (defense in depth for an inline onclick= JS-string context)"

key-files:
  created:
    - solsys_code/tests/test_campaign_site_search.py
    - src/templates/campaigns/partials/site_search_results.html
  modified:
    - solsys_code/campaign_utils.py
    - solsys_code/campaign_views.py
    - solsys_code/campaign_urls.py

key-decisions:
  - "substring_or_fuzzy_match_candidates() placed directly below fuzzy_match_candidates() in campaign_utils.py, not replacing it -- fuzzy_match_candidates() gained a backward-compatible optional n=5 parameter instead of being reimplemented"
  - "_check_and_increment_throttle() stays in campaign_utils.py (not campaign_views.py) -- it only touches the cache framework (no request/HTTP types) and campaign_utils already owns the module's cache import and cache-key conventions"
  - "SiteSearchView exempts request.user.is_staff from the anonymous throttle so staff triaging the approval queue (Plan 02/03 widgets) never trip the public-abuse limit"
  - "Server-side _INPUT_ID_RE allowlist (^[-A-Za-z0-9_:.]+$) replaces a non-matching input_id with the default 'id_site_raw' before it reaches the template, on top of |escapejs in the template itself -- HTML auto-escaping alone is insufficient inside an inline onclick= JS-string context"

requirements-completed: [D-01, D-02, D-03, D-04, D-05]

coverage:
  - id: D1
    description: "Anonymous GET to campaigns:site_search returns HTTP 200 with a rendered HTML fragment (Content-Type text/html, never JSON)"
    requirement: "D-01/D-03"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_anonymous_get_returns_html_fragment_with_suggestion"
        status: pass
    human_judgment: false
  - id: D2
    description: "substring_or_fuzzy_match_candidates('faulkes', pool) surfaces both Faulkes sites via substring containment, beating difflib's 0.6 cutoff; falls back to difflib only when containment finds nothing"
    requirement: "D-04"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_site_search.py#SubstringOrFuzzyMatchCandidatesTest.test_substring_hit_surfaces_all_faulkes_candidates"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_site_search.py#SubstringOrFuzzyMatchCandidatesTest.test_substring_beats_difflib_cutoff_for_lowell"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_site_search.py#SubstringOrFuzzyMatchCandidatesTest.test_difflib_fallback_only_when_containment_finds_nothing"
        status: pass
    human_judgment: false
  - id: D3
    description: "Suggestions render as 'Display Name (obscode)' in both the visible text node and the click-to-fill value"
    requirement: "D-05"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_anonymous_get_returns_html_fragment_with_suggestion"
        status: pass
    human_judgment: false
  - id: D4
    description: "The (LIMIT+1)th request from one IP within the 60s window returns HTTP 429; a staff session is not throttled at the anonymous limit"
    requirement: "D-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_over_limit_anonymous_request_returns_429"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_staff_session_not_throttled_at_anonymous_limit"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_site_search.py#ThrottleTest.test_allows_up_to_limit_then_rejects"
        status: pass
    human_judgment: false
  - id: D5
    description: "A blank or 1-character query returns HTTP 200 with an EMPTY fragment WITHOUT calling build_site_candidates() (22-REVIEWS.md finding 4)"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_blank_query_returns_empty_fragment_without_building_pool"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_one_char_query_returns_empty_fragment_without_building_pool"
        status: pass
    human_judgment: false
  - id: D6
    description: "A hostile input_id GET param is replaced server-side with the default 'id_site_raw', and both input_id JS-string occurrences in the partial are |escapejs-escaped (22-REVIEWS.md finding 2)"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_hostile_input_id_is_replaced_with_default_fallback"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_campaign_site_search.py#SiteSearchViewTest.test_hostile_candidate_text_is_escaped_in_js_string_context"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 01: Shared Live-Search Endpoint Summary

**Anonymous, throttled HTMX live-search endpoint (`campaigns:site_search`) with a substring-first-then-difflib site matcher, backing Plan 02's public form and approval-queue widgets and Plan 03's sites-needing-review row.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2 completed (Task 1 TDD, Task 2 standard)
- **Files modified:** 3 modified, 2 created

## Accomplishments
- `substring_or_fuzzy_match_candidates()` in `campaign_utils.py`: case-insensitive substring containment first (surfaces every candidate containing the query, e.g. both Faulkes sites for `'faulkes'`), falling back to `fuzzy_match_candidates()` only when containment finds zero hits.
- `_check_and_increment_throttle()` + `SITE_SEARCH_THROTTLE_LIMIT`/`SITE_SEARCH_THROTTLE_WINDOW_SECONDS`: a ~15-line zero-dependency per-IP fixed-window throttle on `django.core.cache`, returning HTTP 429 over the limit (40 req/60s).
- `SiteSearchView`: anonymous (no `StaffRequiredMixin`), GET-only, throttled (staff-exempt), returns a rendered `campaigns/partials/site_search_results.html` HTML fragment -- never JSON.
- Server-side hardening beyond the base plan: a 2-character minimum-length gate that returns an empty fragment *before* `build_site_candidates()` is ever called (22-REVIEWS.md finding 4), and a server-side `_INPUT_ID_RE` allowlist that replaces a hostile `input_id` GET param with the safe default before it reaches the template, on top of `|escapejs` in the template itself (22-REVIEWS.md finding 2).

## Task Commits

Each task was committed atomically (Task 1 followed the full TDD RED/GREEN cycle):

1. **Task 1: Substring-first matcher + per-IP throttle in campaign_utils.py**
   - `4778f2f` (test) - RED: failing tests for `substring_or_fuzzy_match_candidates()` and `_check_and_increment_throttle()`
   - `92207e4` (feat) - GREEN: implemented both, plus a backward-compatible `n=5` param on `fuzzy_match_candidates()`
2. **Task 2: SiteSearchView + URL route + suggestion-fragment partial** - `a6119dd` (feat)

**Plan metadata:** (this commit, pending)

## Files Created/Modified
- `solsys_code/campaign_utils.py` - `substring_or_fuzzy_match_candidates()`, `_check_and_increment_throttle()`, throttle constants, `fuzzy_match_candidates(n=5)` param
- `solsys_code/campaign_views.py` - `SiteSearchView`, `_INPUT_ID_RE` allowlist
- `solsys_code/campaign_urls.py` - `campaigns:site_search` route
- `src/templates/campaigns/partials/site_search_results.html` - suggestion fragment (new file, new `partials/` directory)
- `solsys_code/tests/test_campaign_site_search.py` - 17 tests (new file)

## Decisions Made
- `fuzzy_match_candidates()` gained an optional `n: int = 5` keyword parameter (backward-compatible default) rather than being reimplemented inside the new matcher's fallback branch -- the existing single call site (`ApprovalQueueTable.render_site()`) is unaffected.
- `_check_and_increment_throttle()` was kept in `campaign_utils.py` per the plan's pre-decided disposition (LOW #8a from 22-REVIEWS.md): it's a pure cache-touching function, colocated with its `SITE_SEARCH_THROTTLE_*` constants for a single test patch target.
- Staff sessions are exempted from the anonymous throttle (Assumption A3 from 22-RESEARCH.md) so Plan 02/03's staff-facing widgets reusing this same endpoint never trip the public-abuse limit during active queue triage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Verification-command literalism for the `escapejs` grep count**
- **Found during:** Task 2 (writing `site_search_results.html`)
- **Issue:** The plan's acceptance criteria requires `grep -c 'escapejs' ... >= 4` (line-count, not occurrence-count). A single-line `onclick=` attribute with all four `|escapejs` filter applications on one physical line would only satisfy `grep -c` as `1`, failing the literal acceptance check even though the escaping itself was correct.
- **Fix:** Restructured the `onclick=` JS into multiple statements (`var inputEl = ...`, `var displayText = ...`, `var obscodeText = ...`, `inputEl.value = ...`) each on its own line, so each `|escapejs` occurrence lands on a distinct line. Functionally identical to the one-line version; arguably more readable.
- **Files modified:** `src/templates/campaigns/partials/site_search_results.html`
- **Verification:** `grep -c 'escapejs' src/templates/campaigns/partials/site_search_results.html` returns `5` (>= 4); all XSS-escaping tests still pass.
- **Committed in:** `a6119dd` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2, cosmetic/verification-alignment only).
**Impact on plan:** No functional change to the escaping behavior itself -- purely a formatting adjustment to satisfy the plan's literal verification command. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. No new packages installed (per 22-RESEARCH.md's Package Legitimacy Audit: `django-htmx` and `difflib` are both already present).

## Next Phase Readiness
- `campaigns:site_search` is live and fully tested; Plan 02 (public form + approval-queue widget wiring) and Plan 03 (sites-needing-review row) can now consume it directly via `hx-get`.
- `fuzzy_match_candidates()` remains behaviorally unchanged for its existing caller (`ApprovalQueueTable.render_site()`), confirmed by the full 453-test `solsys_code` suite passing with no regressions.
- No blockers for Plan 02/03.

---
*Phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 5 created/modified files verified present on disk; all 3 task commit hashes
(`4778f2f`, `92207e4`, `a6119dd`) verified present in git history.
