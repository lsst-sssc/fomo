---
status: resolved
trigger: "Site-search endpoint (/campaigns/site-search/) reverted to 'No matches for this search' (68-byte 200 response) for G37/'Discovery Chan' terms that worked minutes earlier in the same session. Suspected: SAME CLASS as bug #1 — some other live-MPC record shape raises an exception in build/flatten, silently dropping the full 2712-code pool to the 23-entry local-only fallback; bug #1's 60s fallback TTL now cycles working/broken. Also suspect: selection_to_obscode() fix or its tests corrupted shared cache state."
created: 2026-07-16T08:15:00Z
updated: 2026-07-16T09:05:00Z
resolution: "Confirmed by human browser verification (combined with the site-resolve-list-old-names session's re-test): 'G37'/'Discovery Chan' produced suggestions with the shared cache re-warmed and test isolation in place. Committed as commit 9f99996 (fix) + adfcd4f (tests, incl. @override_settings(CACHES=LocMemCache) isolation + name_utf8/short_name hardening)."
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

reasoning_checkpoint:
  hypothesis: "Bug #3 is test/runserver shared-cache cross-contamination. settings.CACHES is a FileBasedCache at /tmp; Django does NOT isolate CACHES for tests, so the test process and the dev runserver share the same file cache under the same key 'mpc_obscode_candidates'. Running the bug-#2 verification suite (110 tests) — cache.clear() in setUp/tearDown + real build_site_candidates() writes with a mocked-failing query_all — WIPED the runserver's warmed 5733-entry pool. The runserver then cold-rebuilt on the next browser request and (during a burst of rapid requests + a transient/rate-limited MPC fetch, cached 60s per bug #1) served the degraded local-only pool with no G37 -> 'No matches'. Separately, a latent same-class-as-bug-#1 robustness gap exists: _flatten_mpc_candidates coerces old_names but not name_utf8/short_name, so a future non-str shape in those fields would re-trigger a silent full-pool drop."
  confirming_evidence:
    - "DECISIVE: warmed the shared /tmp cache to a full 5733 pool, ran the 110-test suite (and a 19-test subset) the user ran; both left the cache a MISS every time (warmed pool WIPED). Reproducible."
    - "Live query_all() = 2712 codes incl. G37 (old_names=None); _flatten_mpc_candidates() on live data SUCCEEDS (5717 entries, no exception); fresh build_site_candidates() = full 5733 pool with G37. Build/flatten path is healthy live -> the 'new field shape throws' hypothesis is DISCONFIRMED."
    - "Only the local-only degraded pool (26 entries, no G37/'Discovery') reproduces the exact zero-match symptom; BULK_MPC_FIXTURE contains G37 so fixture-poisoning would NOT cause it. Cache is now MISS (not a 24h fixture entry) -> consistent with an expired 60s degraded pool."
    - "Field-shape audit of all 2712 live records: obscode/name_utf8/short_name always str; old_names None(2652)/list(60). A probe with list-shaped name_utf8/short_name makes _flatten_mpc_candidates raise TypeError: unhashable type: 'list' -> latent same-class gap (not the current cause, but the user's 'future shape surprise' worry is valid)."
    - "override_settings(CACHES=LocMemCache) verified to redirect campaign_utils's imported `cache` proxy to LocMem -> decorating the cache-touching test classes fully isolates them from the shared /tmp file cache."
  falsification_test: "If, after decorating the cache-touching test classes with override_settings(LocMemCache), warming the shared /tmp cache to a full pool and re-running the 110-test suite STILL wiped the warmed pool (cache MISS after), the isolation fix would be wrong."
  fix_rationale: "Primary (the actual regression): isolate the test cache via @override_settings(CACHES=LocMemCache) on every cache-touching test class in both modules, so the suite can never wipe/poison the runserver's shared FileBasedCache again — directly addresses the proven root cause (running tests broke live search). Defensive (same-class-as-bug-#1): harden _flatten_mpc_candidates to coerce name_utf8/short_name to strings and skip non-dict records so ANY future MPC field-shape surprise degrades per-field/per-record instead of silently dropping the whole 2712-code pool — closes the 'future shape surprise' door the user asked to guard."
  blind_spots: "Cannot reproduce the exact transient MPC-fetch failure at 07:41 (network is up now, rebuild healthy) — the cache-wipe is the proven, reproducible in-code defect; the transient degraded rebuild on top of it is environmental and already mitigated by bug #1's 60s TTL. The decide-view POST path calls the real build_site_candidates() in un-patched tests (a pre-existing network/cache touch, not introduced by bug #3); cache isolation covers the cache half; the network half is out of scope."
next_action: Implement Fix A (cache isolation decorators, both test modules) + Fix B (_flatten hardening) + 2 regression tests; re-run the warm->test->check experiment to confirm the warmed pool SURVIVES; run full suites + ruff.

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: |
  GET /campaigns/site-search/?site_selection=G37&input_id=site-input-20 returns suggestion
  fragment listing "Lowell Discovery Telescope (G37)". Same for 'Discovery Chan'. Worked
  minutes earlier in the same session.
actual: |
  Returns 200 with a 68-byte body ("No matches for this search" empty-state fragment) for
  G37, G3, Dis, Disc, Disco, Discov, Discove, Discover. Single-char 'G' returns 200/2-byte
  (expected len<2 gate). Regression: search worked earlier this session.
errors: |
  No traceback in the user's runserver log — 200 responses, just empty results. Any
  exception is being swallowed inside build_site_candidates()'s broad except (same swallow
  mechanism as bug #1).
timeline: |
  Regression during browser verification of bug #2's selection_to_obscode() fix. Search
  worked earlier THIS session (that's how bug #2 was found). Now zero matches for the exact
  same terms.
reproduction: |
  GET /campaigns/site-search/?site_selection=G37&input_id=site-input-20 -> 68-byte empty
  result. Reproduce build_site_candidates() in a Django shell and check pool size.

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-07-16T08:25:00Z
  checked: cache.get('mpc_obscode_candidates') live right now (Django shell); local-only pool size
  found: cache MISS (no entry). Live query_all() returns 2712 codes incl. G37 (old_names=None). _flatten_mpc_candidates() on live data SUCCEEDS -> 5717 entries, no exception. Fresh build_site_candidates() -> FULL 5733-entry pool, G37 present, 'Discovery' keys present. NOT degraded.
  implication: The user's "new field shape raises an exception in flatten -> degraded pool" hypothesis is DISCONFIRMED. The build/flatten path is healthy on live data. The failure at 07:41 was a transient cache-state problem, not a code defect in the flatten path.

- timestamp: 2026-07-16T08:30:00Z
  checked: Contents of the two pools that could produce zero G37 matches — BULK_MPC_FIXTURE (what tests write into cache) vs the local-only degraded fallback pool
  found: BULK_MPC_FIXTURE flattened CONTAINS G37 + 'Discovery' (so a fixture-poisoned cache would still match G37 — does NOT explain symptom). The local-only degraded pool (26 entries, obscodes 268/269/309/705/809/E10/F65/G96) does NOT contain G37 or 'Discovery' — serving it yields exactly zero matches for G37/'Discovery Chan'.
  implication: The reported symptom is uniquely explained by the runserver serving the DEGRADED LOCAL-ONLY pool, not a fixture-poisoned pool. Cache is now MISS (not a lingering 24h fixture entry), consistent with a short-TTL (60s, MPC_CANDIDATE_FALLBACK_TTL_SECONDS) degraded pool that has since expired.

- timestamp: 2026-07-16T08:40:00Z
  checked: DECISIVE EXPERIMENT — warmed shared /tmp cache to full 5733 pool (simulating runserver's warmed state), ran the affected test suite, re-inspected the shared cache. Repeated for 19 tests and for the full 110 tests the user ran (test_campaign_approval + test_campaign_site_search).
  found: After BOTH runs the warmed full pool was GONE (cache MISS). settings.CACHES uses FileBasedCache at tempfile.gettempdir() (/tmp); Django's test runner does NOT override CACHES, so the test process and the runserver process share the same file cache. TestSiteFuzzyMatch.setUp/tearDown call cache.clear() (wipes ALL keys incl. the runserver's 'mpc_obscode_candidates'); several tests (e.g. test_build_site_candidates_cold_cache_mpc_failure_falls_back_to_local_pool, line 1555) call the REAL build_site_candidates() with a mocked-failing query_all, writing a test-DB-derived degraded pool into the shared key.
  implication: ROOT CAUSE. Running the test suite (exactly what the user did to verify bug #2 immediately before browser-testing) WIPES the runserver's warmed MPC pool from the shared FileBasedCache. The runserver is then forced into a cold rebuild on the next browser request; that rebuild (subject to a transient/rate-limited MPC fetch during the burst of 8 rapid requests, cached for only 60s per bug #1's fallback TTL) served the degraded local-only pool with no G37 -> the exact "No matches" regression. The trigger is test/runserver cache cross-contamination, NOT a new record shape and NOT bug #2's selection_to_obscode() code (which is purely additive and never touches build_site_candidates/flatten/TTLs).

## Eliminated
<!-- APPEND only - prevents re-investigating -->

- hypothesis: A new live-MPC record shape (not old_names, already handled) raises a different exception in build_site_candidates()/_flatten_mpc_candidates(), silently dropping the full pool to the 23/26-entry local-only fallback (the user's primary hypothesis, same class as bug #1).
  evidence: Live query_all() returns 2712 codes; _flatten_mpc_candidates() on the full live dict SUCCEEDS with no exception (5717 entries); fresh build_site_candidates() returns a full 5733 pool with G37. The build/flatten path is healthy on current live data. No exception is being thrown in that path.
  timestamp: 2026-07-16T08:25:00Z

- hypothesis: The selection_to_obscode() test suite poisoned the shared cache with the small BULK_MPC_FIXTURE pool, and that fixture pool lacks G37.
  evidence: The flattened BULK_MPC_FIXTURE CONTAINS G37 and 'Discovery' — a fixture-poisoned cache would still return G37, contradicting the symptom. The symptom is explained by the local-only degraded pool (no G37), not the fixture pool. (The test suite DOES contaminate the shared cache, but by WIPING it via cache.clear() and by writing degraded local-only pools — not by writing the G37-containing fixture pool as the served state.)
  timestamp: 2026-07-16T08:30:00Z

- hypothesis: bug #2's selection_to_obscode() code change degraded the pool build.
  evidence: git diff shows campaign_utils.py's bug-#2 change is purely additive (new selection_to_obscode() + _SELECTION_DISPLAY_OBSCODE_RE); it does not touch build_site_candidates(), _flatten_mpc_candidates(), _local_observatory_candidates(), or any cache TTL/key. campaign_views.py swaps build_site_candidates().get(...) for selection_to_obscode() only in the resolve/approve POST handlers, not in the search path.
  timestamp: 2026-07-16T08:20:00Z

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: |
  Test/runserver shared-cache cross-contamination. settings.CACHES uses a FileBasedCache at
  tempfile.gettempdir() (/tmp) and Django's test runner does NOT isolate CACHES (unlike the
  DB), so the test process and the dev runserver process read/write the SAME file cache under
  the SAME key ('mpc_obscode_candidates'). The campaign-approval / site-search test classes
  (a) call cache.clear() in setUp/tearDown (TestSiteFuzzyMatch etc.), wiping the runserver's
  warmed 5733-entry MPC candidate pool, and (b) call the real build_site_candidates() with a
  mocked-failing query_all() (e.g. test_build_site_candidates_cold_cache_mpc_failure_falls_back_
  to_local_pool), writing a tiny test-DB-derived degraded local-only pool into the shared key.
  When the user ran the bug-#2 verification suite (110 tests) immediately before browser-testing,
  it WIPED the runserver's warmed pool; the runserver then cold-rebuilt on the next browser
  request and — during a burst of 8 rapid requests with a transient/rate-limited MPC fetch,
  cached for only 60s (bug #1's MPC_CANDIDATE_FALLBACK_TTL_SECONDS) — served the degraded
  local-only pool, which contains no G37/'Discovery' entries, yielding "No matches for this
  search". Proven reproducible: warming the shared /tmp cache to the full 5733 pool then running
  the 110-test suite leaves the cache a MISS every time. NOT a new MPC record shape (flatten is
  healthy live) and NOT bug #2's selection_to_obscode() code (purely additive, never touches the
  search/build path).
fix: |
  (Applied, self-verified, NOT yet committed — pending human browser verification, bundled with bug #2.)

  Bug #3 FIX A (primary root cause — test/runserver cache isolation):
  - solsys_code/tests/test_campaign_approval.py — new module-level ISOLATED_TEST_CACHES
    (LocMemCache) constant; @override_settings(CACHES=ISOLATED_TEST_CACHES) on
    CampaignApprovalTestBase (inherited by every approval subclass whose decide-POST path calls
    the real build_site_candidates()) and on TestSiteFuzzyMatch (the standalone class that calls
    the real build_site_candidates() + cache.clear()). Verified override_settings redirects
    campaign_utils's imported `cache` proxy to LocMem, so all test cache traffic hits an
    in-memory cache, never the shared /tmp FileBasedCache the runserver serves from.
  - solsys_code/tests/test_campaign_site_search.py — imports ISOLATED_TEST_CACHES;
    @override_settings on ThrottleTest and SiteSearchViewTest (both cache.clear()/write throttle keys).

  Bug #3 FIX B (defensive hardening — same failure family as bug #1):
  - solsys_code/campaign_utils.py — new _candidate_str() helper coerces name_utf8/short_name to
    strings (non-str -> absent), and _flatten_mpc_candidates() now skips non-dict records. A
    future MPC field-shape surprise in ANY candidate field (as already happened with old_names)
    now degrades to "skip that field/record" instead of raising TypeError: unhashable type and
    silently dropping the whole ~2712-code pool.

  Bug #3 regression tests (solsys_code/tests/test_campaign_approval.py):
  - TestSiteFuzzyMatch.test_flatten_mpc_candidates_survives_shape_surprise_in_any_field — feeds a
    fixture with every candidate field as each surprising shape (list/dict/int/None/missing + a
    non-dict record); asserts flatten never raises and well-formed records survive. Confirmed it
    FAILS without Fix B (TypeError: unhashable type: 'list').
  - TestSiteSearchCacheIsolationRegression — writes a sentinel into a direct /tmp FileBasedCache
    handle, performs the exact cache.clear()/set() ops the suite performs under isolation, asserts
    the sentinel survives. Confirmed it FAILS without Fix A (default cache is FileBasedCache).

  NOTE: campaign_views.py is UNCHANGED by bug #3 — its diff is entirely bug #2 (selection_to_obscode
  call-site swaps). Bug #2's changes remain uncommitted and pending their own first verification.
verification: |
  - DECISIVE (fix works): warmed the shared /tmp cache to a full 5733 pool, ran the full 112-test
    suite -> the warmed runserver pool SURVIVED (cache HIT, size 5733, G37 present, Discovery keys
    present). Pre-fix the same experiment left the cache a MISS every time. Isolation confirmed
    effective end-to-end.
  - Both new regression tests are genuine guards (proven via negative checks): shape test ERRORs
    with TypeError: unhashable type: 'list' without Fix B; cache-isolation test FAILs with
    "FileBasedCache is not a LocMemCache" without Fix A.
  - Suites: ./manage.py test solsys_code.tests.test_campaign_approval
    solsys_code.tests.test_campaign_site_search -> Ran 112 tests, OK.
  - Quality gates: ruff check -> All checks passed; ruff format --check -> 3 files already formatted.
  - Shared cache re-warmed with the correct full 5733 pool for immediate browser verification.
  - PENDING: human confirmation in the real browser — BOTH (a) site-search suggestions for
    G37/'Discovery Chan' on the DCT placeholder row, and (b) bug #2's G37 Resolve action end-to-end.
files_changed:
  - solsys_code/campaign_utils.py  # bug #3 Fix B (also carries bug #2's additive selection_to_obscode)
  - solsys_code/tests/test_campaign_approval.py  # bug #3 Fix A decorators + 2 regression tests (also bug #2's TestSelectionToObscode)
  - solsys_code/tests/test_campaign_site_search.py  # bug #3 Fix A decorators (NEW to the change set)
</content>
</invoke>
<invoke name="Read">
<parameter name="file_path">/home/tlister/git/fomo_devel/.planning/debug/resolved/site-search-mpc-no-match.md