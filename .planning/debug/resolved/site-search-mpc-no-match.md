---
status: resolved
trigger: "Site search widget on the Sites Needing Review row (CampaignRun pk=30, site_raw='G96') now correctly sends site_selection param (query-param fix from 22-04 worked) but returns \"No matches for this search\" for both 'G96' and 'Mt. Lemmon' — a real MPC obscode/name that should resolve via the MPC candidate pool."
created: 2026-07-16T06:10:00Z
updated: 2026-07-16T07:40:00Z
resolution: "Fixed in commit 95d6244. _flatten_mpc_candidates() now normalizes list-shaped old_names via new _old_name_strings() helper (live MPC bulk API returns old_names as a JSON list for 60/2712 records, e.g. G96 -> ['Mt. Lemmon Survey']); the prior string assumption raised TypeError: unhashable type: 'list' in a dict-membership test, which build_site_candidates()'s broad except swallowed — discarding the whole 2712-code MPC pool and degrading to a 23-string local-only pool cached 24h. Degraded-fallback TTL also cut from 24h to 60s (MPC_CANDIDATE_FALLBACK_TTL_SECONDS). 4 regression tests added; 103 tests OK; ruff clean. User confirmed in browser: CampaignRun pk=30 now suggests 'Mt. Lemmon' (G96)."
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

reasoning_checkpoint:
  hypothesis: "_flatten_mpc_candidates() raises TypeError: unhashable type: 'list' on any MPC record whose old_names is a non-empty list (the live API's actual shape for 60/2712 records); build_site_candidates()'s broad except swallows it and discards the entire MPC pool, so MPC-only codes like G96 never reach the site-search matcher."
  confirming_evidence:
    - "Direct call: _flatten_mpc_candidates(live_2712_dict) raised TypeError at campaign_utils.py:275 ('candidate not in mapping' with a list operand)."
    - "Live old_names shape audit: 60 lists, 0 strings, 2652 None across 2712 records (e.g. '061' -> ['Uzhgorod'], G96 -> ['Mt. Lemmon Survey'])."
    - "build_site_candidates() (fresh, cache cleared, network up) returns 23 local-only strings; G96 absent — the MPC branch produced nothing."
    - "Live query_all() independently returns 2712 codes incl. G96, so network and data are fine; the failure is purely the flatten TypeError being swallowed."
    - "BULK_MPC_FIXTURE (test fixture) has old_names: None on every record — the list shape was never tested, explaining why Phase 22 tests passed."
  falsification_test: "If _flatten_mpc_candidates() were patched to handle list old_names and build_site_candidates() then STILL returned a local-only pool for a network-up rebuild, the hypothesis would be wrong."
  fix_rationale: "Fixing _flatten_mpc_candidates() to iterate list old_names (and tolerate str) makes the flatten succeed on real data, so the full ~2712-code MPC pool builds and caches — G96/'Mt. Lemmon Survey' become matchable. This addresses the root cause (data-shape mismatch), not a symptom. Secondary: cap the TTL of a degraded (MPC-failed) pool so a genuine transient MPC outage can no longer poison search for a full 24h (aligns code with its documented 'degrades gracefully' contract)."
  blind_spots: "to_observatory() (tier-2 resolve path, utils.py) also assigns old_names (possibly a list) to a CharField — a separate latent issue, out of scope for this search symptom. The existing poisoned FileBasedCache entry must be cleared for the fix to take effect immediately (else served stale for up to 24h)."
next_action: RESOLVED — user confirmed in browser that CampaignRun pk=30's site-search widget now suggests 'Mt. Lemmon' (G96). Fix committed (95d6244) and session archived. NOTE: the blind-spot flagged below (to_observatory() assigns list-shaped old_names to a CharField on the tier-2 resolve path) was subsequently hit for real on CampaignRun pk=21 (DCT/G37) and is tracked in a separate debug session: site-resolve-list-old-names.

---

## Symptoms

expected: |
  Typing 2+ characters of a real MPC obscode ('G96', Mt. Lemmon Survey / Catalina Sky Survey)
  or its display name ('Mt. Lemmon') into the Sites Needing Review row's site-search widget
  (CampaignRun pk=30, site_raw='G96', site_needs_review=True) should render a suggestion list,
  since G96 is a genuine MPC observatory code and build_site_candidates() merges local
  Observatory rows with the full live MPC obscode candidate list.
actual: |
  The server now receives the request correctly (query-param fix from 22-04 confirmed working
  — request is `?site_selection=<term>&input_id=site-input-30`, HTTP 200, non-empty body), but
  the rendered fragment says "No matches for this search" for every progressively-typed
  fragment of both 'G96' and 'Mt. Lemmon' ('G9', 'G96', 'Mt. ', 'Mt. L', 'Mt. Lemm').
errors: |
  None visible to the user — no exception, no 500. Runserver access log:
  [16/Jul/2026 06:01:36] "GET /campaigns/site-search/?site_selection=G9&input_id=site-input-30 HTTP/1.1" 200 68
  [16/Jul/2026 06:01:36] "GET /campaigns/site-search/?site_selection=G96&input_id=site-input-30 HTTP/1.1" 200 68
  [16/Jul/2026 06:01:50] "GET /campaigns/site-search/?site_selection=Mt.%20&input_id=site-input-30 HTTP/1.1" 200 68
  [16/Jul/2026 06:01:50] "GET /campaigns/site-search/?site_selection=Mt.%20L&input_id=site-input-30 HTTP/1.1" 200 68
  [16/Jul/2026 06:01:51] "GET /campaigns/site-search/?site_selection=Mt.%20Lemm&input_id=site-input-30 HTTP/1.1" 200 68
  All return 200 with a 68-byte body (consistent with a short "No matches" fragment).
timeline: |
  Reported immediately after Phase 22's gap-closure plan 22-04 fixed the query-param mismatch
  bug (UAT gaps 1 & 3) that previously caused these same widgets to render nothing at all with
  no server-visible request term. This is a NEW/DIFFERENT symptom appearing only after that fix:
  the request now reaches the view with the correct term, but the view's search finds zero
  candidates for a term that should genuinely match. Phase 22 was verified/marked complete
  immediately before this was found via manual browser testing (post-completion UAT).
reproduction: |
  1. Go to the staff approval queue (/campaigns/approval-queue/ or similar).
  2. In the "Sites Needing Review" card, find CampaignRun pk=30 (Test Campaign,
     telescope_instrument='CR-01 UAT Test 1m', site=None, site_raw='G96',
     site_needs_review=True, orchestrator-injected fixture).
  3. Type into that row's site-search input (id="site-input-30" per the logged requests).
  4. Type 'G96' (real MPC obscode, Mt. Lemmon Survey / Catalina Sky Survey) — no suggestions,
     "No matches for this search" shown.
  5. Type 'Mt. Lemmon' (the MPC display name for the same site) — same result.

## Evidence

- timestamp: 2026-07-16T06:20:00Z
  checked: Django shell — cache.get('mpc_obscode_candidates') before any call
  found: cache HIT holding only 23 candidate strings; G96 absent from keys and values; no key contains 'lemmon'.
  implication: The served pool is a tiny local-only pool (not the ~2712-code MPC pool), and it is being served from cache.

- timestamp: 2026-07-16T06:20:00Z
  checked: Local Observatory rows
  found: 8 rows total; no G96; no row with 'lemmon' in name; 1 placeholder ('NEEDS REVIEW: DCT'). _local_observatory_candidates() yields exactly 23 strings.
  implication: 23 cached strings == the local-only candidate count. The cached pool contains ZERO MPC entries. G96 is genuinely absent from every local row (as the fixture design intended).

- timestamp: 2026-07-16T06:20:00Z
  checked: Live MPCObscodeFetcher().query_all() reachability
  found: SUCCEEDED — 2712 codes returned, G96 present with name_utf8 'University of Arizona Mt. Lemmon Survey' and old_names ['Mt. Lemmon Survey'].
  implication: Network to MPC is available in this env; G96 IS in the live data. The network-unavailable hypothesis is FALSE.

- timestamp: 2026-07-16T06:22:00Z
  checked: Cleared the cache key and rebuilt via build_site_candidates()
  found: fresh rebuild STILL produced only 23 (local-only) strings; G96 still absent. Cache backend is FileBasedCache at /tmp (persistent across processes — shared with runserver).
  implication: NOT merely cache poisoning — even a fresh rebuild silently drops the entire MPC pool. build_site_candidates() is swallowing an error on the MPC branch. The cache poisoning is a secondary amplifier that persists the degraded pool for 24h.

- timestamp: 2026-07-16T06:25:00Z
  checked: Called _flatten_mpc_candidates() directly on the live 2712-code dict; audited old_names shapes
  found: old_names shapes across live data — list: 60, str: 0, None: 2652 (e.g. '061': ['Uzhgorod']). _flatten_mpc_candidates() RAISED `TypeError: unhashable type: 'list'` at campaign_utils.py:275 (`if candidate and candidate not in mapping:`) — a non-empty list old_names is used in a dict-membership test.
  implication: ROOT CAUSE. _flatten_mpc_candidates() assumes old_names is a string; the live MPC bulk API returns it as a list. The first record with a non-empty list aborts the whole flatten; the TypeError is caught by build_site_candidates()'s broad `except (..., TypeError, ...)` clause, discarding all 2712 MPC candidates and degrading to the local-only pool — then cached for 24h.

## Eliminated

- hypothesis: MPC network/query_all() is unavailable in this sandboxed dev environment, so only local rows can match (investigation-context hypothesis 1).
  evidence: Live query_all() SUCCEEDED, returned 2712 codes including G96. Network is reachable.
  timestamp: 2026-07-16T06:20:00Z

- hypothesis: CR-02's `.exclude(name__startswith=NEEDS_REVIEW_NAME_PREFIX)` regression broke the merge with the MPC pool (investigation-context hypothesis 2).
  evidence: _local_observatory_candidates() correctly yields 23 strings and excludes only the 1 DCT placeholder as intended; the merge code is fine. The MPC pool is empty for an unrelated reason (the flatten TypeError), not because of the exclude.
  timestamp: 2026-07-16T06:22:00Z

- hypothesis: substring/fuzzy matching threshold or 'Mt. ' normalization fails to match G96/'Mt. Lemmon' (investigation-context hypothesis 3).
  evidence: The matcher is never the problem — the candidate pool it receives contains no MPC entries at all. With a correctly-built pool (proven by the direct query_all result) 'Mt. Lemmon Survey' would be present as a substring-matchable candidate.
  timestamp: 2026-07-16T06:25:00Z

- hypothesis: Pure cache poisoning — a prior local-only fallback was cached for 24h and simply needs the network back.
  evidence: Clearing the cache and rebuilding fresh (with the network confirmed up) STILL produced a local-only pool. Cache poisoning is real but is a downstream amplifier, not the root cause. The MPC branch fails on every rebuild due to the flatten TypeError.
  timestamp: 2026-07-16T06:22:00Z

## Resolution

root_cause: |
  solsys_code/campaign_utils.py::_flatten_mpc_candidates() treats each MPC record's
  `old_names` field as a string, but the live MPC bulk obscodes API
  (MPCObscodeFetcher.query_all()) returns `old_names` as a JSON LIST (60 of 2712 records
  have a non-empty list, e.g. G96 -> ['Mt. Lemmon Survey']; 0 are strings). At line 275,
  `if candidate and candidate not in mapping:` performs a dict-membership test with the
  list as the operand, raising `TypeError: unhashable type: 'list'`. build_site_candidates()
  wraps the flatten in a broad `except (RequestException, ValueError, KeyError, TypeError,
  AttributeError)` that silently swallows the TypeError and falls back to a local-only pool,
  discarding ALL 2712 MPC candidates. That degraded local-only pool is then cached under
  'mpc_obscode_candidates' for 24h (FileBasedCache at /tmp, shared with runserver), so the
  outage persists across requests/restarts. Net effect: any MPC-only obscode (G96 was
  deliberately not seeded as a local Observatory) never appears in the site-search widget.
fix: |
  1. solsys_code/campaign_utils.py — added _old_name_strings() helper that normalizes an
     MPC record's old_names (list | str | None/missing) into a flat list of non-empty
     strings. _flatten_mpc_candidates() now folds each prior name in as its own candidate
     via this helper, so a list-valued old_names no longer reaches a dict key/membership
     operation and no longer raises TypeError: unhashable type: 'list'. The full ~2712-code
     MPC pool now builds successfully (verified: pool grew from 23 -> 5732 strings).
  2. solsys_code/campaign_utils.py — build_site_candidates() now tracks whether the MPC
     fetch succeeded and caches a degraded (local-only) fallback pool for only
     MPC_CANDIDATE_FALLBACK_TTL_SECONDS (60s) instead of the full 24h TTL, so a genuine
     transient MPC outage can no longer poison every site search for a whole day. A full
     pool (MPC ok) still caches for MPC_CANDIDATE_CACHE_TTL_SECONDS (24h).
  3. Cleared the poisoned FileBasedCache entry ('mpc_obscode_candidates' at /tmp, shared
     with runserver) and re-warmed it with the correct 5732-string pool so the fix takes
     effect immediately rather than after the stale 24h entry expired.
  4. solsys_code/tests/test_campaign_approval.py — 4 regression tests: list-valued old_names
     are folded in and the whole MPC pool survives (primary regression); string/None/missing
     old_names still supported (no regression); degraded pool uses the short fallback TTL;
     full pool uses the long TTL.
verification: |
  - Root cause reproduced: _flatten_mpc_candidates() on the live 2712-code dict raised
    TypeError: unhashable type: 'list' at campaign_utils.py:275 (pre-fix).
  - Post-fix cold rebuild: build_site_candidates() returns 5732 candidate strings incl. G96.
  - Every exact browser search term the user typed now returns matches:
    'G96' -> [('G96','G96')]; 'Mt. Lemmon' -> [('Mt. Lemmon Survey','G96')];
    'G9'/'Mt. '/'Mt. L'/'Mt. Lemm' all return non-empty suggestion lists.
  - Full affected suites pass: `./manage.py test solsys_code.tests.test_campaign_approval
    solsys_code.tests.test_campaign_site_search` -> Ran 103 tests, OK (incl. 4 new).
  - Quality gates clean: `ruff check` -> All checks passed; `ruff format --check` -> formatted.
  - Shared cache re-warmed with the correct full pool for immediate browser verification.
  - PENDING: human confirmation in the real browser (approval queue, CampaignRun pk=30 row).
files_changed:
  - solsys_code/campaign_utils.py
  - solsys_code/tests/test_campaign_approval.py
