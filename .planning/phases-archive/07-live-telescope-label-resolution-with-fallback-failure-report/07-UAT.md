---
status: complete
phase: 07-live-telescope-label-resolution-with-fallback-failure-report
source:
  - .planning/phases/07-live-telescope-label-resolution-with-fallback-failure-report/07-01-SUMMARY.md
  - .planning/phases/07-live-telescope-label-resolution-with-fallback-failure-report/07-02-SUMMARY.md
started: 2026-06-24T01:33:41.000Z
updated: 2026-06-24T04:23:04.000Z
---

## Current Test

[testing complete]

## Tests

### 1. Successful live telescope-label resolution for a placed record
expected: |
  Run `./manage.py sync_lco_observation_calendar --proposal <code>` against a proposal
  with a placed (scheduled) record. For a record whose live LCO API call succeeds and
  resolves to a known (site, aperture_class) pair, the resulting CalendarEvent's
  `telescope` field shows the verified `SITECODE-CLASS` label (e.g. `LSC-1m0`,
  `COJ-2m0`), and the event's title has a clean prefix (no `[UNVERIFIED]`, no
  `[QUEUED]`).
result: issue
reported: |
  observation_id=4213127 (after running ./manage.py updatestatus --target_id 36 to
  populate scheduled_start, which was stale/None and itself explained the initial
  [QUEUED] reading) resolved via the live API to site='coj', telescope='1m0a'
  (aperture class 1m0) -- a real, successful API response. But SITE_TELESCOPE_MAP
  only has ('coj', '2m0'), so _derive_telescope('coj', '1m0') returned None and the
  record fell back to [UNVERIFIED] 1m0 instead of the verified COJ-1m0 label.
  Cross-referenced https://lco.global/observatory/sites/mpccodes/ (SITEID + first-3-
  chars-of-TELID, deduped) against the current dict: coj is missing both '1m0' and
  '0m4' entries (Siding Spring hosts 0m4/1m0/2m0 -- all three classes), and ogg is
  missing '0m4' (Haleakala hosts 2m0/0m4, not just 2m0). elp/lsc/cpt/tfn/sor already
  match the public table exactly.
severity: major
resolution: |
  Fixed via quick task 260623-su3 (commits cd3b17e/5583400/2473aa8): added
  ('coj','1m0')->'COJ-1m0', ('coj','0m4')->'COJ-0m4', ('ogg','0m4')->'OGG-0m4' to
  SITE_TELESCOPE_MAP, citing https://lco.global/observatory/sites/mpccodes/ as the
  confirmation source. Re-verified directly against the same real record
  (observation_id=4213127): re-running sync_lco_observation_calendar now produces
  title='COJ-1m0 1M0-SCICAM-SINISTRO', telescope='COJ-1m0', telescope_api_failed: 0
  -- matches Test 1's original expectation exactly. Full test suite 35/35 green.
result_after_fix: pass

### 2. API failure/timeout/unmapped-code falls back, not skipped
expected: |
  For a placed record whose live API call fails, times out, or returns a code not in
  `SITE_TELESCOPE_MAP`, the record still gets a CalendarEvent (it is not dropped/skipped).
  The event's telescope label is a coarse instrument-class token (`1m0`/`0m4`/`2m0`/`4m0`),
  the title is prefixed `[UNVERIFIED]`, and the description notes the lookup failed or was
  unverified.
result: pass
verified: |
  Live test (not mocked): user temporarily swapped in an invalid LCO_APIKEY in
  local_settings.py and re-ran sync_lco_observation_calendar against observation_id=
  4213127, producing a genuine auth failure from the real LCO API. Result: record was
  `updated` (not skipped), telescope='1m0' (coarse), title='[UNVERIFIED] 1m0
  1M0-SCICAM-SINISTRO', description notes "Telescope label unverified: live API
  lookup failed or returned an unmapped code." Summary line showed
  telescope_api_failed: 1, skipped: 0. Stderr log line was the fixed generic message
  naming only the observation_id -- no leaked key/body (also evidences Test 6).

### 3. Banner-stage (unscheduled) record never calls the API
expected: |
  For a record with no `scheduled_start` (not yet placed/queued), no live API call is
  made at all. The event gets the coarse fallback label and a `[QUEUED]` title prefix
  (never `[UNVERIFIED]` — that prefix is reserved for placed records whose API call
  failed).
result: pass

### 4. Summary line separates telescope_api_failed from skipped
expected: |
  The command's final summary line (e.g. `Done. proposal: X, LCO: created: N, updated: N,
  unchanged: N, skipped: N, extraction_failed: N, telescope_api_failed: N | SOAR: ...`)
  reports `telescope_api_failed` as its own count, distinct from `skipped` — a degraded
  (fallback) label is never counted as a hard skip.
result: pass

### 5. A per-record API failure never aborts the run
expected: |
  If one record's live API call fails (timeout, error, or malformed response), the
  command does not crash or stop early — every subsequent record in the same run is still
  processed and synced (with its own success/fallback outcome).
result: pass

### 6. No credential or response-body leakage on API failure
expected: |
  When a live API call fails, nothing printed to the terminal/log (stderr) or written
  into any CalendarEvent field contains the raw API response body or the `LCO_APIKEY`
  value — only a fixed, generic failure message naming the affected `observation_id`.
result: pass
verified: |
  Already evidenced by Test 2's live auth-failure run: stderr showed only "Telescope
  API lookup failed or returned an unmapped code for observation_id='4213127'; using
  fallback label." -- no key/body content. User confirmed pass explicitly.

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0

## Gaps

- truth: "A successfully-resolved (site, aperture_class) pair from the live API maps through SITE_TELESCOPE_MAP to the correct verified label, for every real LCO-network site/class combination."
  status: resolved
  reason: "User reported: real record observation_id=4213127 resolved to ('coj', '1m0') via a successful live API call, but SITE_TELESCOPE_MAP has no ('coj','1m0') entry (only ('coj','2m0')), so it fell back to [UNVERIFIED] instead of the correct verified label. Cross-referenced https://lco.global/observatory/sites/mpccodes/ : coj is missing '1m0' and '0m4'; ogg is missing '0m4'. elp/lsc/cpt/tfn/sor already complete."
  severity: major
  test: 1
  artifacts:
    - solsys_code/management/commands/sync_lco_observation_calendar.py (SITE_TELESCOPE_MAP)
  missing:
    - "('coj', '1m0') -> 'COJ-1m0'"
    - "('coj', '0m4') -> 'COJ-0m4'"
    - "('ogg', '0m4') -> 'OGG-0m4'"
  resolved_by: "quick task 260623-su3 (commits cd3b17e, 5583400, 2473aa8); re-verified live against observation_id=4213127"
