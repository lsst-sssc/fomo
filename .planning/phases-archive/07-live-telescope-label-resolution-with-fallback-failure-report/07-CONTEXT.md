# Phase 7: Live Telescope-Label Resolution with Fallback & Failure Reporting - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace `sync_lco_observation_calendar.py`'s current telescope-label
derivation — `_derive_telescope(record.parameters['site'])`, which reads a
flat `'site'` key that real `ObservationRecord.parameters` data doesn't
reliably have — with a per-record LCO Observation Portal API call that
resolves the actual site/enclosure/telescope for a **placed (scheduled)**
record, mapped through a verified static dict covering all 8 real
LCO-network sites. When that call fails, times out, or returns a code absent
from the dict, the record still gets a `CalendarEvent`, labeled with a
coarse instrument-class fallback (`1m0`/`0m4`/`2m0`) instead of being
skipped — visibly marked as degraded, counted separately from the existing
`skipped_count`, and never aborting the run or leaking credentials/response
bodies into logs. Instrument-type extraction (Phase 6) and
proposal/facility selection (Phase 5) are untouched.

</domain>

<decisions>
## Implementation Decisions

### Resolution timing (when the API call happens)
- **D-01:** The per-record LCO API call is only attempted for **placed**
  records — i.e. `scheduled_start`/`scheduled_end` are populated. A
  queue-banner record (not yet scheduled) gets the coarse fallback label
  immediately, with **no API call attempted** — there is nothing to look up
  yet.
- **D-02:** The new SYNC-06 fallback/API-failure counter only increments on
  an actual failed/timed-out/unmapped-code API call for a **placed** record.
  A banner-stage record's coarse label does **not** increment this counter —
  it isn't a real failure, just "not yet resolvable." Conflating the two
  would make the new counter noisy and useless for spotting genuine API
  degradation.

### Verified static dict (TELESCOPE-01)
- **D-03:** Collapse by **(site, aperture class)** pair, not by individual
  `enclid`/`telid`. A site with only one telescope class (e.g. `coj`, `ogg`,
  `sor`) gets exactly one label; a site with multiple classes present (e.g.
  `lsc` has both 1m0 domes and possibly a 0m4) gets one label **per class**
  — but multiple domes of the *same* class at the same site still collapse
  to one label. This extends the existing collapsing precedent (Phase
  4-6's `'FTS'` covering all `coj` instruments; PROJECT.md's deliberate
  Magellan Baade/Clay ambiguity) rather than breaking from it, while still
  surfacing the one distinction (aperture class) that's operationally
  meaningful and that the fallback label already exposes.
- **D-04:** Label format is `SITECODE-CLASS`, hyphenated, site code
  uppercased, class token drawn from the exact same vocabulary as the
  coarse fallback (`1m0`/`0m4`/`2m0`) — e.g. `'LSC-1m0'`, `'LSC-0m4'`,
  `'CPT-1m0'`. Using the identical class token in both the verified label
  and the fallback label means a label "flip" between verified and fallback
  changes only the site-prefix presence, keeping the aperture-class part
  stable and recognizable either way.
- **D-05:** **Migrate the 3 existing entries too**, for full dict
  consistency: `'coj'` (currently `'FTS'`) becomes `'COJ-2m0'`, `'ogg'`
  (currently `'FTN'`) becomes `'OGG-2m0'`, `'sor'` (currently `'SOAR'`)
  becomes `'SOR-<class>'` (confirm SOAR's real aperture class during
  research — likely `4m0`). This is an explicit, accepted one-time visible
  label change on already-synced historical `CalendarEvent`s at these
  sites going forward. **This dict (`SITE_TELESCOPE_MAP` in
  `sync_lco_observation_calendar.py`) is separate from Stage 1/2's
  `telescope_runs.py:SITES` dict** (classical-run ingest naming, e.g. `FTS`
  at Siding Spring for the classical schedule) — this migration does **not**
  touch that other dict or feature.
- All 8 real LCO-network sites (per PROJECT.md's MPC-code reference table —
  `ogg`, `elp`, `lsc`, `cpt`, `coj`, `tfn`, `tlv`, `sor`) must be covered;
  exact per-site aperture classes present need confirming during
  research/planning (not all sites necessarily have every class).

### Fallback visibility (TELESCOPE-04)
- **D-06:** Add a new title prefix, mirroring the existing
  `[QUEUED]`/`[EXPIRED]`/`[CANCELLED]`/`[FAILED]` convention — the
  calendar's day view only shows the truncated title at a glance (16-18
  chars; confirmed in installed `tom_calendar`'s
  `calendar.html` template), so the `telescope` field's coarse-class token
  and the `description`'s failure note (both already required by
  TELESCOPE-04) aren't visible without opening the event.
- **D-07:** The new prefix applies **only** to a placed record whose API
  call genuinely failed/timed out/returned an unmapped code (same scope as
  D-02's counter). Banner-stage records keep their existing `[QUEUED]`
  prefix unchanged and do **not** get the new prefix — avoids redundancy
  with `[QUEUED]`, which already communicates "not yet resolved."
- **D-08:** Prefix text is `[UNVERIFIED]`.
- **D-09 (open, Claude's discretion):** How `[UNVERIFIED]` combines with the
  existing terminal-state prefixes (`[EXPIRED]`/`[CANCELLED]`/`[FAILED]`,
  per Phase 4's D-04 priority rule that terminal prefixes beat `[QUEUED]`)
  if a record reaches a terminal state after an API-failure fallback was
  already applied. Likely the terminal prefix takes priority/replaces
  `[UNVERIFIED]` too (consistent with D-04's existing precedent), but the
  planner should confirm and document the exact combination rule explicitly
  rather than leaving it implicit.

### API call discipline (SYNC-08/09 — already locked by ROADMAP, restated here for planning)
- **D-10:** Explicit timeout of **10 seconds**, single attempt, no
  retry/backoff loop. There is no existing HTTP-timeout precedent anywhere
  else in this codebase (`JPLSBDBQuery.run_query()` in `views.py:543` calls
  `requests.get(url)` with no timeout at all — a known anti-pattern, not a
  convention to follow) — this is the first explicit timeout introduced in
  `solsys_code/`.
- **D-11:** No raw response body or credential/API-key content may appear
  in any logged error/exception message for a failed API call (SYNC-09).
  Note for the researcher/planner: `tom_observations.facilities.ocs.make_request()`
  (the library helper `LCOFacility`/`SOARFacility` calls internally) raises
  exceptions whose messages **embed `response.content`** directly (e.g.
  `ImproperCredentialsException('OCS: ' + str(response.content))`) — any
  `except`/log path that stringifies and logs that exception verbatim would
  violate SYNC-09. The catch site must construct its own fixed, generic
  message rather than logging the caught exception's `str()` directly.

### Claude's Discretion
- Exact label string for SOAR's real aperture class (D-05) — confirm
  against `tom_observations.facilities.soar` or real data rather than
  guessing.
- Exact per-site aperture-class inventory for the 5 newly-added sites
  (`elp`, `lsc`, `cpt`, `tfn`, `tlv`) — confirm during research, not
  assumed from the bare MPC-code table alone.
- D-09's exact prefix-combination/priority rule when `[UNVERIFIED]` and a
  terminal-state prefix could both apply to the same record.
- Whether the new SYNC-06 fallback counter is reported per-facility in the
  run summary (mirroring the existing `created`/`updated`/`unchanged`/
  `skipped`/`extraction_failed` per-facility breakdown established in
  Phase 5's D-08) — not explicitly discussed, but the established summary
  convention strongly suggests following the same per-facility pattern;
  planner should confirm this is the obvious extension and not a fresh
  decision point.
- Exact helper/method structure for the new per-record API call (e.g.
  whether to extend `OCSFacility.get_observation_status()` — which already
  hits `/api/requests/{id}/observations/` but only returns `state`/
  `scheduled_start`/`scheduled_end`, not site/enclosure/telescope — or add a
  new dedicated call) — this is a HOW/implementation question for
  research, not a user-vision decision.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `.planning/REQUIREMENTS.md` — TELESCOPE-01, TELESCOPE-02, TELESCOPE-03,
  TELESCOPE-04, SYNC-06, SYNC-07, SYNC-08, SYNC-09 (this phase's locked
  acceptance criteria)
- `.planning/ROADMAP.md` §"Phase 7: Live Telescope-Label Resolution with
  Fallback & Failure Reporting" — success criteria 1-5 and dependency on
  Phase 6
- `.planning/milestones/v1.3-ROADMAP.md` §"Phase 7" — same success criteria,
  milestone-scoped copy
- `.planning/PROJECT.md` — "Current Milestone: v1.3 Full LCO Facility Sync"
  section, including the v1.2 real-data bug note (`parameters['site']`
  doesn't reliably exist on real records — the exact problem this phase
  fixes) and the LCO site → MPC code reference table (8 real sites: `ogg`,
  `elp`, `lsc`, `cpt`, `coj`, `tfn`, `tlv`, `sor`)
- `.planning/STATE.md` — "Blockers/Concerns" section: 3 open research gaps
  for this phase (exact JSON key names in `/api/requests/<id>/observations/`
  block responses not confirmed against a live response; whether
  `FACILITIES['SOAR']` needs an explicit settings.py entry — already
  resolved by Phase 5 D-04/D-05; `tlv` site presence in installed
  `LCOSettings.get_sites()`/`SOARSettings.get_sites()` not confirmed)

### Existing code (Phase 4-6)
- `solsys_code/management/commands/sync_lco_observation_calendar.py` —
  `SITE_TELESCOPE_MAP` (lines 18-22, the dict this phase replaces/extends
  per D-03/D-04/D-05); `_derive_telescope()` (lines 71-86, currently takes a
  bare site code and raises `KeyError` if unmapped — signature/behavior
  will need to change to accept the fully-qualified `siteid-enclid-telid`
  code and support a fallback path); `_build_event_fields()` (lines
  225-267, calls `_derive_telescope(record.parameters['site'])` — the exact
  call site this phase must change to use the new API-call+fallback logic
  instead of trusting a flat `parameters['site']` key that often doesn't
  exist); `_FAILURE_PREFIX_BY_STATUS`/`_failure_prefix()` (lines 24-37,
  55-68 — existing title-prefix-priority convention that D-09's
  `[UNVERIFIED]`-vs-terminal-prefix combination rule must integrate with);
  `Command.handle()` (lines 295-402 — existing per-facility counters dict
  and summary-line construction that the new fallback counter extends)
- `solsys_code/tests/test_sync_lco_observation_calendar.py` — existing test
  conventions: `unittest.mock.patch` already used for mocking; `_create_record()`/
  `_parameters()` fixture helpers to extend for placed-vs-banner and
  API-success-vs-failure test cases. Per CLAUDE.md, any new `Target`
  fixtures must use `NonSiderealTargetFactory`, never `SiderealTargetFactory`.
- `.planning/phases/06-correct-instrument-type-extraction/06-CONTEXT.md` —
  Phase 6's decisions this phase builds on (instrument extraction is a
  prerequisite signal for the coarse fallback's aperture-class derivation
  per the milestone's phase-ordering rationale in PROJECT.md)
- `.planning/phases/05-multi-proposal-multi-facility-selection/05-CONTEXT.md`
  — Phase 5's per-facility dispatch dict and per-facility summary
  breakdown (D-08) convention this phase's new fallback counter likely
  extends
- `.planning/phases/04-lco-queue-sync-command/04-CONTEXT.md` — Phase 4's
  D-04 terminal-state-prefix-priority-over-`[QUEUED]` precedent, directly
  relevant to D-09's open combination-rule question

### Third-party library (not in this repo) — confirmed against installed `tomtoolkit`
- `tom_observations/facilities/ocs.py:1548-1568` — `OCSFacility.get_observation_status(observation_id)`
  already calls `GET /api/requests/{id}` and `GET /api/requests/{id}/observations/`,
  but only extracts/returns `state`, `scheduled_start`, `scheduled_end` from
  the response — does **not** currently surface site/enclosure/telescope
  fields. This phase's new call will need to either extend this method or
  add a new one that parses those fields from the same or a related
  response.
- `tom_observations/facilities/ocs.py:180-186` — `make_request(*args, **kwargs)`
  module-level helper used internally by `OCSFacility`/`LCOFacility`/
  `SOARFacility`; raises `ImproperCredentialsException`/`forms.ValidationError`
  with `response.content` embedded directly in the exception message on
  4xx responses, and calls `response.raise_for_status()` (raises
  `requests.exceptions.HTTPError` on other error statuses) — directly
  relevant to D-11's credential/body-leak constraint.
- `tom_observations/facilities/ocs.py:1406-1407` — `get_observation_url()`
  (unchanged, already in use since Phase 4 D-01)
- `src/fomo/settings.py:214-226` — `FACILITIES['LCO']`/`FACILITIES['SOAR']`
  dicts (`portal_url`, `api_key` via `os.environ.get('LCO_APIKEY', '')`) —
  the credential source D-11 must never let leak into logs. Note: this file
  currently shows as modified in `git status` (uncommitted
  `LCO_APIKEY`-related edit per prior session's memory) — confirm its state
  before this phase's planning/execution.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `OCSFacility.get_observation_status()` — already hits the right
  `/api/requests/{id}/observations/` endpoint; likely extend rather than
  duplicate, once the exact block-response field names for
  site/enclosure/telescope are confirmed against a live response (open
  research gap).
- `sync_lco_observation_calendar.py`'s existing per-record
  catch-log-continue error handling (already extended once for D-06's
  `extraction_failed` counter in Phase 6) — extend again for the new
  fallback/API-failure counter, same pattern.
- `test_sync_lco_observation_calendar.py`'s `unittest.mock.patch` usage —
  the established way to simulate a slow/failing API response for the
  SYNC-08 single-attempt-no-retry test and the SYNC-09 no-leak test.

### Established Patterns
- Title-prefix priority convention (terminal-state prefixes > `[QUEUED]`,
  Phase 4 D-04) — D-09 must explicitly extend this ordering to include the
  new `[UNVERIFIED]` prefix.
- Per-facility counters dict + summary-line breakdown (Phase 5 D-08,
  extended in Phase 6 with `extraction_failed`) — the natural place to add
  the new fallback counter.
- No-churn idempotency (only `.save()` when fields differ) — the
  `telescope` field is one of the compared fields, so a label flip between
  verified and fallback (or vice versa) on re-run will already trigger an
  update under the existing comparison logic, satisfying TELESCOPE-04's
  "visible information, not silently hidden churn" requirement without new
  code for that part specifically.

### Integration Points
- `_derive_telescope(record.parameters['site'])` call site in
  `_build_event_fields()` is replaced by: (1) check if the record is placed
  (D-01); (2) if so, attempt the per-record API call with the D-10 timeout;
  (3) on success, map the returned fully-qualified code through the new
  D-03/D-04/D-05 dict; (4) on any failure/timeout/unmapped-code, or if not
  yet placed, fall back to a coarse instrument-class label derived from the
  already-extracted instrument type (Phase 6).
- The new fallback counter slots into the existing per-facility `counters`
  dict in `Command.handle()` alongside `extraction_failed`.

</code_context>

<specifics>
## Specific Ideas

No UI/template changes — this phase remains a management-command-only
change (matches Phase 4-6 precedent); the only "visible" change is the
`CalendarEvent.title`/`telescope`/`description` field content itself,
rendered through `tom_calendar`'s existing unmodified templates. No sample
real LCO API `/observations/` block response was provided during this
discussion — the researcher must confirm exact JSON field names against a
live response or the user-shared `observe.lco.global.json` reference (used
during Phase 6 discussion) if it contains a placed/scheduled example.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. Retry/backoff and
caching/memoizing the per-record API result were already explicitly placed
in REQUIREMENTS.md's "Out of Scope"/v2 sections before this discussion and
were not re-raised.

### Reviewed Todos (not folded)
- `2026-06-18-status-aware-calendar-event-coloring-telescope-proposal-keye.md`
  — reviewed (`todo.match-phase` score 0.9 on this phase, keyword overlap
  on "telescope"/"calendar"/"status"); confirmed still out of scope — it's
  a visual/UI coloring change requiring a `tom_calendar` template override,
  distinct from this phase's label-resolution/fallback-reporting scope.
  Already deferred at v1.2 close and reviewed-not-folded again in Phase 6;
  left pending/deferred unchanged.

</deferred>

---

*Phase: 7-Live Telescope-Label Resolution with Fallback & Failure Reporting*
*Context gathered: 2026-06-21*
