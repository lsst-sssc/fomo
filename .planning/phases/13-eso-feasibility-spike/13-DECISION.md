# Phase 13: ESO Feasibility Spike - Decision

**Investigated:** 2026-07-01
**Status:** Complete. Findings recorded (ESO-01/02/03) against real Paranal production credentials; Recommendation (ESO-04: Bypass) and Future-sync sketch (ESO-05) completed in Plan 02.

This phase is investigation-only. No `sync_eso_observation_calendar` command, no
`FACILITIES['ESO']` settings change, and no other application code is shipped
from this plan — the sole committed deliverable is this findings record,
built from a throwaway, git-excluded probe script (`eso_p2_probe.py`, D-09)
run by the operator against real ESO Phase 2 production credentials
(D-01/D-03), never against a demo/sandbox environment.

## Findings

### ESO-01 — Credential obtainability & usability

Valid ESO Phase 2 production credentials for Paranal (VLT) ARE obtainable and
usable. The operator (Tim) holds a real production account and confirmed the
connection succeeded when constructed via
`tom_eso.eso_api.ESOAPI(environment='production', username, password)`
(D-05), with credentials supplied only via environment variables per D-03
(never hard-coded, never logged, never committed). Connection success is
evidenced directly by the real data returned in ESO-02 below — a live
`getOB()` call against a real OB ID and a live `getNightExecutions()` call
both returned real, well-formed response shapes rather than an authentication
error.

**La Silla stretch result (D-06):** the operator additionally flipped
`ESO_P2_ENVIRONMENT=production_lasilla` and re-ran the same probe. This
connection attempt FAILED:

```
ESOAPI.__init__: Error creating API connections: (500, 'POST', 'production_lasilla', 'environment not supported')
CONNECTION FAILED: P1Error -- unexpected error constructing ESOAPI.
```

credential-adjacent fields redacted per D-04 (none were present in this
diagnostic; it contains only the environment string and error metadata).

Per D-06 this is a valid, documented finding, not a phase blocker: Paranal
credentials work end-to-end. Note the exception surfaced as `P1Error`, not
`p2api.p2api.P2Error` — this is the key diagnostic clue, and root-causing it
changes the finding materially from "La Silla API access doesn't work" to
"`tom_eso`'s `ESOAPI` wrapper is broken for La Silla, but the underlying P2
API itself is very likely fine":

- The operator separately confirmed they can log into the La Silla P2 web
  portal (`https://www.eso.org/p2ls/home`) with the same credentials —
  meaning the account genuinely has La Silla P2 access.
- Reading the installed libraries directly: `tom_eso.eso_api.ESOAPI.__init__`
  unconditionally constructs **two** connections — `self.api1 =
  p1api.ApiConnection(environment, ...)` (Phase 1, proposal submission —
  irrelevant to OB status/execution data) *before* `self.api2 =
  p2api.ApiConnection(environment, ...)` (Phase 2 — what ESO-02 actually
  needs).
- `p1api.p1api.API_URL` only defines `'production'` and `'demo'` — there is
  no La Silla entry at all — so `p1api.ApiConnection('production_lasilla',
  ...)` unconditionally raises `P1Error(500, 'POST', 'production_lasilla',
  'environment not supported')`, and it does this before `self.api2` (the
  connection we actually need) is ever constructed.
- `p2api.p2api.API_URL`, by contrast, **does** define
  `'production_lasilla': 'https://www.eso.org/copls/api/v1'` — so the real
  Phase 2 API for La Silla is very likely reachable; the failure is entirely
  an artifact of `ESOAPI`'s wrapper needlessly demanding a Phase 1 connection
  that doesn't exist for that site.

**Revised finding:** La Silla P2 (Phase 2) access is likely viable for this
account — not confirmed blocked. The practical workaround is to bypass
`tom_eso.eso_api.ESOAPI` entirely and construct `p2api.ApiConnection(
'production_lasilla', username, password)` directly (the probe script already
calls Phase-2-only methods via `eso.api2.*` for the OB-status/execution
calls, so this bypass is a small, targeted change, not a rewrite).

**Direct-`p2api` bypass test (follow-up, run live during this spike):** the
operator ran the bypass directly in a shell session:

```
p2api.ApiConnection('production_lasilla', username, password).getRuns()
```

credential-adjacent fields redacted per D-04 (none were present in this
response). This call succeeded — no `P1Error`, no `P2Error` — and returned:

```
([{'runId': 116232900, 'progId': '116.28N5.001', 'title': 'Characterising newly discovered distant comets to prepare for the ESA Comet Interceptor mission', 'period': 116, 'scheduledPeriod': 116, 'mode': 'SM', 'instrument': 'FORS2', 'telescope': 'UT1', 'ipVersion': 116.07, 'isToO': False, 'owned': False, 'delegated': True, 'itemCount': 0, 'containerId': 1000562023, 'validFrom': '2025-10-01T16:00:00Z', 'validTo': '2026-05-01T16:00:00Z', 'pi': {...}, 'observingConstraints': {'fli': 'n', 'seeing': 0.8, 'skyTransparency': 'Clear'}}], None)
```

This confirms the wrapper diagnosis directly: bypassing `ESOAPI`/`p1api` and
calling `p2api.ApiConnection('production_lasilla', ...)` directly connects
without error. **However, the single run returned is the same run
(`runId=116232900`) already captured under Paranal `production` above** —
same `progId`, and `telescope='UT1'` is a Paranal VLT unit telescope, not a
La Silla/NTT instrument. This is a genuinely ambiguous result, not a clean
confirmation of distinct La Silla content:

- It confirms the *connection mechanism* works — `production_lasilla` is a
  valid, reachable `p2api` environment for this account once `ESOAPI`'s
  `p1api` requirement is bypassed.
- It does NOT confirm the account has any *La-Silla-specific* runs/OBs
  visible through this environment — the one query performed returned a
  Paranal-instrument run, which is either evidence that `getRuns()` isn't
  scoped strictly by the queried environment for this account, or (more
  likely) that this operator's account simply has no dedicated La Silla
  runs to return, and the API surfaced the one run it could find.

**Net La Silla finding:** connectivity is confirmed working via the
`p2api` bypass (not merely "likely viable" as first suspected); genuine
La-Silla-sourced OB/execution data remains uncaptured in this spike. A future
phase attempting La Silla sync should not assume this test alone proves
La Silla OB data is reachable — it proves the connection path is open, using
the same bypass strategy already recommended for Paranal.

### ESO-02 — Real OB status/execution data shape

Live, redacted verbatim responses were captured against Paranal production.

`getOB(4725578)` response:
```
{'obId': 4725578, 'itemType': 'OB', 'obStatus': 'P', 'name': 'TEST_CAL_LTT3218', 'exposureTime': 0, 'executionTime': 0, 'runId': 116232900, 'instrument': 'FORS2', 'ipVersion': 116.07, 'migrate': False, 'grade': '?', 'userPriority': 1, 'parentContainerId': 4725576, 'obsDescription': {'name': 'No name', 'userComments': ''}, 'target': {'name': 'CD-325613', 'ra': '08:41:32.429', 'dec': '-32:56:32.916', 'equinox': 'J2000', 'epoch': 2000.0, 'properMotionRa': 0.0, 'properMotionDec': 0.0, 'differentialRa': 0.0, 'differentialDec': 0.0}, 'constraints': {'name': 'No Name', 'airmass': 2.8, 'skyTransparency': 'Clear', 'fli': 1.0, 'seeing': 0.8, 'moonDistance': 30, 'twilight': 0, 'waterVapour': 30.0}}
```
credential-adjacent fields redacted per D-04 (none were present; this is the
full real response verbatim — the target name, RA/Dec, run ID, and OB ID are
not credential-adjacent per D-04's scope of usernames/passwords/sensitive
program IDs).

`obStatus = 'P'` (Partially Defined — just created, fully editable, never
executed; this particular OB is a draft calibration test OB, consistent with
its name `TEST_CAL_LTT3218`).

`getOBExecutions(4725578, '2026-07-01')` response: `[]` (empty — no
executions recorded, consistent with an OB that has never run).

`getNightExecutions('FORS2', '2026-07-01')` response (real per-night
execution history, captured against a *different* OB on the same
instrument/night):
```
[{'from': '2026-07-01T16:01:41Z', 'to': '2026-07-01T16:31:25Z', 'obId': 200115617, 'obStatus': 'M', 'grade': 'X'}]
```
credential-adjacent fields redacted per D-04 (none were present).

This is real per-night execution data: a time window (`from`/`to`), an
`obId`, an `obStatus` of `M` (Must Repeat — executed outside constraints,
will be requeued), and a `grade` of `X`. Together, the two response shapes
confirm both ends of the OB lifecycle are visible through the read-only P2
API: a never-executed OB's status/execution response (`P` / empty executions
list) and an executed-but-failed OB's per-night execution record (`M`/`X`
with a concrete time window). This is sufficient real-data grounding for
Plan 02 to design against, rather than the guessed 12-code vocabulary from
documentation alone.

### ESO-03 — Headless credential-sourcing path

A headless Django management command COULD source ESO P2 credentials the
same way this probe did. The live connection above succeeded via
`ESOAPI(environment, username, password)` constructed directly from
environment-variable-supplied credentials in `eso_p2_probe.py` — there was no
`ESOProfile` model instance involved, no active Django session, and no
`tom_common.session_utils` decryption anywhere in this path.

This directly demonstrates the viable path: a future headless
`sync_eso_observation_calendar` (or Bridge-record-creation) command could
authenticate the identical way — e.g. backed by a `FACILITIES['ESO']`
settings entry populated from environment variables or `local_settings.py`,
mirroring the LCO/SOAR/GEM facility-config pattern already used elsewhere in
this codebase — without depending on the per-user, session-bound,
Fernet-encrypted `ESOProfile` + Django-session-decryption path. That
alternate path is confirmed non-viable for headless use per existing
research (PITFALLS.md Pitfall 3): `ESOProfile`'s encrypted-field decryption
silently returns `None` outside an active user session, with no exception
raised, which would produce a silent, hard-to-diagnose authentication
failure in any headless command that relied on it.

**Conclusion: viable, with a concrete named path** — add a plaintext
`FACILITIES['ESO']` entry to `src/fomo/settings.py` (credentials sourced from
environment variables, consistent with D-03's "never committed, never
logged" discipline) and construct `ESOAPI(...)` directly from it, exactly as
this probe did. Relying on `ESOProfile` + session decryption for a background
job is confirmed non-viable.

## Recommendation (ESO-04)

**Bypass — sync straight from `p2api` to `CalendarEvent`, skipping
`ObservationRecord` for ESO.**

The evidence gathered in ESO-01/02/03 is direct, real-data support for Bypass,
and offers no support at all for Bridge:

- **ESO-01** confirms Paranal production credentials connect and work
  end-to-end via `tom_eso.eso_api.ESOAPI(environment='production', username,
  password)` — but the connection itself, and every subsequent call
  (`getOB`, `getOBExecutions`, `getNightExecutions`), was made directly
  against the P2 API. Nothing in the probe touched
  `ESOFacility.submit_observation()`, `ObservationCreateView.form_valid()`,
  or any code path that would create an `ObservationRecord(facility='ESO')`
  row. Bridge is specifically "patch/work around `tom_eso` so it populates
  real `ObservationRecord` rows" — that patch was never attempted, let alone
  evidenced, in this spike. Bypass's data path (direct P2 API calls into a
  synced representation) is exactly what ESO-01 exercised and proved works.
- **ESO-02** captured real `getOB()` and `getNightExecutions()` shapes —
  again, both are direct P2-API reads, not `ObservationRecord` fields. The
  fact that this data is reachable *without* an `ObservationRecord` in the
  loop is itself evidence that skipping `ObservationRecord` (Bypass) is not
  just viable but is the natural shape of the data that was actually
  captured. There is no captured evidence describing what a hand-created
  `ObservationRecord(facility='ESO')` row would need to contain, because
  Bridge's premise (working around `submit_observation()`'s empty-list
  return) was out of scope for this investigation-only phase (per D-08's
  read-only guardrail: `submit_observation()`/`saveOB`/any write call was
  never exercised).
- **ESO-03** confirms the headless credential-sourcing path — a future
  `sync_eso_observation_calendar` (or equivalent) command can authenticate
  the same way this probe did, via a `FACILITIES['ESO']` settings entry
  populated from environment variables, mirroring the LCO/SOAR/GEM
  facility-config pattern. This headless path is required by *both* Bridge
  and Bypass equally, so it doesn't discriminate between them — but it does
  confirm neither option is blocked on credentials, which is why the verdict
  can be a definitive Bypass rather than "Not Yet Feasible."

Bridge also carries structural cost that this spike's evidence doesn't
justify paying: it requires teaching a new code path to *create*
`ObservationRecord` rows by hand (working around
`ESOFacility.submit_observation()`'s hardcoded empty-list return), then
running the standard LCO/Gemini downstream pattern on top of those
hand-created rows — an extra responsibility neither existing sync command
has, and one this spike deliberately did not attempt per the D-08 read-only
guardrail. Bypass reaches the same real data (confirmed live in ESO-02)
through a shorter, already-evidenced path: read OBs directly via `p2api`/
`ESOAPI`, build `CalendarEvent`s straight from that data, keyed on a
synthetic identifier. Since the goal is a `CalendarEvent` the astronomer can
see on the calendar — not an `ObservationRecord` row for its own sake —
Bypass gets there with less new surface area and is the option this spike's
real-data evidence actually demonstrates end-to-end.

The La Silla revised finding (ESO-01) reinforces this further: the working
La Silla path identified is "bypass `tom_eso.eso_api.ESOAPI` entirely and
construct `p2api.ApiConnection('production_lasilla', ...)` directly" — i.e.
the same Bypass-shaped strategy already recommended for Paranal, generalizing
cleanly across both sites without needing site-specific `ObservationRecord`
plumbing. A live follow-up test confirmed this bypass connects successfully
(no error) — though the one run it returned was a Paranal-instrument run
already seen under `production`, so genuine La-Silla-sourced OB data remains
unconfirmed. Either way, the connection mechanism a future Bypass-based
command would use for La Silla is now proven reachable, not merely
theorized.

## Future-sync sketch (ESO-05)

Since the verdict is feasible (Bypass), this section sketches what "synced"
could reasonably mean for a **future** `sync_eso_observation_calendar`-style
command. This is scoped as input to a future milestone's requirements —
**nothing here is implemented in this phase**, and none of it should be read
as a v1/simplified/placeholder version of shippable work.

### Reusable landing point

`solsys_code/calendar_utils.py:insert_or_create_calendar_event()` needs no
changes. It is already facility-agnostic (`lookup`/`fields` dict contract),
and Gemini already proved the pattern generalizes to a facility with a
synthetic idempotency key and a single-state (banner-only) window, rather
than LCO's queued->placed two-state machine. A future ESO command would call
this helper exactly as-is.

### Synthetic idempotency key

Following the precedent set by `sync_gemini_observation_calendar.py`'s
`GEM:{prog}/{observation_id}` key (itself needed because Gemini's
`get_observation_url()` is also unusable as a key), a future ESO command
would key `CalendarEvent.url` on `ESO:{p2_environment}/{obId}` — e.g.
`ESO:production/4725578`. This directly matches the real identifiers
captured in ESO-02 (`obId`, `environment`), so the key can be built from data
already confirmed to exist in the shape it was captured.

### Banner-only vs. status-aware sync

ESO Service Mode has no advance per-OB schedule — Paranal Science Operations
chooses which OBs to execute in real time, unlike LCO's queue scheduler which
publishes `scheduled_start`/`scheduled_end` ahead of time. This rules out
reusing LCO's queued->placed two-state pattern (SYNC-02/03) for ESO; any
future sync is at most a single-state banner, the same shape Gemini already
uses. Two options for what that banner conveys, in increasing order of
richness:

1. **Banner-only (window banner, no status).** One `CalendarEvent` per OB,
   window derived from the OB's observing-run/period validity dates (or a
   PI-set absolute time constraint, if present — rare). Title stays clean;
   no attempt to reflect execution state. This is the safe floor: every
   piece of data it needs (`obId`, run period, target) was directly observed
   in ESO-02's `getOB()` response.
2. **Status-aware sync.** Layer the OB's `obStatus` (captured live in
   ESO-02: `'P'` for a never-executed OB, `'M'` for an executed-but-
   must-repeat OB via `getNightExecutions()`) onto the banner as a title
   prefix or badge, the same way LCO's terminal-state prefixing works today.
   This is real-data-supported (ESO-02 captured both a pre-execution and a
   post-execution shape), but requires deciding **which night(s) to poll**
   per OB — unlike LCO/Gemini, there is no single "give me the current
   status" call; `getOBExecutions(obId, night)` and
   `getNightExecutions(instrument, night)` are both queried per night, so a
   future command needs an explicit polling-window policy (e.g. "the OB's
   run period, one call per night") that this spike did not need to resolve
   for a one-off probe.

Given ESO-02 directly captured both a `'P'` (never-executed) and an `'M'`
(executed, must-repeat) real response, status-aware sync is evidenced as
*reachable*, not merely theoretical — but it is meaningfully more complex
than the banner-only floor, and the "which nights to poll" policy is
genuinely open. A future milestone should treat banner-only as the MVP and
status-aware as a should-have layered on top, consistent with the FEATURES.md
research's original P1/P2 prioritization.

### ESO P2 status vocabulary (if status-aware)

The 12-code `obStatus` vocabulary (from ESO's public Phase 2 status docs,
cross-checked against the real `'P'`/`'M'` values captured live in ESO-02):

| Code | Meaning | Terminal? |
|------|---------|-----------|
| `P` | Partially defined (just created) | No |
| `D` | Defined (passed certification, ready for review) | No |
| `-` | Rejected (needs user attention) | No |
| `R` | Review (under revision by support astronomer) | No |
| `+` | Accepted (ready to be observed) | No |
| `C` | Completed (executed successfully, will not repeat) | Yes |
| `X` | Executed (successfully completed, can repeat — e.g. visitor mode) | Yes (per-execution) |
| `M` | Must repeat (executed outside constraints, will be requeued) | No (requeues) |
| `A` | Aborted during execution (will be requeued) | No (requeues) |
| `F` | Failed (absolute time window expired; read-only, irreversible) | Yes |
| `K` | Kancelled (support-astronomer set, irreversible) | Yes |
| `T` | Terminated (run terminated, irreversible) | Yes |

This vocabulary is entirely unrelated to LCO's/Gemini's terminal-state sets —
a future command must build its own `obStatus`-keyed prefix mapping rather
than reusing (or extending) LCO's `_FAILURE_PREFIX_BY_STATUS` or Gemini's
`ready`-flag logic.

### Not included: D-11 effort-sizing

Per D-11, the Bridge effort-sizing estimate (which `tom_eso` methods would
need real implementations, and whether the change reads as a small patch,
moderate fork, or larger undertaking) is only required when the verdict is
Bridge. Since the verdict above is Bypass, that estimate is not applicable
here — a future Bypass-based command's cost driver is the polling-window
policy question above, not a `tom_eso` patch.
