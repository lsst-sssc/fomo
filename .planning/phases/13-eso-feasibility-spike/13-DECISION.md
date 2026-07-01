# Phase 13: ESO Feasibility Spike - Decision

**Investigated:** 2026-07-01
**Status:** Findings recorded (ESO-01/02/03) against real Paranal production credentials; Recommendation (ESO-04) and Future-sync sketch (ESO-05) deferred to Plan 02.

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
credentials work end-to-end; `'production_lasilla'` is NOT a valid/supported
P2 environment string for this account via `p2api`/`ESOAPI` as currently
used. Note the exception surfaced as `P1Error`, not `p2api.p2api.P2Error` —
a different exception path than the Paranal success case, worth flagging for
anyone attempting La Silla connectivity in a future phase (it is not simply
"the same P2Error caught elsewhere," it's a distinct failure mode at
connection-construction time, before any read-style call is even attempted).
La Silla remains untested for real OB data as a result — its Phase 2
environment identifier (if reachable at all) is not what research guessed.

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

<!-- completed in Plan 02 -->

## Future-sync sketch (ESO-05)

<!-- completed in Plan 02 -->
