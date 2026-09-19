# Phase 37 — External API Coverage Matrix

**Decided:** 2026-09-18 (plan time)
**Detector:** `bin/lib/api-coverage.cjs` reported `detected: false` against the ROADMAP section
and the drafted plan bodies. This matrix is written anyway, because CONTEXT.md D-07 is a real
external-API read (the LCO Observation Portal's proposal endpoint) and a decided matrix at plan
time is cheaper than discovering the gap at seal time.

## External API in scope

**LCO Observation Portal** — `GET https://observe.lco.global/api/proposals/<proposal_code>/`,
authenticated with `Authorization: Token <FACILITIES['LCO']['api_key']>` through the existing
`LCOFacility._portal_headers()`. This is the same portal, the same credential and the same
`make_request` client every other portal call in this codebase already uses; no new service, no
new secret name.

Called exclusively from `solsys_code/proposal_allocation.py` inside the unattended runner's
`proposal_allocation` step (plan 37-02, Task 3). **Never called at request time** — D-07's trust
boundary: an anonymous visitor must not be able to trigger a credentialed portal call.

## Capability surface

| Capability / field | Disposition | Reason |
|---|---|---|
| `timeallocation_set[].std_allocation` | INTEGRATE | Standard-time hours awarded — the numerator half of D-06's unused-hours figure. |
| `timeallocation_set[].std_time_used` | INTEGRATE | Standard-time hours used — the subtrahend. |
| `timeallocation_set[].rr_allocation` / `rr_time_used` | INTEGRATE (stored, not summed) | Stored as their own rows so the summation rule can change without a re-fetch (RESEARCH assumption A2 is unverified against live credentials); excluded from the estimate until a live check says otherwise. |
| `timeallocation_set[].tc_allocation` / `tc_time_used` | INTEGRATE (stored, not summed) | Same reason as the rapid-response pair. |
| `timeallocation_set[].semester` | INTEGRATE | Part of the natural key, so one proposal's two semesters do not collide in one row. |
| `timeallocation_set[].instrument_types` | INTEGRATE | Part of the natural key, for the same reason. |
| `timeallocation_set[].ipp_limit` / `ipp_time_available` | OPT-OUT | Intra-proposal priority accounting; it changes which requests get scheduled, not how many nights are unused, so it feeds nothing this phase renders. |
| Proposal membership / PI / co-investigator fields on the proposal object | OPT-OUT | Personal data with no consumer in this phase; the tallies are public, so fetching them would widen the exposure of a public page for no benefit. |
| Proposal title / abstract / tags | OPT-OUT | Not rendered anywhere in this phase; the public pages key on the proposal code alone. |
| `GET /api/proposals/` (list endpoint) | OPT-OUT | The step already knows exactly which proposal codes it needs (the active watch list plus every run's stored code); the list endpoint would return proposals FOMO has no interest in. |
| Write endpoints on the portal (submission, cancellation) | OPT-OUT | Out of scope by the roadmap's own locked constraints; this phase reads only. |

## Notes

- The response shape is `[CITED]` from the official LCO developer documentation and the LCOGT
  example-scripts repository, **not** tool-verified against live FOMO credentials
  (RESEARCH.md assumptions A1/A2). Plan 37-02's `checkpoint:decision` asks the developer to
  confirm the field names and the summation rule before the model schema is finalised.
- Failure handling is not bespoke: the fetch inherits Phase 36's per-step lock, per-step failure
  isolation, failure email and heartbeat. A portal outage shows on the public pages as a
  not-yet-known unused figure, never as a blank page.
- No response body or caught exception is ever logged or stringified — those exception types
  embed the response content, and the request carried the API key (threat T-37-04/T-37-05).
