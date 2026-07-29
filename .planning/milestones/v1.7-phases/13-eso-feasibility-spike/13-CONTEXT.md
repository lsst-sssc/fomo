# Phase 13: ESO Feasibility Spike - Context

**Gathered:** 2026-07-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase is **investigation-only**. It answers "can ESO/VLT observation sync
work at all, and if so how?" by probing the real ESO P2 API for OB status/
execution data and the headless-credential situation, then produces a written
decision doc recommending Bridge, Bypass, or Not Yet Feasible. No
`sync_eso_observation_calendar` command is implemented this milestone — the
deliverable is evidence and a recommendation, not shipped sync code.

</domain>

<decisions>
## Implementation Decisions

### Credentials & environment access
- **D-01 [informational]:** Tim has real ESO Phase 2 production credentials for **Paranal
  (VLT)** and is confident they work (last tested ~February 2026). Background
  fact underlying D-05 (Paranal first); not a standalone actionable decision.
- **D-02 [informational]:** Tim is **not sure whether `tom_eso`/`p2api` has ever been
  exercised against La Silla's separate Phase 2 environment** (research
  identified this is likely a distinct `environment` string, e.g.
  `production_lasilla`, vs. Paranal's `production`). Background fact underlying
  D-06 (La Silla as stretch goal); not a standalone actionable decision.
- **D-03:** Credentials for this spike are supplied via environment
  variable(s) or `local_settings.py` only — never committed, never logged.
  No change to `FACILITIES['ESO']` in `src/fomo/settings.py` is required or
  in scope for this phase (that's Phase 1/Bridge-implementation territory
  per research, contingent on this spike's outcome).
- **D-04:** Any real API response captured verbatim into the decision doc
  (ESO-02) must have credential-adjacent content (usernames, program IDs if
  sensitive) redacted before it's written to a committed file.

### Site scope & sequencing
- **D-05:** **Paranal (VLT) first.** Confirm the full investigation loop —
  connect → capture real OB status/execution data → assess Bridge/Bypass
  viability — against Paranal, the known-working environment.
- **D-06:** **La Silla (NTT) as a stretch goal**, attempted after Paranal
  work is solid. If the `production_lasilla`-style environment fails to
  connect or `tom_eso` chokes on it, that is itself a valid, documented
  finding for the decision doc (not a phase blocker) — e.g. "Paranal:
  feasible per below; La Silla: connection untested/failed, needs separate
  follow-up."
- **D-07:** The dev DB is missing an `Observatory` record for Cerro Paranal
  (obscode `309`) — only La Silla (`809`) exists today. Creating a Paranal
  `Observatory` record is in scope for this spike if needed to support the
  investigation (e.g., for a future sketch of site/telescope resolution),
  but is not a hard requirement of ESO-01..ESO-05 — use judgement based on
  whether the decision doc needs it as supporting evidence.

### Safety guardrail
- **D-08:** **Read-only only, no writes, ever** — against both production
  environments. Only call read-style `p2api`/`ESOAPI` methods (e.g.
  `getRuns`, `getItems`, `getOB`, `getOBExecutions`, `getNightExecutions`).
  Never call save/submit/create operations against the real P2 API, even
  against what might seem like a low-risk demo/test OB — there is no
  demo/sandbox environment in scope here since Tim's credentials are
  production.

### Investigation mechanics
- **D-09:** The investigation itself (connecting to P2, exploring
  `getRuns`/`getOB`/etc.) is exploratory — via a scratch script or Django
  shell session, not a committed management command or module. Only the
  decision doc(s) (see below) are committed deliverables from this phase;
  no paired demo notebook is required (CLAUDE.md's demo-notebook convention
  applies to `telescope_runs.py` / `load_telescope_runs.py` /
  `sync_lco_observation_calendar.py` / `sync_gemini_observation_calendar.py`
  — none of which this phase touches).

### Decision doc scope (ESO-04 / ESO-05)
- **D-10:** **Both** a full-detail version and a durable summary are
  produced (see Canonical References for exact locations).
- **D-11:** If Paranal connects successfully and the evidence supports
  Bridge (patching/forking `tom_eso` to implement a functional observation-
  status retriever) as viable, the decision doc's ESO-05 sketch must
  include a **rough effort-sizing estimate** for that Bridge work — e.g.
  which methods would need real implementations
  (`get_observation_status()`/`get_observation_url()`/`data_products()`),
  roughly how much new code/logic that implies, and whether it reads as a
  small patch, a moderate fork, or a larger undertaking. This is
  order-of-magnitude scoping only — no code is written against it in this
  phase (ESO-05 explicitly defers implementation).

### Claude's Discretion
- Exact wording/structure of the decision doc beyond what D-10/D-11 specify.
- Whether creating the Cerro Paranal `Observatory` record (D-07) is
  actually needed to produce the doc's evidence, vs. noting it as a gap for
  a future phase.
- How to redact captured API responses (D-04) while keeping them useful as
  verbatim evidence for ESO-02.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### v1.7 pre-work research (mandatory — already complete)
- `.planning/research/SUMMARY.md` — executive summary, 3-phase-structure
  rationale (this phase = "Phase 0: Spike" in that document), confidence
  assessment, gaps to address
- `.planning/research/STACK.md` — confirms `tom-eso==0.2.4`/`p2api==1.0.10`
  are already installed; no new dependencies
- `.planning/research/FEATURES.md` — table-stakes vs. deferred feature
  analysis for a future ESO sync command
- `.planning/research/ARCHITECTURE.md` — Bridge vs. Bypass architectural
  tradeoffs
- `.planning/research/PITFALLS.md` — 5 critical pitfalls (status-method
  `NotImplementedError`, missing `ObservationRecord` creation path,
  session-bound encrypted credentials, incompatible status vocabularies,
  Paranal/La Silla site-assumption gap)

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — ESO-01 through ESO-05 (this phase's scope),
  ESO-10/ESO-11 (v2, contingent on this phase's decision)
- `.planning/ROADMAP.md` §"Phase 13: ESO Feasibility Spike" — the 5 success
  criteria this phase's deliverables must satisfy

### Project design docs
- `docs/design/telescope_runs_calendar.rst` — full feasibility study and
  4-stage plan for issue #37 (this feature's parent design doc)
- `docs/design/gsd_experiment.rst` — rationale for using this feature as a
  GSD workflow trial

### Installed source (read directly, not from docs — per research's HIGH-confidence findings)
- `tom_eso/eso.py` (installed `tom-eso==0.2.4`) — confirms
  `submit_observation()`/`get_observation_status()`/`get_observation_url()`
  limitations
- `tom_eso/eso_api.py` — `ESOAPI` wrapper around `p2api.ApiConnection`
- `p2api/p2api.py` (installed `p2api==1.0.10`) — the real client this spike
  will exercise directly

### External (secondary confidence — official ESO docs)
- ESO Phase 2 Status: https://www.eso.org/sci/observing/phase2/p2intro/phase-2-status.html
  — OB status code definitions (12-code vocabulary)
- ESO Phase 2 API Documentation: https://www.eso.org/sci/observing/phase2/p2intro/Phase2API.html
  — schema reference, useful if live credentials produce unexpected shapes

### Decision doc destinations (per D-10)
- `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` — full findings
  (planning record: credential evidence, captured real API response(s),
  detailed rationale)
- `docs/design/eso_feasibility_spike.rst` — durable summary alongside
  `docs/design/telescope_runs_calendar.rst`, for future milestones to
  reference directly without digging into `.planning/`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `solsys_code/calendar_utils.py:insert_or_create_calendar_event()` —
  facility-agnostic idempotent create-or-update helper; research confirms
  it needs no changes regardless of Bridge/Bypass outcome. Not touched by
  this phase, but the decision doc should note it as the eventual landing
  point for a future ESO sync command.
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` —
  the closest precedent for a facility whose `get_observation_url()` isn't
  usable: it builds a synthetic key (`GEM:{prog}/{observation_id}`). ESO's
  future Bridge/Bypass command would follow the same precedent
  (`ESO:{p2_environment}/{obId}`).

### Established Patterns
- Per-record API-failure handling (explicit timeout, single attempt, no
  retry, credential scrubbing) — established in Phase 7
  (`sync_lco_observation_calendar.py`) and reused in Phase 10
  (`sync_gemini_observation_calendar.py`). Not implemented in this phase
  (no command is built), but the decision doc's ESO-05 sketch should note
  this as the pattern a future command would follow.
- `Observatory` model (MPC-obscode-keyed, this repo's site-coordinate
  source) — if a Paranal record is created during this spike (D-07), it
  follows the same pattern as the existing La Silla/Las Campanas/Siding
  Spring records from Phase 1.

### Integration Points
- None for this phase — it produces no code that integrates with the
  running application. The decision doc is the sole deliverable.

</code_context>

<specifics>
## Specific Ideas

- Tim tested the Paranal P2 production connection himself around February
  2026 and is confident it still works; La Silla is the genuine unknown.
- The "Bridge" option's viability question isn't just binary — Tim wants
  the doc to sketch *how much work* a functional observation-status
  retriever patch/fork of `tom_eso` would actually be, not just "yes it's
  possible."

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (The Bridge effort-sizing
request is captured as D-11 above since it's a direct extension of
ESO-04/ESO-05, not a new capability.)

### Reviewed Todos (not folded)
- **"Extract site/telescope mapping and instrument extraction into own
  module"** (`.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`)
  — matched Phase 13 on keyword overlap (site/telescope/instrument/phase)
  but its `resolves_phase: 11` frontmatter and content confirm it was
  already resolved by Phase 11's `calendar_utils.py` extraction. Not
  relevant to ESO investigation; not folded. (The pending file itself
  appears stale and could be cleaned up separately — not this phase's
  concern.)

</deferred>

---

*Phase: 13-eso-feasibility-spike*
*Context gathered: 2026-07-01*
