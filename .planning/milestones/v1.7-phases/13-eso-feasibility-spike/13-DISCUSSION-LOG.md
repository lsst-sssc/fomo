# Phase 13: ESO Feasibility Spike - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-01
**Phase:** 13-eso-feasibility-spike
**Areas discussed:** Credentials, Site scope, Safety guardrail, Decision doc location, Sequencing, Bridge effort-sizing

---

## Credentials

| Option | Description | Selected |
|--------|-------------|----------|
| Production credentials | You have a real ESO PI login (production P2 environment) usable for read-only investigation against your own program(s). | ✓ |
| Public demo/sandbox only | Use ESO's public P2 demo environment (well-known demo/demo-style sandbox credentials, e.g. p2demo) — no real program data, but a real API shape. | |
| Nothing obtainable — use documented shapes | No live credentials at all. The spike falls back to ESO's published Phase 2 API docs/examples. | |
| Not sure — investigate as part of the spike | Let the spike's first task be figuring out what's actually available. | |

**User's choice:** Production credentials.
**Notes:** Tim is confident these work — last tested against Paranal around February 2026.

---

## Site scope

| Option | Description | Selected |
|--------|-------------|----------|
| La Silla (NTT) only | Already has an Observatory record; matches classical-schedule precedent. | |
| Paranal (VLT) only | The higher-value target long-term, but needs a new Observatory record and its own credential environment. | |
| Both, if credentials allow | Broadest investigation, at the cost of needing two credential sets. | |
| You decide based on what's obtainable | Let credential availability drive which site(s) actually get probed. | |

**User's choice:** Free text — "IIRC there are different phase2 environments for La Silla and VLT. Not sure tom_eso has been tested against the La Silla version."
**Notes:** This surfaced a real unknown (untested `production_lasilla`-style environment) rather than picking one of the offered options directly. Resolved via the follow-up "Sequencing" question below.

---

## Safety guardrail

| Option | Description | Selected |
|--------|-------------|----------|
| Read-only only — no writes, ever | Only call read-style methods. Never call save/submit/create against P2, even on demo. | ✓ |
| Reads plus safe writes on demo only | Read-only against production; demo/sandbox may be used more freely. | |
| You decide — use judgement per environment | No hard rule; use reasonable caution. | |

**User's choice:** Read-only only — no writes, ever.
**Notes:** No demo/sandbox environment is even in play here since Tim's credentials are production — this is the correct strict default.

---

## Decision doc location

| Option | Description | Selected |
|--------|-------------|----------|
| Inside the phase directory | e.g. `.planning/phases/13-eso-feasibility-spike/13-DECISION.md` — colocated with planning artifacts. | |
| Alongside the project design doc | e.g. `docs/design/eso_feasibility_spike.rst` — next to `docs/design/telescope_runs_calendar.rst`. | |
| Both | Full findings in the phase directory; a durable summary also under `docs/design/`. | ✓ |

**User's choice:** Both.
**Notes:** Locations locked in CONTEXT.md's Canonical References section.

---

## Sequencing (follow-up to Site scope)

| Option | Description | Selected |
|--------|-------------|----------|
| Paranal first, La Silla as stretch | Confirm the full investigation against Paranal first; attempt La Silla afterward, documenting failure as a finding if it doesn't connect. | ✓ |
| Try both up front, see what connects | Attempt connection to both environments as the very first task. | |
| Paranal only — don't attempt La Silla | Scope the whole spike to VLT/Paranal. | |

**User's choice:** Paranal first, La Silla as stretch.
**Notes:** Avoids blocking the whole spike on an untested environment while still giving La Silla a real attempt.

---

## Bridge effort-sizing (raised during "anything else?" check)

**User's question:** "Assuming we can connect to the production phase2 (I'm pretty sure I tested it back in February) is it worth speccing how much work it would be to make a functional observation status retriever as a new branch/version of tom_eso?"

**Resolution:** Captured as D-11 in CONTEXT.md — if Bridge is found viable, ESO-05's sketch must include a rough (order-of-magnitude) effort estimate for patching/forking `tom_eso` to implement a real observation-status retriever. This directly extends ESO-04/ESO-05 rather than introducing new scope, so it was folded into the decisions rather than deferred.

---

## Claude's Discretion

- Exact wording/structure of the decision doc beyond the D-10/D-11 requirements.
- Whether creating the Cerro Paranal `Observatory` record is actually needed for the doc's evidence vs. noted as a future-phase gap.
- How to redact captured API responses while keeping them useful as verbatim ESO-02 evidence.

## Deferred Ideas

None — discussion stayed within phase scope.
