# Phase 8: Telescope Label Verification Sidecar - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-24
**Phase:** 8-telescope-label-verification-sidecar
**Areas discussed:** Sidecar field shape, Visual cue style, Tooltip wording

---

## Pre-discussion: Todo cross-reference

Two pending todos matched Phase 8 by keyword search, presented to the user for fold/no-fold decision.

| Todo | Match score | Folded? |
|------|-------------|---------|
| Status-aware calendar event coloring (telescope/proposal-keyed) | 0.9 | No |
| Extract site/telescope mapping and instrument extraction into own module | 0.6 | No |

**User's choice:** Fold neither (recommended).
**Notes:** First todo is Phase 9's scope, not Phase 8's; second is an unrelated deferred refactor.

---

## Sidecar field shape

| Option | Description | Selected |
|--------|-------------|----------|
| Boolean only | `is_verified: BooleanField(default=True)` — matches research recommendation, simplest migration | |
| Add a reason/detail field | `is_verified` + `reason` (e.g. 'api_failed', 'unmapped_code') — richer tooltip, more migration surface | |
| You decide | Let the planner/researcher pick based on existing `telescope_api_failed` contract | ✓ |

**User's choice:** You decide.
**Notes:** Resolved in favor of boolean-only — consistent with the existing `telescope_api_failed` signal in `sync_lco_observation_calendar.py`, which Phase 07's Key Decisions deliberately keeps as one shared signal rather than splitting failure causes ("both are the same user-visible degrade signal; splitting them into two differently-labeled failure classes adds complexity without operator value"). Recorded as D-01 in CONTEXT.md.

---

## Visual cue style

| Option | Description | Selected |
|--------|-------------|----------|
| Dashed/dotted border | Research's lead recommendation (Outlook tentative-booking precedent); risk of collision with Phase 9's undecided status-border treatment | ✓ |
| Small badge/icon | Separate channel, no collision risk, but competes for cramped event-block width | |
| You decide | Let the planner pick the lowest-risk option | |

**User's choice:** Dashed/dotted border.
**Notes:** Recorded as D-02 in CONTEXT.md.

### Follow-up: Border-property scope vs. Phase 9

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, reserve dash-style for verification | Constrain Phase 9 to use a different border property (color/thickness) for status, avoiding a CSS-property collision | ✓ |
| No, leave it open | Don't pre-constrain Phase 9; let its sketch session decide everything fresh | |

**User's choice:** Yes, reserve dash-style for verification.
**Notes:** Recorded as D-03 in CONTEXT.md — flagged as an explicit constraint for Phase 9's future CONTEXT.md/planning.

---

## Tooltip wording

| Option | Description | Selected |
|--------|-------------|----------|
| Generic estimate framing | Plain-language sentence explaining why (API verification failed) and what (coarse fallback) | ✓ |
| Short/terse | Minimal one-liner, no explanation | |
| You decide | Let the planner draft exact copy within plain-language constraints | |

**User's choice:** Generic estimate framing.
**Notes:** Working text drafted: "Telescope label is an estimate — could not be verified against the LCO API; showing a coarse fallback label (1m0/0m4/2m0/4m0)." Recorded as D-04 in CONTEXT.md; exact final copy left to Claude's discretion at plan/execute time.

---

## Claude's Discretion

- Exact final tooltip copy (within the plain-language, why+what framing locked by D-04).
- Whether the dashed-border visual cue is implemented via a template-tag-driven CSS class or an inline conditional `style=` attribute.
- Sidecar field shape was explicitly deferred to Claude/planner judgment (resolved toward boolean-only, see above).

## Deferred Ideas

- N+1 batching template tag for the sidecar's reverse-accessor read — not re-opened in this discussion; already deferred to v2 as DISPLAY-09 in REQUIREMENTS.md, confirmed accept-as-is for this phase.
- Per-failure-reason tooltip detail — considered and explicitly not pursued (see Sidecar field shape above).
- Status-aware calendar event coloring todo — reviewed, not folded; belongs to Phase 9.
- Site/telescope mapping extraction todo — reviewed, not folded; unrelated deferred refactor.
