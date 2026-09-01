# Phase 31: Foundation Spikes — Run Identity & Unattended Invocation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-01
**Phase:** 31-foundation-spikes-run-identity-unattended-invocation
**Areas discussed:** Real host/deployment facts, Schema shape for non-campaign runs, Classical adapter's identity surface, What counts as "real data" for the spike

---

## Todo cross-reference (pre-discussion)

Four pending todos scored a weak (0.6) keyword match against Phase 31. Presented to the user with a note that none looked genuinely relevant (generic "campaign"/"test" keyword hits, not schema/scheduling matches).

| Todo | Selected |
|------|----------|
| Extract site/telescope mapping | |
| TTL cache for attribution banner | |
| Guard attribution dismiss action | |
| Skip sun_event computation | |
| **None — leave all for later** | ✓ |

**User's choice:** None — leave all for later.

---

## Real host / deployment facts

### Q1: Where does FOMO actually run today, if anywhere in production/unattended form?

| Option | Description | Selected |
|--------|-------------|----------|
| Not deployed yet | Dev-only so far | |
| Already on a known server | Real host today, user describes it | |
| Hosting platform / PaaS | Managed Python host | |
| **Other (free text)** | | ✓ |

**User's choice:** "Locally deployed on Linux Rocky 9 machine or a WSL2 Ubuntu installation presently, eventually in LCO's AWS Kubernetes cluster."

### Q2: Is a plain OS crontab realistically available, or does policy/access favor systemd timers?

| Option | Description | Selected |
|--------|-------------|----------|
| **Cron is fine / preferred** | Matches research recommendation | ✓ |
| systemd timers preferred | | |
| Don't know yet — spike should check | | |

**User's choice:** Cron is fine / preferred.

### Q3: Is `flock` available, and is an outbound heartbeat ping acceptable? (multiSelect)

| Option | Description | Selected |
|--------|-------------|----------|
| flock is available | | |
| Outbound heartbeat ping OK | | |
| **Neither confirmed — spike must check** | | ✓ |

**User's choice:** Neither confirmed — spike must check.

### Q4: How should credentials be supplied?

| Option | Description | Selected |
|--------|-------------|----------|
| **Environment variables (existing pattern)** | Matches FINK_CREDENTIAL_* | ✓ |
| A secrets file cron reads | | |
| You decide / spike should investigate | | |

**User's choice:** Environment variables (existing pattern).

### Q5 (follow-up): Given the eventual AWS Kubernetes target, should the spike design for the interim host, K8s-native CronJob, or both?

| Option | Description | Selected |
|--------|-------------|----------|
| Design for interim host now, note K8s migration path | | |
| Design directly for K8s CronJob | | |
| Design a mechanism that works under both | | |
| **Other (free text)** | | ✓ |

**User's choice:** "I think it's ok to keep using cron+flock *inside* the FOMO container when it gets deployed to AWS."

**Notes:** This settles SCHED-07's headline question — cron+flock, running inside the container, works identically on the interim host and once deployed to AWS, so no K8s-native `CronJob` redesign is needed. `flock` availability and heartbeat-ping acceptability still need spike verification against the real container image/host.

---

## Schema shape for non-campaign runs

### Q1: How do you lean on giving a non-campaign queue observation a persistent CampaignRun identity?

| Option | Description | Selected |
|--------|-------------|----------|
| Make campaign nullable | | |
| Sentinel/placeholder TargetList row | | |
| Wide open — let the spike investigate from scratch | | |
| **Other (free text)** | | ✓ |

**User's choice:** "Not leaning either way at this stage, maybe one 'No Campaign' TargetList or a per-proposal auto-created default TargetList?"

**Notes:** Introduces a third candidate beyond the presented options — a per-proposal auto-created placeholder `TargetList`, distinct from a single shared sentinel. All three (nullable, single sentinel, per-proposal placeholder) go into CONTEXT.md as options the spike must investigate against real data.

### Q2: How much should the spike weigh migration risk for existing rows?

| Option | Description | Selected |
|--------|-------------|----------|
| **Existing rows are all campaign-linked — low risk** | | ✓ |
| Treat migration as a first-class risk to investigate | | |

**User's choice:** Existing rows are all campaign-linked — low risk.

---

## Classical adapter's identity surface

### Q1: Do classical schedule files carry any per-line ID, or is tolerance-match the ceiling?

| Option | Description | Selected |
|--------|-------------|----------|
| No per-line ID exists — tolerance-match is the ceiling | | |
| There might be one — spike should check the real files | | |
| Yes, there's an ID I know of | | |
| **Other (free text)** | | ✓ |

**User's choice:** "there might be a proposal code available in the classical line but only for specific states of classical schedule e.g. ones that are 'planned' or 'observed'"

**Notes:** Not a flat yes/no — presence is state-dependent. Spike must inspect real classical schedule file samples and confirm which states carry a proposal code.

---

## What counts as "real data" for the spike

### Q1: Does "real dev-DB rows" mean the local src/fomo_db.sqlite3, or a different dataset?

| Option | Description | Selected |
|--------|-------------|----------|
| **Local src/fomo_db.sqlite3** | Standard dev DB per CLAUDE.md | ✓ |
| A different dataset I'll point to | | |

**User's choice:** Local src/fomo_db.sqlite3.

---

## Claude's Discretion

- Exact investigation methodology (which queries to run, how to sample classical schedule files) — left to the researcher/planner.
- `docs/design/` page filename/structure — follows `uncertain_scheduling_spike.rst` / `canonical_record_spike.rst` precedent unless the researcher finds reason to deviate.

## Deferred Ideas

None raised during discussion. Four pending todos were reviewed via cross-reference but explicitly not folded (see CONTEXT.md's "Reviewed Todos (not folded)" section) — all four are unrelated to this phase's schema/scheduling scope.
