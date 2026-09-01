# Phase 31: Foundation Spikes — Run Identity & Unattended Invocation - Research

**Researched:** 2026-09-01
**Domain:** Django schema-identity design spike + OS-level unattended-scheduling spike (no code ships)
**Confidence:** HIGH for what real evidence could settle this session; explicitly flagged where the plan/execute stage must still gather evidence a research pass cannot (real classical schedule files, real AWS deployment)

## Summary

This phase is an investigation, not a build: nothing in `solsys_code/` changes, no migration
is applied, and the only committed artifacts are `31-DECISION.md` (in
`.planning/phases/31-.../`) and a `docs/design/` page, following the Phase 18 and Phase 26
precedent exactly. The two tracks (SCHEMA-01/02/03 and SCHED-07) are independent and can be
planned as parallel investigation tasks inside the same phase.

For the **schema/identity track**, this research session queried the real dev DB
(`src/fomo_db.sqlite3`) directly and confirms the low-migration-risk premise D-06 already
assumed: **all 49 existing `CampaignRun` rows have a non-null `campaign` FK** — nothing to
backfill. It also confirms `CampaignRun.campaign` is `NOT NULL` at the schema level today
(`"campaign_id" integer NOT NULL REFERENCES "tom_targets_targetlist"`), and re-confirms the
exact field tuples of both existing partial `UniqueConstraint`s the identity scheme must not
collide with. None of the three candidate schema shapes (nullable FK / single sentinel
`TargetList` / per-proposal auto-created default `TargetList`) has any precedent anywhere else
in this codebase — there is no existing "default row" or "sentinel campaign" pattern to copy;
whichever shape wins, the plan is designing it from first principles, not adapting an existing
local idiom. The classical adapter's write-time identity surface (SCHEMA-03) genuinely cannot
be settled by research alone: this session confirmed the only classical schedule "file" content
that exists anywhere in this repo is three example lines inside
`docs/design/telescope_runs_calendar.rst`, none of which carry a proposal code, and the current
parser (`ParsedRun`/`parse_run_line`) has no proposal-code field at all — the spike's classical
track must obtain and inspect **real** classical schedule files from the operator, something no
research pass can substitute for.

For the **scheduling track**, this research session ran real probes directly against what
appears to be the actual target interim host (the shell environment for this session reports a
Rocky 9.8 kernel, matching D-01's "local Rocky 9 Linux machine" description) and found
concrete, load-bearing evidence: `flock` (util-linux 2.37.4) is installed and on `PATH`; the
operator's own real `crontab -l` already contains three FOMO-related Django management-command
cron entries running **today**, invoked as
`<venv>/bin/python <checkout>/manage.py <command> [args]`, with **no `flock` guard on any of
them**; and an outbound HTTPS request to `hc-ping.com` (healthchecks.io's ping domain)
succeeded (redirect response, not a network failure) — network egress to a heartbeat service is
not blocked from this host. These are real, falsifiable observations from this session, not
assumptions, but the plan must still (a) confirm this session's shell is in fact the same host
D-01 refers to (not a superficially similar sandbox) and (b) verify `flock` presence and
healthchecks.io reachability from **inside the actual FOMO container image**, and from the
eventual AWS target, separately — this session's shell is not that container.

**Primary recommendation:** Scope Phase 31's plan as two independent investigation tracks that
both write into one shared `31-DECISION.md` (schema track first, since it is more constrained
and has more real data already in hand; scheduling track can run in parallel), each producing
dated, evidence-tagged findings exactly in the style of `26-DECISION.md`/`18-DECISION.md`, then
one `docs/design/` page (single file, two sections) carrying both verdicts forward. Do not let
either track attempt to "recommend from documentation" where D-05/D-07/D-03 explicitly require
real evidence — the roadmap's own success criteria make executable evidence the acceptance bar,
not argument.

## User Constraints

<user_constraints>
### Locked Decisions (verbatim from 31-CONTEXT.md)

**Real host / deployment facts (for the SCHED-07 track)**

None of this is recorded anywhere in CLAUDE.md or `.planning/codebase/*` — confirmed by search; this is exactly the gap STATE.md flagged as a blocker only the operator could close.

- **D-01:** FOMO runs today on a local Rocky 9 Linux machine or a WSL2 Ubuntu install. The eventual production target is **LCO's AWS Kubernetes cluster**.
- **D-02:** The scheduling mechanism is **cron + `flock`, running inside the FOMO container** — the same mechanism on the interim host (Rocky 9 / WSL2) and once deployed to AWS (inside the K8s pod's container), rather than a K8s-native `CronJob`. This settles SCHED-07's headline question: no task-queue dependency, and no K8s-native scheduling redesign — cron+flock travels with the container. — **Reversibility:** costly — **rationale:** Phase 34 will build the actual scheduler entry point against this mechanism; switching to K8s-native `CronJob` afterward would mean re-deriving overlap prevention (`concurrencyPolicy` vs. `flock`) and credential wiring (K8s Secrets vs. env vars) from scratch.
- **D-03:** Whether `flock` is actually present in the container image, and whether an outbound heartbeat ping (e.g. healthchecks.io) is acceptable from the eventual AWS deployment, are **not confirmed** — the spike must verify both against the real container/host rather than assume either.
- **D-04:** Credentials are supplied via **environment variables**, extending the existing `FINK_CREDENTIAL_*` pattern already used elsewhere in this codebase (per `.planning/codebase/CONCERNS.md:21`) — no new secrets-file mechanism.

**Schema shape for non-campaign runs (SCHEMA-01/02)**

- **D-05:** No lean between the candidate approaches — the user is explicitly open to any of: making `campaign` nullable, a single sentinel/placeholder `TargetList` (e.g. "No Campaign"), or a **per-proposal auto-created default `TargetList`** (one placeholder per LCO/Gemini proposal ID, not a single shared bucket). The spike must investigate at least these three against real dev-DB data and recommend one with evidence, per the roadmap's Success Criterion 1. — **Reversibility:** one-way — **rationale:** whichever shape is chosen becomes the write-time identity surface every Phase 32 adapter targets; changing it after adapters ship means a re-migration across all three adapters' writes (the exact risk research's Executive Summary calls out for "guessing it wrong").
- **D-06:** Migration risk for **existing** `CampaignRun` rows is treated as **low** — every row today came through campaign submission / CSV-import / attribution paths, all of which already require a campaign. The spike should confirm this with a real count from `src/fomo_db.sqlite3` rather than assume it, but should not spend investigation budget hardening a migration path for rows that likely don't need one. Nullability/sentinel design only has to serve **new** adapter-written rows.

**Classical adapter's identity surface (SCHEMA-03)**

- **D-07:** A proposal code **may** be present in a classical schedule line, but — per the user's recollection — only for specific run states (e.g. "planned" or "observed"), not necessarily for earlier states (e.g. "requested"). The spike must inspect real classical schedule file samples to confirm which states carry it. Where absent, the existing 5-minute telescope/instrument/start_time tolerance match (Phase 26 Criterion 2, `load_telescope_runs.py:207-216`) remains the fallback — the spike should not assume tolerance-match is either the ceiling or obsolete without checking.

**What counts as "real data" for the spike**

- **D-08:** "Real dev-DB rows" (roadmap Success Criterion 1) means the local **`src/fomo_db.sqlite3`** — the standard dev database per CLAUDE.md conventions. No other dataset or snapshot is in play.

### Claude's Discretion

- Exact investigation methodology (which queries to run against `src/fomo_db.sqlite3`, how to sample classical schedule files) is left to the researcher/planner — the decisions above scope *what* to investigate, not *how*.
- The `docs/design/` page's filename and structure follow the `uncertain_scheduling_spike.rst` / `canonical_record_spike.rst` precedent unless the researcher finds a reason to deviate.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. Four pending todos scored a weak keyword match against this phase during `cross_reference_todos` but were reviewed and **not** folded (none are actually about run-identity schema or unattended invocation): `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`, `2026-09-01-add-ttl-cache-to-attribution-banner-count.md`, `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md`, `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCHEMA-01 | A phase-time investigation spike settles whether `CampaignRun.campaign` becomes nullable (and how) so a routine, non-campaign LCO/Gemini queue observation can still get a persistent `CampaignRun` identity | Real dev-DB confirms `campaign_id` is `NOT NULL` today, 0/49 rows null — see Architecture Patterns §"Three candidate schema shapes" and Common Pitfalls §1 for trade-offs the plan must weigh with real evidence |
| SCHEMA-02 | The spike settles a `source_identifier`-style field (or equivalent) plus its `UniqueConstraint` so each adapter's write-time identity key maps cleanly onto `CampaignRun`, extending Phase 26's read-time identity-key findings to the write path | See Code Context §"Existing constraints the identity scheme must not collide with" (exact field tuples, quoted verbatim) and §"Per-adapter read-time identity keys (Phase 26 baseline)" |
| SCHEMA-03 | The spike confirms whether the classical adapter's tolerance-windowed match is a sufficient write-time identity surface, or needs a facility-specific key | See Open Questions §1 — no real classical schedule file exists in this repo to inspect; this is an execution-time evidence gap, not a research gap |
| SCHED-07 | A phase-time investigation spike settles the scheduling mechanism (cron+flock vs. a task queue) against the real target deployment's constraints, including credential handling and overlap prevention | See Environment Availability and Common Pitfalls §4-6 — this session directly verified flock presence, existing crontab precedent, and outbound network reachability from the interim host |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CampaignRun identity/nullability scheme | Database / Storage (schema + constraints) | API / Backend (write-path adapters that target the scheme) | The identity surface is a schema decision (FK nullability, new field, `UniqueConstraint`); every adapter in Phase 32 is a consumer of whatever the schema decides, not a co-owner of the decision |
| Classical adapter write-time identity | API / Backend (management command) | Database / Storage (tolerance-match query against existing rows) | `load_telescope_runs.py` already owns this decision today via its lookup dict; the spike is deciding whether that ownership boundary needs a new stored key or stays as-is |
| Unattended invocation mechanism (cron+flock) | OS / Host process tier (outside the Django app boundary entirely) | API / Backend (the management commands cron invokes) | Cron and flock are host/container-level primitives that wrap Django management commands; Django itself has no scheduling tier — this capability lives below the application, in the container's init/process layer |
| Credential handling for unattended runs | OS / Host process tier (env vars injected at container/host level) | API / Backend (`os.getenv()` reads inside settings.py / management commands) | Extends the existing `FINK_CREDENTIAL_*` pattern — env vars are supplied externally (host or K8s Secret-as-env-var) and read inside the Django process; no new tier is introduced |
| Missed-invocation visibility (heartbeat) | OS / Host process tier (outbound ping from inside the container) | API / Backend (`_notify_staff()`-style in-command failure path) | Two independent layers per SCHED-09: one lives inside the Django command (in-process exception handling), the other lives outside it entirely (a dead-man's-switch that fires only when the *scheduler itself* never invoked the command) |

## Standard Stack

### Core

No new Python package is required for either track. This is a deliberate, evidence-backed
choice already locked by D-02 and by the project's own v2.3 Out-of-Scope table ("A task-queue
scheduler (Celery/huey/APScheduler) ... no broker/worker infra buys anything for a
single-server, few-jobs-an-hour deployment").

| Tool | Version (verified this session) | Purpose | Why Standard |
|------|-----|---------|--------------|
| `flock` (util-linux) | `2.37.4` `[VERIFIED: flock --version, this session's shell]` | Prevent overlapping cron invocations of the same command | OS-level advisory locking; no Python dependency; already installed on this host — see Environment Availability |
| cron | present as `crontab`, real user entries confirmed `[VERIFIED: crontab -l, this session's shell]` | Unattended recurring invocation | Zero new daemon; the operator's real crontab already runs FOMO management commands this way today (see Common Pitfalls §4 for the gap this reveals) |
| Django `send_mail()` (already in use) | Django `5.2.17` `[VERIFIED: python -c "import django; print(django.VERSION)", this session]` | In-command failure notification | `campaign_views.py::_notify_staff()` already does this — see Code Context correction below; this is the pattern SCHED-09/Phase 34 extends, not `mail_admins()` |
| healthchecks.io ping endpoint (`hc-ping.com`) | n/a (hosted service, no client library needed — a bare HTTPS GET/POST) | Dead-man's-switch / missed-invocation visibility | `curl`-reachable from this host this session (HTTP 301 response, not a connection failure) `[VERIFIED: curl to https://hc-ping.com/, this session's shell]`; free tier, no code dependency — a `requests.get(url, timeout=...)` call from the management command's success path is the entire integration |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cron + flock | Celery/huey/APScheduler task queue | Rejected in `REQUIREMENTS.md`'s Out-of-Scope table: no broker/worker infra buys anything for this deployment's actual job count; only revisit if the spike itself finds a real blocker to cron+flock |
| cron + flock | K8s-native `CronJob` + `concurrencyPolicy` | D-02 explicitly rejects this to keep the mechanism identical across the interim host and the eventual AWS deployment; would require re-deriving overlap prevention and credential wiring from scratch after Phase 34 already builds against cron+flock |
| Nullable `CampaignRun.campaign` FK | Sentinel `TargetList` row / per-proposal auto-created `TargetList` | This is exactly D-05's three-way open question — see Architecture Patterns below; no lean, evidence-driven choice required |

**Installation:** None — `flock` ships with util-linux (already present on Rocky 9 and virtually every Linux distribution including typical container base images); healthchecks.io requires no package, only an outbound HTTP call the command already has the tooling for (`requests`, already a transitive dependency of this Django stack).

**Version verification:** `flock --version` -> `flock from util-linux 2.37.4` (verified this session, interim host). No package-manager version to check for cron itself (it is a system service, not a pip/npm package). Django version for migration-pattern purposes: `5.2.17` (verified this session) — note this is materially newer than CLAUDE.md's "Django 2.1+ (via TOM Toolkit)" tech-stack line; all nullable-FK/`get_or_create`/`on_delete=SET()` patterns below are current for 5.2.

## Package Legitimacy Audit

Not applicable — this phase installs no external packages (no `pip install`, no `npm install`).
`flock` is a pre-existing OS binary; healthchecks.io is a hosted HTTP endpoint requiring no
client library. If a future phase (34) decides a healthchecks-specific Python client is
warranted, that phase's own research must run the Package Legitimacy Gate at that time — a bare
`requests.get()` call needs no such audit.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │  Host / Container process tier (OS-level)    │
                    │                                               │
   cron daemon ───► │  flock <lockfile> \                          │
   (interim host    │    python manage.py sync_lco_observation_... │
    or AWS pod)     │                                               │
                    │  env vars injected here (FINK_CREDENTIAL_*   │
                    │  pattern extended to LCO/Gemini creds)       │
                    └───────────────┬───────────────┬─────────────┘
                                    │               │
                         success/failure      on success only
                                    │               │
                                    ▼               ▼
                    ┌───────────────────────┐  ┌─────────────────────┐
                    │ In-command failure     │  │ Outbound heartbeat  │
                    │ notification           │  │ ping (hc-ping.com)  │
                    │ (extends _notify_staff │  │ — dead-man's-switch │
                    │  send_mail() idiom)    │  │ catches scheduler    │
                    └───────────────────────┘  │ itself never firing  │
                                                 └─────────────────────┘

  Data path this phase investigates but does not write:

   Adapter write path ──► identity lookup (source_identifier / equivalent)
                             │
                             ├─ collides with unique_campaign_run_resolved_window? (checked)
                             ├─ collides with unique_campaign_run_tbd_natural_key? (checked)
                             └─► CampaignRun row (campaign nullable? sentinel? per-proposal default?)
```

### Recommended `31-DECISION.md` / `docs/design/` Structure (precedent-derived)

Both `26-DECISION.md` and `18-DECISION.md` follow the same shape; Phase 31 should follow it
too, adapted for two tracks in one phase:

```
31-DECISION.md
├── Header: Investigated date, Status (built up across N plans, same as 26-DECISION.md:1-13)
├── Investigation-only framing paragraph (no schema migration, no code ships)
├── ## Findings
│   ├── ### Schema/Identity track (SCHEMA-01/02/03)
│   │   ├── Real dev-DB snapshot (counts, dated, fingerprinted — mirror 26-DECISION.md's
│   │   │     `stat -c '%s %Y'` fingerprint-before/after discipline for any real-DB touch)
│   │   ├── Three-way candidate comparison: nullable FK vs. sentinel TargetList vs.
│   │   │     per-proposal auto-created TargetList (evidence for each, confidence-tagged)
│   │   ├── source_identifier field + UniqueConstraint proposal, checked against both
│   │   │     existing partial constraints (quote the constraint definitions verbatim)
│   │   └── Classical adapter identity surface: real schedule-file sample findings (or an
│   │         explicit statement that no real file was available and what was substituted)
│   └── ### Scheduling track (SCHED-07)
│       ├── Real host verification: flock presence, cron precedent, outbound reachability
│       ├── Credential-handling verification (env var injection path confirmed end to end)
│       └── Missed-invocation visibility mechanism confirmed reachable
├── ## Recommendation (one locked decision per SCHEMA-01/02/03/SCHED-07 criterion)
└── ## Durable summary → docs/design/<name>.rst

docs/design/<run-identity-and-unattended-invocation>.rst
├── Background (why this spike, what four questions it answers)
├── Key finding (one paragraph per track)
├── Decisions (list-tables, mirroring canonical_record_spike.rst's per-topic table shape)
└── Future scope (pointer back to 31-DECISION.md for full evidence)
```

Suggested single-page filename following the `_spike.rst` naming convention already
established by both precedents: `docs/design/run_identity_and_unattended_invocation_spike.rst`
(both tracks in one file, two `Decisions` tables, since Success Criterion 5 asks for "a
`docs/design/` page" — singular — carrying both verdicts).

### Pattern 1: Three candidate schema shapes for a non-campaign `CampaignRun` (D-05)

**What:** Django offers three structurally different ways to let a `CampaignRun` exist without
a "real" campaign, each with a different migration and query-time shape.

**When to use:** Exactly this situation — a FK that is `NOT NULL` today, where some future
writers genuinely have no natural value for it, and where the "no value" case still needs a
stable, queryable identity.

**Option A — Make `campaign` nullable.**
```python
# Source: Django docs, ForeignKey.null — https://docs.djangoproject.com/en/5.2/ref/models/fields/#null
campaign = models.ForeignKey(
    TargetList,
    on_delete=models.PROTECT,
    null=True,        # was: null=False
    blank=True,
    related_name='campaign_runs',
)
```
Migration: a single `AlterField` (`makemigrations` autodetects this cleanly — no `RunPython`
step needed since existing rows already have real values, confirmed 0/49 null this session).
Trade-off: every existing reader of `run.campaign.name` (there are several — the `__str__`
method alone does `self.campaign.name` unconditionally, `models.py:352`) must be audited for a
new `AttributeError` risk on `None`. **This is the cheapest migration but the widest blast
radius on read paths** — the spike must grep every `.campaign.` access site, not just the write
paths, before recommending this.

**Option B — Single sentinel `TargetList` (e.g. "No Campaign").**
```python
# Source: Django docs get_or_create — https://docs.djangoproject.com/en/5.2/ref/models/querysets/#get-or-create
def get_no_campaign_sentinel() -> TargetList:
    sentinel, _ = TargetList.objects.get_or_create(name='No Campaign')
    return sentinel
```
No schema change to `CampaignRun.campaign` at all — it stays `NOT NULL`. Every non-campaign
adapter write points `campaign` at this one shared row. Trade-off: `unique_campaign_run_
resolved_window` and `unique_campaign_run_tbd_natural_key` both key on `campaign`, so *every*
non-campaign run across *every* facility shares one `campaign` value in those constraints —
two genuinely distinct non-campaign runs at the same telescope/instrument with the same window
would collide on `unique_campaign_run_resolved_window` today, a real risk this option must
explicitly rule out with evidence (or fold `source_identifier` into that constraint too).

**Option C — Per-proposal auto-created default `TargetList`.**
```python
# Source: Django docs get_or_create, parameterized by proposal id
def get_or_create_default_campaign_for_proposal(proposal_id: str, source: str) -> TargetList:
    name = f'{source} default — {proposal_id}'
    campaign, _ = TargetList.objects.get_or_create(name=name)
    return campaign
```
Splits the collision risk Option B has by proposal, at the cost of one `TargetList` row per
proposal ID ever seen (a new, unbounded-but-slow-growing table of placeholder rows) and a new
question: what happens when a proposal later *does* get a real coordinated campaign — does its
placeholder `TargetList`'s existing `CampaignRun` rows get re-pointed, or does the placeholder
persist forever alongside the real campaign? This is a genuinely open design question Option A/B
do not have, and the spike's Recommendation section must answer it explicitly if Option C wins.

**Anti-pattern to avoid:** picking a schema shape by which is easiest to implement rather than
by the real-data evidence D-05 asks for — the STATE.md rationale for D-05 is explicit that
getting this wrong means a "re-migration across all three adapters' writes" once Phase 32 ships
against whichever shape wins here.

### Pattern 2: `on_delete=SET_NULL` / `SET()` for FK cleanup on identity-owner deletion

**What:** If Option A (nullable FK) is chosen, deciding what happens to a `CampaignRun` if its
now-nullable `campaign` (or, for Option B/C, the sentinel/placeholder `TargetList`) is ever
deleted matters — `on_delete=PROTECT` (today's setting) would then block deletion of a
placeholder row that hundreds of `CampaignRun`s point at, which may or may not be desired.

```python
# Source: Django docs, ForeignKey.on_delete — https://docs.djangoproject.com/en/5.2/ref/models/fields/#django.db.models.ForeignKey.on_delete
campaign = models.ForeignKey(
    TargetList,
    on_delete=models.SET(get_no_campaign_sentinel),  # callable, evaluated at delete time
    null=True,
    blank=True,
)
```
**When to use:** Only if Option A is chosen AND the plan wants "delete a real campaign, its
orphaned runs fall back to a sentinel" behavior rather than the current `PROTECT` (block
deletion entirely). This is a secondary decision the spike should flag as a follow-on question
for whichever option wins, not something to resolve in the abstract.

### Anti-Patterns to Avoid

- **Recommending a schema shape from documentation/training-data reasoning alone.** D-05 is
  explicit: "no lean... investigate at least these three against real dev-DB data." A
  recommendation with no dated dev-DB query behind it fails the roadmap's own Success
  Criterion 1 bar ("backed by executable evidence... not by argument").
- **Treating the classical adapter's tolerance-match as either obviously sufficient or
  obviously obsolete.** D-07 explicitly forbids assuming either without checking real schedule
  files — see Open Questions §1.
- **Assuming flock/network reachability from this research session's shell environment
  automatically satisfies D-03.** This session's probes are real, useful, load-bearing
  evidence for the *interim host*, but D-03 requires verification against **the container
  image** and **the eventual AWS deployment** too — neither of which this session's shell can
  reach or confirm it matches. Treat this session's findings as one data point, not closure.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Overlap prevention for cron-invoked commands | A custom PID-file / lockfile-checking wrapper script | `flock` (already installed, already the util-linux standard tool for exactly this) | `flock`'s advisory locking correctly handles the crash-leaves-stale-lock case (`flock -n` fails fast if held; the lock releases automatically when the holding process exits, including on crash) — a hand-rolled PID file does not, without significant extra code to handle stale PIDs |
| Dead-man's-switch / missed-invocation detection | A custom periodic "did the last run happen" checker inside Django itself | healthchecks.io-style external ping service | A checker running *inside* the same scheduling mechanism that might be the thing failing cannot detect its own scheduler dying — the entire point of a dead-man's-switch is that failure to receive a ping (from *outside* the failing system) is itself the signal, per Critical Pitfall #2 in `.planning/research/SUMMARY.md` |
| A new placeholder-identity concept for non-campaign runs | A bespoke "no campaign" flag/enum bolted onto `CampaignRun` alongside the existing FK | One of the three FK-shape patterns above — all are standard Django ORM idioms (`null=True`, `get_or_create`, `on_delete=SET()`) | Django's FK nullability and sentinel-row idioms are well-trodden; a parallel boolean/enum "has no real campaign" flag would create two sources of truth for the same fact the FK's value (or lack of one) already encodes |

**Key insight:** Both tracks of this phase are deliberately choosing between existing, standard
mechanisms (Django FK patterns; OS cron+flock) rather than inventing anything new — the actual
work is evidence-gathering to pick correctly among them, not engineering a novel mechanism.

## Common Pitfalls

### Pitfall 1: Recommending Option A (nullable FK) without auditing every `.campaign.` read site

**What goes wrong:** A nullable `campaign` FK compiles and migrates cleanly, then raises
`AttributeError: 'NoneType' object has no attribute 'name'` the first time a non-campaign
`CampaignRun` reaches a template or admin page that assumes `run.campaign.name` is always
safe — e.g. `CampaignRun.__str__` (`models.py:352`, `f'#{self.pk} {self.campaign.name} | ...'`)
executes unconditionally today.

**Why it happens:** The write path (adapters) is naturally what D-05's investigation focuses
on, but every existing *read* path was written against the `NOT NULL` invariant and has never
had to handle `None`.

**How to avoid:** If Option A is the spike's chosen candidate, the spike (or its
Recommendation section, handed to Phase 32's planner) must include a grep-verified inventory of
every `.campaign.` / `run.campaign` access site in `solsys_code/` — not just the ones this
research pass happened to notice — as part of its evidence, not left implicit.

**Warning signs:** Any `.campaign.<attr>` access without a `None`-guard, `campaign__isnull` used
inconsistently across querysets, or admin list_display/`__str__` methods assuming the FK is
always populated.

### Pitfall 2: A sentinel/placeholder identity colliding with the write-time `UniqueConstraint`

**What goes wrong:** Under Option B (single sentinel `TargetList`), two genuinely distinct
non-campaign observations at the same `telescope_instrument` with the same resolved
`window_start`/`window_end` would both target `(sentinel_campaign, telescope_instrument,
window_start, window_end)` — the exact tuple `unique_campaign_run_resolved_window` enforces —
and the second write would raise `IntegrityError` where it should have created a second,
independent row.

**Why it happens:** The existing constraint was designed when `campaign` genuinely
distinguished different coordination efforts; collapsing many different "no campaign" runs onto
one shared campaign value removes that constraint's real discriminating power for exactly the
rows this phase is trying to give identity to.

**How to avoid:** The spike must either (a) prove with real/constructed evidence that this
collision cannot occur in practice for the actual telescope/instrument/window combinations LCO
and Gemini queue observations produce, or (b) fold `source_identifier` into the constraint
itself so the new field, not `campaign`, does the discriminating for these rows. This is
precisely what SCHEMA-02's "cannot collide with either existing partial constraint" success
criterion is asking the spike to prove, not assume.

**Warning signs:** Any candidate design where the new identity field is *not* part of a
`UniqueConstraint` at all (mirrors Pitfall from `26-DECISION.md` WR-05: `get_or_create()` is
only race-safe when its lookup fields are backed by a real DB constraint).

### Pitfall 3: Assuming the classical adapter's tolerance-match generalizes to a `source_identifier` scheme designed around LCO/Gemini

**What goes wrong:** `load_telescope_runs.py:207-216`'s lookup
(`{'telescope': ..., 'instrument': ..., 'start_time': ...}` with a 5-minute tolerance) has no
`url`-equivalent single string value the way LCO (`portal request url`) and Gemini
(`GEM:{prog}/{obsid}`) do. If SCHEMA-02's new field is designed as a single string column with
a straightforward equality `UniqueConstraint`, the classical adapter has no natural single value
to write into it without inventing one — and D-07 flags that a proposal code, the most obvious
candidate, is not reliably present in every schedule-line state.

**Why it happens:** LCO/Gemini both already produce a natural unique string (portal URL / API
observation ID); the classical file format was designed around free-text schedule lines with no
equivalent stable identifier baked in.

**How to avoid:** Confirm the design accommodates a facility whose "identity" may remain a
tuple-plus-tolerance match rather than a single equality key — either by making
`source_identifier` nullable/optional (classical rows leave it blank and rely on the existing
tolerance match as today) or by explicitly synthesizing a deterministic string for classical
runs (e.g. `f'CLASSICAL:{telescope}:{instrument}:{start_time.isoformat()}'`) and testing whether
that materially changes match behavior versus today's tolerance window.

**Warning signs:** A schema design that makes `source_identifier` `NOT NULL` for every row
before confirming the classical adapter genuinely has a value to put there.

### Pitfall 4: The operator's real crontab already shows the exact gap this phase must close

**What goes wrong:** This session's real `crontab -l` output (three FOMO-related entries)
reveals that **cron entries already exist for this codebase's management commands with no
`flock` guard at all** — e.g.
`0 * * * * /home/tlister/venv/fomo311_venv/bin/python /home/tlister/git/fomo_fresh/manage.py rundataquery 1`.
If a long-running invocation overruns its hourly slot, the next cron tick launches a second,
overlapping process against the same SQLite database — precisely Critical Pitfall #1 in
`.planning/research/SUMMARY.md` ("SQLite write-lock collision"), already a live risk today, not
a hypothetical one this phase is inventing.

**Why it happens:** These entries predate this milestone's overlap-prevention requirement
entirely; they were written before `flock` was ever considered.

**How to avoid:** The spike's Recommendation should note this concretely as evidence *for*
D-02's cron+flock choice (the mechanism is proven to already work with cron on this exact host;
adding `flock` is a small, low-risk addition, not a new paradigm) and should recommend the exact
flock invocation shape Phase 34 will need, e.g.:

```bash
# Source: util-linux flock(1) man page, standard cron+flock idiom
*/15 * * * * /usr/bin/flock -n /tmp/fomo-sync-lco.lock /path/to/venv/bin/python /path/to/manage.py sync_lco_observation_calendar
```

**Warning signs:** Any recommendation that treats overlap prevention as a purely theoretical
future risk rather than a condition this exact host already demonstrates is possible today.

### Pitfall 5: Verifying flock/network reachability against the wrong host

**What goes wrong:** This research session's shell reports a Rocky 9.8-family kernel and has
direct filesystem/network access, matching D-01's "local Rocky 9 Linux machine" description —
but the actual FOMO **container image** that will run in production is a distinct artifact from
this general-purpose dev shell, and may have a minimal base image that lacks `flock`,
`curl`, or outbound network access even if the host it runs on has all three.

**Why it happens:** It is easy to conflate "verified on a host that looks like the target" with
"verified inside the actual deployment artifact."

**How to avoid:** The spike's execution phase must explicitly run the same two checks
(`flock --version`, outbound ping to a heartbeat endpoint) from *inside* the actual FOMO
container image (e.g. `docker run --rm <image> flock --version`), not just from this general
dev shell, before D-03 can be marked confirmed rather than "confirmed on a similar host."

**Warning signs:** A `31-DECISION.md` that cites this research session's findings as sufficient
evidence for D-03 without a corresponding container-level check.

### Pitfall 6: Credential leakage through the very notification mechanism meant to surface failures

**What goes wrong:** SCHED-10 requires "no credential value appears in any log line or
notification generated by the unattended execution path" — but `_notify_staff()`'s pattern is
to email a message string built from request/exception context. If a future adapter's
exception message includes, e.g., an LCO API key embedded in a failed request URL, extending
`_notify_staff()` naively would leak that credential straight into a staff notification email.

**Why it happens:** Exception messages from HTTP client libraries often include the full
request (headers, sometimes query-string credentials) in their string representation for
debugging convenience.

**How to avoid:** Flag as an explicit finding for Phase 34: any adapter exception must be
sanitized (or logged only, never included verbatim in the notification body) before reaching
`_notify_staff()`'s pattern. Out of this phase's scope to fix, but worth naming in the decision
doc as a concrete constraint Phase 34 must satisfy, given D-04 already commits to env-var
credentials that could appear in a raised exception's `str()`.

**Warning signs:** Any code (present or future) that does `str(exc)` into an email body/log line
without first confirming the exception type cannot carry credential material.

## Code Examples

### Existing classical adapter identity lookup (verbatim, current line numbers)

```python
# Source: solsys_code/management/commands/load_telescope_runs.py:207-216 (read this session)
event, action = insert_or_create_calendar_event(
    {'telescope': parsed.telescope, 'instrument': parsed.instrument, 'start_time': start_time},
    {
        'end_time': end_time,
        'title': title,
        'description': description,
        'target_list': campaign,
    },
    start_time_tolerance=_START_TIME_MATCH_TOLERANCE,
)
```
`_START_TIME_MATCH_TOLERANCE = timedelta(minutes=5)` (`load_telescope_runs.py:22`) — the
existing fallback D-07 says must not be assumed obsolete without checking real files first.

### Existing LCO/Gemini read-time identity keys (verbatim, current line numbers, re-verified this session)

```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:341
event, action = insert_or_create_calendar_event({'url': url}, fields)
```
```python
# Source: solsys_code/management/commands/sync_gemini_observation_calendar.py:150,163
url = f'GEM:{prog}/{record.observation_id}'
_event, action = insert_or_create_calendar_event({'url': url}, fields)
```

### Existing partial `UniqueConstraint`s the new identity field must not collide with (verbatim, read this session)

```python
# Source: solsys_code/models.py:288-303 (CampaignRun.Meta.constraints)
models.UniqueConstraint(
    fields=('campaign', 'telescope_instrument', 'window_start', 'window_end'),
    condition=models.Q(window_start__isnull=False),
    name='unique_campaign_run_resolved_window',
),
models.UniqueConstraint(
    fields=('campaign', 'telescope_instrument', 'contact_person'),
    condition=models.Q(window_start__isnull=True),
    name='unique_campaign_run_tbd_natural_key',
),
```

### Existing `_notify_staff()` idiom — correction to CONTEXT.md's characterization

CONTEXT.md's canonical_refs section describes this as a "`mail_admins()`-based staff-notification
idiom." This is not quite what the code does — `mail_admins()` does not appear anywhere in this
codebase `[VERIFIED: grep -rn mail_admins --include=*.py ., this session, zero matches]`. The
real mechanism, which Phase 34 should extend:

```python
# Source: solsys_code/campaign_views.py:326-345 (read this session)
def _notify_staff(self, run):
    recipients = list(User.objects.filter(is_staff=True).exclude(email='').values_list('email', flat=True))
    if not recipients:
        return  # no staff with an email on file -- nothing to notify, not an error
    queue_url = self.request.build_absolute_uri(reverse('campaigns:approval_queue'))
    send_mail(
        subject='FOMO: new campaign run submission pending review',
        message=f'A new run submission is pending review: {queue_url}',
        from_email=None,
        recipient_list=recipients,
        fail_silently=True,
    )
```
It emails every `is_staff=True` user with a non-blank email on file via `django.core.mail.
send_mail()`, not Django's `mail_admins()` (which sends to the static `settings.ADMINS` list).
This distinction matters for Phase 34: a management-command context has no `self.request` to
build an absolute URL from, so the extension will need a different way to build the
approval-queue-style link (or drop the link entirely for a scheduler-context notification).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual `--proposal`/`--name-prefix` args per invocation (`backfill_lco_observation_records`-style) | Admin-editable watch-list model (`WatchedProposal`, DISCOVER-01) | Phase 34 (not this phase) | Not this phase's concern directly, but the scheduling mechanism this phase settles is what Phase 34's watch-list-driven sweep will run under |
| `CampaignRun.campaign` as an unconditional `NOT NULL` FK | One of three candidate shapes accommodating "no real campaign" | This phase | Every Phase 32 adapter write targets whichever shape wins here |

**Deprecated/outdated:** None yet — this phase is the first to touch this specific gap; nothing
existing is being deprecated, only extended.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | This research session's shell environment is genuinely the same interim host D-01 refers to (Rocky 9), not a distinct sandbox that merely shares a similar kernel version string | Summary, Common Pitfalls §5 | If wrong, the flock/crontab/network findings in this document are evidence about an unrelated machine, not the real interim host — the plan must independently re-confirm on the actual operator machine |
| A2 | `hc-ping.com` reachability implies healthchecks.io (or an equivalent dead-man's-switch service) is an acceptable choice from a policy/compliance standpoint, not just a network-reachability standpoint | Standard Stack, Common Pitfalls §5 | An outbound ping to a third-party SaaS from a scientific/research deployment may need institutional approval (data-sharing, security-review) this research cannot assess — D-03 explicitly calls this "not confirmed," and this session's network probe only resolves the *technical* half of that question |
| A3 | A bare `requests.get()`/`curl` call is a sufficient healthchecks.io integration with no client library needed | Standard Stack | If healthchecks.io's API requires more than a bare ping (e.g. signed payloads, specific headers for a paid tier), a later phase may need an actual client dependency — low risk, since the free-tier ping API is documented as exactly this simple, but not independently verified against healthchecks.io's own docs this session |
| A4 | Django's `on_delete=SET()` callable pattern (Pattern 2) is the recommended idiom for a nullable-FK-with-fallback design, rather than a plain `SET_NULL` | Architecture Patterns Pattern 2 | Low risk — this is standard, well-documented Django ORM behavior (training-knowledge level, not independently fetched from docs this session), but the *choice* of whether Option A even wants fallback-on-delete behavior at all is a genuinely open design question the spike must resolve, not merely a technical detail |

**None of these block planning** — A1/A2 are confirmation checkpoints for the plan/execute
stage (re-verify inside the container, and ask the operator about SaaS-heartbeat acceptability
as a policy question, not just a technical one); A3/A4 are low-risk technical assumptions with
an easy fallback if wrong.

## Open Questions

1. **Does a real classical schedule file ever carry a proposal code, and in which run states?**
   - What we know: The only classical-schedule-line examples anywhere in this repo are the
     three lines in `docs/design/telescope_runs_calendar.rst`'s "Classical Run Input Format"
     section (`NTT EFOSC2 allocation 9-13 July`, `Magellan IMACS 13-19 July (proposed)`,
     `Magellan Proto-Lightspeed Jul 8-12 (proposed)`), none of which carry a proposal code.
     `ParsedRun`/`parse_run_line()` (`solsys_code/telescope_runs.py`) has no proposal-code field
     at all today, and `KNOWN_STATUSES = {'allocation', 'proposed', 'confirmed', 'cancelled',
     'not confirmed'}` (`telescope_runs.py:36`) does not literally contain "planned" or
     "observed" — the words D-07's user recollection used — suggesting D-07's memory may be of
     a different status vocabulary (perhaps `CampaignRun.RunStatus`, or the source ESO/LCO
     schedule tool's own states) than the classical-file parser's own `KNOWN_STATUSES`.
   - What's unclear: Whether real classical schedule files (from ESO's Tatoo tool, or wherever
     these lines originate) ever include a proposal code at all, in which format, and under
     which of the file's own status words.
   - Recommendation: This is a real-evidence gap no research pass can close — the plan/execute
     stage must ask the operator for a real, current classical schedule file (or a representative
     historical one) to inspect, exactly as Phase 18's spike did for the real 3I/ATLAS CSV. If no
     real file is obtainable, the spike should say so explicitly (mirroring 26-DECISION.md's
     honest "0 real rows" framing for the Gemini/CAMPAIGN: cases) rather than reasoning from the
     three documentation examples as if they were a representative sample.

2. **Which of the three D-05 schema shapes has the smallest blast radius on existing read paths?**
   - What we know: `CampaignRun.__str__` (`models.py:352`) unconditionally dereferences
     `self.campaign.name`; this is one concrete site Option A (nullable FK) would need to guard.
     A full inventory of every `.campaign.` access site was not exhaustively enumerated in this
     research pass (out of scope for a research pass; this is exactly the kind of `grep`-then-
     read audit the spike's execution should perform).
   - What's unclear: The total count and location of every such site, and whether any of them
     sit in a hot path (e.g. a queryset `.select_related('campaign')` that would need touching)
     versus a cold path (e.g. an admin `__str__`).
   - Recommendation: The plan should scope a dedicated grep-and-read task for this as part of
     the schema-track investigation, producing the same kind of "confirmed against real rows /
     six integration points, not four" table `26-DECISION.md`'s SPIKE-04 produced for its own
     rename blast-radius question — that is the right evidentiary bar for this question too.

3. **Is this session's shell genuinely the D-01 interim host, or merely similar to it?**
   - What we know: Kernel string (`Linux 5.14.0-687.42.1.el9_8.x86_64`) and `systemctl --version`
     output (`252-67.el9_8.4.rocky.0.1`) both indicate a genuine Rocky Linux 9.8 machine, and the
     real `crontab -l` for the logged-in user already contains FOMO-specific cron entries
     referencing `/home/tlister/git/fomo_fresh` — a plausible sibling checkout of this same
     project, which is strong circumstantial evidence this *is* the real interim host, not a
     disposable sandbox.
   - What's unclear: Research has no way to independently confirm this from inside the sandbox
     alone; only the operator can confirm "yes, this is the machine referenced in D-01."
   - Recommendation: The plan's SCHED-07 track should ask the operator to confirm this
     explicitly as a cheap, high-value checkpoint before treating this session's flock/cron/
     network findings as closing D-03 for the interim host (they would still not close it for
     the container image or the eventual AWS target regardless of the answer).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `flock` (util-linux) | SCHED-07 overlap prevention | ✓ (this session's shell, likely-real interim host — see Open Question 3) | 2.37.4 | Not yet verified inside the actual container image — spike must check separately |
| cron / `crontab` | SCHED-07 unattended invocation | ✓ (real entries already exist for this project) | n/a (systemd 252-67.el9_8.4.rocky.0.1 host) | None needed — cron is confirmed working today for this exact codebase |
| Outbound HTTPS to `hc-ping.com` | SCHED-07/SCHED-09 heartbeat/dead-man's-switch | ✓ (HTTP 301 response to a bare `curl`, this session) | n/a | If blocked from the container/AWS target, fall back to an in-command-only failure notification (SCHED-09's first layer still functions without the second) |
| SQLite dev DB (`src/fomo_db.sqlite3`) | D-08's "real dev-DB rows" requirement | ✓ (queried directly this session via the `sqlite3` CLI) | SQLite (bundled with Python 3.11.13, this session) | None needed |
| Real classical schedule file(s) | SCHEMA-03 | ✗ — none exist anywhere in this repository | — | No fallback: the spike's classical-track finding must either obtain one from the operator or explicitly record that none was available, per Open Question 1 |
| Docker (for container-level flock/network verification, Pitfall 5) | D-03's container-image verification requirement | Not checked this session (no container build was attempted) | — | If Docker/the container image is unavailable during planning, the spike should record D-03 as verified for the interim host only, explicitly flagged as still open for the container/AWS target |

**Missing dependencies with no fallback:**
- Real classical schedule file(s) for SCHEMA-03 — must come from the operator; no substitute
  data source exists in this repository.

**Missing dependencies with fallback:**
- Container-image-level flock/network verification (Pitfall 5) — if the plan/execute stage
  cannot build or access the actual container image, document the gap explicitly in
  `31-DECISION.md` rather than silently treating the interim-host verification as sufficient for
  D-03's full scope.

## Validation Architecture

This phase ships no source-code behavior change (Success Criterion 5: "the test suite is
unchanged because no source behaviour changed"). The precedent phases (18, 26) added zero
`solsys_code/tests/` changes and instead treated their own investigation scripts/queries as the
evidence artifact — this phase should follow the identical pattern, not invent unit tests for
findings that live entirely in a decision document.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django `TestCase` via `python manage.py test` (existing, unchanged by this phase) |
| Config file | none — no new test infrastructure needed |
| Quick run command | Not applicable to this phase's own deliverable; if the spike's investigation scripts touch a real or copied DB, use the Phase 26 precedent (`tmp/26_*_probe.py` run via `python manage.py shell < script.py`, disposable/git-excluded) |
| Full suite command | `python manage.py test` (only relevant as a **regression check**: confirm this phase's investigation activity — reading the DB, running probe scripts — leaves the existing suite green, since nothing in `solsys_code/` is edited) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCHEMA-01 | Decision doc states nullable-vs-sentinel-vs-per-proposal verdict, backed by a real dev-DB count | Investigation script (not a unit test) | `sqlite3 src/fomo_db.sqlite3 "SELECT COUNT(*) FROM solsys_code_campaignrun WHERE campaign_id IS NULL;"` (already run this session: 0/49) | N/A — no test file, evidence lives in `31-DECISION.md` |
| SCHEMA-02 | Candidate `source_identifier` scheme checked against both existing partial constraints | Constructed-input code-path check, mirroring `26-DECISION.md`'s SPIKE-01 `tmp/26_integrity_check.py` pattern (write against a disposable DB copy, confirm no unexpected `IntegrityError`, confirm expected ones still fire) | `python manage.py shell < tmp/31_constraint_probe.py` (script to be written by the executor, disposable/git-excluded) | ❌ — Wave 0: write this throwaway probe script, never commit it |
| SCHEMA-03 | Classical adapter identity surface confirmed against real schedule-file samples | Manual inspection (no automated test possible without a real file) | N/A | ❌ — depends on operator-supplied file, see Open Question 1 |
| SCHED-07 | flock/cron/network verified against real host and container | Shell-level verification script | `flock --version`, `crontab -l`, `curl -sS -o /dev/null -w '%{http_code}' https://hc-ping.com/` (all run this session for the interim host; repeat inside the container image per Pitfall 5) | ❌ Wave 0 for the container-level check — nothing to write, just run and record output |
| Regression | Existing suite still green after any investigation activity | full suite | `python manage.py test` (excluding `test_views.TestEphemeris`, per the known ASSIST segfault gotcha) | ✅ — suite already exists |

### Sampling Rate

- **Per investigation step:** re-run the specific probe/query and record output verbatim in
  `31-DECISION.md`, exactly as `26-DECISION.md` and `18-DECISION.md` do.
- **Per plan merge / phase gate:** `python manage.py test` (excluding the known ASSIST-segfault
  test class) to confirm the investigation left no accidental edits behind — this phase's
  entire correctness bar for the *existing* suite is "unchanged," not "passing new tests."

### Wave 0 Gaps

- [ ] No new `tests/*.py` file — this phase adds no source behavior to test.
- [ ] Disposable, git-excluded probe scripts for the schema-track constraint checks (mirror
      `tmp/26_integrity_check.py`'s pattern: write for real against a **disposable copy** of
      `src/fomo_db.sqlite3`, never against the live file, per the D-08/CLAUDE.md real-dev-DB
      convention and the Phase 26 evidence-posture precedent).
- [ ] A container-level shell check for `flock`/network reachability (Pitfall 5) — not a test
      file, a one-time verification step to run and quote verbatim in `31-DECISION.md`.

*(No gaps in the existing automated test infrastructure — this phase's validation is entirely
the investigation evidence itself, not new automated coverage.)*

## Security Domain

`security_enforcement` is enabled (`.planning/config.json` `workflow.security_enforcement: true`,
`security_asvs_level: 1`). This phase ships no code, so most ASVS categories have nothing to
apply to yet — the categories below are scoped to what the *decisions* this phase locks will
constrain for Phase 34, since a spike's recommendation can foreclose or open a security posture
even without writing code itself.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This phase touches no auth surface |
| V3 Session Management | No | No session-handling code involved |
| V4 Access Control | No | No new endpoints or permission checks |
| V5 Input Validation | No | No new user-facing input surface (classical schedule files are operator-supplied, not user-submitted through the app) |
| V6 Cryptography | Marginally — informs Phase 34 | D-04 commits to plain environment-variable credential injection (extending `FINK_CREDENTIAL_*`), the same posture `.planning/codebase/CONCERNS.md` already flags as a pre-existing weakness ("Hardcoded API Keys in settings.py... Credentials must be injected post-deployment"); this phase should not introduce a *worse* posture (e.g. credentials in a cron crontab line's argument list, visible via `ps`) than the existing env-var pattern |
| V7 Error Handling / Logging | Yes — directly relevant | SCHED-10 ("No credential value... appears in any log line or notification") is this phase's own requirement to ground for Phase 34; see Common Pitfalls §6 |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Credential exposure via `ps`/process listing if a cron line embeds a secret as a CLI argument | Information Disclosure | Keep credentials in environment variables (D-04, already locked) — never pass as a `manage.py` positional/keyword CLI argument, which is visible to any local user via `ps aux` |
| Credential exposure via exception message forwarded into a staff-notification email | Information Disclosure | See Common Pitfall #6 — sanitize exception `str()` before it reaches any `send_mail()`-based notification path |
| SSRF-adjacent: an unattended job's outbound heartbeat ping URL becomes attacker-controlled if ever made configurable per-environment without validation | Tampering / Information Disclosure | Not a risk today (a hardcoded or env-var-configured `hc-ping.com` URL, not user input), but Phase 34 should not make the ping target configurable via any untrusted input surface |
| Overlapping unattended invocations causing a partial/corrupted write under SQLite's single-writer model | Denial of Service (data integrity) | `flock` overlap prevention, per D-02 — already the chosen mitigation, reinforced by Common Pitfall #4's real-host evidence that unguarded cron entries already exist |

## Sources

### Primary (HIGH confidence)
- `solsys_code/models.py:78-460` (`CampaignRun`, `CampaignRunObservation`, `CalendarEventMeta`, `CalendarEventDismissal` — read this session)
- `solsys_code/management/commands/load_telescope_runs.py` (full file, read this session)
- `solsys_code/telescope_runs.py:1-50` (`SITES`, `ESO_NOON_TO_NOON_SITES`, `KNOWN_STATUSES` — read this session)
- `solsys_code/campaign_views.py:300-358` (`_notify_staff`, `ApprovalQueueView` docstring — read this session)
- `src/fomo/settings.py` (FINK_CREDENTIAL_* `os.getenv()` pattern, lines 309-329 — read this session)
- Real dev DB `src/fomo_db.sqlite3` — direct `sqlite3` CLI queries this session (total/null-campaign counts, schema DDL, window/site null counts)
- This session's shell: `flock --version`, `crontab -l`, `systemctl --version`, `curl` to `hc-ping.com`, `python3`/`django` version checks
- `.planning/milestones/v2.2-phases/26-canonical-record-spike/26-DECISION.md` (full document, read this session)
- `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md` (full document, read this session)
- `docs/design/canonical_record_spike.rst`, `docs/design/uncertain_scheduling_spike.rst` (full documents, read this session)
- `docs/design/telescope_runs_calendar.rst:169-209` ("Classical Run Input Format" section, read this session)
- `.planning/codebase/CONCERNS.md`, `.planning/codebase/INTEGRATIONS.md` (full documents, read this session)
- `.planning/research/SUMMARY.md` (full document, read this session)
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `31-CONTEXT.md` (full documents, read this session)

### Secondary (MEDIUM confidence)
- Django 5.2 documentation patterns for `ForeignKey.null`, `on_delete=SET()`, `get_or_create()` — training-knowledge level, consistent with Django's stable public API but not independently re-fetched from docs.djangoproject.com this session

### Tertiary (LOW confidence)
- healthchecks.io's exact ping-API contract (headers, payload requirements beyond a bare GET/POST) — not independently verified against healthchecks.io's own documentation this session, flagged as Assumption A3

## Metadata

**Confidence breakdown:**
- Schema/identity track: HIGH for what real dev-DB evidence could confirm this session (campaign nullability, constraint field tuples, existing row counts); explicitly LOW/open for the classical-file proposal-code question, which requires operator-supplied data no research pass can substitute for
- Scheduling track: HIGH for the interim-host-level findings (flock, cron precedent, network reachability), all independently verified this session; explicitly open for the container-image and AWS-deployment levels, which this session's shell cannot reach
- Pitfalls: HIGH — grounded in this session's direct reads of the real code, the real DB, and this repo's own prior spike precedents (26/18), not speculation

**Research date:** 2026-09-01
**Valid until:** This is investigation-only groundwork for a spike phase whose own execution will re-verify everything against real, possibly-fresher data (per D-08's "real dev-DB rows" requirement) — treat this document as scaffolding for that execution, not as a substitute for it. Estimate 7 days validity for the host-level findings (crontab/flock could change), 30 days for the schema/constraint findings (stable unless another phase edits `models.py` first).
