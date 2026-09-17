# API Coverage — healthchecks-compatible ping endpoint + LCO/SOAR Observation Portal

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
>
> Detector note: `api-coverage.cjs --json` over the Phase 36 scope returned
> `{"detected": false}` this session. This matrix is written anyway because the phase
> genuinely makes outbound calls to two external services (a healthchecks-compatible ping
> endpoint per D-12, and the LCO/SOAR Observation Portal via TOM's facility classes and
> `make_request`), and because a seal-time re-run over the finished PLAN.md bodies may
> classify the phase differently. Decided at plan time, not left for seal time.

## Service 1 — healthchecks-compatible ping API (new integration, D-12)

Capability surface taken from the healthchecks.io Pinging API (RESEARCH.md "Code Examples",
[CITED: healthchecks.io/docs/http_api/]). The runner is deliberately service-agnostic: any
endpoint implementing this ping convention, hosted or self-hosted, must work.

| capability | decision | reason |
|---|---|---|
| `GET <url>/start` | INTEGRATE | |
| `GET <url>/<exit-code>` (0 = success, non-zero = failure) | INTEGRATE | |
| `GET <url>/fail` | OPT-OUT | `/<exit-code>` with a non-zero code already carries the failure signal; D-12 specifies exactly `/start` then `/<exit-code>`, and a second failure verb would give two ways to say one thing |
| `GET <url>/log` (informational ping, no status change) | OPT-OUT | the rotated log file (D-18) is the log surface; the heartbeat carries only the pass/fail signal |
| Run IDs (`?rid=<uuid>` correlating a `/start` with its terminating ping) | OPT-OUT | `flock -n` guarantees one tick at a time (D-01), so start/end pairs cannot interleave and need no correlation id |
| Auto-provisioning (`?create=1` / slug ping URLs) | OPT-OUT | the operator creates the check once from the runbook (D-12's grace-period guidance); the runner never provisions a check |
| POST body / exit-status payload in the ping body | OPT-OUT | D-14 puts the failure detail in the email and D-18 puts it in the log file; sending step summaries to a third-party service would widen the credential-disclosure surface SCHED-10 exists to close |
| Management API (checks CRUD, API-key authenticated) | OPT-OUT | binds the runner to one vendor's control plane; D-12 requires service-agnostic, any-healthchecks-compatible-endpoint behavior |

## Service 2 — LCO/SOAR Observation Portal (existing integration, extended by D-03/D-07)

Capability surface = the portal operations reachable through the `tom_observations` facility
classes and `make_request` that this phase's unattended path either uses or deliberately does not.

| capability | decision | reason |
|---|---|---|
| `update_all_observation_statuses()` — per-record status refresh (LCO) | INTEGRATE | D-03, step 1 |
| `update_all_observation_statuses()` — per-record status refresh (SOAR) | INTEGRATE | D-03, step 1 |
| `update_observation_status(observation_id)` — single-record refresh | INTEGRATE | D-03's class-name re-derivation (RESEARCH.md Open Question 1) |
| `GET /api/requestgroups/` paging (proposal discovery) | INTEGRATE | D-07, step 3 — existing `_iter_request_groups()`, now driven by `WatchedProposal` |
| Observed-block lookup (embedded `observations` block / `resolve_placement_block()` fallback) | INTEGRATE | step 2's site-lookup hook, reused unchanged from `project_observation_calendar` |
| `get_observation_url()` — event key namespace | INTEGRATE | reused unchanged by the projector (spike 002 requirement) |
| `submit_observation()` / `cancel_observation()` | OPT-OUT | the unattended path is read-only by design — FOMO never submits or cancels an observation from cron; nothing in SCHED-08/09/10 or DISCOVER-01 asks for it |
| `GET /api/proposals/` (portal-side proposal/allocation listing) | OPT-OUT | not needed for SC 2 — the watched list is operator-maintained in the admin (D-06); auto-discovering proposals from the portal is a different feature |
| Gemini / ESO facility status refresh | OPT-OUT | explicitly out of scope (36-CONTEXT.md "Out of scope" — no real read-back exists for those stubs); D-03 scopes step 1 to LCO/SOAR |
| `backfill_lco_observation_records` (campaign-bound sibling) portal path | OPT-OUT | explicitly out of scope (36-CONTEXT.md "Out of scope" — unchanged this phase); deferred item: "letting the campaign-bound sibling read the watched list" |

## Service 3 — SMTP (Django mail backend, D-11/D-13/D-14)

| capability | decision | reason |
|---|---|---|
| `send_mail()` failure notification (staff recipients) | INTEGRATE | D-11/D-14 |
| `send_mail()` recovery ("cleared") notification | INTEGRATE | D-11 |
| `send_mail()` test message (`check_unattended --send-test-email`) | INTEGRATE | planner discretion accepted — the only way an operator can verify the mail layer during setup (SC 5) without waiting for a real failure |
| HTML / multipart alternative bodies | OPT-OUT | D-14 specifies a plain-text body with no traceback, no request URLs, no portal response text; an HTML part would duplicate that content with no operator benefit |
| Per-recipient delivery tracking / bounce handling | OPT-OUT | explicitly out of scope — REQUIREMENTS.md Out-of-Scope table: "A full alerting/notification pipeline"; email + heartbeat meets the SCHED-09 bar |
