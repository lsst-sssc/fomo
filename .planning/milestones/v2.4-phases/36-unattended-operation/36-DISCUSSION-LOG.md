# Phase 36: Unattended Operation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-16
**Phase:** 36-Unattended Operation
**Areas discussed:** Schedule & cron layout, Watched-proposal list, Failure signalling, Credential hygiene & logs

---

## Schedule & cron layout

| Option | Description | Selected |
|--------|-------------|----------|
| One FOMO runner command, one cron entry | Fixed sequence updatestatus → sweep → discovery → reconcile in one entry point; logs skips, pings heartbeat, mails on failure; settles same-minute ordering by construction | ✓ |
| Four separate cron entries, staggered minutes | Phase 31's per-command shape ×4; ordering by timing only; wrapping repeated four times | |
| Two entries: a fast chain and a slow discovery job | Status/sweep/reconcile frequent, discovery on its own slower entry | |

| Option | Description | Selected |
|--------|-------------|----------|
| FOMO-owned step: LCO/SOAR only, fails loudly | Calls `update_all_observation_statuses()` for LCO/SOAR, non-empty failed list = failure, logs id + exception class only | ✓ |
| Call stock `updatestatus` via call_command, as-is | Hits every facility incl. stubs, always exits 0, prints `str(e)` | |
| Keep `updatestatus` out of FOMO's schedule | Narrowing depends on a job FOMO neither owns nor monitors | |

| Option | Description | Selected |
|--------|-------------|----------|
| Every 15 min, discovery every tick | Narrowing within a quarter hour; new proposal appears within one tick | ✓ |
| Every 15 min, discovery hourly | Runner flag or wall-clock check; two cron lines for one command | |
| Hourly for everything | Minimal portal load, up to an hour of lag | |

| Option | Description | Selected |
|--------|-------------|----------|
| Committed crontab template + a `check_unattended` command | `deploy/cron/fomo.crontab.example`; command verifies prerequisites and prints the cron line | ✓ |
| Runbook-only: the cron line is documented text | No repo artifact, manual checklist | |
| Template + runbook, no check command | Template committed; verification manual | |

**User's choice:** all four recommended options.
**Notes:** Step-failure isolation (never stop later steps; aggregate to one non-zero exit) and a `--dry-run` on the runner left to Claude's discretion.

---

## Watched-proposal list

| Option | Description | Selected |
|--------|-------------|----------|
| Code + active flag + per-run bookkeeping | `proposal_code`, `is_active`, `target_list_name`, `attributed_to`, read-only `last_run_at`/`last_run_summary`; no created-after bound | ✓ |
| Code + active flag only | Minimal; last-run answered from logs | |
| Code + active + created-after bound | Per-proposal date bound, no bookkeeping | |

| Option | Description | Selected |
|--------|-------------|----------|
| Optional override; default = every active watched proposal | Bare run sweeps active rows; `--proposal X` still works one-off | ✓ |
| Drop it: the watch list is the only source | Literal "replaces"; 30 tests + notebook rewritten | |
| Keep it required; add a separate `--watched` flag | Two mutually exclusive selectors | |

| Option | Description | Selected |
|--------|-------------|----------|
| Quiet no-op, but `check_unattended` warns | One INFO line, exit 0; setup check and runbook flag the empty list | ✓ |
| Warning-level log every tick | Visible in log, no email | |
| Treat as a failure | Mails every 15 min until a row exists | |

| Option | Description | Selected |
|--------|-------------|----------|
| Per-proposal isolation; failure recorded on the row | Own try block per proposal; failure to `last_run_summary`; step non-zero at end | ✓ |
| First failure aborts the discovery step | One bad proposal blocks the rest | |

**User's choice:** all four recommended options.
**Notes:** A `facility` field (recommend none — shared portal/proposal namespace) and admin list/filter layout left to Claude's discretion.

---

## Failure signalling

| Option | Description | Selected |
|--------|-------------|----------|
| Any step exiting non-zero, incl. a non-empty per-record failure count | Raises or failed-record counters mail; expected data-shape skips do not | ✓ |
| Only an uncaught exception / crash | Per-record failures look healthy | |
| Any non-zero counter at all, including skips | Mails on known-benign states | |

| Option | Description | Selected |
|--------|-------------|----------|
| In the runner, once per tick, suppressed while the same failure persists | Extracted request-free `notify_staff`, `fail_silently=False` caught+logged; persisted last-notified state; daily reminder; "cleared" email | ✓ |
| In the runner, every failing tick, no throttling | Mails every 15 min | |
| Inside each command | Manual runs mail too; four call sites | |

| Option | Description | Selected |
|--------|-------------|----------|
| One check for the whole tick: /start then /<exit-code> | Single env var URL; service-agnostic; ping failures never fail the tick | ✓ |
| One check per step (four URLs) | Finer dashboards, four env vars | |
| Success-only ping | Hung and failed look the same | |

| Option | Description | Selected |
|--------|-------------|----------|
| Runner skips pinging; `check_unattended` reports it as missing | Unset URL allowed, logged once per tick | ✓ |
| Runner refuses to start | Blocks deployments where external ping is not allowed | |

| Option | Description | Selected |
|--------|-------------|----------|
| Staff users with an email on file (existing idiom) | Same rule as `_notify_staff()`; check fails if none | ✓ |
| A dedicated operator list setting | `FOMO_OPERATOR_EMAILS` | |
| Both: staff plus an optional operator list | Union | |

**User's choice:** all five recommended options.
**Notes:** Suppression store (state file vs. tiny model), daily-reminder interval and ping timeout left to Claude's discretion.

---

## Credential hygiene & logs

| Option | Description | Selected |
|--------|-------------|----------|
| Discipline + a redacting filter as belt-and-braces | Class-name-only rule plus a root-logger `logging.Filter` replacing known secret values | |
| Discipline + regression tests only | No runtime filter; tests seed a fake key, force each failure path, assert absence in logs and mail | ✓ |
| Runtime filter only | Log lines interpolate freely | |

| Option | Description | Selected |
|--------|-------------|----------|
| Per-tick log via cron redirect to one file, rotated by logrotate | `/var/log/fomo/unattended.log` placeholder; committed logrotate example; check verifies dir | ✓ |
| Runner writes its own `RotatingFileHandler` | Path from a setting | |
| Leave it to cron mail / journald | No persistent per-tick history | |

| Option | Description | Selected |
|--------|-------------|----------|
| Step name, exit code, counters, scrubbed exception class + first line | Plus log path and admin link; no traceback/URLs/response text | ✓ |
| Full scrubbed traceback | Richer, trusts the scrub completely | |
| Bare ping only | "See the log" | |

| Option | Description | Selected |
|--------|-------------|----------|
| Class name only for network/portal/mail errors; full message otherwise | Portal/`requests`/`send_mail` exceptions → class only; FOMO-raised exceptions carry their message | ✓ |
| Class name only, everywhere | Uniform, less informative | |
| Full message everywhere | Relies entirely on tests | |

**User's choice:** the non-recommended "Discipline + regression tests only" for enforcement (runtime filter declined); recommended options for the other three. The exception-text follow-up was asked because, without a runtime filter, the email's "first line" was the one place a portal message could leak.
**Notes:** Log line format, retention length, and a `check_unattended --send-test-email` left to Claude's discretion.

---

## Claude's Discretion

- Runner command name, module layout, `--dry-run` / `--step` flags.
- Lock strategy inside the runner (recommended: cron-line `flock` on the runner + per-command `fcntl` locks in each step).
- Suppression store, reminder interval, ping timeout, log banner format, crontab skip-line shape.
- Test layout; whether the runner gets its own pre-executed notebook (only if executable offline with portal/mail/heartbeat mocked).

## Deferred Ideas

- AWS/Kubernetes scheduling, container image, third-party-heartbeat policy (Phase 31 open items).
- Runtime log-redaction filter.
- Watched list feeding `backfill_lco_observation_records` or `project_observation_calendar --proposal`.
- `check_unattended --send-test-email`.

## Todos reviewed

- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — already shipped as Phase 35 D-13; not folded (closable).
- `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` — keyword-only match; not folded.
