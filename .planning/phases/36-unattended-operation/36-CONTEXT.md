# Phase 36: Unattended Operation - Context

**Gathered:** 2026-09-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 36 makes the pipeline the earlier phases built run on the real interim host with nobody
typing anything. One FOMO-owned runner management command, invoked from one `flock -n`-guarded
cron line every 15 minutes, executes a fixed sequence of four steps: refresh LCO/SOAR observation
statuses (a FOMO-owned replacement for the relevant part of TOM's `updatestatus`), the observation
projector sweep (`project_observation_calendar`), discovery of robotically scheduled observations
(`backfill_lco_observations`, now driven by an admin-editable `WatchedProposal` list instead of a
required `--proposal` argument), and the campaign reconciler sweep (`reconcile_campaign_runs`).

When a step fails, an operator finds out two independent ways: the runner emails staff once per
failing tick (suppressed while the same failure persists, with a "cleared" email on recovery), and
a healthchecks-style heartbeat receives `/start` before the first step and `/<exit-code>` after the
last, so a tick that never ran, hung, or failed all alert. No credential value reaches a log line,
email, or error message on this path — enforced by the existing class-name-only logging rule for
network/portal/mail exceptions plus regression tests, not by a runtime filter.

An operator sets up or verifies the whole schedule on a fresh host from one runbook section: a
committed crontab template and logrotate example, plus a `check_unattended` management command that
verifies every prerequisite (flock, lock/log dirs, email backend and staff recipients, heartbeat URL,
at least one active watched proposal) and prints the exact cron line to install.

**In scope:** SCHED-08, SCHED-09, SCHED-10, DISCOVER-01. Paired docs (CLAUDE.md rule, in
`files_modified` up front): `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (the
command's argument handling and discovery source change) and `docs/runbooks/telescope_runs_calendar.rst`
(a new unattended-operation section: the schedule, the failure signals, what to check when nothing
has appeared; the cheat-sheet row for `backfill_lco_observations` and new rows for the runner and
`check_unattended`; the troubleshooting section).

**Out of scope:** any task-queue scheduler (Celery/huey/APScheduler — Phase 31 settled cron +
`flock`); the AWS/Kubernetes deployment story (`CronJob` with `concurrencyPolicy: Forbid`, the
third-party-ping policy question) — the interim host only, Phase 31's open items stay open; a
container image; a full alerting pipeline beyond email + heartbeat; a runtime log-redaction filter
(declined, see D-16); Gemini/ESO status refresh (no real read-back); changes to the campaign-bound
sibling `backfill_lco_observation_records`; Phase 37's status vocabulary, tallies, UNUSED-01, GAPB-01.

</domain>

<decisions>
## Implementation Decisions

### Schedule & cron layout (SCHED-08)

- **D-01: One FOMO runner command, one cron entry.** A single management command (name is the
  planner's, e.g. `run_unattended`) executes the fixed sequence status-refresh → sweep → discovery →
  reconcile, in that order, in one process. The crontab line is Phase 31's exact shape —
  `/usr/bin/flock -n /var/lock/fomo/<runner>.lock /path/to/venv/bin/python /path/to/checkout/manage.py <runner>`
  with absolute paths — plus a stdout+stderr append-redirect to the log file (D-18). This settles
  Phase 31's open "same-minute ordering" question by construction: the order is the runner's, not
  cron's. A skipped tick (`flock -n` contended) is visible through the heartbeat (D-11) and the
  crontab template's own `|| echo`-style skip line (planner's choice of shape) so a permanently
  contended lock never looks like a healthy no-op.
  — **Reversibility:** reversible — splitting into per-command entries later is a crontab change and
  four copies of the wrapping logic.
- **D-02: Step-failure isolation, one exit code.** A failed step never stops the later steps; the
  runner records each step's outcome, runs every step, and exits non-zero at the end if any step
  failed. Each step runs the shipped command's logic (`call_command()` or the command's own module
  functions — planner's choice), never a subprocess `manage.py` re-invocation.
- **D-03: The status-refresh step is FOMO-owned, LCO/SOAR only, and fails loudly.** The runner does
  not call TOM's stock `updatestatus` (which loops over every registered facility including the
  Gemini/ESO stubs, always exits 0, and prints `str(e)` for every failed record). Instead it calls
  `update_all_observation_statuses()` on a fresh `LCOFacility()` and `SOARFacility()` instance
  (facility instance per facility, never shared — 34 D-10), treats a non-empty failed-records list as
  a step failure, and logs only `observation_id` + `type(exc).__name__` — never the exception text
  TOM returns (SCHED-10). The receiver Phase 34 installed does the narrowing on each record save;
  the sweep that follows makes the one-time observed-site lookups (34 D-08).
- **D-04: Every 15 minutes, discovery on every tick.** Narrowing shows within a quarter hour and a
  newly watched proposal's observations appear within one tick (SC 2). Discovery pages the portal
  each tick (the dev-DB proposal is ~150 records; the sweep itself measured 0.1–0.3 s); the lock
  covers a slow tick. The documented cron expression is `*/15 * * * *`.
- **D-05: Committed host artifacts + a setup-check command (SC 5).** The repo gains a
  `deploy/cron/fomo.crontab.example` (paths as placeholders) and a `deploy/logrotate/fomo.example`;
  a new `check_unattended` management command verifies `flock` is present, the lock and log
  directories exist and are writable, the email backend is not the console backend and at least one
  staff user has an email, the heartbeat URL env var is set (a warning, not a failure — D-12), and at
  least one active `WatchedProposal` exists (a warning — D-08); it exits non-zero on a hard failure
  and prints the exact cron line to install with the real resolved interpreter and `manage.py`
  paths. The runbook's new section walks: copy the template, run `check_unattended`, install the
  line, read the log, watch the heartbeat.

### Watched-proposal list (DISCOVER-01)

- **D-06: `WatchedProposal` model.** Fields: `proposal_code` (unique, the exact LCO/SOAR proposal
  code), `is_active` (default true), `target_list_name` (optional; replaces `--target-list`, default
  derived `<code>_targets` as today), `attributed_to` (optional FK to user; replaces `--username`),
  plus read-only bookkeeping the runner writes after each sweep of that proposal: `last_run_at` and
  `last_run_summary` (the command's per-proposal summary line, or the failure's step + exception
  class). No `created_after` bound: discovery always considers the whole proposal, as the runbook
  documents today. No `facility` field: LCO and SOAR share one portal and one proposal namespace.
  Registered in the admin with `list_display` showing code / active / last run / summary,
  `list_editable` on `is_active`, and a filter on `is_active`. One small additive migration.
  — **Reversibility:** costly — a schema migration; renaming or dropping fields later touches every
  row and the admin.
- **D-07: `--proposal` becomes an optional override on `backfill_lco_observations`.** Bare
  invocation (what the runner uses) sweeps every active `WatchedProposal`, applying each row's
  target-list override and attribution user; `--proposal X` (with the existing `--target-list` /
  `--username` / `--created-after` / `--created-before` flags) still works for a one-off manual run
  and does not require X to be watched. The 30 existing tests and the notebook's `--proposal` cells
  keep working; the notebook gains cells that seed rows, run bare, and show `last_run_summary`.
- **D-08: Empty list is a quiet no-op.** With no active row, discovery logs one INFO line
  ("0 watched proposals, nothing to discover"), exits 0, and the tick is healthy. `check_unattended`
  warns about it, and the runbook's "nothing has appeared" section lists it first.
- **D-09: Per-proposal failure isolation, recorded on the row.** Each watched proposal is swept in
  its own try block; a portal or data error on one is caught, written to that row's
  `last_run_summary`, counted, and the next proposal still runs. The discovery step exits non-zero
  at the end if any proposal failed, and the failure email names the proposal code(s).

### Failure signalling (SCHED-09)

- **D-10: What is a failure.** A step that raises, or that finishes with a non-empty per-record
  failure count — status-refresh errors (D-03), the sweep's `failed:` counter, a discovery
  proposal error (D-09), a reconciler per-run failure — makes the tick non-zero and triggers the
  email. Expected data-shape outcomes the commands already isolate and report (`unprojectable`,
  `skipped`, `detach_declined`, `remint_declined`, `unchanged`) are not failures; they stay in the
  log and the summary.
- **D-11: The runner sends the email, once per failing tick, with suppression.** The runner (not
  the individual commands, so a manual command run never mails) sends through a request-free
  `notify_staff()` helper extracted from `campaign_views.CampaignRunSubmissionView._notify_staff()`
  (Phase 31's instruction): base URL from a setting (e.g. `FOMO_BASE_URL`, used for the admin and
  log links), same staff-with-email recipient rule, `fail_silently=False` with the exception caught
  and logged at ERROR — never re-raised, and reported as class name only (D-17). A persisted
  "last notified" record (store is the planner's — a small state file next to the lock file, or a
  tiny model) suppresses repeat emails while the same set of failing steps persists, sends a daily
  reminder while it does (interval is the planner's), and sends one "cleared" email when a tick
  succeeds after a failure. The existing submission notice switches to the extracted helper (still
  `fail_silently=True` semantics there — a mail outage must not break a submission).
- **D-12: One whole-tick heartbeat: `/start` then `/<exit-code>`.** Before the first step the
  runner pings `<URL>/start`; after the last it pings `<URL>/<exit-code>` (0 on success, the
  runner's non-zero code otherwise), so a never-invoked tick (no start within the grace period), a
  hung tick, and a failed tick all alert on the receiving service. The URL comes from one
  environment variable (e.g. `FOMO_HEARTBEAT_URL`) read via `os.getenv()` like `FINK_*`, is
  service-agnostic (any healthchecks-compatible endpoint, hosted or self-hosted; Phase 31 confirmed
  `hc-ping.com` egress), and uses `requests` with a short timeout. A ping failure is logged and never
  fails the tick. When the variable is unset the runner logs one INFO line per tick and skips
  pinging; `check_unattended` reports the second layer as off (a warning). The runbook documents a
  grace period a little above one 15-minute interval (e.g. 20 minutes) and the recommended
  one-check-per-schedule setup.
- **D-13: Recipients are staff users with an email on file** — the existing `_notify_staff()`
  idiom; nothing new to configure. `check_unattended` fails if no staff user has an email.
- **D-14: Email content.** Subject `FOMO unattended run failed: <step(s)>` (and
  `FOMO unattended run recovered` on clear); body lists each failed step with its exit outcome, the
  step's counters/summary line, the exception class (+ message only where D-17 allows), the
  proposal code(s) for discovery failures, the log file path, and links to the admin
  `WatchedProposal` page and the calendar. No traceback, no request URLs, no portal response text.

### Credential hygiene & logs (SCHED-10)

- **D-15: Credentials are environment variables only.** Any new value the unattended path needs
  (heartbeat URL, base URL if not a setting) is read via `os.getenv()`; nothing is ever a
  command-line argument or embedded in the cron line (Phase 31). `check_unattended` prints
  variable *names* and set/unset status, never values.
- **D-16: SCHED-10 is enforced by discipline plus regression tests — no runtime redacting filter.**
  Every log and email line on the unattended path follows the existing rule (D-17). Tests seed a
  fake API key / email password / heartbeat URL into settings, force each failure path (portal
  error in status refresh, discovery, the sweep's site lookup; mail send failure; heartbeat ping
  failure), and assert the fake values are absent from captured logs, stdout/stderr, and
  `mail.outbox`. A runtime filter was offered and declined.
- **D-17: What an exception may contribute.** An exception raised from a `requests`/facility/portal
  call or from `send_mail()` is reported as `type(exc).__name__` only (the rule
  `resolve_placement_block()` and the projector already follow). Exceptions FOMO raises itself
  (`CommandError`, `sun_event()`'s `ValueError`, a `ValidationError`) carry their message, since
  those strings never embed a response body or header. The runner's per-step wrapper decides which
  by where the exception was caught, not by exception type alone.
- **D-18: Log destination is one file, rotated by logrotate.** The crontab template appends
  stdout+stderr to `/var/log/fomo/unattended.log` (an operator placeholder path); the committed
  logrotate example rotates it daily and keeps 14. Each tick logs a start/end banner with a
  timestamp and each step's summary line so a tick is readable in isolation. `check_unattended`
  verifies the directory is writable. The runbook's "what to check when nothing has appeared"
  section points at this file, the heartbeat dashboard, and the admin's `last_run_summary`.

### Claude's Discretion

- Runner command name and module layout (e.g. `solsys_code/unattended.py` holding the step
  functions and the notification/heartbeat helpers, with thin management commands for the runner
  and `check_unattended`); whether the runner offers `--dry-run` (recommended: yes, passing dry-run
  through to every step and never pinging or mailing) and `--step <name>` for a single step.
- Locking inside the runner. Recommended: the cron line's `flock -n` guards the runner as a whole
  (D-01), and each step also takes its own per-command `fcntl.flock` on
  `/var/lock/fomo/<command>.lock` (non-blocking, skip-and-log when contended) so a manual
  `manage.py <command>` run and a tick never overlap — honouring Phase 31's one-lock-per-command
  rule without four cron lines. Lock directory from a setting with a sensible default.
- Suppression store for D-11 (state file vs. model), the daily-reminder interval, the ping timeout,
  the exact log banner format.
- Shape of the `flock` skip line in the crontab template; whether `check_unattended` can also
  test-send an email (`--send-test-email`) — nice for the runbook, optional.
- Whether `project_observation_calendar` keeps accepting `--proposal` for manual narrowing (yes —
  unchanged; the runner calls it bare).
- Test layout: new `test_unattended.py` / `test_check_unattended.py` / `test_watched_proposal.py`,
  and extension of `test_backfill_lco_observations.py` for D-07..D-09; how the notebook shows the
  watched-list run (mocked portal, as the existing cells do).
- Whether the runner gets its own pre-executed demo notebook. CLAUDE.md pairs a new module with a
  notebook; a runner demo can only execute offline with the portal, mail and heartbeat mocked, so
  add one only if that is achievable in the same style as `backfill_lco_observations_demo.ipynb`
  — otherwise the runbook section is the runner's paired doc and CLAUDE.md's map records that.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decisions this phase executes
- `.planning/ROADMAP.md` §"Phase 36: Unattended Operation" — goal, the five success criteria,
  scope note (cron + `flock -n`, no task queue, PR #43 / SEED-003 bar, DISCOVER-01 replaces
  `--proposal`), paired-docs list; and the milestone's "Locked constraints" block.
- `.planning/REQUIREMENTS.md` — SCHED-08, SCHED-09, SCHED-10, DISCOVER-01 (this phase); the
  Out-of-Scope table rows for a task-queue scheduler and a full alerting pipeline; Phase 37's
  requirements for what not to pre-empt.
- `.planning/PROJECT.md` §"Current Milestone: v2.4 Observation-First Calendar" — the "Unattended
  operation (SEED-003)" target feature, landmines, conventions carried.

### The spike verdict this phase implements
- `docs/design/run_identity_and_unattended_invocation_spike.rst` §"Unattended invocation (SCHED-07)"
  — the invocation mechanism, overlap prevention (one lock per command, skip not queue, log the
  skip), credential handling (env vars, never CLI/cron), missed-invocation visibility (two layers;
  `_notify_staff()` must be extracted request-free with `fail_silently=False`), and the "Future
  scope" list of open items this phase does not close (AWS policy, container image, same-minute
  ordering — the last is closed by D-01).
- `.planning/milestones/v2.3-phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md`
  §"SCHED-07 evidence" and §"SCHED-07 - unattended invocation mechanism" — the real-host probe
  (flock present, 0/3 unguarded cron entries, `hc-ping.com` egress HTTP 301), the exact cron line
  shape D-01 copies, the `local_settings.py` redaction constraint on any evidence-gathering.

### Prior-phase decisions this phase builds on
- `.planning/phases/34-the-observation-projector-trigger/34-CONTEXT.md` — D-08 (the sweep makes
  the site lookup, Phase 36 runs it right after the status refresh), D-13/D-16 (credential-free
  logging, never-raise receiver), D-17 (the sweep is called bare), D-20 (SCHED-06's live proof
  assumed `updatestatus` on cron).
- `.planning/phases/35-allocation-layer-classical-cutover/35-CONTEXT.md` — D-11 (receivers do the
  per-save work; `reconcile_campaign_runs` is the backstop sweep the runner's last step calls),
  D-15 (the cutover is one-time and already run — not scheduled).
- `.planning/milestones/v2.2-phases/29-the-reconciler/29-CONTEXT.md` — D-05/D-06: the sweep's
  summary form and per-run failure isolation D-10 reads.

### The code this phase writes into, reuses, or wraps
- `solsys_code/management/commands/backfill_lco_observations.py` — `Command.add_arguments()`
  (`--proposal` required today; `--created-after/--created-before`, `--username`, `--target-list`,
  `--dry-run`), `Command.handle()` (~line 477; the `<proposal>_targets` `TargetList` step ~line
  678), `_iter_request_groups()`. D-07..D-09 rewrite the proposal selection around `WatchedProposal`.
- `solsys_code/management/commands/project_observation_calendar.py` — `Command.add_arguments()`
  (all optional; "Phase 36 calls this bare"), the per-facility summary line and `failed:` counter.
- `solsys_code/management/commands/reconcile_campaign_runs.py` — `--dry-run` only; per-run failure
  isolation and summary.
- `solsys_code/campaign_views.py` `CampaignRunSubmissionView._notify_staff()` (~line 326) — the
  helper D-11 extracts (recipient rule, no-PII body, `send_mail`).
- `solsys_code/calendar_utils.py` `resolve_placement_block()` (~line 300) — the class-name-only
  network-exception rule D-17 generalises; `solsys_code/observation_projector.py` — the
  `type(exc).__name__` logging idiom throughout.
- `solsys_code/management/commands/sync_gemini_observation_calendar.py` — D-04's password-strip
  idiom (the SYNC-09 discipline D-16 inherits).
- `solsys_code/models.py` — model home for `WatchedProposal`; `solsys_code/admin.py` — registration
  pattern (`CampaignRunAdmin`, `CalendarEventMetaAdmin`); `solsys_code/apps.py` `ready()` — where
  receivers live (no new receiver needed here).
- `src/fomo/settings.py` — `LOGGING` (console handler, root INFO), `EMAIL_BACKEND` (console by
  default; production values in `local_settings.py`), `FACILITIES['LCO'/'SOAR']['api_key']` folding
  from `LCO_API_KEY`, the `FINK_*` `os.getenv()` convention, `FOMO_DATABASE_PATH`.
- Installed tomtoolkit 3.0.1: `tom_observations/management/commands/updatestatus.py` (what D-03
  replaces and why), `tom_observations/facility.py` `update_all_observation_statuses()` (~line 540;
  returns `[(observation_id, str(e))]`, excludes terminal states) and `update_observation_status()`.
- Tests to extend: `solsys_code/tests/test_backfill_lco_observations.py` (30 tests, mocks
  `backfill_lco_observations.make_request`), `test_project_observation_calendar.py`,
  `test_reconcile_campaign_runs.py`, `test_admin.py`.

### Paired docs (CLAUDE.md rule — part of the deliverable)
- `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` — watched-list cells (D-07),
  `last_run_summary` (D-06), per-proposal isolation (D-09); check `docs/notebooks.rst` lists it.
- `docs/runbooks/telescope_runs_calendar.rst` — new §"How do I run everything unattended?" (setup,
  verify, schedule, failure signals, "nothing has appeared" checklist); §"How do I backfill
  ObservationRecords without a campaign?" (D-07); "Command cheat-sheet" rows; "Troubleshooting".
- `docs/installation.rst` §"Running FOMO Management Commands" (`running-management-commands`
  label) — cross-reference to the new runbook section if the planner finds it fits.
- `CLAUDE.md` notebook map — add the runner/`check_unattended` pairing per the discretion note.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- The three sweep commands already have the shape the runner needs: zero required arguments,
  `--dry-run`, per-record/per-run failure isolation, and a single summary line with a `failed:` (or
  equivalent) counter the runner can read to decide D-10.
- `_notify_staff()` is the only mail sender in the codebase; its recipient rule and no-PII body are
  reused as-is once extracted (D-11). `send_mail` + Django's `locmem` backend in tests gives
  `mail.outbox` for the D-16 assertions.
- `resolve_placement_block()` and the projector's `type(exc).__name__` idiom are the class-name-only
  rule D-17 extends to the runner's step wrapper.
- `update_all_observation_statuses()` on an `LCOFacility()`/`SOARFacility()` instance is the whole
  status-refresh step (D-03); it already excludes terminal-state records.
- `backfill_lco_observations` already threads `target_list` and `username` options through its
  `handle()`, so per-row overrides (D-06) are a source change, not new behaviour.
- Phase 31's transcript gives the exact host facts (`/usr/bin/flock`, util-linux 2.37.4) and cron
  line the template copies.

### Established Patterns
- Credential-free logging: exceptions from portal calls are logged by class only; the Gemini
  command strips `password` before anything is logged. Settings quoting is never committed
  (`local_settings.py` holds live secrets).
- Sweeps are idempotent and isolate failures per item; one-time commands are `--dry-run`-able and
  runbook-documented; every management command has a cheat-sheet row.
- Receivers connect in `SolsysCodeConfig.ready()` with `dispatch_uid`; none are needed here.
- Migrations small and additive; Target fixtures use `NonSiderealTargetFactory`; Google docstrings;
  single quotes; 120 cols; `pre-commit run ruff` is the gate.
- Paired notebooks are pre-executed against mocked network (`make_request` patched) so they run
  offline in CI.

### Integration Points
- New: `WatchedProposal` model + migration + admin; runner command; `check_unattended` command;
  `deploy/cron/` and `deploy/logrotate/` examples (new top-level `deploy/` directory — none exists).
- Changed: `backfill_lco_observations.add_arguments()`/`handle()` (D-07..D-09);
  `campaign_views._notify_staff()` → extracted helper; settings gain the base-URL setting and read
  the heartbeat env var.
- Host (documented, not committed): `/var/lock/fomo/`, `/var/log/fomo/`, the crontab entry, the
  logrotate drop-in, `FOMO_HEARTBEAT_URL` in the environment cron sees, a real `EMAIL_BACKEND` in
  `local_settings.py`. Neither directory exists on the developer host today; `check_unattended`
  reports that.
- Dev DB baseline (2026-09-16): one real proposal (`KEY2026B-004`, ~150 LCO records) is the
  obvious first `WatchedProposal` row; the runbook's worked example uses a placeholder code.

</code_context>

<specifics>
## Specific Ideas

- "When it breaks, an operator finds out" is the phase's bar: every choice above prefers a loud,
  deduplicated signal over silence — but a legitimately empty watch list is quiet, and a
  data-shape skip the commands already report is not a failure.
- The runner is the single place that knows about cron, email and heartbeat; the four step commands
  stay usable by hand exactly as today and never mail on their own.
- Phase 31's warning stands as a constraint on this phase's own evidence: no notebook cell,
  transcript, or runbook example may quote live settings or `local_settings.py`; `check_unattended`
  prints names and set/unset status only.
- Phase 34 D-20 assumed TOM's `updatestatus` would be on cron; D-03 replaces that assumption with
  the FOMO-owned LCO/SOAR step — the narrowing proof still holds because the receiver fires on the
  same `update_observation_status()` save.

</specifics>

<deferred>
## Deferred Ideas

- AWS/Kubernetes scheduling (`CronJob` + `concurrencyPolicy: Forbid`), the container image
  (`flock` + HTTP client), and the third-party-heartbeat policy question — Phase 31's open items,
  owned by whoever owns that deployment; the runner's design (one process, env-var config,
  service-agnostic heartbeat) is meant to carry over unchanged.
- A runtime log-redaction filter (offered as belt-and-braces, declined for this phase) — revisit
  if a SCHED-10 regression ever slips past the tests.
- Letting the campaign-bound sibling `backfill_lco_observation_records` read the watched list, or
  letting `project_observation_calendar --proposal` take the watched set — not needed for SC 2.
- `check_unattended --send-test-email` — nice-to-have, planner's call.

### Reviewed Todos (not folded)
- `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md` — already delivered
  as Phase 35 D-13 (`allocation_projector.py` computes `sun_event()` only when minting/re-minting);
  closable, no Phase 36 work.
- `2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md` — attribution-UI
  guard; keyword-only match, unrelated to unattended operation.

</deferred>

---

*Phase: 36-Unattended Operation*
*Context gathered: 2026-09-16*
