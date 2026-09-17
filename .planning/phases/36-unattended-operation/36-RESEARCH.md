# Phase 36: Unattended Operation - Research

**Researched:** 2026-09-16
**Domain:** Django management-command orchestration, cron+flock unattended scheduling, Django email/mail testing, healthchecks-style dead-man's-switch heartbeat, credential-safe logging
**Confidence:** HIGH

## Summary

Phase 36 does not introduce a new technology stack — it wires four already-shipped, already-tested
FOMO management commands (a new FOMO-owned status refresh, `project_observation_calendar`,
`backfill_lco_observations`, `reconcile_campaign_runs`) into one runner command that cron invokes
under `flock -n`. Every mechanism the phase needs (cron+flock invocation shape, credential-via-env
convention, the `_notify_staff()` email idiom, healthchecks-style ping conventions) was already
proven against the real interim host in Phase 31's spike (`docs/design/run_identity_and_unattended_invocation_spike.rst`,
`31-DECISION.md`) and is treated as settled, not open, by 36-CONTEXT.md. The main engineering
work is: (1) building the runner and its per-step wrapper (a thin management command that calls
module-level functions, not `call_command()`, because two of the three sweep commands return
`None` from `handle()` and only print their summary/failure counts to stdout — the module
functions are the only structured-data path); (2) extracting `_notify_staff()` into a request-free,
`fail_silently=False` helper; (3) adding a `WatchedProposal` model and restructuring
`backfill_lco_observations` to loop over active rows with per-row failure isolation instead of a
single required `--proposal`; and (4) a `check_unattended` preflight command plus the committed
`deploy/cron/` and `deploy/logrotate/` templates and a new runbook section, none of which exist yet.

The single most load-bearing research finding: `project_observation_calendar.Command.handle()` and
`reconcile_campaign_runs.Command.handle()` both **return `None`** — their `failed:`/`unprojectable`/
`failed_count` values exist only as text printed via `self.stdout.write()`, never as a return value
or exception. `backfill_lco_observations.Command.handle()` **does** return a summary string, but has
no `failed:` counter at all today. This means the runner cannot learn success/failure from
`call_command()`'s return value for any of the three; it must call the underlying module functions
directly (`project_queryset()`, a per-`CampaignRun` loop over `reconcile_run()`, and a new
per-`WatchedProposal` loop this phase adds to `backfill_lco_observations`) to get a structured
result it can act on for D-10/D-11/D-12.

**Primary recommendation:** Build the runner (e.g. `solsys_code/unattended.py`) as a set of
step functions, each calling the sweep module's own function (`update_all_observation_statuses()`
on a fresh facility instance; `observation_projector.project_queryset()`; a new per-proposal
`backfill_lco_observations` module function; `campaign_reconciler.reconcile_run()` looped) rather
than `call_command()`, so every step yields a structured `(failed: bool, summary: str)` the runner's
notification/heartbeat logic can consume without parsing stdout text.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Scheduled invocation (cron + flock) | OS / Host | — | cron and `flock` are host-level facilities outside Django; Phase 31 confirmed both present on the interim host |
| Runner orchestration (step order, failure aggregation, locking) | API / Backend (Django management command) | — | a Django management command process, in-process, no task queue (locked scope) |
| Status refresh / sweep / discovery / reconcile (the four steps) | API / Backend | Database / Storage | each step reads/writes `ObservationRecord`/`CalendarEvent`/`CampaignRun` via the ORM; discovery also makes portal HTTP calls |
| `WatchedProposal` admin editing | API / Backend (Django admin) | Database / Storage | admin-editable list read at sweep time by the discovery step — no redeploy needed (SC 2) |
| Failure notification (email) | API / Backend | — | `send_mail()` via Django's mail backend, triggered from the runner process only |
| Heartbeat ping | API / Backend → external service | — | outbound HTTP `requests` call to a healthchecks-compatible endpoint; the runner is the sole caller |
| Log file | OS / Host | — | stdout+stderr redirect in the crontab line, rotated by logrotate (host-level, not Django `LOGGING`) |
| `check_unattended` preflight | API / Backend (Django management command) | OS / Host (checks lock/log dirs, `flock` binary) | reads settings/env and the filesystem; never writes |

## Standard Stack

### Core

No new third-party packages. Every dependency this phase needs is already installed and already in
use elsewhere in this codebase:

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `requests` | 2.33.1 [VERIFIED: `pip show requests` this session] | Heartbeat ping (`GET <url>/start`, `GET <url>/<exit-code>`) | Already the HTTP client used by `calendar_utils.resolve_placement_block()` and `backfill_lco_observations`'s portal calls — no new dependency to vet |
| `fcntl` (stdlib) | Python 3.11.13 stdlib [VERIFIED: `python -c "import fcntl"` this session] | Per-command non-blocking file lock (`fcntl.flock(fd, fcntl.LOCK_EX \| fcntl.LOCK_NB)`), if the runner also locks inside the process per Claude's Discretion | Standard library; raises `BlockingIOError` (a subclass of `OSError`) when the lock is already held and `LOCK_NB` is set — well-established POSIX semantics [CITED: Python `fcntl` module docs] |
| `django.core.mail.send_mail` | Django (via tomtoolkit 3.0.1) | Failure notification email | Already the sole mail-sending call in this codebase — `campaign_views.py:340` [VERIFIED: solsys_code/campaign_views.py:340, `send_mail(` — Read this session] |
| `django.core.management.base.BaseCommand`/`call_command` | Django | Thin management-command wrappers for the runner and `check_unattended` | Existing convention for every command in `solsys_code/management/commands/` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `tom_observations.facilities.lco.LCOFacility` | tomtoolkit 3.0.1 [VERIFIED: `pip show tomtoolkit`] | Status-refresh step (D-03) | `LCOFacility()` and `SOARFacility()` instances, one each, never shared (mirrors 34 D-10's rule) |
| `django.test.TestCase` + `django.core.mail` (`mail.outbox`) | Django | D-16 credential-leak regression tests | Django's test runner automatically swaps `EMAIL_BACKEND` to `locmem` during `manage.py test` [VERIFIED: `django/test/utils.py:146-147`, `settings.EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"` — Read this session]; `solsys_code/tests/test_campaign_submission.py` already asserts on `mail.outbox` this way [VERIFIED: solsys_code/tests/test_campaign_submission.py — grep this session, `from django.core import mail` at line 12, `mail.outbox` assertions lines 204-241] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cron + `flock -n` | Celery/huey/APScheduler | Explicitly out of scope — Phase 31 settled this against the real host; a task queue buys nothing for a single-server, few-jobs-an-hour deployment (ROADMAP.md "Locked constraints", `31-DECISION.md` line 913-921) |
| Module-function calls for each step | `call_command('project_observation_calendar')` etc. | `call_command()` returns `handle()`'s return value, which is `None` for 2 of 3 sweep commands — no structured failure signal without parsing stdout text; module functions (`project_queryset()`, `reconcile_run()`) return real Python objects |
| One env var per credential (`FOMO_HEARTBEAT_URL`) | A settings dict | Matches the existing `FINK_*` `os.getenv()` convention in `src/fomo/settings.py:314-319` [VERIFIED: src/fomo/settings.py:314-319, `os.getenv('FINK_CREDENTIAL_URL', ...)` etc. — Read this session] and the credential-hygiene rule from Phase 31's spike |

**Installation:** none — no new packages.

**Version verification:** `requests==2.33.1` and `tomtoolkit==3.0.1` confirmed installed via `pip show` this session; no version bump needed for this phase's work.

## Package Legitimacy Audit

No new external packages are introduced by this phase. `requests` and Django/tomtoolkit are
pre-existing, already-vetted dependencies of this project; `fcntl` is Python stdlib. The Package
Legitimacy Gate is not applicable — no `npm view`/`pip index versions` verification is needed
because nothing new is being installed.

**Packages removed due to [SLOP] verdict:** none — nothing new was evaluated.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
cron (*/15 * * * *)
   |
   v
/usr/bin/flock -n /var/lock/fomo/run_unattended.lock \
   /path/venv/bin/python /path/manage.py run_unattended  >> /var/log/fomo/unattended.log 2>&1
   |
   |-- lock held? -----> [skip, log nothing new -- a permanently-contended lock is caught by
   |                       the crontab template's own skip line + the heartbeat's grace period]
   |
   v (lock acquired)
runner process (management command)
   |
   |-- ping <FOMO_HEARTBEAT_URL>/start  (best-effort; failure logged, never fatal)
   |
   |-- Step 1: status refresh (D-03)
   |     LCOFacility().update_all_observation_statuses()   -> failed_records: list[(obs_id, str(e))]
   |     SOARFacility().update_all_observation_statuses()  -> failed_records: list[(obs_id, str(e))]
   |     each record .save() fires the post_save receiver (Phase 34) -> narrows its CalendarEvent
   |     non-empty failed_records => step failed; log observation_id + type(exc).__name__ only
   |
   |-- Step 2: observation projector sweep (backstop, D-08's one-time observed-site lookup)
   |     observation_projector.project_queryset(ObservationRecord.objects.all(), dry_run=False,
   |         pre_fields_hook=<site-lookup hook, mirrors project_observation_calendar.py's own hook>)
   |     -> result['counters'] per facility, result['rows'] (row['action']=='unprojectable' => failed)
   |
   |-- Step 3: discovery backfill (DISCOVER-01)
   |     for row in WatchedProposal.objects.filter(is_active=True):
   |         try: sweep this proposal (backfill_lco_observations' per-request loop, extracted to
   |              a per-proposal function) -> summary string, exception class on failure
   |         except Exception as exc: row.last_run_summary = f'failed: {type(exc).__name__}'; continue
   |         row.last_run_at = now(); row.last_run_summary = summary; row.save()
   |     0 active rows => one INFO line, exit 0 (D-08, not a failure)
   |
   |-- Step 4: campaign reconciler sweep (backstop)
   |     for run in CampaignRun.objects.all():
   |         try: campaign_reconciler.reconcile_run(run, dry_run=False) -> ReconcileResult
   |         except Exception as exc: failed_count += 1  (mirrors reconcile_campaign_runs.py's own loop)
   |
   |-- aggregate: any step failed? -> exit_code = 1 if any failed else 0
   |
   |-- notify_staff() if exit_code != 0 and not already-notified-for-this-failure-set (D-11
   |     suppression state) -- OR notify "recovered" if exit_code == 0 and previous tick failed
   |
   |-- ping <FOMO_HEARTBEAT_URL>/<exit_code>  (best-effort; failure logged, never fatal)
   |
   v
process exits with exit_code
```

### Recommended Project Structure

```
solsys_code/
├── unattended.py                          # step functions, notify_staff(), heartbeat ping, suppression state
├── management/commands/
│   ├── run_unattended.py                  # thin wrapper: calls unattended.run_tick(), sets exit code
│   ├── check_unattended.py                # preflight: flock present, dirs writable, email/staff/heartbeat/watched-proposal checks
│   └── backfill_lco_observations.py       # D-07..D-09: loop over WatchedProposal when --proposal omitted
├── models.py                              # + WatchedProposal
├── admin.py                               # + WatchedProposalAdmin
└── migrations/0022_watchedproposal.py     # next migration number after 0021 [VERIFIED: solsys_code/migrations/ — `ls` this session shows 0021_alter_calendareventmeta_is_verified_and_more.py as latest]
deploy/
├── cron/fomo.crontab.example              # new top-level directory -- none exists today [VERIFIED: `ls deploy/` this session -> "No such file or directory"]
└── logrotate/fomo.example
```

### Pattern 1: Per-step failure isolation with a structured result, not `call_command()`

**What:** Each runner step calls the underlying module function/class directly and interprets its
return value, rather than invoking `call_command('the_command_name')` and hoping for a non-zero
return or a raised exception.

**When to use:** Always, for this phase's four steps — verified this session that two of the three
existing sweep commands never surface failure through `handle()`'s return value.

**Example (verified from the actual shipped commands, this session):**

```python
# project_observation_calendar.py's Command.handle() -- verified return is always None; the
# `failed: N` count only ever reaches self.stdout.write(), never a return value or raise.
# Source: solsys_code/management/commands/project_observation_calendar.py:164-224 (Read this session)
def handle(self, *args, **options):
    ...
    result = project_queryset(records, dry_run=dry_run, pre_fields_hook=...)
    failed = 0
    for row in result['rows']:
        if row['action'] == 'unprojectable':
            failed += 1
    ...
    self.stdout.write(f'{prefix} failed: {failed} | {summary}')
    return  # <-- always None; call_command('project_observation_calendar') tells the caller nothing

# The runner should instead do what the command itself does, directly:
from solsys_code.observation_projector import project_queryset
result = project_queryset(ObservationRecord.objects.all(), dry_run=False, pre_fields_hook=hook)
failed = sum(1 for row in result['rows'] if row['action'] == 'unprojectable')
```

```python
# reconcile_campaign_runs.py's Command.handle() -- same shape: loops CampaignRun.objects.all(),
# calls reconcile_run(run, dry_run=...) per run, catches exceptions locally, and only ever prints
# the aggregate -- handle() itself returns None.
# Source: solsys_code/management/commands/reconcile_campaign_runs.py:35-177 (Read this session)
from solsys_code.campaign_reconciler import reconcile_run
failed_count = 0
for run in CampaignRun.objects.all().select_related('site', 'campaign').order_by('pk'):
    try:
        result = reconcile_run(run, dry_run=False)
    except Exception as exc:
        failed_count += 1
        continue
    ...
```

```python
# backfill_lco_observations.py's Command.handle() -- the one exception: it DOES return a summary
# string. But --proposal is required=True today (a single proposal per call) and there is no
# failed: counter at all -- D-07..D-09 require restructuring this into a per-WatchedProposal loop
# with its own try/except, since a portal error today propagates unhandled out of handle().
# Source: solsys_code/management/commands/backfill_lco_observations.py:445-451,478-710 (Read this session)
```

### Pattern 2: Credential-free logging via caught-exception classification

**What:** An exception caught from a `requests`/facility/portal call is logged and reported by
`type(exc).__name__` only, never `str(exc)`; an exception FOMO raises itself (`CommandError`,
`ValueError` from validation) may carry its own message.

**When to use:** Every except clause on the unattended path (status refresh, discovery, the sweep,
mail send, heartbeat ping).

**Example (two real, contrasting patterns already in this codebase, both read this session):**

```python
# Network/portal exception -- class name only, never the exception's value.
# Source: solsys_code/calendar_utils.py:317-329 (resolve_placement_block())
try:
    response = make_request('GET', ..., timeout=_API_TIMEOUT_SECONDS)
    blocks = response.json()
except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, ValueError):
    return None  # no str(exc) anywhere -- SYNC-09/D-11's rule

# Local/FOMO-raised exception path in the projector's own generic catch -- DOES log str(exc),
# because everything reaching this except clause is FOMO's own pipeline code (facility_for(),
# event_fields_for(), insert_or_create_calendar_event(), write_event_meta()), not a raw network
# exception whose str() could embed a response body.
# Source: solsys_code/observation_projector.py:364-372
except Exception as exc:  # noqa: BLE001
    logger.warning('unprojectable observation_id=%r: %s: %s', record.observation_id, type(exc).__name__, exc)
    return 'unprojectable', type(exc).__name__
```

The runner's per-step wrapper must replicate `resolve_placement_block()`'s discipline (class name
only) for the status-refresh step, since `update_all_observation_statuses()` returns
`[(observation_id, str(e))]` — that `str(e)` is exactly the value D-03 says must never reach a log
line: `[VERIFIED: tom_observations/facility.py:567-579]`:

```python
def update_all_observation_statuses(self, target=None):
    from tom_observations.models import ObservationRecord
    failed_records = []
    records = ObservationRecord.objects.filter(facility=self.name)
    if target:
        records = records.filter(target=target)
    records = records.exclude(status__in=self.get_terminal_observing_states())
    for record in records:
        try:
            self.update_observation_status(record.observation_id)
        except Exception as e:
            failed_records.append((record.observation_id, str(e)))
    return failed_records
```

The runner must discard the `str(e)` half of each tuple and log/report only `observation_id` plus
`type(exc).__name__` — but `update_all_observation_statuses()` only gives back `str(e)`, not the
exception object itself, so the class name is not recoverable from this return value alone. **Open
question for the planner:** either re-derive the class name by calling
`facility.update_observation_status(observation_id)` directly per failed id inside a fresh
try/except (so the exception object is in hand), or accept that the only thing available from the
stock TOM method is a string and treat every entry in `failed_records` as an opaque count (no class
name at all) rather than trying to parse one out of `str(e)`.

### Pattern 3: `_notify_staff()` extraction — the exact request-dependency to remove

**What:** The only mail sender in the codebase builds its link with `self.request.build_absolute_uri(...)`
and calls `send_mail(..., fail_silently=True)`. Both must change for the unattended path.

**Example (verbatim, read this session):**

```python
# Source: solsys_code/campaign_views.py:327-346
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
        fail_silently=True,  # Pitfall 6: a mail outage must never break the submission
    )
```

The extracted helper must (a) take an explicit base URL (e.g. `settings.FOMO_BASE_URL`, a new
setting — neither `FOMO_BASE_URL` nor `FOMO_HEARTBEAT_URL` exists anywhere in this codebase today
`[VERIFIED: grep -n "FOMO_BASE_URL\|FOMO_HEARTBEAT" src/fomo/settings.py solsys_code/*.py — no output this session]`),
never `self.request`; (b) keep the identical recipient rule
(`User.objects.filter(is_staff=True).exclude(email='')`); and (c) use `fail_silently=False` on the
unattended-path call with the raised exception caught by the runner and logged as `type(exc).__name__`
only (D-17) — while the existing submission-notice call site keeps `fail_silently=True` semantics,
per D-11.

### Pattern 4: Status-refresh step — instantiate facilities fresh, never share

**What:** `update_all_observation_statuses()` is a `BaseRoboticObservationFacility` instance method.
The stock TOM `updatestatus` command loops `facility.get_service_classes()` (all registered
facilities, including the Gemini/ESO stubs) and always exits 0 regardless of `failed_records`.

**Example (verbatim, read this session):**

```python
# Source: tom_observations/management/commands/updatestatus.py:28-56
def handle(self, *args, **options):
    ...
    failed_records = {}
    for facility_name in facility.get_service_classes():
        instance = facility.get_service_class(facility_name)()
        instance.set_user(user)
        failed_records[facility_name] = instance.update_all_observation_statuses(target=target)
    success = True
    for facility_name, errors in failed_records.items():
        if len(errors) > 0:
            success = False
            break
    if success:
        return 'Update completed successfully'
    else:
        return 'Update completed with errors: {0}'.format(str(failed_records))
```

Note: `handle()` **does** distinguish success/failure internally (the `success` flag), but never
raises and never calls `sys.exit()` — a non-zero exit never happens on this path even when
`failed_records` is non-empty; `handle()`'s return value is just a string Django's command runner
prints or discards. This confirms 36-CONTEXT.md D-03's claim that stock `updatestatus` "always exits
0" `[VERIFIED: tom_observations/management/commands/updatestatus.py:28-56]`. D-03's FOMO-owned
replacement must call `LCOFacility().update_all_observation_statuses()` and
`SOARFacility().update_all_observation_statuses()` directly (each a **fresh** instance, matching
Phase 34 D-10's "facility instance per facility, never shared" rule already established for the
projector's own `facility_for()`) and treat a non-empty returned list as a step failure explicitly —
the runner supplies the pass/fail judgement the stock command withholds.

Also confirmed: `update_all_observation_statuses()` already `records.exclude(status__in=self.get_terminal_observing_states())` before iterating `[VERIFIED: tom_observations/facility.py:570-573]` — so it never re-checks a terminal-state record, matching D-03's claim that it "already excludes terminal states."

### Anti-Patterns to Avoid

- **Re-invoking `manage.py <command>` as a subprocess per step:** D-02 explicitly forbids this — each
  step must call the shipped command's logic via `call_command()` or the module's own functions, in
  the same process, so the runner's `flock` and heartbeat wrap all four steps as one unit.
- **Trusting `call_command()`'s return value for pass/fail:** demonstrated above — two of the three
  commands return `None` regardless of outcome.
- **Parsing the `failed: N` text out of captured stdout:** technically possible (`call_command(..., stdout=StringIO())`)
  but fragile — a wording change in any of the three commands' summary lines silently breaks the
  runner's failure detection. Prefer the module-function path (Pattern 1).
- **Logging `str(e)` from `update_all_observation_statuses()`'s returned tuples:** see Pattern 2 —
  this is exactly the credential-leak vector SCHED-10 exists to close; `LCOFacility`/`SOARFacility`
  portal exceptions can embed request/response content in their `str()` form.
- **A shared `LCOFacility()`/`SOARFacility()` instance across steps:** Phase 34 D-10 already
  established one instance per facility per call; the status-refresh step and the discovery step
  each need their own.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-record LCO/SOAR status polling | A custom portal-status loop | `LCOFacility().update_all_observation_statuses()` / `SOARFacility().update_all_observation_statuses()` [VERIFIED: tom_observations/facility.py:567-579] | Already handles per-facility exclusion of terminal states and per-record try/except; only the pass/fail judgement and credential-safe logging are new |
| Event projection for observation-backed nights | A second projector | `observation_projector.project_queryset()` (Phase 34) | Already the backstop sweep; the runner's step 2 is exactly `project_observation_calendar`'s own logic, called directly |
| Campaign-run reconciliation | A second reconciler loop | `campaign_reconciler.reconcile_run()` looped over `CampaignRun.objects.all()` (Phase 33/35) | Already per-run failure-isolated; the runner's step 4 mirrors `reconcile_campaign_runs.py`'s own loop verbatim |
| Non-blocking process-level locking | A custom PID-file scheme | `flock -n` (cron line) plus, optionally, `fcntl.flock(fd, fcntl.LOCK_EX \| fcntl.LOCK_NB)` inside the runner for the one-lock-per-command discretion note | `flock` is confirmed present on the interim host (util-linux 2.37.4) `[VERIFIED: 31-DECISION.md:385, "flock from util-linux 2.37.4"]`; a hand-rolled PID file cannot express the same atomic non-blocking semantics without races |
| A dead-man's-switch | A custom "did the last tick run" table + alert | A healthchecks-compatible external ping service (`/start`, `/<exit-code>`) | The whole point of the second visibility layer (D-12) is that it lives **outside** the process — anything inside the FOMO process cannot observe its own non-execution; `31-DECISION.md:1006-1014` already confirmed egress reachability from the interim host |

**Key insight:** every piece of domain logic this phase needs (status refresh judgement,
projection, reconciliation, mail-sending idiom) already exists in this codebase from Phases 33-35 and
the `_notify_staff()` submission-notice flow — Phase 36's job is orchestration, restructuring one
command's argument handling, and adding the two new small models/commands (`WatchedProposal`,
`check_unattended`), not new domain algorithms.

## Runtime State Inventory

> Rename/refactor/migration inventory — Phase 36 changes an existing command's default argument
> behavior (`backfill_lco_observations`'s `--proposal` becomes optional) and adds host-level state
> (crontab entry, lock files, log file) that does not exist in git today.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — `WatchedProposal` is a brand-new model with no prior rows to migrate; `backfill_lco_observations`'s existing behavior (per-invocation `--proposal`) stays available as an override (D-07), so no existing automation that already passes `--proposal` breaks | None |
| Live service config | The real interim host's crontab today has **0 of 3** existing FOMO-unrelated cron entries guarded by `flock` `[VERIFIED: 31-DECISION.md:958-964]` — this phase does not touch those entries; it adds a new, separate, `flock`-guarded line | None (informational — confirms the host precedent this phase's own line follows) |
| OS-registered state | No cron entry, lock file, or log file for this phase's runner exists on any host yet — `deploy/cron/`, `deploy/logrotate/` and `/var/lock/fomo/`, `/var/log/fomo/` are new (documented, not committed, per 36-CONTEXT.md "code_context" section) `[VERIFIED: ls deploy/ this session -> "No such file or directory"]` | Operator installs via `check_unattended`'s printed cron line (D-05) — no migration of existing state |
| Secrets/env vars | `FOMO_HEARTBEAT_URL` and (if not a Django setting) a base-URL env var are brand new — neither exists in `src/fomo/settings.py` or anywhere in `solsys_code/` today `[VERIFIED: grep across src/fomo/settings.py and solsys_code/*.py this session — no matches]` | New env vars only; no existing key renamed |
| Build artifacts | None — no package renamed or restructured | None |

**Nothing found in category "Stored data" and "Build artifacts":** confirmed by direct search this session; no migration risk from this phase's changes beyond the additive `WatchedProposal` migration itself.

## Common Pitfalls

### Pitfall 1: Assuming `call_command()`'s return value signals success

**What goes wrong:** The runner calls `call_command('project_observation_calendar')` or
`call_command('reconcile_campaign_runs')` and checks the return value for failure — it is always
`None`, so every tick silently reports success regardless of how many records were unprojectable or
runs failed.
**Why it happens:** `call_command()` genuinely does return `handle()`'s return value for commands
that return one (`backfill_lco_observations` does); it is easy to assume the pattern is uniform
across all commands in this codebase without checking each one.
**How to avoid:** Call the module-level functions directly for the two silent commands
(`project_queryset()`, a `reconcile_run()` loop), as Pattern 1 above shows. Verified from source this
session, not assumed.
**Warning signs:** A test that mocks a step to raise and expects the runner's overall exit code to be
non-zero, but the mock never gets exercised because the step function used `call_command()` and
swallowed the outcome.

### Pitfall 2: `update_all_observation_statuses()`'s failure list only carries `str(e)`, not the exception object

**What goes wrong:** The runner tries to log `type(exc).__name__` for a status-refresh failure, but
the only thing `update_all_observation_statuses()` returns is `[(observation_id, str(e))]` — the
exception object itself is gone by the time the runner sees the list.
**Why it happens:** This is a stock TOM Toolkit method (`tom_observations/facility.py:567-579`), not
FOMO's own code — its return shape was fixed before this phase's credential-hygiene requirement
existed.
**How to avoid:** Documented as an open question above (Pattern 4) — the planner must choose between
re-deriving the exception object (calling `update_observation_status()` directly per failed id in a
fresh try/except) or accepting an opaque failure count with no class name for this one step.
**Warning signs:** A D-16 regression test that seeds a fake credential into a mocked portal error and
asserts the class name appears in the log — if the runner is passing `str(e)` through unfiltered, the
credential leaks and the test should catch it; if the runner drops the message entirely without a
class name, the log becomes less useful than the other three steps' logging.

### Pitfall 3: `backfill_lco_observations` has no per-proposal failure isolation today

**What goes wrong:** A portal error on one `WatchedProposal` (e.g. a revoked/expired proposal code)
aborts the whole discovery step, so every other active proposal is skipped for that tick.
**Why it happens:** `_iter_request_groups()` calls `make_request()` directly with no try/except
`[VERIFIED: solsys_code/management/commands/backfill_lco_observations.py:74-78]` — an HTTP or auth
failure propagates straight out of `handle()` today, since the current design only ever processes one
proposal per invocation and a hard failure is acceptable there.
**How to avoid:** D-09 requires wrapping each watched proposal's sweep in its own try/except inside
the new per-proposal loop, recording the failure on that row's `last_run_summary` and continuing to
the next proposal — this is new code, not a refactor of existing error handling (there is none to
refactor).
**Warning signs:** A test that seeds two `WatchedProposal` rows, makes the first one's portal call
raise, and asserts the second one's records still got created/updated.

### Pitfall 4: `docs/notebooks.rst` does not list `backfill_lco_observations_demo.ipynb`

**What goes wrong:** The paired-docs notebook this phase must update
(`docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`) exists on disk
`[VERIFIED: ls docs/notebooks/pre_executed/ this session — backfill_lco_observations_demo.ipynb present]`
but is **not** wired into the Sphinx toctree at `docs/notebooks.rst`
`[VERIFIED: grep -n "backfill_lco_observations_demo" docs/notebooks.rst this session — no match; the
toctree lists telescope_runs_demo, load_telescope_runs_demo, project_observation_calendar_demo,
sync_gemini_observation_calendar_demo, import_campaign_csv_demo, reconcile_campaign_runs_demo,
campaign_lifecycle_demo — seven entries, backfill_lco_observations_demo absent]` — a pre-existing gap,
not something this phase introduces, but visible the moment this phase touches that notebook again.
**Why it happens:** the notebook was added in a prior phase without a toctree entry.
**How to avoid:** the planner should decide whether closing this gap (adding one `toctree` line) rides
along with this phase's already-scoped notebook update, since the CLAUDE.md paired-docs rule requires
the notebook to be part of the deliverable regardless.
**Warning signs:** a Sphinx build that succeeds even though the notebook page is unreachable from the
docs nav — `sphinx-build` does not fail on an orphaned notebook file that is never referenced by a
toctree unless `nitpicky`/orphan-checking is enabled.

### Pitfall 5: `flock -n` skip is silent unless the runner (or the crontab line itself) logs it

**What goes wrong:** A tick that finds the lock held exits non-zero with no log line — indistinguishable
from a healthy no-op unless something logs the skip.
**Why it happens:** `flock -n`'s own non-zero exit code, unobserved, is a silent failure mode — this
was explicitly flagged as a real, not hypothetical, risk in Phase 31's spike, since the real interim
host crontab already has 0 of 3 unrelated Django commands guarded at all
`[VERIFIED: 31-DECISION.md:950-964]`.
**How to avoid:** D-01 requires the crontab template to include a skip-visible shape (an `|| echo`-style
fallback) and D-12's heartbeat is the structural backstop — a contended-forever lock means no
`/<exit-code>` ping ever arrives, so the heartbeat's grace period (D-12 recommends ~20 minutes for a
15-minute schedule) catches it even if the shell-level skip line is somehow lost.
**Warning signs:** A tick that "ran" every time in the log but the heartbeat dashboard shows the last
successful ping days old.

## Code Examples

### The exact cron line shape (proven against the real interim host)

```
# Source: 31-DECISION.md:926-929 (verified against the real interim host, util-linux 2.37.4)
/usr/bin/flock -n /var/lock/fomo/<command-name>.lock \
    /path/to/venv/bin/python /path/to/checkout/manage.py <command-name> [args]
```

D-01 applies this shape with `<command-name>` = the runner's own name (e.g. `run_unattended`) and
appends a stdout+stderr append-redirect to the log file (D-18):

```
*/15 * * * * /usr/bin/flock -n /var/lock/fomo/run_unattended.lock \
    /path/to/venv/bin/python /path/to/checkout/manage.py run_unattended \
    >> /var/log/fomo/unattended.log 2>&1
```

### Healthchecks-style ping shape

```python
# GET <url>/start before the first step
requests.get(f'{heartbeat_url}/start', timeout=<short timeout>)
# ... run all four steps ...
# GET <url>/<exit_code> after the last step (0 on success, non-zero exit code otherwise)
requests.get(f'{heartbeat_url}/{exit_code}', timeout=<short timeout>)
```

`/start` and `/<exit-code>` (0-255; 0 = success, any other value = failure) is the documented
healthchecks.io ping API convention [CITED: healthchecks.io/docs/http_api/, WebSearch this session —
"You can append /start, /fail or /<exitcode> to the base ping URL... Healthchecks.io interprets 0 as
a success and all other values as a failure"]. Phase 31's own probe confirmed outbound egress to
`hc-ping.com` from the real interim host returns HTTP 301 `[VERIFIED: 31-DECISION.md:1010, "Task 1's
transcript recorded HTTP status 301 from this host to hc-ping.com"]` — technical reachability only;
policy/compliance acceptability is explicitly out of scope for this phase (deferred item, carried
from Phase 31).

### Mail testing pattern (`mail.outbox`)

```python
# Source: solsys_code/tests/test_campaign_submission.py (grep-confirmed lines 12, 204-241, this session)
from django.core import mail

class SomeUnattendedTest(TestCase):
    def test_failure_email_sent(self):
        # ... trigger a failing tick ...
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn(staff_with_email.email, mail.outbox[0].to)
```

Django's test runner (`django.test.utils.setup_test_environment()`) automatically swaps
`EMAIL_BACKEND` to `django.core.mail.backends.locmem.EmailBackend` for the duration of `manage.py
test`, regardless of the project's configured `EMAIL_BACKEND` (console, by default, per
`src/fomo/settings.py:403` [VERIFIED: src/fomo/settings.py:401-403]) `[VERIFIED:
django/test/utils.py:146-147, "settings.EMAIL_BACKEND = \"django.core.mail.backends.locmem.EmailBackend\""]`
— no `@override_settings(EMAIL_BACKEND=...)` is needed for D-16's mail-outbox assertions, only for
`check_unattended`'s own test of "is EMAIL_BACKEND the console backend" (which must explicitly set
`EMAIL_BACKEND` back to console or a real backend to exercise that branch, since the test runner's
locmem override would otherwise mask it).

### Existing mock pattern for portal calls (for D-16/D-09 tests)

```python
# Source: solsys_code/tests/test_backfill_lco_observations.py (grep-confirmed, this session)
from unittest.mock import MagicMock, patch

def _page_response(results, next_url=None):
    response = MagicMock()
    response.json.return_value = {'count': len(results), 'next': next_url, 'previous': None, 'results': results}
    return response

class SomeTest(TestCase):
    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_something(self, mock_make_request):
        mock_make_request.return_value = _page_response([...])
        ...
```

Every existing test in `test_backfill_lco_observations.py` patches `make_request` at the point it is
imported into the command module (`solsys_code.management.commands.backfill_lco_observations.make_request`),
not at its source (`tom_observations.facilities.ocs.make_request`) — the same convention applies to
any new per-proposal loop the planner extracts, and to a forced portal-error test for D-09's
per-proposal isolation.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `backfill_lco_observations --proposal <code>` (one required, per-invocation code) | `WatchedProposal` admin rows, bare invocation sweeps every active row | This phase (D-07) | `--proposal` becomes an optional override for a one-off manual run; existing 30 tests and notebook `--proposal` cells keep working `[per 36-CONTEXT.md D-07]` |
| Stock TOM `updatestatus` (always exits 0, prints `str(e)`) | FOMO-owned status-refresh step, non-empty failure list = step failure, class-name-only logging | This phase (D-03) | Closes a real credential-leak vector (`str(e)` from a portal exception) and gives the runner a real pass/fail signal the stock command withholds |
| No scheduled invocation at all (manual `manage.py <command>` runs) | cron + `flock -n`, one runner, `*/15 * * * *` | This phase (D-01/D-04) | SC 1/2/3 — the milestone's unattended-operation bar |

**Deprecated/outdated:** none — this phase adds capability rather than replacing a prior mechanism;
the three sweep commands and `_notify_staff()`'s submission-notice call site are unchanged in their
existing, direct-invocation form.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The recommended `check_unattended` design (a `BaseCommand` that probes `flock` via `shutil.which('flock')` or similar, checks `EMAIL_BACKEND`, and counts active `WatchedProposal` rows) is a standard, idiomatic implementation shape — not verified against any external "preflight command" convention, since none exists in this codebase yet | Recommended Project Structure | Low — the planner has full discretion here per 36-CONTEXT.md's Claude's Discretion section; no external library or pattern is being assumed incorrectly, only an implementation shape |
| A2 | A daily-reminder interval and the suppression-state store's exact shape (state file vs. tiny model) are unresolved by design — 36-CONTEXT.md explicitly leaves both to the planner | Pattern 3 / Common Pitfalls | Low — explicitly flagged as Claude's Discretion in CONTEXT.md, not a research gap |
| A3 | `fcntl.flock`'s `BlockingIOError` behavior under `LOCK_NB` is standard CPython/POSIX behavior, cited from training knowledge and the Python stdlib documentation rather than executed in this session | Standard Stack table | Low — this is well-established, widely-documented stdlib behavior; a quick interactive check (`python -c "import fcntl"`) confirmed the module and its two flag constants are present this session, but the raise-on-contention behavior itself was not exercised live |

**If this table is empty:** N/A — three low-risk assumptions logged above; nothing here touches a
compliance, retention, or security-standard claim, and nothing contradicts a verified finding.

## Open Questions

1. **How does the runner recover a class name for a status-refresh failure, given `update_all_observation_statuses()` only returns `str(e)`?**
   - What we know: the stock method's return shape is fixed (`[VERIFIED: tom_observations/facility.py:567-579]`) and cannot be changed without patching a third-party library.
   - What's unclear: whether the planner wants the runner to re-derive the exception by calling `update_observation_status()` per failed id directly (extra portal calls, real exception object) or accept an opaque per-record failure count with no class name for this one step only.
   - Recommendation: re-derive per failed id — the extra calls are bounded by `len(failed_records)`, which is already small (excludes terminal-state records), and it keeps this step's logging discipline consistent with the other three steps (class-name-only, never the message).

2. **Where does `WatchedProposal` live relative to `CampaignRun` in `models.py`, and does `backfill_lco_observations`'s existing single-proposal code path get restructured in place or split into a new shared function the runner also calls?**
   - What we know: `models.py` has no existing "configuration list" model to pattern-match against; `CampaignRunAdmin` (`solsys_code/admin.py:142-176`) is the closest existing admin-registration example (`list_display`, `list_filter`, no `list_editable` example exists yet in this codebase, though it is standard Django).
   - What's unclear: exact placement (near `CampaignRun` vs. at the end of the file) and whether the per-proposal sweep becomes a new top-level function in `backfill_lco_observations.py` that both the bare-invocation loop and the `--proposal` override call, or whether the existing `handle()` body is inlined into a loop.
   - Recommendation: extract the existing per-proposal request-group loop (currently the bulk of `handle()`, lines ~526-677) into a function taking `(facility, proposal, target_list_override, user)` and returning `(summary: dict, exception: Exception | None)`, called once per watched row and once for a `--proposal` override — this is a refactor of existing logic, not new logic, and keeps the 30 existing tests passing since the wire format each request-group produces is unchanged.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `flock` | cron line overlap prevention (D-01) | ✓ (on the real interim host — confirmed by Phase 31, not re-probed this session; the local dev sandbox this research ran in has no cron/production host to probe) | util-linux 2.37.4 `[VERIFIED: 31-DECISION.md:385]` | — |
| Outbound HTTPS to a healthchecks-compatible service | Heartbeat (D-12) | ✓ (confirmed by Phase 31 against the real interim host — HTTP 301 from `hc-ping.com`; not re-probed this session) | — | If unset, the runner logs one INFO line per tick and skips pinging (D-12); `check_unattended` reports the layer as off, a warning not a failure |
| A real (non-console) `EMAIL_BACKEND` + at least one staff user with an email | Failure notification (D-11/D-13) | Not configured on this dev sandbox (`EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'` by default, `[VERIFIED: src/fomo/settings.py:403]`) | — | `check_unattended` fails hard on this (D-13) — no fallback; an operator must set a real backend in `local_settings.py` before the notification layer works in production |
| `requests` | Heartbeat ping | ✓ | 2.33.1 `[VERIFIED: pip show requests, this session]` | — |

**Missing dependencies with no fallback:** a real `EMAIL_BACKEND` and at least one staff user with an
email — `check_unattended` must fail hard on this per D-13; there is no fallback notification channel
for an unattended run (email + heartbeat is the whole alerting surface, by explicit scope decision).

**Missing dependencies with fallback:** the heartbeat URL — if `FOMO_HEARTBEAT_URL` is unset, the
runner degrades to email-only visibility (still catches most failure modes, just not a dead
scheduler) and `check_unattended` reports it as a warning, not a hard failure.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django's built-in test runner (`django.test.TestCase`), via `python manage.py test` — the only functioning suite per `CLAUDE.md`'s "Testing" section |
| Config file | none — no `pytest.ini`/`setup.cfg` test config governs `solsys_code/tests/`; `pyproject.toml`'s `testpaths` governs the unrelated, legacy `tests/`/`src`/`docs` pytest suite only |
| Quick run command | `python manage.py test solsys_code.tests.test_unattended` (new file; substitute the actual TestCase name for a single test, e.g. `python manage.py test solsys_code.tests.test_unattended.TestRunUnattended.test_step_failure_triggers_email -v 2`) |
| Full suite command | `python manage.py test solsys_code` (excludes `solsys_code.tests.test_views.TestEphemeris`, which segfaults in native ASSIST per the project's own known test-suite gotcha, and any test that would trigger the `~1.6 GB` SPICE download by importing `solsys_code.views`/`solsys_code.ephem_utils`) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCHED-08 | Runner runs the fixed 4-step sequence, one exit code, step-failure isolation (D-02) | unit | `python manage.py test solsys_code.tests.test_unattended` | ❌ Wave 0 |
| SCHED-08 | Two invocations never overlap (`flock -n` behavior, if locking is tested at the Python level via `fcntl`) | unit | `python manage.py test solsys_code.tests.test_unattended.TestLocking` | ❌ Wave 0 |
| SCHED-08 | `check_unattended` reports every prerequisite and prints the exact cron line | unit | `python manage.py test solsys_code.tests.test_check_unattended` | ❌ Wave 0 |
| SCHED-09 | Failed step triggers exactly one email, suppressed on repeat, "cleared" email on recovery | unit | `python manage.py test solsys_code.tests.test_unattended.TestNotification` (asserts on `mail.outbox`) | ❌ Wave 0 |
| SCHED-09 | Heartbeat pings `/start` then `/<exit-code>`; never fails the tick on ping failure | unit | `python manage.py test solsys_code.tests.test_unattended.TestHeartbeat` (mocks `requests.get`) | ❌ Wave 0 |
| SCHED-10 | No credential value in any log line/email/stdout across every forced failure path | unit (regression) | `python manage.py test solsys_code.tests.test_unattended.TestCredentialHygiene` (seeds a fake API key/heartbeat URL, asserts absence from captured logs, `mail.outbox`, stdout/stderr) | ❌ Wave 0 |
| DISCOVER-01 | `WatchedProposal` admin editing, `is_active` filter, `list_editable` | unit | `python manage.py test solsys_code.tests.test_watched_proposal` | ❌ Wave 0 |
| DISCOVER-01 | Bare `backfill_lco_observations` sweeps every active row; per-row failure isolation; `last_run_summary`/`last_run_at` written | unit (extends existing suite) | `python manage.py test solsys_code.tests.test_backfill_lco_observations` | ✓ (extend existing 1018-line file) |
| DISCOVER-01 | Empty watched list = quiet no-op, exit 0, one INFO line | unit | `python manage.py test solsys_code.tests.test_backfill_lco_observations.TestEmptyWatchedList` | ❌ Wave 0 (new test class in existing file) |
| — | `--proposal` override still works, unwatched proposal accepted | unit (regression) | `python manage.py test solsys_code.tests.test_backfill_lco_observations` | ✓ (existing 30 tests must keep passing) |

### Sampling Rate

- **Per task commit:** `python manage.py test solsys_code.tests.test_unattended` (or the specific new/changed test module)
- **Per wave merge:** `python manage.py test solsys_code` (excluding the known-segfaulting `test_views.TestEphemeris`)
- **Phase gate:** Full suite green before `/gsd-verify-work`; plus a live `check_unattended` run against the real developer database (no portal/mail/heartbeat mocking needed for that command specifically, since it only inspects settings/filesystem/DB state) as a manual UAT step

### Wave 0 Gaps

- [ ] `solsys_code/tests/test_unattended.py` — covers SCHED-08, SCHED-09, SCHED-10 (runner orchestration, notification, heartbeat, credential hygiene)
- [ ] `solsys_code/tests/test_check_unattended.py` — covers SCHED-08's SC 5 (preflight command)
- [ ] `solsys_code/tests/test_watched_proposal.py` — covers DISCOVER-01's model/admin surface
- [ ] Extension of `solsys_code/tests/test_backfill_lco_observations.py` — covers DISCOVER-01's D-07..D-09 (bare-invocation loop, per-proposal isolation, empty-list no-op)
- [ ] Extension of `solsys_code/tests/test_admin.py` (1296 lines, existing) — covers `WatchedProposalAdmin`'s `list_editable`/`list_filter`
- [ ] No new test framework install needed — `python manage.py test` already covers everything this phase needs

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | no | The runner runs as a trusted local process (cron), not a request-authenticated actor; no new login surface |
| V3 Session Management | no | No web session involved on the unattended path |
| V4 Access Control | partial | `WatchedProposal` admin editing is gated by Django admin's existing `is_staff`/`is_superuser` permission model — no new access-control surface, reuses Django admin's standard control |
| V5 Input Validation | yes | `WatchedProposal.proposal_code` uniqueness enforced at the model/DB level (`unique=True`); the discovery step's per-proposal error isolation (D-09) treats a malformed/nonexistent proposal code as a per-row failure, never a crash |
| V6 Cryptography | no | No cryptographic operation introduced — credentials are read via `os.getenv()` and passed through to existing, already-authenticated `requests`/`send_mail()` calls; no new crypto primitive is hand-rolled |
| V7 Error Handling and Logging | yes | This is the phase's central security control (SCHED-10): every except clause on the unattended path must log `type(exc).__name__` only for a network/portal/mail exception, never `str(exc)` — see Pattern 2 above |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Credential leakage via exception `str()` (an LCO/SOAR portal auth error, or `ImproperCredentialsException`, embedding request/response content) | Information Disclosure | Class-name-only exception logging (D-17), proven pattern already in `resolve_placement_block()` `[VERIFIED: solsys_code/calendar_utils.py:292-329]` |
| Credential leakage via cron line / process argument vector (`ps aux` visible to any local user) | Information Disclosure | Environment variables only, never a CLI argument or embedded literal in the crontab line (D-15, already the existing `FINK_*` convention) |
| A runtime log-redaction filter as the sole safety net (a false sense of security if the discipline above is ever skipped) | — | Deliberately declined (D-16) — enforced by discipline plus regression tests instead; if this class ever slips past the tests, the fallback is a future runtime filter, not assumed to already exist |
| Silent failure of the scheduler itself (cron misconfigured, the lock permanently contended, the host down) | Denial of Service (from an operator-visibility standpoint) | The external heartbeat's grace period (D-12) — the only layer that can observe non-execution from outside the process |
| A leaked heartbeat/base-URL value in a committed notebook cell or runbook example | Information Disclosure | Explicit constraint carried from Phase 31: no notebook cell, transcript, or runbook example may quote live settings or `local_settings.py`; `check_unattended` prints variable names and set/unset status only, never values (D-15) |

## Sources

### Primary (HIGH confidence — read/executed this session)

- `solsys_code/management/commands/backfill_lco_observations.py` (full file, 711 lines) — current `--proposal` contract, `handle()` return shape, no `failed:` counter today
- `solsys_code/management/commands/project_observation_calendar.py` (full file, 225 lines) — `handle()` returns `None`, structured data only via `project_queryset()`
- `solsys_code/management/commands/reconcile_campaign_runs.py` (full file, 178 lines) — `handle()` returns `None`, structured data only via `reconcile_run()`
- `tom_observations/management/commands/updatestatus.py` (tomtoolkit 3.0.1, full file) — confirms "always exits 0", `str(e)` in the failure dict
- `tom_observations/facility.py:538-579` (tomtoolkit 3.0.1) — `update_observation_status()`/`update_all_observation_statuses()` signatures and return shapes, terminal-state exclusion
- `solsys_code/campaign_views.py:259-346` — `CampaignRunSubmissionView`/`_notify_staff()` verbatim
- `solsys_code/calendar_utils.py:292-341` — `resolve_placement_block()`, the class-name-only exception rule
- `solsys_code/observation_projector.py:355-381` — the contrasting `type(exc).__name__: exc` pattern for FOMO's own exceptions
- `solsys_code/apps.py:1-30` — `ready()` receiver-wiring convention
- `solsys_code/admin.py:1-176,469-472` — `CampaignRunAdmin`/registration pattern (`admin.site.register`, no decorator)
- `solsys_code/models.py` (grep pass) — model layout, no existing "config list" model to pattern-match
- `src/fomo/settings.py:190-420` — `LOGGING`, `FACILITIES`, `TOM_FACILITY_CLASSES`, `EMAIL_BACKEND`, `FINK_*` env-var convention, absence of `FOMO_BASE_URL`/`FOMO_HEARTBEAT_URL`
- `solsys_code/tests/test_backfill_lco_observations.py` (fixture/mocking conventions, `@patch(...make_request)`)
- `solsys_code/tests/test_campaign_submission.py` (`mail.outbox` assertion pattern, grep-confirmed)
- `django/test/utils.py:124-166` (installed Django, via tomtoolkit's dependency) — confirms automatic `locmem` EMAIL_BACKEND swap during tests
- `docs/design/run_identity_and_unattended_invocation_spike.rst` (full file) — the published spike verdict this phase implements
- `.planning/milestones/v2.3-phases/31-foundation-spikes-run-identity-unattended-invocation/31-DECISION.md` (targeted sections, lines 355-1034) — the real-host probe transcript, exact cron line, healthchecks/`hc-ping.com` egress confirmation
- `docs/runbooks/telescope_runs_calendar.rst` (targeted sections + full cheat-sheet/troubleshooting) — existing documented behavior for the three sweep commands, existing cheat-sheet/troubleshooting structure the new section extends
- `docs/notebooks.rst` — confirms `backfill_lco_observations_demo` is absent from the toctree
- `.planning/ROADMAP.md` — Phase 36 goal/success-criteria/locked constraints, milestone-wide "Locked constraints" block
- `.planning/REQUIREMENTS.md` — SCHED-08/09/10, DISCOVER-01 text, traceability table
- `pip show requests`, `pip show tomtoolkit`, `python -c "import fcntl"` — version/module confirmations, this session

### Secondary (MEDIUM confidence)

- healthchecks.io Pinging API documentation — `/start`, `/fail`, `/<exitcode>` conventions [CITED: https://healthchecks.io/docs/http_api/, via WebSearch this session]

### Tertiary (LOW confidence)

- `fcntl.flock`'s `LOCK_NB`/`BlockingIOError` raise behavior — standard, well-documented CPython/POSIX semantics, not independently exercised against a contended lock in this session (see Assumption A3)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; every library already installed and in active use, versions confirmed via `pip show` this session
- Architecture: HIGH — every command/module this phase orchestrates was read in full or in its relevant sections this session; the load-bearing `handle()`-returns-`None` finding was independently confirmed for two of three commands, not assumed
- Pitfalls: HIGH — all five pitfalls trace to a specific, cited line range read this session, not general domain knowledge

**Research date:** 2026-09-16
**Valid until:** 30 days (stable, in-repo domain; no fast-moving external dependency introduced) — but re-verify against the actual shipped `WatchedProposal` model/migration number if this research is consulted after Phase 36 begins executing, since `migrations/0021_...` was the latest at research time and will change.
