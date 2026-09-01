# Stack Research

**Domain:** Unattended/periodic execution of Django management commands, multi-proposal watch-list configuration, and failure visibility for a single-server, low-traffic TOM Toolkit deployment — FOMO v2.3 "Automatic Run Sync & Outcome Propagation"
**Researched:** 2026-09-01
**Confidence:** MEDIUM overall — HIGH on this repo's own architecture (direct source inspection: existing management commands, `_notify_staff`, settings.py, per-run failure isolation already established in `campaign_reconciler.py`); MEDIUM on the ecosystem comparison (cron vs. Celery/huey/APScheduler/django-crontab), which rests on web search rather than a primary-source library read, but is corroborated across multiple independent sources on facts that are stable and not contested (django-crontab's abandonment, Celery's broker requirement, huey's consumer-process requirement).

**This file supersedes the previous contents** (dated 2026-07-26, scoped to the v2.2 "One Canonical Run Record" milestone's ORM/reconciler-idempotency patterns — that milestone shipped 2026-09-01). This is a narrow, question-scoped rewrite for v2.3's specific unattended-scheduling question, not a full project stack audit — see `milestone_context` for what's deliberately out of scope (adapter consolidation, outcome propagation, and the other v2.3 target features have their own research elsewhere in this milestone's research pass).

## Headline Finding

**Plain OS-level cron invoking `python manage.py <command>` — no new Python dependency.** Every one of Celery+celery-beat, huey, APScheduler, django-crontab, and django-cron was considered and rejected for this specific deployment. The reasoning is not "cron is simpler" in the abstract — it is that **this codebase's commands are already built to the shape cron rewards** (idempotent, no-churn, per-item failure isolation — see `campaign_reconciler.reconcile_run()`/`reconcile_campaign_runs`, `insert_or_create_calendar_event()`), and every alternative's complexity buys a capability (distributed workers, sub-second scheduling precision, in-app schedule editing, retry/backoff queues) that a single-server, three-to-five-jobs-a-day astronomy coordination tool does not need. This finding directly informs the milestone's own planned "phase-time investigation spike" — treat it as the spike's starting hypothesis to confirm against the real target host, not a substitute for that spike.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| OS cron (`crontab -e` / `/etc/cron.d/`) | whatever ships with the deployment OS (no version to pin) | Periodic invocation of management commands | Zero new dependency, zero new daemon/process, zero new infra (no broker, no worker, no beat process). Runs `python manage.py <command>` exactly the way an operator already runs it by hand today — the "unattended" story is additive, not a rewrite of how these commands work. |
| `flock` (`util-linux`, present on essentially every Linux distro) | system package, not pip | Prevent overlapping runs if one invocation runs long | A sync command that takes longer than the cron interval (network-dependent — LCO/Gemini API calls) must not run twice concurrently against the same SQLite file; `flock -n /tmp/fomo-sync.lock python manage.py ...` is the standard, dependency-free guard. Matters more on SQLite (single-writer) than Postgres, but correct either way. |
| Django's own `mail_admins()` / staff-email pattern (already in this codebase) | Django 5.2.17 (already installed) | In-command failure notification | No new dependency — reuses the exact idiom `campaign_views.py::_notify_staff()` already established (`send_mail(..., fail_silently=True)` to a staff email list), just triggered from a caught exception in the unattended command path instead of a submission POST. |
| healthchecks.io (hosted free tier) or self-hosted [`healthchecks`](https://github.com/healthchecks/healthchecks) | n/a (external service / self-hosted Django app) | "The whole cron entry never fired" visibility | Catches the one failure class in-command error handling structurally cannot: the job not running at all (cron daemon down, host down, crontab misconfigured, silent SSH-key/permission failure). A single `curl -fsS --retry 3 https://hc-ping.com/<uuid>` appended to each cron line. Free tier (20 checks, no card required as of this research) covers this project's handful of jobs; self-hosted is a fallback if a third-party dependency for something this operationally load-bearing is unwelcome — it is itself a small open-source Django app, which fits this team's existing skill set if self-hosting is preferred. |

### Supporting Libraries

**None required.** No new pip package is warranted for the scheduling mechanism itself. See "What NOT to Use" for the specific packages considered and rejected, with reasons.

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `logger.exception(...)` (stdlib `logging`, already used throughout `solsys_code/`) | Capture full traceback on a caught failure before emailing a summary | Matches this codebase's existing logging convention (`logger = logging.getLogger(__name__)`, debug-level for expected failures) — an unattended-run failure is not "expected" in the same sense as e.g. a JPL query 404, so log at `exception`/`error`, not `debug`. |
| `./manage.py test solsys_code` | Regression gate for the new orchestrator/watch-list code | No new test infra — same Django `TestCase` convention as every other management command in this repo. |

## Installation

```bash
# No new pip package required for the scheduling mechanism.
# The only new artifact is a crontab entry (or systemd timer, if the target host's
# existing conventions prefer that over cron — confirm with the phase-time spike) and,
# optionally, a healthchecks.io account (or self-hosted healthchecks instance).

# Example crontab line (adjust interval to real deployment cadence, e.g. every 15-30 min):
# */15 * * * * flock -n /tmp/fomo-unattended-sync.lock \
#   /path/to/venv/bin/python /path/to/fomo/manage.py run_unattended_sync \
#   >> /var/log/fomo/unattended_sync.log 2>&1 \
#   && curl -fsS --retry 3 https://hc-ping.com/<uuid> \
#   || curl -fsS --retry 3 https://hc-ping.com/<uuid>/fail
```

If a future data-volume increase genuinely outgrows cron (see "Stack Patterns by Variant" below), the next step up is `huey>=2.5,<4` (current stable line is 3.x; latest is `3.3.4` per `huey.readthedocs.io`) with its SQLite storage backend (`huey.contrib.djhuey`, no Redis needed) — but that is a documented escape hatch, not a recommendation to install now.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| OS cron + `flock` | Celery + celery-beat | Only if a real driver for a message broker (Redis/RabbitMQ) and distributed workers already exists elsewhere in the deployment, or job volume/concurrency genuinely requires horizontal worker scaling. Neither is true here: this is a single server, a handful of periodic jobs, and no other part of FOMO uses a broker today. Introducing Celery here means running and operating two new long-lived daemons (a worker and a beat scheduler) plus a broker, purely to run `python manage.py sync_x` on a timer — the same outcome cron already provides for free. |
| OS cron + `flock` | huey (Redis or SQLite-backed) | If the team later adds genuinely asynchronous, latency-sensitive background work (e.g. "kick off X the moment a web request happens, don't block the response") — a real use case distinct from *periodic* sync. Huey's SQLite storage backend removes the Redis dependency, which narrows the gap versus cron, but it still requires a long-running consumer process (`huey_consumer.py`) to be supervised (systemd unit, restart-on-crash, log rotation) — one more daemon to operate for jobs that are inherently "run every N minutes," which cron already does natively with no daemon of its own. |
| OS cron + `flock` | APScheduler (in-process, `BackgroundScheduler`) | Only for a deployment that runs as a single long-lived Python process. This project is a WSGI Django app; if it's ever served by more than one worker process (gunicorn `-w 2+`, which is a normal production default, not an edge case), an in-process scheduler registered at Django startup fires once **per worker process**, silently multiplying every sync — a real, easy-to-miss correctness bug, not a hypothetical. It also stops running the moment the process restarts (deploy, crash, worker recycle) with no persistence of a missed run, unlike cron which is independent of the app process's lifecycle. |
| OS cron + `flock` | django-cron (`django_cron`, DB-backed cron-job registry with admin visibility) | If the team wants job history/success-tracking *inside* Django admin without standing up healthchecks.io. It's a real, still-maintained option (ships `FailedRunsNotificationCronJob` for exactly the "N failures in a row -> email" pattern this milestone needs) and is lighter than Celery/huey. It was not chosen as the primary recommendation because it still requires *something* to invoke `python manage.py runcrons` on a schedule — i.e., it doesn't remove cron, it adds a DB-polling layer and a new dependency on top of cron, for a job-history UI this milestone's smaller healthchecks-ping + email-on-exception combination already covers at lower cost. Worth a second look if the operator later wants an in-app dashboard of pass/fail history rather than an external service. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `django-crontab` | Last released 2016 (`0.7.1`), effectively abandoned (no commits in ~8 years per package-health scans). It is also, mechanically, just a Python-side wrapper that writes real crontab entries on your behalf — for a project with no existing scheduler at all, hand-writing the crontab entry directly is strictly simpler and has no unmaintained dependency in the critical path. | A plain crontab entry, hand-managed (or provisioned via whatever config-management/deploy tooling the target host already uses). |
| Celery + celery-beat + Redis/RabbitMQ | Requires standing up and operating a message broker plus two new long-lived daemons (worker, beat) for a workload that is "run 3-5 idempotent management commands a few times an hour." This is the textbook over-engineering case the milestone context explicitly warns against ("avoid a heavyweight broker... unless there's a concrete reason") — there is no concrete reason here: no other part of FOMO needs async task execution, and the sync commands are already synchronous, idempotent, and fast enough to run inline under cron. | OS cron. |
| Bare `huey` with a Redis backend | Same broker-dependency problem as Celery, one notch smaller — still a new service (Redis) with no other consumer in this deployment. | If a queue is later justified, huey's SQLite backend removes this specific objection (see Alternatives table), but that's a distinct, deferred decision. |
| APScheduler `BackgroundScheduler` registered at Django app-config startup | Multi-worker WSGI deployments (gunicorn, uWSGI with >1 worker) cause it to fire once per worker process, silently duplicating every sync — a correctness bug that's easy to ship and hard to notice until duplicate `CampaignRun`s/calendar events appear. It also has no execution history independent of the app process. | OS cron, which is worker-count-agnostic by construction (one crontab entry regardless of how many app workers are running). |
| A bespoke in-house "scheduler" (e.g. a Django management command that sleeps in a loop, run once at boot under systemd) | Reinvents cron's job (interval scheduling, skip-if-still-running, restart-on-host-reboot) worse than cron already does it, and needs its own supervision (a `systemd` unit with `Restart=always`) to survive a crash — cron already survives process crashes because it isn't a process, it's invoked fresh each time by the system's own cron daemon. | OS cron (or, if the target host's own conventions already lean on `systemd`, a `systemd` timer unit — mechanically equivalent to cron for this purpose; confirm which the deployment already uses before introducing the other). |
| `mail_admins()`/Django's `ADMINS` setting as the *sole* failure-visibility mechanism | `ADMINS` is currently unset in `src/fomo/settings.py` (confirmed by direct read), and even if configured, it only fires on an *exception the command actually raises and Django actually reports* — it cannot detect "the cron entry never fired," "the host was down," or "the process was OOM-killed before it could send mail." | Combine in-command email-on-exception (reusing the existing `_notify_staff`-style staff-email pattern, not a fresh `ADMINS` config) with an external dead-man's-switch ping — the two failure classes are genuinely different and neither alone is sufficient. |

## Stack Patterns by Variant

**If deployment stays single-server SQLite (current dev reality) or moves to single-server Postgres (CLAUDE.md's stated production expectation):** the cron + `flock` recommendation is unchanged either way — nothing about the scheduling mechanism is database-specific. `flock` matters slightly more on SQLite (single-writer file lock contention is a real, visible failure mode under `database is locked` errors) but is correct and harmless on Postgres too.

**If job volume or latency needs genuinely grow** (e.g. dozens of proposals polled every minute, or a requirement for sub-cron-granularity scheduling): re-evaluate huey with its SQLite backend first (smallest step up, no Redis), and only reach for Celery if true multi-worker horizontal scaling of the *sync* workload itself becomes necessary — not just because "task queues are the standard approach" in general. This project's current real numbers (a handful of proposals, 3-4 management commands, dev-DB scale in the tens of rows) are nowhere near that threshold.

**If the target production host already runs `systemd` and the team's operational convention is timers over crontabs:** a `systemd.timer` + `systemd.service` pair is a mechanically equivalent substitute for the crontab entry described above — same "no new Python dependency, no new daemon beyond what the OS already runs" property. This is a deployment-convention choice, not a different recommendation; confirm which the real target host already uses before the implementation phase, since CLAUDE.md documents no production deployment infra for this repo today.

## Version Compatibility

| Package | Compatible With | Notes |
|---------|------------------|-------|
| Django 5.2.17 (confirmed installed via `pip show django`) | Python 3.10-3.12 (project's tested range) | `mail_admins()`, `send_mail()`, and `django.core.management.call_command()` are long-stable, unchanged Django APIs — nothing about this recommendation is gated on Django 5.2 specifically. |
| OS cron / `flock` | Any Linux server (dev machine or production host) | Not a Python dependency at all — no version-compatibility surface with Django/TOM Toolkit/`tomtoolkit` to track. |
| `huey` (only if the escape hatch is later taken) | `huey==3.3.4` (current, per `huey.readthedocs.io`) supports Python 3.7+ and Django via `huey.contrib.djhuey` | Not needed now; recorded here only so a future re-evaluation starts from a current version, not a stale one found via search. |

## New Django Convention This Milestone Needs: the Multi-Proposal Watch-List

The milestone context is explicit that `backfill_lco_observation_records`-style discovery currently requires an explicit `--proposal`/name-prefix per invocation, and unattended discovery needs a configured watch-list instead. Four conventions were weighed against this codebase's existing patterns (Django/TOM Toolkit norms, not general Django advice):

| Approach | Fits this codebase? | Verdict |
|----------|----------------------|---------|
| A small Django model (e.g. `WatchedProposal`: `proposal_code`, `facility` choice, `is_active`, optional `notes`), registered in `admin.py` | **Yes — this is the established convention here.** `Observatory` (site config) and `CampaignRun` (run config) are both plain models, editable via Django admin, with no code deploy needed to add a row. Staff already use Django admin for comparable operational data (`CalendarEventMetaAdmin`, `CampaignRunObservationInline`). | **Recommended.** Adding/removing a watched proposal is then an admin action, not a deploy — matches how this team already operates (staff-facing approval queues, admin actions like `mark_cancelled`), and the model can grow a `last_synced_at`/`last_result` field later without a settings-file migration. |
| `settings.py` list (e.g. `WATCHED_LCO_PROPOSALS = [...]`) | Technically simplest to implement, but requires a code change + deploy/restart to add a proposal — proposals are seasonal/per-semester in this domain, so this would reintroduce exactly the kind of "operator can't self-serve" friction this milestone is trying to remove from `backfill_lco_observation_records`'s current `--proposal`-per-invocation requirement. | Rejected as the primary mechanism — acceptable only as a *bootstrap default* seeded once via a data migration, not as the ongoing edit surface. |
| Environment variable (comma-separated proposal codes) | Same self-service problem as `settings.py`, worse ergonomics (no admin UI, easy to typo, invisible to anyone without shell/deploy access), and this codebase's existing env-var usage (`FINK_CREDENTIAL_*`, `LASAIR_TOKEN`) is reserved for *credentials/secrets*, not operational config that changes routinely — mixing the two would blur an existing, sensible convention. | Rejected. |
| A YAML/JSON config file checked into the repo or mounted on the server | No existing precedent anywhere in this codebase (grep confirms no config-file-loading pattern exists today) — would be a genuinely new convention introduced solely for this one feature, and still requires filesystem/deploy access to edit, same self-service gap as the env-var option. | Rejected. |

**Concrete shape**, deliberately as small as the existing `Observatory` model:

```python
class WatchedProposal(models.Model):
    """A proposal/facility pair the unattended discovery sweep should poll without an
    explicit --proposal argument (v2.3: closes the "one invocation per proposal" gap
    backfill_lco_observation_records currently has).
    """

    proposal_code = models.CharField(max_length=64)
    facility = models.CharField(max_length=32, choices=[('LCO', 'LCO'), ('SOAR', 'SOAR')])
    is_active = models.BooleanField(default=True)
    notes = models.CharField(max_length=255, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=('proposal_code', 'facility'), name='unique_watched_proposal'),
        ]
```

The sweep command keeps its existing `--proposal` flag for one-off manual runs (backward compatible, useful for debugging a single proposal), and falls back to `WatchedProposal.objects.filter(is_active=True)` when no `--proposal` is supplied — the same "explicit argument overrides configured default" shape already used elsewhere in Django management commands, not a new pattern to learn.

## Failure Visibility Pattern (concrete recommendation)

Two independent layers, because they catch two independent, non-overlapping failure classes:

1. **In-command exception -> staff email.** Wrap the unattended entry point in a `try`/`except Exception`, `logger.exception(...)` the traceback, and send a short email to staff using the exact idiom already shipped in `campaign_views.py::_notify_staff()` (`User.objects.filter(is_staff=True).exclude(email='')`, `send_mail(..., fail_silently=True)`) — reuse that helper (factor it out to a shared location if it isn't already importable) rather than inventing a second notification mechanism or introducing Django's separate, currently-unused `ADMINS`/`AdminEmailHandler` machinery. This catches "the job ran and broke."

2. **Dead-man's-switch ping -> healthchecks.io (or self-hosted).** Append a ping to the *end* of the cron line itself (not inside Django) using the two-URL pattern (`.../ping/<uuid>` on success, `.../ping/<uuid>/fail` on any non-zero exit), with a grace period configured generously past the expected run time. This catches "the job never ran at all" — a class of failure no amount of in-app error handling can detect, because the app process never started.

**Recommended orchestration shape:** rather than wiring `flock` + email + a healthchecks ping around *each* of the three ingest commands and the reconciler separately (four to five near-identical cron lines), introduce one small orchestrating management command (e.g. `run_unattended_sync`) that calls each step (`load_telescope_runs`, `sync_lco_observation_calendar`, `sync_gemini_observation_calendar`, then `reconcile_campaign_runs`) with **per-step failure isolation** — directly mirroring the per-run failure isolation `reconcile_campaign_runs` already established in v2.2 (one bad run/step doesn't abort the sweep). The orchestrator collects a pass/fail summary, emails staff once if anything failed (a single digest, not N separate emails), and exits non-zero only if something failed, which is what drives the healthchecks `/fail` ping. This gives cron exactly one line to manage and keeps the per-adapter isolation logic in Python (testable with `./manage.py test`) instead of shell.

## Sources

- Direct inspection of this repository (HIGH confidence): `pyproject.toml` (no celery/huey/apscheduler/django-crontab/django-cron dependency present), `src/fomo/settings.py` (SQLite `DATABASES`, `EMAIL_BACKEND` console default, no `ADMINS`/`MANAGERS` configured, `TIME_ZONE='UTC'`), `solsys_code/campaign_views.py` (`_notify_staff()` — the existing staff-email idiom this recommendation reuses), `solsys_code/management/commands/backfill_lco_observation_records.py` (`--proposal` argument this milestone needs to make optional/watch-list-driven), `solsys_code/campaign_reconciler.py`/`reconcile_campaign_runs.py` (existing per-run failure isolation precedent the orchestrator recommendation mirrors), `.planning/PROJECT.md` (v2.3 milestone context, target features, explicit out-of-scope list).
- [django-crontab package health (Snyk Advisor)](https://snyk.io/advisor/python/django-crontab) — MEDIUM confidence; corroborated by the PyPI/Cloudsmith listing showing `0.7.1` (2016-03-07) as latest.
- [Huey documentation, latest](https://huey.readthedocs.io/en/latest/) and [Huey documentation, Django integration](https://huey.readthedocs.io/en/latest/django.html) — MEDIUM confidence (official project docs); confirms current `3.x` line, SQLite/Redis/Postgres/filesystem storage backends, `djhuey` periodic-task decorators.
- [Healthchecks.io docs — Monitoring Cron Jobs](https://healthchecks.io/docs/monitoring_cron_jobs/) and [Healthchecks.io docs](https://healthchecks.io/docs/) — MEDIUM confidence (official project docs); confirms the ping-on-success/ping-on-fail pattern and grace-period semantics. Self-hosted alternative: [healthchecks/healthchecks on GitHub](https://github.com/healthchecks/healthchecks).
- General web search corroboration (cron vs. Celery/huey/APScheduler tradeoffs; django-cron's `FailedRunsNotificationCronJob`; `mail_admins()`/`AdminEmailHandler` as the general Django failure-notification primitive) — LOW-to-MEDIUM confidence individually (blog/tutorial sources), used only where the claim was corroborated across multiple independent results and is consistent with this repo's own architecture, never as the sole basis for a recommendation.

---
*Stack research for: FOMO v2.3 "Automatic Run Sync & Outcome Propagation" — unattended scheduling mechanism, multi-proposal watch-list, and failure visibility*
*Researched: 2026-09-01*
