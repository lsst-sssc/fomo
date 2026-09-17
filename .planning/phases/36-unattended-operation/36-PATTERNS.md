# Phase 36: Unattended Operation - Pattern Map

**Mapped:** 2026-09-16
**Files analyzed:** 14 (new/modified)
**Analogs found:** 12 / 14

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `solsys_code/unattended.py` | service (step functions, notify/heartbeat helpers) | batch / event-driven | `solsys_code/management/commands/reconcile_campaign_runs.py` (loop-and-isolate shape) + `solsys_code/campaign_reconciler.py` (pure-function-under-a-command pattern) | role-match |
| `solsys_code/management/commands/run_unattended.py` | route (thin management command) | batch | `solsys_code/management/commands/reconcile_campaign_runs.py` | exact |
| `solsys_code/management/commands/check_unattended.py` | route (preflight command) | request-response (read-only checks) | `solsys_code/management/commands/project_observation_calendar.py` (bare-invocation, all-optional-args shape) | role-match |
| `solsys_code/models.py` → `WatchedProposal` | model | CRUD | `solsys_code/models.py` `CalendarEventDismissal` (small config/audit model, FK + bookkeeping fields) | role-match |
| `solsys_code/migrations/00NN_watchedproposal.py` | migration | CRUD | `solsys_code/migrations/0021_alter_calendareventmeta_is_verified_and_more.py` | exact (structural shape only — this is a `CreateModel`, not `AlterField`) |
| `solsys_code/admin.py` → `WatchedProposalAdmin` | config (Django admin) | CRUD | `solsys_code/admin.py` `CampaignRunAdmin` (list_display/list_filter) — no `list_editable` example exists yet, standard Django | role-match |
| `solsys_code/management/commands/backfill_lco_observations.py` (D-07..D-09 rewrite) | controller (management command) | CRUD + event-driven (portal poll) | itself (existing file, in-place restructure) | exact |
| `solsys_code/campaign_views.py` → extract `notify_staff()` | service (extracted from controller) | request-response → event-driven | `solsys_code/campaign_views.py` `CampaignRunSubmissionView._notify_staff()` (lines ~327-346) | exact |
| `src/fomo/settings.py` → `FOMO_BASE_URL`, `FOMO_HEARTBEAT_URL` | config | — | `src/fomo/settings.py` `DATA_SERVICES`/`ALERT_STREAMS` `FINK_*` block (lines 314-319) and `FOMO_DATABASE_PATH` (line 134) | exact |
| `deploy/cron/fomo.crontab.example` | config | — | none in-repo (new top-level `deploy/` dir) — pattern comes from `31-DECISION.md:926-929` cron-line transcript, not a source file | no analog |
| `deploy/logrotate/fomo.example` | config | — | none in-repo | no analog |
| `solsys_code/tests/test_unattended.py` | test | unit | `solsys_code/tests/test_reconcile_campaign_runs.py` + `solsys_code/tests/test_campaign_submission.py` (`mail.outbox` pattern) | role-match |
| `solsys_code/tests/test_check_unattended.py` | test | unit | `solsys_code/tests/test_admin.py` (settings/DB-state assertions) | partial |
| `solsys_code/tests/test_watched_proposal.py` | test | unit | `solsys_code/tests/test_admin.py` | role-match |
| `solsys_code/tests/test_backfill_lco_observations.py` (extend) | test | unit | itself (existing file, `@patch(...make_request)` convention) | exact |

## Pattern Assignments

### `solsys_code/unattended.py` (service, batch/event-driven)

**Analog:** `solsys_code/management/commands/reconcile_campaign_runs.py` (loop shape) + `solsys_code/management/commands/project_observation_calendar.py` (module-function-not-call_command discipline)

**Imports pattern** (from `reconcile_campaign_runs.py` lines 1-9):
```python
import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.models import CampaignRun

logger = logging.getLogger(__name__)
```
For `unattended.py` itself (not a `BaseCommand`), drop the `BaseCommand`/`CommandParser` imports and instead import the three step primitives directly, per RESEARCH.md's verified finding:
```python
from solsys_code.observation_projector import project_queryset
from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.calendar_utils import resolve_placement_block  # for D-03's class-name recovery, if adopted
from solsys_code.models import CampaignRun, WatchedProposal
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility  # confirm actual import path before use
```

**Core per-step failure-isolation pattern** (`reconcile_campaign_runs.py` lines 61-74 — copy this try/except-and-continue shape for every one of the runner's four steps):
```python
for run in runs:
    run_count += 1
    try:
        result = reconcile_run(run, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 -- the only catch point, D-06
        logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)
        self.stderr.write(f'Run pk={run.pk}: reconcile failed ({exc}) -- skipping')
        failed_count += 1
        continue
```
Note: this analog logs `str(exc)` because it is FOMO's own `reconcile_run()` exception (D-17's second bucket — FOMO-raised exceptions may carry a message). The runner's status-refresh and discovery steps wrap **network/portal** exceptions instead, which must follow the class-name-only rule below, not this one.

**Credential-free exception pattern** (`solsys_code/calendar_utils.py` lines ~317-329, `resolve_placement_block()` — copy for every portal/network except clause on the unattended path):
```python
try:
    response = make_request('GET', ..., timeout=_API_TIMEOUT_SECONDS)
    blocks = response.json()
except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, ValueError):
    return None  # no str(exc) anywhere -- SYNC-09/D-11's rule
```
Contrast — FOMO's own pipeline exceptions **may** carry `str(exc)` (`solsys_code/observation_projector.py` lines 364-372):
```python
except Exception as exc:  # noqa: BLE001
    logger.warning('unprojectable observation_id=%r: %s: %s', record.observation_id, type(exc).__name__, exc)
    return 'unprojectable', type(exc).__name__
```

**Module-function-not-`call_command()` pattern** (`project_observation_calendar.py` lines 190-199 — the hook/closure shape the runner's status-refresh and projector steps should mirror):
```python
def hook(record: ObservationRecord, facility: Any) -> dict[str, int] | None:
    increment, message = resolve_observed_site(record, facility)
    if message:
        self.stderr.write(message)
    return increment

result = project_queryset(records, dry_run=dry_run, pre_fields_hook=None if dry_run else hook)
```
For the runner, call `project_queryset(ObservationRecord.objects.all(), dry_run=False, pre_fields_hook=hook)` directly (never `call_command('project_observation_calendar')` — verified `handle()` always returns `None`).

**Gemini password-strip idiom** (`solsys_code/management/commands/sync_gemini_observation_calendar.py` line 47-48 — same "strip before anything is logged" discipline, useful template for any place `unattended.py` must scrub a dict before logging):
```python
# D-04: strip password immediately, before any logging or field derivation.
safe_params = {k: v for k, v in (record.parameters or {}).items() if k != 'password'}
```

---

### `solsys_code/management/commands/run_unattended.py` (route, batch)

**Analog:** `solsys_code/management/commands/reconcile_campaign_runs.py` (full structure)

**Imports + docstring + `add_arguments()` pattern** (lines 1-32, adapt for `--dry-run` / `--step`):
```python
import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.models import CampaignRun

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """<docstring: what this command orchestrates and why>"""

    help = '<one-line help>'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would run without writing anything, and without mailing/pinging.',
        )
        # No return statement — BaseCommand.add_arguments() returns None
```
`handle()` should call a single `unattended.run_tick(dry_run=...)` function living in `unattended.py` (keeps the command thin, matching every other command in this codebase — the domain logic lives in a module, the command is a wrapper) and `sys.exit(exit_code)`.

---

### `solsys_code/management/commands/check_unattended.py` (route, request-response/read-only)

**Analog:** `solsys_code/management/commands/project_observation_calendar.py` (all-optional-args, `CommandError` for a hard failure)

**CommandError-on-hard-failure pattern** (`project_observation_calendar.py` lines 179-187):
```python
if not codes:
    # Fail closed: ... silently widening would sweep the opposite of what the operator asked for
    raise CommandError(f'--proposal {proposal_raw!r} names no usable proposal code.')
```
`check_unattended` should raise `CommandError` (non-zero exit) for each hard-failure check (D-13's "no staff email" case) while using `self.stdout.write()`/`self.stderr.write()` for warnings (heartbeat unset, empty watched list — D-08/D-12), never raising for those.

---

### `solsys_code/models.py` → `WatchedProposal` (model, CRUD)

**Analog:** `solsys_code/models.py` `CalendarEventDismissal` (lines 639-694) for FK/bookkeeping field shape; `CampaignRun`'s `TextChoices` pattern (lines 201-217) if `is_active` ever needs richer states (not needed here — plain `BooleanField`).

**FK + bookkeeping field pattern** (`CalendarEventDismissal`, lines 654-682):
```python
event = models.ForeignKey(
    CalendarEvent,
    on_delete=models.CASCADE,
    related_name='attribution_dismissals',
    verbose_name='Calendar event',
)
dismissed_by = models.ForeignKey(
    settings.AUTH_USER_MODEL,
    on_delete=models.SET_NULL,
    null=True,
    blank=True,
    related_name='dismissed_calendar_event_attributions',
    verbose_name='Dismissed by',
)
dismissed_at = models.DateTimeField(null=True, blank=True, verbose_name='Dismissed at')
reason = models.TextField(blank=True, default='', verbose_name='Why this candidate was rejected')

class Meta:  # noqa: D106
    constraints = [
        models.UniqueConstraint(fields=('event', 'run'), name='unique_calendar_event_dismissal_pair'),
    ]

def __str__(self):
    return f'dismissed {self.event} for {self.run}'
```
For `WatchedProposal`, apply this shape with: `proposal_code = models.CharField(max_length=..., unique=True)`, `is_active = models.BooleanField(default=True)`, `target_list_name = models.CharField(blank=True, default='')`, `attributed_to = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)`, `last_run_at = models.DateTimeField(null=True, blank=True)`, `last_run_summary = models.TextField(blank=True, default='')`. Use `unique=True` on the field itself rather than a `UniqueConstraint` (single-field uniqueness — `CalendarEventDismissal`'s constraint is for the two-FK pair case, not applicable here).

---

### Migration (structural template)

**Analog:** `solsys_code/migrations/0021_alter_calendareventmeta_is_verified_and_more.py` (header/dependency shape only — this migration is an `AlterField`; the new one is `CreateModel`, use Django's `makemigrations` output, don't hand-write `CreateModel` from scratch):
```python
# Generated by Django 5.2.17 on 2026-09-16 17:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('solsys_code', '0021_alter_calendareventmeta_is_verified_and_more'),
    ]

    operations = [
        # migrations.CreateModel(...) -- run `python manage.py makemigrations solsys_code`
    ]
```
Next migration number after `0021` (confirm the actual latest at execution time — RESEARCH.md flags this as a moving target).

---

### `solsys_code/admin.py` → `WatchedProposalAdmin` (config, CRUD)

**Analog:** `CampaignRunAdmin` (lines 142-176)

**Registration pattern**:
```python
class WatchedProposalAdmin(admin.ModelAdmin):  # noqa: D101
    list_display = ['proposal_code', 'is_active', 'last_run_at', 'last_run_summary']
    list_filter = ['is_active']
    list_editable = ['is_active']  # no existing example in this codebase; standard Django API
    ordering = ['proposal_code']


admin.site.register(WatchedProposal, WatchedProposalAdmin)
```
Registration call site pattern (`admin.py` lines 469-472):
```python
admin.site.register(CampaignRun, CampaignRunAdmin)
admin.site.register(CalendarEventMeta, CalendarEventMetaAdmin)
```

---

### `solsys_code/management/commands/backfill_lco_observations.py` (D-07..D-09 rewrite)

**Analog:** itself — `add_arguments()` (lines 445-476) and `handle()` (lines 478-524+)

**Current required-`--proposal` shape to relax to optional** (lines 447-451):
```python
parser.add_argument(
    '--proposal',
    required=True,
    help='LCO proposal code to filter RequestGroups by (exact match).',
)
```
Change to `required=False, default=None`; when `None`, loop `WatchedProposal.objects.filter(is_active=True)` instead (D-07/D-09). Per RESEARCH.md's Open Question 2 recommendation: extract the existing per-proposal request-group loop (`handle()` body, roughly lines 497-710) into a function taking `(facility, proposal, target_list_override, user)` returning `(summary: dict, exception: Exception | None)`, called once per watched row (wrapped in its own try/except per D-09) and once for a `--proposal` override — this keeps the 30 existing tests passing since the request-group wire format is unchanged.

**User-resolution / `CommandError` pattern to reuse per-row** (lines 490-498):
```python
user = None
if options.get('username'):
    try:
        user = get_user_model().objects.get(username=options['username'])
    except get_user_model().DoesNotExist as exc:
        raise CommandError(f'Invalid username: {options["username"]!r}') from exc

facility = LCOFacility()
facility.set_user(user)
```

**Test mocking convention** (`solsys_code/tests/test_backfill_lco_observations.py`):
```python
from unittest.mock import MagicMock, patch

def _page_response(results, next_url=None):
    response = MagicMock()
    response.json.return_value = {'count': len(results), 'next': next_url, 'previous': None, 'results': results}
    return response

class SomeTest(TestCase):
    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_something(self, mock_make_request):
        mock_make_request.return_value = _page_response([...])
```
Patch `make_request` at its import point inside the command module, not at its source (`tom_observations.facilities.ocs.make_request`).

---

### `solsys_code/campaign_views.py` → extract `notify_staff()`

**Analog:** `CampaignRunSubmissionView._notify_staff()` (verbatim, lines ~327-346 per RESEARCH.md):
```python
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
        fail_silently=True,  # a mail outage must never break the submission
    )
```
Extracted helper (e.g. in `unattended.py` or a shared module) must: take an explicit `base_url` (from `settings.FOMO_BASE_URL`) instead of `self.request`; keep the identical recipient rule (`User.objects.filter(is_staff=True).exclude(email='')`); default `fail_silently=False` for the unattended-path caller (the runner catches and logs `type(exc).__name__` per D-17), while `CampaignRunSubmissionView` keeps calling it with `fail_silently=True` at its own call site.

Imports at the top of `campaign_views.py` to mirror when adding the extraction (lines 1-25):
```python
import logging
import re
from datetime import date, datetime
from datetime import timezone as dt_timezone

from django.contrib import messages
from django.contrib.auth.models import User
```

**Mail testing pattern** (`solsys_code/tests/test_campaign_submission.py`):
```python
from django.core import mail

class SomeUnattendedTest(TestCase):
    def test_failure_email_sent(self):
        # ... trigger a failing tick ...
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn(staff_with_email.email, mail.outbox[0].to)
```

---

### `src/fomo/settings.py` → `FOMO_BASE_URL`, `FOMO_HEARTBEAT_URL`

**Analog:** the `FINK_*` `os.getenv()` block (lines 314-319):
```python
'URL': os.getenv('FINK_CREDENTIAL_URL', 'set FINK_CREDENTIAL_URL value in environment'),
'USERNAME': os.getenv('FINK_CREDENTIAL_USERNAME', 'set FINK_CREDENTIAL_USERNAME value in environment'),
'GROUP_ID': os.getenv('FINK_CREDENTIAL_GROUP_ID', 'set FINK_CREDENTIAL_GROUP_ID value in environment'),
```
and the single top-level setting shape (`FOMO_DATABASE_PATH`, line 134):
```python
'NAME': os.getenv('FOMO_DATABASE_PATH') or os.path.join(BASE_DIR, 'fomo_db.sqlite3'),
```
New settings:
```python
FOMO_BASE_URL = os.getenv('FOMO_BASE_URL', 'http://localhost:8000')
FOMO_HEARTBEAT_URL = os.getenv('FOMO_HEARTBEAT_URL')  # None when unset -- D-12's "off" branch
```
`EMAIL_BACKEND` default to check against in `check_unattended` (line 403):
```python
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

---

### `deploy/cron/fomo.crontab.example` / `deploy/logrotate/fomo.example`

**No in-repo analog** — new top-level directory. Source the exact shape from `31-DECISION.md:926-929` / `36-RESEARCH.md` "Code Examples" section (already verified against the real interim host):
```
*/15 * * * * /usr/bin/flock -n /var/lock/fomo/run_unattended.lock \
    /path/to/venv/bin/python /path/to/checkout/manage.py run_unattended \
    >> /var/log/fomo/unattended.log 2>&1
```

---

## Shared Patterns

### Credential-free exception logging (SCHED-10 / D-17)
**Source:** `solsys_code/calendar_utils.py` `resolve_placement_block()` (~lines 292-329)
**Apply to:** every except clause in `unattended.py`'s status-refresh and discovery steps, and the extracted `notify_staff()` / heartbeat-ping helpers.
```python
except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, ValueError):
    return None  # never str(exc)
```
Contrast for FOMO's own raised exceptions (message allowed): `solsys_code/observation_projector.py` lines 364-372, `reconcile_campaign_runs.py` lines 65-69.

### Per-item failure isolation with a running summary
**Source:** `solsys_code/management/commands/reconcile_campaign_runs.py` lines 61-176 (loop + per-field counters + final `Done. runs: N, created: N, ...` summary line)
**Apply to:** the runner's four steps, `backfill_lco_observations`'s new per-`WatchedProposal` loop, `check_unattended`'s per-prerequisite report.

### Module-function-not-`call_command()` for structured results
**Source:** `solsys_code/management/commands/project_observation_calendar.py` lines 190-224 (`project_queryset()` called directly, result dict inspected)
**Apply to:** every step `unattended.py` wraps — never `call_command('project_observation_calendar')` / `call_command('reconcile_campaign_runs')`, since `handle()` returns `None` for both (RESEARCH.md-verified).

### `BaseCommand` file skeleton (docstring, `help`, `add_arguments`, typed `handle`)
**Source:** `solsys_code/management/commands/reconcile_campaign_runs.py` (whole file, 177 lines) — the cleanest, most recently-modified example of the shape both `run_unattended.py` and `check_unattended.py` should follow (module docstring explaining what/why, `help` string, `add_arguments(self, parser: CommandParser) -> None`, `handle(self, *args: Any, **options: Any) -> str | None`).

### Admin registration (no decorator form)
**Source:** `solsys_code/admin.py` lines 142-176, 469-472 (`class XAdmin(admin.ModelAdmin)` + `admin.site.register(Model, ModelAdmin)` at file end — this codebase never uses `@admin.register()`).

### Env-var-only credentials
**Source:** `src/fomo/settings.py` lines 314-319 (`FINK_*` `os.getenv()` convention)
**Apply to:** `FOMO_BASE_URL`, `FOMO_HEARTBEAT_URL` — never a CLI arg, never embedded in the crontab line (D-15).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `deploy/cron/fomo.crontab.example` | config | — | No `deploy/` directory exists in this repo yet; pattern sourced from `31-DECISION.md`'s real-host transcript instead of a codebase analog |
| `deploy/logrotate/fomo.example` | config | — | Same — no logrotate example exists anywhere in-repo; standard logrotate stanza (daily, rotate 14) is external convention, not a local pattern |

## Metadata

**Analog search scope:** `solsys_code/management/commands/`, `solsys_code/models.py`, `solsys_code/admin.py`, `solsys_code/campaign_views.py`, `solsys_code/calendar_utils.py`, `solsys_code/observation_projector.py`, `solsys_code/migrations/`, `src/fomo/settings.py`, `solsys_code/tests/`
**Files scanned:** ~14 read directly this session (plus RESEARCH.md's own verified excerpts reused where a re-read would have duplicated an already-in-context range)
**Pattern extraction date:** 2026-09-16
