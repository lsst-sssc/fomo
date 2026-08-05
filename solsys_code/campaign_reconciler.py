"""Pure-logic reconciler core (D-01/D-03, 29-CONTEXT.md).

Projects and refreshes a ``CampaignRun``'s calendar events as a function of the run's own
state, computed by one idempotent per-run function (``reconcile_run()``) shared by the batch
command (plan 29-03) and the four staff-action call sites (plan 29-04) -- this is the only
way RECON-01's "running it a second time changes nothing" and RECON-08's "a staff decision
reconciles immediately" can be guaranteed to agree with each other.

Two coexisting key families (26-DECISION.md "Criterion 3 / SPIKE-03"): a bare
``RUN:{run_pk}`` whole-window container for queue-scheduled, class-wide and
satellite/space runs (RECON-02 queue half, RECON-03), and a date-bearing
``RUN:{run_pk}:{date}`` key per observing night for classically-scheduled runs (RECON-02
classical half) -- always date-bearing, including for a single-night run, so the key form
alone says which family an event belongs to.

Like ``campaign_gap.py``/``campaign_utils.py``, this module must NEVER import the views
module or the heavy SPICE-loading ephemeris module -- the latter triggers a ~1.6 GB SPICE
kernel download at module load (CLAUDE.md "Heavy import side effect", v2.2 milestone-locked
module-home constraint).

Field authority differs deliberately between the two branches (see
``_reconcile_container()``/``_reconcile_classical_nights()`` docstrings below): the container
branch is the sole writer of its key and is authoritative for every field on both create and
update, while the per-night branch only refreshes ``title``/``description``/``target_list`` on
update -- ``start_time``/``end_time``/``telescope`` are never rewritten after creation.
"""

import logging
from datetime import datetime, timedelta
from datetime import time as dt_time
from datetime import timezone as dt_timezone
from typing import Any, NamedTuple

from django.db.models import Q
from tom_calendar.models import CalendarEvent

from solsys_code.calendar_utils import (
    insert_or_create_calendar_event,
    preview_calendar_event_action,
    update_calendar_event_key_and_fields,
)
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import sun_event

logger = logging.getLogger(__name__)

RUN_URL_NAMESPACE = 'RUN:'

# Moved verbatim from campaign_views._RUN_STATUS_CALENDAR_PREFIX (D-01) -- must stay
# byte-identical to calendar_display_extras._TERMINAL_PREFIXES so the box-shadow status ring
# still applies. Public (no leading underscore) because the reconciler now owns titles, and
# test_campaign_approval.py asserts on these strings directly. Plan 29-04 deletes the
# campaign_views copy.
RUN_STATUS_CALENDAR_PREFIX = {
    CampaignRun.RunStatus.CANCELLED: '[CANCELLED]',
    CampaignRun.RunStatus.WEATHER_TECH_FAILURE: '[WEATHERED]',
}

# D-07: the reconciler branches purely on this set -- never a text heuristic over
# telescope_instrument/site_raw (26-DECISION.md's domain correction; see 29-RESEARCH.md
# Pitfall 1 for why the real dev-DB rows do not yet reflect this split -- that is a data-fix
# task, not a code task).
QUEUE_SOURCES = frozenset({CampaignRun.Source.LCO_QUEUE, CampaignRun.Source.GEMINI_QUEUE})


class ReconcileResult(NamedTuple):
    """Outcome of one ``reconcile_run()`` call.

    ``skipped_reason is None`` is the successor to ``_project_calendar_event()``'s bool
    return: ``_resolve_site()`` (plan 29-04) uses it to pick between its two success
    messages (D-04).
    """

    created: int = 0
    updated: int = 0
    unchanged: int = 0
    blocked: int = 0
    skipped_reason: str | None = None


def run_container_url(run: CampaignRun) -> str:
    """The bare whole-window container key (queue/class-wide/satellite branches, RECON-02/03)."""
    return f'{RUN_URL_NAMESPACE}{run.pk}'


def run_night_url(run: CampaignRun, night) -> str:
    """The per-night classical key -- always date-bearing, including a single-night run.

    26-DECISION.md's "Criterion 3 / SPIKE-03" locks the classical form as
    ``RUN:{run_pk}:{date}`` and the bare form as the queue/container family; this is a
    deliberate divergence from the ported ``_project_calendar_event()`` code (which used
    the bare key when ``n_nights == 1``), so the key form alone says which family an event
    belongs to. ``night`` must be the site-local observing night (the same night
    ``sun_event()``'s sunset is computed for), never the naive UTC date.
    """
    return f'{RUN_URL_NAMESPACE}{run.pk}:{night.isoformat()}'


def owned_events(run: CampaignRun):
    """Every ``CalendarEvent`` keyed in this run's ``RUN:`` namespace (identity check).

    The trailing colon on the ``startswith`` prefix is required: without it, run pk=3 also
    matches run pk=34's per-night events.
    """
    container_url = run_container_url(run)
    return CalendarEvent.objects.filter(Q(url=container_url) | Q(url__startswith=f'{container_url}:'))


def event_title(run: CampaignRun) -> str:
    """Byte-identical to ``campaign_views._calendar_event_title()``'s cancelled/weathered
    output, so ``calendar_display_extras``' terminal-prefix ring keeps applying."""
    base = f'{run.campaign.name}: {run.telescope_instrument}'
    if run.window_start != run.window_end:
        base = f'{base} (window {run.window_start}..{run.window_end})'
    prefix = RUN_STATUS_CALENDAR_PREFIX.get(run.run_status)
    if prefix:
        return f'{prefix} {base}'
    return base


def event_description(run: CampaignRun) -> str:
    """Appends the run-status line ``campaign_views._set_run_status()`` already writes,
    only when a status prefix applies to this run's current ``run_status``."""
    if run.run_status in RUN_STATUS_CALENDAR_PREFIX:
        return f'{run.observation_details}\nRun status: {run.get_run_status_display()}'
    return run.observation_details


def _skip_reason(run: CampaignRun) -> str | None:
    """Stage-0 guard (D-05's itemized skip vocabulary), evaluated in this order.

    Preserves today's exact "no event yet" cases from ``_project_calendar_event()``, plus
    the new approval gate (an unapproved web submission must never reach the calendar).
    """
    if run.approval_status != CampaignRun.ApprovalStatus.APPROVED:
        return 'not approved'
    if not run.telescope_instrument:
        return 'missing telescope/instrument'
    if run.window_start is None or run.window_end is None:
        return 'TBD window'
    if run.site is None and not run.telescope_class:
        return 'unresolved site'
    return None


def _may_write(event: CalendarEvent | None, run: CampaignRun) -> bool:
    """RECON-05's ownership rule -- the first condition checked in every write path.

    Returns True when ``event`` is None. Otherwise looks up this event's
    ``CalendarEventMeta`` companion row: when it exists and its ``run`` is set, ownership is
    exact-match only; when there is no companion row or its ``run`` is unset, this run may
    still write it if the event's ``url`` already lives in this run's ``RUN:`` namespace.
    Everything else returns False -- a hand-created entry, a conference, a proposal deadline
    or an un-attributed sync-command event is never created, modified or deleted.
    """
    if event is None:
        return True
    meta = CalendarEventMeta.objects.filter(event=event).first()
    if meta is not None and meta.run_id is not None:
        return meta.run_id == run.pk
    container_url = run_container_url(run)
    return event.url == container_url or event.url.startswith(f'{container_url}:')


def _link_event_to_run(event: CalendarEvent, run: CampaignRun) -> None:
    """Writer WR-03 (27-REVIEW.md): set/keep ``CalendarEventMeta.run``, nothing else.

    Never writes ``is_verified``, ``confirmed_by`` or ``confirmed_at`` -- an adopted row's
    telescope-label verification history and Phase 28 attribution audit must survive
    untouched.
    """
    meta, _created = CalendarEventMeta.objects.get_or_create(event=event)
    if meta.run_id != run.pk:
        meta.run = run
        meta.save(update_fields=['run'])


def _reconcile_container(run: CampaignRun, *, dry_run: bool) -> ReconcileResult:
    """The whole-window branch shared by queue-scheduled, class-wide and satellite runs
    (RECON-02 queue half, RECON-03).

    The container is the ONLY writer of the bare ``RUN:{pk}`` key, so it is authoritative
    for every field on both create and update -- its span must track window edits.
    """
    url = run_container_url(run)
    fields: dict[str, Any] = {
        'title': event_title(run),
        'description': event_description(run),
        'target_list': run.campaign,
        'telescope': run.telescope_instrument,
        'start_time': datetime.combine(run.window_start, dt_time(0, 0), tzinfo=dt_timezone.utc),
        'end_time': datetime.combine(run.window_end, dt_time(23, 59), tzinfo=dt_timezone.utc),
    }
    existing = CalendarEvent.objects.filter(url=url).first()
    if not _may_write(existing, run):
        logger.warning('Reconcile blocked: event pk=%s is not owned by run pk=%s.', existing.pk, run.pk)
        return ReconcileResult(blocked=1)

    if dry_run:
        action = preview_calendar_event_action(existing, fields)
        return ReconcileResult(**{action: 1})

    if existing is None:
        event, action = insert_or_create_calendar_event({'url': url}, fields=fields)
    else:
        event, action = update_calendar_event_key_and_fields(existing, url, fields)
    _link_event_to_run(event, run)
    return ReconcileResult(**{action: 1})


def _reconcile_classical_nights(run: CampaignRun, *, dry_run: bool) -> ReconcileResult:
    """The per-night branch (RECON-02 classical half).

    Ports ``_project_calendar_event()``'s ground loop: iterates every night in
    ``[window_start, window_end]`` inclusive, calling ``sun_event(run.site, night,
    kind='sun')`` (never ``kind='dark'``) for the dip-corrected sunset/sunrise. Per D-06,
    the ``ValueError`` ``sun_event()`` raises (e.g. a blank ``Observatory.timezone``) is
    NOT caught here -- it propagates uncaught out of ``reconcile_run()`` so the batch loop
    (plan 29-03) and the staff-action call sites (plan 29-04) can each apply their own
    already-differentiated handling.

    Field authority deliberately differs from the container branch: on **create**, this
    writes ``title``, ``description``, ``target_list``, ``telescope``, ``start_time``,
    ``end_time``; on **update of an event that already exists**, it writes only ``title``,
    ``description`` and ``target_list``. ``start_time``, ``end_time`` and ``telescope`` are
    never rewritten after creation, for two reasons: (a) a night adopted from
    ``load_telescope_runs`` (plan 29-02's D-02 step) carries that command's file-derived
    BoN/EoN window, which is more precise than this coarse sunset-to-sunrise span and must
    not be overwritten; (b) ``load_telescope_runs`` keys its own idempotent lookup on
    ``(telescope, instrument, start_time +/- 5 min)``, so moving ``start_time`` off the
    value it keys on would make its next re-ingest create a second event for the same
    night. Refreshing ``title``/``description`` is still required so a ``mark_cancelled``/
    ``mark_weather_failure`` decision reaches this run's events, exactly as
    ``_set_run_status()`` does today. Accepted consequence: for an adopted night,
    ``title``/``description`` alternate between this module and ``load_telescope_runs`` on
    re-ingest -- key stability, not field-write exclusivity, is what D-02's churn analysis
    guaranteed.
    """
    totals = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0}
    n_nights = (run.window_end - run.window_start).days + 1
    for i in range(n_nights):
        night = run.window_start + timedelta(days=i)
        sunset, sunrise = sun_event(run.site, night, kind='sun')
        url = run_night_url(run, night)
        existing = CalendarEvent.objects.filter(url=url).first()
        if not _may_write(existing, run):
            logger.warning('Reconcile blocked: event pk=%s is not owned by run pk=%s.', existing.pk, run.pk)
            totals['blocked'] += 1
            continue

        common_fields: dict[str, Any] = {
            'title': event_title(run),
            'description': event_description(run),
            'target_list': run.campaign,
        }
        if existing is None:
            fields = {
                **common_fields,
                'telescope': run.telescope_instrument,
                'start_time': sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0),
                'end_time': sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0),
            }
        else:
            fields = common_fields

        if dry_run:
            totals[preview_calendar_event_action(existing, fields)] += 1
            continue

        if existing is None:
            event, action = insert_or_create_calendar_event({'url': url}, fields=fields)
        else:
            event, action = update_calendar_event_key_and_fields(existing, url, fields)
        _link_event_to_run(event, run)
        totals[action] += 1

    return ReconcileResult(**totals)


def reconcile_run(run: CampaignRun, *, dry_run: bool = False) -> ReconcileResult:
    """The public D-03 entry point: implements all of this run's calendar projection.

    Safe to call redundantly (idempotent), as defence-in-depth for the staff-action call
    sites (plan 29-04) -- calling it twice against unchanged run state must report
    ``unchanged`` the second time and write nothing (RECON-01).

    Args:
        run: the ``CampaignRun`` to reconcile.
        dry_run: when True, report what would change without writing anything.

    Returns:
        ReconcileResult: ``skipped_reason`` set (all counts 0) when the stage-0 guard
            fires; otherwise the outcome of whichever branch this run dispatches to.
    """
    reason = _skip_reason(run)
    if reason is not None:
        return ReconcileResult(skipped_reason=reason)

    if run.telescope_class:
        # RECON-03: a class-wide allocation (2m0/1m0/0m4) or a SPACE-classed run shares
        # this branch -- the whole-window math is identical either way (RESEARCH.md
        # Assumption A2, resolved in favour of one branch, not two).
        return _reconcile_container(run, dry_run=dry_run)
    if run.site is not None and run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        # The ported satellite case: no fixed horizon, so no per-night sun_event() math.
        return _reconcile_container(run, dry_run=dry_run)
    if run.source in QUEUE_SOURCES:
        # RECON-02 queue half, D-07: branch purely on source, never a text heuristic.
        return _reconcile_container(run, dry_run=dry_run)
    return _reconcile_classical_nights(run, dry_run=dry_run)
