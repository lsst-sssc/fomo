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
alone says which family an event belongs to. ``reconcile_run()`` re-derives which family a
run belongs to from its *current* state on every call, so a re-classification (an admin
correction to ``source``/``telescope_class``/``site`` on an already-reconciled run) is
detected and converged on: ``_detach_stale_family_events()`` detaches (never deletes) any
event left over from the family the run no longer belongs to, back into Phase 28's
attribution queue (29-REVIEW.md CR-01).

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
from zoneinfo import ZoneInfo

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
#
# ESO_QUEUE added in plan 29-06 (explicit user-directed deviation, not part of this phase's
# original scope -- see 29-06-SUMMARY.md): the real 3I/ATLAS dev-DB data has ESO VLT rows
# (obscode 309) that 26-DECISION.md's own classification rule names as a shared-queue
# network alongside LCO/Gemini/SOAR, but CampaignRun.Source had no dedicated value for it.
QUEUE_SOURCES = frozenset({CampaignRun.Source.LCO_QUEUE, CampaignRun.Source.GEMINI_QUEUE, CampaignRun.Source.ESO_QUEUE})


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
    the new approval gate (an unapproved web submission must never reach the calendar), plus
    a ``window_end < window_start`` data-integrity guard (29-REVIEW.md WR-02): without it,
    ``_reconcile_classical_nights()``'s ``n_nights = (window_end - window_start).days + 1``
    goes non-positive and ``range(n_nights)`` silently iterates zero times -- no event, no
    skip reason, indistinguishable from an already-``unchanged`` run in the summary.
    """
    if run.approval_status != CampaignRun.ApprovalStatus.APPROVED:
        return 'not approved'
    if not run.telescope_instrument:
        return 'missing telescope/instrument'
    if run.window_start is None or run.window_end is None:
        return 'TBD window'
    if run.window_end < run.window_start:
        return 'window_end before window_start'
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


def _adopted_event_for_night(run: CampaignRun, night, site_zone: ZoneInfo) -> CalendarEvent | None:
    """D-02's adopt step: find a classical night already attributed to this run.

    Queries ``CalendarEventMeta.objects.filter(run_id=run.pk)`` -- rows Phase 28's
    attribution queue writes once a staff member confirms a ``load_telescope_runs``-created
    (blank-``url``) event belongs to this run -- and returns the first candidate whose
    ``start_time``, converted into ``site_zone`` and reduced to ``.date()``, equals
    ``night``. ``site_zone`` is a ``ZoneInfo`` built once per run (from ``run.site.timezone``)
    by the caller, not once per night. Ordered by ``event__start_time`` so the pick is
    deterministic in the (should-never-happen) case that two candidates share a night.
    Returns None when nothing matches -- the caller then falls through to minting.

    Matches on ``CalendarEventMeta.run_id`` plus the site-local night ONLY (RESEARCH.md
    Assumption A3, resolved explicitly here): this deliberately does NOT also filter the
    candidate's stored telescope field against the run's telescope_instrument. The
    ``CalendarEventMeta.run`` FK is already an explicit, human-confirmed attribution
    statement written by Phase 28's queue
    -- layering a free-text telescope comparison on top of an already-confirmed link can
    only cause a miss (a ``CampaignRun.telescope_instrument`` string like ``'FTN/MuSCAT3'``
    need not equal a ``load_telescope_runs`` ``telescope`` value like ``'FTN'``), and a miss
    silently contradicts D-02 by falling through to minting a duplicate event for a night
    that already has one.

    ``event__url=''`` (29-REVIEW.md CR-01): candidates are restricted to
    ``load_telescope_runs``-created, blank-``url`` events, matching this docstring's own
    stated intent. Without this filter, a run's own STALE bare container event from a prior
    reconcile under the other key family (see ``_detach_stale_family_events()``) also
    carries ``CalendarEventMeta.run == run`` and would match here, "adopting" it into a
    per-night slot -- re-keying its ``url`` while leaving its ``start_time``/``end_time`` at
    the original whole-window span, since neither field is rewritten on adopt. That silently
    produced a calendar entry keyed and titled as one observing night but timed as the
    entire original multi-night window.

    Why re-keying the matched event's ``url`` is safe for the other writer:
    ``load_telescope_runs`` looks up its own events by ``(telescope, instrument, start_time
    +/- 5 min)`` and never reads or writes ``url`` at all, so moving the ``url`` here does
    not break its idempotent lookup -- this is exactly the asymmetry D-02 cites as the
    reason adopt (not gap-fill) was chosen for classical runs. Accepted consequence carried
    over from plan 29-01: ``title``/``description`` on an adopted night alternate between
    this module and ``load_telescope_runs`` on re-ingest -- key stability, not field-write
    exclusivity, is what D-02's churn analysis guaranteed.
    """
    candidates = (
        CalendarEventMeta.objects.filter(run_id=run.pk, event__url='')
        .select_related('event')
        .order_by('event__start_time')
    )
    for meta in candidates:
        event = meta.event
        if event.start_time.astimezone(site_zone).date() == night:
            return event
    return None


def _reconcile_classical_nights(run: CampaignRun, *, dry_run: bool) -> ReconcileResult:
    """The per-night branch (RECON-02 classical half).

    Ports ``_project_calendar_event()``'s ground loop: iterates every night in
    ``[window_start, window_end]`` inclusive, calling ``sun_event(run.site, night,
    kind='sun')`` (never ``kind='dark'``) for the dip-corrected sunset/sunrise. Per D-06,
    the ``ValueError`` ``sun_event()`` raises (e.g. a blank ``Observatory.timezone``) is
    NOT caught here -- it propagates uncaught out of ``reconcile_run()`` so the batch loop
    (plan 29-03) and the staff-action call sites (plan 29-04) can each apply their own
    already-differentiated handling.

    Per-night resolution order (D-02): (1) an existing event already keyed at
    ``run_night_url(run, night)`` -- the common idempotent-rerun case; (2) failing that,
    ``_adopted_event_for_night(...)`` -- a ``load_telescope_runs``-created event already
    attributed to this run for this night via Phase 28's confirmation queue, re-keyed in
    place; (3) failing both, mint a new event. Both (1) and (2) go through ``_may_write()``
    before any write -- the ownership rule stays the first condition checked (RECON-05
    defence in depth; see T-29-05).

    Field authority deliberately differs from the container branch: on **create**, this
    writes ``title``, ``description``, ``target_list``, ``telescope``, ``start_time``,
    ``end_time``; on **update of an event that already exists (including an adopted one)**,
    it writes only ``title``, ``description``, ``target_list`` and (for an adopted event)
    the new ``url`` -- ``start_time``, ``end_time`` and ``telescope`` are never rewritten
    after creation, for two reasons: (a) a night adopted from ``load_telescope_runs``
    carries that command's file-derived BoN/EoN window, which is more precise than this
    coarse sunset-to-sunrise span and must not be overwritten; (b) ``load_telescope_runs``
    keys its own idempotent lookup on ``(telescope, instrument, start_time +/- 5 min)``, so
    moving ``start_time`` off the value it keys on would make its next re-ingest create a
    second event for the same night. Refreshing ``title``/``description`` is still required
    so a ``mark_cancelled``/``mark_weather_failure`` decision reaches this run's events,
    exactly as ``_set_run_status()`` does today. Accepted consequence: for an adopted
    night, ``title``/``description`` alternate between this module and
    ``load_telescope_runs`` on re-ingest -- key stability, not field-write exclusivity, is
    what D-02's churn analysis guaranteed.
    """
    totals = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0}
    n_nights = (run.window_end - run.window_start).days + 1
    site_zone = ZoneInfo(run.site.timezone)
    for i in range(n_nights):
        night = run.window_start + timedelta(days=i)
        sunset, sunrise = sun_event(run.site, night, kind='sun')
        url = run_night_url(run, night)

        existing = CalendarEvent.objects.filter(url=url).first()
        adopting = False
        if existing is None:
            existing = _adopted_event_for_night(run, night, site_zone)
            adopting = existing is not None

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
            # An about-to-be-adopted night must report 'updated', not 'created' -- merge
            # the new url into the preview fields so preview_calendar_event_action() sees
            # the re-key, exactly as the real write below would.
            preview_fields = {**fields, 'url': url} if adopting else fields
            totals[preview_calendar_event_action(existing, preview_fields)] += 1
            continue

        if existing is None:
            event, action = insert_or_create_calendar_event({'url': url}, fields=fields)
        else:
            event, action = update_calendar_event_key_and_fields(existing, url, fields)
        _link_event_to_run(event, run)
        totals[action] += 1

    return ReconcileResult(**totals)


def _detach_stale_family_events(run: CampaignRun, active_urls: set[str]) -> None:
    """Convergence step (29-REVIEW.md CR-01, user-directed fix: DETACH, not delete or
    flag-only).

    ``reconcile_run()`` dispatches a run to exactly one of the two mutually-exclusive key
    families (bare ``RUN:{pk}`` container vs. date-bearing ``RUN:{pk}:{date}`` per-night)
    based on the run's *current* ``source``/``telescope_class``/``site`` state. Nothing else
    detects a re-classification (an admin correction to one of those fields on an
    already-reconciled run): without this, the OLD family's events would either be silently
    orphaned forever -- no code path ever revisits them again -- or, worse, accidentally
    "adopted" into the new family and corrupted (see ``_adopted_event_for_night()``).

    Detaching (``CalendarEventMeta.run = None``) -- rather than deleting the
    ``CalendarEvent`` rows outright, or merely logging/flagging -- returns them to Phase 28's
    attribution queue for a human to re-confirm or discard, matching every other
    "un-owned" row's meaning (module docstring: "unset ... never 'touch me'").

    A bulk ``.update()``, not a per-instance ``.save()`` loop, deliberately: this only ever
    clears a FK on rows already known to belong to this run, so there is no per-row business
    logic to run, and a bulk update avoids firing save-related signal handlers for every row.

    Args:
        run: the ``CampaignRun`` just reconciled.
        active_urls: the exact set of ``CalendarEvent.url`` values the branch just run
            considers current for this run (one container url, or one url per night).
    """
    stale = owned_events(run).exclude(url__in=active_urls)
    CalendarEventMeta.objects.filter(event__in=stale).update(run=None)


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
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    elif run.site is not None and run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        # The ported satellite case: no fixed horizon, so no per-night sun_event() math.
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    elif run.source in QUEUE_SOURCES:
        # RECON-02 queue half, D-07: branch purely on source, never a text heuristic.
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    else:
        result = _reconcile_classical_nights(run, dry_run=dry_run)
        n_nights = (run.window_end - run.window_start).days + 1
        active_urls = {run_night_url(run, run.window_start + timedelta(days=i)) for i in range(n_nights)}

    # CR-01 convergence step: detach (never delete) any of this run's owned events left
    # over from a family it no longer belongs to. A no-op whenever the run has not been
    # re-classified since its last reconcile (RECON-01 idempotency: every url this branch
    # just wrote/confirmed is already in active_urls, so exclude() finds nothing stale).
    # Skipped entirely in dry_run -- detaching is a write, and dry_run must write nothing.
    if not dry_run:
        _detach_stale_family_events(run, active_urls)

    return result
