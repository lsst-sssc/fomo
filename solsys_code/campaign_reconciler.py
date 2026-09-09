"""Pure-logic reconciler core (D-01/D-03, 29-CONTEXT.md).

Projects and refreshes a ``CampaignRun``'s calendar events as a function of the run's own
state, computed by one idempotent per-run function (``reconcile_run()``) shared by the batch
command (plan 29-03) and the four staff-action call sites (plan 29-04) -- this is the only
way RECON-01's "running it a second time changes nothing" and RECON-08's "a staff decision
reconciles immediately" can be guaranteed to agree with each other.

Two coexisting key families (26-DECISION.md "Criterion 3 / SPIKE-03"): a bare
``RUN:{run_pk}`` whole-window container for class-wide and satellite/space runs
(RECON-02 queue half, RECON-03), and a date-bearing ``RUN:{run_pk}:{date}`` key per
observing night for every other approved, windowed run with a resolved ground site --
queue-scheduled or classically-scheduled alike (RECON-02 classical half) -- always
date-bearing, including for a single-night run, so the key form alone says which family
an event belongs to. What decides "no fixed observing site" is a non-blank
``telescope_class`` or a satellite ``site`` -- never the run's ``source`` field, which is
provenance only and does not change an event's window shape (corrected by quick task
``260805-tad``, 2026-08-05, after the field it originally read only ever fired for a
run that already had a resolved, non-satellite site -- see 29-CONTEXT.md D-07's dated
forward-pointer). ``reconcile_run()`` re-derives which family a run belongs to from its
*current* state on every call, so a re-classification (an admin correction to
``telescope_class``/``site`` on an already-reconciled run) is detected and converged on:
``_detach_stale_family_events()`` detaches (never deletes) any event left over from the
family the run no longer belongs to, back into Phase 28's attribution queue
(29-REVIEW.md CR-01).

Like ``campaign_gap.py``/``campaign_utils.py``, this module must NEVER import the views
module or the heavy SPICE-loading ephemeris module -- the latter triggers a ~1.6 GB SPICE
kernel download at module load (CLAUDE.md "Heavy import side effect", v2.2 milestone-locked
module-home constraint).

D-17 (Phase 33): a set ``CalendarEventMeta.run`` means the event is ATTRIBUTED to that run,
never that the run OWNS it -- what this module owns is the ``RUN:`` key namespace, and
namespace identity is exactly what ``owned_events()``/``writable_events()`` express. An
attributed event outside that namespace is read-only from this module's point of view: it
informs the skip-the-night rule in ``_reconcile_classical_nights()`` (D-01) but is never
created, modified, re-keyed or deleted here.

Field authority differs deliberately between the two branches (see
``_reconcile_container()``/``_reconcile_classical_nights()`` docstrings below): the container
branch is the sole writer of its key and is authoritative for every field on both create and
update, while the per-night branch only refreshes ``title``/``description``/``target_list`` on
update -- ``start_time``/``end_time``/``telescope``/``instrument`` are never rewritten after
creation.
"""

import logging
import re
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


class ReconcileResult(NamedTuple):
    """Outcome of one ``reconcile_run()`` call.

    ``skipped_reason is None`` is this module's success signal: ``_resolve_site()``
    (plan 29-04) uses it to pick between its two success messages (D-04).
    """

    created: int = 0
    updated: int = 0
    unchanged: int = 0
    blocked: int = 0
    skipped_nights: int = 0
    """Classical nights left untouched because a non-``RUN:`` event is already attributed
    to this run for that night (D-01, ANNOT-01) -- the reconciler wrote nothing for them."""
    detached: int = 0
    """Companion rows whose attribution the reconciler cleared during the stale/superseded
    detach step (CR-03, 33-REVIEW.md WR-03) -- including the confirmation stamps
    (``confirmed_by``/``confirmed_at``) that went with them. Includes both a re-classified
    run's old-family events (29-REVIEW.md CR-01) and a classical night that became
    attributed through a non-``RUN:`` event after this reconciler had already minted its
    own event for it: that superseded event is detached, never deleted, back into Phase
    28's attribution queue."""
    skipped_reason: str | None = None


def run_container_url(run: CampaignRun) -> str:
    """The bare whole-window container key (class-wide/satellite branches only, RECON-02/03)."""
    return f'{RUN_URL_NAMESPACE}{run.pk}'


def run_night_url(run: CampaignRun, night) -> str:
    """The per-night classical key -- always date-bearing, including a single-night run.

    26-DECISION.md's "Criterion 3 / SPIKE-03" locks the classical form as
    ``RUN:{run_pk}:{date}`` and the bare form as the class-wide/satellite container
    family; this is a deliberate divergence from the retired pre-reconciler projection
    helper in ``campaign_views`` (which used the bare key when ``n_nights == 1``), so the
    key form alone says which family an event belongs to. ``night`` must be the
    site-local observing night (the same night ``sun_event()``'s sunset is computed for),
    never the naive UTC date.
    """
    return f'{RUN_URL_NAMESPACE}{run.pk}:{night.isoformat()}'


def owned_events(run: CampaignRun):
    """Every ``CalendarEvent`` keyed in this run's ``RUN:`` namespace (identity check).

    The trailing colon on the ``startswith`` prefix is required: without it, run pk=3 also
    matches run pk=34's per-night events.
    """
    container_url = run_container_url(run)
    return CalendarEvent.objects.filter(Q(url=container_url) | Q(url__startswith=f'{container_url}:'))


def writable_events(run: CampaignRun):
    """The queryset-level twin of ``_may_write()`` -- T-29-19: namespace identity alone is
    NOT ownership.

    A companion row that points at a DIFFERENT run means a staff member attributed that
    event elsewhere via Phase 28's queue, and that attribution outranks a ``url`` string left
    over from an earlier keying -- so a write path must never touch it. Narrows
    ``owned_events(run)`` to the rows this run may actually write: no companion row at all,
    a companion row whose ``run`` is unset, or a companion row that already points at this
    run.

    ``owned_events()`` (namespace identity) remains the right query for read-only inspection
    and counting -- it stays unchanged for its existing consumers (``test_campaign_approval.py``
    and the demo notebook). Every write path (reconcile's detach step, the run-deletion
    cascade) must go through ``writable_events()`` instead.
    """
    return owned_events(run).filter(
        Q(telescope_label_meta__isnull=True)
        | Q(telescope_label_meta__run__isnull=True)
        | Q(telescope_label_meta__run=run)
    )


def _split_telescope_instrument(text: str) -> tuple[str, str]:
    """Splits a ``CampaignRun.telescope_instrument`` free-text value into its telescope and
    instrument halves.

    Submitters write this field as ``<telescope>/<instrument>`` or
    ``<telescope>+<instrument>``. Splits on the FIRST ``/`` or ``+`` delimiter (``maxsplit=1``,
    so ``'A/B/C'`` returns ``('A', 'B/C')``, never splitting a second time), and strips
    whitespace from both halves.

    When no delimiter is present, the whole string is returned as the telescope half with a
    blank instrument -- the safe fallback, since guessing which part of an un-delimited
    value like ``'NTT EFOSC2'`` is the telescope and which is the instrument would be
    inventing structure that isn't there. A leading delimiter (``'/MuSCAT3'``) therefore
    yields a blank telescope half by the same logic -- deliberately not special-cased,
    because ``_skip_reason()`` already rejects a wholly blank ``telescope_instrument`` and a
    second fallback rule here would be un-asked-for behaviour.

    Args:
        text: the run's ``telescope_instrument`` value.

    Returns:
        tuple[str, str]: ``(telescope, instrument)``.
    """
    parts = re.split(r'[/+]', text, maxsplit=1)
    if len(parts) == 1:
        return text.strip(), ''
    telescope, instrument = parts
    return telescope.strip(), instrument.strip()


def event_title(run: CampaignRun) -> str:
    """No longer embeds a campaign label, with or without a campaign (D-12, Phase 33): the
    campaign an event is attributed to is rendered from ``CalendarEventMeta.run`` at display
    time by ``calendar_display_extras.campaign_decoration()`` instead -- this is the single
    campaign label now, for every attributed event, ``RUN:`` or not. Must keep the terminal
    cancelled/weathered prefix form (``RUN_STATUS_CALENDAR_PREFIX``) that
    ``calendar_display_extras``' terminal-prefix ring matches on, so a cancelled/weathered
    run's event still gets the status ring.
    """
    base = run.telescope_instrument
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

    Preserves today's exact "no event yet" cases from the retired pre-reconciler
    projection helper in ``campaign_views``, plus
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

    Setting ``run`` here records ATTRIBUTION (D-17), not ownership -- this module's own
    ``RUN:``-keyed events happen to be self-attributed this way so the display-time
    decoration path (``calendar_display_extras.campaign_decoration()``) covers them too.
    Never writes ``is_verified``, ``confirmed_by`` or ``confirmed_at`` -- an already-linked
    row's telescope-label verification history and Phase 28 attribution audit must survive
    untouched.
    """
    meta, _created = CalendarEventMeta.objects.get_or_create(event=event)
    if meta.run_id != run.pk:
        meta.run = run
        meta.save(update_fields=['run'])


def _reconcile_container(run: CampaignRun, *, dry_run: bool) -> ReconcileResult:
    """The whole-window branch shared by class-wide and satellite runs
    (RECON-02 queue half, RECON-03) -- a run's ``source`` field never selects this branch.

    The container is the ONLY writer of the bare ``RUN:{pk}`` key, so it is authoritative
    for every field on both create and update -- its span must track window edits.
    """
    url = run_container_url(run)
    telescope, instrument = _split_telescope_instrument(run.telescope_instrument)
    fields: dict[str, Any] = {
        'title': event_title(run),
        'description': event_description(run),
        'target_list': run.campaign,
        'telescope': telescope,
        'instrument': instrument,
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


def _observing_night(start_time: datetime, site_zone: ZoneInfo):
    """The site-local observing night a ``start_time`` belongs to, anchored at local noon.

    This is the same anchor ``telescope_runs._local_noon_utc()`` uses:
    ``sun_event(site, date)`` computes sunset for the EVENING of ``date``, so the observing
    night runs from local noon of ``date`` through local noon of ``date + 1``. Converting
    ``start_time`` into ``site_zone`` and subtracting twelve hours before taking ``.date()``
    maps any local time from noon through noon-plus-24-hours onto the date the night
    started on -- in particular, a 02:00 local start belongs to the PREVIOUS date's night,
    not the date its own naive site-local ``.date()`` would name.

    This supersedes 26-DECISION.md D-10's plain site-local ``.date()`` derivation, which is
    correct only for a start before local midnight (CR-02, 33-REVIEW.md): D-10's measured
    comparison called event ``pk=54`` (``2026-07-08T14:08:19Z``, Sydney, 00:08 local on
    2026-07-09) a 2026-07-09 night; under this anchor it is 2026-07-08 -- the night whose
    sunset the run was actually scheduled against.

    Forward-pointer: Phase 34/35 should promote this to a shared public helper next to
    ``sun_event()`` when the observation projector needs the same event-to-night mapping.

    Args:
        start_time: an event's ``start_time`` (timezone-aware).
        site_zone: the run's site timezone.

    Returns:
        date: the site-local observing night ``start_time`` belongs to.
    """
    local = start_time.astimezone(site_zone)
    return (local - timedelta(hours=12)).date()


def _attributed_nights(run: CampaignRun, site_zone: ZoneInfo) -> set:
    """The set of site-local observing nights already covered by an attributed non-``RUN:``
    event (D-01, ANNOT-01): a night with an attributed non-``RUN:`` event has no reconciler
    event -- the same rule Phase 35's allocation handoff will use.

    Runs ONE query, called once per ``_reconcile_classical_nights()`` call, before the
    per-night loop is entered (33-REVIEWS.md Agreed Concern 5): a predicate called inside
    ``for i in range(n_nights)`` would issue one ORM round-trip per night, and a multi-week
    allocation would pay that cost for every night of the window.

    Carries NO blank-url restriction, unlike the retired per-night adopt helper this
    supersedes: a facility-URL-keyed attributed event (a Phase 34 observation event) must
    match too -- any url outside the ``RUN:`` namespace counts, blank or not. That relaxed
    restriction is what made ``.date()``'s post-local-midnight defect (CR-02) reachable:
    the retired helper's blank-url-only scope meant every matching event started at
    beginning-of-night (before local midnight), so the derivation error never fired.
    ``_observing_night()`` closes that gap.

    Args:
        run: the ``CampaignRun`` being reconciled.
        site_zone: the run's site timezone, built once by the caller.

    Returns:
        set: the site-local observing ``date``s already covered by an attributed
        non-``RUN:`` event.
    """
    metas = (
        CalendarEventMeta.objects.filter(run_id=run.pk)
        .exclude(event__url__startswith=RUN_URL_NAMESPACE)
        .select_related('event')
    )
    return {_observing_night(meta.event.start_time, site_zone) for meta in metas}


def _reconcile_classical_nights(run: CampaignRun, *, dry_run: bool) -> tuple[ReconcileResult, set[str]]:
    """The per-night branch (RECON-02 classical half).

    Ports the retired pre-reconciler projection helper's ground loop from
    ``campaign_views``: iterates every night in
    ``[window_start, window_end]`` inclusive, calling ``sun_event(run.site, night,
    kind='sun')`` (never ``kind='dark'``) for the dip-corrected sunset/sunrise. Per D-06,
    the ``ValueError`` ``sun_event()`` raises (e.g. a blank ``Observatory.timezone``) is
    NOT caught here -- it propagates uncaught out of ``reconcile_run()`` so the batch loop
    (plan 29-03) and the staff-action call sites (plan 29-04) can each apply their own
    already-differentiated handling.

    Per-night resolution order (D-01, ANNOT-01 -- retires the D-02 adopt/re-key contract;
    CR-03 fix, 33-REVIEW.md): (1) if the night is already attributed to this run through a
    non-``RUN:`` event (``_attributed_nights()``), skip the night entirely -- no event is
    created, modified or re-keyed for it, only the ``skipped_nights`` counter moves. This is
    now UNCONDITIONAL on whether a ``RUN:{pk}:{date}`` event already exists for that night,
    so the reconcile-then-attribute ordering and the attribute-then-reconcile ordering
    converge on the same result: a night that becomes attributed after this reconciler
    already minted its own event for it drops that event's url out of the returned active-url
    set, so ``_detach_stale_family_events()`` reclaims it. (2) otherwise, an event already
    keyed at ``run_night_url(run, night)`` -- the common idempotent-rerun case; (3) otherwise,
    mint a new event. ``_may_write()`` remains the first condition checked on every write --
    resolution step (2) still goes through it before any write (RECON-05 defence in depth;
    see T-29-05).

    Field authority deliberately differs from the container branch: on **create**, this
    writes ``title``, ``description``, ``target_list``, ``telescope``, ``instrument``,
    ``start_time``, ``end_time``; on **update of an event that already exists at its own
    ``RUN:`` key**, it writes only ``title``, ``description`` and ``target_list`` --
    ``start_time``, ``end_time``, ``telescope`` and ``instrument`` are never rewritten after
    creation. Refreshing ``title``/``description`` is still required so a
    ``mark_cancelled``/``mark_weather_failure`` decision reaches this run's events, exactly
    as ``_set_run_status()`` does today.

    Returns:
        tuple[ReconcileResult, set[str]]: the outcome, and the exact set of
        ``CalendarEvent.url`` values this branch considers current -- every night that is
        NOT skipped, including a blocked night and every night visited in ``dry_run``. This
        is what lets ``reconcile_run()`` detach a superseded night's own event instead of
        re-deriving the whole window a second time (IN-02, 33-REVIEW.md).
    """
    totals = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0, 'skipped_nights': 0}
    n_nights = (run.window_end - run.window_start).days + 1
    site_zone = ZoneInfo(run.site.timezone)
    attributed_nights = _attributed_nights(run, site_zone)
    active_urls: set[str] = set()
    for i in range(n_nights):
        night = run.window_start + timedelta(days=i)
        url = run_night_url(run, night)

        if night in attributed_nights:
            totals['skipped_nights'] += 1
            continue

        active_urls.add(url)
        sunset, sunrise = sun_event(run.site, night, kind='sun')
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
            telescope, instrument = _split_telescope_instrument(run.telescope_instrument)
            fields = {
                **common_fields,
                'telescope': telescope,
                'instrument': instrument,
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

    return ReconcileResult(**totals), active_urls


def _detach_stale_family_events(run: CampaignRun, active_urls: set[str]) -> int:
    """Convergence step (29-REVIEW.md CR-01, user-directed fix: DETACH, not delete or
    flag-only).

    ``reconcile_run()`` dispatches a run to exactly one of the two mutually-exclusive key
    families (bare ``RUN:{pk}`` container vs. date-bearing ``RUN:{pk}:{date}`` per-night)
    based on the run's *current* ``telescope_class``/``site`` state (never its ``source``
    field, corrected by quick task ``260805-tad`` -- see 29-CONTEXT.md D-07's dated
    forward-pointer). Nothing else detects a re-classification (an admin correction to one
    of those fields on an already-reconciled run): without this, the OLD family's events
    would either be silently orphaned forever -- no code path ever revisits them again --
    or, worse, silently miscounted as belonging to a family they no longer match.

    A second case (CR-03, 33-REVIEW.md): a classical night that has become attributed
    through a non-``RUN:`` event drops out of ``active_urls`` (see
    ``_reconcile_classical_nights()``), so its reconciler-minted event is detached back
    into Phase 28's queue here too, rather than lingering as a second attributed entry for
    the same night.

    Detaching -- rather than deleting the ``CalendarEvent`` rows outright, or merely
    logging/flagging -- returns them to Phase 28's attribution queue for a human to
    re-confirm or discard, matching every other unattributed row's meaning (D-17: an unset
    ``run`` means "not attributed to any CampaignRun" -- never "touch me"). Plan 33-04
    (D-16) routes this through the shared :func:`~solsys_code.campaign_utils.
    unlink_event_from_run` helper -- the single writer of what clearing an attribution
    means -- rather than this module's own ad-hoc update, which is a deliberate behaviour
    change: the helper also clears ``confirmed_by``/``confirmed_at``, which this step did
    not do before. A detached row that kept "confirmed by X at T" was displaying a
    confirmation for an attribution that no longer exists.

    The extra ``run=run`` filter term (T-29-19) is not redundant, and the helper preserves
    it: without it, a stale-family event that staff have since re-attributed to a DIFFERENT
    run gets its confirmed attribution silently cleared by a reconcile of the run whose
    namespace the url happens to carry. It also loses nothing -- rows with ``run`` already
    unset are a no-op, and rows with no companion row were never in the queryset.

    Args:
        run: the ``CampaignRun`` just reconciled.
        active_urls: the exact set of ``CalendarEvent.url`` values the branch just run
            considers current for this run (one container url, or one url per night).

    Returns:
        int: the number of companion rows actually cleared (WR-03, 33-REVIEW.md) --
        including the confirmation stamps that went with them.
    """
    # Local import to avoid a circular import at module load time: campaign_utils.py
    # imports reconcile_run/ReconcileResult from this module at its own top level, so a
    # top-level import here of campaign_utils would deadlock on whichever module Python
    # loads first.
    from solsys_code.campaign_utils import unlink_event_from_run

    stale = owned_events(run).exclude(url__in=active_urls)
    detached = unlink_event_from_run(stale, run)
    if detached:
        logger.warning(
            'Reconcile detached %s stale/superseded event(s) from run pk=%s; confirmation stamps cleared.',
            detached,
            run.pk,
        )
    return detached


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
    else:
        result, active_urls = _reconcile_classical_nights(run, dry_run=dry_run)

    # CR-01 convergence step: detach (never delete) any of this run's owned events left
    # over from a family it no longer belongs to, OR a classical night's event superseded
    # by a later attribution (CR-03, 33-REVIEW.md). A no-op whenever the run has not been
    # re-classified and no night has been superseded since its last reconcile (RECON-01
    # idempotency: every url this branch just wrote/confirmed/skipped-for is already in
    # active_urls, so exclude() finds nothing stale). Skipped entirely in dry_run --
    # detaching is a write, and dry_run must write nothing, so `detached` stays 0 there.
    detached = 0
    if not dry_run:
        detached = _detach_stale_family_events(run, active_urls)

    return result._replace(detached=detached)
