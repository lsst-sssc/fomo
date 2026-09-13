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
informs the skip-the-night rule in ``_attributed_nights()`` (D-01) but is never created,
modified, re-keyed or deleted here.

Field authority differs deliberately between this module's own container branch and the
per-night allocation branch it now dispatches to (D-09, Phase 35): the container branch is
the sole writer of its key and is authoritative for every field on both create and update,
while ``allocation_projector.project_allocation()`` only refreshes
``title``/``description``/``target_list`` on update -- ``start_time``/``end_time``/
``telescope``/``instrument`` are never rewritten after creation.
"""

import logging
import re
from datetime import datetime
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
from solsys_code.telescope_runs import observing_night

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
    """Companion rows the sweep actually released during the stale/superseded detach step
    (CR-03, 33-REVIEW.md WR-03). By construction (Task 1, 33-10) a released row never had
    ``confirmed_by`` set -- a human-confirmed attribution is never counted here, see
    ``detach_declined`` instead. Includes both a re-classified run's old-family events
    (29-REVIEW.md CR-01) and a classical night that became attributed through a non-``RUN:``
    event after this reconciler had already minted its own event for it: that superseded
    event is detached, never deleted, back into Phase 28's attribution queue."""
    detach_declined: int = 0
    """Superseded or stale companion rows the sweep deliberately did NOT release because
    ``confirmed_by`` is set (UAT decision, 2026-09-09 ``## Decisions``: option B -- a human
    decision outranks an automated sweep). Exists so an operator is told a release was
    declined rather than left to infer it from an unchanged ``detached`` -- silence and
    'nothing to release' are otherwise indistinguishable."""
    retired: int = 0
    """Allocation nights deleted because a linked record's placed or observed block now
    occupies them (D-05/D-07, Phase 35), or because a re-classification left them out of
    this reconcile's active set (D-14)."""
    rekeyed: int = 0
    """Legacy ``RUN:{pk}:{night}`` events re-keyed in place into the ``ALLOC:`` namespace,
    keeping their primary key, start_time and end_time (D-16, Phase 35)."""
    skipped_reason: str | None = None


def run_container_url(run: CampaignRun) -> str:
    """The bare whole-window container key (class-wide/satellite branches only, RECON-02/03)."""
    return f'{RUN_URL_NAMESPACE}{run.pk}'


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


def split_telescope_instrument(text: str) -> tuple[str, str]:
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
    a ``window_end < window_start`` data-integrity guard (29-REVIEW.md WR-02): without it, a
    per-night branch's ``n_nights = (window_end - window_start).days + 1`` goes non-positive
    and a range-based per-night loop silently iterates zero times -- no event, no skip
    reason, indistinguishable from an already-``unchanged`` run in the summary.
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
    telescope, instrument = split_telescope_instrument(run.telescope_instrument)
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


def _attributed_nights(run: CampaignRun, site_zone: ZoneInfo) -> set:
    """The set of site-local observing nights already covered by an attributed non-``RUN:``
    event (D-01, ANNOT-01): a night with an attributed non-``RUN:`` event has no reconciler
    event -- the same rule Phase 35's allocation handoff uses.

    Runs ONE query, called once per caller (33-REVIEWS.md Agreed Concern 5): a predicate
    called inside a per-night loop would issue one ORM round-trip per night, and a
    multi-week allocation would pay that cost for every night of the window.

    Carries NO blank-url restriction, unlike the retired per-night adopt helper this
    supersedes: a facility-URL-keyed attributed event (a Phase 34 observation event) must
    match too -- any url outside the ``RUN:`` namespace counts, blank or not. That relaxed
    restriction is what made a plain ``.date()``'s post-local-midnight defect (CR-02)
    reachable: the retired helper's blank-url-only scope meant every matching event started
    at beginning-of-night (before local midnight), so the derivation error never fired.
    ``observing_night()`` closes that gap.

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
    return {observing_night(meta.event.start_time, site_zone) for meta in metas}


def _stale_attributions(run: CampaignRun, active_urls: set[str]) -> tuple[list[int], int]:
    """Read-only split of this run's stale/superseded owned events into what an automated
    sweep may release and what it must leave alone (33-UAT.md ``## Decisions``, 2026-09-09):
    *"Option B -- human outranks machine. The reconciler sweep detaches only rows with no
    confirmed_by; a human-confirmed attribution is never cleared by an automated sweep."*

    This closes the CR-04 confirm/erase loop: ``campaign_attribution.orphan_calendar_events()``
    re-offers a detached row to the very run that released it at HIGH band the moment a
    staff member re-confirms the obvious match, and -- before this guard -- the very next
    unattended sweep erased that confirmation again, with only a ``logger.warning`` as a
    record. That contradicted the phase goal ("the reconciler annotates instead of owning"),
    ANNOT-01 and ``unlink_event_from_run()``'s own "a human attribution always outranks an
    automated clear".

    The guard deliberately lives HERE, on the reconciler side, and NOT in
    :func:`~solsys_code.campaign_utils.unlink_event_from_run` or its ``UNLINK_CLEARED_FIELDS``
    declaration: that helper's other callers -- Phase 28's undo view and the admin's
    standalone clear branch -- are human-initiated and must keep being able to clear a
    confirmed row. Only an *automated* sweep needs to defer to a prior human decision.

    No dismissal-row model instance is written anywhere on this path, nor by this
    function's caller: the UAT decision explicitly rejected the dismissal-row remedy
    proposed in 33-REVIEW.md CR-04's fix block -- a dismissal is the trace of a human's own
    decision, and an automated sweep declining to act is not one.

    Performs reads only (one queryset built off ``owned_events()``, one companion-row
    filter, one ``.values_list()`` and one ``.count()``) -- no ``.save()``, ``.update()``,
    ``.create()`` or ``.delete()`` runs here, so ``reconcile_run()``'s dry-run branch can
    call this directly to preview the detach without any write occurring (WR-11).

    Args:
        run: the ``CampaignRun`` just reconciled.
        active_urls: the exact set of ``CalendarEvent.url`` values the branch just run
            considers current for this run (one container url, or one url per night).

    Returns:
        tuple[list[int], int]: ``(clearable_event_ids, declined)`` -- the primary keys of
        events an automated sweep may release (their companion row's ``confirmed_by`` is
        unset), and the count of companion rows left attributed because ``confirmed_by``
        is set.
    """
    stale = owned_events(run).exclude(url__in=active_urls)
    metas = CalendarEventMeta.objects.filter(run_id=run.pk, event__in=stale)
    clearable_event_ids = list(metas.filter(confirmed_by__isnull=True).values_list('event_id', flat=True))
    declined = metas.filter(confirmed_by__isnull=False).count()
    return clearable_event_ids, declined


def _detach_stale_family_events(run: CampaignRun, active_urls: set[str]) -> tuple[int, int]:
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
    through a non-``RUN:`` event drops out of ``active_urls`` (see the per-night dispatch
    branch's own attribution handling), so its reconciler-minted event is detached back
    into Phase 28's queue here too, rather than lingering as a second attributed entry for
    the same night.

    Detaching -- rather than deleting the ``CalendarEvent`` rows outright, or merely
    logging/flagging -- returns them to Phase 28's attribution queue where a staff member
    can re-confirm them. Re-confirming a released entry is now safe and permanent (Task 1,
    33-10): once ``confirmed_by`` is set on the re-linked row, no later automated sweep
    detaches it again -- see ``_stale_attributions()``. Plan 33-04 (D-16) routes the actual
    clear through the shared :func:`~solsys_code.campaign_utils.unlink_event_from_run`
    helper -- the single writer of what clearing an attribution means -- rather than this
    module's own ad-hoc update.

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
        tuple[int, int]: ``(detached, declined)`` -- the number of companion rows actually
        cleared (WR-03, 33-REVIEW.md), and the number left attributed because a human had
        confirmed them (see :func:`_stale_attributions`).
    """
    # Local import to avoid a circular import at module load time: campaign_utils.py
    # imports reconcile_run/ReconcileResult from this module at its own top level, so a
    # top-level import here of campaign_utils would deadlock on whichever module Python
    # loads first.
    from solsys_code.campaign_utils import unlink_event_from_run

    clearable_event_ids, declined = _stale_attributions(run, active_urls)
    detached = unlink_event_from_run(clearable_event_ids, run) if clearable_event_ids else 0
    if detached:
        logger.warning(
            'Reconcile detached %s stale/superseded event(s) from run pk=%s.',
            detached,
            run.pk,
        )
    if declined:
        logger.warning(
            'Reconcile declined to detach %s stale/superseded event(s) from run pk=%s: '
            'a human confirmation outranks the automated sweep.',
            declined,
            run.pk,
        )
    return detached, declined


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
    elif run.source in {
        CampaignRun.Source.LCO_QUEUE,
        CampaignRun.Source.SOAR_QUEUE,
        CampaignRun.Source.GEMINI_QUEUE,
        CampaignRun.Source.ESO_QUEUE,
    }:
        # D-09/D-10 (Phase 35): a queue-scheduled run keeps its single whole-window
        # container regardless of its resolved ground site -- this dispatch reads the
        # stored `source` field only and never infers provenance from a telescope name
        # or a site.
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}
    else:
        # D-09 (Phase 35): every other approved, windowed run with a resolved ground
        # site (WEB/CSV_IMPORT/CLASSICAL_FILE/LEGACY) is a per-night allocation, owned
        # entirely by the peer allocation_projector module. Local import: that module
        # imports this one at its own top level (to reuse split_telescope_instrument()
        # and _may_write()/_link_event_to_run()), so a top-level import here would
        # deadlock on whichever module Python loads first -- the same idiom
        # `_detach_stale_family_events()` already uses for campaign_utils.
        from solsys_code.allocation_projector import project_allocation

        result, active_urls = project_allocation(run, dry_run=dry_run)

    # CR-01 convergence step: detach (never delete) any of this run's owned events left
    # over from a family it no longer belongs to, OR a classical night's event superseded
    # by a later attribution (CR-03, 33-REVIEW.md). A no-op whenever the run has not been
    # re-classified and no night has been superseded since its last reconcile (RECON-01
    # idempotency: every url this branch just wrote/confirmed/skipped-for is already in
    # active_urls, so exclude() finds nothing stale). WR-11: the count is a pure read either
    # way -- `_stale_attributions()` only builds querysets and counts, so a dry run can
    # preview exactly what a real sweep would detach/decline without writing anything; the
    # one irreversible step in the sweep is precisely the step the preview used to refuse to
    # show.
    if dry_run:
        clearable_event_ids, declined = _stale_attributions(run, active_urls)
        detached, detach_declined = len(clearable_event_ids), declined
    else:
        detached, detach_declined = _detach_stale_family_events(run, active_urls)

    return result._replace(detached=detached, detach_declined=detach_declined)
