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
    legacy_deleted: int = 0
    """Date-bearing ``RUN:{pk}:{date}`` events left over from the retired per-night key
    family, belonging to a run that now dispatches to the whole-window container, deleted
    as one-time cutover churn (D-16, Phase 35 Task 1). Deliberately a DELETE, not a
    DETACH: the whole per-night form of this module's own ``RUN:`` namespace retires in
    this phase, so every one of its events is either re-keyed by the allocation projector
    (a per-night-dispatched run's own takeover, counted under ``rekeyed``) or removed here
    -- there is no third outcome that leaves the calendar coherent. The
    detach-never-delete rule (CR-01) continues to govern the bare ``RUN:{pk}`` container
    key, still counted under ``detached``/``detach_declined``."""
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
    still write it if the event's ``url`` already lives in one of this run's OWN key
    namespaces -- the ``RUN:`` container/per-night form, or the ``ALLOC:`` per-night
    allocation form. Everything else returns False -- a hand-created entry, a conference, a
    proposal deadline or an un-attributed sync-command event is never created, modified or
    deleted.

    The fallback covers BOTH of this run's own key namespaces (35-REVIEW.md NF-06): before
    this widening, an unattributed ``ALLOC:{pk}:{night}`` event (shape (a)/(b)) could never
    match the ``RUN:``-only fallback, so this predicate diverged from
    ``writable_allocation_events()``'s queryset twin, which admits exactly those shapes. The
    divergence's cost: a night with no attribution at all was permanently ``blocked``, never
    refreshed again, and the operator was told it was "owned by someone else" when nobody
    owned it -- while the ``pre_delete`` cascade (``models.py``) would have happily deleted
    the very same event. ``_may_write()`` and both queryset twins (``writable_events()`` for
    ``RUN:``, ``writable_allocation_events()`` for ``ALLOC:``) now state the same rule.
    """
    if event is None:
        return True
    meta = CalendarEventMeta.objects.filter(event=event).first()
    if meta is not None and meta.run_id is not None:
        return meta.run_id == run.pk
    # Local import: allocation_projector imports this module at its own top level (to reuse
    # split_telescope_instrument()/_may_write()/_link_event_to_run()), so a top-level import
    # here would deadlock on whichever module Python loads first -- the same idiom
    # _stale_allocation_events() and models.py's pre_delete cascade already use.
    from solsys_code.allocation_projector import ALLOC_URL_NAMESPACE

    container_url = run_container_url(run)
    return (
        event.url == container_url
        or event.url.startswith(f'{container_url}:')
        or event.url.startswith(f'{ALLOC_URL_NAMESPACE}{run.pk}:')
    )


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


def dispatches_per_night(run: CampaignRun) -> bool:
    """True when this run's calendar form is the per-night ``ALLOC:`` family, as opposed to
    the whole-window ``RUN:`` container (class-wide, satellite, or queue-sourced) --
    extracted so ``reconcile_run()``'s dispatch decision has exactly one owner, reused by
    ``allocation_projector.reproject_allocation_if_dispatched()`` (35-REVIEW.md CR-01) and
    by this module's own convergence step (CR-02) so a signal-triggered re-project and a
    sweep can never disagree about which branch a run belongs to.

    Mirrors ``reconcile_run()``'s own dispatch chain exactly: a run with a
    ``telescope_class``, a satellite site, or a queue ``source`` all take the whole-window
    container instead. A ``None`` site is defensive only -- ``_skip_reason()`` already
    refuses a run with no ``telescope_class`` and no resolved ``site`` before dispatch ever
    runs, so this function is never called with that combination in production, but callers
    outside that guard (a convergence read, a future trigger) must not crash on it.

    Args:
        run: the ``CampaignRun`` being dispatched.

    Returns:
        bool: True when ``run`` belongs to the per-night ``ALLOC:`` family.
    """
    if run.telescope_class:
        return False
    if run.site is None:
        return False
    if run.site.observations_type == Observatory.SATELLITE_OBSTYPE:
        return False
    return run.source not in {
        CampaignRun.Source.LCO_QUEUE,
        CampaignRun.Source.SOAR_QUEUE,
        CampaignRun.Source.GEMINI_QUEUE,
        CampaignRun.Source.ESO_QUEUE,
    }


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


def _split_stale_owned_events(run: CampaignRun, active_urls: set[str]) -> tuple[Any, Any]:
    """Splits this run's stale owned events (``owned_events(run)`` minus ``active_urls``)
    into the bare ``RUN:{pk}`` container form and the date-bearing ``RUN:{pk}:{date}``
    per-night form (Task 1, Phase 35). ``owned_events()`` never returns a third shape, so
    the two querysets this returns are mutually exclusive and exhaustive over the stale set.

    Args:
        run: the ``CampaignRun`` just reconciled.
        active_urls: the exact set of ``CalendarEvent.url`` values the branch just run
            considers current for this run (one container url, or one url per night).

    Returns:
        tuple: ``(stale_bare, stale_dated)`` querysets.
    """
    container_url = run_container_url(run)
    stale = owned_events(run).exclude(url__in=active_urls)
    stale_bare = stale.filter(url=container_url)
    stale_dated = stale.exclude(url=container_url)
    return stale_bare, stale_dated


def _clearable_and_declined(run: CampaignRun, candidates) -> tuple[list[int], int]:
    """Shared confirmed_by split (33-UAT.md ``## Decisions``, 2026-09-09, Option B -- human
    outranks machine) over an explicit candidate ``CalendarEvent`` queryset.

    Scoped to companion rows whose ``run`` is EXACTLY this run (``run_id=run.pk``): an event
    within ``candidates`` whose ``CalendarEventMeta.run`` points at a DIFFERENT run, or that
    carries no companion row at all, is excluded from both halves -- neither released nor
    counted as declined, since it was never this run's attribution to clear (T-29-19).

    Performs reads only -- no ``.save()``, ``.update()``, ``.create()`` or ``.delete()`` runs
    here, so a dry-run preview can call this directly without any write occurring (WR-11).

    Args:
        run: the ``CampaignRun`` just reconciled.
        candidates: a ``CalendarEvent`` queryset of stale events to split.

    Returns:
        tuple[list[int], int]: ``(clearable_event_ids, declined)`` -- the primary keys of
        events an automated sweep may release/delete (their companion row's ``confirmed_by``
        is unset), and the count of companion rows left alone because ``confirmed_by`` is
        set.
    """
    metas = CalendarEventMeta.objects.filter(run_id=run.pk, event__in=candidates)
    clearable_event_ids = list(metas.filter(confirmed_by__isnull=True).values_list('event_id', flat=True))
    declined = metas.filter(confirmed_by__isnull=False).count()
    return clearable_event_ids, declined


def _clearable_declined_and_unattributed(run: CampaignRun, candidates) -> tuple[list[int], int]:
    """Total-partition twin of :func:`_clearable_and_declined` (35-REVIEW.md NF-01, NF-06):
    the shape-(c)-this-run split ALONE is not the whole story, because shapes (a) (no
    ``CalendarEventMeta`` companion row at all) and (b) (a companion row whose ``run`` is
    unset) are candidates too -- every one of this module's four delete/detach call sites
    used to route through :func:`_clearable_and_declined` alone, which starts from
    ``CalendarEventMeta.objects.filter(run_id=run.pk, ...)`` and therefore sees only
    shape-(c)-this-run. Shapes (a) and (b) fell between that filter and
    ``foreign_stale_count``'s namespace-identity check: not deleted, not declined, and not
    counted foreign either -- silently permanent, with no log line. This is D-16's forbidden
    "third outcome" (NF-01): every candidate must land in exactly one of deletable, declined,
    or left-alone-because-attributed-elsewhere, and a candidate in none of the three is the
    bug this helper closes.

    Deletable means: an unconfirmed companion row attributed to THIS run (shape (c)-this-run,
    via :func:`_clearable_and_declined`), OR no attribution at all (shape (a) or (b)) --
    shape (a)/(b) has nothing to preserve, so it is exactly as safe to delete as a
    shape-(c)-this-run row with no ``confirmed_by``.

    The concatenation of the two halves cannot double-count: ``CalendarEventMeta.event`` is a
    ``OneToOneField`` (``models.py:39-43``), so a row in the ``run_id=run.pk`` half (shape
    (c)-this-run) can never also match "no companion row or ``run IS NULL``" (shape (a)/(b))
    -- the two halves are disjoint by construction.

    The unattributed half is ALSO split on ``confirmed_by``, even though
    :func:`~solsys_code.campaign_utils.unlink_event_from_run`'s ``UNLINK_CLEARED_FIELDS``
    clears ``run``, ``confirmed_by`` and ``confirmed_at`` together (D-16) -- so an unattributed
    row carrying ``confirmed_by`` should not normally arise through that single writer. A
    direct admin edit can still produce it, though, and a human stamp outranks an automated
    sweep whatever the row's shape: routing that stray combination to ``declined`` keeps the
    partition TOTAL rather than re-opening a narrower version of the very hole this helper
    closes.

    Performs reads only -- no ``.save()``, ``.update()``, ``.create()`` or ``.delete()`` runs
    here, so a dry-run preview may call this directly, same contract as
    :func:`_clearable_and_declined`.

    Args:
        run: the ``CampaignRun`` just reconciled.
        candidates: a ``CalendarEvent`` queryset of stale events to split.

    Returns:
        tuple[list[int], int]: ``(deletable_event_ids, declined)`` -- the primary keys of
        events an automated sweep may release/delete (shape (c)-this-run with no
        ``confirmed_by``, plus shape (a)/(b) with no ``confirmed_by``), and the count of
        companion rows left alone because ``confirmed_by`` is set (across both halves).
    """
    clearable, declined = _clearable_and_declined(run, candidates)
    unattributed = candidates.filter(Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True))
    unattributed_deletable_ids = list(
        unattributed.exclude(telescope_label_meta__confirmed_by__isnull=False).values_list('pk', flat=True)
    )
    stray_confirmed = unattributed.filter(telescope_label_meta__confirmed_by__isnull=False).count()
    return clearable + unattributed_deletable_ids, declined + stray_confirmed


def _stale_attributions(run: CampaignRun, active_urls: set[str]) -> tuple[list[int], int]:
    """Read-only split of this run's stale/superseded BARE-CONTAINER owned event into what
    an automated sweep may detach and what it must leave alone (33-UAT.md ``## Decisions``,
    2026-09-09): *"Option B -- human outranks machine. The reconciler sweep detaches only
    rows with no confirmed_by; a human-confirmed attribution is never cleared by an
    automated sweep."*

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

    Scoped to the bare-container shape only since Task 1 (Phase 35): the date-bearing
    counterpart is :func:`_stale_dated_events`, which deletes rather than detaches.

    Args:
        run: the ``CampaignRun`` just reconciled.
        active_urls: the exact set of ``CalendarEvent.url`` values the branch just run
            considers current for this run (one container url, or one url per night).

    Returns:
        tuple[list[int], int]: ``(clearable_event_ids, declined)`` -- see
        :func:`_clearable_and_declined`.
    """
    stale_bare, _stale_dated = _split_stale_owned_events(run, active_urls)
    return _clearable_and_declined(run, stale_bare)


def _stale_dated_events(
    run: CampaignRun, active_urls: set[str], claimed_legacy_urls: frozenset[str] = frozenset()
) -> tuple[list[int], int, int]:
    """The date-bearing counterpart of :func:`_stale_attributions` (Task 1, Phase 35, D-16):
    leftover ``RUN:{pk}:{date}`` events from the retired per-night key family. Unlike the
    bare-container group, these are DELETED rather than detached -- see
    ``ReconcileResult.legacy_deleted``'s docstring for why. Same guards as
    :func:`_stale_attributions`: an event attributed to a different run is left alone, and a
    human-confirmed attribution is reported under ``declined``, never cleared.

    NF-01 item 2 (35-REVIEW.md): before the shared total-partition helper existed, this
    function unioned in ONLY the no-companion-row half of the unattributed candidates
    (``stale_dated.filter(telescope_label_meta__isnull=True)``) -- shape (a). Shape (b), a
    ``RUN:{pk}:{date}`` event with a companion row whose ``run IS NULL``, was in NEITHER
    ``_clearable_and_declined()``'s own scope (which starts from
    ``CalendarEventMeta.objects.filter(run_id=run.pk, ...)`` and therefore sees only
    shape-(c)-this-run) NOR that partial union, so it survived every sweep forever with no
    counter moved and no log line. D-16's stated contract is that every ``RUN:{pk}:{date}``
    event is *"either re-keyed (elsewhere, by the projector) or removed (here) -- no third
    outcome"* -- a meta-less (shape (a)) or unset-``run`` (shape (b)) legacy event is exactly
    that third outcome. :func:`_clearable_declined_and_unattributed` now covers both shapes in
    one pass, so no separate union is needed here.

    NF-15 (35-REVIEW.md): ``_clearable_declined_and_unattributed()`` is a total partition
    over shapes (a)/(b)/(c)-this-run, but ``stale_dated`` -- derived from ``owned_events()``,
    i.e. namespace identity ALONE -- also contains shape (d): a ``RUN:{pk}:{date}`` event in
    THIS run's own namespace whose companion row attributes it to a DIFFERENT run. Shape (d)
    matches neither half of the partition (excluded from the clearable half by
    ``run_id=run.pk`` scoping, and from the unattributed half by its companion row's ``run``
    being set), so it used to land in NEITHER deletable NOR declined -- and NEITHER this
    function nor its caller (:func:`_detach_stale_family_events`) counted it at all: a
    silent, permanently-orphaned third outcome, in a key family this phase retires entirely,
    that no code path will ever revisit again. This now computes and logs that shape as
    ``foreign`` -- the same read-only, no-side-effect-except-logging contract
    ``project_allocation()``'s own ``foreign_stale_count`` already uses for the mirror
    ``ALLOC:`` case.

    Read-only except for the one ``logger.warning()`` call below (matching
    ``project_allocation()``'s own convergence step) -- callers (the real detach/delete step
    and the dry-run preview) both build on this without any database write occurring here.

    Args:
        run: the ``CampaignRun`` just reconciled.
        active_urls: the exact set of ``CalendarEvent.url`` values the branch just run
            considers current for this run (one container url, or one url per night).
        claimed_legacy_urls: legacy ``RUN:{pk}:{date}`` urls the allocation projector's own
            per-night loop already decided the fate of THIS call (a takeover re-key or a
            retirement delete) -- excluded here so a ``dry_run`` preview never
            double-counts the SAME url under both ``rekeyed``/``retired`` and
            ``legacy_deleted``. Empty for a container-dispatched run, which never takes
            over a legacy night at all.

    Returns:
        tuple[list[int], int, int]: ``(deletable_event_ids, declined, foreign)`` --
        ``deletable_event_ids``/``declined`` as :func:`_clearable_declined_and_unattributed`
        returns them; ``foreign`` is the count of shape-(d) events (NF-15) -- left alone
        entirely (a human attribution outranks a sweep, T-29-19), but reported rather than
        silently dropped.
    """
    _stale_bare, stale_dated = _split_stale_owned_events(run, active_urls)
    if claimed_legacy_urls:
        stale_dated = stale_dated.exclude(url__in=claimed_legacy_urls)
    writable_dated = stale_dated.filter(
        Q(telescope_label_meta__isnull=True)
        | Q(telescope_label_meta__run__isnull=True)
        | Q(telescope_label_meta__run=run)
    )
    foreign = stale_dated.count() - writable_dated.count()
    if foreign:
        logger.warning(
            'Reconcile found %s leftover per-night event(s) for run pk=%s attributed to a '
            'different run: left alone, not deleted.',
            foreign,
            run.pk,
        )
    deletable_event_ids, declined = _clearable_declined_and_unattributed(run, writable_dated)
    return deletable_event_ids, declined, foreign


def _stale_allocation_events(run: CampaignRun) -> tuple[list[int], int]:
    """Read-only split of this run's leftover ``ALLOC:`` nights when the run no longer
    dispatches to the per-night branch AT ALL (35-REVIEW.md CR-02): a re-classification
    into the whole-window container leaves the old per-night ``ALLOC:`` family behind with
    no other code path left to reach it -- ``project_allocation()``'s own convergence step
    (D-14) only runs for a per-night-dispatched run, so a run that moves OUT of that branch
    never revisits its own old nights again without this.

    Deliberately unscoped by ``active_urls``: a container-dispatched run's active set is
    always exactly ``{run_container_url(run)}``, a ``RUN:`` url that can never collide with
    an ``ALLOC:`` one, so every ``ALLOC:`` event this run still owns is stale by
    construction the moment this function is even reached.

    Same two guards as :func:`_stale_attributions`/:func:`_stale_dated_events`, PLUS the
    unattributed shapes NF-01 item 1 (35-REVIEW.md) names: an event attributed to a
    DIFFERENT run is left alone entirely (neither deleted nor counted), a human-confirmed
    attribution is reported under ``declined``, never cleared (UAT decision, 2026-09-09,
    Option B), and an ``ALLOC:`` night with NO companion row at all, or one whose ``run`` is
    unset, is now DELETED here too (shape (a)/(b)) -- before
    :func:`_clearable_declined_and_unattributed` existed, ``_clearable_and_declined()`` alone
    left those two shapes unreachable forever, the exact gap NF-01 item 1 reports for a run
    re-classified INTO container dispatch.

    Args:
        run: the ``CampaignRun`` just reconciled.

    Returns:
        tuple[list[int], int]: ``(deletable_event_ids, declined)`` -- empty/zero for a
        per-night-dispatched run, whose own ``ALLOC:`` convergence stays entirely inside
        :func:`~solsys_code.allocation_projector.project_allocation`.
    """
    if dispatches_per_night(run):
        return [], 0
    # Local import: allocation_projector imports this module at its own top level (to reuse
    # split_telescope_instrument()/_may_write()/_link_event_to_run()), so a top-level import
    # here would deadlock on whichever module Python loads first -- the same idiom this
    # module already uses for campaign_utils.
    from solsys_code.allocation_projector import writable_allocation_events

    return _clearable_declined_and_unattributed(run, writable_allocation_events(run))


def _detach_stale_family_events(
    run: CampaignRun, active_urls: set[str], claimed_legacy_urls: frozenset[str] = frozenset()
) -> tuple[int, int, int]:
    """Convergence step (29-REVIEW.md CR-01, user-directed fix: DETACH, not delete or
    flag-only, for the bare-container group; Task 1/D-16, Phase 35, adds a DELETE branch for
    the date-bearing group).

    ``reconcile_run()`` dispatches a run to exactly one of two branches (whole-window
    container vs. the peer allocation projector's per-night loop) based on the run's
    *current* state (never its ``source`` field alone for the container-by-``telescope_class``
    case, corrected by quick task ``260805-tad`` -- see 29-CONTEXT.md D-07's dated
    forward-pointer; ``source`` alone DOES decide the D-10 queue case). Nothing else detects
    a re-classification (an admin correction to a run's dispatch-deciding fields after it was
    already reconciled): without this, the OLD family's events would either be silently
    orphaned forever -- no code path ever revisits them again -- or, worse, silently
    miscounted as belonging to a family they no longer match.

    A second case (CR-03, 33-REVIEW.md): a classical night that has become attributed
    through a non-``RUN:`` event drops out of ``active_urls`` (see the per-night dispatch
    branch's own attribution handling), so its reconciler-minted event is detached back
    into Phase 28's queue here too, rather than lingering as a second attributed entry for
    the same night.

    **Bare-container group (unchanged):** detaching -- rather than deleting the
    ``CalendarEvent`` rows outright, or merely logging/flagging -- returns them to Phase 28's
    attribution queue where a staff member can re-confirm them. Re-confirming a released
    entry is now safe and permanent (Task 1, 33-10): once ``confirmed_by`` is set on the
    re-linked row, no later automated sweep detaches it again -- see
    :func:`_stale_attributions`. Plan 33-04 (D-16) routes the actual clear through the shared
    :func:`~solsys_code.campaign_utils.unlink_event_from_run` helper -- the single writer of
    what clearing an attribution means -- rather than this module's own ad-hoc update.

    **Date-bearing group (new, Task 1, Phase 35):** this is the retired ``RUN:{pk}:{date}``
    per-night key family's last remaining half -- the half the allocation projector cannot
    reach, because a container-dispatched run never enters the projector at all (a
    per-night-dispatched run's OWN leftover ``RUN:{pk}:{date}`` event is instead re-keyed in
    place by the projector's legacy-night takeover, counted under ``rekeyed``, and never
    reaches this convergence step at all). What DOES reach here -- surviving as a
    ``RUN:{pk}:{date}``-shaped event in this run's namespace after dispatch -- is, by
    construction, no longer a night the projector's takeover claimed: it belongs to a run
    that now dispatches to the whole-window container (D-10's 8 single-night queue runs
    being the primary case), so it is one-time churn deleted rather than detached, exactly
    like the bare-container group's events would be if their whole family were retiring.
    This is a real ``.delete()``, not a detach: the whole per-night form of this module's
    own namespace retires in this phase, and every one of its events is either re-keyed
    (elsewhere, by the projector) or removed (here) -- no third outcome.

    **``ALLOC:`` family (new, CR-02, 35-REVIEW.md):** the mirror image of the date-bearing
    group above, for a run that just went the OTHER way -- one that used to dispatch to the
    per-night allocation branch and has now been re-classified into the whole-window
    container. Its old ``ALLOC:{pk}:{night}`` nights are exactly as unreachable afterwards
    as a container-to-per-night re-classification's old ``RUN:{pk}:{date}`` nights are:
    ``project_allocation()`` is never called again for this run, so nothing else ever
    revisits them. Also a real ``.delete()``, counted into the same ``legacy_deleted``
    return value as the date-bearing group -- both are one-time churn from the SAME kind of
    event (a stale per-night key family, in the same run's namespace, left behind by a
    dispatch change), so the operator-facing runbook's existing ``legacy_deleted`` promise
    ("deleting the run's leftover per-night events ... and replacing them with a single
    whole-window entry") stays literally true for both origin families rather than needing
    a second, parallel counter.

    **Two guards, both groups (unchanged in shape):** the attribution must not point at a
    DIFFERENT run (:func:`_clearable_and_declined`'s ``run_id=run.pk`` scoping -- the
    T-29-19 concern), and a companion row carrying ``confirmed_by`` is left completely alone
    and counted under ``declined``/``detach_declined``, never cleared or deleted by an
    automated sweep -- a human confirmation always outranks it.

    Args:
        run: the ``CampaignRun`` just reconciled.
        active_urls: the exact set of ``CalendarEvent.url`` values the branch just run
            considers current for this run (one container url, or one url per night).
        claimed_legacy_urls: forwarded to :func:`_stale_dated_events` -- empty for a
            container-dispatched run (the only branch this delete path is normally reached
            for); real-mode is unaffected either way since a claimed legacy url has already
            left the ``RUN:`` namespace in the database by the time this function runs.

    Returns:
        tuple[int, int, int, int]: ``(detached, declined, legacy_deleted, foreign_blocked)``
        -- the number of bare-container companion rows actually cleared (WR-03,
        33-REVIEW.md), the number of rows across BOTH groups left attributed because a
        human had confirmed them (see
        :func:`_stale_attributions`/:func:`_stale_dated_events`), the number of date-bearing
        ``CalendarEvent`` rows actually deleted, and ``foreign_blocked`` -- the NF-15
        (35-REVIEW.md) shape-(d) count :func:`_stale_dated_events` reports: date-bearing
        events in this run's own namespace attributed to a DIFFERENT run, left alone and
        folded into the caller's own ``blocked`` total rather than silently dropped.
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

    legacy_deleted_ids, legacy_declined, foreign_blocked = _stale_dated_events(run, active_urls, claimed_legacy_urls)
    legacy_deleted = 0
    if legacy_deleted_ids:
        CalendarEvent.objects.filter(pk__in=legacy_deleted_ids).delete()
        legacy_deleted = len(legacy_deleted_ids)
        logger.warning(
            'Reconcile deleted %s leftover per-night event(s) from run pk=%s: this run now '
            'dispatches to the whole-window container, and its retired RUN:-namespaced '
            'per-night family is one-time churn.',
            legacy_deleted,
            run.pk,
        )

    alloc_deleted_ids, alloc_declined = _stale_allocation_events(run)
    if alloc_deleted_ids:
        CalendarEvent.objects.filter(pk__in=alloc_deleted_ids).delete()
        legacy_deleted += len(alloc_deleted_ids)
        logger.warning(
            'Reconcile deleted %s leftover allocation night(s) from run pk=%s: this run now '
            'dispatches to the whole-window container, and its per-night ALLOC: family is '
            'one-time churn.',
            len(alloc_deleted_ids),
            run.pk,
        )

    declined += legacy_declined + alloc_declined
    if declined:
        logger.warning(
            'Reconcile declined to detach/delete %s stale/superseded event(s) from run pk=%s: '
            'a human confirmation outranks the automated sweep.',
            declined,
            run.pk,
        )
    return detached, declined, legacy_deleted, foreign_blocked


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

    # `claimed_legacy_urls`: legacy RUN:{pk}:{date} urls the allocation projector's own
    # per-night loop already decided the fate of THIS call (Task 1, Phase 35) -- always
    # empty for a container-dispatched run, which never takes over a legacy night.
    claimed_legacy_urls: frozenset[str] = frozenset()

    if dispatches_per_night(run):
        # D-09 (Phase 35): every approved, windowed run with a resolved, non-satellite
        # ground site and a non-queue source (WEB/CSV_IMPORT/CLASSICAL_FILE/LEGACY) is a
        # per-night allocation, owned entirely by the peer allocation_projector module.
        # Local import: that module imports this one at its own top level (to reuse
        # split_telescope_instrument() and _may_write()/_link_event_to_run()), so a
        # top-level import here would deadlock on whichever module Python loads first --
        # the same idiom `_detach_stale_family_events()` already uses for campaign_utils.
        from solsys_code.allocation_projector import project_allocation

        result, active_urls, claimed_legacy_urls = project_allocation(run, dry_run=dry_run)
    else:
        # RECON-03/RECON-02 queue half/D-09-D-10: a class-wide allocation
        # (2m0/1m0/0m4/SPACE), a satellite-sited run, or a queue-scheduled run all share
        # this single whole-window branch -- `dispatches_per_night()` is the one place
        # that decides which of the two forms a run belongs to (CR-01, 35-REVIEW.md).
        result = _reconcile_container(run, dry_run=dry_run)
        active_urls = {run_container_url(run)}

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
        legacy_deleted_ids, legacy_declined, foreign_blocked = _stale_dated_events(
            run, active_urls, claimed_legacy_urls
        )
        alloc_deleted_ids, alloc_declined = _stale_allocation_events(run)
        detached = len(clearable_event_ids)
        detach_declined = declined + legacy_declined + alloc_declined
        legacy_deleted = len(legacy_deleted_ids) + len(alloc_deleted_ids)
    else:
        detached, detach_declined, legacy_deleted, foreign_blocked = _detach_stale_family_events(
            run, active_urls, claimed_legacy_urls
        )

    # NF-15 (35-REVIEW.md): fold the date-bearing shape-(d) foreign count into `blocked` --
    # the SAME total `project_allocation()`'s own convergence step already folds its
    # mirror-image ALLOC: shape into (allocation_projector.py's `foreign_stale_count` +
    # `declined_stale`) -- rather than leaving it uncounted anywhere on this branch.
    return result._replace(
        blocked=result.blocked + foreign_blocked,
        detached=detached,
        detach_declined=detach_declined,
        legacy_deleted=legacy_deleted,
    )
