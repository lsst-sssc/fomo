"""Allocation projector -- one sunset->sunrise `ALLOC:` event per un-observed night (D-09).

This module owns the ``ALLOC:`` key namespace alone. ``RUN:{pk}`` (the bare whole-window
container, and the retired ``RUN:{pk}:{date}`` per-night family it is taking over from)
stays ``campaign_reconciler``'s; a facility observation url (e.g. an LCO/SOAR portal link)
stays the observation projector's. This module never creates, modifies or deletes an event
outside its own namespace.

Like ``campaign_reconciler.py``/``campaign_utils.py``/``campaign_gap.py``, this module must
NEVER import the views module or the heavy SPICE-loading ephemeris module -- the latter
triggers a ~1.6 GB SPICE kernel download at module load (CLAUDE.md "Heavy import side
effect", v2.2 milestone-locked module-home constraint).

Attribution is a link on ``CalendarEventMeta``, never a write to a ``CalendarEvent`` field:
this module's own ``ALLOC:``-keyed nights are self-attributed to their run the same way the
reconciler's ``RUN:``-keyed events are, and the observation-record attribution bridge (Task
2) reads/writes attribution exclusively through ``campaign_utils.adopt_event_into_run()`` /
``campaign_utils.unlink_event_from_run()`` -- never a direct assignment to the companion
row's ``run`` field.

Import discipline (the reason two different import styles appear below): a pure, stateless
helper this module calls with its own data is promoted to public in its home module and
imported by its public name (``campaign_reconciler.split_telescope_instrument()``,
``telescope_runs.observing_night()``, ``calendar_utils.coerce_schedule_datetime()``). The
reconciler's ownership and attribution POLICY (``_may_write()``, ``_link_event_to_run()``)
is imported under its private name deliberately, as a signal that it is one rule with one
owner -- promoting it would invite a second implementation here, and two projectors that
disagree about who may write an event is the defect this module exists downstream of.
"""

import logging
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any
from zoneinfo import ZoneInfo

from django.db.models import Q
from tom_calendar.models import CalendarEvent

from solsys_code.calendar_utils import (
    coerce_schedule_datetime,
    insert_or_create_calendar_event,
    preview_calendar_event_action,
    record_time_window,
    update_calendar_event_key_and_fields,
)
from solsys_code.campaign_reconciler import RUN_STATUS_CALENDAR_PREFIX as _RUN_STATUS_CALENDAR_PREFIX
from solsys_code.campaign_reconciler import (
    ReconcileResult,
    _clearable_declined_and_unattributed,
    _link_event_to_run,
    _may_write,
    event_description,
    run_container_url,
    split_telescope_instrument,
)
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.telescope_runs import observing_night, sun_event

logger = logging.getLogger(__name__)

ALLOC_URL_NAMESPACE = 'ALLOC:'

_DARK_WINDOW_PREFIX = 'Dark window (-15 deg, UTC): '


def allocation_night_url(run: CampaignRun, night) -> str:
    """The per-night allocation key.

    Args:
        run: the ``CampaignRun`` being projected.
        night: the site-local observing night (the same night ``sun_event()``'s sunset is
            computed for), never the naive UTC date.

    Returns:
        str: ``f'ALLOC:{run.pk}:{night.isoformat()}'``.
    """
    return f'{ALLOC_URL_NAMESPACE}{run.pk}:{night.isoformat()}'


def allocation_events(run: CampaignRun):
    """Every ``CalendarEvent`` keyed in this run's ``ALLOC:`` namespace.

    The trailing colon on the ``startswith`` prefix is required: without it, run pk=3 also
    matches run pk=34's allocation nights.
    """
    return CalendarEvent.objects.filter(url__startswith=f'{ALLOC_URL_NAMESPACE}{run.pk}:')


def writable_allocation_events(run: CampaignRun):
    """The ``ALLOC:`` twin of ``campaign_reconciler.writable_events()`` -- namespace
    identity alone is NOT ownership.

    Corrected relationship (35-REVIEW.md NF-06): this function's filter -- no companion row
    at all, a companion row whose ``run`` is unset, or a companion row that already points at
    this run -- was previously claimed to mirror ``writable_events()``'s queryset filter
    "exactly", which was true of the two querysets but false of the row-level predicate the
    pair is supposed to express: ``_may_write()`` used to admit only the ``RUN:`` namespace
    fallback, so it diverged from this function for exactly the shapes it claims to admit.
    ``_may_write()`` is now the single row-level predicate for BOTH namespaces;
    ``writable_events()`` is its ``RUN:``-namespace queryset twin and this function its
    ``ALLOC:``-namespace queryset twin -- all three now state the same rule. Used by the
    ``CampaignRun`` ``pre_delete`` cascade (``models.py``) alongside ``writable_events()`` so
    deleting run A never destroys an allocation night whose companion row attributes it to
    run B.
    """
    return allocation_events(run).filter(
        Q(telescope_label_meta__isnull=True)
        | Q(telescope_label_meta__run__isnull=True)
        | Q(telescope_label_meta__run=run)
    )


def allocation_night_title(run: CampaignRun) -> str:
    """Allocation night title: ``<telescope> <instrument>``, with the optional
    ``RUN_STATUS_CALENDAR_PREFIX`` -- exactly what ``load_telescope_runs`` writes today
    (``'NTT EFOSC2'``, ``'[CANCELLED] NTT EFOSC2'``). Deliberately NO ``(window a..b)``
    suffix -- that form belongs to the container branch's ``event_title()`` only (D-12).
    """
    telescope, instrument = split_telescope_instrument(run.telescope_instrument)
    base = f'{telescope} {instrument}'.strip()
    prefix = _RUN_STATUS_CALENDAR_PREFIX.get(run.run_status)
    if prefix:
        return f'{prefix} {base}'
    return base


def allocation_night_description(run: CampaignRun, dark_line: str | None) -> str:
    """Allocation night description: the -15 deg dark-window line (when given), followed by
    the shared ``event_description()`` body -- reused deliberately so a staff
    ``mark_cancelled``/``mark_weather_failure`` action reaches allocation nights the same way
    it reaches container events.

    Args:
        run: the ``CampaignRun`` being projected.
        dark_line: the literal first line ``f'Dark window (-15 deg, UTC): {start} to {end}'``,
            or ``None`` when no dark-window line applies (e.g. an update that has none to
            preserve).

    Returns:
        str: ``f'{dark_line}\\n{event_description(run)}'`` when ``dark_line`` is given, else
        ``event_description(run)``.
    """
    if dark_line is not None:
        return f'{dark_line}\n{event_description(run)}'
    return event_description(run)


def preserved_dark_window_line(event: CalendarEvent) -> str | None:
    """The event's stored description's first line, when it is a dark-window line.

    This is what makes D-13's "no ``sun_event()`` on an unchanged night" reachable while
    still refreshing the description on update: reuse the stored dark-window line verbatim
    instead of recomputing it.

    Args:
        event: the already-identified allocation ``CalendarEvent``.

    Returns:
        str | None: the first line of ``event.description`` when it starts with the literal
        prefix ``'Dark window (-15 deg, UTC): '``, else ``None``.
    """
    first_line = (event.description or '').split('\n', 1)[0]
    if first_line.startswith(_DARK_WINDOW_PREFIX):
        return first_line
    return None


def _night_span_utc(run: CampaignRun, night) -> tuple[datetime, datetime]:
    """The site's own observing-night UTC span for ``night`` (D-04, 35-REVIEW.md NF-03): the
    site's nominal local 18:00 through the local wall-clock instant twelve hours later, both
    converted to UTC.

    Supersedes ``_site_runs_behind_utc()``'s sign-of-offset boolean (35-REVIEW.md NF-03): the
    property a per-boundary date resolution needs is WHERE the site's observing night sits
    relative to UTC midnight, not the SIGN of its UTC offset -- and there are three bands,
    not two. A local night runs roughly local 18:00 -> local 06:00, i.e.
    ``(18 - offset) -> (30 - offset)`` in UTC: entirely inside its own UTC date only when
    ``offset > +6`` (Siding Spring, +10); straddling UTC midnight for
    ``-6 < offset <= +6`` (La Silla -4, SAAO Sutherland +2, Hanle +5:30); entirely inside the
    NEXT UTC date when ``offset <= -6`` (Maunakea/FTN, -10). The fixed 12:00 UTC threshold the
    old rule used was correct only by coincidence for the two bands this project's fixture
    sites happened to occupy.

    This is a ``zoneinfo`` lookup only -- it must NEVER call ``sun_event()``. D-13 forbids any
    astropy work on ``_span_needs_remint()``'s update path, and using the site's nominal local
    18:00 rather than its true sunset is exactly what buys that. The twelve-hour offset is
    added to the zone-carrying local datetime BEFORE converting to UTC, so a DST shift that
    falls inside the night is applied by the conversion rather than assumed away.

    Args:
        run: the ``CampaignRun`` being projected -- its ``site.timezone`` selects the zone.
        night: the site-local observing night (evening date).

    Returns:
        tuple[datetime, datetime]: ``(span_start, span_end)``, both UTC-aware -- the site's
        nominal local 18:00 on ``night`` and the wall-clock instant twelve hours later.
    """
    site_zone = ZoneInfo(run.site.timezone)
    local_evening = datetime(night.year, night.month, night.day, 18, tzinfo=site_zone)
    local_morning = local_evening + timedelta(hours=12)
    return local_evening.astimezone(dt_timezone.utc), local_morning.astimezone(dt_timezone.utc)


def _time_of_day_to_datetime(t, night, night_span: tuple[datetime, datetime]) -> datetime:
    """A stored sub-night `TimeField` value -> a UTC datetime for one observing night (D-04,
    35-REVIEW.md NF-03).

    Builds two candidate UTC datetimes for ``t`` -- one on ``night``, one on
    ``night + timedelta(days=1)`` -- and picks the one closer to the site's own
    observing-night UTC span (``night_span``, from ``_night_span_utc()``): a candidate that
    lies within the span, inclusive of both endpoints, has distance zero; otherwise its
    distance is the smaller of its distances to the two span endpoints. On an exact tie the
    candidate on ``night`` wins. There is no hour comparison anywhere in this body -- the
    superseded rule's hard-coded ``t.hour < 12`` threshold is gone entirely, which is why a
    half-hour offset (Asia/Kolkata, +5:30) needs no special case.

    Args:
        t: a ``datetime.time`` (a stored ``night_start_utc``/``night_end_utc`` value).
        night: the site-local observing night (evening date).
        night_span: ``_night_span_utc(run, night)`` -- the site's own observing-night UTC
            span this stored time-of-day is resolved against.

    Returns:
        datetime: the UTC-aware datetime for that time-of-day on the correct date.
    """
    span_start, span_end = night_span
    next_night = night + timedelta(days=1)
    candidates = [
        datetime(night.year, night.month, night.day, t.hour, t.minute, t.second, tzinfo=dt_timezone.utc),
        datetime(next_night.year, next_night.month, next_night.day, t.hour, t.minute, t.second, tzinfo=dt_timezone.utc),
    ]

    def _distance(candidate: datetime) -> timedelta:
        if span_start <= candidate <= span_end:
            return timedelta(0)
        return min(abs(candidate - span_start), abs(candidate - span_end))

    return min(candidates, key=_distance)


def night_bounds(run: CampaignRun, night, sunset, sunrise) -> tuple[datetime, datetime]:
    """Resolve one allocation night's UTC start/end from the run's sub-night window fields
    (D-04, 35-REVIEW.md NF-03), each end independently.

    This is the same rule ``load_telescope_runs._resolve_window_time()`` applied per
    schedule line, moved behind the run so the allocation projector applies it per night
    instead of the command re-deriving it. The two ends are resolved independently, so a
    line may name one boundary and leave the other computed from the sun event. Each set
    boundary is resolved against the site's own observing-night UTC span
    (``_night_span_utc()``), computed once and shared by both ends: entirely inside its own
    UTC date for a site with an offset above +6 (Siding Spring), straddling UTC midnight for
    an offset above -6 and at or below +6 (La Silla, SAAO Sutherland, Hanle), entirely inside
    the NEXT UTC date for an offset at or below -6 (Maunakea/FTN).

    Args:
        run: the ``CampaignRun`` being projected.
        night: the site-local observing night (evening date).
        sunset: astropy Time of sunset for this night -- used when ``run.night_start_utc``
            is null.
        sunrise: astropy Time of sunrise for this night -- used when ``run.night_end_utc``
            is null.

    Returns:
        tuple[datetime, datetime]: ``(start, end)``, both UTC-aware, seconds precision.

    Raises:
        ValueError: the resolved ``start`` is not strictly before ``end`` (CR-06,
            35-REVIEW.md) -- refuses to hand the caller an inverted span to write, rather
            than silently minting a ``CalendarEvent`` whose ``start_time`` is after its
            ``end_time``.
    """
    night_span = _night_span_utc(run, night)
    if run.night_start_utc is None:
        start = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    else:
        start = _time_of_day_to_datetime(run.night_start_utc, night, night_span)
    if run.night_end_utc is None:
        end = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    else:
        end = _time_of_day_to_datetime(run.night_end_utc, night, night_span)
    _raise_if_inverted(run, night, start, end)
    return start, end


def _raise_if_inverted(run: CampaignRun, night, start: datetime, end: datetime) -> None:
    """Shared guard (CR-06, 35-REVIEW.md; extracted for NF-10, 35-REVIEW.md): raises the
    SAME ``ValueError`` :func:`night_bounds` raises when ``start`` is not strictly before
    ``end``. Extracted into its own function so :func:`night_bounds`'s real-mode check and
    the dry-run boundary preview below (NF-10) cannot drift apart the way ``night_bounds``
    and its own dry-run short-circuit already had -- a dry run must see the same failure a
    real run would, over the SAME two datetimes, checked the SAME way.

    Args:
        run: the ``CampaignRun`` being projected.
        night: the site-local observing night (evening date).
        start: the resolved UTC start of the night's span.
        end: the resolved UTC end of the night's span.

    Raises:
        ValueError: ``start`` is not strictly before ``end``.
    """
    if start >= end:
        logger.error(
            'Allocation night_bounds inverted for run pk=%s night=%s: start=%s >= end=%s '
            '(night_start_utc=%s, night_end_utc=%s).',
            run.pk,
            night,
            start,
            end,
            run.night_start_utc,
            run.night_end_utc,
        )
        raise ValueError(
            f'Computed an inverted allocation-night span for run pk={run.pk} night={night}: '
            f'start={start.isoformat()} >= end={end.isoformat()}. Check night_start_utc/'
            'night_end_utc against the site timezone.'
        )


def _span_needs_remint(run: CampaignRun, night, existing: CalendarEvent) -> bool:
    """D-13's cheap, astropy-free re-mint check: whether ``existing``'s stored boundaries no
    longer match what the run's CURRENT sub-night fields say they should be (35-REVIEW.md
    NF-03).

    A null sub-night field means the expected boundary is the sun-event pair, which cannot
    be known without calling ``sun_event()`` -- so a null field is deliberately never
    checked; D-13 forbids rewriting an existing night's stored boundary for astropy drift,
    and a null-null run therefore always reports "no re-mint needed" on this check. A SET
    field's expected boundary is computable with no astropy call at all -- a ``zoneinfo``
    span lookup (``_night_span_utc()``) plus the same per-boundary resolution
    ``night_bounds()`` uses -- so it is compared directly against the stored boundary; a
    mismatch on either end marks the night for re-mint.

    Args:
        run: the ``CampaignRun`` being projected.
        night: the site-local observing night (evening date).
        existing: the already-identified allocation ``CalendarEvent``.

    Returns:
        bool: True when the night must be deleted and re-created fresh.
    """
    if run.night_start_utc is None and run.night_end_utc is None:
        return False
    night_span = _night_span_utc(run, night)
    if run.night_start_utc is not None and existing.start_time != _time_of_day_to_datetime(
        run.night_start_utc, night, night_span
    ):
        return True
    if run.night_end_utc is not None and existing.end_time != _time_of_day_to_datetime(
        run.night_end_utc, night, night_span
    ):
        return True
    return False


def _mint_fields(run: CampaignRun, night) -> dict[str, Any]:
    """The full field set for a brand-new allocation night -- the only place `sun_event()`
    (both `'sun'` and `'dark'`) is called for a per-night create (D-13).

    Args:
        run: the ``CampaignRun`` being projected.
        night: the site-local observing night (evening date) being minted.

    Returns:
        dict[str, Any]: the ``insert_or_create_calendar_event()``-ready field set.
    """
    sunset, sunrise = sun_event(run.site, night, kind='sun')
    dark_start, dark_end = sun_event(run.site, night, kind='dark')
    dark_start_iso = dark_start.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0).isoformat()
    dark_end_iso = dark_end.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0).isoformat()
    dark_line = f'{_DARK_WINDOW_PREFIX}{dark_start_iso} to {dark_end_iso}'
    telescope, instrument = split_telescope_instrument(run.telescope_instrument)
    start, end = night_bounds(run, night, sunset, sunrise)
    return {
        'title': allocation_night_title(run),
        'description': allocation_night_description(run, dark_line),
        'target_list': run.campaign,
        'telescope': telescope,
        'instrument': instrument,
        'start_time': start,
        'end_time': end,
    }


def retired_nights(run: CampaignRun, site_zone: ZoneInfo) -> set:
    """The set of site-local observing nights a linked, placed-or-observed record retires
    (D-05).

    Iterates ``run.observation_links`` once. A record whose ``scheduled_start``/
    ``scheduled_end`` are not BOTH set is intent that may still move -- a queue window is
    not a set of owned nights -- so it retires nothing and the loop continues. No status
    filtering at all: a terminal-negative record that still carries a block keeps its night
    retired (D-06), because its own marked event already occupies that night.

    Args:
        run: the ``CampaignRun`` being projected.
        site_zone: the run's site timezone, built once by the caller.

    Returns:
        set: the site-local observing ``date``s a linked record's placed/observed block
        retires.
    """
    nights: set = set()
    for link in run.observation_links.select_related('observation_record'):
        record = link.observation_record
        try:
            start = coerce_schedule_datetime(record.scheduled_start)
            end = coerce_schedule_datetime(record.scheduled_end)
        except ValueError as exc:
            # G-34-2 portal-string case: the projector never raises on a record it cannot
            # read -- the record simply retires nothing.
            logger.warning(
                'retired_nights: could not coerce schedule bounds for observation_record ' 'pk=%s: %s',
                record.pk,
                type(exc).__name__,
            )
            continue
        if start is None or end is None:
            continue
        window_start, _window_end = record_time_window(record)
        nights.add(observing_night(window_start, site_zone))
    return nights


def _sync_observation_attribution(run: CampaignRun, *, dry_run: bool) -> int:
    """D-08's attribution bridge, both directions -- the only place this module writes an
    observation-record-derived event's attribution, and it never writes any of that event's
    ``title``/``description``/``start_time``/``end_time`` fields.

    Link half: every surviving link whose record's facility the observation projector owns
    gets its own event adopted into this run via ``campaign_utils.adopt_event_into_run()``,
    which refuses (logged, counted under ``blocked``) when the event is already attributed
    to a DIFFERENT run -- a staff confirmation elsewhere outranks this automated write.

    Unlink half: resolved by convergence, not by diffing (D-08's other sentence -- nothing
    else in the system clears an attribution whose link went away, and iterating surviving
    links can by construction never see the link that vanished). Each filter clause is
    load-bearing: ``run=run`` (plus ``unlink_event_from_run()``'s own filter) is "only when
    attributed to THIS run"; ``confirmed_by__isnull=True`` is the human guard -- a
    staff-confirmed attribution is left alone; ``observation_record__isnull=False`` keeps
    this step inside the observation projector's own namespace, out of reach of this
    module's self-attributed ``ALLOC:`` nights.

    Args:
        run: the ``CampaignRun`` being projected.
        dry_run: when True, do nothing and report zero blocked -- neither half is
            reachable under ``dry_run``.

    Returns:
        int: the number of link-half adoptions refused because the event already belongs
        to a different run.
    """
    if dry_run:
        return 0

    from solsys_code import campaign_utils, observation_projector

    blocked = 0
    for link in run.observation_links.select_related('observation_record'):
        record = link.observation_record
        if record.facility not in observation_projector.PROJECTED_FACILITIES:
            continue
        facility = observation_projector.facility_for(record)
        event = CalendarEvent.objects.filter(url=observation_projector.event_url(record, facility)).first()
        if event is None:
            continue
        if not campaign_utils.adopt_event_into_run(event, run):
            logger.warning(
                'Allocation attribution blocked: observation event pk=%s (record pk=%s) is '
                'already attributed to a different run (run pk=%s could not adopt it).',
                event.pk,
                record.pk,
                run.pk,
            )
            blocked += 1

    linked_record_pks = set(run.observation_links.values_list('observation_record_id', flat=True))
    stale_event_pks = list(
        CalendarEventMeta.objects.filter(run=run, confirmed_by__isnull=True, observation_record__isnull=False)
        .exclude(observation_record_id__in=linked_record_pks)
        .values_list('event_id', flat=True)
    )
    if stale_event_pks:
        cleared = campaign_utils.unlink_event_from_run(stale_event_pks, run)
        if cleared:
            logger.info(
                'Allocation attribution: cleared %s stale observation-event attribution(s) for run pk=%s.',
                cleared,
                run.pk,
            )

    return blocked


def project_allocation(run: CampaignRun, *, dry_run: bool = False) -> tuple[ReconcileResult, set[str], set[str]]:
    """Project (or refresh) every night in ``[run.window_start, run.window_end]`` inclusive
    into its own ``ALLOC:{run.pk}:{night}`` sunset->sunrise ``CalendarEvent``, retiring a
    night the moment a linked record's placed or observed block occupies it (D-05/D-07) and
    taking over a legacy ``RUN:{pk}:{night}`` event in place rather than duplicating it
    (D-16).

    Per-night resolution order: ownership (``_may_write()``) is decided BEFORE any other
    outcome -- a blocked night is counted and its url added to the active set (so a foreign
    attribution is never detached out from under it), never written. A retired night is
    counted once regardless of whether an event existed to delete, its url is added to
    neither the active set nor kept as a live night, and no legacy-takeover or mint logic
    runs for it. A night with no existing ``ALLOC:`` event but a writable legacy
    ``RUN:{pk}:{night}`` event is re-keyed in place (title/description/target_list only,
    same primary key, no ``sun_event()`` call) rather than minted fresh. ``sun_event()``
    (both ``'sun'`` and ``'dark'``) is called only when a brand-new night is being minted --
    never on the update or re-key paths (D-13): an existing night's ``start_time``/
    ``end_time`` are never rewritten.

    Field authority: on **create**, writes ``title``, ``description``, ``target_list``,
    ``telescope``, ``instrument``, ``start_time``, ``end_time``. On **update** (including a
    re-key), writes only ``title``, ``description`` (with the preserved dark-window line)
    and ``target_list``.

    After the per-night loop, the attribution bridge (``_sync_observation_attribution()``)
    links every surviving observation record's own event to this run, and clears the
    attribution of an event whose link is gone (D-08). Finally, convergence (D-14) deletes
    any ``ALLOC:`` event left over from a night this reconcile no longer visits (e.g. a
    re-classification that shrank the window) -- excluding nights already handled as
    retired above, so a dry-run preview never double-counts the same retirement twice.

    ``sun_event()``'s ``ValueError`` (e.g. a blank ``Observatory.timezone``) is deliberately
    NOT caught here -- it keeps propagating out of ``reconcile_run()`` for the staff-action
    call sites, unchanged from 29 D-06.

    Args:
        run: the ``CampaignRun`` to project. Must have a resolved ``site``, an approved
            status and a non-null ``window_start``/``window_end`` (``reconcile_run()``'s
            stage-0 guard, ``_skip_reason()``, already enforces this before dispatch).
        dry_run: when True, report what would change without writing anything.

    Returns:
        tuple[ReconcileResult, set[str], set[str]]: the outcome; the exact set of
        ``CalendarEvent.url`` values this call considers current -- every non-retired night
        visited, including a blocked night and every night visited in ``dry_run``; and
        ``legacy_urls_claimed`` -- every ``RUN:{pk}:{date}`` legacy url this per-night loop
        has already decided the fate of AT ALL (a takeover re-key, a retirement delete, OR
        (NF-09, 35-REVIEW.md) a decision to leave the row alone -- blocked because a
        different run owns it, or declined because a human confirmed it), in EITHER real or
        ``dry_run`` mode. The caller (``campaign_reconciler.reconcile_run()``) excludes this
        set from its own date-bearing convergence step (Task 1, Phase 35, D-16). NF-17
        (35-REVIEW.md): the exclusion is a no-op in real mode ONLY for a re-keyed or
        deleted url -- the write already happened by the time that step runs, so the url
        has already left the ``RUN:`` namespace -- but it is LOAD-BEARING in real mode for
        a blocked or declined url (NF-09 widened this set to claim those too): a
        blocked/declined event is, by definition, NOT written, so it is still sitting in
        the ``RUN:`` namespace when that step runs, and dropping the exclusion would restore
        NF-09's double count for exactly that shape. In ``dry_run`` mode nothing was
        written at all, so without this exclusion the SAME legacy url would be
        double-counted regardless of shape -- once here as
        ``rekeyed``/``retired``/``blocked``, and again there as
        ``legacy_deleted``/``detach_declined`` for the very same single decision (NF-09's
        double-count regression: a blocked or declined legacy event used to be reported
        under ``blocked`` here AND ``detach_declined`` downstream for one event, one
        decision).
    """
    totals: dict[str, int] = {
        'created': 0,
        'updated': 0,
        'unchanged': 0,
        'blocked': 0,
        'retired': 0,
        'rekeyed': 0,
        # NF-16 (35-REVIEW.md): seeded so the confirmed_declined branch below can route to
        # it -- nobody else owns this legacy event (it is in THIS run's own namespace,
        # confirmed_by-stamped to THIS run), so 'blocked' (whose reconcile_campaign_runs
        # message reads "owned by someone else") was false twice over for this shape.
        'detach_declined': 0,
    }
    site_zone = ZoneInfo(run.site.timezone)
    n_nights = (run.window_end - run.window_start).days + 1
    retired = retired_nights(run, site_zone)
    active_urls: set[str] = set()
    retired_urls: set[str] = set()
    legacy_urls_claimed: set[str] = set()

    for i in range(n_nights):
        night = run.window_start + timedelta(days=i)
        url = allocation_night_url(run, night)
        legacy_url = f'{run_container_url(run)}:{night.isoformat()}'
        existing = CalendarEvent.objects.filter(url=url).first()

        if not _may_write(existing, run):
            logger.warning('Allocation blocked: event pk=%s is not owned by run pk=%s.', existing.pk, run.pk)
            totals['blocked'] += 1
            active_urls.add(url)
            continue

        if night in retired:
            retired_urls.add(url)
            # CR-03 (35-REVIEW.md): the legacy RUN:{pk}:{night} event this retirement would
            # also delete gets the SAME two guards the takeover branch below already applies
            # to the same class of row -- ownership first (_may_write()), then a
            # human-confirmed attribution (reused via _clearable_declined_and_unattributed(),
            # the same UAT-2026-09-09 Option B rule every other delete/detach path in this
            # module and campaign_reconciler.py honours). Neither guard was applied here
            # before, so an automated re-projection could delete a companion row a staff
            # member had just confirmed, or one re-attributed to a different run entirely.
            legacy_event = CalendarEvent.objects.filter(url=legacy_url).first()
            legacy_deletable = False
            if legacy_event is not None:
                # NF-09 (35-REVIEW.md): claim the url the moment this loop has decided the
                # legacy event's fate AT ALL -- deleted, blocked or declined -- not only on
                # the deletable path. This cannot over-delete: the downstream
                # _stale_dated_events() step would have refused these same rows anyway (a
                # foreign attribution is excluded from both halves of
                # _clearable_declined_and_unattributed(); a confirmed_by row is counted
                # declined there too), so excluding them here removes only the DOUBLE COUNT
                # a blocked/declined legacy event used to produce (reported under `blocked`
                # here AND `detach_declined` downstream for the same single decision).
                legacy_urls_claimed.add(legacy_url)
                if not _may_write(legacy_event, run):
                    logger.warning(
                        'Allocation retire blocked: legacy event pk=%s is not owned by run pk=%s.',
                        legacy_event.pk,
                        run.pk,
                    )
                    totals['blocked'] += 1
                else:
                    deletable_ids, confirmed_declined = _clearable_declined_and_unattributed(
                        run, CalendarEvent.objects.filter(pk=legacy_event.pk)
                    )
                    if deletable_ids:
                        legacy_deletable = True
                    elif confirmed_declined:
                        logger.warning(
                            'Allocation retire declined: legacy event pk=%s is human-confirmed '
                            'to run pk=%s -- an automated retirement never clears it.',
                            legacy_event.pk,
                            run.pk,
                        )
                        # NF-16 (35-REVIEW.md): 'detach_declined', not 'blocked' -- the
                        # event is in THIS run's own namespace and confirmed_by-stamped to
                        # THIS run, so nobody else owns it; 'blocked's reconcile_campaign_runs
                        # message ("owned by someone else") was false for this shape, while
                        # 'detach_declined's message ("a person confirmed them...") is true.
                        totals['detach_declined'] += 1
            if not dry_run:
                if existing is not None:
                    existing.delete()
                if legacy_deletable:
                    legacy_event.delete()
            totals['retired'] += 1
            continue

        active_urls.add(url)

        legacy_event = CalendarEvent.objects.filter(url=legacy_url).first() if existing is None else None

        if existing is None and legacy_event is not None:
            if not _may_write(legacy_event, run):
                logger.warning(
                    'Allocation blocked: legacy event pk=%s is not owned by run pk=%s.',
                    legacy_event.pk,
                    run.pk,
                )
                totals['blocked'] += 1
                continue
            legacy_urls_claimed.add(legacy_url)
            dark_line = preserved_dark_window_line(legacy_event)
            rekey_fields: dict[str, Any] = {
                'title': allocation_night_title(run),
                'description': allocation_night_description(run, dark_line),
                'target_list': run.campaign,
            }
            if dry_run:
                totals['rekeyed'] += 1
                continue
            event, _action = update_calendar_event_key_and_fields(legacy_event, url, rekey_fields)
            _link_event_to_run(event, run)
            totals['rekeyed'] += 1
            continue

        if existing is not None and _span_needs_remint(run, night, existing):
            # D-13: a sub-night field change never rewrites start_time/end_time in place --
            # the night is deleted and re-created fresh, counted as retired + created, never
            # updated. Both halves are skipped under dry_run (no sun_event() call either),
            # so a dry-run preview and a real run agree on the same pair of counters.
            totals['retired'] += 1
            totals['created'] += 1
            if dry_run:
                continue
            existing.delete()
            event, _action = insert_or_create_calendar_event({'url': url}, fields=_mint_fields(run, night))
            _link_event_to_run(event, run)
            continue

        if existing is None:
            # WR-03 (35-REVIEW.md): `preview_calendar_event_action(None, fields)` always
            # returns 'created' without reading `fields` at all -- so under dry_run, calling
            # `_mint_fields()` (two `sun_event()` calls) here would compute and discard the
            # same astropy work for every brand-new night in the previewed window, and could
            # raise `sun_event()`'s own `ValueError` (e.g. a blank `Observatory.timezone`)
            # on what the module's own docstring documents as a read-only preview.
            #
            # NF-10 (35-REVIEW.md): `_mint_fields()` is also the only caller of
            # `night_bounds()`, where CR-06's inversion guard lives -- skipping it entirely
            # under `dry_run` hid the one failure mode an OPERATOR-set (not site-derived)
            # `night_start_utc`/`night_end_utc` pair can raise: a preview reported
            # `would_create` for a night whose immediately following real run failed with
            # an inverted-span `ValueError`. When BOTH sub-night fields are set, this can be
            # checked with no `sun_event()` call at all -- the same `zoneinfo`-only span
            # `night_bounds()` itself uses for a set boundary -- via the shared
            # `_raise_if_inverted()` guard, so the two passes cannot drift apart on this
            # check either. A null field's boundary depends on the sun event and is never
            # checked here, matching `_span_needs_remint()`'s own null-field convention.
            if dry_run:
                if run.night_start_utc is not None and run.night_end_utc is not None:
                    night_span = _night_span_utc(run, night)
                    start = _time_of_day_to_datetime(run.night_start_utc, night, night_span)
                    end = _time_of_day_to_datetime(run.night_end_utc, night, night_span)
                    _raise_if_inverted(run, night, start, end)
                totals['created'] += 1
                continue
            fields: dict[str, Any] = _mint_fields(run, night)
        else:
            dark_line = preserved_dark_window_line(existing)
            fields = {
                'title': allocation_night_title(run),
                'description': allocation_night_description(run, dark_line),
                'target_list': run.campaign,
            }

        if dry_run:
            totals[preview_calendar_event_action(existing, fields)] += 1
            continue

        if existing is None:
            event, action = insert_or_create_calendar_event({'url': url}, fields=fields)
        else:
            event, action = update_calendar_event_key_and_fields(existing, url, fields)
        _link_event_to_run(event, run)
        totals[action] += 1

    totals['blocked'] += _sync_observation_attribution(run, dry_run=dry_run)

    # CR-04 (35-REVIEW.md): namespace identity (allocation_events()) alone is NOT
    # ownership -- an event attributed to a DIFFERENT run, or human-confirmed to THIS run,
    # must survive this convergence exactly like every other delete/detach path in this
    # module and campaign_reconciler.py already requires. writable_allocation_events()
    # narrows to what this run may actually write; _clearable_declined_and_unattributed()
    # then splits that into what an automated sweep may delete (shape (c)-this-run
    # unconfirmed, OR shape (a)/(b) unattributed -- NF-01's fix) vs. what a human
    # confirmation protects. foreign_stale_count stays the only remaining "left alone"
    # bucket after this swap, so len(stale_ids) + declined_stale == writable_stale.count()
    # now holds, where before the NF-01 fix it did not (shapes (a)/(b) were neither).
    stale_qs = allocation_events(run).exclude(url__in=active_urls | retired_urls)
    writable_stale = writable_allocation_events(run).exclude(url__in=active_urls | retired_urls)
    foreign_stale_count = stale_qs.count() - writable_stale.count()
    stale_ids, declined_stale = _clearable_declined_and_unattributed(run, writable_stale)
    if foreign_stale_count or declined_stale:
        logger.warning(
            'Allocation convergence left %s event(s) alone for run pk=%s: %s attributed to a '
            'different run, %s human-confirmed to this run.',
            foreign_stale_count + declined_stale,
            run.pk,
            foreign_stale_count,
            declined_stale,
        )
    totals['blocked'] += foreign_stale_count + declined_stale
    if stale_ids:
        if not dry_run:
            CalendarEvent.objects.filter(pk__in=stale_ids).delete()
        totals['retired'] += len(stale_ids)

    return ReconcileResult(**totals), active_urls, legacy_urls_claimed


def reproject_allocation_if_dispatched(run: CampaignRun) -> None:
    """Trigger-side entry point (D-11) -- the only way a signal receiver may reach
    ``project_allocation()`` (35-REVIEW.md CR-01).

    ``project_allocation()``'s own precondition is that ``reconcile_run()``'s stage-0 guard
    (``_skip_reason()``) and its four-way dispatch have already run -- a signal receiver
    firing below an unrelated write (a ``CampaignRunObservation`` save/delete, or an
    ``ObservationRecord`` save) never goes through ``reconcile_run()`` at all, so calling
    ``project_allocation()`` straight from a receiver bypassed both the approval gate and
    the container/per-night dispatch rule. This function is the single place both are
    re-applied outside a sweep, so every trigger and the sweep itself agree on exactly one
    dispatch decision.

    Args:
        run: the ``CampaignRun`` a receiver wants to re-project.
    """
    from solsys_code.campaign_reconciler import _skip_reason, dispatches_per_night

    if _skip_reason(run) is not None or not dispatches_per_night(run):
        return
    project_allocation(run)


def receiver_on_run_observation_save(sender: Any, instance: Any, created: bool, raw: bool, **kwargs: Any) -> None:
    """post_save receiver on ``CampaignRunObservation`` (D-11): re-projects the linked run so
    an allocation night retires the moment a staff member confirms an attribution -- no
    operator command, no sweep.

    Fires downstream of an action that is already access-controlled
    (``AttributionDecisionView`` sits behind ``StaffRequiredMixin``); this is a receiver
    below an access-controlled action, not a new entry point. Makes no network call of its
    own, and reaches ``sun_event()`` only through ``project_allocation()``'s own
    create-or-re-mint branch -- the common transition here (linking a placed record) is a
    delete, not a mint.

    Returns immediately for a fixture load (``raw=True``). Resolves the run defensively
    (``CampaignRun.objects.filter(pk=instance.run_id).first()``) and returns when it is
    None -- the row's own ``run`` foreign key is required and a save is never a CASCADE
    consequence of the run's own deletion the way a delete can be, so in practice this only
    guards a race with a concurrent run deletion. See
    ``receiver_on_run_observation_delete()``'s own docstring for why its post-delete
    equivalent needs a stronger, ``origin``-based guard instead of relying on this lookup
    alone.

    Args:
        sender: the model class Django's signal framework passes (``CampaignRunObservation``).
        instance: the ``CampaignRunObservation`` that was just saved.
        created: True if this save created a new row (unused -- the run is re-projected
            either way, since an edit to an existing link's schedule-relevant fields would
            matter too).
        raw: True if this save came from a fixture load (``loaddata``).
        **kwargs: the remaining signal kwargs (``using``, ``update_fields``), unused.
    """
    if raw:
        return
    run = CampaignRun.objects.filter(pk=instance.run_id).first()
    if run is None:
        return
    try:
        reproject_allocation_if_dispatched(run)
    except Exception as exc:  # noqa: BLE001 -- never abort the caller's save
        logger.warning(
            'receiver_on_run_observation_save failed for link pk=%s run pk=%s: %s',
            instance.pk,
            instance.run_id,
            type(exc).__name__,
        )
        return
    logger.debug(
        'post_save re-projected run pk=%s after CampaignRunObservation pk=%s save',
        instance.run_id,
        instance.pk,
    )


def receiver_on_run_observation_delete(sender: Any, instance: Any, **kwargs: Any) -> None:
    """post_delete receiver on ``CampaignRunObservation`` (D-11): re-projects the linked run
    so an allocation night returns the moment a staff member undoes an attribution -- no
    operator command, no sweep.

    No ``raw`` parameter -- ``post_delete`` never sends one.

    A ``CampaignRun`` delete cascade (``run.delete()``) must project nothing, per its own
    ``pre_delete`` receiver already having cleared every one of the run's writable ``ALLOC:``
    events (``models.py``'s ``CampaignRun`` ``pre_delete`` receiver) before this signal ever
    fires -- re-projecting here would just re-mint fresh nights moments before the run row
    itself disappears. A plain ``CampaignRun.objects.filter(pk=instance.run_id).exists()``
    check cannot detect this case: Django's ``Collector`` deletes a CASCADE child (this row)
    and sends its ``post_delete`` *before* the parent row's own DELETE statement runs, in the
    same transaction -- so the run still exists in the database at this exact moment
    (verified against this project's Django version; the plan's own draft assumed the
    opposite). What Django DOES give a cascade-fired signal that a standalone delete lacks is
    ``kwargs['origin']`` -- the model instance ``.delete()`` was originally called on, the
    same for every signal the resulting ``Collector`` run fires. When ``origin`` is a
    ``CampaignRun`` (not this ``CampaignRunObservation`` itself), this delete is a cascade
    side effect of the run's own deletion, and this receiver returns without projecting
    anything. Only when ``origin`` is NOT a ``CampaignRun`` (a standalone
    ``link.delete()``/``CampaignRunObservation.objects.filter(...).delete()`` call, where
    ``origin`` is the link/queryset itself) does the run-existence lookup below apply, as a
    second, defensive check.

    CR-05 (35-REVIEW.md): Django sets ``origin`` to the object ``.delete()`` was called on
    for an instance-level ``Model.delete()``, but to the QUERYSET for a ``QuerySet.delete()``
    -- the Django admin's "Delete selected" bulk action goes through
    ``ModelAdmin.delete_queryset()`` -> ``queryset.delete()``, so a plain
    ``isinstance(origin, CampaignRun)`` check is False for that path even though it is every
    bit as much a cascade side effect of a run deletion as the single-object path. Checking
    the ORIGIN'S MODEL CLASS (``getattr(origin, 'model', type(origin))``) covers both forms:
    a bare ``CampaignRun`` instance's own type, and a ``QuerySet[CampaignRun]``'s ``.model``
    attribute.

    This receiver deliberately does NOT clear the removed link's own event attribution
    itself, and must not start doing so: ``project_allocation()`` already converges
    attributions against the run's surviving links (35-01 Task 2 step 4b /
    ``_sync_observation_attribution()``), so the one call this receiver makes both restores
    the night and clears the stale ``CalendarEventMeta.run`` together, and the sweep gets
    the same result without a signal. A second clearing writer here would be a second place
    for the human-confirmation guard to be forgotten.

    Fires downstream of an action that is already access-controlled
    (``AttributionDecisionView`` sits behind ``StaffRequiredMixin``). Makes no network call
    of its own, and reaches ``sun_event()`` only through ``project_allocation()``'s own
    create-or-re-mint branch.

    Args:
        sender: the model class Django's signal framework passes (``CampaignRunObservation``).
        instance: the ``CampaignRunObservation`` that was just deleted (already removed from
            the database by the time this fires, but its in-memory ``pk``/``run_id`` are
            still populated).
        **kwargs: the remaining signal kwargs (``using``, ``origin``); ``origin`` is read,
            ``using`` is unused.
    """
    origin = kwargs.get('origin')
    origin_model = getattr(origin, 'model', type(origin))
    if origin_model is CampaignRun or isinstance(origin, CampaignRun):
        return
    run = CampaignRun.objects.filter(pk=instance.run_id).first()
    if run is None:
        return
    try:
        reproject_allocation_if_dispatched(run)
    except Exception as exc:  # noqa: BLE001 -- never abort the caller's delete
        logger.warning(
            'receiver_on_run_observation_delete failed for link pk=%s run pk=%s: %s',
            instance.pk,
            instance.run_id,
            type(exc).__name__,
        )
        return
    logger.debug(
        'post_delete re-projected run pk=%s after CampaignRunObservation pk=%s delete',
        instance.run_id,
        instance.pk,
    )
