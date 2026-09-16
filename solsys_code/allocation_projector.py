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

import hashlib
import logging
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any
from zoneinfo import ZoneInfo

from django.db import transaction
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

# CR-01 (35-REVIEW.md iteration 7, plan 35-19 Task 2): the unrecorded-provenance branch of
# `_span_needs_remint()` compares a stored boundary against a freshly resolved sun_event()
# result to decide whether a LEGACY night (no recorded mint provenance) is still correct.
# Astropy drift between sessions (an IERS Earth-orientation refresh) moves a sun_event()
# result by seconds -- D-13's actual concern, and the loader's drifted-reimport regression
# test pins that drift alone must never re-mint a night. A stale OPERATOR value, by
# contrast, sits minutes to hours away from the true sun event. One minute separates the two
# classes cleanly: comfortably above any realistic astropy drift, comfortably below any
# real stale-boundary shape.
_UNRECORDED_PROVENANCE_TOLERANCE = timedelta(minutes=1)

# CR-02 (35-REVIEW.md iteration 8, plan 35-21): what the version marker is FOR. A night's
# boundaries are minted from every input `_mint_fields()`'s boundary computation reads, and
# that set can grow over time (CR-02 itself added the site to a token that used to carry
# only the sub-night pair). A token written before an input joined that set could not have
# carried it, so it must never be trusted to agree or disagree with a current-format token
# built from the wider identity -- reading it as "provenance unrecorded" is what lets a row
# that predates this version re-resolve once through the bounded legacy branch below,
# instead of a data migration rewriting every stored token by hand (D-15's no-data-migration
# rule). Bump this marker, and nothing else, the next time a new input joins the identity.
#
# T-35-24-01 (35-REVIEW.md iteration 9, plan 35-24; the round-5 verifier's escalated
# decision, 35-VERIFICATION.md "Human Verification Required" #1): bumped from `v2` to `v3`
# because `_sub_night_provenance_token()` now carries a site POSITION fingerprint alongside
# `site_id` -- an in-place `Observatory` correction (`lat`/`lon`/`altitude`/`timezone`
# edited, `run.site` untouched) changes the fingerprint but not `site_id`, and a `v2|` token
# could not have carried it. The bump is what makes every `v2|` token already stored read as
# unrecorded, so each such night resolves once through the bounded legacy branch
# (`_span_needs_remint()` step 4) and re-records in the current `v3` format -- the same
# read-time transition CR-02 used one release earlier, with no `RunPython` data migration
# (D-15) and no stored value ever rewritten in place.
_PROVENANCE_TOKEN_VERSION = 'v3'


def _site_position_fingerprint(run: CampaignRun) -> str:
    """A stable fingerprint of ``run.site``'s boundary-relevant POSITION (T-35-24-01,
    35-REVIEW.md iteration 9, plan 35-24; the round-5 verifier's escalated decision).

    Covers exactly the four ``Observatory`` fields ``sun_event()`` reads: ``lat`` and
    ``lon`` (through ``to_earth_location()``, which also needs ``altitude``) and
    ``timezone`` (through ``_local_noon_utc()`` and ``sun_event()``'s own blank-timezone
    guard). Those four, and nothing else, are what an in-place site correction can change
    that the boundaries actually depend on -- so those four, and nothing else, are what this
    fingerprint must cover.

    Reads ``run.site``, which ``project_allocation()`` has already loaded at the top of its
    sweep (building the site's ``ZoneInfo``), so the projector's own call path issues no
    additional query; a caller that has NOT already loaded the relation pays one lazy FK
    fetch. This corrects, rather than repeats, the "no database access" claim
    :func:`_sub_night_provenance_token` makes for itself: that claim is true of its own body
    (a plain FK-id read), but this sibling function may touch the database depending on the
    caller's own state, and the two must not be conflated.

    Each coordinate is rendered with ``repr()``, which in Python 3 gives the shortest
    decimal string that round-trips back to the identical float -- stable across process
    restarts and across a save/reload cycle, unlike ``str()`` for a float in general. A null
    coordinate renders as ``repr(None)`` (``'None'``) rather than a special case, so a
    satellite/space `Observatory` (no fixed lon/lat/altitude, see
    ``Observatory.to_earth_location()``) still produces a well-defined, comparable
    fingerprint rather than raising.

    The digest is TRUNCATED to 16 hexadecimal characters (64 bits) because the column this
    token feeds must stay bounded (see `CalendarEventMeta.minted_sub_night_window`'s
    `max_length`); a collision here costs at most a missed re-mint on a position change,
    which the next genuine position change still catches (the fingerprint is recomputed and
    compared on every sweep, never trusted twice for the same claim).

    Args:
        run: the ``CampaignRun`` being projected.

    Returns:
        str: the literal ``'none'`` when ``run.site_id`` is None; otherwise the first 16
        characters of the lowercase hexadecimal SHA-256 digest of
        ``f'{lat!r},{lon!r},{altitude!r},{timezone!r}'``.
    """
    if run.site_id is None:
        return 'none'
    site = run.site
    canonical = f'{site.lat!r},{site.lon!r},{site.altitude!r},{site.timezone!r}'
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


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


def _raise_if_set_window_inverted(run: CampaignRun, night) -> None:
    """Shared dry-run inversion guard (NF-20, WR-01, 35-REVIEW.md), serving BOTH
    ``_mint_fields()`` caller branches' dry-run short-circuits: the re-mint branch
    (``_span_needs_remint()`` returns True) and the create branch (``existing is None``).
    Before this helper existed, only the create branch checked for an inverted set sub-night
    pair under ``dry_run`` (NF-10) -- the re-mint branch's own ``if dry_run: continue``
    skipped the check entirely, so an operator edit that both changed an already-minted
    night's boundary AND inverted it previewed clean while the immediately following real run
    raised. A third caller of ``_mint_fields()`` added later has this one guard to call,
    rather than a third inline copy -- that recurrence is exactly what turned NF-10 into
    NF-20.

    Checks a span only when BOTH ``night_start_utc`` and ``night_end_utc`` are set. A
    HALF-null run (exactly one field set -- a half-night classical line such as
    ``1130-EoN``/``BoN-0230`` produces this shape via ``_window_token_to_time()``) is NOT
    previewed for inversion on EITHER caller branch, create or re-mint: the null boundary is
    a sun event the preview must not compute (D-13), and a boundary already stored on the
    existing event is NOT a sound substitute for it. A second gap-closure round (35-13) tried
    exactly that substitution, on the premise that the stored boundary was produced by the
    same deterministic ``sun_event()`` for the same site and night -- false whenever the
    now-null field was previously SET, because the re-mint branch is entered precisely when
    the sub-night fields CHANGED, and nulling a previously-set field leaves that field's old
    operator value sitting in the stored event, not a sunset or sunrise. Both directions of
    the resulting false parity were reproduced against a real Django test database and
    recorded in ``35-VERIFICATION.md``: PROBE-P1 (the preview RAISES on a night the real run
    creates cleanly) and PROBE-P6 (the preview is clean while the real run raises). The guard
    returns without raising whenever either resolved boundary is unknown -- restoring parity
    by silence, not by a guessed answer -- for a half-null run on EITHER branch; only the
    immediately following real run can detect that run's inversion.

    Args:
        run: the ``CampaignRun`` being projected.
        night: the site-local observing night (evening date).

    Raises:
        ValueError: the same error ``_raise_if_inverted()`` raises, when both resolved
            boundaries are known and the span is inverted.
    """
    if run.night_start_utc is None and run.night_end_utc is None:
        return
    night_span = _night_span_utc(run, night)
    start = (
        _time_of_day_to_datetime(run.night_start_utc, night, night_span) if run.night_start_utc is not None else None
    )
    end = _time_of_day_to_datetime(run.night_end_utc, night, night_span) if run.night_end_utc is not None else None
    if start is None or end is None:
        return
    _raise_if_inverted(run, night, start, end)


def _sub_night_provenance_token(run: CampaignRun) -> str:
    """The canonical text form of the full identity a run's ``ALLOC:`` night boundaries are
    CURRENTLY minted from (CR-01, 35-REVIEW.md iteration 7, plan 35-19; widened by CR-02,
    35-REVIEW.md iteration 8, plan 35-21; widened again by T-35-24-01, 35-REVIEW.md
    iteration 9, plan 35-24 -- the round-5 verifier's escalated decision).

    ``_mint_fields()`` computes a night's ``start_time``/``end_time`` from exactly four
    inputs: ``sun_event(run.site, night, kind='sun')`` and ``night_bounds(run, night,
    sunset, sunrise)`` read ``run.night_start_utc``, ``run.night_end_utc``, ``run.site`` and
    ``night`` -- so those four, and nothing else, are the boundary inputs this token must
    carry. All four are carried here, ``run.site`` now as two parts:

    - A leading version marker (:data:`_PROVENANCE_TOKEN_VERSION`), so a token written
      before an input joined this identity reads as unrecorded rather than as agreement --
      see the constant's own comment.
    - ``run.site_id`` -- the plain foreign-key column, read directly rather than through
      ``run.site`` so this function stays pure and database-free exactly as its predecessor
      was, without depending on a caller happening to have already resolved the relation.
      Rendered as the literal ``'none'`` when null, the same convention the two time fields
      already use, so this function does not assume a caller that enforces a non-null site
      (``reconcile_run()``'s stage-0 guard happens to, but this function must not assume it).
    - :func:`_site_position_fingerprint` -- the site's boundary-relevant POSITION
      (``lat``/``lon``/``altitude``/``timezone``), not merely its identity. ``site_id`` alone
      cannot detect an in-place ``Observatory`` correction: the round-5 verifier reproduced a
      ~15-hour error reported as `unchanged` forever, because the token recorded WHICH row
      supplied the position, never the position itself. ``site_id`` is KEPT alongside the
      fingerprint, not replaced by it -- it keeps a stored token legible in a log line and
      preserves the documented behaviour that a site SWAP (two different `Observatory` rows,
      even with byte-identical position) still re-mints.
    - The sub-night window pair, unchanged from the pre-CR-02 token: each side is
      ``isoformat()`` of the ``TimeField`` when set and the literal ``'none'`` when null.

    ``night`` is deliberately ABSENT. The event's own key already carries it
    (``ALLOC:{run.pk}:{night.isoformat()}``), and :func:`_span_needs_remint` is only ever
    called with the night that key encodes -- so ``night`` is a constant of every comparison
    this token feeds, not a variable the token could fail to carry.

    ``run.telescope_instrument`` and ``run.campaign`` are deliberately ABSENT too. They feed
    ``title``/``description``/``target_list``, which the plain-update path (the branch that
    runs when :func:`_span_needs_remint` returns False) rewrites on every sweep regardless --
    they are not boundary inputs, and including them would make an ordinary title or
    campaign change delete and re-create the night for no boundary reason at all.

    All five parts are joined by the existing single ``'|'`` separator. Pure and astropy-free
    itself; database access is only what ``run.site_id``/``run.site`` already cost the
    caller -- see :func:`_site_position_fingerprint`'s own docstring for the honest
    statement of when that is zero (this module's own call path) versus one lazy fetch (a
    caller that has not already resolved the relation).

    A sub-night side reading ``'none'`` is itself a RECORDED value, not an absence of one --
    it means "this boundary was minted from the sun event", unchanged from 35-19. That is
    what makes the null case decidable at all once a run's current token is compared against
    a previously recorded one (see :func:`_span_needs_remint`).

    Args:
        run: the ``CampaignRun`` being projected.

    Returns:
        str: e.g. ``'v3|3|a1b2c3d4e5f6a7b8|23:00:00|05:00:00'``,
            ``'v3|3|a1b2c3d4e5f6a7b8|none|05:00:00'``, or
            ``'v3|3|a1b2c3d4e5f6a7b8|none|none'``.
    """
    start_token = run.night_start_utc.isoformat() if run.night_start_utc is not None else 'none'
    end_token = run.night_end_utc.isoformat() if run.night_end_utc is not None else 'none'
    site_token = run.site_id if run.site_id is not None else 'none'
    fingerprint_token = _site_position_fingerprint(run)
    return f'{_PROVENANCE_TOKEN_VERSION}|{site_token}|{fingerprint_token}|{start_token}|{end_token}'


def _record_sub_night_provenance(event: CalendarEvent, token: str) -> None:
    """Writer (CR-01, 35-REVIEW.md iteration 7; plan 35-19): set ONLY
    ``CalendarEventMeta.minted_sub_night_window`` on ``event``'s companion row -- mirrors
    ``_link_event_to_run()``'s "Writer WR-03" docstring contract.

    Uses ``update_or_create`` keyed on ``event`` so this is safe whether or not
    ``_link_event_to_run()`` has already created the row for this event. Never touches
    ``run``, ``is_verified``, ``confirmed_by``, ``confirmed_at``, ``observation_record`` or
    ``observation_group`` -- an already-linked row's attribution and audit history must
    survive untouched.

    Args:
        event: the just-minted or just-re-minted allocation ``CalendarEvent``.
        token: :func:`_sub_night_provenance_token` for the run this event was minted from.
    """
    CalendarEventMeta.objects.update_or_create(event=event, defaults={'minted_sub_night_window': token})


def _span_needs_remint(run: CampaignRun, night, existing: CalendarEvent, *, dry_run: bool = False) -> bool:
    """D-13's re-mint check: whether ``existing``'s stored boundaries no longer match what
    the run's CURRENT sub-night fields say they should be (35-REVIEW.md NF-03, CR-01
    iteration 7).

    Proves CR-01 closed against all three probe shapes 35-VERIFICATION.md reproduced: (A) a
    set/set window cleared to null/null, (B) a half-null window's remaining set field
    cleared, and (C) clearing only ONE of two set fields while the other stays set and still
    matches its stored boundary -- the shape a naive "check whether BOTH fields are null"
    fix would still miss, since the per-field ``is not None`` gate reaches the second
    comparison, finds it unchanged, and used to return False for a window that genuinely
    changed. Comparing a null side against a stored boundary WITHOUT recorded provenance is
    what round 2 did and round 3 correctly deleted -- this function never does that; it
    proves the null case in two ways, neither of them a guess:

    1. Both existing SET-field comparisons run FIRST, byte-identical in meaning to before:
       a SET field whose expected boundary differs from the stored one returns True. These
       stay because they also catch a boundary edited directly on the ``CalendarEvent``,
       which provenance cannot see. ``_night_span_utc()`` is computed only when at least one
       field is set, so a null/null run does no ``zoneinfo`` work it will not use.
    2. If BOTH fields are set, return False -- fully decided, astropy-free, exactly as
       before. T-35-24-02 (35-REVIEW.md iteration 9, plan 35-24, WR-05) states what this
       actually means for a site correction, which the runbook, this docstring itself
       (before this correction) and plan 35-21's own success criterion all got wrong: a
       fully-set sub-night pair pins BOTH boundaries here, before the token is ever read --
       so the token, and therefore the site, is NEVER consulted for such a run. An in-place
       site correction on a fully-set run therefore changes NOTHING this function decides;
       the only site-derived field left for a correction to reach is the event's stored
       dark-window line, which ``project_allocation()``'s plain-update path now refreshes
       via :func:`_site_provenance_differs` (a separate, later read this function does not
       perform). A correction that moves the run's site to a DIFFERENT TIMEZONE is caught
       one step EARLIER, by step 1 above, because the resolved boundaries themselves move --
       not by this step. Operator remedy for that case: a sub-night window pinned to one
       site's observing night is not portable to another site's timezone, so the window
       fields (``night_start_utc``/``night_end_utc``) must be corrected together with the
       site, not left as-is.
    3. Otherwise read the recorded provenance from ``existing``'s companion row
       (``CalendarEventMeta.minted_sub_night_window``). CR-02 (35-REVIEW.md iteration 8,
       plan 35-21) changed this from a presence test to a VERSION test; T-35-24-01
       (35-REVIEW.md iteration 9, plan 35-24 -- the round-5 verifier's escalated decision,
       35-VERIFICATION.md "Human Verification Required" #1) adds a PART-COUNT test on top of
       it. The token is trusted only when it starts with the current
       :data:`_PROVENANCE_TOKEN_VERSION` marker followed by the separator AND splits into
       exactly the number of parts the current format has (five) -- a token written in an
       older format could not have carried every current boundary input (CR-02's own
       defect: a pre-CR-02 token carried the sub-night pair alone, so a ``run.site``
       correction produced no comparison that could detect it, forever), and a
       CURRENT-version token with the wrong part count is no more trustworthy than an
       old-version one. ``None``, the empty string, any pre-release token, and a
       current-version token with the wrong part count all fail this test and fall through
       to step 4 exactly as an unrecorded token always has.

       When the token is trusted, the comparison is COMPONENT-WISE rather than a single
       string equality -- this is what closes the escalated decision. The two sub-night
       sides differing, or ``site_id`` differing, returns True immediately: this is today's
       behaviour for both of those inputs and the documented meaning of a sub-night edit or
       a site SWAP (two different `Observatory` rows, even with byte-identical position --
       the accepted cost of keeping ``site_id`` in the token for log legibility). When only
       the POSITION FINGERPRINT differs, this function does NOT return True here: it falls
       through to step 4's resolution branch below, which makes one real ``sun_event()``
       call and compares the stored boundary against it at the established tolerance,
       either re-minting or re-recording a current-format token. This is the line that
       closes the escalated decision itself: an in-place ``Observatory`` correction
       (``lat``/``lon``/``altitude``/``timezone`` edited, ``run.site`` untouched, `site_id`
       therefore unchanged) now produces a fingerprint difference that routes to
       resolution -- proving whether the BOUNDARY actually moved -- rather than being
       invisible forever (the verifier's probe: a ``v2|`` token recording a stale
       ``site_id``-only identity, read as agreement regardless of how far the position
       drifted). Routing to resolution rather than to an outright re-mint is deliberate: an
       INPUT moving is not the same fact as the BOUNDARY moving, and a trivial one-metre
       altitude correction must not destroy and re-create every night at that site for no
       boundary reason. When every component matches, return False, astropy-free, exactly
       as before. No ``sun_event()`` call is ever made inside this step (prohibition 3 of
       the plan that introduced it still names this constraint).
    4. When provenance is NOT recorded (a version-or-part-count-test failure, including
       plain absence) OR a trusted token's position fingerprint alone differed (step 3
       above), this branch resolves the night against a real sun event rather than trusting
       or guessing. The two entry paths share this branch because they share the same
       proof obligation: neither a truly-unrecorded night nor a trusted-but-repositioned one
       has yet had its CURRENT boundary proven correct, so both are resolved identically,
       once, against ``sun_event()``. The unrecorded case covers a night minted before this
       column existed, taken over by the legacy re-key path while preserving the legacy
       event's own boundaries, or minted before an input (CR-02's site, or T-35-24-01's
       position fingerprint) joined the identity this token records. Round 2 SUBSTITUTED a
       stored boundary for an uncomputed sun event here and was reverted for it; this branch
       instead COMPUTES the sun event, exactly once, and records only what that computation
       confirms -- and what it records is always a CURRENT-format token, which is how
       existing rows migrate to the wider identity with no ``RunPython`` data migration
       (D-15's no-data-migration rule): the first reconcile after an upgrade resolves each
       such night once, the same bounded cost CR-01 already established, two releases wider.
       Calls ``sun_event(run.site,
       night, kind='sun')`` once for the night and compares only the NULL side or sides
       against the returned sunset/sunrise, resolved with the SAME expression
       :func:`night_bounds` uses so the two can never drift apart, against
       ``_UNRECORDED_PROVENANCE_TOLERANCE``: astropy drift between sessions moves a
       ``sun_event()`` result by seconds (D-13's actual concern), while a stale operator
       value -- or a genuinely relocated site -- sits minutes to hours away, so the
       one-minute tolerance separates the two cleanly. Outside the tolerance, logs a warning
       naming the run, the night, the stored boundary and the resolved sun event, and
       returns True -- this projector no longer reports a run/calendar disagreement it can
       detect as a silent ``unchanged`` (35-VERIFICATION.md `missing` item 4, and the
       escalated decision's own probe). Within the tolerance, the null side is PROVEN
       sun-derived by the call just made and the set side (if any) was already proven to
       match by step 1 -- so the run's current token is now an established fact, recorded
       via :func:`_record_sub_night_provenance` (skipped only under ``dry_run``, so a
       preview and the real run still reach the identical decision), and False is returned.
       Cost bound, qualified honestly (WR-01/WR-07, 35-REVIEW.md iteration 9, plan 35-24):
       "once ever" is true only when this branch's resolution is ALLOWED to record what it
       proves, and that has two escapes, named here rather than left for a reader to
       discover.

       - A night whose re-mint is DECLINED (``_remint_decline_reason()`` returns non-None,
         reported under ``remint_declined`` after plan 35-23's counter split) never
         receives a token from the re-mint path, and THIS branch deliberately records
         nothing on the stale path above (``return True`` before any write) -- so such a
         night resolves once PER SWEEP, indefinitely, paying exactly one
         ``sun_event(kind='sun')`` call and emitting both its warnings (this branch's
         staleness warning and ``_remint_decline_reason``'s own) every time, never more.
         This is accepted rather than fixed, for a stated reason: the repeated warning is
         the standing report that a night a person confirmed disagrees with its run, and
         the operator's remedy is in the runbook's ``remint_declined`` section. The
         alternative was considered and rejected: recording a token for boundaries that
         were not re-minted would claim a fact this sweep never proved -- the exact
         false-provenance mistake round 2 was reverted for (see this function's own
         module-level history).
       - Under ``--dry_run``, this resolution repeats on EVERY invocation, because a
         preview may not record what it proves (the ``if not dry_run:`` guard above).
         This is WR-01, a separate, still-open finding -- NOT claimed fixed here.

       Neither escape widens step 3's own bound above (component-wise, zero-astropy once
       every component matches): both apply only to nights that reach THIS branch at all,
       which step 3 already gates.

    Args:
        run: the ``CampaignRun`` being projected.
        night: the site-local observing night (evening date).
        existing: the already-identified allocation ``CalendarEvent``.
        dry_run: when True, step 4 still performs its comparison (so a preview and the real
            run reach the identical decision) but skips recording what it proves.

    Returns:
        bool: True when the night must be deleted and re-created fresh.
    """
    night_span = None
    if run.night_start_utc is not None or run.night_end_utc is not None:
        night_span = _night_span_utc(run, night)
    if run.night_start_utc is not None and existing.start_time != _time_of_day_to_datetime(
        run.night_start_utc, night, night_span
    ):
        return True
    if run.night_end_utc is not None and existing.end_time != _time_of_day_to_datetime(
        run.night_end_utc, night, night_span
    ):
        return True
    if run.night_start_utc is not None and run.night_end_utc is not None:
        return False
    try:
        recorded_token = existing.telescope_label_meta.minted_sub_night_window
    except CalendarEventMeta.DoesNotExist:
        recorded_token = None
    # CR-02 (35-REVIEW.md iteration 8, plan 35-21), extended by T-35-24-01 (35-REVIEW.md
    # iteration 9, plan 35-24): a version-AND-part-count test, not a presence test and not a
    # version-prefix-only test. `None`, `''`, any pre-release token, and a current-version
    # token with the wrong part count all fail this and fall through to the legacy branch
    # below, which resolves and re-records them in the current format.
    recorded_parts = recorded_token.split('|') if recorded_token is not None else None
    token_trusted = (
        recorded_parts is not None and len(recorded_parts) == 5 and recorded_parts[0] == _PROVENANCE_TOKEN_VERSION
    )
    if token_trusted:
        current_parts = _sub_night_provenance_token(run).split('|')
        # T-35-24-01: component-wise, not a single string equality -- this is the line that
        # closes the escalated decision. A sub-night side or `site_id` differing returns
        # True immediately (today's behaviour, the documented meaning of a sub-night edit or
        # a site swap). Only the position fingerprint differing does NOT return True here --
        # it falls through to step 4's resolution branch below, which proves whether the
        # BOUNDARY actually moved rather than treating "an input moved" as "the boundary
        # moved". Everything matching returns False, astropy-free.
        (_version, site_id_part, fingerprint_part, start_part, end_part) = recorded_parts
        (_c_version, c_site_id_part, c_fingerprint_part, c_start_part, c_end_part) = current_parts
        if start_part != c_start_part or end_part != c_end_part or site_id_part != c_site_id_part:
            return True
        if fingerprint_part == c_fingerprint_part:
            return False
        # else: only the position fingerprint differs -- fall through to step 4.

    # Legacy night, provenance unrecorded, OR a trusted token whose position fingerprint
    # alone differed (CR-01 Task 2; T-35-24-01): resolve the unknown side(s) against ONE
    # real sun_event() call, never inferring provenance from the stored value.
    sunset, sunrise = sun_event(run.site, night, kind='sun')
    expected_sunset = sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    expected_sunrise = sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0)
    stale = False
    if run.night_start_utc is None and abs(existing.start_time - expected_sunset) > _UNRECORDED_PROVENANCE_TOLERANCE:
        stale = True
    if run.night_end_utc is None and abs(existing.end_time - expected_sunrise) > _UNRECORDED_PROVENANCE_TOLERANCE:
        stale = True
    if stale:
        logger.warning(
            'Allocation unrecorded-provenance night pk=%s run pk=%s night=%s: stored '
            'boundary start=%s end=%s disagrees beyond tolerance with the resolved sun '
            'event sunset=%s sunrise=%s.',
            existing.pk,
            run.pk,
            night,
            existing.start_time,
            existing.end_time,
            expected_sunset,
            expected_sunrise,
        )
        return True
    if not dry_run:
        _record_sub_night_provenance(existing, _sub_night_provenance_token(run))
    return False


def _site_provenance_differs(run: CampaignRun, existing: CalendarEvent) -> bool:
    """A pure read (T-35-24-02, 35-REVIEW.md iteration 9, plan 35-24, WR-05): whether
    ``existing``'s recorded provenance token proves the run's site component has moved --
    either ``site_id`` or the position fingerprint -- since it was last recorded.

    This is the one read ``project_allocation()``'s plain-update path needs to decide
    whether a FULLY-SET sub-night pair's dark-window line is stale: such a run's boundaries
    are pinned by :func:`_span_needs_remint`'s step 2 before its token is ever read, so this
    function performs the one site-only comparison that function never reaches. No astropy,
    no writes -- exactly the same contract :func:`_remint_decline_reason` states for itself.

    Returns True only when the recorded token passes the current version-AND-part-count
    test (:func:`_span_needs_remint` step 3's own test, repeated here rather than shared
    because the two callers act on the result differently) AND either its ``site_id``
    component or its position-fingerprint component differs from the run's current values.

    A token that FAILS the version-or-part-count test returns False -- a deliberate
    limitation, stated here with its reason: an UNRECORDED token is no evidence that the
    stored dark-window line is stale, since the boundaries it describes may never have been
    proven against the current site at all. Refreshing on an unrecorded token would pay one
    ``sun_event(kind='dark')`` call per legacy night on EVERY sweep, forever -- the same
    permanent per-sweep cost WR-07 documents for a declined re-mint's ``kind='sun'`` call.
    The operator's escape for such a night: clear one sub-night field, which routes it
    through the re-mint path instead, and that path recomputes the whole description (both
    the boundaries and the dark-window line) from scratch.

    Args:
        run: the ``CampaignRun`` being projected.
        existing: the already-identified allocation ``CalendarEvent``.

    Returns:
        bool: True only when a TRUSTED recorded token's ``site_id`` or position fingerprint
        differs from the run's current values.
    """
    try:
        recorded_token = existing.telescope_label_meta.minted_sub_night_window
    except CalendarEventMeta.DoesNotExist:
        recorded_token = None
    recorded_parts = recorded_token.split('|') if recorded_token is not None else None
    if recorded_parts is None or len(recorded_parts) != 5 or recorded_parts[0] != _PROVENANCE_TOKEN_VERSION:
        return False
    current_parts = _sub_night_provenance_token(run).split('|')
    _version, site_id_part, fingerprint_part, _start_part, _end_part = recorded_parts
    _c_version, c_site_id_part, c_fingerprint_part, _c_start_part, _c_end_part = current_parts
    return site_id_part != c_site_id_part or fingerprint_part != c_fingerprint_part


def _remint_decline_reason(run: CampaignRun, existing: CalendarEvent) -> str | None:
    """Decides whether an automated re-mint may destroy and re-create `existing` (CR-01,
    35-REVIEW.md iteration 8; plan 35-20).

    Two rules, in order:

    1. Reuse ``campaign_reconciler._clearable_declined_and_unattributed()``, unmodified --
       the same UAT-2026-09-09 "Option B: human outranks machine" rule the four sibling
       delete/detach paths in this module and in ``campaign_reconciler.py`` already apply.
       When it reports no deletable primary key for ``existing``, this re-mint would destroy
       a companion row a human has confirmed -- return the ``'confirmed'`` reason. That
       helper's third possible outcome (a companion row attributed to a DIFFERENT run) cannot
       arise at this call site: the per-night loop's own ``_may_write(existing, run)`` gate
       (``:765``) has already routed a foreign-owned night to ``blocked`` and ``continue``d
       before this branch is ever reached. So the partition here is two-way (deletable vs.
       confirmed-declined), and the absence of a foreign arm is deliberate, not an omission
       (D-16 / NF-01's "no third outcome" rule).
    2. Otherwise, check the companion row for staff-set facts a freshly created row would not
       reproduce: ``observation_record`` set, ``observation_group`` set, or ``is_verified``
       False. Any of them returns the ``'staff_state'`` reason.
       ``CalendarEventMeta.DoesNotExist`` reads as "no staff state", the same convention
       ``_span_needs_remint()`` already uses. This branch is stricter than its four siblings
       for a reason worth stating: the sibling delete/detach paths delete a night that is
       genuinely going away, while THIS branch destroys a row it intends to immediately
       re-create, so it owes the row's contents a decision. ``is_verified`` is the
       production-reachable half of that companion-row state -- the one companion-row field
       neither admin surface lists in ``readonly_fields`` -- and the two link fields are
       covered for the same reason at no extra cost. WR-08 (35-REVIEW.md iteration 9, plan
       35-24): setting ``is_verified`` False PERMANENTLY vetoes an automated correction of
       this night's boundaries through this rule, while NOT vetoing the night being retired
       when a linked observation places a block on it -- that is plan 35-23's separate
       decision for the retirement branch (see the Cross-reference paragraph below).
       ``CalendarEventMeta``'s own class docstring is where this field's full meaning --
       both halves -- is now recorded for a reader who starts there instead of here.

    Performs reads only -- no ``.save()``, ``.update()``, ``.create()`` or ``.delete()`` runs
    here, so a dry-run preview may call this directly, the same contract
    ``_clearable_declined_and_unattributed()`` states for itself.

    Args:
        run: the ``CampaignRun`` being projected.
        existing: the already-identified ``ALLOC:`` ``CalendarEvent`` the re-mint branch is
            about to destroy and re-create.

    Returns:
        str | None: ``None`` when this automated re-mint may proceed; otherwise a short
        reason token (``'confirmed'`` or ``'staff_state'``) naming why it may not.

    Cross-reference (CR-05, 35-REVIEW.md, plan 35-23): the retirement branch's own delete of
    an ``existing`` allocation night answers this same "may an automated write destroy this
    row" question DIFFERENTLY, deliberately, and does NOT call this function. It declines
    ONLY on ``confirmed_by`` (rule 1 above, via ``_clearable_declined_and_unattributed()``
    directly) -- never on ``is_verified=False`` or an observation-record/observation-group
    link (rule 2 above). This branch destroys a row it intends to immediately re-create, so
    it owes the row's contents a decision; the retirement branch removes a night genuinely
    superseded by the linked observation's own calendar entry, and extending rule 2's veto
    there would leave a permanent duplicate night on the calendar beside the very observation
    that retired it.
    """
    deletable_ids, _declined = _clearable_declined_and_unattributed(run, CalendarEvent.objects.filter(pk=existing.pk))
    if existing.pk not in deletable_ids:
        return 'confirmed'
    try:
        meta = existing.telescope_label_meta
    except CalendarEventMeta.DoesNotExist:
        return None
    if meta.observation_record_id is not None or meta.observation_group_id is not None or meta.is_verified is False:
        return 'staff_state'
    return None


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
        # message reads "owned by someone else") was false twice over for this shape. The
        # legacy-retire decline below stays on 'detach_declined' (its cause is confirmed_by,
        # so NF-16's reasoning and its message are both still correct).
        'detach_declined': 0,
        # 35-23 (CR-04/WR-06): the re-mint decline moves here -- a separate cause from the
        # legacy-retire decline above, so each printed message can name its own causes
        # truthfully instead of one counter claiming two things.
        'remint_declined': 0,
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
            # CR-05 (35-REVIEW.md, plan 35-23): the allocation night's OWN delete gets the
            # same two-way split the legacy event above already receives -- before this fix
            # the only gate `existing` passed was `_may_write()`, which admits this run's own
            # night regardless of `confirmed_by`, so a confirmed night was destroyed
            # silently (its companion row's stamp and both observation links cascaded away
            # with it, `CalendarEventMeta.event` being a `OneToOneField(on_delete=CASCADE)`),
            # counted as ordinary `retired` work. Deliberately NOT `_remint_decline_reason()`:
            # only `confirmed_by` declines a retirement here, never `is_verified=False` or an
            # observation_record/observation_group link -- the re-mint branch destroys a row
            # it intends to immediately re-create, so it owes the row's contents a decision,
            # while this branch removes a night genuinely superseded by the linked
            # observation's own calendar entry, and extending the veto here would leave a
            # permanent duplicate night on the calendar beside the very observation that
            # retired it. Operator remedy: clear the confirmation on that night's companion
            # row and re-run the sweep.
            existing_deletable = existing is None
            if existing is not None:
                deletable_ids, confirmed_declined = _clearable_declined_and_unattributed(
                    run, CalendarEvent.objects.filter(pk=existing.pk)
                )
                if existing.pk in deletable_ids:
                    existing_deletable = True
                elif confirmed_declined:
                    logger.warning(
                        'Allocation retire declined: night pk=%s night=%s is human-confirmed '
                        'to run pk=%s -- an automated retirement never destroys it.',
                        existing.pk,
                        night,
                        run.pk,
                    )
                    totals['detach_declined'] += 1
            if not dry_run:
                if existing_deletable and existing is not None:
                    existing.delete()
                if legacy_deletable:
                    legacy_event.delete()
            # "retired" counts a night that went away, not a night we walked past (CR-05):
            # it fires when there was nothing to delete to begin with (existing is None,
            # today's preserved behaviour) and when the allocation night was actually
            # deletable -- never when its delete was declined.
            if existing_deletable:
                totals['retired'] += 1
            # No active_urls.add(url) here (IN-05, 35-REVIEW.md): retired_urls.add(url) at
            # the top of this branch already excludes this url from the D-14 convergence
            # step at the bottom of this function -- adding it to active_urls too would be a
            # second no-op of exactly the kind IN-05 removed.
            continue

        active_urls.add(url)

        legacy_event = CalendarEvent.objects.filter(url=legacy_url).first() if existing is None else None

        if existing is None and legacy_event is not None:
            # NF-22 (35-REVIEW.md): claim the url the moment this loop has decided the
            # legacy event's fate AT ALL, exactly as the retired branch above already does
            # (NF-09) -- including the blocked-because-a-different-run-owns-it outcome, not
            # only the re-key path. Without this, a blocked takeover legacy event stayed
            # visible to campaign_reconciler._stale_dated_events()'s downstream `foreign`
            # count too, so the SAME single decision was reported under `blocked` here AND
            # `foreign`/`blocked` there -- one event, two counts.
            legacy_urls_claimed.add(legacy_url)
            if not _may_write(legacy_event, run):
                logger.warning(
                    'Allocation blocked: legacy event pk=%s is not owned by run pk=%s.',
                    legacy_event.pk,
                    run.pk,
                )
                totals['blocked'] += 1
                continue
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

        if existing is not None and _span_needs_remint(run, night, existing, dry_run=dry_run):
            # CR-01 (35-REVIEW.md iteration 8, plan 35-20): the guard runs FIRST, before
            # either counter moves and before the dry_run short-circuit below -- it is a pure
            # read (same contract as _clearable_declined_and_unattributed()), so a dry-run
            # preview and a real run reach the identical decision and report the identical
            # counters for a declined re-mint, exactly as they already do for a re-minted one.
            decline_reason = _remint_decline_reason(run, existing)
            if decline_reason is not None:
                if decline_reason == 'confirmed':
                    logger.warning(
                        'Allocation re-mint declined: event pk=%s night=%s is human-confirmed '
                        'to run pk=%s -- an automated re-mint never clears it.',
                        existing.pk,
                        night,
                        run.pk,
                    )
                else:
                    logger.warning(
                        'Allocation re-mint declined: event pk=%s night=%s carries staff-set '
                        'state (an observation_record/observation_group link, or '
                        'is_verified=False) for run pk=%s -- an automated re-mint never '
                        'clears it.',
                        existing.pk,
                        night,
                        run.pk,
                    )
                totals['remint_declined'] += 1
                # active_urls.add(url) is unnecessary here (IN-05, 35-REVIEW.md): the
                # unconditional add a few lines above (this branch is only reachable after
                # it) already covers this url -- a second add here would be exactly the kind
                # of redundant no-op IN-05 removed.
            else:
                # D-13: a sub-night field change never rewrites start_time/end_time in place
                # -- the night is deleted and re-created fresh, counted as retired + created,
                # never updated. Both halves are skipped under dry_run (no sun_event() call
                # either), so a dry-run preview and a real run agree on the same pair of
                # counters.
                totals['retired'] += 1
                totals['created'] += 1
                if dry_run:
                    # NF-20/WR-01 (35-REVIEW.md, 35-VERIFICATION.md gap 1): both this branch
                    # and the create branch below call the SAME two-argument guard. It raises
                    # only for a set/set span; a half-null span here is left to the real run
                    # to detect, exactly as it already is on the create branch, since round 2's
                    # attempt to preview it via a stored-boundary fallback produced a false
                    # positive (PROBE-P1) and did not fix the original false negative (PROBE-P6).
                    _raise_if_set_window_inverted(run, night)
                    continue
                # CR-03 (35-REVIEW.md): compute BEFORE destroying. _mint_fields() reaches
                # night_bounds() -> _raise_if_inverted(), so an inverted span raises HERE, with
                # nothing deleted yet -- the reorder protects against a failure in the one
                # movable failure point.
                remint_fields = _mint_fields(run, night)
                with transaction.atomic():
                    # CR-03: contain the rest. The wrap protects against a failure in the
                    # three write steps below, which cannot be moved ahead of the delete -- a
                    # failure anywhere in this block rolls the delete back, so no night is
                    # ever left with zero calendar events. Scoped to exactly this one night's
                    # delete/create pair and nothing wider (prohibition 7): a sweep over many
                    # runs still commits the runs and nights it has already finished when a
                    # later one raises.
                    existing.delete()
                    event, _action = insert_or_create_calendar_event({'url': url}, fields=remint_fields)
                    _link_event_to_run(event, run)
                    # CR-01 (35-REVIEW.md iteration 7, plan 35-19): record what this re-mint's
                    # boundaries were minted from, so the next sweep's _span_needs_remint()
                    # can decide a future null-side change astropy-free.
                    _record_sub_night_provenance(event, _sub_night_provenance_token(run))
                continue

        # CR-04 (35-REVIEW.md, plan 35-23): a declined re-mint falls through to here instead
        # of `continue`-ing out of the loop. The decline refuses only the DESTRUCTIVE half
        # (the delete/create pair and its boundary rewrite, handled above) -- the night still
        # travels the ordinary update path below, which writes title/description/target_list
        # and is how a staff mark_cancelled/mark_weather_failure action reaches an allocation
        # night at all (allocation_night_description()'s own docstring). A declined night
        # therefore reports remint_declined AND an updated/unchanged, deliberately, because
        # two things happened to it in the same sweep.

        if existing is None:
            # WR-03 (35-REVIEW.md): `preview_calendar_event_action(None, fields)` always
            # returns 'created' without reading `fields` at all -- so under dry_run, calling
            # `_mint_fields()` (two `sun_event()` calls) here would compute and discard the
            # same astropy work for every brand-new night in the previewed window, and could
            # raise `sun_event()`'s own `ValueError` (e.g. a blank `Observatory.timezone`)
            # on what the module's own docstring documents as a read-only preview.
            #
            # NF-10/NF-20/WR-01 (35-REVIEW.md, 35-VERIFICATION.md gap 1): `_mint_fields()` is
            # also the only caller of `night_bounds()`, where CR-06's inversion guard lives --
            # skipping it entirely under `dry_run` hid the one failure mode an OPERATOR-set
            # (not site-derived) `night_start_utc`/`night_end_utc` pair can raise: a preview
            # reported `would_create` for a night whose immediately following real run failed
            # with an inverted-span `ValueError`. `_raise_if_set_window_inverted()` is the
            # shared guard also called from the re-mint branch above, so the two passes
            # cannot drift apart on this check for a set/set pair on either branch. A
            # half-null pair is left to the real run to detect on EITHER branch, this one
            # included: there is no stored counterpart this preview may trust (a round-2
            # attempt to fall back to one produced PROBE-P1's false positive without fixing
            # PROBE-P6's false negative), and resolving the null field here directly would
            # need the `sun_event()` call D-13 forbids on a preview -- documented here rather
            # than silently skipped.
            if dry_run:
                _raise_if_set_window_inverted(run, night)
                totals['created'] += 1
                continue
            fields: dict[str, Any] = _mint_fields(run, night)
            refresh_dark_window = False
        else:
            # T-35-24-02 (35-REVIEW.md iteration 9, plan 35-24, WR-05): a fully-set
            # sub-night pair pins both boundaries at _span_needs_remint()'s step 2, so a
            # site correction never reaches this run's night through the token at all --
            # the stored dark-window line is the only site-derived field left for a
            # correction to reach. The both-set condition below is not decoration: for a
            # fully-set run, step 1 of _span_needs_remint() has already compared both
            # stored boundaries against what the current site and the current sub-night
            # fields produce and found them equal on THIS SAME sweep, so refreshing the
            # description and recording the current token below are both claims this sweep
            # just proved. For any other sub-night case the token comparison has already
            # decided the night, so this refresh is unreachable and must stay so.
            refresh_dark_window = (
                run.night_start_utc is not None
                and run.night_end_utc is not None
                and _site_provenance_differs(run, existing)
            )
            if refresh_dark_window and not dry_run:
                # D-13 (verbatim): "`sun_event()` (both `'sun'` and `'dark'`) runs only for
                # a night being created or re-minted." This call is the ONE stated exception
                # to that clause. The night reaching this line is neither created nor
                # re-minted; the call fires ONLY on the transition where a recorded
                # current-format token proves the site component moved AND both sub-night
                # fields are set; it is bounded to exactly one call per night per site
                # correction, because this same sweep records the current token below (see
                # the comment on that write); and it never fires in a preview (the `dry_run`
                # half of this condition) or on an idempotent sweep (the flag is False
                # whenever the site component has not moved). What is NOT narrowed: D-13's
                # purpose -- no `sun_event()` call on an idempotent re-reconcile of an
                # existing night, the todo
                # `2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md`
                # asked for -- is untouched, and `TestNoSunEventRecompute` staying green
                # unedited is what proves it. WR-05 is the reason this exception exists at
                # all: for a fully-set pair the boundaries are correctly pinned, so the
                # dark-window line is the only site-derived field a correction can still
                # reach.
                dark_start, dark_end = sun_event(run.site, night, kind='dark')
                dark_start_iso = dark_start.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0).isoformat()
                dark_end_iso = dark_end.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0).isoformat()
                dark_line = f'{_DARK_WINDOW_PREFIX}{dark_start_iso} to {dark_end_iso}'
            else:
                dark_line = preserved_dark_window_line(existing)
            fields = {
                'title': allocation_night_title(run),
                'description': allocation_night_description(run, dark_line),
                'target_list': run.campaign,
            }

        if dry_run:
            if existing is not None and refresh_dark_window:
                # The preview already KNOWS the description will change, because the site
                # component of the recorded token differs from the run's current site, and
                # it reports that without paying the astropy call the real run pays above --
                # keeping WR-02's deferred preview-raises-ValueError surface exactly as wide
                # as it was (35-23-PLAN.md's ledger; WR-02 stays deferred). Accepted
                # divergence, pinned by its own test rather than left for a reader to
                # discover: when the corrected position happens to produce an identical
                # dark window, this preview still reports `updated` where the real run
                # reports `unchanged`
                # (test_preview_may_over_report_updated_by_one_on_a_site_correction).
                totals['updated'] += 1
                continue
            totals[preview_calendar_event_action(existing, fields)] += 1
            continue

        if existing is None:
            event, action = insert_or_create_calendar_event({'url': url}, fields=fields)
            _link_event_to_run(event, run)
            # CR-01 (35-REVIEW.md iteration 7, plan 35-19): record provenance only on the
            # create path -- the plain-update path's boundaries are whatever is already
            # stored, not this run's current sub-night window, so recording there would
            # claim a fact this write never proved. T-35-24-02 (plan 35-24) adds this
            # branch's single exception, just below: when the dark-window refresh fired,
            # step 1 of _span_needs_remint() already compared both boundaries against the
            # current inputs on this SAME sweep, so recording here claims only what this
            # sweep just proved.
            _record_sub_night_provenance(event, _sub_night_provenance_token(run))
        else:
            event, action = update_calendar_event_key_and_fields(existing, url, fields)
            _link_event_to_run(event, run)
            if refresh_dark_window:
                # T-35-24-02: the plain-update path's one exception to "provenance is
                # recorded on the create path only" -- see the comment on the create
                # branch's own call above for why this is sound.
                _record_sub_night_provenance(event, _sub_night_provenance_token(run))
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
