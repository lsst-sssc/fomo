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
``campaign_utils.unlink_event_from_run()`` -- never ``meta.run = ...`` directly.

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
from datetime import timedelta
from datetime import timezone as dt_timezone
from typing import Any

from tom_calendar.models import CalendarEvent

from solsys_code.calendar_utils import (
    insert_or_create_calendar_event,
    preview_calendar_event_action,
    update_calendar_event_key_and_fields,
)
from solsys_code.campaign_reconciler import RUN_STATUS_CALENDAR_PREFIX as _RUN_STATUS_CALENDAR_PREFIX
from solsys_code.campaign_reconciler import (
    ReconcileResult,
    _link_event_to_run,
    _may_write,
    event_description,
    split_telescope_instrument,
)
from solsys_code.models import CampaignRun
from solsys_code.telescope_runs import sun_event

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


def project_allocation(run: CampaignRun, *, dry_run: bool = False) -> tuple[ReconcileResult, set[str]]:
    """Project (or refresh) every night in ``[run.window_start, run.window_end]`` inclusive
    into its own ``ALLOC:{run.pk}:{night}`` sunset->sunrise ``CalendarEvent``.

    Per-night resolution order: ownership (``_may_write()``) is decided BEFORE any write --
    a blocked night is counted and its url added to the active set (so a foreign
    attribution is never detached out from under it), never written. ``sun_event()`` (both
    ``'sun'`` and ``'dark'``) is called only when a night is being CREATED -- never on the
    update path (D-13): an existing night's ``start_time``/``end_time`` are never rewritten.

    Field authority: on **create**, writes ``title``, ``description``, ``target_list``,
    ``telescope``, ``instrument``, ``start_time``, ``end_time``. On **update**, writes only
    ``title``, ``description`` (with the preserved dark-window line) and ``target_list``.

    ``sun_event()``'s ``ValueError`` (e.g. a blank ``Observatory.timezone``) is deliberately
    NOT caught here -- it keeps propagating out of ``reconcile_run()`` for the staff-action
    call sites, unchanged from 29 D-06.

    Args:
        run: the ``CampaignRun`` to project. Must have a resolved ``site``, an approved
            status and a non-null ``window_start``/``window_end`` (``reconcile_run()``'s
            stage-0 guard, ``_skip_reason()``, already enforces this before dispatch).
        dry_run: when True, report what would change without writing anything.

    Returns:
        tuple[ReconcileResult, set[str]]: the outcome, and the exact set of
        ``CalendarEvent.url`` values this call considers current -- every night visited,
        including a blocked night and every night visited in ``dry_run``.
    """
    totals: dict[str, int] = {'created': 0, 'updated': 0, 'unchanged': 0, 'blocked': 0}
    n_nights = (run.window_end - run.window_start).days + 1
    active_urls: set[str] = set()

    for i in range(n_nights):
        night = run.window_start + timedelta(days=i)
        url = allocation_night_url(run, night)
        existing = CalendarEvent.objects.filter(url=url).first()

        if not _may_write(existing, run):
            logger.warning('Allocation blocked: event pk=%s is not owned by run pk=%s.', existing.pk, run.pk)
            totals['blocked'] += 1
            active_urls.add(url)
            continue

        active_urls.add(url)

        if existing is None:
            sunset, sunrise = sun_event(run.site, night, kind='sun')
            dark_start, dark_end = sun_event(run.site, night, kind='dark')
            dark_start_iso = dark_start.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0).isoformat()
            dark_end_iso = dark_end.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0).isoformat()
            dark_line = f'{_DARK_WINDOW_PREFIX}{dark_start_iso} to {dark_end_iso}'
            telescope, instrument = split_telescope_instrument(run.telescope_instrument)
            fields: dict[str, Any] = {
                'title': allocation_night_title(run),
                'description': allocation_night_description(run, dark_line),
                'target_list': run.campaign,
                'telescope': telescope,
                'instrument': instrument,
                'start_time': sunset.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0),
                'end_time': sunrise.to_datetime(timezone=dt_timezone.utc).replace(microsecond=0),
            }
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

    return ReconcileResult(**totals), active_urls
