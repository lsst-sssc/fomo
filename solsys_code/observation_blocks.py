"""FOMO's own rule for which observation block a request's schedule comes from.

Why this module exists (UAT gaps G-37.1-1-alloc and G-37.1-6, and the user's 2026-10-04 and 2026-10-05
decisions): TOM Toolkit's ``OCSFacility.get_observation_status()`` keeps only a COMPLETED block, else the last
PENDING block, and returns empty times for everything else -- including IN_PROGRESS and the two states the
portal gives a block that started, took data and stopped early: ABORTED, and, as on every such July 2026
Didymos block (verified on the live portal 2026-10-05), FAILED. A request whose only block stopped early was
therefore stored with no ``scheduled_start``/``scheduled_end``, Phase 35 D-05's ``retired_nights()`` retired
nothing, and the run's ``ALLOC:`` night stayed beside the observation's own calendar entry. Under Phase 35
D-05/D-06 a block that was placed and stopped early after taking data counts as "placed-or-observed" (ABORTED,
IN_PROGRESS and FAILED-with-data), so it retires its night like a COMPLETED one whenever it is the block the rule
below chooses: a request that can still run follows a later pending block instead, and that earlier night then
comes back, while a finished request keeps the block that took data over a pending block the portal still lists.

A FAILED block counts only when it took data: some configuration's summary reports ``time_completed`` above
zero (developer decision 2026-10-05). It then ranks with ABORTED and IN_PROGRESS. A FAILED block with nothing
completed gives no times, and neither does a block the portal never attempted (NOT_ATTEMPTED) or cancelled
(CANCELED), whatever else it carries.

The rule, in tiers (see :func:`select_schedule_block`), depends on whether the request is finished: its state is
one of the facility's terminal observing states (:func:`is_request_finished`; for LCO and SOAR, COMPLETED,
WINDOW_EXPIRED, CANCELED, FAILURE_LIMIT_REACHED and NOT_ATTEMPTED). While the request can still run, the order is the
first COMPLETED block, else the last PENDING block, else the last ABORTED, IN_PROGRESS or FAILED-with-data block, else
none (developer decision 2026-10-05 on review WR-19, "placed block wins"; it reverses the 2026-10-04/05 order, which
put the started tier above PENDING). Once the request is finished, the order is the first COMPLETED block, else the
last ABORTED, IN_PROGRESS or FAILED-with-data block, else the last leftover PENDING block, else none (developer
decision 2026-10-05 on review WR-20: a finished request runs no further block, so a block that took data outranks a
pending block the portal still lists, and the tick at which the request finishes is the last one that reads its
record). While a request can still run (its state is not terminal; for LCO and SOAR that is PENDING), its record
follows the block the scheduler has placed: drawn as scheduled on the upcoming night, whose allocation night retires.
An aborted, in-progress or failed-with-data block counts once no pending block remains, or once the request is
finished. A record carries one block, so while a request is placed again the night of its earlier aborted or failed
block is not retired, and it stays unretired if the placed block completes (Phase 35 D-06 gives ground here); if the
request instead finishes with the placed block still listed as pending, the record goes back to the block that took
data and that night retires again. A finished request whose only timed block is a leftover PENDING block keeps that
block's times, as every earlier rule did (37.1-15 A-33). A request with no block that counts keeps no times and
retires nothing (Phase 35 D-05 unchanged).

FOMO never edits or monkeypatches TOM Toolkit's installed code. The rule lives only on FOMO's own
facility subclasses, :class:`FomoLCOFacility` and :class:`FomoSOARFacility`, which override the one
method TOM's ``update_observation_status()`` and ``update_all_observation_statuses()`` both call.

Those subclasses are deliberately NOT registered in ``settings.TOM_FACILITY_CLASSES``, so the rule
covers FOMO-owned paths only: FOMO's commands and the unattended runner build these classes
directly. Every TOM Toolkit route that resolves the facility through ``get_service_class()`` keeps
TOM's rule: its stock ``updatestatus`` command, the "Update status" button on its observation list
page (``tom_observations/views.py``, which runs that command), the Cancel button on its observation
page (``ObservationRecordCancelView``), its REST cancel route (``PATCH /api/observations/<pk>/cancel/``,
``tom_observations/api_views.py``) and ``ObservationRecord.update_status()``. A request moved to a
finished state through one of them is stored without the times of a block that took data, and nothing on a
tick looks at a finished record again, so its allocation night comes back. An operator recovers an
LCO record by re-running the backfill command for its proposal with ``--recheck-unscheduled``.

A portal reply whose block list is not a list (a dict error body, or a paginated ``{'results': [...]}``
envelope) is a failed lookup, not "no block": :meth:`ScheduleBlockRuleMixin.get_observation_status` raises
:class:`UnexpectedBlockPayloadError`, so every caller's existing failure handling applies. The discovery
sweep holds the status change back and keeps the stored times, TOM's update loop writes nothing and the
unattended runner counts the record as failed, and the Didymos command reports it under
``status sync failed``. :func:`select_schedule_block` itself stays tolerant (anything that is not a list
gives None) for the embedded-list caller and ``calendar_utils.resolve_placement_block()``, which must never
raise.
"""

from typing import Any
from urllib.parse import urljoin

from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.ocs import make_request
from tom_observations.facilities.soar import SOARFacility


class BlockState:
    """Observation-BLOCK states from ``/api/requests/{id}/observations/``.

    These are states of a scheduled block, not of a request; request states stay in
    ``status_vocabulary.OCSState``. The portal's block states are PENDING, IN_PROGRESS, NOT_ATTEMPTED,
    COMPLETED, ABORTED, FAILED and CANCELED. The rule names five of them; NOT_ATTEMPTED and CANCELED never
    give times, so they have no member here.
    """

    COMPLETED = 'COMPLETED'
    ABORTED = 'ABORTED'
    IN_PROGRESS = 'IN_PROGRESS'
    PENDING = 'PENDING'
    FAILED = 'FAILED'


class UnexpectedBlockPayloadError(ValueError):
    """Raised when the portal's ``/observations/`` reply for a request is not a list.

    A dict error body or a paginated ``{'results': [...]}`` envelope is a failed lookup, never "no
    block": reading it as an empty list would erase a stored time and commit a status change without
    its block. It subclasses ``ValueError`` so the handlers that already catch it (and
    ``calendar_utils``'s own ``ValueError`` catch) keep working, and it is named so the unattended
    runner's failure line says what happened. The message carries only the request id and the body's
    Python type name, never the body, which can hold portal text.
    """


def _block_took_data(block: dict[str, Any]) -> bool:
    """Say whether a block's configurations report any completed time.

    True when ``block['configuration_statuses']`` is a list with at least one entry that is a dict whose
    ``summary`` is a dict whose ``time_completed`` is an ``int`` or ``float`` (not a ``bool``) above zero.
    Every other value at every level -- missing, None, the wrong type, a numeric string, a bool, NaN, zero or
    a negative number -- reads as "no data", so a block with an unusable summary never gives times and the
    helper never raises. It reads only ``time_completed``: never the free-text ``reason`` and never the
    summary's own ``state`` (the portal puts ABORTED inside some FAILED blocks).

    Args:
        block: one block dict from the portal's observations list.

    Returns:
        bool: True when some configuration completed some time.
    """
    statuses = block.get('configuration_statuses')
    if not isinstance(statuses, list):
        return False
    for status in statuses:
        if not isinstance(status, dict):
            continue
        summary = status.get('summary')
        if not isinstance(summary, dict):
            continue
        completed = summary.get('time_completed')
        if isinstance(completed, bool) or not isinstance(completed, int | float):
            continue
        # NaN is not greater than zero, so it reads as no data without a separate check.
        if completed > 0:
            return True
    return False


def is_request_finished(state: Any, facility: Any) -> bool:
    """Say whether a request is finished, that is, will run no further block.

    A finished request is one whose state is among the facility's terminal observing states (developer decision
    2026-10-05, review WR-20). For LCO and SOAR those are COMPLETED, WINDOW_EXPIRED, CANCELED,
    FAILURE_LIMIT_REACHED and NOT_ATTEMPTED: TOM Toolkit's ``get_terminal_observing_states()``, the list its own
    update loop uses to stop refreshing a record. Any other value, including PENDING, None or a state the portal
    adds later, reads as "can still run". List membership never hashes, so an odd state value never raises.

    Args:
        state: the portal's request state, or a stored record's status.
        facility: any facility instance that has ``get_terminal_observing_states()``.

    Returns:
        bool: True when the request is finished.
    """
    return state in facility.get_terminal_observing_states()


def select_schedule_block(blocks: Any, *, request_finished: bool = False) -> dict[str, Any] | None:
    """Choose the block a request's scheduled start and end come from.

    The first COMPLETED block wins. Otherwise the LAST PENDING block: while the request can still run
    (``request_finished`` False, the default: its state is not one of the facility's terminal states, which for
    LCO and SOAR means PENDING), the block the scheduler has placed is its live schedule (developer decision
    2026-10-05, review WR-19 "placed block wins"; this reverses the 2026-10-04/05 order, in which an earlier
    started block beat a later pending one). Otherwise the LAST block that is ABORTED or IN_PROGRESS (a block
    that is running now, or that started and was aborted after taking data), or that is FAILED after taking data
    (some configuration's summary reports ``time_completed`` above zero; the portal reports every such July 2026
    Didymos block as FAILED, not ABORTED). Otherwise None. An IN_PROGRESS block yields to a PENDING one too: the
    LCO scheduler places a new block for a request whose block is still running only once that block has had a
    configuration fail, so it is about to stop early (37.1-14 A-24).

    Once the request is finished (``request_finished`` True: COMPLETED, WINDOW_EXPIRED, CANCELED,
    FAILURE_LIMIT_REACHED or NOT_ATTEMPTED; see :func:`is_request_finished`) it will run no further block, so the
    two middle tiers swap: after the first COMPLETED block comes the LAST ABORTED, IN_PROGRESS or
    FAILED-with-data block, and only then the LAST PENDING block, a leftover the portal still lists (developer
    decision 2026-10-05, review WR-20). This matters at the one tick that is never repeated, the request's change
    to a finished state: a finished record that holds both times is not looked up again. A finished request whose
    only timed block is a leftover PENDING block still gets that block's times (37.1-15 A-33).

    The portal lists a request's blocks in creation order (the observation portal orders them by block id).
    "First" and "last" lean on that order only within a tier -- the first COMPLETED block, the last PENDING
    block and the last block of the started tier, mirroring TOM's last-PENDING convention. Which tier wins never
    depends on position; it depends only on whether the request is finished.

    A FAILED block with no usable ``time_completed`` above zero (missing, None, zero, negative, NaN, a
    string, a bool, or a summary or ``configuration_statuses`` of the wrong type) gives no times and is
    skipped exactly like an unknown state. A NOT_ATTEMPTED or CANCELED block never gives times, even when it
    carries a ``time_completed``.

    Args:
        blocks: the decoded JSON of the portal's observations list; anything that is not a list gives None,
            and entries that are not dicts or carry no state are ignored.
        request_finished: keyword-only; True when the request's state is one of the facility's terminal
            observing states, so a block that took data outranks a leftover PENDING block. Defaults to False,
            the order for a request that can still run.

    Returns:
        dict[str, Any] | None: the chosen block dict, or None when no block is COMPLETED, PENDING, ABORTED,
            IN_PROGRESS, or FAILED after taking data.
    """
    if not isinstance(blocks, list):
        return None
    last_started = None
    last_pending = None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        state = block.get('state')
        if state == BlockState.COMPLETED:
            return block
        if state in (BlockState.ABORTED, BlockState.IN_PROGRESS) or (
            state == BlockState.FAILED and _block_took_data(block)
        ):
            last_started = block
        elif state == BlockState.PENDING:
            last_pending = block
    if request_finished:
        return last_started if last_started is not None else last_pending
    return last_pending if last_pending is not None else last_started


class ScheduleBlockRuleMixin:
    """Replace TOM's block choice in ``get_observation_status()`` with :func:`select_schedule_block`.

    A block-list reply that is not a list raises :class:`UnexpectedBlockPayloadError` instead of reading
    as "no block".
    """

    def get_observation_status(self, observation_id: str) -> dict[str, Any]:
        """Return the request state and the chosen block's start and end.

        Makes the same two portal GETs TOM's own implementation makes; ``make_request``'s
        exceptions propagate exactly as TOM's do. An empty block list still means "no block"; a block
        list that is not a list at all is a failed lookup. The rule is told whether the request is finished
        from the state the first GET returns (:func:`is_request_finished`, review WR-20).

        Args:
            observation_id: the LCO request id.

        Returns:
            dict[str, Any]: ``{'state', 'scheduled_start', 'scheduled_end'}`` with the raw portal
                strings, or None for both times when no block qualifies.

        Raises:
            UnexpectedBlockPayloadError: when the ``/observations/`` reply is not a list (for example a
                dict error body or a paginated envelope). The message names only the request id and the
                body's Python type.
        """
        portal_url = self.facility_settings.get_setting('portal_url')
        response = make_request(
            'GET',
            urljoin(portal_url, f'/api/requests/{observation_id}'),
            headers=self._portal_headers(),
        )
        state = response.json()['state']

        response = make_request(
            'GET',
            urljoin(portal_url, f'/api/requests/{observation_id}/observations/'),
            headers=self._portal_headers(),
        )
        blocks = response.json()
        if not isinstance(blocks, list):
            raise UnexpectedBlockPayloadError(
                f'observations reply for request {observation_id} is a {type(blocks).__name__}, not a list'
            )
        block = select_schedule_block(blocks, request_finished=is_request_finished(state, self))
        if block is None:
            return {'state': state, 'scheduled_start': None, 'scheduled_end': None}
        return {'state': state, 'scheduled_start': block.get('start'), 'scheduled_end': block.get('end')}


class FomoLCOFacility(ScheduleBlockRuleMixin, LCOFacility):
    """TOM's LCO facility reading the observed block with FOMO's rule (a block that took data counts)."""


class FomoSOARFacility(ScheduleBlockRuleMixin, SOARFacility):
    """TOM's SOAR facility reading the observed block with FOMO's rule (a block that took data counts)."""
