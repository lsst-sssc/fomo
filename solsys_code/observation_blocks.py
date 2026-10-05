"""FOMO's own rule for which observation block a request's schedule comes from.

Why this module exists (UAT gaps G-37.1-1-alloc and G-37.1-6, and the user's 2026-10-04 and 2026-10-05
decisions): TOM Toolkit's ``OCSFacility.get_observation_status()`` keeps only a COMPLETED block, else the last
PENDING block, and returns empty times for everything else -- including IN_PROGRESS and the two states the
portal gives a block that started, took data and stopped early: ABORTED, and, as on every such July 2026
Didymos block (verified on the live portal 2026-10-05), FAILED. A request whose only block stopped early was
therefore stored with no ``scheduled_start``/``scheduled_end``, Phase 35 D-05's ``retired_nights()`` retired
nothing, and the run's ``ALLOC:`` night stayed beside the observation's own calendar entry. Under Phase 35
D-05/D-06 a block that was placed and stopped early after taking data retires its night exactly like a
COMPLETED one: ABORTED, IN_PROGRESS and FAILED-with-data count as "placed-or-observed".

A FAILED block counts only when it took data: some configuration's summary reports ``time_completed`` above
zero (developer decision 2026-10-05). It then ranks with ABORTED and IN_PROGRESS. A FAILED block with nothing
completed gives no times, and neither does a block the portal never attempted (NOT_ATTEMPTED) or cancelled
(CANCELED), whatever else it carries.

The rule, in tiers (see :func:`select_schedule_block`): the first COMPLETED block, else the last PENDING block,
else the last ABORTED, IN_PROGRESS or FAILED-with-data block, else none (developer decision 2026-10-05 on review
WR-19, "placed block wins"; it reverses the 2026-10-04/05 order, which put the started tier above PENDING). While a
request is still pending, its record follows the block the scheduler has placed: drawn as scheduled on the
upcoming night, whose allocation night retires. An aborted, in-progress or failed-with-data block counts only once
no pending block remains. A record carries one block, so while a request is placed again the night of its earlier
aborted or failed block is not retired, and it stays unretired if the placed block completes (Phase 35 D-06 gives
ground here). A request with no block that counts keeps no times and retires nothing (Phase 35 D-05 unchanged).

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


def select_schedule_block(blocks: Any) -> dict[str, Any] | None:
    """Choose the block a request's scheduled start and end come from.

    The first COMPLETED block wins. Otherwise the LAST PENDING block: while the request is still pending, the
    block the scheduler has placed is its live schedule (developer decision 2026-10-05, review WR-19 "placed
    block wins"; this reverses the 2026-10-04/05 order, in which an earlier started block beat a later pending
    one). Otherwise the LAST block that is ABORTED or IN_PROGRESS (a block that is running now, or that started
    and was aborted after taking data), or that is FAILED after taking data (some configuration's summary
    reports ``time_completed`` above zero; the portal reports every such July 2026 Didymos block as FAILED, not
    ABORTED). Otherwise None. An IN_PROGRESS block yields to a PENDING one too: the LCO scheduler places a new
    block for a request whose block is still running only once that block has had a configuration fail, so it is
    about to stop early (37.1-14 A-24).

    The portal lists a request's blocks in creation order (the observation portal orders them by block id).
    "First" and "last" lean on that order only within a tier -- the first COMPLETED block, the last PENDING
    block and the last block of the started tier, mirroring TOM's last-PENDING convention. Which tier wins never
    depends on position.

    A FAILED block with no usable ``time_completed`` above zero (missing, None, zero, negative, NaN, a
    string, a bool, or a summary or ``configuration_statuses`` of the wrong type) gives no times and is
    skipped exactly like an unknown state. A NOT_ATTEMPTED or CANCELED block never gives times, even when it
    carries a ``time_completed``.

    Args:
        blocks: the decoded JSON of the portal's observations list; anything that is not a list gives None,
            and entries that are not dicts or carry no state are ignored.

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
        list that is not a list at all is a failed lookup.

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
        block = select_schedule_block(blocks)
        if block is None:
            return {'state': state, 'scheduled_start': None, 'scheduled_end': None}
        return {'state': state, 'scheduled_start': block.get('start'), 'scheduled_end': block.get('end')}


class FomoLCOFacility(ScheduleBlockRuleMixin, LCOFacility):
    """TOM's LCO facility reading the observed block with FOMO's rule (a block that took data counts)."""


class FomoSOARFacility(ScheduleBlockRuleMixin, SOARFacility):
    """TOM's SOAR facility reading the observed block with FOMO's rule (a block that took data counts)."""
