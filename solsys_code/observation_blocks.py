"""FOMO's own rule for which observation block a request's schedule comes from.

Why this module exists (UAT gap G-37.1-1-alloc, and the user's 2026-10-04 decision): TOM Toolkit's
``OCSFacility.get_observation_status()`` keeps only a COMPLETED block, else the last PENDING block,
and returns empty times for everything else -- including ABORTED, the portal state of a block that
started, took data and stopped early, and IN_PROGRESS. A request whose only block aborted was
therefore stored with no ``scheduled_start``/``scheduled_end``, Phase 35 D-05's ``retired_nights()``
retired nothing, and the run's ``ALLOC:`` night stayed beside the observation's own calendar entry.
Under Phase 35 D-05/D-06 a block that was placed and ABORTED after taking data retires its night
exactly like a COMPLETED one: ABORTED and IN_PROGRESS count as "placed-or-observed".

The rule, in tiers (see :func:`select_schedule_block`): the first COMPLETED block, else the last
ABORTED or IN_PROGRESS block, else the last PENDING block, else none. A request that never got a
block at all therefore keeps no times and retires nothing (Phase 35 D-05 unchanged).

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
finished state through one of them is stored without its aborted block's times, and nothing on a
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
    ``status_vocabulary.OCSState``.
    """

    COMPLETED = 'COMPLETED'
    ABORTED = 'ABORTED'
    IN_PROGRESS = 'IN_PROGRESS'
    PENDING = 'PENDING'


class UnexpectedBlockPayloadError(ValueError):
    """Raised when the portal's ``/observations/`` reply for a request is not a list.

    A dict error body or a paginated ``{'results': [...]}`` envelope is a failed lookup, never "no
    block": reading it as an empty list would erase a stored time and commit a status change without
    its block. It subclasses ``ValueError`` so the handlers that already catch it (and
    ``calendar_utils``'s own ``ValueError`` catch) keep working, and it is named so the unattended
    runner's failure line says what happened. The message carries only the request id and the body's
    Python type name, never the body, which can hold portal text.
    """


def select_schedule_block(blocks: Any) -> dict[str, Any] | None:
    """Choose the block a request's scheduled start and end come from.

    The first COMPLETED block wins. Otherwise the LAST block that is ABORTED or IN_PROGRESS (a
    block that is running now, or that started and was aborted after taking data). Otherwise the
    LAST PENDING block. Otherwise None. "Last" mirrors TOM's last-PENDING convention. An earlier
    aborted block beats a later pending block that has not run yet: data already taken outranks
    intent, and once the pending block runs it becomes COMPLETED, which wins, or ABORTED, which as
    the later block replaces the earlier one.

    Args:
        blocks: the decoded JSON of the portal's observations list; anything that is not a list
            gives None, and entries that are not dicts or carry no state are ignored.

    Returns:
        dict[str, Any] | None: the chosen block dict, or None when no block is COMPLETED, ABORTED,
            IN_PROGRESS or PENDING.
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
        if state in (BlockState.ABORTED, BlockState.IN_PROGRESS):
            last_started = block
        elif state == BlockState.PENDING:
            last_pending = block
    return last_started if last_started is not None else last_pending


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
    """TOM's LCO facility reading the observed block with FOMO's rule (an aborted block counts)."""


class FomoSOARFacility(ScheduleBlockRuleMixin, SOARFacility):
    """TOM's SOAR facility reading the observed block with FOMO's rule (an aborted block counts)."""
