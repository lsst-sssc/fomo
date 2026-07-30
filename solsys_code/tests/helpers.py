"""Fixture builders shared by more than one test module in ``solsys_code/tests/``.

IN-02: ``test_calendar_utils`` previously imported ``_observations_block_response`` directly
from ``test_sync_lco_observation_calendar``. That coupled the two modules' import graphs --
any import-time failure in the sync command's test module (a renamed facility import, a
missing fixture) would also fail ``test_calendar_utils``, for a helper that has nothing to do
with the sync command. Shared fixture builders belong here instead.

This module is deliberately NOT named ``test_*.py``, so neither the Django test runner's
default ``test*.py`` pattern nor pytest collects it as a test module.
"""

from unittest.mock import MagicMock


def observations_block_response(
    site: str = 'lsc',
    enclosure: str = 'doma',
    telescope: str = '1m0a',
    state: str = 'COMPLETED',
) -> MagicMock:
    """Build a mock make_request() response for /api/requests/{id}/observations/.

    Args:
        site: 3-letter site code for the single returned block.
        enclosure: 4-char enclosure code for the single returned block.
        telescope: 4-char telescope code for the single returned block.
        state: the block's 'state' value (e.g. 'COMPLETED', 'PENDING').

    Returns:
        MagicMock: a response double whose .json() returns a one-element list
            containing the block dict built from the given keyword args.
    """
    response = MagicMock()
    response.json.return_value = [{'site': site, 'enclosure': enclosure, 'telescope': telescope, 'state': state}]
    return response
