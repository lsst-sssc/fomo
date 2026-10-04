from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.soar import SOARFacility
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.observation_blocks import (
    BlockState,
    FomoLCOFacility,
    FomoSOARFacility,
    select_schedule_block,
)


def _block(state, start='2026-07-01T01:00:00Z', end='2026-07-01T01:40:00Z'):
    return {'state': state, 'start': start, 'end': end}


def portal_side_effect(request_states, blocks_by_request):
    """Build a make_request side_effect that answers by URL, as the two portal GETs do.

    Args:
        request_states: dict of request id (str) to the request state string.
        blocks_by_request: dict of request id (str) to the list of block dicts.

    Returns:
        A function usable as the ``side_effect`` of a patched ``make_request``.
    """

    def _make_request(method, url, **kwargs):
        response = MagicMock()
        if url.endswith('/observations/'):
            request_id = url.rsplit('/', 3)[-3]
            response.json.return_value = blocks_by_request.get(request_id, [])
        else:
            request_id = url.rsplit('/', 1)[-1]
            response.json.return_value = {'state': request_states[request_id]}
        return response

    return _make_request


class TestSelectScheduleBlock(SimpleTestCase):
    def test_empty_and_non_list_give_none(self):
        self.assertIsNone(select_schedule_block([]))
        self.assertIsNone(select_schedule_block(None))
        self.assertIsNone(select_schedule_block({'state': 'COMPLETED'}))
        self.assertIsNone(select_schedule_block('COMPLETED'))

    def test_blocks_that_never_ran_give_none(self):
        self.assertIsNone(select_schedule_block([_block('CANCELED')]))
        self.assertIsNone(select_schedule_block([_block('NOT_ATTEMPTED'), _block('FAILED')]))

    def test_single_aborted_block_is_chosen(self):
        aborted = _block(BlockState.ABORTED)
        self.assertIs(select_schedule_block([aborted]), aborted)

    def test_aborted_beats_pending_in_either_order(self):
        aborted, pending = _block(BlockState.ABORTED), _block(BlockState.PENDING, start='2026-07-05T01:00:00Z')
        self.assertIs(select_schedule_block([pending, aborted]), aborted)
        # An earlier aborted block (data already taken) outranks a later pending block that has not run.
        self.assertIs(select_schedule_block([aborted, pending]), aborted)

    def test_last_of_two_aborted_blocks_is_chosen(self):
        first, last = _block(BlockState.ABORTED), _block(BlockState.ABORTED, start='2026-07-02T01:00:00Z')
        self.assertIs(select_schedule_block([first, last]), last)

    def test_completed_beats_aborted(self):
        aborted, completed = _block(BlockState.ABORTED), _block(BlockState.COMPLETED, start='2026-07-03T01:00:00Z')
        self.assertIs(select_schedule_block([aborted, completed]), completed)
        self.assertIs(select_schedule_block([completed, aborted]), completed)

    def test_first_of_two_completed_blocks_is_chosen(self):
        first, last = _block(BlockState.COMPLETED), _block(BlockState.COMPLETED, start='2026-07-02T01:00:00Z')
        self.assertIs(select_schedule_block([first, last]), first)

    def test_last_of_two_pending_blocks_is_chosen(self):
        first, last = _block(BlockState.PENDING), _block(BlockState.PENDING, start='2026-07-02T01:00:00Z')
        self.assertIs(select_schedule_block([first, last]), last)

    def test_in_progress_beats_pending(self):
        in_progress, pending = _block(BlockState.IN_PROGRESS), _block(BlockState.PENDING, start='2026-07-02T01:00:00Z')
        self.assertIs(select_schedule_block([in_progress, pending]), in_progress)

    def test_entries_that_are_not_dicts_or_carry_no_state_are_ignored(self):
        pending = _block(BlockState.PENDING)
        self.assertIs(select_schedule_block(['junk', None, {'start': 'x'}, pending, 42]), pending)
        self.assertIsNone(select_schedule_block(['junk', None, {'start': 'x'}]))


class TestFomoFacilityStatus(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.target = NonSiderealTargetFactory.create(name='Didymos')

    def test_class_identity(self):
        self.assertEqual(FomoLCOFacility().name, 'LCO')
        self.assertEqual(FomoSOARFacility().name, 'SOAR')
        self.assertIsInstance(FomoLCOFacility(), LCOFacility)
        self.assertIsInstance(FomoSOARFacility(), SOARFacility)

    @patch('solsys_code.observation_blocks.make_request')
    def test_status_carries_the_aborted_block_times(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect({'123': 'WINDOW_EXPIRED'}, {'123': [_block('ABORTED')]})

        status = FomoLCOFacility().get_observation_status('123')

        self.assertEqual(
            status,
            {
                'state': 'WINDOW_EXPIRED',
                'scheduled_start': '2026-07-01T01:00:00Z',
                'scheduled_end': '2026-07-01T01:40:00Z',
            },
        )
        self.assertEqual(mock_make_request.call_count, 2)
        self.assertTrue(mock_make_request.call_args_list[1].args[1].endswith('/api/requests/123/observations/'))

    @patch('solsys_code.observation_blocks.make_request')
    def test_status_with_no_blocks_has_no_times(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect({'123': 'CANCELED'}, {'123': []})

        status = FomoLCOFacility().get_observation_status('123')

        self.assertEqual(status, {'state': 'CANCELED', 'scheduled_start': None, 'scheduled_end': None})

    @patch('solsys_code.observation_blocks.make_request')
    def test_update_observation_status_saves_the_aborted_block_times(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect({'123': 'WINDOW_EXPIRED'}, {'123': [_block('ABORTED')]})
        record = ObservationRecord.objects.create(
            target=self.target, facility='LCO', observation_id='123', status='WINDOW_EXPIRED', parameters={}
        )
        self.assertIsNone(record.scheduled_start)

        FomoLCOFacility().update_observation_status('123')

        record.refresh_from_db()
        self.assertEqual(record.status, 'WINDOW_EXPIRED')
        self.assertIsNotNone(record.scheduled_start)
        self.assertIsNotNone(record.scheduled_end)
