from datetime import datetime, timezone
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
    is_request_finished,
    select_schedule_block,
)


def _block(state, start='2026-07-01T01:00:00Z', end='2026-07-01T01:40:00Z'):
    return {'state': state, 'start': start, 'end': end}


# Real block dicts, one per request, from read-only portal GETs of /api/requests/{id}/observations/ made on
# 2026-10-05 for the LCO2026A-003 Didymos requests (UAT gap G-37.1-6): the portal reports a block that started,
# took data and stopped early as FAILED, not ABORTED. Each request's only block is FAILED, its one configuration
# status carries a summary with time_completed (seconds). The block ``id`` and ``priority`` values were not
# recorded in the reply notes, so those two are arbitrary; ``site``, ``enclosure`` and ``telescope`` are the
# Siding Spring 2 m values the run implies. The rule reads none of these.
REAL_FAILED_BLOCKS = {
    '4253588': {
        'id': 900000001,
        'priority': 10,
        'request': 4253588,
        'site': 'coj',
        'enclosure': 'clma',
        'telescope': '2m0a',
        'state': 'FAILED',
        'start': '2026-07-12T08:55:50Z',
        'end': '2026-07-12T15:00:31Z',
        'configuration_statuses': [
            {
                'state': 'FAILED',
                'summary': {
                    'state': 'FAILED',
                    'start': '2026-07-12T08:55:50Z',
                    'end': '2026-07-12T14:10:03Z',
                    'time_completed': 18060.0,
                    'reason': 'Error while executing OffsetCommand (only a fragment was recorded)',
                },
            }
        ],
    },
    '4272067': {
        'id': 900000002,
        'priority': 10,
        'request': 4272067,
        'site': 'coj',
        'enclosure': 'clma',
        'telescope': '2m0a',
        'state': 'FAILED',
        'start': '2026-07-17T09:07:48Z',
        'end': '2026-07-17T14:54:29Z',
        'configuration_statuses': [
            {
                'state': 'FAILED',
                'summary': {
                    'state': 'ABORTED',
                    'end': '2026-07-17T09:47:30Z',
                    'time_completed': 2160.0,
                    'reason': 'Aborting observation: Enclosure no longer open.',
                },
            }
        ],
    },
    '4276100': {
        'id': 900000003,
        'priority': 10,
        'request': 4276100,
        'site': 'coj',
        'enclosure': 'clma',
        'telescope': '2m0a',
        'state': 'FAILED',
        'start': '2026-07-19T08:57:45Z',
        'end': '2026-07-19T14:44:26Z',
        'configuration_statuses': [{'state': 'FAILED', 'summary': {'time_completed': 19440.0}}],
    },
    '4282342': {
        'id': 900000004,
        'priority': 10,
        'request': 4282342,
        'site': 'coj',
        'enclosure': 'clma',
        'telescope': '2m0a',
        'state': 'FAILED',
        'start': '2026-07-20T09:11:47Z',
        'end': '2026-07-20T14:16:28Z',
        'configuration_statuses': [{'state': 'FAILED', 'summary': {'time_completed': 14850.0}}],
    },
}


def failed_block(time_completed, start='2026-07-01T01:00:00Z', end='2026-07-01T01:40:00Z', summary_state='FAILED'):
    """Build a FAILED block with one configuration status whose summary reports ``time_completed``.

    Args:
        time_completed: the value of the summary's ``time_completed`` (any type, to test odd portal data).
        start: the block's start string.
        end: the block's end string.
        summary_state: the configuration summary's own state (the portal says ABORTED inside some FAILED blocks).

    Returns:
        dict: a block dict in the portal's data model.
    """
    return {
        'state': 'FAILED',
        'start': start,
        'end': end,
        'configuration_statuses': [
            {'state': 'FAILED', 'summary': {'state': summary_state, 'time_completed': time_completed}}
        ],
    }


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
        self.assertIsNone(select_schedule_block([_block('NOT_ATTEMPTED'), _block('CANCELED')]))

    def test_failed_block_without_time_completed_gives_none(self):
        """WR-04: a FAILED block counts only when it took data; every unusable value reads as no data."""
        cases = {
            'bare FAILED block': _block(BlockState.FAILED),
            'time_completed 0': failed_block(0),
            'time_completed 0.0': failed_block(0.0),
            'time_completed negative': failed_block(-1.0),
            'time_completed None': failed_block(None),
            'time_completed numeric string': failed_block('18060.0'),
            'time_completed bool': failed_block(True),
            'time_completed NaN': failed_block(float('nan')),
        }
        for status_value in (None, 'missing', [5.0]):
            block = failed_block(5.0)
            if status_value == 'missing':
                del block['configuration_statuses'][0]['summary']
            else:
                block['configuration_statuses'][0]['summary'] = status_value
            cases[f'summary {status_value!r}'] = block
        no_time = failed_block(5.0)
        del no_time['configuration_statuses'][0]['summary']['time_completed']
        cases['time_completed missing'] = no_time
        for value in (None, {'summary': {'time_completed': 5.0}}, ['junk', None, 3]):
            block = failed_block(5.0)
            block['configuration_statuses'] = value
            cases[f'configuration_statuses {value!r}'] = block
        for label, block in cases.items():
            with self.subTest(label):
                self.assertIsNone(select_schedule_block([block]))

    def test_each_real_failed_block_is_chosen(self):
        # 4272067's configuration summary says ABORTED inside a FAILED block; its time completed is what counts.
        for request_id, block in REAL_FAILED_BLOCKS.items():
            with self.subTest(request_id):
                self.assertIs(select_schedule_block([block]), block)

    def test_failed_block_is_chosen_when_any_configuration_completed_time(self):
        block = failed_block(0.0)
        block['configuration_statuses'].append({'state': 'FAILED', 'summary': {'time_completed': 120.0}})
        self.assertIs(select_schedule_block([block]), block)
        self.assertIsNotNone(select_schedule_block([failed_block(30)]))

    def test_not_attempted_and_canceled_blocks_never_give_times(self):
        for state in ('NOT_ATTEMPTED', 'CANCELED'):
            with self.subTest(state):
                block = failed_block(18060.0)
                block['state'] = state
                self.assertIsNone(select_schedule_block([block]))

    def test_failed_with_data_ranks_with_aborted_and_in_progress(self):
        aborted = _block(BlockState.ABORTED)
        failed = failed_block(18060.0, start='2026-07-02T01:00:00Z')
        self.assertIs(select_schedule_block([aborted, failed]), failed)
        self.assertIs(select_schedule_block([failed, aborted]), aborted)
        in_progress = _block(BlockState.IN_PROGRESS)
        self.assertIs(select_schedule_block([failed, in_progress]), in_progress)

    def test_a_pending_block_beats_a_failed_block_that_took_data_in_either_order(self):
        """WR-19, developer decision 2026-10-05: the placed block wins while the request is pending."""
        failed, pending = failed_block(18060.0), _block(BlockState.PENDING, start='2026-07-05T01:00:00Z')
        self.assertIs(select_schedule_block([failed, pending]), pending)
        self.assertIs(select_schedule_block([pending, failed]), pending)

    def test_completed_beats_failed_with_data_in_either_order(self):
        failed, completed = failed_block(18060.0), _block(BlockState.COMPLETED, start='2026-07-03T01:00:00Z')
        self.assertIs(select_schedule_block([failed, completed]), completed)
        self.assertIs(select_schedule_block([completed, failed]), completed)

    def test_failed_without_data_never_displaces_another_block(self):
        aborted, pending = _block(BlockState.ABORTED), _block(BlockState.PENDING)
        self.assertIs(select_schedule_block([aborted, failed_block(0.0)]), aborted)
        self.assertIs(select_schedule_block([failed_block(0.0), pending]), pending)

    def test_single_aborted_block_is_chosen(self):
        aborted = _block(BlockState.ABORTED)
        self.assertIs(select_schedule_block([aborted]), aborted)

    def test_a_pending_block_beats_an_aborted_block_in_either_order(self):
        aborted, pending = _block(BlockState.ABORTED), _block(BlockState.PENDING, start='2026-07-05T01:00:00Z')
        self.assertIs(select_schedule_block([pending, aborted]), pending)
        self.assertIs(select_schedule_block([aborted, pending]), pending)

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

    def test_a_pending_block_beats_an_in_progress_block_in_either_order(self):
        """A-24: an IN_PROGRESS block yields to a PENDING one, as the developer's chosen code does."""
        in_progress, pending = _block(BlockState.IN_PROGRESS), _block(BlockState.PENDING, start='2026-07-02T01:00:00Z')
        self.assertIs(select_schedule_block([in_progress, pending]), pending)
        self.assertIs(select_schedule_block([pending, in_progress]), pending)

    def test_every_pair_of_block_states_follows_the_order(self):
        """Every ordered pair of the nine block kinds gives the winner the order names (WR-19, A-25)."""
        # kind -> (builder, rank). Rank 0: the first of equal rank wins; rank 1 (PENDING) and rank 2 (the
        # started tier) -- the last of equal rank wins; rank None never counts.
        kinds = {
            'COMPLETED': (lambda day: _block('COMPLETED', start=f'2026-07-{day:02d}T01:00:00Z'), 0),
            'PENDING': (lambda day: _block('PENDING', start=f'2026-07-{day:02d}T01:00:00Z'), 1),
            'IN_PROGRESS': (lambda day: _block('IN_PROGRESS', start=f'2026-07-{day:02d}T01:00:00Z'), 2),
            'ABORTED': (lambda day: _block('ABORTED', start=f'2026-07-{day:02d}T01:00:00Z'), 2),
            'FAILED with data': (lambda day: failed_block(18060.0, start=f'2026-07-{day:02d}T01:00:00Z'), 2),
            'FAILED without data': (lambda day: failed_block(0.0, start=f'2026-07-{day:02d}T01:00:00Z'), None),
            'NOT_ATTEMPTED': (lambda day: _block('NOT_ATTEMPTED', start=f'2026-07-{day:02d}T01:00:00Z'), None),
            'CANCELED': (lambda day: _block('CANCELED', start=f'2026-07-{day:02d}T01:00:00Z'), None),
            'unknown state': (lambda day: _block('SOMETHING_NEW', start=f'2026-07-{day:02d}T01:00:00Z'), None),
        }
        for first_name, (first_build, first_rank) in kinds.items():
            for second_name, (second_build, second_rank) in kinds.items():
                with self.subTest(first=first_name, second=second_name):
                    first, second = first_build(1), second_build(2)
                    counting = [
                        (rank, block)
                        for rank, block in ((first_rank, first), (second_rank, second))
                        if rank is not None
                    ]
                    if not counting:
                        expected = None
                    else:
                        best = min(rank for rank, _ in counting)
                        winners = [block for rank, block in counting if rank == best]
                        expected = winners[0] if best == 0 else winners[-1]
                    self.assertIs(select_schedule_block([first, second]), expected)

    def test_every_pair_of_block_states_follows_the_order_once_the_request_is_finished(self):
        """Every ordered pair of the nine block kinds, for a finished request (WR-20, developer decision 2026-10-05).

        Once the request is finished the started tier comes before a leftover PENDING block. For a request that
        can still run, the order is checked against an expectation computed independently from the same ranks with
        the PENDING and started tiers swapped (review IN-33 (3)).
        """
        # kind -> (builder, rank). Rank 0 (COMPLETED): the first of equal rank wins. Rank 1 (the started tier) and
        # rank 2 (PENDING): the last of equal rank wins. Rank None never counts.
        kinds = {
            'COMPLETED': (lambda day: _block('COMPLETED', start=f'2026-07-{day:02d}T01:00:00Z'), 0),
            'PENDING': (lambda day: _block('PENDING', start=f'2026-07-{day:02d}T01:00:00Z'), 2),
            'IN_PROGRESS': (lambda day: _block('IN_PROGRESS', start=f'2026-07-{day:02d}T01:00:00Z'), 1),
            'ABORTED': (lambda day: _block('ABORTED', start=f'2026-07-{day:02d}T01:00:00Z'), 1),
            'FAILED with data': (lambda day: failed_block(18060.0, start=f'2026-07-{day:02d}T01:00:00Z'), 1),
            'FAILED without data': (lambda day: failed_block(0.0, start=f'2026-07-{day:02d}T01:00:00Z'), None),
            'NOT_ATTEMPTED': (lambda day: _block('NOT_ATTEMPTED', start=f'2026-07-{day:02d}T01:00:00Z'), None),
            'CANCELED': (lambda day: _block('CANCELED', start=f'2026-07-{day:02d}T01:00:00Z'), None),
            'unknown state': (lambda day: _block('SOMETHING_NEW', start=f'2026-07-{day:02d}T01:00:00Z'), None),
        }
        for first_name, (first_build, first_rank) in kinds.items():
            for second_name, (second_build, second_rank) in kinds.items():
                with self.subTest(first=first_name, second=second_name):
                    first, second = first_build(1), second_build(2)

                    def expected_for(ranks, first=first, second=second):
                        counting = [
                            (rank, block)
                            for rank, block in zip(ranks, (first, second), strict=True)
                            if rank is not None
                        ]
                        if not counting:
                            return None
                        best = min(rank for rank, _ in counting)
                        winners = [block for rank, block in counting if rank == best]
                        return winners[0] if best == 0 else winners[-1]

                    # While the request can still run, PENDING (1) comes before the started tier (2).
                    swap = {0: 0, 1: 2, 2: 1, None: None}
                    self.assertIs(
                        select_schedule_block([first, second], request_finished=True),
                        expected_for((first_rank, second_rank)),
                    )
                    self.assertIs(
                        select_schedule_block([first, second], request_finished=False),
                        expected_for((swap[first_rank], swap[second_rank])),
                    )

    def test_a_block_that_took_data_beats_a_leftover_pending_block_once_the_request_is_finished(self):
        """WR-20: a finished request runs no further block, so a block that took data beats a pending one."""
        pending = _block(BlockState.PENDING, start='2026-07-05T01:00:00Z')
        started = {
            'FAILED with data': failed_block(18060.0),
            'ABORTED': _block(BlockState.ABORTED),
            'IN_PROGRESS': _block(BlockState.IN_PROGRESS),
        }
        for label, block in started.items():
            with self.subTest(label, order='started first'):
                self.assertIs(select_schedule_block([block, pending], request_finished=True), block)
            with self.subTest(label, order='pending first'):
                self.assertIs(select_schedule_block([pending, block], request_finished=True), block)
            with self.subTest(label, order='request can still run'):
                self.assertIs(select_schedule_block([block, pending]), pending)

    def test_a_finished_request_with_only_a_pending_block_keeps_that_block(self):
        """A-33: with no block that took data, a leftover PENDING block still gives its times, as every earlier
        rule did. This is flagged for the developer, not decided silently."""
        pending = _block(BlockState.PENDING, start='2026-07-05T01:00:00Z')
        self.assertIs(select_schedule_block([pending], request_finished=True), pending)
        self.assertIs(select_schedule_block([failed_block(0.0), pending], request_finished=True), pending)
        self.assertIs(select_schedule_block([pending, _block('NOT_ATTEMPTED')], request_finished=True), pending)
        self.assertIs(
            select_schedule_block([_block('CANCELED'), pending, _block('SOMETHING_NEW')], request_finished=True),
            pending,
        )

    def test_finished_request_sequences(self):
        """A request that finished after being placed again: which block the record carries."""
        failed_day1 = failed_block(18060.0, start='2026-07-01T01:00:00Z')
        failed_day3 = failed_block(3600.0, start='2026-07-03T01:00:00Z')
        no_data_day3 = failed_block(0.0, start='2026-07-03T01:00:00Z')
        canceled = _block('CANCELED', start='2026-07-02T01:00:00Z')
        pending_day4 = _block('PENDING', start='2026-07-04T01:00:00Z')
        pending_day1 = _block('PENDING', start='2026-07-01T01:00:00Z')
        completed_day2 = _block('COMPLETED', start='2026-07-02T01:00:00Z')
        finished = {'request_finished': True}
        self.assertIs(select_schedule_block([failed_day1, canceled, pending_day4], **finished), failed_day1)
        self.assertIs(select_schedule_block([failed_day1, failed_day3, pending_day4], **finished), failed_day3)
        self.assertIs(select_schedule_block([failed_day1, no_data_day3, pending_day4], **finished), failed_day1)
        self.assertIs(select_schedule_block([pending_day1, completed_day2], **finished), completed_day2)

    def test_rescheduled_request_sequences(self):
        """A request placed again after a block stopped early: which block the record carries."""
        failed_day1 = failed_block(18060.0, start='2026-07-01T01:00:00Z')
        failed_day3 = failed_block(3600.0, start='2026-07-03T01:00:00Z')
        no_data_day3 = failed_block(0.0, start='2026-07-03T01:00:00Z')
        canceled = _block('CANCELED', start='2026-07-02T01:00:00Z')
        pending = _block('PENDING', start='2026-07-04T01:00:00Z')
        self.assertIs(select_schedule_block([failed_day1, canceled, pending]), pending)
        self.assertIs(select_schedule_block([failed_day1, failed_day3]), failed_day3)
        self.assertIs(select_schedule_block([failed_day1, no_data_day3]), failed_day1)
        self.assertIs(select_schedule_block([failed_day1, canceled]), failed_day1)

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
    def test_status_carries_the_failed_block_times_when_it_took_data(self, mock_make_request):
        block = REAL_FAILED_BLOCKS['4253588']
        mock_make_request.side_effect = portal_side_effect({'123': 'WINDOW_EXPIRED'}, {'123': [block]})

        status = FomoLCOFacility().get_observation_status('123')

        self.assertEqual(
            status,
            {'state': 'WINDOW_EXPIRED', 'scheduled_start': block['start'], 'scheduled_end': block['end']},
        )

    @patch('solsys_code.observation_blocks.make_request')
    def test_status_has_no_times_for_a_failed_block_that_took_no_data(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect({'123': 'WINDOW_EXPIRED'}, {'123': [failed_block(0.0)]})

        status = FomoLCOFacility().get_observation_status('123')

        self.assertEqual(status, {'state': 'WINDOW_EXPIRED', 'scheduled_start': None, 'scheduled_end': None})

    @patch('solsys_code.observation_blocks.make_request')
    def test_update_observation_status_saves_the_failed_block_times(self, mock_make_request):
        block = REAL_FAILED_BLOCKS['4253588']
        mock_make_request.side_effect = portal_side_effect({'123': 'WINDOW_EXPIRED'}, {'123': [block]})
        record = ObservationRecord.objects.create(
            target=self.target, facility='LCO', observation_id='123', status='WINDOW_EXPIRED', parameters={}
        )

        FomoLCOFacility().update_observation_status('123')

        record.refresh_from_db()
        self.assertEqual(record.scheduled_start, datetime(2026, 7, 12, 8, 55, 50, tzinfo=timezone.utc))
        self.assertEqual(record.scheduled_end, datetime(2026, 7, 12, 15, 0, 31, tzinfo=timezone.utc))

    @patch('solsys_code.observation_blocks.make_request')
    def test_update_observation_status_stores_the_placed_block_over_a_failed_one(self, mock_make_request):
        """WR-19: the stored times are the placed PENDING block's, not the earlier failed block's."""
        placed = {'state': 'PENDING', 'start': '2026-07-17T09:00:00Z', 'end': '2026-07-17T14:00:00Z'}
        mock_make_request.side_effect = portal_side_effect(
            {'123': 'PENDING'}, {'123': [REAL_FAILED_BLOCKS['4253588'], placed]}
        )
        record = ObservationRecord.objects.create(
            target=self.target, facility='LCO', observation_id='123', status='PENDING', parameters={}
        )

        FomoLCOFacility().update_observation_status('123')

        record.refresh_from_db()
        self.assertEqual(record.scheduled_start, datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc))
        self.assertEqual(record.scheduled_end, datetime(2026, 7, 17, 14, 0, tzinfo=timezone.utc))

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

    @patch('solsys_code.observation_blocks.make_request')
    def test_a_dict_reply_for_the_block_list_raises_and_names_only_the_request_and_type(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect(
            {'123': 'WINDOW_EXPIRED'}, {'123': {'detail': 'secret-body-text'}}
        )

        with self.assertRaises(ValueError) as cm:
            FomoLCOFacility().get_observation_status('123')

        self.assertEqual(type(cm.exception).__name__, 'UnexpectedBlockPayloadError')
        self.assertIn('123', str(cm.exception))
        self.assertIn('dict', str(cm.exception))
        self.assertNotIn('secret-body-text', str(cm.exception))
        self.assertEqual(mock_make_request.call_count, 2)

    @patch('solsys_code.observation_blocks.make_request')
    def test_a_paginated_envelope_for_the_block_list_raises(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect(
            {'123': 'WINDOW_EXPIRED'}, {'123': {'count': 1, 'results': [_block('ABORTED')]}}
        )

        with self.assertRaises(ValueError) as cm:
            FomoLCOFacility().get_observation_status('123')

        self.assertEqual(type(cm.exception).__name__, 'UnexpectedBlockPayloadError')

    def test_the_facilitys_terminal_states_are_finished(self):
        for facility in (FomoLCOFacility(), FomoSOARFacility()):
            for state in ('COMPLETED', 'WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED'):
                with self.subTest(facility=facility.name, state=state):
                    self.assertTrue(is_request_finished(state, facility))

    def test_the_terminal_states_are_a_list_so_an_odd_state_never_raises(self):
        """IN-34: is_request_finished() relies on get_terminal_observing_states() returning a list, as TOM
        Toolkit's OCS facilities do. A list is searched with ==, so a dict, a list or a set state gives False
        without raising; a set or frozenset of states would raise TypeError for an unhashable one."""
        for facility in (FomoLCOFacility(), FomoSOARFacility()):
            with self.subTest(facility=facility.name):
                self.assertIs(type(facility.get_terminal_observing_states()), list)
                for odd_state in ({'odd': 1}, ['COMPLETED'], {'COMPLETED'}):
                    self.assertFalse(is_request_finished(odd_state, facility))

    def test_pending_and_unknown_states_can_still_run(self):
        for facility in (FomoLCOFacility(), FomoSOARFacility()):
            for state in ('PENDING', '', None, 'SOMETHING_NEW', {'odd': 1}):
                with self.subTest(facility=facility.name, state=state):
                    self.assertFalse(is_request_finished(state, facility))

    @patch('solsys_code.observation_blocks.make_request')
    def test_status_on_a_finished_request_reads_the_block_that_took_data_over_a_leftover_pending_one(
        self, mock_make_request
    ):
        """WR-20: the portal still lists the placed block as PENDING when the request finishes."""
        failed = REAL_FAILED_BLOCKS['4253588']
        pending = {'state': 'PENDING', 'start': '2026-07-17T09:00:00Z', 'end': '2026-07-17T14:00:00Z'}
        for state in ('COMPLETED', 'WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED'):
            with self.subTest(state):
                mock_make_request.side_effect = portal_side_effect({'123': state}, {'123': [failed, pending]})

                status = FomoLCOFacility().get_observation_status('123')

                self.assertEqual(
                    status, {'state': state, 'scheduled_start': failed['start'], 'scheduled_end': failed['end']}
                )
        mock_make_request.side_effect = portal_side_effect({'123': 'PENDING'}, {'123': [failed, pending]})
        self.assertEqual(
            FomoLCOFacility().get_observation_status('123'),
            {'state': 'PENDING', 'scheduled_start': pending['start'], 'scheduled_end': pending['end']},
        )

    @patch('solsys_code.observation_blocks.make_request')
    def test_status_on_a_completed_request_still_reads_its_completed_block(self, mock_make_request):
        """Q2: a COMPLETED request is finished, and its COMPLETED block still wins first, in either order."""
        pending = {'state': 'PENDING', 'start': '2026-07-17T09:00:00Z', 'end': '2026-07-17T14:00:00Z'}
        completed = {'state': 'COMPLETED', 'start': '2026-07-19T09:00:00Z', 'end': '2026-07-19T14:00:00Z'}
        for label, blocks in (
            ('pending then completed', [pending, completed]),
            ('failed then completed', [REAL_FAILED_BLOCKS['4253588'], completed]),
            ('completed then pending', [completed, pending]),
            ('completed then failed', [completed, REAL_FAILED_BLOCKS['4253588']]),
        ):
            with self.subTest(label):
                mock_make_request.side_effect = portal_side_effect({'123': 'COMPLETED'}, {'123': blocks})

                status = FomoLCOFacility().get_observation_status('123')

                self.assertEqual(status['scheduled_start'], completed['start'])
                self.assertEqual(status['scheduled_end'], completed['end'])

    def _placed_pending_record(self):
        return ObservationRecord.objects.create(
            target=self.target,
            facility='LCO',
            observation_id='123',
            status='PENDING',
            parameters={},
            scheduled_start=datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 7, 1, 1, 40, tzinfo=timezone.utc),
        )

    def _assert_record_unchanged(self, record):
        record.refresh_from_db()
        self.assertEqual(record.status, 'PENDING')
        self.assertEqual(record.scheduled_start, datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc))
        self.assertEqual(record.scheduled_end, datetime(2026, 7, 1, 1, 40, tzinfo=timezone.utc))

    @patch('solsys_code.observation_blocks.make_request')
    def test_update_observation_status_writes_nothing_when_the_block_list_is_not_a_list(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect({'123': 'WINDOW_EXPIRED'}, {'123': {'detail': 'Not found.'}})
        record = self._placed_pending_record()

        with self.assertRaises(ValueError):
            FomoLCOFacility().update_observation_status('123')

        self._assert_record_unchanged(record)

    @patch('solsys_code.observation_blocks.make_request')
    def test_update_all_observation_statuses_reports_the_record_and_keeps_its_times(self, mock_make_request):
        mock_make_request.side_effect = portal_side_effect(
            {'123': 'WINDOW_EXPIRED'}, {'123': {'count': 1, 'results': [_block('ABORTED')]}}
        )
        record = self._placed_pending_record()

        failures = FomoLCOFacility().update_all_observation_statuses()

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], '123')
        self._assert_record_unchanged(record)
