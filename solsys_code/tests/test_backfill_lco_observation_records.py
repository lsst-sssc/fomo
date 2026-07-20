from unittest.mock import MagicMock, patch

from django.core.management import CommandError, call_command
from django.test import TestCase
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory


def _configuration(instrument_type='1M0-SCICAM-SINISTRO', target_name='Didymos'):
    return {
        'type': 'EXPOSE',
        'instrument_type': instrument_type,
        'instrument_configs': [{'exposure_time': 30.0, 'exposure_count': 1}],
        'target': {'name': target_name, 'type': 'ORBITAL_ELEMENTS'},
    }


def _request(request_id, target_name='Didymos', state='COMPLETED'):
    return {
        'id': request_id,
        'state': state,
        'windows': [{'start': '2026-07-01T00:00:00', 'end': '2026-07-02T00:00:00'}],
        'configurations': [_configuration(target_name=target_name)],
    }


def _request_group(group_id, name, proposal='LCO2026A-003', requests=None):
    return {
        'id': group_id,
        'name': name,
        'proposal': proposal,
        'state': 'COMPLETED',
        'requests': requests if requests is not None else [_request(group_id * 10)],
    }


def _page_response(results, next_url=None):
    response = MagicMock()
    response.json.return_value = {'count': len(results), 'next': next_url, 'previous': None, 'results': results}
    return response


class TestBackfillLcoObservationRecords(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.target = NonSiderealTargetFactory.create(name='Didymos')
        cls.campaign = TargetList.objects.create(name='Didymos 2026 Campaign')
        cls.campaign.targets.add(cls.target)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_creates_record_for_matching_group_and_target(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.target, self.target)
        self.assertEqual(record.status, 'COMPLETED')
        self.assertEqual(record.parameters['proposal'], 'LCO2026A-003')
        self.assertEqual(record.parameters['instrument_type'], '1M0-SCICAM-SINISTRO')
        self.assertEqual(record.parameters['start'], '2026-07-01T00:00:00')

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_name_prefix_is_rechecked_client_side(self, mock_make_request):
        # Server-side 'name' filter is icontains, so a group containing but not
        # starting with the prefix could come back from the API -- must be excluded.
        mock_make_request.return_value = _page_response([_request_group(1, 'Not a Didymos run')])

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        self.assertFalse(ObservationRecord.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_skips_request_with_existing_observation_record(self, mock_make_request):
        ObservationRecord.objects.create(
            target=self.target, facility='LCO', observation_id='10', status='COMPLETED', parameters={}
        )
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        self.assertEqual(ObservationRecord.objects.filter(facility='LCO', observation_id='10').count(), 1)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_skips_request_whose_target_is_not_a_campaign_member(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, target_name='Some Other Object')])]
        )

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        self.assertFalse(ObservationRecord.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_dry_run_creates_nothing(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--dry-run',
        )

        self.assertFalse(ObservationRecord.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_follows_pagination(self, mock_make_request):
        mock_make_request.side_effect = [
            _page_response([_request_group(1, 'Didymos 2026 - ELP')], next_url='https://observe.lco.global/next'),
            _page_response([_request_group(2, 'Didymos 2026 - LSC', requests=[_request(20)])]),
        ]

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        self.assertEqual(mock_make_request.call_count, 2)
        self.assertEqual(ObservationRecord.objects.filter(facility='LCO').count(), 2)

    def test_unknown_campaign_raises_command_error(self):
        with self.assertRaises(CommandError):
            call_command(
                'backfill_lco_observation_records',
                '--proposal=LCO2026A-003',
                '--name-prefix=Didymos',
                '--campaign=Not A Real Campaign',
            )

    def test_campaign_with_no_targets_raises_command_error(self):
        TargetList.objects.create(name='Empty Campaign')
        with self.assertRaises(CommandError):
            call_command(
                'backfill_lco_observation_records',
                '--proposal=LCO2026A-003',
                '--name-prefix=Didymos',
                '--campaign=Empty Campaign',
            )

    def test_unknown_username_raises_command_error(self):
        with self.assertRaises(CommandError):
            call_command(
                'backfill_lco_observation_records',
                '--proposal=LCO2026A-003',
                '--name-prefix=Didymos',
                '--campaign=Didymos 2026 Campaign',
                '--username=nonexistent-user',
            )
