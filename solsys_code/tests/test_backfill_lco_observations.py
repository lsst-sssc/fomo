import io
from unittest.mock import MagicMock, patch

from django.core.management import CommandError, call_command
from django.test import TestCase
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import Target
from tom_targets.tests.factories import NonSiderealTargetFactory

# A complete, correctly-scoped ORBITAL_ELEMENTS wire-key payload (D-E), used as the default
# for every fixture request unless a test deliberately builds an incomplete one.
_DEFAULT_ELEMENTS = {
    'scheme': 'MPC_MINOR_PLANET',
    'orbinc': 3.4,
    'longascnode': 73.5,
    'argofperih': 319.2,
    'meandist': 2.5,
    'meananom': 45.6,
    'eccentricity': 0.2,
    'epochofel': 58600.0,
}


def _configuration(
    instrument_type='1M0-SCICAM-SINISTRO', target_name='Didymos', target_type='ORBITAL_ELEMENTS', elements=None
):
    target = {'name': target_name, 'type': target_type}
    if target_type == 'ORBITAL_ELEMENTS':
        target.update(_DEFAULT_ELEMENTS if elements is None else elements)
    return {
        'type': 'EXPOSE',
        'instrument_type': instrument_type,
        'instrument_configs': [{'exposure_time': 30.0, 'exposure_count': 1}],
        'target': target,
    }


def _request(
    request_id,
    target_name='Didymos',
    state='COMPLETED',
    target_type='ORBITAL_ELEMENTS',
    elements=None,
    observations=None,
    windows=None,
):
    request = {
        'id': request_id,
        'state': state,
        'windows': windows if windows is not None else [{'start': '2026-07-01T00:00:00', 'end': '2026-07-02T00:00:00'}],
        'configurations': [
            _configuration(target_name=target_name, target_type=target_type, elements=elements),
        ],
    }
    if observations is not None:
        request['observations'] = observations
    return request


def _request_group(group_id, name, proposal='LCO2026A-003', requests=None, created='2026-07-01T00:00:00Z'):
    return {
        'id': group_id,
        'name': name,
        'proposal': proposal,
        'state': 'COMPLETED',
        'created': created,
        'requests': requests if requests is not None else [_request(group_id * 10)],
    }


def _page_response(results, next_url=None):
    response = MagicMock()
    response.json.return_value = {'count': len(results), 'next': next_url, 'previous': None, 'results': results}
    return response


class TestBackfillLcoObservations(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    def setUp(self):
        # The command constructs its own LCOFacility() instance, so the D-B fallback lookup
        # must be patched at the class level, exactly as the sibling test patches
        # update_observation_status. A sensible default return value keeps every test that
        # doesn't care about block-derived times passing without extra setup; tests that do
        # care override .return_value/.side_effect explicitly.
        patcher = patch('tom_observations.facilities.lco.LCOFacility.get_observation_status')
        self.mock_get_observation_status = patcher.start()
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        self.addCleanup(patcher.stop)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_creates_record_for_matching_group_and_request(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        stdout, stderr = io.StringIO(), io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout, stderr=stderr)

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.target, self.existing_target)
        self.assertEqual(record.status, 'COMPLETED')
        self.assertEqual(record.parameters['proposal'], 'LCO2026A-003')
        self.assertEqual(record.parameters['instrument_type'], '1M0-SCICAM-SINISTRO')
        self.assertEqual(record.parameters['start'], '2026-07-01T00:00:00')
        # scheduled_start/end came from the mocked D-B fallback, since this fixture carries
        # no embedded 'observations' block.
        self.assertIsNotNone(record.scheduled_start)
        self.assertIsNotNone(record.scheduled_end)
        self.assertIn('requestgroups seen: 1', stdout.getvalue())
        self.assertIn('created: 1', stdout.getvalue())
        self.assertEqual(stderr.getvalue(), '')

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_uses_embedded_observation_block_when_present(self, mock_make_request):
        # D-B: an embedded 'observations' block list on the request payload is read
        # directly and takes priority over the live facility fallback -- no fallback call
        # should be made at all.
        observations = [{'state': 'COMPLETED', 'start': '2026-07-01T00:05:00', 'end': '2026-07-01T00:15:00'}]
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, observations=observations)])]
        )

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T00:05:00+00:00')
        self.assertEqual(record.scheduled_end.isoformat(), '2026-07-01T00:15:00+00:00')
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_idempotent_rerun_updates_status_and_schedule_in_place(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='PENDING')])]
        )
        self.mock_get_observation_status.return_value = {
            'state': 'PENDING',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(ObservationRecord.objects.filter(facility='LCO', observation_id='10').count(), 1)

        # Second pass: the request's own state changed, and the observed block moved.
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='COMPLETED')])]
        )
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T01:10:00+00:00',
            'scheduled_end': '2026-07-01T01:20:00+00:00',
        }
        stdout2 = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout2, stderr=io.StringIO())

        self.assertEqual(ObservationRecord.objects.filter(facility='LCO', observation_id='10').count(), 1)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.status, 'COMPLETED')
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T01:10:00+00:00')
        self.assertEqual(record.scheduled_end.isoformat(), '2026-07-01T01:20:00+00:00')
        self.assertIn('updated: 1', stdout2.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_third_pass_with_no_changes_reports_unchanged(self, mock_make_request):
        page = _page_response([_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='COMPLETED')])])
        mock_make_request.return_value = page

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        stdout2 = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout2, stderr=io.StringIO())

        self.assertEqual(ObservationRecord.objects.filter(facility='LCO', observation_id='10').count(), 1)
        self.assertIn('unchanged: 1', stdout2.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_missing_target_creates_non_sidereal_target_from_orbital_elements(self, mock_make_request):
        new_target_elements = dict(_DEFAULT_ELEMENTS, orbinc=12.5, meandist=1.9)
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'New Object 2026 - ELP',
                    requests=[_request(10, target_name='2026 AB1', elements=new_target_elements)],
                )
            ]
        )

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        new_target = Target.objects.get(name='2026 AB1')
        self.assertEqual(new_target.type, Target.NON_SIDEREAL)
        self.assertEqual(new_target.inclination, 12.5)
        self.assertEqual(new_target.semimajor_axis, 1.9)
        self.assertEqual(new_target.scheme, 'MPC_MINOR_PLANET')
        self.assertFalse(Target.objects.filter(type=Target.SIDEREAL).exists())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_existing_target_reused_by_exact_name(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(Target.objects.filter(name='Didymos').count(), 1)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.target, self.existing_target)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_existing_target_reused_by_alias(self, mock_make_request):
        self.existing_target.aliases.create(name='65803 Didymos')
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, target_name='65803 Didymos')])]
        )

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(Target.objects.filter(name='65803 Didymos').count(), 0)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.target, self.existing_target)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_unmappable_request_is_skipped_and_reported(self, mock_make_request):
        incomplete_elements = {'scheme': 'MPC_MINOR_PLANET', 'orbinc': 3.4}  # missing meandist/meananom/etc.
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Unmappable 2026 - ELP',
                    requests=[_request(10, target_name='Unmappable Object', elements=incomplete_elements)],
                )
            ]
        )

        stderr = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=stderr)

        self.assertFalse(ObservationRecord.objects.exists())
        self.assertFalse(Target.objects.filter(name='Unmappable Object').exists())
        self.assertFalse(Target.objects.filter(type=Target.SIDEREAL).exists())
        self.assertIn('10', stderr.getvalue())
        self.assertIn('Skipping request', stderr.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_multi_request_group_creates_one_reusable_observation_group(self, mock_make_request):
        page = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - Multi',
                    requests=[_request(10), _request(11)],
                )
            ]
        )
        mock_make_request.return_value = page

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(ObservationGroup.objects.count(), 1)
        group = ObservationGroup.objects.get()
        self.assertEqual(group.observation_records.count(), 2)

        # Second run over the same (now unchanged) data reuses the same group -- no second one.
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(ObservationGroup.objects.count(), 1)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_single_request_group_creates_no_observation_group(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertFalse(ObservationGroup.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_writes_nothing_but_reports_summary(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - Multi', requests=[_request(10), _request(11)])]
        )

        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        self.assertFalse(ObservationRecord.objects.exists())
        # setUpTestData already created 'Didymos'; dry-run must create no *new* target.
        self.assertEqual(Target.objects.count(), 1)
        self.assertFalse(ObservationGroup.objects.exists())
        self.assertIn('requestgroups seen: 1', stdout.getvalue())
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_created_date_filter_excludes_group_outside_window_even_if_portal_returns_it(self, mock_make_request):
        # The mock returns this group regardless of the created_after/created_before query
        # params sent, proving the client-side re-check (not the query param) is what
        # actually excludes it.
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Old Run 2026 - ELP', created='2020-01-01T00:00:00Z')]
        )

        call_command(
            'backfill_lco_observations',
            '--proposal=LCO2026A-003',
            '--created-after=2026-01-01T00:00:00',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertFalse(ObservationRecord.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_created_date_filter_includes_group_inside_window(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'New Run 2026 - ELP', created='2026-06-15T00:00:00Z')]
        )

        call_command(
            'backfill_lco_observations',
            '--proposal=LCO2026A-003',
            '--created-after=2026-01-01T00:00:00',
            '--created-before=2026-12-31T00:00:00',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertTrue(ObservationRecord.objects.filter(facility='LCO', observation_id='10').exists())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_follows_pagination(self, mock_make_request):
        mock_make_request.side_effect = [
            _page_response([_request_group(1, 'Didymos 2026 - ELP')], next_url='https://observe.lco.global/next'),
            _page_response([_request_group(2, 'Didymos 2026 - LSC', requests=[_request(20)])]),
        ]

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(mock_make_request.call_count, 2)
        self.assertEqual(ObservationRecord.objects.filter(facility='LCO').count(), 2)

    def test_unknown_username_raises_command_error(self):
        with self.assertRaises(CommandError):
            call_command(
                'backfill_lco_observations',
                '--proposal=LCO2026A-003',
                '--username=nonexistent-user',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )

    def test_invalid_created_after_raises_command_error(self):
        with self.assertRaises(CommandError):
            call_command(
                'backfill_lco_observations',
                '--proposal=LCO2026A-003',
                '--created-after=not-a-date',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
