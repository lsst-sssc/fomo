import io
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import requests
from django.core.management import CommandError, call_command
from django.test import TestCase
from tom_observations.models import ObservationRecord
from tom_targets.models import Target, TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.allocation_projector import allocation_events
from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.campaign_utils import create_system_link
from solsys_code.models import CampaignRun, CampaignRunObservation
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import observing_night
from solsys_code.tests.test_observation_blocks import REAL_FAILED_BLOCKS, failed_block, portal_side_effect


def _configuration(
    instrument_type='1M0-SCICAM-SINISTRO',
    target_name='Didymos',
    target_type='ORBITAL_ELEMENTS',
    ra=None,
    dec=None,
    epoch=None,
    pm_ra=None,
    pm_dec=None,
    parallax=None,
):
    target = {'name': target_name, 'type': target_type}
    if ra is not None:
        target['ra'] = ra
    if dec is not None:
        target['dec'] = dec
    if epoch is not None:
        target['epoch'] = epoch
    if pm_ra is not None:
        target['proper_motion_ra'] = pm_ra
    if pm_dec is not None:
        target['proper_motion_dec'] = pm_dec
    if parallax is not None:
        target['parallax'] = parallax
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
    ra=None,
    dec=None,
    epoch=None,
    pm_ra=None,
    pm_dec=None,
    parallax=None,
):
    return {
        'id': request_id,
        'state': state,
        'windows': [{'start': '2026-07-01T00:00:00', 'end': '2026-07-02T00:00:00'}],
        'configurations': [
            _configuration(
                target_name=target_name,
                target_type=target_type,
                ra=ra,
                dec=dec,
                epoch=epoch,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                parallax=parallax,
            )
        ],
    }


def _field_request(
    request_id, target_name, ra=170.1, dec=-24.3, state='COMPLETED', epoch=None, pm_ra=None, pm_dec=None, parallax=None
):
    """Build a request for a fixed-sky field target, carrying ra/dec like a real ICRS pointing."""
    return _request(
        request_id,
        target_name=target_name,
        state=state,
        target_type='ICRS',
        ra=ra,
        dec=dec,
        epoch=epoch,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        parallax=parallax,
    )


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
    FIELD_NAME = 'Didymos COJ 2026 Field #14'

    @classmethod
    def setUpTestData(cls):
        cls.target = NonSiderealTargetFactory.create(name='Didymos')
        cls.campaign = TargetList.objects.create(name='Didymos 2026 Campaign')
        cls.campaign.targets.add(cls.target)

    def setUp(self):
        # Prevent every non-dry-run test from making a real live API call via the new
        # post-create facility.update_observation_status(observation_id) call (260722-ux0)
        # -- patched at the class level since the command constructs its own LCOFacility()
        # instance. Tests that care about call count/args/side effects override this mock.
        patcher = patch('tom_observations.facilities.lco.LCOFacility.update_observation_status')
        self.mock_update_observation_status = patcher.start()
        self.addCleanup(patcher.stop)

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

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_flag_off_unmatched_field_target_still_skipped(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - COJ', requests=[_field_request(10, self.FIELD_NAME)])]
        )

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        self.assertFalse(ObservationRecord.objects.exists())
        self.assertFalse(Target.objects.filter(name=self.FIELD_NAME).exists())

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_flag_on_creates_new_field_target(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - COJ', requests=[_field_request(10, self.FIELD_NAME)])]
        )

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--create-missing-targets',
        )

        field_target = Target.objects.get(name=self.FIELD_NAME)
        self.assertEqual(field_target.type, Target.SIDEREAL)
        self.assertEqual(field_target.ra, 170.1)
        self.assertEqual(field_target.dec, -24.3)
        self.assertTrue(self.campaign.targets.filter(name=self.FIELD_NAME).exists())
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.target, field_target)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_flag_on_reuses_existing_field_target(self, mock_make_request):
        existing_field_target = Target.objects.create(name=self.FIELD_NAME, type=Target.SIDEREAL, ra=170.1, dec=-24.3)
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - COJ', requests=[_field_request(10, self.FIELD_NAME)])]
        )

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--create-missing-targets',
        )

        self.assertEqual(Target.objects.filter(name=self.FIELD_NAME).count(), 1)
        self.assertTrue(self.campaign.targets.filter(name=self.FIELD_NAME).exists())
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.target, existing_field_target)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_flag_on_dry_run_creates_nothing(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - COJ', requests=[_field_request(10, self.FIELD_NAME)])]
        )
        campaign_target_count_before = self.campaign.targets.count()

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--create-missing-targets',
            '--dry-run',
        )

        self.assertFalse(Target.objects.filter(name=self.FIELD_NAME).exists())
        self.assertEqual(self.campaign.targets.count(), campaign_target_count_before)
        self.assertFalse(ObservationRecord.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_flag_on_creates_new_field_target_with_epoch_pm_parallax(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - COJ',
                    requests=[
                        _field_request(10, self.FIELD_NAME, epoch=2451545.0, pm_ra=12.3, pm_dec=-45.6, parallax=7.89)
                    ],
                )
            ]
        )

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--create-missing-targets',
        )

        field_target = Target.objects.get(name=self.FIELD_NAME)
        self.assertEqual(field_target.epoch, 2451545.0)
        self.assertEqual(field_target.pm_ra, 12.3)
        self.assertEqual(field_target.pm_dec, -45.6)
        self.assertEqual(field_target.parallax, 7.89)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_flag_on_creates_new_field_target_without_epoch_pm_parallax(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - COJ', requests=[_field_request(10, self.FIELD_NAME)])]
        )

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--create-missing-targets',
        )

        field_target = Target.objects.get(name=self.FIELD_NAME)
        self.assertIsNone(field_target.epoch)
        self.assertIsNone(field_target.pm_ra)
        self.assertIsNone(field_target.pm_dec)
        self.assertIsNone(field_target.parallax)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_flag_on_reuse_never_overwrites_epoch_pm_parallax(self, mock_make_request):
        existing_field_target = Target.objects.create(name=self.FIELD_NAME, type=Target.SIDEREAL, ra=170.1, dec=-24.3)
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - COJ',
                    requests=[
                        _field_request(10, self.FIELD_NAME, epoch=2451545.0, pm_ra=12.3, pm_dec=-45.6, parallax=7.89)
                    ],
                )
            ]
        )

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--create-missing-targets',
        )

        existing_field_target.refresh_from_db()
        self.assertIsNone(existing_field_target.epoch)
        self.assertIsNone(existing_field_target.pm_ra)
        self.assertIsNone(existing_field_target.pm_dec)
        self.assertIsNone(existing_field_target.parallax)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_created_record_triggers_status_refresh(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        def _refresh_status(observation_id):
            record = ObservationRecord.objects.get(facility='LCO', observation_id=observation_id)
            record.scheduled_start = '2026-07-01T00:10:00+00:00'
            record.scheduled_end = '2026-07-01T00:20:00+00:00'
            record.save()

        self.mock_update_observation_status.side_effect = _refresh_status

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        self.mock_update_observation_status.assert_called_once_with('10')
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertIsNotNone(record.scheduled_start)
        self.assertIsNotNone(record.scheduled_end)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_dry_run_makes_zero_status_refresh_calls(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
            '--dry-run',
        )

        self.mock_update_observation_status.assert_not_called()
        self.assertFalse(ObservationRecord.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_skipped_existing_record_makes_zero_status_refresh_calls(self, mock_make_request):
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

        self.mock_update_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_status_refresh_failure_is_non_fatal_and_does_not_roll_back(self, mock_make_request):
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - ELP',
                    requests=[_request(10), _request(11)],
                )
            ]
        )
        self.mock_update_observation_status.side_effect = Exception('LCO API unavailable')

        call_command(
            'backfill_lco_observation_records',
            '--proposal=LCO2026A-003',
            '--name-prefix=Didymos',
            '--campaign=Didymos 2026 Campaign',
        )

        self.assertEqual(self.mock_update_observation_status.call_count, 2)
        self.assertTrue(ObservationRecord.objects.filter(facility='LCO', observation_id='10').exists())
        self.assertTrue(ObservationRecord.objects.filter(facility='LCO', observation_id='11').exists())


class TestBackfillSystemLinks(TestCase):
    """ALLOC-06: the Didymos backfill links records it creates and records it re-encounters."""

    FIELD_NAME = 'Didymos COJ 2026 Field #14'
    ARGS = ('--proposal=LCO2026A-003', '--name-prefix=Didymos', '--campaign=Didymos 2026 Campaign')

    @classmethod
    def setUpTestData(cls):
        cls.target = NonSiderealTargetFactory.create(name='Didymos')
        cls.field_target = NonSiderealTargetFactory.create(name=cls.FIELD_NAME)
        cls.campaign = TargetList.objects.create(name='Didymos 2026 Campaign')
        cls.campaign.targets.add(cls.target, cls.field_target)
        cls.container_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            target=cls.target,
            proposal_code='LCO2026A-003',
            telescope_class='1m0',
            source=CampaignRun.Source.LCO_QUEUE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='1m0/Sinistro',
            window_start=date(2026, 6, 25),
            window_end=date(2026, 7, 1),
        )

    def setUp(self):
        patcher = patch('tom_observations.facilities.lco.LCOFacility.update_observation_status')
        self.mock_update_observation_status = patcher.start()
        self.addCleanup(patcher.stop)

    def _existing_record(self):
        return ObservationRecord.objects.create(
            target=self.field_target,
            facility='LCO',
            observation_id='10',
            status='COMPLETED',
            parameters={
                'proposal': 'LCO2026A-003',
                'instrument_type': '1M0-SCICAM-SINISTRO',
                'start': '2026-06-28T00:00:00',
                'end': '2026-06-29T00:00:00',
            },
        )

    def _page(self):
        return _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, target_name=self.FIELD_NAME)])]
        )

    def _run_command(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        summary = call_command('backfill_lco_observation_records', *self.ARGS, *extra, stdout=stdout, stderr=stderr)
        return summary, stdout.getvalue(), stderr.getvalue()

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_existing_unlinked_record_links_by_campaign_fallback(self, mock_make_request):
        mock_make_request.return_value = self._page()
        record = self._existing_record()

        summary, stdout, _ = self._run_command()

        link = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link.run_id, self.container_run.pk)
        self.assertIsNone(link.confirmed_by)
        self.assertIn('(proposal unique within campaign + window)', stdout)
        self.assertIn('already existed: 1', summary)
        self.assertTrue(summary.endswith('system links: 1, links skipped: 0'), summary)
        self.mock_update_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_created_record_links_on_its_refreshed_placed_block(self, mock_make_request):
        mock_make_request.return_value = self._page()

        def _refresh_status(observation_id):
            record = ObservationRecord.objects.get(facility='LCO', observation_id=observation_id)
            record.scheduled_start = '2026-07-01T00:10:00+00:00'
            record.scheduled_end = '2026-07-01T00:20:00+00:00'
            record.save()

        self.mock_update_observation_status.side_effect = _refresh_status

        summary, _, _ = self._run_command()

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        # The request window (2026-07-01..07-02) is not inside the run's window; the placed block is.
        link = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link.run_id, self.container_run.pk)
        self.assertTrue(summary.endswith('system links: 1, links skipped: 0'), summary)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_rerun_is_idempotent(self, mock_make_request):
        mock_make_request.return_value = self._page()
        record = self._existing_record()
        self._run_command()
        link = CampaignRunObservation.objects.get(observation_record=record)

        summary, _, _ = self._run_command()

        self.assertEqual(CampaignRunObservation.objects.filter(observation_record=record).count(), 1)
        link_again = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link_again.confirmed_at, link.confirmed_at)
        self.assertTrue(summary.endswith('system links: 0, links skipped: 0'), summary)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_dry_run_reports_would_link_and_writes_nothing(self, mock_make_request):
        mock_make_request.return_value = self._page()
        self._existing_record()

        summary, stdout, _ = self._run_command('--dry-run')

        self.assertIn("Would system-link ObservationRecord observation_id='10'", stdout)
        self.assertTrue(summary.endswith('would link: 1, links skipped: 0'), summary)
        self.assertFalse(CampaignRunObservation.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_second_same_proposal_run_in_campaign_blocks_the_fallback(self, mock_make_request):
        mock_make_request.return_value = self._page()
        record = self._existing_record()
        CampaignRun.objects.create(
            campaign=self.campaign,
            target=NonSiderealTargetFactory.create(name='Didymos Other'),
            proposal_code='LCO2026A-003',
            telescope_class='1m0',
            source=CampaignRun.Source.LCO_QUEUE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='1m0/Sinistro',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 5),
        )

        summary, _, _ = self._run_command()

        self.assertFalse(CampaignRunObservation.objects.filter(observation_record=record).exists())
        self.assertTrue(summary.endswith('system links: 0, links skipped: 0'), summary)

    @patch('solsys_code.management.commands.backfill_lco_observation_records.make_request')
    def test_summary_line_is_printed_exactly_once(self, mock_make_request):
        # Quick task 261004-c7x: Django's BaseCommand.execute() already writes a returned string to
        # stdout, so handle() must return the summary and not also write it itself.
        mock_make_request.return_value = self._page()
        self._existing_record()

        # Dry run first so it writes nothing before the real run links the record.
        for extra, prefix in ((('--dry-run',), 'Would create:'), ((), 'Created:')):
            with self.subTest(extra=extra):
                summary, stdout, _ = self._run_command(*extra)

                self.assertIsInstance(summary, str)
                self.assertTrue(summary)
                self.assertEqual(stdout.count(summary), 1, stdout)
                self.assertEqual(stdout.splitlines()[-1], summary)
                self.assertTrue(summary.startswith(prefix), summary)


class TestRecheckUnscheduledRetiresAllocationNight(TestCase):
    """G-37.1-1-alloc: --recheck-unscheduled gives an existing record its aborted block's times, which
    retires the linked run's allocation night; the status call is NOT patched, so FOMO's block rule runs."""

    FIELD_NAME = 'Didymos COJ 2026 Field #14'
    ARGS = ('--proposal=LCO2026A-003', '--name-prefix=Didymos', '--campaign=Didymos 2026 Campaign')
    ABORTED = {'state': 'ABORTED', 'start': '2026-07-01T01:00:00Z', 'end': '2026-07-01T01:40:00Z'}
    LISTING = 'solsys_code.management.commands.backfill_lco_observation_records.make_request'
    PORTAL = 'solsys_code.observation_blocks.make_request'

    @classmethod
    def setUpTestData(cls):
        cls.target = NonSiderealTargetFactory.create(name='Didymos')
        cls.field_target = NonSiderealTargetFactory.create(name=cls.FIELD_NAME)
        cls.campaign = TargetList.objects.create(name='Didymos 2026 Campaign')
        cls.campaign.targets.add(cls.target, cls.field_target)
        cls.site = Observatory.objects.create(
            obscode='809',
            name='ESO, La Silla',
            short_name='NTT',
            lat=-29.2567,
            lon=-70.7300,
            altitude=2347,
            timezone='America/Santiago',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        cls.per_night_run = CampaignRun.objects.create(
            campaign=None,
            target=cls.target,
            proposal_code='LCO2026A-003',
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            site=cls.site,
            site_raw='809',
            window_start=date(2026, 6, 29),
            window_end=date(2026, 7, 2),
        )

    def setUp(self):
        reconcile_run(self.per_night_run)
        self.alloc_before = allocation_events(self.per_night_run).count()
        self.record_a = self._record('10', 'WINDOW_EXPIRED')
        self.record_b = self._record('20', 'CANCELED')
        for record in (self.record_a, self.record_b):
            create_system_link(record, self.per_night_run)

    def _record(self, observation_id, status):
        return ObservationRecord.objects.create(
            target=self.field_target,
            facility='LCO',
            observation_id=observation_id,
            status=status,
            parameters={
                'proposal': 'LCO2026A-003',
                'instrument_type': '1M0-SCICAM-SINISTRO',
                'start': '2026-06-29T00:00:00',
                'end': '2026-07-02T00:00:00',
            },
        )

    def _listing(self):
        return _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - ELP',
                    requests=[_request(10, target_name=self.FIELD_NAME), _request(20, target_name=self.FIELD_NAME)],
                )
            ]
        )

    def _portal(self):
        return portal_side_effect({'10': 'WINDOW_EXPIRED', '20': 'CANCELED'}, {'10': [dict(self.ABORTED)], '20': []})

    def _run_command(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        summary = call_command('backfill_lco_observation_records', *self.ARGS, *extra, stdout=stdout, stderr=stderr)
        return summary, stdout.getvalue(), stderr.getvalue()

    def _alloc_url(self, record):
        night = observing_night(record.scheduled_start, ZoneInfo('America/Santiago'))
        return f'ALLOC:{self.per_night_run.pk}:{night.isoformat()}'

    @patch(PORTAL)
    @patch(LISTING)
    def test_recheck_gives_the_aborted_block_times_and_retires_its_night(self, mock_listing, mock_portal):
        mock_listing.return_value = self._listing()
        mock_portal.side_effect = self._portal()

        summary, _, stderr = self._run_command('--recheck-unscheduled')

        self.record_a.refresh_from_db()
        self.record_b.refresh_from_db()
        self.assertIsNotNone(self.record_a.scheduled_start)
        self.assertIsNotNone(self.record_a.scheduled_end)
        self.assertEqual(self.record_a.scheduled_start, datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc))
        self.assertFalse(allocation_events(self.per_night_run).filter(url=self._alloc_url(self.record_a)).exists())
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before - 1)
        # The never-scheduled request gets no times and keeps its night (Phase 35 D-05).
        self.assertIsNone(self.record_b.scheduled_start)
        self.assertIsNone(self.record_b.scheduled_end)
        self.assertIn('already existed: 2', summary)
        self.assertIn('schedules rechecked: 2, blocks found: 1', summary)
        self.assertTrue(summary.endswith('system links: 0, links skipped: 0'), summary)
        self.assertEqual(stderr, '')

    @patch(PORTAL)
    @patch(LISTING)
    def test_without_the_flag_an_existing_record_is_not_touched(self, mock_listing, mock_portal):
        mock_listing.return_value = self._listing()
        mock_portal.side_effect = self._portal()

        summary, _, _ = self._run_command()

        mock_portal.assert_not_called()
        self.record_a.refresh_from_db()
        self.assertIsNone(self.record_a.scheduled_start)
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before)
        self.assertIn('already existed: 2', summary)
        self.assertIn('schedules rechecked: 0, blocks found: 0', summary)

    @patch(PORTAL)
    @patch(LISTING)
    def test_dry_run_recheck_previews_and_writes_nothing(self, mock_listing, mock_portal):
        mock_listing.return_value = self._listing()
        mock_portal.side_effect = self._portal()

        summary, stdout, _ = self._run_command('--dry-run', '--recheck-unscheduled')

        mock_portal.assert_not_called()
        for observation_id in ('10', '20'):
            self.assertIn(
                f"Would recheck the observed block of ObservationRecord observation_id='{observation_id}'", stdout
            )
        self.record_a.refresh_from_db()
        self.assertIsNone(self.record_a.scheduled_start)
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before)
        self.assertIn('schedules would recheck: 2, blocks found: n/a (dry-run)', summary)

    @patch(PORTAL)
    @patch(LISTING)
    def test_portal_failure_is_reported_by_class_only_and_does_not_stop_the_run(self, mock_listing, mock_portal):
        mock_listing.return_value = self._listing()
        working = self._portal()

        def _flaky(method, url, **kwargs):
            if url.endswith('/api/requests/10'):
                raise requests.HTTPError('portal said SECRET-MARKER-123')
            return working(method, url, **kwargs)

        mock_portal.side_effect = _flaky

        summary, stdout, stderr = self._run_command('--recheck-unscheduled')

        self.assertIn('HTTPError', stderr)
        self.assertNotIn('SECRET-MARKER-123', stderr)
        self.assertNotIn('SECRET-MARKER-123', stdout)
        self.assertNotIn('SECRET-MARKER-123', summary)
        self.assertIn('status sync failed: 1', summary)
        self.assertIn('schedules rechecked: 1, blocks found: 0', summary)
        # The other record was still rechecked.
        self.record_b.refresh_from_db()
        self.assertTrue(any(c.args[1].endswith('/api/requests/20') for c in mock_portal.call_args_list))

    @patch(PORTAL)
    @patch(LISTING)
    def test_a_record_with_both_times_is_not_rechecked(self, mock_listing, mock_portal):
        mock_listing.return_value = self._listing()
        mock_portal.side_effect = self._portal()
        ObservationRecord.objects.filter(pk=self.record_a.pk).update(
            scheduled_start=datetime(2026, 7, 1, 2, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 7, 1, 2, 30, tzinfo=timezone.utc),
        )

        summary, _, _ = self._run_command('--recheck-unscheduled')

        urls = [c.args[1] for c in mock_portal.call_args_list]
        self.assertFalse([u for u in urls if '/requests/10' in u], urls)
        self.assertTrue([u for u in urls if '/requests/20' in u], urls)
        self.assertIn('schedules rechecked: 1, blocks found: 0', summary)


class TestRecheckUnscheduledFailedBlocks(TestCase):
    """G-37.1-6: the portal reports a block that started, took data and stopped early as FAILED. The Didymos
    command, run with --recheck-unscheduled on the real portal blocks, stores their times and retires exactly
    their allocation nights; the status call is NOT patched, so FOMO's block rule runs."""

    FIELD_NAME = 'Didymos COJ 2026 Field #14'
    ARGS = ('--proposal=LCO2026A-003', '--name-prefix=Didymos', '--campaign=Didymos 2026 Campaign')
    LISTING = 'solsys_code.management.commands.backfill_lco_observation_records.make_request'
    PORTAL = 'solsys_code.observation_blocks.make_request'
    # The four real requests whose only block the portal reports FAILED after taking data, a synthetic request
    # whose FAILED block took nothing (4999001), and a real never-scheduled request (4253584).
    REAL_IDS = ('4253588', '4272067', '4276100', '4282342')
    NO_DATA_ID = '4999001'
    NEVER_SCHEDULED_ID = '4253584'
    RETIRED_NIGHTS = ('2026-07-12', '2026-07-17', '2026-07-19', '2026-07-20')

    @classmethod
    def setUpTestData(cls):
        cls.target = NonSiderealTargetFactory.create(name='Didymos')
        cls.field_target = NonSiderealTargetFactory.create(name=cls.FIELD_NAME)
        cls.campaign = TargetList.objects.create(name='Didymos 2026 Campaign')
        cls.campaign.targets.add(cls.target, cls.field_target)
        cls.site = Observatory.objects.create(
            obscode='E10',
            name='Siding Spring (FTS)',
            short_name='FTS',
            lon=149.0708,
            lat=-31.2733,
            altitude=1165.0,
            timezone='Australia/Sydney',
        )
        cls.per_night_run = CampaignRun.objects.create(
            campaign=None,
            target=cls.target,
            proposal_code='LCO2026A-003',
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='FTS/MuSCAT4',
            site=cls.site,
            site_raw='E10',
            window_start=date(2026, 7, 11),
            window_end=date(2026, 7, 20),
        )

    def setUp(self):
        reconcile_run(self.per_night_run)
        self.alloc_before = allocation_events(self.per_night_run).count()
        self.records = {
            observation_id: self._record(observation_id)
            for observation_id in (*self.REAL_IDS, self.NO_DATA_ID, self.NEVER_SCHEDULED_ID)
        }
        for record in self.records.values():
            create_system_link(record, self.per_night_run)

    def _record(self, observation_id):
        return ObservationRecord.objects.create(
            target=self.field_target,
            facility='LCO',
            observation_id=observation_id,
            status='WINDOW_EXPIRED',
            parameters={
                'proposal': 'LCO2026A-003',
                'instrument_type': '2M0-SCICAM-MUSCAT',
                'start': '2026-07-11T00:00:00',
                'end': '2026-07-21T00:00:00',
            },
        )

    def _listing(self):
        requests_ = [
            _request(int(observation_id), target_name=self.FIELD_NAME, state='WINDOW_EXPIRED')
            for observation_id in self.records
        ]
        return _page_response([_request_group(1, 'Didymos 2026 - FTS', requests=requests_)])

    def _portal(self):
        states = {observation_id: 'WINDOW_EXPIRED' for observation_id in self.records}
        states['4276100'] = 'COMPLETED'
        blocks = {observation_id: [dict(REAL_FAILED_BLOCKS[observation_id])] for observation_id in self.REAL_IDS}
        # Synthetic: a FAILED block with nothing completed.
        blocks[self.NO_DATA_ID] = [failed_block(0.0, start='2026-07-13T09:00:00Z', end='2026-07-13T14:00:00Z')]
        blocks[self.NEVER_SCHEDULED_ID] = []
        return portal_side_effect(states, blocks)

    def _run_command(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        summary = call_command('backfill_lco_observation_records', *self.ARGS, *extra, stdout=stdout, stderr=stderr)
        return summary, stdout.getvalue(), stderr.getvalue()

    def _alloc_url(self, night):
        return f'ALLOC:{self.per_night_run.pk}:{night}'

    @patch(PORTAL)
    @patch(LISTING)
    def test_recheck_stores_failed_blocks_that_took_data_and_retires_their_nights(self, mock_listing, mock_portal):
        mock_listing.return_value = self._listing()
        mock_portal.side_effect = self._portal()

        summary, _, stderr = self._run_command('--recheck-unscheduled')

        self.assertIn('already existed: 6', summary)
        self.assertIn('status sync failed: 0', summary)
        self.assertIn('schedules rechecked: 6, blocks found: 4', summary)
        self.assertEqual(stderr, '')
        for observation_id in self.REAL_IDS:
            record = ObservationRecord.objects.get(observation_id=observation_id)
            block = REAL_FAILED_BLOCKS[observation_id]
            self.assertEqual(
                record.scheduled_start, datetime.fromisoformat(block['start'].replace('Z', '+00:00')), observation_id
            )
            self.assertEqual(
                record.scheduled_end, datetime.fromisoformat(block['end'].replace('Z', '+00:00')), observation_id
            )
        self.assertEqual(ObservationRecord.objects.get(observation_id='4276100').status, 'COMPLETED')
        for observation_id in (self.NO_DATA_ID, self.NEVER_SCHEDULED_ID):
            record = ObservationRecord.objects.get(observation_id=observation_id)
            self.assertIsNone(record.scheduled_start, observation_id)
            self.assertIsNone(record.scheduled_end, observation_id)
        urls = set(allocation_events(self.per_night_run).values_list('url', flat=True))
        for night in self.RETIRED_NIGHTS:
            self.assertNotIn(self._alloc_url(night), urls, night)
        # The FAILED block that took nothing keeps its night. The never-scheduled request's record has no times
        # (asserted above), and the count below shows no fifth night retired.
        self.assertIn(self._alloc_url('2026-07-13'), urls)
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before - 4)

    @patch(PORTAL)
    @patch(LISTING)
    def test_dry_run_recheck_previews_and_writes_nothing(self, mock_listing, mock_portal):
        mock_listing.return_value = self._listing()
        mock_portal.side_effect = self._portal()

        summary, _, _ = self._run_command('--dry-run', '--recheck-unscheduled')

        mock_portal.assert_not_called()
        self.assertIn('schedules would recheck: 6, blocks found: n/a (dry-run)', summary)
        for record in ObservationRecord.objects.filter(observation_id__in=list(self.records)):
            self.assertIsNone(record.scheduled_start, record.observation_id)
            self.assertIsNone(record.scheduled_end, record.observation_id)
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before)
