import copy
import io
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import requests
from django.contrib.auth.models import User
from django.core.management import CommandError, call_command
from django.db import IntegrityError
from django.test import SimpleTestCase, TestCase
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import Target, TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.allocation_projector import allocation_events
from solsys_code.campaign_attribution import record_attribution_backlog
from solsys_code.campaign_reconciler import reconcile_run
from solsys_code.campaign_utils import create_system_link
from solsys_code.management.commands.backfill_lco_observations import (
    _build_parameters,
    _preserve_observed_site_keys,
    _schedule_lookup_is_needed,
    sweep_proposal,
)
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation, WatchedProposal
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import observing_night
from solsys_code.tests.test_observation_blocks import portal_side_effect

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


def _expected_summary(
    dry_run,
    requestgroups_seen,
    created,
    updated,
    unchanged,
    skipped,
    targets,
    groups_created,
    groups_reused,
    embedded_blocks,
    fallback_lookups_needed,
    block_lookups_failed,
    fallback_lookups_skipped=0,
    list_name='LCO2026A-003_targets',
    list_reused=False,
    targets_added=0,
    system_links=0,
    links_skipped=0,
):
    """Build the exact summary line a run over these counts should produce.

    Deliberately independent of the command module -- it spells every label out literally
    and never imports or calls anything from backfill_lco_observations -- so this helper
    is the test suite's own copy of the summary-line contract. A helper that re-derived the
    labels from the command would agree with the command even when the command is wrong,
    which defeats the point of an exact-line assertion (T-ik7-02).
    """
    created_label = 'would create' if dry_run else 'created'
    updated_label = 'would update' if dry_run else 'updated'
    targets_label = 'targets would create' if dry_run else 'targets created'
    groups_created_label = 'groups would create' if dry_run else 'groups created'
    groups_reused_label = 'groups would reuse' if dry_run else 'groups reused'
    block_lookups_failed_value = 'n/a (dry-run)' if dry_run else str(block_lookups_failed)
    if dry_run:
        list_verb = 'would reuse' if list_reused else 'would create'
    else:
        list_verb = 'reused' if list_reused else 'created'
    targets_added_label = 'targets would add to list' if dry_run else 'targets added to list'
    system_links_label = 'would link' if dry_run else 'system links'
    return (
        f'requestgroups seen: {requestgroups_seen}, '
        f'{created_label}: {created}, '
        f'{updated_label}: {updated}, '
        f'unchanged: {unchanged}, skipped: {skipped}, '
        f'{targets_label}: {targets}, '
        f'{groups_created_label}: {groups_created}, '
        f'{groups_reused_label}: {groups_reused}, '
        f'embedded blocks: {embedded_blocks}, fallback lookups needed: {fallback_lookups_needed}, '
        f'fallback lookups skipped: {fallback_lookups_skipped}, '
        f'block lookups failed: {block_lookups_failed_value}, '
        f'target list: {list_verb} {list_name!r}, '
        f'{targets_added_label}: {targets_added}, '
        f'{system_links_label}: {system_links}, links skipped: {links_skipped}'
    )


class TestBackfillLcoObservations(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    def setUp(self):
        # The command constructs its own FomoLCOFacility() instance, so the D-B fallback lookup
        # must be patched at the class level, exactly as the sibling test patches
        # update_observation_status. A sensible default return value keeps every test that
        # doesn't care about block-derived times passing without extra setup; tests that do
        # care override .return_value/.side_effect explicitly.
        patcher = patch('solsys_code.observation_blocks.FomoLCOFacility.get_observation_status')
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
        # T-kpy-01: a dry run performs zero TargetList writes -- no row at all.
        self.assertFalse(TargetList.objects.exists())
        self.assertIn('requestgroups seen: 1', stdout.getvalue())
        self.mock_get_observation_status.assert_not_called()
        # Exact-line assertion, not just a fragment -- both requests are fresh, share the
        # already-existing 'Didymos' target, and neither carries an embedded block.
        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=2,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())
        # Django's BaseCommand.execute() writes handle()'s return value to self.stdout; an
        # explicit self.stdout.write(summary) inside handle() would double the line on the
        # operator's terminal, so pin the count at exactly once.
        self.assertEqual(stdout.getvalue().count(expected), 1)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_would_update_when_status_differs(self, mock_make_request):
        # Pins the would-update counter: a pre-existing record whose portal status differs
        # is counted 'would update: 1'.
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='PENDING')])]
        )
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='COMPLETED')])]
        )
        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=0,
            updated=1,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=True,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())
        # The dry run must not have actually changed the record.
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.status, 'PENDING')

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_unchanged_for_identical_embedded_block_request(self, mock_make_request):
        # Pins the unchanged counter on the full four-field comparison path (embedded
        # block present, so schedule times are compared too).
        observations = [{'state': 'COMPLETED', 'start': '2026-07-01T00:05:00', 'end': '2026-07-01T00:15:00'}]
        request_group = _request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, observations=observations)])
        mock_make_request.return_value = _page_response([request_group])
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        mock_make_request.return_value = _page_response([request_group])
        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=0,
            updated=0,
            unchanged=1,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=1,
            fallback_lookups_needed=0,
            block_lookups_failed=0,
            list_reused=True,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_unchanged_for_identical_fallback_path_request(self, mock_make_request):
        # Pins the unchanged counter on the fallback path (no embedded block, so the dry
        # run compares status/parameters only -- per the compare_schedule=embedded caveat).
        # The request is PENDING, not COMPLETED, so that the fallback path still NEEDS a
        # lookup: a finished record at the same portal state is skipped instead (F2).
        request_group = _request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='PENDING')])
        mock_make_request.return_value = _page_response([request_group])
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        mock_make_request.return_value = _page_response([request_group])
        self.mock_get_observation_status.reset_mock()
        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=0,
            updated=0,
            unchanged=1,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=True,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_targets_would_create_deduped_within_group(self, mock_make_request):
        # Pins the targets counter's same-invocation de-duplication: two requests in one
        # group naming the same absent target report 'targets would create: 1', matching
        # what a real run reports.
        elements = dict(_DEFAULT_ELEMENTS)
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'New Object 2026 - Multi',
                    requests=[
                        _request(10, target_name='2026 CD2', elements=elements),
                        _request(11, target_name='2026 CD2', elements=elements),
                    ],
                )
            ]
        )

        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=1,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=2,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())
        self.assertEqual(Target.objects.filter(name='2026 CD2').count(), 0)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_groups_would_create_then_would_reuse(self, mock_make_request):
        # Pins the groups counters: no existing ObservationGroup reports 'groups would
        # create: 1'; after a real run has created it, a second dry run reports
        # 'groups would reuse: 1'.
        request_group = _request_group(1, 'Didymos 2026 - Multi', requests=[_request(10), _request(11)])
        mock_make_request.return_value = _page_response([request_group])

        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )
        expected_create = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=2,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertIn(expected_create, stdout.getvalue())
        self.assertFalse(ObservationGroup.objects.exists())

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(ObservationGroup.objects.count(), 1)

        stdout2 = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout2, stderr=io.StringIO()
        )
        expected_reuse = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=0,
            updated=0,
            unchanged=2,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=1,
            embedded_blocks=0,
            fallback_lookups_needed=0,
            fallback_lookups_skipped=2,
            block_lookups_failed=0,
            list_reused=True,
            targets_added=1,
        )
        self.assertIn(expected_reuse, stdout2.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_reports_mixed_schedule_path_counters(self, mock_make_request):
        # Pins the schedule-path counters and the dry-run failed-lookup label together, for
        # a fixture with one embedded-block request and one fallback-path request.
        observations = [{'state': 'COMPLETED', 'start': '2026-07-01T00:05:00', 'end': '2026-07-01T00:15:00'}]
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - Multi',
                    requests=[_request(10, observations=observations), _request(11)],
                )
            ]
        )

        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=1,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_real_run_reports_mixed_schedule_path_counters(self, mock_make_request):
        # Pins the schedule-path counters in real-run mode, and the exact-count failed-
        # lookup value (not the dry-run 'n/a' string): the fallback request triggers
        # exactly one live get_observation_status call.
        observations = [{'state': 'COMPLETED', 'start': '2026-07-01T00:05:00', 'end': '2026-07-01T00:15:00'}]
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - Multi',
                    requests=[_request(10, observations=observations), _request(11)],
                )
            ]
        )

        stdout = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout, stderr=io.StringIO())

        expected = _expected_summary(
            dry_run=False,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=1,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())
        self.mock_get_observation_status.assert_called_once()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_reports_accurate_counters_end_to_end(self, mock_make_request):
        # DRYRUN-01..04: a single-request fresh RequestGroup naming an absent target must
        # report real counts, not structural zeros -- this is the whole bug being fixed.
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'New Object 2026 - ELP',
                    requests=[_request(10, target_name='2026 AB1', elements=dict(_DEFAULT_ELEMENTS))],
                )
            ]
        )

        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        expected_summary = (
            'requestgroups seen: 1, would create: 1, would update: 0, unchanged: 0, '
            'skipped: 0, targets would create: 1, groups would create: 0, groups would reuse: 0, '
            'embedded blocks: 0, fallback lookups needed: 1, fallback lookups skipped: 0, '
            'block lookups failed: n/a (dry-run), '
            "target list: would create 'LCO2026A-003_targets', targets would add to list: 1"
        )
        self.assertIn(expected_summary, stdout.getvalue())
        self.assertFalse(ObservationRecord.objects.exists())
        self.assertEqual(Target.objects.count(), 1)  # only setUpTestData's 'Didymos'
        self.assertFalse(ObservationGroup.objects.exists())
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_target_list_created_with_matched_and_new_targets(self, mock_make_request):
        # TL-01: a real sweep collects both the fuzzy-matched existing target and the
        # newly built one into the derived '<proposal>_targets' list.
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos and New 2026 - Multi',
                    requests=[
                        _request(10, target_name='Didymos'),
                        _request(11, target_name='2026 AB1', elements=dict(_DEFAULT_ELEMENTS)),
                    ],
                )
            ]
        )

        stdout = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout, stderr=io.StringIO())

        target_list = TargetList.objects.get(name='LCO2026A-003_targets')
        new_target = Target.objects.get(name='2026 AB1')
        self.assertEqual(set(target_list.targets.all()), {self.existing_target, new_target})
        expected = _expected_summary(
            dry_run=False,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=1,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=2,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=2,
        )
        self.assertIn(expected, stdout.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_target_list_rerun_is_idempotent(self, mock_make_request):
        # TL-02: re-running the same sweep creates no second TargetList and adds no
        # duplicate membership; run two's summary reports the reused verb with the same
        # added count as run one (D-04).
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos and New 2026 - Multi',
                    requests=[
                        _request(10, target_name='Didymos'),
                        _request(11, target_name='2026 AB1', elements=dict(_DEFAULT_ELEMENTS)),
                    ],
                )
            ]
        )
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(TargetList.objects.count(), 1)
        count_after_run_one = TargetList.objects.get(name='LCO2026A-003_targets').targets.count()

        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos and New 2026 - Multi',
                    requests=[
                        _request(10, target_name='Didymos'),
                        _request(11, target_name='2026 AB1', elements=dict(_DEFAULT_ELEMENTS)),
                    ],
                )
            ]
        )
        stdout2 = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout2, stderr=io.StringIO())

        self.assertEqual(TargetList.objects.count(), 1)
        count_after_run_two = TargetList.objects.get(name='LCO2026A-003_targets').targets.count()
        self.assertEqual(count_after_run_two, count_after_run_one)

        expected = _expected_summary(
            dry_run=False,
            requestgroups_seen=1,
            created=0,
            updated=0,
            unchanged=2,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=1,
            embedded_blocks=0,
            fallback_lookups_needed=0,
            fallback_lookups_skipped=2,
            block_lookups_failed=0,
            list_reused=True,
            targets_added=2,
        )
        self.assertIn(expected, stdout2.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_target_list_override_name(self, mock_make_request):
        # TL-03: --target-list overrides the derived name; the derived name is never created.
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos and New 2026 - Multi',
                    requests=[
                        _request(10, target_name='Didymos'),
                        _request(11, target_name='2026 AB1', elements=dict(_DEFAULT_ELEMENTS)),
                    ],
                )
            ]
        )

        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations',
            '--proposal=LCO2026A-003',
            '--target-list=SweepList',
            stdout=stdout,
            stderr=io.StringIO(),
        )

        self.assertFalse(TargetList.objects.filter(name='LCO2026A-003_targets').exists())
        target_list = TargetList.objects.get(name='SweepList')
        new_target = Target.objects.get(name='2026 AB1')
        self.assertEqual(set(target_list.targets.all()), {self.existing_target, new_target})
        expected = _expected_summary(
            dry_run=False,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=1,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=2,
            block_lookups_failed=0,
            list_name='SweepList',
            list_reused=False,
            targets_added=2,
        )
        self.assertIn(expected, stdout.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_target_list_dry_run_zero_writes_matches_real_pass_count(self, mock_make_request):
        # TL-05/TL-06: a dry run over the same payload as
        # test_target_list_created_with_matched_and_new_targets writes no TargetList row at
        # all, yet reports the would-add count (2) matching what the real pass reports.
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos and New 2026 - Multi',
                    requests=[
                        _request(10, target_name='Didymos'),
                        _request(11, target_name='2026 AB1', elements=dict(_DEFAULT_ELEMENTS)),
                    ],
                )
            ]
        )

        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        self.assertFalse(TargetList.objects.exists())
        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=2,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=1,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=2,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=2,
        )
        self.assertIn(expected, stdout.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_target_list_dry_run_would_reuse_after_real_run(self, mock_make_request):
        # TL-05: after a real run creates the list, a following dry run reports the
        # would-reuse verb and leaves the list's membership count unchanged.
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        count_after_real_run = TargetList.objects.get(name='LCO2026A-003_targets').targets.count()

        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])
        stdout = io.StringIO()
        call_command(
            'backfill_lco_observations', '--proposal=LCO2026A-003', '--dry-run', stdout=stdout, stderr=io.StringIO()
        )

        self.assertEqual(TargetList.objects.count(), 1)
        self.assertEqual(TargetList.objects.get(name='LCO2026A-003_targets').targets.count(), count_after_real_run)
        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=0,
            updated=0,
            unchanged=1,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=0,
            fallback_lookups_skipped=1,
            block_lookups_failed=0,
            list_reused=True,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_target_list_excludes_targets_from_skipped_requests(self, mock_make_request):
        # TL-04: a target-step skip (Unmappable Object, wrong target type) and a
        # parameters-step skip (Didymos, matched before its instrument_type is emptied)
        # contribute nothing to the list -- proving collection sits after every skip branch,
        # not before the parameters check.
        request_c = _request(12, target_name='Didymos')
        request_c['configurations'][0]['instrument_type'] = ''
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Mixed 2026 - Multi',
                    requests=[
                        _request(10, target_name='2026 AB1', elements=dict(_DEFAULT_ELEMENTS)),
                        _request(11, target_name='Unmappable Object', target_type='SIDEREAL'),
                        request_c,
                    ],
                )
            ]
        )

        stdout = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout, stderr=io.StringIO())

        target_list = TargetList.objects.get(name='LCO2026A-003_targets')
        self.assertEqual(list(target_list.targets.values_list('name', flat=True)), ['2026 AB1'])
        expected = _expected_summary(
            dry_run=False,
            requestgroups_seen=1,
            created=1,
            updated=0,
            unchanged=0,
            skipped=2,
            targets=1,
            groups_created=1,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertIn(expected, stdout.getvalue())

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

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_proposal_override_works_without_a_watched_proposal_row(self, mock_make_request):
        """D-07: the --proposal override never requires the named code to be watched."""
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertTrue(ObservationRecord.objects.filter(facility='LCO', observation_id='10').exists())
        self.assertFalse(WatchedProposal.objects.exists())


class TestSweepProposalFunction(TestCase):
    """Task 2 (36-RESEARCH.md Open Question 2, resolution): sweep_proposal() called
    directly -- no management command involved -- returns a summary string
    byte-identical to the one handle() returned before the extraction, for both the
    dry-run and the real pass."""

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    def setUp(self):
        patcher = patch('solsys_code.observation_blocks.FomoLCOFacility.get_observation_status')
        self.mock_get_observation_status = patcher.start()
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        self.addCleanup(patcher.stop)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_direct_call_returns_the_same_summary_as_handle_for_a_real_pass(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        summary = sweep_proposal('LCO2026A-003')

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.target, self.existing_target)
        expected = _expected_summary(
            dry_run=False,
            requestgroups_seen=1,
            created=1,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertEqual(summary, expected)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_direct_call_returns_the_same_summary_as_handle_for_a_dry_run(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        summary = sweep_proposal('LCO2026A-003', dry_run=True)

        self.assertFalse(ObservationRecord.objects.exists())
        expected = _expected_summary(
            dry_run=True,
            requestgroups_seen=1,
            created=1,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
        )
        self.assertEqual(summary, expected)


def _watched_side_effect(*proposal_to_group_kwargs):
    """Build a make_request side_effect routing each proposal's query to its own page.

    Args:
        *proposal_to_group_kwargs: any number of (proposal_code, request_group_kwargs)
            pairs; request_group_kwargs is passed to _request_group() (id/name/requests).

    Returns:
        Callable: a make_request(method, url, **kwargs) side_effect. Raises AssertionError
            for a URL that names none of the given proposal codes -- a test-authoring bug,
            never a real portal response, so it must never be silently swallowed.
    """
    by_code = dict(proposal_to_group_kwargs)

    def side_effect(method, url, **kwargs):
        for code, group_kwargs in by_code.items():
            if f'proposal={code}' in url:
                return _page_response([_request_group(proposal=code, **group_kwargs)])
        raise AssertionError(f'unexpected proposal queried: {url}')

    return side_effect


class TestWatchedListSweep(TestCase):
    """Task 3 (36-CONTEXT.md D-06/D-07): a bare invocation sweeps every active
    WatchedProposal row, applying each row's overrides, in proposal_code order."""

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    def setUp(self):
        patcher = patch('solsys_code.observation_blocks.FomoLCOFacility.get_observation_status')
        self.mock_get_observation_status = patcher.start()
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        self.addCleanup(patcher.stop)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_bare_invocation_sweeps_every_active_row(self, mock_make_request):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        WatchedProposal.objects.create(proposal_code='CCC-2026-003', is_active=False)
        mock_make_request.side_effect = _watched_side_effect(
            ('AAA-2026-001', {'group_id': 1, 'name': 'A run', 'requests': [_request(10)]}),
            ('BBB-2026-002', {'group_id': 2, 'name': 'B run', 'requests': [_request(20)]}),
        )

        call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(mock_make_request.call_count, 2)
        queried_urls = [call.args[1] for call in mock_make_request.call_args_list]
        self.assertTrue(any('proposal=AAA-2026-001' in url for url in queried_urls))
        self.assertTrue(any('proposal=BBB-2026-002' in url for url in queried_urls))
        self.assertFalse(any('proposal=CCC-2026-003' in url for url in queried_urls))

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_row_overrides_are_applied(self, mock_make_request):
        user = User.objects.create_user(username='attributee-watched')
        WatchedProposal.objects.create(proposal_code='AAA-2026-001', target_list_name='custom_list', attributed_to=user)
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        mock_make_request.side_effect = _watched_side_effect(
            ('AAA-2026-001', {'group_id': 1, 'name': 'A run', 'requests': [_request(10)]}),
            ('BBB-2026-002', {'group_id': 2, 'name': 'B run', 'requests': [_request(20)]}),
        )

        call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertTrue(TargetList.objects.filter(name='custom_list').exists())
        record_a = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record_a.user, user)

        self.assertTrue(TargetList.objects.filter(name='BBB-2026-002_targets').exists())
        record_b = ObservationRecord.objects.get(facility='LCO', observation_id='20')
        self.assertIsNone(record_b.user)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_bookkeeping_written_per_row(self, mock_make_request):
        row_a = WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        row_b = WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        mock_make_request.side_effect = _watched_side_effect(
            ('AAA-2026-001', {'group_id': 1, 'name': 'A run', 'requests': [_request(10)]}),
            ('BBB-2026-002', {'group_id': 2, 'name': 'B run', 'requests': [_request(20)]}),
        )

        call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        row_a.refresh_from_db()
        row_b.refresh_from_db()
        self.assertIsNotNone(row_a.last_run_at)
        self.assertIsNotNone(row_b.last_run_at)
        counters = {
            'dry_run': False,
            'requestgroups_seen': 1,
            'created': 1,
            'updated': 0,
            'unchanged': 0,
            'skipped': 0,
            'targets': 0,
            'groups_created': 0,
            'groups_reused': 0,
            'embedded_blocks': 0,
            'fallback_lookups_needed': 1,
            'block_lookups_failed': 0,
            'list_reused': False,
            'targets_added': 1,
        }
        self.assertEqual(row_a.last_run_summary, _expected_summary(list_name='AAA-2026-001_targets', **counters))
        self.assertEqual(row_b.last_run_summary, _expected_summary(list_name='BBB-2026-002_targets', **counters))

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_rows_are_swept_in_code_order(self, mock_make_request):
        # Insertion order is the reverse of alphabetical order.
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        mock_make_request.side_effect = _watched_side_effect(
            ('AAA-2026-001', {'group_id': 1, 'name': 'A run', 'requests': []}),
            ('BBB-2026-002', {'group_id': 2, 'name': 'B run', 'requests': []}),
        )

        call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        queried_urls = [call.args[1] for call in mock_make_request.call_args_list]
        self.assertEqual(len(queried_urls), 2)
        self.assertIn('proposal=AAA-2026-001', queried_urls[0])
        self.assertIn('proposal=BBB-2026-002', queried_urls[1])

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_override_and_watched_row_produce_the_same_summary(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        stdout = io.StringIO()
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=stdout, stderr=io.StringIO())
        override_summary = stdout.getvalue().strip()

        # Reset the writes from the override run so the watched-path sweep below starts
        # from the same clean state over the identical mocked portal payload.
        ObservationRecord.objects.all().delete()
        TargetList.objects.all().delete()

        WatchedProposal.objects.create(proposal_code='LCO2026A-003')
        call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        row = WatchedProposal.objects.get(proposal_code='LCO2026A-003')
        self.assertEqual(row.last_run_summary, override_summary)


class TestPerProposalIsolation(TestCase):
    """Task 3 (36-CONTEXT.md D-09): a portal or data error on one watched proposal is
    caught, recorded on that row, and never stops the rest of the sweep."""

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    def setUp(self):
        patcher = patch('solsys_code.observation_blocks.FomoLCOFacility.get_observation_status')
        self.mock_get_observation_status = patcher.start()
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        self.addCleanup(patcher.stop)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_portal_error_on_one_row_does_not_stop_the_others(self, mock_make_request):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')

        def side_effect(method, url, **kwargs):
            if 'proposal=AAA-2026-001' in url:
                raise requests.exceptions.HTTPError('boom')
            if 'proposal=BBB-2026-002' in url:
                return _page_response([_request_group(2, 'B run', proposal='BBB-2026-002', requests=[_request(20)])])
            raise AssertionError(f'unexpected proposal queried: {url}')

        mock_make_request.side_effect = side_effect

        with self.assertRaises(CommandError):
            call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertTrue(ObservationRecord.objects.filter(facility='LCO', observation_id='20').exists())
        row_a = WatchedProposal.objects.get(proposal_code='AAA-2026-001')
        self.assertTrue(row_a.last_run_summary.startswith('failed:'))
        self.assertIn('HTTPError', row_a.last_run_summary)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_failure_summary_carries_no_exception_message(self, mock_make_request):
        fake_key = 'sk-FAKE-API-KEY-jz8f0q2x9v'
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')

        def side_effect(method, url, **kwargs):
            raise requests.exceptions.HTTPError(f'401 Unauthorized: token={fake_key}')

        mock_make_request.side_effect = side_effect

        stderr = io.StringIO()
        with self.assertLogs('solsys_code.management.commands.backfill_lco_observations', level='DEBUG') as captured:
            with self.assertRaises(CommandError):
                call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=stderr)

        row = WatchedProposal.objects.get(proposal_code='AAA-2026-001')
        self.assertNotIn(fake_key, row.last_run_summary)
        self.assertNotIn(fake_key, stderr.getvalue())
        self.assertFalse(any(fake_key in message for message in captured.output))


class TestEmptyWatchedList(TestCase):
    """Task 3 (36-CONTEXT.md D-08): zero active WatchedProposal rows is a quiet no-op."""

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_no_active_rows_is_a_quiet_no_op(self, mock_make_request):
        stdout = io.StringIO()

        call_command('backfill_lco_observations', stdout=stdout, stderr=io.StringIO())

        mock_make_request.assert_not_called()
        self.assertFalse(TargetList.objects.exists())
        self.assertFalse(ObservationRecord.objects.exists())
        self.assertEqual(stdout.getvalue().count('0 watched proposals, nothing to discover'), 1)


class TestBareInvocationRejectsProposalOnlyFlags(TestCase):
    """CR-02 (36-REVIEW.md): --created-after/--created-before/--username/--target-list
    were silently discarded on the bare (no --proposal) invocation, running a
    full-history sweep of every watched proposal with no error and no mention in the
    summary. Fail closed instead: each flag requires --proposal.
    """

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_created_after_without_proposal_raises_command_error(self, mock_make_request):
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'backfill_lco_observations',
                '--created-after=2026-01-01T00:00:00',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        self.assertIn('--created-after', str(ctx.exception))
        self.assertIn('--proposal', str(ctx.exception))
        # IN-19 (36-REVIEW.md): the message used the plural verb unconditionally,
        # reading "--created-after require --proposal" for the single-flag case --
        # by far the common one, and the one the paired notebook exercises.
        self.assertIn('--created-after requires --proposal', str(ctx.exception))
        mock_make_request.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_multiple_proposal_only_flags_use_the_plural_verb(self, mock_make_request):
        # IN-19 (36-REVIEW.md): two or more ignored flags is the one case where "require"
        # is grammatically correct -- pin it so the singular/plural branch is exercised.
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'backfill_lco_observations',
                '--created-after=2026-01-01T00:00:00',
                '--username=someuser',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        self.assertIn('--created-after, --username require --proposal', str(ctx.exception))
        mock_make_request.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_created_before_without_proposal_raises_command_error(self, mock_make_request):
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'backfill_lco_observations',
                '--created-before=2026-12-31T00:00:00',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        self.assertIn('--created-before', str(ctx.exception))
        mock_make_request.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_username_without_proposal_raises_command_error(self, mock_make_request):
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)
        User.objects.create_user(username='someone')
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'backfill_lco_observations',
                '--username=someone',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        self.assertIn('--username', str(ctx.exception))
        mock_make_request.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_target_list_without_proposal_raises_command_error(self, mock_make_request):
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'backfill_lco_observations',
                '--target-list=override_targets',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        self.assertIn('--target-list', str(ctx.exception))
        mock_make_request.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_bare_invocation_with_no_proposal_only_flags_still_sweeps(self, mock_make_request):
        """The fix must not touch the legitimate bare invocation the unattended runner uses."""
        mock_make_request.return_value = _page_response([])
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)

        call_command('backfill_lco_observations', '--dry-run', stdout=io.StringIO(), stderr=io.StringIO())

        mock_make_request.assert_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_unknown_username_without_proposal_reports_the_proposal_guard_not_invalid_username(self, mock_make_request):
        # IN-12 (36-REVIEW.md): the --proposal-only guard must run before --username is
        # resolved to a User -- otherwise an unknown username on the bare invocation
        # reports the less useful "Invalid username" error instead of the guard that
        # actually explains why the command is rejecting the invocation.
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'backfill_lco_observations',
                '--username=ghost-user-that-does-not-exist',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        self.assertIn('--username', str(ctx.exception))
        self.assertIn('--proposal', str(ctx.exception))
        self.assertNotIn('Invalid username', str(ctx.exception))
        mock_make_request.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_empty_string_target_list_without_proposal_still_raises(self, mock_make_request):
        # IN-12 (36-REVIEW.md): the guard must test "was the flag supplied at all"
        # (options.get(key) is not None), not truthiness -- an empty-string flag was
        # still supplied and must still trip the guard.
        WatchedProposal.objects.create(proposal_code='LCO2026A-003', is_active=True)
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'backfill_lco_observations',
                '--target-list=',
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        self.assertIn('--target-list', str(ctx.exception))
        mock_make_request.assert_not_called()


_MOVED_WINDOW = [{'start': '2026-07-03T00:00:00', 'end': '2026-07-04T00:00:00'}]


class TestObservedSiteKeysSurviveDiscovery(TestCase):
    """Discovery must carry the projector sweep's observed-site keys forward (F1).

    The three key names are spelled literally on purpose: they are the contract with the
    sweep's resolve_observed_site(), so renaming the shared constant must fail these tests.
    """

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    def setUp(self):
        patcher = patch('solsys_code.observation_blocks.FomoLCOFacility.get_observation_status')
        self.mock_get_observation_status = patcher.start()
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        self.addCleanup(patcher.stop)

    def _first_pass(self, mock_make_request, windows=None):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, windows=windows)])]
        )
        call_command('backfill_lco_observations', '--proposal=LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

    def _tag_as_swept(self, observation_id='10', site='elp', telescope='1m0a', enclosure='doma'):
        """Write the three keys exactly as resolve_observed_site() does, and reload."""
        record = ObservationRecord.objects.get(facility='LCO', observation_id=observation_id)
        record.parameters.update(
            {'observed_site': site, 'observed_telescope': telescope, 'observed_enclosure': enclosure}
        )
        record.save(update_fields=['parameters'])
        # update_fields=['parameters'] does not write 'modified', so reload before capturing
        # it: otherwise a no-churn assertion compares against a value that was never stored.
        record.refresh_from_db()
        return record

    def _second_pass(self, mock_make_request, dry_run, windows=None):
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, windows=windows)])]
        )
        args = ['--proposal=LCO2026A-003']
        if dry_run:
            args.append('--dry-run')
        stdout = io.StringIO()
        call_command('backfill_lco_observations', *args, stdout=stdout, stderr=io.StringIO())
        return stdout.getvalue()

    def _summary(self, dry_run, updated, unchanged):
        return _expected_summary(
            dry_run=dry_run,
            requestgroups_seen=1,
            created=0,
            updated=updated,
            unchanged=unchanged,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=0,
            fallback_lookups_skipped=1,
            block_lookups_failed=0,
            list_reused=True,
            targets_added=1,
        )

    def _assert_keys(self, record, site='elp', telescope='1m0a', enclosure='doma'):
        self.assertEqual(record.parameters['observed_site'], site)
        self.assertEqual(record.parameters['observed_telescope'], telescope)
        self.assertEqual(record.parameters['observed_enclosure'], enclosure)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_real_run_over_unchanged_data_keeps_keys_and_does_not_save(self, mock_make_request):
        self._first_pass(mock_make_request)
        tagged = self._tag_as_swept()
        modified_before = tagged.modified

        out = self._second_pass(mock_make_request, dry_run=False)

        self.assertIn(self._summary(dry_run=False, updated=0, unchanged=1), out)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self._assert_keys(record)
        self.assertEqual(record.parameters['proposal'], 'LCO2026A-003')
        self.assertEqual(record.parameters['instrument_type'], '1M0-SCICAM-SINISTRO')
        self.assertEqual(record.parameters['start'], '2026-07-01T00:00:00')
        self.assertEqual(record.parameters['end'], '2026-07-02T00:00:00')
        self.assertEqual(record.modified, modified_before)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_real_run_with_moved_window_updates_portal_keys_and_keeps_observed_keys(self, mock_make_request):
        self._first_pass(mock_make_request)
        self._tag_as_swept()

        out = self._second_pass(mock_make_request, dry_run=False, windows=_MOVED_WINDOW)

        self.assertIn(self._summary(dry_run=False, updated=1, unchanged=0), out)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.parameters['start'], '2026-07-03T00:00:00')
        self.assertEqual(record.parameters['end'], '2026-07-04T00:00:00')
        self._assert_keys(record)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_over_unchanged_data_reports_unchanged_and_touches_nothing(self, mock_make_request):
        self._first_pass(mock_make_request)
        tagged = self._tag_as_swept()
        modified_before = tagged.modified

        out = self._second_pass(mock_make_request, dry_run=True)

        self.assertIn(self._summary(dry_run=True, updated=0, unchanged=1), out)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self._assert_keys(record)
        self.assertEqual(record.modified, modified_before)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_with_moved_window_would_update_and_writes_nothing(self, mock_make_request):
        self._first_pass(mock_make_request)
        self._tag_as_swept()

        out = self._second_pass(mock_make_request, dry_run=True, windows=_MOVED_WINDOW)

        self.assertIn(self._summary(dry_run=True, updated=1, unchanged=0), out)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.parameters['start'], '2026-07-01T00:00:00')
        self._assert_keys(record)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_none_valued_enclosure_is_carried_forward_as_none(self, mock_make_request):
        self._first_pass(mock_make_request)
        self._tag_as_swept(enclosure=None)

        out = self._second_pass(mock_make_request, dry_run=False)

        self.assertIn(self._summary(dry_run=False, updated=0, unchanged=1), out)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertIn('observed_enclosure', record.parameters)
        self.assertIsNone(record.parameters['observed_enclosure'])


class TestPreserveObservedSiteKeys(SimpleTestCase):
    def test_present_keys_are_copied_including_none_and_portal_keys_come_from_rebuilt(self):
        existing = {
            'proposal': 'LCO2026A-003',
            'start': '2026-07-01T00:00:00',
            'observed_site': 'elp',
            'observed_telescope': '1m0a',
            'observed_enclosure': None,
        }
        rebuilt = {'proposal': 'LCO2026A-003', 'start': '2026-07-03T00:00:00'}

        result = _preserve_observed_site_keys(existing, rebuilt)

        self.assertEqual(
            result,
            {
                'proposal': 'LCO2026A-003',
                'start': '2026-07-03T00:00:00',
                'observed_site': 'elp',
                'observed_telescope': '1m0a',
                'observed_enclosure': None,
            },
        )

    def test_existing_without_keys_gives_an_equal_new_dict(self):
        rebuilt = {'proposal': 'LCO2026A-003', 'start': '2026-07-03T00:00:00'}

        result = _preserve_observed_site_keys({'proposal': 'old'}, rebuilt)

        self.assertEqual(result, rebuilt)
        self.assertIsNot(result, rebuilt)

    def test_partial_existing_copies_only_the_keys_it_has(self):
        result = _preserve_observed_site_keys({'observed_site': 'elp'}, {'proposal': 'P'})

        self.assertEqual(result, {'proposal': 'P', 'observed_site': 'elp'})
        self.assertNotIn('observed_telescope', result)
        self.assertNotIn('observed_enclosure', result)

    def test_neither_argument_is_mutated(self):
        existing = {'proposal': 'old', 'observed_site': 'elp', 'observed_telescope': '1m0a'}
        rebuilt = {'proposal': 'P', 'start': 's'}
        existing_snapshot, rebuilt_snapshot = copy.deepcopy(existing), copy.deepcopy(rebuilt)

        _preserve_observed_site_keys(existing, rebuilt)

        self.assertEqual(existing, existing_snapshot)
        self.assertEqual(rebuilt, rebuilt_snapshot)

    def test_non_dict_existing_gives_a_copy_of_rebuilt(self):
        rebuilt = {'proposal': 'P'}

        result = _preserve_observed_site_keys(None, rebuilt)

        self.assertEqual(result, rebuilt)
        self.assertIsNot(result, rebuilt)


_MOVED_SCHEDULE = {
    'state': 'COMPLETED',
    'scheduled_start': '2026-07-01T05:00:00+00:00',
    'scheduled_end': '2026-07-01T05:10:00+00:00',
}


class TestSkipLookupForFinishedRecords(TestCase):
    """Discovery skips the live block lookup for a record finished at the same portal state (F2).

    Before every second pass the mock is reset and, unless a test says otherwise, made to
    return a MOVED schedule, so a lookup that should not have happened would show up as
    changed stored times as well as a recorded call.
    """

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    def setUp(self):
        patcher = patch('solsys_code.observation_blocks.FomoLCOFacility.get_observation_status')
        self.mock_get_observation_status = patcher.start()
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        self.addCleanup(patcher.stop)

    def _summary(
        self, dry_run, created=0, updated=0, unchanged=0, needed=0, skipped=0, embedded=0, failed=0, list_reused=True
    ):
        return _expected_summary(
            dry_run=dry_run,
            requestgroups_seen=1,
            created=created,
            updated=updated,
            unchanged=unchanged,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=embedded,
            fallback_lookups_needed=needed,
            fallback_lookups_skipped=skipped,
            block_lookups_failed=failed,
            list_reused=list_reused,
            targets_added=1,
        )

    def _pass(self, mock_make_request, dry_run=False, state='COMPLETED', windows=None, observations=None):
        mock_make_request.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - ELP',
                    requests=[_request(10, state=state, windows=windows, observations=observations)],
                )
            ]
        )
        args = ['--proposal=LCO2026A-003']
        if dry_run:
            args.append('--dry-run')
        stdout = io.StringIO()
        call_command('backfill_lco_observations', *args, stdout=stdout, stderr=io.StringIO())
        return stdout.getvalue()

    def _moved_lookup(self):
        self.mock_get_observation_status.reset_mock()
        self.mock_get_observation_status.return_value = dict(_MOVED_SCHEDULE)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_a_finished_record_at_same_state_is_skipped_and_untouched(self, mock_make_request):
        self._pass(mock_make_request)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        record.refresh_from_db()
        modified_before = record.modified
        self._moved_lookup()

        out = self._pass(mock_make_request)

        self.assertIn(self._summary(False, unchanged=1, skipped=1), out)
        self.mock_get_observation_status.assert_not_called()
        record.refresh_from_db()
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T00:10:00+00:00')
        self.assertEqual(record.modified, modified_before)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_b_pending_record_is_still_looked_up(self, mock_make_request):
        self._pass(mock_make_request, state='PENDING')
        self._moved_lookup()

        out = self._pass(mock_make_request, state='PENDING')

        self.assertIn(self._summary(False, updated=1, needed=1), out)
        self.assertEqual(self.mock_get_observation_status.call_count, 1)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T05:00:00+00:00')

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_c_state_change_from_terminal_to_terminal_is_looked_up(self, mock_make_request):
        self._pass(mock_make_request)
        self.mock_get_observation_status.reset_mock()
        self.mock_get_observation_status.return_value = {
            'state': 'CANCELED',
            'scheduled_start': None,
            'scheduled_end': None,
        }

        out = self._pass(mock_make_request, state='CANCELED')

        self.assertIn(self._summary(False, updated=1, needed=1), out)
        self.assertEqual(self.mock_get_observation_status.call_count, 1)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.status, 'CANCELED')
        self.assertIsNone(record.scheduled_start)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_d_brand_new_request_is_looked_up(self, mock_make_request):
        out = self._pass(mock_make_request)

        self.assertIn(self._summary(False, created=1, needed=1, list_reused=False), out)
        self.assertEqual(self.mock_get_observation_status.call_count, 1)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_e1_dry_run_and_real_run_agree_over_a_finished_record(self, mock_make_request):
        self._pass(mock_make_request)
        self._moved_lookup()

        dry_out = self._pass(mock_make_request, dry_run=True)
        real_out = self._pass(mock_make_request)

        self.assertIn(self._summary(True, unchanged=1, skipped=1), dry_out)
        self.assertIn(self._summary(False, unchanged=1, skipped=1), real_out)
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_e2_dry_run_over_a_pending_record_counts_needed_without_a_call(self, mock_make_request):
        self._pass(mock_make_request, state='PENDING')
        self._moved_lookup()

        out = self._pass(mock_make_request, dry_run=True, state='PENDING')

        self.assertIn(self._summary(True, unchanged=1, needed=1), out)
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_f_moved_window_updates_parameters_but_not_the_schedule(self, mock_make_request):
        self._pass(mock_make_request)
        self._moved_lookup()
        moved = [{'start': '2026-07-03T00:00:00', 'end': '2026-07-04T00:00:00'}]

        out = self._pass(mock_make_request, windows=moved)

        self.assertIn(self._summary(False, updated=1, skipped=1), out)
        self.mock_get_observation_status.assert_not_called()
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.parameters['start'], '2026-07-03T00:00:00')
        self.assertEqual(record.parameters['end'], '2026-07-04T00:00:00')
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T00:10:00+00:00')
        self.assertEqual(record.scheduled_end.isoformat(), '2026-07-01T00:20:00+00:00')

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_g_observed_site_keys_survive_a_skipped_update(self, mock_make_request):
        self._pass(mock_make_request)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        record.parameters.update({'observed_site': 'elp', 'observed_telescope': '1m0a', 'observed_enclosure': 'doma'})
        record.save(update_fields=['parameters'])
        self._moved_lookup()
        moved = [{'start': '2026-07-03T00:00:00', 'end': '2026-07-04T00:00:00'}]

        out = self._pass(mock_make_request, windows=moved)

        self.assertIn(self._summary(False, updated=1, skipped=1), out)
        record.refresh_from_db()
        self.assertEqual(record.parameters['observed_site'], 'elp')
        self.assertEqual(record.parameters['observed_telescope'], '1m0a')
        self.assertEqual(record.parameters['observed_enclosure'], 'doma')
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T00:10:00+00:00')

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_h1_completed_record_with_no_times_is_looked_up_until_they_arrive(self, mock_make_request):
        self.mock_get_observation_status.side_effect = Exception('portal down')

        out = self._pass(mock_make_request)

        self.assertIn(self._summary(False, created=1, needed=1, failed=1, list_reused=False), out)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.status, 'COMPLETED')
        self.assertIsNone(record.scheduled_start)

        self.mock_get_observation_status.reset_mock()
        self.mock_get_observation_status.side_effect = None
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }

        out = self._pass(mock_make_request)

        self.assertIn(self._summary(False, updated=1, needed=1), out)
        self.assertEqual(self.mock_get_observation_status.call_count, 1)
        record.refresh_from_db()
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T00:10:00+00:00')

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_h2_failed_state_with_no_times_stays_skipped(self, mock_make_request):
        self.mock_get_observation_status.return_value = {
            'state': 'WINDOW_EXPIRED',
            'scheduled_start': None,
            'scheduled_end': None,
        }
        self._pass(mock_make_request, state='WINDOW_EXPIRED')
        self.mock_get_observation_status.reset_mock()

        out = self._pass(mock_make_request, state='WINDOW_EXPIRED')

        self.assertIn(self._summary(False, unchanged=1, skipped=1), out)
        self.mock_get_observation_status.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_i_the_facilitys_own_terminal_list_decides(self, mock_make_request):
        self.mock_get_observation_status.return_value = {
            'state': 'WINDOW_EXPIRED',
            'scheduled_start': None,
            'scheduled_end': None,
        }
        self._pass(mock_make_request, state='WINDOW_EXPIRED')
        self.mock_get_observation_status.reset_mock()

        with patch(
            'tom_observations.facilities.lco.LCOFacility.get_terminal_observing_states', return_value=['COMPLETED']
        ):
            out = self._pass(mock_make_request, state='WINDOW_EXPIRED')

        self.assertIn(self._summary(False, unchanged=1, needed=1), out)
        self.assertEqual(self.mock_get_observation_status.call_count, 1)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_j_embedded_block_is_always_compared(self, mock_make_request):
        first = [{'state': 'COMPLETED', 'start': '2026-07-01T00:05:00', 'end': '2026-07-01T00:15:00'}]
        second = [{'state': 'COMPLETED', 'start': '2026-07-01T00:06:00', 'end': '2026-07-01T00:16:00'}]
        self._pass(mock_make_request, observations=first)
        self._moved_lookup()

        out = self._pass(mock_make_request, observations=second)

        self.assertIn(self._summary(False, updated=1, embedded=1), out)
        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.scheduled_start.isoformat(), '2026-07-01T00:06:00+00:00')
        self.mock_get_observation_status.assert_not_called()


_TERMINAL = frozenset({'COMPLETED', 'WINDOW_EXPIRED', 'CANCELED', 'FAILURE_LIMIT_REACHED', 'NOT_ATTEMPTED'})
_FAILED = _TERMINAL - {'COMPLETED'}
_START = datetime(2026, 7, 1, 0, 10, tzinfo=timezone.utc)
_END = datetime(2026, 7, 1, 0, 20, tzinfo=timezone.utc)


class TestScheduleLookupIsNeeded(SimpleTestCase):
    def _needed(self, record, portal_state, terminal=_TERMINAL, failed=_FAILED, recheck=False):
        # The keyword is only passed when set, so every default-path test also pins that it is optional.
        extra = {'recheck_unscheduled': True} if recheck else {}
        return _schedule_lookup_is_needed(record, portal_state, terminal, failed, **extra)

    def test_no_existing_record_needs_a_lookup(self):
        self.assertTrue(self._needed(None, 'COMPLETED'))

    def test_pending_record_needs_a_lookup(self):
        record = ObservationRecord(status='PENDING')
        self.assertTrue(self._needed(record, 'PENDING'))

    def test_completed_record_with_times_at_same_state_is_skipped(self):
        record = ObservationRecord(status='COMPLETED', scheduled_start=_START, scheduled_end=_END)
        self.assertFalse(self._needed(record, 'COMPLETED'))

    def test_state_change_needs_a_lookup(self):
        record = ObservationRecord(status='COMPLETED', scheduled_start=_START, scheduled_end=_END)
        self.assertTrue(self._needed(record, 'CANCELED'))

    def test_failed_state_with_no_times_is_skipped(self):
        record = ObservationRecord(status='WINDOW_EXPIRED')
        self.assertFalse(self._needed(record, 'WINDOW_EXPIRED'))

    def test_completed_record_missing_one_time_needs_a_lookup(self):
        record = ObservationRecord(status='COMPLETED', scheduled_start=_START, scheduled_end=None)
        self.assertTrue(self._needed(record, 'COMPLETED'))

    def test_failed_state_with_no_times_needs_a_lookup_when_rechecking(self):
        # An aborted block can sit under a request that expired or was cancelled, so the operator's
        # one-time catch-up looks at it once even though the per-tick skip does not.
        record = ObservationRecord(status='WINDOW_EXPIRED')
        self.assertFalse(self._needed(record, 'WINDOW_EXPIRED'))
        self.assertTrue(self._needed(record, 'WINDOW_EXPIRED', recheck=True))

    def test_failed_state_missing_one_time_needs_a_lookup_when_rechecking(self):
        record = ObservationRecord(status='CANCELED', scheduled_start=_START, scheduled_end=None)
        self.assertFalse(self._needed(record, 'CANCELED'))
        self.assertTrue(self._needed(record, 'CANCELED', recheck=True))

    def test_failed_state_with_both_times_needs_no_lookup_even_when_rechecking(self):
        record = ObservationRecord(status='WINDOW_EXPIRED', scheduled_start=_START, scheduled_end=_END)
        self.assertFalse(self._needed(record, 'WINDOW_EXPIRED', recheck=True))

    def test_recheck_changes_nothing_for_records_that_already_needed_a_lookup(self):
        missing_time = ObservationRecord(status='COMPLETED', scheduled_start=_START, scheduled_end=None)
        self.assertTrue(self._needed(missing_time, 'COMPLETED'))
        self.assertTrue(self._needed(missing_time, 'COMPLETED', recheck=True))
        pending = ObservationRecord(status='PENDING')
        self.assertTrue(self._needed(pending, 'PENDING'))
        self.assertTrue(self._needed(pending, 'PENDING', recheck=True))
        changed = ObservationRecord(status='COMPLETED', scheduled_start=_START, scheduled_end=_END)
        self.assertTrue(self._needed(changed, 'CANCELED'))
        self.assertTrue(self._needed(changed, 'CANCELED', recheck=True))
        self.assertTrue(self._needed(None, 'COMPLETED', recheck=True))

    def test_recheck_leaves_a_finished_record_with_both_times_skipped(self):
        record = ObservationRecord(status='COMPLETED', scheduled_start=_START, scheduled_end=_END)
        self.assertFalse(self._needed(record, 'COMPLETED', recheck=True))

    def test_function_trusts_the_state_lists_it_is_given(self):
        record = ObservationRecord(status='DONE')
        self.assertFalse(self._needed(record, 'DONE', terminal=frozenset({'DONE'}), failed=frozenset({'DONE'})))
        self.assertTrue(self._needed(record, 'DONE', terminal=frozenset(), failed=frozenset()))


class TestSweepSystemLinks(TestCase):
    """ALLOC-06: the discovery sweep links a record that exactly matches one approved run."""

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')
        cls.chilean_site = Observatory.objects.create(
            obscode='809',
            name='ESO, La Silla',
            short_name='NTT',
            lat=-29.2567,
            lon=-70.7300,
            altitude=2347,
            timezone='America/Santiago',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

    def setUp(self):
        patcher = patch('solsys_code.observation_blocks.FomoLCOFacility.get_observation_status')
        self.mock_get_observation_status = patcher.start()
        self.mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        self.addCleanup(patcher.stop)

    def _make_per_night_run(self, **overrides):
        kwargs = {
            'campaign': None,
            'target': self.existing_target,
            'proposal_code': 'LCO2026A-003',
            'source': CampaignRun.Source.CLASSICAL_FILE,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'telescope_instrument': 'NTT/EFOSC2',
            'site': self.chilean_site,
            'site_raw': '809',
            'window_start': date(2026, 6, 29),
            'window_end': date(2026, 7, 2),
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_exact_target_match_links_and_retires_the_night(self, mock_make_request):
        run = self._make_per_night_run()
        reconcile_run(run)
        events_before = allocation_events(run).count()
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])
        stdout, stderr = io.StringIO(), io.StringIO()

        summary = sweep_proposal('LCO2026A-003', stdout=stdout, stderr=stderr)

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        link = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link.run_id, run.pk)
        self.assertIsNone(link.confirmed_by)
        self.assertIsNotNone(link.confirmed_at)
        retired_night = observing_night(record.scheduled_start, ZoneInfo('America/Santiago'))
        self.assertFalse(allocation_events(run).filter(url=f'ALLOC:{run.pk}:{retired_night.isoformat()}').exists())
        self.assertEqual(allocation_events(run).count(), events_before - 1)
        self.assertNotIn(record.pk, {g.orphan.pk for g in record_attribution_backlog()})
        self.assertIn(
            f"System-linked ObservationRecord observation_id='10' to CampaignRun #{run.pk} "
            f'(proposal + target + window).',
            stdout.getvalue(),
        )
        self.assertEqual(stderr.getvalue(), '')
        expected = _expected_summary(
            dry_run=False,
            requestgroups_seen=1,
            created=1,
            updated=0,
            unchanged=0,
            skipped=0,
            targets=0,
            groups_created=0,
            groups_reused=0,
            embedded_blocks=0,
            fallback_lookups_needed=1,
            block_lookups_failed=0,
            list_reused=False,
            targets_added=1,
            system_links=1,
            links_skipped=0,
        )
        self.assertEqual(summary, expected)

    def _make_container_run(self, **overrides):
        """An APPROVED class-wide queue run (no site); the class is what lets the receiver resolve it."""
        kwargs = {
            'campaign': None,
            'target': self.existing_target,
            'proposal_code': 'LCO2026A-003',
            'source': CampaignRun.Source.LCO_QUEUE,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'telescope_instrument': '1m0/Sinistro',
            'telescope_class': '1m0',
            'window_start': date(2026, 6, 29),
            'window_end': date(2026, 7, 2),
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_over_an_existing_record_previews_the_link_and_writes_nothing(self, mock_make_request):
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])
        sweep_proposal('LCO2026A-003')
        run = self._make_per_night_run()
        links_before = CampaignRunObservation.objects.count()
        records_before = ObservationRecord.objects.count()
        stdout, stderr = io.StringIO(), io.StringIO()

        summary = sweep_proposal('LCO2026A-003', dry_run=True, stdout=stdout, stderr=stderr)

        self.assertIn(
            f"Would system-link ObservationRecord observation_id='10' to CampaignRun #{run.pk} "
            f'(proposal + target + window).',
            stdout.getvalue(),
        )
        self.assertTrue(summary.endswith('would link: 1, links skipped: 0'), summary)
        self.assertEqual(CampaignRunObservation.objects.count(), links_before)
        self.assertEqual(ObservationRecord.objects.count(), records_before)
        self.assertEqual(stderr.getvalue(), '')

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_over_a_new_request_with_an_existing_target_previews_the_link(self, mock_make_request):
        run = self._make_per_night_run()
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])
        stdout = io.StringIO()

        summary = sweep_proposal('LCO2026A-003', dry_run=True, stdout=stdout, stderr=io.StringIO())

        self.assertIn(
            f"Would system-link ObservationRecord observation_id='10' to CampaignRun #{run.pk} "
            f'(proposal + target + window).',
            stdout.getvalue(),
        )
        self.assertTrue(summary.endswith('would link: 1, links skipped: 0'), summary)
        self.assertFalse(ObservationRecord.objects.exists())
        self.assertFalse(CampaignRunObservation.objects.exists())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_dry_run_over_a_would_be_new_target_asks_nothing(self, mock_make_request):
        self._make_per_night_run()
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Brand New Comet - ELP', requests=[_request(10, target_name='Brand New Comet')])]
        )
        stdout = io.StringIO()

        summary = sweep_proposal('LCO2026A-003', dry_run=True, stdout=stdout, stderr=io.StringIO())

        self.assertNotIn('Would system-link', stdout.getvalue())
        self.assertTrue(summary.endswith('would link: 0, links skipped: 0'), summary)
        self.assertFalse(Target.objects.filter(name='Brand New Comet').exists())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_container_run_adopts_the_records_own_event(self, mock_make_request):
        run = self._make_container_run()
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        summary = sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        link = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link.run_id, run.pk)
        self.assertIsNone(link.confirmed_by)
        meta = CalendarEventMeta.objects.get(observation_record=record)
        self.assertEqual(meta.run_id, run.pk)
        self.assertIsNone(meta.confirmed_by)
        self.assertTrue(summary.endswith('system links: 1, links skipped: 0'), summary)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_a_second_sweep_writes_no_second_row_and_leaves_confirmed_at_alone(self, mock_make_request):
        self._make_per_night_run()
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])
        sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        first = CampaignRunObservation.objects.get()
        stdout = io.StringIO()

        summary = sweep_proposal('LCO2026A-003', stdout=stdout, stderr=io.StringIO())

        self.assertEqual(CampaignRunObservation.objects.count(), 1)
        self.assertEqual(CampaignRunObservation.objects.get().confirmed_at, first.confirmed_at)
        self.assertTrue(summary.endswith('system links: 0, links skipped: 0'), summary)
        self.assertNotIn('System-linked', stdout.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_an_unchanged_record_is_linked_once_its_run_exists(self, mock_make_request):
        """D-01: runs are often created after their records."""
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])
        first_summary = sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())
        self.assertTrue(first_summary.endswith('system links: 0, links skipped: 0'), first_summary)
        self.assertFalse(CampaignRunObservation.objects.exists())
        run = self._make_per_night_run()

        summary = sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertIn('unchanged: 1', summary)
        self.assertTrue(summary.endswith('system links: 1, links skipped: 0'), summary)
        self.assertEqual(CampaignRunObservation.objects.get().run_id, run.pk)

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_a_failing_link_write_is_counted_and_the_sweep_continues(self, mock_make_request):
        """D-08: the first write raises, the second record still links, the sweep returns."""
        self._make_per_night_run()
        mock_make_request.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10), _request(20)])]
        )
        calls = []

        def flaky_create_system_link(record, run):
            calls.append(record.observation_id)
            if len(calls) == 1:
                raise IntegrityError('boom')
            return create_system_link(record, run)

        stderr = io.StringIO()
        with patch('solsys_code.campaign_system_links.create_system_link', side_effect=flaky_create_system_link):
            summary = sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=stderr)

        first = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        second = ObservationRecord.objects.get(facility='LCO', observation_id='20')
        self.assertFalse(CampaignRunObservation.objects.filter(observation_record=first).exists())
        self.assertTrue(CampaignRunObservation.objects.filter(observation_record=second).exists())
        self.assertTrue(summary.endswith('system links: 1, links skipped: 1'), summary)
        self.assertIn('IntegrityError', stderr.getvalue())
        self.assertNotIn('boom', stderr.getvalue())

    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_bare_invocation_records_the_link_count_on_the_watched_row(self, mock_make_request):
        self._make_per_night_run()
        row = WatchedProposal.objects.create(proposal_code='LCO2026A-003')
        mock_make_request.side_effect = _watched_side_effect(
            ('LCO2026A-003', {'group_id': 1, 'name': 'Didymos 2026 - ELP', 'requests': [_request(10)]}),
        )

        call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        row.refresh_from_db()
        self.assertTrue(row.last_run_summary.endswith('system links: 1, links skipped: 0'), row.last_run_summary)


_ABORTED_BLOCK = {'state': 'ABORTED', 'start': '2026-07-01T01:00:00Z', 'end': '2026-07-01T01:40:00Z'}
_LISTING = 'solsys_code.management.commands.backfill_lco_observations.make_request'
_PORTAL = 'solsys_code.observation_blocks.make_request'


class TestDiscoveryUsesFomoBlockRule(TestCase):
    """G-37.1-1-alloc: a new request whose block was aborted is stored with that block's times.

    Nothing here patches get_observation_status, so FOMO's own block rule runs for both the embedded
    block list and the live lookup.
    """

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')

    @patch(_LISTING)
    def test_an_embedded_aborted_block_gives_the_record_its_times(self, mock_listing):
        mock_listing.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - ELP',
                    requests=[_request(10, state='WINDOW_EXPIRED', observations=[dict(_ABORTED_BLOCK)])],
                )
            ]
        )

        summary = sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.status, 'WINDOW_EXPIRED')
        self.assertEqual(record.scheduled_start, datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc))
        self.assertEqual(record.scheduled_end, datetime(2026, 7, 1, 1, 40, tzinfo=timezone.utc))
        self.assertIn('embedded blocks: 1', summary)

    @patch(_LISTING)
    def test_an_embedded_block_list_with_only_a_cancelled_block_gives_no_times(self, mock_listing):
        mock_listing.return_value = _page_response(
            [
                _request_group(
                    1,
                    'Didymos 2026 - ELP',
                    requests=[
                        _request(
                            10,
                            state='CANCELED',
                            observations=[{'state': 'CANCELED', 'start': _ABORTED_BLOCK['start'], 'end': None}],
                        )
                    ],
                )
            ]
        )

        sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertIsNone(record.scheduled_start)
        self.assertIsNone(record.scheduled_end)

    @patch(_PORTAL)
    @patch(_LISTING)
    def test_the_live_lookup_for_a_new_expired_request_uses_the_aborted_block(self, mock_listing, mock_portal):
        mock_listing.return_value = _page_response(
            [_request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='WINDOW_EXPIRED')])]
        )
        mock_portal.side_effect = portal_side_effect({'10': 'WINDOW_EXPIRED'}, {'10': [dict(_ABORTED_BLOCK)]})

        summary = sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        record = ObservationRecord.objects.get(facility='LCO', observation_id='10')
        self.assertEqual(record.scheduled_start, datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc))
        self.assertEqual(record.scheduled_end, datetime(2026, 7, 1, 1, 40, tzinfo=timezone.utc))
        self.assertEqual(mock_portal.call_count, 2)
        self.assertIn('fallback lookups needed: 1', summary)


class TestRecheckUnscheduledSweep(TestCase):
    """G-37.1-1-alloc: --recheck-unscheduled through sweep_proposal() re-resolves records stored under
    TOM's old rule, retires a linked run's night in the same sweep, and leaves never-scheduled requests alone."""

    @classmethod
    def setUpTestData(cls):
        cls.existing_target = NonSiderealTargetFactory.create(name='Didymos')
        cls.chilean_site = Observatory.objects.create(
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
            target=cls.existing_target,
            proposal_code='LCO2026A-003',
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            site=cls.chilean_site,
            site_raw='809',
            window_start=date(2026, 6, 29),
            window_end=date(2026, 7, 2),
        )

    def setUp(self):
        reconcile_run(self.per_night_run)
        self.alloc_before = allocation_events(self.per_night_run).count()

    def _stored_record(self, request_id, state):
        """A record as the sweep stored it under TOM's old rule: finished, linked, and without times."""
        request = _request(request_id, state=state)
        group = _request_group(1, 'Didymos 2026 - ELP', requests=[request])
        record = ObservationRecord.objects.create(
            target=self.existing_target,
            facility='LCO',
            observation_id=str(request_id),
            status=state,
            parameters=_build_parameters(group, request),
        )
        create_system_link(record, self.per_night_run)
        return group, record

    def _alloc_url(self, record):
        night = observing_night(record.scheduled_start, ZoneInfo('America/Santiago'))
        return f'ALLOC:{self.per_night_run.pk}:{night.isoformat()}'

    @patch(_PORTAL)
    @patch(_LISTING)
    def test_recheck_gives_the_aborted_block_its_times_and_retires_its_night(self, mock_listing, mock_portal):
        group, record = self._stored_record(10, 'WINDOW_EXPIRED')
        mock_listing.return_value = _page_response([group])
        mock_portal.side_effect = portal_side_effect({'10': 'WINDOW_EXPIRED'}, {'10': [dict(_ABORTED_BLOCK)]})

        summary = sweep_proposal('LCO2026A-003', recheck_unscheduled=True, stdout=io.StringIO(), stderr=io.StringIO())

        record.refresh_from_db()
        self.assertEqual(record.scheduled_start, datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc))
        self.assertEqual(record.scheduled_end, datetime(2026, 7, 1, 1, 40, tzinfo=timezone.utc))
        self.assertFalse(allocation_events(self.per_night_run).filter(url=self._alloc_url(record)).exists())
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before - 1)
        self.assertEqual(mock_portal.call_count, 2)
        self.assertIn('updated: 1', summary)
        self.assertIn('fallback lookups needed: 1', summary)
        self.assertIn('fallback lookups skipped: 0', summary)

    @patch(_PORTAL)
    @patch(_LISTING)
    def test_without_the_flag_the_finished_record_is_not_looked_up(self, mock_listing, mock_portal):
        group, record = self._stored_record(10, 'WINDOW_EXPIRED')
        mock_listing.return_value = _page_response([group])
        mock_portal.side_effect = portal_side_effect({'10': 'WINDOW_EXPIRED'}, {'10': [dict(_ABORTED_BLOCK)]})

        summary = sweep_proposal('LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO())

        mock_portal.assert_not_called()
        record.refresh_from_db()
        self.assertIsNone(record.scheduled_start)
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before)
        self.assertIn('unchanged: 1', summary)
        self.assertIn('fallback lookups skipped: 1', summary)
        self.assertIn('fallback lookups needed: 0', summary)

    @patch(_PORTAL)
    @patch(_LISTING)
    def test_a_never_scheduled_request_keeps_its_night_when_rechecked(self, mock_listing, mock_portal):
        group, record = self._stored_record(20, 'CANCELED')
        mock_listing.return_value = _page_response([group])
        mock_portal.side_effect = portal_side_effect({'20': 'CANCELED'}, {'20': []})

        summary = sweep_proposal('LCO2026A-003', recheck_unscheduled=True, stdout=io.StringIO(), stderr=io.StringIO())

        record.refresh_from_db()
        self.assertIsNone(record.scheduled_start)
        self.assertIsNone(record.scheduled_end)
        self.assertEqual(allocation_events(self.per_night_run).count(), self.alloc_before)
        self.assertEqual(mock_portal.call_count, 2)
        self.assertIn('unchanged: 1', summary)
        self.assertIn('updated: 0', summary)
        self.assertIn('fallback lookups needed: 1', summary)

    @patch(_PORTAL)
    @patch(_LISTING)
    def test_a_failed_lookup_is_counted_and_never_writes_the_error_text(self, mock_listing, mock_portal):
        group, record = self._stored_record(10, 'WINDOW_EXPIRED')
        mock_listing.return_value = _page_response([group])
        mock_portal.side_effect = requests.exceptions.ConnectionError('secret-token-in-the-url')
        stdout, stderr = io.StringIO(), io.StringIO()

        summary = sweep_proposal('LCO2026A-003', recheck_unscheduled=True, stdout=stdout, stderr=stderr)

        record.refresh_from_db()
        self.assertIsNone(record.scheduled_start)
        self.assertIn('block lookups failed: 1', summary)
        self.assertIn("Failed to resolve observed block for observation_id='10'.", stderr.getvalue())
        self.assertNotIn('secret-token-in-the-url', stderr.getvalue() + stdout.getvalue() + summary)


class TestRecheckUnscheduledCommand(TestCase):
    """The flag reaches sweep_proposal() from both invocation forms and is not one of the proposal-only flags."""

    SWEEP = 'solsys_code.management.commands.backfill_lco_observations.sweep_proposal'

    @patch(SWEEP)
    def test_proposal_form_passes_the_flag_through(self, mock_sweep):
        mock_sweep.return_value = 'summary line'

        call_command(
            'backfill_lco_observations',
            '--proposal',
            'LCO2026A-003',
            '--recheck-unscheduled',
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertIs(mock_sweep.call_args.kwargs['recheck_unscheduled'], True)

    @patch(SWEEP)
    def test_proposal_form_without_the_flag_passes_false(self, mock_sweep):
        mock_sweep.return_value = 'summary line'

        call_command(
            'backfill_lco_observations', '--proposal', 'LCO2026A-003', stdout=io.StringIO(), stderr=io.StringIO()
        )

        self.assertIs(mock_sweep.call_args.kwargs['recheck_unscheduled'], False)

    @patch(SWEEP)
    def test_bare_form_passes_the_flag_for_every_watched_row(self, mock_sweep):
        mock_sweep.return_value = 'summary line'
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')

        call_command('backfill_lco_observations', '--recheck-unscheduled', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(mock_sweep.call_count, 1)
        self.assertEqual(mock_sweep.call_args.args[0], 'AAA-2026-001')
        self.assertIs(mock_sweep.call_args.kwargs['recheck_unscheduled'], True)

    @patch(SWEEP)
    def test_bare_form_without_the_flag_passes_false(self, mock_sweep):
        mock_sweep.return_value = 'summary line'
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')

        call_command('backfill_lco_observations', stdout=io.StringIO(), stderr=io.StringIO())

        self.assertIs(mock_sweep.call_args.kwargs['recheck_unscheduled'], False)

    @patch(SWEEP)
    def test_bare_form_still_rejects_the_proposal_only_flags(self, mock_sweep):
        for flag, value in (
            ('--created-after', '2026-01-01'),
            ('--created-before', '2026-12-31'),
            ('--username', 'someone'),
            ('--target-list', 'My list'),
        ):
            with self.subTest(flag=flag):
                with self.assertRaises(CommandError) as ctx:
                    call_command(
                        'backfill_lco_observations',
                        '--recheck-unscheduled',
                        flag,
                        value,
                        stdout=io.StringIO(),
                        stderr=io.StringIO(),
                    )
                self.assertIn(flag, str(ctx.exception))
                self.assertNotIn('--recheck-unscheduled', str(ctx.exception))
        mock_sweep.assert_not_called()
