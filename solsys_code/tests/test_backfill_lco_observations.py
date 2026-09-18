import io
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth.models import User
from django.core.management import CommandError, call_command
from django.test import TestCase
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import Target, TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.management.commands.backfill_lco_observations import sweep_proposal
from solsys_code.models import WatchedProposal

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
    list_name='LCO2026A-003_targets',
    list_reused=False,
    targets_added=0,
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
    return (
        f'requestgroups seen: {requestgroups_seen}, '
        f'{created_label}: {created}, '
        f'{updated_label}: {updated}, '
        f'unchanged: {unchanged}, skipped: {skipped}, '
        f'{targets_label}: {targets}, '
        f'{groups_created_label}: {groups_created}, '
        f'{groups_reused_label}: {groups_reused}, '
        f'embedded blocks: {embedded_blocks}, fallback lookups needed: {fallback_lookups_needed}, '
        f'block lookups failed: {block_lookups_failed_value}, '
        f'target list: {list_verb} {list_name!r}, '
        f'{targets_added_label}: {targets_added}'
    )


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
        request_group = _request_group(1, 'Didymos 2026 - ELP', requests=[_request(10, state='COMPLETED')])
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
            fallback_lookups_needed=2,
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
            'embedded blocks: 0, fallback lookups needed: 1, block lookups failed: n/a (dry-run), '
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
            fallback_lookups_needed=2,
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
            fallback_lookups_needed=1,
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
        patcher = patch('tom_observations.facilities.lco.LCOFacility.get_observation_status')
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
        patcher = patch('tom_observations.facilities.lco.LCOFacility.get_observation_status')
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
        patcher = patch('tom_observations.facilities.lco.LCOFacility.get_observation_status')
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
