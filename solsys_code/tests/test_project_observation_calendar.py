"""project_observation_calendar: the TRIG-03 backstop sweep -- zero required arguments,
--dry-run, per-record failure isolation, and the no-churn idempotency contract (D-17).

Uses ``tom_targets.tests.factories.NonSiderealTargetFactory`` for every target fixture --
FOMO is exclusively a Solar System TOM, so a sidereal fixture would misrepresent what this
code handles (CLAUDE.md convention). No test performs a real HTTP call.
"""

import re
from datetime import datetime
from datetime import timezone as dt_timezone
from io import StringIO
from unittest.mock import patch

import requests
from django.core.management import CommandError, call_command
from django.db.models.signals import post_save
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import observation_projector as op
from solsys_code.management.commands.project_observation_calendar import _parse_proposal_arg, resolve_observed_site
from solsys_code.tests.helpers import observations_block_response


def _parse_summary(output: str) -> dict[str, dict[str, int]]:
    """Parse the command's ``Done[...]. failed: N | LCO: key: N, ... | SOAR: key: N, ...``
    summary line into a per-facility dict of counters (the ``failed`` scalar is folded into
    every facility dict under its own 'failed' key for convenience)."""
    lines = [line for line in output.strip().splitlines() if line.startswith('Done')]
    assert lines, f'no summary line found in output: {output!r}'
    body = re.sub(r'^Done(?:\s*\(dry run\))?\.\s*', '', lines[-1])
    segments = body.split(' | ')
    failed_segment, facility_segments = segments[0], segments[1:]
    failed = int(re.search(r'failed:\s*(-?\d+)', failed_segment).group(1))

    result: dict[str, dict[str, int]] = {}
    for segment in facility_segments:
        facility, _, rest = segment.partition(': ')
        counters = {key: int(value) for key, value in re.findall(r'(\w+):\s*(-?\d+)', rest)}
        counters['failed'] = failed
        result[facility.strip()] = counters
    return result


class _ProjectObservationCalendarTestBase(TestCase):
    """Shared fixture: one NonSiderealTargetFactory target and a record-builder helper."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target = NonSiderealTargetFactory.create()

    def setUp(self) -> None:
        op.reset_facility_cache()

    def _make_record(
        self,
        observation_id: str,
        status: str = 'PENDING',
        scheduled_start: datetime | None = None,
        scheduled_end: datetime | None = None,
        facility: str = 'LCO',
        proposal: str = 'TESTPROP',
        start: str | None = '2026-09-01T00:00:00',
        end: str | None = '2026-09-02T00:00:00',
        instrument_type: str = '2M0-SCICAM-MUSCAT',
    ) -> ObservationRecord:
        """Create an ObservationRecord fixture sharing the class-level target.

        This module exercises the sweep command's own effect on a queryset, so fixture
        creation must not itself pre-populate the CalendarEvent this command is meant to
        write -- the projector's post_save receiver is wired globally (34-01), so disconnect
        it around fixture creation, mirroring the retired sync command's own test module.
        """
        params: dict = {'proposal': proposal, 'instrument_type': instrument_type}
        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end
        post_save.disconnect(
            op.receiver_on_record_save,
            sender=ObservationRecord,
            dispatch_uid='solsys_code.observation_projector.post_save',
        )
        try:
            return ObservationRecord.objects.create(
                target=self.target,
                facility=facility,
                observation_id=observation_id,
                status=status,
                scheduled_start=scheduled_start,
                scheduled_end=scheduled_end,
                parameters=params,
            )
        finally:
            post_save.connect(
                op.receiver_on_record_save,
                sender=ObservationRecord,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.post_save',
            )


class TestBareInvocationAndSummary(_ProjectObservationCalendarTestBase):
    def test_bare_invocation_projects_every_record_and_prints_full_summary(self) -> None:
        self._make_record('bare-lco', facility='LCO')
        self._make_record('bare-soar', facility='SOAR', instrument_type='SOAR_GHTS_REDCAM')

        out = StringIO()
        call_command('project_observation_calendar', stdout=out, stderr=StringIO())
        output = out.getvalue()

        for label in ('created:', 'updated:', 'unchanged:', 'unprojectable:', 'site_lookups:', 'site_lookup_failed:'):
            self.assertIn(label, output)
        self.assertEqual(CalendarEvent.objects.count(), 2)

    def test_second_sweep_over_unchanged_data_reports_created_zero_updated_zero(self) -> None:
        self._make_record('nochurn-lco', facility='LCO')
        self._make_record('nochurn-soar', facility='SOAR', instrument_type='SOAR_GHTS_REDCAM')

        call_command('project_observation_calendar', stdout=StringIO(), stderr=StringIO())
        second_out = StringIO()
        call_command('project_observation_calendar', stdout=second_out, stderr=StringIO())

        summary = _parse_summary(second_out.getvalue())
        for facility_counters in summary.values():
            self.assertEqual(facility_counters['created'], 0)
            self.assertEqual(facility_counters['updated'], 0)

    def test_empty_record_set_reports_every_counter_zero(self) -> None:
        out = StringIO()
        call_command('project_observation_calendar', stdout=out, stderr=StringIO())
        summary = _parse_summary(out.getvalue())
        self.assertEqual(set(summary), {'LCO', 'SOAR'})
        for facility_counters in summary.values():
            for key, value in facility_counters.items():
                self.assertEqual(value, 0, f'{key} was {value}, expected 0')


class TestProposalAndFacilityFiltering(_ProjectObservationCalendarTestBase):
    def test_proposal_filter_matches_exact_code_only(self) -> None:
        self._make_record('prop-match', proposal='A')
        self._make_record('prop-decoy', proposal='AB')

        call_command('project_observation_calendar', '--proposal', 'A', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(CalendarEvent.objects.count(), 1)
        facility = op.facility_for(ObservationRecord.objects.get(observation_id='prop-match'))
        matched_url = facility.get_observation_url('prop-match')
        self.assertTrue(CalendarEvent.objects.filter(url=matched_url).exists())

    def test_parse_proposal_arg_dedupes_and_drops_trailing_empty_segment(self) -> None:
        self.assertEqual(_parse_proposal_arg('A,B,A,'), ['A', 'B'])

    def test_facility_filter_projects_only_that_facility(self) -> None:
        self._make_record('facility-lco', facility='LCO')
        self._make_record('facility-soar', facility='SOAR', instrument_type='SOAR_GHTS_REDCAM')

        call_command('project_observation_calendar', '--facility', 'SOAR', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(CalendarEvent.objects.count(), 1)
        soar_facility = op.facility_for(ObservationRecord.objects.get(observation_id='facility-soar'))
        self.assertTrue(CalendarEvent.objects.filter(url=soar_facility.get_observation_url('facility-soar')).exists())

    def test_facility_gem_is_rejected_by_argparse(self) -> None:
        with self.assertRaises(CommandError):
            call_command('project_observation_calendar', '--facility', 'GEM', stdout=StringIO(), stderr=StringIO())

    def test_comma_list_matches_any_no_substring_leakage(self) -> None:
        """Migrated from the retired sync command's SELECT-02 test (34-02 Task 2)."""
        self._make_record('comma-a', proposal='A')
        self._make_record('comma-b', proposal='B')
        self._make_record('comma-c', proposal='C')
        self._make_record('comma-decoy', proposal='AB')

        call_command('project_observation_calendar', '--proposal', 'A,B,C', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(CalendarEvent.objects.count(), 3)
        decoy_facility = op.facility_for(ObservationRecord.objects.get(observation_id='comma-decoy'))
        self.assertFalse(CalendarEvent.objects.filter(url=decoy_facility.get_observation_url('comma-decoy')).exists())

    def test_proposal_with_only_empty_segments_raises_instead_of_sweeping_everything(self) -> None:
        """WR-04: --proposal ',,' parses to no usable code -- this must fail closed
        (CommandError) rather than silently widening to the full unfiltered corpus, which
        is the opposite of what an operator naming --proposal at all is asking for."""
        self._make_record('proposal-guard-a', proposal='A')
        self._make_record('proposal-guard-b', proposal='B')

        with self.assertRaises(CommandError):
            call_command('project_observation_calendar', '--proposal', ',,', stdout=StringIO(), stderr=StringIO())

        # Fails before writing anything -- the guard raises before the queryset is swept.
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_zero_matching_proposal_reports_created_zero_no_error(self) -> None:
        """Migrated from the retired sync command's zero-match test (34-02 Task 2)."""
        self._make_record('zero-match', proposal='SOMEOTHERCODE')

        out = StringIO()
        call_command('project_observation_calendar', '--proposal', 'NOMATCHCODE', stdout=out, stderr=StringIO())

        self.assertEqual(CalendarEvent.objects.count(), 0)
        summary = _parse_summary(out.getvalue())
        for facility_counters in summary.values():
            self.assertEqual(facility_counters['created'], 0)


class TestDryRun(_ProjectObservationCalendarTestBase):
    def test_dry_run_writes_nothing_and_matches_the_subsequent_real_run(self) -> None:
        self._make_record('dry-a', facility='LCO')
        self._make_record('dry-b', facility='SOAR', instrument_type='SOAR_GHTS_REDCAM')
        record_count_before = ObservationRecord.objects.count()

        dry_out = StringIO()
        call_command('project_observation_calendar', '--dry-run', stdout=dry_out, stderr=StringIO())
        dry_summary = _parse_summary(dry_out.getvalue())

        self.assertEqual(CalendarEvent.objects.count(), 0)
        self.assertEqual(ObservationRecord.objects.count(), record_count_before)

        real_out = StringIO()
        call_command('project_observation_calendar', stdout=real_out, stderr=StringIO())
        real_summary = _parse_summary(real_out.getvalue())

        for facility in ('LCO', 'SOAR'):
            self.assertEqual(dry_summary[facility]['created'], real_summary[facility]['created'])


class TestFailureIsolation(_ProjectObservationCalendarTestBase):
    def test_one_raising_record_is_reported_and_isolated_command_exits_0(self) -> None:
        good = self._make_record('fail-good')
        bad = self._make_record('fail-bad')
        original = op.event_fields_for

        def flaky(record, facility):
            if record.observation_id == bad.observation_id:
                raise RuntimeError('unexpected failure')
            return original(record, facility)

        err = StringIO()
        with patch('solsys_code.observation_projector.event_fields_for', side_effect=flaky):
            call_command('project_observation_calendar', stdout=StringIO(), stderr=err)

        self.assertIn('fail-bad', err.getvalue())
        good_facility = op.facility_for(good)
        self.assertTrue(CalendarEvent.objects.filter(url=good_facility.get_observation_url('fail-good')).exists())

    def test_missing_window_is_unprojectable_and_leaves_preexisting_event_untouched(self) -> None:
        record = self._make_record('missing-window', start=None, end=None)
        facility = op.facility_for(record)
        url = facility.get_observation_url('missing-window')
        stale_event = CalendarEvent.objects.create(
            url=url,
            title='pre-existing',
            description='untouched',
            start_time=datetime(2020, 1, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2020, 1, 2, tzinfo=dt_timezone.utc),
        )
        before = (stale_event.title, stale_event.description, stale_event.start_time, stale_event.end_time)

        err = StringIO()
        call_command('project_observation_calendar', stdout=StringIO(), stderr=err)

        stale_event.refresh_from_db()
        after = (stale_event.title, stale_event.description, stale_event.start_time, stale_event.end_time)
        self.assertEqual(before, after)
        self.assertIn('missing-window', err.getvalue())

    def test_write_failure_is_counted_unprojectable_not_updated(self) -> None:
        """CR-01: project_queryset() must count what project_record()'s write actually did,
        not what the preview predicted. A pre-existing duplicate-url CalendarEvent pair
        makes CalendarEvent.objects.get_or_create(url=...) raise MultipleObjectsReturned
        inside project_record() -- before the fix, project_queryset() discarded that
        outcome, counted the preview's 'updated' guess, and reported unprojectable: 0,
        failed: 0 for a record whose event was never touched."""
        record = self._make_record('sweep-dup-url')
        facility = op.facility_for(record)
        url = facility.get_observation_url('sweep-dup-url')
        for i in range(2):
            CalendarEvent.objects.create(
                url=url,
                title=f'dup {i}',
                start_time=datetime(2020, 1, 1, tzinfo=dt_timezone.utc),
                end_time=datetime(2020, 1, 2, tzinfo=dt_timezone.utc),
            )

        result = op.project_queryset(ObservationRecord.objects.filter(pk=record.pk))

        self.assertEqual(result['counters']['LCO']['unprojectable'], 1)
        self.assertEqual(result['counters']['LCO']['updated'], 0)
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(result['rows'][0]['action'], 'unprojectable')
        titles = set(CalendarEvent.objects.filter(url=url).values_list('title', flat=True))
        self.assertEqual(titles, {'dup 0', 'dup 1'})  # the write never touched either row

        out = StringIO()
        err = StringIO()
        call_command('project_observation_calendar', stdout=out, stderr=err)
        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['LCO']['failed'], 1)
        self.assertEqual(summary['LCO']['unprojectable'], 1)
        self.assertIn('sweep-dup-url', err.getvalue())

    def test_dry_run_agrees_with_the_real_run_on_a_duplicate_url_failure(self) -> None:
        """A duplicate-url CalendarEvent pair is a write failure --dry-run CAN see without
        writing: the real run's get_or_create() is certain to raise
        MultipleObjectsReturned. A --dry-run pass over this fixture must report
        unprojectable: 1 / failed: 1, the same outcome the real run that follows it
        reports -- not the preview's 'updated' guess, which is what a dry run without this
        detection would (wrongly) predict."""
        record = self._make_record('dry-dup-url')
        facility = op.facility_for(record)
        url = facility.get_observation_url('dry-dup-url')
        for i in range(2):
            CalendarEvent.objects.create(
                url=url,
                title=f'dup {i}',
                start_time=datetime(2020, 1, 1, tzinfo=dt_timezone.utc),
                end_time=datetime(2020, 1, 2, tzinfo=dt_timezone.utc),
            )

        dry_result = op.project_queryset(ObservationRecord.objects.filter(pk=record.pk), dry_run=True)

        self.assertEqual(dry_result['counters']['LCO']['unprojectable'], 1)
        self.assertEqual(dry_result['counters']['LCO']['updated'], 0)
        self.assertEqual(len(dry_result['rows']), 1)
        self.assertEqual(dry_result['rows'][0]['action'], 'unprojectable')
        titles_after_dry_run = set(CalendarEvent.objects.filter(url=url).values_list('title', flat=True))
        self.assertEqual(titles_after_dry_run, {'dup 0', 'dup 1'})  # dry run wrote nothing

        out = StringIO()
        err = StringIO()
        call_command('project_observation_calendar', '--dry-run', stdout=out, stderr=err)
        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['LCO']['failed'], 1)
        self.assertEqual(summary['LCO']['unprojectable'], 1)
        self.assertIn('dry-dup-url', err.getvalue())

        # The real run over the same fixture must agree with what --dry-run just reported.
        real_result = op.project_queryset(ObservationRecord.objects.filter(pk=record.pk))
        self.assertEqual(real_result['counters']['LCO']['unprojectable'], 1)
        self.assertEqual(real_result['rows'][0]['action'], dry_result['rows'][0]['action'])


class TestNamespaceIsolation(_ProjectObservationCalendarTestBase):
    def test_run_gem_and_blank_url_events_are_untouched_by_a_sweep(self) -> None:
        run_event = CalendarEvent.objects.create(
            url='RUN:1:2026-09-01',
            title='[CANCELLED] classical run',
            description='reconciler-owned',
            start_time=datetime(2026, 9, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, tzinfo=dt_timezone.utc),
            telescope='FTN',
            instrument='muscat',
        )
        gem_event = CalendarEvent.objects.create(
            url='GEM:gemini-obs-1',
            title='Gemini submission echo',
            description='gemini-owned',
            start_time=datetime(2026, 9, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, tzinfo=dt_timezone.utc),
            telescope='GN',
            instrument='GMOS',
        )
        blank_url_event = CalendarEvent.objects.create(
            url='',
            title='classical schedule night',
            description='load_telescope_runs-owned',
            start_time=datetime(2026, 9, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, tzinfo=dt_timezone.utc),
            telescope='FTS',
            instrument='',
        )
        snapshot = {
            event.pk: (
                event.url,
                event.title,
                event.description,
                event.start_time,
                event.end_time,
                event.telescope,
                event.instrument,
            )
            for event in (run_event, gem_event, blank_url_event)
        }

        self._make_record('namespace-isolation-record')
        call_command('project_observation_calendar', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(CalendarEvent.objects.count(), 4)
        for event in (run_event, gem_event, blank_url_event):
            event.refresh_from_db()
            after = (
                event.url,
                event.title,
                event.description,
                event.start_time,
                event.end_time,
                event.telescope,
                event.instrument,
            )
            self.assertEqual(snapshot[event.pk], after)


class TestProjectQuerysetOrdering(_ProjectObservationCalendarTestBase):
    """project_queryset()'s own row order -- tested directly rather than through the command,
    since the command does not print a per-row line for every record."""

    def test_rows_are_returned_in_primary_key_order(self) -> None:
        third = self._make_record('order-c')
        first = self._make_record('order-a')
        second = self._make_record('order-b')
        self.assertLess(third.pk, first.pk)
        self.assertLess(first.pk, second.pk)

        records = ObservationRecord.objects.filter(pk__in=[third.pk, first.pk, second.pk])
        result = op.project_queryset(records)

        self.assertEqual(
            [row['observation_id'] for row in result['rows']],
            ['order-c', 'order-a', 'order-b'],
        )


class TestObservedSiteLookup(_ProjectObservationCalendarTestBase):
    """resolve_observed_site()/the sweep's pre_fields_hook wiring -- the D-07/D-08 one-time
    observed-site lookup. No test performs a real HTTP call."""

    def _make_observed_record(
        self, observation_id: str, facility: str = 'LCO', instrument_type: str = '2M0-SCICAM-MUSCAT'
    ) -> ObservationRecord:
        """Create a COMPLETED, with-block record via a NORMAL save (receiver connected).

        Unlike ``_make_record()``, this deliberately does NOT disconnect the post_save
        receiver: the sweep's own site-lookup tests need the base event to already exist
        (with the coarse token, from this creation save) before the sweep runs, so that the
        sweep's own action is 'updated' -- the title/telescope actually changing -- rather
        than 'created', which would be the case for a record whose FIRST-ever event write is
        the sweep itself (per D-08's "that same first sweep counts the record updated").
        """
        return ObservationRecord.objects.create(
            target=self.target,
            facility=facility,
            observation_id=observation_id,
            status='COMPLETED',
            scheduled_start=datetime(2026, 9, 6, 10, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 6, 10, 19, tzinfo=dt_timezone.utc),
            parameters={
                'proposal': 'TESTPROP',
                'instrument_type': instrument_type,
                'start': '2026-09-01T00:00:00',
                'end': '2026-09-02T00:00:00',
            },
        )

    def test_first_sweep_resolves_site_once_and_counts_updated_and_site_lookups(self) -> None:
        record = self._make_observed_record('site-first')

        with patch(
            'solsys_code.calendar_utils.make_request',
            return_value=observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
        ) as mocked:
            out = StringIO()
            call_command('project_observation_calendar', stdout=out, stderr=StringIO())
            self.assertEqual(mocked.call_count, 1)

        record.refresh_from_db()
        self.assertEqual(record.parameters['observed_site'], 'coj')
        self.assertEqual(record.parameters['observed_telescope'], '2m0a')
        self.assertEqual(record.parameters['observed_enclosure'], 'doma')

        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['LCO']['site_lookups'], 1)
        self.assertEqual(summary['LCO']['updated'], 1)
        self.assertEqual(summary['LCO']['created'], 0)

        facility = op.facility_for(record)
        event = CalendarEvent.objects.get(url=facility.get_observation_url('site-first'))
        self.assertEqual(event.telescope, 'FTS')
        self.assertTrue(event.title.startswith('[O] FTS '))

    def test_second_sweep_issues_no_portal_call_and_reports_unchanged(self) -> None:
        self._make_observed_record('site-second')
        with patch(
            'solsys_code.calendar_utils.make_request',
            return_value=observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
        ):
            call_command('project_observation_calendar', stdout=StringIO(), stderr=StringIO())

        with patch('solsys_code.calendar_utils.make_request') as mocked:
            out = StringIO()
            call_command('project_observation_calendar', stdout=out, stderr=StringIO())
            mocked.assert_not_called()

        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['LCO']['site_lookups'], 0)
        self.assertEqual(summary['LCO']['unchanged'], 1)
        self.assertEqual(summary['LCO']['updated'], 0)

    def test_none_block_and_unmapped_pair_both_count_site_lookup_failed_and_are_retried(self) -> None:
        none_block_record = self._make_observed_record('site-none-block')
        unmapped_record = self._make_observed_record('site-unmapped')

        with patch(
            'solsys_code.calendar_utils.make_request',
            side_effect=[
                requests.exceptions.Timeout,
                observations_block_response(site='zzz', telescope='9x9x', state='COMPLETED'),
            ],
        ):
            out = StringIO()
            call_command('project_observation_calendar', stdout=out, stderr=StringIO())

        none_block_record.refresh_from_db()
        unmapped_record.refresh_from_db()
        self.assertNotIn('observed_site', none_block_record.parameters)
        self.assertNotIn('observed_site', unmapped_record.parameters)

        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['LCO']['site_lookup_failed'], 2)
        self.assertEqual(summary['LCO']['site_lookups'], 0)

        for record in (none_block_record, unmapped_record):
            facility = op.facility_for(record)
            event = CalendarEvent.objects.get(url=facility.get_observation_url(record.observation_id))
            self.assertEqual(event.telescope, '2m0')

        # Retried on the next sweep -- both succeed this time.
        with patch(
            'solsys_code.calendar_utils.make_request',
            side_effect=[
                observations_block_response(site='coj', telescope='2m0a', state='COMPLETED'),
                observations_block_response(site='ogg', telescope='2m0a', state='COMPLETED'),
            ],
        ):
            second_out = StringIO()
            call_command('project_observation_calendar', stdout=second_out, stderr=StringIO())
        second_summary = _parse_summary(second_out.getvalue())
        self.assertEqual(second_summary['LCO']['site_lookups'], 2)
        self.assertEqual(second_summary['LCO']['site_lookup_failed'], 0)

    def test_pending_record_never_triggers_a_lookup(self) -> None:
        self._make_record('site-pending', status='PENDING')
        with patch('solsys_code.calendar_utils.make_request') as mocked:
            call_command('project_observation_calendar', stdout=StringIO(), stderr=StringIO())
            call_command('project_observation_calendar', stdout=StringIO(), stderr=StringIO())
            mocked.assert_not_called()

    def test_dry_run_performs_no_lookup_and_writes_nothing_to_parameters(self) -> None:
        record = self._make_observed_record('site-dry-run')
        with patch('solsys_code.calendar_utils.make_request') as mocked:
            call_command('project_observation_calendar', '--dry-run', stdout=StringIO(), stderr=StringIO())
            mocked.assert_not_called()
        record.refresh_from_db()
        self.assertNotIn('observed_site', record.parameters)

    def test_failure_message_is_fixed_and_never_leaks_exception_content(self) -> None:
        self._make_observed_record('site-leak')
        leak_marker = 'LEAK_MARKER_apikey_body'
        with patch(
            'solsys_code.calendar_utils.make_request',
            side_effect=ImproperCredentialsException(f'OCS: {leak_marker}'),
        ):
            err = StringIO()
            call_command('project_observation_calendar', stdout=StringIO(), stderr=err)
        error_output = err.getvalue()
        self.assertNotIn(leak_marker, error_output)
        self.assertIn('site-leak', error_output)

    def test_resolve_observed_site_returns_none_none_for_non_terminal_stage(self) -> None:
        record = self._make_record('site-direct-queued', status='PENDING')
        facility = op.facility_for(record)
        increment, message = resolve_observed_site(record, facility)
        self.assertIsNone(increment)
        self.assertIsNone(message)

    def test_resolve_observed_site_returns_none_none_when_already_resolved(self) -> None:
        record = self._make_observed_record('site-direct-resolved')
        record.parameters['observed_site'] = 'coj'
        record.save(update_fields=['parameters'])
        facility = op.facility_for(record)
        increment, message = resolve_observed_site(record, facility)
        self.assertIsNone(increment)
        self.assertIsNone(message)
