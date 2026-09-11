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

from django.core.management import CommandError, call_command
from django.db.models.signals import post_save
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import observation_projector as op
from solsys_code.management.commands.project_observation_calendar import _parse_proposal_arg


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
