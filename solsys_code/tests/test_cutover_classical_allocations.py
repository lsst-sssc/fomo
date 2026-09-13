"""Tests for cutover_classical_allocations (Phase 35 Task 2, D-17/D-18).

Fixtures build their own hand-made blank-url `CalendarEvent` rows, matching the pre-cutover
`load_telescope_runs`'s own three-line description shape (`Dark window (-15 deg, UTC): ...`,
`Status: ...`, `Source line: ...`) -- never dependent on developer-database content. Uses
`tom_targets.tests.factories.NonSiderealTargetFactory` for any `Target` (CLAUDE.md), though
none of these tests need one directly since a classical `CampaignRun` carries `target=None`.
"""

from datetime import date, datetime
from datetime import timezone as dt_timezone
from io import StringIO

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.allocation_projector import allocation_night_url
from solsys_code.management.commands.load_telescope_runs import _source_identifier
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import parse_run_line

_DARK_LINE = 'Dark window (-15 deg, UTC): 2026-07-09T00:00:00+00:00 to 2026-07-09T10:00:00+00:00'

# NTT (809) uses the ESO noon-to-noon date-range convention, so 'NTT EFOSC2 allocation
# 9-12 July' yields 3 observing nights (12 - 9 + 1 - 1 for the ESO closing-boundary drop):
# July 9, 10, 11.
_THREE_NIGHT_LINE = 'NTT EFOSC2 allocation 9-12 July'
_THREE_NIGHTS = [date(date.today().year, 7, day) for day in (9, 10, 11)]


class CutoverClassicalAllocationsTestBase(TestCase):
    """Shared fixtures: an NTT Observatory (obscode 809, real timezone -- matches
    `test_write_and_reconcile.TestLoadTelescopeRunsWritesAllocations`'s own fixture) and an
    FTS Observatory (obscode E10, blank timezone) for the D-18 blank-timezone case."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.other_campaign = TargetList.objects.create(name='Other Campaign')
        cls.ntt = Observatory.objects.create(
            obscode='809',
            name='ESO, La Silla',
            short_name='NTT',
            lat=-29.2567,
            lon=-70.7300,
            altitude=2347,
            timezone='America/Santiago',
        )
        cls.blank_tz_site = Observatory.objects.create(
            obscode='E10',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='',
        )

    def _make_legacy_event(
        self,
        *,
        source_line: str,
        start_time: datetime,
        end_time: datetime,
        status: str = 'allocation',
        telescope: str = 'NTT',
        instrument: str = 'EFOSC2',
        target_list: TargetList | None = None,
        dark_line: str | None = _DARK_LINE,
        title: str | None = None,
        description: str | None = None,
    ) -> CalendarEvent:
        """A hand-built legacy blank-url classical CalendarEvent, matching the pre-cutover
        load_telescope_runs command's own three-line description shape."""
        if description is None:
            body = f'Status: {status}\nSource line: {source_line}'
            description = f'{dark_line}\n{body}' if dark_line is not None else body
        return CalendarEvent.objects.create(
            title=title or f'{telescope} {instrument}',
            url='',
            description=description,
            telescope=telescope,
            instrument=instrument,
            start_time=start_time,
            end_time=end_time,
            target_list=target_list,
        )

    def _make_three_night_group(self, **overrides) -> list[CalendarEvent]:
        """Three blank-url events sharing `_THREE_NIGHT_LINE`, one per night, each starting
        at 23:00 UTC of its own night's evening date (well within the site-local night for
        America/Santiago, so `observing_night()` maps each back to the intended date)."""
        kwargs = {'source_line': _THREE_NIGHT_LINE, 'target_list': self.campaign}
        kwargs.update(overrides)
        events = []
        for night in _THREE_NIGHTS:
            start_time = datetime(night.year, night.month, night.day, 23, 0, tzinfo=dt_timezone.utc)
            end_time = datetime(night.year, night.month, night.day + 1, 9, 0, tzinfo=dt_timezone.utc)
            events.append(self._make_legacy_event(start_time=start_time, end_time=end_time, **kwargs))
        return events


class TestThreeEventGroupConvertsToOneRun(CutoverClassicalAllocationsTestBase):
    """Test 1: three blank-url classical events sharing one parseable Source line: produce
    exactly one new CampaignRun and three events re-keyed to ALLOC:{pk}:{night}, with their
    primary keys unchanged."""

    def test_three_events_produce_one_run_and_three_rekeyed_events(self):
        events = self._make_three_night_group()
        pks = [event.pk for event in events]

        call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(CampaignRun.objects.count(), 1)
        run = CampaignRun.objects.get()
        parsed = parse_run_line(_THREE_NIGHT_LINE)
        expected_key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        self.assertEqual(run.source_identifier, expected_key)
        self.assertEqual(run.source, CampaignRun.Source.CLASSICAL_FILE)
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.campaign_id, self.campaign.pk)

        for pk, night in zip(pks, _THREE_NIGHTS, strict=True):
            event = CalendarEvent.objects.get(pk=pk)
            self.assertEqual(event.url, allocation_night_url(run, night))

    def test_source_identifier_matches_what_a_fresh_import_would_compute(self):
        """A converted run's source_identifier equals what
        load_telescope_runs._source_identifier() computes for the same line -- guarantees a
        re-import of the same line matches this converted run rather than duplicating it."""
        self._make_three_night_group()

        call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())

        run = CampaignRun.objects.get()
        parsed = parse_run_line(_THREE_NIGHT_LINE)
        expected_key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        self.assertEqual(run.source_identifier, expected_key)


class TestConvertedEventFields(CutoverClassicalAllocationsTestBase):
    """Test 2: each converted event keeps its start_time and end_time exactly, and gains a
    CalendarEventMeta row whose run is the new run."""

    def test_start_and_end_time_are_byte_identical_and_meta_row_created(self):
        events = self._make_three_night_group()
        spans_before = [(event.start_time, event.end_time) for event in events]
        pks = [event.pk for event in events]

        call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())

        run = CampaignRun.objects.get()
        for pk, (start_before, end_before) in zip(pks, spans_before, strict=True):
            event = CalendarEvent.objects.get(pk=pk)
            self.assertEqual(event.start_time, start_before)
            self.assertEqual(event.end_time, end_before)
            meta = CalendarEventMeta.objects.get(event=event)
            self.assertEqual(meta.run_id, run.pk)


class TestSecondInvocationIsANoOp(CutoverClassicalAllocationsTestBase):
    """Test 3: a second invocation converts nothing, reports zero conversions and exits 0
    -- no blank-url classical event remains for it to find."""

    def test_second_run_converts_nothing(self):
        self._make_three_night_group()
        call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())
        run_count_after_first = CampaignRun.objects.count()

        out = StringIO()
        call_command('cutover_classical_allocations', stdout=out, stderr=StringIO())

        self.assertEqual(CampaignRun.objects.count(), run_count_after_first)
        self.assertIn('candidates: 0', out.getvalue())
        self.assertIn('events re-keyed: 0', out.getvalue())
        self.assertFalse(CalendarEvent.objects.filter(url='').exists())


class TestUnparseableEventLeftUntouched(CutoverClassicalAllocationsTestBase):
    """Test 4: a blank-url event with no parseable Source line: is left byte-identical
    (url, title, description, start, end, modified all unchanged), is listed with its
    reason, and the command exits non-zero."""

    def test_no_source_line_marker_is_reported_and_left_untouched(self):
        event = self._make_legacy_event(
            source_line='unused',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 9, 0, tzinfo=dt_timezone.utc),
            description='A hand-typed note with no Source line marker at all.',
        )
        fields_before = (event.url, event.title, event.description, event.start_time, event.end_time, event.modified)

        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        event.refresh_from_db()
        self.assertEqual(
            (event.url, event.title, event.description, event.start_time, event.end_time, event.modified),
            fields_before,
        )
        self.assertIn(f'pk={event.pk}', err.getvalue())
        self.assertEqual(CampaignRun.objects.count(), 0)


class TestUnexplainableGroupsGetDistinctReasons(CutoverClassicalAllocationsTestBase):
    """Test 5: a blank-url event whose Source line: names an unknown telescope, and one
    whose site has a blank timezone, are each reported with their own distinct reason and
    left untouched; the command exits non-zero."""

    def test_unknown_telescope_and_blank_timezone_get_distinct_reasons(self):
        unknown_telescope_event = self._make_legacy_event(
            source_line='Bogus EFOSC2 allocation 9-12 July',
            telescope='Bogus',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 9, 0, tzinfo=dt_timezone.utc),
        )
        blank_timezone_event = self._make_legacy_event(
            source_line='FTS MuSCAT3 allocation 9-12 July',
            telescope='FTS',
            instrument='MuSCAT3',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 9, 0, tzinfo=dt_timezone.utc),
        )

        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        error_output = err.getvalue()
        lines_by_pk = {}
        for line in error_output.splitlines():
            for event in (unknown_telescope_event, blank_timezone_event):
                if f'pk={event.pk} ' in line:
                    lines_by_pk[event.pk] = line
        self.assertIn(unknown_telescope_event.pk, lines_by_pk)
        self.assertIn(blank_timezone_event.pk, lines_by_pk)
        self.assertNotEqual(lines_by_pk[unknown_telescope_event.pk], lines_by_pk[blank_timezone_event.pk])
        unknown_telescope_event.refresh_from_db()
        blank_timezone_event.refresh_from_db()
        self.assertEqual(unknown_telescope_event.url, '')
        self.assertEqual(blank_timezone_event.url, '')
        self.assertEqual(CampaignRun.objects.count(), 0)


class TestDryRunMatchesRealRun(CutoverClassicalAllocationsTestBase):
    """Test 6: --dry-run over the Test 1 fixture creates no CampaignRun row and re-keys no
    event, and its reported counts equal what the subsequent real run then performs."""

    def test_dry_run_creates_nothing_and_predicts_the_real_run(self):
        self._make_three_night_group()

        dry_out = StringIO()
        call_command('cutover_classical_allocations', '--dry-run', stdout=dry_out, stderr=StringIO())

        self.assertEqual(CampaignRun.objects.count(), 0)
        self.assertTrue(CalendarEvent.objects.filter(url='').count() == 3)
        self.assertIn('events re-keyed: 3', dry_out.getvalue())
        self.assertIn('runs created: 1', dry_out.getvalue())

        real_out = StringIO()
        call_command('cutover_classical_allocations', stdout=real_out, stderr=StringIO())

        self.assertIn('events re-keyed: 3', real_out.getvalue())
        self.assertIn('runs created: 1', real_out.getvalue())
        self.assertEqual(CampaignRun.objects.count(), 1)

    def test_dry_run_still_exits_non_zero_on_an_unexplainable_event(self):
        """A dry run must not silently succeed while an unexplainable event exists -- that
        is exactly the condition the operator must clear before the real run."""
        self._make_legacy_event(
            source_line='unused',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 9, 0, tzinfo=dt_timezone.utc),
            description='No Source line marker here.',
        )

        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', '--dry-run', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(CampaignRun.objects.count(), 0)


class TestNeverDeletesACalendarEvent(CutoverClassicalAllocationsTestBase):
    """Test 7: the command never deletes a CalendarEvent -- the total row count before and
    after any invocation, including the failure paths, is identical."""

    def test_row_count_unchanged_across_success_and_failure_paths(self):
        self._make_three_night_group()
        self._make_legacy_event(
            source_line='unused',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 9, 0, tzinfo=dt_timezone.utc),
            description='No Source line marker here.',
        )
        count_before = CalendarEvent.objects.count()

        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(CalendarEvent.objects.count(), count_before)


class TestSummaryPrintsFinalAllocationCount(CutoverClassicalAllocationsTestBase):
    """Test 8: the summary prints a final per-night allocation event count."""

    def test_summary_names_the_total_alloc_event_count(self):
        self._make_three_night_group()

        out = StringIO()
        call_command('cutover_classical_allocations', stdout=out, stderr=StringIO())

        self.assertIn('total ALLOC:-keyed calendar events now in the database: 3', out.getvalue())


class TestForeignAttributionLeftUntouched(CutoverClassicalAllocationsTestBase):
    """D-18's foreign-attribution reason (bonus coverage beyond the plan's 8 named tests,
    referenced by the threat model T-35-05 and the prohibitions block): an event already
    attributed to a DIFFERENT CampaignRun is reported and left alone, its own group's other
    events still convert normally."""

    def test_foreign_attributed_event_is_skipped_others_in_group_still_convert(self):
        events = self._make_three_night_group()
        other_run = CampaignRun.objects.create(
            campaign=self.other_campaign,
            telescope_instrument='Other/Instrument',
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        foreign_event = events[0]
        CalendarEventMeta.objects.create(event=foreign_event, run=other_run)
        foreign_event_url_before = foreign_event.url

        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        self.assertIn(f'pk={foreign_event.pk}', err.getvalue())
        foreign_event.refresh_from_db()
        self.assertEqual(foreign_event.url, foreign_event_url_before)
        meta = CalendarEventMeta.objects.get(event=foreign_event)
        self.assertEqual(meta.run_id, other_run.pk)

        run = CampaignRun.objects.get(source__isnull=False, source=CampaignRun.Source.CLASSICAL_FILE)
        for event in events[1:]:
            event.refresh_from_db()
            self.assertNotEqual(event.url, '')
            self.assertTrue(event.url.startswith(f'ALLOC:{run.pk}:'))


class TestCampaignMismatchGroupLeftUntouched(CutoverClassicalAllocationsTestBase):
    """D-18's campaign-mismatch reason (bonus coverage): a group whose events disagree on
    their campaign (target_list) is unexplainable and left completely alone."""

    def test_group_disagreeing_on_campaign_is_reported_and_left_untouched(self):
        night = _THREE_NIGHTS[0]
        start_time = datetime(night.year, night.month, night.day, 23, 0, tzinfo=dt_timezone.utc)
        end_time = datetime(night.year, night.month, night.day + 1, 9, 0, tzinfo=dt_timezone.utc)
        one = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.campaign
        )
        two = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.other_campaign
        )

        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())

        one.refresh_from_db()
        two.refresh_from_db()
        self.assertEqual(one.url, '')
        self.assertEqual(two.url, '')
        self.assertEqual(CampaignRun.objects.count(), 0)
