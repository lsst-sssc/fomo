"""Tests for cutover_classical_allocations (Phase 35 Task 2, D-17/D-18).

Fixtures build their own hand-made blank-url `CalendarEvent` rows, matching the pre-cutover
`load_telescope_runs`'s own three-line description shape (`Dark window (-15 deg, UTC): ...`,
`Status: ...`, `Source line: ...`) -- never dependent on developer-database content. Uses
`tom_targets.tests.factories.NonSiderealTargetFactory` for any `Target` (CLAUDE.md), though
none of these tests need one directly since a classical `CampaignRun` carries `target=None`.
"""

import re
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from io import StringIO
from unittest.mock import patch
from uuid import uuid4

from django.contrib.auth.models import User
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.allocation_projector import allocation_events, allocation_night_url
from solsys_code.campaign_reconciler import owned_events
from solsys_code.management.commands.load_telescope_runs import _source_identifier
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import observing_night as real_observing_night
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


class TestCutoverSequenceContract(CutoverClassicalAllocationsTestBase):
    """Task 3 (35-VALIDATION.md's Manual-Only Verifications, second row): pins the same
    end-state properties the real-database proof (this plan's SUMMARY) measured, against
    synthetic fixtures, so the guarantee survives without that database.

    Builds one world containing: a convertible blank-url group (this class's own
    three-night NTT fixture); one unexplainable blank-url event; a per-night-dispatched
    run that stays per-night, with a legacy RUN:{pk}:{night} event inside its window
    (re-keyed by the sweep, D-16 first half); a per-night-dispatched run a queue source
    now sends to the container (D-10), with a legacy RUN:{pk}:{night} event (deleted by
    the sweep, Task 1/D-16 second half); and a per-night-dispatched run with a linked,
    placed ObservationRecord that retires one of its nights (D-05) automatically, before
    the cutover or the sweep ever runs. Runs the cutover then the reconciler sweep once,
    and asserts the pinned end-state."""

    def test_cutover_then_sweep_reaches_the_pinned_end_state(self):
        # 1. Convertible blank-url group (this base class's own 3-night NTT fixture).
        convertible_events = self._make_three_night_group()

        # 2. Unexplainable blank-url event -- no Source line: marker at all.
        unexplained_event = self._make_legacy_event(
            source_line='unused',
            start_time=datetime(2026, 7, 9, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 9, 0, tzinfo=dt_timezone.utc),
            description='A hand-typed note with no Source line marker at all.',
        )

        # 3. A per-night-dispatched run that STAYS per-night, with a legacy
        # RUN:{pk}:{night} event inside its current window -- never reconciled yet, so
        # only the hand-made legacy artifact exists (the pre-cutover shape). The sweep's
        # first-ever reconcile of this run re-keys it in place (D-16 first half).
        stays_per_night_run = CampaignRun.objects.create(
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            site=self.ntt,
            site_raw='NTT',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 1),
        )
        rekey_legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=f'RUN:{stays_per_night_run.pk}:2026-09-01',
            start_time=datetime(2026, 9, 1, 23, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 9, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=rekey_legacy_event, run=stays_per_night_run)
        rekey_legacy_pk = rekey_legacy_event.pk

        # 4. A per-night-dispatched run a queue source now sends to the container (D-10),
        # with a legacy RUN:{pk}:{night} event -- also never reconciled yet. The sweep's
        # first-ever reconcile creates the bare container AND deletes this leftover
        # per-night artifact (Task 1/D-16 second half).
        now_container_run = CampaignRun.objects.create(
            source=CampaignRun.Source.LCO_QUEUE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            site=self.ntt,
            site_raw='NTT',
            window_start=date(2026, 9, 2),
            window_end=date(2026, 9, 2),
        )
        delete_legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact',
            url=f'RUN:{now_container_run.pk}:2026-09-02',
            start_time=datetime(2026, 9, 2, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=delete_legacy_event, run=now_container_run)
        delete_legacy_pk = delete_legacy_event.pk

        # 5. A per-night-dispatched run with a linked, placed record -- creating the link
        # retires its night automatically via the D-11 post_save receiver, before either
        # the cutover or the sweep below ever runs.
        retire_run = CampaignRun.objects.create(
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            site=self.ntt,
            site_raw='NTT',
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
        )
        target = NonSiderealTargetFactory.create()
        owner = User.objects.create(username=f'contract-owner-{uuid4().hex[:8]}')
        # 23:00 UTC maps (via observing_night()'s local-noon anchor, America/Santiago) to
        # the SAME evening date -- matching retire_run.window_start=2026-09-03 exactly
        # (the same pattern _make_three_night_group() uses for its own fixtures).
        scheduled_start = datetime(2026, 9, 3, 23, 0, tzinfo=dt_timezone.utc)
        record = ObservationRecord.objects.create(
            target=target,
            user=owner,
            facility='LCO',
            observation_id=f'contract-{uuid4().hex[:8]}',
            status='COMPLETED',
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_start + timedelta(hours=2),
            parameters={'proposal': 'TEST'},
        )
        CampaignRunObservation.objects.create(run=retire_run, observation_record=record)

        # Step 3: the cutover command (exits non-zero -- the unexplainable event, as designed).
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())

        # Step 4: the reconciler sweep, once.
        call_command('reconcile_campaign_runs', stdout=StringIO(), stderr=StringIO())

        # -- Pinned end-state --

        # Zero date-bearing RUN:{pk}:{date} events remain anywhere.
        self.assertFalse(CalendarEvent.objects.filter(url__regex=r'^RUN:[0-9]+:').exists())

        # Zero blank-url events remain apart from the one reported unexplainable.
        blank_url_pks = list(CalendarEvent.objects.filter(url='').values_list('pk', flat=True))
        self.assertEqual(blank_url_pks, [unexplained_event.pk])

        # The stays-per-night run's legacy night is now ALLOC:-keyed, same primary key,
        # never a duplicate row.
        rekeyed_event = CalendarEvent.objects.get(pk=rekey_legacy_pk)
        self.assertEqual(rekeyed_event.url, allocation_night_url(stays_per_night_run, date(2026, 9, 1)))
        self.assertEqual(allocation_events(stays_per_night_run).count(), 1)

        # The queue run's legacy night is gone entirely -- deleted, not rekeyed elsewhere --
        # and its bare container now exists.
        self.assertFalse(CalendarEvent.objects.filter(pk=delete_legacy_pk).exists())
        self.assertTrue(CalendarEvent.objects.filter(url=f'RUN:{now_container_run.pk}').exists())
        self.assertEqual(owned_events(now_container_run).count(), 1)

        # The linked record's night is retired: no allocation event at all for it.
        self.assertEqual(allocation_events(retire_run).count(), 0)

        # The convertible group is fully re-keyed to ALLOC:, same primary keys.
        parsed = parse_run_line(_THREE_NIGHT_LINE)
        converted_key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        converted_run = CampaignRun.objects.get(source_identifier=converted_key)
        for event in convertible_events:
            event.refresh_from_db()
            self.assertTrue(event.url.startswith(f'ALLOC:{converted_run.pk}:'))

        # Three-group reconciliation over the two hand-made legacy artifacts: one re-keyed,
        # one deleted, zero retired-by-observation among THEM specifically (the retire_run
        # fixture never had a legacy RUN:-event to begin with -- it demonstrates the third
        # group exists as a mechanism, not that it applies to a pre-existing legacy row).
        rekeyed_count = 1
        legacy_deleted_count = 1
        self.assertEqual(rekeyed_count + legacy_deleted_count, 2)


class TestGroupTransactionBoundary(CutoverClassicalAllocationsTestBase):
    """35-REVIEW.md WR-06: each group's writes (the run write plus every event's re-key)
    are wrapped in a savepoint, so a genuinely unexpected group-level failure rolls back
    cleanly, while a per-event failure (D-18's own contract) still leaves the run and every
    OTHER event in the group converted."""

    def test_group_level_failure_rolls_back_the_run_and_leaves_every_event_untouched(self):
        events = self._make_three_night_group()
        pks = [event.pk for event in events]

        with patch(
            'solsys_code.management.commands.cutover_classical_allocations.insert_or_create_campaign_run',
            side_effect=RuntimeError('simulated connection drop'),
        ):
            err = StringIO()
            with self.assertRaises(CommandError):
                call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        self.assertIn('RuntimeError', err.getvalue())
        parsed = parse_run_line(_THREE_NIGHT_LINE)
        key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        self.assertFalse(CampaignRun.objects.filter(source_identifier=key).exists())
        for pk in pks:
            event = CalendarEvent.objects.get(pk=pk)
            self.assertEqual(event.url, '')

    def test_one_event_exception_does_not_roll_back_the_run_or_other_events(self):
        events = self._make_three_night_group()

        def _fail_first_only(start_time, site_zone):
            if start_time == events[0].start_time:
                raise ValueError('simulated per-event failure')
            return real_observing_night(start_time, site_zone)

        with patch(
            'solsys_code.management.commands.cutover_classical_allocations.observing_night',
            side_effect=_fail_first_only,
        ):
            err = StringIO()
            with self.assertRaises(CommandError):
                call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        self.assertIn('ValueError', err.getvalue())
        parsed = parse_run_line(_THREE_NIGHT_LINE)
        key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        run = CampaignRun.objects.get(source_identifier=key)  # the group's run was still created

        events[0].refresh_from_db()
        self.assertEqual(events[0].url, '')  # the failing event is left byte-identical
        for event in events[1:]:
            event.refresh_from_db()
            self.assertTrue(event.url.startswith(f'ALLOC:{run.pk}:'))  # the other two still converted


class TestWindowContainmentGuard(CutoverClassicalAllocationsTestBase):
    """35-REVIEW.md WR-08: a re-keyed event's own independently-derived night must lie
    inside the run's own window (from the schedule line's day range) -- nothing else
    asserts the two agree, and a mismatch silently re-keyed a url outside the window that
    the very next sweep's convergence step would then classify as stale and delete."""

    def test_event_whose_derived_night_falls_outside_the_window_is_reported_not_rekeyed(self):
        in_window_events = self._make_three_night_group()
        year = date.today().year
        outlier_start = datetime(year, 7, 20, 23, 0, tzinfo=dt_timezone.utc)
        outlier_end = datetime(year, 7, 21, 9, 0, tzinfo=dt_timezone.utc)
        outlier = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE,
            start_time=outlier_start,
            end_time=outlier_end,
            target_list=self.campaign,
        )

        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        self.assertIn("falls outside the run's window", err.getvalue())
        outlier.refresh_from_db()
        self.assertEqual(outlier.url, '')  # never re-keyed to a url outside the window

        parsed = parse_run_line(_THREE_NIGHT_LINE)
        key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        run = CampaignRun.objects.get(source_identifier=key)
        for event in in_window_events:
            event.refresh_from_db()
            self.assertTrue(event.url.startswith(f'ALLOC:{run.pk}:'))  # the in-window nights still converted


class TestDryRunAppliesTheWindowCheck(CutoverClassicalAllocationsTestBase):
    """35-REVIEW.md NF-02: the WR-08 window-containment check must also apply on the
    --dry-run path, and both paths must classify the failure under the dedicated
    window_mismatch reason rather than the generic unexpected-error bucket -- a dry run
    that exits 0 over a fixture the immediately following real run rejects hides the one
    thing a dry run exists to surface."""

    def _make_outlier_fixture(self) -> tuple[list[CalendarEvent], CalendarEvent]:
        in_window_events = self._make_three_night_group()
        year = date.today().year
        outlier_start = datetime(year, 7, 20, 23, 0, tzinfo=dt_timezone.utc)
        outlier_end = datetime(year, 7, 21, 9, 0, tzinfo=dt_timezone.utc)
        outlier = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE,
            start_time=outlier_start,
            end_time=outlier_end,
            target_list=self.campaign,
        )
        return in_window_events, outlier

    def test_dry_run_reports_window_mismatch_and_does_not_count_it_as_rekeyed(self):
        _in_window_events, outlier = self._make_outlier_fixture()

        out = StringIO()
        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', '--dry-run', stdout=out, stderr=err)

        stdout_value = out.getvalue()
        self.assertIn('unexplained (window_mismatch): 1', stdout_value)
        self.assertIn('events re-keyed: 3', stdout_value)  # the outlier is NOT counted as would-be re-keyed
        self.assertIn("falls outside the run's window", err.getvalue())
        self.assertEqual(CampaignRun.objects.count(), 0)
        outlier.refresh_from_db()
        self.assertEqual(outlier.url, '')
        self.assertFalse(CalendarEventMeta.objects.filter(event=outlier).exists())

    def test_real_run_also_reports_window_mismatch_not_the_generic_bucket(self):
        """Pins the real path's reclassification away from the generic `other` bucket --
        both passes must name the same category."""
        _in_window_events, outlier = self._make_outlier_fixture()

        out = StringIO()
        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=out, stderr=err)

        stdout_value = out.getvalue()
        self.assertIn('unexplained (window_mismatch): 1', stdout_value)
        self.assertIn('events re-keyed: 3', stdout_value)
        self.assertIn("falls outside the run's window", err.getvalue())
        outlier.refresh_from_db()
        self.assertEqual(outlier.url, '')
        self.assertFalse(CalendarEventMeta.objects.filter(event=outlier).exists())


class TestAllForeignAttributedGroupWritesNothing(CutoverClassicalAllocationsTestBase):
    """35-REVIEW.md WR-07: a `Source line:` group whose EVERY event is already attributed
    to a different CampaignRun must not get a run created or updated for it at all -- an
    unconditional write here would leave an APPROVED, site-resolved, windowed run owning
    zero events, which the next `reconcile_campaign_runs` sweep would then project a full
    duplicate set of `ALLOC:` nights for, over nights the foreign run already owns."""

    def _make_all_foreign_fixture(self) -> tuple[CampaignRun, list[CalendarEvent]]:
        other_run = CampaignRun.objects.create(
            campaign=self.other_campaign,
            telescope_instrument='Other/Instrument',
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        events = []
        for night in _THREE_NIGHTS[:2]:
            start_time = datetime(night.year, night.month, night.day, 23, 0, tzinfo=dt_timezone.utc)
            end_time = datetime(night.year, night.month, night.day + 1, 9, 0, tzinfo=dt_timezone.utc)
            event = self._make_legacy_event(
                source_line=_THREE_NIGHT_LINE,
                start_time=start_time,
                end_time=end_time,
                target_list=self.campaign,
            )
            CalendarEventMeta.objects.create(event=event, run=other_run)
            events.append(event)
        return other_run, events

    def test_all_foreign_attributed_group_creates_no_run(self):
        other_run, events = self._make_all_foreign_fixture()
        run_count_before = CampaignRun.objects.count()

        out = StringIO()
        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=out, stderr=err)

        self.assertEqual(CampaignRun.objects.count(), run_count_before)
        error_output = err.getvalue()
        for event in events:
            event.refresh_from_db()
            self.assertEqual(event.url, '')
            self.assertIn(f'pk={event.pk}', error_output)
            meta = CalendarEventMeta.objects.get(event=event)
            self.assertEqual(meta.run_id, other_run.pk)

        stdout_value = out.getvalue()
        self.assertIn('runs created: 0', stdout_value)
        self.assertIn('groups: 1', stdout_value)
        self.assertIn(f'unexplained (foreign_attribution): {len(events)}', stdout_value)

    def test_all_foreign_attributed_group_dry_run_predicts_no_run(self):
        self._make_all_foreign_fixture()
        run_count_before = CampaignRun.objects.count()

        out = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', '--dry-run', stdout=out, stderr=StringIO())

        self.assertIn('runs created: 0', out.getvalue())
        self.assertEqual(CampaignRun.objects.count(), run_count_before)


class TestUnknownClassicalStatusGuard(CutoverClassicalAllocationsTestBase):
    """35-REVIEW.md WR-09: a status word with no `_CLASSICAL_RUN_STATUS` mapping must be
    reported per-group (D-18), not escape as an uncaught `KeyError` that aborts the whole
    cutover mid-run."""

    def test_missing_status_mapping_is_reported_per_group_not_uncaught(self):
        events = self._make_three_night_group()
        pks = [event.pk for event in events]

        with patch.dict(
            'solsys_code.management.commands.cutover_classical_allocations._CLASSICAL_RUN_STATUS', clear=True
        ):
            err = StringIO()
            with self.assertRaises(CommandError):
                call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        self.assertIn('unknown classical status', err.getvalue())
        for pk in pks:
            event = CalendarEvent.objects.get(pk=pk)
            self.assertEqual(event.url, '')  # left byte-identical, never partially converted


class TestKeyCollisionDetection(CutoverClassicalAllocationsTestBase):
    """WR-11 (35-REVIEW.md): a second event whose derived night is already claimed --
    either within this run (two schedule lines colliding on the same night) or by a row
    another CalendarEvent already holds (an import of the same schedule file that ran
    before the cutover) -- is reported under its own named reason category and left
    byte-identical, never re-keyed onto a url another row already holds."""

    def _first_night_span(self) -> tuple[date, datetime, datetime]:
        night = _THREE_NIGHTS[0]
        start_time = datetime(night.year, night.month, night.day, 23, 0, tzinfo=dt_timezone.utc)
        end_time = datetime(night.year, night.month, night.day + 1, 9, 0, tzinfo=dt_timezone.utc)
        return night, start_time, end_time

    def test_in_run_collision_rekeys_the_first_claimant_reports_the_second(self):
        _night, start_time, end_time = self._first_night_span()
        first = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.campaign
        )
        second = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.campaign
        )

        out = StringIO()
        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=out, stderr=err)

        first.refresh_from_db()
        second.refresh_from_db()
        self.assertNotEqual(first.url, '')  # first claimant converted
        self.assertEqual(second.url, '')  # second claimant left byte-identical
        self.assertFalse(CalendarEventMeta.objects.filter(event=second).exists())

        self.assertIn(f'pk={second.pk}', err.getvalue())

        stdout_value = out.getvalue()
        self.assertIn('unexplained (key_collision): 1', stdout_value)

        alloc_urls = list(CalendarEvent.objects.filter(url__startswith='ALLOC:').values_list('url', flat=True))
        self.assertEqual(len(alloc_urls), len(set(alloc_urls)))  # no repeated ALLOC: url anywhere

    def test_existing_url_collision_reports_legacy_event_leaves_pre_existing_row_untouched(self):
        night, start_time, end_time = self._first_night_span()
        parsed = parse_run_line(_THREE_NIGHT_LINE)
        key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        run = CampaignRun.objects.create(
            source_identifier=key,
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            campaign=self.campaign,
            site=self.ntt,
            site_raw='NTT',
            window_start=_THREE_NIGHTS[0],
            window_end=_THREE_NIGHTS[-1],
        )
        alloc_url = allocation_night_url(run, night)
        pre_existing_event = CalendarEvent.objects.create(
            title='Pre-existing ALLOC event (import ran first)',
            url=alloc_url,
            start_time=start_time,
            end_time=end_time,
        )
        legacy_event = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.campaign
        )

        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=err)

        legacy_event.refresh_from_db()
        self.assertEqual(legacy_event.url, '')  # left byte-identical, never re-keyed onto the taken url
        self.assertIn(f'pk={legacy_event.pk}', err.getvalue())

        self.assertEqual(CalendarEvent.objects.filter(url=alloc_url).count(), 1)
        pre_existing_event.refresh_from_db()
        self.assertEqual(pre_existing_event.url, alloc_url)  # untouched

    def test_dry_run_predicts_in_run_collision_without_writing(self):
        _night, start_time, end_time = self._first_night_span()
        first = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.campaign
        )
        second = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.campaign
        )

        out = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', '--dry-run', stdout=out, stderr=StringIO())

        self.assertEqual(CampaignRun.objects.count(), 0)
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual(first.url, '')
        self.assertEqual(second.url, '')

        stdout_value = out.getvalue()
        self.assertIn('unexplained (key_collision): 1', stdout_value)
        self.assertIn('events re-keyed: 1', stdout_value)  # same count the real run then performs


def _parse_cutover_summary(text: str) -> dict:
    """Parses a cutover_classical_allocations stdout summary into a comparable structure
    (counters plus a `{category: count}` reason breakdown), so the dry-run/real-run
    parity assertion below compares structures rather than a hand-maintained list of
    substrings."""
    done_line_match = re.search(
        r'candidates: (\d+), groups: (\d+), runs created: (\d+), updated: (\d+), unchanged: (\d+), '
        r'events re-keyed: (\d+), unexplained: (\d+)',
        text,
    )
    assert done_line_match is not None, f'summary line not found in: {text!r}'
    (candidates, groups, runs_created, runs_updated, runs_unchanged, events_rekeyed, unexplained_total) = (
        int(value) for value in done_line_match.groups()
    )
    reasons = {category: int(count) for category, count in re.findall(r'unexplained \((\w+)\): (\d+)', text)}
    return {
        'candidates': candidates,
        'groups': groups,
        'runs_created': runs_created,
        'runs_updated': runs_updated,
        'runs_unchanged': runs_unchanged,
        'events_rekeyed': events_rekeyed,
        'unexplained_total': unexplained_total,
        'reasons': reasons,
    }


class TestDryRunAndRealRunAgree(CutoverClassicalAllocationsTestBase):
    """35-REVIEW.md NF-02: the dry run and the immediately following real run agree on
    every counter, every reason category and the exit status over a fixture that trips
    all three per-event preconditions at once -- the property that would have caught the
    WR-11 rebuild copying only two of the real path's three checks."""

    def _make_all_three_preconditions_fixture(self) -> dict:
        parsed = parse_run_line(_THREE_NIGHT_LINE)
        key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        year = date.today().year
        # Deliberately WIDER than the schedule line's own July 9..11 window: an
        # implementation that read the window off this pre-existing run (rather than the
        # freshly re-derived fields['window_start']/fields['window_end']) would wave the
        # July 20 outlier through on the dry-run path while the real pass -- which writes
        # the correct narrow window onto `run` before its own check runs -- rejected it.
        # Do not "tidy" this window narrower; it is what makes the NF-02 regression
        # visible to this test.
        run = CampaignRun.objects.create(
            source_identifier=key,
            source=CampaignRun.Source.CLASSICAL_FILE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='NTT/EFOSC2',
            campaign=self.campaign,
            site=self.ntt,
            site_raw='NTT',
            window_start=date(year, 7, 1),
            window_end=date(year, 7, 31),
        )

        def _span(night: date) -> tuple[datetime, datetime]:
            start_time = datetime(night.year, night.month, night.day, 23, 0, tzinfo=dt_timezone.utc)
            end_time = datetime(night.year, night.month, night.day + 1, 9, 0, tzinfo=dt_timezone.utc)
            return start_time, end_time

        existing_url = allocation_night_url(run, _THREE_NIGHTS[1])
        existing_start, existing_end = _span(_THREE_NIGHTS[1])
        pre_existing_event = CalendarEvent.objects.create(
            title='Pre-existing ALLOC event (import ran first)',
            url=existing_url,
            start_time=existing_start,
            end_time=existing_end,
        )

        def _event_for(night: date) -> CalendarEvent:
            start_time, end_time = _span(night)
            return self._make_legacy_event(
                source_line=_THREE_NIGHT_LINE, start_time=start_time, end_time=end_time, target_list=self.campaign
            )

        first_claimant = _event_for(_THREE_NIGHTS[0])
        in_run_collision = _event_for(_THREE_NIGHTS[0])
        existing_url_collision = _event_for(_THREE_NIGHTS[1])
        third_night = _event_for(_THREE_NIGHTS[2])
        outlier_start = datetime(year, 7, 20, 23, 0, tzinfo=dt_timezone.utc)
        outlier_end = datetime(year, 7, 21, 9, 0, tzinfo=dt_timezone.utc)
        window_mismatch = self._make_legacy_event(
            source_line=_THREE_NIGHT_LINE, start_time=outlier_start, end_time=outlier_end, target_list=self.campaign
        )
        return {
            'run': run,
            'pre_existing_event': pre_existing_event,
            'first_claimant': first_claimant,
            'in_run_collision': in_run_collision,
            'existing_url_collision': existing_url_collision,
            'third_night': third_night,
            'window_mismatch': window_mismatch,
        }

    def test_dry_run_and_real_run_report_identical_counts_and_reasons(self):
        self._make_all_three_preconditions_fixture()

        dry_out = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', '--dry-run', stdout=dry_out, stderr=StringIO())
        dry_summary = _parse_cutover_summary(dry_out.getvalue())

        real_out = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=real_out, stderr=StringIO())
        real_summary = _parse_cutover_summary(real_out.getvalue())

        self.assertEqual(dry_summary, real_summary)
        self.assertEqual(dry_summary['events_rekeyed'], 2)
        self.assertEqual(dry_summary['unexplained_total'], 3)
        self.assertEqual(dry_summary['reasons'], {'key_collision': 2, 'window_mismatch': 1})

    def test_out_of_window_event_is_byte_identical_after_both_passes(self):
        fixture = self._make_all_three_preconditions_fixture()
        outlier = fixture['window_mismatch']
        pre_existing_event = fixture['pre_existing_event']
        existing_url = pre_existing_event.url

        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', '--dry-run', stdout=StringIO(), stderr=StringIO())
        outlier.refresh_from_db()
        self.assertEqual(outlier.url, '')
        self.assertFalse(CalendarEventMeta.objects.filter(event=outlier).exists())

        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=StringIO(), stderr=StringIO())
        outlier.refresh_from_db()
        self.assertEqual(outlier.url, '')
        self.assertFalse(CalendarEventMeta.objects.filter(event=outlier).exists())

        pre_existing_event.refresh_from_db()
        self.assertEqual(pre_existing_event.url, existing_url)  # untouched by either pass
        alloc_urls = list(CalendarEvent.objects.filter(url__startswith='ALLOC:').values_list('url', flat=True))
        self.assertEqual(len(alloc_urls), len(set(alloc_urls)))  # no repeated ALLOC: url anywhere


class TestDuplicateIdentityKeyAcrossGroups(CutoverClassicalAllocationsTestBase):
    """NF-14 (35-REVIEW.md): `_source_identifier()` deliberately ignores `parsed.status`
    (`load_telescope_runs.py`), so two GROUPS -- keyed on the raw `source_line` string --
    whose lines differ only in status word resolve to the SAME run identity key. Before
    the fix, the second group silently overwrote the first group's `CampaignRun` (a
    find-or-update on the same `source_identifier`), and its own events were then rejected
    as `key_collision` -- a reason whose documented remedy (delete/re-attribute the
    duplicate row) is wrong for a legitimate second schedule line. The exact status-only
    difference the review reproduced: an 'allocation' line and a 'cancelled' line for the
    same telescope, instrument and window."""

    _ALLOCATION_LINE = 'NTT EFOSC2 allocation 9-12 July'
    _CANCELLED_LINE = 'NTT EFOSC2 cancelled 9-12 July'

    def _make_two_groups_same_identity_key_fixture(self) -> dict:
        """Three blank-url events per line, one per night, sharing all three nights --
        group A (`_ALLOCATION_LINE`) created first so it wins the identity key under
        insertion-order dict processing, matching candidates.order_by('pk')."""

        def _events_for(source_line: str) -> list[CalendarEvent]:
            events = []
            for night in _THREE_NIGHTS:
                start_time = datetime(night.year, night.month, night.day, 23, 0, tzinfo=dt_timezone.utc)
                end_time = datetime(night.year, night.month, night.day + 1, 9, 0, tzinfo=dt_timezone.utc)
                events.append(
                    self._make_legacy_event(
                        source_line=source_line,
                        start_time=start_time,
                        end_time=end_time,
                        status='allocation' if source_line == self._ALLOCATION_LINE else 'cancelled',
                        target_list=self.campaign,
                    )
                )
            return events

        group_a = _events_for(self._ALLOCATION_LINE)
        group_b = _events_for(self._CANCELLED_LINE)
        return {'group_a': group_a, 'group_b': group_b}

    def test_second_group_is_rejected_as_duplicate_identity_not_merged(self):
        fixture = self._make_two_groups_same_identity_key_fixture()
        group_a, group_b = fixture['group_a'], fixture['group_b']

        out = StringIO()
        err = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=out, stderr=err)

        # Exactly one CampaignRun for the shared identity key -- group B never merged into
        # it, and never created a second one either.
        parsed = parse_run_line(self._ALLOCATION_LINE)
        key = _source_identifier(parsed, _THREE_NIGHTS[0], _THREE_NIGHTS[-1])
        self.assertEqual(CampaignRun.objects.filter(source_identifier=key).count(), 1)
        run = CampaignRun.objects.get(source_identifier=key)

        # The run's fields are group A's (the first claimant), never overwritten by group
        # B's differing status.
        self.assertEqual(run.run_status, CampaignRun.RunStatus.PLANNED)  # 'allocation' -> PLANNED
        self.assertIn('Status: allocation', run.observation_details)

        # Group A's events are re-keyed; group B's are left byte-identical and reported
        # under duplicate_identity, never merged onto group A's run.
        for event in group_a:
            event.refresh_from_db()
            self.assertTrue(event.url.startswith(f'ALLOC:{run.pk}:'))
        for event in group_b:
            event.refresh_from_db()
            self.assertEqual(event.url, '')
            self.assertFalse(CalendarEventMeta.objects.filter(event=event).exists())
            self.assertIn(f'pk={event.pk}', err.getvalue())

        stdout_value = out.getvalue()
        self.assertIn('runs created: 1', stdout_value)
        self.assertIn('events re-keyed: 3', stdout_value)
        self.assertIn('unexplained (duplicate_identity): 3', stdout_value)
        self.assertIn('already claimed', err.getvalue())

    def test_dry_run_and_real_run_agree_on_duplicate_identity(self):
        self._make_two_groups_same_identity_key_fixture()

        dry_out = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', '--dry-run', stdout=dry_out, stderr=StringIO())
        dry_summary = _parse_cutover_summary(dry_out.getvalue())

        real_out = StringIO()
        with self.assertRaises(CommandError):
            call_command('cutover_classical_allocations', stdout=real_out, stderr=StringIO())
        real_summary = _parse_cutover_summary(real_out.getvalue())

        self.assertEqual(dry_summary, real_summary)
        self.assertEqual(dry_summary['runs_created'], 1)
        self.assertEqual(dry_summary['events_rekeyed'], 3)
        self.assertEqual(dry_summary['reasons'], {'duplicate_identity': 3})
