"""Command-level tests for reconcile_campaign_runs (plan 29-03, Task 2).

Covers RECON-01 (idempotency), RECON-06 (--dry-run parity and per-run failure isolation)
and RECON-07 (the measured real 3I/ATLAS 8 QUEUE / 11 CLASSICAL / 0 SPACE split becomes
calendar-visible in one command run), all expressed via django.core.management.call_command
against StringIO-captured stdout/stderr, mirroring test_backfill_range_calendar_events.py's
shape (that command is deleted in plan 29-04).

Per research Pitfall 1 (29-RESEARCH.md): the real dev-DB rows for this split all carry
source='legacy' today, so TestRealDataShapeScenario sets `source` explicitly on every
fixture row and proves the reconciler's branch dispatch is correct given correct data --
it is not evidence that the live DB renders correctly (that is plan 29-07's D-07 checkpoint).

This module never fixtures an individual tom_targets.models.Target (CampaignRun.target is
nullable and left unset throughout), so CLAUDE.md's non-sidereal-only target-factory
convention doesn't arise here.
"""

import re
from datetime import date, timedelta
from io import StringIO

from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.campaign_reconciler import owned_events
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory


def _parse_summary(output: str) -> dict[str, int]:
    """Parse the command's final ``Done[...]. key: N, key: N, ...`` line into a dict.

    Works for both the real-run and --dry-run summary shapes (the label vocabulary differs
    -- created/would_create etc. -- but both are ``label: integer`` pairs on one line).
    """
    lines = [line for line in output.strip().splitlines() if line.startswith('Done')]
    assert lines, f'no summary line found in output: {output!r}'
    return {key: int(value) for key, value in re.findall(r'(\w+):\s*(-?\d+)', lines[-1])}


class ReconcileCampaignRunsTestBase(TestCase):
    """Shared fixture: one campaign and one Tier-1-resolvable ground Observatory."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.ground_site = Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

    def _make_run(self, **overrides) -> CampaignRun:
        kwargs = {
            'campaign': self.campaign,
            'telescope_instrument': 'FTN/MuSCAT3',
            'site': self.ground_site,
            'site_raw': 'F65',
            'window_start': date(2026, 8, 1),
            'window_end': date(2026, 8, 1),
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    def _seed_mixed_runs(self) -> tuple[CampaignRun, CampaignRun, CampaignRun]:
        """One classical multi-night run and one queue-sourced run -- both site-resolved and
        both taking the per-night branch (260805-tad: `source` does not change the calendar
        shape) -- plus one class-wide run, which is the only one of the three that gets a
        single whole-window container."""
        classical_run = self._make_run(
            telescope_instrument='FTN/MuSCAT3 classical',
            source=CampaignRun.Source.CLASSICAL_FILE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 3),
        )
        queue_run = self._make_run(
            telescope_instrument='FTS/Spectral queue',
            source=CampaignRun.Source.LCO_QUEUE,
            window_start=date(2026, 8, 5),
            window_end=date(2026, 8, 6),
        )
        class_wide_run = self._make_run(
            telescope_instrument='LCO 1m0 network',
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
            window_start=date(2026, 8, 12),
            window_end=date(2026, 8, 20),
        )
        return classical_run, queue_run, class_wide_run


class TestIdempotency(ReconcileCampaignRunsTestBase):
    """RECON-01: a second identical sweep reports zero created/updated with no modified churn."""

    def test_second_sweep_reports_zero_created_and_zero_updated_with_no_modified_churn(self):
        classical_run, queue_run, class_wide_run = self._seed_mixed_runs()

        first_out = StringIO()
        call_command('reconcile_campaign_runs', stdout=first_out)
        first_summary = _parse_summary(first_out.getvalue())
        self.assertGreater(first_summary['created'], 0)

        count_after_first = CalendarEvent.objects.count()
        modified_by_pk = {event.pk: event.modified for event in CalendarEvent.objects.all()}
        self.assertGreater(len(modified_by_pk), 0)

        second_out = StringIO()
        call_command('reconcile_campaign_runs', stdout=second_out)
        second_summary = _parse_summary(second_out.getvalue())

        self.assertEqual(second_summary['created'], 0)
        self.assertEqual(second_summary['updated'], 0)
        self.assertEqual(CalendarEvent.objects.count(), count_after_first)
        for event in CalendarEvent.objects.all():
            self.assertEqual(event.modified, modified_by_pk[event.pk])

        # Sanity: every seeded run actually got at least one event out of the first sweep.
        for run in (classical_run, queue_run, class_wide_run):
            self.assertGreaterEqual(owned_events(run).count(), 1)


class TestDryRun(ReconcileCampaignRunsTestBase):
    """RECON-06: --dry-run reports the same counts the real run would, and writes nothing."""

    def test_dry_run_matches_real_run_and_writes_nothing(self):
        self._seed_mixed_runs()

        dry_first_out = StringIO()
        call_command('reconcile_campaign_runs', '--dry-run', stdout=dry_first_out)
        dry_first_summary = _parse_summary(dry_first_out.getvalue())

        self.assertEqual(CalendarEvent.objects.count(), 0)
        self.assertEqual(CalendarEventMeta.objects.count(), 0)

        real_first_out = StringIO()
        call_command('reconcile_campaign_runs', stdout=real_first_out)
        real_first_summary = _parse_summary(real_first_out.getvalue())

        # The preview path and the write path agree on the first (never-reconciled) sweep.
        self.assertEqual(dry_first_summary['would_create'], real_first_summary['created'])

        real_second_out = StringIO()
        call_command('reconcile_campaign_runs', stdout=real_second_out)
        real_second_summary = _parse_summary(real_second_out.getvalue())
        self.assertEqual(real_second_summary['created'], 0)

        dry_second_out = StringIO()
        call_command('reconcile_campaign_runs', '--dry-run', stdout=dry_second_out)
        dry_second_summary = _parse_summary(dry_second_out.getvalue())

        self.assertEqual(dry_second_summary['would_create'], 0)
        self.assertEqual(dry_second_summary['would_update'], real_second_summary['updated'])
        self.assertEqual(dry_second_summary['would_leave_unchanged'], real_second_summary['unchanged'])
        # --dry-run must never write, even against already-reconciled state.
        self.assertEqual(CalendarEvent.objects.count(), real_first_summary['created'])
        self.assertEqual(CalendarEventMeta.objects.count(), real_first_summary['created'])


class TestFailureIsolation(ReconcileCampaignRunsTestBase):
    """RECON-06/D-06: one run's failure is reported by pk and the batch continues."""

    def test_middle_run_failure_is_reported_and_the_other_two_still_reconcile(self):
        blank_tz_site = Observatory.objects.create(
            obscode='T99',
            name='Blank Timezone Site',
            short_name='BTS',
            lat=-30.0,
            lon=149.0,
            altitude=1000.0,
            timezone='',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        run_a = self._make_run(
            telescope_instrument='FTN/MuSCAT3 run A',
            source=CampaignRun.Source.CLASSICAL_FILE,
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 1),
        )
        run_b = self._make_run(
            telescope_instrument='FTN/MuSCAT3 run B (blank tz)',
            source=CampaignRun.Source.CLASSICAL_FILE,
            site=blank_tz_site,
            site_raw='T99',
            window_start=date(2026, 9, 2),
            window_end=date(2026, 9, 2),
        )
        run_c = self._make_run(
            telescope_instrument='FTN/MuSCAT3 run C',
            source=CampaignRun.Source.CLASSICAL_FILE,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
        )

        out = StringIO()
        err = StringIO()
        call_command('reconcile_campaign_runs', stdout=out, stderr=err)

        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['failed'], 1)
        self.assertIn(f'Run pk={run_b.pk}', err.getvalue())

        self.assertEqual(owned_events(run_a).count(), 1)
        self.assertEqual(owned_events(run_b).count(), 0)
        self.assertEqual(owned_events(run_c).count(), 1)


class TestRealDataShapeScenario(ReconcileCampaignRunsTestBase):
    """RECON-07: the measured real 8 QUEUE / 11 CLASSICAL / 0 SPACE split of 19 runs becomes
    fully calendar-visible in one command run (26-DECISION.md "Run-type inventory").

    Corrected by quick task 260805-tad: queue-sourcing alone no longer determines the
    calendar shape -- `telescope_class`/`site` do. Of the 8 queue-sourced runs, 5 mirror the
    real site-resolved ESO VLT queue rows (a fixed, resolved, non-satellite site -> one
    per-night event each) and 3 mirror the real LCO class-wide queue allocations (no fixed
    site -> one whole-window container each), matching the real mix recorded in
    29-06-SUMMARY.md."""

    def test_19_run_fixture_matching_the_real_split_becomes_calendar_visible(self):
        site_resolved_queue_runs = []
        for i in range(5):
            source = CampaignRun.Source.LCO_QUEUE if i % 2 == 0 else CampaignRun.Source.ESO_QUEUE
            window_start = date(2026, 10, 1) + timedelta(days=3 * i)
            window_end = window_start + timedelta(days=1)
            run = self._make_run(
                telescope_instrument=f'Site-resolved queue run {i}',
                source=source,
                window_start=window_start,
                window_end=window_end,
            )
            site_resolved_queue_runs.append(run)

        class_wide_queue_runs = []
        for i in range(3):
            run = self._make_run(
                telescope_instrument=f'Class-wide queue run {i}',
                source=CampaignRun.Source.LCO_QUEUE,
                site=None,
                site_raw='',
                telescope_class=CampaignRun.TelescopeClass.ONE_M0,
                window_start=date(2026, 10, 20) + timedelta(days=5 * i),
                window_end=date(2026, 10, 20) + timedelta(days=5 * i + 4),
            )
            class_wide_queue_runs.append(run)

        classical_runs = []
        classical_window_lengths = []
        for i in range(11):
            n_nights = 2 if i < 3 else 1
            window_start = date(2026, 11, 1) + timedelta(days=3 * i)
            window_end = window_start + timedelta(days=n_nights - 1)
            run = self._make_run(
                telescope_instrument=f'Classical run {i}',
                source=CampaignRun.Source.CLASSICAL_FILE,
                window_start=window_start,
                window_end=window_end,
            )
            classical_runs.append(run)
            classical_window_lengths.append(n_nights)

        out = StringIO()
        call_command('reconcile_campaign_runs', stdout=out)
        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['runs'], 19)
        self.assertEqual(summary['failed'], 0)
        self.assertEqual(summary['skipped'], 0)
        self.assertEqual(summary['blocked'], 0)

        total_events_written = 0
        for run in site_resolved_queue_runs:
            expected_n = (run.window_end - run.window_start).days + 1
            events = owned_events(run)
            self.assertEqual(events.count(), expected_n)
            for event in events:
                self.assertTrue(event.url.startswith(f'RUN:{run.pk}:'))
            total_events_written += expected_n

        for run in class_wide_queue_runs:
            events = owned_events(run)
            self.assertEqual(events.count(), 1)
            self.assertEqual(events.get().url, f'RUN:{run.pk}')
            total_events_written += 1

        for run, n_nights in zip(classical_runs, classical_window_lengths, strict=True):
            events = owned_events(run)
            self.assertEqual(events.count(), n_nights)
            for event in events:
                self.assertTrue(event.url.startswith(f'RUN:{run.pk}:'))
            total_events_written += n_nights

        for run in (*site_resolved_queue_runs, *class_wide_queue_runs, *classical_runs):
            self.assertGreaterEqual(owned_events(run).count(), 1)

        self.assertEqual(CalendarEventMeta.objects.filter(run__isnull=False).count(), total_events_written)
