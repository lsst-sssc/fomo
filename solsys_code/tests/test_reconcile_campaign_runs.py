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

Migrated for Phase 35 plan 35-02: D-10 inverted queue-sourced dispatch to the whole-window
`RUN:{pk}` container regardless of a resolved ground site, and a resolved-site, non-queue
run (`CLASSICAL_FILE`/`LEGACY`) now dispatches to the peer `allocation_projector` module's
`ALLOC:{pk}:{date}` namespace instead of `_reconcile_classical_nights()`'s retired
`RUN:{pk}:{date}` family. See 35-02-SUMMARY.md's classification table.
"""

import re
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from io import StringIO
from uuid import uuid4

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.allocation_projector import allocation_events
from solsys_code.campaign_reconciler import owned_events
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation
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

    def _link_placed_record(self, run: CampaignRun, *, scheduled_start: datetime, scheduled_end: datetime):
        """Create a linked, placed ObservationRecord (NonSiderealTargetFactory target --
        CLAUDE.md) whose block is expected to retire one of `run`'s allocation nights
        (D-05)."""
        target = NonSiderealTargetFactory.create()
        owner = User.objects.create(username=f'obs-owner-{uuid4().hex[:8]}')
        record = ObservationRecord.objects.create(
            target=target,
            user=owner,
            facility='LCO',
            observation_id=f'obs-{uuid4().hex[:8]}',
            status='COMPLETED',
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters={'proposal': 'TEST'},
        )
        return CampaignRunObservation.objects.create(run=run, observation_record=record)

    def _seed_mixed_runs(self) -> tuple[CampaignRun, CampaignRun, CampaignRun]:
        """One allocation-dispatched (resolved-site, non-queue) multi-night run, one
        queue-sourced run (D-10: container-dispatched regardless of its resolved site), and
        one class-wide run (also container-dispatched) -- exercising all three branches
        `reconcile_run()` can take."""
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

        # Sanity: every seeded run actually got at least one event out of the first sweep --
        # the classical (allocation-dispatched) run in the ALLOC: namespace, the other two
        # (both container-dispatched under D-10) in the RUN: namespace.
        self.assertGreaterEqual(allocation_events(classical_run).count(), 1)
        for run in (queue_run, class_wide_run):
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
    """RECON-06/D-06: one run's failure is reported by pk and the batch continues. All
    three runs are resolved-site, non-queue (CLASSICAL_FILE) -- allocation-dispatched under
    D-09, so the blank-timezone failure now surfaces from `ZoneInfo(run.site.timezone)`
    inside `project_allocation()` rather than the retired classical branch, and the
    surviving runs' events are asserted in the `ALLOC:` namespace."""

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

        self.assertEqual(allocation_events(run_a).count(), 1)
        self.assertEqual(allocation_events(run_b).count(), 0)
        self.assertEqual(allocation_events(run_c).count(), 1)


class TestRealDataShapeScenario(ReconcileCampaignRunsTestBase):
    """RECON-07: the measured real 8 QUEUE / 11 CLASSICAL / 0 SPACE split of 19 runs becomes
    fully calendar-visible in one command run (26-DECISION.md "Run-type inventory").

    D-10 (Phase 35) supersedes quick task 260805-tad's finding for this scenario: EVERY
    queue-sourced run now dispatches to the single whole-window `RUN:{pk}` container
    regardless of site resolution -- `source` alone decides for the four queue values, read
    directly off the stored field. The 11 classical (resolved-site, non-queue) runs are
    allocation-dispatched (`ALLOC:{pk}:{date}`, D-09)."""

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
            # D-10: every queue-sourced run gets exactly one bare container, regardless of
            # window length or site resolution.
            events = owned_events(run)
            self.assertEqual(events.count(), 1)
            self.assertEqual(events.get().url, f'RUN:{run.pk}')
            total_events_written += 1

        for run in class_wide_queue_runs:
            events = owned_events(run)
            self.assertEqual(events.count(), 1)
            self.assertEqual(events.get().url, f'RUN:{run.pk}')
            total_events_written += 1

        for run, n_nights in zip(classical_runs, classical_window_lengths, strict=True):
            events = allocation_events(run)
            self.assertEqual(events.count(), n_nights)
            for event in events:
                self.assertTrue(event.url.startswith(f'ALLOC:{run.pk}:'))
            total_events_written += n_nights

        for run in (*site_resolved_queue_runs, *class_wide_queue_runs):
            self.assertGreaterEqual(owned_events(run).count(), 1)
        for run in classical_runs:
            self.assertGreaterEqual(allocation_events(run).count(), 1)

        self.assertEqual(CalendarEventMeta.objects.filter(run__isnull=False).count(), total_events_written)


class TestSummaryCounters(ReconcileCampaignRunsTestBase):
    """WR-01/WR-03 (33-REVIEW.md) plus D-05/D-07/D-16 (Phase 35): the sweep surfaces
    `detached` (a `RUN:`-namespace event released to the attribution queue) and the two
    counters 35-01 added, `retired` and `rekeyed`, in both the real-run and dry-run summary
    lines.

    `skipped_nights` is now a permanently-zero field: the skip-the-night rule it counted
    (`_attributed_nights()`) has no caller left in `campaign_reconciler.py` after 35-01 --
    superseded outright by the handoff, which deletes (retires) a night rather than skipping
    it. The two tests that exercised it are retired (named reason, no destination module);
    a new pair of tests below proves `retired`/`rekeyed` instead, per Task 2's own
    instruction to add a case where a linked, placed record reports a non-zero `retired`."""

    def test_real_sweep_reports_retired_for_a_run_with_a_linked_placed_record(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.CLASSICAL_FILE)
        scheduled_start = datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=2)
        self._link_placed_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)

        out = StringIO()
        call_command('reconcile_campaign_runs', stdout=out)

        output = out.getvalue()
        self.assertIn(f'Run pk={run.pk}', output)
        self.assertIn('allocation night(s) were removed', output)
        summary = _parse_summary(output)
        self.assertEqual(summary['retired'], 1)
        self.assertEqual(summary['rekeyed'], 0)
        self.assertFalse(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{night.isoformat()}').exists())

    def test_dry_run_names_would_retire_and_would_rekey(self):
        """The dry-run line uses `would_retire`/`would_rekey` -- the preview vocabulary --
        rather than `retired`/`rekeyed`, but both underlying counters are the same
        `ReconcileResult` fields, and a legacy `RUN:{pk}:{night}` event is re-keyed
        (previewed) in place of a fresh mint."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.CLASSICAL_FILE)
        legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=datetime(2026, 8, 1, 9, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 19, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)

        out = StringIO()
        call_command('reconcile_campaign_runs', '--dry-run', stdout=out)

        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['would_retire'], 0)
        self.assertEqual(summary['would_rekey'], 1)
        # Regression (Task 1, Phase 35): a legacy night the projector's own takeover would
        # rekey must NOT also be double-counted by the new date-bearing delete preview --
        # caught by 35-06 Task 3's real-database proof run, where every legacy night due
        # for an in-place takeover was also being previewed as `would_delete_legacy`.
        self.assertEqual(summary['would_delete_legacy'], 0)
        # --dry-run never writes: the legacy event is still keyed under RUN:, not ALLOC:.
        self.assertTrue(CalendarEvent.objects.filter(url=f'RUN:{run.pk}:{night.isoformat()}').exists())
        self.assertFalse(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{night.isoformat()}').exists())

    def test_real_sweep_reports_legacy_deleted_for_a_superseded_container_run_night(self):
        """Task 1 (D-16, Phase 35): a container-dispatched (queue-sourced) run's leftover
        legacy `RUN:{pk}:{date}` event -- outside its own `active_urls` (always
        `{RUN:{pk}}`) -- is DELETED (not detached) on the next sweep, since the whole
        per-night `RUN:` family retires for a container-dispatched run."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.LCO_QUEUE)
        call_command('reconcile_campaign_runs', stdout=StringIO())
        self.assertTrue(CalendarEvent.objects.filter(url=f'RUN:{run.pk}').exists())

        legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)
        legacy_pk = legacy_event.pk

        out = StringIO()
        err = StringIO()
        call_command('reconcile_campaign_runs', stdout=out, stderr=err)

        self.assertIn(f'Run pk={run.pk}', out.getvalue())
        self.assertIn('leftover per-night event(s) deleted', out.getvalue())
        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['legacy_deleted'], 1)
        self.assertEqual(summary['detached'], 0)
        self.assertFalse(CalendarEvent.objects.filter(pk=legacy_pk).exists())
        self.assertFalse(CalendarEventMeta.objects.filter(event_id=legacy_pk).exists())

    def test_dry_run_previews_the_would_delete_legacy_count_and_writes_nothing(self):
        """Test 6 (Task 1): the dry-run and real summary lines both name `legacy_deleted`
        (the dry-run line as `would_delete_legacy`, the same underlying `ReconcileResult`
        field), and a dry run deletes nothing."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.LCO_QUEUE)
        call_command('reconcile_campaign_runs', stdout=StringIO())

        legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)

        event_count_before = CalendarEvent.objects.count()
        meta_count_before = CalendarEventMeta.objects.count()

        out = StringIO()
        call_command('reconcile_campaign_runs', '--dry-run', stdout=out)

        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['would_delete_legacy'], 1)
        self.assertEqual(summary['detach_declined'], 0)
        self.assertEqual(CalendarEvent.objects.count(), event_count_before)
        self.assertEqual(CalendarEventMeta.objects.count(), meta_count_before)

        real_out = StringIO()
        call_command('reconcile_campaign_runs', stdout=real_out)
        real_summary = _parse_summary(real_out.getvalue())
        self.assertEqual(real_summary['legacy_deleted'], 1)

    def test_dry_run_retired_message_says_would_be_removed_not_past_tense(self):
        """35-REVIEW.md WR-04: under --dry-run nothing has been written yet -- the per-run
        message must not claim a night was already removed."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.CLASSICAL_FILE)
        scheduled_start = datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=2)
        self._link_placed_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)

        out = StringIO()
        call_command('reconcile_campaign_runs', '--dry-run', stdout=out)

        output = out.getvalue()
        self.assertIn(f'Run pk={run.pk}: 1 allocation night(s) would be removed', output)
        self.assertNotIn('night(s) were removed', output)

    def test_retired_message_does_not_claim_a_single_cause(self):
        """35-REVIEW.md WR-05: `result.retired` is incremented from three unrelated causes
        (observation handoff, sub-night re-mint, window-shrink convergence) -- the message
        must not claim the observation-handoff cause specifically."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.CLASSICAL_FILE)
        scheduled_start = datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=2)
        self._link_placed_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)

        out = StringIO()
        call_command('reconcile_campaign_runs', stdout=out)

        output = out.getvalue()
        self.assertIn(f'Run pk={run.pk}: 1 allocation night(s) were removed', output)
        self.assertIn('observation handoff, sub-night re-mint or window change', output)
        self.assertNotIn('now covered by a real observation', output)

    def test_dry_run_rekeyed_message_says_would_be_re_keyed_not_past_tense(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.CLASSICAL_FILE)
        legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=datetime(2026, 8, 1, 9, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 19, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)

        out = StringIO()
        call_command('reconcile_campaign_runs', '--dry-run', stdout=out)

        output = out.getvalue()
        self.assertIn(f'Run pk={run.pk}: 1 legacy RUN:-keyed night(s) would be re-keyed into ALLOC:', output)
        self.assertNotIn('night(s) re-keyed into ALLOC:', output)

    def test_dry_run_legacy_deleted_message_says_would_be_deleted_not_past_tense(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.LCO_QUEUE)
        call_command('reconcile_campaign_runs', stdout=StringIO())
        legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)

        out = StringIO()
        call_command('reconcile_campaign_runs', '--dry-run', stdout=out)

        output = out.getvalue()
        self.assertIn(f'Run pk={run.pk}: 1 leftover per-night event(s) would be deleted', output)
        self.assertNotIn('event(s) deleted --', output)

    def test_real_sweep_reports_declined_for_a_human_confirmed_superseded_row(self):
        """33-10 Task 1 (UAT option B, 2026-09-09): a superseded RUN:-keyed row a human has
        already confirmed is left attributed, not detached -- the sweep reports it via the
        declined per-run line on stderr and the summary's `detach_declined` counter."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, source=CampaignRun.Source.LCO_QUEUE)
        call_command('reconcile_campaign_runs', stdout=StringIO())

        legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        staffer = User.objects.create(username='declined-counter-staffer')
        CalendarEventMeta.objects.create(
            event=legacy_event,
            run=run,
            confirmed_by=staffer,
            confirmed_at=datetime(2026, 8, 1, 9, 0, tzinfo=dt_timezone.utc),
        )

        out = StringIO()
        err = StringIO()
        call_command('reconcile_campaign_runs', stdout=out, stderr=err)

        error_output = err.getvalue()
        self.assertIn(f'Run pk={run.pk}', error_output)
        self.assertIn('left attributed', error_output)
        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['detached'], 0)
        self.assertEqual(summary['detach_declined'], 1)
        legacy_meta = CalendarEventMeta.objects.get(event=legacy_event)
        self.assertEqual(legacy_meta.run_id, run.pk)
        self.assertEqual(legacy_meta.confirmed_by_id, staffer.pk)

    def test_sweep_with_nothing_retired_or_detached_reports_zero_and_no_per_run_lines(self):
        self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 2))

        out = StringIO()
        err = StringIO()
        call_command('reconcile_campaign_runs', stdout=out, stderr=err)

        summary = _parse_summary(out.getvalue())
        self.assertEqual(summary['retired'], 0)
        self.assertEqual(summary['rekeyed'], 0)
        self.assertEqual(summary['detached'], 0)
        self.assertEqual(summary['legacy_deleted'], 0)
        self.assertNotIn('night(s) retired', out.getvalue())
        self.assertNotIn('event(s) detached', err.getvalue())
        self.assertNotIn('leftover per-night event(s) deleted', out.getvalue())

    def test_retired_and_legacy_deleted_are_both_reported_in_the_same_sweep(self):
        """The realistic mixed sweep this phase's summary line exists to surface: one
        allocation-dispatched run's night retires because a linked record was placed before
        its first reconcile, and a second, container-dispatched run's leftover legacy night
        is deleted (Task 1) -- one sweep, both counters non-zero."""
        retire_night = date(2026, 8, 1)
        retire_run = self._make_run(
            telescope_instrument='Retire run',
            window_start=retire_night,
            window_end=retire_night,
            source=CampaignRun.Source.CLASSICAL_FILE,
        )
        scheduled_start = datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc)
        self._link_placed_record(
            retire_run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2)
        )

        legacy_night = date(2026, 8, 2)
        legacy_run = self._make_run(
            telescope_instrument='Legacy-delete run',
            window_start=legacy_night,
            window_end=legacy_night,
            source=CampaignRun.Source.LCO_QUEUE,
        )
        call_command('reconcile_campaign_runs', stdout=StringIO())
        self.assertTrue(CalendarEvent.objects.filter(url=f'RUN:{legacy_run.pk}').exists())
        legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact',
            url=f'RUN:{legacy_run.pk}:{legacy_night.isoformat()}',
            start_time=datetime(2026, 8, 2, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 2, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=legacy_run)

        out = StringIO()
        err = StringIO()
        call_command('reconcile_campaign_runs', stdout=out, stderr=err)

        summary = _parse_summary(out.getvalue())
        self.assertGreaterEqual(summary['retired'], 1)
        self.assertEqual(summary['legacy_deleted'], 1)
        self.assertIn(f'Run pk={retire_run.pk}', out.getvalue())
        self.assertIn(f'Run pk={legacy_run.pk}', out.getvalue())
