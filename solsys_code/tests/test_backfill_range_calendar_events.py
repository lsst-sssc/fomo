"""Tests for the backfill_range_calendar_events management command (D-07/FIX-08).

Covers: a qualifying APPROVED, site-resolved, range-window CampaignRun with no existing
CAMPAIGN:{pk}* CalendarEvent gets one dip-corrected event per night via delegation to
campaign_views._project_calendar_event(); non-qualifying runs (single-night, TBD,
unresolved-site, PENDING_REVIEW) get none; a re-run is idempotent (no duplicates);
--dry-run writes nothing; and a per-candidate sun_event() ValueError is reported and
skipped, never aborting the whole backfill run.

This module never fixtures an individual tom_targets.models.Target at all
(CampaignRun.target is nullable and left unset throughout), so CLAUDE.md's
non-sidereal-only target-factory convention doesn't arise here.
"""

from datetime import date
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.db.models import Q
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory


class TestBackfillRangeCalendarEvents(TestCase):
    """D-07/FIX-08: the one-off backfill command for already-APPROVED range-window runs."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        # Tier-1-resolvable ground site so sun_event() succeeds deterministically without a
        # live MPC call.
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

    def _make_approved_run(self, **overrides):
        """Create an APPROVED CampaignRun; kwargs override the default 4-night ground window."""
        kwargs = {
            'campaign': self.campaign,
            'telescope_instrument': 'FTN/MuSCAT3',
            'site': self.ground_site,
            'window_start': date(2026, 8, 1),
            'window_end': date(2026, 8, 4),
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    def _event_count(self, run):
        return CalendarEvent.objects.filter(
            Q(url=f'CAMPAIGN:{run.pk}') | Q(url__startswith=f'CAMPAIGN:{run.pk}:')
        ).count()

    def test_backfill_projects_per_night_events_for_qualifying_range_run(self):
        run = self._make_approved_run()
        self.assertEqual(self._event_count(run), 0)

        call_command('backfill_range_calendar_events', stdout=StringIO())

        self.assertEqual(self._event_count(run), 4)
        self.assertTrue(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}:2026-08-01').exists())
        self.assertTrue(CalendarEvent.objects.filter(url=f'CAMPAIGN:{run.pk}:2026-08-04').exists())

    def test_backfill_is_idempotent_on_second_run(self):
        run = self._make_approved_run()
        call_command('backfill_range_calendar_events', stdout=StringIO())
        self.assertEqual(self._event_count(run), 4)

        call_command('backfill_range_calendar_events', stdout=StringIO())

        self.assertEqual(self._event_count(run), 4)

    def test_backfill_skips_non_qualifying_runs(self):
        single_night_run = self._make_approved_run(window_start=date(2026, 8, 10), window_end=date(2026, 8, 10))
        tbd_run = self._make_approved_run(window_start=None, window_end=None)
        unresolved_site_run = self._make_approved_run(
            site=None, window_start=date(2026, 8, 20), window_end=date(2026, 8, 23)
        )
        pending_run = self._make_approved_run(
            window_start=date(2026, 8, 25),
            window_end=date(2026, 8, 28),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        qualifying_run = self._make_approved_run()

        call_command('backfill_range_calendar_events', stdout=StringIO())

        self.assertEqual(self._event_count(qualifying_run), 4)
        self.assertEqual(self._event_count(single_night_run), 0)
        self.assertEqual(self._event_count(tbd_run), 0)
        self.assertEqual(self._event_count(unresolved_site_run), 0)
        self.assertEqual(self._event_count(pending_run), 0)

    def test_backfill_dry_run_writes_nothing(self):
        run = self._make_approved_run()
        out = StringIO()

        call_command('backfill_range_calendar_events', '--dry-run', stdout=out)

        self.assertEqual(self._event_count(run), 0)
        self.assertIn(str(run.pk), out.getvalue())
        self.assertIn('would', out.getvalue().lower())

    def test_backfill_skips_and_continues_on_sun_event_valueerror(self):
        run_a = self._make_approved_run()
        run_b = self._make_approved_run(telescope_instrument='FTS/Spectral')

        with patch('solsys_code.campaign_views.sun_event', side_effect=ValueError('blank timezone')):
            call_command('backfill_range_calendar_events', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(self._event_count(run_a), 0)
        self.assertEqual(self._event_count(run_b), 0)
