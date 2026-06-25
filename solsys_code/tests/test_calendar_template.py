"""First view-level rendering test for tom_calendar's calendar.html override.

Asserts the DISPLAY-02/03 dashed-border + tooltip markers appear for fallback-labeled
events only, on both the all-day and timed render branches, and that a CalendarEvent
with no CalendarEventTelescopeLabel sidecar row renders without raising (DISPLAY-01
read-side default, A1).
"""

from datetime import datetime
from datetime import timezone as dt_timezone

from django.test import Client, TestCase
from django.urls import reverse
from tom_calendar.models import CalendarEvent

from solsys_code.models import CalendarEventTelescopeLabel

DASHED_BORDER_MARKER = '2px dashed rgba(0, 0, 0, 0.65)'
TOOLTIP_SUBSTRING = 'estimate'


class CalendarTemplateTest(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        self.year = 2026
        self.month = 6

        # All-day branch: start/end dates differ.
        self.all_day_fallback = CalendarEvent.objects.create(
            title='All-day fallback',
            start_time=datetime(2026, 6, 10, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 11, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventTelescopeLabel.objects.create(event=self.all_day_fallback, is_verified=False)

        self.all_day_verified = CalendarEvent.objects.create(
            title='All-day verified',
            start_time=datetime(2026, 6, 12, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 13, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventTelescopeLabel.objects.create(event=self.all_day_verified, is_verified=True)

        self.all_day_no_row = CalendarEvent.objects.create(
            title='All-day no sidecar row',
            start_time=datetime(2026, 6, 14, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 15, 6, 0, tzinfo=dt_timezone.utc),
        )

        # Timed branch: start/end share the same date.
        self.timed_fallback = CalendarEvent.objects.create(
            title='Timed fallback',
            start_time=datetime(2026, 6, 16, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 16, 23, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventTelescopeLabel.objects.create(event=self.timed_fallback, is_verified=False)

        self.timed_verified = CalendarEvent.objects.create(
            title='Timed verified',
            start_time=datetime(2026, 6, 17, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 17, 23, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventTelescopeLabel.objects.create(event=self.timed_verified, is_verified=True)

        self.timed_no_row = CalendarEvent.objects.create(
            title='Timed no sidecar row',
            start_time=datetime(2026, 6, 18, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 18, 23, 0, tzinfo=dt_timezone.utc),
        )

        # The all-day fallback event spans 2 calendar days (Jun 10-11), so the calendar
        # view's day-cell bucketing (offset_date(start) <= d <= offset_date(end)) renders
        # it once per day cell it touches; the timed fallback event renders exactly once.
        self.num_fallback_day_cell_occurrences = 2 + 1

    def _get_calendar(self):
        return self.client.get(reverse('calendar:calendar'), {'year': self.year, 'month': self.month})

    def test_calendar_renders_200_including_no_sidecar_row_events(self):
        """Proves the silenced DoesNotExist path (A1): no-row events don't 500."""
        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)

    def test_fallback_events_get_dashed_border_and_tooltip(self):
        response = self._get_calendar()
        self.assertContains(response, DASHED_BORDER_MARKER)
        self.assertContains(response, TOOLTIP_SUBSTRING)

    def test_dashed_border_count_matches_fallback_event_count_only(self):
        """Verified and no-sidecar-row events (all-day and timed) must NOT get the dashed border.

        The all-day fallback event spans 2 day cells, so it contributes 2 occurrences of the
        marker on its own; the timed fallback event contributes exactly 1. Verified and
        no-sidecar-row events (both branches) must contribute 0.
        """
        response = self._get_calendar()
        content = response.content.decode()
        self.assertEqual(content.count(DASHED_BORDER_MARKER), self.num_fallback_day_cell_occurrences)
