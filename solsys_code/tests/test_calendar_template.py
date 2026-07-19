"""First view-level rendering test for tom_calendar's calendar.html override.

Asserts the DISPLAY-02/03 dashed-border + tooltip markers appear for fallback-labeled
events only, on both the all-day and timed render branches, and that a CalendarEvent
with no CalendarEventTelescopeLabel sidecar row renders without raising (DISPLAY-01
read-side default, A1).

Phase 9 additions cover DISPLAY-04/05/06/07: proposal-color fills, [QUEUED] override
fix, status box-shadow rings, composition with Phase 8 dashed border, and the footer
legend with click-to-filter infrastructure.
"""

from datetime import datetime
from datetime import timezone as dt_timezone

from django.db import connection
from django.test import Client, TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from tom_calendar.models import CalendarEvent

from solsys_code.models import CalendarEventTelescopeLabel
from solsys_code.templatetags.calendar_display_extras import proposal_color

DASHED_BORDER_MARKER = '2px dashed rgba(0, 0, 0, 0.65)'
TOOLTIP_SUBSTRING = 'estimate'

# Phase 9 marker constants (DISPLAY-05/06) — note: NO trailing semicolon so these work
# as substring matches against the CSS the tags emit (which does include the semicolon).
QUEUED_BOX_SHADOW = 'box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.45)'
TERMINAL_BOX_SHADOW = 'box-shadow: 0 0 0 3px rgba(160, 0, 0, 0.55)'
# This is the old [QUEUED] background-color override that DISPLAY-05 requires removing.
# Note: assert the full `background-color:` prefix — the new queued box-shadow
# legitimately contains the bare rgba value as a substring (see plan Task 3 note).
OLD_QUEUED_GREY = 'background-color: rgba(0, 0, 0, 0.45)'
NEUTRAL_HEX = '#5a6268'


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

        # Phase 9 fixtures — proposal-color, status rings, composition (DISPLAY-04/05/06/07).
        # All use June 2026 dates not already taken by Phase 8 fixtures above.
        self.queued_event = CalendarEvent.objects.create(
            title='[QUEUED] LTP2025A run',
            proposal='LTP2025A-004',
            start_time=datetime(2026, 6, 20, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 21, 6, 0, tzinfo=dt_timezone.utc),
        )

        self.terminal_event = CalendarEvent.objects.create(
            title='[FAILED] LTP2025B run',
            proposal='LTP2025B-012',
            start_time=datetime(2026, 6, 22, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 23, 6, 0, tzinfo=dt_timezone.utc),
        )

        # Timed event with a proposal — exercises the timed proposal bullet (DISPLAY-04 both-branches).
        self.timed_with_proposal = CalendarEvent.objects.create(
            title='LTP2025A timed run',
            proposal='LTP2025A-004',
            start_time=datetime(2026, 6, 25, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 25, 11, 0, tzinfo=dt_timezone.utc),
        )

        # Empty-proposal all-day event — exercises the neutral slot (DISPLAY-04, DISPLAY-07).
        self.no_proposal_event = CalendarEvent.objects.create(
            title='Classical block',
            proposal='',
            start_time=datetime(2026, 6, 24, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 25, 6, 0, tzinfo=dt_timezone.utc),
        )

        # Pitfall 3 composition fixture: queued AND fallback-labeled timed event.
        # Carries both the QUEUED box-shadow ring AND the Phase 8 dashed border.
        # Contributes exactly 1 additional day-cell occurrence of DASHED_BORDER_MARKER.
        self.queued_fallback_timed = CalendarEvent.objects.create(
            title='[QUEUED] fallback run',
            proposal='LTP2025A-004',
            start_time=datetime(2026, 6, 27, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 27, 11, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventTelescopeLabel.objects.create(event=self.queued_fallback_timed, is_verified=False)

        # The all-day fallback event spans 2 calendar days (Jun 10-11), so the calendar
        # view's day-cell bucketing (offset_date(start) <= d <= offset_date(end)) renders
        # it once per day cell it touches; the timed fallback event renders exactly once;
        # queued_fallback_timed (Phase 9) is a timed fallback event contributing exactly 1.
        self.num_fallback_day_cell_occurrences = 2 + 1 + 1

    def _get_calendar(self):
        return self.client.get(reverse('calendar:calendar'), {'year': self.year, 'month': self.month})

    def test_calendar_renders_200_including_no_sidecar_row_events(self):
        """Proves the silenced DoesNotExist path (A1): no-row events don't 500."""
        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)

    def test_calendar_partial_data_url_carries_utc_offset(self):
        """Regression for BUGFIX-CAL-UTC: the calRefresh reload URL must carry utc_offset.

        A non-zero offset proves the user's actual selection is threaded through the
        data-url (not just that a literal '0' happens to appear).
        """
        response = self.client.get(
            reverse('calendar:calendar'), {'year': self.year, 'month': self.month, 'utc_offset': 5}
        )
        url = reverse('calendar:calendar')
        self.assertContains(response, f'data-url="{url}?month=6&year=2026&utc_offset=5"')

    def test_fallback_events_get_dashed_border_and_tooltip(self):
        response = self._get_calendar()
        self.assertContains(response, DASHED_BORDER_MARKER)
        self.assertContains(response, TOOLTIP_SUBSTRING)

    def test_dashed_border_count_matches_fallback_event_count_only(self):
        """Verified and no-sidecar-row events (all-day and timed) must NOT get the dashed border.

        The all-day fallback event spans 2 day cells, so it contributes 2 occurrences of the
        marker on its own; the timed fallback event contributes exactly 1; the Phase 9
        queued_fallback_timed event (is_verified=False) contributes 1 more. Verified and
        no-sidecar-row events (both branches) must contribute 0.
        """
        response = self._get_calendar()
        content = response.content.decode()
        self.assertEqual(content.count(DASHED_BORDER_MARKER), self.num_fallback_day_cell_occurrences)

    # --- Phase 9 tests: DISPLAY-04/05/06/07 ---

    def test_display05_old_queued_grey_background_color_is_gone(self):
        """DISPLAY-05: the flat-grey [QUEUED] background-color override no longer appears.

        Asserts the full 'background-color: rgba(0, 0, 0, 0.45)' string is absent.
        The new queued box-shadow legitimately contains the bare rgba value as a substring,
        so only the background-color-prefixed form is checked here (plan Task 3 note, D-05).
        """
        response = self._get_calendar()
        content = response.content.decode()
        self.assertNotIn(OLD_QUEUED_GREY, content)

    def test_display05_queued_event_renders_proposal_background_color(self):
        """DISPLAY-05: [QUEUED] all-day event keeps its proposal-keyed background-color."""
        qhex = proposal_color('LTP2025A-004')
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(f'background-color: {qhex}', content)

    def test_display04_neutral_slot_color_present_for_empty_proposal_event(self):
        """DISPLAY-04: empty-proposal event renders the neutral slot color (#5a6268)."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(NEUTRAL_HEX, content)

    def test_display04_timed_proposal_bullet_rendered(self):
        """DISPLAY-04 (timed branch): timed event with proposal gets a proposal-color bullet."""
        qhex = proposal_color('LTP2025A-004')
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(f'color: {qhex}', content)

    def test_display06_queued_box_shadow_present(self):
        """DISPLAY-06: [QUEUED] events carry the 2px queued ring."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(QUEUED_BOX_SHADOW, content)

    def test_display06_terminal_box_shadow_present(self):
        """DISPLAY-06: terminal-failure events carry the 3px red ring."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(TERMINAL_BOX_SHADOW, content)

    def test_display06_queued_and_terminal_rings_are_visually_distinct(self):
        """DISPLAY-06: the two status rings must be different strings (visual distinction)."""
        self.assertNotEqual(QUEUED_BOX_SHADOW, TERMINAL_BOX_SHADOW)

    def test_display06_pitfall3_composition_dashed_and_queued_coexist(self):
        """DISPLAY-06 + Pitfall 3: queued_fallback_timed carries BOTH the dashed border
        (Phase 8 is_verified=False) AND the queued box-shadow ring (Phase 9 status)."""
        response = self._get_calendar()
        content = response.content.decode()
        # Both signals coexist — Phase 8 signal not overwritten by Phase 9 status.
        self.assertIn(DASHED_BORDER_MARKER, content)
        self.assertIn(QUEUED_BOX_SHADOW, content)
        # Exact count: 2 (all_day_fallback spans 2 days) + 1 (timed_fallback) + 1 (queued_fallback_timed)
        self.assertEqual(content.count(DASHED_BORDER_MARKER), self.num_fallback_day_cell_occurrences)

    def test_display07_legend_swatch_markup_present(self):
        """DISPLAY-07: the footer proposal legend contains .cal-legend-swatch elements."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn('cal-legend-swatch', content)

    def test_display07_classical_schedule_label_present_when_empty_proposal_events_visible(self):
        """DISPLAY-07 D-06: the neutral-slot legend entry 'Classical schedule' appears
        because no_proposal_event (proposal='') is visible this month."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn('Classical schedule', content)

    # --- Phase 12 tests: DISPLAY-08/09 ---

    def test_display08_inline_text_color_present_for_all_day_events(self):
        """DISPLAY-08: all-day event divs carry an inline computed text color."""
        # DISPLAY-08: palette colors are dark, so computed text color is #fff.
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn('color: #fff', content)

    def test_display08_important_color_rule_absent(self):
        """DISPLAY-08: the hardcoded !important color override no longer appears in the page."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertNotIn('color: #fff !important', content)

    def test_display09_query_count_bounded(self):
        """DISPLAY-09: query count does not grow when additional CalendarEvents are added."""
        # Baseline: count queries with setUp fixtures already present.
        with CaptureQueriesContext(connection) as baseline_ctx:
            self._get_calendar()
        baseline_count = len(baseline_ctx)

        # Add one more CalendarEvent in the visible month and recount.
        CalendarEvent.objects.create(
            title='Extra event for N+1 test',
            start_time=datetime(2026, 6, 28, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 29, 6, 0, tzinfo=dt_timezone.utc),
        )
        with CaptureQueriesContext(connection) as extra_ctx:
            self._get_calendar()

        # DISPLAY-09: query count must not grow with additional events.
        self.assertEqual(len(extra_ctx), baseline_count)

    def test_display09_active_todo_count_renders_in_event_title(self):
        """DISPLAY-09: active_todo_count annotation still shows todo parenthetical."""
        from tom_calendar.models import EventTodo

        # Create an event with an incomplete todo so the count parenthetical renders.
        event_with_todo = CalendarEvent.objects.create(
            title='Event with todo',
            start_time=datetime(2026, 6, 28, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 29, 6, 0, tzinfo=dt_timezone.utc),
        )
        EventTodo.objects.create(event=event_with_todo, description='Test task', is_completed=False)

        response = self._get_calendar()
        content = response.content.decode()
        # DISPLAY-09: the todo count parenthetical must appear in the rendered output.
        self.assertIn('(1)', content)
