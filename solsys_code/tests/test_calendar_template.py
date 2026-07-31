"""First view-level rendering test for tom_calendar's calendar.html override.

Asserts the DISPLAY-02/03 dashed-border + tooltip markers appear for fallback-labeled
events only, on both the all-day and timed render branches, and that a CalendarEvent
with no CalendarEventMeta sidecar row renders without raising (DISPLAY-01
read-side default, A1).

Phase 9 additions cover DISPLAY-04/05/06/07: proposal-color fills, [QUEUED] override
fix, status box-shadow rings, composition with Phase 8 dashed border, and the footer
legend with click-to-filter infrastructure.
"""

from datetime import date, datetime
from datetime import timezone as dt_timezone
from pathlib import Path

from django.contrib.auth.models import User
from django.db import connection
from django.test import Client, SimpleTestCase, TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils.formats import date_format
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.templatetags.calendar_display_extras import proposal_color, telescope_color, telescope_stripe_color

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
        CalendarEventMeta.objects.create(event=self.all_day_fallback, is_verified=False)

        self.all_day_verified = CalendarEvent.objects.create(
            title='All-day verified',
            start_time=datetime(2026, 6, 12, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 13, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=self.all_day_verified, is_verified=True)

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
        CalendarEventMeta.objects.create(event=self.timed_fallback, is_verified=False)

        self.timed_verified = CalendarEvent.objects.create(
            title='Timed verified',
            start_time=datetime(2026, 6, 17, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 17, 23, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=self.timed_verified, is_verified=True)

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
        CalendarEventMeta.objects.create(event=self.queued_fallback_timed, is_verified=False)

        # quick-260724-osc fixture: classical-schedule (empty-proposal) all-day event with
        # a telescope set — exercises the per-telescope left-edge stripe + legend.
        # June 1-2 is a free date range not touched by any other fixture above.
        self.classical_with_telescope = CalendarEvent.objects.create(
            title='Classical NTT run',
            proposal='',
            telescope='NTT',
            start_time=datetime(2026, 6, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 6, 2, 6, 0, tzinfo=dt_timezone.utc),
        )

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

    # --- quick-260724-osc: per-telescope left-edge stripe + legend ---

    def _event_div_html(self, content, event, window=500):
        """Return a slice of rendered HTML starting at the given event's update-event link.

        Isolates one event's markup so stripe assertions can be scoped to a single
        event's div rather than the whole page.
        """
        marker = f'/calendar/update/{event.id}/"'
        idx = content.index(marker)
        return content[idx : idx + window]

    def test_osc_classical_event_renders_telescope_stripe(self):
        """quick-260724-tiz: classical-schedule all-day event gets the cal-event-classical
        class and a --tel-color custom property (pseudo-element stripe, no inline border-left).

        quick-260724-vb0: --tel-color is fed from telescope_stripe_color()
        (TELESCOPE_STRIPE_PALETTE), not telescope_color() -- the stripe and the legend
        chip now resolve through different palettes gated against different backgrounds.
        """
        tel_hex = telescope_stripe_color('NTT')
        response = self._get_calendar()
        content = response.content.decode()
        div_html = self._event_div_html(content, self.classical_with_telescope)
        self.assertIn('cal-event-classical', div_html)
        self.assertIn(f'--tel-color: {tel_hex};', div_html)

    def test_osc_proposal_having_event_has_no_telescope_stripe(self):
        """quick-260724-tiz: proposal-having all-day events render neither the
        cal-event-classical class nor a --tel-color custom property."""
        response = self._get_calendar()
        content = response.content.decode()
        for event in (self.queued_event, self.terminal_event):
            with self.subTest(event=event.title):
                div_html = self._event_div_html(content, event)
                self.assertNotIn('cal-event-classical', div_html)
                self.assertNotIn('--tel-color', div_html)

    def test_tiz_legends_render_chip_swatches(self):
        """quick-260724-tiz: both legends render .cal-legend-chip swatches with the
        correct background-color, replacing the thin ▌ glyph."""
        tel_hex = telescope_color('NTT')
        prop_hex = proposal_color(self.queued_event.proposal)
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(f'<span class="cal-legend-chip" style="background-color: {tel_hex};">', content)
        self.assertIn(f'<span class="cal-legend-chip" style="background-color: {prop_hex};">', content)

    def test_osc_telescope_legend_renders_when_classical_event_visible(self):
        """quick-260724-osc: the display-only telescope legend renders and decodes NTT."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn('cal-legend-telescope', content)
        self.assertIn('NTT', content)

    def test_osc_telescope_legend_is_not_click_to_filter_wired(self):
        """quick-260724-osc: the telescope legend must not hook into the proposal
        click-to-filter JS (no data-proposal attribute, no cal-legend-swatch class)."""
        response = self._get_calendar()
        content = response.content.decode()
        # Skip the <style> block's own class-definition occurrence and find the
        # first rendered <span class="cal-legend-telescope..."> markup instance.
        body_start = content.index('</style>')
        idx = content.index('cal-legend-telescope', body_start)
        # Look at the opening <span ...> tag containing the class to confirm it
        # carries no data-proposal attribute and isn't also tagged cal-legend-swatch.
        tag_start = content.rindex('<span', 0, idx)
        tag_end = content.index('>', idx)
        tag_html = content[tag_start:tag_end]
        self.assertNotIn('data-proposal', tag_html)
        self.assertNotIn('cal-legend-swatch', tag_html)

    # --- quick-260724-vb0: two-palette split (legend vs stripe) ---

    def test_vb0_legend_and_stripe_render_different_hex_for_same_telescope(self):
        """quick-260724-vb0: the legend chip renders telescope_color() (TELESCOPE_PALETTE,
        gated against white) while the stripe renders telescope_stripe_color()
        (TELESCOPE_STRIPE_PALETTE, gated against the gray fill) -- for the same telescope
        name these must resolve to two different hex values, so a future accidental
        re-merge of the two paths fails loudly here rather than silently."""
        legend_hex = telescope_color('NTT')
        stripe_hex = telescope_stripe_color('NTT')
        self.assertNotEqual(legend_hex, stripe_hex)

        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(f'<span class="cal-legend-chip" style="background-color: {legend_hex};">', content)
        div_html = self._event_div_html(content, self.classical_with_telescope)
        self.assertIn(f'--tel-color: {stripe_hex};', div_html)


class EventModalCampaignRunLinkTest(TestCase):
    """Phase 27 Plan 05 (CANON-05/D-08/D-09/D-10): the event_form.html override links a
    calendar event back to its owning CampaignRun when the run is publicly visible, and
    renders nothing for a run that has not yet been approved -- for both a non-staff and a
    staff visitor.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.staff_user = User.objects.create_user(username='modalstaff', password='pw', is_staff=True)

        cls.approved_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 7, 4),
            window_end=date(2026, 7, 4),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.pending_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Should Stay Hidden Scope',
            window_start=date(2026, 7, 5),
            window_end=date(2026, 7, 5),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )

        cls.event_with_approved_run = CalendarEvent.objects.create(
            title='Event with approved run',
            start_time=datetime(2026, 7, 4, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 5, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_with_approved_run, run=cls.approved_run)

        cls.event_with_pending_run = CalendarEvent.objects.create(
            title='Event with pending run',
            start_time=datetime(2026, 7, 5, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 6, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_with_pending_run, run=cls.pending_run)

        cls.event_with_null_run = CalendarEvent.objects.create(
            title='Event with null run',
            start_time=datetime(2026, 7, 6, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 7, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_with_null_run, run=None)

        cls.event_with_no_meta_row = CalendarEvent.objects.create(
            title='Event with no companion row at all',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
        )

        # WR-04: a TBD run (window_start/window_end both NULL) linked to a
        # publicly-visible event must still render its telescope/instrument and
        # campaign link, but never the literal "(None-None)" window.
        cls.tbd_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='TBD Window Scope',
            window_start=None,
            window_end=None,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.event_with_tbd_run = CalendarEvent.objects.create(
            title='Event with TBD-window run',
            start_time=datetime(2026, 7, 8, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 9, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_with_tbd_run, run=cls.tbd_run)

    def _modal_url(self, event):
        return reverse('calendar:update-event', args=[event.id])

    def _campaign_table_href(self):
        return reverse('campaigns:table', args=[self.campaign.pk])

    def test_approved_run_shows_run_block_to_anonymous_visitor(self):
        response = self.client.get(self._modal_url(self.event_with_approved_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('FTN/MuSCAT3', content)
        self.assertIn(self._campaign_table_href(), content)

    def test_pending_run_shows_no_run_block_to_anonymous_visitor(self):
        response = self.client.get(self._modal_url(self.event_with_pending_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Should Stay Hidden Scope', content)
        self.assertNotIn(self._campaign_table_href(), content)

    def test_pending_run_shows_no_run_block_to_staff_visitor(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self._modal_url(self.event_with_pending_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Should Stay Hidden Scope', content)
        self.assertNotIn(self._campaign_table_href(), content)

    def test_null_run_companion_row_renders_200_with_no_run_block(self):
        response = self.client.get(self._modal_url(self.event_with_null_run))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(self._campaign_table_href(), response.content.decode())

    def test_no_companion_row_at_all_renders_200_with_no_exception(self):
        """The read-side default path (D-08): a conference/proposal-deadline event with no
        CalendarEventMeta row at all must render exactly as it does today."""
        response = self.client.get(self._modal_url(self.event_with_no_meta_row))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(self._campaign_table_href(), response.content.decode())

    def test_template_source_never_contains_pending_review_literal(self):
        """Asserted against the template file's own contents (source-level), not rendered
        output, so a future inline-literal regression is caught even if no test scenario
        happens to render it (D-10)."""
        template_path = (
            Path(__file__).resolve().parents[2] / 'src' / 'templates' / 'tom_calendar' / 'partials' / 'event_form.html'
        )
        content = template_path.read_text()
        self.assertNotIn('pending_review', content)

    def test_modal_renders_no_django_comment_delimiters(self):
        """The exact defect the UAT reporter saw: a multi-line {# ... #} block renders
        literally into the modal instead of being parsed as a comment. Covers both the
        approved-run modal and the TBD-window modal, since the third comment block sits
        inside the {% if run.is_publicly_visible %} branch and only that branch's
        rendering exercises it."""
        for event in (self.event_with_approved_run, self.event_with_tbd_run):
            with self.subTest(event=event.title):
                response = self.client.get(self._modal_url(event))
                self.assertEqual(response.status_code, 200)
                content = response.content.decode()
                self.assertNotIn('{#', content)
                self.assertNotIn('#}', content)
                self.assertNotIn('FOMO override of the upstream tom_calendar partial', content)

    def test_calendar_page_renders_no_django_comment_delimiters(self):
        """Covers FOMO's OTHER tom_calendar override (calendar.html), so the render-level
        assertion spans both surfaces the phase criterion names, not just the modal."""
        response = self.client.get(reverse('calendar:calendar'), {'year': 2026, 'month': 7})
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('{#', content)
        self.assertNotIn('#}', content)

    def test_tbd_run_renders_no_none_window(self):
        response = self.client.get(self._modal_url(self.event_with_tbd_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('TBD Window Scope', content)
        self.assertIn(self._campaign_table_href(), content)
        self.assertNotIn('(None', content)
        self.assertNotIn('None&ndash;None', content)

    def test_resolved_run_still_renders_its_window(self):
        """The window render must be byte-identical to before Task 1's Edit B for a run
        that has a resolved window -- only the TBD case changes."""
        response = self.client.get(self._modal_url(self.event_with_approved_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        expected = date_format(date(2026, 7, 4))
        self.assertIn(f'({expected}&ndash;{expected})', content)


class TemplateCommentSyntaxSweepTest(SimpleTestCase):
    """Repo-wide sweep for the class of defect fixed in event_form.html: Django's
    {# ... #} comment syntax is single-line only, so a multi-line block renders as
    literal text instead of being parsed as a comment. This makes the 27-UAT.md
    grep-based survey a permanent, automated guard rather than a one-off check.

    Scoped to src/templates/ because src/fomo/settings.py:96 names it as the only
    entry in TEMPLATES[0]['DIRS'], and no installed FOMO app ships its own
    templates/ directory (APP_DIRS=True is set, but solsys_code/ has no
    templates/ subdirectory) -- so "anywhere in the repo" and "everything under
    src/templates/" are the same search space today.
    """

    def test_no_multiline_django_comment_blocks_in_fomo_templates(self):
        repo_root = Path(__file__).resolve().parents[2]
        templates_root = repo_root / 'src' / 'templates'
        html_files = sorted(templates_root.rglob('*.html'))
        # Guard against a path typo silently making this test vacuously green.
        self.assertGreater(len(html_files), 0, f'No .html files found under {templates_root}')

        failures = []
        for html_file in html_files:
            content = html_file.read_text()
            search_from = 0
            while True:
                start = content.find('{#', search_from)
                if start == -1:
                    break
                end = content.find('#}', start + 2)
                if end == -1:
                    # WR-05: record the unterminated marker and keep scanning the SAME file
                    # from just past it. Breaking out here abandoned the rest of the file, so
                    # a single stray '{#' -- in a JS object literal, a CSS selector, or a
                    # {% verbatim %} block -- silently disabled this guard for every later
                    # comment block in that template, which is exactly the content that makes
                    # the heuristic fire in the first place.
                    line_no = content.count('\n', 0, start) + 1
                    failures.append(f'{html_file}:{line_no}: unterminated "{{#" (no matching "#}}")')
                    search_from = start + 2
                    continue
                span = content[start:end]
                if '\n' in span:
                    line_no = content.count('\n', 0, start) + 1
                    failures.append(f'{html_file}:{line_no}: multi-line {{# ... #}} block renders literally')
                search_from = end + 2

        self.assertEqual(failures, [], 'Multi-line Django comment blocks found:\n' + '\n'.join(failures))
