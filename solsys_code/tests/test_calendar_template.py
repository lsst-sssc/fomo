"""First view-level rendering test for tom_calendar's calendar.html override.

Asserts the DISPLAY-02/03 dashed-border + tooltip markers appear for fallback-labeled
events only, on both the all-day and timed render branches, and that a CalendarEvent
with no CalendarEventMeta sidecar row renders without raising (DISPLAY-01
read-side default, A1).

Phase 9 additions cover DISPLAY-04/05/06/07: proposal-color fills, [QUEUED] override
fix, status box-shadow rings, composition with Phase 8 dashed border, and the footer
legend with click-to-filter infrastructure.
"""

import difflib
import re
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from pathlib import Path

import tom_calendar
from django.contrib.auth.models import User
from django.db import connection
from django.db.models.signals import m2m_changed, post_save
from django.test import Client, SimpleTestCase, TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from django.utils.formats import date_format
from django.utils.html import escape
from tom_calendar.models import CalendarEvent, EventTodo
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.allocation_projector import ALLOC_URL_NAMESPACE
from solsys_code.campaign_reconciler import RUN_URL_NAMESPACE
from solsys_code.models import NO_CAMPAIGN_LABEL, CalendarEventMeta, CampaignRun
from solsys_code.observation_projector import receiver_on_group_membership_changed, receiver_on_record_save
from solsys_code.templatetags.calendar_display_extras import (
    observation_status_legend,
    proposal_color,
    telescope_color,
    telescope_stripe_color,
)

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
            # Phase 37 STATUS-01: [F] is the final short-letter marker; the legacy
            # bracket-word [FAILED] form was retired in plan 37-07 once a re-title sweep
            # proved the developer database held none of it.
            title='[F] LTP2025B run',
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

    def test_display07_no_proposal_label_present_when_empty_proposal_events_visible(self):
        """DISPLAY-07 (relabelled by F12, quick task 261006-lsf): the neutral-slot legend entry
        'No proposal recorded' appears because no_proposal_event (proposal='') is visible this
        month, and the v1.4 'Classical schedule' wording no longer does."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn('No proposal recorded', content)
        self.assertNotIn('Classical schedule', content)

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

        # Phase 33 (ANNOT-02, D-11): an approved run whose campaign is None must still
        # render its decoration -- table_url is None (no href), campaign_name is
        # NO_CAMPAIGN_LABEL.
        cls.no_campaign_run = CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            window_start=date(2026, 7, 10),
            window_end=date(2026, 7, 10),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.event_with_no_campaign_run = CalendarEvent.objects.create(
            title='Event with no-campaign run',
            start_time=datetime(2026, 7, 10, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 11, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_with_no_campaign_run, run=cls.no_campaign_run)

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

    def test_approved_run_shows_attributed_label_and_anchored_campaign_link(self):
        """D-13/D-17: the block's label reads 'Attributed campaign run' (never 'owned'),
        and the campaign-table link is anchored to this run's row."""
        response = self.client.get(self._modal_url(self.event_with_approved_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('Attributed campaign run', content)
        self.assertIn(f'{self._campaign_table_href()}#run-{self.approved_run.pk}', content)

    def test_no_campaign_run_renders_200_with_telescope_instrument_and_no_campaign_link(self):
        """D-11/T-33-01: a run with no campaign still decorates (telescope/instrument,
        run status), but table_url is None so no campaign-table link is emitted -- the
        modal must never raise NoReverseMatch on a null campaign pk."""
        response = self.client.get(self._modal_url(self.event_with_no_campaign_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('NTT/EFOSC2', content)
        self.assertIn('Attributed campaign run', content)
        self.assertNotIn('View campaign', content)
        self.assertNotIn(self._campaign_table_href(), content)

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
        # ANNOT-02: a pending-review run's event renders no decoration for any visitor,
        # staff included.
        self.assertNotIn('Attributed campaign run', content)

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


class EventModalRunTallyTest(TestCase):
    """Phase 37 Plan 06 (D-09, TALLY-01): the event_form.html override renders the run's
    live tally inside the same attributed-run block the campaign_decoration() gate already
    applies -- an anonymous visitor sees it for an approved run's event, and sees no
    attributed-run block at all for a pending-review run's event.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Run Tally Modal Campaign')
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
            title='Event with approved run tally',
            start_time=datetime(2026, 7, 4, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 5, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_with_approved_run, run=cls.approved_run)

        cls.event_with_pending_run = CalendarEvent.objects.create(
            title='Event with pending run tally',
            start_time=datetime(2026, 7, 5, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 6, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_with_pending_run, run=cls.pending_run)

    def _modal_url(self, event):
        return reverse('calendar:update-event', args=[event.id])

    def test_approved_run_shows_tally_to_anonymous_visitor(self):
        response = self.client.get(self._modal_url(self.event_with_approved_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('0 groups', content)
        self.assertIn('0 records', content)
        self.assertIn('[O]', content)
        self.assertIn('[S]', content)
        self.assertIn('[X/F]', content)
        self.assertIn('[U]', content)

    def test_pending_review_run_shows_no_attributed_run_block_at_all(self):
        """T-37-21: the whole {% if deco %} block -- including the tally sub-line -- must
        not render at all for a pending-review run, for an anonymous visitor."""
        response = self.client.get(self._modal_url(self.event_with_pending_run))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Attributed campaign run', content)
        self.assertNotIn('groups', content)

    def test_not_yet_known_unused_figure_renders_a_word_not_a_zero(self):
        """The approved run has no allocation events and no proposal code -- the unused
        segment must render as not-yet-known text, never a bare zero."""
        response = self.client.get(self._modal_url(self.event_with_approved_run))
        content = response.content.decode()
        self.assertIn('not yet known', content)


class MonthCellUnusedNightRenderTest(TestCase):
    """Phase 37 Plan 06 (UNUSED-01, D-12/D-13/D-14): both of calendar.html's event loops
    call unused_night_decoration() and render its two channels -- the cal-event-unused
    style class plus the [U] text token -- for an elapsed, still-standing allocation night,
    and neither channel for a future one. The token is added at render time only; the
    stored CalendarEvent.title never changes.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Unused Night Month View')
        cls.active_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='NTT/EFOSC2',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )

        # All-day branch: an overnight span, safely elapsed relative to the real clock
        # (system date is far past August 2026 by the time this test runs). Blank
        # proposal -> neutral-slot color -> the cal-event-classical branch, so the
        # class-list assertions below can pin the exact combined class string.
        cls.elapsed_all_day_event = CalendarEvent.objects.create(
            title='NTT/EFOSC2',
            start_time=datetime(2026, 8, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 2, 6, 0, tzinfo=dt_timezone.utc),
            url=f'ALLOC:{cls.active_run.pk}:2026-08-01',
        )
        CalendarEventMeta.objects.create(event=cls.elapsed_all_day_event, run=cls.active_run)

        # Timed branch: a same-day span, also elapsed.
        cls.elapsed_timed_event = CalendarEvent.objects.create(
            title='NTT/EFOSC2',
            start_time=datetime(2026, 8, 3, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 3, 21, 0, tzinfo=dt_timezone.utc),
            url=f'ALLOC:{cls.active_run.pk}:2026-08-03',
        )
        CalendarEventMeta.objects.create(event=cls.elapsed_timed_event, run=cls.active_run)

        # Timed branch, computed relative to the real clock so this never goes stale.
        future_start = timezone.now() + timedelta(days=400)
        cls.future_year = future_start.year
        cls.future_month = future_start.month
        cls.future_event = CalendarEvent.objects.create(
            title='NTT/EFOSC2',
            start_time=future_start,
            end_time=future_start + timedelta(hours=1),
            url=f'ALLOC:{cls.active_run.pk}:{future_start.date().isoformat()}',
        )
        CalendarEventMeta.objects.create(event=cls.future_event, run=cls.active_run)

    def _get_calendar(self, year, month):
        return self.client.get(reverse('calendar:calendar'), {'year': year, 'month': month})

    def test_elapsed_allocation_nights_render_unused_class_attribute_and_token(self):
        response = self._get_calendar(2026, 8)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('cal-event-classical cal-event-unused', content)
        self.assertIn('cal-event-timed cal-event-unused', content)
        self.assertIn('data-unused="1"', content)
        self.assertIn('[U] NTT/EFOSC2', content)

    def test_future_allocation_night_renders_none_of_the_three_channels(self):
        response = self._get_calendar(self.future_year, self.future_month)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('cal-event-classical cal-event-unused', content)
        self.assertNotIn('cal-event-timed cal-event-unused', content)
        self.assertNotIn('data-unused="1"', content)
        self.assertNotIn('[U] NTT/EFOSC2', content)

    def test_rendering_the_month_view_leaves_stored_titles_byte_identical(self):
        before_all_day_title = self.elapsed_all_day_event.title
        before_timed_title = self.elapsed_timed_event.title
        self._get_calendar(2026, 8)
        self.elapsed_all_day_event.refresh_from_db()
        self.elapsed_timed_event.refresh_from_db()
        self.assertEqual(self.elapsed_all_day_event.title, before_all_day_title)
        self.assertEqual(self.elapsed_timed_event.title, before_timed_title)


class EventModalAttributionHintTest(TestCase):
    """27-07 gap closure (27-UAT.md Test 9, .planning/debug/calendar-event-run-link-inconsistent.md):
    the event_form.html modal for an unlinked event with a HIGH-band attribution-queue
    candidate now surfaces a staff-only "Possible campaign run match" hint naming the
    candidate and linking to the attribution queue filtered to band=high.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Didymos 2026')
        cls.staff_user = User.objects.create_user(username='attrmodalstaff', password='pw', is_staff=True)

        cls.matched_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )

        # unlinked_event_with_candidate: mirrors the real dev-DB pk=59 shape exactly --
        # instrument is deliberately IDENTICAL to matched_run.telescope_instrument so
        # instrument_similarity() is a deterministic 1.0, not dependent on fuzzy-match tuning.
        cls.unlinked_event_with_candidate = CalendarEvent.objects.create(
            title='[EXPIRED] 2m0 2M0-SCICAM-MUSCAT',
            telescope='2m0',
            instrument='FTS/MuSCAT4',
            target_list=cls.campaign,
            start_time=datetime(2026, 7, 16, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 16, 11, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.unlinked_event_with_candidate, is_verified=True, run=None)

        # unlinked_event_no_campaign: a conference/proposal-deadline shape -- no target_list,
        # no CalendarEventMeta row at all, so candidates_for_event() returns [].
        cls.unlinked_event_no_campaign = CalendarEvent.objects.create(
            title='SBAG Meeting',
            start_time=datetime(2026, 7, 17, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 17, 11, 0, tzinfo=dt_timezone.utc),
        )

        # linked_event: already attributed to matched_run.
        cls.linked_event = CalendarEvent.objects.create(
            title='Didymos 2026: FTS/MuSCAT4 (window 2026-07-07..2026-07-21)',
            target_list=cls.campaign,
            start_time=datetime(2026, 7, 18, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 18, 11, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.linked_event, is_verified=True, run=cls.matched_run)

    def _modal_url(self, event):
        return reverse('calendar:update-event', args=[event.id])

    def test_staff_sees_high_band_hint_for_unlinked_event(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self._modal_url(self.unlinked_event_with_candidate))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('Possible campaign run match', content)
        self.assertIn(f'{reverse("campaigns:attribution")}?band=high', content)

    def test_record_backed_event_shows_no_hint(self):
        """37.1 WR-06 (D-09): a record's own event is attributed only through its record and is
        never on the event worklist, so the pop-up must not point at a queue that omits it."""
        record = ObservationRecord.objects.create(
            target=NonSiderealTargetFactory.create(),
            user=User.objects.create(username='hint-record-owner'),
            facility='LCO',
            observation_id='HINT-1',
            status='PENDING',
            parameters={},
        )
        # The observation projector may already have given the record its own event; hand the
        # record to the candidate-bearing event instead, so only the record link differs.
        CalendarEventMeta.objects.filter(observation_record=record).update(observation_record=None)
        CalendarEventMeta.objects.filter(event=self.unlinked_event_with_candidate).update(observation_record=record)
        self.client.force_login(self.staff_user)

        response = self.client.get(self._modal_url(self.unlinked_event_with_candidate))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn('Possible campaign run match', response.content.decode())

    def test_anonymous_does_not_see_hint(self):
        response = self.client.get(self._modal_url(self.unlinked_event_with_candidate))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Possible campaign run match', content)
        self.assertNotIn(f'{reverse("campaigns:attribution")}?band=high', content)

    def test_no_candidate_event_shows_no_hint(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self._modal_url(self.unlinked_event_no_campaign))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Possible campaign run match', content)

    def test_linked_event_shows_run_block_not_hint(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(self._modal_url(self.linked_event))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Possible campaign run match', content)
        self.assertIn(reverse('campaigns:table', args=[self.campaign.pk]), content)

    def test_stale_wr03_comment_removed_from_template_source(self):
        """Asserted against the template file's own contents (source-level), same pattern as
        test_template_source_never_contains_pending_review_literal above."""
        template_path = (
            Path(__file__).resolve().parents[2] / 'src' / 'templates' / 'tom_calendar' / 'partials' / 'event_form.html'
        )
        content = template_path.read_text()
        self.assertNotIn('no production code writes CalendarEventMeta.run yet', content)


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


class MonthCellCampaignMarkerTest(TestCase):
    """Phase 33 Plan 02 Task 1 (ANNOT-02, D-10/D-11): the month grid's two event loops
    (day.all_day_events and day.events) each render a compact campaign marker for an
    event attributed to an approved, publicly-visible run with a campaign -- sourced
    from campaign_decoration(), never from CalendarEvent.title, and carrying the
    campaign name only in the marker's title= tooltip.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Month Marker Campaign')
        cls.approved_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 8, 4),
            window_end=date(2026, 8, 4),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )

        # All-day branch: start/end dates differ. Title is exactly 18 characters -- the
        # all-day loop's truncatechars:18 budget -- so WR-05.2 can prove the chip is a
        # sibling of the filtered title, never folded inside the filter expression.
        cls.all_day_event = CalendarEvent.objects.create(
            title='AllDayAttrEighteen',
            start_time=datetime(2026, 8, 3, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 4, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.all_day_event, run=cls.approved_run)

        # Timed branch: start/end dates are the same day. Title is exactly 16
        # characters -- the timed loop's truncatechars:16 budget.
        cls.timed_event = CalendarEvent.objects.create(
            title='TimedAttrSixteen',
            start_time=datetime(2026, 8, 4, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 4, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.timed_event, run=cls.approved_run)

    def _get_calendar(self):
        return self.client.get(reverse('calendar:calendar'), {'year': 2026, 'month': 8})

    def test_month_view_shows_campaign_chip_and_name_tooltip(self):
        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('cal-campaign-chip', content)
        self.assertIn(f'title="{self.campaign.name}"', content)

    def test_chip_does_not_consume_title_truncation_budget(self):
        """WR-05.2: each fixture title sits exactly at its own filter's budget (18 for
        all-day, 16 for timed), so any character the chip contributed inside the
        truncatechars filter expression would visibly shorten it. Asserting the chip's
        own attribute string is also present proves the chip is a sibling of the
        filtered title, not part of it."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn(self.all_day_event.title, content)
        self.assertIn(self.timed_event.title, content)
        self.assertIn(f'title="{self.campaign.name}"', content)


class CalendarModalOpenerRenderTest(TestCase):
    """Phase 33 Plan 11 (UAT G-33-2): the served month partial must open `#cal-modal`
    through the Bootstrap 5 API and must never contain a jQuery-style selector call.

    The tomtoolkit 3.x base page (`tom_common/base.html`) loads the Bootstrap 5.3.3
    bundle, htmx and Alpine and no jQuery, so a jQuery call in this partial is dead code
    that throws a `ReferenceError` at click time instead of opening the calendar pop-up
    -- the exact defect this plan's Task 1 fixed. This class pins the served-output form
    of that fix so a regression is caught even if no browser test happens to click the
    element that regressed.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Modal Opener Guard Campaign')
        cls.approved_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 9, 4),
            window_end=date(2026, 9, 4),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.timed_event = CalendarEvent.objects.create(
            title='ModalGuardEvent',
            start_time=datetime(2026, 9, 4, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 4, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.timed_event, run=cls.approved_run)

    def _get_calendar(self):
        return self.client.get(reverse('calendar:calendar'), {'year': 2026, 'month': 9})

    def test_calendar_partial_contains_no_jquery_selector_call(self):
        """Before the fix, this exact two-character sequence ('$(') appeared 71 times in
        the rendered page -- one per day-cell click target plus the '+ New Event' button
        and the inner event-container div -- so this assertion is sensitive rather than
        vacuous."""
        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn(
            '$(',
            content,
            'UAT G-33-2: the served month partial must never call a jQuery-style '
            "selector ('$(') -- the tomtoolkit 3.x base page loads Bootstrap 5, htmx "
            'and Alpine and no jQuery, so a jQuery call here is dead code that throws '
            'at runtime instead of opening the calendar pop-up.',
        )

    def test_calendar_partial_opens_modal_via_bootstrap5_api(self):
        """Positive counterpart to the guard above -- asserted explicitly so deleting the
        handlers altogether (rather than fixing them) cannot satisfy the no-jQuery test."""
        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('bootstrap.Modal.getOrCreateInstance', content)


class DecorationSurvivalAndGuardsTest(TestCase):
    """Phase 33 Plan 02 Task 3: proves the month-cell + modal decoration is display-time
    only (survives a from-scratch rewrite of the event's own fields), and exercises the
    campaign-less, non-public, PII, and N+1 boundaries the decoration must respect.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Survival Guard Campaign')
        cls.staff_user = User.objects.create_user(username='survivalstaff', password='pw', is_staff=True)

        cls.approved_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 9, 4),
            window_end=date(2026, 9, 4),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.linked_event = CalendarEvent.objects.create(
            title='Original',
            description='Original description',
            start_time=datetime(2026, 9, 4, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 4, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.linked_event, run=cls.approved_run)

        cls.no_campaign_run = CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            window_start=date(2026, 9, 5),
            window_end=date(2026, 9, 5),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.no_campaign_event = CalendarEvent.objects.create(
            title='No-campaign attributed event',
            start_time=datetime(2026, 9, 5, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 5, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.no_campaign_event, run=cls.no_campaign_run)

        # WR-05.1: pending_run gets its own campaign, distinct from Survival Guard
        # Campaign, so test_pending_review_run_shows_no_marker_for_staff_and_anonymous
        # can discriminate on this campaign's own name -- if pending_run shared
        # cls.campaign, the campaign-name discriminator would also match linked_event's
        # and pii_event's chips, contributing nothing to the outcome.
        cls.pending_campaign = TargetList.objects.create(name='Pending Review Campaign')
        cls.pending_run = CampaignRun.objects.create(
            campaign=cls.pending_campaign,
            telescope_instrument='Should Stay Hidden Scope',
            window_start=date(2026, 9, 6),
            window_end=date(2026, 9, 6),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        cls.pending_event = CalendarEvent.objects.create(
            title='Pending attributed event',
            start_time=datetime(2026, 9, 6, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 6, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.pending_event, run=cls.pending_run)

        cls.pii_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='PII Guard Scope',
            window_start=date(2026, 9, 7),
            window_end=date(2026, 9, 7),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            contact_person='Do Not Leak Person',
            contact_email='donotleak@example.org',
            source=CampaignRun.Source.CSV_IMPORT,
        )
        cls.pii_event = CalendarEvent.objects.create(
            title='PII guard event',
            start_time=datetime(2026, 9, 7, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 7, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.pii_event, run=cls.pii_run)

    def _get_calendar(self, year=2026, month=9):
        return self.client.get(reverse('calendar:calendar'), {'year': year, 'month': month})

    def _modal_url(self, event):
        return reverse('calendar:update-event', args=[event.id])

    def _campaign_table_href(self, campaign=None):
        return reverse('campaigns:table', args=[(campaign or self.campaign).pk])

    def test_decoration_survives_from_scratch_rewrite_of_title_and_description(self):
        """ROADMAP criterion 3: the decoration lives on the link (CalendarEventMeta.run),
        never on the event's own fields, so rewriting title/description from scratch
        cannot erase it."""
        response = self._get_calendar()
        content = response.content.decode()
        self.assertIn('cal-campaign-chip', content)
        modal_response = self.client.get(self._modal_url(self.linked_event))
        self.assertIn('Attributed campaign run', modal_response.content.decode())

        self.linked_event.title = 'Rewritten'
        self.linked_event.description = 'Brand-new description after re-projection'
        self.linked_event.save()

        fresh_response = self._get_calendar()
        fresh_content = fresh_response.content.decode()
        self.assertIn('cal-campaign-chip', fresh_content)
        self.assertIn('Rewritten', fresh_content)

        fresh_modal_response = self.client.get(self._modal_url(self.linked_event))
        self.assertIn('Attributed campaign run', fresh_modal_response.content.decode())

    def test_no_campaign_run_renders_marker_and_no_table_href(self):
        """WR-05.3: asserts on the no-campaign chip's own tooltip string -- linked_event
        and pii_event in the same September grid already carry the shared
        cal-campaign-chip class, so that class alone proves nothing about
        no_campaign_run specifically. campaign_decoration()'s table_url is never
        rendered into the month-cell chip (only the modal's <a href> uses it), so
        _campaign_table_href() retargeted at no_campaign_run.campaign (None, which the
        helper falls back to self.campaign for) is a namespace guard on the whole page
        rather than a per-event assertion."""
        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn(f'Attributed run #{self.no_campaign_run.pk} {NO_CAMPAIGN_LABEL}', content)
        self.assertNotIn(self._campaign_table_href(self.no_campaign_run.campaign), content)

    def test_pending_review_run_shows_no_marker_for_staff_and_anonymous(self):
        """WR-05.1: discriminates on pending_campaign's own name -- a value only the
        pending fixture can produce -- so this test fails if the is_publicly_visible
        gate in campaign_decoration() is deleted. telescope_instrument is never
        rendered by the month cell, so asserting on it (as this test previously did)
        cannot detect a regression of the visibility gate."""
        anon_response = self._get_calendar()
        self.assertEqual(anon_response.status_code, 200)
        anon_content = anon_response.content.decode()
        self.assertNotIn(f'title="{self.pending_campaign.name}"', anon_content)
        self.assertNotIn(f'aria-label="Campaign: {self.pending_campaign.name}"', anon_content)

        self.client.force_login(self.staff_user)
        staff_response = self._get_calendar()
        self.assertEqual(staff_response.status_code, 200)
        staff_content = staff_response.content.decode()
        self.assertNotIn(f'title="{self.pending_campaign.name}"', staff_content)
        self.assertNotIn(f'aria-label="Campaign: {self.pending_campaign.name}"', staff_content)

    def test_pii_fields_never_render_on_month_view(self):
        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Do Not Leak Person', content)
        self.assertNotIn('donotleak@example.org', content)
        self.assertNotIn(CampaignRun.Source.CSV_IMPORT.value, content)

    def test_query_count_does_not_grow_with_number_of_attributed_events(self):
        """Count-comparison form (1 attributed event vs. N), never a hard-coded number,
        so an unrelated future query addition to the month view does not make this test
        brittle -- the assertion that matters is that the count does not grow with N."""
        single_campaign = TargetList.objects.create(name='Single Query Campaign')
        single_run = CampaignRun.objects.create(
            campaign=single_campaign,
            telescope_instrument='Single Query Scope',
            window_start=date(2026, 10, 1),
            window_end=date(2026, 10, 1),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        single_event = CalendarEvent.objects.create(
            title='Single query event',
            start_time=datetime(2026, 10, 1, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 10, 1, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=single_event, run=single_run)

        with CaptureQueriesContext(connection) as single_ctx:
            self._get_calendar(year=2026, month=10)
        single_count = len(single_ctx)

        for i in range(4):
            campaign_n = TargetList.objects.create(name=f'N+1 Guard Campaign {i}')
            run_n = CampaignRun.objects.create(
                campaign=campaign_n,
                telescope_instrument=f'N+1 Guard Scope {i}',
                window_start=date(2026, 10, 2 + i),
                window_end=date(2026, 10, 2 + i),
                approval_status=CampaignRun.ApprovalStatus.APPROVED,
            )
            event_n = CalendarEvent.objects.create(
                title=f'N+1 guard event {i}',
                start_time=datetime(2026, 10, 2 + i, 20, 0, tzinfo=dt_timezone.utc),
                end_time=datetime(2026, 10, 2 + i, 21, 0, tzinfo=dt_timezone.utc),
            )
            CalendarEventMeta.objects.create(event=event_n, run=run_n)

        with CaptureQueriesContext(connection) as multi_ctx:
            self._get_calendar(year=2026, month=10)
        multi_count = len(multi_ctx)

        self.assertEqual(multi_count, single_count)

    def test_campaign_name_encoding_edge_escapes_consistently_in_title_and_aria_label(self):
        """ANNOT-02 encoding edge: a campaign name containing &, < and " must be
        HTML-escaped identically in the chip's title= and aria-label= attributes by
        Django's autoescape -- the raw characters never reach the rendered attribute
        values, and the two attributes agree on what the escaped campaign name is."""
        raw_name = 'A & B < C "D"'
        escaped_name = escape(raw_name)
        encoding_campaign = TargetList.objects.create(name=raw_name)
        encoding_run = CampaignRun.objects.create(
            campaign=encoding_campaign,
            telescope_instrument='Encoding Guard Scope',
            window_start=date(2026, 9, 8),
            window_end=date(2026, 9, 8),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        encoding_event = CalendarEvent.objects.create(
            title='Encoding guard event',
            start_time=datetime(2026, 9, 8, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 8, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=encoding_event, run=encoding_run)

        response = self._get_calendar()
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn(raw_name, content)
        self.assertIn(f'title="{escaped_name}"', content)
        self.assertIn(f'aria-label="Campaign: {escaped_name}"', content)


class CalendarStatusLegendRenderTest(TestCase):
    """PROJ-03/D-02 (Phase 34 Plan 03): the observation-status marker legend renders on
    the calendar page itself -- reached through the Django test client, never by
    importing solsys_code.views directly (that module transitively loads SPICE kernels)."""

    def test_calendar_page_renders_every_legend_marker_and_label(self):
        response = self.client.get(reverse('calendar:calendar'), {'year': 2026, 'month': 9})
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        for entry in observation_status_legend():
            with self.subTest(marker=entry['marker']):
                self.assertIn(entry['marker'], content)
                self.assertIn(entry['label'], content)


class EventModalSeriesDecorationTest(TestCase):
    """PROJ-04/PROJ-05 (Phase 34 Plan 03, D-04): the event modal renders series identity
    ("night n of N") from CalendarEventMeta.observation_group at request time, alongside
    any campaign decoration, and the month view's query count does not grow with the
    number of grouped observation events (PROJ-05 performance edge). The observation
    projector's post_save/m2m_changed receivers are globally wired (34-01), so they are
    disconnected around fixture-creation calls here -- this class tests rendering, not
    the projector (34-01 precedent)."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign = TargetList.objects.create(name='Series Modal Campaign')
        # Not named `cls.run` -- unittest.TestCase.run() is the test-execution entry
        # point, and shadowing it with a class attribute breaks the test runner.
        cls.campaign_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        # An authenticated (not necessarily staff) viewer -- observation_series_decoration()'s
        # anonymous-viewer gate for an un-attributed (run=None) event only checks
        # is_authenticated, not is_staff, so a plain user is the right fixture here.
        cls.authenticated_user = User.objects.create_user(username='seriesmodalviewer', password='pw')

    def _make_record(self, observation_id: str, start: datetime, end: datetime) -> ObservationRecord:
        post_save.disconnect(
            receiver_on_record_save,
            sender=ObservationRecord,
            dispatch_uid='solsys_code.observation_projector.post_save',
        )
        try:
            return ObservationRecord.objects.create(
                target=self.target,
                facility='LCO',
                observation_id=observation_id,
                status='COMPLETED',
                parameters={
                    'proposal': 'TESTPROP',
                    'instrument_type': '2M0-SCICAM-MUSCAT',
                    'start': start.isoformat(),
                    'end': end.isoformat(),
                },
            )
        finally:
            post_save.connect(
                receiver_on_record_save,
                sender=ObservationRecord,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.post_save',
            )

    def _add_to_group(self, group: ObservationGroup, *records: ObservationRecord) -> None:
        m2m_changed.disconnect(
            receiver_on_group_membership_changed,
            sender=ObservationGroup.observation_records.through,
            dispatch_uid='solsys_code.observation_projector.m2m_changed',
        )
        try:
            group.observation_records.add(*records)
        finally:
            m2m_changed.connect(
                receiver_on_group_membership_changed,
                sender=ObservationGroup.observation_records.through,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.m2m_changed',
            )

    def _modal_url(self, event: CalendarEvent):
        return reverse('calendar:update-event', args=[event.id])

    def test_grouped_event_modal_hides_group_name_from_anonymous_viewer(self):
        """An un-attributed (run=None) grouped event -- the common case, since most
        projector-owned events never go through campaign attribution -- must not publish
        the observation-group's own name (an internal portal RequestGroup identifier) to
        an anonymous visitor of the unauthenticated event-update view."""
        r1 = self._make_record(
            'modal-series-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r2 = self._make_record(
            'modal-series-2',
            datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Series Modal Group')
        self._add_to_group(group, r1, r2)

        event = CalendarEvent.objects.create(
            title='Series modal event',
            start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=event, observation_record=r1, observation_group=group)

        response = self.client.get(self._modal_url(event))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Series Modal Group', content)
        self.assertNotIn('Night 1 of 2', content)

    def test_grouped_event_modal_shows_group_name_and_night_n_of_n_to_authenticated_viewer(self):
        r1 = self._make_record(
            'modal-series-auth-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r2 = self._make_record(
            'modal-series-auth-2',
            datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Series Modal Group (authenticated)')
        self._add_to_group(group, r1, r2)

        event = CalendarEvent.objects.create(
            title='Series modal event',
            start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=event, observation_record=r1, observation_group=group)

        self.client.force_login(self.authenticated_user)
        response = self.client.get(self._modal_url(event))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('Series Modal Group (authenticated)', content)
        self.assertIn('Night 1 of 2', content)

    def test_grouped_and_attributed_event_shows_both_decorations(self):
        """The series block is gated on the viewer being authenticated, full stop --
        including for an attributed, approved run. Only the campaign block is visible to
        an anonymous viewer (its own gate is CampaignRun.is_publicly_visible, unrelated to
        the series block's viewer check). See
        test_grouped_and_attributed_event_hides_group_name_from_anonymous_viewer for the
        anonymous-viewer counterpart this same fixture must satisfy."""
        r1 = self._make_record(
            'modal-both-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r2 = self._make_record(
            'modal-both-2',
            datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Both Decorations Group')
        self._add_to_group(group, r1, r2)

        event = CalendarEvent.objects.create(
            title='Series and campaign event',
            start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(
            event=event, observation_record=r1, observation_group=group, run=self.campaign_run
        )

        self.client.force_login(self.authenticated_user)
        response = self.client.get(self._modal_url(event))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('Both Decorations Group', content)
        self.assertIn('Night 1 of 2', content)
        self.assertIn('Attributed campaign run', content)
        self.assertIn('FTN/MuSCAT3', content)

    def test_grouped_and_attributed_event_hides_group_name_from_anonymous_viewer(self):
        """An attributed AND approved run does not bypass the series block's viewer check
        -- the group name is an internal portal RequestGroup identifier regardless of
        campaign attribution, so an anonymous visitor must not see it even though the
        campaign block itself (a different value, gated by a different rule) is public for
        an approved run."""
        r1 = self._make_record(
            'modal-both-anon-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r2 = self._make_record(
            'modal-both-anon-2',
            datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Both Decorations Group Anon')
        self._add_to_group(group, r1, r2)

        event = CalendarEvent.objects.create(
            title='Series and campaign event (anonymous)',
            start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(
            event=event, observation_record=r1, observation_group=group, run=self.campaign_run
        )

        response = self.client.get(self._modal_url(event))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Both Decorations Group Anon', content)
        self.assertNotIn('Night 1 of 2', content)
        # The campaign block's own visibility rule is unrelated and unaffected: an
        # approved run's campaign attribution is still public to an anonymous viewer.
        self.assertIn('Attributed campaign run', content)
        self.assertIn('FTN/MuSCAT3', content)

    def test_grouped_event_hides_group_name_when_attributed_run_is_pending_review(self):
        """WR-03: observation_series_decoration() must apply the same is_publicly_visible
        gate campaign_decoration() already applies -- a pending-review run's attribution
        must not leak the observation-group's own name (an internal portal RequestGroup
        id) to an anonymous visitor of the unauthenticated event-update view."""
        r1 = self._make_record(
            'modal-pending-review-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r2 = self._make_record(
            'modal-pending-review-2',
            datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Pending Review Group Name')
        self._add_to_group(group, r1, r2)
        pending_run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTS/MuSCAT3',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )

        event = CalendarEvent.objects.create(
            title='Pending review series event',
            start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=event, observation_record=r1, observation_group=group, run=pending_run)

        response = self.client.get(self._modal_url(event))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertNotIn('Pending Review Group Name', content)
        self.assertNotIn('Night 1 of 2', content)

    def _make_modal_group_event(self, group_name: str, month: int, size: int) -> CalendarEvent:
        """Build a `size`-member ObservationGroup and a CalendarEvent linked to its first
        member, all in `month` (2026) so distinct calls never collide on window times."""
        records = [
            self._make_record(
                f'{group_name}-{i}',
                datetime(2026, month, 1 + i, 20, 0, tzinfo=dt_timezone.utc),
                datetime(2026, month, 1 + i, 21, 0, tzinfo=dt_timezone.utc),
            )
            for i in range(size)
        ]
        group = ObservationGroup.objects.create(name=group_name)
        self._add_to_group(group, *records)
        event = CalendarEvent.objects.create(
            title=f'{group_name} event',
            start_time=datetime(2026, month, 1, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, month, 1, 21, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=event, observation_record=records[0], observation_group=group)
        return event

    def test_modal_query_count_does_not_grow_with_group_size(self):
        """WR-05: observation_series_decoration() is reachable only from the event-update
        modal (tom_calendar.views.update_event fetches its one CalendarEvent by pk with no
        select_related of its own) -- calendar:calendar never renders it, so the previous
        version of this test measured the month view and passed unconditionally regardless
        of the tag's real per-modal fan-out. Compares a 2-member group against a 10-member
        group, count-comparison form (never a hard-coded number), per the sibling
        campaign-attribution query-count test's own convention. Logs in first: these fixture
        events carry no `run` (see _make_modal_group_event), and an anonymous viewer would
        now be gated out before the per-member query fan-out this test exists to measure
        ever runs -- which would make the comparison trivially equal for the wrong reason."""
        self.client.force_login(self.authenticated_user)
        small_event = self._make_modal_group_event('Query Guard Small Group', month=10, size=2)

        with CaptureQueriesContext(connection) as small_ctx:
            self.client.get(self._modal_url(small_event))
        small_count = len(small_ctx)

        large_event = self._make_modal_group_event('Query Guard Large Group', month=11, size=10)

        with CaptureQueriesContext(connection) as large_ctx:
            self.client.get(self._modal_url(large_event))
        large_count = len(large_ctx)

        self.assertEqual(large_count, small_count)


class EventFormUrlLinkTest(TestCase):
    """UAT G-37.1-1-allocurl: the signed-in editor's event form links only http(s) addresses.

    An anonymous visitor gets the read-only card instead of the form (see EventCardUrlLinkTest).

    The allocation layer (``ALLOC:{run.pk}:{night}``) and the campaign reconciler (``RUN:{pk}``)
    keep namespace keys in ``CalendarEvent.url``; those must show as plain values, never as a
    dead link, and a stored ``javascript:`` url must never land in an href.
    """

    PORTAL_URL = 'https://observe.lco.global/requests/4229878'

    @classmethod
    def setUpTestData(cls) -> None:
        cls.editor = User.objects.create_user(username='urlcase-editor', password='pw')

    def _form_html(self, url: str) -> str:
        self.client.force_login(self.editor)
        event = CalendarEvent.objects.create(
            title='URL case',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
            url=url,
        )
        response = self.client.get(reverse('calendar:update-event', args=[event.id]))
        self.assertEqual(response.status_code, 200)
        return response.content.decode()

    def test_allocation_key_is_not_a_link(self):
        key = f'{ALLOC_URL_NAMESPACE}1:2026-07-07'
        content = self._form_html(key)
        self.assertNotIn(f'href="{ALLOC_URL_NAMESPACE}', content)
        self.assertIn('not a web link', content)
        self.assertIn(f'value="{key}"', content)

    def test_campaign_run_key_is_not_a_link(self):
        content = self._form_html(f'{RUN_URL_NAMESPACE}5')
        self.assertNotIn(f'href="{RUN_URL_NAMESPACE}', content)
        self.assertIn('not a web link', content)

    def test_portal_url_still_links_with_noopener(self):
        content = self._form_html(self.PORTAL_URL)
        self.assertIn(f'href="{self.PORTAL_URL}"', content)
        self.assertIn('rel="noopener noreferrer"', content)
        self.assertIn('View', content)
        self.assertNotIn('not a web link', content)

    def test_javascript_url_is_never_a_link(self):
        content = self._form_html('javascript:alert(1)')
        self.assertNotIn('href="javascript:', content)

    def test_empty_url_shows_neither_link_nor_note(self):
        content = self._form_html('')
        self.assertNotIn('not a web link', content)
        self.assertNotIn(self.PORTAL_URL, content)


class CalendarMonthViewReadOnlyTest(TestCase):
    """ACCESS-02 / D-07: the month view offers create click targets only to a signed-in user."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.event = CalendarEvent.objects.create(
            title='Month Event',
            start_time=datetime(2026, 8, 4, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 4, 21, 0, tzinfo=dt_timezone.utc),
        )
        cls.editor = User.objects.create_user(username='month-editor', password='pw')

    def _month(self, client=None, **extra) -> str:
        response = (client or self.client).get(reverse('calendar:calendar'), {'year': 2026, 'month': 8}, **extra)
        self.assertEqual(response.status_code, 200)
        return response.content.decode()

    def test_anonymous_month_view_has_no_create_target(self):
        content = self._month()
        self.assertNotIn('/calendar/create/', content)
        self.assertNotIn('+ New Event', content)
        self.assertIn(reverse('calendar:update-event', args=[self.event.pk]), content)
        self.assertIn('bootstrap.Modal.getOrCreateInstance', content)
        self.assertIn('cal-header-spacer', content)

    def test_signed_in_month_view_keeps_both_create_targets(self):
        self.client.force_login(self.editor)
        content = self._month()
        self.assertIn('+ New Event', content)
        self.assertIn('/calendar/create/?date=2026-08-01', content)
        self.assertNotIn('cal-header-spacer', content)

    def test_anonymous_month_partial_offers_no_login_prompt(self):
        content = self._month(headers={'HX-Request': 'true'})
        lowered = content.lower()
        self.assertNotIn('log in', lowered)
        self.assertNotIn('login', lowered)
        self.assertNotIn('/accounts/login/', content)


class EventModalReadOnlyCardTest(TestCase):
    """ACCESS-02 / D-04 / D-05: an anonymous visitor's pop-up is a read-only card, not the form."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target_list = TargetList.objects.create(name='Card Target List')
        cls.campaign = TargetList.objects.create(name='Card Campaign')
        cls.card_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 8, 4),
            window_end=date(2026, 8, 4),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.event = CalendarEvent.objects.create(
            title='Card Event',
            description='Line one\nLine two',
            start_time=datetime(2026, 8, 4, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 4, 21, 0, tzinfo=dt_timezone.utc),
            url='https://observe.lco.global/requests/4229878',
            target_list=cls.target_list,
            user='tlister',
            proposal='KEY2026B-004',
            telescope='FTN',
            instrument='MuSCAT3',
        )
        CalendarEventMeta.objects.create(event=cls.event, run=cls.card_run)
        cls.done_todo = EventTodo.objects.create(event=cls.event, description='Check guider', is_completed=True)
        cls.open_todo = EventTodo.objects.create(event=cls.event, description='Reduce frames', is_completed=False)
        cls.bare_event = CalendarEvent.objects.create(
            title='Bare Event',
            start_time=datetime(2026, 8, 5, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 5, 21, 0, tzinfo=dt_timezone.utc),
        )
        cls.markup_event = CalendarEvent.objects.create(
            title='Card <b>bold</b> title',
            description='Line one\nLine two <script>alert(1)</script>',
            start_time=datetime(2026, 8, 6, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 6, 21, 0, tzinfo=dt_timezone.utc),
        )
        cls.editor = User.objects.create_user(username='card-editor', password='pw')

    def _popup_url(self, event) -> str:
        return reverse('calendar:update-event', args=[event.pk])

    def _popup(self, event, client=None) -> str:
        response = (client or self.client).get(self._popup_url(event))
        self.assertEqual(response.status_code, 200)
        return response.content.decode()

    def test_anonymous_card_has_no_form_controls(self):
        content = self._popup(self.event)
        for forbidden in ('<form', '<input', '<select', '<textarea', '<button', 'hx-post', 'csrfmiddlewaretoken'):
            self.assertNotIn(forbidden, content)
        pk = self.event.pk
        for url in (
            reverse('calendar:delete-event', args=[pk]),
            reverse('calendar:update-event', args=[pk]),
            reverse('calendar:create-todo', args=[pk]),
            reverse('calendar:update-todo', args=[self.done_todo.pk]),
            reverse('calendar:update-todo', args=[self.open_todo.pk]),
        ):
            self.assertNotIn(url, content)

    def test_anonymous_card_shows_every_field_as_text(self):
        content = self._popup(self.event)
        self.assertIn('id="cal-event-card"', content)
        for label in (
            'Title',
            'Start',
            'End',
            'Description',
            'URL',
            'Target list',
            'User',
            'Proposal',
            'Telescope',
            'Instrument',
        ):
            self.assertIn(f'<dt class="col-sm-3">{label}</dt>', content)
        for value in (
            'Card Event',
            '2026-08-04 20:00 UTC',
            '2026-08-04 21:00 UTC',
            'Line one<br>Line two',
            'tlister',
            'KEY2026B-004',
            'FTN',
            'MuSCAT3',
            'Card Target List',
            f'{reverse("targets:list")}?targetlist__name={self.target_list.id}',
        ):
            self.assertIn(value, content)

    def test_anonymous_card_omits_empty_fields(self):
        content = self._popup(self.bare_event)
        for label in ('Title', 'Start', 'End'):
            self.assertIn(f'<dt class="col-sm-3">{label}</dt>', content)
        for label in ('Description', 'URL', 'Target list', 'User', 'Proposal', 'Telescope', 'Instrument'):
            self.assertNotIn(f'<dt class="col-sm-3">{label}</dt>', content)

    def test_anonymous_card_renders_attributed_run_block_once(self):
        content = self._popup(self.event)
        self.assertEqual(content.count('Attributed campaign run'), 1)
        self.assertIn('FTN/MuSCAT3', content)
        self.assertIn(f'{reverse("campaigns:table", args=[self.campaign.pk])}#run-{self.card_run.pk}', content)

    def test_anonymous_card_lists_todos_read_only(self):
        content = self._popup(self.event)
        self.assertIn('id="cal-todos-readonly"', content)
        self.assertRegex(content, r'Check guider</span>\s*<small[^>]*>\(done\)')
        self.assertRegex(content, r'Reduce frames\s*<small[^>]*>\(not done\)')
        self.assertNotIn('type="checkbox"', content)

    def test_anonymous_card_with_no_todos_says_so(self):
        content = self._popup(self.bare_event)
        self.assertIn('No todos yet.', content)

    def test_anonymous_card_escapes_markup(self):
        content = self._popup(self.markup_event)
        self.assertIn(escape(self.markup_event.title), content)
        self.assertNotIn('<b>bold</b>', content)
        self.assertIn('&lt;script&gt;', content)
        self.assertNotIn('<script>alert(1)', content)
        self.assertIn('Line one<br>Line two', content)

    def test_anonymous_card_offers_no_login_prompt(self):
        lowered = self._popup(self.event).lower()
        self.assertNotIn('log in', lowered)
        self.assertNotIn('login', lowered)

    def test_signed_in_user_gets_the_editable_form(self):
        self.client.force_login(self.editor)
        content = self._popup(self.event)
        self.assertIn('<form', content)
        self.assertIn('csrfmiddlewaretoken', content)
        self.assertIn('>Save</button>', content)
        self.assertIn(reverse('calendar:delete-event', args=[self.event.pk]), content)
        self.assertIn(reverse('calendar:create-todo', args=[self.event.pk]), content)
        self.assertNotIn('cal-event-card', content)
        self.assertEqual(content.count('Attributed campaign run'), 1)

    def _row_counts(self) -> tuple[int, int, int]:
        return (CalendarEvent.objects.count(), EventTodo.objects.count(), CalendarEventMeta.objects.count())

    def test_anonymous_reads_write_nothing(self):
        before_counts = self._row_counts()
        before_modified = CalendarEvent.objects.get(pk=self.event.pk).modified
        for _ in range(2):
            self.assertEqual(self.client.get(reverse('calendar:calendar'), {'year': 2026, 'month': 8}).status_code, 200)
            self._popup(self.event)
        self.assertEqual(self._row_counts(), before_counts)
        self.assertEqual(CalendarEvent.objects.get(pk=self.event.pk).modified, before_modified)

    def test_editor_then_anonymous_render_share_no_output(self):
        self.client.force_login(self.editor)
        editor_month = self.client.get(reverse('calendar:calendar'), {'year': 2026, 'month': 8}).content.decode()
        editor_popup = self._popup(self.event)
        self.assertIn('/calendar/create/', editor_month)
        self.assertIn('<form', editor_popup)

        visitor = Client()
        visitor_month = visitor.get(reverse('calendar:calendar'), {'year': 2026, 'month': 8}).content.decode()
        visitor_popup = self._popup(self.event, client=visitor)
        self.assertNotIn('/calendar/create/', visitor_month)
        self.assertNotIn('<form', visitor_popup)
        self.assertIn('cal-event-card', visitor_popup)


class EventCardUrlLinkTest(TestCase):
    """ACCESS-02 / D-05: the anonymous card links only http(s) addresses and never echoes other values."""

    PORTAL_URL = 'https://observe.lco.global/requests/4229878'

    def _card_html(self, url: str) -> str:
        event = CalendarEvent.objects.create(
            title='URL card case',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
            url=url,
        )
        response = self.client.get(reverse('calendar:update-event', args=[event.id]))
        self.assertEqual(response.status_code, 200)
        return response.content.decode()

    def test_allocation_key_is_not_a_link_and_not_echoed(self):
        key = f'{ALLOC_URL_NAMESPACE}1:2026-07-07'
        content = self._card_html(key)
        self.assertNotIn(f'href="{ALLOC_URL_NAMESPACE}', content)
        self.assertIn('not a web link', content)
        self.assertNotIn(key, content)

    def test_campaign_run_key_is_not_a_link_and_not_echoed(self):
        key = f'{RUN_URL_NAMESPACE}5'
        content = self._card_html(key)
        self.assertNotIn(f'href="{RUN_URL_NAMESPACE}', content)
        self.assertIn('not a web link', content)
        self.assertNotIn(key, content)

    def test_portal_url_links_with_noopener(self):
        content = self._card_html(self.PORTAL_URL)
        self.assertIn(f'href="{self.PORTAL_URL}"', content)
        self.assertIn('rel="noopener noreferrer"', content)
        self.assertIn('View', content)
        self.assertNotIn('not a web link', content)

    def test_javascript_url_is_never_a_link_and_not_echoed(self):
        content = self._card_html('javascript:alert(1)')
        self.assertNotIn('href="javascript:', content)
        self.assertNotIn('javascript:alert(1)', content)

    def test_empty_url_shows_no_url_row(self):
        content = self._card_html('')
        self.assertNotIn('<dt class="col-sm-3">URL</dt>', content)
        self.assertNotIn('not a web link', content)


class EventFormHeaderMatchesUpstreamTest(SimpleTestCase):
    """WARN-01 / D-10: event_form.html's header lists exactly the blocks that differ from the installed upstream."""

    TEMPLATE = Path(__file__).resolve().parents[2] / 'src/templates/tom_calendar/partials/event_form.html'
    # Per header item: literals that must occur in a differing region and in that item's own header text.
    ANCHORS = {
        1: ('attribution_display_extras',),
        2: ('is_web_url', 'noopener noreferrer', 'not a web link'),
        3: ('<button',),
        4: ('observation_series_decoration', 'campaign_decoration', 'high_band_attribution_candidates'),
        5: ('request.user.is_authenticated', 'cal-event-card'),
        6: ('request.user.is_authenticated', 'event.todos.all'),
    }

    @classmethod
    def _source(cls) -> str:
        return cls.TEMPLATE.read_text()

    @classmethod
    def _header(cls) -> str:
        source = cls._source()
        return source.split('{% endcomment %}', 1)[0]

    @classmethod
    def _body_lines(cls) -> list[str]:
        return cls._source().split('{% endcomment %}\n', 1)[1].splitlines()

    @classmethod
    def _upstream_lines(cls) -> list[str]:
        upstream = (
            Path(tom_calendar.__file__).resolve().parent / 'templates' / 'tom_calendar' / 'partials' / 'event_form.html'
        )
        assert upstream.exists(), f'installed upstream partial not found at {upstream}'
        return upstream.read_text().splitlines()

    def test_header_names_the_pinned_upstream(self):
        source = self._source()
        self.assertTrue(source.startswith('{% comment %}'))
        header = self._header()
        for needed in (
            'tomtoolkit 3.1.0',
            'tom_calendar/templates/tom_calendar/partials/event_form.html',
            'FOMO override of the upstream tom_calendar partial',
        ):
            self.assertIn(needed, header)
        for stale in ('exact copy', 'one new block', '3.0.1', '3.0.0a9'):
            self.assertNotIn(stale, header)
        # The header sits inside a {% comment %} block, so it must not contain template syntax itself.
        inner = header[len('{% comment %}') :]
        self.assertNotIn('{%', inner)
        self.assertNotIn('#}', inner)

    def test_header_items_are_numbered_one_to_six(self):
        markers = re.findall(r'^\s+(\d+)\.\s', self._header(), flags=re.MULTILINE)
        self.assertEqual(markers, ['1', '2', '3', '4', '5', '6'])

    def _item_texts(self) -> dict[int, str]:
        header = self._header()
        starts = {int(m.group(1)): m.start() for m in re.finditer(r'^\s+(\d+)\.\s', header, flags=re.MULTILINE)}
        self.assertEqual(sorted(starts), [1, 2, 3, 4, 5, 6], 'header items must be numbered 1 to 6')
        texts = {}
        for number in range(1, 7):
            end = starts[number + 1] if number < 6 else len(header)
            texts[number] = header[starts[number] : end]
        return texts

    def test_every_differing_region_is_listed_and_every_item_differs(self):
        upstream = self._upstream_lines()
        body = self._body_lines()
        opcodes = difflib.SequenceMatcher(None, upstream, body, autojunk=False).get_opcodes()
        regions = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                continue
            lines = body[j1:j2] if j2 > j1 else upstream[i1:i2]
            regions.append('\n'.join(lines))
        all_anchors = [anchor for anchors in self.ANCHORS.values() for anchor in anchors]
        for region in regions:
            self.assertTrue(
                any(anchor in region for anchor in all_anchors),
                f'a differing region is not covered by any header item:\n{region}',
            )
        for anchor in all_anchors:
            self.assertTrue(
                any(anchor in region for region in regions),
                f'header anchor {anchor!r} does not occur in any region that differs from upstream',
            )
        texts = self._item_texts()
        for number, anchors in self.ANCHORS.items():
            for anchor in anchors:
                self.assertIn(anchor, texts[number], f'header item {number} does not mention {anchor!r}')


class CalendarTemplateBootstrap5ClassTest(SimpleTestCase):
    """D-11: calendar.html uses the Bootstrap 5 utility names tomtoolkit 3.1.0's partial uses."""

    TEMPLATE = Path(__file__).resolve().parents[2] / 'src/templates/tom_calendar/partials/calendar.html'

    def test_calendar_partial_uses_bootstrap5_utility_names(self):
        source = self.TEMPLATE.read_text()
        self.assertIsNone(re.search(r'(?<![\w-])(?:mr|ml)-[0-9]', source))
        bootstrap4_names = ('border-' + 'left', 'border-' + 'right', 'font-weight-' + 'bold', 'var(--' + 'white)')
        for name in bootstrap4_names:
            self.assertNotIn(name, source)
        for name in ('border-start', 'border-end', 'fw-bold', 'me-2', 'me-3', 'var(--bs-white)'):
            self.assertIn(name, source)
        self.assertIn('data-url=', source)
        self.assertNotIn('data-bs-url', source)
