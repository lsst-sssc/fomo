"""Unit tests for solsys_code.templatetags.calendar_display_extras.

Wave 0 scaffold — written before the module exists (RED). Tests cover the three
public tags: proposal_color (DISPLAY-04, D-04/D-05), status_border_css (DISPLAY-06,
D-08/D-09), and visible_proposals (DISPLAY-07, D-02/D-04/D-06).
"""

from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from types import SimpleNamespace

from django.db.models.signals import m2m_changed, post_save
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import observation_projector as op
from solsys_code.models import CalendarEventMeta, CampaignRun
from solsys_code.observation_projector import receiver_on_group_membership_changed, receiver_on_record_save
from solsys_code.templatetags.calendar_display_extras import (
    CLASSICAL_SCHEDULE_LABEL,
    NEUTRAL_SLOT_COLOR,
    PROPOSAL_PALETTE,
    STRIPE_OUTER_EDGE_COLOR,
    TELESCOPE_PALETTE,
    TELESCOPE_STRIPE_PALETTE,
    _contrast_ratio,
    _relative_luminance,
    neutral_slot_color,
    observation_series_decoration,
    observation_status_legend,
    proposal_color,
    run_tally,
    status_border_css,
    telescope_color,
    telescope_stripe_color,
    text_color_for_bg,
    unused_night_decoration,
    visible_classical_telescopes,
    visible_proposals,
)

QUEUED_BOX_SHADOW = 'box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.45);'
TERMINAL_BOX_SHADOW = 'box-shadow: 0 0 0 3px rgba(160, 0, 0, 0.55);'

# observation_series_decoration() is a takes_context=True tag: calling it directly (as this
# module does, rather than through a template) requires supplying a context dict of our own.
# These two stand in for what django.contrib.auth.context_processors.auth would put there for
# a logged-in vs anonymous request.
AUTHENTICATED_CONTEXT = {'user': SimpleNamespace(is_authenticated=True)}
ANONYMOUS_CONTEXT = {'user': SimpleNamespace(is_authenticated=False)}


class ProposalColorTest(TestCase):
    def test_same_input_same_output(self):
        # DISPLAY-04: deterministic — same proposal always returns the same color.
        self.assertEqual(proposal_color('LTP2025A-004'), proposal_color('LTP2025A-004'))

    def test_normalization_case_insensitive(self):
        # D-04 premise: .strip().upper() applied before hashing.
        self.assertEqual(proposal_color('LTP2025A-004'), proposal_color('ltp2025a-004'))

    def test_normalization_trailing_space(self):
        # D-04 premise: whitespace stripped before hashing.
        self.assertEqual(proposal_color('LTP2025A-004'), proposal_color('LTP2025A-004 '))

    def test_empty_string_returns_neutral_slot(self):
        # D-05: empty proposal → dedicated neutral slot, not hash-of-empty.
        self.assertEqual(proposal_color(''), NEUTRAL_SLOT_COLOR)

    def test_blank_string_returns_neutral_slot(self):
        # D-05: whitespace-only proposal → neutral slot after .strip().
        self.assertEqual(proposal_color('   '), NEUTRAL_SLOT_COLOR)

    def test_none_returns_neutral_slot(self):
        # D-05: None proposal → neutral slot.
        self.assertEqual(proposal_color(None), NEUTRAL_SLOT_COLOR)

    def test_nonempty_proposal_returns_palette_member(self):
        # D-04: non-empty proposals map to one of the 8 curated palette entries.
        color = proposal_color('LTP2025A-004')
        self.assertIn(color, PROPOSAL_PALETTE)

    def test_neutral_slot_not_in_palette(self):
        # D-05: neutral slot is a separate slot — not a palette hash target.
        self.assertNotIn(NEUTRAL_SLOT_COLOR, PROPOSAL_PALETTE)


class StatusBorderCssTest(TestCase):
    def test_queued_returns_queued_box_shadow(self):
        # D-08: [QUEUED]-prefixed title → queued ring.
        result = status_border_css('[QUEUED] LTP run')
        self.assertEqual(result, QUEUED_BOX_SHADOW)

    def test_expired_returns_terminal_box_shadow(self):
        # Phase 37 STATUS-01: [X]-prefixed title (final short-letter marker, the legacy
        # bracket-word [EXPIRED] form was retired once plan 37-07's re-title sweep proved
        # the developer database held none of it) → terminal-failure ring.
        self.assertEqual(status_border_css('[X] x'), TERMINAL_BOX_SHADOW)

    def test_cancelled_returns_terminal_box_shadow(self):
        # Phase 37 STATUS-01: [C]-prefixed title (final short-letter marker, the legacy
        # bracket-word [CANCELLED] form was retired) → terminal-failure ring.
        self.assertEqual(status_border_css('[C] x'), TERMINAL_BOX_SHADOW)

    def test_failed_returns_terminal_box_shadow(self):
        # Phase 37 STATUS-01: [F]-prefixed title (final short-letter marker, the legacy
        # bracket-word [FAILED] form was retired) → terminal-failure ring.
        self.assertEqual(status_border_css('[F] x'), TERMINAL_BOX_SHADOW)

    def test_weathered_returns_terminal_box_shadow(self):
        # Phase 37 STATUS-01 (D-02, migrated from the Phase 23 Plan 02 [WEATHERED] bracket-
        # word form): [W]-prefixed title → terminal-failure ring, same as [C] -- both
        # CampaignRun terminal run_status outcomes get the ring.
        self.assertEqual(status_border_css('[W] x'), TERMINAL_BOX_SHADOW)

    def test_unverified_returns_empty_string(self):
        # D-09: placed bucket → '' (Phase 8's dashed border owns this distinction).
        self.assertEqual(status_border_css('[UNVERIFIED] x'), '')

    def test_clean_title_returns_empty_string(self):
        # D-09: no known prefix → '' (placed, no extra ring).
        self.assertEqual(status_border_css('Some title'), '')

    def test_queued_box_shadow_differs_from_terminal(self):
        # D-08: queued and terminal-failure are visually distinct.
        self.assertNotEqual(QUEUED_BOX_SHADOW, TERMINAL_BOX_SHADOW)

    def test_no_dashed_in_queued_result(self):
        # D-09: dashed border-style is reserved for Phase 8's is_verified cue.
        self.assertNotIn('dashed', status_border_css('[QUEUED] x'))

    def test_no_dashed_in_terminal_result(self):
        # D-09: terminal ring must not use dashed border-style.
        self.assertNotIn('dashed', status_border_css('[X] x'))
        self.assertNotIn('dashed', status_border_css('[C] x'))
        self.assertNotIn('dashed', status_border_css('[F] x'))
        self.assertNotIn('dashed', status_border_css('[W] x'))

    def test_no_dashed_in_placed_result(self):
        # D-09: placed events return '' — inherently no dashed.
        self.assertNotIn('dashed', status_border_css('[UNVERIFIED] x'))
        self.assertNotIn('dashed', status_border_css('clean title'))


def _make_weeks(proposals):
    """Build a minimal fake weeks structure from a flat list of proposal strings."""
    events = [SimpleNamespace(proposal=p) for p in proposals]
    day = SimpleNamespace(all_day_events=events, events=[])
    return [[day]]


class VisibleProposalsTest(TestCase):
    def test_groups_by_color_with_collision_handling(self):
        # D-04: colliding proposal codes share one legend entry.
        # Build expected mapping dynamically so the test is robust regardless
        # of whether the chosen proposals actually collide.
        proposals = ['PROP-A', 'PROP-B', 'PROP-C', '']
        weeks = _make_weeks(proposals)

        expected_by_color = {}
        for p in proposals:
            color = proposal_color(p)
            normalized = (p or '').strip().upper()
            label = normalized if normalized else CLASSICAL_SCHEDULE_LABEL
            expected_by_color.setdefault(color, set()).add(label)

        result = visible_proposals(weeks)
        self.assertEqual(len(result), len(expected_by_color))

        for entry in result:
            self.assertIn(entry['color'], expected_by_color)
            actual_labels = set(entry['label'].split(', '))
            self.assertEqual(actual_labels, expected_by_color[entry['color']])

    def test_neutral_slot_color_for_empty_proposal(self):
        # D-05: empty-proposal event → NEUTRAL_SLOT_COLOR entry.
        weeks = _make_weeks([''])
        result = visible_proposals(weeks)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['color'], NEUTRAL_SLOT_COLOR)

    def test_neutral_slot_label_is_classical_schedule(self):
        # D-06: empty-proposal legend entry is labeled 'Classical schedule'.
        weeks = _make_weeks([''])
        result = visible_proposals(weeks)
        self.assertEqual(result[0]['label'], CLASSICAL_SCHEDULE_LABEL)

    def test_neutral_slot_ordered_last(self):
        # D-06 / 09-UI-SPEC Legend Layout: Classical schedule entry appears last.
        weeks = _make_weeks(['PROP-A', ''])
        result = visible_proposals(weeks)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[-1]['color'], NEUTRAL_SLOT_COLOR)
        self.assertEqual(result[-1]['label'], CLASSICAL_SCHEDULE_LABEL)

    def test_absent_proposal_not_in_result(self):
        # D-02: only proposals present in weeks appear in the legend.
        weeks = _make_weeks(['PROP-A'])
        result = visible_proposals(weeks)
        all_labels = ' '.join(e['label'] for e in result)
        self.assertNotIn('PROP-B', all_labels)


class TextColorForBgTest(TestCase):
    def test_all_palette_colors_return_white(self):
        # DISPLAY-08: all 8 PROPOSAL_PALETTE entries achieve WCAG AA 4.5:1 with white text.
        for hex_color in PROPOSAL_PALETTE:
            with self.subTest(hex_color=hex_color):
                self.assertEqual(text_color_for_bg(hex_color), '#fff')

    def test_neutral_slot_returns_white(self):
        # DISPLAY-08: NEUTRAL_SLOT_COLOR (#5a6268) achieves WCAG AA with white text.
        self.assertEqual(text_color_for_bg(NEUTRAL_SLOT_COLOR), '#fff')

    def test_bright_background_returns_black(self):
        # DISPLAY-08: formula correctness — pure white background yields black text.
        self.assertEqual(text_color_for_bg('#ffffff'), '#000')

    def test_pure_black_returns_white(self):
        # DISPLAY-08: pure black background yields white text (maximum contrast).
        self.assertEqual(text_color_for_bg('#000000'), '#fff')


class NeutralSlotColorTagTest(TestCase):
    def test_returns_neutral_slot_color(self):
        # quick-260724-osc: assignment tag exposes NEUTRAL_SLOT_COLOR to templates
        # without a magic literal, so calendar.html can compare bg_color against it.
        self.assertEqual(neutral_slot_color(), NEUTRAL_SLOT_COLOR)


class TelescopeColorTest(TestCase):
    def test_same_input_same_output(self):
        # quick-260724-osc: deterministic — same telescope always returns the same color.
        self.assertEqual(telescope_color('NTT'), telescope_color('NTT'))

    def test_normalization_case_insensitive(self):
        # quick-260724-osc: .strip().upper() applied before hashing.
        self.assertEqual(telescope_color('NTT'), telescope_color('ntt'))

    def test_normalization_trailing_space(self):
        # quick-260724-osc: whitespace stripped before hashing.
        self.assertEqual(telescope_color('NTT'), telescope_color('NTT '))

    def test_empty_string_returns_neutral_slot(self):
        # quick-260724-osc: empty telescope → dedicated neutral slot, mirrors proposal_color.
        self.assertEqual(telescope_color(''), NEUTRAL_SLOT_COLOR)

    def test_blank_string_returns_neutral_slot(self):
        # quick-260724-osc: whitespace-only telescope → neutral slot after .strip().
        self.assertEqual(telescope_color('   '), NEUTRAL_SLOT_COLOR)

    def test_none_returns_neutral_slot(self):
        # quick-260724-osc: None telescope → neutral slot (defensive fallback).
        self.assertEqual(telescope_color(None), NEUTRAL_SLOT_COLOR)

    def test_nonempty_telescope_returns_palette_member(self):
        # quick-260724-tiz: non-empty telescopes map to one of the TELESCOPE_PALETTE entries.
        color = telescope_color('NTT')
        self.assertIn(color, TELESCOPE_PALETTE)


def _make_classical_weeks(entries):
    """Build a minimal fake weeks structure from a flat list of (proposal, telescope) tuples."""
    events = [SimpleNamespace(proposal=p, telescope=t) for p, t in entries]
    day = SimpleNamespace(all_day_events=events, events=[])
    return [[day]]


class VisibleClassicalTelescopesTest(TestCase):
    def test_only_classical_events_contribute(self):
        # quick-260724-osc: proposal-having events are excluded even when their
        # telescope field is set.
        weeks = _make_classical_weeks([('', 'NTT'), ('LTP2025A-004', 'FTS')])
        result = visible_classical_telescopes(weeks)
        all_labels = ' '.join(e['label'] for e in result)
        self.assertIn('NTT', all_labels)
        self.assertNotIn('FTS', all_labels)

    def test_groups_by_color_with_collision_handling(self):
        # quick-260724-osc: colliding telescope names share one legend entry.
        # Build expected mapping dynamically so the test is robust regardless
        # of whether the chosen telescopes actually collide.
        telescopes = ['NTT', 'FTS', 'DUPONT']
        entries = [('', t) for t in telescopes]
        weeks = _make_classical_weeks(entries)

        expected_by_color = {}
        for t in telescopes:
            color = telescope_color(t)
            normalized = t.strip().upper()
            expected_by_color.setdefault(color, set()).add(normalized)

        result = visible_classical_telescopes(weeks)
        self.assertEqual(len(result), len(expected_by_color))

        for entry in result:
            self.assertIn(entry['color'], expected_by_color)
            actual_labels = set(entry['label'].split(', '))
            self.assertEqual(actual_labels, expected_by_color[entry['color']])

    def test_absent_telescope_not_in_result(self):
        # quick-260724-osc: telescopes absent from the visible weeks do not appear.
        weeks = _make_classical_weeks([('', 'NTT')])
        result = visible_classical_telescopes(weeks)
        all_labels = ' '.join(e['label'] for e in result)
        self.assertNotIn('FTS', all_labels)

    def test_supports_dict_based_day_objects(self):
        # quick-260724-osc: mirrors visible_proposals's dual dict-vs-attribute support.
        event = SimpleNamespace(proposal='', telescope='NTT')
        day = {'all_day_events': [event], 'events': []}
        weeks = [[day]]
        result = visible_classical_telescopes(weeks)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'NTT')


class TestTelescopeStripeContrast(TestCase):
    """quick-260724-vb0: WCAG contrast audit gating TELESCOPE_PALETTE (vs #ffffff, the
    legend chip's only background), TELESCOPE_STRIPE_PALETTE (vs NEUTRAL_SLOT_COLOR, the
    stripe's only fill neighbour) and STRIPE_OUTER_EDGE_COLOR, plus a live-arithmetic
    proof that one 8-color palette cannot clear 3:1 against both backgrounds at once.
    """

    def test_contrast_ratio_white_vs_black_is_21(self):
        # quick-260724-vb0: formula sanity check -- maximum possible contrast.
        self.assertAlmostEqual(_contrast_ratio('#ffffff', '#000000'), 21.0, places=1)

    def test_contrast_ratio_self_is_one(self):
        # quick-260724-vb0: a color against itself has no contrast.
        self.assertAlmostEqual(_contrast_ratio('#3987e5', '#3987e5'), 1.0, places=6)

    def test_contrast_ratio_is_order_independent(self):
        # quick-260724-vb0: (L1+0.05)/(L2+0.05) with L1/L2 as lighter/darker means
        # swapping arguments must not change the result.
        self.assertAlmostEqual(
            _contrast_ratio('#3987e5', '#ffffff'),
            _contrast_ratio('#ffffff', '#3987e5'),
            places=6,
        )

    def test_white_vs_neutral_slot_color_is_6_21(self):
        # quick-260724-vb0: verified starting value from the plan's key_finding table.
        self.assertAlmostEqual(_contrast_ratio('#ffffff', NEUTRAL_SLOT_COLOR), 6.21, places=2)

    def test_stripe_palette_clears_neutral_slot_gate(self):
        # quick-260724-vb0: every TELESCOPE_STRIPE_PALETTE entry clears 3.4:1 against
        # NEUTRAL_SLOT_COLOR -- the stripe's only fill neighbour (its right/inner edge).
        # This passes on the values TELESCOPE_STRIPE_PALETTE ships with in this task.
        for hex_color in TELESCOPE_STRIPE_PALETTE:
            with self.subTest(hex_color=hex_color):
                self.assertGreaterEqual(_contrast_ratio(hex_color, NEUTRAL_SLOT_COLOR), 3.4)

    def test_legend_palette_clears_white_gate(self):
        # quick-260724-vb0: this is the RED gate. Four TELESCOPE_PALETTE entries
        # (#199e70, #c98500, #9085e9, #e66767) measure 3.41/3.07/3.13/3.23 against
        # white on the palette as shipped by quick-260724-tiz -- Task 2 retunes them.
        # Do not weaken this threshold to make it pass prematurely.
        for hex_color in TELESCOPE_PALETTE:
            with self.subTest(hex_color=hex_color):
                self.assertGreaterEqual(_contrast_ratio(hex_color, '#ffffff'), 3.5)

    def test_stripe_outer_edge_clears_white_gate(self):
        # quick-260724-vb0: the opaque outward-facing edge must read as a line
        # against the white day cell.
        self.assertGreaterEqual(_contrast_ratio(STRIPE_OUTER_EDGE_COLOR, '#ffffff'), 3.0)

    def test_stripe_outer_edge_clears_terminal_ring_adjacency(self):
        """quick-260724-vb0 Task 3: status_border_css's terminal branch emits an
        outward box-shadow ring on .cal-event-all-day, painted outside the chip's
        border box -- so on a [C]/[X]/[F]/[W] classical
        chip, the ring's inner neighbour along the chip's left flank is the stripe's
        outward-facing edge (STRIPE_OUTER_EDGE_COLOR), not the fill-facing side.
        rgba(160, 0, 0, 0.55) composited over the white day cell is #cb7373; the
        outer edge must clear 3:1 against that composited color."""
        ring_composited_over_white = '#cb7373'
        self.assertGreaterEqual(_contrast_ratio(STRIPE_OUTER_EDGE_COLOR, ring_composited_over_white), 3.0)

    def test_palettes_are_parallel_arrays(self):
        # quick-260724-vb0: equal length, and a telescope name resolves to the same
        # index in both palettes -- the two lists are one hash away from each other,
        # not independently shuffled sets.
        self.assertEqual(len(TELESCOPE_PALETTE), len(TELESCOPE_STRIPE_PALETTE))
        for name in ('NTT', 'FTS', 'DUPONT', 'Aqawan 1: Turbina', 'SOAR'):
            with self.subTest(name=name):
                legend_idx = TELESCOPE_PALETTE.index(telescope_color(name))
                stripe_idx = TELESCOPE_STRIPE_PALETTE.index(telescope_stripe_color(name))
                self.assertEqual(legend_idx, stripe_idx)

    def test_stripe_color_normalization_mirrors_telescope_color(self):
        # quick-260724-vb0: casing and surrounding-whitespace variants of one name
        # return one stripe color, same normalization contract as telescope_color.
        self.assertEqual(telescope_stripe_color('NTT'), telescope_stripe_color('ntt'))
        self.assertEqual(telescope_stripe_color('NTT'), telescope_stripe_color('NTT '))

    def test_stripe_color_blank_returns_neutral_slot(self):
        # quick-260724-vb0: blank/None telescope -> a deliberately invisible stripe
        # (NEUTRAL_SLOT_COLOR matches the chip's own fill). The >= 3.4:1 gate above
        # covers palette entries only, never this fallback.
        self.assertEqual(telescope_stripe_color(''), NEUTRAL_SLOT_COLOR)
        self.assertEqual(telescope_stripe_color('   '), NEUTRAL_SLOT_COLOR)
        self.assertEqual(telescope_stripe_color(None), NEUTRAL_SLOT_COLOR)

    def test_legend_and_stripe_resolve_to_different_hex_for_same_telescope(self):
        # quick-260724-vb0: a classical chip renders a TELESCOPE_STRIPE_PALETTE member
        # while the legend chip renders a TELESCOPE_PALETTE member -- for the same
        # telescope these are two different hex values.
        self.assertNotEqual(telescope_color('NTT'), telescope_stripe_color('NTT'))

    def test_one_palette_cannot_clear_both_backgrounds(self):
        # quick-260724-vb0 key_finding: derive the luminance bands live from
        # _relative_luminance rather than hardcoding the ceiling/floor, so this stays
        # true if NEUTRAL_SLOT_COLOR is ever revisited.
        l_white = _relative_luminance('#ffffff')
        l_neutral = _relative_luminance(NEUTRAL_SLOT_COLOR)

        # 3:1 against white requires L <= (l_white + 0.05) / 3 - 0.05.
        max_l_for_white_gate = (l_white + 0.05) / 3 - 0.05
        # 3:1 against NEUTRAL_SLOT_COLOR on the lighter side requires
        # L >= 3 * (l_neutral + 0.05) - 0.05.
        min_l_for_neutral_lighter_side = 3 * (l_neutral + 0.05) - 0.05
        # The lighter band never intersects the white constraint.
        self.assertLess(max_l_for_white_gate, min_l_for_neutral_lighter_side)

        # The darker-side band (L <= (l_neutral + 0.05) / 3 - 0.05) is effectively
        # black-only: the darkest fully-saturated sRGB primary (#0000ff) already
        # exceeds that ceiling, so no set of 8 mutually distinguishable hues fits.
        max_l_for_neutral_darker_side = (l_neutral + 0.05) / 3 - 0.05
        l_pure_blue = _relative_luminance('#0000ff')
        self.assertGreater(l_pure_blue, max_l_for_neutral_darker_side)


class TestProjectorMarkerRings(TestCase):
    """PROJ-03/D-02 (Phase 34 Plan 03): status_border_css() extended to recognize the
    observation projector's own terse bracket-letter marker vocabulary, without touching
    any existing bracket-word assertion."""

    def test_q_returns_queued_box_shadow(self):
        self.assertEqual(status_border_css('[Q] 2m0 3I/ATLAS'), QUEUED_BOX_SHADOW)

    def test_x_returns_terminal_box_shadow(self):
        self.assertEqual(status_border_css('[X] 2m0 3I/ATLAS'), TERMINAL_BOX_SHADOW)

    def test_c_returns_terminal_box_shadow(self):
        self.assertEqual(status_border_css('[C] 2m0 3I/ATLAS'), TERMINAL_BOX_SHADOW)

    def test_f_returns_terminal_box_shadow(self):
        self.assertEqual(status_border_css('[F] 2m0 3I/ATLAS'), TERMINAL_BOX_SHADOW)

    def test_question_mark_returns_terminal_box_shadow(self):
        # D-13: an inconsistent record reads as needing attention, not as normal.
        self.assertEqual(status_border_css('[?] 2m0 3I/ATLAS'), TERMINAL_BOX_SHADOW)

    def test_s_returns_empty_string(self):
        # The placed/observed bucket keeps no ring -- same reasoning as [UNVERIFIED].
        self.assertEqual(status_border_css('[S] 2m0 3I/ATLAS'), '')

    def test_o_returns_empty_string(self):
        self.assertEqual(status_border_css('[O] FTS 3I/ATLAS'), '')

    def test_no_dashed_in_any_projector_marker_result(self):
        for title in ('[Q] a', '[X] a', '[C] a', '[F] a', '[?] a', '[S] a', '[O] a'):
            with self.subTest(title=title):
                self.assertNotIn('dashed', status_border_css(title))

    def test_all_pre_existing_bracket_word_assertions_still_hold(self):
        # Belt-and-suspenders re-assertion alongside the individual tests above (T-34-18).
        # The legacy bracket-word [EXPIRED]/[CANCELLED]/[FAILED]/[WEATHERED] forms were
        # retired in Phase 37 Plan 07 once the re-title sweep proved the developer database
        # held none of them -- [QUEUED] is a separate, deliberately preserved word-form
        # literal (37-01 SUMMARY key-decision) that this deletion never touched.
        self.assertEqual(status_border_css('[QUEUED] x'), QUEUED_BOX_SHADOW)
        self.assertEqual(status_border_css('[X] x'), TERMINAL_BOX_SHADOW)
        self.assertEqual(status_border_css('[C] x'), TERMINAL_BOX_SHADOW)
        self.assertEqual(status_border_css('[F] x'), TERMINAL_BOX_SHADOW)
        self.assertEqual(status_border_css('[W] x'), TERMINAL_BOX_SHADOW)
        self.assertEqual(status_border_css('[UNVERIFIED] x'), '')
        self.assertEqual(status_border_css('bare title'), '')


class TestObservationStatusLegend(TestCase):
    """PROJ-03/D-02/D-04 (Phase 37): observation_status_legend() exposes the fixed
    marker vocabulary (read from status_vocabulary.LEGEND, Phase 37 STATUS-01/02) to
    calendar.html."""

    def test_returns_nine_entries_covering_every_marker(self):
        legend = observation_status_legend()
        self.assertEqual(len(legend), 9)
        markers = [entry['marker'] for entry in legend]
        self.assertEqual(markers, ['[Q]', '[S]', '[O]', '[X]', '[C]', '[F]', '[W]', '[?]', '[U]'])

    def test_every_entry_has_a_non_empty_label(self):
        for entry in observation_status_legend():
            with self.subTest(marker=entry['marker']):
                self.assertTrue(entry['label'])

    def test_never_raises_and_takes_no_arguments_and_reads_no_database(self):
        # Two calls in a row with no setup and no queryset touch -- proves it's a fixed
        # vocabulary, not data-driven.
        first = observation_status_legend()
        second = observation_status_legend()
        self.assertEqual(first, second)

    def test_only_the_unused_entry_is_filterable(self):
        """WR-09 (37-REVIEW.md): calendar.html's click-to-filter swatch branches on
        entry.filterable, never on a bare `entry.marker == '[U]'` literal comparison."""
        legend = observation_status_legend()
        filterable_markers = [entry['marker'] for entry in legend if entry['filterable']]
        self.assertEqual(filterable_markers, ['[U]'])


class TestObservationSeriesDecoration(TestCase):
    """PROJ-04/PROJ-05 (Phase 34 Plan 03, D-04): observation_series_decoration() reads
    series identity from CalendarEventMeta.observation_group/.observation_record at
    request time, mirroring campaign_decoration()'s guards.

    Builds ObservationRecord fixtures with NonSiderealTargetFactory and
    ObservationRecord.objects.create(), and CalendarEventMeta companion rows directly --
    this class tests the tag, not the projector (see TestRenderThenReprojectByteIdentical
    below for the one test that DOES exercise the real projector end-to-end). The
    observation projector's post_save receiver is globally wired (34-01), so it is
    disconnected around every ObservationRecord.objects.create() call here (34-01
    precedent, test_campaign_attribution.py) -- otherwise it would auto-create a second,
    unwanted CalendarEvent+CalendarEventMeta for the same record and collide with this
    class's own directly-built companion rows on the OneToOneField.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign = TargetList.objects.create(name='Series Decoration Campaign')
        cls.approved_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.pending_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT3',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )

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

    def _make_unwindowed_record(self, observation_id: str) -> ObservationRecord:
        """A record record_time_window() cannot derive a window for (no scheduled_start/
        end, no parameters['start']/['end']) -- the "cannot derive window" sibling."""
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
                status='PENDING',
                parameters={'proposal': 'TESTPROP', 'instrument_type': '2M0-SCICAM-MUSCAT'},
            )
        finally:
            post_save.connect(
                receiver_on_record_save,
                sender=ObservationRecord,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.post_save',
            )

    def _add_to_group(self, group: ObservationGroup, *records: ObservationRecord) -> None:
        """group.observation_records.add() fires the projector's m2m_changed receiver
        (D-15, closes the add-after-save gap), which would auto-create a CalendarEvent+
        CalendarEventMeta per added record -- disconnected around every .add() call here
        for the same reason _make_record() disconnects post_save."""
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

    def _make_malformed_window_record(self, observation_id: str) -> ObservationRecord:
        """WR-04: parameters['start']/['end'] are JSON numbers, not ISO strings --
        datetime.fromisoformat() raises TypeError (not ValueError) for this shape."""
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
                status='PENDING',
                parameters={
                    'proposal': 'TESTPROP',
                    'instrument_type': '2M0-SCICAM-MUSCAT',
                    'start': 12345,
                    'end': 12346,
                },
            )
        finally:
            post_save.connect(
                receiver_on_record_save,
                sender=ObservationRecord,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.post_save',
            )

    def _make_event(self, title: str = 'series test event') -> CalendarEvent:
        return CalendarEvent.objects.create(
            title=title,
            start_time=datetime(2026, 9, 1, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 1, 21, 0, tzinfo=dt_timezone.utc),
        )

    def test_three_member_group_returns_name_index_size_and_links(self):
        r1 = self._make_record(
            'series-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r2 = self._make_record(
            'series-2',
            datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )
        r3 = self._make_record(
            'series-3',
            datetime(2026, 9, 3, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 4, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='3I/ATLAS nightly cadence')
        self._add_to_group(group, r1, r2, r3)

        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, observation_record=r2, observation_group=group)

        result = observation_series_decoration(AUTHENTICATED_CONTEXT, event)
        self.assertIsNotNone(result)
        self.assertEqual(result['group_name'], '3I/ATLAS nightly cadence')
        self.assertEqual(result['group_pk'], group.pk)
        self.assertEqual(result['size'], 3)
        self.assertEqual(result['index'], 2)
        self.assertEqual(result['group_list_url'], reverse('tom_observations:group-list'))
        self.assertEqual(result['record_url'], reverse('tom_observations:detail', args=[r2.pk]))

    def test_unattributed_group_hides_name_from_anonymous_viewer_but_shows_it_authenticated(self):
        """A companion row with no `run` at all (the common, projector-only case, never
        routed through campaign attribution) must not publish the group name to an
        anonymous viewer -- the same protection campaign_decoration() already gives an
        attributed-but-not-yet-public run, extended to the far more common un-attributed
        case."""
        r1 = self._make_record(
            'unattributed-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r2 = self._make_record(
            'unattributed-2',
            datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Unattributed portal RequestGroup name')
        self._add_to_group(group, r1, r2)
        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, observation_record=r1, observation_group=group, run=None)

        self.assertIsNone(observation_series_decoration(ANONYMOUS_CONTEXT, event))

        result = observation_series_decoration(AUTHENTICATED_CONTEXT, event)
        self.assertIsNotNone(result)
        self.assertEqual(result['group_name'], 'Unattributed portal RequestGroup name')

    def test_viewer_gate_applies_regardless_of_run_and_run_visibility_is_a_second_gate(self):
        """The viewer check is a single, unconditional rule -- it does not matter whether
        `run` is None, approved, or pending review, an anonymous viewer never sees the
        group name. `run.is_publicly_visible` is an ADDITIONAL constraint layered on top
        for an authenticated viewer, not an alternative to the viewer check. Six cases:
        (run None / approved / pending) x (anonymous / authenticated)."""

        def make_group_event(group_name: str, run: CampaignRun | None) -> CalendarEvent:
            r1 = self._make_record(
                f'{group_name}-1',
                datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
                datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
            )
            r2 = self._make_record(
                f'{group_name}-2',
                datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
                datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
            )
            group = ObservationGroup.objects.create(name=group_name)
            self._add_to_group(group, r1, r2)
            event = self._make_event(title=f'{group_name} event')
            CalendarEventMeta.objects.create(event=event, observation_record=r1, observation_group=group, run=run)
            return event

        no_run_event = make_group_event('Gate matrix: no run', None)
        approved_event = make_group_event('Gate matrix: approved run', self.approved_run)
        pending_event = make_group_event('Gate matrix: pending run', self.pending_run)

        # Anonymous: hidden in all three cases -- the viewer check alone is enough to
        # reject an anonymous request before run visibility is even considered.
        self.assertIsNone(observation_series_decoration(ANONYMOUS_CONTEXT, no_run_event))
        self.assertIsNone(observation_series_decoration(ANONYMOUS_CONTEXT, approved_event))
        self.assertIsNone(observation_series_decoration(ANONYMOUS_CONTEXT, pending_event))

        # Authenticated: no run and an approved run both show the group name; a
        # pending-review run's own is_publicly_visible gate still hides it even from an
        # authenticated viewer -- that gate protects the run's own attribution, not the
        # viewer identity.
        self.assertIsNotNone(observation_series_decoration(AUTHENTICATED_CONTEXT, no_run_event))
        self.assertIsNotNone(observation_series_decoration(AUTHENTICATED_CONTEXT, approved_event))
        self.assertIsNone(observation_series_decoration(AUTHENTICATED_CONTEXT, pending_event))

    def test_members_numbered_by_window_start_independent_of_pk_order(self):
        # Created in reverse chronological order, so pk order is the OPPOSITE of window
        # order -- proves the sort key is window_start, not pk or creation order.
        r_latest = self._make_record(
            'reverse-1',
            datetime(2026, 9, 3, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 4, 6, 0, tzinfo=dt_timezone.utc),
        )
        r_earliest = self._make_record(
            'reverse-2',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Reverse-order group')
        self._add_to_group(group, r_latest, r_earliest)

        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, observation_record=r_earliest, observation_group=group)

        result = observation_series_decoration(AUTHENTICATED_CONTEXT, event)
        self.assertEqual(result['index'], 1)  # earliest window, despite the later pk

    def test_unwindowed_sibling_sorts_last_without_raising(self):
        r_windowed = self._make_record(
            'unwindowed-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r_unwindowed = self._make_unwindowed_record('unwindowed-2')
        group = ObservationGroup.objects.create(name='Half-projectable group')
        self._add_to_group(group, r_windowed, r_unwindowed)

        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, observation_record=r_windowed, observation_group=group)

        result = observation_series_decoration(AUTHENTICATED_CONTEXT, event)
        self.assertIsNotNone(result)
        self.assertEqual(result['size'], 2)
        self.assertEqual(result['index'], 1)  # windowed sibling sorts first, unwindowed last

    def test_malformed_window_sibling_sorts_last_without_raising(self):
        """WR-04: a sibling whose parameters['start']/['end'] are JSON numbers (fromisoformat()
        raises TypeError, not ValueError) must sort last, not 500 the modal."""
        r_windowed = self._make_record(
            'malformed-window-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        r_malformed = self._make_malformed_window_record('malformed-window-2')
        group = ObservationGroup.objects.create(name='Malformed-window group')
        self._add_to_group(group, r_windowed, r_malformed)

        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, observation_record=r_windowed, observation_group=group)

        result = observation_series_decoration(AUTHENTICATED_CONTEXT, event)
        self.assertIsNotNone(result)
        self.assertEqual(result['size'], 2)
        self.assertEqual(result['index'], 1)  # windowed sibling sorts first, malformed one last

    def test_returns_none_for_no_companion_row(self):
        event = self._make_event()
        self.assertIsNone(observation_series_decoration(AUTHENTICATED_CONTEXT, event))

    def test_returns_none_for_companion_row_with_no_group(self):
        r1 = self._make_record(
            'nogroup-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, observation_record=r1, observation_group=None)
        self.assertIsNone(observation_series_decoration(AUTHENTICATED_CONTEXT, event))

    def test_returns_none_for_single_member_group(self):
        r1 = self._make_record(
            'solo-1',
            datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        group = ObservationGroup.objects.create(name='Solo group')
        self._add_to_group(group, r1)
        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, observation_record=r1, observation_group=group)
        self.assertIsNone(observation_series_decoration(AUTHENTICATED_CONTEXT, event))

    def test_returns_none_for_non_calendar_event_value(self):
        self.assertIsNone(observation_series_decoration(AUTHENTICATED_CONTEXT, None))
        self.assertIsNone(observation_series_decoration(AUTHENTICATED_CONTEXT, 'not an event'))

    def test_render_then_reproject_leaves_title_and_description_byte_identical(self):
        # Exercises the real projector end-to-end (no receiver disconnect here) -- the
        # exact scenario spike 003 found broken when campaign text was written into event
        # fields: rendering this tag must never itself write, and re-projecting the
        # record afterwards must not erase what the tag already read from the link.
        r1 = ObservationRecord.objects.create(
            target=self.target,
            facility='LCO',
            observation_id='byte-1',
            status='COMPLETED',
            scheduled_start=datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 1, 1, 19, tzinfo=dt_timezone.utc),
            parameters={'proposal': 'TESTPROP', 'instrument_type': '2M0-SCICAM-MUSCAT'},
        )
        r2 = ObservationRecord.objects.create(
            target=self.target,
            facility='LCO',
            observation_id='byte-2',
            status='COMPLETED',
            scheduled_start=datetime(2026, 9, 2, 1, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 2, 1, 19, tzinfo=dt_timezone.utc),
            parameters={'proposal': 'TESTPROP', 'instrument_type': '2M0-SCICAM-MUSCAT'},
        )
        group = ObservationGroup.objects.create(name='Byte-identical group')
        group.observation_records.add(r1, r2)

        meta = r1.calendar_event_meta
        meta.observation_group = group
        meta.save(update_fields=['observation_group'])
        event = meta.event

        before_title = event.title
        before_description = event.description

        result = observation_series_decoration(AUTHENTICATED_CONTEXT, event)
        self.assertIsNotNone(result)

        op.project_record(r1)
        event.refresh_from_db()
        self.assertEqual(event.title, before_title)
        self.assertEqual(event.description, before_description)


class TestRunTally(TestCase):
    """Phase 37 Plan 06 (D-09, TALLY-01): run_tally() mirrors campaign_decoration()'s guard
    shape and renders campaign_tally.tally_segments() for the calendar pop-up's
    attributed-run block -- read from CalendarEventMeta.run at request time, never written
    anywhere.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Run Tally Campaign')
        cls.approved_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.pending_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT3',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )

    def _make_event(self, title: str = 'run tally event') -> CalendarEvent:
        return CalendarEvent.objects.create(
            title=title,
            start_time=datetime(2026, 9, 1, 20, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 1, 21, 0, tzinfo=dt_timezone.utc),
        )

    def test_non_calendar_event_returns_none_without_raising(self):
        self.assertIsNone(run_tally('not-an-event'))
        self.assertIsNone(run_tally(None))

    def test_event_with_no_companion_row_returns_none(self):
        event = self._make_event()
        self.assertIsNone(run_tally(event))

    def test_companion_row_with_no_run_returns_none(self):
        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, run=None)
        self.assertIsNone(run_tally(event))

    def test_pending_review_run_returns_none(self):
        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, run=self.pending_run)
        self.assertIsNone(run_tally(event))

    def test_approved_run_returns_tally_dict_with_ordered_segments(self):
        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, run=self.approved_run)
        result = run_tally(event)
        self.assertIsNotNone(result)
        self.assertEqual(result['groups'], 0)
        self.assertEqual(result['records'], 0)
        markers = [segment['marker'] for segment in result['segments']]
        self.assertEqual(markers, ['[O]', '[S]', '[X/F]', '[U]'])
        self.assertIn('0 groups', result['summary'])

    def test_not_yet_known_unused_figure_renders_a_word_not_a_zero(self):
        """A run with no allocation events and no proposal code leaves nights_unused as
        None/unused_known=False -- the summary word must never claim zero."""
        event = self._make_event()
        CalendarEventMeta.objects.create(event=event, run=self.approved_run)
        result = run_tally(event)
        unused_segment = result['segments'][-1]
        self.assertFalse(unused_segment['known'])
        self.assertIsNone(unused_segment['count'])
        self.assertIn('not yet known', result['summary'])


class TestUnusedNightDecoration(TestCase):
    """Phase 37 Plan 06 (D-12/D-13/D-14, UNUSED-01): unused_night_decoration() delegates to
    campaign_tally.is_unused_allocation_night() -- the single shared classifier the campaign
    table's unused count also reads (D-15) -- and never writes CalendarEvent.title.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Unused Night Campaign')
        cls.active_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='NTT/EFOSC2',
            window_start=date(2026, 9, 1),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.cancelled_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='NTT/EFOSC2',
            window_start=date(2026, 9, 4),
            window_end=date(2026, 9, 6),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.CANCELLED,
        )
        cls.weathered_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='NTT/EFOSC2',
            window_start=date(2026, 9, 7),
            window_end=date(2026, 9, 9),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.WEATHER_TECH_FAILURE,
        )

    def _make_alloc_event(self, url: str, end_time: datetime, title: str = 'alloc event') -> CalendarEvent:
        return CalendarEvent.objects.create(
            title=title,
            start_time=end_time - timedelta(hours=8),
            end_time=end_time,
            url=url,
        )

    def test_non_calendar_event_returns_none_without_raising(self):
        self.assertIsNone(unused_night_decoration('not-an-event'))
        self.assertIsNone(unused_night_decoration(None))

    def test_event_with_no_companion_row_returns_none(self):
        event = self._make_alloc_event(f'ALLOC:{self.active_run.pk}:2026-09-01', timezone.now() - timedelta(days=1))
        self.assertIsNone(unused_night_decoration(event))

    def test_non_allocation_url_returns_none(self):
        """A RUN: container event (or any non-ALLOC: url) is never classified at all."""
        event = CalendarEvent.objects.create(
            title='RUN container event',
            start_time=timezone.now() - timedelta(days=2),
            end_time=timezone.now() - timedelta(days=1),
            url=f'RUN:{self.active_run.pk}',
        )
        CalendarEventMeta.objects.create(event=event, run=self.active_run)
        self.assertIsNone(unused_night_decoration(event))

    def test_companion_row_with_no_run_returns_none(self):
        event = self._make_alloc_event(f'ALLOC:{self.active_run.pk}:2026-09-01', timezone.now() - timedelta(days=1))
        CalendarEventMeta.objects.create(event=event, run=None)
        self.assertIsNone(unused_night_decoration(event))

    def test_future_night_returns_none(self):
        event = self._make_alloc_event(f'ALLOC:{self.active_run.pk}:2026-09-02', timezone.now() + timedelta(days=1))
        CalendarEventMeta.objects.create(event=event, run=self.active_run)
        self.assertIsNone(unused_night_decoration(event))

    def test_elapsed_night_on_active_run_returns_token_and_label(self):
        event = self._make_alloc_event(f'ALLOC:{self.active_run.pk}:2026-09-01', timezone.now() - timedelta(days=1))
        CalendarEventMeta.objects.create(event=event, run=self.active_run)
        result = unused_night_decoration(event)
        self.assertIsNotNone(result)
        self.assertEqual(result['token'], '[U]')
        self.assertEqual(result['label'], 'Unused awarded night')

    def test_elapsed_night_on_cancelled_run_returns_none(self):
        """D-14: staff run status always wins -- a cancelled run's elapsed night is never
        unused, whatever the time."""
        event = self._make_alloc_event(f'ALLOC:{self.cancelled_run.pk}:2026-09-04', timezone.now() - timedelta(days=1))
        CalendarEventMeta.objects.create(event=event, run=self.cancelled_run)
        self.assertIsNone(unused_night_decoration(event))

    def test_elapsed_night_on_weathered_run_returns_none(self):
        """D-14: a weather/technical-failure run's elapsed night is never unused either."""
        event = self._make_alloc_event(f'ALLOC:{self.weathered_run.pk}:2026-09-07', timezone.now() - timedelta(days=1))
        CalendarEventMeta.objects.create(event=event, run=self.weathered_run)
        self.assertIsNone(unused_night_decoration(event))

    def test_elapsed_night_on_pending_review_run_returns_none(self):
        """WR-06 (37-REVIEW.md): every sibling tally surface (run_tally(), the campaign
        roll-up's queryset-level exclude) gates on is_publicly_visible -- this tag must too,
        so a pending-review run's elapsed allocation night never surfaces the public [U]
        token, chip or tooltip on the anonymous calendar."""
        pending_run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='NTT/EFOSC2',
            window_start=date(2026, 9, 10),
            window_end=date(2026, 9, 12),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        event = self._make_alloc_event(f'ALLOC:{pending_run.pk}:2026-09-10', timezone.now() - timedelta(days=1))
        CalendarEventMeta.objects.create(event=event, run=pending_run)
        self.assertIsNone(unused_night_decoration(event))

    def test_rendering_does_not_change_stored_title(self):
        event = self._make_alloc_event(
            f'ALLOC:{self.active_run.pk}:2026-09-01', timezone.now() - timedelta(days=1), title='NTT/EFOSC2'
        )
        CalendarEventMeta.objects.create(event=event, run=self.active_run)
        before_title = event.title
        unused_night_decoration(event)
        event.refresh_from_db()
        self.assertEqual(event.title, before_title)
