"""Unit tests for solsys_code.templatetags.calendar_display_extras.

Wave 0 scaffold — written before the module exists (RED). Tests cover the three
public tags: proposal_color (DISPLAY-04, D-04/D-05), status_border_css (DISPLAY-06,
D-08/D-09), and visible_proposals (DISPLAY-07, D-02/D-04/D-06).
"""

from types import SimpleNamespace

from django.test import TestCase

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
    proposal_color,
    status_border_css,
    telescope_color,
    telescope_stripe_color,
    text_color_for_bg,
    visible_classical_telescopes,
    visible_proposals,
)

QUEUED_BOX_SHADOW = 'box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.45);'
TERMINAL_BOX_SHADOW = 'box-shadow: 0 0 0 3px rgba(160, 0, 0, 0.55);'


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
        # D-08: [EXPIRED]-prefixed title → terminal-failure ring.
        self.assertEqual(status_border_css('[EXPIRED] x'), TERMINAL_BOX_SHADOW)

    def test_cancelled_returns_terminal_box_shadow(self):
        # D-08: [CANCELLED]-prefixed title → terminal-failure ring.
        self.assertEqual(status_border_css('[CANCELLED] x'), TERMINAL_BOX_SHADOW)

    def test_failed_returns_terminal_box_shadow(self):
        # D-08: [FAILED]-prefixed title → terminal-failure ring.
        self.assertEqual(status_border_css('[FAILED] x'), TERMINAL_BOX_SHADOW)

    def test_weathered_returns_terminal_box_shadow(self):
        # D-03/D-08 (Phase 23 Plan 02): [WEATHERED]-prefixed title → terminal-failure ring,
        # same as [CANCELLED] -- both CampaignRun terminal run_status outcomes get the ring.
        self.assertEqual(status_border_css('[WEATHERED] x'), TERMINAL_BOX_SHADOW)

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
        self.assertNotIn('dashed', status_border_css('[EXPIRED] x'))
        self.assertNotIn('dashed', status_border_css('[CANCELLED] x'))
        self.assertNotIn('dashed', status_border_css('[FAILED] x'))
        self.assertNotIn('dashed', status_border_css('[WEATHERED] x'))

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
        border box -- so on a [CANCELLED]/[EXPIRED]/[FAILED]/[WEATHERED] classical
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
