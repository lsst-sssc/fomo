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
    proposal_color,
    status_border_css,
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
