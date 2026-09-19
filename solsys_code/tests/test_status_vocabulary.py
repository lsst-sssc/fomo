"""One marker, one module -- the end-to-end agreement test for Phase 37 (STATUS-01/02).

For a real ``ObservationRecord`` fixture in a queued state and in the portal ``CANCELED``
state, proves the marker at the front of ``observation_projector.title_for()``'s output:

1. is a member of ``status_vocabulary.MARKER.values()``,
2. resolves back to the same display state through ``status_vocabulary.state_for_title()``,
3. appears in ``status_vocabulary.LEGEND``, and
4. lands in the ring bucket ``calendar_display_extras.status_border_css()`` gives it today.

Always uses ``tom_targets.tests.factories.NonSiderealTargetFactory`` for any ``Target``
fixture -- never ``SiderealTargetFactory`` (CLAUDE.md: FOMO is exclusively for Solar
System / non-sidereal targets).
"""


from django.test import TestCase
from tom_observations.models import ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import observation_projector as op
from solsys_code import status_vocabulary
from solsys_code.templatetags.calendar_display_extras import status_border_css


class TestOneMarkerOneModule(TestCase):
    """STATUS-01 tracer: the full path -- record state -> projector title marker ->
    template-tag ring -> rendered legend entry -- is sourced end-to-end from
    ``status_vocabulary``."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target = NonSiderealTargetFactory.create()

    def setUp(self) -> None:
        op.reset_facility_cache()

    def _make_record(self, observation_id: str, status: str) -> ObservationRecord:
        params = {
            'proposal': 'TESTPROP',
            'instrument_type': '2M0-SCICAM-MUSCAT',
            'start': '2026-09-01T00:00:00',
            'end': '2026-09-02T00:00:00',
        }
        return ObservationRecord.objects.create(
            target=self.target,
            facility='LCO',
            observation_id=observation_id,
            status=status,
            parameters=params,
        )

    def _assert_marker_agrees_end_to_end(self, record: ObservationRecord) -> None:
        facility = op.facility_for(record)
        stage = op.stage_for(record, facility)
        token = op.telescope_token(record, stage, 'MUSCAT')
        title = op.title_for(record, stage, token, record.target.name, facility)
        marker = title.split(' ', 1)[0]

        # 1. the marker is a member of MARKER.values()
        self.assertIn(marker, status_vocabulary.MARKER.values())

        # 2. resolves back to the same display state through state_for_title()
        resolved_state = status_vocabulary.state_for_title(title)
        self.assertIsNotNone(resolved_state)
        self.assertEqual(status_vocabulary.MARKER[resolved_state], marker)

        # 3. appears in LEGEND
        legend_markers = [entry['marker'] for entry in status_vocabulary.LEGEND]
        self.assertIn(marker, legend_markers)

        # 4. lands in the ring bucket status_border_css() gives it today
        css = status_border_css(title)
        if resolved_state in status_vocabulary.RING_TERMINAL_STATES:
            self.assertNotEqual(css, '')
        elif resolved_state in status_vocabulary.RING_QUEUED_STATES:
            self.assertNotEqual(css, '')
        else:
            self.assertEqual(css, '')

    def test_queued_record_agrees_end_to_end(self) -> None:
        record = self._make_record('status-vocab-queued', status='PENDING')
        self._assert_marker_agrees_end_to_end(record)

    def test_portal_cancelled_record_agrees_end_to_end(self) -> None:
        record = self._make_record('status-vocab-cancelled', status='CANCELED')
        self._assert_marker_agrees_end_to_end(record)


class TestModuleDocstringInvariants(TestCase):
    """The module docstring states the TALLY-03 invariant and the heavy-import
    constraint (Task 1 acceptance criteria)."""

    def test_docstring_states_tally_03_invariant(self) -> None:
        self.assertIn('run_status', status_vocabulary.__doc__)
        self.assertIn('never', status_vocabulary.__doc__.lower())

    def test_docstring_states_no_heavy_import_constraint(self) -> None:
        self.assertIn('solsys_code.views', status_vocabulary.__doc__)
        self.assertIn('ephem_utils', status_vocabulary.__doc__)

    def test_module_source_never_imports_heavy_modules(self) -> None:
        # Static guard mirroring the plan's own verification step: the module's source
        # text never references the heavy SPICE-loading modules at import time, order-
        # independent (unlike a sys.modules check, which would depend on what other test
        # modules already imported in this same test process).
        import inspect

        source = inspect.getsource(status_vocabulary)
        self.assertNotIn('solsys_code.views', source.replace(status_vocabulary.__doc__, ''))
        self.assertNotIn('import solsys_code.ephem_utils', source)
        self.assertNotIn('from solsys_code.ephem_utils', source)
        self.assertNotIn('from solsys_code import ephem_utils', source)


class TestVocabularyStructure(TestCase):
    """STATUS-01/D-01..D-04: the shape of the one shared vocabulary."""

    def test_legend_has_nine_entries_in_fixed_order(self) -> None:
        markers = [entry['marker'] for entry in status_vocabulary.LEGEND]
        self.assertEqual(markers, ['[Q]', '[S]', '[O]', '[X]', '[C]', '[F]', '[W]', '[?]', '[U]'])

    def test_scheduled_label_is_literally_scheduled(self) -> None:
        self.assertEqual(status_vocabulary.LABEL[status_vocabulary.DisplayState.SCHEDULED], 'Scheduled')

    def test_state_for_title_resolves_markers(self) -> None:
        for state, marker in status_vocabulary.MARKER.items():
            with self.subTest(state=state):
                self.assertEqual(status_vocabulary.state_for_title(f'{marker} NTT EFOSC2'), state)

    def test_state_for_title_resolves_retired_bracket_word_prefixes(self) -> None:
        cases = {
            '[EXPIRED] x': status_vocabulary.DisplayState.WINDOW_EXPIRED,
            '[CANCELLED] x': status_vocabulary.DisplayState.CANCELLED,
            '[FAILED] x': status_vocabulary.DisplayState.FAILED,
            '[WEATHERED] x': status_vocabulary.DisplayState.WEATHERED,
        }
        for title, expected_state in cases.items():
            with self.subTest(title=title):
                self.assertEqual(status_vocabulary.state_for_title(title), expected_state)

    def test_state_for_title_returns_none_for_unknown_title(self) -> None:
        self.assertIsNone(status_vocabulary.state_for_title('Some title'))
        self.assertIsNone(status_vocabulary.state_for_title(''))
        self.assertIsNone(status_vocabulary.state_for_title(None))

    def test_unused_is_in_neither_ring_bucket(self) -> None:
        self.assertNotIn(status_vocabulary.DisplayState.UNUSED, status_vocabulary.RING_QUEUED_STATES)
        self.assertNotIn(status_vocabulary.DisplayState.UNUSED, status_vocabulary.RING_TERMINAL_STATES)
