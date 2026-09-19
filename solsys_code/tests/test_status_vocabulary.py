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


class TestRunStatusMarker(TestCase):
    """Task 2 (D-02): run-level RunStatus values join the same short-letter vocabulary --
    [C] for CANCELLED, [W] for WEATHER_TECH_FAILURE, no marker for the other six values."""

    def test_run_status_marker_has_exactly_two_entries(self) -> None:
        from solsys_code.models import CampaignRun
        from solsys_code.status_vocabulary import RUN_STATUS_MARKER

        self.assertEqual(len(RUN_STATUS_MARKER), 2)
        self.assertEqual(sorted(RUN_STATUS_MARKER.values()), ['[C]', '[W]'])
        self.assertEqual(RUN_STATUS_MARKER[CampaignRun.RunStatus.CANCELLED], '[C]')
        self.assertEqual(RUN_STATUS_MARKER[CampaignRun.RunStatus.WEATHER_TECH_FAILURE], '[W]')

    def test_no_other_run_status_value_has_a_marker(self) -> None:
        from solsys_code.models import CampaignRun
        from solsys_code.status_vocabulary import RUN_STATUS_MARKER

        marked = {CampaignRun.RunStatus.CANCELLED, CampaignRun.RunStatus.WEATHER_TECH_FAILURE}
        for value in CampaignRun.RunStatus:
            with self.subTest(value=value):
                if value in marked:
                    self.assertIn(value, RUN_STATUS_MARKER)
                else:
                    self.assertNotIn(value, RUN_STATUS_MARKER)

    def test_cancelled_and_legacy_cancelled_share_the_same_terminal_ring(self) -> None:
        from solsys_code.templatetags.calendar_display_extras import status_border_css

        self.assertEqual(status_border_css('[C] NTT EFOSC2'), status_border_css('[CANCELLED] NTT EFOSC2'))
        self.assertNotEqual(status_border_css('[C] NTT EFOSC2'), '')


class _StubFacility:
    """Minimal facility double exposing name/get_terminal_observing_states/
    get_failed_observing_states -- neither GEM nor ESO's real package is installed in this
    environment, so a real facility class cannot be imported for those two cases."""

    def __init__(self, name, terminal_states, failed_states):
        self.name = name
        self._terminal_states = terminal_states
        self._failed_states = failed_states

    def get_terminal_observing_states(self):
        return self._terminal_states

    def get_failed_observing_states(self):
        return self._failed_states


class TestFacilityAwareClassifier(TestCase):
    """Task 3 (D-05): one facility-aware terminal classifier -- observed_states_for(),
    failed_states_for(), classify_record(), OBSERVED_STATES_BY_FACILITY."""

    def test_observed_states_for_lco_is_terminal_minus_failed(self) -> None:
        from tom_observations.facilities.lco import LCOFacility

        from solsys_code.status_vocabulary import observed_states_for

        self.assertEqual(observed_states_for(LCOFacility()), frozenset({'COMPLETED'}))

    def test_observed_states_for_gem_is_empty(self) -> None:
        from solsys_code.status_vocabulary import observed_states_for

        gem = _StubFacility('GEM', {'TRIGGERED', 'ON_HOLD'}, {'ON_HOLD'})
        self.assertEqual(observed_states_for(gem), frozenset())

    def test_observed_states_for_eso_is_empty(self) -> None:
        from solsys_code.status_vocabulary import observed_states_for

        eso = _StubFacility('ESO', {'COMPLETED'}, set())
        self.assertEqual(observed_states_for(eso), frozenset())

    def test_observed_states_by_facility_has_exactly_gem_and_eso(self) -> None:
        from solsys_code.status_vocabulary import OBSERVED_STATES_BY_FACILITY

        self.assertEqual(sorted(OBSERVED_STATES_BY_FACILITY), ['ESO', 'GEM'])
        self.assertEqual(OBSERVED_STATES_BY_FACILITY['GEM'], frozenset())
        self.assertEqual(OBSERVED_STATES_BY_FACILITY['ESO'], frozenset())

    def test_classify_record_inconsistent_for_half_set_schedule(self) -> None:
        from unittest.mock import MagicMock

        from solsys_code.status_vocabulary import DisplayState, classify_record

        record = MagicMock(scheduled_start=None, scheduled_end='not-none', status='PENDING')
        facility = _StubFacility('LCO', {'COMPLETED'}, set())
        self.assertEqual(classify_record(record, facility), DisplayState.INCONSISTENT)

    def test_classify_record_observed_for_completed_status(self) -> None:
        from unittest.mock import MagicMock

        from solsys_code.status_vocabulary import DisplayState, classify_record

        record = MagicMock(scheduled_start=None, scheduled_end=None, status='COMPLETED')
        facility = _StubFacility('LCO', {'COMPLETED'}, set())
        self.assertEqual(classify_record(record, facility), DisplayState.OBSERVED)

    def test_classify_record_scheduled_for_full_block_unresolved_status(self) -> None:
        from unittest.mock import MagicMock

        from solsys_code.status_vocabulary import DisplayState, classify_record

        record = MagicMock(scheduled_start='a', scheduled_end='b', status='PENDING')
        facility = _StubFacility('LCO', {'COMPLETED'}, set())
        self.assertEqual(classify_record(record, facility), DisplayState.SCHEDULED)

    def test_classify_record_queued_otherwise(self) -> None:
        from unittest.mock import MagicMock

        from solsys_code.status_vocabulary import DisplayState, classify_record

        record = MagicMock(scheduled_start=None, scheduled_end=None, status='PENDING')
        facility = _StubFacility('LCO', {'COMPLETED'}, set())
        self.assertEqual(classify_record(record, facility), DisplayState.QUEUED)

    def test_classify_record_failure_state_maps_through_failure_marker_states(self) -> None:
        from unittest.mock import MagicMock

        from solsys_code.status_vocabulary import DisplayState, classify_record

        facility = _StubFacility('LCO', {'CANCELED'}, {'CANCELED'})
        record = MagicMock(scheduled_start=None, scheduled_end=None, status='CANCELED')
        self.assertEqual(classify_record(record, facility), DisplayState.CANCELLED)
