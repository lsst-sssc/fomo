"""
Tests for solsys_code/ssodnet.py.

THIS IS A SCAFFOLD: only the scaffold itself is tested (imports cleanly, class exists
with the expected method names). Add real tests here as SsODNetDataService gets
implemented -- e.g. mock `rocks.Rock()` the same way test_views.py's
TestJPLSBDBQuery mocks `requests.get` for JPLSBDBQuery, and use
tom_targets.tests.factories.NonSiderealTargetFactory for any Target fixtures (per
CLAUDE.md -- FOMO is exclusively non-sidereal targets).
"""

from django.test import SimpleTestCase

from solsys_code.ssodnet import SsODNetDataService


class TestSsODNetDataServiceScaffold(SimpleTestCase):
    def test_class_has_expected_hooks(self):
        self.assertTrue(hasattr(SsODNetDataService, 'build_query_parameters_from_target'))
        self.assertTrue(hasattr(SsODNetDataService, 'query_service'))

    def test_build_query_parameters_from_target_returns_name(self):
        class FakeTarget:
            name = '(433) Eros'

        params = SsODNetDataService().build_query_parameters_from_target(FakeTarget())
        self.assertEqual(params, {'name': '(433) Eros'})
