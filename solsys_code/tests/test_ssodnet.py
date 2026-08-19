"""
Tests for solsys_code/ssodnet.py.
"""

from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from solsys_code.ssodnet import SsODNetDataService


class FakeTarget:
    def __init__(self, name):
        self.name = name


class TestBuildQueryParametersFromTarget(SimpleTestCase):
    def test_returns_target_name(self):
        params = SsODNetDataService().build_query_parameters_from_target(FakeTarget('(433) Eros'))
        self.assertEqual(params, {'name': '(433) Eros'})


class TestQueryService(SimpleTestCase):
    @patch('solsys_code.ssodnet.rocks.Rock')
    def test_returns_rock_for_known_object(self, mock_rock_cls):
        mock_rock = Mock(id_='Eros')
        mock_rock_cls.return_value = mock_rock

        result = SsODNetDataService().query_service({'name': '(433) Eros'})

        mock_rock_cls.assert_called_once_with('(433) Eros')
        self.assertIs(result, mock_rock)

    @patch('solsys_code.ssodnet.rocks.Rock')
    def test_returns_none_when_not_found(self, mock_rock_cls):
        mock_rock_cls.return_value = Mock(id_=None)

        result = SsODNetDataService().query_service({'name': 'not a real object'})

        self.assertIsNone(result)

    @patch('solsys_code.ssodnet.rocks.Rock')
    def test_returns_none_on_lookup_error(self, mock_rock_cls):
        mock_rock_cls.side_effect = Exception('boom')

        result = SsODNetDataService().query_service({'name': '(433) Eros'})

        self.assertIsNone(result)

    def test_returns_none_without_a_name(self):
        result = SsODNetDataService().query_service({})
        self.assertIsNone(result)
