"""
Tests for solsys_code/ssodnet.py.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from solsys_code.ssodnet import SsODNetDataService, build_card_context
from src.templatetags.solsys_code_extras import ssodnet_card


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

    @patch('solsys_code.ssodnet.rocks.Rock')
    def test_returns_none_for_unresolvable_name(self, mock_rock_cls):
        # Confirmed real-world behaviour (2026-08-19): rocks.Rock() raises
        # KeyError('ssocard') for a name it can't resolve at all, rather than
        # returning a Rock with an empty id.
        mock_rock_cls.side_effect = KeyError('ssocard')

        result = SsODNetDataService().query_service({'name': 'not a real name'})

        self.assertIsNone(result)

    def test_returns_none_without_a_name(self):
        result = SsODNetDataService().query_service({})
        self.assertIsNone(result)


def _fake_bibref(shortbib='Someone+2020', bibcode='2020ABC..1234S', doi='10.0/x'):
    return SimpleNamespace(shortbib=shortbib, bibcode=bibcode, doi=doi)


def _fake_rock(
    name='Eros',
    number=433,
    class_='NEA>Amor',
    parent='Sun',
    system='Sun',
    taxonomy_class='S',
    taxonomy_bibref=None,
    abs_mag=10.4,
    abs_mag_bibref=None,
    diameter=17.6,
    diameter_bibref=None,
):
    """Build a minimal SimpleNamespace mimicking the real rocks.Rock structure
    (confirmed against a live `rocks.Rock('Eros')` call on 2026-08-19) -- just the
    fields build_card_context() actually reads."""
    return SimpleNamespace(
        name=name,
        number=number,
        class_=class_,
        parent=parent,
        system=system,
        parameters=SimpleNamespace(
            physical=SimpleNamespace(
                taxonomy=SimpleNamespace(
                    class_=SimpleNamespace(value=taxonomy_class),
                    bibref=taxonomy_bibref if taxonomy_bibref is not None else [_fake_bibref()],
                ),
                absolute_magnitude=SimpleNamespace(
                    H=SimpleNamespace(value=abs_mag),
                    bibref=abs_mag_bibref if abs_mag_bibref is not None else [_fake_bibref()],
                ),
                diameter=SimpleNamespace(
                    value=diameter,
                    bibref=diameter_bibref if diameter_bibref is not None else [_fake_bibref()],
                ),
            )
        ),
    )


class TestBuildCardContext(SimpleTestCase):
    def test_returns_none_for_none_rock(self):
        self.assertIsNone(build_card_context(None))

    def test_shapes_a_fully_populated_rock(self):
        context = build_card_context(_fake_rock())

        self.assertEqual(context['name'], 'Eros')
        self.assertEqual(context['number'], 433)
        self.assertEqual(context['class_'], 'NEA>Amor')
        self.assertEqual(context['parent'], 'Sun')
        self.assertEqual(context['system'], 'Sun')
        self.assertEqual(context['taxonomy']['value'], 'S')
        self.assertEqual(context['taxonomy']['references'][0]['shortbib'], 'Someone+2020')
        self.assertEqual(context['absolute_magnitude']['value'], 10.4)
        self.assertEqual(context['diameter']['value'], 17.6)
        self.assertEqual(context['diameter']['unit'], 'km')

    def test_nan_diameter_and_magnitude_become_none(self):
        context = build_card_context(_fake_rock(abs_mag=float('nan'), diameter=float('nan')))

        self.assertIsNone(context['absolute_magnitude']['value'])
        self.assertIsNone(context['diameter']['value'])

    def test_empty_taxonomy_class_becomes_none(self):
        context = build_card_context(_fake_rock(taxonomy_class=''))

        self.assertIsNone(context['taxonomy']['value'])

    def test_empty_bibref_entries_are_dropped(self):
        context = build_card_context(_fake_rock(taxonomy_bibref=[_fake_bibref(shortbib='')]))

        self.assertEqual(context['taxonomy']['references'], [])


class TestSsodnetCardTemplateTag(SimpleTestCase):
    @patch('solsys_code.ssodnet.rocks.Rock')
    def test_includes_shaped_ssodnet_data_in_context(self, mock_rock_cls):
        mock_rock_cls.return_value = _fake_rock()

        result = ssodnet_card({'target': FakeTarget('Eros')})

        self.assertIsNotNone(result['ssodnet'])
        self.assertEqual(result['ssodnet']['name'], 'Eros')

    @patch('solsys_code.ssodnet.rocks.Rock')
    def test_none_ssodnet_when_not_found(self, mock_rock_cls):
        mock_rock_cls.side_effect = KeyError('ssocard')

        result = ssodnet_card({'target': FakeTarget('not a real name')})

        self.assertIsNone(result['ssodnet'])
