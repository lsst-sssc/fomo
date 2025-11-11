from pathlib import Path

from astropy.table import Table
from django.test import SimpleTestCase

from solsys_code.utils import convert_ades_to_table, read_psv


class TestReadPSV(SimpleTestCase):
    def setUp(self) -> None:
        self.test_psv_file = Path(__file__).parent / 'test_data' / 'sample.psv'
        return super().setUp()

    def test_read_psv(self):
        expected_items = ['header', 'keywords', 'observations']
        expected_data = [
            [
                '217P',
                '',
                '',
                'CCD',
                'Z24',
                '2025-11-11T05:27:02.7Z',
                '138.036656',
                '+12.403909',
                '0.144',
                '0.100',
                'Gaia3',
                '18.78',
                '0.270',
                'G',
                'g',
                'Gaia3',
                '1.4',
                '1.45',
                '1.1',
                '170',
                '0.17',
                '332',
                '',
                '',
            ],
            [
                '217P',
                '',
                '',
                'CCD',
                'Z24',
                '2025-11-11T05:43:44.6Z',
                '138.038423',
                '+12.403937',
                '0.119',
                '0.072',
                'Gaia3',
                '17.59',
                '0.244',
                'G',
                'i',
                'Gaia3',
                '2.9',
                '1.94',
                '1.1',
                '170',
                '0.13',
                '474',
                '',
                '',
            ],
        ]

        psv_data = read_psv(self.test_psv_file)
        self.assertEqual(len(expected_items), len(list(psv_data.keys())))
        for key in psv_data:
            self.assertIn(key, expected_items)
        self.assertEqual(len(expected_data[0]), len(psv_data['keywords']))
        for row in [psv_data['observations'][0], psv_data['observations'][-1]]:
            self.assertEqual(row, expected_data.pop(0))


class TestConvertADESToTable(SimpleTestCase):
    def setUp(self) -> None:
        self.test_psv_file = Path(__file__).parent / 'test_data' / 'sample.psv'
        self.ades_data = read_psv(self.test_psv_file)
        return super().setUp()

    def test_convert_ades_to_table(self):
        ades_table = convert_ades_to_table(self.ades_data)
        self.assertEqual(Table, type(ades_table))
        self.assertEqual(len(ades_table), 36)
        expected_columns = [
            'permID',
            'provID',
            'trkSub',
            'mode',
            'stn',
            'obsTime',
            'ra',
            'dec',
            'rmsRA',
            'rmsDec',
            'astCat',
            'mag',
            'rmsMag',
            'band',
            'fltr',
            'photCat',
            'photAp',
            'logSNR',
            'seeing',
            'exp',
            'rmsFit',
            'nStars',
            'notes',
            'remarks',
        ]
        self.assertEqual(len(ades_table.colnames), len(expected_columns))
        for col in ades_table.colnames:
            self.assertIn(col, expected_columns)
