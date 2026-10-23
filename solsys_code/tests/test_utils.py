import tempfile
from pathlib import Path

import pandas as pd
from astropy.table import Table
from django.test import SimpleTestCase

from solsys_code.utils import (
    ades_dataframe_to_ades_data,
    apply_zero_aperture_extrapolation,
    convert_ades_to_table,
    perform_zero_aperture_extrapolation_on_psv,
    read_psv,
    write_psv,
    zero_aperture_extrapolation,
)


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
                '',
                'C/2025 K1',
                '',
                'CCD',
                'Z24',
                '2025-11-11T05:43:44.6Z',
                '138.038423',
                '-12.403937',
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
                'K',
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


class TestZeroApertureExtrapolation(SimpleTestCase):
    def setUp(self) -> None:
        test_psv_file = Path(__file__).parent / 'test_data' / 'sample.psv'
        self.ades_data = read_psv(test_psv_file)
        self.ades_table = convert_ades_to_table(self.ades_data)
        return super().setUp()

    def test_zero_aperture_extrapolation(self):
        expected_dts = [
            '2025-11-11T05:27:02.7Z',
            '2025-11-11T05:30:22.0Z',
            '2025-11-11T05:33:42.3Z',
            '2025-11-11T05:37:02.8Z',
            '2025-11-11T05:40:24.0Z',
            '2025-11-11T05:43:44.6Z',
        ]

        derived_zaa = zero_aperture_extrapolation(self.ades_table, make_plots=False)
        self.assertEqual(len(derived_zaa), len(self.ades_table) // 6)  # 6 frames, 6 apertures per frame
        self.assertEqual(expected_dts, list(derived_zaa.keys()))
        # Check known values (pre-computed)
        expected_zaa_ra_dec = [
            (138.036628, 12.403883),
            (138.037016, 12.403832),
            (138.037354, 12.403898),
            (138.037738, 12.404076),
            (138.038080, 12.403972),
            (138.038430, -12.403934),
        ]
        for expected_ra_dec, values in zip(expected_zaa_ra_dec, derived_zaa.values(), strict=False):
            self.assertAlmostEqual(expected_ra_dec[0], values['zero_ap_ra'], places=6)
            self.assertAlmostEqual(expected_ra_dec[1], values['zero_ap_dec'], places=6)


class TestWritePSV(SimpleTestCase):
    def setUp(self) -> None:
        self.test_psv_file = Path(__file__).parent / 'test_data' / 'sample.psv'
        self.ades_data = read_psv(self.test_psv_file)
        self.expected_zaa_ra_dec = {
            '2025-11-11T05:27:02.7Z': {'zero_ap_ra': 138.036628, 'zero_ap_dec': 12.403883},
            '2025-11-11T05:30:22.0Z': {'zero_ap_ra': 138.037016, 'zero_ap_dec': 12.403832},
            '2025-11-11T05:33:42.3Z': {'zero_ap_ra': 138.037354, 'zero_ap_dec': 12.403898},
            '2025-11-11T05:37:02.8Z': {'zero_ap_ra': 138.037738, 'zero_ap_dec': 12.404076},
            '2025-11-11T05:40:24.0Z': {'zero_ap_ra': 138.038080, 'zero_ap_dec': 12.403972},
            '2025-11-11T05:43:44.6Z': {'zero_ap_ra': 138.038430, 'zero_ap_dec': -12.403934},
        }
        self.expected_header_line = (
            'permID |provID     |trkSub  |mode|stn |obsTime                |ra         |dec        |rmsRA'
            '|rmsDec|astCat  |mag  '
            '|rmsMag|band|fltr|photCat |photAp|logSNR|seeing|exp |rmsFit|nStars|notes|remarks'
        )

        self.maxDiff = None
        return super().setUp()

    def _write_and_read_lines(self, ades_data):
        with tempfile.NamedTemporaryFile(prefix='fomo', suffix='.psv') as tmp:
            output_psv_file = Path(tmp.name)
            write_psv(ades_data, output_psv_file)
            with open(output_psv_file, 'r') as f:
                return f.readlines()

    def test_apply_zero_aperture_extrapolation(self):
        extrapolated = apply_zero_aperture_extrapolation(self.ades_data, self.expected_zaa_ra_dec)

        self.assertEqual(extrapolated['header'], self.ades_data['header'])
        self.assertEqual(extrapolated['keywords'], self.ades_data['keywords'])
        # Only the reference aperture (1 of the 6 apertures per frame) survives
        self.assertEqual(len(extrapolated['observations']), len(self.ades_data['observations']) // 6)
        keywords = extrapolated['keywords']
        first = dict(zip(keywords, extrapolated['observations'][0], strict=False))
        self.assertEqual(first['photAp'], '1.8')
        self.assertEqual(first['ra'], '138.036628')
        self.assertEqual(first['dec'], '+12.403883')
        self.assertEqual(first['notes'], 'e')
        last = dict(zip(keywords, extrapolated['observations'][-1], strict=False))
        self.assertEqual(last['dec'], '-12.403934')
        self.assertEqual(last['notes'], 'eK')
        # Input is not modified
        self.assertEqual(len(self.ades_data['observations']), 36)

    def test_write_psv_roundtrip(self):
        written_lines = self._write_and_read_lines(self.ades_data)

        self.assertEqual(written_lines[0].rstrip(), '# version=2022')
        self.assertIn('# observatory\n', written_lines)
        self.assertIn('! mpcCode Z24\n', written_lines)
        with tempfile.NamedTemporaryFile(prefix='fomo', suffix='.psv', mode='w') as tmp:
            tmp.writelines(written_lines)
            tmp.flush()
            reread = read_psv(Path(tmp.name))
        self.assertEqual(reread['header'], self.ades_data['header'])
        self.assertEqual(reread['keywords'], self.ades_data['keywords'])
        self.assertEqual(
            [[v.strip() for v in obs] for obs in reread['observations']],
            [[v.strip() for v in obs] for obs in self.ades_data['observations']],
        )

    def test_write_psv_with_zero_aperture_extrapolation(self):
        extrapolated = apply_zero_aperture_extrapolation(self.ades_data, self.expected_zaa_ra_dec)
        written_lines = self._write_and_read_lines(extrapolated)

        check_217p = False
        for line in written_lines:
            if line.startswith('permID'):
                self.assertEqual(line.strip(), self.expected_header_line)
            elif line.lstrip().startswith('217P') and not check_217p:
                self.assertEqual(
                    line.rstrip(),
                    (
                        '   217P|           |        | CCD|Z24 |2025-11-11T05:27:02.7Z |138.036628 |+12.403883 |0.145'
                        '|0.102 |   Gaia3|18.22'
                        '|0.269 |   G|   g|   Gaia3| 1.8  |1.51  |1.1   | 170|  0.17|   332|e    |'
                    ),
                )
                check_217p = True
            elif line.lstrip().startswith('|C/2025 K1'):
                self.assertEqual(
                    line.rstrip(),
                    (
                        '       |C/2025 K1  |        | CCD|Z24 |2025-11-11T05:43:44.6Z |138.038430 |-12.403934 |0.119'
                        '|0.071 |   Gaia3|18.52'
                        '|0.244 |   G|   i|   Gaia3| 1.8  |1.76  |1.1   | 170|  0.13|   474|eK   |'
                    ),
                )
                break
        self.assertTrue(check_217p)

    def test_perform_zero_aperture_extrapolation_on_psv(self):
        with tempfile.NamedTemporaryFile(prefix='fomo', suffix='.psv') as tmp:
            output_psv_file = Path(tmp.name)
            perform_zero_aperture_extrapolation_on_psv(self.test_psv_file, output_psv_file, make_plots=False)
            reread = read_psv(output_psv_file)

        self.assertEqual(reread['keywords'], self.ades_data['keywords'])
        self.assertEqual(len(reread['observations']), 6)
        for observation in reread['observations']:
            fields = dict(zip(reread['keywords'], observation, strict=False))
            self.assertEqual(fields['photAp'].strip(), '1.8')
            self.assertTrue(fields['notes'].startswith('e'))
            expected = self.expected_zaa_ra_dec[fields['obsTime'].strip()]
            self.assertAlmostEqual(float(fields['ra']), expected['zero_ap_ra'], places=6)
            self.assertAlmostEqual(float(fields['dec']), expected['zero_ap_dec'], places=6)


class TestADESDataFrameToADESData(SimpleTestCase):
    """Conversion of the MPC Observations API `ADES_DF` DataFrames (all-lowercase columns) to PSV."""

    def setUp(self) -> None:
        test_csv_file = Path(__file__).parent / 'test_data' / 'test_ades_df.csv'
        self.df = pd.read_csv(test_csv_file, index_col=0, dtype=str)
        self.maxDiff = None
        return super().setUp()

    def test_ades_dataframe_to_ades_data(self):
        ades_data = ades_dataframe_to_ades_data(self.df)

        self.assertEqual(ades_data['header'], {'version': '2022'})
        # Keywords are the populated columns, renamed to ADES camelCase and in standard ADES order
        self.assertEqual(
            ades_data['keywords'],
            [
                'permID', 'provID', 'trkSub', 'obsID', 'trkID', 'mode', 'stn', 'prog', 'obsTime', 'ra', 'dec',
                'rmsRA', 'rmsDec', 'astCat', 'mag', 'rmsMag', 'band', 'logSNR', 'ref', 'disc', 'subFrm', 'subFmt',
                'precTime', 'precRA', 'precDec', 'remarks',
            ],
        )  # fmt: skip
        self.assertNotIn('Obstype', ades_data['keywords'])
        self.assertEqual(len(ades_data['observations']), 6)
        first = dict(zip(ades_data['keywords'], ades_data['observations'][0], strict=False))
        self.assertEqual(first['permID'], '33933')
        self.assertEqual(first['obsTime'], '1971-09-16T04:20:36.672Z')
        self.assertEqual(first['ra'], '348.84408')
        self.assertEqual(first['disc'], '*')
        self.assertEqual(first['stn'], '808')
        # Missing values become empty fields
        self.assertEqual(first['provID'], '1971 SO')
        self.assertEqual(first['trkSub'], '')
        self.assertEqual(first['mag'], '')

    def test_camelcase_columns_pass_through(self):
        df = self.df.rename(columns={'obstime': 'obsTime', 'rmsra': 'rmsRA', 'permid': 'permID'})
        ades_data = ades_dataframe_to_ades_data(df)

        self.assertEqual(ades_data['keywords'], ades_dataframe_to_ades_data(self.df)['keywords'])

    def test_deprecated_observations_dropped(self):
        df = self.df.copy()
        df.loc[df.index[1], 'deprecated'] = 'X'

        ades_data = ades_dataframe_to_ades_data(df)
        self.assertEqual(len(ades_data['observations']), 5)
        self.assertNotIn('deprecated', ades_data['keywords'])

        ades_data = ades_dataframe_to_ades_data(df, drop_deprecated=False)
        self.assertEqual(len(ades_data['observations']), 6)
        self.assertIn('deprecated', ades_data['keywords'])

    def test_write_psv_from_dataframe(self):
        with tempfile.NamedTemporaryFile(prefix='fomo', suffix='.psv') as tmp:
            output_psv_file = Path(tmp.name)
            write_psv(self.df, output_psv_file)
            with open(output_psv_file, 'r') as f:
                written_lines = f.readlines()
            reread = read_psv(output_psv_file)

        self.assertEqual(written_lines[0].rstrip(), '# version=2022')
        self.assertTrue(written_lines[1].startswith('permID |provID     |trkSub  |'))
        self.assertEqual(len(written_lines), 8)
        self.assertEqual(reread['keywords'], ades_dataframe_to_ades_data(self.df)['keywords'])
        self.assertEqual(len(reread['observations']), 6)
        last = dict(zip(reread['keywords'], reread['observations'][-1], strict=False))
        self.assertEqual(last['stn'].strip(), 'R17')
        self.assertEqual(last['obsTime'].strip(), '2025-04-27T21:51:58.89Z')
        self.assertEqual(last['ra'].strip(), self.df['ra'].iloc[-1])
