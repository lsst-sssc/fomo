import logging
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from astropy.time import Time
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from tom_dataproducts.data_processor import run_data_processor
from tom_dataproducts.models import AstrometryReducedDatum, DataProduct
from tom_observations.tests.factories import NonSiderealTargetFactory

from solsys_code.processors.ades_processor import ADESProcessor

logger = logging.getLogger(__name__)

TEST_DATA = Path(__file__).parent / 'test_data'


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class TestADESProcessor(TestCase):
    """Test the ADESProcessor(DataProcessor) class."""

    def setUp(self):
        self.target = NonSiderealTargetFactory.create()
        self.data_product = DataProduct.objects.create(target=self.target, data_product_type='astrometry')
        self.data_product_filefield_data = SimpleUploadedFile('nonsense.psv', b'somedata')

        self.maxDiff = None

    @patch('solsys_code.processors.ades_processor.ADESProcessor._process_astrometry_from_plaintext')
    def test_process_astrometry_with_plaintext_file(self, mocked_method):
        """Test that ADESProcessor.process_data() calls ADESProcessor._process_astrometry_from_plaintext()."""
        self.data_product.data.save('ades_astrometry.psv', self.data_product_filefield_data)

        # this is the call under test
        ADESProcessor().process_data(self.data_product)
        mocked_method.assert_called_with(self.data_product)

    def test_read(self):
        """Test reading ADES astrometry from PSV.

        The test data is from a query on https://data.minorplanetcenter.net/explorer for '33933'.
        The resulting ADES XML was converted to PSV using `xmltopsv.py` from `iau-ades`
        and trimmed down for the test.
        """
        # read the test data in as a data_product's data
        with open(TEST_DATA / 'test_ades.psv') as ades_file:
            self.data_product.data.save('test_data.psv', ades_file)

        # this is the call under test
        astrometry = ADESProcessor()._process_astrometry_from_plaintext(self.data_product)

        expected_count = 9  # known a priori from test data in test_ades.psv
        self.assertEqual(expected_count, len(astrometry))
        expected_dt = Time(datetime(1971, 9, 16, 4, 20, 36, int(1e6 * 0.672)))
        expected_mag = expected_magerr = None
        self.assertEqual(expected_dt, astrometry[0]['timestamp'])
        self.assertEqual(expected_mag, astrometry[0]['magnitude'])
        self.assertEqual(expected_magerr, astrometry[0]['mag_error'])
        # the 1971 observation has no reported uncertainties, so no units are tagged on either
        self.assertIsNone(astrometry[0]['ra_error'])
        self.assertIsNone(astrometry[0]['dec_error'])
        self.assertNotIn('ra_error_units', astrometry[0])
        self.assertNotIn('dec_error_units', astrometry[0])

        expected_dt = Time(datetime(2025, 4, 27, 21, 51, 58, int(1e6 * 0.890)))
        expected_mag = 19.39
        expected_magerr = 0.137
        self.assertEqual(expected_dt, astrometry[-1]['timestamp'])
        self.assertAlmostEqual(expected_mag, astrometry[-1]['magnitude'])
        self.assertEqual(expected_magerr, astrometry[-1]['mag_error'])
        self.assertAlmostEqual(0.535, astrometry[-1]['ra_error'])
        self.assertAlmostEqual(0.535, astrometry[-1]['dec_error'])
        self.assertEqual('arcsec', astrometry[-1]['ra_error_units'])
        self.assertEqual('arcsec', astrometry[-1]['dec_error_units'])

    def test_read_mpcexplorer(self):
        """Test reading ADES astrometry from PSV with a different format. The MPC's Observations API
        returns the underlying DB table names which aren't camelCase like the ADES standard, so test
        that we can read in that format as well.

        The test data is a query on https://data.minorplanetcenter.net/api/get-obs
        for '2009 DP2' with the `output_format`: `ADES_DF` and the resulting JSON read into
        a pandas DataFrame and then written out to PSV with `ades_df.to_csv('file.psv', sep='|')`
        and heavily trimmed down.
        """
        # read the test data in as a data_product's data
        with open(TEST_DATA / 'test_ades_mpcexplorer.psv') as ades_file:
            self.data_product.data.save('test_data.psv', ades_file)

        # this is the call under test
        astrometry = ADESProcessor()._process_astrometry_from_plaintext(self.data_product)

        expected_count = 5  # known a priori from test data in test_ades_mpcexplorer.psv
        self.assertEqual(expected_count, len(astrometry))
        expected_dt = Time(datetime(2009, 2, 17, 1, 29, 22, int(1e6 * 0.848)))
        expected_mag = 19.8
        expected_magerr = None
        self.assertEqual(expected_dt, astrometry[0]['timestamp'])
        self.assertEqual(expected_mag, astrometry[0]['magnitude'])
        self.assertEqual(expected_magerr, astrometry[0]['mag_error'])
        self.assertIsNone(astrometry[0]['ra_error'])
        self.assertNotIn('ra_error_units', astrometry[0])

        expected_dt = Time(datetime(2026, 3, 10, 3, 39, 4, int(1e6 * 0.900)))
        expected_mag = 21.12
        expected_magerr = 0.202
        self.assertEqual(expected_dt, astrometry[-1]['timestamp'])
        self.assertAlmostEqual(expected_mag, astrometry[-1]['magnitude'])
        self.assertEqual(expected_magerr, astrometry[-1]['mag_error'])
        self.assertAlmostEqual(0.256, astrometry[-1]['ra_error'])
        self.assertAlmostEqual(0.200, astrometry[-1]['dec_error'])
        self.assertEqual('arcsec', astrometry[-1]['ra_error_units'])
        self.assertEqual('arcsec', astrometry[-1]['dec_error_units'])

    def test_read_df(self):
        """Test that astrometry can be read from a Pandas DataFrame

        The test data is from the following query:
        ```
        response = requests.get("https://data.minorplanetcenter.net/api/get-obs",
        json={"desigs": ["33933"], "ades_version": "2022", "output_format": "ADES_DF"})
        if response.ok:
            ades_df = pd.DataFrame(response.json()[0]['ADES_DF'])
            top_n_tail = pd.concat([ades_df.head(3) , ades_df.tail(3)])
            top_n_tail.to_csv("test_ades_df.csv")
        ```
        """
        # read the test data in as a Pandas DataFrame
        self.test_ades_df = pd.read_csv(TEST_DATA / 'test_ades_df.csv', index_col=0)

        # this is the call under test
        astrometry = ADESProcessor()._process_astrometry_from_df(self.test_ades_df)

        expected_count = 6  # known a priori from test data in test_ades_df.csv
        self.assertEqual(expected_count, len(astrometry))
        expected_dt = Time(datetime(1971, 9, 16, 4, 20, 36, int(1e6 * 0.672)))
        expected_mag = expected_magerr = None
        self.assertEqual(expected_dt, astrometry[0]['timestamp'])
        self.assertEqual(expected_mag, astrometry[0]['magnitude'])
        self.assertEqual(expected_magerr, astrometry[0]['mag_error'])
        self.assertIsNone(astrometry[0]['ra_error'])
        self.assertNotIn('ra_error_units', astrometry[0])

        expected_dt = Time(datetime(2025, 4, 27, 21, 51, 58, int(1e6 * 0.890)))
        expected_mag = 19.39
        expected_magerr = 0.137
        self.assertEqual(expected_dt, astrometry[-1]['timestamp'])
        self.assertAlmostEqual(expected_mag, astrometry[-1]['magnitude'])
        self.assertEqual(expected_magerr, astrometry[-1]['mag_error'])
        self.assertAlmostEqual(0.535, astrometry[-1]['ra_error'])
        self.assertEqual('arcsec', astrometry[-1]['ra_error_units'])

    def test_read_all_numeric_station_codes(self):
        """Station codes must survive as strings, including any leading zero.

        NEOCP candidate observation files (e.g. the Observation File linked from a JPL Scout
        object page) are often short and from a single site, so the `stn` column can be entirely
        numeric and would otherwise be parsed as an integer column. The PSV below is synthetic,
        constructed to exercise exactly that case.
        """
        psv = (
            b'# version=2022\n'
            b'permID |trkSub |mode|stn |obsTime                 |ra          |dec         |'
            b'rmsRA  |rmsDec |mag  |rmsMag|band\n'
            b'       |C46HTM1| CCD|703 |2026-09-10T05:27:02.700Z|138.03662   | 12.40388   |'
            b'0.450  |0.450  |19.2 |0.150 |   G\n'
            b'       |C46HTM1| CCD|046 |2026-09-10T05:30:22.000Z|138.03701   | 12.40383   |'
            b'0.470  |0.470  |19.3 |0.160 |   G\n'
        )
        self.data_product.data.save('neocp.psv', SimpleUploadedFile('neocp.psv', psv))

        # this is the call under test
        astrometry = ADESProcessor()._process_astrometry_from_plaintext(self.data_product)

        self.assertEqual(2, len(astrometry))
        self.assertEqual(['703', '046'], [datum['telescope'] for datum in astrometry])
        for datum in astrometry:
            self.assertIsInstance(datum['telescope'], str)

    def test_run_data_processor_creates_astrometry_reduced_datums(self):
        """End-to-end test that a DataProduct of type 'astrometry' is routed to ADESProcessor and
        that the emitted keys land on AstrometryReducedDatum's own fields rather than in `value`.
        """
        with open(TEST_DATA / 'test_ades.psv') as ades_file:
            self.data_product.data.save('test_data.psv', ades_file)

        # this is the call under test
        run_data_processor(self.data_product)

        datums = AstrometryReducedDatum.objects.filter(target=self.target).order_by('timestamp')
        self.assertEqual(9, datums.count())
        for datum in datums:
            self.assertIsNotNone(datum.ra)
            self.assertIsNotNone(datum.dec)
            self.assertEqual('MPC', datum.source_name)

        first = datums.first()
        self.assertAlmostEqual(348.84408, first.ra)
        self.assertAlmostEqual(-6.81600, first.dec)
        self.assertEqual('808', first.telescope)
        self.assertIsNone(first.ra_error)
        self.assertEqual('', first.ra_error_units)

        last = datums.last()
        self.assertAlmostEqual(123.508580, last.ra)
        self.assertAlmostEqual(0.535, last.ra_error)
        self.assertEqual('arcsec', last.ra_error_units)
        self.assertEqual('arcsec', last.dec_error_units)
        self.assertAlmostEqual(19.39, last.value['magnitude'])
        self.assertAlmostEqual(0.137, last.value['mag_error'])
        self.assertEqual('w', last.value['filter'])
