import logging
import mimetypes

import astropy.io.ascii
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time, TimezoneInfo
from django.core.files.storage import default_storage
from tom_dataproducts.data_processor import DataProcessor
from tom_dataproducts.exceptions import InvalidFileFormatException

logger = logging.getLogger(__name__)

# ADES Pipe Separated Value files aren't known to the stdlib; without this they are
# guessed as `None` and `process_data` would reject them as an unsupported file type.
mimetypes.add_type('text/csv', '.psv')

# ADES `rmsRA`/`rmsDec` are in arcseconds, and `rmsRA` already includes the cos(dec)
# factor (see the ADES standard), so both map directly onto the model's error fields.
ERROR_UNITS = 'arcsec'


class ADESProcessor(DataProcessor):
    """DataProcessor for astrometry (and any accompanying photometry) in IAU ADES format."""

    def data_type_override(self):
        """Returns the DataProduct type whose ReducedDatums this processor emits."""
        return 'astrometry'

    def process_data(self, data_product):
        """
        Routes an ADES processing call to a method specific to a file-format.

        :param data_product: ADES Astrometry DataProduct or pandas.DataFrame which will be processed into
            the specified format for database ingestion
        :type data_product: DataProduct|pandas.DataFrame

        :returns: python list of 3-tuples, each with a timestamp and corresponding data, and source
        :rtype: list
        """

        try:
            mimetype = mimetypes.guess_type(data_product.data.path)[0]
        except NotImplementedError:
            mimetype = 'text/plain'
        logger.debug(f'Processing ADES data with mimetype {mimetype}')

        if mimetype in self.PLAINTEXT_MIMETYPES:
            astrometry = self._process_astrometry_from_plaintext(data_product)
            return [(datum.pop('timestamp'), datum, datum.pop('source', 'MPC')) for datum in astrometry]
        else:
            raise InvalidFileFormatException('Unsupported file type')

    def _process_astrometry_from_plaintext(self, data_product):
        """
        Processes the ADES astrometry and photometry data from a plaintext file (in ADES Pipe Separated Value (PSV)
        format) into a list of dicts.
        Details on the ADES standard: https://data.minorplanetcenter.net/mpcops/documentation/ades/

        :param data_product: DataProduct with a populated 'data' FileField containing the ADES PSV file to be processed
        :type data_product: DataProduct
        """
        astrometry = []
        data_file = default_storage.open(data_product.data.name, 'r')
        # NEOCP/Scout observation files can carry all-numeric station codes (e.g. '703'), which
        # would otherwise be parsed as integers, dropping the leading zero of codes like '046'.
        data = astropy.io.ascii.read(data_file.read(), converters={'stn': str})
        if len(data) < 1:
            raise InvalidFileFormatException('Empty table or invalid file type')

        # Mapping between returned quantities and ADES columns.
        # There two versions, one for the actual ADES standard compliant from e.g.
        # 'xmltopsv.py' or Astrometrica and one for the data from MPC Explorer/the Observations API
        # which has all lowercase column names.
        mapping_ades = {
            'time': 'obsTime',
            'ra_error': 'rmsRA',
            'dec_error': 'rmsDec',
            'magnitude': 'mag',
            'mag_error': 'rmsMag',
        }
        mapping_mpcx = {
            'time': 'obstime',
            'ra_error': 'rmsra',
            'dec_error': 'rmsdec',
            'magnitude': 'mag',
            'mag_error': 'rmsmag',
        }
        mapping = mapping_ades
        if 'obstime' in data.colnames:
            mapping = mapping_mpcx
        try:
            utc = TimezoneInfo(utc_offset=0 * u.hour)

            for row in data:
                time = Time(row[mapping['time']], format='isot', scale='utc')
                time.format = 'datetime'
                value = {
                    'timestamp': time.to_datetime(timezone=utc),
                    'filter': str(row['band']),
                    'telescope': row['stn'],
                }
                value['ra'] = float(row['ra'])
                value['dec'] = float(row['dec'])
                for key, col in mapping.items():
                    if key != 'time':
                        value[key] = None
                        if np.ma.is_masked(row[col]) is False:
                            value[key] = float(row[col])
                self._add_error_units(value)
                astrometry.append(value)
        except Exception as e:
            raise InvalidFileFormatException(e) from e
        return astrometry

    def _process_astrometry_from_df(self, df):
        """
        Processes the ADES astrometry and photometry data from a pandas DataFrame into a list of dicts.

        The `stn` column should be read as a string by the caller (e.g. `dtype={'stn': str}`);
        an all-numeric station code that pandas has already parsed as an integer has lost any
        leading zero by the time it gets here.

        :param df: ADES pandas.DataFrame which will be processed into a list of dicts for the measurements
        :type df: pandas.DataFrame
        :return: python list containing the astrometric data from the DataFrame
        :rtype: list
        """
        astrometry = []

        try:
            utc = TimezoneInfo(utc_offset=0 * u.hour)

            for row in df.itertuples(index=False):
                time = Time(row.obstime, format='isot', scale='utc')
                time.format = 'datetime'
                value = {
                    'timestamp': time.to_datetime(timezone=utc),
                    'filter': str(row.band),
                    'telescope': str(row.stn),
                }
                value['ra'] = float(row.ra)
                value['ra_error'] = None
                if pd.isna(row.rmsra) is False:
                    value['ra_error'] = row.rmsra
                value['dec'] = float(row.dec)
                value['dec_error'] = None
                if pd.isna(row.rmsdec) is False:
                    value['dec_error'] = row.rmsdec
                value['magnitude'] = None
                if pd.isna(row.mag) is False:
                    value['magnitude'] = row.mag
                value['mag_error'] = None
                if pd.isna(row.rmsmag) is False:
                    value['mag_error'] = row.rmsmag
                self._add_error_units(value)
                astrometry.append(value)
        except Exception as e:
            raise InvalidFileFormatException(e) from e
        return astrometry

    @staticmethod
    def _add_error_units(value):
        """
        Tags the astrometric uncertainties with their units so that they are stored on
        `AstrometryReducedDatum` rather than falling through into its free-form `value` field.

        :param value: dict of measurements for a single observation, modified in place
        :type value: dict
        """
        for key in ('ra_error', 'dec_error'):
            if value.get(key) is not None:
                value[f'{key}_units'] = ERROR_UNITS
