import os
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import requests
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from solsys_code.utils import read_psv


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.json.return_value = payload
    response.text = str(payload)
    return response


@patch('solsys_code.management.commands.fetch_mpc_obs.requests.get')
class TestFetchMPCObs(SimpleTestCase):
    def setUp(self) -> None:
        test_csv_file = Path(__file__).parent / 'test_data' / 'test_ades_df.csv'
        df = pd.read_csv(test_csv_file, index_col=0, dtype=str)
        # The API returns a list of records (one per observation) with None for missing fields
        self.records = df.where(df.notna(), None).to_dict('records')
        self.records[1]['deprecated'] = 'X'
        return super().setUp()

    def test_writes_psv(self, mock_get):
        mock_get.return_value = _mock_response([{'ADES_DF': self.records}])
        out = StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / '33933.psv'
            call_command('fetch_mpc_obs', '2000 LE29', '--output', str(output), stdout=out)
            ades_data = read_psv(output)

        mock_get.assert_called_once()
        self.assertEqual(mock_get.call_args.kwargs['json'], {'desigs': ['2000 LE29'], 'output_format': ['ADES_DF']})
        self.assertEqual(out.getvalue().strip(), f'Wrote 5 observations of 2000 LE29 to {output}')
        self.assertEqual(ades_data['header'], {'version': '2022'})
        self.assertEqual(ades_data['keywords'][:3], ['permID', 'provID', 'trkSub'])
        self.assertNotIn('deprecated', ades_data['keywords'])
        self.assertEqual(len(ades_data['observations']), 5)
        first = dict(zip(ades_data['keywords'], ades_data['observations'][0], strict=False))
        self.assertEqual(first['permID'].strip(), '33933')
        self.assertEqual(first['obsTime'].strip(), '1971-09-16T04:20:36.672Z')
        self.assertEqual(first['ra'].strip(), '348.84408')

    def test_keep_deprecated_and_default_filename(self, mock_get):
        mock_get.return_value = _mock_response([{'ADES_DF': self.records}])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Default output is relative to the current directory; chdir so the test stays self-contained
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                call_command('fetch_mpc_obs', 'P/2026 T1', '--keep-deprecated', stdout=StringIO())
                output = Path(tmpdir) / 'P_2026_T1.psv'
                self.assertTrue(output.exists())
                ades_data = read_psv(output)
            finally:
                os.chdir(cwd)

        self.assertIn('deprecated', ades_data['keywords'])
        self.assertEqual(len(ades_data['observations']), 6)

    def test_api_error(self, mock_get):
        mock_get.return_value = _mock_response(
            {'error': 'general_error', 'message': 'get-obs failed with Bad Label'}, status_code=500
        )

        with self.assertRaisesMessage(CommandError, "returned 500 for '2P/Encke': get-obs failed with Bad Label"):
            call_command('fetch_mpc_obs', '2P/Encke', stdout=StringIO())

    def test_no_observations(self, mock_get):
        mock_get.return_value = _mock_response([{'ADES_DF': []}])

        with self.assertRaisesMessage(CommandError, "No observations found for 'ZZ99'"):
            call_command('fetch_mpc_obs', 'ZZ99', stdout=StringIO())

    def test_request_exception(self, mock_get):
        mock_get.side_effect = requests.ConnectionError('boom')

        with self.assertRaisesMessage(CommandError, 'MPC Observations API request failed: boom'):
            call_command('fetch_mpc_obs', '2P', stdout=StringIO())
