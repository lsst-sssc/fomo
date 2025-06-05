from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from django.test import TestCase

# Import models to test
from solsys_code.solsys_code_observatory.views import MPCObscodeFetcher


class TestMPCObscodeFetcher(TestCase):
    def setUp(self) -> None:
        self.fetcher = MPCObscodeFetcher()
        # Is this better than a json.load from a file...
        self.fetcher.obs_data = {
            'created_at': 'Sat, 25 May 2019 00:11:26 GMT',
            'firstdate': '2007-01-05',
            'lastdate': None,
            'longitude': '149.07085',
            'name': 'Siding Spring-Faulkes Telescope South',
            'name_latex': 'Siding Spring-Faulkes Telescope South',
            'name_utf8': 'Siding Spring-Faulkes Telescope South',
            'obscode': 'E10',
            'observations_type': 'optical',
            'old_names': None,
            'rhocosphi': '0.855632',
            'rhosinphi': '-0.516198',
            'short_name': 'Siding Spring-Faulkes Telescope South',
            'updated_at': 'Tue, 15 Apr 2025 20:52:50 GMT',
            'uses_two_line_observations': False,
            'web_link': None,
        }

    @patch('requests.get')
    def test_query_failure_invalid_code(self, mock_get):
        """test query of invalid code"""
        mock_response = MagicMock()
        mock_response.status_code = 501
        mock_response.ok = False
        mock_response.json.return_value = (
            b'{\n  "error": "input_error",\n  "message": "Malformed input: bad obscode\n}\n'
        )
        mock_get.return_value = mock_response

        result = self.fetcher.query('FOO')
        self.assertEqual(result, None)
        self.assertIsNone(self.fetcher.obs_data)

    @patch('requests.get')
    def test_query_failure_unknown_code(self, mock_get):
        """test query of valid but unknown code"""
        mock_response = MagicMock()
        mock_response.status_code = 501
        mock_response.ok = False
        mock_response.json.return_value = (
            b'{\n  "error": "input_error",\n  "message": "obscodes failed: No obscode \'Y69\'"\n}\n'
        )
        mock_get.return_value = mock_response

        result = self.fetcher.query('Y69')
        self.assertEqual(result, None)
        self.assertIsNone(self.fetcher.obs_data)

    def test_to_observatory(self):
        obs = self.fetcher.to_observatory()

        tobs_data = self.fetcher.obs_data
        self.assertEqual(obs.obscode, 'E10')
        self.assertEqual(obs.name, tobs_data['name_utf8'])
        self.assertEqual(obs.short_name, tobs_data['short_name'])
        self.assertEqual(obs.old_names, '')
        self.assertAlmostEqual(obs.lon, 149.07085)
        self.assertAlmostEqual(obs.lat, -31.272803156)
        self.assertAlmostEqual(
            obs.altitude, 1154.26336, 5
        )  # Doesn't quite match find_orb due to different axis ratio/flattening factor used
        self.assertEqual(obs.observations_type, 0)
        self.assertEqual(obs.uses_two_line_obs, False)
        self.assertEqual(obs.created, datetime(2019, 5, 25, 0, 11, 16, tzinfo=timezone.utc))
        self.assertEqual(obs.modified, datetime(2025, 4, 15, 20, 52, 50, tzinfo=timezone.utc))
