import json
import logging
from unittest.mock import patch, Mock
from django.test import SimpleTestCase
from importlib.resources import files
from solsys_code.views import JPLSBDBQuery
from astropy.table import QTable
from urllib.parse import urlparse, parse_qs, unquote


## Silence logging during tests
logging.disable(logging.CRITICAL)


class TestJPLSBDBQuery(SimpleTestCase):
    def setUp(self):
        self.query = JPLSBDBQuery(
            orbit_class="IEO",
            orbital_constraints=["q<1.3", "i<10.5"]
        )
        self.base_url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
        self.fields = "full_name,first_obs,epoch,e,a,q,i,om,w"
        self.maxDiff = None

        # Load sample response JSON from file
        test_json_fp = files('solsys_code.tests.data').joinpath('test_query_jplsbdb.json')
        self.test_json = json.loads(test_json_fp.read_text())

    def test_translate_constraints(self):
        raw = ["q<1.3", "i<=10.5", "6<=H<=7"]
        query = JPLSBDBQuery(orbital_constraints=raw)
        expected = ["q|LT|1.3", "i|LE|10.5", "H|RG|6|7"]
        self.assertEqual(query.orbital_constraints, expected)

    def test_build_query_url(self):
        url = self.query.build_query_url()
        self.assertTrue(url.startswith(self.base_url))
        self.assertIn("fields=", url)
        self.assertIn("sb-class=IEO", url)
        self.assertIn("sb-cdata=", url)

        # Parse sb-cdata from URL
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        sb_cdata_encoded = query_params.get("sb-cdata", [None])[0]
        self.assertIsNotNone(sb_cdata_encoded, "sb-cdata not found in URL")

        sb_cdata_json = json.loads(unquote(sb_cdata_encoded))
        self.assertIn("AND", sb_cdata_json)
        self.assertIn("q|LT|1.3", sb_cdata_json["AND"])
        self.assertIn("i|LT|10.5", sb_cdata_json["AND"])

    @patch("requests.get")
    def test_run_query_success(self, mock_get):
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = self.test_json
        mock_get.return_value = mock_response

        results = self.query.run_query()
        self.assertIsInstance(results, dict)
        self.assertIn("fields", results)
        self.assertIn("data", results)

    @patch("requests.get")
    def test_run_query_failure(self, mock_get):
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        results = self.query.run_query()
        self.assertIsNone(results)

    def test_parse_results_valid(self):
        table = self.query.parse_results(self.test_json)
        self.assertIsInstance(table, QTable)
        self.assertEqual(table.colnames, self.test_json["fields"])
        self.assertEqual(len(table), len(self.test_json["data"]))
        self.assertEqual(table["full_name"][0], self.test_json["data"][0][0])

    def test_parse_results_empty(self):
        empty_table = self.query.parse_results({})
        self.assertIsInstance(empty_table, QTable)
        self.assertEqual(len(empty_table), 0)

        none_table = self.query.parse_results(None)
        self.assertIsInstance(none_table, QTable)
        self.assertEqual(len(none_table), 0)