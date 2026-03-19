import os
import tempfile

import pandas as pd
from django.test import SimpleTestCase

from fomo.compare_utils import compare_ades_with_csv


class CompareUtilsTests(SimpleTestCase):
    def test_compare_match(self):
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': ['x', 'y']})
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, 'ades.csv')
            df.to_csv(csv, index=False)

            res = compare_ades_with_csv(df, csv)
            self.assertTrue(res['match'])
            self.assertEqual(res['hash_ades'], res['hash_disk'])

    def test_compare_mismatch(self):
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': ['x', 'y']})
        df2 = df.copy()
        df2.loc[1, 'a'] = 2.1
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, 'ades.csv')
            df2.to_csv(csv, index=False)

            res = compare_ades_with_csv(df, csv)
            self.assertFalse(res['match'])
            self.assertTrue(('diff' in res) or ('diff_rows' in res))

    def test_float_rounding(self):
        # Values that differ only in sub-6th-decimal places but are not
        # identical; using strict tolerances (rtol=0, atol=0) ensures the
        # unrounded comparison fails while rounding to 6 decimals makes
        # them equal.
        df = pd.DataFrame({'dec': [1.234567123], 'b': ['x']})
        df2 = pd.DataFrame({'dec': [1.234567456], 'b': ['x']})
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, 'ades.csv')
            df2.to_csv(csv, index=False)

            # Strict comparison (no tolerance) should detect mismatch
            res_no_round = compare_ades_with_csv(df, csv, rtol=0, atol=0)
            self.assertFalse(res_no_round['match'])

            # With rounding to 6 decimals, frames should match under strict check
            res_round = compare_ades_with_csv(df, csv, float_decimals=6, rtol=0, atol=0)
            self.assertTrue(res_round['match'])
