from django.test import SimpleTestCase

import fomo


class TestPackaging(SimpleTestCase):
    def test_version(self):
        """Check to see that we can get the package version"""
        self.assertIsNotNone(fomo.__version__)
