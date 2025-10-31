from django.test import Client, TestCase
from tom_catalogs.harvester import MissingDataException


class CreateObservatoryTest(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_form_view(self):
        response = self.client.post('/observatory/create/', {'obscode': 'Z31'})
        self.assertEqual(response.status_code, 302)

    def test_form_view_bad_code(self):
        with self.assertRaises(MissingDataException):
            response = self.client.post('/observatory/create/', {'obscode': 'FOO'})
            self.assertEqual(response.status_code, 422)
