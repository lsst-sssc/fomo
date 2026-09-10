from unittest import mock

from django.db.utils import IntegrityError
from django.test import Client, TestCase

from solsys_code.solsys_code_observatory.models import Observatory


class CreateObservatoryTest(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_form_view(self):
        response = self.client.post('/observatory/create/', {'obscode': 'Z31'})
        self.assertEqual(response.status_code, 302)

    def test_form_view_bad_code(self):
        response = self.client.post('/observatory/create/', {'obscode': 'FOO'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response, 'Malformed input: &quot;obscode&quot;=&quot;FOO&quot; does not match regular expression'
        )

    def test_form_view_duplicate_code(self):
        """A code already in the DB is rejected by the form's unique validation (no API call)."""
        Observatory.objects.create(obscode='Z31', name='Tenerife Observatory-LCO A, Tenerife')

        response = self.client.post('/observatory/create/', {'obscode': 'Z31'})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Observatory with this MPC observatory code already exists.')
        self.assertEqual(Observatory.objects.filter(obscode='Z31').count(), 1)

    def test_form_view_duplicate_code_race(self):
        """A duplicate created between form validation and save() is reported as a duplicate."""

        def create_then_fail():
            Observatory.objects.create(obscode='W85', name='Cerro Tololo-LCO A')
            raise IntegrityError('UNIQUE constraint failed: solsys_code_observatory_observatory.obscode')

        with mock.patch('solsys_code.solsys_code_observatory.views.MPCObscodeFetcher') as mock_fetcher:
            mock_fetcher.return_value.query.return_value = None
            mock_fetcher.return_value.to_observatory.side_effect = create_then_fail
            response = self.client.post('/observatory/create/', {'obscode': 'W85'})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Observatory with MPC code W85 already exists')

    def test_form_view_integrity_error_not_duplicate(self):
        with mock.patch('solsys_code.solsys_code_observatory.views.MPCObscodeFetcher') as mock_fetcher:
            mock_fetcher.return_value.query.return_value = None
            mock_fetcher.return_value.to_observatory.side_effect = IntegrityError(
                'NOT NULL constraint failed: solsys_code_observatory_observatory.timezone'
            )
            response = self.client.post('/observatory/create/', {'obscode': 'W85'})

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'already exists')
        self.assertContains(
            response,
            'Could not create Observatory: NOT NULL constraint failed: solsys_code_observatory_observatory.timezone',
        )
