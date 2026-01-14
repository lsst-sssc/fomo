from django.test import Client, TestCase


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
