"""Tests for solsys_code/admin.py -- proves the load-bearing admin constraints via the
admin test client rather than by eyeballing the ModelAdmin class definitions:

- CampaignRun and CalendarEventTelescopeLabel are both reachable under /admin/solsys_code/.
- approval_status is visible-but-non-editable in the CampaignRun change form (T-jpd-01: no
  admin path to APPROVED that bypasses CampaignRunDecisionView.post()'s calendar projection
  + D-06 clobber guard).
- contact_person/contact_email never appear in the CampaignRun change-list (T-jpd-02: PII is
  not scannable across rows) but remain editable in the detail/change view.
- CalendarEventTelescopeLabel's event__title search path resolves without a FieldError.
"""

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from tom_targets.models import TargetList

from solsys_code.models import CampaignRun

PII_CONTACT_PERSON = 'Zztestcontact'
PII_CONTACT_EMAIL = 'pii-secret@example.test'


class AdminRegistrationAndGatingTests(TestCase):
    """T-jpd-01/T-jpd-02: approval_status read-only, PII gated from the change-list."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(username='adminuser', email='admin@example.test', password='pw')
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        # NOTE: not named `cls.run` -- unittest.TestCase.run() is the method the test
        # framework itself invokes to execute each test; shadowing it with an attribute
        # breaks test execution with `TypeError: 'CampaignRun' object is not callable`.
        cls.campaign_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='LCO-1m-Sinistro',
            contact_person=PII_CONTACT_PERSON,
            contact_email=PII_CONTACT_EMAIL,
        )

    def setUp(self) -> None:
        self.client.force_login(self.superuser)

    def test_campaignrun_changelist_loads(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_changelist'))
        self.assertEqual(response.status_code, 200)

    def test_calendareventtelescopelabel_changelist_loads(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_calendareventtelescopelabel_changelist'))
        self.assertEqual(response.status_code, 200)

    def test_calendareventtelescopelabel_search_resolves(self) -> None:
        response = self.client.get(
            reverse('admin:solsys_code_calendareventtelescopelabel_changelist'), {'q': 'anything'}
        )
        self.assertEqual(response.status_code, 200)

    def test_approval_status_is_readonly_in_change_form(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.campaign_run.pk]))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('Pending Review', content)
        self.assertNotIn('name="approval_status"', content)

    def test_contact_fields_editable_in_change_form(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.campaign_run.pk]))
        content = response.content.decode()
        self.assertIn('name="contact_person"', content)
        self.assertIn('name="contact_email"', content)

    def test_pii_not_rendered_in_changelist(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_changelist'))
        content = response.content.decode()
        self.assertNotIn(PII_CONTACT_PERSON, content)
        self.assertNotIn(PII_CONTACT_EMAIL, content)
        self.assertIn('LCO-1m-Sinistro', content)
