"""Tests for `CampaignRunSubmissionForm` (SUBMIT-01/SUBMIT-04, D-05/D-06).

Uses `TargetList.objects.create(...)` for the campaign fixture (never
`SiderealTargetFactory`/`Target` -- CLAUDE.md mandates non-sidereal-only fixtures for this
project; campaigns are `TargetList` objects, not `Target`, so no Target factory is needed here
at all).
"""

from django import forms
from django.test import TestCase
from tom_targets.models import TargetList

from solsys_code.campaign_forms import CampaignRunSubmissionForm

CONTACT_PERSON = 'Jane Coordinator'
CONTACT_EMAIL = 'jane@example.org'


class CampaignRunSubmissionFormTest(TestCase):
    """Behaviors from 16-01-PLAN.md Task 2 <behavior>."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')

    def _minimal_data(self, **overrides):
        data = {
            'campaign': self.campaign.pk,
            'contact_person': CONTACT_PERSON,
            'contact_email': CONTACT_EMAIL,
        }
        data.update(overrides)
        return data

    def test_minimal_valid_submission(self):
        """A form bound to only campaign/contact_person/contact_email (valid campaign pk) is valid."""
        form = CampaignRunSubmissionForm(data=self._minimal_data())
        self.assertTrue(form.is_valid(), form.errors)

    def test_missing_campaign_invalid(self):
        """campaign is the only required model-backed field."""
        data = self._minimal_data()
        del data['campaign']
        form = CampaignRunSubmissionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('campaign', form.errors)

    def test_missing_contact_person_invalid(self):
        """contact_person is required at the form level (D-06), even though blank=True on the model."""
        data = self._minimal_data()
        del data['contact_person']
        form = CampaignRunSubmissionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('contact_person', form.errors)

    def test_missing_contact_email_invalid(self):
        """contact_email is required at the form level (D-06), even though blank=True on the model."""
        data = self._minimal_data()
        del data['contact_email']
        form = CampaignRunSubmissionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('contact_email', form.errors)

    def test_honeypot_filled_still_valid(self):
        """alt_contact_info is required=False; a filled honeypot does not fail validation (SUBMIT-04)."""
        form = CampaignRunSubmissionForm(data=self._minimal_data(alt_contact_info='I am a bot'))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['alt_contact_info'], 'I am a bot')

    def test_honeypot_widget_is_hidden_input(self):
        """alt_contact_info renders as a HiddenInput, never a normally-typed field (Pitfall 5)."""
        form = CampaignRunSubmissionForm()
        self.assertIsInstance(form.fields['alt_contact_info'].widget, forms.HiddenInput)
        self.assertFalse(form.fields['alt_contact_info'].required)

    def test_telescope_instrument_not_required(self):
        """Every field except campaign/contact_person/contact_email is optional (D-05)."""
        form = CampaignRunSubmissionForm()
        self.assertFalse(form.fields['telescope_instrument'].required)

    def test_contact_fields_required(self):
        form = CampaignRunSubmissionForm()
        self.assertTrue(form.fields['contact_person'].required)
        self.assertTrue(form.fields['contact_email'].required)

    def test_site_raw_label_is_observing_site(self):
        """site_raw's form label is 'Observing site', not the model verbose_name."""
        form = CampaignRunSubmissionForm()
        self.assertEqual(form.fields['site_raw'].label, 'Observing site')

    def test_is_plain_form_not_model_form(self):
        """CampaignRunSubmissionForm must be a plain forms.Form, never a ModelForm (Pitfall 3)."""
        self.assertTrue(issubclass(CampaignRunSubmissionForm, forms.Form))
        self.assertFalse(issubclass(CampaignRunSubmissionForm, forms.ModelForm))
