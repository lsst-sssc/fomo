"""Tests for `CampaignRunSubmissionForm` (SUBMIT-01/SUBMIT-04, D-05/D-06).

Uses `TargetList.objects.create(...)` for the campaign fixture (never
`SiderealTargetFactory`/`Target` -- CLAUDE.md mandates non-sidereal-only fixtures for this
project; campaigns are `TargetList` objects, not `Target`, so no Target factory is needed here
at all).
"""

from datetime import date

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

    def test_contact_public_opt_in_present_and_not_required(self):
        """VIEW-05/D-07: the opt-in checkbox is not required, so an unchecked box validates."""
        form = CampaignRunSubmissionForm()
        self.assertIn('contact_public_opt_in', form.fields)
        self.assertFalse(form.fields['contact_public_opt_in'].required)

    def test_contact_public_opt_in_unchecked_defaults_false(self):
        """VIEW-05: omitting the checkbox from POST data (unchecked box) cleans to False."""
        form = CampaignRunSubmissionForm(data=self._minimal_data())
        self.assertTrue(form.is_valid(), form.errors)
        self.assertFalse(form.cleaned_data['contact_public_opt_in'])

    def test_contact_public_opt_in_checked_cleans_true(self):
        """VIEW-05: submitting the checkbox as checked cleans to True."""
        form = CampaignRunSubmissionForm(data=self._minimal_data(contact_public_opt_in='on'))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertTrue(form.cleaned_data['contact_public_opt_in'])

    def test_contact_public_opt_in_label(self):
        form = CampaignRunSubmissionForm()
        self.assertEqual(form.fields['contact_public_opt_in'].label, 'Show contact info publicly?')

    def test_is_plain_form_not_model_form(self):
        """CampaignRunSubmissionForm must be a plain forms.Form, never a ModelForm (Pitfall 3)."""
        self.assertTrue(issubclass(CampaignRunSubmissionForm, forms.Form))
        self.assertFalse(issubclass(CampaignRunSubmissionForm, forms.ModelForm))


class CampaignRunSubmissionFormObsDateWindowTest(TestCase):
    """260714-ilz: obs_date accepts flexible date/range text, parsed via parse_obs_window()
    into cleaned_data['window_start']/['window_end'] (requirements 1/2/4/5/6).
    """

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

    def test_single_date_collapses_to_start_equals_end(self):
        form = CampaignRunSubmissionForm(data=self._minimal_data(obs_date='2027-04-20'))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['window_start'], date(2027, 4, 20))
        self.assertEqual(form.cleaned_data['window_end'], date(2027, 4, 20))

    def test_identical_double_hyphen_range_collapses_to_single_night(self):
        """Requirement 4: an explicit start==end range still collapses to a single night."""
        form = CampaignRunSubmissionForm(data=self._minimal_data(obs_date='2027-04-20 -- 2027-04-20'))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['window_start'], date(2027, 4, 20))
        self.assertEqual(form.cleaned_data['window_end'], date(2027, 4, 20))

    def test_identical_to_separated_range_collapses_to_single_night(self):
        """The 'to'-separated equal-endpoint range exercises the second separator path."""
        form = CampaignRunSubmissionForm(data=self._minimal_data(obs_date='2027-04-20 to 2027-04-20'))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['window_start'], date(2027, 4, 20))
        self.assertEqual(form.cleaned_data['window_end'], date(2027, 4, 20))

    def test_genuine_multi_night_range_is_valid(self):
        """Requirement 1: a real multi-night range no longer hard-fails Django date validation."""
        form = CampaignRunSubmissionForm(data=self._minimal_data(obs_date='2027-04-20 -- 2027-05-11'))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['window_start'], date(2027, 4, 20))
        self.assertEqual(form.cleaned_data['window_end'], date(2027, 5, 11))

    def test_blank_obs_date_is_valid_and_yields_tbd_window(self):
        """Requirement 5: blank obs_date still produces a TBD run, both window fields None."""
        form = CampaignRunSubmissionForm(data=self._minimal_data())
        self.assertTrue(form.is_valid(), form.errors)
        self.assertIsNone(form.cleaned_data['window_start'])
        self.assertIsNone(form.cleaned_data['window_end'])

    def test_unparseable_obs_date_text_is_invalid_with_friendly_error(self):
        """Requirement 2: genuinely unparseable non-blank text errors, never a silent TBD."""
        form = CampaignRunSubmissionForm(data=self._minimal_data(obs_date='sometime next spring'))
        self.assertFalse(form.is_valid())
        self.assertIn('obs_date', form.errors)
        self.assertTrue(form.errors['obs_date'][0])

    def test_reversed_range_is_invalid_with_friendly_error(self):
        """A reversed range (end < start) falls through to the unparseable-non-blank branch."""
        form = CampaignRunSubmissionForm(data=self._minimal_data(obs_date='2027-05-11 -- 2027-04-20'))
        self.assertFalse(form.is_valid())
        self.assertIn('obs_date', form.errors)
