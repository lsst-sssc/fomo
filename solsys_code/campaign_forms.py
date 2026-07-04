"""Public-facing campaign run submission form (SUBMIT-01/SUBMIT-04, D-05/D-06).

A plain `forms.Form` -- NEVER a `ModelForm`. `CampaignRun.telescope_instrument` has no
`blank=True` on the model, so a `ModelForm` would derive `required=True` from the model field and
wrongly force it required, contradicting D-05 ("everything except `campaign` is optional").
Explicit `required=False` on every non-`campaign` field sidesteps this entirely.
"""

from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Div, Fieldset, Layout, Submit
from django import forms
from tom_targets.models import TargetList


class CampaignRunSubmissionForm(forms.Form):
    """Public intake form for a single campaign observing run, pending staff review."""

    campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(), required=True)
    telescope_instrument = forms.CharField(max_length=255, required=False, label='Telescope / instrument')
    site_raw = forms.CharField(max_length=255, required=False, label='Observing site')
    obs_date = forms.DateField(required=False, label='Observation date')
    ut_start = forms.DateTimeField(required=False, label='UT start time')
    ut_end = forms.DateTimeField(required=False, label='UT end time')
    filters_bandpass = forms.CharField(max_length=255, required=False, label='Filter(s) / bandpass')
    observation_details = forms.CharField(widget=forms.Textarea, required=False, label='Observation details')
    open_to_collaboration = forms.BooleanField(required=False, label='Open to collaboration?')
    contact_person = forms.CharField(max_length=255, required=True, label='Contact person')  # D-06
    contact_email = forms.EmailField(required=True, label='Contact email')  # D-06
    comments = forms.CharField(widget=forms.Textarea, required=False, label='Other comments')
    # SUBMIT-04: hidden honeypot, non-obvious name, never rendered visibly to a human.
    alt_contact_info = forms.CharField(required=False, widget=forms.HiddenInput())

    def clean_alt_contact_info(self):
        """Never raise -- SUBMIT-04: a tripped bot must get no error signal. The view (Plan 02)
        decides what to do with a filled value; the form only passes it through.
        """
        return self.cleaned_data.get('alt_contact_info', '')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'campaign',
            Fieldset(
                'Run details',
                'telescope_instrument',
                'site_raw',
                'obs_date',
                'ut_start',
                'ut_end',
                'filters_bandpass',
                'observation_details',
                'open_to_collaboration',
            ),
            Fieldset('Contact', 'contact_person', 'contact_email', 'comments'),
            # Hidden via widget=HiddenInput above; belt-and-suspenders CSS hiding too.
            Div('alt_contact_info', css_class='d-none'),
            FormActions(Submit('submit', 'Submit run for review')),
        )
