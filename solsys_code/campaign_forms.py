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
from tom_targets.models import Target, TargetList

from solsys_code.solsys_code_observatory.models import Observatory


class CampaignRunSubmissionForm(forms.Form):
    """Public intake form for a single campaign observing run, pending staff review."""

    campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(), required=True)
    telescope_instrument = forms.CharField(max_length=255, required=False, label='Telescope / instrument')
    site_raw = forms.CharField(max_length=255, required=False, label='Observing site')
    # A3: collapses to a single observing-date field -- the window schema has no time-of-
    # night component, so the UT start/end DateTimeField inputs have no home here and are
    # dropped entirely (not repurposed). The view maps this single date to both
    # window_start and window_end on save (single-night collapse, SCHED-02).
    obs_date = forms.DateField(required=False, label='Observation date')
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
                'filters_bandpass',
                'observation_details',
                'open_to_collaboration',
            ),
            Fieldset('Contact', 'contact_person', 'contact_email', 'comments'),
            # Hidden via widget=HiddenInput above; belt-and-suspenders CSS hiding too.
            Div('alt_contact_info', css_class='d-none'),
            FormActions(Submit('submit', 'Submit run for review')),
        )


class CampaignGapAnalysisForm(forms.Form):
    """Campaign-scoped target/site/date-range selection form for coverage-gap analysis (GAP-02).

    A plain `forms.Form` -- NOT a `ModelForm` -- matching `CampaignRunSubmissionForm`'s style.
    The `target`/`site` querysets MUST be scoped to a specific campaign at instantiation time
    (via `campaign=` in `__init__`), never via a class-level unscoped queryset (D-12/D-13's
    dropdown-population rules). The view (`CampaignGapAnalysisView`) re-validates any submitted
    `target`/`site` pk server-side regardless of what these querysets offer -- this form only
    controls what's *offered*, not what a raw request can submit (Pitfall 3, IDOR).
    """

    target = forms.ModelChoiceField(queryset=Target.objects.none(), required=False, label='Target')
    site = forms.ModelChoiceField(queryset=Observatory.objects.none(), required=True, label='Site')
    end_date = forms.DateField(required=False, label='End date (optional)')

    def __init__(self, *args, campaign=None, **kwargs):
        super().__init__(*args, **kwargs)
        if campaign is not None:
            # D-12: target is optional (auto-selected server-side) for a single-target
            # campaign; required when there's more than one target to disambiguate.
            self.fields['target'].queryset = campaign.targets.all()
            self.fields['target'].required = campaign.targets.count() > 1
            # D-13: only Observatory records actually used by this campaign's CampaignRuns.
            self.fields['site'].queryset = Observatory.objects.filter(campaign_runs__campaign=campaign).distinct()
        self.helper = FormHelper()
        self.helper.form_method = 'get'  # D-09: gap analysis is a plain GET, not an htmx POST
        self.helper.layout = Layout(
            'target',
            'site',
            'end_date',
            FormActions(Submit('submit', 'Update Results')),
        )
