"""Public-facing campaign run submission form (SUBMIT-01/SUBMIT-04, D-05/D-06).

A plain `forms.Form` -- NEVER a `ModelForm`. `CampaignRun.telescope_instrument` has no
`blank=True` on the model, so a `ModelForm` would derive `required=True` from the model field and
wrongly force it required, contradicting D-05 ("everything except `campaign` is optional").
Explicit `required=False` on every non-`campaign` field sidesteps this entirely.
"""

from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Div, Fieldset, Layout, Submit
from django import forms
from django.urls import reverse_lazy
from tom_targets.models import Target, TargetList

from solsys_code.campaign_utils import parse_obs_window
from solsys_code.solsys_code_observatory.models import Observatory


class CampaignRunSubmissionForm(forms.Form):
    """Public intake form for a single campaign observing run, pending staff review."""

    campaign = forms.ModelChoiceField(queryset=TargetList.objects.all(), required=True)
    telescope_instrument = forms.CharField(max_length=255, required=False, label='Telescope / instrument')
    # D-09: live-search widget, no create-new-site link (public submitters never get a
    # site-creation path; unmatched free text is allowed and never blocks submission).
    # NOTE (htmx hx-trigger grammar): the `[...]` event filter goes IMMEDIATELY AFTER the
    # event name, with modifiers (`changed`, `delay:300ms`) following -- 22-REVIEWS.md
    # finding 1. Do NOT reorder this to `input changed delay:300ms[...]`; htmx does not
    # parse a filter placed after the modifiers.
    site_raw = forms.CharField(
        max_length=255,
        required=False,
        label='Observing site',
        widget=forms.TextInput(
            attrs={
                'hx-get': reverse_lazy('campaigns:site_search'),
                'hx-trigger': 'input[this.value.length >= 2] changed delay:300ms',
                'hx-target': '#site-suggestions-id_site_raw',
                'hx-swap': 'innerHTML',
                'hx-vals': '{"input_id": "id_site_raw"}',
                'autocomplete': 'off',
                'placeholder': 'MPC code or site name…',
                'class': 'form-control',
            }
        ),
    )
    # A3: collapses to a single observing-date free-text field -- the window schema has no
    # time-of-night component, so the UT start/end DateTimeField inputs have no home here
    # and are dropped entirely (not repurposed). This is now free text parsed by
    # `parse_obs_window()` (SUBMIT-01 date-format gap fix) rather than a strict single
    # `DateField` -- clean() below maps the parsed window onto cleaned_data['window_start']/
    # ['window_end'], which the view reads (single-night collapse or a genuine multi-night
    # range; blank -> TBD, both None).
    obs_date = forms.CharField(
        required=False,
        max_length=255,
        label='Observation date',
        help_text=(
            'A single date (YYYY-MM-DD), a date range (YYYY-MM-DD -- YYYY-MM-DD or '
            'YYYY-MM-DD to YYYY-MM-DD), or leave blank if not yet scheduled (TBD).'
        ),
    )
    filters_bandpass = forms.CharField(max_length=255, required=False, label='Filter(s) / bandpass')
    observation_details = forms.CharField(widget=forms.Textarea, required=False, label='Observation details')
    open_to_collaboration = forms.BooleanField(required=False, label='Open to collaboration?')
    contact_person = forms.CharField(max_length=255, required=True, label='Contact person')  # D-06
    contact_email = forms.EmailField(required=True, label='Contact email')  # D-06
    # VIEW-05/D-07: default-opt-out combined contact-visibility flag. required=False means an
    # unchecked box (the default) cleans to False.
    contact_public_opt_in = forms.BooleanField(
        required=False,
        label='Show contact info publicly?',
        help_text=(
            'If checked, your name and email will be shown on the public campaign table. '
            'Leave unchecked to keep them visible to staff only (default).'
        ),
    )
    comments = forms.CharField(widget=forms.Textarea, required=False, label='Other comments')
    # SUBMIT-04: hidden honeypot, non-obvious name, never rendered visibly to a human.
    alt_contact_info = forms.CharField(required=False, widget=forms.HiddenInput())

    def clean_alt_contact_info(self):
        """Never raise -- SUBMIT-04: a tripped bot must get no error signal. The view (Plan 02)
        decides what to do with a filled value; the form only passes it through.
        """
        return self.cleaned_data.get('alt_contact_info', '')

    def clean(self):
        """Parse the free-text `obs_date` into a window via `parse_obs_window()`.

        Mirrors `import_campaign_csv`'s needs-review discipline (act on the parser's flag,
        never raise) but adapted to Django form-error convention: non-blank unparseable text
        surfaces a friendly `obs_date` error (`form.add_error`, not a silent skip); blank text
        also comes back with `window_needs_review=True` but is an intentional TBD and must NOT
        error. `parse_obs_window()` always returns `window_start`/`window_end` both `None` or
        both set (single-night collapse or a genuine range), never one-`None` -- this keeps the
        model's `campaign_run_window_start_end_null_together` CheckConstraint invariant intact
        with no extra code needed here. There is no UT-time field on this public form (per the
        `obs_date` field's A3 comment), so an empty `ut_range_raw` is passed through.
        """
        cleaned_data = super().clean()
        obs_date_raw = cleaned_data.get('obs_date', '') or ''
        (
            window_start,
            window_end,
            _original_raw,
            window_needs_review,
            _ut_start,
            _ut_end,
            _ut_needs_review,
        ) = parse_obs_window(obs_date_raw, '')
        cleaned_data['window_start'] = window_start
        cleaned_data['window_end'] = window_end
        if window_needs_review and obs_date_raw.strip():
            self.add_error(
                'obs_date',
                "Couldn't understand this date. Use a single date (YYYY-MM-DD), a range "
                '(YYYY-MM-DD -- YYYY-MM-DD or YYYY-MM-DD to YYYY-MM-DD), or leave it blank '
                'if the observing date is not yet scheduled.',
            )
        return cleaned_data

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'campaign',
            Fieldset(
                'Run details',
                'telescope_instrument',
                'site_raw',
                HTML('<div id="site-suggestions-id_site_raw" class="mt-2"></div>'),
                'obs_date',
                'filters_bandpass',
                'observation_details',
                'open_to_collaboration',
            ),
            Fieldset('Contact', 'contact_person', 'contact_email', 'contact_public_opt_in', 'comments'),
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
