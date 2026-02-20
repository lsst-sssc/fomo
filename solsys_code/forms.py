from datetime import datetime, timedelta, timezone

from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import HTML, Div, Fieldset, Layout, Submit
from django import forms
from django.urls import reverse

from solsys_code.solsys_code_observatory.models import Observatory


class EphemerisForm(forms.Form):
    """
    This form is for requesting an ephemeris of a Target object
    """

    time_text = 'date (and optionally time, all parts are optional) in the format YYYY-MM-DD[THH:MM:SS] or YYYY-MM-DD [HH:MM:SS]'  # noqa: E501

    target_id = forms.IntegerField(required=True, widget=forms.HiddenInput())
    start_date = forms.DateTimeField(required=True, help_text=f'Start {time_text}')
    end_date = forms.DateTimeField(required=True, help_text=f'End {time_text}')
    step = forms.CharField(
        required=False,
        initial='1d',
        help_text="Step size; any combination of number and parseable unit e.g. '1d', '0.5hour', '5m'",
    )
    site_code = forms.ModelChoiceField(
        Observatory.objects.filter(altitude__gt=0).order_by('name'), blank=False, required=True
    )
    full_precision = forms.BooleanField(
        required=False, initial=True, help_text='Whether to show the full results precision'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        dt = datetime.now(timezone.utc)
        start_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=20)
        self.fields['start_date'].initial = start_date
        self.fields['end_date'].initial = end_date
        target_id = self.initial.get('target_id')
        self.helper.form_action = reverse('makeephem', kwargs={'pk': target_id})
        cancel_url = reverse('home')
        if target_id:
            cancel_url = reverse('tom_targets:detail', kwargs={'pk': target_id})  # + '?tab=ephemeris'
        self.helper.layout = Layout(
            HTML(
                """
                <p>
                Fill in the form to generate an ephemeris for the object from the selected site.
                </p>
            """
            ),
            'target_id',
            Fieldset(
                'Ephemeris Parameters',
                Div(
                    Div(
                        'start_date',
                        'end_date',
                        css_class='col',
                    ),
                    Div(
                        'site_code',
                        'step',
                        'full_precision',
                        css_class='col',
                    ),
                    css_class='form-row',
                ),
            ),
            FormActions(
                Submit('confirm', 'Create Ephemeris'),
                HTML(f'<a class="btn btn-outline-primary" href={cancel_url}>Cancel</a>'),
            ),
        )
