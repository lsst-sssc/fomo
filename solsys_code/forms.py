from crispy_forms.helper import FormHelper
from crispy_forms.layout import ButtonHolder, Column, Layout, Row, Submit
from django import forms
from django.urls import reverse


class EphemerisForm(forms.Form):
    """
    This form is for requesting an ephemeris of a Target object
    """

    target_id = forms.IntegerField(required=True, widget=forms.HiddenInput())
    start_date = forms.DateTimeField(required=True)
    end_date = forms.DateTimeField(required=True)
    step = forms.CharField(required=False, initial='1d')
    site_code = forms.CharField(required=True, max_length=3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_action = reverse('makeephem', kwargs={'pk': self.initial.get('target_id')})
        self.helper.layout = Layout(
            'target_id',
            Row(
                Column('start_date'),
                Column('end_date'),
                Column('site_code'),
                Column('step'),
                Column(ButtonHolder(Submit('submit', 'Create Ephemeris'))),
            ),
        )
