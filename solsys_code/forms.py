from django import forms


class EphemerisForm(forms.Form):
    """
    This form is for requesting an ephemeris of a Target object
    """

    target_id = forms.IntegerField(required=True, widget=forms.HiddenInput())
    start_date = forms.DateTimeField(required=True)
    end_date = forms.DateTimeField(required=True)
    site_code = forms.CharField(required=True, max_length=3)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.helper = FormHelper()
    #     self.helper.form_action = reverse('makeephem', kwargs={'pk': self.initial.get('target_id')})
    #     self.helper.layout = Layout(
    #         'target_id',
    #         Row(
    #             Column(
    #                 'start_date'
    #             ),
    #             Column(
    #                 'end_date'
    #             ),
    #             Column(
    #                 'site_code'
    #             ),
    #             Column(
    #                 ButtonHolder(
    #                     Submit('submit', 'Generate Ephemeris')
    #                 )
    #             )
    #         )
    #     )
