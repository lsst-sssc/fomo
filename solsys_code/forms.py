from django import forms


class EphemerisForm(forms.Form):
    """
    This form is for requesting an ephemeris of a Target object
    """

    target_id = forms.IntegerField(required=True, widget=forms.HiddenInput())
    start_date = forms.DateTimeField(required=True)
    end_date = forms.DateTimeField(required=True)
    site_code = forms.CharField(required=True, max_length=3)
