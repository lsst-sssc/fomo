from django import forms

from solsys_code.solsys_code_observatory.models import Observatory


class CreateObservatoryForm(forms.ModelForm):
    """
    Form for creating observatories by specifying the MPC code
    """

    obscode = forms.CharField(max_length=3, min_length=3)

    class Meta:  # noqa: D106
        model = Observatory
        fields = [
            'obscode',
        ]

    def clean_obscode(self):
        """Light cleaning of passed obscode to ensure it's uppercase as needed by the MPC"""
        cleaned_data = self.clean()
        obscode = cleaned_data.get('obscode', '')
        if obscode is not None:
            obscode = obscode.upper()
        return obscode
