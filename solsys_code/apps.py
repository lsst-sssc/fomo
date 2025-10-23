from django.apps import AppConfig


class SolsysCodeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'solsys_code'

    def target_detail_buttons(self):
        """
        Integration point for adding buttons to the Target detail view
        """
        return {'namespace': 'makeephem', 'title': 'FOMO Target Button', 'class': 'btn btn-info', 'text': 'Ephemeris'}
