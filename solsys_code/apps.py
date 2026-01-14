from django.apps import AppConfig


class SolsysCodeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'solsys_code'

    def target_detail_buttons(self):
        """
        Integration point for adding buttons to the Target detail view
        """
        return [{'partial': f'{self.name}/partials/ephem_button.html',
                 'context': f'src.templatetags.solsys_code_extras.ephem_button'}]

