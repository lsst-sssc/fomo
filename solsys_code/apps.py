from django.apps import AppConfig


class SolsysCodeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'solsys_code'

    def target_detail_buttons(self):
        """
        Integration point for adding buttons to the Target detail view
        """
        return [
            {
                'partial': f'{self.name}/partials/ephem_button.html',
                'context': 'src.templatetags.solsys_code_extras.ephem_button',
            },
            {
                'partial': f'{self.name}/partials/campaign_links.html',
                'context': 'src.templatetags.solsys_code_extras.campaign_links',
            },
        ]

    def nav_items(self):
        """
        Integration point for adding entries to the navbar (VIEW-02/D-03).
        """
        return [
            {
                'partial': f'{self.name}/partials/campaigns_nav_link.html',
                'context': 'src.templatetags.solsys_code_extras.campaigns_nav_link',
                'position': 'left',
            }
        ]

    def data_services(self):
        """
        integration point for including data services in the TOM
        This method should return a list of dictionaries containing dot separated DataService classes
        """
        return [{'class': 'tom_fink.fink.FinkDataService'}]
