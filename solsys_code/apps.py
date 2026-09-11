from django.apps import AppConfig


class SolsysCodeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'solsys_code'

    def ready(self):
        """Wire the observation projector's signal receivers (TRIG-01).

        A Django ``post_save`` signal, not ``settings.HOOKS['observation_change_state']``,
        because that hook is silent for a schedule-only placement save -- exactly the
        narrowing step that matters for this project's calendar (``ObservationRecord.save()``
        only runs the hook when ``status`` changes or on creation). Imports are function-local
        to match the lazy-import convention the existing ``pre_delete`` receiver in
        ``models.py`` already uses to dodge app-loading circular imports.
        """
        from django.db.models.signals import post_save
        from tom_observations.models import ObservationRecord

        from solsys_code.observation_projector import receiver_on_record_save

        post_save.connect(
            receiver_on_record_save,
            sender=ObservationRecord,
            weak=False,
            dispatch_uid='solsys_code.observation_projector.post_save',
        )

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
