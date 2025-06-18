from django.apps import AppConfig


class SolsysCodeObservatoryConfig(AppConfig):  # noqa: D101
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'solsys_code.solsys_code_observatory'

    def nav_items(self):
        """
        Integration point for adding items to the navbar.
        This method should return a list of dictionaries that include a `partial` key pointing to the html templates to
        be included in the navbar. The `position` key, if included, should be either "left" or "right" to specify which
        side of the navbar the partial should be included on. If not included, a left side nav item is assumed.
        """
        # TODO: These filenames probably don't need 'demo' in them b/c they're namespaced in the app folder
        # XXX: Map the '.' in the module hierarchy to a '/'
        return [
            {'partial': f'{self.name}/partials/navbar_demo.html', 'position': 'right'},
            {'partial': f'{self.name}/partials/navbar_list_demo.html'},
        ]
