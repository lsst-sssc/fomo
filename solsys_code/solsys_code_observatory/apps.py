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

        template_path = self.name.split('.')[-1]
        return [
            {'partial': f'{template_path}/partials/navbar.html', 'position': 'right'},
            {'partial': f'{template_path}/partials/navbar_list.html'},
        ]
