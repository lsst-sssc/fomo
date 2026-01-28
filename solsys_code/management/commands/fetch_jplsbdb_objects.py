from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.views import JPLSBDBQuery


class Command(BaseCommand):
    """
    Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones
    """
    help = 'Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones'

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Make task run when doing `python manage.py fetch_jplsbdb_objects`

        :return: Status value
        :rtype: str | None
        """
        new_objects = JPLSBDBQuery(orbit_class=orbit_class, orbital_constraints=orbital_constraints)
        self.stdout.write(f"Querying JPL SBDB for new_objects with constraints= {new_objects.orbit_class, new_objects.orbital_constraints}.")
        results = new_objects.run_query()
        _ = new_objects.parse_results(results)
        if new_objects.results_table is not None:
            new_objects.create_targets()
        else:
            self.stderr.write(self.style.ERROR(f"No results found for new_objects"))

        return super().handle(*args, **options)