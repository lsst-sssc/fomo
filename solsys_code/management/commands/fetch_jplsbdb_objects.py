from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.views import JPLSBDBQuery

class Command(BaseCommand):
    """
    Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones
    """
    help = 'Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments
        """
        parser.add_argument('--orbit_class', action='store', type=str, default=None,
                            help='Orbit class constraints as a comma separated string')
        parser.add_argument('--orbital_constraints', action='store', type=str, default=None,
                            help='Orbital constraints as a comma separated string')
        return super().add_arguments(parser)

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Make task run when doing `python manage.py fetch_jplsbdb_objects`

        :return: Status value
        :rtype: str | None
        """
        if options['orbit_class'] is not None:
            orbit_class = options['orbit_class'].split(',')
        else:
            orbit_class = None
        if options['orbital_constraints'] is not None:
            orbital_constraints = options['orbital_constraints'].split(',')
        else:
            orbital_constraints = None
        if orbit_class == None and orbital_constraints == None:
            orbital_constraints = 'e>=1.2'
        new_objects = JPLSBDBQuery(orbit_class=orbit_class, orbital_constraints=orbital_constraints)
        self.stdout.write(f"Querying JPL SBDB for new_objects with constraints= {new_objects.orbit_class}, {new_objects.orbital_constraints}.")
        results = new_objects.run_query()
        _ = new_objects.parse_results(results)
        if new_objects.results_table is not None:
            new_objects.create_targets()
        else:
            self.stderr.write(self.style.ERROR(f"No results found for new_objects"))

        return super().handle(*args, **options)