from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from tom_targets.models import TargetList

from solsys_code.views import JPLSBDBQuery


class Command(BaseCommand):
    """
    Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones
    """

    help = 'Fetch new objects matching filter criteria from JPL SBDB and make Targets from the new ones'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments"""
        parser.add_argument(
            '--orbit_class',
            action='store',
            type=str,
            default=None,
            help='Orbit class constraints as a comma separated string',
        )
        parser.add_argument(
            '--orbital_constraints',
            action='store',
            type=str,
            default=None,
            help='Orbital constraints as a comma separated string',
        )
        parser.add_argument(
            '--group_name',
            action='store',
            type=str,
            default=None,
            help='Name of the target group to put the filtered objects in',
        )
        return super().add_arguments(parser)

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Make task run when doing `python manage.py fetch_jplsbdb_objects`

        :return: Status value
        :rtype: str | None
        """
        orbit_class = ','.join(options['orbit_class'].split(',')) if options['orbit_class'] is not None else None
        if options['orbital_constraints'] is not None:
            orbital_constraints = options['orbital_constraints'].split(',')
        else:
            orbital_constraints = None
        if orbit_class is None and orbital_constraints is None:
            orbital_constraints = ['e>=1.2']
        new_objects = JPLSBDBQuery(orbit_class=orbit_class, orbital_constraints=orbital_constraints)
        msg = f'Querying JPL SBDB for new objects with constraints= {new_objects.orbit_class}, {new_objects.orbital_constraints}.'  # noqa: E501
        self.stdout.write(msg)
        results = new_objects.run_query()
        if results is not None:
            _ = new_objects.parse_results(results)
            if new_objects.results_table is not None:
                new_targets = new_objects.create_targets()
                self.stdout.write(f'Created {len(new_targets)} new Targets')
                if options['group_name'] is not None:
                    group_name = options['group_name']
                    targetlist = TargetList.objects.get_or_create(name=group_name)
                    targetlist.targets.add(new_targets)
                    self.stdout.write(f'Added {len(new_targets)} new Targets to Target Group: {group_name}')
            else:
                self.stderr.write(self.style.ERROR('No results found for new_objects'))
        else:
            self.stderr.write('Error running query')

        return
