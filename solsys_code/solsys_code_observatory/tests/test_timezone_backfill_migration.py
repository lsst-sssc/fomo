"""D-23: regression coverage for migration 0003's Observatory.timezone backfill.

Seeds rows against the pre-0003 schema (mirrors test_window_schema_migration.py's
MigrationExecutor shape), migrates forward, and asserts the derived-timezone outcome
directly -- rather than relying entirely on the unrepeatable manual dev-DB run.
"""

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase


class TestTimezoneBackfillMigration(TransactionTestCase):
    """Exercises migration 0003's backfill_observatory_timezone RunPython step end-to-end."""

    migrate_from = [('solsys_code_observatory', '0002_observatory_timezone_seed')]
    migrate_to = [('solsys_code_observatory', '0003_backfill_observatory_timezone')]

    def setUp(self):
        # Start from the pre-0003 schema.
        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_from)
        old_apps = executor.loader.project_state(self.migrate_from).apps
        Observatory = old_apps.get_model('solsys_code_observatory', 'Observatory')

        # D-23's named target: real Siding Spring coordinates, blank timezone.
        self.e10_pk = Observatory.objects.create(
            obscode='E10',
            name='Siding Spring-Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2728,
            lon=149.0709,
            timezone='',
        ).pk

        # Coordinate-less satellite-shaped row (mirrors HST, obscode 250) -- must stay blank.
        self.satellite_pk = Observatory.objects.create(
            obscode='250',
            name='Hubble Space Telescope',
            short_name='HST',
            lat=None,
            lon=None,
            timezone='',
        ).pk

        # A row with a pre-set timezone -- authoritative, must be left byte-identical even
        # though its (deliberately implausible) coordinates would derive a different name.
        self.preset_pk = Observatory.objects.create(
            obscode='TST',
            name='Pre-Set Timezone Test Site',
            short_name='TST',
            lat=0.0,
            lon=0.0,
            timezone='Etc/UTC',
        ).pk

        # Migrate forward through 0003.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.migrate_to)
        self.new_apps = executor.loader.project_state(self.migrate_to).apps

    def tearDown(self):
        # Leave the DB on the latest migration state for any tests that run after this one.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())

    def test_e10_backfilled_to_australia_sydney(self):
        """D-23: Siding Spring's real coordinates derive 'Australia/Sydney'."""
        Observatory = self.new_apps.get_model('solsys_code_observatory', 'Observatory')
        obs = Observatory.objects.get(pk=self.e10_pk)

        self.assertEqual(obs.timezone, 'Australia/Sydney')

    def test_coordinate_less_row_stays_blank(self):
        """A satellite-shaped row with null lat/lon is skipped and stays blank."""
        Observatory = self.new_apps.get_model('solsys_code_observatory', 'Observatory')
        obs = Observatory.objects.get(pk=self.satellite_pk)

        self.assertEqual(obs.timezone, '')

    def test_preset_timezone_is_never_overwritten(self):
        """A row with a timezone already set is left byte-identical, not re-derived."""
        Observatory = self.new_apps.get_model('solsys_code_observatory', 'Observatory')
        obs = Observatory.objects.get(pk=self.preset_pk)

        self.assertEqual(obs.timezone, 'Etc/UTC')
