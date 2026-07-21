"""WR-03: regression coverage for migration 0004's RunPython data-transformation logic.

``backfill_window_fields``, ``dedupe_tbd_collisions``, and ``dedupe_resolved_window_collisions``
(added by CR-01) are non-reversible, data-deleting functions with real production consequences.
Everything in ``test_campaign_models.py`` only asserts the *post-migration* model shape via
``CampaignRun.objects.create()`` against the already-migrated test schema; it never exercises the
migration's own ``RunPython`` steps against a simulated pre-migration state. This module seeds rows
against the historical (pre-0004) schema -- including the resolved-window collision scenario CR-01
describes (two rows sharing campaign/telescope_instrument/obs_date but with distinct, non-null
``ut_start`` values) -- migrates forward to 0004, and asserts the backfill/dedup outcome directly,
rather than relying entirely on an unrepeatable manual dry run against a copy of the dev DB.
"""

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase


class TestWindowSchemaMigrationDataTransform(TransactionTestCase):
    """Exercises migration 0004's backfill + both dedup RunPython steps end-to-end."""

    migrate_from = [('solsys_code', '0003_campaignrun_natural_key_unique_constraint')]
    migrate_to = [('solsys_code', '0004_campaignrun_window_schema')]

    def setUp(self):
        # Start from the pre-0004 schema (obs_date/ut_start/ut_end still present).
        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_from)
        old_apps = executor.loader.project_state(self.migrate_from).apps

        TargetList = old_apps.get_model('tom_targets', 'TargetList')
        CampaignRun = old_apps.get_model('solsys_code', 'CampaignRun')
        campaign = TargetList.objects.create(name='3I/ATLAS')

        # CR-01 scenario: two rows sharing campaign+telescope_instrument+obs_date but with
        # distinct, non-null ut_start values -- legal under the old (campaign,
        # telescope_instrument, ut_start) unique constraint, but collide once
        # backfill_window_fields collapses both onto the same (window_start, window_end).
        self.resolved_collision_a = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='FTN/MuSCAT3',
            obs_date='2025-07-04',
            ut_start='2025-07-04T02:00:00Z',
            contact_person='Alice Observer',
        ).pk
        self.resolved_collision_b = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='FTN/MuSCAT3',
            obs_date='2025-07-04',
            ut_start='2025-07-04T09:00:00Z',
            contact_person='Bob Observer',
        ).pk
        # A non-colliding resolved row on a different night should survive untouched.
        self.resolved_unique = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='FTN/MuSCAT3',
            obs_date='2025-07-05',
            ut_start='2025-07-05T02:00:00Z',
            contact_person='Carol Observer',
        ).pk
        # TBD-branch duplicate pair, to confirm the pre-existing dedupe_tbd_collisions step
        # still works correctly alongside the new resolved-window step.
        CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='JWST',
            obs_date=None,
            contact_person='Dana Requester',
        )
        CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='JWST',
            obs_date=None,
            contact_person='Dana Requester',
        )

        # Migrate forward through 0004.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.migrate_to)
        self.new_apps = executor.loader.project_state(self.migrate_to).apps

    def tearDown(self):
        # Leave the DB on the latest migration state for any tests that run after this one.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())

    def test_resolved_window_collision_deduped_to_single_row(self):
        """CR-01: same-night, distinct-ut_start rows collapse to one survivor, not an IntegrityError."""
        CampaignRun = self.new_apps.get_model('solsys_code', 'CampaignRun')
        runs = CampaignRun.objects.filter(telescope_instrument='FTN/MuSCAT3', window_start='2025-07-04')

        self.assertEqual(runs.count(), 1)
        survivor = runs.get()
        # dedupe keeps the lowest pk (the ordering both dedup functions use).
        self.assertEqual(survivor.pk, self.resolved_collision_a)
        self.assertEqual(str(survivor.window_start), '2025-07-04')
        self.assertEqual(str(survivor.window_end), '2025-07-04')

    def test_non_colliding_resolved_row_survives_untouched(self):
        """A resolved-window row with no collision is left alone by the new dedup step."""
        CampaignRun = self.new_apps.get_model('solsys_code', 'CampaignRun')
        run = CampaignRun.objects.get(pk=self.resolved_unique)

        self.assertEqual(str(run.window_start), '2025-07-05')
        self.assertEqual(str(run.window_end), '2025-07-05')

    def test_tbd_branch_dedup_still_runs_alongside_resolved_dedup(self):
        """Pre-existing dedupe_tbd_collisions behavior is unaffected by the new step."""
        CampaignRun = self.new_apps.get_model('solsys_code', 'CampaignRun')
        runs = CampaignRun.objects.filter(telescope_instrument='JWST')

        self.assertEqual(runs.count(), 1)
        self.assertIsNone(runs.get().window_start)

    def test_backfill_sets_window_start_equal_window_end_for_every_surviving_row(self):
        """SCHED-05: window_start == window_end == the row's former obs_date, for all survivors."""
        CampaignRun = self.new_apps.get_model('solsys_code', 'CampaignRun')

        for run in CampaignRun.objects.filter(window_start__isnull=False):
            self.assertEqual(run.window_start, run.window_end)
