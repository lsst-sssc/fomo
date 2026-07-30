"""CANON-03: regression coverage for migrations 0008 (RenameModel:
CalendarEventTelescopeLabel -> CalendarEventMeta) and 0009 (AddField: run).

Everything else exercising CalendarEventMeta asserts the *post-migration* model shape
directly against the already-migrated test schema; it never proves that the rename itself
preserves the real companion rows' history. This module seeds rows against the historical
(pre-0008) schema under the model's pre-rename name, migrates forward through 0009, and
asserts row count, per-row is_verified, pk identity, and null run survive -- rather than
relying entirely on the unrepeatable manual proof already run once against a scratch copy
of the dev DB (26-DECISION.md Criterion 4, Task 2 of this plan).
"""

from datetime import datetime
from datetime import timezone as dt_timezone

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase


class TestCompanionRecordRenamePreservesHistory(TransactionTestCase):
    """Exercises migrations 0008 (rename) and 0009 (AddField run) end-to-end."""

    migrate_from = [('solsys_code', '0007_campaignrun_contact_public_opt_in')]
    migrate_to = [('solsys_code', '0009_calendareventmeta_run')]

    def setUp(self):
        # Start from the pre-rename schema (CalendarEventTelescopeLabel still present).
        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_from)
        old_apps = executor.loader.project_state(self.migrate_from).apps

        CalendarEvent = old_apps.get_model('tom_calendar', 'CalendarEvent')
        # The pre-rename name is correct here -- that is the whole point of a historical
        # model state; this is not a stray reference to the old class that should have been
        # updated by Task 1.
        CalendarEventTelescopeLabel = old_apps.get_model('solsys_code', 'CalendarEventTelescopeLabel')

        event_a = CalendarEvent.objects.create(
            title='Event A',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
        )
        event_b = CalendarEvent.objects.create(
            title='Event B',
            start_time=datetime(2026, 7, 8, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 9, 6, 0, tzinfo=dt_timezone.utc),
        )
        event_c = CalendarEvent.objects.create(
            title='Event C',
            start_time=datetime(2026, 7, 9, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 10, 6, 0, tzinfo=dt_timezone.utc),
        )

        # Seed a mix of is_verified=True/False across two events, so a silent table drop
        # (row count -> 0) or a default-value reset (every row landing on True) would both
        # be caught.
        self.verified_pk = event_a.pk
        self.fallback_pk = event_b.pk
        self.verified_2_pk = event_c.pk
        CalendarEventTelescopeLabel.objects.create(event=event_a, is_verified=True)
        CalendarEventTelescopeLabel.objects.create(event=event_b, is_verified=False)
        CalendarEventTelescopeLabel.objects.create(event=event_c, is_verified=True)

        # Migrate forward through 0009 (rename, then AddField run).
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.migrate_to)
        self.new_apps = executor.loader.project_state(self.migrate_to).apps

    def tearDown(self):
        # Leave the DB on the latest migration state for any test that runs after this one.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())

    def test_row_count_survives_the_rename(self):
        """A DeleteModel/CreateModel pair would have made this zero -- the regression this
        test exists to catch."""
        CalendarEventMeta = self.new_apps.get_model('solsys_code', 'CalendarEventMeta')
        self.assertEqual(CalendarEventMeta.objects.count(), 3)

    def test_is_verified_survives_per_row_matched_by_pk(self):
        """Matching by event_id (the model's primary key) proves pk identity is preserved,
        not just the aggregate row count."""
        CalendarEventMeta = self.new_apps.get_model('solsys_code', 'CalendarEventMeta')

        self.assertTrue(CalendarEventMeta.objects.get(event_id=self.verified_pk).is_verified)
        self.assertFalse(CalendarEventMeta.objects.get(event_id=self.fallback_pk).is_verified)
        self.assertTrue(CalendarEventMeta.objects.get(event_id=self.verified_2_pk).is_verified)

    def test_run_is_null_on_every_migrated_row(self):
        """The AddField in 0009 must not fabricate an owner for pre-existing rows."""
        CalendarEventMeta = self.new_apps.get_model('solsys_code', 'CalendarEventMeta')

        for row in CalendarEventMeta.objects.all():
            self.assertIsNone(row.run_id)
        self.assertEqual(CalendarEventMeta.objects.filter(run__isnull=False).count(), 0)

    def test_old_model_name_gone_from_post_migration_app_state(self):
        """A rename that left the old name reachable would mean RenameModel didn't take --
        the post-migration app state must raise LookupError for the pre-rename name."""
        with self.assertRaises(LookupError):
            self.new_apps.get_model('solsys_code', 'CalendarEventTelescopeLabel')


class TestSourceAndTelescopeClassBackfill(TransactionTestCase):
    """CANON-01/CANON-02: regression coverage for migrations 0010 (AddField source/
    telescope_class + CreateModel CampaignRunObservation) and 0011 (the telescope_class
    backfill).

    Seeds CampaignRun rows against the historical (pre-0010) model, mirroring D-16's real
    dev-DB row shapes, then migrates through 0011 and asserts both the static `source`
    default and the derived-rule `telescope_class` backfill landed correctly.
    """

    migrate_from = [('solsys_code', '0009_calendareventmeta_run')]
    migrate_to = [('solsys_code', '0011_backfill_campaignrun_telescope_class')]

    def setUp(self):
        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_from)
        old_apps = executor.loader.project_state(self.migrate_from).apps

        TargetList = old_apps.get_model('tom_targets', 'TargetList')
        Observatory = old_apps.get_model('solsys_code_observatory', 'Observatory')
        CampaignRun = old_apps.get_model('solsys_code', 'CampaignRun')

        campaign = TargetList.objects.create(name='3I/ATLAS')
        resolved_site = Observatory.objects.create(obscode='F65', name='Haleakala', short_name='FTN')

        # D-16's dev-DB row shapes, all site-less unless stated, each given a distinct
        # window so no two rows can collide on unique_campaign_run_resolved_window.
        self.lco_1m_pk = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='LCO 1m',
            site_raw='',
            window_start='2025-07-01',
            window_end='2025-07-01',
        ).pk
        self.lco_2m_pk = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='LCO 2m',
            site_raw='',
            window_start='2025-07-02',
            window_end='2025-07-02',
        ).pk
        self.juice_pk = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='JUICE',
            site_raw='',
            window_start='2025-07-03',
            window_end='2025-07-03',
        ).pk
        self.jwst_pk = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='JWST',
            site_raw='500@-170',
            window_start='2025-07-04',
            window_end='2025-07-04',
        ).pk
        self.hst_pk = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='HST STIS/COS',
            site_raw='250',
            window_start='2025-07-05',
            window_end='2025-07-05',
        ).pk
        self.swift_pk = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='Swift/UVOT',
            site_raw='',
            window_start='2025-07-06',
            window_end='2025-07-06',
        ).pk
        # Control row: text would otherwise match the site-less 'LCO 1m' row above, but this
        # one has a resolved site -- a run with a resolved site never gets a telescope_class
        # even when its instrument text names one.
        self.resolved_control_pk = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='LCO 1m',
            site=resolved_site,
            site_raw='F65',
            window_start='2025-07-07',
            window_end='2025-07-07',
        ).pk

        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.migrate_to)
        self.new_apps = executor.loader.project_state(self.migrate_to).apps

    def tearDown(self):
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())

    def _campaign_run_model(self):
        return self.new_apps.get_model('solsys_code', 'CampaignRun')

    def test_source_defaults_to_legacy_on_every_seeded_row_with_no_runpython(self):
        """26-DECISION.md Criterion 4: the static field default backfills every
        pre-milestone row with no RunPython step at all."""
        CampaignRun = self._campaign_run_model()

        self.assertEqual(CampaignRun.objects.exclude(source='legacy').count(), 0)
        self.assertEqual(CampaignRun.objects.count(), 7)

    def test_telescope_class_backfill_matches_d16_row_shapes(self):
        CampaignRun = self._campaign_run_model()

        self.assertEqual(CampaignRun.objects.get(pk=self.lco_1m_pk).telescope_class, '1m0')
        self.assertEqual(CampaignRun.objects.get(pk=self.lco_2m_pk).telescope_class, '2m0')
        self.assertEqual(CampaignRun.objects.get(pk=self.juice_pk).telescope_class, 'SPACE')
        # JWST/HST/Swift all have real MPC obscodes -- not SPACE -- and no aperture-class
        # signal in their text (D-11): they stay blank.
        self.assertEqual(CampaignRun.objects.get(pk=self.jwst_pk).telescope_class, '')
        self.assertEqual(CampaignRun.objects.get(pk=self.hst_pk).telescope_class, '')
        self.assertEqual(CampaignRun.objects.get(pk=self.swift_pk).telescope_class, '')

    def test_site_resolved_control_row_stays_blank_even_with_matching_text(self):
        """T-27-15: a site-resolved run must never get a telescope_class, even when its
        telescope_instrument text would otherwise derive one."""
        CampaignRun = self._campaign_run_model()

        self.assertEqual(CampaignRun.objects.get(pk=self.resolved_control_pk).telescope_class, '')
