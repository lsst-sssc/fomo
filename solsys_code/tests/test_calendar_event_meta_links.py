"""PROJ-04 (33-CONTEXT.md, ROADMAP.md Phase 33 criterion 1): regression coverage for
``CalendarEventMeta.observation_record``/``.observation_group`` -- the two real foreign keys
Phase 34's observation projector will write.

Two classes:

- ``CalendarEventMetaObservationLinksFieldTests`` exercises the field-level behaviour
  (nullability, the one-to-one collision, the plain-FK non-collision, and SET_NULL survival
  on delete) against the already-migrated test schema, the same way every other model-level
  test in this project works.
- ``TestCalendarEventMetaObservationLinksMigration`` exists because no other test in this
  project exercises migration 0017 against pre-existing rows: it seeds ``CalendarEventMeta``
  rows against the historical (pre-0017) schema with ``run``/``is_verified``/``confirmed_by``/
  ``confirmed_at`` populated, migrates forward, and asserts those four values survive
  byte-identical while both new link columns land NULL -- ROADMAP criterion 1's load-bearing
  proof.
"""

from datetime import datetime
from datetime import timezone as dt_timezone

from django.db import IntegrityError, connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.test import TestCase, TransactionTestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import CalendarEventMeta


class CalendarEventMetaObservationLinksFieldTests(TestCase):
    """Field-level behaviour of ``observation_record`` (OneToOneField, D-05) and
    ``observation_group`` (ForeignKey, D-06), against the already-migrated test schema."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target = NonSiderealTargetFactory.create()
        cls.record_a = ObservationRecord.objects.create(
            target=cls.target,
            facility='LCO',
            observation_id='cem-links-record-a',
            status='PENDING',
            parameters={'proposal': 'TEST'},
        )
        cls.record_b = ObservationRecord.objects.create(
            target=cls.target,
            facility='LCO',
            observation_id='cem-links-record-b',
            status='PENDING',
            parameters={'proposal': 'TEST'},
        )
        cls.group = ObservationGroup.objects.create(name='cem-links-group')

    def _make_event(self, title: str) -> CalendarEvent:
        return CalendarEvent.objects.create(
            title=title,
            start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )

    def test_both_links_unset_by_default(self) -> None:
        meta = CalendarEventMeta.objects.create(event=self._make_event('unset-links'))
        self.assertIsNone(meta.observation_record_id)
        self.assertIsNone(meta.observation_group_id)

    def test_two_rows_may_both_leave_observation_record_null(self) -> None:
        """NULLs are not equal for the one-to-one constraint -- any number of rows may leave
        observation_record unset without colliding."""
        meta_1 = CalendarEventMeta.objects.create(event=self._make_event('null-record-1'))
        meta_2 = CalendarEventMeta.objects.create(event=self._make_event('null-record-2'))
        self.assertIsNone(meta_1.observation_record_id)
        self.assertIsNone(meta_2.observation_record_id)

    def test_second_row_pointing_at_same_observation_record_raises_integrity_error(self) -> None:
        CalendarEventMeta.objects.create(event=self._make_event('duplicate-record-1'), observation_record=self.record_a)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CalendarEventMeta.objects.create(
                    event=self._make_event('duplicate-record-2'), observation_record=self.record_a
                )

    def test_several_rows_may_point_at_the_same_observation_group(self) -> None:
        """observation_group is a plain foreign key (D-06) -- several events in a series
        share one group."""
        meta_1 = CalendarEventMeta.objects.create(
            event=self._make_event('shared-group-1'), observation_group=self.group
        )
        meta_2 = CalendarEventMeta.objects.create(
            event=self._make_event('shared-group-2'), observation_group=self.group
        )
        meta_3 = CalendarEventMeta.objects.create(
            event=self._make_event('shared-group-3'), observation_group=self.group
        )
        self.assertEqual(
            {meta_1.observation_group_id, meta_2.observation_group_id, meta_3.observation_group_id},
            {self.group.pk},
        )

    def test_deleting_observation_record_clears_only_that_link(self) -> None:
        event = self._make_event('delete-record')
        meta = CalendarEventMeta.objects.create(
            event=event,
            observation_record=self.record_a,
            observation_group=self.group,
            is_verified=False,
        )

        self.record_a.delete()

        meta.refresh_from_db()
        self.assertIsNone(meta.observation_record_id)
        self.assertEqual(meta.observation_group_id, self.group.pk)
        self.assertFalse(meta.is_verified)
        self.assertIsNone(meta.run_id)
        self.assertIsNone(meta.confirmed_by_id)
        self.assertIsNone(meta.confirmed_at)
        self.assertTrue(CalendarEvent.objects.filter(pk=event.pk).exists())

    def test_deleting_observation_group_clears_only_that_link(self) -> None:
        event = self._make_event('delete-group')
        meta = CalendarEventMeta.objects.create(
            event=event,
            observation_record=self.record_b,
            observation_group=self.group,
            is_verified=False,
        )

        self.group.delete()

        meta.refresh_from_db()
        self.assertIsNone(meta.observation_group_id)
        self.assertEqual(meta.observation_record_id, self.record_b.pk)
        self.assertFalse(meta.is_verified)
        self.assertIsNone(meta.run_id)
        self.assertIsNone(meta.confirmed_by_id)
        self.assertIsNone(meta.confirmed_at)
        self.assertTrue(CalendarEvent.objects.filter(pk=event.pk).exists())


class TestCalendarEventMetaObservationLinksMigration(TransactionTestCase):
    """Exercises migration 0017 (two AddFields + one AlterField) against pre-existing rows.

    Seeds rows against the historical (pre-0017) schema mirroring the real dev-DB mix (one
    row with the audit fields populated, one with them NULL), migrates forward, and asserts
    each row's four pre-existing values are byte-identical while both new link columns land
    NULL.
    """

    migrate_from = [('solsys_code', '0016_alter_campaignrun_source_soar_queue')]
    migrate_to = [('solsys_code', '0017_calendareventmeta_observation_links')]

    def setUp(self):
        # Start from the pre-0017 schema (observation_record/observation_group not yet
        # present, run's verbose_name still the pre-D-17 wording).
        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_from)
        old_apps = executor.loader.project_state(self.migrate_from).apps

        CalendarEvent = old_apps.get_model('tom_calendar', 'CalendarEvent')
        TargetList = old_apps.get_model('tom_targets', 'TargetList')
        CampaignRun = old_apps.get_model('solsys_code', 'CampaignRun')
        CalendarEventMeta = old_apps.get_model('solsys_code', 'CalendarEventMeta')
        User = old_apps.get_model('auth', 'User')

        campaign = TargetList.objects.create(name='PROJ-04 Migration Campaign')
        run = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start='2026-09-01',
            window_end='2026-09-01',
        )
        confirming_user = User.objects.create(username='cem-links-migration-confirmer')

        event_confirmed = CalendarEvent.objects.create(
            title='PROJ-04 migration event (confirmed)',
            start_time=datetime(2026, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        event_unconfirmed = CalendarEvent.objects.create(
            title='PROJ-04 migration event (unconfirmed)',
            start_time=datetime(2026, 9, 2, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 3, 6, 0, tzinfo=dt_timezone.utc),
        )

        confirmed_at = datetime(2026, 9, 1, 12, 0, tzinfo=dt_timezone.utc)
        self.confirmed_pk = CalendarEventMeta.objects.create(
            event=event_confirmed,
            run=run,
            is_verified=True,
            confirmed_by=confirming_user,
            confirmed_at=confirmed_at,
        ).pk
        self.confirmed_expected = {
            'run_id': run.pk,
            'is_verified': True,
            'confirmed_by_id': confirming_user.pk,
            'confirmed_at': confirmed_at,
        }
        self.unconfirmed_pk = CalendarEventMeta.objects.create(
            event=event_unconfirmed,
            run=None,
            is_verified=False,
            confirmed_by=None,
            confirmed_at=None,
        ).pk
        self.unconfirmed_expected = {
            'run_id': None,
            'is_verified': False,
            'confirmed_by_id': None,
            'confirmed_at': None,
        }

        # Migrate forward through 0017.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(self.migrate_to)
        self.new_apps = executor.loader.project_state(self.migrate_to).apps

    def tearDown(self):
        # Leave the DB on the latest migration state for any tests that run after this one.
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())

    def _calendar_event_meta_model(self):
        return self.new_apps.get_model('solsys_code', 'CalendarEventMeta')

    def test_pre_existing_values_survive_byte_identical(self):
        CalendarEventMeta = self._calendar_event_meta_model()

        confirmed = CalendarEventMeta.objects.get(pk=self.confirmed_pk)
        for field, expected in self.confirmed_expected.items():
            self.assertEqual(getattr(confirmed, field), expected, field)

        unconfirmed = CalendarEventMeta.objects.get(pk=self.unconfirmed_pk)
        for field, expected in self.unconfirmed_expected.items():
            self.assertEqual(getattr(unconfirmed, field), expected, field)

    def test_both_new_link_columns_are_null_on_every_migrated_row(self):
        CalendarEventMeta = self._calendar_event_meta_model()

        for row in CalendarEventMeta.objects.all():
            self.assertIsNone(row.observation_record_id)
            self.assertIsNone(row.observation_group_id)
        self.assertEqual(CalendarEventMeta.objects.filter(observation_record__isnull=False).count(), 0)
        self.assertEqual(CalendarEventMeta.objects.filter(observation_group__isnull=False).count(), 0)
