"""Unit tests for the Phase 32 groundwork: schema (SCHEMA-01/02), the
``event_title()`` null-campaign guard, and ``write_and_reconcile_campaign_run()``
(32-01-PLAN.md Task 1).

Fixture style mirrors ``CampaignReconcilerTestBase`` in test_campaign_reconciler.py.
"""

import io
import pathlib
import tempfile
from datetime import date, time

from django.core.management import call_command
from django.db import IntegrityError, transaction
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.allocation_projector import ALLOC_URL_NAMESPACE
from solsys_code.campaign_reconciler import event_title
from solsys_code.campaign_utils import preview_campaign_run_action, write_and_reconcile_campaign_run
from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory


class WriteAndReconcileTestBase(TestCase):
    """Shared fixture: one campaign, one resolvable ground Observatory."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.ground_site = Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )


class TestNullableCampaignAndSourceIdentifierSchema(WriteAndReconcileTestBase):
    """SCHEMA-01/02: null campaign, source_identifier partial unique constraint."""

    def test_null_campaign_run_with_source_identifier_saves(self):
        run = CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            site=self.ground_site,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            source=CampaignRun.Source.CLASSICAL_FILE,
            source_identifier='CLASSICAL:NTT:EFOSC2:2026-09-03',
        )

        reloaded = CampaignRun.objects.get(pk=run.pk)

        self.assertIsNone(reloaded.campaign_id)
        self.assertEqual(reloaded.source_identifier, 'CLASSICAL:NTT:EFOSC2:2026-09-03')

    def test_duplicate_source_identifier_raises_integrity_error(self):
        CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            site=self.ground_site,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            source=CampaignRun.Source.CLASSICAL_FILE,
            source_identifier='CLASSICAL:NTT:EFOSC2:2026-09-03',
        )

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CampaignRun.objects.create(
                    campaign=None,
                    telescope_instrument='NTT/EFOSC2',
                    site=self.ground_site,
                    window_start=date(2026, 9, 4),
                    window_end=date(2026, 9, 4),
                    approval_status=CampaignRun.ApprovalStatus.APPROVED,
                    source=CampaignRun.Source.CLASSICAL_FILE,
                    source_identifier='CLASSICAL:NTT:EFOSC2:2026-09-03',
                )

    def test_two_null_source_identifier_rows_do_not_collide(self):
        first = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
        )
        with transaction.atomic():
            second = CampaignRun.objects.create(
                campaign=self.campaign,
                telescope_instrument='FTN/MuSCAT3',
                window_start=date(2026, 9, 4),
                window_end=date(2026, 9, 4),
            )

        self.assertIsNone(first.source_identifier)
        self.assertIsNone(second.source_identifier)
        self.assertNotEqual(first.pk, second.pk)

    def test_soar_queue_source_value(self):
        self.assertEqual(CampaignRun.Source.SOAR_QUEUE.value, 'soar_queue')
        self.assertEqual(CampaignRun.Source.SOAR_QUEUE.label, 'SOAR queue')


class TestEventTitleNullCampaignGuard(WriteAndReconcileTestBase):
    """event_title() must not raise on a null-campaign run, and no longer embeds a
    campaign label whether or not the run has a campaign (D-12, Phase 33 -- the decoration
    tag is now the single campaign label)."""

    def test_null_campaign_run_title_has_no_campaign_label(self):
        run = CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            site=self.ground_site,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            source=CampaignRun.Source.CLASSICAL_FILE,
            source_identifier='CLASSICAL:NTT:EFOSC2:2026-09-03-title',
        )

        self.assertEqual(event_title(run), 'NTT/EFOSC2')

    def test_with_campaign_title_also_has_no_campaign_label(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            site=self.ground_site,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )

        self.assertEqual(event_title(run), 'FTN/MuSCAT3')


class TestWriteAndReconcileCampaignRun(WriteAndReconcileTestBase):
    """write_and_reconcile_campaign_run() round-trips a null-campaign run through
    reconcile_run() into exactly one CalendarEvent, idempotently."""

    def _fields(self, **overrides) -> dict:
        fields = {
            'campaign': None,
            'telescope_instrument': 'NTT/EFOSC2',
            'site': self.ground_site,
            'window_start': date(2026, 9, 3),
            'window_end': date(2026, 9, 3),
            'observation_details': 'Photometric monitoring',
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'source': CampaignRun.Source.CLASSICAL_FILE,
        }
        fields.update(overrides)
        return fields

    def test_fresh_write_creates_run_and_one_calendar_event(self):
        lookup = {'source_identifier': 'CLASSICAL:NTT:EFOSC2:2026-09-03-fresh'}
        fields = self._fields(source_identifier=lookup['source_identifier'])

        result = write_and_reconcile_campaign_run(lookup, fields)

        self.assertEqual(result.action, 'created')
        self.assertEqual(result.reconcile.created, 1)
        self.assertIsNone(result.reconcile.skipped_reason)
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_second_identical_call_reports_unchanged_no_churn(self):
        lookup = {'source_identifier': 'CLASSICAL:NTT:EFOSC2:2026-09-03-nochurn'}
        fields = self._fields(source_identifier=lookup['source_identifier'])
        write_and_reconcile_campaign_run(lookup, fields)

        result = write_and_reconcile_campaign_run(lookup, fields)

        self.assertEqual(result.action, 'unchanged')
        self.assertEqual(result.reconcile.unchanged, 1)
        self.assertEqual(CalendarEvent.objects.count(), 1)

    def test_missing_approval_status_surfaces_skipped_reason_to_caller(self):
        lookup = {'source_identifier': 'CLASSICAL:NTT:EFOSC2:2026-09-03-unapproved'}
        fields = self._fields(source_identifier=lookup['source_identifier'])
        del fields['approval_status']

        result = write_and_reconcile_campaign_run(lookup, fields)

        self.assertEqual(result.reconcile.skipped_reason, 'not approved')
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_existing_web_sourced_row_keeps_its_source_and_approval_status(self):
        lookup = {'source_identifier': 'CLASSICAL:NTT:EFOSC2:2026-09-03-web'}
        existing = CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            site=self.ground_site,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
            source=CampaignRun.Source.WEB,
            source_identifier=lookup['source_identifier'],
        )
        fields = self._fields(
            source_identifier=lookup['source_identifier'],
            observation_details='Updated details from the classical adapter',
        )

        result = write_and_reconcile_campaign_run(lookup, fields)

        result.run.refresh_from_db()
        self.assertEqual(result.run.pk, existing.pk)
        self.assertEqual(result.run.source, CampaignRun.Source.WEB)
        self.assertEqual(result.run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)
        # Non-guarded fields still apply.
        self.assertEqual(result.run.observation_details, 'Updated details from the classical adapter')


class TestPreviewCampaignRunAction(WriteAndReconcileTestBase):
    """preview_campaign_run_action() -- the run-level twin of
    calendar_utils.preview_calendar_event_action() (Phase 35 Task 2): a --dry-run preview
    must never disagree with what insert_or_create_campaign_run() would report."""

    def test_none_run_returns_created(self):
        action = preview_campaign_run_action(None, {'telescope_instrument': 'NTT/EFOSC2'})

        self.assertEqual(action, 'created')

    def test_differing_field_returns_updated_and_writes_nothing(self):
        run = CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            site=self.ground_site,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            source=CampaignRun.Source.CLASSICAL_FILE,
            source_identifier='CLASSICAL:NTT:EFOSC2:2026-09-03-preview-updated',
            observation_details='Old details',
        )
        run_count_before = CampaignRun.objects.count()

        action = preview_campaign_run_action(run, {'observation_details': 'New details'})

        self.assertEqual(action, 'updated')
        reloaded = CampaignRun.objects.get(pk=run.pk)
        self.assertEqual(reloaded.observation_details, 'Old details')
        self.assertEqual(CampaignRun.objects.count(), run_count_before)

    def test_identical_fields_returns_unchanged_and_writes_nothing(self):
        run = CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            site=self.ground_site,
            window_start=date(2026, 9, 3),
            window_end=date(2026, 9, 3),
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            source=CampaignRun.Source.CLASSICAL_FILE,
            source_identifier='CLASSICAL:NTT:EFOSC2:2026-09-03-preview-unchanged',
            observation_details='Same details',
        )
        run_count_before = CampaignRun.objects.count()

        action = preview_campaign_run_action(run, {'observation_details': 'Same details'})

        self.assertEqual(action, 'unchanged')
        self.assertEqual(CampaignRun.objects.count(), run_count_before)


class TestLoadTelescopeRunsWritesAllocations(TestCase):
    """load_telescope_runs writes allocations, not calendar events (Phase 35 Task 2):
    each schedule line becomes one campaign-less CampaignRun, routed through
    write_and_reconcile_campaign_run(), and the allocation projector draws the nights."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.ntt = Observatory.objects.create(
            obscode='809',
            name='ESO, La Silla',
            short_name='NTT',
            lat=-29.2567,
            lon=-70.7300,
            altitude=2347,
            timezone='America/Santiago',
        )

    def _write_schedule_file(self, lines: list[str]) -> tuple[str, tempfile.TemporaryDirectory]:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmpdir_ctx.name) / 'schedule.txt'
        path.write_text('\n'.join(lines) + '\n')
        return str(path), tmpdir_ctx

    def test_import_creates_one_campaign_run_with_expected_fields(self):
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(CampaignRun.objects.count(), 1)
        run = CampaignRun.objects.get()
        year = date.today().year
        self.assertIsNone(run.campaign_id)
        self.assertEqual(run.source, CampaignRun.Source.CLASSICAL_FILE)
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.run_status, CampaignRun.RunStatus.PLANNED)
        self.assertEqual(run.telescope_instrument, 'NTT/EFOSC2')
        self.assertEqual(run.site_id, self.ntt.pk)
        self.assertEqual(run.site.obscode, '809')
        self.assertEqual(run.site_raw, 'NTT')
        self.assertEqual(run.window_start, date(year, 7, 9))
        self.assertEqual(run.window_end, date(year, 7, 12))
        self.assertIsNone(run.night_start_utc)
        self.assertIsNone(run.night_end_utc)

    def test_import_creates_four_alloc_events_and_zero_blank_url_events(self):
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(CalendarEvent.objects.filter(url__startswith=ALLOC_URL_NAMESPACE).count(), 4)
        self.assertEqual(CalendarEvent.objects.filter(url='').count(), 0)

    def test_cancelled_line_gives_cancelled_run_status_and_prefixed_titles(self):
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 9-13 July (cancelled)'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        run = CampaignRun.objects.get()
        self.assertEqual(run.run_status, CampaignRun.RunStatus.CANCELLED)
        events = CalendarEvent.objects.filter(url__startswith=ALLOC_URL_NAMESPACE)
        self.assertGreater(events.count(), 0)
        for event in events:
            self.assertTrue(event.title.startswith('[CANCELLED]'), event.title)

    def test_partial_night_bon_hhmm_stores_only_end_sub_night_field(self):
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July BoN-0626'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        run = CampaignRun.objects.get()
        self.assertIsNone(run.night_start_utc)
        self.assertEqual(run.night_end_utc, time(6, 26))

    def test_colliding_lines_produce_one_run_and_a_reported_collision(self):
        path, tmpdir_ctx = self._write_schedule_file(
            [
                'NTT EFOSC2 allocation 9-13 July',
                'NTT EFOSC2 allocation 9-13 July',
            ]
        )
        with tmpdir_ctx:
            stderr_buf = io.StringIO()
            stdout_buf = io.StringIO()
            call_command('load_telescope_runs', path, stdout=stdout_buf, stderr=stderr_buf)

        self.assertEqual(CampaignRun.objects.count(), 1)
        err = stderr_buf.getvalue()
        self.assertIn('Line 2', err)
        self.assertIn('line 1', err)
        self.assertIn('skipped_collision: 1', stdout_buf.getvalue())

    def test_dry_run_writes_nothing_and_reports_same_vocabulary(self):
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            stdout_buf = io.StringIO()
            call_command('load_telescope_runs', path, '--dry-run', stdout=stdout_buf, stderr=io.StringIO())

        self.assertEqual(CampaignRun.objects.count(), 0)
        self.assertEqual(CalendarEvent.objects.count(), 0)
        summary = stdout_buf.getvalue()
        self.assertIn('created: 1', summary)
        self.assertIn('skipped_collision: 0', summary)
        self.assertIn('nights -- created: 4', summary)

    def test_existing_web_sourced_run_is_never_relabelled_or_auto_approved(self):
        year = date.today().year
        key = f'CLASSICAL:NTT:EFOSC2:{year}-07-09:{year}-07-12:BoN:EoN'
        CampaignRun.objects.create(
            campaign=None,
            telescope_instrument='NTT/EFOSC2',
            site=self.ntt,
            site_raw='NTT',
            window_start=date(year, 7, 9),
            window_end=date(year, 7, 12),
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
            source=CampaignRun.Source.WEB,
            source_identifier=key,
        )
        path, tmpdir_ctx = self._write_schedule_file(['NTT EFOSC2 allocation 9-13 July'])
        with tmpdir_ctx:
            call_command('load_telescope_runs', path, stdout=io.StringIO(), stderr=io.StringIO())

        run = CampaignRun.objects.get(source_identifier=key)
        self.assertEqual(run.source, CampaignRun.Source.WEB)
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)
