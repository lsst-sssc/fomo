"""Unit tests for the Phase 32 groundwork: schema (SCHEMA-01/02), the
``event_title()`` null-campaign guard, and ``write_and_reconcile_campaign_run()``
(32-01-PLAN.md Task 1).

Fixture style mirrors ``CampaignReconcilerTestBase`` in test_campaign_reconciler.py.
"""

from datetime import date

from django.db import IntegrityError, transaction
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_targets.models import TargetList

from solsys_code.campaign_reconciler import event_title
from solsys_code.campaign_utils import write_and_reconcile_campaign_run
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
