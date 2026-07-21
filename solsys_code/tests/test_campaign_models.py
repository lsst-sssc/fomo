from django.db import IntegrityError, transaction
from django.test import TestCase
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.models import CampaignRun


class TestCampaignRunFieldInventory(TestCase):
    """CAMP-01: CampaignRun stores the full field inventory and re-fetches it correctly."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')

    def test_full_field_inventory_persists_and_reloads(self):
        """SCHED-02: a single-night run (window_start == window_end) persists and reloads."""
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            site_raw='F65',
            site_needs_review=False,
            window_start='2025-07-04',
            window_end='2025-07-04',
            filters_bandpass='griz',
            observation_details='Photometric monitoring',
            weather='Clear',
            observation_outcome='Detected',
            publication_plans='TBD',
            open_to_collaboration=True,
            comments='Nothing unusual',
            contact_person='Test Person',
            contact_email='test@example.com',
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            run_status=CampaignRun.RunStatus.OBSERVED,
        )

        reloaded = CampaignRun.objects.get(pk=run.pk)

        self.assertEqual(reloaded.campaign, self.campaign)
        self.assertEqual(reloaded.telescope_instrument, 'FTN/MuSCAT3')
        self.assertEqual(reloaded.site_raw, 'F65')
        self.assertFalse(reloaded.site_needs_review)
        self.assertEqual(str(reloaded.window_start), '2025-07-04')
        self.assertEqual(reloaded.window_start, reloaded.window_end)
        self.assertFalse(hasattr(reloaded, 'obs_date'))
        self.assertFalse(hasattr(reloaded, 'ut_start'))
        self.assertFalse(hasattr(reloaded, 'ut_end'))
        self.assertEqual(reloaded.filters_bandpass, 'griz')
        self.assertEqual(reloaded.observation_details, 'Photometric monitoring')
        self.assertEqual(reloaded.weather, 'Clear')
        self.assertEqual(reloaded.observation_outcome, 'Detected')
        self.assertEqual(reloaded.publication_plans, 'TBD')
        self.assertTrue(reloaded.open_to_collaboration)
        self.assertEqual(reloaded.comments, 'Nothing unusual')
        self.assertEqual(reloaded.contact_person, 'Test Person')
        self.assertEqual(reloaded.contact_email, 'test@example.com')
        self.assertEqual(reloaded.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(reloaded.run_status, CampaignRun.RunStatus.OBSERVED)


class TestCampaignRunWindowSchema(TestCase):
    """SCHED-03/SCHED-04: TBD runs and the two partial UniqueConstraints."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')

    def test_tbd_run_saves_with_both_window_fields_null(self):
        """SCHED-03: a fully-TBD run (both window fields null) persists and reloads."""
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='JWST',
            window_start=None,
            window_end=None,
            contact_person='Carrie Holt',
        )

        reloaded = CampaignRun.objects.get(pk=run.pk)

        self.assertIsNone(reloaded.window_start)
        self.assertIsNone(reloaded.window_end)

    def test_tbd_same_contact_person_collides(self):
        """SCHED-04 (TBD branch): same campaign+telescope_instrument+contact_person collides."""
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='JWST',
            window_start=None,
            window_end=None,
            contact_person='Carrie Holt',
        )

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CampaignRun.objects.create(
                    campaign=self.campaign,
                    telescope_instrument='JWST',
                    window_start=None,
                    window_end=None,
                    contact_person='Carrie Holt',
                )

    def test_tbd_differing_contact_person_both_save(self):
        """SCHED-04 (TBD branch): contact_person discriminates otherwise-identical TBD rows."""
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='JWST',
            window_start=None,
            window_end=None,
            contact_person='Carrie Holt',
        )
        with transaction.atomic():
            second = CampaignRun.objects.create(
                campaign=self.campaign,
                telescope_instrument='JWST',
                window_start=None,
                window_end=None,
                contact_person='Martin Cordiner',
            )

        self.assertIsNotNone(second.pk)
        self.assertEqual(CampaignRun.objects.filter(telescope_instrument='JWST').count(), 2)

    def test_resolved_window_same_key_collides(self):
        """SCHED-04 (resolved branch): same campaign+telescope_instrument+window_start+window_end collides."""
        CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start='2025-07-04',
            window_end='2025-07-04',
        )

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CampaignRun.objects.create(
                    campaign=self.campaign,
                    telescope_instrument='FTN/MuSCAT3',
                    window_start='2025-07-04',
                    window_end='2025-07-04',
                )

    def test_mismatched_window_start_end_pair_rejected_by_db(self):
        """WR-02: window_start/window_end must be null together at the DB level."""
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                CampaignRun.objects.create(
                    campaign=self.campaign,
                    telescope_instrument='FTN/MuSCAT3',
                    window_start='2025-07-04',
                    window_end=None,
                )


class TestCampaignRunOptionalTarget(TestCase):
    """CAMP-02: target is nullable; single-target campaigns work without ever setting it."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')

    def test_campaign_run_without_target_persists_and_reloads(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='VLT/MUSE',
        )

        reloaded = CampaignRun.objects.get(pk=run.pk)

        self.assertIsNone(reloaded.target)

    def test_campaign_run_with_linked_target_persists_and_reloads(self):
        target = NonSiderealTargetFactory.create()
        self.campaign.targets.add(target)

        run = CampaignRun.objects.create(
            campaign=self.campaign,
            target=target,
            telescope_instrument='FTN/MuSCAT3',
        )

        reloaded = CampaignRun.objects.get(pk=run.pk)

        self.assertEqual(reloaded.target, target)


class TestCampaignRunStatusVocabulary(TestCase):
    """CAMP-03: two-field status with correct defaults and controlled vocabulary sizes."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')

    def test_default_statuses_on_fresh_campaign_run(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
        )

        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self.assertEqual(run.run_status, CampaignRun.RunStatus.REQUESTED)

    def test_approval_status_has_exactly_three_members(self):
        self.assertEqual(len(CampaignRun.ApprovalStatus.choices), 3)

    def test_run_status_has_exactly_eight_members(self):
        self.assertEqual(len(CampaignRun.RunStatus.choices), 8)


class TestCampaignRunWindowNeedsReviewFields(TestCase):
    """IMPORT-01/IMPORT-02: original_obs_date_raw/window_needs_review defaults and persistence."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')

    def test_defaults_on_fresh_campaign_run(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
        )

        self.assertEqual(run.original_obs_date_raw, '')
        self.assertFalse(run.window_needs_review)

    def test_fields_are_assignable_and_persist(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='JWST',
            contact_person='Carrie Holt',
            original_obs_date_raw='TBD pending Cycle 2',
            window_needs_review=True,
        )

        reloaded = CampaignRun.objects.get(pk=run.pk)

        self.assertEqual(reloaded.original_obs_date_raw, 'TBD pending Cycle 2')
        self.assertTrue(reloaded.window_needs_review)
