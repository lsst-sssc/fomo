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
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            site_raw='F65',
            site_needs_review=False,
            obs_date='2025-07-04',
            ut_start='2025-07-04T08:50:00Z',
            ut_end='2025-07-04T11:50:00Z',
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
        self.assertEqual(str(reloaded.obs_date), '2025-07-04')
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
