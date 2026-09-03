"""Unit tests for the five sites that dereference ``run.campaign.name``/``run.campaign_id``
(32-01-PLAN.md Task 2). Each site gets a null-campaign case (renders without raising) and a
with-campaign regression case (today's output is unchanged).

Fixture style mirrors ``CampaignReconcilerTestBase`` in test_campaign_reconciler.py.
"""

from datetime import date
from types import SimpleNamespace

from django.test import TestCase
from tom_targets.models import TargetList

from solsys_code.campaign_attribution import _campaign_evidence
from solsys_code.campaign_reconciler import event_title
from solsys_code.campaign_tables import AttributionConfirmedTable, AttributionDismissedTable
from solsys_code.models import NO_CAMPAIGN_LABEL, CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory


class NullCampaignGuardTestBase(TestCase):
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

    def _make_run(self, **overrides) -> CampaignRun:
        kwargs = {
            'campaign': self.campaign,
            'telescope_instrument': 'NTT/EFOSC2',
            'site': self.ground_site,
            'window_start': date(2026, 9, 3),
            'window_end': date(2026, 9, 3),
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)


class TestCampaignRunStrGuard(NullCampaignGuardTestBase):
    """CampaignRun.__str__() (Task 1)."""

    def test_null_campaign_run_str_uses_no_campaign_label(self):
        run = self._make_run(campaign=None)

        label = str(run)

        self.assertIn(NO_CAMPAIGN_LABEL, label)

    def test_with_campaign_run_str_uses_campaign_name(self):
        run = self._make_run()

        label = str(run)

        self.assertIn('3I/ATLAS', label)
        self.assertNotIn(NO_CAMPAIGN_LABEL, label)


class TestEventTitleGuard(NullCampaignGuardTestBase):
    """campaign_reconciler.event_title() (Task 1, the hot-path guard)."""

    def test_null_campaign_run_title_has_no_prefix(self):
        run = self._make_run(campaign=None)

        self.assertEqual(event_title(run), 'NTT/EFOSC2')

    def test_with_campaign_title_is_unchanged(self):
        run = self._make_run()

        self.assertEqual(event_title(run), '3I/ATLAS: NTT/EFOSC2')


class TestDismissalHistoryTableRenderRunGuard(NullCampaignGuardTestBase):
    """AttributionDismissedTable.render_run() (the dismissal-history table)."""

    def test_null_campaign_run_renders_bare_telescope_instrument(self):
        run = self._make_run(campaign=None)
        record = SimpleNamespace(run=run)

        self.assertEqual(AttributionDismissedTable(data=[]).render_run(record), 'NTT/EFOSC2')

    def test_with_campaign_run_renders_parenthetical_unchanged(self):
        run = self._make_run()
        record = SimpleNamespace(run=run)

        self.assertEqual(AttributionDismissedTable(data=[]).render_run(record), 'NTT/EFOSC2 (3I/ATLAS)')


class TestConfirmedPairTableRenderRunGuard(NullCampaignGuardTestBase):
    """AttributionConfirmedTable.render_run() (the confirmed-pair table)."""

    def test_null_campaign_run_renders_bare_telescope_instrument(self):
        run = self._make_run(campaign=None)
        record = SimpleNamespace(run=run)

        self.assertEqual(AttributionConfirmedTable(data=[]).render_run(record), 'NTT/EFOSC2')

    def test_with_campaign_run_renders_parenthetical_unchanged(self):
        run = self._make_run()
        record = SimpleNamespace(run=run)

        self.assertEqual(AttributionConfirmedTable(data=[]).render_run(record), 'NTT/EFOSC2 (3I/ATLAS)')


class TestCampaignEvidenceGuard(NullCampaignGuardTestBase):
    """campaign_attribution._campaign_evidence()."""

    def test_null_campaign_run_evidence_states_no_campaign(self):
        run = self._make_run(campaign=None)

        evidence = _campaign_evidence(run)

        self.assertIn('no campaign', evidence)
        self.assertIn(str(run.pk), evidence)

    def test_with_campaign_run_evidence_is_unchanged(self):
        run = self._make_run()

        evidence = _campaign_evidence(run)

        self.assertEqual(
            evidence,
            f"run belongs to campaign '3I/ATLAS' (pk={run.campaign_id}), matching the orphan's campaign",
        )
