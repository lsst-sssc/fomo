"""Offline, mocked-network tests for repair_stale_campaign_run_sites (D-16).

Mirrors test_import_campaign_csv.py's `_MPC_OBS_DATA_E10` fixture shape (every key
`MPCObscodeFetcher.to_observatory()` reads) for two satellite MPC records -- HST (`250`)
and Swift (`C52`). Both are satellite records, so `longitude`/`rhocosphi`/`rhosinphi` are
`None`, matching real MPC satellite rows and exercising `to_observatory()`'s
coordinate-less path (quick task 260725-kn4) -- a fixture with real coordinates would not
reproduce the live behaviour these rows actually have.
"""

import io
from unittest.mock import MagicMock, patch

import requests
from django.core.management import call_command
from django.test import TestCase
from tom_targets.models import TargetList

from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory

# Satellite MPC record for Hubble Space Telescope (obscode 250). Coordinate fields are
# None, matching every real MPC space-based site (verified live for 250, per
# solsys_code_observatory/utils.py's to_observatory() comment).
_MPC_OBS_DATA_250 = {
    'created_at': 'Sat, 25 May 2019 00:11:26 GMT',
    'longitude': None,
    'name_utf8': 'Hubble Space Telescope',
    'obscode': '250',
    'observations_type': 'satellite',
    'old_names': None,
    'rhocosphi': None,
    'rhosinphi': None,
    'short_name': 'HST',
    'updated_at': 'Tue, 15 Apr 2025 20:52:50 GMT',
    'uses_two_line_observations': True,
}

# Satellite MPC record for Swift (obscode C52) -- same coordinate-less shape as _MPC_OBS_DATA_250.
_MPC_OBS_DATA_C52 = {
    'created_at': 'Sat, 25 May 2019 00:11:26 GMT',
    'longitude': None,
    'name_utf8': 'Swift',
    'obscode': 'C52',
    'observations_type': 'satellite',
    'old_names': None,
    'rhocosphi': None,
    'rhosinphi': None,
    'short_name': 'Swift',
    'updated_at': 'Tue, 15 Apr 2025 20:52:50 GMT',
    'uses_two_line_observations': True,
}


def _mock_response(payload: dict) -> MagicMock:
    response = MagicMock(ok=True)
    response.json.return_value = payload
    return response


class TestRepairStaleCampaignRunSites(TestCase):
    """Covers every D-16 dev-DB row shape plus the D-22 network-failure proof and --dry-run."""

    def setUp(self):
        self.campaign = TargetList.objects.create(name='3I/ATLAS')

    def _make_run(self, **overrides) -> CampaignRun:
        defaults = dict(
            campaign=self.campaign,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            site=None,
            site_raw='',
            site_needs_review=True,
            window_start='2025-07-04',
            window_end='2025-07-04',
        )
        defaults.update(overrides)
        return CampaignRun.objects.create(**defaults)

    def test_hst_resolves_via_live_tier2_lookup(self):
        """D-16 HST shape: site_raw='250', tier-2 hit sets a real site, clears the flag."""
        run = self._make_run(telescope_instrument='HST STIS/COS', site_raw='250')

        with patch('requests.get', return_value=_mock_response(_MPC_OBS_DATA_250)):
            call_command('repair_stale_campaign_run_sites')

        run.refresh_from_db()
        self.assertIsNotNone(run.site)
        self.assertEqual(run.site.obscode, '250')
        self.assertFalse(run.site_needs_review)

    def test_swift_gets_owner_supplied_site_raw_then_resolves(self):
        """D-16 Swift shape: blank site_raw becomes 'C52' (D-16b) then resolves via tier 2."""
        run = self._make_run(
            telescope_instrument='Swift/UVOT',
            site_raw='',
            window_start='2025-07-05',
            window_end='2025-07-05',
        )

        with patch('requests.get', return_value=_mock_response(_MPC_OBS_DATA_C52)):
            call_command('repair_stale_campaign_run_sites')

        run.refresh_from_db()
        self.assertEqual(run.site_raw, 'C52')
        self.assertIsNotNone(run.site)
        self.assertEqual(run.site.obscode, 'C52')
        self.assertFalse(run.site_needs_review)

    def test_jwst_resolves_offline_via_alias_no_network_call(self):
        """D-16 JWST shape: site_raw='500@-170' resolves offline through the alias table."""
        Observatory.objects.create(obscode='274', name='James Webb Space Telescope', short_name='JWST')
        run = self._make_run(
            telescope_instrument='JWST/NIRSpec',
            site_raw='500@-170',
            window_start='2025-07-06',
            window_end='2025-07-06',
        )

        with patch('requests.get') as mock_get:
            call_command('repair_stale_campaign_run_sites')
            mock_get.assert_not_called()

        run.refresh_from_db()
        self.assertIsNotNone(run.site)
        self.assertEqual(run.site.obscode, '274')
        self.assertFalse(run.site_needs_review)

    def test_juice_stays_site_less_no_site_code(self):
        """D-16 JUICE shape: blank site_raw, no owner correction -- skipped, untouched, no network call."""
        run = self._make_run(
            telescope_instrument='JUICE',
            site_raw='',
            window_start=None,
            window_end=None,
            contact_person='Juice Requester',
        )

        with patch('requests.get') as mock_get:
            call_command('repair_stale_campaign_run_sites')
            mock_get.assert_not_called()

        run.refresh_from_db()
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)  # unchanged from creation -- row was never touched

    def test_tier2_network_failure_leaves_row_flagged_no_placeholder(self):
        """D-22 proof: RequestException leaves site=None, site_needs_review=True, zero new Observatory rows."""
        run = self._make_run(
            telescope_instrument='HST STIS/COS',
            site_raw='250',
            window_start='2025-07-07',
            window_end='2025-07-07',
        )
        before_count = Observatory.objects.count()

        with patch('requests.get', side_effect=requests.exceptions.RequestException('network is down')):
            call_command('repair_stale_campaign_run_sites')

        run.refresh_from_db()
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)
        self.assertEqual(Observatory.objects.count(), before_count)

    def test_dry_run_writes_nothing(self):
        """--dry-run performs no CampaignRun save and creates no Observatory row."""
        run = self._make_run(
            telescope_instrument='HST STIS/COS',
            site_raw='250',
            window_start='2025-07-08',
            window_end='2025-07-08',
        )
        before_obs_count = Observatory.objects.count()

        with patch('requests.get') as mock_get:
            call_command('repair_stale_campaign_run_sites', '--dry-run')
            mock_get.assert_not_called()

        run.refresh_from_db()
        self.assertIsNone(run.site)
        self.assertEqual(run.site_raw, '250')
        self.assertEqual(Observatory.objects.count(), before_obs_count)

    def test_class_carrying_row_skipped_entirely(self):
        """260730-jty/D-06: a candidate row that already carries a telescope_class is
        permanently site-less by design (the class IS the answer to "why is there no
        site") -- there is no site to repair, so it is skipped entirely: site, site_raw,
        and site_needs_review are all left untouched, and it is reported under its own
        skipped_class_wide counter rather than resolved/still_flagged/skipped_no_site_code.
        """
        run = self._make_run(
            telescope_instrument='JUICE',
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.SPACE,
            window_start=None,
            window_end=None,
            contact_person='Juice Requester',
        )

        stdout_buf = io.StringIO()
        with patch('requests.get') as mock_get:
            call_command('repair_stale_campaign_run_sites', stdout=stdout_buf)
            mock_get.assert_not_called()

        run.refresh_from_db()
        self.assertIsNone(run.site)
        self.assertEqual(run.site_raw, '')
        self.assertTrue(run.site_needs_review)  # untouched -- migration 0012 clears this, not this command
        self.assertEqual(run.telescope_class, CampaignRun.TelescopeClass.SPACE)
        self.assertIn('skipped_class_wide: 1', stdout_buf.getvalue())

    def test_rejected_row_untouched(self):
        """D-15: a rejected row with a resolvable site_raw is never touched (not in scope)."""
        Observatory.objects.create(obscode='X05', name='Simonyi Survey Telescope', short_name='Rubin')
        run = self._make_run(
            telescope_instrument='Rubin Observatory',
            site_raw='X05',
            approval_status=CampaignRun.ApprovalStatus.REJECTED,
            window_start='2025-07-09',
            window_end='2025-07-09',
        )

        with patch('requests.get') as mock_get:
            call_command('repair_stale_campaign_run_sites')
            mock_get.assert_not_called()

        run.refresh_from_db()
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)
