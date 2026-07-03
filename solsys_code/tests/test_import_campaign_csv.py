import csv
import io
import pathlib
import tempfile
from datetime import date, datetime
from datetime import timezone as dt_timezone
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.campaign_utils import (
    insert_or_create_campaign_run,
    map_observation_status,
    parse_obs_window,
    resolve_site,
)
from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory

# Full 14-column header set, exact order/spelling verified against the real 3I/ATLAS
# sheet export (RESEARCH.md "Real 3I/ATLAS Sheet -- Verified Shape").
_HEADERS = [
    'Contact Person',
    'Email',
    'Telescope / Instrument',
    'Site Code',
    'Obs. Date',
    'UT Time Range',
    'Filter(s)/Bandpass',
    'Observation Details',
    'Weather conditions or forecast',
    'Observation Status',
    'Observation Outcome',
    'Publication Plans',
    'Open to collaboration?',
    'Other comments',
]

# Same shape as solsys_code_observatory/tests/test_utils.py's obs_data fixture --
# every key MPCObscodeFetcher.to_observatory() reads.
_MPC_OBS_DATA_E10 = {
    'created_at': 'Sat, 25 May 2019 00:11:26 GMT',
    'longitude': '149.07085',
    'name_utf8': 'Siding Spring-Faulkes Telescope South',
    'obscode': 'E10',
    'observations_type': 'optical',
    'old_names': None,
    'rhocosphi': '0.855632',
    'rhosinphi': '-0.516198',
    'short_name': 'Siding Spring-Faulkes Telescope South',
    'updated_at': 'Tue, 15 Apr 2025 20:52:50 GMT',
    'uses_two_line_observations': False,
}


def _row(**overrides):
    """Build one full-header CSV row dict, blank by default, with the given overrides."""
    row = dict.fromkeys(_HEADERS, '')
    row.update(overrides)
    return row


class _WriteCsvMixin:
    def _write_csv(self, rows: list[dict]) -> tuple[str, tempfile.TemporaryDirectory]:
        """Write a campaign CSV to a temporary directory and return (path, tmpdir_ctx).

        The caller must use tmpdir_ctx as a context manager to ensure cleanup:

            path, tmpdir_ctx = self._write_csv([...])
            with tmpdir_ctx:
                call_command(...)
        """
        tmpdir_ctx = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmpdir_ctx.name) / 'campaign.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        return str(path), tmpdir_ctx


class TestCampaignUtils(TestCase):
    """Pure-helper edge cases for campaign_utils.py (Task 1's behavior block)."""

    def test_resolve_site_blank_returns_none_needs_review(self):
        site, needs_review = resolve_site('')
        self.assertIsNone(site)
        self.assertTrue(needs_review)
        self.assertEqual(Observatory.objects.count(), 0)

    def test_resolve_site_oversized_returns_none_needs_review(self):
        # JWST's spacecraft-style designation -- 8 chars, exceeds obscode's max_length=4.
        site, needs_review = resolve_site('500@-170')
        self.assertIsNone(site)
        self.assertTrue(needs_review)
        self.assertEqual(Observatory.objects.count(), 0)

    def test_resolve_site_existing_observatory_hit(self):
        obs = Observatory.objects.create(
            obscode='F65', name='FTN', short_name='FTN', lat=20.7, lon=-156.3, altitude=3055
        )
        site, needs_review = resolve_site('F65')
        self.assertEqual(site, obs)
        self.assertFalse(needs_review)

    @patch('requests.get')
    def test_resolve_site_mpc_miss_creates_placeholder(self, mock_get):
        mock_response = MagicMock(ok=False, status_code=501)
        mock_response.json.return_value = {'error': 'input_error', 'message': "obscodes failed: No obscode 'Z99'"}
        mock_get.return_value = mock_response

        site, needs_review = resolve_site('Z99')

        self.assertIsNotNone(site)
        self.assertEqual(site.obscode, 'Z99')
        self.assertIn('NEEDS REVIEW', site.name)
        self.assertTrue(needs_review)

    @patch('requests.get')
    def test_resolve_site_mpc_hit_creates_observatory(self, mock_get):
        mock_response = MagicMock(ok=True)
        mock_response.json.return_value = _MPC_OBS_DATA_E10
        mock_get.return_value = mock_response

        site, needs_review = resolve_site('E10')

        self.assertEqual(site.obscode, 'E10')
        self.assertFalse(needs_review)

    def test_parse_obs_window_hhmm_range(self):
        obs_date, start, end = parse_obs_window('2025-07-04', '08:50 - 11:50')
        self.assertEqual(obs_date, date(2025, 7, 4))
        self.assertEqual(start, datetime(2025, 7, 4, 8, 50, tzinfo=dt_timezone.utc))
        self.assertEqual(end, datetime(2025, 7, 4, 11, 50, tzinfo=dt_timezone.utc))

    def test_parse_obs_window_semicolon_typo(self):
        obs_date, start, end = parse_obs_window('2025-07-06', '17:45 - 18;55')
        self.assertEqual(obs_date, date(2025, 7, 6))
        self.assertEqual(start, datetime(2025, 7, 6, 17, 45, tzinfo=dt_timezone.utc))
        self.assertEqual(end, datetime(2025, 7, 6, 18, 55, tzinfo=dt_timezone.utc))

    def test_parse_obs_window_approximate_hour(self):
        obs_date, start, end = parse_obs_window('2025-07-03', '~1 am')
        self.assertEqual(obs_date, date(2025, 7, 3))
        self.assertEqual(start, datetime(2025, 7, 3, 1, 0, tzinfo=dt_timezone.utc))
        self.assertIsNone(end)

    def test_parse_obs_window_blank_time_falls_back_to_midnight(self):
        obs_date, start, end = parse_obs_window('2025-07-06', '')
        self.assertEqual(obs_date, date(2025, 7, 6))
        self.assertEqual(start, datetime(2025, 7, 6, 0, 0, tzinfo=dt_timezone.utc))
        self.assertIsNone(end)

    def test_parse_obs_window_unparseable_date_raises(self):
        with self.assertRaises(ValueError):
            parse_obs_window('', '08:50 - 11:50')

    def test_map_observation_status_completed(self):
        self.assertEqual(map_observation_status('completed'), CampaignRun.RunStatus.OBSERVED)

    def test_map_observation_status_upcoming(self):
        self.assertEqual(map_observation_status('Upcoming'), CampaignRun.RunStatus.PLANNED)

    def test_map_observation_status_unknown_defaults_requested(self):
        self.assertEqual(map_observation_status('???'), CampaignRun.RunStatus.REQUESTED)

    def test_insert_or_create_campaign_run_unchanged_on_second_call(self):
        campaign = TargetList.objects.create(name='Test Campaign')
        lookup = {
            'campaign': campaign,
            'telescope_instrument': 'FTN',
            'ut_start': datetime(2025, 7, 4, 8, 50, tzinfo=dt_timezone.utc),
        }
        fields = {'obs_date': date(2025, 7, 4), 'run_status': CampaignRun.RunStatus.OBSERVED}

        run1, action1 = insert_or_create_campaign_run(lookup, fields)
        self.assertEqual(action1, 'created')

        run2, action2 = insert_or_create_campaign_run(lookup, fields)
        self.assertEqual(action2, 'unchanged')
        self.assertEqual(run1.pk, run2.pk)


class TestImportCampaignCsv(_WriteCsvMixin, TestCase):
    """Integration tests for the import_campaign_csv management command."""

    def test_creates_campaignrun_with_existing_observatory(self):
        Observatory.objects.create(obscode='F65', name='FTN', short_name='FTN', lat=20.7, lon=-156.3, altitude=3055)
        path, ctx = self._write_csv(
            [
                _row(
                    **{
                        'Telescope / Instrument': 'FTN/MuSCAT3',
                        'Site Code': 'F65',
                        'Obs. Date': '2025-07-04',
                        'UT Time Range': '08:50 - 11:50',
                        'Observation Status': 'completed',
                    }
                )
            ]
        )
        with ctx:
            call_command(
                'import_campaign_csv', '--campaign', 'Test Campaign', path, stdout=io.StringIO(), stderr=io.StringIO()
            )

        self.assertEqual(CampaignRun.objects.count(), 1)
        run = CampaignRun.objects.first()
        self.assertEqual(run.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(run.site.obscode, 'F65')
        self.assertFalse(run.site_needs_review)
        self.assertEqual(run.run_status, CampaignRun.RunStatus.OBSERVED)

    def test_auto_resolves_single_target_campaign(self):
        """D-07/CAMP-02: a single-Target campaign auto-assigns that Target to every imported row."""
        campaign = TargetList.objects.create(name='Single Target Campaign')
        target = NonSiderealTargetFactory.create()
        campaign.targets.add(target)

        path, ctx = self._write_csv(
            [
                _row(
                    **{
                        'Telescope / Instrument': 'FTN/MuSCAT3',
                        'Obs. Date': '2025-07-04',
                        'UT Time Range': '08:50 - 11:50',
                    }
                )
            ]
        )
        with ctx:
            call_command(
                'import_campaign_csv',
                '--campaign',
                'Single Target Campaign',
                path,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )

        run = CampaignRun.objects.first()
        self.assertEqual(run.target, target)

    @patch('requests.get')
    def test_tier2_mpc_lookup_creates_observatory(self, mock_get):
        mock_response = MagicMock(ok=True)
        mock_response.json.return_value = _MPC_OBS_DATA_E10
        mock_get.return_value = mock_response

        path, ctx = self._write_csv(
            [
                _row(
                    **{
                        'Telescope / Instrument': 'FTS test',
                        'Site Code': 'E10',
                        'Obs. Date': '2025-07-04',
                        'UT Time Range': '08:50 - 11:50',
                    }
                )
            ]
        )
        with ctx:
            call_command(
                'import_campaign_csv', '--campaign', 'Test Campaign', path, stdout=io.StringIO(), stderr=io.StringIO()
            )

        mock_get.assert_called_once()
        self.assertTrue(Observatory.objects.filter(obscode='E10').exists())
        run = CampaignRun.objects.first()
        self.assertFalse(run.site_needs_review)

    def test_unresolvable_site_flags_needs_review_without_skipping_row(self):
        """D-09: JWST's oversized Site Code doesn't skip the row -- just flags it."""
        path, ctx = self._write_csv(
            [
                _row(
                    **{
                        'Telescope / Instrument': 'JWST',
                        'Site Code': '500@-170',
                        'Obs. Date': '2025-08-06',
                        'UT Time Range': '11:01 - 11:20',
                    }
                )
            ]
        )
        with ctx:
            call_command(
                'import_campaign_csv', '--campaign', 'Test Campaign', path, stdout=io.StringIO(), stderr=io.StringIO()
            )

        self.assertEqual(CampaignRun.objects.count(), 1)
        run = CampaignRun.objects.first()
        self.assertIsNone(run.site)
        self.assertTrue(run.site_needs_review)
        self.assertEqual(run.site_raw, '500@-170')

    def test_natural_key_failure_skipped_and_logged(self):
        """D-05: a bad Obs. Date row is skipped and logged; a good sibling row still imports."""
        Observatory.objects.create(obscode='F65', name='FTN', short_name='FTN', lat=20.7, lon=-156.3, altitude=3055)
        path, ctx = self._write_csv(
            [
                _row(**{'Telescope / Instrument': 'JUICE', 'Obs. Date': '2025-11-02 -25'}),  # malformed date
                _row(
                    **{
                        'Telescope / Instrument': 'FTN/MuSCAT3',
                        'Site Code': 'F65',
                        'Obs. Date': '2025-07-04',
                        'UT Time Range': '08:50 - 11:50',
                    }
                ),
            ]
        )
        with ctx:
            stderr_buf = io.StringIO()
            call_command(
                'import_campaign_csv', '--campaign', 'Test Campaign', path, stdout=io.StringIO(), stderr=stderr_buf
            )

        self.assertEqual(CampaignRun.objects.count(), 1)
        err = stderr_buf.getvalue()
        self.assertIn('Row 2', err)

    def test_idempotent_rerun_no_duplicates(self):
        """D-04: running the command twice over the same CSV produces no duplicate CampaignRuns."""
        Observatory.objects.create(obscode='F65', name='FTN', short_name='FTN', lat=20.7, lon=-156.3, altitude=3055)
        path, ctx = self._write_csv(
            [
                _row(
                    **{
                        'Telescope / Instrument': 'FTN/MuSCAT3',
                        'Site Code': 'F65',
                        'Obs. Date': '2025-07-04',
                        'UT Time Range': '08:50 - 11:50',
                    }
                )
            ]
        )
        with ctx:
            call_command(
                'import_campaign_csv', '--campaign', 'Test Campaign', path, stdout=io.StringIO(), stderr=io.StringIO()
            )
            first_count = CampaignRun.objects.count()

            stdout2 = io.StringIO()
            call_command(
                'import_campaign_csv', '--campaign', 'Test Campaign', path, stdout=stdout2, stderr=io.StringIO()
            )
            second_count = CampaignRun.objects.count()

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertIn('created: 0', stdout2.getvalue())
