"""Tests for the Phase 36 unattended runner tracer slice (Plan 01, Task 1).

Covers ``run_unattended``'s single ``reconcile`` step end-to-end: locking (whole-run and
per-step), heartbeat bracketing, the D-11 mail-once-per-newly-failing-set rule with
suppression/reminder/recovery, and SCHED-10 credential hygiene. Any ``Target`` fixture
uses ``tom_targets.tests.factories.NonSiderealTargetFactory`` per CLAUDE.md -- this module
does not fixture one directly since ``CampaignRun.target`` is nullable and left unset,
matching ``test_reconcile_campaign_runs.py``'s own convention.
"""

import contextlib
import fcntl
import json
import tempfile
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from io import StringIO
from pathlib import Path
from smtplib import SMTPAuthenticationError
from unittest.mock import patch

import requests
from django.contrib.auth.models import User
from django.core import mail
from django.core.management import call_command
from django.db.models.signals import post_save
from django.test import TestCase, override_settings
from tom_common.exceptions import ImproperCredentialsException
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import notifications, unattended
from solsys_code import observation_projector as op
from solsys_code.models import CampaignRun, WatchedProposal
from solsys_code.solsys_code_observatory.models import Observatory

_FAKE_HEARTBEAT_URL = 'https://hc.example/UUID-TEST'


class UnattendedTestBase(TestCase):
    """Shared fixture: a temp lock/state dir, a fake heartbeat URL, and ``requests.get``
    patched to a no-op mock -- per the plan's own ``<behavior>`` preamble.
    """

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        settings_override = override_settings(
            FOMO_LOCK_DIR=self.tmp_dir.name,
            FOMO_STATE_DIR=self.tmp_dir.name,
            FOMO_HEARTBEAT_URL=_FAKE_HEARTBEAT_URL,
        )
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        requests_get_patcher = patch('solsys_code.unattended.requests.get')
        self.mock_requests_get = requests_get_patcher.start()
        self.addCleanup(requests_get_patcher.stop)

        self.campaign = TargetList.objects.create(name='Test Campaign')
        self.ground_site = Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )

    def _make_campaign_run(self, **overrides) -> CampaignRun:
        kwargs = {
            'campaign': self.campaign,
            'telescope_instrument': 'FTN/MuSCAT3',
            'site': self.ground_site,
            'site_raw': 'F65',
            'window_start': date(2026, 8, 1),
            'window_end': date(2026, 8, 1),
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)


class TestRunUnattended(UnattendedTestBase):
    """The reconcile step end-to-end, plus --dry-run and --step semantics."""

    def test_healthy_tick_exits_zero(self):
        self._make_campaign_run()
        call_command('run_unattended')  # must not raise SystemExit
        self.assertEqual(len(mail.outbox), 0)

    def test_step_failure_sets_exit_code(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with self.assertRaises(SystemExit) as cm:
                call_command('run_unattended')
        self.assertEqual(cm.exception.code, 1)

    def test_step_failure_does_not_abort_the_tick(self):
        calls = []

        def failing_step(dry_run):
            calls.append('a')
            return unattended.StepResult(name='a', failed=True, summary='boom')

        def ok_step(dry_run):
            calls.append('b')
            return unattended.StepResult(name='b', failed=False, summary='fine')

        with patch.object(unattended, 'STEPS', (('a', failing_step), ('b', ok_step))):
            with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
                with self.assertRaises(SystemExit):
                    call_command('run_unattended')

        self.assertEqual(calls, ['a', 'b'])
        joined = '\n'.join(captured.output)
        self.assertIn('step a', joined)
        self.assertIn('step b', joined)
        self.assertLess(joined.index('step a'), joined.index('step b'))

    def test_empty_database_tick_is_healthy(self):
        call_command('run_unattended')
        self.assertEqual(len(mail.outbox), 0)
        urls = [call.args[0] for call in self.mock_requests_get.call_args_list]
        self.assertTrue(any(url.endswith('/start') for url in urls))
        self.assertTrue(any(url.endswith('/0') for url in urls))

    def test_dry_run_neither_pings_nor_mails(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with contextlib.suppress(SystemExit):
                call_command('run_unattended', '--dry-run')
        self.assertEqual(len(mail.outbox), 0)
        self.mock_requests_get.assert_not_called()

    def test_step_flag_runs_one_step_only(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run') as mock_reconcile:
            call_command('run_unattended', '--step', 'reconcile')
        mock_reconcile.assert_called()
        self.assertEqual(len(mail.outbox), 0)
        self.mock_requests_get.assert_not_called()

    def test_all_four_steps_run_in_order(self):
        calls = []

        def make_step(name):
            def step(dry_run):
                calls.append(name)
                return unattended.StepResult(name=name, failed=(name == 'status_refresh'), summary='x')

            return step

        fake_steps = tuple(
            (name, make_step(name)) for name in ('status_refresh', 'project_sweep', 'discovery', 'reconcile')
        )
        with patch.object(unattended, 'STEPS', fake_steps):
            with contextlib.suppress(SystemExit):
                call_command('run_unattended')

        self.assertEqual(calls, ['status_refresh', 'project_sweep', 'discovery', 'reconcile'])

    def test_expected_data_shape_outcomes_are_not_failures(self):
        def sweep_step(dry_run):
            return unattended.StepResult(name='project_sweep', failed=False, summary='LCO: unchanged: 5, skipped: 0')

        def reconcile_step(dry_run):
            return unattended.StepResult(
                name='reconcile', failed=False, summary='detach_declined: 2, remint_declined: 1'
            )

        with patch.object(unattended, 'STEPS', (('project_sweep', sweep_step), ('reconcile', reconcile_step))):
            call_command('run_unattended')  # must not raise
        self.assertEqual(len(mail.outbox), 0)


class TestNotification(UnattendedTestBase):
    """D-11: mail once per newly-failing tick, with suppression, reminder, and recovery."""

    def setUp(self):
        super().setUp()
        self.staff_with_email = User.objects.create_user(
            username='staff-with-email', email='staff@example.org', is_staff=True
        )
        User.objects.create_user(username='staff-no-email', email='', is_staff=True)
        User.objects.create_user(username='non-staff', email='nonstaff@example.org', is_staff=False)

    def test_failing_tick_mails_staff_once(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, [self.staff_with_email.email])
        self.assertTrue(mail.outbox[0].subject.startswith('FOMO unattended run failed:'))
        self.assertIn('reconcile', mail.outbox[0].subject)

    def test_repeat_failure_is_suppressed(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
        self.assertEqual(len(mail.outbox), 1)

    def test_reminder_after_interval(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
            old_time = datetime.now(dt_timezone.utc) - timedelta(hours=25)
            unattended.save_state(['reconcile'], old_time)
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
        self.assertEqual(len(mail.outbox), 2)

    def test_recovery_mails_once(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
        call_command('run_unattended')  # now healthy
        self.assertEqual(len(mail.outbox), 2)
        self.assertEqual(mail.outbox[1].subject, 'FOMO unattended run recovered')
        call_command('run_unattended')  # still healthy
        self.assertEqual(len(mail.outbox), 2)

    def test_mail_failure_never_raises(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with patch(
                'solsys_code.notifications.send_mail',
                side_effect=SMTPAuthenticationError(535, b'user=svc pass=hunter2'),
            ):
                with self.assertLogs('solsys_code', level='INFO') as captured:
                    stdout = StringIO()
                    stderr = StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as cm:
                            call_command('run_unattended')
        self.assertEqual(cm.exception.code, 1)
        joined = '\n'.join(captured.output)
        self.assertNotIn('hunter2', joined)
        self.assertNotIn('hunter2', stdout.getvalue())
        self.assertNotIn('hunter2', stderr.getvalue())

    def test_failed_send_does_not_save_state(self):
        # WR-02 (36-REVIEW.md): a failed/unsent notification must never be recorded as
        # "staff were notified" -- otherwise a down mail relay on the first failing tick
        # suppresses every subsequent notification for the same failing set for 24 hours.
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with patch(
                'solsys_code.notifications.send_mail',
                side_effect=SMTPAuthenticationError(535, b'user=svc pass=hunter2'),
            ):
                with self.assertRaises(SystemExit):
                    call_command('run_unattended')
            self.assertEqual(len(mail.outbox), 0)
            state = unattended.load_state()
            self.assertEqual(state['failing_steps'], [])
            self.assertIsNone(state['notified_at'])

            # A later tick, with mail working again, must still mail staff -- the failed
            # send above must not have suppressed it.
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
        self.assertEqual(len(mail.outbox), 1)


class TestNotifyStaffReturnValue(UnattendedTestBase):
    """IN-07 (36-REVIEW.md): ``notify_staff()`` must return whether ``send_mail()``
    actually sent a message, not just whether a recipient existed and nothing raised --
    ``unattended._send_notification()``'s docstring promises exactly that stronger
    claim, and WR-02's whole suppression decision rests on it being true."""

    def setUp(self):
        super().setUp()
        User.objects.create_user(username='staff-with-email', email='staff@example.org', is_staff=True)

    def test_returns_true_when_a_message_is_actually_sent(self):
        self.assertTrue(notifications.notify_staff('subject', 'body'))
        self.assertEqual(len(mail.outbox), 1)

    def test_returns_false_when_send_mail_reports_zero_sent(self):
        with patch('solsys_code.notifications.send_mail', return_value=0):
            self.assertFalse(notifications.notify_staff('subject', 'body'))

    def test_returns_false_when_fail_silently_suppresses_a_raised_exception(self):
        with patch('solsys_code.notifications.send_mail', side_effect=RuntimeError('smtp outage')):
            self.assertFalse(notifications.notify_staff('subject', 'body', fail_silently=True))


class TestHeartbeat(UnattendedTestBase):
    """D-12: one whole-tick heartbeat, /start then /<exit-code>."""

    def test_pings_start_then_exit_code(self):
        self._make_campaign_run()
        call_command('run_unattended')
        urls = [call.args[0] for call in self.mock_requests_get.call_args_list]
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].endswith('/start'))
        self.assertTrue(urls[1].endswith('/0'))

        self.mock_requests_get.reset_mock()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
        urls = [call.args[0] for call in self.mock_requests_get.call_args_list]
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].endswith('/start'))
        self.assertTrue(urls[1].endswith('/1'))

    def test_ping_failure_never_fails_the_tick(self):
        self._make_campaign_run()
        self.mock_requests_get.side_effect = requests.exceptions.ConnectionError('https://hc.example/UUID-SECRET')
        with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
            call_command('run_unattended')  # must not raise
        joined = '\n'.join(captured.output)
        self.assertNotIn('UUID-SECRET', joined)

    def test_unset_url_skips_pinging(self):
        self._make_campaign_run()
        with override_settings(FOMO_HEARTBEAT_URL=None):
            call_command('run_unattended')  # must not raise
        self.mock_requests_get.assert_not_called()


class TestLocking(UnattendedTestBase):
    """One lock per command (run_unattended, and reconcile_campaign_runs underneath it)."""

    def test_contended_lock_skips_every_step(self):
        self._make_campaign_run()
        lock_dir = Path(self.tmp_dir.name)
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / 'run_unattended.lock'
        fh = lock_path.open('a+')
        self.addCleanup(fh.close)
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch('solsys_code.unattended.reconcile_run') as mock_reconcile:
                stderr = StringIO()
                with contextlib.redirect_stderr(stderr):
                    call_command('run_unattended')  # must not raise
            mock_reconcile.assert_not_called()
            self.assertIn('lock', stderr.getvalue().lower())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
        self.assertEqual(len(mail.outbox), 0)


class TestEndBannerTimestamp(UnattendedTestBase):
    """WR-04 (36-REVIEW.md): the END banner must report a freshly-sampled time, not the
    tick's START time -- otherwise no tick's duration is ever readable in the log."""

    def test_end_banner_timestamp_differs_from_start_banner_timestamp(self):
        self._make_campaign_run()
        start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt_timezone.utc)
        end_time = datetime(2026, 1, 1, 0, 15, 0, tzinfo=dt_timezone.utc)

        class _FakeDateTime(datetime):
            _values = iter([start_time, end_time])

            @classmethod
            def now(cls, tz=None):
                return next(cls._values)

        with patch('solsys_code.unattended.datetime', _FakeDateTime):
            with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
                call_command('run_unattended')

        joined = '\n'.join(captured.output)
        self.assertIn(f'START {start_time.isoformat()}', joined)
        self.assertIn(f'END {end_time.isoformat()}', joined)
        self.assertNotIn(f'END {start_time.isoformat()}', joined)


class TestStateFileRobustness(UnattendedTestBase):
    """WR-03 (36-REVIEW.md): the suppression-state file must be fail-safe -- a
    malformed file must never raise out of ``load_state()``, and a state-handling
    failure must never stop the tick's own END banner or exit-code heartbeat ping."""

    def _write_state_file(self, content: str) -> None:
        state_path = Path(self.tmp_dir.name) / 'unattended-state.json'
        state_path.write_text(content)

    def test_non_dict_state_file_is_treated_as_no_prior_failure(self):
        self._write_state_file('[1, 2]')
        self.assertEqual(unattended.load_state(), {'failing_steps': [], 'notified_at': None})

    def test_malformed_notified_at_is_treated_as_no_prior_notification(self):
        self._write_state_file(json.dumps({'failing_steps': ['reconcile'], 'notified_at': 'not-a-date'}))
        state = unattended.load_state()
        self.assertEqual(state['failing_steps'], ['reconcile'])
        self.assertIsNone(state['notified_at'])

    def test_naive_notified_at_is_assumed_utc(self):
        self._write_state_file(json.dumps({'failing_steps': ['reconcile'], 'notified_at': '2026-01-01T00:00:00'}))
        state = unattended.load_state()
        self.assertIsNotNone(state['notified_at'].tzinfo)

    def test_mixed_type_failing_steps_are_coerced_not_raised(self):
        # WR-10 (36-REVIEW.md): a hand-edited or foreign-written state file can carry a
        # 'failing_steps' list with non-string elements -- sorted() must never be asked
        # to compare a str to an int/bool, which raised TypeError before this fix and
        # skipped save_state() for the rest of the tick's life (every subsequent tick
        # repeated the same crash and never repaired the file).
        self._write_state_file(json.dumps({'failing_steps': [1, 'a'], 'notified_at': None}))
        self.assertEqual(unattended.load_state(), {'failing_steps': ['a'], 'notified_at': None})

        self._write_state_file(json.dumps({'failing_steps': [True, 'a']}))
        self.assertEqual(unattended.load_state(), {'failing_steps': ['a'], 'notified_at': None})

    def test_state_handling_failure_never_blocks_the_end_banner_or_heartbeat(self):
        self._make_campaign_run()
        with (
            patch('solsys_code.unattended.save_state', side_effect=OSError('disk full')),
            patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')),
            self.assertLogs('solsys_code.unattended', level='INFO') as captured,
        ):
            with self.assertRaises(SystemExit) as cm:
                call_command('run_unattended')
        self.assertEqual(cm.exception.code, 1)
        joined = '\n'.join(captured.output)
        self.assertIn('=== FOMO unattended run END', joined)
        urls = [call.args[0] for call in self.mock_requests_get.call_args_list]
        self.assertTrue(any(url.endswith('/1') for url in urls))


class TestNoneSettingGuards(UnattendedTestBase):
    """IN-14 (36-REVIEW.md): a local_settings.py deriving FOMO_LOCK_DIR/FOMO_STATE_DIR
    from an unset environment variable with no default of its own yields None -- the same
    WR-07 hazard FOMO_BASE_URL already guards against -- and Path(None) must never raise
    out of the very first thing run_tick() does."""

    def test_none_lock_dir_falls_back_to_the_documented_default(self):
        # Patch the fallback constant itself to a writable temp dir rather than letting
        # the real /var/lock/fomo default fire -- this test only needs to prove
        # Path(None) never raises, not that this sandbox can write to a real system path.
        fallback_dir = tempfile.TemporaryDirectory()
        self.addCleanup(fallback_dir.cleanup)
        with (
            override_settings(FOMO_LOCK_DIR=None),
            patch('solsys_code.unattended._DEFAULT_LOCK_DIR', fallback_dir.name),
        ):
            with unattended.command_lock('none-lock-dir-guard-test'):
                pass  # must not raise TypeError from Path(None)

    def test_none_state_dir_falls_back_to_lock_dir(self):
        with override_settings(FOMO_STATE_DIR=None):
            # Must not raise -- falls back to FOMO_LOCK_DIR (still the writable temp dir
            # UnattendedTestBase sets up), not the hardcoded /var/lock/fomo default.
            self.assertEqual(unattended.load_state(), {'failing_steps': [], 'notified_at': None})
            unattended.save_state(['reconcile'], None)
            state_path = Path(self.tmp_dir.name) / 'unattended-state.json'
            self.assertTrue(state_path.exists())


class TestStatusRefreshStep(UnattendedTestBase):
    """Task 1 (D-03): the FOMO-owned LCO/SOAR status refresh step."""

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_calls_both_facilities_with_fresh_instances(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = []
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        unattended.step_status_refresh(dry_run=False)

        mock_lco_cls.assert_called_once_with()
        mock_soar_cls.assert_called_once_with()
        mock_lco_cls.return_value.update_all_observation_statuses.assert_called_once_with()
        mock_soar_cls.return_value.update_all_observation_statuses.assert_called_once_with()
        self.assertIsNot(mock_lco_cls.return_value, mock_soar_cls.return_value)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_clean_refresh_is_not_a_failure(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = []
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertFalse(result.failed)
        self.assertIn('LCO: failed 0', result.summary)
        self.assertIn('SOAR: failed 0', result.summary)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_non_empty_failure_list_is_a_step_failure(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', 'boom')]
        mock_lco_cls.return_value.update_observation_status.return_value = None
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertTrue(result.failed)
        self.assertIn('LCO: failed 1', result.summary)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_failure_is_reported_by_class_name_not_message(self, mock_lco_cls, mock_soar_cls):
        fake_key = 'FAKE-API-KEY-CREDHYG-STATUS-1'
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [
            ('obs-1', f'portal error key={fake_key}')
        ]
        mock_lco_cls.return_value.update_observation_status.side_effect = requests.exceptions.HTTPError(
            f'portal error key={fake_key}'
        )
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        with self.assertLogs('solsys_code.unattended', level='WARNING') as captured:
            result = unattended.step_status_refresh(dry_run=False)

        joined = '\n'.join(captured.output)
        self.assertIn('obs-1', joined)
        self.assertIn('HTTPError', joined)
        self.assertNotIn(fake_key, joined)
        self.assertTrue(result.failed)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_transient_failure_still_counts(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', 'boom')]
        mock_lco_cls.return_value.update_observation_status.return_value = None
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        with self.assertLogs('solsys_code.unattended', level='WARNING') as captured:
            result = unattended.step_status_refresh(dry_run=False)

        self.assertTrue(result.failed)
        self.assertIn('LCO: failed 1', result.summary)
        joined = '\n'.join(captured.output)
        self.assertIn('no exception on re-check', joined)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_facility_exception_is_isolated_per_facility(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.side_effect = RuntimeError('lco down')
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        mock_soar_cls.return_value.update_all_observation_statuses.assert_called_once_with()
        self.assertTrue(result.failed)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_dry_run_makes_no_facility_call(self, mock_lco_cls, mock_soar_cls):
        result = unattended.step_status_refresh(dry_run=True)

        mock_lco_cls.assert_not_called()
        mock_soar_cls.assert_not_called()
        self.assertFalse(result.failed)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_whole_facility_outage_caps_the_recheck_calls(self, mock_lco_cls, mock_soar_cls):
        # WR-08 (36-REVIEW.md): a whole-facility outage fails every non-terminal record,
        # so the re-check must be capped -- not one portal call per failed record with
        # no bound, no backoff, and no per-tick time budget.
        many_failures = [(f'obs-{i}', 'boom') for i in range(unattended._MAX_STATUS_RECHECKS + 5)]
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = many_failures
        mock_lco_cls.return_value.update_observation_status.return_value = None
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertEqual(
            mock_lco_cls.return_value.update_observation_status.call_count, unattended._MAX_STATUS_RECHECKS
        )
        self.assertTrue(result.failed)
        self.assertIn(f'LCO: failed {len(many_failures)}', result.summary)
        self.assertIn('recheck capped: 5 omitted', result.summary)

    @patch('solsys_code.unattended.SOARFacility')
    @patch('solsys_code.unattended.LCOFacility')
    def test_under_cap_failure_count_omits_the_capped_note(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', 'boom')]
        mock_lco_cls.return_value.update_observation_status.return_value = None
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertNotIn('recheck capped', result.summary)


class TestProjectSweepStep(UnattendedTestBase):
    """Task 2: the projector sweep step, reproducing project_observation_calendar's own
    logic directly rather than going through the management command."""

    @classmethod
    def setUpTestData(cls):
        cls.target = NonSiderealTargetFactory.create()

    def _make_record(self, observation_id: str, facility: str = 'LCO') -> ObservationRecord:
        """Create an ObservationRecord fixture with the projector's post_save receiver
        disconnected -- mirroring test_project_observation_calendar.py's own convention --
        so fixture creation does not itself pre-populate the CalendarEvent this step's
        sweep is meant to write.
        """
        post_save.disconnect(
            op.receiver_on_record_save,
            sender=ObservationRecord,
            dispatch_uid='solsys_code.observation_projector.post_save',
        )
        try:
            return ObservationRecord.objects.create(
                target=self.target,
                facility=facility,
                observation_id=observation_id,
                status='PENDING',
                parameters={
                    'proposal': 'TESTPROP',
                    'instrument_type': '2M0-SCICAM-MUSCAT',
                    'start': '2026-09-01T00:00:00',
                    'end': '2026-09-02T00:00:00',
                },
            )
        finally:
            post_save.connect(
                op.receiver_on_record_save,
                sender=ObservationRecord,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.post_save',
            )

    def test_clean_sweep_is_not_a_failure(self):
        self._make_record('sweep-clean')

        result = unattended.step_project_sweep(dry_run=False)

        self.assertFalse(result.failed)
        self.assertIn('LCO', result.summary)

    @patch('solsys_code.unattended.project_queryset')
    def test_unprojectable_row_is_a_failure(self, mock_project_queryset):
        mock_project_queryset.return_value = {
            'counters': {
                'LCO': {
                    'created': 0,
                    'updated': 0,
                    'unchanged': 0,
                    'unprojectable': 1,
                    'site_lookups': 0,
                    'site_lookup_failed': 0,
                }
            },
            'rows': [
                {'observation_id': 'obs-x', 'status': 'PENDING', 'stage': 'ValueError', 'action': 'unprojectable'}
            ],
        }

        result = unattended.step_project_sweep(dry_run=False)

        self.assertTrue(result.failed)
        self.assertIn('failed: 1', result.summary)

    @patch('solsys_code.unattended.project_queryset')
    def test_site_lookup_hook_is_passed_on_a_real_run_and_omitted_on_dry_run(self, mock_project_queryset):
        mock_project_queryset.return_value = {'counters': {}, 'rows': []}

        unattended.step_project_sweep(dry_run=False)
        self.assertTrue(callable(mock_project_queryset.call_args.kwargs['pre_fields_hook']))

        mock_project_queryset.reset_mock()
        unattended.step_project_sweep(dry_run=True)
        self.assertIsNone(mock_project_queryset.call_args.kwargs['pre_fields_hook'])

    @patch('django.core.management.call_command')
    def test_step_never_calls_the_management_command(self, mock_call_command):
        self._make_record('sweep-no-command')

        unattended.step_project_sweep(dry_run=False)

        mock_call_command.assert_not_called()


class TestDiscoveryStep(UnattendedTestBase):
    """Task 2 (36-CONTEXT.md D-07..D-09): the watched-proposal discovery step."""

    @patch('solsys_code.management.commands.backfill_lco_observations.sweep_proposal')
    def test_sweeps_every_active_row(self, mock_sweep_proposal):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        WatchedProposal.objects.create(proposal_code='CCC-2026-003', is_active=False)
        mock_sweep_proposal.return_value = 'swept ok'

        result = unattended.step_discovery(dry_run=False)

        self.assertEqual(
            [call.args[0] for call in mock_sweep_proposal.call_args_list],
            ['AAA-2026-001', 'BBB-2026-002'],
        )
        self.assertFalse(result.failed)

    @patch('solsys_code.management.commands.backfill_lco_observations.sweep_proposal')
    def test_empty_list_is_healthy(self, mock_sweep_proposal):
        result = unattended.step_discovery(dry_run=False)

        self.assertFalse(result.failed)
        self.assertIn('0 watched proposals', result.summary)
        mock_sweep_proposal.assert_not_called()

    @patch('solsys_code.management.commands.backfill_lco_observations.sweep_proposal')
    def test_one_failing_row_does_not_stop_the_others(self, mock_sweep_proposal):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        mock_sweep_proposal.side_effect = [requests.exceptions.HTTPError('boom'), 'requestgroups seen: 1']

        result = unattended.step_discovery(dry_run=False)

        row_a = WatchedProposal.objects.get(proposal_code='AAA-2026-001')
        row_b = WatchedProposal.objects.get(proposal_code='BBB-2026-002')
        self.assertTrue(row_a.last_run_summary.startswith('failed:'))
        self.assertEqual(row_b.last_run_summary, 'requestgroups seen: 1')
        self.assertTrue(result.failed)

    @patch('solsys_code.management.commands.backfill_lco_observations.sweep_proposal')
    def test_failing_row_names_the_proposal_in_the_summary(self, mock_sweep_proposal):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        mock_sweep_proposal.side_effect = requests.exceptions.HTTPError('boom')

        result = unattended.step_discovery(dry_run=False)

        self.assertIn('AAA-2026-001', result.summary)

    @patch('solsys_code.management.commands.backfill_lco_observations.sweep_proposal')
    def test_dry_run_writes_no_bookkeeping(self, mock_sweep_proposal):
        row = WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        mock_sweep_proposal.return_value = 'would sweep'

        unattended.step_discovery(dry_run=True)

        row.refresh_from_db()
        self.assertIsNone(row.last_run_at)
        self.assertEqual(row.last_run_summary, '')


_FAKE_LCO_API_KEY = 'FAKE-API-KEY-DO-NOT-LOG-a1b2c3'
_FAKE_MAIL_PASSWORD = 'FAKE-MAIL-PW-d4e5f6'
_FAKE_HEARTBEAT_PING_URL = 'https://hc.example/FAKE-PING-UUID-g7h8i9'


class TestCredentialHygiene(UnattendedTestBase):
    """SCHED-10/D-16/D-17: no seeded credential leaks into any output surface, across
    every forced failure path on the unattended tick (Task 3)."""

    def setUp(self):
        super().setUp()
        from django.conf import settings as django_settings

        self._original_lco_api_key = django_settings.FACILITIES['LCO'].get('api_key')
        self._original_soar_api_key = django_settings.FACILITIES.get('SOAR', {}).get('api_key')
        django_settings.FACILITIES['LCO']['api_key'] = _FAKE_LCO_API_KEY
        if 'SOAR' in django_settings.FACILITIES:
            django_settings.FACILITIES['SOAR']['api_key'] = _FAKE_LCO_API_KEY

        def _restore_facilities():
            django_settings.FACILITIES['LCO']['api_key'] = self._original_lco_api_key
            if 'SOAR' in django_settings.FACILITIES:
                django_settings.FACILITIES['SOAR']['api_key'] = self._original_soar_api_key

        self.addCleanup(_restore_facilities)

        settings_override = override_settings(
            EMAIL_HOST_PASSWORD=_FAKE_MAIL_PASSWORD,
            FOMO_HEARTBEAT_URL=_FAKE_HEARTBEAT_PING_URL,
        )
        settings_override.enable()
        self.addCleanup(settings_override.disable)

        self.staff_with_email = User.objects.create_user(
            username='staff-with-email-credhyg', email='staff-credhyg@example.org', is_staff=True
        )

    def _assert_no_secrets_leaked(self, log_output: list[str], stdout_value: str, stderr_value: str) -> None:
        """Assert none of the three seeded literals appear in the log, stdout, stderr, or
        any sent email's subject/body."""
        joined_logs = '\n'.join(log_output)
        for secret in (_FAKE_LCO_API_KEY, _FAKE_MAIL_PASSWORD, _FAKE_HEARTBEAT_PING_URL):
            self.assertNotIn(secret, joined_logs)
            self.assertNotIn(secret, stdout_value)
            self.assertNotIn(secret, stderr_value)
        for sent in mail.outbox:
            for secret in (_FAKE_LCO_API_KEY, _FAKE_MAIL_PASSWORD, _FAKE_HEARTBEAT_PING_URL):
                self.assertNotIn(secret, sent.subject)
                self.assertNotIn(secret, sent.body)

    def _run_tick_capturing(self) -> tuple[list[str], str, str]:
        """Run one tick through the real ``run_unattended`` command, capturing the log at
        the project's own root level (INFO -- src/fomo/settings.py ``LOGGING``), stdout,
        and stderr. A ``SystemExit`` from a failing tick is expected and swallowed."""
        stdout = StringIO()
        stderr = StringIO()
        with self.assertLogs('solsys_code', level='INFO') as captured:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with contextlib.suppress(SystemExit):
                    call_command('run_unattended')
        return captured.output, stdout.getvalue(), stderr.getvalue()

    def test_status_refresh_portal_error_leaks_nothing(self):
        error_message = (
            f'portal error key={_FAKE_LCO_API_KEY} pass={_FAKE_MAIL_PASSWORD} url={_FAKE_HEARTBEAT_PING_URL}'
        )
        with (
            patch('solsys_code.unattended.LCOFacility') as mock_lco_cls,
            patch('solsys_code.unattended.SOARFacility') as mock_soar_cls,
        ):
            mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', error_message)]
            mock_lco_cls.return_value.update_observation_status.side_effect = requests.exceptions.HTTPError(
                error_message
            )
            mock_soar_cls.return_value.update_all_observation_statuses.return_value = []
            log_output, stdout_value, stderr_value = self._run_tick_capturing()

        self._assert_no_secrets_leaked(log_output, stdout_value, stderr_value)

    def test_project_sweep_site_lookup_error_leaks_nothing(self):
        self._make_record_for_sweep('sweep-credhyg')
        error_message = f'site lookup failed key={_FAKE_LCO_API_KEY} url={_FAKE_HEARTBEAT_PING_URL}'
        with patch(
            'solsys_code.unattended.resolve_observed_site',
            side_effect=requests.exceptions.RequestException(error_message),
        ):
            log_output, stdout_value, stderr_value = self._run_tick_capturing()

        self._assert_no_secrets_leaked(log_output, stdout_value, stderr_value)

    def _make_record_for_sweep(self, observation_id: str):
        target = NonSiderealTargetFactory.create()
        post_save.disconnect(
            op.receiver_on_record_save,
            sender=ObservationRecord,
            dispatch_uid='solsys_code.observation_projector.post_save',
        )
        try:
            return ObservationRecord.objects.create(
                target=target,
                facility='LCO',
                observation_id=observation_id,
                status='PENDING',
                parameters={
                    'proposal': 'TESTPROP',
                    'instrument_type': '2M0-SCICAM-MUSCAT',
                    'start': '2026-09-01T00:00:00',
                    'end': '2026-09-02T00:00:00',
                },
            )
        finally:
            post_save.connect(
                op.receiver_on_record_save,
                sender=ObservationRecord,
                weak=False,
                dispatch_uid='solsys_code.observation_projector.post_save',
            )

    def test_discovery_portal_error_leaks_nothing(self):
        row_a = WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        row_b = WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        error_message = f'portal error key={_FAKE_LCO_API_KEY} url={_FAKE_HEARTBEAT_PING_URL}'
        with patch(
            'solsys_code.management.commands.backfill_lco_observations.sweep_proposal',
            side_effect=[ImproperCredentialsException(error_message), 'requestgroups seen: 1'],
        ):
            log_output, stdout_value, stderr_value = self._run_tick_capturing()

        self._assert_no_secrets_leaked(log_output, stdout_value, stderr_value)
        row_a.refresh_from_db()
        row_b.refresh_from_db()
        for secret in (_FAKE_LCO_API_KEY, _FAKE_MAIL_PASSWORD, _FAKE_HEARTBEAT_PING_URL):
            self.assertNotIn(secret, row_a.last_run_summary)
            self.assertNotIn(secret, row_b.last_run_summary)

    def test_reconcile_error_leaks_nothing(self):
        self._make_campaign_run()
        error_message = f'reconcile error key={_FAKE_LCO_API_KEY} pass={_FAKE_MAIL_PASSWORD}'
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError(error_message)):
            log_output, stdout_value, stderr_value = self._run_tick_capturing()

        # D-17's second bucket: reconcile_run() is FOMO's own call, so its message MAY
        # reach the (DEBUG-only) log -- but DEBUG is below the project's root INFO level
        # (src/fomo/settings.py LOGGING), so it never reaches this capture, stdout,
        # stderr, or the email either way. No credential literal appears anywhere here.
        self._assert_no_secrets_leaked(log_output, stdout_value, stderr_value)

    def test_mail_send_failure_leaks_nothing(self):
        self._make_campaign_run()
        stdout = StringIO()
        stderr = StringIO()
        with (
            patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')),
            patch(
                'solsys_code.notifications.send_mail',
                side_effect=SMTPAuthenticationError(535, f'user=svc pass={_FAKE_MAIL_PASSWORD}'.encode()),
            ),
            self.assertLogs('solsys_code', level='INFO') as captured,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as cm:
                call_command('run_unattended')

        self.assertEqual(cm.exception.code, 1)
        self._assert_no_secrets_leaked(captured.output, stdout.getvalue(), stderr.getvalue())

    def test_heartbeat_ping_failure_leaks_nothing(self):
        self.mock_requests_get.side_effect = requests.exceptions.ConnectionError(_FAKE_HEARTBEAT_PING_URL)

        log_output, stdout_value, stderr_value = self._run_tick_capturing()

        self._assert_no_secrets_leaked(log_output, stdout_value, stderr_value)

    def test_failure_email_body_carries_no_secret_and_no_traceback(self):
        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            self._run_tick_capturing()

        self.assertEqual(len(mail.outbox), 1)
        body = mail.outbox[0].body
        from django.conf import settings as django_settings

        self.assertIn(django_settings.FOMO_LOG_FILE, body)
        self.assertIn('reconcile', body)
        self.assertNotIn('Traceback', body)
        for secret in (_FAKE_LCO_API_KEY, _FAKE_MAIL_PASSWORD, _FAKE_HEARTBEAT_PING_URL):
            self.assertNotIn(secret, body)

    def test_no_seeded_secret_in_any_output(self):
        self._make_campaign_run()
        fake_api_key = 'FAKE-LCO-API-KEY-CREDHYG'
        fake_email_password = 'FAKE-EMAIL-PASSWORD-CREDHYG'
        fake_heartbeat_url = 'https://hc.example/FAKE-HEARTBEAT-CREDHYG'
        error_message = f'portal error key={fake_api_key} pass={fake_email_password} url={fake_heartbeat_url}'

        from django.conf import settings as django_settings

        original_api_key = django_settings.FACILITIES['LCO'].get('api_key')
        django_settings.FACILITIES['LCO']['api_key'] = fake_api_key
        try:
            with override_settings(
                EMAIL_HOST_PASSWORD=fake_email_password,
                FOMO_HEARTBEAT_URL=fake_heartbeat_url,
            ):
                with patch(
                    'solsys_code.unattended.reconcile_run',
                    side_effect=requests.exceptions.HTTPError(error_message),
                ):
                    stdout = StringIO()
                    stderr = StringIO()
                    with self.assertLogs('solsys_code', level='INFO') as captured:
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            with self.assertRaises(SystemExit):
                                call_command('run_unattended')
        finally:
            django_settings.FACILITIES['LCO']['api_key'] = original_api_key

        joined_logs = '\n'.join(captured.output)
        for secret in (fake_api_key, fake_email_password, fake_heartbeat_url):
            self.assertNotIn(secret, joined_logs)
            self.assertNotIn(secret, stdout.getvalue())
            self.assertNotIn(secret, stderr.getvalue())
        for sent in mail.outbox:
            for secret in (fake_api_key, fake_email_password, fake_heartbeat_url):
                self.assertNotIn(secret, sent.subject)
                self.assertNotIn(secret, sent.body)
