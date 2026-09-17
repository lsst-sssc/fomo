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
from django.test import TestCase, override_settings
from tom_targets.models import TargetList

from solsys_code import unattended
from solsys_code.models import CampaignRun
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


class TestCredentialHygiene(UnattendedTestBase):
    """SCHED-10/D-16/D-17: no seeded credential leaks into any output surface."""

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
