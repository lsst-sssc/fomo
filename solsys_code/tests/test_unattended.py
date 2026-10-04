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
import importlib.resources
import importlib.util
import json
import os
import re
import stat
import tempfile
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from io import StringIO
from pathlib import Path
from smtplib import SMTPAuthenticationError
from unittest import skipIf
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

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
from solsys_code.tests.test_backfill_lco_observations import _page_response, _request_group
from solsys_code.tests.test_observation_blocks import portal_side_effect

_FAKE_HEARTBEAT_URL = 'https://hc.example/UUID-TEST'


def _http_error(
    status_code: int,
    *,
    body: bytes = b'',
    url: str = 'https://observe.lco.global/api/observations/',
    headers: dict | None = None,
    reason: str = 'Error',
) -> requests.exceptions.HTTPError:
    """Build a genuine ``requests.exceptions.HTTPError`` by raising it from a real
    ``requests.Response`` -- so its message really embeds the URL, exactly as TOM's
    ``make_request()`` produces it (quick task 260927-eqs, Task 1)."""
    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = url
    response.encoding = 'utf-8'
    response._content = body
    response.headers.update(headers or {})
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        return exc
    raise AssertionError(f'raise_for_status() did not raise for status_code={status_code!r}')


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

    def test_unknown_only_step_raises_instead_of_reporting_a_healthy_no_op(self):
        # IN-06 (36-REVIEW.md): run_unattended's own --step argparse choices already
        # reject an unknown name from the CLI, but run_tick() is a public function any
        # other caller can invoke directly -- before this check, an unrecognized
        # only_step silently matched zero STEPS entries, ran nothing, and returned
        # exit_code=0: a typo'd step name looked exactly like a healthy tick.
        with self.assertRaises(ValueError) as ctx:
            unattended.run_tick(only_step='typo-not-a-real-step')
        self.assertIn('typo-not-a-real-step', str(ctx.exception))
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

    @skipIf(os.geteuid() == 0, 'unwritable-directory tests are meaningless as root')
    def test_unwritable_state_dir_after_setup_still_suppresses_repeat_mail(self):
        # WR-17 (36-REVIEW.md): WR-15's fix only covered the SETUP-time preflight
        # (check_state_dir()) -- a state directory that becomes unwritable AFTER setup
        # (a full /var/lock tmpfs is the realistic trigger) previously made
        # save_state() raise, so the NEXT tick's load_state() saw no persisted failure
        # and reached "newly failing" again -- one identical failure email per tick,
        # forever (D-11 failing open). save_state() now falls back to a location
        # outside FOMO_STATE_DIR when the primary write fails, so the suppression
        # decision survives the outage across two separate `run_unattended` processes.
        unwritable_state_dir = tempfile.TemporaryDirectory()
        self.addCleanup(unwritable_state_dir.cleanup)
        state_path = Path(unwritable_state_dir.name)
        os.chmod(state_path, stat.S_IRUSR | stat.S_IXUSR)
        self.addCleanup(lambda: os.chmod(state_path, stat.S_IRWXU))

        # WR-34 (36-REVIEW.md): the fallback location used to be a module-level constant
        # resolving to the real, shared /tmp/fomo-unattended-state.fallback.json -- this
        # test read, wrote and deleted that live path, which would destroy a real
        # deployment's suppression state if this suite ever ran on a host that also runs
        # the cron schedule, and would race a concurrent test run or tick on the same
        # file. CR-05/WR-33 turned the fallback location into a function
        # (`_fallback_state_path()`) precisely so it can be patched to an isolated
        # `TemporaryDirectory()` path instead, matching every other filesystem-touching
        # test in this module.
        fallback_dir = tempfile.TemporaryDirectory()
        self.addCleanup(fallback_dir.cleanup)
        fallback_path = Path(fallback_dir.name) / 'fallback.json'
        fallback_path_patcher = patch('solsys_code.unattended._fallback_state_path', return_value=fallback_path)
        fallback_path_patcher.start()
        self.addCleanup(fallback_path_patcher.stop)

        self._make_campaign_run()
        with (
            override_settings(FOMO_STATE_DIR=str(state_path)),
            patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')),
        ):
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
            self.assertTrue(fallback_path.exists(), 'expected save_state() to fall back when the primary is unwritable')
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

    def test_state_file_with_unparseable_notified_at_still_mails(self):
        # CR-06 (36-REVIEW.md): a state file with a valid failing_steps but an
        # unparseable notified_at -- reachable via a hand edit (the runbook's own
        # troubleshooting section names these files), a truncated/legacy file, or the
        # CR-05 fallback path -- must not permanently suppress notification for that
        # failing set. Write exactly such a file directly (bypassing save_state(), which
        # never itself produces one), then run a tick with the SAME step still failing;
        # before the fix this reached decide_notification() as "same set, never
        # notified" and returned None forever, so no mail was ever sent again.
        state_path = Path(self.tmp_dir.name) / 'unattended-state.json'
        state_path.write_text(json.dumps({'failing_steps': ['reconcile'], 'notified_at': 'not-a-date'}))

        self._make_campaign_run()
        with patch('solsys_code.unattended.reconcile_run', side_effect=RuntimeError('boom')):
            with self.assertRaises(SystemExit):
                call_command('run_unattended')
        self.assertEqual(len(mail.outbox), 1)
        self.assertTrue(mail.outbox[0].subject.startswith('FOMO unattended run failed:'))

    def test_decide_notification_treats_a_none_notified_at_as_due_now(self):
        # CR-06 (36-REVIEW.md): belt-and-braces unit test for decide_notification()'s own
        # guard, independent of load_state()'s fix above -- a previous_state with the
        # SAME failing set as this tick but notified_at=None (however that combination
        # arose) must be treated as due for notification now, not as "already notified,
        # indefinitely" (previously unreachable by either the 'failure' branch, since the
        # sets match, or the 'reminder' branch, since notified_at is None).
        previous_state = {'failing_steps': ['reconcile'], 'notified_at': None}
        decision = unattended.decide_notification(previous_state, ['reconcile'], datetime.now(dt_timezone.utc))
        self.assertEqual(decision, 'failure')


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

    def test_non_2xx_response_is_logged_as_a_failed_ping(self):
        # IN-01 (36-REVIEW.md): requests.get() does not raise on its own for a non-2xx
        # response -- an expired/rotated healthchecks.io URL (404) or a rate limit (429)
        # was silently treated as a delivered ping. raise_for_status() must catch it.
        self._make_campaign_run()
        response = requests.Response()
        response.status_code = 404
        self.mock_requests_get.return_value = response
        with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
            call_command('run_unattended')  # must not raise
        joined = '\n'.join(captured.output)
        self.assertIn('heartbeat ping failed', joined)
        self.assertIn('HTTPError', joined)


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
            stderr_value = stderr.getvalue()
            self.assertIn('lock', stderr_value.lower())
            # Quick task 260927-eqs, Task 2 (DEC-5): the runner's own in-process skip
            # line now carries a trailing, parenthesised timestamp: UTC, then host-local.
            self.assertIsNotNone(
                re.search(
                    r'run_unattended: lock held -- skipping this tick '
                    r'\(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00 '
                    r'\(local \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\)\)',
                    stderr_value,
                )
            )
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
        self.assertEqual(len(mail.outbox), 0)


class TestLocalTimestamp(UnattendedTestBase):
    """Quick task 260927-eqs, Task 2 (DEC-2/DEC-3): ``_local_timestamp()`` reads the
    host's real zone from ``/etc/localtime``, never from the process's own local-time
    conversion (which Django's settings loader pins to UTC)."""

    @skipIf(importlib.util.find_spec('tzdata') is None, 'tzdata package not installed')
    def test_local_timestamp_uses_summer_offset_from_the_localtime_file(self):
        zoneinfo_path = Path(self.tmp_dir.name) / 'localtime-la'
        zoneinfo_path.write_bytes(
            importlib.resources.files('tzdata').joinpath('zoneinfo/America/Los_Angeles').read_bytes()
        )
        with patch.object(unattended, '_HOST_LOCALTIME_PATH', zoneinfo_path):
            result = unattended._local_timestamp(datetime(2026, 7, 1, 12, 0, 0, 123456, tzinfo=dt_timezone.utc))
        self.assertEqual(result, '2026-07-01T05:00:00-07:00')

    @skipIf(importlib.util.find_spec('tzdata') is None, 'tzdata package not installed')
    def test_local_timestamp_uses_winter_offset_from_the_localtime_file(self):
        zoneinfo_path = Path(self.tmp_dir.name) / 'localtime-la'
        zoneinfo_path.write_bytes(
            importlib.resources.files('tzdata').joinpath('zoneinfo/America/Los_Angeles').read_bytes()
        )
        with patch.object(unattended, '_HOST_LOCALTIME_PATH', zoneinfo_path):
            result = unattended._local_timestamp(datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(result, '2025-12-31T16:00:00-08:00')

    def test_missing_localtime_file_falls_back_to_utc_without_raising(self):
        missing_path = Path(self.tmp_dir.name) / 'does-not-exist'
        with patch.object(unattended, '_HOST_LOCALTIME_PATH', missing_path):
            result = unattended._local_timestamp(datetime(2026, 7, 1, 12, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(result, '2026-07-01T12:00:00+00:00')

    def test_corrupt_localtime_file_falls_back_to_utc_without_raising(self):
        corrupt_path = Path(self.tmp_dir.name) / 'localtime-corrupt'
        corrupt_path.write_bytes(b'not a tzif file')
        with patch.object(unattended, '_HOST_LOCALTIME_PATH', corrupt_path):
            result = unattended._local_timestamp(datetime(2026, 7, 1, 12, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(result, '2026-07-01T12:00:00+00:00')


class TestEndBannerTimestamp(UnattendedTestBase):
    """WR-04 (36-REVIEW.md): the END banner must report a freshly-sampled time, not the
    tick's START time -- otherwise no tick's duration is ever readable in the log.

    Quick task 260927-eqs, Task 2: both banners now carry the host-local timestamp
    (DEC-2/DEC-3), and the END banner also carries ``duration=`` (DEC-4)."""

    def test_end_banner_timestamp_differs_from_start_banner_timestamp(self):
        self._make_campaign_run()
        start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt_timezone.utc)
        end_time = datetime(2026, 1, 1, 0, 15, 0, tzinfo=dt_timezone.utc)

        class _FakeDateTime(datetime):
            _values = iter([start_time, end_time])

            @classmethod
            def now(cls, tz=None):
                return next(cls._values)

        with (
            patch('solsys_code.unattended.datetime', _FakeDateTime),
            patch('solsys_code.unattended._host_timezone', return_value=ZoneInfo('America/Los_Angeles')),
        ):
            with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
                call_command('run_unattended')

        joined = '\n'.join(captured.output)
        # UTC first, then the host-local time -- which here falls on the previous date.
        self.assertIn('START 2026-01-01T00:00:00+00:00 (local 2025-12-31T16:00:00-08:00) ===', joined)
        self.assertIn(
            'END 2026-01-01T00:15:00+00:00 (local 2025-12-31T16:15:00-08:00) exit=0 duration=900s ===', joined
        )
        self.assertNotIn('END 2026-01-01T00:00:00+00:00', joined)

    def test_dry_run_banners_carry_local_offset_and_no_fractional_seconds(self):
        stdout = StringIO()
        stderr = StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
                call_command('run_unattended', '--dry-run')

        joined = '\n'.join(captured.output)
        self.assertIsNotNone(
            re.search(
                r'=== FOMO unattended run START \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00 '
                r'\(local \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\) ===',
                joined,
            )
        )
        self.assertIsNotNone(
            re.search(
                r'=== FOMO unattended run END \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00 '
                r'\(local \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\) exit=0 duration=\d+s ===',
                joined,
            )
        )
        # Both regexes above require seconds to be immediately followed by the UTC
        # offset, with no `.` fractional-seconds component in between.


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
        # CR-06 (36-REVIEW.md): an unparseable notified_at makes the WHOLE record
        # untrustworthy, not just this one field. Before this fix, failing_steps was
        # kept (only notified_at was nulled), which wedged decide_notification() on
        # "same set, never notified" forever -- no failure mail, and no reminder either,
        # since the reminder branch also requires a non-None notified_at. This function's
        # own docstring promises "no prior failure" for the whole record; assert that
        # promise, not the partial-preservation behavior it previously violated.
        self._write_state_file(json.dumps({'failing_steps': ['reconcile'], 'notified_at': 'not-a-date'}))
        self.assertEqual(unattended.load_state(), {'failing_steps': [], 'notified_at': None})

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


class TestStateFileAtomicWrite(UnattendedTestBase):
    """IN-03 (36-REVIEW.md): the state file must be written atomically (temp file + a
    rename), with an explicit mode rather than relying on the process umask."""

    def test_written_file_has_explicit_0o600_mode(self):
        unattended.save_state(['reconcile'], None)
        state_path = Path(self.tmp_dir.name) / 'unattended-state.json'
        self.assertEqual(stat.S_IMODE(state_path.stat().st_mode), 0o600)

    def test_no_leftover_temp_file_after_a_successful_write(self):
        unattended.save_state(['reconcile'], None)
        leftover_temp_files = [p for p in Path(self.tmp_dir.name).iterdir() if p.name.endswith('.tmp')]
        self.assertEqual(leftover_temp_files, [])

    def test_content_round_trips_through_the_atomic_write(self):
        now = datetime.now(dt_timezone.utc)
        unattended.save_state(['reconcile', 'discovery'], now)
        state = unattended.load_state()
        self.assertEqual(state['failing_steps'], ['discovery', 'reconcile'])
        self.assertEqual(state['notified_at'], now)

    def test_stale_temp_file_from_a_killed_write_is_reaped(self):
        # IN-21 (36-REVIEW.md): a SIGKILL/OOM kill between mkstemp() and os.replace()
        # leaves a `.<filename>.<random>.tmp` file behind forever, since load_state()
        # only ever reads the exact target filename -- nothing else ever cleaned these
        # up. A file older than one cron tick interval can only be such a leftover
        # (a single write is milliseconds of work), so the next write reaps it.
        state_dir = Path(self.tmp_dir.name)
        stale_tmp = state_dir / '.unattended-state.json.stale123.tmp'
        stale_tmp.write_text('{}')
        stale_age = unattended._CRON_TICK_INTERVAL + timedelta(minutes=1)
        stale_mtime = (datetime.now(dt_timezone.utc) - stale_age).timestamp()
        os.utime(stale_tmp, (stale_mtime, stale_mtime))

        fresh_tmp = state_dir / '.unattended-state.json.fresh456.tmp'
        fresh_tmp.write_text('{}')

        unattended.save_state(['reconcile'], None)

        self.assertFalse(stale_tmp.exists(), 'stale temp file older than one tick interval should be reaped')
        self.assertTrue(fresh_tmp.exists(), 'a temp file younger than one tick interval must not be touched')


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

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_calls_both_facilities_with_fresh_instances(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = []
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        unattended.step_status_refresh(dry_run=False)

        mock_lco_cls.assert_called_once_with()
        mock_soar_cls.assert_called_once_with()
        mock_lco_cls.return_value.update_all_observation_statuses.assert_called_once_with()
        mock_soar_cls.return_value.update_all_observation_statuses.assert_called_once_with()
        self.assertIsNot(mock_lco_cls.return_value, mock_soar_cls.return_value)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_clean_refresh_is_not_a_failure(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = []
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertFalse(result.failed)
        self.assertIn('LCO: failed 0', result.summary)
        self.assertIn('SOAR: failed 0', result.summary)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_non_empty_failure_list_is_a_step_failure(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', 'boom')]
        mock_lco_cls.return_value.update_observation_status.return_value = None
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertTrue(result.failed)
        self.assertIn('LCO: failed 1', result.summary)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
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

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
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

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_facility_exception_is_isolated_per_facility(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.side_effect = RuntimeError('lco down')
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        mock_soar_cls.return_value.update_all_observation_statuses.assert_called_once_with()
        self.assertTrue(result.failed)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_dry_run_makes_no_facility_call(self, mock_lco_cls, mock_soar_cls):
        result = unattended.step_status_refresh(dry_run=True)

        mock_lco_cls.assert_not_called()
        mock_soar_cls.assert_not_called()
        self.assertFalse(result.failed)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
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

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_under_cap_failure_count_omits_the_capped_note(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', 'boom')]
        mock_lco_cls.return_value.update_observation_status.return_value = None
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertNotIn('recheck capped', result.summary)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_whole_facility_outage_reads_as_outage_not_failed_one(self, mock_lco_cls, mock_soar_cls):
        # IN-05 (36-REVIEW.md): a whole-facility outage (update_all_observation_statuses()
        # itself raised) must never read as "failed 1" -- the true affected-record count
        # is unknown, and that phrasing understates a systemic outage as a single record.
        mock_lco_cls.return_value.update_all_observation_statuses.side_effect = RuntimeError('lco down')
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertTrue(result.failed)
        self.assertIn('LCO: outage (RuntimeError)', result.summary)
        self.assertNotIn('LCO: failed 1', result.summary)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_clean_refresh_summary_has_no_dangling_classes_fragment(self, mock_lco_cls, mock_soar_cls):
        # IN-05 (36-REVIEW.md): an empty classes list must omit the whole 'classes: '
        # segment rather than leaving a dangling, content-free fragment in the summary.
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = []
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertNotIn('classes:', result.summary)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_per_record_http_status_code_is_reported(self, mock_lco_cls, mock_soar_cls):
        # Quick task 260927-eqs, Task 1: a portal HTTPError's numeric status code must
        # travel to the per-record log line and the step summary's `classes:` field.
        error = _http_error(502, reason='Bad Gateway')
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', str(error))]
        mock_lco_cls.return_value.update_observation_status.side_effect = error
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        with self.assertLogs('solsys_code.unattended', level='WARNING') as captured:
            result = unattended.step_status_refresh(dry_run=False)

        joined = '\n'.join(captured.output)
        self.assertIn('observation_id=obs-1 HTTPError 502', joined)
        self.assertIn('classes: HTTPError 502', result.summary)
        self.assertTrue(result.failed)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_multiple_status_codes_are_deduplicated_by_label(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.return_value = [
            ('obs-1', 'boom'),
            ('obs-2', 'boom'),
            ('obs-3', 'boom'),
        ]
        mock_lco_cls.return_value.update_observation_status.side_effect = [
            _http_error(502),
            _http_error(502),
            _http_error(504),
        ]
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        result = unattended.step_status_refresh(dry_run=False)

        self.assertIn('classes: HTTPError 502, HTTPError 504', result.summary)
        self.assertEqual(result.summary.count('HTTPError 502'), 1)
        self.assertEqual(result.summary.count('HTTPError 504'), 1)

    @patch('solsys_code.unattended.FomoSOARFacility')
    @patch('solsys_code.unattended.FomoLCOFacility')
    def test_outage_reports_the_http_status_code(self, mock_lco_cls, mock_soar_cls):
        mock_lco_cls.return_value.update_all_observation_statuses.side_effect = _http_error(
            503, reason='Service Unavailable'
        )
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []

        with self.assertLogs('solsys_code.unattended', level='WARNING') as captured:
            result = unattended.step_status_refresh(dry_run=False)

        self.assertIn('LCO: outage (HTTPError 503)', result.summary)
        self.assertNotIn('LCO: failed 1', result.summary)
        joined = '\n'.join(captured.output)
        self.assertTrue(any(line.endswith('HTTPError 503') for line in joined.splitlines()))


class TestStatusRefreshKeepsBlockTimes(UnattendedTestBase):
    """G-37.1-1-alloc: the poll reads blocks with FOMO's rule, so it neither erases nor fails to record the
    times of an in-progress or aborted block. A real FomoLCOFacility runs; only the portal calls are mocked."""

    PORTAL = 'solsys_code.observation_blocks.make_request'

    def setUp(self):
        super().setUp()
        soar_patcher = patch('solsys_code.unattended.FomoSOARFacility')
        mock_soar_cls = soar_patcher.start()
        self.addCleanup(soar_patcher.stop)
        mock_soar_cls.return_value.update_all_observation_statuses.return_value = []
        self.target = NonSiderealTargetFactory.create(name='Didymos')

    def _pending_record(self):
        return ObservationRecord.objects.create(
            target=self.target, facility='LCO', observation_id='500', status='PENDING', parameters={}
        )

    def test_in_progress_block_times_survive_two_polls(self):
        record = self._pending_record()
        block = {'state': 'IN_PROGRESS', 'start': '2026-07-01T01:00:00Z', 'end': '2026-07-01T01:40:00Z'}
        with patch(self.PORTAL, side_effect=portal_side_effect({'500': 'PENDING'}, {'500': [block]})):
            unattended.step_status_refresh(dry_run=False)
            record.refresh_from_db()
            first = (record.scheduled_start, record.scheduled_end)
            unattended.step_status_refresh(dry_run=False)
            record.refresh_from_db()

        self.assertIsNotNone(first[0])
        self.assertIsNotNone(first[1])
        self.assertEqual((record.scheduled_start, record.scheduled_end), first)
        self.assertEqual(record.status, 'PENDING')

    def test_window_expired_transition_stores_the_aborted_block(self):
        record = self._pending_record()
        block = {'state': 'ABORTED', 'start': '2026-07-01T01:00:00Z', 'end': '2026-07-01T01:40:00Z'}
        with patch(self.PORTAL, side_effect=portal_side_effect({'500': 'WINDOW_EXPIRED'}, {'500': [block]})):
            result = unattended.step_status_refresh(dry_run=False)

        record.refresh_from_db()
        self.assertFalse(result.failed)
        self.assertEqual(record.status, 'WINDOW_EXPIRED')
        self.assertIsNotNone(record.scheduled_start)
        self.assertIsNotNone(record.scheduled_end)


class TestExceptionLabel(UnattendedTestBase):
    """Quick task 260927-eqs, Task 1: ``_exception_label()``'s scope and hygiene."""

    def test_http_error_with_502_response_includes_the_status_code(self):
        exc = _http_error(
            502,
            reason='Bad Gateway',
            url='https://observe.lco.global/api/observations/?api_key=FAKE-KEY-LABEL-TEST',
            headers={'Authorization': 'Token FAKE-KEY-LABEL-TEST'},
            body=b'body-marker-label-test',
        )
        self.assertEqual(unattended._exception_label(exc), 'HTTPError 502')

    def test_http_error_with_404_response_includes_the_status_code(self):
        exc = _http_error(404, reason='Not Found')
        self.assertEqual(unattended._exception_label(exc), 'HTTPError 404')

    def test_falsy_4xx_response_is_still_read_not_treated_as_no_response(self):
        # requests.Response.__bool__ returns .ok, which is False for every 4xx/5xx --
        # proves the helper never tests the response's truthiness.
        exc = _http_error(404)
        self.assertFalse(exc.response)
        self.assertEqual(unattended._exception_label(exc), 'HTTPError 404')

    def test_http_error_with_no_response_falls_back_to_bare_class_name(self):
        exc = requests.exceptions.HTTPError('boom')
        self.assertEqual(unattended._exception_label(exc), 'HTTPError')

    def test_http_error_with_non_int_status_code_falls_back_to_bare_class_name(self):
        exc = requests.exceptions.HTTPError(response=MagicMock())
        self.assertEqual(unattended._exception_label(exc), 'HTTPError')

    def test_generic_exception_returns_bare_class_name(self):
        self.assertEqual(unattended._exception_label(RuntimeError('x')), 'RuntimeError')

    def test_improper_credentials_exception_stays_name_only(self):
        exc = ImproperCredentialsException('OCS: b"secret body"')
        self.assertEqual(unattended._exception_label(exc), 'ImproperCredentialsException')


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

    @patch('solsys_code.management.commands.backfill_lco_observations.sweep_proposal')
    def test_per_request_skip_reasons_are_logged_not_discarded(self, mock_sweep_proposal):
        # IN-02 (36-REVIEW.md): sweep_proposal()'s own per-request skip-reason lines
        # sink into a throwaway io.StringIO() when this step passes no stdout/stderr --
        # every such reason was silently discarded. They must now reach the log.
        #
        # WR-18 (36-REVIEW.md): asserted at INFO, not DEBUG -- this project's own
        # LOGGING config pins the root logger to INFO, so a record logged at DEBUG is
        # filtered out before it ever reaches the crontab's redirected unattended.log.
        # Asserting only that logger.debug() was CALLED (the previous version of this
        # test) passed even though the fix's stated outcome -- "they must now reach the
        # log" -- was not actually achieved in production. A future downgrade back to
        # DEBUG must fail this test.
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')

        def _write_skip_reason(_proposal, **kwargs):
            kwargs['stderr'].write('Skipping request 123: no configuration with a named target.')
            return 'requestgroups seen: 1, skipped: 1'

        mock_sweep_proposal.side_effect = _write_skip_reason

        with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
            unattended.step_discovery(dry_run=False)

        joined = '\n'.join(captured.output)
        self.assertIn('no configuration with a named target', joined)

    @patch('solsys_code.management.commands.backfill_lco_observations.sweep_proposal')
    def test_multiline_sweep_output_is_logged_as_one_record_per_line(self, mock_sweep_proposal):
        # IN-38 (36-REVIEW.md): the captured sweep buffer was previously logged as ONE
        # multi-line INFO record -- a --dry-run preview writes one "Would create/reuse
        # ..." line per portal request, so a large proposal produced one enormous log
        # line that no line-oriented tool (grep, logrotate's size accounting, journald's
        # field limits) handles gracefully. Each line must instead be its own record.
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')

        def _write_two_lines(_proposal, **kwargs):
            kwargs['stdout'].write('Would create observation for obs-1.\nWould reuse observation for obs-2.\n')
            return 'requestgroups seen: 1, would create: 1, would reuse: 1'

        mock_sweep_proposal.side_effect = _write_two_lines

        with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
            unattended.step_discovery(dry_run=True)

        discovery_records = [record for record in captured.output if 'discovery stdout' in record]
        self.assertEqual(len(discovery_records), 2, discovery_records)
        self.assertTrue(any('obs-1' in record for record in discovery_records))
        self.assertTrue(any('obs-2' in record for record in discovery_records))

    @patch('tom_observations.facilities.lco.LCOFacility.get_observation_status')
    @patch('solsys_code.management.commands.backfill_lco_observations.make_request')
    def test_system_links_reach_the_log_and_leave_the_step_summary_unchanged(
        self, mock_make_request, mock_get_observation_status
    ):
        # ALLOC-06 (37.1-01, D-04 as corrected): no sweep_proposal() mock -- the real sweep
        # links a matching record, its per-link stdout line reaches the log as its own INFO
        # record, and the step's own summary keeps its 'swept: N, failed: M' shape.
        mock_get_observation_status.return_value = {
            'state': 'COMPLETED',
            'scheduled_start': '2026-07-01T00:10:00+00:00',
            'scheduled_end': '2026-07-01T00:20:00+00:00',
        }
        target = NonSiderealTargetFactory.create(name='Didymos')
        WatchedProposal.objects.create(proposal_code='LCO2026A-003')
        run = CampaignRun.objects.create(
            campaign=None,
            target=target,
            proposal_code='LCO2026A-003',
            source=CampaignRun.Source.LCO_QUEUE,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            telescope_instrument='1m0/Sinistro',
            telescope_class='1m0',
            window_start=date(2026, 6, 29),
            window_end=date(2026, 7, 2),
        )
        mock_make_request.return_value = _page_response([_request_group(1, 'Didymos 2026 - ELP')])

        with self.assertLogs('solsys_code.unattended', level='INFO') as captured:
            result = unattended.step_discovery(dry_run=False)

        link_records = [
            record for record in captured.output if 'discovery stdout: System-linked ObservationRecord' in record
        ]
        self.assertEqual(len(link_records), 1, captured.output)
        self.assertIn(f'CampaignRun #{run.pk}', link_records[0])
        self.assertEqual(result.summary, 'swept: 1, failed: 0')
        self.assertFalse(result.failed)


class TestProposalAllocationStep(UnattendedTestBase):
    """Task 3 (D-07): the proposal-allocation fetch step joins the runner's fixed step order."""

    def test_registered_as_the_fifth_and_final_step(self):
        names = [name for name, _fn in unattended.STEPS]
        self.assertEqual(names[-1], 'proposal_allocation')
        self.assertEqual(names[:-1], ['status_refresh', 'project_sweep', 'discovery', 'reconcile'])

    def test_dry_run_makes_no_network_call(self):
        with patch('solsys_code.proposal_allocation.make_request') as mock_make_request:
            result = unattended.step_proposal_allocation(dry_run=True)

        mock_make_request.assert_not_called()
        self.assertFalse(result.failed)
        self.assertIn('dry run', result.summary)

    def test_lock_contended_is_a_non_failing_skip(self):
        lock_dir = Path(self.tmp_dir.name)
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / 'proposal_allocation.lock'
        fh = lock_path.open('a+')
        self.addCleanup(fh.close)
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = unattended.step_proposal_allocation(dry_run=False)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

        self.assertFalse(result.failed)
        self.assertIn('lock', result.summary.lower())

    def test_success_reports_counters(self):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        response = MagicMock()
        response.json.return_value = {
            'timeallocation_set': [
                {'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 10.0, 'std_time_used': 2.0}
            ]
        }
        with patch('solsys_code.proposal_allocation.make_request', return_value=response):
            result = unattended.step_proposal_allocation(dry_run=False)

        self.assertFalse(result.failed)
        self.assertIn('proposals: 1', result.summary)
        self.assertIn('rows written: 1', result.summary)

    def test_portal_outage_is_a_step_failure(self):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        with patch('solsys_code.proposal_allocation.make_request', side_effect=requests.exceptions.Timeout('boom')):
            result = unattended.step_proposal_allocation(dry_run=False)

        self.assertTrue(result.failed)
        self.assertIn('failed: 1', result.summary)

    def test_summary_leaks_no_api_key_or_response_body(self):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        fake_key = 'FAKE-API-KEY-CREDHYG-PROPOSAL-1'
        with patch(
            'solsys_code.proposal_allocation.make_request',
            side_effect=ImproperCredentialsException(f'OCS: {fake_key}'),
        ):
            result = unattended.step_proposal_allocation(dry_run=False)

        self.assertNotIn(fake_key, result.summary)
        self.assertTrue(result.failed)

    def _watched_plus_eso_run(self):
        """An active watched proposal plus a classical NTT run carrying an ESO code (the F7 shape)."""
        la_silla = Observatory.objects.create(
            obscode='809',
            name='La Silla',
            short_name='La Silla',
            lat=-29.2563,
            lon=-70.7380,
            altitude=2400.0,
            timezone='America/Santiago',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        CampaignRun.objects.create(
            telescope_instrument='NTT/EFOSC2',
            source=CampaignRun.Source.CLASSICAL_FILE,
            site=la_silla,
            proposal_code='117.2A2N.001',
        )

    def test_non_lco_code_is_reported_not_fetchable_and_the_step_stays_ok(self):
        self._watched_plus_eso_run()
        response = MagicMock()
        response.json.return_value = {
            'timeallocation_set': [
                {'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 10.0, 'std_time_used': 2.0}
            ]
        }
        with patch('solsys_code.proposal_allocation.make_request', return_value=response) as mock_make_request:
            result = unattended.step_proposal_allocation(dry_run=False)

        self.assertFalse(result.failed)
        self.assertIn('proposals: 1', result.summary)
        self.assertIn('failed: 0', result.summary)
        self.assertIn('not fetchable: 1', result.summary)
        self.assertNotIn('first error', result.summary)
        self.assertFalse(any('117.2A2N.001' in str(c.args[1]) for c in mock_make_request.call_args_list))

    def test_a_real_portal_failure_still_fails_the_step_alongside_an_exclusion(self):
        self._watched_plus_eso_run()
        with patch('solsys_code.proposal_allocation.make_request', side_effect=requests.exceptions.Timeout('boom')):
            result = unattended.step_proposal_allocation(dry_run=False)

        self.assertTrue(result.failed)
        self.assertIn('failed: 1', result.summary)
        self.assertIn('not fetchable: 1', result.summary)
        self.assertIn('first error: Timeout', result.summary)


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
            patch('solsys_code.unattended.FomoLCOFacility') as mock_lco_cls,
            patch('solsys_code.unattended.FomoSOARFacility') as mock_soar_cls,
            # The full tick's proposal_allocation step still builds TOM's own LCOFacility.
            patch('solsys_code.unattended.LCOFacility'),
        ):
            mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', error_message)]
            mock_lco_cls.return_value.update_observation_status.side_effect = requests.exceptions.HTTPError(
                error_message
            )
            mock_soar_cls.return_value.update_all_observation_statuses.return_value = []
            log_output, stdout_value, stderr_value = self._run_tick_capturing()

        self._assert_no_secrets_leaked(log_output, stdout_value, stderr_value)

    def test_status_refresh_http_status_code_is_reported_without_leaking(self):
        # Quick task 260927-eqs, Task 1's own end-to-end proof: the portal's HTTP status
        # code travels to the log line, the step summary and the failure email, while no
        # response body, URL, query string, header, reason phrase or exception message
        # escapes anywhere.
        body_marker = 'BODY-MARKER-CREDHYG-502'
        full_url = f'https://observe.lco.global/api/observations/?api_key={_FAKE_LCO_API_KEY}'
        auth_header_value = f'Token {_FAKE_LCO_API_KEY}'
        error = _http_error(
            502,
            reason='Bad Gateway',
            url=full_url,
            headers={'Authorization': auth_header_value},
            body=body_marker.encode(),
        )
        with (
            patch('solsys_code.unattended.FomoLCOFacility') as mock_lco_cls,
            patch('solsys_code.unattended.FomoSOARFacility') as mock_soar_cls,
            # The full tick's proposal_allocation step still builds TOM's own LCOFacility.
            patch('solsys_code.unattended.LCOFacility'),
        ):
            mock_lco_cls.return_value.update_all_observation_statuses.return_value = [('obs-1', str(error))]
            mock_lco_cls.return_value.update_observation_status.side_effect = error
            mock_soar_cls.return_value.update_all_observation_statuses.return_value = []
            log_output, stdout_value, stderr_value = self._run_tick_capturing()

        joined_logs = '\n'.join(log_output)
        self.assertIn('HTTPError 502', joined_logs)
        self.assertEqual(len(mail.outbox), 1)
        sent = mail.outbox[0]
        self.assertIn('status_refresh', sent.subject)
        self.assertIn('HTTPError 502', sent.body)

        self._assert_no_secrets_leaked(log_output, stdout_value, stderr_value)

        leaked_values = (
            body_marker,
            full_url,
            f'api_key={_FAKE_LCO_API_KEY}',
            auth_header_value,
            'Bad Gateway',
            str(error),
        )
        for leaked in leaked_values:
            self.assertNotIn(leaked, joined_logs)
            self.assertNotIn(leaked, stdout_value)
            self.assertNotIn(leaked, stderr_value)
            self.assertNotIn(leaked, sent.subject)
            self.assertNotIn(leaked, sent.body)

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
