"""Tests for `check_unattended` (Phase 36 Plan 04).

Covers the six prerequisite checks, their aggregation into a single pass/fail run
(``TestHardChecks``/``TestWarningChecks``), the printed cron line (``TestCronLine``),
the ``--send-test-email`` flag (``TestTestEmail``), and SCHED-10/D-15 credential-hygiene
(``TestNoValueLeakage``). No ``Target`` fixture is used anywhere in this module.
"""

import io
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.conf import settings as django_settings
from django.contrib.auth.models import User
from django.core import mail
from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings

from solsys_code.management.commands.check_unattended import cron_line
from solsys_code.models import WatchedProposal

_FAKE_HEARTBEAT_URL = 'https://hc.example/UUID-TEST-CHECK-UNATTENDED'
_FAKE_MAIL_PASSWORD = 'sk-fake-mail-password-check-unattended'  # noqa: S105 -- fixture literal, not a real secret
_FAKE_LCO_API_KEY = 'fake-lco-api-key-check-unattended'


def _run(*args, **kwargs):
    """Call `check_unattended`, returning (stdout, stderr) as strings.

    Any raised ``CommandError`` propagates -- callers that expect a failure use
    ``assertRaises`` around this helper.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    call_command('check_unattended', *args, stdout=stdout, stderr=stderr, **kwargs)
    return stdout.getvalue(), stderr.getvalue()


class CheckUnattendedTestBase(TestCase):
    """Shared fixture: writable temp lock/log directories and a staff user with an
    email, so every hard check passes unless a test deliberately breaks one."""

    def setUp(self):
        super().setUp()
        self.lock_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.lock_dir.cleanup)
        self.log_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.log_dir.cleanup)

        settings_override = override_settings(
            FOMO_LOCK_DIR=self.lock_dir.name,
            FOMO_STATE_DIR=self.lock_dir.name,
            FOMO_LOG_FILE=str(Path(self.log_dir.name) / 'unattended.log'),
        )
        settings_override.enable()
        self.addCleanup(settings_override.disable)

        self.staff_user = User.objects.create_user(
            username='staff-with-email', email='staff@example.org', is_staff=True
        )

    def _make_unwritable_parent(self) -> Path:
        """Return a directory whose contents cannot be written to (mode 0o500), with
        its permissions restored before the enclosing TemporaryDirectory tears down."""
        readonly_dir = tempfile.TemporaryDirectory()
        self.addCleanup(readonly_dir.cleanup)
        path = Path(readonly_dir.name)
        os.chmod(path, stat.S_IRUSR | stat.S_IXUSR)
        self.addCleanup(lambda: os.chmod(path, stat.S_IRWXU))
        return path


class TestHardChecks(CheckUnattendedTestBase):
    def test_missing_flock_fails(self):
        with patch('solsys_code.management.commands.check_unattended.shutil.which', return_value=None):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('flock', str(ctx.exception))

    def test_unwritable_lock_dir_fails(self):
        readonly_parent = self._make_unwritable_parent()
        with override_settings(FOMO_LOCK_DIR=str(readonly_parent / 'sublock')):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('FOMO_LOCK_DIR', str(ctx.exception))

    def test_unwritable_log_dir_fails(self):
        readonly_parent = self._make_unwritable_parent()
        with override_settings(FOMO_LOG_FILE=str(readonly_parent / 'sublog' / 'unattended.log')):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('FOMO_LOG_FILE', str(ctx.exception))

    def test_console_email_backend_fails(self):
        # The test runner's automatic locmem swap must be overridden explicitly for this
        # branch to be reachable at all.
        with override_settings(EMAIL_BACKEND='django.core.mail.backends.console.EmailBackend'):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('EMAIL_BACKEND', str(ctx.exception))

    def test_no_staff_email_fails(self):
        self.staff_user.delete()
        User.objects.create_user(username='staff-no-email', is_staff=True, email='')
        User.objects.create_user(username='regular-with-email', is_staff=False, email='regular@example.org')
        with self.assertRaises(CommandError) as ctx:
            _run()
        self.assertIn('staff_recipients', str(ctx.exception))

    def test_all_hard_checks_passing_exits_zero(self):
        # Does not raise -- that is the assertion.
        stdout, _stderr = _run()
        self.assertIn('[ok] flock', stdout)
        self.assertIn('[ok] FOMO_LOCK_DIR', stdout)
        self.assertIn('[ok] FOMO_LOG_FILE', stdout)
        self.assertIn('[ok] EMAIL_BACKEND', stdout)
        self.assertIn('[ok] staff_recipients', stdout)

    def test_command_writes_nothing(self):
        lock_path = Path(self.lock_dir.name) / 'does-not-exist-yet'
        log_path = Path(self.log_dir.name) / 'does-not-exist-yet'
        with override_settings(FOMO_LOCK_DIR=str(lock_path), FOMO_LOG_FILE=str(log_path / 'unattended.log')):
            self.assertFalse(lock_path.exists())
            self.assertFalse(log_path.exists())
            _run()
            self.assertFalse(lock_path.exists())
            self.assertFalse(log_path.exists())
        self.assertEqual(WatchedProposal.objects.count(), 0)

    def test_writable_result_names_the_uid_that_was_actually_tested(self):
        # WR-06 (36-REVIEW.md): os.access() only answers "can *this* process's uid
        # write here" -- report the resolved owner/mode alongside the verdict, and say
        # explicitly whose write access was tested, so an operator running the
        # preflight as root does not mistake that [ok] for one tested as the cron
        # account.
        stdout, _stderr = _run()
        self.assertIn(f'writable by uid {os.geteuid()}', stdout)
        self.assertIn('run this check as the account that will actually run unattended', stdout)


class TestWarningChecks(CheckUnattendedTestBase):
    def test_unset_heartbeat_is_a_warning(self):
        with override_settings(FOMO_HEARTBEAT_URL=None):
            stdout, stderr = _run()
        self.assertIn('FOMO_HEARTBEAT_URL', stdout)
        self.assertIn('[WARN]', stdout)
        self.assertIn('FOMO_HEARTBEAT_URL', stderr)

    def test_empty_watched_list_is_a_warning(self):
        stdout, _stderr = _run()
        self.assertIn('[WARN] watched_proposals', stdout)

        WatchedProposal.objects.create(proposal_code='KEY2026B-004', is_active=True)
        stdout, _stderr = _run()
        self.assertNotIn('[WARN] watched_proposals', stdout)
        self.assertIn('[ok] watched_proposals', stdout)

    def test_warnings_do_not_mask_a_hard_failure(self):
        self.staff_user.delete()
        stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
        with override_settings(FOMO_HEARTBEAT_URL=None):
            with self.assertRaises(CommandError):
                call_command('check_unattended', stdout=stdout_capture, stderr=stderr_capture)
        combined = stdout_capture.getvalue() + stderr_capture.getvalue()
        self.assertIn('FOMO_HEARTBEAT_URL', combined)
        self.assertIn('staff_recipients', combined)

    def test_localhost_default_base_url_is_a_warning(self):
        # WR-07 (36-REVIEW.md): FOMO_BASE_URL left at its localhost dev default makes
        # every emailed link (failure notice, campaign approval-queue notice) unusable
        # off this host -- nothing else checks it, so this preflight must.
        with override_settings(FOMO_BASE_URL='http://localhost:8000'):
            stdout, _stderr = _run()
        self.assertIn('[WARN] FOMO_BASE_URL', stdout)

        with override_settings(FOMO_BASE_URL='https://fomo.example.org'):
            stdout, _stderr = _run()
        self.assertIn('[ok] FOMO_BASE_URL', stdout)

    def test_unset_base_url_is_a_warning(self):
        with override_settings(FOMO_BASE_URL=None):
            stdout, _stderr = _run()
        self.assertIn('[WARN] FOMO_BASE_URL', stdout)


class TestCronLine(CheckUnattendedTestBase):
    def test_line_has_real_paths(self):
        line = cron_line()
        self.assertIn(sys.executable, line)
        manage_py_path = str(Path(django_settings.BASE_DIR).parent / 'manage.py')
        self.assertIn(manage_py_path, line)
        self.assertNotIn('/path/to/venv/bin/python', line)
        self.assertNotIn('/path/to/checkout/manage.py', line)

    def test_line_matches_the_committed_template_shape(self):
        line = cron_line()
        for element in (
            '*/15 * * * *',
            f'{shutil.which("flock")} -n -E 99',
            'run_unattended.cron.lock',
            'run_unattended',
            '2>&1',
            'rc=$?',
            '[ $rc -eq 99 ]',
            'lock held',
            'exit $rc',
        ):
            self.assertIn(element, line)
        self.assertIn('>>', line)

    def test_line_ends_with_an_explicit_exit_of_the_captured_status(self):
        # WR-09 (36-REVIEW.md): the skip-tail's own `[ ... ] && echo ...` must not be the
        # line's last command -- that made the *tail's* exit status (1 unless it actually
        # fired) the line's reported status, inverting cron's view of a healthy tick (1)
        # vs. a skipped one (0). The line must capture flock's status into `$rc` and end
        # with an explicit `exit $rc` so cron always sees `run_unattended`'s own status.
        line = cron_line()
        self.assertTrue(line.rstrip().endswith('exit $rc'), line)

    def test_skip_tail_is_gated_on_exit_code_99_not_any_failure(self):
        # WR-01 (36-REVIEW.md): the skip tail must be gated on flock's own -E 99 exit
        # code, never on a bare '||' that would also fire on run_unattended's own exit 1
        # (a step failure) -- that would mislabel a failing-but-genuinely-ran tick as
        # "lock held" in the log.
        line = cron_line()
        self.assertNotIn('|| echo', line)
        self.assertIn('-E 99', line)

    def test_flock_path_is_resolved_not_hardcoded(self):
        # WR-05 (36-REVIEW.md): cron_line() must print the same resolved `flock` path
        # check_flock() already verified is on PATH -- not a hardcoded '/usr/bin/flock'
        # that can be wrong on a non-merged-/usr layout, a venv-provided util-linux, or
        # a container image that only has it in /bin.
        with patch(
            'solsys_code.management.commands.check_unattended.shutil.which', return_value='/opt/util-linux/flock'
        ):
            line = cron_line()
        self.assertIn('/opt/util-linux/flock -n', line)
        self.assertNotIn('/usr/bin/flock', line)

    def test_cron_lock_differs_from_the_runner_internal_lock(self):
        # CR-01 (36-REVIEW.md): the cron guard and `command_lock('run_unattended')`'s own
        # lock file must never be the same path -- `flock(2)` locks are per open file
        # description, so a shared name would deny the child's own lock attempt and
        # silently no-op every scheduled tick.
        line = cron_line()
        runner_internal_lock = str(Path(django_settings.FOMO_LOCK_DIR) / 'run_unattended.lock')
        self.assertNotIn(runner_internal_lock, line)
        self.assertIn(str(Path(django_settings.FOMO_LOCK_DIR) / 'run_unattended.cron.lock'), line)

    def test_line_carries_no_setting_value(self):
        with override_settings(FOMO_HEARTBEAT_URL=_FAKE_HEARTBEAT_URL):
            original_lco_api_key = django_settings.FACILITIES['LCO'].get('api_key')
            django_settings.FACILITIES['LCO']['api_key'] = _FAKE_LCO_API_KEY
            try:
                stdout, _stderr = _run()
            finally:
                django_settings.FACILITIES['LCO']['api_key'] = original_lco_api_key
        with override_settings(EMAIL_HOST_PASSWORD=_FAKE_MAIL_PASSWORD):
            line = cron_line()
        self.assertNotIn(_FAKE_HEARTBEAT_URL, line)
        self.assertNotIn(_FAKE_MAIL_PASSWORD, line)
        self.assertNotIn(_FAKE_LCO_API_KEY, line)
        self.assertNotIn(_FAKE_HEARTBEAT_URL, stdout)
        self.assertNotIn(_FAKE_LCO_API_KEY, stdout)


class TestTestEmail(CheckUnattendedTestBase):
    def test_send_test_email_sends_one_message(self):
        _run('--send-test-email')
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn(self.staff_user.email, mail.outbox[0].to)
        self.assertIn('FOMO', mail.outbox[0].subject)
        self.assertIn('test', mail.outbox[0].subject.lower())

    def test_send_test_email_without_recipients_fails(self):
        self.staff_user.delete()
        with self.assertRaises(CommandError) as ctx:
            _run('--send-test-email')
        self.assertIn('send_test_email', str(ctx.exception))

    def test_flag_absent_sends_nothing(self):
        _run()
        self.assertEqual(len(mail.outbox), 0)


class TestNoValueLeakage(CheckUnattendedTestBase):
    def _seed_fake_values(self):
        original_lco_api_key = django_settings.FACILITIES['LCO'].get('api_key')
        django_settings.FACILITIES['LCO']['api_key'] = _FAKE_LCO_API_KEY

        def _restore():
            django_settings.FACILITIES['LCO']['api_key'] = original_lco_api_key

        self.addCleanup(_restore)

        settings_override = override_settings(
            FOMO_HEARTBEAT_URL=_FAKE_HEARTBEAT_URL,
            EMAIL_HOST_PASSWORD=_FAKE_MAIL_PASSWORD,
        )
        settings_override.enable()
        self.addCleanup(settings_override.disable)

    def _assert_no_leak(self, *outputs: str) -> None:
        for output in outputs:
            for secret in (_FAKE_HEARTBEAT_URL, _FAKE_MAIL_PASSWORD, _FAKE_LCO_API_KEY):
                self.assertNotIn(secret, output)

    def test_output_never_contains_a_seeded_value(self):
        self._seed_fake_values()

        # All-passing configuration.
        stdout, stderr = _run()
        self._assert_no_leak(stdout, stderr)

        # Hard-failing configuration.
        self.staff_user.delete()
        stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
        with self.assertRaises(CommandError) as ctx:
            call_command('check_unattended', stdout=stdout_capture, stderr=stderr_capture)
        self._assert_no_leak(stdout_capture.getvalue(), stderr_capture.getvalue(), str(ctx.exception))
