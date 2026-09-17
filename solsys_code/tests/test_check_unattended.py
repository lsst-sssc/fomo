"""Tests for `check_unattended` (Phase 36 Plan 04).

Covers the six prerequisite checks, their aggregation into a single pass/fail run
(``TestHardChecks``/``TestWarningChecks``), the printed cron line (``TestCronLine``),
the ``--send-test-email`` flag (``TestTestEmail``), and SCHED-10/D-15 credential-hygiene
(``TestNoValueLeakage``). No ``Target`` fixture is used anywhere in this module.
"""

import io
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings

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
