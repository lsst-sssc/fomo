"""Tests for `check_unattended` (Phase 36 Plan 04).

Covers the eight prerequisite checks, their aggregation into a single pass/fail run
(``TestHardChecks``/``TestWarningChecks``), the printed cron line (``TestCronLine``),
the ``--send-test-email`` flag (``TestTestEmail``), and SCHED-10/D-15 credential-hygiene
(``TestNoValueLeakage``). No ``Target`` fixture is used anywhere in this module.
"""

import io
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import skipIf
from unittest.mock import patch

from django.conf import settings as django_settings
from django.contrib.auth.models import User
from django.core import mail
from django.core.mail.backends.locmem import EmailBackend as _LocmemEmailBackend
from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings

from solsys_code.management.commands.check_unattended import _owner_mode, cron_line
from solsys_code.models import WatchedProposal

_FAKE_HEARTBEAT_URL = 'https://hc.example/UUID-TEST-CHECK-UNATTENDED'
_FAKE_MAIL_PASSWORD = 'sk-fake-mail-password-check-unattended'  # noqa: S105 -- fixture literal, not a real secret
_FAKE_LCO_API_KEY = 'fake-lco-api-key-check-unattended'


class _FakeDeliveringEmailBackend(_LocmemEmailBackend):
    """A stand-in for "some real, delivering backend" in tests (WR-19, 36-REVIEW.md).

    Behaves exactly like ``locmem`` (inherits ``send_messages()`` unchanged, so
    ``django.core.mail.outbox`` is still populated the same way Django's test runner
    already relies on) but is not one of the dotted paths
    ``_NON_DELIVERING_EMAIL_BACKENDS`` checks for by name -- so ``check_email()`` reports
    it as OK, the same way it would report any real third-party SMTP-backed backend.
    """


_FAKE_DELIVERING_BACKEND_PATH = f'{__name__}.{_FakeDeliveringEmailBackend.__qualname__}'


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
            # WR-19 (36-REVIEW.md): Django's test runner swaps EMAIL_BACKEND to `locmem`
            # for the whole suite -- which is now itself one of the non-delivering
            # backends this check must fail on. Override it here to a stand-in that
            # behaves exactly like locmem (mail.outbox still works) but is not one of
            # the four dotted paths the check knows by name, so "every hard check
            # passes by default" continues to hold; individual tests below override it
            # back to each non-delivering backend explicitly.
            EMAIL_BACKEND=_FAKE_DELIVERING_BACKEND_PATH,
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

    def test_flock_probe_raising_oserror_fails_cleanly(self):
        # WR-20 (36-REVIEW.md): a dangling symlink target, a noexec mount, or a
        # TOCTOU delete between shutil.which() and subprocess.run() can raise OSError
        # out of the -E probe. Because check_flock() is the first check run, an
        # uncaught OSError here would abort the whole command -- losing every other
        # check's result and the printed cron line -- instead of degrading to a
        # reported detail, which is what this read-only preflight must always do.
        with patch('solsys_code.management.commands.check_unattended.subprocess.run', side_effect=OSError('boom')):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('flock', str(ctx.exception))

    def test_flock_outside_system_directories_gets_a_sanity_note(self):
        # IN-23 (36-REVIEW.md): shutil.which('flock') resolves against the preflight
        # process's own PATH -- a stale or user-writable directory early in PATH (a
        # conda/venv bin, a ~/bin) could resolve a non-system flock that then gets
        # pasted into a persistent, scheduled crontab entry. WR-35 (36-REVIEW.md): the
        # note must render as [WARN] and reach stderr -- an [ok] line (the previous
        # behavior) is invisible to both a "grep FAIL/WARN" scan and anything watching
        # stderr, exactly the workflow this note exists to catch. Still advisory only
        # (it works and supports -E): the command must not exit non-zero for it.
        fake_probe = subprocess.CompletedProcess(args=[], returncode=0, stdout='--conflict-exit-code', stderr='')
        with (
            patch(
                'solsys_code.management.commands.check_unattended.shutil.which',
                return_value='/home/operator/.conda/envs/fomo/bin/flock',
            ),
            patch('solsys_code.management.commands.check_unattended.subprocess.run', return_value=fake_probe),
        ):
            stdout, stderr = _run()
        self.assertIn('[WARN] flock', stdout)
        self.assertIn('outside the usual system directories', stdout)
        self.assertIn('[WARN] flock', stderr)

    def test_flock_in_usr_bin_gets_no_sanity_note(self):
        with patch(
            'solsys_code.management.commands.check_unattended.subprocess.run',
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout='--conflict-exit-code', stderr=''),
        ):
            stdout, _stderr = _run()
        flock_line = next(line for line in stdout.splitlines() if line.startswith('[ok] flock'))
        self.assertNotIn('outside the usual system directories', flock_line)

    def test_flock_probe_timing_out_fails_cleanly(self):
        # WR-20 (36-REVIEW.md): a flock binary on a stalled NFS mount could otherwise
        # hang the preflight indefinitely.
        with patch(
            'solsys_code.management.commands.check_unattended.subprocess.run',
            side_effect=subprocess.TimeoutExpired(cmd=['flock', '--help'], timeout=5),
        ):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('flock', str(ctx.exception))

    @skipIf(os.geteuid() == 0, 'unwritable-directory tests are meaningless as root')
    def test_unwritable_lock_dir_fails(self):
        readonly_parent = self._make_unwritable_parent()
        with override_settings(FOMO_LOCK_DIR=str(readonly_parent / 'sublock')):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('FOMO_LOCK_DIR', str(ctx.exception))

    @skipIf(os.geteuid() == 0, 'unwritable-directory tests are meaningless as root')
    def test_unwritable_log_dir_fails(self):
        readonly_parent = self._make_unwritable_parent()
        with override_settings(FOMO_LOG_FILE=str(readonly_parent / 'sublog' / 'unattended.log')):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('FOMO_LOG_FILE', str(ctx.exception))

    def test_console_email_backend_fails(self):
        with override_settings(EMAIL_BACKEND='django.core.mail.backends.console.EmailBackend'):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('EMAIL_BACKEND', str(ctx.exception))

    def test_dummy_email_backend_fails(self):
        # WR-19 (36-REVIEW.md): `dummy` is the canonical "turn email off" idiom and a
        # realistic production setting -- it previously passed this check (and
        # --send-test-email "succeeded" against it, since dummy.EmailBackend.
        # send_messages() returns len(email_messages) without sending anything).
        with override_settings(EMAIL_BACKEND='django.core.mail.backends.dummy.EmailBackend'):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('EMAIL_BACKEND', str(ctx.exception))

    def test_locmem_email_backend_fails(self):
        # WR-19 (36-REVIEW.md): what a half-finished local_settings.py copied from a
        # test config carries -- previously passed this check.
        with override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend'):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('EMAIL_BACKEND', str(ctx.exception))

    def test_filebased_email_backend_fails(self):
        # WR-19 (36-REVIEW.md): writes to a local file nobody reads -- previously
        # passed this check.
        with tempfile.TemporaryDirectory() as file_backend_dir:
            with override_settings(
                EMAIL_BACKEND='django.core.mail.backends.filebased.EmailBackend',
                EMAIL_FILE_PATH=file_backend_dir,
            ):
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
        self.assertIn('[ok] FOMO_STATE_DIR', stdout)
        self.assertIn('[ok] EMAIL_BACKEND', stdout)
        self.assertIn('[ok] staff_recipients', stdout)

    @skipIf(os.geteuid() == 0, 'unwritable-directory tests are meaningless as root')
    def test_unwritable_state_dir_fails(self):
        # WR-15 (36-REVIEW.md): an unwritable FOMO_STATE_DIR must be caught here, before
        # it turns into the every-15-minutes duplicate-failure-email loop an unpersistable
        # state file causes at runtime.
        readonly_parent = self._make_unwritable_parent()
        with override_settings(FOMO_STATE_DIR=str(readonly_parent / 'substate')):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('FOMO_STATE_DIR', str(ctx.exception))

    def test_flock_without_conflict_exit_code_support_fails(self):
        # WR-11 (36-REVIEW.md): an older flock (util-linux < 2.27) has no -E option --
        # the cron line's whole skip-detection scheme silently no-ops on such a host, so
        # this must be a hard failure, not just a PATH lookup.
        with patch(
            'solsys_code.management.commands.check_unattended.subprocess.run',
            return_value=type('Probe', (), {'stdout': 'Usage: flock [options] ...', 'stderr': ''})(),
        ):
            with self.assertRaises(CommandError) as ctx:
                _run()
        self.assertIn('flock', str(ctx.exception))

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

    def test_none_lock_dir_falls_back_to_the_documented_default_instead_of_raising(self):
        # IN-14 (36-REVIEW.md): a local_settings.py deriving FOMO_LOCK_DIR from an unset
        # environment variable with no default of its own yields None, and Path(None)
        # would raise TypeError from inside check_lock_dir() and cron_line() before this
        # fix -- an uncaught traceback out of a read-only reporting command, not the
        # reported failure a bad prerequisite should produce.
        fallback_dir = tempfile.TemporaryDirectory()
        self.addCleanup(fallback_dir.cleanup)
        with (
            override_settings(FOMO_LOCK_DIR=None),
            patch('solsys_code.management.commands.check_unattended._DEFAULT_LOCK_DIR', fallback_dir.name),
        ):
            stdout, _stderr = _run()  # must not raise TypeError
        self.assertIn('[ok] FOMO_LOCK_DIR', stdout)
        self.assertIn(fallback_dir.name, stdout)


class TestOwnerModeRobustness(TestCase):
    """IN-14 (36-REVIEW.md): `_owner_mode()`'s `stat()` call is not atomic with its
    caller's own `path.exists()` check -- a concurrent delete or an `EACCES` on a parent
    directory between the two must degrade to a reported detail, never an uncaught
    traceback out of this read-only preflight."""

    def test_stat_oserror_reports_unavailable_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            with patch.object(Path, 'stat', side_effect=OSError('stat failed')):
                self.assertEqual(_owner_mode(path), 'owner/mode unavailable')


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

    def test_missing_facility_credentials_are_a_warning(self):
        # WR-31 (36-REVIEW.md): the fresh-host runbook names the LCO/SOAR api_key as a
        # prerequisite, but nothing checked it until this check existed -- a green
        # preflight followed by status_refresh failing on every non-terminal record.
        original_lco = django_settings.FACILITIES['LCO'].get('api_key')
        original_soar = django_settings.FACILITIES['SOAR'].get('api_key')
        django_settings.FACILITIES['LCO']['api_key'] = ''
        django_settings.FACILITIES['SOAR']['api_key'] = ''
        try:
            stdout, stderr = _run()
        finally:
            django_settings.FACILITIES['LCO']['api_key'] = original_lco
            django_settings.FACILITIES['SOAR']['api_key'] = original_soar
        self.assertIn('[WARN] facility_credentials', stdout)
        self.assertIn('LCO', stdout)
        self.assertIn('SOAR', stdout)
        self.assertIn('facility_credentials', stderr)

    def test_configured_facility_credentials_are_ok(self):
        original_lco = django_settings.FACILITIES['LCO'].get('api_key')
        original_soar = django_settings.FACILITIES['SOAR'].get('api_key')
        django_settings.FACILITIES['LCO']['api_key'] = _FAKE_LCO_API_KEY
        django_settings.FACILITIES['SOAR']['api_key'] = _FAKE_LCO_API_KEY
        try:
            stdout, _stderr = _run()
        finally:
            django_settings.FACILITIES['LCO']['api_key'] = original_lco
            django_settings.FACILITIES['SOAR']['api_key'] = original_soar
        self.assertIn('[ok] facility_credentials', stdout)
        self.assertNotIn(_FAKE_LCO_API_KEY, stdout)

    def test_set_heartbeat_reminds_about_the_check_period(self):
        # G-36-3: the runbook once named only the check's grace time, so an operator
        # left the check's own expected ping interval (Period) at its 1-day default
        # and never got an alert. This reminder is the last line of defense against
        # that regenerating -- keep it if this check is ever refactored.
        #
        # IN-20 (36-REVIEW.md): also asserts Grace, not Period alone -- the detail
        # previously reminded about only one of the two knobs the runbook and crontab
        # template both document, so an operator following the preflight's reminder
        # alone left Grace at healthchecks.io's 1-hour default.
        with override_settings(FOMO_HEARTBEAT_URL=_FAKE_HEARTBEAT_URL):
            stdout, _stderr = _run()
        self.assertIn('[ok] heartbeat', stdout)
        self.assertIn('Period', stdout)
        self.assertIn('Grace', stdout)
        self.assertNotIn(_FAKE_HEARTBEAT_URL, stdout)


class TestCronLine(CheckUnattendedTestBase):
    def _assert_lock_held_exit_matrix_is_normalized(self, line: str) -> None:
        """Shared 0/1/99 matrix: splice a stub in place of the real `flock ...
        run_unattended` invocation in ``line``, keeping its own skip-tail/exit logic
        verbatim, and run it in a real shell.

        Anchor on ' run_unattended >>' specifically (not the bare command name), which
        also appears inside the lock file path (`run_unattended.cron.lock`) and inside
        the skip line's own echoed text ("run_unattended skipped: lock held") -- only
        the real invocation is immediately followed by ' >>'.
        """
        _head, sep, tail = line.partition(' run_unattended >>')
        self.assertTrue(sep, 'expected exactly one " run_unattended >>" in the line')
        for stub_exit, expected_final_exit in ((0, 0), (1, 1), (99, 0)):
            with self.subTest(stub_exit=stub_exit):
                # The stub's extra positional argument ("run_unattended") is harmless --
                # `sh -c "exit N" $0 ...` ignores it, since "exit N" never references $0.
                script = f'sh -c "exit {stub_exit}"{sep}{tail}'
                result = subprocess.run(['sh', '-c', script], check=False)
                self.assertEqual(result.returncode, expected_final_exit)

    def test_lock_held_exit_is_normalized_to_zero(self):
        # WR-16 (36-REVIEW.md): run_tick()'s own contract is that lock contention is NOT
        # a failure -- it returns exit_code=0, and the heartbeat (D-12) is the
        # structural backstop. Before this fix, a lock-held skip made the WHOLE cron
        # line exit 99, which any supervisor (cron's own syslog line, an OnFailure=
        # hook, a monitoring wrapper) reads as a failure on a routine tick overlap.
        # Exercises the actual shipped tail in a real shell, not a hand-copied
        # re-implementation.
        self._assert_lock_held_exit_matrix_is_normalized(cron_line())

    def test_committed_template_lock_held_exit_is_normalized_to_zero(self):
        # WR-37 (36-REVIEW.md): the previous version of this test only ever exercised
        # cron_line()'s OWN generated output -- never the committed
        # deploy/cron/fomo.crontab.example line an operator might instead copy-paste
        # directly. The WR-16 rc=0 normalization could regress in the committed template
        # alone (e.g. losing the `{ ... ; rc=0; }` grouping) and every test would still
        # pass, in a phase whose CR-01/WR-01/WR-09 history is entirely about these two
        # artifacts drifting apart. Runs the identical 0/1/99 matrix against the
        # template's own `*/15` line, read from disk.
        template_path = Path(django_settings.BASE_DIR).parent / 'deploy' / 'cron' / 'fomo.crontab.example'
        template_line = next(
            (
                stripped_line
                for raw_line in template_path.read_text().splitlines()
                if (stripped_line := raw_line.strip()).startswith('*/15')
            ),
            None,
        )
        self.assertIsNotNone(template_line, f'no */15 line found in {template_path}')
        self._assert_lock_held_exit_matrix_is_normalized(template_line)

    def test_line_has_real_paths(self):
        line = cron_line()
        self.assertIn(sys.executable, line)
        manage_py_path = str(Path(django_settings.BASE_DIR).parent / 'manage.py')
        self.assertIn(manage_py_path, line)
        self.assertNotIn('/path/to/venv/bin/python', line)
        self.assertNotIn('/path/to/checkout/manage.py', line)

    def test_line_matches_the_committed_template_shape(self):
        # IN-08 (36-REVIEW.md): guard against a host with no `flock` on PATH -- `shutil.
        # which('flock')` would then return None, and the un-guarded f-string built an
        # assertion for the literal 'None -n -E 99', which can never match `cron_line()`'s
        # `/usr/bin/flock` fallback and fails the test for a reason unrelated to what it
        # actually checks.
        flock_path = shutil.which('flock') or '/usr/bin/flock'
        line = cron_line()
        for element in (
            '*/15 * * * *',
            f'{flock_path} -n -E 99',
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

    def test_line_matches_the_committed_template_token_for_token(self):
        # IN-08 (36-REVIEW.md): the previous version of this test's name promised
        # agreement with deploy/cron/fomo.crontab.example but never actually read it,
        # comparing only a hand-maintained fragment list instead -- drift between the two
        # is exactly what CR-01/WR-01/WR-09 were about. Read the committed file's own
        # `*/15` line and compare option tokens, ignoring the two host-specific
        # placeholder paths this test doesn't resolve.
        # IN-23 (36-REVIEW.md): resolve the repo root from settings.BASE_DIR (the same
        # way cron_line() itself resolves manage.py's path) rather than
        # Path(__file__).resolve().parents[2], which silently assumes this test file's
        # own depth below the repo root and breaks if the file is ever moved.
        template_path = Path(django_settings.BASE_DIR).parent / 'deploy' / 'cron' / 'fomo.crontab.example'
        template_line = next(
            (
                stripped_line
                for raw_line in template_path.read_text().splitlines()
                if (stripped_line := raw_line.strip()).startswith('*/15')
            ),
            None,
        )
        self.assertIsNotNone(template_line, f'no */15 line found in {template_path}')
        line = cron_line()
        for token in (
            '-n',
            '-E 99',
            '.cron.lock',
            '>>',
            '2>&1',
            'rc=$?',
            '[ $rc -eq 99 ]',
            'lock held',
            # WR-37 (36-REVIEW.md): the previous token list did not cover WR-16's rc=0
            # normalization or the `{ ... ; }` grouping that makes it work -- the
            # committed template could lose either and this test would still pass.
            'rc=0',
            '; }',
            'exit $rc',
        ):
            self.assertIn(token, template_line, f'{token!r} missing from the committed template line')
            self.assertIn(token, line, f'{token!r} missing from cron_line()')

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
