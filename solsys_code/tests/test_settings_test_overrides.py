"""Tests for the test-only overrides in ``src/fomo/settings.py`` (Phase 39.1: the cache
override for SPEED-03 and the fast MD5 password hasher for SPEED-04).

``src/fomo/settings.py`` swaps in test-only overrides only when Django's test command is
the entry point (``sys.argv[1] == 'test'``). These tests prove the live settings under
this test run, and execute the settings file afresh under other command lines to prove
production and runserver settings are unchanged -- in particular the weak MD5 test hasher
must never reach a web, shell or WSGI process.

Uses ``django.test.SimpleTestCase`` -- no database row is touched anywhere in this module.
No ``Target`` fixture is created, so the ``NonSiderealTargetFactory`` rule in CLAUDE.md
does not arise.
"""

import importlib
import runpy
import sys
import tempfile
import types
from unittest import mock

from django.conf import global_settings, settings
from django.contrib.auth.hashers import make_password
from django.core.cache import caches
from django.core.cache.backends.locmem import LocMemCache
from django.test import SimpleTestCase

LOCMEM_BACKEND = 'django.core.cache.backends.locmem.LocMemCache'
FILE_BACKEND = 'django.core.cache.backends.filebased.FileBasedCache'
MD5_HASHER = 'django.contrib.auth.hashers.MD5PasswordHasher'
PBKDF2_HASHER = 'django.contrib.auth.hashers.PBKDF2PasswordHasher'

# Command lines whose argv[1] is Django's 'test' subcommand: the test-only block applies.
TEST_COMMAND_ARGVS = (
    ['manage.py', 'test'],
    ['manage.py', 'test', 'solsys_code.tests.test_urls'],
    ['/venv/lib/python3.11/site-packages/django/__main__.py', 'test'],
)

# One step either side of 'test': runserver/shell never match; 'testserver' and 'tests'
# share its prefix; a bare manage.py and a WSGI server have no subcommand.
OTHER_COMMAND_ARGVS = (
    ['manage.py', 'runserver'],
    ['manage.py', 'shell'],
    ['manage.py', 'testserver'],
    ['manage.py', 'tests'],
    ['manage.py'],
    ['gunicorn', 'fomo.wsgi:application'],
)

# The exact anchor the settings module's fold tail begins with (same value as
# test_settings_api_key_fold): the test-only block must sit above it.
_FOLD_TAIL_ANCHOR = 'try:\n    from fomo.local_settings import *'

# The guard line that selects the test-only block.
_TEST_BLOCK_GUARD = "if len(sys.argv) > 1 and sys.argv[1] == 'test':"


class TestTestCommandSettingsInEffect(SimpleTestCase):
    """The live settings, under this very test run, carry the test-only overrides."""

    def test_default_cache_is_a_per_process_locmem_cache(self):
        self.assertEqual(settings.CACHES['default']['BACKEND'], LOCMEM_BACKEND)
        self.assertIsInstance(caches['default'], LocMemCache)

    def test_password_hasher_is_md5_under_the_test_command(self):
        # Django's documented "speeding up the tests" setting: test users hash with MD5, not
        # PBKDF2 at 1,000,000 iterations.
        self.assertEqual(settings.PASSWORD_HASHERS, [MD5_HASHER])
        self.assertTrue(make_password('probe').startswith('md5$'))


class TestSettingsOverridesOnlyUnderTestCommand(SimpleTestCase):
    """Executing the settings file afresh applies the overrides under the test command only."""

    def _settings_namespace(self, argv):
        """Execute the live settings file under ``argv`` and return its namespace.

        An empty synthetic ``fomo.local_settings`` module is injected into ``sys.modules``
        (restored with ``addCleanup``) so the operator's real local settings never
        influence the result.
        """
        settings_module = importlib.import_module(settings.SETTINGS_MODULE)

        previous_module = sys.modules.get('fomo.local_settings')
        sys.modules['fomo.local_settings'] = types.ModuleType('fomo.local_settings')

        def _restore_module():
            if previous_module is None:
                sys.modules.pop('fomo.local_settings', None)
            else:
                sys.modules['fomo.local_settings'] = previous_module

        self.addCleanup(_restore_module)

        with mock.patch.object(sys, 'argv', argv):
            return runpy.run_path(settings_module.__file__, run_name='fomo_settings_probe')

    def test_test_command_selects_the_locmem_cache(self):
        for argv in TEST_COMMAND_ARGVS:
            with self.subTest(argv=argv):
                namespace = self._settings_namespace(argv)
                self.assertEqual(namespace['CACHES']['default']['BACKEND'], LOCMEM_BACKEND)

    def test_test_command_selects_the_md5_hasher(self):
        for argv in TEST_COMMAND_ARGVS:
            with self.subTest(argv=argv):
                namespace = self._settings_namespace(argv)
                self.assertEqual(namespace.get('PASSWORD_HASHERS'), [MD5_HASHER])

    def test_other_commands_keep_the_shared_file_cache(self):
        for argv in OTHER_COMMAND_ARGVS:
            with self.subTest(argv=argv):
                namespace = self._settings_namespace(argv)
                self.assertEqual(namespace['CACHES']['default']['BACKEND'], FILE_BACKEND)
                self.assertEqual(namespace['CACHES']['default']['LOCATION'], tempfile.gettempdir())

    def test_other_commands_keep_the_production_hashers(self):
        # No MD5 hasher anywhere in the executed settings, and the effective first hasher is
        # Django's own default (PBKDF2), whether PASSWORD_HASHERS is unset or set explicitly.
        for argv in OTHER_COMMAND_ARGVS:
            with self.subTest(argv=argv):
                namespace = self._settings_namespace(argv)
                self.assertNotIn(MD5_HASHER, namespace.get('PASSWORD_HASHERS', []))
                effective = namespace.get('PASSWORD_HASHERS') or global_settings.PASSWORD_HASHERS
                self.assertEqual(effective[0], PBKDF2_HASHER)

    def test_block_sits_above_the_local_settings_fold_tail(self):
        # test_settings_api_key_fold executes the file from the fold-tail anchor to the end
        # in a bare namespace, so the test-only block must stay above that anchor.
        settings_module = importlib.import_module(settings.SETTINGS_MODULE)
        with open(settings_module.__file__) as fh:
            source = fh.read()
        guard_index = source.find(_TEST_BLOCK_GUARD)
        anchor_index = source.find(_FOLD_TAIL_ANCHOR)
        self.assertNotEqual(guard_index, -1, 'test-only block guard not found in settings')
        self.assertNotEqual(anchor_index, -1, 'fold-tail anchor not found in settings')
        self.assertLess(guard_index, anchor_index)
