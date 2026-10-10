"""Tests that run the shared notebook scratch-database helper, ``docs/notebooks/fomo_nb_utils.py``.

Phase 40 (D-11) moved every pre-executed notebook's database setup out of eight hand-copied setup
cells and into one helper, so its routing, read-only snapshot and refusal behavior is proven here by
running the real helper -- replacing the old guard's source-pattern checks of each setup cell.

Pure parts (snapshot, path check, missing-database refusal, teardown refusal) run in this process
against throwaway SQLite files. The helper imports Django lazily (D-07), so loading it here neither
touches the app registry nor the test database. Anything that really calls ``django.setup()`` runs in
a subprocess: the test process is already configured against the in-memory test database, and a
second ``setup()`` there would test nothing. Every test owns its temp directory, so parallel workers
never share one.
"""

import ast
import hashlib
import importlib.util
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

_REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = _REPO_ROOT / 'docs' / 'notebooks' / 'fomo_nb_utils.py'

_SCRATCH_PREFIX = 'fomo-notebook-db-'


def _load_helper():
    """Load the helper file as a module without touching ``sys.path``.

    Returns:
        module: the executed helper module (not registered in ``sys.modules``).
    """
    spec = importlib.util.spec_from_file_location('fomo_nb_utils_under_test', HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_helper_script(script, env_overrides=None, python_args=()):
    """Run ``script`` with ``python -c`` in a fresh interpreter, cwd at the repo root.

    The environment is a copy of this process's with ``FOMO_DATABASE_PATH`` removed, so an inherited
    value can never decide where the child writes.

    Args:
        script: Python source passed to ``-c``.
        env_overrides: mapping of environment variables to set in the child.
        python_args: extra interpreter flags placed before ``-c`` (for example ``('-O',)``).

    Returns:
        subprocess.CompletedProcess: with captured text ``stdout`` and ``stderr``.
    """
    env = {key: value for key, value in os.environ.items() if key != 'FOMO_DATABASE_PATH'}
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, *python_args, '-c', script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_sqlite_file(path):
    """Create a tiny SQLite file with one table and one row, standing in for a developer database."""
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE marker (id INTEGER PRIMARY KEY, label TEXT)')
        connection.execute("INSERT INTO marker (label) VALUES ('developer-row')")
        connection.commit()
    finally:
        connection.close()


def _write_override_settings(directory, extra_source):
    """Write a settings module that extends ``src.fomo.settings`` and return its module name.

    Args:
        directory: directory to write ``override_settings_for_helper_test.py`` into; put it on the
            child's ``PYTHONPATH`` and name the module in ``DJANGO_SETTINGS_MODULE``.
        extra_source: Python source appended after the ``from src.fomo.settings import *`` line.

    Returns:
        str: the module name, ``'override_settings_for_helper_test'``.
    """
    (Path(directory) / 'override_settings_for_helper_test.py').write_text(
        'from src.fomo.settings import *  # noqa: F403\n' + extra_source,
        encoding='utf-8',
    )
    return 'override_settings_for_helper_test'


def _refusal_probe_script():
    """Return child-process source that runs the setup, then probes what the interpreter can still reach.

    The child prints ``REFUSED <message>`` when the setup refuses, then ``APPS_READY``, the two routing
    variables, one ``WRITE_SUCCEEDED <alias>`` or ``WRITE_REFUSED <alias> <exception class>`` line per
    configured alias (a raw ``CREATE TABLE`` through that alias's cursor), and finally whether a second
    setup call is refused.

    Returns:
        str: the script text, for ``_run_helper_script``.
    """
    return (
        'import os, sys\n'
        'os.environ.pop("DJANGO_ALLOW_ASYNC_UNSAFE", None)\n'
        f'sys.path.insert(0, {str(HELPER_PATH.parent)!r})\n'
        'from fomo_nb_utils import setup_scratch_django\n'
        'try:\n'
        f'    setup_scratch_django({str(_REPO_ROOT)!r}, mode="fresh")\n'
        'except RuntimeError as exc:\n'
        '    print("REFUSED", exc)\n'
        'import django.apps\n'
        'print("APPS_READY", django.apps.apps.ready)\n'
        'print("FOMO_DATABASE_PATH_AFTER", repr(os.environ.get("FOMO_DATABASE_PATH")))\n'
        'print("ASYNC_UNSAFE_AFTER", repr(os.environ.get("DJANGO_ALLOW_ASYNC_UNSAFE")))\n'
        'from django.db import connections\n'
        'for alias in list(connections):\n'
        '    try:\n'
        '        with connections[alias].cursor() as cursor:\n'
        '            cursor.execute("CREATE TABLE written_after_refusal (id INTEGER)")\n'
        '        print("WRITE_SUCCEEDED", alias)\n'
        '    except Exception as exc:\n'
        '        print("WRITE_REFUSED", alias, type(exc).__name__)\n'
        'try:\n'
        f'    setup_scratch_django({str(_REPO_ROOT)!r}, mode="fresh")\n'
        '    print("SECOND_SETUP_SUCCEEDED")\n'
        'except RuntimeError as exc:\n'
        '    print("SECOND_SETUP_REFUSED", exc)\n'
    )


class TestNotebookHelperModule(SimpleTestCase):
    """Static and import-time properties of the helper file."""

    def test_importing_the_module_does_not_import_django(self):
        script = (
            'import sys\n'
            f'sys.path.insert(0, {str(HELPER_PATH.parent)!r})\n'
            'import fomo_nb_utils\n'
            'bad = sorted(n for n in sys.modules if n in ("django", "solsys_code") '
            'or n.startswith(("django.", "solsys_code.", "tom_")))\n'
            'print(bad)\n'
        )
        result = _run_helper_script(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '[]')

    def test_module_has_no_assert_statements(self):
        tree = ast.parse(HELPER_PATH.read_text(encoding='utf-8'))
        self.assertEqual([node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)], [])

    def test_module_level_imports_are_standard_library_or_astropy(self):
        tree = ast.parse(HELPER_PATH.read_text(encoding='utf-8'))
        allowed = {'os', 'shutil', 'sqlite3', 'tempfile', 'pathlib', 'astropy'}
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            else:
                continue
            for name in names:
                self.assertIn(name.split('.')[0], allowed, f'module-level import of {name!r}')

    def test_special_locations_still_returns_the_eight_rubin_fields(self):
        helper = _load_helper()
        locations = helper.special_locations()
        self.assertEqual(len(locations), 8)
        self.assertIn('ELAISS1', locations)
        self.assertIn('M49 (Virgo)', locations)
        # The seed writes this declination with a U+2212 minus sign; it must parse as negative.
        self.assertLess(locations['ELAISS1'][1], -44)


class TestSnapshotAndRouting(SimpleTestCase):
    """The helper's pure parts, run in this process against throwaway files."""

    def test_snapshot_copies_rows_and_leaves_the_source_untouched(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.sqlite3'
            target = Path(tmp) / 'target.sqlite3'
            _make_sqlite_file(source)
            before = _sha256(source)
            helper.snapshot_database(source, target)
            connection = sqlite3.connect(target)
            try:
                rows = connection.execute('SELECT label FROM marker').fetchall()
            finally:
                connection.close()
            self.assertEqual(rows, [('developer-row',)])
            self.assertEqual(_sha256(source), before)

    def test_snapshot_opens_the_source_read_only(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.sqlite3'
            target = Path(tmp) / 'target.sqlite3'
            _make_sqlite_file(source)
            with mock.patch.object(helper.sqlite3, 'connect', wraps=sqlite3.connect) as connect:
                helper.snapshot_database(source, target)
            first_call = connect.call_args_list[0]
            self.assertTrue(str(first_call.args[0]).endswith('?mode=ro'), first_call)
            self.assertIs(first_call.kwargs.get('uri'), True)

    def test_snapshot_mode_without_a_developer_database_raises_before_creating_anything(self):
        helper = _load_helper()
        before = os.environ.get('FOMO_DATABASE_PATH')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(helper.tempfile, 'mkdtemp') as mkdtemp:
            with self.assertRaises(RuntimeError):
                helper.prepare_scratch_database(tmp, mode='snapshot')
        mkdtemp.assert_not_called()
        self.assertEqual(os.environ.get('FOMO_DATABASE_PATH'), before)

    def test_unknown_mode_raises_before_creating_anything(self):
        helper = _load_helper()
        before = os.environ.get('FOMO_DATABASE_PATH')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(helper.tempfile, 'mkdtemp') as mkdtemp:
            with self.assertRaises(ValueError):
                helper.prepare_scratch_database(tmp, mode='copy')
        mkdtemp.assert_not_called()
        self.assertEqual(os.environ.get('FOMO_DATABASE_PATH'), before)

    def test_resolved_path_check_accepts_the_same_file_spelled_another_way(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / 'fomo_db.sqlite3'
            scratch.write_bytes(b'')
            (Path(tmp) / 'sub').mkdir()
            link = Path(tmp) / 'link.sqlite3'
            link.symlink_to(scratch)
            helper.check_resolved_database(str(link), scratch)
            helper.check_resolved_database(str(Path(tmp) / 'sub' / '..' / 'fomo_db.sqlite3'), scratch)
            helper.check_resolved_database(str(scratch), scratch)

    def test_resolved_path_check_raises_runtime_error_for_any_other_file(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / 'fomo_db.sqlite3'
            other = Path(tmp) / 'other.sqlite3'
            scratch.write_bytes(b'')
            other.write_bytes(b'')
            for resolved in (str(other), ''):
                with self.subTest(resolved=resolved):
                    with self.assertRaises(RuntimeError) as caught:
                        helper.check_resolved_database(resolved, scratch)
                    self.assertNotIsInstance(caught.exception, AssertionError)

    def test_setup_refuses_when_django_is_already_set_up(self):
        """Refuse before anything is created, and never disable a Django the helper did not start."""
        from django.conf import settings
        from django.db import connections

        helper = _load_helper()
        before = os.environ.get('FOMO_DATABASE_PATH')
        default_before = dict(settings.DATABASES['default'])
        engine_before = connections['default'].settings_dict['ENGINE']
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(helper.tempfile, 'mkdtemp') as mkdtemp:
            with self.assertRaises(RuntimeError):
                helper.setup_scratch_django(tmp)
        mkdtemp.assert_not_called()
        self.assertEqual(os.environ.get('FOMO_DATABASE_PATH'), before)
        self.assertEqual(settings.DATABASES['default']['ENGINE'], default_before['ENGINE'])
        self.assertEqual(settings.DATABASES['default']['NAME'], default_before['NAME'])
        self.assertEqual(connections['default'].settings_dict['ENGINE'], engine_before)

    def test_teardown_refuses_a_directory_without_the_scratch_prefix(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory(prefix='other-') as tmp:
            with self.assertRaises(RuntimeError):
                helper.teardown_scratch_django(tmp)
            self.assertTrue(Path(tmp).is_dir())

    def test_a_failed_snapshot_removes_the_scratch_directory(self):
        helper = _load_helper()
        real_mkdtemp = tempfile.mkdtemp
        created = []

        def recording_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created.append(Path(path))
            return path

        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'src').mkdir()
            _make_sqlite_file(Path(root) / 'src' / 'fomo_db.sqlite3')
            with (
                mock.patch.object(helper.tempfile, 'mkdtemp', side_effect=recording_mkdtemp),
                mock.patch.object(
                    helper, 'snapshot_database', side_effect=sqlite3.OperationalError('forced snapshot failure')
                ),
            ):
                with self.assertRaises(sqlite3.OperationalError):
                    helper.prepare_scratch_database(root, mode='snapshot')
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].name.startswith(_SCRATCH_PREFIX), created[0])
        self.assertFalse(created[0].exists(), 'the failed snapshot left its scratch directory behind')


class TestDatabaseSettingsCheck(SimpleTestCase):
    """WR-02 (40-REVIEW.md): the pure check that refuses a second alias or a router before Django starts.

    Developer decision of 2026-10-09: a notebook cannot prove that every write reaches the scratch
    file once a second database alias or a database router exists, so both are refused rather than
    supported. The check runs on the settings values alone, in this process.
    """

    def test_accepts_only_the_scratch_default_database(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / 'fomo_db.sqlite3'
            scratch.write_bytes(b'')
            databases = {'default': {'NAME': str(scratch)}}
            helper.check_database_settings(databases, [], scratch)
            helper.check_database_settings(databases, (), scratch)

    def test_refuses_a_second_alias_even_when_it_names_the_scratch_file(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / 'fomo_db.sqlite3'
            scratch.write_bytes(b'')
            databases = {'default': {'NAME': str(scratch)}, 'other': {'NAME': str(scratch)}}
            with self.assertRaises(RuntimeError) as caught:
                helper.check_database_settings(databases, [], scratch)
            self.assertIn("'other'", str(caught.exception))

    def test_refuses_database_routers(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / 'fomo_db.sqlite3'
            scratch.write_bytes(b'')
            with self.assertRaises(RuntimeError) as caught:
                helper.check_database_settings({'default': {'NAME': str(scratch)}}, ['x.Router'], scratch)
            self.assertIn('DATABASE_ROUTERS', str(caught.exception))

    def test_refuses_a_missing_or_empty_default(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / 'fomo_db.sqlite3'
            scratch.write_bytes(b'')
            for databases in ({}, {'default': {}}, {'default': {'NAME': ''}}):
                with self.subTest(databases=databases):
                    with self.assertRaises(RuntimeError) as caught:
                        helper.check_database_settings(databases, [], scratch)
                    self.assertNotIsInstance(caught.exception, AssertionError)

    def test_names_the_extra_alias_before_the_routers(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / 'fomo_db.sqlite3'
            scratch.write_bytes(b'')
            databases = {'default': {'NAME': str(scratch)}, 'zeta': {'NAME': 'z'}, 'alpha': {'NAME': 'a'}}
            with self.assertRaises(RuntimeError) as caught:
                helper.check_database_settings(databases, ['x.Router'], scratch)
            message = str(caught.exception)
            self.assertIn("'alpha'", message)
            self.assertIn("'zeta'", message)
            self.assertLess(message.index("'alpha'"), message.index("'zeta'"))
            self.assertNotIn('DATABASE_ROUTERS', message)


class TestHelperRunsForReal(SimpleTestCase):
    """Subprocess runs of the real helper, each in a fresh interpreter and a test-owned temp directory."""

    def test_fresh_setup_routes_migrates_and_tears_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                'import sqlite3, sys\n'
                'from pathlib import Path\n'
                f'sys.path.insert(0, {str(HELPER_PATH.parent)!r})\n'
                'from fomo_nb_utils import setup_scratch_django, teardown_scratch_django\n'
                f'scratch = setup_scratch_django({str(_REPO_ROOT)!r}, mode="fresh")\n'
                'connection = sqlite3.connect(Path(scratch) / "fomo_db.sqlite3")\n'
                'print("MIGRATION_ROWS", connection.execute("SELECT COUNT(*) FROM django_migrations").fetchone()[0])\n'
                'connection.close()\n'
                'teardown_scratch_django(scratch)\n'
            )
            result = _run_helper_script(script, {'TMPDIR': tmp})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertRegex(
                result.stdout,
                rf"Resolved database: '{re.escape(tmp)}/{_SCRATCH_PREFIX}[^/']+/fomo_db\.sqlite3' "
                r'\(fresh scratch database, migrated from empty\)',
            )
            # IN-02: the evidence line names the settings module and no longer publishes the checkout path.
            self.assertIn("Django ready: settings module='src.fomo.settings'\n", result.stdout)
            self.assertNotIn(str(_REPO_ROOT), result.stdout)
            rows = int(result.stdout.split('MIGRATION_ROWS')[1].split()[0])
            self.assertGreater(rows, 0)
            self.assertIn('Removed scratch database directory:', result.stdout)
            self.assertEqual([p.name for p in Path(tmp).glob(f'{_SCRATCH_PREFIX}*')], [])

    def test_a_databases_override_is_refused_before_migrate(self):
        variants = (
            ('plain', {}, ()),
            ('dash_O', {}, ('-O',)),
            ('PYTHONOPTIMIZE', {'PYTHONOPTIMIZE': '1'}, ()),
        )
        for label, extra_env, python_args in variants:
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                other_db = Path(tmp) / 'other.sqlite3'
                (Path(tmp) / 'override_settings_for_helper_test.py').write_text(
                    'from src.fomo.settings import *  # noqa: F403\n'
                    "DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', "
                    f"'NAME': {str(other_db)!r}}}}}\n",
                    encoding='utf-8',
                )
                script = (
                    'import sys\n'
                    f'sys.path.insert(0, {str(HELPER_PATH.parent)!r})\n'
                    'from fomo_nb_utils import setup_scratch_django\n'
                    f'setup_scratch_django({str(_REPO_ROOT)!r}, mode="fresh")\n'
                )
                env = {
                    'TMPDIR': tmp,
                    'PYTHONPATH': tmp,
                    'DJANGO_SETTINGS_MODULE': 'override_settings_for_helper_test',
                    **extra_env,
                }
                result = _run_helper_script(script, env, python_args)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('RuntimeError', result.stderr)
                self.assertNotIn('Resolved database:', result.stdout)
                self.assertFalse(other_db.exists(), 'the override database was created')
                self.assertEqual([p.name for p in Path(tmp).glob(f'{_SCRATCH_PREFIX}*')], [])


class TestHelperRefusalLeavesNoDatabase(SimpleTestCase):
    """CR-01 (40-REVIEW.md, 40-VERIFICATION.md gap 1): a refusal must leave no database reachable.

    The helper used to call ``django.setup()`` before it checked which database Django resolved, so
    a refusal left Django set up against the refused database and the next notebook cell wrote
    there. Each test runs in a fresh interpreter and a test-owned temp directory, because only a
    child process can show what is reachable after the helper has raised.
    """

    def test_a_refused_setup_leaves_django_unable_to_reach_any_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            other_db = Path(tmp) / 'other.sqlite3'
            module = _write_override_settings(
                tmp,
                f"DATABASES = {{'default': {{'ENGINE': 'django.db.backends.sqlite3', 'NAME': {str(other_db)!r}}}}}\n",
            )
            env = {'TMPDIR': tmp, 'PYTHONPATH': tmp, 'DJANGO_SETTINGS_MODULE': module}
            result = _run_helper_script(_refusal_probe_script(), env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertRegex(result.stdout, r'REFUSED .*is not the scratch database')
            self.assertIn('APPS_READY False', result.stdout)
            self.assertIn('WRITE_REFUSED default ImproperlyConfigured', result.stdout)
            self.assertIn('FOMO_DATABASE_PATH_AFTER None', result.stdout)
            self.assertIn('ASYNC_UNSAFE_AFTER None', result.stdout)
            self.assertRegex(result.stdout, r'SECOND_SETUP_REFUSED .*restart the kernel')
            self.assertNotIn('WRITE_SUCCEEDED', result.stdout)
            self.assertNotIn('SECOND_SETUP_SUCCEEDED', result.stdout)
            self.assertNotIn('Resolved database:', result.stdout)
            self.assertFalse(other_db.exists(), 'the override database was created')
            self.assertEqual([p.name for p in Path(tmp).glob(f'{_SCRATCH_PREFIX}*')], [])

    def test_a_failure_after_django_setup_leaves_no_usable_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                'import os, sys\n'
                'os.environ.pop("DJANGO_ALLOW_ASYNC_UNSAFE", None)\n'
                'import django.core.management\n'
                'def _fail(*args, **kwargs):\n'
                '    raise RuntimeError("migrate failed (forced by the test)")\n'
                'django.core.management.call_command = _fail\n'
                f'sys.path.insert(0, {str(HELPER_PATH.parent)!r})\n'
                'from fomo_nb_utils import setup_scratch_django\n'
                'try:\n'
                f'    setup_scratch_django({str(_REPO_ROOT)!r}, mode="fresh")\n'
                'except RuntimeError as exc:\n'
                '    print("REFUSED", exc)\n'
                'import django.apps\n'
                'print("APPS_READY", django.apps.apps.ready)\n'
                'from django.db import connections\n'
                'try:\n'
                '    with connections["default"].cursor() as cursor:\n'
                '        cursor.execute("CREATE TABLE written_after_failure (id INTEGER)")\n'
                '    print("WRITE_SUCCEEDED default")\n'
                'except Exception as exc:\n'
                '    print("WRITE_REFUSED default", type(exc).__name__)\n'
                'from django.contrib.auth.models import User\n'
                'try:\n'
                '    User.objects.count()\n'
                '    print("ORM_SUCCEEDED")\n'
                'except Exception as exc:\n'
                '    print("ORM_REFUSED", type(exc).__name__)\n'
            )
            result = _run_helper_script(script, {'TMPDIR': tmp})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('REFUSED migrate failed (forced by the test)', result.stdout)
            self.assertIn('APPS_READY True', result.stdout)
            self.assertIn('WRITE_REFUSED default ImproperlyConfigured', result.stdout)
            self.assertIn('ORM_REFUSED ImproperlyConfigured', result.stdout)
            self.assertNotIn('WRITE_SUCCEEDED', result.stdout)
            self.assertNotIn('ORM_SUCCEEDED', result.stdout)
            self.assertNotIn('Resolved database:', result.stdout)
            self.assertEqual([p.name for p in Path(tmp).glob(f'{_SCRATCH_PREFIX}*')], [])

    def test_a_second_alias_or_a_router_is_refused_before_django_setup(self):
        cases = (
            ('second alias', 'other'),
            ('router', 'router'),
        )
        for label, kind in cases:
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                other_db = Path(tmp) / 'other.sqlite3'
                if kind == 'other':
                    extra = (
                        "DATABASES = {**DATABASES, 'other': {'ENGINE': 'django.db.backends.sqlite3', "
                        f"'NAME': {str(other_db)!r}}}}}\n"
                    )
                else:
                    extra = "DATABASE_ROUTERS = ['not_a_module_for_helper_test.Router']\n"
                module = _write_override_settings(tmp, extra)
                env = {'TMPDIR': tmp, 'PYTHONPATH': tmp, 'DJANGO_SETTINGS_MODULE': module}
                result = _run_helper_script(_refusal_probe_script(), env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertRegex(
                    result.stdout, r"REFUSED .*'other'" if kind == 'other' else r'REFUSED .*DATABASE_ROUTERS'
                )
                self.assertIn('APPS_READY False', result.stdout)
                self.assertIn('WRITE_REFUSED default ImproperlyConfigured', result.stdout)
                if kind == 'other':
                    self.assertIn('WRITE_REFUSED other ImproperlyConfigured', result.stdout)
                self.assertNotIn('WRITE_SUCCEEDED', result.stdout)
                self.assertFalse(other_db.exists(), 'the second database was created')
                self.assertEqual([p.name for p in Path(tmp).glob(f'{_SCRATCH_PREFIX}*')], [])


class TestHelperSnapshotRunsForReal(SimpleTestCase):
    """IN-07 helper-test part (D-11: tests run the real helper): snapshot mode, executed for real.

    Snapshot mode had only its pure part tested. This runs it in a child process against a throwaway
    "developer database" in a test-owned root, never the real ``src/fomo_db.sqlite3``, and confirms
    the read-only online backup (D-02) leaves the source bytes unchanged.
    """

    def test_snapshot_setup_copies_the_developer_rows_migrates_and_tears_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'src').mkdir()
            source = Path(tmp) / 'src' / 'fomo_db.sqlite3'
            _make_sqlite_file(source)
            before = _sha256(source)
            script = (
                'import sqlite3, sys\n'
                'from pathlib import Path\n'
                f'sys.path.insert(0, {str(HELPER_PATH.parent)!r})\n'
                'from fomo_nb_utils import setup_scratch_django, teardown_scratch_django\n'
                f'scratch = setup_scratch_django({tmp!r}, mode="snapshot")\n'
                'connection = sqlite3.connect(Path(scratch) / "fomo_db.sqlite3")\n'
                'print("MARKER_ROWS", connection.execute("SELECT label FROM marker").fetchall())\n'
                'print("MIGRATION_ROWS", connection.execute("SELECT COUNT(*) FROM django_migrations").fetchone()[0])\n'
                'connection.close()\n'
                'teardown_scratch_django(scratch)\n'
            )
            result = _run_helper_script(script, {'TMPDIR': tmp})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertRegex(
                result.stdout,
                rf"Resolved database: '{re.escape(tmp)}/{_SCRATCH_PREFIX}[^/']+/fomo_db\.sqlite3' "
                r'\(scratch copy of the developer database',
            )
            self.assertIn("MARKER_ROWS [('developer-row',)]", result.stdout)
            self.assertGreater(int(result.stdout.split('MIGRATION_ROWS')[1].split()[0]), 0)
            self.assertEqual(_sha256(source), before, 'the source database changed')
            self.assertIn('Removed scratch database directory:', result.stdout)
            self.assertEqual([p.name for p in Path(tmp).glob(f'{_SCRATCH_PREFIX}*')], [])
