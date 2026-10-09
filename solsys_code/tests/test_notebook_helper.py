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
        helper = _load_helper()
        before = os.environ.get('FOMO_DATABASE_PATH')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(helper.tempfile, 'mkdtemp') as mkdtemp:
            with self.assertRaises(RuntimeError):
                helper.setup_scratch_django(tmp)
        mkdtemp.assert_not_called()
        self.assertEqual(os.environ.get('FOMO_DATABASE_PATH'), before)

    def test_teardown_refuses_a_directory_without_the_scratch_prefix(self):
        helper = _load_helper()
        with tempfile.TemporaryDirectory(prefix='other-') as tmp:
            with self.assertRaises(RuntimeError):
                helper.teardown_scratch_django(tmp)
            self.assertTrue(Path(tmp).is_dir())


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
                rf"Resolved database: '{tmp}/{_SCRATCH_PREFIX}[^/']+/fomo_db\.sqlite3' "
                r'\(fresh scratch database, migrated from empty\)',
            )
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
