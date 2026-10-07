"""Guard tests: every Django-backed pre-executed notebook runs on a scratch database.

Why this module exists (UAT G-37.1-3): four of the notebooks under
``docs/notebooks/pre_executed/`` opened the developer database ``src/fomo_db.sqlite3`` and
wrote demo rows into it, and prose alone -- a README and a CLAUDE.md paired-docs rule -- did
not stop that. Regenerating such a notebook is routine, so each regeneration would have
written to the developer database again. This module turns the rule into something the
suite enforces.

The rule: a notebook whose code calls ``django.setup()`` must, before that call, create a
``fomo-notebook-db-*`` directory with ``tempfile.mkdtemp`` and point ``FOMO_DATABASE_PATH`` at
a database inside it with a plain assignment (never ``os.environ.setdefault``, which would
let an inherited value win). Its committed output must show Django resolved a path inside
such a directory, and its last code cell must remove the directory.

These tests read notebook JSON only and never execute a notebook. Code is inspected with
``ast`` so a comment that merely mentions ``setdefault`` or the variable cannot satisfy or
trip a check.
"""

import ast
import json
import re
from pathlib import Path

from django.test import SimpleTestCase

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / 'docs' / 'notebooks' / 'pre_executed'
SCRATCH_PREFIX = 'fomo-notebook-db-'
DB_ENV_VAR = 'FOMO_DATABASE_PATH'
RESOLVED_LINE = re.compile(r"Resolved database: '([^']+)'")
TEARDOWN_OUTPUT = 'Removed scratch database directory:'


def _cell_source(cell: dict) -> str:
    """Return a cell's source as one string (``.ipynb`` stores it as a string or a list of lines)."""
    source = cell.get('source', '')
    return ''.join(source) if isinstance(source, list) else source


def _cell_output_text(cell: dict) -> str:
    """Return the text of a code cell's committed stream outputs as one string."""
    parts = []
    for output in cell.get('outputs', []):
        text = output.get('text', '')
        parts.append(''.join(text) if isinstance(text, list) else text)
    return ''.join(parts)


def _parse(source: str) -> ast.Module | None:
    """Parse a cell's source, or return None when it is not plain Python (e.g. IPython magics)."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _dotted_name(node: ast.expr) -> str:
    """Return ``a.b.c`` for a chain of attribute accesses on a name, or an empty string."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return ''


def _is_constant(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _calls(tree: ast.Module, dotted_name: str) -> list[ast.Call]:
    """Return every call in ``tree`` to the function named ``dotted_name`` (e.g. ``django.setup``)."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _dotted_name(node.func) == dotted_name]


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']


def django_setup_cells(notebook: dict) -> list[dict]:
    """Return the code cells of ``notebook`` that call ``django.setup()``."""
    cells = []
    for cell in _code_cells(notebook):
        tree = _parse(_cell_source(cell))
        if tree is not None and _calls(tree, 'django.setup'):
            cells.append(cell)
    return cells


def _routes_before(tree: ast.Module, setup_line: int) -> tuple[bool, bool]:
    """Report whether a scratch directory is created, and the variable assigned, before ``setup_line``."""
    creates_scratch_dir = any(
        node.lineno < setup_line
        and any(keyword.arg == 'prefix' and _is_constant(keyword.value, SCRATCH_PREFIX) for keyword in node.keywords)
        for node in _calls(tree, 'tempfile.mkdtemp')
    )
    assigns_variable = any(
        node.lineno < setup_line
        and any(
            isinstance(target, ast.Subscript)
            and _dotted_name(target.value) == 'os.environ'
            and _is_constant(target.slice, DB_ENV_VAR)
            for target in node.targets
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
    )
    return creates_scratch_dir, assigns_variable


def _reads_resolved_database_name(tree: ast.Module) -> bool:
    """Return True when the cell reads ``<settings>.DATABASES['default']['NAME']``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_constant(node.slice, 'NAME'):
            inner = node.value
            if (
                isinstance(inner, ast.Subscript)
                and _is_constant(inner.slice, 'default')
                and _dotted_name(inner.value).split('.')[-1] == 'DATABASES'
            ):
                return True
    return False


def isolation_problems(notebook: dict) -> list[str]:
    """Return human-readable reasons ``notebook`` could touch a real database.

    Args:
        notebook: A parsed ``.ipynb`` file (the JSON object, not a path).

    Returns:
        A list of problems; empty when the notebook never calls ``django.setup()`` or is
        correctly isolated onto a scratch database.
    """
    problems = []
    code_cells = _code_cells(notebook)
    parsed = {id(cell): _parse(_cell_source(cell)) for cell in code_cells}

    for cell in code_cells:
        if parsed[id(cell)] is None and 'django.setup' in _cell_source(cell):
            problems.append('a code cell mentions django.setup but is not parseable Python, so it cannot be checked')

    # An inherited FOMO_DATABASE_PATH could point at a real database, so no cell may defer to it.
    for number, cell in enumerate(code_cells, start=1):
        tree = parsed[id(cell)]
        if tree is None:
            continue
        for call in _calls(tree, 'os.environ.setdefault'):
            if call.args and _is_constant(call.args[0], DB_ENV_VAR):
                problems.append(
                    f'code cell {number} routes with os.environ.setdefault({DB_ENV_VAR!r}, ...); '
                    'assign it directly so an inherited value cannot win'
                )

    setup_cells = django_setup_cells(notebook)
    if not setup_cells:
        return problems
    if len(setup_cells) > 1:
        problems.append(f'{len(setup_cells)} code cells call django.setup(); expected exactly one setup cell')
        return problems

    setup = setup_cells[0]
    tree = parsed[id(setup)]
    setup_line = min(call.lineno for call in _calls(tree, 'django.setup'))
    creates_scratch_dir, assigns_variable = _routes_before(tree, setup_line)
    if not creates_scratch_dir:
        problems.append(
            f'the setup cell does not call tempfile.mkdtemp(prefix={SCRATCH_PREFIX!r}) before django.setup()'
        )
    if not assigns_variable:
        problems.append(f'the setup cell does not assign os.environ[{DB_ENV_VAR!r}] before django.setup()')
    if not _reads_resolved_database_name(tree):
        problems.append("the setup cell never reads DATABASES['default']['NAME'] to check what Django resolved")

    resolved = RESOLVED_LINE.search(_cell_output_text(setup))
    if resolved is None:
        problems.append("the setup cell's committed output has no \"Resolved database: '<path>'\" line")
    elif not Path(resolved.group(1)).parent.name.startswith(SCRATCH_PREFIX):
        problems.append(
            f'the committed run resolved {resolved.group(1)!r}, which is not inside a {SCRATCH_PREFIX}* directory'
        )

    last = code_cells[-1]
    last_tree = parsed[id(last)]
    removes_scratch_dir = last_tree is not None and any(
        call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == 'scratch_db_dir'
        for call in _calls(last_tree, 'shutil.rmtree')
    )
    if not removes_scratch_dir:
        problems.append('the last code cell is not the scratch teardown (shutil.rmtree(scratch_db_dir, ...))')
    if TEARDOWN_OUTPUT not in _cell_output_text(last):
        problems.append(f"the last code cell's committed output does not contain {TEARDOWN_OUTPUT!r}")
    return problems


def _notebook(setup_source=None, setup_output=None, last_source=None, last_output=None, extra_cells=()):
    """Build a small notebook dict in the reference shape, with any part overridden."""
    setup_source = REFERENCE_SETUP if setup_source is None else setup_source
    setup_output = REFERENCE_SETUP_OUTPUT if setup_output is None else setup_output
    last_source = REFERENCE_TEARDOWN if last_source is None else last_source
    last_output = REFERENCE_TEARDOWN_OUTPUT if last_output is None else last_output

    def code(source, output):
        outputs = [{'output_type': 'stream', 'name': 'stdout', 'text': output.splitlines(keepends=True)}]
        return {'cell_type': 'code', 'source': source.splitlines(keepends=True), 'outputs': outputs}

    cells = [{'cell_type': 'markdown', 'source': ['# A demo']}]
    cells.append(code(setup_source, setup_output))
    cells.extend(extra_cells)
    cells.append(code(last_source, last_output))
    return {'cells': cells}


REFERENCE_SETUP = """import os
import tempfile
from pathlib import Path

import django

scratch_db_dir = Path(tempfile.mkdtemp(prefix='fomo-notebook-db-'))
scratch_db_path = scratch_db_dir / 'fomo_db.sqlite3'
os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)

django.setup()

from django.conf import settings as django_settings

resolved_db_name = django_settings.DATABASES['default']['NAME']
print(f'Resolved database: {resolved_db_name!r} (fresh scratch database)')
"""

REFERENCE_SETUP_OUTPUT = "Resolved database: '/tmp/fomo-notebook-db-abc123/fomo_db.sqlite3' (fresh scratch database)\n"

REFERENCE_TEARDOWN = """import shutil

shutil.rmtree(scratch_db_dir, ignore_errors=True)
print(f'Removed scratch database directory: {scratch_db_dir}')
"""

REFERENCE_TEARDOWN_OUTPUT = 'Removed scratch database directory: /tmp/fomo-notebook-db-abc123\n'


class TestPreExecutedNotebooksUseAScratchDatabase(SimpleTestCase):
    """Every Django-backed notebook under ``pre_executed/`` is routed to a scratch database."""

    def test_every_real_pre_executed_notebook_is_isolated(self):
        notebooks = sorted(NOTEBOOK_DIR.glob('*.ipynb'))
        self.assertTrue(notebooks, f'no notebooks found under {NOTEBOOK_DIR}')
        for path in notebooks:
            with self.subTest(notebook=path.name):
                problems = isolation_problems(json.loads(path.read_text()))
                self.assertEqual(problems, [], f'{path.name} is not isolated:\n  ' + '\n  '.join(problems))

    def test_the_real_notebook_check_is_not_vacuous(self):
        """The four notebooks the UAT named are present and are really recognised as Django-backed."""
        required = {
            'import_campaign_csv_demo.ipynb',
            'sync_gemini_observation_calendar_demo.ipynb',
            'project_observation_calendar_demo.ipynb',
            'telescope_runs_demo.ipynb',
        }
        present = {path.name for path in NOTEBOOK_DIR.glob('*.ipynb')}
        self.assertLessEqual(required, present)
        for name in sorted(required):
            with self.subTest(notebook=name):
                notebook = json.loads((NOTEBOOK_DIR / name).read_text())
                self.assertEqual(len(django_setup_cells(notebook)), 1)

    def test_reference_shape_has_no_problems(self):
        self.assertEqual(isolation_problems(_notebook()), [])

    def test_a_notebook_that_never_sets_django_up_is_not_checked(self):
        notebook = {'cells': [{'cell_type': 'code', 'source': ['print(1)'], 'outputs': []}]}
        self.assertEqual(isolation_problems(notebook), [])

    def test_cell_source_may_be_a_single_string(self):
        notebook = _notebook()
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                cell['source'] = ''.join(cell['source'])
        self.assertEqual(isolation_problems(notebook), [])

    def test_no_routing_before_django_setup_is_a_problem(self):
        setup = REFERENCE_SETUP.replace(
            "scratch_db_dir = Path(tempfile.mkdtemp(prefix='fomo-notebook-db-'))\n"
            "scratch_db_path = scratch_db_dir / 'fomo_db.sqlite3'\n"
            "os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)\n",
            '',
        )
        self.assertNotIn('FOMO_DATABASE_PATH', setup)
        self.assertTrue(isolation_problems(_notebook(setup_source=setup)))

    def test_routing_with_setdefault_is_a_problem(self):
        setup = REFERENCE_SETUP.replace(
            "os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)",
            "os.environ.setdefault('FOMO_DATABASE_PATH', str(scratch_db_path))",
        )
        problems = isolation_problems(_notebook(setup_source=setup))
        self.assertTrue(any('setdefault' in problem for problem in problems), problems)

    def test_assigning_the_variable_only_after_django_setup_is_a_problem(self):
        assignment = "os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)\n"
        setup = REFERENCE_SETUP.replace(assignment, '').replace('django.setup()\n', 'django.setup()\n' + assignment)
        self.assertTrue(isolation_problems(_notebook(setup_source=setup)))

    def test_creating_the_scratch_directory_only_after_django_setup_is_a_problem(self):
        mkdtemp = "scratch_db_dir = Path(tempfile.mkdtemp(prefix='fomo-notebook-db-'))\n"
        setup = REFERENCE_SETUP.replace(mkdtemp, "scratch_db_dir = Path('/tmp/somewhere')\n").replace(
            'django.setup()\n', 'django.setup()\n' + mkdtemp
        )
        self.assertTrue(isolation_problems(_notebook(setup_source=setup)))

    def test_a_scratch_directory_with_another_prefix_is_a_problem(self):
        setup = REFERENCE_SETUP.replace("prefix='fomo-notebook-db-'", "prefix='something-else-'")
        self.assertTrue(isolation_problems(_notebook(setup_source=setup)))

    def test_a_comment_cannot_stand_in_for_the_routing(self):
        setup = REFERENCE_SETUP.replace(
            "os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)",
            "# os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)",
        )
        self.assertTrue(isolation_problems(_notebook(setup_source=setup)))

    def test_a_comment_mentioning_setdefault_is_not_a_problem(self):
        setup = REFERENCE_SETUP.replace(
            'django.setup()\n', "# never os.environ.setdefault('FOMO_DATABASE_PATH', ...)\ndjango.setup()\n"
        )
        self.assertEqual(isolation_problems(_notebook(setup_source=setup)), [])

    def test_setdefault_in_a_later_cell_is_a_problem(self):
        later = {
            'cell_type': 'code',
            'source': ["import os\nos.environ.setdefault('FOMO_DATABASE_PATH', '/x')\n"],
            'outputs': [],
        }
        self.assertTrue(isolation_problems(_notebook(extra_cells=[later])))

    def test_two_cells_calling_django_setup_is_a_problem(self):
        second = {'cell_type': 'code', 'source': ['import django\ndjango.setup()\n'], 'outputs': []}
        self.assertTrue(isolation_problems(_notebook(extra_cells=[second])))

    def test_missing_resolved_path_check_is_a_problem(self):
        setup = REFERENCE_SETUP.replace("django_settings.DATABASES['default']['NAME']", "'unchecked'")
        self.assertTrue(isolation_problems(_notebook(setup_source=setup)))

    def test_committed_output_on_the_developer_database_is_a_problem(self):
        output = "Resolved database: '/home/someone/src/fomo_db.sqlite3' (the developer database itself)\n"
        self.assertTrue(isolation_problems(_notebook(setup_output=output)))

    def test_committed_output_with_no_resolved_database_line_is_a_problem(self):
        self.assertTrue(isolation_problems(_notebook(setup_output='Django ready\n')))

    def test_last_code_cell_that_is_not_the_scratch_teardown_is_a_problem(self):
        self.assertTrue(isolation_problems(_notebook(last_source="print('the end')\n", last_output='the end\n')))

    def test_teardown_without_its_committed_output_is_a_problem(self):
        self.assertTrue(isolation_problems(_notebook(last_output='')))

    def test_a_code_cell_that_does_not_parse_but_mentions_django_setup_is_a_problem(self):
        broken = {'cell_type': 'code', 'source': ['def (:\n    django.setup()\n'], 'outputs': []}
        self.assertTrue(isolation_problems(_notebook(extra_cells=[broken])))

    def test_ipython_magics_in_unrelated_cells_are_tolerated(self):
        magic = {'cell_type': 'code', 'source': ['%matplotlib inline\n'], 'outputs': []}
        self.assertEqual(isolation_problems(_notebook(extra_cells=[magic])), [])
