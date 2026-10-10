"""Guard tests: every Django-backed pre-executed notebook runs on a scratch database through the shared helper.

Why this module exists (UAT G-37.1-3, then review findings WR-13, WR-14 and WR-17 of the 37.1 review):
notebooks under ``docs/notebooks/pre_executed/`` once opened the developer database
``src/fomo_db.sqlite3`` and wrote demo rows into it, and a guard that only matched the setup
cell's text was easy to get around (an own ``FOMO_DATABASE_PATH`` assignment with stale good-looking
output, ``from django import setup as _s``, a setup call hidden in a ``def``). Regenerating such a
notebook is routine, so each regeneration would have written to the developer database again.

The rule, enforced by ``isolation_problems()``: every Django-backed notebook calls the shared helper
(``docs/notebooks/fomo_nb_utils.py``) first -- one top-level
``scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh' | 'snapshot')`` before any Django
import -- never sets Django up or routes the database itself, never byte-copies a database, ends with
``teardown_scratch_django(scratch_db_dir)``, and shows output from one fresh run (execution counts
1..N, the ``Resolved database:`` line inside a ``fomo-notebook-db-*`` directory). The helper's own
behaviour is proven by running it in ``test_notebook_helper.py``.

These tests read notebook JSON only and never execute a notebook. Code is inspected with ``ast`` so a
comment that merely mentions a name cannot satisfy or trip a check. A static guard cannot see code the
AST cannot read (for example ``exec()`` of a built string), and does not claim to.
"""

import ast
import copy
import importlib.util
import json
import re
from pathlib import Path

from django.test import SimpleTestCase

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / 'docs' / 'notebooks' / 'pre_executed'
HELPER_PATH = REPO_ROOT / 'docs' / 'notebooks' / 'fomo_nb_utils.py'

HELPER_MODULE = 'fomo_nb_utils'
SETUP_FUNCTION = 'setup_scratch_django'
TEARDOWN_FUNCTION = 'teardown_scratch_django'
SCRATCH_MODES = ('fresh', 'snapshot')
SCRATCH_PREFIX = 'fomo-notebook-db-'
DEVELOPER_DB_NAME = 'fomo_db.sqlite3'
DB_ENV_VAR = 'FOMO_DATABASE_PATH'
# R12: names a notebook never needs, because only the helper may choose or change the database.
REROUTE_NAMES = frozenset({'DATABASES', 'settings_dict', 'override_settings', 'developer_database_path'})
# R12: attribute chains (matched when they equal, or end with ``.`` plus, one of these) that reach the same dicts.
REROUTE_DOTTED_SUFFIXES = ()
# Text every R12 problem carries, so a test can tell an R12 report from any other rule's.
REROUTE_MARKER = '(R12:'
RESOLVED_LINE = re.compile(r"Resolved database: '([^']+)'")
TEARDOWN_OUTPUT = 'Removed scratch database directory:'
EXPECTED_NOTEBOOKS = frozenset(
    {
        'backfill_lco_observations_demo.ipynb',
        'campaign_lifecycle_demo.ipynb',
        'import_campaign_csv_demo.ipynb',
        'load_telescope_runs_demo.ipynb',
        'project_observation_calendar_demo.ipynb',
        'reconcile_campaign_runs_demo.ipynb',
        'sync_gemini_observation_calendar_demo.ipynb',
        'telescope_runs_demo.ipynb',
    }
)
FORBIDDEN_COPY_NAMES = ('copy', 'copy2', 'copyfile')
FORBIDDEN_COPY_CALLS = tuple(f'shutil.{name}' for name in FORBIDDEN_COPY_NAMES)
# Call names that write: an ``assert`` wrapping one would vanish under ``python -O`` (WARN-03).
WRITE_CALL_NAMES = frozenset(
    {
        'call_command',
        'save',
        'create',
        'delete',
        'update',
        'bulk_create',
        'bulk_update',
        'get_or_create',
        'update_or_create',
        'post',
        'put',
        'patch',
        'execute',
        'executemany',
        'write',
        'write_text',
        'unlink',
        'rmtree',
    }
)
# An unparseable cell (IPython magics) is tolerated unless it is about Django or a database.
UNPARSEABLE_DJANGO_IMPORT = re.compile(r'^\s*(from|import)\s+(django|solsys_code|tom_)', re.MULTILINE)
UNPARSEABLE_SUSPECT_TEXT = ('django.setup', DB_ENV_VAR, 'shutil.copy', DEVELOPER_DB_NAME)
# Text that must never reach a committed notebook (snapshot output comes from the developer database).
CREDENTIAL_TEXT = re.compile(r'api_key|Authorization|token=|pbkdf2_sha256\$', re.IGNORECASE)
HORIZON_DIP_LINE = re.compile(r'Las Campanas \(2402 m\): ([0-9.]+) deg')
README_PATH = REPO_ROOT / 'docs' / 'notebooks' / 'README.md'
README_SNAPSHOT_HEADING = '### Notebooks that read a read-only snapshot'
README_FRESH_HEADING = '### Notebooks that start from a fresh migrated database'
# The rule the notebooks actually follow (D-04), in the same words in the README, ROADMAP.md and REQUIREMENTS.md.
ISOLATION_RULE = (
    'No pre-executed notebook byte-copies or writes the developer database (`src/fomo_db.sqlite3`); '
    'a notebook that needs real rows reads it only through a read-only online-backup snapshot, '
    'and every other notebook starts from a fresh migrated database.'
)


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


def _all_output_text(cell: dict) -> str:
    """Return every textual output of a code cell: streams, plain/HTML results and error tracebacks."""
    parts = []
    for output in cell.get('outputs', []):
        values = [output.get('text', '')]
        values.extend(output.get('data', {}).get(mime, '') for mime in ('text/plain', 'text/html'))
        values.append(output.get('traceback', ''))
        for value in values:
            parts.append(''.join(value) if isinstance(value, list) else value)
    return '\n'.join(parts)


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


def _call_name(call: ast.Call) -> str:
    """Return the bare function name of a call: the attribute for ``a.b()``, the id for ``b()``."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ''


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']


def _parsed_code_cells(notebook: dict) -> list[tuple[int, dict, ast.Module | None]]:
    """Return ``(number, cell, tree)`` for each code cell, numbered from 1 in notebook order."""
    return [(number, cell, _parse(_cell_source(cell))) for number, cell in enumerate(_code_cells(notebook), start=1)]


def _statements(parsed):
    """Yield ``(cell number, statement index, statement)`` for each top-level statement, in run order."""
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        for index, statement in enumerate(tree.body):
            yield number, index, statement


def _is_django_module(name: str | None) -> bool:
    if not name:
        return False
    top = name.split('.')[0]
    return top in ('django', 'solsys_code') or top.startswith('tom_')


def _django_imports(node: ast.AST) -> list[ast.AST]:
    """Return every import of Django, the project app or a TOM plugin anywhere inside ``node``."""
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Import) and any(_is_django_module(alias.name) for alias in child.names):
            found.append(child)
        elif isinstance(child, ast.ImportFrom) and child.level == 0 and _is_django_module(child.module):
            found.append(child)
    return found


def is_django_backed(notebook: dict) -> bool:
    """Return True when any code cell imports Django, the project app or a TOM plugin (D-13)."""
    return any(tree is not None and _django_imports(tree) for _number, _cell, tree in _parsed_code_cells(notebook))


def _helper_bindings(parsed) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Return the helper names bound by top-level ``from fomo_nb_utils import ...`` and any import problems."""
    bound: dict[str, tuple[int, int]] = {}
    problems = []
    for number, index, statement in _statements(parsed):
        if not (isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module == HELPER_MODULE):
            continue
        for alias in statement.names:
            if alias.name == '*':
                problems.append(f'code cell {number} star-imports {HELPER_MODULE}; import the helper names explicitly')
            elif alias.asname:
                problems.append(
                    f'code cell {number} imports {alias.name} from {HELPER_MODULE} under the alias {alias.asname!r}; '
                    'import it without an alias'
                )
            else:
                bound.setdefault(alias.name, (number, index))
    return bound, problems


def _setup_matches(parsed, bound) -> list[tuple[int, int, ast.Assign]]:
    """Return the top-level ``name = setup_scratch_django(...)`` statements whose function was imported earlier."""
    matches = []
    imported_at = bound.get(SETUP_FUNCTION)
    for number, index, statement in _statements(parsed):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == SETUP_FUNCTION
            and imported_at is not None
            and imported_at < (number, index)
        ):
            matches.append((number, index, statement))
    return matches


def setup_call(notebook: dict) -> tuple[int, int, ast.Assign] | None:
    """Return ``(cell number, statement index, assignment)`` of the notebook's one valid helper call, or None.

    Only a top-level ``name = setup_scratch_django(...)`` whose function was imported by an earlier top-level
    ``from fomo_nb_utils import ...`` without an alias counts. A call inside a ``def``, ``if`` or ``try`` body
    does not.
    """
    parsed = _parsed_code_cells(notebook)
    bound, _problems = _helper_bindings(parsed)
    matches = _setup_matches(parsed, bound)
    return matches[0] if len(matches) == 1 else None


def notebook_mode(notebook: dict) -> str | None:
    """Return the ``mode`` constant of the notebook's setup call ('fresh' or 'snapshot'), or None."""
    match = setup_call(notebook)
    if match is None:
        return None
    call = match[2].value
    keyword = call.keywords[0] if len(call.keywords) == 1 else None
    if (
        keyword is not None
        and keyword.arg == 'mode'
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value in SCRATCH_MODES
    ):
        return keyword.value.value
    return None


def _django_setup_problems(parsed) -> list[str]:
    """R4: a notebook must not set Django up itself, under any spelling or alias."""
    problems = []
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted_name(node.func) == 'django.setup':
                problems.append(f'code cell {number} calls django.setup(); only {SETUP_FUNCTION} may set Django up')
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'django':
                if any(alias.name == 'setup' for alias in node.names):
                    problems.append(f'code cell {number} imports django.setup (under any alias); use {SETUP_FUNCTION}')
            elif isinstance(node, ast.Import) and any(alias.name == 'django' and alias.asname for alias in node.names):
                problems.append(f'code cell {number} imports django under an alias; use a plain `import django`')
    return problems


def _database_routing_problems(parsed) -> list[str]:
    """R5 and R7: no code string or keyword names the database variable or the developer database file."""
    problems = []
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if DB_ENV_VAR in node.value:
                    problems.append(f'code cell {number} names {DB_ENV_VAR} in code; only {SETUP_FUNCTION} may route')
                if DEVELOPER_DB_NAME in node.value:
                    problems.append(f'code cell {number} names {DEVELOPER_DB_NAME} in code; only the helper may')
            elif isinstance(node, ast.keyword) and node.arg == DB_ENV_VAR:
                problems.append(f'code cell {number} passes a {DB_ENV_VAR} keyword; only {SETUP_FUNCTION} may route')
    return problems


def _byte_copy_problems(parsed) -> list[str]:
    """R6: no ``shutil.copy*`` call and no import of those names (a byte copy can catch a database mid-write)."""
    problems = []
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        shutil_names = {'shutil'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                shutil_names.update(alias.asname for alias in node.names if alias.name == 'shutil' and alias.asname)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                dotted = _dotted_name(node.func)
                if dotted.split('.')[0] in shutil_names and dotted.split('.')[-1] in FORBIDDEN_COPY_NAMES:
                    problems.append(f'code cell {number} calls {dotted}(); a notebook must never byte-copy a database')
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'shutil':
                if any(alias.name in (*FORBIDDEN_COPY_NAMES, '*') for alias in node.names):
                    problems.append(f'code cell {number} imports a shutil copy function; a notebook must never copy')
    return problems


def _database_rerouting_problems(parsed) -> list[str]:
    """R12: no notebook re-routes Django's database after the helper call, under any of the names it could use."""
    problems = []
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in REROUTE_NAMES:
                found.add(node.id)
            elif isinstance(node, ast.Attribute):
                if node.attr in REROUTE_NAMES:
                    found.add(node.attr)
                dotted = _dotted_name(node)
                found.update(
                    suffix for suffix in REROUTE_DOTTED_SUFFIXES if dotted == suffix or dotted.endswith(f'.{suffix}')
                )
            elif isinstance(node, ast.keyword) and node.arg in REROUTE_NAMES:
                found.add(node.arg)
            elif isinstance(node, ast.alias):
                found.update(name for name in (node.name.split('.')[-1], node.asname) if name in REROUTE_NAMES)
        problems.extend(
            f'code cell {number} references {name}; only {SETUP_FUNCTION} may choose the database '
            f'{REROUTE_MARKER} no re-routing after setup)'
            for name in sorted(found)
        )
    return problems


def isolation_problems(notebook: dict) -> list[str]:
    """Return human-readable reasons ``notebook`` could touch a real database or show stale output.

    Args:
        notebook: A parsed ``.ipynb`` file (the JSON object, not a path).

    Returns:
        A list of problems, in statement order where it matters; empty when the notebook is clean. A notebook
        that imports nothing from Django, the project app or a TOM plugin is only checked for the rules that
        apply to every notebook (no own database routing, no byte copy, no developer database file name, and
        none of the names R12 lists for re-routing the database after setup).
    """
    problems = []
    code_cells = _code_cells(notebook)
    parsed = _parsed_code_cells(notebook)

    # R0: a cell that is not plain Python is tolerated (IPython magics) unless it is about Django or a database.
    for number, cell, tree in parsed:
        if tree is not None:
            continue
        source = _cell_source(cell)
        if UNPARSEABLE_DJANGO_IMPORT.search(source) or any(text in source for text in UNPARSEABLE_SUSPECT_TEXT):
            problems.append(
                f'code cell {number} is not parseable Python but mentions Django or a database, so it cannot be checked'
            )

    # R5, R6, R7, R12 apply to every notebook.
    problems.extend(_database_routing_problems(parsed))
    problems.extend(_byte_copy_problems(parsed))
    problems.extend(_database_rerouting_problems(parsed))

    if not is_django_backed(notebook):
        return problems

    # R4, R8.
    problems.extend(_django_setup_problems(parsed))
    counts = [cell.get('execution_count') for cell in code_cells]
    if counts != list(range(1, len(code_cells) + 1)):
        problems.append(
            f"the code cells' execution counts are {counts}, not 1..{len(code_cells)}: "
            'the committed output is not from one fresh Restart and Run All'
        )
    for number, cell, _tree in parsed:
        if any(output.get('output_type') == 'error' for output in cell.get('outputs', [])):
            problems.append(f'code cell {number} has an error output')

    # R2: exactly one valid top-level helper call.
    bound, import_problems = _helper_bindings(parsed)
    problems.extend(import_problems)
    matches = _setup_matches(parsed, bound)
    total_calls = sum(
        1
        for _number, _cell, tree in parsed
        if tree is not None
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == SETUP_FUNCTION
    )
    call_shape = f'`scratch_db_dir = {SETUP_FUNCTION}(repo_root_path, mode=...)`'
    if not matches:
        problems.append(
            f'no top-level {call_shape} call after `from {HELPER_MODULE} import {SETUP_FUNCTION}` '
            '(a call inside a def, if or try body does not count)'
        )
        return problems
    if len(matches) > 1:
        problems.append(f'{len(matches)} top-level {SETUP_FUNCTION} calls; expected exactly one')
        return problems
    if total_calls > 1:
        problems.append(f'{SETUP_FUNCTION} is also called somewhere other than the one top-level call')
    setup_number, setup_index, setup_statement = matches[0]
    call = setup_statement.value
    keyword = call.keywords[0] if len(call.keywords) == 1 else None
    if len(call.args) != 1 or keyword is None or keyword.arg != 'mode':
        problems.append(f'code cell {setup_number}: the setup call must be {call_shape} with exactly those arguments')
    elif not (isinstance(keyword.value, ast.Constant) and keyword.value.value in SCRATCH_MODES):
        problems.append(f'code cell {setup_number}: mode must be a constant, one of {SCRATCH_MODES}')

    # R3: no Django import (even nested in an if or try) before the helper call.
    for number, index, statement in _statements(parsed):
        if (number, index) >= (setup_number, setup_index):
            break
        if _django_imports(statement):
            problems.append(f'code cell {number} imports Django before the {SETUP_FUNCTION} call')

    # R9: the setup cell's committed output shows a scratch database.
    setup_cell = code_cells[setup_number - 1]
    resolved = RESOLVED_LINE.search(_cell_output_text(setup_cell))
    if resolved is None:
        problems.append("the setup cell's committed output has no \"Resolved database: '<path>'\" line")
    else:
        resolved_path = Path(resolved.group(1))
        if not resolved_path.parent.name.startswith(SCRATCH_PREFIX) or resolved_path.name != DEVELOPER_DB_NAME:
            problems.append(
                f'the committed run resolved {resolved.group(1)!r}, which is not {DEVELOPER_DB_NAME} '
                f'inside a {SCRATCH_PREFIX}* directory'
            )

    # R10: the last code cell ends with the teardown of the same scratch directory, with its output.
    last_number, last_cell, last_tree = parsed[-1]
    scratch_name = setup_statement.targets[0].id
    last_statement = last_tree.body[-1] if last_tree is not None and last_tree.body else None
    ends_with_teardown = (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == TEARDOWN_FUNCTION
        and TEARDOWN_FUNCTION in bound
        and bound[TEARDOWN_FUNCTION] < (last_number, len(last_tree.body) - 1)
        and len(last_statement.value.args) == 1
        and not last_statement.value.keywords
        and isinstance(last_statement.value.args[0], ast.Name)
        and last_statement.value.args[0].id == scratch_name
    )
    if not ends_with_teardown:
        problems.append(f'the last code cell does not end with `{TEARDOWN_FUNCTION}({scratch_name})`')
    if TEARDOWN_OUTPUT not in _cell_output_text(last_cell):
        problems.append(f"the last code cell's committed output does not contain {TEARDOWN_OUTPUT!r}")

    # R11 (WARN-03): an assert vanishes under `python -O`, so none may sit in the setup cell or wrap a write.
    setup_tree = parsed[setup_number - 1][2]
    if any(isinstance(node, ast.Assert) for node in ast.walk(setup_tree)):
        problems.append(f'the setup cell (code cell {setup_number}) contains an assert; use a raise')
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and any(
                isinstance(sub, ast.Call) and _call_name(sub) in WRITE_CALL_NAMES for sub in ast.walk(node.test)
            ):
                problems.append(f'code cell {number} has an assert wrapping a write call, which `python -O` would drop')
    return problems


def _code(source: str, output: str = ''):
    outputs = [{'output_type': 'stream', 'name': 'stdout', 'text': output.splitlines(keepends=True)}] if output else []
    return {'cell_type': 'code', 'source': source.splitlines(keepends=True), 'outputs': outputs}


def _notebook(
    setup_source=None,
    setup_output=None,
    last_source=None,
    last_output=None,
    django_source=None,
    pre_cells=(),
    extra_cells=(),
    counts=None,
):
    """Build a small notebook dict in the reference shape, with any part overridden.

    Code cells are numbered 1..N in run order unless ``counts`` gives the execution counts explicitly.
    ``pre_cells`` run before the setup cell and ``extra_cells`` between the Django cell and the teardown.
    """
    setup_source = REFERENCE_SETUP if setup_source is None else setup_source
    setup_output = REFERENCE_SETUP_OUTPUT if setup_output is None else setup_output
    last_source = REFERENCE_TEARDOWN if last_source is None else last_source
    last_output = REFERENCE_TEARDOWN_OUTPUT if last_output is None else last_output
    django_source = REFERENCE_DJANGO_CELL if django_source is None else django_source

    cells = [{'cell_type': 'markdown', 'source': ['# A demo']}]
    cells.extend(pre_cells)
    cells.append(_code(setup_source, setup_output))
    cells.append(_code(django_source))
    cells.extend(extra_cells)
    cells.append(_code(last_source, last_output))
    code_cells = [cell for cell in cells if cell['cell_type'] == 'code']
    for position, cell in enumerate(code_cells):
        cell['execution_count'] = position + 1 if counts is None else counts[position]
    return {'cells': cells}


REFERENCE_SETUP = """import sys
from pathlib import Path

repo_root_path = Path.cwd().resolve().parents[2]
sys.path.insert(0, str(repo_root_path))
sys.path.insert(0, str(repo_root_path / 'docs' / 'notebooks'))

from fomo_nb_utils import setup_scratch_django, teardown_scratch_django  # noqa: E402

# fresh scratch database: the reason goes here
scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh')
"""

REFERENCE_SETUP_OUTPUT = (
    "Django ready: settings module='src.fomo.settings', repo_root='/repo'\n"
    "Resolved database: '/tmp/fomo-notebook-db-abc123/fomo_db.sqlite3' (fresh scratch database, migrated from empty)\n"
)

REFERENCE_DJANGO_CELL = 'from django.conf import settings\n'

REFERENCE_TEARDOWN = 'teardown_scratch_django(scratch_db_dir)\n'

REFERENCE_TEARDOWN_OUTPUT = (
    'Removed scratch database directory: /tmp/fomo-notebook-db-abc123\n'
    'The developer database (src/fomo_db.sqlite3) was never opened for writing.\n'
)


def _first_problem_mentions(testcase, problems, text):
    testcase.assertTrue(problems, 'expected the guard to report a problem')
    testcase.assertTrue(any(text in problem for problem in problems), problems)


class TestIsolationRulesOnSyntheticNotebooks(SimpleTestCase):
    """Each WR-13/WR-14/WR-17 bypass, built as a small notebook, is reported -- whatever its output says."""

    def test_reference_shape_has_no_problems(self):
        self.assertEqual(isolation_problems(_notebook()), [])
        self.assertEqual(notebook_mode(_notebook()), 'fresh')

    def test_cell_source_may_be_a_single_string(self):
        notebook = _notebook()
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                cell['source'] = ''.join(cell['source'])
        self.assertEqual(isolation_problems(notebook), [])

    def test_ipython_magics_in_unrelated_cells_are_tolerated(self):
        magic = _code('%matplotlib inline\n')
        self.assertEqual(isolation_problems(_notebook(extra_cells=[magic])), [])

    def test_a_notebook_that_imports_nothing_from_django_is_not_django_backed(self):
        notebook = {'cells': [{'cell_type': 'code', 'source': ['import astropy.units as u\n'], 'outputs': []}]}
        self.assertFalse(is_django_backed(notebook))
        self.assertEqual(isolation_problems(notebook), [])
        self.assertTrue(is_django_backed(_notebook()))

    def test_own_fomo_database_path_assignment_with_stale_output_is_a_problem(self):
        # WR-13: the stale committed output still says the scratch database was used.
        setup = REFERENCE_SETUP.replace(
            'scratch_db_dir = setup',
            "import os\nos.environ['FOMO_DATABASE_PATH'] = '/home/someone/src/fomo_db.sqlite3'\nscratch_db_dir = setup",
        )
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), DB_ENV_VAR)

    def test_a_comment_mentioning_fomo_database_path_is_not_a_problem(self):
        setup = REFERENCE_SETUP.replace(
            '# fresh scratch database: the reason goes here',
            '# the helper, never this notebook, sets FOMO_DATABASE_PATH',
        )
        self.assertEqual(isolation_problems(_notebook(setup_source=setup)), [])

    def test_aliased_django_setup_import_is_a_problem(self):
        setup = REFERENCE_SETUP + '\nfrom django import setup as _s\n_s()\n'
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'django.setup')

    def test_aliased_django_import_is_a_problem(self):
        setup = REFERENCE_SETUP + '\nimport django as dj\ndj.setup()\n'
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'alias')

    def test_calling_django_setup_directly_is_a_problem(self):
        setup = REFERENCE_SETUP + '\nimport django\ndjango.setup()\n'
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'django.setup()')

    def test_setup_call_inside_a_def_or_if_false_does_not_count(self):
        bodies = {
            'def': "def run():\n    scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh')\n",
            'if False': "if False:\n    scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh')\n",
        }
        for label, body in bodies.items():
            with self.subTest(label):
                setup = REFERENCE_SETUP.replace(
                    "scratch_db_dir = setup_scratch_django(repo_root_path, mode='fresh')\n", body
                )
                self.assertIsNone(setup_call(_notebook(setup_source=setup)))
                _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'no top-level')

    def test_django_import_before_the_helper_call_is_a_problem(self):
        in_setup = REFERENCE_SETUP.replace('from fomo_nb_utils', 'import django\nfrom fomo_nb_utils')
        cases = {
            'earlier statement in the setup cell': _notebook(setup_source=in_setup),
            'an earlier cell': _notebook(pre_cells=[_code('from django.conf import settings\n')]),
        }
        for label, notebook in cases.items():
            with self.subTest(label):
                _first_problem_mentions(self, isolation_problems(notebook), 'before the setup_scratch_django call')

    def test_django_import_nested_in_a_block_before_the_helper_call_is_a_problem(self):
        blocks = {
            'if True:': 'if True:\n    from django.conf import settings\n',
            'try:': 'try:\n    import django\nexcept ImportError:\n    pass\n',
        }
        for label, block in blocks.items():
            with self.subTest(label):
                setup = REFERENCE_SETUP.replace('from fomo_nb_utils', block + 'from fomo_nb_utils')
                _first_problem_mentions(
                    self, isolation_problems(_notebook(setup_source=setup)), 'before the setup_scratch_django call'
                )

    def test_missing_helper_call_is_a_problem(self):
        setup = 'import sys\n'
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'no top-level')

    def test_two_helper_calls_is_a_problem(self):
        setup = REFERENCE_SETUP + "other_dir = setup_scratch_django(repo_root_path, mode='fresh')\n"
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'expected exactly one')

    def test_aliased_helper_import_is_a_problem(self):
        setup = REFERENCE_SETUP.replace(
            'import setup_scratch_django, teardown_scratch_django',
            'import setup_scratch_django as s, teardown_scratch_django',
        )
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'alias')

    def test_unknown_or_missing_mode_is_a_problem(self):
        call = "setup_scratch_django(repo_root_path, mode='fresh')"
        cases = {
            "mode='copy'": "setup_scratch_django(repo_root_path, mode='copy')",
            'mode=chosen_mode': 'setup_scratch_django(repo_root_path, mode=chosen_mode)',
            'no mode': 'setup_scratch_django(repo_root_path)',
        }
        for label, replacement in cases.items():
            with self.subTest(label):
                setup = REFERENCE_SETUP.replace(call, replacement)
                notebook = _notebook(setup_source=setup)
                self.assertTrue(isolation_problems(notebook))
                self.assertIsNone(notebook_mode(notebook))

    def test_byte_copy_is_a_problem(self):
        cases = {
            'shutil.copy2(a, b)': 'import shutil\nshutil.copy2(a, b)\n',
            'shutil.copyfile(a, b)': 'import shutil\nshutil.copyfile(a, b)\n',
            'from shutil import copy as c': 'from shutil import copy as c\n',
            'import shutil as sh': 'import shutil as sh\nsh.copy(a, b)\n',
        }
        for label, source in cases.items():
            with self.subTest(label):
                _first_problem_mentions(self, isolation_problems(_notebook(extra_cells=[_code(source)])), 'copy')

    def test_developer_database_literal_in_code_is_a_problem(self):
        cell = _code("db = open('src/fomo_db.sqlite3', 'rb')\n")
        _first_problem_mentions(self, isolation_problems(_notebook(extra_cells=[cell])), DEVELOPER_DB_NAME)

    def test_developer_database_name_in_a_comment_is_not_a_problem(self):
        cell = _code('# src/fomo_db.sqlite3 is never opened\nprint(1)\n')
        self.assertEqual(isolation_problems(_notebook(extra_cells=[cell])), [])

    def test_non_contiguous_execution_counts_are_a_problem(self):
        for counts in ([1, 2, 4], [1, None, 3], [2, 3, 4]):
            with self.subTest(counts=counts):
                _first_problem_mentions(self, isolation_problems(_notebook(counts=counts)), 'execution counts')

    def test_an_error_output_is_a_problem(self):
        cell = _code('1 / 0\n')
        cell['outputs'] = [{'output_type': 'error', 'ename': 'ZeroDivisionError', 'evalue': 'x', 'traceback': []}]
        _first_problem_mentions(self, isolation_problems(_notebook(extra_cells=[cell])), 'error output')

    def test_committed_output_without_a_resolved_line_is_a_problem(self):
        _first_problem_mentions(self, isolation_problems(_notebook(setup_output='Django ready\n')), 'Resolved database')

    def test_committed_output_outside_a_scratch_directory_is_a_problem(self):
        output = "Resolved database: '/home/someone/src/fomo_db.sqlite3' (fresh scratch database)\n"
        _first_problem_mentions(self, isolation_problems(_notebook(setup_output=output)), 'inside a fomo-notebook-db-')

    def test_last_code_cell_that_is_not_the_teardown_is_a_problem(self):
        notebook = _notebook(last_source="print('the end')\n", last_output='the end\n')
        _first_problem_mentions(self, isolation_problems(notebook), 'does not end with')

    def test_teardown_of_another_name_is_a_problem(self):
        notebook = _notebook(last_source='teardown_scratch_django(other_dir)\n')
        _first_problem_mentions(self, isolation_problems(notebook), 'does not end with')

    def test_teardown_without_its_committed_output_is_a_problem(self):
        _first_problem_mentions(self, isolation_problems(_notebook(last_output='x\n')), 'committed output')

    def test_assert_in_the_setup_cell_is_a_problem(self):
        setup = REFERENCE_SETUP + 'assert scratch_db_dir.exists()\n'
        _first_problem_mentions(self, isolation_problems(_notebook(setup_source=setup)), 'contains an assert')

    def test_assert_wrapping_a_write_is_a_problem(self):
        cell = _code('assert client.post(url).status_code == 302\n')
        _first_problem_mentions(self, isolation_problems(_notebook(extra_cells=[cell])), 'wrapping a write')

    def test_a_demonstration_assert_that_only_reads_is_not_a_problem(self):
        cell = _code('assert Thing.objects.count() == 3\n')
        self.assertEqual(isolation_problems(_notebook(extra_cells=[cell])), [])

    def test_an_unparseable_cell_that_imports_django_is_a_problem(self):
        cases = {
            'broken syntax mentioning django.setup': 'def (:\n    django.setup()\n',
            'a magic followed by a Django import': '%%time\nfrom django.conf import settings\n',
        }
        for label, source in cases.items():
            with self.subTest(label):
                _first_problem_mentions(
                    self, isolation_problems(_notebook(extra_cells=[_code(source)])), 'not parseable'
                )


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _append_to_setup_cell(notebook: dict, source: str) -> dict:
    """Return a copy of ``notebook`` with ``source`` appended to the code cell that holds the helper call.

    Appending to the setup cell, rather than inserting a new cell, keeps the execution counts contiguous and
    leaves every other cell and every committed output as it was, so only the rule under test can fire.

    Args:
        notebook: A parsed ``.ipynb`` file with one valid helper call.
        source: Python source to add after the existing setup cell source, each line ending in a newline.

    Returns:
        A deep copy of ``notebook``; the argument is not modified.
    """
    modified = copy.deepcopy(notebook)
    setup_number = setup_call(modified)[0]
    cell = _code_cells(modified)[setup_number - 1]
    cell['source'] = _cell_source(cell).rstrip('\n') + '\n' + source
    return modified


def _rerouting_names(problems: list[str]) -> set[str]:
    """Return the names that R12 problems say a notebook references (``references <name>;``)."""
    return {match.group(1) for problem in problems for match in re.finditer(r'references (\S+);', problem)}


class TestRerouteRuleOnARealNotebook(SimpleTestCase):
    """WR-01 / VERIFICATION gap 2: a notebook that calls the helper and then re-routes Django's database.

    The two probes are the verifier's own, appended to the setup cell of a copy of a real notebook with its
    committed output kept. Before rule R12 existed ``isolation_problems()`` returned ``[]`` for both.
    """

    SETTINGS_DICT_PROBE = (
        'from django.db import connections\n'
        'from fomo_nb_utils import developer_database_path\n'
        "connections['default'].close()\n"
        "connections['default'].settings_dict['NAME'] = str(developer_database_path(repo_root_path))\n"
    )
    OVERRIDE_SETTINGS_PROBE = (
        'from django.test import override_settings\n'
        'from fomo_nb_utils import developer_database_path\n'
        "override_settings(DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', "
        "'NAME': str(developer_database_path(repo_root_path))}}).enable()\n"
    )

    def setUp(self):
        self.notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')

    def test_the_unmodified_real_notebook_has_no_problems(self):
        self.assertEqual(isolation_problems(self.notebook), [])

    def test_a_settings_dict_reroute_after_setup_is_reported_on_a_real_notebook(self):
        problems = isolation_problems(_append_to_setup_cell(self.notebook, self.SETTINGS_DICT_PROBE))
        self.assertTrue(problems, 'the guard missed a settings_dict re-route after the helper call')
        self.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)
        self.assertTrue({'settings_dict', 'developer_database_path'} <= _rerouting_names(problems), problems)

    def test_an_override_settings_reroute_after_setup_is_reported_on_a_real_notebook(self):
        problems = isolation_problems(_append_to_setup_cell(self.notebook, self.OVERRIDE_SETTINGS_PROBE))
        self.assertTrue(problems, 'the guard missed an override_settings re-route after the helper call')
        self.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)
        self.assertTrue({'override_settings', 'DATABASES', 'developer_database_path'} <= _rerouting_names(problems))


class TestPreExecutedNotebooksUseTheHelper(SimpleTestCase):
    """Every real notebook under ``pre_executed/`` is routed to a scratch database by the shared helper."""

    def test_every_real_pre_executed_notebook_is_isolated(self):
        notebooks = sorted(NOTEBOOK_DIR.glob('*.ipynb'))
        self.assertTrue(notebooks, f'no notebooks found under {NOTEBOOK_DIR}')
        for path in notebooks:
            with self.subTest(notebook=path.name):
                problems = isolation_problems(_load_notebook(path))
                self.assertEqual(problems, [], f'{path.name} is not isolated:\n  ' + '\n  '.join(problems))

    def test_the_real_notebook_check_is_not_vacuous(self):
        """All eight notebooks are present, recognised as Django-backed, and each has its one helper call."""
        present = {path.name for path in NOTEBOOK_DIR.glob('*.ipynb')}
        self.assertEqual(present, set(EXPECTED_NOTEBOOKS))
        for name in sorted(EXPECTED_NOTEBOOKS):
            with self.subTest(notebook=name):
                notebook = _load_notebook(NOTEBOOK_DIR / name)
                self.assertTrue(is_django_backed(notebook))
                self.assertIsNotNone(setup_call(notebook))
                self.assertIn(notebook_mode(notebook), SCRATCH_MODES)

    def test_the_guard_names_match_the_helper(self):
        spec = importlib.util.spec_from_file_location(HELPER_MODULE, HELPER_PATH)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        self.assertTrue(callable(getattr(helper, SETUP_FUNCTION)))
        self.assertTrue(callable(getattr(helper, TEARDOWN_FUNCTION)))
        self.assertEqual(helper.SCRATCH_MODES, SCRATCH_MODES)
        self.assertEqual(helper.SCRATCH_DIR_PREFIX, SCRATCH_PREFIX)
        self.assertEqual(helper.SCRATCH_DB_NAME, DEVELOPER_DB_NAME)
        self.assertEqual(helper.TEARDOWN_LINE_PREFIX, TEARDOWN_OUTPUT)

    def test_no_committed_output_reports_arcmin_deg(self):
        """WR-17: a dip printed in arcminutes under a degrees label (``arcmin deg``) is a unit error."""
        for path in sorted(NOTEBOOK_DIR.glob('*.ipynb')):
            with self.subTest(notebook=path.name):
                for cell in _code_cells(_load_notebook(path)):
                    self.assertNotIn('arcmin deg', _all_output_text(cell))

    def test_telescope_runs_demo_horizon_dip_matches_the_stage1_contract(self):
        """WR-17: the committed Las Campanas dip is in degrees and inside 1.44 +/- 0.02 (EPHEM-03)."""
        notebook = _load_notebook(NOTEBOOK_DIR / 'telescope_runs_demo.ipynb')
        values = [
            float(match.group(1))
            for cell in _code_cells(notebook)
            for match in HORIZON_DIP_LINE.finditer(_all_output_text(cell))
        ]
        self.assertEqual(len(values), 1, 'expected exactly one "Las Campanas (2402 m): <x> deg" line')
        self.assertTrue(1.42 <= values[0] <= 1.46, values)

    def test_no_credential_shaped_text_in_any_notebook(self):
        """A snapshot notebook's output comes from the developer database and is published with the docs."""
        for path in sorted(NOTEBOOK_DIR.glob('*.ipynb')):
            with self.subTest(notebook=path.name):
                match = CREDENTIAL_TEXT.search(path.read_text(encoding='utf-8'))
                self.assertIsNone(match, f'{path.name} contains credential-shaped text {match and match.group(0)!r}')


class TestReadmeStatesTheIsolationRule(SimpleTestCase):
    """``docs/notebooks/README.md`` says what the notebooks do, and its two lists match each notebook's mode."""

    @staticmethod
    def _listed_notebooks(readme: str, heading: str) -> set[str]:
        """Return the backticked ``*.ipynb`` names in the bullets under ``heading``, up to the next heading."""
        start = readme.index(heading) + len(heading)
        following = re.search(r'^#{1,6} ', readme[start:], re.MULTILINE)
        section = readme[start : start + following.start()] if following else readme[start:]
        return set(re.findall(r'^- `([^`]+\.ipynb)`', section, re.MULTILINE))

    def test_readme_states_the_rule_in_the_d04_words(self):
        self.assertIn(ISOLATION_RULE, README_PATH.read_text(encoding='utf-8'))

    def test_readme_lists_each_notebook_under_its_mode(self):
        readme = README_PATH.read_text(encoding='utf-8')
        listed = {
            'snapshot': self._listed_notebooks(readme, README_SNAPSHOT_HEADING),
            'fresh': self._listed_notebooks(readme, README_FRESH_HEADING),
        }
        actual = {mode: set() for mode in SCRATCH_MODES}
        for path in sorted(NOTEBOOK_DIR.glob('*.ipynb')):
            actual[notebook_mode(_load_notebook(path))].add(path.name)
        self.assertEqual(listed['snapshot'], actual['snapshot'])
        self.assertEqual(listed['fresh'], actual['fresh'])
        self.assertEqual(listed['snapshot'] | listed['fresh'], set(EXPECTED_NOTEBOOKS))
        self.assertFalse(listed['snapshot'] & listed['fresh'])
        self.assertEqual(
            listed['snapshot'], {'project_observation_calendar_demo.ipynb', 'reconcile_campaign_runs_demo.ipynb'}
        )
