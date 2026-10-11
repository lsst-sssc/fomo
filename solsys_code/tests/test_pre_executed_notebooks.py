"""Guard tests: every Django-backed pre-executed notebook runs on a scratch database through the shared helper.

Why this module exists (UAT G-37.1-3, then review findings WR-13, WR-14 and WR-17 of the 37.1 review,
WR-01 of the Phase 40 review, WR-01 of the 2026-10-10 re-review, CR-01 and CR-02 of the third review, CR-01, WR-01 and
WR-02 of the fourth review, and CR-01, CR-02, WR-01 and WR-02 of the fifth review):
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
comment that merely mentions a name cannot satisfy or trip a check.

What the guard checks: R0, every code cell must be Python that IPython runs unchanged, apart from IPython's autocall
(below). Once ``%matplotlib inline`` lines and a first-line ``%%time`` or ``%%capture`` (with only its
``--no-stderr``, ``--no-stdout`` and ``--no-display`` flags and an output name) are removed, the cell must parse, and
IPython's own static input transformer (``TransformerManager``) must leave its meaning unchanged, so an alias, an
escape on any line (also after a line magic, as in ``%time !ls``), any other magic and a prompt prefix (``In [1]:``)
are reported. ``%config`` is not harmless, because ``%config AliasManager.user_aliases = [...]`` defines a shell alias
that a later one-line cell runs; ``%matplotlib`` with any other argument is not harmless either, because a backend
such as ``module://name`` imports a module by name. A one-line cell whose first word, read with IPython's
``split_user_input``, is an IPython line magic or alias (the set comes from IPython itself, including the kernel's and
storemagic's magics) is reported, because IPython's automagic runs it unless that name is bound when the cell runs,
which a static check cannot tell: a bare annotation, an ``if False:`` branch, an empty loop or an ``except ... as``
name binds nothing, and ``del`` unbinds, so the cell is reported whatever an earlier cell assigns, imports or defines.
Only a first word that is a Python builtin or keyword, or one followed by ``=`` or ``,``, is not reported, because
IPython always runs such a line as Python. R0 does not see one rewrite, IPython's autocall: a one-line cell that
starts with an autocall object runs as a call (``exit -1`` runs as ``exit(-1)``). The kernel's only autocall objects,
``exit`` and ``quit``, end the kernel; R12 reports the name ``IPyAutocall``; and an autocall object a notebook makes
without that name (from a subclass of ``type(exit)``) runs its own call method, which is notebook code the rules read
like any other. A cell that does not parse as written and mentions Django, a database, a process, a shell command or
the environment (an IPython shell escape, a process cell magic such as ``%%bash``, a process or environment line magic
such as ``%system`` or ``%env``, ``manage.py``, ``env -i``, ``env -u`` or any name R12 lists) is still reported by its
own message. Every later rule reads each cell's Python view, so the code under a harmless magic is read like any other
code. Whether R2-R4 and R8-R11 apply is decided by ``is_django_backed`` from the cells that parse as written, before
any magic line is removed (D-13): a notebook whose only import of Django, the project app or a TOM plugin sits in a
cell under a harmless magic gets only the rules every notebook gets, and every real notebook makes such an import in a
cell that parses as written, as the not-vacuous test requires. R2-R4 and R8-R11: the helper call comes first and once,
no own Django setup, one fresh run with evidence and teardown output, and no ``assert`` guarding a write. R5 and R7:
no str or bytes constant in a cell's code (a bytes constant is read as latin-1) names the database variable
``FOMO_DATABASE_PATH`` or the developer database file ``fomo_db.sqlite3``, and no keyword argument is named
``FOMO_DATABASE_PATH``. R6, no byte copy or byte write, reads a cell's code and the code inside its strings, to two
levels as R12 does. It reports the module names ``shutil`` and ``_shutil`` (tempfile's name for shutil), shutil's
``copy2``, ``copyfile``, ``copytree`` and ``copyfileobj``, and ``write_bytes`` at every identifier the parser records,
as R12 matches its names, and as a '.'- or ':'-separated part of a str or bytes constant, so code that names shutil is
reported however it then hands the module on (to a parameter, a lambda, a ``match`` capture or a class pattern). It
also reports ``copy`` and ``move``, which are also common method names (a DataFrame's ``copy``), when read as an
attribute of an expression that mentions a shutil name, called or not, when imported with ``from shutil import``, and
as a string part (``getattr(shutil, 'move')``). A shutil name is ``shutil``, ``_shutil`` or a name the notebook binds
to one in any cell through ``import shutil as X`` or ``from <any module> import shutil [as X]``, or as the target of
an assignment, a loop, a ``with``, a comprehension or a parameter default whose value mentions a shutil name, where a
mention is a name, an attribute or a string part, so ``zipfile.shutil`` and ``sys.modules['shutil']`` count. Not
reported: ``copy`` or ``move`` read from anything that mentions no shutil name, such as another object's method of
that name (a DataFrame's ``copy``, or a path's ``copy`` and ``move`` on Python 3.14 and later, which no Python this
project supports has); a shutil module, a shutil function or a ``write_bytes`` method that the notebook obtains at
runtime without writing its name (from ``inspect.getmodule``, a function's ``__module__``, or a search over
``sys.modules`` or a module's ``vars()``); and code in a string nested more than two strings deep, which R6 and R12
match only by its '.'- and ':'-separated parts. R12 reports, in every notebook, any use of five capabilities that can
route around the scratch database (WR-01, D-G1): Django's database settings and connection objects (DATABASES,
DATABASE_ROUTERS, settings_dict, override_settings, modify_settings, the helper's internals, DatabaseWrapper,
load_backend, ``connections``, ``connection`` and ``_connections``, ConnectionHandler, create_connection,
get_connection_params, get_new_connection, ``Database``, ``connections.settings`` and ``connections.databases``, and
any ``sqlite3`` import); attaching another database through raw SQL (a string holding ATTACH followed by DATABASE, a
quote or a placeholder, or by an opening parenthesis, ``ATTACH (``); starting a process (subprocess, multiprocessing,
pty, pexpect, ptyprocess, psutil, Jupyter's kernel clients jupyter_client, nbclient and nbconvert, ``system``,
``popen``, ``fork``, ``forkpty``, ``startfile``, ``posix_spawn``, ``posix_spawnp``, every ``exec*`` and ``spawn*``
function of ``os``, Popen, ProcessPoolExecutor, asyncio's ``create_subprocess_exec`` and ``create_subprocess_shell``,
``getoutput``, ``getstatusoutput``, IPython's ``get_ipython``, ``system_raw``, ``run_line_magic`` and
``run_cell_magic``, and the IPython.utils helpers ``getoutputerror``, ``get_output_error_code`` and
``process_handler``); the environment (``os.environ``, ``os.environb``, ``os.putenv``, ``os.unsetenv``; ``os.getenv``
reads are allowed); and running code from a string or another file (exec, eval, compile, ``__import__``,
``__builtins__``, importlib, runpy, builtins, ctypes, cffi, execute_from_command_line, ManagementUtility, the lookup
functions ``import_string``, ``resolve_name`` and ``locate``, IPython's ``Macro`` and ``IPyAutocall``, and
``call_command`` of shell, shell_plus or runscript). Each R12 name is matched at every identifier the parser records
-- a name, an attribute, a keyword argument, a parameter, a function, class or exception name, a global or nonlocal
name, a match capture or class-pattern attribute, a type parameter, every part of an imported module's path, and every
imported name and alias, with attributes and class-pattern attributes never matched against the builtins exec, eval,
compile, ``__import__`` and ``__builtins__`` (so ``re.compile`` stays allowed) -- and against every '.'- or
':'-separated part of a string or bytes constant; a string that is itself Python code (it parses and holds a statement
or a call) is read like the cell's own code, to two levels. So importing a listed name from another module (``from
webbrowser import subprocess``), binding it to another name first (``c = connections``, ``run = os.system``) or
looking it up by a literal string (``import_string('subprocess.run')``) is reported too. The guard makes no claim of
completeness.

What the guard does not see: it reads only the notebook's own source, so deliberate obfuscation is out of its
reach. (1) Values assembled or looked up at runtime: a name, module, SQL statement or file path built from pieces
(string concatenation, f-strings, joined path parts, ``str.join``, ``chr``) or fetched with ``getattr``,
``globals()``, ``vars()`` or ``sys.modules`` using such a string, and code decoded at runtime (pickle, marshal,
codecs, base64). (2) Code outside the notebook: a module it imports, IPython's profile and start-up files (with the
extensions, aliases and magics they define), or a library function that starts a process or writes a file under a
name these lists do not contain (for example a path's ``replace`` or ``rename``, ``open`` in a write mode, or a
``webbrowser`` controller given a command line). (3) Left out on purpose: JavaScript or HTML in a cell's output,
which a notebook front end may run when the notebook is opened; an nbconvert regeneration does not run it.
"""

import ast
import builtins
import copy
import functools
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from keyword import iskeyword
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
REROUTE_NAMES = frozenset(
    {
        'DATABASES',
        'DATABASE_ROUTERS',
        'settings_dict',
        'override_settings',
        'modify_settings',
        'developer_database_path',
        'prepare_scratch_database',
        'snapshot_database',
        'DatabaseWrapper',
        'load_backend',
        'sqlite3',
        # D-G1 (b): a fresh connection wrapper stored into ``connections`` replaces the scratch database.
        'connections',
        'ConnectionHandler',
        'create_connection',
        # Replacing either method on a connection wrapper re-points it at another database.
        'get_connection_params',
        'get_new_connection',
        # D-G6 (a): Django's connection proxy, the handler behind it, and the DB-API module a wrapper connects through.
        'connection',
        '_connections',
        'Database',
    }
)
# R12 (D-G1 a): modules that start a process or drive a child terminal or kernel; the import is what is reported.
PROCESS_MODULES = frozenset(
    {
        'subprocess',
        'multiprocessing',
        'pty',
        'pexpect',
        'ptyprocess',
        'psutil',
        'jupyter_client',
        'nbclient',
        'nbconvert',
    }
)
# R12 (D-G1 a): names that start a process, matched as a name, an attribute, an imported name or a string.
PROCESS_NAMES = frozenset(
    {
        'Popen',
        'ProcessPoolExecutor',
        'system',
        'popen',
        'fork',
        'forkpty',
        'startfile',
        'posix_spawn',
        'posix_spawnp',
        *(f'exec{suffix}' for suffix in ('l', 'le', 'lp', 'lpe', 'v', 've', 'vp', 'vpe')),
        *(f'spawn{suffix}' for suffix in ('l', 'le', 'lp', 'lpe', 'v', 've', 'vp', 'vpe')),
        'create_subprocess_exec',
        'create_subprocess_shell',
        'getoutput',
        'getstatusoutput',
        # IPython's own shell access.
        'system_raw',
        'run_line_magic',
        'run_cell_magic',
        'get_ipython',
        # IPython.utils helpers that start a shell.
        'getoutputerror',
        'get_output_error_code',
        'process_handler',
    }
)
# R12 (D-G1 c): names that read or change the environment the helper set (``os.getenv`` stays allowed).
ENVIRON_NAMES = frozenset({'environ', 'environb', 'putenv', 'unsetenv'})
# R12: modules that run code loaded from a string, another file or a native library; the import is reported.
DYNAMIC_CODE_MODULES = frozenset({'importlib', 'runpy', 'builtins', 'ctypes', 'cffi'})
# R12: builtins that run code from a string, matched as a name, an imported name or a string, never as an attribute
# (so ``re.compile`` and a DataFrame's ``eval`` stay allowed).
DYNAMIC_CODE_BUILTINS = frozenset({'exec', 'eval', 'compile', '__import__', '__builtins__'})
# R12: Django entry points that run a command line, and so any management command, in this process.
DYNAMIC_CODE_NAMES = frozenset(
    {
        'execute_from_command_line',
        'ManagementUtility',
        # Functions that look a callable up by a dotted string.
        'import_string',
        'resolve_name',
        'locate',
        # IPython runs a Macro's text when a one-line cell names it.
        'Macro',
        # IPython calls an IPyAutocall object named first on a one-line cell (``exit -1`` runs as ``exit(-1)``), a
        # rewrite R0 cannot see (WR-02 of the fifth review, D-G13).
        'IPyAutocall',
    }
)
# R12: management commands that run code (``shell`` runs a string with ``exec``); ``call_command`` of one is reported.
CODE_RUNNING_COMMANDS = frozenset({'shell', 'shell_plus', 'runscript'})
# Every name R12 lists, so R0's whole-word search of an unparseable cell and R12 cannot drift apart.
R12_ALL_NAMES = (
    REROUTE_NAMES
    | PROCESS_MODULES
    | PROCESS_NAMES
    | ENVIRON_NAMES
    | DYNAMIC_CODE_MODULES
    | DYNAMIC_CODE_BUILTINS
    | DYNAMIC_CODE_NAMES
)
# A string constant that attaches another database to the connection through raw SQL; SQLite takes any expression
# after ATTACH, so an opening parenthesis (with or without whitespace) counts too (D-G6 a).
ATTACH_SQL = re.compile(r"\battach(?:\s+(?:database\b|['\"?:@$])|\s*\()", re.IGNORECASE)
# The reason each R12 capability gives in its problem text; 'database' is the text rule R12 always had.
R12_REASONS = {
    'database': f'only {SETUP_FUNCTION} may choose the database',
    'process': 'a notebook never starts a process, which would run without the scratch database settings',
    'environment': 'a notebook never changes the environment the helper set; read a variable with os.getenv',
    'dynamic code': (
        'a notebook never runs code built from a string or loaded from another file, which the guard cannot read'
    ),
}
# R12: attribute chains (matched when they equal, or end with ``.`` plus, one of these) that reach the same dicts.
REROUTE_DOTTED_SUFFIXES = ('connections.settings', 'connections.databases')
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
# R6: shutil functions that byte-copy or move a file or a whole directory (D-G6 b).
FORBIDDEN_COPY_NAMES = ('copy', 'copy2', 'copyfile', 'copytree', 'copyfileobj', 'move')
FORBIDDEN_COPY_CALLS = tuple(f'shutil.{name}' for name in FORBIDDEN_COPY_NAMES)
# R6: shutil copy functions whose names nothing else in a notebook uses, so R6 reports them wherever they are
# written; ``copy`` and ``move`` are also common method names and stay tied to shutil names (WR-01, fourth review).
DISTINCTIVE_COPY_NAMES = ('copy2', 'copyfile', 'copytree', 'copyfileobj')
# R6: methods that write a file's bytes, so a scratch file's bytes can land on the developer database (D-G6 b).
BYTE_WRITE_NAMES = ('write_bytes',)
# R6: the shutil module's name and tempfile's name for it. A notebook never needs shutil (the helper does every
# copy), so R6 reports either wherever it is written, which also covers shutil handed on to a parameter, a lambda or
# a match pattern (CR-01 of the fifth review, D-G11).
SHUTIL_MODULE_NAMES = ('shutil', '_shutil')
# How many levels of code inside a string (a string holding code that holds a string of code) the rules read.
STRING_CODE_DEPTH = 2
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
# An IPython shell escape (``!ls``, ``files = !ls``, but not ``a != b``) or a line magic that starts a process
# or changes the environment, in a cell that is not parseable Python (D-G1 d).
SHELL_ESCAPE = re.compile(
    r'^\s*!(?!=)|=\s*!(?!=)|'
    r'^\s*%%?(?:system|sx|sc|bash|sh|script|python|python2|python3|pypy|perl|ruby|run|env|set_env)\b',
    re.MULTILINE,
)
# Any name R12 lists, as a whole word, so ``retrieval`` or ``execute_query`` do not trip ``eval`` or ``execute``.
UNPARSEABLE_SUSPECT_NAMES = re.compile(r'\b(?:' + '|'.join(re.escape(name) for name in sorted(R12_ALL_NAMES)) + r')\b')
UNPARSEABLE_SUSPECT_TEXT = (
    'django.setup',
    DB_ENV_VAR,
    'shutil.copy',
    DEVELOPER_DB_NAME,
    *REROUTE_DOTTED_SUFFIXES,
    # D-G1 (d): the project's management entry point and a cleared child environment.
    'manage.py',
    'env -i',
    'env -u',
)
# R0 allow-list (D-G5 2): a whole line ``%matplotlib inline``; any other backend argument, such as module://name,
# imports a module by name.
HARMLESS_LINE_MAGIC = re.compile(r'^%matplotlib inline[ \t]*$', re.MULTILINE)
# R0 allow-list: a first-line ``%%time`` alone (IPython refuses a statement after it) or ``%%capture`` with only its
# --no-stderr, --no-stdout and --no-display flags and an output name. IPython honours a cell magic only on the first
# line. ``%config`` is not harmless: ``%config AliasManager.user_aliases = [...]`` defines a shell alias that a later
# one-line cell runs, and ``%config`` reconfigures other magics.
HARMLESS_CELL_MAGIC = re.compile(
    r'\A%%(?:time|capture(?:[ \t]+(?:--no-stderr|--no-stdout|--no-display|[A-Za-z_]\w*))*)[ \t]*$', re.MULTILINE
)
# The harmless magic lines, in the words R0's problem text uses.
HARMLESS_MAGICS_TEXT = '`%matplotlib inline` and a first-line `%%time` or `%%capture`'
# Magic aliases InteractiveShell.init_magics registers at start-up, on no Magics class.
IPYTHON_STARTUP_MAGIC_ALIASES = frozenset({'ed', 'hist', 'rep'})
# Text R0's two Python-but-not-as-read problems carry, so a test can tell them from the parse problems.
R0_REWRITE_MARKER = '(R0: IPython rewrites'
R0_AUTOMAGIC_MARKER = '(R0: automagic'
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


def _python_view(source: str) -> str:
    """Return the Python IPython would run for a cell: the source when it parses, else without harmless magic lines.

    The harmless lines are ``%matplotlib inline`` and a first-line ``%%time`` or ``%%capture``
    (HARMLESS_LINE_MAGIC, HARMLESS_CELL_MAGIC). Each removed line is left blank, so line numbers stay put.
    """
    if _parse(source) is not None:
        return source
    return HARMLESS_LINE_MAGIC.sub('', HARMLESS_CELL_MAGIC.sub('', source, count=1))


@functools.cache
def _ipython_transformer():
    """Return IPython's own static input transformer (prompt stripping, dedent, escapes, magics); needs no shell."""
    from IPython.core.inputtransformer2 import TransformerManager

    return TransformerManager()


def _ipython_rewrites(view: str) -> bool:
    """Return True unless IPython's static input transformer leaves the meaning of ``view`` unchanged.

    ``In [1]: rm -r -- 'x'`` parses as an annotated assignment, but IPython strips the prompt and runs the alias,
    so parsing alone is not enough: the transformed text must parse to the same tree.

    Args:
        view: A cell's Python view, which parses.

    Returns:
        True when the transformed text does not parse or parses to a different tree.
    """
    transformed = _parse(_ipython_transformer().transform_cell(view))
    return transformed is None or ast.dump(transformed) != ast.dump(ast.parse(view))


@functools.cache
def _ipython_line_magic_names() -> frozenset[str]:
    """Return every IPython line magic and alias name that automagic can run from a one-line cell.

    Built from IPython itself: the line magics of every Magics class in ``IPython.core.magics``, of the default
    storemagic extension and of ipykernel's KernelMagics, the default shell aliases, and the magic aliases a shell
    registers at start-up (IPYTHON_STARTUP_MAGIC_ALIASES). A child-process test pins it against a fresh shell.
    """
    from ipykernel.zmqshell import KernelMagics
    from IPython.core import magics
    from IPython.core.alias import default_aliases
    from IPython.core.magic import Magics
    from IPython.extensions.storemagic import StoreMagics

    classes = [value for value in vars(magics).values() if isinstance(value, type) and issubclass(value, Magics)]
    names = set(IPYTHON_STARTUP_MAGIC_ALIASES)
    for magics_class in (*classes, StoreMagics, KernelMagics):
        names.update(magics_class.magics.get('line', {}))
    names.update(name for name, _command in default_aliases())
    return frozenset(names)


def _automagic_word(view: str) -> str | None:
    """Return the magic or alias IPython's automagic would run for a one-line cell, or None.

    IPython applies automagic only to a cell that is one line, when its first word (as ``split_user_input`` reads
    it) is a line magic or alias that is not shadowed by a name bound when the cell runs, a builtin or a keyword
    (AutoMagicChecker), and is not followed by ``=`` or ``,`` (AssignmentChecker). ``rm -r -- 'x'`` is valid
    Python, yet IPython runs ``rm``. Whether a name is bound when the cell runs cannot be read from the source: a
    bare annotation, an ``if False:`` branch, an empty loop target or an ``except ... as`` name binds nothing, and
    ``del`` unbinds. So the word is reported whatever the notebook binds; only a builtin, a keyword and a following
    ``=`` or ``,`` stay exempt, because they hold at runtime whatever the notebook's source says. Blank lines do
    not count, so the check errs towards reporting.

    Args:
        view: A cell's Python view, which parses and which IPython does not rewrite.

    Returns:
        The first word when IPython may run it as a magic or alias, otherwise None.
    """
    lines = [line for line in view.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    from IPython.core.splitinput import split_user_input

    _pre, escape, word, rest = split_user_input(lines[0])
    if (
        escape
        or not word.isidentifier()
        or iskeyword(word)
        or word in vars(builtins)
        or rest.lstrip()[:1] in ('=', ',')
    ):
        return None
    return word if word in _ipython_line_magic_names() else None


def _constant_text(node: ast.AST) -> str | None:
    """Return the text of a str constant, a bytes constant decoded as latin-1, or None for any other node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            return node.value.decode('latin-1')
    return None


def _string_parts(text: str) -> list[str]:
    """Split a string on '.' and ':', the separators of a dotted path (``os.system``, ``subprocess:run``).

    A string with neither separator is its own only part, so matching parts keeps the exact-string match.
    """
    return re.split(r'[.:]', text)


def _parse_string_code(text: str) -> ast.Module | None:
    """Parse a string's text as Python and return the tree when it is code, otherwise None.

    A string counts as code when it parses and holds a statement other than a bare expression, or a call. A string
    that parses only as an expression with no call (prose such as ``'subprocess-free'``, a dotted path, a date
    ``'2026-10-10'``) runs nothing even when a library runner executes it, and its dotted parts are matched anyway.
    """
    with warnings.catch_warnings():
        # A string holding code may hold invalid escape sequences, which only warn.
        warnings.simplefilter('ignore')
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None
    is_code = any(not isinstance(statement, ast.Expr) for statement in tree.body) or any(
        isinstance(node, ast.Call) for node in ast.walk(tree)
    )
    return tree if is_code else None


def _string_code_trees(tree: ast.AST, depth: int = STRING_CODE_DEPTH):
    """Yield the parsed tree of every str or bytes constant in ``tree`` that is itself Python code.

    Code in a literal string runs as code under ``timeit``, ``pdb``, ``cProfile``, ``code.InteractiveInterpreter``
    and similar library runners the R12 lists do not name, so the rules read it like the cell's own code. Strings
    inside those strings are read too, down to ``depth`` levels.

    Args:
        tree: A parsed cell, or a tree parsed from a string.
        depth: How many levels of string code to read below ``tree``.

    Yields:
        Each string-code tree, outer strings before the strings they hold.
    """
    if depth <= 0:
        return
    for node in ast.walk(tree):
        text = _constant_text(node)
        inner = None if text is None else _parse_string_code(text)
        if inner is not None:
            yield inner
            yield from _string_code_trees(inner, depth - 1)


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
    """R5 and R7: no code string or keyword names the database variable or the developer database file.

    No str or bytes constant (bytes read as latin-1) in a cell's code names the database variable or the developer
    database file, and no keyword argument is named after the variable.
    """
    problems = []
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        for node in ast.walk(tree):
            text = _constant_text(node)
            if text is not None:
                if DB_ENV_VAR in text:
                    problems.append(f'code cell {number} names {DB_ENV_VAR} in code; only {SETUP_FUNCTION} may route')
                if DEVELOPER_DB_NAME in text:
                    problems.append(f'code cell {number} names {DEVELOPER_DB_NAME} in code; only the helper may')
            elif isinstance(node, ast.keyword) and node.arg == DB_ENV_VAR:
                problems.append(f'code cell {number} passes a {DB_ENV_VAR} keyword; only {SETUP_FUNCTION} may route')
    return problems


def _mentions_shutil(node: ast.AST, names: set[str]) -> bool:
    """Return True when an expression names the shutil module or a name bound to it.

    Args:
        node: The expression to read, for example the value an attribute is read from.
        names: The names that stand for shutil, from ``_shutil_names``.

    Returns:
        True when any node under ``node`` is a name in ``names``, an attribute whose attribute name is in
        ``names``, or a str or bytes constant one of whose '.'/':' parts is in ``names``. The last two matter
        because ``zipfile.shutil`` and ``sys.modules['shutil']`` reach the module without a plain name.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            return True
        if isinstance(child, ast.Attribute) and child.attr in names:
            return True
        text = _constant_text(child)
        if text is not None and any(part in names for part in _string_parts(text)):
            return True
    return False


def _binding_pairs(node: ast.AST):
    """Yield ``(target, value)`` for every way ``node`` binds names to a value (assignment-shaped bindings)."""
    if isinstance(node, ast.Assign):
        for target in node.targets:
            yield target, node.value
    elif isinstance(node, ast.AnnAssign):
        if node.value is not None:
            yield node.target, node.value
    elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
        yield node.target, node.value
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        yield node.target, node.iter
    elif isinstance(node, ast.withitem):
        if node.optional_vars is not None:
            yield node.optional_vars, node.context_expr
    elif isinstance(node, ast.arguments):
        positional = [*node.posonlyargs, *node.args]
        for parameter, default in zip(positional[len(positional) - len(node.defaults) :], node.defaults, strict=True):
            yield parameter, default
        for parameter, default in zip(node.kwonlyargs, node.kw_defaults, strict=True):
            if default is not None:
                yield parameter, default


def _target_names(target: ast.AST) -> set[str]:
    """Return every name and every str or bytes constant text under a binding target, and a parameter's name."""
    if isinstance(target, ast.arg):
        return {target.arg}
    names = set()
    for child in ast.walk(target):
        if isinstance(child, ast.Name):
            names.add(child.id)
        text = _constant_text(child)
        if text is not None:
            names.add(text)
    return names


def _shutil_names(trees) -> set[str]:
    """Return the names that stand for the shutil module anywhere in a notebook's code (WR-01, D-G7 2, D-G11).

    Starts from ``shutil`` and ``_shutil`` (tempfile's name for it, SHUTIL_MODULE_NAMES) and adds, over every tree
    given (every cell and the code inside its strings), the alias
    of every ``import shutil as X`` and of every ``from <any module> import shutil [as X]``, and every name or
    string in the target of an assignment (plain, chained, tuple, annotated with a value, augmented or ``:=``), a
    ``for`` loop, a comprehension, a ``with ... as`` or a parameter default whose value mentions a shutil name
    (``_mentions_shutil``). Repeats until the set stops growing, so ``a = shutil`` then ``b = a`` is followed, and
    a binding in any cell counts for every cell (it errs towards reporting).

    Not followed for ``copy`` and ``move``: shutil handed on as a call argument, to a lambda, through a match
    capture or class pattern, as a function's return value or through a call's side effect (``setattr``, a
    container's ``append``). Each of those names ``shutil`` or ``_shutil``, which R6 reports wherever it is written
    (SHUTIL_MODULE_NAMES).

    Args:
        trees: Every parsed cell tree and every tree parsed from code inside a string.

    Returns:
        The set of names, always including ``shutil`` and ``_shutil``.
    """
    nodes = [node for tree in trees for node in ast.walk(tree)]
    names = set(SHUTIL_MODULE_NAMES)
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or 'shutil' for alias in node.names if alias.name == 'shutil')
    pairs = [pair for node in nodes for pair in _binding_pairs(node)]
    grew = True
    while grew:
        grew = False
        for target, value in pairs:
            if _mentions_shutil(value, names) and not _target_names(target) <= names:
                names |= _target_names(target)
                grew = True
    return names


def _byte_copy_problems(parsed) -> list[str]:
    """R6: no byte copy or byte write that can land on a database, under any spelling.

    A byte copy can catch a database mid-write, and the scratch file has the developer database's base name, so a
    copy or write of a whole directory or file can land on it without naming it. Reported in the cell's code and in
    code inside its strings (read to STRING_CODE_DEPTH levels):

    - SHUTIL_MODULE_NAMES, the distinctive copy functions (DISTINCTIVE_COPY_NAMES) and BYTE_WRITE_NAMES at
      every identifier the parser records (names, attributes, import paths and aliases, keyword arguments,
      parameters, def and class names, match captures and class-pattern attributes), as R12 matches its names,
      and as a '.'/':' part of a str or bytes constant, so shutil handed to a parameter, a lambda or a match
      pattern is reported by its name, and a write_bytes method is reported however it is reached;
    - ``copy`` and ``move`` (and the distinctive names too) read as an attribute of an expression that mentions a
      shutil name, called or not: ``shutil`` itself, ``_shutil``, a name bound to either by an import or any
      assignment-shaped binding in any cell (``_shutil_names``), ``zipfile.shutil``, ``sys.modules['shutil']``;
    - a ``from shutil import`` of a copy function;
    - any string whose dotted parts name a copy or write function (``getattr(shutil, 'copytree')``).

    One problem per finding per cell, in cell order; a function a call, reference or import problem already names
    in a cell is not named again by the identifier pass, and a cell has at most one write_bytes problem.

    Not reported: ``copy`` or ``move`` read from anything that mentions no shutil name (another object's method of
    that name, such as a DataFrame's copy, or Python 3.14's Path.copy and Path.move); a shutil module, shutil
    function or write_bytes method the notebook obtains at runtime without writing its name (through
    inspect.getmodule, a function's ``__module__``, or a search over sys.modules or a module's vars()); and code
    nested deeper than STRING_CODE_DEPTH strings, which is matched only by its '.'/':' parts.
    """
    scanned_by_cell = [
        (number, (tree, *_string_code_trees(tree))) for number, _cell, tree in parsed if tree is not None
    ]
    shutil_names = _shutil_names([scanned for _number, trees in scanned_by_cell for scanned in trees])
    problems = []
    for number, scanned_trees in scanned_by_cell:
        found: dict[str, None] = {}
        named: set[str] = set()
        for scanned in scanned_trees:
            called = set()
            for node in ast.walk(scanned):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr in FORBIDDEN_COPY_NAMES
                        and _mentions_shutil(func.value, shutil_names)
                    ):
                        found[
                            f'code cell {number} calls {ast.unparse(func)}(); '
                            'a notebook must never byte-copy a database'
                        ] = None
                        called.add(id(func))
                        named.add(func.attr)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'shutil':
                    names = [alias.name for alias in node.names if alias.name in (*FORBIDDEN_COPY_NAMES, '*')]
                    if names:
                        found[
                            f'code cell {number} imports a shutil copy function ({", ".join(names)}); '
                            'a notebook must never copy'
                        ] = None
                        named.update(names)
            for node in ast.walk(scanned):
                if (
                    isinstance(node, ast.Attribute)
                    and id(node) not in called
                    and node.attr in FORBIDDEN_COPY_NAMES
                    and _mentions_shutil(node.value, shutil_names)
                ):
                    found[
                        f'code cell {number} references {ast.unparse(node)}; a notebook must never byte-copy a database'
                    ] = None
                    named.add(node.attr)
                name = node.attr if isinstance(node, ast.Attribute) else getattr(node, 'id', None)
                if isinstance(node, (ast.Attribute, ast.Name)) and name in BYTE_WRITE_NAMES:
                    found[
                        f'code cell {number} references {name}; '
                        'a notebook must never byte-copy or byte-write a file over a database'
                    ] = None
                text = _constant_text(node)
                for part in _string_parts(text) if text is not None else ():
                    if part in FORBIDDEN_COPY_NAMES or part in BYTE_WRITE_NAMES:
                        found[
                            f'code cell {number} names {part} in a string; '
                            'a notebook must never look up a byte-copy or byte-write function by name'
                        ] = None
                    elif part in SHUTIL_MODULE_NAMES:
                        found[
                            f'code cell {number} names {part} in a string; '
                            'a notebook never uses shutil, whose functions byte-copy a database'
                        ] = None
        # The shutil module names, the distinctive copy names and the write names are reported at every identifier,
        # unless a problem above already names the function. A write name gets exactly the first pass's text, so a
        # cell never has two write_bytes problems.
        for scanned in scanned_trees:
            for node in ast.walk(scanned):
                for identifier, _groups in _identifiers(node):
                    if identifier in BYTE_WRITE_NAMES:
                        found[
                            f'code cell {number} references {identifier}; '
                            'a notebook must never byte-copy or byte-write a file over a database'
                        ] = None
                    elif identifier in SHUTIL_MODULE_NAMES and identifier not in named:
                        found[
                            f'code cell {number} references {identifier}; '
                            'a notebook never uses shutil, whose functions byte-copy a database'
                        ] = None
                        named.add(identifier)
                    elif identifier in DISTINCTIVE_COPY_NAMES and identifier not in named:
                        found[
                            f'code cell {number} references {identifier}; a notebook must never byte-copy a database'
                        ] = None
                        named.add(identifier)
        problems.extend(found)
    return problems


# R12: (names, capability) pairs matched as a name, an attribute, a keyword, an imported name or an exact string.
_R12_NAME_TABLES = (
    (REROUTE_NAMES, 'database'),
    (PROCESS_NAMES, 'process'),
    (ENVIRON_NAMES, 'environment'),
    (DYNAMIC_CODE_NAMES, 'dynamic code'),
)
# R12: (module names, capability) pairs matched as an import's first component, an attribute or an exact string.
_R12_MODULE_TABLES = ((PROCESS_MODULES, 'process'), (DYNAMIC_CODE_MODULES, 'dynamic code'))
# R12: builtins matched as a name, an imported name or an exact string, but never as an attribute.
_R12_BUILTIN_TABLES = ((DYNAMIC_CODE_BUILTINS, 'dynamic code'),)
# R12: the table groups an identifier is matched against -- every table, or every table but the builtins for an
# attribute (so ``re.compile`` and a DataFrame's ``eval`` stay allowed).
_R12_EVERY_TABLE = (_R12_NAME_TABLES, _R12_MODULE_TABLES, _R12_BUILTIN_TABLES)
_R12_ATTRIBUTE_TABLES = (_R12_NAME_TABLES, _R12_MODULE_TABLES)
# Type parameter nodes (``def f[T]()``), which only Python 3.12 and later define.
_TYPE_PARAM_NODES = tuple(
    node_type
    for node_type in (getattr(ast, name, None) for name in ('TypeVar', 'ParamSpec', 'TypeVarTuple'))
    if node_type is not None
)


def _identifiers(node: ast.AST):
    """Yield ``(identifier, table groups)`` for every identifier an AST node carries.

    The parser records identifiers in many places besides a plain name: attributes, imported module paths and
    names, keyword arguments, parameters, function, class and exception names, global and nonlocal names, match
    captures and class-pattern attributes, and type parameters. A listed name written in any of them is reported,
    so importing it from another module (``from webbrowser import subprocess``) or binding it under another name
    cannot hide it. An attribute, and a class-pattern attribute (``case object(subprocess=sp)``, an attribute
    lookup), is never matched against the builtins.

    Args:
        node: Any node of a parsed cell.

    Yields:
        ``(identifier, table groups)`` pairs, the groups being the R12 tables to match the identifier against.
    """
    if isinstance(node, ast.Name):
        yield node.id, _R12_EVERY_TABLE
    elif isinstance(node, ast.Attribute):
        yield node.attr, _R12_ATTRIBUTE_TABLES
    elif isinstance(node, ast.alias):
        for name in (node.name, node.asname):
            for part in (name or '').split('.'):
                yield part, _R12_EVERY_TABLE
    elif isinstance(node, ast.ImportFrom):
        for part in (node.module or '').split('.'):
            yield part, _R12_EVERY_TABLE
    elif isinstance(node, (ast.keyword, ast.arg)):
        if node.arg is not None:
            yield node.arg, _R12_EVERY_TABLE
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, *_TYPE_PARAM_NODES)):
        yield node.name, _R12_EVERY_TABLE
    elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        if node.name is not None:
            yield node.name, _R12_EVERY_TABLE
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        for name in node.names:
            yield name, _R12_EVERY_TABLE
    elif isinstance(node, ast.MatchMapping):
        if node.rest is not None:
            yield node.rest, _R12_EVERY_TABLE
    elif isinstance(node, ast.MatchClass):
        for name in node.kwd_attrs:
            yield name, _R12_ATTRIBUTE_TABLES


def _capability_of(name: str | None, tables) -> str | None:
    """Return the capability of the first table in ``tables`` that lists ``name``, or None."""
    for names, capability in tables:
        if name in names:
            return capability
    return None


def _call_command_target(call: ast.Call) -> str | None:
    """Return the constant command of a ``call_command`` call (first argument or ``command_name=``), or None."""
    if _call_name(call) != 'call_command':
        return None
    command = call.args[0] if call.args else None
    for keyword in call.keywords:
        if keyword.arg == 'command_name':
            command = keyword.value
    if isinstance(command, ast.Constant) and isinstance(command.value, str):
        return command.value
    return None


def _database_rerouting_problems(parsed) -> list[str]:
    """R12: no notebook uses a capability that can route around the scratch database, under any spelling.

    The capabilities are Django's database settings and connection objects, starting a process, changing the
    environment, running code built from a string or loaded from another file, and attaching another database
    through raw SQL. Each listed name is matched at every identifier the parser records (a name, an attribute, every
    part of an imported module path, an imported name or alias, a keyword argument, a parameter, a function, class
    or exception name, a global or nonlocal name, a match capture or class-pattern attribute, a type parameter --
    attributes and class-pattern attributes never against the builtins) and as every '.' or ':' part of a str or
    bytes constant, so importing one from another module, binding it to another name first or looking it up by a
    literal dotted string is reported too. Code inside a string is read like the cell's own code
    (``_string_code_trees``).
    """
    problems = []
    for number, _cell, tree in parsed:
        if tree is None:
            continue
        found: dict[str, str] = {}

        def note(name, *table_groups, found=found):
            capability = _capability_of(name, [pair for tables in table_groups for pair in tables])
            if capability is not None:
                found[name] = capability

        nodes = itertools.chain.from_iterable(ast.walk(scanned) for scanned in (tree, *_string_code_trees(tree)))
        for node in nodes:
            for name, table_groups in _identifiers(node):
                note(name, *table_groups)
            text = _constant_text(node)
            if isinstance(node, ast.Attribute):
                dotted = _dotted_name(node)
                for suffix in REROUTE_DOTTED_SUFFIXES:
                    if dotted == suffix or dotted.endswith(f'.{suffix}'):
                        found[suffix] = 'database'
            elif text is not None:
                for part in _string_parts(text):
                    note(part, *_R12_EVERY_TABLE)
                if ATTACH_SQL.search(text):
                    found['ATTACH'] = 'database'
            elif isinstance(node, ast.Call):
                command = _call_command_target(node)
                if command in CODE_RUNNING_COMMANDS:
                    found[f"call_command('{command}')"] = 'dynamic code'
            if isinstance(node, ast.Import):
                if any('sqlite3' in alias.name.split('.') for alias in node.names):
                    found['sqlite3'] = 'database'
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and 'sqlite3' in (node.module or '').split('.'):
                found['sqlite3'] = 'database'
        problems.extend(
            f'code cell {number} references {name}; {R12_REASONS[capability]} '
            f'{REROUTE_MARKER} no re-routing after setup)'
            for name, capability in sorted(found.items())
        )
    return problems


def isolation_problems(notebook: dict) -> list[str]:
    """Return human-readable reasons ``notebook`` could touch a real database or show stale output.

    Args:
        notebook: A parsed ``.ipynb`` file (the JSON object, not a path).

    Returns:
        A list of problems, in statement order where it matters; empty when the notebook is clean. A notebook
        in which no code cell that parses as written (before any magic line is removed) imports Django, the
        project app or a TOM plugin (``is_django_backed``, D-13) is only checked for the rules that apply to every
        notebook (no own database routing, no byte copy, no developer database file name, and none of the names
        R12 lists for re-routing the database after setup); R2-R4 and R8-R11 apply only to a Django-backed
        notebook. R0 also applies to every notebook: each code cell must be Python that IPython runs unchanged once
        ``%matplotlib inline`` and a first-line ``%%time`` or ``%%capture`` are removed, and a one-line cell may not
        start with an IPython line magic or alias name, whatever an earlier cell binds (only a builtin, a keyword or
        a word followed by ``=`` or ``,`` is exempt). R0 does not see IPython's autocall of an object such as
        ``exit`` or ``quit`` (``exit -1`` runs as ``exit(-1)``), a rewrite IPython makes from the namespace at run
        time; R12 reports the IPyAutocall name. Every later rule reads each cell's Python view.
    """
    problems = []
    code_cells = _code_cells(notebook)
    parsed = _parsed_code_cells(notebook)

    # R0: a cell must be Python that IPython runs unchanged (D-G5 2), apart from IPython's autocall, which R0 cannot
    # see (D-G13). A cell that is not plain Python is reduced to its Python view (harmless magic lines removed); the
    # deny-list message comes first, then the allow-list. At most one R0 problem per cell, in cell order.
    views = []
    for number, cell, tree in parsed:
        source = _cell_source(cell)
        view = _python_view(source)
        view_tree = tree if tree is not None else _parse(view)
        if tree is None and (
            UNPARSEABLE_DJANGO_IMPORT.search(source)
            or SHELL_ESCAPE.search(source)
            or UNPARSEABLE_SUSPECT_NAMES.search(source)
            or any(text in source for text in UNPARSEABLE_SUSPECT_TEXT)
        ):
            problems.append(
                f'code cell {number} is not parseable Python but mentions Django, a database, a process, '
                'a shell command or the environment, so it cannot be checked'
            )
        elif view_tree is None:
            problems.append(
                f'code cell {number} is not parseable Python once its harmless magic lines are removed '
                f'(only {HARMLESS_MAGICS_TEXT} are allowed), so IPython would run something the guard cannot check'
            )
        elif _ipython_rewrites(view):
            problems.append(
                f'code cell {number} is Python that IPython would rewrite before running it (a prompt prefix, an '
                f'escape or a magic), so it cannot be checked {R0_REWRITE_MARKER})'
            )
        elif (word := _automagic_word(view)) is not None:
            problems.append(
                f'code cell {number} is one line that starts with {word!r}, an IPython line magic or alias name, '
                "which IPython's automagic runs as a magic or a shell command unless that name is bound when the "
                f'cell runs, so the guard reports it {R0_AUTOMAGIC_MARKER})'
            )
        views.append((number, cell, view_tree))
    # Every later rule reads the Python views, so the code under a harmless magic is read like any other cell's;
    # whether R2-R4 and R8-R11 apply is still decided from each cell as written (is_django_backed, D-13, D-G12).
    parsed = views

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
    "Django ready: settings module='src.fomo.settings'\n"
    "Resolved database: '/tmp/fomo-notebook-db-abc123/fomo_db.sqlite3' (fresh scratch database, migrated from empty)\n"
)

REFERENCE_DJANGO_CELL = 'from django.conf import settings\n'

REFERENCE_TEARDOWN = 'teardown_scratch_django(scratch_db_dir)\n'

REFERENCE_TEARDOWN_OUTPUT = (
    'Removed scratch database directory: /tmp/fomo-notebook-db-abc123\n'
    "Before django.setup() ran, the helper checked that Django's only configured database was the scratch file "
    'in that directory.\n'
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

    def test_a_bytes_literal_naming_the_developer_database_or_its_variable_is_a_problem(self):
        # WR-02 of the fourth review, D-G8 (1): R5 and R7 read bytes constants as well as str ones.
        cases = [
            ("db = open(b'src/fomo_db.sqlite3', 'rb')\n", DEVELOPER_DB_NAME),
            ("x = b'FOMO_DATABASE_PATH'\n", DB_ENV_VAR),
        ]
        for source, text in cases:
            with self.subTest(source=source):
                problems = isolation_problems(_notebook(extra_cells=[_code(source)]))
                self.assertTrue(any(text in problem and 'in code' in problem for problem in problems), problems)

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


def _insert_code_cell_after_setup(notebook: dict, source: str) -> dict:
    """Return a copy of ``notebook`` with ``source`` as a new code cell right after the setup cell.

    The new cell has no outputs, and every code cell's execution count is renumbered 1..N in order. Renumbering
    keeps rule R8 quiet, so only the rule under test can fire.

    Args:
        notebook: A parsed ``.ipynb`` file with one valid helper call.
        source: Python (or IPython) source for the new cell, each line ending in a newline.

    Returns:
        A deep copy of ``notebook``; the argument is not modified.
    """
    modified = copy.deepcopy(notebook)
    setup_number = setup_call(modified)[0]
    setup_cell = _code_cells(modified)[setup_number - 1]
    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'id': 'inserted-cell',
        'metadata': {},
        'outputs': [],
        'source': source.splitlines(keepends=True),
    }
    modified['cells'].insert(modified['cells'].index(setup_cell) + 1, new_cell)
    for position, cell in enumerate(_code_cells(modified), start=1):
        cell['execution_count'] = position
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


class TestReReviewFormsOnARealNotebook(SimpleTestCase):
    """WR-01 of the 2026-10-10 re-review, the verification gap and developer decision D-G1.

    Four plain-source ways to reach the developer database after the helper call used to pass the guard. Each
    is added to a copy of a real notebook with its committed output kept; before rule R12 learned whole
    capabilities and R0 learned shell escapes, ``isolation_problems()`` returned ``[]`` for every one of them.
    """

    FORM_A = (
        'from django.db import connections\n'
        'from django.db.utils import ConnectionHandler\n'
        "dev = str(repo_root_path / 'src' / ('fomo_db' + '.sqlite3'))\n"
        "connections['default'] = ConnectionHandler({'default': {'ENGINE': 'django.db.backends.' + 'sqlite' + '3', "
        "'NAME': dev}})['default']\n"
    )
    FORM_B = (
        'import subprocess\n'
        "subprocess.run([sys.executable, 'manage.py', 'migrate'], cwd=repo_root_path, env={'PATH': '/usr/bin'})\n"
    )
    FORM_C = (
        'import os, subprocess\n'
        'os.environ.clear()\n'
        "subprocess.run([sys.executable, str(repo_root_path / 'manage.py'), 'flush', '--noinput'])\n"
    )
    FORM_D = '!cd ../.. && env -i PATH=/usr/bin python manage.py flush --noinput\n'

    def setUp(self):
        self.notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')

    def assert_only_r12_names(self, source, expected):
        """Check that appending ``source`` to the setup cell yields only R12 problems naming ``expected``."""
        problems = isolation_problems(_append_to_setup_cell(self.notebook, source))
        self.assertTrue(problems, 'the guard missed a re-route after the helper call')
        self.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)
        self.assertTrue(expected <= _rerouting_names(problems), problems)

    def test_a_connection_handler_store_is_reported_on_a_real_notebook(self):
        self.assert_only_r12_names(self.FORM_A, {'connections', 'ConnectionHandler'})

    def test_a_child_manage_py_with_a_bare_environment_is_reported_on_a_real_notebook(self):
        self.assert_only_r12_names(self.FORM_B, {'subprocess'})

    def test_a_cleared_environment_and_a_child_manage_py_are_reported_on_a_real_notebook(self):
        self.assert_only_r12_names(self.FORM_C, {'environ', 'subprocess'})

    def test_a_shell_escape_running_manage_py_is_reported_on_a_real_notebook(self):
        with self.subTest('inserted as its own cell'):
            problems = isolation_problems(_insert_code_cell_after_setup(self.notebook, self.FORM_D))
            self.assertTrue(problems, 'the guard missed a shell escape in its own cell')
            self.assertTrue(all('not parseable' in problem for problem in problems), problems)
        with self.subTest('appended to the setup cell'):
            problems = isolation_problems(_append_to_setup_cell(self.notebook, self.FORM_D))
            self.assertTrue(any('not parseable' in problem for problem in problems), problems)


class TestThirdReviewFormsOnARealNotebook(SimpleTestCase):
    """CR-01 and CR-02 of the third review (2026-10-10), verification gap 1 and developer decision D-G5.

    Readable, literal ways to reach the developer database that the extended guard still passed: a process module
    imported from another module, a process looked up by a literal string, the connection proxy, ``ATTACH (``,
    IPython aliases, automagic and escapes after a line magic. Each is added to a copy of a real notebook with its
    committed output kept.
    """

    RUN_ARGS = "([sys.executable, 'manage.py', 'flush', '--noinput'], cwd=repo_root_path, env={'PATH': '/usr/bin'})\n"

    def setUp(self):
        self.notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')

    def assert_only_r12_names(self, source, expected):
        """Check that appending ``source`` to the setup cell yields only R12 problems naming ``expected``."""
        problems = isolation_problems(_append_to_setup_cell(self.notebook, source))
        self.assertTrue(problems, 'the guard missed a re-route after the helper call')
        self.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)
        self.assertTrue(set(expected) <= _rerouting_names(problems), problems)

    def test_subprocess_imported_from_another_module_is_reported_on_a_real_notebook(self):
        forms = {
            'webbrowser': 'from webbrowser import subprocess\nsubprocess.run' + self.RUN_ARGS,
            'autoreload': 'from django.utils.autoreload import subprocess as sp\nsp.run' + self.RUN_ARGS,
        }
        for label, source in forms.items():
            with self.subTest(label):
                self.assert_only_r12_names(source, {'subprocess'})

    def test_a_process_looked_up_by_a_literal_string_is_reported_on_a_real_notebook(self):
        forms = {
            'import_string': (
                'from django.utils.module_loading import import_string\n'
                "import_string('subprocess.run')" + self.RUN_ARGS,
                {'import_string', 'subprocess'},
            ),
            'resolve_name': (
                "import pkgutil\npkgutil.resolve_name('subprocess:run')" + self.RUN_ARGS,
                {'resolve_name', 'subprocess'},
            ),
        }
        for label, (source, expected) in forms.items():
            with self.subTest(label):
                self.assert_only_r12_names(source, expected)

    def test_ipython_process_helpers_are_reported_on_a_real_notebook(self):
        forms = {
            'getoutputerror': (
                'from IPython.utils.process import getoutputerror\n'
                "getoutputerror('cd ../.. && env --ignore-environment PATH=/usr/bin "
                "python manage.py flush --noinput')\n"
            ),
            'get_output_error_code': (
                "from IPython.utils.process import get_output_error_code\nget_output_error_code('ls')\n"
            ),
            'process_handler': (
                "from IPython.utils._process_common import process_handler\nprocess_handler('ls', print)\n"
            ),
        }
        for name, source in forms.items():
            with self.subTest(name):
                self.assert_only_r12_names(source, {name})

    def test_the_connection_proxy_database_connect_and_attach_are_reported_on_a_real_notebook(self):
        forms = {
            'CR-01 (b)': (
                'from django.db import connection\n'
                "connection._connections.settings['default']['NAME'] = '/home/someone/other.sqlite3'\n"
                'connection.close()\n',
                {'connection', '_connections'},
            ),
            'CR-01 (c)': (
                'from django.db import connection\n'
                "connection.connection = connection.Database.connect('/home/someone/other.sqlite3', "
                'isolation_level=None)\n',
                {'connection', 'Database'},
            ),
            'CR-01 (d)': (
                'from django.db import connection\n'
                'with connection.cursor() as cursor:\n'
                "    cursor.execute('ATTACH (%s) AS dev', ['/home/someone/other.sqlite3'])\n",
                {'connection', 'ATTACH'},
            ),
        }
        for label, (source, expected) in forms.items():
            with self.subTest(label):
                self.assert_only_r12_names(source, expected)

    def test_copytree_and_write_bytes_are_reported_on_a_real_notebook(self):
        forms = {
            'copytree': "import shutil\nshutil.copytree(scratch_db_dir, repo_root_path / 'src', dirs_exist_ok=True)\n",
            'write_bytes': (
                'for f in scratch_db_dir.iterdir():\n'
                "    (repo_root_path / 'src' / f.name).write_bytes(f.read_bytes())\n"
            ),
        }
        for name, source in forms.items():
            with self.subTest(name):
                problems = isolation_problems(_append_to_setup_cell(self.notebook, source))
                self.assertTrue(any(name in problem and 'byte-copy' in problem for problem in problems), problems)

    def assert_inserted_cell_reports_only(self, sources, text):
        """Check that each source, inserted as its own cell, yields problems that all contain ``text``."""
        for source in sources:
            with self.subTest(source=source):
                problems = isolation_problems(_insert_code_cell_after_setup(self.notebook, source))
                self.assertTrue(problems, 'the guard missed a cell IPython would run as something else')
                self.assertTrue(all(text in problem for problem in problems), problems)

    def test_a_shell_escape_after_a_line_magic_is_reported_on_a_real_notebook(self):
        self.assert_inserted_cell_reports_only(
            [
                '%time !cd ../.. && env --ignore-environment PATH=/usr/bin python -m django flush --noinput '
                '--settings=src.fomo.settings\n'
            ],
            'not parseable',
        )

    def test_ipython_aliases_and_shell_magics_are_reported_on_a_real_notebook(self):
        self.assert_inserted_cell_reports_only(
            [
                'cp /tmp/fomo-notebook-db-*/* ../../src/\n',
                '%cp /tmp/fomo-notebook-db-*/* ../../src/\n',
                'rm -r ../../src\n',
                '%time !rm -r ../../src\n',
                '%alias wipe rm -r ../../src\n',
                '%rehashx\n',
                '%pip install -e ../..\n',
            ],
            'not parseable',
        )

    def test_a_one_line_alias_or_magic_that_is_valid_python_is_reported_on_a_real_notebook(self):
        # Each is valid Python, but IPython's automagic runs it as an alias or a magic (rm deletes the directory).
        self.assert_inserted_cell_reports_only(
            [
                "rm -r -- '../../src'\n",
                'rehashx\n',
                'store -r\n',
                'reset -f\n',
                "sx('rm' ' -r' ' ../../src')\n",
                "%%time\nrm -r -- '../../src'\n",
            ],
            R0_AUTOMAGIC_MARKER,
        )
        # Valid Python (an annotated assignment), but IPython strips the prompt and then runs the alias.
        self.assert_inserted_cell_reports_only(["In [1]: rm -r -- '../../src'\n"], R0_REWRITE_MARKER)

    def test_harmless_magic_lines_leave_a_real_notebook_clean(self):
        for source in ('%matplotlib inline\n', '%%time\nx = 1\n', '%%capture captured\nx = 1\n'):
            with self.subTest(source=source):
                self.assertEqual(isolation_problems(_insert_code_cell_after_setup(self.notebook, source)), [])
        self.assert_inserted_cell_reports_only(
            [
                "%config InlineBackend.figure_format = 'svg'\n",
                '%matplotlib module://matplotlib_inline.backend_inline\n',
            ],
            'not parseable',
        )


def _assert_r12_reports(testcase, cases):
    """Check that each ``(source, expected names)`` case yields only R12 problems naming the expected names.

    Each case is the reference notebook plus one extra code cell holding ``source``; one subTest per case.

    Args:
        testcase: The running ``SimpleTestCase``, which owns the subTests and assertions.
        cases: ``(source, names)`` pairs; every name must appear as ``references <name>;`` in some problem.
    """
    for source, expected in cases:
        with testcase.subTest(source=source):
            problems = isolation_problems(_notebook(extra_cells=[_code(source)]))
            testcase.assertTrue(problems, 'expected the guard to report a problem')
            testcase.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)
            for name in expected:
                _first_problem_mentions(testcase, problems, f'references {name};')


class TestRerouteRuleOnSyntheticNotebooks(SimpleTestCase):
    """WR-01: each way a notebook could re-route Django's database after the helper call, built as a small notebook.

    Every case is the reference notebook plus one extra cell, so the only problems reported are R12's.
    """

    def assert_reported(self, cases):
        """Check that each ``(source, expected names)`` case yields only R12 problems naming the expected names."""
        _assert_r12_reports(self, cases)

    def test_settings_dict_in_any_spelling_is_a_problem(self):
        self.assert_reported(
            [
                ("connections['default'].settings_dict['NAME'] = x\n", ['settings_dict']),
                ("getattr(connection, 'settings_dict')['NAME'] = x\n", ['settings_dict']),
            ]
        )

    def test_databases_or_routers_in_any_spelling_is_a_problem(self):
        self.assert_reported(
            [
                ("settings.DATABASES['default']['NAME'] = x\n", ['DATABASES']),
                ("vars(settings)['DATABASES']['default']['NAME'] = x\n", ['DATABASES']),
                ('settings.configure(DATABASES={})\n', ['DATABASES']),
                ("settings.DATABASE_ROUTERS = ['a.Router']\n", ['DATABASE_ROUTERS']),
            ]
        )

    def test_connection_handler_settings_are_a_problem(self):
        # Django's ConnectionHandler returns the same dict object as settings.DATABASES from both properties.
        self.assert_reported(
            [
                ("connections.settings['default']['NAME'] = x\n", ['connections.settings']),
                ("django.db.connections.databases['default']['NAME'] = x\n", ['connections.databases']),
            ]
        )

    def test_override_or_modify_settings_is_a_problem(self):
        self.assert_reported(
            [
                ('from django.test import override_settings\n', ['override_settings']),
                ('django.test.utils.modify_settings(INSTALLED_APPS={}).enable()\n', ['modify_settings']),
                ('@override_settings(DEBUG=True)\ndef demo():\n    pass\n', ['override_settings']),
            ]
        )

    def test_helper_internals_are_a_problem(self):
        self.assert_reported(
            [
                ('from fomo_nb_utils import developer_database_path\n', ['developer_database_path']),
                ("prepare_scratch_database(repo_root_path, mode='fresh')\n", ['prepare_scratch_database']),
                ('snapshot_database(a, b)\n', ['snapshot_database']),
            ]
        )

    def test_sqlite3_or_a_backend_wrapper_is_a_problem(self):
        self.assert_reported(
            [
                ('import sqlite3\n', ['sqlite3']),
                ('import sqlite3 as lite\n', ['sqlite3']),
                ('from sqlite3 import connect\n', ['sqlite3']),
                ('import sqlite3\nsqlite3.connect(path)\n', ['sqlite3']),
                ('from django.db.backends.sqlite3.base import DatabaseWrapper\n', ['sqlite3', 'DatabaseWrapper']),
                ('from django.db.utils import load_backend\n', ['load_backend']),
            ]
        )

    def test_the_rule_applies_to_a_notebook_that_does_not_import_django(self):
        notebook = {'cells': [{'cell_type': 'code', 'source': ['import sqlite3\n'], 'outputs': []}]}
        self.assertFalse(is_django_backed(notebook))
        problems = isolation_problems(notebook)
        _first_problem_mentions(self, problems, 'references sqlite3;')
        self.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)

    def test_names_in_comments_or_markdown_are_not_a_problem(self):
        comment = _code('# settings_dict, DATABASES and sqlite3 are never touched here\nprint(1)\n')
        notebook = _notebook(extra_cells=[comment])
        notebook['cells'].append(
            {'cell_type': 'markdown', 'source': ['The helper uses override_settings and connections.settings.']}
        )
        self.assertEqual(isolation_problems(notebook), [])

    def test_an_unparseable_cell_that_reroutes_is_a_problem(self):
        source = "%%time\nconnections['default'].settings_dict['NAME'] = x\n"
        _first_problem_mentions(self, isolation_problems(_notebook(extra_cells=[_code(source)])), 'not parseable')


class TestCapabilityRulesOnSyntheticNotebooks(SimpleTestCase):
    """WR-01 and D-G1: each capability that could reach the developer database without naming it is reported.

    The capabilities are a process, the environment, a new connection, code from a string or another file and an
    attached database. Each is reported wherever it is named, so binding it to another name first is reported
    too. Every case is the reference notebook plus one extra cell, so the only problems reported are R12's.
    """

    def test_starting_a_process_in_any_spelling_is_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ('import subprocess\n', ['subprocess']),
                ('from subprocess import Popen as P\nP(cmd)\n', ['subprocess', 'Popen']),
                ('import os\nos.system(cmd)\n', ['system']),
                ('import os as o\no.system(cmd)\n', ['system']),
                ('run = os.system\n', ['system']),
                ('from os import system as s\n', ['system']),
                ('os.popen(cmd)\n', ['popen']),
                ('os.execv(a, b)\n', ['execv']),
                ('os.execlpe(a, b, env)\n', ['execlpe']),
                ('os.spawnlp(m, f, a)\n', ['spawnlp']),
                ('os.posix_spawn(p, a, e)\n', ['posix_spawn']),
                ('os.posix_spawnp(p, a, e)\n', ['posix_spawnp']),
                ('os.fork()\n', ['fork']),
                ('os.forkpty()\n', ['forkpty']),
                ('os.startfile(p)\n', ['startfile']),
                ('import pty\npty.spawn(a)\n', ['pty']),
                ('asyncio.create_subprocess_exec(a)\n', ['create_subprocess_exec']),
                ('asyncio.create_subprocess_shell(a)\n', ['create_subprocess_shell']),
                ('import multiprocessing\n', ['multiprocessing']),
                ('from concurrent.futures import ProcessPoolExecutor\n', ['ProcessPoolExecutor']),
                ('import pexpect\n', ['pexpect']),
                ('import ptyprocess\n', ['ptyprocess']),
                ('import psutil\npsutil.Popen(a)\n', ['psutil', 'Popen']),
                ('from jupyter_client import KernelManager\n', ['jupyter_client']),
                ('import nbclient\n', ['nbclient']),
                ('from nbconvert.preprocessors import ExecutePreprocessor\n', ['nbconvert']),
                ('from subprocess import getoutput\n', ['subprocess', 'getoutput']),
                ("import sys\nsys.modules['subprocess'].run(a)\n", ['subprocess']),
            ],
        )

    def test_ipython_shell_access_in_a_parseable_cell_is_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ("get_ipython().system('ls')\n", ['get_ipython', 'system']),
                ("get_ipython().getoutput('ls')\n", ['getoutput']),
                ("get_ipython().system_raw('ls')\n", ['system_raw']),
                ("get_ipython().run_line_magic('system', 'ls')\n", ['run_line_magic', 'system']),
                ("get_ipython().run_cell_magic('bash', '', 'ls')\n", ['run_cell_magic']),
                ('from IPython import get_ipython\n', ['get_ipython']),
                ('from IPython.utils.process import getoutput\n', ['getoutput']),
            ],
        )

    def test_shell_escapes_and_process_magics_are_a_problem(self):
        sources = [
            '!ls\n',
            'files = !ls\n',
            '%%bash\nls\n',
            '%%sh\nls\n',
            '%%script bash\nls\n',
            '%%python3\nprint(1)\n',
            '%system ls\n',
            '%sx ls\n',
            '%run helper.py\n',
            '%env HOME=/tmp\n',
            '%set_env HOME=/tmp\n',
            '%%time\nos.environ.clear()\n',
            "%%time\npath = 'manage.py'\n",
            "%%time\nprint('env -i')\n",
            "%%time\nprint('env -u HOME')\n",
            TestReReviewFormsOnARealNotebook.FORM_D,
        ]
        for source in sources:
            with self.subTest(source=source):
                _first_problem_mentions(
                    self, isolation_problems(_notebook(extra_cells=[_code(source)])), 'not parseable'
                )
        for source in ('%matplotlib inline\n', '%%time\nok = a != b\n'):
            with self.subTest(source=source):
                self.assertIsNone(SHELL_ESCAPE.search(source))

    def test_replacing_a_database_connection_in_any_spelling_is_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ("connections['default'] = x\n", ['connections']),
                ("connections['default'] += x\n", ['connections']),
                ("connections['default']: object = x\n", ['connections']),
                ("del connections['default']\n", ['connections']),
                ("django.db.connections['default'] = x\n", ['connections']),
                ("c = connections\nc['default'] = x\n", ['connections']),
                ('from django.db import connections as c\n', ['connections']),
                ('ConnectionHandler({})\n', ['ConnectionHandler']),
                ('from django.db.utils import ConnectionHandler\n', ['ConnectionHandler']),
                ("handler.create_connection('default')\n", ['create_connection']),
                ('transaction.get_connection().get_connection_params = f\n', ['get_connection_params']),
                ('wrapper.get_new_connection = f\n', ['get_new_connection']),
            ],
        )

    def test_changing_the_environment_in_any_spelling_is_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ('os.environ.clear()\n', ['environ']),
                ("os.environ.pop('HOME')\n", ['environ']),
                ('os.environ.popitem()\n', ['environ']),
                ("os.environ.update({'A': 'b'})\n", ['environ']),
                ("os.environ.setdefault('A', 'b')\n", ['environ']),
                ("os.environ['A'] = 'b'\n", ['environ']),
                ("del os.environ['A']\n", ['environ']),
                ("os.environ |= {'A': 'b'}\n", ['environ']),
                ("os.environb[b'A'] = b'b'\n", ['environb']),
                ("os.putenv('A', 'b')\n", ['putenv']),
                ("os.unsetenv('A')\n", ['unsetenv']),
                ('from os import environ as e\ne.clear()\n', ['environ']),
                ('e = os.environ\ne.clear()\n', ['environ']),
            ],
        )

    def test_running_code_from_a_string_or_another_file_is_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ('exec(src)\n', ['exec']),
                ('eval(src)\n', ['eval']),
                ("compile(src, 'x', 'single')\n", ['compile']),
                ("__import__('os')\n", ['__import__']),
                ('__builtins__.exec(src)\n', ['__builtins__']),
                ('f = exec\n', ['exec']),
                ('import importlib\nimportlib.import_module(name)\n', ['importlib']),
                ('from importlib import import_module\n', ['importlib']),
                ('import runpy\nrunpy.run_path(p)\n', ['runpy']),
                ('import builtins\n', ['builtins']),
                ('import ctypes\n', ['ctypes']),
                ('import cffi\n', ['cffi']),
                ("call_command('shell', command=src)\n", ["call_command('shell')"]),
                ("management.call_command('shell_plus')\n", ["call_command('shell_plus')"]),
                ("call_command(command_name='runscript')\n", ["call_command('runscript')"]),
                (
                    'from django.core.management import execute_from_command_line\n',
                    ['execute_from_command_line'],
                ),
                ('django.core.management.ManagementUtility(argv).execute()\n', ['ManagementUtility']),
            ],
        )

    def test_attaching_another_database_through_raw_sql_is_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ('cursor.execute("ATTACH DATABASE \'x.sqlite3\' AS dev")\n', ['ATTACH']),
                ("cursor.execute('attach ? as dev', [path])\n", ['ATTACH']),
                ('cursor.execute("ATTACH \'x.sqlite3\' AS dev")\n', ['ATTACH']),
            ],
        )

    def test_allowed_look_alikes_are_not_a_problem(self):
        sources = [
            "import os\nprint(os.getenv('HOME'))\n",
            "import re\npattern = re.compile('x')\n",
            "cursor.execute('SELECT 1')\n",
            "call_command('load_telescope_runs', 'f.txt')\n",
            "frame.eval('a + b')\n",
            "print('attach the observation to its run')\n",
            'connection_count = 3\n',
            'system_link = None\n',
            'from django.db import transaction\n',
            '%%time\nretrieval = execute_query()\n',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(isolation_problems(_notebook(extra_cells=[_code(source)])), [])

    def test_r12_name_lists_are_disjoint_and_reach_r0(self):
        sets = [
            REROUTE_NAMES,
            PROCESS_MODULES,
            PROCESS_NAMES,
            ENVIRON_NAMES,
            DYNAMIC_CODE_MODULES,
            DYNAMIC_CODE_BUILTINS,
            DYNAMIC_CODE_NAMES,
        ]
        for first, second in itertools.combinations(sets, 2):
            self.assertEqual(first & second, frozenset())
        self.assertEqual(R12_ALL_NAMES, frozenset().union(*sets))
        for name in sorted(R12_ALL_NAMES):
            with self.subTest(name=name):
                _first_problem_mentions(
                    self, isolation_problems(_notebook(extra_cells=[_code(f'%%time\nx = {name}\n')])), 'not parseable'
                )


class TestR12MatchesEveryIdentifierAndString(SimpleTestCase):
    """CR-01 of the third review and developer decision D-G5 (1).

    Every name R12 lists is reported at every identifier the parser records and in every part of a string or bytes
    constant and in code inside a string, so importing it from another module or binding it under another name is
    reported. Every case is the reference notebook plus one extra cell.
    """

    def test_a_listed_name_is_reported_wherever_a_name_appears(self):
        _assert_r12_reports(
            self,
            [
                ('x = subprocess\n', ['subprocess']),
                ('x = runpy\n', ['runpy']),
                ('from webbrowser import subprocess\n', ['subprocess']),
                ('from webbrowser import importlib\n', ['importlib']),
                ('import json as subprocess\n', ['subprocess']),
                ('import pkg.subprocess.thing\n', ['subprocess']),
                ('from pkg.subprocess import thing\n', ['subprocess']),
                ('f(subprocess=1)\n', ['subprocess']),
                ('f(environ=1)\n', ['environ']),
                ('def f(subprocess):\n    pass\n', ['subprocess']),
                ('def importlib():\n    pass\n', ['importlib']),
                ('class Popen:\n    pass\n', ['Popen']),
                ('try:\n    pass\nexcept OSError as environ:\n    pass\n', ['environ']),
                ('def f():\n    global environ\n', ['environ']),
                ('match webbrowser:\n    case object(subprocess=sp):\n        pass\n', ['subprocess']),
                ('match x:\n    case [*environ]:\n        pass\n', ['environ']),
                ('match x:\n    case {**environ}:\n        pass\n', ['environ']),
                ('match x:\n    case _ as subprocess:\n        pass\n', ['subprocess']),
                ('f = exec\n', ['exec']),
            ],
        )

    def test_a_listed_name_inside_a_string_or_bytes_constant_is_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ("import_string('os.system')\n", ['import_string', 'system']),
                ("pydoc.locate('os.system')\n", ['locate', 'system']),
                ("operator.attrgetter('environ.clear')(os)()\n", ['environ']),
                ("pkgutil.resolve_name('subprocess:run')\n", ['resolve_name', 'subprocess']),
                ("sys.modules[b'subprocess'.decode()].run(a)\n", ['subprocess']),
                ("x = 'django.db:connection'\n", ['connection']),
                ("m = Macro('!ls')\n", ['Macro']),
            ],
        )

    def test_code_inside_a_string_is_checked_like_code(self):
        _assert_r12_reports(
            self,
            [
                ("timeit.timeit('from webbrowser import subprocess as sp; sp.run(cmd)')\n", ['subprocess']),
                ("code.InteractiveInterpreter().runsource('import os; os.system(c)')\n", ['system']),
                ("cProfile.run('x = __import__(m)')\n", ['__import__']),
                ("timeit.timeit(stmt='pass', setup='import multiprocessing')\n", ['multiprocessing']),
                # Two levels: a string holding a string holding code.
                ('outer = "inner = \'import subprocess\'"\n', ['subprocess']),
            ],
        )

    def test_the_connection_proxy_and_attach_with_a_parenthesis_are_a_problem(self):
        _assert_r12_reports(
            self,
            [
                ('from django.db import connection\n', ['connection']),
                ('django.db.connection.close()\n', ['connection']),
                ('c = connection\n', ['connection']),
                ("import_string('django.db.connection')\n", ['connection']),
                ('dev = type(connection.connection)(p)\n', ['connection']),
                ('from django.db.backends.sqlite3.base import Database\n', ['Database', 'sqlite3']),
                ("connection._connections.settings['default']['NAME'] = x\n", ['connection', '_connections']),
                ('cursor.execute("ATTACH(\'/x\') AS dev")\n', ['ATTACH']),
                ("cursor.execute('attach  (?) as dev', [p])\n", ['ATTACH']),
                ('cursor.execute("ATTACH DATABASE \'x.sqlite3\' AS dev")\n', ['ATTACH']),
                ("cursor.execute('attach ? as dev', [path])\n", ['ATTACH']),
                ('cursor.execute("ATTACH \'x.sqlite3\' AS dev")\n', ['ATTACH']),
            ],
        )

    def test_names_that_only_look_like_listed_ones_are_not_a_problem(self):
        sources = [
            'subprocesses = 3\n',
            'connection_count = 3\n',
            'system_link = None\n',
            'from django.db import transaction\n',
            "import re\npattern = re.compile('x')\n",
            "frame.eval('a + b')\n",
            'frame.copy()\n',
            "print('subprocess-free')\n",
            "cursor.execute('SELECT 1')\n",
            "call_command('load_telescope_runs', 'f.txt')\n",
            "print('attach the observation to its run')\n",
            "print('a copy of the run')\n",
            "print('move on to the next cell')\n",
            'df = frame.copy()\n',
            "items.move_to_end('k')\n",
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(isolation_problems(_notebook(extra_cells=[_code(source)])), [])


class TestR6CopyAndWriteFunctions(SimpleTestCase):
    """WR-01 of the third review with D-G6 (b), WR-01 of the fourth review with D-G7 (2), CR-01 and CR-02 of the fifth.

    The scratch file has the developer database's base name, so any copy or byte write of a whole directory or file
    can land on ``src/fomo_db.sqlite3`` without naming it. Every copy and write function is reported as a call, a
    reference, an import and a name looked up by string. The distinctive copy functions are reported wherever they
    are named, and ``copy`` and ``move`` wherever they are read from shutil or from a name the notebook binds to it.
    With developer decision D-G11 (CR-01 and CR-02 of the fifth review) the shutil module's names (shutil, tempfile's
    _shutil) and write_bytes are reported wherever they are named too, as the distinctive copy functions are, so
    shutil handed to a parameter, a lambda or a match pattern is reported; ``copy`` and ``move`` stay tied to shutil
    names, and the look-alike ``import shutil`` then ``shutil.which('ls')`` was superseded on purpose (it names
    shutil, so it is reported). Every case is the reference notebook plus one extra cell, or a copy of a real
    notebook.
    """

    def test_every_copy_and_write_function_in_any_spelling_is_a_problem(self):
        cases = [
            ('import shutil\nshutil.copytree(a, b)\n', 'copytree'),
            ('import shutil\nshutil.copyfileobj(f, g)\n', 'copyfileobj'),
            ('import shutil\nshutil.move(a, b)\n', 'move'),
            ('import shutil as sh\nsh.move(a, b)\n', 'move'),
            ('from shutil import copytree\n', 'copytree'),
            ('import shutil\ncp = shutil.copytree\n', 'copytree'),
            ('(dst / f.name).write_bytes(data)\n', 'write_bytes'),
            ('w = p.write_bytes\n', 'write_bytes'),
            ("getattr(shutil, 'copytree')(a, b)\n", 'copytree'),
            ("getattr(shutil, 'move')(a, b)\n", 'move'),
            ("operator.methodcaller('write_bytes', data)(p)\n", 'write_bytes'),
            ("timeit.timeit('import shutil; shutil.copytree(a, b)')\n", 'copytree'),
        ]
        for source, name in cases:
            with self.subTest(source=source):
                _first_problem_mentions(self, isolation_problems(_notebook(extra_cells=[_code(source)])), name)

    def assert_r6_names(self, problems, name):
        """Assert some problem names ``name`` as the function it calls or references, before its first ';'."""
        pattern = re.compile(r'(?:calls|references) \S*\b' + re.escape(name) + r'(?:\(\))?$')
        self.assertTrue(problems, f'expected R6 to report {name}')
        self.assertTrue(
            any(pattern.search(problem.split(';')[0]) and 'byte-copy' in problem for problem in problems),
            (name, problems),
        )

    def test_a_rebound_shutil_or_a_distinctive_copy_name_is_a_problem(self):
        destination = "(scratch_db_dir, repo_root_path / 'src', dirs_exist_ok=True)\n"
        on_a_real_notebook = [
            ('review: X = shutil', 'import shutil\nsh = shutil\nsh.copytree' + destination, 'copytree'),
            ('review: from a module', 'from zipfile import shutil as s\ns.copytree' + destination, 'copytree'),
            (
                'X = shutil, copy',
                'import shutil\nsh = shutil\nfor f in scratch_db_dir.iterdir():\n'
                "    sh.copy(f, repo_root_path / 'src')\n",
                'copy',
            ),
            (
                'from a module, move',
                "from zipfile import shutil as s\ns.move(scratch_db_dir / 'x', repo_root_path / 'src')\n",
                'move',
            ),
        ]
        notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')
        for label, source, name in on_a_real_notebook:
            with self.subTest(label):
                self.assert_r6_names(isolation_problems(_append_to_setup_cell(notebook, source)), name)
        cases = [
            ('a = b = shutil\nb.copy(x, y)\n', 'copy'),
            ('sh: object = shutil\nsh.copy(a, b)\n', 'copy'),
            ('if (sh := shutil):\n    sh.copy(a, b)\n', 'copy'),
            ('sh, other = shutil, None\nsh.copy(a, b)\n', 'copy'),
            ('a = shutil\nb = a\nb.copy(x, y)\n', 'copy'),
            ('for sh in [shutil]:\n    sh.copy(a, b)\n', 'copy'),
            ('with contextlib.nullcontext(shutil) as sh:\n    sh.copy(a, b)\n', 'copy'),
            ('def f(sh=shutil):\n    sh.copy(a, b)\n', 'copy'),
            ('import zipfile\nzipfile.shutil.copy(a, b)\n', 'copy'),
            ("import sys\nsys.modules['shutil'].copy(a, b)\n", 'copy'),
            ('mods = [shutil]\nmods[0].move(a, b)\n', 'move'),
            ('class K:\n    sh = shutil\nK.sh.copy(a, b)\n', 'copy'),
            ("globals()['sh'] = shutil\nsh.copy(a, b)\n", 'copy'),
            ('x = frame.copytree\n', 'copytree'),
            ('def copyfile(a, b):\n    pass\n', 'copyfile'),
            ('f(copy2=1)\n', 'copy2'),
            ('from somewhere import copyfileobj\n', 'copyfileobj'),
            ('import pkg.copytree\n', 'copytree'),
        ]
        for source, name in cases:
            with self.subTest(source=source):
                self.assert_r6_names(isolation_problems(_notebook(extra_cells=[_code(source)])), name)

    def test_a_shutil_name_bound_in_another_cell_or_in_string_code_is_followed(self):
        cases = [
            (['import shutil as sh\n', 'sh.copy(a, b)\n'], 'copy'),
            (['sh = shutil\n', 'sh.move(a, b)\n'], 'move'),
            (["import shutil as sh\ntimeit.timeit('sh.copy(a, b)', globals=globals())\n"], 'copy'),
        ]
        for sources, name in cases:
            with self.subTest(sources=sources):
                problems = isolation_problems(_notebook(extra_cells=[_code(source) for source in sources]))
                self.assert_r6_names(problems, name)

    def test_copy_and_move_not_read_from_shutil_are_not_a_problem(self):
        sources = [
            'df = frame.copy()\n',
            'sh = frame\nsh.copy()\n',
            'import copy\nc = copy.copy(x)\n',
            "print('a copy of the run')\n",
            "items.move_to_end('k')\n",
            "print('copytree is not used here')\n",
            'x = shutil_free\nx.copy()\n',
            "print('shutil-free')\n",
            'x = my_shutil\n',
            'x = write_bytes_count\n',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(isolation_problems(_notebook(extra_cells=[_code(source)])), [])

    def test_shutil_passed_on_in_any_way_is_reported_on_a_real_notebook(self):
        # CR-01 of the fifth review, D-G11: shutil handed on without a binding R6 follows is reported by its name.
        loop = '    for f in scratch_db_dir.iterdir():\n'
        cases = {
            'parameter given shutil': (
                f"import shutil\ndef put(sh):\n{loop}        sh.copy(f, repo_root_path / 'src')\nput(shutil)\n"
            ),
            'lambda called at once': (
                'import shutil\n'
                "(lambda sh: [sh.copy(f, repo_root_path / 'src') for f in scratch_db_dir.iterdir()])(shutil)\n"
            ),
            'map over a lambda': (
                'import shutil\n'
                "list(map(lambda sh: [sh.move(f, repo_root_path / 'src') for f in scratch_db_dir.iterdir()], "
                '[shutil]))\n'
            ),
            'keyword-only parameter': (
                f"import shutil\ndef put(*, sh):\n{loop}        sh.move(f, repo_root_path / 'src')\nput(sh=shutil)\n"
            ),
            'match capture': (
                'import shutil\nmatch shutil:\n    case sh:\n        for f in scratch_db_dir.iterdir():\n'
                "            sh.copy(f, repo_root_path / 'src')\n"
            ),
            'class pattern': (
                'import shutil\nmatch shutil:\n    case object(copy=c):\n        for f in scratch_db_dir.iterdir():\n'
                "            c(f, repo_root_path / 'src')\n"
            ),
        }
        notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')
        for label, source in cases.items():
            with self.subTest(label):
                self.assert_r6_names(isolation_problems(_append_to_setup_cell(notebook, source)), 'shutil')
        with self.subTest('control: parameter default'):
            source = f"import shutil\ndef put(sh=shutil):\n{loop}        sh.move(f, repo_root_path / 'src')\nput()\n"
            problems = isolation_problems(_append_to_setup_cell(notebook, source))
            self.assert_r6_names(problems, 'move')
            self.assert_r6_names(problems, 'shutil')

    def test_write_bytes_at_any_identifier_is_reported_on_a_real_notebook(self):
        # CR-02 of the fifth review, D-G11: write_bytes is matched at every identifier, once per cell.
        cases = {
            'class pattern': (
                "for f in scratch_db_dir.iterdir():\n    match repo_root_path / 'src' / f.name:\n"
                '        case object(write_bytes=w):\n            w(f.read_bytes())\n'
            ),
            'parameter': 'def put(p, write_bytes=None):\n    pass\n',
            'keyword argument': 'put(p, write_bytes=data)\n',
            'control: method': (
                'for f in scratch_db_dir.iterdir():\n'
                "    (repo_root_path / 'src' / f.name).write_bytes(f.read_bytes())\n"
            ),
        }
        notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')
        for label, source in cases.items():
            with self.subTest(label):
                problems = isolation_problems(_append_to_setup_cell(notebook, source))
                self.assert_r6_names(problems, 'write_bytes')
                self.assertEqual(sum('write_bytes' in problem for problem in problems), 1, problems)

    def test_shutil_and_tempfile_shutil_are_reported_wherever_they_are_named(self):
        # D-G11: the module's names are reported like the distinctive copy functions; shutil.which is superseded.
        named = [
            ('import shutil\n', 'shutil'),
            ("import shutil\npath = shutil.which('ls')\n", 'shutil'),
            ('from shutil import which\n', 'shutil'),
            ('def f(shutil=None):\n    pass\n', 'shutil'),
            ('f(shutil=1)\n', 'shutil'),
            ('match m:\n    case object(shutil=s):\n        pass\n', 'shutil'),
        ]
        for source, name in named:
            with self.subTest(source=source):
                self.assert_r6_names(isolation_problems(_notebook(extra_cells=[_code(source)])), name)
        with self.subTest('tempfile._shutil'):
            problems = isolation_problems(
                _notebook(extra_cells=[_code('import tempfile\ntempfile._shutil.copy(a, b)\n')])
            )
            self.assert_r6_names(problems, '_shutil')
            self.assert_r6_names(problems, 'copy')
        with self.subTest('a string'):
            _first_problem_mentions(
                self, isolation_problems(_notebook(extra_cells=[_code("x = 'shutil'\n")])), 'shutil'
            )
        with self.subTest('a dotted string'):
            problems = isolation_problems(_notebook(extra_cells=[_code("x = 'tempfile._shutil'\n")]))
            _first_problem_mentions(self, problems, '_shutil')
        with self.subTest('one problem per cell'):
            problems = isolation_problems(_notebook(extra_cells=[_code('import shutil\n'), _code('x = shutil\n')]))
            numbers = [problem.split()[2] for problem in problems if 'references shutil' in problem]
            self.assertEqual(len(numbers), 2, problems)
            self.assertEqual(len(set(numbers)), 2, problems)
        with self.subTest('the start set'):
            self.assertEqual(_shutil_names([]), {'shutil', '_shutil'})


class TestR0AllowList(SimpleTestCase):
    """CR-02 of the third review (D-G5 2) and CR-01 of the fourth (D-G7 1); WR-01, WR-02 of the fifth review.

    The fifth review's findings are settled by developer decisions D-G12 (WR-01) and D-G13 (WR-02).

    IPython runs more than the escapes R0 listed -- aliases, automagic, escapes after a line magic, prompt prefixes
    -- so every cell must be Python that IPython runs unchanged, apart from a few harmless magic lines and IPython's
    autocall, which R0 cannot see (the IPyAutocall name is reported by R12, and the kernel's only autocall objects
    are exit and quit). A one-line cell may not start with an IPython magic or alias name whatever an earlier cell
    binds (only a builtin, a keyword or a word followed by ``=`` or ``,`` is exempt), because IPython decides from
    the names bound when the cell runs. Whether R2-R4 and R8-R11 apply is decided from each cell as written (D-G12).
    Every case is the reference notebook plus the extra cells shown, or a copy of a real notebook.
    """

    def problems_with(self, *sources):
        return isolation_problems(_notebook(extra_cells=[_code(source) for source in sources]))

    def test_a_cell_is_a_problem_unless_it_is_python_that_ipython_runs_unchanged(self):
        not_parseable = [
            '!ls\n',
            '%ls\n',
            '%time !ls\n',
            '%%time\n!ls\n',
            '%%capture\n%cd ..\n',
            '%%writefile ../../src/x\nx\n',
            'ls ?\n',
            '%matplotlib qt\n',
            # A cell magic with a leading space, or not on the first line, is not one IPython strips.
            ' %%time\nx = 1\n',
            'x = 1\n%%time\ny = 1\n',
            # %%time takes no argument.
            '%%time --flag\nx = 1\n',
        ]
        for source in not_parseable:
            with self.subTest(source=source):
                _first_problem_mentions(self, self.problems_with(source), 'not parseable')
        with self.subTest(source='In [1]: x = 1'):
            _first_problem_mentions(self, self.problems_with('In [1]: x = 1\n'), R0_REWRITE_MARKER)
        allowed = [
            '%matplotlib inline\n',
            '%%time\nx = 1\n',
            '%%capture --no-stderr out\nx = 1\n',
            '%%time\nretrieval = execute_query()\n',
            '',
        ]
        for source in allowed:
            with self.subTest(source=source):
                self.assertEqual(self.problems_with(source), [])

    def test_code_under_a_harmless_magic_is_checked_by_every_rule(self):
        cases = [
            ('%%capture\nfrom shutil import copytree\ncopytree(a, b)\n', 'copytree'),
            ('%%time\nimport sqlite3\n', 'references sqlite3;'),
            ("%%capture\ndb = open('src/fomo_db.sqlite3')\n", DEVELOPER_DB_NAME),
        ]
        for source, text in cases:
            with self.subTest(source=source):
                _first_problem_mentions(self, self.problems_with(source), text)

    def test_autocall_is_not_seen_by_r0_and_the_ipyautocall_name_is_reported(self):
        # IPython's IPyAutocallChecker runs a one-line cell that starts with an autocall object as a call (``exit -1``
        # as ``exit(-1)``), a rewrite IPython makes from the namespace at run time that R0 cannot see. The kernel's
        # only autocall objects are exit and quit; the name IPyAutocall is an R12 name (D-G13, WR-02 of the fifth
        # review).
        for source in ('exit -1\n', 'quit -1\n'):
            with self.subTest(source=source):
                self.assertEqual(self.problems_with(source), [])
        bye = 'class Bye(type(exit)):\n    def __call__(self, *args):\n        {body}\nbye = Bye()\n'
        with self.subTest('an autocall object made without the name'):
            self.assertEqual(self.problems_with(bye.format(body='pass'), 'bye -1\n'), [])
        with self.subTest('its call method is notebook code'):
            problems = self.problems_with(bye.format(body='shutil.copy(a, b)'), 'bye -1\n')
            self.assertTrue(
                any(p.split(';')[0].endswith('calls shutil.copy()') and 'byte-copy' in p for p in problems), problems
            )
            self.assertFalse(any(R0_REWRITE_MARKER in p or R0_AUTOMAGIC_MARKER in p for p in problems), problems)
        forms = {
            'import': 'from IPython.core.autocall import IPyAutocall\n',
            'subclass': (
                'import IPython.core.autocall as autocall\n'
                'class Bye(autocall.IPyAutocall):\n    def __call__(self):\n        pass\n'
            ),
            'string': "name = 'IPython.core.autocall.IPyAutocall'\n",
        }
        notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')
        for label, source in forms.items():
            with self.subTest(label):
                problems = isolation_problems(_append_to_setup_cell(notebook, source))
                self.assertTrue(problems, 'the guard missed an IPyAutocall name')
                self.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)
                self.assertEqual(_rerouting_names(problems), {'IPyAutocall'}, problems)

    def test_the_django_backed_gate_reads_each_cell_as_written(self):
        # Pins the documented fact behind D-G12: whether R2-R4 and R8-R11 apply is decided from each cell as written
        # (D-13, unchanged); every real notebook passes the gate (test_the_real_notebook_check_is_not_vacuous).
        magic_cell = '%%capture\nx = 1; import django as dj\ndj.setup()\n'
        with self.subTest('under a magic the gate does not see the import'):
            bare = {'cells': [_code(magic_cell)]}
            problems = isolation_problems(bare)
            self.assertFalse(is_django_backed(bare))
            self.assertFalse(
                any('imports django under an alias' in p or 'no top-level' in p for p in problems), problems
            )
        with self.subTest('without the magic the same code is Django-backed'):
            plain = {'cells': [_code(magic_cell.split('\n', 1)[1])]}
            problems = isolation_problems(plain)
            self.assertTrue(is_django_backed(plain))
            _first_problem_mentions(self, problems, 'imports django under an alias')
            _first_problem_mentions(self, problems, 'no top-level')
        with self.subTest('on a copy of a real notebook'):
            notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')
            inserted = _insert_code_cell_after_setup(notebook, magic_cell)
            _first_problem_mentions(self, isolation_problems(inserted), 'imports django under an alias')

    def test_a_one_line_magic_or_alias_is_a_problem_even_when_an_earlier_cell_binds_the_name(self):
        # IPython decides from the names bound when the cell runs (is_shadowed), and a bare annotation, an
        # `if False:` branch, an empty loop or an `except ... as` name binds nothing at runtime, so the guard cannot
        # trust any binding it reads in the source (D-G7 1, CR-01 of the fourth review).
        reported = [
            "rm -r -- 'x'\n",
            'cp\n',
            'rehashx\n',
            'time -x\n',
            'rm += 3\n',
            'man -k\n',  # an ipykernel magic
            'store -r\n',  # storemagic, a default extension
            'ed\n',  # an alias IPython registers at start-up
            "rm -r -- 'x'  # trailing comment\n",
            "rm -r -- 'x'\n\n",  # blank lines do not count
        ]
        for source in reported:
            with self.subTest(source=source):
                problems = self.problems_with(source)
                self.assertTrue(problems, 'the guard missed a one-line magic or alias')
                self.assertTrue(all(R0_AUTOMAGIC_MARKER in problem for problem in problems), problems)
        with self.subTest('time bound by an earlier import'):
            problems = self.problems_with('import time\n', 'time\n')
            self.assertTrue(problems, 'an earlier import must not exempt a one-line cell')
            self.assertTrue(all(R0_AUTOMAGIC_MARKER in problem for problem in problems), problems)
            self.assertTrue(all('bound when the cell runs' in problem for problem in problems), problems)
        allowed = {
            'time = 5': ['time = 5\n'],
            'rm = 3': ['rm = 3\n'],
            'rm, cp = 1, 2': ['rm, cp = 1, 2\n'],
            'run_report()': ['run_report()\n'],
            'print(1)': ['print(1)\n'],
            'two lines': ["rm -r -- 'x'\ny = 1\n"],
        }
        for label, sources in allowed.items():
            with self.subTest(label):
                self.assertEqual(self.problems_with(*sources), [])
        with self.subTest('a dotted first word is never a magic'):
            problems = self.problems_with('os.system\n')
            self.assertTrue(problems)
            self.assertTrue(all(REROUTE_MARKER in problem for problem in problems), problems)
            self.assertFalse(any(R0_AUTOMAGIC_MARKER in problem for problem in problems), problems)
        notebook = _load_notebook(NOTEBOOK_DIR / 'import_campaign_csv_demo.ipynb')
        bindings = {
            'bare annotation': 'rm: int\n',
            'if False branch': 'if False:\n    rm = 1\n',
            'empty for loop': 'for rm in []:\n    pass\n',
            'except as': 'try:\n    pass\nexcept OSError as rm:\n    pass\n',
            'plain assignment': 'rm = 1\n',
            'import': 'from os import remove as rm\n',
            'assignment then del': 'rm = 1\ndel rm\n',
        }
        for label, binding in bindings.items():
            with self.subTest(label):
                bound = _append_to_setup_cell(notebook, binding)
                self.assertEqual(isolation_problems(bound), [], 'the binding alone must be clean')
                problems = isolation_problems(_insert_code_cell_after_setup(bound, "rm -r -- '../../src'\n"))
                self.assertTrue(problems, 'the alias cell after the binding was not reported')
                self.assertTrue(all(R0_AUTOMAGIC_MARKER in problem for problem in problems), problems)

    def test_the_automagic_rule_knows_every_line_magic_a_fresh_ipython_shell_has(self):
        child = (
            'import json\n'
            'from traitlets.config import Config\n'
            'from IPython.core.interactiveshell import InteractiveShell\n'
            'config = Config()\n'
            'config.HistoryManager.enabled = False\n'
            'shell = InteractiveShell.instance(config=config)\n'
            "shell.extension_manager.load_extension('storemagic')\n"
            "print(json.dumps(sorted(shell.magics_manager.magics['line'])))\n"
        )
        with tempfile.TemporaryDirectory() as ipython_dir:
            result = subprocess.run(
                [sys.executable, '-I', '-c', child],
                cwd=ipython_dir,
                env={**os.environ, 'IPYTHONDIR': ipython_dir},
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        shell_names = set(json.loads(result.stdout.strip().splitlines()[-1]))
        known = _ipython_line_magic_names()
        self.assertEqual(shell_names - known, set())
        expected = {'cp', 'rm', 'mv', 'rehashx', 'pip', 'alias', 'sx', 'system', 'store', 'reset', 'run', 'time'}
        expected |= {'env', 'ed', 'hist', 'rep', 'man', 'less'}
        self.assertTrue(expected <= known, expected - known)


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
