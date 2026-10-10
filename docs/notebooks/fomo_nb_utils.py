import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

logger = logging.getLogger(__name__)


def special_locations(skycoords=False):
    """Special locations for Rubin. Adapted and condensed into one routine from
    the originals in `rubin_scheduler.utils`

    Parameters
    ----------
    skycoords : `bool`
        Return locations as astropy.SkyCoords. If False, returns
        as a tuple of floats. Default False.
    """

    # The DDF locations here are from Neil Brandt's white paper
    # submitted in response to the 2018 Call for White Papers on observing
    # strategy
    # Document-30468 -- AGN-DDF-WP-02.pdf
    # The locations are chosen based on existing multi-wavelength
    # coverage, plus an offset to avoid the bright star Mira near XMM-LSS

    result = {}
    result['ELAISS1'] = SkyCoord('00:37:48 −44:01:30', unit=(u.hourangle, u.deg), frame='icrs')
    result['XMM_LSS'] = SkyCoord('02:22:18  −04:49:00', unit=(u.hourangle, u.deg), frame='icrs')
    result['ECDFS'] = SkyCoord('03:31:55  −28:07:00', unit=(u.hourangle, u.deg), frame='icrs')
    result['COSMOS'] = SkyCoord('10:00:26  +02:14:01', unit=(u.hourangle, u.deg), frame='icrs')
    result['EDFS_a'] = SkyCoord(ra=58.90 * u.deg, dec=-49.32 * u.deg, frame='icrs')
    result['EDFS_b'] = SkyCoord(ra=63.60 * u.deg, dec=-47.60 * u.deg, frame='icrs')
    result['Roman_bulge_location'] = SkyCoord(268.708 * u.deg, -28.975 * u.deg, frame='icrs')
    result['M49 (Virgo)'] = SkyCoord(187.4 * u.deg, +8 * u.deg, frame='icrs')

    if not skycoords:
        # replace SkyCoord with ra/deg tuple in degrees
        result = dict([(key, (result[key].ra.deg, result[key].dec.deg)) for key in result])

    return result


# --- Scratch-database setup shared by every pre-executed notebook (Phase 40, WARN-05 / WARN-03) ---
#
# A notebook under docs/notebooks/pre_executed/ may create, update and delete rows (the runner
# demos do), so it must never run against the developer database. These functions own every step
# of that setup: the scratch directory, FOMO_DATABASE_PATH, a fresh migrate or a read-only online
# backup of the developer database, django.setup(), the resolved-path check, the evidence line and
# the teardown. Django is imported only inside function bodies so that importing this module (and
# special_locations() above) never needs a configured Django.

SCRATCH_DIR_PREFIX = 'fomo-notebook-db-'
SCRATCH_DB_NAME = 'fomo_db.sqlite3'
SCRATCH_MODES = ('fresh', 'snapshot')
TEARDOWN_LINE_PREFIX = 'Removed scratch database directory:'

# The environment variables setup_scratch_django() changes; a failed setup and the teardown put them back.
_ROUTING_ENVIRON = ('FOMO_DATABASE_PATH', 'DJANGO_ALLOW_ASYNC_UNSAFE')

# The scratch directory and the previous environment values of the one setup this interpreter allows,
# read by teardown_scratch_django(). setup_scratch_django() refuses a second setup per interpreter, so this
# record cannot mix two setups. It is a deliberate exception to the project note about module-level mutable
# state: returning the values instead would change the `scratch_db_dir = setup_scratch_django(...)` /
# `teardown_scratch_django(scratch_db_dir)` call shape that every notebook and the guard depend on (D-06).
_active_setup = {}

_MODE_DESCRIPTIONS = {
    'fresh': 'fresh scratch database, migrated from empty',
    'snapshot': (
        "scratch copy of the developer database, taken read-only with SQLite's online backup and migrated to head"
    ),
}


def developer_database_path(repo_root):
    """Return the path of the developer database under a repository root.

    Args:
        repo_root: path of the repository (the directory holding manage.py).

    Returns:
        pathlib.Path: ``<repo_root>/src/fomo_db.sqlite3``. The file may not exist.
    """
    return Path(repo_root) / 'src' / SCRATCH_DB_NAME


def snapshot_database(source_path, target_path):
    """Copy a SQLite database with the online backup API, opening the source read-only.

    The source is opened through a ``?mode=ro`` URI, so this function cannot write it even by
    accident, and ``Connection.backup()`` takes a consistent snapshot even while another process
    (the unattended runner) is writing the source, which a byte copy cannot promise.

    Args:
        source_path: path of the SQLite file to read.
        target_path: path of the SQLite file to create or overwrite with the snapshot.

    Raises:
        sqlite3.Error: the source cannot be opened or read.
    """
    source_connection = sqlite3.connect(f'{Path(source_path).resolve().as_uri()}?mode=ro', uri=True)
    try:
        target_connection = sqlite3.connect(target_path)
        try:
            source_connection.backup(target_connection)
        finally:
            target_connection.close()
    finally:
        source_connection.close()


def check_resolved_database(resolved_name, scratch_db_path):
    """Raise unless Django resolved exactly the scratch database file.

    Both paths are resolved before they are compared, so the same file reached through a symlink or
    another spelling is accepted, and any other file (including an empty name) is refused. This is a
    ``raise``, never an ``assert``: ``python -O`` strips assertions, and ``local_settings.py`` can
    override ``DATABASES`` regardless of ``FOMO_DATABASE_PATH``.

    Args:
        resolved_name: ``settings.DATABASES['default']['NAME']`` as Django resolved it.
        scratch_db_path: path of the scratch database this run created.

    Raises:
        RuntimeError: the resolved database is not the scratch database.
    """
    if not resolved_name or Path(resolved_name).resolve() != Path(scratch_db_path).resolve():
        raise RuntimeError(
            'A notebook must never write to the developer database -- resolved DB '
            f'{resolved_name!r} is not the scratch database {str(scratch_db_path)!r}.'
        )


def check_database_settings(databases, database_routers, scratch_db_path):
    """Raise unless Django's settings reach the scratch database file and nothing else.

    Checked in a fixed order, so the message always names the first problem: any database alias other
    than ``'default'`` (even one whose ``NAME`` is the scratch file), then any non-empty
    ``DATABASE_ROUTERS``, then the ``default`` file itself. A notebook cannot prove that every write
    reaches the scratch file once a second alias or a router exists, so both are refused rather than
    supported. These are ``raise`` statements, never ``assert``: ``python -O`` strips assertions.

    Args:
        databases: ``settings.DATABASES`` as Django would use it.
        database_routers: ``settings.DATABASE_ROUTERS`` as Django would use it.
        scratch_db_path: path of the scratch database this run created.

    Raises:
        RuntimeError: an extra alias or a router is configured, or the ``default`` database is
            missing, has no ``NAME``, or is not the scratch database.
    """
    extra_aliases = sorted(alias for alias in databases if alias != 'default')
    if extra_aliases:
        named = ', '.join(repr(alias) for alias in extra_aliases)
        raise RuntimeError(
            "A notebook scratch setup supports only the 'default' database, but the settings also define "
            f'{named}; a notebook cannot prove every write reaches the scratch database.'
        )
    if database_routers:
        raise RuntimeError(
            f'DATABASE_ROUTERS is set ({database_routers!r}); a notebook cannot prove every write reaches the '
            'scratch database once a router can send queries elsewhere.'
        )
    check_resolved_database((databases.get('default') or {}).get('NAME'), scratch_db_path)


def _restore_environ(previous):
    """Put environment variables back to the values they had before the setup.

    Args:
        previous: mapping of variable name to its earlier value, or ``None`` when the variable was
            not set; such a variable is removed, never set to an empty string.
    """
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _disable_django_databases():
    """Leave Django unable to reach any database, so a refused or half-finished setup cannot be used.

    Every configured alias is switched to Django's dummy backend first, which raises
    ``ImproperlyConfigured`` on any query. The alias dicts are changed in place because Django's
    connection handler and its cached connection wrappers hold the same dict objects as
    ``settings.DATABASES``, so the switch alone leaves no real database reachable and a close that fails
    afterwards cannot undo it. Then every connection is closed on a best-effort basis (a failure,
    including Django's own refusal when no ``default`` alias exists, is logged and never raised), and
    every cached wrapper is dropped so the next access builds a dummy one. When the setup failed before
    the settings were read, a dummy-only configuration is installed instead, so a later query still
    cannot reach a real database.
    """
    from django.conf import settings as django_settings
    from django.db import connections

    if not django_settings.configured:
        django_settings.configure(DATABASES={'default': {'ENGINE': 'django.db.backends.dummy'}})
    aliases = list(django_settings.DATABASES)
    for alias in aliases:
        django_settings.DATABASES[alias]['ENGINE'] = 'django.db.backends.dummy'
        django_settings.DATABASES[alias]['NAME'] = ''
    try:
        connections.close_all()
    except Exception as exc:
        logger.warning(
            f'Closing the database connections failed ({type(exc).__name__}: {exc}); '
            'every alias is already on the dummy backend, so no database is reachable.'
        )
    for alias in aliases:
        try:
            del connections[alias]
        except AttributeError:
            # No wrapper was cached for this alias, so there is nothing to drop.
            pass


def prepare_scratch_database(repo_root, mode='fresh'):
    """Create the scratch directory and, in snapshot mode, fill it with a read-only snapshot.

    Both refusals happen before anything is created, so a failure leaves no directory behind and
    leaves the environment untouched.

    Args:
        repo_root: path of the repository (the directory holding manage.py).
        mode: ``'fresh'`` for an empty database that setup migrates, or ``'snapshot'`` for an online
            backup of ``src/fomo_db.sqlite3``.

    Returns:
        pathlib.Path: the scratch database file inside the new ``fomo-notebook-db-*`` directory.

    Raises:
        ValueError: ``mode`` is not one of ``SCRATCH_MODES``.
        RuntimeError: snapshot mode was requested but the developer database does not exist.
    """
    if mode not in SCRATCH_MODES:
        raise ValueError(f'Unknown scratch database mode {mode!r}; expected one of {SCRATCH_MODES}.')
    developer_db = developer_database_path(repo_root)
    if mode == 'snapshot' and not developer_db.is_file():
        raise RuntimeError(f'No developer database at {developer_db}; run `python manage.py migrate` first.')
    scratch_dir = Path(tempfile.mkdtemp(prefix=SCRATCH_DIR_PREFIX))
    scratch_db_path = scratch_dir / SCRATCH_DB_NAME
    if mode == 'snapshot':
        try:
            snapshot_database(developer_db, scratch_db_path)
        except BaseException:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            raise
    return scratch_db_path


def setup_scratch_django(repo_root, mode='fresh'):
    """Route Django to a scratch database, check the route, set Django up, migrate and print the evidence.

    Call this once, as the first Django-related statement of a notebook. ``repo_root`` must already
    be on ``sys.path`` so that ``src.fomo.settings`` imports. The settings are read and checked
    before ``django.setup()`` runs, so a refused database is never made reachable.

    Args:
        repo_root: path of the repository (the directory holding manage.py).
        mode: ``'fresh'`` migrates an empty database; ``'snapshot'`` takes a read-only online-backup
            copy of ``src/fomo_db.sqlite3`` and migrates it to head.

    Returns:
        pathlib.Path: the scratch directory, to hand to ``teardown_scratch_django()``.

    Raises:
        RuntimeError: Django is already set up or configured in this interpreter (restart the
            kernel), snapshot mode found no developer database, or the settings define a database
            alias other than ``default``, a ``DATABASE_ROUTERS`` entry, or a ``default`` database
            other than the scratch one. Once the settings are being read, any failure (the refusal
            itself, or a later one such as in ``migrate``) switches every database alias to Django's
            dummy backend, then closes the connections on a best-effort basis (a failure there is
            logged, never raised in place of the original error), before the setup's own error
            propagates, so no database is reachable until the kernel is restarted. The scratch
            directory is removed and the routing environment variables are put back. A failure before
            the settings are read (for example snapshot mode with no developer database) leaves
            Django unconfigured and routes nothing, so the setup cell can simply be run again.
        ValueError: ``mode`` is not one of ``SCRATCH_MODES``.
    """
    from django.apps import apps
    from django.conf import settings as django_settings

    # Outside the try below on purpose: the disable step must only ever run on a setup this function
    # started, never on a Django that was already running (for example the test runner's).
    if apps.ready or django_settings.configured:
        raise RuntimeError(
            'Django is already set up or configured in this interpreter (a setup in this kernel already ran, '
            "or failed after reading Django's settings), so the database cannot be re-routed; restart the "
            'kernel (Restart & Run All) instead of re-running the setup cell.'
        )
    scratch_db_path = prepare_scratch_database(repo_root, mode)
    previous_environ = {name: os.environ.get(name) for name in _ROUTING_ENVIRON}
    try:
        # Plain assignment, never setdefault, so an inherited value can never point a notebook at a
        # real database. The value is non-empty: settings.py treats an empty one as "unset" and
        # falls back to the developer database.
        os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')
        # Jupyter's ipykernel runs inside an asyncio event loop, but Django's ORM is sync-only by
        # default and refuses to run there; this opts back in.
        os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

        # Reading the lazy settings loads the settings module without populating the app registry or
        # opening a connection, so the database Django would use is checked before django.setup().
        check_database_settings(django_settings.DATABASES, django_settings.DATABASE_ROUTERS, scratch_db_path)
        resolved_name = django_settings.DATABASES['default']['NAME']

        import django

        django.setup()

        from django.core.management import call_command

        # A snapshot can be behind head, and a fresh database has no tables yet.
        call_command('migrate', verbosity=0)
    except BaseException:
        # Order matters: closing connections is async_unsafe, so DJANGO_ALLOW_ASYNC_UNSAFE must still
        # be set while _disable_django_databases() runs; the environment is restored last. A clean-up
        # failure is logged and never replaces the setup's own error, which the bare raise re-raises.
        try:
            _disable_django_databases()
        except Exception as cleanup_exc:
            logger.warning(
                f'The clean-up after the failed setup also raised ({type(cleanup_exc).__name__}: {cleanup_exc}); '
                "the setup's own error is re-raised."
            )
        finally:
            shutil.rmtree(scratch_db_path.parent, ignore_errors=True)
            _restore_environ(previous_environ)
        raise
    _active_setup['scratch_dir'] = scratch_db_path.parent
    _active_setup['previous_environ'] = previous_environ
    print(f'Django ready: settings module={os.environ["DJANGO_SETTINGS_MODULE"]!r}')
    print(f'Resolved database: {resolved_name!r} ({_MODE_DESCRIPTIONS[mode]})')
    return scratch_db_path.parent


def teardown_scratch_django(scratch_dir):
    """Close Django's database connections, remove the scratch directory and restore the environment.

    Only the directory this interpreter's ``setup_scratch_django()`` created is accepted, and only when
    it sits directly inside the temporary directory; every check runs before Django is imported or
    anything is closed or removed. The second printed line reports what the setup checked before
    ``django.setup()`` ran, not a measurement taken at teardown.

    Args:
        scratch_dir: the directory ``setup_scratch_django()`` returned.

    Raises:
        RuntimeError: ``scratch_dir`` is not an existing directory named ``fomo-notebook-db-*``, is
            not directly inside the resolved temporary directory, or is not the directory a
            successful ``setup_scratch_django()`` created in this interpreter (including when no
            setup ran or the setup failed); nothing is closed or removed in that case.
        OSError: the directory could not be removed (errors are not swallowed).
    """
    path = Path(scratch_dir)
    if not path.name.startswith(SCRATCH_DIR_PREFIX) or not path.is_dir():
        raise RuntimeError(f'Refusing to remove {str(path)!r}: not an existing directory named {SCRATCH_DIR_PREFIX}*.')
    temp_root = Path(tempfile.gettempdir()).resolve()
    if path.resolve().parent != temp_root:
        raise RuntimeError(
            f'Refusing to remove {str(path)!r}: not directly inside the temporary directory {str(temp_root)!r}.'
        )
    recorded = _active_setup.get('scratch_dir')
    if recorded is None or Path(recorded).resolve() != path.resolve():
        raise RuntimeError(
            f'Refusing to remove {str(path)!r}: it is not the scratch directory setup_scratch_django() created in '
            'this interpreter.'
        )
    from django.apps import apps

    if apps.ready:
        from django.db import connections

        connections.close_all()
    shutil.rmtree(path)
    # Restored only after close_all(), which is async_unsafe and needs DJANGO_ALLOW_ASYNC_UNSAFE still set.
    _restore_environ(_active_setup['previous_environ'])
    _active_setup.clear()
    print(f'{TEARDOWN_LINE_PREFIX} {path}')
    print(
        "Before django.setup() ran, the helper checked that Django's only configured database was the scratch "
        'file in that directory.'
    )
