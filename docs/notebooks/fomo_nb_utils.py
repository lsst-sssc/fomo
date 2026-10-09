import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord


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
    """Route Django to a scratch database, set it up, check the route, migrate and print the evidence.

    Call this once, as the first Django-related statement of a notebook. ``repo_root`` must already
    be on ``sys.path`` so that ``src.fomo.settings`` imports.

    Args:
        repo_root: path of the repository (the directory holding manage.py).
        mode: ``'fresh'`` migrates an empty database; ``'snapshot'`` takes a read-only online-backup
            copy of ``src/fomo_db.sqlite3`` and migrates it to head.

    Returns:
        pathlib.Path: the scratch directory, to hand to ``teardown_scratch_django()``.

    Raises:
        RuntimeError: Django is already set up in this interpreter (restart the kernel), snapshot
            mode found no developer database, or Django resolved a database other than the scratch
            one. The scratch directory is removed before the error propagates.
        ValueError: ``mode`` is not one of ``SCRATCH_MODES``.
    """
    from django.apps import apps

    if apps.ready:
        raise RuntimeError(
            'Django is already set up in this interpreter, so the database cannot be re-routed; '
            'restart the kernel (Restart & Run All) instead of re-running the setup cell.'
        )
    scratch_db_path = prepare_scratch_database(repo_root, mode)
    try:
        # Plain assignment, never setdefault, so an inherited value can never point a notebook at a
        # real database. The value is non-empty: settings.py treats an empty one as "unset" and
        # falls back to the developer database.
        os.environ['FOMO_DATABASE_PATH'] = str(scratch_db_path)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.fomo.settings')
        # Jupyter's ipykernel runs inside an asyncio event loop, but Django's ORM is sync-only by
        # default and refuses to run there; this opts back in.
        os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

        import django

        django.setup()

        from django.conf import settings as django_settings
        from django.core.management import call_command

        resolved_name = django_settings.DATABASES['default']['NAME']
        check_resolved_database(resolved_name, scratch_db_path)
        # A snapshot can be behind head, and a fresh database has no tables yet.
        call_command('migrate', verbosity=0)
    except BaseException:
        shutil.rmtree(scratch_db_path.parent, ignore_errors=True)
        raise
    print(f'Django ready: settings module={os.environ["DJANGO_SETTINGS_MODULE"]!r}, repo_root={str(repo_root)!r}')
    print(f'Resolved database: {resolved_name!r} ({_MODE_DESCRIPTIONS[mode]})')
    return scratch_db_path.parent


def teardown_scratch_django(scratch_dir):
    """Close Django's database connections and remove the scratch directory.

    Args:
        scratch_dir: the directory ``setup_scratch_django()`` returned.

    Raises:
        RuntimeError: ``scratch_dir`` is not an existing directory named ``fomo-notebook-db-*``;
            nothing is closed or removed in that case.
        OSError: the directory could not be removed (errors are not swallowed).
    """
    path = Path(scratch_dir)
    if not path.name.startswith(SCRATCH_DIR_PREFIX) or not path.is_dir():
        raise RuntimeError(f'Refusing to remove {str(path)!r}: not an existing directory named {SCRATCH_DIR_PREFIX}*.')
    from django.apps import apps

    if apps.ready:
        from django.db import connections

        connections.close_all()
    shutil.rmtree(path)
    print(f'{TEARDOWN_LINE_PREFIX} {path}')
    print('The developer database (src/fomo_db.sqlite3) was never opened for writing.')
