"""Gunicorn configuration for running FOMO as a long-lived service.

Used by the systemd unit that ``deploy/install_service.sh`` generates. To run it by hand from the repo root
(with the venv active): ``FOMO_BIND=tlister-thinkmate.lco.gtn:6729 gunicorn -c deploy/gunicorn.conf.py``
"""

import os
from pathlib import Path

# gunicorn chdirs here and puts it on sys.path, which makes `solsys_code` importable; `src/` (for `fomo.*`
# and `local_settings`) is on sys.path via the editable install's .pth file.
chdir = str(Path(__file__).resolve().parent.parent)
wsgi_app = 'fomo.wsgi:application'

bind = os.environ.get('FOMO_BIND', '127.0.0.1:8000').split(',')
proc_name = 'fomo'
# systemd manages the process; the gunicorn >=26 control socket defaults to a per-user path shared by every
# gunicorn instance, so turn it off.
control_socket_disable = True

# Each worker holds its own copy of the SPICE kernels/ASSIST ephemeris (~450 MB RSS). Threads give
# concurrency within a worker; keep the process count low because the database is SQLite.
workers = int(os.environ.get('FOMO_WORKERS', 3))
worker_class = 'gthread'
threads = int(os.environ.get('FOMO_THREADS', 4))

# Do NOT enable preload_app: importing solsys_code.ephem_utils opens the SPICE kernels and ASSIST ephemeris
# files, and forked workers would then share those file descriptors (and their seek offsets).
preload_app = False

# Let an in-flight ephemeris calculation finish on restart/reload.
graceful_timeout = 120

# Recycle workers periodically to bound any slow memory growth.
max_requests = 1000
max_requests_jitter = 100

# Log to stdout/stderr, i.e. the systemd journal (`journalctl --user -u fomo`).
accesslog = '-'
errorlog = '-'
loglevel = 'info'


def post_worker_init(worker):
    """Load the URLconf (and with it the ephemeris machinery) before the worker accepts its first request."""
    from importlib import import_module

    from django.conf import settings

    import_module(settings.ROOT_URLCONF)
