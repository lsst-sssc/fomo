Single-host Deployment with gunicorn and systemd
================================================

This page describes a "halfway house" between ``manage.py runserver`` and a full
containerized deployment: FOMO runs under the `gunicorn <https://gunicorn.org/>`_
WSGI server, supervised by a systemd *user* service, so it starts at boot,
restarts if it crashes, and logs to the systemd journal. Static files are served
by `WhiteNoise <https://whitenoise.readthedocs.io/>`_ from inside the Django
process, so no separate web server (nginx/Apache) is required and, apart from one
command, nothing needs root.

All the files involved live in the ``deploy/`` directory of the repository:

``deploy/gunicorn.conf.py``
   gunicorn configuration (workers, threads, logging, worker warm-up).
``deploy/install_service.sh``
   Generates, installs and enables the systemd user unit
   ``~/.config/systemd/user/fomo.service``.

How it fits together
--------------------

.. code-block:: text

   browser ──HTTP──> gunicorn arbiter (bind address, e.g. host:6729)
                       ├── worker 1 (4 threads) ─┐
                       ├── worker 2 (4 threads) ─┼─> Django + WhiteNoise ─> SQLite
                       └── worker 3 (4 threads) ─┘        │
                                                          └─> /static/ from src/_static/
   systemd --user ──supervises──> gunicorn   (ExecStartPre: collectstatic)

* Each worker process imports :mod:`solsys_code.ephem_utils` and therefore holds
  its own copy of the SPICE kernels and ASSIST ephemeris (roughly 450 MB RSS per
  worker). Concurrency within a worker comes from threads (``gthread`` worker
  class).
* Workers load the full URLconf in gunicorn's ``post_worker_init`` hook, so the
  ephemeris machinery is set up before the first request rather than during it.
* ``collectstatic`` runs on every service start, so ``src/_static/`` (gitignored)
  is always current.

Prerequisites
-------------

* A working FOMO install in a virtual environment, as described in
  :doc:`../installation`, with the database migrated and the SPICE kernels
  already downloaded to ``~/.cache/sorcha/`` (run the dev server once, or
  generate one ephemeris, to trigger the ~1.6 GB download before switching over).
* systemd (any modern Linux distribution).
* ``sudo`` rights for one command (enabling lingering, below).

Installation
------------

1. Install the deployment extra (gunicorn) into the FOMO environment:

   .. code-block:: console

      >> pip install -e '.[deploy]'

2. Add production overrides to ``src/fomo/local_settings.py``. This file is
   gitignored and star-imported at the end of ``src/fomo/settings.py``. At a
   minimum:

   .. code-block:: python

      DEBUG = False
      ALLOWED_HOSTS = ['your-host.example.org']

   ``ALLOWED_HOSTS`` must contain the hostname that users type into their
   browser. You should also replace the ``SECRET_KEY`` (see
   :ref:`secret-key-rotation`).

3. Generate and enable the service, passing the address to listen on (it
   defaults to ``$(hostname -f):6729``):

   .. code-block:: console

      >> deploy/install_service.sh your-host.example.org:6729

   The script must be run with the FOMO virtualenv (or conda env) active: the
   unit file it writes points at that environment's ``python`` and
   ``gunicorn`` and at the current checkout. Re-run it if either moves; don't
   edit the generated unit by hand.

4. Stop any ``manage.py runserver`` already using the port, then start the
   service:

   .. code-block:: console

      >> systemctl --user start fomo
      >> systemctl --user status fomo

5. Enable lingering (once per user, needs ``sudo``). Without it, systemd only
   runs your user services while you are logged in, so FOMO would not start
   at boot:

   .. code-block:: console

      >> sudo loginctl enable-linger $USER

Day-to-day operation
--------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Task
     - Command
   * - Status / which workers are running
     - ``systemctl --user status fomo``
   * - Follow the logs (access log + Django logging)
     - ``journalctl --user -u fomo -f``
   * - Logs since the last boot
     - ``journalctl --user -u fomo -b``
   * - Stop / start / restart
     - ``systemctl --user stop|start|restart fomo``
   * - Disable starting at boot
     - ``systemctl --user disable fomo``

Deploying a code update:

.. code-block:: console

   >> git pull
   >> pip install -e '.[deploy]'      # only if dependencies changed
   >> python manage.py migrate        # only if there are new migrations
   >> systemctl --user restart fomo

Use ``restart`` rather than ``reload``: ``reload`` (``SIGHUP``) restarts the
workers gracefully and picks up new code, but does not re-run
``collectstatic``.

Configuration
-------------

The bind address and worker counts are read from environment variables by
``deploy/gunicorn.conf.py``:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Variable
     - Default
     - Meaning
   * - ``FOMO_BIND``
     - ``127.0.0.1:8000``
     - Address(es) to listen on; comma-separated for several. Set in the unit
       file by ``install_service.sh``.
   * - ``FOMO_WORKERS``
     - ``3``
     - Number of worker processes (~450 MB each).
   * - ``FOMO_THREADS``
     - ``4``
     - Threads per worker.

Extra environment for the service, such as the ``TNS_*`` and ``FINK_*``
variables read by ``settings.py``, goes in ``~/.config/fomo/fomo.env`` (one
``NAME=value`` per line). Remember that a systemd service does **not** inherit
your login shell's environment. Restart the service after editing it.

To run the same configuration by hand from the repository root (e.g. to debug a
startup problem with the service stopped):

.. code-block:: console

   >> FOMO_BIND=127.0.0.1:8001 gunicorn -c deploy/gunicorn.conf.py

.. _secret-key-rotation:

Rotating the SECRET_KEY
-----------------------

The ``SECRET_KEY`` in ``src/fomo/settings.py`` is committed to a public
repository and must not be used in a deployment. TOM Toolkit derives the key it
uses to encrypt stored credentials (``EncryptedModelField`` values, e.g. users'
facility API keys) from ``SECRET_KEY``, so changing it naively would make that
data unreadable. Rotate it gracefully instead:

1. Generate a new key:

   .. code-block:: console

      >> python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"

2. In ``src/fomo/local_settings.py`` set the new key as primary and keep the old
   one as a fallback:

   .. code-block:: python

      SECRET_KEY = '<new key>'
      SECRET_KEY_FALLBACKS = ['<old key>']

3. Re-encrypt all stored values under the new key, then restart:

   .. code-block:: console

      >> python manage.py rotate_encryption_key
      >> systemctl --user restart fomo

4. Once the command reports no failures, the fallback can be removed from
   ``local_settings.py`` (and the service restarted again). Existing login
   sessions are invalidated when the fallback is removed.

Design decisions
----------------

WhiteNoise rather than nginx/Apache
   Keeps the whole stack inside the virtualenv and the user's account, with no
   root-owned web-server configuration, and carries straight over to a future
   container. Static assets are a few MB of CSS/JS for a small number of
   users, which WhiteNoise handles comfortably. With ``DEBUG=True`` (development)
   WhiteNoise serves directly from the static finders, so ``runserver`` behaves
   as before.

No ``preload_app``
   Importing :mod:`solsys_code.ephem_utils` opens the SPICE kernel and ASSIST
   ephemeris files. If that happened in the gunicorn arbiter before forking,
   all workers would share those file descriptors (and their seek offsets).
   Each worker therefore loads them itself (in ``post_worker_init``).

Few processes, several threads
   The database is SQLite, which serializes writes across processes. Keeping
   the process count low limits ``database is locked`` errors under concurrent
   writes.

gunicorn control socket disabled
   gunicorn 26+ opens a control socket at a per-user default path
   (``$XDG_RUNTIME_DIR/gunicorn.ctl``) shared by every gunicorn instance that
   user runs. systemd already provides process control, so it is turned off.

Limitations
-----------

* **No HTTPS.** gunicorn serves plain HTTP. Put a TLS-terminating reverse proxy
  in front before exposing FOMO beyond a trusted network (and then set
  ``SESSION_COOKIE_SECURE``/``CSRF_COOKIE_SECURE`` and
  ``CSRF_TRUSTED_ORIGINS``).
* **Uploaded files are not served.** Data products under ``MEDIA_ROOT``
  (``/data/``) are only served by Django when ``DEBUG=True``; WhiteNoise
  handles static files, not uploads. Once data products are uploaded, a reverse
  proxy (or similar) needs to serve ``/data/``.
* **Startup ordering.** systemd user units cannot wait on the system's
  ``network-online.target``. If the bind address's interface is not up yet,
  gunicorn retries for a few seconds and then exits, and systemd restarts it
  10 seconds later (``Restart=on-failure``).
* **SQLite.** Adequate for a handful of users; a busier deployment should move
  to PostgreSQL.
