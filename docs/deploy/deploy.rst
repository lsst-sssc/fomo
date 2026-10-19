Deploying FOMO
==============

FOMO can be run in three increasingly robust ways:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Approach
     - Suited to
     - Documentation
   * - ``manage.py runserver``
     - Local development and testing. Single-threaded, ``DEBUG=True``, and it
       must be restarted by hand after every reboot.
     - :doc:`../installation`
   * - gunicorn + systemd user service
     - A long-running instance on a single host that should survive reboots and
       crashes, without needing a container runtime or root-owned web server
       configuration.
     - :doc:`gunicorn_systemd`
   * - Containerized (Docker)
     - Not yet available.
     - --

.. toctree::
   :maxdepth: 1

   gunicorn + systemd (single host) <gunicorn_systemd>
