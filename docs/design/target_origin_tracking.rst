Target Origin Tracking
======================

This document records the research and design decisions around tracking where a
FOMO target was first ingested from.

Background
----------

As FOMO grows to ingest targets from multiple sources — JPL Scout, the MPC, and
eventually Rubin alert brokers such as Fink, Lasair, and SNAPS — it becomes
useful to record the origin of each target.  Knowing where a target came from
informs how it should be displayed, filtered, and managed.  For example, Scout
candidates are pre-MPC objects with unconfirmed orbits; Fink or Lasair alerts
flag dynamic events on objects that MPC already tracks; hand-ingested MPC targets
were deliberately added by a user.

Reference: SNEx2
----------------

To inform the design, we studied `SNEx2 <https://github.com/LCOGT/snex2>`_, an
LCO application that also uses the TOM Toolkit and ingests targets from multiple
broker sources (TNS, ALeRCE, Lasair/IRIS).

SNEx2 tracks target origin via two mechanisms:

* A **`BrokerTarget` staging model** — a separate Django model (not a
  :class:`~tom_targets.models.Target` subclass) with a ``stream_name``
  CharField recording which alert stream first flagged the object.  Targets live
  here until a human promotes them to the main :class:`~tom_targets.models.Target`
  table.

* A **`SNExTarget` model subclass** of :class:`~tom_targets.models.BaseTarget`
  — for promoted targets, domain-specific attributes (redshift, classification,
  first detection date, etc.) are stored as typed model columns, not as
  :class:`~tom_targets.models.TargetExtra` key-value pairs.

Critically, SNEx2 **does not use** :class:`~tom_targets.models.TargetExtra` for
anything it needs to filter or display reliably.  The pattern is considered too
untyped and awkward to query for properties that actually matter.

A search across all open FOMO branches confirmed that FOMO has no existing
``TargetExtra`` usage and no source or origin tracking of any kind (checked
2026-06-08).

Taxonomy of Sources
-------------------

Not all sources play the same role:

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Source
     - Role
     - Status
   * - MPC
     - Canonical orbital authority.  Source of truth for confirmed orbits,
       astrometric, and photometric data.  FOMO measurements are reported back
       to MPC.
     - Existing (via ``fetch_jplsbdb_objects``)
   * - Scout
     - JPL real-time service for unconfirmed candidates.  Objects here have not
       yet received an MPC designation and carry uncertain orbits.
     - Existing (Scout data service)
   * - Fink
     - Rubin alert broker.  Will flag interesting dynamic events (outbursts,
       unexpected brightening) on MPC-known objects.
     - Planned
   * - Lasair
     - Rubin alert broker.  Same role as Fink.
     - Planned
   * - SNAPS
     - Rubin alert broker.  Same role as Fink.
     - Planned
   * - Manual
     - Target added directly by a user, not via any automated source.
     - Implicit

The key distinction is that **MPC is infrastructure**, not a broker.  It is the
orbital authority and the clearing house for all survey discoveries.
Fink/Lasair/SNAPS, by contrast, are **event-driven**: they flag that something
interesting happened to an object that MPC already knows about.

Design Decision
---------------

We record the **first ingestion source** for each target.  Ongoing provenance
(e.g. a target later appearing in a second broker) is not tracked at this time,
because the interesting question is why a target entered FOMO, not every
subsequent place it was seen.

We use a dedicated ``TargetIngestion`` model with a one-to-one relationship to
:class:`~tom_targets.models.Target`:

.. code-block:: python

    class TargetIngestion(models.Model):
        target = models.OneToOneField(
            Target,
            on_delete=models.CASCADE,
            related_name='ingestion',
        )
        source = models.CharField(max_length=50)
        # e.g. 'Scout', 'MPC', 'Fink', 'Lasair', 'SNAPS', 'Manual'
        ingested_at = models.DateTimeField(auto_now_add=True)

Rejected alternatives:

* **``TargetExtra``** — The TOM Toolkit's built-in key-value store for
  supplementary target data.  Rejected because key-value lookups are awkward
  (``TargetExtra.objects.filter(key='source', value='Scout')``), untyped, and
  not idiomatic for a property as fundamental as origin.

* **Field on a Target subclass** — SNEx2 takes this approach, but it requires
  subclassing :class:`~tom_targets.models.BaseTarget` across the whole project.
  A one-to-one model achieves the same queryability
  (``Target.objects.filter(ingestion__source='Scout')``) without touching the
  TOM core target model.

Deferred Decisions
------------------

Broker-specific metadata (e.g. Scout's orbit-fit probability, a Fink
classification confidence score) has been deferred.  Solar system object alerts
through the Rubin brokers have not been observed in practice as of the time of
writing (2026-06-08), so the shape of that metadata is unknown.  The
``TargetIngestion`` model can be extended with additional fields when the need
becomes concrete.

Implementation Plan
-------------------

The touch points below are stable enough to record, but deliberately avoid naming
specific functions or line numbers — those details will drift as the codebase
evolves.  Resolve the open questions before starting.

Model and migration
^^^^^^^^^^^^^^^^^^^

* Add ``TargetIngestion`` to ``solsys_code/models.py`` (or a dedicated
  ``solsys_code/ingestion.py`` if the model grows).
* Generate and commit a migration.
* Register the model in ``solsys_code/admin.py`` so ingestion source is visible
  in the Django admin.

Ingestion touch points
^^^^^^^^^^^^^^^^^^^^^^

Each place that creates a :class:`~tom_targets.models.Target` needs a
corresponding ``TargetIngestion`` row created in the same transaction:

* **MPC / JPL SBDB** — the ``fetch_jplsbdb_objects`` management command; set
  ``source='MPC'``.
* **Scout** — the Scout data service (``solsys_code/`` Scout integration); set
  ``source='Scout'``.
* **Future Rubin brokers** — each new broker data service or alert-stream
  ingestion command; set ``source`` to the broker name (``'Fink'``,
  ``'Lasair'``, ``'SNAPS'``).
* **Manual / web form** — any view that creates a target via the TOM Toolkit
  create-target form; set ``source='Manual'``.  Consider using a TOM
  ``post_save`` signal or an overridden ``TargetCreateView`` so manual targets
  are never missed.

View implications
^^^^^^^^^^^^^^^^^

* The Scout "live candidates" view already implicitly restricts to Scout-sourced
  objects by querying the Scout API at render time.  Once ``TargetIngestion``
  exists, views that show database-backed target lists can add
  ``ingestion__source='Scout'`` filters to be explicit and consistent.
* The target list view may benefit from a source filter widget so users can
  narrow the table by origin.
* Target detail pages could display the ingestion source and timestamp as a
  read-only field.

Open questions to resolve before implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **``source`` as free text or ``choices``?**  A ``CharField`` with a
  ``choices`` tuple makes valid values explicit and enables validation, but
  requires a code change to add new brokers.  Free text is more flexible but
  easier to mistype.  A ``choices`` field is recommended unless broker names are
  expected to be highly dynamic.

* **What happens when the same object arrives from a second source?**  The
  current design records only the first ingestion and ignores later arrivals.
  If cross-source provenance becomes important, the ``OneToOneField`` would need
  to become a ``ForeignKey`` (one target, many ingestion events).  Decide
  before creating the migration; changing this later requires a schema change.

* **Pre-existing targets** — the database already contains targets with no
  ``TargetIngestion`` row.  The migration could backfill a ``source='Unknown'``
  row for each, or leave them without a row and let code handle the missing case
  with ``hasattr(target, 'ingestion')``.  Explicit backfill is cleaner for
  filtering.
