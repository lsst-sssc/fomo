Fink Solar System Object Support
================================

This document records research into what solar system object (SSO) support is
available to FOMO from the Fink broker's Rubin/LSST alert processing, how it
compares to the JPL SBDB and MPC Explorer ingestion FOMO already has, and what
would need to change before Fink becomes a useful target source.

Findings reflect the state of the Fink repositories' ``main`` branches as of
2026-07-25, which may run ahead of what is deployed to the production Kafka
cluster.

Summary
-------

Fink currently offers FOMO **no discovery capability** for solar system objects
beyond what MPC and JPL already provide, and **no real-time SSO substream** for
Rubin.  The one genuine benefit is access to per-epoch Rubin photometry for
objects FOMO has already selected, available today only as a batch download.

The recommendation is to keep JPL SBDB and MPC Explorer as the target selection
layer, treat Fink as a potential *enrichment* step on existing targets, and
revisit when the announced real-time SSO filter ships.

Package Landscape
-----------------

Fink support reaches FOMO through two separately versioned packages.

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Package
     - FOMO pin
     - Installed
     - Latest
   * - ``tomtoolkit``
     - ``==3.0.0a9``
     - 3.0.0a9
     - 3.0.0 (2026-07-15)
   * - ``tom_alertstreams``
     - ``>=1.2.1``
     - 1.2.1
     - 1.3.0 (2026-07-16)
   * - ``tom_fink``
     - ``>=1.0.0``
     - 1.0.0
     - 2.0.1 (2026-07-16)

Fink has **not** moved into ``tom_alertstreams``.  That package ships only
``antares.py``, ``gcn.py``, and ``hopskotch.py``; the ``fink`` extra it
advertises merely pulls in ``fink-client``.  Fink remains its own ``tom_fink``
package.

The tom_fink 2.0 bump
^^^^^^^^^^^^^^^^^^^^^

``tom_fink/alertstream.py`` is byte-identical between 1.0.0 and 2.0.1, so
FOMO's ``ALERT_STREAMS`` configuration and the
``tom_fink.alertstream.FinkAlertStream`` path need no changes across the major
version bump.

The entire 2.0 change is in the data service half, ``tom_fink/fink.py``, which
FOMO registers via ``SolsysCodeConfig.data_services()``:

* :class:`~tom_dataproducts.models.ReducedDatum` becomes
  ``PhotometryReducedDatum`` (TOM Toolkit 3.0's typed reduced datum models).
* ``data_type`` is dropped in favour of explicit ``brightness``,
  ``brightness_error``, and ``bandpass`` fields.
* Timestamps become timezone-aware via ``TimezoneInfo()``.
* Minor hygiene: request ``timeout=60``, ``assert`` replaced by
  ``QueryServiceError``, mutable default argument fixed.

Upgrade blocker
^^^^^^^^^^^^^^^

``tom_fink`` 2.0.1 requires ``tomtoolkit>=3.0.0,<4.0.0``.  FOMO pins
``==3.0.0a9``, and under PEP 440 a pre-release sorts *below* the final release,
so the pin does not satisfy the requirement.  The pinned 3.0.0a9 exposes only
:class:`~tom_dataproducts.models.ReducedDatum`, with no
``PhotometryReducedDatum``, so ``tom_fink`` 2.x would fail on import.

The Fink upgrade is therefore gated behind completing the bootstrap work and
moving to ``tomtoolkit>=3.0.0``.

``tom_fink`` 2.0.1 also requires ``tom-alertstreams>=1.3.0`` and
``fink-client>=11,<12`` (8.10 is currently installed).

.. warning::

   The ``tom_alertstreams[fink]`` extra pins ``fink-client<9`` while
   ``tom_fink`` 2.x requires ``>=11``.  These are unresolvable together.  Take
   the Fink client from ``tom_fink`` only; never install the ``fink`` extra.

SSO Support in Fink's Rubin Processing
--------------------------------------

Rubin changed the alert data model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under ZTF, all alerts shared one schema and moving objects were separated by
post-hoc cuts.  LSST performs MPC association *before* the alert is sent, so
packet contents differ by type:

* Non-SSO alerts carry ``diaObject``.
* SSO alerts carry ``mpc_orbit`` and ``ssSource``.
* ``ssSource`` includes ``diaDistanceRank``, ranking each source by closeness to
  the predicted SSO position, for crowded matches.

The identifier of interest is ``ssObjectId``, not ``diaObject``.

No SSO science module has been ported to Rubin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Modules
   * - ``fink_science/rubin/``
     - ad_features, cats, hostless_detection, orphans, random_forest_snia,
       slsn, snn, xmatch
   * - ``fink_science/ztf/``
     - the same plus **asteroids** and **ssoft**

The ``asteroids`` module (the ZTF ``roid`` classifier) and ``ssoft`` (the Solar
System Object Fink Table, providing HG / HG1G2 / sHG1G2 phase curve fits) exist
only under ``ztf/``.  Fink-FAT orbit linking is likewise ZTF-only.

No SSO livestream filter exists for Rubin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All nine filters under ``fink_filters/rubin/livestream/`` target extragalactic,
supernova, hostless, TNS, or random-sample selections:

.. list-table::
   :header-rows: 1
   :widths: 35 50 15

   * - Filter
     - Selects
     - Uses ``is_sso``
   * - ``filter_extragalactic_lt20mag_candidate``
     - Rising, bright (mag < 20) extragalactic candidates
     - veto
   * - ``filter_extragalactic_new_candidate``
     - New extragalactic candidates, requiring two or more detections in the
       same band
     - no
   * - ``filter_extragalactic_svom``
     - New (< 5 d), bright (mag < 24), potentially extragalactic, for SVOM
       follow-up
     - veto
   * - ``filter_hostless_candidate``
     - Hostless alerts via ``elephant_kstest_template``
     - no
   * - ``filter_in_tns``
     - Alerts with a known TNS counterpart at time of emission
     - no
   * - ``filter_most_likely_sn``
     - Likely SNe from SuperNNova and CATS classifier scores
     - veto
   * - ``filter_remove_unlikely_transients``
     - Drops alerts unlikely to be transients of interest to DESC
     - veto
   * - ``filter_sn_near_galaxy_candidate``
     - Catalog-matched to a galaxy, properties consistent with SNe
     - veto
   * - ``filter_uniform_sample``
     - 1% uniformly random sample of all live alerts
     - no

Five of the nine accept ``is_sso``, and every one uses it to *exclude* solar
system objects as contamination.  The ZTF SSO topics
(``fink_sso_ztf_candidates_ztf``, ``fink_sso_fink_candidates_ztf``) have no
Rubin equivalent.

Note that ``filter_remove_unlikely_transients`` is explicitly scoped to DESC's
interests, which indicates who the current Rubin filter set was written for.

What ``is_sso`` actually means
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``fink_broker/rubin/science.py``:

.. code-block:: python

    df = df.withColumn("is_sso", df["ssSource"].isNotNull())

``is_sso`` is **not** a classifier or a Fink prediction.  It is simply "Rubin
attached an ``ssSource`` block", i.e. Rubin's own MPC pre-association succeeded.
The ``b_is_solar_system`` block in ``fink_filters/rubin/blocks.py`` is a
one-line passthrough of that flag.

This is the decisive point: **by construction, every object Fink can return is
already in the MPC catalog.**  Unlinked moving objects never receive the flag,
so Fink cannot surface an object that MPC and JPL do not already know about.

Available Route Today: Data Transfer
------------------------------------

Per Julien Peloton (Fink) on the Rubin community forum, there are two routes:

* **Data Transfer service — available now, batch only.**  Select nights, apply
  the ``b_is_solar_system`` block, choose fields, and stream the result out via
  the Fink/Rubin Science Portal.  Further filtering on orbital parameters such
  as semi-major axis is supported.
* **A real-time LSST SSO filter is announced as "coming"**, which would enable
  direct streaming and API database queries, mirroring the ZTF setup.  No date
  has been given.

No one in that discussion describes obtaining SSO alerts from Fink's Kafka
livestream, and no Rubin SSO topic name is published.  The ``fink-filters``
repository contains no filter-to-topic mapping, so topic names should be
confirmed against the live schema page before being wired into
``ALERT_STREAMS``.

Comparison with Existing JPL SBDB and MPC Explorer Ingestion
------------------------------------------------------------

The sources answer different questions and are complementary rather than
competing.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - JPL SBDB / MPC Explorer
     - Fink Data Transfer
   * - Nature
     - Orbit catalog
     - Per-epoch Rubin measurements
   * - Answers
     - "Which objects satisfy ``e>=1.2, q<1.3``?"
     - "What did Rubin detect, when, and how bright?"
   * - Provides
     - Elements, H/G, orbit class
     - ``diaSource`` photometry, ``ssSource`` geometry, ``mpc_orbit``
   * - Discovery
     - Yes, authoritative
     - No, MPC-known objects only
   * - Latency
     - On demand
     - Batch
   * - Role in FOMO
     - Target selection (``fetch_jplsbdb_objects``)
     - Enrichment of existing targets

Arguments for adding Fink
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Rubin photometry.**  Neither SBDB nor MPC Explorer provides a Rubin
  lightcurve.  This would allow validating ``add_magnitude`` in
  ``ephem_utils.py`` against measured magnitudes rather than assumed H/G values.
* **Observed rather than predicted brightness.**  Knowing an object was actually
  detected last night at a given magnitude is a far stronger follow-up trigger
  than a predicted magnitude.
* **Combined queries.**  Orbital-parameter cuts during transfer would allow
  "high-eccentricity objects Rubin detected recently, with photometry" in a
  single pass, which neither existing service can answer.

Arguments against, for now
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Batch only, so no benefit for rapid follow-up until the real-time filter
  ships.
* ``mpc_orbit`` is MPC's orbit, so there is no independent orbital information;
  JPL's own fits are arguably better and FOMO already retrieves them.
* No SSOFT phase curves and no Fink-FAT linking for Rubin.
* An additional credential path and ingestion pipeline to maintain.

Potential Future Developments
-----------------------------

Roughly in order of increasing effort and payoff:

#. **Unblock the upgrade.**  Complete the bootstrap work, move to
   ``tomtoolkit>=3.0.0``, and take ``tom_fink`` 2.0.1 with
   ``tom-alertstreams>=1.3.0``.  Required before anything else here, and needed
   regardless of the SSO question.
#. **Monitor for the Rubin SSO livestream filter.**  When it ships, FOMO's
   existing ``ALERT_STREAMS`` machinery should handle it with configuration
   changes only, given ``alertstream.py`` is unchanged across the bump.
#. **Prototype a Data Transfer enrichment path.**  A management command in the
   style of ``fetch_jplsbdb_objects`` that pulls Rubin photometry for existing
   targets by ``ssObjectId`` and stores it as ``PhotometryReducedDatum``.  Batch
   semantics suit a scheduled job better than a streaming consumer.
#. **Contribute an SSO filter to fink-filters.**  Since no Rubin SSO substream
   exists, the path to real-time Fink SSO data is to write one against
   ``is_sso``, ``mpc_orbit``, and ``ssSource`` and submit it upstream.  This
   would benefit the wider SSSC community, not just FOMO.
#. **Evaluate consuming Rubin's SSO stream directly.**  Given the MPC
   association is already performed upstream and Fink adds no SSO-specific
   processing for Rubin, going direct may be simpler than routing through a
   broker that currently treats these objects as contamination.

References
----------

* `Migration from ZTF to LSST <https://doc.lsst.fink-broker.org/data/ztf_to_lsst/>`_
  — Fink/LSST documentation.
* `Using brokers to get solar system object alerts <https://community.lsst.org/t/using-brokers-to-get-solar-system-object-alerts/11628>`_
  — Rubin community forum, February 2026.
* `Are comets in the alert stream? <https://community.lsst.org/t/are-comets-in-the-alert-stream/11791>`_
* `How can I reliably check Rubin moving candidates against MPC via API? <https://community.lsst.org/t/how-can-i-reliably-check-rubin-moving-candidates-against-mpc-via-api/11943>`_
* `astrolabsoftware/fink-filters <https://github.com/astrolabsoftware/fink-filters>`_
* `astrolabsoftware/fink-science <https://github.com/astrolabsoftware/fink-science>`_
* `astrolabsoftware/fink-broker <https://github.com/astrolabsoftware/fink-broker>`_
* `Fink livestream service <https://fink-broker.readthedocs.io/en/latest/services/livestream/>`_
* `Fink/LSST science roadmap <https://doc.lsst.fink-broker.org/broker/roadmap/>`_
* `Enabling discoveries of Solar System objects in large alert data streams <https://arxiv.org/html/2305.01123>`_
