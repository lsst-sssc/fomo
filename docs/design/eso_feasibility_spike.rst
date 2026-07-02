ESO/VLT Calendar Sync — Feasibility Spike
==========================================

This document records the feasibility spike investigating whether ESO/VLT
observation sync can work at all for FOMO's telescope-runs calendar. It was
written after an investigation (2026-07-01) that connected to the real ESO
Phase 2 (P2) API with production Paranal credentials, captured live OB
status/execution data, and confirmed a headless credential-sourcing path for
a future management command. No sync command was built during this spike —
the deliverable is this decision record and its full-detail companion,
``.planning/phases/13-eso-feasibility-spike/13-DECISION.md``.

Background
----------

FOMO already syncs queue-scheduled observation blocks to the calendar for
LCO (queue banner -> placed block, Stage 3) and Gemini (submission-time
window banner, Stage 3b). ESO/VLT observing is different again: Paranal
(VLT) and La Silla (NTT) both run in **Service Mode**, where Paranal Science
Operations staff choose which Observation Blocks (OBs) to execute in real
time based on current conditions — there is no advance per-OB schedule
published ahead of time, unlike LCO's queue scheduler. (ESO/NTT *classical*
scheduling — whole nights assigned in advance — is a separate, already-solved
problem handled by Stage 2's ``load_telescope_runs``; this spike is about
*queue*/OB-level sync, not classical nights.)

The core constraint that reshaped this milestone: the installed
``tom_eso==0.2.4`` plugin cannot create ``ObservationRecord`` rows through
the standard TOM submission flow (``submit_observation()`` always returns an
empty ID list) and does not implement the TOM Toolkit status/URL interface
(``get_observation_status()``, ``get_observation_url()``, and
``data_products()`` all raise ``NotImplementedError``). This meant the usual
"read ``ObservationRecord`` rows, sync them to the calendar" pattern used by
LCO and Gemini had no data to read for ESO — the open question this spike
had to answer was whether *any* real ESO data was reachable at all, and if
so, how.

Key finding
-----------

**Bypass — sync straight from the ESO P2 API (** ``p2api`` **) to**
``CalendarEvent``, **skipping** ``ObservationRecord`` **for ESO entirely.**
Real Paranal production credentials connect and return real OB data through
direct P2 API calls (``getOB()``, ``getOBExecutions()``,
``getNightExecutions()``); a headless credential-sourcing path (a
``FACILITIES['ESO']`` settings entry, mirroring LCO/SOAR/GEM) was confirmed
viable without needing the session-bound, per-user ``ESOProfile`` path. No
evidence was gathered (or needed to be gathered, given this phase's
read-only guardrail) for creating hand-built ``ObservationRecord`` rows —
that is the Bridge option, and this spike's real-data path never touched it.

Investigation summary
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Capability
     - Status
     - Notes
   * - Paranal (VLT) P2 connection
     - Working
     - ``ESOAPI(environment='production', ...)`` connects and returns real
       data; confirmed via live ``getOB()``/``getNightExecutions()`` calls.
   * - La Silla (NTT) P2 connection via ``tom_eso.eso_api.ESOAPI``
     - Fails (wrapper bug, not API access)
     - Fails at ``ESOAPI.__init__`` because it unconditionally constructs a
       Phase-1 (``p1api``) connection first, and ``p1api``'s ``API_URL`` has
       no ``production_lasilla`` entry. ``p2api``'s own ``API_URL`` *does*
       support ``production_lasilla``.
   * - La Silla (NTT) P2 connection via direct ``p2api`` bypass
     - Connects; La Silla-specific data unconfirmed
     - ``p2api.ApiConnection('production_lasilla', ...)`` (bypassing
       ``ESOAPI``/``p1api``) connects without error and returns real data —
       confirming the wrapper-bug diagnosis. The one run returned was a
       Paranal-instrument run already seen under ``production``, so this
       proves the connection path is open but does not yet confirm distinct
       La-Silla-sourced OB data is reachable for this account.
   * - ``get_observation_status()`` / ``get_observation_url()`` /
       ``data_products()`` (``tom_eso``-level)
     - Not usable; unimplemented in ``tom_eso``
     - All three raise ``NotImplementedError`` in the installed
       ``tom_eso==0.2.4``. Not needed under the Bypass path — a future sync
       command reads ``obStatus`` directly from ``p2api``'s ``getOB()`` /
       ``getOBExecutions()`` / ``getNightExecutions()`` responses instead.
   * - Real OB status data (``p2api``-level, direct)
     - Reachable
     - A never-executed OB returned ``obStatus='P'`` with an empty executions
       list; a separately-queried, already-executed OB's per-night execution
       record returned ``obStatus='M'`` (Must Repeat) with a concrete
       ``from``/``to`` time window and ``grade='X'``.
   * - Headless credential-sourcing (for a future management command)
     - Viable
     - Direct ``ESOAPI(environment, username, password)`` construction from
       environment-variable-supplied credentials works with no active
       Django session and no ``ESOProfile`` involved — the same pattern
       LCO/SOAR/GEM already use via ``FACILITIES[...]`` settings entries.

ESO P2 ``obStatus`` vocabulary (12 codes)
--------------------------------------------

If a future sync command layers OB status onto the banner (status-aware
sync, see Future scope below), this is the vocabulary it would map, entirely
distinct from LCO's or Gemini's terminal-state sets:

.. list-table::
   :header-rows: 1
   :widths: 12 48 20

   * - Code
     - Meaning
     - Terminal?
   * - ``P``
     - Partially defined (just created)
     - No
   * - ``D``
     - Defined (passed certification, ready for review)
     - No
   * - ``-``
     - Rejected (needs user attention)
     - No
   * - ``R``
     - Review (under revision by support astronomer)
     - No
   * - ``+``
     - Accepted (ready to be observed)
     - No
   * - ``C``
     - Completed (executed successfully, will not repeat)
     - Yes
   * - ``X``
     - Executed (successfully completed, can repeat — e.g. visitor mode)
     - Yes (per-execution)
   * - ``M``
     - Must repeat (executed outside constraints, will be requeued)
     - No (requeues)
   * - ``A``
     - Aborted during execution (will be requeued)
     - No (requeues)
   * - ``F``
     - Failed (absolute time window expired; read-only, irreversible)
     - Yes
   * - ``K``
     - Kancelled (support-astronomer set, irreversible)
     - Yes
   * - ``T``
     - Terminated (run terminated, irreversible)
     - Yes

Future scope
------------

See ``.planning/phases/13-eso-feasibility-spike/13-DECISION.md`` for the full
recommendation rationale and future-sync sketch. In brief, a future
``sync_eso_observation_calendar``-style command (not built in this
milestone) would:

* Reuse ``solsys_code/calendar_utils.py:insert_or_create_calendar_event()``
  unchanged — it is already facility-agnostic.
* Key idempotency on a synthetic identifier, ``ESO:{p2_environment}/{obId}``,
  following the precedent set by Gemini's ``GEM:{prog}/{observation_id}``.
* Choose between a banner-only window sync (OB run-period dates, no status)
  or a status-aware sync (layering the ``obStatus`` vocabulary above onto the
  banner) — the latter is real-data-supported by this spike but requires a
  polling-window policy (which night(s) to check per OB) that this
  investigation did not need to resolve.

This is input to a future milestone's requirements, not implemented here.
