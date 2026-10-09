``tom_calendar`` vs YSE_PZ Calendar Support
============================================

This note compares the calendar support already used by FOMO
(:doc:`telescope_runs_calendar`, built on the TOM Toolkit's ``tom_calendar``
package) against the calendar views in `YSE_PZ
<https://github.com/berkeleyseti/YSE_PZ>`_ (Young Supernova Experiment -
PhotoMetry and Spectroscopy), a sibling time-domain follow-up TOM. Both
projects solve the same underlying problem -- show telescope time on a
calendar -- but with architectures different enough to be worth recording
before deciding how Stages 2-4 of issue #37 should evolve.

Sources: ``tom_calendar`` is installed in the FOMO virtualenv
(``site-packages/tom_calendar``); YSE_PZ was read from a local clone
(``~/git/YSE_PZ``, ``YSE_App/models/``, ``YSE_App/views.py``,
``YSE_App/templates/YSE_App/*calendar*.html``).

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``tom_calendar``
     - YSE_PZ
   * - Data model
     - One generic ``CalendarEvent`` model (+ ``EventTodo``), independent of
       any domain model
     - No calendar-specific model at all; each calendar view queries
       existing domain models directly (``OnCallDate``,
       ``ClassicalObservingDate``, ``ToOResource``, ``SurveyObservation``)
   * - Number of calendars
     - One generic calendar, reusable for any event type
     - Five separate, purpose-built calendar pages (on-call, classical
       observing, ToO, PS1/ZTF survey, DECam survey)
   * - Create / edit / delete
     - Inline, via htmx modals on the calendar grid itself
       (``create_event`` / ``update_event`` / ``delete_event``)
     - Not on the grid. On-call dates go through a separate
       ``add_oncall_observer`` form view or the DRF API; the other four
       calendars are read-only renders of data created elsewhere (proposals,
       survey scheduler, ToO allocation)
   * - Rendering
     - Server-renders a month grid in Django; htmx swaps partials in place
       (no full page reloads, no calendar JS library)
     - Client-side `FullCalendar.js <https://fullcalendar.io/>`_ (an old
       ~2015-era jQuery build vendored via Bower); Django only emits a
       JS ``events: [...]`` literal embedded in the template
   * - Astronomy content
     - None built in; moon phase is the only astronomy ``tom_calendar``
       itself computes (``MoonPhase.from_date``, used for the grid icon)
     - Each observing calendar computes sunset/sunrise itself per view
       (``astroplan.Observer.sun_set_time`` / ``sun_rise_time``), duplicated
       across ``too_requests``, ``yse_home``, ``yse_observing_calendar``,
       ``decam_observing_calendar``
   * - Multi-telescope handling
     - Free-text ``telescope`` / ``instrument`` ``CharField`` per event; no
       enforced relationship to a site model
     - A real ``Telescope`` model with lat/lon/elevation; resources
       (``ToOResource``, ``ClassicalResource``) hold an FK to it, so
       coordinates are looked up, not duplicated
   * - "Color" semantics
     - ``CalendarEvent.color`` is a read-only property derived from ``pk``
       -- cannot be used to encode status
     - Colors are computed per-view in Python (cycling through a fixed
       palette, keyed by user or telescope) and passed into the JS event
       objects, so color *can* encode meaning (e.g. one color per telescope)

The Data Model
---------------

``tom_calendar.models.CalendarEvent`` is a single, domain-agnostic table:
``title``, ``description``, ``start_time``, ``end_time``, ``url``,
``telescope``, ``instrument``, ``proposal``, ``user`` (free text, not an FK),
an optional ``target_list`` FK, and a related ``EventTodo`` checklist. It
carries no opinion about *what kind* of event it represents -- an on-call
shift, a classical night, and a queue window would all be rows in the same
table, distinguished only by their field values. This is exactly what
:doc:`telescope_runs_calendar` exploits: Stage 1-4 reuse the existing fields
with no migration.

YSE_PZ instead has dedicated models per concept, with real foreign keys:

* ``OnCallDate`` / ``YSEOnCallDate`` -- a date plus a ``ManyToManyField`` to
  ``User``.
* ``TelescopeResource`` (abstract) -- FK to ``Telescope``, optional FK to
  ``PrincipalInvestigator``, ``begin_date_valid`` / ``end_date_valid``
  (the "semester"). Subclassed by ``ToOResource`` (adds awarded/used ToO
  hours and triggers), ``QueuedResource`` (awarded/used hours), and
  ``ClassicalResource`` (adds nothing beyond the base).
* ``ClassicalObservingDate`` -- FK to ``ClassicalResource`` and to
  ``ClassicalNightType``, plus a single ``obs_date`` (one row **per night**,
  not a date range).
* No model represents a "calendar event" in the abstract; the survey
  calendars (``yse_observing_calendar``, ``decam_observing_calendar``) are
  built directly from ``SurveyObservation`` rows that already exist for
  scheduling purposes, with no calendar-specific persistence at all.

The tradeoff: ``tom_calendar``'s generic model is reusable with zero schema
changes (good for FOMO's incremental Stage 1-4 plan) but pushes all
telescope/instrument/PI structure into untyped ``CharField`` text. YSE_PZ's
typed models give referential integrity (a ``Telescope`` has one
authoritative lat/lon/elevation; a ``ClassicalResource`` really does belong
to one PI) at the cost of a calendar view per concept and no generic
create/edit UI.

The Views
---------

``tom_calendar.views`` (summarized fully in the prior chat turn) is six
small functions behind one ``app_name='tom_calendar'`` URL namespace:
``render_calendar`` (month grid, htmx partial or full page), and
``create_event`` / ``update_event`` / ``delete_event`` / ``create_todo`` /
``update_todo``, all of which mutate ``CalendarEvent``/``EventTodo`` and
re-render either a form partial or the calendar itself, firing htmx custom
events (``calRefresh``, ``calClose``) to update the page without a reload.
There is exactly one ``EventForm`` (a ``ModelForm`` on ``CalendarEvent``)
covering every event, classical or queue alike.

YSE_PZ's calendar views (``YSE_App/views.py``) are five separate,
hand-written, read-only renders, each tied to one model and visualization:

* ``calendar`` / ``yse_oncall_calendar`` -- on-call rosters, colored per
  user, built from ``OnCallDate`` / ``YSEOnCallDate``.
* ``observing_calendar`` -- classical nights, colored per telescope, from
  ``ClassicalObservingDate``.
* ``too_calendar`` -- a 60-days-back/60-days-forward window of
  ``ToOResource`` validity ranges, with per-day date lists computed in
  Python and handed to the template.
* ``yse_observing_calendar`` / ``decam_observing_calendar`` -- a hardcoded
  40-day window (today - 30 to +9) of ``SurveyObservation`` rows for PS1/ZTF
  or DECam, with sunset/sunrise and moon illumination recomputed per day
  inside the view, and pre-formatted summary strings built with manual
  string concatenation rather than template logic.

None of the YSE_PZ calendar views accept a month/year query parameter the
way ``render_calendar`` does -- each hardcodes its own date window in Python
and is not navigable forward/backward. Creating new entries happens outside
the calendar entirely (a separate form view for on-call dates, the Django
admin or DRF API for the resource models).

What This Means for Issue #37
------------------------------

FOMO's existing plan (:doc:`telescope_runs_calendar`) already chose the
``tom_calendar`` approach -- reuse the generic ``CalendarEvent`` model,
synced/ingested by management commands -- and this comparison does not
surface anything that would change that choice for Stages 2-4:

* FOMO has one generic need (telescope runs as calendar blocks), not five
  visually distinct calendar products, so ``tom_calendar``'s single reusable
  model is the better fit; replicating YSE_PZ's "one bespoke view per
  data source" pattern would mean writing (and maintaining) a separate
  calendar per telescope/scheduling-model combination.
* ``tom_calendar``'s htmx create/edit/delete-on-the-grid UI is more capable
  than anything in YSE_PZ's calendars (which have no inline editing at all),
  so Stage 2's idempotent management-command ingest is additive on top of
  an already-richer UI, not a gap to fill.
* YSE_PZ's pattern of recomputing sunset/sunrise/twilight independently in
  four different views (``too_requests``, ``yse_home``,
  ``yse_observing_calendar``, ``decam_observing_calendar``) is the duplication
  Stage 1's shared ``solsys_code/telescope_runs.py`` helper is explicitly
  designed to avoid -- worth treating as a cautionary example rather than a
  pattern to copy.
* The one idea worth borrowing: YSE_PZ's ``Telescope`` model (FK'd from
  every resource) instead of free-text fields. FOMO already has the
  equivalent in ``Observatory`` (MPC-obscode keyed); Stage 1 deliberately
  looks sites up from it rather than hardcoding coordinates, and the Open
  Items in :doc:`telescope_runs_calendar` already flag making the
  ``SITES`` registry data-driven from ``Observatory`` -- the same direction
  YSE_PZ's FK-based design points to, just not yet applied to
  ``CalendarEvent.telescope`` itself (which remains free text, as
  ``tom_calendar`` defines it).
