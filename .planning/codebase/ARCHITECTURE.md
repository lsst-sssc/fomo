<!-- refreshed: 2026-09-04 -->
# Architecture

**Analysis Date:** 2026-09-04

## System Overview

FOMO is a Django web application built on the TOM Toolkit (Target & Observation Manager) that coordinates follow-up observations of Solar System targets. It combines three major feature areas:

1. **Ephemeris Generation** — Orbital mechanics, coordinate transforms, magnitude calculations
2. **JPL Discovery** — Query JPL SBDB for Solar System objects, create Targets
3. **Telescope Runs Calendar** (active focus) — Classical telescope runs, campaign-based observations, calendar event coordination, observation record synchronization

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Client Layer (Django Templates)                       │
│  Ephemeris Form / Calendar Month View / Campaign Tables / Attribution Queue     │
│  `src/templates/` — project-level templates + TOM template overrides             │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
┌────────────────────────────────┴───────────────────────────────────────────────┐
│                          View / Handler Layer                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│ • Ephemeris: `MakeEphemerisView`, `Ephemeris` (`solsys_code/views.py`)          │
│ • Calendar: `fomo_render_calendar()` (`solsys_code/views.py`)                   │
│ • Campaigns: `CampaignRunTableView`, `CampaignListView`,                        │
│   `CampaignRunSubmissionView`, `ApprovalQueueView`, `AttributionQueueView`      │
│   (`solsys_code/campaign_views.py`)                                             │
│ • Observatory CRUD: `CreateObservatory`, `ObservatoryList`, etc.                │
│   (`solsys_code/solsys_code_observatory/views.py`)                              │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
┌────────────────────────────────┴───────────────────────────────────────────────┐
│                         Form / Handler Layer                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│ • `EphemerisForm` - Date range, observatory, output options                     │
│ • `CampaignRunSubmissionForm` - New run intake, site resolution                 │
│ • `CampaignGapAnalysisForm` - Coverage gap query parameters                     │
│ (`solsys_code/forms.py`, `campaign_forms.py`)                                   │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
┌────────────────────────────────┴───────────────────────────────────────────────┐
│                    Orchestration / Business Logic Layer                        │
├────────────────────────────────────────────────────────────────────────────────┤
│ EPHEMERIS CHAIN:                                                               │
│   ephem_utils.py — n-body integration setup, coordinate transforms,            │
│   magnitude/sky-motion calculation (uses REBOUND + ASSIST + ERFA + Sorcha)     │
│                                                                                 │
│ CAMPAIGN CHAIN:                                                                │
│   • campaign_views.py — approval/attribution queue orchestration               │
│   • campaign_attribution.py — confirmation/dismissal/undo logic                │
│   • campaign_reconciler.py — idempotent run→event projection + key namespacing │
│   • calendar_utils.py — site-telescope mapping, calendar event create-or-update│
│   • campaign_utils.py — site resolution, throttling, link management           │
│   • campaign_gap.py — coverage gap analysis                                    │
│   • telescope_runs.py — classical-run parsing, sun-event calculation           │
│   • campaign_tables.py — django-tables2 table definitions                      │
│   • campaign_filters.py — django-filters filtersets                            │
│                                                                                 │
│ MANAGEMENT COMMANDS:                                                           │
│   • load_telescope_runs.py — ESO classical schedule ingest                     │
│   • sync_lco_observation_calendar.py — LCO queue observation records           │
│   • sync_gemini_observation_calendar.py — Gemini queue observation records     │
│   • sync_soar_observation_calendar.py — SOAR queue observation records         │
│   • backfill_lco_observations.py — Backfill LCO records for past runs          │
│   • reconcile_campaign_runs.py — Batch reconciliation of all runs              │
│   • fetch_jplsbdb_objects.py — Target ingestion from JPL                       │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
┌────────────────────────────────┴───────────────────────────────────────────────┐
│                           Model / Data Layer                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│ SOLSYS MODELS (`solsys_code/models.py`):                                        │
│   • CampaignRun — single observing run within a coordination campaign           │
│   • CalendarEventMeta — companion record: telescope-label verification +        │
│     attribution to CampaignRun + observation record/group links                 │
│   • CalendarEventDismissal — audit trail for event dismissals                  │
│   • ObservationRecordDismissal — audit trail for record dismissals              │
│   • CampaignRunObservation — observatory-level detail for multi-site runs      │
│                                                                                 │
│ OBSERVATORY MODELS:                                                            │
│   • Observatory (solsys_code_observatory/models.py) — MPC obscode + geodetic   │
│     position + coordinate conversion methods                                    │
│                                                                                 │
│ TOM MODELS (external, wrapped/extended):                                       │
│   • CalendarEvent (tom_calendar) — observatory event record                     │
│   • Target (tom_targets) — non-sidereal Solar System targets                    │
│   • TargetList (tom_targets) — campaign container                              │
│   • ObservationRecord (tom_observations) — submitted observation                │
│   • ObservationGroup (tom_observations) — grouped observation records           │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
┌────────────────────────────────┴───────────────────────────────────────────────┐
│                    External Services & Integrations                            │
├────────────────────────────────────────────────────────────────────────────────┤
│ ASTRONOMY LIBRARIES:                                                           │
│   • astropy — Sun position, AltAz coordinates, time conversion                 │
│   • REBOUND + ASSIST — N-body orbital integration                              │
│   • Sorcha — Solar System ephemeris simulation                                 │
│   • ERFA/SOFA — Coordinate geometry, Earth orientation                         │
│   • spiceypy — SPICE kernel access (caches ~1.6GB at module load)             │
│                                                                                 │
│ FACILITY QUEUES (read + reconcile):                                            │
│   • LCO API — queue scheduling portal read-back + live reconciliation          │
│   • Gemini API — queue scheduling portal read-back                             │
│   • SOAR API — queue scheduling portal read-back + live reconciliation         │
│                                                                                 │
│ DATA SERVICES:                                                                 │
│   • Fink alert stream (`tom_fink`)                                             │
│   • JPL Scout / Horizons API (external)                                        │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File(s) |
|-----------|----------------|---------|
| Ephemeris View | HTTP handler for ephemeris request + form validation | `solsys_code/views.py:MakeEphemerisView`, `Ephemeris` |
| Ephemeris Math | Orbital integration, coordinate transforms, magnitude/rates | `solsys_code/ephem_utils.py` |
| Campaign Table | Per-campaign run display (sortable, filterable, paginated) | `solsys_code/campaign_views.py:CampaignRunTableView` |
| Campaign Submission | Public intake form + site resolution attempt | `solsys_code/campaign_views.py:CampaignRunSubmissionView` |
| Approval Queue | Staff-only pending-review worklist | `solsys_code/campaign_views.py:ApprovalQueueView` |
| Attribution Queue | Orphan calendar events + runs awaiting link-decision | `solsys_code/campaign_views.py:AttributionQueueView` |
| Attribution Decision | Confirm/dismiss/undo event-run link (audit trail) | `solsys_code/campaign_views.py:AttributionDecisionView` |
| Reconciler | Idempotent run→calendar-event projection with namespace | `solsys_code/campaign_reconciler.py:reconcile_run()` |
| Calendar Sync | Insert-or-create calendar events from observation records | `solsys_code/calendar_utils.py:insert_or_create_calendar_event()` |
| Site Resolution | MPC obscode lookup, fallback guessing, confidence scoring | `solsys_code/campaign_utils.py:resolve_site()` |
| Sun Events | Sunset/sunrise/dark-window times (astropy + zoneinfo) | `solsys_code/telescope_runs.py:sun_event()` |
| Run Parser | Classical-schedule line parsing (DSL → ParsedRun) | `solsys_code/telescope_runs.py:parse_run_line()` |
| Observatory Model | MPC site lookup + coordinate conversions | `solsys_code/solsys_code_observatory/models.py:Observatory` |
| Calendar Rendering | Month-cell prefetch + campaign decoration injection | `solsys_code/views.py:fomo_render_calendar()` |
| CampaignRun Model | Run lifecycle, approval/run status, source tracking | `solsys_code/models.py:CampaignRun` |
| CalendarEventMeta Model | Telescope-label metadata + run attribution + obs links | `solsys_code/models.py:CalendarEventMeta` |
| JPL Ingest | SBDB query + Target creation (asteroids vs. comets) | `solsys_code/views.py:JPLSBDBQuery` |

## Pattern Overview

**Overall:** Multi-layer Plugin Architecture

**Key Characteristics:**

- **Plugin Substrate** — Extends TOM Toolkit via Django app hooks (`apps.py:target_detail_buttons()`, `nav_items()`, `data_services()`)
- **Idempotent Reconciliation** — `campaign_reconciler.reconcile_run()` is the single source of truth for run→event projection; called by both batch commands and live staff actions to guarantee consistency
- **Separation of Heavy/Light Modules** — Core ephemeris module (`ephem_utils.py`) is intentionally isolated; views defer importing it until request time to avoid ~1.6 GB SPICE kernel download at startup
- **No Circular Imports** — `campaign_*` modules deliberately avoid importing `views.py` or `ephem_utils.py`
- **Multi-Source Event Ingestion** — Calendar events come from classical files, LCO queue, Gemini queue, SOAR queue, and manual submission; all converge on `insert_or_create_calendar_event()` with idempotent tolerance windows
- **Namespace-Based Event Ownership** — Reconciler owns events keyed `RUN:{run_pk}` (class-wide/satellite) or `RUN:{run_pk}:{date}` (classical/ground-site); attribution via `CalendarEventMeta.run` is orthogonal and read-only from reconciler's perspective (Phase 33)
- **Attribute-Based Access Control** — PII fields (`contact_person`, `contact_email`) gated at SQL SELECT via `Case/When` annotations, not template conditionals (D-13/VIEW-05)

## Layers

**Presentation Layer:**

- **Purpose:** Render user-facing HTML/forms
- **Location:** `src/templates/`, `src/templatetags/` (project-level), plus per-app overrides
- **Contains:** Django templates for ephemeris form, campaign tables, calendar month view, attribution queue, admin customizations
- **Depends on:** Django context from views, Crispy Forms, django-tables2 renderers
- **Used by:** Django template rendering system
- **Key Pattern:** Minimal logic in templates; heavy lifting in views/templatetags

**View Handler Layer:**

- **Purpose:** HTTP request/response orchestration, form validation, queryset optimization
- **Location:** `solsys_code/views.py`, `solsys_code/campaign_views.py`, `solsys_code/solsys_code_observatory/views.py`
- **Contains:** Django class-based views (FormView, ListView, TemplateView, FilterView), context assembly, prefetch/select_related optimization (DISPLAY-09 pattern)
- **Depends on:** Forms, models, business logic modules, TOM views
- **Used by:** URL dispatcher
- **Key Pattern:** Queryset optimization via prefetch_related/select_related; non-staff PII gating at SQL SELECT; table view suppresses default sort to preserve manual ordering

**Form & Filtering Layer:**

- **Purpose:** Input validation, form rendering, query filtering
- **Location:** `solsys_code/forms.py`, `solsys_code/campaign_forms.py`, `solsys_code/campaign_filters.py`, `solsys_code/solsys_code_observatory/forms.py`
- **Contains:** Crispy Forms layouts, Crispy Bootstrap fields, django-filters FilterSets, form.clean() validators
- **Depends on:** Models, Crispy Forms, django-filters
- **Used by:** Views for initialization/validation and template rendering
- **Key Pattern:** Timezone selection via Crispy widgets; site resolution UI with HTMX search fragment

**Business Logic Layer — Orchestration:**

- **Purpose:** Coordinate complex multi-step operations (campaign approval, event reconciliation, site resolution)
- **Location:** `solsys_code/campaign_views.py`, `solsys_code/campaign_attribution.py`, `solsys_code/campaign_reconciler.py`, `solsys_code/campaign_utils.py`
- **Contains:** State transitions (approval/run status updates), attribution logic, reconciliation driver, site lookup/fallback, link management
- **Depends on:** Models, ORM queries, pure-logic utility modules
- **Used by:** View handlers, management commands
- **Key Pattern:** Idempotent functions with audit trail; transaction wrapping; no side effects (function composition)

**Pure Logic Layer — Scientific/Utility:**

- **Purpose:** Stateless computation (astronomical calculations, text parsing, algorithm)
- **Location:** `solsys_code/telescope_runs.py`, `solsys_code/campaign_gap.py`, `solsys_code/ephem_utils.py`, `solsys_code/calendar_utils.py`
- **Contains:** Sun-event calculation, classical-run parsing, coverage-gap analysis, coordinate transforms, n-body integration setup
- **Depends on:** Astropy, external astronomy libraries, stdlib
- **Used by:** Views, orchestration layer, management commands
- **Key Pattern:** Stateless; composition over classes; returns dataclasses/tuples; no ORM access except calendar_utils (which is a thin adapter)

**Data Access Layer — Models & ORM:**

- **Purpose:** Define data schema and query interface
- **Location:** `solsys_code/models.py`, `solsys_code/solsys_code_observatory/models.py`, `solsys_code/migrations/`
- **Contains:** Django ORM models (CampaignRun, CalendarEventMeta, Observatory) with Meta constraints, properties, __str__ methods
- **Depends on:** Django ORM, external TOM Toolkit models
- **Used by:** Views, forms, business logic, management commands
- **Key Pattern:** One-way ForeignKey (PROTECT/SET_NULL); UniqueConstraint for get_or_create race-safety; CheckConstraint for invariants

**Management Command Layer:**

- **Purpose:** Offline batch operations and data ingestion
- **Location:** `solsys_code/management/commands/`
- **Contains:** Classical-run ingest, queue synchronization, backfill, reconciliation, JPL discovery
- **Depends on:** Models, pure-logic modules, external APIs
- **Used by:** Operators via `python manage.py <command>`
- **Key Pattern:** Tolerance windows for matching (start_time ±5min); counter dict for summary; --dry-run flag support

**External Integration Layer:**

- **Purpose:** Bridge to external services
- **Location:** Hidden in business logic & utility modules; explicit in management commands
- **Contains:** Facility API calls (LCO, Gemini, SOAR), JPL SBDB queries, Horizons API integration, SPICE kernel caching
- **Depends on:** requests, tom_observations facilities, astropy, spiceypy
- **Used by:** Orchestration, utilities, commands
- **Key Pattern:** Explicit timeout configuration; graceful degradation (fallback labels); error categorization (extraction_failed vs. skipped)

## Data Flow

### Primary Flow: Campaign Submission → Approval → Reconciliation → Calendar Events

1. **Submission** (user-facing)
   - POST to `CampaignRunSubmissionView` (`solsys_code/campaign_views.py:390-600`)
   - Form validates dates, resolves site via `resolve_site()` (`campaign_utils.py:120-180`)
   - CampaignRun created with `approval_status=PENDING_REVIEW`, `source=WEB`
   - Redirect to submission-thanks template

2. **Approval Queue** (staff-only)
   - GET `ApprovalQueueView` (`campaign_views.py:900-950`)
   - Displays pending CampaignRuns awaiting approval
   - Staff POST via `CampaignRunDecisionView` to approve/reject
   - Approval sets `approval_status=APPROVED` or `REJECTED`

3. **Reconciliation** (batch or live)
   - Called by `reconcile_campaign_runs.py` command or live-action handlers
   - `reconcile_run(run)` (`campaign_reconciler.py:400-500`) generates events:
     - Container key `RUN:{run_pk}` for class-wide/satellite runs
     - Per-night keys `RUN:{run_pk}:{date}` for classical/ground-site runs
   - Events inserted via `insert_or_create_calendar_event()` with ±5min tolerance
   - Returns `ReconcileResult` with created/updated/unchanged/blocked counts

4. **Calendar Display** (user-facing read)
   - GET calendar month via `fomo_render_calendar()` (`views.py:60-150`)
   - Prefetch CalendarEventMeta + select_related run.campaign (DISPLAY-09)
   - Inject campaign-decoration marker for attributed events
   - Render month grid with event popups

### Secondary Flow: Observation Record Sync → Calendar → Attribution Queue

1. **Queue Sync** (batch command)
   - `sync_lco_observation_calendar.py` / `sync_gemini_observation_calendar.py`
   - Fetch ObservationRecord list via facility API
   - Extract telescope label, instrument, window via `calendar_utils.py` helpers
   - Create CalendarEvent via `insert_or_create_calendar_event()` with `is_verified=False`
   - Store observation link in `CalendarEventMeta.observation_record` / `.observation_group`

2. **Attribution Queue** (staff-only)
   - GET `AttributionQueueView` (`campaign_views.py:1200-1400`)
   - Show orphan CalendarEvents (unattributed) + orphan CampaignRuns (no events)
   - Staff confirms link via `AttributionDecisionView` POST
   - Link written to `CalendarEventMeta.run`, audit fields set (`confirmed_by`, `confirmed_at`)

3. **Undo Path** (staff-only)
   - POST to `AttributionDecisionView` with action='undo'
   - Clears `CalendarEventMeta.run` only; event survives (SET_NULL)
   - Event returns to orphan queue for re-attribution

### Tertiary Flow: Classical Run Ingest

1. **File Parsing** (batch command)
   - `load_telescope_runs.py` reads ESO classical schedule text
   - `parse_run_line()` (`telescope_runs.py:850-950`) tokenizes each line
   - Extract: telescope, start-date, end-date, status, run-name

2. **Night Enumeration**
   - `_iter_run_nights()` expands date range to per-night CalendarEvents
   - For each night: compute UTC sunset/sunrise via `sun_event()` (`telescope_runs.py:350-400`)
   - Resolve timezone via `get_site()` → Observatory lookup by MPC obscode
   - Handles ESO noon-to-noon convention (E-S nights vs. E-S+1)

3. **Event Creation**
   - `insert_or_create_calendar_event()` creates event per night
   - Title format: `[CANCELLED]? {telescope} {instrument}`
   - Key: `RUN:{run_pk}:{date}` (classical family)
   - CalendarEventMeta created with `is_verified=True` (classical runs are trusted input)

### Quaternary Flow: Gap Analysis

1. **Query Composition** (form submission)
   - User selects campaign + date range
   - Form validates via `CampaignGapAnalysisForm`
   - GET `CampaignGapAnalysisView` with params in query string

2. **Gap Computation** (cached at campaign level)
   - `get_or_compute_gap()` (`campaign_gap.py`) filters CampaignRuns:
     - `approval_status=APPROVED`
     - `run_status` not in {CANCELLED, WEATHER_TECH_FAILURE, NOT_AWARDED}
     - window_start/window_end overlap with query range
   - For each Target in campaign, compute ephemeris time slots
   - Identify nights with no scheduled observation

3. **Result Display**
   - Render table: Target | Site(s) | Gap Nights | Coverage %
   - Link to CampaignRunTableView for adding new runs

**State Management:**

- **Request-local:** Form data, context dict passed between view methods
- **Persistent:** CampaignRun, CalendarEvent, CalendarEventMeta, Observatory models (Django ORM)
- **Module-level (HEAVY):** Sorcha `ephem` object, SPICE kernels cached in `~/.cache/sorcha/` (loaded once at `ephem_utils.py` import)
- **Per-command:** Counter dict in management command handle() method

## Key Abstractions

**ReconcileResult Named Tuple:**

- **Purpose:** Encapsulates reconciliation outcome (created/updated/unchanged/blocked counts + skip reason)
- **Examples:** `solsys_code/campaign_reconciler.py:100-120`
- **Pattern:** Return value for idempotent function; used by both batch command and live actions to report results consistently

**ParsedRun Data Class:**

- **Purpose:** Intermediate representation of a parsed classical-run line
- **Examples:** `solsys_code/telescope_runs.py:950-1000` (contains: telescope name, start date, end date, status)
- **Pattern:** Validation bridge between free-text input and structured CampaignRun creation

**Observatory Coordinate Conversions:**

- **Purpose:** Transform between geodetic (lat/lon/alt), geocentric XYZ, and MPC parallax constants
- **Examples:** `solsys_code/solsys_code_observatory/models.py:Observatory.to_parallax_constants()`, `.to_geocentric()`
- **Pattern:** Method on model; lazy-computed properties; used by n-body integration setup

**CampaignRun.is_publicly_visible Property:**

- **Purpose:** Centralized source of truth for approval-status visibility rule (PENDING_REVIEW hidden from non-staff)
- **Examples:** `solsys_code/models.py:CampaignRun:318-330`
- **Pattern:** Property that mirrors queryset-level exclude() discipline (D-10); prevents template drift

**Telescope-Label Fallback Chain:**

- **Purpose:** Attempt live LCO API verification, fall back to SITE_TELESCOPE_MAP lookup, fall back to coarse label
- **Examples:** `solsys_code/calendar_utils.py:derive_telescope()`, `coarse_telescope_label()`
- **Pattern:** Staged degradation; confidence tracked in `CalendarEventMeta.is_verified`

## Entry Points

**Web Application:**

- Location: `src/fomo/wsgi.py`
- Triggers: Web server (runserver, gunicorn, uWSGI)
- Responsibilities: Create Django WSGI application via `get_wsgi_application()`

**ASGI (Async Support):**

- Location: `src/fomo/asgi.py`
- Triggers: ASGI server (Daphne, Hypercorn)
- Responsibilities: Create Django ASGI application for async views

**URL Dispatcher:**

- Location: `src/fomo/urls.py`
- Routes: Project-level namespace hierarchy
  - `/observatory/` → `solsys_code_observatory` app URLs
  - `/calendar/` → `solsys_code` calendar URLs (shadows TOM's calendar with prefetch optimization)
  - `/campaigns/` → `solsys_code` campaign URLs (VIEW-01 positioned before tom_common)
  - `/ephem/<int:pk>/` → Ephemeris result display
  - `/targets/<int:pk>/makeephem/` → Ephemeris form (TOM target detail button integration)
  - `/alerts/` → TOM alerts namespace
  - `''` → TOM common URLs (default)

**Management Commands:**

- Location: `solsys_code/management/commands/`
- Triggered by: `python manage.py <command_name> [options]`
- Examples:
  - `load_telescope_runs --site NTT --filepath <file>` — Classical run ingest
  - `sync_lco_observation_calendar --dry-run` — LCO queue sync
  - `reconcile_campaign_runs --campaign-pk 123` — Batch reconciliation
  - `fetch_jplsbdb_objects --orbital_constraints "e>=1.2"` — JPL discovery

**App Config Integration Hooks:**

- Location: `solsys_code/apps.py:SolsysCodeConfig`
- Methods:
  - `target_detail_buttons()` → Injects "Make Ephemeris" button into TOM target detail
  - `nav_items()` → Injects "Campaigns" link into TOM navbar (position: left)
  - `data_services()` → Registers Fink alert stream as data service

## Architectural Constraints

- **Threading:** Django is single-threaded at request level. Ephemeris computation is synchronous and blocks request handling (ASSIST n-body integration can be slow for large date ranges). No background worker queue (Celery-like) in scope.

- **Global State — HEAVY IMPORT:** Importing `solsys_code.ephem_utils` (transitively via `solsys_code.views`) triggers `fomo_furnish_spiceypy()` at module load, which downloads ~1.6 GB of SPICE kernels to `~/.cache/sorcha/` on first use. This is intentional and unavoidable for ephemeris computation, but forces deliberate separation: `campaign_*` modules avoid importing `views.py` to keep command/form startup fast.

- **Database:** SQLite3 by default (dev); production deployments can use PostgreSQL. Concurrent writes are limited on SQLite.

- **Coordinate Frames:** All ephemeris computations assume J2000 equatorial; ecliptic transforms are done in-function.

- **Observatory Selection:** Form restricts to observatories with `altitude > 0` (validates server-side; no underwater sites).

- **Unique Constraints:** Multiple partial UniqueConstraints on CampaignRun enforce invariants:
  - `(campaign, telescope_instrument, window_start, window_end)` for resolved windows
  - `(campaign, telescope_instrument, contact_person)` for TBD (window_start/window_end both NULL)
  - `(source_identifier,)` for adapter-sourced rows (source != WEB/CSV_IMPORT)

- **Permanent Campaign-Less State:** Queue/classical-file-sourced runs may permanently carry `campaign=None` (CANON-02, D-05/D-06). `NO_CAMPAIGN_LABEL = '(no campaign)'` is the display substitute. Every reader that dereferences `run.campaign.name` must guard on `campaign_id is None` first.

- **Permanent Telescope-Class State:** A run with `telescope_class` set (e.g., '2m0') has `site=None` by design (class-wide allocation). The class is NEVER cleared even if a different site later resolves elsewhere (27-REVIEW-FIX.md CR-01 rejection).

## Anti-Patterns

### Silent Fallback in Site Resolution

**What happens:** `resolve_site()` returns None if obscode lookup fails; caller may then dereference `.site.short_name` without null-check. At form-render time, `is_placeholder_observatory()` pre-screens for fallback sites; at async write time (e.g., HTMX search fragment), NULL site is possible.

**Why it's wrong:** Deferred null-checks can crash at render time; inconsistent null handling between form and async paths invites bugs.

**Do this instead:** Explicit null-checks in all callers; distinguish between "site lookup in progress" (form state) and "site lookup failed" (form error). Consider a `SiteResolutionError` exception for hard failures.

### Hardcoded Observatory Codes in Fixtures

**What happens:** Tests reference Observatory.obscode directly (e.g., `Observatory.objects.get(obscode='268')`); if the fixture is missing or the constant changes, the test silently fails to create its dependency.

**Why it's wrong:** Test setup is fragile; missing fixture shows up as a cascade of downstream errors, not a clear "Observatory does not exist" message.

**Do this instead:** Use factory methods (e.g., `tom_targets.tests.factories.ObservatoryFactory`) or explicit fixture setup with assertion (`assertIsNotNone(Observatory.objects.get(obscode='268'))`).

### Form Initialization with Missing Timezone Fallback

**What happens:** `EphemerisForm.initial['timezone']` is hardcoded to 'UTC'; if a user's browser or server zone differs, the form displays wrong default. No server-side timezone auto-detection for the logged-in user.

**Why it's wrong:** User experience degrades silently; staff cannot assume their default is the same as a colleague's.

**Do this instead:** Detect request.user's timezone preference (if stored in UserProfile) or use browser's `Intl.DateTimeFormat().resolvedOptions().timeZone` + HTMX fetch to update default.

### N+1 Query Pattern — CalendarEventMeta OneToOne Reverse

**What happens:** Iterating over CalendarEvents and dereferencing `.telescope_label_meta` causes one query per event in the calendar month view.

**Why it's wrong:** Month views with 30+ events trigger 30+ extra queries; page render slows to O(N).

**Do this instead:** Use `prefetch_related(Prefetch('telescope_label_meta', queryset=CalendarEventMeta.objects.select_related('run__campaign')))` as in `fomo_render_calendar()` (DISPLAY-09 pattern); reduces to one prefetch query + one select_related.

### CampaignRun.__str__ Includes PII Without Gating

**What happens:** `__str__` is rendered in admin changelist, autocomplete JSON endpoints, and delete-confirmation pages for all users. Contact fields are NOT included (intentional), but the label shows campaign name + telescope + site.

**Why it's wrong:** In future milestones if contact fields are added to `__str__`, they will leak to non-staff users via admin autocomplete.

**Do this instead:** Comments in `__str__` docstring (already present, `models.py:383-397`) explain the PII discipline; never add contact fields to the label. Use a separate method `admin_str()` if admin-only context is needed.

## Error Handling

**Strategy:** Defensive, layered, with audit trail

**Patterns:**

- **Form Validation:** `clean()` methods raise `ValidationError` for user-facing messages; async HTMX endpoints return 400 BadRequest for malformed queries.
- **Site Resolution Failure:** Form resets site to None, sets `site_needs_review=True`, appends validation error; user sees "Could not automatically resolve site" message + staff worklist reminder.
- **Observation Record Extraction:** `extract_instrument()` raises `InstrumentExtractionError` (caught separately) for fully-malformed records; routed to 'extraction_failed' counter, not merged into 'skipped'.
- **Reconciliation Invariant Violation:** `reconcile_run()` raises `Exception` if `window_start/window_end` are partial (violates CheckConstraint); message describes the invariant.
- **API Timeout/Failure:** Facility sync commands log at error level, increment 'telescope_api_failed' counter, continue processing next record (graceful degradation).
- **External Process (SPICE/ASSIST):** Ephemeris computation may raise AttributeError or ValueError; caught at view level and rendered as "Computation failed" error page.

## Cross-Cutting Concerns

**Logging:**

- Django logging via `logger = logging.getLogger(__name__)`
- Levels: DEBUG for diagnostic info (e.g., "Resolving site FOO-1m0"), INFO for milestones (e.g., "Reconciled run 123: created 5, updated 2"), ERROR for failures
- Example: `logger.debug(f'Query failed with status {resp.status_code}')`

**Validation:**

- Model-level: Django `clean()` methods + Meta.constraints (UniqueConstraint, CheckConstraint)
- Form-level: Crispy Forms field validators + custom `clean()` methods
- View-level: Explicit null-checks and type guards (e.g., `if run.campaign_id is None`)

**Authentication & Authorization:**

- TOM Toolkit integration: `django.contrib.auth` + `django_guardian` object-level permissions
- `StaffRequiredMixin` gatekeeper for approval/attribution views
- Queryset-level PII gating via `Case/When` annotations (non-staff never SELECT contact fields)

**Pagination & Performance:**

- Table views: `django-tables2` with `table_pagination={'per_page': 25}`
- Large querysets: Use `select_related()` for ForeignKey traversal, `prefetch_related()` for reverse Many-to-One
- SQL-level sorting via `F()` expressions (e.g., `order_by(F('window_start').desc(nulls_last=True))`) for portable NULL ordering

**Caching:**

- Gap analysis: Cached at campaign level via `get_or_compute_gap()` memoization (no persistent cache backend; in-memory only)
- SPICE kernels: Cached at `~/.cache/sorcha/` by Sorcha library (downloaded once, re-used across requests)

---

*Architecture analysis: 2026-09-04*
