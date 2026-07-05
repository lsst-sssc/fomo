# Phase 14: Campaign Data Model & Bootstrap Import - Research

**Researched:** 2026-07-02
**Domain:** Django model design (two-field status vocabulary), tiered external-lookup resolution
(local DB → external API → placeholder), CSV ingestion of messy real-world spreadsheet data,
PII-safe demo-notebook fixtures — all within the existing `solsys_code` app.
**Confidence:** HIGH — this phase is almost entirely internal-pattern-driven (direct precedent
already exists in this codebase for every sub-problem) plus one grounding pass against the real
3I/ATLAS sheet's actual column headers and data shapes.

## Summary

Phase 14 has no new external dependencies and no genuinely novel technical risk — every piece
(status-choices field, tiered site resolution, CSV-to-model import with skip-and-log, paired demo
notebook) has a direct precedent already committed in `solsys_code/`. The real work is disciplined
adaptation of those precedents, plus handling the real 3I/ATLAS sheet's messiness, which this
research fetched and inspected directly (column headers + non-PII format patterns for all 24 rows,
without pulling real names/emails into this document — see "Real 3I/ATLAS Sheet — Verified Shape"
below).

The single most consequential finding: **the sheet already has a dedicated "Site Code" column, and
22 of 24 real rows populate it with values that look exactly like MPC observatory codes** (`F65`,
`309`, `675`, `705`, `568`, `X09`, `X07`, `C65`, `N50`, `500@-170`). This means D-08's "3-tier site
resolution" is *not* a free-text-to-site fuzzy-match problem — it's a direct `obscode` lookup
against `Observatory`, then `MPCObscodeFetcher.query(obscode)`, exactly mirroring the existing
`CreateObservatory.form_valid()` code path. The free-text facility names CONTEXT.md's examples cite
("Palomar P200", "VLT/MUSE") live in the separate `Telescope / Instrument` column and are *not* what
gets resolved against `Observatory` — they're stored as-is in a `telescope_instrument` field.

**Primary recommendation:** Follow `load_telescope_runs.py`'s command shape (per-row try/except,
stdout/stderr summary counters) verbatim, adapted from lines to `csv.DictReader` rows; resolve site
via `Observatory.objects.get(obscode=row['Site Code'])` → `MPCObscodeFetcher` → placeholder
`Observatory` (same three tiers `CreateObservatory.form_valid()` already implements, minus the
Django form layer); use `django.db.models.TextChoices` for both status fields; store the CSV's raw
`Site Code`, `Obs. Date`, and `UT Time Range` strings verbatim in `*_raw` fields alongside every
parsed/resolved field, mirroring `CalendarEventTelescopeLabel.is_verified`'s "flag, don't silently
guess" pattern.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `CampaignRun` schema & persistence | Database/Storage | API/Backend (Django ORM) | New Django model + first-ever `solsys_code` migration beyond `CalendarEventTelescopeLabel`; pure data layer, no view built this phase |
| Two-field status vocabulary (`approval_status`/`run_status`) | API/Backend | Database/Storage | Enforced via `TextChoices` + `CharField(choices=...)` at the model layer before any DB write |
| CSV bootstrap import (`import_campaign_csv` management command) | API/Backend | — | A `BaseCommand`, backend-only; no HTTP request path, invoked by the operator via CLI |
| Site resolution (Observatory lookup → MPC API → placeholder) | API/Backend | External Service (MPC Obscodes API) | Backend orchestrates a 3-tier lookup; tier 2 makes one outbound HTTPS GET to `data.minorplanetcenter.net` |
| PII-safe demo notebook + synthetic fixture | API/Backend (test/demo harness) | — | Notebook exercises the backend command against a local, network-free fixture — no browser/CDN tier involved this phase |

## User Constraints (from CONTEXT.md)

<user_constraints>

### Locked Decisions

- **D-01:** CAMP-03's status applies **per-`CampaignRun`** (one telescope run), not per-campaign —
  `TargetList` (the campaign) has no status field in this milestone.
- **D-02:** Split into **two fields**: `approval_status` (`pending_review`, `approved`, `rejected`)
  and `run_status` (`requested` → `planned` → `observed` → `reduced` → `published`, plus
  `cancelled`, `not_awarded`, `weather_tech_failure` — 8 values total).
- **D-03:** Bootstrap-imported rows get `approval_status='approved'` (vetted historical backfill,
  not fresh submissions). The `pending_review` → `approved`/`rejected` lifecycle is demonstrated in
  the demo notebook using synthetic data, not the real import.
- **D-04:** Natural key for create-or-update idempotency: **(campaign, telescope, obs date/UT
  start)**. Mirrors `CalendarEvent`'s existing (telescope, instrument, start_time) find-or-create
  key pattern.
- **D-05:** Row-skip granularity: only a failure in a **natural-key field** (campaign, telescope,
  obs date/UT start) skips the whole row. A malformed **non-key** field nulls just that column —
  the row still imports.
- **D-06:** The import command takes a **required `--campaign` CLI argument**; the `TargetList` is
  found-or-created by name. One CSV run = one campaign.
- **D-07:** `CampaignRun.target` is **auto-resolved**: if the campaign `TargetList` has exactly one
  `Target`, every imported row gets that `Target` assigned automatically.
- **D-08:** `CampaignRun.site` uses **3-tier resolution**: (1) match against existing `Observatory`
  records, (2) query the MPC Obscodes API and create an `Observatory` row if found, (3) create a
  **placeholder** `Observatory` row and flag for review.
- **D-09:** A failed/partial site resolution must not skip the row (site is not part of the natural
  key). Storage shape: `CampaignRun.site` (FK, nullable) + `CampaignRun.site_raw` (text, preserves
  original sheet string) + `CampaignRun.site_needs_review` (bool) — mirrors
  `CalendarEventTelescopeLabel.is_verified`.
- **D-10:** Demo-notebook fixture is a small hand-built synthetic CSV, same column shape as the real
  sheet, obviously-fake contact info (e.g. `test@example.com`), ~5-10 rows covering field variety.
- **D-11:** Fixture lives at `docs/notebooks/pre_executed/fixtures/` (new subdirectory — first of
  its kind). Fixture rows use **only sites already seeded in the local `Observatory` table** — the
  notebook does not make a live MPC API call. Tier-2/tier-3 resolution logic is covered by the
  Django test suite with mocked API responses instead.

### Claude's Discretion

- Exact management command name — not locked, follow the `load_telescope_runs`/
  `fetch_jplsbdb_objects` naming convention (this research recommends `import_campaign_csv`).
- Exact CSV column-name-to-model-field mapping and date/time parsing strategy for the real sheet's
  free-text columns — this research provides the verified real column headers and format patterns
  (see "Real 3I/ATLAS Sheet — Verified Shape" below) plus a recommended mapping.
- Whether `site_needs_review` rows get a distinct counter in the command's created/updated/skipped
  summary output — recommended (see Code Examples), following the Phase 7 `[UNVERIFIED]`-style
  counter precedent.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within Phase 14's scope this session. Phase 15 (per-campaign table view),
Phase 16 (submission form/approval queue UI/calendar projection), and Phase 17 (coverage-gap
analysis) are separate phases, not deferred ideas within this one.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAMP-01 | `CampaignRun` stores campaign `TargetList` (required FK), optional observed `Target`, full 3I-sheet field inventory | Verified real column headers (see below); recommended field list in "Standard Stack"/"Code Examples"; `TargetList`/`Target`/`Observatory` FK shapes confirmed via live model introspection |
| CAMP-02 | Optional `Target` FK; single-target campaigns work without it | D-07 auto-resolve logic + `TargetList.targets` M2M confirmed (`tom_targets.models.TargetList_targets` through table); nullable `SET_NULL` FK recommended |
| CAMP-03 | Two-field status (`approval_status`/`run_status`), 8 `run_status` values | `django.db.models.TextChoices` pattern confirmed against official Django docs; concrete class layout in Code Examples |
| CAMP-04 | Management command imports real 3I/ATLAS CSV; created/updated/skipped summary; skip-and-log unparseable rows | `load_telescope_runs.py` structure adapted line→CSV-row; `csv.DictReader` skip-and-log pattern confirmed; real CSV messiness documented below with concrete parsing recommendations |
| CAMP-05 | Paired demo notebook runs against synthetic/redacted fixture, no real PII in git history | `load_telescope_runs_demo.ipynb`/`telescope_runs_demo.ipynb` structure documented as template; `docs/notebooks/pre_executed/fixtures/` confirmed not excluded by `.gitignore` |
</phase_requirements>

## Real 3I/ATLAS Sheet — Verified Shape

Fetched directly from the CSV export of the sheet named in CONTEXT.md/seed doc
(`docs.google.com/spreadsheets/d/1INhxLWlHoa-JkW-uKRzmSyms06wI80wEXTqBJSR3YAI`). **Contact
Person/Email values were deliberately excluded from this document** — only column headers, row
counts, and non-PII column values are reproduced below. `[VERIFIED: real 3I/ATLAS sheet export, 2026-07-02]`

**Exact column headers (14 columns, in order):**
`Contact Person, Email, Telescope / Instrument, Site Code, Obs. Date, UT Time Range,
Filter(s)/Bandpass, Observation Details, Weather conditions or forecast, Observation Status,
Observation Outcome, Publication Plans, Open to collaboration?, Other comments`

**24 total data rows.** Non-PII columns, verbatim (this table is safe to commit — no names/emails):

| Telescope / Instrument | Site Code | Obs. Date | UT Time Range | Filter(s)/Bandpass | Observation Status |
|---|---|---|---|---|---|
| FTN/MusCAT3 | F65 | 2025-07-04 | 08:50 - 11:50 | griz | completed |
| ESO VLT FORS2 | 309 | 2025-07-04 | 06:50 - 07:15 | R | completed |
| Deep Random Survey / 43cm | X09 | 2025-07-04 | 04:11 - 10:00 | Lum | completed |
| HCT | N50 | 2025-07-06 | 17:45 - 18;55 *(typo: semicolon not colon)* | r | completed |
| Palomar P200/NGPS | 675 | 2025-07-03 | free-text garbage ("...is for the coordination...") | g/r/i | completed |
| Palomar P200/NGPS | 675 | 2025-07-03 | `~7:00:00 AM` | R/I | completed |
| Apache Point Observatory/ARCTIC | 705 | 2025-07-06 | *(blank)* | g/r/i/z | completed |
| Apache Point Observatory/KOSMOS | 705 | 2025-07-06 | *(blank)* | Blue GRISM 3800-6600 Å | completed |
| HST STIS/COS | 250 | 2025-11-27 | `2025-11-27 to 2025-12-10` *(a date range, in the UT-time column)* | G160M/G230L | Upcoming |
| Swift/UVOT | *(blank)* | 2025-07-11 | `2025-07-11 to 2025-07-13` | UVW1/V | completed |
| VLT/MUSE | 309 | 2025-07-03 | `~ 1am` | 480-930 nm | completed |
| VLT/MUSE | 309 | 2025-07-16 | `~1 am` | 480-930 nm | completed |
| Deep Sky Chile at Rio Hurtado Valley | X07 | 2025-07-03 | `5 UTC` | *(blank)* | completed |
| Telescope Joan Oró, Montsec, Catalonia | C65 | 2025-07-05 | `1 UTC` | V,R,I | completed |
| VLT/MUSE | 309 | 2025-07-29 | `~1 am` | 480-930 nm | completed |
| VLT/MUSE | 309 | 2025-08-10 | `~1 am` | 480-930 nm | Upcoming |
| VLT/UVES | 309 | 2025-08-11 | `2025-08-11 to 2025-08-19` | 300 to 700nm | Upcoming |
| JWST | 500@-170 *(spacecraft-style MPC designation, 8 chars)* | 2025-08-06 | 11:01 - 11:20 | NIRSpec Prism | completed |
| NASA IRTF/SpeX | 568 | 2025-07-03 | 7:42-11:51 | Prism (0.7-2.5 um) | completed |
| NASA IRTF/SpeX | 568 | 2025-07-04 | 7:34-11:40 | Prism (0.7-2.5 um) | completed |
| NASA IRTF/SpeX | 568 | 2025-07-25 | 5:58-9:02 | Prism (0.7-2.5 um) | completed |
| NASA IRTF/SpeX | 568 | 2025-08-05 | 5:39-8:51 | Prism (0.7-2.5 um) | completed |
| JUICE | *(blank)* | `2025-11-02 -25` *(malformed range)* | *(blank)* | UVS, SWI, JANUS, MAJIS, PEP/JENI | *(blank)* |
| LCO 1m | *(blank)* | `2025-07-05 to 2025-09-22` | *(blank)* | g, r, i, z | completed |

Also confirmed: 2 rows have an entirely blank `Obs. Date` with a populated `Telescope/Instrument`
(planned/TBD future observations — not reproduced above to avoid any incidental contact-column
leakage risk in this table).

**What this means for the planner:**

1. **`Site Code` is already MPC-obscode-shaped for 22/24 rows** — direct `obscode` lookup, not
   fuzzy free-text matching. Blank `Site Code` occurs for space-based/mobile facilities (Swift,
   JUICE, generic "LCO 1m" with no specific site) — these correctly fall through to tier 3
   (placeholder + flag), which is exactly what D-08/D-09 already designed for.
2. **`500@-170` (JWST) will not fit `Observatory.obscode` (`max_length=4`).** Both
   `Observatory.objects.get(obscode=...)` and `MPCObscodeFetcher.query(obscode)` — and any
   placeholder-creation attempt using this string as `obscode` — will fail or truncate
   destructively. **Recommendation (flag for planner decision, not locked by CONTEXT.md):** when
   the raw `Site Code` value exceeds `Observatory.obscode`'s max length, skip tiers 1-2 entirely and
   go straight to storing `site_raw` + `site_needs_review=True` with `site` left `null` (do not
   attempt placeholder `Observatory` creation with a truncated/invalid obscode — that would silently
   fabricate a wrong site, violating the "flag, don't silently guess" principle D-08/D-09 already
   establish for the *no-match* case).
3. **`UT Time Range` is the messiest column**: colon typos (`18;55`), approximate times (`~1 am`,
   `~7:00:00 AM`), bare-hour shorthand (`5 UTC`, `1 UTC`), a date range mistakenly in the time
   column, and outright unparseable prose. See "Common Pitfalls" and "Code Examples" below for a
   concrete, permissive parsing strategy plus an explicit fallback for the natural-key implications.
4. **`Observation Status` values in the real data (`completed`, `Upcoming`) don't map 1:1 onto the
   8-value `run_status` vocabulary** — a translation table is needed. See "Code Examples".
5. Only 24 rows total — small enough that a `--dry-run`-style stdout preview (optional; not
   required by CAMP-04) would let the operator manually sanity-check the mapping before committing
   the real import, but is not required to satisfy the locked decisions.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Django (`django.db.models`) | 5.2.14 (already installed, `[VERIFIED: solsys_code/migrations/0001_calendareventtelescopelabel.py header]`) | `TextChoices` status fields, model/migration | Already the project's ORM; no new dependency |
| `csv` (stdlib) | Python 3.10+ stdlib | Row-based CSV parsing with per-row skip-and-log | Matches `load_telescope_runs.py`'s existing per-line try/except shape far more naturally than `pandas`' vectorized/DataFrame model; no new dependency |
| `datetime`/`zoneinfo` (stdlib) | Python 3.10+ stdlib | UTC-aware datetime construction for `ut_start`/`ut_end` | Matches `load_telescope_runs.py`'s `dt_timezone.utc` pattern exactly; `USE_TZ=True`/`TIME_ZONE='UTC'` confirmed in `src/fomo/settings.py` `[VERIFIED: settings.py:158,164]` |
| `re` (stdlib) | Python 3.10+ stdlib | Best-effort UT-time-range regex extraction from free text | No parsing library needed for a handful of known patterns (see Code Examples) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas` | 2.3.1 (already installed transitively via `sorcha`, `[VERIFIED: pip show pandas]`, imported directly today in `ephem_utils.py`/`views.py`) | Available if the planner prefers DataFrame-style CSV inspection for one-off exploration (e.g. a notebook cell listing unique `Observation Status` values) | Not recommended for the *command's* row-by-row skip-and-log logic (see Alternatives below) — fine for ad hoc notebook exploration only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `csv.DictReader` row loop | `pandas.read_csv` + `.iterrows()` | `pandas` normalizes blank cells to `NaN` (not `''`), silently coerces types, and its natural failure mode is a DataFrame-wide parse abort rather than a single bad row — fighting the CAMP-04 "skip only the bad row, keep the rest" requirement (D-05) instead of helping with it. `csv.DictReader` is the closer match to the existing `load_telescope_runs.py` per-line pattern CLAUDE.md/CONTEXT.md explicitly ask to adapt. |
| Manual regex UT-time parsing | `dateutil.parser.parse` (not currently a dependency) | Would add a new third-party dependency for a handful of known real-world formats; regex is enough given the verified format inventory above, and avoids `dateutil`'s tendency to "successfully" parse nonsense strings into wrong-but-plausible dates (a real risk given rows like the free-text-garbage UT Time Range row above) |

**Installation:** None — every recommended tool above is already available in this environment (stdlib or already-installed transitive dependency). No new entries needed in `pyproject.toml`.

## Package Legitimacy Audit

**No new external packages are introduced by this phase.** `csv`, `datetime`, `zoneinfo`, `re` are
Python stdlib; `pandas` is already installed and already imported directly elsewhere in
`solsys_code/` (`ephem_utils.py:14`, `views.py:15`). The Package Legitimacy Gate protocol does not
apply — there is nothing to check against a registry.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
Operator (CLI)
   │  manage.py import_campaign_csv --campaign "3I/ATLAS" path/to/sheet.csv
   ▼
Command.handle()  (solsys_code/management/commands/import_campaign_csv.py)
   │
   ├─ 1. TargetList.objects.get_or_create(name=campaign)         [D-06]
   ├─ 2. Resolve auto-target: campaign.targets.count()==1 ?      [D-07]
   │        target = campaign.targets.first() : target = None
   │
   ▼
   for row in csv.DictReader(open(filepath)):                    [row-by-row, CAMP-04]
       │
       ├─ parse natural-key fields: telescope_instrument,
       │    obs_date, ut_start  ───fail───▶ log to stderr,        [D-04/D-05]
       │                                    skipped_count += 1, continue
       │  (success)
       │
       ├─ resolve site (3 tiers, D-08/D-09):
       │    Observatory.objects.get(obscode=site_code)  ──miss──▶
       │    MPCObscodeFetcher().query(site_code) → .to_observatory() ──miss/error──▶
       │    Observatory placeholder (obscode=site_code, name="NEEDS REVIEW: <raw>",
       │                             site_needs_review=True)
       │  (site resolution NEVER skips the row — D-09)
       │
       ├─ parse non-key fields (filters, weather, outcome, ...)
       │    each failure → that field = None/blank, row continues [D-05]
       │
       ├─ map Observation Status → run_status (translation table)
       │
       ▼
       CampaignRun.objects.get_or_create(
           campaign=campaign, telescope_instrument=..., ut_start=...,
           defaults={...all other fields..., approval_status='approved'}  [D-03]
       )
       │
       └─▶ created_count / updated_count / unchanged_count += 1
   │
   ▼
   stdout summary: created / updated / unchanged / skipped / site_needs_review counts
```

### Recommended Project Structure
```
solsys_code/
├── models.py                          # add CampaignRun here (see "Model File" note below)
├── campaign_utils.py                  # NEW — CSV column parsing helpers (obs_date/ut_start
│                                       #   regex extraction, Observation Status translation,
│                                       #   site resolution helper) — mirrors calendar_utils.py's
│                                       #   role for load_telescope_runs/sync_* commands
├── management/commands/
│   └── import_campaign_csv.py         # NEW — the bootstrap-import command
├── migrations/
│   └── 0002_campaignrun.py            # NEW — first schema addition since 0001
└── tests/
    └── test_import_campaign_csv.py    # NEW — mirrors test_load_telescope_runs.py shape

docs/notebooks/pre_executed/
├── import_campaign_csv_demo.ipynb     # NEW — paired demo notebook (CLAUDE.md convention)
└── fixtures/
    └── campaign_sample.csv            # NEW — D-10/D-11 synthetic fixture (first fixtures/ dir)
```

**Model file placement:** `solsys_code` is a *flat* Django app — `models.py` is a single file, not
a `models/` package. Django's migration autodetector only inspects `solsys_code.models`. If
`CampaignRun` is defined in a separate module (e.g. `campaign_models.py`), it **must** be imported
into `solsys_code/models.py` (e.g. `from solsys_code.campaign_models import CampaignRun  # noqa: F401`)
or `makemigrations` will silently see no new model. Given `CampaignRun`'s field count (~18 fields)
is large relative to the current single-model `models.py` (26 lines), either approach works, but the
simpler, lower-risk choice — and this research's recommendation — is to add `CampaignRun` directly
to the existing `solsys_code/models.py` alongside `CalendarEventTelescopeLabel`, avoiding the
import-forwarding gotcha entirely.

### Pattern 1: Two-field TextChoices status model (CAMP-03/D-02)
**What:** Two independent `CharField(choices=...)` attributes on one model, each backed by a
`django.db.models.TextChoices` subclass.
**When to use:** Exactly this case — two status dimensions that vary independently (admin gate vs.
real-world lifecycle) rather than a single flat enum.
**Example:**
```python
# Source: docs.djangoproject.com "Field choices enumeration types" (verified 2026-07-02)
# [CITED: docs.djangoproject.com/en/5.2/ref/models/fields/#field-choices-enum-types]
from django.db import models


class CampaignRun(models.Model):
    class ApprovalStatus(models.TextChoices):
        PENDING_REVIEW = 'pending_review', 'Pending Review'
        APPROVED = 'approved', 'Approved'
        REJECTED = 'rejected', 'Rejected'

    class RunStatus(models.TextChoices):
        REQUESTED = 'requested', 'Requested'
        PLANNED = 'planned', 'Planned'
        OBSERVED = 'observed', 'Observed'
        REDUCED = 'reduced', 'Reduced'
        PUBLISHED = 'published', 'Published'
        CANCELLED = 'cancelled', 'Cancelled'
        NOT_AWARDED = 'not_awarded', 'Not Awarded'
        WEATHER_TECH_FAILURE = 'weather_tech_failure', 'Weather/Technical Failure'

    approval_status = models.CharField(
        max_length=20, choices=ApprovalStatus, default=ApprovalStatus.PENDING_REVIEW
    )
    run_status = models.CharField(max_length=30, choices=RunStatus, default=RunStatus.REQUESTED)
```
Django 5.0+ allows passing the `TextChoices` class directly to `choices=` (no `.choices` needed);
this codebase's installed Django is 5.2.14, so the direct-class form above is safe to use.

### Pattern 2: Tiered site resolution reusing `MPCObscodeFetcher` (D-08)
**What:** Adapt `CreateObservatory.form_valid()`'s exact tier-1/tier-2 logic (currently coupled to a
Django form/view) into a plain function usable from a management command.
**When to use:** Any time an obscode-shaped string needs resolving to an `Observatory`, outside the
request/response cycle.
**Example:**
```python
# Source: adapted from solsys_code/solsys_code_observatory/views.py:CreateObservatory.form_valid
# and solsys_code/solsys_code_observatory/utils.py:MPCObscodeFetcher (both read directly,
# 2026-07-02) [VERIFIED: solsys_code/solsys_code_observatory/{views,utils}.py]
from django.db.utils import IntegrityError
from tom_dataservices.dataservices import MissingDataException

from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.solsys_code_observatory.utils import MPCObscodeFetcher

_MAX_OBSCODE_LEN = Observatory._meta.get_field('obscode').max_length  # 4


def resolve_site(site_code_raw: str) -> tuple[Observatory | None, bool]:
    """Resolve a raw Site Code string to an Observatory (D-08 3-tier resolution).

    Returns:
        tuple[Observatory | None, bool]: (observatory_or_None, needs_review).
    """
    code = (site_code_raw or '').strip()
    if not code:
        return None, True  # tier 3: no code at all -- flag, no placeholder possible

    if len(code) > _MAX_OBSCODE_LEN:
        # e.g. JWST's '500@-170' -- can't fit Observatory.obscode; don't fabricate
        # a truncated/wrong site. Flag for manual review instead (see Pitfall 2).
        return None, True

    # Tier 1: existing Observatory record
    try:
        return Observatory.objects.get(obscode=code), False
    except Observatory.DoesNotExist:
        pass

    # Tier 2: MPC Obscodes API (same call CreateObservatory.form_valid makes)
    fetcher = MPCObscodeFetcher()
    errors = fetcher.query(code)
    try:
        return fetcher.to_observatory(), False
    except MissingDataException:
        pass  # no such obscode at MPC either -- fall through to tier 3
    except IntegrityError:
        # Race: another row in this same import already created it (D-08 tier 2/3
        # collision) -- re-fetch instead of losing the row.
        return Observatory.objects.get(obscode=code), False

    # Tier 3: placeholder, flagged for review (D-09 -- flag, don't silently guess)
    placeholder = Observatory.objects.create(
        obscode=code,
        name=f'NEEDS REVIEW: {code}',
        short_name=code,
    )
    return placeholder, True
```
Note `errors` from `fetcher.query()` is intentionally unused above beyond triggering the
`MissingDataException` path — `MPCObscodeFetcher.query()` already logs the API error internally
(`logging.error(...)` in `utils.py:53`); don't double-log.

### Pattern 3: CSV row skip-and-log adapted from `load_telescope_runs.py`
**What:** Replace the file's per-line `for line_num, line in enumerate(...)` loop with a
`csv.DictReader` per-row loop; keep the same try/except-around-natural-key-fields structure.
**When to use:** This command's `handle()`.
**Example:**
```python
# Source: solsys_code/management/commands/load_telescope_runs.py:92-127, adapted (2026-07-02)
import csv

created_count = updated_count = unchanged_count = skipped_count = site_needs_review_count = 0

with open(filepath, encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row_num, row in enumerate(reader, start=2):  # start=2: header is row 1
        try:
            telescope_instrument = _require(row['Telescope / Instrument'])
            obs_date, ut_start, ut_end = parse_obs_window(row['Obs. Date'], row['UT Time Range'])
        except ValueError as exc:
            self.stderr.write(f'Row {row_num}: {exc} (row: {row!r})')
            skipped_count += 1
            continue

        site, needs_review = resolve_site(row.get('Site Code', ''))
        if needs_review:
            site_needs_review_count += 1

        run, action = CampaignRun.objects.get_or_create(
            campaign=campaign,
            telescope_instrument=telescope_instrument,
            ut_start=ut_start,
            defaults=dict(
                target=auto_target,
                site=site,
                site_raw=row.get('Site Code', ''),
                site_needs_review=needs_review,
                ut_end=ut_end,
                filters_bandpass=row.get('Filter(s)/Bandpass', '') or '',
                observation_details=row.get('Observation Details', '') or '',
                weather=row.get('Weather conditions or forecast', '') or '',
                run_status=map_observation_status(row.get('Observation Status', '')),
                approval_status=CampaignRun.ApprovalStatus.APPROVED,  # D-03
                observation_outcome=row.get('Observation Outcome', '') or '',
                publication_plans=row.get('Publication Plans', '') or '',
                open_to_collaboration=(row.get('Open to collaboration?', '') or '').strip().lower() == 'yes',
                contact_person=row.get('Contact Person', '') or '',
                contact_email=row.get('Email', '') or '',
                comments=row.get('Other comments', '') or '',
            ),
        )
        if action_counter := {'created': created_count, ...}:  # illustrative only
            ...
```
(Elided: the real command should reuse the exact "compare `defaults` against existing field values,
`.save(update_fields=...)` only if changed" idiom from `insert_or_create_calendar_event()` in
`calendar_utils.py:296-332`, rather than hand-rolling it — that function is generic enough over
`lookup`/`fields` dicts that it could be extended to `CampaignRun` too, or its diff-and-save body
copied verbatim into a `CampaignRun`-specific equivalent. Either is reasonable; picking one is a
planner-level call, not locked by CONTEXT.md.)

### Anti-Patterns to Avoid
- **Using `dateutil.parser.parse` on the `UT Time Range` column:** given real rows contain
  free-text garbage (see verified sheet data above), a permissive general-purpose date parser will
  "succeed" at parsing nonsense into a plausible-looking but wrong datetime — worse than failing
  loudly. Use targeted regex for the known formats instead (Code Examples below).
- **Truncating an oversized `Site Code` to fit `Observatory.obscode`'s 4-char limit and creating a
  placeholder anyway:** silently fabricates a wrong site record. Flag for review instead (Pattern 2
  above, Pitfall 2 below).
- **Reusing `pandas.read_csv`'s NaN-for-blank-cell semantics directly in model field assignment:**
  `CampaignRun.filters_bandpass = float('nan')` is a real footgun if `pandas` is used — always
  convert blank/NaN to `''`/`None` explicitly before assignment, regardless of which CSV reader is
  chosen.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MPC Obscodes API client | A new HTTP client for `data.minorplanetcenter.net` | `MPCObscodeFetcher` (`solsys_code/solsys_code_observatory/utils.py`) | Already implements the exact query→parse→`Observatory` conversion this phase needs; reusing it also means test mocking follows the existing `@patch('requests.get')` pattern already proven in `test_utils.py` |
| Coordinate/parallax math for placeholder `Observatory` rows | Custom lat/lon computation | `Observatory.from_parallax_constants()` (called internally by `MPCObscodeFetcher.to_observatory()`) | Tier-3 placeholders don't need coordinates at all (they're flagged for manual review), and tier-2 rows get correct coordinates for free via the existing conversion |
| Status-transition validation/state machine | A custom FSM library or `django-fsm` | Plain `TextChoices` + one `CharField` per status dimension | Per `.planning/research/FEATURES.md`'s own explicit anti-feature call-out: "a flat single vocabulary... no state-machine library needed... a small closed vocabulary, one field, one direction of travel" — this applies equally to the two-field split |
| create-or-update idempotency ("upsert") diff logic | Hand-rolled dirty-field tracking | `get_or_create()` + explicit field-by-field diff (mirror `insert_or_create_calendar_event()`) | Established pattern across Phases 3/4/10/11 per CONTEXT.md's own "Established Patterns" section — avoids `modified`-timestamp churn on unchanged re-runs |

**Key insight:** every "hard" sub-problem in this phase (external API client, coordinate math,
idempotent upsert, status vocabulary) already has a working, tested implementation somewhere in
`solsys_code/`. The actual engineering task is composition and CSV-shaped adaptation, not new
algorithm design.

## Common Pitfalls

### Pitfall 1: Treating "UT start" as always-parseable and skipping too aggressively
**What goes wrong:** D-04's natural key includes "obs date/UT start." If the planner interprets this
as "the row must have a cleanly parseable UT start time or it's skipped," roughly 4-5 of the 24 real
rows (blank `UT Time Range`, garbage text, or a stray date-range in that column — see verified sheet
data above) would be silently dropped from the bootstrap import, defeating CAMP-04's own goal of
validating the schema against the real, messy sheet.
**Why it happens:** The natural-key wording is intentionally terse; it doesn't specify a fallback.
**How to avoid:** Recommended fallback (flag for planner/discuss confirmation, not locked by
CONTEXT.md): if `Obs. Date` parses but `UT Time Range` doesn't yield a time-of-day, use `obs_date`
at `00:00:00 UTC` as the `ut_start` component of the natural key (still deterministic and
idempotent — a second run with the same unparseable range produces the same fallback value) rather
than skipping the row. Only skip when `Obs. Date` itself is unparseable/empty (true natural-key
failure — `telescope_instrument` and `campaign` alone aren't unique enough across repeat visits to
the same facility).
**Warning signs:** A `skipped_count` unexpectedly close to the total real row count on the live
import.

### Pitfall 2: Oversized `Site Code` values breaking placeholder creation
**What goes wrong:** JWST's real `Site Code` value (`500@-170`, 8 characters) exceeds
`Observatory.obscode`'s `max_length=4`. A naive tier-3 placeholder-creation call
(`Observatory.objects.create(obscode=site_code_raw, ...)`) either raises `django.db.utils.DataError`
(most backends enforce `max_length` at the DB level) or, worse, silently truncates depending on
backend/validation configuration — either aborts the whole row unexpectedly or fabricates a bogus
4-character site.
**Why it happens:** MPC's spacecraft-designation convention (`500@<naifid>`) predates and doesn't
fit this codebase's terrestrial-observatory-shaped `obscode` field.
**How to avoid:** Length-check the raw `Site Code` before attempting any tier; if it exceeds
`Observatory._meta.get_field('obscode').max_length`, skip straight to `site=None`,
`site_needs_review=True`, preserving the full raw string in `site_raw` (a `TextField`/longer
`CharField`, not itself 4-char-limited). See Pattern 2's `resolve_site()` above.
**Warning signs:** A `DataError`/`IntegrityError` traceback naming `obscode` during the real import
run, or an `Observatory` row with a suspiciously truncated `obscode` after import.

### Pitfall 3: `Observation Status` sheet values not matching the 8-value `run_status` vocabulary
**What goes wrong:** The real sheet's `Observation Status` column contains free-text-ish values
(`completed`, `Upcoming`, and likely others in cells not sampled above) that have no 1:1 mapping to
`requested`/`planned`/`observed`/`reduced`/`published`/`cancelled`/`not_awarded`/
`weather_tech_failure`. An unhandled/default case could silently mis-tag historical data.
**Why it happens:** The sheet's vocabulary predates and is coarser than the new `run_status`
8-value ladder (this is literally D-02's own stated rationale for building the richer vocabulary).
**How to avoid:** Build an explicit, case-insensitive substring-based translation table (Code
Examples below) with a documented, conservative default (`REQUESTED` — the lowest/most-neutral rung)
for anything unrecognized, and log a distinct counter (or at minimum a stderr note) for
unrecognized-but-imported status strings so the operator can manually review/upgrade them after the
real import — this is a non-key field per D-05, so it must never cause a row skip, only an
imprecise-but-safe default.
**Warning signs:** After the real import, spot-checking a `completed`-status sheet row and finding
its `CampaignRun.run_status` at the default `requested` value.

### Pitfall 4: `docs/notebooks/pre_executed/fixtures/` conflicting with `.gitignore`
**What goes wrong:** Assuming a new `fixtures/` directory needs a `.gitignore` exception.
**Why it happens:** `docs/notebooks/data/**` *is* gitignored (confirmed
`[VERIFIED: .gitignore:157]`), which could be mistaken for a blanket "anything under
`docs/notebooks/` is ignored" rule.
**How to avoid:** No action needed — `docs/notebooks/pre_executed/**` (where D-11 places the
fixture) is not covered by any `.gitignore` rule; the fixture CSV will commit normally like the
existing `.ipynb` files in that directory.
**Warning signs:** `git status` not showing the new fixture file as untracked after creating it (an
actual gitignore hit would show this) — not currently a risk per the verification above, but worth
a sanity `git add -n` check during execution.

## Code Examples

### `Observation Status` → `run_status` translation (Pitfall 3)
```python
# Recommendation only -- not verified against operator intent; flag as [ASSUMED] for
# discuss/plan confirmation. Case-insensitive substring match, most-specific first.
_STATUS_MAP = [
    ('cancel', CampaignRun.RunStatus.CANCELLED),
    ('not awarded', CampaignRun.RunStatus.NOT_AWARDED),
    ('weather', CampaignRun.RunStatus.WEATHER_TECH_FAILURE),
    ('technical', CampaignRun.RunStatus.WEATHER_TECH_FAILURE),
    ('publish', CampaignRun.RunStatus.PUBLISHED),
    ('reduc', CampaignRun.RunStatus.REDUCED),
    ('complet', CampaignRun.RunStatus.OBSERVED),   # sheet's 'completed' -> OBSERVED (safe minimum claim)
    ('observ', CampaignRun.RunStatus.OBSERVED),
    ('upcoming', CampaignRun.RunStatus.PLANNED),
    ('planned', CampaignRun.RunStatus.PLANNED),
]


def map_observation_status(raw: str) -> str:
    normalized = (raw or '').strip().lower()
    for needle, status in _STATUS_MAP:
        if needle in normalized:
            return status
    return CampaignRun.RunStatus.REQUESTED  # conservative default (D-05: never blocks the row)
```

### UT Time Range best-effort parsing (Pitfall 1)
```python
# Recommendation only -- [ASSUMED], covers the verified real-format inventory above.
import re
from datetime import date, datetime, time
from datetime import timezone as dt_timezone

_HHMM_RANGE = re.compile(r'(\d{1,2})[:;](\d{2})\s*(?:AM|PM)?\s*-\s*(\d{1,2})[:;](\d{2})', re.IGNORECASE)
_APPROX_HOUR = re.compile(r'~?\s*(\d{1,2})\s*(?:am|pm)?\s*UTC?', re.IGNORECASE)


def parse_obs_window(obs_date_raw: str, ut_range_raw: str) -> tuple[date, datetime, datetime | None]:
    """Best-effort parse of the sheet's Obs. Date + UT Time Range columns.

    Raises:
        ValueError: if obs_date_raw itself can't be parsed to a date (true natural-key
            failure per D-05) -- a bad/missing ut_range_raw does NOT raise; it falls
            back to obs_date at 00:00 UTC (Pitfall 1).
    """
    obs_date = datetime.strptime(obs_date_raw.strip(), '%Y-%m-%d').date()  # ValueError propagates

    match = _HHMM_RANGE.search(ut_range_raw or '')
    if match:
        h1, m1, h2, m2 = (int(x) for x in match.groups())
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h1, m1, tzinfo=dt_timezone.utc)
        end = datetime(obs_date.year, obs_date.month, obs_date.day, h2, m2, tzinfo=dt_timezone.utc)
        return obs_date, start, end

    match = _APPROX_HOUR.search(ut_range_raw or '')
    if match:
        h = int(match.group(1))
        start = datetime(obs_date.year, obs_date.month, obs_date.day, h, 0, tzinfo=dt_timezone.utc)
        return obs_date, start, None

    # Fallback: obs_date is valid but UT range isn't parseable at all (blank, garbage
    # text, or a misplaced date-range) -- use midnight UTC (Pitfall 1), never skip here.
    start = datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0, tzinfo=dt_timezone.utc)
    return obs_date, start, None
```

### Management command test pattern (mirrors `test_load_telescope_runs.py`)
```python
# Source: adapted from solsys_code/tests/test_load_telescope_runs.py (read directly, 2026-07-02)
import csv
import io
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase

from solsys_code.models import CampaignRun
from solsys_code.solsys_code_observatory.models import Observatory
from tom_targets.models import TargetList


class TestImportCampaignCsv(TestCase):
    def _write_csv(self, rows: list[dict]) -> tuple[str, tempfile.TemporaryDirectory]:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmpdir_ctx.name) / 'campaign.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return str(path), tmpdir_ctx

    def test_creates_campaignrun_with_existing_observatory(self):
        Observatory.objects.create(obscode='F65', name='FTN', short_name='FTN', lat=20.7, lon=-156.3, altitude=3055)
        path, ctx = self._write_csv([{'Telescope / Instrument': 'FTN/MuSCAT3', 'Site Code': 'F65', ...}])
        with ctx:
            call_command('import_campaign_csv', '--campaign', 'Test Campaign', path,
                          stdout=io.StringIO(), stderr=io.StringIO())
            self.assertEqual(CampaignRun.objects.count(), 1)
            self.assertEqual(CampaignRun.objects.first().approval_status, CampaignRun.ApprovalStatus.APPROVED)

    @patch('requests.get')
    def test_tier2_mpc_lookup_creates_observatory(self, mock_get):
        # Mirrors test_utils.py's @patch('requests.get') mocking pattern for MPCObscodeFetcher
        mock_response = MagicMock(ok=True)
        mock_response.json.return_value = {...}  # same shape as test_utils.py's obs_data fixture
        mock_get.return_value = mock_response
        ...

    def test_unresolvable_site_flags_needs_review_without_skipping_row(self):
        # D-09: site failure must not skip the row
        ...

    def test_idempotent_rerun_no_duplicates(self):
        # Mirrors test_load_telescope_runs.py's test_idempotent_rerun_no_duplicates
        ...

    def test_natural_key_failure_skipped_and_logged(self):
        # D-05: a row missing/malformed Obs. Date is skipped and logged; other rows still import
        ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Google Sheet manual coordination | `CampaignRun` model + bootstrap CSV import | This phase | Replaces link-shared spreadsheet with a queryable, FOMO-native record per the milestone's core value |
| `CalendarEvent`-only tracking (telescope/instrument/start/end, no campaign or PII fields) | `CampaignRun` as a distinct model, cross-linked later (Phase 16 CAL-01) rather than widening `CalendarEvent` | This milestone (design decision predates this phase, documented in the seed) | Keeps `CalendarEvent` scoped to its existing sync-command consumers; avoids a second sidecar-on-sidecar pattern |

**Deprecated/outdated:** none — this is new functionality, not a replacement of existing FOMO code.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Recommendation to fall back to `obs_date` at `00:00 UTC` when `UT Time Range` is unparseable (rather than skipping the row) | Common Pitfalls #1, Code Examples | If the operator instead wants strict skip-on-unparseable-time, several real historical rows would be silently dropped from the bootstrap import instead of imported with a documented fallback — worth confirming at `/gsd-discuss-phase` or plan-review time if not already settled by CONTEXT.md (CONTEXT.md left this as "Claude's Discretion" for date/time parsing strategy) |
| A2 | Recommendation to skip Observatory-resolution tiers entirely (go straight to flagged/`site=None`) for `Site Code` values exceeding `Observatory.obscode`'s 4-char `max_length` (e.g. JWST's `500@-170`), rather than truncating | Common Pitfalls #2, Pattern 2 | If the operator prefers extending `Observatory.obscode`'s max_length in a new migration to accommodate spacecraft codes, this changes the `solsys_code_observatory` app (a different app than this phase's canonical scope) and needs an explicit decision, not an assumption |
| A3 | `Observation Status` → `run_status` translation table content (which sheet strings map to which of the 8 values, and `REQUESTED` as the conservative default) | Common Pitfalls #3, Code Examples | A different mapping choice (e.g. mapping `'completed'` to `PUBLISHED` instead of `OBSERVED`) would change the imported data's semantics; low risk since `run_status` is a non-key field the operator can bulk-correct post-import via Django admin, but worth a quick discuss-phase confirmation given only 2 status strings were directly observed in the sample (`completed`, `Upcoming`) — the full 24-row set may contain others |
| A4 | Single `telescope_instrument` CharField (not split into separate `telescope`/`instrument` fields like `CalendarEvent`) matches the sheet's own single-column shape and is what D-04's natural-key "telescope" component refers to | Architecture Patterns, Standard Stack | If the planner instead wants a `CalendarEvent`-style split, the natural-key implementation and CSV column mapping both need adjusting; the sheet itself never separates these two, so a split would require inventing a parsing rule that doesn't exist in the source data |
| A5 | `CampaignRun.campaign` FK uses `on_delete=PROTECT` (not `CASCADE` or `SET_NULL`) since `campaign` is required (`null=False`) per CAMP-01, unlike `CalendarEvent.target_list`'s `SET_NULL` (which is nullable) | Code Examples (not shown inline — flagged here for completeness) | `SET_NULL` is invalid on a non-nullable FK; `CASCADE` would silently delete all campaign history if a `TargetList` is ever deleted — `PROTECT` is the conservative choice preventing accidental data loss, but is this research's inference, not an explicit CONTEXT.md decision |

## Open Questions

1. **Does `CampaignRun` need a DB-level `UniqueConstraint` on the natural key, or is
   `get_or_create()` alone (matching `CalendarEvent`'s precedent — no explicit unique constraint
   exists there either) sufficient?**
   - What we know: `get_or_create()` correctly translates a `None` lookup value to `IS NULL` in the
     generated SQL, so app-level idempotency works even when `ut_start` falls back to midnight UTC
     rather than being genuinely absent (there's always a value once `obs_date` parses).
   - What's unclear: whether the planner wants defense-in-depth against concurrent/manual
     `CampaignRun.objects.create()` calls bypassing the command (e.g. from a future Phase 16
     submission form) creating natural-key duplicates.
   - Recommendation: follow `CalendarEvent`'s precedent (no DB constraint, app-level `get_or_create`
     only) for consistency; revisit if Phase 16's submission form reveals a real race condition risk.

2. **Full 24-row `Observation Status` and `Publication Plans` value inventory beyond the samples
   captured in this document.**
   - What we know: the samples above show `completed`/`Upcoming` for status and
     URL/`TBD`/`survey`-shaped free text for publication plans (per the earlier general-format pass,
     not reproduced verbatim here since it wasn't re-verified column-by-column).
   - What's unclear: whether any additional `Observation Status` strings exist that the translation
     table (Code Examples) doesn't recognize.
   - Recommendation: the real bootstrap-import run itself (CAMP-04's stated purpose) is the correct
     place to discover this — log unrecognized status strings distinctly (a `stderr.write` note, not
     necessarily a full counter) so the operator can review post-import, rather than trying to
     enumerate every value in advance from this research pass.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Network access to `data.minorplanetcenter.net` | D-08 tier 2 (MPC Obscodes API) during the *real* CSV import (not the demo notebook, per D-11) | Not verified in this research session (existing `MPCObscodeFetcher` already assumes this is reachable in production use) | — | If unreachable, tier 2 fails and every unresolved site correctly falls to tier 3 (placeholder + flag) — this is already the designed degraded-mode behavior, not a blocker |
| The real 3I/ATLAS sheet, exported to CSV | CAMP-04's live bootstrap import (not required for tests or the demo notebook) | Confirmed fetchable this session via the sheet's public export URL (`[VERIFIED: direct WebFetch of the sheet's CSV export, 2026-07-02]`); not currently committed to the repo per CONTEXT.md's own note | 24 data rows as of 2026-07-02 | The operator provides this file at execution time via the command's positional CSV-path argument; it is explicitly never a test/notebook dependency |

**Missing dependencies with no fallback:** none — the real-sheet CSV is operator-supplied at
execution time by design (CONTEXT.md), not something this phase's tests or notebook depend on.

**Missing dependencies with fallback:** MPC API reachability (degrades gracefully to tier 3, by
design).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Django `TestCase` via `./manage.py test` (this app's Django-test-suite half; NOT the `pytest` suite, which excludes `solsys_code/` per `pyproject.toml`'s `testpaths`) |
| Config file | none dedicated — relies on `src.fomo.settings` (same as every existing `solsys_code` test) |
| Quick run command | `./manage.py test solsys_code.tests.test_import_campaign_csv` |
| Full suite command | `./manage.py test solsys_code` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CAMP-01 | `CampaignRun` stores full field inventory + required `campaign` FK | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ❌ Wave 0 |
| CAMP-02 | Optional `target`; single-target campaign auto-resolves (D-07) without manual setting | unit | `./manage.py test solsys_code.tests.test_import_campaign_csv.TestImportCampaignCsv.test_auto_resolves_single_target_campaign` | ❌ Wave 0 |
| CAMP-03 | Two independent status fields, 8 `run_status` values, correct defaults | unit | `./manage.py test solsys_code.tests.test_campaign_models` | ❌ Wave 0 |
| CAMP-04 | Command reports created/updated/skipped; skip-and-log on natural-key failure; non-key failures null just that field; idempotent re-run | integration | `./manage.py test solsys_code.tests.test_import_campaign_csv` | ❌ Wave 0 |
| CAMP-05 | Demo notebook executes end-to-end with no live network call, against the synthetic fixture only | manual/notebook-execution | `jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` | ❌ Wave 0 (notebook + fixture both new) |

### Sampling Rate
- **Per task commit:** `./manage.py test solsys_code.tests.test_import_campaign_csv` (and
  `test_campaign_models` once split out)
- **Per wave merge:** `./manage.py test solsys_code` (full Django suite — this phase doesn't touch
  anything under `pytest`'s `testpaths`, so `python -m pytest` is not required to re-run for this
  phase's own changes, but should still pass unchanged)
- **Phase gate:** Full `./manage.py test solsys_code` green, plus `ruff check .` /
  `ruff format --check .` clean, plus the demo notebook re-executed and committed with output,
  before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `solsys_code/tests/test_campaign_models.py` — covers CAMP-01/CAMP-02/CAMP-03 model-level
      behavior (field presence, status defaults, optional `target`)
- [ ] `solsys_code/tests/test_import_campaign_csv.py` — covers CAMP-04 (mirrors
      `test_load_telescope_runs.py`'s shape: temp-file CSV fixtures, `call_command`,
      stdout/stderr assertions)
- [ ] `docs/notebooks/pre_executed/fixtures/campaign_sample.csv` — the D-10/D-11 synthetic fixture
      (also doubles as a manually-inspectable input for hand-verifying the notebook's demonstrated
      behavior)
- [ ] `docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb` — covers CAMP-05
- [ ] Migration `solsys_code/migrations/0002_campaignrun.py` (generated via `./manage.py
      makemigrations solsys_code`, not hand-written — see `0001_calendareventtelescopelabel.py`
      for the expected auto-generated shape)
- Framework install: none — `./manage.py test` is already fully configured, no new packages needed.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This phase has no HTTP-facing views/forms — the only entry point is an operator-run CLI management command |
| V3 Session Management | No | Same reason as V2 |
| V4 Access Control | No | Same reason as V2 — access control on *displaying* the PII fields this phase stores is Phase 15's VIEW-03, not this phase's concern |
| V5 Input Validation | Yes | CSV field parsing (obs date/UT time regex, `Site Code` length check before any DB write, `Observation Status` translation with a safe default) — see Common Pitfalls/Code Examples above; Django's `EmailField` for basic `contact_email` format validation at the model layer |
| V6 Cryptography | No | Nothing in this phase touches secrets, tokens, or encrypted storage |
| V8 Data Protection (PII at rest) | Partial — applies to *storage*, not yet *display* | `CampaignRun.contact_person`/`contact_email` store real PII once the live 3I/ATLAS import runs (D-03's real historical data). This phase's own guard is CAMP-05 (never commit real PII to git via the demo notebook/fixture, D-10/D-11) — the *display*-side guard (auth-gated rendering) is explicitly Phase 15's VIEW-03, out of scope here. No additional encryption-at-rest control is implied — SQLite dev DB (per CLAUDE.md's documented conventions) is not itself gitignored differently than any other model's data, consistent with existing `Observatory`/`Target` PII-adjacent handling in this codebase |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| CSV/spreadsheet formula injection (a cell value like `=cmd\|'/c calc'!A0` or `=HYPERLINK(...)` in a free-text field such as `comments`/`observation_details`, later re-exported to CSV/Excel by staff) | Tampering | Not a risk *this phase introduces* (this phase only *imports* CSV, never exports it back to a spreadsheet format) — flag for the planner's awareness only, relevant if/when a future phase adds CSV export of `CampaignRun` data; no mitigation needed in Phase 14 itself |
| Untrusted CLI file path (`import_campaign_csv <path>`) used to read arbitrary filesystem paths | Information Disclosure | Already the existing, accepted pattern for `load_telescope_runs <filepath>` — this is an operator-run CLI tool (not a web-facing upload), so the same trust boundary applies unchanged; `open(filepath)` wrapped in `except OSError` mirrors the existing command's error handling |
| Outbound SSRF via `Site Code` values fed into `MPCObscodeFetcher.query(obscode)` | Tampering / Information Disclosure | Low risk in this phase: `requests.get()` targets a hardcoded MPC URL (`utils.py:43`) with the obscode passed as a JSON body parameter, not URL-interpolated — no attacker-controlled URL construction exists in the current `MPCObscodeFetcher` implementation, and this phase's CSV input is operator-supplied, not web-form-submitted (that's Phase 16's SUBMIT-04 honeypot-guarded surface) |

## Sources

### Primary (HIGH confidence)
- `solsys_code/models.py`, `solsys_code/calendar_utils.py`,
  `solsys_code/management/commands/load_telescope_runs.py`,
  `solsys_code/solsys_code_observatory/{models,utils,views}.py`,
  `solsys_code/tests/test_load_telescope_runs.py`,
  `solsys_code/solsys_code_observatory/tests/test_utils.py`,
  `solsys_code/migrations/0001_calendareventtelescopelabel.py`,
  `solsys_code/solsys_code_observatory/migrations/*.py` — read directly this session
- Live Django model introspection (`CalendarEvent`, `TargetList`, `Target`, `Observatory` field
  lists and `on_delete` behaviors) — run directly via `./manage.py shell`-equivalent Python this
  session
- Direct CSV export fetch of the real 3I/ATLAS coordination sheet (column headers + non-PII data
  patterns for all 24 rows) — fetched this session, 2026-07-02
- `.gitignore`, `pyproject.toml`, `src/fomo/settings.py` (`TIME_ZONE`/`USE_TZ`) — read directly

### Secondary (MEDIUM confidence)
- [Django model field reference — Field choices enumeration types](https://docs.djangoproject.com/en/5.2/ref/models/fields/#field-choices-enum-types) — confirmed `TextChoices` usage pattern and Django 5.0+ direct-class `choices=` support

### Tertiary (LOW confidence)
- General web search on `csv.DictReader` skip-and-log patterns — confirms this is a widely-used,
  unremarkable pattern; not treated as authoritative, only as a sanity check against the
  `load_telescope_runs.py`-derived approach already recommended above

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; every tool choice grounded in existing, working code
  in this exact codebase
- Architecture: HIGH — direct precedent (`load_telescope_runs.py`, `CreateObservatory.form_valid`,
  `calendar_utils.py`) read and adapted, not invented
- Pitfalls: HIGH — grounded in the actual, verified real-sheet data (not hypothetical messiness)

**Research date:** 2026-07-02
**Valid until:** 30 days (stable internal codebase patterns; the external dependency — the live
Google Sheet's exact row contents — could change if the operator/community keep editing it before
the real bootstrap import runs, so re-verify row count/shape immediately before executing CAMP-04's
live import if significant time has passed)
