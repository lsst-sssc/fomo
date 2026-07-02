# Stack Research

**Domain:** Campaign-coordination data model, CSV bootstrap import, table view, moderated submission form, and coverage-gap analysis inside an existing Django + TOM Toolkit app (FOMO)
**Researched:** 2026-07-02
**Confidence:** HIGH

## Bottom Line

**No new third-party packages are required for v2.0.** Every one of the five target
features (campaign-run data model, CSV bootstrap import, per-target campaign table,
submission form + approval queue, coverage-gap analysis) is fully coverable with
packages already installed in this environment (`pip show` confirmed versions below)
plus the Python/Django standard library and code already in `solsys_code/`. This
matches the project's existing pattern of solving problems with plain Django models
and management commands (`CalendarEventTelescopeLabel`, `fetch_jplsbdb_objects`,
`load_telescope_runs`) rather than pulling in generic frameworks.

## Recommended Stack

### Core Technologies (already installed — reuse, don't reinstall)

| Technology | Installed Version | Purpose in v2.0 | Why Recommended |
|------------|---------|---------|-----------------|
| Django | 5.2.15 | `CampaignRun` model, migrations, admin, forms, views | Already the app framework; new model is a normal Django app addition, no framework decision to make |
| django-tables2 | 3.0.0 | Per-target campaign table view (spreadsheet-replacement display) | Already installed and in `INSTALLED_APPS`; purpose-built for exactly this "sortable/paginated model table" use case, same tool that would be reached for on a green-field Django project today |
| django-filter | 24.3 | Column filters on the campaign table (status, filter/bandpass, telescope) if/when needed | Already installed; pairs natively with django-tables2's `FilterView`/`SingleTableMixin` pattern |
| django-crispy-forms 2.4 + crispy-bootstrap4 2024.10 | 2.4 / 2024.10 | Renders the community submission `ModelForm` with the project's existing Bootstrap 4 styling | Already installed and used for `EphemerisForm`/`CreateObservatoryForm`; a new form should look consistent, not introduce a second form-styling system |
| pandas | 2.3.1 | Parsing the 3I/ATLAS Google-Sheets CSV export in the one-off bootstrap-import management command | Already a direct import in `solsys_code/views.py` and `solsys_code/ephem_utils.py` (not just a transitive sorcha dependency) — `pd.read_csv` handles BOM/encoding/blank-row/NaN quirks typical of a Sheets export far more robustly than hand-rolled `csv.reader` code, with zero new dependency cost |
| astropy | 6.1.7 | Coverage-gap analysis (observable-but-unclaimed dates) | Already the project's sun/ephemeris library (`solsys_code/telescope_runs.py`'s `sun_event`/`get_site`); gap analysis is "which nights in the observable window have no `CampaignRun`", built on the same primitives, not a new astronomy stack |
| django.contrib.admin | (bundled with Django 5.2.15) | Admin approval queue for pending community submissions | Already installed/wired (`django.contrib.admin` in `INSTALLED_APPS`); a custom `ModelAdmin` with an "Approve selected" `admin.action` is the standard, zero-dependency way to build a moderation queue for a single model |

### Supporting Libraries (already installed — situational use)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| django-guardian | 2.4.0 | Object-level permission if a *specific* campaign run needs per-object visibility control (e.g. "only the submitting PI + admins can edit this row") | Only if per-row edit permissions are required; do **not** reach for it just to gate the contact-email *field* — that's a display-time check, not an object-permission problem (see "What NOT to Use") |
| django_htmx | 1.23.2 | Optional: live-refresh the campaign table or approval queue without a full page reload | Nice-to-have polish, not required for MVP; only pull in if the submission-form UX needs partial-page updates (mirrors the existing calendar htmx pattern) |
| stdlib `csv` | n/a (stdlib) | Fallback/companion to pandas for the bootstrap import if a lighter, streaming, dependency-free path is preferred for one specific edge case (e.g. reading a huge sheet row-by-row) | Use pandas as the default per row above; only drop to stdlib `csv` if a specific row-streaming or memory concern arises — this is a one-off import command, not a hot path |
| stdlib `datetime`/`itertools` | n/a (stdlib) | Computing the observable-but-unclaimed date gaps (sort `CampaignRun` date ranges per target, diff against the ephemeris-observable window from `telescope_runs.py`) | Core implementation of coverage-gap analysis; no interval-arithmetic package needed at this scale (one target, one observing season's worth of nights) |
| stdlib `django.core.mail` | n/a (stdlib) | Notifying admins when a new submission needs approval | Django ships a working `send_mail`; use the existing `EMAIL_BACKEND` setting rather than adding a notifications package for a single "new submission" email |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff | Lint/format (already configured in `pyproject.toml`) | No new config needed; new files fall under existing `[tool.ruff]` rules |
| `./manage.py test solsys_code` | DB-dependent tests for the new `CampaignRun` model, import command, table view, and submission/approval views | Follows the existing two-suite split (CLAUDE.md) — all new tests are DB-dependent and belong here, not in `tests/` |

## Installation

```bash
# Nothing to install — all of the above are already present in this environment.
# Confirm with:
pip show django-tables2 django-filter django-crispy-forms crispy-bootstrap4 django-guardian django-htmx pandas astropy

# If any milestone-scope decision later adds change-history/audit-trail tracking
# for the lifecycle status field (planned -> observed -> reduced -> published),
# that would be the one plausible genuine addition — see "Alternatives Considered".
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| Hand-rolled `status` `CharField(choices=...)` on `CampaignRun` + a Django admin "Approve" action | A generic moderation framework (`django-moderation`, `django-approval`, `djangocms-moderation`) | Only if FOMO needs moderation across *many* unrelated model types with diffing/versioned-change review UI. These packages moderate arbitrary `GenericForeignKey`-linked objects and are built for CMS-scale editorial workflows; for one model with a linear pending→approved/rejected gate they add real complexity (extra migrations, extra admin surface, and `django-approval` is explicitly "beta quality" per its own README) for no benefit over a four-line admin action. Confirmed via package survey (2026): both are still narrowly-scoped for change-review, not a good fit for a single always-new (not "edited-in-place") submission record |
| Plain `status` choices field (`planned`/`observed`/`reduced`/`published`) | `django-fsm` / `django-model-utils` `StatusField` | Only if the lifecycle needs enforced transition rules (e.g. "can't go from `planned` directly to `published`") with guard functions and signals. Neither is installed, neither is currently needed — the 4-state lifecycle described in the seed is a simple forward progression an admin/PI sets directly; a state-machine library is justified when transitions have side effects or need to be *prevented*, not just recorded |
| pandas `read_csv` for the bootstrap import | stdlib `csv` module only | Use stdlib `csv` if the import needs to stream a very large file row-by-row with minimal memory, or if a future recurring (not one-off) import wants zero extra dependency surface. For this milestone's *one-off* real Google-Sheets export, pandas (already a direct project dependency) is more robust against the export's real-world messiness (blank rows, merged-cell artifacts, inconsistent encoding) |
| django-tables2 + django-filter for the campaign table | Hand-rolled `ListView` + manual pagination/sorting templates | Only if the table needs a bespoke, highly custom layout that fights django-tables2's column model. The per-target campaign table is a straightforward "one row per run, sortable/filterable columns" case — exactly django-tables2's design target, and it's already the installed, in-use pattern for tabular Django views in this stack |
| Django's own `django.contrib.admin` approval queue | A dedicated workflow/ticketing package | Only if approval needs multi-step review (e.g. two-person sign-off) beyond a single admin action. The seed's approval model is single-gate ("admin reviews and approves before public") — `admin.action` on `CampaignRunAdmin` covers it |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| django-guardian for gating the contact-email **field** | Object-level permissions answer "can this user touch this row," not "should this field render for this viewer." Using guardian here is solving a display-time redaction problem with a data-layer permission system — extra migrations (per-object permission rows) for something a template `{% if request.user.is_staff %}` / view-level check does in one line | A plain auth check (`request.user.is_staff` or `is_authenticated`, per the phase's PII policy decision) in the view/template that renders `CampaignRun.contact_email` |
| django-fsm / django-model-utils for the lifecycle status | Adds a new dependency + migration surface for a 4-value linear status field with no transition-guard requirements described in scope | `models.CharField(choices=CampaignRun.Status.choices)` using Django's built-in `TextChoices` (stdlib to Django, no package) |
| django-moderation / django-approval / djangocms-moderation for the approval queue | Generic content-moderation frameworks aimed at arbitrary/multiple model types with diff-based change review; `django-approval` is explicitly "beta quality" per its own docs, `django-moderation`'s primary activity predates this project's Django 5.2 baseline — mismatched maturity and scope for a single always-new submission record | A `status` field + `ModelAdmin.actions = ['approve_selected']` (or a small dedicated pending-queue `ModelAdmin` queryset filter) |
| A CAPTCHA/anti-spam package (`django-recaptcha`, `django-honeypot`) for the submission form | Not called for by the seed's own design: the admin-approval gate *is* the spam/error control ("Approval step catches accidental duplicates... uncoordinated community submissions" — seed doc, `target-linked-run-submission-form.md`). Adding bot-defense infrastructure before there's evidence of bot traffic is premature for a niche astronomy-coordination form | Defer; revisit only if real spam submissions are observed post-launch |
| A dedicated interval/date-range package (e.g. `python-intervals`) for coverage-gap analysis | Overkill for "per target, per observing season, find nights with no claimed run" — a handful of date ranges, not a general interval-algebra problem | stdlib `datetime` + sorting/set-difference logic layered on `telescope_runs.py`'s existing `sun_event`/dark-window output |
| Introducing a REST/GraphQL layer for the campaign table or submission form | Not required — the feature is server-rendered Django views/templates matching every other FOMO screen (ephemeris form, observatory CRUD, calendar). `rest_framework` is installed for other purposes but adding an API surface here is scope creep with no stated consumer | Standard Django `CreateView`/`ListView` (or django-tables2 `SingleTableView`) + templates, consistent with the rest of the app |

## Stack Patterns by Variant

**If the PII policy (open question in the seed) lands on "auth-gated, staff-only display":**
- Use a plain `request.user.is_staff` check in the table/detail template — no new package.
- If it instead lands on "opt-in per-submitter display flag," add a `show_contact_publicly = BooleanField(default=False)` on `CampaignRun` itself — still no new package, just an extra field + template branch.

**If coverage-gap analysis needs to render a visual (calendar-heatmap-style) view later:**
- Reuse the existing `tom_calendar`/`calendar_display_extras` template-tag rendering pattern (already installed, already WCAG-checked via `text_color_for_bg`) rather than adding a JS charting library — the v1.4 visual-clarity work already solved "render date-keyed colored blocks in this app."

**If the bootstrap CSV import needs to run more than once (recurring, not one-off):**
- Revisit `load_telescope_runs`'s idempotent `get_or_create` + no-churn-save pattern (already proven in this codebase) rather than reaching for a dedicated ETL package — the volume (one campaign spreadsheet) never approaches what would justify one.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| Django 5.2.15 | django-tables2 3.0.0, django-filter 24.3, django-crispy-forms 2.4, crispy-bootstrap4 2024.10, django-guardian 2.4.0, django_htmx 1.23.2 | All already resolved and running together in this environment (`pip show` confirmed 2026-07-02); no version bump needed for v2.0 |
| pandas 2.3.1 | astropy 6.1.7 / sorcha 1.1.0 | Already coexisting (sorcha's `sbpy>=0.6.0` pin in `pyproject.toml` exists specifically for astropy 7.2.0+ compatibility per its inline comment) — no new interaction introduced by using `pd.read_csv` for the bootstrap import |
| tomtoolkit 3.0.0a9 | `tom_calendar.models.CalendarEvent` (no direct `Target` FK — only `TargetList`) | Confirmed by reading the installed `tom_calendar/models.py`: a new `CampaignRun` model must hold its **own** `ForeignKey` to `tom_targets.Target` (the seed's premise — "CalendarEvent has no Target link today") rather than assuming one exists on `CalendarEvent` |

## Sources

- `pip show` against the project's active venv (`/home/tlister/venv/fomo_venv`) — HIGH confidence, ground truth for installed versions (django-tables2 3.0.0, django-filter 24.3, django-guardian 2.4.0, django_htmx 1.23.2, crispy-bootstrap4 2024.10, django-crispy-forms 2.4, Django 5.2.15, pandas 2.3.1, astropy 6.1.7, sorcha 1.1.0, tomtoolkit 3.0.0a9)
- Direct read of installed `tom_calendar/models.py` — HIGH confidence, confirms `CalendarEvent` has no `Target` FK (only `TargetList`), informing the "own FK on `CampaignRun`" integration point
- `solsys_code/views.py`, `solsys_code/ephem_utils.py` — confirmed `pandas` is already a direct (not merely transitive) import in this codebase
- Web search, "django content moderation approval queue package 2026" — MEDIUM confidence, informs the "What NOT to Use" rationale on `django-moderation`/`django-approval`/`djangocms-moderation` maturity/scope mismatch: [Django Packages: Moderation grid](https://djangopackages.org/grids/g/moderation/), [django-approval (PyPI)](https://pypi.org/project/django-approval/), [django-moderation (GitHub)](https://github.com/dominno/django-moderation)
- `.planning/seeds/target-linked-run-submission-form.md`, `.planning/notes/web-form-vs-file-ingest.md`, `.planning/PROJECT.md` — HIGH confidence, project-internal source of truth for feature scope and the open PII-policy question

---
*Stack research for: Campaign coordination for rare/urgent Solar System objects (FOMO v2.0)*
*Researched: 2026-07-02*
