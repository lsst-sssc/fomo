# Architecture Research

**Domain:** Campaign coordination for rare/urgent Solar System objects — new feature area on an existing Django + TOM Toolkit app (FOMO v2.0)
**Researched:** 2026-07-02

This file supersedes the previous contents (dated 2026-07-01, about the v1.7 ESO/VLT feasibility spike) — that topic shipped as an investigation-only decision doc and is now closed; this is a full rewrite for the v2.0 Campaign Coordination milestone.

## Recommended Architecture

Add one new first-class model (`CampaignRun`) plus a small module family alongside the
existing `solsys_code/calendar_utils.py` / `calendar_urls.py` / `templatetags/calendar_display_extras.py`
trio, rather than widening `CalendarEvent` or spinning up a new Django app.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         Target detail page (TOM)                          │
│  target_detail_buttons() hook → "Campaign Runs" button (NEW, apps.py)     │
│  nav_items() hook → "Campaigns" navbar item (NEW, apps.py)                │
└───────────────┬─────────────────────────────┬─────────────────────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│ Campaign table view (per-Target)│   │ Submission form + approval queue view  │
│ campaign_views.py (NEW)         │   │ campaign_views.py (NEW)                │
│ read path, PII-gated column     │   │ write path, status-gated visibility    │
└───────────────┬─────────────────┘   └───────────────┬─────────────────────┘
                │                                     │
                ▼                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     CampaignRun model (NEW, solsys_code/models.py)        │
│  FK → tom_targets.Target (required)                                       │
│  FK/OneToOne → tom_calendar.CalendarEvent (nullable, projection link)     │
│  status: PENDING_REVIEW / APPROVED / OBSERVED / REDUCED / PUBLISHED /     │
│          REJECTED                                                         │
│  contact_person, contact_email (PII), telescope, instrument, site,        │
│  filters, obs_type, obs_date_start/end, outcome, publication_plans,       │
│  open_to_collaboration, comments                                          │
└───────────────┬───────────────────────────┬──────────────────────────────┘
                │                           │
                ▼                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│ import_campaign_csv command    │   │ On approval + telescope/dates present: │
│ (NEW, one-off 3I bootstrap)    │   │ insert_or_create_calendar_event()      │
│ management/commands/           │   │ (REUSED, calendar_utils.py)            │
└───────────────────────────────┘   └───────────────┬─────────────────────────┘
                                                     ▼
                                    ┌───────────────────────────────────────┐
                                    │ Existing calendar rendering path       │
                                    │ fomo_render_calendar / calendar.html   │
                                    │ (REUSED, no changes needed)            │
                                    └───────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Coverage-gap view (LAST) — telescope_runs.sun_event()/get_site() (REUSED, │
│ light) crossed against CampaignRun date ranges per Target/site.           │
│ MUST NOT import solsys_code.ephem_utils (1.6 GB SPICE download trigger).  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `CampaignRun` (new model, `solsys_code/models.py`) | Source-of-truth record for one observing run: target link, lifecycle status, PII contact fields, outcome/publication metadata | `tom_targets.Target` (FK, required), `tom_calendar.CalendarEvent` (FK/OneToOne, nullable projection link) |
| `campaign_views.py` (new) | Per-target table view (read), submission form (write, unauthenticated allowed per `AUTH_STRATEGY='READ_ONLY'`), staff-gated approval queue | `CampaignRun`, `campaign_extras` template tags, `calendar_utils.insert_or_create_calendar_event` |
| `campaign_urls.py` (new, namespaced `campaigns`) | Routes `/campaigns/target/<pk>/`, `/campaigns/submit/`, `/campaigns/review/`, `/campaigns/target/<pk>/coverage/` | Included from `src/fomo/urls.py`, mirrors `calendar_urls.py` inclusion pattern |
| `templatetags/campaign_extras.py` (new) | PII-visibility gate (`contact_visible`) and status→label/color mapping for table/badge rendering | Templates only; mirrors `calendar_display_extras.py`'s tag-library pattern |
| `management/commands/import_campaign_csv.py` (new) | One-off CLI ingest of the real 3I/ATLAS sheet CSV export into `CampaignRun` rows | `CampaignRun`, `tom_targets.Target` (lookup by name), `Observatory` (site validation) |
| `calendar_utils.insert_or_create_calendar_event` (existing, reused) | Projects an approved `CampaignRun` onto the calendar as a `CalendarEvent`, no-churn | Called from `campaign_views.py` on approval transition, not from a periodic sync command |
| `telescope_runs.sun_event`/`get_site` (existing, reused) | Per-site dark-window times for coverage-gap comparison | `Observatory` model; explicitly does **not** import `ephem_utils` |
| `apps.py` hooks (modified) | Add a second `target_detail_buttons()` entry ("Campaign Runs") and a new `nav_items()` method ("Campaigns") | TOM Toolkit's hook-running machinery (`run_hook`), same mechanism as the existing "Make Ephemeris" button |

## Key Architecture Decisions

### 1. First-class model, not a `CalendarEvent` sidecar

**Decision:** `CampaignRun` is a standalone model with its own table and a required `ForeignKey` to `Target`, plus an optional `ForeignKey`/`OneToOneField` to `CalendarEvent`.

**Why not the `CalendarEventTelescopeLabel` sidecar pattern:** that pattern (`OneToOneField(primary_key=True)` on a third-party model) exists specifically to attach *derived metadata about an event that is guaranteed to already exist* (verification status of a telescope label) without touching `tom_calendar`'s migrations. Campaign runs are the opposite: they are the **primary submitted record**, most of them (per the real 3I sheet — FTN/MuSCAT3, Palomar P200/NGPS, VLT/MUSE) come from facilities **outside** FOMO's sync commands and will often have no corresponding `CalendarEvent` at all (pending review, calendar-only-optional, or a facility FOMO never syncs). Verified in codebase: `CalendarEvent` (`tom_calendar/models.py`, read directly) has no `Target` FK today — only `target_list` (a `TargetList` FK) — so there is no way to anchor "all runs for object X" through `CalendarEvent` alone, confirming the seed's own observation.

**Confidence:** HIGH — read `tom_calendar/models.py` directly; `CalendarEvent` fields confirmed (`title`, `description`, `start_time`, `end_time`, `url`, `target_list`, `user`, `proposal`, `telescope`, `instrument`, `created`, `modified` — no `target`).

### 2. Calendar surfacing: generate a paired `CalendarEvent`, reuse `insert_or_create_calendar_event`

**Decision:** When a `CampaignRun` passes approval **and** has `telescope` + a date range populated, call the existing `insert_or_create_calendar_event(lookup, fields)` helper to create/update a paired `CalendarEvent`, keyed on a synthetic URL like `CAMPAIGN:{campaign_run.pk}` (same idiom as `GEM:{prog}/{observation_id}` in `sync_gemini_observation_calendar.py`). Store the resulting event's PK back on `CampaignRun.calendar_event` so re-approval/edits update in place through the same no-churn helper, and so the table view can link out to the calendar.

**Why not render independently:** v1.4-v1.6 already built a whole visual language for `CalendarEvent` — proposal-keyed WCAG-AA color hashing (`calendar_display_extras.py`), status box-shadow rings, click-to-filter legend, N+1-safe prefetch (`fomo_render_calendar`). A parallel rendering path for campaign runs would fork that language and immediately drift. Projecting into `CalendarEvent` gets all of that for free.

**Why triggered from the approval view, not a management command:** every existing sync command (`sync_lco_observation_calendar`, `sync_gemini_observation_calendar`) exists because its source of truth is an **external** system polled periodically (LCO/Gemini portals). `CampaignRun` originates **inside** FOMO via the submission form — there is no external system to poll, so a periodic sync command would just be a slower, indirect way of reacting to an in-app state transition. Call `insert_or_create_calendar_event` directly from the approval view's `form_valid()` (or a `CampaignRun` status-transition method). No `@receiver`/signal precedent exists anywhere in `solsys_code/` today (verified — `grep -rn "@receiver\|post_save.connect" solsys_code/` returns zero matches) — stay consistent with that and use an explicit call, not an implicit signal.

**Confidence:** HIGH — `insert_or_create_calendar_event` signature and no-churn contract read directly from `solsys_code/calendar_utils.py`; signal-usage claim verified by direct grep.

### 3. Approval state: a single `status` field, not a separate pending model

**Decision:** `CampaignRun.status` is one `CharField` with choices spanning both the approval gate and the observing lifecycle: `PENDING_REVIEW`, `APPROVED`, `OBSERVED`, `REDUCED`, `PUBLISHED`, `REJECTED`. "Visible on the public table/calendar" is simply `status not in {PENDING_REVIEW, REJECTED}` — no second table, no copy-on-approve step.

**Why not a separate `RunSubmission` staging model** (the seed's original open question): a copy-on-approve design means two schemas to keep in sync and a migration/copy step at approval time — extra surface area for the same information. The seed's own field inventory already frames status as a lifecycle (`planned → observed → data reduced → published`); folding "pending review" and "rejected" into that same enum costs one extra pair of choices, not a new model. External validation: this single-status-field pattern is the common baseline in Django moderation write-ups (see Sources) — dedicated moderation packages exist mainly to generalize across *many* models, which FOMO doesn't need for one campaign-run type.

**Confidence:** MEDIUM — codebase precedent (`[QUEUED]`/`[UNVERIFIED]` title-prefix vocabulary already living as data on `CalendarEvent`, not a parallel table) is suggestive but this is a genuinely new area of the codebase with no direct model precedent to grep for; the general Django-community pattern is corroborating, not codebase-verified.

### 4. PII gating: view-level `is_staff` gate for the queue, template-tag gate for the table

**Decision:** Two different mechanisms for two different problems:
- **Approval queue view** (`/campaigns/review/`): gate the whole view with Django's built-in `UserPassesTestMixin`/`is_staff` check (or ship it as a Django admin `ModelAdmin` with an "Approve selected" action first, deferring a custom queue UI to a later increment). This is an all-or-nothing per-view gate, not a per-object permission — `django-guardian` (already installed and configured — `guardian.backends.ObjectPermissionBackend` in `AUTHENTICATION_BACKENDS`, used by `tom_targets` for per-object target permissions) is the wrong tool here because there is no "which users own which campaign run" question, only "is this user staff."
- **Per-target campaign table** (public, `AUTH_STRATEGY='READ_ONLY'` compatible): gate only the `contact_email`/`contact_person` **column**, via a new `campaign_extras.contact_visible(user)` template tag — mirrors the existing `calendar_display_extras.py` tag-library convention (`proposal_color`, `status_border_css`, `text_color_for_bg`) rather than inline `{% if request.user.is_authenticated %}` scattered across the template. Keeps the PII policy in one testable, greppable place.

**Why not row-level guardian permissions:** `solsys_code/` has zero existing `assign_perm`/object-permission calls (verified — guardian is used internally by `tom_targets` for `Target` visibility, but nothing FOMO-authored in `solsys_code` extends that pattern). Introducing it here for a binary "staff can see PII, everyone else can't" rule would be new machinery for a problem `is_staff`/a template tag already solves.

**Confidence:** MEDIUM — `AUTH_STRATEGY='READ_ONLY'`, `TARGET_PERMISSIONS_ONLY=True`, and `guardian` installation/backend registration confirmed directly in `src/fomo/settings.py`; the specific gating mechanism recommendation is an architectural choice informed by, not dictated by, that config — the seed itself flags this as an open question, so confirm the exact policy during phase discussion.

### 5. Coverage-gap analysis: reuse `telescope_runs.py`, do not touch `ephem_utils.py`

**Decision:** Build the coverage-gap view on `telescope_runs.sun_event()`/`get_site()` (dark-window times per site/date, already precise to ≤2 min vs. skycalc) crossed against `CampaignRun.obs_date_start/end` per `(Target, site)`. For v2.0, treat "observable" as "the site is in its −15° dark window on that date and a campaign for this target is otherwise active" rather than computing true target altitude/airmass — the interstellar-object use case is a short, already-known visibility window, so per-site dark-window coverage is a reasonable and cheap first cut.

**Why this must not import `ephem_utils`:** importing `solsys_code.ephem_utils` (transitively, anything importing `solsys_code.views`) triggers `fomo_furnish_spiceypy()` at module load — a ~1.6 GB SPICE kernel download and ASSIST ephemeris build on first use (documented in `CLAUDE.md`, confirmed by the module's own docstring/comment). A campaign coverage-gap **view**, hit on ordinary page loads, must never pay that cost implicitly. True target-altitude filtering (the more accurate version) is exactly the kind of thing that *would* need `ephem_utils`, which is precisely why the milestone context scopes coverage-gap last and allows deferral to v2.1 — flag this as needing its own phase-specific research spike before committing to "true observability" scope; do not silently reach for `ephem_utils` to get there.

**Confidence:** HIGH — the SPICE download trigger and its avoidance are directly documented in `CLAUDE.md` and mirrored by `telescope_runs.py`'s existing design (it already deliberately avoids this import, per `.planning/PROJECT.md`'s Context section: "`telescope_runs.py` avoids importing `solsys_code.ephem_utils`").

### 6. Integration hooks

**Decision:** Extend `solsys_code/apps.py`'s `SolsysCodeConfig`:
- Add a second entry to the list returned by `target_detail_buttons()` (already a list, currently one entry — the "Make Ephemeris" button) for a "Campaign Runs" button linking to `/campaigns/target/<pk>/`, mirroring the existing `ephem_button.html` partial + `src.templatetags.solsys_code_extras.ephem_button` context pattern.
- Add a new `nav_items()` method returning `[{'partial': 'solsys_code/partials/campaign_nav_item.html'}]` — this hook exists and is used by `tom_calendar.apps.TomCalendarConfig.nav_items()` (confirmed by reading `tom_calendar/apps.py`: `return [{'partial': 'tom_calendar/partials/navbar_item.html'}]`); `solsys_code/apps.py` does not implement `nav_items()` yet (it only has `target_detail_buttons()` and `data_services()`), so this is a net-new method, not a modification of an existing one.

**Confidence:** HIGH — `target_detail_buttons()` return shape and current `SolsysCodeConfig` contents read directly from `solsys_code/apps.py`; `nav_items()` hook existence and return shape confirmed by reading `tom_calendar/apps.py`.

## Patterns to Follow

### Pattern 1: No-churn create-or-update for the calendar projection

**What:** Reuse `calendar_utils.insert_or_create_calendar_event(lookup, fields)` verbatim for the `CampaignRun` → `CalendarEvent` projection, exactly as `sync_lco_observation_calendar`/`sync_gemini_observation_calendar`/`load_telescope_runs` already do.

**When:** Any time a `CampaignRun`'s calendar-relevant fields (telescope, date range, status) change after approval.

**Example:**
```python
event, action = insert_or_create_calendar_event(
    lookup={'url': f'CAMPAIGN:{campaign_run.pk}'},
    fields={
        'title': f'{campaign_run.target.name} — {campaign_run.telescope}',
        'start_time': campaign_run.obs_date_start,
        'end_time': campaign_run.obs_date_end,
        'telescope': campaign_run.telescope,
        'instrument': campaign_run.instrument,
        'proposal': campaign_run.proposal_code or '',
    },
)
campaign_run.calendar_event = event
campaign_run.save(update_fields=['calendar_event'])
```

### Pattern 2: Template-tag-based cross-cutting display logic

**What:** New display concerns (PII visibility, status→badge color) belong in a tag library (`campaign_extras.py`), not inline template conditionals — following `calendar_display_extras.py`'s `proposal_color`/`status_border_css`/`text_color_for_bg` precedent.

**When:** Any rendering decision that depends on more than the object being rendered (e.g., current user, WCAG contrast) or that needs a dedicated unit test.

### Pattern 3: CLI-first CSV ingest, validated against real data before UI is built

**What:** Ship `import_campaign_csv` (a `BaseCommand`, mirroring `load_telescope_runs.py`'s structure: `add_arguments` → per-row parse → per-row `(ValueError, Target.DoesNotExist)` skip-and-log, never abort-on-first-error) and run it against the real 3I/ATLAS sheet export **before** building the table/form views on top of a guessed schema.

**When:** This is a deliberate build-order choice (see below), not just a nice-to-have — the codebase's own history (v1.2's `SITE_TELESCOPE_MAP`/`_extract_instrument` shipped against assumed data shapes, requiring the v1.3 follow-up once real `ObservationRecord` rows were checked) shows what happens when schema assumptions aren't validated against real data before building on top of them. Validating `CampaignRun`'s schema against the real sheet early avoids repeating that.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Importing `solsys_code.ephem_utils` (or `solsys_code.views`) from any campaign module

**What happens:** Any module-level or view-level `import` that transitively pulls in `ephem_utils` triggers a ~1.6 GB SPICE kernel download on first use, and loads a module-level Sorcha/ASSIST ephemeris object.

**Why bad:** A page load (campaign table view, coverage-gap view) is not an acceptable place to pay a multi-GB one-time download cost; it would also make `./manage.py test solsys_code` slow for every campaign-related test unless carefully isolated.

**Instead:** Use `telescope_runs.py`'s `sun_event()`/`get_site()` for anything site/dark-window related; if true target-altitude ephemeris is needed later, isolate it behind an explicit, documented, separately-tested code path (own phase, own research) rather than an incidental import.

### Anti-Pattern 2: Widening `CalendarEvent` with campaign-specific fields

**What happens:** Adding `contact_email`, `outcome`, `publication_plans`, etc. directly onto `tom_calendar.CalendarEvent` (a third-party model) to avoid a new model.

**Why bad:** `CalendarEvent` is shared infrastructure for classical-schedule, LCO/SOAR, and Gemini sync — none of those consumers need or want campaign-specific/PII fields on every row, and it would require patching a third-party model's schema (exactly what the `CalendarEventTelescopeLabel` sidecar was built to avoid doing for a much narrower case).

**Instead:** `CampaignRun` as its own model with an optional link, per Decision 1 above.

### Anti-Pattern 3: `django-guardian` object permissions for the PII/approval gate

**What happens:** Reaching for `assign_perm`/per-object guardian permissions to decide who can see `contact_email` or who can approve a run.

**Why bad:** Guardian solves "which of N users/groups can access this specific object" — there is no such multi-owner requirement here, only a binary staff/non-staff split. It would add a permission-assignment step to every submission/approval flow for no behavioral benefit, and there's no existing `solsys_code`-authored code using it (it's used internally by `tom_targets`, not by any FOMO-authored code).

**Instead:** `is_staff`/`UserPassesTestMixin` for the approval queue view; a template tag for the PII column, per Decision 4.

## Scalability Considerations

Campaign volume is inherently low (one active interstellar-object campaign is a rare, bounded event — the 3I/ATLAS sheet this replaces had on the order of tens of rows), so the usual high-traffic scale tables don't really apply. The one real constraint is per-page-load cost:

| Concern | At current scale (tens of runs) | If it ever grows (hundreds of runs, multiple concurrent campaigns) |
|---------|----------------------------------|----------------------------------------------------------------------|
| Per-target table view queries | Single `CampaignRun.objects.filter(target=...)` query, no prefetch needed | Add `select_related('target', 'calendar_event')` if the table starts rendering event links inline (same N+1 lesson already learned in DISPLAY-09/`fomo_render_calendar`) |
| Coverage-gap computation | Compute on request (small date ranges, single site lookups via `sun_event`) | If it starts scanning many targets × many sites × wide date ranges, memoize per-site dark-window results (they don't depend on the target) rather than recomputing `sun_event` per row |
| Calendar projection writes | One `insert_or_create_calendar_event` call per approval transition, synchronous | Still fine synchronous at this volume; no queueing/async infra needed for v2.0 |

## Suggested Build Order

1. **`CampaignRun` model + migration** — everything else depends on this; no external consumers yet, so it's the cheapest place to get the schema right and iterate.
2. **`import_campaign_csv` bootstrap command** — validates the model's field shapes against the *real* 3I/ATLAS sheet data before any UI is built on top of guessed columns (mirrors the lesson already learned in this codebase's own history: shipping against assumed data shapes before checking real records caused a follow-up milestone in v1.2→v1.3). Cheap to build (CLI-only, no auth/PII-display concerns yet) and produces real fixtures for every later phase's tests.
3. **Per-target campaign table view (read path)** — lowest-risk UI increment; surfaces value immediately (the actual replacement for the spreadsheet), and is a natural place to build/test the PII-gating template tag before the submission form needs the same status vocabulary.
4. **Submission form + approval queue (write path)** — depends on the model (step 1) and benefits from having something real to look at already (step 3); this is the step where `status` transitions and the staff-gated queue view get built.
5. **Calendar projection wiring** — depends on approval existing (step 4) since the trigger is "status transitions to approved-and-schedulable"; low risk because it's pure reuse of `insert_or_create_calendar_event`.
6. **Coverage-gap analysis** — explicitly scoped last per the milestone context ("so it can defer to v2.1 if needed"); depends on steps 1 and 3 for data, and needs its own phase-specific research spike to settle how much of "observability" beyond per-site dark windows is worth building without touching `ephem_utils`.

## Integration Points

### External Services

None new. Unlike v1.7's ESO spike, this milestone introduces no new external API dependency — the CSV import is a one-off local file read, and the submission/approval flow is entirely in-app.

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `campaign_views.py` ↔ `CampaignRun` | Direct ORM (`objects.filter(target=...)`, `objects.create(...)`) | Standard Django CBV pattern, same shape as `MakeEphemerisView`/`CreateObservatory`. |
| `campaign_views.py` (approval transition) ↔ `calendar_utils.insert_or_create_calendar_event()` | Direct function call, `(lookup, fields)` dicts | Unchanged contract — no modification needed to this function for campaign runs; same reuse pattern already proven across LCO/SOAR/Gemini. |
| `import_campaign_csv` ↔ `tom_targets.Target` | ORM lookup by name/designation, skip-and-log on `Target.DoesNotExist` | Mirrors `load_telescope_runs.py`'s per-line `(ValueError, Observatory.DoesNotExist)` skip-and-log discipline (D-02 precedent) — a malformed/unmatched CSV row should not abort the whole import. |
| Coverage-gap view ↔ `telescope_runs.sun_event()`/`get_site()` | Direct function call | Must not transitively import `ephem_utils`/`views.py` — see Anti-Pattern 1. |

## Sources

- `tom_calendar/models.py` (installed package, read directly) — confirms `CalendarEvent` has no `Target` FK, only `target_list`
- `tom_calendar/apps.py` (installed package, read directly) — confirms `nav_items()` hook shape
- `solsys_code/apps.py`, `solsys_code/models.py`, `solsys_code/calendar_utils.py` (this repo, read directly) — existing integration-hook, sidecar-model, and no-churn-projection patterns
- `src/fomo/urls.py` (this repo, read directly) — confirms `calendar_urls.py` inclusion pattern to mirror for `campaign_urls.py`
- `src/fomo/settings.py` (this repo, read directly) — `AUTH_STRATEGY='READ_ONLY'`, `TARGET_PERMISSIONS_ONLY=True`, `TARGET_DEFAULT_PERMISSION='OPEN'`, `guardian` installation and `ObjectPermissionBackend` registration confirmed
- `solsys_code/management/commands/load_telescope_runs.py` (this repo, read directly) — CLI ingest structure to mirror for `import_campaign_csv`
- `.planning/PROJECT.md`, `.planning/seeds/target-linked-run-submission-form.md` (this repo) — milestone scope, field inventory from the real 3I/ATLAS sheet, open questions
- `CLAUDE.md` (this repo) — SPICE kernel download side effect of importing `ephem_utils`
- [Model Approval Workflow – Django (B2's Tech Blog)](https://b2techblog.wordpress.com/2020/09/08/model-approval-workflow-django/) — general single-status-field moderation pattern, MEDIUM confidence (not project-specific)
- [django-moderation (PyPI)](https://pypi.org/project/django-moderation/) — corroborates status-field-with-admin-actions as the common baseline; not recommended for adoption here (overkill for one model type)

---
*Architecture research for: Campaign coordination for rare/urgent Solar System objects (FOMO v2.0)*
*Researched: 2026-07-02*
