# Research Summary — FOMO v2.0 Campaign Coordination

**Project:** Campaign Coordination for Rare/Urgent Solar System Objects (FOMO v2.0)
**Domain:** Django + TOM Toolkit web app — adding campaign-coordination data model, community submission, admin approval, and coverage-gap analysis
**Researched:** 2026-07-02
**Confidence:** HIGH

## Executive Summary

FOMO v2.0 adds campaign coordination to replace an ad-hoc Google Sheet (3I/ATLAS reference) with a proper data model and workflow. A rare interstellar object passes through the inner solar system once; dozens of observatory teams want to observe it; today they coordinate via a link-shared spreadsheet. FOMO's solution: a `CampaignRun` model tracking observing runs per target, a community submission form with admin approval, per-target campaign table view surfacing what's already claimed, and (as a stretch goal) ephemeris-aware coverage-gap analysis showing unobserved nights. The recommended approach reuses already-installed Django tooling (`django-tables2`, `django-crispy-forms`, `pandas`, `astropy`) — no new packages needed — and mirrors established patterns in this codebase (sidecar models, CSV import commands, template-tag display logic, calendar projection). The key risks are two-fold: (1) silent PII exposure on an `AUTH_STRATEGY='READ_ONLY'` public site (must gate contact fields explicitly), and (2) messy real-world CSV data from the sheet (must validate schema against the actual file before building UI on top, echoing a v1.2→v1.3 lesson already learned the hard way).

The milestone is achievable as five moderately-scoped phases: model + import (validation gate), read path (table), write path (form + queue), calendar integration, and gap analysis. The first three are core to launch; gap analysis is explicitly scopable to v2.1 if time runs short, and the research flags this phase as needing its own spike (to clarify whether per-site dark-window gaps or true target-altitude ephemeris is worth the `ephem_utils` import cost).

## Key Findings

### Recommended Stack

**No new dependencies required.** Every core feature is buildable with Django 5.2, `django-tables2` 3.0, `django-filter` 24.3, `django-crispy-forms` 2.4, `pandas` 2.3, `astropy` 6.1, and standard library—all already installed and proven across this codebase. The one nuance: `coverage-gap analysis` should use `telescope_runs.py`'s lightweight `sun_event()`/`get_site()` helpers, **not** `ephem_utils`, which downloads ~1.6 GB of SPICE kernels at import time and would make the campaign table view unacceptably slow (confirmed in CLAUDE.md; see Pitfalls below).

**Core technologies:**
- **Django 5.2 + Django ORM** — `CampaignRun` model (target FK, status field, PII contact columns) and migrations; standard CBV pattern for forms/views
- **django-tables2 3.0** — per-target campaign table (sortable/paginated, no custom rendering framework needed)
- **django-crispy-forms 2.4 + crispy-bootstrap4** — community submission form, matching existing ephemeris/observatory form styling
- **pandas 2.3** — robust CSV parsing for the one-off 3I bootstrap import (handles Sheets export quirks: BOM, blank rows, NaN)
- **astropy 6.1** — sun-event and dark-window calculations (coverage-gap baseline, already used in `telescope_runs.py`)

### Expected Features

**Must have (table stakes):**
- `CampaignRun` data model — target FK, telescope/instrument/site, obs date + UT range, status (planned→observed→reduced→published), contact person/email (PII-guarded), comments, outcome, publication plans
- One-off CSV bootstrap import of the real 3I/ATLAS sheet — essential to validate schema against real messy data before UI is built
- Per-target campaign table view — the actual spreadsheet-replacement deliverable (coordinator sees all runs for one target)
- Community submission form — web form replaces direct sheet editing; Target mandatory, everything else optional/free-text
- Admin approval gate — submissions invisible until approved; prevents spam/errors polluting public view

**Should have (competitive differentiators):**
- Ephemeris-aware coverage-gap analysis — the key value-add over the original sheet; surfaces "observable but unclaimed dates" (reuses `telescope_runs.py`, doesn't touch `ephem_utils`)
- "Open to collaboration?" flag — filterable in table; lets searchers find runs that want help
- Admin notification on new submission — prevents queue from silently filling unmonitored
- Self-service approval bypass for trusted PIs — reduces admin bottleneck for known contributors (requires submitter-trust matching, flagged as open design question)

**Defer to v2.1+ or anti-features:**
- Full data-file/product upload (ExoFOP-style archive) — turns FOMO into data-storage system; 3I sheet is metadata-only
- Generic "any facility can push status via bot" layer (TNS-style) — campaign runs are the *non-synced-facility* case
- Booking/locking coverage gaps as reservable slots — coverage-gap is advisory display, not a reservation system
- Heavyweight third-party moderation package — single `status` field + Django admin action sufficient

### Architecture Approach

`CampaignRun` is a first-class model (not a sidecar on `CalendarEvent`, which has no `Target` FK) with an optional link to `CalendarEvent` for calendar surfacing. When approved **and** telescope/dates are present, the view calls `insert_or_create_calendar_event()` (reused from LCO/Gemini sync commands) with a namespaced key (`CAMPAIGN:{pk}`) to project the run onto the calendar without risking collisions with synced events. The feature reuses existing Django patterns throughout: sidecar-model precedent from `CalendarEventTelescopeLabel`, template-tag display logic from `calendar_display_extras.py`, CSV ingest structure from `load_telescope_runs`, and PII gating via view-level context filtering + template tags (avoiding `django-guardian`, which is not needed for a binary staff/non-staff split).

**Major components:**
1. `CampaignRun` model — persistent storage, target FK, status lifecycle, PII fields
2. `campaign_views.py` (CBV + forms) — per-target table (read), submission form + approval queue (write), optional coverage-gap view
3. `import_campaign_csv` command — one-off CLI ingest, skip-and-log error handling, validates model schema against real data
4. `campaign_extras.py` (template tags) — PII visibility gate, status→badge rendering
5. Calendar projection trigger — approval view calls `insert_or_create_calendar_event()` with `CAMPAIGN:` namespace
6. `apps.py` hooks — target-detail "Campaign Runs" button + navbar "Campaigns" item

### Critical Pitfalls to Avoid

1. **PII in demo-notebook output** — The `import_campaign_csv` command will receive a paired demo notebook per CLAUDE.md convention, but that notebook's output cells will contain real contact names/emails from the 3I sheet. Decide upfront: redact the notebook's displayed output, use a synthetic/fake-data fixture instead, or get explicit exception-to-convention approval. Do not commit real people's emails into git history by default.

2. **PII exposure on the campaign table** — FOMO's `AUTH_STRATEGY='READ_ONLY'` means unauthenticated users can view target pages. Unless contact email/name are explicitly excluded from the view's context for anonymous requests *and* verified by an anonymous-client test, they will be visible to any web crawler. Gating must happen at the view layer, not just in the template.

3. **Collision with existing calendar sync commands** — `insert_or_create_calendar_event()` is shared by LCO, Gemini, and classical-schedule sync commands, each with carefully-chosen lookup keys. If campaign code writes to `CalendarEvent` directly or reuses a key scheme, re-approval or concurrent syncs will create duplicates. Route through the helper with a distinctly-namespaced key (`CAMPAIGN:{pk}`).

4. **Transitive SPICE import cost** — Importing `solsys_code.ephem_utils` downloads ~1.6 GB of SPICE kernels at module load. Coverage-gap analysis should use `telescope_runs.py` (lightweight, `astropy`-only) for per-site dark windows. If full target-altitude ephemeris becomes unavoidable later, import `ephem_utils` lazily inside the function, not at module scope.

5. **Messy CSV import schema mismatches** — The real 3I sheet has free-text date formats, status vocab that won't cleanly map to lifecycle states, multi-observer emails comma-separated in one cell, and (typical of Sheets exports) blank header rows or merged cells. Download and inspect the real CSV first; build the importer with per-row try/except that logs and skips on unparseable rows; use `pandas.to_datetime(..., errors='coerce')` for date tolerance.

6. **Approval-queue race conditions** — Two admins double-clicking "approve" on the same pending submission can create duplicates. Use `CampaignRun.objects.filter(pk=pk, status='pending').update(status='approved')` (atomic conditional update) or `select_for_update()` inside `transaction.atomic()`. Write a test that calls approval twice on the same record and asserts the second is a no-op.

7. **Silent rejection + spam exposure** — The public submission form needs at least a honeypot field to deflect cheap spam traffic. Admin notification (a simple email) prevents the queue from silently filling unmonitored. Rejected submitters should get *some* visibility (private status-check link) so they can tell a submission was actually seen.

## Implications for Roadmap

### Suggested Phase Structure

**Phase 1: `CampaignRun` Data Model + Bootstrap CSV Import**
- **Rationale:** Everything downstream depends on the model; CSV import validates schema against real data before any UI is built
- **Delivers:** `CampaignRun` model with all fields; migration; `import_campaign_csv` management command; real data fixtures from the 3I sheet
- **Avoids:** Pitfall 5 (messy CSV) — must include explicit real-file inspection + dry-run/skip-and-log reporting
- **Design decision:** Resolve Pitfall 1 (demo-notebook PII strategy) in phase discussion before code starts

**Phase 2: Per-Target Campaign Table View (Read Path)**
- **Rationale:** Lowest-risk UI; surfaces value immediately (spreadsheet replacement); good place to test PII-gating template tag before submission form needs it
- **Delivers:** Per-target campaign table (sortable/paginated, `django-tables2`), linked from target-detail page; contact email gated to staff-only via template tag and view-level context filtering
- **Avoids:** Pitfall 2 (PII exposure) — contact email must be excluded from view context for anonymous requests and verified by anonymous-client test
- **Research flags:** None; `django-tables2` established. Verify PII-gating strategy in phase discussion if Pitfall 2 policy not already resolved.

**Phase 3: Submission Form + Approval Queue (Write Path)**
- **Rationale:** Depends on model (Phase 1); benefits from having a live table (Phase 2) so admins can see new submissions appear
- **Delivers:** Community submission form (Target mandatory, rest optional); approval-queue view (staff-only); Django admin action for bulk approve; admin notification email; honeypot field for spam prevention
- **Avoids:** Pitfall 6 (race conditions) — approval must use conditional update, tested with double-approval; Pitfall 7 (silent rejection + spam) — honeypot + notification must be present in v1
- **Research flags:** **Moderate** — admin notification and honeypot are straightforward; confirm PII-policy if not settled in Phase 2. Submitter status-check is optional but recommended.

**Phase 4: Calendar Projection Wiring**
- **Rationale:** Depends on approval existing (Phase 3); low risk because it reuses `insert_or_create_calendar_event()` unchanged
- **Delivers:** When a `CampaignRun` transitions to `APPROVED` **and** has telescope + date range, create/update paired `CalendarEvent` with key `CAMPAIGN:{campaign_run.pk}`
- **Avoids:** Pitfall 3 (calendar collisions) — no direct `CalendarEvent.objects.create()`; must use shared helper with distinct namespace
- **Research flags:** None; reuse pattern proven across LCO/Gemini/classical commands. Optional polish if time short.

**Phase 5: Ephemeris-Aware Coverage-Gap Analysis (Stretch Goal / Deferrable to v2.1)**
- **Rationale:** Scoped last per milestone context ("can defer to v2.1 if needed"); depends on working data model + table view
- **Delivers:** View showing observable-but-unclaimed dates for a target + site; reuses `telescope_runs.sun_event()`/`get_site()` for dark-window times; cached or explicitly-triggered (not computed inline)
- **Avoids:** Pitfall 4 (SPICE import) — must **not** import `ephem_utils` at module scope; never compute inline; explicit user action or cached results only
- **Research flags:** **HIGH — requires dedicated research spike.** Key question: is per-site dark-window coverage sufficient for v2.0, or does the feature need true target-altitude/airmass filtering (which pulls in `ephem_utils`)? This decision gates whether gap analysis ships in v2.0 or defers to v2.1.

### Phase Ordering Rationale

Model + import first (no external consumers, cheap iteration); read path before write path (table is simpler, admins see data working before public form); write path before calendar (approval workflow must exist before projection is triggered); calendar before coverage-gap (gap analysis needs both claimed and observable sides stable). Gap analysis is last and deferrable per scope context.

### Research Flags

| Phase | Flag | Reason |
|-------|------|--------|
| 1 (Model + Import) | Design decision: Pitfall 1 (demo-notebook PII strategy) | Decide before code starts whether notebook redacts PII, uses synthetic fixture, or gets convention exception |
| 2 (Table View) | Policy confirmation: Pitfall 2 (PII visibility gate) | Confirm auth-gated vs. opt-in vs. store-never-render approach if not settled in Phase 1 |
| 3 (Form + Queue) | Best-practice: Pitfalls 6 + 7 (race conditions, spam) | Conditional-update approval + honeypot + notification are straightforward but must be in v1 |
| 4 (Calendar Integration) | None | Established reuse pattern; no research needed |
| 5 (Coverage-Gap Analysis) | **HIGH PRIORITY SPIKE:** Pitfall 4 (SPICE cost) | Dark windows only vs. true target-altitude filtering? This settles whether gap analysis is v2.0 or v2.1 scope |

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | HIGH | All packages already installed and verified (`pip show` 2026-07-02). No incompatibilities. No new packages needed. |
| **Features** | MEDIUM-HIGH | Must-have features sourced from real 3I sheet's field inventory (HIGH). Differentiators are solid patterns (MEDIUM) but some are open policy questions needing phase discussion. |
| **Architecture** | HIGH | Core decisions sourced from direct code inspection. `CalendarEvent` has no `Target` FK confirmed by reading source. No assumptions. |
| **Pitfalls** | HIGH | Facts sourced from direct code inspection (CLAUDE.md, SPICE cost, OPEN/READ_ONLY settings, calendar patterns, v1.2 CSV history) or established Django best-practices consistent across sources. |
| **Overall** | **HIGH** | Solid research with clear dependencies. Main unknowns (PII policy, coverage-gap altitude scope) are flagged as phase-discussion decisions, not research gaps. |

### Gaps to Address

1. **PII-display policy** — Exact approach for contact email/name visibility. Should be resolved in Phase 1 or Phase 2 planning. Recommend default: auth-gated (staff only), with opt-in checkbox on submission form as future enhancement.

2. **Submitter-trust criteria for PI approval bypass** — If self-service approval is scoped into v2.0, what matches a submitter to their credentials? Only needed if this optional feature is in scope.

3. **Coverage-gap altitude scope** — Is per-site dark-window coverage sufficient for v2.0, or does the feature require true target RA/Dec + airmass filtering? Phase 5's research spike settles this.

4. **Demo-notebook redaction strategy** — Before bootstrap-import phase is coded, resolve whether notebook will redact/synthesize PII columns, use synthetic fixtures, or get convention exception documented in CLAUDE.md.

## Sources

### Primary (HIGH confidence)

- `.planning/PROJECT.md` — Milestone scope, v1.2→v1.3 CSV lesson, existing `insert_or_create_calendar_event` pattern, `CalendarEventTelescopeLabel` sidecar precedent
- `.planning/seeds/target-linked-run-submission-form.md` — Enriched seed with real 3I/ATLAS field inventory, PII framing, submission/approval shape
- `CLAUDE.md` — Demo-notebook convention and Pitfall 1 enforcement gaps, `AUTH_STRATEGY='READ_ONLY'`/PII risk, SPICE cost, telescope_runs.py avoidance of `ephem_utils`
- `.planning/codebase/CONCERNS.md` — SQLite concurrency risk, `MakeEphemerisView` blocking cost, calendar-sync visual patterns
- Installed package inventory (`pip show`, 2026-07-02) — All core technologies present and version-compatible
- Direct code inspection: `tom_calendar/models.py` (no `Target` FK), `solsys_code/calendar_utils.py`, `solsys_code/apps.py`, `src/fomo/settings.py`

### Secondary (MEDIUM confidence)

- Competitor analysis: IAWN, ExoFOP-TESS, TNS, YSE-PZ — corroborates "no existing system does exactly this" and feature-combination from multiple reference systems
- Django community best-practices — conditional-update/race-condition handling, honeypot fields, CSV/date parsing tolerances
- [Django Packages: Moderation grid](https://djangopackages.org/grids/g/moderation/) — corroborates avoiding heavyweight moderation packages for single-model use case

---

*Research completed: 2026-07-02*
*All research files (STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md) synthesized*
*Ready for roadmap creation*
