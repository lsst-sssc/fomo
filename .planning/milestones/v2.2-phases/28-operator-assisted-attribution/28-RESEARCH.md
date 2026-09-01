# Phase 28: Operator-Assisted Attribution - Research

**Researched:** 2026-08-01
**Domain:** Django staff worklist (django-tables2/Bootstrap4), `difflib`-based fuzzy scoring, two new typed link/dismissal models, admin `save_formset` audit stamping
**Confidence:** HIGH (all code-location and live-DB claims verified directly by reading the file/running a read-only query against `src/fomo_db.sqlite3`; a small number of design choices Claude retains discretion over are marked ASSUMED/LOW and listed in the Assumptions Log)

## Summary

Phase 28 adds exactly one new write surface (a staff worklist) over data that already exists:
`CampaignRun` (Phase 27), `CalendarEvent`/`CalendarEventMeta` (Phase 27 CANON-03), and
`ObservationRecord`/`CampaignRunObservation` (Phase 27 CANON-04). Nothing about the storage
layer needs inventing from scratch — it needs two small typed dismissal models (D-08), two new
audit fields on `CalendarEventMeta` (D-12), one matcher module producing scored candidates
(D-11), and one staff view/template pair (D-01..D-04) that writes into the two link models
Phase 27 already built. The codebase's own `difflib`-based fuzzy-match precedent
(`campaign_utils.fuzzy_match_candidates()`), its atomic-conditional-`.update()` staff-decision
idiom (`CampaignRunDecisionView`), its `save_formset` audit-stamping idiom
(`CampaignRunAdmin.save_formset`), and its multi-table staff-page idiom
(`approval_queue.html`/`ApprovalQueueTable`) are all directly reusable precedents — this phase
should extend them, not invent parallel ones.

The live dev DB was queried read-only and confirms `CampaignRun` pk=1 (FTS/MuSCAT4, Didymos 2026
campaign, 2026-07-07..2026-07-21, site=Observatory E10/Siding Spring, approved, source=legacy) is
real, and that all 11 real LCO calendar events and all 11 real matching `ObservationRecord`s exist
with the instrument/date signatures CONTEXT.md describes. Two corrections to CONTEXT.md's framing,
both discovered by the live query, matter for planning: (1) **one of the 11 calendar events
(pk=53) already carries a correct `CalendarEventMeta.run=1` link** — only 10 of the 11 are true
orphans today, not 11; and (2) **one of the 11 matching `ObservationRecord`s (pk=58) already
carries a `CampaignRunObservation` link, but to the WRONG run** (`CampaignRun` pk=3, campaign
"3I/ATLAS" — a different campaign from Didymos 2026, whose `TargetList` pk=3 does not even contain
the record's target). This pre-existing cross-campaign link is dev-DB noise from before this
phase (out of scope to fix — read-only research), but the acceptance test for criterion 5 must be
written against one of the 10 genuinely-orphaned calendar events and one of the 10
genuinely-orphaned observation records, not against pk=53/pk=58, which the unique constraint
already blocks from re-linking.

A second concrete, code-verified finding matters for D-11's telescope-match signal: there is
**no existing code mapping from `Observatory.obscode` (MPC code, e.g. `E10`) to the LCO 3-letter
site codes `calendar_utils.SITE_TELESCOPE_MAP` uses (e.g. `coj`)** — the two vocabularies are
bridged today only by human-readable code comments ("`'coj'` (Siding Spring)"), never by a
queryable structure. `telescope_runs.SITES` maps a *classical run-file* telescope nickname
(`'FTS'`) to an obscode (`'E10'`) — confirmed to match `CampaignRun.telescope_instrument`'s
`"FTS/MuSCAT4"` prefix exactly — but that dict has only 4 entries and doesn't cover LCO site
codes at all. The matcher's telescope-match signal needs a small new alias table (mirroring the
existing `HORIZONS_OBSERVER_TO_OBSCODE` extension-rule discipline) or must fall back to comparing
`CampaignRun.site` (already resolved for the pk=1 case) against a reverse lookup through
`SITE_TELESCOPE_MAP`. This is the single piece of new "vocabulary bridging" logic this phase
must write that has no existing precedent to copy — everything else does.

**Primary recommendation:** Build one new module `solsys_code/campaign_attribution.py` (peer of
`campaign_utils.py`/`campaign_gap.py`, never a private helper inside `campaign_views.py` — the
milestone reserves `campaign_reconciler.py` for Phase 29's reconciliation logic specifically) that
computes scored `(orphan, candidate_run)` pairs on the fly with no persistence beyond the two new
dismissal models, reuses `difflib.get_close_matches`/`SequenceMatcher` exactly as
`campaign_utils.py` already does, and writes into `CalendarEventMeta.run` /
`CampaignRunObservation` (never a new ownership model) via the existing atomic-conditional-update
idiom `CampaignRunDecisionView` already established.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Candidate matching/scoring | API / Backend (`campaign_attribution.py`) | — | Pure Python over already-loaded querysets; no DB writes, no client logic — same tier as `campaign_utils.fuzzy_match_candidates()` |
| Attribution worklist page | Frontend Server (SSR, Django view + template) | API / Backend | Server-rendered Bootstrap4/django-tables2, same as `approval_queue.html`; no client-side framework, no new JS beyond existing Bootstrap `collapse`/htmx precedent |
| Confirm / dismiss / undo POST actions | API / Backend (Django view) | Database | Atomic conditional `.update()` / `get_or_create` against SQLite, mirroring `CampaignRunDecisionView` |
| Dismissal persistence | Database / Storage | — | Two new typed models with named `UniqueConstraint`s, same tier as `CampaignRunObservation` |
| Audit stamping (confirmed_by/at) | API / Backend (view POST handler + admin `save_formset`) | Database | Both write paths (staff attribution view, admin inline) must stamp identically — this is the tier boundary Phase 27.1's admin fix already established |
| Count banner on campaign list | Frontend Server (template) | API / Backend (shared queryset helper) | Mirrors `runs_needing_site_review()` → `CampaignListView.get_context_data()` → `campaign_list.html` |
| Runbook documentation | Docs (Sphinx) | — | `docs/runbooks/telescope_runs_calendar.rst`, CLAUDE.md paired-docs rule |

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: Orphan-centric, not run-centric.** One row per un-attributed calendar event or
  observation record, expanded to its candidate runs.
- **D-02: A standalone staff page plus a count banner on the campaign list**, reusing the
  27.1-03 shared-queryset-helper mechanism, not a second counting path.
- **D-03: An orphan enters the queue only if it has at least one candidate run** — the
  campaign/target boundary check doubles as the noise filter; no separate exclusion list.
- **D-04: Two tables on one page** ("Calendar events awaiting attribution" /
  "Observation records awaiting attribution"), same shape as `approval_queue.html`'s tables.
- **D-05: A rejected suggestion is persisted, per (orphan, run) pair.** A dismissal is not an
  association.
- **D-06: A dismissal records who, when, and a free-text reason** — mirrors
  `confirmed_by`/`confirmed_at`, adds an optional note.
- **D-07: A dismissal is reversible from a collapsed "Dismissed" section on the same page.**
- **D-08: Two small typed dismissal models, one per orphan kind** — real FKs, named
  `UniqueConstraint` per pair, no `GenericForeignKey`.
- **D-09: Multi-select confirmation, gated to the high-confidence band only.** Checkboxes render
  only on High-band candidates; everything else is single-confirm only.
- **D-10: Score is filtered by named band (High/Medium/Low) and displayed as a number too.**
  Band gates D-09's checkboxes; the number is always additional to the evidence columns.
- **D-11: A pure weighted sum over evidence signals, with the campaign/target boundary as the
  single hard gate.** Telescope match, date overlap, instrument-string similarity all
  contribute; none is individually disqualifying. This is what makes criterion 5 work — a
  design where instrument similarity or exact date-span equality gates would fail it.
- **D-12: `CalendarEventMeta` gains `confirmed_by`/`confirmed_at`**, revisiting Phase 27's D-05.
  The `models.py` D-05 comment ("do not fix it here") must be updated, not left contradicting
  the code.
- **D-13: An undo writes a dismissal row** (not a soft-undo flag, not a log line).
- **D-14: Confirmed associations appear in a "Confirmed" section on the same page** —
  fourth collapsed section beside the two worklists and Dismissed.
- **D-15: "Done" is an empty queue plus a stated remaining count.** No backlog-reporting
  management command (declined).

### Claude's Discretion

- Names of the two dismissal models, the attribution view/URL/template, and the matcher module.
  Milestone reserves `campaign_reconciler.py` for Phase 29; this phase's matcher goes in a peer
  module or `campaign_utils.py`, never a private helper inside `campaign_views.py`.
- The actual weights in D-11's sum and the High/Medium/Low cut-points — subject to D-09's
  warning that the high cut-point is a correctness decision (it gates multi-select), and the
  requirement that criterion 5's real pk=1 case lands high enough to be surfaced.
- How date overlap is scored for a TBD or unresolved window (`window_start`/`window_end`
  nullable on `CampaignRun`; `scheduled_start`/`scheduled_end` nullable on `ObservationRecord` —
  see Runtime evidence below, this applies to 10 of the 11 real records too).
- Pagination, sort order, whether Dismissed/Confirmed paginate separately.
- Test organisation.

### Deferred Ideas (OUT OF SCOPE)

- A backlog-reporting management command (declined under D-15; Phase 29's call if needed).
- A full staff run-detail page (Phase 27 D-06 deferred it here; D-01's orphan-centric queue
  doesn't need one — admin inlines cover it).
- The other three multi-select guardrails (no select-all, checkbox only when exactly one
  candidate, evidence forced inline) — offered under D-09, not chosen, available later.
- Bulk *dismissal* — not in scope; dismissal is per-pair and individual.
- v2.3 items: adapter rewiring (ADAPT-01..03), provenance-blind gap analysis (GAPB-01), status
  vocabulary unification (STATUS-01/02), unused-allocation display (UNUSED-01).
- The reconciler and calendar-event projection (Phase 29).
- Automatic merging of suspected duplicate associations, at any confidence.
- `GenericForeignKey` for either link.
- Any new third-party dependency (`rapidfuzz` rejected twice — `difflib` is the tool).

## Project Constraints (from CLAUDE.md)

- **GSD workflow enforcement**: file edits must go through a GSD command (`/gsd:execute-phase`
  etc.) — the planner should assume execution runs under that discipline; no direct action needed
  in the plan itself beyond noting it.
- **Plain-English planning terminology**: write "create or update" / "find-or-create" instead of
  "upsert" in CONTEXT/RESEARCH/PLAN/PATTERNS docs (this document follows that rule throughout).
- **`NonSiderealTargetFactory` only, never `SiderealTargetFactory`**, when fixturing a `Target` —
  applies to every test this phase adds. Note: the real dev-DB orphan records' targets
  (`Didymos COJ 2026 Field #NN`) are themselves `SIDEREAL` field targets (per-pointing targets
  created by `--create-missing-targets`, distinct from the campaign's own `NON_SIDEREAL` moving
  target) — this is existing, correct, real data, not something new tests should imitate; new
  tests fixturing the *campaign's* target must still use `NonSiderealTargetFactory`.
- **Paired docs are part of the deliverable**: `docs/runbooks/telescope_runs_calendar.rst` is
  required in `files_modified` from the start (already flagged in ROADMAP/CONTEXT). No new demo
  notebook is required — this phase ships no management command.
- **Test invocation**: Django app tests run via `python manage.py test solsys_code` (not
  `./manage.py`, per project memory), excluding `test_views.TestEphemeris` (segfaults in native
  ASSIST — irrelevant to this phase, which does not import `ephem_utils`).
- **Ruff**: single quotes, 120-col line length, Rubin-DM naming exceptions already configured —
  no special handling needed for this phase's new modules.

## Standard Stack

No new dependency is needed or permitted for this phase (`rapidfuzz` rejected twice; the Out of
Scope table forbids "any new dependency for reconciliation or field-diffing" and this phase's
matching is squarely in that family). Every library below is already installed and in production
use elsewhere in this codebase.

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `difflib` (stdlib) | Python 3.11 stdlib | Instrument-string similarity scoring | `[VERIFIED: local codebase]` `campaign_utils.fuzzy_match_candidates()` already wraps `difflib.get_close_matches(cutoff=0.6)` and `substring_or_fuzzy_match_candidates()` layers substring-first matching on top — reuse this shape, don't reinvent |
| `django-tables2` | 3.0.0 | Worklist tables (two orphan tables, Dismissed, Confirmed) | `[VERIFIED: pip show django-tables2]` Already the table library for `CampaignRunTable`/`ApprovalQueueTable`; installed version confirmed in this environment |
| `django-filter` | 24.3 | Optional band-filter control | `[VERIFIED: pip show django-filter]` Already used by `CampaignRunFilterSet`; **caveat below** — this phase's "table" rows are computed candidate pairs, not a plain `CampaignRun` queryset, so a `FilterSet`-backed `FilterView` may not fit cleanly; a plain GET-param band filter handled in the view (like `CampaignGapAnalysisView`'s manual param validation) is the more direct fit. Confirm during planning, don't default to `FilterSet` just because it's the existing pattern for `CampaignRunTable` |
| Bootstrap 4 + `crispy_forms`/`crispy-bootstrap4` | project-pinned | Page layout, badges, collapse sections | `[VERIFIED: local codebase]` Same as every other `campaigns/` template; UI-SPEC already locks every visual token to existing Bootstrap4 utility classes |
| `django-htmx` | 1.27.0 | (not required by this phase, but available) | `[VERIFIED: pip show django-htmx]` UI-SPEC's Multi-Select Form Shape doesn't need htmx (plain multi-row `<form>` POST); only the existing site-search widget uses htmx, unaffected by this phase |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Django's own `messages` framework | Django (project-pinned) | Success/warning/error feedback on confirm/dismiss/undo | Already the pattern in `CampaignRunDecisionView` — UI-SPEC's copywriting contract explicitly mirrors `campaign_views.py:658`/`:660`'s existing message wording |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `difflib` | `rapidfuzz` | Rejected twice at milestone level (v2.1 and this milestone) — no new dependency permitted; `difflib` already proven sufficient for the harder site-name fuzzy-match problem |
| Computed-candidate view over `django-tables2`+`FilterSet` | A materialized `AttributionCandidate` model refreshed by a batch job | Rejected implicitly by D-01/D-05 (candidates are computed on the fly, only dismissals/confirmations persist) — a materialized candidate table would need its own refresh/staleness logic this milestone explicitly avoids (no Celery, no new background job machinery) |

**Installation:** None required — no new packages.

**Version verification:** `pip show django-tables2 django-filter django-htmx` run directly in
this repo's virtualenv (`/home/tlister/venv/devel_fomo311_venv`) on 2026-08-01, confirming
django-tables2==3.0.0, django-filter==24.3, django-htmx==1.27.0 — all already `pyproject.toml`
dependencies, no version bump needed for this phase.

## Package Legitimacy Audit

No external packages are being added by this phase — the Package Legitimacy Gate is not
triggered. All libraries above are already installed project dependencies, verified in the
running virtualenv (not merely asserted). `difflib` is a Python stdlib module, not a registry
package. No `npm view`/`pip index versions` check applies since nothing new is being installed.

**Packages removed due to [SLOP] verdict:** none (none proposed).
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
Staff browser
    |
    | GET /campaigns/attribution/
    v
AttributionQueueView (StaffRequiredMixin, TemplateView)  -- new, campaign_views.py or peer module
    |
    | 1. orphan_calendar_events()  -- CalendarEventMeta.run IS NULL, minus dismissed pairs
    | 2. orphan_observation_records() -- no CampaignRunObservation row, minus dismissed pairs
    v
campaign_attribution.py (new matcher module)
    |
    | for each orphan: candidate_runs(orphan)
    |   1. HARD GATE: campaign/target boundary (same TargetList membership) -- D-11
    |   2. score = w1*telescope_match + w2*date_overlap + w3*instrument_similarity
    |   3. band = High/Medium/Low from score cut-points
    v
Two django-tables2 Tables (or hand-grouped candidate rows per UI-SPEC's
"Candidate Grouping Contract") rendered into approval_queue.html-style template
    |
    | staff ticks checkboxes (High band only) or clicks single Confirm/Dismiss
    v
POST /campaigns/attribution/<pair>/confirm|dismiss|undo/
    |
    |-- confirm (event side):  CalendarEventMeta.objects.filter(event=pk, run__isnull=True)
    |                          .update(run=run_pk, confirmed_by=request.user, confirmed_at=now())
    |                          -- atomic conditional update, mirrors CampaignRunDecisionView
    |
    |-- confirm (record side): CampaignRunObservation.objects.get_or_create(
    |                          observation_record=record, defaults={run, confirmed_by, confirmed_at})
    |                          -- IntegrityError on race -> messages.warning (D-08's UniqueConstraint)
    |
    |-- dismiss (either side): CalendarEventDismissal / ObservationRecordDismissal
    |                          .objects.get_or_create(event=.., run=.., defaults={dismissed_by, dismissed_at, reason})
    |
    |-- undo (D-13):           writes a NEW dismissal row for the pair just un-confirmed,
    |                          then clears CalendarEventMeta.run / deletes the CampaignRunObservation row
    v
Django admin (CampaignRunAdmin.save_formset) -- unchanged code path, but now must ALSO
stamp CalendarEventMeta.confirmed_by/confirmed_at on newly created inline rows (D-12)
    |
    v
campaign_list.html count banner (D-02) <- runs_needing_attribution_count()-style shared helper,
mirrors runs_needing_site_review()
```

### Recommended Project Structure

```
solsys_code/
├── campaign_attribution.py   # NEW: matcher (candidate_runs(), score(), band_for_score()),
│                              #      orphan querysets, dismissal-aware filtering
├── models.py                  # gains 2 dismissal models + 2 CalendarEventMeta fields
├── migrations/
│   └── 0013_..._dismissal_models_and_calendar_event_meta_audit.py  # or split into 2
├── campaign_views.py           # gains AttributionQueueView + 3 POST action views (or one
│                              #   dispatching View, mirroring CampaignRunDecisionView's shape)
├── campaign_tables.py          # gains 2 (or 4, if Dismissed/Confirmed use dedicated tables)
│                              #   new django-tables2 Table classes
├── campaign_urls.py            # gains attribution/, attribution/<pk>/confirm/, /dismiss/, /undo/
├── admin.py                    # save_formset gains a CalendarEventMeta isinstance branch
└── tests/
    ├── test_campaign_attribution.py   # matcher unit tests (scoring, gating, banding)
    └── test_campaign_attribution_views.py  # (or extend test_campaign_approval.py-style file)
        # confirm/dismiss/undo, double-submit, two-staff race, criterion-5 acceptance test

src/templates/campaigns/
├── attribution_queue.html      # NEW: 4 sections per UI-SPEC
docs/runbooks/
└── telescope_runs_calendar.rst # gains "How do I attribute existing events/records to a run?"
```

### Pattern 1: `difflib`-based scoring, reused not reinvented

**What:** `campaign_utils.py:549` `fuzzy_match_candidates()` wraps
`difflib.get_close_matches(text, pool.keys(), n=5, cutoff=0.6)`; `:580`
`substring_or_fuzzy_match_candidates()` layers a case-insensitive substring pre-pass ahead of
the fuzzy fallback, sorted shortest-match-first, because whole-string `SequenceMatcher` ratios
often can't bridge a short acronym against a long official name.

**When to use:** D-11's instrument-similarity signal is exactly this same shape of problem
(`"FTS/MuSCAT4"` vs `"2M0-SCICAM-MUSCAT"`) — reuse `difflib.SequenceMatcher(None, a, b).ratio()`
directly for a continuous 0..1 score (not `get_close_matches`, which is a
best-N-picks-over-a-cutoff API suited to *site* search, not *pairwise* scoring). Measured
evidence below shows the naive whole-string ratio undershoots what a human would call "obviously
the same instrument" — apply the same substring/token-extraction discipline
`substring_or_fuzzy_match_candidates()` already established before falling back to a raw ratio.

**Measured evidence** (`[VERIFIED: local computation against real strings from the live DB]`,
using Python's stdlib `difflib.SequenceMatcher`):

| String A | String B | Raw ratio |
|----------|----------|-----------|
| `"FTS/MuSCAT4"` | `"2M0-SCICAM-MUSCAT"` (real `CalendarEvent.instrument`/`ObservationRecord.parameters['instrument_type']` value) | **0.500** |
| `"FTS/MuSCAT4"` | `"2m0"` (real `CalendarEvent.telescope` value) | 0.143 |
| `"FTS/MuSCAT4"` | `"COJ-2m0"` (real `CalendarEvent.telescope` value on one of the 11) | 0.111 |
| `"MuSCAT4"` (right-of-slash token) | `"MUSCAT"` (right-of-hyphen token from the instrument string) | **0.923** |

This is the concrete reason D-11 forbids treating instrument similarity as a gate: the raw
whole-string ratio (0.500) sits *below* this codebase's own established 0.6 difflib cutoff
convention — reusing that cutoff naively as a threshold would silently suppress criterion 5's
own reference case. Tokenising each string first (splitting on `/`, `-`, whitespace and
comparing token sets, or comparing only the trailing "instrument family" token) recovers a much
higher, criterion-5-passing similarity (0.923 in the example above). **This tokenisation step is
new logic this phase must write — no existing helper does it** — but it should live beside
`fuzzy_match_candidates()`'s existing `difflib` usage, not reinvent a different comparison
library.

```python
# Source: solsys_code/campaign_utils.py:549-577 (existing pattern to extend, not the code to add)
def fuzzy_match_candidates(site_raw: str, candidate_pool: dict[str, str], n: int = 5) -> list[tuple[str, str]]:
    text = (site_raw or '').strip()
    if not text:
        return []
    matches = difflib.get_close_matches(text, candidate_pool.keys(), n=n, cutoff=0.6)
    return [(match, candidate_pool[match]) for match in matches]
```

### Pattern 2: Telescope-match signal needs a NEW small alias table — no existing bridge

**What:** `CampaignRun.site` is an `Observatory` FK (MPC obscode vocabulary, e.g. `E10`).
`CalendarEvent.telescope` and `calendar_utils.SITE_TELESCOPE_MAP` use LCO's own 3-letter site
codes (`coj`, `ogg`, `sor`, `elp`, `lsc`, `cpt`, `tfn`) combined with an aperture-class token
(e.g. `COJ-2m0`). **These two vocabularies are never bridged by code today** — only by prose in
code comments (`calendar_utils.py:28-39`: "`'coj'` (Siding Spring)... `'ogg'` (Haleakala)").

A *different*, narrower bridge exists but does not cover this: `telescope_runs.SITES` (`[VERIFIED:
local codebase]` `solsys_code/telescope_runs.py:17-22`) maps a **classical run-file** telescope
nickname to an obscode — `{'Magellan-Clay': '268', 'Magellan-Baade': '269', 'NTT': '809', 'FTS':
'E10'}`. This confirms `'FTS'` (the token before the `/` in `CampaignRun.telescope_instrument`
`"FTS/MuSCAT4"`) maps to obscode `E10` — matching `CampaignRun` pk=1's already-resolved
`site.obscode`. But this dict has only 4 entries, is scoped to classical-file ingest, and has no
entries at all for the 7 LCO/SOAR `SITE_TELESCOPE_MAP` site codes.

**When to use / what to build:** For the pk=1 case specifically, the telescope-match signal
doesn't need any new bridge at all — `CampaignRun.site` is already resolved (`E10`), so the
signal just needs to know `E10 == "Siding Spring"` maps to LCO site code `coj`. For the general
case (any run whose site resolves to one of the 7 `SITE_TELESCOPE_MAP` sites), the matcher needs
a small new reverse-lookup table (obscode -> LCO site code, or vice versa), mirroring the
existing extension-rule discipline used for `HORIZONS_OBSERVER_TO_OBSCODE`
(`solsys_code/observer_codes.py`) and `telescope_runs.SITES`: **verify each obscode against the
real `Observatory` table or the MPC API before adding it — never infer from the site name
alone.** Only `coj -> E10` is confirmed here (via the live DB query below); the other 6 (`ogg`,
`sor`, `elp`, `lsc`, `cpt`, `tfn`) are not independently re-verified in this research pass and
must not be hand-typed from memory. `[ASSUMED]` — flagged in the Assumptions Log.

```python
# Confirmed by live query against src/fomo_db.sqlite3 (read-only):
# CampaignRun pk=1: site_id -> Observatory(obscode='E10', short_name='Siding Spring-Faulkes
#   Telescope South'); CalendarEvent rows for the same run show telescope='2m0' (10 of 11,
#   coarse/unverified fallback label) or telescope='COJ-2m0' (1 of 11, resolved label) --
#   'COJ-2m0' is exactly calendar_utils.SITE_TELESCOPE_MAP[('coj', '2m0')].
```

### Pattern 3: Atomic conditional `.update()` for confirm — event side vs. record side differ

**What:** `CampaignRunDecisionView.post()` (`campaign_views.py:534-661`) proves the "double-submit
is a no-op" and "race lands correctly" guarantees via
`CampaignRun.objects.filter(pk=pk, approval_status=PENDING_REVIEW).update(approval_status=new)`
— a single conditional queryset `.update()`, never a read-then-write pair, and the returned row
count (`updated_count`) distinguishes "already decided" from "never existed"
(`campaign_views.py:653-660`).

**When to use:** CONTEXT.md's research priority #3 requires this pattern generalised to two
**structurally different** targets:

- **`CalendarEventMeta.run`** is a **field being set on an existing row** (the
  `CalendarEventMeta` row for a `CalendarEvent` always exists once the event has gone through
  telescope-label resolution — confirmed: all 11 real rows here already have a
  `CalendarEventMeta` row, with `run` either `NULL` or already `1`). Confirmation is therefore
  exactly `CalendarEventMeta.objects.filter(event_id=event_pk, run__isnull=True).update(run=run_pk,
  confirmed_by=request.user, confirmed_at=timezone.now())` — the same shape as
  `CampaignRunDecisionView`'s conditional update, keyed on `run__isnull=True` (the "unclaimed"
  precondition) instead of `approval_status=PENDING_REVIEW`. A `updated_count == 0` result means
  "already attributed or dismissed by someone else" (D-08's `UniqueConstraint` isn't even the
  guard here — this conditional update *is* the guard, since `CalendarEventMeta.run` has no
  uniqueness constraint of its own, only nullability).
- **`CampaignRunObservation`** is a **row being created** (D-01: no row exists until
  confirmation). Confirmation is `CampaignRunObservation.objects.get_or_create(observation_record=record,
  defaults={'run': run, 'confirmed_by': request.user, 'confirmed_at': timezone.now()})`, and the
  race is caught by `unique_campaign_run_observation_record`'s `IntegrityError`, not by a
  conditional-update row count — `get_or_create()` under the existing `UniqueConstraint` is
  itself race-safe (confirmed as the established idiom by
  `insert_or_create_campaign_run()`/`insert_or_create_calendar_event()`'s docstrings, which
  explicitly note "only race-safe when its lookup fields are backed by a real DB constraint").
  Two racing staff confirming the same orphan-record pair against *different* runs must be
  wrapped the same way `CampaignRunSubmissionView.form_valid()` already handles a natural-key
  collision (`campaign_views.py:288-302`): catch `IntegrityError` inside its own
  `transaction.atomic()` savepoint (never let it poison the outer test/request transaction),
  and surface `messages.warning` per UI-SPEC's copy contract.

```python
# Source: solsys_code/campaign_views.py:558-561 (the pattern to generalise, not to copy verbatim)
updated_count = CampaignRun.objects.filter(
    pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
).update(approval_status=new_status)
```

**Multi-select bulk confirm (D-09):** for N checked `CalendarEventMeta` pks, either loop the
single conditional `.update()` per pk inside one `transaction.atomic()` block (simplest, matches
the existing per-row idiom, and still lets each row succeed/fail independently since `.update()`
doesn't raise on 0 rows matched), or issue one combined
`CalendarEventMeta.objects.filter(pk__in=checked_pks, run__isnull=True).update(run=...)` **only
if every checked row is being attributed to the SAME candidate run** — which is not true in
general (D-09's bulk button confirms N *different* orphan→run pairs in one submit, each row
potentially pointing at a different run). The per-row loop is therefore the correct shape, not
a single combined `.update()`; report back to the user how many of the N actually applied
(`updated_count` summed across the loop) versus how many were already claimed by someone else,
per UI-SPEC's "N candidates confirmed" success copy.

### Recommended Project Structure — Django admin stamping (Pattern 4)

**What:** `CampaignRunAdmin.save_formset()` (`admin.py:220-252`) already stamps
`confirmed_by`/`confirmed_at` on **newly created** `CampaignRunObservation` instances only
(`isinstance(instance, CampaignRunObservation) and instance.pk is None`), and explicitly does
**not** touch `CalendarEventMeta` instances flowing through the same method (the isinstance gate
exists precisely so the two inline formsets, which both call `save_formset` once each, don't
cross-contaminate).

**When to use:** D-12 requires extending this exact method with a second `isinstance` branch:
`CalendarEventMeta` instances need a "was `run` just set (newly non-null) on this save" check —
**not** a `pk is None` check, since `CalendarEventMeta.event` is the primary key and every row
already exists (created at telescope-label-resolution time) before a staff member ever links a
run to it via the admin. The correct condition is closer to: "the pre-save `run_id` was `None`
and the post-`formset.save(commit=False)` instance's `run_id` is not `None`" — this requires
diffing against the DB value before overwriting it, since `formset.save(commit=False)` returns
already-mutated in-memory instances. A clean approach: fetch each changed instance's *prior*
`run_id` via `CalendarEventMeta.objects.filter(pk=instance.pk).values_list('run_id', flat=True)`
(or capture it from `formset.initial_forms` before `save(commit=False)` mutates the instance)
before stamping — mirroring the "stamp only on a genuine transition" discipline
`CampaignRunObservationInline`'s `pk is None` check already established for the *creation* case.
This is genuinely new logic (Phase 27's `save_formset` only had to handle the creation case,
because `CampaignRunObservation` rows didn't exist before confirmation) — flag this as a real
gap, not a copy-paste extension.

```python
# Source: solsys_code/admin.py:230-246 (existing method to extend with a second branch)
def save_formset(self, request, form, formset, change):
    instances = formset.save(commit=False)
    for instance in instances:
        if isinstance(instance, CampaignRunObservation) and instance.pk is None:
            instance.confirmed_by = request.user
            instance.confirmed_at = timezone.now()
        # D-12 gap: a CalendarEventMeta branch must go here, keyed on a run_id
        # None -> not-None TRANSITION, not on instance.pk (which is never None for this model).
        instance.save()
    for obj in formset.deleted_objects:
        obj.delete()
    formset.save_m2m()
```

### Anti-Patterns to Avoid

- **Reinventing a fuzzy-match library or scoring mechanism.** `difflib` (stdlib) is the only
  permitted tool; `rapidfuzz` is explicitly rejected twice at the milestone level.
- **A materialized/cached candidate table refreshed by a scheduled job.** D-01/D-05 already
  settle this: candidates are computed live, only confirmations and dismissals persist. No
  Celery, no new background-job machinery (explicitly Out of Scope).
- **`GenericForeignKey` for either dismissal model.** Explicitly rejected (D-08, and the
  milestone's Out of Scope table) — two typed models with real FKs instead.
  **A single dismissal model with two nullable FKs and an "exactly one is set"
  `CheckConstraint`.** Also explicitly rejected by D-08 (branch-on-column reads, worse than two
  small models).
- **Treating instrument-string similarity or exact date-span equality as a gate rather than a
  weighted contributor.** D-11 is explicit and criterion 5 is the proof: gating on either would
  fail day one on the real pk=1 case (measured evidence above: raw instrument ratio 0.500,
  below the codebase's own 0.6 convention; date span differs by a day).
- **A single combined queryset `.update()` for bulk confirm across heterogeneous target runs.**
  Each checked row may point at a different candidate run — a combined `.update(run=X)` can only
  set one value for all matched rows, so the per-row-loop pattern above is required, not a
  shortcut.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| String similarity scoring | A custom Levenshtein/Jaro-Winkler implementation, or `rapidfuzz` | `difflib.SequenceMatcher`/`get_close_matches`, exactly as `campaign_utils.py` already does | Milestone-locked; stdlib already proven sufficient for a harder version of this same problem (free-text site names) |
| ObservationRecord active time window | A new date/time extraction helper | `sync_lco_observation_calendar.py:108` `_time_window(record)` (currently module-private) — prefers `scheduled_start`/`scheduled_end`, falls back to `parameters['start']`/`parameters['end']` as naive-UTC ISO strings | This exact fallback logic is needed for the record-side date-overlap signal (measured: 10 of the 11 real matching records have `scheduled_start`/`scheduled_end` = `NULL`, only the one COMPLETED record has them populated) — reuse or promote this helper rather than re-deriving the parsing rules |
| Staff-only page gating | A new permission class or decorator | `StaffRequiredMixin` (`solsys_code/mixins.py`) | Already the gate for `ApprovalQueueView`; zero reason to diverge |
| Atomic race-safe state transition | A `transaction.atomic()` + `select_for_update()` pair | The conditional-`.filter().update()` idiom `CampaignRunDecisionView` already established | Documented, tested, and proven correct against this codebase's SQLite deployment; introducing row-locking here would be a second, inconsistent concurrency-control style |
| Shared "count of items needing staff action" | A second ad hoc `.count()` call inline in `CampaignListView`/the attribution page | A single new shared function (mirroring `runs_needing_site_review()`) that both the banner and the worklist call | Prevents the exact silent-drift hazard 27.1-03's docstring calls out by name |

**Key insight:** every hard technical problem this phase touches (fuzzy string matching, atomic
staff-decision transitions, staff gating, shared-queryset-for-banner-and-page) already has a
proven, tested precedent in this exact codebase from the last three phases. The only genuinely
new logic is (1) the telescope-obscode-to-LCO-site-code bridge (Pattern 2 above) and (2) the
`run_id` None→not-None transition detection needed for `save_formset`'s new
`CalendarEventMeta` branch (Pattern 4 above) — both should be written carefully and tested
directly, since neither has an existing implementation to lean on.

## Runtime State Inventory

> This phase adds new fields/models and a new write surface, but it is not a rename/refactor/
> migration-of-existing-data phase in the sense the inventory below is designed for (no strings
> are being renamed; no existing rows change meaning). Included briefly for completeness since
> the phase does carry schema migrations.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | The one pre-existing `CampaignRunObservation` row (pk=1, run=3, observation_record=58) is a cross-campaign anomaly already committed to the dev DB before this phase — `CampaignRun` pk=3 belongs to campaign "3I/ATLAS" (`TargetList` pk=3, containing only target 142), but `observation_record` pk=58's target (141) belongs to campaign "Didymos 2026" (`TargetList` pk=2). This is dev-DB noise from earlier manual/test admin activity, not a defect this phase is asked to fix (read-only research; the existing `UniqueConstraint` already prevents a second link to the same record, so this row simply removes pk=58 from the orphan pool). | None — code edit not required; the planner should simply avoid writing a criterion-5 acceptance test against this specific record and should not assume all 11 real records are available candidates |
| Live service config | None found — nothing in this phase depends on externally-configured service state (n8n/Datadog/Tailscale-style concerns don't apply to this codebase) | None |
| OS-registered state | None found — no OS-level task registration touches `CampaignRunObservation`/`CalendarEventMeta` | None |
| Secrets/env vars | None found — this phase introduces no new secrets or environment variables | None |
| Build artifacts | None found — no package rename, no stale egg-info concern; this is additive schema + one new module | None |

## Common Pitfalls

### Pitfall 1: Reusing the 0.6 `difflib` cutoff as an instrument-match threshold

**What goes wrong:** The instrument-similarity signal silently fails to surface criterion 5's
real case (or any pair with a comparably "obviously the same, differently formatted" instrument
string), because a naive whole-string comparison scores below the codebase's own established
0.6 convention.
**Why it happens:** `fuzzy_match_candidates()`'s 0.6 cutoff is well-suited to *site name*
matching (long descriptive strings vs. their own abbreviations/typos) but LCO instrument-type
codes and free-text telescope/instrument descriptions (`"FTS/MuSCAT4"` vs `"2M0-SCICAM-MUSCAT"`)
are a different, much noisier comparison — one string carries a facility nickname, the other an
aperture-class-prefixed LCO instrument code, and neither directly contains the other as a
substring or a close whole-string match.
**How to avoid:** Treat instrument similarity as a *contributing weighted signal* (D-11 already
mandates this) computed from a tokenised/normalized comparison (e.g. strip the aperture-class
prefix, compare on the "instrument family" token — `MUSCAT` vs `MuSCAT4`), not a raw whole-string
`SequenceMatcher` ratio compared against 0.6.
**Warning signs:** A test asserting criterion 5's real pk=1 pair appears at all (any band) is the
canary — if it doesn't, the instrument signal (or its weight) is too strict.

### Pitfall 2: Treating `CalendarEventMeta.run__isnull=True` as the whole orphan definition

**What goes wrong:** The orphan-calendar-event queryset misses that a `CalendarEvent` may not
have a `CalendarEventMeta` row at all (Phase 27's CANON-03 docstring: "no row at all means
'verified' by documented default... classically-scheduled events from `load_telescope_runs`,
which never go through telescope-label resolution").
**Why it happens:** `CalendarEventMeta` is a `OneToOneField(primary_key=True)` companion, created
only by the LCO/SOAR/Gemini telescope-label resolution path — a classical run's `CalendarEvent`
(from `load_telescope_runs`) has no companion row and therefore no `run` field to check at all.
**How to avoid:** The orphan-calendar-event queryset needs `CalendarEvent.objects.filter(
Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True))` (or an
equivalent two-branch definition), not a single `.filter(telescope_label_meta__run__isnull=True)`
which silently excludes every classical event that has no companion row at all. Confirm which of
the two orphan populations (classical vs. LCO/Gemini-derived) D-03's "at least one candidate"
noise filter is actually meant to admit — classical events are exactly the kind whose
`CampaignRun` would be `source=CLASSICAL_FILE` (not yet produced by any adapter until v2.3
ADAPT-01, per the `Source` TextChoices docstring), so in practice today no classical event will
ever have a candidate run and D-03 already filters it out downstream — but the queryset must be
written to reach that correct conclusion, not accidentally exclude classical events from
consideration for the wrong reason.
**Warning signs:** A count mismatch between "total un-owned `CalendarEvent`s" and "orphans shown
in the queue" that isn't explained by D-03's candidate-count filter alone.

### Pitfall 3: `CampaignRunObservation`'s `UniqueConstraint` is on `observation_record` alone

**What goes wrong:** Code assumes the constraint is on `(run, observation_record)` and therefore
believes two different runs can each separately confirm the same record (a "confirm race" that
looks preventable per-run rather than globally).
**Why it happens:** `models.py:384-387`'s `unique_campaign_run_observation_record` constraint is
declared with `fields=('observation_record',)` only — **one run per record, globally**, not one
row per (run, record) pair. This is deliberate (D-02, Phase 27): "behaviour today is identical [to
a `OneToOneField`], but broadening to many-runs-per-record later is a single `RemoveConstraint`".
**How to avoid:** The confirm-race test (research priority #3(b)) must simulate two *different*
runs both being confirmed against the *same* record concurrently (not two confirmations of the
same run/record pair) — the second `get_or_create()` call must hit the `IntegrityError` on
`observation_record` uniqueness alone, regardless of which run it names.
**Warning signs:** A test that confirms the same `(run, record)` pair twice "proves" idempotency
but says nothing about the actual two-different-runs race this constraint guards against.

### Pitfall 4: Confusing `CalendarEventMeta.run__isnull` transitions with `CampaignRunObservation`'s existence check

**What goes wrong:** `save_formset`'s new `CalendarEventMeta` audit-stamping branch (D-12) is
written with the same `instance.pk is None` gate `CampaignRunObservation`'s branch uses, and
never stamps anything — because every `CalendarEventMeta` instance already has a non-`None` pk
(the row was created at telescope-label-resolution time, long before any staff member touches
`run`).
**Why it happens:** Copy-pasting Pattern 4's existing branch structure without noticing the two
models differ in exactly this respect (D-01: `CampaignRunObservation`'s existence *is* the
confirmation; `CalendarEventMeta`'s existence is unrelated to attribution — only its `run` field
being newly non-null is).
**How to avoid:** Detect the transition (prior `run_id` was `None`, new `run_id` is not `None`)
rather than gating on `pk is None`. See Pattern 4 above for the exact mechanics.
**Warning signs:** An admin-created event→run link in a test never gets a `confirmed_by`/
`confirmed_at` value even though the test asserts `run_id` was set correctly — this is exactly
the gap D-12 exists to close, so a passing "run got linked" test alongside a failing "audit
fields got stamped" test is the signature of this bug.

## Code Examples

### Existing atomic conditional-update idiom to generalise (confirm, event side)

```python
# Source: solsys_code/campaign_views.py:558-561 (CampaignRunDecisionView.post, existing code)
updated_count = CampaignRun.objects.filter(
    pk=pk, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW
).update(approval_status=new_status)
# ... updated_count == 0 distinguishes "already decided" (row exists) from "row deleted"
# (campaign_views.py:653-660) -- the same 3-way branch (success / already-claimed / gone)
# applies to the new CalendarEventMeta.run confirm action.
```

### Existing `difflib` fuzzy-match idiom to extend for instrument similarity

```python
# Source: solsys_code/campaign_utils.py:549-577 (existing code, precedent only)
matches = difflib.get_close_matches(text, candidate_pool.keys(), n=n, cutoff=0.6)
```

### Existing `save_formset` stamping idiom to extend with a second branch

```python
# Source: solsys_code/admin.py:230-246 (existing code, precedent -- see Pattern 4 above
# for what the new CalendarEventMeta branch must do differently)
instances = formset.save(commit=False)
for instance in instances:
    if isinstance(instance, CampaignRunObservation) and instance.pk is None:
        instance.confirmed_by = request.user
        instance.confirmed_at = timezone.now()
    instance.save()
```

### Existing `ObservationRecord` time-window extraction to reuse (currently module-private)

```python
# Source: solsys_code/management/commands/sync_lco_observation_calendar.py:108-136
def _time_window(record: ObservationRecord) -> tuple[datetime, datetime]:
    if record.scheduled_start is None and record.scheduled_end is None:
        start_time = datetime.fromisoformat(record.parameters['start']).replace(tzinfo=dt_timezone.utc)
        end_time = datetime.fromisoformat(record.parameters['end']).replace(tzinfo=dt_timezone.utc)
    elif record.scheduled_start is not None and record.scheduled_end is not None:
        start_time = record.scheduled_start
        end_time = record.scheduled_end
    else:
        raise ValueError(...)
    return start_time, end_time
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual admin FK picker is the only way to create a run<->event link (Phase 27.1-02's legible `__str__`/autocomplete was the interim mitigation) | Staff-facing scored worklist with confirm/dismiss/undo (this phase) | This phase | The admin FK picker remains available (and still the only path for a run<->event link with no candidate at all, e.g. a genuinely obscure match), but the worklist becomes the primary, evidence-backed path for the common case |
| `CampaignRunObservation` rows can only be created via the Django admin inline (Phase 27 CANON-04/CANON-05) | Also creatable via the staff attribution worklist's confirm action | This phase | Both paths must stamp `confirmed_by`/`confirmed_at` identically (this is exactly D-12's motivating concern, generalised: any write path onto these two link models must stamp audit fields, not just the admin) |

**Deprecated/outdated:** none — this phase adds a new surface, it does not retire an old one
(unlike Phase 29's `backfill_range_calendar_events` retirement).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 6 unverified LCO/SOAR site-code-to-obscode mappings (`ogg`→?, `sor`→?, `elp`→?, `lsc`→?, `cpt`→?, `tfn`→?) beyond the confirmed `coj`→`E10` — not independently re-verified in this research pass, deliberately left unstated rather than guessed | Architecture Patterns, Pattern 2 | If the planner hand-types these from memory without verifying against the live `Observatory` table or the MPC API, a wrong obscode silently breaks the telescope-match signal for any site other than Siding Spring — verify each one against `Observatory.objects.filter(...)` or the MPC Obscodes API before adding to any new alias table, mirroring the existing `HORIZONS_OBSERVER_TO_OBSCODE` extension-rule discipline |
| A2 | `django-filter`'s `FilterSet`/`FilterView` may not be the right fit for the band filter, since attribution rows are computed candidate pairs rather than a plain model queryset — recommended a manual GET-param approach instead, but this is a recommendation, not a verified constraint | Standard Stack | If the planner defaults to `FilterSet` out of habit (matching `CampaignRunTableView`'s precedent) without checking whether the computed-candidate data shape actually supports it, time may be lost discovering the mismatch mid-implementation |
| A3 | The exact weighting formula and High/Medium/Low cut-points are unspecified by design (Claude's Discretion per CONTEXT.md) — this research provides measured raw signal values (Pattern 1's ratio table) but does not commit to specific weights or thresholds | Standard Stack / Architecture Patterns | The planner must choose weights such that criterion 5's real pk=1 pair clears the High band (for D-09's bulk-confirm) or at minimum is visibly offered (any band, per criterion 1) — untested weight choices risk either hiding the reference case or making the High band too loose (D-09's own stated residual risk) |
| A4 | Whether a combined-migration (`CreateModel` x2 + `AddField` x2 on `CalendarEventMeta` in one file) or split-migrations sequencing is preferred is left to the planner — no strong precedent either way in this codebase's migration history (0008 kept the rename separate from 0009's field-add specifically "so a rename regression and a new-field regression can never be confused", but that rationale doesn't obviously transfer to two independent new models plus two new fields on an existing model) | Recommended Project Structure | Low risk either way; flagged only so the planner makes a deliberate choice rather than defaulting without considering the 0008/0009 precedent |

**If this table is empty:** N/A — see entries above.

## Open Questions

1. **Should the orphan-calendar-event queryset's `Q(telescope_label_meta__isnull=True)` branch
   (Pitfall 2) ever produce a real candidate today?**
   - What we know: only LCO/SOAR/Gemini-synced events get a `CalendarEventMeta` companion row at
     all; classical events (`load_telescope_runs`) never do, and no adapter produces
     `source=CLASSICAL_FILE` `CampaignRun`s until v2.3's ADAPT-01.
   - What's unclear: whether any *pre-milestone* classical `CalendarEvent` could still coincidentally
     match a `LEGACY`-sourced `CampaignRun` on telescope/date signals, making this branch produce a
     real (if rare) candidate today.
   - Recommendation: write the queryset correctly for completeness/correctness (per Pitfall 2), but
     don't block the phase on finding a real example — D-03's "must have >=1 candidate" filter
     will naturally suppress it if none exists.

2. **Does the campaign/target boundary check (D-11's hard gate) need to account for the
   pre-existing cross-campaign `CampaignRunObservation` anomaly (pk=1, run=3)?**
   - What we know: this row already exists and already violates the boundary the matcher is being
     built to enforce for *new* writes.
   - What's unclear: whether any downstream Phase 28/29 logic reads `CampaignRunObservation` rows
     assuming they always respect campaign boundaries (a "every existing link is boundary-clean"
     invariant this row already breaks).
   - Recommendation: the matcher's gate only needs to apply to *new* candidate generation — it should
     not need to validate or clean up this pre-existing row, and the planner should not build a
     migration to "fix" it (out of scope, read-only research, and the row causes no crash — it just
     removes one record from the orphan pool).

## Environment Availability

No new external dependency, service, or CLI tool is required by this phase — it is a pure
Django/Python change over the existing SQLite dev database and installed package set. Skipped
per the stated skip condition.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Django test runner (`django.test.TestCase`/`TransactionTestCase`) via `python manage.py test solsys_code` |
| Config file | none dedicated — `pytest.ini`/`pyproject.toml` `testpaths` excludes `solsys_code/` entirely (Django app tests run under the Django runner, not pytest, per CLAUDE.md) |
| Quick run command | `python manage.py test solsys_code.tests.test_campaign_attribution` (matcher unit tests only, no DB fixtures needed if pure functions are tested directly) |
| Full suite command | `python manage.py test solsys_code` (excluding `test_views.TestEphemeris`, which segfaults in native ASSIST and is unrelated to this phase) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ATTRIB-01 | Worklist shows evidence columns (telescope/date/campaign/instrument) per candidate, not a bare score | unit + view | `python manage.py test solsys_code.tests.test_campaign_attribution_views.TestEvidenceColumns` | ❌ Wave 0 |
| ATTRIB-02 | Candidates confidence-scored, filterable by band | unit | `python manage.py test solsys_code.tests.test_campaign_attribution.TestScoringAndBanding` | ❌ Wave 0 |
| ATTRIB-03 | No association without explicit confirmation; no cross-boundary suggestion | unit + view | `python manage.py test solsys_code.tests.test_campaign_attribution.TestCampaignBoundaryGate` | ❌ Wave 0 |
| ATTRIB-04 | Confirm/undo, both attributable to person+time | view + model | `python manage.py test solsys_code.tests.test_campaign_attribution_views.TestConfirmUndo` | ❌ Wave 0 |
| ATTRIB-05 | Real pk=1 case surfaced against its real LCO events | integration (real-DB-shaped fixture) | `python manage.py test solsys_code.tests.test_campaign_attribution.TestCriterion5RealCase` | ❌ Wave 0 |
| ATTRIB-06 | Queue can drain to zero before first reconcile sweep | view (end-to-end confirm-all-then-assert-empty) | `python manage.py test solsys_code.tests.test_campaign_attribution_views.TestQueueDrainsToEmpty` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `python manage.py test solsys_code.tests.test_campaign_attribution` (fast,
  no-DB-fixture matcher tests) plus the specific new test file(s) touched by that task.
- **Per wave merge:** `python manage.py test solsys_code` (excluding `TestEphemeris`).
- **Phase gate:** Full suite green (`ruff check .`, `ruff format --check .`, full
  `manage.py test solsys_code` run) before `/gsd-verify-work`.

### Wave 0 Gaps

- [ ] `solsys_code/tests/test_campaign_attribution.py` — matcher unit tests: scoring formula,
  band cut-points, campaign/target boundary hard gate, instrument-similarity tokenisation
  (covers the measured-evidence Pitfall 1 case directly with the real strings above).
- [ ] `solsys_code/tests/test_campaign_attribution_views.py` (or extend
  `test_campaign_approval.py`-style conventions in a new file) — view/POST integration tests:
  confirm (event side, atomic conditional update), confirm (record side, `get_or_create` +
  `IntegrityError`), dismiss, undo, double-submit no-op, two-staff race on both link types,
  staff gating (`StaffRequiredMixin`), and the criterion-5 acceptance test built against one of
  the 10 genuinely-orphaned real-shaped fixture rows (not pk=53/pk=58 verbatim — build an
  equivalent fixture using `NonSiderealTargetFactory`/`ObservationRecord.objects.create`, per
  this codebase's established fixture style in `test_campaign_run_observation.py`).
- [ ] Admin `save_formset` `CalendarEventMeta` branch test — extend
  `solsys_code/tests/test_admin.py`'s `CampaignRunAdminInlinesTests` class with a
  `test_save_formset_stamps_calendar_event_meta_on_run_transition` case (Pitfall 4).
- [ ] No new framework install needed — `django.test.TestCase`/`TransactionTestCase` already
  cover every case above; no `pytest` fixtures needed since this is entirely Django-app-test
  territory per CLAUDE.md's testing split.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no (session auth already handled by Django/TOM) | n/a |
| V3 Session Management | no (unchanged) | n/a |
| V4 Access Control | yes | `StaffRequiredMixin` gates the entire attribution page and every POST action, exactly as `ApprovalQueueView`/`CampaignRunDecisionView` already do — every new view class must declare it explicitly, it is not inherited automatically |
| V5 Input Validation | yes | Every POST action must re-validate the target orphan/run pks server-side (never trust that a checkbox/button was only rendered for an eligible row) — mirrors `CampaignRunDecisionView._resolve_site()`'s "business-logic bypass guard" comment discipline; the multi-select bulk-confirm endpoint must re-check each submitted pk is still in the High band server-side, not merely trust that the checkbox only rendered for High-band rows client-side |
| V6 Cryptography | no | n/a — no new crypto surface |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Business-logic bypass: a staff user POSTs a confirm/dismiss/undo action for a pair that no longer qualifies (already confirmed, already dismissed, or now crosses the campaign/target boundary because the underlying data changed between page render and POST) | Tampering | Server-side re-validation on every POST (never trust the rendered button/checkbox state) — exactly the discipline `CampaignRunDecisionView._resolve_site()`'s guard comment documents: "validate state server-side, never just trust the button was only offered on eligible rows" |
| IDOR: a staff user (or a non-staff user bypassing `StaffRequiredMixin` client-side) submits an arbitrary orphan pk / run pk pair not actually offered as a candidate | Information Disclosure / Elevation of Privilege | `StaffRequiredMixin` on every view class; additionally, since D-11's campaign/target boundary is a *hard* gate, the confirm action must independently re-derive "is this run actually a valid candidate for this orphan" server-side (not merely check "does this orphan exist" and "does this run exist") — otherwise a tampered POST could create a cross-boundary association the UI would never have offered, defeating criterion 3's absolute guarantee |
| Stored-XSS via the free-text dismissal reason (D-06) | Tampering / Information Disclosure | Django's auto-escaping template rendering handles this by default (same as `CampaignRunTable.render_window_start()`'s existing `original_obs_date_raw` tooltip precedent, which explicitly notes "mitigates stored-XSS from community-editable sheet text... never `mark_safe` or string concatenation") — the dismissal reason must be rendered the same way, never `mark_safe`d |
| CSRF on confirm/dismiss/undo POST actions | Tampering | Django's standard CSRF middleware + `{% csrf_token %}`/`get_token(request)` in each form, exactly as `ApprovalQueueTable.render_actions()` already does |

## Sources

### Primary (HIGH confidence)

- Local codebase read directly: `solsys_code/campaign_utils.py`, `solsys_code/calendar_utils.py`,
  `solsys_code/models.py`, `solsys_code/campaign_views.py`, `solsys_code/campaign_tables.py`,
  `solsys_code/campaign_urls.py`, `solsys_code/mixins.py`, `solsys_code/admin.py`,
  `solsys_code/telescope_runs.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py`,
  `src/templates/campaigns/approval_queue.html`, `src/templates/campaigns/campaign_list.html`,
  `solsys_code/tests/test_campaign_run_observation.py`, `solsys_code/tests/test_admin.py`,
  `solsys_code/tests/test_campaign_approval.py`, `solsys_code/tests/test_campaign_views.py`,
  `solsys_code/migrations/` (directory listing, latest = 0012).
- Live read-only SQLite query against `src/fomo_db.sqlite3` (`CampaignRun`, `CalendarEvent`,
  `CalendarEventMeta`, `ObservationRecord`, `CampaignRunObservation`,
  `tom_targets_targetlist`/`tom_targets_targetlist_targets`, `tom_targets_basetarget`,
  `solsys_code_observatory_observatory` tables) — confirms `CampaignRun` pk=1's real values, the
  11 real `CalendarEvent`/`ObservationRecord` rows, and the two anomalies (pk=53 already linked,
  pk=58 wrongly linked) documented above.
- `pip show django-tables2 django-filter django-htmx` run in this repo's virtualenv — confirms
  installed versions 3.0.0 / 24.3 / 1.27.0.
- Third-party package model source read directly: `tom_calendar/models.py`,
  `tom_observations/models.py` (installed package files in this repo's virtualenv).

### Secondary (MEDIUM confidence)

- `.planning/phases/28-operator-assisted-attribution/28-CONTEXT.md` and
  `28-UI-SPEC.md` (locked project decisions, cited throughout as `D-XX`/UI-SPEC references).
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/STATE.md` (project decision
  history, phase sequencing rationale).
- `docs/runbooks/telescope_runs_calendar.rst` (existing runbook structure/tone, read to determine
  where a new "How do I attribute..." section fits).

### Tertiary (LOW confidence)

- None of the LCO/SOAR site-code-to-obscode mappings beyond `coj`→`E10` were independently
  re-verified against the MPC API or the live `Observatory` table in this research pass — flagged
  in the Assumptions Log (A1) rather than asserted.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no new dependency; every library version directly confirmed via
  `pip show` in the project's own virtualenv.
- Architecture: HIGH for reused patterns (all read directly from source); MEDIUM for the two
  genuinely new pieces of logic (the obscode/site-code bridge, the `save_formset` transition
  detection) since no existing implementation could be read as ground truth for those.
- Pitfalls: HIGH — all four are derived from directly-read code plus a live-DB-verified measured
  example (the `difflib` ratio table), not speculation.

**Research date:** 2026-08-01
**Valid until:** 30 days (stable internal codebase; the live-DB snapshot itself is a point-in-time
read and will drift as more runs/events/records are created — re-verify criterion 5's exact pks
at plan time if execution is delayed significantly past this date).
