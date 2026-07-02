# Feature Research

**Domain:** Campaign coordination for rare/urgent Solar System objects (FOMO v2.0 — target-linked campaign-run tracking, community submission + approval, per-target coverage view)
**Researched:** 2026-07-02
**Confidence:** MEDIUM for "what the feature category needs" (cross-corroborated against multiple real astronomy coordination platforms — ExoFOP-TESS, TNS, YSE-PZ, IAWN, generic Django moderation patterns); HIGH for "what the 3I/ATLAS reference sheet actually contains" (verified directly against the enriched seed's field inventory, which was built from the real spreadsheet, not from search); LOW-confidence items are called out inline (general web search only, no hands-on verification of any of the reference systems' internals).

## Headline Finding (read this first)

**None of the reference systems (IAWN, ExoFOP-TESS, TNS, YSE-PZ) implement the specific thing this milestone needs — an admin-gated, target-linked *campaign coordination table* for a single rare object with a lifecycle status and a coverage-gap view.** Each is close on one axis and absent on others:

- **IAWN** is the closest conceptual match (ad-hoc rapid-response campaign for a specific object) but is entirely informal/human-coordinated — no software artifact to copy, MPC is the astrometry sink, not a campaign tracker.
- **ExoFOP-TESS** has the right per-target aggregation + upload/follow pattern, but is a full data-product archive (raw file uploads, derived parameters) — heavier than what FOMO needs and a different failure mode (data archive vs. coordination sheet).
- **TNS** has the right dual-intake shape (bot API + human form feeding one canonical registry) but no approval gate for humans and no per-object campaign grouping — it's a one-row-per-discovery ledger, not a many-runs-per-object table.
- **YSE-PZ** (same TOM Toolkit family as FOMO) has the coordination *tools* (finding charts, airmass plots, trigger forms, group permissions) but no evidence of a moderated community-submission-with-approval-queue feature — it assumes trusted collaboration members, not open/unvetted community intake.

**The actual shape of this milestone is closest to "TNS's dual-intake pattern, scoped to one target at a time, with a Django-moderation-style pending/approved gate, replacing a spreadsheet whose exact columns are already known."** This is a smaller, more tractable problem than any single reference system solves whole — which is good news for scoping but means there is no off-the-shelf pattern to import wholesale; the design should synthesize from the pieces below rather than clone one system.

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = the spreadsheet-replacement fails and coordinators fall back to the Google Sheet.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `CampaignRun` data model: target FK, telescope/instrument, site code, obs date + UT time range, filter/bandpass, observation details, status, outcome, publication plans, collaboration flag, contact person/email, comments | This is a direct 1:1 mapping of the real 3I/ATLAS sheet's column inventory (verified against the enriched seed, not inferred) — anything less and the bootstrap import loses data | MEDIUM | Model as its own entity (not a `CalendarEvent` widening), per the seed's own recommendation, echoing the Phase 8 `CalendarEventTelescopeLabel` sidecar precedent of extending via a separate linked model rather than growing an existing one; site code should FK/lookup against the existing `Observatory` model exactly like `CalendarEvent`'s existing site resolution does |
| Lifecycle status field (planned → observed → reduced → published) | The sheet's own "Observation Status" column is the seed's explicit motivating gap — today's title-prefix vocabulary (`[QUEUED]`/`[UNVERIFIED]`) is calendar-sync-specific and doesn't cover the post-observation lifecycle (reduced, published) that campaign coordination cares about | LOW-MEDIUM | A single `status` choices field is enough; no state-machine library needed — this is TNS's AT→SN transition pattern (a small closed vocabulary, one field, one direction of travel) |
| Per-target campaign table view — all runs for one object, one page | This *is* the spreadsheet-replacement ask; a coordinator today opens one Google Sheet and sees every row for 3I/ATLAS. Without this, the data model is just inert rows in the admin | LOW-MEDIUM | Straightforward `ListView`/template filtered by `target`, linked from the target detail page (same integration point pattern as the existing "Make Ephemeris" button injected via `target_detail_buttons()`) |
| One-off CSV bootstrap import of the real 3I/ATLAS sheet | Explicitly required by the milestone scope and by the seed ("validate the model against real campaign data before a 4I ever appears") — an unvalidated schema is a real risk given the sheet's free-text columns (weather, comments, observation details) | MEDIUM | One-shot management command in the `load_telescope_runs`/`fetch_jplsbdb_objects` tradition (existing precedent for CSV/API-to-model ingest commands in this codebase); expect messy free-text normalization (multi-band imaging rows, mixed date formats) — budget research/spike time, don't assume clean input |
| Community submission form (Target required, most other fields optional) | The sheet itself *is* a submission form today — anyone with the link edits a row directly. A web form is table stakes, not a nice-to-have, because it's literally what's being replaced | MEDIUM | Per the original seed: Target is the only hard-mandatory field; proposal code stays optional (many campaign runs — Palomar P200/NGPS, VLT/MUSE — are outside FOMO's facility-sync scope entirely, so requiring a FOMO-known proposal would reject legitimate real rows) |
| Submitter-visible-but-gated contact info (PII guard) | The sheet already stores Contact Person + Email; FOMO's `OPEN`/`AUTH_STRATEGY='READ_ONLY'` model means anyone can view target pages today, so unguarded email display is a real PII leak this milestone must not introduce | MEDIUM | Exact policy is an open question per the seed (auth-gated column vs. opt-in display vs. store-but-never-render) — but *some* guard is table stakes, not optional; TNS's own registration-gate-for-submission (not for reading) is one workable reference point |
| Approval gate before a submission is publicly visible | Directly requested by the milestone ("admin approval queue"); also matches the general Django moderation pattern (pending record invisible until approved) found across every reviewed reference — this is the single most consistently-expected shape in the research | LOW-MEDIUM | Simplest correct implementation is a `status` value (`pending_review`/`approved`/`rejected`) on `CampaignRun` itself, filtered out of the public table view — no need for a heavyweight package (`django-moderation` et al.) or a dual-model draft/live pair; see Anti-Features |

### Differentiators (Competitive Advantage — Defer if Needed)

Features that set FOMO apart from "just use the Google Sheet again." Should align with the milestone's stated Core Value (FOMO replaces the ad-hoc sheet as the community hub).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Ephemeris-aware coverage-gap analysis (observable-but-unclaimed dates) | This is the milestone's own stated differentiator and the reason to build in FOMO rather than spin up another sheet — no reference system found (IAWN, ExoFOP, TNS, YSE-PZ, or the general visibility-tool ecosyston: NASA Exoplanet Archive ephemeris service, ESO Object Observability, Visplot, ESA NEOCC feed) cross-references *observability* against a *claimed-run registry* to surface gaps. Every tool found computes one half of this problem, never both | HIGH | Composes two things this codebase already has: `solsys_code/telescope_runs.py` (`sun_event`, dark-window calc, per-`Observatory` geometry) for the observable side, and the new `CampaignRun` table (approved, non-cancelled rows) for the claimed side; genuinely novel synthesis, not a lookup — budget real design time, and the milestone's own scoping ("last, so it can defer to v2.1") is sound risk management |
| "Open to collaboration?" flag surfaced prominently (e.g. in the table view, filterable) | Directly actionable: a new observer looking for where to help can filter to "wants collaborators" runs, which the raw sheet supports only via manual scanning | LOW | Straightforward boolean field + filter; higher value once combined with the coverage-gap view (i.e., "here's a gap, and here's who's said they want collaborators nearby") |
| Self-service approval bypass for approved-program PIs (trusted submitters skip the queue) | From the original seed: committed-time PIs shouldn't face the same friction as a first-time community submitter — reduces admin bottleneck for the highest-trust, highest-volume submitter class | MEDIUM | Requires a submitter-trust concept (e.g. matching submitter's Django user/email against known approved-program proposal codes) — real design work, correctly flagged as an open question in the seed rather than assumed |
| Admin notification on new pending submission | Every moderation reference (Django moderation packages, general best-practice writeups) treats "moderator gets notified" as a first-class feature, not an afterthought — an approval queue nobody checks is worse than no queue | LOW | Django's existing email backend is sufficient; no new infrastructure — in-app notification is a nicer follow-up, not required for v1 of this feature |
| Cross-link from `CampaignRun` rows to the existing telescope-runs calendar (where the run's facility is one FOMO already syncs, e.g. LCO/Gemini) | Lets a coordinator see "this campaign run is *also* a synced `CalendarEvent`" rather than two disconnected views of the same reality | MEDIUM | Only applicable to the subset of campaign runs on FOMO-synced facilities; most 3I-sheet rows (Palomar, VLT/MUSE) have nothing to link to — genuinely optional polish, not core |
| "My Targets"-style follow/subscribe with email updates on new campaign activity | ExoFOP-TESS's most-cited engagement feature; would let community observers get notified when new runs are added for an object they care about | MEDIUM-HIGH | Real value but a distinct feature with its own notification-infrastructure cost; not mentioned anywhere in the milestone's stated scope — correctly a future consideration, not v2.0 |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but would create problems or scope creep for this milestone.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Full data-file/data-product upload and hosting (ExoFOP-style) | ExoFOP-TESS's most visible feature is exactly this, and "let people attach their FITS/light-curve files" sounds like a natural extension of "track the run" | Turns FOMO into a data archive with storage, format-validation, and access-control concerns far outside this milestone's scope (and outside anything the 3I sheet itself does — it's metadata-only, no attachments) | Keep `CampaignRun` metadata-only, matching the sheet it replaces; "Publication Plans"/"Observation Details" stay free-text fields describing where the data lives externally |
| Generic "any facility can push status via API" bot-ingestion layer, mirroring TNS's bot-heavy intake | TNS's dual bot+human intake looks like the "right" scalable design, and FOMO already has 3 API-sync commands (LCO, Gemini, SOAR) it could pattern-match against | Campaign runs are explicitly the *out-of-sync-command* case — the seed notes example 3I rows spanning FTN/MuSCAT3, Palomar P200/NGPS, VLT/MUSE, i.e. mostly facilities with no FOMO API integration at all. Building a generic bot-ingestion abstraction here solves a problem this milestone doesn't have | Community submission form is the correct, and only necessary, intake path for campaign runs (this is explicitly called out in the seed's own reasoning) |
| Mandatory proposal-code validation against a live LCO/ESO proposal database | Feels like it would prevent bad data at the source, mirroring how the existing sync commands trust structured `ObservationRecord` data | Campaign runs routinely have no FOMO-known proposal at all (external facilities) or are calendar-only per the original seed ("Optional: Proposal code — required only if the submitter wants to trigger observations through FOMO"); mandatory cross-validation would reject legitimate real rows and contradicts the seed's own design | Keep proposal code optional free text; validate only structurally (non-empty if provided), never against an external database, in v2.0 |
| Heavyweight third-party moderation package (`django-moderation`, `django-approval`, `djangocms-moderation`) for the approval queue | These exist specifically for "Django approval workflow" and search results surface them as the go-to answer | They're built for arbitrary-model, arbitrary-field moderation (diffing granular field-level changes across any model) — overkill for one model (`CampaignRun`) with one clear pending→approved/rejected transition; adds a dependency and abstraction layer the milestone doesn't need, inconsistent with this codebase's existing preference for small hand-rolled helpers (`calendar_utils.py`) over heavy frameworks | A single `status` choices field on `CampaignRun`, filtered in the public queryset, with admin-integrated approve/reject actions (Django admin already supports custom admin actions natively, no package needed) |
| Unauth-gated public display of submitter email (matching the sheet's current link-shared-anyone-can-see behavior) | "Just show what the sheet shows" is the path of least resistance and avoids a design decision | Directly conflicts with FOMO's `OPEN`/`AUTH_STRATEGY='READ_ONLY'` model — FOMO target pages are far more publicly discoverable than a link-shared sheet was ever intended to be; shipping this would be a real PII regression, not a neutral parity choice | Some explicit guard (auth-gated field, opt-in display checkbox at submission time, or store-but-never-render in v2.0 with display deferred) — the seed leaves the exact policy open, but "unguarded" is not on the table |
| Treating coverage-gap results as bookable/reservable slots with locking | Natural next thought once gaps are visible ("let people claim a gap") | Turns an advisory display into a reservation system with concurrency/locking/conflict-resolution concerns — a different, much larger feature, and the milestone's own scoping explicitly treats coverage-gap as a display/analysis feature, not a booking engine | Coverage-gap analysis stays read-only/advisory; claiming a gap means submitting a normal `CampaignRun` through the existing submission form, same as any other run |

## Feature Dependencies

```
[CampaignRun data model]
    ├──requires──> [Observatory model site-code lookup] (existing, no new work)
    ├──requires──> [Target FK] (tom_targets.Target, existing)
    │
    ├──blocks──> [CSV bootstrap import]          (needs the model to import into)
    ├──blocks──> [Per-target campaign table view] (needs the model to query)
    ├──blocks──> [Submission form]                (needs the model to write to)
    └──blocks──> [Coverage-gap analysis]          (needs "claimed dates" data source)

[Submission form] ──requires──> [Approval-gate status field on CampaignRun]
[Approval-gate status field] ──enhances──> [Per-target campaign table view]
                                                (public view filters to approved only)

[Coverage-gap analysis] ──requires──> [CampaignRun data model]  (claimed side)
[Coverage-gap analysis] ──requires──> [telescope_runs.py sun_event/dark-window] (existing, observable side)
[Coverage-gap analysis] ──requires──> [Observatory model geometry]              (existing)

[CSV bootstrap import] ──validates──> [CampaignRun data model]
                                          (de-risks schema before submission form is built on top of it)

[Self-service PI bypass] ──enhances──> [Submission form + approval queue]
[Admin notification]     ──enhances──> [Submission form + approval queue]
[Collaboration-flag filtering] ──enhances──> [Per-target campaign table view]
[Collaboration-flag filtering] ──enhances──> [Coverage-gap analysis]

[Full data-file upload]        ──conflicts──> [Milestone scope: metadata-only tracker]
[Generic bot-ingestion layer]  ──conflicts──> [Milestone scope: form-based intake for non-synced facilities]
[Booking/locking on gaps]      ──conflicts──> [Coverage-gap analysis: advisory-only]
```

### Dependency Notes

- **Everything downstream needs the data model first, so it should be phase one.** The CSV bootstrap import doubles as schema validation — running it early (against real, messy data) surfaces free-text-column edge cases before the submission form's validation rules are designed, which is the same "validate against reality before building on top" lesson this codebase already learned the hard way with `SITE_TELESCOPE_MAP` (v1.2→v1.3) and multi-configuration instrument extraction.
- **The approval-gate status field is small but structurally required before the submission form ships**, not an add-on after — a submission form that writes directly to a publicly-visible table has no oversight, which is the explicit problem this milestone exists to solve (per the seed: "web form lowers the barrier to entry without bypassing oversight").
- **Coverage-gap analysis is correctly the last dependency to resolve** because it needs both halves (claimed + observable) to exist and be trustworthy first; the milestone's own note that it "can defer to v2.1 if needed" is consistent with this dependency shape — it is additive on top of a working data model + table view, not a blocker for anything else.
- **The anti-features above are dependency-shaped too, not just scope calls:** a bot-ingestion layer would need per-facility API integration work this milestone has no budget for; a booking/locking system would need concurrency-safety work that doesn't exist anywhere else in this codebase yet. Both are correctly excluded rather than partially built.

## MVP Definition

### Launch With (v1 = v2.0 milestone)

- [ ] `CampaignRun` data model — target FK, telescope/instrument, site code (Observatory lookup), obs date + UT range, filter/bandpass, observation details, status (planned→observed→reduced→published), outcome, publication plans, collaboration flag, contact person/email (PII-guarded), comments, and a separate approval-gate status (pending/approved/rejected) — essential foundation, nothing else in scope works without it
- [ ] One-off CSV bootstrap import of the real 3I/ATLAS sheet — essential to validate the schema against real messy data before a 4I ever appears (the milestone's own stated validation gate)
- [ ] Per-target campaign table view — the actual spreadsheet-replacement deliverable; without it the data model is invisible
- [ ] Community submission form (Target mandatory, everything else optional/free-text) — essential intake path per the seed, replacing direct sheet editing
- [ ] Admin approval queue (Django-admin-native actions on the pending-status subset) — essential oversight per the seed's core motivation; keep it to a single status field, no third-party package

### Add After Validation (v1.x / later in v2.0 if time allows)

- [ ] Ephemeris-aware coverage-gap analysis — trigger: once the data model + table view are proven against the bootstrap-imported real data; milestone explicitly allows deferring to v2.1
- [ ] "Open to collaboration?" filter/highlight in the table view — trigger: once basic table view is live, cheap to layer on
- [ ] Admin email notification on new pending submission — trigger: once the approval queue exists and has real submitters to notify about
- [ ] Self-service approval bypass for approved-program PIs — trigger: once the basic (always-gated) approval flow is validated and the submitter-trust design question (open in the seed) is resolved

### Future Consideration (v2.1+)

- [ ] Full data-file/data-product upload and hosting — defer: different feature class (data archive, not coordination tracker), no evidence the 3I sheet itself needed this
- [ ] Generic multi-facility bot-ingestion layer for campaign runs — defer: campaign runs are specifically the non-synced-facility case; building this would solve a problem that doesn't exist yet
- [ ] "My Targets" follow/subscribe with email alerts — defer: real value (ExoFOP's most-cited feature) but a distinct notification-infrastructure investment not mentioned in this milestone's scope
- [ ] Cross-linking `CampaignRun` rows to synced `CalendarEvent`s for the FOMO-facility subset — defer: only applicable to a minority of real campaign rows, cheap to add once both views are mature
- [ ] Booking/locking coverage gaps as reservable slots — defer indefinitely unless a real coordination-conflict problem emerges; advisory display is the correct scope

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|----------------------|----------|
| `CampaignRun` data model (incl. PII-guard + approval-gate status) | HIGH | MEDIUM | P1 |
| CSV bootstrap import of 3I sheet | HIGH (de-risks everything downstream) | MEDIUM | P1 |
| Per-target campaign table view | HIGH | LOW-MEDIUM | P1 |
| Community submission form | HIGH | MEDIUM | P1 |
| Admin approval queue (status-field + admin actions) | HIGH | LOW-MEDIUM | P1 |
| Ephemeris-aware coverage-gap analysis | HIGH (the differentiator) | HIGH | P2 |
| "Open to collaboration?" filter | MEDIUM | LOW | P2 |
| Admin notification on submission | MEDIUM | LOW | P2 |
| Self-service PI approval bypass | MEDIUM | MEDIUM | P2 |
| Cross-link campaign runs ↔ synced CalendarEvents | LOW-MEDIUM | MEDIUM | P3 |
| "My Targets" follow/subscribe | MEDIUM | MEDIUM-HIGH | P3 |
| Full data-file upload/hosting | LOW (out of stated scope) | HIGH | P3 (anti-feature, not just deferred) |
| Generic bot-ingestion layer | LOW (out of stated scope) | HIGH | P3 (anti-feature, not just deferred) |

**Priority key:**
- P1: Must have for v2.0 launch
- P2: Should have, add when possible within v2.0 or immediately after
- P3: Nice to have / explicitly out of scope, future consideration only

## Competitor Feature Analysis

| Feature | IAWN | ExoFOP-TESS | TNS | YSE-PZ | FOMO v2.0 Approach |
|---------|------|-------------|-----|--------|--------------------|
| Per-object campaign grouping | Yes, but informal (workshop/registration gate, no software artifact) | Yes, per-TOI aggregation of files/notes | No — one row per discovery, not grouped by ongoing campaign | Partial — target-centric but built for trusted collaboration members | `CampaignRun.target` FK, dedicated per-target table view |
| Community submission | Self-organized, advocacy-based, no form | Registered-user upload (data/files) | Interactive form or bulk upload, bot-dominant | Not evidenced as open/public — collaboration-member-scoped | Open web form, Target mandatory, rest optional |
| Approval/moderation gate | None (informal) | Not evidenced (upload-and-share model) | None for human submissions (immediate registry entry) | Not evidenced | Explicit pending/approved status field, admin-native actions |
| Lifecycle status | N/A | Implicit (has data or doesn't) | AT→SN reclassification only | Not evidenced beyond internal survey workflow | Explicit planned→observed→reduced→published field |
| PII handling | N/A (MPC submissions, not public profile) | Registered users, not evidenced as public-facing PII risk | Registered users required to submit | Group-permissioned, not open | Explicit PII guard on contact fields, policy TBD at phase discussion |
| Observability-vs-claimed gap analysis | No | No | No | No (has visibility tools, not gap analysis) | Genuinely novel synthesis — no reference implementation found |

## Sources

- `.planning/PROJECT.md` — milestone scope, existing FOMO capabilities (telescope_runs.py, Observatory model, calendar sync commands), requirement IDs
- `.planning/seeds/target-linked-run-submission-form.md` — the enriched seed; authoritative source for the real 3I/ATLAS sheet's field inventory and the milestone's own scoping rationale (read directly, not inferred)
- [iawn.net](https://iawn.net/) and [99942 Apophis Observing Campaigns](https://iawn.net/obscamp/Apophis/) — IAWN campaign structure (LOW confidence, general web search only)
- [The International Asteroid Warning Network Initiated a Campaign to Monitor 3I/ATLAS — Avi Loeb, Medium](https://avi-loeb.medium.com/the-international-asteroid-warning-network-initiated-a-campaign-to-monitor-3i-atlas-d2a698859747) (LOW confidence)
- [ExoFOP-TESS — MIT TESS](https://tess.mit.edu/followup/exofop-tess/), [Exoplanet Follow-up Observing Program — IPAC/Caltech](https://www.ipac.caltech.edu/project/exofop) (LOW confidence, general web search only)
- [TNS — Getting Started](https://www.wis-tns.org/content/tns-getting-started), [Home — Transient Name Server](https://www.wis-tns.org/) (LOW confidence)
- [YSE-PZ: A Transient Survey Management Platform that Empowers the Human-in-the-loop — PASP/IOPscience](https://iopscience.iop.org/article/10.1088/1538-3873/acd662), [YSE-PZ — Zenodo](https://zenodo.org/records/7278430) (LOW confidence)
- [NASA Science — Comet 3I/ATLAS](https://science.nasa.gov/solar-system/comets/3i-atlas/), [ESA — ESA observations of interstellar Comet 3I/ATLAS](https://www.esa.int/Science_Exploration/Space_Science/ESA_observations_of_interstellar_Comet_3I_ATLAS) (LOW confidence; confirms mission-level coordination exists, does not confirm the community sheet itself, which is instead sourced from the project's own seed doc)
- [The NASA Exoplanet Archive and Exoplanet Follow-up Observing Program: Data, Tools, and Usage — arXiv](https://arxiv.org/pdf/2506.03299), [ESO — Calendars and Calculators](https://www.eso.org/sci/observing/tools/ephemerides.html), [Visplot: A visibility plot and observation scheduling tool — arXiv](https://arxiv.org/pdf/2604.14151) (LOW confidence — used to confirm the *absence* of a claimed-vs-observable gap-analysis tool in the reviewed literature, not to describe a pattern to copy)
- [GitHub — django-approval](https://github.com/artscoop/django-approval), [django-moderation — PyPI](https://pypi.org/project/django-moderation/), [Implementing the Four-Eyes Principle in Django — Django Forum](https://forum.djangoproject.com/t/implementing-the-four-eyes-principle-in-django-approvals-moderation/29750) (LOW confidence, general web search only — used to describe the generic Django moderation-queue shape, deliberately recommended against as a dependency in this project; see Anti-Features)

---
*Feature research for: Campaign coordination for rare/urgent Solar System objects (FOMO v2.0)*
*Researched: 2026-07-02*
