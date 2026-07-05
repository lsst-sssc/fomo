# Research Summary — FOMO v2.1 "Uncertain Scheduling & Site Disambiguation"

**Project:** FOMO (Follow-up Observations of Moving Objects)  
**Milestone:** v2.1 "Uncertain Scheduling & Site Disambiguation"  
**Domain:** Django/TOM Toolkit campaign-coordination feature for space-mission scheduling representation  
**Researched:** 2026-07-05  
**Overall Confidence:** MEDIUM-HIGH

## POST-RESEARCH CORRECTION (operator-provided, 2026-07-05)

**Every finding below about `Observatory.obscode` needing to widen from 4 to 8+ characters is built on a false premise and should NOT be treated as a blocker.** The `'500@-170'` string this research (and the pre-existing `campaign_utils.py` docstring it was quoting) treated as "JWST's MPC obscode" is actually JPL Horizons/SPICE observer notation (`500` = geocentric-observer flag, `@-170` = JWST's NAIF SPK ID) — **not an MPC observatory code at all.**

Per the official MPC Observatory Codes list (https://www.minorplanetcenter.net/iau/lists/ObsCodes.html), real space telescopes already have standard, short MPC obscodes: **250 = Hubble, 274 = JWST, 289 = Nancy Grace Roman** — all 3 characters, well within the existing `Observatory.obscode` `max_length=4`.

**Practical effect on scope:** `Observatory.obscode` widening is very likely NOT required. The actual open questions for the phase-time spike are narrower: (a) confirm `resolve_site()`'s tier 1/tier 2 (exact match, then live MPC Obscodes API query) correctly resolves these real space-observatory codes the same way it resolves ground codes — no evidence yet either way; (b) `CreateObservatoryForm`'s hardcoded `max_length=3, min_length=3` may need to become `max_length=4` (matching the model) only if some real code needs the 4th character, not 8. Treat "does the obscode length need to change at all" as a spike question with a *default answer of no*, not a presumed P1 blocker. Everywhere below that frames the obscode length as "the hardest blocking dependency in the whole milestone" should be read with this correction in mind.

## Executive Summary

FOMO v2.1 must extend the existing campaign-coordination system to handle space-mission scheduling workflows, where observation dates are uncertain during early proposal stages and narrow over time from wide windows to fixed dates. This is a genuine schema and integration challenge, not a simple feature add: the current `CampaignRun` model assumes a single `obs_date` with optional UTC time bounds, a representation that fails entirely for "TBD pending Cycle 2" rows or "Aug 1-15" ranges — exactly the rows the real 3I/ATLAS community sheet currently contains for JWST/HST observations and that v2.0 silently drops.

Research across reference systems (JWST APT, HST Long Range Plan, space-mission ToO literature) confirms that every professional scheduler uses the same pattern: representing uncertain dates as windows that narrow over time, never as single-point null fields. FOMO's approach mirrors this exactly — replacing `obs_date` with `window_start`/`window_end` DateField pairs (both nullable to represent "TBD, no dates yet"), and adding an asset-type distinction (`Observatory.observations_type`) so ground runs claim every date in their window while space missions claim nothing until scheduling narrows them to a single concrete night.

**Core technical decisions:** One new package (`rapidfuzz` for fuzzy-match quality) is recommended, though stdlib `difflib` is a viable alternative to avoid a new dependency — this disagreement is explicitly flagged for the spike to decide. Four critical integration hazards emerged that must be guarded in implementation: (1) making `window_start` nullable while keeping the existing `UniqueConstraint` unchanged reopens the exact duplicate-row race Phase 14 already fixed — requires a `condition=Q()` partial-index constraint; (2) widening `Observatory.obscode` in the model alone is insufficient, the `CreateObservatoryForm` has independent hardcoded `max_length=3` that must be updated in parallel; (3) `CampaignRunDecisionView.post()` unconditionally re-resolves sites, silently clobbering a staff member's manual disambiguation choice — needs a guard; (4) any fuzzy-match layer that auto-selects the top candidate undermines the exact "never fabricate, always flag" invariant quick task 260705-l1v established. These are not hypotheticals — each is grounded in current production code and the milestone's own constraints.

## Key Findings

### Recommended Stack

A **single new third-party package** is necessary, with a deliberate choice between quality and zero-new-dependencies tradeoffs:

**DISAGREEMENT FLAGGED:** STACK.md recommends `rapidfuzz>=3.9` for superior fuzzy-match quality (handles transposed/reordered names, partial matches; MIT-licensed, zero runtime dependencies, prebuilt wheels). ARCHITECTURE.md recommends stdlib `difflib.SequenceMatcher` + `get_close_matches` to avoid any new dependency — a slower but sufficient approach for "a few hundred Observatory rows" scaled per-request, matching this project's existing bias toward stdlib. **Action for spike:** decide this explicitly based on match-quality testing against the real 3I sheet's actual site-name messy input; both are viable, choose deliberately.

**Everything else is existing Django:**
- `DateField(null=True, blank=True)` pair for window boundaries (no new DB-specific field type needed)
- `UniqueConstraint(fields=[...], condition=Q(...))` for the nullable natural key (requires django.db.models.Q, not a new package)
- Plain `forms.Form` (not ModelForm) for the site-disambiguation dropdown, following the existing `CampaignRunSubmissionForm` convention
- Existing `django_htmx` (already installed, used by Phase 12's calendar work) — optional for future "live re-search" interactions, not required for MVP

### Expected Features

**Table stakes (everything must-have for space-mission rows to be importable at all):**
- Window-first scheduling (replace single `obs_date` with `window_start`/`window_end` pair; single classical night modeled as `start == end`)
- Explicit "TBD" state (both fields `NULL`, orthogonal to `contact_person` attribution — a row can be TBD and still have a named person accountable)
- CSV/form intake that accepts ranges ("Aug 1-15") and TBD strings ("TBD pending Cycle 2") instead of silently dropping them
- Natural key that survives nullable `window_start` (current `(campaign, telescope_instrument, ut_start)` constraint breaks entirely)
- Ground vs. space-mission asset distinction via `Observatory.observations_type` (derived at read-time, no new field)
- `Observatory.obscode` length expansion from 4 to 8+ characters (hard blocker: JWST's `500@-170` cannot exist as a row without this)

**Differentiators (next-level competitive advantage):**
- Asset-aware coverage-gap analysis (ground window claims every date in range; space mission claims none until window narrows to `start == end`)
- Approval-queue site-disambiguation UI (fuzzy-ranked `Observatory` candidates + free-text resolve-or-create, never auto-fabricating)

**Explicitly deferred (out of v2.1 scope):**
- Full JWST-style visit-status vocabulary (duplicates existing `run_status`/`approval_status` fields FOMO already has)
- Auto-narrowing via APT/Visit-Status scraping (out-of-scope "scheduler integration" anti-pattern, per milestone intent)
- Continuous confidence-score field (no reference system uses this; discrete window/null-window model is sufficient)

### Architecture Approach

The schema change is the foundation: replace `obs_date` (single DateField) with `window_start`/`window_end` (pair of nullable DateFields), and add `window_needs_review` boolean sidecar-flag (matching the existing `site_needs_review` pattern already on `CampaignRun`). Keep `ut_start`/`ut_end` as optional DateTimeFields — they serve a *different* purpose (precise time for calendar projection) and must not be conflated with the date window. Replace the natural-key constraint `UniqueConstraint(campaign, telescope_instrument, ut_start)` with `UniqueConstraint(campaign, telescope_instrument, window_start)`, but crucially add a `condition=Q(window_start__isnull=False)` to avoid the NULL uniqueness trap (see Pitfalls section).

Three major integration patterns emerge:

1. **`campaign_gap.claimed_dates()` asset-aware rewrite** — currently it expands a single `obs_date` into a one-day claim; the new version must distinguish ground vs. space-mission runs: ground claims every date in `[window_start, window_end]`, space missions claim nothing unless `window_start == window_end` (narrowed to a single night). This is the real differentiator work.
2. **CSV import range/TBD parsing** — `parse_obs_window()` currently raises on any non-`YYYY-MM-DD` input; must now accept ranges ("Aug 1-15", "2026-08-01 to 2026-08-15") and TBD prose ("TBD pending Cycle 2"), setting `window_needs_review=True` for anything not exactly matched and never raising (no more "true natural-key failure" skip-and-log — TBD rows are *valid* entries with `window_start=NULL`).
3. **Fuzzy-match site-disambiguation UI** — add `fuzzy_match_observatories()` helper and interactive `render_site()` branch to `ApprovalQueueTable`, gated on the `show_actions` flag already used by the read-only decided table; generates ranked `Observatory` candidates for a staff member to click, never auto-selecting.

**Critical integration bug to fix in parallel:** `CampaignRunDecisionView.post()` unconditionally re-calls `resolve_site()` on every approve, overwriting `run.site`/`run.site_needs_review` regardless of whether a human already resolved it. Must add a guard: `if run.site_id is None: run.site = resolve_site(...)`  — this fix is essential to ship with the new site-disambiguation UI, or staff disambiguation gets silently clobbered.

### Critical Pitfalls (Risk Mitigation Required)

**Pitfall 1: NULL window_start defeats `UniqueConstraint` — reopens Phase 14's WR-05 race condition.** Both SQLite and PostgreSQL treat NULL as never-equal-to-itself, so two `CampaignRun` rows with identical `(campaign, telescope_instrument)` and `window_start=NULL` do not collide — unlimited duplicate TBD rows can silently accumulate under concurrent imports. **Fix:** Use `condition=Q(window_start__isnull=False)` on the constraint; TBD rows need a *different* dedup key (e.g. `window_start` itself is no longer the full story, a spike decision needed). Without this, the migration phase must explicitly test the "two TBD rows don't silently merge" scenario.

**Pitfall 2: Widening `Observatory.obscode` model field alone is insufficient.** `solsys_code_observatory/forms.py`'s `CreateObservatoryForm` independently declares `max_length=3, min_length=3` (hardcoded, not derived from the model). After a migration widens the model field, a staff member trying to create an `Observatory` for JWST's `500@-170` through the web form still gets a validation error. Also, `resolve_site()`'s call to `MPCObscodeFetcher.query()` tier 2 will uselessly hit the MPC API for every spacecraft code (which can never resolve). **Fix:** Update `CreateObservatoryForm` to accept the new max length *in parallel* with the model migration; short-circuit tier 2 for codes that don't look like real MPC codes (contain `@`, exceed the traditional 3-4 char convention).

**Pitfall 3: Fuzzy-match UI must never auto-select.** The temptation is to auto-select the top-scoring candidate above a threshold (fewer clicks for staff). This silently reintroduces the exact "fabricate a placeholder" bug quick task 260705-l1v just fixed, except now it's "silently pick the wrong existing site" which looks correct everywhere downstream (ephemeris, timezone, coverage-gap) with no warning flag. Ambiguous names like "VLT" or truncated spellings are exactly where fuzzy matching fails most, and they're exactly the inputs a public submitter (unvetted free text on the form) will provide. **Fix:** Fuzzy-match must always present *candidates for a human to pick*, never auto-select. Keep `resolve_site()`'s exact-match and API tiers as the only code paths that set `site` without human interaction; fuzzy is UI-only.

**Pitfall 4: Natural-key dedup mechanism breaks for TBD rows if reusing CR-02's offset hack.** Phase 14's CR-02 workaround adds a per-batch second-offset to `ut_start` to dedup rows that both fail UT-time parsing (making them look different). Once TBD rows genuinely have `ut_start=NULL`, there's nothing to offset — the same rows collide again. **Fix:** CSV import needs an explicit TBD-row disambiguator separate from the offset trick (e.g. a content-hash of the raw cells), decided in the spike.

**Pitfall 5: New window fields can break `insert_or_create_campaign_run()`'s `lookup` / `fields` contract.** The function treats `lookup` (natural key) and `fields` (update-only) as disjoint sets by contract. Once the natural key changes and new fields are added, it's easy to accidentally include a field in both dicts without noticing. **Fix:** Re-derive the full field list for the CSV import call site explicitly before implementation; verify no field name appears in both `lookup` and `fields` dicts via a test assertion at the actual call site.

## Implications for Roadmap

Research indicates a clear, dependency-ordered build sequence. The milestone explicitly includes a phase-time investigation spike (not a separate preceding research phase) — everything below assumes the spike confirms the architecture recommendations.

### Suggested Phase Sequence

**Phase 0: Investigation Spike (Within Milestone Scope)**
- **Rationale:** Multiple open decisions (exact window schema, TBD natural-key replacement, CSV range/TBD parsing patterns, fuzzy-library choice, obscode max-length target) must be decided against real 3I sheet rows before implementation, not discovered afterward.
- **Delivers:** Confirmations of: window field names/nullability, natural-key for TBD rows, space-mission "narrowing trigger" rule (when does `window_start == window_end` claim dates?), CSV range/TBD shapes found in the real sheet, fuzzy-library tradeoff decision, safe obscode length target (8 chars sufficient for all spacecraft?).
- **Avoids:** Pitfalls 1, 2, 5 (all require spike-decided schema before implementation).
- **Research flag:** None — spike is research by definition; proceed with the ARCHITECTURE.md recommendations as the default proposal to validate.

**Phase 1: `Observatory.obscode` Max-Length Widening**
- **Rationale:** Small, independent, hard-blocker for all space-mission work downstream. No space-mission `Observatory` row can exist (and no asset-type distinction can be validated) until this lands. Ship it *before* the `CampaignRun` schema migration to reduce blast radius.
- **Delivers:** `Observatory.obscode` CharField widened (per spike confirmation, likely 8 chars); `CreateObservatoryForm.obscode` updated in parallel; tier-2 MPC API short-circuited for spacecraft-style codes; migration verified against existing `Observatory` rows.
- **Implements:** ARCHITECTURE A1's "Blocking prerequisite" section, fully addressing Pitfall 3.
- **Testing:** Manual/UAT — a staff user creates an `Observatory` for an 8-character spacecraft code through the actual web form (not just ORM), and `resolve_site()` respects it without re-querying the MPC API.

**Phase 2: `CampaignRun` Window-Field Schema Migration**
- **Rationale:** Largest blast-radius change (touches `CampaignRunTable`, `ApprovalQueueTable`, `CampaignRunSubmissionForm`, `CampaignRunDecisionView`, import pipeline, gap analysis). Land it as its own phase before downstream features, per ARCHITECTURE A4's checklist. Prerequisite: Phase 1 complete (obscode has no dependency on this, so Phase 1 is truly independent; vice versa is not true).
- **Delivers:** `window_start`/`window_end` DateFields added, `window_needs_review` sidecar flag; `obs_date` removed; `UniqueConstraint` replaced with window-keyed version + `condition=Q(window_start__isnull=False)` partial-index guard (both SQLite 3.8.0+ and PostgreSQL); data migration existing `obs_date` rows to `window_start=window_end=obs_date`; `ut_start`/`ut_end` retained and re-audited for dependencies; all consumers updated (table columns, form fields, view kwargs, import pipeline).
- **Implements:** ARCHITECTURE A1 schema recommendation, addressing Pitfall 1.
- **Testing:** Two explicit test cases: (a) a single TBD row round-trips correctly; (b) two distinct TBD rows for the same campaign+telescope via concurrent/re-run import both fail to merge into one row and don't silently duplicate under the DB constraint.

**Phase 3: Asset-Aware Coverage-Gap Analysis**
- **Rationale:** Depends only on Phases 1 & 2 (needs window schema + asset-type to be checkable). Can run in parallel with Phase 4 (CSV parsing) once Phase 2 lands. This is the real differentiator work — the complex, value-add logic.
- **Delivers:** `campaign_gap.claimed_dates()` rewritten to distinguish ground vs. space-mission runs; ground claims every date in window, space missions claim nothing unless narrowed to single night (per spike's narrowing-trigger decision). New `pending_narrowing_runs` list bucket surfaced in gap-analysis view alongside existing `undated_runs`/`unattributed_runs`.
- **Implements:** ARCHITECTURE A2, Features differentiator "asset-aware coverage-gap analysis".
- **Testing:** Gap analysis against a real mix of ground and space-mission rows shows ground dates claimed, space-mission wide windows not claimed, narrow windows claimed.

**Phase 4: CSV Import Range/TBD Parsing**
- **Rationale:** Depends only on Phase 2 (needs window schema). Can run in parallel with Phase 3. Mirrors the existing narrow-regex precedent from `_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC` — enumerate actual shapes from real 3I sheet during spike, one pattern per shape, fall back to `window_needs_review=True` never raise.
- **Delivers:** `parse_obs_window()` rewritten to accept ranges and TBD text; "true natural-key failure" case eliminated (no more raises, TBD rows with `window_start=NULL` are valid); explicit TBD-row disambiguator (content-hash per spike) replaces CR-02's offset trick for the new case; paired demo notebook `import_campaign_csv_demo.ipynb` updated if behavior changes.
- **Implements:** ARCHITECTURE A3, addressing Pitfall 2 and Pitfall 4.
- **Testing:** Two distinct TBD rows in one CSV import produce two distinct `CampaignRun` rows (not merged); range like "Aug 1-15" parsed correctly; unrecognized text like "TBD pending Cycle 2" accepted with `window_needs_review=True`.

**Phase 5: Site-Disambiguation UI (Fuzzy-Match Dropdown)**
- **Rationale:** Has *no dependency* on Phases 2-4 (it only touches `Observatory` resolution, not scheduling fields). Can run first, last, or in parallel with the window/asset work. The one hard rule: the `CampaignRunDecisionView.post()` guard fix (B4) must ship in the *same phase* as the new `CampaignRunSiteResolutionView` endpoint (B3), never split.
- **Delivers:** `fuzzy_match_observatories()` helper in `campaign_utils.py` (using difflib or rapidfuzz per spike decision) accepting `name`/`short_name`/`old_names` candidates; `ApprovalQueueTable.render_site()` interactive branch gated on `show_actions`; new `CampaignRunSiteResolutionView` (StaffRequiredMixin, POST-only) for resolving a site to a fuzzy candidate or free-text fallback (→ existing `CreateObservatory` form with `?next=` back to approval queue); **critical fix**: `CampaignRunDecisionView.post()` guarded to skip `resolve_site()` if `run.site_id is not None` (avoid clobbering human choices).
- **Implements:** ARCHITECTURE B1-B4, Features differentiator "approval-queue site-disambiguation UI", addressing Pitfall 3.
- **Testing:** Ambiguous free-text site name (not exact typo) always requires explicit human click before `site` is set; `site_needs_review` never silently flips to `False` via auto-select. Staff can resolve a site via the UI, then click Approve without the site being clobbered.

**Phase 6: VIEW-05 Submitter Contact Opt-In**
- **Rationale:** Fully independent (one new form field + conditional submission logic). Good low-risk final phase or can run anywhere. No dependency on any above phases.
- **Delivers:** New `contact_person_opt_in` (or equivalent) BooleanField on submission form; contact fields only populated if opt-in checked; submission-form and table-view logic updated.
- **Implements:** Features table-stakes "VIEW-05 combined submitter contact opt-in".
- **Testing:** Opting out omits contact from the submitted row; opting in populates it.

### Phase Ordering Rationale

1. **Spike first** (implicit, within milestone scope) — settles all open decisions.
2. **Phase 1 (obscode)** is the true root blocker and smallest-blast-radius change; ship it independently before the big schema migration.
3. **Phase 2 (window schema)** is the foundation every downstream feature depends on; land it early and thoroughly.
4. **Phases 3-4 (gap analysis + CSV parsing)** can run in parallel once Phase 2 lands — both are "consumers" of the new schema, not interdependent.
5. **Phase 5 (site UI)** and **Phase 6 (contact opt-in)** are independent of the above; can run any time, but Phase 5's B4 guard fix is entangled with B3 (must ship same phase).

### Parallelization Opportunity

Once Phase 2 (schema migration) lands:
- Phases 3 & 4 can run concurrently (asset-gap vs. CSV parsing, both ready immediately after schema).
- Phase 5 & 6 can run any time, independent of everything else, even starting before Phase 2 if preferred.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Fuzzy-library choice is explicitly disagreed between STACK.md and ARCHITECTURE.md (rapidfuzz quality vs. stdlib zero-dependency); both viable, spike must decide. Everything else (plain Django, DateFields, forms) is HIGH-confidence established pattern. |
| Features | MEDIUM | Table-stakes features are well-grounded in reference systems and milestone intent (HIGH confidence). Differentiators (asset-gap, fuzzy UI) are less-commonly-implemented but well-understood pattern (MEDIUM confidence). Anti-features are sound (MEDIUM confidence). |
| Architecture | HIGH | All findings grounded directly in this repo's current source code (models, views, forms, utils, migrations). No external ecosystem research — this is pure integration design against known dependencies. |
| Pitfalls | HIGH | All pitfalls are grounded in repo code inspection and empirically verified against SQL NULL-uniqueness behavior. Cross-verified against Django documentation and existing Phase 14 decisions (WR-05). |
| **Overall** | **MEDIUM-HIGH** | Roll-up of above. The spike (resolving the MEDIUM stack point and confirming architecture recommendations) is the only significant risk item before implementation can proceed with HIGH confidence. |

## Gaps to Address

**During the phase-time investigation spike:**
- Confirm window schema (field names, nullability, sidecar flags) against real 3I sheet rows imported with ranges/TBD text.
- Decide the TBD-row natural key explicitly (is `window_start` alone sufficient, or does it need a secondary disambiguator?).
- Enumerate actual CSV range/TBD text patterns in the real 3I sheet (don't guess a generic parser).
- Decide fuzzy-library tradeoff (rapidfuzz quality vs. stdlib zero-dependency) via match-quality testing.
- Confirm `Observatory.obscode` max-length target is safe for all real spacecraft codes (Gaia, Spitzer, JWST, HST, IceSat-2, etc. all use `@`-prefixed heliocentric/L2 MPC-style codes of similar length).

**Before the site-disambiguation UI implementation phase:**
- Establish logging/acceptance-rate tracking for fuzzy suggestions (for future threshold-tuning passes).
- Document the "never auto-select, always require human click" rule explicitly in the plan's UAT criteria.

**Before the gap-analysis phase:**
- Decide the exact "narrowing trigger" rule (this document recommends `window_start == window_end`, but spike must validate against real JWST scheduling practices).

## Sources

### Primary Research Files (HIGH confidence)

- `.planning/research/STACK.md` — Technology stack recommendations for v2.1, including the rapidfuzz vs. difflib disagreement, version compatibility, and alternative considerations. Read 2026-07-05.
- `.planning/research/FEATURES.md` — Feature landscape research covering table-stakes, differentiators, anti-features, reference-system comparisons (JWST APT, HST LRP, space-mission ToO literature). Read 2026-07-05.
- `.planning/research/ARCHITECTURE.md` — Integration architecture for range/window scheduling, asset-type distinction, gap analysis, and fuzzy-match UI, grounded in current repo source. Read 2026-07-05.
- `.planning/research/PITFALLS.md` — Five critical pitfalls with risk mitigation strategies, grounded in repo code inspection and SQL/Django documentation. Read 2026-07-05.

### Project Context (HIGH confidence)

- `.planning/PROJECT.md` — v2.1 milestone scope, Active requirements, Key Decisions log (D-04, D-05, D-08, D-09 re: natural keys; WR-01–WR-08 re: timeouts; CR-01, CR-02 re: offset disambiguation; 260705-l1v re: "never fabricate" invariant).
- `solsys_code/models.py`, `solsys_code/campaign_*.py`, `solsys_code/solsys_code_observatory/models.py` — Current schema, views, forms, and utils (read 2026-07-05 for this research pass).

### External Reference Systems (MEDIUM-LOW confidence)

- [STScI Visit Status Help — JWST](https://www.stsci.edu/public/help/visit-help-JWST.html) — Official documentation for "plan window not yet assigned" state and ~8-week initial window. Used to validate the milestone's own window-narrowing architecture approach.
- [JWST/HST documentation on scheduling windows and visit status](https://jwst-docs.stsci.edu, https://www.stsci.edu/hst/) — General web search, treated as MEDIUM-LOW confidence per this project's generic-webfetch-provider classification.
- Space-mission ToO literature (JWST, ESO, Gemini public documentation) — Confirms response-time-class pattern (Rapid/Hard/Soft ToO) as orthogonal to this milestone's window-narrowing pattern; used to justify the "do not reuse Rap/Std model for community-submitted space-mission rows" anti-feature recommendation.

---

**Research completed:** 2026-07-05  
**Ready for roadmap planning:** Yes — spike must precede all implementation phases to confirm the MEDIUM-confidence stack and architecture decisions.
