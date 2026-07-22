# Feature Research

**Domain:** Uncertain scheduling representation for community campaign coordination (FOMO v2.1 — window-first `CampaignRun` scheduling, ground-vs-space-mission asset distinction, range/TBD CSV import, asset-aware coverage-gap, site disambiguation)
**Researched:** 2026-07-05
**Confidence:** MEDIUM overall. HIGH for "what the real 3I/ATLAS sheet needs" (the milestone brief itself, in `.planning/PROJECT.md`, directly describes the reference rows — Carrie Holt/Martin Cordiner JWST rows with uncertain dates — as first-hand ground truth, not inferred). LOW-MEDIUM for "how space-mission scheduling tools formally represent windows/TBD" (general web search + one direct fetch of STScI's own public JWST visit-status help page — official-source content, but the tool-classified confidence tier for `websearch`/`webfetch` providers is LOW per this project's source-hierarchy classifier, so treat specifics as directionally right, not verbatim-verified). No access to the actual 3I/ATLAS Google Sheet or to APT/Visit Planner software itself — everything about JWST/HST internals is from public documentation, not hands-on use.

## Headline Finding (read this first)

**Every space-mission scheduling system found (JWST APT, HST LRP) solves this exact problem — representing "we don't have an exact date yet" — with the same shape: a *window that narrows over time*, never a single nullable date field.** JWST's own visit-status vocabulary makes the "not scheduled yet" state explicit and named (`Flight Ready` / "plan window not yet assigned"), then assigns a wide window (~8 weeks) once long-range planning acts, then narrows to an exact start/end only about a week before execution. HST's Long Range Plan does the same two-stage narrowing (annual candidate windows → week-by-week exact schedule), with windows from half a day to eight weeks wide. Neither system ever has a row that is simultaneously "approved" and "has one fixed date" before execution is imminent — the date field's *cardinality* changes over the object's lifecycle (none → range → point), not just its value.

**This directly validates the milestone's own architectural instinct** (a window replaces the single `obs_date`/`ut_start`/`ut_end` triple, with a classically-scheduled single night modeled as a degenerate 1-day window) rather than inventing something novel. The one thing neither JWST nor HST needs, but this milestone does, is a **third state below "window": no window at all yet** ("TBD pending Cycle 2" in the milestone's own example) — closer to ToO literature's practice of tracking a request's *validity period* (start/expiry, from trigger to deadline) with no target execution date at all until a night-by-night observability computation narrows it. FOMO's own `sync_gemini_observation_calendar.py` (GEM-WINDOW-02) already implements exactly this narrowing pattern for Rap:/Std: ToO defaults — this milestone is applying the same idea to the community-submitted, human-typed side of the sheet instead of an API-supplied one.

## Feature Landscape

### Table Stakes (Users Expect These)

Features a coordination tool must have to match what the real 3I/ATLAS sheet and comparable space-mission scheduling tools already do. Missing these = the sheet's harder rows either get silently dropped (current v2.0 behavior) or coordinators stop trusting the tool for space-mission rows.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Window (start date + end date) as the *primary* scheduling representation, with a single classical night modeled as `start == end` (1-day window) | Matches JWST/HST scheduling windows (8 weeks / half-day–8-weeks) and the real sheet's own "Aug 1–15" style cells; a single nullable `obs_date` cannot represent a range at all, which is exactly why v2.0's `parse_obs_window()` currently drops these rows as natural-key failures | MEDIUM | This is a real schema migration (per milestone scope), not additive — every existing `CampaignRun` row (currently one obs_date + one UT range) needs a migration path to `start == end`. Depends on: existing `CampaignRun` model, `import_campaign_csv`/`parse_obs_window()` |
| An explicit "no date yet" state, distinct from both "window" and "single night" | JWST's own vocabulary names this state (`Flight Ready`, "plan window not yet assigned") rather than leaving it implicit; the real sheet's "TBD pending Cycle 2" cells are exactly this — an approved run with zero temporal information yet | LOW-MEDIUM | Should be representable without inventing sentinel dates (e.g. `NULL` start/end, not `1900-01-01`) — a null window is a real, named lifecycle state, not "missing data"; downstream code (coverage-gap, calendar projection) must treat "no window" as a defined branch, not an unhandled case |
| Ground vs. space-mission distinction driving date-parsing and coverage-gap behavior differently, without a new `CampaignRun` field | Every reference system's date-certainty behavior is driven by *mission/facility type*, not a per-row flag a submitter fills in — JWST rows are inherently long-lead/narrowing, classical ground nights are inherently exact. Reusing `Observatory.observations_type` (`SATELLITE_OBSTYPE`) is the correct place for this, matching the milestone's own decision | LOW-MEDIUM | Pure derivation from `site` FK — no new column, no submitter-facing choice to get wrong. Depends on: existing `Observatory.observations_type` |
| CSV/form intake that accepts a range or a TBD-style free-text cell instead of skip-and-log | Directly what breaks today — real 3I sheet rows for space-mission observations use exactly this style, and v2.0's importer currently treats "doesn't parse as `YYYY-MM-DD`" as a natural-key failure per D-05, silently dropping legitimate rows | MEDIUM-HIGH | The parsing itself is the milestone's own flagged spike question ("range like 'Aug 1-15' or a 'TBD pending Cycle 2' cell") — free text in the wild will not be fully enumerable; budget for a best-effort parser with an honest "still couldn't parse, needs human review" fallback bucket (same shape as the existing `site_needs_review` summary bucket in `import_campaign_csv`), not a silent drop |
| A stable natural key that survives a row having no fixed `ut_start` | Current `(campaign, telescope_instrument, ut_start)` uniqueness breaks the moment a legitimate row has no start time at all — this is called out explicitly in the milestone's Key Context | MEDIUM | Needs a replacement key that doesn't require an exact timestamp — e.g. `(campaign, telescope_instrument, window_start_or_null, submitted PI/contact)` or a stable identity taken from the CSV row itself (row index / hash of telescope+contact) for TBD rows. This is squarely the milestone's own spike question, not something general research resolves — flag for phase-time investigation, not implementation-by-analogy |
| Named-individual attribution surviving on uncertain-date rows (the "Carrie Holt"/"Martin Cordiner" pattern) | The real sheet keeps a PI/contact name attached to a row even when the date is completely unknown — this is exactly what `contact_person` already exists for in `CampaignRun`; a TBD row is not a reason to lose the human accountable for it | LOW | No new field needed — `contact_person`/`contact_email` already exist on `CampaignRun` from v2.0 and are orthogonal to whether a date is known. Just needs the CSV/form path to not require a date before accepting a contact |
| Site-code length that accommodates spacecraft identifiers, not just MPC ground-site codes | `Observatory.obscode` is `CharField(max_length=4)` but JWST's own MPC-style code (`500@-170`) is 8 characters — flagged directly in the milestone's Key Context as a hard blocker; without this fix, no space-mission `Observatory` row can exist at all, which blocks every other feature in this list | LOW-MEDIUM (schema) / spike-dependent (design) | This is a genuine pre-existing constraint bug surfaced by adding space-mission rows, not new functionality — straightforward `max_length` migration once the spike confirms 8 characters is sufficient for all needed spacecraft codes (Gaia, Spitzer, JWST, HST, etc. all use `@`-prefixed heliocentric/L2 MPC-style codes of similar length) |

### Differentiators (Competitive Advantage — Defer if Needed)

Features that go beyond "match what the sheet already does" toward something a spreadsheet or the reference space-mission tools don't offer.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Asset-aware coverage-gap analysis (ground window claims every date in range; space-mission claims nothing until scheduling narrows to something concrete) | No reference system computes "observable AND unclaimed" at all (per v2.0 research), and none of them additionally distinguish *how confidently claimed* a date is by asset type. This is the natural, and harder, evolution of the v2.0 differentiator — a coordinator sees "this JWST row exists but claims zero dates yet, so don't treat those nights as spoken-for" | HIGH | The exact narrowing trigger (when does a space-mission row's window collapse from "claims nothing" to "claims dates"?) is explicitly flagged as a spike question in the milestone, mirroring how JWST's own visit-status vocabulary treats this as a discrete lifecycle transition (`Flight Ready`→`Scheduled`) rather than a continuous confidence score — recommend modeling it the same way (a status/threshold transition, not a probability), rather than inventing a novel confidence metric. Depends on: v2.0 `campaign_gap.py`, `Observatory.observations_type` |
| Approval-queue site-disambiguation UI (fuzzy-matched `Observatory` candidates + free-text resolve-or-create, never auto-fabricating) | Directly extends quick task `260705-l1v`'s visibility fix into an actual resolution workflow — no reference system was found doing fuzzy site-name matching against a controlled MPC-obscode registry in a moderation UI; this is FOMO-specific tooling, not adapted from an external pattern | MEDIUM | Not really informed by external research (no comparable reference system) — this is closing FOMO's own internal gap, best resolved by direct product-design work at plan time rather than further ecosystem research. Depends on: `resolve_site()`'s existing tier 1/2 matching, `CampaignRun.site` FK |
| Visibly distinguishing "window narrowing over time" in the UI (e.g. showing a TBD row transition to a range, then to a fixed night, as the real mission schedule firms up) | JWST's own Visit Status page literally does this (shows plan-window-not-yet-assigned → window → exact start/end as time passes) — replicating that same progressive-disclosure idea in the per-campaign table would be a genuine UX improvement over the flat spreadsheet, which just gets manually edited in place with no history | MEDIUM-HIGH | Requires either an edit-history mechanism or accepting that "narrowing" is just a normal update to the same row (simplest, matches the sheet's own editing model) — recommend the latter for this milestone; flag full history/audit trail as a future consideration, not v2.1 scope |

### Anti-Features (Commonly Requested, Often Problematic)

Features that would seem natural extensions of "handle uncertain dates better" but would over-scope this milestone.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Building a full JWST-APT-style "Visit Status" state machine (Pending Submission → Implementation → Flight Ready → Scheduled → Executed → Collecting → Completed → Archived, 14 states) | JWST's vocabulary is detailed, official, and directly analogous — tempting to import wholesale as "the" answer for uncertain-scheduling status | This is a proposal-*execution*-tracking vocabulary built for STScI's own internal scheduling pipeline (guide-star availability, flight scheduling, archive ingest) — FOMO has no equivalent internal pipeline for space-mission observations; it is a passive coordination hub, not a scheduler. Importing 14 states would create fields nothing in FOMO ever transitions, and would duplicate/conflict with `CampaignRun`'s existing `run_status` (planned→observed→reduced→published) and `approval_status` (pending/approved/rejected), which already cover FOMO's actual lifecycle needs (per D-02 in v2.0) | Reuse the existing two-status-field model; represent scheduling *uncertainty* purely through the window/null-window date representation (this milestone's actual scope), not a parallel execution-status vocabulary |
| A generic "confidence score" or probability field for how likely a date range is to hold | Feels like a natural way to express "this JWST window is pretty firm" vs. "this is a wild guess" on a continuous scale | No reference system (JWST, HST, ToO literature) uses a continuous confidence score for scheduling — they all use discrete, named lifecycle states (has a window / doesn't have a window / window has narrowed to an exact date). A confidence score invites false precision this milestone has no data to support and no consumer (coverage-gap, table view) that would meaningfully use a float over a simple three-state model | The window's own presence/absence and width already encode this: no window = least certain, wide window = some certainty, 1-day window = fully certain. No separate field needed |
| Auto-narrowing a space-mission window automatically via a scheduled sync against STScI's public APT/Visit Status pages | Since the vocabulary/pattern exists publicly, it's tempting to think FOMO could scrape or poll it directly, closing the loop without a human editing the sheet-replacement table | Out of scope for a "lightweight coordination hub" — this would require a new per-mission scraping/API-integration surface (STScI has no public bulk API for this, only per-program HTML help pages), mirroring the exact over-scope anti-pattern the v2.0 research already flagged ("generic bot-ingestion layer... campaign runs are explicitly the out-of-sync-command case") | Keep human-submitted/human-edited updates as the only path for space-mission row narrowing, same as every other campaign-run field; if STScI integration is ever wanted, treat it as a wholly separate future milestone, not bundled into v2.1 |
| Response-time-class buckets (Rapid/Hard/Soft ToO, hours-to-days) as the vocabulary for space-mission uncertainty | ToO literature and FOMO's own existing Gemini sync (`Rap:`/`Std:`) already use exactly this pattern, so it's tempting to reuse the same buckets for campaign-run space-mission rows | Wrong shape for this milestone's actual data: Gemini ToO windows are computed from a *known trigger time* (`ObservationRecord.created`) plus a fixed offset; 3I/ATLAS space-mission rows have no trigger timestamp at all, just a human-typed "TBD pending Cycle 2" — there is nothing to offset from. Applying the Rap/Std bucket model here would fabricate false precision from data that doesn't exist | Model space-mission uncertainty as the window/null-window representation (this milestone's scope), not as a trigger-relative offset; reserve the existing Rap:/Std: pattern for what it already correctly serves (Gemini ToO sync), don't generalize it to community-submitted rows with no trigger data |

## Feature Dependencies

```
[Observatory.obscode max_length fix (4 -> 8+ chars)]
    └──blocks──> [Any space-mission Observatory row existing at all]
                     └──blocks──> [Ground vs. space-mission asset distinction]
                                      └──blocks──> [Asset-aware coverage-gap analysis]

[Window-first CampaignRun schema (start/end replaces obs_date/ut_start/ut_end)]
    ├──requires──> [Migration path for every existing CampaignRun row] (start == end degenerate case)
    ├──blocks──> [CSV import range/TBD parsing]
    ├──blocks──> [Asset-aware coverage-gap analysis]        (needs a window, not a point, to reason about)
    └──blocks──> [Natural-key replacement]                  (ut_start no longer guaranteed to exist)

[Natural-key replacement] ──requires──> [Window-first schema] + [milestone spike decision]

[CSV import range/TBD parsing] ──requires──> [Window-first schema]
                               ──requires──> [best-effort parser + honest "needs review" fallback bucket]
                                                 (mirrors existing site_needs_review pattern)

[Ground vs. space-mission asset distinction] ──requires──> [Observatory.observations_type] (existing, no new work)
                                              ──enhances──> [Asset-aware coverage-gap analysis]

[Site-disambiguation UI] ──requires──> [resolve_site() tier 1/2] (existing)
                         ──extends──> [quick task 260705-l1v fix]

[VIEW-05 contact opt-in] ──independent of the above──> [pure submission-form + table-view change]
```

### Dependency Notes

- **The `Observatory.obscode` length fix is the true root blocker for this milestone**, more so than the schema migration itself — nothing space-mission-related (asset distinction, coverage-gap asset-awareness, even just creating a JWST `Observatory` row to resolve a site against) can exist until it lands. It should be the first concrete implementation task, immediately after the spike settles on a safe max length.
- **The window-schema migration is the second root blocker.** It touches every existing row (per milestone scope, explicitly not additive), so it should land before CSV-parsing changes and before coverage-gap logic is touched — trying to add range/TBD parsing on top of the old single-`obs_date` schema would mean building throwaway logic twice.
- **The natural-key replacement is entangled with the window schema, not separable from it** — per the milestone's own Key Context, the current `(campaign, telescope_instrument, ut_start)` key assumes a fixed start time exists; this can only be resolved once the spike decides what a "no window yet" row's stable identity actually is. Recommend resolving this as part of the same spike, not a separate design pass.
- **CSV range/TBD parsing depends on, but is more open-ended than, the schema migration** — this is squarely a "spike, then implement" item, since real-world free text (per the milestone's own "Aug 1-15" / "TBD pending Cycle 2" examples) will have long-tail formats no fixed grammar fully covers. Recommend the same skip-and-log-with-summary pattern `import_campaign_csv` already uses for site resolution (`site_needs_review`), applied to date parsing, rather than assuming a parser can be complete.
- **Asset-aware coverage-gap analysis is correctly the most downstream item** — it needs the site-type distinction, the window schema, and a settled narrowing-trigger decision all to exist first. This mirrors the exact same "last, because it composes everything else" dependency shape v2.0's research already identified for the original coverage-gap feature.
- **Site-disambiguation UI and VIEW-05 are structurally independent of the scheduling-representation work** — they can be built and shipped in parallel with the window/asset-distinction phases without blocking or being blocked by them, since neither touches `CampaignRun`'s date fields.

## MVP Definition

### Launch With (v1 = v2.1 milestone, per PROJECT.md's Active requirements)

- [ ] Phase-time investigation spike settling: window schema shape, range/TBD parsing rules, natural-key replacement, fuzzy-match approach, and the `Observatory.obscode` 4-vs-8-char fix — essential, this milestone explicitly puts the spike inside the milestone rather than deferring it
- [ ] `Observatory.obscode` length fix — hard blocker for every space-mission-related feature below
- [ ] Window-first `CampaignRun` scheduling (start/end replaces obs_date/ut_start/ut_end; single night = 1-day window) — essential foundation, real schema migration
- [ ] Ground vs. space-mission asset distinction via `Observatory.observations_type` — essential, no new field, pure derivation
- [ ] CSV import handles range/TBD `Obs. Date` text — essential, this is what breaks today against the real sheet
- [ ] Asset-aware coverage-gap analysis — essential per milestone scope (ground claims every date in window; space-mission claims none until narrowed)
- [ ] Approval-queue site-disambiguation UI (fuzzy-match dropdown + free-text resolve-or-create, no auto-fabrication) — essential, closes the quick-task-260705-l1v loose end
- [ ] VIEW-05 combined submitter contact opt-in — essential per milestone scope, independent of the scheduling work

### Add After Validation (not explicitly in v2.1 scope, but natural next steps if time allows)

- [ ] Progressive-disclosure UI showing a row's window narrowing over time (TBD → range → exact night) in the per-campaign table — trigger: once the underlying window schema is proven correct against re-imported real data

### Future Consideration (beyond v2.1)

- [ ] Any STScI APT/Visit-Status scraping or sync integration — defer indefinitely; explicitly an anti-feature for this milestone's scope (lightweight coordination hub, not a scheduler-integration project)
- [ ] Continuous confidence-score field for date certainty — defer indefinitely; no reference system uses this pattern and the discrete window/null-window model already covers the real need
- [ ] Full JWST-style multi-state visit-status vocabulary — defer indefinitely; duplicates existing `run_status`/`approval_status` fields, solves a scheduling-pipeline problem FOMO doesn't have

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|----------------------|----------|
| Phase-time investigation spike | HIGH (de-risks everything downstream) | MEDIUM | P1 |
| `Observatory.obscode` length fix | HIGH (hard blocker) | LOW | P1 |
| Window-first `CampaignRun` scheduling | HIGH | MEDIUM-HIGH (real migration) | P1 |
| Ground vs. space-mission asset distinction | HIGH | LOW | P1 |
| CSV import range/TBD parsing | HIGH | MEDIUM-HIGH (open-ended free text) | P1 |
| Asset-aware coverage-gap analysis | HIGH (differentiator) | HIGH | P1 |
| Site-disambiguation UI | HIGH (closes known gap) | MEDIUM | P1 |
| VIEW-05 contact opt-in | MEDIUM | LOW | P1 |
| Progressive-disclosure "window narrowing" UI | MEDIUM | MEDIUM | P2 |
| STScI APT/Visit-Status integration | LOW (out of stated scope) | HIGH | P3 (anti-feature) |
| Continuous confidence-score field | LOW (out of stated scope) | MEDIUM | P3 (anti-feature) |
| Full JWST-style visit-status vocabulary | LOW (duplicates existing fields) | HIGH | P3 (anti-feature) |

**Priority key:**
- P1: Must have for v2.1 launch (all currently in PROJECT.md's Active requirements)
- P2: Should have if time allows, natural next step
- P3: Explicitly out of scope, future consideration only or anti-feature

## Reference System Comparison

| Dimension | JWST APT / Visit Planner | HST Long Range Plan | General ToO literature | 3I/ATLAS sheet (this milestone's actual target) | FOMO v2.1 Approach |
|-----------|---------------------------|----------------------|--------------------------|---------------------------------------------------|---------------------|
| "No date yet" state | Named, explicit: `Flight Ready`, "plan window not yet assigned" | Implicit (pre-LRP-assignment) | Request validity period (start/expiry), no target date | Free-text "TBD pending Cycle 2" | Null start/end window, a defined branch not an error case |
| Range representation | Plan window, ~8 weeks wide | Scheduling window, half-day to 8 weeks | Union of per-night "night windows" across validity period | Free-text "Aug 1-15" | `start`/`end` date pair; single night = `start == end` |
| Narrowing trigger | Time-based: ~7 days before execution | Week-by-week schedule draw from annual LRP | Observability computation per night | Unstated/manual (sheet is hand-edited) | Explicit spike decision required (milestone's own flagged open question) |
| Attribution on uncertain rows | N/A (internal STScI pipeline, not community-facing) | N/A | N/A | PI/contact name kept regardless of date certainty (Carrie Holt/Martin Cordiner) | `contact_person`/`contact_email` already exist, orthogonal to date certainty (no change needed) |
| Asset-type-driven behavior | N/A (single mission) | N/A | Facility-type-driven (ground vs. space largely separate literatures) | Implicit — ground rows have exact dates, space-mission rows don't, by nature of the facility | Explicit, derived from `Observatory.observations_type`, no submitter-facing flag |
| Coverage/claim semantics | N/A (not a multi-observer coordination tool) | N/A | N/A | N/A (no gap-analysis concept in the sheet itself) | Genuinely novel — ground window claims every date, space-mission claims none until narrowed |

## Sources

- `.planning/PROJECT.md` — v2.1 milestone scope (`Current Milestone: v2.1` section), Active requirements, Key Context (`Observatory.obscode` 4-vs-8-char constraint, natural-key TBD-breakage), existing v2.0 shipped capabilities (`CampaignRun`, `import_campaign_csv`, `campaign_gap.py`) — read directly, authoritative for this project's own scope and constraints
- [Visit Status Help — STScI](https://www.stsci.edu/public/help/visit-help-JWST.html) — direct fetch, official STScI public documentation; source of the full 14-state visit-status vocabulary and the "plan window not yet assigned" / 8-week-window / ~7-day-narrowing quotes. Classified LOW confidence by this project's generic webfetch-provider tier, but is itself STScI's own authoritative public help page — treat as directionally reliable, not independently cross-verified against a second source
- [APT Visit Planner — JWST User Documentation](https://jwstcf.stsci.edu/jwst-astronomers-proposal-tool-overview/apt-workflow-articles/apt-visit-planner) (LOW confidence, general web search)
- [Scheduling — STScI](https://www.stsci.edu/hst/observing/scheduling), [Orbital Visibility and Scheduling — STScI](https://www.stsci.edu/hst/proposing/phase-i/proposal-planning-toolbox/orbital-visibility-and-scheduling) — source of the "half a day to eight weeks" HST scheduling-window figure and the two-stage LRP-then-weekly-schedule process (LOW confidence, general web search)
- [JWST Target of Opportunity Observations — JWST User Documentation](https://jwst-docs.stsci.edu/methods-and-roadmaps/jwst-target-of-opportunity-observations), [Target of Opportunity (ToO) — Gemini Observatory](https://www.gemini.edu/observing/phase-i/too), [ESO — Target of Opportunity Proposals](https://www.eso.org/sci/observing/policies/too_policy.html) — source of the Rapid/Hard/Soft ToO response-time-class pattern and "night window"/"observation window" (union across validity period) terminology (LOW confidence, general web search)
- [JWST NIRSpec: 3I/ATLAS Interstellar Probe](https://www.emergentmind.com/topics/jwst-nirspec-campaign-for-3i-atlas), [JWST detection of a carbon dioxide dominated gas coma surrounding interstellar object 3I/ATLAS — arXiv](https://arxiv.org/abs/2508.18209) — confirms the real JWST/Cordiner NIRSpec campaign for 3I/ATLAS exists and is a genuine, active multi-instrument coordination effort; did not surface the community coordination Google Sheet itself (not indexed/public), so the sheet's actual field content is sourced from the milestone's own first-hand description in PROJECT.md/prior seeds, not from this search (LOW confidence, general web search)
- v2.0 prior research (`.planning/research/FEATURES.md`, superseded by this file) — carried-forward context on IAWN/ExoFOP-TESS/TNS/YSE-PZ as reference community-coordination systems; none of those systems were found (in that prior research pass) to implement range/TBD scheduling representation at all, which is consistent with this pass's finding that the closest real analogues are space-mission proposal tools (JWST/HST), not other community transient/follow-up coordination platforms

---
*Feature research for: Uncertain Scheduling & Site Disambiguation (FOMO v2.1)*
*Researched: 2026-07-05*
