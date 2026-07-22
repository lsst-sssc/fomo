---
title: v2.1 Phase 18 (Uncertain-Scheduling Investigation Spike) — implementation hints
date: 2026-07-05
context: Captured at the end of a /gsd-new-milestone session for v2.1 (Uncertain Scheduling & Site Disambiguation), before hitting the weekly usage limit, so a future session can resume Phase 18 planning/execution without re-deriving this context.
---

# Note: v2.1 Phase 18 implementation hints

## Correction that overrides the research files (critical — read this first)

`research/SUMMARY.md`, `STACK.md`, `ARCHITECTURE.md`, `PITFALLS.md`, and `FEATURES.md` all repeat a
false premise before a correction note was added on top of them: the `'500@-170'` string in
`resolve_site()`'s docstring/comments (`campaign_utils.py` lines ~95, ~117) is JPL Horizons/SPICE
observer notation (NAIF SPK ID for JWST), **not an MPC obscode**. Real space telescopes have
standard 3-character MPC codes: **250 = Hubble, 274 = JWST, 289 = Nancy Grace Roman** (per
https://www.minorplanetcenter.net/iau/lists/ObsCodes.html). `Observatory.obscode`'s
`max_length=4` almost certainly does **not** need widening. Full correction is already recorded
at the top of `.planning/research/SUMMARY.md` and in `PROJECT.md`'s v2.1 Key Context — check
there for the canonical version before trusting the rest of the research files' framing.

## Phase 18 spike must settle these open questions before Phase 19 starts

- **Window field schema**: `window_start`/`window_end` as a nullable `DateField` pair (both null
  = TBD state); single classical night modeled as `window_start == window_end`. `ut_start`/`ut_end`
  should likely stay as separate optional precise-time fields feeding only the `CalendarEvent`
  projection (CAL-01/02), decoupled from "what dates are claimed" (Architecture research
  recommendation).
- **Replacement natural key** for `CampaignRun` once `ut_start`/`window_start` can be null: current
  `UniqueConstraint` is `(campaign, telescope_instrument, ut_start)`. SQLite and PostgreSQL both
  treat NULL as never-equal in a `UniqueConstraint`, so a naive swap would silently reopen the exact
  duplicate-row race Phase 14's WR-05 fix closed. Needs a partial/conditional constraint, e.g.
  `UniqueConstraint(fields=[...], condition=Q(window_start__isnull=False))` for scheduled rows,
  plus a separate explicit dedup mechanism for TBD rows (a content hash was suggested, not decided).
- **CSV range/TBD parsing rules** for `parse_obs_window` (`campaign_utils.py`) must be enumerated
  from real 3I/ATLAS sheet cell shapes (examples given: "Aug 1-15", "TBD pending Cycle 2"), not
  guessed generically. Mirror the existing `site_needs_review` pattern: unparseable-after-best-effort
  rows get flagged "needs review" and counted in the import summary, never silently dropped
  (IMPORT-02).
- **Fuzzy-match library choice**: Stack research recommends adding `rapidfuzz>=3.9` (MIT, zero
  deps, better matching quality, not currently a direct project dependency — only present
  transitively via poetry/cleo). Architecture research recommends stdlib
  `difflib.SequenceMatcher`/`get_close_matches` instead to avoid a new dependency, given the small
  scale (a few hundred `Observatory` rows). This is an explicit open disagreement — the spike
  should resolve it via match-quality testing against real messy site-name input from the actual
  sheet.
- **Confirm `resolve_site()`'s tier 1** (exact `Observatory` match) **and tier 2** (live MPC
  Obscodes API query) **actually resolve real space-observatory codes** (250/274/289) correctly —
  untested either way as of this note.

## Real bugs/gaps already found — fix as part of the relevant phase, don't rediscover later

- **`CampaignRunDecisionView.post()`** (`campaign_views.py` ~line 296) unconditionally re-calls
  `resolve_site()` on every approve. Once Phase 21 lets staff manually resolve a site via the new
  disambiguation UI, this must be guarded (e.g. `if run.site_id is None: resolve_site(...)`) or it
  will silently clobber the human's manual choice. Must ship atomically with Phase 21's SITE-01/02,
  not as a separate cleanup task.
- **`CreateObservatoryForm`** (`solsys_code/solsys_code_observatory/forms.py`) independently
  hardcodes `max_length=3, min_length=3` on its obscode field, separate from the `Observatory`
  model's own `max_length=4`. If the spike ever does decide obscode needs widening (unlikely per
  the correction above), this form must be updated in parallel or Phase 21's SITE-02 free-text
  "create a new Observatory" fallback will keep rejecting valid codes even after a model migration.
- **A fuzzy-match layer must never auto-select the top candidate** above some confidence threshold
  — always present candidates for human choice via the dropdown. This preserves the "never
  fabricate, always flag" invariant quick task `260705-l1v` (approval-queue site-visibility fix,
  shipped 2026-07-05) established for `resolve_site()`'s `create_placeholder=False` path.
  Auto-selecting would reintroduce the same bug class one level up, worse: there'd be no
  `site_needs_review`-style flag to catch a wrong-but-real auto-selected `Observatory`.

## Build order / dependency spine

Phase 18 (spike) → Phase 19 (window-schema migration, largest blast radius, must land before any
consumer touches the new schema) → Phase 20 (CSV range/TBD import + asset-aware coverage-gap, both
consumers of Phase 19's schema, can run concurrently within the phase). Phase 21 (site
disambiguation UI + VIEW-05 contact opt-in) is structurally independent of the scheduling work and
only depends on Phase 18's fuzzy-library decision — can be planned/executed in parallel with 19-20.

## Reference locations for anyone resuming

- `.planning/PROJECT.md`'s "Current Milestone: v2.1" section — full target-feature list + key context
- `.planning/REQUIREMENTS.md` — 13 requirements: SCHED-01..05, ASSET-01..02, IMPORT-01..02, SITE-01..03, VIEW-05
- `.planning/ROADMAP.md` Phase 18-21 detail sections — goal/success-criteria per phase
- `.planning/research/SUMMARY.md` — full synthesis, correction note at top
- `.planning/research/{STACK,FEATURES,ARCHITECTURE,PITFALLS}.md` — per-dimension detail

Next command once resumed: `/gsd-discuss-phase 18` or `/gsd-plan-phase 18`.
