# Phase 18: Uncertain-Scheduling Investigation Spike - Context

**Gathered:** 2026-07-08
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase is **investigation-only**, mirroring Phase 13's ESO feasibility spike. It
settles 5 open design decisions — window field schema (already locked, see below), the
replacement natural key for TBD rows, the CSV range/TBD text-parsing rules, the
fuzzy-match library choice (`rapidfuzz` vs. stdlib `difflib`), and whether `resolve_site()`
correctly resolves real space-observatory MPC codes — against the **real 3I/ATLAS
coordination sheet**, not synthetic/guessed data. The deliverable is a decision doc (or
docs); no `CampaignRun` schema migration, no CSV importer changes, no fuzzy-match UI code
ships this phase (that's Phases 19-21).

**Already locked (not open for this phase to reconsider):**
- Window schema is a nullable `window_start`/`window_end` `DateField` pair (not datetime,
  not a `django.contrib.postgres.fields.DateRangeField`) — per SCHED-01/PROJECT.md.
- `Observatory.obscode` widening is presumed unnecessary — real space-observatory MPC
  codes (250=Hubble, 274=JWST, 289=Nancy Grace Roman) are standard 3-char codes that
  already fit `CharField(max_length=4)`. The spike's job is to confirm this, default
  answer is "no widening needed," not to design a widening migration.

</domain>

<decisions>
## Implementation Decisions

### Real-data access
- **D-01:** The real 3I/ATLAS sheet CSV export lives locally at
  `/mnt/c/Users/liste/OneDrive/Documents/Asteroids/3I/3I_ATLAS Observations and Observing
  Plans - Sheet1.csv` (Tim's machine, same filesystem this session runs on). Read it
  **directly from that path** during Plan 18 execution — do **not** copy it into the repo
  or `.planning/` (real names/emails, PII-gated per project convention). Any verbatim cell
  text quoted into a committed decision doc must have `Contact Person`/`Email` redacted or
  omitted, matching Phase 13's D-04 API-response-redaction precedent. Real people's names
  used to describe a finding (e.g., "the Belyakov/Cordiner JWST rows") are acceptable
  without redaction — `PROJECT.md`'s own Current Milestone section already names these
  contributors; only email addresses and full contact-person+email pairings need
  redaction.
- **D-02:** This discussion already read the real CSV once (2026-07-08 snapshot) to
  ground the gray-area questions below. The findings captured as decisions D-03..D-09
  are real, not representative/constructed — Plan 18 execution should re-read the file
  directly (it may have been edited further, it's a live publicly-editable Google Sheet
  export) rather than trusting this document as the final word on cell shapes.

### CSV range/TBD parsing rules (enumerated from real cell shapes, SCHED-01 criterion 3)
- **D-03:** Real `Obs. Date` column shapes observed, beyond the existing exact
  `YYYY-MM-DD` case:
  - Blank entirely (obs date itself unknown, e.g. an "upcoming" VLT row with no date yet)
  - `" to "`-separated full-date range: `2025-07-05 to 2025-09-22`, `2026-01-15 to
    2026-01-22` (Carrie Holt's and Adina Feinstein's LCO-network rows)
  - Compact same-month range: `2025-11-02 -25` (meaning Nov 2–25, day2 only, no repeated
    month/year) — JUICE row
  - Month-known-day-TBD marker: `2025-12-?` (two separate real JWST rows, see D-06)
- **D-04:** **Important real finding:** a multi-day window is sometimes typed into the
  **`UT Time Range`** column instead of `Obs. Date` — e.g. `2025-11-27 to 2025-12-10`
  (John Noonan's HST row, `Obs. Date` = exact `2025-11-27`), `2025-07-11 to 2025-07-13`
  (Dennis Bodewits's Swift/UVOT row), `2025-08-11 to 2025-08-19` (Cyrielle Opitom's
  VLT/UVES row). A parser that only inspects `Obs. Date` for range syntax will silently
  miss these real multi-day windows. Any range-detection logic built in Phase 20 must
  check **both** columns, not just `Obs. Date`.
- **D-05:** Real `UT Time Range` free-text shapes beyond the already-handled HH:MM range /
  semicolon-typo / approx-hour / bare-hour-UTC cases: literal `TBD` text (several rows),
  blank, `~7:00:00 AM` (approx marker with seconds baked in), and — separately — one row
  (Matthew Belyakov, Palomar) where the cell contains a stray, unrelated fragment of the
  sheet's own preamble/description text (an apparent copy-paste artifact in the live
  sheet). This last case is genuine unparseable garbage, not a date/time shape at all —
  confirms the existing "never raise on this column, fall back and flag needs-review"
  discipline (Pitfall 1/D-09 from Phase 14) is still the right posture, extended to
  range/TBD detection.
- **D-06 (locks the TBD-collision open item from initial discussion):** Real collision
  case found: Matthew Belyakov's JWST/MIRI row and Martin Cordiner's JWST/NIRSpec row are
  both `Telescope / Instrument = "JWST"` (the instrument distinction lives only in the
  `Filter(s)/Bandpass` column, not in the telescope field), both `Obs. Date = "2025-12-?"`
  (day unknown → `window_start = window_end = NULL` in the new schema). **Decision:**
  extend the TBD-row natural key with `contact_person` (already a `CampaignRun` field,
  populated on every real row seen) so these two genuinely distinct rows don't collide
  under `(campaign, telescope_instrument, window_start)` alone, even after SCHED-04's
  partial/conditional constraint closes the plain NULL-uniqueness gap. Phase 19's
  migration design should treat this as the concrete evidence for why `contact_person`
  must be part of the key for null-window rows (exact mechanism — e.g. a
  conditional/partial `UniqueConstraint` including `contact_person` only when
  `window_start IS NULL` — is Phase 19's job to design; this phase just needs to record
  the decision and the real evidence for it in the decision doc).
- **D-07:** Some real rows have entirely blank `Site Code` for legitimately ground-based
  entries expressed at the network level rather than a specific site (`"LCO 1m"`, `"LCO
  2m"` — Carrie Holt's and Adina Feinstein's rows) as well as for space missions with no
  MPC site concept at all (Swift, JUICE). Flag this for Phase 20's ASSET-01/02 research:
  when `site` never resolves (blank or unresolvable), there's no `Observatory` to read
  `observations_type` from, so the ground-vs-space derivation has a real gap case beyond
  "resolves to a ground site" vs. "resolves to a space-mission site." Not this phase's
  decision to solve — just document it as a finding so Phase 20 doesn't discover it cold.

### Fuzzy-match library & resolve_site() confirmation (SCHED-01 criteria 4-5)
- **D-08:** Both should be **live-tested**, not reasoned from documentation alone —
  mirrors Phase 13's precedent of capturing real, verbatim evidence rather than
  theoretical comparison. `rapidfuzz` is not added to `pyproject.toml` in this phase —
  install it temporarily/scratch-only for the comparison (like Phase 13's git-excluded
  `eso_p2_probe.py`), same as a throwaway investigation script, not a committed
  dependency change.
- **D-09:** Real messy `Site Code` values to use as the fuzzy-match test corpus (from the
  actual sheet, not invented): `X09` (Sam Deen, "Deep Random Survey / 43cm"), `N50` (HCT),
  `X07` (Josep Trigo-Rodríguez, "Deep Sky Chile"), `C65` ("Telescope Joan Oró, Montsec,
  Catalonia"), and blank/missing codes (Swift, JUICE, "LCO 1m"/"LCO 2m" — see D-07, not a
  fuzzy-match case, a no-code case). **Important real finding for the `resolve_site()`
  MPC-code confirmation (criterion 5):** the real JWST rows (Belyakov's, Cordiner's, and
  Martin Cordiner's earlier Aug 2025 row) all use `Site Code = "500@-170"` — the JPL
  Horizons/SPICE observer notation — **never** the correct standard MPC code `274`. This
  is exactly the over-length code `resolve_site()` already flags for manual review rather
  than truncating/fabricating (Pitfall 2/D-09 from Phase 14) — confirms that behavior is
  correct and needed against real data. It also means confirming `resolve_site('274')`
  works is a code-path check using a constructed input (no real row in the current sheet
  snapshot actually types the plain code `274`), while `resolve_site('250')` (Hubble) can
  be confirmed directly against real rows (David Jewitt's and John Noonan's, both use
  `250`). Document both explicitly as different confidence levels in the decision doc —
  don't conflate "confirmed against a real row" with "confirmed via constructed input."
  Note this `500@-170`-vs-`274` mismatch as important context for Phase 21's fuzzy-match
  design (a straight fuzzy-string match against `Observatory.name`/`short_name`/
  `old_names` won't bridge JPL/SPICE notation to an MPC code — that's a distinct future
  problem, not something to solve in this phase).
- **D-10:** The real sheet's `Telescope / Instrument` column contains at least one
  embedded-newline quoted CSV cell (`"Hubble\nWFC3/UVIS"`, David Jewitt's row) — confirms
  the importer must keep using Python's `csv` module (which already handles this
  correctly) rather than any naive line-based parsing; not a new finding requiring code
  change, just a confirmation the existing approach is already correct.

### Claude's Discretion
- Exact wording/structure of the decision doc(s) beyond what D-01..D-10 specify.
- Whether to produce a full-detail doc plus a durable summary (Phase 13's D-10 pattern)
  or a single doc — this phase's scope is narrower than Phase 13's, a single doc may
  suffice; use judgement based on how much evidence accumulates during live-testing.
- How exactly to redact the decision doc's quoted real-sheet examples (D-01) while
  keeping them useful as verbatim evidence.
- Exact regex/parsing-rule design implementing the shapes enumerated in D-03/D-04/D-05 —
  this phase documents the shapes; designing the actual parsing rules is legitimately
  shared between this spike's decision doc (recommending an approach) and Phase 19/20's
  planning (implementing it). Use judgement on how far to take it here vs. leaving as a
  documented recommendation for Phase 20.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Real data (read directly, never commit or copy)
- `/mnt/c/Users/liste/OneDrive/Documents/Asteroids/3I/3I_ATLAS Observations and Observing
  Plans - Sheet1.csv` — the real, live, publicly-editable 3I/ATLAS coordination sheet
  export this entire phase validates against (D-01/D-02)

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` §"Scheduling Window Model (SCHED)" — SCHED-01 (this
  phase's scope) and SCHED-02..05 (Phase 19, downstream of this phase's decisions)
- `.planning/ROADMAP.md` §"Phase 18: Uncertain-Scheduling Investigation Spike" — the 5
  success criteria this phase's deliverable(s) must satisfy
- `.planning/PROJECT.md` §"Current Milestone: v2.1 Uncertain Scheduling & Site
  Disambiguation" — full milestone goal, target features, key context
- `.planning/STATE.md` §"Accumulated Context" — v2.1 roadmap decisions (4-phase
  compression rationale, obscode-widening-is-a-spike-question decision)

### Prior spike precedent (structure/format reference)
- `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-CONTEXT.md` — the only
  prior spike-type phase; established the "decision doc only, no shippable code,
  throwaway/git-excluded investigation script, redact before committing" pattern this
  phase follows
- `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-DECISION.md` — the
  decision-doc format/depth precedent (full findings + explicit recommendation)

### Existing code this phase's decisions are about
- `solsys_code/campaign_utils.py:resolve_site()` (lines 84-182) — 3-tier resolution,
  `_MAX_OBSCODE_LEN` derived from `Observatory._meta.get_field('obscode').max_length`
- `solsys_code/campaign_utils.py:parse_obs_window()` (lines 185-243) — current
  exact-date-only + best-effort UT-time parsing this phase's findings extend
- `solsys_code/models.py:CampaignRun` (lines 30-127) — current `obs_date`/`ut_start`/
  `ut_end` fields and `unique_campaign_run_natural_key` constraint this phase's decisions
  will replace (in Phase 19)
- `solsys_code/solsys_code_observatory/models.py` — `Observatory.obscode`
  (`CharField(max_length=4)`) and `observations_type`/`SATELLITE_OBSTYPE`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `resolve_site()`'s existing length/blank guard (D-09 from Phase 14, "never truncate or
  fabricate an over-length code") already does the right thing for `"500@-170"`-style
  JPL/SPICE observer strings — no change needed there, this phase just needs to confirm
  it with real data (D-09 above).
- `parse_obs_window()`'s existing "never raise on the non-key time column, fall back and
  flag needs-review" discipline is the right foundation to extend to range/TBD detection
  (D-05 above) — same pattern, wider set of shapes to recognize.

### Established Patterns
- Phase 14's per-batch deterministic disambiguating offset (for the unparseable-UT-time
  collision, `14-REVIEW.md`) is the closest precedent for "two distinct rows would
  otherwise collide on the natural key" — this phase's `contact_person`-in-the-key
  decision (D-06) is a variant of that same problem, solved with a real field instead of
  a synthetic offset since one is available on `CampaignRun`.
- Phase 13's "read-only investigation, decision doc is the sole deliverable, redact
  before committing" pattern applies directly to this phase.

### Integration Points
- None for this phase — like Phase 13, it produces no code that integrates with the
  running application. The decision doc(s) are the sole deliverable; Phase 19-21 consume
  the decisions.

</code_context>

<specifics>
## Specific Ideas

- The real sheet is genuinely messier than the synthetic `campaign_sample.csv` fixture
  anticipated in several concrete ways: a date range typed into the wrong column (D-04),
  a copy-paste garbage artifact in a data cell (D-05), and a real natural-key collision
  between two distinct people's rows (D-06) — all found on a single read of the current
  snapshot, not hypothesized.
- Tim pointed directly at the real sheet file rather than having Claude construct
  representative examples — the decision doc should lean on this real evidence rather
  than hedging with "representative, not verbatim" caveats.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (D-07's "blank site code" finding is
adjacent to this phase's SCHED-01 scope but is explicitly logged as a note for Phase 20's
ASSET-01/02 research, not solved here — captured as a decision-doc finding, not a scope
addition to this phase.)

### Reviewed Todos (not folded)
- **"Extract site/telescope mapping and instrument extraction into own module"**
  (`.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`)
  — weak keyword match (score 0.4, "site"/"phase" overlap) on the phase-matcher. Already
  reviewed and rejected as not-relevant during Phase 13's discussion (its
  `resolves_phase: 11` frontmatter shows it was resolved by Phase 11's
  `calendar_utils.py` extraction). Still not relevant to this phase's scheduling/
  fuzzy-match/obscode scope; not folded.

</deferred>

---

*Phase: 18-uncertain-scheduling-investigation-spike*
*Context gathered: 2026-07-08*
