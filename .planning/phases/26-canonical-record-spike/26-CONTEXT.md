# Phase 26: Canonical-Record Spike - Context

**Gathered:** 2026-07-27
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase is **investigation-only**, following the Phase 13 (ESO) and Phase 18
(uncertain-scheduling) precedents. It settles the four questions milestone questioning
deliberately left open — the `source` vocabulary and its interaction with `CampaignRun`'s
two existing partial unique constraints (SPIKE-01), how each ingest adapter's existing
calendar-event identity key maps onto a run (SPIKE-02), the canonical reconciler event-key
scheme and the stage-2 class-wide fan-out question (SPIKE-03), and the migration and
attribution strategy including the companion-record rename checklist (SPIKE-04).

The deliverables are a decision doc plus a durable `docs/design/` page. **No schema
migration, no reconciler, no attribution UI ships from this phase** — Phases 27, 28 and 29
consume the decisions. Any migration or model code written during the spike is throwaway,
git-excluded, and discarded when the phase closes.

**Already locked — this phase executes these, it does not re-open them** (ROADMAP.md
"Locked constraints", REQUIREMENTS.md "Out of Scope"):

- The spike blocks everything; Phases 27-29 are gated on it.
- `related_name='telescope_label_meta'` is **not** renamed. Only the model class is.
- No new dependencies (`django-dirtyfields`/`FieldTracker`, `django-fsm`, Celery,
  `rapidfuzz`, `GenericForeignKey` all explicitly rejected by research).
- New reconciler logic will live in `solsys_code/campaign_reconciler.py`, a peer of
  `campaign_gap.py`/`campaign_utils.py` — never a private helper in `campaign_views.py`,
  and never importing `solsys_code.views` or `solsys_code.ephem_utils` (the latter triggers
  a ~1.6 GB SPICE kernel download at module load).
- The `run` FK is `null=True, blank=True, on_delete=SET_NULL`; no `RunPython` backfill.
- `source` stays out of both existing partial unique constraints — attribution, not the
  constraint, connects same-physical-run rows from different sources.
- No automatic merging of suspected duplicate associations; no upstreaming the event→run
  link into `tom_calendar`.

</domain>

<decisions>
## Implementation Decisions

### Evidence standard — how the spike proves things

- **D-01:** The evidence vehicle is a **throwaway migration applied to a copy of the real
  dev DB**. Write the real `source`/`telescope_class` migration plus the companion-record
  rename on a scratch branch, apply it to a copy of `src/fomo_db.sqlite3`, record verbatim
  results in the decision doc, then discard. This is the only way to satisfy SPIKE-01's
  "demonstrated by ... coexisting without an `IntegrityError`" — the field does not exist
  yet, so it must actually be created. Follows Phase 13's git-excluded `eso_p2_probe.py`
  precedent: throwaway investigation code, decision doc is the deliverable. Phase 27 writes
  the real migration afterwards.

- **D-02:** The spike **executes and measures the companion-record rename**, it does not
  just enumerate it. On the throwaway branch: perform the rename, migrate the DB copy, run
  the full `./manage.py test solsys_code` suite, and load `/calendar/` in the dev server —
  then record which of the four integration points actually broke. This converts research
  Pitfall #1 (the top-ranked risk) from a prose checklist into measured evidence, and
  Phase 27 executes a de-risked rename.

  **Analytical prediction to test, not assume:** because `related_name='telescope_label_meta'`
  is locked unchanged, the calendar template lookups and the view's `prefetch_related()`
  string are safe *by construction* — only the two class-name imports (`solsys_code/admin.py`
  and `solsys_code/management/commands/sync_lco_observation_calendar.py`) are actually at
  risk, and those fail loudly as `ImportError`. The spike should confirm or refute this
  rather than restating the research's four-way checklist unexamined.

- **D-03:** The renamed model class is **`CalendarEventMeta`**. Chosen for generality — it
  absorbs `run`, `is_verified`, and whatever v2.3 adds, without needing a second rename.
  Reads naturally against the locked accessor (`event.telescope_label_meta` → a
  `CalendarEventMeta`), with the understanding that the accessor name itself remains
  legacy-flavoured until some future phase changes it. `CalendarEventRunLink` was
  considered and rejected: it misdescribes the 11 existing rows, which are pure
  telescope-label metadata with no run at all.

- **D-04:** Take a **dated, git-excluded snapshot** of `src/fomo_db.sqlite3` and pin every
  number in the decision doc to it (e.g. "as of 2026-07-27: 31 runs, 20 calendar events,
  0 `CAMPAIGN:` events"). The dev DB is a moving target — it was re-imported after Phase 25
  — so undated counts will rot. Record the PROJECT.md discrepancy (D-16) as an explicit
  finding so Phases 27-29 do not trust it, and open a **separate todo** to correct
  PROJECT.md. The spike itself stays investigation-only and does not edit PROJECT.md.

### Stage-2 class-wide semantics

- **D-05:** A class-wide run produces a **single class-wide event** per day (00:00–23:59,
  labelled with the class, no site) — **not** one event per candidate site. Grounded in
  measured cost: `CampaignRun` pk=29 (`LCO 1m`) is an 80-night window and `SITE_TELESCOPE_MAP`
  carries `1m0` at five sites (coj, elp, lsc, cpt, tfn), so fan-out would be 80 × 5 = 400
  events for one run, four-fifths of them describing observations that will never happen
  there — LCO's scheduler picks exactly one site. Stage 3 narrows to the real site when an
  `ObservationRecord` appears, which is the pipeline working as designed.

- **D-06:** CANON-02's field **widens from a telescope-class-only field to a "why is there
  no site" vocabulary** covering all three real meanings: telescope-class allocation
  (`1m0`/`2m0`/`0m4`), space mission, and unresolved/failed-to-resolve. The requirement as
  written names only two, but the live data has **five** space-mission rows (pk=8 Hubble,
  pk=12 HST, pk=13 Swift, pk=21 JWST, pk=26 JUICE) against **two** class-wide ones (pk=29
  `LCO 1m`, pk=30 `LCO 2m`). v2.1 distinguished ground from space via
  `Observatory.observations_type`/`SATELLITE_OBSTYPE`, which needs a resolved `Observatory`
  these rows do not have — this closes the gap Phase 18's D-07 explicitly deferred. The
  spike settles the vocabulary; Phase 27 implements it.

- **D-07:** A space-mission run gets **one spanning event covering the whole window**, not
  one event per day. pk=26 (JUICE, 2025-11-02→11-25) becomes one 24-day event rather than
  24 daily events — honestly representing "sometime in this window" instead of asserting 24
  observing days. This keeps the calendar consistent with v2.1's asset-aware
  `campaign_gap.claimed_dates()`, which refuses to claim those dates at all; the uniform
  "treat every site-less run as daily 00:00–23:59" alternative was rejected precisely
  because the calendar and the gap analysis would then disagree about the same run.

- **D-08:** The spike defines an explicit **stage 0 — "allocated but unscheduled"**. Runs
  with `window_start IS NULL` produce no calendar event (there is no date to place), but
  the reconciler **counts and reports them in its summary**, the way `import_campaign_csv`
  already reports `site_needs_review` — visibly pending rather than silently skipped. Three
  real rows need this: pk=4 (ESO VLT FORS2, site-resolved, approved) and pk=27/28 (JWST,
  no site). Gives RECON-06's "reported and skipped" a defined case and stops each
  downstream phase inventing its own answer.

### Canonical event key & ownership

- **D-09:** **Identity and ownership are two separate mechanisms.** Identity: the reconciler
  passes a namespaced `url` of the form `RUN:{run_pk}:{date}` as the
  `insert_or_create_calendar_event()` lookup, giving stage-stable idempotency. Ownership
  (RECON-05): the new companion `run` FK, as a hard rule — **no companion row, or a
  companion row with `run=NULL`, means "not mine, never touch"**. That rule is already
  provable against the live DB: the 9 classical events have no companion row at all, and
  all 11 LCO events have companion rows that will be `run=NULL` until attribution sets them.

  `insert_or_create_calendar_event()` takes a caller-supplied `lookup` dict
  (`solsys_code/calendar_utils.py:317`), so the reconciler is free to choose its own key
  without touching the shared helper.

- **D-10:** The `{date}` component is **always the site-local observing night**, derived in
  the site's timezone — never the UTC date of whatever `start_time` the current stage
  happens to produce. Stages 3 and 4 therefore change an event's *times* but never its
  *key*. This is the direct mitigation for research Pitfall #5 (idempotency breaking across
  stage transitions): at Siding Spring (UTC+10) the night of 7 July begins ~09:00 UTC on
  7 July but can run into 8 July UTC, so a UTC-date key would mint a fresh event and orphan
  the stage-1 one on every re-run. Matches how the codebase already thinks —
  `load_telescope_runs` writes `start_time=sunset(d)`, `end_time=sunrise(d+1)` for night `d`.

- **D-11:** The **adopt-vs-gap-fill question is prototyped, not decided in the abstract.**
  Against the real pk=1 case (15-night window 7–21 July; 11 of those nights already carry
  LCO events), build both on the throwaway DB copy and recommend one based on what the
  calendar actually looks like:
  - *Adopt* — update the attributed event in place, keeping its LCO-URL key, and mint
    `RUN:1:{date}` only for the 4 uncovered nights. 15 events. Makes the reconciler
    authoritative over the whole pipeline; biggest blast radius, since it writes to events
    the LCO sync command also writes to.
  - *Gap-fill* — create nothing for nights with an attributed adapter event; fill only the
    4 uncovered nights. Also 15 events, respects RECON-05 more strictly, but stages 3-4 for
    the 11 LCO nights keep coming from the sync command until v2.3 rewires the adapters.

  The decision doc must also record the **rejected baseline** — reconciler always mints its
  own, giving 26 events for one run — and state explicitly that this is the visible
  double-booking ATTRIB-06 exists to prevent.

### `source` vocabulary & approval gating

- **D-12:** The 31 existing runs get a distinct **`LEGACY`** value. Nothing in the data can
  discriminate provenance — `original_obs_date_raw` is set on only 2 rows (pk=27/28), so it
  is a parse-failure marker, not an import signature — meaning any per-row inference would
  be guesswork. `LEGACY` is honest about what is actually known (these rows predate
  provenance tracking), is never produced by any new code path, and stops a future reader
  mistaking an assumed provenance for a recorded one. Blanket `CSV_IMPORT` was rejected as
  an unverifiable assertion written into 31 rows, and doubly uncertain for pk=1, which is on
  the Didymos 2026 campaign rather than 3I/ATLAS.

- **D-13:** Phase 27 declares the **full vocabulary — all five roadmap values plus
  `LEGACY`** — with the three adapter values (classical file, LCO queue, Gemini queue)
  explicitly documented as *not yet produced by any code path*, awaiting v2.3's
  ADAPT-01..03. Downstream code (approval gating, attribution evidence) is then written once
  against the final vocabulary instead of being widened later. The cost is near zero:
  Django `TextChoices` values are validation-only, so adding or removing them later is a
  no-op `AlterField`.

- **D-14:** Non-web runs keep `approval_status = APPROVED`; **`source` is the
  disambiguator.** The derivation rule — `APPROVED` **and** `source != WEB` means *no
  approval was required*, as distinct from *a human approved this* — is recorded explicitly
  in the decision doc so downstream code does not re-invent it. A fourth `NOT_REQUIRED`
  approval value was considered and rejected: every existing reader of `approval_status`
  (approval-queue filters, the non-staff visibility gate, `CampaignRunTable`,
  `ApprovalQueueTable`, `CampaignRunDecisionView`'s conditional `.update()`) would have to
  handle it, for a distinction `source` already carries.

### Measured findings that correct the planning docs

These were verified against `src/fomo_db.sqlite3` during this discussion. Downstream agents
should trust these over the corresponding planning-doc statements.

- **D-15:** **There are zero `CAMPAIGN:`-namespaced calendar events in the dev DB.** The
  20 events are 9 classical (blank `url`) and 11 LCO (`https://observe.lco.global/...`).
  The reconciler's key scheme therefore has a clean slate — no existing `CAMPAIGN:{pk}` or
  `CAMPAIGN:{pk}:{date}` rows to migrate or reconcile with.

- **D-16:** **PROJECT.md's Phase 25 claim does not reproduce.** It states "`CampaignRun`
  pk=34 (GS-2026A-FT-115) ... now has its 4 per-night `CalendarEvent`s", but the maximum
  run pk is 31, there is no FT-115 row, and there are no `CAMPAIGN:` events at all. The DB
  was re-imported after Phase 25's UAT. Record as a finding; correct PROJECT.md via a
  separate todo (D-04).

- **D-17:** **There are zero `PENDING_REVIEW` runs** (30 approved, 1 rejected). CANON-01's
  approval-gating change therefore disturbs no live pending rows. Relatedly,
  `import_campaign_csv.py:194` **already** writes `ApprovalStatus.APPROVED` ("D-03:
  bootstrap rows are vetted backfill"), so the roadmap's claim that Phase 27 changes what
  the importer writes for `approval_status` is only true if a new value is introduced —
  which D-14 declines to do. The importer's real behaviour change is writing `source`.

- **D-18:** **There are zero `GEM:`-namespaced events**, so SPIKE-02's Gemini identity
  mapping can only be reasoned from code, not confirmed against real rows — state that
  confidence difference explicitly, following Phase 18's D-09 precedent of never conflating
  "confirmed against a real row" with "confirmed via constructed input".

- **D-19:** **The 9 classical events have a blank `url`.** SPIKE-02's per-adapter identity
  mapping must account for the classical adapter having no string identity key at all — it
  is keyed on `(telescope, instrument, start_time ± tolerance)` via
  `insert_or_create_calendar_event`'s `start_time_tolerance` path. RECON-05's ownership
  scoping consequently cannot lean on `url` for these events, which is precisely why D-09
  puts ownership on the companion FK instead.

- **D-20:** Confirmed exactly as the roadmap claims: `CampaignRun` pk=1 (FTS/MuSCAT4,
  2026-07-07→2026-07-21, site 7, campaign 2 "Didymos 2026", approved/observed); its 11 LCO
  queue events (ids 53-63, `2m0`/`COJ-2m0` + `2M0-SCICAM-MUSCAT`, 7–20 July, 8 `[EXPIRED]`,
  1 `[CANCELLED]`); 11 companion rows, all `is_verified=1`; and **exactly 19** approved,
  site-resolved, windowed 3I/ATLAS runs with zero calendar presence. Also present: 13 LCO
  `ObservationRecord`s (4 COMPLETED, 8 WINDOW_EXPIRED, 1 CANCELED) — the only real data
  stages 3 and 4 have to act on.

### Claude's Discretion

- Exact structure, wording and section ordering of the decision doc and the `docs/design/`
  page beyond what D-01..D-20 specify, and whether they are one document or two (Phase 13
  used full-detail-plus-durable-summary; Phase 18 folded both into one).
- How to redact real 3I/ATLAS `contact_person`/`contact_email` values in any quoted
  evidence — carry forward Phase 18's D-01 posture: real people's names may be used to
  describe a finding, but email addresses and full name+email pairings must be omitted or
  redacted. This was not re-asked; it is an established project convention.
- Mechanics of the throwaway branch and DB copy (branch name, where the copy lives, how it
  is git-excluded) — Phase 13's `eso_p2_probe.py` is the precedent, not a prescription.
- How deep to take attribution-scoring prototyping against pk=1's 11 events beyond what
  D-11 requires for the adopt-vs-gap-fill comparison.

### Folded Todos

- **"Rename `calendar_utils.py` private helpers to reflect shared-module API"**
  (`.planning/todos/pending/2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`)
  — folded as a **recommendation to record, not code to write** (this phase ships no code).
  The todo notes that `_derive_telescope`, `_extract_instrument`, `_resolve_placement_block`,
  `_coarse_telescope_label` and `_aperture_class_from_telescope_code` are imported across
  module boundaries while still carrying a leading underscore, and that
  `calendar_utils.py`-owned tests still live in `test_sync_lco_observation_calendar.py`.
  It is adjacent to this phase because the spike is already deciding a rename (D-03) and
  already touching `calendar_utils.py`'s create-or-update contract (D-09). The decision doc
  should state a recommended naming posture so Phase 27 — which will be editing these
  modules anyway — can execute it as part of its own work rather than as a separate cleanup.
  **Note:** a recent commit explicitly corrected a false claim that v2.2 closes this todo;
  folding it here as a recommendation does not close it, and the todo stays open until code
  actually lands.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` §"Phase 26: Canonical-Record Spike" — the 5 success criteria this
  phase's deliverables must satisfy; §"Locked constraints" — the six constraints this phase
  executes rather than re-opens
- `.planning/REQUIREMENTS.md` §"Spike" — SPIKE-01..04; §"Out of Scope" — the six explicitly
  excluded approaches (auto-merge, upstreaming, `related_name` rename, new dependencies,
  `GenericForeignKey`, required `run` link)
- `.planning/PROJECT.md` §"Current Milestone: v2.2 One Canonical Run Record" — milestone
  goal, the four-stage window pipeline table, key context. **Read with D-16 in hand** — its
  Phase 25 paragraph contains a claim that no longer reproduces.
- `.planning/STATE.md` — current position

### Milestone research (direct source inspection, HIGH confidence)
- `.planning/research/SUMMARY.md` — executive summary, top-5 pitfalls, per-phase
  implications. Note its suggested 6-phase structure was compressed to 4 in ROADMAP.md;
  the roadmap wins.
- `.planning/research/PITFALLS.md` — measured dev-DB hazards; Pitfalls #1 (rename breaks
  silent integration points) and #5 (idempotency across stage transitions) are what D-02
  and D-10 respectively mitigate
- `.planning/research/ARCHITECTURE.md` — the `campaign_gap.py`/`campaign_utils.py`
  pure-logic-module pattern `campaign_reconciler.py` must follow
- `.planning/research/STACK.md` — why no new dependency is warranted

### Prior spike precedent (structure, format, discipline)
- `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-CONTEXT.md`
  — the closest precedent; D-01 (real-data access + PII redaction posture) and D-07 (the
  site-less/space-mission gap this phase's D-06 closes) carry forward directly
- `.planning/milestones/v2.1-phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`
  — decision-doc format and depth; note its explicit separation of "confirmed against a real
  row" from "confirmed via constructed input" (D-18 reuses this)
- `.planning/milestones/v1.7-phases/13-eso-feasibility-spike/13-CONTEXT.md` — established
  the throwaway/git-excluded investigation-script pattern D-01 follows
- `docs/design/uncertain_scheduling_spike.rst` and `docs/design/eso_feasibility_spike.rst` —
  the two existing durable `docs/design/` spike pages; ROADMAP criterion 5 requires this
  phase to produce a third

### Existing code the decisions are about
- `solsys_code/models.py:7-27` — `CalendarEventTelescopeLabel`, the `OneToOneField(primary_key=True)`
  sidecar being renamed to `CalendarEventMeta` (D-03) and gaining the `run` FK
- `solsys_code/models.py:120-160` — `CampaignRun.Meta`, the two partial `UniqueConstraint`s
  (`unique_campaign_run_resolved_window`, `unique_campaign_run_tbd_natural_key`) and the
  `campaign_run_window_start_end_null_together` `CheckConstraint` that SPIKE-01 must leave
  unchanged
- `solsys_code/calendar_utils.py:317-377` — `insert_or_create_calendar_event()`, the
  caller-supplied-`lookup` create-or-update contract D-09's key scheme plugs into, including
  the `start_time_tolerance` proximity path the classical adapter uses (D-19)
- `solsys_code/calendar_utils.py:36-53` — `SITE_TELESCOPE_MAP`, the source of the 5-site
  `1m0` / 2-site `2m0` counts behind D-05
- `solsys_code/campaign_views.py` — `_project_calendar_event()`, `_calendar_event_title()`,
  `_set_run_status()`, `CampaignRunDecisionView`; the three current callers of calendar
  projection the reconciler replaces
- `solsys_code/management/commands/backfill_range_calendar_events.py` — retired by RECON-09;
  the existing anti-pattern of importing a private helper from the views module
- `solsys_code/management/commands/import_campaign_csv.py:194` — already writes
  `ApprovalStatus.APPROVED` (D-17)
- `solsys_code/admin.py` and
  `solsys_code/management/commands/sync_lco_observation_calendar.py` — the two class-name
  import sites the rename actually puts at risk (D-02)
- `solsys_code/campaign_gap.py` — asset-aware `claimed_dates()`, whose ground-vs-space
  treatment D-07 keeps the calendar consistent with

### Real data (read directly; snapshot, never commit)
- `src/fomo_db.sqlite3` — the live dev DB all D-15..D-20 findings were measured against.
  Snapshot and date-pin it per D-04. Contains real `contact_person`/`contact_email` values
  on 3I/ATLAS rows — redact per Phase 18's D-01 before quoting anything into a committed doc.

### Folded todo
- `.planning/todos/pending/2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`
  — folded as a recommendation to record (see Folded Todos above)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `insert_or_create_calendar_event()` (`calendar_utils.py:317`) already implements the
  no-churn create-or-update contract RECON-01 needs, and takes an arbitrary caller-supplied
  `lookup` dict — so the reconciler adopts D-09's key scheme without modifying the shared
  helper at all. Its `start_time_tolerance` proximity-window path (built for
  `load_telescope_runs`' IERS-drifting sun-event times) is existing precedent for keying on
  something other than an exact string.
- `_update_or_unchanged()` in the same module already does explicit field diffing — this is
  why research rejected `django-dirtyfields`/`FieldTracker`.
- `campaign_gap.py`'s asset-aware `claimed_dates()` already encodes the ground-vs-space
  distinction D-07 keeps the reconciler consistent with.
- `import_campaign_csv`'s `site_needs_review` summary counter is the existing pattern D-08's
  stage-0 reporting should mirror.
- `test_calendar_template.py` (17 tests, incl. dashed-border and `CaptureQueriesContext`
  N+1 assertions) is the existing safety net that would catch template/prefetch regressions
  from the rename — relevant to D-02's measurement.

### Established Patterns
- **Spike phases produce a decision doc plus a durable `docs/design/` page, and nothing
  else** — Phase 13 and Phase 18 both did exactly this; ROADMAP criterion 5 requires it here.
- **Throwaway investigation code is git-excluded and never committed** (Phase 13's
  `eso_p2_probe.py`) — D-01's scratch migration follows this.
- **Never conflate confirmation levels** — Phase 18's D-09 explicitly separated "confirmed
  against a real row" from "confirmed via constructed input". D-18 reapplies this to the
  Gemini mapping.
- **Pure-logic modules are imported by views, never the reverse** (`campaign_gap.py`,
  `campaign_utils.py`) — `campaign_reconciler.py` joins this family.
- **PII discipline** — real contact names/emails are never copied into `.planning/` or
  committed docs (Phase 18 D-01, Phase 14/15 PII gating).

### Integration Points
- **None for this phase.** Like Phases 13 and 18, it produces no code that integrates with
  the running application. Every integration point named in D-02 is *examined* on a
  throwaway branch and then discarded; Phases 27-29 consume the decisions.

</code_context>

<specifics>
## Specific Ideas

- The discussion was deliberately grounded in read-only queries against the real dev DB
  rather than in the planning docs, and that turned up four corrections (D-15, D-16, D-17,
  D-18) plus one requirement gap (D-06, the three-way meaning of `site=None`). The decision
  doc should lean on this measured evidence the way Phase 18's did on the real 3I sheet —
  stating counts and pks, not hedged generalities.
- The recurring instinct across three separate decisions was **refusing to conflate distinct
  meanings in one value**: `site=None` meaning three different things (D-06), `APPROVED`
  meaning both "human approved" and "no approval needed" (D-14), and an assumed provenance
  being indistinguishable from a recorded one (D-12). Where the fix was cheap the spike
  splits the meanings; where it was expensive (D-14) it records an explicit derivation rule
  instead. Downstream phases should apply the same test.
- D-11 was deliberately left to measurement rather than decided in the abstract — the user
  chose "prototype both and recommend after" over picking adopt or gap-fill up front.

</specifics>

<deferred>
## Deferred Ideas

- **Correcting PROJECT.md's stale Phase 25 paragraph** (D-16) — a separate todo, not this
  phase's work. The spike records the discrepancy as a finding; the doc fix happens outside
  the investigation-only boundary.
- **Renaming `related_name='telescope_label_meta'`** to match the new `CalendarEventMeta`
  class — explicitly out of scope per REQUIREMENTS.md; revisit only if a future phase has a
  reason to accept the silent-breakage risk.
- **`CalendarEvent.url` non-uniqueness** (noted in `23-REVIEW.md` as a pre-existing
  structural issue) — D-09 relies on the `url` key for identity while putting ownership on
  the companion FK, which sidesteps it. If a later phase wants `get_or_create` on `url` to
  be race-safe, that is its own change.
- **v2.3 items confirmed still deferred and untouched by this discussion:** status
  vocabulary unification (STATUS-01/02), adapter rewiring (ADAPT-01..03), provenance-blind
  gap analysis (GAPB-01), unused-allocation visualisation (UNUSED-01).

### Reviewed Todos (not folded)
- **"Extract site/telescope mapping and instrument extraction into own module"**
  (`.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`)
  — matched on weak keyword overlap (score 0.6: "site", "telescope", "instrument",
  "module"). Its own frontmatter carries `resolves_phase: 11`, and Phase 11's
  `calendar_utils.py` extraction already did the work; it was reviewed and rejected as
  not-relevant during both Phase 14's and Phase 18's discussions. Folding it would mean
  re-opening a closed extraction. Not folded.

</deferred>

---

*Phase: 26-canonical-record-spike*
*Context gathered: 2026-07-27*
