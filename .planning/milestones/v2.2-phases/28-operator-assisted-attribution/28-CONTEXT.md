# Phase 28: Operator-Assisted Attribution - Context

**Gathered:** 2026-07-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 28 connects the calendar events and observation records that already exist to the
`CampaignRun`s that own them — through a **staff worklist of suggested associations**, each
showing its evidence, confirmed by a human and reversible. It writes exactly two things:
`CalendarEventMeta.run` (the event side) and `CampaignRunObservation` rows (the record side),
both of which Phase 27 already built. It never guesses silently and never merges automatically.

**In scope:** the candidate matcher and its confidence scoring; a staff-only attribution page
with the two orphan worklists, their evidence columns, and a score-band filter; per-candidate
and high-band multi-select confirmation; dismissal of a wrong suggestion with a reason; undo of
a confirmed association; the count banner that makes the backlog visible from the campaign list;
audit fields on the event side; and the operator runbook section for the new surface.

**Out of scope:** the reconciler and any calendar-event projection (Phase 29 — attribution ships
first so RECON stages 3-4 have real data and so ATTRIB-06 is structural); rewiring the four
ingest adapters to create runs (v2.3, ADAPT-01..03); automatic merging at any confidence
(Out of Scope table); any new dependency — `difflib` is the fuzzy-match tool, `rapidfuzz` was
rejected in v2.1 and again for this milestone.

</domain>

<decisions>
## Implementation Decisions

### The queue's shape and where staff reach it (ATTRIB-01)

- **D-01: Orphan-centric, not run-centric.** One row per un-attributed calendar event or
  observation record, expanded to its candidate runs — not one row per run expanded to its
  candidate events. The backlog *is* the set of unowned orphans, so an orphan-centric list
  drains to empty and gives ATTRIB-06 ("attribution can be completed before the first reconcile
  sweep") a measurable meaning. A run-centric list over ~31 runs, most with no candidates at
  all, never visibly finishes.

  Note this reverses the framing of ROADMAP criterion 5, which reads run-first ("pk=1 is offered
  against its 11 LCO queue events"). The criterion is about *which pairs are surfaced*, not about
  which column the table is keyed on — it is satisfied as long as each of the 11 events offers
  pk=1 as a candidate. Do not "fix" the criterion's wording or the queue's orientation to match
  each other.

- **D-02: A standalone staff page plus a count banner on the campaign list.** Its own route
  behind `StaffRequiredMixin`, because `approval_queue.html` already carries three tables
  (pending / decided / sites needing review) and this is a long filterable worklist rather than
  ongoing per-submission review. The banner is driven the same way 27.1-03's two-queue banner
  already is — reuse that mechanism rather than inventing a second counting path.

- **D-03: An orphan enters the queue only if it has at least one candidate run.** The
  campaign/target boundary check from criterion 3 doubles as the noise filter: a conference or
  proposal-deadline `CalendarEvent` has no target, so it produces no candidate and never appears.
  There is deliberately **no separate exclusion list** of event kinds to maintain — one rule,
  used twice. Consequence to state plainly in the UI: the queue shows attributable orphans, not
  every un-attributed row.

- **D-04: Two tables on one page, not one merged table and not two pages.** "Calendar events
  awaiting attribution" and "Observation records awaiting attribution" as sibling tables — the
  same shape `approval_queue.html` already uses. Their evidence differs (an event has a title and
  a window; a record has a facility, a status and parameters), so a merged table would carry
  mostly-blank columns, which is exactly the defect 16-05 had to trim off `ApprovalQueueTable`.
  Both write to different link models, so the confirm action differs anyway.

### Dismissals — the open question D-01 of Phase 27 handed to this phase (ATTRIB-03)

- **D-05: A rejected suggestion is persisted, per (orphan, run) pair.** Phase 27's D-01 wrote
  nothing until confirmation; the cost it accepted was that a rejected candidate returns on every
  page load. Without a dismissal record the queue cannot drain, so ATTRIB-06 would be unreachable
  for any orphan carrying a wrong candidate. **A dismissal is not an association** — persisting
  one does not weaken Phase 27's D-01, and an unconfirmed guess still can never be mistaken for
  ownership.

  Dismissal is per *pair*, not per orphan: dismissing the whole orphan would also hide candidates
  the matcher surfaces later as new runs arrive.

- **D-06: A dismissal records who, when, and a free-text reason.** `confirmed_by`/`confirmed_at`
  on `CampaignRunObservation` is the existing in-house shape; mirror it and add an optional note.
  The reason is what stops the next person re-deriving why pk=1 was rejected against a given
  event, and it is the evidence any future change to the matcher's weights would be judged
  against.

- **D-07: A dismissal is reversible from a collapsed "Dismissed" section on the same page.**
  Same shape as the approval queue's Decided table. Symmetric with criterion 4's undo
  requirement — a mis-click must never need a database shell, and a dismissal must not end up
  the one irreversible action on a page whose whole point is reversibility.

- **D-08: Two small typed dismissal models, one per orphan kind.** `GenericForeignKey` is in the
  milestone's Out of Scope table, and the two alternatives are both worse: one model with two
  nullable FKs needs an "exactly one is set" `CheckConstraint` and branch-on-column reads, and a
  synthetic `EVENT:{pk}` string key throws away referential integrity — a deleted event would
  leave a dangling dismissal, which is the precise failure `GenericForeignKey` was rejected to
  avoid. Two models means real FKs, plain JOINs, a named `UniqueConstraint` on each pair, and
  some duplicated field definitions, which is the accepted cost.

### Confidence, filtering, and how confirmation happens (ATTRIB-02, ATTRIB-03)

- **D-09: Multi-select confirmation, gated to the high-confidence band.** Checkboxes and one
  submit — but checkboxes render **only** on candidates above the high band; anything ambiguous
  is single-confirm only.

  **This resolves a real conflict in the source documents, and the resolution needs recording
  rather than reconciling by wording.** REQUIREMENTS ATTRIB-02 says staff can "bulk-confirm the
  confident tail"; ROADMAP criteria 3 and 4 say "explicit per-candidate staff confirmation" and
  "one candidate at a time". The project owner chose multi-select, on the reading that ticking an
  individual box *is* a per-candidate act. The score gate is the guardrail that keeps that
  reading honest: bulk speed applies exactly where ATTRIB-02 wanted it and nowhere else.

  The accepted residual risk, stated so nobody has to rediscover it: a high band that is tuned
  too loose turns select-and-submit into silent guessing wearing a confirmation button. The band
  cut-point is therefore a correctness decision, not a display preference. The owner declined the
  three other offered guardrails (no select-all control; checkbox only when an orphan has exactly
  one candidate; evidence forced inline on every checkboxable row) — planning may still adopt any
  of them, but must not treat the absence of one as an oversight.

- **D-10: Score is filtered by named band and displayed as a number too.** High / Medium / Low
  derived from the numeric score, with the band as the filter control and the underlying number
  rendered in the row so staff can sanity-check the banding while the matcher is new. The band is
  also what gates D-09's checkboxes. ROADMAP criterion 1 is explicit that staff see the evidence
  "not a bare score" — the number is *additional* to the evidence columns, never a replacement.

- **D-11: A pure weighted sum over the evidence signals, with the campaign/target boundary as the
  single hard gate.** Telescope match, date overlap and instrument-string similarity all
  contribute to one number and **none of them is disqualifying**. The campaign/target boundary
  alone pre-filters which pairs get scored at all — because ROADMAP criterion 3 forbids a
  cross-boundary suggestion absolutely, and a weight can always be re-tuned over a threshold
  whereas a filter cannot. This costs nothing extra: D-03's eligibility rule is the same check.

  **This is what makes criterion 5 work.** `CampaignRun` pk=1 (FTS/MuSCAT4, 7-21 July, Siding
  Spring E10) must be offered against its 11 LCO queue events (`2m0`/`2M0-SCICAM-MUSCAT`, 7-20
  July) *despite* mismatched instrument strings and a one-day span difference. Under a pure sum
  those mismatches cost score but cannot disqualify. A design where instrument similarity or exact
  date-span equality gates would fail criterion 5 on day one — do not add such a gate.

### Undo and the event-side audit gap (ATTRIB-04)

- **D-12: `CalendarEventMeta` gains `confirmed_by`/`confirmed_at`, revisiting Phase 27's D-05.**
  ROADMAP criterion 4 requires both the confirmation and the undo to be attributable to a person
  and a time. `CampaignRunObservation` already satisfies that; `CalendarEventMeta.run` was made a
  bare FK on purpose, and 27-CONTEXT states the consequence outright — "Phase 28's undo on the
  event side will be untraceable" — while listing the revisit as a deferred idea for *this* phase
  ("if Phase 28's undo flow proves the event-side gap painful, that is the phase to revisit it").
  It is painful; revisit it.

  This is a **deliberate, evidence-backed reopening of a locked Phase 27 decision**, not drift.
  Accepted cost: the companion record now mixes telescope-label metadata with attribution audit,
  which is a slightly wider job than its `CalendarEventMeta` name implied — though 26-DECISION
  chose that general name precisely so a third concern could be added without a second rename.
  Phase 27's D-05 comment in `models.py` says "do not fix it here"; the planner must **update that
  comment** rather than leave it contradicting the code.

- **D-13: An undo writes a dismissal row.** Undoing means clearing `run` or deleting the link row,
  which erases the very fields recording who confirmed it — so the trace has to live elsewhere.
  The dismissal model from D-05/D-06 already carries who, when and a reason, and "this pair was
  rejected by X at T" is exactly what an undo means. It also stops the matcher immediately
  re-suggesting the pair just undone, which is otherwise the obvious next bug.

  Rejected alternatives, with reasons: a soft-undo flag (`undone_by`/`undone_at` on the link row)
  would break Phase 27 D-01's invariant that a row's existence means "confirmed", forcing every
  reader of "is this attributed?" to carry an extra condition; a log line puts the trace where no
  staff member will see it, which does not satisfy "attributable".

- **D-14: Confirmed associations appear in a "Confirmed" section on the same page.** A third
  collapsed section beside the two worklists and the Dismissed one — same shape as the approval
  queue's Decided table. Satisfies criterion 4's "undone from the same screen that created it"
  literally, and doubles as the record of what the attribution pass actually did, which is what
  someone reviewing it before the Phase 29 sweep needs. Deliberately not inline state-flipping in
  the worklist: that would keep the queue from visibly shrinking, undercutting D-15.

- **D-15: "Done" is an empty queue plus a stated remaining count.** Both worklists drain to zero,
  the page states how many orphans remain unattributed, and the D-02 banner reads zero. ATTRIB-06
  and criterion 5 both require the pass to be *completable* before the first reconcile sweep —
  this makes that a checkable fact rather than a judgement, and gives Phase 29 an unambiguous
  precondition to point at. A backlog-reporting management command was offered and **declined**;
  if Phase 29 wants to assert the precondition programmatically, that is Phase 29's call.

### Claude's Discretion

- The names of the two dismissal models, the attribution view/URL/template, and the matcher
  module. Note the milestone's locked constraint that new reconciliation logic lives in
  `solsys_code/campaign_reconciler.py` — planning should decide whether the matcher belongs
  there, in a peer `campaign_attribution.py`, or in `campaign_utils.py`, and must not bury it as
  a private helper inside `campaign_views.py`.
- The actual weights in D-11's sum and the High/Medium/Low cut-points — subject to D-09's warning
  that the high cut-point is a correctness decision because it gates multi-select, and to the
  requirement that criterion 5's real pk=1 case lands high enough to be surfaced.
- How date overlap is scored for a TBD or unresolved window (`window_start`/`window_end` are
  nullable).
- Pagination, sort order, and whether the Dismissed and Confirmed sections paginate separately.
- Test organisation.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decisions this phase executes

- `.planning/ROADMAP.md` §Phase 28 — the five success criteria and the "Depends on" rationale for
  why attribution ships before the reconciler. Criterion 3's campaign/target boundary is absolute
  (see D-11); criterion 4's undo-attributability is what forces D-12.
- `.planning/REQUIREMENTS.md` — ATTRIB-01..06, and the Out of Scope table, which rejects automatic
  merging, `GenericForeignKey` for either link, making the `run` link required, and any new
  dependency. **ATTRIB-02's "bulk-confirm" wording conflicts with ROADMAP criteria 3/4; D-09 is
  the resolution — do not re-litigate it, and consider correcting ATTRIB-02's wording to match.**
- `.planning/phases/27-the-canonical-run-record/27-CONTEXT.md` — D-01 (nothing written until
  confirmation, and the explicit hand-off of the dismissal question to this phase), D-02/D-03/D-04
  (the observation link's shape and audit fields), **D-05 (the event-side audit gap this phase
  reopens — read alongside D-12 above)**, D-06 (the staff run-detail page deferred to this phase),
  and the Deferred Ideas list, which names this revisit.
- `.planning/phases/26-canonical-record-spike/26-DECISION.md` §"Criterion 4 / SPIKE-04" — the
  attribution strategy: ownership lives on the companion record's `run` link, no automatic
  merging, per-candidate confirmation only, attribution completable before the first sweep. Also
  the ownership rule Phase 29 reads: a companion row whose `run` is unset means "not mine".
- `docs/design/canonical_record_spike.rst` — the durable, redaction-free form of the same
  decisions.

### Code this phase changes or depends on

- `solsys_code/models.py` — `CalendarEventMeta` (lines 9-58; its `run` FK and the D-05 comment
  D-12 supersedes), `CampaignRunObservation` (lines 329-390; the confirmation target and its
  `unique_campaign_run_observation_record` constraint), `CampaignRun` (`Source`,
  `TelescopeClass`, `is_publicly_visible`, both partial unique constraints).
- `solsys_code/campaign_views.py` — `ALLOWED_FIELDS_FOR_NON_STAFF` (line 70), `ApprovalQueueView`
  and `runs_needing_site_review()` (the two-queue pattern D-02's banner reuses),
  `CampaignRunDecisionView` (the existing staff POST-action shape, including its atomic
  conditional `.update()` idiom).
- `solsys_code/campaign_tables.py` — `CampaignRunTable`/`ApprovalQueueTable`, including the
  `Meta.exclude`/`Meta.sequence` column trimming from 16-05 that D-04 cites, and
  `render_actions()`'s single-form refactor.
- `solsys_code/campaign_urls.py` — the `campaigns` namespace; today list / submit /
  submission-thanks / approval-queue / site-search / decide / gaps / table. No run-detail route
  exists.
- `solsys_code/campaign_utils.py` — `fuzzy_match_candidates()` (line 549) and
  `substring_or_fuzzy_match_candidates()` (line 580), the existing `difflib` scoring precedent
  D-11's instrument-similarity signal should follow rather than reinvent.
- `solsys_code/calendar_utils.py` — `derive_telescope_class()`, `SITE_TELESCOPE_MAP`,
  `aperture_class_from_telescope_code`, `extract_instrument` — the telescope/instrument
  vocabulary D-11's telescope and instrument signals must speak.
- `solsys_code/mixins.py` — `StaffRequiredMixin`, the existing staff gate for D-02's page.
- `solsys_code/admin.py` — `CalendarEventMetaInline` / `CampaignRunObservationInline` and the
  `save_formset` attribution stamping from Phase 27 D-07. **D-12's new audit fields must be
  stamped by that same `save_formset` path**, or admin-created event links reproduce exactly the
  audit hole D-07 was written to close.
- `src/templates/campaigns/approval_queue.html` — the multi-table page D-04 and D-07/D-14 mirror.

### Paired docs (CLAUDE.md rule — required in `files_modified` up front)

- `docs/runbooks/telescope_runs_calendar.rst` — documents the staff approval-queue actions; this
  phase adds a new staff decision surface alongside them, and the runbook must describe the
  attribution pass, the meaning of a dismissal, and the D-15 "done" signal Phase 29 depends on.
  In `files_modified` from the start, not as a follow-up.
- No new demo notebook: this phase ships no management command (D-15 declined one), so nothing in
  CLAUDE.md's notebook pairing map is touched. If planning reintroduces a command, the pairing
  rule applies and the map gains an entry.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`fuzzy_match_candidates()` / `substring_or_fuzzy_match_candidates()`**
  (`campaign_utils.py:549,580`) — `difflib`-based scoring already in production for site
  disambiguation, with a cached candidate pool and a per-IP throttle. D-11's instrument-similarity
  signal should reuse this approach; the milestone forbids adding `rapidfuzz`.
- **`runs_needing_site_review()`** (`campaign_views.py`) — one shared definition feeding both a
  queue and a campaign-list banner, added by 27.1-03. This is the exact pattern D-02's count
  banner should follow rather than a second counting path.
- **`CampaignRunObservation`** — already exists with `confirmed_by`/`confirmed_at` and a named
  `UniqueConstraint`. This phase writes rows into it; it does not design it.
- **`CalendarEventMeta.run`** — already exists, nullable, `SET_NULL`. This phase sets it; D-12
  only adds the audit fields beside it.
- **`StaffRequiredMixin`** (`mixins.py`) — the staff gate, already used by the approval queue.
- **`ApprovalQueueTable`'s `render_actions()`** — the single-form-per-row POST idiom; D-09's
  multi-select needs a different shape (one form spanning many rows), so this is a reference for
  conventions, not a template to copy.

### Established Patterns

- **Named `UniqueConstraint`s with explanatory comments** — `CampaignRun.Meta.constraints` and
  `CampaignRunObservation.Meta.constraints`. D-08's two dismissal models follow this.
- **Hand-enumerated non-staff allow-list** — `ALLOWED_FIELDS_FOR_NON_STAFF` is deliberately not
  introspected, so a new field is invisible to non-staff unless explicitly added. Nothing in this
  phase is public, but any new field must be a deliberate omission, commented.
- **Queryset-level gating, not template conditionals** — the non-staff run table excludes pending
  rows in SQL. The attribution page is staff-only end to end, so this is a floor not a ceiling.
- **Atomic conditional `.update()` for staff decisions** — `CampaignRunDecisionView.post()` keys
  its update on the current state so a double-submit is a proven no-op. D-09's multi-select
  confirm must survive the same double-submit test, and a race on two staff confirming the same
  orphan must land on the `UniqueConstraint`, not on a lost update.
- **Multi-table staff pages** — `approval_queue.html`'s three tables; D-04/D-07/D-14 give the
  attribution page four sections (two worklists, Dismissed, Confirmed).

### Integration Points

- `campaign_urls.py` gains the attribution route (and the confirm / dismiss / undo POST targets).
- `campaign_list.html` gains the D-02 count banner, driven off the shared definition.
- `admin.py`'s `save_formset` must stamp D-12's new `CalendarEventMeta` audit fields.
- `models.py` gains two dismissal models and two fields on `CalendarEventMeta` — so this phase
  carries migrations, which the planner should sequence before the UI work.
- `docs/runbooks/telescope_runs_calendar.rst` gains the attribution section.

</code_context>

<specifics>
## Specific Ideas

- **Criterion 5 is the acceptance test, and it is a real pair of rows.** `CampaignRun` pk=1 —
  FTS/MuSCAT4, 7-21 July, Siding Spring E10 — against its 11 LCO queue events
  (`2m0`/`2M0-SCICAM-MUSCAT`, 7-20 July). The instrument strings differ and the span differs by a
  day. If the matcher does not surface this pair, the phase has not shipped, regardless of what
  the tests say.
- **These are not duplicates to be merged.** pk=1 is the run; the 11 events are how it was
  realised. The fix is attribution, not deduplication — this distinction is the reason the whole
  milestone exists, and it is why automatic merging is out of scope at any confidence.
- **The measured backlog is small** — 11 LCO queue events with a companion row whose `run` is
  unset. Design for legibility over throughput; the bulk path exists for the confident tail, not
  because the volume demands it.
- **Beware stale pks in planning docs.** 26-DECISION records that the dev DB was re-imported;
  PROJECT.md's Phase 25 paragraph does not reproduce against it. Verify any concrete pk against
  the live DB before building a fixture or a test around it.

</specifics>

<deferred>
## Deferred Ideas

- **A backlog-reporting management command** — offered under D-15 and declined. If Phase 29 wants
  to assert the attribution precondition programmatically rather than by looking at the page,
  that is Phase 29's call.
- **A full staff run-detail page** — Phase 27's D-06 deferred it here, but D-01's orphan-centric
  queue does not need one, so it is not built. Revisit if a run-first view of its own links is
  wanted; the admin inlines cover it for now.
- **The other three multi-select guardrails** — no select-all control, checkbox only when an
  orphan has exactly one candidate, and evidence forced inline on every checkboxable row. Offered
  under D-09 and not chosen; available if the score gate alone proves too loose in use.
- **Bulk *dismissal*** — not discussed and not in scope; dismissal is per-pair and individual.
- **v2.3 items untouched here:** adapter rewiring (ADAPT-01..03), provenance-blind gap analysis
  (GAPB-01), status vocabulary unification (STATUS-01/02), unused-allocation display (UNUSED-01).

### Reviewed Todos (not folded)

- **`2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`** — extract
  `SITE_TELESCOPE_MAP` and instrument extraction into their own module (matched at score 0.6 on
  telescope/instrument keywords). Not folded: it has no ATTRIB requirement behind it, Phase 27's
  D-24 already dropped it back out for exactly that reason and left it open for a later cleanup
  pass, and this phase carries two new models, a field addition, migrations and a new staff page.
  This phase *reads* `calendar_utils`' telescope/instrument vocabulary for D-11's signals; it does
  not need to move it.

</deferred>

---

*Phase: 28-operator-assisted-attribution*
*Context gathered: 2026-07-31*
