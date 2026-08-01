# Phase 28: Operator-Assisted Attribution - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-31
**Phase:** 28-operator-assisted-attribution
**Areas discussed:** Queue shape & entry point, Where dismissals live, Confirm granularity & score filter, Undo & the event-side audit gap

---

## Queue shape & entry point

**Q1: Which way round should the attribution queue be organised?**

| Option | Description | Selected |
|--------|-------------|----------|
| Orphan-centric (Recommended) | One row per un-attributed event/record, expanded to candidate runs; drains to empty, makes ATTRIB-06 measurable | ✓ |
| Run-centric | One row per `CampaignRun`, expanded to candidate events/records; reads with criterion 5 but ~31 runs never drain | |
| Both directions | Orphan worklist plus a per-run panel; two places a confirmation can happen | |

**Q2: Where do staff reach the attribution queue?**

| Option | Description | Selected |
|--------|-------------|----------|
| New standalone staff page (Recommended) | Own route behind `StaffRequiredMixin`; keeps a long worklist off the already-dense approval queue | |
| Fourth table on the approval queue | One destination for all staff work; cheapest, but that page carries three tables already | |
| Both — page plus a count banner | Standalone page plus a count banner on the campaign list, driven like 27.1-03's two-queue banner | ✓ |

**Q3: What makes an orphan eligible to appear at all?**

| Option | Description | Selected |
|--------|-------------|----------|
| Only those with ≥1 candidate run (Recommended) | Criterion 3's boundary check doubles as the noise filter; no separate exclusion list | ✓ |
| Everything un-attributed, candidates may be empty | Honest about the full backlog; fills with conferences and deadlines that never drain | |
| Only rows sharing a campaign target | Pre-filter then match; a second rule that can drift from the boundary check | |

**Q4: One merged table or two?**

| Option | Description | Selected |
|--------|-------------|----------|
| Two tables on one page (Recommended) | Matches `approval_queue.html`'s shape; avoids the mostly-blank columns 16-05 had to trim | ✓ |
| One merged table with a type column | One sort order and one thing to drain; evidence columns only half-apply per row | |
| Two separate pages | Cleanest separation; splits the "is attribution done?" answer across two places | |

**Notes:** The orphan-centric choice reverses ROADMAP criterion 5's run-first framing. Recorded in CONTEXT.md D-01 as a framing difference, not a conflict — the criterion is about which pairs get surfaced.

---

## Where dismissals live

**Q1: What happens to a suggestion a staff member rejects?**

| Option | Description | Selected |
|--------|-------------|----------|
| Persist the dismissal (Recommended) | A record of the rejected (orphan, run) pair; without it the queue cannot drain and ATTRIB-06 is unreachable | ✓ |
| Nothing — recompute every time | Zero new models; a wrong candidate is immortal and "done" becomes a judgement call | |
| Dismiss the whole orphan | Fewer rows; also hides candidates the matcher would surface later as new runs arrive | |

**Q2: What should the dismissal record carry?**

| Option | Description | Selected |
|--------|-------------|----------|
| Who, when, and a free-text reason (Recommended) | Mirrors `CampaignRunObservation`'s audit fields; the reason is the evidence a future matcher change is judged against | ✓ |
| Who and when only | Exact parity, no free text; a dismissal becomes an unexplained veto | |
| Just the pair | Minimal; the one staff action in the campaign area not attributable to a person | |

**Q3: Is a dismissal reversible, and where?**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, from a Dismissed list on the same page (Recommended) | Same shape as the approval queue's Decided table; symmetric with criterion 4's undo | ✓ |
| Admin only | No extra UI; recovery is a raw row delete for admin-access staff only | |
| No — dismissals are permanent | Simplest rule; makes the cheapest action the only irreversible one | |

**Q4: How is the dismissal modelled, given `GenericForeignKey` is rejected?**

| Option | Description | Selected |
|--------|-------------|----------|
| Two small typed models (Recommended) | Real FKs, plain JOINs, a named `UniqueConstraint` each; some duplicated field definitions | ✓ |
| One model, two nullable FKs | Half the models; needs an "exactly one is set" `CheckConstraint` and branch-on-column reads | |
| One model keyed by a synthetic string | Fewest tables; throws away referential integrity — the failure `GenericForeignKey` was rejected to avoid | |

**Notes:** Phase 27's D-01 explicitly handed the dismissal question to this phase. A dismissal is not an association, so persisting one does not weaken D-01.

---

## Confirm granularity & score filter

**Q1: ATTRIB-02's "bulk-confirm" vs ROADMAP criterion 3's "explicit per-candidate confirmation"?**

| Option | Description | Selected |
|--------|-------------|----------|
| One at a time, filter for speed (Recommended) | Keeps ATTRIB-03 structural; matches criteria 3 and 4; requires correcting ATTRIB-02's wording | |
| Checkbox multi-select, one submit | Each association still individually ticked; the ergonomic that can turn into select-all | ✓ |
| One at a time now, bulk deferred | Honest about not knowing whether the tail is long enough to need it | |

**Q2: What keeps multi-select from degenerating into select-all-and-click?** *(multi-select question)*

| Option | Description | Selected |
|--------|-------------|----------|
| No select-all control | Every checkbox ticked individually; cheapest guardrail | |
| Only above a score threshold | Checkboxes render only on the high band; bulk speed exactly where ATTRIB-02 wanted it | ✓ |
| Only one candidate per orphan | Checkbox only when there is nothing to choose between | |
| Evidence visible in the row | Nothing confirmable without its evidence on screen | |

**Q3: How is confidence surfaced and filtered?**

| Option | Description | Selected |
|--------|-------------|----------|
| Named bands, filter by band (Recommended) | A band is a boundary a person can reason about; avoids false precision about a `difflib` ratio | |
| Numeric score with a threshold slider | Maximum control; exposes a knob whose right value nobody knows yet | |
| Both — band as filter, number shown | Filter by band, display the number so staff can sanity-check the banding while the matcher is new | ✓ |

**Q4: How do the signals combine, given criterion 5's real case?**

| Option | Description | Selected |
|--------|-------------|----------|
| Hard gates plus a weighted score (Recommended) | Boundary and telescope gate; date and instrument only move the score | |
| Pure weighted sum | Simplest to tune; nothing disqualifying | ✓ |
| All signals are gates, no score | Very legible; fails criterion 5 immediately | |

**Q5 (follow-up, raised by Claude): criterion 3 forbids cross-boundary suggestions absolutely — how does a pure sum handle that?**

| Option | Description | Selected |
|--------|-------------|----------|
| Boundary is the one hard gate (Recommended) | Everything else stays a pure weighted sum; the boundary pre-filters which pairs get scored; costs nothing extra | ✓ |
| Truly pure sum, boundary as a heavy weight | Uniform model; makes criterion 3 a tuning value a future weight change could breach | |
| Weighted sum plus telescope gate too | Fewer junk candidates; risks hiding a pair whose telescope string is recorded oddly | |

**Notes:** Q1's answer contradicts ROADMAP criteria 3/4 as literally worded. Flagged to the user at the time; the owner's reading is that ticking an individual box *is* a per-candidate act, with the score gate as the guardrail. Recorded in CONTEXT.md D-09 including the residual risk and the three declined guardrails. Q5 was asked because the Q4 answer, taken literally, would have breached a fixed ROADMAP criterion.

---

## Undo & the event-side audit gap

**Q1: Criterion 4 needs the event-side confirmation and undo attributable, but Phase 27 D-05 gave `CalendarEventMeta.run` no audit fields.**

| Option | Description | Selected |
|--------|-------------|----------|
| Add audit fields to `CalendarEventMeta` (Recommended) | Revisits D-05 exactly as 27-CONTEXT's deferred-ideas list anticipated; symmetric with the observation link | ✓ |
| A separate attribution audit log | Preserves D-05; full history; a third new model and a second place to look | |
| Accept the asymmetry | Costs nothing; leaves a fixed ROADMAP criterion knowingly unmet | |

**Q2: Where does the undo's own trace live, since undoing erases the confirmation fields?**

| Option | Description | Selected |
|--------|-------------|----------|
| Undo writes a dismissal row (Recommended) | Reuses the who/when/reason record; also stops the matcher re-suggesting the pair just undone | ✓ |
| Soft-undo: keep the row, mark it undone | Full history in one place; breaks D-01's "row exists means confirmed" invariant | |
| Undo is logged, not stored | Cheapest; the trace lives where no staff member will see it | |

**Q3: Where do confirmed associations appear for undo?**

| Option | Description | Selected |
|--------|-------------|----------|
| A Confirmed section on the same page (Recommended) | Satisfies criterion 4 literally; doubles as the record of what the pass did | ✓ |
| Inline — the row flips state | Strongest "same screen" reading; the queue never visibly shrinks | |
| From the run's page and the admin | Least new UI; contradicts "the same screen that created it" | |

**Q4: How is "done" made visible for ATTRIB-06?**

| Option | Description | Selected |
|--------|-------------|----------|
| Empty queue plus a stated count (Recommended) | "Done" becomes a checkable fact; Phase 29 gets an unambiguous precondition | ✓ |
| A management command that reports the backlog | Scriptable; a second surface to build and keep honest | |
| Both | Belt and braces; one more command in this phase's scope | |

**Notes:** Q1 is a deliberate, evidence-backed reopening of a locked Phase 27 decision — 27-CONTEXT names this phase as the place to revisit it. The planner must update the "do not fix it here" comment in `models.py` rather than leave it contradicting the code.

---

## Claude's Discretion

- Names for the two dismissal models, the attribution view/URL/template, and the matcher module — subject to the milestone's constraint that new logic is not a private helper inside `campaign_views.py`.
- The weights in the sum and the High/Medium/Low cut-points, subject to the high cut-point gating multi-select and to criterion 5's pk=1 case landing high enough to surface.
- Date-overlap scoring for a TBD or unresolved window.
- Pagination, sort order, and whether Dismissed/Confirmed paginate separately.
- Test organisation.

## Deferred Ideas

- A backlog-reporting management command (declined under D-15; Phase 29's call if it wants a programmatic precondition).
- A full staff run-detail page (Phase 27 D-06 deferred it here; the orphan-centric queue does not need one).
- The three declined multi-select guardrails — no select-all control, checkbox only for single-candidate orphans, evidence forced inline.
- Bulk dismissal — not discussed, not in scope.
- v2.3 items: ADAPT-01..03, GAPB-01, STATUS-01/02, UNUSED-01.
- Reviewed but not folded: `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — no ATTRIB requirement, already dropped from Phase 27 by D-24 for the same reason.
