# Phase 21: Site Disambiguation & Submitter Contact Opt-In - Context

**Gathered:** 2026-07-11
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers two structurally independent pieces, both requirements-locked
(SITE-01/02/03, VIEW-05):

1. **Staff-facing site-disambiguation UI in the approval queue** — when a submitted
   `site_raw` doesn't resolve via `resolve_site()`'s tier 1 (exact `Observatory` match) or
   tier 2 (live MPC Obscodes API), staff get a fuzzy-matched dropdown to pick from, a
   free-text/create-new fallback, and a guarantee that a site they've already manually
   resolved is never silently re-resolved/overwritten on approve (SITE-03's clobbering bug
   fix).
2. **Submitter contact opt-in** — a single combined flag (default opt-out) on the
   submission form that, when set, makes `contact_person`/`contact_email` visible to
   anonymous visitors on the per-campaign table (currently staff-only via
   `ALLOWED_FIELDS_FOR_NON_STAFF`).

No window-schema/scheduling work — that's Phases 19-20, already shipped. Fuzzy-match
*library* choice (`difflib`, not `rapidfuzz`) is already locked by Phase 18's spike
(`18-DECISION.md` Criterion 4) — not open for reconsideration here.

</domain>

<decisions>
## Implementation Decisions

### Fuzzy-match scope & trigger
- **D-01:** Candidate pool is widened to the **live MPC Obscodes list**, not just the
  local `Observatory` table (~8 rows today) that Phase 18's spike tested against. The
  spike's evidence (`18-DECISION.md` Criterion 4) explicitly flagged the local table as
  too narrow a candidate pool to meaningfully fuzzy-match arbitrary external site codes —
  this decision acts on that finding.
- **D-02:** Fetch via the existing `MPCObscodeFetcher`
  (`solsys_code/solsys_code_observatory/utils.py`), **cached locally** (DB table or
  Django cache) rather than fetched live on every approval-queue page render. Refresh
  periodically (management command or lazy-on-stale), not per-request — avoids adding
  network latency/failure risk to a page staff load often. Exact cache mechanism
  (dedicated cache table vs. Django's cache framework vs. a periodic sync command
  populating a lookup table) is Claude's discretion during planning/research.
- **D-03:** The fuzzy match still only runs after `resolve_site()`'s own tier 1 (exact
  match) and tier 2 (live single-obscode MPC lookup) have both missed — per SITE-01's
  literal wording and the spike's note. This is not a new decision, just confirmed scope.

### Staff resolution UI pattern
- **D-04:** Site column becomes an inline `<select>` of fuzzy-matched candidates plus a
  free-text fallback input, directly in the existing approval-queue table row. The
  chosen/typed value rides along with the **existing** approve/reject form POST to
  `CampaignRunDecisionView` — no new endpoint, no AJAX, no separate "resolve site" action
  decoupled from the decision.
- **D-05:** SITE-02's "create a new Observatory" (when no fuzzy-matched candidate is
  correct) reuses the **existing `CreateObservatory` form/flow**
  (`solsys_code_observatory`), not a new lightweight inline form. Link/redirect to it and
  return to the approval queue afterward — avoids duplicating obscode-driven creation
  validation that already exists.

### SITE-03 clobber-fix mechanism
- **D-06:** `CampaignRunDecisionView.post()` (lines 291-413 of `campaign_views.py`)
  currently *always* calls `resolve_site(run.site_raw, create_placeholder=False)` on
  approve, unconditionally overwriting `run.site`/`run.site_needs_review`. Fix: **skip the
  `resolve_site()` call whenever `run.site` is already set** (not `None`) at approve time
  — trust an already-resolved site (whether resolved at CSV-import time, tier 1/2
  auto-resolution, or staff's new manual-resolution UI from D-04) rather than adding a new
  `site_manually_resolved` field/migration. Only a run with `site=None` still gets
  auto-resolved on approve.

### VIEW-05 opt-in placement & scope
- **D-07:** A single checkbox on `CampaignRunSubmissionForm`
  (`solsys_code/campaign_forms.py`), placed immediately after the existing
  `contact_person`/`contact_email` fields, default **unchecked** (opt-out, matches
  today's staff-only behavior when unset). Set once at submission — there is no submitter
  self-service edit view today (confirmed), so no "editable after submission" mechanism
  is built this phase. Exact field name, verbose label, and help text are Claude's
  discretion (mirror `open_to_collaboration`'s existing style/placement precedent).
- **D-08:** When set, the new flag adds `contact_person`/`contact_email` to
  `ALLOWED_FIELDS_FOR_NON_STAFF` (or an equivalent per-row conditional) so anonymous
  visitors see them on the per-campaign table for that run only — runs that didn't opt in
  stay staff-only exactly as today.

### Claude's Discretion
- Exact MPC-list cache mechanism (D-02) — dedicated table, Django cache, or a periodic
  sync command.
- Fuzzy-match candidate count/threshold shown in the dropdown (not discussed — use
  judgement, consistent with `difflib.get_close_matches`'s `cutoff`/`n` defaults unless
  research surfaces a reason to change them).
- VIEW-05 checkbox field name, verbose label, help text (D-07).
- Whether the per-row "opted in" state needs its own visible indicator in the approval
  queue, or is purely a per-campaign-table display concern.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` §"Site Disambiguation (SITE)" — SITE-01/02/03
- `.planning/REQUIREMENTS.md` §"Visibility/Display (VIEW)" — VIEW-05
- `.planning/ROADMAP.md` §"Phase 21: Site Disambiguation & Submitter Contact Opt-In" —
  this phase's 4 success criteria
- `.planning/PROJECT.md` §"Current Milestone: v2.1 Uncertain Scheduling & Site
  Disambiguation" — full milestone goal, target features

### Phase 18 spike findings (locks the fuzzy-match library choice)
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-DECISION.md`
  Criterion 4 ("Fuzzy-match library: split verdict, difflib primary") — the actual
  library decision this phase must follow (`difflib.get_close_matches`, not
  `rapidfuzz`), plus the candidate-pool-too-narrow finding this phase's D-01 acts on,
  and the `500@-170`-vs-`274` JPL/SPICE-notation gap (neither library bridges this —
  out of scope, a distinct future problem)
- `.planning/phases/18-uncertain-scheduling-investigation-spike/18-CONTEXT.md` D-09 —
  real messy `Site Code` test corpus used during the spike, useful as a starting fixture
  set for this phase's tests

### Existing code this phase's decisions are about
- `solsys_code/campaign_views.py:CampaignRunDecisionView` (lines 291-413) — the
  approve/reject POST endpoint containing SITE-03's clobbering bug (D-06)
- `solsys_code/campaign_views.py:ApprovalQueueView` (lines 238-288) — builds the pending/
  decided tables this phase's dropdown UI extends
- `solsys_code/campaign_views.py:ALLOWED_FIELDS_FOR_NON_STAFF` (line 52) — the
  staff-only field allowlist VIEW-05's opt-in (D-08) conditionally extends
  per-row
- `solsys_code/campaign_tables.py:render_site()` (line 108) — current site-cell
  rendering (resolved short_name vs. raw text + needs-review styling) this phase's
  inline dropdown replaces for unresolved rows
- `solsys_code/campaign_utils.py:resolve_site()` (lines 84-182) — the 3-tier resolver;
  D-03 confirms the fuzzy match only runs after both existing tiers miss
- `solsys_code/campaign_forms.py:CampaignRunSubmissionForm` (lines 17-61) — submission
  form VIEW-05's opt-in checkbox (D-07) is added to
- `solsys_code/solsys_code_observatory/` — existing `CreateObservatory` form/view (D-05)
  and `MPCObscodeFetcher` (D-02) this phase reuses rather than reimplementing
- `.planning/quick/260705-l1v` (referenced by SITE-01/02, not yet inspected directly) —
  the prior fix that stopped auto-fabrication of placeholder `Observatory` rows on
  approval; this phase's UI is the "natural next step" after it

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `MPCObscodeFetcher` (`solsys_code_observatory/utils.py`) — already fetches/parses the
  MPC Observatory Codes API; D-02 reuses it rather than writing a new fetcher.
- Existing `CreateObservatory` form/view — D-05 reuses it for SITE-02's "create new"
  path instead of a bespoke inline form.
- `open_to_collaboration` field on `CampaignRunSubmissionForm`/`CampaignRun` — an
  existing boolean-opt-in-on-a-public-form precedent VIEW-05's new flag (D-07) should
  mirror stylistically.
- `resolve_site()`'s existing 3-tier structure and `create_placeholder` kwarg — D-06's
  fix works within this existing function's contract, no signature change needed beyond
  the call-site skip logic in `CampaignRunDecisionView`.

### Established Patterns
- `render_site()`'s dict-vs-model dual-accessor pattern (handles both `.values()`
  querysets and model instances) — the inline dropdown (D-04) must preserve this dual
  compatibility since `CampaignRunTableView`'s non-staff path uses `.values()`.
- Phase 14's D-09 "never truncate or fabricate an over-length/unresolvable code, flag
  for manual review instead" discipline — SITE-02's explicit-create-or-resolve
  requirement is a direct continuation of this discipline for the approval-queue path
  specifically (the CSV-import path already got this in Phase 14; the public-submission
  approval path got a partial fix via quick task `260705-l1v`; this phase completes it
  with an actual resolution UI).

### Integration Points
- New/changed UI lives inside the existing `approval_queue.html` template and
  `ApprovalQueueTable`/`CampaignRunTable` (`campaign_tables.py`) — not a new page.
- `CampaignRunDecisionView.post()` gains the D-06 skip-if-already-resolved check and
  must read the new site-selection field(s) from the same POST body the dropown/free-text
  submits (D-04).
- `CampaignRunSubmissionForm`/`CampaignRunSubmissionView` gain the D-07 opt-in field;
  `CampaignRunTableView`'s `ALLOWED_FIELDS_FOR_NON_STAFF`-gated queryset (or its
  per-row rendering) gains the D-08 conditional contact-field exposure.

</code_context>

<specifics>
## Specific Ideas

- No specific visual/UX references given beyond the four locked decisions above — user
  went with the recommended option on 3 of 4 areas discussed, and explicitly chose to
  widen the fuzzy-match candidate pool to the live MPC list (D-01) rather than staying
  local-table-only, prioritizing match quality over avoiding the caching complexity.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

### Reviewed Todos (not folded)
- **"Extract site/telescope mapping and instrument extraction into own module"**
  (`.planning/todos/pending/2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md`)
  — weak keyword match (score 0.6, "site"/"telescope"/"instrument" overlap) on the
  phase-matcher. Already reviewed and rejected as not-relevant during Phase 13 and Phase
  18's discussions (its `resolves_phase: 11` frontmatter shows it was resolved by Phase
  11's `calendar_utils.py` extraction — an LCO/Gemini calendar-sync concern, not
  campaign-approval-queue site resolution). Still not relevant; not folded.
- **"Rename calendar_utils.py private helpers to reflect shared-module API"**
  (`.planning/todos/pending/2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md`)
  — weak keyword match (score 0.6, "calendar"/"api"/"sync" overlap). About
  `calendar_utils.py`'s LCO/Gemini-sync-facing helper naming, unrelated to this phase's
  `campaign_utils.py`/approval-queue/submission-form scope. Not folded.

</deferred>

---

*Phase: 21-site-disambiguation-submitter-contact-opt-in*
*Context gathered: 2026-07-11*
