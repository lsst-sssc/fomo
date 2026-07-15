---
status: resolved
trigger: "Investigate issue: sites-needing-review-placement-and-placeholder-correction — The \"Sites Needing Review\" table is positioned confusingly on the approval-queue page, and a row whose site is a placeholder Observatory has no way to correct it."
created: 2026-07-15T19:15:00Z
updated: 2026-07-15T20:30:00Z
resolution: "Placement (2A) fixed by 22-05-PLAN.md: Sites Needing Review wrapped in a distinct border-warning action-required card while preserving D-07's locked table order (commit 936f565). Placeholder correction (2B) fixed by 22-06-PLAN.md: NEEDS_REVIEW_NAME_PREFIX + is_placeholder_observatory() helper, render_site() correction widget for placeholder sites, and _resolve_site() placeholder-replacement path (commits ef97bd2, 03bb0e9, 7bd649e). A follow-up deep code review then found and fixed 2 further Critical gaps this fix exposed one layer down (resolve_site() misreporting a placeholder hit as genuine, and placeholders polluting the search candidate pool) — see 22-REVIEW.md / 22-REVIEW-FIX.md."

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: CONFIRMED (two distinct root causes, both in Phase 22's own design/decision layer, not accidental code defects).
  (A) Placement: 22-CONTEXT.md D-07 explicitly locked the table order as "pending / decided /
  sites-needing-review", so approval_queue.html correctly implements the design as specified --
  the actionable "Sites Needing Review" table was placed AFTER the historical/read-only "Recently
  Decided" table by design decision, not by coding accident, and that design decision was never
  visually validated by a human before UAT.
  (B) Placeholder correction: ApprovalQueueTable.render_site()'s resolve-mode branch treats
  "site__short_name is truthy" (ANY Observatory set on run.site) as proof of "genuinely resolved,
  only needs projection retry" -- it never checks whether that Observatory is itself a placeholder
  (name starts with 'NEEDS REVIEW: ', created via resolve_site(create_placeholder=True)). No model
  field/helper exists anywhere in the codebase to distinguish a placeholder Observatory from a
  genuine one, and CampaignRunDecisionView.post()'s resolve_site action explicitly skips
  resolution whenever run.site is not None (D-06 never-re-resolve guard) -- a guard designed to
  prevent a race between two concurrent staff POSTs, but which also has the side effect of
  permanently blocking manual correction of an already-set (even if wrong/placeholder) site.
test: n/a -- root cause confirmed via direct code + planning-doc read, cross-referenced against
  the orchestrator's Django-shell finding that Observatory(obscode='DCT') has
  name='NEEDS REVIEW: DCT' and blank timezone, matching campaign_utils.py's placeholder-creation
  naming convention exactly.
expecting: n/a
next_action: n/a -- diagnose-only mode (goal: find_root_cause_only), returning ROOT CAUSE FOUND.

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: The "Sites Needing Review" table is clearly grouped/visible under the "Approval Queue" page heading; a row can be corrected via the live-search widget if its already-set site turns out to be a placeholder or wrong value.
actual: User reports (with a screenshot) that the "Sites Needing Review" table renders at the bottom of the approval-queue page, after the approved/decided runs section, not visually grouped under the "Approval Queue" banner -- described as "unhelpful" for finding it. Separately, in the screenshot, one row (telescope "DCT", Site column showing plain text "DCT") has ONLY a "Resolve" button -- no search input -- because its `site` field is already set. Investigation via Django shell confirmed the local `Observatory` row with `obscode='DCT'` has `name='NEEDS REVIEW: DCT'` and a blank `timezone` -- i.e. it is itself a placeholder record (very likely created via an earlier `resolve_site(create_placeholder=True)` call from the Phase 21 approve flow, not a genuine MPC-resolved site), yet the current "Sites Needing Review" UI treats "site already set" as equivalent to "correctly resolved" and offers no way to search/replace it -- only to retry the (already-succeeded, in this case) calendar projection.
errors: None -- this is a UX/workflow design gap, not a crash or exception.
reproduction: UAT Test 2 in .planning/phases/22-site-matching-at-submission-and-unmatched-site-resolution-wo/22-UAT.md. Screenshot saved at /home/tlister/.claude/image-cache/945378df-779f-43c5-8119-fb584930b1b2/1.png (not readable by this agent -- described above). Relevant template: src/templates/campaigns/approval_queue.html (table ordering/headings). Relevant view/table logic: solsys_code/campaign_views.py ApprovalQueueView.get_context_data() (review_table construction) and solsys_code/campaign_tables.py ApprovalQueueTable.render_site() (the `site_short_name` early-return path for resolve-mode rows whose site is already set, per 22-03-PLAN.md's explicit by-design choice -- investigate whether that design choice itself is the "root cause", vs. a distinct bug of not distinguishing a placeholder Observatory from a genuinely-resolved one).
started: Discovered during Phase 22 UAT (2026-07-15), after Phase 22's automated verification (VERIFICATION.md) passed 20/20 must-haves and a deep code review + fixes were applied.

## Eliminated
<!-- APPEND only - prevents re-investigating -->

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-07-15T19:20:00Z
  checked: src/templates/campaigns/approval_queue.html (full file, 17 lines)
  found: |
    Page structure is: <h4>Approval Queue</h4>, then sequentially <h5>Pending Review</h5> +
    pending_table, <h5>Recently Decided</h5> + decided_table, <h5>Sites Needing Review</h5> +
    review_table. All three h5 sections are DOM siblings under the same h4/content block -- there
    is no separate page, no distinct card/border grouping distinguishing "Sites Needing Review"
    from "Recently Decided" visually.
  implication: The template is a literal, faithful implementation of the spec below (not an
    implementation bug) -- "Sites Needing Review" is placed last, after the historical/read-only
    "Recently Decided" table, with no visual differentiation (card/border/color) to signal it's an
    actionable work-queue like "Pending Review" rather than more historical content.

- timestamp: 2026-07-15T19:21:00Z
  checked: .planning/phases/22-.../22-CONTEXT.md decision D-07
  found: "D-07: \"Sites needing review\" is a third table on the existing approval-queue page
    (pending / decided / sites-needing-review), listing approved runs with
    site_needs_review=True. ... No new page or navbar entry."
  implication: The table ORDER (pending, then decided, then sites-needing-review) was explicitly
    locked during Phase 22's discuss-phase, not left to implementation discretion. This is a
    genuine design decision, made in the abstract (text-only CONTEXT.md), never checked against
    how it would actually read once rendered with real historical decided-run data above it.

- timestamp: 2026-07-15T19:22:00Z
  checked: .planning/phases/22-.../22-UI-SPEC.md lines 216-224 ("Sites Needing Review — new third
    table" section)
  found: "Placed on the existing approval-queue page, after the \"Recently Decided\" section, same
    page (no new URL, no navbar entry — D-07)" followed by the exact <h5>+{% render_table %}
    markup that ended up in approval_queue.html verbatim.
  implication: UI-SPEC carried D-07's ordering forward unchanged and specified no alternate visual
    grouping (no card, no distinct banner, no placement ahead of "Recently Decided"). 22-03-PLAN.md
    Task 2 then implemented this UI-SPEC snippet exactly. The chain design-decision -> UI-SPEC ->
    PLAN -> template is fully consistent and traceable; nothing drifted between docs and code.

- timestamp: 2026-07-15T19:24:00Z
  checked: solsys_code/campaign_tables.py ApprovalQueueTable.render_site() (lines 247-281) and
    _render_site_search_widget() (lines 212-245)
  found: "`site_short_name = Accessor('site__short_name').resolve(record, quiet=True); if
    site_short_name: return super().render_site(record)` -- this early-return fires for ANY row
    whose run.site is set to ANY Observatory, with no check of whether that Observatory is a
    placeholder. Only when site_short_name is falsy (run.site is None) does the method proceed to
    render the live-search widget. There is no branch, anywhere in render_site() or
    _render_site_search_widget(), that renders a search input for a resolve-mode row whose site is
    already set."
  implication: This is the exact code responsible for the DCT row showing only plain text + a
    Resolve button, matching the screenshot description precisely.

- timestamp: 2026-07-15T19:25:00Z
  checked: .planning/phases/22-.../22-03-PLAN.md Task 2 <action> ("KEEP the site_short_name
    early-return for resolve-mode rows whose site IS already set (the projection-failed retry
    state — 22-REVIEWS.md finding 8): such a row shows its resolved site as plain text ... and its
    Resolve button alone re-attempts the projection — no site input is needed or rendered.")
  found: This is an explicit, deliberate design choice from 22-REVIEWS.md finding 8c / 22-03-PLAN.md
    Task 2 -- the plan's own mental model treats "run.site is set" as synonymous with "this is the
    projection-failed retry state" (a resolved site, just a failed calendar write). It never
    considers the case where run.site is set to a placeholder Observatory (itself an unresolved,
    flagged-for-review record) rather than a genuinely resolved one.
  implication: The gap is a conflation in the phase's own design model between two states that
    look identical from render_site()'s single site_short_name check but are semantically
    different: (1) site correctly resolved, only the CalendarEvent projection needs retry; (2)
    site is itself a placeholder/wrong Observatory record that needs correcting via search, not
    just a projection retry.

- timestamp: 2026-07-15T19:27:00Z
  checked: solsys_code/campaign_utils.py resolve_site() (lines 115-213), tier 3
    (lines 196-213)
  found: "When create_placeholder=True (the DEFAULT parameter value) and tiers 1/2 both miss,
    resolve_site() creates `Observatory.objects.create(obscode=code, name=f'NEEDS REVIEW: {code}',
    short_name=code)` -- a placeholder record with a blank timezone (never set) and no `is_real`/
    `is_placeholder` field anywhere on the Observatory model. The only signal that a given
    Observatory row is a placeholder is the 'NEEDS REVIEW: ' prefix baked into its `name` string at
    creation time."
  implication: Confirms exactly how a placeholder Observatory like DCT ('NEEDS REVIEW: DCT', blank
    timezone) gets created in the first place (very likely an earlier CSV import or an
    earlier/pre-Phase-22 approve-flow call with create_placeholder defaulted True, before Phase 21/
    22 introduced create_placeholder=False for the staff resolve/approve paths) -- and confirms
    there is NO existing model-level helper anywhere in the codebase (grepped
    solsys_code_observatory/models.py: zero hits for 'NEEDS REVIEW'/'placeholder'/'is_placeholder')
    that render_site() (or anything else) could have reused to detect this state. The gap is
    structural: no representation of "this Observatory is a placeholder" exists outside the name
    string convention, and nothing reads that convention at render time.

- timestamp: 2026-07-15T19:28:00Z
  checked: solsys_code/campaign_views.py CampaignRunDecisionView.post() resolve_site branch, per
    22-03-PLAN.md Task 1 step (2): "Resolution — only when run.site is None (D-06 never-re-resolve)"
  found: The resolve_site view action calls resolve_site()/writes a new site ONLY when
    `run.site is None`; when run.site is already set (step 3 of the plan's action), resolution is
    skipped entirely -- "resolve_site is never called" -- and the code falls straight to
    re-attempting the projection.
  implication: Even if render_site() DID render a search widget for an already-set placeholder
    site, the current resolve_site view action would refuse to act on a new selection for that row
    -- the D-06 "never re-resolve an already-set site" guard (originally scoped to prevent a race
    between two concurrent staff POSTs resolving the SAME unresolved run -- 22-REVIEWS.md finding
    5 / SITE-03 bug class) has been extended, by omission, to also mean "never let staff manually
    replace a previously-set site value in a later, unrelated request." Both the rendering layer
    and the view layer independently lack any "this site is wrong, let me search again" path.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: |
  Two distinct root causes, both traceable to Phase 22's own design/planning artifacts rather than
  an implementation slip -- the code faithfully implements what was planned; the plans themselves
  left a gap that only surfaced once a human looked at the rendered page with real data.

  (A) Placement: 22-CONTEXT.md D-07 explicitly locked the "Sites Needing Review" table as the
  THIRD table on the approval-queue page, in the fixed order "pending / decided /
  sites-needing-review". 22-UI-SPEC.md and 22-03-PLAN.md Task 2 carried this ordering forward
  verbatim into src/templates/campaigns/approval_queue.html: <h4>Approval Queue</h4>, then
  <h5>Pending Review</h5>, <h5>Recently Decided</h5>, <h5>Sites Needing Review</h5>, all as plain
  DOM siblings with no visual differentiation (card/border/priority styling) separating the
  actionable "Sites Needing Review" work queue from the purely historical "Recently Decided" audit
  table above it. Placing it after (potentially many) decided-run rows, with only an <h5> and
  Bootstrap's default vertical spacing to distinguish it, is why a real user scrolling the actual
  page perceives it as buried at the bottom / not grouped under the "Approval Queue" banner, even
  though structurally it IS a direct child of that page. This design decision was never checked
  against an actual rendered page (only described in text across CONTEXT/UI-SPEC/PLAN docs) before
  Phase 22's automated verification passed and UAT caught it.

  (B) Placeholder correction: ApprovalQueueTable.render_site() (solsys_code/campaign_tables.py)
  gates its resolve-mode live-search widget rendering solely on whether `site__short_name` is
  truthy -- i.e., whether `run.site` points to ANY Observatory at all. This check cannot
  distinguish a genuinely-resolved Observatory from a placeholder one (created by
  campaign_utils.resolve_site()'s tier-3 fallback -- `name=f'NEEDS REVIEW: {code}'`, blank
  timezone -- when `create_placeholder=True`, its DEFAULT). 22-03-PLAN.md Task 2 (informed by
  22-REVIEWS.md finding 8c) explicitly designed this early-return to treat "site already set" as
  synonymous with "the projection-failed retry state" (a genuinely resolved site whose calendar
  write merely failed and needs retrying) -- the plan's mental model never considered the case
  where the already-set site is itself a placeholder/wrong record. Compounding this, the
  CampaignRunDecisionView.post() `resolve_site` action (same plan, Task 1 step 2/3) only calls
  `resolve_site()`/writes a new site when `run.site is None` (the D-06 "never re-resolve" guard,
  originally scoped to prevent two concurrent staff POSTs from racing to resolve the SAME
  unresolved run -- 22-REVIEWS.md finding 5). That guard has the side effect of also blocking any
  LATER, deliberate staff correction of an already-set site -- there is no server-side action that
  accepts a new `site_selection` for a run whose `run.site` is already non-null. No Observatory
  model field/helper exists anywhere in the codebase to mark/detect a placeholder record (grepped
  solsys_code_observatory/models.py -- zero hits); the only signal is the ad-hoc `'NEEDS REVIEW: '`
  name-string prefix set at creation time, which nothing at render or resolve time currently reads.
  The result: a row whose site was set via an earlier create_placeholder=True call (e.g. Phase 21's
  approve flow, or CSV import, before Phase 22 introduced create_placeholder=False for staff
  paths) is permanently locked out of correction through the Sites Needing Review UI -- both the
  rendering layer (no widget shown) and the view layer (resolution skipped when site is set)
  independently block it.
fix:
verification:
files_changed: []
