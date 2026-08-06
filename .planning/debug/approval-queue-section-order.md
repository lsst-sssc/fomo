---
status: diagnosed
trigger: "UAT Test 8 (.planning/phases/27-the-canonical-run-record/27-UAT.md): 'Yes, I would still like the \"Sites Needing Review - action required\" section to be at the top not the bottom'"
created: 2026-08-06T00:00:00Z
updated: 2026-08-06T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — pure template ordering oversight, no functional/data dependency between the three sections.
test: Read approval_queue.html, ApprovalQueueView.get_context_data, and 27-UAT.md for any documented rationale for current order.
expecting: N/A — root cause confirmed.
next_action: Return diagnosis (goal: find_root_cause_only). Do not fix.

## Symptoms

expected: Sites Needing Review is the first thing staff see on /campaigns/approval-queue/ when it's the only actionable table (e.g. when pending_count is 0 but site_review_count is non-zero).
actual: User reported (verbatim): "Yes, I would still like the 'Sites Needing Review - action required' section to be at the top not the bottom" — confirming the section currently renders at the bottom of the page, below "Pending Review" and "Recently Decided" tables.
errors: None reported
reproduction: Test 8 in .planning/phases/27-the-canonical-run-record/27-UAT.md. Open /campaigns/approval-queue/ as a staff user. The template is src/templates/campaigns/approval_queue.html — current order is: h5 "Pending Review" + pending_table, h5 "Recently Decided" + decided_table, then the "Sites Needing Review — action required" card.
started: Discovered during a UAT re-verification pass on 2026-08-06, while confirming an earlier fix (widening the campaign_list.html nav banner condition to `pending_count or site_review_count`, commit 8da4060) actually made the queue reachable.

## Eliminated

- hypothesis: The three sections have a rendering/data dependency on each other (e.g. review_table computation depends on pending_table or decided_table context, so reordering could break something).
  evidence: ApprovalQueueView.get_context_data (solsys_code/campaign_views.py:361-423) builds pending_table, decided_table, and review_table from three fully independent querysets/context vars (pending_qs, decided_qs, review_qs). The only shared object is `candidate_pool` (built once, reused by pending_table and review_table, decided_table doesn't need it) — an optimization to avoid a second build_site_candidates() call, not an ordering constraint. All three are added to context unconditionally and independently; reordering the template blocks that render them has zero effect on this Python logic.
  timestamp: 2026-08-06T00:00:00Z

- hypothesis: The order is a deliberate design decision recorded somewhere (PLAN/RESEARCH/CONTEXT docs, class docstring, or CLAUDE.md/PATTERNS.md convention) that would need to be overridden knowingly.
  evidence: Searched .planning/phases/27-the-canonical-run-record/*.md and the ApprovalQueueView docstring. The docstring literally describes the page as a "Staff-only two-section approval queue: pending review + recently decided (D-01/D-02)" -- it predates the "Sites Needing Review" card entirely. The review_table/card was added later per the "D-07/27.1-03" comment directly above its context-building block (campaign_views.py:402-416), simply appended after the two pre-existing sections in both the view method and the template. No planning doc, comment, or docstring states any priority/ordering rationale for the card being last. This is consistent with straightforward append-at-the-end-of-file drift, not intentional design.
  timestamp: 2026-08-06T00:00:00Z

## Evidence

- timestamp: 2026-08-06T00:00:00Z
  checked: src/templates/campaigns/approval_queue.html (full file, 25 lines)
  found: |
    Block order top-to-bottom: (1) h5 "Pending Review" + {% render_table pending_table %},
    (2) h5 "Recently Decided" + {% render_table decided_table %},
    (3) a `<div class="card border-warning mt-4">` with header "Sites Needing Review — action required"
    wrapping {% render_table review_table %}.
    No {% if %} conditionals gate any of the three blocks -- all three always render regardless
    of row counts (pending_count/site_review_count are NOT even in this view's context; the
    template renders unconditionally and each django-tables2 table shows its own empty_text
    when its queryset is empty, e.g. "No submissions waiting for review.").
  implication: The "top vs bottom" issue is purely a static HTML block-ordering problem in one file. There is no conditional layout logic to preserve or extend.

- timestamp: 2026-08-06T00:00:00Z
  checked: solsys_code/campaign_views.py:348-423 (ApprovalQueueView.get_context_data)
  found: |
    pending_table (from pending_qs, filter approval_status=PENDING_REVIEW), decided_table
    (from decided_qs, exclude PENDING_REVIEW, capped -pk[:20]), and review_table (from
    runs_needing_site_review(), D-07/27.1-03) are three independently-built ApprovalQueueTable
    instances added to context unconditionally: context['pending_table'], context['decided_table'],
    context['review_table']. No pending_count / site_review_count vars are added here (those
    only exist on the separate CampaignListView, for the /campaigns/ nav banner -- confirmed via
    grep, campaign_views.py:238-253).
  implication: Reordering the three template blocks is safe from a data/context standpoint -- none of the tables' correctness depends on render order, and there's no existing count-based conditional to adapt.

- timestamp: 2026-08-06T00:00:00Z
  checked: .planning/phases/27-the-canonical-run-record/27-UAT.md lines 62-94 and 171-181
  found: |
    Test 8's `note` explicitly frames this as a NEW finding distinct from the original nav-banner
    gap (already fixed by commit 8da4060 widening `pending_count` to `pending_count or
    site_review_count` in campaign_list.html). The structured re-verification entry (lines
    171-181) records: truth="Sites Needing Review is the first thing staff see on the approval
    queue when it's the only actionable table", status=failed, artifacts=[{path:
    "src/templates/campaigns/approval_queue.html", issue: "Sites Needing Review card renders
    after Pending Review and Recently Decided tables, not before"}], root_cause="" (not yet filled
    in -- this debug session fills it), missing=[] (UAT authors already identified the fix
    location, just hadn't run root-cause diagnosis).
  implication: UAT documentation already correctly scoped this to approval_queue.html's block order; investigation confirms no other files/logic are implicated.

## Resolution

root_cause: |
  In src/templates/campaigns/approval_queue.html, the three sections (Pending Review, Recently
  Decided, Sites Needing Review) are hardcoded in a fixed, unconditional HTML order that simply
  reflects the chronological order the features were built in -- "Pending Review" + "Recently
  Decided" were the original two-section page (view docstring: "Staff-only two-section approval
  queue... D-01/D-02"), and the "Sites Needing Review — action required" card was appended
  afterward, at the bottom of both the template and ApprovalQueueView.get_context_data, when
  D-07/27.1-03 introduced it. No conditional logic, shared state, or documented design rationale
  ties the sections' visual order to their append order -- it is an unreviewed ordering oversight
  from incremental feature addition, not a functional dependency. Because pending_count is not
  even passed into ApprovalQueueView's context (only CampaignListView's nav banner uses it), the
  template can't and doesn't already adapt position based on which table has actionable rows --
  it always renders Pending Review first regardless of whether it has any rows, which is exactly
  the symptom Test 8 describes (Sites Needing Review buried at the bottom even when it's the only
  actionable section, since pending_count is 0 and decided_table is purely informational).
fix: (not applied — find_root_cause_only mode)
verification: (not applicable — diagnosis only)
files_changed: []
