---
phase: 28-operator-assisted-attribution
plan: 04
subsystem: ui
tags: [django, templates, django-tables2, staff-workflow, xss]

# Dependency graph
requires:
  - phase: 28-01
    provides: CalendarEventDismissal/ObservationRecordDismissal models, CalendarEventMeta audit fields
  - phase: 28-02
    provides: campaign_attribution matcher module (candidates_for_event/candidates_for_record, band scoring)
  - phase: 28-03
    provides: AttributionQueueView/AttributionDecisionView, campaigns:attribution routes, banner count on CampaignListView
provides:
  - Four-section staff attribution page (events worklist, records worklist, Dismissed table, Confirmed table)
  - Confidence-band badges and D-09 checkbox gate rendered in the template, mirroring 28-03's server-side sole-high-candidate rule
  - Campaign-list attribution banner (staff-only) linking to the queue
  - Operator runbook section documenting the attribution pass, dismissal semantics, and the Phase 29 done signal
affects: [phase-29-reconcile-sweep]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Template comment convention: {% comment %}...{% endcomment %}, never multi-line {# ... #} (TemplateCommentSyntaxSweepTest enforces this repo-wide)"
    - "Confidence badge vs checkbox are mutually exclusive in a row's leading cell -- checkbox only for the sole High-band candidate, badge otherwise"

key-files:
  created:
    - src/templates/campaigns/attribution_queue.html
    - solsys_code/campaign_tables.py
  modified:
    - src/templates/campaigns/campaign_list.html
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_attribution_views.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "TestEvidenceColumns' single-candidate fixture is the sole High-band candidate for its orphan, so its leading cell renders a checkbox, not a badge-success chip -- the row's border-left border-success is the High-band signal in that case. Corrected the test to assert border-success there instead of badge-success, and added an explicit badge-success assertion to the existing two-High-candidates (ambiguous, non-checkboxable) test in TestBandFilterAndBanner, so both branches of the Checkbox Gate Contract (28-UI-SPEC.md) are exercised."

patterns-established:
  - "Confidence Band & Checkbox Gate Contract: checkbox in the leading cell only when candidate.run.pk == group.sole_high_candidate_pk; every other row (any band) renders the confidence badge in that same cell instead -- never both, never neither."

requirements-completed: [ATTRIB-01, ATTRIB-02, ATTRIB-06]

# Metrics
duration: 95min
completed: 2026-08-01
---

# Phase 28: Operator-Assisted Attribution Summary — Plan 04

**Staff attribution queue page with evidence-column worklists, confidence-band badges, a D-09 checkbox gate, a campaign-list banner, and the operator runbook's attribution-pass section.**

## Performance

- **Duration:** ~95 min (includes a mid-plan halt/resume; see Issues Encountered)
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- `attribution_queue.html`: four sections (events worklist, records worklist, collapsible Dismissed table, collapsible Confirmed table), band filter, evidence columns (telescope/date/campaign/instrument) with the score as a subordinate `.badge-light.border` chip, confidence badges (`.badge-success`/`.badge-warning.text-dark`/`.badge-secondary`), the D-09 checkbox gate, per-candidate Confirm/Dismiss forms, and the "Attribution complete" drain-to-empty state (D-15).
- `campaign_tables.py`: `DismissedAttributionTable` / `ConfirmedAttributionTable` (django-tables2) for the Dismissed/Confirmed sections.
- `campaign_list.html` + `CampaignListView`: staff-only banner showing the attribution backlog count with a link to the queue; anonymous visitors see neither.
- `docs/runbooks/telescope_runs_calendar.rst`: new section documenting the attribution pass, what confirming/dismissing means, and the "done" signal Phase 29's reconcile sweep depends on.
- 3 new test classes (`TestEvidenceColumns`, `TestBandFilterAndBanner`, `TestQueueDrainsToEmpty`) covering ATTRIB-01/02/06, including a stored-XSS control on the free-text dismissal reason.

## Task Commits

Each task was committed atomically:

1. **Task 1: Attribution page template + Dismissed/Confirmed tables** - `8bedd01` (feat)
2. **Task 2: Campaign-list banner + operator runbook section** - `0a2f461` (docs)
3. **Task 3: Evidence-column, band-filter, banner and drain-to-empty tests** - `54b4cab` (test)

_Note: Task 3 also fixed a leftover multi-line `{# ... #}` Django comment in the template (line 196) that the repo's `TemplateCommentSyntaxSweepTest` catches -- the other three in the file had already been converted to `{% comment %}` blocks._

## Files Created/Modified
- `src/templates/campaigns/attribution_queue.html` - The four-section staff attribution page
- `solsys_code/campaign_tables.py` - `DismissedAttributionTable`, `ConfirmedAttributionTable`
- `src/templates/campaigns/campaign_list.html` - Staff-only attribution banner
- `solsys_code/campaign_views.py` - Banner count wiring in `CampaignListView`
- `solsys_code/tests/test_campaign_attribution_views.py` - 3 new test classes (43 tests total in file)
- `docs/runbooks/telescope_runs_calendar.rst` - Attribution-pass runbook section

## Decisions Made
- **Checkbox-vs-badge assertion fix:** `TestEvidenceColumns`'s single-candidate scenario produces the sole High-band candidate for its orphan, which per the Checkbox Gate Contract (28-UI-SPEC.md) renders a checkbox in the leading cell instead of a `badge-success` chip. The original (halted-agent-authored) assertion expected `badge-success` unconditionally and failed. Fixed by asserting `border-success` (the row's actual High-band signal when checkboxable) in that test, and added an explicit `badge-success` assertion to `test_checkbox_absent_when_orphan_has_two_high_band_candidates` in `TestBandFilterAndBanner`, which is the ambiguous (non-checkboxable) High-band scenario where the badge genuinely renders. This keeps both branches of the mutually-exclusive checkbox/badge design under test without touching the already-verified template logic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test/spec mismatch] Corrected TestEvidenceColumns' band-badge assertion**
- **Found during:** Task 3 resume (post-halt), verification run
- **Issue:** The test asserted `badge-success` for a candidate that is the sole High-band candidate for its orphan; per the template's (already-committed, Task-1) checkbox gate, that row renders a checkbox instead of the badge, so the assertion could never pass.
- **Fix:** Asserted `border-success` (the correct High-band signal for a checkboxable row) instead, and added a `badge-success` assertion to the existing ambiguous-candidates test where the badge does render.
- **Files modified:** solsys_code/tests/test_campaign_attribution_views.py
- **Verification:** Full `test_campaign_attribution_views` module (43/43) and the broader `solsys_code` suite (767/767, excluding `test_views.TestEphemeris` per project convention) pass.
- **Committed in:** 54b4cab (Task 3 commit)

**2. [Rule 1 - Repo convention] Converted the last multi-line `{# ... #}` template comment**
- **Found during:** Task 3 resume, full-suite verification
- **Issue:** `TemplateCommentSyntaxSweepTest` (repo-wide) failed on one remaining multi-line `{# ... #}` block in `attribution_queue.html:196` that had not yet been converted to `{% comment %}...{% endcomment %}`.
- **Fix:** Converted the block to `{% comment %}` syntax, matching the other three in the same file.
- **Files modified:** src/templates/campaigns/attribution_queue.html
- **Verification:** `TemplateCommentSyntaxSweepTest` and the full suite pass.
- **Committed in:** 54b4cab (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — completing/correcting already-in-flight Task 3 work, no scope creep beyond the plan's own test task).
**Impact on plan:** Both fixes were required to make the plan's own Task 3 acceptance criteria (full suite green) actually pass. No new behavior was added beyond what Task 3 specified.

## Issues Encountered
- **Mid-plan halt/resume:** The original executor agent for this plan was stopped mid-Task-3 (by explicit user instruction, due to session usage) after committing Tasks 1 and 2 and leaving Task 3's test/template edits uncommitted-but-complete in the worktree, with one failing test. On resume, the orchestrator inspected the halted worktree directly (rather than re-running Task 3 from scratch), found the uncommitted work coherent and nearly correct, diagnosed the one failing assertion and the one remaining template-comment-syntax violation against the plan/UI-SPEC contracts, fixed both, reran the full verification suite, and committed. No work was discarded or redone.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 plans of Phase 28 (28-01 through 28-04) are now complete and merged to `issue37-telescope-runs-calendar`.
- ATTRIB-01, ATTRIB-02, ATTRIB-06 requirements satisfied by this plan; ATTRIB-03/04/05 were covered by 28-01/28-03 per their own SUMMARY.md files.
- Phase 29's reconcile sweep can rely on the "done" signal (`is_drained`/`unattributable_count`) this plan's template surfaces.

---
*Phase: 28-operator-assisted-attribution*
*Completed: 2026-08-01*
