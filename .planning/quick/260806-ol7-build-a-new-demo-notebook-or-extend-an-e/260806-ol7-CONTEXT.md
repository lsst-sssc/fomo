# Quick Task 260806-ol7: Build a new demo notebook exercising the full v2.2 workflow - Context

**Gathered:** 2026-08-07
**Status:** Ready for planning

<domain>
## Task Boundary

Build a new demo notebook under `docs/notebooks/pre_executed/` that walks through creating a
Campaign (`tom_targets.TargetList`) and CampaignRuns covering four source/dispatch variations --
classical (`source=CLASSICAL_FILE`), LCO generic queue (`source=LCO_QUEUE`), ESO/other queue
(`source=ESO_QUEUE`), and a class-wide/site-agnostic run (`telescope_class` set, `site=None`) --
to demonstrate and exercise the complete workflow added/changed across milestone v2.2 (Phases
26-29: canonical run record, staff surfaces, operator-assisted attribution, and the reconciler).

</domain>

<decisions>
## Implementation Decisions

### Depth: view-driven, not direct-ORM shortcut
The notebook drives the real staff-facing views via Django's test client (or equivalent
request-cycle simulation from within the notebook), not `CampaignRun.objects.create(...,
approval_status=APPROVED)` shortcuts. Specifically it must exercise:
- The public submission form (creates a `source=WEB`... no -- for CSV/legacy/queue sources this
  step doesn't apply the same way; see per-run-type notes below) / or the relevant creation path
  for each source type
- The approval queue view (approve a pending run)
- Site-review resolution (for a run that needs it)
- The attribution queue (confirm at least one candidate association)

This is the whole point of the ask -- `reconcile_campaign_runs_demo.ipynb` already covers
direct-ORM-created, pre-approved runs plus the reconciler itself (Phase 29 only). This notebook's
job is to cover what THAT one doesn't: the Phase 27/27.1/28 staff-facing workflow surfaces.

### Scope: new standalone notebook
Do not extend `reconcile_campaign_runs_demo.ipynb` -- that notebook's docstring and scope are
specifically the reconciler (Phase 29). Create a new notebook instead. Suggested name:
`campaign_lifecycle_demo.ipynb` (planner may adjust if a clearer name fits FOMO's existing naming
convention better -- check the other five notebook names in `docs/notebooks/pre_executed/` for
precedent).

### Run-type coverage: four variations, not three
Cover all four reconciler dispatch branches in one place:
1. Classical (`source=CLASSICAL_FILE`, resolved ground site) -- per-night dip-corrected events
2. LCO generic queue (`source=LCO_QUEUE`, resolved ground site) -- per-night dip-corrected events
   (same branch as classical per Phase 29's window-shape-dispatch fix from quick task 260805-tad --
   NOT a whole-window container; only class-wide/site-agnostic and satellite runs get the
   whole-window container treatment)
3. ESO/other queue (`source=ESO_QUEUE`, resolved ground site, e.g. modeled on the real VLT/FORS2
   MPC 309 Paranal pattern) -- also per-night dip-corrected events, same branch as classical/LCO
   for the same reason
4. Class-wide/site-agnostic (`telescope_class` set, `site=None`) -- whole-window `RUN:{pk}`
   container event, the one branch genuinely different from the other three

Note for the notebook's own narration: because of the 260805-tad fix, only run #4 will visibly
differ in calendar rendering (whole-window container vs. per-night events) -- #1/#2/#3 will all
render per-night once site-resolved. The notebook should call this out explicitly rather than
implying source alone determines the rendering shape (a common misconception the reconciler
demo notebook and runbook already correct -- match that framing).

### Runbook cross-reference
Add a short pointer in `docs/runbooks/telescope_runs_calendar.rst` (e.g. near the top intro or in
a "See also" section) linking to the new notebook as "the full campaign-lifecycle walkthrough."
One sentence/link, not a restructure.

</decisions>

<specifics>
## Specific Ideas

- Model the ESO_QUEUE run on the real VLT/FORS2 @ MPC 309 (Paranal) pattern already used in
  `29-VERIFICATION.md`'s and `29-UAT.md`'s real dev-DB evidence (RUN:3), for narrative
  consistency with the rest of the milestone's documentation -- doesn't need to be the literal
  same pk, just the same shape (ESO VLT instrument, resolved ground site).
- The notebook should end with a view of `/campaigns/<pk>/` (or the equivalent table/queryset)
  and a calendar render showing all four runs' events side by side, so the "complete workflow"
  claim is visibly true, not just asserted in prose.
- Per CLAUDE.md's paired-docs rule, this notebook itself doesn't pair 1:1 with a single module
  (it's a cross-cutting milestone demo, not tied to one file like the other five notebooks are)
  -- treat `docs/runbooks/telescope_runs_calendar.rst`'s new cross-reference as satisfying the
  paired-docs spirit for this deliverable, and add this notebook to CLAUDE.md's pairing map as
  covering "the full v2.2 campaign lifecycle (Phases 26-29), not a single module" if the planner
  judges that update is in scope for this quick task (optional -- flag as a deviation if skipped).

</specifics>

<canonical_refs>
## Canonical References

- `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` -- closest existing precedent
  for notebook structure, Django setup boilerplate, and the four-CampaignRun seeding pattern
  (though that one skips straight to direct-ORM pre-approved creation; this one must not).
- `docs/runbooks/telescope_runs_calendar.rst` -- existing Q&A-style runbook; sections on "How do
  I reach the approval queue?", "How do I attribute existing calendar events and observation
  records to a run?", and "Can I correct a run's source?" describe the views this notebook must
  actually drive.
- `solsys_code/models.py` `CampaignRun.Source` / `CampaignRun.TelescopeClass` TextChoices --
  authoritative vocabulary for the four run-type variations.
- Quick task `260805-tad` (window-shape dispatch fix) -- authoritative source for why
  LCO_QUEUE/ESO_QUEUE resolved-site runs render per-night, not whole-window, contrary to what an
  earlier design doc might imply.

</canonical_refs>
