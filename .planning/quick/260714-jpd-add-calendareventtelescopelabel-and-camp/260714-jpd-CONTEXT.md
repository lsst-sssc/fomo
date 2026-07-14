# Quick Task 260714-jpd: Add CalendarEventTelescopeLabel and CampaignRun to solsys_code/admin.py - Context

**Gathered:** 2026-07-14
**Status:** Ready for planning

<domain>
## Task Boundary

Add CalendarEventTelescopeLabel and CampaignRun to solsys_code/admin.py

`solsys_code/admin.py` has been the untouched `django-admin startproject` stub since the repo's
first commit — no models in the main `solsys_code` app are registered with Django admin, even
though the sibling `solsys_code_observatory/admin.py` properly registers `Observatory` with a
custom `ObservatoryAdmin`. This task closes that gap for `CalendarEventTelescopeLabel` (the
telescope-label verification sidecar, v1.4) and `CampaignRun` (the campaign-coordination model,
v2.0+), so staff have an admin escape hatch for data they currently can't fix any other way (e.g.
a leftover `NEEDS REVIEW: DCT` placeholder Observatory, or a `CampaignRun`'s `window_start`/
`window_end` when it needs manual correction — no UI in the app currently supports editing those).

</domain>

<decisions>
## Implementation Decisions

### CampaignRun field editability
- `window_start`, `window_end`, and `site` are admin-editable — this closes the real gaps found
  during earlier discussion (no in-app way to fix a mis-resolved site or correct a TBD/range
  window after submission).
- `approval_status` is **read-only** in admin. Reason: changing it normally triggers side effects
  (calendar projection via `insert_or_create_calendar_event`, the D-06 `if run.site is None`
  clobber guard) that live entirely in `CampaignRunDecisionView.post()`, not in the model or a
  signal. Admin must never be able to silently flip a run to `APPROVED` with no `CalendarEvent`
  ever created — that transition must always go through the real approval-queue flow.
- `run_status` (the separate lifecycle field: planned/observed/reduced/published) was not
  discussed explicitly — no side effects are known to be tied to it, so treat it as editable
  unless the planner finds a reason otherwise (Claude's Discretion, see below).

### List display / filters (CampaignRun admin)
- `list_display`: `pk`, `campaign`, `telescope_instrument`, `approval_status`, `run_status`,
  `site`, `window_start`, `window_end` — triage-focused, mirrors the approval queue's
  triage-first column ordering (16-05 precedent: actions/status-forward, not a flat field dump).
- `list_filter`: `approval_status`, `run_status`, `campaign`.
- `search_fields`: `telescope_instrument`, `site_raw`, `contact_person`.

### PII handling (contact_person / contact_email)
- Excluded from `list_display` entirely — never appears in the change-list table, so PII is never
  scannable across many rows at once (extends the app's established SQL-level PII-gating
  discipline from VIEW-03/VIEW-05 to the admin list view).
- Editable in the detail/change view — staff can correct a submitter's typo'd contact info
  directly if needed. Not made read-only (rejected the stricter "read-only in detail" option).

### Claude's Discretion
- `CalendarEventTelescopeLabel` admin config (list_display/list_filter/search_fields) — not
  discussed; this model is a small OneToOneField sidecar on `tom_calendar.CalendarEvent`
  (verified/fallback telescope-label metadata), register it standalone (not as an inline — no
  local admin for `tom_calendar.CalendarEvent` exists to inline it onto) with a sensible minimal
  list_display (e.g. the linked calendar event, verified/fallback status fields — planner to
  confirm actual field names from the model).
- `CampaignRun.run_status` editability (no explicit side-effect concern raised — default to
  editable unless the planner finds one).
- Whether `CampaignRun`'s other non-PII, non-status fields (e.g. `telescope_instrument`,
  `filters_bandpass`, `observation_details`, `open_to_collaboration`, `comments`,
  `contact_public_opt_in`, `site_raw`, `site_needs_review`, `weather`, `observation_outcome`,
  `publication_plans`) are editable in the detail view — not discussed, default to Django admin's
  normal behavior (editable) since no side-effect concern was raised for them, only for the four
  fields called out above.

</decisions>

<specifics>
## Specific Ideas

No specific UI mockups or exact admin class code discussed — the decisions above (editability,
list_display/filters, PII handling) fully scope the implementation.

</specifics>

<canonical_refs>
## Canonical References

- `solsys_code/solsys_code_observatory/admin.py` — the sibling app's existing `ObservatoryAdmin`
  pattern to mirror (custom `ModelAdmin` class + `admin.site.register`), for style consistency.
- `solsys_code/campaign_views.py:CampaignRunDecisionView.post()` — the source of truth for which
  side effects live outside the model (calendar projection, D-06 clobber guard) and therefore
  which fields must stay read-only in admin.
- `solsys_code/campaign_views.py:57-59` (`ALLOWED_FIELDS_FOR_NON_STAFF`) and the VIEW-03/VIEW-05
  PII-gating discipline — the precedent this task's PII handling extends to the admin list view.

</canonical_refs>
