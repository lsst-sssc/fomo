# Phase 15: Per-Campaign Table View (Read Path) - Context

**Gathered:** 2026-07-03
**Status:** Ready for planning

<domain>
## Phase Boundary

A coordinator can see every `CampaignRun` for a campaign (`TargetList`) in one sortable,
filterable, paginated table that replaces the shared spreadsheet — reachable from a target's
detail page and from a new navbar "Campaigns" entry — with contact details visible only to
authenticated staff.

**Scoping correction vs. research docs:** `.planning/research/SUMMARY.md`/`FEATURES.md` were
written before Phase 14 locked the schema and describe this as a "per-target" table. The actual
model makes `CampaignRun.campaign` (→ `TargetList`) the required FK and `CampaignRun.target`
optional/nullable (Phase 14 D-07: only auto-filled for single-target campaigns). ROADMAP.md and
REQUIREMENTS.md — finalized after research, during roadmap creation — correctly say
"per-campaign." **The table is scoped by `campaign` (TargetList), not `target`.** This is not a
re-litigated decision; it's a correction downstream agents need so they don't follow the stale
research framing.

Out of scope for this phase (belongs to later phases): the community submission form and
approval-queue UI (Phase 16), calendar projection (Phase 16), coverage-gap analysis (Phase 17),
and VIEW-05 (submitter opt-in public contact display — deferred per REQUIREMENTS.md).

</domain>

<decisions>
## Implementation Decisions

### Campaign discovery & navigation
- **D-01:** A Target's detail page finds "its" campaign(s) via **TargetList membership**, not
  via `CampaignRun.target`: `TargetList.objects.filter(targets=this_target,
  campaign_runs__isnull=False).distinct()`. This works even for rows where the optional
  `CampaignRun.target` FK was never set (D-07 from Phase 14 only guarantees auto-fill for
  single-target campaigns going forward, not universal population).
- **D-02:** If a Target belongs to 2+ qualifying campaigns, the target-detail integration point
  shows **one button/link per matching campaign** (each labeled with the `TargetList` name) —
  not a single button to an intermediate chooser page. Mostly future-proofing: 3I/ATLAS today is
  single-target/single-campaign.
- **D-03:** The navbar "Campaigns" entry links to a **new dedicated campaigns list page** — a
  view listing every `TargetList` that has ≥1 `CampaignRun`, each linking to its per-campaign
  table. Not a reuse of TOM Toolkit's existing `tom_targets:targetgrouping` view (that view is
  login-required and lists every `TargetList` regardless of campaign status — noisier and
  inconsistent with this feature's open read path).
- **D-04:** The campaign list page and per-campaign table are **open to anonymous visitors**,
  matching FOMO's existing `AUTH_STRATEGY='READ_ONLY'`/`OPEN` targets convention and VIEW-03's
  wording ("excluded for anonymous requests" implies anonymous users can reach the page — just
  not the contact fields).

### Approval-status visibility
- **D-05:** The table **shows all `CampaignRun` rows regardless of `approval_status`** for
  everyone (staff and non-staff/anonymous alike) in this phase — it does **not** filter to
  `approved`-only. Deliberate choice over the "forward-compatible" filtered option: every row
  today is bootstrap-imported and already `approved` (Phase 14 D-03), so filtering has no visible
  effect yet; the operator chose to defer the approval-status filter to Phase 16, when pending
  submissions actually start to exist. **Phase 16 planning must not assume Phase 15 already
  gates on `approval_status` — it doesn't.**
- **D-06:** Consistent with D-05: staff and non-staff both see `rejected` rows in the main table
  (no hidden review-queue-only treatment in this phase).
- **D-07:** Default `run_status` filter state on page load is **unfiltered — show everything**.
  The `run_status` filter (VIEW-04) is opt-in narrowing, not a default-hide of dead-end statuses
  (`cancelled`/`not_awarded`/`weather_tech_failure`).
- **D-08:** `approval_status` gets a **visually distinct badge/highlight** in the table (not a
  plain unstyled column) — since non-approved rows are mixed in with approved ones (per D-05),
  staff need to spot them at a glance. Follow the existing `calendar_display_extras.py`
  badge/color precedent for implementation style (planner/researcher's call on exact mechanism).

### Table columns, sort & paging
- **D-09:** Column set is **spreadsheet-parity** — mirror the real 3I sheet closely rather than a
  curated subset with detail-expansion. Include telescope_instrument, site, obs_date, UT
  start/end, filters_bandpass, run_status (badged per D-08), open_to_collaboration, contact
  (staff-only, see below), plus the free-text fields (observation_details, weather,
  observation_outcome, publication_plans, comments) as columns rather than hidden behind a
  detail link.
- **D-10:** Default sort order is **`obs_date`, most recent first** (newest activity at the top).
  `django-tables2` still lets users re-sort by any column.
- **D-11:** Pagination is **25 rows per page**.
- **D-12:** The `run_status` filter (VIEW-04) is **multi-select** (e.g., show `planned` OR
  `observed` at once) — not a single-value dropdown. Use `django-filter`'s multi-select-capable
  filter type (e.g. `MultipleChoiceFilter`).

### Staff-only contact gating
- **D-13:** Non-staff/anonymous viewers get `contact_person`/`contact_email` columns **omitted
  from the table entirely** — not shown as masked/blank placeholders. Gate at the view layer (the
  columns never reach the template context for non-staff), per the research doc's Pitfall 2
  framing: defense in depth, not just a template-level hide.
- **D-14:** Phase 15 adds **no contact/reach-out path** for anonymous visitors interested in an
  `open_to_collaboration` run (no generic "contact FOMO admins" link either). That problem is
  explicitly VIEW-05's scope (deferred per REQUIREMENTS.md: "Submitter can opt in to public
  display of their contact details"). Phase 15 only surfaces `open_to_collaboration` as a
  filterable/visible column.

### Claude's Discretion
- **Staff check mechanism (D-13's "staff"):** operator deferred to Claude. Use
  `request.user.is_staff` (Django's built-in flag) — no new permission/group needed, matches the
  admin-style gating pattern already implied elsewhere in FOMO (e.g. `CreateObservatory`). Do not
  introduce a dedicated permission/group for this unless research surfaces a reason to.
- Exact `django-tables2`/`django-filter` implementation details (Table subclass structure,
  FilterSet field wiring, template used) — not asked; both libraries are already installed and
  unused elsewhere in this codebase, first real consumer here.
- Exact URL names/paths for the new campaigns list view and per-campaign table view.
- Exact badge/styling mechanism for D-08's approval_status treatment (new template tag vs. reuse
  of `calendar_display_extras.py` patterns).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` §"Phase 15: Per-Campaign Table View (Read Path)" — goal, success criteria
- `.planning/REQUIREMENTS.md` §"Per-Campaign Table View (VIEW)" (VIEW-01..04) — full requirement
  text; also note VIEW-05 (deferred: submitter opt-in public contact display) as the boundary for
  D-14

### Research (read with the Phase 14 scoping correction in mind — see `<domain>`)
- `.planning/research/SUMMARY.md` — stack recommendation (`django-tables2`, `django-filter`, no
  new dependencies), Pitfall 2 (PII exposure on `AUTH_STRATEGY='READ_ONLY'`) — still accurate
  despite the stale "per-target" framing
- `.planning/research/FEATURES.md` — field-value rationale for `open_to_collaboration` filtering

### Prior phase context
- `.planning/phases/14-campaign-data-model-bootstrap-import/14-CONTEXT.md` — D-01/D-02 (status
  vocabulary split), D-07 (optional Target auto-assignment), D-08/D-09 (site resolution/flagging)
- `.planning/seeds/target-linked-run-submission-form.md` — original seed + 2026-07-02 enrichment;
  3I/ATLAS sheet as the reference model for spreadsheet-parity columns (D-09)

### Existing code precedent
- `solsys_code/models.py` — `CampaignRun` model (D-09's field inventory), `ApprovalStatus`/
  `RunStatus` TextChoices (D-05/D-07/D-12's filter/badge values)
- `solsys_code/apps.py` (`SolsysCodeConfig.target_detail_buttons`) — existing integration-point
  pattern for D-01/D-02 (target-detail campaign links), same mechanism as the "Make Ephemeris"
  button
- `tom_common`'s `nav_items()` app-config hook (see `tom_common/templatetags/tom_common_extras.py`
  `show_individual_app_partial`) — mechanism for D-03's navbar "Campaigns" entry; FOMO's
  `solsys_code/apps.py` does not yet define `nav_items()`, this phase adds it
- `tom_targets.models.TargetList` — has no "is this a campaign" flag; "campaign" is defined
  operationally as "a `TargetList` with ≥1 `CampaignRun`" (D-01/D-03 both rely on this)
- `solsys_code/templatetags/calendar_display_extras.py` — badge/color precedent referenced by
  D-08 (`proposal_color`, `status_border_css` pattern)
- `src/fomo/settings.py` — confirms `django_filters`/`django_tables2` already in `INSTALLED_APPS`,
  unused elsewhere; first real consumer is this phase
- `src/fomo/urls.py` — existing URL-wiring convention (e.g. `calendar_urls.py` included before
  `tom_common.urls`) to follow for the new campaigns list/table URLs

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `django_tables2`/`django_filters` — installed, unused, directly applicable to VIEW-01/VIEW-04
  (sortable/paginated table + lifecycle-status/collaboration-flag filtering).
- `SolsysCodeConfig.target_detail_buttons()` (`solsys_code/apps.py`) — existing pattern for
  injecting a button into the target detail view; extend with a similar `nav_items()` method for
  D-03.
- `calendar_display_extras.py` template tags — precedent for D-08's approval_status badge styling
  (proposal-keyed color, WCAG-aware text color).

### Established Patterns
- "Flag, don't silently guess" precedent (`CalendarEventTelescopeLabel.is_verified`,
  `CampaignRun.site_needs_review`) — conceptually similar to D-08's "make non-approved rows
  visually distinct rather than silently blending in."
- View-layer PII gating (not template-only) — matches D-13 and the research doc's Pitfall 2
  guidance directly.

### Integration Points
- New view(s) likely belong in a new `solsys_code/campaign_views.py` (per research doc's
  suggested module name) or alongside existing views in `solsys_code/views.py` — planner's call.
- `solsys_code/apps.py` needs both `target_detail_buttons()` extended (D-01/D-02) and a new
  `nav_items()` method (D-03).
- New URLs registered in `src/fomo/urls.py`, following the `calendar_urls.py` inclusion pattern.

</code_context>

<specifics>
## Specific Ideas

- The real 3I/ATLAS sheet
  (https://docs.google.com/spreadsheets/d/1INhxLWlHoa-JkW-uKRzmSyms06wI80wEXTqBJSR3YAI/) is the
  reference model for D-09's spreadsheet-parity column set — coordinators already know this
  layout, and the table should feel like a direct replacement, not a re-designed subset.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 15's scope. VIEW-05 (submitter contact opt-in) and the
Phase 16 `approval_status` filter (D-05 note) are explicitly out of scope here and already
tracked in REQUIREMENTS.md/ROADMAP.md, not newly deferred by this discussion.

### Reviewed Todos (not folded)
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — proposes renaming
  `calendar_utils.py` private helpers. Matched at score 0.4 (keyword overlap only); unrelated to
  the campaign table view. Reviewed, not folded.
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — proposes
  extracting `SITE_TELESCOPE_MAP`/instrument-extraction logic. Matched at score 0.2; already
  resolved per Phase 14's context (extraction already happened in `calendar_utils.py`) and
  unrelated to this phase anyway. Reviewed, not folded.

</deferred>

---

*Phase: 15-Per-Campaign Table View (Read Path)*
*Context gathered: 2026-07-03*
