# Phase 16: Submission Form, Approval Queue & Calendar Projection (Write Path) - Context

**Gathered:** 2026-07-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Community members (PIs and external observers) can submit a `CampaignRun` via a public web
form (campaign mandatory, everything else optional) that stays hidden from public views until
a staff member approves it via a dedicated approval-queue page. Approving a submission that has
telescope + date range creates or updates a paired `CalendarEvent` (via
`insert_or_create_calendar_event()`, keyed `CAMPAIGN:{pk}`), and staff get an email notification
when a genuine submission lands (honeypot-tripped submissions are dropped silently, never
processed).

This phase also **extends Phase 15's read path**: the per-campaign table (`CampaignRunTableView`)
currently shows every `CampaignRun` row to everyone regardless of `approval_status` (Phase 15
D-05 deliberately deferred approval-status filtering to this phase — Phase 16 owns adding it).

Out of scope for this phase (belongs to later phases or v2): coverage-gap analysis (Phase 17),
self-service approval bypass for trusted PIs (SUBMIT-06, v2), submitter status-check link
(SUBMIT-07, v2), submitter opt-in public contact display (VIEW-05, v2).

</domain>

<decisions>
## Implementation Decisions

### Approval queue interface
- **D-01:** Staff review pending submissions through a **dedicated staff-facing approval-queue
  page** (a FOMO view, not Django admin bulk actions, not both) with Approve/Reject actions,
  reachable from the existing "Campaigns" navbar entry (`SolsysCodeConfig.nav_items()`). Matches
  this phase's `UI hint: yes` in ROADMAP.md; Django admin moderation actions and third-party
  moderation packages (`django-moderation` etc.) were both considered and rejected — the latter
  is already in REQUIREMENTS.md's Out of Scope table.
- **D-02:** The approval-queue page shows **both** the pending queue (`approval_status
  ='pending_review'`) **and** a "recently decided" section (recently approved/rejected) so staff
  can spot-check or catch a mis-click without leaving the page or falling back to the Phase 15
  table.

### Staff notification (SUBMIT-05)
- **D-03:** Notification recipients are **every `User` with `is_staff=True` and a non-empty
  email** — `User.objects.filter(is_staff=True).exclude(email='')`. No new settings-based
  address (`settings.ADMINS`/a `CAMPAIGN_NOTIFICATION_EMAIL`-style setting) is needed; this
  matches FOMO's existing `is_staff`-only gating convention (no groups/permissions model) already
  used for VIEW-03/D-13's contact-field staff check.
- **D-04:** The email body is a **bare "new submission pending review" ping with a link to the
  approval-queue page** — it does **not** include submission details (telescope, campaign) or
  submitter contact PII (`contact_person`/`contact_email`) in the subject or body. Keeps PII out
  of email infrastructure (inboxes, mail server logs) that sits outside FOMO's own PII-gating
  boundary, consistent with VIEW-03's staff-only contact-field precedent.

### Submission form shape (SUBMIT-01)
- **D-05:** The public form exposes an **intake-relevant subset** of `CampaignRun` fields:
  `campaign` (required) + `telescope_instrument`, `site_raw` (see D-07), `obs_date`, `ut_start`,
  `ut_end`, `filters_bandpass`, `observation_details`, `open_to_collaboration`, `contact_person`,
  `contact_email`, `comments`. **Excluded:** `run_status`, `observation_outcome`, `weather`,
  `publication_plans`, `site` (FK), `site_needs_review` — these are staff/post-observation fields
  that don't make sense to ask a submitter proposing a not-yet-executed run to fill in.
- **D-06:** `contact_person` and `contact_email` are **required at the form-validation level**
  (not the DB level — `CampaignRun.contact_person`/`contact_email` stay `blank=True` at the model
  layer for staff-created rows and CSV imports, per Phase 14). Every public submission must carry
  reachable contact info so staff can follow up on a pending or rejected submission.
- **D-07:** The form captures the observing site as **free text only** (`site_raw`) — it does
  **not** attempt FK resolution against `Observatory` at submission time. `CampaignRun.site`
  resolution (existing `Observatory` → MPC Obscodes API → flagged placeholder; Phase 14 D-08's
  3-tier logic, already implemented in `campaign_utils.resolve_site`) runs when **staff approves**
  the submission, reusing the existing resolution path rather than adding new resolution
  machinery to the form.

### Post-approval visibility scope (SUBMIT-02, extends Phase 15's table)
- **D-09:** Once this phase adds approval-status filtering to the Phase 15 per-campaign table,
  **non-staff (anonymous) visitors see `approved` AND `rejected` rows — only `pending_review` is
  hidden**. A rejected run's row still appears (consistent with Phase 15 D-06's original intent
  that rejected rows aren't hidden). Staff (`is_staff`, same check as Phase 15 D-13) continue to
  see every row regardless of status, unchanged from today.
- **D-10:** The campaigns **list** page (Phase 15 D-03: lists every `TargetList` with ≥1
  `CampaignRun`, for anyone) is **not** changed by this phase. A campaign whose only
  `CampaignRun`s are still `pending_review` continues to appear in the public campaigns list even
  though its table would show no rows to a non-staff visitor (per D-09) until something is
  approved or rejected. This is intentional continuity with Phase 15's existing behavior, not a
  new gap this phase needs to close.

### Claude's Discretion
- **Honeypot mechanics (SUBMIT-04):** a hidden, non-required, non-obviously-named form field
  (not literally `honeypot`) that silently drops the submission on trip — no `CampaignRun`
  created, no error shown to the bot, no notification email sent. Informed by web-search best
  practice during discussion (django-honeypot conventions); no third-party honeypot package
  needed given CLAUDE.md's minimal-dependency convention — a plain Django form field suffices.
- **Approval atomicity mechanism (SUBMIT-03):** research (`SUMMARY.md` Pitfall 6) already
  recommends a conditional atomic update (e.g.
  `CampaignRun.objects.filter(pk=pk, approval_status='pending_review').update(approval_status=
  'approved')`) so a double-approve is a proven no-op; not re-litigated with the user, planner
  should follow the research recommendation and write the double-approve test it specifies.
- Exact URL names/paths for the submission-form view and the approval-queue view.
- Exact crispy-forms layout/field ordering for the submission form (follow the existing
  `EphemerisForm`/`FormHelper` pattern in `solsys_code/forms.py`).
- `EMAIL_BACKEND`/`EMAIL_HOST` configuration for local dev and tests — no email backend is
  currently configured in `src/fomo/settings.py`; planner/researcher's call whether to default to
  Django's console backend for dev or document it as a `local_settings.py` deployment requirement
  (tests should use Django's `locmem` backend / `django.test.override_settings`, not hit a real
  SMTP server).

### Reviewed Todos (not folded)
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — matched by
  keyword overlap only (`calendar`, `shared`); unrelated to submission/approval/calendar
  projection. Already reviewed-not-folded in Phase 15 for the same reason.
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — matched by
  keyword overlap only (`telescope`, `phase`, `calendar`); already resolved per Phase 14/15
  context (extraction already happened in `calendar_utils.py`) and unrelated to this phase.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/ROADMAP.md` §"Phase 16: Submission Form, Approval Queue & Calendar Projection
  (Write Path)" — goal, success criteria
- `.planning/REQUIREMENTS.md` §"Community Submission & Approval (SUBMIT)" (SUBMIT-01..05) and
  §"Calendar Projection (CAL)" (CAL-01..03) — full requirement text

### Research
- `.planning/research/SUMMARY.md` — Pitfall 3 (`CalendarEvent` collision namespace, `CAMPAIGN:
  {pk}` key requirement for CAL-01), Pitfall 6 (approval race conditions — conditional atomic
  update, double-approve test), Pitfall 7 (honeypot + admin notification), Phase 3/Phase 4
  delivery breakdown (maps to this phase's SUBMIT/CAL scope)
- `.planning/seeds/target-linked-run-submission-form.md` — original seed; open questions this
  discussion resolved (notification mechanism → D-03/D-04; self-service bypass → confirmed v2/
  SUBMIT-06, not this phase)

### Prior phase context
- `.planning/phases/15-per-campaign-table-view-read-path/15-CONTEXT.md` — D-05/D-06 (Phase 15
  deliberately does not filter by `approval_status`; explicit note that Phase 16 owns adding the
  filter — this discussion's D-09 is that filter's shape), D-03 (campaigns list page definition,
  unchanged per D-10), D-13 (`is_staff` gating precedent reused for D-01's approval-queue page and
  D-09's visibility split)
- `.planning/phases/14-campaign-data-model-bootstrap-import/14-CONTEXT.md` — D-02 (`approval_status`/
  `run_status` split — `approval_status` already defaults to `PENDING_REVIEW`), D-08/D-09 (3-tier
  site resolution, reused at approval time per D-07)

### Existing code precedent
- `solsys_code/models.py` — `CampaignRun` model (D-05's field inventory to select from),
  `ApprovalStatus`/`RunStatus` TextChoices (`approval_status` already defaults to
  `PENDING_REVIEW`)
- `solsys_code/campaign_utils.py` — `resolve_site` (3-tier resolution, reused at approval time per
  D-07), `insert_or_create_campaign_run`
- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event()` (CAL-01 must route through
  this unchanged, with a `CAMPAIGN:{pk}`-namespaced key, consistent with LCO/Gemini/classical sync
  commands — never construct a `CalendarEvent` directly)
- `solsys_code/campaign_views.py`, `campaign_tables.py`, `campaign_filters.py`, `campaign_urls.py`
  — Phase 15's view/table/filter/URL structure; `CampaignRunTableView.get_queryset()` is where
  D-09's approval-status filter for non-staff needs to land
- `solsys_code/apps.py` (`SolsysCodeConfig.nav_items`, `target_detail_buttons`) — the "Campaigns"
  nav entry already exists; D-01's approval-queue page needs its own reachable entry point (nav
  item, staff-only banner/link on the campaign table, or both — planner's call)
- `solsys_code/forms.py` — existing `crispy_forms`/`FormHelper` pattern (`EphemerisForm`) to follow
  for the new submission form
- `src/fomo/settings.py` — no `EMAIL_BACKEND`/`EMAIL_HOST`/`ADMINS` currently configured (see
  Claude's Discretion above)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `insert_or_create_calendar_event()` (`solsys_code/calendar_utils.py`) — directly reusable for
  CAL-01/CAL-03's create-or-update-with-no-churn requirement; already used by three other sync
  commands with the same idempotency contract this phase needs.
- `campaign_utils.resolve_site` — directly reusable for D-07's approval-time site resolution;
  same 3-tier logic Phase 14's CSV import already exercises.
- `crispy_forms`/`FormHelper` (`solsys_code/forms.py`) — established form-layout pattern to follow
  for the new submission form.
- `django_filters`/`django_tables2` (already used by Phase 15's `CampaignRunTableView`) — the
  approval-queue page (D-01/D-02) can likely reuse the same table/filter machinery for its
  pending + recently-decided sections.

### Established Patterns
- Conditional atomic update for no-double-processing (`.filter(...).update(...)`) — new pattern
  for this phase (SUBMIT-03/D-nothing-new-needed, per research Pitfall 6), but consistent with
  the codebase's existing "no silent churn" discipline (`insert_or_create_calendar_event`'s
  no-churn comparison, `insert_or_create_campaign_run`'s natural-key idempotency).
- `is_staff` as the sole staff-gating mechanism (Phase 15 D-13) — reused for D-01 (who can reach
  the approval-queue page) and D-09 (what non-staff can see in the table).
- View-layer PII/status gating, not template-only (Phase 15 D-13's `ALLOWED_FIELDS_FOR_NON_STAFF`
  pattern) — D-09's approval-status filter should follow the same "exclude at the queryset level"
  discipline, not just hide rows via template conditionals.

### Integration Points
- `CampaignRunTableView.get_queryset()` (`solsys_code/campaign_views.py`) — needs the D-09
  approval-status filter added for non-staff requests.
- New submission-form view and approval-queue view likely belong in `solsys_code/campaign_views.py`
  alongside the existing Phase 15 views, or a new module if the file is getting large (planner's
  call, same discretion Phase 14/15 already exercised for file organization).
- New URLs registered in `solsys_code/campaign_urls.py` (existing `campaigns` namespace).
- Approval action needs to call `insert_or_create_calendar_event()` when telescope + date range
  are present (CAL-01) — this is the phase's one write path into `CalendarEvent`.

</code_context>

<specifics>
## Specific Ideas

No new specific references beyond what's already captured in canonical refs — the discussion
stayed at the decision level (interface choice, field scope, visibility scope) rather than naming
new external examples.

</specifics>

<deferred>
## Deferred Ideas

None newly deferred by this discussion. SUBMIT-06 (self-service approval bypass), SUBMIT-07
(submitter status-check link), and VIEW-05 (submitter opt-in public contact display) were
mentioned during research context but are already tracked as v2 requirements in
`.planning/REQUIREMENTS.md` — not new deferrals from this session.

### Reviewed Todos (not folded)
- `2026-07-02-rename-calendar-utils-py-private-helpers-to-reflect-shared-m.md` — proposes renaming
  `calendar_utils.py` private helpers. Matched at score 0.6 (keyword overlap: `calendar`,
  `shared`); unrelated to submission/approval/calendar-projection scope. Reviewed, not folded
  (also reviewed-not-folded in Phase 15).
- `2026-06-23-extract-site-telescope-mapping-and-instrument-extraction-int.md` — proposes
  extracting `SITE_TELESCOPE_MAP`/instrument-extraction logic. Matched at score 0.6 (keyword
  overlap: `telescope`, `phase`, `calendar`); already resolved per Phase 14's context (extraction
  already happened in `calendar_utils.py`) and unrelated to this phase. Reviewed, not folded.

</deferred>

---

*Phase: 16-Submission Form, Approval Queue & Calendar Projection (Write Path)*
*Context gathered: 2026-07-03*
