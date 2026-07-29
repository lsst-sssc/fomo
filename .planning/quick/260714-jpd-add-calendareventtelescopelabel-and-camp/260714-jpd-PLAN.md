---
phase: quick-260714-jpd
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/admin.py
  - solsys_code/tests/test_admin.py
autonomous: true
requirements: [QUICK-260714-jpd]

must_haves:
  truths:
    - "Both CalendarEventTelescopeLabel and CampaignRun are registered with Django admin: their change-list pages under /admin/solsys_code/ load (HTTP 200) for a logged-in superuser (today solsys_code/admin.py is the empty startproject stub and neither model is reachable)."
    - "The CampaignRun change (detail) page renders approval_status READ-ONLY: its current value (e.g. 'Pending Review') is displayed but there is no editable form widget for it, so a staff admin cannot flip a run to APPROVED and silently bypass the calendar-projection side effects + D-06 clobber guard that live only in CampaignRunDecisionView.post()."
    - "contact_person and contact_email never appear as columns in the CampaignRun change-list (PII not scannable across many rows at once, extending the VIEW-03/VIEW-05 PII-gating discipline to the admin list view), yet both remain editable in the change (detail) view so staff can correct a typo'd contact."
    - "CampaignRun admin exposes list_display = pk, campaign, telescope_instrument, approval_status, run_status, site, window_start, window_end; list_filter = approval_status, run_status, campaign; search_fields = telescope_instrument, site_raw, contact_person."
    - "CalendarEventTelescopeLabel is registered standalone (not inline) with a minimal, sensible config (the linked event plus is_verified), and searching its change-list by event title executes without a FieldError."
  artifacts:
    - solsys_code/admin.py
    - solsys_code/tests/test_admin.py
  key_links:
    - "CampaignRunAdmin.readonly_fields = ['approval_status'] -> Django renders approval_status non-editable in the change form -> no admin path to APPROVED that bypasses CampaignRunDecisionView.post()'s calendar projection + D-06 site guard."
    - "CampaignRunAdmin.list_display excludes contact_person/contact_email -> the change-list never renders PII columns; both fields stay in the default change-form fieldset -> still editable in the detail view."
    - "admin.site.register(CampaignRun, CampaignRunAdmin) and admin.site.register(CalendarEventTelescopeLabel, CalendarEventTelescopeLabelAdmin) -> both models reachable at /admin/solsys_code/, mirroring the sibling ObservatoryAdmin registration."
---

<objective>
Fill in `solsys_code/admin.py` (currently the untouched `django-admin startproject` stub) so the
main app's `CalendarEventTelescopeLabel` and `CampaignRun` models are manageable through Django
admin, mirroring the sibling `solsys_code_observatory/admin.py` `ObservatoryAdmin` style precedent
(custom `ModelAdmin` class + `admin.site.register`).

This gives staff an admin escape hatch for data that has no other fix-it path today — hand-correcting
a mis-resolved `site` or a `window_start`/`window_end` range after submission — while preserving two
hard invariants: `approval_status` must stay read-only in admin (its normal transition triggers
calendar-projection side effects and the D-06 clobber guard that live entirely in
`CampaignRunDecisionView.post()`, not the model), and contact PII (`contact_person`/`contact_email`)
must never be scannable across rows in the admin change-list.

Purpose: close the admin-registration gap without opening a side-effect-free approval bypass or a PII
list-view leak.
Output: a populated `solsys_code/admin.py` and a `solsys_code/tests/test_admin.py` that proves the
three load-bearing constraints via the admin test client (not by eyeballing the class definitions).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@.planning/quick/260714-jpd-add-calendareventtelescopelabel-and-camp/260714-jpd-CONTEXT.md

# Style precedent to mirror (custom ModelAdmin + admin.site.register):
@solsys_code/solsys_code_observatory/admin.py

# Actual field names/types for list_display/list_filter/search_fields/readonly_fields:
@solsys_code/models.py

# Why approval_status must stay read-only (side effects live here, not on the model):
@solsys_code/campaign_views.py

# Fixture/style precedent for the admin test (TargetList + CampaignRun fixtures, Client, reverse):
@solsys_code/tests/test_campaign_approval.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Register CampaignRun and CalendarEventTelescopeLabel in solsys_code/admin.py</name>
  <files>solsys_code/admin.py</files>
  <action>
Replace the stub `solsys_code/admin.py` with two custom `ModelAdmin` classes plus their
`admin.site.register(...)` calls, mirroring the sibling `ObservatoryAdmin` shape
(`class XAdmin(admin.ModelAdmin):  # noqa: D101` then `admin.site.register(Model, XAdmin)`).
Import both models from `solsys_code.models` (`CalendarEventTelescopeLabel`, `CampaignRun`) and
`from django.contrib import admin`. Single quotes, 120-col, ruff-clean.

CampaignRunAdmin (all field names verified against solsys_code/models.py):
- `list_display = ['pk', 'campaign', 'telescope_instrument', 'approval_status', 'run_status', 'site', 'window_start', 'window_end']`
  (triage-forward ordering, per CONTEXT decision — mirrors the approval queue's status-forward column ordering).
- `list_filter = ['approval_status', 'run_status', 'campaign']`.
- `search_fields = ['telescope_instrument', 'site_raw', 'contact_person']`.
- `readonly_fields = ['approval_status']` — use Django's standard `ModelAdmin.readonly_fields`
  mechanism (NOT `exclude`): the field must still be VISIBLE in the change form (showing its current
  value) but non-editable, so admin cannot flip a run to APPROVED and bypass the calendar-projection
  side effects + D-06 `if run.site is None` guard that live only in `CampaignRunDecisionView.post()`.
  Add a short block comment on `readonly_fields` explaining that approval transitions must go through
  the real approval-queue flow (name `CampaignRunDecisionView.post()` as the reason).
- Do NOT add `contact_person`/`contact_email` to `list_display` — they stay out of the change-list so
  PII is never scannable across rows (extends the VIEW-03/VIEW-05 PII-gating discipline to admin).
  Do NOT add them to `readonly_fields` either — they must remain editable in the detail view so staff
  can correct a typo (the stricter read-only-in-detail option was explicitly rejected in CONTEXT).
- Leave `window_start`, `window_end`, `site`, and `run_status` editable (default admin behavior) —
  these are the gaps this task exists to close; `run_status` has no known side effects (Claude's
  Discretion: default editable). Do not set `exclude`/`fields`/`fieldsets` — the remaining non-PII,
  non-status fields keep Django admin's default editable behavior per CONTEXT.

CalendarEventTelescopeLabelAdmin (small OneToOneField sidecar on tom_calendar.CalendarEvent — fields
verified: `event` = OneToOneField primary_key, `is_verified` = BooleanField):
- Register standalone (its own ModelAdmin + `admin.site.register`), NOT as an inline — no local admin
  for tom_calendar.CalendarEvent exists to inline it onto.
- `list_display = ['event', 'is_verified']`.
- `list_filter = ['is_verified']`.
- `search_fields = ['event__title']` — CalendarEvent has a `title` field (used in the model's
  `__str__` and in campaign_views' calendar projection), so this related lookup is valid.
Keep it minimal and sensible; do not invent extra config.
  </action>
  <verify>
    <automated>ruff check solsys_code/admin.py && ruff format --check solsys_code/admin.py && ./manage.py check</automated>
  </verify>
  <done>solsys_code/admin.py registers both models with the field lists above; `ruff check`/`ruff format --check` are clean and `./manage.py check` (which runs Django's admin E10x system checks over list_display/list_filter/readonly_fields) reports no issues.</done>
</task>

<task type="auto">
  <name>Task 2: Add solsys_code/tests/test_admin.py verifying read-only approval_status, PII gating, and registration</name>
  <files>solsys_code/tests/test_admin.py</files>
  <action>
Create a new Django `TestCase` (runs under `./manage.py test solsys_code`, DB-dependent) that
exercises the admin through the test client — this is the required proof of the three load-bearing
constraints, not a class-definition eyeball. Import only lightweight modules (django test `TestCase`,
`django.contrib.auth.models.User`, `django.urls.reverse`, `solsys_code.models.CampaignRun`,
`tom_targets.models.TargetList`); do NOT import `solsys_code.views`/`ephem_utils` (heavy SPICE import,
per CLAUDE.md) — none are needed here.

Fixtures (mirror test_campaign_approval.py style): create a superuser via
`User.objects.create_superuser(...)` and `self.client.force_login(...)`; create a campaign via
`TargetList.objects.create(name=...)`; create a CampaignRun via `CampaignRun.objects.create(...)` with
`campaign=<the TargetList>`, `telescope_instrument='LCO-1m-Sinistro'`, and DISTINCTIVE PII sentinels
`contact_person='Zztestcontact'`, `contact_email='pii-secret@example.test'` so the PII assertions are
unambiguous. (Required fields are only `campaign` and `telescope_instrument`; `approval_status`
defaults to PENDING_REVIEW.)

Test cases (use `reverse` with the admin URL names):
1. Change-lists load: GET `reverse('admin:solsys_code_campaignrun_changelist')` and
   GET `reverse('admin:solsys_code_calendareventtelescopelabel_changelist')` each return HTTP 200.
2. Search path valid: GET the CalendarEventTelescopeLabel changelist with `?q=anything` returns 200
   (proves `search_fields = ['event__title']` resolves without FieldError even with zero rows).
3. approval_status read-only in detail: GET
   `reverse('admin:solsys_code_campaignrun_change', args=[run.pk])` returns 200; assert the displayed
   value 'Pending Review' IS present (field still visible) AND `'name="approval_status"'` is NOT in the
   rendered HTML (no editable widget — Django renders a readonly field as a static div, not an input/
   select). This is the anti-approval-bypass guarantee.
4. contact fields editable in detail: on that same change-page HTML, assert `'name="contact_person"'`
   and `'name="contact_email"'` ARE present (still editable so staff can fix a typo).
5. PII gated in list view: GET the CampaignRun changelist (200) and assert the sentinel strings
   `'Zztestcontact'` and `'pii-secret@example.test'` are NOT in the response body (contact PII never
   rendered as a change-list column), while `'LCO-1m-Sinistro'` IS present (confirms the row rendered,
   so the absence of PII is real gating, not an empty page).

Give the test class/methods clear names; test files are exempt from D101/D102 docstring rules.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_admin -v 2</automated>
  </verify>
  <done>`./manage.py test solsys_code.tests.test_admin` passes: both change-lists return 200, the CampaignRun change page shows 'Pending Review' with no editable approval_status widget, contact_person/contact_email are editable in the detail view but absent from the change-list, and the CalendarEventTelescopeLabel search path resolves without error.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| staff admin user -> CampaignRun state | A privileged staff user acts on approval state + PII through the Django admin change form, bypassing the app's own CampaignRunDecisionView flow. |
| admin change-list -> viewer | Row data (potentially many rows) is rendered at once to any admin viewer; contact PII must not be among the rendered columns. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-jpd-01 | Elevation of Privilege / Tampering | CampaignRunAdmin change form (approval_status) | high | mitigate | `readonly_fields = ['approval_status']` — admin can view but never edit approval_status, so it cannot flip a run to APPROVED bypassing CampaignRunDecisionView.post()'s calendar projection + D-06 site guard. Verified by Task 2 case 3. |
| T-jpd-02 | Information Disclosure | CampaignRunAdmin change-list (contact_person/contact_email) | medium | mitigate | Exclude both contact fields from `list_display` so PII is never scannable across rows; they remain editable only in the single-record detail view. Verified by Task 2 case 5. |
| T-jpd-03 | Information Disclosure | Django admin access control | low | accept | Admin is already gated by Django's staff/superuser auth + per-model permissions; this task adds registrations under that existing gate and introduces no new anonymous surface. |

No package-manager installs in this task — no supply-chain (T-jpd-SC) threat applies.
</threat_model>

<verification>
- `ruff check solsys_code/admin.py` and `ruff format --check solsys_code/admin.py` clean.
- `./manage.py check` reports no admin system-check (E10x) errors for the new ModelAdmins.
- `./manage.py test solsys_code.tests.test_admin` passes all cases.
- Manual sanity (optional): `./manage.py runserver`, visit `/admin/solsys_code/campaignrun/<pk>/change/`
  and confirm approval_status shows its value with no editable control, while site/window_start/
  window_end/run_status are editable.
</verification>

<success_criteria>
- Both models are registered and reachable under `/admin/solsys_code/` (change-lists return 200).
- `approval_status` is read-only (visible, non-editable) in the CampaignRun change form.
- `contact_person`/`contact_email` are absent from the change-list but editable in the detail view.
- CampaignRun list_display/list_filter/search_fields match the CONTEXT decisions exactly.
- CalendarEventTelescopeLabel is registered standalone with a minimal `event` + `is_verified` config
  and a valid `event__title` search path.
- All quality gates (ruff, `manage.py check`, the new admin test) pass.
</success_criteria>

<output>
Create `.planning/quick/260714-jpd-add-calendareventtelescopelabel-and-camp/260714-jpd-SUMMARY.md` when done.
</output>
