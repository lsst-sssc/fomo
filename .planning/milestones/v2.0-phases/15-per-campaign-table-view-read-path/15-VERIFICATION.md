---
phase: 15-per-campaign-table-view-read-path
verified: 2026-07-03T00:00:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 15: Per-Campaign Table View (Read Path) Verification Report

**Phase Goal:** A coordinator can see every run for a campaign in one sortable, filterable table
that replaces the shared spreadsheet — with contact details visible only to staff.
**Verified:** 2026-07-03
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can view a per-campaign table listing all of its runs, sortable and paginated | ✓ VERIFIED | `CampaignRunTableView` (`solsys_code/campaign_views.py:47-80`) + `CampaignRunTable` (`solsys_code/campaign_tables.py`, `Meta.order_by = ('-obs_date',)`, `table_pagination={'per_page': 25}`). Executed `python manage.py test solsys_code.tests.test_campaign_views.TestCampaignRunTableView` — 4/4 pass, including `test_first_page_shows_25_rows_and_second_page_exists` (30-row fixture, 25/page, 2 pages) and `test_default_sort_is_obs_date_descending`. |
| 2 | User can reach a campaign's table from the relevant target-detail page, and a navbar entry exposes campaigns | ✓ VERIFIED | `SolsysCodeConfig.target_detail_buttons()` second entry + new `nav_items()` (`solsys_code/apps.py:17-33`); `campaign_links`/`campaigns_nav_link` inclusion tags (`src/templatetags/solsys_code_extras.py`) query via `TargetList` membership. Executed `TestCampaignDetailIntegration` — 3/3 pass: link present on member target's detail page, absent on non-member's, navbar "Campaigns" entry present on every page. |
| 3 | Contact person/email are excluded from view context for anonymous requests (proven by an anonymous-client test) and shown only to authenticated staff | ✓ VERIFIED | `CampaignRunTableView.get_queryset()` returns `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` for non-staff — `ALLOWED_FIELDS_FOR_NON_STAFF` (`campaign_views.py:26-44`) omits `contact_person`/`contact_email` from the SQL SELECT itself; `get_table_kwargs()` adds `exclude=('contact_person','contact_email')` as defense-in-depth. Executed `TestContactFieldGating` — 4/4 pass: anonymous context rows are dicts missing both keys, anonymous rendered content lacks the seeded contact strings, staff context/content both include them. |
| 4 | User can filter the table by lifecycle status and by the open-to-collaboration flag | ✓ VERIFIED | `CampaignRunFilterSet.run_status` is an explicit `MultipleChoiceFilter` (OR semantics) with `CheckboxSelectMultiple`; `open_to_collaboration` is an auto-generated `BooleanFilter` (`solsys_code/campaign_filters.py`). Executed `TestCampaignRunFilterSet` — 3/3 pass: unfiltered default returns all 30 rows, `?run_status=planned&run_status=observed` returns exactly the 2 matching rows (OR), `?open_to_collaboration=true` returns exactly the 1 seeded row. |

### Observable Truths (Plan frontmatter must_haves, additive)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 5 | The table shows all rows regardless of `approval_status` for both staff and anonymous — no filtering to approved-only | ✓ VERIFIED | No `approval_status` filter exists anywhere in `campaign_views.py`/`campaign_filters.py`; `test_default_load_shows_every_seeded_run_status_value` confirms every seeded `run_status` value appears unfiltered. |
| 6 | `approval_status` renders as a colored Bootstrap badge on every row (D-08); `run_status` renders as a muted badge | ✓ VERIFIED | `render_approval_status`/`render_run_status` (`campaign_tables.py:82-107`) emit `format_html('<span class="badge {}"...')` with fixed `APPROVAL_BADGE_CLASSES`/`RUN_STATUS_BADGE_CLASSES` lookups. Confirmed by an ad-hoc one-off spot-check test executed during this verification (created, run, and deleted — no repo state change): a `REJECTED`/`CANCELLED` row's rendered HTML contains both `badge-danger` and `badge-light`. Visual contrast/legibility itself is explicitly deferred to a future UI-review pass per 15-01-SUMMARY.md and 15-VALIDATION.md's Manual-Only Verifications table — this is a cosmetic quality judgment, not a functional gap, and does not block the phase goal. |
| 7 | `GET /campaigns/` lists every `TargetList` with >=1 `CampaignRun`, each linking to its per-campaign table (D-03), and never lists a `TargetList` with zero runs | ✓ VERIFIED | `CampaignListView.queryset` filters `campaign_runs__isnull=False` (`campaign_views.py:91-93`), never `.objects.all()`. `TestCampaignListView` — 2/2 pass. |
| 8 | Neither view has a login/staff check gating access itself — anonymous visitors can reach both pages; only contact fields are staff-gated | ✓ VERIFIED | No `LoginRequiredMixin`/`staff_member_required`/`permission_required` anywhere in `campaign_views.py`; `test_anonymous_get_returns_200` confirms 200 for anonymous. |
| 9 | No contact/reach-out path added for anonymous visitors interested in an `open_to_collaboration` run (VIEW-05 explicitly out of scope) | ✓ VERIFIED | Grep of all phase-touched files for a reach-out/contact-request form or mailto action found none; `open_to_collaboration` is rendered as a read-only yes/no icon column only. |

**Score:** 9/9 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `solsys_code/campaign_tables.py` | `CampaignRunTable` with badge render methods | ✓ VERIFIED | Exists, substantive, wired into `campaign_views.py` |
| `solsys_code/campaign_filters.py` | `CampaignRunFilterSet` multi-select | ✓ VERIFIED | Exists, substantive, wired |
| `solsys_code/campaign_views.py` | `CampaignRunTableView`, `CampaignListView` | ✓ VERIFIED | Exists, substantive, wired into `campaign_urls.py` |
| `solsys_code/campaign_urls.py` | `campaigns` namespace | ✓ VERIFIED | Exists, wired into `src/fomo/urls.py:26` before the `tom_common.urls` catch-all |
| `src/templates/campaigns/campaign_list.html` | Campaigns list page | ✓ VERIFIED | Renders `{% for campaign in campaigns %}` list with run-count badge and D-03 empty state |
| `src/templates/campaigns/campaignrun_table.html` | Table + filter panel | ✓ VERIFIED | Renders `{% render_table table %}`, filter form (checkboxes for `run_status`, select for `open_to_collaboration`) |
| `solsys_code/tests/test_campaign_views.py` | Tests proving VIEW-01/02/03/04 | ✓ VERIFIED | 16 tests across 5 classes, all pass (executed live, not just claimed) |
| `src/templates/solsys_code/partials/campaign_links.html` | Per-campaign target-detail link(s) | ✓ VERIFIED | Loops over `campaigns`, emits one "View {name} Runs" link each |
| `src/templates/solsys_code/partials/campaigns_nav_link.html` | Navbar "Campaigns" entry | ✓ VERIFIED | Static `<li>` with active-nav check, reverses `campaigns:list` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `src/fomo/urls.py` | `solsys_code/campaign_urls.py` | `include(..., namespace='campaigns')` before catch-all | ✓ WIRED | Confirmed at `src/fomo/urls.py:26` |
| `CampaignRunTableView.get_queryset` | SQL SELECT | `.values(*ALLOWED_FIELDS_FOR_NON_STAFF)` for non-staff | ✓ WIRED | `contact_person`/`contact_email` absent from the list; proven by `TestContactFieldGating` |
| `CampaignRunTable.render_*` methods | `record` (dict or model instance) | `Accessor(...).resolve(record)` | ✓ WIRED | Works identically for staff (model) and anonymous (dict) rows — proven by both staff and anonymous test paths passing |
| `apps.py target_detail_buttons()` | `campaign_links` inclusion tag | `'context': 'src.templatetags.solsys_code_extras.campaign_links'` | ✓ WIRED | Confirmed via source read + `test_target_detail_shows_campaign_link` |
| `apps.py nav_items()` | `campaigns_nav_link` inclusion tag | same pattern | ✓ WIRED | Confirmed via source read + `test_navbar_shows_campaigns_entry` |
| `campaign_links()` tag | `TargetList` membership | `TargetList.objects.filter(targets=target, campaign_runs__isnull=False)` | ✓ WIRED | Never via `CampaignRun.target` (grep confirms 0 occurrences); proven by non-member-target test |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full `test_campaign_views` suite (16 tests) | `python manage.py test solsys_code.tests.test_campaign_views` | 16/16 pass | ✓ PASS |
| Full `solsys_code` app suite (regression check) | `python manage.py test solsys_code` | 258/258 pass | ✓ PASS |
| Badge markup renders with correct CSS classes | ad-hoc one-off test (created, run, deleted during verification) | `badge-danger` present for REJECTED, `badge-light` present for CANCELLED | ✓ PASS |
| `ruff check` on all phase-touched Python files | `ruff check <files>` | All checks passed | ✓ PASS |
| `ruff format --check` on all phase-touched Python files | `ruff format --check <files>` | 8 files already formatted | ✓ PASS |
| Commits referenced in SUMMARY.md exist in git history | `git cat-file -e <hash>` for all 6 hashes | all 6 present | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| VIEW-01 | 15-01 | Per-campaign table, sortable/paginated | ✓ SATISFIED | `TestCampaignRunTableView` (4/4 pass) |
| VIEW-02 | 15-02 | Target-detail link + navbar entry | ✓ SATISFIED | `TestCampaignDetailIntegration` (3/3 pass) |
| VIEW-03 | 15-01 | Contact PII staff-only | ✓ SATISFIED | `TestContactFieldGating` (4/4 pass) |
| VIEW-04 | 15-01 | Filter by lifecycle status + collaboration flag | ✓ SATISFIED | `TestCampaignRunFilterSet` (3/3 pass) |

No orphaned requirements — REQUIREMENTS.md maps exactly VIEW-01..04 to Phase 15, and both plans' `requirements:` frontmatter together cover all four with no gaps or overlaps.

### Anti-Patterns Found

No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers, no "coming soon"/"not yet implemented" strings, and no empty-implementation stubs found in any of the 12 phase-touched files.

Carried forward from `15-REVIEW.md` (deep code review, `issues_found`: 0 critical / 2 warnings / 3 info) — none rise to blocker level for this phase's goal:

- **WR-01** (warning): `CampaignRunTable.Meta.order_by = ('-obs_date',)` has no tiebreaker; under PostgreSQL (the documented production target per CLAUDE.md), ties on `obs_date` could cause unstable pagination (a row could be duplicated or skipped across pages). Currently masked by SQLite's incidental rowid fallback in dev/test. This is a real latent correctness risk for the "listing all of its runs" guarantee once the project migrates off SQLite, but does not affect current observable behavior. Recommend a follow-up quick task to add `-pk` as a tiebreaker (`order_by = ('-obs_date', '-pk')`).
- **WR-02** (warning): No regression test proves PII-excluded columns can't be reached via `?sort=contact_person` tampering. The underlying security boundary (`.values()` restriction) is unaffected by sort params, but the safety is emergent from two independently-maintained mechanisms rather than locked in by a test.
- **IN-01/IN-02/IN-03** (info): duplicated `is_staff` check, `CampaignRunTableView` doesn't 404 non-campaign `TargetList` pks, `open_to_collaboration=false` filter path untested. None are must-haves for this phase's goal.

These do not block phase sign-off (the review explicitly found 0 critical issues and confirmed the two must-have security properties hold), but are worth tracking as follow-up work, particularly WR-01 given the explicit Postgres migration path noted in CLAUDE.md's Architectural Constraints.

### Human Verification Required

None. All roadmap success criteria and plan must-haves are proven by automated tests executed live during this verification (not merely claimed in SUMMARY.md), and all touched code passes `ruff check`/`ruff format --check`. The one item flagged as `human_judgment: true` in 15-01-SUMMARY.md (visual badge color/contrast legibility) is a cosmetic UI-review item explicitly deferred by design (documented in 15-VALIDATION.md's Manual-Only Verifications table as out of scope for this phase's functional test suite) — the functional truth ("renders as a colored/muted badge") is independently proven by source inspection and a live spot-check, so it does not block `passed` status.

### Gaps Summary

No gaps found. Both plans' commits exist in git history, all 16 phase-specific tests pass, the full 258-test `solsys_code` suite is green (no regressions), all four requirement IDs (VIEW-01..04) trace to passing tests, PII gating is enforced at the SQL layer (not just template hiding) and proven by an anonymous-client test, and navigation wiring (target-detail button + navbar entry) is proven end-to-end. The phase goal — "a coordinator can see every run for a campaign in one sortable, filterable table that replaces the shared spreadsheet, with contact details visible only to staff" — is achieved in the codebase.

---

*Verified: 2026-07-03*
*Verifier: Claude (gsd-verifier)*
