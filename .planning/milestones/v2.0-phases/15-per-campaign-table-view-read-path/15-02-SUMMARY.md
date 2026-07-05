---
phase: 15-per-campaign-table-view-read-path
plan: 02
subsystem: ui
tags: [tom-toolkit, appconfig-hooks, navbar, target-detail, django-templatetags]

requires:
  - phase: 15-01
    provides: campaigns:list / campaigns:table URL names (CampaignListView, CampaignRunTableView)
provides:
  - Per-campaign "View {campaign.name} Runs" link(s) on the target-detail page, one per matching campaign (D-01/D-02)
  - Navbar "Campaigns" entry on every page via a new AppConfig.nav_items() hook (D-03, first nav_items() consumer in FOMO)
  - Integration tests proving VIEW-02 (member-target link, non-member-target absence, navbar presence)
affects: [16-submission-form-approval-queue-calendar-projection]

tech-stack:
  added: []
  patterns:
    - "AppConfig.nav_items() hook -- first consumer in this codebase; all future navbar entries should follow the same static-context inclusion-tag shape"
    - "target_detail_buttons() context method receives the full render context (context['target'] already set) -- a single dict entry + a loop in its partial satisfies 'one link per matching campaign' (D-02) without multiple dict entries"

key-files:
  created:
    - src/templates/solsys_code/partials/campaign_links.html
    - src/templates/solsys_code/partials/campaigns_nav_link.html
  modified:
    - solsys_code/apps.py
    - src/templatetags/solsys_code_extras.py
    - solsys_code/tests/test_campaign_views.py

key-decisions:
  - "reverse('tom_targets:detail', ...) resolves correctly via Django's application-namespace fallback (tom_targets/urls.py sets app_name='tom_targets') even though src/fomo/urls.py -> tom_common.urls registers the actual instance namespace as 'targets' -- confirmed live (both reverse to the same /targets/<pk>/ URL) rather than assumed, per the plan's explicit verification instruction; no adjustment was needed."
  - "campaign_links.html applies 'mr-2 mb-2' unconditionally to every link, not only when 2+ campaigns are present (as UI-SPEC's wrapping-div language for the 2+ case might suggest) -- simpler, harmless for the single-link case (matches the pre-existing 'Make Ephemeris' button visually), and matches the plan's literal Task 2 action text verbatim."

patterns-established:
  - "Pattern: an AppConfig hook's inclusion-tag context method returns {} when it needs no per-request data (campaigns_nav_link) -- the surrounding page context (via 'django.template.context_processors.request') still supplies 'request' to the rendered partial for active-nav-link checks."

requirements-completed: [VIEW-02]

coverage:
  - id: D1
    description: "Target-detail page for a Target that belongs to a campaign shows a 'View {campaign.name} Runs' link, linking to that campaign's table"
    requirement: "VIEW-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignDetailIntegration.test_target_detail_shows_campaign_link"
        status: pass
    human_judgment: false
  - id: D2
    description: "Target-detail page for a Target in zero campaigns shows no campaign link (empty partial output, no placeholder)"
    requirement: "VIEW-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignDetailIntegration.test_target_detail_no_campaign_for_unrelated_target"
        status: pass
    human_judgment: false
  - id: D3
    description: "Campaign discovery is via TargetList membership, not CampaignRun's target FK -- proven by a fixture where the member target is never set as any CampaignRun's target"
    requirement: "VIEW-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignDetailIntegration.test_target_detail_shows_campaign_link"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every page's navbar shows a single-word 'Campaigns' entry linking to campaigns:list"
    requirement: "VIEW-02"
    verification:
      - kind: integration
        ref: "solsys_code/tests/test_campaign_views.py#TestCampaignDetailIntegration.test_navbar_shows_campaigns_entry"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-07-03
status: complete
---

# Phase 15 Plan 02: Campaign Navigation Integration (Target Detail + Navbar) Summary

**Per-campaign "View {name} Runs" links on target-detail pages via a second `target_detail_buttons()` entry, plus FOMO's first `AppConfig.nav_items()` navbar hook for a global "Campaigns" entry -- completing VIEW-02.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-07-03T16:03:00Z (approx, immediately following Plan 01's 16:02:39Z completion)
- **Completed:** 2026-07-03T16:18:18Z
- **Tasks:** 3
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments
- `SolsysCodeConfig.target_detail_buttons()` gains a second entry (`campaign_links.html`) rendering one "View {campaign.name} Runs" button per campaign the rendered target belongs to
- `SolsysCodeConfig.nav_items()` -- FOMO's first `nav_items()` consumer -- adds a global navbar "Campaigns" entry linking to `campaigns:list`
- `campaign_links`/`campaigns_nav_link` inclusion tags added to `solsys_code_extras.py`; campaign discovery strictly via `TargetList` membership (D-01), proven even when `CampaignRun.target` is never populated for any row
- Two new partials (`campaign_links.html`, `campaigns_nav_link.html`); `module_buttons.html` required zero changes (confirmed via `git diff --stat` -- falls through to the existing generic `show_individual_app_partial` branch)
- `TestCampaignDetailIntegration`: 3 new integration tests proving VIEW-02 end-to-end; 258/258 `solsys_code` tests green (255 + 3 new), `ruff check`/`ruff format --check` clean on every file this plan touched

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend apps.py hooks + add campaign_links / campaigns_nav_link inclusion tags** - `11f6194` (feat)
2. **Task 2: Create campaign_links.html and campaigns_nav_link.html partials** - `7cb6b39` (feat)
3. **Task 3: Integration tests for VIEW-02 (target-detail links + navbar entry)** - `b205c09` (test)

**Plan metadata:** (this commit, following SUMMARY.md write)

## Files Created/Modified
- `solsys_code/apps.py` - second `target_detail_buttons()` entry (`campaign_links`) + new `nav_items()` method
- `src/templatetags/solsys_code_extras.py` - `campaign_links`/`campaigns_nav_link` inclusion tags
- `src/templates/solsys_code/partials/campaign_links.html` - one "View {name} Runs" `btn-info` link per matching campaign
- `src/templates/solsys_code/partials/campaigns_nav_link.html` - single-word "Campaigns" navbar entry
- `solsys_code/tests/test_campaign_views.py` - `TestCampaignDetailIntegration` (3 new tests)

## Decisions Made
- `reverse('tom_targets:detail', ...)` resolves correctly via Django's application-namespace fallback (`tom_targets/urls.py` sets `app_name = 'tom_targets'`) even though `src/fomo/urls.py` (via `tom_common.urls`) registers the actual instance namespace as `'targets'` -- confirmed live in a shell (`reverse('targets:detail', ...)` and `reverse('tom_targets:detail', ...)` both resolve to the same `/targets/<pk>/` URL) rather than assumed, per the plan's explicit "confirm ... if the reverse fails" instruction. No adjustment was needed; the plan's literal `tom_targets:detail` name works as written.
- `campaign_links.html` applies `mr-2 mb-2` unconditionally to every rendered link, not only when 2+ campaigns are present (as UI-SPEC's "wrap in `<div class="d-flex flex-wrap">`" language for the 2+ case might otherwise suggest). This is simpler, harmless for the single-link case (visually identical to the pre-existing "Make Ephemeris" button, which sits in the same button row), and matches the plan's own Task 2 `<action>` text verbatim (which already reconciles this simplification against UI-SPEC).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Running the plan's own overall `ruff check .` / `ruff format --check .` verification step (repo-wide, not per-file) surfaced pre-existing issues in 7 files: `docs/notebooks/pre_executed/{load_telescope_runs_demo,sync_gemini_observation_calendar_demo,sync_lco_observation_calendar_demo,import_campaign_csv_demo}.ipynb`, `src/fomo/settings.py`, and two Phase-5-era scratch scripts under `.planning/quick/260619-f7u-.../`. Confirmed via `git log --oneline -1 -- <file>` and `git show --stat` on this plan's 3 commits that none of these files were touched by this plan (all last modified by commits `adc5a61`/`9ca8a29`/`bc5bfdf`, predating Phase 15). Per the executor's scope boundary, these are out-of-scope pre-existing repo debt -- logged to `.planning/phases/15-per-campaign-table-view-read-path/deferred-items.md`, not fixed here. All 5 files this plan actually created/modified individually pass `ruff check`/`ruff format --check`.

## Known Stubs
None -- `campaign_links.html`/`campaigns_nav_link.html` render real data (queried campaigns / static link), no hardcoded empty/placeholder values.

## Threat Flags
None -- the implementation matches the threat model's T-15-04 mitigation exactly (`TargetList.objects.filter(targets=target, campaign_runs__isnull=False).distinct()`, scoped to the specific rendered target); no new unmodeled surface was introduced. T-15-05 (anonymous read access) is the same accepted-by-design D-04 posture as Plan 01.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- VIEW-02 complete. Phase 15 (VIEW-01..04) is now fully implemented across both plans: Plan 01 shipped the read path (table/list/PII-gating/filtering), Plan 02 shipped navigation (target-detail links + navbar). Ready for phase-level verification (`/gsd-verify-work`).
- Phase 16 (submission form + approval queue + calendar projection) can proceed without further Phase 15 work -- it can rely on `campaigns:table`/`campaigns:list` URLs, `CampaignRunTableView`/`CampaignRunTable`, and the target-detail/navbar discoverability entry points shipped here.
- No blockers.

---
*Phase: 15-per-campaign-table-view-read-path*
*Completed: 2026-07-03*

## Self-Check: PASSED

All 5 plan files (`solsys_code/apps.py`, `src/templatetags/solsys_code_extras.py`,
`src/templates/solsys_code/partials/campaign_links.html`,
`src/templates/solsys_code/partials/campaigns_nav_link.html`,
`solsys_code/tests/test_campaign_views.py`) plus this SUMMARY.md and
`deferred-items.md` confirmed present on disk. All 3 task commits (`11f6194`,
`7cb6b39`, `b205c09`) confirmed present in `git log`.
