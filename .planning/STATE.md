---
gsd_state_version: "1.0"
milestone: v2.5
milestone_name: Main Sync & Consolidation
current_phase: 39
current_phase_name: Calendar Write Access
status: executing
stopped_at: Completed 39-05-PLAN.md
last_updated: "2026-10-08T22:55:54.478Z"
last_activity: 2026-10-08
last_activity_desc: Phase 39 execution started
state_head: ba8a53a4075905eaaddbf7a1c195d743fcdbf790
progress:
  total_phases: 5
  completed_phases: 40
  total_plans: 12
  completed_plans: 12
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-08 — Phase 38 complete)

**Core value:** The `issue37-telescope-runs-calendar` branch is back in step with `main` — same dependency floors, same tooling, same CI runner — and the debt v2.4 carried forward is either fixed or has a written decision, so the next feature milestone starts from a current, clean base.
**Current focus:** Phase 39 — Calendar Write Access

## Current Position

Phase: 39 (Calendar Write Access) — EXECUTING
Plan: 2 of 5
Status: Ready to execute
Last activity: 2026-10-08 — Phase 39 execution started

Progress: [████████████████████] 7/7 plans ([██████████] 100%)

## Performance Metrics

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| - | - | - | - |
| Phase 38 P01 | 35min | 3 tasks | 9 files |
| Phase 38 P02 | 8 min | 3 tasks | 31 files |
| Phase 38 P03 | 21 min | 3 tasks | 1 files |
| Phase 38 P04 | 32 min | 3 tasks | 1 files |
| Phase 38 P05 | 21 min | 2 tasks | 6 files |
| Phase 38 P06 | 55 min | 3 tasks | 0 files |
| Phase 38 P07 | 80min elapsed | 3 tasks | 3 files |
| Phase 39 P01 | 34 min | 2 tasks | 5 files |
| Phase 39 P02 | 36 min | 3 tasks | 7 files |
| Phase 39 P03 | 45 min | 2 tasks | 3 files |
| Phase 39 P04 | 2h | 3 tasks | 9 files |
| Phase 39 P05 | 30 min | 2 tasks | 6 files |

*v2.4 per-plan timings are in the v2.4 phase summaries under `.planning/milestones/v2.4-phases/`.*

## Accumulated Context

### Decisions

Full decision log: `.planning/PROJECT.md` (Key Decisions). Roadmap decisions for v2.5:

- [Roadmap]: Order is 38 sync → 39 calendar write access → 40 notebooks + attribution page → 41 triage → (inserted 41.1 if anything is fix-now) → 42 re-verify. Sync first and re-verify last are the developer's decisions.
- [Roadmap]: ACCESS-01/02 share Phase 39 with WARN-01 (all three are FOMO's `tom_calendar` overrides, compared against tomtoolkit 3.1.0's upstream copies). WARN-04/07 sit with the notebook work in Phase 40 because `campaign_lifecycle_demo.ipynb` is both a byte-copier (WARN-05) and the attribution page's paired notebook, so it is rebuilt and re-executed once.
- [Roadmap]: Five phases under `granularity: coarse` — four are forced by the ordering constraints; the fifth keeps the security gate (Phase 39) verifiable on its own.
- [Phase 38]: 38-01: merge landed with SKIP=django-test,ruff,ruff-format so no reformat rides in the merge commit (D-02); developer approved staged resolution at Task 2
- [Phase 38]: 38-01: run_unattended cron line stays paused (restore: crontab $HOME/tmp/phase38-crontab.bak) until 38-03 restores it after the database migrate
- [Phase 38]: 38-02: SIM103 fixed in code (not by widening ruff ignores); smoke-test.yml gets the same ephemeris_segfault exclusion as the CI matrix; codebase maps edited alongside CLAUDE.md
- [Phase 38]: 38-03: no upstream file FOMO shadows changed between tomtoolkit 3.0.1 and 3.1.0; no SYNC-07 fix from the override comparison
- [Phase 38]: 38-03: fresh venv $HOME/venv/fomo_phase38_fresh kept for /gsd-verify-work (tomtoolkit 3.1.0, tom_jpl 0.3.0, full suite 2178 OK); dev DB migrated, backup src/fomo_db_20261007_pre_phase38.sqlite3; cron restored 18:29Z
- [Phase 38]: 38-04: PR #43 refreshed by publishing 372d02c, a merge -s ours of origin/main and a tree snapshot on issue37-code-only (plain fast-forward pushes); developer answered publish; CI (Django runner, pre-commit, docs) green on the snapshot
- [Phase 38]: 38-05: took main's side for the alerts/ include (deleted the four lines, ada2000) instead of re-adding tom_alerts to INSTALLED_APPS; D-03 not reopened
- [Phase 38]: 38-06: developer approved publish; issue37-code-only re-snapshotted (846be34) without the alerts/ route, plain fast-forward pushes, PR #43 still a draft, CI green on 3.10-3.12
- [Phase 38]: 38-07: G-38-1 closed by documentation only; src/fomo/local_settings.py is the canonical location and the wording names both old locations (repo root and src/) and main's PR #58
- [Phase 39]: 39-01: any logged-in user may write the calendar (D-01); five routes guarded at FOMO's URL conf, require_POST on delete-event/create-todo/update-todo, login next=/calendar/
- [Phase 39]: 39-01: CSRF failure on a signed-in calendar POST surfaces as 302 to login (tom_common Raise403Middleware), not 403; nothing is written
- [Phase 39]: 39-02: read-only visitor card lives inside event_form.html (one file, series/campaign blocks render once); a non-web URL value is never echoed on the anonymous card; todos stay readable to visitors
- [Phase 39]: 39-03: edit step counts form[hx-post*='/calendar/update/'] because the saved-event pop-up also holds upstream's add-a-todo form
- [Phase 39]: 39-04: gap 1 closed by docs and tests, not a CSRF_FAILURE_VIEW; CR-01 open self-registration recorded as an accepted risk (Tim Lister 2026-10-08); upstream Save and Edit label restored with a pinned body diff snapshot
- [Phase 39]: wave 3 ui.safety-gate block ("UI files changed, no UI-SPEC.md") overridden by the developer (2026-10-08): the only UI change is 65ba57c restoring a one-word button label, and 39-UI-REVIEW.md already audits the phase against the abstract standards
- [Phase 39]: 39-05: G-39-4 fixed minimally with an isinstance guard on high_band_attribution_candidates plus an action-first gate on the staff hint in event_form.html; header item 4 and the pinned snapshot updated in the same commit
- [Phase 39]: 39-05: G-39-3 is runbook-only; the template method=post hardening (39-REVIEW WR-01 item 2) was offered at UAT and not requested. campaign_attribution.candidates_for_event missing guard surfaced for Phase 41 todo triage

### Pending Todos

- [2026-09-01] [general] Add TTL cache to attribution banner count — [todo file](.planning/todos/pending/2026-09-01-add-ttl-cache-to-attribution-banner-count.md)
- [2026-09-01] [general] Guard attribution dismiss action with is_offered_candidate — [todo file](.planning/todos/pending/2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md)
- [2026-09-01] [general] Skip sun_event computation for already-existing reconciler nights — [todo file](.planning/todos/pending/2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md)
- [2026-09-30] [unattended-discovery] Fetch LCO observation blocks in bulk per proposal instead of one call per request — [todo file](.planning/todos/pending/2026-09-30-fetch-lco-observation-blocks-in-bulk-per-proposal.md)
- [2026-10-02] [telescope-runs] "load_telescope_runs: skip comment lines and warn on a bare proposal token" — [todo file](.planning/todos/pending/2026-10-02-load-telescope-runs-skip-comment-lines-and-warn-on-a-bare-pr.md)
- [2026-10-02] [docs] Run pre-executed demo notebooks against a scratch DB copy, never the live dev DB — [todo file](.planning/todos/pending/2026-10-02-run-pre-executed-demo-notebooks-against-a-scratch-db-copy-ne.md)
- [2026-10-07] [observation-projector] "A failed or aborted record keeps its last scheduled window instead of the original re… — [todo file](.planning/todos/pending/2026-10-07-a-failed-or-aborted-record-keeps-its-last-scheduled-window-i.md)
- [2026-10-07] [campaign-runs] "Decide whether CampaignRun.run_status needs an awarded-and-in-progress value" — [todo file](.planning/todos/pending/2026-10-07-decide-whether-campaignrun-run-status-needs-an-awarded-and-i.md)
- [2026-10-07] [campaign-reconciler] "Delete the reconciler's own stale RUN:{pk} container on a container-to-per-night re-cla… — [todo file](.planning/todos/pending/2026-10-07-delete-the-reconciler-s-own-stale-run-pk-container-on-a-cont.md)
- [2026-10-07] [campaign-runs] "Explain CampaignRun.telescope_class: setting it on a site-resolved run switches it to one con… — [todo file](.planning/todos/pending/2026-10-07-explain-campaignrun-telescope-class-setting-it-on-a-site-res.md)
- [2026-10-07] [campaign-gap] "Give the campaign gap analysis a start/end date control" — [todo file](.planning/todos/pending/2026-10-07-give-the-campaign-gap-analysis-a-start-end-date-control.md)
- [2026-10-07] [tests] "Isolate the campaign table query-count test from the shared file cache" — [todo file](.planning/todos/pending/2026-10-07-isolate-the-campaign-table-query-count-test-from-the-shared.md)
- [2026-10-07] [observation-projector] "Keep a request's site restriction and show it in the event title from the start (LCO-… — [todo file](.planning/todos/pending/2026-10-07-keep-a-request-s-site-restriction-and-show-it-in-the-event-t.md)
- [2026-10-07] [campaign-table] "Link each campaign table row to its run, or give the target its own column" — [todo file](.planning/todos/pending/2026-10-07-link-each-campaign-table-row-to-its-run-or-give-the-target-i.md)
- [2026-10-07] [observation-projector] "Mark site lookups as 'not attempted' in project_observation_calendar --dry-run output" — [todo file](.planning/todos/pending/2026-10-07-mark-site-lookups-as-not-attempted-in-project-observation-ca.md)
- [2026-10-07] [unattended-discovery] "Report system-link outcomes in the discovery step's tick summary" — [todo file](.planning/todos/pending/2026-10-07-report-system-link-outcomes-in-the-discovery-step-s-tick-sum.md)
- [2026-10-07] [allocation-projector] "Revisit 'a human-confirmed allocation night is not retired' if a real doubled night ap… — [todo file](.planning/todos/pending/2026-10-07-revisit-a-human-confirmed-allocation-night-is-not-retired-if.md)
- [2026-10-07] [unattended-discovery] "Say in WatchedProposal.attributed_to help text and the runbook that it applies only to… — [todo file](.planning/todos/pending/2026-10-07-say-in-watchedproposal-attributed-to-help-text-and-the-runbo.md)
- [2026-10-07] [campaign-tally] "Show a proposal-level unused-nights figure once per proposal, not on every run row" — [todo file](.planning/todos/pending/2026-10-07-show-a-proposal-level-unused-nights-figure-once-per-proposal.md)
- [2026-10-08] [telescope-runs] Cache telescope_runs.sun_event() and speed up the test suite — [todo file](.planning/todos/pending/2026-10-08-cache-telescope-runs-sun-event-and-speed-up-the-test-suite.md)

### Blockers/Concerns

- [Phase 39+]: PR #58 (`production-deploy` → `main`) switches `main`'s `settings.py` to `from .local_settings import *`. At the next sync with `main`, keep the branch's absolute `from fomo.local_settings import *` — its `ImportError` guard tolerates only that module name, and the relative form crashes every checkout without the file under `manage.py` (38-REVIEW WR-02).
- [Phase 38 follow-ups]: the next `issue37-code-only` snapshot refresh (D-11 recipe) carries 38-07's two docs commits onto PR #43's diff; `CLAUDE.md:106` names `local_settings.py` without a path (PR #58 rewrites that line on `main`); REQUIREMENTS.md traceability table lacks PROP-01, SITE-10, DEP-01 (phase.complete warning).
- [Phase 40]: WARN-05 names `docs/notebooks/pre_executed/README`; the real file is `docs/notebooks/README.md`.
- [Phase 42]: REVERIFY-02 routes fixes into "the TRIAGE-02 gap-closure phase", which runs before Phase 42. A re-verification gap fixed this milestone needs a further phase inserted after 42, then a re-run of that report.
- [Phase 41]: TRIAGE-03 needs the TOM Toolkit Slack "multi proposal support" thread content from the developer.

## Deferred Items

Items acknowledged and deferred at the v2.4 close, most recent first. (Before that close, 7 quick tasks
and 4 debug sessions the audit flagged were found to be complete and closed in f45e17e rather than deferred.)

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | 33/deferred-items.md: flaky `test_observatory_create_form_submits_to_observatory_url` (live MPC call timed out) | acknowledged | 2026-10-06 | v2.4 |
| deferred_items | 37/deferred-items.md: flaky `test_observatory_create_form_submits_to_observatory_url` (order-dependent Playwright failure in full run) | acknowledged | 2026-10-06 | v2.4 |
| seeds | SEED-003 | dormant | 2026-10-06 | v2.4 |
| seeds | SEED-004 | dormant | 2026-10-06 | v2.4 |
| seeds | SEED-261007-5pe | dormant | 2026-10-06 | v2.4 |
| seeds | SEED-261007-j63 | dormant | 2026-10-06 | v2.4 |
| todos | 2026-09-01-add-ttl-cache-to-attribution-banner-count.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-09-01-guard-attribution-dismiss-action-with-is-offered-candidate.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-09-01-skip-sun-event-computation-for-already-existing-reconciler-n.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-09-30-fetch-lco-observation-blocks-in-bulk-per-proposal.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-02-load-telescope-runs-skip-comment-lines-and-warn-on-a-bare-pr.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-02-run-pre-executed-demo-notebooks-against-a-scratch-db-copy-ne.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-a-failed-or-aborted-record-keeps-its-last-scheduled-window-i.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-decide-whether-campaignrun-run-status-needs-an-awarded-and-i.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-delete-the-reconciler-s-own-stale-run-pk-container-on-a-cont.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-explain-campaignrun-telescope-class-setting-it-on-a-site-res.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-give-the-campaign-gap-analysis-a-start-end-date-control.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-isolate-the-campaign-table-query-count-test-from-the-shared.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-keep-a-request-s-site-restriction-and-show-it-in-the-event-t.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-link-each-campaign-table-row-to-its-run-or-give-the-target-i.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-mark-site-lookups-as-not-attempted-in-project-observation-ca.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-report-system-link-outcomes-in-the-discovery-step-s-tick-sum.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-revisit-a-human-confirmed-allocation-night-is-not-retired-if.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-say-in-watchedproposal-attributed-to-help-text-and-the-runbo.md | (presence-only) | 2026-10-06 | v2.4 |
| todos | 2026-10-07-show-a-proposal-level-unused-nights-figure-once-per-proposal.md | (presence-only) | 2026-10-06 | v2.4 |

## Session

**Last session:** 2026-10-08T22:55:54.433Z
**Stopped at:** Completed 39-05-PLAN.md
**Resume file:** None

## Operator Next Steps

- Discuss Phase 39 with /gsd-discuss-phase 39 (calendar write access: who may still write — any logged-in user or staff only — and whether the todo URLs are guarded too)
- Carried from Phase 38: next `issue37-code-only` snapshot refresh (D-11) for PR #43; `CLAUDE.md:106` path; keep `fomo.local_settings` import at the next main sync (PR #58)
