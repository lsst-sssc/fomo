---
gsd_state_version: "1.0"
milestone: v2.5
milestone_name: Main Sync & Consolidation
current_phase: 38
current_phase_name: Sync with main
status: executing
stopped_at: Completed 38-06-PLAN.md
last_updated: "2026-10-07T23:32:53.742Z"
last_activity: 2026-10-07
last_activity_desc: Phase 38 execution started
state_head: d27ccf4dd8793c5d8e98c3eda5c1ae2070a5724c
progress:
  total_phases: 5
  completed_phases: 39
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-06 — v2.5 milestone started)

**Core value:** The `issue37-telescope-runs-calendar` branch is back in step with `main` — same dependency floors, same tooling, same CI runner — and the debt v2.4 carried forward is either fixed or has a written decision, so the next feature milestone starts from a current, clean base.
**Current focus:** Phase 38 — Sync with main

## Current Position

Phase: 38 (Sync with main) — EXECUTING
Plan: 6 of 6 complete
Status: Phase 38 execution complete, ready for verification
Last activity: 2026-10-07 — Phase 38 execution started

Progress: [██████████] 100%

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

### Pending Todos

19 files under `.planning/todos/pending/` at milestone start (listed in Deferred Items below). Phase 41 (TRIAGE-01) gives each one a decision; WARN-05 (Phase 40) closes the 2026-10-02 scratch-DB todo.

### Blockers/Concerns

- [Phase 38]: PR #43's head is `issue37-code-only`, last synced "through v2.2" (2026-09-01). A v2.4 description (SYNC-08) only matches the PR's diff if that branch is refreshed (e.g. `/gsd-pr-branch`) — decide in discuss-phase.
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

**Last session:** 2026-10-07T23:32:53.719Z
**Stopped at:** Completed 38-06-PLAN.md
**Resume file:** None

## Operator Next Steps

- Discuss Phase 38 with /gsd-discuss-phase 38 (settle the PR #43 head-branch question there)
