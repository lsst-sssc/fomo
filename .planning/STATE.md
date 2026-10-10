---
gsd_state_version: "1.0"
milestone: v2.5
milestone_name: Main Sync & Consolidation
current_phase: 40
current_phase_name: Notebook Isolation & Attribution Page
status: executing
stopped_at: Completed 40-14-PLAN.md
last_updated: "2026-10-10T23:34:44.302Z"
last_activity: 2026-10-10
last_activity_desc: Phase 40 execution started
state_head: f38a7b776be8e634b783007ff080945930c5fa5b
progress:
  total_phases: 6
  completed_phases: 42
  total_plans: 31
  completed_plans: 31
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-09 — Phase 39.1 complete)

**Core value:** The `issue37-telescope-runs-calendar` branch is back in step with `main` — same dependency floors, same tooling, same CI runner — and the debt v2.4 carried forward is either fixed or has a written decision, so the next feature milestone starts from a current, clean base.
**Current focus:** Phase 40 — Notebook Isolation & Attribution Page

## Current Position

Phase: 40 (Notebook Isolation & Attribution Page) — EXECUTING
Plan: 14 of 14
Status: Ready to execute
Last activity: 2026-10-10 — Phase 40 execution started

Progress: [████████████████████] 17/17 plans ([██████████] 100%)

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
| Phase 39 P06 | 41 min | 2 tasks | 2 files |
| Phase 39.1 P01 | 90 min | 3 tasks | 6 files |
| Phase 39.1 P02 | 4h 28m | 3 tasks | 9 files |
| Phase 39.1 P03 | 22 min | 3 tasks | 11 files |
| Phase 39.1 P04 | 45 min | 3 tasks | 2 files |
| Phase 40 P01 | 70 min | 2 tasks | 7 files |
| Phase 40 P02 | 20 min | 2 tasks | 9 files |
| Phase 40 P03 | 30 min | 3 tasks | 12 files |
| Phase 40 P04 | 11 min | 3 tasks | 10 files |
| Phase 40 P05 | 15min | 3 tasks | 3 files |
| Phase 40 P06 | 6 min | 2 tasks | 1 files |
| Phase 40 P07 | 20 min | 3 tasks | 9 files |
| Phase 40 P08 | 12 min | 3 tasks | 5 files |
| Phase 40 P09 | 19 min | 3 tasks | 10 files |
| Phase 40 P10 | 16 min | 3 tasks | 2 files |
| Phase 40 P11 | 9 min | 2 tasks | 8 files |
| Phase 40 P12 | 2 min | 3 tasks | 1 files |
| Phase 40 P13 | 11 min | 3 tasks | 2 files |
| Phase 40 P14 | 12 min | 3 tasks | 4 files |

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
- [Phase 39]: 39-06: G-39-7 closed test-only; the hint gate in event_form.html is correct and the staff conjunct is now pinned at view and template level (WR-04 fixed)
- [Phase 39.1]: 39.1-01: sun_event() crossing search memoised per process via lru_cache(maxsize=4096) helper below the validation, returning Time copies; _find_crossing is a 10-minute scan plus 14-step vectorised bisection, within 0.044 s of the pre-phase 1-minute scan
- [Phase 39.1]: Phase 39.1: race fixed by a per-process LocMemCache under manage.py test (argv[1]=='test' block above the local_settings fold tail) plus override_settings on the two cache-clearing classes
- [Phase 39.1]: Phase 39.1: --parallel with no worker count in the django-test hook and CI, hook keeps coverage, daily smoke test stays serial on purpose
- [Phase 39.1]: Phase 39.1: coverage config-file form (concurrency multiprocessing, parallel true) works; hook starts with coverage erase and runs coverage combine before reporting
- [Phase 39.1]: Phase 39.1: tblib approved by the developer and added to the dev extra
- [Phase 39.1]: 39.1-03: migration tests = the six MigrationExecutor TransactionTestCase classes, tagged @tag('migration'); excluded in the django-test hook only, CI keeps all 18
- [Phase 39.1]: 39.1-03: MD5PasswordHasher lives in the argv[1]=='test' block of settings.py only; django-test hook measured 2276 s -> 78 s (29.2x), coverage 93% -> 93%, so coverage stays in the hook
- [Phase 39.1]: 39.1-04: developer published the snapshot (commit 8cb2d2f) to draft PR #43; CI run 37941525172 green on 3.10-3.12 with --parallel, unit step 1019/1171/1140 s -> 163/110/160 s
- [Phase 39.1]: UAT (2026-10-09): code-review WR-01 (a dark window shorter than one 10-minute coarse step, possible above about 51.5° latitude, returns no crossings and campaign_gap files that date as unknown) accepted as shipped; no code or doc change requested; no currently stored site is affected
- [Phase 40]: Phase 40-01: notebook DB setup moved into docs/notebooks/fomo_nb_utils.py (setup_scratch_django / teardown_scratch_django); RuntimeError path check survives python -O; reconcile_campaign_runs_demo uses mode='snapshot' (mode=ro online backup), load_telescope_runs, backfill_lco and campaign_lifecycle use mode='fresh'
- [Phase 40]: 40-02: notebook guard checks current source (AST rules R0-R11) rather than committed output; all eight notebooks call setup_scratch_django; README/ROADMAP/REQUIREMENTS share one isolation-rule sentence pinned by a test
- [Phase 40]: 40-03: High-band attribution marker is border-0 border-start border-4 border-success; template class guard scans every repo-owned template directory (DIRS plus in-repo app dirs), never third-party apps
- [Phase 40]: 40-04: attribution return state travels as hidden form inputs plus one whitelisted redirect (attribution_return_state); the table prefixes live in campaign_tables.py so an unconfigured table still posts dismissed-page/confirmed-page — Keeps the tables free of any view import, and makes request=None tables post the same keys; a page past the end is kept as posted and django-tables2 renders the last page with rows
- [Phase 40]: 40-04: UI-SPEC F1 (scroll-margin-top 5rem on both section divs) and F3 (Undo-confirmation prompt says above; flash says the pair is now in Dismissed) applied in 66c1285 — User adopted both on 2026-10-09; they post-date the plan text
- [Phase 40]: 40-05: WR-02 decision -- notebook setup refuses any DATABASES alias other than default and any DATABASE_ROUTERS (refuse, not support)
- [Phase 40]: 40-05: helper fails closed -- settings checked before django.setup(); any later failure switches every alias to the dummy backend; _active_setup module-level record carries scratch dir and previous env to teardown
- [Phase 40]: 40-06: R12 reports database-routing names (not intent) in every notebook, Django-backed or not; its names also feed R0's unparseable-cell list so the two cannot drift
- [Phase 40]: 40-07: spiceypy 'Using fallback library' stderr line (a /home/ path) in campaign_lifecycle_demo is third-party log output, listed in the SUMMARY and left in place
- [Phase 40]: 40-08: R12/R0 guard extended to whole capabilities (process, environment, connection replacement, code from a string, ATTACH); the 40-06 completeness claim is withdrawn and the residual named in the docstring (D-G1)
- [Phase 40]: 40-08: helper switches every alias to the dummy backend before a best-effort close and re-raises its own error; teardown second line says what setup checked (D-G2)
- [Phase 40]: 40-09: all eight pre-executed notebooks re-executed on the exception-safe helper; teardown and refusal prose narrowed to what the helper checks (D-G2, D-G3); NF-14 labels and requirement-table citations made data-independent (D-G4); developer override for the residual WARN-02 risk recorded last (D-G1)
- [Phase 40]: 40-10: R12 matches every listed name at every identifier, in every dotted part of a str or bytes constant and in code inside strings (two levels); a string counts as code only when it holds a statement or a call
- [Phase 40]: 40-10: R0 is an allow-list over IPython's own TransformerManager (harmless only: %matplotlib inline, a first-line %%time or %%capture; %config is not harmless) plus an automagic rule for one-line cells built from IPython's live magic registry
- [Phase 40]: 40-10: R12 adds connection, _connections, Database, getoutputerror, get_output_error_code, process_handler, import_string, resolve_name, locate, Macro and ATTACH (; R6 adds copytree, copyfileobj, move and write_bytes
- [Phase 40]: 40-11: the guard module docstring and README describe the guard as 40-10 left it, rule by rule, and name exactly the residual RES; string code is described with 40-10's test (a string that parses and holds a statement or a call)
- [Phase 40]: 40-11: the _disable_django_databases docstring change is a docstring only, so no notebook is re-executed for IN-01 (D-G6 c); the matching warning text in the helper's code is left for Phase 41 triage
- [Phase 40]: D-G5 (5): Tim Lister approved the corrected WARN-02 override reason verbatim (2026-10-10T19:14:57Z); it names only what 40-10's tests and 40-11's guard docstring support, and the accepted residual is exactly the docstring's 'What the guard does not see' paragraph — Recorded in 40-VERIFICATION.md's one overrides: entry (44c33c6); the verifier applies or rejects it at re-verification
- [Phase 40]: 40-13: R0 automagic rule no longer exempts a name an earlier cell binds (IPython decides from the namespace at run time); R6 follows shutil names notebook-wide and reports distinctive copy names everywhere; R5/R7 read bytes; ipykernel>=7 in dev extra
- [Phase 40]: 40-14: developer (Tim Lister) approved the guard docstring, README paragraph and reworded override reason as drafted (accepted_at 2026-10-10T23:20:47Z); IN-01 stays open and out of scope, recorded for Phase 41

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
- [Phase 38 follow-ups]: 38-07's docs commits reached PR #43 in the 39.1-04 snapshot (8cb2d2f); still open: `CLAUDE.md:106` names `local_settings.py` without a path (PR #58 rewrites that line on `main`); REQUIREMENTS.md traceability table lacks PROP-01, SITE-10, DEP-01 (phase.complete warning).
- [Phase 40]: WARN-05 names `docs/notebooks/pre_executed/README`; the real file is `docs/notebooks/README.md`.
- [Phase 42]: REVERIFY-02 routes fixes into "the TRIAGE-02 gap-closure phase", which runs before Phase 42. A re-verification gap fixed this milestone needs a further phase inserted after 42, then a re-run of that report.
- [Phase 41]: TRIAGE-03 needs the TOM Toolkit Slack "multi proposal support" thread content from the developer.

### Roadmap Evolution

- Phase 39.1 inserted after Phase 39: Test-Suite Speed-Up: memoise sun_event(), cheaper crossing search, --parallel in CI and the django-test hook, fast test hasher and tagged migration tests, so the hook drops from ~10 min to ~1 min and commits stop needing SKIP=django-test (source: todo 2026-10-08) (URGENT)

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

**Last session:** 2026-10-10T23:34:44.085Z
**Stopped at:** Completed 40-14-PLAN.md
**Resume file:** None

## Operator Next Steps

- Discuss Phase 40 with /gsd-discuss-phase 40 (notebook isolation: every pre-executed demo notebook builds its own scratch DB with a guard test; attribution page under Bootstrap 5.3 — WR-13 to WR-18). Code commits now run the full django-test hook in about 80 s; `SKIP=django-test` is only for .planning-only commits or a deliberately red work-in-progress commit
- Carried from Phase 39.1: review WR-01 accepted as shipped at UAT (sub-10-minute dark windows above about 51.5° latitude); info items IN-01..IN-06 in 39.1-REVIEW.md go to Phase 41 triage with the Phase 39 ledger; Phase 41 closes todo 2026-10-08 (sun_event cache) as fixed in 39.1 and todo 2026-10-07 (campaign table query-count test / shared cache) the same way
- Carried from Phase 39: review ledger findings WR-02, WR-03, IN-01..IN-08 go to Phase 41 triage; UAT decisions D-01 (any logged-in user may write) and AR-39-01/AR-39-02 stand
- Carried from Phase 38: `CLAUDE.md:106` path; keep `fomo.local_settings` import at the next main sync (PR #58); REQUIREMENTS.md traceability table lacks PROP-01, SITE-10, DEP-01
