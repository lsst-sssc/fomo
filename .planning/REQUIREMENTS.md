# Requirements: Telescope Runs Calendar — v2.5 Main Sync & Consolidation

**Defined:** 2026-10-06
**Core Value (this milestone):** The `issue37-telescope-runs-calendar` branch is back in step with `main` — same dependency floors, same tooling, same CI runner — and the debt v2.4 carried forward is either fixed or has a written disposition, so the next feature milestone starts from a current, clean base.

## v1 Requirements

Requirements for this milestone. Each maps to a roadmap phase.

### Sync with main (SYNC)

- [ ] **SYNC-01**: `origin/main` is merged into `issue37-telescope-runs-calendar` with `git merge` (not rebase); the merge commit's parents are the branch head and `origin/main`'s head at merge time, and no `main` commit since the `756680f` merge base is missing from the branch history.
- [ ] **SYNC-02**: `pyproject.toml` carries `main`'s dependency floors — `tomtoolkit>=3.1.0` and `tom_jpl>=0.3.0` — and the dev environment runs tomtoolkit 3.1.0 (`pip show tomtoolkit` reports 3.1.0 or later). No other version floor is raised beyond what `main` already requires.
- [ ] **SYNC-03**: ruff is 0.16.9 everywhere it is referenced — `.pre-commit-config.yaml`'s `ruff-pre-commit` rev, the dev extra (`ruff>=0.16`, as `main` has it), and the lint/format commands and D-07 note in `CLAUDE.md` — and `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` are clean on the merged tree.
- [x] **SYNC-04**: The repository follows LINCC python-project-template v2.2.0 as `main` does (PR template, hooks, `.gitignore` for the collectstatic output directory), with no FOMO-specific file from the branch lost in the merge.
- [x] **SYNC-05**: CI (`.github/workflows/`) runs `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` as `main` does, with `coverage` reporting; the pytest-based CI job is gone.
- [ ] **SYNC-06**: The dead pytest configuration is removed — `[tool.pytest.ini_options]`, the `pytest`/`pytest-cov` dev extras, and the legacy `tests/` directory — and `CLAUDE.md`'s Testing section no longer describes a pytest suite as present.
- [x] **SYNC-07**: The full suite (`python manage.py test solsys_code --exclude-tag=ephemeris_segfault`) passes on the merged tree with tomtoolkit 3.1.0; any failure caused by a tomtoolkit 3.0.1→3.1.0 or `main` change is fixed in FOMO code, not by skipping the test.
- [x] **SYNC-08**: Draft PR #43's description is rewritten to describe what the branch delivers as of v2.4 (observation projector, allocation layer, unattended operation, tallies) and links `docs/runbooks/telescope_runs_calendar.rst`; the PR remains a draft.

### Calendar write access (ACCESS)

- [ ] **ACCESS-01**: An anonymous `POST` to any write endpoint wired in `solsys_code/calendar_urls.py` — `create-event`, `update-event`, `delete-event`, `create-todo`, `update-todo` — does not create, change or delete a row; the request is redirected to login (or refused), and a test per endpoint asserts the row count and the targeted row are unchanged. Which signed-in users may still write (any, or staff only) is decided in discuss-phase. (Phase 33 review WR-05, never fixed.)
- [ ] **ACCESS-02**: The month-view template's create/update click targets (`src/templates/tom_calendar/partials/calendar.html`) are hidden from anonymous users, so the public calendar does not advertise a write it will refuse.

### Review warnings (WARN) — Phase 37.1 ledger, all still `open`

- [ ] **WARN-01** (WR-05): The header comment on `src/templates/tom_calendar/partials/event_form.html` states accurately which blocks differ from the upstream `tom_calendar` template, instead of claiming it is an exact copy apart from one block.
- [ ] **WARN-02** (WR-13): The pre-executed-notebook guard fails a notebook whose *current source* routes Django to the developer database, regardless of whether its committed output is stale.
- [ ] **WARN-03** (WR-14): No pre-executed notebook relies on `assert` to protect the resolved database path before running `migrate`; the check raises an ordinary exception that survives `python -O`.
- [ ] **WARN-04** (WR-15): The attribution page (`src/templates/campaigns/attribution_queue.html` and partials) uses Bootstrap 5.3 class names only; the High-band row marker renders again, and a template test asserts it is present in the rendered HTML.
- [ ] **WARN-05** (WR-16): Every notebook under `docs/notebooks/pre_executed/` is audited for live developer-database access; every one that reads or copies `src/fomo_db.sqlite3` is changed to build its own scratch database the way the compliant notebooks already do, and `docs/notebooks/README.md` states the isolation rule the notebooks actually follow. (Also closes pending todo 2026-10-02 "Run pre-executed demo notebooks against a scratch DB copy".)
- [ ] **WARN-06** (WR-17): `telescope_runs_demo.ipynb`'s committed output reports the horizon dip in a single unit (degrees or arcminutes), not "arcmin deg"; the notebook is re-executed and committed.
- [ ] **WARN-07** (WR-18): Pressing Undo on a Dismissed row in the attribution queue leaves the Dismissed section open and on the same page, so working down the list needs no re-open or re-page per row.

### Re-verify the stale v2.4 reports (REVERIFY)

- [ ] **REVERIFY-01**: After the `main` sync (SYNC-07) and the cleanup phases have landed, each v2.4 verification report that `verification.status` reads as `stale` (phases 34, 35, 36, 37 and 37.1, under `.planning/milestones/v2.4-phases/`) is re-run by the verifier against HEAD, goal-backward against that phase's own must-haves, and the refreshed report is written back into the archived phase directory with a status of `passed`, `gaps_found` or `human_needed` — never left `stale`.
- [ ] **REVERIFY-02**: Any gap a re-verification finds is either fixed in this milestone — as a REQ-ID in a gap-closure phase inserted after Phase 42, followed by a re-run of that phase's report — or recorded with a reason in the report's gaps section and in `.planning/MILESTONES.md`'s v2.4 "Known Gaps" entry, so the v2.4 override close is replaced by an honest statement of what still holds.
- [ ] **REVERIFY-03**: `.planning/MILESTONES.md`'s v2.4 entry no longer lists "stale verification reports" as a known gap; it records the re-verification date and outcome per phase.

### Todo triage (TRIAGE)

- [ ] **TRIAGE-01**: Every file under `.planning/todos/pending/` (19 at milestone start) and backlog Phase 999.1 has a written disposition — *fix now*, *drop* or *park* — with a one-line reason, recorded in a single triage note under `.planning/`; dropped todos are closed, parked ones stay pending with the reason added to the file.
- [ ] **TRIAGE-02**: Each *fix now* item is turned into a requirement with its own REQ-ID and inserted as a gap-closure phase (`/gsd-phase --insert`) after the triage phase, so the fixes are planned and verified like any other work rather than done ad hoc.
- [ ] **TRIAGE-03**: `SEED-261007-5pe` records (done early, 2026-10-06, commit be11549 — the triage phase confirms the entry reads correctly) the gist of the TOM Toolkit Slack "multi proposal support" thread (who is driving it, what model or API is proposed, expected release) so the Proposal-record design in a later milestone starts from it. No code change.

## v2 Requirements

Deferred to a later milestone. Tracked but not in this roadmap.

### Proposal record

- **PROP-01**: A `Proposal` record links `WatchedProposal`, `ProposalTimeAllocation` and runs (SEED-261007-5pe), aligned with whatever TOM Toolkit's multi-proposal work lands.

### Site codes

- **SITE-10**: Per-site obscode sets for LCO site codes with a membership check (SEED-261007-j63).

### Dependency currency

- **DEP-01**: Django, astropy, sorcha and the other scientific dependencies are bumped to current releases, with the suite re-run — only after v2.5 has settled the `main` sync.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Merging PR #43 to `main` | A human decision about the repository's release line; this milestone makes the PR current and keeps it a draft. |
| Version bumps beyond `main`'s floors (Django 5.2.18 etc.) | One source of breakage at a time: first match `main`, then bump separately (DEP-01). |
| Implementing SEED-261007-5pe or SEED-261007-j63 | Feature work; the Proposal record should be designed after upstream's multi-proposal direction is known. |
| ESO sync (SEED-001/002), SEED-004 (upstreaming the projector), SUBMIT-06/07 | Unchanged from the v2.4 close; not part of this housekeeping ask. |
| Fixing every pending todo | TRIAGE-01 decides which are worth fixing; the rest are dropped or parked with a reason. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SYNC-01 | Phase 38 | Gaps Found |
| SYNC-02 | Phase 38 | Gaps Found |
| SYNC-03 | Phase 38 | Gaps Found |
| SYNC-04 | Phase 38 | Complete |
| SYNC-05 | Phase 38 | Complete |
| SYNC-06 | Phase 38 | Gaps Found |
| SYNC-07 | Phase 38 | Complete |
| SYNC-08 | Phase 38 | Complete |
| ACCESS-01 | Phase 39 | Pending |
| ACCESS-02 | Phase 39 | Pending |
| WARN-01 | Phase 39 | Pending |
| WARN-02 | Phase 40 | Pending |
| WARN-03 | Phase 40 | Pending |
| WARN-04 | Phase 40 | Pending |
| WARN-05 | Phase 40 | Pending |
| WARN-06 | Phase 40 | Pending |
| WARN-07 | Phase 40 | Pending |
| TRIAGE-01 | Phase 41 | Pending |
| TRIAGE-02 | Phase 41 | Pending |
| TRIAGE-03 | Phase 41 | Pending |
| REVERIFY-01 | Phase 42 | Pending |
| REVERIFY-02 | Phase 42 | Pending |
| REVERIFY-03 | Phase 42 | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23 (Phase 38: 8, Phase 39: 3, Phase 40: 6, Phase 41: 3, Phase 42: 3)
- Unmapped: 0 ✓

TRIAGE-02 is delivered by Phase 41 (the triage produces the fix-now REQ-IDs and inserts the gap-closure phase, expected 41.1); the fix-now REQ-IDs themselves are added to this file and mapped to that inserted phase when it is created.

---
*Requirements defined: 2026-10-06*
*Last updated: 2026-10-06 after roadmap creation (v2.5 Phases 38-42)*
