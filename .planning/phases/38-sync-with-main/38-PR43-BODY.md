## Change Description
- [x] My PR includes a link to the issue that I am addressing

Related to #37.

This branch is the telescope runs calendar for FOMO. As of v2.4 the calendar is observation-first: it is driven by what the LCO and SOAR telescopes actually observed, with the planned classical and allocated nights shown until a real observation replaces them. As of v2.5 Phase 38 the branch also carries main's tooling (tomtoolkit 3.1.0, tom_jpl 0.3.0, ruff 0.16.9, the LINCC template v2.2.0 and the Django test runner with coverage in CI), so the diff here is the calendar work on top of current `main`, not a revert of main's newest changes.

Replaces #41 (closed), which also carried the full `.planning/` workflow history. This branch carries no `.planning/` history.

**This PR stays a draft until v2.5's Phases 39-42 land.**

## Solution Description

Operator documentation: [`docs/runbooks/telescope_runs_calendar.rst`](https://github.com/lsst-sssc/fomo/blob/issue37-code-only/docs/runbooks/telescope_runs_calendar.rst) is the tutorial for everything below. The design write-up is `docs/design/telescope_runs_calendar.rst`.

### Observation projector
`solsys_code/observation_projector.py` gives every LCO and SOAR `ObservationRecord` exactly one calendar event. The event starts as the scheduling window and narrows each time the record is saved, through a `post_save` receiver, until it is the block that actually ran. The `project_observation_calendar` management command sweeps the same logic over existing records as a backstop, so a missed save is repaired on the next pass.

### Allocation layer
`solsys_code/allocation_projector.py` shows awarded time as planned nights (`ALLOC:` events, from `ProposalTimeAllocation` records) before anything has been observed. A planned night retires as soon as a real observation links to it. The link is made automatically when an observation's identity matches exactly, at the time the observation is ingested. `cutover_classical_allocations` is the one-time command that moves existing classical allocations onto this layer.

### Unattended operation
`run_unattended` runs the whole pipeline (status refresh, the projection sweep, discovery, reconcile and proposal allocation) in order from one cron line, and `check_unattended` reports on it. The proposals to watch are an admin-editable `WatchedProposal` list, so adding a proposal needs no code change. `deploy/cron/` and `deploy/logrotate/` hold example templates. Failures are reported two ways: an email on a failing tick, and a heartbeat ping that alerts when the schedule stops altogether.

### Public tallies
`campaign_tally.py` and `status_vocabulary.py` give every user, logged in or not, a read-only tally for each campaign run and for each campaign. It counts the observation groups and records linked to the run and four kinds of night (observed, scheduled, expired or failed, and unused), all computed from the linked records and using one shared status vocabulary across the calendar, the tables and the tallies. A tally never changes a run's own status.

### How to try it
```
python manage.py migrate
python manage.py load_telescope_runs <schedule-file> --dry-run
python manage.py run_unattended --dry-run
```
`--dry-run` reports what would happen without writing, pinging or mailing. The runbook section "How do I run everything unattended?" walks through a first tick and the two failure signals.

## Code Quality
- [ ] I have read the Contribution Guide and agree to the Code of Conduct
- [x] My code follows the code style of this project (ruff 0.16.9 through `pre-commit run`)
- [ ] My code builds (or compiles) cleanly without any errors or warnings
- [x] My code contains relevant comments and necessary documentation
- [x] I have added or updated tests under `solsys_code/` for the change (`python manage.py test`)

## FOMO-Specific Checklist
- [x] **Models:** I ran `python manage.py makemigrations` and the resulting migration(s) are committed
- [x] **Settings:** Changes to `src/fomo/settings.py` that a deployment must mirror in `local_settings.py` are called out above. A deployment sets `FOMO_BASE_URL`, `FOMO_HEARTBEAT_URL`, `FOMO_LOCK_DIR`, `FOMO_STATE_DIR` and `FOMO_LOG_FILE` in its environment, and a real `EMAIL_BACKEND` with its host settings in `local_settings.py`. `tom_registration` is gone from `INSTALLED_APPS` and the middleware list, so a `local_settings.py` must not reference it.
- [x] **Dependencies:** Changes to `pyproject.toml` dependencies are reflected in the Requirements list in `docs/installation.rst` (it lists `timezonefinder`)
- [x] **Docs:** New or changed user-facing behaviour is documented under `docs/` (the operator runbook and the pre-executed demo notebooks)
