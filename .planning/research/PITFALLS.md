# Pitfalls Research

**Domain:** Adding unattended (cron/scheduled) execution to a Django/TOM-Toolkit app with a SQLite dev DB and live external-API sync commands; migrating three independently-keyed "adapter writes CalendarEvent directly" commands into "adapter writes CampaignRun, reconciler projects the event"; deriving `CampaignRun.run_status` from linked `ObservationRecord` outcomes.
**Researched:** 2026-09-01
**Confidence:** HIGH (grounded directly in this repo's own code — `campaign_reconciler.py`, the three sync commands, `models.py`, `settings.py` — plus MEDIUM/HIGH-confidence general Django/SQLite/cron sources for the non-repo-specific claims)

## Critical Pitfalls

### Pitfall 1: SQLite write-lock collision between a scheduled sync and a concurrent staff action

**What goes wrong:**
`src/fomo/settings.py`'s `DATABASES['default']` has no `OPTIONS['timeout']` set, so Django's sqlite3 backend uses the driver default (5 s) before raising `django.db.utils.OperationalError: database is locked`. Today every write to `CampaignRun`/`CalendarEvent` happens inside an HTTP request-response cycle, so writes are naturally spaced out by human click speed. Once a scheduler runs `sync_lco_observation_calendar`/`sync_gemini_observation_calendar`/`load_telescope_runs` every few minutes, each invocation opens a write transaction for every record/night it touches (`insert_or_create_calendar_event`, and after the adapter rewrite, `CampaignRun` create/update plus `reconcile_run()`'s own writes) at the same time a staff member might be approving a run or resolving a site in the approval queue UI. SQLite allows exactly one writer at a time; a long-running sync (many records, each triggering a live telescope-API lookup per `resolve_placement_block`) can hold the write lock long enough that a concurrent UI action times out and 500s, or vice versa.

**Why it happens:**
SQLite's file-level locking model is invisible during manual testing (one developer, one browser tab) and only surfaces under real overlapping writers — exactly the scenario unattended scheduling introduces for the first time in this project's life. The `sync_lco_observation_calendar` per-record loop also does a live API call (`resolve_placement_block`) *before* its DB write per record, which under "BEGIN then slow I/O then COMMIT" would hold a lock for the API round-trip; this codebase's `insert_or_create_calendar_event` writes are already short (see `calendar_utils.py`), but the risk grows if the adapter rewrite folds more logic between the API call and the commit.

**How to avoid:**
- Set `DATABASES['default']['OPTIONS'] = {'timeout': 20}` (or higher) so a lock contention resolves by waiting rather than raising immediately — cheap, should ship in the same phase that adds scheduling, not left as an incident-driven fix.
- Ensure the scheduling layer never runs two instances of the *same* sync command concurrently (a `flock`-style lock file or the scheduler's own overlap-prevention, e.g. cron `flock(1)` or a task-queue's single-worker-per-task-name setting) — the phase-time investigation spike (cron vs. task queue) should make this an explicit selection criterion, not an afterthought.
- Keep external API calls outside the DB transaction boundary — resolve the live telescope-API lookup, then do the DB write as a short, separate step (already true of `resolve_placement_block` calls happening before `insert_or_create_calendar_event`; preserve this ordering when the adapter is rewritten to write `CampaignRun` instead).
- Do not widen this into a Postgres migration as part of this milestone — CLAUDE.md already documents SQLite's concurrent-write limit as a known constraint for production; the fix here is bounded (timeout + no-overlap scheduling), not a database migration.

**Warning signs:**
`OperationalError: database is locked` in scheduler logs or Django error emails; staff reports of approval-queue actions failing intermittently only around the sync schedule's cadence; test suite passing but a live scheduled run failing only when a human is using the approval queue at the same time.

**Phase to address:**
The unattended-execution/scheduling-mechanism phase (the phase-time investigation spike plus its implementation) — this is a property of *how* the sync commands are invoked unattended, not of the adapter rewrite itself, so it must be settled before or alongside the scheduling phase, not deferred to the adapter-rewiring phase.

---

### Pitfall 2: A scheduled sync job fails and no one ever finds out

**What goes wrong:**
Today, a failed sync is visible because a human ran it and saw stderr/a traceback in their terminal. Once a scheduler (cron or a task queue) runs these commands unattended, a failure — a network timeout to the LCO Observation Portal, a Gemini API credential rejection, an unhandled exception in the adapter's new `CampaignRun` create/update path — produces output that goes nowhere a person will look. Cron's default behavior mails command output to a local mailbox that is very often unconfigured or unread on this kind of deployment; a task-queue worker's failure likewise just increments an internal counter unless something is watching it. The result: the calendar/CampaignRun data silently goes stale, and by the time a staff member notices (a run that should have synced didn't), the causal event is long gone from any log rotation window.

**Why it happens:**
"Wire this into cron/a scheduler" is naturally scoped as "make the command runnable on a schedule" and stops there; alerting-on-failure is a separate concern that's easy to treat as out of scope because it doesn't block the command from technically working. This project's own repo has zero precedent for this — no celery/cron/APScheduler infrastructure exists yet — so there is no existing convention to inherit; it must be designed from scratch in this milestone.

**How to avoid:**
- Treat "a failure is visible to an operator" as a first-class, explicitly planned requirement in the scheduling phase (the milestone's own target-feature list already states this — "a failure is visible to an operator rather than silently disappearing between runs" — so the plan-checker should flag any plan that implements the schedule but not the failure-visibility path as incomplete).
- Prefer a heartbeat/dead-man's-switch pattern over pure exception-alerting: the job pings a monitoring endpoint (or writes a last-success timestamp somewhere staff can see, e.g. a `runbook`-documented log line or a simple status row) only on success; absence of the expected ping is itself the alert. This also catches the case pure try/except alerting misses — the scheduler itself failing to invoke the job at all (crontab deleted, worker process dead).
- Each sync command should exit with a non-zero status and a clear stderr message on failure (already partially true — `CommandError` is used for fatal cases in `load_telescope_runs`) so the scheduler's own failure-detection (cron's `MAILTO`, or the task queue's retry/alerting config) has something to key off; per-record `try/except` blocks that log-and-continue (as `sync_lco_observation_calendar` already does for `InstrumentExtractionError`/`KeyError`/`ValueError`) are correct for *partial* failures but must not swallow whole-command failures (e.g. total API unavailability) the same way.
- Document the chosen alerting path in the operator runbook (`docs/runbooks/telescope_runs_calendar.rst`) as this milestone's CLAUDE.md paired-docs rule already requires for any behavior-changing module this touches.

**Warning signs:**
A `CampaignRun`/calendar event that should reflect a recent LCO schedule change doesn't, with no error anyone saw; the only way anyone finds out is a PI complaining their observation isn't on the calendar.

**Phase to address:**
The unattended-invocation/scheduling phase — must ship failure-visibility in the same phase as the schedule itself, not as a follow-up once a real silent failure has already happened in production.

---

### Pitfall 3: Credential leakage through unattended-job logs

**What goes wrong:**
`sync_gemini_observation_calendar.py` already had to add explicit password-scrubbing (`safe_params = {k: v for k, v in (record.parameters or {}).items() if k != 'password'}`, D-04) because Gemini ToO parameters can carry a password field that must never reach a description string, log line, or the calendar UI. When these commands run unattended, their stdout/stderr is captured somewhere new (a cron mail spool, a task-queue's log aggregator, a systemd journal) that likely has different retention, access-control, and rotation properties than a developer's interactive terminal — and is more likely to be piped into a shared log viewer, alerting tool, or ticket attached to an incident report. Any credential or token used to authenticate the *unattended job itself* to the LCO Observation Portal or Gemini API (as opposed to a per-record password, which is already scrubbed) is a second, distinct leakage surface: if the scheduling mechanism reads that credential from an environment variable and something logs `os.environ` or a full exception traceback that includes request headers, it ends up wherever the job's output lands.

**Why it happens:**
The existing scrubbing (D-04) was scoped to the one known field (a per-observation `password` parameter surfacing through Gemini's API) at the time it was written — it is not a general policy applied to every place a credential could leak, and a *new* execution path (the scheduler's own log capture) was never in scope when that fix landed, so it's easy to assume "we already handled credentials" and not re-check the assumption against the new unattended context.

**How to avoid:**
- Audit every place these three adapters currently log or write descriptions (`self.stderr.write(...)`, `description +=`, exception messages) for anything that could carry a credential once the *job-level* auth secret (not just the per-record `password` field) is added for unattended API access — confirm no code path ever does a bare `str(exc)`/`repr(exc)` on an exception raised by an authenticated HTTP client library (which can embed request headers/auth in its exception message).
- Keep the job-level credential itself out of command-line arguments (visible in `ps`, shell history, and often in scheduler job listings) — use environment variables or a secrets file read at process start, matching how `os.getenv()` is already used elsewhere in this project per its Configuration conventions.
- Confirm the chosen scheduler's own logging (cron mail, task-queue worker logs) does not echo the full command line (some schedulers log the invoked command including arguments) — if credentials must be passed as arguments for some reason, this becomes a hard blocker on that scheduler choice, not a detail to patch later.
- Extend the D-04-style scrubbing pattern explicitly to any new field the `CampaignRun`-writing adapters read that Gemini/LCO could plausibly source-populate with a secret, rather than assuming the original scrub covers a superset it doesn't.

**Warning signs:**
A credential or password string visible in a cron mail, a task-queue dashboard, or a pasted log snippet in a bug report/Slack thread; `git grep` for `password` in the adapters' write paths turning up a new field D-04 didn't anticipate.

**Phase to address:**
The unattended-invocation phase (for job-level credential handling) and the adapter-rewiring phase (for auditing whether the rewrite introduces any new field that could carry a secret into a `CampaignRun`/log line) — both need an explicit check, not just inherited confidence from the existing Gemini scrub.

---

### Pitfall 4: Dual-write window during the adapter migration produces duplicate or orphaned calendar events

**What goes wrong:**
Today each of the three adapters is both the writer and the sole authority for its `CalendarEvent`s, keyed by its own idempotent lookup (`sync_lco_observation_calendar` on the LCO portal `url`; `load_telescope_runs` on `(telescope, instrument, start_time ± 5 min)`; `sync_gemini_observation_calendar` similarly on its own record identity). The migration goal is: each adapter instead creates/updates a `CampaignRun`, and `campaign_reconciler.reconcile_run()` projects the calendar event. If the rewrite ships adapter-by-adapter (a very likely sequencing, since they're independent commands) or if a rollback/rollout accident leaves the *old* direct-write code path live for one adapter while `reconcile_run()` is *also* now running against `CampaignRun`s created by another already-migrated adapter, both code paths can write a calendar event for the same real observing night — one keyed the old way (e.g. the LCO portal URL, or `(telescope, instrument, start_time)`), one keyed the new way (`RUN:{pk}` or `RUN:{pk}:{date}`). The reconciler's `_may_write()`/ownership guard (RECON-05) specifically protects against the reconciler clobbering an event it doesn't own, but it does **not** prevent the *old* adapter code from continuing to mint its own separate event in parallel — that's a second, independently-keyed event for the same night, i.e. exactly the "Double representation" scenario the v2.2 milestone already had to solve once for `CampaignRun` pk=1 (FTS/MuSCAT4 vs. the 11 LCO queue events) via Phase 28's attribution queue, not deduplication.

**Why it happens:**
A phased/incremental cutover (migrate one adapter, verify, migrate the next) is the natural and lower-risk implementation approach, but "natural" here silently reintroduces a coexistence window each already-migrated adapter's `CampaignRun` and each not-yet-migrated adapter's directly-written `CalendarEvent` both target the same real night, with no shared identity between them until a human runs the attribution queue again. Unlike the Phase 26-29 case (which was existing historical data reconciled once), this is an *ongoing* production window: every sync cycle during the migration period keeps minting fresh instances of the collision, not just a one-time backlog.

**How to avoid:**
- Scope the migration window explicitly in the adapter-rewiring phase's plan, not left implicit: state upfront whether the three adapters cut over atomically (all three switch to writing `CampaignRun`s in the same deploy) or sequentially, and if sequential, what happens to the *not-yet-migrated* adapters' existing direct writes during that window — they keep running exactly as today (acceptable, since it's the status quo, not a regression) as long as nothing else assumes they've already stopped.
- Never run the reconciler's sweep (`reconcile_campaign_runs`) against a `CampaignRun` created by a migrated adapter for the same real night an unmigrated adapter is *still* independently writing to, without confirming the reconciler's `_may_write()` ownership check correctly leaves the unmigrated adapter's event alone (it should, since that event has no `CalendarEventMeta.run` pointing elsewhere and the reconciler will only touch events already in `RUN:` url namespace it owns) — but explicitly test this interaction, not just each adapter's rewrite in isolation, since the pre-existing test suites (`test_sync_lco_observation_calendar.py`, `test_load_telescope_runs.py`) each assume the *other* adapters aren't running.
- Reuse Phase 28's attribution queue as the intended resolution mechanism for any transition-period duplicate (same pattern as the pk=1 case), rather than inventing a new merge/dedup tool — the existing infrastructure already treats "same night, two representations" as an attribution problem to be staff-confirmed, and that framing should extend to duplicates the migration itself introduces, not just historical ones.
- Prefer a feature-flag-per-adapter or one-adapter-at-a-time rollout with a verification step (compare the new `CampaignRun`-derived calendar output against the old direct-write output for a sample of real records) before moving to the next adapter, rather than flipping all three at once with no checkpoint.

**Warning signs:**
Two `CalendarEvent`s for what is obviously the same night/telescope/instrument appearing during the migration period that weren't there before it started; the attribution queue's candidate count spiking right after an adapter cutover instead of trending toward zero; `claimed_dates()`/coverage-gap analysis reporting the same night as both claimed (via the old event) and claimed again (via the new `RUN:` event).

**Phase to address:**
The adapter-rewiring phase's plan must state the cutover sequencing and the dual-write window's handling explicitly (per the milestone's own downstream-consumer note) — this is not something the plan-checker should have to infer; a plan that rewires all three adapters without addressing sequencing/coexistence should be treated as incomplete, not merely under-specified.

---

### Pitfall 5: An adapter's existing idempotency key silently stops working once it targets `CampaignRun` instead of `CalendarEvent`

**What goes wrong:**
Each adapter's "no-churn idempotency" guarantee (`sync_lco_observation_calendar`'s SYNC-04, `load_telescope_runs`'s tolerance-windowed start-time match, `sync_gemini_observation_calendar`'s per-record find-or-create) was designed and tested against `CalendarEvent`'s field set and its `insert_or_create_calendar_event()`/`insert_or_create_campaign_run()`-style compare-and-update helper. `CampaignRun` has a *different* schema and a *different* set of natural-key constraints already in the model (`unique_campaign_run_resolved_window` on `(campaign, telescope_instrument, window_start, window_end)`, and the TBD-branch constraint on `(campaign, telescope_instrument, contact_person)`), neither of which was designed with the LCO portal URL, the Gemini ToO record's own identity, or `load_telescope_runs`' `(telescope, instrument, start_time ± tolerance)` key in mind. Naively mapping each adapter's existing key onto `CampaignRun`'s fields risks two distinct failure modes: (a) the mapping is *looser* than the original key, so two genuinely distinct LCO records (e.g. two different proposals on the same telescope the same week) collide onto the same `CampaignRun` row and overwrite each other's data; or (b) the mapping is *stricter* or misaligned (e.g. keying on `window_start`/`window_end` computed from a `scheduled_start` that isn't set yet during the banner stage), so every re-sync creates a *new* `CampaignRun` instead of updating the existing one — reintroducing duplicate-run churn that Phase 26's spike was specifically convened to prevent (26-DECISION.md: "each adapter's existing identity key mapped onto a run" was one of the four settled spike questions).

**Why it happens:**
This is exactly the kind of decision Phase 26's spike already flagged as needing settlement before implementation for the *v2.2* work — but v2.2's spike settled it for the *existing* historical data and the reconciler's key scheme, not for what happens when live, ongoing adapter writes target `CampaignRun` going forward. The temptation is to assume the spike's settled mapping trivially extends to "and now the adapter writes there directly too," when in fact the adapter's write-time key (what it has *before* a full schedule/placement is known — e.g. a banner-stage LCO record with no `scheduled_start` yet) is a different, earlier-stage question than the reconciler's read-time key (what event to project once the `CampaignRun`'s state is fully known).

**How to avoid:**
- Treat the mapping from each adapter's write-time identity key to `CampaignRun`'s natural key as a design decision that needs its own explicit verification per adapter, not an assumption inherited from Phase 26's reconciler-facing spike — write a test per adapter that re-syncs the *same* underlying source data twice and asserts exactly one `CampaignRun` row results, with no field churn on the second run (mirroring the existing SYNC-04 test pattern, just against `CampaignRun` fields instead of `CalendarEvent` fields).
- Explicitly decide, per adapter, what identifies "the same run" *before* a window is fully resolved (e.g. an LCO banner-stage record with no `scheduled_start`) — if `CampaignRun`'s natural key requires `window_start`/`window_end` to be non-null for the resolved-window constraint to apply, an adapter that runs during the banner stage may need to write into the TBD branch first and then transition the same row to the resolved branch once the schedule places it, rather than creating a fresh row at each stage.
- Confirm neither of `CampaignRun`'s two existing partial `UniqueConstraint`s is silently bypassed by the new adapter-write path in a way that lets two adapter runs create two rows for what a human would call "the same run" — this was exactly the failure mode WR-05 in the model's own `Meta.constraints` comments already calls out get_or_create()-without-a-backing-constraint as unsafe for.

**Warning signs:**
Re-running a sync command against unchanged upstream data creates a *new* `CampaignRun` instead of leaving the existing one `unchanged`; the same LCO proposal's queue record produces a different `telescope_instrument`/window pairing across two consecutive syncs, splitting what should be one run into two.

**Phase to address:**
The adapter-rewiring phase — each adapter's key-mapping decision should be an explicit, tested design point in that phase's plan (not merely "adapter now writes CampaignRun"), and the plan-checker should confirm a no-churn idempotency test exists per adapter against `CampaignRun`, mirroring the existing `CalendarEvent`-facing tests.

---

### Pitfall 6: One bad `ObservationRecord` regresses an otherwise-good run's status

**What goes wrong:**
Deriving `CampaignRun.run_status` automatically from its linked `ObservationRecord`s' outcomes is a many-to-one aggregation (a run can realise through several records — e.g. one per night, or one per rescheduled attempt after a weather loss). A naive "most recent record's status wins" or "any failed record sets the run to failed" rule regresses a run that actually succeeded: e.g. a 3-night classical run where night 1 was weathered out (a real, terminal `WEATHER_TECH_FAILURE`-style outcome on that night's record) but nights 2-3 completed successfully should not report the whole run as failed, and a run whose *first* placement attempt expired/was cancelled before the LCO scheduler successfully replaced it with a completed observation should not stay pinned to the stale, superseded terminal status of the abandoned attempt. The existing terminal-state handling in `sync_lco_observation_calendar.py` (`_FAILURE_PREFIX_BY_STATUS`, `get_failed_observing_states()`/`get_terminal_observing_states()`) is scoped to *one record's* title, and has no aggregation rule at all today — this milestone is the first time a many-records-to-one-run rollup rule must be designed.

**Why it happens:**
The single-record case (today's title-prefix logic) generalizes deceptively easily to "just run the same logic per record and combine somehow," but combining independent per-record verdicts into one parent status is a genuinely different problem (what dominance order do statuses have? does one COMPLETED record's success outweigh two other records' failures for the same run, or vice versa? does a still-`PENDING`/non-terminal record block the run from ever reaching a terminal status, or does it get ignored?) that has no existing precedent in this codebase to copy.

**How to avoid:**
- Design run-status derivation as an explicit, small state-transition table (analogous to the existing terminal/failure-state dictionaries) rather than an ad hoc "last write wins" or "any failure wins" rule — decide up front, in the adapter/outcome-propagation phase's plan, the dominance order across `REQUESTED`/`PLANNED`/`OBSERVED`/`REDUCED`/`PUBLISHED`/`CANCELLED`/`NOT_AWARDED`/`WEATHER_TECH_FAILURE` when a run has multiple linked records at different outcomes, and write it down as a table, not just code.
- Treat a run with mixed outcomes (some records succeeded, some failed/weathered) as "partially observed" rather than collapsing it to either extreme — if the existing `RunStatus` vocabulary has no state for "partially realised," that gap itself is a design decision for this phase to make explicitly (extend the vocabulary, or define a documented convention for how partial success maps onto the existing choices) rather than silently picking whichever `RunStatus` value happens to fall out of whatever aggregation code gets written first.
- Never let a *superseded* record (e.g. an expired/cancelled placement attempt that the LCO scheduler later replaced with a successful one) contribute its terminal failure status to the run once a later record for the same night/window has succeeded — this requires the derivation logic to reason about record recency/supersession, not just "does any linked record have status=FAILED."
- Write the mixed-outcome and superseded-record cases as explicit test fixtures (using `NonSiderealTargetFactory` per this repo's CLAUDE.md convention) before implementing the derivation rule, so the "one bad observation regresses a good run" failure mode is caught by a test, not discovered live.

**Warning signs:**
A `CampaignRun` that genuinely completed successfully shows `run_status=WEATHER_TECH_FAILURE`/`CANCELLED` because one of its several linked records had that terminal status even though a later or different record for the same run succeeded; staff have to manually override an auto-derived status back to what they know is correct, defeating the purpose of automating it.

**Phase to address:**
The outcome-propagation phase — the dominance/aggregation rule must be an explicit design artifact (a table or documented function contract) reviewed before implementation, not inferred from whichever record happens to be processed last.

---

### Pitfall 7: Automatic status derivation fires on an unconfirmed (not-yet-attributed) `ObservationRecord`, or bypasses the human-confirmed link entirely

**What goes wrong:**
`CampaignRunObservation` (CANON-04, Phase 27) links a `CampaignRun` to the `ObservationRecord` that realises it, and by design (D-01/D-03 in `models.py`) that row is written *only* once a staff member confirms the attribution via Phase 28's queue — "a row exists only once a staff member confirms the attribution," with no boolean confirmation flag because the row's mere existence *is* the confirmation. If outcome-propagation code derives `run_status` by querying `ObservationRecord`s some other way — e.g. by re-running the same date/instrument/telescope heuristic `campaign_attribution.py` uses for *candidate scoring* (which is explicitly designed to be an unconfirmed guess, not a real link) rather than strictly through `CampaignRunObservation` rows — it silently reintroduces exactly the unconfirmed-merge risk ATTRIB-03 ("no association is created without explicit staff confirmation") was built to prevent, just one hop removed: no `CampaignRunObservation` row is created, but the run's *status* still changes based on an unconfirmed guess, which is functionally the same trust violation with a different symptom.

**Why it happens:**
Outcome propagation is naturally implemented by a query that finds "records relevant to this run," and the *scoring* heuristic in `campaign_attribution.py` is the most readily available, already-tested piece of code that answers a similar-sounding question ("which records plausibly belong to this run") — reusing it for status derivation is an easy, wrong shortcut, since its purpose is explicitly to surface *candidates for a human to confirm*, not to identify records safe to act on programmatically.

**How to avoid:**
- Outcome propagation must read exclusively through confirmed `CampaignRunObservation` rows (`run.campaignrunobservation_set` or equivalent) — never through `campaign_attribution.py`'s candidate-scoring functions, and never through a fresh ad hoc date/instrument-overlap query that reimplements a weaker version of the same heuristic.
- A run with zero confirmed `CampaignRunObservation` links must not have its status auto-derived at all (it stays whatever it already is — most likely `REQUESTED`/`PLANNED`) — this should be an explicit early-return/guard in the derivation function, mirroring `campaign_reconciler._skip_reason()`'s pattern of an itemized, testable skip-reason vocabulary, not an implicit fallthrough.
- Sequence outcome propagation strictly after a confirmed link exists — this is a data-availability dependency (there is nothing to propagate from until Phase 28's queue produces a `CampaignRunObservation` row), not merely a nice-to-have ordering; a plan that ships outcome-derivation logic against records with no guarantee a confirmed link exists yet will either no-op silently (acceptable, if guarded) or crash/misbehave (not acceptable) depending on how defensively it's written.
- Add a test asserting that an `ObservationRecord` with a plausible attribution *candidate* score (as `campaign_attribution.py` would compute it) but no confirmed `CampaignRunObservation` row never influences `run_status` — this directly guards against the reuse-the-scorer shortcut described above.

**Warning signs:**
A `CampaignRun.run_status` changes before any staff member has visited the attribution queue for the records involved; a run's status reflects an `ObservationRecord` that a later attribution-queue review actually dismisses as *not* belonging to that run (i.e., the status derivation trusted a guess the human then rejected).

**Phase to address:**
The outcome-propagation phase, and it should be explicitly sequenced (in ROADMAP.md / the phase dependency graph) as depending on Phase 28's attribution queue already existing and being the sole write path for `CampaignRunObservation` — not merely assumed compatible because both features happen to touch the same models.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Ship all three adapter rewrites in one big-bang deploy instead of sequencing/flagging them | Avoids designing a dual-write coexistence window (Pitfall 4) | Untested interaction risk concentrated into one deploy; a bug in one adapter's rewrite blocks/rolls back all three | Only if each adapter's `CampaignRun`-write path can be fully verified against real historical data in a staging DB before the single cutover — otherwise never |
| Increase SQLite's `timeout` and call the concurrency problem solved | Removes the immediate `database is locked` errors observed in testing | Masks, rather than fixes, an underlying overlap between the scheduler and staff UI writes; long enough contention still degrades UX (slow responses) even without an outright error | Acceptable as a first-line mitigation alongside — never instead of — no-overlap scheduling |
| Derive `run_status` from the single most-recently-created `ObservationRecord` rather than designing a real dominance-order aggregation | Fastest to implement; passes a simple single-record test | Produces Pitfall 6's regression the first time a run has 2+ records with different terminal outcomes — a near-certainty once robotic queue scheduling means every weathered night gets a rescheduled replacement record | Never for a run with more than one linked record; only defensible for a run design known to always have exactly one record (not this milestone's case) |
| Log the full `ObservationRecord.parameters` dict for debugging during the scheduling rollout | Speeds up diagnosing sync failures during initial rollout | Reopens the exact credential-leakage risk D-04 already fixed once, now surfacing through a new (scheduler) log destination (Pitfall 3) | Never in a code path that ships; acceptable only as a throwaway local `print()` during development, never committed |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|-------------------|
| LCO Observation Portal (live queue API, real credentials) | Treating a slow/timed-out API response the same as "no data" and letting the sync silently skip a record with no operator-visible signal, now that no human is watching stderr | Keep the existing skip-and-log-with-a-dedicated-counter pattern (`extraction_failed`/`telescope_api_failed` in `sync_lco_observation_calendar.py`) but route the per-run *summary line* (not just per-record stderr) to whatever failure-visibility mechanism the scheduling phase adds, so a run with an elevated failure count is flagged even though the command still exits 0 |
| Gemini ToO API | Assuming the one already-known secret field (`password`) is the only credential-shaped value that can appear in `record.parameters` from a new adapter code path | Re-audit `record.parameters` handling specifically for the new `CampaignRun`-writing code path added by the adapter rewrite — don't assume D-04's scrub was exhaustive for fields that didn't exist in the parameters shape it was written against |
| Task queue / cron scheduler itself (whichever the spike selects) | Assuming the scheduler's own failure (crashed worker, deleted crontab entry, disabled systemd timer) will be caught by the same in-job error handling that catches an LCO API failure | A dead-man's-switch/heartbeat check (see Pitfall 2) is required specifically because it is the only mechanism that also detects the scheduler failing to invoke the job at all, which in-job exception handling structurally cannot see |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Reconciler sweep (`reconcile_campaign_runs`) invoked on every scheduling cycle against every `CampaignRun` in the DB, not just runs the adapters just touched | Each sync cycle's wall-clock time grows with total historical `CampaignRun` count, not with new activity, even though `RECON-01`'s idempotency means most of that work produces `unchanged` results | Scope the scheduled reconciler invocation (or the adapters' own post-write call to `reconcile_run()`) to the runs actually created/updated in that cycle, mirroring how `campaign_views.py`'s four staff actions already call `reconcile_run()` on the single affected run rather than sweeping everything (RECON-08) | Becomes noticeable once the campaign/run count grows into the hundreds and the schedule runs every few minutes — a full sweep every cycle competes for the same SQLite write lock Pitfall 1 already flags |
| Per-record live telescope-API lookup (`resolve_placement_block`) run serially inside the sync loop, now on an unattended cadence tight enough to overlap the previous run's still-in-flight API calls | A scheduled run takes longer than the interval between scheduled runs, so two invocations of the same command run concurrently, doubling the SQLite write-lock contention risk from Pitfall 1 | The scheduling mechanism must guarantee no-overlap (a lock file, or the task queue's own single-concurrency-per-task-name setting) chosen explicitly in the investigation spike, not assumed from the schedule interval alone | Breaks as soon as API latency (real network conditions, not local dev) exceeds the chosen schedule interval — worth sizing against observed real API response times, not assumed local-dev speed |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Storing the unattended job's LCO/Gemini API credential in the same place/format as a per-request Django setting readable by the web process | A web-process compromise (e.g. an SSRF or template-injection bug elsewhere in this Django app) also exposes the scheduler's own service credential, not just user session data | Keep the scheduler's credential scoped as narrowly as the API allows (read-only queue-status scope if LCO/Gemini offer it) and stored via the same `os.getenv()`/`local_settings.py`-override convention this project already uses, not hardcoded or embedded in a scheduler-specific config file with looser permissions |
| Letting a scheduler's job-invocation logging (which some cron/task-queue configurations enable by default) capture the full command line | If credentials are ever passed as command-line arguments (even briefly, during development), they leak into scheduler job history/logs which often have weaker access control than application logs | Confirm the chosen scheduler's logging configuration does not record full command-line arguments, or ensure no adapter/scheduling code ever accepts a credential as a CLI argument in the first place |
| Auto-approving/auto-publishing whatever a scheduled sync creates, because "it's now automated so it must be trusted" | Undermines the existing approval-gate distinction (`CampaignRun.Source` only requires `approval_status` review for `WEB` submissions; CSV/LCO/Gemini-sourced runs are not treated as needing human approval) if the adapter rewrite accidentally routes scheduler-created runs through a path that skips whatever gating *should* apply to them | Confirm the adapter rewrite's `CampaignRun.source` value for each of the three adapters continues to receive exactly the approval treatment CANON-01/the `Source` `TextChoices` docstring already specifies — don't let "this now runs unattended" quietly change which runs require staff review |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| A run's calendar event and its `run_status` disagree during the migration/rollout window (e.g. the calendar still shows `[QUEUED]` from the old adapter-write path while the new outcome-propagation logic has already set `run_status=OBSERVED`) | Staff lose trust in the calendar as the single source of truth — the whole point of this milestone's canonical-record model | Cut over the title/status-vocabulary unification (STATUS-01/02) in lockstep with the adapter rewrite and outcome propagation, not as a separately-timed follow-up, so the calendar title prefix and `run_status` are always derived from the same underlying state at every point in the rollout |
| A `run_status` auto-derived from records changes with no visible explanation of *why*, right after staff manually set a status via `_set_run_status()`'s `mark_cancelled`/`mark_weather_failure` actions | Staff who deliberately marked a run cancelled/weathered see it silently revert or conflict once outcome propagation runs, undermining the manual-override affordance the v2.1/v2.3 weather-handling feature depends on | Define an explicit precedence rule: does a manual staff status override auto-derivation permanently, or only until the next record-outcome change? Document and test whichever is chosen — don't leave it as an emergent property of code execution order |
| Operators discover a scheduled sync has been silently failing only when a PI complains an observation isn't reflected on the calendar | Erodes confidence in "the feature-completeness bar for PR #43" this milestone exists to satisfy — the whole point is *removing* the need for someone to notice and run something manually | Ship the failure-visibility/heartbeat mechanism (Pitfall 2) in the same phase as the scheduling itself, and mention its existence in the operator runbook so staff know where to look |

## "Looks Done But Isn't" Checklist

- [ ] **Scheduled sync commands:** Often missing a no-overlap guarantee — verify two invocations of the same command can never run concurrently (lock file, or scheduler-native single-concurrency setting), not just that the command "runs on a schedule."
- [ ] **Failure visibility:** Often missing the case where the *scheduler itself* fails to invoke the job (not just the job failing once invoked) — verify a heartbeat/dead-man's-switch check exists, not only in-job try/except alerting.
- [ ] **Adapter rewrite (`CampaignRun`-writing):** Often missing a no-churn idempotency test against `CampaignRun`'s own fields — verify re-syncing identical upstream data twice produces zero field changes on the second run, mirroring the existing `CalendarEvent`-facing SYNC-04-style tests, not just "a `CampaignRun` gets created."
- [ ] **Dual-write migration window:** Often missing an explicit statement of what happens to not-yet-migrated adapters' direct writes during a sequential cutover — verify the phase plan states the cutover order and coexistence handling, not just "adapter now writes CampaignRun."
- [ ] **Outcome propagation:** Often missing the multi-record aggregation rule for a run with mixed-outcome linked records — verify a documented dominance-order table exists and is tested against a run with both a failed and a later successful record for the same window.
- [ ] **Outcome propagation, confirmed-link guard:** Often missing the check that status derivation reads only confirmed `CampaignRunObservation` rows — verify a test proves a high-scoring but unconfirmed attribution candidate never changes `run_status`.
- [ ] **Credential handling in the new unattended path:** Often missing a re-audit of what new job-level (not just per-record) credentials the scheduler introduces — verify no adapter/scheduler code logs a full exception message or command line that could carry one.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|-----------------|
| Duplicate calendar events from the migration dual-write window (Pitfall 4) | MEDIUM | Run the existing attribution queue (Phase 28) against the affected window — this is exactly the mechanism already proven on the pk=1 (FTS/MuSCAT4 vs. LCO queue) case; no new dedup tool needed, just staff confirmation time |
| A `run_status` regression from a bad aggregation rule (Pitfall 6) already shipped and misreported several runs | MEDIUM | Because `CampaignRunObservation` and its confirmed records remain the source of truth, a corrected aggregation function can be re-run against every run with 2+ linked records to recompute `run_status` — the fix is a one-off management command re-deriving status from the (already-correct) confirmed links, not a data-recovery exercise, as long as the underlying `CampaignRunObservation` rows were never themselves corrupted |
| SQLite lock contention causing intermittent staff-UI failures after scheduling went live (Pitfall 1) | LOW | Raise `OPTIONS['timeout']`, and add scheduler-level no-overlap enforcement if it was missing — no data recovery needed, since a lock timeout raises before any partial write commits |
| A scheduled job silently failed for an extended period with no alerting (Pitfall 2) | MEDIUM–HIGH depending on how long it went unnoticed | Once caught, run the affected adapter manually (as before this milestone) to backfill the missed window's `CampaignRun`s, then let the reconciler project the resulting calendar events — the manual-run fallback this milestone is designed to eliminate is also its own recovery path for a scheduling outage |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| SQLite write-lock collision (Pitfall 1) | Investigation spike + unattended-invocation/scheduling phase | `OPTIONS['timeout']` set in `settings.py`; a test or documented mechanism proves the scheduler never runs two overlapping instances of the same command |
| Silent scheduled-job failure (Pitfall 2) | Unattended-invocation/scheduling phase | A heartbeat/dead-man's-switch (or equivalent) mechanism exists and is exercised by a deliberately-failed test run; documented in the operator runbook |
| Credential leakage via unattended-job logs (Pitfall 3) | Unattended-invocation phase (job-level credentials) + adapter-rewiring phase (per-record field audit) | `git grep` for credential-shaped field names across the adapters' new write/log paths finds nothing unscrubbed; scheduler logging config confirmed not to echo full command lines |
| Dual-write duplicate/orphaned events during migration (Pitfall 4) | Adapter-rewiring phase | The phase plan states cutover sequencing explicitly; a test exercises the coexistence window (one migrated + one unmigrated adapter targeting the same night) and confirms no duplicate event is silently created without landing in the attribution queue |
| Adapter's idempotency key breaks against `CampaignRun`'s schema (Pitfall 5) | Adapter-rewiring phase | Per-adapter no-churn test against `CampaignRun` fields (re-sync identical data twice, assert zero field changes second time); confirms interaction with both existing `UniqueConstraint`s |
| One bad observation regresses a good run's status (Pitfall 6) | Outcome-propagation phase | A documented dominance-order table exists; a test with 2+ linked records at different terminal outcomes for the same run asserts the expected (non-regressed) `run_status` |
| Status derivation bypasses the human-confirmed link (Pitfall 7) | Outcome-propagation phase, sequenced after the attribution queue (Phase 28) already exists | A test proves a high-scoring but unconfirmed `campaign_attribution.py` candidate never changes `run_status`, and that a run with zero confirmed `CampaignRunObservation` rows is never auto-derived |

## Sources

- Primary (HIGH confidence, direct repo inspection): `/home/tlister/git/fomo_devel/solsys_code/campaign_reconciler.py`, `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/models.py` (`CampaignRun`, `CampaignRunObservation`, dismissal models), `src/fomo/settings.py` (`DATABASES`), `.planning/PROJECT.md` (v2.2 SHIPPED RECON-01..09/ATTRIB-01..06 section, v2.3 Current Milestone section).
- [Django, SQLite, and the Database Is Locked Error](https://blog.pecar.me/django-sqlite-dblock/) — MEDIUM confidence, general Django/SQLite timeout guidance.
- [OperationalError: database is locked Python SQLite [Solved]](https://bobbyhadz.com/blog/operational-error-database-is-locked) — MEDIUM confidence, general SQLite locking behavior.
- [Dead man's switch, explained for developers (and how to actually use one)](https://crontap.com/blog/dead-man-switch-explained-for-developers) — MEDIUM confidence, general cron/scheduler monitoring pattern.
- [5 Ways Your Cron Jobs Are Failing Silently (and How to Catch Them)](https://dev.to/deadping/5-ways-your-cron-jobs-are-failing-silently-and-how-to-catch-them-2njp) — MEDIUM confidence, general cron failure-mode catalogue.

---
*Pitfalls research for: FOMO v2.3 — unattended scheduling + canonical-record adapter rewiring + outcome propagation*
*Researched: 2026-09-01*
