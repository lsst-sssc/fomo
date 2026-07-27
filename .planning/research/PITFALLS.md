# Pitfalls Research

**Domain:** Consolidating a canonical run record + idempotent derived-projection reconciler, retrofitted onto a live Django/TOM-Toolkit system with real operator data (FOMO v2.2 "One Canonical Run Record")
**Researched:** 2026-07-26
**Confidence:** HIGH (grounded directly in `solsys_code/models.py`, `solsys_code/campaign_views.py`, `solsys_code/calendar_utils.py`, `solsys_code/admin.py`, `tom_calendar.models.CalendarEvent`, and the measured dev-DB hazards in PROJECT.md's Current Milestone section) / MEDIUM on the general reconciler-pattern claims (cross-checked against the Kubernetes controller literature, which is the closest well-documented analogue to what the spike needs to design)

This file supersedes the v2.1-scoped `PITFALLS.md`. It is scoped to the six v2.2 target features named in `.planning/PROJECT.md`'s Current Milestone section: the `CalendarEventTelescopeLabel` companion-record generalisation, `source`/`telescope_class` on `CampaignRun`, `ObservationRecord` linkage, operator-assisted attribution, the idempotent four-stage reconciler, and the settling spike.

## Critical Pitfalls

### Pitfall 1: Renaming `CalendarEventTelescopeLabel` breaks one of its four live call sites silently

**What goes wrong:**
The plan says "generalise `CalendarEventTelescopeLabel` into a companion record." If that's implemented as a Django `RenameModel` (or a drop-and-recreate under a new name), any of the four concrete places that reference the old name or its `related_name='telescope_label_meta'` can silently stop working instead of erroring: `solsys_code/admin.py` (`admin.site.register(CalendarEventTelescopeLabel, ...)`), `solsys_code/management/commands/sync_lco_observation_calendar.py` (`CalendarEventTelescopeLabel.objects.update_or_create(...)`, line 369), `solsys_code/views.py`'s `fomo_render_calendar` (`.prefetch_related('telescope_label_meta')`, DISPLAY-09's N+1 fix), and `src/templates/tom_calendar/partials/calendar.html` (`event.telescope_label_meta.is_verified`, lines 228/244). A template reference to a renamed reverse accessor doesn't raise — Django templates resolve missing attributes to empty string, so the dashed-border/tooltip fallback-label UI (DISPLAY-01/02/03) just silently stops showing the fallback indicator for every future sync, with no test failure unless there's an explicit assertion on rendered HTML content (which `test_calendar_template.py` does have — but only if that test file itself is updated to reference the new name/accessor).

**Why it happens:**
`RenameModel` migrations feel mechanical (Django's migration autodetector proposes them and they "just work" for the FK/PK plumbing), so the temptation is to trust the migration to handle every consumer. It handles the DB schema and the model graph; it does **not** touch template strings, admin registrations that reference the class by import, or `related_name` strings that are hardcoded in `.prefetch_related()`/template lookups rather than derived from the model's `Meta`. Known Django tickets confirm `RenameModel` has historical gaps even in the parts it does own (M2M/through-table columns, `related_name='+'` FKs) — the four non-migration consumers here are entirely outside its scope.

**How to avoid:**
- Grep-verify every consumer before writing the migration: `grep -rn "CalendarEventTelescopeLabel\|telescope_label_meta"` across `.py`, `.html` (already done for this research — see the four sites above) and treat that as the fixed checklist for the phase.
- Decide explicitly whether `related_name='telescope_label_meta'` is kept as-is (safest — zero template/view changes) or renamed; if renamed, it must change in the model, the template (2 lines), and `views.py`'s `prefetch_related()` string in the same commit, and the existing `test_calendar_template.py`/`test_sync_lco_observation_calendar.py` sidecar-write tests must be run (not just added to) to catch a stale reference.
- Prefer Django's `RenameModel` operation (which preserves the underlying table via `ALTER TABLE RENAME`, keeping existing rows and their PK values — the `CalendarEvent` OneToOne PK relationship survives untouched) over drop-and-recreate, specifically because this table has live production rows keyed on `event_id`.
- Add or extend a regression test that renders the calendar template against a `CalendarEventTelescopeLabel`-equivalent row with `is_verified=False` and asserts the dashed-border class is present — this is the one consumer (the template) that a Python-level `manage.py check` cannot catch.

**Warning signs:** `ruff`/`manage.py check` pass but `./manage.py test solsys_code` regresses `test_calendar_template.py` or the sidecar-write tests in `test_sync_lco_observation_calendar.py`; or worse, tests stay green but a manual UAT pass shows fallback-labeled events losing their dashed border (exactly the class of gap Phase 22's human UAT caught previously in this project).

**Phase to address:** The spike phase (settling the migration strategy) should enumerate this exact 4-site checklist as an explicit spike deliverable; the implementation phase that performs the rename should treat "all 4 sites updated in the same commit, `related_name` decision documented" as a must-have, not a follow-up.

---

### Pitfall 2: Adding the `run` FK to the companion record as `on_delete=CASCADE` or NOT NULL

**What goes wrong:**
The companion record's new `run` field must be a **nullable** FK to `CampaignRun`, because most existing `CalendarEventTelescopeLabel` rows (all LCO/SOAR/Gemini/classical-sync-created events) have no `CampaignRun` at all — they come from the other three ingest paths this milestone deliberately does not touch (`sync_lco_observation_calendar`, `sync_gemini_observation_calendar`, `load_telescope_runs`). If the migration adds `run` as `null=False` with a naive default, either the migration fails outright (SQLite/Postgres both reject a NOT NULL add without a default when rows exist) or every existing row gets force-linked to an arbitrary/placeholder run, corrupting the very attribution this milestone is supposed to establish carefully. Separately, `on_delete` on this FK matters in the wrong direction from the sidecar's own `event` FK: if a `CampaignRun` is ever deleted (not currently exposed in the UI, but `CampaignRunAdmin` doesn't forbid deletion) with `on_delete=CASCADE` on the `run` field, every companion record — and by extension the operator's verified/fallback label history for those events — disappears along with it, when the calendar event itself should almost certainly survive.

**Why it happens:**
The instinct when adding "the FK that makes the relation real" is to reach for the same `on_delete=CASCADE` pattern already used for `event` (which is correct there, because the sidecar's whole reason to exist is 1:1 with the event). But `run` is a different direction of ownership — the companion record's lifecycle is owned by the event, not by the run, and a run's deletion should not be able to destroy the operator's attribution/verification history silently.

**How to avoid:**
- `run = models.ForeignKey(CampaignRun, null=True, blank=True, on_delete=models.SET_NULL, related_name='calendar_labels')` (or equivalent) — nullable so the migration is a no-op data-wise for every pre-existing row, `SET_NULL` so deleting a `CampaignRun` un-links the calendar bookkeeping instead of deleting it.
- Write the migration as two Django operations in sequence (`AddField` with `null=True`, no backfill in the same migration) — do not try to backfill `run` values as part of the schema migration; attribution is a separate, operator-assisted, auditable step (see Pitfall 5), not something a migration should guess.
- Add a test asserting no migration in this milestone ever ships a default that links `run` to a real `CampaignRun` pk without an explicit operator action.

**Warning signs:** Migration file contains a `default=` on the new FK field, or a `RunPython` step in the *same* migration that assigns `run` based on any heuristic (date overlap, telescope-instrument string match) — that heuristic belongs in the attribution tool, reviewable and undoable, not baked into a one-shot migration.

**Phase to address:** Schema/migration phase (companion-record generalisation). The spike should explicitly decide `on_delete` and nullability before the migration is written, and the migration-writing phase's plan should show `AddField(null=True)` with zero `RunPython` backfill logic.

---

### Pitfall 3: `source` field's "value for pre-existing rows" collides with the two partial unique constraints

**What goes wrong:**
`CampaignRun` currently has two mutually-exclusive partial unique constraints keyed on `(campaign, telescope_instrument, window_start, window_end)` and `(campaign, telescope_instrument, contact_person)`, chosen specifically as the natural key for detecting a re-import/re-submission of the *same* run. Every existing row (imported CSV, web-submitted, or otherwise) currently satisfies exactly one of those constraints without `source` in the key at all. If `source` is added as a genuinely new discriminator dimension of "sameness" (e.g. because a future adapter — the LCO/Gemini/classical ingest rewiring deferred to v2.3 — will write its own `CampaignRun` rows keyed by the *same* `(campaign, telescope_instrument, window)` tuple as an existing CSV-imported row for the same physical run, which is explicitly the pk=1/11-LCO-events scenario), then two outcomes are both wrong: (a) leaving `source` out of the constraint means a future adapter's write of "the same run, but now recorded as `source=lco_queue`" collides with the existing CSV-imported row and either raises `IntegrityError` or (under `get_or_create`) silently mutates the CSV row's `source` value, corrupting provenance; (b) adding `source` into the constraint fields means the *existing* rows (which all predate `source` and would all get the same backfilled default, e.g. `'csv_import'` or `'unknown'`) now have a wider natural key, which is fine for existing rows but does nothing to prevent the pk=1-style double-representation this milestone exists to fix — the LCO-queue-derived run, if ever materialized as a `CampaignRun`, would get `source='lco_queue'` and sail right past the constraint as "clearly a different row," never surfacing as a candidate for the operator-assisted attribution pass at all.

**Why it happens:**
`source` reads as "just another column" but it is semantically an identity dimension for some future adapters and purely descriptive metadata for others (a web submission's `source` is always `web`, never ambiguous). Treating it uniformly in the constraint is the natural first instinct and is wrong for exactly the case (LCO/Gemini/classical-derived runs matching an existing manually-entered run) this milestone is designed around.

**How to avoid:**
- Backfill value for existing rows: this milestone's target features already imply the answer — every row predating this milestone was either CSV-imported or web-submitted; `import_campaign_csv.py` and `CampaignRunSubmissionView` are the only two writers of `CampaignRun` today. The migration should set `source` from a `RunPython` step that inspects which of those two code paths plausibly created the row (e.g. web submissions are distinguishable by having gone through the pending-review flow, or more robustly, a single explicit sentinel value like `'legacy_unknown'` for all pre-milestone rows if per-row inference is too unreliable to trust — see the Technical Debt table below). Never leave `source` NULL for old rows if the reconciler or approval logic branches on it (`approval_status` becomes meaningful only for web submissions — a NULL `source` makes that branch undefined for every pre-existing row).
- Do NOT add `source` to either existing `UniqueConstraint`'s field list. Keep `source` purely descriptive/provenance metadata; let the *attribution* mechanism (a separate nullable link or merge marker) be the place two same-physical-run rows from different sources get connected, not the natural-key constraint. This preserves the existing constraints' job (detecting a duplicate *re-import from the same source*) while leaving cross-source duplicate detection to the explicitly human-reviewed attribution pass — matching the milestone's own framing ("The fix is attribution, not deduplication").
- If `telescope_class` similarly needs constraint treatment (a class-wide run and a site-resolved run for the same telescope/window are legitimately different rows, per the milestone's stated purpose for the field), verify explicitly against the pk=1/LCO-events scenario before deciding either way — write a test that imports both representations and asserts they do NOT collide.

**Warning signs:** A migration or spike doc proposes putting `source` in `unique_campaign_run_resolved_window`'s field list "for correctness" — that's the moment to stop and re-derive the pk=1/LCO-11-events scenario against the proposed constraint by hand.

**Phase to address:** The spike phase must explicitly answer this (it is literally one of the spike's stated deliverables — "the migration and attribution strategy"). The migration-writing phase should include the pk=1 double-representation scenario as an executable test case: import/create both representations and assert the intended outcome (two distinct rows, no `IntegrityError`, both flagged as attribution candidates).

---

### Pitfall 4: Backfill/attribution auto-links records to the wrong parent run using a loose heuristic

**What goes wrong:**
The obvious first cut at "operator-assisted attribution" is a script that proposes matches by overlapping `(telescope_instrument, date range)` and lets staff bulk-confirm. Given the measured hazards, two distinct failure shapes are already latent in the real data: (1) **false merge** — 3I/ATLAS's 19 site-resolved `CampaignRun`s and 20 Didymos-2026 events are for *different targets*; a heuristic keyed only on telescope+date (not target/proposal) could propose cross-target matches if two campaigns share a telescope in overlapping weeks, and if the UI makes "confirm" one click per page of suggestions, staff fatigue makes false-positive bulk-confirmation a real risk; (2) **false split / missed match** — pk=1 (FTS/MuSCAT4, Siding Spring E10, 7–21 July) vs. 11 LCO queue events (2m0/2M0-SCICAM-MUSCAT, 7–20 July) is the *same* run, but the run's `telescope_instrument` string (free text) and the events' `telescope`/`instrument` fields (populated from `SITE_TELESCOPE_MAP` coarse labels or verified LCO API values) are not the same string, and their date ranges are off by one day (21 vs 20 July) — a naive exact-match or exact-date-range heuristic finds *zero* candidates for the one case the milestone explicitly calls out as its motivating example, while an over-loose fuzzy heuristic risks proposing wrong matches elsewhere.

**Why it happens:**
Attribution heuristics get validated against the easy cases (exact string match, exact date match) because those are what's convenient to unit-test, and the hard case — the one that actually motivated the milestone — is exactly the case where the naive heuristic fails silently (produces no suggestion, or a low-confidence one buried below higher-confidence noise).

**How to avoid:**
- Design the candidate-generation heuristic against the pk=1 example specifically as a fixture from day one (not as an afterthought) — the milestone context already hands you the exact real pks/labels/date-off-by-one; use it as the acceptance test for "does the attribution tool even surface this."
- Never auto-confirm. The milestone context is explicit — "never a silent merge" — so the write path must require an explicit staff action per proposed link (or an explicit "confirm all N" only after individually rendering each with its evidence: matched telescope/instrument, date overlap, target/campaign check), and the write must be logged with who confirmed it and when (a plain `confirmed_by`/`confirmed_at` pair on the attribution record, or Django's admin log, is enough — it doesn't need a full audit-log framework).
- Make target/campaign a hard filter, not just a scoring signal — never propose a `CalendarEvent`/`ObservationRecord` as belonging to a `CampaignRun` whose campaign's targets don't include (or aren't consistent with) whatever target information the event/record carries, even loosely. This directly prevents the 3I/Didymos cross-contamination case.
- Score/present date-overlap and instrument-string-similarity as separate, visible signals (not collapsed into one opaque score) so staff reviewing pk=1-style off-by-one-day, different-string-representation matches can see *why* the tool thinks they're the same run, rather than trusting a black-box confidence number.

**Warning signs:** The attribution tool's test suite only exercises exact-match fixtures; a dry run against the real dev DB produces zero suggestions for the pk=1/11-LCO-events pair (the one case everyone already knows should match); or the UI has a single "confirm all" button with no per-row evidence.

**Phase to address:** The operator-assisted-attribution phase. Its plan should list the pk=1 scenario as an explicit acceptance criterion, and its "must-have" verification should include "the confirmation action is per-candidate and logged, never bulk-silent."

---

### Pitfall 5: Attribution has no undo, so a wrong confirmation is permanent

**What goes wrong:**
If confirming an attribution suggestion directly sets the FK (`run` on the companion record, or the M2M/FK from `ObservationRecord`) with no history and no way to un-link, a staff mis-click during a bulk-review session (plausible given the 19+20 event volume already measured) permanently and silently corrupts provenance — and because the reconciler in this same milestone is designed to project calendar events *from* run state, a wrong attribution doesn't just mislabel one row, it can cause the reconciler to modify/delete a `CalendarEvent` that never should have been touched (see Pitfall 8).

**Why it happens:**
"Just set the FK when staff clicks confirm" is the minimal implementation, and undo is the kind of feature that's easy to defer as "we'll add it if someone asks."

**How to avoid:**
- Every attribution write should be reversible through the same UI that made it — a plain "unlink" action (set `run` back to `None`) requires no new infrastructure since the field is already nullable (Pitfall 2), and should be exposed next to every confirmed attribution, not hidden in Django admin.
- Log the attribution action itself (who, when, what was linked) separately from the current FK value, so even after an unlink there's a record that a mistake happened and was corrected — this can be as simple as Django's built-in `LogEntry` (already used implicitly via `ModelAdmin`) if the confirm action goes through admin, or an explicit small audit model if it goes through a custom staff view (which, given `CampaignRunDecisionView`'s precedent in this codebase, is the more likely shape).
- Treat "confirmed" as a separate boolean from "linked" if useful — i.e., a suggested-but-unconfirmed link and a staff-confirmed link should be visually and behaviorally distinct, so the reconciler (once it exists) can be scoped to only touch confirmed links deterministically.

**Warning signs:** The attribution UI's plan has no "undo"/"unlink" affordance listed as a requirement; code review finds the confirm action is a plain `.update(run=candidate)` with no logging.

**Phase to address:** Same phase as attribution (Pitfall 4) — undo is not a separate feature, it's a correctness requirement of the same deliverable.

---

### Pitfall 6: The reconciler is not actually idempotent — reruns create duplicate or drifting events

**What goes wrong:**
The four-stage window pipeline (twilight night -> class-wide day -> scheduled record window -> completed range) means a single run's *correct* event window changes over its lifecycle. A reconciler that recomputes "what should exist" and then does a plain `create()` (rather than routing every write through the existing `insert_or_create_calendar_event()` no-churn find-or-create, which already solves exactly this problem for the three ingest commands) will duplicate events on every rerun, or — if it does update but forgets that stage transitions change the *key* (bare `CAMPAIGN:{pk}` vs. per-night `CAMPAIGN:{pk}:{date}` vs. whatever key scheme records at stage 3/4 use) — will leave orphaned events from an earlier stage sitting alongside the new ones instead of replacing them. `_project_calendar_event()`'s existing per-night key scheme (`CAMPAIGN:{pk}` for a single night, `CAMPAIGN:{pk}:{date.isoformat()}` for a range) already shows this is a real, non-trivial problem *within a single stage* (ground vs. satellite branch, single-night vs. range) — a four-stage pipeline multiplies the ways the key scheme can drift between reconciler runs.

**Why it happens:**
"Idempotent" gets treated as "safe to call `insert_or_create_calendar_event()` for what I compute today," but true idempotency for a *multi-stage* projection also requires reconciling away events from a stage the run has since moved past (e.g. a run that had a stage-1 twilight event, then acquired a scheduled `ObservationRecord` and should now show the stage-3 narrowed window instead) — a naive reconciler that only ever calls `insert_or_create_calendar_event()` for the *current* stage's key will leave the stage-1 event behind as a stale duplicate, because its key never gets touched again.

**How to avoid:**
- Design one canonical key scheme per `CampaignRun` that's stable across stage transitions (e.g. always `CAMPAIGN:{pk}` for a single-night run or `CAMPAIGN:{pk}:{date}` per night, regardless of which pipeline stage currently determines that night's window) so a stage transition is an *update* to the same row via `insert_or_create_calendar_event()`, never a new key — this is exactly the "no-churn create-or-update" contract that function already implements; reuse it rather than reimplementing reconciliation logic inline in the new command.
- If a stage transition genuinely requires a different *number* of events (e.g. stage 2's single class-wide day collapses into stage 3's single narrowed-to-a-record window — one event either way, just a different window) versus stage 1's site-specific twilight window for potentially multiple nights, write an explicit test matrix: for every `(previous_stage, new_stage)` pair, assert the reconciler ends with exactly the expected event set (no leftover keys from the previous stage, no duplicate keys for the new stage).
- Run the reconciler twice in a row in every test and against the same input and assert zero `CalendarEvent.objects.count()` change and zero `modified` timestamp change on the second run (see Pitfall 7) — this is the cheapest, highest-value regression test for the whole feature and should be a phase must-have, not incidental coverage.

**Warning signs:** `CalendarEvent.objects.filter(url__startswith='CAMPAIGN:{pk}')` grows over successive reconciler runs against unchanged run state; or a manual test that walks a single run through all four stages ends with more than the expected number of events for that run.

**Phase to address:** The reconciler phase itself — must-have acceptance criteria should explicitly include "two consecutive runs against unchanged state produce zero writes" and "a full stage-1-through-4 walk of one run ends with exactly the events the current stage implies, no stage-N-1 leftovers."

---

### Pitfall 7: Reconciler churns `modified` timestamps on every run, defeating the "no-churn" idempotency it's supposed to have

**What goes wrong:**
`tom_calendar.CalendarEvent` has `modified = models.DateTimeField(auto_now=True)`. Any `.save()` call — even one that writes back the exact same field values it read — bumps `modified`. The existing `insert_or_create_calendar_event()` helper already guards against this (its docstring explicitly frames the contract as "leave it unchanged if no fields differ," and its return value distinguishes `'unchanged'` from `'updated'`) precisely because SYNC-04/CAL-03/GEM-NOCHURN-01 all required it — spurious `modified` churn was already identified as a real problem in this codebase for the three existing sync commands. A reconciler that recomputes derived fields with even slightly different values each run — e.g. recomputing `sunset`/`sunrise` via `sun_event()`, which the codebase's own comments note can drift by a second or two between runs as astropy's IERS Earth-orientation data refreshes (this drift is explicitly documented as the reason `insert_or_create_calendar_event()`'s `start_time_tolerance` proximity-match parameter exists for `load_telescope_runs`) — will silently reintroduce this exact bug for the reconciler's stage-1/2 window computation unless it reuses the same tolerance-aware comparison, not a fresh exact-equality diff.

**Why it happens:**
The reconciler is new code, and it's easy to write its own comparison logic ("does this field differ from what's in the DB") without noticing that the *existing* comparison logic already has hard-won tolerance handling baked in for exactly this astronomy-specific problem (float/time drift between independent computations of "the same" sunset).

**How to avoid:**
- Route every reconciler write through `insert_or_create_calendar_event()` (with `start_time_tolerance` set for any stage whose window comes from `sun_event()`) rather than writing new create/update logic — this is a "don't hand-roll" instance the codebase already flags as a convention (`campaign_views.py`'s own comments: "Never construct CalendarEvent directly -- always route through the shared helper").
- Add an explicit regression test: run the reconciler twice against the same `CampaignRun` state with `sun_event()` computations that differ by a couple of seconds between calls (mock or tolerate real IERS drift) and assert `modified` is unchanged on the second run.
- Treat spurious `modified` churn as a genuine defect class in code review for this phase, not a cosmetic nit — downstream consumers (calendar UI "recently changed" sorting, any future webhook/notification on event changes) depend on `modified` meaning "this event's displayed content actually changed."

**Warning signs:** A reconciler dry-run against the live dev DB (19 site-resolved 3I runs is a real fixture for this) shows non-zero updates on a second consecutive run with identical `CampaignRun` state.

**Phase to address:** Reconciler phase; explicitly reuse (don't reimplement) `insert_or_create_calendar_event()`'s no-churn contract as a stated design constraint in that phase's plan.

---

### Pitfall 8: Reconciler deletes or mutates a `CalendarEvent` it does not own

**What goes wrong:**
`CalendarEvent` is general-purpose — conferences and proposal deadlines are events with no `run` at all, and today's LCO/Gemini/classical sync commands also write `CalendarEvent`s with no `CampaignRun` link (that link doesn't exist yet in the data model, and even after this milestone, those adapters are explicitly deferred to v2.3 to rewire). A reconciler that, in service of "idempotency," tries to clean up events under some derived key pattern (e.g. "delete any `CAMPAIGN:{pk}:*` event that no longer matches the run's current stage window") is safe *only* as long as its delete/update scope is provably restricted to events it created or owns. If the reconciler's cleanup logic is instead keyed on anything looser — a date-range sweep, a telescope-instrument match, or (worse) a blanket "recompute everything under this campaign's target" — it risks touching or deleting a hand-created conference event, a proposal deadline, or (until v2.3's adapter rewiring ships) an LCO/Gemini/classical-sync-created event that happens to share a telescope and date window with a `CampaignRun` the reconciler is processing — exactly the pk=1/11-LCO-events collision the milestone is built around, except now automated and destructive instead of a one-time double-representation to be manually attributed.

**Why it happens:**
"Idempotent" reconcilers (by analogy with Kubernetes controllers, the closest well-documented pattern for this exact problem) are expected to reconcile toward a *complete* desired state, which naturally suggests "delete anything I find that shouldn't be there" — but that's only safe when every managed resource carries an unambiguous ownership marker the reconciler can filter on. In Kubernetes this is an owner-reference/label; here, an unattributed, un-owned `CalendarEvent` has no such marker at all pre-attribution, and post-attribution the marker is exactly the nullable `run` FK this milestone is adding.

**How to avoid:**
- The reconciler must scope every write/delete strictly to `CalendarEvent`s whose companion record's `run` FK equals the `CampaignRun` currently being reconciled (or, for creation, to keys it itself derives deterministically from that run's pk, per Pitfall 6) — never a broader match on telescope/instrument/date.
- Never delete an event outright to "clean up" a stage transition unless it is provably one the reconciler itself created (traceable via the `run` FK) — prefer updating an existing owned event in place (per Pitfall 6's stable-key design) over delete-and-recreate, and if an owned event genuinely needs to disappear (e.g. a run reverts from stage 3 back toward stage 1 because a scheduled record was cancelled), require that the deletion path is exercised by an explicit test proving it only ever removes events keyed to that one run's pk.
- The pk=1/11-LCO-events collision is the canonical test case here too: before attribution, the reconciler processing `CampaignRun` pk=1 must not touch, merge, or delete any of the 11 unrelated `sync_lco_observation_calendar`-owned events, even though they represent "the same" physical telescope time — only after an explicit attribution confirms the link should the reconciler be allowed to treat them as related, and even then the milestone's own framing is "attribution, not deduplication," implying the reconciler still should not delete the sync-command-owned events even post-attribution.
- Extend `insert_or_create_calendar_event()`'s contract (or wrap it in a reconciler-specific helper) so a "no matching owned event, nothing to do" outcome is explicit and distinguishable from "found and updated" — this makes the ownership-scoping testable in isolation.

**Warning signs:** Reconciler test suite has no fixture combining an unrelated hand-created `CalendarEvent` (no run link, e.g. a conference) in the same date range as a `CampaignRun` under test and asserting it survives untouched; or a dry-run reconciler pass against the live dev DB reports any delete/update touching an event whose `url` doesn't start with a key scheme traceable to the run being processed.

**Phase to address:** Reconciler phase — this should be the single most heavily tested must-have (ownership-scoped mutation, proven against a fixture that deliberately includes unowned events in the same window) given it's explicitly called out as a live hazard in the milestone context.

---

### Pitfall 9: Unbounded/eager event creation from a wide or unresolved window

**What goes wrong:**
Stage 2's class-wide window (00:00–23:59 that day, telescope-class-only resolution, "many sites worldwide") is already a case where the reconciler cannot pick one site — but if it errs toward "create an event per candidate site" instead of one class-wide event, a single class-wide run could explode into dozens of events (one per site that has a telescope of that class). Separately, if the reconciler's window pipeline is driven off a naive `for date in date_range` loop without an upper bound (mirroring the existing `_project_calendar_event()`'s `n_nights = (window_end - window_start).days + 1` loop, which today has no sanity cap), a `CampaignRun` with a data-entry error in `window_end` (e.g. a typo'd year making the window decades long) would create an unbounded number of events synchronously in one reconciler pass, which is both a performance and a data-integrity hazard — and unlike the current single approve-click flow, a reconciler that runs unattended/on a schedule means nobody is watching in real time when it happens.

**Why it happens:**
Existing code (`_project_calendar_event()`) already has this exact unbounded-loop shape and has worked fine so far because run windows in the real 3I/ATLAS and awarded-Gemini-time data have been observationally small (days, not years) — but the reconciler generalizes this to run automatically and to every `CampaignRun` in the system, raising both the likelihood of hitting a bad row and the blast radius of not noticing.

**How to avoid:**
- Add an explicit sanity bound on window length (e.g. reject/flag/log-and-skip rather than silently create hundreds of events for any single run whose `window_end - window_start` exceeds a generous but finite threshold — a month or two is already generous for any known real use case in this system) as a defensive guard in the reconciler, independent of whatever the UI-side form validation does today (form validation doesn't protect CSV-imported or attributed-from-legacy-data rows).
- For the class-wide (stage 2) window specifically: confirm explicitly in the spike whether "many sites worldwide" means one event per candidate site or one event with no fixed site (the milestone's stage-2 description — "00:00→23:59 that day" with no site qualifier — reads as the latter, a single class-wide event, matching how `telescope_class` is described as distinguishing "a legitimately class-wide run from a site that failed to resolve" rather than as a multi-site fan-out) — get this into the spike's decision doc explicitly rather than leaving it to be inferred during implementation.
- Make the reconciler's per-run event count part of its own logging/summary output (mirroring `import_campaign_csv`'s created/updated/skipped summary convention already established in this codebase) so an operator running it can immediately see an anomalous count for any one run.

**Warning signs:** A reconciler dry-run against the dev DB shows any single run's event count in the double or triple digits; no upper-bound test exists for window length.

**Phase to address:** Reconciler phase (bound enforcement) and the spike (resolving the stage-2 fan-out question before implementation starts).

---

### Pitfall 10: Reconciler partial failure leaves a run half-projected with no visible retry surface

**What goes wrong:**
`_project_calendar_event()`'s existing range-window loop already documents this exact tradeoff deliberately: "A mid-loop `sun_event()` `ValueError` re-raises immediately, leaving any already-created earlier nights' events in place — accepted partial projection, no `transaction.atomic()` wrap." That's a reasonable choice for a single approve-click UI action (the run stays in the "Sites Needing Review" queue as its retry surface, per `_resolve_site()`'s design). A reconciler processing *every* `CampaignRun` in one batch run needs the equivalent retry surface at the *reconciler's* level, not just the per-run level — if one run's mid-loop failure (e.g. the already-known blank-`Observatory.timezone` hazard) aborts the whole batch instead of being caught, logged, and skipped so the batch continues to the next run, then a single bad row (already known to exist in this dev DB — the FTN-timezone-gap rows `backfill_range_calendar_events` already had to gracefully skip) silently blocks reconciliation for every *other* run in the same invocation, and there's no per-run "needs review" surface analogous to today's `site_needs_review` flag for the *reconciler's* failure mode specifically (as opposed to the unresolved-site failure mode it already covers).

**Why it happens:**
The single-run code this reconciler generalizes was written and tested against one-run-at-a-time flows (approve click, resolve_site retry, one-off backfill command with `--dry-run`); scaling that to "reconcile every run, unattended, on a schedule (or on every trigger)" introduces a batch-failure-isolation requirement that didn't exist before and is easy to miss if the reconciler is implemented as a thin loop calling the existing per-run projection logic unchanged.

**How to avoid:**
- Wrap each run's reconciliation in its own try/except inside the batch loop (mirroring the graceful-skip pattern `backfill_range_calendar_events` already established for the timezone-gap case, FIX-08) so one run's failure logs and continues rather than aborting the batch.
- Track and surface a per-run "reconciliation failed, needs attention" state distinct from the existing `site_needs_review` — reuse the `Observatory`-timezone-gap example as a concrete test fixture, since it's a documented, currently-unfixed real data issue in this very database.
- Report a summary (processed/succeeded/failed/skipped counts, and which runs failed) at the end of every reconciler invocation, matching the established convention (`import_campaign_csv`, `backfill_range_calendar_events`) rather than silent success/failure.
- Decide explicitly (spike-level question) whether the reconciler is triggered per-run (e.g. on approve, on record-schedule-change) or as a full-database sweep, or both — the partial-failure isolation requirement differs in shape between the two (a per-run trigger only ever risks one run's partial projection, matching today's behavior; a full sweep needs the batch-isolation design above).

**Warning signs:** Reconciler implementation is "one function, no try/except around the per-run loop body"; no test exercises "one run raises mid-batch, assert every other run still gets processed."

**Phase to address:** Reconciler phase; the spike should settle the per-run-vs-sweep trigger question, since it changes the shape of this pitfall's fix.

---

### Pitfall 11: `approval_status` "becomes meaningful only for web submissions" is read inconsistently across old and new code paths

**What goes wrong:**
Today, `approval_status` is a single field with uniform meaning for every `CampaignRun` regardless of how it was created — every existing consumer (`CampaignRunTableView`'s D-09 non-staff filter excluding `PENDING_REVIEW`, `ApprovalQueueView`'s pending/decided split, `gap_analysis_available()`, the admin's read-only-field guard) treats it as globally meaningful. If `source`/`approval_status` semantics change so that non-web-submission runs (CSV import, future LCO/Gemini/classical adapters) are meant to bypass approval entirely (e.g. auto-`APPROVED`, or the field becomes irrelevant/ignored for them), every one of those existing consumers needs to be re-audited against the new semantics, or a CSV-imported run that defaults to `PENDING_REVIEW` (today's model default, unconditionally) becomes permanently invisible to non-staff on the per-campaign table (D-09's exact filter) even though nobody is ever going to "approve" it through the queue UI, since CSV import doesn't create pending-review rows the human approval flow expects to process.

**Why it happens:**
"Approval required only for web submissions" is a clean product statement but touches a field with several milestones' worth of accreted consumers that all assumed uniform semantics; a change here is a classic "one field, many silent readers" situation, and the risk is asymmetric — old CSV-imported rows silently stuck in a `PENDING_REVIEW` limbo they can never exit (because nobody reviews them) is a worse and quieter failure than a new bug that throws visibly.

**How to avoid:**
- Change the *default value* CSV import and future non-web adapters write, not the meaning readers assign to the field — i.e., have `import_campaign_csv.py` (and any future adapter) explicitly set `approval_status=APPROVED` at creation time for non-web sources, rather than trying to teach every reader "ignore this field when source != web." This keeps every existing consumer's logic correct unchanged, and is a much smaller, more auditable diff.
- Explicitly re-run (not just re-read) the existing SUBMIT-02/D-09 non-staff-visibility tests against a CSV-imported-source fixture to prove old and new source rows behave identically at every visibility boundary.
- For the migration backfill of `source` (Pitfall 3), backfill existing rows' `approval_status` unchanged — do not retroactively flip existing `PENDING_REVIEW` CSV-imported rows to `APPROVED` as part of the same migration; that's a policy change that deserves its own explicit, reviewed step (if wanted at all), not a side effect of adding a column.

**Warning signs:** A CSV import test fixture that used to assert `approval_status == PENDING_REVIEW` (today's model default) starts silently getting a different value with no corresponding code change flagged in review; or a manual UAT check shows CSV-imported 3I/ATLAS runs vanishing from the non-staff campaign table view after this milestone ships.

**Phase to address:** The `source`/`telescope_class` field-addition phase — should include an explicit decision + test for what `approval_status` value each non-web adapter writes going forward, separate from (but adjacent to) the migration backfill decision in Pitfall 3.

---

### Pitfall 12: `ObservationRecord` FK/M2M direction and cascade choice orphans records or blocks run deletion

**What goes wrong:**
Both `CalendarEvent` and `ObservationRecord` are third-party models (`tom_calendar`, `tom_observations`), so — as the milestone context already notes — the link must live on FOMO's side. The natural implementation is a `ForeignKey`/`ManyToManyField` declared on `CampaignRun` pointing at `ObservationRecord` (or a FOMO-side through-model, mirroring the `CalendarEventTelescopeLabel` companion-record pattern). Two concrete cascade mistakes are easy to make here: (1) if the FK direction is `ObservationRecord -> CampaignRun` (rather than `CampaignRun -> ObservationRecord` M2M, or a companion record like the calendar one), any `on_delete=CASCADE` on that FK means deleting a `CampaignRun` would delete real third-party `ObservationRecord` rows — telescope-time accounting records that belong to `tom_observations`, not to this app, and that other TOM Toolkit features (visibility plots, existing sync commands) depend on continuing to exist; (2) even with the safer direction/cascade, if `ObservationRecord` rows are matched to a `CampaignRun` via a loose heuristic during attribution (Pitfall 4) rather than an explicit confirmed link, the existing `sync_lco_observation_calendar`/`sync_gemini_observation_calendar` commands' own idempotency keys (the LCO portal URL, `GEM:{prog}/{observation_id}`) could get silently reused/reinterpreted by the new linkage logic in a way that breaks those commands' own no-churn re-run guarantees if the new FK/M2M field is added without auditing whether either sync command's `update_or_create`/lookup logic needs to preserve or ignore it.

**Why it happens:**
It's tempting to add the FK on the "cheaper" side (whichever model is easier to migrate) without separately reasoning about ownership direction and blast radius — `ObservationRecord` rows are real telescope-time records other parts of TOM Toolkit read, and this app must never be in a position to cascade-delete them as a side effect of tidying up a `CampaignRun`.

**How to avoid:**
- Never put `on_delete=CASCADE` from `ObservationRecord`'s side toward `CampaignRun`-owned data; if the FK must live on `ObservationRecord`-adjacent FOMO-side data (a companion/through model, mirroring `CalendarEventTelescopeLabel`), the FK back to `CampaignRun` should be `SET_NULL` (same reasoning as Pitfall 2), and there should be no FK the other direction that could cascade-delete `ObservationRecord` itself.
- Prefer an M2M (`CampaignRun.observation_records = models.ManyToManyField(ObservationRecord, ...)`) or a lightweight FOMO-side through-model over a direct FK on `ObservationRecord`, precisely because `ObservationRecord` is third-party and this app should not add required/cascading fields to a model it doesn't own.
- Audit `sync_lco_observation_calendar.py`/`sync_gemini_observation_calendar.py` for any place their `update_or_create`/lookup logic could be affected by a new FOMO-side field being added to (or M2M'd against) `ObservationRecord`, and add a regression test proving both commands' existing no-churn re-run guarantees (SYNC-04, GEM-NOCHURN-01) are unaffected by the new linkage.

**Warning signs:** Migration adds a field directly to `tom_observations.ObservationRecord` (a third-party model this app doesn't own — a strong signal something's wrong architecturally) or adds `on_delete=CASCADE` pointed at a third-party model.

**Phase to address:** The `ObservationRecord` linkage phase; the spike should settle the FK-vs-M2M-vs-companion-model shape (this is explicitly one of the spike's listed deliverables — "how each adapter's existing identity key maps onto a run").

---

### Pitfall 13: Timezone/UTC-midnight errors specific to the four-stage pipeline's date semantics

**What goes wrong:**
The pipeline mixes two genuinely different date semantics that are easy to conflate: stage 1 (site twilight, sunset→sunrise, which straddles UTC midnight for most real sites — Las Campanas/Siding Spring/La Silla local evening is often UTC-late-day-into-next-UTC-day) versus stage 2 (00:00→23:59 **that day**, which the milestone spec states as a bare UTC calendar day with no site to anchor a local evening against). A reconciler that computes "the night of `window_start`" using a `CampaignRun.window_start` `DateField` (a bare date, no timezone) and then narrows from stage 2's UTC-day window down to stage 1's site-twilight window (once a site resolves) risks an off-by-one-day error exactly like the one already measured in the real data (pk=1's 21 July vs. the LCO events' 20 July) — because a `DateField` "night of 20 July" at a Southern-Hemisphere site conventionally means "the night that starts on the evening of 20 July local and ends the morning of 21 July local," which in UTC is a *different* calendar date than 20 July for sites west of Greenwich in local-evening terms, and the existing `sun_event()`/`load_telescope_runs` machinery already encodes this convention correctly for stage 1 — but stage 2's literal "00:00→23:59 that day" is a genuinely different, UTC-calendar-day convention that a careless narrowing-transition implementation could silently apply the wrong convention to the wrong stage.

**Why it happens:**
"That day" reads as unambiguous in the milestone spec's stage-2 row, but the codebase's own established convention (from Stage 1/`telescope_runs.py`, `load_telescope_runs`) is that "the observing night of date D" means sunset(D)→sunrise(D+1), a *local-evening-anchored* concept — the two conventions (bare UTC calendar day vs. local-evening-anchored night) look identical for sites near Greenwich but diverge for every real site in this system (Chile, Australia — both far from UTC+0), and a stage-2→stage-1 narrowing transition is exactly the seam where a developer might reuse stage 2's bare-date value as if it were already stage 1's night-anchor date without re-deriving it through `sun_event()`.

**How to avoid:**
- Treat stage 2's "00:00→23:59 that day" window explicitly as UTC-calendar-day semantics (matching the milestone's literal wording and the existing satellite-branch precedent in `_project_calendar_event()`, which already uses `datetime.combine(window_start, dt_time(0,0), tzinfo=UTC)`/`datetime.combine(window_end, dt_time(23,59), tzinfo=UTC)` for the no-fixed-horizon case) — do not silently reinterpret it as a local-evening-anchored night once a site becomes known; the stage-2→stage-1 transition should explicitly *replace* the UTC-day window with a freshly-computed `sun_event()` result for the resolved site and the run's window date, never adjust/reuse the stage-2 window's endpoints.
- Any date arithmetic that compares a `CampaignRun.window_start`/`window_end` `DateField` against a `CalendarEvent`'s UTC `start_time`/`end_time` `DateTimeField` should go through an explicit, tested conversion step (reusing `sun_event()` for the site-anchored case), never a bare `.date()` truncation of a UTC datetime compared directly to the `DateField` — that comparison is exactly where a night starting late UTC-yesterday gets attributed to the wrong calendar date.
- Add the pk=1-vs-11-LCO-events date-off-by-one directly as a reconciler/attribution test fixture (it already exists in real data) and assert the reconciler's own stage transitions never introduce a *new* one-day discrepancy on top of the attribution one.
- For the reconciler's window-length loop (`n_nights = (window_end - window_start).days + 1`, the existing pattern), confirm it is being reused for the multi-night stage-1 case, not reimplemented — this loop already encodes the correct inclusive-range/per-night convention that a from-scratch reimplementation could easily get wrong (off-by-one on the inclusive end).

**Warning signs:** A reconciler test using a UTC-friendly fixture (e.g. a mocked site near Greenwich) passes, but the same test against Las Campanas/Siding Spring/La Silla data shows an event window shifted by one day from the expected value; or the reconciler's stage-2→stage-1 transition code path never calls `sun_event()` at all (a sure sign it's reusing the UTC-day window instead of recomputing).

**Phase to address:** The reconciler phase (stage-2→stage-1 transition logic specifically) and its test suite — should include at least one non-UTC-friendly real site (Las Campanas or Siding Spring, both already `Observatory` fixtures in this codebase) in every date-semantics test, never only a UTC-convenient mock site.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Backfilling `source` with a single blanket sentinel (`'legacy_unknown'`) instead of per-row inference | Migration ships fast, no per-row guessing logic to review | Coarser provenance for pre-milestone rows forever; any future `source`-keyed query/report for old data is unreliable | Acceptable — explicitly preferred over a guessing `RunPython` step (Pitfall 3); document the sentinel and its meaning in the migration's docstring |
| Reconciler triggered only per-run (on approve/on record change), no full-database sweep in v2.2 | Smaller surface area, reuses today's single-run call pattern, avoids Pitfall 10's batch-isolation design entirely | The 19 already-approved-but-unprojected 3I/ATLAS runs (the motivating defect) need a one-off catch-up pass regardless — deferring the sweep just re-invents `backfill_range_calendar_events` under a new name unless the reconciler is explicitly runnable in "sweep everything" mode too | Acceptable only if a one-off management-command sweep invocation ships in the same milestone to actually close the 19-run gap; not acceptable if it just becomes another bespoke backfill script |
| Leaving `telescope_class` un-enforced against the natural-key constraints (no test proving class-wide vs. site-resolved runs for the same telescope/window coexist) | One less test to write in the migration phase | Silent reintroduction of the class-wide/site-resolved ambiguity the field was added to fix, discovered only when a real collision happens in production data | Never — this is a stated purpose of the field; the spike should settle it explicitly |
| Skipping a per-run "reconciliation failed" surface in v2.2, relying on log output alone | Less UI work | Operators have no queue-style page (unlike `site_needs_review`'s "Sites Needing Review" precedent) to notice and retry failed runs; failures rot silently exactly like the pre-milestone calendar-projection gap did | Acceptable short-term only if the reconciler is still triggered per-run interactively (so failure is immediately visible in the response); not acceptable once/if it runs unattended on a schedule |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| `tom_calendar.CalendarEvent` (third-party model) | Adding ownership/attribution fields directly onto `CalendarEvent` via a fork or monkeypatch | Keep all FOMO-side state in the companion record (extending the existing `CalendarEventTelescopeLabel` pattern), exactly as this milestone already plans |
| `tom_observations.ObservationRecord` (third-party model) | Adding a FK/field directly onto `ObservationRecord`, or an `on_delete=CASCADE` that could delete real observation records | FOMO-side M2M or companion/through-model only, `SET_NULL`/no-cascade toward the third-party model (Pitfall 12) |
| `Observatory.timezone` (existing, sometimes-blank field) | Reconciler crashes or aborts the whole batch on the first `sun_event()` `ValueError` from a blank-timezone site | Reuse `backfill_range_calendar_events`'s already-established graceful-skip-and-log pattern per run, not per batch (Pitfall 10) |
| `django.db` unique constraints on `CampaignRun` | Adding `source`/`telescope_class` into the existing partial `UniqueConstraint`s without re-deriving the pk=1 scenario by hand | Keep `source` out of the natural-key constraints; treat cross-source duplicate detection as attribution, not constraint-level dedup (Pitfall 3) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Reconciler recomputes `sun_event()` (astropy `get_sun`/`AltAz`, non-trivial cost) for every night of every run on every invocation, including runs whose state hasn't changed since the last run | Reconciler wall-clock time grows linearly with total run-nights in the system, not with actual state changes | Consider a per-run "needs reconciliation" marker (dirty flag set on run/record/event state changes) so a full sweep can skip runs with nothing new, at least once the number of `CampaignRun`s grows past the current dozens-of-rows scale | Not urgent at today's ~30-40 `CampaignRun` scale (per the 19+ site-resolved 3I rows already measured); becomes real once campaign count or run-per-campaign count grows an order of magnitude |
| Attribution candidate generation does an O(n×m) scan of all unattributed events against all runs with no target/date pre-filter | Attribution page slow to load or times out as more historical data accumulates | Pre-filter candidates by campaign/target and a coarse date-overlap window before any string-similarity scoring (Pitfall 4's target-hard-filter recommendation also solves this) | Noticeable once unattributed-event backlog (currently ~39 events: 19 site-resolved runs + 20 Didymos events, per measured hazards) grows into the hundreds |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Attribution confirmation endpoint reachable without `StaffRequiredMixin` (mirroring the existing precedent everywhere else PII/state-changing actions live in this codebase) | Anonymous or non-staff user could confirm/unlink run-event-record attributions, corrupting provenance | Gate the attribution UI/endpoint with the same `StaffRequiredMixin` pattern already used for `ApprovalQueueView`/`CampaignRunDecisionView` |
| Reconciler exposed as a web-triggerable endpoint without the same staff/business-logic guards `CampaignRunDecisionView` already applies (conditional `.update()`, POST-only, staleness checks) | A crafted request could trigger unbounded reconciliation work (Pitfall 9) or race concurrent reconciler runs against the same run | If the reconciler is web-triggerable at all (vs. management-command-only), apply the same POST-only + staff-gated + conditional-update discipline already established by `CampaignRunDecisionView` |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| Reconciler runs silently in the background with no visible summary of what it changed | Staff can't tell whether a calendar change was human-made, sync-command-made, or reconciler-made, undermining trust in the calendar as a source of truth — exactly the confusion this milestone is meant to eliminate | Surface a per-invocation summary (created/updated/unchanged/failed counts, mirroring `import_campaign_csv`'s convention) and consider a lightweight "last reconciled" indicator on the run/companion-record admin views |
| Attribution suggestions presented with no context (bare pk/date/telescope string) | Staff can't judge whether a suggested match is right without leaving the page to cross-reference the campaign table | Render each suggestion with the same evidentiary detail a human reviewer needs: campaign name, target, both records' telescope/instrument strings side by side, date overlap visualized — not just a raw confidence score |

## "Looks Done But Isn't" Checklist

- [ ] **Companion-record rename:** Often missing an updated template reference — verify `src/templates/tom_calendar/partials/calendar.html`'s two `telescope_label_meta` lookups still resolve correctly, not just that migrations run clean.
- [ ] **`source` backfill:** Often missing an explicit, documented value for pre-existing rows — verify no `CampaignRun.source` is left NULL/blank after the migration, and that the value chosen doesn't silently collide with (or get excluded from) the natural-key constraints.
- [ ] **Reconciler idempotency:** Often "looks idempotent" because it uses `get_or_create`/`update_or_create` somewhere, but still churns `modified` on unchanged runs — verify with an explicit two-consecutive-runs-zero-diff test, not just "no exception raised twice."
- [ ] **Attribution "never silent merge":** Often looks safe because there's a confirm button, but the confirm action itself does a bulk `.update()` with no per-row evidence or undo — verify each confirmation is individually attributable and reversible.
- [ ] **Reconciler ownership scoping:** Often looks correct against `CampaignRun`-derived fixtures alone — verify explicitly against a fixture containing an unrelated hand-created `CalendarEvent` (a conference, no `run` link) in the same date window, and assert it survives every reconciler pass untouched.
- [ ] **Timezone/date-boundary correctness:** Often validated only against a UTC-convenient mock site — verify against at least one real Southern-Hemisphere site (Las Campanas or Siding Spring) whose local evening spans a UTC calendar-date boundary.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|-----------------|------------------|
| Companion-record rename breaks a template/prefetch reference silently | LOW | Grep for the old class/related-name string across `.py`/`.html`, fix the missed reference, add the regression test that should have caught it, ship a follow-up commit |
| `source` added into a unique constraint, causing false collisions on future adapter writes | MEDIUM | Migration to drop `source` from the constraint's field list; audit for any rows that were silently merged/rejected under the wrong constraint in the interim and manually re-attribute |
| Reconciler duplicates events across stage transitions | MEDIUM | One-off cleanup command (mirroring `backfill_range_calendar_events`'s precedent) that finds and merges/deletes duplicate-keyed events for the same run, dry-run first; fix the underlying key-scheme bug before rerunning the reconciler |
| Reconciler deletes/mutates an unowned `CalendarEvent` (a conference, or a not-yet-attributed sync-command event) | HIGH | No generic recovery — depends entirely on whether the event's original data (title/window/description) is recoverable from another source (git history of a fixture, the originating sync command's re-run, or manual staff memory); this is exactly why Pitfall 8's prevention must be airtight before the reconciler ever runs against production data |
| Attribution mis-links a run to the wrong event/record | LOW (if undo exists per Pitfall 5) / HIGH (if it doesn't) | With undo: unlink via the same UI, done. Without undo: manual DB-level correction plus an audit of any reconciler runs that acted on the bad link in the interim |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| 1. Companion-record rename breaks template/admin/prefetch/command references | Spike + companion-record migration phase | 4-site checklist re-verified by running `./manage.py test solsys_code` and manually confirming rendered calendar HTML still shows the dashed-border fallback indicator |
| 2. `run` FK added as required/CASCADE | Companion-record migration phase | Migration reviewed for `null=True`/`SET_NULL`; no `RunPython` backfill of `run` in the schema migration |
| 3. `source` collides with existing unique constraints | Spike (decision) + migration phase (implementation) | pk=1/11-LCO-events scenario reproduced as an explicit test: both representations coexist without `IntegrityError` |
| 4. Attribution auto-links wrong parent run | Attribution phase | pk=1/11-LCO-events pair surfaced as a candidate with visible evidence; target/campaign hard-filter proven by a cross-target-no-match test |
| 5. Attribution has no undo | Attribution phase (same deliverable as #4) | Unlink action present in UI/tests; confirmation logged with who/when |
| 6. Reconciler not idempotent across stage transitions | Reconciler phase | Two-consecutive-runs-zero-diff test; full 4-stage walk-through test with exact expected event-set assertions |
| 7. Reconciler churns `modified` on unchanged runs | Reconciler phase | Reuses `insert_or_create_calendar_event()`/`start_time_tolerance`; explicit `modified`-unchanged assertion in tests |
| 8. Reconciler mutates/deletes unowned `CalendarEvent`s | Reconciler phase | Fixture with an unrelated hand-created event proven untouched across every reconciler test |
| 9. Unbounded event creation from a wide/bad window | Spike (stage-2 fan-out decision) + reconciler phase (bound enforcement) | Window-length sanity bound test; stage-2 fan-out behavior explicitly documented and tested |
| 10. Partial reconciler failure blocks the whole batch | Spike (trigger-shape decision) + reconciler phase | Per-run try/except proven by a mid-batch-failure test asserting other runs still process; summary output includes failed-run list |
| 11. `approval_status` semantics drift across old/new sources | `source`/`telescope_class` field-addition phase | CSV-import (and any other non-web adapter) fixture proven to write `APPROVED` at creation, not left in `PENDING_REVIEW` limbo; SUBMIT-02/D-09 visibility tests rerun against it |
| 12. `ObservationRecord` linkage cascade/direction mistakes | Spike (linkage-shape decision) + linkage phase | No field added directly to `ObservationRecord`; no cascade path from `CampaignRun` deletion to real observation-record deletion; existing sync commands' no-churn tests rerun unaffected |
| 13. UTC-midnight/date-boundary errors across pipeline stages | Reconciler phase (stage-2→stage-1 transition) | Non-UTC-friendly real site (Las Campanas/Siding Spring) included in every date-semantics test; stage-2→stage-1 transition proven to call `sun_event()` fresh, never reuse the UTC-day window |

## Sources

- `.planning/PROJECT.md` — Current Milestone v2.2 section, measured live-data hazards (2026-07-26), prior-migration review finding (0004/CR-01)
- `solsys_code/models.py` — `CalendarEventTelescopeLabel`, `CampaignRun`, existing partial `UniqueConstraint`s and `CheckConstraint`
- `solsys_code/campaign_views.py` — `_project_calendar_event()`, `_calendar_event_title()`, `_set_run_status()`, `CampaignRunDecisionView` (existing partial-failure/no-churn/staleness-guard precedents this milestone's reconciler must generalize)
- `solsys_code/calendar_utils.py` — `insert_or_create_calendar_event()` no-churn create-or-update contract, `start_time_tolerance` drift handling
- `solsys_code/admin.py` — confirmed 4th consumer of `CalendarEventTelescopeLabel` (admin registration)
- `tom_calendar/models.py` (installed package) — `CalendarEvent.modified = models.DateTimeField(auto_now=True)`, confirming the churn hazard in Pitfall 7
- [Django ticket #29000 — RenameModel does not rename M2M column when run after RenameField](https://code.djangoproject.com/ticket/29000)
- [Django ticket #27903 — RenameModel does not change ForeignKey with related_name='+'](https://code.djangoproject.com/ticket/27903)
- [Django Forum — Backwards compatible migrations](https://forum.djangoproject.com/t/backwards-compatible-migrations/1406)
- [The Reconciler Pattern (farishuskovic.dev) — level-triggered idempotent reconciliation, ownership-scoped mutation](https://www.farishuskovic.dev/blog/k8s-reconciler-pattern/)
- [Understanding and Implementing the Reconciliation Loop Pattern (oneuptime.com)](https://oneuptime.com/blog/post/2026-02-09-operator-reconciliation-loop/view)

---
*Pitfalls research for: FOMO v2.2 "One Canonical Run Record" — canonical-record consolidation + idempotent reconciler retrofit onto a live Django/TOM Toolkit system*
*Researched: 2026-07-26*
