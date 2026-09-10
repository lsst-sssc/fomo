---
phase: 33-series-identity-reconciler-inversion
verified: 2026-09-10T07:05:00Z
status: human_needed
score: 108/109 must-haves verified
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "No code added by this phase depends on the iteration order of the `observation_group` reverse manager: `CalendarEventMeta` gains no `Meta.ordering`, and no reader added here iterates `group.calendar_event_metas` expecting a stable order (PROJ-04 ordering edge)."
    reason: >-
      `verification: backstop` / `insufficient_spec` item from the 2026-09-08 verification,
      explicitly resolved by the human at UAT test 3 (33-UAT.md `## Decisions`, 2026-09-09):
      "Accept absence-by-grep evidence for observation_group reverse-manager ordering
      (option A); carry a 'set ordering or add shuffled-insertion test if a reader is added'
      requirement into Phase 34 context." Re-checked at HEAD: `CalendarEventMeta` still
      declares no `class Meta` (solsys_code/models.py) and the gap-closure wave added no
      production reader of `group.calendar_event_metas`.
    accepted_by: "tlister@lco.global"
    accepted_at: "2026-09-09T00:00:00Z"
re_verification:
  previous_status: gaps_found
  previous_score: 80/83
  gaps_closed:
    - "Gap 1 / CR-04 (the confirm/erase loop): `_stale_attributions()` splits this run's stale owned events into `clearable_event_ids` (companion row's `confirmed_by IS NULL`) and `declined` (`confirmed_by` set). A human-confirmed attribution is never cleared by an automated sweep. Independently reproduced end-to-end through the REAL staff view (throwaway probe module under the Django test runner, then deleted; no source file modified): R1 created=1 -> attribute facility event -> R2 detached=1, row orphaned, re-offered to run 1 at HIGH band 0.82 -> POST to `campaigns:attribution_decide` (action=confirm) stamps `run=1, confirmed_by=2, confirmed_at=...` -> R3 detached=0, detach_declined=1, stamp intact -> R4 detached=0, detach_declined=1, stamp intact. The loop the previous verification reproduced is closed at its actual production trigger."
    - "Gap 2 / WR-13 (attributed-AND-contested night): `_may_write(existing, run)` is now evaluated FIRST in `_reconcile_classical_nights()`, before the attributed-night skip, and the blocked branch adds the url to `active_urls` (campaign_reconciler.py:438-446). `test_attributed_and_contested_night_is_blocked_not_skipped_and_never_detached` asserts `blocked==1, skipped_nights==0, detached==0, detach_declined==0` and that the foreign run's `run`/`confirmed_by`/`confirmed_at` are untouched. All three clauses of 33-08 truth 9 now hold together."
    - "UAT G-33-2 (calendar pop-up dead on click): the three `hx-on::after-request` handlers in `src/templates/tom_calendar/partials/calendar.html` now call `bootstrap.Modal.getOrCreateInstance(document.getElementById('cal-modal')).show()`. Four Playwright tests in `test_bootstrap5_rendering.py` click a real rendered calendar in headless Chromium (attributed entry, unattributed entry, empty day cell, '+ New Event'), assert `#cal-modal.show` becomes visible, assert the 'Attributed campaign run' block and the 'View campaign' link are present, and assert `pageerror` collected == []. I ran them: 7 tests, OK."
    - "UAT G-33-4 first half (notebook residue): both demo notebooks copy `src/fomo_db.sqlite3` to a `tempfile.mkdtemp()` scratch file and export `FOMO_DATABASE_PATH` BEFORE `django.setup()`, assert the resolved `DATABASES['default']['NAME']` IS that copy, and `rmtree` it at the end. Committed output proves it (`Resolved database: '/tmp/fomo-notebook-db-vs5we3ua/fomo_db.sqlite3' (scratch copy...)`). Residue verified gone by direct sqlite query of the developer database: event pk 335 does not exist; no `Reconciler Demo Campaign` / `Campaign Lifecycle Demo` TargetList; 0 attributed events in September 2026; 0 `RUN:` events in September 2026."
    - "UAT G-33-4 second half (contact fields): `contact_person`/`contact_email` no longer appear in ANY committed output of `campaign_lifecycle_demo.ipynb` (they survive only as form INPUT in two submission code cells, which is the form's required payload). Confirmed by parsing every cell's outputs at HEAD and at cf4a916."
    - "The previously-abstained `insufficient_spec` backstop ordering item — resolved by explicit human decision at UAT test 3 (recorded as an override above, not silently passed)."
  gaps_remaining: []
  regressions: []
flagged_prohibitions:
  - statement: "No notebook cell may print or store a run's `contact_person`, `contact_email` or `source` into committed output (33-09 P1, inherited from 33-05 P1 / 33-08 P5)."
    verdict: "contact half CLOSED; `source` half technically violated, pre-existing and benign (NON-AUTHORITATIVE LLM-judge verdict)"
    observed: >-
      The contact-field breach the UAT ordered fixed is genuinely fixed (0 occurrences in any
      committed output). The prohibition's third named field is not: `campaign_lifecycle_demo.ipynb`
      cell 10 prints `source='web'` four times and cell 12 prints
      `source='classical_file'|'lco_queue'|'eso_queue'|'legacy'` — the provenance enum, which is
      the point of those cells. Byte-identical to cf4a916, so this is inherited, not introduced by
      the gap-closure wave, and no PII is involved. Either the prohibition's `source` clause should
      be narrowed or the prints removed.
    flag: "unverified-prohibition — human review recommended"
deferred:
  - truth: "The superseded night no longer shows two calendar entries (WR-09 — the detach removes the attribution, the `CalendarEvent` row survives by design and still renders, now with no campaign chip and no campaign name in its title)"
    addressed_in: "Phase 35"
    evidence: "Phase 35 success criterion 5: 'After the stated cutover step runs, an operator looking at the calendar sees one event per night: no duplicate and no orphan left behind from the old load_telescope_runs events or the reconciler's RUN:{pk}:{date} events'. Restated in the 33-10 UAT decision: 'Leftover RUN:{pk}:{date} duplicates on a night remain Phase 35 SC 5's responsibility.'"
  - truth: "PROJ-04's second clause — the shared title stem for a series"
    addressed_in: "Phase 34"
    evidence: "REQUIREMENTS.md line 139: 'PROJ-04 is the only requirement whose clauses land in two phases — Phase 33 [carrier fields], Phase 34 [shared title stem]'. Phase 33's half (the `observation_record`/`observation_group` foreign keys) is verified below."
human_verification:
  - test: "Open http://<dev-server>/calendar/?year=2026&month=7 (July 2026 — 15 campaign-attributed entries, all belonging to run pk=1; verified present in `src/fomo_db.sqlite3` by direct query, so NO fixture seeding is needed). Click one of the ⚑ entries, then click 'View campaign ↗' in the 'Attributed campaign run' block."
    expected: "The pop-up opens (this half is now machine-proven by the Playwright tests, so it should just work), and the campaign table page then loads SCROLLED to that run's own row with the row visibly highlighted by the `tr:target` rule."
    why_human: "Browser anchor-scroll plus `:target` highlight rendering is real-browser behaviour no server-side or headless-assertion test observes. This is 33-11 Task 2's deferred `<human-check>` and UAT G-33-2's third `missing:` item. NOTE: 33-09's fixture receipt says July 2025 – July 2026; the surviving attributed months are 2025-07 (26), 2025-08 (21), 2025-11 (2), 2026-01 (1), 2026-07 (15) — none in the current month, so navigate deliberately."
  - test: "Review the flagged judgment-tier prohibition above: `campaign_lifecycle_demo.ipynb` prints each run's `source` enum value into committed output, and the 33-09/33-05 prohibition text names `source` alongside the two contact fields."
    expected: "Either the `source` clause of that prohibition is narrowed (it is provenance metadata, not contact data, and the cells exist to demonstrate it), or the prints are dropped and the notebook re-executed."
    why_human: "unverified-prohibition — a judgment-tier must-NOT carries no wired enforcement test; a model verdict is never authoritative. Pre-existing (byte-identical to the pre-wave notebook), so this does not block the phase on its own."
---

# Phase 33: Series Identity & Reconciler Inversion — Verification Report (second re-verification)

**Phase Goal:** `CalendarEventMeta` carries real series identity and attribution links, and the campaign reconciler annotates instead of owning — so the observation projector can land next phase without the campaign layer stealing its events.
**Verified:** 2026-09-10
**Status:** human_needed
**Re-verification:** Yes — after gap-closure plans 33-09, 33-10 and 33-11 (second gap-closure wave)

## Headline

**Both blocking gaps are closed in the code, not just in the SUMMARYs, and I proved the important
one myself rather than taking the tests' word for it.** The CR-04 confirm/erase loop — the
goal-level failure of the previous verification — is closed at its real production trigger: I wrote
a throwaway probe that drives the actual staff attribution view (`POST campaigns:attribution_decide`,
`action=confirm`), not a direct model write, and watched a human confirmation survive two further
unattended sweeps with `detach_declined=1` each time. The probe was deleted; no source file was
modified. WR-13 is closed by reordering ownership ahead of the night's outcome, with a test that
asserts all three clauses together. G-33-2 (the calendar pop-up dead on every click) is closed with
four real headless-Chromium tests that I ran, and G-33-4's notebook residue is verifiably gone from
the developer database by direct sqlite query.

The one thing that keeps this from `passed` is not a defect: **plan 33-11 deliberately deferred one
browser observation** — that following 'View campaign ↗' scrolls to and highlights the run's row —
to the end-of-phase human checkpoint, because anchor-scroll and `:target` rendering are not
observable server-side. Everything upstream of it (the anchor `id`, the `tr:target` rule in the
served HTML, the link's `href`, and now the pop-up opening at all) is machine-verified.

Independent quality gates at HEAD `fb99f9a`: `pre-commit run ruff --all-files` Passed,
`pre-commit run ruff-format --all-files` Passed, and 270 targeted tests run by this verifier
(`test_campaign_reconciler` 67, `test_reconcile_campaign_runs` + `test_campaign_approval` +
`test_calendar_template` + `test_calendar_event_meta_links` 196, `test_bootstrap5_rendering` 7) — all OK.

## Goal Achievement

### ROADMAP Success Criteria (the contract) — regression re-check

| # | Success criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Event links to `ObservationRecord`/`ObservationGroup` by real FKs; every companion row survives the migration with `run`/`is_verified`/`confirmed_by` intact | ✓ VERIFIED | `solsys_code/models.py:54-68` still declares `observation_record` (OneToOne, SET_NULL) and `observation_group` (FK, SET_NULL); `migrations/0017_calendareventmeta_observation_links.py` present; `test_calendar_event_meta_links` re-run in this verification, passing. `UNLINK_CLEARED_FIELDS` still excludes both carriers and `is_verified` (campaign_utils.py:868-872) |
| 2 | `reconcile_campaign_runs` no longer adopts, re-keys or detaches any event **outside** the `RUN:` namespace | ✓ VERIFIED | The attributed-night skip still `continue`s before any write (campaign_reconciler.py:444-446); the new blocked-first branch (:438-442) also writes nothing. `_stale_attributions()` filters `owned_events(run)` — the `RUN:` namespace only — and `unlink_event_from_run()` keeps its `run_id=run.pk` term. My probe's facility event (`https://observe.lco.global/...`) kept its url, fields, link and stamps across four sweeps |
| 3 | Campaign decoration rendered from the link, surviving a from-scratch rewrite of title/description | ✓ VERIFIED | `campaign_decoration()` (`solsys_code/templatetags/calendar_display_extras.py:434+`) has zero `.save()`/`.update()`/`.create()`/`get_or_create()`/`.delete()` calls in the whole module (grep). The wave's only change here is a defensive `isinstance(event, CalendarEvent)` early `return None` — still read-only. `test_calendar_template` re-run, passing |
| 4 | Clearing `CalendarEventMeta.run` removes only the decoration; the event is untouched, nothing deleted | ✓ VERIFIED | `unlink_event_from_run()` unchanged by this wave: one `.filter(run_id=..., **event_filter).update(**UNLINK_CLEARED_FIELDS)`, no `CalendarEvent` write, no delete. 33-10 deliberately did NOT push the human-confirmation guard into this helper (its human callers — the undo view, the admin clear — must still clear a confirmed row); the guard lives on the reconciler side in `_stale_attributions()` |
| — | **Derived goal truth**: a human-made attribution survives an unattended sweep ("annotates instead of owning") | ✓ VERIFIED | **Previously the sole FAILED truth.** See the probe transcript under 33-10 truth 1 below |

### 33-09 — notebook scratch-database isolation and demo residue cleanup (8/8 verified)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Executing either notebook leaves `src/fomo_db.sqlite3` untouched — writes land in a scratch copy | ✓ | Setup cell copies `src/fomo_db.sqlite3` with `shutil.copy2` into `tempfile.mkdtemp(prefix='fomo-notebook-db-')` and sets `os.environ['FOMO_DATABASE_PATH']` **before** `django.setup()` (the named key link). Teardown cell `rmtree`s it. Executed output confirms the ordering worked in the committed run |
| 2 | Each notebook proves where it wrote, in its own committed output | ✓ | Setup cell `assert resolved_db_name == str(scratch_db_path)` then prints it. Committed output: `Resolved database: '/tmp/fomo-notebook-db-vs5we3ua/fomo_db.sqlite3'` (reconcile) and `'/tmp/fomo-notebook-db-sm72cbt8/...'` (lifecycle) |
| 3 | `settings.py` reads `FOMO_DATABASE_PATH` when set and non-empty; unset leaves today's default exactly | ✓ | `src/fomo/settings.py:134` — `os.getenv('FOMO_DATABASE_PATH') or os.path.join(BASE_DIR, 'fomo_db.sqlite3')`; the `or` covers the empty-string case. Exercised implicitly by all 270 tests I ran with the variable unset |
| 4 | The lifecycle notebook's public-table cell prints no contact fields | ✓ | Parsed every cell's outputs at HEAD: 0 occurrences of `contact_person`/`contact_email` in any output. Remaining source occurrences are the form POST payload in two submission cells (cells 10 and 22) and one markdown paragraph — input, not committed output |
| 5 | The developer database holds no residue from either demo notebook, and pk 335 is gone | ✓ | Direct sqlite query of `src/fomo_db.sqlite3`: `select id from ...calendarevent where id=335` → empty; no TargetList named `Reconciler Demo Campaign` or `Campaign Lifecycle Demo`; 0 attributed metas and 0 `RUN:` events in September 2026 |
| 6 | The cleanup leaves a fixture receipt, not a silence | ✓ | 33-09-SUMMARY.md records `ATTRIBUTED_SURVIVING=65`, earliest `2025-07-03 22:00:47`, latest `2026-07-21 07:27:08`. **My independent query returns exactly 65 and exactly that date range** — the receipt is truthful, not narrated |
| 7 | Notebook prose describes the scratch-copy mechanism | ✓ | Markdown cell 1 in both notebooks now states the copy-before-`django.setup()` mechanism and cites UAT G-33-4 |
| 8 | Both notebooks remain re-runnable evidence for ANNOT-01/ANNOT-02 | ✓ | 11/11 and 19/19 code cells carry executed output; a re-run now costs nothing but a temp directory |

### 33-11 — the calendar pop-up opens again (6/7 verified, 1 human)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Clicking a campaign-attributed entry opens `#cal-modal` in a real browser, carrying the 'Attributed campaign run' block and the 'View campaign ↗' link | ✓ | `test_calendar_modal_opens_for_campaign_attributed_event_with_no_page_errors` — waits for `#cal-modal.show` visible, asserts the block text and `>=1` 'View campaign' anchor in `#cal-modal-body`. **Run by this verifier: OK** |
| 2 | The click raises no JavaScript error (`pageerror` list empty) | ✓ | All four modal tests register `page.on('pageerror', ...)` and assert `page_errors == []` — a handler calling a global the page never loads fails the test instead of failing silently |
| 3 | Every click target opens the pop-up, not just an attributed entry | ✓ | Three further passing tests: unattributed entry, empty day cell (clicks `.day-num`, outside the inner `event.stopPropagation()` guard), '+ New Event' button |
| 4 | The served partial contains no jQuery selector call and calls `bootstrap.Modal.getOrCreateInstance` | ✓ | `test_calendar_template.py:844-867` asserts `'$('` absent and `'bootstrap.Modal.getOrCreateInstance'` present in the rendered partial. grep of `src/templates/tom_calendar/partials/calendar.html` shows three handlers, all the identical fixed string (also satisfying the "no template variable in an inline handler" prohibition) |
| 5 | Following 'View campaign ↗' lands on the campaign table scrolled to the run's row, visibly highlighted by `tr:target` | ? **NEEDS HUMAN** | Server side fully verified (33-06: the rule is inside `{% block additional_css %}` and is served; the anchor `id="run-{pk}"` and the link's `href` are asserted by `test_campaign_views.py:634-684`). Anchor scroll + `:target` paint are browser-rendering behaviour. Deliberately deferred by 33-11 Task 2's `<human-check>` — see Human Verification below |
| 6 | The browser check has a fixture it can actually reach | ✓ | 33-09's receipt says 65 attributed rows survive; I independently confirmed and localised them: 2025-07 (26), 2025-08 (21), 2025-11 (2), 2026-01 (1), 2026-07 (15, all run pk=1). No seeding needed — contrary to the plan's expected-zero case |
| 7 | The runbook's pop-up section matches the restored behaviour | ✓ | `docs/runbooks/telescope_runs_calendar.rst:815-828` — states the pop-up opens through the Bootstrap 5 modal API because the TOM Toolkit 3.x base loads no jQuery, and distinguishes "opens but shows no block" (missing attribution) from "does not open at all" (client-side fault), naming Phase 33 / UAT G-33-2 |

### 33-10 — a human confirmation outranks the machine (11/11 verified)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A human-confirmed attribution survives an unattended sweep | ✓ | `_stale_attributions()` (campaign_reconciler.py:482-527) splits stale owned events on `confirmed_by__isnull`. **Independently reproduced end-to-end through the real staff view** (probe, then deleted): `R1 created=1` → attribute facility event → `R2 detached=1` (row orphaned; `candidates_for_event` re-offers it to run 1 at `high` 0.82) → `POST campaigns:attribution_decide action=confirm` → `run_id=1 confirmed_by=2 confirmed_at=…` → `R3 detached=0 detach_declined=1`, stamp intact → `R4 detach_declined=1`, stamp intact. The production confirm path really does stamp `confirmed_by` (`campaign_views.py:1224` — `.filter(event_id=…, run__isnull=True).update(run_id=…, confirmed_by=request.user, confirmed_at=…)`), which is what makes the guard reach the real world and not just the test fixture |
| 2 | The loop is closed at its actual trigger (reconcile → attribute → reconcile → RE-CONFIRM → reconcile) | ✓ | `test_staff_reconfirmation_of_the_detached_run_keyed_event_survives_every_later_sweep` walks exactly that sequence plus a fourth sweep, asserting `detached==0, detach_declined==1`, the event pk, `run_id`, `confirmed_by_id` and `confirmed_at` all survive, and `CalendarEventDismissal.objects.count()==0`. Plus my probe above |
| 3 | An unconfirmed stale row is still detached exactly as 33-08 shipped it — the fix is narrowed, not weakened | ✓ | `test_second_reconcile_detaches_the_superseded_run_keyed_event_and_restore_on_third` appears in **no** hunk of the wave's diff (`git diff cf4a916..HEAD` — 10 removed lines total in the file, all an import line and one renamed test's name/docstring). `test_unconfirmed_reattribution_of_the_detached_run_keyed_event_is_still_reclaimable` pins the negative control: an automated re-link with `confirmed_by` null is still reclaimed (`detached=1`) |
| 4 | The sweep reports what it declined to release | ✓ | `ReconcileResult.detach_declined` (:104); `logger.warning('Reconcile declined to detach %s … a human confirmation outranks the automated sweep.')` — **observed in my own test-run stderr**; command summary line (`detach_declined: N`) and per-run stderr line (`reconcile_campaign_runs.py:95-100`); `test_real_sweep_reports_declined_for_a_human_confirmed_superseded_row` |
| 5 | No `CalendarEventDismissal` is ever written by an automated detach | ✓ | grep for `Dismissal` in `campaign_reconciler.py` → zero hits; the CR-04 test asserts `CalendarEventDismissal.objects.count() == 0`. The UAT explicitly rejected the dismissal-row remedy and the code respects that |
| 6 | An attributed-AND-contested night reports `blocked==1`, keeps its url active, is never detached, foreign stamps untouched (WR-13) | ✓ | `_may_write()` is now first (:438), the blocked branch does `active_urls.add(url)` before `continue`, and only then is the attributed-night skip evaluated. `test_attributed_and_contested_night_is_blocked_not_skipped_and_never_detached` asserts `blocked==1, skipped_nights==0, detached==0, detach_declined==0` and the foreign run's `run_id`/`confirmed_by_id`/`confirmed_at` |
| 7 | `--dry-run` previews the detach from the same predicate, and still writes nothing (WR-11) | ✓ | `reconcile_run()` calls `_stale_attributions()` directly on the dry-run branch (:640-644) — one predicate, two consumers. `_stale_attributions()` is reads only (one `exclude`, one `filter`, one `values_list`, one `count`). `test_dry_run_previews_the_detach_count_and_writes_nothing` + `test_dry_run_previews_the_would_detach_count_and_writes_nothing`; the notebook's executed output shows `would_detach`/`detach_declined` matching the live counters |
| 8 | `detached` is described by BOTH its causes, in the per-run line and the runbook (WR-10) | ✓ | Command per-run line: "released back into the attribution queue -- superseded by another attributed entry, or left over from a key family this run no longer belongs to". Runbook :676-685 says the same and adds that `would_detach` reports the same number a real sweep would detach |
| 9 | All three `reconcile_run()` call sites in `campaign_views.py` surface `result.detached`, and `_resolve_site()`'s success message is keyed on `created`/`updated`/`skipped_nights` (WR-12) | ✓ | One shared `_message_reconcile_side_effects()` (:435-461, `messages.warning` for `detached`, `messages.info` for `detach_declined`) called at :566 (approve), :727 (`_resolve_site`), :811 (`_set_run_status`) — the four staff actions. `_resolve_site()`'s message is now a three-way branch on `skipped_reason` / `created or updated` / else (:728-741), so "run added to the calendar" can no longer be shown when nothing was added. `test_resolve_that_detaches_something_shows_the_warning` covers the warning path (see Anti-Patterns for the two uncovered message branches) |
| 10 | The runbook no longer steers the operator into the destructive path | ✓ | Both places now say re-confirming is permanent: :466-469 ("Re-confirming a released entry is a permanent decision -- once a person has confirmed it, no later automated sweep releases it again") and :778-780 in the skip-rule section. `detach_declined` is documented at :687-700 including "there is nothing for an operator to do about a non-zero `detach_declined`". Section split with 33-11 held: 33-11's runbook commit is a pure 13-line insertion in the pop-up section; 33-10's is 41/-31 in its own three sections |
| 11 | Both notebooks re-executed post-fix, and the reconcile notebook shows a confirmed attribution surviving a sweep alongside the dry-run preview | ✓ | `reconcile_campaign_runs_demo.ipynb` cell 21's committed output: "Reconcile declined to detach 1 … a human confirmation outranks the automated sweep" → `ReconcileResult(… detached=0, detach_declined=1 …)` after re-confirmation, again on a further sweep, and again from `dry_run=True`. Its demo-scoped cleanup deletes only the stand-in event pk captured earlier in the same cell |

### Carried-forward truths (33-01 … 33-08)

The 2026-09-08 verification verified 80 of 83 must-haves across plans 33-01 – 33-08 and itemised
the evidence there. Regression check at HEAD: the gap-closure diff touches
`campaign_reconciler.py`, `campaign_views.py`, `reconcile_campaign_runs.py`,
`calendar_display_extras.py` (+8 lines), the calendar partial, `settings.py`, both notebooks, the
runbook and four test modules — nothing in `models.py`, `campaign_utils.py`, `admin.py`,
`campaign_attribution.py`, the migrations or the campaign templates. The four ROADMAP criteria
were re-verified above; 270 targeted tests re-run clean; both ruff gates clean. **No regressions found.**

**Score:** 108/109 truths verified (83 carried forward — 80 previously verified + 2 closed gaps + 1
human-accepted override — plus 25 of the 26 new gap-closure truths; the 1 outstanding is 33-11
truth 5, deferred to the human browser check by design). 0 present-but-behavior-unverified.

### Requirements Coverage

| Requirement | Source plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| PROJ-04 | 33-03 (+33-09 indirectly) | Series identity carried by real FKs on `CalendarEventMeta` (`observation_record`, `observation_group`), not a title-suffix stopgap | ✓ SATISFIED (Phase 33 half) | `models.py:54-68` + `migrations/0017_calendareventmeta_observation_links.py`; `test_calendar_event_meta_links` passing; both carriers excluded from `UNLINK_CLEARED_FIELDS`. The shared-title-stem clause is Phase 34's by the recorded scope split (REQUIREMENTS.md:139-141) — listed under `deferred` |
| ANNOT-01 | 33-01, 33-08, 33-09, 33-10 | `CalendarEventMeta.run` means "attributed to", not "owned by"; `reconcile_run()` only annotates | ✓ SATISFIED | ROADMAP SC2 + the derived goal truth, both verified above; the confirm/erase loop that contradicted this requirement is closed and independently reproduced as closed |
| ANNOT-02 | 33-01, 33-02, 33-06, 33-09, 33-11 | Campaign decoration rendered from the link at display time, never written into the event's fields | ✓ SATISFIED | ROADMAP SC3 verified; and the decoration is now demonstrably *reachable* — the pop-up that renders it opens again, proven in a real browser |

No orphaned requirements: REQUIREMENTS.md maps exactly PROJ-04, ANNOT-01, ANNOT-02 to Phase 33, and
every one is claimed by at least one plan.

**Bookkeeping note (not a gap):** REQUIREMENTS.md still carries `- [ ] PROJ-04` and a
`PROJ-04 | … | Gaps Found` row from commit `562fc1c` ("revert premature Complete requirements after
gaps found"), and ROADMAP.md still says "Plans: 9/11 plans executed" while all 11 are checked.
Both are stale status text that the phase-completion step should refresh now that the gaps are closed.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| notebook setup cell | `src/fomo/settings.py` `DATABASES['default']['NAME']` | `os.environ['FOMO_DATABASE_PATH']` set **before** `django.setup()` | ✓ WIRED | Ordering verified in source and proved by the committed output printing the scratch path |
| `shutil.copy2(dev db, scratch)` | every notebook write → `rmtree(scratch_dir)` | scratch copy | ✓ WIRED | Teardown cell present in both notebooks with executed output |
| `_stale_attributions(run, active_urls)` | `_detach_stale_family_events()` (write) **and** `reconcile_run()` dry-run branch (read) | one predicate, two consumers | ✓ WIRED | :580 and :641 — the preview cannot drift from the real sweep |
| `_stale_attributions()` | `campaign_utils.unlink_event_from_run(clearable_event_ids, run)` | `UNLINK_CLEARED_FIELDS` | ✓ WIRED | Guard is on the reconciler side only; the helper still clears a confirmed row for its human callers |
| `ReconcileResult.detach_declined` | command summary + per-run line + `logger.warning` | operator surfaces | ✓ WIRED | Observed in this verifier's own test-run stderr |
| `reconcile_run()` result | `_message_reconcile_side_effects(request, result)` | 3 call sites / 4 staff actions | ✓ WIRED | :566, :727, :811 |
| `_may_write(existing, run)` evaluated BEFORE the attributed-night skip | `blocked` + `active_urls` → `exclude(url__in=active_urls)` | ordering is the mechanism | ✓ WIRED | :438-446 |
| `.cal-event` / `.cal-day` / '+ New Event' `hx-on::after-request` | the `bootstrap` global from tomtoolkit 3.0.1's base | `#cal-modal` from `tom_calendar` `calendar_page.html` | ✓ WIRED | Proven in a real browser, not by grep |
| Staff confirm view | `CalendarEventMeta.confirmed_by` | `AttributionDecisionView._do_confirm_event()` | ✓ WIRED | `campaign_views.py:1224` — this is what makes 33-10's guard reach production; verified by probe, not by reading alone |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|-----------|---------|--------|--------|
| CR-04 loop closed through the REAL staff view | throwaway probe module under `python manage.py test` (created, run, deleted) | `R2 detached=1` → confirm via `campaigns:attribution_decide` → `R3/R4 detached=0, detach_declined=1`, `run_id`/`confirmed_by`/`confirmed_at` intact | ✓ PASS |
| Reconciler behaviour suite | `python manage.py test solsys_code.tests.test_campaign_reconciler` | Ran 67 tests, OK | ✓ PASS |
| Command / approval / calendar / link-field suites | `python manage.py test …test_reconcile_campaign_runs …test_campaign_approval …test_calendar_template …test_calendar_event_meta_links` | Ran 196 tests, OK | ✓ PASS |
| Calendar pop-up opens in a real browser | `python manage.py test solsys_code.tests.test_bootstrap5_rendering` | Ran 7 tests, OK (4 are the modal-open tests) | ✓ PASS |
| Demo residue absent from the developer DB | direct `sqlite3` query of `src/fomo_db.sqlite3` | pk 335 gone; 0 demo campaigns; 0 attributed events and 0 `RUN:` events in 2026-09 | ✓ PASS |
| Fixture receipt truthful | direct `sqlite3` count + min/max | 65 rows, `2025-07-03 22:00:47` … `2026-07-21 07:27:08` — exactly what 33-09-SUMMARY claims | ✓ PASS |
| Lint gate | `pre-commit run ruff --all-files` | Passed | ✓ PASS |
| Format gate | `pre-commit run ruff-format --all-files` | Passed | ✓ PASS |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `solsys_code/campaign_views.py` | 736-741 | `_resolve_site()`'s else branch always phrases the outcome as "`{skipped_nights}` night(s) are already covered…", so a retry that produces `unchanged>0` or `blocked>0` with `skipped_nights==0` reads "0 night(s) are already covered by entries attributed to this run" | ℹ️ Info | Cosmetic only. The truth it was written for holds absolutely — "run added to the calendar" is never shown when nothing was added. Worth a wording pass keyed on `unchanged`/`blocked` |
| `solsys_code/campaign_views.py` | 456-461, 728-741 | The `messages.info` `detach_declined` staff message and the reworded `_resolve_site()` success branches have no test asserting their strings; only the `detached` warning path is pinned (`test_campaign_approval.py:1154`) | ⚠️ Warning | Wiring is verified by reading all three call sites, and the message text is directly readable — but a future edit to these strings would not be caught. Not a gap; a test-coverage note for Phase 34+ |
| `docs/notebooks/pre_executed/campaign_lifecycle_demo.ipynb` | cells 10, 12 | Committed output prints each run's `source` enum, which the 33-09/33-05 prohibition text names alongside the contact fields | ⚠️ Warning | Pre-existing and byte-identical to `cf4a916`; no PII. Routed to human review as a flagged judgment-tier prohibition |
| — | — | Debt markers (`TBD`/`FIXME`/`XXX`) in the wave's modified files | ℹ️ None found | The only `TBD` hits are the domain skip reason `'TBD window'` and prose about TBD-window runs; the only `PLACEHOLDER` hit is the tier-3 placeholder-Observatory concept. No unreferenced debt markers |

### Scope Deviation (disclosed, accepted)

`solsys_code/templatetags/calendar_display_extras.py` was modified by 33-11 (`ee9957a`, +8 lines)
although it is not in 33-11's `files_modified`. It adds `if not isinstance(event, CalendarEvent):
return None` at the top of `campaign_decoration()` — the create-event form context has no `event`
key, so Django resolves the tag argument to the empty-string invalid-variable placeholder and the
tag raised `AttributeError` on the '+ New Event' path. Disclosed as a Rule 1 auto-fix in
33-11-SUMMARY.md (lines 223, 233-258). It is read-only, it makes the docstring's "never raises"
promise true, and it is exercised by the passing '+ New Event' browser test. Accepted, not a gap.

### Human Verification Required

#### 1. 'View campaign ↗' lands on the highlighted run row (UAT test 2, second half)

**Test:** Open `/calendar/?year=2026&month=7` on the dev server — July 2026 holds 15
campaign-attributed entries (all run pk=1), confirmed present in the developer database by direct
query, so **no fixture seeding is needed** (33-09's cleanup did not empty this month; only the
September 2026 demo entries went). Click a ⚑ entry, then click 'View campaign ↗' in the
'Attributed campaign run' block.
**Expected:** The pop-up opens (now machine-proven — this is the part that was broken), and the
campaign table page loads scrolled to that run's own row with the row visibly highlighted.
**Why human:** Browser anchor-scroll and `:target` paint are not observable by any server-side or
headless-assertion test. Deferred deliberately by 33-11 Task 2's `<human-check>`; it is UAT
G-33-2's third `missing:` item.
**Other attributed months if July 2026 is inconvenient:** 2025-07 (26 entries), 2025-08 (21),
2025-11 (2), 2026-01 (1).

#### 2. The `source`-in-output prohibition (judgment tier)

**Test:** Review whether `campaign_lifecycle_demo.ipynb` printing `source='web'` /
`source='classical_file'` etc. into committed output should still count as a breach of the
"no `contact_person`, `contact_email` or `source` in committed output" prohibition.
**Expected:** Either the prohibition's `source` clause is narrowed (those cells exist precisely to
demonstrate provenance), or the prints are dropped and the notebook re-executed.
**Why human:** `unverified-prohibition` — judgment-tier must-NOTs carry no wired enforcement test
and a model verdict is never authoritative. Pre-existing, no PII, so it does not block the phase.

### Gaps Summary

**None.** Both blocking gaps from the 2026-09-08 verification (CR-04's confirm/erase loop, WR-13's
masked `blocked` signal) and both UAT gaps (G-33-2 the dead calendar pop-up, G-33-4 the notebook
residue and contact fields) are closed in the codebase and independently verified here — the CR-04
closure by a fresh end-to-end probe through the real staff view rather than by trusting the wave's
own tests, and G-33-4 by querying the developer database directly rather than trusting the SUMMARY's
counts. The previously-abstained backstop ordering item was resolved by an explicit human decision at
UAT and is recorded as an override, not silently passed.

The phase is `human_needed` rather than `passed` for exactly one reason: plan 33-11 deliberately
routed one browser observation (anchor scroll + `tr:target` highlight) to the end-of-phase human
checkpoint, and a second, non-blocking judgment-tier prohibition wants a ruling. Neither is a code
defect.

---

_Verified: 2026-09-10T07:05:00Z_
_Verifier: Claude (gsd-verifier)_
