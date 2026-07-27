---
phase: 26-canonical-record-spike
plan: 04
subsystem: investigation-spike
tags: [django, campaign-run, calendar-utils, campaign-gap, spike, gap-closure]

# Dependency graph
requires:
  - phase: 26-canonical-record-spike (plans 26-01..26-03)
    provides: "the locked D-09/D-10 event-key mechanism, the D-11 adopt-vs-gap-fill prototype, and the domain-correction reopening of SPIKE-03 for queue-scheduled runs that 26-VERIFICATION.md flagged as a gap"
provides:
  - "26-DECISION.md's new '### SPIKE-03 gap closure -- queue-run projection, measured' Findings subsection: the run-type inventory (QUEUE=12/CLASSICAL=12/SPACE=7, RECON-07 split 8 QUEUE/11 CLASSICAL/0 SPACE), the campaign_gap.claimed_dates() over-claim finding with a real file:line citation, D-05's three arithmetic inputs, the three-way span/none/per-night scenario comparison with idempotency and window-narrowing key-stability tokens, the calendar-render measurement, and the key-scheme consequence -- all without a verdict"
affects: [26-05-canonical-record-spike]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bare vs. date-bearing calendar-event key stability measured directly by editing a CampaignRun's window on a disposable scratch copy and re-running the candidate projection a third time, rather than reasoning about stability abstractly"
    - "Per-event hx-get=\"{% url 'calendar:update-event' event.id %}\" URL token used as an exact per-instance calendar-render search key, since rendered title text is truncated (truncatechars:18) and several distinct real classical rows share identical truncated text"

key-files:
  created: []
  modified:
    - .planning/phases/26-canonical-record-spike/26-DECISION.md

key-decisions:
  - "Run-type categorization (QUEUE/CLASSICAL/SPACE) is a judgment applied to real field values, tagged explicitly as such -- CLASSICAL is not limited to the NTT/EFOSC2/Magellan family that already has real blank-url calendar events; any single-facility ground run with no shared-queue network (HCT, Palomar, Apache Point, IRTF, Deep Sky Chile, Joan Oró) defaults into the same bucket, since it likewise owns a specific awarded night at one telescope"
  - "The RECON-07 'no existing classical calendar presence' exclusion must be computed from the real blank-url CalendarEvent telescopes (a DB query), not from the run-type CLASSICAL bucket built in block (A) -- conflating the two undercounted RECON-07 from 19 to 8 on the first pass; caught and fixed before recording any figure in the decision doc"
  - "In-window CalendarEvent count for the three-way scenario comparison must be scoped to pk=1's own telescope/instrument/RUN:1-prefix footprint, not every CalendarEvent whose date happens to overlap the window -- the 9 real classical NTT/Magellan events coincidentally fall inside pk=1's July window and would otherwise inflate the count from 12/11/15 to 21/20/24"

requirements-completed: [SPIKE-03]

# Metrics
duration: ~90min
completed: 2026-07-27
---

# Phase 26 Plan 04: SPIKE-03 Gap Closure -- Queue-Run Projection, Measured Summary

**Measured, D-11-grade evidence for the queue-run projection question 26-VERIFICATION.md left open: all 31 CampaignRun rows categorized (12 QUEUE/12 CLASSICAL/7 SPACE, RECON-07 split 8/11/0), the shipped `campaign_gap.claimed_dates()` over-claim reproduced with a file:line citation, and three candidate calendar-projection strategies (span/none/per-night) built and counted against pk=1's real 15-night window and its real 11 LCO events, with the bare `RUN:1` key proven stable under a window-narrowing stage transition and the rejected per-night candidate proven unstable (one orphaned key) -- no verdict stated, leaving the choice to plan 26-05.**

## Performance

- **Duration:** ~90 min
- **Started:** 2026-07-27 (approx.)
- **Completed:** 2026-07-27
- **Tasks:** 3 completed
- **Files modified:** 1 (`26-DECISION.md`, across three task commits, pure insertions -- 198 lines added, 0 removed, 0 files other than this one touched in the whole plan's committed diff)

## Accomplishments

- **Task 1:** a read-only probe against the real, unmodified `src/fomo_db.sqlite3` categorized all 31 `CampaignRun` rows into QUEUE/CLASSICAL/SPACE (12/12/7), named pk=1/29/30 all QUEUE, and split the 19-row RECON-07 baseline 8 QUEUE / 11 CLASSICAL / 0 SPACE -- answering the "corner case or dominant mechanism" question the plan posed: neither, a genuine mixed population. The same probe called the shipped, unmodified `campaign_gap.claimed_dates()` against pk=1's real window and found it claims all 15 nights while only 11 were ever scheduled -- the identical category error the domain correction found in the reconciler's own key scheme, quoted with real `campaign_gap.py:207-209` line numbers, and explicitly flagged (not fixed) as v2.3's GAPB-01.
- **Task 2:** three candidate queue-run calendar projections (`span`: one bare `RUN:1` whole-window event; `none`: mint nothing; `per-night`: the rejected `RUN:1:{date}` candidate for each of the 4 site-local-uncovered nights) were built and counted against pk=1's real window and real 11 LCO events on three disposable scratch DB copies. All three left the 11 real LCO events completely untouched (`LCO_ROWS_UNTOUCHED=True` for all three). The key-stability probe -- narrowing pk=1's window by one night at each end and re-running -- found the bare `span` key stable (`KEY_SET_STABLE=True`, only the event's times changed) and the `per-night` key **not** stable (`KEY_SET_STABLE=False`, `RUN:1:2026-07-21` orphaned once the window no longer covered that date). A `django.test.Client()` calendar-render measurement found a reader cannot visually distinguish "one span row" from "N per-night rows" -- both render identically per covered day cell -- so the real difference lives in stored rows, ownership surface, and machine consumers, not the rendered grid.
- **Task 3:** the real dev DB's fingerprint (`946176 1785094461`) was confirmed unchanged across the whole plan and recorded verbatim inside the committed decision doc before `tmp/` was deleted. `local_settings.py` and `tmp/` (three scratch DB copies, two probe scripts, four captured output files) were removed and confirmed literally absent from disk and from every git ref. `python -m pytest` and `./manage.py test solsys_code.tests.test_calendar_template` both pass from the pristine post-teardown tree.
- The real `src/fomo_db.sqlite3` fingerprint (`946176 1785094461`) never changed across any of the three tasks -- verified at every task boundary, matching the value already recorded in `26-DECISION.md`'s D-04 Snapshot Finding exactly (no drift since plan 26-01/26-02/26-03).

## Task Commits

Real phase-26 branch (`issue37-telescope-runs-calendar`):

1. **Task 1: Measure the queue-versus-classical run inventory and the existing over-claim, read-only against the real dev DB** - `0eb8141` (docs)
2. **Task 2: Build and count the three candidate queue-run projections against pk=1's real window** - `8793e1b` (docs)
3. **Task 3: Record the surviving fingerprint, discard every scratch artifact, and prove the tree is pristine** - `66a2433` (docs)

No scratch-branch commits this plan -- unlike plans 26-01/26-02/26-03, this plan creates no scratch git branch at all and edits no `solsys_code/` file at any point (per the plan's own stricter scratch-discipline rules).

## Files Created/Modified

**Committed on the real branch:**
- `.planning/phases/26-canonical-record-spike/26-DECISION.md` -- extended with `### SPIKE-03 gap closure -- queue-run projection, measured`, containing `#### Run-type inventory`, `#### The same category error already exists in campaign_gap.claimed_dates()`, `#### D-05's 400-event figure`, `#### Three-way comparison against pk=1's real window`, `#### What a calendar reader actually sees`, and `#### The key-scheme consequence`, plus a closing surviving-fingerprint sentence. 198 lines added across three commits; zero lines of any pre-existing content modified or deleted.

**Throwaway, git-excluded (`tmp/`, `local_settings.py` -- discarded in task 3, never committed anywhere):**
- `tmp/26_queue_inventory_probe.py`, `tmp/26-queue-inventory.txt`, `tmp/26-realdb-fingerprint-before.txt`
- `tmp/26_queue_projection_probe.py`, `tmp/26_calendar_render_probe.py`, `tmp/26-queue-projection-counts.txt`, `tmp/26-calendar-render.txt`
- `tmp/26-queue-span-copy.sqlite3`, `tmp/26-queue-none-copy.sqlite3`, `tmp/26-queue-pernight-copy.sqlite3`
- `local_settings.py` (repointed five times across the plan: task 1 never created it at all; task 2 pointed it at span, then none, then per-night for the projection probe, then re-pointed across the same three copies again for the render probe)

## Decisions Made

- **Run-type categorization is explicitly a "judgment applied to real field values," not an executed check** -- the `source` field that would decide this mechanically does not exist until Phase 27's CANON-01. CLASSICAL was deliberately scoped broader than the NTT/EFOSC2/Magellan family D-19 already ties to real blank-url calendar events: any single-facility ground run with no shared-queue network (HCT, Palomar P200, Apache Point, IRTF, Deep Sky Chile, Joan Oró) defaults into the same bucket, since the distinguishing question is "does this facility schedule through a shared queue, or does the run own a specific awarded night" -- not "is this one of the 3 telescopes already visible on the calendar."
- **The bare `RUN:1` key is measurably stable under a window-narrowing stage transition; the rejected per-night `RUN:1:{date}` key is measurably not.** This is the direct, code-level answer to whether a single canonical key scheme, stable across all four pipeline stages, can cover queue-scheduled runs at all -- it can, but only with a bare (non-date-bearing) key form, mirrored by `campaign_views.py:797`'s existing `Q(url=...) | Q(url__startswith=...)` ownership-query precedent.
- **No verdict is stated on which projection option (span/none/per-night) a queue run should actually use** -- per the plan's explicit scope, that decision belongs to plan 26-05 task 1, which now has the same D-11-grade measured evidence bar to reason from that D-11 itself already produced for the write-strategy question.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] RECON-07 exclusion query conflated two different meanings of "classical"**
- **Found during:** Task 1, first run of `tmp/26_queue_inventory_probe.py`
- **Issue:** Block (B)'s RECON-07 split excluded rows whose `telescope_instrument` matched block (A)'s broader run-type CLASSICAL bucket (which includes HCT, Palomar, IRTF, etc.), instead of the narrower set of telescopes that actually have a pre-existing blank-`url` classical `CalendarEvent` (NTT/EFOSC2, Magellan-Baade/IMACS, Magellan-Clay/Lightspeed -- D-19). This undercounted the RECON-07 baseline from the expected 19 down to 8.
- **Fix:** Rebuilt the exclusion set from a direct query (`CalendarEvent.objects.filter(url='').values_list('telescope', flat=True).distinct()`) instead of reusing block (A)'s run-type category, restoring the correct 19-row baseline matching D-20 exactly.
- **Files modified:** `tmp/26_queue_inventory_probe.py` (git-excluded, not committed)
- **Commit:** N/A (throwaway file)

**2. [Rule 1 - Bug] Run-type classifier left 12 of 31 rows uncategorized on the first pass**
- **Found during:** Task 1, first run of the same probe
- **Issue:** The initial classifier only recognized the NTT/EFOSC2/Magellan-family keywords for CLASSICAL, leaving 12 single-facility ground rows (HCT, Palomar, Apache Point, IRTF, Deep Sky Chile, Joan Oró, Deep Random Survey, and the rejected `FOO / BAR` test row) with no assigned category, violating the plan's "assign each row exactly one of three categories" requirement.
- **Fix:** Changed the classifier's fallback branch from "UNCATEGORIZED" to CLASSICAL by default (with an explicit reason string), since a single-facility ground run with no shared-queue-network keyword match is, by the plan's own general rule ("the run owns specific awarded nights at one telescope"), a classically-scheduled run even when it isn't one of the three telescopes already visible on the calendar.
- **Files modified:** `tmp/26_queue_inventory_probe.py` (git-excluded, not committed)
- **Commit:** N/A (throwaway file)

**3. [Rule 1 - Bug] "In-window CalendarEvent count" for the three-way scenario comparison initially counted unrelated events**
- **Found during:** Task 2, first run of `tmp/26_queue_projection_probe.py` (span scenario)
- **Issue:** The initial `in_window_urls()` query filtered only by `start_time__date` range, which coincidentally also matched the 9 real classical NTT/Magellan `CalendarEvent`s (an entirely different campaign/run) whose dates happen to fall inside pk=1's July window -- inflating the measured in-window count from the expected 12/11/15 to 21/20/24.
- **Fix:** Scoped the query to pk=1's own footprint specifically: events matching `telescope__icontains='2m0', instrument='2M0-SCICAM-MUSCAT'` (the 11 real LCO events) OR `url='RUN:1'` OR `url__startswith='RUN:1:'` (this scenario's own minted keys).
- **Files modified:** `tmp/26_queue_projection_probe.py` (git-excluded, not committed)
- **Commit:** N/A (throwaway file)

**4. [Rule 3 - Blocking] DB-path guard's scenario-to-filename mapping didn't account for the `per-night` scenario's hyphen**
- **Found during:** Task 2, first attempt to run the `per-night` scenario
- **Issue:** The guard asserted `db_name.endswith(f'tmp/26-queue-{SCENARIO}-copy.sqlite3')`, but `SCENARIO='per-night'` produces `tmp/26-queue-per-night-copy.sqlite3`, which does not match the plan's actual scratch filename `tmp/26-queue-pernight-copy.sqlite3` (no hyphen) -- the guard correctly aborted rather than silently proceeding, exactly as designed, but blocked forward progress.
- **Fix:** Added an explicit `{'span': 'span', 'none': 'none', 'per-night': 'pernight'}` mapping from scenario name to filename suffix.
- **Files modified:** `tmp/26_queue_projection_probe.py` (git-excluded, not committed)
- **Commit:** N/A (throwaway file)

**5. [Rule 3 - Blocking] `django.test.Client()` calendar fetch returned HTTP 400 DisallowedHost with `SERVER_NAME='localhost'`**
- **Found during:** Task 2, first run of `tmp/26_calendar_render_probe.py`
- **Issue:** Unlike plan 26-01/26-02 (where `ALLOWED_HOSTS=[]` let Django's DEBUG-only "allow localhost automatically" fallback apply), this working tree's uncommitted local `src/fomo/settings.py` modification (per the executor's `working_tree_warning`) sets `ALLOWED_HOSTS=['tlister-thinkmate.lco.gtn', '127.0.0.1']` -- non-empty, so the automatic-localhost fallback no longer applies and `SERVER_NAME='localhost'` was rejected.
- **Fix:** Used `SERVER_NAME='127.0.0.1'` instead, which is explicitly present in this environment's current `ALLOWED_HOSTS`.
- **Files modified:** `tmp/26_calendar_render_probe.py` (git-excluded, not committed)
- **Commit:** N/A (throwaway file)

**6. [Rule 1 - Bug] Calendar-render title-text search initially overcounted due to template truncation and shared titles**
- **Found during:** Task 2, first run of `tmp/26_calendar_render_probe.py` (per-night scenario)
- **Issue:** Searching the rendered page body for the literal minted-event title text ("Didymos 2026") produced misleading counts because (a) `calendar.html:230`'s `truncatechars:18` filter chops every rendered title, and (b) the real classical NTT/EFOSC2 events share an identical short title ("NTT EFOSC2") across multiple distinct rows, so title-text counting conflated separate real events.
- **Fix:** Switched to counting occurrences of each event's exact per-instance `hx-get="{% url 'calendar:update-event' event.id %}"` URL token (`/calendar/update/<pk>/`), which uniquely identifies one specific `CalendarEvent` row regardless of shared or truncated title text.
- **Files modified:** `tmp/26_calendar_render_probe.py` (git-excluded, not committed)
- **Commit:** N/A (throwaway file)

---

**Total deviations:** 6 auto-fixed (3 Rule 1 measurement-methodology bugs in throwaway evidence-gathering scripts, 2 Rule 3 blocking invocation/mapping fixes, 1 Rule 1 render-measurement-precision fix). None touched `26-DECISION.md`'s eventual content incorrectly -- every fix was caught and corrected before any figure was written into the committed doc, and the final recorded numbers (RECON-07=19, in-window counts 12/11/15, `KEY_SET_STABLE` True/True/False) are the corrected, verified values.
**Impact on plan:** All six were mechanical fixes to evidence-gathering tooling, not to the plan's scope or the underlying findings. No scope creep.

## Issues Encountered

- The working tree's pre-existing, uncommitted local modification to `src/fomo/settings.py` (real `ALLOWED_HOSTS`/LCO API key, per the executor's `working_tree_warning` -- never staged, never committed, never touched by this plan) caused `git status --porcelain | grep -v -e '.planning/'` (part of Task 3's literal `<automated>` verify command) to report one non-`.planning/` dirty file. This is the documented, allowed exception the executor's `working_tree_warning` explicitly names, not a plan violation -- confirmed the only three dirty paths at task 3's close were `.planning/STATE.md` (pre-existing orchestrator state, not this plan's), `.planning/phases/26-canonical-record-spike/26-DECISION.md` (this plan's own committed deliverable), and `src/fomo/settings.py` (the named exception).
- No blockers beyond the above and the six auto-fixed deviations. No PII leaked (every grep gate in the plan's own verify commands passed on the final committed state), and `src/fomo_db.sqlite3` never changed across any of the three tasks.

## User Setup Required

None -- no external service configuration required. No checkpoint in this plan (`autonomous: true`, no `type="checkpoint:*"` tasks).

## Next Phase Readiness

- Plan 26-05 can proceed directly: `26-DECISION.md` now carries the full D-11-grade measured evidence bar for the queue-run projection question -- the run-type inventory, the `campaign_gap` over-claim finding, the three-way scenario comparison with idempotency and window-narrowing key-stability tokens, the calendar-render measurement, and the key-scheme consequence -- with no verdict stated. Plan 26-05 task 1 is where that verdict gets locked, using this plan's measurements as its evidence base, mirroring how D-11's write-strategy question was handled.
- The repository is provably identical to its pre-plan-26-04 state apart from the one committed file: `local_settings.py`/`tmp/` are gone from disk and from every git ref, no scratch branch was ever created (this plan created none, unlike 26-01/26-02/26-03), `solsys_code/` has zero diff, and both `python -m pytest` and `./manage.py test solsys_code.tests.test_calendar_template` pass from the pristine tree.
- The real dev DB (`src/fomo_db.sqlite3`) is confirmed byte-identical to its pre-phase-26 fingerprint (`946176 1785094461`) throughout -- unchanged since plan 26-01's original D-04 snapshot, three plans and one gap-closure plan later.
- No blockers.

---
*Phase: 26-canonical-record-spike*
*Completed: 2026-07-27*

## Self-Check: PASSED

- FOUND: `.planning/phases/26-canonical-record-spike/26-DECISION.md`
- FOUND: `.planning/phases/26-canonical-record-spike/26-04-SUMMARY.md`
- FOUND commit: `0eb8141`, `8793e1b`, `66a2433` (real branch)
