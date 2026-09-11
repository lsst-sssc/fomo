---
phase: 34-the-observation-projector-trigger
fixed_at: 2026-09-11T12:13:33Z
review_path: .planning/phases/34-the-observation-projector-trigger/34-REVIEW.md
iteration: 1
findings_in_scope: 17
fixed: 17
skipped: 0
status: all_fixed
---

# Phase 34: Code Review Fix Report

**Fixed at:** 2026-09-11T12:13:33Z
**Source review:** .planning/phases/34-the-observation-projector-trigger/34-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 17 (CR-01, CR-02, WR-01 through WR-09, IN-01 through IN-06 --
  the `critical_warning` pass fixed the first 11; this `--all` pass added the six
  Info findings)
- Fixed: 17
- Skipped: 0 (CR-01's facility-URL-namespace sub-part is closed by analysis, not code --
  see that entry below)

**Verification environment:** every fix below was edited, linted, and test-run inside
an isolated git worktree this run created, then fast-forwarded onto
`issue37-telescope-runs-calendar` and the worktree removed as part of this run's
cleanup. The critical/warning pass (11 findings, first run) used
`.claude/worktrees/rf-34-1025690-1789123413` (branch `gsd-reviewfix/34-1025690`); this
`--all` pass (six Info findings, second run) used
`.claude/worktrees/rf-34-1094896-1789128415` (branch `gsd-reviewfix/34-1094896`). Both
worktrees' `src/fomo/_version.py` (a gitignored, `setuptools_scm`-generated file
`manage.py` needs at import time) does not exist in a fresh worktree checkout; it was
copied over from the main checkout's copy (not committed, not part of any fix) purely
so `python manage.py test` could run at all -- this is a build artifact, not a source
change. The numbers below are reproducible from the main checkout
(`issue37-telescope-runs-calendar`) after all 17 commits landed (both passes).

## Fixed Issues

### CR-01: The projector's identity key is not unique — records with a blank or duplicate `observation_id` silently overwrite each other's event and steal the companion link

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/tests/test_observation_projector.py`
**Commit:** `5767a7a`
**Applied fix:** `event_fields_for()` now raises `ValueError` for a blank/whitespace-only
`observation_id` before it can be handed to `event_url()`, which would otherwise map
every such record to the same bare facility listing URL and silently collapse them onto
one `CalendarEvent`, stealing each other's `CalendarEventMeta.observation_record` link.
`project_record()` already catches this and reports the record as `unprojectable`
(consistent with the fix suggestion's code snippet). Two regression tests added: two
blank-`observation_id` records never collide, and a whitespace-only id is also rejected.

**Resolved by analysis:** the review also flagged that LCO and SOAR share the identical
`portal_url` (`settings.py:230/238`), raising the same event URL for a same-id LCO/SOAR
pair as a possible collision. Analysis on 2026-09-11 closes this as intended behaviour,
not a defect, for five reasons.

1. `SOARFacility` subclasses `LCOFacility` and `SOARSettings` subclasses `LCOSettings`
   (`tom_observations/facilities/soar.py`), so SOAR observations are scheduled through the
   same LCO Observation Portal and their `observation_id` values live in one shared
   request-ID space.
2. An LCO record and a SOAR record carrying the same `observation_id` are therefore the
   same portal request, so `https://observe.lco.global/requests/<id>` is that request's
   correct single identity -- namespacing `event_url()` per facility would wrongly split
   one real request into two calendar events.
3. Every FOMO writer -- the projector, the reconciler and `load_telescope_runs` -- finds
   the event by `url` and creates it only if missing, so the only remaining way to get two
   rows with the same non-blank url is a person editing `url` by hand in tom_calendar's
   event form, which CR-02's fix (above) already catches and reports as `unprojectable`.
4. The evidence, observed 2026-09-11 on the developer database: 241 calendar events, 0
   duplicate non-blank urls, 10 blank urls.
5. The decision, made by the user: no schema change. A partial unique index on
   `tom_calendar_calendarevent(url)` limited to non-blank urls stays available as a
   possible later hardening, but it was considered and explicitly not chosen -- so
   nothing here is outstanding work.

### CR-02: A single duplicate-url CalendarEvent permanently breaks a record's projection, and `project_record()` does not catch it despite promising "Never raises"

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/tests/test_observation_projector.py`
**Commit:** `e3b274b`
**Applied fix:** every write `project_record()` makes -- resolving the facility, building
the field dict, the create-or-update itself (`insert_or_create_calendar_event()`), and
the companion-row write (`write_event_meta()`) -- now lives inside one `try`, so a
`CalendarEvent.objects.get_or_create(url=...)` collision (`MultipleObjectsReturned` from
a duplicate-url row, reachable through the unauthenticated event form per the review's own
repro) is caught here and reported as `unprojectable`, matching the docstring's "Never
raises" promise for the first time. Left `insert_or_create_calendar_event()`'s
`get_or_create()` as-is (the fix's "and/or" alternative) -- catching the exception here is
sufficient and does not require touching `calendar_utils.py`'s shared helper, which
`load_telescope_runs`/`sync_gemini_observation_calendar` also depend on. Added a
regression test creating two duplicate-url events directly and asserting
`project_record()` returns `('unprojectable', 'MultipleObjectsReturned')` instead of
raising.

### WR-01: The never-raise wrappers cannot protect the caller's save from a database error, and three CharFields are written unbounded

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/tests/test_observation_projector.py`
**Commit:** `2794928`
**Applied fix:** `telescope`/`instrument`/`proposal` are now truncated to `[:200]` in
`event_fields_for()`'s field dict, matching `title`'s existing truncation, so an
externally-sourced over-length value (SQLite accepts it silently; PostgreSQL -- the
documented production target -- raises `DataError`) can no longer reach the database
unbounded. `receiver_on_record_save()` now runs `project_record()` inside its own
`transaction.atomic()` savepoint: per Django's documented `needs_rollback` behavior, this
lets a database error be recovered from (the atomic block's own `__exit__` issues a
`ROLLBACK TO SAVEPOINT` when the connection is marked as needing one, even though the
exception is caught inside the block by `project_record()`'s own `except`), so the
receiver's existing broad `except` continues to protect the caller's outer transaction
instead of leaving it unusable for every later query. Added a regression test proving
proposal/instrument truncate to exactly 200 characters.

### WR-02: `--dry-run` provably *can* disagree with a real run — the "structurally unable to disagree" claim is false whenever a site lookup fires

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/management/commands/project_observation_calendar.py`, `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `8747ca3`
**Applied fix:** softened the docstrings in both Python files and added a paragraph to
the runbook naming the one documented exception explicitly: a dry run never performs the
one-time observed-site lookup, so `site_lookups` is always 0 in a dry run and a record
whose only pending change is the coarse-to-observed telescope token is reported
`unchanged` by `--dry-run` but `updated` by the real sweep that follows it. Chose the
"soften the docs" option over "make the dry run predict the lookup" -- the latter would
require either making the live network call `--dry-run` exists to avoid, or faking a
result, both of which change dry-run's own no-network-call guarantee.

### WR-03: `observation_series_decoration()` publishes observation-group identity on an unauthenticated view with no visibility gate, unlike its sibling tag

**Files modified:** `solsys_code/templatetags/calendar_display_extras.py`, `solsys_code/tests/test_calendar_template.py`
**Commit:** `9f96920`
**Applied fix:** added the same `run is not None and not run.is_publicly_visible` gate
`campaign_decoration()` already applies, so a pending-review run's attribution no longer
leaks the observation-group's own name (an internal portal `RequestGroup` identifier) to
an anonymous visitor of the unauthenticated event-update modal. Added a cross-reference
comment at each tag's own gate, per the finding's own request to keep the two visibility
rules legible side by side. Added a regression test (`CampaignRun.ApprovalStatus.
PENDING_REVIEW`) proving both the group name and the "Night n of N" text are suppressed.

### WR-04: `_window_start_or_max()` catches a narrower exception set than the code path it guards, so the modal can still 500

**Files modified:** `solsys_code/templatetags/calendar_display_extras.py`, `solsys_code/tests/test_calendar_display_extras.py`
**Commit:** `10072e0`
**Applied fix:** widened the `except (KeyError, ValueError)` to a bare `except Exception`
(matching the discipline `event_fields_for()` already uses for the same
`datetime.fromisoformat()` parsing call), since a JSON number/boolean/null stored in
`parameters['start']` raises `TypeError`, not `ValueError`. Added a regression test with
`parameters={'start': 12345, 'end': 12346}` proving the malformed sibling sorts last
instead of raising out of `list.sort()`.

### WR-05: The PROJ-05 prefetch widening targets the wrong view, joins a relation nothing reads, and its regression test is vacuous

**Files modified:** `solsys_code/views.py`, `solsys_code/tests/test_calendar_template.py`
**Commit:** `19aeecc`
**Applied fix:** dropped `observation_record__target` and `observation_group` from the
month view's `select_related` (kept `run__campaign`, which `campaign_decoration()` does
dereference per attributed event in `calendar.html`) and corrected the comment to say the
tag runs in the modal view (`calendar:update-event`, `tom_calendar.views.update_event`),
which fetches its one event by pk with no `select_related` of its own -- so there was
never a month-cell consumer for the two dropped joins. Replaced the vacuous
`test_month_view_query_count_does_not_grow_with_second_grouped_event` (which measured
`calendar:calendar`, a view the tag never renders, and so passed unconditionally) with
`test_modal_query_count_does_not_grow_with_group_size`, which measures the actual view
the tag runs in across a 2-member vs. a 10-member group.

### WR-06: `is_verified` is now write-only-`True`, leaving two dead template branches and a misleading model field

**Files modified:** `solsys_code/models.py`, `src/templates/tom_calendar/partials/calendar.html`
**Commit:** `41d9510`
**Applied fix:** chose the finding's alternative option (documentation, not deletion) --
deleting the two `calendar.html` branches would break several existing tests (and
`admin.py`'s own `list_filter`) that deliberately construct a `CalendarEventMeta` with
`is_verified=False` directly, so removal is a larger, separately-scoped change than this
finding covers. Added an explicit code comment at each branch (`{% comment %}` block for
the multi-line one -- Django's `{# #}` tag does not support multi-line content, caught by
this repo's own `test_modal_renders_no_django_comment_delimiters`-style source scan) and a
note in `CalendarEventMeta`'s class docstring describing what the field currently means:
no writer in this codebase sets it `False` any more. `verbose_name` text is left
unchanged, since editing it would need a migration -- out of scope for a docstring/comment
fix. Real branch removal is filed as a later-phase follow-up, per the finding's own text.

### WR-07: The D-07 telescope rename moves the projector's token into `load_telescope_runs`' classical lookup vocabulary

**Files modified:** `solsys_code/management/commands/load_telescope_runs.py`, `solsys_code/tests/test_load_telescope_runs.py`
**Commit:** `135d98a`
**Applied fix:** added `url=''` to `load_telescope_runs`' find-or-create lookup dict, so
this classical writer can only ever match a blank-url (classically scheduled) event -- the
same namespace discipline the reconciler and the projector both already apply. A
projector-owned event whose `telescope`/`instrument` happen to collide with the classical
vocabulary (now structurally possible after the D-07 rename put `'FTS'`/`'FTN'`/`'SOAR'`
in both writers' vocabularies) can no longer be adopted and rewritten. Added a regression
test that seeds a projector-owned event at the exact `(telescope, instrument, start_time)`
triple a classical schedule line computes, then asserts the classical run leaves it
untouched and creates a separate blank-url event instead.

### WR-08: `_cleared_group_members` is an unbounded, process-lifetime, non-thread-safe module global

**Files modified:** `solsys_code/observation_projector.py`
**Commit:** `9b62d67`
**Applied fix:** chose the finding's "re-derive from the DB" alternative over
bounding/scoping the pre_clear capture. `post_clear` now queries
`CalendarEventMeta.objects.filter(observation_group_id=instance.pk)` directly:
`observation_group` is only ever written by `write_event_meta()` at projection time, so
any companion row still pointing at the just-cleared group is exactly a record that needs
re-projecting. This removes the module global, the `pre_clear` branch, and the whole
leak/thread-safety class of failure entirely, rather than mitigating it. Existing coverage
(`test_group_clear_reprojects_every_former_member`) already exercises this path and
continues to pass unchanged, so no new test was needed for the behavior itself.

### WR-09: The notebook's convergence assertion claims more than it checks

**Files modified:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
**Commit:** `da0354d`
**Applied fix:** replaced the substring-in-the-whole-string loop with the fix's own
suggested code -- split `second_sweep_summary` on `' | '` and assert each facility
segment independently, so a non-converged LCO sweep can no longer hide behind the
all-zero SOAR segment. This was a **source-only edit, not a re-execution**: the cell's
`assert` produces no output on success, so the committed output (two `print()` blocks) is
unaffected by the corrected logic either way, and re-running the corrected assertion
against the exact committed `second_sweep_summary` string (verified by hand, see the
commit message) confirms it still passes silently -- the committed output remains a
faithful record of what the corrected code does. Full re-execution
(`jupyter nbconvert --to notebook --execute --inplace
project_observation_calendar_demo.ipynb`) was not performed in this run: this notebook
makes real LCO Observation Portal API calls (the one-time observed-site lookup) and runs
directly against the developer database (not a scratch copy, per the notebook's own
framing), neither of which is reproducible or safe inside this automated fix session. If
a maintainer re-executes this notebook for an unrelated reason, back up
`project_observation_calendar_demo.sched06-baseline.json` first and restore it afterward
per IN-06's own note (still open, out of this fix's scope).

## Fixed Issues (Info, --all pass)

### IN-01: `telescope_match_score()`'s documented resolution order is now stale

**Files modified:** `solsys_code/campaign_attribution.py`
**Commit:** `18c2bf0`
**Applied fix:** the step-2 worked example (`'FTS'`) was stale -- D-07 (34-02 Task 3)
made `_extract_lco_site_code()` resolve an *observed* telescope token via
`OBSERVED_TELESCOPE_SITE_CODES` at step 1, so `'FTS'`/`'FTN'` can never reach step 2's
classical-alias branch any more. Swapped the example to `'NTT'` (a `telescope_runs.SITES`
key with no observed-site entry, so it genuinely still reaches step 2), added a note at
step 1 explaining why `'FTS'` no longer falls through, and documented the evidence-string
wording difference between the two steps ("LCO site code '...' resolves to obscode ..."
at step 1 vs. "classical site alias for ..." at step 2 -- same score, different text).

### IN-02: The runbook's ring description omits the failure markers

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `371fc1b`
**Applied fix:** appended the finding's own suggested clause -- "...and an expired,
cancelled or failed entry carries the terminal ring" -- to the status-legend paragraph,
so the ring sentence now covers all three ringed cases `status_border_css()` actually
implements (`[Q]`, `[?]`, and the `_TERMINAL_PREFIXES` set covering `[X]`/`[C]`/`[F]`),
not just the first two.

### IN-03: `[?]` is unreachable for a record that is both inconsistent and in a failure state

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `bfb9350`
**Applied fix:** took the finding's lower-risk option (runbook caveat, not a precedence
change) per this run's own instruction -- changing `title_for()`'s marker precedence
would be a behavior change requiring the paired notebook to be re-executed, which this
run does not do (real LCO API calls, direct developer-database writes). Added a sentence
to the `[?]` row stating that a failure marker always wins over `[?]` and that a record
that is both inconsistent and window-expired/cancelled/failed therefore does not get the
"visible on the calendar" treatment the row otherwise promises.

### IN-04: Two hand-maintained copies of the same six counter keys

**Files modified:** `solsys_code/observation_projector.py`, `solsys_code/management/commands/project_observation_calendar.py`
**Commit:** `91bd188`
**Applied fix:** renamed `observation_projector._SWEEP_COUNTER_KEYS` to the public
`SWEEP_COUNTER_KEYS` (dropped the leading underscore, per this run's instruction to keep
the exported name public) and had the command import it instead of hand-copying the same
six-tuple as `_COUNTER_KEYS`. `_COUNTER_KEYS` is kept as a local alias inside the command
(`_COUNTER_KEYS = SWEEP_COUNTER_KEYS`) so every existing call site (`_new_counters()`)
needed no further edit. Updated the explanatory comment on both sides of the import.
Behavior-neutral: same six values, same order.

### IN-05: `observed_enclosure` is written and never read; `select_related('target')` is fetched and never used

**Files modified:** `solsys_code/templatetags/calendar_display_extras.py`, `solsys_code/management/commands/project_observation_calendar.py`
**Commit:** `13aec1d`
**Applied fix:** dropped the unused `select_related('target')` from
`observation_series_decoration()`'s group-member query -- the tag only ever dereferences
`member.pk` and `record_time_window(member)` (which reads `member.parameters`, never
`member.target`), so the join added a LEFT JOIN per group member for data nothing reads.
For the `observed_enclosure` write: checked `.planning/ROADMAP.md`'s Phase 35 (Allocation
Layer & Classical Cutover), Phase 36 (Unattended Operation) and Phase 37 (Status
Vocabulary, Public Tallies & Provenance-Blind Gaps) entries, plus the Phase 34 planning
docs (`34-RESEARCH.md`, `34-CONTEXT.md`, `34-02-PLAN.md`) that introduced the field --
**none of them names `observed_enclosure` as something a later phase plans to consume.**
Per this run's instruction, the write was left in place rather than removed (removing on
a guess risks silently discarding data a not-yet-planned future consumer might want), and
a comment was added at the write site stating plainly that no phase currently plans to
read it, and why it is kept anyway (same portal placement block as the two keys that
already have a reader; the key is already reserved by `OBSERVED_SITE_PARAMETER_KEYS`).

### IN-06: Re-executing the demo notebook overwrites the SCHED-06 baseline it tells you to diff against

**Files modified:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`
**Commit:** `0c71828`
**Applied fix:** took the finding's second option -- amended cell 17's instruction text
rather than changing cell 15's filename -- since renaming the baseline file's output
would itself be a behavior change needing re-execution to verify. The new wording states
plainly that cell 15's baseline write overwrites
`project_observation_calendar_demo.sched06-baseline.json` in place on every run, and
directs the reader to `git diff` on that file instead of the old "diffable by eye against
the JSON file this run wrote" phrasing, which was true only before the file described
already got replaced by the run producing it. **Source-only edit, not a re-execution**:
verified with a targeted text replacement (`git diff` on the commit shows only the
markdown `source` array changed, no output cell touched, valid notebook JSON confirmed by
reloading with `json.load()`). The notebook was not re-executed for the same reason
WR-09's fix was not: it makes real LCO Observation Portal API calls and writes directly
to the developer database, neither of which this automated session can safely trigger.

## Notes and Follow-ups

- **CR-01 facility-URL-namespace question** (LCO and SOAR sharing `portal_url`) is closed --
  see the CR-01 entry above's **Resolved by analysis** paragraph for the shared-request-ID
  rationale and the user's no-schema-change decision.
- **IN-01 through IN-06** were out of scope for the first (`critical_warning`) pass and
  are now closed by this `--all` pass, above. None of the earlier WR-*/CR-* fixes had
  happened to resolve any of them as a side effect. IN-06 (the SCHED-06 baseline JSON
  getting overwritten on re-execution) is now fixed as prose, but the underlying
  re-execution itself -- for both WR-09 and IN-06 -- remains a manual follow-up an
  operator must perform once real observing nights have passed (see `34-UAT.md`).
- IN-05's `observed_enclosure` write is left in place with no current planned reader
  (see that entry above) -- if a future phase decides it will never be read, removing the
  write and the `observed_enclosure` key from `OBSERVED_SITE_PARAMETER_KEYS` is a small,
  separately-scoped follow-up at that point.
- All 184 tests across the four modules this `--all` pass's own findings touch
  (`test_observation_projector`, `test_project_observation_calendar`,
  `test_calendar_display_extras`, `test_campaign_attribution`) pass after all six IN-*
  commits, run together in one invocation. Combined with the first pass's 526-test run,
  every module this phase's review touched has been exercised green after all 17 fixes.
  `pre-commit run ruff`/`ruff-format` are clean on every changed Python file (checked
  per-commit, not repo-wide). `pre-commit run sphinx-build` (part of every commit's own
  hook run) built cleanly after each `.rst`/docstring edit (IN-01, IN-02, IN-03).

---

_Fixed: 2026-09-11T12:13:33Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
