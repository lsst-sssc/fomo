---
phase: quick-260805-sgf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_reconciler.py
  - solsys_code/tests/test_campaign_reconciler.py
  - solsys_code/tests/test_campaign_approval.py
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
autonomous: true
requirements: [SGF-01]

must_haves:
  truths:
    - "A reconciler-created calendar event for a run whose telescope_instrument is 'Apache Point Observatory/ARCTIC' has telescope='Apache Point Observatory' and instrument='ARCTIC', not the whole combined string under telescope with instrument blank"
    - "Both write branches split: the container branch (queue/class-wide/satellite) and the classical per-night create path"
    - "A '+'-separated telescope_instrument splits the same way a '/'-separated one does"
    - "A telescope_instrument with no delimiter at all still lands wholly in telescope with instrument blank -- today's single-token and space-separated values do not regress"
    - "The calendar event TITLE still carries the full combined 'campaign: telescope/instrument' string -- only the two fields are split"
    - "The classical branch's update path still never rewrites telescope or instrument, so an adopted load_telescope_runs event keeps its own telescope/instrument values"
    - "All 161 pre-existing tests across test_campaign_reconciler.py, test_reconcile_campaign_runs.py and test_campaign_approval.py still pass (one assertion in test_campaign_approval.py is deliberately revised to assert the split)"
  artifacts:
    - path: "solsys_code/campaign_reconciler.py"
      provides: "_split_telescope_instrument() pure helper plus its use at both write sites"
      contains: "def _split_telescope_instrument"
    - path: "solsys_code/tests/test_campaign_reconciler.py"
      provides: "Helper-level and branch-level tests for the split"
      contains: "_split_telescope_instrument"
    - path: "docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb"
      provides: "Paired demo notebook exercising the split with real executed output"
      contains: "instrument="
    - path: "docs/runbooks/telescope_runs_calendar.rst"
      provides: "Operator-facing note on what the pop-up's Telescope / Instrument fields now show"
      contains: "Instrument"
  key_links:
    - from: "solsys_code/campaign_reconciler.py::_reconcile_container"
      to: "_split_telescope_instrument"
      via: "fields dict builds 'telescope' and 'instrument' from the split halves"
      pattern: "_split_telescope_instrument\\(run\\.telescope_instrument\\)"
    - from: "solsys_code/campaign_reconciler.py::_reconcile_classical_nights"
      to: "_split_telescope_instrument"
      via: "create-only fields dict builds 'telescope' and 'instrument' from the split halves"
      pattern: "'instrument':"
---

<objective>
`CampaignRun.telescope_instrument` is free text submitters fill in as `<telescope>/<instrument>`
or `<telescope>+<instrument>` (`'FTN/MuSCAT3'`, `'Apache Point Observatory/ARCTIC'`) -- a
convention `campaign_reconciler.py`'s own `_adopted_event_for_night()` docstring already names.
But both reconciler write sites push the ENTIRE combined string into `CalendarEvent.telescope`
and never populate `CalendarEvent.instrument`. The event-detail pop-up
(`src/templates/tom_calendar/partials/event_form.html:76-81`) renders those as two separate
"Telescope" / "Instrument" form fields, so every reconciler-created event shows the whole
combined string under Telescope and leaves Instrument blank. Confirmed live on the dev DB:
RUN:10's event has Telescope="Apache Point Observatory/ARCTIC", Instrument="".

Purpose: the pop-up shows the right value in the right field, and the two fields carry the
same content shape that `load_telescope_runs`-created events already carry (`telescope='FTN'`,
`instrument='MuSCAT3'`) -- which also gives Phase 28's attribution scoring real instrument text
to compare on a detached event.

Output: one small pure helper in `campaign_reconciler.py` used at both write sites, tests
covering both branches and both delimiters plus the no-delimiter fallback, and the paired demo
notebook and operator runbook updated per CLAUDE.md's paired-docs rule.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md
@solsys_code/campaign_reconciler.py
@solsys_code/tests/test_campaign_reconciler.py
</context>

<interfaces>

## The two write sites (both currently `'telescope': run.telescope_instrument`)

- `solsys_code/campaign_reconciler.py:231` -- inside `_reconcile_container()`'s `fields` dict.
  This branch is authoritative for every field on **both create and update**, so existing
  container events self-heal on the next sweep.
- `solsys_code/campaign_reconciler.py:369` -- inside `_reconcile_classical_nights()`'s
  **create-only** `fields` dict (the `if existing is None:` arm). The `else:` arm
  (`fields = common_fields`) must stay as it is: `telescope`/`instrument`/`start_time`/`end_time`
  are deliberately never rewritten after creation, because an adopted `load_telescope_runs`
  event already carries its own correct `telescope='FTN'` / `instrument='MuSCAT3'` and its own
  more-precise file-derived window. Operational consequence to accept and document: existing
  per-night classical events keep their combined string until they are recreated; container
  events fix themselves on the next sweep.

## Target model fields (verified)

`tom_calendar.models.CalendarEvent` has both `telescope` and `instrument` as
`CharField(max_length=200, blank=True, default="")`. Splitting can only shorten each half, so
no length concern is introduced.

## Explicitly out of scope -- do not touch

- `event_title()` (line 144) -- the title must keep showing the full combined
  `"campaign: telescope/instrument"` string. Only the two FIELDS split.
- `_adopted_event_for_night()` (line 252) -- its docstring explicitly says matching must NOT
  compare telescope text; that is a deliberate, unrelated decision about a different code path.
- `_skip_reason()` (line 164), `QUEUE_SOURCES` (line 75) and all `source`-based branching --
  a separate, larger fix is planned for that dispatch logic.

## Pre-existing assertion that MUST be revised (found during planning)

`solsys_code/tests/test_campaign_approval.py:380` asserts
`self.assertEqual(event.telescope, run.telescope_instrument)` against a run whose
`_make_pending_run()` fixture value is `'FTN/MuSCAT3'` (line 149). This assertion becomes false
by design once the split lands. It is the only such assertion in the repo (grep confirmed: no
other non-test consumer reads a reconciler-written `event.telescope`, and
`test_calendar_template.py:484`'s `assertIn('FTN/MuSCAT3', content)` reads
`run.telescope_instrument` through the modal, not the event field). Revise it to assert the
split -- that turns it into an end-to-end proof of the fix through the real approval path.

## Other readers of these fields (no change needed, no regression)

- `solsys_code/templatetags/calendar_display_extras.py:409` buckets the telescope legend on
  `(event.telescope or '').strip().upper()`. A `RUN:` event's bucket becomes `FTN` instead of
  `FTN/MUSCAT3` -- it now shares a bucket with `load_telescope_runs` events for the same
  telescope, which is the intended behaviour, not a regression.
- `solsys_code/campaign_attribution.py:541` scores a detached orphan's
  `(event.telescope, event.instrument)` against `run.telescope_instrument` with a tokenised
  similarity. A populated `instrument` half can only raise that score.

## Measured test baseline (planning time, 2026-08-05)

`python manage.py test solsys_code.tests.test_campaign_reconciler
solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval`
reports `Ran 161 tests` / `OK` in ~162s. Use `python manage.py` (not `./manage.py`).

</interfaces>

<paired_docs_assessment>
CLAUDE.md's paired-docs rule applies. `solsys_code/campaign_reconciler.py` is paired with
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`, and `docs/runbooks/` is in
scope by directory (today's only page: `docs/runbooks/telescope_runs_calendar.rst`). This is new
extraction logic, not a pure refactor, so the trigger fires -- both artifacts are in
`files_modified` up front, and Task 3 covers them.

Grepped, not assumed:

- **Notebook**: its only executed field dump (code cell index 10) prints `url`, `title`, `start`
  and `end` per event -- it never prints `.telescope` or `.instrument`, and all three seeded
  fixtures use delimiter-free strings (`'RDGS EFOSC2'`, `'RDGS 1m0-SciCam-Sinistro'`,
  `'LCO 1m0 Network (demo)'`), so no committed output line reads differently by itself. It
  therefore does not currently exercise the new behaviour at all, which is exactly what the rule
  requires it to do -- Task 3 makes it demonstrate the split with real executed output.
- **Runbook**: it never quotes a `CalendarEvent.telescope` field value, so nothing in it is
  stale. But its "What an operator sees on the calendar afterwards, in plain terms" paragraph
  (lines 548-553) is precisely the documented behaviour this change alters from the operator's
  point of view, so Task 3 adds a short paragraph there.
</paired_docs_assessment>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add _split_telescope_instrument() and use it at both reconciler write sites</name>
  <files>solsys_code/campaign_reconciler.py, solsys_code/tests/test_campaign_reconciler.py</files>
  <behavior>
    Helper-level tests (new class `TestSplitTelescopeInstrumentHelper` in
    `test_campaign_reconciler.py`, importing `_split_telescope_instrument` alongside the existing
    `campaign_reconciler` imports at line 21):
    - Test 1: a slash-separated value splits and strips both halves --
      `' FTN / MuSCAT3 '` returns `('FTN', 'MuSCAT3')`.
    - Test 2: a plus-separated value splits the same way --
      `'Apache Point Observatory+ARCTIC'` returns `('Apache Point Observatory', 'ARCTIC')`.
    - Test 3: no delimiter falls back to the whole string as telescope with a blank instrument --
      `'NTT EFOSC2'` returns `('NTT EFOSC2', '')` and `'SomeScope'` returns `('SomeScope', '')`.
    - Test 4: only the FIRST delimiter splits -- `'A/B/C'` returns `('A', 'B/C')` and `'A/B+C'`
      returns `('A', 'B+C')`.
  </behavior>
  <action>
Add a module-level pure helper to `solsys_code/campaign_reconciler.py`, placed just above
`event_title()` (i.e. after `writable_events()`, before line 144) so it reads next to the other
small run-to-event derivations:

    def _split_telescope_instrument(text: str) -> tuple[str, str]

It splits `text` on the FIRST `/` or `+` delimiter using `re.split` with the character class
`[/+]` and `maxsplit=1`, strips whitespace from both halves, and returns
`(telescope, instrument)`. When no delimiter is present it returns `(text.strip(), '')` -- the
whole string as telescope, empty instrument -- the safe fallback that keeps today's single-token
and space-separated values (`'NTT EFOSC2'`, `'LCO 1m0 Network (demo)'`) landing exactly where
they land now. Add `import re` to the stdlib import block at the top of the module (it is not
imported today).

Give it a Google-style docstring in this module's established voice: name the free-text
`<telescope>/<instrument>` submitter convention `_adopted_event_for_night()`'s docstring already
cites, say plainly why the fallback returns the whole string rather than guessing, and note that
a leading delimiter (`'/MuSCAT3'`) therefore yields a blank telescope half -- deliberately not
special-cased, because `_skip_reason()` already rejects a wholly blank `telescope_instrument`
and inventing a second fallback rule here would be un-asked-for behaviour.

Then use it at both write sites, replacing `'telescope': run.telescope_instrument` with the two
split halves:

1. `_reconcile_container()` (the `fields` dict at line 227-234) -- unpack the helper once above
   the dict and set both `'telescope'` and `'instrument'` keys from the halves.
2. `_reconcile_classical_nights()` (the create-only `fields` dict at line 367-372, inside the
   `if existing is None:` arm) -- same two keys. Leave the `else: fields = common_fields` arm
   untouched, and leave `common_fields` untouched.

Change nothing else in the module: `event_title()` keeps emitting the full combined string,
`_adopted_event_for_night()`'s matching logic is untouched, and `_skip_reason()`,
`QUEUE_SOURCES` and every `source`-based branch are out of scope.

Two docstrings enumerate the per-branch field lists and become inaccurate without a word each --
update both, minimally:
- The module header docstring's closing paragraph (lines 26-30): its
  "``start_time``/``end_time``/``telescope`` are never rewritten after creation" list gains
  `instrument`.
- `_reconcile_classical_nights()`'s docstring "Field authority" paragraph (lines 326-333): both
  its create list ("writes ``title``, ``description``, ``target_list``, ``telescope``,
  ``start_time``, ``end_time``") and its never-rewritten list gain `instrument`, and add one
  clause to reason (a) making the consequence explicit: an adopted `load_telescope_runs` event
  already carries its own correct `telescope`/`instrument` pair, which must not be overwritten
  by this run's free text.

Write the four helper-level tests from `<behavior>` as a new test class at the end of
`solsys_code/tests/test_campaign_reconciler.py`. These are pure-function tests -- subclass
`TestCase` directly rather than `CampaignReconcilerTestBase`; no DB fixture is needed.
  </action>
  <verify>
    <automated>python manage.py test solsys_code.tests.test_campaign_reconciler.TestSplitTelescopeInstrumentHelper 2>&1 | tail -5</automated>
  </verify>
  <done>`_split_telescope_instrument()` exists, is used at both write sites, both affected docstrings name `instrument`, and the four helper-level tests pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Prove the split on real events through both branches; revise the one stale assertion</name>
  <files>solsys_code/tests/test_campaign_reconciler.py, solsys_code/tests/test_campaign_approval.py</files>
  <behavior>
    New class `TestTelescopeInstrumentSplitOnEvents(CampaignReconcilerTestBase)` in
    `test_campaign_reconciler.py` -- five tests, each reconciling a real run and asserting on the
    written `CalendarEvent`:
    - Test 1 (container branch): a queue run (`source=CampaignRun.Source.LCO_QUEUE`) with the
      base fixture's `'FTN/MuSCAT3'` produces a bare `RUN:{pk}` event with `telescope == 'FTN'`
      and `instrument == 'MuSCAT3'`.
    - Test 2 (classical create path): a single-night classical run with `'FTN/MuSCAT3'` produces
      a `RUN:{pk}:{date}` event with `telescope == 'FTN'` and `instrument == 'MuSCAT3'`.
    - Test 3 (plus delimiter): a queue run with
      `telescope_instrument='Apache Point Observatory+ARCTIC'` produces
      `telescope == 'Apache Point Observatory'`, `instrument == 'ARCTIC'`.
    - Test 4 (no-delimiter fallback): a classical run with `telescope_instrument='NTT EFOSC2'`
      produces `telescope == 'NTT EFOSC2'` and `instrument == ''` -- today's behaviour preserved.
    - Test 5 (title guard): the same run as Test 1 still has the full combined string in its
      title -- assert `'FTN/MuSCAT3' in event.title` and `event.title == event_title(run)`, so a
      future change to `event_title()` cannot silently split the title too.
  </behavior>
  <action>
Add the five tests from `<behavior>` as a new class at the end of
`solsys_code/tests/test_campaign_reconciler.py`, using the existing
`CampaignReconcilerTestBase._make_run(**overrides)` fixture helper and matching the surrounding
file's style: a one-line class docstring naming what the class proves, `reconcile_run(run)` then
assertions on `CalendarEvent.objects.get(url=...)`. Reuse `run_night_url`-shaped f-string keys
the way the neighbouring `TestClassicalStage1` tests do
(`f'RUN:{run.pk}:{night.isoformat()}'`) rather than introducing a new import.

Then revise the one pre-existing assertion this fix deliberately invalidates:
`solsys_code/tests/test_campaign_approval.py:380`, currently
`self.assertEqual(event.telescope, run.telescope_instrument)` inside
`test_approve_single_night_ground_run_creates_dip_corrected_calendar_event`. Its run fixture is
`'FTN/MuSCAT3'` (`_make_pending_run()`, line 149), so replace that single line with the split
pair -- assert `event.telescope == 'FTN'` and `event.instrument == 'MuSCAT3'` -- and add a short
inline comment noting that the run's own `telescope_instrument` still holds the full combined
string, which the event title still carries. This is the fix proved end-to-end through the real
staff-approval path, not just the unit path. Change nothing else in that file.

Then run the full three-module suite and the lint gates.
  </action>
  <verify>
    <automated>python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval 2>&1 | tail -5 && ruff check . && ruff format --check .</automated>
  </verify>
  <done>The three-module run reports OK with at least 170 tests (161 baseline + 9 new), and both ruff gates are clean.</done>
</task>

<task type="auto">
  <name>Task 3: Update the paired demo notebook and the operator runbook</name>
  <files>docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb, docs/runbooks/telescope_runs_calendar.rst</files>
  <action>
**Notebook** (`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`), three edits then
a regeneration:

1. Code cell index 6 (the three `CampaignRun` seeds): change the classical run's
   `telescope_instrument` from `'RDGS EFOSC2'` to `'RDGS/EFOSC2'` so the executed output shows a
   real split. Leave the queue run (`'RDGS 1m0-SciCam-Sinistro'`) and the class-wide run
   (`'LCO 1m0 Network (demo)'`) exactly as they are -- they are the no-delimiter fallback, and
   having both cases visible in one output block is the point. Accepted consequence, already
   true of any seed-key change in this notebook: on a database where the old `'RDGS EFOSC2'` row
   already exists, the `update_or_create` natural key no longer matches it and a new row is
   created alongside; the regeneration below runs against an empty DB, where this does not arise.
2. Code cell index 10 (the per-event print loop after the real sweep): add
   `telescope=` and `instrument=` to the printed lines, e.g. a third print line
   `f'    telescope={ev.telescope!r}  instrument={ev.instrument!r}'` under the existing
   `title=` line, so every event's two fields appear in committed output.
3. The markdown cell immediately above it (index 9, "## The real sweep"): add two or three
   sentences in the notebook's existing plain-English voice explaining that submitters write
   `Telescope / Instrument` as free text using `/` or `+`, that the reconciler splits it on the
   first delimiter into the calendar event's two separate `telescope` and `instrument` fields --
   which is what the event-detail pop-up renders as its "Telescope" and "Instrument" boxes -- and
   that a value with no delimiter goes wholly into `telescope`, as the queue and class-wide rows
   below show.

Then regenerate with real executed output, committed with output per the `pre_executed/`
convention (`.pre-commit-config.yaml:20` excludes this directory from the output-clearing hook).

The notebook MUST be executed against an empty, freshly-migrated database. Against the populated
dev DB the sweep would report all ~26 real campaign runs and the committed pks, counters and
`CalendarEvent.objects.count()` lines would become meaningless. From the repo root:

- If `src/fomo_db.sqlite3` exists in the working tree, move it aside first (it is gitignored --
  never commit it, never delete the operator's copy); in a fresh worktree it will not exist.
- Run `python manage.py migrate` to create an empty one.
- Run `jupyter nbconvert --to notebook --execute --inplace reconcile_campaign_runs_demo.ipynb`
  with the working directory set to `docs/notebooks/pre_executed/` -- the setup cell's
  `parents[2]` repo-root resolution depends on it, as its own inline note says.
- Afterwards, remove the scratch `src/fomo_db.sqlite3` the notebook wrote to and move the saved
  copy back.

Sanity-check the regenerated output before moving on: the classical run's three events must show
`telescope='RDGS'  instrument='EFOSC2'`, the queue and class-wide events must show the whole
string under `telescope` with `instrument=''`, all three titles must still carry the full
combined string, and the second sweep must still report `created: 0, updated: 0`.

**Runbook** (`docs/runbooks/telescope_runs_calendar.rst`): in the "How do I get every campaign
run onto the calendar?" section, immediately after the "What an operator sees on the calendar
afterwards, in plain terms" paragraph that ends at line 553, add a short paragraph in the page's
existing operator-facing voice covering three things:

- The run's free-text ``Telescope / Instrument`` value is split on the first ``/`` or ``+`` into
  the calendar entry's separate **Telescope** and **Instrument** fields in the event pop-up; a
  value with no delimiter goes wholly into Telescope. The entry's title still shows the full
  combined text either way.
- Entries for queue-scheduled, class-wide and satellite runs pick this up automatically on the
  next sweep, because that whole-window entry is rewritten from the run every time.
- Per-night entries for classically-scheduled runs created before this change keep their old
  combined value, because a per-night entry's Telescope/Instrument and its sunset/sunrise window
  are deliberately never rewritten after it is first created -- that is what protects a night
  adopted from ``load_telescope_runs`` from having its own more precise values overwritten.

Do not restructure the section or renumber anything else on the page.
  </action>
  <verify>
    <automated>python -c "import json; nb=json.load(open('docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb')); out=''.join(''.join(o.get('text','')) for c in nb['cells'] for o in c.get('outputs',[])); assert \"telescope='RDGS'\" in out and \"instrument='EFOSC2'\" in out, 'split not visible in executed output'; assert \"instrument=''\" in out, 'no-delimiter fallback not visible in executed output'; assert 'created: 0, updated: 0' in out, 'idempotency output lost'; print('notebook output OK')" && grep -c "Instrument" docs/runbooks/telescope_runs_calendar.rst && python -m sphinx -b html docs docs/_build/html -q</automated>
  </verify>
  <done>The demo notebook demonstrates the split (and the no-delimiter fallback) with real committed output, the runbook documents what the pop-up's two fields now show and which existing entries do and do not self-heal, and the Sphinx build is clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| public submitter -> `CampaignRun.telescope_instrument` | Untrusted free text enters at the public submission form and is stored verbatim |
| `CampaignRun` -> `CalendarEvent.telescope` / `.instrument` | This change routes that same untrusted text into two stored fields instead of one |
| `CalendarEvent` -> event-detail pop-up | Both fields are rendered back to every calendar visitor, including anonymous ones |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-SGF-01 | Information disclosure | `event_form.html` Telescope/Instrument fields | accept | No new data is exposed and no new rendering path is added: `instrument` is already rendered by the same `{% bootstrap_field %}` block (`event_form.html:79-81`) for `load_telescope_runs` events, and the combined text is already shown today in the Telescope field and the event title. Django template autoescaping is unchanged. |
| T-SGF-02 | Tampering | `CalendarEvent.telescope` / `.instrument` `max_length=200` | accept | Both halves of a split are strictly shorter than the source string that already fits `telescope`, so the change can only reduce stored length. No new truncation or overflow path. |
| T-SGF-03 | Tampering | `_reconcile_classical_nights()` update path | mitigate | Task 1 explicitly leaves the `else: fields = common_fields` arm untouched, so an adopted `load_telescope_runs` event's own `telescope`/`instrument` values cannot be overwritten by a run's free text; Task 2 Test 2 covers only the create path, and the existing `TestAdoptAndRekey` tests continue to guard the adopt path. |
| T-SGF-04 | Denial of service | `re.split(r'[/+]', text, maxsplit=1)` | accept | Fixed character-class split with `maxsplit=1` -- linear, no backtracking, no ReDoS surface. |
| T-SGF-SC | Tampering | package installs | n/a | This plan installs no npm/pip/cargo packages; no legitimacy gate applies. |
</threat_model>

<verification>
- `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs solsys_code.tests.test_campaign_approval` reports `OK` with at least 170 tests (measured baseline 2026-08-05: `Ran 161 tests` / `OK`).
- `ruff check .` and `ruff format --check .` both clean.
- `python -m sphinx -b html docs docs/_build/html -q` completes without errors.
- `grep -n "'telescope': run.telescope_instrument" solsys_code/campaign_reconciler.py` returns nothing -- both write sites converted.
- `grep -c "def event_title" solsys_code/campaign_reconciler.py` is 1 and `event_title()`'s body still builds `f'{run.campaign.name}: {run.telescope_instrument}'` -- the title is unchanged.
</verification>

<success_criteria>
- A run with `telescope_instrument='Apache Point Observatory/ARCTIC'` reconciles to a calendar event with `telescope='Apache Point Observatory'` and `instrument='ARCTIC'` -- the exact live dev-DB case in the bug report (RUN:10).
- Both the container branch and the classical create path split; `+` works the same as `/`; a delimiter-free value still lands wholly in `telescope`.
- The event title still carries the full combined string; `_adopted_event_for_night()`, `_skip_reason()` and all `source`-based dispatch are byte-unchanged.
- Every pre-existing test still passes, with exactly one deliberately revised assertion (`test_campaign_approval.py:380`) that now proves the split end-to-end through the staff-approval path.
- The paired demo notebook and the operator runbook are updated in the same change, not as a follow-up.
</success_criteria>

<output>
Create `.planning/quick/260805-sgf-split-campaignrun-telescope-instrument-i/260805-sgf-SUMMARY.md` when done
</output>
