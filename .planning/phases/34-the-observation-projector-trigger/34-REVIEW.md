---
phase: 34-the-observation-projector-trigger
reviewed: 2026-09-11T00:00:00Z
depth: deep
iteration: 5
files_reviewed: 7
files_reviewed_list:
  - docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb
  - solsys_code/calendar_utils.py
  - solsys_code/observation_projector.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_observation_projector.py
  - solsys_code/tests/test_observation_projector_signals.py
  - solsys_code/tests/test_projector_demo_notebook.py
findings:
  critical: 1
  warning: 11
  info: 4
  total: 16
status: issues_found
---

# Phase 34: Code Review Report (incremental re-review, iteration 5)

**Reviewed:** 2026-09-11
**Depth:** deep
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Incremental review of everything since `50e0959` (the commit carrying the previous
`34-REVIEW.md`): the CR-01/CR-02/CR-03/WR-01/WR-03/WR-04/WR-05/WR-06..WR-10 fix commits
(`1d6ef6c`..`46d8390`) and the plan 34-07 gap-closure commits (`37ffe2b`, `8757750`,
`667ba21`, `4501677`, `025d741`). `WR-02` (duplicate datetime parser) and `WR-11`
(hand-rolled monkeypatching in the signal tests) were explicitly skipped by the fix pass
and are not re-litigated here.

**Verification performed.** `pre-commit run ruff --files` and `pre-commit run ruff-format
--files` are clean on all six changed Python files. `python manage.py test
solsys_code.tests.test_calendar_utils solsys_code.tests.test_projector_demo_notebook`
(60 tests, 1 skipped) and `... test_observation_projector
test_observation_projector_signals` (72 tests) are green. Two behavioural claims below
were reproduced against the real code with a throwaway script (no database writes, no
source file modified): `record_time_window()` returning `(None, None)`, and a date-only
request window now raising. The notebook was parsed with `nbformat`-level JSON and every
guarded cell's *output* was compared against its *source* and against the committed
`project_observation_calendar_demo.sched06-baseline.json`.

**What is genuinely closed.** CR-01 (`astimezone()` conversion) is real and the new
`event_fields_for()` regression test pins the `Window (UTC):` wall clock. CR-02's routing
of `parameters['start']/['end']` through the shared coercion does fix the `Z`-suffix
rejection. WR-08's scratch-database guard now compares `Path.resolve()`d paths and raises
instead of asserting, closing a genuine "relative override silently opens the developer
database" hole. WR-10's in-memory-instance test no longer INSERTs. CR-03/WR-06/WR-07 leave
the committed SCHED-06 baseline intact on a scratch-routed run, and the new
`test_projector_demo_notebook.py` does fail for the right reason when cell `05528b38`'s
outputs are emptied.

**What this pass found.** One BLOCKER: the CR-02 fix made `record_time_window()` able to
return `(None, None)`, contradicting its own annotation, its `Raises:` section and
`coerce_schedule_datetime()`'s explicit "never returns `None` for an unusable value"
promise — and the value lands in a web-request path whose handler only catches
`(KeyError, ValueError)`. Beyond that: the WR-03 date-rejection was applied to a field
(the request window) whose semantics the WR-03 rationale never covered, and it creates the
exact receiver-vs-sweep divergence `coerce_schedule_datetime()` was written to prevent;
the WR-01 fix now logs caught exception *values* in two bare `except Exception` handlers,
contradicting this module's own SYNC-09/D-11 rule; and the new notebook-evidence guard has
three holes big enough to pass the failure it exists to catch, plus one false-failure mode
that fires on exactly the re-execution the notebook instructs the operator to perform. Two
of the notebook's printed statements contradict the same notebook's own evidence.

## Narrative Findings (AI reviewer)

### Critical Issues

#### CR-01: `record_time_window()` now returns `(None, None)` for a JSON-null request window, breaking its annotation, its docstring and the caller that only catches `(KeyError, ValueError)`

**Severity:** BLOCKER
**File:** `solsys_code/calendar_utils.py:548-549` (behaviour), `:515,531-540` (contract),
`:486-498` (`coerce_schedule_datetime()`'s own promise); consumed at
`solsys_code/campaign_attribution.py:680-683`, reached from
`solsys_code/campaign_views.py:1151`

**Issue:** The CR-02 fix replaced

```python
start_time = datetime.fromisoformat(record.parameters['start']).replace(tzinfo=dt_timezone.utc)
```

with `coerce_schedule_datetime(record.parameters['start'])`. That helper's *first* line is
`if value is None: return None`. So when a record carries `parameters = {'start': None,
'end': None}` (a JSON null, which `JSONField` stores and returns verbatim),
`record_time_window()` now returns `(None, None)` instead of raising. Reproduced against
the real code:

```
null start/end -> (None, None)
```

Three shipped statements are false as a result:

1. The return annotation `-> tuple[datetime, datetime]` (`:515`). The sibling branch was
   given `cast(datetime, ...)` by WR-05 *specifically* so the annotation "stays honest
   about None never being reachable" (`:551-553`) — while the branch immediately above it
   was, in the same commit, made able to return `None`.
2. `record_time_window()`'s `Raises:` section (`:534-540`), which promises `ValueError`
   "if `parameters['start']/['end']` are not valid ISO datetime strings". `None` is not a
   valid ISO datetime string and no `ValueError` is raised.
3. `coerce_schedule_datetime()`'s own docstring (`:490-498`): "This function never returns
   `None` for an unusable value … silently degrading it to `None` here would draw a
   queued-looking event over the wrong window." That reasoning is exactly why the *caller*
   must not hand it a value whose `None` means "absent request window" rather than
   "absent schedule field" — the two `None`s mean different things and this change
   conflates them.

The consequence is not contained to the projector. `campaign_attribution._record_window()`
(`:665-683`) documents itself "Never raises", catches `(KeyError, ValueError)` — and then
calls `start.date()`. With `(None, None)` that is an `AttributeError`, which escapes the
helper, escapes `candidates_for_record()`, and 500s the attribution worklist view at
`campaign_views.py:1151`. (Honest caveat: pre-change this same input raised `TypeError`
from `datetime.fromisoformat(None)`, which that handler also did not catch — so the 500 is
not new. What *is* new is that the failure is now silent inside `record_time_window()`,
returns a wrong-typed value, and is invisible to both the annotation and every
`except ValueError` written against this function. The one-line fix below closes both.)

**Fix:** make the request-window branch as strict as the schedule branch it now shares a
parser with — a missing window is exactly as unusable as a malformed one:

```python
    if record.scheduled_start is None and record.scheduled_end is None:
        start_time = coerce_schedule_datetime(record.parameters['start'])
        end_time = coerce_schedule_datetime(record.parameters['end'])
        if start_time is None or end_time is None:
            raise ValueError(
                f'Request window is null: start={record.parameters["start"]!r}, '
                f'end={record.parameters["end"]!r}'
            )
```

and add a test alongside `test_both_scheduled_none_falls_back_to_z_suffixed_parameters_start_end`
asserting `ValueError` for `parameters={'start': None, 'end': None}`. Keeping the `cast()`
on the other branch is then consistent rather than selective.

### Warnings

#### WR-01: the WR-03 "a schedule field is a block boundary, not a day" rule was applied to the request window as well, where a day boundary is legitimate

**Severity:** WARNING
**File:** `solsys_code/calendar_utils.py:505-506` (the guard), reached for request windows
via `:548-549` and `solsys_code/observation_projector.py:280-281`

**Issue:** WR-03's rationale, quoted verbatim in the code and the docstring, is about
`scheduled_start`/`scheduled_end`: "a schedule field is a block boundary, not a day". CR-02
then routed `parameters['start']/['end']` — the *request window*, which is a user-chosen
span that legitimately may be expressed day-granularly — through the same function. The
two changes landed in the same increment and neither mentions the other. Reproduced:

```
date-only request window -> ValueError Schedule value is a date, not a datetime: '2026-09-18'
```

Before this increment such a record projected fine (`datetime.fromisoformat('2026-09-18')`
succeeds and yields midnight); now it is `unprojectable`, its event is never created or is
frozen at its last good state, and the only trace is a warning line and a counter. Today's
two known producers happen to be safe (`OCSBaseObservationForm.clean_start()` returns
`SplitDateTimeField.isoformat()`, and `backfill_lco_observations._build_parameters()`
copies portal window strings), so this is a latent regression rather than an observed one —
but it applies to every hand-created, imported or fixture-loaded record too, and nothing
enforces the assumption.

Secondary: `docs/runbooks/telescope_runs_calendar.rst:173-177` enumerates what makes a
record `unprojectable` ("for example, an unparsable request window"). A bare ISO date is
*parsable* by every field in the stack — Django's `DateTimeField` accepts it, the ORM
stores it, `parse_datetime()` returns a datetime for it — and is rejected only by this new
guard. Per CLAUDE.md's paired-docs rule (any `docs/runbooks/` page whose documented
behaviour the change affects), that new rejection class should have been named in the
runbook in the same increment.

**Fix:** either scope the guard to the schedule fields it was written for (pass a flag, or
keep a thin `coerce_request_window_datetime()` that accepts midnight), or accept the
widening deliberately — and then say so in `coerce_schedule_datetime()`'s docstring, in
`record_time_window()`'s `Raises:` section, and in the runbook's `unprojectable` paragraph.

#### WR-02: `len(value) <= 10` is a fragile proxy for "this string has no time component"

**Severity:** WARNING
**File:** `solsys_code/calendar_utils.py:501-507`

**Issue:** The guard infers "date-only" from string length, after the value has already
been parsed, and is therefore coupled to two implementation details it does not name:

- Django ≥ 4.1's `parse_datetime()` tries `datetime.fromisoformat()` *first* (installed
  here: Django 5.2.17). On Django < 4.1 the regex-only implementation rejects a bare date
  outright, so the branch is unreachable and the new
  `test_bare_iso_date_string_raises_value_error` would fail on its message assertion. CI
  tests Python 3.10-3.12 against whatever Django `tomtoolkit` resolves to.
- Python's accepted date-only spellings all happen to be ≤ 10 characters *today*
  (`'2026-09-18'`, `'20260918'`, `'2026-W38-5'` — all verified to parse to midnight, all
  ≤ 10). Nothing guarantees that stays true; a future accepted spelling longer than 10
  characters silently reverts to the pre-WR-03 "midnight" behaviour the guard exists to
  prevent, with no test failing.

**Fix:** test the property directly instead of a proxy for it — e.g. reject when the raw
string contains neither a `T` nor a space separator followed by a digit:

```python
        parsed = parse_datetime(value)
        if parsed is None:
            raise ValueError(f'Unparseable schedule datetime string: {value!r}')
        if not re.search(r'[T ]\d', value):  # no time component -- a day, not a block boundary
            raise ValueError(f'Schedule value is a date, not a datetime: {value!r}')
```

and add `'20260918'` / `'2026-W38-5'` to `test_bare_iso_date_string_raises_value_error`.

#### WR-03: the bare-date rejection creates the exact receiver-vs-sweep divergence `coerce_schedule_datetime()` was written to prevent

**Severity:** WARNING
**File:** `solsys_code/calendar_utils.py:460-512` (docstring premise at `:466-476`);
demonstrated by `solsys_code/tests/test_observation_projector_signals.py:170-199`

**Issue:** `coerce_schedule_datetime()`'s stated reason to exist is: "The sweep, by
contrast, re-fetches the record from the database and sees a real `datetime` — both paths
must produce the same window, or the receiver and the sweep would write different spans to
the same event forever."

For a bare-date schedule value the two paths now produce *different* outcomes by
construction:

- Receiver (in-memory string): `ValueError` → `unprojectable` → the event is left at its
  previous (queued) span. This is what the new test asserts.
- Sweep (DB-fetched): Django's `DateTimeField` already coerced `'2026-09-16'` to
  `2026-09-16T00:00:00Z` on save, so the sweep sees a real `datetime`, classifies the
  record `placed`, and draws the midnight-to-05:19 block the receiver refused.

So the record the new test creates is not "left alone"; it is left as a time bomb whose
next sweep writes precisely the window WR-03 argues is wrong. The test asserts only the
receiver half and stops. Whatever the resolution of WR-01/WR-02 above, both paths must
agree.

**Fix:** decide once. If a bare date is unusable, the sweep must reject it too — which
means the record's *stored* value has to be recognisable as day-granular (it no longer is
after the DB round-trip), so the realistic option is to accept midnight and log, rather
than reject. If it stays rejected, extend the new signals test to also assert what the
next `project_observation_calendar` sweep does with the same record, so the divergence is
at least pinned rather than undiscovered.

#### WR-04: the WR-01 fix logs the caught exception's value in two bare `except Exception` handlers, contradicting this code's own SYNC-09/D-11 rule

**Severity:** WARNING
**File:** `solsys_code/observation_projector.py:370-371`, `:487-490`; rule stated at
`solsys_code/calendar_utils.py:310-315`

**Issue:** `resolve_placement_block()`'s docstring states the convention in absolute terms:
"The except clause never references, stringifies, or logs the caught exception
(SYNC-09/D-11) — `ImproperCredentialsException`/`forms.ValidationError` embed
`response.content` directly and must never be logged verbatim." The previous review pass
verified this held across the whole phase ("No exception *value* is logged anywhere in the
new code"). The WR-01 fix changed both projector handlers from `'%s', type(exc).__name__`
to `'%s: %s', type(exc).__name__, exc`, and neither the code, the commit, nor
`34-REVIEW-FIX.md` reconciles that with the rule.

Both handlers are bare `except Exception` over a whole call graph, so what they stringify
is not bounded by what today's code raises: `project_record()`'s `try` covers
`facility_for()` (which instantiates `LCOFacility`/`SOARFacility` — the classes that own
the portal credentials), `insert_or_create_calendar_event()` (database errors, whose
messages embed row values) and `write_event_meta()`. Today nothing on those paths raises a
credential-bearing exception, so this is a convention breach and a latent leak into the
production log rather than an observed one — but the value of a blanket "never stringify"
rule is precisely that it does not require this analysis to be redone on every future
change.

**Fix:** log the diagnostic without stringifying an arbitrary exception — either restrict
the message to the exception types this module raises deliberately:

```python
    except Exception as exc:  # noqa: BLE001
        detail = str(exc) if isinstance(exc, (ValueError, KeyError, InstrumentExtractionError)) else ''
        logger.warning('unprojectable observation_id=%r: %s %s', record.observation_id, type(exc).__name__, detail)
```

or amend the SYNC-09/D-11 note at `calendar_utils.py:310-315` to say the rule is scoped to
the portal-call handler and explain why the projector's handlers are exempt. Either is
acceptable; silently having both statements in the tree is not.

#### WR-05: the new notebook guard's "converges to zero" check omits `unprojectable`, `failed` and `site_lookup_failed` — the committed run already hides a non-zero counter behind it

**Severity:** WARNING
**File:** `solsys_code/tests/test_projector_demo_notebook.py:135-147`; mirrored in the
notebook at cell `556d2a9f`

**Issue:** The test's own docstring says "a second sweep over an already-projected database
must always report no further work", but it only asserts `created: 0`, `updated: 0`,
`site_lookups: 0`, and it drops the first `' | '` segment — the one carrying `failed: N`.
Two consequences, one already realised:

1. The committed second-sweep line is
   `LCO: created: 0, updated: 0, unchanged: 159, unprojectable: 0, site_lookups: 0,
   site_lookup_failed: 1`. `site_lookup_failed: 1` means the sweep made a live portal call
   for `observation_id='4276100'` and failed again — i.e. the corpus is *not* fully
   converged, and every future sweep (including Phase 36's cron) repeats that call. The
   notebook's prose calls this state "convergence, not silence"; the guard now codifies
   that reading permanently.
2. A regression in which the second sweep reports
   `created: 0, updated: 0, unchanged: 0, unprojectable: 159, site_lookups: 0` — total
   failure — satisfies all three asserted tokens and passes the guard unchanged. An
   evidence guard that passes on total failure is not a guard.

**Fix:** assert the full counter set, and assert the one that actually proves work
happened:

```python
        for token in ('created: 0', 'updated: 0', 'site_lookups: 0', 'unprojectable: 0'):
            self.assertIn(token, segment, ...)
        self.assertRegex(second_line, r'Done\. failed: 0')
```

and either assert `site_lookup_failed: 0` too, or state in the notebook prose why a
permanently-retried lookup is acceptable and add its `observation_id` to the closing
requirement table so it is tracked rather than normalised.

#### WR-06: the guard fails a legitimate un-routed run that finds real takeover work — the exact re-execution the notebook instructs the operator to perform

**Severity:** WARNING
**File:** `solsys_code/tests/test_projector_demo_notebook.py:118-133`; notebook cell
`556d2a9f` (the `elif first_sweep_work == 0:` branch)

**Issue:** `test_unrouted_run_states_the_run_was_already_converged` skips only when the run
*was* scratch-routed. For every un-routed run it unconditionally requires the string
"found nothing to take over" in cell `556d2a9f`'s output. But the notebook only prints that
sentence when `first_sweep_work == 0`; an un-routed run whose first sweep does real work
prints nothing of the sort and the test fails — reporting "the notebook does not state that
this un-routed run found nothing to take over" for a notebook whose evidence is *better*
than the converged case. `docs/.../project_observation_calendar_demo.ipynb`'s own "What
happens next" section instructs exactly that run ("re-execute this notebook end to end …
and commit it again"), so the first operator to follow the documented procedure with any
residual sweep work breaks the test suite.

**Fix:** make the un-routed assertion conditional on the same signal the notebook branches
on, rather than on routing alone:

```python
        sweep_text = self.cell_text[_CELL_SECOND_SWEEP]
        work = re.search(r'^First sweep work \(created \+ updated, every facility\): (\d+)$', sweep_text, re.M)
        self.assertIsNotNone(work, 'Second-sweep cell does not report the first sweep work total.')
        if int(work.group(1)) > 0:
            self.skipTest('This un-routed run did real takeover work -- the converged statement does not apply.')
        self.assertIn('found nothing to take over', sweep_text, ...)
```

#### WR-07: the corpus-reconciliation cell claims "the developer database" on a run the same notebook proves ran against a scratch copy

**Severity:** WARNING
**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`, cell
`5b5a036e` (final `else:` branch) — printed output: "(none -- every LCO/SOAR record in the
developer database has its own facility-url-keyed event)"

**Issue:** Cell `7022f987` of the same run prints "Resolved database:
'/home/tlister/git/fomo_devel/tmp/34-07-fresh-clone.sqlite3' -- routed to a scratch copy,
not the developer database", and the entire WR-08/CR-03 fix set exists to stop scratch-copy
state being presented as developer-database state. This cell's string is the one place that
still does it, and it is the sentence backing ROADMAP criterion 1 ("the whole real corpus,
not a sample") in the closing evidence table. A reader auditing the notebook top-to-bottom
gets two contradictory answers to "which database is this?".

**Fix:** derive the noun from the same routing signal every other cell now re-reads:

```python
db_label = 'the scratch copy' if os.environ.get('FOMO_DATABASE_PATH') else 'the developer database'
...
    print(f'(none -- every LCO/SOAR record in {db_label} has its own facility-url-keyed event)')
```

and re-execute. (Cell `12`'s heading and the closing table's PROJ-01 row should get the same
treatment.)

#### WR-08: the SCHED-06 cell blames the sweep for a record-count gap the sweep cannot cause, leaving the committed baseline's validity unexplained

**Severity:** WARNING
**File:** `docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`, cell
`250b5d0b` — printed output: "(the scratch copy currently holds 52 pending records, already
mutated by the sweep cells above; they are deliberately not shown …)"

**Issue:** The committed baseline (`project_observation_calendar_demo.sched06-baseline.json`,
`captured_at='2026-09-11T04:44:59.526430+00:00'`) has `record_count: 74` (56 queued + 18
placed) — the same tally the cell prints. The scratch clone, taken "fresh immediately
before it runs" from the developer database on the same day, yields 52. The printed
explanation is that the sweep mutated them, which is not possible: `project_queryset()`
writes `CalendarEvent`/`CalendarEventMeta` rows and, via `resolve_observed_site()`, only
`parameters` on records already at a *successful-terminal* stage
(`project_observation_calendar.py:80-103`). It never changes `status`, so it cannot change
how many records match `status='PENDING'`. The real cause — 22 records left `PENDING`
between the baseline capture and this clone — is the very thing SCHED-06 is watching, and
the notebook neither reports it nor reconciles it with "the committed baseline is the
evidence UAT Test 4 diffs against".

**Fix:** replace the misattribution with the actual comparison, so the reader can see
whether the baseline is still the right thing to diff against:

```python
    print(f'(the scratch copy currently holds {len(scratch_records)} pending records vs. '
          f'{existing_baseline["record_count"]} at baseline capture -- '
          f'{existing_baseline["record_count"] - len(scratch_records)} have left PENDING since; '
          'the copy\'s rows are deliberately not shown -- see the framing above.)')
```

and say in the "What happens next" prose whether a corpus that has moved on invalidates the
committed baseline or is exactly the SCHED-06 signal being waited for.

#### WR-09: the guard ignores error outputs and execution counts, so a notebook committed with failing in-cell asserts still passes

**Severity:** WARNING
**File:** `solsys_code/tests/test_projector_demo_notebook.py:41-59`

**Issue:** `_cell_output_text_by_id()` concatenates only top-level `text` keys, which
exist on `stream` outputs. An `error` output (type `error`, carrying `ename`/`evalue`/
`traceback`) contributes the empty string and is otherwise invisible. The notebook's own
evidence rests on in-cell `assert` statements (cells `05528b38`, `556d2a9f`) — but
`jupyter nbconvert --execute --allow-errors` commits a notebook whose asserts *failed*,
with every `print()` before the assert still present in the stream output. Such a notebook
passes every test in this module. The module docstring's claim that it "makes the paired
demo notebook's takeover demonstration checkable on every test run" does not hold for the
one failure mode the notebook's own asserts are meant to produce.

**Fix:** add a whole-notebook check that no cell carries an error output, and that the
guarded cells were actually executed:

```python
    def test_no_cell_recorded_an_error_output(self):
        with open(NOTEBOOK_PATH) as fh:
            notebook = json.load(fh)
        errored = [
            cell.get('id')
            for cell in notebook['cells']
            if any(output.get('output_type') == 'error' for output in cell.get('outputs', []))
        ]
        self.assertFalse(errored, f'Notebook cells recorded error outputs: {errored}')
```

#### WR-10: the guard reads three files with the locale default encoding

**Severity:** WARNING
**File:** `solsys_code/tests/test_projector_demo_notebook.py:52`, `:75`, `:164`

**Issue:** `open(notebook_path)`, `open(BASELINE_PATH)` and `NOTEBOOK_PATH.read_text()` all
use `locale.getpreferredencoding()`. `.ipynb` and `.json` are UTF-8 by specification. The
committed notebook happens to be pure ASCII today, so this is latent — but the notebook's
output embeds real target names and portal strings, and a single non-ASCII character (a
degree sign, a `µ`, an accented observer name) in a future re-execution turns every test in
this module into a `UnicodeDecodeError` under a `C`/`POSIX` locale, which is a plausible CI
or container default.

**Fix:** `open(path, encoding='utf-8')` in both places and
`NOTEBOOK_PATH.read_text(encoding='utf-8')`.

#### WR-11: two guard assertions raise `StopIteration` instead of failing, and cells without an `id` collapse onto one dict key

**Severity:** WARNING
**File:** `solsys_code/tests/test_projector_demo_notebook.py:110-111`, `:142`, `:58`

**Issue:**

- `next(line for line in sweep_text.splitlines() if line.startswith('First sweep'))` raises
  `StopIteration` when the notebook's summary prefix changes (a rename, a reflow, a
  `print()` edit). The test then *errors* with a bare `StopIteration` and no message
  naming the cell or the expected prefix — the opposite of the diagnostic style every other
  assertion in this file was written with. Both `next(...)` call sites, and both
  `first_line`/`second_line` uses, share this.
- `text_by_id[cell.get('id')] = ...` uses `None` as the key for any cell lacking an `id`
  (legal in nbformat 4.4 and earlier). Several such cells silently overwrite each other.
  Harmless today (`nbformat_minor` is 5 and all ids are unique and present) but it defeats
  the module's own "addressed by nbformat id, never by position" guarantee for a
  downgraded file.

**Fix:** extract a helper that fails cleanly, and skip id-less cells explicitly:

```python
def _line_starting_with(text, prefix, cell_id):
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f'Cell {cell_id} has no {prefix!r} line; output was: {text!r}')
```

### Info

#### IN-01: the guard never runs in CI or pre-commit

**File:** `solsys_code/tests/test_projector_demo_notebook.py:1-17`

Per CLAUDE.md, `pyproject.toml`'s `testpaths = ["tests", "src", "docs"]` means
`python -m pytest` (what pre-commit and `.github/workflows/` run) does not collect
`solsys_code/tests/`. The module docstring's "checkable on every test run" is true only for
`python manage.py test`. Worth a sentence in the docstring, or a pytest-collected shim, so
nobody assumes the committed notebook is gated by CI.

#### IN-02: the credential scan is a keyword heuristic that cannot see an actual secret

**File:** `solsys_code/tests/test_projector_demo_notebook.py:162-166`

`re.findall(r'api_key|Authorization|token=', raw_text, re.I)` catches the *words*, not the
values. An LCO API key pasted or printed as a bare 40-character string passes; conversely a
legitimate prose mention of "Authorization" in a markdown cell fails the build. A value-
shaped check (e.g. a long high-entropy hex/base64 run outside a URL) would match the test's
stated intent better; at minimum the docstring should say what it does *not* detect.

#### IN-03: dead capture group, and the baseline path is coupled to the notebook's directory

**File:** `solsys_code/tests/test_projector_demo_notebook.py:30`, `:99`

`re.search(r'^([1-9][0-9]*) of ([0-9]+) pre-existing', ...)`'s second group is never read —
the total is not compared to anything, so a run that re-titles 1 of 10000 events passes the
same as 33 of 159. `BASELINE_PATH = NOTEBOOK_PATH.parent / ...` silently requires anyone
using `FOMO_DEMO_NOTEBOOK_PATH` to copy the baseline JSON alongside the notebook; a missing
file surfaces as a bare `FileNotFoundError` from `setUpClass`, unlike the neighbouring
missing-cell check which raises a diagnostic `AssertionError`.

#### IN-04: review-finding IDs continue to accumulate in shipped source

**File:** `solsys_code/calendar_utils.py:505,543,551`;
`solsys_code/observation_projector.py` (passim); every new test docstring in this increment

This increment added `WR-03`, `WR-04`, `WR-05`, `WR-07`, `WR-08`, `WR-09`, `WR-10`,
`CR-01`, `CR-02`, `CR-03` and `G-34-2`/`G-34-3` markers to comments, docstrings and test
names. The prior review raised this as IN-04 and it has grown. These IDs are scoped to one
phase's review cycle and are meaningless to a reader six months from now — the *reasons*
they encode are valuable and should stay; the identifiers should not.

---

_Reviewed: 2026-09-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
