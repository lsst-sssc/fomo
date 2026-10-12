# Override draft for /gsd-verify-work 40 (quick task 261012-g4y)

Draft only. It rests on the 8ab5247 guard and the 819321e helper as restored on 2026-10-12, and it replaces the
override recorded on 2026-10-11 (accepted_at 2026-10-11T13:37:50Z), which described the rolled-back R12.

## Proposed override 1 -- WARN-02 guard completeness

must_have: "ROADMAP SC 2 (first half) / WARN-02: The notebook guard test fails for a notebook whose current source routes Django to the developer database even when that notebook's committed output is stale"

reason: >-
  The guard reads each notebook's JSON (its literal code and its committed output) and never runs it. In ordinary
  code it catches a notebook that skips the helper or calls it twice (R2), imports Django before the helper call
  (R3), sets Django up itself with `django.setup()`, `from django import setup` or `import django as x` (R4), names
  `FOMO_DATABASE_PATH` or `fomo_db.sqlite3` in a string or passes a `FOMO_DATABASE_PATH=` keyword (R5/R7), or calls
  `shutil.copy`, `copy2` or `copyfile`, also under an import alias (R6). R0, R5, R6 and R7 apply to every notebook;
  the other rules apply only to a notebook that imports Django, the project app or a TOM plugin. It also checks the
  committed output: counts 1..N and no error (R8), a Resolved line inside a `fomo-notebook-db-*` directory (R9), the
  matching teardown at the end (R10), and no `assert` around a write (R11).

  Deliberate evasion is out of scope (developer decision 2026-10-12). The guard does not see code inside a string
  (`exec`, `eval`, a `timeit` string, including the setup call's own argument), IPython shell lines, magics and
  aliases, monkey-patching (for example replacing `tempfile.mkdtemp`), or names, paths and values built at run time.
  It also misses a helper call made through its module attribute, re-pointing Django's settings or connections
  after the helper call without writing `FOMO_DATABASE_PATH` or `fomo_db.sqlite3`, and a byte copy by anything other
  than shutil's `copy`, `copy2` and `copyfile` when the file name is not written out.

  Another process is not seen either (a `!` shell line, `%run`, `subprocess`, `os.system`). Unlike the rest,
  ordinary code can do this: a `!python manage.py ...` cell before the helper call or after the teardown runs with
  `FOMO_DATABASE_PATH` unset, so `settings.py` falls back to `src/fomo_db.sqlite3`.
  [Developer: accept or ask for a fix]

  The run-time backstop is the 819321e helper. It refuses to start if Django is already set up or configured in the
  kernel. Before `django.setup()` it raises `RuntimeError` unless `default` is the only alias, `DATABASE_ROUTERS` is
  empty and `default` resolves to the scratch file it created. After a later failure it closes the connections,
  switches every alias to the dummy backend, removes the scratch directory, restores the environment and re-raises.
  If closing the connections itself fails, the switch is skipped and Django's own error replaces the helper's. A
  `DATABASES` setting with no `default` alias does this (round-2 WR-02, left unfixed by the developer's decision); no
  database is reachable then, because Django refuses every connection without a `default` alias. The teardown's
  second line is fixed text, not a measurement (round-2 WR-01). The helper checks once, at setup, so code that runs
  before the call, after the teardown or in another process is outside its checks.

accepted_by: [developer to fill in]
accepted_at: [developer to fill in]

## Proposed override 2 -- guard wording

must_have: "40-08 must_have 8.14, 40-11 must_have 11.1 and 40-18 must_have 6 (every form the texts claim is reported is reported): the guard docstring and README state what R0-R12 check"

reason: >-
  The texts those must-haves judged (the 40-08..40-18 guard docstrings, the README paragraph and the R12 they
  describe) were rolled back with the guard. The texts in force are the 8ab5247 module docstring and the README
  (819321e plus the two IN-01 sentence corrections). Read against the code, they claim more than is checked in four
  places: (1) the docstring says a comment that merely mentions a name cannot satisfy or trip a check, but a cell
  that does not parse is searched as text, comments included (R0; this only ever reports more; round-7 WR-03);
  (2) the README says `/tmp/fomo-notebook-db-*`, but the helper uses `tempfile`, which follows `TMPDIR` (the README's
  own command sets `TMPDIR=/tmp`; round-7 WR-04); (3) the README and docstring say a notebook never sets Django up,
  routes the database itself or byte-copies, which means the forms in override 1 and no others; (4) the README and
  `setup_scratch_django`'s docstring say a refusal or failure switches every alias to the dummy backend, which does
  not hold when closing the connections fails (round-2 WR-02, accepted by dropping c45edbc).

accepted_by: [developer to fill in]
accepted_at: [developer to fill in]

## What each claim rests on

| Claim | Rests on |
|-------|----------|
| Skipped or doubled helper call | `isolation_problems` R2, test_pre_executed_notebooks.py:363-393 |
| Django imported before the call | R3, :394-399 |
| Own `django.setup()` | `_django_setup_problems` (R4), :264-278 |
| Routing by string or keyword | `_database_routing_problems` (R5/R7), :281-295 |
| Byte copy | `_byte_copy_problems` (R6), :298-316 |
| Applies to every notebook / Django-backed only | `isolation_problems` :344-349, `is_django_backed` :190 |
| Output checks | R8 :357-361, R9 :401-412, R10 :414-433, R11 :435 |
| Comments searched as text | R0, :334-342 (`UNPARSEABLE_SUSPECT_TEXT` :83) |
| Refuses a running Django | `setup_scratch_django`, fomo_nb_utils.py:275 |
| Settings checked before `django.setup()` | :295, `check_database_settings` :137, `check_resolved_database` :115 |
| Switch to dummy backend after failure | `except BaseException` :306, `_disable_django_databases` :184 |
| Close can fail first (WR-02) | `connections.close_all()` :199, before the alias switch |
| Teardown line is fixed text | `teardown_scratch_django`, fomo_nb_utils.py:364 |
| Another process falls back to the developer file | `src/fomo/settings.py:100` |
| Wording gaps | guard docstring :19; README.md:59, :69, :90; fomo_nb_utils.py:265 |

## Must-haves the rollback withdraws

The guard, pyproject and helper work of these plans is undone: 40-06 (R12 first form), 40-08 (R12 whole-capability
bans and c45edbc's helper change), 40-10 (R12 at every identifier, R0 allow-list), 40-11 (7af64d0 docstrings),
40-13 (R0 automagic, R6 copy names, ccfc578 ipykernel dev extra), 40-14, 40-15 (R6 and R12 additions), 40-16, 40-17
(R12 shell forms, R3 and R4 reading strings) and 40-18 (docstring and README wording). Their README and docstring
text goes with them.

What stays: 40-05's helper and its tests; the README's two IN-01 sentences from c45edbc; 40-07's and 40-09's
notebook work; and 40-09/40-11's notebook prose (17168ff), whose eight teardown sentences now match 819321e.

The verifier decides at /gsd-verify-work 40 which of those must-haves count as withdrawn.
