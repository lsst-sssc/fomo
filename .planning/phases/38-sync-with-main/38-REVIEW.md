---
phase: 38-sync-with-main
reviewed: 2026-10-08T02:39:40Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - docs/installation.rst
  - docs/runbooks/telescope_runs_calendar.rst
findings:
  critical: 0
  warning: 2
  info: 6
  total: 8
status: issues_found
---

# Phase 38: Code Review Report

**Reviewed:** 2026-10-08T02:39:40Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

This is an incremental review of gap-closure plan 38-07 (`git diff b07109a HEAD -- docs`). The plan adds a
`.. _local-settings:` section to `docs/installation.rst`, adds `src/fomo/local_settings.py` and
`:ref:`local-settings`` to the `FOMO_BASE_URL` note, and changes one line in step 2 of the runbook's "Setting it up
on a fresh host".

I checked the text against the code that actually runs: `src/fomo/settings.py:414-423` (the import plus its
`exc.name` guard), `.gitignore:61`, `manage.py`, the editable-install `.pth` (which puts `src/` on `sys.path`),
`origin/main` and every other remote branch's `settings.py`, and PR #58 (`origin/production-deploy`: its
`settings.py`, `deploy/gunicorn.conf.py` and `deploy/install_service.sh`).

The rst is structurally sound. I parsed `installation.rst` with docutils (`:ref:` stubbed out) and got zero system
messages:
- The section underline is 46 characters, matching its 46-character title.
- Both `.. code-block:: console` blocks and the trailing paragraph sit inside the `.. warning::` node.
- The label is defined once (`installation.rst:79`) and referenced from `installation.rst:150` and
  `telescope_runs_calendar.rst:2048`.
- No added line is over 120 columns.
- The added text contains no credential and no absolute local path.
- The `>>` prompt and `python3 manage.py` conventions are followed.
- Both `mv` commands are correct when run from the repository root.

I checked the `shell -c` check with a scratch module. When the file is missing, the guard in `settings.py`
swallows the error during settings import. The `-c` import then retries, because Python does not cache failed
imports, and raises `ModuleNotFoundError: No module named 'fomo.local_settings'`. So the check works as described
for the plain "file is missing" case.

The real defects are in two factual claims. The closing sentence of the warning is false today. It also calls
PR #58's relative import "the same change", but under `manage.py` that import gives the module a different name,
and that name breaks this branch's WR-32 guard. That matters for exactly this phase, which syncs with `main`.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: "`src/fomo/local_settings.py` is the location on every current FOMO branch" is false, and PR #58 is not merged

**File:** `docs/installation.rst:111-112`
**Issue:** The sentence says, in the present tense, that "`main` makes the same change through pull request #58". It
then concludes that `src/fomo/local_settings.py` "is the location on every current FOMO branch". Both statements
are untrue right now:
- `gh pr view 58` reports the PR as `OPEN` (not merged).
- `git show origin/main:src/fomo/settings.py:366` is still `from local_settings import *`.
- So are the `v1.6` tag and 11 other remote branches (`origin/feature/*`, `origin/experiment/*`,
  `origin/eso-paf-example`, `origin/issue27-geocenter-ephem`, `origin/scout-kafka-bridge-feasibility`).

An operator running a `main`-based host (or a v1.x release) who reads this page, for example on the branch's
GitHub view or a PR docs preview, is told that `src/fomo/` is correct for their checkout too. If they move the
file there, the warning's own text describes the result: the committed `SECRET_KEY`, `DEBUG = True`, the console
email backend and empty API keys, all silently. The sentence is also built to go stale: "pull request #58" is a
bare, unlinked number whose meaning changes once it merges.

**Fix:** Say what is true and limit it to versions:
```rst
   From this release on, FOMO reads the file only from ``src/fomo/local_settings.py``. Releases before it,
   including ``main`` until `pull request #58 <https://github.com/<org>/fomo/pull/58>`_ is merged, still read
   the top-level ``local_settings`` module described above.
```
Or drop the sentence: the rest of the warning already holds without it.

### WR-02: PR #58's `from .local_settings import *` is not "the same change"; under `manage.py` it breaks the WR-32 guard this page relies on

**File:** `docs/installation.rst:88`, `docs/installation.rst:111` (cross-ref `src/fomo/settings.py:415-423`,
`origin/production-deploy:src/fomo/settings.py:373-376`)
**Issue:** Line 88 says `settings.py` "imports it as ``fomo.local_settings``". Line 111 says PR #58's relative
import is "the same change". The file is the same, but the module name is not.
- `manage.py` sets `DJANGO_SETTINGS_MODULE=src.fomo.settings`. Under it, `from .local_settings import *` resolves
  to `src.fomo.local_settings`.
- Only the WSGI entry point (`fomo.settings`) gets `fomo.local_settings`.
- I confirmed this with a scratch package. A missing file raises `ImportError` with `exc.name ==
  'src.fomo.local_settings'` under the `src.fomo.settings` import path, and `'fomo.local_settings'` under
  `fomo.settings`.

This branch's guard re-raises anything except `exc.name == 'fomo.local_settings'`. If the phase-38 merge takes
PR #58's import line and keeps this branch's guard, every checkout without a `local_settings.py` crashes at
settings import for every `manage.py` command (`migrate`, `test`, the cron runner). That breaks the first sentence
of this section ("A development checkout needs no settings file of its own"). If the merge takes PR #58's whole
block (`except ImportError: pass`), it quietly undoes WR-32 instead.

The doc tells whoever resolves that conflict that the two are equivalent. Line 88 is also only accurate while the
absolute spelling survives.

**Fix:** Don't call them the same. Either:
- Drop the PR #58 sentence (see WR-01), or
- Note that the merge must keep the absolute `from fomo.local_settings import *` spelling, because the
  `exc.name` guard in `settings.py` compares against that exact name.

Also record the merge constraint in the phase's sync plan so the `settings.py` conflict is not resolved toward
`.local_settings`.

## Info

### IN-01: The check's "`ModuleNotFoundError` means the file is not where FOMO looks" also catches a missing dependency inside the file

**File:** `docs/installation.rst:104-109`
**Issue:** Suppose the file is in the right place but imports a package that is missing from this venv (for
example, a moved production file that imports `whitenoise`). The `settings.py` guard re-raises it
(`exc.name != 'fomo.local_settings'`). `manage.py shell` then dies before `-c` runs, with `ModuleNotFoundError: No
module named 'whitenoise'`. The text as written sends the operator looking for a misplaced file. The check also
never tells them to confirm that the printed path is `.../src/fomo/local_settings.py`. That matters if a
non-editable install has a stale copy under `site-packages/fomo/`.
**Fix:** "A ``ModuleNotFoundError: No module named 'fomo.local_settings'`` means the file is not where FOMO looks;
any other missing module is an import inside the file itself. The printed path should end in
``src/fomo/local_settings.py``."

### IN-02: "nothing reports it" is overstated; `check_unattended` (and `check --deploy`) do flag the effects

**File:** `docs/installation.rst:94`
**Issue:** `solsys_code/management/commands/check_unattended.py:298` flags the console `EMAIL_BACKEND`, and `:529-536`
flags an unset LCO/SOAR `api_key`. Django's `manage.py check --deploy` flags `DEBUG = True`. Nothing reports the
*misplaced file*, but the upgrade path has detection tools, and the warning doesn't point to them.
**Fix:** Change it to "nothing reports the misplaced file itself" and add: "After moving it, run ``python3
manage.py check_unattended`` (and ``python3 manage.py check --deploy``) to confirm the production values took
effect."

### IN-03: The upgrade warning doesn't cover what already ran against the development defaults

**File:** `docs/installation.rst:91-97`
**Issue:** On an upgraded host, anything run between the `git pull` and the `mv` used the default SQLite file
`src/fomo_db.sqlite3` and the console mail backend. That includes a `migrate`, which the page later tells
upgraders to re-run (line 191), and any cron `run_unattended` tick. After the move, the production database may
be missing those migrations, and those ticks' writes and notices went to the dev database and stdout.
**Fix:** Add one sentence: "If ``migrate`` or the unattended runner ran before the move, run ``python3 manage.py
migrate`` again afterwards; anything those runs wrote went to the default SQLite file, not this host's database."

### IN-04: The location rule in the parentheses is a guess, and `mv` overwrites silently

**File:** `docs/installation.rst:92-102`
**Issue:**
- The parentheses ("repository root (when ... `manage.py`) or in `src/` (when ... gunicorn or WSGI)") don't match
  how the old bare import resolved. Under `manage.py`, both the repo root (`sys.path[0]`) and `src/` (the
  editable-install `.pth`) were searched. PR #58's own `deploy/gunicorn.conf.py` `chdir`s to the repo root.
- Both `mv` commands sit in one console block and run without `-n`/`-i`. Pasting both on a host that has two
  copies replaces the copy that was actually in effect (the root one, which is first on `sys.path`) with the
  `src/` one. Either command also clobbers an existing `src/fomo/local_settings.py`.

**Fix:** "Check both places (``ls local_settings.py src/local_settings.py``); if both exist, the repository-root
copy was the one in effect." Use `mv -n` (or `mv -i`) in both commands.

### IN-05: "anything set there replaces the default" contradicts the runbook's FOMO_STATE_DIR trap

**File:** `docs/installation.rst:88-89` (vs `docs/runbooks/telescope_runs_calendar.rst:1986-1991`)
**Issue:** The new sentence promises that any setting in `local_settings.py` overrides the default. The runbook
explains that `FOMO_STATE_DIR` takes its default from `FOMO_LOCK_DIR` *before* the import. So overriding
`FOMO_LOCK_DIR` alone does not move `FOMO_STATE_DIR`, and `LCO_API_KEY` is folded in *after* the import. The import
is also not literally "at the end of the file": `settings.py:429-444` comes after it.
**Fix:** "...so a name assigned there replaces that setting (defaults derived from another setting are not
recomputed -- see the runbook's FOMO_STATE_DIR note), near the end of the file."

### IN-06: Lines edited in place break the surrounding wrap width

**File:** `docs/runbooks/telescope_runs_calendar.rst:2048`, `docs/installation.rst:150`
**Issue:** Runbook line 2048 is 110 columns in a paragraph wrapped at about 70. Installation line 150 is 116 columns
in a note wrapped at about 85. Both are under 120, so this is cosmetic, but the raw-source diffs and later reflows
look uneven.
**Fix:** Re-wrap both paragraphs to their existing widths.

---

_Reviewed: 2026-10-08T02:39:40Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
