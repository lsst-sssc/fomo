---
phase: 36-unattended-operation
fixed_at: 2026-09-18T15:10:00Z
review_path: .planning/phases/36-unattended-operation/36-REVIEW.md
iteration: 5
findings_in_scope: 32
fixed: 32
skipped: 0
status: all_fixed
---

# Phase 36: Code Review Fix Report (iteration 5)

**Fixed at:** 2026-09-18T15:10:00Z
**Source review:** `.planning/phases/36-unattended-operation/36-REVIEW.md`
**Iteration:** 5

**Summary:**
- Findings in scope: 32 -- the 10 "this iteration" findings (CR-04, WR-28..WR-32,
  IN-29..IN-32) plus the 22 carried-forward findings the review's own frontmatter
  reports as still open (9 from iteration 4, 13 from iteration 3). `fix_scope: all`.
- Fixed: 32
- Skipped: 0

Two findings the review lists as carried-forward were **not** re-fixed here because the
review itself already records them as closed/accepted, and re-checking confirmed that
still holds:
- **CR-02** and **WR-23** -- the review's own "Status of every finding carried into this
  iteration" section records both as CLOSED by plan 36-08 (the commits immediately
  before this fix pass). Verified: `grep -n FACILITIES docs/runbooks/telescope_runs_calendar.rst`
  returns nothing, and `settings.py`'s SOAR fold (subsequently hardened by WR-28 in this
  pass) is in place.
- **WR-22** -- carries a recorded acceptance (2026-09-18, 36-UAT.md Test 2) conditional
  on `LOGGING`'s root level staying at `INFO`. Verified the condition still holds
  (`src/fomo/settings.py` still has `'level': 'INFO'`) and left it alone, per the
  review's own disposition. Not counted in `findings_in_scope`.

All fixes were applied inside an isolated git worktree
(`.claude/worktrees/rf-36-1613182-1789739938`, branch `gsd-reviewfix/36-1613182`) and
fast-forwarded onto `issue37-telescope-runs-calendar`. Every commit was individually
verified before being made: the affected test module(s)
(`solsys_code.tests.test_settings_api_key_fold`, `solsys_code.tests.test_check_unattended`,
`solsys_code.tests.test_unattended`, `solsys_code.tests.test_backfill_lco_observations` --
160 tests total, run together as a final combined pass, all green), `pre-commit run ruff
--all-files`, `pre-commit run ruff-format --all-files`, and `pre-commit run sphinx-build
--all-files` for every runbook/conf.py edit. `python -c "import ast; ast.parse(...)"`
syntax-checked every edited `.py` file before its test run.

**A note on where the gates ran:** all verification above ran inside the isolated
worktree, which is a fresh checkout with no `local_settings.py` and no pre-existing
`src/fomo_db.sqlite3` -- `python manage.py migrate` was run once to create a throwaway
dev database for the one paired-notebook regeneration (IN-19), and the generated
`src/fomo/_version.py` (gitignored, setuptools_scm-generated, absent in a fresh worktree
checkout) was copied in from the main checkout so the package would import at all. None
of this is reproducible from the main checkout after the worktree is torn down, but
nothing here is a source-of-truth artifact either -- the migrations and copied
`_version.py` are pre-existing project machinery the worktree needed to bootstrap, not
part of the fix.

## Fixed Issues

### CR-04: Fresh-host procedure never told the operator to override SECRET_KEY/DEBUG/ALLOWED_HOSTS

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `76ed56e`
**Applied fix:** Added a paragraph to step 2 requiring a fresh `SECRET_KEY`,
`DEBUG = False`, and a real `ALLOWED_HOSTS`, and noted the `DEBUG`/`ALLOWED_HOSTS`
coupling (turning `DEBUG` off without also fixing `ALLOWED_HOSTS` makes every request
400). Verified with `pre-commit run sphinx-build`.

### WR-28: LCO_API_KEY fold's destination guard didn't match its source guard

**Files modified:** `src/fomo/settings.py`
**Commit:** `1105620`
**Applied fix:** Replaced the direct `FACILITIES['LCO']['api_key'] = ...` /
`FACILITIES['SOAR']['api_key'] = ...` assignments with a `setdefault()` loop over both
facility names, so a `local_settings.py` that replaces `FACILITIES` wholesale and omits
`SOAR` no longer crashes settings import with `KeyError: 'SOAR'`. Reproduced the exact
failure scenario from the review (a synthetic `FACILITIES` with only `LCO`/`GEM`) and
confirmed the fix resolves it with no exception.

### WR-29: Fold test's synthetic namespace couldn't detect a missing real SOAR entry

**Files modified:** `solsys_code/tests/test_settings_api_key_fold.py`
**Commit:** `9d1f66e`
**Applied fix:** Added `TestLiveFacilitiesCarriesBothFoldTargets`, asserting against the
live, already-imported `django.conf.settings.FACILITIES` (not the synthetic namespace
`_run_fold()` seeds) that both `LCO` and `SOAR` entries exist with an `api_key` key.

### WR-30: NameError test pinned a property of Python, not of this codebase

**Files modified:** `solsys_code/tests/test_settings_api_key_fold.py`
**Commit:** `55a3025`
**Applied fix:** Replaced `TestBracketedDictSubscriptRaisesNameError` (which executed
`exec("d['k']=1", {})` and would pass regardless of any change to FOMO's own fold logic)
with `TestFoldTailUsesStarImportIntoOwnNamespace`, which asserts the fold tail's actual
source contains the star-import statement and does not use a bare `except:`. Verified
this new test would catch the exact regression the old one could not (a hypothetical
regression to a bare `except:` makes the new regex assertion fail).

### WR-31: No preflight check for the LCO/SOAR portal credential

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`, `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `c62dc01`
**Applied fix:** Added `check_facility_credentials()` (a soft check matching
D-08/D-12's precedent), wired it into `Command.handle()`, added two tests
(missing/configured), and named it in the runbook's step 6 enumeration and step 7's
warning-count summary (updated "three" to "four").

### WR-32: `except ImportError: pass` swallowed errors from inside local_settings.py

**Files modified:** `src/fomo/settings.py`
**Commit:** `a0b2ecf`
**Applied fix:** Narrowed the except clause to check `exc.name != 'fomo.local_settings'`
before re-raising, so only the module's own absence is swallowed. Verified with the full
fold test suite (all 4, now 6, tests green).

### IN-29: Step 2's consequence sentence over-generalized and named the mechanism, not the symptom

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `8ff75a0`
**Applied fix:** Rewrote to name LCO/SOAR specifically (not "any portal call" -- GEM/ESO
have their own credentials), and to describe the symptom (`status_refresh` fails, a
failure email every 15 minutes) rather than the mechanism.

### IN-30: Step 2's NameError explanation used undefined terms and the wrong register

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `feaa201`
**Applied fix:** Rewrote in plain operator-facing language, per the review's suggested
text.

### IN-31: SOAR key-copy safety premise lived only in a comment far from the edit site

**Files modified:** `src/fomo/settings.py`, `solsys_code/tests/test_settings_api_key_fold.py`
**Commit:** `386735c`
**Applied fix:** Added the warning directly at the `FACILITIES['SOAR']` entry (where a
future `portal_url` repoint would actually happen), and added
`TestSoarPortalUrlMatchesLcoBeforeKeyIsCopied` asserting the two `portal_url` values
still match.

### IN-32: Three test-module hygiene items

**Files modified:** `solsys_code/tests/test_settings_api_key_fold.py`
**Commit:** `4e5b263`
**Applied fix:** Switched to `django.conf.settings.SETTINGS_MODULE`; replaced the bare
`assert` locating the fold-tail anchor with `self.fail(...)` inside an `if` guard; and
dropped the two inert `# noqa: S102` comments (`S` is not in this project's ruff
`select` list).

### CR-03: `local_settings.py` rendered verbatim into generated HTML docs

**Files modified:** `docs/conf.py`, `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `e2ed553`
**Applied fix:** Added `'*/local_settings.py'` to `autoapi_ignore`, and corrected the
runbook's boundary claim (credentials "must never go into a committed file" -> also
warns against serving a docs build from a configured host). **Verified end-to-end**: ran
a real `sphinx-build` with a `local_settings.py` containing a marker secret injected
into the worktree, confirmed `fomo.local_settings` no longer appears among the
highlighted modules and the marker string appears nowhere in the built HTML, then
removed the injected file before committing. Note: this does not retroactively clean the
`_readthedocs/html/`/`docs/_build/html/` trees already on disk in the **main checkout**
(not present in this worktree) -- per the review's fix part 2, that requires a human
judgment call (were those builds ever served/copied/shared?) and, if so, ping-URL
rotation, both outside this pass's scope.

### WR-24: Heartbeat alert arithmetic attached to the wrong check-type route

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `49023d2`
**Applied fix:** Split the formula so it applies only to the Simple-check route it's
actually true for, and gave the Cron-type alternative its own (correct) alerting
description (missed slot + grace, no interval to set) instead of inheriting the wrong one.

### WR-25: Step 3's opening sentence misattributed the failure class FOMO cannot report itself

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `a495a3b`
**Applied fix:** Moved the relative clause to its correct antecedent, per the review's
suggested rewrite.

### WR-26: Runbook never mentioned FOMO_STATE_DIR or the flock -E version requirement

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `c8b81e2`
**Applied fix:** Added a sentence to step 1 about `FOMO_STATE_DIR`'s default and when it
needs its own directory; extended step 6's enumeration to name the suppression-state
directory and the util-linux 2.27+ requirement for `-E`.

### WR-27: "Either route produces the same line" was false when flock/paths diverge

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`,
`deploy/cron/fomo.crontab.example`
**Commit:** `f850dbd`
**Applied fix:** Rewrote step 8 to state the printed line is authoritative and the
template is a fallback; extended the template header to list all the values an operator
must verify (not just the two named placeholders) before trusting a hand-edited line.

### IN-25: 15/20/35-minute alert window restated in five places

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `4ba6e3a`
**Applied fix:** Dropped the derived-arithmetic restatement from step 3, keeping the two
input values and delegating the arithmetic to "The two failure signals" (the one place
it should live).

### IN-26: "Name each concept first" drafting directives leaked into operator prose

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `df7eb85`
**Applied fix:** Removed both instances of the directive phrasing, keeping only the
naming's product (the parenthetical spelling).

### IN-27: Stale `[ $? -eq 99 ]` comment and "a later plan in this phase" reference

**Files modified:** `deploy/cron/fomo.crontab.example`
**Commit:** `fcaa8a0` (the header's "a later plan in this phase" for `check_unattended`
was incidentally already fixed by the WR-27 rewrite in the prior commit)
**Applied fix:** Updated the stale `[ $? -eq 99 ]` reference to `[ $rc -eq 99 ]`, and
dropped the remaining "(a later plan in this phase)" parenthetical for
`deploy/logrotate/fomo.example`.

### IN-28: Single-backtick deploy-file paths render as italics under Sphinx's default role

**Files modified:** `docs/runbooks/telescope_runs_calendar.rst`
**Commit:** `5003b58`
**Applied fix:** Switched both to double backticks.

### WR-16: Lock-held skip propagated as a non-zero exit code, contradicting run_tick()'s own contract

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`, `deploy/cron/fomo.crontab.example`
**Commit:** `eafda16`
**Applied fix:** Normalized `$rc` back to 0 inside the skip-tail's brace group, in both
`cron_line()` and the committed template, and corrected both prose claims. **Verified
end-to-end**: new `TestCronLine.test_lock_held_exit_is_normalized_to_zero` splices a
stub in place of the real `flock ... run_unattended` invocation (keeping the shipped
tail verbatim) and runs it in a real `sh` subprocess for all three cases (healthy=0,
failing=1, lock-held=0). **Flagging as requiring human verification per the
verification_strategy's logic-bug limitation** -- this changes exit-code semantics a
production cron supervisor/monitoring wrapper may already depend on; the test proves the
new contract is internally consistent, not that no external tooling assumed the old one.

### WR-17: save_state() had no runtime fallback for a directory that goes bad after setup

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`,
`solsys_code/management/commands/check_unattended.py`
**Commit:** `72115bb`
**Applied fix:** **Adapted from the review's literal suggestion.** The review proposed a
module-level "already failed" flag reset per process; `run_unattended` is a fresh
process per cron tick with no in-process loop, so a module-level flag cannot survive
between ticks and would have been dead code in production (only exercisable by calling
`run_tick()` twice within one test process). Implemented a persistent fallback instead:
`save_state()` writes to a location outside `FOMO_STATE_DIR` (the system temp directory)
when the primary write fails, `load_state()` reads whichever of the two was written most
recently, and the fallback is deleted once the primary becomes writable again. **Verified
end-to-end** with a real unwritable directory across two separate `run_unattended`
processes (`call_command` invocations): exactly one email, not two. **Flagging as
requiring human verification** -- this is a genuine behavioral/persistence-semantics
change beyond the review's original suggestion, and the test cannot fully substitute for
observing an actual multi-tick outage in production.

### WR-18 / IN-18: Discovery skip reasons logged at DEBUG, filtered out by this project's own LOGGING config

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `9ec25d0`
**Applied fix:** Promoted the log call to INFO (matching the function's sibling
operator-facing line), updated the test to assert at INFO instead of DEBUG (the old
version passed even though the fix's own stated goal -- "they must now reach the log" --
wasn't achieved), and corrected the accompanying comment's overstated D-17 claim (per
IN-18, folded into the same commit since both findings describe the same three lines).

### WR-19: check_email() rejected only the console backend, not dummy/locmem/filebased

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`
**Commit:** `7535ebf`
**Applied fix:** Replaced the single string comparison with a lookup against all four
Django-shipped non-delivering backends. **Adapted for an interaction the review didn't
flag**: Django's test runner swaps `EMAIL_BACKEND` to `locmem` for the whole suite, which
is now itself one of the rejected backends -- without a fix, every other test in this
module would have started failing the `EMAIL_BACKEND` check by default. Added a
same-behavior stand-in (`_FakeDeliveringEmailBackend`, a `locmem.EmailBackend` subclass
under a different dotted path, so `mail.outbox` capture still works) as the base
fixture's `EMAIL_BACKEND`, and gave each of the three newly-rejected backends its own
explicit test.

### WR-20: check_flock()'s subprocess probe could hang or crash the whole preflight

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`
**Commit:** `6656529`
**Applied fix:** Added `timeout=5` and a `try/except (OSError, subprocess.TimeoutExpired)`
degrading to a reported `CheckResult`, matching the hardening `_owner_mode()` already
had. Added tests for both the `OSError` and `TimeoutExpired` cases.

### IN-17: sweep_watched_rows()'s docstring/type hints stale after the runner started using stdout

**Files modified:** `solsys_code/management/commands/backfill_lco_observations.py`
**Commit:** `4e49b00`
**Applied fix:** Corrected the docstring (the runner passes a live `io.StringIO()`, not
`None`) and widened the type hints from `io.StringIO | None` to `typing.TextIO | None`
(the bare-invocation caller passes Django `OutputWrapper` instances).

### IN-19: Ungrammatical --proposal guard message, baked into the paired notebook

**Files modified:** `solsys_code/management/commands/backfill_lco_observations.py`,
`solsys_code/tests/test_backfill_lco_observations.py`,
`docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`
**Commit:** `3321f4b`
**Applied fix:** Added a singular/plural verb branch, a test for each case, and
regenerated the paired notebook (`jupyter nbconvert --to notebook --execute --inplace`)
per CLAUDE.md's paired-docs rule -- confirmed the diff contains only the grammar fix and
expected re-execution timestamps.

### IN-20: Preflight's heartbeat reminder named only Period, never Grace

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`
**Commit:** `c3a545f`
**Applied fix:** Extended the `[ok] heartbeat` detail to name both knobs, and introduced
`_CRON_INTERVAL_MINUTES`/`_RECOMMENDED_HEARTBEAT_GRACE_MINUTES` as the single source both
`cron_line()`'s schedule and this reminder read, closing the "third hardcoded copy of 15"
gap the review also named.

### IN-21: save_state()'s temp files leaked forever on a process kill

**Files modified:** `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py`
**Commit:** `8fe7813`
**Applied fix:** Added `_reap_stale_temp_files()`, called at the top of
`_atomic_write_json()`, removing any of its own leftover temp files older than one cron
tick interval (any that old can only be a kill-leftover). Verified with a test seeding
both a stale and a fresh temp file and confirming only the stale one is removed.

### IN-22: Default-path constants duplicated in three places with nothing enforcing sync

**Files modified:** `solsys_code/management/commands/check_unattended.py`
**Commit:** `576b4ff`
**Applied fix:** **Chose the "export once" remedy** (the review offered this or "add one
test" as alternatives) -- `check_unattended.py` now imports `_DEFAULT_LOCK_DIR`/
`_DEFAULT_LOG_FILE` from `unattended.py` instead of redefining them, making drift
structurally impossible rather than merely tested against. Verified the existing test
that patches `check_unattended._DEFAULT_LOCK_DIR` still passes (patching an imported
name overrides the binding in the importing module's own namespace).

### IN-23: Non-system flock path unflagged; template-test could fail with an opaque StopIteration

**Files modified:** `solsys_code/management/commands/check_unattended.py`,
`solsys_code/tests/test_check_unattended.py`
**Commit:** `d7402ae`
**Applied fix:** Added an allow-list check in `check_flock()`'s success path that notes
when the resolved `flock` falls outside `/usr/bin`, `/bin`, `/usr/sbin`, `/sbin`; and
hardened `test_line_matches_the_committed_template_token_for_token` with `next(..., None)`
+ `assertIsNotNone` (a named failure instead of a bare `StopIteration`) and resolved the
repo root from `settings.BASE_DIR` instead of `Path(__file__).resolve().parents[2]`.

### IN-24: Backfill demo notebook was the only one missing kernelspec metadata

**Files modified:** `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb`
**Commit:** `f6be59b`
**Applied fix:** Added the same `kernelspec` block every sibling `pre_executed/`
notebook already carries. Metadata-only edit (5-line diff); did not require
re-execution.

## Skipped Issues

None — all in-scope findings were fixed.

## Items requiring human verification (not "skipped", but flagged per the
verification_strategy's logic-bug limitation)

Two fixes involve behavioral/semantic changes beyond syntax-level verification, each with
an end-to-end test proving internal consistency but not full production observation:

- **WR-16** (`eafda16`) -- normalizing the lock-held cron exit code from 99 to 0 changes
  what any external supervisor/monitoring wrapper watching the cron line's exit status
  sees. Recommend confirming no external tooling (outside this repo) depended on
  observing 99.
- **WR-17** (`72115bb`) -- the state-persistence fallback is a genuine design addition
  (a second on-disk location `save_state()`/`load_state()` now read/write), adapted from
  the review's suggestion because the original (a module-level flag) would have been
  inert in this project's one-shot-process-per-tick architecture. Recommend a human
  read-through of `_primary_state_path()`/`_newest_existing_state_path()`/`save_state()`
  in `solsys_code/unattended.py` before relying on it in production.

## Follow-up not addressed by this pass (out of scope for a code fixer)

- **CR-03's remedy part 2**: the real `_readthedocs/html/` and `docs/_build/html/` trees
  in the **main checkout** (not present in this isolated worktree) may still contain a
  real heartbeat ping URL and other credentials rendered before the `autoapi_ignore` fix
  landed. A human should judge whether either build was ever served, copied, or shared
  and, if so, delete both trees and rotate the healthchecks.io check (new check, re-export
  `FOMO_HEARTBEAT_URL`).

---

_Fixed: 2026-09-18T15:10:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 5_
