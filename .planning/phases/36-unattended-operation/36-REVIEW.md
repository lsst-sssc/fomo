---
phase: 36-unattended-operation
reviewed: 2026-09-18T12:00:00Z
depth: deep
iteration: 5
files_reviewed: 3
files_reviewed_list:
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/tests/test_settings_api_key_fold.py
  - src/fomo/settings.py
findings:
  critical: 1
  warning: 5
  info: 4
  total: 10
carried_forward_open: 22
status: issues_found
---

# Phase 36: Code Review Report (iteration 5 — incremental review of plan 36-08, gap closure G-36-4)

**Reviewed:** 2026-09-18T12:00:00Z
**Depth:** deep
**Files Reviewed:** 3 (everything changed since `a3e5556`, the commit iteration 4 was written against — commits `62d4d78` and `39312f4`; `96701a3` and `c04e245` are planning artifacts and out of scope)
**Status:** issues_found

## Summary

Plan 36-08 closes G-36-4 in two commits: `62d4d78` adds one line to the settings fold
(`src/fomo/settings.py:442-443`) plus a new 126-line test module, and `39312f4` rewrites
the fresh-host procedure's step 2 (`docs/runbooks/telescope_runs_calendar.rst:1458-1470`).
The whole source diff is **+127 / -2 lines**. Everything below was checked by execution,
not by reading the plan's prose.

**Both findings the plan claims to close are genuinely closed.**

- **CR-02 (iteration 4) — CLOSED.** The `FACILITIES['LCO']['api_key']` instruction is gone
  from the runbook *entirely*: `grep -n FACILITIES docs/runbooks/telescope_runs_calendar.rst`
  now returns nothing, so there is no copy-pasteable nested key path left anywhere on the
  page. The replacement names the flat `LCO_API_KEY` with a bracketed placeholder
  (`'<your key>'`), and its mechanism sentence is a faithful paraphrase of
  `settings.py:436-439`. An operator following the new step 2 literally gets a working
  host instead of a settings module that refuses to import.
- **WR-23 (iteration 4) — CLOSED, by remedy (b), and the code half verifies end to end.**
  `settings.py:443` now folds the same flat name into `FACILITIES['SOAR']['api_key']`, and
  that is demonstrably the key the consumer reads: `OCSSettings.get_setting()` resolves
  `settings.FACILITIES.get('SOAR', default_settings).get('api_key', …)`
  (`tom_observations/facilities/ocs.py:98-99`), and `step_status_refresh()` really does
  instantiate `SOARFacility()` alongside `LCOFacility()`
  (`solsys_code/unattended.py:267-268`). Without the `FACILITIES['SOAR']` entry the
  accessor silently falls back to `LCOSettings.default_settings`, whose `api_key` is `''` —
  so the entry plus the fold are both load-bearing, and both are present.

**The new test really does execute the live settings tail, and it does not leak state.**
It reads `settings_module.__file__`, slices from the `try:\n    from fomo.local_settings
import *` anchor to EOF, and `exec`s that slice — so deleting `settings.py:443` fails
`test_flat_key_fills_lco_and_soar`, and moving the fold above the import guard fails it
too. I probed the two leak vectors that matter rather than assuming: a star-import of an
injected submodule does **not** rebind the parent package's attribute (verified in this
interpreter), so `addCleanup`'s `sys.modules` restore is complete; and the cold case (no
`fomo.local_settings` in `sys.modules` at all, i.e. a CI host with no local settings file)
resolves against the injected module correctly, so the module is portable and never reads
the real file on disk. `django.conf.settings` is untouched — the fold executes into a
throwaway `dict`, and the one `override_settings` use is properly scoped. All four tests
pass (`python manage.py test solsys_code.tests.test_settings_api_key_fold`, 4 tests, 0.003 s,
no database). `pre-commit run ruff` and `ruff-format` on both changed Python files: clean.

**But the test buys less assurance than its own docstring claims, in two specific ways
(WR-29, WR-30), and the fold it protects has an unguarded destination path (WR-28).** The
namespace the test `exec`s into is *seeded* with `{'LCO': {…}, 'SOAR': {…}}`, so the real
`FACILITIES['SOAR']` entry at `settings.py:244-247` — which the new fold line now requires
to exist — is the one thing the test cannot see. Delete that entry and the test still
passes, while a configured host dies at settings import with `KeyError: 'SOAR'`
(reproduced). The asymmetry is what makes it worth fixing: the guard is `if 'LCO_API_KEY'
in globals()`, so CI (no key) never executes the subscript and never notices, while
production (key set) crashes on every process start.

**RST and gate status: clean.** A `docutils` parse of the whole page at `report_level=1`
produces no structural message (only three "hyperlink target not referenced" INFOs, all
pre-existing) — the rewritten step 2 keeps the enumerated list, the three-space
continuation indent and the ~70-column wrap intact. Plan 36-07's slice gate still holds in
full: every presence clause (`Period`, `Grace`, `*/15 * * * *`, `hc-ping.com/<uuid>`,
`healthchecks.io`, `The two failure signals`, `35 min`, a `^9. ` step) passes, and the
order clause still passes with room to spare — `Period` at slice line 39 and the ping-URL
placeholder at slice line 51 both precede the export anchor at slice line 55 (step 2 grew
by 8 lines, which moved all three anchors down together). The canonical-paragraph
exclusion and the UUID negative-grep also still pass. I deliberately did **not** run the
gate's `pre-commit run sphinx-build --all-files` clause: it writes into the working tree,
and per **CR-03** (still open) that build is what renders `local_settings.py` — including
this host's real credentials — into HTML.

**Credential hygiene inside the three files: clean for the change, with one pre-existing
exception.** No UUID-shaped, key-shaped or token-shaped literal appears in the diff; the
runbook shows only `'<your key>'`, and the test's literal is
`'fake-portal-key-test-settings-api-key-fold'` — obviously non-credential and asserted only
by equality, never printed. The exception is not in the diff but is in a reviewed file: the
committed `SECRET_KEY` and `DEBUG = True` at `settings.py:25,28`, which the fresh-host
procedure this change edits never tells the operator to override — see **CR-04**.

**Scope fence respected.** Plan 36-08 touched nothing outside G-36-4, exactly as it said it
would; none of iteration 4's other open findings are re-opened or regressed by it, and all
are carried forward below by ID. The CLAUDE.md paired-docs rule is satisfied: the changed
behavior's documented surface is `docs/runbooks/telescope_runs_calendar.rst` and it was
updated in the same plan; no module in the notebook pairing map was touched.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-04: The fresh-host procedure stands up a gunicorn/uWSGI host on a `SECRET_KEY` that is published in this git repository, with `DEBUG = True` — step 2 enumerates what goes in `local_settings.py` and omits every one of them

**File:** `docs/runbooks/telescope_runs_calendar.rst:1458-1470` (step 2, rewritten by
`39312f4`); cf. `src/fomo/settings.py:25`, `:28`, `:30-32`, and
`docs/runbooks/telescope_runs_calendar.rst:1507-1509`
**Issue:** Step 2 is the only place in the entire published doc set that enumerates what a
real host must put in `local_settings.py`. `grep -rniE 'secret_key|allowed_hosts' docs/`
returns **nothing** — `docs/installation.rst` never mentions them either (its one "For a
real deployment" note, `:110-118`, covers `FOMO_BASE_URL` only). The list step 2 gives is
`EMAIL_BACKEND`, `EMAIL_HOST_*`, and now `LCO_API_KEY`. It stops there.

That this is a production host, serving the web app, is the runbook's own claim: step 5
tells the operator to export `FOMO_BASE_URL` "in **both** the cron environment and the web
server's (gunicorn/uWSGI) environment" because "the campaign-submission approval-queue link
is built inside the web process" (`:1507-1510`). So an operator who follows this page
end-to-end runs a public Django site with:

```python
SECRET_KEY = '1c1nvy&amp;t@z+wq16gbfag8_-t&amp;e#mppk4h=syp*i*fs^hi&amp;7ihi'   # settings.py:25 — in git
DEBUG = True                                                                 # settings.py:28
ALLOWED_HOSTS = ['tlister-thinkmate.lco.gtn']                                # settings.py:30-32
```

A `SECRET_KEY` anyone can read in the public repository is a signing-key compromise, not a
style issue: it forges session cookies (`django.contrib.sessions` signed cookies and the
session-id-independent `_auth_user_hash`), password-reset tokens, and any
`signing.dumps()` payload — i.e. staff-account takeover on the very accounts this phase
mails failure notices to and which approve campaigns. `DEBUG = True` adds full traceback
and settings disclosure on any unhandled exception. Neither is detected by
`check_unattended` (see WR-31), so the operator gets a green preflight.

I am recording this as a BLOCKER on the procedure, not as a regression: the omission
pre-dates `39312f4`, and `CLAUDE.md` legitimately calls the two values dev defaults whose
"production overrides belong in a `local_settings.py`". The defect is that the one step
which tells an operator what to put in `local_settings.py` — the step this change opened up
and expanded — still does not say so, and no other document does either. It should not
ship in that state.
**Fix:** add the three settings to the step that already exists, and note the
`ALLOWED_HOSTS` coupling (turning `DEBUG` off without it makes every request 400):

```rst
2. Put the real ``EMAIL_BACKEND`` (and its ``EMAIL_HOST_*`` settings) and
   the LCO/SOAR API key in this host's ``local_settings.py`` ...

   This host must also override three development defaults in the same file, or it
   will serve the site with a signing key that is public in this repository:
   ``SECRET_KEY`` (generate a fresh one -- ``python -c "from django.core.management.utils
   import get_random_secret_key; print(get_random_secret_key())"``), ``DEBUG = False``,
   and ``ALLOWED_HOSTS`` set to this host's real names -- ``DEBUG = False`` with the
   committed single-entry ``ALLOWED_HOSTS`` makes every request return 400, so the two
   must change together.
```

## Warnings

### WR-28: The new fold line guards its source name but not its destination path — a `FACILITIES` override that omits `SOAR` now kills settings import with an uncaught `KeyError`

**File:** `src/fomo/settings.py:440-443`
**Issue:** The fold is

```python
if 'LCO_API_KEY' in globals():
    FACILITIES['LCO']['api_key'] = LCO_API_KEY
    FACILITIES['SOAR']['api_key'] = LCO_API_KEY   # new
```

The presence guard covers only the *source* name. `FACILITIES` at this point is whatever
survived the star import three lines above: `local_settings.py` can legally assign a whole
new `FACILITIES` dict (that is precisely the one thing it *can* do, per the comment at
`:436-439`), and a TOM deployment's stock `FACILITIES` block carries `LCO` and `GEM` — not
`SOAR`, which is a FOMO-local addition (`:240-247`). Reproduced against the real tail
source with a `local_settings` that sets `LCO_API_KEY` and a `FACILITIES` of
`{'LCO': …, 'GEM': …}`:

```
RAISED KeyError 'SOAR'
```

Nothing catches it — the `except ImportError` guard is already closed by then — so the
settings module fails to import and gunicorn, every `manage.py` command, the
`run_unattended` tick, `check_unattended` and the failure mail all die at start-up. That is
the exact blast radius iteration 4's CR-02 described, arriving by a different door. The
pre-existing `['LCO']` subscript has the same shape, but `LCO` is the one key any
`FACILITIES` override is overwhelmingly likely to keep; `SOAR` is not. Not a BLOCKER only
because it takes a second, undocumented operator action to trigger and fails loudly with a
traceback that names the line.
**Fix:** make the destination as tolerant as the source guard, in one line each:

```python
if 'LCO_API_KEY' in globals():
    # SOAR authenticates against the same LCO Observation Portal (see the FACILITIES['SOAR'] entry above).
    for _facility in ('LCO', 'SOAR'):
        FACILITIES.setdefault(_facility, {})['api_key'] = LCO_API_KEY  # noqa: F405
```

(or `if 'SOAR' in FACILITIES:` if silently skipping is preferred to creating the entry —
but then say so in the comment, because a silently skipped SOAR fold is WR-23 all over
again).

### WR-29: The test seeds its own `FACILITIES`, so the one prerequisite the new fold line added — the `FACILITIES['SOAR']` entry — is exactly what it cannot detect, and CI structurally cannot either

**File:** `solsys_code/tests/test_settings_api_key_fold.py:81-83`; cf.
`src/fomo/settings.py:244-247`
**Issue:** `_run_fold()` executes the real tail source, but into

```python
namespace = {'FACILITIES': {'LCO': {'api_key': ''}, 'SOAR': {'api_key': ''}}}
```

so the `FACILITIES` the fold mutates is synthetic. Delete `settings.py:244-247` (the real
`SOAR` entry, whose own comment says it exists only so `SOARSettings('SOAR')` resolves a
real `api_key`) and all four tests still pass — while every configured host raises
`KeyError: 'SOAR'` at import (WR-28). The module docstring's promise, "a future edit that
drops the SOAR line fails this test instead of passing a source-token grep", is true for
`:443` and false for `:244-247`, which is the half a future editor is more likely to prune
as "unused duplication of LCO".

The failure mode is also invisible to CI by construction: the fold is behind `if
'LCO_API_KEY' in globals()`, and a CI checkout has no `local_settings.py`, so the subscript
is never executed there. The only environment that can notice is production, at start-up.
**Fix:** assert against the live settings object, not only against the synthetic namespace —
two lines in a new case:

```python
class TestLiveFacilitiesCarriesBothFoldTargets(SimpleTestCase):
    def test_live_facilities_has_both_entries_the_fold_writes_into(self):
        from django.conf import settings as live

        for facility in ('LCO', 'SOAR'):
            self.assertIn(facility, live.FACILITIES)
            self.assertIn('api_key', live.FACILITIES[facility])
```

### WR-30: `TestBracketedDictSubscriptRaisesNameError` asserts a property of Python, not a property of this codebase — it cannot fail if the behavior it claims to pin regresses

**File:** `solsys_code/tests/test_settings_api_key_fold.py:106-115`
**Issue:** The case is

```python
with self.assertRaises(NameError):
    exec("FACILITIES['LCO']['api_key'] = 'placeholder'", {})
```

Nothing in that statement touches FOMO. It executes a string literal in an empty dict and
observes that Python raises `NameError` for an unbound name — which is true of every
Python program ever written. Its docstring nevertheless claims it "pins G-36-4's failure
mode as an executable case so the class of defect it belongs to cannot return silently."
It cannot: change `settings.py:431-434` to `exec(open(path).read(), globals())` (a real and
tempting "fix" for the flat-name awkwardness), and the nested form would start working, the
runbook's `NameError` explanation would become false — and this test would still pass,
green. A test that cannot fail for any change to the system under test is not a regression
guard; here it is worse than absent, because the docstring invites a future maintainer to
trust it.
**Fix:** exercise the real import path — write a throwaway module *file* containing the
nested assignment, put its directory on `sys.path` under the `fomo` package name, and
assert the live tail raises. If that is judged too heavy, the honest cheap version is to
pin the mechanism the claim actually rests on: assert that the tail slice contains
`from fomo.local_settings import *` and that its guard catches `ImportError` only, and
rewrite the docstring to say the case documents Python's scoping rule rather than pinning
FOMO's behavior.

### WR-31: The new prerequisite is one the preflight cannot check, while step 6 promises it "reports every prerequisite in one pass" and step 7 says to iterate until it is green

**File:** `docs/runbooks/telescope_runs_calendar.rst:1524-1547` (step 6), `:1548-1549`
(step 7), `:1468-1470` (step 2's consequence sentence); cf.
`solsys_code/management/commands/check_unattended.py` (checks: `check_flock`,
`check_lock_dir`, `check_log_dir`, `check_state_dir`, `check_email`, `check_heartbeat`,
`check_base_url`, `check_watched_proposals` — no facility-credential check)
**Issue:** Step 2 now ends: "Leave the setting out and both facility entries stay empty, so
any portal call FOMO makes -- including the unattended tick's status refresh -- goes out
unauthenticated." Step 6, four steps later, says `check_unattended` "reports every
prerequisite in one pass", and step 7 says "Fix whatever it reports, re-running
``check_unattended`` until every hard check passes." An operator who skipped or fat-fingered
step 2 therefore gets a fully green preflight and a printed cron line, and only discovers
the problem when `status_refresh` starts failing on live records. The check is a one-liner
away: TOM already ships `OCSSettings.get_unconfigured_settings()`
(`tom_observations/facilities/ocs.py:101-105`), which returns the blank required keys for a
facility — exactly the shape `check_unattended`'s other checks use.

This is the second time in this phase that step 6's enumeration has been found to describe a
preflight that has moved on (see WR-26, still open, for `FOMO_STATE_DIR` and the `flock -E`
probe). Both should be fixed in the same pass.
**Fix:** add a soft check and name it in step 6's enumeration:

```python
def check_facility_credentials() -> CheckResult:
    """LCO/SOAR portal credentials -- a warning, not a hard failure: a host with no
    LCO/SOAR ObservationRecords ticks fine without them."""
    missing = [f for f in ('LCO', 'SOAR') if not SOARSettings(f).get_setting('api_key')]
    ...
```

and, in step 6, extend the list with "whether the LCO/SOAR portal API key is configured
(a warning: the tick's `status_refresh` step fails on every non-terminal record without
it)". Never print the value — set/unset only, as `check_heartbeat()` already does (D-15).

### WR-32: `except ImportError: pass` swallows import failures raised *inside* `local_settings.py`, silently reverting a configured host to every dev default the new step 2 just told the operator to override

**File:** `src/fomo/settings.py:431-434`
**Issue:**

```python
try:
    from fomo.local_settings import *  # noqa
except ImportError:
    pass
```

The guard's intent is "the file may not exist", but its reach is "any `ImportError` or
`ModuleNotFoundError` raised anywhere while executing that module". A `local_settings.py`
that does `from fomo.secrets import LCO_API_KEY` (typo, un-deployed sibling file, a package
missing from this host's venv) is discarded **whole and silently**: the host then runs with
`DEBUG = True`, the committed `SECRET_KEY`, `EMAIL_BACKEND` still pointed at the console
backend (so every failure email this phase exists to send is written to cron's stdout and
lost), and both facility `api_key` entries empty. `check_unattended`'s `check_email()` would
catch the console backend — but only if the operator runs it again after the breakage, and
step 7 is a setup-time step. Pre-existing, and `CLAUDE.md` does describe the import as
"fallback: no error on missing"; but this change is what makes `local_settings.py` the
documented home of a second credential, which raises the cost of losing it silently.
**Fix:** catch only the absence of that one module, and let anything else propagate:

```python
try:
    from fomo.local_settings import *  # noqa
except ImportError as exc:
    if exc.name != 'fomo.local_settings':
        raise
```

## Info

### IN-29: Step 2's consequence sentence over-generalizes to facilities the key cannot reach, and describes the mechanism rather than the symptom the operator will see

**File:** `docs/runbooks/telescope_runs_calendar.rst:1468-1470`
**Issue:** "Leave the setting out and both facility entries stay empty, so **any portal
call FOMO makes** -- including the unattended tick's status refresh -- goes out
unauthenticated." Two inaccuracies. (1) "any portal call" is wrong for GEM and ESO, which
have their own credentials (`settings.py:248-265` — `FACILITIES['GEM']['api_key']` is a
nested `{'GS': …, 'GN': …}` dict with no flat fold and no documentation anywhere, i.e. the
same G-36-4 trap, still open for Gemini); the page's own "What runs, and when" section is
careful about this, saying the tick "never touches Gemini or ESO" (`:1433-1434`). (2)
"goes out unauthenticated" is the mechanism; the symptom is that
`update_all_observation_statuses()` gets a 4xx and raises `ImproperCredentialsException`,
which `_refresh_one_facility()` converts into an outage result
(`solsys_code/unattended.py:222-226`) — so the operator sees `status_refresh` fail and a
failure email every 15 minutes, which is what they will actually be debugging.
**Fix:** "…both the LCO and SOAR facility entries stay empty, and the tick's
``status_refresh`` step then fails on every non-terminal LCO/SOAR record — a failure email
every 15 minutes. (Gemini and ESO have their own credentials; this key does not reach
them.)"

### IN-30: Step 2's new explanation uses three terms an operator has not been given

**File:** `docs/runbooks/telescope_runs_calendar.rst:1462-1466`
**Issue:** "because **this module** is imported into its own namespace, so it can only
**ASSIGN** new settings: reaching into a setting already built **above it** raises
``NameError``, which **the import guard** does not catch". The reader has never been told
there is an import guard, "above it" is meaningful only if you know where
`local_settings.py` is imported from, "this module" can plausibly be read as
`src/fomo/settings.py` (named two sentences later), and the shouted `ASSIGN` is carried
over from the source comment at `:436-437`, where its audience is a developer reading code.
Everything factual is correct; it is the register that slipped.
**Fix:** "…because ``settings.py`` imports ``local_settings.py`` into a namespace of its
own, near the end of the file: names you set there become settings, but anything you try to
reach *into* -- a dictionary settings.py already built -- is not visible, and the attempt
raises ``NameError`` before Django finishes starting."

### IN-31: The fold copies the LCO key into the SOAR entry unconditionally, while the premise that makes that safe lives only in a comment nothing re-checks

**File:** `src/fomo/settings.py:240-247`, `:442-443`
**Issue:** Both the entry's comment and the fold's comment assert that SOAR authenticates
against the same LCO Observation Portal, which is true today because
`FACILITIES['SOAR']['portal_url']` is literally `https://observe.lco.global` (`:245`).
Nothing ties the two together: an editor who later repoints that `portal_url` at a
NOIRLab-hosted SOAR portal (the obvious future edit, and the comment sits four lines above
it) silently starts sending the LCO Observation Portal key to a third-party host, with no
test and no comment at the edit site to stop them. Low likelihood, but the blast radius is a
credential disclosure to a new party.
**Fix:** put the warning where the edit will happen — extend the `:240-243` comment with
"if this `portal_url` is ever repointed away from `observe.lco.global`, remove the
`FACILITIES['SOAR']['api_key']` line in the fold at the end of this file: it copies the LCO
portal key" — and add the coupling as an assertion in the new test module.

### IN-32: Three small hygiene items in the new test module

**File:** `solsys_code/tests/test_settings_api_key_fold.py:56`, `:63`, `:82`, `:115`
**Issue:** (1) `:56` reads `environ['DJANGO_SETTINGS_MODULE']` directly; Django's own
`django.conf.settings.SETTINGS_MODULE` is the supported accessor and is correct even under a
runner that configures settings without the environment variable — as written the module
raises `KeyError` rather than skipping. (2) `:63` uses a bare `assert` for the anchor
check, which `python -O` strips; the slice then becomes the file's last character and
`test_flat_key_fills_lco_and_soar` fails with an opaque `'' != 'fake-portal-key-…'` instead
of "anchor not found". `self.fail(...)` or `assertNotEqual` is immune. (3) `:82` and `:115`
carry `# noqa: S102`, but `S` (flake8-bandit) is not in this project's `select` list
(`pyproject.toml:87-109`) and `RUF100` is not enabled either, so both directives are inert
noise that reads as if a real rule were being suppressed.
**Fix:** `settings.SETTINGS_MODULE`; `self.fail(f'fold-tail anchor not found in {settings_path}')`
inside an `if anchor_index == -1:`; and drop the two `noqa` codes, keeping the explanatory
half of the comment.

## Security review (explicitly requested scope)

| Check | Result |
|---|---|
| Credential-shaped literal anywhere in the three files' diff | **None.** Runbook shows `'<your key>'` only; test literal is `'fake-portal-key-test-settings-api-key-fold'`, never printed |
| Real UUID / ping token in the runbook | **None** (`! grep -qiE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'` passes on the whole page) |
| Copy-pasteable nested `FACILITIES[...]` key path in the runbook | **None** — `grep -n FACILITIES` on the page returns nothing (CR-02 closed) |
| Secret steered into a process argument vector or an env var | **No.** Step 2 still says "never in the crontab line, never in an environment variable" |
| New credential written to a wider surface by the fold | **No.** `FACILITIES['SOAR']['portal_url']` is the same LCO host (see IN-31 for the forward-looking caveat) |
| Test leaking the operator's real `local_settings.py` | **No.** Injection resolves from `sys.modules`; verified the parent package attribute is not rebound and the cold case works. The real file is never opened |
| Secret steered into a file the toolchain publishes | **YES — CR-03 still open**, and this change increases traffic on that path by making `local_settings.py` the documented home of a second credential |
| Committed signing key used by the documented production procedure | **YES — CR-04 (new)** |

## Status of every finding carried into this iteration

**Closed by plan 36-08 (2):**

- **CR-02** — CLOSED. No `FACILITIES` token remains on the runbook page; flat `LCO_API_KEY`
  with a bracketed placeholder, mechanism correctly stated.
- **WR-23** — CLOSED via remedy (b). `settings.py:443` folds the key into the SOAR entry;
  consumer path verified through `OCSSettings.get_setting()` and `step_status_refresh()`.
  See WR-28/WR-29 for the robustness and coverage caveats on the closure.

**Still open from iteration 4 (9) — plan 36-08's scope fence deliberately excluded all of
them; none is a regression:**

- **CR-03** — still open, unremediated and unchanged: `docs/conf.py:63` `autoapi_ignore` is
  still `['*/__main__.py', '*/_version.py']`, and
  `_readthedocs/html/autoapi/fomo/local_settings/`,
  `_readthedocs/html/_modules/fomo/local_settings.html` and the `docs/_build/html/`
  equivalents are all still present in this working tree (confirmed by directory listing
  only — not opened). Aggravated in kind by this change, which makes `local_settings.py`
  the documented home of the LCO/SOAR key.
- **WR-24** — still open. Step 3 still reads "Create ONE check for this schedule and set
  both of its settings" with the Period/Grace arithmetic attached to a sentence that offers
  the Cron-type alternative (`:1484-1494`).
- **WR-25** — still open, verbatim: "not only when one failed outright, which is the one
  failure class FOMO's own error handling cannot report itself" (`:1473-1474`).
- **WR-26** — still open. Step 6's enumeration still says "the lock and log directories" and
  "(flock, the directories, or email)"; `grep -rn FOMO_STATE_DIR docs/` still returns
  nothing. Now compounded by WR-31.
- **WR-27** — still open. Step 8 still says "either route produces the same line"
  (`:1550-1552`).
- **IN-25** — still open; the change added no sixth copy of the 15/20/35 triple.
- **IN-26** — still open, verbatim at `:1484-1486` ("Name each concept first, giving
  healthchecks.io's spelling in parentheses").
- **IN-27** — still open; `deploy/cron/fomo.crontab.example` is unchanged since `a3e5556`.
- **IN-28** — still open; `` `deploy/cron/fomo.crontab.example` `` (`:1551`) and
  `` `deploy/logrotate/fomo.example` `` (`:1554`) are still single-backticked.

**Still open from iteration 3 (13), in source files unchanged since `a3e5556` and not
re-examined by this incremental pass:** WR-16, WR-17, WR-18, WR-19, WR-20, IN-17, IN-18,
IN-19, IN-20, IN-21, IN-22, IN-23, IN-24. Their iteration-3 records below remain
authoritative. **WR-22 is excluded from the open count** — it carries a recorded acceptance
(2026-09-18, 36-UAT.md Test 2), on the standing condition that `LOGGING`'s root level stays
at `INFO`; `settings.py:200-209` still has `'level': 'INFO'`, so the condition holds. IN-15
and IN-16 are excluded as duplicates of IN-27.

`carried_forward_open: 22` = 9 (iteration 4) + 13 (iteration 3).

---

_Reviewed: 2026-09-18T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Iteration: 5 (incremental review of plan 36-08, gap closure G-36-4, against `a3e5556`)_

---

# Retained: iteration 4 report (unchanged, for history)

The full iteration-4 report follows verbatim, including the iteration-3 report it in turn
retained. Its frontmatter has been removed; the frontmatter at the top of this file
describes iteration 5.


# Phase 36: Code Review Report (iteration 4 — incremental review of plan 36-07, gap closure G-36-1)

**Reviewed:** 2026-09-18T00:00:00Z
**Depth:** deep
**Files Reviewed:** 2 (everything changed since `67e6f68`, the commit iteration 3 was written against — commits `280962b`, `a2f1ee9`, `71cdec2`, `95b08ed`)
**Status:** issues_found

## Summary

Plan 36-07 adds a "create and configure the heartbeat check" step to the runbook's
"Setting it up on a fresh host" subsection (renumbering its steps 1–7 to 1–9), adds two
`sudo` notes, names the LCO/SOAR API-key setting, and re-points the crontab template's
three cross-references at the new subsection. Both files are documentation; the review
treated them as an operator *procedure* and executed/verified every claim they make
against `src/fomo/settings.py`, `solsys_code/management/commands/check_unattended.py`,
`solsys_code/unattended.py`, `docs/conf.py` and `.pre-commit-config.yaml`.

**The secret-hygiene requirement the review was asked to enforce holds inside the two
files.** Neither file contains a real ping URL, UUID, API key, host name or host path: the
runbook's only heartbeat URL is the placeholder `https://hc-ping.com/<uuid>`
(`:1490`), the template still carries `/path/to/venv/bin/python` and
`/path/to/checkout/manage.py`, and the template names every environment variable without
ever giving it a value (D-15). The runbook's new "the `<uuid>` part is the ping token, so
this URL is itself a credential" sentence (`:1491-1493`) is a genuine improvement.

**But the procedure those two files describe leaks credentials by a route neither file
warns about, and the leak has already happened in this working tree.** `docs/conf.py`
sets `autoapi_dirs = ['../src']` with no ignore for `local_settings.py`, so every
`pre-commit` run renders that file — the exact file setup step 2 tells the operator to put
the LCO API key and the mail password in, and where this host has also put the real
heartbeat ping URL — verbatim into generated HTML. See **CR-03**. Both output trees are
`.gitignore`d, so nothing reached git; the exposure is on-disk and on anything that serves
those builds.

**The single most damaging defect is the new API-key sentence itself (CR-02): following it
literally stops Django from starting at all.** `settings.py:436-441` documents, in its own
comment, that `from fomo.local_settings import *` cannot mutate a dict built above it —
and the `NameError` it raises is not caught by the `except ImportError` guard. Reproduced
empirically. The supported name is the flat `LCO_API_KEY`, which is what this host's own
`local_settings.py` and `.planning/codebase/INTEGRATIONS.md` both use. The same sentence
also promises a `FACILITIES['SOAR']['api_key']` route that does not exist in any form
(**WR-23**) — `settings.py` folds `LCO_API_KEY` only.

**The new heartbeat step is substantively right but states the alert arithmetic in a way
that is wrong for the Cron-type option it offers in the same breath** (**WR-24**), and its
opening sentence's relative clause attaches to the wrong failure class, asserting the
opposite of the truth about what FOMO can report itself (**WR-25**).

**On the specific question of duplicated alert-window reasoning:** the *reasoning* is
correctly delegated ("see 'The two failure signals' below for why these numbers…"), but
the *values and the formula* are restated, which is where WR-24's inaccuracy entered. The
15/20/35-minute triple now appears in five places across the two files (**IN-25**).

**RST validity: clean.** A `docutils` parse of the whole runbook at `report_level=1`
produces no structural message — no enumerated-list, indentation or block-quote warning
from the renumbered 1–9 list or the new multi-paragraph step 3. Only the expected
Sphinx-role notices (`:doc:`, `:ref:`) appear. No document references the step numbers
this change shifted (`grep -niE "step [0-9]"` across `docs/` finds nothing), so the
renumbering is safe. The template's comment-only edits do not disturb
`test_check_unattended.py`'s two template-agreement tests, which key off the `*/15` line
(unchanged) and a token list (all still present).

**Three iteration-3 findings are still open in these two files, and this change edited the
very lines two of them name** — see **WR-26** (WR-21) and **IN-27** (IN-15/IN-16).

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-02: Setup step 2's new API-key sentence tells the operator to write a line that makes Django refuse to start

**File:** `docs/runbooks/telescope_runs_calendar.rst:1458-1462`
**Issue:** The sentence added by `a2f1ee9` reads:

> Put the real ``EMAIL_BACKEND`` (and its ``EMAIL_HOST_*`` settings) and the LCO/SOAR API
> key in this host's ``local_settings.py`` … The API key setting is nested:
> ``FACILITIES['LCO']['api_key']`` and ``FACILITIES['SOAR']['api_key']``.

`local_settings.py` is imported as its own module (`src/fomo/settings.py:431-434`), so
`FACILITIES` is not in its namespace. Writing `FACILITIES['LCO']['api_key'] = '…'` there
raises `NameError`, and the guard around the import catches `ImportError` only —
`settings.py` itself is what `settings.py:436-439` already spells out:

```python
# `from fomo.local_settings import *` executes that module in its own namespace, so it can only
# ASSIGN new settings -- it cannot mutate ones already built above (FACILITIES['LCO']['api_key']
# = ... there raises NameError, which the ImportError guard does not catch). Secrets that belong
# inside an existing dict therefore arrive as flat names and are folded in here.
if 'LCO_API_KEY' in globals():
    FACILITIES['LCO']['api_key'] = LCO_API_KEY  # noqa: F405
```

Reproduced in isolation (same module shape, same guard):

```
  File ".../local_settings.py", line 1, in <module>
    FACILITIES['LCO']['api_key'] = 'abc'
NameError: name 'FACILITIES' is not defined
```

The blast radius is the whole deployment, not just the cron path: the settings module
fails to import, so gunicorn/uWSGI, every `manage.py` command, the `run_unattended` tick
and `check_unattended` itself all die before running — and the failure email that would
otherwise report it cannot be sent either. This is also the *only* place in `docs/` that
documents the API key at all (`grep -rn "api_key\|API key" docs/*.rst` finds nothing
else), so there is no competing correct instruction; and it contradicts both this host's
own working `local_settings.py` (which sets `LCO_API_KEY`) and
`.planning/codebase/INTEGRATIONS.md:12,178,326`.
**Fix:** name the flat setting the fold actually reads, and say why it is flat:

```rst
2. Put the real ``EMAIL_BACKEND`` (and its ``EMAIL_HOST_*`` settings) and the LCO/SOAR
   API key in this host's ``local_settings.py`` -- never in the crontab line, never in
   an environment variable, and never committed to git. The API key goes in as the
   **flat** name ``LCO_API_KEY = '...'``: ``local_settings.py`` is imported into its own
   namespace, so assigning into ``FACILITIES[...]`` there raises ``NameError`` and stops
   Django from starting. ``settings.py`` folds ``LCO_API_KEY`` into
   ``FACILITIES['LCO']['api_key']`` for you.
```

(and see **WR-23** for the SOAR half of the same sentence).

### CR-03: The credential home this procedure mandates is rendered verbatim into generated HTML by the project's own docs build — the real ping URL, API key and mail password are in this working tree's build output now

**File:** `docs/runbooks/telescope_runs_calendar.rst:1458-1462` and `:1488-1498`
(procedure); root cause `docs/conf.py:62` (`autoapi_dirs = ['../src']`, `autoapi_ignore`
does not exclude `local_settings.py`); triggered by `.pre-commit-config.yaml:67-89`
**Issue:** Step 2 puts the LCO API key and the mail credentials in
`src/fomo/local_settings.py`; step 5 recommends the same file for `FOMO_BASE_URL`; step 4
tells the operator to export the heartbeat ping URL "exactly as copied" and calls it a
credential that "must never go into a committed file (D-15)". That assurance is scoped to
*committed* files, and it is the wrong boundary: `sphinx-autoapi` scans `../src`, and
`sphinx.ext.viewcode`'s module pages reproduce the source. The `sphinx-build` pre-commit
hook therefore writes the real values into `_readthedocs/html/` on every commit, and a
manual build writes them into `docs/_build/html/`. Both are present in this working tree
right now:

```
_readthedocs/html/autoapi/fomo/local_settings/index.html
_readthedocs/html/_modules/fomo/local_settings.html
_readthedocs/html/_sources/autoapi/fomo/local_settings/index.rst.txt
docs/_build/html/... (same three)
```

Those pages render this host's live `FOMO_HEARTBEAT_URL` (a real `hc-ping.com` UUID —
deliberately not quoted here, since this report is committed), and `local_settings.py`
also defines `LCO_API_KEY`, `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD`. `_readthedocs/` and
`docs/_build/` are both `.gitignore`d (`.gitignore:76-77`), so nothing has reached git,
and ReadTheDocs builds from a checkout with no `local_settings.py`, so the published site
is unaffected — but any operator who builds and *serves* HTML from a production checkout
(the only reason to build HTML) publishes the LCO API key, the SMTP password and the ping
token. Iteration 3's credential-hygiene scan looked at the twelve changed source files
only, which is why this never surfaced.
**Fix:** three parts, in order of urgency:

1. Exclude the file from autoapi in `docs/conf.py`:
   ```python
   autoapi_ignore = ['*/__main__.py', '*/_version.py', '*/local_settings.py']
   ```
2. Treat the ping token currently rendered in `_readthedocs/html/` and
   `docs/_build/html/` as exposed: delete both trees and rotate the check (create a new
   check, re-export `FOMO_HEARTBEAT_URL`) if either build was ever served, copied or
   shared.
3. Correct the runbook's own boundary claim in step 3/4 — "must never go into a committed
   file" becomes "must never go into a committed file, and note that `local_settings.py`
   is rendered into the HTML docs unless `autoapi_ignore` excludes it, so never serve a
   docs build produced from a configured host".

## Warnings

### WR-23: Step 2 documents a `FACILITIES['SOAR']['api_key']` route that exists nowhere — SOAR cannot be given a key by any supported mechanism

**File:** `docs/runbooks/telescope_runs_calendar.rst:1461-1462`; cf.
`src/fomo/settings.py:244-247`, `:440-441`
**Issue:** Beyond CR-02's `NameError`, the SOAR half of the sentence has no correct form
at all. `settings.py` folds exactly one flat name — `LCO_API_KEY` — into
`FACILITIES['LCO']['api_key']`. There is no `SOAR_API_KEY` fold anywhere in the repo
(`grep -rn "SOAR_API_KEY" --include=*.py` finds only planning prose deciding *not* to add
one), so `FACILITIES['SOAR']['api_key']` stays `''` no matter what the operator writes in
`local_settings.py`. An operator following step 2 will believe SOAR is authenticated when
the unattended `status_refresh` step is in fact calling the portal with an empty key.
**Fix:** either (a) document reality — one key, `LCO_API_KEY`, and note that SOAR
authenticates against the same LCO Observation Portal credentials (settings.py:240-243),
which means `FACILITIES['SOAR']['api_key']` is a separate, currently unfilled slot; or
(b) close the gap in code by extending the fold, and then document the flat name:

```python
if 'LCO_API_KEY' in globals():
    FACILITIES['LCO']['api_key'] = LCO_API_KEY  # noqa: F405
    FACILITIES['SOAR']['api_key'] = LCO_API_KEY  # SOAR shares LCO portal credentials (D-05)
```

(b) is the one that makes the runbook's current sentence true; whichever is chosen, the
runbook must match it.

### WR-24: The new step 3 states the alert arithmetic as if it applied to the Cron-type check it offers in the same sentence, and calls Period+Grace "both of its settings"

**File:** `docs/runbooks/telescope_runs_calendar.rst:1476-1486`; cf. the authoritative
paragraph at `:1588-1591`
**Issue:** Step 3 says "Create ONE check for this schedule and set **both of its
settings**", then gives Period = 15 min "-- or, as the drift-free alternative, a Cron-type
check carrying the same ``*/15 * * * *`` expression … -- and the grace time (``Grace``) =
about 20 minutes. **The service alerts at last ping + expected interval + grace**, so with
these values a stopped schedule alerts about 35 minutes after the last successful ping."
The later, authoritative paragraph is explicit that the Cron alternative "pegs lateness to
the wall-clock slot instead of to the last ping" (`:1590-1591`), i.e. it alerts at
*next scheduled slot + grace*, and a Cron-type check has no Period field at all — so on
the very route step 3 recommends as "drift-free", both "both of its settings" and the
stated formula are wrong. An operator who configures the Cron route and then reasons from
step 3's arithmetic will mis-predict when (and from what baseline) an alert fires.
**Fix:** attach the formula to the route it belongs to, and drop "both" when offering a
one-setting alternative:

```rst
   Create ONE check for this schedule. Name each concept with healthchecks.io's spelling
   in parentheses: set the expected interval between pings (``Period``) to 15 minutes,
   matching this cron schedule, and the grace time (``Grace``) to about 20 minutes. A
   Simple check of that shape alerts at last ping + interval + grace -- about 35 minutes
   after the last successful ping. The drift-free alternative is a Cron-type check
   carrying the same ``*/15 * * * *`` expression the crontab line uses, with the same
   grace; it has no interval to set and alerts at the missed slot + grace instead. See
   "The two failure signals" below for why these numbers …
```

### WR-25: Step 3's opening sentence attributes "the one failure class FOMO's own error handling cannot report itself" to outright failure — the opposite of the truth

**File:** `docs/runbooks/telescope_runs_calendar.rst:1463-1466`
**Issue:** The sentence reads:

> This is a dead-man's switch on a third-party service that alerts when a tick never ran
> at all or hung partway through -- not only when one failed outright, which is the one
> failure class FOMO's own error handling cannot report itself.

The `which` clause attaches to the nearest noun phrase, "one failed outright" — asserting
that an outright failure is what FOMO cannot report. The code says the reverse: an
outright failure is exactly what `run_tick()` reports, via `_send_notification()` and the
`/<exit-code>` ping (`unattended.py:630-661`); the class FOMO cannot report is the tick
that never started or hung, because the process that would have mailed never reached the
mail call. Getting this backwards is not cosmetic — it is the whole justification for
setting the check up at all, in the step whose job is to convince the operator to do so.
**Fix:** move the clause to its antecedent:

```rst
   This is a dead-man's switch on a third-party service: it catches a tick that never ran
   at all or hung partway through -- the one failure class FOMO's own error handling
   cannot report itself, because the process that would have mailed never got there. A
   tick that runs and fails is already covered by the failure email.
```

### WR-26: WR-21 (iteration 3) is still open, and this change edited the exact sentence it names without closing it

**File:** `docs/runbooks/telescope_runs_calendar.rst:1451-1456` (step 1, edited by
`a2f1ee9`) and `:1516-1530` (step 6's enumeration); cf.
`solsys_code/management/commands/check_unattended.py:66-96`, `:177-191`, `:411-418`
**Issue:** Iteration 3's WR-21 recorded two drifts: step 1 says "Create the **two**
directories" while `check_state_dir()` makes `FOMO_STATE_DIR` a third *hard* check, and
the step-6 enumeration lists neither `FOMO_STATE_DIR` nor `check_flock()`'s `-E`/util-linux
2.27 probe, and still summarises the hard set as "(flock, the directories, or email)".
Plan 36-07 rewrote step 1's first sentence (to add the `sudo` note) and renumbered step 4
to step 6, so both passages were in hand, and both were left stale. `grep -rn
FOMO_STATE_DIR docs/` still returns nothing. A host that points `FOMO_STATE_DIR` somewhere
other than its `FOMO_LOCK_DIR` default (`settings.py:423-425`) therefore gets a hard
preflight failure for a directory the runbook never mentioned.
**Fix:** as WR-21 specified — in step 1 add "``FOMO_STATE_DIR`` defaults to
``FOMO_LOCK_DIR``; create it separately only if this host points it elsewhere"; in step
6's enumeration replace "the lock and log directories" with "the lock, log and
suppression-state directories" and "whether ``flock`` is on ``PATH``" with "whether
``flock`` is on ``PATH`` *and* new enough to support ``-E`` (util-linux 2.27+), which the
cron line's skip detection needs".

### WR-27: Both files claim the two install routes produce the same cron line; the template hardcodes a third host-specific path (`/usr/bin/flock`) that its own header never tells you to replace

**File:** `docs/runbooks/telescope_runs_calendar.rst:1542-1545` (step 8),
`deploy/cron/fomo.crontab.example:3-11` and `:56`; cf.
`solsys_code/management/commands/check_unattended.py:316-320`, `:66-96`
**Issue:** Step 8 says starting from the template "and replac[ing] its two placeholder
paths by hand … either route produces the same line", and the template header says
`check_unattended` "prints this same line with the real resolved interpreter and manage.py
paths already filled in". Neither is true in general. `cron_line()` substitutes five
resolved values — interpreter, `manage.py`, **`shutil.which('flock')`**, `FOMO_LOCK_DIR`
and `FOMO_LOG_FILE` — and its own comment says the template's literal is a placeholder:

```python
    # WR-05 (36-REVIEW.md): resolve the real `flock` path the same way check_flock()
    # already verified it -- the committed template's hardcoded '/usr/bin/flock' is
    # only a placeholder for a host with a non-merged-/usr layout, a venv-provided
    # util-linux, or a container image that keeps it in /bin only.
```

The template's header, by contrast, names exactly two placeholders ("replace BOTH
placeholder paths below") and lists only the interpreter and `manage.py`. On a host whose
`flock` is not at `/usr/bin/flock`, or that overrides `FOMO_LOCK_DIR`/`FOMO_LOG_FILE`, the
template route installs a line whose `sh` lookup fails (exit 127 — not 99, so no
"lock held" line is written either) and every tick is a silent no-op. That is precisely
the CR-01/WR-11 failure class the phase has already been bitten by twice.
**Fix:** in the template header, add `/usr/bin/flock` to the list of values to check
("replace the two placeholder paths, and confirm `flock` really is at `/usr/bin/flock` —
`command -v flock` — or substitute the real path"); in runbook step 8, replace "either
route produces the same line" with "the printed line is authoritative: it carries this
host's resolved `flock`, lock-file and log-file paths as well as the interpreter and
`manage.py`, which the template can only guess at".

## Info

### IN-25: The 15/20/35-minute alert window is now restated in five places across the two files

**File:** `docs/runbooks/telescope_runs_calendar.rst:1476-1486` (new), `:1588-1604`
(authoritative), `:1620-1623`, `:2131-2144`; `deploy/cron/fomo.crontab.example:34-38`
**Issue:** Plan 36-07 correctly delegates the *reasoning* ("see 'The two failure signals'
below for why these numbers, why the grace must not be shrunk, and what the interval's
default does"), but restates the two input values and the derived 35-minute figure in
full, making five copies in these two files. WR-24 is what that duplication already cost:
the copy drifted from the original in its treatment of the Cron-type option. All five
copies presently agree on the numbers themselves.
**Fix:** in step 3, keep the two values (an operator configuring the check needs them in
hand) and drop the derived arithmetic sentence, leaving the existing pointer to carry it —
the arithmetic is the part that has to stay in exactly one place.

### IN-26: "Name each concept first, giving healthchecks.io's spelling in parentheses" is an instruction to the doc's author, not to the operator

**File:** `docs/runbooks/telescope_runs_calendar.rst:1476-1477` (and the pre-existing
"Name the concept first:" at `:1584-1585`)
**Issue:** Both sentences are drafting directives that survived into operator-facing
prose. An operator reading "Name each concept first" in a numbered setup step reasonably
wonders what they are meant to name, and where. Every other step in this subsection is a
plain imperative aimed at the reader.
**Fix:** delete the directive and keep only its product, e.g. "Set the expected interval
between pings (healthchecks.io calls this ``Period``) to 15 minutes…".

### IN-27: IN-15 and IN-16 are still open in the template, and the change rewrote one of the two IN-16 sentences while preserving its stale parenthetical

**File:** `deploy/cron/fomo.crontab.example:32-33` (IN-15), `:9-11` and `:62-65` (IN-16)
**Issue:** (1) IN-15: line 32-33 still describes "the ``[ $? -eq 99 ] && echo ... skipped``
tail below", while line 56 reads `[ $rc -eq 99 ]` and lines 52-55 correctly describe
`[ $? -eq 99 ]` as the superseded form — the file still says both. (2) IN-16: line 9-11
still calls `check_unattended` "(a later plan in this phase)", and line 62-65 — a sentence
this very change rewrote, in `a2f1ee9` — still calls `deploy/logrotate/fomo.example` "(a
later plan in this phase)". Both shipped; `deploy/logrotate/fomo.example` is on disk.
`grep -rn "later plan in this phase"` outside `.planning/` still finds only these two.
**Fix:** as previously specified — line 32 becomes `` `[ $rc -eq 99 ] && echo ...
skipped` ``, and both parentheticals are dropped.

### IN-28: Step 8 and step 9 mark the two deploy-file paths with single backticks, which Sphinx renders as italic title references rather than literals

**File:** `docs/runbooks/telescope_runs_calendar.rst:1543` and `:1546`
**Issue:** `` `deploy/cron/fomo.crontab.example` `` and `` `deploy/logrotate/fomo.example` ``
use one backtick. `docs/conf.py` sets no `default_role`, so Sphinx's default
(`title-reference`) applies and both render italic, unlike every other path in the
subsection (``` ``/var/lock/fomo`` ```, ``` ``local_settings.py`` ```, ``` ``/etc/logrotate.d/fomo`` ```).
Pre-existing wording, but both lines were rewritten by this change (the renumbering) and
step 9's sentence gained text.
**Fix:** use double backticks on both.

## Security review (explicitly requested scope)

| Check | Result |
|---|---|
| Real heartbeat ping URL / UUID in either file | **None.** Only `https://hc-ping.com/<uuid>` (`:1490`) |
| Real API key, token or password in either file | **None.** Environment variables are named, never valued (`fomo.crontab.example:13-27`) |
| Real host path or host name in the template | **None.** `/path/to/venv/bin/python`, `/path/to/checkout/manage.py` intact; `/var/lock/fomo`, `/var/log/fomo` are the shipped defaults (`settings.py:422`, `:429`), not host-specific |
| Secret steered into a process argument vector | **No.** Template `:25-27` still forbids it (T-36-02) |
| Secret steered into a file the toolchain publishes | **YES — see CR-03** (`local_settings.py` → autoapi/viewcode → `_readthedocs/html/`, `docs/_build/html/`) |

## Previously reported, still open

- **WR-21** (iteration 3) — re-raised as **WR-26** above; the plan-36-07 diff touched both
  passages and closed neither.
- **IN-15**, **IN-16** (iteration 3) — re-raised as **IN-27** above; IN-16's second
  instance sits inside a sentence this change rewrote.
- **WR-16, WR-17, WR-18, WR-19, WR-20, WR-22, IN-17…IN-24** (iteration 3) — not re-opened
  by this incremental review; the code files they name are unchanged since `67e6f68`.
  Their records below remain authoritative, including **WR-22's recorded acceptance**
  (2026-09-18, 36-UAT.md Test 2).

---

_Reviewed: 2026-09-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Iteration: 4 (incremental review of plan 36-07, gap closure G-36-1, against `67e6f68`)_

---

# Retained: iteration 3 report (unchanged, for history)

The full iteration-3 report follows verbatim. Its frontmatter has been removed; the
frontmatter at the top of this file describes iteration 4.

# Phase 36: Code Review Report (iteration 3 — re-review after the iteration-2 fix pass and the 36-06 gap closure)

**Reviewed:** 2026-09-17T23:55:00Z
**Depth:** deep
**Files Reviewed:** 12 (everything changed since `fe719d6d`, the commit iteration 2 was written against)
**Status:** issues_found

## Summary

This is an incremental re-review of the 21 fixes recorded in `36-REVIEW-FIX.md`
(iteration 2) plus gap-closure plan 36-06 (`12c51c6`, `f075a7f`, `1f3bbac`), which
corrected the heartbeat alert-window guidance.

**Verification of the prior 21 fixes.** All were checked against the real code rather than
against the fix report's prose. Twenty are genuinely and completely fixed. The two I
re-derived empirically:

- **WR-09 (inverted cron-line exit status) is fixed.** Running the committed line's exact
  shape in a real `sh` now yields `0` on a healthy tick, `1` on a failing tick, and `99`
  on a contended one — the inversion is gone. But the *semantics* of the new third case
  contradict the runner's own documented contract; see **WR-16**.
- **WR-10 (`load_state()` raising on a mixed-type `failing_steps`) is fixed** by the
  `isinstance(step, str)` filter at `unattended.py:434`, which sits before the `sorted()`
  at `:445`.

**One prior finding is only partially fixed and is carried forward.** WR-15's own
recommended remedies were "(a) suppress further mail when the state cannot be persisted"
**or** "(b) add a `check_state_dir()` preflight", with the review explicitly noting that
"(a) is the one that survives a directory that becomes unwritable after setup". The fix
pass implemented (b) only, and `36-REVIEW-FIX.md` records WR-15 as fixed. The runtime
email-storm loop is unchanged — see **WR-17**.

**The 36-06 gap closure is substantively correct.** The runbook's rewritten "Heartbeat."
paragraph now names both knobs, states the alert arithmetic, explains why the grace time
also bounds the `/start`→completion gap (accurate for healthchecks.io's start-signal
semantics), and adds a dedicated troubleshooting entry. Its propagation into
`check_unattended.py` is thinner than into the docs (see **IN-20**), and the runbook's
own description of what the preflight reports was not refreshed for the two *other*
checks the fix pass added (see **WR-21**).

**Credential hygiene (D-15/D-17/SCHED-10) mostly holds.** A scan of all twelve files for
`hc-ping`, healthchecks-shaped URLs, and bare UUIDs finds nothing; every fixture literal is
an obvious `example`/`FAKE-` placeholder; `check_heartbeat()` still prints set/unset only
and the new test pins `assertNotIn(_FAKE_HEARTBEAT_URL, stdout)`; the regenerated notebook
carries no absolute host path and no worktree path. `check_unattended` prints set/unset
only, never a value, so D-15 is satisfied. The one hole is **WR-22**: two `logger.debug()`
call sites interpolate a raw `str(exc)` from a portal call, which D-17 forbids — latent
only because the shipped `LOGGING` config drops `DEBUG`. No injection, path-traversal, or
deserialization defect was found. `ruff`'s 120-column limit is respected in every changed
Python file.

**Where the remaining defects cluster.** Four of the seven warnings are consequences of the
fix pass itself: a contract contradiction the WR-09 fix introduced (WR-16), the half-fix
of WR-15 (WR-17), an IN-02 fix whose output the project's own `LOGGING` config discards
(WR-18), and a `subprocess` call the WR-11 fix added without the error handling the same
commit gave `_owner_mode()` (WR-20). The remaining three are pre-existing gaps the earlier
iterations did not catch (WR-19, WR-21, WR-22). **WR-18 and WR-22 must be fixed together**:
WR-18's obvious remedy (raise the verbosity of the unattended log) is exactly what would
activate WR-22's leak.

## Narrative Findings (AI reviewer)

### Verification of the prior 21 fixes

| Prior finding | Verdict | Evidence |
|---|---|---|
| WR-09 inverted cron-line exit status | **Fixed** (but see **WR-16**) | Live `sh` probe: healthy `0`, failing `1`, contended `99`; `check_unattended.py:325-330`, `fomo.crontab.example:52`, `test_check_unattended.py` `test_line_ends_with_an_explicit_exit_of_the_captured_status` |
| WR-10 `load_state()` raises on mixed types | **Fixed** | `unattended.py:434` filters before `sorted()` at `:445`; `test_mixed_type_failing_steps_are_coerced_not_raised` |
| WR-11 `check_flock()` didn't prove `-E` | **Fixed** (but see **WR-20**) | `check_unattended.py:85-95` |
| WR-12 runbook flag docs | **Fixed** | `telescope_runs_calendar.rst:388-391`, `:1779-1780` |
| WR-13 `FOMO_BASE_URL` scoped to cron only | **Fixed** | `telescope_runs_calendar.rst:1463-1473`, `fomo.crontab.example:16-21`, `docs/installation.rst:110-117` |
| WR-14 unreachable stale-lock remedy | **Fixed** | `telescope_runs_calendar.rst:2045-2057` |
| WR-15 unwritable `FOMO_STATE_DIR` mail storm | **Partly fixed** — see **WR-17** | `check_unattended.py:177-191` adds the setup-time check; `unattended.py:646-660` is unchanged |
| IN-01 heartbeat ignored HTTP status | **Fixed** | `unattended.py:156` `raise_for_status()` |
| IN-02 skip reasons discarded | **Partly fixed** — see **WR-18** | `unattended.py:374-381` captures them, but at `DEBUG` |
| IN-03 non-atomic state write | **Fixed** | `unattended.py:473-483`; mkstemp + `chmod 0o600` + `os.replace` |
| IN-04 stale `run_unattended` docstring | **Fixed** | `run_unattended.py:20-25` (but see **IN-16** for two siblings missed) |
| IN-05 outage read as "failed 1" / dangling `classes:` | **Fixed** | `unattended.py:226`, `:281-290` |
| IN-06 unknown `only_step` silent no-op | **Fixed** | `unattended.py:605-606` |
| IN-07 `notify_staff()` discarded `send_mail()`'s result | **Fixed** | `notifications.py:84-95` |
| IN-08 cron-line shape test | **Fixed** | `test_check_unattended.py:263`, new token-for-token test |
| IN-09 stale `check_unattended` docstrings | **Fixed** | `check_unattended.py:15-19`, `:376-381`, `test_check_unattended.py:3` |
| IN-10 stale runbook `flock -n` passages | **Fixed** | `telescope_runs_calendar.rst:1587-1600` |
| IN-11 notebook prose-only | **Fixed** | Cell 18 is a real executed cell (`execution_count` 1..12 is sequential across all 12 code cells, so the notebook was genuinely re-run) |
| IN-12 guard ordering/truthiness | **Fixed** | `backfill_lco_observations.py:879-899` (`is not None`, above username resolution) |
| IN-13 duplicated watched-proposal loop | **Fixed** | `backfill_lco_observations.py:697-769`, both callers |
| IN-14 `None` path settings, `_owner_mode()` | **Fixed** | `unattended.py:121`, `:417`, `:466`, `:551`; `check_unattended.py:108-112` |

## Warnings

### WR-16: The cron line now reports a benign lock-contended skip as exit 99 — a failure to any supervisor — and both the docstring and the crontab comment misattribute that code to `run_unattended`

**File:** `solsys_code/management/commands/check_unattended.py:295-300` and `:325-330`,
`deploy/cron/fomo.crontab.example:47-52`; cf. `solsys_code/unattended.py:88-91`, `:590-594`,
`:665-668`
**Issue:** The WR-09 fix ends the line with `exit $rc`. Verified in a real `sh` against a
real `flock`:

```
healthy tick   -> exit=0
failing tick   -> exit=1
lock contended -> exit=99
```

That is correct for the first two cases and fixes the inversion. The third case is a new
contract violation. `run_tick()`'s own docstring is explicit that contention is *not* a
failure — "A contended whole-run lock is NOT a failure -- it returns ``exit_code=0`` with
no results ... the heartbeat (D-12) is the structural backstop" (`unattended.py:590-594`),
and `TickResult.exit_code` documents "0 on a healthy tick (including a lock-contended
skip)" (`:88-91`). The cron line now overrides that decision from the outside: any
supervisor reading the line's status — cron's own syslog `CMD exit status`, a systemd
timer if this is migrated, an `OnFailure=` hook, a `run-parts` harness, or a monitoring
wrapper — sees a **non-zero status on a routine tick overlap**, which the runbook itself
calls normal ("One occurrence is normal (an overrunning tick colliding with the next
scheduled one)", `telescope_runs_calendar.rst:1600`). The previous iteration's WR-09
complained that a healthy tick was indistinguishable from a failing one; this iteration's
line makes a *healthy skip* indistinguishable from a failing tick.

Both prose claims about the new line are also wrong in the same direction:

- `check_unattended.py:299-300`: "the line's own status is always `run_unattended`'s (0
  healthy, 1 failing, 99 skipped)". `run_unattended` never exits 99 — it exits **0** on
  contention. 99 is `flock`'s own `-E` code and is a status `run_unattended` cannot
  produce.
- `fomo.crontab.example:48-49`: the same sentence, same error.

**Fix:** decide which contract wins and make all three surfaces agree. The runner's own
contract (contention is benign, the heartbeat is the backstop) is the one the whole phase
is built on, so normalize the skip to 0 after the log line is written, in `cron_line()`
and the committed template together:

```
*/15 * * * * <flock> -n -E 99 <lock> <python> <manage.py> run_unattended >> <log> 2>&1; rc=$?; \
  [ $rc -eq 99 ] && { echo "$(date -Is) run_unattended skipped: lock held" >> <log>; rc=0; }; exit $rc
```

and correct both prose claims to "the line's own status is `run_unattended`'s (0 healthy,
1 failing); a lock-held skip is normalized to 0 after the skip line is logged, matching
`run_tick()`'s own decision that contention is not a failure". If the project instead
*wants* 99 surfaced, say so explicitly in both places ("99 means the tick was skipped —
benign in isolation, investigate only if repeated") and update `run_tick()`'s docstring to
note the divergence. Either way, add a `TestCronLine` case pinning whichever choice is made.

### WR-17: WR-15's runtime email storm is still open — only the setup-time preflight was added

**File:** `solsys_code/unattended.py:646-660`, `solsys_code/management/commands/check_unattended.py:177-191`
**Issue:** `36-REVIEW-FIX.md` records WR-15 as fixed, citing the new `check_state_dir()`.
That closes the *setup-time* case only. WR-15's own text named two remedies and said which
one mattered: "(a) is the one that survives a directory that becomes unwritable after
setup." (a) was not implemented, and the runtime path is byte-for-byte unchanged. Traced
against the current code with a state directory that becomes unwritable or full **after**
the preflight passed (a full `/var/lock` tmpfs is the realistic trigger; `FOMO_STATE_DIR`
defaults to `FOMO_LOCK_DIR`, `settings.py:418`):

1. `load_state()` (`:647`) → its `except (OSError, ValueError)` at `:421` returns
   `{'failing_steps': [], 'notified_at': None}`.
2. `decide_notification()` (`:648`) sees an empty previous set and a non-empty current one
   → `'failure'`.
3. `_send_notification()` (`:654`) **sends**.
4. `save_state()` (`:656`) raises `OSError` → caught at `:659`, logged as one
   `unattended notification/state handling raised: OSError` line.
5. Nothing records that staff were told, so step 2 reaches the identical conclusion on the
   next tick.

At the D-04 15-minute cadence that is 96 identical emails per staff address per day, for
as long as one step keeps failing — D-11's suppression rule failing open in the most
visible possible way, and the exact scenario `check_state_dir()`'s own docstring describes
("makes every tick send the same failure email again, forever") without preventing it once
the host is past setup.
**Fix:** implement remedy (a) alongside the preflight — a process-lifetime fallback so an
unpersistable state cannot re-notify:

```python
_state_write_failed = False  # module-level, reset per process


def _persist(failing_steps, when):
    global _state_write_failed
    try:
        save_state(failing_steps, when)
    except OSError:
        _state_write_failed = True
        logger.error(
            'could not persist unattended suppression state to %s -- further notifications '
            'for this failing set are suppressed for this process',
            Path(settings.FOMO_STATE_DIR or settings.FOMO_LOCK_DIR or _DEFAULT_LOCK_DIR) / _STATE_FILENAME,
        )
```

and skip `_send_notification()` when `_state_write_failed` is set and the decision is
`'failure'`/`'reminder'`. Add a `TestStateFileRobustness` case patching `save_state` to
raise `OSError` across two consecutive `run_tick()` calls and asserting
`len(mail.outbox) == 1`, not 2. Either way, un-mark WR-15 in `36-REVIEW-FIX.md`.

### WR-18: IN-02's "skip reasons are no longer discarded" fix logs at `DEBUG`, which this project's own `LOGGING` config drops — the reasons are still discarded in production

**File:** `solsys_code/unattended.py:374-381`; cf. `src/fomo/settings.py:193-202`,
`solsys_code/tests/test_unattended.py` `test_per_request_skip_reasons_are_logged_not_discarded`
**Issue:** The IN-02 fix captures `sweep_proposal()`'s per-request skip lines into
`io.StringIO()` sinks and re-emits them with `logger.debug('discovery %s: %s', ...)`
(`:381`). The project ships exactly one logging configuration, and its root logger is
pinned to `INFO`:

```python
LOGGING = {
    ...
    'loggers': {'': {'handlers': ['console'], 'level': 'INFO'}},
}
```

`solsys_code.unattended` declares no logger of its own in that config, so it inherits the
root level. Every one of those `DEBUG` records is therefore filtered out before it reaches
the `StreamHandler` whose stderr the crontab line redirects into
`/var/log/fomo/unattended.log`. Net effect on a real deployment: identical to before the
fix — the operator still sees only the bare `swept: N, failed: M` summary, and the skip
reasons are still gone. The new test passes only because
`self.assertLogs('solsys_code.unattended', level='DEBUG')` temporarily installs its own
handler at `DEBUG`, which is exactly the condition that does not hold at runtime; it
therefore verifies that `logger.debug()` was *called*, not that IN-02's stated outcome
("They must now reach the log") is achieved.
**Fix:** promote this one re-emission to `INFO`, which is the level the same function
already uses for its sibling operator-facing line at `:383` (`'0 watched proposals,
nothing to discover'`), and which is what the redirected log actually captures:

```python
                if captured_text:
                    logger.info('discovery %s: %s', sink_name, captured_text)
```

Skip reasons are structural (`Skipping request <id>: no configuration with a named
target.`), not credentials, so `INFO` is consistent with D-17 — see IN-18 for the wording.
Then assert the level in the test (`assertLogs(..., level='INFO')`) so a future downgrade
back to `DEBUG` fails. **Do not implement this by lowering the global log level to `DEBUG`
instead** — that would activate **WR-22**.

### WR-19: `check_email()`'s backend check rejects only the console backend, so `dummy`, `locmem` and `filebased` pass a check whose own docstring promises "the email backend can actually deliver"

**File:** `solsys_code/management/commands/check_unattended.py:194-217`
**Issue:** The check is a single equality test:

```python
    is_console = backend == 'django.core.mail.backends.console.EmailBackend'
```

Django ships four other non-delivering backends. `django.core.mail.backends.dummy.EmailBackend`
is the canonical "turn email off" idiom and is a realistic production setting on a host
where someone wanted to silence mail temporarily; `locmem` is what a half-finished
`local_settings.py` copied from a test config carries; `filebased` writes to a directory
nobody reads. All three pass this hard check and are reported `[ok] EMAIL_BACKEND` with
their own dotted path as the detail, and `--send-test-email` also "succeeds" against all
three (`dummy.EmailBackend.send_messages()` returns `len(email_messages)`, so
`notify_staff()` returns True and `_send_test_email()` reports
`sent one test email to staff recipients`). The operator then installs the crontab line
believing D-11's primary alert channel is proven, and no failure notice will ever arrive.
This is a strictly worse outcome than the console backend the check does catch, because
`--send-test-email` actively confirms it.
**Fix:** reject the whole non-delivering set by name rather than one member of it:

```python
_NON_DELIVERING_BACKENDS = {
    'django.core.mail.backends.console.EmailBackend',
    'django.core.mail.backends.dummy.EmailBackend',
    'django.core.mail.backends.locmem.EmailBackend',
    'django.core.mail.backends.filebased.EmailBackend',
}
...
    is_non_delivering = backend in _NON_DELIVERING_BACKENDS
```

with a detail naming which one and why it cannot deliver, and add one test per backend.
(A custom third-party backend still passes, which is the right default — the check can
only prove a *known* non-deliverer.)

### WR-20: The `check_flock()` probe the WR-11 fix added can hang or traceback the whole read-only preflight — the exact hardening the same commit gave `_owner_mode()`

**File:** `solsys_code/management/commands/check_unattended.py:85`
**Issue:**

```python
    probe = subprocess.run([path, '--help'], capture_output=True, text=True, check=False)  # noqa: S603
```

has neither a `timeout=` nor a `try/except OSError`. `shutil.which()` returning a path is
an `os.access(..., X_OK)` test, not a guarantee the `execve` will succeed: a dangling
symlink target, a `noexec` mount, an `ENOEXEC` wrapper script with a bad shebang, an
`ETXTBSY`, or a plain TOCTOU delete between the `which()` at `:77` and the `run()` at
`:85` all raise `OSError`/`PermissionError` out of `check_flock()`. Because `check_flock()`
is the **first** entry in `Command.handle()`'s list (`:411`), that exception aborts the
whole command: the operator loses the other seven check results *and* the printed cron
line, and gets a traceback out of a command whose module docstring opens with "This
command is read-only by construction" and whose whole purpose (`:8-10`) is "naming every
failed hard check in one `CommandError` so a fresh-host operator sees the whole list of
problems in one run". Separately, with no `timeout=` a `flock` binary on a stalled NFS
mount hangs the preflight indefinitely. The same commit range explicitly hardened
`_owner_mode()`'s `stat()` against precisely this class ("an unavailable stat should
degrade to a reported detail, never an uncaught traceback", `:105-106`); the new
`subprocess.run()` did not get the same treatment.
**Fix:**

```python
    try:
        probe = subprocess.run([path, '--help'], capture_output=True, text=True, check=False, timeout=5)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name='flock',
            ok=False,
            hard=True,
            detail=f'{path} could not be executed to verify -E support: {type(exc).__name__}',
        )
```

(reporting the class name only, matching D-17), and add a test patching `subprocess.run`
with `side_effect=OSError` that asserts a `CommandError` naming `flock` rather than an
`OSError` escaping.

### WR-21: The runbook's description of what `check_unattended` reports was not updated for the two hard checks the same fix pass added

**File:** `docs/runbooks/telescope_runs_calendar.rst:1449-1454` and `:1480-1496`; cf.
`solsys_code/management/commands/check_unattended.py:66-96`, `:177-191`, `:411-418`
**Issue:** The iteration-2 fixes added two hard checks — `FOMO_STATE_DIR` writability
(WR-15) and `flock -E` support (WR-11) — taking the total from six to eight. The
`check_unattended` module docstring and the test module docstring were updated (IN-09);
the runbook, which is the page an operator actually follows, was not:

- Step 4's enumeration (`:1480-1496`) still lists exactly the old set: "whether ``flock``
  is on ``PATH``, whether the lock and log directories exist and are writable, whether the
  email backend can actually deliver and at least one staff user has an email on file,
  whether ``FOMO_HEARTBEAT_URL`` is set ..., whether ``FOMO_BASE_URL`` has been changed
  ..., and whether at least one ``WatchedProposal`` row is active". Neither the state
  directory nor the `-E` probe appears, and the closing sentence still says the hard set is
  "(flock, the directories, or email)" without naming which directories.
- Setup step 1 (`:1449-1454`) still says "Create the **two** directories the schedule below
  assumes exist", listing `/var/lock/fomo` and `/var/log/fomo`. `FOMO_STATE_DIR` is a
  separately settable path (`settings.py:418`) that merely *defaults* to `FOMO_LOCK_DIR`; a
  host that points it elsewhere now gets a hard preflight failure for a directory the
  runbook never told the operator to create, and `grep -n FOMO_STATE_DIR docs/runbooks/`
  returns nothing at all.

CLAUDE.md makes any `docs/runbooks/` page whose documented behavior a change affects part
of the deliverable, not follow-up polish — the same rule WR-12 was raised under, and the
same rule quick task `260726-kdp` is recorded as breaching.
**Fix:** in step 4's enumeration, replace "whether the lock and log directories exist and
are writable" with "whether the lock, log and suppression-state directories exist and are
writable" and "whether ``flock`` is on ``PATH``" with "whether ``flock`` is on ``PATH``
*and* new enough to support ``-E`` (util-linux 2.27+), which the cron line's skip
detection needs"; in step 1, add a sentence that `FOMO_STATE_DIR` defaults to
`FOMO_LOCK_DIR` and only needs creating separately if it has been pointed elsewhere.

### WR-22: Two `logger.debug()` sites interpolate a raw portal-exception message, which D-17 forbids — latent only because the shipped config drops `DEBUG`

**File:** `solsys_code/management/commands/backfill_lco_observations.py:349`,
`solsys_code/unattended.py:191`; cf. `solsys_code/tests/test_unattended.py`
`TestCredentialHygiene`
**Issue:** The phase's D-17 discipline is "only the exception's class name ever reaches the
row, stderr, or the log — never `str(exc)`", and it is honoured at every `warning`/`error`
site in `unattended.py` (`:158`, `:225`, `:236`, `:576`, `:625`, `:660`) and in
`sweep_watched_rows()` (`backfill_lco_observations.py:742`). Two `DEBUG` sites break it:

```python
# backfill_lco_observations.py:346-349 -- the exception is from a live portal call
    try:
        result = facility.get_observation_status(observation_id)
    except Exception as exc:
        logger.debug(f'Observed-block lookup failed for observation_id={observation_id!r}: {exc}')

# unattended.py:188-191 -- `except Exception`, so not necessarily FOMO's own exception class
                except Exception as exc:  # noqa: BLE001 -- FOMO's own reconcile_run(), D-17's
                    logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, exc)
```

`facility.get_observation_status()` is an authenticated LCO Observation Portal call. This
codebase's own credential-hygiene test models exactly what such an exception's message can
carry — `TestCredentialHygiene` constructs
`ImproperCredentialsException(f'portal error key={_FAKE_LCO_API_KEY} url={_FAKE_HEARTBEAT_PING_URL}')`
— so `str(exc)` here can put the LCO API key and a heartbeat ping URL into
`/var/log/fomo/unattended.log`, a file `deploy/logrotate/fomo.example` keeps on disk and
that an operator is told to read first when triaging. Today this is inert because
`settings.LOGGING` pins the root logger to `INFO`; it becomes live the moment anyone sets
`DEBUG` to chase a problem — which is the natural response to WR-18, and which the runbook
does not warn against. The second site's `# noqa` comment claims the exception is "FOMO's
own `reconcile_run()`, D-17's second bucket", but the `except` clause is bare `Exception`,
so anything `reconcile_run()` propagates (including a `requests` error from deeper in the
call chain) is logged with its full message.
**Fix:** use the class name at both sites, matching every other call site in the phase:

```python
        logger.debug('Observed-block lookup failed for observation_id=%r: %s', observation_id, type(exc).__name__)
...
                    logger.debug('reconcile_run() raised for run pk=%s: %s', run.pk, type(exc).__name__)
```

and extend `TestCredentialHygiene` with a case that runs a tick under
`self.assertLogs(level='DEBUG')` and asserts the fake key and ping URL appear nowhere in
the captured output — the current suite only captures at the default level, which is why
this survived two review iterations.
**Disposition (2026-09-18, 36-UAT.md Test 2):** **accepted, not fixed.** The developer recorded
an explicit acceptance that the class-name-only discipline holds only while `settings.LOGGING`
keeps the root logger at `INFO`. The constraint is documented at the point an operator would
change it — a comment directly above `LOGGING` in `src/fomo/settings.py` naming both sites and
the required fix — and in `36-VERIFICATION.md` § Acknowledged Gaps. If the level is ever raised,
WR-22 and WR-18 must be fixed together before the change ships.

## Info

### IN-15: The crontab template's own comment contradicts the command line directly below it

**File:** `deploy/cron/fomo.crontab.example:29`
**Issue:** Line 29 still describes "the ``[ $? -eq 99 ] && echo ... skipped`` tail below",
but line 52 was rewritten by the WR-09 fix and now reads `[ $rc -eq 99 ]`. The WR-09
explanation block at `:47-51` correctly describes `[ $? -eq 99 ]` as the *earlier* form, so
the file now says both that `$?` is what the line uses and that `$?` is what the line no
longer uses.
**Fix:** change `:29` to `` `[ $rc -eq 99 ] && echo ... skipped` ``.

### IN-16: Two stale "a later plan in this phase" references survive in the committed crontab template

**File:** `deploy/cron/fomo.crontab.example:9-11` and `:60`
**Issue:** `:9-11` describes `python manage.py check_unattended` as "(a later plan in this
phase)" and `:60` describes `deploy/logrotate/fomo.example` as "(a later plan in this
phase)". Both shipped — `deploy/logrotate/fomo.example` exists on disk and was reviewed in
iteration 2. This is the same defect IN-04 raised and the fix pass corrected in
`run_unattended.py`; the sweep stopped at the Python file and missed the two instances in
the operator-facing template. `grep -rn "later plan in this phase"` outside `.planning/`
finds only these.
**Fix:** drop both parentheticals.

### IN-17: `sweep_watched_rows()`'s docstring and type hints were invalidated by the IN-02 fix that landed two commits later

**File:** `solsys_code/management/commands/backfill_lco_observations.py:697-714`, `:926-928`
**Issue:** Two drifts, both introduced by the ordering of the fix commits (`3ac974d`
IN-13, then `cf5c7f4` IN-02):
(1) `:712-713` says "``stdout``: forwarded to ``sweep_proposal()``, **unused (``None``) by
the runner**, which has no stdout of its own to write progress lines to". The runner now
passes a live `io.StringIO()` (`unattended.py:374-377`) precisely so it is *not* unused.
(2) the signature annotates `stdout: io.StringIO | None` / `stderr: io.StringIO | None`,
but `Command.handle()` passes `self.stdout`/`self.stderr`, which are Django
`OutputWrapper` instances, not `io.StringIO` — so the annotation is wrong for one of the
two callers it was extracted to serve.
**Fix:** reword (1) to describe the runner's capture-and-log use, and widen (2) to
`typing.TextIO | None` (matching `sweep_proposal()`'s own `Any` parameters at `:424-425`).

### IN-18: The IN-02 comment's D-17 justification overstates what the captured sinks contain

**File:** `solsys_code/unattended.py:366-373`
**Issue:** The comment asserts the captured text holds skip reasons that are "never portal
response content or a credential (D-17)". The credential half is right; the portal half is
not. `sweep_proposal()` writes portal-derived values to both sinks:
`stderr.write(f'Skipping request {observation_id}: ...')` (`backfill_lco_observations.py:519`,
`:529`, `:536`, `:550`) carries request ids from the payload, and the dry-run
`stdout.write(...)` at `:589-593` carries `target_name` and `status` straight from the
RequestGroup JSON. None of that is a credential and none is PII, so the *decision* to log
it is fine — but the stated reason is not the true one, and a future reader relying on
"never portal response content" to widen what gets logged would be relying on something
false.
**Fix:** reword to "these are structural skip reasons carrying only portal identifiers
(request/observation ids, target names, states) — never a credential and never a raw
response body, request URL, or caught exception's message (D-17)".

### IN-19: The `--proposal` guard's message is ungrammatical for the single-flag case, and that wording is now committed in the notebook's executed output

**File:** `solsys_code/management/commands/backfill_lco_observations.py:896-899`;
`docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (cell 18 output)
**Issue:** `f'{", ".join(ignored)} require --proposal; ...'` uses the plural verb
unconditionally. The single-flag case — by far the common one, and the only one the
notebook and the two new tests exercise — reads
`--created-after require --proposal; the watched-list sweep takes its overrides from each
WatchedProposal row.` That exact string is now baked into the notebook's committed output
and is what an operator sees.
**Fix:**

```python
                verb = 'requires' if len(ignored) == 1 else 'require'
                raise CommandError(
                    f'{", ".join(ignored)} {verb} --proposal; the watched-list sweep takes its '
                    'overrides from each WatchedProposal row.'
                )
```

and regenerate the notebook (`jupyter nbconvert --to notebook --execute --inplace`) so the
committed output matches.

### IN-20: The 36-06 heartbeat guidance reached the runbook and the crontab with both knobs, but the preflight with only one

**File:** `solsys_code/management/commands/check_unattended.py:242-246`;
`solsys_code/tests/test_check_unattended.py` `test_set_heartbeat_reminds_about_the_check_period`;
cf. `deploy/cron/fomo.crontab.example:31-35`, `docs/runbooks/telescope_runs_calendar.rst:1548-1569`
**Issue:** 36-06's stated purpose was "correct heartbeat guidance with **both** alert-window
knobs". The runbook and the crontab template both name the expected ping interval
(`Period`, 15 min) *and* the grace time (`Grace`, ~20 min). The `[ok] heartbeat` detail —
the one surface the operator actually executes, and the only one that fires at setup time —
names only `Period`, and the new test pins only `assertIn('Period', stdout)`, so a future
edit that drops the grace half entirely would still pass. An operator who follows the
preflight's reminder alone sets `Period=15` and leaves healthchecks.io's 1-hour default
grace, producing a 75-minute alert window instead of the documented ~35. Separately, the
literal `15 min` in this string is now a third hardcoded copy of the schedule constant
(alongside `cron_line()`'s `*/15` and the template's), with nothing tying them together.
**Fix:** extend the detail to "confirm the check's own expected ping interval (Period) is
15 min, not its 1-day default, and its grace time is about 20 min", assert both substrings
in the test, and derive the "15" from the same source `cron_line()`'s `*/15` uses (a
module constant) so the three cannot drift.

### IN-21: `save_state()`'s temp files leak on a process kill, and nothing ever reaps them

**File:** `solsys_code/unattended.py:473-483`
**Issue:** The IN-03 fix's `except BaseException: tmp_path.unlink()` covers the exception
path only. A `SIGKILL`/OOM kill between `mkstemp()` (`:473`) and `os.replace()` (`:479`) —
the same failure class the fix's own docstring cites for the lock file — leaves a
`.unattended-state.json.<random>.tmp` file in `FOMO_STATE_DIR` forever. `load_state()`
reads only the exact `_STATE_FILENAME`, so there is no correctness impact, but on a host
that OOM-kills ticks the directory (which defaults to `/var/lock/fomo`, often a small
tmpfs) accumulates one file per occurrence with nothing to clean them.
**Fix:** at the top of `save_state()`, unlink any `.{_STATE_FILENAME}.*.tmp` older than a
tick interval, or note the leak in the docstring so an operator knows the files are safe
to delete.

### IN-22: `_DEFAULT_LOCK_DIR`/`_DEFAULT_LOG_FILE` now exist in three places

**File:** `solsys_code/unattended.py:58-59`,
`solsys_code/management/commands/check_unattended.py:42-43`, `src/fomo/settings.py:415`, `:422`
**Issue:** The IN-14 fix duplicated the same two literals into both modules, each with a
near-identical comment saying it "mirrors settings.py's own `os.getenv(..., <default>)`
defaults". Nothing enforces the mirroring, so a change to `settings.py`'s defaults now
silently desynchronizes two fallback paths whose entire purpose is to match it.
**Fix:** export them once — e.g. `solsys_code/unattended.py` as the single owner, imported
by `check_unattended.py` — or add one test asserting
`unattended._DEFAULT_LOCK_DIR == check_unattended._DEFAULT_LOCK_DIR` and that both match
`settings.py`'s literal.

### IN-23: `cron_line()` bakes a `PATH`-resolved binary into a line destined for a service crontab, and the new template test fails opaquely if the template's schedule line is renamed

**File:** `solsys_code/management/commands/check_unattended.py:320`;
`solsys_code/tests/test_check_unattended.py` `test_line_matches_the_committed_template_token_for_token`
**Issue:** Two small ones.
(1) WR-05's `shutil.which('flock')` resolves against the *preflight process's* `PATH` and
the result is printed for the operator to paste into a persistent, scheduled command. An
operator with a stale or user-writable directory early in `PATH` (a conda/venv `bin`, a
`~/bin`) can end up installing a non-system `flock` into a service crontab — a
low-likelihood but persistent outcome, and one the committed template's hardcoded
`/usr/bin/flock` did not have. A one-line sanity note ("resolved outside the usual system
directories — confirm this is the `flock` you want in a crontab") would keep WR-05's
benefit without the silent case.
(2) the new test's `next(...)` has no default, so if the template's `*/15` line is ever
reformatted or the file moved, the test fails with a bare `StopIteration` instead of an
assertion naming the problem; and `Path(__file__).resolve().parents[2]` assumes an editable
checkout layout.
**Fix:** (1) compare `flock_path` against a small allow-list of system directories and add
a note to the detail when it falls outside; (2) `next(..., None)` plus
`self.assertIsNotNone(template_line, f'no */15 line in {template_path}')`.

### IN-24: The regenerated notebook is the only pre-executed notebook with no `kernelspec` metadata

**File:** `docs/notebooks/pre_executed/backfill_lco_observations_demo.ipynb` (top-level
`metadata`)
**Issue:** All seven other notebooks under `docs/notebooks/pre_executed/` carry
`metadata.kernelspec` = `python3`; this one carries `language_info` only. Verified
**pre-existing** — the same metadata was already missing at `fe719d6d`, so the IN-11
regeneration preserved rather than caused it. Flagged because the file is in scope and
because nbsphinx and JupyterLab both use `kernelspec` to pick an interpreter if the
notebook is ever re-executed by a reader or by a future `--execute` build.
**Fix:** add the standard block on the next regeneration:

```json
  "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
```

### Previously reported, still open

- **WR-15** (iteration 2) — recorded as fixed in `36-REVIEW-FIX.md`, but only remedy (b)
  was applied. Carried forward as **WR-17** above.

---

_Reviewed: 2026-09-17T23:55:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
_Iteration: 3 (re-review of the `36-REVIEW-FIX.md` fix pass and the 36-06 gap closure)_
