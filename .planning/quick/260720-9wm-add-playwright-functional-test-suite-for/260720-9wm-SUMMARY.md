---
phase: 260720-9wm
plan: 01
subsystem: testing
tags: [playwright, django, staticliveservertestcase, crispy-forms, bootstrap5, ci, github-actions]

# Dependency graph
requires:
  - phase: 260719-rmx
    provides: tomtoolkit 3.0.0 final + Bootstrap 4 -> 5 migration (django_bootstrap5/crispy_bootstrap5)
provides:
  - Playwright-based Django functional test suite proving BS5 JS + crispy markup render correctly
  - Dedicated functional-tests CI job (Python 3.11, cached SPICE kernels, chromium)
affects: [issue-45-tomtoolkit-3.0-upgrade]

# Tech tracking
tech-stack:
  added: ["playwright (dev-only, plain package, no pytest-playwright/pytest-django)"]
  patterns:
    - "StaticLiveServerTestCase + playwright.sync_api.sync_playwright for browser-driven functional tests, kept in solsys_code/tests/ (Django test runner, not pytest) per repo testpaths convention"
    - "os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true') to work around a Playwright-sync-API / Django async-safety false positive"

key-files:
  created:
    - solsys_code/tests/test_bootstrap5_rendering.py
  modified:
    - pyproject.toml
    - src/fomo/settings.py
    - .github/workflows/testing-and-coverage.yml

key-decisions:
  - "Plain `playwright` only in dev deps — no pytest-playwright/pytest-django — functional tests are driven directly from Django's own test runner, keeping the pytest suite (testpaths=tests,src,docs) free of Django/DB plugins per CLAUDE.md."
  - "Fixed a genuine crispy_forms bootstrap5 config gap (CRISPY_ALLOWED_TEMPLATE_PACKS) discovered by the new ephemeris-form test — this was a live-rendering regression from the prior BS4->BS5 migration that no unit-level template-tag test had caught."

requirements-completed: [issue-45]

coverage:
  - id: D1
    description: "Navbar dropdown functional test proves BS5 JS runs (click reveals .dropdown-menu.show)"
    requirement: "issue-45"
    verification:
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#test_navbar_dropdown_toggle_shows_menu"
        status: pass
    human_judgment: false
  - id: D2
    description: "Ephemeris-form functional test proves BS4->BS5 crispy migration (zero .form-row, >=1 form .row)"
    requirement: "issue-45"
    verification:
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#test_ephemeris_form_uses_bs5_crispy_layout"
        status: pass
    human_judgment: false
  - id: D3
    description: "Observatory-create functional test proves the obscode form submits to a URL under /observatory/"
    requirement: "issue-45"
    verification:
      - kind: e2e
        ref: "solsys_code/tests/test_bootstrap5_rendering.py#test_observatory_create_form_submits_to_observatory_url"
        status: pass
    human_judgment: false
  - id: D4
    description: "Dedicated functional-tests CI job (Python 3.11) runs the suite with a cached SPICE kernel dir, leaving the build job's 3.10/3.11/3.12 matrix untouched"
    requirement: "issue-45"
    verification:
      - kind: other
        ref: "python -c \"import yaml; ...\" validating jobs.build and jobs.functional-tests structure (plan Task 3 automated verify)"
        status: pass
    human_judgment: true
    rationale: "The workflow YAML's structural correctness was verified locally, but the job has not actually been run on GitHub Actions (no CI run triggered yet from this local session) — a human should confirm the first real CI run succeeds."

duration: ~20min
completed: 2026-07-20
status: complete
---

# Phase 260720-9wm: Add Playwright Functional Test Suite for BS5 Rendering Summary

**Playwright-driven Django functional test suite (3 tests) proving BS5 JS and crispy BS5 layout markup actually render, plus a dedicated CI job — and along the way, caught and fixed a live crispy_forms `CRISPY_ALLOWED_TEMPLATE_PACKS` regression from the prior BS4->BS5 migration that broke the makeephem page.**

## Performance

- **Duration:** ~20 min (plan dispatch to final task commit)
- **Started:** 2026-07-20T07:12:25-07:00
- **Completed:** 2026-07-20T07:26:04-07:00
- **Tasks:** 3
- **Files modified:** 4 (3 planned + 1 auto-fixed via Rule 1)

## Accomplishments
- Added plain `playwright` to `pyproject.toml`'s `dev` optional-dependencies (no pytest-playwright, no pytest-django), keeping the pytest suite Django/DB-plugin-free.
- Created `solsys_code/tests/test_bootstrap5_rendering.py`: a `StaticLiveServerTestCase` launching one headless chromium browser per test class (page per test) with 3 tests — navbar dropdown JS, ephemeris-form crispy BS5 layout, observatory-create form submission.
- Added a new `functional-tests` GitHub Actions job (Python 3.11 only) that caches `~/.cache/sorcha` (key `sorcha-kernels-v1`), installs chromium with OS deps, and runs the new suite — the existing `build` job's 3.10/3.11/3.12 pytest matrix is completely untouched.
- Discovered and fixed a genuine bug: `CRISPY_ALLOWED_TEMPLATE_PACKS` was missing from `src/fomo/settings.py`, so every `{% crispy form %}` render (including the makeephem page) raised `TemplateSyntaxError` since the prior migration set `CRISPY_TEMPLATE_PACK = 'bootstrap5'` without allow-listing it.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add playwright to dev optional-dependencies** - `51750a8` (chore)
2. **Task 2: Create the Playwright BS5-rendering functional test suite** - `50de062` (feat, includes Rule 1 auto-fix to `src/fomo/settings.py`)
3. **Task 3: Add a dedicated functional-tests CI job** - `1879144` (ci)

**Plan metadata:** `a8da9c7` (docs: pre-dispatch plan, prior to execution)

## Files Created/Modified
- `pyproject.toml` - Added `playwright` to `dev` optional-dependencies.
- `solsys_code/tests/test_bootstrap5_rendering.py` - New Playwright functional test suite (3 tests) driving a real headless chromium browser against a live Django test server.
- `src/fomo/settings.py` - Added `CRISPY_ALLOWED_TEMPLATE_PACKS = ('bootstrap4', 'bootstrap5')` (Rule 1 auto-fix; see Deviations).
- `.github/workflows/testing-and-coverage.yml` - Added a new `functional-tests` job (Python 3.11, cached SPICE kernels, chromium install, runs the new suite); `build` job left untouched.

## Decisions Made
- Plain `playwright` dependency (not `pytest-playwright`/`pytest-django`) — functional tests live in the Django test runner (`manage.py test`), driven directly by `playwright.sync_api`, per this repo's convention of keeping the pytest suite (`testpaths = tests, src, docs`) free of Django/DB dependencies.
- Set `os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')` at module import time in the new test file — a targeted, documented workaround for a known Playwright-sync-API / Django async-safety false positive (see Issues Encountered).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing `CRISPY_ALLOWED_TEMPLATE_PACKS` breaking all `{% crispy %}` renders**
- **Found during:** Task 2 (writing/running the ephemeris-form functional test)
- **Issue:** `django-crispy-forms` 2.4 validates the `{% crispy %}` tag's `template_pack` argument against `CRISPY_ALLOWED_TEMPLATE_PACKS`, which defaults to `('uni_form', 'bootstrap3', 'bootstrap4')`. The prior BS4->BS5 migration (quick-260719-rmx) set `CRISPY_TEMPLATE_PACK = 'bootstrap5'` but never added this allow-list entry, so every `{% crispy form %}` render — including the live makeephem page — raised `django.template.exceptions.TemplateSyntaxError: crispy tag's template_pack argument should be in ('uni_form', 'bootstrap3', 'bootstrap4')`. No unit-level template-tag test caught this because it only manifests when the template is actually rendered end-to-end.
- **Fix:** Added `CRISPY_ALLOWED_TEMPLATE_PACKS = ('bootstrap4', 'bootstrap5')` to `src/fomo/settings.py`, immediately above the existing `CRISPY_TEMPLATE_PACK = 'bootstrap5'` line.
- **Files modified:** `src/fomo/settings.py`
- **Verification:** `test_ephemeris_form_uses_bs5_crispy_layout` failed with the `TemplateSyntaxError` before the fix and passes (along with the other 2 tests) after it; `ruff check .` / `ruff format --check .` stayed clean.
- **Committed in:** `50de062` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug, Rule 1)
**Impact on plan:** The fix was necessary for the ephemeris-form functional test (an explicit `must_haves.truths` item) to pass at all, and it fixes a real production bug (the makeephem page was broken). No scope creep beyond what was required to satisfy the plan's own done criteria.

## Issues Encountered

- **Playwright sync API vs. Django async-safety guard:** Starting `sync_playwright()` in `setUpClass` left an asyncio event loop appearing "running" in the test thread (a documented Playwright/Django interaction caused by Playwright's greenlet-based sync dispatch), which made Django's `async_unsafe` guard raise `SynchronousOnlyOperation` on ordinary synchronous DB access (e.g. `NonSiderealTargetFactory.create()`, `flush` between tests) — a false positive, not an actual concurrency issue. Resolved by setting `DJANGO_ALLOW_ASYNC_UNSAFE=true` at the top of the test module (see Decisions Made). This is scoped to this one file's process-local environment variable and does not affect any other test module.
- **Local sandbox lacked chromium's OS-level shared libraries** (`libnspr4.so` and friends) and had no passwordless `sudo`, so `playwright install --with-deps chromium` could not run. Worked around it in this session by `apt-get download`-ing the needed `.deb` packages (no root required for download) and extracting them into a scratch prefix, then setting `LD_LIBRARY_PATH` for the local test run. This was a session-local workaround only — it is not encoded anywhere in the repo; the CI job's `playwright install --with-deps chromium` step runs as root on the GitHub Actions runner and needs no such workaround.

## User Setup Required

None - no external service configuration required. (Local dev machines without root/sudo access will need their OS package manager to install chromium's shared-library dependencies before `playwright install chromium` browsers can launch — this is an environment prerequisite, not something the plan's 3 files can encode.)

## Next Phase Readiness

- The functional-tests CI job has not yet been exercised on an actual GitHub Actions run (it will trigger on the next push/PR against `main`) — flagged as `human_judgment: true` in the coverage block above for confirmation once CI runs.
- No blockers for further work on issue #45 or the tomtoolkit 3.0/BS5 upgrade; the BS5 rendering regression this suite exists to guard against is now both fixed and covered by a regression test.

---
*Phase: 260720-9wm*
*Completed: 2026-07-20*

## Self-Check: PASSED

All created/modified files and all 3 task commit hashes (51750a8, 50de062, 1879144) verified present in the working tree and `git log`.
