## Deferred Items

- Flaky `test_observatory_create_form_submits_to_observatory_url` under the full
  `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` run (37-08)
  status: open
  **What:** Observed during 37-08 Task 3's full-suite verification gate. The full 1746-test
  run reported `FAILED (failures=1, skipped=1)` with a single failure in
  `solsys_code.tests.test_bootstrap5_rendering.TestBootstrap5Rendering.test_observatory_create_form_submits_to_observatory_url`
  (`AssertionError: assert '/observatory/create/' not in self.page.url` -- a Playwright
  browser-page assertion). This file is entirely outside 37-08's `files_modified`
  (`solsys_code/campaign_tally.py`, `solsys_code/campaign_views.py`,
  `solsys_code/tests/test_campaign_tally.py`, `solsys_code/tests/test_campaign_views.py`,
  `docs/runbooks/telescope_runs_calendar.rst`) and imports none of them.
  **Why deferred, not fixed:** out of scope per the executor's scope-boundary rule (only
  fix issues directly caused by the current task's changes). Confirmed pre-existing and
  order-dependent, not a 37-08 regression: (1) the test passes in isolation
  (`python manage.py test solsys_code.tests.test_bootstrap5_rendering.TestBootstrap5Rendering.test_observatory_create_form_submits_to_observatory_url`
  -> OK); (2) it also passes when run together with both files this plan touched
  (`test_bootstrap5_rendering` + `test_campaign_tally` + `test_campaign_views`, 154 tests,
  OK); (3) Django's alphabetical test-module ordering runs `test_bootstrap5_rendering`
  before `test_campaign_tally`/`test_campaign_views` in a full-app run, so nothing this
  plan changed could have influenced it. Most likely a Playwright browser-page state leak
  from an unrelated earlier test in the full 1746-test run, not reproducible in a smaller
  run.
  **Follow-up:** a quick task to investigate Playwright test isolation in
  `test_bootstrap5_rendering.py` (or the full-suite runner's browser-context teardown) when
  this recurs.
  **Confirmed transient:** an identical immediate re-run of
  `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` (same 1746 tests, no
  code changes in between) completed `OK (skipped=1)` with no failure at all -- this
  specific occurrence did not reproduce, reinforcing that it is order/timing-dependent
  flakiness rather than a real regression from 37-08.
