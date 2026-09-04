# Deferred Items — Phase 33

Out-of-scope discoveries logged per the executor's scope-boundary rule (do not fix issues
unrelated to the current plan's changes).

## Plan 33-03

- **`solsys_code.tests.test_bootstrap5_rendering.TestBootstrap5Rendering.test_observatory_create_form_submits_to_observatory_url`
  is flaky against live network access.** Running the project's full-suite `test_command`
  (`.planning/config.json`) after plan 33-03's changes landed produced one failure in this
  test: `requests.exceptions.ReadTimeout:
  HTTPSConnectionPool(host='data.minorplanetcenter.net', port=443): Read timed out.` This test
  drives a headless Chromium browser through the Observatory-create form, which resolves an
  MPC obscode via a **live** call to `data.minorplanetcenter.net` (`MPCObscodeFetcher`), with
  no mock/VCR fixture. The file (`solsys_code/tests/test_bootstrap5_rendering.py`) and the code
  it exercises (`solsys_code/solsys_code_observatory/`) are both untouched by plan 33-03 — no
  commit in this plan modified either. `solsys_code.tests.test_calendar_event_meta_links` and
  `solsys_code.tests.test_admin` (the two modules plan 33-03 changed or added) both pass in
  isolation and combined. Out of scope for this plan per the deviation-rule scope boundary; not
  fixed here.
