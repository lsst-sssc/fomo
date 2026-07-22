Findings

High: run_status is persisted before an unguarded calendar write, so failures can leave partial state and a 500.
solsys_code/campaign_views.py:736 updates CampaignRun.run_status, then solsys_code/campaign_views.py:752/solsys_code/campaign_views.py:754 calls insert_or_create_calendar_event(...) without try/except.
If calendar update fails (DB/network/runtime), user gets an error after status is already changed; no recovery message/revert path like _resolve_site() has.

Resolved: fixed in quick task 260718-dih. `_set_run_status` (now starting at
solsys_code/campaign_views.py:740) still commits the run_status change first, via the
conditional `.update()` at solsys_code/campaign_views.py:775-777, but the calendar-sync loop
that follows is now wrapped in a non-reverting try/except (solsys_code/campaign_views.py:801
opens the try block, the `insert_or_create_calendar_event(...)` call is at
solsys_code/campaign_views.py:804, and the except at solsys_code/campaign_views.py:815 logs
via `logger.exception` and shows a warning telling the user the status change was saved and to
retry the same Mark Cancelled / Mark Weathered action). This mirrors `_resolve_site()`'s
existing non-reverting projection guard. A sync failure now redirects with a warning instead of
raising an uncaught 500.

Medium (behavioral mismatch): parser accepts cross-month ranges, but loader rejects them.
solsys_code/telescope_runs.py:90–solsys_code/telescope_runs.py:99 and solsys_code/telescope_runs.py:437 parse 28 December-2 January.
solsys_code/management/commands/load_telescope_runs.py:76–solsys_code/management/commands/load_telescope_runs.py:77 raises on day2 < day1.
Result: syntactically valid lines are parsed but always skipped in ingestion; this is an architectural contract mismatch between modules.

Resolved: this cross-month rejection was originally a documented, intentional Phase-3
deferral, not an accidental gap -- the fix in quick task 260718-dih changes it from a fail-late
design (the parser succeeds and builds a `ParsedRun`, then the loader always rejects it) to a
fail-fast one (the parser rejects it immediately). `_CROSS_MONTH_RANGE` is still defined at
solsys_code/telescope_runs.py:91, but `parse_run_line` now raises a `ValueError` naming the
unsupported cross-month range as soon as it matches, at
solsys_code/telescope_runs.py:445 -- no `ParsedRun` is ever built for a genuine cross-month
line, and the now-dead December-to-January year-rollover block was removed. The loader's
`_iter_run_nights` guard at solsys_code/management/commands/load_telescope_runs.py:79 is kept
(still load-bearing for a descending same-month day range, e.g. a typo like '20-5 July', which
`parse_run_line` does not reject), with its docstring and message reworded to describe that
narrower remaining case rather than cross-month ranges. The original "architectural contract
mismatch" framing no longer applies -- both layers now agree that a cross-month range is
unsupported, and it is caught as early as possible.

Low/Medium (edge-case parser bug): partial-night token validation is too permissive.
solsys_code/telescope_runs.py:106 defines _PARTIAL_NIGHTS without anchors and solsys_code/telescope_runs.py:478 uses .search(...).
Tokens with extra surrounding garbage can still match (substring match), so malformed inputs may be accepted instead of rejected.

Resolved: fixed in quick task 260718-dih. `_PARTIAL_NIGHTS`'s definition is unchanged at
solsys_code/telescope_runs.py:106, but its call site now uses `.fullmatch(...)` instead of
`.search(...)`, at solsys_code/telescope_runs.py:485. Because the matched value is always a
single whitespace-delimited token, `fullmatch` requires the whole token to match the pattern,
so a garbage-wrapped token like `'xBoN-0626'` is rejected with a `ValueError` while a
well-formed token like `'BoN-0626'` or `'0646-EoN'` still parses exactly as before.

Coverage gaps
No explicit test for mark_cancelled/mark_weather_failure when calendar update throws (partial-write risk path).

Resolved: closed by `TestRunStatusChange.test_mark_cancelled_survives_calendar_sync_failure`
in solsys_code/tests/test_campaign_approval.py, which patches
`insert_or_create_calendar_event` to raise and asserts the response redirects (200 after
follow), the run's `run_status` is committed as `CANCELLED`, and a warning message mentioning
the calendar sync failure is present.

No command-level test for cross-month run lines (accepted by parser but rejected downstream).

Resolved: closed by `TestLoadTelescopeRuns.test_cross_month_line_logged_and_skipped` in
solsys_code/tests/test_load_telescope_runs.py, which runs `load_telescope_runs` against a
schedule file containing a genuine cross-month line alongside a valid line, and asserts the
cross-month line is skipped and logged with its line number and text to stderr while the valid
line still produces its CalendarEvents.

No parser test asserting rejection of partial-night tokens with extra prefix/suffix text.

Resolved: closed by
`TestTelescopeRuns.test_parse_run_line_partial_night_token_with_garbage_prefix_raises` in
solsys_code/tests/test_telescope_runs.py, which asserts a run line whose trailing window token
is `'xBoN-0626'` raises `ValueError`, proving the `fullmatch` anchoring.