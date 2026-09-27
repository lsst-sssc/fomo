---
phase: 260927-eqs
plan: 01
subsystem: infra
tags: [django, cron, unattended-runner, zoneinfo, requests, logging, sphinx-runbook]

# Dependency graph
requires:
  - phase: 36-unattended-operation
    provides: run_tick()/run_unattended, the START/END banners, and the per-step lock/heartbeat/notification machinery this plan reformats
provides:
  - "_exception_label() -- a credential-free exception label (class name, plus HTTP status code when the portal returned one)"
  - "_host_timezone()/_local_timestamp() -- host-local ISO-8601 timestamps read from /etc/localtime, immune to Django's process-wide UTC TZ"
  - "_write_banner() with duration= on the END banner, and a timestamped internal lock-skip line"
  - "Updated docs/runbooks/telescope_runs_calendar.rst matching the new log/email output exactly"
affects: [unattended-operation, status_refresh troubleshooting, cron log legibility]

# Actuals (#2632)
actuals:
  tokens: 9900
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Read the host's real timezone directly from /etc/localtime via zoneinfo.ZoneInfo.from_file() rather than the process's own local-time conversion, since Django's settings loader pins the process TZ to settings.TIME_ZONE and calls time.tzset()."
    - "A credential-hygiene helper reads only exc.response.status_code (isinstance int check, bool excluded) and never the response's message/body/URL/headers/reason -- proven by an end-to-end test seeding all of those in a real requests.Response."

key-files:
  created: []
  modified:
    - solsys_code/unattended.py
    - solsys_code/tests/test_unattended.py
    - docs/runbooks/telescope_runs_calendar.rst

key-decisions:
  - "DEC-1: reformat the existing START/END banner text in place; no second header line, and the exact prefixes stay byte-identical for greps and the existing state-handling test."
  - "DEC-2: read the host zone from /etc/localtime with ZoneInfo.from_file(), falling back to UTC on any failure; no new Django setting."
  - "DEC-3: isoformat(timespec='seconds'), matching the cron guard's own date -Is precision."
  - "DEC-4: duration=<N>s (round() of START-to-END) added to the existing END banner, after exit=."
  - "DEC-5: the in-process LockContended skip line gets a trailing ' (<timestamp>)', on both the stderr write and the logger.warning call."
  - "DEC-6: _exception_label()'s HTTP-status-code scope is status_refresh's four exception sites only -- ping_heartbeat(), step_proposal_allocation()/refresh_all(), and run_tick()'s generic 'step %s raised' line stay class-name-only, per the plan's explicit scope boundary."

requirements-completed: [SCHED-09, SCHED-10]

coverage:
  - id: D1
    description: "A status_refresh portal HTTPError is reported as 'HTTPError <code>' in the per-record log line, the step summary's classes:/outage(...) field, and the failure email; a missing response or non-integer status code falls back to the bare class name."
    requirement: SCHED-10
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py#TestExceptionLabel"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py#TestStatusRefreshStep::test_per_record_http_status_code_is_reported"
        status: pass
      - kind: integration
        ref: "solsys_code/tests/test_unattended.py#TestCredentialHygiene::test_status_refresh_http_status_code_is_reported_without_leaking"
        status: pass
    human_judgment: false
  - id: D2
    description: "START/END log banners carry the host's local ISO-8601 time with UTC offset, to the second; END also carries duration=<N>s; a missing/corrupt /etc/localtime degrades to UTC without raising."
    requirement: SCHED-09
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py#TestLocalTimestamp"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py#TestEndBannerTimestamp"
        status: pass
      - kind: other
        ref: "python manage.py run_unattended --dry-run (live host check, -07:00 offset + duration= confirmed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The in-process lock-skip line carries the same host-local timestamp, distinguishable from the cron guard's own leading-timestamp skip line."
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_unattended.py#TestLocking::test_contended_lock_skips_every_step"
        status: pass
    human_judgment: false
  - id: D4
    description: "The paired runbook section and its troubleshooting entries show the new banner/duration/lock-skip/status-code output and explain the startup-chatter lines, matching what the code actually emits."
    verification:
      - kind: other
        ref: "Task 3 <verify> script (grep counts) and sphinx-build -M html (0 new warnings for this file)"
        status: pass
    human_judgment: false

# Metrics
duration: ~20min
completed: 2026-09-27
status: complete
---

# Quick Task 260927-eqs: Timestamp Each Unattended Run and Log HTTP Status Codes Summary

**Host-local ISO-8601 timestamps with `duration=` on the unattended runner's START/END banners, and `HTTPError <code>` (never the response body/URL/headers) on status_refresh portal failures, plus a matching runbook update.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-09-27T17:55Z (approx.)
- **Completed:** 2026-09-27T18:12:40-07:00
- **Tasks:** 3/3
- **Files modified:** 3

## Accomplishments
- A portal `HTTPError` during `status_refresh` now reports its numeric HTTP status code (`HTTPError 502`) in the per-record log line, the step summary's `classes:`/`outage(...)` field, and the failure email -- distinguishing a transient 502/503/504 from a 404 or 429 -- while `_exception_label()` proves (via a new end-to-end test) that no response body, URL, query string, header, reason phrase, or exception message ever escapes.
- Every tick's START/END banners now report the host's own local time with its UTC offset, to the second, read directly from `/etc/localtime` (never from Django's process-wide `TIME_ZONE=UTC`); the END banner additionally carries `duration=<N>s`, and the in-process lock-contention skip line carries the same trailing timestamp.
- `docs/runbooks/telescope_runs_calendar.rst`'s unattended-operation walkthrough and troubleshooting entries now quote the actual new output character-for-character, including a new entry explaining how to read an `HTTPError <code>` failure.

## Task Commits

Each task was committed atomically:

1. **Task 1 (tracer): status_refresh HTTP status codes without leaking credentials** - `ee8e47b` (feat)
2. **Task 2: host-local banner timestamps and duration=** - `911e770` (feat)
3. **Task 3: paired runbook update** - `2da32fe` (docs)

_All three tasks were TDD-driven (RED confirmed before each GREEN implementation) except Task 3, which is docs-only._

## Files Created/Modified
- `solsys_code/unattended.py` - Added `_exception_label()`, `_host_timezone()`, `_local_timestamp()`, `_HOST_LOCALTIME_PATH`; `_refresh_one_facility()` now uses `_exception_label()` at all four exception-reporting sites; `_write_banner()` renders host-local timestamps and `duration=`; `run_tick()`'s `LockContended` branch stamps its skip line and passes `duration_seconds` to the END banner.
- `solsys_code/tests/test_unattended.py` - Added `_http_error()` test helper, `TestExceptionLabel`, `TestLocalTimestamp`; extended `TestStatusRefreshStep` (per-record/multi-code/outage status-code cases), `TestCredentialHygiene` (end-to-end hygiene proof), `TestLocking` and `TestEndBannerTimestamp` (timestamp/duration assertions).
- `docs/runbooks/telescope_runs_calendar.rst` - Updated the walkthrough's dry-run banner example, added a reading-the-log note (startup-chatter lines + a failing-tick example), updated "The two failure signals" email paragraph with an `HTTPError 502` example and hygiene explanation, updated "When nothing has appeared" items 2 and 4, updated "Repeated 'lock held' lines" with the timestamped internal message, and added a new "A status_refresh failure names an HTTP status code" troubleshooting entry.

## Decisions Made
See `key-decisions` in the frontmatter (DEC-1 through DEC-6) -- all were recorded in the plan itself (no CONTEXT.md existed for this quick task) and implemented exactly as specified. No new decisions were needed during execution.

## Deviations from Plan

None - plan executed exactly as written. All `<behavior>` cases in both TDD tasks were implemented as specified; the `<action>` sections' RED-then-GREEN sequencing was followed for both Task 1 and Task 2 (confirmed RED failures before implementing GREEN); Task 3's edits (a)-(f) were all applied.

## Issues Encountered
None. The live-checkout precautions in the plan's objective (never leave `unattended.py` non-importable across a quarter-hour boundary) were followed: an import check ran after each edit to `unattended.py`, and both Task 1 and Task 2 committed helper-additions and call-site switch-overs together within a single GREEN step, never in a half-applied intermediate state.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- The unattended runner's log is now directly comparable to the cron guard's own `date -Is` lines (same clock, same precision), and a status_refresh failure email now distinguishes transient portal errors (5xx) from client-side ones (4xx) without opening the log.
- No blockers. The next real cron tick on this host (every 15 minutes) will pick up the new banner/timestamp/status-code behavior automatically, since this is a live checkout.

---
*Phase: 260927-eqs*
*Completed: 2026-09-27*

## Self-Check: PASSED

All modified files and all three task commit hashes were confirmed present on disk / in `git log --oneline --all` before this SUMMARY was finalized.
