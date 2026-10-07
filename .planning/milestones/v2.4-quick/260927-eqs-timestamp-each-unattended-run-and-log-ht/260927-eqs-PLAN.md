---
phase: 260927-eqs
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/unattended.py
  - solsys_code/tests/test_unattended.py
  - docs/runbooks/telescope_runs_calendar.rst
autonomous: true
requirements: [SCHED-09, SCHED-10]

estimate:
  tokens: 70000
  raw_tokens: 70000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "Every tick's START banner in `/var/log/fomo/unattended.log` carries the host's local ISO-8601 time with its UTC offset, to the second (e.g. `=== FOMO unattended run START 2026-09-26T11:45:02-07:00 ===`), even though Django's `TIME_ZONE` is `'UTC'`."
    - "Every END banner uses the same timestamp format and carries `exit=<code> duration=<N>s` (e.g. `=== FOMO unattended run END 2026-09-26T11:45:34-07:00 exit=1 duration=32s ===`)."
    - "The in-process lock-contended skip line carries the same timestamp, e.g. `run_unattended: lock held -- skipping this tick (2026-09-26T11:45:02-07:00)`, so no line the runner itself writes for a tick is left without a time."
    - "If `/etc/localtime` is missing, unreadable, or corrupt, the tick still runs every step and stamps its banners in UTC (`+00:00`). The timestamp lookup can never abort a tick."
    - "A status_refresh per-record portal `HTTPError` is logged as `observation_id=<id> HTTPError <code>` and summarized as `classes: HTTPError <code>` in both the `step status_refresh: FAILED | ...` log line and the failure email's `- status_refresh: ...` line. A whole-facility outage reads `LCO: outage (HTTPError <code>)`."
    - "An `HTTPError` with no response, or a response whose `status_code` is not an integer, falls back to the bare class name (`HTTPError`)."
    - "SCHED-10 still holds. No response body, request URL or query string, request/response header, reason phrase, or exception message reaches the log, stdout, stderr, or the email. This is proven by an extended credential-hygiene test that seeds a real `requests.Response` carrying the fake API key in its body, URL query string and `Authorization` header."
    - "The runbook's \"How do I run everything unattended?\" section, and its unattended troubleshooting entries, show the new banner, END-duration, lock-skip and status-code forms. They also explain how to read them, including the process-startup lines that come before each START banner."
  artifacts:
    - "solsys_code/unattended.py — new `_exception_label()`, `_host_timezone()`, `_local_timestamp()`, `_HOST_LOCALTIME_PATH`. `_refresh_one_facility()` uses `_exception_label()` at all four exception-reporting sites. `_write_banner()` gains local timestamps and `duration=`. `run_tick()`'s LockContended branch stamps its skip line."
    - "solsys_code/tests/test_unattended.py — helper-level, step-level and end-to-end status-code tests. A new credential-hygiene test with a real `requests.Response`. Local-timestamp and fallback tests. An updated `TestEndBannerTimestamp`. A timestamped lock-skip assertion."
    - "docs/runbooks/telescope_runs_calendar.rst — updated example log and email output, a reading-the-log note, updated lock-held wording, and one new `^^^^`-level troubleshooting entry for a status-code failure."
  key_links:
    - "`_refresh_one_facility()` -> `_exception_label()` -> `step_status_refresh()` summary `classes:` field -> `_build_notification_body()` `- status_refresh: <summary>` email line (no change needed in the email builder; it already quotes the summary verbatim)"
    - "`run_tick()` -> `_write_banner()` -> `_local_timestamp()` -> `_host_timezone()` -> `/etc/localtime` (`_HOST_LOCALTIME_PATH`), with UTC fallback"
    - "Runbook example lines <-> the `_write_banner()` format strings, the LockContended skip string, and `_exception_label()` output. The runbook must quote what the code actually emits after Tasks 1-2."
---

<objective>
Make each unattended tick easy to place in time in `/var/log/fomo/unattended.log`, and make a status_refresh failure say which HTTP status the LCO portal returned. SCHED-10's credential guarantee must not weaken. Then update the paired runbook section to match.

Purpose: the 2026-09-26 failure email read `- status_refresh: LCO: failed 1 | SOAR: failed 0 | classes: HTTPError`. Nothing in it separated a transient 502 gateway blip from a 404 or 429, and the log's own banners were on a different clock from the operator and from the cron guard's own lines.

Output: code changes in `solsys_code/unattended.py`, tests in `solsys_code/tests/test_unattended.py`, and runbook changes in `docs/runbooks/telescope_runs_calendar.rst`. The cron line, lock handling, notification transport and step order are NOT changed. This is output legibility only.

**Planning-time correction to the task statement (read before starting).** The orchestrator's evidence said the rotated log has "NO timestamps on any line". That is not quite right, and the difference shapes the fix. Checked at planning time against `/var/log/fomo/unattended.log-20260927`, all 96 ticks already have a `=== FOMO unattended run START <ts> ===` / `=== FOMO unattended run END <ts> exit=N ===` pair from `unattended._write_banner()`. The problems are:
(a) The timestamps are UTC with microseconds (`2026-09-26T07:00:07.532207+00:00`). The host runs America/Los_Angeles, and the cron guard's own skip line uses `date -Is` local time (`-07:00`), so the file mixes two clocks.
(b) Each START banner comes after about 13 lines of process-startup output: `Note: NumExpr detected 32 cores ...`, `NumExpr defaulting to 16 threads.`, `Using fallback library next to module: .../libcspice.so`, eight `registering new views: ...` lines, then Django's `System check identified some issues:` / `WARNINGS:` / `?: (urls.W005) URL namespace 'calendar' isn't unique ...` block. The banner is easy to miss after all that.
(c) A tick's duration can only be worked out by subtraction. The 2026-09-27T00:45 UTC tick took 4m39s, and nothing in the log says so.
So this plan changes the format of the EXISTING banners rather than adding a second header line, adds `duration=` to END, stamps the in-process lock-skip line, and documents the startup lines.

**Stream check (asked for by the orchestrator).** Banners, step lines and the per-record `observation_id=... HTTPError` warnings all go through `logging`. That goes to the project's `LOGGING` `console` handler (`src/fomo/settings.py:200-209`, a bare `logging.StreamHandler`, whose default stream is stderr). Django's system-check `WARNINGS:` block is written to stderr by `BaseCommand.check()` before `handle()` runs. The cron line redirects `>> /var/log/fomo/unattended.log 2>&1`. So everything lands in the one file, in the order it was emitted, and no stream change is needed.

**Timezone pitfall (verified at planning time; this is the easy thing to get wrong).** Django's settings loader sets `os.environ['TZ']` to `settings.TIME_ZONE` (`'UTC'`, `src/fomo/settings.py:172`) and calls `time.tzset()` before any management command runs. Inside `python manage.py shell` on this host, the TZ environment variable reads `UTC`, and a no-argument `datetime.now().astimezone()` gives `2026-09-27T17:40:23+00:00`. Reading the zone from `/etc/localtime` with `zoneinfo.ZoneInfo.from_file()` gives `2026-09-27T10:40:23-07:00`, which matches `date -Is` in the cron shell. `/etc/localtime` is a symlink to `../usr/share/zoneinfo/America/Los_Angeles`. So the host zone MUST be read from the file, never from the process's local-time conversion.

**Where HTTPError comes from (verified at planning time).** TOM's `tom_observations/facilities/ocs.py` `make_request()` (lines 180-187) handles portal errors as follows:
- 401-403 become `ImproperCredentialsException('OCS: ' + str(response.content))`. The message embeds the response body; it stays class-name-only.
- 400 becomes `forms.ValidationError`.
- Every other 4xx/5xx goes through `response.raise_for_status()`, which raises a real `requests.HTTPError`. Its `.response` is set, and its message is `"<code> Server Error: <reason> for url: <full URL>"`.
So in practice `HTTPError <code>` means 404, 429 or 5xx, and the exception's message always contains the request URL. Only the integer `exc.response.status_code` is safe to read.

**Operational note: this checkout is live.** The real host's crontab runs `/home/tlister/git/fomo_devel/manage.py run_unattended` every 15 minutes from THIS working tree (confirmed with `crontab -l` at planning time). Any edit to `solsys_code/unattended.py` therefore takes effect on the next tick. A non-importable intermediate state (a half-applied edit, a call to a helper not yet defined) would make that tick fail and mail staff. Two rules follow:
- In each GREEN step, add the new helper functions and constants first, and only then switch call sites over to them.
- Never leave `unattended.py` non-importable across a quarter-hour boundary.
Test-only edits (the RED steps) have no runtime effect. If the executor runs in a separate git worktree, this does not apply until merge.

No CONTEXT.md exists for this quick task, so there are no D-XX decisions to cite. The choices below are Claude's discretion, recorded so the executor, checker and verifier all apply the same ones.
</objective>

<decisions_this_plan_records>
- DEC-1 (header): change the format of the existing START/END banners; do not add a second header line. Keep the exact prefixes `=== FOMO unattended run START ` and `=== FOMO unattended run END ` byte-for-byte. `test_state_handling_failure_never_blocks_the_end_banner_or_heartbeat`, the runbook, and operator greps all key on them.
- DEC-2 (clock): read the host's local zone from `/etc/localtime` with `zoneinfo.ZoneInfo.from_file()`. Fall back to UTC on ANY failure. Add no new Django setting: "the host's clock" is what the cron guard's `date -Is` already uses, and a setting would add a second place for that to disagree.
- DEC-3 (precision): `isoformat(timespec='seconds')`, matching `date -Is` and the user's example `2026-09-26T11:45:02-07:00`.
- DEC-4 (end/duration): add `duration=<N>s` (integer seconds, `round()` of the START-to-END difference) to the EXISTING END banner, after `exit=`. Do not add a separate end line. It is one field, and it makes an overrunning tick easy to grep for (anything over 900s overlapped the next cron slot). That is exactly what the runbook's "lock held" entry asks the operator to diagnose.
- DEC-5 (lock-skip line): the in-process `LockContended` skip text gets a trailing ` (<timestamp>)`. This applies to both the `sys.stderr.write` and the `logger.warning` emission. The existing substring `run_unattended: lock held -- skipping this tick` is kept intact for greps, and the line stays distinguishable from the cron guard's `<timestamp> run_unattended skipped: lock held`.
- DEC-6 (status-code scope): `_exception_label()` appends the code for any `requests.exceptions.RequestException` whose `.response` has an integer `status_code`. It is applied at all four exception-reporting sites inside `_refresh_one_facility()`: the whole-facility outage log line and return value, and the per-record log line and `class_names` entry. `ping_heartbeat()`, `step_proposal_allocation()` / `refresh_all()`, and `run_tick()`'s generic `step %s raised: %s` line stay class-name-only. The task names status_refresh only, and this plan does not widen that.
- DEC-7 (startup chatter): do not suppress Django system checks or library import output. That would change command behaviour, which is outside this legibility-only scope. The runbook documents those lines instead.
</decisions_this_plan_records>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@solsys_code/unattended.py
@solsys_code/tests/test_unattended.py
@docs/runbooks/telescope_runs_calendar.rst

Relevant line ranges (all as of planning time, HEAD c677406):
- `solsys_code/unattended.py:199-313`: `_refresh_one_facility()` and `step_status_refresh()`. The four exception sites are lines 235-237 (outage) and 246-248 (per-record).
- `solsys_code/unattended.py:752-758`: `_write_banner()`.
- `solsys_code/unattended.py:761-788`: `_build_notification_body()`. It quotes each failed step's `summary` verbatim, so it needs no change.
- `solsys_code/unattended.py:813-901`: `run_tick()`. `now` is taken at line 841; `end_time` at line 869; the `LockContended` branch is at lines 898-901.
- `solsys_code/tests/test_unattended.py:435-481`: `TestLocking.test_contended_lock_skips_every_step` and `TestEndBannerTimestamp`. The latter patches `solsys_code.unattended.datetime` with a fake class whose `now()` yields exactly two values.
- `solsys_code/tests/test_unattended.py:613-767`: `TestStatusRefreshStep`.
- `solsys_code/tests/test_unattended.py:1038-1115`: the `_FAKE_*` credential constants, plus `TestCredentialHygiene`'s `setUp`, `_assert_no_secrets_leaked()`, `_run_tick_capturing()` and `test_status_refresh_portal_error_leaks_nothing`.
- `docs/runbooks/telescope_runs_calendar.rst:1454-2045`: "How do I run everything unattended?". The walkthrough's step 7 banner example is at lines 1870-1880. "The two failure signals" email paragraph is at lines 1931-1942. "When nothing has appeared" items 2 and 4 are at lines 1981-2007.
- `docs/runbooks/telescope_runs_calendar.rst:2567-2740`: the unattended troubleshooting entries. "Repeated 'lock held' lines" is at lines 2579-2627; "A failure email arrived once, then went quiet" is at line 2628.
- `/home/tlister/venv/devel_fomo311_venv/lib/python3.11/site-packages/tom_observations/facilities/ocs.py:180-187`: TOM's `make_request()`, cited above.
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1 (tracer): the portal's HTTP status code travels from the status_refresh re-check to its log line, the step summary, and the failure email; nothing else from the response does</name>
  <files>solsys_code/unattended.py, solsys_code/tests/test_unattended.py</files>
  <read_first>
    - solsys_code/unattended.py lines 199-313 (`_refresh_one_facility`, `step_status_refresh`) and 761-788 (`_build_notification_body`)
    - solsys_code/tests/test_unattended.py lines 11-42 (imports, `_FAKE_HEARTBEAT_URL`), 613-767 (`TestStatusRefreshStep`), 1038-1115 (`_FAKE_*` constants, `TestCredentialHygiene` fixtures and helpers)
    - /home/tlister/venv/devel_fomo311_venv/lib/python3.11/site-packages/tom_observations/facilities/ocs.py lines 180-187 (`make_request`)
  </read_first>
  <behavior>
    - Build a real `requests.Response`: status 502, reason `Bad Gateway`, a URL with an `api_key=<seeded key>` query string, an `Authorization: Token <seeded key>` header, and a body containing the seeded key and a body marker. Call its `raise_for_status()` and catch the genuine `requests.exceptions.HTTPError` it raises. `unattended._exception_label()` of that exception returns exactly `HTTPError 502`.
    - The same construction with status 404 returns `HTTPError 404`. Any 4xx/5xx `requests.Response` is falsy, because `Response.__bool__` returns `.ok`, so this case proves the helper never tests the response's truthiness.
    - `requests.exceptions.HTTPError('boom')` (no response) returns `HTTPError`.
    - `requests.exceptions.HTTPError(response=MagicMock())` (its `status_code` is not an int) returns `HTTPError`.
    - `RuntimeError('x')` returns `RuntimeError`. `tom_common.exceptions.ImproperCredentialsException('OCS: b"..."')` returns `ImproperCredentialsException`.
    - The step, per-record path: LCO's `update_all_observation_statuses` returns `[('obs-1', <str of the 502 error>)]` and `update_observation_status` raises the 502 error. A WARNING log line then contains `observation_id=obs-1 HTTPError 502`, `result.summary` contains `classes: HTTPError 502`, and `result.failed` is True.
    - The step, three failed records re-checked with 502, 502, 504: the summary contains `classes: HTTPError 502, HTTPError 504`, and each label appears exactly once (the existing de-duplication now works on labels).
    - The step, outage path: LCO's `update_all_observation_statuses` itself raises a 503 `HTTPError`. The summary then contains `LCO: outage (HTTPError 503)`, a WARNING log line ends with `HTTPError 503`, and the summary does not contain `LCO: failed 1`.
    - End to end (the tracer's own proof), in `TestCredentialHygiene`: `_run_tick_capturing()` runs `call_command('run_unattended')` with the 502 error seeded as above. Assert all of the following:
      - `HTTPError 502` appears in the captured log output AND in a sent email body.
      - That email's subject names `status_refresh`.
      - `_assert_no_secrets_leaked()` passes.
      - None of these appear in the log output, stdout, stderr, or any sent email's subject or body: the body marker, the full URL, the `api_key=` query-string fragment, the full `Authorization` header value, the reason phrase `Bad Gateway`, and `str()` of the exception.
    - Every pre-existing test in the module stays green unchanged. In particular `test_failure_is_reported_by_class_name_not_message` (no response, bare `HTTPError`), `test_whole_facility_outage_reads_as_outage_not_failed_one` (`LCO: outage (RuntimeError)`) and `test_status_refresh_portal_error_leaks_nothing` must not change.
  </behavior>
  <action>
    RED first. In `solsys_code/tests/test_unattended.py`:
    - Add a module-level test helper named `_http_error`. It takes a status code plus optional keyword arguments for body bytes, URL, headers and reason. It builds a real `requests.Response`: set `status_code`, `reason`, `url` and `encoding='utf-8'`, set the private `_content` to the body bytes, and update `headers`. It then calls `raise_for_status()` inside try/except and returns the caught `requests.exceptions.HTTPError`. Using the genuine `raise_for_status()` means the exception's message really embeds the URL, exactly as TOM's `make_request()` produces it.
    - Put the helper-level cases in a new `TestExceptionLabel(UnattendedTestBase)` class.
    - Add the step-level cases to `TestStatusRefreshStep`, following its existing `@patch('solsys_code.unattended.SOARFacility')` / `@patch('solsys_code.unattended.LCOFacility')` decorator pattern.
    - Add the end-to-end case to `TestCredentialHygiene` as `test_status_refresh_http_status_code_is_reported_without_leaking`, reusing `_run_tick_capturing()` and `_assert_no_secrets_leaked()` and seeding with `_FAKE_LCO_API_KEY` / `_FAKE_MAIL_PASSWORD`. That class's `setUp` already creates a staff user with an email, so the failure email is actually sent.
    - Import `ImproperCredentialsException` from the existing `tom_common.exceptions` import.
    - Run the module and confirm the new tests fail with `AttributeError` for `_exception_label` or summary mismatches. Do not commit a red test.

    GREEN. In `solsys_code/unattended.py`:
    - Add a private function `_exception_label(exc: BaseException) -> str` directly above `_refresh_one_facility()`. It returns `type(exc).__name__`. When `exc` is a `requests.exceptions.RequestException` (`requests` is already imported), and its `response` attribute is not None, and that response's `status_code` is an `int` but not a `bool`, it returns `f'{type(exc).__name__} {status_code}'` instead.
    - Read the response with `getattr(exc, 'response', None)` and compare it with `is None`. NEVER test the response object's truthiness: `requests.Response.__bool__` returns `.ok`, which is False for every 4xx/5xx, the very case this helper exists for.
    - The helper must not read or format anything else from the exception or the response. That excludes the message, `str()` or `repr()` of either object, `.text`/`.content`, `.url`, `.reason`, `.headers`, and anything on `.request`.
    - Give it a Google-style docstring stating the SCHED-10 contract: only an integer can be appended, so no credential-bearing text can pass through. Also document the `Response.__bool__` pitfall and why the scope is `RequestException`: TOM's `make_request()` turns 401-403 into `ImproperCredentialsException`, whose message embeds the response body, so that class stays name-only. Cite quick task 260927-eqs, in the style of the module's existing WR-/IN- review citations.
    - Inside `_refresh_one_facility()` (per DEC-6), use `_exception_label(exc)` in place of `type(exc).__name__` at all four sites: the outage `logger.warning(...)` argument, the outage return value's fourth element, the per-record `logger.warning('observation_id=%s %s', ...)` argument, and the `class_names.append(...)` argument. Compute the label once per caught exception into a local variable.
    - Update `_refresh_one_facility()`'s docstring so `class_names` and `outage_class_name` are described as exception labels (class name, plus the HTTP status code when the portal returned one).
    - Leave `step_status_refresh()`'s summary-building code, the `classes:` field name, `_MAX_STATUS_RECHECKS`, and `_build_notification_body()` unchanged. The email already quotes the summary verbatim, which is what makes this a one-path end-to-end change.
    - Do not touch `ping_heartbeat()`, `step_proposal_allocation()`, or `run_tick()`'s `step %s raised` line (DEC-6).

    Run the test module and ruff (commands in verify) until all pass, then commit tests and implementation together.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel && python manage.py test solsys_code.tests.test_unattended --exclude-tag=ephemeris_segfault && pre-commit run ruff --all-files && pre-commit run ruff-format --all-files</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c '_exception_label(' solsys_code/unattended.py` prints at least 3: the definition plus one call in each of `_refresh_one_facility()`'s two except clauses. Each clause computes its label once into a local variable and reuses it for both of its report sites.
    - `python manage.py test solsys_code.tests.test_unattended.TestExceptionLabel solsys_code.tests.test_unattended.TestStatusRefreshStep solsys_code.tests.test_unattended.TestCredentialHygiene --exclude-tag=ephemeris_segfault` reports OK, and `TestCredentialHygiene` contains `test_status_refresh_http_status_code_is_reported_without_leaking`.
    - The whole `solsys_code.tests.test_unattended` module reports OK, and its test count is higher than the 74 found at planning time.
  </acceptance_criteria>
  <done>A portal HTTPError during status_refresh is reported as `HTTPError <code>` in the per-record log line, the step summary's `classes:` / `outage (...)` field, and the failure email. A missing response or non-integer code falls back to the bare class name. The new end-to-end hygiene test proves that no body, URL, query string, header, reason or message text escapes. Ruff is clean, and tests and implementation are committed together.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: stamp every tick's START/END banners and lock-skip line with the host's local ISO-8601 time and UTC offset, and add duration= to END</name>
  <files>solsys_code/unattended.py, solsys_code/tests/test_unattended.py</files>
  <read_first>
    - solsys_code/unattended.py lines 19-78 (imports and module constants), 752-758 (`_write_banner`), 813-901 (`run_tick`)
    - solsys_code/tests/test_unattended.py lines 435-481 (`TestLocking`, `TestEndBannerTimestamp`)
  </read_first>
  <behavior>
    - Point `unattended._HOST_LOCALTIME_PATH` (patched) at a file in the test's temp directory holding the bytes of `America/Los_Angeles`, read from the installed `tzdata` package via `importlib.resources.files('tzdata').joinpath('zoneinfo/America/Los_Angeles').read_bytes()`. Presence was verified at planning time; guard the test with `skipIf(importlib.util.find_spec('tzdata') is None, ...)`. With that zone:
      - `unattended._local_timestamp(datetime(2026, 7, 1, 12, 0, 0, 123456, tzinfo=dt_timezone.utc))` returns exactly `2026-07-01T05:00:00-07:00` (no fractional seconds, summer offset). This proves the zone comes from the file even though Django's settings have set the process TZ to UTC.
      - The same call with `datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt_timezone.utc)` returns exactly `2025-12-31T16:00:00-08:00` (winter offset).
    - With `_HOST_LOCALTIME_PATH` patched to a path that does not exist, `_local_timestamp()` of the July instant returns `2026-07-01T12:00:00+00:00` and raises nothing.
    - With `_HOST_LOCALTIME_PATH` patched to a file containing `b'not a tzif file'`, it returns the same UTC string and raises nothing.
    - Rewrite `TestEndBannerTimestamp.test_end_banner_timestamp_differs_from_start_banner_timestamp`. It keeps the existing two-value fake `datetime` (start 2026-01-01T00:00Z, end 2026-01-01T00:15Z), additionally patches `solsys_code.unattended._host_timezone` to return `ZoneInfo('America/Los_Angeles')`, and asserts that the captured log:
      - contains `START 2025-12-31T16:00:00-08:00 ===`;
      - contains `END 2025-12-31T16:15:00-08:00 exit=0 duration=900s ===`;
      - does not contain `END 2025-12-31T16:00:00-08:00`.
    - A new test runs a real dry-run tick with `call_command('run_unattended', '--dry-run')` and no clock patching. The captured START line matches the regex `=== FOMO unattended run START \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} ===`, and the END line matches `=== FOMO unattended run END \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2} exit=0 duration=\d+s ===`. Neither line contains a `.` fractional-seconds component.
    - Extend `TestLocking.test_contended_lock_skips_every_step`. Its captured stderr now matches the regex `run_unattended: lock held -- skipping this tick \(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\)`. Its existing assertions (`reconcile_run` not called, no mail) are unchanged.
    - Every other pre-existing test stays green. That includes `test_state_handling_failure_never_blocks_the_end_banner_or_heartbeat` (the `=== FOMO unattended run END` prefix) and the state-file tests. The state file keeps storing UTC `notified_at`, because only human-facing log text changes.
  </behavior>
  <action>
    RED first. Write or modify the tests listed in behavior in `solsys_code/tests/test_unattended.py`:
    - Add `re` and `importlib.util`/`importlib.resources` imports, and `from zoneinfo import ZoneInfo`.
    - Put the `_local_timestamp`/fallback cases in a new `TestLocalTimestamp(UnattendedTestBase)` class, and patch the module attribute with `patch.object(unattended, '_HOST_LOCALTIME_PATH', <Path>)`.
    - Run the module and confirm the new and changed tests fail. Do not commit a red test.

    GREEN. In `solsys_code/unattended.py`:
    - Add `from zoneinfo import ZoneInfo` and import `tzinfo` from `datetime`, alongside the existing datetime imports.
    - Add a module constant `_HOST_LOCALTIME_PATH = Path('/etc/localtime')` next to the other private constants. Give it a comment explaining why the file is read directly, per DEC-2 and the planning-time verification in this plan's objective: Django's settings loader sets the process TZ environment variable to `settings.TIME_ZONE` (`'UTC'`) and calls `time.tzset()`, so the process's own local-time conversion reports UTC, not the host's zone.
    - Add private `_host_timezone() -> tzinfo`. It opens `_HOST_LOCALTIME_PATH` in binary mode and returns `ZoneInfo.from_file(fh)`. On ANY exception it returns `dt_timezone.utc`, via a bare `except Exception:` carrying the module's existing `# noqa: BLE001 -- <reason>` comment style. The reason is that a truncated TZif file can raise `struct.error`, which is not an `OSError` or `ValueError`. Its docstring must say why it can never raise: the START banner is written inside the run lock before the first step, so an exception there would skip every step of the tick, and a legibility feature must never do that. Do not add caching; the file is read twice per tick.
    - Add private `_local_timestamp(moment: datetime) -> str`. It returns `moment.astimezone(_host_timezone()).isoformat(timespec='seconds')`. It must NOT call `datetime.now()` itself: `TestEndBannerTimestamp`'s fake clock supplies exactly two `now()` values, and `run_tick()` already samples both. Always pass the explicit tz argument, never the no-argument conversion; see the objective's timezone pitfall.
    - Change `_write_banner()` to take `(kind: str, now: datetime, *, exit_code: int | None = None, duration_seconds: int | None = None)`.
      - The START line becomes `'=== FOMO unattended run START %s ==='` with `_local_timestamp(now)`.
      - The END line becomes `'=== FOMO unattended run END %s exit=%s duration=%ss ==='` with `_local_timestamp(now)`, `exit_code` and `duration_seconds`.
      - Keep both prefixes byte-identical (DEC-1).
      - Update its docstring: host-local time with offset, seconds precision, and what `duration=` measures.
    - In `run_tick()`:
      - Pass `duration_seconds=round((end_time - now).total_seconds())` to the END `_write_banner()` call (DEC-4). Leave `now` and `end_time` as the UTC-aware values they are today, and leave every other use of them unchanged. `decide_notification()` and `save_state()` keep receiving UTC `end_time`.
      - In the `except LockContended:` branch, build one stamped message: the existing text `run_unattended: lock held -- skipping this tick` followed by ` (` + `_local_timestamp(now)` + `)`. Use it for both the `sys.stderr.write(...)` (keeping its trailing newline) and the `logger.warning(...)`, the latter with `%s` formatting (DEC-5).
      - Add no new `datetime.now()` call anywhere in `run_tick()`.
      - Do not move `_write_banner('START', now)` relative to the lock, the heartbeat ping, or the steps. Lock handling and step order are out of scope.

    Run the test module and ruff until all pass, then commit tests and implementation together.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel && python manage.py test solsys_code.tests.test_unattended --exclude-tag=ephemeris_segfault && pre-commit run ruff --all-files && pre-commit run ruff-format --all-files</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "_HOST_LOCALTIME_PATH = Path('/etc/localtime')" solsys_code/unattended.py` matches exactly one line.
    - `grep -c 'duration=%ss' solsys_code/unattended.py` prints at least 1.
    - `grep -c 'lock held -- skipping this tick' solsys_code/unattended.py` prints at least 1, and both skip emissions use the one stamped message.
    - `python manage.py test solsys_code.tests.test_unattended.TestLocalTimestamp solsys_code.tests.test_unattended.TestEndBannerTimestamp solsys_code.tests.test_unattended.TestLocking --exclude-tag=ephemeris_segfault` reports OK.
    - A live check on this host (TZ America/Los_Angeles), `python manage.py run_unattended --dry-run 2>&1 | grep -E '=== FOMO unattended run (START|END) '`, prints a START line ending `-07:00 ===` (or `-08:00 ===` outside DST) and an END line containing `exit=` and `duration=`. If it prints the timestamped `run_unattended: lock held -- skipping this tick (...)` line instead, a cron tick is running at that moment. That line is itself a valid check of DEC-5; rerun after the tick finishes to see the banners.
  </acceptance_criteria>
  <done>START and END banners carry the host's local ISO-8601 time with its UTC offset, to the second. END also carries `duration=<N>s`. The in-process lock-skip line carries the same timestamp. A missing or corrupt `/etc/localtime` degrades to `+00:00` without raising. The state file and notification timing are unchanged. Tests are green, ruff is clean, and the work is committed.</done>
</task>

<task type="auto">
  <name>Task 3: update the paired runbook section and unattended troubleshooting entries to show and explain the new banner, duration, lock-skip and status-code output</name>
  <files>docs/runbooks/telescope_runs_calendar.rst</files>
  <read_first>
    - docs/runbooks/telescope_runs_calendar.rst lines 1454-2045 ("How do I run everything unattended?") and 2567-2740 (unattended troubleshooting entries)
    - solsys_code/unattended.py as changed by Tasks 1-2: `_exception_label`, `_host_timezone`, `_local_timestamp`, `_write_banner`, and `run_tick()`'s `LockContended` branch. Quote the output these actually produce, not this plan's paraphrase.
  </read_first>
  <action>
    This is the CLAUDE.md paired-docs rule. `solsys_code/unattended.py` and the `run_unattended`/`check_unattended` commands pair to this runbook section, not to a notebook, so no notebook is in scope. All example timestamps across this section and its troubleshooting entries should read as one Pacific-time host: use a `-07:00` offset (a `date -Is` / `_local_timestamp()` output on such a host), seconds precision, and no fractional seconds. Keep the page's existing RST conventions: `::` literal blocks indented three spaces under list items, double-backtick inline literals, and `^` underlines at least as long as their title. Make these edits:

    (a) Walkthrough step 7 (the dry-run whole-tick example, around lines 1870-1880):
      - Change the example's START and END banner lines to the new form, e.g. START `2026-09-22T07:00:00-07:00` and END `2026-09-22T07:00:03-07:00 exit=0 duration=3s`. The five step lines between them stay as they are.
      - In the paragraph after it, which explains `exit=`, add that the banner time is the host's own local time with its UTC offset, read from `/etc/localtime`, not from Django's `TIME_ZONE` (which is UTC). That is the same clock the cron guard's `date -Is` skip line uses, so the two kinds of line compare directly.
      - Also add that `duration=` is the tick's wall-clock seconds from START to END.

    (b) A short reading-the-log note in the same section, e.g. as a paragraph after the step-7 material or folded into "When nothing has appeared" item 2. It says that in the cron log each tick's START banner comes after a block of process-startup lines, and names them: a `Note: NumExpr detected ...` line and the thread-count line after it, a `Using fallback library next to module: ...libcspice.so` line, several `registering new views: ...` lines, and Django's system-check block (`System check identified some issues:` / `WARNINGS:` / `?: (urls.W005) URL namespace 'calendar' isn't unique ...`). It explains that these come from library imports and Django's own system check, go to stderr like the runner's own lines (hence `2>&1`), belong to the tick whose START banner follows them, and are not failures. Include one literal-block example of a FAILING tick, with lines in this order:
      - the START banner, e.g. `2026-09-26T11:45:02-07:00`;
      - `observation_id=4378036 HTTPError 502`;
      - `step status_refresh: FAILED | LCO: failed 1 | SOAR: failed 0 | classes: HTTPError 502`;
      - the remaining four `step ...: ok | ...` lines in registry order, which may be shortened to realistic summaries matching the step-7 example;
      - an END banner with `exit=1 duration=32s`.

    (c) "The two failure signals", the **Email** paragraph:
      - Add a literal-block example of a failed-step line as it appears in the email body: `- status_refresh: LCO: failed 1 | SOAR: failed 0 | classes: HTTPError 502`.
      - State that for an HTTP error from the portal, the `classes:` field (or `outage (...)` for a whole-facility failure, e.g. `LCO: outage (HTTPError 503)`) carries only the exception class and its numeric HTTP status code. It never carries the response body, the request URL or its query string, request/response headers, or the exception's message, and that is what keeps SCHED-10's no-credential guarantee.
      - State that a bare `HTTPError` with no number means the exception carried no response. Keep the paragraph's existing "never a traceback, a request URL, or portal response text" sentence true and consistent.

    (d) "When nothing has appeared", item 2: update "Every tick writes a START/per-step/END banner with a timestamp" to say the START/END banners carry host-local time with a UTC offset, and that END also carries `exit=` and `duration=`.

    (e) "When nothing has appeared", item 4, and the troubleshooting entry "Repeated 'lock held' lines in the unattended log":
      - Change both cron-guard example lines to the host-local form, e.g. `2026-09-17T08:00:03-07:00 run_unattended skipped: lock held`.
      - Quote the runner's internal-lock message with its new trailing timestamp, e.g. `run_unattended: lock held -- skipping this tick (2026-09-17T08:00:03-07:00)`.
      - Reword the "only the cron guard's line above (with the `run_unattended skipped:` prefix and a leading timestamp)" sentence. Both lines now carry a timestamp, so the distinguishing marks are the wording (`run_unattended skipped: lock held` versus `run_unattended: lock held -- skipping this tick`) and the timestamp's position: leading for the cron guard, trailing in parentheses for the runner.
      - In that troubleshooting entry's Cause paragraph, where it says to look for the tick's own START/END banner, add that the END banner's `duration=` shows how long the overrunning tick took, and that a value above 900 seconds (one 15-minute interval) confirms it overlapped the next scheduled tick.

    (f) Add one new `^^^^`-level troubleshooting entry immediately before "A failure email arrived once, then went quiet while the problem continued", titled `A status_refresh failure names an HTTP status code`. Follow the page's **Cause:** / **Fix:** shape.
      - **Cause:** status_refresh re-checks up to 20 failed records one by one and reports each as `observation_id=<id> <exception class> <HTTP status>`. The step line and email carry the de-duplicated labels (e.g. `classes: HTTPError 502, HTTPError 504`). TOM's facility code turns 401-403 into `ImproperCredentialsException` and 400 into `ValidationError`, so an `HTTPError <code>` is always some other 4xx (for example 404 or 429) or a 5xx.
      - **Fix:**
        - For 5xx (502/503/504): portal- or gateway-side trouble, usually transient. If later ticks succeed, a single recovery email follows and no action is needed. If it persists across many ticks, check the LCO portal's own status before suspecting FOMO.
        - For a 4xx: the portal rejected that particular request, so look up the named `observation_id` on the portal.
        - For `ImproperCredentialsException`: check the LCO API key configured for the host, and never paste the key into a ticket, a log excerpt or an email.
        - Cross-reference the email paragraph in (c) instead of restating it.

    Do not change any other section of the page. Do not change the documented cron line, lock files, heartbeat or email transport; this is legibility only.
  </action>
  <verify>
    <automated>cd /home/tlister/git/fomo_devel && test "$(grep -c 'HTTPError 502' docs/runbooks/telescope_runs_calendar.rst)" -ge 3 && test "$(grep -cE 'unattended run END [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}-07:00 exit=[01] duration=[0-9]+s ===' docs/runbooks/telescope_runs_calendar.rst)" -ge 2 && test "$(grep -cE 'unattended run START [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}-07:00 ===' docs/runbooks/telescope_runs_calendar.rst)" -ge 2 && test "$(grep -cE 'unattended run (START|END) [0-9T:.-]+\+00:00' docs/runbooks/telescope_runs_calendar.rst)" -eq 0 && grep -q 'lock held -- skipping this tick (' docs/runbooks/telescope_runs_calendar.rst && grep -q '^A status_refresh failure names an HTTP status code$' docs/runbooks/telescope_runs_calendar.rst && test "$(sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees -D 'exclude_patterns=notebooks/*,_build' 2>&1 | grep 'runbooks/telescope_runs_calendar.rst:.*WARNING' | grep -vc "unknown document: '/notebooks/")" -eq 0</automated>
  </verify>
  <acceptance_criteria>
    - The runbook contains at least 3 occurrences of `HTTPError 502` (the failing-tick log example twice, plus the email example).
    - At least 2 START and 2 END banner examples use the `-07:00` seconds-precision form, and every END example carries `duration=<N>s`.
    - No START/END banner example in the runbook still shows a `+00:00` offset.
    - The internal-lock message is quoted with its trailing parenthesised timestamp.
    - The new troubleshooting entry's title line exists, and its `^` underline is at least as long as the title.
    - The Sphinx build reports zero warnings for `runbooks/telescope_runs_calendar.rst` other than the pre-existing `unknown document: '/notebooks/pre_executed/campaign_lifecycle_demo'` one. That warning exists at planning time only because the pre-commit Sphinx configuration excludes `notebooks/*`, and it is unrelated to this plan.
    - Every example line quoted in the runbook matches what Tasks 1-2's code actually emits, character for character, apart from the illustrative values.
  </acceptance_criteria>
  <done>The paired runbook section and its unattended troubleshooting entries show the host-local START/END banners with `duration=`, the timestamped internal lock-skip line, and the `HTTPError <code>` form in the log and email. They explain where the pre-banner startup lines come from, and how to act on a 5xx, a 4xx, or a credentials failure. The page builds with no new Sphinx warnings, and the change is committed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| LCO/SOAR portal -> FOMO runner | Untrusted HTTP responses. TOM's `make_request()` wraps them in exceptions whose message embeds the request URL (401-403: the response body). |
| FOMO runner -> `/var/log/fomo/unattended.log` | The log file is mode `-rw-r--r--` on the host (world-readable; checked at planning time), so anything written there is disclosed to every local account. |
| FOMO runner -> staff email | The failure/reminder email body quotes each failed step's summary verbatim. |
| Host filesystem (`/etc/localtime`) -> FOMO runner | A root-owned file, read on every tick. It is trusted for content but can be missing or corrupt on an unusual host or container. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-eqs-01 | Information disclosure | `_exception_label()` -> status_refresh log line, summary, email | high | mitigate | Only `exc.response.status_code` is read, and it is formatted only when it is an `int` (not a `bool`). The message, `str`/`repr`, body, URL/query string, headers, reason and request are never read or formatted. Proven by Task 1's end-to-end hygiene test: a real `requests.Response` seeded with the fake API key in its body, URL query string and `Authorization` header, run through `call_command('run_unattended')`, with log, stdout, stderr and email subject/body all asserted free of every seeded value, the URL, the reason phrase and `str(exc)`. The existing `TestCredentialHygiene` tests must stay green (SCHED-10). |
| T-eqs-02 | Information disclosure | `ImproperCredentialsException` (401-403) path | high | mitigate | The helper's scope is `requests.exceptions.RequestException` only. `ImproperCredentialsException`, whose message embeds the response body, stays class-name-only, asserted by a Task 1 helper-level test. |
| T-eqs-03 | Spoofing / Tampering | A response-like object whose `status_code` is a non-int (a mock, a string carrying data) | medium | mitigate | The `isinstance(..., int)` guard falls back to the bare class name, asserted by the `HTTPError(response=MagicMock())` test. |
| T-eqs-04 | Denial of service | `_host_timezone()` inside `run_tick()`'s lock, before the first step | medium | mitigate | A broad `except Exception` falls back to UTC, so a missing, unreadable or truncated `/etc/localtime` can never raise out of the banner and skip a tick's steps. Asserted by Task 2's missing-file and corrupt-file tests. |
| T-eqs-05 | Information disclosure | The host timezone offset in a world-readable log | low | accept | The offset reveals only the host's configured zone, which `date -Is` already writes to the same file through the cron guard line. No credential or PII is involved. |
| T-eqs-06 | Repudiation | Tick timing in the log | low | accept | This plan improves traceability (local timestamps plus `duration=`). No audit-trail property is weakened, and the state file keeps UTC. |
| T-eqs-SC | Tampering | Package installs | low | accept | No npm/pip/cargo install in this plan. `zoneinfo` is stdlib, and `tzdata` is already installed (test-only resource read). |
</threat_model>

<verification>
Run from `/home/tlister/git/fomo_devel` after all three tasks:
- `python manage.py test solsys_code.tests.test_unattended solsys_code.tests.test_check_unattended --exclude-tag=ephemeris_segfault` reports OK. At planning time this was 74 + 48 tests in about 36s; there are more after this plan.
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` both report Passed. These are pinned via pre-commit per CLAUDE.md (D-07); do not use an unpinned `ruff`.
- `pre-commit run sphinx-build --all-files` reports Passed, and Task 3's Sphinx warning filter prints 0.
- The live dry run `python manage.py run_unattended --dry-run 2>&1 | grep -E '=== FOMO unattended run (START|END) '` shows host-local offsets and `duration=`. It never pings or mails.
- `git diff --stat` touches only `solsys_code/unattended.py`, `solsys_code/tests/test_unattended.py` and `docs/runbooks/telescope_runs_calendar.rst`. No crontab, lock, heartbeat, notification-transport or step-order change appears.
</verification>

<success_criteria>
- An operator reading `/var/log/fomo/unattended.log` can place any failure in local time from the tick's own START banner, which uses the same clock and offset as the cron guard's `date -Is` lines. They can also read each tick's duration from its END banner.
- A status_refresh failure email distinguishes a transient `HTTPError 502` from an `HTTPError 404` or `HTTPError 429` without anyone opening the log.
- SCHED-10 holds: all pre-existing credential-hygiene tests stay green, and the new end-to-end test proves that no response body, URL, query string, header, reason or message text escapes.
- The paired runbook section and troubleshooting entries match the new output exactly.
</success_criteria>

<output>
Create `.planning/quick/260927-eqs-timestamp-each-unattended-run-and-log-ht/260927-eqs-SUMMARY.md` when done
</output>
