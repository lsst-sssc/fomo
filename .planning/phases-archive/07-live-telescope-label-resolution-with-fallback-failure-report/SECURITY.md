# SECURITY.md — Phase 07: Live Telescope Label Resolution with Fallback / Failure Report

**ASVS Level:** 1
**Block on:** high
**Audit date:** 2026-06-24 (re-audit; original audit 2026-06-23)
**Threats closed:** 7/7
**Threats open (BLOCKER):** 0

This document is produced by retroactive security audit of the implemented
Phase 07 code (`solsys_code/management/commands/sync_lco_observation_calendar.py`,
`solsys_code/tests/test_sync_lco_observation_calendar.py`) against the threat
registers declared in `07-01-PLAN.md` and `07-02-PLAN.md`. Implementation files
were not modified by this audit.

**Re-audit note (2026-06-24):** the original audit (2026-06-23) found T-07-03
OPEN (BLOCKER). Quick task `260623-ocs` (commits `3fc6554` fix + `2fa0300`
test) applied the required fix. This re-audit re-verifies T-07-03 only against
the current code, plus a sanity check that the other 6 threats' mechanisms are
unchanged. See "Audit Trail" at the bottom for the full history.

## Threat Verification

| Threat ID | Category | Disposition | Status | Evidence |
|-----------|----------|-------------|--------|----------|
| T-07-01 | Information Disclosure | mitigate | CLOSED | `sync_lco_observation_calendar.py:174` — `_resolve_placement_block`'s except clause (`except (requests.exceptions.RequestException, ImproperCredentialsException, forms.ValidationError, ValueError):`) returns `None` with zero reference to the caught exception (no `as exc`, no logging). Re-confirmed unchanged in 2026-06-24 re-audit — quick task `260623-ocs` did not touch this function's except clause. `test_sync_lco_observation_calendar.py:716-738` (`test_sync_09_no_credential_or_body_leak_in_logs`) drives both `ImproperCredentialsException` and `forms.ValidationError` with an embedded leak marker and asserts the helper returns `None`. |
| T-07-02 | Denial of Service | mitigate | CLOSED | `sync_lco_observation_calendar.py:44,164,171` — single `make_request(...)` call with `timeout=_API_TIMEOUT_SECONDS` (=10), no loop/retry/backoff in the function body. Re-confirmed unchanged in 2026-06-24 re-audit. `test_sync_lco_observation_calendar.py:701-714` (`test_sync_08_single_attempt_no_retry`) patches `make_request` with `side_effect=requests.exceptions.Timeout` and asserts `mock_make_request.assert_called_once()`. |
| T-07-03 | Denial of Service / Tampering | mitigate | **CLOSED (re-verified 2026-06-24)** | See "T-07-03 Re-Verification (2026-06-24)" below. |
| T-07-SC | Tampering (supply chain) | accept | CLOSED (logged below) | "Accepted Risks" section of this document. Quick task `260623-ocs` introduced no new packages (confirmed: only edits to the two files already covered by the existing accepted-risk entry). |
| T-07-04 | Information Disclosure | mitigate | CLOSED | `sync_lco_observation_calendar.py:576-584` — `Command.handle()`'s fallback stderr line is a fixed f-string interpolating only `record.observation_id!r`; no exception variable is referenced on that line. Re-confirmed unchanged in 2026-06-24 re-audit. `test_sync_lco_observation_calendar.py:894-924` (`test_sync_09_log_line_is_fixed_generic_message`) raises `ImproperCredentialsException` with an embedded `leak_marker`, asserts the marker is absent from captured stderr and `observation_id` ('800106') and 'fallback' are present. |
| T-07-05 | Information Disclosure | mitigate | CLOSED | `sync_lco_observation_calendar.py:435` — the fallback description note is the fixed literal string `'\nTelescope label unverified: live API lookup failed or returned an unmapped code.'`, never built from a caught exception. Re-confirmed unchanged in 2026-06-24 re-audit. `test_sync_lco_observation_calendar.py:770-818` (`test_telescope_04_fallback_label_visibly_distinguishable`) asserts `'unverified' in event.description.lower()` for the fallback case. |
| T-07-06 | Denial of Service | mitigate | CLOSED | `_resolve_placement_block` never raises (returns `None` on every failure path); `_build_event_fields` routes `block is None` to the coarse-fallback branch rather than propagating, and (post-fix) also routes a present-but-incomplete block to the same fallback branch via `.get()`. Re-confirmed unchanged in 2026-06-24 re-audit (mechanism untouched by the T-07-03 fix; the fix only widened the set of inputs that correctly reach this fallback). `test_sync_lco_observation_calendar.py:849-892` (`test_sync_07_api_failure_does_not_abort_run`) drives two placed records where the first-processed record's `make_request` call raises `Timeout`, and asserts both records still produce `CalendarEvent` rows (`CalendarEvent.objects.count() == 2`) with no exception propagating out of `call_command`. |

## T-07-03 Re-Verification (2026-06-24)

**Declared mitigation (07-01-PLAN.md threat register, restated by quick-task
`260623-ocs` threat register):**
> Read block fields via `block.get('site')`/`block.get('telescope')` (not
> `[]`) so a missing key yields None and routes to the existing
> coarse-fallback (Pitfall-4) bucket instead of raising KeyError into the
> generic `skipped` counter. None-guard `_aperture_class_from_telescope_code`
> so a None telescope_code cannot raise TypeError. Regression test asserts
> fallback, not skip.

**Verification performed (current code, post-fix, commits `3fc6554` +
`2fa0300`):**

1. **`.get()` at point of consumption — CONFIRMED.**
   `sync_lco_observation_calendar.py:411`:
   ```python
   resolved = _derive_telescope(block.get('site'), block.get('telescope')) if block is not None else None
   ```
   `grep -n "block\['"` against the whole file returns zero matches — no
   bracket indexing of the resolved block remains anywhere in the module.

2. **`_aperture_class_from_telescope_code` and `_derive_telescope` are
   genuinely None-safe — CONFIRMED by reading the guard, not the docstring.**
   `sync_lco_observation_calendar.py:108-112`:
   ```python
   def _aperture_class_from_telescope_code(telescope_code: str | None) -> str | None:
       ...
       if not telescope_code:
           return None
       if len(telescope_code) >= 4 and telescope_code[:3] in {'0m4', '1m0', '2m0', '4m0'}:
           return telescope_code[:3]
       return None
   ```
   The `if not telescope_code: return None` guard runs *before* the
   `len(telescope_code)` call, so a `None` argument cannot reach `len(None)`
   (which would raise `TypeError`). `_derive_telescope` (line 115) calls this
   helper unconditionally and then does
   `SITE_TELESCOPE_MAP.get((site, aperture_class))` — a dict `.get()` with a
   `None` site in the tuple key simply misses and returns `None`; no code path
   in either function raises for a `None` `site` or `None` `telescope_code`.

3. **Regression test exists and asserts the claimed behavior — CONFIRMED.**
   `test_sync_lco_observation_calendar.py:955-992`,
   `test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped`:
   builds `malformed_response.json.return_value = [{'state': 'COMPLETED', 'telescope': '1m0a'}]`
   (no `'site'` key — a syntactically valid COMPLETED block missing the field
   T-07-03 describes), patches `make_request` to return it, runs the full
   `sync_lco_observation_calendar` command via `call_command`, and asserts:
   - `CalendarEvent.objects.count() == 1` (record is NOT skipped)
   - `event.telescope == '1m0'` (coarse fallback label)
   - `event.title.startswith('[UNVERIFIED]')`
   - captured stdout contains `'telescope_api_failed: 1'` and `'skipped: 0'`

   These are exactly the assertions the declared mitigation requires (fallback
   bucket, not skip bucket) — not a weaker substitute.

4. **Empirical confirmation, not reading-only.** Ran
   `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar -v 1`
   in this audit: **34 tests, all green** (`Ran 34 tests in 0.751s / OK`),
   including the new regression test. This is direct evidence, not inference
   from code structure.

**Disposition:** CLOSED. The declared mitigation pattern (`.get()` field
access + None-safe downstream helpers + regression test proving fallback, not
skip) is present in the code exactly as described, at the actual point of
consumption, and is exercised by a passing test.

**Sanity check on other 6 threats (no regression):** `git show --stat
3fc6554` confirms the diff touched only
`solsys_code/management/commands/sync_lco_observation_calendar.py` (24
insertions, 13 deletions — the `.get()` line, the
`_aperture_class_from_telescope_code` guard, and docstring updates); `git show
--stat 2fa0300` confirms the test-file diff is a pure addition (39 insertions,
0 deletions). The except-clause mechanisms for T-07-01/T-07-04 (no exception
variable referenced), the single-attempt/timeout mechanism for T-07-02, the
fixed-string fallback description for T-07-05, and the never-raises contract
of `_resolve_placement_block` for T-07-06 were independently re-grepped in
this audit and are byte-for-byte unchanged from the original 2026-06-23 audit
evidence (line numbers shifted slightly due to the fix's insertions but the
code itself is identical).

## Accepted Risks Log

| Threat ID | Category | Disposition | Rationale | Approved |
|-----------|----------|-------------|-----------|----------|
| T-07-SC | Tampering (supply chain) | accept | No new third-party packages were installed in Phase 07. All new imports (`requests`, `django.forms`, `tom_common.exceptions.ImproperCredentialsException`, `tom_observations.facilities.ocs.make_request`, `urllib.parse.urljoin`) resolve to packages already present in the existing dependency set (per `07-01-PLAN.md`/`07-02-PLAN.md` Package Legitimacy Audit: "not applicable — `requests`/`tomtoolkit`/Django already installed dependencies"). Verified: no diff to `pyproject.toml` / lockfile in either phase's `files_modified` list. | Phase-time (plan-authored); logged retroactively by this audit since no prior SECURITY.md existed for this phase. |

## Unregistered Flags

No `## Threat Flags` section was present in either `07-01-SUMMARY.md` or
`07-02-SUMMARY.md` — the executors did not flag any new attack surface during
implementation.

One informational item was independently identified during this audit and is
recorded here for completeness (not a blocker, since it is pre-existing
surface, not new surface introduced by this phase):

- **`record.observation_id` interpolated into the API request path**
  (`urljoin(..., f'/api/requests/{observation_id}/observations/')`,
  `sync_lco_observation_calendar.py:159-162`) and into log/description strings.
  `observation_id` originates from `ObservationRecord.observation_id`, a field
  populated by a prior LCO sync, not directly from end-user input at this
  command's boundary — the same trust assumption the pre-existing
  `facility.get_observation_url(record.observation_id)` call (line 412,
  unchanged by this phase) already makes. No path-traversal or injection
  concern beyond what already existed before Phase 07. Logged here for
  visibility, not as a new finding requiring remediation in this phase.

## Trust Boundaries (from PLAN.md, both plans)

| Boundary | Description |
|----------|--------------|
| sync command -> LCO Observation Portal API | Untrusted remote response (JSON body, HTTP status, latency) crosses into the command via `make_request`, consumed by `_resolve_placement_block` and `_build_event_fields`. |
| credential store -> log/stdout | `LCO_APIKEY` must never cross into any logged/returned string. |
| caught API exception -> stderr / CalendarEvent.description | The failure log line and the fallback description must never carry the response body or API key. |

## Verification Commands Used

Original audit (2026-06-23):
```
grep -n "block\['site'\]\|block\['telescope'\]\|block.get(" solsys_code/management/commands/sync_lco_observation_calendar.py
grep -n "def test_telescope_01\|def test_telescope_02\|def test_telescope_03\|def test_telescope_04\|def test_sync_08\|def test_sync_09\|def test_sync_06\|def test_sync_07\|def test_d01_banner" solsys_code/tests/test_sync_lco_observation_calendar.py
```

Re-audit (2026-06-24), against post-fix code:
```
grep -n "_aperture_class_from_telescope_code\|^def _derive_telescope\|SITE_TELESCOPE_MAP" solsys_code/management/commands/sync_lco_observation_calendar.py
grep -n "block\['" solsys_code/management/commands/sync_lco_observation_calendar.py   # zero matches
grep -n "def test_telescope_03_block_missing" -A 60 solsys_code/tests/test_sync_lco_observation_calendar.py
git show --stat 3fc6554
git show --stat 2fa0300
python manage.py test solsys_code.tests.test_sync_lco_observation_calendar -v 1   # 34 tests, OK
```

## Audit Trail

| Date | Action | Result |
|------|--------|--------|
| 2026-06-23 | Original audit | 6/7 closed; T-07-03 OPEN (BLOCKER) — `_build_event_fields` read the resolved block via bracket indexing, raising `KeyError` (misrouted to `skipped`) on a COMPLETED/PENDING block missing `'site'`/`'telescope'`, instead of the declared `.get()`-based fallback routing. |
| 2026-06-23/24 | Quick task `260623-ocs` | Fixed T-07-03: `block.get('site')`/`block.get('telescope')` at the consumption point (`sync_lco_observation_calendar.py:411`), None-guard added to `_aperture_class_from_telescope_code`, regression test added (`test_telescope_03_block_missing_site_or_telescope_falls_back_not_skipped`). Commits `3fc6554` (fix), `2fa0300` (test). |
| 2026-06-24 | Re-audit (this document) | T-07-03 re-verified CLOSED: `.get()` access confirmed (zero bracket-indexing matches remain), None-safety of `_aperture_class_from_telescope_code`/`_derive_telescope` confirmed by reading the actual guard code (not docstring), regression test confirmed to exist and assert the exact claimed behavior, full 34-test suite run and passing. Other 6 threats sanity-checked via `git show --stat` (diff scope confirmed minimal) and re-grep of unchanged mechanisms — no regression. **Phase 07 security gate fully resolved: 7/7 closed, 0 open.** |

## Next Steps

None. All 7 threats are CLOSED. No further action required for this phase's
security gate.
