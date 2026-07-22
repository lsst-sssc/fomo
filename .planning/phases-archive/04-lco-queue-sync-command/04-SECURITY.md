---
phase: 04
slug: lco-queue-sync-command
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-17
---

# Phase 04 — LCO Queue Sync Command — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Operator shell → command | `--proposal <code>` is free-text CLI input used directly in an ORM filter | Operator-supplied string, single-operator local tool |
| `ObservationRecord.parameters` (third-party-populated JSON) → `CalendarEvent` fields | `parameters` values (`proposal`/`start`/`end`/`instrument_type`/`site`) are populated by the TOM submission layer and flow into `CalendarEvent` title/description/times/url | Untrusted/external-sourced JSON keys and values |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-04-01 | Tampering (injection) | `--proposal` → `parameters__proposal=code` ORM filter | accept | Django ORM parameterizes JSONField key-value lookups (value bound, not string-interpolated); no hand-rolled escaping required | closed (accepted) |
| T-04-02 | Tampering / Information Disclosure | `parameters['start']`/`['end']`/`['site']`/`['instrument_type']` → `CalendarEvent` fields | mitigate | Per-record `try/except (KeyError, ValueError)` skip-and-log path; `datetime.fromisoformat` rejects malformed time strings into the skip path | closed |
| T-04-03 | Tampering | `url` field used as create-or-update lookup key | mitigate | `url` built exclusively via `LCOFacility().get_observation_url(record.observation_id)`; no `requestgroups` literal present in source | closed |
| T-04-04 | Repudiation | Silent no-op run (zero matches) appears successful | mitigate | Unconditional `self.stdout.write(...)` summary outside the loop, always includes `created: {n}` even when `n == 0`; no `CommandError` raised on zero matches | closed |
| T-04-05 | Denial of Service (low) | Very large matching result set | accept | Single-operator CLI tool, local SQLite, bounded by one proposal's records; no untrusted remote trigger | closed (accepted) |
| T-04-SC | Tampering | npm/pip/cargo installs | accept | No new packages installed this phase — all classes (`LCOFacility`, `CalendarEvent`, `ObservationRecord`) imported from already-installed `tomtoolkit`; no new dependency surface | closed (accepted) |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

### Verification Evidence (mitigate threats)

| Threat ID | Evidence |
|-----------|----------|
| T-04-02 | `solsys_code/management/commands/sync_lco_observation_calendar.py:199-202` — `except (KeyError, ValueError) as exc: self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}'); skipped_count += 1; continue`, wrapping the single call to `_build_event_fields(record, facility)` (line 198) which is the only path that reads `parameters['site']`/`['instrument_type']`/`['proposal']` (line 138-140) and parses `parameters['start']`/`['end']` via `datetime.fromisoformat` (line 109-110, inside `_time_window`). One entry point (the per-record loop, line 196), fully wrapped. Confirmed by passing tests `test_skip_path_missing_site_logged_and_skipped` and `test_skip_path_inconsistent_scheduled_times_logged_and_skipped` (independently re-run during audit: both `ok`). |
| T-04-03 | `solsys_code/management/commands/sync_lco_observation_calendar.py:141` — `url = facility.get_observation_url(record.observation_id)` inside `_build_event_fields`, the only site that constructs `url`; this is the sole value used as the `get_or_create(url=url, ...)` lookup key (line 205). `grep -n requestgroups` on the file returns no matches — no hardcoded/derivable-from-parameters URL path exists. Confirmed by passing test `test_sync_01_d01_url_uses_requests_path_not_requestgroups` (independently re-run: `ok`), which asserts `event.url == LCOFacility().get_observation_url(observation_id)` and `'requestgroups' not in event.url`. |
| T-04-04 | `solsys_code/management/commands/sync_lco_observation_calendar.py:218-224` — `self.stdout.write(...)` is the last statement in `handle()`, outside and after the `for record in records:` loop (line 196-216), so it executes unconditionally regardless of how many records matched, including zero. The f-string includes `f'created: {created_count}'` with `created_count` initialized to `0` (line 189) and never gated behind a non-zero check. Confirmed by passing test `test_zero_match_reports_created_zero_no_command_error` (independently re-run: `ok`), which asserts `'created: 0' in stdout_buf.getvalue()` and that no exception/CommandError is raised. |

All three `mitigate` threats verified against the actual entry point (the single per-record loop in `Command.handle`) — there is only one call path into `_build_event_fields` and only one `stdout.write` call, so single-grep-match coverage is complete coverage here (no additional entry points to check).

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-04-01 | T-04-01 | `--proposal` flows into `ObservationRecord.objects.filter(parameters__proposal=proposal)`. Django's ORM binds JSONField key-value lookups as query parameters at the DB-API level; the value is never string-interpolated into raw SQL. No additional escaping is required for ASVS V5 (Level 1) on this code path. Risk accepted as framework-mitigated, not independently re-verified by this audit (per plan disposition + audit constraints). | Phase 04 plan (PLAN.md `<threat_model>`) | 2026-06-17 |
| AR-04-02 | T-04-05 | Command is a single-operator, locally-run Django management command against local SQLite, with no network-facing trigger and no untrusted remote caller. Result-set size is bounded by one proposal's `ObservationRecord` rows. Likelihood and impact of a DoS via large result set are both low; no rate limiting or pagination added. | Phase 04 plan (PLAN.md `<threat_model>`) | 2026-06-17 |
| AR-04-03 | T-04-SC | This phase introduced no new third-party packages. `LCOFacility`, `CalendarEvent`, `ObservationRecord` are all imported from `tom_observations`/`tom_calendar`, which ship as part of the already-installed `tomtoolkit` dependency (confirmed: no `pyproject.toml`/lockfile diff for this phase). No supply-chain legitimacy checkpoint applies. | Phase 04 plan (PLAN.md `<threat_model>`) | 2026-06-17 |

*Accepted risks do not resurface in future audit runs.*

---

## Threat Flags (from SUMMARY.md)

SUMMARY.md `## Threat Flags` section states: "None. All three threat-register items disposed `mitigate` ... are implemented as specified ... No new network endpoints, auth paths, or schema changes were introduced."

No unmapped/unregistered new attack surface flagged by the executor. No `unregistered_flag` entries required.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-17 | 6 | 6 | 0 | gsd-security-auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-17
