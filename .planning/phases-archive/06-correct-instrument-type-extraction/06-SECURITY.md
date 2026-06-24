---
phase: 06
slug: correct-instrument-type-extraction
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-21
---

# Phase 06 Security Audit — Correct Instrument-Type Extraction

**Audited:** 2026-06-21
**ASVS Level:** 1
**Block on:** high
**Plan:** `.planning/phases/06-correct-instrument-type-extraction/06-01-PLAN.md`
**Implementation:** `solsys_code/management/commands/sync_lco_observation_calendar.py`, `solsys_code/tests/test_sync_lco_observation_calendar.py`

## Verdict: SECURED

**Threats Closed:** 2/2

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|--------------|------------|--------|
| T-06-01 | Denial of Service (availability) | `_extract_instrument` parsing `record.parameters` | mitigate | Whitelist-based `.get(...)`/membership checks; D-06 sentinel routed via dedicated `InstrumentExtractionError` to `extraction_failed`, never aborts the batch | closed |
| T-06-02 | Tampering (npm/pip/cargo installs) | dependency surface | accept | No new packages installed this phase — stdlib/existing imports only | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

## Threat Verification

| Threat ID | Category | Disposition | Evidence |
|-----------|----------|-------------|----------|
| T-06-01 | Denial of Service (availability) | mitigate | CLOSED — see below |
| T-06-02 | Tampering (npm/pip/cargo installs) | accept | CLOSED — see below |

### T-06-01 — `_extract_instrument` parsing `record.parameters` (mitigate)

Claimed mitigation: all `c_N_*` lookups use `.get(...)`/membership checks against the
fixed `_SCIENCE_CONFIGURATION_TYPES` whitelist, never blind indexing; a malformed/partial
shape returns the D-06 sentinel, is caught by the per-record catch-log-continue
convention, routed to a dedicated counter, and the run continues.

Verified directly against the implemented code, not the plan's intent:

- **No blind indexing.** `grep -n "\['c_\|\[f'c_"` over
  `sync_lco_observation_calendar.py` returns zero matches — every `c_N_*` read goes
  through `.get(...)`:
  - `_has_muscat_exposure_signal` (line 99): `parameters.get(f'c_{n}_ic_1_exposure_time_{suffix}')`
  - `_find_science_config` (line 114): `parameters.get(f'c_{n}_configuration_type')`
  - `_find_exposure_signal_config` (line 132): `parameters.get(f'c_{n}_exposure_time')`
  - `_extract_instrument` (line 162): `parameters.get(f'c_{n}_instrument_type')`
- **Whitelist membership, not blind indexing.** `_SCIENCE_CONFIGURATION_TYPES` (line 46)
  is a fixed module-level set `{'EXPOSE', 'REPEAT_EXPOSE', 'SPECTRUM', 'REPEAT_SPECTRUM',
  'STANDARD'}`; the only access pattern against it is `configuration_type in
  _SCIENCE_CONFIGURATION_TYPES` (line 115) — pure membership test, ARC/LAMP_FLAT/
  NRES_SPECTRUM never recognized.
- **D-06 sentinel routed via dedicated exception, not absorbed into a generic catch.**
  `_extract_instrument` returns `None` on total failure (line 163); `_build_event_fields`
  converts that into `InstrumentExtractionError` (lines 245-249), a dedicated exception
  class (line 218) distinct from the pre-existing `except (KeyError, ValueError)` block.
  `handle()`'s per-record loop (lines 365-376) has a dedicated `except
  InstrumentExtractionError` clause that logs `observation_id` to stderr and increments
  `counters[record.facility]['extraction_failed']`, then `continue`s the loop — the
  malformed record never propagates past the per-record `try`, and the `for record in
  records:` loop keeps processing subsequent records regardless.
- **Dedicated counter wired into all 4 locations**, confirmed by direct read of
  `handle()`: both eager dict literals (lines 340-341), the `setdefault` defensive
  default (lines 358-361, for the unrelated unrecognized-facility path), the per-record
  except-clause increment (line 371), and the summary `' | '.join(...)` f-string
  (line 399) — `extraction_failed: {count}` appears in the final stdout summary.
- **Test evidence, not just code-shape inspection.** Ran
  `python manage.py test solsys_code.tests.test_sync_lco_observation_calendar` — all 22
  tests pass (19 pre-existing regression tests + 3 new), including
  `test_d06_no_extractable_config_logged_and_counted_separately`, which asserts: (a) a
  fully-malformed record (only an `ARC` calibration config, no exposure signal, no
  flat `instrument_type`) does NOT create a `CalendarEvent` and does NOT crash the run,
  (b) its `observation_id` (`710005`) appears in stderr, (c) `extraction_failed: 1`
  appears in the stdout summary, and (d) a coexisting well-formed record
  (`710006`) still produces its `CalendarEvent` in the same run — proving one malformed
  record cannot abort the whole sync.

**Disposition:** CLOSED. Evidence: `solsys_code/management/commands/sync_lco_observation_calendar.py:46,99,114,115,132,162-163,218-249,340-341,358-371,399`; `solsys_code/tests/test_sync_lco_observation_calendar.py:561-591`; test run output (22/22 passing).

### T-06-02 — npm/pip/cargo installs (accept)

Claimed disposition: accept — no new packages installed by this phase; uses only stdlib
dict access and existing imports.

Verified by checking the actual commits attributed to this phase, not just the plan's
stated intent:

- `git show aaf4d1f --stat` (Task 1 commit) touches only
  `solsys_code/tests/test_sync_lco_observation_calendar.py` (118 insertions, 0 deletions).
- `git show 5e1489c --stat` (Task 2 commit) touches only
  `solsys_code/management/commands/sync_lco_observation_calendar.py` (130 insertions,
  11 deletions).
- Neither commit touches `pyproject.toml`, any lockfile, or any requirements file.
- The module's import block (lines 1-9) is unchanged from before the phase: `datetime`,
  `typing.Any` (stdlib), `django.core.management.base`, `tom_calendar.models`,
  `tom_observations.facilities.{lco,soar}`, `tom_observations.models` — all pre-existing
  Django/TOM imports, no new third-party package.

An `accept` disposition requires only that the accepted-risk entry exists and that no
contradicting evidence (e.g. an undisclosed new dependency) is found. Both hold here.

**Disposition:** CLOSED. Evidence: `git show aaf4d1f --stat`, `git show 5e1489c --stat` (file lists confirm no `pyproject.toml`/lockfile changes); `solsys_code/management/commands/sync_lco_observation_calendar.py:1-9` (import block unchanged).

## Unregistered Flags

None. `06-01-SUMMARY.md` has no `## Threat Flags` section, and no new attack surface
was found during verification beyond the two registered threats. The one documented
deviation in SUMMARY.md (the added flat-`instrument_type` fallback tier for the legacy
single-config shape) is a correctness fix to satisfy the plan's own regression
requirement, not new attack surface — it adds one more `.get('instrument_type')` read
(line 163), which follows the same `.get(...)`-only discipline already covered by
T-06-01's verification.

## Additional Note (informational, not a finding)

`_derive_telescope` (lines 71-86) still raises a bare `KeyError` on an unmapped site
code, which is caught by the pre-existing generic `except (KeyError, ValueError)` block
(line 373) and routed to `'skipped'`, not `'extraction_failed'`. This is outside this
phase's declared scope (the threat register and PLAN.md scope `_extract_instrument`
specifically; `_derive_telescope`/`SITE_TELESCOPE_MAP` are explicitly deferred to
Phase 7 per `06-01-SUMMARY.md` "Next Phase Readiness"). No action required for Phase 6;
noted for completeness only.

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-06-01 | T-06-02 | No new packages installed by this phase — confirmed against actual commits (`aaf4d1f`, `5e1489c`); import block unchanged | gsd-security-auditor | 2026-06-21 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-21 | 2 | 2 | 0 | gsd-security-auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-21

---

**Result: SECURED — 2/2 threats CLOSED, 0 open, 0 unregistered flags.**
