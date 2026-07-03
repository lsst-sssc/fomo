---
phase: 14-campaign-data-model-bootstrap-import
reviewed: 2026-07-03T00:00:00Z
depth: deep
files_reviewed: 8
files_reviewed_list:
  - solsys_code/migrations/0002_campaignrun.py
  - solsys_code/tests/test_campaign_models.py
  - solsys_code/models.py
  - solsys_code/campaign_utils.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/tests/test_import_campaign_csv.py
  - docs/notebooks/pre_executed/fixtures/campaign_sample.csv
  - docs/notebooks/pre_executed/import_campaign_csv_demo.ipynb
findings:
  critical: 2
  warning: 9
  info: 2
  total: 13
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-07-03T00:00:00Z
**Depth:** deep
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed the `CampaignRun` model/migration, `campaign_utils.py`'s site-resolution/time-parsing/status-mapping/create-or-update helpers, the `import_campaign_csv` management command, their test suites, and the paired demo notebook + synthetic CSV fixture. All 26 Django tests in `solsys_code/tests/test_campaign_models.py` and `solsys_code/tests/test_import_campaign_csv.py` pass, `ruff check`/`ruff format --check` are clean on these files, and the model/migration are in sync field-for-field.

However, tracing the two "never raise, never fabricate, never silently merge distinct rows" design invariants stated in the module's own docstrings against actual behavior (confirmed by direct reproduction, not just reading) turned up two confirmed data-correctness bugs that violate those invariants:

1. `parse_obs_window`'s regexes capture `am`/`pm` markers but never apply them — a "PM" UT time silently parses as if it were "AM" (12 hours off), with no error and no flag.
2. The documented natural key `(campaign, telescope_instrument, ut_start)` collides whenever `UT Time Range` is blank or unparseable for two distinct rows sharing the same telescope and date, because the fallback always resolves to the same midnight timestamp — the second row silently overwrites or "unchanges" into the first, losing a real observation record.

Several further gaps in `resolve_site`'s exception handling, `map_observation_status`'s substring matching, and the lack of a DB-level uniqueness constraint backing the natural key round out the warnings below.

## Critical Issues

### CR-01: PM/AM time-of-day markers are matched but never applied — UT times silently parsed 12 hours wrong

**File:** `solsys_code/campaign_utils.py:35-36, 140-157`
**Issue:** `_HHMM_RANGE` and `_APPROX_HOUR` both include an optional `(?:am|pm)?` capture group, and the docstring explicitly cites `'~7:00:00 AM'` as a real, verified format from the source sheet (`RESEARCH.md`). But neither `match.group()` for the am/pm marker is ever read in `parse_obs_window` — the parsed hour is used as-is, effectively always treated as 24-hour/AM. Confirmed by direct reproduction:

```
>>> parse_obs_window('2025-07-16', '~7:00:00 PM')
(..., datetime(2025, 7, 16, 7, 0, tzinfo=utc), None)   # should be hour=19
>>> parse_obs_window('2025-07-16', '08:50 pm - 11:50 pm')
(..., datetime(2025, 7, 16, 8, 50, ...), datetime(2025, 7, 16, 11, 50, ...))  # should be 20:50-23:50
```

This directly contradicts the module's own stated design principle ("never ... succeeds into a wrong-but-plausible time") — a PM entry doesn't fail or get flagged, it silently produces a wrong-but-plausible UTC time 12 hours off. No test exercises a PM input, so this is unexercised in CI.

**Fix:** Capture and apply the am/pm marker in all three regex paths, e.g.:
```python
_HHMM_RANGE = re.compile(
    r'(\d{1,2})[:;](\d{2})\s*(am|pm)?\s*-\s*(\d{1,2})[:;](\d{2})\s*(am|pm)?', re.IGNORECASE
)

def _to_24h(hour: int, meridiem: str | None) -> int:
    if not meridiem:
        return hour
    meridiem = meridiem.lower()
    if meridiem == 'am':
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12
```
and use `_to_24h(h1, m1_marker)` / `_to_24h(h2, m2_marker)` when building `start`/`end`. If am/pm can't be reliably disambiguated for the range case, prefer flagging the row (analogous to `site_needs_review`) over silently guessing.

### CR-02: Natural key collides for distinct rows when UT Time Range is unparseable — one CampaignRun silently overwrites/absorbs the other

**File:** `solsys_code/campaign_utils.py:159-162` (fallback), `solsys_code/management/commands/import_campaign_csv.py:101-104` (natural key usage), `solsys_code/campaign_utils.py:211-220` (`insert_or_create_campaign_run`)
**Issue:** `parse_obs_window`'s fallback path (blank or unparseable `ut_range_raw`) always returns `datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0, tzinfo=utc)` regardless of the actual (unparseable) text. Confirmed:

```
>>> parse_obs_window('2025-07-06', '')
(..., datetime(2025, 7, 6, 0, 0, tzinfo=utc), None)
>>> parse_obs_window('2025-07-06', 'redo later')
(..., datetime(2025, 7, 6, 0, 0, tzinfo=utc), None)
>>> parse_obs_window('2025-07-06', 'some garbled text, no time info')
(..., datetime(2025, 7, 6, 0, 0, tzinfo=utc), None)
```

The import command's D-04 natural key is `(campaign, telescope_instrument, ut_start)` (`import_campaign_csv.py:101-104`). If a real sheet has two distinct rows for the same telescope on the same date, both with a blank/garbled `UT Time Range` (a realistic scenario — the sample fixture's own row 5 comment says "exact start time not logged"), both rows resolve to the identical natural key. `insert_or_create_campaign_run` (`campaign_utils.py:211-220`) will then treat the second row as an update-in-place (or a no-op "unchanged") of the first via `get_or_create`, **silently discarding one of the two real observation records** rather than creating two rows or flagging the collision. This is a genuine data-loss risk for a tool whose entire purpose is to replace an error-prone spreadsheet with a reliable coordination record — no test covers two same-telescope/same-date rows with unparseable times.

**Fix:** Don't let the un-parseable-time fallback collide silently. Options: (a) disambiguate the fallback timestamp by combining it with a distinguishing signal (e.g. a stable hash of the raw row content, stored in a dedicated field, folded into the natural key) instead of a fixed midnight; or (b) detect the collision in the command before calling `insert_or_create_campaign_run` — if a lookup key repeats within the same import batch, log it distinctly (similar to `site_needs_review`) so the operator knows two rows were merged, e.g.:
```python
seen_keys = set()
...
key = (campaign.pk, telescope_instrument, ut_start)
if key in seen_keys:
    self.stderr.write(f'Row {row_num}: WARNING duplicate natural key {key} within this import; row may be merged')
seen_keys.add(key)
```
At minimum, surface this as a counted/flagged outcome rather than a silent merge.

## Warnings

### WR-01: `resolve_site`'s Tier 2 MPC lookup has no timeout and doesn't catch network exceptions

**File:** `solsys_code/campaign_utils.py:94-99`
**Issue:** `fetcher.query(code)` (→ `solsys_code/solsys_code_observatory/utils.py:43`, `requests.get(...)` with no `timeout=`) can hang indefinitely on an unresponsive MPC API, and any `requests.exceptions.RequestException`/`ConnectionError`/`Timeout` raised inside `query()` is not caught anywhere in `resolve_site`. Since `import_campaign_csv` calls `resolve_site` once per CSV row inside a synchronous loop, a single flaky network call can hang or crash the entire batch import, losing progress on all remaining rows (only rows already committed before the failure survive).
**Fix:** Wrap the Tier 2 call in a `try/except requests.exceptions.RequestException` that falls through to Tier 3 (treat network failure like an MPC miss), and pass an explicit `timeout=` through `MPCObscodeFetcher.query()`.

### WR-02: IntegrityError handler in Tier 2 assumes any conflict is an obscode race — a name-collision would raise uncaught `DoesNotExist`

**File:** `solsys_code/campaign_utils.py:100-103`
**Issue:** `Observatory.name` is also `unique=True` (`solsys_code_observatory/models.py:30-32`). If `fetcher.to_observatory()`'s `obs.save()` raises `IntegrityError` because of a `name` collision (a different obscode with a duplicate `name_utf8`) rather than an `obscode` race, the `except IntegrityError` handler's `Observatory.objects.get(obscode=code)` will raise `Observatory.DoesNotExist` (since no observatory was actually created for `code`), which is uncaught and propagates out of `resolve_site`, crashing the import for a scenario the code believes it has handled defensively.
**Fix:** Catch `Observatory.DoesNotExist` around the re-fetch and fall through to Tier 3 rather than letting it propagate; or inspect the IntegrityError more precisely before assuming it's an obscode race.

### WR-03: Tier 3 placeholder creation has no race protection, unlike Tier 2

**File:** `solsys_code/campaign_utils.py:105-111`
**Issue:** Tier 2's comment explicitly acknowledges and handles "another row in this same import (or a concurrent process)" racing to create the same `Observatory` (`campaign_utils.py:100-103`), but the Tier 3 `Observatory.objects.create(...)` call three lines later has no equivalent `except IntegrityError` handling. A concurrent process resolving the same unresolvable site code would crash the import at Tier 3 despite the module's stated "never raise" contract.
**Fix:** Apply the same `except IntegrityError: return Observatory.objects.get(obscode=code), True` pattern used in Tier 2.

### WR-04: `resolve_site` doesn't catch malformed-but-"ok" MPC API responses (KeyError)

**File:** `solsys_code/campaign_utils.py:94-99`, `solsys_code_observatory/utils.py:58-97`
**Issue:** `to_observatory()` reads several dict keys (`short_name`, `longitude`, `rhocosphi`, `rhosinphi`, `observations_type`, ...) with `self.obs_data['...']` (no `.get()`/default). If the live MPC API ever returns a `200 OK` response missing one of these keys, `to_observatory()` raises `KeyError`, which `resolve_site` only guards against via `except MissingDataException` — not `KeyError`/`TypeError`/`ValueError`. Confirmed by direct reproduction: mocking `response.json()` to return `{'obscode': 'Z99', 'name_utf8': 'Test'}` (missing `short_name` etc.) raises `KeyError: 'short_name'` out of `resolve_site`, crashing the whole command mid-import. This directly contradicts the docstring's "Never raises for expected messy-data cases" contract.
**Fix:** Broaden the except clause (e.g. `except (MissingDataException, KeyError, ValueError, TypeError)`) around the Tier 2 block, falling through to Tier 3 on any malformed response.

### WR-05: No DB-level uniqueness constraint backs the documented natural key

**File:** `solsys_code/models.py:31-115`, `solsys_code/migrations/0002_campaignrun.py`
**Issue:** `insert_or_create_campaign_run`'s docstring and `import_campaign_csv.py`'s D-04 comment both describe `(campaign, telescope_instrument, ut_start)` as the natural key relied on for idempotent re-imports, and it's used as the `get_or_create(**lookup, ...)` filter. But neither the model nor the migration declares a matching `unique_together`/`UniqueConstraint`. Per Django's own documented caveat, `get_or_create` is only race-safe when backed by a real DB constraint on the lookup fields; without one, two processes (or two overlapping runs of the management command) racing on the same row can both miss the existing row and both attempt to create it, producing duplicate `CampaignRun` rows — defeating the "idempotent re-run" guarantee this feature is designed around (D-04).
**Fix:** Add a migration with `models.UniqueConstraint(fields=['campaign', 'telescope_instrument', 'ut_start'], name='unique_campaign_run_natural_key')` (or `unique_together`) to `CampaignRun.Meta`.

### WR-06: Skipped-row error log includes the full raw CSV row, including contact PII

**File:** `solsys_code/management/commands/import_campaign_csv.py:72`
**Issue:** `self.stderr.write(f'Row {row_num}: {exc} (row: {row!r})')` dumps the entire row dict — including `Contact Person` and `Email` — to stderr whenever a row is skipped for an unrelated reason (e.g. a malformed `Obs. Date`). The project explicitly calls out PII care for this feature (the paired demo notebook states "The fixture used here is entirely synthetic and PII-free (CAMP-05)"), implying the real 3I/ATLAS sheet this command is meant to ingest does contain real names/emails. Logging the full row to stderr for every skip risks that PII ending up in shell history, CI logs, or log aggregation for no functional reason (only the natural-key fields are actually needed to diagnose the skip).
**Fix:** Log only the fields relevant to the failure, e.g.:
```python
self.stderr.write(
    f"Row {row_num}: {exc} (Telescope/Instrument={telescope_instrument!r}, "
    f"Obs. Date={row.get('Obs. Date')!r})"
)
```

### WR-07: Re-importing always overwrites `target` to the auto-resolved value, silently reverting manual corrections

**File:** `solsys_code/management/commands/import_campaign_csv.py:82`
**Issue:** `fields['target']` is unconditionally set to `auto_target` for every row on every run. On a re-import (`action == 'updated'`), this will overwrite a `CampaignRun.target` that a staff user may have manually corrected (e.g. cleared, or pointed at a more specific sub-target) after the first import, since `insert_or_create_campaign_run` treats any field difference — including `target` — as a reason to update and save. There's no way to re-run the bootstrap import without clobbering manual target corrections.
**Fix:** Either exclude `target` from `fields` on update when it was previously set by a human (would require tracking provenance), or document this explicitly as expected/acceptable behavior in the command's docstring so it isn't a silent surprise; at minimum flag it in the command's help text.

### WR-08: `map_observation_status` substring matching has no negation awareness

**File:** `solsys_code/campaign_utils.py:165-184`
**Issue:** The translation table matches by unordered substring containment. A status string like `"Not observed"` (no other keyword present) doesn't match any of `cancel`/`not awarded`/`weather`/`technical`/`publish`/`reduc`/`complet` but does contain `observ`, so it maps to `RunStatus.OBSERVED` — the opposite of its actual meaning. The ordering only helps when a more-specific keyword happens to co-occur (e.g. "Not observed — weather" correctly hits `weather` first); a bare negation with no other signal is silently mis-classified.
**Fix:** Add an explicit negation check before the substring table (e.g. reject/route separately any string matching `r'\bnot\s+observ'` or similar), or require a stricter keyword list that doesn't include a bare positive substring capable of being negated.

### WR-09: No upfront validation of the CSV's column shape

**File:** `solsys_code/management/commands/import_campaign_csv.py:65-99`
**Issue:** Every column is read via `row.get('Exact Header Text', '')`. If the input CSV's header doesn't match exactly (e.g. a renamed column, or the real sheet export changes a header string), every row's corresponding field silently defaults to blank rather than the command failing fast. In the worst case (`'Telescope / Instrument'` header missing/renamed), every row is silently skipped one-by-one with only a per-row stderr line and a `skipped: N` summary — there's no single, clear top-level diagnostic that the header shape itself is wrong.
**Fix:** Validate `reader.fieldnames` against the expected header set up front and raise `CommandError` with a clear message if required columns are missing, before processing any rows.

## Info

### IN-01: Misleading comment referencing a nonexistent local variable

**File:** `solsys_code/campaign_utils.py:90-93`
**Issue:** The comment says "The `errors` return value is intentionally unused..." but `resolve_site` never binds a variable named `errors` — that name only exists inside `MPCObscodeFetcher.query()`'s own local scope. The comment is explaining `query()`'s internals rather than anything visible in `resolve_site`, which is confusing on a re-read.
**Fix:** Reword to something like "`fetcher.query()`'s return value (an error dict on failure) is intentionally unused here — it already logs the API error internally; don't double-log."

### IN-02: `handle()` return type annotated `str | None` but never returns a `str`

**File:** `solsys_code/management/commands/import_campaign_csv.py:36, 119`
**Issue:** The type hint `-> str | None` combined with the trailing comment `# No return statement...`/bare `return` suggests the `str` branch is vestigial. Not incorrect, just slightly misleading for a reader checking the signature for meaning.
**Fix:** Simplify to `-> None` to match actual behavior, or drop the explicit `return` since it returns `None` implicitly anyway.

---

_Reviewed: 2026-07-03T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
