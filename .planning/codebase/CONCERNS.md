# Codebase Concerns

**Analysis Date:** 2026-09-04

## Phase 33 Critical Issues (Blockers)

**Phase 33 has three load-bearing defects requiring fixes before Phase 34 can proceed safely.**

### CR-01: Row-highlight CSS silently discarded from campaign table template

**File:** `src/templates/campaigns/campaignrun_table.html:1-13`
**Severity:** BLOCKER
**Status:** Unresolved

The `<style>` block containing the D-13 row-highlight CSS sits outside every Django template block in an extends-based child template. Django's `ExtendsNode` discards top-level nodes outside blocks, so the `tr:target { background-color: #fff3cd; ... }` rule is never emitted. The campaign-table anchor link from the calendar decoration lands on the correct row `id="run-{pk}"`, but the visual highlight never renders.

**Impact:** D-13 behavior (highlight the run row the calendar decoration lands on) is completely absent; users cannot visually distinguish which run their calendar entry points to.

**Fix approach:** Move the `<style>` block inside `{% block content %}` or use the parent template's CSS block if available. Add a test assertion for the presence of `tr:target` CSS in the rendered response.

---

### CR-02: Observing night calculation uses wrong boundary for post-midnight events

**File:** `solsys_code/campaign_reconciler.py:328`
**Severity:** BLOCKER
**Status:** Unresolved

The night key is derived as `meta.event.start_time.astimezone(site_zone).date()` — a plain site-local `.date()`. This is only correct for events starting before local midnight. The rest of the codebase (e.g., `telescope_runs._local_noon_utc`) anchors the observing night at local noon, so an event starting after local midnight (e.g., `02:00` Sydney on 2026-08-02) belongs to the **previous** observing night (2026-08-01), not the current calendar day. A facility observation window routinely starts after midnight.

**Impact:** The reconciler mints a duplicate `RUN:` event for the real night *and* spuriously skips the following night. Both failures are blocked by D-01/ANNOT-01. Reproduced with Sydney site + event at `16:00Z` (= `02:00` local, Aug 2 calendar date but Aug 1 observing night).

**Fix approach:** Derive the observing night with the same noon-anchor used by `telescope_runs._local_noon_utc`: `(local - timedelta(hours=12)).date()`. Add a regression test with a post-midnight fixture asserting correct night assignment.

---

### CR-03: Skip rule never fires once `RUN:` event exists — creates permanent duplicate entries

**File:** `solsys_code/campaign_reconciler.py:370-373`
**Severity:** BLOCKER
**Status:** Unresolved

The skip is gated on `existing is None`. The contract states unconditionally: *"a night with an attributed non-`RUN:` event has no reconciler event."* The implementation only honors that when attribution happens **before** the first reconcile.

The normal Phase 34 handoff ordering (reconcile → attribute observation → reconcile again) leaves two calendar entries for the same night forever, both now rendering the campaign chip. `skipped_nights` incorrectly reports 0, giving no signal the issue exists.

**Impact:** Users see duplicated nights in the month calendar, both marked with the campaign chip. The feature advertises a skip rule that only works in one ordering, silent in the other.

**Fix approach:** Make the skip unconditional on the attribution, and have the stale-detach step reclaim the reconciler's superseded event by only building `active_urls` from nights the reconcile actually wrote. Add a test that reconciles, attributes, reconciles again, and asserts exactly one event remains.

---

## Phase 33 Warnings (Secondary Concerns)

### WR-01: `ReconcileResult.skipped_nights` never surfaced to operators

**File:** `solsys_code/management/commands/reconcile_campaign_runs.py:49-95`
**Severity:** WARNING
**Status:** Unresolved

The `skipped_nights` counter is computed by the reconciler but dropped on the floor by the batch command. A run with all nights already attributed reports `created: 0, updated: 0` with no explanation — indistinguishable from "already converged". The demo notebook documents this as a gap rather than closing it.

**Impact:** Operators cannot observe the skip rule working; D-04 documentation promises visibility that doesn't exist.

**Fix approach:** Accumulate and print `skipped_nights` in the batch summary. Log per-run when the skip occurs, similar to how `blocked` is already logged.

---

### WR-02: Shared unlink helper documented but not used by admin clear path

**File:** `solsys_code/admin.py:391-406`
**Severity:** WARNING
**Status:** Unresolved

The admin's in-memory clear path keeps a hand-written copy of the field nulling (`obj.confirmed_by = None; obj.confirmed_at = None`) instead of calling `unlink_event_from_run()`. The comment explains why (pre-persisted stale values), which is correct — but the consequence is that the definition of "what clearing means" now lives in two places. If a fourth audit field is ever added to the helper, the admin path will silently stop clearing it.

**Impact:** Maintenance hazard: silent drift between two sources of truth if the link/audit schema grows.

**Fix approach:** Export the field set from `campaign_utils` as a constant and consume it on both sides, so they cannot drift independently.

---

### WR-03: Reconciler's detach silently destroys audit stamps with no trace

**File:** `solsys_code/campaign_reconciler.py:411-453`
**Severity:** WARNING
**Status:** Unresolved

The detach step clears `confirmed_by`/`confirmed_at` during an unattended batch sweep with no log line, no count, and no compensating trace record (unlike the attribution-undo view, which creates a `CalendarEventDismissal` for auditing). The helper's return value (changed count) is discarded, and `_detach_stale_family_events()` never reports back to `ReconcileResult`, so command summaries and staff-action sites cannot tell a human that a confirmation was just erased.

**Impact:** Loss of auditability during batch operations; humans cannot detect when confirmation evidence was cleared.

**Fix approach:** Log the detach event with the count of cleared rows. At minimum, thread the count into `ReconcileResult` as a `detached` field.

---

### WR-04: Unlink helper type dispatch silently mis-filters on unexpected scalar types

**File:** `solsys_code/campaign_utils.py:901-910`
**Severity:** WARNING
**Status:** Unresolved

The helper's `else` branch treats any non-`CalendarEvent`/`int` input as an iterable: `event_filter = {'event__in': events}`. A `str` pk — the natural shape from a POST parameter — is iterable, so `unlink_event_from_run('12', run)` produces `event__in=['1', '2']` and clears events 1 and 2 instead of event 12. No exception is raised.

Today's call sites happen to be safe (`campaign_views._as_pk_or_none()` returns `int`), but the helper is documented as the shared entry point and its signature advertises `CalendarEvent | int | Any`.

**Impact:** Silent mis-filtering hazard if a future call site passes a string pk.

**Fix approach:** Narrow the dispatch and fail loudly on strings: add a type check that raises `TypeError` if `isinstance(events, str)` with a message explaining why strings are iterable and dangerous.

---

### WR-05: New decoration tests are satisfied by wrong fixtures

**File:** `solsys_code/tests/test_calendar_template.py:768-905`
**Severity:** WARNING
**Status:** Unresolved

Three month-view decoration tests cannot fail for the reason they claim:

1. `test_pending_review_run_shows_no_marker_for_staff_and_anonymous` — asserts the pending run's `telescope_instrument` is absent, but the month cell never renders that field at all; the pending run shares a campaign with the approved run, so the assertion passes for the wrong reason.
2. `test_chip_does_not_consume_title_truncation_budget` — uses titles shorter than the truncation threshold, so they're never truncated regardless of the chip's placement in the filter expression.
3. `test_no_campaign_run_renders_marker_and_no_table_href` — other fixtures in the same month grid also emit chips, making the no-campaign case not isolated.

**Impact:** False confidence in test coverage; the actual month-view isolation gaps are uncovered.

**Fix approach:** Assert on values only the fixture under test produces. Give the pending run its own campaign so the campaign name is the discriminator; use a title exactly at the truncation boundary.

---

### WR-06: Runbook and admin docstring document an inline field that does not exist

**File:** `docs/runbooks/telescope_runs_calendar.rst:812-818`; `solsys_code/admin.py:75-78`
**Severity:** WARNING
**Status:** Unresolved

The runbook instructs users to clear the **Attributed campaign run** value on the inline (`CalendarEventMetaInline`). Django's inline formsets exclude the parent foreign key from the child form, so the `run` field is **not rendered as editable at all** — there is no value to clear there. The audit-stamp clearing the prose describes is only implemented on the standalone `CalendarEventMetaAdmin` change page, not on the inline.

**Impact:** Users following the runbook on the inline will find no such field; the documented workflow is impossible.

**Fix approach:** Point the bullet at the standalone surface where the operation actually exists. Correct the inline docstring to note that deletion of the row (not field clearing) is how to un-attribute.

---

### WR-07: Modal decoration gates the tag twice, in two places

**File:** `src/templates/tom_calendar/partials/event_form.html:118-136`
**Severity:** WARNING
**Status:** Unresolved

The template keeps a pre-existing gate on `run.is_publicly_visible` and adds a second, independent gate on the tag's return value. `campaign_decoration()` applies the exact same rule internally, so the visibility logic lives in two places. A future change to the tag's rule would not take effect in the modal — the template gate silently wins.

**Impact:** D-10 drift risk: the template gate can become stale independently of the tag's actual rule.

**Fix approach:** Remove the redundant template gate and restructure to use the tag's return value exclusively. Lift the `run` dereference out to a single, outer gate if needed.

---

### WR-08: Campaign-table anchor link fails silently for runs past page 1

**File:** `solsys_code/templatetags/calendar_display_extras.py:466-473`
**Severity:** WARNING
**Status:** Unresolved

The `#run-{pk}` anchor is built with no page parameter. `CampaignRunTableView` paginates at 25 rows. A campaign with >25 runs — the 3I/ATLAS case this feature exists for — lands the browser on page 1 with no matching `id="run-{pk}"` anchor in the document, so the browser scrolls nowhere and (once CR-01 is fixed) nothing highlights. The failure is completely silent.

**Impact:** The decoration's link to the run table is broken for large campaigns; users cannot reach the run row they clicked on.

**Fix approach:** Compute the run's page number when building the link (or at least clear filters), so the link lands on the page that contains the row. At minimum, document the limitation and add a test pinning the >25-run behavior as a known constraint.

---

## Phase 33 Info Items

### IN-01: Unused `run_pk` key in decoration return value

**File:** `solsys_code/templatetags/calendar_display_extras.py:477`
**Impact:** Dead payload in a documented "exactly these keys" contract.

---

### IN-02: Window arithmetic computed twice per classical reconcile

**File:** `solsys_code/campaign_reconciler.py:362,487`
**Impact:** Maintenance hazard; two copies must agree exactly for detach to be a no-op.

---

### IN-03: Campaign marker glyph has no accessible name for screen readers

**File:** `src/templates/tom_calendar/partials/calendar.html:252-283`
**Impact:** Visual-only attribution for users relying on assistive technology.

---

### IN-04: Reconcile demo notebook deletes rows from the live dev database

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (cell 18)
**Impact:** Live database mutation in a doc artifact; users executing the notebook will delete rows.

---

### IN-05: Null-campaign runs still render a campaign marker with "(no campaign)" tooltip

**File:** `solsys_code/templatetags/calendar_display_extras.py:476`
**Impact:** Visual inconsistency; a campaign marker whose tooltip says there is no campaign.

---

## Module-Load Side Effects

### SPICE Kernel Download on Import

**File:** `solsys_code/ephem_utils.py:62-69`
**Severity:** HIGH
**Status:** Design limitation

Importing `solsys_code.ephem_utils` (and transitively, `solsys_code.views`) runs `fomo_furnish_spiceypy()` at module load time, which downloads **~1.6 GB of SPICE kernels** to `~/.cache/sorcha/` on first use and builds the ASSIST ephemeris object. This blocks test collection: `python manage.py test solsys_code` cannot even collect tests without paying this cost.

**Impact:**
- Test collection is unexpectedly slow and requires network access
- Any import of `solsys_code.views` pulls in the full SPICE ephemeris setup
- Cannot mock ephemeris in tests without careful import ordering

**Workaround:** The SPICE download only happens once; subsequent test runs reuse the cache.

**Note:** This is a design decision in how Sorcha's ephemeris is used, not a bug. It trades startup overhead for correctness (full GR integration for solar system targets). Addressed in CLAUDE.md as a known pattern.

---

## Test Suite Issues

### TestEphemeris Segfault in Native ASSIST

**File:** `solsys_code/tests/test_views.py:72-170` (class `TestEphemeris`)
**Severity:** HIGH
**Status:** Known exclusion

The native ASSIST library used by Sorcha occasionally segfaults during the `TestEphemeris` test class. The exact root cause is unclear (likely an interaction between ASSIST's C++ backend and Python's memory management), but the failure is intermittent and non-deterministic.

**Impact:** Test runs may fail sporadically with a segmentation fault, blocking CI/CD and development.

**Mitigation:** `CLAUDE.md` explicitly marks this test as excluded from the regular test suite. The REQUIREMENTS.md testing section documents this exclusion.

---

### Legacy Pytest Suite Runs Against Non-Django Tests

**File:** `.pre-commit-config.yaml:94-102`; `pyproject.toml` (testpaths config)
**Severity:** MEDIUM
**Status:** Maintenance debt

The pre-commit hook and GitHub workflows run pytest against paths `["tests", "src", "docs"]`, which collects only the legacy `tests/fomo/test_packaging.py` suite. The real test suite — 379+ tests in `solsys_code/tests/` — lives in Django's test runner and is **not** invoked by pytest.

The comment in `.pre-commit-config.yaml` states "the pytest suite will likely be removed — do not add tests to it."

**Impact:**
- Pre-commit runs a suite that does not test the core application
- Coverage reports from GitHub workflows do not reflect actual test coverage of the main codebase
- Maintenance burden: two separate test runners with different semantics

**Fix approach:** Consolidate to the Django test runner. Remove the legacy pytest suite or update pyproject.toml to exclude `tests/` and only test documentation/packaging.

---

## Security and Configuration Concerns

### Hardcoded Secret Key and DEBUG=True in settings.py

**File:** `src/fomo/settings.py:25-28`
**Severity:** MEDIUM (dev only, not production)
**Status:** Design pattern (acceptable for dev, dangerous if deployed)

The Django secret key is hardcoded in version control, and `DEBUG=True` is set unconditionally. This is appropriate for development but a critical security issue if the application were ever deployed to production without overrides.

**Impact:**
- FOMO is not suitable for production deployment as-is
- Requires manual configuration via environment variables or `local_settings.py` override

**Current mitigation:** `src/fomo/settings.py` imports `local_settings` at the end (if present), allowing dev and production to diverge. CLAUDE.md documents the pattern. The codebase is explicitly development-focused.

---

### SQLite Database in Version Control

**File:** `src/fomo_db.sqlite3`
**Severity:** LOW (dev only)
**Status:** Expected pattern for single-dev projects

The SQLite database file is committed to git. This is normal for small development projects but:

- Merge conflicts are difficult when multiple branches have local DB changes
- The dev database may contain stale fixture data
- Production would need a migration path

**Current state:** This is documented as the expected development setup. Production deployments would use PostgreSQL per the comment in `src/fomo/settings.py`.

---

## Anti-Patterns and Debt Markers

### Observatory Creation Shortcut Not Implemented

**File:** `solsys_code/views.py:298`
**Severity:** LOW
**Status:** Unimplemented feature

```python
# XXX Could replace this by a creation of the missing Observatory
# relatively easily
observatory = get_object_or_404(Observatory, obscode=obscode)
```

The ephemeris view requires an Observatory record to exist for the requested obscode. A user requesting an unknown obscode gets a 404, when the code could auto-create the Observatory via MPC lookups.

**Impact:** Minor UX friction; users must manually create observatories or request staff to add them.

**Fix approach:** Low priority; would require fetching MPC observatory data inline and creating the record.

---

### Repeated Coordinate Transforms

**File:** `solsys_code/ephem_utils.py:217`
**Severity:** LOW
**Status:** Efficiency debt

```python
# XXX This is done repeatedly in both directions... this must lose speed and precision Shirley...
```

Coordinate transforms are computed repeatedly in forward and reverse. No immediate impact, but a refactoring opportunity for performance.

---

## Infrastructure and API Integration Gaps

### LCO/Gemini Sync Commands: Limited Error Logging

**Files:** `solsys_code/management/commands/sync_lco_observation_calendar.py`; `sync_gemini_observation_calendar.py`
**Severity:** MEDIUM
**Status:** Partial mitigation (timeout added, logging incomplete)

Recent versions added explicit HTTP timeouts (`_API_TIMEOUT_SECONDS = 10` in `calendar_utils.py`), but error handling is still minimal:

- Network timeouts are caught and logged, but no retry/backoff logic
- Some API failures result in silently skipped records with a generic stderr message
- The reconciler's detach step (WR-03) clears audit stamps with no log

**Impact:** Operator visibility into sync failures is limited; debugging network issues requires code inspection.

**Partial fix:** Timeouts are now explicit (Phase 28+ work). Further improvements would require structured error logging and per-error categorization.

---

### Timezone Handling in Calendar Sync

**Files:** `solsys_code/campaign_reconciler.py`, `solsys_code/telescope_runs.py`
**Severity:** HIGH (addressed by CR-02)
**Status:** Broken (see CR-02)

See CR-02 above: observing night boundary calculations are incorrect for post-midnight events. This affects both the reconciler and the telescopes-runs helper.

---

## Database and ORM Concerns

### Concurrent Write Limitations with SQLite

**File:** `src/fomo/settings.py:126-129`
**Severity:** MEDIUM (dev only)
**Status:** Expected limitation

SQLite has strict concurrent write locking. Multiple background sync commands (LCO, Gemini, reconcile) running simultaneously can timeout on database locks.

**Current mitigation:** FOMO is single-developer, so concurrent writes are rare. Production would use PostgreSQL.

**Note:** GitHub workflows run tests serially; CI does not encounter write contention.

---

## Testing Coverage Gaps

### CalendarEventMeta Observation-Link Reverse Manager Ordering

**File:** `solsys_code/models.py:12-88` (CalendarEventMeta, no `Meta.ordering`)
**Severity:** LOW
**Status:** Abstained verification (Phase 33)

Phase 33 verification abstained on whether the `observation_group` reverse manager's iteration order is stable. The model declares no `Meta.ordering`, and no production code iterates `group.calendar_event_metas` expecting stable order.

**Impact:** Theoretical: if Phase 34's projector adds code that depends on insertion order (e.g., `.first()` to get the "first" linked event), it could fail non-deterministically.

**Fix approach:** Add a held-out/property-based test that shuffles insertion order and asserts consuming code is unchanged. Or explicitly accept the risk if the order-independent semantics are clear.

---

## Open Items from Phase 33

Per the Phase 33 VERIFICATION.md report (2026-09-04), three items require human verification:

1. **Visual legibility of campaign chip** (month-cell glyph against dynamic fill colors) — no server-side test can observe this.
2. **Browser anchor-scroll and `:target` highlight rendering** — both ends are server-asserted; the browser rendering is real-behavior only.
3. **Prohibition review** — 11 judgment-tier prohibitions with no wired enforcement (see Phase 33 VERIFICATION.md section "Prohibitions").

Additionally, two flagged items in the prohibitions review:
- Reconciler notebook cell 36 prints empty `contact_person=''`/`contact_email=''` values (PII gate confirmed, no values leaked).
- Notebooks write to and delete rows in the real developer database `src/fomo_db.sqlite3` (cell 18, scoped to the notebook's own created row).

---

---

*Concerns audit: 2026-09-04*
