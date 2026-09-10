---
phase: "33"
slug: "series-identity-reconciler-inversion"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
block_on: high
register_authored_at_plan_time: true
created: "2026-09-10"
updated: "2026-09-10"
---

# Phase 33 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Register origin: every one of the 11 plans (33-01 … 33-11) carried a `<threat_model>` block at
> plan time, so this audit **verifies mitigations exist** rather than hunting for new threats.
> Audit depth: ASVS L1 (grep-level presence of each named control in the implementation, plus the
> passing tests each plan wired as proof). No SUMMARY.md in this phase carries a `## Threat Flags`
> section, so no execution-time threats were added to the plan-time register.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| anonymous visitor -> calendar event modal | `tom_calendar`'s `update-event` view renders `event_form.html` for any visitor; whatever the decoration tag returns is public | see plan 33-01 threat register |
| management command / staff action -> CalendarEvent rows | `reconcile_run()` writes shared calendar state on behalf of a run; a write outside its namespace silently corrupts another writer's data | see plan 33-01 threat register |
| anonymous visitor -> calendar month view | `fomo_render_calendar` renders `calendar.html` for any visitor; every event's decoration is public | see plan 33-02 threat register |
| anonymous visitor -> campaign run table | `CampaignRunTableView` serves non-staff a PII-gated `.values()` queryset; anything added to a row must survive that gate | see plan 33-02 threat register |
| staff form submission -> CalendarEventMeta row | Django admin renders every non-readonly model field as an editable widget; a submitted value is bound unless the field is declared read-only | see plan 33-03 threat register |
| migration -> existing production rows | Migration 0017 runs against a database already holding 85 companion rows carrying real attribution and audit history | see plan 33-03 threat register |
| staff POST -> attribution undo view | `_undo_confirmation()` validates its two path values only as integers, so a stale resubmit or a tampered POST can name a pair that was never confirmed to that run | see plan 33-04 threat register |
| unattended reconcile -> attributed companion rows | The detach step runs with no operator watching and writes to rows that carry human confirmation history | see plan 33-04 threat register |
| staff form -> CalendarEventMeta clear branch | The admin change form is a second, independent write path onto the same row | see plan 33-04 threat register |
| developer database -> committed notebook output | Pre-executed notebooks in this directory are committed WITH their output, so whatever a cell prints becomes a permanent public artifact in the repository | see plan 33-05 threat register |
| notebook execution -> developer database | These notebooks write real rows against the developer database when executed | see plan 33-05 threat register |
| anonymous browser -> `campaigns:table` | An unauthenticated reader renders the campaign run table; the view already excludes pending-review runs. | see plan 33-06 threat register |
| anonymous browser -> `/calendar/` month view | An unauthenticated reader renders every attributed event's chip; campaign names and run identity cross this boundary. | see plan 33-06 threat register |
| campaign/run data -> HTML attribute values | `campaign_name` is operator-supplied text rendered into `title=` and `aria-label=`. | see plan 33-06 threat register |
| staff browser -> Django admin change form | A staff user submits field values that reach `CalendarEventMetaAdmin.save_model()`; `confirmed_by` is server-derived from `request.user`, never bound from the form. | see plan 33-07 threat register |
| POST parameter -> `unlink_event_from_run()` | `campaign_views._undo_confirmation()` converts a submitted orphan primary key through `_as_pk_or_none()` before it reaches the helper. | see plan 33-07 threat register |
| unattended management command -> bulk companion-row write | `reconcile_campaign_runs` reaches the same helper with a queryset, with no human in the loop. | see plan 33-07 threat register |
| unattended cron/CLI sweep -> companion-row writes | `reconcile_campaign_runs` clears attributions and their confirmation stamps with no human in the loop. | see plan 33-08 threat register |
| operator-facing summary output -> operator decision | The command's stdout/stderr is the only signal an operator has about what the sweep did. | see plan 33-08 threat register |
| pre-executed notebook -> real developer database | Both notebooks execute against `src/fomo_db.sqlite3`, not a test database, and a reader is invited to run them. | see plan 33-08 threat register |
| other writers' calendar events -> reconciler | Facility-keyed and blank-url events attributed to a run inform the skip rule and must stay read-only to this module. | see plan 33-08 threat register |
| process environment -> Django settings | `FOMO_DATABASE_PATH` crosses from the shell into `DATABASES['default']['NAME']` | see plan 33-09 threat register |
| notebook execution -> developer database file | `jupyter nbconvert --execute` runs arbitrary project code with write access to whatever sqlite file settings resolve | see plan 33-09 threat register |
| notebook committed output -> public git history | every printed value in a `pre_executed/` notebook is committed verbatim and published through the Sphinx docs | see plan 33-09 threat register |
| unattended cron sweep -> attribution audit data | `reconcile_campaign_runs` runs with no human present and can clear `run`/`confirmed_by`/`confirmed_at` on companion rows | see plan 33-10 threat register |
| staff HTTP action -> the same sweep | approve, resolve site, mark cancelled and mark weather failure each call `reconcile_run()` inside a request | see plan 33-10 threat register |
| one run's namespace -> another run's attribution | a `RUN:{pk}` url is a string; a staff member may have attributed that event to a different run | see plan 33-10 threat register |
| server-rendered template -> browser JavaScript execution | an `hx-on::after-request` attribute body is executed as JavaScript in the visitor's browser | see plan 33-11 threat register |
| calendar data (campaign names, event titles) -> rendered month grid | untrusted-ish text authored by public campaign submitters crosses into HTML attributes near the handler | see plan 33-11 threat register |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-33-01 | Denial of Service | `campaign_decoration()` link building, `event_form.html` | high | mitigate | (plan 33-01) Build the campaign-table URL in Python with `reverse()` and return `None` when `run.campaign_id` is None, so a campaign-less run renders no link instead of raising `NoReverseMatch` on a public, unauthenticated page. Task 1 adds a `campaign=None` fixture asserting the modal responds 200 (RESEARCH.md Pitfall 1) | closed |
| T-33-02 | Information Disclosure | `campaign_decoration()` return dict | medium | mitigate | (plan 33-01) The tag returns a fixed seven-key dict and never reads `contact_person`, `contact_email`, `contact_public_opt_in` or `CampaignRun.source`; the `is_publicly_visible` gate keeps a pending-review run's campaign name off the public surface. Task 1 acceptance criteria grep for the contact/source names | closed |
| T-33-03 | Tampering | `_reconcile_classical_nights()` | high | mitigate | (plan 33-01) The skip branch removes the only reconciler write path that could re-key or field-write an event outside `RUN:`; `_may_write()` stays the first condition on every remaining write, and Task 2 adds a byte-identical regression for both blank-url and facility-URL-keyed attributed events | closed |
| T-33-04 | Tampering | `_may_write()` on a foreign attribution | medium | mitigate | (plan 33-01) D-02 keeps the existing behaviour: a `RUN:` event whose companion row points at another run stays blocked and is never reset to the reconciling run; Task 2 adds the explicit regression | closed |
| T-33-SC | Tampering | package installs | low | accept | (plan 33-01) This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | closed |
| T-33-05 | Denial of Service | month-cell marker for a campaign-less run | high | mitigate | (plan 33-02) The marker consumes `campaign_decoration()`, which builds the campaign-table URL with `reverse()` and returns `None` for it when `run.campaign_id` is None — the month view never emits a `{% url %}` call against a null campaign pk. Task 3 adds a month-view regression for the `campaign=None` case | closed |
| T-33-06 | Information Disclosure | month-cell marker tooltip | medium | mitigate | (plan 33-02) `campaign_decoration()` returns `None` for a run that is not publicly visible, so a pending-review run's campaign name never reaches the public month grid; Task 3 asserts the absence for both a staff and an anonymous client, plus the absence of `contact_person` / `contact_email` / `source` | closed |
| T-33-07 | Information Disclosure | `run-{pk}` row ids on the campaign table | low | accept | (plan 33-02) The row id exposes only a `CampaignRun` primary key on rows the non-staff queryset already returns (`pk` is already in `ALLOWED_FIELDS_FOR_NON_STAFF` and already reachable through existing links); no new field crosses the PII gate | closed |
| T-33-08 | Denial of Service | month-view query volume | medium | mitigate | (plan 33-02) The `Prefetch(...select_related('run__campaign'))` chain keeps the marker's run/campaign dereference off the per-event path; Task 3's count-comparison test fails if the count grows with the number of attributed events | closed |
| T-33-SC | Tampering | package installs | low | accept | (plan 33-02) This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | closed |
| T-33-09 | Tampering | `CalendarEventMetaAdmin` / `CalendarEventMetaInline` form binding | medium | mitigate | (plan 33-03) Both new fields are added to `readonly_fields` on both surfaces (D-09), the same mechanism already protecting `confirmed_by`/`confirmed_at`; Task 2 adds a POST-level regression proving a submitted value is not stored | closed |
| T-33-10 | Tampering | migration `0017` against existing rows | high | mitigate | (plan 33-03) The migration carries only `AddField`/`AlterField` operations and no data step (D-08); Task 1 greps for any `RunPython`/`RunSQL`, and Task 3's `MigrationExecutor` test seeds pre-0017 rows and asserts their `run`/`is_verified`/`confirmed_by`/`confirmed_at` values survive byte-identical | closed |
| T-33-11 | Tampering | duplicate projection onto one `ObservationRecord` | medium | mitigate | (plan 33-03) `observation_record` is a `OneToOneField` (D-05), so a second event for the same record raises `IntegrityError` at the database rather than producing a silent duplicate; Task 3 asserts both the collision and the NULL non-collision | closed |
| T-33-12 | Information Disclosure | the two new fields in the Django admin | low | accept | (plan 33-03) Both point at data already visible to staff throughout this codebase; the admin is a staff-only surface, and read-only exposure adds no disclosure beyond what `run` already established | closed |
| T-33-SC | Tampering | package installs | low | accept | (plan 33-03) This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | closed |
| T-33-13 | Tampering | `unlink_event_from_run()` as consumed by `_undo_confirmation()` | high | mitigate | (plan 33-04) The helper's clear is a single conditional update filtered on BOTH the event and the run and returns the changed count, preserving 28-REVIEW WR-01's property that the write itself is the proof the pair was confirmed and that the dismissal row is gated on it. Task 1 tests the wrong-run call returning 0 and changing nothing; Task 2 keeps the `if changed_count:` gate intact | closed |
| T-33-14 | Tampering | `_detach_stale_family_events()` reaching a foreign attribution | high | mitigate | (plan 33-04) The stale queryset stays scoped to `owned_events(run)` and the helper keeps the `run` filter term (T-29-19), so a re-attributed event is never cleared by a reconcile of the run whose namespace its url happens to carry; Task 3 adds the explicit regression | closed |
| T-33-15 | Denial of Service | data loss through an unlink | high | mitigate | (plan 33-04) The helper performs one `.update()` of three fields and contains no delete; Task 1 greps its body for `.delete(` and Task 3 asserts both object counts are unchanged across a real detaching reconcile | closed |
| T-33-16 | Repudiation | a cleared link keeping a stale confirmation stamp | medium | mitigate | (plan 33-04) The helper clears `confirmed_by`/`confirmed_at` with `run` at every call site, so no row can display a confirmation for an attribution that no longer exists (D-16); Task 3 covers the reconciler path, which does not clear them today | closed |
| T-33-21 | Tampering | `unlink_event_from_run()` invoked with a null run | high | mitigate | (plan 33-04) Without a guard, a None run builds `filter(..., run_id=None)`, which matches every already-unlinked companion row and nulls audit stamps the caller never named — silent destruction of confirmation history across unrelated rows. Task 1 makes an early `return 0` on an unresolvable run pk the first statement in the body, verifies it appears above any queryset construction, and seeds a second already-unlinked row with populated audit fields to prove it is untouched (33-REVIEWS.md Agreed Concern 2) | closed |
| T-33-22 | Repudiation | admin `save_model` re-persisting stale audit values over a helper `.update()` | medium | mitigate | (plan 33-04) `super().save_model()` calls `obj.save()`, writing every in-memory field back, so a helper-only clear would be silently reverted and the row would keep displaying a confirmation for a cleared attribution. Task 2 requires the in-memory `obj.confirmed_by`/`obj.confirmed_at` nulling to survive alongside the helper call, with a source verify and a re-fetch-from-database behavior assertion in Task 3 (33-REVIEWS.md Agreed Concern 3) | closed |
| T-33-SC | Tampering | package installs | low | accept | (plan 33-04) This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | closed |
| T-33-17 | Information Disclosure | committed notebook output | high | mitigate | (plan 33-05) No cell prints `contact_person`, `contact_email` or `CampaignRun.source`; the D-04 diff cell snapshots only url, title, description, times and the run id. Tasks 1 and 2 each grep the committed JSON for the contact field names | closed |
| T-33-18 | Tampering | notebook execution against the developer database | medium | mitigate | (plan 33-05) Every new fixture is created with find-or-create semantics inside the notebook's own demo-scoped reset, and no cell deletes or edits a row it did not create — the pattern the lifecycle notebook already establishes. The reconciler notebook's diff cell is read-only apart from the sweep it is measuring | closed |
| T-33-19 | Denial of Service | notebook execution importing the ephemeris path | medium | mitigate | (plan 33-05) No cell imports `solsys_code.views` or `solsys_code.ephem_utils`; importing either downloads roughly 1.6 GB of SPICE kernels at module load. Task 2 greps the committed notebook for the views module import | closed |
| T-33-20 | Repudiation | stale operator documentation | medium | mitigate | (plan 33-05) The runbook's three affected sections are rewritten in this plan rather than left to drift; a stale runbook led an operator astray once before in this project (quick task 260726-kdp), which is why the paired-docs rule now covers `docs/runbooks/` by directory | closed |
| T-33-SC | Tampering | package installs | low | accept | (plan 33-05) This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | closed |
| T-33-06-01 | Information disclosure | `campaign_chip.html` accessible name | medium | mitigate | (plan 33-06) The chip is rendered only when `campaign_decoration()` returns a value, and that tag returns `None` for a run that is not publicly visible. Task 3's rewritten pending-run test is the wired proof that adding a second, screen-reader-reachable label does not widen the disclosure surface. | closed |
| T-33-06-02 | Tampering (stored XSS) | campaign name -> `title=` / `aria-label=` | high | mitigate | (plan 33-06) Both attributes rely on Django's autoescape; the partial adds no `\|safe` and no manual escaping. Task 3's encoding test asserts `&`, `<` and `"` appear only in entity form in both attributes. ASVS L1 output-encoding control. | closed |
| T-33-06-03 | Information disclosure | `event_form.html` single-gate restructure | medium | mitigate | (plan 33-06) Collapsing two gates into one could widen what the modal shows. The surviving `{% elif not event.telescope_label_meta.run and request.user.is_staff %}` keeps the attribution-queue hint staff-only, and the existing pending-run modal tests (`test_pending_run_shows_no_run_block_to_anonymous_visitor` / `..._to_staff_visitor`) must stay green — they are re-run by Task 2's verify. | closed |
| T-33-06-04 | Information disclosure | paginated campaign table | low | accept | (plan 33-06) WR-08's dead link exposes nothing; page 2 is reachable by any reader who could reach page 1. The failure mode is navigational, not a disclosure. | closed |
| T-33-06-SC | Tampering | npm/pip/cargo installs | low | accept | (plan 33-06) This plan installs no package from any package manager — every change is to files already tracked in this repository — so no package-legitimacy audit or blocking human checkpoint applies. | closed |
| T-33-07-01 | Tampering | `unlink_event_from_run()` type dispatch | high | mitigate | (plan 33-07) Task 2 rejects `str`/`bytes` with `TypeError` before any filter is built, so a string primary key can no longer be expanded per-character into an `event__in` filter that clears attributions on unrelated events. ASVS L1 input-validation control; the rejection tests assert no companion row changed. | closed |
| T-33-07-02 | Repudiation | audit-stamp clearing drift between the two writers | medium | mitigate | (plan 33-07) Task 1 makes both writers derive from `UNLINK_CLEARED_FIELDS`, so a future audit field added to the helper cannot be silently missed by the admin path, leaving a row displaying a confirmation for an attribution that no longer exists. | closed |
| T-33-07-03 | Elevation of privilege | staff form binding a carrier or audit field | medium | accept | (plan 33-07) Unchanged by this plan and already mitigated: `confirmed_by`, `confirmed_at`, `observation_record` and `observation_group` are in `readonly_fields` on both admin surfaces, with POST-does-not-bind tests. Task 1's prohibition forbids adding any of them to `UNLINK_CLEARED_FIELDS`. | closed |
| T-33-07-04 | Information disclosure | corrected inline docstring | low | accept | (plan 33-07) Documentation-only; the docstring names no credential, no primary key and no operator identity. | closed |
| T-33-07-SC | Tampering | npm/pip/cargo installs | low | accept | (plan 33-07) This plan installs no package from any package manager — every change is to files already tracked in this repository — so no package-legitimacy audit or blocking human checkpoint applies. | closed |
| T-33-08-01 | Repudiation | `_detach_stale_family_events()` erasing `confirmed_by`/`confirmed_at` unattended | high | mitigate | (plan 33-08) Task 1 returns the cleared-row count and emits `logger.warning` naming the run and the count; Task 2 threads it into `ReconcileResult.detached`, a per-run stderr line and the summary line, so a discarded confirmation is never silent. The row itself is detached, not deleted, so the event returns to Phase 28's queue for a human to re-confirm (WR-03). | closed |
| T-33-08-02 | Tampering | night-derivation change re-mapping existing attributed events | high | mitigate | (plan 33-08) The change is confined to `_observing_night()` and its single call site; the forward url mapping is untouched. Task 3 re-runs the D-04 real-database before/after diff over every non-`RUN:` event and the `<human-check>` requires that diff to stay empty, so a re-mapping that touched a foreign event would be visible before the change is accepted. | closed |
| T-33-08-03 | Denial of service | per-night ORM round trips | low | accept | (plan 33-08) `_attributed_nights()` keeps its single pre-loop query and `select_related('event')`; `_observing_night()` is pure arithmetic on an already-loaded value and adds no query. | closed |
| T-33-08-04 | Information disclosure | committed notebook output | medium | mitigate | (plan 33-08) Prohibitions 33-05 P1/P2 remain in force: no cell may print `contact_person`, `contact_email` or `source`, and Task 3's rewritten cell prints only urls, counters and link state. | closed |
| T-33-08-05 | Tampering | notebook deletion against the real developer database | medium | mitigate | (plan 33-08) Task 3 removes the unconditional `.delete()` outright and replaces it with a demo-scoped removal keyed on a primary key captured in the same cell and guarded by an `assert`, with a printed banner and a markdown warning naming the database (IN-04). | closed |
| T-33-08-06 | Elevation of privilege | reconciler writing outside its namespace | high | mitigate | (plan 33-08) Unchanged and re-asserted: every write is keyed through `run_container_url`/`run_night_url`, the detach step operates on `owned_events(run)` only, and the `_may_write()` exact-match on a foreign `meta.run` still blocks (D-02). Task 1's blocked-night test and the existing `TestAttributedEventsSurviveReconcile` suite are the wired proof. | closed |
| T-33-08-SC | Tampering | npm/pip/cargo installs | low | accept | (plan 33-08) This plan installs no package from any package manager — every change is to files already tracked in this repository — so no package-legitimacy audit or blocking human checkpoint applies. | closed |
| T-33-09-01 | Tampering | `src/fomo/settings.py` `DATABASES` | low | accept | (plan 33-09) An attacker able to set `FOMO_DATABASE_PATH` in the server process environment already controls the process. The variable is read once at settings import, is never taken from a request, and defaults to today's path when unset or empty. | closed |
| T-33-09-02 | Tampering | demo notebook execution vs. `src/fomo_db.sqlite3` | medium | mitigate | (plan 33-09) Task 1 copies the database before Django opens it and points `FOMO_DATABASE_PATH` at the copy; the setup cell asserts the resolved path is the copy, and the task's own gate compares the developer file's md5 across a full execution of both notebooks. | closed |
| T-33-09-03 | Information disclosure | `campaign_lifecycle_demo.ipynb` committed output | medium | mitigate | (plan 33-09) Task 2 removes `contact_person`/`contact_email` from the printed row, and a grep gate keeps both field names out of the notebook file. Values observed to date were empty strings, so no contact data has been published. | closed |
| T-33-09-04 | Tampering | task 3's deletions against the developer database | medium | mitigate | (plan 33-09) A backup copy is taken outside the repository first; every delete is filtered by an explicit demo name, obscode or url prefix; the gate asserts the residue count is 0 AND the total event count is non-zero, so an over-broad delete fails the gate. | closed |
| T-33-09-05 | Repudiation | demo staff user removal | low | mitigate | (plan 33-09) The `campaign_lifecycle_demo_staff` user is removed only when no surviving companion row still names it as `confirmed_by`, so no attribution audit trail loses its actor. | closed |
| T-33-10-01 | Repudiation | `_detach_stale_family_events()` erasing `confirmed_by`/`confirmed_at` | high | mitigate | (plan 33-10) Task 1's `_stale_attributions()` splits on `confirmed_by__isnull` so a human-confirmed row is never cleared by an automated sweep; the confirm/erase regression test asserts the stamp survives repeated sweeps, and the declined count is logged and reported. | closed |
| T-33-10-02 | Tampering | a foreign run's attribution on a `RUN:`-keyed event | high | mitigate | (plan 33-10) `_may_write()` is evaluated before the night's outcome and the contested url is kept in `active_urls`, so the foreign attribution is neither written nor detached; `unlink_event_from_run()`'s `run_id` filter remains as the second layer; a test asserts the other run's three fields are unchanged. | closed |
| T-33-10-03 | Information disclosure | staff-action `messages` text | low | mitigate | (plan 33-10) The new warning and info messages name only counts and the run's own calendar state — no contact field, no `source`, no other run's identity. | closed |
| T-33-10-04 | Tampering | `--dry-run` preview | medium | mitigate | (plan 33-10) The preview calls the read-only `_stale_attributions()` helper; the dry-run branch performs no `.save()`, `.update()`, `.create()` or `.delete()`, and the preview test asserts the stale row is still attributed afterwards. | closed |
| T-33-10-05 | Repudiation | notebook demo writing to shared state | low | mitigate | (plan 33-10) Inherited from plan 33-09: both notebooks execute against a scratch copy, so a doc artifact can no longer seed the queue state that starts this loop (IN-08). | closed |
| T-33-11-01 | Tampering | inline `hx-on::after-request` handler in `calendar.html` | medium | mitigate | (plan 33-11) The handler body is a fixed literal with no template variable interpolated into it, identical on all three click targets; a grep gate pins the exact call count at 3 so a variant handler cannot appear unnoticed. | closed |
| T-33-11-02 | Tampering | campaign name / event title rendered next to the handler | medium | mitigate | (plan 33-11) The chip and title continue to rely wholly on Django autoescape (no `\|safe`, no manual escaping); 33-06's escaping test for `&`, `<` and `"` in the chip's `title`/`aria-label` stays green in this plan's full-suite run. | closed |
| T-33-11-03 | Information disclosure | the pop-up body opened by the fixed handler | low | accept | (plan 33-11) The pop-up loads `calendar:update-event`, whose template already gates staff-only and PII fields (contact fields and `source` are never rendered there, verified in 33-01/33-05). Opening the modal changes no server-side authorisation. | closed |
| T-33-11-04 | Denial of service | a click target whose handler throws | low | mitigate | (plan 33-11) The browser test asserts an empty `pageerror` list, so a handler that throws — the exact defect being closed — fails the suite instead of silently disabling every click on the page. | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on (`high`) count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

### Evidence consulted for the `mitigate` closures (2026-09-10, L1)

| Control | Where it was found |
|---------|--------------------|
| Campaign-table link built with `reverse()` in Python, `None` for a campaign-less run (T-33-01, T-33-05) | `solsys_code/templatetags/calendar_display_extras.py:472-487` |
| No `contact_person` / `contact_email` / `.source` read in the decoration tag, the event modal or the chip partial (T-33-02, T-33-06, T-33-06-01, T-33-11-03) | grep count 0 across `calendar_display_extras.py`, `src/templates/tom_calendar/event_form.html`, `partials/campaign_chip.html` |
| Chip relies on autoescape only — no `\|safe` in any `tom_calendar` template (T-33-06-02, T-33-11-02) | only hit is the comment in `partials/campaign_chip.html:16` saying so; 33-06 encoding test green |
| `_may_write()` gate, `owned_events(run)` scoping, unconditional attributed-night skip, noon-anchored `_observing_night()` (T-33-03, T-33-04, T-33-14, T-33-08-02, T-33-08-06, T-33-10-02) | `solsys_code/campaign_reconciler.py:132, 245, 315, 380`; `TestAttributedEventsSurviveReconcile` green |
| Human-confirmed rows never cleared by the sweep — `_stale_attributions()` splits on `confirmed_by__isnull` and reports `declined`; `logger.warning` on detach (T-33-08-01, T-33-10-01) | `campaign_reconciler.py:482-526, 583-589` |
| `--dry-run` reads only (T-33-10-04) | `management/commands/reconcile_campaign_runs.py:29-102`; `test_dry_run_previews_*` green |
| `unlink_event_from_run()`: early `return 0` on a null run, `TypeError` on `str`/`bytes`, one `.update()` filtered on `run_id` **and** the event, no `.delete()` (T-33-13, T-33-15, T-33-21, T-33-07-01) | `solsys_code/campaign_utils.py:875-946` |
| Single `UNLINK_CLEARED_FIELDS` declaration consumed by both writers; admin `save_model` nulls the in-memory audit fields alongside the helper call (T-33-16, T-33-22, T-33-07-02) | `campaign_utils.py:868`, `admin.py:354-430` |
| `readonly_fields` carry `confirmed_by`, `confirmed_at`, `observation_record`, `observation_group` on both admin surfaces (T-33-09, T-33-07-03) | `solsys_code/admin.py:115, 321`; POST-does-not-bind tests in `test_admin` green |
| Migration `0017` has no `RunPython`/`RunSQL`; `observation_record` is a `OneToOneField` (T-33-10, T-33-11) | `migrations/0017_calendareventmeta_observation_links.py`; `test_calendar_event_meta_links` green |
| Notebooks import neither `solsys_code.views` nor `ephem_utils` (T-33-19) | grep count 0 in both `docs/notebooks/pre_executed/*.ipynb` |
| Notebooks run against a scratch copy via `FOMO_DATABASE_PATH` set before `django.setup()`; residue cleanup gated on residue = 0 and total events > 0 (T-33-18, T-33-08-05, T-33-09-02, T-33-09-04, T-33-09-05, T-33-10-05) | `src/fomo/settings.py:126-134`; both notebooks' setup/teardown cells; 33-VERIFICATION.md 33-09 rows 1-6 |
| No contact field in any committed notebook output (T-33-17, T-33-08-04, T-33-09-03) | 33-VERIFICATION.md: 0 occurrences parsed at HEAD. The `source` clause of that prohibition was **narrowed by human decision** at UAT test 2 (2026-09-10): `source` is the provenance enum, not contact data — see 33-UAT.md `## Decisions` |
| Fixed-literal `hx-on::after-request` handler, exactly 3 occurrences; Playwright asserts empty `pageerror` (T-33-11-01, T-33-11-04) | `src/templates/tom_calendar/partials/calendar.html` (count 3); `test_bootstrap5_rendering` 7 tests OK |
| Staff-action messages name only counts and this run's state (T-33-10-03) | `solsys_code/campaign_views.py:451-717` — 0 hits for contact/source in any `messages.*` call |
| Runbook sections rewritten in-phase rather than left to drift (T-33-20) | `docs/runbooks/telescope_runs_calendar.rst`; 33-VERIFICATION.md 33-10 rows 8-10 and 33-11 row 7 |
| Month-view prefetch keeps the marker off the N+1 path (T-33-08) | `test_calendar_template` query-count test green |

Test evidence at code HEAD (`fb99f9a` — no non-planning file has changed since): verifier ran 270
targeted tests OK plus both ruff gates on 2026-09-10; this audit re-ran the VALIDATION.md quick
command (`test_campaign_reconciler` + `test_calendar_template` + `test_campaign_attribution_views`):
**180 tests, OK**.

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-T-33-SC | T-33-SC | This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | planner (plan 33-01), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-07 | T-33-07 | The row id exposes only a `CampaignRun` primary key on rows the non-staff queryset already returns (`pk` is already in `ALLOWED_FIELDS_FOR_NON_STAFF` and already reachable through existing links); no new field crosses the PII gate | planner (plan 33-02), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-SC | T-33-SC | This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | planner (plan 33-02), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-12 | T-33-12 | Both point at data already visible to staff throughout this codebase; the admin is a staff-only surface, and read-only exposure adds no disclosure beyond what `run` already established | planner (plan 33-03), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-SC | T-33-SC | This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | planner (plan 33-03), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-SC | T-33-SC | This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | planner (plan 33-04), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-SC | T-33-SC | This plan runs no `pip`/`npm`/`cargo` install step; RESEARCH.md `## Package Legitimacy Audit` records that the phase installs no external package | planner (plan 33-05), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-06-04 | T-33-06-04 | WR-08's dead link exposes nothing; page 2 is reachable by any reader who could reach page 1. The failure mode is navigational, not a disclosure. | planner (plan 33-06), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-06-SC | T-33-06-SC | This plan installs no package from any package manager — every change is to files already tracked in this repository — so no package-legitimacy audit or blocking human checkpoint applies. | planner (plan 33-06), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-07-03 | T-33-07-03 | Unchanged by this plan and already mitigated: `confirmed_by`, `confirmed_at`, `observation_record` and `observation_group` are in `readonly_fields` on both admin surfaces, with POST-does-not-bind tests. Task 1's prohibition forbids adding any of them to `UNLINK_CLEARED_FIELDS`. | planner (plan 33-07), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-07-04 | T-33-07-04 | Documentation-only; the docstring names no credential, no primary key and no operator identity. | planner (plan 33-07), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-07-SC | T-33-07-SC | This plan installs no package from any package manager — every change is to files already tracked in this repository — so no package-legitimacy audit or blocking human checkpoint applies. | planner (plan 33-07), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-08-03 | T-33-08-03 | `_attributed_nights()` keeps its single pre-loop query and `select_related('event')`; `_observing_night()` is pure arithmetic on an already-loaded value and adds no query. | planner (plan 33-08), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-08-SC | T-33-08-SC | This plan installs no package from any package manager — every change is to files already tracked in this repository — so no package-legitimacy audit or blocking human checkpoint applies. | planner (plan 33-08), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-09-01 | T-33-09-01 | An attacker able to set `FOMO_DATABASE_PATH` in the server process environment already controls the process. The variable is read once at settings import, is never taken from a request, and defaults to today's path when unset or empty. | planner (plan 33-09), verified by secure-phase L1 audit | 2026-09-10 |
| AR-T-33-11-03 | T-33-11-03 | The pop-up loads `calendar:update-event`, whose template already gates staff-only and PII fields (contact fields and `source` are never rendered there, verified in 33-01/33-05). Opening the modal changes no server-side authorisation. | planner (plan 33-11), verified by secure-phase L1 audit | 2026-09-10 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-10 | 58 (42 mitigate, 16 accept) | 58 | 0 | secure-phase orchestrator (L1 short-circuit: plan-time register, ASVS 1, `threats_open: 0`) — dispatched from `/gsd-verify-work 33` verify:post |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-10
