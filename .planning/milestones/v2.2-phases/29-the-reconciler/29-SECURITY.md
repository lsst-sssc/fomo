---
phase: 29
slug: the-reconciler
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-08-05
---

# Phase 29 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Register authored at plan time (`register_authored_at_plan_time: true`) across
> `29-01-PLAN.md` .. `29-06-PLAN.md`; this document verifies each declared disposition
> against the shipped code, not against the plans' intent.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| reconciler -> `CalendarEvent` table | The reconciler writes rows on a general-purpose calendar it shares with hand-created entries, conferences, proposal deadlines, `load_telescope_runs` ingest and two sync commands. | Trusted server-side `CampaignRun` fields only |
| `CampaignRun` row state -> projection decision | `telescope_class` / `site` / window drive which key family is written (corrected 2026-08-05, quick task `260805-tad`: `source` never did); no HTTP input crosses here. | Staff-editable model fields (admin-gated; `source` withheld on `web` rows) |
| `CalendarEventMeta.run` (staff-confirmed attribution) -> write authority | A Phase 28 confirmation is what grants the reconciler permission to adopt and re-key an ingest-created event. | Staff decision -> FK |
| operator shell -> `reconcile_campaign_runs` | Only input is the boolean `--dry-run`; no user-supplied identifier reaches the ORM. | Shell access (already DB-equivalent) |
| staff POST -> `CampaignRunDecisionView` | Untrusted request data; unchanged by this phase (`StaffRequiredMixin`, POST-only, fixed action whitelist). | Staff session -> fixed action string |
| Django admin delete -> `CampaignRun` | **New in this phase (post-plan review fix):** a run delete now hard-deletes calendar rows. | Staff session -> destructive DB write |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Evidence verified in code | Status |
|-----------|----------|-----------|----------|-------------|---------------------------|--------|
| T-29-01 | Tampering | `_reconcile_container` / `_reconcile_classical_nights` write paths | high | mitigate | `_may_write()` defined `campaign_reconciler.py:164-180` and is the **first** condition in both declared write paths: `campaign_reconciler.py:213` (container, before any create/update) and `campaign_reconciler.py:333` (per-night, before the field set is even built). Both declared fixtures exist and pass: un-owned same-window event `tests/test_campaign_reconciler.py:224-247`, wrong-run companion `:249-277`, plus the trailing-colon namespace guard `:279-293`. 34/34 reconciler + command tests green (`python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs`, 2026-08-05). | closed |
| T-29-02 | Tampering | `_link_event_to_run` | medium | mitigate | `campaign_reconciler.py:183-193`: `get_or_create(event=event)` then, only when `meta.run_id != run.pk`, `meta.save(update_fields=['run'])`. `update_fields` structurally excludes `is_verified` / `confirmed_by` / `confirmed_at`; no assignment to those fields exists anywhere in the module (grep: zero hits). Fixture proof for `is_verified` survival on adopt: `tests/test_campaign_reconciler.py:357,372`. Note: no test asserts `confirmed_by` / `confirmed_at` survival — closed on the code-level `update_fields` guarantee, not on test coverage. | closed |
| T-29-03 | Elevation of Privilege | `reconcile_run()` invoked directly for an invalid-state run | high | mitigate | `_skip_reason()` approval gate is the first check in the function, `campaign_reconciler.py:151-152` (`approval_status != APPROVED -> 'not approved'`), and `reconcile_run()` calls it before any branch dispatch (`:416-418`). Covers PENDING_REVIEW and REJECTED alike (negative test, not allow-list). Test: `tests/test_campaign_reconciler.py:69-76`. Idempotency: `:298-317` (second pass `unchanged`, no `modified` churn) and `tests/test_reconcile_campaign_runs.py:102-127`. | closed |
| T-29-04 | Information Disclosure | Calendar titles/descriptions built from run fields | low | accept | `event_title()` `campaign_reconciler.py:121-130` reads only `campaign.name`, `telescope_instrument`, `window_start/end`, status prefix. `event_description()` `:133-138` reads only `observation_details` + the run-status line — byte-equivalent to the retired helper's fields (`git show 6e9da58^:solsys_code/campaign_views.py:499,894`). `grep -n contact solsys_code/campaign_reconciler.py` -> zero hits; `contact_person` / `contact_email` / `contact_public_opt_in` never reach a title or description. See AR-29-01. | closed |
| T-29-05 | Tampering | `_adopted_event_for_night` -> re-key write | high | mitigate | Candidate set is `CalendarEventMeta.objects.filter(run_id=run.pk, event__url='')` (`campaign_reconciler.py:271-275`) — narrower than declared (the `event__url=''` restriction was added by 29-REVIEW CR-01), so an event with no companion row, an unset `run`, or a `run` pointing elsewhere is structurally outside the query. `_may_write()` still applies afterwards as defence in depth (`:333`). Tests: `tests/test_campaign_reconciler.py:351-424` (adopt/re-key/site-local night), `:657-686` (a stale own-container event is NOT adopted into a night). | closed |
| T-29-06 | Tampering | Re-key moving `start_time` off `load_telescope_runs`' ±5-min lookup key | medium | mitigate | On the existing/adopted path the field set is `fields = common_fields` (`campaign_reconciler.py:350-351`), where `common_fields` = `title`/`description`/`target_list` only (`:338-342`); `start_time`/`end_time`/`telescope` appear only in the create branch (`:344-349`). Asserted by `TestAdoptAndRekey`: `tests/test_campaign_reconciler.py:369-371` (file-derived window byte-identical after adopt) and `:379-397` (re-key sticky, second pass `unchanged`, `modified` frozen). | closed |
| T-29-07 | Tampering | Reconciler overwriting an `ObservationRecord`-derived event | high | mitigate | **Corrected on re-verification 2026-08-05** (quick task `260805-tad` removed the `source`-driven dispatch branch that this row's evidence used to cite). The real mechanism: `_may_write()` (`campaign_reconciler.py:215`) is the **first** condition checked in both write paths -- `campaign_reconciler.py:266` (container, before any create/update) and `:387` (per-night, before the field set is even built) -- so the protection does not depend on which branch a given run takes. `TestRecordEventNonInterference.test_reconciler_never_touches_the_record_derived_event` (`tests/test_campaign_reconciler.py`, per-night branch: a queue-sourced, site-resolved run) and `TestContainerRecordEventNonInterference.test_reconciler_never_touches_the_record_derived_event` (`tests/test_campaign_reconciler.py`, container branch: a class-wide run) each assert the LCO-portal-keyed record event's `url`/`title`/`start`/`end` and `modified` are unchanged after **two** reconcile passes, and that the run's own event(s) coexist beside it without disturbing it. | closed |
| T-29-08 | Repudiation | Mid-batch failure with no record of what did/didn't reconcile | medium | mitigate | Per-run failure: `management/commands/reconcile_campaign_runs.py:58-62` writes `Run pk=…: reconcile failed (…) -- skipping` to `stderr` and increments `failed`. Per-run skip: `:64-67` writes `Run pk=…: skipped (reason)`. Blocked events: `:73-74`. Summary carries `skipped:` / `failed:` in both dry-run and real branches (`:82-83`, `:92-93`). `TestFailureIsolation` (`tests/test_reconcile_campaign_runs.py:169-212`) asserts both halves: `summary['failed'] == 1` **and** `Run pk={run_b.pk}` present in captured stderr, with runs A and C still reconciled. | closed |
| T-29-09 | Tampering | A `--dry-run` invocation that nonetheless writes | high | mitigate | Container dry-run returns from `preview_calendar_event_action()` before any ORM write (`campaign_reconciler.py:217-219`); per-night dry-run `continue`s before the write block (`:353-359`); the CR-01 detach write is explicitly skipped under dry-run (`:444-445`). `TestDryRun` asserts `CalendarEvent.objects.count() == 0` **and** `CalendarEventMeta.objects.count() == 0` after a dry sweep (`tests/test_reconcile_campaign_runs.py:137-138`) and re-asserts no writes against already-reconciled state (`:160-163`). Real-data confirmation: dev-DB `CalendarEvent` count 20 before and after the pre-fix dry run (`29-06-SUMMARY.md:253-255`). | closed |
| T-29-10 | Elevation of Privilege | Command projecting an unapproved web submission | high | mitigate | The command queryset is deliberately unfiltered — `CampaignRun.objects.all().select_related('site','campaign').order_by('pk')` (`management/commands/reconcile_campaign_runs.py:47`, with the "must never grow a second, divergent copy of that rule" comment at `:43-46`); the only gate is `reconcile_run()` -> `_skip_reason()` (`campaign_reconciler.py:151`). Real-data proof of the single decision point firing: `Run pk=31: skipped (not approved)`, `Run pk=43: skipped (not approved)` in both live sweeps (`29-06-SUMMARY.md:225,231,404,410`). | closed |
| T-29-11 | Denial of Service | Unbounded sweep over `CampaignRun` | low | accept | `select_related('site', 'campaign')` present at `management/commands/reconcile_campaign_runs.py:47`; command is operator-invoked, never request-triggered (no URL route imports it). Measured live: 44 runs, 63 creates, exit 0 (`29-06-SUMMARY.md:393-403`). See AR-29-02. | closed |
| T-29-12 | Elevation of Privilege | `reconcile_run()` fired for an invalid-state run by a rewired call site | high | mitigate | All three declared guards verified untouched by the rewire commit — `git show 6e9da58 -- solsys_code/campaign_views.py` contains **no** hunk touching `StaffRequiredMixin`, `http_method_names`, the action whitelist, or any `approval_status=` conditional filter. Present today: approve's conditional `.update()` on `PENDING_REVIEW` (`campaign_views.py:458-460`), `_resolve_site()`'s `APPROVED and site_needs_review` precondition (`:591-593`) plus the `site_id=previous_site_id` conditional claim (`:643-651`), `_set_run_status()`'s `APPROVED` precondition (`:733-735`) and `updated_count == 0` staleness short-circuit (`:738-747`). `reconcile_run()` re-checks approval itself (`campaign_reconciler.py:151`) and is idempotent. | closed |
| T-29-13 | Spoofing / Tampering | The four POST actions | medium | accept (unchanged) | `class CampaignRunDecisionView(StaffRequiredMixin, View)` `campaign_views.py:434`, `http_method_names = ['post']` `:446`, fixed whitelist `if action not in ('approve','reject','resolve_site','mark_cancelled','mark_weather_failure')` -> `HttpResponseBadRequest()` `:451-452`, `_ACTION_TO_RUN_STATUS` fixed dict `:428-431`. Confirmed unchanged by this phase (diff of the only phase-29 commit touching the file, `6e9da58`, shows zero changes to these lines). See AR-29-03. | closed |
| T-29-14 | Repudiation | Reconcile failure silently leaving a run approved with no calendar entry | medium | mitigate | D-04's asymmetry present verbatim: `approve()` catches **only** `ValueError` and logs it, leaving the approval standing (`campaign_views.py:526-532`), with the broader revert-to-PENDING_REVIEW `except Exception` still beneath it (`:534-548`); `_resolve_site()` wraps the reconcile in a non-reverting `except Exception` using `logger.exception`, warns the user, returns **before** `site_needs_review` is cleared (`:679-689`), and clears the flag only after a clean return (`:691-692`); `_set_run_status()` warns "retry the same action" and never reverts the status (`:757-767`). | closed |
| T-29-15 | Tampering | Deleting the backfill command while something still imports it | medium | mitigate | `solsys_code/management/commands/backfill_range_calendar_events.py` and `solsys_code/tests/test_backfill_range_calendar_events.py` are both gone (commit `6e9da58`, `-103` lines of command). No definition and no import of `_project_calendar_event` / `_calendar_event_title` / `_RUN_STATUS_CALENDAR_PREFIX` survives in `solsys_code/` or `src/` (grep for `def _project_calendar_event` / `def _calendar_event_title` -> zero hits). Residual matches are prose only (docstrings at `campaign_reconciler.py:81,103,144,286`, `reconcile_campaign_runs.py:14,17`, `calendar_utils.py:553`) plus generated build artefacts in `src/fomo.egg-info/`. See OBS-29-02 for one stale prose comment that is now factually wrong. | closed |
| T-29-16 | Repudiation | Runbook describing a deleted command / superseded workaround | medium | mitigate | `grep -n backfill_range_calendar_events docs/runbooks/telescope_runs_calendar.rst` -> zero hits. "Why doesn't the calendar pop-up show a 'Campaign run' block?" rewritten at `docs/runbooks/telescope_runs_calendar.rst:564-603`: states the link is now set automatically by `reconcile_campaign_runs` and by the four staff actions, and demotes the admin inline to the un-adopted case. `reconcile_campaign_runs` documented as its own section (`:492-562`) and in the cheat-sheet (`:636`). Reclassification/detach behaviour documented `:312-330`. See OBS-29-01 for an undocumented behaviour change. | closed |
| T-29-17 | Information Disclosure | Contact PII / real proposal strings in docs or committed notebook output | medium | mitigate | No email address, `contact_person` or `contact_email` string occurs in `docs/runbooks/telescope_runs_calendar.rst` or `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` (case-insensitive grep -> zero hits). The notebook seeds its own fixtures (`Observatory` `X29`/`X30`, "Reconciler Demo Campaign", 3 demo runs) and its committed output shows `runs: 3` only — no real campaign, telescope or proposal string appears in any output cell. | closed |
| T-29-18 | Tampering | Notebook execution writing fixture rows into a real database | low | accept | Executed against an isolated worktree DB (output cell 2 records `repo_root='…/.claude/worktrees/agent-ab02ad4…'`), and `src/fomo_db.sqlite3` is gitignored (`.gitignore:64-65`). Rows are clearly-labelled demo fixtures created with `update_or_create` (idempotent re-execution). See AR-29-04. | closed |
| T-29-19 | Tampering | The first live sweep overwriting or deleting a real hand-created / un-attributed calendar event | high | mitigate | **Closed on re-verification 2026-08-05** (was open; fixed by quick task `260805-qdc`, commits `7070fae` + `009195f`). Branch half unchanged and still verified: `_may_write()` first in both branches (`campaign_reconciler.py:213,333`), real-data `modified` evidence (`29-06-SUMMARY.md:486-509`). The two post-plan write paths now apply the same rule: `writable_events()` (`campaign_reconciler.py:121-141`) narrows `owned_events()` by `Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True) | Q(telescope_label_meta__run=run)`; the `pre_delete` cascade calls it (`models.py:385,387`) and `_detach_stale_family_events()` filters its bulk update with `run=run` (`campaign_reconciler.py:427`). Fixtures: `tests/test_campaign_reconciler.py:705-789` `TestCrossRunOwnershipGuards` — `:714` (delete spares run B's event), `:735` (reconcile does not clear run B's attribution), `:758` (no over-narrowing: container + `run`-unset + no-companion rows still deleted). 37/37 green (`python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs`, 2026-08-05). Independently re-probed: see "Resolved Threats" below. | closed |
| T-29-20 | Tampering | A wrong `source` edit sending a run down the wrong branch | medium | mitigate | Convergence implemented: `reconcile_run()` re-derives the family from current state every call and `_detach_stale_family_events()` detaches (never deletes) the old family (`campaign_reconciler.py:371-398`, called `:444-445`), proven by `tests/test_campaign_reconciler.py:627-655` (old per-night events detached, still present) and `:657-686` (stale container not adopted). Idempotency against real data: second live sweep `created: 0, updated: 0, unchanged: 64` (`29-06-SUMMARY.md:415-436`). Task-2 checkpoint reasoning and the "no classification disagreed with" record: `29-06-SUMMARY.md:296-340`. Per-run outcome table: `:438-485`. | closed |
| T-29-21 | Repudiation | A live-data change with no durable record (dev DB gitignored) | medium | mitigate | `29-06-SUMMARY.md` carries the exact edited pks with old/new `source` values in a 10-row table (`:300-315`), the verification query and its result (`:325-340`), the verbatim pre-fix dry-run (`:185-256`), the verbatim post-fix dry-run (`:343-392`), both real sweeps (`:393-436`) and event counts before/after (`:486-509`). | closed |
| T-29-22 | Elevation of Privilege | Sweep projecting an unapproved web submission onto the public calendar | high | mitigate | `_skip_reason()` gate `campaign_reconciler.py:151-152`; every skip recorded by pk and reason both by the command at runtime (`reconcile_campaign_runs.py:64-67`) and in the summary's itemised 8-row skip table with expected-vs-genuine classification (`29-06-SUMMARY.md:510-528`). Two `not approved` skips (pk=31, pk=43) appear in both live sweeps; neither appears in the created counts. | closed |
| T-29-SC | Tampering | npm / pip / cargo installs | low | accept | `git diff 2f1e8cd~1..HEAD -- pyproject.toml` (whole phase-29 commit range) is empty; no `requirements*.txt` / lockfile exists or was added. No new dependency was installed. See AR-29-05. | closed |

*Status: open · closed — only open threats at or above `workflow.security_block_on` count toward `threats_open`*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Open Threats

None.

---

## Resolved Threats

### T-29-19 — RECON-05's ownership rule was not applied to two shipped write paths — **CLOSED 2026-08-05**

**Resolution:** implemented (option 1 of the two the original finding offered). Quick task
`260805-qdc`, commits `7070fae` (RED regression tests) and `009195f` (fix). Re-verified by a
second audit pass on 2026-08-05; the original finding is retained verbatim below for the
record, followed by the re-verification evidence.

#### Original finding (2026-08-05, first audit pass)

**Declared mitigation (29-06-PLAN.md):** "RECON-05's ownership rule is the first filter in
every write path (plan 29-01)".

**What is present:** `_may_write()` guards the two *branch* write paths
(`campaign_reconciler.py:213`, `:333`) and the fixture + real-data evidence for those paths
is complete.

**What is missing:** two write paths introduced by the 29-REVIEW fix round
(commits `9db22f0`, `8dcdf58`) — i.e. after every `<threat_model>` block was authored —
select rows by **URL-namespace identity alone** and never call `_may_write()`:

| Path | Location | Selection | Write |
|------|----------|-----------|-------|
| `_detach_stale_family_events()` | `campaign_reconciler.py:397-398` | `owned_events(run).exclude(url__in=active_urls)` | `CalendarEventMeta.objects.filter(event__in=stale).update(run=None)` |
| `_delete_owned_calendar_events_on_campaign_run_delete()` (`pre_delete` signal) | `solsys_code/models.py:355-379` | `owned_events(instance)` | `.delete()` — hard-deletes `CalendarEvent` rows, cascading to their `CalendarEventMeta` |

`owned_events()` (`campaign_reconciler.py:111-118`) is a URL-prefix identity check only; it
does **not** consult `CalendarEventMeta.run`, which is the half of `_may_write()` that
distinguishes "keyed in my namespace" from "attributed to someone else".

**Empirical proof** (throwaway test DB, probe module run under `python manage.py test`,
never committed to the repo — `…/scratchpad/audit_probe.py`):

- Probe 1 — an event keyed `RUN:{A.pk}:2026-09-01` whose `CalendarEventMeta.run` is run **B**
  (the exact fixture `tests/test_campaign_reconciler.py:249` proves the reconciler refuses to
  *modify*) → after `run_a.delete()`: `CalendarEvent` exists = **False**, run B's
  `CalendarEventMeta` exists = **False**. The row the write path correctly blocks is hard-deleted
  by the delete path.
- Probe 2 — a stale bare-key `RUN:{A.pk}` event attributed to run **B** → after
  `reconcile_run(run_a)` (a classical run, so the bare key is "stale family"):
  `CalendarEventMeta.run_id` = **None**, i.e. run B's staff-confirmed Phase 28 attribution is
  silently cleared by run A's reconcile.
- Probe 3 — a `load_telescope_runs`-shaped ingest event (blank `url`) adopted and re-keyed by
  run A → after `run_a.delete()`: the ingest-created `CalendarEvent` is **gone**. Before this
  phase, `CalendarEventMeta.run`'s `on_delete=SET_NULL` left that row on the calendar.

**Reachability:** Probe 2's precondition is produced by the reconciler itself — the detach step
returns stale-family events to Phase 28's attribution queue, where staff may attribute them to a
*different* run while the `url` still names the original run's namespace. Both paths require staff
privileges (admin delete / operator sweep), so this is accidental-destruction and
cross-run-attribution loss, not an unauthenticated attack.

**Not patched here** (implementation files are read-only for this audit). Options for the
coordinator:
1. Implement — route both paths through `_may_write()` (e.g. filter the stale/delete set to rows
   where the companion `run` is unset or equals this run), with fixtures mirroring probes 1-3; or
2. Accept — record an explicit accepted risk narrowing T-29-19's declared "every write path"
   wording to the two branch write paths, stating that namespace-keyed rows attributed to another
   run are considered collateral of a staff-initiated delete/reclassify.

#### Re-verification (2026-08-05, second audit pass)

**The mitigation now shipped.** `writable_events(run)` (`campaign_reconciler.py:121-141`) is the
queryset-level twin of `_may_write()`: it narrows `owned_events(run)` by
`Q(telescope_label_meta__isnull=True) | Q(telescope_label_meta__run__isnull=True) | Q(telescope_label_meta__run=run)`.
`owned_events()` is unchanged — `git diff 8dcdf58 HEAD -- solsys_code/campaign_reconciler.py`
removes exactly one line in the whole module (the old detach write), so the read-only consumers
(`test_campaign_approval.py`, the demo notebook) are provably untouched.

| Path | Location | Selection now | Verified |
|------|----------|---------------|----------|
| `_detach_stale_family_events()` | `campaign_reconciler.py:427` | `CalendarEventMeta.objects.filter(event__in=stale, run=run).update(run=None)` — the `run=run` term is the ownership filter | yes |
| `pre_delete` cascade | `models.py:385,387` | `from solsys_code.campaign_reconciler import writable_events` … `writable_events(instance).delete()` | yes |

**Write-path completeness sweep** (the original finding's failure mode was a missed path, so the
sweep was repeated rather than assumed): the only production consumers of `owned_events()` in
`solsys_code/` + `src/` are the two above (grep over `*.py`, `*.html`, `*.ipynb`, `*.rst`,
tests and `docs/_build/` excluded). Every other `CalendarEventMeta` write in the tree carries its
own ownership filter and is not namespace-keyed: `campaign_views.py:1179`
(`event_id=orphan_pk, run__isnull=True`), `:1326` and `:1334` (`event_id=orphan_pk, run_id=run_pk`),
`campaign_reconciler.py:213` (`get_or_create(event=event)` behind `_may_write()`),
`sync_lco_observation_calendar.py:349` (`is_verified` only, never `run`).

**Regression tests exist and are real guards.** `TestCrossRunOwnershipGuards`
(`tests/test_campaign_reconciler.py:705-789`) mirrors probes 1-3:

| Test | Line | Asserts |
|------|------|---------|
| `test_deleting_a_run_never_deletes_an_event_attributed_to_a_different_run` | `:714` | after `run_a.delete()` the `RUN:{A}:2026-08-01` event still exists **and** its `CalendarEventMeta.run_id == run_b.pk` (both the row and the attribution) |
| `test_reconcile_never_detaches_an_event_attributed_to_a_different_run` | `:735` | after a second `reconcile_run(run_a)` the stale `RUN:{A}:2026-09-15` event's `run_id` is still `run_b.pk`, not `None` |
| `test_deleting_a_run_still_deletes_the_events_it_genuinely_owns` | `:758` | no over-narrowing — the container event (`run` = this run), a previously-detached event (`run` unset) and a namespaced event with **no** companion row are all still deleted, preserving WR-01 |

**These tests were not taken on trust.** A throwaway probe module (run under
`PYTHONPATH=…/scratchpad python manage.py test audit_probe_t2919`, never committed) re-executed the
**pre-fix expressions verbatim against the same fixtures**, to rule out tests that pass for the
wrong reason:

- `owned_events(run_a).delete()` (the old `models.py` line) **does** destroy the run-B-attributed
  event in test `:714`'s exact fixture → that test genuinely fails without the fix. The post-fix
  `writable_events(run_a).delete()` on the same fixture leaves both row and attribution intact.
- The detach step was spied on through `reconcile_run(run_a)` to capture the real `active_urls`;
  test `:735`'s stale event **is** inside the pre-fix selection `owned_events(run).exclude(url__in=active_urls)`,
  and the pre-fix `update(run=None)` on that selection clears it → that test genuinely fails without
  the fix.
- `writable_events()` was exercised over all four ownership cases plus an out-of-namespace event:
  it includes no-companion / `run`-unset / `run`-is-this-run, excludes `run`-is-another-run and
  excludes non-`RUN:` events; it returns no duplicate rows (the FK is a `OneToOneField`,
  `models.py:26-32`); and it agrees with `_may_write()` event-for-event. The emitted SQL is a
  `LEFT OUTER JOIN` on `solsys_code_calendareventmeta` — which is what makes the "no companion row
  at all" case (WR-01) survive the join rather than being silently dropped by an inner join.

**Test run:** `python manage.py test solsys_code.tests.test_campaign_reconciler solsys_code.tests.test_reconcile_campaign_runs`
→ `Ran 37 tests … OK` (34 before the fix + the 3 new guards), 2026-08-05, this audit pass.

**Residual behaviours, both re-probed and both non-blocking** (neither is an ownership violation):

- Deleting run A *and then* run B: the cross-attributed event survives **both** deletes; `run`
  is cleared by the FK's `on_delete=SET_NULL`, returning it to Phase 28's attribution queue —
  the documented meaning of an unset `run`. No row is destroyed.
- Original probe 3 is unchanged by design: an ingest-shaped (blank-`url`) event adopted and
  re-keyed by run A has `CalendarEventMeta.run = A`, so it is genuinely writable by A and the
  cascade still hard-deletes it with the run. This is WR-01's intended outcome, not the T-29-19
  gap; the operator-facing documentation gap it leaves is tracked as OBS-29-01.

---

## Unregistered Flags (WARNING — informational, non-blocking)

None of the six `29-0N-SUMMARY.md` files contains a `## Threat Flags` section, so no new attack
surface was declared by the executor. The following surfaced during this audit instead:

| Flag ID | Surface | Introduced | Threat mapping | Note |
|---------|---------|------------|----------------|------|
| UF-29-01 | `_detach_stale_family_events()` — bulk `CalendarEventMeta.run = None` write | `9db22f0` (29-REVIEW CR-01) | folded into T-29-19 | **Resolved 2026-08-05** (`009195f`): the bulk update now carries a `run=run` filter (`campaign_reconciler.py:427`), so it clears only attributions this run already holds |
| UF-29-02 | `pre_delete` signal hard-deleting `owned_events(instance)` | `8dcdf58` (29-REVIEW WR-01) | folded into T-29-19 | **Resolved 2026-08-05** (`009195f`): now cascades through `writable_events(instance)` (`models.py:387`). Still a destructive path reachable from the admin changelist's bulk "Delete selected", but scoped to events the run may write; the operator-facing doc gap remains open as OBS-29-01 |
| UF-29-03 | `CampaignRun.Source.ESO_QUEUE` new enum value + migration | `fb9c70c` (user-directed, plan 29-06) | none | **Assessed, no new untrusted surface:** `source` is absent from `campaign_forms.py` (grep -> zero hits) so no public submission can set it, and `admin.py:211-214` withholds `source` on every `web` row. Staff-only vocabulary widening. |

---

## Observations (non-threat, documentation accuracy)

- **OBS-29-01** — `docs/runbooks/telescope_runs_calendar.rst` documents the reclassification
  *detach* behaviour (`:312-330`) but nowhere states that deleting a `CampaignRun` now deletes its
  calendar events (including an adopted `load_telescope_runs` row). Operator-facing behaviour
  change shipped without runbook coverage. **Still open after the T-29-19 fix** (re-checked
  2026-08-05: `grep -n -i delet docs/runbooks/telescope_runs_calendar.rst` returns only the
  detach paragraph at `:322` and the companion-record prose at `:593-596`). The fix narrows *which*
  events the cascade destroys but does not change the fact that it destroys them, so the runbook
  gap is unaffected. Documentation-only; not a threat.
- **OBS-29-02** — `src/templates/tom_calendar/partials/event_form.html:104` still asserts "no
  production code writes `CalendarEventMeta.run` yet"; as of this phase `_link_event_to_run()`
  writes it on every create and adopt. Stale prose, no functional impact.
- **OBS-29-03** — `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` cell 2 output
  embeds a local absolute path (`/home/tlister/git/fomo_devel/.claude/worktrees/agent-…`). Developer
  path disclosure only; no credentials, no PII.

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-29-01 | T-29-04 | Calendar titles/descriptions expose only campaign name, telescope/instrument, window dates, run status and `observation_details` — the identical field set the retired `_project_calendar_event()` already published. Contact fields are never read by `campaign_reconciler.py`. | Phase 29 plan author (29-01-PLAN.md threat model) | 2026-08-05 |
| AR-29-02 | T-29-11 | The real `CampaignRun` table is tens of rows (44 measured); the sweep is operator-invoked, not request-triggered, and `select_related('site','campaign')` removes the dominant N+1. | Phase 29 plan author (29-03-PLAN.md threat model) | 2026-08-05 |
| AR-29-03 | T-29-13 | `StaffRequiredMixin` + POST-only + fixed action whitelist are untouched by this phase; no new access-control surface (verified against commit `6e9da58`). | Phase 29 plan author (29-04-PLAN.md threat model) | 2026-08-05 |
| AR-29-04 | T-29-18 | The demo notebook seeds clearly-labelled fixture rows idempotently; the dev database is gitignored and disposable, and this execution used an isolated worktree DB. | Phase 29 plan author (29-05-PLAN.md threat model) | 2026-08-05 |
| AR-29-05 | T-29-SC | No package was installed in this phase; `pyproject.toml` is unchanged across the entire phase-29 commit range, so no dependency-legitimacy audit applies. | Phase 29 plan author (all six plans) | 2026-08-05 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-08-05 | 23 | 22 | 1 | gsd-security-auditor (register_authored_at_plan_time: true, asvs_level: 1; verification included a live run of the 34 mitigation tests and a throwaway-DB probe of the two unguarded write paths) |
| 2026-08-05 | 23 | 23 | 0 | gsd-security-auditor — **re-verification of T-29-19 only** after quick task `260805-qdc` (`7070fae`, `009195f`). Outcome: **CLOSED**. Scope deliberately narrow: the other 22 threats were not re-scanned, since the fix commit touches only `campaign_reconciler.py`, `models.py` and `test_campaign_reconciler.py`, and `owned_events()` (their shared dependency) is provably unchanged. Verification: code read of `writable_events()` + both call sites; a repeat write-path completeness sweep for any remaining namespace-only writer; a live run of all 37 reconciler/command tests (OK); and two throwaway probe modules that re-executed the pre-fix expressions against the new tests' own fixtures to confirm the tests are genuine RED-without-the-fix guards, that the join is `LEFT OUTER` (WR-01's no-companion-row case), and that `writable_events()` agrees with `_may_write()` event-for-event. |
| 2026-08-05 | 23 | 23 | 0 | gsd-security-auditor — **re-verification of T-29-07 only**, scoped to quick task `260805-tad` (dispatch-branch removal, `campaign_reconciler.py`). Outcome: **still CLOSED, evidence corrected**. T-29-07's evidence previously cited the now-removed `run.source in QUEUE_SOURCES` branch as the reason the per-night branch was "unreachable" for queue runs; that branch is deleted (it only ever fired for a run that already had a resolved, non-satellite site) and the real, unchanged protection is `_may_write()`'s ownership check, confirmed still the first condition in both `_reconcile_container()` (`campaign_reconciler.py:266`) and `_reconcile_classical_nights()` (`:387`). Tests run: the full 45-test `test_campaign_reconciler` suite plus `test_reconcile_campaign_runs` and `test_campaign_approval` (173 tests total, OK), including the new `TestRecordEventNonInterference` (per-night branch) and `TestContainerRecordEventNonInterference` (container branch) non-interference pair and the `TestReclassificationConvergence.test_pre_fix_container_event_converges_to_per_night_on_next_reconcile` live-shaped convergence test (confirmed to fail when the removed branch was temporarily restored, then reverted). The other 22 threats were not re-scanned: this change removes a branch selector only and touches no ownership check, approval gate, or dry-run code path, so none of them depend on the removed branch. |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed — T-29-19 closed on re-verification 2026-08-05
- [x] `status: verified` set in frontmatter

**Approval:** approved. All 23 threats resolve to closed (18 mitigate verified in code, 5 accepted
risks logged). Three non-blocking documentation observations remain (OBS-29-01 — the runbook still
does not tell operators that deleting a `CampaignRun` deletes its calendar events; OBS-29-02;
OBS-29-03); none gates the phase.
