---
phase: 33-series-identity-reconciler-inversion
verified: 2026-09-08T19:40:00Z
status: gaps_found
score: 80/83 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 49/50
  gaps_closed:
    - "CR-01: the D-13 `tr:target` highlight rule is now inside `{% block additional_css %}` and is actually served (3 tests)"
    - "CR-02: the observing night is anchored at local noon, matching `telescope_runs._local_noon_utc()` (5 boundary tests)"
    - "CR-03 (attribution half): the skip is unconditional; exactly one entry per night is attributed to the run"
    - "WR-01/WR-03: `skipped_nights` and `detached` reach `ReconcileResult` and the sweep's summary + per-run lines"
    - "WR-02: `UNLINK_CLEARED_FIELDS` is the single declaration, consumed by both writers (sentinel-key test proves the loop)"
    - "WR-04: `unlink_event_from_run()` raises `TypeError` on `str`/`bytes`"
    - "WR-05.1/.2/.3: the three month-view tests now discriminate on fixture-specific values"
    - "WR-06: `CalendarEventMetaInline` renders `run` only as a hidden `InlineForeignKeyField`; docstring and runbook corrected"
    - "WR-07: one visibility gate, in `campaign_decoration()` only"
    - "IN-01/IN-03/IN-05: one chip definition with `role=\"img\"`, an `aria-label`, and a no-campaign tooltip that consumes `run_pk`"
    - "IN-02: the window arithmetic lives only in `_reconcile_classical_nights()`"
    - "IN-04: the reconcile notebook's `.delete()` is gated on a pk captured in the same run"
  gaps_remaining:
    - "The abstained `verification: backstop` ordering item (unchanged: still no held-out test, still no production reader)"
  regressions:
    - "CR-04: the CR-03 detach + Phase 28 queue form a confirm/erase loop that erases human `confirmed_by`/`confirmed_at` on every sweep (reproduced independently by this verifier)"
    - "WR-13: a night that is both attributed to this run and contested by another run now reports `skipped_nights=1, blocked=0` and drops out of the active url set — two clauses of 33-08 truth 9 no longer hold in that combination"
gaps:
  - truth: "A human-made attribution of a calendar event to a run survives an unattended reconcile sweep — the reconciler annotates, and never repeatedly clears a run link plus its confirmation stamps that a staff member has (re-)confirmed (phase goal, REQUIREMENTS.md ANNOT-01, reconciler D-17, `unlink_event_from_run()`'s own 'a human attribution always outranks an automated clear')."
    status: failed
    reason: >-
      Independently reproduced (throwaway probe run under the Django test runner, then deleted;
      no source file modified). After the reconciler mints `RUN:{pk}:{date}` and a facility event
      for the same night is later attributed to the run, the sweep detaches the reconciler's own
      event and clears `run`/`confirmed_by`/`confirmed_at`. No `CalendarEventDismissal` row is
      written, `orphan_calendar_events()` treats the row as an orphan, and `candidates_for_event()`
      re-offers it to the SAME run at HIGH band (score 0.82). A staff member draining the Phase 28
      queue confirms it; the next sweep silently erases the stamp again — and again. Observed:
      R2 detached=1 -> candidates [(run 1, 'high', 0.82)] -> staff re-confirm -> R3 detached=1,
      run=None, confirmed_by=None, confirmed_at=None. Only a `logger.warning` records the loss.
      The loop is newly reachable — before 33-08 every night stayed in `active_urls`, so a
      re-attributed `RUN:` event was never stale. It is reachable from an interactive staff surface
      (`_resolve_site()`) as well as from the unattended cron sweep Phase 36 will schedule, and its
      trigger shape (a facility/observation-keyed event attributed to the run) is exactly what
      Phase 34's projector starts producing next phase.
    artifacts:
      - path: "solsys_code/campaign_reconciler.py"
        issue: "`_detach_stale_family_events()` (:469-528) clears a human confirmation with no compensating dismissal row and no once-only guard; the unconditional skip at :425-427 makes it repeat on every sweep"
      - path: "solsys_code/campaign_views.py"
        issue: "`_resolve_site()` (:681-699) drops `result.detached` and reports 'Site resolved — run added to the calendar.' even when created=0 and every night was skipped (WR-12); the approve (:527) and `_set_run_status()` (:759) call sites discard the result entirely"
      - path: "solsys_code/tests/test_campaign_reconciler.py"
        issue: "`test_second_reconcile_detaches_the_superseded_run_keyed_event_and_restore_on_third` (:686-746) clears the FACILITY event's link before the third reconcile, so it never exercises the staff-re-confirms-the-detached-RUN:-event path the queue actually steers operators into"
      - path: "docs/runbooks/telescope_runs_calendar.rst"
        issue: "Tells the operator the released entry is there 'for a human to re-confirm or discard' — re-confirming is the loop trigger, and discarding (a dismissal) never removes the entry"
    missing:
      - "Make the release un-re-offerable or non-repeatable: write the `CalendarEventDismissal` row inside the same write (mirroring `AttributionDecisionView._undo_confirmation()`), or skip an event whose `confirmed_by` is set and count it under a separate operator-visible counter"
      - "A regression test that reconciles -> attributes -> reconciles -> RE-CONFIRMS the detached `RUN:` event to the same run -> reconciles, asserting the stamp survives or the pair is no longer offered"
      - "Surface `result.detached` as a warning at all four staff-action call sites, and key `_resolve_site()`'s message on `created`/`updated`/`skipped_nights` rather than on `skipped_reason is None` alone (WR-12)"
      - "Correct the runbook's 're-confirm or discard' instruction so it does not steer the operator into the destructive path"
  - truth: "A night whose `RUN:`-keyed event is attributed to a DIFFERENT run stays `blocked`, keeps its url in the active set, and is never detached — the foreign attribution survives the refactor (33-08 truth 9, D-02)."
    status: partial
    reason: >-
      Holds in the tested configuration, but not when the night is ALSO attributed to the
      reconciling run through a non-`RUN:` event. `_reconcile_classical_nights()` performs the skip
      `continue` (:425-427) BEFORE `active_urls.add(url)` and before `_may_write()`, so that
      combination reports `skipped_nights=1, blocked=0` and the contested url drops out of the
      active set. Two of the truth's three clauses fail there. The data is still safe — but only
      because of one remaining layer (`unlink_event_from_run()`'s `run_id=run.pk` filter); the
      `blocked` diagnostic that tells an operator 'someone else owns this night's entry' is lost,
      and no test covers the combination.
    artifacts:
      - path: "solsys_code/campaign_reconciler.py"
        issue: "Ownership is evaluated after the skip decision, so the two signals mask instead of composing (:425-436)"
    missing:
      - "Evaluate `_may_write()` on any existing event before the attributed-night skip, keeping the contested url in `active_urls`"
      - "A test asserting `blocked == 1` for a night that is both attributed to this run and carries a foreign-attributed `RUN:` event"
deferred:
  - truth: "The superseded night no longer shows two calendar entries (WR-09 — CR-03's original 'a visibly duplicated night, forever' complaint; the detach removes the attribution, the `CalendarEvent` row survives by design and still renders, now with no campaign chip and no campaign name in its title)"
    addressed_in: "Phase 35"
    evidence: "Phase 35 success criterion 5: 'After the stated cutover step runs, an operator looking at the calendar sees one event per night: no duplicate and no orphan left behind from the old load_telescope_runs events or the reconciler's RUN:{pk}:{date} events'; criterion 3 replaces this handoff with the allocation layer's own link/unlink."
insufficient_spec_items:
  - truth: "No code added by this phase depends on the iteration order of the `observation_group` reverse manager: `CalendarEventMeta` gains no `Meta.ordering`, and no reader added here iterates `group.calendar_event_metas` expecting a stable order (PROJ-04 ordering edge)."
    reason: insufficient_spec
    tier: backstop
    observed: "Unchanged since the 2026-09-04 verification and re-checked against the gap-closure diff: `CalendarEventMeta` still declares no `class Meta` (solsys_code/models.py:12-88), and repo-wide grep finds no production reader of `group.calendar_event_metas` — the only hits are `related_name` declarations and a string assertion on an admin inline prefix in solsys_code/tests/test_admin.py:503-539. Plans 33-06/07/08 added no such reader."
    why_human: "Tagged `verification: backstop` — non-inferable. Absence-by-grep plus symbol presence is explicitly NOT sufficient; only a wired held-out/property-based test (shuffle insert order, assert a stable outcome) or directly observed ordering behavior can confirm it."
flagged_prohibitions:
  - statement: "No notebook cell may delete a row it did not create earlier in the same notebook run, nor leave demo rows behind outside its own demo-scoped reset (33-08 P4 / 33-05 P2)."
    verdict: "partially violated (NON-AUTHORITATIVE LLM-judge verdict)"
    observed: "The delete half is clean — `reconcile_campaign_runs_demo.ipynb`'s only `.delete()` is gated on `blank_url_event_pk`, captured at creation in the same cell (IN-04 closed). But the cell's committed output shows it detaching the PRE-EXISTING `RUN:59:2026-09-02` (pk=335) and leaving it detached in `src/fomo_db.sqlite3`, where it appears in the real attribution queue as a HIGH-band candidate — i.e. a doc artifact seeds CR-04's starting state into the dev DB (IN-08)."
    flag: "unverified-prohibition — human review recommended"
  - statement: "No notebook cell may print or store a run's `contact_person`, `contact_email` or `source` into committed output (33-08 P5 / 33-05 P1)."
    verdict: "technically violated, no PII leaked (NON-AUTHORITATIVE LLM-judge verdict)"
    observed: "`campaign_lifecycle_demo.ipynb` cell 36's committed output prints `contact_person='' contact_email=''` for five demo runs (pre-existing cell, carried forward from 33-05 and flagged in the previous verification). The field names are printed; every value is the empty string, so no contact data is committed."
    flag: "unverified-prohibition — human review recommended"
human_verification:
  - test: "Open the month calendar (`/calendar/`) on a month containing at least one campaign-attributed all-day entry AND one attributed timed entry, across several different proposal fill colours. Look at the ⚑ campaign chip."
    expected: "The chip is legible against every proposal fill (it inherits the entry's foreground via `color: currentColor`), does not compress or clip in the timed entry's flex row (`flex-shrink: 0`), and hovering it shows the campaign name as a tooltip."
    why_human: "Visual legibility and layout across dynamic, data-driven fill colours. Tests assert the CSS declarations and the tooltip attribute are present in the rendered HTML, but cannot judge whether the chip reads clearly on every fill."
  - test: "Click a campaign-attributed calendar entry to open its pop-up, then click 'View campaign ↗' in the 'Attributed campaign run' block."
    expected: "The campaign table page loads scrolled to that run's own row, and the row is visibly highlighted (the `tr:target` rule now actually renders — it moved inside `{% block additional_css %}`)."
    why_human: "The anchor `id=\"run-{pk}\"`, the `tr:target` rule and its presence in the served HTML are all asserted by tests (test_campaign_views.py:634-684), but browser anchor-scroll plus `:target` highlight rendering is real-browser behaviour no server-side test observes."
  - test: "Decide the CR-04 remedy: dismissal-row-on-detach (make the released pair un-re-offerable) vs. once-only detach that yields to a human `confirmed_by` (make it non-repeatable)."
    expected: "A chosen approach recorded as a decision, so the closure plan implements one of the two rather than inventing a third."
    why_human: "A product/audit-policy choice — whether an automated release should be recorded as a dismissal or should simply lose to a human decision — not a code fact."
  - test: "Review the two flagged judgment-tier prohibitions above (notebook residue in the dev DB; `contact_person=''`/`contact_email=''` in committed notebook output)."
    expected: "Each is confirmed as still acceptable, or the deviation is corrected."
    why_human: "unverified-prohibition — judgment-tier prohibitions carry no wired enforcement test; a model verdict is never authoritative for a must-NOT."
  - test: "Review the abstained backstop ordering item: decide whether a held-out test pinning `observation_group` reverse-manager ordering independence is wanted before Phase 34's projector starts writing these links, or accept the absence evidence as-is."
    expected: "Either a held-out/property-based test is added (shuffle insertion order of several `CalendarEventMeta` rows sharing one `ObservationGroup`; assert the consuming code's outcome is unchanged), or the item is explicitly accepted."
    why_human: "`verification: backstop` — non-inferable by design; routing, not diagnosis. reason: insufficient_spec (NOT ordinary manual UAT)."
---

# Phase 33: Series Identity & Reconciler Inversion — Verification Report (re-verification)

**Phase Goal:** `CalendarEventMeta` carries real series identity and attribution links, and the campaign reconciler annotates instead of owning — so the observation projector can land next phase without the campaign layer stealing its events.
**Verified:** 2026-09-08
**Status:** gaps_found
**Re-verification:** Yes — after gap-closure plans 33-06, 33-07 and 33-08

## Headline

The gap-closure wave genuinely closed what it set out to close: 12 of the 13 prior blockers/warnings
are verified closed in the code (not in the SUMMARYs), the full project suite passes (1005 tests, OK),
and both ruff gates are clean. **But 33-08's CR-03 fix introduced a reproducible regression** — the
detach step plus Phase 28's attribution queue form a confirm/erase loop that destroys a human's
`confirmed_by`/`confirmed_at` on every subsequent sweep. I reproduced it independently (throwaway probe
module, run then deleted; no source file modified), so this is not taken on the review's word.

That is a goal-level failure, not just a code-quality finding. The phase goal is "the reconciler
**annotates instead of owning**"; D-17 in the reconciler's own module header says a set
`CalendarEventMeta.run` "means the event is ATTRIBUTED to that run, never that the run OWNS it"; and
`unlink_event_from_run()`'s docstring states "a human attribution always outranks an automated or
stale-POST clear." The shipped behaviour is the opposite for the reconciler's own namespace: the
machine overrides the human, silently, on every sweep, and the runbook instructs the operator straight
into it. The trigger shape — a facility/observation-keyed event attributed to the run — is precisely
what Phase 34's projector starts producing next phase.

## Goal Achievement

### ROADMAP Success Criteria (the contract)

| # | Success criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Event links to `ObservationRecord`/`ObservationGroup` by real FKs; every companion row survives the migration with `run`/`is_verified`/`confirmed_by` intact | ✓ VERIFIED | `models.py:54-68` declares `observation_record` (OneToOne, SET_NULL) and `observation_group` (FK, SET_NULL); `migrations/0017_calendareventmeta_observation_links.py` is two nullable `AddField`s plus an `AlterField` on `run`'s verbose name/related_name only; `test_calendar_event_meta_links.py:229/240` assert pre-existing values survive byte-identical and both new columns are NULL on every migrated row |
| 2 | `reconcile_campaign_runs` no longer adopts, re-keys or detaches any event **outside** the reconciler's own `RUN:` namespace — an attributed event keeps its key and its fields | ✓ VERIFIED | The attributed-night skip `continue`s before any write (`campaign_reconciler.py:425-427`); `test_attributed_night_is_skipped_and_event_untouched` asserts a 7-field byte-identical snapshot; `test_facility_url_keyed_attributed_event_skips_its_night` covers the Phase 34 url shape. My probe confirms the facility event's url/fields/link/stamps are untouched across three sweeps |
| 3 | The campaign decoration is rendered from the link and survives a from-scratch rewrite of the event's title/description | ✓ VERIFIED | `campaign_decoration()` (`calendar_display_extras.py:433-489`) reads only — no `.save()`/`.update()`/`.create()`/`get_or_create()`/`.delete()` anywhere in the module; `test_decoration_survives_from_scratch_rewrite_of_title_and_description` (test_calendar_template.py:898) |
| 4 | Clearing `CalendarEventMeta.run` removes only the decoration; the event is untouched and nothing is deleted | ✓ VERIFIED | `unlink_event_from_run()` issues a single `CalendarEventMeta.objects.filter(...).update(**UNLINK_CLEARED_FIELDS)` — three keys, no `CalendarEvent` write, no delete; 12 unlink/undo tests in `test_campaign_attribution_views.py` |
| — | **Derived goal truth**: a human-made attribution survives an unattended sweep (the "annotates instead of owning" clause) | ✗ FAILED | Reproduced confirm/erase loop — see Gap 1 |

### Gap-closure plan truths

**33-06 — display layer (11/11 verified)**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `tr:target` rule served to staff and anonymous | ✓ | `campaignrun_table.html:5-20` inside `{% block additional_css %}`; `tom_common/base.html:18` really declares that block; `test_staff_get_contains_tr_target_highlight_rule` / `test_anonymous_...` |
| 2 | Still served with zero run rows | ✓ | `test_empty_campaign_still_serves_tr_target_highlight_rule` |
| 3 | `&`, `<`, `"` escaped identically in `title=` and `aria-label=` | ✓ | Chip relies wholly on autoescape (no `\|safe`); `test_campaign_name_encoding_edge_escapes_consistently_in_title_and_aria_label` |
| 4 | No-campaign chip names the run distinctly | ✓ | `campaign_chip.html:23` renders `Attributed run #{{ deco.run_pk }} {{ deco.campaign_name }}`; `test_no_campaign_run_renders_marker_and_no_table_href` asserts that exact string |
| 5 | Chip carries `aria-label` and `role="img"` | ✓ | `campaign_chip.html:21,23` |
| 6 | `event_form.html` has no second visibility gate | ✓ | Single `{% if deco %}` at :140-141; no `is_publicly_visible` in the template |
| 7 | `run_pk` has a rendered consumer | ✓ | The no-campaign chip branch |
| 8 | WR-05.1 sensitivity | ✓ | `test_pending_review_run_shows_no_marker_for_staff_and_anonymous` asserts absence of `title="{pending_campaign.name}"` — a value only the pending fixture produces, so deleting the gate makes it fail |
| 9 | WR-05.2 sensitivity | ✓ | Fixture titles are exactly 18 (`AllDayAttrEighteen`) and 16 (`TimedAttrSixteen`) chars, matching their `truncatechars` budgets; the chip is a sibling `{% include %}`, not inside the filter expression |
| 10 | WR-05.3 sensitivity | ✓ | Asserts that run's own tooltip string, not the shared `cal-campaign-chip` class |
| 11 | Page-1-only anchor pinned by test, not a silent dead link | ✓ | `TestCampaignRunAnchorPagination` (`test_campaign_views.py:686-730`) + runbook §"That 'View campaign ↗' link carries the run's row anchor but no page number" |

**33-07 — one declaration of what clearing an attribution means (7/7 verified)**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `UNLINK_CLEARED_FIELDS` declared once, both writers derive from it | ✓ | `campaign_utils.py:868-872`; consumed at `:946` (`.update(**...)`) and `admin.py:430-433` (loop) |
| 2 | A fourth key reaches the admin path with no `admin.py` edit | ✓ | `test_admin.py:342` iterates the exported set; `:373` `patch.dict`s a `_wr02_sentinel` key and asserts the admin clears it — this distinguishes a loop from three hand-written assignments |
| 3 | Admin standalone clear nulls every declared field, leaves `is_verified`/`observation_record`/`observation_group`/event untouched | ✓ | `admin.py:406-434` branch 2 + tests |
| 4 | `str`/`bytes` raises `TypeError`, writes nothing | ✓ | `campaign_utils.py:928-937`, checked after the `run_pk` guard |
| 5 | `int` and queryset branches both directly asserted | ✓ | Dedicated tests in `test_campaign_attribution_views.py` |
| 6 | Inline renders no editable attribution field | ✓ | `fk_name = 'run'` (`admin.py:110`); `test_admin.py:537-539` asserts `type="hidden" name="calendar_event_metas-0-run"` present and `<select name="calendar_event_metas-0-run"` absent |
| 7 | Inline docstring states the operation as it exists | ✓ | `admin.py:74-106`; mirrored in the runbook's inline bullets |

**33-08 — noon anchor, unconditional skip, detach (13/14 verified, 1 partial)**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Night anchored at local noon, same as `_local_noon_utc()` | ✓ | `_observing_night()` (`campaign_reconciler.py:309-337`) does `(local - timedelta(hours=12)).date()`; `telescope_runs._local_noon_utc()` anchors at `time(12, 0)` in the site zone. Both use wall-clock arithmetic |
| 2 | 12:00:00 vs 11:59:59 boundary exact on both sides | ✓ | `test_exact_local_noon_boundary_belongs_to_the_date_that_just_started`, `test_one_second_before_local_noon_belongs_to_the_previous_date` |
| 3 | Negative-offset site matches the observing night, not the UTC date | ✓ | `test_negative_utc_offset_site_resolves_by_observing_night_not_naive_utc_date` |
| 4 | No attributed event ⇒ skips nothing, detaches nothing, one event per night (incl. single-night) | ✓ | `test_no_attributed_events_skips_nothing_and_mints_one_event_per_night`, `test_single_night_window_with_no_attribution_creates_exactly_one_event` |
| 5 | Skip fires regardless of ordering; both orderings converge | ✓ | Skip is unconditional at `:425-427`; probe R2 = `skipped_nights=1, detached=1` |
| 6 | Superseded `RUN:` event is DETACHED, never deleted; row survives; stamps cleared; back in the queue | ✓ | Probe: event pk survives with its url, `run`/`confirmed_by`/`confirmed_at` all None, `is orphan: True`. (This truth is met exactly as written — its downstream consequence is Gap 1) |
| 7 | Exactly one entry attributed to the run for that night after the second reconcile | ✓ | Probe + `test_second_reconcile_detaches_...` assert `CalendarEventMeta.objects.filter(run=run).count() == 1` |
| 8 | Clearing the attributed link restores the reconciler's entry in place (same pk, same url, created 0) | ✓ | Third-reconcile half of `test_second_reconcile_detaches_the_superseded_run_keyed_event_and_restore_on_third` |
| 9 | A night whose `RUN:` event is attributed to a DIFFERENT run stays blocked, keeps its url active, is never detached | ⚠️ PARTIAL → counted FAILED | Holds when tested in isolation (`test_blocked_night_keeps_its_url_active_and_is_never_detached`), but the skip `continue`s before `active_urls.add()` and before `_may_write()`, so an attributed-AND-contested night reports `blocked=0` and drops out of the active set. Data still safe via the helper's `run_id` filter. See Gap 2 |
| 10 | `_detach_stale_family_events()` returns the count, warns, and it reaches `ReconcileResult.detached` | ✓ | `:513-528`; probe shows the warning line and `detached=1` |
| 11 | The sweep reports `skipped_nights` and `detached` in the summary and per-run lines | ✓ | `reconcile_campaign_runs.py:76-112`. (The four staff-action call sites do not — see WR-12; the truth names only the command) |
| 12 | Window arithmetic computed in exactly one place | ✓ | `n_nights` only in `_reconcile_classical_nights()`; `reconcile_run()` consumes the returned `active_urls` |
| 13 | Notebook has no unconditional `.delete()` | ✓ | The single `.delete()` is gated on `blank_url_event_pk` captured at creation, with an assert and a printed banner |
| 14 | Both notebooks re-executed and committed with output; runbook describes the new counters, noon anchor, detach-on-supersede, the inline's real operation, and the page-1 anchor | ✓ (with WR-09/WR-10 accuracy warnings) | Commit `40109b8`; 9/9 and 18/18 code cells carry output showing post-fix values (`skipped_nights=1, detached=1`); all five runbook topics present at lines 666-690, 745-770, 800-815, 845-875 |

**33-01..33-05 regression check (49 previously verified truths):** spot-checked, no regressions.
The carrier fields, migration proof, decoration-survives-rewrite, PII exclusions, N+1 guard, unlink
semantics and the paired-docs deliverables are all still in place and all covered by the passing suite.

### Requirements Coverage

| Requirement | Source plans | Status | Evidence |
|-------------|--------------|--------|----------|
| PROJ-04 (Phase 33 half: carrier FKs) | 33-03 | ✓ SATISFIED | `observation_record` / `observation_group` on `CalendarEventMeta`, migration 0017, read-only in both admin surfaces, 8 link tests. Ordering edge abstained (backstop, see `insufficient_spec_items`) |
| ANNOT-01 | 33-01, 33-08 | ⚠️ PARTIAL | Satisfied for the milestone's actual landmine — no adopt, no re-key, no touch of any non-`RUN:` event (SC 2 verified). NOT satisfied for its literal text "no longer ... detaches an event attributed to a run": inside the reconciler's own namespace a human's re-attribution is detached on every sweep (Gap 1) |
| ANNOT-02 | 33-01, 33-02, 33-06 | ✓ SATISFIED | Decoration rendered from the link at request time, survives a from-scratch field rewrite, one chip definition with an accessible name, one visibility gate, escaping and empty-input edges tested |

No orphaned requirement IDs: REQUIREMENTS.md maps exactly PROJ-04 / ANNOT-01 / ANNOT-02 to Phase 33,
and every one is claimed by a plan.

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `campaignrun_table.html` `<style>` | `tom_common/base.html:18` | `{% block additional_css %}` | ✓ WIRED (block confirmed present in the installed `tom_common`) |
| `campaign_decoration(event)` | `campaign_chip.html` | `{% include %}` from both month-grid loops (`calendar.html:252, :279`) | ✓ WIRED |
| `campaign_decoration(event)` | `event_form.html` | single `{% if deco %}` gate (:140) | ✓ WIRED |
| `UNLINK_CLEARED_FIELDS` | `unlink_event_from_run()` | `.update(**...)` (`campaign_utils.py:946`) | ✓ WIRED |
| `UNLINK_CLEARED_FIELDS` | `CalendarEventMetaAdmin.save_model()` | local import + loop (`admin.py:430-433`) | ✓ WIRED |
| `CalendarEventMetaInline.fk_name` | Django inline formset field exclusion | hidden `InlineForeignKeyField` | ✓ WIRED (asserted on rendered HTML) |
| `_observing_night()` | `_attributed_nights()` | single call site (`:371`) on `event.start_time` | ✓ WIRED |
| `_reconcile_classical_nights()` active urls | `_detach_stale_family_events()` | `reconcile_run()` (`:560-574`) | ✓ WIRED |
| `_detach_stale_family_events()` | `ReconcileResult.detached` → sweep summary | `result._replace(detached=...)` | ✓ WIRED |
| `_detach_stale_family_events()` | Phase 28 queue (`orphan_calendar_events` / `candidates_for_event`) | **no `CalendarEventDismissal` written** | ✗ MISSING LINK — this is Gap 1's mechanism |
| `reconcile_run()` result | four staff-action call sites | `result.detached` | ✗ NOT WIRED (WR-12) |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|-----------|---------|--------|--------|
| Full project suite | `python manage.py test $LABELS` (config `workflow.test_command`, first half) | `Ran 1005 tests in 457s — OK` | ✓ PASS |
| Lint gate (D-07) | `pre-commit run ruff --all-files` | Passed | ✓ PASS |
| Format gate (D-07) | `pre-commit run ruff-format --all-files` | Passed | ✓ PASS |
| CR-04 confirm/erase loop | throwaway probe module under `solsys_code/tests/`, run then deleted | `R2 detached=1` → candidates `[(1,'high',0.82)]` → staff re-confirm → `R3 detached=1, run=None, confirmed_by=None, confirmed_at=None`; two entries remain on the night | ✗ FAIL (defect reproduced) |
| Notebook outputs are post-fix | JSON inspection of both notebooks | 9/9 and 18/18 code cells carry output; reconcile demo prints `skipped_nights=1, detached=1` | ✓ PASS |

### Anti-Patterns / Review Findings weighed against what the phase promised

| Finding | Severity here | Is it a gap against the phase's promise? |
|---------|---------------|------------------------------------------|
| CR-04 confirm/erase loop | 🛑 BLOCKER | **Yes.** Contradicts the goal's "annotates instead of owning", ANNOT-01's literal text, and the module's own D-17/helper docstrings. Reproduced independently. Gap 1 |
| WR-13 blocked signal swallowed | 🛑 gap (partial) | **Yes** — two clauses of declared truth 33-08.9 do not hold in a reachable configuration. Gap 2 |
| WR-12 staff actions surface neither counter; "run added to the calendar" is false when nothing was added | ⚠️ WARNING | Partly — no must-have names the staff surfaces, but it is (a) CR-04's interactive reach and (b) a user-visible message regression created by 33-08's unconditional skip. Folded into Gap 1's closure list |
| WR-09 duplicate entry survives; runbook's remedy removes neither | ⚠️ WARNING | Not a gap against this phase's must-haves (33-08 truth 7 was scoped to *attribution*, and it holds) — **deferred to Phase 35 SC 5**, which explicitly promises no leftover `RUN:{pk}:{date}` duplicates after cutover. The misleading "re-confirm or discard" wording is folded into Gap 1 |
| WR-10 `detached` described as one cause, counts two | ⚠️ WARNING | No. Truth 33-08.14 required the counters to be described, and they are; the description is incomplete, not absent. Recommend fixing with Gap 1 since both touch the same operator wording |
| WR-11 `--dry-run` cannot preview the detach | ⚠️ WARNING | No must-have required it, and the runbook documents the behaviour. But the reviewer is right that the count is a pure read and the one irreversible step is the one the preview hides — worth doing in the closure plan |
| WR-08 page-1-only anchor (carried forward) | ℹ️ accepted | No. Plan 33-06's must-have was explicitly "pinned by a test rather than a silent dead link", and it is (2 tests + runbook). Recommend a backlog item, not a gap |
| IN-06..IN-11 | ℹ️ INFO | No. Docstring drift, a `setUpTestData` without `super()`, `bool`/`0` pk edges, chip aria-label prefix inconsistency, notebook residue, DST-fixture thinness. IN-08 also feeds the flagged notebook prohibition |
| Debt markers (`TBD`/`FIXME`/`XXX`) in phase-modified files | ✓ clean | The only `TBD` occurrences are the `CampaignRun` "TBD window" domain vocabulary, not debt markers |

### Human Verification Required

5 items — see `human_verification` in the frontmatter (chip legibility across fills; browser anchor
scroll + `:target` highlight; the CR-04 remedy decision; the two flagged judgment-tier prohibitions;
the abstained backstop ordering item).

## Gaps Summary

Two gaps, one root cause between them: `_reconcile_classical_nights()` now makes the night's outcome
decision (skip) *before* it establishes ownership and before it records the url as active, and
`_detach_stale_family_events()` then acts on that decision with a write that erases human audit data
and leaves no trace the rest of the system can read.

- **Gap 1 (BLOCKER)** — the released pair is immediately re-offered to the run that released it, so a
  human confirmation is destroyed on every sweep. The fix is one of two shapes (write the dismissal
  row, or let a human `confirmed_by` outrank the automated release), plus the regression test that the
  existing test stops one step short of, plus the four staff-action surfaces and the runbook wording.
- **Gap 2 (PARTIAL)** — evaluate ownership before the skip so `blocked` and `skipped_nights` compose
  instead of masking, and keep a contested url in the active set.

Everything else the gap-closure wave promised is in the code, wired, and covered by a passing suite.
Phase 34 should not start on top of Gap 1: the projector's own attributed events are exactly what makes
the loop routine rather than rare.

---

_Verified: 2026-09-08_
_Verifier: Claude (gsd-verifier)_
_Supersedes: 33-VERIFICATION.md of 2026-09-04 (49/50, human_needed)_
