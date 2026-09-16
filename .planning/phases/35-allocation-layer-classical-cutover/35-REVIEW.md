---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-16T19:09:02Z
depth: deep
iteration: 10
diff_base: 1f35efe7ed85ca719ec9bd6c41fcc9aff7008bfb
files_reviewed: 10
files_reviewed_list:
  - docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/allocation_projector.py
  - solsys_code/campaign_reconciler.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/reconcile_campaign_runs.py
  - solsys_code/migrations/0021_alter_calendareventmeta_is_verified_and_more.py
  - solsys_code/models.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_reconcile_campaign_runs.py
findings:
  critical: 1
  warning: 11
  info: 7
  total: 19
status: issues_found
---

# Phase 35: Code Review Report (iteration 10 — sixth gap-closure round)

**Reviewed:** 2026-09-16T19:09:02Z
**Depth:** deep
**Files Reviewed:** 10
**Status:** issues_found

## Summary

**Both of iteration 9's criticals are genuinely closed**, verified line-by-line at HEAD
(`0bc1ccd`) rather than against the SUMMARY claims.

| Iteration-9 finding | Verdict at HEAD |
|---|---|
| **CR-04** (declined re-mint froze title/description/target_list forever) | **CLOSED.** The `continue` is gone. `allocation_projector.py:1256-1280` increments `remint_declined` and falls through; the `else` arm at `:1281-1318` is the only one that `continue`s. Execution reaches `:1329`'s `existing is None` test, takes the else at `:1357`, and writes `title`/`description`/`target_list` at `:1435`. Five tests (`TestDeclinedRemintStillUpdatesLabels`) pin the `[CANCELLED]` title, the untouched pk/boundaries, the surviving `confirmed_by` stamp, the `remint_declined + updated` pair, the second-sweep `unchanged`, and dry-run parity. |
| **CR-05** (retirement branch deleted the `ALLOC:` night with no human guard) | **CLOSED for the delete itself.** `:1182-1208` now splits `existing` two ways through `_clearable_declined_and_unattributed()` before `:1199`'s `existing.delete()`, logs `Allocation retire declined: night pk=...`, counts `detach_declined`, and withholds `retired`. The deliberate narrowing (only `confirmed_by`, never `is_verified=False`/an observation link) is stated at the call site, in `_remint_decline_reason()`'s Cross-reference paragraph, and pinned by `test_is_verified_false_with_no_confirmation_still_retires`. Six tests including a dry-run parity case. **But the night the guard now keeps alive is never refreshed again — see CR-01 below.** |

Also closed: **WR-06** (the counter is genuinely split — `ReconcileResult.remint_declined`
exists at `campaign_reconciler.py:114`, `reconcile_run()`'s `_replace()` at `:912-917`
preserves it, and both operator surfaces got their own truthful message,
`reconcile_campaign_runs.py:134-141` and `campaign_views.py:476-484`, with
`detach_declined`'s wording left byte-identical); **WR-08** (`models.py:22-38` and
`:82-101` now document the `is_verified` veto, and migration 0021 ships the corrected
`verbose_name`/`help_text`); **IN-05** (`active_urls.add(url)` removed from the decline
branch, with a comment at `:1277-1280` and `:1209-1212` explaining why it is redundant in
*both* decline branches — the distinction IN-05 asked to be got right, got right); and
**IN-06** (`_remint_decline_reason()`'s rule 2 now names `is_verified` as the
production-reachable arm). **WR-05** and **WR-07** are partially closed and carried forward
with only their residue. **WR-01/WR-02/WR-03/WR-04, IN-01 and IN-03 are unchanged** and are
carried forward below.

**This round opened one critical defect and seven new warnings.** The critical is CR-04's
own defect, one branch over: 35-23 created a brand-new *permanent-retention* path (the
declined retirement) and did not give it the fall-through it had just finished arguing was
mandatory — so a night the sweep now deliberately keeps is frozen at the labels it had when
it was confirmed, forever (CR-01). The warnings cluster around 35-24's site-position
fingerprint: the `v2`→`v3` bump cannot re-audit a fully-set sub-night run at all, so the
WR-05 dark-window refresh it was built to enable is unreachable for every night that
already exists (WR-02); the new fingerprint path reuses a log message that calls a
current-format recorded token "unrecorded-provenance" (WR-04); and `project_allocation()`'s
own docstring still states the D-13 absolute the same round deliberately carved an
exception into (WR-03).

### Verification method

- Read `allocation_projector.py` in full (1632 lines) at HEAD, plus every cross-module
  symbol its new code touches: `_may_write()`, `_clearable_and_declined()`,
  `_clearable_declined_and_unattributed()`, `_split_stale_owned_events()`,
  `owned_events()`, `_stale_attributions()`, `_stale_dated_events()`,
  `_stale_allocation_events()`, `reconcile_run()`'s aggregation, and
  `Observatory.to_earth_location()`.
- Traced the CR-04 fall-through by hand for all four reachable shapes (null/null,
  half-null, set/set, dry-run) and its previously-unconsidered intersection with 35-24's
  `refresh_dark_window` (WR-01).
- Traced `_span_needs_remint()`'s four steps for every combination of sub-night shape ×
  token state (absent / `''` / `v2` / `v3`-wrong-part-count / `v3`-current) × site change
  (swap / in-place position / in-place timezone) to establish which populations the `v3`
  bump can and cannot reach (WR-02).
- Checked for a double count of the newly-surviving declined `ALLOC:` night in
  `reconcile_run()`'s downstream convergence: `owned_events()` is `RUN:`-namespace-only and
  `_stale_allocation_events()` returns early for a per-night-dispatched run, so there is
  none. This is correct; recorded here because it is the NF-09 failure mode a new
  retention path invites.
- Confirmed `ReconcileResult`'s new field is only ever constructed by keyword
  (`ReconcileResult(**totals)`, `ReconcileResult(**{action: 1})`, `_replace()`), so the
  insertion between `detach_declined` and `retired` breaks no positional caller.
- Read the notebook's stored outputs programmatically: 47 cells, every code cell carries a
  non-null `execution_count` and at least one real output; no placeholder cells.
- Read the full runbook diff and re-derived each new operator claim against the code.
- Line-length check (120 cols) clean on all eight changed Python files; the one >120 line is
  in migration 0021, which `pyproject.toml` per-file-ignores for `E501`.
- No file other than this REVIEW.md was written; no test suite was run (per instructions).
- Iteration-7 deferrals recorded as `user_deferred:` in `35-VERIFICATION.md` are not
  re-raised.

## Structural Findings (fallow)

No structural pre-pass was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: the retirement decline this round added keeps the night alive but never refreshes it again — `mark_cancelled` never reaches a confirmed retired night, which is CR-04's defect in the branch created to fix CR-05

**File:** `solsys_code/allocation_projector.py:1182-1213` (specifically the unconditional
`continue` at `:1213`)

**Issue:** Before this round the retirement branch always ended in a delete, so "a retired
night keeps stale labels" was not a reachable state. 35-23 created that state — a confirmed
night now *survives* its own retirement — and left the branch's `continue` at `:1213`
untouched. The declined night therefore never reaches the plain-update path at `:1399-1442`
that writes `title`, `description` and `target_list`, and the decline is not transient:
nothing about it changes on a later sweep, so the same night is declined and skipped again,
indefinitely.

This is the identical argument 35-23 itself made three commits earlier for the re-mint
branch, and which iteration 9 rated Critical. `allocation_night_description()`'s docstring
(`:211-216`) still states the property being broken:

> "reused deliberately so a staff ``mark_cancelled``/``mark_weather_failure`` action reaches
> allocation nights the same way it reaches container events."

Trace (every step is an ordinary staff action, and the first two are exactly the ones
`TestRetirePathAllocationEventGuard.test_retiring_a_night_never_deletes_a_human_confirmed_alloc_event`
already performs):

1. Staff confirms an `ALLOC:` night's attribution (`CalendarEventMetaAdmin.save_model()`
   stamps `confirmed_by`/`confirmed_at`).
2. Anyone links an `ObservationRecord` with a placed block on that night.
   `receiver_on_run_observation_save()` fires immediately — no sweep needed — and
   `night in retired` routes to `:1118`. The delete is declined (correct), `detach_declined`
   is counted (correct), and `:1213` `continue`s (the defect).
3. The run is later cancelled, or its `campaign` changes, or its
   `telescope_instrument` is corrected. `allocation_night_title(run)` would now return
   `'[CANCELLED] NTT EFOSC2'`.
4. Step 2's decision repeats on every subsequent sweep, so step 3's write never happens.
   The calendar shows a cancelled run's night as an ordinary observing night, permanently,
   *and* the runbook (`:1178-1190`) has just told the operator to expect that entry to sit
   there beside the observation's own entry until someone clears the confirmation — so the
   stale one is the entry an operator is being trained to leave in place.

The severity matches iteration 9's CR-04 exactly: admin-reachable, silent, permanent stale
calendar data, in an event the sweep reports no counter for (`retired` is deliberately
withheld at `:1207`, and `updated`/`unchanged` are never reached).

**Fix:** give the retirement decline the same two-way split the re-mint decline just
received. Replace the unconditional `continue` at `:1213` with a fall-through for the
declined case only, so the ordinary update path still refreshes the three non-destructive
fields:

```python
            if existing_deletable:
                totals['retired'] += 1
                continue
            # Declined: the night survives, so it must keep receiving the ordinary
            # title/description/target_list refresh -- the mechanism by which a staff
            # mark_cancelled reaches an allocation night (CR-04's own argument, applied to
            # the branch CR-05 created). retired_urls already keeps it out of the D-14
            # convergence step; active_urls must NOT be added, or the night would be treated
            # as live in the RUN:-namespace convergence too.
            if existing is None:
                continue
            # fall through to the plain-update path below
```

which needs the plain-update path reached with `existing` bound (the simplest shape is to
hoist `:1399-1442` into a small `_refresh_labels(run, existing, url, dry_run)` helper and
call it from both the decline branch and the main path, rather than restructuring the loop).
Add `test_declined_retirement_still_receives_a_cancelled_title` and a dry-run parity case,
mirroring `TestDeclinedRemintStillUpdatesLabels`. Whatever counter the refresh reports
(`updated`/`unchanged` alongside `detach_declined`) must be stated in the runbook's
`detach_declined` section, exactly as the `remint_declined` section already states its own
pair.

If instead the deliberate decision is that a superseded-but-confirmed night is frozen on
purpose, that has to be written down at `:1203-1212`, in the runbook's `detach_declined`
section and in `allocation_night_description()`'s docstring (whose stated property it
contradicts) — and pinned by a test asserting the frozen title, so the next reader does not
read it as the oversight it currently looks like.

## Warnings

### WR-01: a declined re-mint that also has a moved site records a provenance token for boundaries the sweep did not mint — and the comment authorising the write asserts a premise that is false on exactly that path

**File:** `solsys_code/allocation_projector.py:1358-1373` (the comment and the
`refresh_dark_window` expression), `:1437-1441` (the write)

**Issue:** `:1362-1368` justifies recording a current-format token on the plain-update path
like this:

> "for a fully-set run, step 1 of `_span_needs_remint()` has already compared both stored
> boundaries against what the current site and the current sub-night fields produce and
> found them equal on THIS SAME sweep, so refreshing the description and recording the
> current token below are both claims this sweep just proved."

That is true of the path 35-24 was written against (`_span_needs_remint()` returned False),
and false of the path 35-23 opened three commits earlier. After CR-04's fall-through, a
fully-set run reaches `:1369` having had `_span_needs_remint()` return **True** — and for a
fully-set run, True can only come from step 1's boundary comparison *disagreeing*
(`:695-696` returns False for every other both-set outcome). So the premise is inverted:
step 1 proved the boundaries do NOT match.

Reachable shape: a run with `night_start_utc`/`night_end_utc` both set, whose `ALLOC:` night
is confirmed (or `is_verified=False`), where an operator corrects the `Observatory` position
in place *and* edits a sub-night field in the same window. `refresh_dark_window` is True
(`_site_provenance_differs()` compares only the site components), the re-mint is declined,
and `:1441` writes `v3|{site}|{new_fp}|{new_start}|{new_end}` onto a night whose
`start_time` is still the old value — the false-provenance claim round 2 was reverted for,
and which `_span_needs_remint()`'s step-4 docstring (`:661-665`) explicitly forbids.

The consequence is contained rather than harmful today — step 1 re-catches the mismatch on
every subsequent sweep for as long as both fields stay set, and a later transition to
half-null/null-null compares the recorded sub-night sides and still returns True — but
nothing in the code says so, no test covers the intersection (every
`TestSetWindowSiteCorrection` case has an unconfirmed night, every
`TestDeclinedRemintStillUpdatesLabels` case has an unchanged site), and the next reader is
told the opposite by the comment.

**Fix:** make the write conditional on the fact it claims. Compute the decline in a variable
the update path can see and gate the provenance write on it:

```python
        if existing is not None and _span_needs_remint(run, night, existing, dry_run=dry_run):
            decline_reason = _remint_decline_reason(run, existing)
            ...
            remint_declined = decline_reason is not None
        else:
            remint_declined = False
        ...
            if refresh_dark_window and not remint_declined:
                _record_sub_night_provenance(event, _sub_night_provenance_token(run))
```

and correct `:1362-1368` to state both entry paths ("…unless this night arrived here through
a declined re-mint, in which case step 1 proved the opposite and nothing may be recorded").
Add `test_declined_remint_with_a_moved_site_records_no_provenance`.

---

### WR-02: the `v3` bump can never re-audit a fully-set sub-night run, so WR-05's dark-window refresh is unreachable for every night that already exists — and both the constant's comment and the runbook say the opposite

**File:** `solsys_code/allocation_projector.py:88-99` (`_PROVENANCE_TOKEN_VERSION`'s comment),
`:695-696` (the step-2 short-circuit), `:757-801` (`_site_provenance_differs()`);
`docs/runbooks/telescope_runs_calendar.rst:1107-1140`

**Issue:** `_PROVENANCE_TOKEN_VERSION`'s comment states the migration mechanism as fact:

> "The bump is what makes every `v2|` token already stored read as unrecorded, so each such
> night resolves once through the bounded legacy branch (`_span_needs_remint()` step 4) and
> re-records in the current `v3` format"

For a run with **both** sub-night fields set, step 2 (`:695-696`) returns before step 3 ever
reads the token, so step 4 is unreachable — and every other writer of
`minted_sub_night_window` is unreachable too for such a night: `:1317` is the re-mint path
(not taken, the boundaries match), `:1433` is the create path (the event exists), and
`:1441` requires `refresh_dark_window`, which requires `_site_provenance_differs()`, which
returns False for any token failing the `v3`-and-5-parts test (`:796-797`). The set is
closed: **a fully-set run's night that is carrying a `v2` token or no token can never
acquire a `v3` one.**

The population that describes is precisely the one WR-05 was raised about — a classical run
with a fixed `1130-0530` window, the shape `night_start_utc`/`night_end_utc` exist for — and
it is every such night in the database at upgrade time. For all of them,
`_site_provenance_differs()` returns False forever, so the dark-window refresh 35-24 added
never fires, and the stale `Dark window (-15 deg, UTC): ...` line WR-05 reported stays stale
permanently. `_site_provenance_differs()`'s own docstring (`:773-781`) states the
unrecorded-token limitation honestly, but describes the operator escape as "clear one
sub-night field, which routes it through the re-mint path" — i.e. deliberately corrupt the
run's data and revert it, which also destroys and re-creates the night. That is not a
remedy an operator can be told to run.

The runbook then states the un-caveated version twice: `:1107-1112` ("a run with BOTH fields
set … a same-timezone site correction changes only the dark-window line, **refreshed
automatically on the next sweep**") and `:1128-1140` ("now re-mints **every** allocation
night already projected at that site").

**Fix:** pick one.
1. Let a fully-set run record provenance without a re-mint, by moving the `:695-696`
   short-circuit *after* step 3's component-wise comparison and returning False there when
   only the site components match — the token then re-records through step 4's
   within-tolerance write, at the same bounded one-`sun_event()`-per-night cost the runbook
   already documents for reason (5); or
2. accept the limitation and say so at all three sites: `_PROVENANCE_TOKEN_VERSION`'s
   comment ("the re-audit reaches null/null and half-null runs only; a fully-set run's
   boundaries are pinned so it needs no re-audit, at the cost of never re-recording, which
   means the dark-window refresh below applies only to nights minted after this release"),
   and both runbook paragraphs.

Either way add a test that mints a fully-set night, forces its token to `v2`/NULL (the
post-upgrade shape), applies a same-timezone position correction, and asserts the outcome
the docs claim.

---

### WR-03: `project_allocation()`'s docstring still states the D-13 absolute the same round carved an exception into, and still describes the update path's field authority as it was before

**File:** `solsys_code/allocation_projector.py:1028-1036`

**Issue:** The module's main entry point documents two rules that HEAD no longer obeys:

> "``sun_event()`` (both ``'sun'`` and ``'dark'``) is called only when a brand-new night is
> being minted -- never on the update or re-key paths (D-13)"
>
> "On **update** (including a re-key), writes only ``title``, ``description`` (with the
> preserved dark-window line) and ``target_list``."

`:1393` calls `sun_event(run.site, night, kind='dark')` on the update path, and `:1396`
builds a *fresh* dark-window line rather than a preserved one. The exception is argued
carefully at the call site (`:1375-1392`), but a reader who checks the function's contract
first — the normal order — is told the exception does not exist, and the invariant is
load-bearing enough that `TestNoSunEventRecompute` exists to defend it.

`_span_needs_remint()` also still calls `sun_event()` on the update path for any unrecorded
night (`:730`), which the same sentence has denied since iteration 7 (see WR-08 below); this
round's change makes the sentence wrong in two independent ways.

**Fix:** state both exceptions in the docstring, pointing at the two call sites:
"…except (a) `_span_needs_remint()`'s step-4 resolution for a night whose provenance is
unrecorded or whose position fingerprint moved, and (b) the plain-update path's dark-window
refresh for a fully-set run whose recorded site component moved — see each call site for its
bound." Amend the Field-authority paragraph to say the update path's `description` carries
the preserved dark-window line *except* on that one transition.

---

### WR-04: the step-4 staleness warning calls a current-format recorded token "unrecorded-provenance" on the new fingerprint path — the false operator message NF-16 and WR-06 were both raised for, now in the notebook's committed output

**File:** `solsys_code/allocation_projector.py:739-750`

**Issue:** 35-24 routes a *trusted, current-format* token whose position fingerprint differs
into the same step-4 branch the unrecorded case uses (`:723-730`, deliberate and correct).
It also reuses the branch's log line verbatim:

```
Allocation unrecorded-provenance night pk=%s run pk=%s night=%s: stored boundary start=%s
end=%s disagrees beyond tolerance with the resolved sun event sunset=%s sunrise=%s.
```

For the new entry path every word of "unrecorded-provenance" is false: the token is present,
current-version, and correct-part-count — the run's *site position* moved. An operator
grepping for this message is sent to the runbook's `retired` reason (5) ("a night minted
before this release … whose recorded mint inputs are absent or were recorded in a
pre-release format"), which is not what happened and offers no relevant remedy; the correct
section is the new "Correcting a site's own definition" paragraph.

This is not hypothetical wording: the notebook's own executed output at
`docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` cell 43 shows the message
emitted for exactly the in-place-correction demo, directly above prose explaining that this
is a *position* correction.

**Fix:** pass the reason into the log line, e.g. compute
`reason = 'position-fingerprint' if token_trusted else 'unrecorded-provenance'` at the two
entry points and emit `'Allocation boundary re-resolution (%s) for night pk=%s ...'`. Keep
the existing token in the unrecorded case so
`TestDeclinedNightResolutionCostIsBounded`'s `'unrecorded-provenance night'` substring
assertions still mean what they say, and add a companion assertion for the new token.

---

### WR-05: two runbook claims are broader than the code — "re-mints every allocation night at that site" and "a cross-timezone correction makes the whole run fail to reconcile"

**File:** `docs/runbooks/telescope_runs_calendar.rst:1128-1140` and `:1107-1119`

**Issue:** (a) "Editing an ``Observatory`` row's latitude, longitude, altitude or timezone
… **now re-mints every allocation night already projected at that site**, on the next
sweep" has three unstated exceptions, two of them normal: a fully-set sub-night run never
re-mints from a position change at all (WR-02 above — and the runbook's own reason-(2)
paragraph twelve lines earlier says so); a night whose re-mint is declined reports
`remint_declined` instead (the section two pages down); and a correction inside the
one-minute tolerance is excluded (this paragraph does say so).

(b) "a correction that moves such a run to a site in a different timezone **makes the whole
run fail to reconcile** instead — reported ``Run pk=N: reconcile failed (...) -- skipping``"
is stated as a general rule, but it is a property of the *particular* pair of timezones and
times, not of cross-timezone moves. The test that pins it says so in its own docstring
("For **this fixture** the resolved span INVERTS"). Hand-tracing `_time_of_day_to_datetime()`
for the same 23:00/05:00 pair moved `America/Santiago` (−4) → `Africa/Johannesburg` (+2)
gives span `[N 16:00Z, N+1 04:00Z]`, start `N 23:00Z`, end `N+1 05:00Z` — not inverted, so
that run re-mints silently to boundaries that are now 6 hours out of place relative to the
new site's night. An operator told "this always fails loudly" will not look for the case
where it succeeds quietly.

**Fix:** (a) add the two missing exceptions to the paragraph, or scope its first sentence to
"every allocation night whose boundaries are derived from the sun event (i.e. a run whose
sub-night window is empty or half-set)". (b) rewrite as "…may fail to reconcile, with an
inverted-span `ValueError`, when the two sites' observing nights sit in different UTC bands;
when they do not, the night is re-minted against boundaries that are no longer meaningful at
the new site. In both cases the remedy is the same: correct
`night_start_utc`/`night_end_utc` together with `site`." Consider making that a hard guard
instead — refusing a site change that crosses `_night_span_utc()` bands while both sub-night
fields are set — which would turn a silent wrong answer into a loud one.

---

### WR-06: the paired demo notebook has no cell for the fully-set dark-window refresh — the round's one new astropy call and its one documented exception to D-13

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb`

**Issue:** CLAUDE.md's paired-docs rule maps `allocation_projector.py` →
`reconcile_campaign_runs_demo.ipynb` and requires cells "exercising the new behavior with
real executed output". The notebook gained four new sections this round (CR-04/CR-05, the
`remint_declined` summary line, and the escalated in-place correction) — but
T-35-24-02/WR-05's dark-window refresh, which is the round's only *new write* on the
plain-update path and the only sanctioned exception to D-13's "no `sun_event()` on an
existing night", has no cell. Searching the notebook JSON: `Dark window` appears once (in a
stored description string, incidentally), `dark-window` zero times,
`_site_provenance_differs` zero times.

This matters more than a coverage gap, because WR-02 above says the feature is unreachable
for pre-existing nights: a notebook cell that had to construct a freshly-minted `v3` night
to make the demo work would have surfaced that limitation during authoring.

**Fix:** add a section after cell 42 that mints a fully-set (`night_start_utc`/
`night_end_utc`) single-night run, prints its stored dark-window line and its token, applies
a same-timezone in-place position correction, reconciles, and shows the refreshed line with
unchanged `pk`/`start_time`/`end_time` and `updated: 1, retired: 0` — the executed
counterpart of `test_same_timezone_correction_on_a_set_window_run_refreshes_the_dark_window_line`.
Regenerate with `jupyter nbconvert --to notebook --execute --inplace`.

---

### WR-07: `load_telescope_runs`'s night summary now silently under-reports — a declined retirement no longer increments `retired`, and neither decline counter is aggregated there

**File:** `solsys_code/management/commands/load_telescope_runs.py:300-307` and `:359-366`
(consumers); `solsys_code/allocation_projector.py:1207`

**Issue:** The ingest command aggregates `created`/`updated`/`unchanged`/`retired`/
`rekeyed`/`blocked`/`skipped_nights` from each `reconcile_run()` result and prints them as
its `nights -- ...` summary. It has never aggregated `detach_declined`, which was tolerable
while that counter only described attribution releases. This round changed the arithmetic
underneath it: `:1207` withholds `retired` when a retirement is declined, and the new
`remint_declined` is not read either. So after this release a `load_telescope_runs` run over
a file containing a confirmed night reports one fewer retirement than actually happened,
with no line anywhere in its output explaining the difference — the exact "silence and
'nothing to release' are indistinguishable" failure `detach_declined`'s own docstring says
the counter exists to prevent.

**Fix:** aggregate and print both counters in the `nights -- ...` line, and add the matching
per-line stderr note the reconcile command already emits. If that is out of scope for this
round, record it in the runbook's `load_telescope_runs` section so an operator comparing the
two commands' summaries is not left to guess.

---

### WR-08 (carried forward from iteration 9's WR-01, partially closed): `--dry-run` still pays one `sun_event()` call per unrecorded night on every invocation; the docstring is now honest, the inline comment is not

**File:** `solsys_code/allocation_projector.py:1282-1286` (the comment), `:730` + `:752-753`
(the behaviour)

**Issue:** 35-24 closed the docstring half: `_span_needs_remint()`'s cost bound at
`:647-668` now names both escapes (a declined night, and a dry run) explicitly and
truthfully, which is what WR-01 asked for. The inline comment in the re-mint branch was not
touched and still reads:

> "Both halves are skipped under dry_run (no `sun_event()` call either), so a dry-run
> preview and a real run agree on the same pair of counters."

`_span_needs_remint(..., dry_run=True)` reaches `:730` and calls `sun_event()` for every
unrecorded night before that branch is entered at all, and `:752` deliberately skips the
recording, so the next preview pays it again. The `v3` bump widens this further than the
`v2` bump did: after the upgrade every null/null and half-null night in the database is
unrecorded, so the first post-upgrade `--dry-run` pays one astropy call per such night, and
so does every repeat preview.

**Fix:** correct `:1284-1286` to say the re-mint *write* is skipped under `dry_run` while the
unrecorded-provenance *resolution* above it still runs, and pin the per-dry-run call count
with a test (`TestDeclinedNightResolutionCostIsBounded` already has the shape; it needs a
dry-run twin).

---

### WR-09 (carried forward from iteration 9's WR-02, explicitly deferred): a read-only `--dry-run` preview can still raise `sun_event()`'s `ValueError`

**File:** `solsys_code/allocation_projector.py:730`; conflicting rationale at `:1330-1350`

**Issue:** Unchanged, and knowingly so — 35-23's plan ledger records WR-02 as deferred, and
`:1405-1416`'s new comment explicitly preserves the surface ("keeping WR-02's deferred
preview-raises-ValueError surface exactly as wide as it was"). The create branch still
carries the comment explaining why `_mint_fields()` must not be called under `dry_run`
("…could raise `sun_event()`'s own `ValueError` … on what the module's own docstring
documents as a read-only preview") while `:730` does exactly that on the resolution branch.
The module contradicts itself in two comments 600 lines apart.

Iteration 9's correction to the impact claim still stands: the blast radius is one run per
bad site, not the sweep (`reconcile_campaign_runs.py`'s per-run `try/except` catches it),
but `campaign_views._resolve_site()` still swallows it into a "use Resolve to retry" message
that can never succeed.

**Fix:** as before — either state the trade-off at `:730` and amend the `:1330-1350`
comment, or degrade a preview to "cannot decide, report `unchanged`" on `ValueError` when
`dry_run` is set.

---

### WR-10 (carried forward from iteration 9's WR-03, unaddressed): the `rekeyed` paragraph still promises a stability the next sweep can withdraw

**File:** `docs/runbooks/telescope_runs_calendar.rst:1140-1144`

**Issue:** Byte-identical to iteration 9. The paragraph still tells operators a re-keyed
night keeps "same primary key, same start/end time, just re-keyed", while the re-key path
(`allocation_projector.py:1236-1248`) deliberately records no provenance — so on the next
sweep that night enters `_span_needs_remint()`'s step-4 resolution and is re-minted (new
primary key, new boundaries) whenever its carried-over boundary sits more than a minute from
the computed sun event. The promise is true of the re-key itself and false one sweep later,
which is exactly when an operator checks.

**Fix:** append the caveat iteration 9 proposed: *"The 'same primary key, same start/end
time' guarantee covers the re-key itself. A re-keyed night carries no mint provenance, so
the next sweep resolves it once under ``retired`` reason (5) and may re-mint it at the
computed sun event — see the deploy note below."*

---

### WR-11 (carried forward from iteration 9's WR-04, parts (a) and (c) unaddressed): the tolerance constant's premise is still unverifiable from the tree, and the odd-one-out fixture convention is still odd

**File:** `solsys_code/allocation_projector.py:68-77`;
`solsys_code/tests/test_cutover_classical_allocations.py` (`_make_three_night_group()`)

**Issue:** Part (b) (the post-upgrade deploy note) was closed in 35-22 and has been extended
for the `v3` bump this round (`telescope_runs_calendar.rst:1245-1260`). Parts (a) and (c) are
untouched: `_UNRECORDED_PROVENANCE_TOLERANCE`'s comment is byte-identical, still names no
retired writer that could have produced a stale operator boundary and gives no audit query
an operator could run to check the premise before the sweep
(`grep -rn 'sun_event' solsys_code/management/commands/` still returns nothing), and
`_make_three_night_group()`'s round-hour convention is still the odd one out relative to the
one test rebuilt around real `sun_event()`-derived boundaries.

**Fix:** as before — (a) name the retired writer and its release in the constant's comment,
or give the operator a one-off audit query; (c) either propagate realistic boundaries into
`_make_three_night_group()` or comment there why the round-hour convention remains
acceptable for the tests that never run a sweep.

## Info

### IN-01: the notebook's counter section is headed "Six counters" and lists seven

**File:** `docs/notebooks/pre_executed/reconcile_campaign_runs_demo.ipynb` cell 31

The heading and its lead-in ("report six numbers alongside
`created`/`updated`/`unchanged`/`blocked`") were not updated when `remint_declined` was
added as a bullet. The bullets now enumerate `skipped_nights`, `detached`,
`detach_declined`, `remint_declined`, `retired`, `rekeyed`, `legacy_deleted` — seven.

**Fix:** "Seven counters on ReconcileResult", and "report seven numbers".

---

### IN-02: `_site_position_fingerprint()`'s `repr()` stability claim does not survive a Django `FloatField` assigned an `int`

**File:** `solsys_code/allocation_projector.py:122-127`, `:146`

The docstring claims `repr()` is "stable across process restarts and across a save/reload
cycle". Django does not coerce on assignment, so `site.altitude = 2347; site.save()` leaves
the in-memory value as `int` (`repr` → `'2347'`) while a reload gives `float`
(`repr` → `'2347.0'`) — two different fingerprints for one stored value. The notebook's own
demo cell assigns integer `altitude=2347`. Production paths always load the site fresh from
the database (the sweep, the receivers, the admin round-trip), so this cannot oscillate
today; a caller that mutates a site in memory and reconciles in the same process would pay
one spurious `sun_event()` resolution per sweep.

**Fix:** normalise before hashing — `float(x) if x is not None else None` — or narrow the
docstring claim to "stable for any value loaded from the database".

---

### IN-03: a within-tolerance in-place position correction leaves a null/half-null night's dark-window line stale, and now hides it behind a refreshed token

**File:** `solsys_code/allocation_projector.py:734-753`

When step 4 is entered via the fingerprint-differs path and the resolved sun event lands
within `_UNRECORDED_PROVENANCE_TOLERANCE`, `:753` records the *current* token and returns
False. The night reports `unchanged`, its boundaries are (correctly) left alone — and its
stored `Dark window (-15 deg, UTC): ...` line keeps the pre-correction site's numbers, with
no path left to refresh it (the update-path refresh at `:1369` requires both sub-night
fields set, which this night by construction does not have). The runbook's new paragraph
presents the within-tolerance case as purely beneficial ("does not churn the calendar")
without mentioning the residue.

**Fix:** one sentence in the runbook's within-tolerance sentence, or refresh the dark-window
line on this transition too (it costs one `sun_event(kind='dark')` call on exactly the
transition that already paid for a `kind='sun'` call).

---

### IN-04: both retirement guards leave an unreachable third outcome silently uncounted, in a module whose own rule is "no third outcome"

**File:** `solsys_code/allocation_projector.py:1152-1166` and `:1182-1197`

Each guard is written as `if <deletable>: ... elif confirmed_declined: ...` with no `else`.
If a candidate were ever neither, the event would be silently left alone with no counter and
no log — D-16/NF-01's forbidden third outcome, which
`_clearable_declined_and_unattributed()`'s docstring is entirely about. It is provably
unreachable today (`_may_write()` has already excluded the only shape the partition does not
cover, a companion row attributed to a different run), but that proof lives in a different
module and is exactly the kind of cross-module invariant a later change breaks quietly.

**Fix:** add an `else:` that logs at `error` and counts under `blocked`, or assert the
invariant with a comment naming `_may_write()` as the reason the third arm cannot fire.

---

### IN-05 (carried forward from iteration 9's IN-01, unchanged): `null=True` + `blank=True` on a `CharField` still leaves `''` as a third state

**File:** `solsys_code/models.py:165-167`,
`solsys_code/migrations/0021_alter_calendareventmeta_is_verified_and_more.py:20`

The field is now `CharField(max_length=128, null=True, blank=True)` — widened, but the
three-state ambiguity (`NULL`, `''`, token) is unchanged, for a field no form is allowed to
bind. The version-AND-part-count test keeps defusing it (`''.split('|')` has one part, so it
reads as unrecorded), and `test_empty_string_token_reads_as_unrecorded_and_resolves_once`
still pins that.

**Fix:** drop `blank=True` in a follow-up schema-only migration, or note in the field comment
that `''` is treated identically to `NULL` by design.

---

### IN-06 (carried forward from iteration 9's IN-03, unaddressed): the class docstring still says "no row at all means verified by documented default" without noting that the projector materialises rows purely to record provenance

**File:** `solsys_code/models.py:16-18`; `solsys_code/allocation_projector.py:523-538`

`_record_sub_night_provenance()` still uses `update_or_create(event=event, ...)`, and step 4
now calls it for both the unrecorded and the fingerprint-differs paths — so the "no row at
all" shape is converted to "row with `run IS NULL`" on the first post-upgrade sweep, at
scale. Verified harmless (`_may_write()` and `_clearable_declined_and_unattributed()` treat
the two shapes identically). The class docstring was rewritten again this round and still
does not mention it.

**Fix:** one sentence: "the allocation projector may materialise a row solely to record
`minted_sub_night_window`, so the absence of a row is not evidence that no projection has
touched the event."

---

### IN-07: `_remint_decline_reason()`'s docstring cites a line number that moved

**File:** `solsys_code/allocation_projector.py:817`

Rule 1's text points at "the per-night loop's own `_may_write(existing, run)` gate
(``:765``)". That gate is now at `:1112`. The reasoning is sound and the reference is the
only thing wrong with it, but a wrong line number in the one paragraph explaining why a
partition is two-way rather than three-way is worth correcting while the surrounding text is
being edited anyway.

**Fix:** cite the symbol rather than the line (`the per-night loop's own
``_may_write(existing, run)`` gate`), so it cannot go stale again.

---

_Reviewed: 2026-09-16T19:09:02Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
