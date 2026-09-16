---
phase: 35-allocation-layer-classical-cutover
reviewed: 2026-09-15T21:40:00Z
depth: deep
iteration: 8
diff_base: b9383b58c0ec2b80be991e18b90ca879724e641c
files_reviewed: 6
files_reviewed_list:
  - solsys_code/admin.py
  - solsys_code/allocation_projector.py
  - solsys_code/migrations/0019_calendareventmeta_minted_sub_night_window.py
  - solsys_code/models.py
  - solsys_code/tests/test_allocation_projector.py
  - solsys_code/tests/test_cutover_classical_allocations.py
findings:
  critical: 3
  warning: 4
  info: 4
  total: 11
status: issues_found
---

# Phase 35: Code Review Report (iteration 8 — fourth gap-closure round)

**Reviewed:** 2026-09-15T21:40:00Z
**Depth:** deep
**Files Reviewed:** 6
**Status:** issues_found

## Summary

**Iteration 7's CR-01 is genuinely closed.** Every claim in `35-19-SUMMARY.md` was
re-verified against the real source and against a real Django test database, and all of them
hold:

| Claim checked | Verdict |
|---|---|
| All three probe shapes (A both-cleared, B half-null's remaining field cleared, C one-of-two cleared) re-mint | **TRUE.** Reproduced independently, plus the **mirror of shape C** the tests do not cover (clearing `night_end_utc` instead of `night_start_utc`): `retired=1, created=1, unchanged=0`, new pk, token refreshed to `23:00:00\|none`, `start_time` preserved at the still-set boundary, `end_time` = the real `sun_event()` sunrise. |
| `TestNoSunEventRecompute` exists unedited and still passes | **TRUE.** `git diff b9383b5..HEAD -- solsys_code/tests/test_allocation_projector.py \| grep -c '^-[^-]'` → **0 deletions**, purely additive. All three of its tests green. |
| `minted_sub_night_window` is admin-readonly on both surfaces | **TRUE.** `CalendarEventMetaAdmin.readonly_fields` (`admin.py:335-341`) and `CalendarEventMetaInline.readonly_fields` (`admin.py:119-125`). No other `ModelForm` binds it. |
| The unrecorded-provenance backfill is bounded to once-per-night-ever | **TRUE in real mode** (second reconcile makes 0 `sun_event()` calls), **FALSE under `--dry-run`** — see WR-01. |
| The token is refreshed correctly on an unrelated re-mint, so a later genuine null-clear is still detected | **TRUE.** `23:00:00\|05:00:00` → window change → `22:00:00\|05:00:00` → clear → detected → `none\|05:00:00`, `start_time` = the real sunset. |
| The `test_cutover_classical_allocations.py` Rule-1 deviation is legitimate | **Legitimate in mechanism, unverifiable in premise** — see WR-04. |
| 212/212 across five modules, both ruff hooks, `makemigrations --check` clean | **TRUE.** Re-ran all three: 212 tests OK (113 s); `ruff` and `ruff-format` Passed; "No changes detected". |

**But this round opened three new defects of its own, and the review scope has been too
narrow for seven iterations to see one of them.** The provenance design correctly records
*which sub-night pair* a night was minted from — and nothing else. The boundaries are a
function of **(sub-night pair, site, night)**, so a site correction now reads as "nothing
changed" forever (CR-02). Separately, the re-mint delete path — which this round *widens* to
cover three new transition shapes plus every legacy night whose stored boundary drifts more
than a minute — is the one delete path in this module that never received the
human-confirmation guard CR-03/CR-04 added everywhere else (CR-01), and it deletes before it
can fail (CR-03).

### Verification method

- Re-read every file in scope in full, plus the call graph they depend on:
  `campaign_reconciler._may_write()` / `_link_event_to_run()` /
  `_clearable_declined_and_unattributed()`, `campaign_utils.UNLINK_CLEARED_FIELDS` /
  `unlink_event_from_run()`, `campaign_views._resolve_site()` and the approve branch,
  `management/commands/repair_stale_campaign_run_sites.py`,
  `docs/runbooks/telescope_runs_calendar.rst`.
- Wrote a throwaway probe module against `AllocationProjectorTestBase` (real migrated test
  DB, real `Observatory` rows, real `sun_event()` calls), ran nine probes, then **deleted the
  probe file** — `git status --short` confirms no source file was added or modified by this
  review.
- Re-ran the full five-module regression surface (212 tests, OK), both ruff hooks (Passed),
  and `makemigrations --check --dry-run` (no drift).
- Deferred items WR-01/WR-02/WR-03 from iteration 7 are recorded as `user_deferred:` in
  `35-VERIFICATION.md` and are **deliberately not re-raised** here.

## Structural Findings (fallow)

No structural pre-pass was supplied with this review request.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: the re-mint delete path is the only delete path in this module with no human-confirmation guard — a re-mint silently destroys `confirmed_by`/`confirmed_at` and the `observation_record` link, and this round widens what reaches it

**File:** `solsys_code/allocation_projector.py:863-886` (specifically `:879`)

**Issue:** Every other delete/detach path in this module applies the UAT-2026-09-09 "Option B
— human outranks machine" rule via `_clearable_declined_and_unattributed()`:

- the retired branch (`:802-819`, added as CR-03 in an earlier iteration),
- the final convergence step (`:952-969`, added as CR-04),
- the legacy-takeover branch (`:841-848`, via `_may_write()`).

The re-mint branch does not. `_span_needs_remint()` returns True and line 879 calls
`existing.delete()` unconditionally. `CalendarEventMeta.event` is
`OneToOneField(primary_key=True, on_delete=CASCADE)`, so the companion row goes with it —
including `confirmed_by`, `confirmed_at`, `observation_record`, `observation_group` and
`is_verified`. `_may_write()` does **not** cover this: a row confirmed to *this* run passes
`_may_write()` (`meta.run_id == run.pk`), which is exactly the shape CR-03/CR-04 exist to
protect.

Reproduced against a real test DB (probe 8):

```
PROBE8 meta before: {'run_id': 1, 'confirmed_by_id': 2}
PROBE8 result= ReconcileResult(created=1, ..., retired=1, ...)
PROBE8 old event still exists? False
PROBE8 old meta still exists? False
PROBE8 new meta confirmed_by= None confirmed_at= None
```

and probe 9 (same run, `observation_record` + `is_verified=False` on the companion row):

```
PROBE9 new meta observation_record= None is_verified= True
```

Both were staff facts. Both were destroyed by an automated sweep, with no warning log, no
`detach_declined` counter, and `retired=1 / created=1` reported as ordinary work.

This is pre-existing code, but **this round materially widens its reach**, which is why it
now blocks: before 35-19 only a *changed SET* sub-night field could re-mint; after 35-19 the
three cleared-to-null shapes re-mint too, **and** the new unrecorded-provenance branch
(`:505-531`) re-mints any legacy night whose stored boundary sits more than a minute from the
computed sun event — on the first sweep after deploy that is every existing `ALLOC:` night in
the database, since migration 0019 leaves them all `NULL` (see WR-04).

**Fix:** apply the same guard the retired branch already applies, before deleting:

```python
if existing is not None and _span_needs_remint(run, night, existing, dry_run=dry_run):
    deletable_ids, confirmed_declined = _clearable_declined_and_unattributed(
        run, CalendarEvent.objects.filter(pk=existing.pk)
    )
    if not deletable_ids:
        if confirmed_declined:
            logger.warning(
                'Allocation re-mint declined: night pk=%s is human-confirmed to run '
                'pk=%s -- an automated re-mint never destroys it.',
                existing.pk,
                run.pk,
            )
            totals['detach_declined'] += 1
        active_urls.add(url)
        continue
    totals['retired'] += 1
    totals['created'] += 1
    ...
```

(If a confirmed night genuinely must be re-minted, the alternative is to preserve the
companion row across the delete/create rather than to skip — but silently discarding a human
stamp is not an option either way.)

---

### CR-02: the provenance token omits the site, so a site correction on an already-projected run is now permanently invisible — the same silent-stale-calendar defect class as iteration 7's CR-01

**File:** `solsys_code/allocation_projector.py:382-402` (`_sub_night_provenance_token`),
`:496-503` (the decision that consumes it)

**Issue:** An allocation night's boundaries are a function of three inputs —
`(run.night_start_utc, run.night_end_utc)`, `run.site`, and `night` —
because `_mint_fields()` computes them via `sun_event(run.site, night, kind='sun')`
(`:545`). The token records only the first. So once provenance is recorded, changing
`run.site` produces no comparison that can detect it: both SET-field comparisons are skipped
(null/null), step 2 is skipped, and step 3 finds `'none|none' == 'none|none'` → `False`,
astropy-free, forever.

Reproduced against a real test DB (probe 1), moving a run from La Silla (`America/Santiago`)
to Siding Spring (`Australia/Sydney`) after one reconcile:

```
PROBE1 token= none|none
PROBE1 chile start/end= 2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE1 sun_event calls after site change = 0
PROBE1 result = ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0, ...)
PROBE1 after start/end= 2026-07-09 22:06:35+00:00 2026-07-10 11:29:46+00:00
PROBE1 TRUE Siding Spring sunset/sunrise= 2026-07-09 07:20:39+00:00 2026-07-09 20:57:12+00:00
PROBE1 same pk? True
```

A ~15-hour error, reported as `unchanged`, permanently. Probe 1b proves the code *can* detect
it — the same site change on a night whose provenance is **not** recorded goes through the
new `:505-531` branch and correctly reports `retired=1, created=1`. It is the recorded token
that masks it.

This is reachable by two ordinary staff actions, not a contrived one:

1. `CampaignRunAdmin` (`admin.py:149-176`) does **not** list `site` in `readonly_fields` —
   only `approval_status` and (for web rows) `source`. A staff member correcting a
   mis-resolved site in the admin lands here directly.
2. `campaign_views._resolve_site()` (`campaign_views.py:651-694`) rewrites `run.site` on an
   already-**APPROVED** run whenever the current site is a tier-3 placeholder, then calls
   `reconcile_run(run)` at `:733`. A CSV-imported run approved with a placeholder site has
   already had its nights projected against the placeholder's coordinates by the approve
   branch's own `reconcile_run(run)` (`campaign_views.py:569`); the later real resolution now
   leaves those placeholder-derived boundaries in place and reports `unchanged`.

Iteration 7's CR-01 was raised for exactly this signature — "admin-reachable, silent,
permanent stale data" — and this round shipped a fix that re-creates it one input over.

**Fix:** make the token carry every input the mint depended on, and treat an old-format token
as unrecorded so existing rows re-resolve once through the `:505-531` branch:

```python
_PROVENANCE_TOKEN_VERSION = 'v2'

def _sub_night_provenance_token(run: CampaignRun) -> str:
    start_token = run.night_start_utc.isoformat() if run.night_start_utc is not None else 'none'
    end_token = run.night_end_utc.isoformat() if run.night_end_utc is not None else 'none'
    return f'{_PROVENANCE_TOKEN_VERSION}|{run.site_id}|{start_token}|{end_token}'
```

and in `_span_needs_remint()`:

```python
if recorded_token is not None and recorded_token.startswith(f'{_PROVENANCE_TOKEN_VERSION}|'):
    return recorded_token != _sub_night_provenance_token(run)
```

`models.CalendarEventMeta.minted_sub_night_window` is `max_length=32`, which the widened
token overflows — bump it (a schema-only `AlterField`, same shape as migration 0019) and add
a test pinning the worst-case token length. Update the field's own docstring
(`models.py:38-47`) and `_sub_night_provenance_token`'s, both of which currently say the
column records the sub-night pair alone.

---

### CR-03: the real-run re-mint deletes the night *before* `_mint_fields()` can raise, so an inverted-window edit destroys the event and then fails — no transaction anywhere on this path

**File:** `solsys_code/allocation_projector.py:879-880`

**Issue:**

```python
existing.delete()
event, _action = insert_or_create_calendar_event({'url': url}, fields=_mint_fields(run, night))
```

`_mint_fields()` calls `night_bounds()` (`:551`), which calls `_raise_if_inverted()` and
raises `ValueError` for an inverted span (the CR-06 guard). The delete has already committed
by then. There is **no** `transaction.atomic` in `allocation_projector.py`,
`campaign_reconciler.py` or `management/commands/reconcile_campaign_runs.py`, and
`ATOMIC_REQUESTS` is not set in `src/fomo/settings.py` — so this is a real, committed delete
in production, not a rolled-back one.

Reproduced against a real test DB (probe 6), editing an already-minted night's sub-night pair
to an inverted pair:

```
PROBE6 exists before = True
PROBE6 RAISED ValueError: Computed an inverted allocation-night span for run pk=1
  night=2026-07-09: start=2026-07-09T23:30:00+00:00 >= end=2026-07-09T23:00:00+00:00.
PROBE6 exists after = False
```

The night — and, per CR-01, its companion row's entire audit history — is gone, and the
caller gets an exception instead of the event. `_raise_if_set_window_inverted()`'s dry-run
guard does not protect this: it only fires if the operator happens to run `--dry-run` first,
and for a **half-null** inverted pair the dry run is silent by design (iteration 7's WR-01,
user-deferred), so the real run destroys the event with no prior warning at all. Both signal
receivers (`:1034`, `:1115`) swallow the exception and log only `type(exc).__name__`, so a
`CampaignRunObservation` save can reach this and leave no diagnosable trace.

**Fix:** compute before destroying, and wrap the pair so a later failure cannot leave a hole:

```python
from django.db import transaction
...
fields = _mint_fields(run, night)       # may raise -- nothing has been deleted yet
with transaction.atomic():
    existing.delete()
    event, _action = insert_or_create_calendar_event({'url': url}, fields=fields)
    _link_event_to_run(event, run)
    _record_sub_night_provenance(event, _sub_night_provenance_token(run))
```

Add a test asserting the event survives when `_mint_fields()` raises.

## Warnings

### WR-01: `--dry-run` now pays one `sun_event()` call per unrecorded night on *every* invocation, and the branch's own comment says the opposite

**File:** `solsys_code/allocation_projector.py:867` (the comment), `:507` + `:529-530` (the
behaviour), `:472-473` (the docstring's cost bound)

**Issue:** The re-mint branch's inline comment reads:

> "Both halves are skipped under dry_run (no `sun_event()` call either), so a dry-run preview
> and a real run agree on the same pair of counters."

That is now false. `_span_needs_remint(..., dry_run=True)` reaches `:507` and calls
`sun_event()` for every unrecorded night, and `:529` deliberately skips the recording — so
nothing is ever learned and the next dry run pays the same cost again. The docstring's stated
cost bound ("at most one `sun_event(kind='sun')` call per unrecorded night, **once ever**",
`:472-473`) holds only in real mode.

Reproduced (probe 2), five-night window, provenance cleared:

```
PROBE2 dry-run #1 sun_event calls = 5
PROBE2 dry-run #2 sun_event calls = 5
PROBE2 rows still unrecorded after 2 dry runs = 5
```

The skip-recording-under-dry-run choice is defensible (a preview must not write). The stale
comment and the unqualified cost bound are not — they are the exact kind of
docstring-diverged-from-code that this phase's iteration-7 CR-01 was found underneath.

**Fix:** correct the comment at `:865-868` to say the re-mint *write* is skipped under
`dry_run` while the unrecorded-provenance *resolution* still runs, and qualify the `:472-473`
cost bound with "in real mode; a `--dry-run` preview repeats the call because it may not
record what it proves". Optionally add a test pinning the per-dry-run call count so the claim
cannot drift again.

---

### WR-02: a read-only `--dry-run` preview can now raise `sun_event()`'s `ValueError`, partially reintroducing the failure mode iteration 6's WR-03 closed

**File:** `solsys_code/allocation_projector.py:507`; conflicting rationale at `:888-894`

**Issue:** The create branch carries a comment, written for WR-03, explaining why
`_mint_fields()` must not be called under `dry_run`:

> "...and could raise `sun_event()`'s own `ValueError` (e.g. a blank `Observatory.timezone`)
> on what the module's own docstring documents as a read-only preview."

The new `:507` call reintroduces exactly that on a different branch. Reproduced (probe 3):
one already-minted night, provenance cleared, `Observatory.timezone` blanked:

```
PROBE3 dry run RAISED: ValueError ZoneInfo keys must be normalized relative paths, got:
```

Before this round the same preview completed cleanly. `reconcile_campaign_runs` has no
per-run `try/except` around this, so one bad `Observatory.timezone` now aborts the whole
preview sweep for every run after it.

**Fix:** either state the trade-off explicitly at `:507` (and amend the `:888-894` comment so
the module stops contradicting itself), or wrap the resolution so a preview degrades to
"cannot decide, report `unchanged`" rather than raising:

```python
try:
    sunset, sunrise = sun_event(run.site, night, kind='sun')
except ValueError:
    if dry_run:
        logger.warning(
            'Allocation dry-run could not resolve unrecorded provenance for run pk=%s '
            'night=%s (sun_event failed); previewing as unchanged.', run.pk, night,
        )
        return False
    raise
```

---

### WR-03: the operator runbook's `retired` enumeration is now wrong — this round adds a fifth reason and `docs/` was deliberately left untouched

**File:** `docs/runbooks/telescope_runs_calendar.rst:1078-1091`

**Issue:** The runbook tells operators, in the section that interprets the sweep's printed
counters:

> "``retired`` counts an allocation night removed from the calendar for any of **four**
> reasons (35-REVIEW.md NF-07): (1) ... (2) a sub-night window field ... changed since the
> night was last minted ... (3) the night no longer falls inside the run's window ... or (4)
> ... a leftover night whose companion row was deleted outright or had its ``run`` cleared"

This round adds a fifth, which none of the four cover: *a night whose mint provenance was
never recorded and whose stored boundary disagrees with the freshly computed sun event by
more than one minute*. An operator seeing `retired: 37` on the first post-deploy sweep will
look for a window shrink or a sub-night edit that never happened. The `rekeyed` paragraph
(`:1094-1098`) is also now misleading — it promises "same primary key, same start/end time",
which holds for the re-key itself but not for the *next* sweep, when the just-re-keyed night
(provenance deliberately not recorded on that path) may be re-minted under the new rule.

`35-19-SUMMARY.md` records `git status --short -- docs/ solsys_code/management/` as clean as
an *achievement*. Under CLAUDE.md's paired-docs rule this is the opposite: the rule is
directory-scoped ("any page under `docs/runbooks/` whose documented behavior the change
affects"), the page is wired into the toctree at `docs/index.rst:24`, and the rule explicitly
names the code-reviewer as a subagent it binds.

**Fix:** change "four reasons" to five and add the new one, e.g.: *"(5) a night minted before
this release (or carried across by the re-key path) whose stored boundary disagrees with the
computed sun event by more than one minute — a one-time audit that happens at most once per
night; the same night reports `unchanged` on every sweep after it."* Add a matching note to
the `rekeyed` paragraph, and a short deploy note (see WR-04).

---

### WR-04: the one-minute tolerance silently rewrites existing calendar boundaries on the first sweep after deploy, on a premise that cannot be verified from the current tree

**File:** `solsys_code/allocation_projector.py:66-75` (`_UNRECORDED_PROVENANCE_TOLERANCE`),
`:505-531`; `solsys_code/tests/test_cutover_classical_allocations.py:434-454`

**Issue:** Migration 0019 is a bare `AddField(null=True)`, so **every** existing `ALLOC:`
event in production has unrecorded provenance. The first sweep after deploy therefore audits
each one against a live `sun_event()` and re-mints (delete + create, per CR-01 without any
human-confirmation guard, per CR-03 with the delete preceding the failure point) any whose
boundary is more than 60 s out.

The plan's own Rule-1 deviation is direct evidence that real fixtures fall outside that
tolerance: `TestCutoverSequenceContract.test_cutover_then_sweep_reaches_the_pinned_end_state`
had to be rebuilt because `_make_three_night_group()`'s 23:00/09:00 boundaries sit ~53
minutes from the true La Silla sun event and were re-minted by the new branch. The
justification given — *"The real pre-cutover `load_telescope_runs` writer always computed
`sun_event()`-derived boundaries, so the round-hour convention never represented a genuine
legacy night"* — is a claim about a **writer that no longer exists in this repository**:
`grep -rn 'sun_event' solsys_code/management/commands/` returns nothing, and the current
`load_telescope_runs.py` delegates boundaries to `reconcile_run()` entirely. Nobody can check
that premise from the tree, and ~30 other tests in the same file still use the round-hour
convention the deviation calls unrealistic — the shared helper was left as the odd one out
rather than the deviation being propagated or the convention justified.

The deviation itself is mechanically fine (scoped to the one test that runs a sweep, uses
real `sun_event()`-derived boundaries, does not weaken an assertion). The problem is that it
is the *only* place this behaviour change is recorded, and it is recorded as a test-fixture
detail rather than as a production-data consequence.

**Fix:** (a) state the premise where it can be challenged — a module-level comment naming the
retired writer and the release it belonged to, or a one-off audit query an operator can run
before the sweep; (b) add the deploy note to the runbook alongside WR-03's fix ("run
`reconcile_campaign_runs --dry-run` first after upgrading; a non-zero `would_retire` on nights
you did not edit is the one-time provenance audit"); (c) either propagate the realistic
boundaries into `_make_three_night_group()` or add a comment there explaining why the
round-hour convention is still acceptable for the ~30 tests that never run a sweep.

## Info

### IN-01: `null=True` on a `CharField` makes `''` a third, undecidable state

**File:** `solsys_code/models.py:117-119`

`minted_sub_night_window` is `CharField(max_length=32, null=True, blank=True)`. The design
depends on exactly two states (`NULL` = not recorded, a token = recorded), but Django's own
convention is that a blank `CharField` stores `''`, and `blank=True` means any future
`ModelForm` that binds this field will write `''`, not `None`. `recorded_token is not None`
(`:502`) would then be True and `'' != token` would re-mint the night once on every sweep
until it self-heals. Safe today only because both admin surfaces list it in
`readonly_fields` — which is a policy, not a constraint.

**Fix:** drop `blank=True` (nothing needs to submit it), or normalise on read:
`recorded_token = existing.telescope_label_meta.minted_sub_night_window or None`.

---

### IN-02: `max_length=32` leaves exactly one character of headroom, untested

**File:** `solsys_code/models.py:117-119`, `solsys_code/migrations/0019_...py:16`

The worst-case token is two microsecond-precision `time.isoformat()` values plus a separator:
`'23:00:00.123456|05:00:00.123456'` = 31 characters. SQLite does not enforce `max_length`, so
a regression here would surface only on the PostgreSQL deployment CLAUDE.md names as the
production target. No test pins the bound.

**Fix:** use a round `max_length=64` and add
`self.assertLessEqual(len(_sub_night_provenance_token(run)), 32)` for a microsecond-valued
run. (CR-02's fix requires a widening anyway.)

---

### IN-03: the provenance write creates a companion row where the model documents "no row at all" as a meaningful state

**File:** `solsys_code/allocation_projector.py:405-420`

`_record_sub_night_provenance()` uses `update_or_create(event=event, ...)`, so on the
unrecorded-provenance branch it will create a `CalendarEventMeta` row for an event that had
none. Verified harmless today (probe 7 — the new row lands with `run=None`,
`is_verified=True`, and both `_may_write()` and `_clearable_declined_and_unattributed()`
treat "no row" and "row with `run IS NULL`" identically), but
`CalendarEventMeta`'s class docstring still says *"no row at all means 'verified' by
documented default"*, and this is now a second writer that silently converts shape (a) into
shape (b).

**Fix:** one sentence in the class docstring noting the allocation projector may materialise
a row solely to record provenance.

---

### IN-04: `allocation_projector.py` has no paired demo notebook, so a behaviour change of this size reached no executed documentation

**File:** `CLAUDE.md` (notebook pairing map), `docs/notebooks/pre_executed/`

The pairing map lists `campaign_reconciler.py` → `reconcile_campaign_runs_demo.ipynb` but has
no entry for `allocation_projector.py`, even though it now owns the entire per-night
projection and this round added a database column and a new re-mint rule to it. The runbook
clause (WR-03) is directory-scoped and does apply; the notebook clause does not, purely
because the map was never extended.

**Fix:** extend the map — `solsys_code/allocation_projector.py` →
`reconcile_campaign_runs_demo.ipynb` is the natural home, since that notebook already drives
`reconcile_run()`, which dispatches into this module.

---

_Reviewed: 2026-09-15T21:40:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
