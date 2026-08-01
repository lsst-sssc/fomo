---
phase: 28-operator-assisted-attribution
reviewed: 2026-08-01T20:06:16Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - docs/runbooks/telescope_runs_calendar.rst
  - solsys_code/admin.py
  - solsys_code/calendar_utils.py
  - solsys_code/campaign_attribution.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_urls.py
  - solsys_code/campaign_views.py
  - solsys_code/management/commands/sync_lco_observation_calendar.py
  - solsys_code/migrations/0013_attribution_dismissals_and_calendar_event_meta_audit.py
  - solsys_code/models.py
  - solsys_code/tests/test_admin.py
  - solsys_code/tests/test_attribution_dismissals.py
  - solsys_code/tests/test_calendar_utils.py
  - solsys_code/tests/test_campaign_attribution.py
  - solsys_code/tests/test_campaign_attribution_views.py
  - src/templates/campaigns/attribution_queue.html
  - src/templates/campaigns/campaign_list.html
findings:
  critical: 2
  warning: 2
  info: 1
  total: 5
status: issues_found
---

# Phase 28: Code Review Report

**Reviewed:** 2026-08-01T20:06:16Z
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Phase 28 adds the operator-assisted attribution worklist (matcher in
`campaign_attribution.py`, views/table classes in `campaign_views.py`/`campaign_tables.py`,
two new dismissal models plus `CalendarEventMeta` audit fields in `models.py`, and the
`attribution_queue.html` template). The matcher itself (scoring, banding, the campaign/target
boundary gate, dismissal exclusion) is well-tested and the arithmetic/edge-case handling holds
up under review. The migration matches the model diff exactly. Server-side re-validation on
every write path (`is_offered_candidate()`, `_is_sole_high_candidate()`) is genuinely present,
not just claimed in comments.

However, two BLOCKER-level defects were found: a client-side HTML validation bug that blocks
the primary "Confirm" action in the rendered template, and a gap in the new
`CalendarEventMeta` admin audit fields where the standalone admin page — which the code's own
docstring calls "the primary staff surface for hand-linking a run to an event" — does not
protect `confirmed_by`/`confirmed_at` the way the inline does, silently defeating the
audit-trail guarantee this phase exists to add. Two further WARNING-level issues concern write
ordering in `_undo_confirmation()` and a band-filter interaction with the checkbox-gate helper.

## Critical Issues

### CR-01: The "Confirm" button is blocked by the same `required` field meant only for Dismiss

**File:** `src/templates/campaigns/attribution_queue.html:91-103` (events table) and
`src/templates/campaigns/attribution_queue.html:177-189` (records table)
**Issue:** Both per-candidate action forms put the Confirm button, the free-text `reason`
input, and the Dismiss button inside one `<form>`:

```html
<form method="post" action="{% url 'campaigns:attribution_decide' %}" ...>
  ...
  <button type="submit" name="action" value="confirm" class="btn btn-sm btn-success">Confirm</button>
  <input type="text" name="reason" class="form-control form-control-sm" required
         placeholder="Why doesn't this candidate match? (required)">
  <button type="submit" name="action" value="dismiss" class="btn btn-sm btn-danger">Dismiss</button>
</form>
```

`reason` carries the HTML5 `required` attribute with no `formnovalidate` on the Confirm button
and no `novalidate` on the `<form>` itself. HTML5 constraint validation runs for *any*
submitter inside a form unless that submitter (or the form) opts out — so clicking **Confirm**
in a real browser is blocked by the browser's native validation UI until the operator types
something into the "Why doesn't this candidate match?" box, even though `reason` is only
meaningful for Dismiss (server-side, `AttributionDecisionView._confirm()`/`_do_confirm_event()`/
`_do_confirm_record()` never read `reason` at all). This is invisible to the test suite because
`self.client.post()` bypasses browser-side constraint validation entirely — every
`test_confirm_*` test in `test_campaign_attribution_views.py` posts directly and never exercises
the rendered `<form>`'s validation behaviour.
**Fix:** Add `formnovalidate` to the Confirm button (or move `reason` outside the shared form
context, e.g. give it `form="..."` targeting only a per-row dismiss action, or split Confirm
into its own minimal form):
```html
<button type="submit" name="action" value="confirm" class="btn btn-sm btn-success" formnovalidate>Confirm</button>
```

### CR-02: `CalendarEventMeta`'s standalone admin page lets staff hand-type `confirmed_by`/`confirmed_at`, defeating the D-12 audit trail

**File:** `solsys_code/admin.py:279-318` (`CalendarEventMetaAdmin`)
**Issue:** `CalendarEventMetaInline` (line 94) explicitly marks the new D-12 audit fields
read-only:

```python
readonly_fields = ['confirmed_by', 'confirmed_at']
```

with a docstring claiming "a staff member can never hand-type either value through this form."
But `CalendarEventMetaAdmin` — the **standalone** admin page, which its own
`get_readonly_fields()` docstring (line 291-313) calls "a second, independent write path onto
the same row, and 27.1-02 made it **the primary staff surface for hand-linking a run to an
event**" — only protects `event` (the frozen primary key). It never adds `confirmed_by`/
`confirmed_at` to `readonly_fields`, and there is no `save_model()` override to stamp them
automatically the way `CampaignRunAdmin.save_formset()` does for the inline path. Since Django's
`ModelAdmin` renders every non-readonly, non-excluded model field as an editable widget by
default, a staff user linking a run through this "primary" page can:

1. Leave `confirmed_by`/`confirmed_at` unset, so the newly-linked association carries no
   attribution at all — directly contradicting ROADMAP Phase 28 criterion 4 ("both a
   confirmation and an undo... attributable to a person and a time").
2. Submit an arbitrary `confirmed_by` (any User via the raw FK widget) and an arbitrary
   `confirmed_at` timestamp, fabricating attribution to someone who never made the decision.

No test in `test_admin.py`'s `CalendarEventMetaStandaloneAdminPkFreezeTests` (or elsewhere)
exercises this — that class only pins the `event` pk freeze, not the audit fields.
**Fix:** Add `confirmed_by`/`confirmed_at` to `CalendarEventMetaAdmin.get_readonly_fields()`
(mirroring the inline), and give the standalone admin its own `save_model()` override that
stamps `confirmed_by=request.user`/`confirmed_at=timezone.now()` on a genuine `run` None ->
not-None transition, the same way `CampaignRunAdmin.save_formset()` does today:
```python
readonly_fields = ['confirmed_by', 'confirmed_at']  # class-level, alongside the pk freeze

def save_model(self, request, obj, form, change):
    prior_run_id = None
    if change:
        prior_run_id = CalendarEventMeta.objects.filter(pk=obj.pk).values_list('run_id', flat=True).first()
    if prior_run_id is None and obj.run_id is not None:
        obj.confirmed_by = request.user
        obj.confirmed_at = timezone.now()
    super().save_model(request, obj, form, change)
```

## Warnings

### WR-01: `_undo_confirmation()` can write a dismissal row for a pair that was never actually undone

**File:** `solsys_code/campaign_views.py:1441-1485`
**Issue:** `_undo_confirmation()` first writes the "undo implies dismissal" row
(`CalendarEventDismissal`/`ObservationRecordDismissal.objects.get_or_create(...)`), and only
*afterward* runs the conditional update/delete that actually clears the confirmed link
(`CalendarEventMeta.objects.filter(event_id=orphan_pk, run_id=run_pk).update(...)` or the
equivalent `CampaignRunObservation` delete). If `run_pk` does not match the pair's actual
currently-confirmed run (a stale re-submit after a re-point, or a tampered/malformed POST —
`AttributionDecisionView` only validates `orphan_pk`/`run_pk` are integers, not that they
describe a real current confirmation), `changed_count` ends up `0` and the view reports "This
candidate no longer exists," but the dismissal row for that (possibly unrelated) pair has
already been committed inside the same `transaction.atomic()` block. Unlike `_dismiss()` (which
gates the same write on `is_offered_candidate()`-equivalent state via the offered-candidate
check upstream) and `_confirm()`/`_confirm_selected()` (which always call
`is_offered_candidate()` before writing), `_undo_confirmation()` performs no upstream check
that `(kind, orphan_pk, run_pk)` is a real, currently-confirmed pair before writing the
dismissal side effect. The result is a dismissal record for a pair that was never actually
associated, which permanently removes that pair from future candidate lists
(`candidates_for_event`/`candidates_for_record` exclude any dismissed run) even though nothing
was ever confirmed for it.
**Fix:** Only write the dismissal row after confirming the update/delete actually matched a
row (reorder the two operations), or gate the whole method on first checking
`CalendarEventMeta.objects.filter(event_id=orphan_pk, run_id=run_pk).exists()` /
`CampaignRunObservation.objects.filter(observation_record_id=orphan_pk, run_id=run_pk).exists()`
before writing the dismissal.

### WR-02: `sole_high_candidate_pk` violates its own "full uncapped list" contract under a band filter

**File:** `solsys_code/campaign_attribution.py:621-638` (`_sole_high_candidate_pk`) and its
call sites at lines 669 and 703 (`event_attribution_backlog()` / `record_attribution_backlog()`)
**Issue:** `_sole_high_candidate_pk()`'s docstring states its `candidates` argument must be
"the orphan's FULL candidate list (uncapped)". But both call sites compute it *after* the
band filter has already been applied:

```python
candidates = candidates_for_event(event)
if band is not None:
    candidates = [c for c in candidates if c.band == band]
if not candidates:
    continue
groups.append(AttributionOrphanGroup(
    ...,
    sole_high_candidate_pk=_sole_high_candidate_pk(candidates),  # band-filtered, not full
))
```

When `band` is `'medium'` or `'low'`, every High-band candidate has already been filtered out,
so `_sole_high_candidate_pk()` always returns `None` for that view — even when the orphan
genuinely has exactly one High-band candidate elsewhere in its full list. Today this is inert
(the checkbox only renders for rows in the already-filtered `group.candidates`, none of which
are High-band under a medium/low filter, so no visibly-wrong checkbox appears), but it silently
breaks the function's documented invariant and is a latent trap for the next person who reads
`_sole_high_candidate_pk`'s docstring and trusts it, or who adds any other consumer of
`group.sole_high_candidate_pk`.
**Fix:** Compute `sole_high_candidate_pk` from the unfiltered `candidates_for_event(event)` /
`candidates_for_record(record)` result before applying the band filter, e.g.:
```python
full_candidates = candidates_for_event(event)
candidates = [c for c in full_candidates if c.band == band] if band is not None else full_candidates
if not candidates:
    continue
...
sole_high_candidate_pk=_sole_high_candidate_pk(full_candidates),
```

## Info

### IN-01: Candidate filtering compares the rounded display score, not the raw signal score

**File:** `solsys_code/campaign_attribution.py:539-541` (`candidates_for_event`) and
`598-600` (`candidates_for_record`)
**Issue:** `_build_candidate()` stores `score=round(score, 2)` on the returned
`AttributionCandidate`, and both `candidates_for_event()`/`candidates_for_record()` then drop a
candidate via `if candidate.score <= 0.0: continue` — i.e. the *rounded* score, not the raw
weighted sum. A raw score of e.g. `0.004` (genuine, if faint, evidence on some signal) rounds to
`0.0` and gets dropped as if it had none at all, contradicting the docstring's "drops candidates
whose total score is 0.0 (no evidence on any signal at all... not a per-signal gate)." In
practice the weights (0.25/0.35/0.40) make sub-0.005 nonzero totals rare, so this is unlikely to
change real-world results, but it is a real deviation from the stated contract.
**Fix:** Filter on the raw (unrounded) `score` local variable inside `_build_candidate()`
(e.g. return the raw score alongside the rounded display value, or filter in
`_build_candidate()` itself before rounding) rather than the already-rounded
`AttributionCandidate.score` field.

---

_Reviewed: 2026-08-01T20:06:16Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
