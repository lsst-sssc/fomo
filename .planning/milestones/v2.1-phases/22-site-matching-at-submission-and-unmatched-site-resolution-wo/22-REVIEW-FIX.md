---
phase: 22-site-matching-at-submission-and-unmatched-site-resolution-wo
fixed_at: 2026-07-15T00:00:00Z
review_path: .planning/phases/22-site-matching-at-submission-and-unmatched-site-resolution-wo/22-REVIEW.md
iteration: 1
findings_in_scope:
  - CR-01
  - CR-02
  - WR-01
  - WR-02
  - WR-03
fixed:
  - CR-01
  - CR-02
  - WR-01
  - WR-02
  - WR-03
skipped:
  - IN-01
status: fixed
---

# Phase 22: Code Review Fix Report (iteration 1)

Fix scope: `critical_warning` (CR-01, CR-02, WR-01, WR-02, WR-03). IN-01 (info) was out of
scope but its suggested regression test was added anyway as a natural fallout of fixing CR-01
(same scenario the review's suggested test targets).

## CR-01: `resolve_site()` reports a placeholder hit as a genuine resolution

**Commit:** `bd80c0d` — `fix(22): CR-01 resolve_site() no longer reports a placeholder hit as genuine`

**Fix applied as suggested.** All three success paths in `resolve_site()`
(`solsys_code/campaign_utils.py`) now derive `needs_review` from
`is_placeholder_observatory(obs)` instead of returning `False` unconditionally:

- Tier 1 (existing-Observatory match)
- Tier 2 success (`fetcher.to_observatory()`) — changed for consistency per the review's
  note, even though this path never itself produces a placeholder-named row
- Tier 2's `IntegrityError` race-recovery re-fetch

**Regression test added:** `TestApprovalSiteResolution.test_resolve_site_tier1_hit_on_existing_placeholder_still_flags_review`
in `solsys_code/tests/test_campaign_approval.py` — this is the exact test IN-01 suggested
(creates a pre-existing placeholder Observatory, calls `resolve_site()` again for the same
obscode, asserts `needs_review=True` and no second placeholder fabricated).

## CR-02: Placeholder Observatories pollute the site-search candidate pool

**Commit:** `c36657c` — `fix(22): CR-02 exclude placeholder Observatories from the search candidate pool`

**Fix applied as suggested.** `_local_observatory_candidates()` now iterates
`Observatory.objects.exclude(name__startswith=NEEDS_REVIEW_NAME_PREFIX)` instead of
`Observatory.objects.all()`.

**Regression test added:** `TestSiteFuzzyMatch.test_build_site_candidates_excludes_placeholder_observatories`
— creates a placeholder Observatory and asserts neither its obscode/short_name nor its
`'NEEDS REVIEW: ...'` display name appears in `build_site_candidates()`'s merged pool.

## WR-01: Approve branch isn't placeholder-aware

**Commit:** `ed432ec` — `fix(22): WR-01 approve branch re-resolves an existing placeholder site`

**Fix applied as suggested.** `CampaignRunDecisionView.post()`'s approve branch guard is now
`if run.site is None or is_placeholder_observatory(run.site):`, mirroring `_resolve_site()`.

**Regression test added:** `TestSiteSelectionResolution.test_approve_re_resolves_when_existing_site_is_a_placeholder`
— a PENDING_REVIEW run pre-seeded with a placeholder `site` is approved with a valid
`site_selection`; asserts the site is replaced and `site_needs_review` clears. Without the
fix, `run.site is None` is `False` (a placeholder Observatory is still an Observatory row),
so resolution would never run.

## WR-02: Placeholder detection is a magic string prefix on a user-editable field

**Commit:** `d6bc732` — `fix(22): WR-02 reject the reserved placeholder name prefix on form-validated saves`

**Deviation from the suggested default, with rationale.** The task brief asked for the
lighter-weight option (reserved-prefix validation in `CreateObservatoryForm`) to be
preferred over a new model field/migration "unless a clearly better approach". Neither of
those two options was actually the right fix once I traced the real attack surface:

- `CreateObservatoryForm` (`solsys_code/solsys_code_observatory/forms.py`) has exactly one
  field, `obscode` (3 characters). `name` is never staff-typed through that form — it's
  always populated from the MPC API response inside `CreateObservatory.form_valid()` /
  `MPCObscodeFetcher.to_observatory()`. Adding prefix validation to that form's
  `clean_obscode()` would not protect `name` at all (a 3-char obscode can't carry the
  ~14-character reserved prefix anyway), so it would be a no-op fix for the scenario the
  review actually describes.
- The real staff-accessible surface where `Observatory.name` is directly editable is the
  Django admin change form (`solsys_code/solsys_code_observatory/admin.py`'s
  `ObservatoryAdmin`, which exposes all model fields with no `fields`/`exclude`
  restriction). There is no other create/edit view or form in the codebase that exposes
  `name` for free-text entry.
- A new dedicated boolean field would need a migration and, more importantly, still
  requires *some* validation layer to keep it in sync when the field is set by hand (e.g.
  via admin) — it doesn't remove the need for a guard, just relocates the marker.

Given that, the fix is a model-level `Observatory.clean()` override
(`solsys_code/solsys_code_observatory/models.py`) that rejects a `name` starting with the
reserved prefix. This:

- Fires on any form-validated save (`full_clean()`), which covers the Django admin change
  form — the actual reachable vector — without a migration.
- Never blocks `resolve_site()`'s own tier-3 placeholder creation, because that path uses a
  plain `Observatory.objects.create()` (bypasses `full_clean()`), confirmed by a new
  regression test.
- Keeps `NEEDS_REVIEW_NAME_PREFIX` duplicated (not imported) between
  `solsys_code/solsys_code_observatory/models.py` and `solsys_code/campaign_utils.py`,
  documented inline: importing it from `campaign_utils` into `models.py` would create a
  circular import, since `campaign_utils` already imports `Observatory` from `models.py`.

**Regression tests added** in `solsys_code/solsys_code_observatory/tests/test_models.py`:
- `test_clean_rejects_reserved_needs_review_name_prefix` — `clean()` raises `ValidationError`
  keyed on `'name'` for a reserved-prefixed name (called directly, not via `full_clean()`, to
  isolate the check from unrelated required-field validation on `lat`/`lon`).
- `test_clean_allows_ordinary_name` — an ordinary name passes `clean()` unaffected.
- `test_tier3_placeholder_create_bypasses_full_clean` — `Observatory.objects.create()` with
  a reserved-prefixed name still succeeds (the legitimate placeholder-creation path is
  unaffected).

## WR-03: Replaced placeholder Observatory rows are never cleaned up

**Commit:** `63b7cef` — `fix(22): WR-03 delete orphaned placeholder Observatory after replacement`

**Fix applied as suggested**, with an added safety guard beyond the review's minimal
suggestion. In `CampaignRunDecisionView._resolve_site()`
(`solsys_code/campaign_views.py`), after the site-write claim succeeds, the previously-set
placeholder Observatory (captured as `previous_site_id` before the write) is deleted if:

1. it still exists,
2. `is_placeholder_observatory()` confirms it actually is a placeholder, and
3. no other `CampaignRun` still has `site_id` pointing at it.

Condition 3 wasn't in the review's one-line suggestion but is necessary: the same
placeholder obscode can be shared by more than one still-unresolved `CampaignRun` (e.g.
several CSV-imported rows at one still-unconfigured site, per CR-01's own bootstrap-import
scenario) — deleting it as soon as the first one is replaced would silently orphan the
`site` foreign key on every sibling row still pointing at it.

**Regression tests added** in `TestPlaceholderSiteReplacement`
(`solsys_code/tests/test_campaign_approval.py`):
- `test_placeholder_replacement_deletes_orphaned_placeholder_observatory` — the placeholder
  row no longer exists after a successful replacement.
- `test_placeholder_replacement_keeps_placeholder_still_referenced_by_another_run` — a
  second `CampaignRun` still pointing at the same placeholder prevents its deletion, and
  that other run's `site` foreign key is left intact.

## Verification

- `ruff check .` and `ruff format --check .` clean on all 5 touched files.
- `python manage.py test solsys_code`: **495 tests, all passing** (487 pre-existing + 8 new
  regression tests: 1 for CR-01, 1 for CR-02, 1 for WR-01, 2 for WR-03, 3 for WR-02).
- All previously-verified invariants re-checked and still intact after these fixes:
  - The *original-cycle* CR-01 (a genuinely-resolved projection-failed retry row still
    renders plain text) — unaffected; these fixes don't touch `render_site()`.
  - WR-01 from the original cycle (`show_actions=False` never renders the widget) —
    unaffected; these fixes don't touch table rendering.
  - D-06 (a genuinely-resolved, non-placeholder site is never silently re-resolved) — the
    `TestPlaceholderSiteReplacement.test_genuine_site_still_never_re_resolved_when_replacing_placeholder_would_apply`
    test (pre-existing, unmodified) still passes: `is_placeholder_observatory()` on a real
    site still returns `False`, so none of these guard changes touch a genuinely-resolved
    site's resolution path.
- Target test factory convention (CLAUDE.md): not applicable — no test added or modified in
  this pass fixtures a `tom_targets.models.Target`.

## Commits

| Finding | Commit | Files |
|---|---|---|
| CR-01 | `bd80c0d` | `campaign_utils.py`, `tests/test_campaign_approval.py` |
| CR-02 | `c36657c` | `campaign_utils.py`, `tests/test_campaign_approval.py` |
| WR-01 | `ed432ec` | `campaign_views.py`, `tests/test_campaign_approval.py` |
| WR-03 | `63b7cef` | `campaign_views.py`, `tests/test_campaign_approval.py` |
| WR-02 | `d6bc732` | `solsys_code_observatory/models.py`, `solsys_code_observatory/tests/test_models.py` |

_Fixed: 2026-07-15_
_Fixer: Claude (gsd-code-fixer)_
