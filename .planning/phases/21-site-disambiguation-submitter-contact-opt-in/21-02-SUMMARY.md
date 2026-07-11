---
phase: 21-site-disambiguation-submitter-contact-opt-in
plan: 02
subsystem: api
tags: [django, django-tables2, django-orm, case-when, pii-gating, forms]

# Dependency graph
requires:
  - phase: 16 (Submission Form, Approval Queue & Calendar Projection)
    provides: CampaignRunSubmissionForm, CampaignRunSubmissionView.form_valid, CampaignRunTableView PII-gating discipline
  - phase: 15 (Per-Campaign Table View)
    provides: ALLOWED_FIELDS_FOR_NON_STAFF / get_queryset() / get_table_kwargs() non-staff PII-safe .values() pattern
provides:
  - "CampaignRun.contact_public_opt_in BooleanField (default=False) + migration 0007"
  - "Submission-form opt-in checkbox persisting onto the created CampaignRun"
  - "Queryset-level per-row PII exposure gate (Case/When keyed on contact_public_opt_in)"
affects: [21-03, 21-04 (site-disambiguation UI plans sharing campaign_views.py, but scoped away from this plan's touched methods)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Django .values()-before-.annotate() ordering to alias an annotation with a name that
      collides with a real model field (Case/When PII gate)"

key-files:
  created:
    - solsys_code/migrations/0007_campaignrun_contact_public_opt_in.py
  modified:
    - solsys_code/models.py
    - solsys_code/campaign_forms.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_forms.py
    - solsys_code/tests/test_campaign_submission.py
    - solsys_code/tests/test_campaign_views.py

key-decisions:
  - "Reordered the Case/When annotation to run AFTER .values() narrows the field list, not
    before as PATTERNS.md literally specified -- Django raises ValueError ('annotation
    conflicts with a field on the model') when an annotation alias (contact_person/
    contact_email) matches a real model field name and .values() hasn't already restricted
    QuerySet._fields. Confirmed via a live shell query before committing the fix."
  - "Updated Phase 15's TestContactFieldGating.test_anonymous_context_rows_have_no_contact_fields
    to assert the keys are present-but-blank rather than absent -- VIEW-05 intentionally
    changes the non-staff .values() dict shape from 'no contact keys at all' to 'always
    present, blank unless opted in', a design consequence of the annotation gate, not a
    regression."

requirements-completed: [VIEW-05]

coverage:
  - id: D1
    description: "Submitter can tick a single 'show my contact info publicly' checkbox on the submission form (default unchecked); value persists onto the created CampaignRun"
    requirement: "VIEW-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_contact_public_opt_in_unchecked_defaults_false"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_forms.py#test_contact_public_opt_in_checked_cleans_true"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_submission.py#test_contact_public_opt_in_checked_persists_true"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_submission.py#test_contact_public_opt_in_unchecked_persists_false"
        status: pass
    human_judgment: false
  - id: D2
    description: "An opted-in run's contact_person/contact_email are visible to anonymous visitors on the per-campaign table; an opted-out run's contact fields are never present in the non-staff queryset SELECT (blanked at SQL, not template)"
    requirement: "VIEW-05"
    verification:
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestContactPublicOptIn.test_opted_in_row_exposes_contact_in_non_staff_values"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestContactPublicOptIn.test_opted_out_row_blanks_contact_in_non_staff_values"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestContactPublicOptIn.test_opted_in_content_visible_to_anonymous_visitor"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestContactPublicOptIn.test_opted_out_content_not_visible_to_anonymous_visitor"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestContactPublicOptIn.test_staff_sees_both_regardless_of_opt_in"
        status: pass
      - kind: unit
        ref: "solsys_code/tests/test_campaign_views.py#TestContactPublicOptIn.test_allowed_fields_for_non_staff_does_not_list_contact_fields"
        status: pass
    human_judgment: false

# Metrics
duration: 14min
completed: 2026-07-11
status: complete
---

# Phase 21 Plan 02: Submitter Contact Opt-In Summary

**A default-opt-out `contact_public_opt_in` checkbox on the public submission form drives a Case/When queryset annotation that exposes `contact_person`/`contact_email` to anonymous visitors only for opted-in `CampaignRun` rows, gated at the SQL SELECT.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-07-11T11:49:47Z
- **Completed:** 2026-07-11T12:02:57Z
- **Tasks:** 3
- **Files modified:** 7 (1 created: migration; 6 modified: model, form, view, 3 test files)

## Accomplishments
- `CampaignRun.contact_public_opt_in` field + additive migration `0007`, applied clean with no backfill
- `CampaignRunSubmissionForm` opt-in checkbox (label "Show contact info publicly?", default unchecked) persisted through `CampaignRunSubmissionView.form_valid`
- `CampaignRunTableView.get_queryset()` non-staff branch gates `contact_person`/`contact_email` via a per-row `Case`/`When` annotation keyed on `contact_public_opt_in` -- opted-out rows' real PII is never fetched by the SQL SELECT, only an empty string
- `ALLOWED_FIELDS_FOR_NON_STAFF` deliberately left untouched (contact fields never added to the allow-list directly), preserving the existing "restrict the queryset, not the template" discipline

## Task Commits

Each task was committed atomically:

1. **Task 1: Model field + additive migration** - `eb2ec05` (feat)
2. **Task 2: Submission-form checkbox + persist through form_valid** - `4e4fa94` (test, RED) → `1eccfac` (feat, GREEN)
3. **Task 3: Queryset-level per-row PII gate (Case/When)** - `5733bd8` (test, RED) → `8e95b14` (feat, GREEN)

_TDD tasks (2 and 3) each have a test → feat commit pair._

## Files Created/Modified
- `solsys_code/models.py` - `CampaignRun.contact_public_opt_in` BooleanField
- `solsys_code/migrations/0007_campaignrun_contact_public_opt_in.py` - additive AddField migration
- `solsys_code/campaign_forms.py` - `CampaignRunSubmissionForm.contact_public_opt_in` checkbox in the Contact fieldset
- `solsys_code/campaign_views.py` - `form_valid` persists the opt-in value; `get_queryset()`/`get_table_kwargs()` Case/When PII gate
- `solsys_code/tests/test_campaign_forms.py` - field presence/required/clean-value/label tests
- `solsys_code/tests/test_campaign_submission.py` - checked/unchecked persistence tests
- `solsys_code/tests/test_campaign_views.py` - `TestContactPublicOptIn` (6 new tests) + updated Phase 15 `TestContactFieldGating` assertion

## Decisions Made
- **Annotation ordering fix (Rule 1 - bug):** `21-PATTERNS.md`'s literal code sample calls `.annotate(_public_contact_person=..., ...)` then `.values(..., contact_person=F('_public_contact_person'), ...)`. Running that exact sequence raises `ValueError: The annotation 'contact_person' conflicts with a field on the model` because `contact_person`/`contact_email` are real `CampaignRun` fields and the alias-collision check only skips fields already excluded via a prior `.values()` call. Fixed by calling `.values(*base_fields_without_contact)` first (narrowing `QuerySet._fields`), then `.annotate(contact_person=Case(...), contact_email=Case(...))` — verified live in a Django shell before committing, and confirmed the Case/When condition can still reference `contact_public_opt_in` and the raw `contact_person`/`contact_email` columns via `F()` even though they aren't in the `.values()` field list.
- **Updated a Phase 15 test's assumption (Rule 1 - bug/regression fix):** `TestContactFieldGating.test_anonymous_context_rows_have_no_contact_fields` previously asserted `contact_person`/`contact_email` keys were entirely absent from non-staff `.values()` rows. VIEW-05 intentionally changes this contract — the keys are now always present (blank string when not opted in). Updated the assertion to check for blank values rather than key absence, since none of that test's fixture rows have `contact_public_opt_in=True`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Django annotation/field-name collision in the Case/When PII gate**
- **Found during:** Task 3 (Queryset-level per-row PII gate)
- **Issue:** The plan's/PATTERNS.md's specified `.annotate()`-then-`.values()` ordering raises `ValueError` because the new annotation aliases (`contact_person`, `contact_email`) collide with real `CampaignRun` model field names, and Django's collision check isn't bypassed until `.values()` has already narrowed the field set.
- **Fix:** Reordered to `.values(*fields_without_contact)` first, then `.annotate(contact_person=Case(...), contact_email=Case(...))` — functionally identical output (verified against the plan's acceptance criteria), different call order.
- **Files modified:** `solsys_code/campaign_views.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_views` passes; live shell query confirmed the reordered queryset returns correct blank/populated values.
- **Committed in:** `8e95b14` (Task 3 GREEN commit)

**2. [Rule 1 - Bug] Outdated Phase 15 test assertion**
- **Found during:** Task 3 (Queryset-level per-row PII gate)
- **Issue:** `TestContactFieldGating.test_anonymous_context_rows_have_no_contact_fields` asserted the non-staff `.values()` dict never contains `contact_person`/`contact_email` keys at all — true under Phase 15's design, no longer true once VIEW-05's annotation always includes those keys (blank by default).
- **Fix:** Updated the assertion to check the values are empty strings for this fixture's (all opted-out) rows, preserving the PII-safety intent while matching the new, intended dict shape.
- **Files modified:** `solsys_code/tests/test_campaign_views.py`
- **Verification:** `python manage.py test solsys_code.tests.test_campaign_views` — 33/33 pass.
- **Committed in:** `5733bd8` (Task 3 RED commit, alongside the new test class)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bug fixes necessary for correctness; no scope creep)
**Impact on plan:** Both fixes were required to make the plan's specified behavior actually work under Django's ORM constraints and to keep the pre-existing test suite internally consistent with the new intended contract. No architectural changes, no new files beyond the planned migration.

## Issues Encountered
- Django's `QuerySet.annotate()` rejects an alias matching a real model field name unless `.values()` has already restricted the field list — not obvious from the plan's code sample; resolved via the ordering fix documented above (Rule 1).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- VIEW-05 fully shipped: opt-in checkbox + persist + queryset-level PII gate, all GREEN with dedicated regression tests.
- `solsys_code/campaign_views.py` was touched only in `CampaignRunTableView.get_queryset()`/`get_table_kwargs()` and `CampaignRunSubmissionView.form_valid` — the approval-queue/decision views (`ApprovalQueueView`, `CampaignRunDecisionView`) that 21-01/21-03/21-04 touch were not modified, so this plan stays structurally independent of the site-disambiguation plans as designed.
- Full `./manage.py test solsys_code` suite (405 tests) and `ruff check .`/`ruff format --check .` (on all files this plan touched) both clean.
- No paired demo notebook update required — this plan doesn't touch any of the four CLAUDE.md-listed notebook-paired modules (`telescope_runs.py`, `load_telescope_runs.py`, `sync_lco_observation_calendar.py`, `sync_gemini_observation_calendar.py`).

---
*Phase: 21-site-disambiguation-submitter-contact-opt-in*
*Completed: 2026-07-11*
