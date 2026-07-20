---
phase: quick-260714-ilz
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_forms.py
  - solsys_code/campaign_views.py
  - solsys_code/tests/test_campaign_forms.py
  - solsys_code/tests/test_campaign_submission.py
autonomous: true
requirements: [SUBMIT-01]

must_haves:
  truths:
    - "Submitting a genuine multi-night range (e.g. '2027-04-20 -- 2027-05-11') through the public form creates one CampaignRun with window_start=2027-04-20 and window_end=2027-05-11 — no Django 'Enter a valid date' validation failure."
    - "A single date, an identical start==end range (e.g. '2027-04-20 -- 2027-04-20'), and a 'to'-separated equal-endpoint range all still collapse to window_start == window_end (single-night behavior preserved, requirement 4)."
    - "Blank obs_date still produces a TBD run: window_start and window_end both None (requirement 5), consistent with the CSV importer's TBD handling."
    - "Genuinely unparseable non-blank obs_date text (e.g. 'sometime next spring', or a reversed range) produces a friendly obs_date form error and re-renders the form (HTTP 200), never a 500, and creates no CampaignRun."
    - "Two submissions proposing the same resolved window (single OR range) surface the existing friendly IntegrityError non-field form error, not a 500 — the existing except-IntegrityError handler already covers the range case (requirement 7)."
  artifacts:
    - solsys_code/campaign_forms.py
    - solsys_code/campaign_views.py
    - solsys_code/tests/test_campaign_forms.py
    - solsys_code/tests/test_campaign_submission.py
  key_links:
    - "CampaignRunSubmissionForm.clean() -> parse_obs_window(obs_date_raw, '') -> cleaned_data['window_start'] / cleaned_data['window_end']"
    - "CampaignRunSubmissionView.form_valid() -> form.cleaned_data['window_start'] / ['window_end'] -> CampaignRun.objects.create(window_start=..., window_end=...)"
    - "window_needs_review True AND non-blank raw text -> form.add_error('obs_date', ...) (friendly, non-500); window_needs_review True AND blank -> TBD run, no error"
---

<objective>
Close the date-format gap on the public campaign-run submission form. Today
`CampaignRunSubmissionForm.obs_date` is a `forms.DateField` that only accepts a single
`YYYY-MM-DD` date, and `CampaignRunSubmissionView.form_valid()` unconditionally collapses it to
`window_start=obs_date, window_end=obs_date`. A submitter proposing an extended/multi-night run
(e.g. `'2027-04-20 -- 2027-05-11'`) gets a hard Django "Enter a valid date" failure and cannot
submit a range at all.

Wire the already-built, already-tested `campaign_utils.parse_obs_window()` (Phase 20, IMPORT-01/02 —
proven against real 3I/ATLAS sheet shapes: single dates, full/compact ranges, TBD text) into the
public form, mirroring the CSV importer's needs-review discipline but adapted to Django form-error
conventions. Do NOT invent new parsing logic.

Purpose: give public submitters the same flexible date/range intake the CSV bootstrap already has,
without regressing single-night collapse or blank->TBD behavior.
Output: obs_date accepts flexible text; parsed window mapped to window_start/window_end on save;
friendly non-500 error for unparseable non-blank text; full test coverage for the six required cases.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@solsys_code/campaign_forms.py
@solsys_code/campaign_views.py
@solsys_code/campaign_utils.py
@solsys_code/models.py
@solsys_code/management/commands/import_campaign_csv.py
@solsys_code/tests/test_campaign_forms.py
@solsys_code/tests/test_campaign_submission.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Accept flexible date/range input on the public form and map the parsed window on save</name>
  <files>solsys_code/campaign_forms.py, solsys_code/campaign_views.py</files>
  <action>
In `solsys_code/campaign_forms.py`, on `CampaignRunSubmissionForm`:

1. Change the `obs_date` field from `forms.DateField(required=False, label='Observation date')` to
   `forms.CharField(required=False, label='Observation date', max_length=255, help_text=...)`. The
   `max_length=255` mirrors the model's `original_obs_date_raw` CharField length and bounds untrusted
   free-text input (T-ilz-02). The help_text must tell submitters the accepted shapes: a single
   `YYYY-MM-DD` date, a date range (`YYYY-MM-DD -- YYYY-MM-DD` or `YYYY-MM-DD to YYYY-MM-DD`), or blank
   if not yet scheduled (TBD). Update the existing A3 comment above the field to note it is now free
   text parsed by `parse_obs_window()` rather than a strict single date. Keep the field in the same
   layout position.

2. Add a whole-form `clean()` method (import `parse_obs_window` from `solsys_code.campaign_utils`).
   It runs after per-field cleaning, calls `super().clean()`, reads the raw obs_date string from
   `cleaned_data.get('obs_date', '') or ''`, and calls `parse_obs_window(obs_date_raw, '')` — pass an
   empty UT-range string because the public form has no UT-time field (the dropped UT start/end inputs,
   per the field's existing A3 comment). Unpack window_start, window_end, and window_needs_review from
   the 7-tuple (ignore original_obs_date_raw and the ut_* values). Always set
   `cleaned_data['window_start']` and `cleaned_data['window_end']` to the parsed values so the view
   reads a consistent contract. If `window_needs_review` is True AND `obs_date_raw.strip()` is
   non-empty, call `self.add_error('obs_date', <friendly message>)` telling the submitter the date
   couldn't be understood and to use a single date or a `start -- end` / `start to end` range (or leave
   it blank for TBD). This is the ONLY error path: blank text also yields window_needs_review True but
   MUST NOT error — it is an intentional TBD (requirement 5). Mirror `import_campaign_csv`'s
   needs-review discipline (never raise; act on the parser's flag) but adapt to Django convention
   (`form.add_error`, not a silent skip — requirement 2). Return `cleaned_data`.

   Note (do not add code for this — it is a correctness invariant to preserve): `parse_obs_window`
   always returns both-None (TBD/blank) or both-set (single-night collapse when start==end, OR a
   range), never one-None, so the model's `campaign_run_window_start_end_null_together` CheckConstraint
   invariant always holds. A reversed range (end<start) and a `'YYYY-MM-?'` marker both fall through to
   the unparseable-non-blank branch and become a friendly error, which is the desired behavior.

In `solsys_code/campaign_views.py`, in `CampaignRunSubmissionView.form_valid()`:

3. Replace the two lines `window_start=form.cleaned_data['obs_date']` and
   `window_end=form.cleaned_data['obs_date']` inside `CampaignRun.objects.create(...)` with
   `window_start=form.cleaned_data['window_start']` and `window_end=form.cleaned_data['window_end']`
   (the parsed values from the form's clean()). Update the SCHED-02 inline comment: single-night
   collapse and blank->TBD now come from `parse_obs_window`'s contract (start==end for a single date;
   both None for blank), not from a raw date field. Requirement 3.

4. Requirement 7 (investigate + flag, not necessarily fix): the existing `except IntegrityError` block
   already covers the range case. Two identical resolved windows (single OR range) collide on the
   `unique_campaign_run_resolved_window` partial constraint keyed on
   `(campaign, telescope_instrument, window_start, window_end)`; a blank/TBD row collides on
   `unique_campaign_run_tbd_natural_key`. Both raise IntegrityError caught by the existing handler. A
   single-night entry and a range starting the same day do NOT collide because window_end differs
   (per the model's own constraint comment) — correctly distinct rows, no false merge. There is NO new
   IntegrityError gap. Make one small wording fix: soften the existing friendly message so it reads
   correctly for ranges as well as single dates (e.g. "on this observing date" -> "for this observing
   window"). Record this finding explicitly in the SUMMARY: "requirement 7 investigated — existing
   except-IntegrityError handler covers the range case unchanged; only the user-facing message wording
   was broadened; no schema/handler change needed."

Keep single quotes and 120-col style (ruff). Do not touch the honeypot, notification, or any other
field. This change touches none of the four CLAUDE.md notebook-paired modules, so no demo notebook
update is required (requirement 9).
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_campaign_forms solsys_code.tests.test_campaign_submission -v1 && ruff check solsys_code/campaign_forms.py solsys_code/campaign_views.py && ruff format --check solsys_code/campaign_forms.py solsys_code/campaign_views.py</automated>
  </verify>
  <done>
obs_date is a CharField(required=False, max_length=255) with range-aware help_text; the form's
clean() calls parse_obs_window(obs_date_raw, '') and sets cleaned_data['window_start'] /
['window_end']; non-blank unparseable text produces an 'obs_date' form error while blank yields both
None (no error); form_valid() maps the parsed window into CampaignRun.create; the IntegrityError
message is broadened for windows. The pre-existing campaign form/submission tests still pass and ruff
is clean on both touched files.
  </done>
</task>

<task type="auto">
  <name>Task 2: Cover single/identical-range/1-night/multi-night/TBD/unparseable + range-duplicate collision</name>
  <files>solsys_code/tests/test_campaign_forms.py, solsys_code/tests/test_campaign_submission.py</files>
  <action>
Add tests for every case in requirement 6, split across the two existing files, reusing their existing
fixtures and helpers (`CampaignRunSubmissionFormTest._minimal_data(**overrides)` and
`CampaignSubmissionTestBase.minimal_valid_data(**overrides)`; campaigns are `TargetList` objects, so no
Target fixture is needed — but if any new fixture touches `Target`, use
`tom_targets.tests.factories.NonSiderealTargetFactory` per CLAUDE.md, requirement 10).

In `solsys_code/tests/test_campaign_forms.py` (form-level: assert `is_valid()` and the derived
`cleaned_data['window_start'] / ['window_end']`, accessed after `form.is_valid()`):
- Single date `'2027-04-20'` -> valid; window_start == window_end == date(2027, 4, 20).
- Identical-range `'2027-04-20 -- 2027-04-20'` (double-hyphen/en-dash separator) -> valid; start == end
  (requirement 4).
- Equal-endpoint 'to'-separated range `'2027-04-20 to 2027-04-20'` -> valid; start == end (exercises the
  second separator path collapsing to a single night — the "1-night range" case).
- Genuine multi-night range `'2027-04-20 -- 2027-05-11'` -> valid; window_start == date(2027, 4, 20),
  window_end == date(2027, 5, 11).
- Blank obs_date (omit it from `_minimal_data`) -> valid; window_start is None and window_end is None
  (TBD, requirement 5).
- Unparseable non-blank text `'sometime next spring'` -> NOT valid; 'obs_date' in form.errors with a
  non-empty message (requirement 2).
- Reversed range `'2027-05-11 -- 2027-04-20'` -> NOT valid; 'obs_date' in form.errors (documents the
  reversed-range -> friendly-error behavior).

In `solsys_code/tests/test_campaign_submission.py` (end-to-end POST through `campaigns:submit`, DB
effect, HTTP status — proving no 500):
- Multi-night range POST (`obs_date='2027-04-20 -- 2027-05-11'`) -> exactly one CampaignRun with
  window_start == date(2027, 4, 20), window_end == date(2027, 5, 11); redirects to the thanks page.
- Blank obs_date POST -> one CampaignRun that is TBD: window_start is None and window_end is None;
  redirects to thanks (requirement 5).
- Unparseable obs_date POST (`obs_date='sometime next spring'`) -> response status 200 (form
  re-rendered, NOT 302, NOT 500); zero CampaignRun rows created; the response contains an obs_date
  form error (requirement 2/6).
- Duplicate identical range: POST `obs_date='2027-04-20 -- 2027-05-11'` twice -> first creates one run
  and redirects; second returns status 200 (re-rendered form, not a 500), still exactly one
  CampaignRun, and surfaces the friendly non-field IntegrityError message — proving requirement 7's
  existing handler covers the range case.

Follow the existing modules' assertion style (assertRedirects, assertEqual on CampaignRun.objects
counts/fields, assertIn on form.errors / response content). Keep single quotes and 120-col style.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_campaign_forms solsys_code.tests.test_campaign_submission -v1 && ruff check solsys_code/tests/test_campaign_forms.py solsys_code/tests/test_campaign_submission.py && ruff format --check solsys_code/tests/test_campaign_forms.py solsys_code/tests/test_campaign_submission.py</automated>
  </verify>
  <done>
New tests cover all six required cases (single date, identical start==end range, equal-endpoint
'to'-range, genuine multi-night range, TBD/blank, unparseable string with friendly non-500 error) plus
reversed-range and range-duplicate collision. The full campaign form + submission suite passes and ruff
is clean on both touched test files.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| anonymous public browser -> CampaignRunSubmissionView | Untrusted free-text obs_date (and other fields) cross into the write path; no auth required for submission. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-ilz-01 | Tampering | CampaignRunSubmissionForm.clean() obs_date parse | low | mitigate | Reuse the already-hardened `parse_obs_window()` (never raises; anchored `^...$` linear regexes, no eval). Non-blank unparseable text is rejected via `form.add_error` (friendly, HTTP 200), never persisted, never a 500. |
| T-ilz-02 | Denial of Service | obs_date free-text length | low | mitigate | Add `max_length=255` on the CharField (mirrors `original_obs_date_raw`) to bound input; parse regexes are anchored and linear (no catastrophic backtracking). |
| T-ilz-03 | Tampering | Natural-key IntegrityError on range submit | low | accept | Existing `except IntegrityError` savepoint block already degrades a duplicate resolved-window (single or range) to a friendly non-field error; no new handler needed (requirement 7 finding). |
| T-ilz-SC | Tampering | package installs | low | accept | No new npm/pip/cargo packages installed — parse logic and all imports already exist in-repo (parse_obs_window from Phase 20). No supply-chain surface added. |
</threat_model>

<verification>
- `./manage.py test solsys_code.tests.test_campaign_forms solsys_code.tests.test_campaign_submission -v1` passes (scoped to these two modules to avoid the ~1.6 GB SPICE kernel import that broad `solsys_code` collection triggers — neither module imports ephem_utils).
- `ruff check .` and `ruff format --check .` stay clean on all four touched files (requirement 8).
- Manual reasoning confirmed (record in SUMMARY): parse_obs_window's both-None-or-both-set contract preserves the model's null-together CheckConstraint; the existing IntegrityError handler covers the range collision case unchanged (requirement 7 — no gap).
</verification>

<success_criteria>
- A multi-night range submitted through the public form creates a CampaignRun spanning window_start..window_end (requirement 1/3).
- Single date, identical start==end range, and equal-endpoint range all still collapse to window_start == window_end (requirement 4).
- Blank input still yields a TBD run with both window fields None (requirement 5).
- Unparseable non-blank text yields a friendly obs_date form error, HTTP 200, no CampaignRun, no 500 (requirement 2).
- Requirement 7 investigated and explicitly recorded in the SUMMARY (existing handler covers ranges; only message wording broadened).
- All six required test cases (plus reversed-range and range-duplicate collision) green; ruff clean.
</success_criteria>

<output>
Create `.planning/quick/260714-ilz-close-date-format-gap-on-public-campaign/260714-ilz-SUMMARY.md` when done.
</output>
