---
phase: 20-range-tbd-import-asset-aware-coverage-gap
reviewed: 2026-07-10T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - solsys_code/campaign_gap.py
  - solsys_code/campaign_tables.py
  - solsys_code/campaign_utils.py
  - solsys_code/management/commands/import_campaign_csv.py
  - solsys_code/migrations/0006_campaignrun_original_obs_date_raw_and_window_needs_review.py
  - solsys_code/models.py
  - solsys_code/tests/test_campaign_gap.py
  - solsys_code/tests/test_campaign_models.py
  - solsys_code/tests/test_campaign_views.py
  - solsys_code/tests/test_import_campaign_csv.py
  - src/templates/campaigns/campaignrun_gap_analysis.html
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: issues_found
---

# Phase 20: Code Review Report

**Reviewed:** 2026-07-10
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed the four Phase 20 plans' source changes: asset-aware `claimed_dates()` bucketing
(`campaign_gap.py`), the two new `CampaignRun` review fields + TBD tooltip (`models.py`,
migration `0006`, `campaign_tables.py`), and the range/TBD `parse_obs_window()` rewrite plus
`import_campaign_csv`'s resolved-vs-TBD natural-key branching (`campaign_utils.py`,
`import_campaign_csv.py`). The ASSET-01/ASSET-02 coverage-gap work (Plan 01) is solid — the
ground-vs-space classification is computed once from the `site` parameter exactly as designed,
the PII-minimizing `.only()` queryset is untouched, and the new bucket is well covered by
tests. The TBD-badge tooltip (Plan 02) correctly escapes `original_obs_date_raw` via
`format_html`'s positional-argument auto-escaping and is proven against a hostile-markup test
case.

The range/TBD import path (Plan 03) is where the real defect lives: the new TBD-branch
natural-key `lookup` dict passed to `CampaignRun.objects.get_or_create()` is missing the
`window_start__isnull=True` condition that the model's own partial `UniqueConstraint`
requires to disambiguate a TBD row from a resolved one. Depending on what's already in the
table, this either silently corrupts an unrelated resolved `CampaignRun` row or crashes the
entire import with an uncaught `MultipleObjectsReturned` — precisely the "malformed CSV data
must never abort the whole batch" failure mode this phase's own threat model (T-20-01) set out
to close. No test in `test_import_campaign_csv.py` exercises a TBD row sharing
`(campaign, telescope_instrument, contact_person)` with an existing resolved row, so the
defect is invisible to the current test suite. A secondary gap: `parse_obs_window()`'s new
full-date-range shape never validates `window_start <= window_end`, so a reversed range
(operator typo in the source sheet) silently produces an inverted, unflagged window.

## Critical Issues

### CR-01: TBD-branch natural-key lookup can corrupt or crash on a shared (campaign, telescope_instrument, contact_person) key

**File:** `solsys_code/management/commands/import_campaign_csv.py:196-204`
**Issue:**

For a row that resolves to TBD (`window_start is None`), the `lookup` dict passed to
`insert_or_create_campaign_run()` — and from there to `CampaignRun.objects.get_or_create(**lookup, defaults=fields)`
— is:

```python
lookup = {
    'campaign': campaign,
    'telescope_instrument': telescope_instrument,
    'contact_person': contact_person,
}
```

This omits `window_start__isnull=True`. The model's actual DB-level natural key for a TBD
row (`CampaignRun.Meta.constraints`, `unique_campaign_run_tbd_natural_key`) is
`(campaign, telescope_instrument, contact_person)` **conditioned on**
`window_start__isnull=True` — the Python-level lookup dict must reproduce that condition, but
doesn't.

`Django's `QuerySet.get_or_create()` does `self.get(**kwargs)` first (confirmed against the
installed Django 5.2.15 source — `get_or_create` only catches `self.model.DoesNotExist`
around that call, nothing else). With the condition missing:

- If exactly one existing `CampaignRun` already shares `(campaign, telescope_instrument,
  contact_person)` — including a **resolved** row with a concrete `window_start`/`window_end`
  (e.g. an earlier import of the same telescope+PI's completed run) — `get()` returns that
  resolved row with `created=False`. `insert_or_create_campaign_run()` then diffs `fields`
  (which for the TBD branch includes `original_obs_date_raw` and `window_needs_review=True`
  but never `window_start`/`window_end`) against the existing row and saves the changed
  fields. The result: the pre-existing **resolved** run gets silently overwritten with
  `window_needs_review=True` and an unrelated `original_obs_date_raw` string, while its real
  `window_start`/`window_end` are left untouched — an internally contradictory row (looks
  resolved and TBD-flagged at once), and the new TBD row's own information (a second,
  unscheduled observation request exists) is discarded entirely, not persisted anywhere.
- If **more than one** existing row shares that triple (e.g. two prior resolved runs on the
  same telescope for a PI who is frequently credited with blank `Contact Person`, which is the
  common real-sheet default per `contact_person = row.get('Contact Person', '') or ''`),
  `self.get(**kwargs)` raises `CampaignRun.MultipleObjectsReturned`. Nothing in
  `Command.handle()`'s row loop catches this — it propagates straight out of `handle()`,
  aborting the entire CSV import (every row after the offending one is never processed), which
  is exactly the "one malformed row must never abort the whole batch" failure this phase's own
  D-13 "never raise" contract and threat model (T-20-01, RESEARCH.md Security Domain) exist to
  prevent.

No existing test exercises this: `test_two_tbd_rows_different_contact_both_import` and
`test_two_tbd_rows_same_contact_second_collides` only cover TBD-vs-TBD collisions within the
same batch; nothing imports a resolved row and a later TBD row sharing the same
`(campaign, telescope_instrument, contact_person)` triple.

**Fix:**
```python
else:
    # TBD branch (Pitfall 2): contact_person is promoted into the lookup key
    # instead, so it's deliberately left out of `fields` to avoid
    # lookup/defaults key-overlap ambiguity. `window_start__isnull=True` must be
    # part of the lookup itself -- without it, get_or_create()'s internal
    # `self.get(**lookup)` can match (and silently corrupt) an unrelated
    # *resolved* row sharing the same telescope/contact, or raise an uncaught
    # MultipleObjectsReturned if more than one does.
    lookup = {
        'campaign': campaign,
        'telescope_instrument': telescope_instrument,
        'contact_person': contact_person,
        'window_start__isnull': True,
    }
```
Add a regression test: create a resolved `CampaignRun` for `(campaign, telescope_instrument,
contact_person='')`, then import a CSV row with the same `telescope_instrument`, blank
`Contact Person`, and an unparseable `Obs. Date` — assert a **second** `CampaignRun` is
created (count == 2), the resolved row is untouched, and the command doesn't raise.

## Warnings

### WR-01: Full-date range parsing never validates `window_start <= window_end`

**File:** `solsys_code/campaign_utils.py:241-248`
**Issue:** `_DATE_RANGE_FULL` parses both sides of a `' to '`/en-dash/em-dash/hyphen range
independently via `strptime`, with no check that the parsed end date is not before the start
date. A source-sheet typo like `'2025-09-22 to 2025-07-05'` (operands swapped) parses
successfully into `window_start=2025-09-22, window_end=2025-07-05` — `window_needs_review`
stays `False` (it's treated as an ordinary resolved range), so nothing flags it for staff
review. Downstream, `claimed_dates()`'s `n_days = (run.window_end - run.window_start).days + 1`
goes negative and `range(n_days)` silently iterates zero times, so the run claims **no**
dates at all — a plausible-looking `CampaignRun` that, unbeknownst to anyone, contributes
nothing to coverage and displays as `2025-09-22 -&gt; 2025-07-05` in the table (`render_window_start`,
`campaign_tables.py:159`), which reads as nonsensical rather than as an error. The compact
range (`_DATE_RANGE_COMPACT`) is not affected — its rollover algorithm always produces
`window_end >= window_start` by construction — but the full-date-range shape has no equivalent
guarantee.

**Fix:** After parsing both sides of `_DATE_RANGE_FULL`, treat `window_end < window_start` the
same as any other unparseable shape (fall through to TBD) rather than accepting it silently:
```python
try:
    window_start = datetime.strptime(start_s, '%Y-%m-%d').date()
    window_end = datetime.strptime(end_s, '%Y-%m-%d').date()
    if window_end < window_start:
        window_start = window_end = None
except ValueError:
    window_start = window_end = None
```

### WR-02: `insert_or_create_campaign_run()`'s docstring understates the natural key it's actually given

**File:** `solsys_code/campaign_utils.py:352-356`
**Issue:** The `lookup` parameter's docstring still reads `"the unique lookup key ... (D-04:
campaign, telescope_instrument, window_start)"` — a single-field-window description that
predates this phase. It doesn't mention `window_end`, and doesn't mention that callers must
now pass one of *two* structurally different key shapes (resolved-window vs. TBD/
`contact_person`, per `CampaignRun.Meta.constraints`'s two partial `UniqueConstraint`s). This
function has no assertion or validation of its own that `lookup` matches either real DB
constraint shape — it fully trusts the caller — so an out-of-date docstring here is exactly
the kind of gap that let CR-01's incorrect lookup dict go unnoticed.

**Fix:** Update the docstring to state both natural-key shapes explicitly, e.g.: "keyword
mapping used as the unique lookup key for `get_or_create` — must match one of
`CampaignRun.Meta.constraints`'s two partial `UniqueConstraint`s exactly: either
`(campaign, telescope_instrument, window_start, window_end)` for a resolved window, or
`(campaign, telescope_instrument, contact_person, window_start__isnull=True)` for a TBD row."

## Info

### IN-01: `parse_obs_window()`'s cyclomatic complexity has grown past this project's usual range

**File:** `solsys_code/campaign_utils.py:201-309`
**Issue:** `ruff check --select C901` (not part of this project's enabled rule set, but a
useful signal) flags `parse_obs_window()` at complexity 12 (default threshold 10) after this
phase's additions — it now nests the exact-date attempt, the full-range attempt, and the
compact-range-with-rollover attempt inside one `except ValueError:` block, followed by the
three pre-existing UT-time branches. The function is still readable today, but further Obs.
Date shapes would push it further past a single-responsibility function's usual size for this
codebase (CLAUDE.md: "methods typically 10-50 lines... extract complex logic into helper
functions").
**Fix:** Consider extracting the three date/range-parsing attempts into a small helper (e.g.
`_parse_obs_date_shape(text: str) -> tuple[date | None, date | None]`) returning just
`(window_start, window_end)`, leaving `parse_obs_window()` to own only the TBD/UT-time
branching. Not required for this phase — purely a maintainability suggestion for the next
change to this function.

---

_Reviewed: 2026-07-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
