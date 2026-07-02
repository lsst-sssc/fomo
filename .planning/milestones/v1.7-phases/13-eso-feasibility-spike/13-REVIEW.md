---
phase: 13-eso-feasibility-spike
reviewed: 2026-07-02T08:53:04Z
depth: deep
files_reviewed: 1
files_reviewed_list:
  - docs/design/eso_feasibility_spike.rst
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-07-02T08:53:04Z
**Depth:** deep
**Files Reviewed:** 1
**Status:** issues_found

## Summary

Phase 13 is investigation-only; the sole non-`.planning/` artifact is
`docs/design/eso_feasibility_spike.rst`, a durable summary of the ESO/VLT
feasibility spike. No application code was written this phase, so the usual
bug/security/logic-error checks don't apply. Review focused on RST
correctness and factual/internal consistency instead:

- **RST syntax**: Parsed cleanly with `docutils.core.publish_doctree` /
  `publish_string` (both `pseudoxml` and `html5` writers) — no docutils
  warnings or errors, title/section underlines all correctly sized, both
  `list-table` directives have matching column counts and header rows.
- **Cross-file factual accuracy**: Every concrete technical claim in the
  document was checked against the installed packages and sibling planning
  docs rather than taken on faith:
  - `tom_eso==0.2.4` is the actually-installed version (`pip show tom_eso`).
  - `ESOAPI.__init__` really does construct `p1api.ApiConnection` before
    `p2api.ApiConnection` (confirmed in installed `tom_eso/eso_api.py`).
  - `p1api.p1api.API_URL` really has no `production_lasilla` key (only
    `production`/`demo`); `p2api.p2api.API_URL` really does define
    `production_lasilla` (confirmed via live `import p1api, p2api`).
  - `submit_observation()`, `get_observation_status()`,
    `get_observation_url()`, `data_products()` all behave exactly as
    claimed (empty-list return / bare `raise NotImplementedError`,
    confirmed in installed `tom_eso/eso.py`).
  - `getOB`, `getOBExecutions`, `getNightExecutions`, `getRuns` are all real
    `p2api.p2api` methods.
  - `insert_or_create_calendar_event()`, the `GEM:{prog}/{observation_id}`
    key, and `_FAILURE_PREFIX_BY_STATUS` all exist exactly as described in
    `solsys_code/calendar_utils.py` / `sync_gemini_observation_calendar.py`
    / `sync_lco_observation_calendar.py`.
  - `FACILITIES` in `src/fomo/settings.py` really does have `LCO`/`SOAR`/
    `GEM` entries and no `ESO` entry, matching the "mirrors LCO/SOAR/GEM"
    and "future `FACILITIES['ESO']`" claims.
  - The obStatus vocabulary table (12 codes) matches the corresponding
    table in `13-DECISION.md` verbatim, and the "Stage 1/2/3/3b/4" language
    matches `PROJECT.md`'s established Stage/Phase terminology exactly.
  - Investigation-summary claims (La Silla wrapper-bug root cause, the
    `production_lasilla` bypass result, the `'P'`/`'M'` real OB status
    values) all match `13-DECISION.md`'s ESO-01/ESO-02 findings with no
    contradictions.

No factual errors or internal inconsistencies were found. Two doc-hygiene
issues were found, both about the document's long-term durability/
discoverability rather than its content's correctness — worth fixing before
this becomes stale, but not blockers to merging investigation findings.

## Warnings

### WR-01: Document is not included in any toctree — orphaned from the docs site

**File:** `docs/design/eso_feasibility_spike.rst` (whole file); missing entry in `docs/design/design.rst:39-44`
**Issue:** Actually running the project's own Sphinx build (`sphinx-build -M html ./docs ... -D exclude_patterns=notebooks/*,_build`, the same invocation the pre-commit `sphinx-build` hook uses) produces:

```
checking consistency... /home/tlister/git/fomo_devel/docs/design/eso_feasibility_spike.rst: WARNING: document isn't included in any toctree
```

`docs/design/design.rst` has a `Design Notes` toctree that lists the other
three docs in `docs/design/`:

```rst
.. toctree::
   :maxdepth: 1

   telescope_runs_calendar
   tom_calendar_vs_yse_pz_calendar
   gsd_experiment
```

`eso_feasibility_spike` is missing from that list. The page is still built
(reachable by a direct URL) but is unreachable via the docs site's navigation
— it is the only file in `docs/design/` that isn't linked from anywhere,
breaking the pattern every other doc in this directory follows. This doesn't
fail the pre-commit hook today (the hook doesn't pass `-W`/`--fail-on-warning`,
so the warning is silent unless someone reads the build log), but it
undermines the doc's own stated purpose of being a durable, findable decision
record.
**Fix:**
```rst
.. toctree::
   :maxdepth: 1

   telescope_runs_calendar
   tom_calendar_vs_yse_pz_calendar
   gsd_experiment
   eso_feasibility_spike
```

### WR-02: Load-bearing cross-reference points into `.planning/phases/`, which this project routinely archives

**File:** `docs/design/eso_feasibility_spike.rst:9-11`, `docs/design/eso_feasibility_spike.rst:153-154`
**Issue:** The document's opening paragraph and its closing "Future scope"
section both cite `.planning/phases/13-eso-feasibility-spike/13-DECISION.md`
as "its full-detail companion" / where "the full recommendation rationale and
future-sync sketch" live — i.e. this path is explicitly called out as part
of the phase's deliverable, not an incidental aside.

That path is not durable. `.planning/phases-archive/` already contains
phases 01 through 09 (`ls .planning/phases-archive/` shows
`01-site-ephemeris-helper` … `09-proposal-color-status-visual-treatment`),
confirming this project's `gsd-cleanup` workflow routinely moves completed
phase directories out of `.planning/phases/` once their milestone is
archived. When that happens to phase 13, the cited path
`.planning/phases/13-eso-feasibility-spike/13-DECISION.md` will 404/no
longer exist at that location (it will move under `.planning/phases-archive/…`
instead), silently breaking the one concrete pointer this "durable" `docs/`
artifact gives readers to the full-detail rationale. No other file in
`docs/design/` references a `.planning/` path at all (checked via
`grep -rln '\.planning/phases/' docs/design/*.rst`) — this document is
uniquely exposed to this failure mode, and it's the kind of staleness that
is easy to miss because nothing (no linter, no `sphinx-build` warning) flags
a plain literal string that happens to look like a path.
**Fix:** Either (a) inline the durable parts of ESO-04/ESO-05 that matter
long-term directly into this `.rst` file so it doesn't depend on the
planning tree surviving archival, or (b) add a note here (and/or update this
sentence when phase 13 is archived) pointing at wherever
`13-DECISION.md` lands post-archive, e.g.
`.planning/phases-archive/13-eso-feasibility-spike/13-DECISION.md`. At
minimum, flag this doc for a follow-up pass when the v1.7 milestone is
archived so the reference doesn't quietly go stale.

## Info

### IN-01: Interleaved bold/literal markup in "Key finding" is hard to scan

**File:** `docs/design/eso_feasibility_spike.rst:41-44`
**Issue:** The opening sentence of "Key finding" alternates `**bold**` and
` ``literal`` ` spans mid-clause to keep specific terms monospaced:

```rst
**Bypass — sync straight from the ESO P2 API (** ``p2api`` **) to**
``CalendarEvent``, **skipping** ``ObservationRecord`` **for ESO entirely.**
```

This parses and renders correctly (verified via docutils `pseudoxml`/`html5`
output — no warnings, spans nest as intended), so it's not a bug, but the
raw source is unusually fragmented and easy to mis-edit (an unbalanced
`**`/` `` ` pair here would silently swallow the rest of the paragraph as
literal/emphasis text rather than raising an error). A simpler phrasing
would be more maintainable.
**Fix:** Simplify to a single bold lead-in followed by plain text with
inline literals only where needed, e.g.:
```rst
**Bypass:** sync straight from the ESO P2 API (``p2api``) to
``CalendarEvent``, skipping ``ObservationRecord`` for ESO entirely.
```

---

_Reviewed: 2026-07-02T08:53:04Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
