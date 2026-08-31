---
phase: 02-run-line-parsing
reviewed: 2026-06-13T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - solsys_code/telescope_runs.py
  - solsys_code/tests/test_telescope_runs.py
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-06-13T00:00:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed the additions to `solsys_code/telescope_runs.py` (`KNOWN_STATUSES`,
`ParsedRun`, `parse_run_line()`, `_resolve_telescope`, `_resolve_status`, and
the date-range regexes) and the 10 new test methods in
`solsys_code/tests/test_telescope_runs.py`, against the locked design
decisions D-01..D-07 in `02-CONTEXT.md`.

The three acceptance fixtures from the design doc all behave as documented
(NTT/EFOSC2 success path, both ambiguous-Magellan error paths, hyphenated
"Proto-Lightspeed" instrument). However, two correctness gaps were found by
constructing inputs slightly outside the test suite's coverage: (1) text
trailing the matched date range is silently discarded instead of being
validated as D-06 requires, and (2) a cross-month date range (e.g. "28
July-2 August") silently drops the end month, producing a `ParsedRun` whose
`day2` does not actually belong to `month`. Both are correctness bugs that
will produce wrong `CalendarEvent` dates downstream in Phase 3 without
raising any error.

## Critical Issues

### CR-01: Trailing text after the date range is silently dropped, not validated

**File:** `solsys_code/telescope_runs.py:407-418`
**Issue:**
`leftover` is computed only from `before_range = remainder[: match.start()]`
(tokens between the instrument and the date range). Any text appearing
*after* the matched date range in `remainder` is never inspected. Per D-06,
"any remaining word(s) ... must be in KNOWN_STATUSES" — but this is only
enforced for words *before* the date range, not after.

Concretely:
```python
parse_run_line('NTT EFOSC2 9-13 July (confirmed) extra')
# -> ParsedRun(telescope='NTT', instrument='EFOSC2', status='confirmed',
#               year=2026, month=7, day1=9, day2=13)
```
The trailing `extra` token is silently swallowed instead of raising
`ValueError`. This means malformed or partially-garbled lines (e.g. OCR
artifacts, accidental duplicate words, a second instrument/status that
wasn't recognized) pass through parsing successfully with data quietly
discarded — directly contradicting D-07's intent that "anything unparseable
... raises ValueError with a message describing what failed."

**Fix:**
```python
before_range = remainder[: match.start()]
after_range = remainder[match.end() :].strip()
tokens = before_range.split()
if len(tokens) < 2:
    raise ValueError(f'Could not find telescope and instrument tokens in {line!r}')
telescope_token, instrument = tokens[0], tokens[1]

leftover = ' '.join(tokens[2:]).strip()
if after_range:
    leftover = (leftover + ' ' + after_range).strip() if leftover else after_range
if leftover:
    raise ValueError(f'Unrecognized status {leftover!r} in {line!r}; known statuses are {sorted(KNOWN_STATUSES)}')
```

---

### CR-02: Cross-month date ranges silently drop the end month, producing an internally inconsistent `ParsedRun`

**File:** `solsys_code/telescope_runs.py:81-90, 387-391`
**Issue:**
`_CROSS_MONTH_RANGE` captures both `month1` and `month2` (e.g. "28
July-2 August" -> `month1='July'`, `day1=28`, `day2=2`, `month2='August'`).
But `parse_run_line()` only ever reads `match.group('month1')` and assigns it
to `ParsedRun.month`; `month2` is parsed by the regex but then discarded.

The only cross-month case that happens to "work" is the December->January
roll-over (`month==12 and day2 < day1` -> `year += 1`), because
`_MONTH_NAMES['december'] == 12` is the only month value where the
single-`month` field combined with the year-rollover heuristic still yields
a usable (month, day1)/(12, day2-next-year-January-is-implied) pair *for that
specific test's assertions* — but even there, `ParsedRun.month=12` and
`day2=2` together describe "December 2", not "January 2", which is simply
wrong if `month` is read at face value by Phase 3.

For any *other* cross-month range, e.g.:
```python
parse_run_line('NTT EFOSC2 28 July-2 August')
# -> ParsedRun(telescope='NTT', instrument='EFOSC2', status='allocation',
#               year=2026, month=7, day1=28, day2=2)
```
This claims the run spans "July 28" to "July 2" (day2 < day1, same month,
no rollover applied since month != 12) — both dates are nonsensical/reversed
within the reported month. Phase 3, which is documented to consume
`ParsedRun.month`/`day1`/`day2` directly to build `CalendarEvent` date
ranges, will compute a backwards or invalid date range with no error raised
at any point.

**Fix:** Either (a) add a `month2`/`day2_month` field to `ParsedRun` (a
`ParsedRun` schema change, but D-03 only says "any named-field structure
satisfying D-03 is acceptable" — an additive field doesn't violate that), and
populate it from `match.group('month2')` when `_CROSS_MONTH_RANGE` matches
(falling back to `month1`'s value for non-cross-month matches); or (b) if a
single-month field is truly intentional for Stage 2, raise `ValueError` for
any cross-month match where `month1 != month2` is meaningful (i.e. not the
Dec->Jan special case), so this gap surfaces explicitly rather than producing
silently-wrong dates. Given D-07's emphasis on raising rather than silently
mis-parsing, option (b) is the minimal safe fix; option (a) is more complete
for Phase 3's needs.

## Warnings

### WR-01: No validation that day2 >= day1 within a single month (non-December)

**File:** `solsys_code/telescope_runs.py:400-404`
**Issue:** The year-rollover check only fires for `month == 12 and day2 <
day1`. For any other month, a line like `'NTT EFOSC2 13-9 July'` parses
successfully to `ParsedRun(month=7, day1=13, day2=9, year=2026)` — a
backwards date range — with no error. While D-07 leaves exact validation
scope to discretion, a backwards range within the same month is clearly
malformed input and silently producing it defeats the "raise on anything
unparseable" intent.

**Fix:** After computing `day1`/`day2`/`month`, add:
```python
if month != 12 and day2 < day1:
    raise ValueError(f'Date range day2 ({day2}) precedes day1 ({day1}) in {line!r}')
```
(or more generally, validate using `calendar.monthrange` to also catch
invalid day-of-month values like day=31 for a 30-day month).

---

### WR-02: `_resolve_status`'s word-boundary regex can match a status word embedded inside an unrelated capitalized token

**File:** `solsys_code/telescope_runs.py:343-347`
**Issue:** The regex `(?<!\S){re.escape(status)}(?!\S)` uses whitespace
boundaries, which is correct for separating `'Proposed-Cam'` (no match,
verified) from a bare `'proposed'` token. However, combined with CR-01, if a
genuine status word appears as the *entire* instrument token (e.g. an
instrument literally named `"Confirmed"` — admittedly unlikely but not
excluded by any earlier validation), `_resolve_status` will consume it as the
status before `_resolve_telescope`/instrument-token logic ever sees it,
silently reinterpreting the instrument as a status. This is a low-probability
edge case but stems from the same root issue as CR-01/WR-01: there's no
cross-check that the matched status token is positioned *after* the
instrument token (between instrument and date-range, or in parens after the
date range) rather than potentially being a token that should be the
instrument itself.

**Fix:** Lower priority given how unlikely real instrument names matching
`KNOWN_STATUSES` are; document this assumption in `_resolve_status`'s
docstring (e.g. "Assumes instrument names do not collide with
KNOWN_STATUSES entries") so it's a documented constraint rather than a silent
assumption.

---

### WR-03: `_resolve_telescope` prefix-match against `SITES` is case-sensitive but `_MONTH_NAME_PATTERN`/status matching is case-insensitive — inconsistent case-handling could surprise callers

**File:** `solsys_code/telescope_runs.py:292-314` vs `317-349`, `58-90`
**Issue:** Month names and statuses are matched with `re.IGNORECASE` /
`.lower()`, but `_resolve_telescope` does an exact/prefix `str.startswith`
comparison against `SITES` keys with no case-folding — `'ntt'` (lowercase)
raises `Unknown telescope 'ntt'` even though `'NTT'` is a valid SITES key
(verified). This is plausibly intentional (telescope names are likely always
written in the canonical case in source schedules), but it's an inconsistency
worth calling out: a user typo in case for the telescope token produces a
generic "unknown telescope" error rather than a case-insensitive resolution,
while the same typo in a month name or status word would be silently
tolerated.

**Fix:** If case-insensitive telescope matching is desired, normalize both
`token` and `SITES` keys to a common case before comparison (taking care that
`SITES` keys like `'Magellan-Clay'` vs `'Magellan-Baade'` still need their
original case preserved in the *return value*, only the comparison should be
case-folded). If intentional, consider a one-line docstring note in
`_resolve_telescope` clarifying that telescope tokens are matched
case-sensitively, since this differs from the other two matching strategies
in the same function call chain.

## Info

### IN-01: `_resolve_status`'s parenthesized-status branch does not anchor to end-of-line, so a stray `(...)` anywhere in the line is treated as the status

**File:** `solsys_code/telescope_runs.py:332-340`
**Issue:** `_PAREN_STATUS = re.compile(r'\(([^)]+)\)')` and
`_resolve_status` takes the *first* parenthesized group anywhere in the line
via `.search()`. If a future run line legitimately contained a parenthesized
aside that isn't a status (e.g. `"NTT EFOSC2 (replaces IFOSC) 9-13 July
(confirmed)"`), the first paren group `(replaces IFOSC)` would be checked
against `KNOWN_STATUSES`, fail, and raise `ValueError` for `'replaces ifosc'`
— even though a valid status `(confirmed)` is also present later in the
line. Not a bug against any current fixture, but worth noting as a latent
fragility if line formats evolve in Phase 3+.

**Fix:** No action needed for Phase 2 fixtures; if Phase 3 encounters lines
with non-status parenthetical asides, consider restricting `_PAREN_STATUS` to
match only near the end of the line, or iterating all paren groups and
picking the one matching `KNOWN_STATUSES`.

---

### IN-02: `ParsedRun.year` defaults silently to `date.today().year` with no way to override for historical/future schedules

**File:** `solsys_code/telescope_runs.py:402`
**Issue:** `year = date_cls.today().year` ties parsing output to wall-clock
time at call time. This matches PARSE-03 as documented and is presumably
intentional, but it means `parse_run_line()` is not a pure function of its
input string alone — the same line parsed on different days (crossing a
year boundary) yields different `ParsedRun.year` values. This is fine for
the documented use case but worth a one-line note in the module/function
docstring flagging the non-determinism for anyone writing snapshot tests
against `parse_run_line()` output in Phase 3.

**Fix:** Documentation-only; no code change required. Consider adding "Note:
year is derived from the current date and is therefore not pure/deterministic
across calendar-year boundaries" to `parse_run_line()`'s docstring.

---

_Reviewed: 2026-06-13T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
