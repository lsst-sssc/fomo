---
phase: quick-260718-dih
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/campaign_views.py
  - solsys_code/telescope_runs.py
  - solsys_code/management/commands/load_telescope_runs.py
  - solsys_code/tests/test_campaign_approval.py
  - solsys_code/tests/test_telescope_runs.py
  - solsys_code/tests/test_load_telescope_runs.py
  - .planning/Findings.md
autonomous: true
requirements:
  - PR-REVIEW-F1   # High: unguarded calendar write after run_status commit
  - PR-REVIEW-F2   # Medium: cross-month parser/loader contract mismatch
  - PR-REVIEW-F3   # Low/Medium: permissive partial-night token regex

must_haves:
  truths:
    - Marking an APPROVED run cancelled/weathered never returns an uncaught 500 when the calendar sync raises; run_status is already committed and a warning tells the user to retry the same action.
    - parse_run_line rejects a genuine cross-month range (e.g. '28 December-2 January') at parse time with a clear ValueError, instead of returning a ParsedRun its only caller always rejects.
    - A partial-night token with surrounding garbage (e.g. 'xBoN-0626') is rejected with ValueError, not silently accepted by substring match.
    - Findings.md records each original finding plus a plain-English resolution note with corrected post-fix line numbers.
  artifacts:
    - solsys_code/campaign_views.py
    - solsys_code/telescope_runs.py
    - solsys_code/management/commands/load_telescope_runs.py
    - solsys_code/tests/test_campaign_approval.py
    - solsys_code/tests/test_telescope_runs.py
    - solsys_code/tests/test_load_telescope_runs.py
    - .planning/Findings.md
  key_links:
    - try/except wrapping the calendar-sync loop in _set_run_status (mirrors _resolve_site's non-reverting projection guard).
    - parse_run_line's cross-month branch raising ValueError before building a ParsedRun.
    - _PARTIAL_NIGHTS call site using fullmatch (whole-token match) instead of search.
---

<objective>
Fix three confirmed PR-review findings on branch issue37-telescope-runs-calendar, close the
three coverage gaps the review lists, and correct the stale line-number citations in
.planning/Findings.md.

Purpose: harden already-shipped code — guard a partial-write path (run_status committed, then
an unguarded calendar-sync loop that can 500), turn a fail-late parser/loader contract mismatch
into a fail-fast parse-time rejection, and reject malformed partial-night tokens instead of
silently stripping surrounding garbage.

Output: guarded calendar sync in _set_run_status; cross-month fail-fast in parse_run_line;
anchored partial-night token match; updated loader docstring; new coverage tests in three test
modules; and a corrected, resolution-annotated Findings.md.

Scope note (CLAUDE.md paired-notebook rule): the two touched behavior modules
(telescope_runs.py, load_telescope_runs.py) have paired demo notebooks
(docs/notebooks/pre_executed/telescope_runs_demo.ipynb and load_telescope_runs_demo.ipynb).
Confirmed during planning that NO notebook cell exercises a cross-month range or a malformed
partial-night token — the only partial-night token in either notebook is the well-formed
'BoN-0626', which parses identically under fullmatch, and '18-20 July' is a same-month range
unaffected by the cross-month rejection. These fixes only add fail-fast rejection of
already-broken input and a try/except around an existing call; they do not change external
behavior for well-formed input. The paired-notebook rule therefore does not apply, and the
notebooks are deliberately excluded from files_modified. Do NOT touch
docs/notebooks/pre_executed/*.ipynb.

Do NOT touch solsys_code/tests/test_import_campaign_csv.py — its lines 381/390 exercise the
CSV importer's parse_obs_window() (a different parser with its own D-11 rollover rules), not
telescope_runs.py's parse_run_line.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/Findings.md
@./CLAUDE.md

@solsys_code/campaign_views.py
@solsys_code/telescope_runs.py
@solsys_code/management/commands/load_telescope_runs.py
@solsys_code/tests/test_campaign_approval.py
@solsys_code/tests/test_telescope_runs.py
@solsys_code/tests/test_load_telescope_runs.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Guard the calendar-sync loop in _set_run_status + Finding-1 coverage test</name>
  <files>solsys_code/campaign_views.py, solsys_code/tests/test_campaign_approval.py</files>
  <action>
In CampaignRunDecisionView._set_run_status (currently starts at campaign_views.py:740), wrap ONLY
the `if matching_events.exists(): ... insert_or_create_calendar_event(...)` block (currently lines
791-804) in a try/except Exception that mirrors the sibling _resolve_site method's non-reverting
projection guard (currently campaign_views.py:720-729). Do NOT touch the run_status conditional
.update() (currently line 769-771) or its updated_count==0 short-circuit — that write is already
correctly guarded (T-23-04) and its commit is the intended, non-reverted behavior.

On exception inside the new except block: call logger.exception with a message that identifies the
run pk (module-level logger already exists at campaign_views.py:54); then call messages.warning with
wording that makes clear the run_status change WAS saved but the calendar entry could not be updated,
and that the user should retry the SAME Mark Cancelled / Mark Weathered action (which is idempotent —
the conditional .update() will simply re-match the already-set status and the sync loop will re-run).
Deliberately differ from _resolve_site's wording: do not imply the status change can be retried from
scratch or that the run reverted — only the calendar sync needs retrying. Then redirect to
'campaigns:approval_queue', the same target as the success path.

Update the method docstring's calendar-sync paragraph (currently campaign_views.py:752-758) to note
the sync loop is now wrapped in a non-reverting try/except so a sync failure leaves run_status
committed and surfaces a warning instead of a 500.

Add a coverage test to the TestRunStatusChange class in test_campaign_approval.py (the class already
has _make_approved_single_night_run at line 479, which creates+approves a resolved-site run so a
CAMPAIGN:{pk} CalendarEvent exists, and setUp logs in staffcoordinator). The test: build an approved
single-night run via _make_approved_single_night_run(), assert its CAMPAIGN:{pk} event exists, then
patch('solsys_code.campaign_views.insert_or_create_calendar_event', side_effect=Exception(...)) ONLY
around a POST of {'action': 'mark_cancelled'} to reverse('campaigns:decide', kwargs={'pk': run.pk})
with follow=True. Assert: (a) response.status_code == 200 (the redirect was followed, not a 500);
(b) after run.refresh_from_db(), run.run_status == CampaignRun.RunStatus.CANCELLED (the status write
survived the sync failure); (c) a warning message is present, using the established idiom
`messages_list = [str(m) for m in response.context['messages']]` and asserting one mentions the
calendar sync failure. Use patch/imports already present in this test module (patch is imported;
CampaignRun, CalendarEvent, reverse, date are in scope).
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_campaign_approval 2>&1 | tail -20</automated>
  </verify>
  <done>_set_run_status wraps only the calendar-sync loop in a non-reverting try/except (logger.exception + messages.warning + redirect to approval_queue); the run_status .update() is unchanged; the new TestRunStatusChange test proves a raising sync yields a redirect (not a 500), a committed CANCELLED run_status, and a warning message; the full test_campaign_approval module passes.</done>
</task>

<task type="auto">
  <name>Task 2: Fail-fast cross-month rejection + anchored partial-night token + parser/loader tests</name>
  <files>solsys_code/telescope_runs.py, solsys_code/management/commands/load_telescope_runs.py, solsys_code/tests/test_telescope_runs.py, solsys_code/tests/test_load_telescope_runs.py</files>
  <action>
In telescope_runs.py parse_run_line (date-range block currently lines 429-448): keep the match-order
precedence unchanged — _MONTH_AFTER_RANGE is still tried first (line 431). In the cross-month branch
(currently lines 436-441), when _CROSS_MONTH_RANGE matches, raise ValueError immediately with a
message naming the unsupported cross-month range and including the offending line via {line!r}
(e.g. 'Cross-month run ranges not yet supported: ...'), INSTEAD of building day1/day2/month from it.
Restructure so that when _CROSS_MONTH_RANGE does not match, _MONTH_BEFORE_RANGE is tried next exactly
as today (raising the existing 'Could not find a date range ...' ValueError when it also fails), and
the later `before_range = remainder[: match.start()]` / `after_range = remainder[match.end():]` logic
still binds `match` to the _MONTH_BEFORE_RANGE (or _MONTH_AFTER_RANGE) result.

Remove the now-effectively-dead year-rollover block (currently lines 450-454:
`if month == 12 and day2 < day1: year += 1`), replacing it with a plain
`year = date_cls.today().year`. Rationale to record in the SUMMARY: the only case this rollover ever
served was the genuine December-to-January cross-month range, which now fails fast above; any
remaining descending same-month range that could still reach this line (a typo like '20-5 December'
via _MONTH_BEFORE_RANGE / _MONTH_AFTER_RANGE) is always rejected downstream by _iter_run_nights, so
the year value it would have bumped is never observable in a successful ingest. Update the inline
date-range comment (currently lines 429-430) and the docstring Raises section (currently lines
417-421) to state that a cross-month range now raises.

Switch the partial-night token match from search to fullmatch: change
`_PARTIAL_NIGHTS.search(window_tokens[0])` (currently line 478) to
`_PARTIAL_NIGHTS.fullmatch(window_tokens[0])`. This is the smaller, clearer diff than adding anchors;
leave the _PARTIAL_NIGHTS regex definition (lines 106-111) unchanged. Because window_tokens[0] is a
single whitespace-delimited token, fullmatch rejects any token with embedded prefix/suffix garbage
(e.g. 'xBoN-0626') while still accepting exact well-formed tokens ('BoN-0626', '0646-EoN').

In load_telescope_runs.py _iter_run_nights: KEEP the `if parsed.day2 < parsed.day1` guard (currently
lines 76-77) — it is still load-bearing for descending same-month day ranges (e.g. a typo'd
'20-5 July' that _MONTH_BEFORE_RANGE / _MONTH_AFTER_RANGE parse into day1 > day2), which
parse_run_line does NOT reject upstream. Update the docstring Raises wording (currently lines 71-77)
and the ValueError message string so they describe the real remaining reason — a descending or
malformed same-month day range where day2 < day1 — rather than 'cross-month ranges not yet supported
in Phase 3', since genuine cross-month ranges now fail fast in parse_run_line.

Tests in test_telescope_runs.py: rewrite the existing test_parse_run_line_december_january_rolls_over_year
(currently lines 361-368) so it asserts parse_run_line('NTT EFOSC2 28 December-2 January') raises
ValueError (use assertRaises); rename it to reflect the new contract (e.g.
test_parse_run_line_cross_month_range_raises) and update its docstring. Add a new test asserting a
partial-night token with surrounding garbage raises ValueError, e.g. parse_run_line for a
well-formed run line whose trailing window token is 'xBoN-0626' (garbage prefix around an otherwise
valid BoN-HHMM pair) raises ValueError — this proves the fullmatch anchoring. Confirm the existing
partial-night tests (test_parse_run_line_magellan_first_half / _second_half /
_second_half_bare_time / _missing_EoN / _wrong_EoN, currently lines 290-336) still pass unmodified.

Tests in test_load_telescope_runs.py: add a command-level test modeled on
test_unparseable_line_logged_and_skipped (currently line 292) that writes a schedule file containing
a genuine cross-month line ('NTT EFOSC2 28 December-2 January'; the 809/NTT Observatory fixture
already exists in setUpTestData) alongside a valid line, runs call_command('load_telescope_runs', ...)
capturing stderr, and asserts the cross-month line is skipped and logged (its line number and text
appear in stderr) while the valid line still produces its CalendarEvents — i.e. a skip, not a crash.
  </action>
  <verify>
    <automated>./manage.py test solsys_code.tests.test_telescope_runs solsys_code.tests.test_load_telescope_runs 2>&1 | tail -25</automated>
  </verify>
  <done>parse_run_line raises ValueError for a genuine cross-month range and the dead year-rollover block is removed; _PARTIAL_NIGHTS is matched via fullmatch so garbage-wrapped tokens are rejected; _iter_run_nights keeps its guard with docstring/message reworded for descending same-month ranges; the rewritten cross-month test, the new garbage-token test, and the new command-level cross-month skip test all pass; existing partial-night tests pass unmodified.</done>
</task>

<task type="auto">
  <name>Task 3: Correct Findings.md line numbers + resolution notes; final ruff + full affected-suite run</name>
  <files>.planning/Findings.md</files>
  <action>
After Tasks 1-2 land, re-grep the FINAL post-fix line numbers (do not reuse the pre-fix numbers) in
solsys_code/campaign_views.py (the _set_run_status method start, the run_status .update() call, and
the guarded insert_or_create_calendar_event call), solsys_code/telescope_runs.py (the _CROSS_MONTH_RANGE
raise in parse_run_line, the _PARTIAL_NIGHTS definition and its fullmatch call site), and
solsys_code/management/commands/load_telescope_runs.py (the _iter_run_nights day2 < day1 guard).

Edit .planning/Findings.md to keep the three original findings and the Coverage gaps section AS the
historical record (do not delete them — this file is a project record of what was found and fixed),
and append a short plain-English 'Resolved:' note to each finding and each coverage-gap item stating
what changed and citing the corrected post-fix line numbers. Specifically: correct finding #1's stale
citations (campaign_views.py:736 / :752 / :754) to the real post-fix locations, and note the calendar
sync loop is now wrapped in a non-reverting try/except so a sync failure leaves the status committed
and surfaces a warning instead of a 500. For finding #2, soften the framing: note the cross-month
rejection was a documented, intentional Phase-3 deferral, and that this task changed it from a
fail-late design (parse succeeds, ingest rejects) to a fail-fast one (parse rejects immediately), so
the original 'contract mismatch' is now resolved rather than open. For finding #3, note the token
match is now anchored via fullmatch so garbage-wrapped tokens are rejected. For the three coverage-gap
items, note each is now closed by the corresponding new test. Write plain English throughout per
CLAUDE.md's planning-doc terminology convention — no DB jargon such as 'upsert' (say 'create or update'
/ 'find-or-create' if needed).

Then run the repo's quality gates and the full affected suite (see verify).
  </action>
  <verify>
    <automated>ruff check . --fix && ruff format . && ruff check . && ruff format --check . && ./manage.py test solsys_code.tests.test_campaign_approval solsys_code.tests.test_telescope_runs solsys_code.tests.test_load_telescope_runs 2>&1 | tail -25</automated>
  </verify>
  <done>Findings.md retains the three original findings and coverage-gap list, each annotated with a plain-English 'Resolved:' note carrying corrected post-fix line numbers; finding #2's framing is softened to a resolved fail-late→fail-fast change; ruff check and ruff format --check are clean; all three affected test modules pass.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| staff browser → CampaignRunDecisionView (decide endpoint) | Mark-status POST triggers a DB write plus a calendar-sync side effect. |
| schedule file → load_telescope_runs / parse_run_line | Untrusted free-text run lines cross into parse + ingest. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-DIH-01 | Denial of Service | _set_run_status calendar-sync loop | medium | mitigate | Wrap the sync loop in a non-reverting try/except so a sync exception becomes a warning + redirect, not an uncaught 500; run_status stays committed and the idempotent action can be retried (Task 1). |
| T-DIH-02 | Tampering | parse_run_line cross-month range | low | mitigate | Fail fast at parse time on a genuine cross-month range instead of emitting a ParsedRun the loader always rejects, removing the fail-late contract mismatch (Task 2). |
| T-DIH-03 | Tampering | _PARTIAL_NIGHTS partial-night token | low | mitigate | Match the whole token with fullmatch so a garbage-wrapped token is rejected rather than silently stripped to a plausible-but-wrong window (Task 2). |

No package-manager installs are introduced by this task; no package legitimacy gate applies.
</threat_model>

<verification>
- ./manage.py test solsys_code.tests.test_campaign_approval passes (Finding 1 + coverage test).
- ./manage.py test solsys_code.tests.test_telescope_runs passes (cross-month raise, garbage-token reject, existing partial-night tests unmodified).
- ./manage.py test solsys_code.tests.test_load_telescope_runs passes (command-level cross-month skip test).
- ruff check . and ruff format --check . are clean.
- .planning/Findings.md carries corrected post-fix line numbers and plain-English resolution notes.
- docs/notebooks/pre_executed/*.ipynb are untouched (paired-notebook rule confirmed not to apply).
</verification>

<success_criteria>
- Marking an APPROVED run cancelled/weathered when the calendar sync raises returns a redirect with a warning, not a 500, and leaves run_status committed.
- A genuine cross-month range raises ValueError at parse time; the dead year-rollover block is removed; the loader's day2 < day1 guard remains for descending same-month ranges with reworded docstring/message.
- A partial-night token with surrounding garbage is rejected via fullmatch; well-formed tokens still parse.
- Findings.md preserves the original findings with appended plain-English resolution notes and corrected post-fix line numbers.
- ruff clean; all three affected test modules pass.
</success_criteria>

<output>
Create `.planning/quick/260718-dih-fix-pr-review-findings-unguarded-calenda/260718-dih-SUMMARY.md` when done.
Record in the SUMMARY: the decision to remove the year-rollover block (with rationale), the reworded
_iter_run_nights guard reason, and the final corrected line numbers written into Findings.md.
</output>
