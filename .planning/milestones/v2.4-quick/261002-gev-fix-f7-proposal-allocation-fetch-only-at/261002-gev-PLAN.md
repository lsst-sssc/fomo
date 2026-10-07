---
phase: 261002-gev
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - solsys_code/proposal_allocation.py
  - solsys_code/unattended.py
  - solsys_code/tests/test_proposal_allocation.py
  - solsys_code/tests/test_unattended.py
  - docs/runbooks/telescope_runs_calendar.rst
  - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb
  - CLAUDE.md
  - .planning/v2.4-INTENT-REVIEW.md
autonomous: true
requirements: [TALLY-01, SCHED-08, SCHED-09, SCHED-10]

estimate:
  tokens: 100000
  raw_tokens: 100000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "A proposal code is sent to the LCO Observation Portal only when it can be a portal proposal. That means every active `WatchedProposal` code, plus a `CampaignRun.proposal_code` whose run has `source` `lco_queue`/`soar_queue` or whose resolved `site.obscode` is one of the LCO/SOAR observatories named in `campaign_attribution`'s two alias tables. An ESO code on a `classical_file` run at La Silla (809) is never requested."
    - "`refresh_all()` returns `(attempted, rows_written, failed, first_exception, not_fetchable)`. `not_fetchable` counts the distinct run codes that are never fetched. A code that is also an active watched proposal, or that also sits on a fetchable run, is fetched and is not counted."
    - "The `step_proposal_allocation()` summary always reads `proposals: N, rows written: N, failed: N, not fetchable: N`, followed by `, first error: <ClassName>` only on a real failure. `failed` is `failed > 0`, so an exclusion alone leaves the step ok. A real portal failure still fails the step when an exclusion is present."
    - "An excluded code has no stored `ProposalTimeAllocation` row after a refresh. `unused_hours_for()` and `estimated_unused_nights()` return `None` for it, so the tally shows 'not yet known', never zero (D-07 / TALLY-01)."
    - "Choosing codes still loads no `CampaignRun` row and no contact column: both selection functions use `values_list` only (T-37-06), and a `CaptureQueriesContext` test pins this."
    - "The runbook's unattended section and its 'not yet known' troubleshooting entry say which codes are fetched, and that a non-LCO code reads `not fetchable` and is not an error. Every sample `proposal_allocation` log line carries `not fetchable:`. CLAUDE.md's paired-docs map gains a `proposal_allocation.py` entry. The load_telescope_runs notebook's proposal-token prose no longer says an NTT/ESO token is looked up at the LCO portal."
  artifacts:
    - "solsys_code/proposal_allocation.py: imports `Q` and imports `LCO_SITE_CODE_TO_OBSCODE`/`OBSERVED_TELESCOPE_OBSCODES` from `solsys_code.campaign_attribution`. Adds the `_PORTAL_RUN_SOURCES`/`_PORTAL_SITE_OBSCODES` constants, a narrowed `proposal_codes_to_fetch()`, the new `proposal_codes_not_fetchable()` and a 5-tuple `refresh_all()`. The module docstring gains an F7 paragraph."
    - "solsys_code/unattended.py: `step_proposal_allocation()` unpacks 5 values and renders `not fetchable: N`. The docstring says exclusions are never a failure."
    - "solsys_code/tests/test_proposal_allocation.py: new `ProposalCodeFetchabilityTests` with 7 tests, and 3 pre-existing tests updated (named in planning finding 4)."
    - "solsys_code/tests/test_unattended.py: 2 new tests in `TestProposalAllocationStep`."
    - "docs/runbooks/telescope_runs_calendar.rst: step 5, one new paragraph, the sample log line, and the troubleshooting entry."
    - "CLAUDE.md: one new paired-docs map entry."
    - "docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb: markdown cell `94a24de0` only, with no re-execution."
    - ".planning/v2.4-INTENT-REVIEW.md: an F7 'Fix landed' paragraph. Working tree only, never committed."
  key_links:
    - "`unattended.STEPS` → `step_proposal_allocation()` → `refresh_all(LCOFacility())` → `proposal_codes_to_fetch()` for the request loop, and `proposal_codes_not_fetchable()` for the count. `fetch_proposal_allocations()` (with its WR-07 `_PROPOSAL_CODE_RE` guard) is reached only for fetchable codes."
    - "`_PORTAL_SITE_OBSCODES` is derived from `campaign_attribution.OBSERVED_TELESCOPE_OBSCODES` values plus `LCO_SITE_CODE_TO_OBSCODE` values. It is never a new literal list. `campaign_gap.py:26-29` already imports the same two tables, which sets the import-discipline precedent."
    - "`campaign_tally` reads `proposal_allocation.estimated_unused_nights(run.proposal_code)` (about lines 423 and 621). It is not edited: with no stored rows for an excluded code it gets `None` and renders 'not yet known'."
    - "The cron runs `run_unattended` from THIS checkout every 15 minutes. If either production module fails to import, every step of the tick fails, not just this one."
---

<objective>
This plan fixes intent-review finding F7 (`.planning/v2.4-INTENT-REVIEW.md`, "### F7 (2026-10-02)"). Since 10:00 PDT every tick has logged `step proposal_allocation: FAILED | proposals: 3, rows written: 16, failed: 1, first error: HTTPError` and ended with `exit=1`, so staff get a failure email every 15 minutes. Run 76 (NTT/EFOSC2, `source=classical_file`, site La Silla 809) carries the ESO code `117.2A2N.001`. `proposal_codes_to_fetch()` sends every distinct non-blank `CampaignRun.proposal_code` to the LCO portal. After the fix, only codes that can be LCO/SOAR portal proposals are fetched. The rest are counted as `not fetchable: N`, are never a step failure, and their runs' tally stays "not yet known".

Purpose: the SCHED-09 failure email has to mean something again. A permanent false failure hides real ones.

Output: Task 1 (tracer) is the fix plus 9 new tests. Task 2 is the paired docs, per CLAUDE.md "Paired docs". Task 3 is the full gates and the F7 "Fix landed" note.

**Planning-time findings. Read these before starting; each one changes how a task is done.**

1. **The cron runs from this checkout, so the fix goes live on the first tick after the fix commit.** A tick takes about 60 s, at :00/:15/:30/:45; the last observed tick logged `duration=56s`. `solsys_code/unattended.py` imports `refresh_all` at module level. If `proposal_allocation.py` fails to import, `run_unattended` cannot start and every step fails. So:
   - Write the tests first. The runner never imports test files.
   - Make the production edits in the order Task 1 gives. That order keeps every saved state importable.
   - Make the two edits that change the `refresh_all()` return arity together (edits d and e) back-to-back, starting right after an `=== FOMO unattended run END` banner in `tail -n 3 /var/log/fomo/unattended.log`. A tick that lands between them fails only this step, with a ValueError caught by the runner's per-step try/except (D-02). This step is already failing every tick.
   - Run the import smoke check after every production edit.

   A read-only query of `src/fomo_db.sqlite3` at planning time found these runs carrying a code:
   - run 1: `legacy`, site E10, `LCO2026A-003`
   - runs 69-75: `lco_queue`, `site` NULL, `1m0`, `KEY2026B-004`
   - run 76: `classical_file`, site 809, `117.2A2N.001`

   The only active watched proposal is `KEY2026B-004`. So after the fix the next tick should read `proposals: 2, rows written: <about 16>, failed: 0, not fetchable: 1`, with `exit=0`. If the operator has already blanked run 76's code as interim relief, it reads `not fetchable: 0`. Never run anything against the live database yourself. The live confirmation is the operator's.

2. **Rule chosen: run provenance (the brief's option 1), derived from the codebase's existing LCO/SOAR obscode tables.** A run code is fetchable when `Q(source__in=_PORTAL_RUN_SOURCES) | Q(site__obscode__in=_PORTAL_SITE_OBSCODES)` holds. It is added as one more `.filter()` on the existing `values_list` query.
   - `_PORTAL_RUN_SOURCES` is `(CampaignRun.Source.LCO_QUEUE, CampaignRun.Source.SOAR_QUEUE)`. SOAR proposals live on the same LCO portal, which is why the step uses one `LCOFacility` for every code.
   - `_PORTAL_SITE_OBSCODES` is `frozenset(OBSERVED_TELESCOPE_OBSCODES.values()) | frozenset(LCO_SITE_CODE_TO_OBSCODE.values())`, taken from `solsys_code/campaign_attribution.py` (lines 75-77 and 93). That is the FTN, FTS and SOAR obscodes. It is the codebase's only verified Observatory-obscode-level LCO/SOAR definition, and `campaign_gap.py` already imports both tables.

   The other candidates fail:
   - `Observatory` has no network or facility field. Its fields are obscode, name, short_name, lat, lon, altitude, timezone and observations_type.
   - `calendar_utils.SITE_TELESCOPE_MAP` is keyed by LCO 3-letter site codes, which cannot be matched against `run.site`.
   - `observation_projector.PROJECTED_FACILITIES` classifies `ObservationRecord.facility`, not runs.
   - `telescope_runs.SITES` is the classical file's own vocabulary, not an LCO definition.

   Option 2 (a code-shape regex) is rejected. Option 1 is cleanly expressible with `values_list`. Provenance is a recorded fact, whereas a shape regex is a guess about the portal's naming that nothing in the codebase records. And WR-07's comment (`proposal_allocation.py:68-79`) deliberately keeps `_PROPOSAL_CODE_RE` a charset guard, not a format check.

   **Discretion: `telescope_class` is NOT a third branch.**
   - F6 shows operators set it by hand with a different meaning.
   - `derive_telescope_class()` infers it from free text.
   - A false positive here is exactly F7's failure. A false negative is benign: it is counted as `not fetchable`, the tally reads "not yet known", and the runbook tells the operator what to check.
   - Known false-negative shapes: a non-queue run whose site is an LCO 1m-network observatory. Those obscodes are deliberately unseeded, see `campaign_attribution.py:60-74`. Also a site-less run that is not from a queue. No such run carrying a code exists live.

3. **API decisions.**
   - `proposal_codes_to_fetch()` keeps its name and signature; only its set narrows.
   - New `proposal_codes_not_fetchable() -> list[str]` returns `sorted(all distinct non-blank run codes - set(proposal_codes_to_fetch()))`.
   - `refresh_all()` appends `not_fetchable` (an int, the length of that list) as a 5th tuple element.
   - Grep at planning time confirmed what else is affected. The only production caller of `refresh_all()` is `unattended.step_proposal_allocation()`. No notebook or other module calls `refresh_all()`/`proposal_codes_to_fetch()`. No standalone management command runs this fetch. `WatchedProposal.last_run_summary` belongs to the discovery sweep, not this step, so it needs no wording change.
   - The step summary stays integer counters plus the exception class name (T-37-04/T-37-05/SCHED-10). The excluded codes themselves are never rendered.

4. **Three pre-existing tests change, because they pin the pre-F7 contract.**
   - `ProposalCodesToFetchTests.test_union_of_watched_and_run_codes_sorted_and_deduped`: its two code-carrying runs are created with the `legacy` default and no site, so neither would be fetchable any more. Give those two runs `source=CampaignRun.Source.LCO_QUEUE` and keep the assertion unchanged.
   - `RefreshAllTests.test_isolates_one_failing_proposal_from_the_rest` and `RefreshAllTests.test_empty_code_list_is_a_no_op`: they unpack 4 values. Unpack 5 and assert `not_fetchable == 0`.

   Every other pre-existing test in both edited test modules must stay byte-for-byte unchanged in body. The Task 1 pin gate checks this.

5. **Paired docs: the CLAUDE.md rule, applied rather than skipped.**
   - `solsys_code/unattended.py` maps to the runbook section "How do I run everything unattended?" in `docs/runbooks/telescope_runs_calendar.rst`, not to a notebook (CLAUDE.md says so explicitly). That section is updated.
   - `solsys_code/proposal_allocation.py` has no mapped doc. Its behavior is documented in that same section's step 5 and in the troubleshooting entry "The unused figure says it is not yet known", so both are updated. CLAUDE.md's map gains one entry for it, as the map asks to be extended.
   - `solsys_code/allocation_projector.py` is NOT changed, so `reconcile_campaign_runs_demo.ipynb` is not touched.
   - `docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` markdown cell `94a24de0` claims the step "can look up this run's LCO Observation Portal time allocation". Its very next code cell, `b9c3b23d`, demonstrates NTT lines carrying ESO tokens (`[0110.C-0234]`, `[0111.C-9999]`), which is exactly the F7 shape. That prose becomes wrong, so it is corrected. This is a markdown-only edit: no code cell changes, so the stored outputs stay valid and the notebook is NOT re-executed. A gate proves that every other cell, and the metadata, are unchanged.

   Editing CLAUDE.md is a change to the project's instruction file. It is limited to that single map entry.

6. **`.planning/v2.4-INTENT-REVIEW.md` belongs to the operator and has uncommitted edits.** Task 3 adds its paragraph with one scoped Edit and must NOT stage or commit the file. Do not tick either F7 checkbox.

7. **Branch and staging.** Run `git branch --show-current` before the first commit; it must print `issue37-telescope-runs-calendar`. Stage every commit by explicit path, and never use `git add -A` or `git add .`. The working tree has unrelated untracked files (`.gsd/`, `.planning/agent-history.json`, `reqgroup_2682493.json`, `src/fomo_db_20260929.sqlite3`) that must stay out of every commit. End every commit message with the attribution trailer lines (`Co-Authored-By:` and `Claude-Session:`) that your session's system reminder specifies.

Source coverage audit:
- GOAL (F7: only LCO-fetchable codes are attempted; a non-LCO code is reported, never a step failure): Task 1.
- REQ TALLY-01 (the "not yet known" contract): Task 1, test P5.
- REQ SCHED-09 (the failure email is meaningful again): Task 1, tests U1 and U2.
- REQ SCHED-10 (summary stays integer-only): Task 1 summary format, plus the unchanged `test_summary_leaks_no_api_key_or_response_body`, which the pin gate covers.
- REQ SCHED-08 (the unattended step keeps running on the schedule): Task 1 ordering and smoke checks.
- CONTEXT, the brief's fix direction:
  - provenance rule reusing the existing obscode definition: Task 1, edits a-c, AST gate;
  - exclusions counted and surfaced: edits d-e;
  - command and `last_run_summary` wording checked: finding 3, nothing to change.
- CONTEXT, required tests 1-5: P1 with U1, P2, P3, P4, P5. Extras: P6 for other non-LCO shapes, P7 for T-37-06, U2 for a real failure alongside an exclusion.
- Paired docs and the CLAUDE.md map: Task 2.
- After-landing note: Task 3.
- RESEARCH: none (no research phase).

Nothing is unplanned, and nothing deferred is present.
</objective>

<execution_context>
@/home/tlister/git/fomo_devel/.claude/gsd-core/workflows/execute-plan.md
@/home/tlister/git/fomo_devel/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/v2.4-INTENT-REVIEW.md
@CLAUDE.md

<interfaces>
`solsys_code/proposal_allocation.py` (290 lines), confirmed at planning time:
- Module docstring at 1-18. Imports at 20-33: `from django.db.models import F, Sum` is at 27, and `from solsys_code.models import CampaignRun, ProposalTimeAllocation, WatchedProposal` is at 33.
- `_PROPOSAL_CODE_RE` is at 80, with its WR-07 comment at 68-79.
- `proposal_codes_to_fetch() -> list[str]` is at 83-95. Body: `watched = set(WatchedProposal.objects.filter(is_active=True).values_list('proposal_code', flat=True))`; `run_codes = set(CampaignRun.objects.exclude(proposal_code='').values_list('proposal_code', flat=True).distinct())`; `return sorted(watched | run_codes)`.
- `fetch_proposal_allocations(proposal_code, facility)` is at 98-152, and makes the call `make_request('GET', urljoin(...), headers=..., timeout=...)`, so the URL is `call.args[1]` on a patched mock. Do not change it.
- `unused_hours_for()` is at 208 and `estimated_unused_nights()` at 243. Do not change them.
- `refresh_all(facility) -> tuple[int, int, int, str | None]` is at 264-289. It loops over `proposal_codes_to_fetch()` and catches `PortalUnavailable` per code.

`solsys_code/campaign_attribution.py`: `LCO_SITE_CODE_TO_OBSCODE: dict[str, str]` is at 75 and `OBSERVED_TELESCOPE_OBSCODES: dict[str, str]` at 93, both module-level. The module imports only `calendar_utils`, `models` and `telescope_runs` from the project, and never the views or ephemeris modules. It does not import `proposal_allocation` or `campaign_tally`, so there is no cycle.

`solsys_code/unattended.py` (1034 lines): `from solsys_code.proposal_allocation import refresh_all` is at 52. `step_proposal_allocation(dry_run)` is at 474-501, with its docstring "Returns:" paragraph at 481-487. The body at 493-501 is `with command_lock('proposal_allocation'):`, then `attempted, rows_written, failed, first_exception = refresh_all(LCOFacility())`, then `summary = f'proposals: {attempted}, rows written: {rows_written}, failed: {failed}'`, then `if first_exception: summary += f', first error: {first_exception}'`, then `return StepResult(name='proposal_allocation', failed=failed > 0, summary=summary)`, with the `except LockContended` skip after it.

`solsys_code/models.py` `CampaignRun`:
- `Source` (220-257) has `CLASSICAL_FILE`, `LCO_QUEUE`, `SOAR_QUEUE`, `GEMINI_QUEUE`, `ESO_QUEUE`, `WEB`, `CSV_IMPORT` and `LEGACY`; the default is `LEGACY`.
- `TelescopeClass.ONE_M0` is `'1m0'`.
- `site` is an FK to `Observatory`, null/blank.
- `contact_person` and `contact_email` are at 370-371.
- `proposal_code` is a CharField, blank with a default of ''.
- `telescope_instrument` is the only field without a default.

`Observatory` fixture idiom (`solsys_code/tests/test_allocation_projector.py:51-70`): `Observatory.objects.create(obscode=..., name=..., short_name=..., lat=..., lon=..., altitude=..., timezone=..., observations_type=Observatory.OPTICAL_OBSTYPE)`.

`solsys_code/tests/test_proposal_allocation.py` (443 lines):
- Imports at 8-18: `MagicMock, patch`, `requests`, `forms`, `IntegrityError, transaction` from django.db, `TestCase`, `timezone`, `ImproperCredentialsException`, `proposal_allocation as pa`, and `CampaignRun, ProposalTimeAllocation, WatchedProposal`.
- `_mock_facility()` is at 103.
- `ProposalCodesToFetchTests` is at 110-122.
- `RefreshAllTests` is at 413-443; it is the last class. Its ok-response idiom is `MagicMock()` with `.json.return_value = {'timeallocation_set': [{'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 10.0, 'std_time_used': 2.0}]}`.

`solsys_code/tests/test_unattended.py`:
- Imports at 11-43 include `MagicMock, patch`, `requests`, `ImproperCredentialsException`, `CampaignRun, WatchedProposal`, and `Observatory`.
- `UnattendedTestBase` (73-) patches `solsys_code.unattended.requests.get` and creates `self.ground_site` (obscode F65) and `self.campaign`.
- `TestProposalAllocationStep` is at 1179-1246. Its last test is `test_summary_leaks_no_api_key_or_response_body`; `_FAKE_LCO_API_KEY` follows at about 1248.
</interfaces>
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Only LCO/SOAR-fetchable proposal codes reach the portal; the rest are counted "not fetchable" and never fail the step (tests first, then ordered production edits)</name>
  <files>solsys_code/tests/test_proposal_allocation.py, solsys_code/tests/test_unattended.py, solsys_code/proposal_allocation.py, solsys_code/unattended.py</files>
  <read_first>
    - solsys_code/proposal_allocation.py (whole file, 290 lines)
    - solsys_code/unattended.py lines 1-55 and 470-512
    - solsys_code/campaign_attribution.py lines 1-40 and 60-95
    - solsys_code/tests/test_proposal_allocation.py lines 1-122 and 410-443
    - solsys_code/tests/test_unattended.py lines 1-110 and 1176-1250
  </read_first>
  <behavior>
    New `ProposalCodeFetchabilityTests(TestCase)` at the end of test_proposal_allocation.py. Its class docstring says it pins F7 / quick task 261002-gev.

    `setUpTestData` creates four `Observatory` rows with the fixture idiom: obscode 809 (La Silla, `America/Santiago`), E10 (FTS, Siding Spring, `Australia/Sydney`), F65 (FTN, Haleakala, `Pacific/Honolulu`) and I33 (SOAR, Cerro Pachon, `America/Santiago`). Plausible lat/lon/altitude values are enough.

    Helpers:
    - `_run(self, code, *, source, site=None, **extra)` creates a `CampaignRun` with a unique `telescope_instrument=f'F7 fixture {source} {code}'`.
    - `_refresh(self)` patches `solsys_code.proposal_allocation.make_request` with `return_value=` the ok-response idiom, calls `pa.refresh_all(_mock_facility())`, and returns `(result, urls)`, where `urls = [c.args[1] for c in mock.call_args_list]`.

    The mock answers EVERY code with a row. A wrongly-fetched code therefore shows up as a stored row, not as a failure.

    - P1 `test_eso_code_on_a_classical_run_at_la_silla_is_never_sent_to_the_portal`:
      - Fixtures: a `classical_file` run at 809 with `117.2A2N.001`, and a `classical_file` run at E10 with `LCO2026A-003`.
      - Unpack 5 from `_refresh()`: `(attempted, rows_written, failed, first_exception, not_fetchable) == (1, 1, 0, None, 1)`.
      - No url contains `117.2A2N.001`, and exactly one contains `LCO2026A-003`.
      - `pa.proposal_codes_to_fetch() == ['LCO2026A-003']` and `pa.proposal_codes_not_fetchable() == ['117.2A2N.001']`.
    - P2 `test_lco_code_on_a_run_at_each_known_lco_or_soar_site_is_fetched`:
      - Fixtures: `classical_file` at E10 with `LCO2026A-003`, `legacy` at F65 with `LCO2026A-004` (the live run-1 shape), and `web` at I33 with `SOAR2026A-005`.
      - `proposal_codes_to_fetch()` is all three, sorted. `proposal_codes_not_fetchable() == []`.
      - `_refresh()` gives `attempted == 3` and `not_fetchable == 0`.
    - P3 `test_queue_sourced_run_codes_are_fetched_with_no_site`:
      - Fixtures: `lco_queue` with `site=None`, `telescope_class=CampaignRun.TelescopeClass.ONE_M0` and `KEY2026B-004`; and `soar_queue` with `site=None` and `SOAR2026B-001`.
      - Both are in `proposal_codes_to_fetch()`.
      - This is a guard and passes before the fix.
    - P4 `test_active_watched_proposal_is_always_fetched`:
      - Fixtures:
        - active `WatchedProposal` `LCO2026B-010`, with no run;
        - active `WatchedProposal` `SHARED2026A-001`, which is also on a `classical_file` run at 809;
        - inactive `WatchedProposal` `OFF2026A-002`.
      - `proposal_codes_to_fetch() == ['LCO2026B-010', 'SHARED2026A-001']` and `proposal_codes_not_fetchable() == []`.
    - P5 `test_excluded_code_stays_not_yet_known_after_a_refresh`:
      - Fixture: only a `classical_file` run at 809 with `117.2A2N.001`.
      - Call `self._refresh()` without unpacking it.
      - `ProposalTimeAllocation.objects.filter(proposal_code='117.2A2N.001').exists()` is False, and `pa.unused_hours_for('117.2A2N.001')` and `pa.estimated_unused_nights('117.2A2N.001')` are both `None`. This is D-07 / TALLY-01: never zero.
    - P6 `test_eso_gemini_and_site_less_runs_are_not_fetchable`:
      - Fixtures: `eso_queue`, site None, `0110.C-0234`; `gemini_queue`, site None, `GS-2026A-FT-115`; `classical_file`, site None, `NOSITE2026A-001`; `legacy` at 809, `LEG2026A-002`.
      - First `assertNotIn` each code in `proposal_codes_to_fetch()`, then `proposal_codes_not_fetchable()` equals the four, sorted.
    - P7 `test_code_selection_never_reads_a_contact_field` (T-37-06):
      - Fixtures: one `lco_queue` run with a code and `contact_person='Secret Person'`, `contact_email='secret@example.org'`; and one `classical_file` run at 809 with a code and the same contact values.
      - Inside `CaptureQueriesContext(connection)`, call both selection functions.
      - At least one captured SQL string contains `campaignrun` (case-insensitive), and none contains `contact`.

    Two new tests at the end of `TestProposalAllocationStep` in test_unattended.py. Each creates its own La Silla `Observatory` (obscode 809), an active `WatchedProposal('AAA-2026-001')`, and a `CampaignRun(source=CLASSICAL_FILE, site=<La Silla>, proposal_code='117.2A2N.001', telescope_instrument='NTT/EFOSC2')`:
    - U1 `test_non_lco_code_is_reported_not_fetchable_and_the_step_stays_ok`:
      - `make_request` is patched with an ok response.
      - `result.failed` is False. The summary contains `proposals: 1`, `failed: 0` and `not fetchable: 1`, and does not contain `first error`.
      - No `make_request` call url contains `117.2A2N.001`.
    - U2 `test_a_real_portal_failure_still_fails_the_step_alongside_an_exclusion`:
      - `make_request` is patched with `side_effect=requests.exceptions.Timeout('boom')`.
      - `result.failed` is True, and the summary contains `failed: 1`, `not fetchable: 1` and `first error: Timeout`.

    Pre-existing tests updated, per planning finding 4: the union test's two code-carrying runs get `source=CampaignRun.Source.LCO_QUEUE`, with the assertion unchanged; the two `RefreshAllTests` unpack 5 values and assert `not_fetchable == 0`. In `test_empty_code_list_is_a_no_op` the expected tuple becomes `(0, 0, 0, None, 0)`.
  </behavior>
  <action>
    **Step 0: baseline.**
    - Run the first `<automated>` command (the import smoke check). It must print `importable, no heavy imports`. If it fails before any edit, stop and report: the heavy-import check is then unreliable on this host.
    - Run `python manage.py test solsys_code.tests.test_proposal_allocation solsys_code.tests.test_unattended --exclude-tag=ephemeris_segfault`. Record its `Ran N tests` figure in the SUMMARY; it is the "before" half of the delta Task 3 quotes. It must pass. If it does not, stop and report.

    **Step 1: RED. Test-only edits; the runner never imports these files.**
    - In test_proposal_allocation.py:
      - Merge `connection` into the existing `from django.db import ...` line.
      - Add `from django.test.utils import CaptureQueriesContext` and `from solsys_code.solsys_code_observatory.models import Observatory`.
      - Write `ProposalCodeFetchabilityTests` exactly as `<behavior>` lists it, after `RefreshAllTests`.
      - Apply the three planning-finding-4 updates. Update `ProposalCodesToFetchTests`'s class docstring to say the run half is limited to LCO/SOAR-fetchable runs (F7).
    - In test_unattended.py, add U1 and U2 at the end of `TestProposalAllocationStep`. The existing imports suffice.
    - Never use `SiderealTargetFactory`. These tests create no `Target` at all (CLAUDE.md).
    - Run the second `<automated>` command. Expected RED:
      - 4 failures (AssertionError): P5, P6, U1, U2.
      - 6 errors. Three are `ValueError: not enough values to unpack (expected 5, got 4)`: P1 and the two edited `RefreshAllTests`. The other three are `AttributeError` naming `proposal_codes_not_fetchable`: P2, P4, P7.
      - P3 and the edited union test pass (guards).
      - Every other test passes.
      - If a failure is anything else (ImportError, NameError, a fixture IntegrityError, or an AttributeError naming a different attribute), fix the test before committing.
    - Run `git branch --show-current` (finding 7). Commit the two test files by explicit path with the message `test(261002-gev): pin that only LCO/SOAR-fetchable proposal codes reach the portal (F7)`.

    **Step 2: GREEN. Ordered production edits.** Make each edit in ONE Edit tool call, and run the first `<automated>` smoke check after each one (finding 1). The order keeps every saved state importable.
    - **(a)** In proposal_allocation.py:
      - Change the `django.db.models` import to `F, Q, Sum`.
      - Add `from solsys_code.campaign_attribution import LCO_SITE_CODE_TO_OBSCODE, OBSERVED_TELESCOPE_OBSCODES` in isort position, before the `solsys_code.models` import.
      - Do both in one Edit spanning lines 27-33. Unused for one edit is fine; ruff runs only at the end.
    - **(b)** Directly after `_PROPOSAL_CODE_RE`, add `_PORTAL_RUN_SOURCES` and `_PORTAL_SITE_OBSCODES` (finding 2).
      - Derive the obscode set ONLY from the two imported tables' `.values()`. Never write an obscode string literal in this module; the AST gate rejects one.
      - Give the block one comment explaining:
        - F7 (quick task 261002-gev);
        - why provenance and not code shape;
        - that the obscode set is the FTN/FTS/SOAR observatories `campaign_attribution` already verifies, and grows when its alias tables grow;
        - why `telescope_class` is deliberately not a branch;
        - that a false negative is benign (counted, and the tally says "not yet known").
    - **(c)** Replace `proposal_codes_to_fetch()` and add `proposal_codes_not_fetchable()` directly after it, in one Edit over lines 83-95.
      - `proposal_codes_to_fetch()` keeps the watched half unchanged. Its run half adds `.filter(Q(source__in=_PORTAL_RUN_SOURCES) | Q(site__obscode__in=_PORTAL_SITE_OBSCODES))` to the existing `values_list(...).distinct()` chain.
      - `proposal_codes_not_fetchable()` builds every distinct non-blank run code with the same `values_list` idiom and returns the sorted set difference against `proposal_codes_to_fetch()`.
      - Both docstrings state the rule, the T-37-06 `values_list`-only constraint, and that a code also watched or also on a fetchable run is fetched rather than counted. Google-style docstrings, single quotes, 120 columns.
    - **(d) + (e)**, back-to-back, right after an END banner (finding 1):
      - (d) `refresh_all()` returns `tuple[int, int, int, str | None, int]`. It computes `not_fetchable = len(proposal_codes_not_fetchable())` and returns it as the 5th element. The loop is unchanged. Its docstring's Returns section names the 5th element and says excluded codes are never requested and never counted as failed.
      - (e) In unattended.py `step_proposal_allocation()`, unpack the 5 values and render `summary = f'proposals: {attempted}, rows written: {rows_written}, failed: {failed}, not fetchable: {not_fetchable}'`. Keep the `first error` append and `failed=failed > 0` exactly as they are. Do the docstring's Returns paragraph in the same Edit: a code that cannot be an LCO/SOAR portal proposal is never requested and is counted under `not fetchable`, never as a failure (F7, quick task 261002-gev). The summary is still built only from integer counters and an exception class name.
    - **(f)** Prose only. In the proposal_allocation.py module docstring, add one paragraph after the existing second paragraph. It says which codes are fetched, that the rest are counted and never sent, and that their tally stays "not yet known".
    - Do NOT touch `fetch_proposal_allocations()`, `store_proposal_allocations()`, `unused_hours_for()`, `estimated_unused_nights()`, `campaign_tally.py` or `campaign_attribution.py`.

    **Step 3: verify, then commit.**
    - Run every `<automated>` command below.
    - If any pre-existing test fails, stop and report. The only exceptions are the three tests named in finding 4.
    - Commit the two production files by explicit path with the message `fix(261002-gev): fetch only LCO/SOAR-fetchable proposal codes; count the rest as not fetchable, never a step failure (F7)`.
    - From this commit on, the next cron tick should report `ok` for this step. That is intended.
  </action>
  <verify>
    <automated>python manage.py shell -c "import sys, solsys_code.proposal_allocation as pa, solsys_code.unattended as u; heavy = [m for m in ('solsys_code.views', 'solsys_code.ephem_utils') if m in sys.modules]; assert not heavy, heavy; print('importable, no heavy imports:', pa.refresh_all.__name__, u.step_proposal_allocation.__name__)"</automated>
    <automated>python manage.py test solsys_code.tests.test_proposal_allocation solsys_code.tests.test_unattended solsys_code.tests.test_campaign_tally --exclude-tag=ephemeris_segfault</automated>
    <automated>python -c "
import ast
heavy = ('solsys_code.views', 'solsys_code.ephem_utils')
for p in ('solsys_code/proposal_allocation.py', 'solsys_code/unattended.py'):
    for n in ast.walk(ast.parse(open(p).read())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(heavy):
            raise SystemExit(f'{p} imports {n.module}')
        if isinstance(n, ast.Import) and any(a.name.startswith(heavy) for a in n.names):
            raise SystemExit(f'{p} imports a heavy module')
tree = ast.parse(open('solsys_code/proposal_allocation.py').read())
imported = {a.name for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'solsys_code.campaign_attribution' for a in n.names}
assert {'OBSERVED_TELESCOPE_OBSCODES', 'LCO_SITE_CODE_TO_OBSCODE'} <= imported, imported
banned = {'E10', 'F65', 'I33', '809', 'W85', 'K92'}
lits = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value in banned]
assert not lits, f'hard-coded obscode literal(s) in proposal_allocation.py: {lits}'
fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
assert 'proposal_codes_not_fetchable' in fns, 'missing proposal_codes_not_fetchable'
called = {c.func.id for c in ast.walk(fns['refresh_all']) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
assert {'proposal_codes_to_fetch', 'proposal_codes_not_fetchable', 'fetch_proposal_allocations'} <= called, called
print('OK: obscode set derived from campaign_attribution, no literals, no heavy imports, refresh_all reaches both selectors')
"</automated>
    <automated>python -c "
import ast, subprocess
red = subprocess.check_output(['git', 'log', '--format=%H', '--grep=^test(261002-gev)', '-1'], text=True).strip()
assert red, 'RED commit test(261002-gev) not found'
allowed = {('ProposalCodesToFetchTests', 'test_union_of_watched_and_run_codes_sorted_and_deduped'), ('RefreshAllTests', 'test_isolates_one_failing_proposal_from_the_rest'), ('RefreshAllTests', 'test_empty_code_list_is_a_no_op')}
def bodies(src):
    out = {}
    for c in ast.parse(src).body:
        if isinstance(c, ast.ClassDef):
            for f in c.body:
                if isinstance(f, ast.FunctionDef):
                    out[(c.name, f.name)] = ast.dump(f)
    return out
checked = 0
for path in ('solsys_code/tests/test_proposal_allocation.py', 'solsys_code/tests/test_unattended.py'):
    old = bodies(subprocess.check_output(['git', 'show', f'{red}^:{path}'], text=True))
    new = bodies(open(path).read())
    for key, dump in old.items():
        assert key in new, f'{path}::{key} removed'
        if key not in allowed:
            assert new[key] == dump, f'{path}::{key} changed'
            checked += 1
print(f'OK: {checked} pre-existing test methods unchanged; only the 3 planned edits differ')
"</automated>
    <automated>pre-commit run ruff --files solsys_code/proposal_allocation.py solsys_code/unattended.py solsys_code/tests/test_proposal_allocation.py solsys_code/tests/test_unattended.py && pre-commit run ruff-format --files solsys_code/proposal_allocation.py solsys_code/unattended.py solsys_code/tests/test_proposal_allocation.py solsys_code/tests/test_unattended.py</automated>
  </verify>
  <done>
    - The RED commit and the fix commit exist, each touching only its two files.
    - The RED run matched the expected 4 failures, 6 errors and 2 passing guards, and the SUMMARY records it.
    - All three test modules pass, with the 9 new tests green.
    - The AST gate confirms the obscode set comes from `campaign_attribution`, with no literals, and that `refresh_all()` reaches both selectors.
    - The pin gate confirms every other pre-existing test method is unchanged.
    - Both ruff hooks are clean on the four files.
    - The SUMMARY records the baseline and post-fix `Ran N tests` for the two edited test modules.
  </done>
</task>

<task type="auto">
  <name>Task 2: Paired docs. The runbook says which codes are fetched and what "not fetchable" means; CLAUDE.md maps proposal_allocation.py; the load_telescope_runs notebook's token prose is narrowed</name>
  <files>docs/runbooks/telescope_runs_calendar.rst, CLAUDE.md, docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb</files>
  <read_first>
    - docs/runbooks/telescope_runs_calendar.rst lines 1481-1525 ("How do I run everything unattended?" / "What runs, and when"), 1980-1993 (the sample failing tick), and 2891-2911 ("The unused figure says it is not yet known")
    - CLAUDE.md lines 120-150 (the paired-docs map)
    - docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb cell `94a24de0`, and code cell `b9c3b23d` for context
  </read_first>
  <action>
    **Part A: runbook.** Plain English, no DB jargon (CLAUDE.md "Planning-doc terminology" applies to prose too). Make four edits, each a scoped Edit, and change nothing else:
    1. **Step 5** in "What runs, and when". Reword the opening of item `5. **proposal_allocation**`. It refreshes the time allocation of every active watched proposal, and of every proposal code carried by a run that can hold an LCO/SOAR portal proposal: a run that came from the LCO or SOAR queue, or a run whose resolved site is one of the LCO/SOAR telescopes FOMO has a verified observatory code for (today FTN, FTS and SOAR). Keep the rest of the item (stored rows; public pages never call the portal).
    2. **New paragraph** directly after the paragraph that begins `A step that fails never stops the later ones` and ends `only as an honest "not yet known".` It says:
       - A proposal code on any other run is never sent to the portal. Example: an ESO code bracketed onto an NTT line of a classical schedule file.
       - The step counts it under `not fetchable: N` in its log line, and this is not a failure: the step stays `ok` and the tick can still exit 0.
       - That run's unused-nights figure stays "not yet known", because there is nothing to fetch it from.
       - The code is still kept on the run; it is what tells two otherwise-identical classical lines apart.
       - If a run you know is at an LCO or SOAR telescope is being counted as not fetchable, check its ingest source and resolved site in the admin.
       - Before quick task 261002-gev (2026-10-02), such a code was sent to the portal and failed this step on every tick.
    3. **The sample failing tick** (about line 1991). Append `, not fetchable: 0` to the `step proposal_allocation: ok | proposals: 1, rows written: 1, failed: 0` line so it matches the new summary format.
    4. **"The unused figure says it is not yet known"**:
       - The **Cause:** paragraph gains a third cause: the run's proposal code is not an LCO/SOAR portal proposal, so the step never fetches it and counts it as `not fetchable`. This is expected for a run at a non-LCO telescope.
       - The **Fix:** paragraph gains one sentence: nothing needs fixing for a non-LCO run. For a run that should be fetchable, check its ingest source and resolved site, as in "How do I run everything unattended?".

    **Part B: CLAUDE.md.** One scoped Edit, inserting ONE new map entry between the line ending `sanctions the runbook section as the paired doc in exactly that case;` and the line beginning `` `solsys_code/campaign_reconciler.py`, ``. Match the surrounding two-space-indented, semicolon-terminated list style. The entry says:
    - `solsys_code/proposal_allocation.py` maps to the runbook's `How do I run everything unattended?` section (its **proposal_allocation** step, and which proposal codes it fetches).
    - It also maps to that runbook's `The unused figure says it is not yet known` troubleshooting entry (`docs/runbooks/telescope_runs_calendar.rst`).
    - It is **not** a notebook: no demo notebook exercises the credentialed portal fetch.

    Change nothing else in CLAUDE.md (finding 5).

    **Part C: notebook prose, markdown only, no re-execution (finding 5).** Use a throwaway Python script in your session scratchpad, never in the repo. Read the notebook with `nbformat.read(path, as_version=4)`, find the cell with id `94a24de0`, and change ONLY its source. Write it back with `nbformat.write`.
    - Keep the first two paragraphs.
    - Narrow the third (the one beginning `As of Phase 37 (TALLY-01)`): the token is stored on `CampaignRun.proposal_code` for every line. But the unattended runner's `proposal_allocation` step looks a code up at the LCO Observation Portal only for a run that can hold an LCO/SOAR portal proposal. Of this file's telescopes that is FTS; an LCO/SOAR queue run also qualifies.
    - Then add: an NTT or Magellan line's token, like the ESO codes in the next cell, still keeps two lines distinct and is still stored, but the step counts it as `not fetchable` and never sends it. That run's unused-time figure stays "not yet known" (quick task 261002-gev, F7).

    Do not execute the notebook. No code cell changes, so every stored output stays valid. If `nbformat.write` changes any other cell or the metadata, which the second `<automated>` gate detects, revert the file with `git checkout -- docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb` and instead edit that one cell's `source` with `json.load` and `json.dump(nb, f, indent=1, ensure_ascii=False)`, followed by a trailing newline.

    **Part D: gates, then commit.** Run every `<automated>` command below, then commit the three files by explicit path with the message `docs(261002-gev): runbook and CLAUDE.md map cover not-fetchable proposal codes; load_telescope_runs notebook prose narrowed (F7)`.
  </action>
  <verify>
    <automated>python -c "
t = open('docs/runbooks/telescope_runs_calendar.rst').read()
a = t.index('How do I run everything unattended?\n---')
b = t.index('Setting it up on a fresh host', a)
sec = t[a:b]
for token in ('not fetchable', '261002-gev', 'not yet known'):
    assert token in sec, f'unattended section lacks {token!r}'
assert sec.find('not fetchable', sec.index('A step that fails never stops the later ones')) != -1, 'new paragraph must follow the step-failure paragraph'
lines = [l for l in t.splitlines() if 'step proposal_allocation: ok | proposals:' in l or 'step proposal_allocation: FAILED | proposals:' in l]
assert lines and all('not fetchable:' in l for l in lines), lines
c = t.index('The unused figure says it is not yet known\n^^^')
d = t.index('See also', c)
assert 'not fetchable' in t[c:d], 'troubleshooting entry lacks the not-fetchable cause'
print('OK: runbook covers not-fetchable codes in all three places')
"</automated>
    <automated>python -c "
import json, subprocess
p = 'docs/notebooks/pre_executed/load_telescope_runs_demo.ipynb'
fix = subprocess.check_output(['git', 'log', '--format=%H', '-1', '--grep=^fix(261002-gev)'], text=True).strip()
assert fix, 'fix commit not found'
old = json.loads(subprocess.check_output(['git', 'show', f'{fix}:{p}'], text=True))
new = json.load(open(p))
assert old['metadata'] == new['metadata'] and old['nbformat'] == new['nbformat'] and len(old['cells']) == len(new['cells']), 'notebook structure changed'
changed = [n.get('id') for o, n in zip(old['cells'], new['cells']) if o != n]
assert changed == ['94a24de0'], f'unexpected changed cells: {changed}'
cell = [c for c in new['cells'] if c.get('id') == '94a24de0'][0]
src = ''.join(cell['source'])
assert cell['cell_type'] == 'markdown'
for token in ('not fetchable', 'not yet known', '261002-gev', 'source_identifier'):
    assert token in src, f'cell 94a24de0 lacks {token!r}'
print('OK: only markdown cell 94a24de0 changed; every output untouched')
"</automated>
    <automated>python -c "
t = open('CLAUDE.md').read()
a = t.index('sanctions the runbook section as the paired doc in exactly that case;')
m = t.index('solsys_code/proposal_allocation.py', a)
b = t.index('solsys_code/campaign_reconciler.py', a)
assert a < m < b, 'map entry is not between the unattended entry and the reconciler entry'
entry = t[m:b]
assert 'How do I run everything unattended?' in entry and 'not yet known' in entry, entry
print('OK: CLAUDE.md map entry in place')
"</automated>
    <automated>pre-commit run sphinx-build --files docs/runbooks/telescope_runs_calendar.rst</automated>
  </verify>
  <done>
    - The runbook covers not-fetchable codes in step 5, in the new paragraph, in every sample log line and in the troubleshooting entry, and Sphinx builds.
    - The CLAUDE.md map has the new entry in the right place.
    - The notebook differs from the fix commit only in markdown cell `94a24de0`.
    - One docs commit touches exactly these three files.
  </done>
</task>

<task type="auto">
  <name>Task 3: Full quality gates, then the F7 "Fix landed" note (working tree only)</name>
  <files>.planning/v2.4-INTENT-REVIEW.md</files>
  <read_first>
    - .planning/v2.4-INTENT-REVIEW.md: the F1 "Fix landed" paragraph (about lines 388-398) and the F5 one (about lines 479-488) for house style, and "### F7 (2026-10-02)" (about lines 515-532)
    - CLAUDE.md, "Commands" and "Testing" (always `python manage.py`; ruff through pre-commit, per D-07)
  </read_first>
  <action>
    **Lint gates.**
    - From the repo root, run `pre-commit run ruff --all-files`, then `pre-commit run ruff-format --all-files`.
    - If either rewrites a file outside this plan's seven code/doc files, stop and report rather than committing someone else's file.
    - If either reformats one of this plan's files, commit only that change, by explicit path, with the message `style(261002-gev): apply pre-commit formatting`.

    **Full suite.**
    - Run the second `<automated>` command with `SCRATCH` set to your session scratchpad directory, never a path inside the repo.
    - Use a Bash timeout of 600000 ms. The suite took about 24 minutes alongside cron ticks last time. If the harness moves it to the background at the limit, wait for its completion notification, then read the log's tail and its `exit=` line.
    - Never start a second concurrent run.
    - The suite uses Django's test database and never opens the live one.
    - Quote the `Ran N tests` and `OK` lines in the SUMMARY. The last recorded full-suite count was 1833 (261001-smo); this plan adds 9.

    **F7 note. Docs only.**
    - Make ONE scoped Edit on `.planning/v2.4-INTENT-REVIEW.md`; never Write. Insert one paragraph immediately before the line ``- [ ] Ticks back to `exit=0`.`` inside F7, separated from it by a blank line, the same layout as F5.
    - Leave both F7 checkboxes unticked.
    - In the F1/F5 house style the paragraph contains:
      - `**Fix landed (quick task 261002-gev, 2026-10-02):**`, followed by the short SHAs from `git log --oneline --grep=261002-gev`, each labelled: tests / fix / runbook + CLAUDE.md map + notebook prose (plus style, if one exists).
      - One parenthetical on the mechanism. `proposal_codes_to_fetch()` now keeps a run's code only when the run came from the LCO/SOAR queue or its resolved site is one of the FTN/FTS/SOAR observatories `campaign_attribution` already verifies. Every other run code is counted by `refresh_all()` as `not fetchable` and never requested. `step_proposal_allocation()` renders `not fetchable: N`, and `failed` still counts only real portal failures.
      - The two edited test modules' `Ran N` before and after (Task 1's figures), and the full-suite count with `OK`.
      - One sentence of planning-time read-only evidence (finding 1): runs 1 (E10, `LCO2026A-003`) and 69-75 (`lco_queue`, `KEY2026B-004`) stay fetchable, and run 76 (809, `117.2A2N.001`) becomes not fetchable. The bracketed code stays on run 76 because it keeps that line's natural key distinct, so the interim relief of blanking it is no longer needed.
      - A closing clause: the live confirmation is the next cron tick reading `step proposal_allocation: ok | proposals: 2, rows written: …, failed: 0, not fetchable: 1` (or `not fetchable: 0` if run 76's code was blanked meanwhile) and `exit=0`, and the operator ticks the box.
    - Do NOT stage or commit this file (finding 6). Run nothing against the live database. In the SUMMARY, say the note is left uncommitted beside the operator's own edits.
  </action>
  <verify>
    <automated>pre-commit run ruff --all-files && pre-commit run ruff-format --all-files</automated>
    <automated>python manage.py test solsys_code --exclude-tag=ephemeris_segfault 2>&1 | tee "$SCRATCH/261002-gev-full-suite.log" | tail -n 5; echo "exit=${PIPESTATUS[0]}"</automated>
    <automated>python -c "
t = open('.planning/v2.4-INTENT-REVIEW.md').read()
f7 = t.index('### F7 (2026-10-02)')
note = t.index('**Fix landed (quick task 261002-gev', f7)
box = t.index('- [ ] Ticks back to', f7)
f8 = t.index('### F8 (2026-10-02)')
assert f7 < note < box < f8, 'note must sit inside F7, before its unticked checkbox'
seg = t[note:box]
for token in ('not fetchable', 'exit=0', 'operator', '117.2A2N.001'):
    assert token in seg, f'note lacks {token!r}'
assert '- [ ] Routed.' in t[box:f8], 'second F7 checkbox missing or ticked'
print('OK: F7 note in place, both checkboxes untouched')
"</automated>
    <automated>git status --porcelain .planning/v2.4-INTENT-REVIEW.md</automated>
    <human-check>Operator: after the first unattended tick that starts after the fix commit, run `tail -n 3 /var/log/fomo/unattended.log`. Confirm that `step proposal_allocation: ok | proposals: 2, rows written: …, failed: 0, not fetchable: 1` appears, followed by an END banner with `exit=0`, and that no failure email arrives (a recovery notice is expected instead). Then tick "Ticks back to `exit=0`" under F7 in `.planning/v2.4-INTENT-REVIEW.md`.</human-check>
  </verify>
  <done>
    - Both ruff hooks are clean repo-wide.
    - The full `solsys_code` suite passes with exit 0 and the segfault tag excluded, and its count is quoted in the SUMMARY.
    - The F7 note sits inside F7 above its unticked checkboxes, naming the commits, the test-count delta and the operator's live confirmation.
    - `git status --porcelain` shows the intent-review file still modified and unstaged (` M`), never committed by this plan.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| operator-supplied `proposal_code` (classical file token, admin edit) -> credentialed LCO portal request | Free text decides whether an API-keyed GET is sent, and to which path |
| unattended step summary -> log file and failure email | Whatever the summary renders leaves the process |
| working tree -> running cron | `run_unattended` imports this checkout at every tick, so an edit on disk is live code |
| `CampaignRun` table -> the allocation process | Contact fields must never be loaded into the fetch process (T-37-06) |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-gev-01 | Denial of service | `step_proposal_allocation()`: a permanent false failure floods staff with an email every 15 minutes and hides real failures (SCHED-09) | high | mitigate | Excluded codes are never requested and never reach `failed`. The step is `failed=failed > 0`, unchanged. U1 pins ok-with-exclusion and U2 pins that a real portal failure still fails the step. |
| T-gev-02 | Information disclosure | `proposal_codes_to_fetch()`/`proposal_codes_not_fetchable()` loading `CampaignRun` rows or contact fields (T-37-06) | medium | mitigate | `values_list('proposal_code', flat=True)` only. The site filter is a JOIN in the WHERE clause, not a loaded column. P7 asserts through `CaptureQueriesContext` that no SQL mentions `contact`. |
| T-gev-03 | Information disclosure | free text (codes) or response content entering the step summary (SCHED-10, T-37-04/05) | medium | mitigate | The summary adds only the integer `not_fetchable`. Excluded codes are never rendered. `test_summary_leaks_no_api_key_or_response_body` stays byte-for-byte unchanged (pin gate). |
| T-gev-04 | Tampering | a crafted code redirecting the credentialed request | medium | mitigate | Unchanged: `fetch_proposal_allocations()` still applies WR-07's `_PROPOSAL_CODE_RE` to every fetched code. This plan only narrows which codes reach it. |
| T-gev-05 | Denial of service | a half-edited `proposal_allocation.py` breaking `unattended.py`'s import, which fails every step of the tick | high | mitigate | Tests come first. Production edits (a)-(f) are ordered so every saved state imports, with the smoke check after each. The arity-changing pair (d)+(e) is done back-to-back after an END banner, and the worst case is one more failing tick of this step only (D-02 per-step isolation). |
| T-gev-06 | Repudiation | a false negative (an LCO code on a run with an unseeded LCO 1m site, or a site-less non-queue run) silently never fetched | low | accept | It is counted in `not fetchable: N` on every tick's log line. The tally reads "not yet known", never zero (P5). The runbook tells the operator to check the run's source and site. No such run exists live (finding 1). |
| T-gev-07 | Tampering | a task commit sweeping the operator's uncommitted intent-review edits into history | low | mitigate | Task 3 edits `.planning/v2.4-INTENT-REVIEW.md` with one scoped Edit and never stages it. The gate checks `git status --porcelain` still shows ` M`. All commits stage by explicit path. |
| T-gev-SC | Tampering | package installs | low | accept | No npm/pip/cargo install in this plan. `nbformat` 5.10.4 is already installed. |
</threat_model>

<verification>
- `python manage.py test solsys_code.tests.test_proposal_allocation solsys_code.tests.test_unattended solsys_code.tests.test_campaign_tally --exclude-tag=ephemeris_segfault` passes, with the 9 new tests green.
- `python manage.py test solsys_code --exclude-tag=ephemeris_segfault` passes (exit 0).
- `pre-commit run ruff --all-files` and `pre-commit run ruff-format --all-files` are clean.
- The AST gate shows the obscode set derived from `campaign_attribution`, no obscode literals and no heavy imports. The pin gate shows every pre-existing test method unchanged except the 3 planned ones.
- The runbook, CLAUDE.md and notebook gates pass, and Sphinx builds.
- `git log` shows `test`, `fix` and `docs` commits for 261002-gev (plus `style` only if a gate reformatted something), touching only the seven code/doc paths. `.planning/v2.4-INTENT-REVIEW.md` stays unstaged, and the pre-existing untracked files stay uncommitted.
</verification>

<success_criteria>
- A run's proposal code is fetched from the LCO portal only when the run is LCO/SOAR queue-sourced or sits at a known LCO/SOAR observatory. Every active watched proposal is always fetched.
- Any other code is counted as `not fetchable: N` in the step summary, is never requested, and never fails the step. Its run's tally stays "not yet known".
- After the fix commit, which is live at the next tick, the unattended runner's ticks return to `exit=0` and the every-15-minutes failure email stops. The operator confirms this and ticks F7's first box.
</success_criteria>

<output>
Create `.planning/quick/261002-gev-fix-f7-proposal-allocation-fetch-only-at/261002-gev-SUMMARY.md` when done. Include:
- the baseline and post-fix `Ran N` for `test_proposal_allocation` + `test_unattended`, and the full-suite `Ran N` / `OK` line;
- the RED result (4 failures, 6 errors, 2 guards passing) and the commit SHAs;
- the three pre-existing tests changed under planning finding 4, named as deliberate deviations;
- the rule chosen (provenance via `campaign_attribution`'s alias tables) and why code shape was rejected, in one or two sentences;
- a statement that the F7 note is in `.planning/v2.4-INTENT-REVIEW.md` uncommitted, and that the live confirmation and the F7 checkboxes are the operator's.
</output>
