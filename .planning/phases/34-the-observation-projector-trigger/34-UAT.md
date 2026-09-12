---
status: complete
phase: 34-the-observation-projector-trigger
source: [34-01-SUMMARY.md, 34-02-SUMMARY.md, 34-03-SUMMARY.md, 34-04-SUMMARY.md, 34-05-SUMMARY.md, 34-06-SUMMARY.md, 34-07-SUMMARY.md, 34-VERIFICATION.md]
started: 2026-09-12T02:39:51Z
updated: 2026-09-12T22:24:09Z
---

## Current Test

[testing complete]

## Tests

### 1. Interleaved-save re-run: two overlapping `updatestatus` runs leave each event matching its record's final persisted state
expected: Drive two overlapping `python manage.py updatestatus` runs against a COPY of the developer
database (`FOMO_DATABASE_PATH=<absolute scratch path>`; never `src/fomo_db.sqlite3`, which is
the SCHED-06 evidence) and inspect the surviving CalendarEvents. Each LCO event's span and title
match its record's final persisted `scheduled_start` / `scheduled_end` / `status` -- no event
describes a superseded intermediate state -- and the two run logs carry zero
`unprojectable ... AttributeError` lines. A single `OperationalError` from SQLite's write lock
under the deliberate double run is expected and is absorbed by `project_record()`'s savepoint.
(Declared `verification: backstop` in 34-01; no automated test exercises concurrency. The
original run of this test was contaminated by G-34-2, now closed by 34-05/34-06/34-07.)
result: pass
reclassified_from: issue
reported: "Step 3: 0 reported for all the 'grep -c's, 'Update completed successfully' report for the 'grep -E'. Step 4 reports "Done (dry run). failed: 0 | LCO: created: 0, updated: 14, unchanged: 145, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0 | SOAR: created: 0, updated: 0, unchanged: 0, unprojectable: 0, site_lookups: 0, site_lookup_failed: 0""
note: "Reclassified to pass on 2026-09-12 after inspection of the scratch copy. The 14 events the dry-run sweep would update are KEY2026B-004 records that went COMPLETED on 2026-09-11 20:02-20:04 UTC (record.modified), i.e. under the pre-34-05 receiver, and were NOT touched by either interleaved run: tom_observations.facility.update_all_observation_statuses() excludes terminal states (facility.py:573), so updatestatus never re-saves a COMPLETED record. Every record the two runs did save projects correctly (0 AttributeError / 0 unprojectable / 0 OperationalError; the freshly COMPLETED 4378046 and 4378332 carry [O] events on their observed windows). The residue is legacy, pre-fix state that only the sweep -- its designed backstop role -- can repair. See Operational Findings F-34-1."
evidence: "scratch copy /tmp/fomo_uat_test1.sqlite3; logs tmp/uat-test1-run-a.txt, tmp/uat-test1-run-b.txt, tmp/uat-test1-dry-run.txt -- keep for diagnosis"

### 2. Runbook determination: `docs/runbooks/telescope_runs_calendar.rst` already describes post-fix behavior and needs no edit
expected: Open `docs/runbooks/telescope_runs_calendar.rst` and read the three regions 34-06 checked:
lines 47-65 ("How do LCO/SOAR queue observations get onto the calendar?"), lines 144-198 ("When
would I run the sweep?"), and lines 1144-1155 (troubleshooting table, `unprojectable` row).
The prose describes the intended behavior -- a saved ObservationRecord (submitted, refreshed by
`updatestatus`, or backfilled) draws/updates its own CalendarEvent through the `post_save`
receiver with no operator command; the sweep's counters and the sweep-vs-receiver division of
labour -- and nowhere documents or implies the broken pre-34-05 path (real `updatestatus`
saves failing to project). Conclusion "no edit needed" holds. (34-06 D6, human judgment.)
result: pass
note: "Runbook accurate for the 34-05 fix; reviewer raised two behaviour-change ideas (recorded under Deferred Follow-Ups, not gaps): failed/aborted records keeping their last scheduled window rather than the original; a clearer dry-run marker for site_lookups. Lines 1144-1155 confirmed good."

### 3. SCHED-06 -- a pending KEY2026B-004 record narrows over real nights with nobody running anything
expected: From the baseline in the SCHED-06 section below, run ONLY `python manage.py updatestatus`
against `src/fomo_db.sqlite3` over several real observing nights -- never
`project_observation_calendar`. Then re-execute
`docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb` end to end UN-ROUTED
(no `FOMO_DATABASE_PATH`) and `git diff` the SCHED-06 baseline JSON. Read the re-execution's
OWN first sweep summary line before anything else: `created: 0, updated: 0` for the narrowed
records proves the `post_save` receiver -- not the sweep -- did the narrowing. At least one
record has moved queued -> placed (or placed -> observed) with its event span/title following.
Fill in the dated re-check row below and flip the verdict from PARTIAL. This is a
verification-over-time item (34-04 D5, 34-05 D5, 34-VERIFICATION.md human item 1) and is
expected to stay pending until real observing nights have elapsed since the
2026-09-11T04:44Z baseline.
result: pass
note: "Closed 2026-09-12 on developer-database evidence gathered read-only (scratch copies only; src/fomo_db.sqlite3 was never swept). After a single `python manage.py updatestatus` at 2026-09-12 22:10 UTC (log: 'Observation change state hook: 10P @ LCO from PENDING to COMPLETED', '11P @ LCO from PENDING to COMPLETED', 'Update completed successfully'), two baseline records narrowed queued -> observed with their events following, via the post_save receiver alone: 4378332 (11P) baseline [Q] 1m0 11P / no schedule -> [O] 1m0 11P 2026-09-12 01:06:46-01:26:04, CalendarEvent.modified == record.modified == 22:10:48; 4378046 (10P) baseline [Q] 1m0 10P -> [O] 1m0 10P 2026-09-12 02:08:39-02:22:01. A real sweep on a throwaway copy changed only their site token (1m0 -> TFN-1m0 / LSC-1m0), which is sweep-only by design (34-02 D4). Paired-docs follow-up: the un-routed notebook re-execution still has to be done and committed; read its first sweep line PER RECORD, because the 14 legacy events (F-34-1) will make it read updated: 16, not 0."

### 4. [34-01 D1] A real ObservationRecord.save() reaches a real CalendarEvent/CalendarEventMeta row end-to-end: creation projects a queued event, a schedule-only save narrows the same event in place, and the real LCOFacility().update_observation_status() path (which TOM's own hook misses) narrows it too -- all with no operator command.
expected: A real ObservationRecord.save() reaches a real CalendarEvent/CalendarEventMeta row end-to-end: creation projects a queued event, a schedule-only save narrows the same event in place, and the real LCOFacility().update_observation_status() path (which TOM's own hook misses) narrows it too -- all with no operator command.
result: pass
source: automated
coverage_id: 34-01/D1
covered_by: solsys_code/tests/test_observation_projector_signals.py#TestPostSaveReceiver, TestUpdateObservationStatusPath

### 5. [34-01 D2] Every lifecycle stage (queued/placed/observed/completed-no-block/terminal-negative/inconsistent) classifies correctly, spans the right window, and carries exactly one marker with failure taking priority over stage; no-churn holds on an unchanged re-projection; two adjacent-window records and an LCO+SOAR pair each resolve independently; group links and existing campaign attribution survive a projection; RUN:/GEM:/blank-url events are never touched.
expected: Every lifecycle stage (queued/placed/observed/completed-no-block/terminal-negative/inconsistent) classifies correctly, spans the right window, and carries exactly one marker with failure taking priority over stage; no-churn holds on an unchanged re-projection; two adjacent-window records and an LCO+SOAR pair each resolve independently; group links and existing campaign attribution survive a projection; RUN:/GEM:/blank-url events are never touched.
result: pass
source: automated
coverage_id: 34-01/D2
covered_by: solsys_code/tests/test_observation_projector.py#TestStageFor, TestTitleAndToken, TestEventFieldsFor, TestProjectRecordWrites, TestMetaLinks, TestNamespaceIsolation

### 6. [34-01 D3] Group membership changes (add/remove/clear/reverse-direction, Gemini-skip) and record deletion (own-event delete, RUN:-namespace safety, no-companion-row safety) are wired and never raise out of the caller's operation, even when the projector itself is made to raise; raw=True saves, QuerySet.update(), and a rolled-back transaction each correctly bypass or undo projection; no network call is ever made during a save.
expected: Group membership changes (add/remove/clear/reverse-direction, Gemini-skip) and record deletion (own-event delete, RUN:-namespace safety, no-companion-row safety) are wired and never raise out of the caller's operation, even when the projector itself is made to raise; raw=True saves, QuerySet.update(), and a rolled-back transaction each correctly bypass or undo projection; no network call is ever made during a save.
result: pass
source: automated
coverage_id: 34-01/D3
covered_by: solsys_code/tests/test_observation_projector_signals.py#TestGroupMembershipReceiver, TestRecordDeleteReceiver, TestReceiverSafetyContract

### 7. [34-01 D4] The whole pre-existing test suite (1069 + 40 tests across solsys_code) still passes with all three receivers live in every test's fixture-creation path, including the three test files whose pre-existing assertions were affected by the new global signal.
expected: The whole pre-existing test suite (1069 + 40 tests across solsys_code) still passes with all three receivers live in every test's fixture-creation path, including the three test files whose pre-existing assertions were affected by the new global signal.
result: pass
source: automated
coverage_id: 34-01/D4
covered_by: workflow.test_command (.planning/config.json) -- python manage.py test over every solsys_code test module

### 8. [34-02 D1] project_observation_calendar sweeps every LCO/SOAR ObservationRecord with zero required arguments, supports --proposal (exact-code, comma list)/--facility/--dry-run, isolates one bad record's failure from the rest of the sweep (reported on stderr, sweep still exits 0), and a second sweep over unchanged data reports created:0/updated:0 for every facility.
expected: project_observation_calendar sweeps every LCO/SOAR ObservationRecord with zero required arguments, supports --proposal (exact-code, comma list)/--facility/--dry-run, isolates one bad record's failure from the rest of the sweep (reported on stderr, sweep still exits 0), and a second sweep over unchanged data reports created:0/updated:0 for every facility.
result: pass
source: automated
coverage_id: 34-02/D1
covered_by: solsys_code/tests/test_project_observation_calendar.py#TestBareInvocationAndSummary, TestProposalAndFacilityFiltering, TestDryRun, TestFailureIsolation, TestNamespaceIsolation, TestProjectQuerysetOrdering

### 9. [34-02 D2] sync_lco_observation_calendar (command + its 38-test module + its registered command name) is retired; every one of its 38 tests is classified as already-covered, migrated (with a named destination test), or deliberately retired with a reason -- no behaviour silently dropped.
expected: sync_lco_observation_calendar (command + its 38-test module + its registered command name) is retired; every one of its 38 tests is classified as already-covered, migrated (with a named destination test), or deliberately retired with a reason -- no behaviour silently dropped.
result: pass
source: automated
coverage_id: 34-02/D2
covered_by: solsys_code/tests/test_calendar_utils.py#TestExtractInstrument; solsys_code/tests/test_observation_projector.py#TestFieldPopulation, TestProjectRecordWrites; solsys_code/tests/test_project_observation_calendar.py#TestProposalAndFacilityFiltering, TestBareInvocationAndSummary; grep -rn 'sync_lco_observation_calendar' --include='*.py' solsys_code/ src/ returns no matches; django.core.management.get_commands() no longer lists 'sync_lco_observation_calendar'

### 10. [34-02 D3] PROJ-01 (one CalendarEvent per record) and PROJ-05 (no-churn, never writes outside its own namespace) hold with the sweep as backstop live: the sweep's own RUN:/GEM:/blank-url namespace-isolation test and the no-churn second-sweep test close the loop 34-01 opened for these two requirements, which this plan also declares (shared-ID gate).
expected: PROJ-01 (one CalendarEvent per record) and PROJ-05 (no-churn, never writes outside its own namespace) hold with the sweep as backstop live: the sweep's own RUN:/GEM:/blank-url namespace-isolation test and the no-churn second-sweep test close the loop 34-01 opened for these two requirements, which this plan also declares (shared-ID gate).
result: pass
source: automated
coverage_id: 34-02/D3
covered_by: solsys_code/tests/test_project_observation_calendar.py#TestNamespaceIsolation.test_run_gem_and_blank_url_events_are_untouched_by_a_sweep, TestBareInvocationAndSummary.test_second_sweep_over_unchanged_data_reports_created_zero_updated_zero

### 11. [34-02 D4] Once a record reaches a successful terminal state, the observation projector resolves its observed telescope exactly once (site_lookups: 1 on the first sweep, 0 on every sweep after), stores it under generic ObservationRecord.parameters keys, titles the event with the observed token ([O] FTN/FTS/SOAR/SITE-aperture), and campaign_attribution's telescope-match signal still scores a site-level match for the three renamed labels instead of degrading to aperture-only.
expected: Once a record reaches a successful terminal state, the observation projector resolves its observed telescope exactly once (site_lookups: 1 on the first sweep, 0 on every sweep after), stores it under generic ObservationRecord.parameters keys, titles the event with the observed token ([O] FTN/FTS/SOAR/SITE-aperture), and campaign_attribution's telescope-match signal still scores a site-level match for the three renamed labels instead of degrading to aperture-only.
result: pass
source: automated
coverage_id: 34-02/D4
covered_by: solsys_code/tests/test_observation_projector.py#TestObservedToken; solsys_code/tests/test_project_observation_calendar.py#TestObservedSiteLookup; solsys_code/tests/test_calendar_utils.py#test_telescope_01_d07_renamed_observed_telescope_labels; solsys_code/tests/test_campaign_attribution.py#test_telescope_match_d07_renamed_observed_labels_still_resolve_site_level

### 12. [34-02 D5] The whole project test suite still passes with the new sweep command, the renamed telescope labels, and the retired sync command all live at once -- no regression anywhere else in the codebase.
expected: The whole project test suite still passes with the new sweep command, the renamed telescope labels, and the retired sync command all live at once -- no regression anywhere else in the codebase.
result: pass
source: automated
coverage_id: 34-02/D5
covered_by: workflow.test_command (.planning/config.json) -- 1070 + 40 tests, both runs OK

### 13. [34-03 D1] Every projector marker paints the right status ring in a month cell ([Q]/[X]/[C]/[F]/[?] get a ring, [S]/[O] stay ring-free), every pre-existing bracket-word ring is untouched, and a visitor can read the vocabulary off a fixed legend rendered on the calendar page itself.
expected: Every projector marker paints the right status ring in a month cell ([Q]/[X]/[C]/[F]/[?] get a ring, [S]/[O] stay ring-free), every pre-existing bracket-word ring is untouched, and a visitor can read the vocabulary off a fixed legend rendered on the calendar page itself.
result: pass
source: automated
coverage_id: 34-03/D1
covered_by: solsys_code/tests/test_calendar_display_extras.py#TestProjectorMarkerRings, TestObservationStatusLegend; solsys_code/tests/test_calendar_template.py#CalendarStatusLegendRenderTest.test_calendar_page_renders_every_legend_marker_and_label

### 14. [34-03 D2] A grouped observation record's event modal shows which night of how many it is, the group's name, and links back to the group list and the record's own detail page -- rendered at request time from CalendarEventMeta.observation_group/.observation_record, numbered by window start (not pk or insertion order), with an unprojectable sibling sorting last rather than raising, and returning None for every documented empty-input edge (no companion row, no group, single-member group, non-CalendarEvent argument).
expected: A grouped observation record's event modal shows which night of how many it is, the group's name, and links back to the group list and the record's own detail page -- rendered at request time from CalendarEventMeta.observation_group/.observation_record, numbered by window start (not pk or insertion order), with an unprojectable sibling sorting last rather than raising, and returning None for every documented empty-input edge (no companion row, no group, single-member group, non-CalendarEvent argument).
result: pass
source: automated
coverage_id: 34-03/D2
covered_by: solsys_code/tests/test_calendar_display_extras.py#TestObservationSeriesDecoration; solsys_code/tests/test_calendar_template.py#EventModalSeriesDecorationTest

### 15. [34-03 D3] Series identity is never written into the event's own title/description -- projecting a record, rendering the modal, and re-projecting the record leaves event.title and event.description byte-identical -- and the campaign and series decorations render side by side in the same modal without either overwriting the other.
expected: Series identity is never written into the event's own title/description -- projecting a record, rendering the modal, and re-projecting the record leaves event.title and event.description byte-identical -- and the campaign and series decorations render side by side in the same modal without either overwriting the other.
result: pass
source: automated
coverage_id: 34-03/D3
covered_by: solsys_code/tests/test_calendar_display_extras.py#TestObservationSeriesDecoration.test_render_then_reproject_leaves_title_and_description_byte_identical; solsys_code/tests/test_calendar_template.py#EventModalSeriesDecorationTest.test_grouped_and_attributed_event_shows_both_decorations

### 16. [34-03 D4] Rendering a month of grouped observation events issues no per-event query fan-out for the new tag: fomo_render_calendar's Prefetch was widened to select observation_record__target and observation_group, and a second grouped event does not add a query to the month view.
expected: Rendering a month of grouped observation events issues no per-event query fan-out for the new tag: fomo_render_calendar's Prefetch was widened to select observation_record__target and observation_group, and a second grouped event does not add a query to the month view.
result: pass
source: automated
coverage_id: 34-03/D4
covered_by: solsys_code/tests/test_calendar_template.py#EventModalSeriesDecorationTest.test_month_view_query_count_does_not_grow_with_second_grouped_event

### 17. [34-03 D5] The month-cell truncatechars:18/:16 budget (PROJ-06) is unchanged, no existing status_border_css() assertion was modified, and the whole pre-existing project test suite plus both ruff gates stay green with the new marker vocabulary, legend and series decoration live.
expected: The month-cell truncatechars:18/:16 budget (PROJ-06) is unchanged, no existing status_border_css() assertion was modified, and the whole pre-existing project test suite plus both ruff gates stay green with the new marker vocabulary, legend and series decoration live.
result: pass
source: automated
coverage_id: 34-03/D5
covered_by: workflow.test_command (.planning/config.json) -- full solsys_code + observatory suite, exit 0; pre-commit run ruff --all-files && pre-commit run ruff-format --all-files

### 18. [34-04 D1] project_observation_calendar_demo.ipynb, executed against the real developer database, demonstrates the receiver narrowing one throwaway record's event through [Q]->[S]->[O] with no command run (rolled back, confirmed absent afterward), then the real one-time takeover: 156 legacy facility-url-keyed events re-titled, 3 new events created, and the RUN:/GEM:/blank-url families proven byte-identical (72/0/10 events, before == after) via an explicit assertion.
expected: project_observation_calendar_demo.ipynb, executed against the real developer database, demonstrates the receiver narrowing one throwaway record's event through [Q]->[S]->[O] with no command run (rolled back, confirmed absent afterward), then the real one-time takeover: 156 legacy facility-url-keyed events re-titled, 3 new events created, and the RUN:/GEM:/blank-url families proven byte-identical (72/0/10 events, before == after) via an explicit assertion.
result: pass
source: automated
coverage_id: 34-04/D1
covered_by: docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb -- executed cell outputs (cells 4, 8, 9, 10), real database, no mocks

### 19. [34-04 D2] The second sweep over the now-converged corpus reports created: 0, updated: 0, site_lookups: 0 for every facility (LCO and SOAR), printed side by side with the first sweep's non-zero created/updated/site_lookups line, with an explicit assertion; the per-corpus reconciliation cell shows 159 LCO/SOAR records against 159 facility-url-keyed events with zero unprojectable rows, and the per-marker tally sums to 159 across all seven markers.
expected: The second sweep over the now-converged corpus reports created: 0, updated: 0, site_lookups: 0 for every facility (LCO and SOAR), printed side by side with the first sweep's non-zero created/updated/site_lookups line, with an explicit assertion; the per-corpus reconciliation cell shows 159 LCO/SOAR records against 159 facility-url-keyed events with zero unprojectable rows, and the per-marker tally sums to 159 across all seven markers.
result: pass
source: automated
coverage_id: 34-04/D2
covered_by: docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb -- executed cell outputs (cells 10, 12, 13)

### 20. [34-04 D3] sync_lco_observation_calendar_demo.ipynb is deleted (git rm), its docs/notebooks.rst toctree entry and CLAUDE.md notebook-map pairing both replaced with project_observation_calendar_demo.ipynb's pairing, and a grep for the retired command name returns 0 across docs/runbooks/telescope_runs_calendar.rst, docs/notebooks.rst, and CLAUDE.md.
expected: sync_lco_observation_calendar_demo.ipynb is deleted (git rm), its docs/notebooks.rst toctree entry and CLAUDE.md notebook-map pairing both replaced with project_observation_calendar_demo.ipynb's pairing, and a grep for the retired command name returns 0 across docs/runbooks/telescope_runs_calendar.rst, docs/notebooks.rst, and CLAUDE.md.
result: pass
source: automated
coverage_id: 34-04/D3
covered_by: grep -rc 'sync_lco_observation_calendar' docs/runbooks/telescope_runs_calendar.rst docs/notebooks.rst CLAUDE.md -- 0 for every file

### 21. [34-04 D4] The runbook's LCO/SOAR section is replaced with a projector-and-sweep narrative (marker table, on-page legend/ring documentation, Observation series subsection, One-time title change note, a 'When would I run the sweep?' subsection with real flags and counter meanings), the cheat-sheet and troubleshooting sections are updated to the real vocabulary (unprojectable/site_lookups/site_lookup_failed, no [UNVERIFIED]/telescope_api_failed), and the Gemini section plus its demo notebook both state the no-read-back caveat and that the projector ignores Gemini records by design.
expected: The runbook's LCO/SOAR section is replaced with a projector-and-sweep narrative (marker table, on-page legend/ring documentation, Observation series subsection, One-time title change note, a 'When would I run the sweep?' subsection with real flags and counter meanings), the cheat-sheet and troubleshooting sections are updated to the real vocabulary (unprojectable/site_lookups/site_lookup_failed, no [UNVERIFIED]/telescope_api_failed), and the Gemini section plus its demo notebook both state the no-read-back caveat and that the projector ignores Gemini records by design.
result: pass
source: automated
coverage_id: 34-04/D4
covered_by: grep-based verify commands over docs/runbooks/telescope_runs_calendar.rst and docs/notebooks/pre_executed/sync_gemini_observation_calendar_demo.ipynb (see plan 34-04 Task 2 <verify>), all passing; sphinx-build and pre-commit ruff/ruff-format green

### 22. [34-05 D1] coerce_schedule_datetime() coerces datetime/str/None schedule values to aware UTC, raising ValueError on anything unusable
expected: coerce_schedule_datetime() coerces datetime/str/None schedule values to aware UTC, raising ValueError on anything unusable
result: pass
source: automated
coverage_id: 34-05/D1
covered_by: solsys_code/tests/test_calendar_utils.py#TestCoerceScheduleDatetime (8 tests: trailing-Z, +00:00, non-UTC offset, naive-as-UTC, aware-passthrough, naive-gets-UTC, None, unparseable-raises)

### 23. [34-05 D2] record_time_window() returns the aware-UTC pair for an in-memory record whose schedule fields hold portal ISO strings, matching a DB-fetched record's result
expected: record_time_window() returns the aware-UTC pair for an in-memory record whose schedule fields hold portal ISO strings, matching a DB-fetched record's result
result: pass
source: automated
coverage_id: 34-05/D2
covered_by: solsys_code/tests/test_calendar_utils.py#TestRecordTimeWindow.test_in_memory_instance_with_portal_iso_strings_returns_aware_utc_pair

### 24. [34-05 D3] A real update_observation_status() save whose payload carries portal ISO strings narrows the record's own CalendarEvent in place with no unprojectable warning (G-34-2 closed at the unit/integration level)
expected: A real update_observation_status() save whose payload carries portal ISO strings narrows the record's own CalendarEvent in place with no unprojectable warning (G-34-2 closed at the unit/integration level)
result: pass
source: automated
coverage_id: 34-05/D3
covered_by: solsys_code/tests/test_observation_projector_signals.py#TestUpdateObservationStatusPath.test_updatestatus_narrows_the_event_with_no_command_run; solsys_code/tests/test_observation_projector_signals.py#TestUpdateObservationStatusPath.test_updatestatus_with_datetime_valued_facility_still_narrows_the_event; solsys_code/tests/test_observation_projector_signals.py#TestUpdateObservationStatusPath.test_updatestatus_event_span_matches_the_reloaded_record_no_churn

### 25. [34-05 D4] The receiver never raises out of a save: project_record()'s existing catch is untouched and observation_projector.py is byte-identical to before this plan
expected: The receiver never raises out of a save: project_record()'s existing catch is untouched and observation_projector.py is byte-identical to before this plan
result: pass
source: automated
coverage_id: 34-05/D4
covered_by: git diff 7877a2e04ea57d4d4e41eb2bf3ea439718c144bd -- solsys_code/observation_projector.py (empty)

### 26. [34-06 D1] project_observation_calendar_demo.ipynb documents and demonstrates the G-34-2 fix with real executed output: the receiver-demo cell assigns scheduled_start/scheduled_end as the portal's own ISO-8601 strings and the event still narrows
expected: project_observation_calendar_demo.ipynb documents and demonstrates the G-34-2 fix with real executed output: the receiver-demo cell assigns scheduled_start/scheduled_end as the portal's own ISO-8601 strings and the event still narrows
result: pass
source: automated
coverage_id: 34-06/D1
covered_by: notebook cell 4 output (this plan's re-execution): '2. After a schedule-only save ... assigned scheduled_start=\\'2026-09-20T03:00:00Z\\' scheduled_end=\\'2026-09-20T05:00:00Z\\' ... title=\\'[S] 2m0 observation-projector-demo-target\\''

### 27. [34-06 D2] The notebook is re-executable against a scratch copy of the developer database without writing to src/fomo_db.sqlite3 or overwriting the committed SCHED-06 baseline JSON, and names which database the committed run used
expected: The notebook is re-executable against a scratch copy of the developer database without writing to src/fomo_db.sqlite3 or overwriting the committed SCHED-06 baseline JSON, and names which database the committed run used
result: pass
source: automated
coverage_id: 34-06/D2
covered_by: test \"$(git status --porcelain docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json src/fomo_db.sqlite3 | wc -l)\" -eq 0 (Task 1 automated verify); notebook cell 2 output: 'Resolved database: .../tmp/fomo_g34_2_copy.sqlite3 -- routed to a scratch copy, not the developer database.'

### 28. [34-06 D3] The 33 stale LCO events remain on the developer database, untouched by this plan, available for the operator's next real updatestatus run to repair through the receiver (SCHED-06/UAT Test 4 evidence)
expected: The 33 stale LCO events remain on the developer database, untouched by this plan, available for the operator's next real updatestatus run to repair through the receiver (SCHED-06/UAT Test 4 evidence)
result: pass
source: automated
coverage_id: 34-06/D3
covered_by: src/fomo_db.sqlite3 mtime/size unchanged (1789157248 / 1232896 bytes, matching the value recorded at dispatch) and git status --porcelain src/fomo_db.sqlite3 empty, checked before and after every step in this plan

### 29. [34-06 D4] A real, portal-backed updatestatus run against a copy of the developer database logs zero unprojectable lines, and the following dry-run sweep reports updated: 0, unprojectable: 0 for LCO -- the direct, live proof G-34-2 is closed
expected: A real, portal-backed updatestatus run against a copy of the developer database logs zero unprojectable lines, and the following dry-run sweep reports updated: 0, unprojectable: 0 for LCO -- the direct, live proof G-34-2 is closed
result: pass
source: automated
coverage_id: 34-06/D4
covered_by: grep -c 'unprojectable' tmp/34-06-updatestatus.txt == 0; grep -o 'LCO: [^|]*' tmp/34-06-dry-run.txt | tail -1 matches 'updated: 0,' and 'unprojectable: 0' (Task 2 automated verifies)

### 30. [34-06 D5] The receiver never raises out of a real save (TRIG-02) -- updatestatus completed successfully with zero AttributeError/unprojectable lines over 52 non-terminal LCO records
expected: The receiver never raises out of a real save (TRIG-02) -- updatestatus completed successfully with zero AttributeError/unprojectable lines over 52 non-terminal LCO records
result: pass
source: automated
coverage_id: 34-06/D5
covered_by: tmp/34-06-updatestatus.txt: 'Update completed successfully', 0 AttributeError, 0 OperationalError, 0 unprojectable lines

### 31. [34-07 D1] The committed notebook's takeover cell (05528b38) reports a non-zero re-titled count with sample before -> after title pairs, proving a real takeover on a scratch copy cloned from an un-swept database
expected: The committed notebook's takeover cell (05528b38) reports a non-zero re-titled count with sample before -> after title pairs, proving a real takeover on a scratch copy cloned from an un-swept database
result: pass
source: automated
coverage_id: 34-07/D1
covered_by: notebook cell 05528b38 committed output: '33 of 159 pre-existing facility-url-keyed events were re-titled by the takeover.' plus 8 sample before/after pairs; solsys_code/tests/test_projector_demo_notebook.py#TestProjectorDemoNotebookEvidence.test_scratch_routed_run_shows_a_real_takeover_and_diverging_sweeps

### 32. [34-07 D2] The notebook's first and second sweep summary lines differ, with the second all-zero per facility segment, and both cells now assert this instead of silently allowing a vacuous re-execution
expected: The notebook's first and second sweep summary lines differ, with the second all-zero per facility segment, and both cells now assert this instead of silently allowing a vacuous re-execution
result: pass
source: automated
coverage_id: 34-07/D2
covered_by: notebook cell 556d2a9f committed output: 'First sweep work (created + updated, every facility): 33'; First/Second sweep lines differ; every facility segment of Second sweep reports created: 0, updated: 0, site_lookups: 0; solsys_code/tests/test_projector_demo_notebook.py#TestProjectorDemoNotebookEvidence.test_second_sweep_reports_zero_per_facility_segment

### 33. [34-07 D3] A scratch-routed re-execution that takes nothing over fails loudly: cells 05528b38 and 556d2a9f each raise when routed to a copy that already converged
expected: A scratch-routed re-execution that takes nothing over fails loudly: cells 05528b38 and 556d2a9f each raise when routed to a copy that already converged
result: pass
source: automated
coverage_id: 34-07/D3
covered_by: Code read of both cells' new guard blocks (scratch_routed branch asserts changed_keys/changed_titles/first_sweep_work non-empty and first_sweep_summary != second_sweep_summary); the real re-execution never had to hit this path since the fresh clone had genuine work (verified: no assertion raised, real evidence produced)

### 34. [34-07 D4] The notebook's own prose (markdown cells 8eeddc83, 6a9bd576, 7e7bd66e) matches its own output and states the un-swept-clone rule; the closing table (35debc54) carries this run's numbers in its PROJ-05/TRIG-03 evidence entries
expected: The notebook's own prose (markdown cells 8eeddc83, 6a9bd576, 7e7bd66e) matches its own output and states the un-swept-clone rule; the closing table (35debc54) carries this run's numbers in its PROJ-05/TRIG-03 evidence entries
result: pass
source: automated
coverage_id: 34-07/D4
covered_by: notebook cell 8eeddc83 quotes 'LCO created: 0, updated: 33' and '33 of 159'; cell 35debc54 committed output: 'PROJ-05 ... 33 of 159 facility-url-keyed events re-titled this run' and 'TRIG-03 ... first sweep created+updated total 33; second sweep reported all zeros for every facility'

### 35. [34-07 D5] The SCHED-06 baseline JSON stays byte-identical, and src/fomo_db.sqlite3 is unmodified by this plan
expected: The SCHED-06 baseline JSON stays byte-identical, and src/fomo_db.sqlite3 is unmodified by this plan
result: pass
source: automated
coverage_id: 34-07/D5
covered_by: git status --porcelain docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json (empty, checked before/after both nbconvert runs and after all commits); stat -c '%Y %s' src/fomo_db.sqlite3 == 1789157248 1232896 unchanged throughout (tmp/34-07-devdb-stamp.txt)

### 36. [34-07 D6] The notebook's evidence is checkable without executing the notebook: a repo-level test passes against the committed artifact and fails against a copy whose takeover evidence has been emptied
expected: The notebook's evidence is checkable without executing the notebook: a repo-level test passes against the committed artifact and fails against a copy whose takeover evidence has been emptied
result: pass
source: automated
coverage_id: 34-07/D6
covered_by: python manage.py test solsys_code.tests.test_projector_demo_notebook (6 tests, 5 pass + 1 skip, OK); FOMO_DEMO_NOTEBOOK_PATH pointed at an emptied copy fails (setUpClass FileNotFoundError without a baseline sibling; AssertionError on the re-titled-count check when a baseline sibling is present -- see key-decisions)

### 37. [34-07 D7] docs/runbooks/telescope_runs_calendar.rst is checked against this change and the outcome recorded (no edit needed -- this plan changes no operator-visible behaviour)
expected: docs/runbooks/telescope_runs_calendar.rst is checked against this change and the outcome recorded (no edit needed -- this plan changes no operator-visible behaviour)
result: pass
source: automated
coverage_id: 34-07/D7
covered_by: grep -n -E 'updatestatus|post_save|unprojectable' docs/runbooks/telescope_runs_calendar.rst -- lines 51/55/147/171-196/1145 read and confirmed to describe post-fix behaviour already (34-06's determination, re-confirmed here)

## SCHED-06: live narrowing over real observing nights

Spike 004 left SCHED-06 as a PARTIAL verdict -- the projector and sweep exist and are
tested, but nothing had yet proven a real, pending `KEY2026B-004` record narrow on the
calendar purely from `python manage.py updatestatus`, with no sweep run in between.
Plan 34-04 Task 1 captured the baseline this section tracks; the verdict closes only
when the re-check below is filled in. (Carried forward from the previous UAT session,
commit 84e8936.)

### Baseline

- **Captured at:** 2026-09-11T04:44:59.526430+00:00 (UTC)
- **Proposal:** `KEY2026B-004`
- **Pending record count at baseline:** 74 (56 `queued`, 18 `placed`)
- **Baseline artifact:** `docs/notebooks/pre_executed/project_observation_calendar_demo.sched06-baseline.json`
  (per-record `observation_id`, target name, status, `scheduled_start`/`scheduled_end`,
  and the projected event's start/end/title, all as of the baseline capture)
- **Notebook cell:** `project_observation_calendar_demo.ipynb`, "SCHED-06 baseline: the
  pending `KEY2026B-004` records (D-20)"
- **Standing evidence (corrected 2026-09-12):** the earlier premise that the 33 stale LCO events would
  all repair themselves under `updatestatus` alone was wrong for terminal records --
  `updatestatus` skips terminal states, so the 14 that completed under the pre-fix receiver
  never get re-saved (F-34-1). The 19 non-terminal ones did repair on the 2026-09-12 run.
  An un-routed re-execution rewrites the baseline JSON in place by design (that is what
  `git diff` reads).

### The rule between baseline and re-check

Over the coming nights, run **only**:

```console
$ python manage.py updatestatus
```

Nothing else -- and specifically **not** `python manage.py project_observation_calendar`
(the sweep). Running the sweep in between would let a bulk write path narrow the
records instead of the `post_save` receiver alone, which is the opposite of what
SCHED-06 needs proven. If the sweep is run for any other operational reason before the
re-check below happens, note it in the table and treat the verdict as still open rather
than closed by a mixed cause.

### Dated re-check table

Fill in one row per re-check. A re-check re-executes
`project_observation_calendar_demo.ipynb`
(`jupyter nbconvert --to notebook --execute --inplace
docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb`) and diffs its
SCHED-06 section against the baseline JSON above -- committing the re-executed notebook
each time. Record the re-execution's first sweep summary line in the Notes column.

| Date | Nights watched (updatestatus only?) | Records narrowed queued->placed | Records narrowed placed->observed | Notes | Verdict |
|------|--------------------------------------|----------------------------------|-------------------------------------|-------|---------|
| 2026-09-12 | 1 night since baseline (2026-09-11T04:44Z -> 2026-09-12T22:10Z); updatestatus only -- the sweep has never run on src/fomo_db.sqlite3 | 0 | 2 (queued -> observed directly: 4378046 10P, 4378332 11P) | Evidence gathered from scratch copies, not yet from the notebook re-execution. Dry-run sweep on a fresh copy: `LCO: created: 0, updated: 14, unchanged: 145` -- the 14 are the pre-fix legacy residue (F-34-1), none of them the narrowed records. Notebook re-execution + baseline diff still to be committed (paired docs). | CLOSED (receiver narrowed real records with no command run) |

**Current verdict: CLOSED (2026-09-12).** Two `KEY2026B-004` records narrowed with nothing but
`updatestatus` run in between. Remaining paired-docs step: re-execute the notebook un-routed,
commit it, and read its first sweep line per record (the legacy residue in F-34-1 will show
as `updated: 16`, which is expected and not a narrowing writer).

## Summary

total: 37
passed: 37
issues: 0
pending: 0
skipped: 0
blocked: 0

## Deferred Follow-Ups

- test: 2
  idea: "Line 63 (runbook) / projector stage semantics: distinguish window_expired/record expiration (keep the ORIGINAL window, as today) from failed or aborted records, which should keep the last scheduled / partly executed window to aid failure diagnosis and reporting."
  deferred_at: 2026-09-12
- test: 2
  idea: "Lines 185-188 (runbook) / sweep summary line: make the --dry-run site_lookups value visibly 'not attempted' rather than 0. Reviewer floated -1/-99; a textual marker such as `site_lookups: n/a (dry run)` keeps the counters summable and the dry-run-vs-real agreement property intact."
  deferred_at: 2026-09-12
- test: 3
  idea: "Paired docs (CLAUDE.md): re-execute docs/notebooks/pre_executed/project_observation_calendar_demo.ipynb UN-ROUTED against src/fomo_db.sqlite3, commit it with the rewritten sched06-baseline.json, and read the first sweep line per record. The notebook's own sweeps will also repair the 14 legacy events (F-34-1); if the notebook is not re-run soon, run `python manage.py project_observation_calendar` once instead."
  deferred_at: 2026-09-12
- test: 1
  idea: "Runbook/verification wording: 34-06 SUMMARY, 34-VERIFICATION.md and the runbook's sweep section assume the next updatestatus run repairs every stale event; state instead that updatestatus never re-saves terminal-state records (tom_observations facility.py:573), so events that went stale on a terminal transition need one sweep."
  deferred_at: 2026-09-12
- reported_verbatim: "for line 63, we might want to distinguish between window_expired/record expiration, which should have the original window, as stated and ones that fail or aborted, which should probably keep the last scheduled/partly executed window to aid in failure diagnosis/reporting. For lines 185-188, is there mileage in reporting '-1' or '-99' rather than '0' in the case of '--dry-run' ? Lines 1144-1155 look good"

## Operational Findings

- finding_id: F-34-1
  title: "14 legacy-stale LCO events on the developer database that updatestatus can never repair"
  detail: "Records 4378021-4378025 (220P), 4378038-4378040, 4378042-4378045 (10P), 4378323, 4378331 (11P) -- all KEY2026B-004 -- reached COMPLETED at 2026-09-11 20:02-20:04 UTC under the pre-34-05 receiver, so their events still read [S]/[Q] on the old windows. update_all_observation_statuses() excludes terminal states, so no updatestatus run re-saves them; one sweep (or the notebook's own sweeps) repairs them. Not a receiver defect; not caused by interleaving. Discovered while diagnosing the Test 1 result on 2026-09-12."
  action: "Covered by the Deferred Follow-Ups above (notebook re-execution or a single sweep)."

## Gaps

[none -- G-34-1 (raised from Test 1 on 2026-09-12) was withdrawn the same day after inspection showed the 14 mismatches predate both interleaved runs; see F-34-1. Previous-session gaps G-34-2 and G-34-3 were closed and verified before this restart (commit 84e8936).]
