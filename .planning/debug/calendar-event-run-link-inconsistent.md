---
status: diagnosed
trigger: "calendar-event-run-link-inconsistent: Two companion calendar events for the same underlying instrument class are inconsistently linked to their owning CampaignRun via CalendarEventMeta.run — one has the link, one doesn't."
created: 2026-08-06T15:51:34Z
updated: 2026-08-06T16:20:00Z
---

## Current Focus

hypothesis: CONFIRMED (revised from initial framing). The two 2026-07-16 events are not "companion events of the same run" at all — they are two structurally different CalendarEvent rows from two independent, never-deduplicated ingestion pipelines: (1) pk=59, a raw `sync_lco_observation_calendar`-synced event (url=a real observe.lco.global request URL, telescope='2m0', instrument='2M0-SCICAM-MUSCAT') that has never been attributed to any CampaignRun; (2) pk=72, a Phase 29 reconciler-owned per-night event (url='RUN:1:2026-07-16') for CampaignRun pk=1 ("Didymos 2026", telescope_instrument='FTS/MuSCAT4'), which the reconciler auto-links via `_link_event_to_run()` on every create/update. pk=59 is not a reconciler bug -- it is an un-confirmed orphan already sitting in Phase 28's attribution queue as a HIGH-band (score 0.8) candidate for CampaignRun pk=1, verified directly via `candidates_for_event()`.
test: Queried the real dev DB (python manage.py shell against src/fomo_db.sqlite3) for both events' CalendarEventMeta rows, CampaignRun pk=1's fields, and ran `campaign_attribution.candidates_for_event()` on the unlinked orphan.
expecting: n/a — hypothesis confirmed by direct DB query, not further testing needed.
next_action: none — diagnosis complete, goal is find_root_cause_only; returning ROOT CAUSE FOUND without applying a fix.

## Symptoms

expected: Companion calendar events for the same instrument class are consistently linked to their owning CampaignRun (per Phase 27 WR-03 / Phase 29 reconciler behavior).
actual: User reported (verbatim): "yes the '[EXPIRED] 2m0 2M0-SCICAM-MUSCAT' calendar event doesn't have a link to the CampaignRun but the other one for FTS/MuSCAT4 (which is the same as 2m0 2M0-SCICAM-MUSCAT; FTS is a '2m0', MuSCAT4 is an instance of a 2M0-SCICAM-MUSCAT class of instruments) does have the link" — specifically the 2026-07-16 event titled "[EXPIRED] 2m0 2M0-SCICAM-MUSCAT" (a telescope-CLASS-level label, no linked CampaignRun) versus an FTS/MuSCAT4 event (a specific telescope INSTANCE of that same class, which IS linked).
errors: None reported
reproduction: Test 9 in .planning/phases/27-the-canonical-run-record/27-UAT.md. Open /calendar/, navigate to July 2026, click the 2026-07-16 event titled "[EXPIRED] 2m0 2M0-SCICAM-MUSCAT" — the modal partial (src/templates/tom_calendar/partials/event_form.html) shows no run link. Compare against the FTS/MuSCAT4 event, which does show a run link.
started: Discovered during a UAT re-verification pass on 2026-08-06, while confirming an earlier fix (commit 6b9aab3, converting the leading multi-line {# #} template comment in event_form.html to {% comment %}) actually cleared the leaked-template-source defect in the modal.

## Eliminated

- hypothesis: "The reconciler's window-shape dispatch (container vs classical-nights branch) only calls `_link_event_to_run()` for one branch, leaving class-level runs unlinked."
  evidence: "Read campaign_reconciler.py in full: `_reconcile_container()` (line 278) and `_reconcile_classical_nights()` (line 421) BOTH call `_link_event_to_run(event, run)` unconditionally on every create/update. There is no code path in reconcile_run() where an event it creates or updates is left unlinked. The 260805-tad quick task (STATE.md) already removed the one dispatch bug of this general shape (QUEUE_SOURCES branch)."
  timestamp: 2026-08-06T16:05:00Z

- hypothesis: "The two July-16 events are two different CampaignRun rows (class-wide telescope_class='2m0' allocation vs. a site-resolved FTS/MuSCAT4 instance run), per CLAUDE.md's class-wide-campaign-keeps-site=None-forever memory."
  evidence: "Queried CampaignRun pk=1 directly: telescope_class='' (blank), site=E10 (Siding Spring-Faulkes Telescope South, resolved), telescope_instrument='FTS/MuSCAT4'. There is only ONE relevant CampaignRun (pk=1) in play, not two. The '2m0 2M0-SCICAM-MUSCAT' unlinked event (pk=59) is not itself a CampaignRun or tied to any class-wide run at all -- it is a raw, run-less CalendarEvent from the LCO sync pipeline that has simply never been attributed to CampaignRun pk=1."
  timestamp: 2026-08-06T16:10:00Z

## Evidence

- timestamp: 2026-08-06T15:58:00Z
  checked: "grep for '[EXPIRED]' and '2M0-SCICAM-MUSCAT' across solsys_code/"
  found: "'[EXPIRED] 2m0 2M0-SCICAM-MUSCAT' is the title format produced by sync_lco_observation_calendar.py's WINDOW_EXPIRED status prefix ('{prefix} {telescope} {instrument}'), NOT anything in campaign_reconciler.py's RUN_STATUS_CALENDAR_PREFIX (which only has [CANCELLED]/[WEATHERED]). test_admin.py's fixture data confirms 7 of the 11 near-identical companion CalendarEventMeta rows share this exact title (same finding as the 27-UAT gap re: illegible admin picker)."
  implication: "The unlinked '[EXPIRED] 2m0 2M0-SCICAM-MUSCAT' event is a raw LCO-sync-derived event, not a reconciler-derived one."

- timestamp: 2026-08-06T16:12:00Z
  checked: "campaign_reconciler.py in full (reconcile_run, _reconcile_container, _reconcile_classical_nights, _link_event_to_run, _skip_reason)"
  found: "Both branches of reconcile_run() (class-wide/satellite container branch AND the per-night classical branch) call _link_event_to_run(event, run) unconditionally on every create or update. _link_event_to_run() get_or_creates a CalendarEventMeta and sets .run = run whenever it differs. There is no telescope_instrument-format check (class-level vs instance-level) anywhere in the dispatch logic -- dispatch is purely on run.telescope_class / run.site.observations_type."
  implication: "Reconciler-owned events are always linked; the dispatch-gap hypothesis (a) from the task instructions is disproven."

- timestamp: 2026-08-06T16:22:00Z
  checked: "Real dev DB (src/fomo_db.sqlite3, queried via python manage.py shell with DATABASES overridden -- this worktree has no DB file of its own) for CalendarEvent rows matching '2M0-SCICAM-MUSCAT' and 'FTS|MuSCAT4', their CalendarEventMeta companions, and CampaignRun pk=1"
  found: |
    pk=59: title='[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', url='https://observe.lco.global/requests/4269958' (real LCO sync URL, not RUN:-namespaced), start=2026-07-16 00:00 UTC, telescope='2m0', instrument='2M0-SCICAM-MUSCAT', target_list_id=2. CalendarEventMeta pk=59 exists (is_verified=True) but run_id=None.
    pk=72: title='Didymos 2026: FTS/MuSCAT4 (window 2026-07-07..2026-07-21)', url='RUN:1:2026-07-16' (reconciler-owned key), start=2026-07-16 07:24 UTC, telescope='FTS/MuSCAT4', instrument='', target_list_id=2. CalendarEventMeta pk=72 has run_id=1.
    CampaignRun pk=1: campaign='Didymos 2026' (campaign_id=2), telescope_instrument='FTS/MuSCAT4', telescope_class='' (blank), site=E10 (resolved, not None), window 2026-07-07..2026-07-21, approval_status='approved', source='legacy'.
  implication: "pk=59 and pk=72 are two DIFFERENT CalendarEvent rows for the same real-world observing night, produced by two independent, never-deduplicated pipelines: raw LCO sync (pk=59) vs. the Phase 29 reconciler acting on CampaignRun pk=1 (pk=72). pk=72 is linked because the reconciler created/owns it. pk=59 is unlinked because nothing has ever attributed it to pk=1 -- there is no CampaignRun whose telescope_instrument is '2m0 2M0-SCICAM-MUSCAT'; that string is only ever a raw LCO instrument_type label."

- timestamp: 2026-08-06T16:25:00Z
  checked: "solsys_code.campaign_attribution.candidates_for_event(event pk=59) run against the real dev DB"
  found: "Returns exactly one candidate: run pk=1, score=0.8, band='high' (HIGH_BAND_MIN=0.75). date_evidence: orphan window 2026-07-16..2026-07-16 overlaps 100% of run window 2026-07-07..2026-07-21. instrument_evidence: orphan instrument '2M0-SCICAM-MUSCAT' vs run telescope/instrument 'FTS/MuSCAT4' -- tokenised similarity 0.92 (the muscat4/muscat token-pair case instrument_similarity()'s own docstring names as its worked example)."
  implication: "pk=59 is not stuck or broken -- it is a live, correctly-scored HIGH-band candidate already sitting in the Phase 28 attribution queue for exactly the run the user expected it to link to. It only needs a staff member to open the attribution queue and confirm it; nothing in the current code path does this automatically, and nothing does it wrongly."

- timestamp: 2026-08-06T16:28:00Z
  checked: "src/templates/tom_calendar/partials/event_form.html (full file) and docs/runbooks/telescope_runs_calendar.rst 'Why doesn't the calendar pop-up show a Campaign run block?' section"
  found: |
    event_form.html's run-link block (lines 112-149) renders only when event.telescope_label_meta.run is set and run.is_publicly_visible -- it renders NOTHING else when unset (no hint of a pending attribution candidate). Its own inline {% comment %} (lines 104-110) is now stale: it says 'no production code writes CalendarEventMeta.run yet ... deferred to the Phase 29 reconciler', but Phase 29 is complete and does write it.
    The runbook section (already correct, lines 585-609) explicitly documents this exact scenario: "every event the reconciler creates or adopts gets that link set automatically... The manual admin path below still exists, and remains the right tool for an event the reconciler never touches at all -- a load_telescope_runs- or sync-command-created event that has not (yet) been attributed to a run through the attribution queue."
  implication: "This is documented, by-design behavior, not a bug -- confirmed by the project's own runbook (which predates this UAT finding). The only real gap is UX: the modal for an unlinked event gives the user no indication that a HIGH-confidence attribution-queue candidate already exists for it, so the 'inconsistency' looks alarming even though it is expected and already resolvable in one click via /campaigns/attribution/ (or equivalent queue URL)."

## Resolution

root_cause: |
  NOT a linking bug in the Phase 29 reconciler or in CalendarEventMeta.run assignment logic. The two "companion" events the user compared are not companions of each other at all -- they are two independently-created CalendarEvent rows from two separate, never-deduplicated ingestion pipelines that happen to describe the same real-world 2026-07-16 FTS/MuSCAT4 observing night for CampaignRun pk=1 ("Didymos 2026"):
    1. pk=59 ("[EXPIRED] 2m0 2M0-SCICAM-MUSCAT") was created by sync_lco_observation_calendar directly from the raw LCO API request/instrument_type data. This pipeline has no concept of CampaignRun and never sets CalendarEventMeta.run -- linking an orphan like this to a run is a human-confirmed step via Phase 28's attribution queue (or the admin FK picker), which has simply not happened yet for this specific event. It is not stuck or silently dropped: candidates_for_event() already scores it as a HIGH-band (0.8) candidate for CampaignRun pk=1.
    2. pk=72 ("Didymos 2026: FTS/MuSCAT4 ...") was created and IS linked because it is a Phase 29 reconciler-owned event (url='RUN:1:2026-07-16'), derived independently from CampaignRun pk=1's own fields (telescope_instrument, window, site) via sun_event() dip-corrected sunset/sunrise math. Both of reconcile_run()'s branches (_reconcile_container and _reconcile_classical_nights) call _link_event_to_run() unconditionally on every event they create or update, so every reconciler-owned event is linked by construction -- there is no dispatch gap for class-level vs instance-level runs (confirmed by reading campaign_reconciler.py in full; the QUEUE_SOURCES-style dispatch bug this shape would resemble was already fixed by quick task 260805-tad).
  The runbook (docs/runbooks/telescope_runs_calendar.rst, "Why doesn't the calendar pop-up show a 'Campaign run' block?") already documents this exact two-pipeline split accurately. The genuine, unaddressed gap is UX/discoverability, not correctness: event_form.html's run-link block gives no hint, for an orphan event, that a specific high-confidence attribution-queue candidate already exists -- so an operator has no way to know, from the calendar modal alone, that "click Save in the attribution queue" is the one action needed to close this specific case. The template's own inline {% comment %} (WR-03, lines 104-110) is also now stale, since it still says "no production code writes CalendarEventMeta.run yet" post-Phase-29.
fix: (not applied -- goal is find_root_cause_only, diagnosis only)
verification: (n/a -- no fix applied)
files_changed: []
