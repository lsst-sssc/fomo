---
status: diagnosed
trigger: "G-35-4 observatory-edit-leaves-nights-stale: Correcting an Observatory row's lat/lon/altitude/timezone in place (without changing run.site) should cause the next reconcile sweep to re-mint already-projected allocation nights at that site with boundaries derived from the corrected coordinates, counted and logged, rather than leaving them silently stale with unchanged=1. Round-5 repro showed ReconcileResult(created=0, updated=0, unchanged=1, retired=0) with token v2|1|none|none unchanged before and after. Owner decided 2026-09-16 to fix in round 6. Mode: find_root_cause_only."
created: 2026-09-16T00:00:00Z
updated: 2026-09-16T00:00:00Z
---

## Current Focus
<!-- OVERWRITE on each update - always reflects NOW -->

hypothesis: |
  G-35-4 is ALREADY CLOSED at HEAD (5b0f431) by round-6 plan 35-24, INCLUDING the owner's
  >1-minute threshold, which pre-existed as `_UNRECORDED_PROVENANCE_TOLERANCE = timedelta(minutes=1)`
  (allocation_projector.py:77, introduced by CR-01/plan 35-19) and is REUSED by the new
  fingerprint fall-through. The UAT "issue" entry records the owner's DECISION ("fix in round 6"),
  not an outstanding defect — round 6 already executed. Remaining delta is confined to
  (a) the operator-facing warning text and (b) the by-design fully-set/confirmed exceptions.
test: |
  Scratch Django TestCase probe against a migrated test DB at HEAD exercising all four paths:
  (1) null/null unconfirmed + >1min correction, (2) null/null unconfirmed + sub-minute
  coordinate tweak, (3) fully-set sub-night + correction, (4) confirmed companion row + correction.
expecting: |
  (1) retired=1/created=1 + corrected boundaries; (2) unchanged=1, no churn, token refreshed;
  (3) no boundary re-mint (boundaries are timezone-derived only, lat/lon/alt cannot move them);
  (4) remint_declined=1 + warning, night stale but counted.
  If all four hold, the root cause is "gap closed; remaining delta is the warning text".
next_action: |
  DIAGNOSIS COMPLETE — hypothesis CONFIRMED on all four paths. No fix applied (diagnose-only mode).
  Handed back to the caller for /gsd-plan-phase --gaps. Nothing further to investigate.
bug_class: bohrbug  # fully deterministic; every probe scenario reproduced on demand
reasoning_checkpoint: null
tdd_checkpoint: null

## Symptoms
<!-- Written during gathering, then immutable -->

expected: |
  Correcting an Observatory row's lat/lon/altitude/timezone in place (without changing run.site) causes the next
  reconcile sweep to re-mint already-projected allocation nights at that site with boundaries derived from the
  corrected coordinates, counted and logged, rather than leaving them silently stale with unchanged=1.

  OWNER'S BINDING DESIGN CONSTRAINT (for the eventual fix): the re-mint must trigger ONLY when the recomputed
  night boundaries differ from the projected ones by MORE THAN 1 MINUTE. A sub-minute drift from a trivial
  coordinate tweak must NOT churn every night at the site. Owner verbatim: "someone might create an Observatory
  with a rough position in a hurry with runs associated with it and the Observatory position could be refined
  later which would be worth a re-sweep and reproject with new times if the positions changed enough to make
  >1 minute differences".

actual: |
  User reported: "fix in round 6"

  Round-5 verifier reproduction (at an earlier HEAD): editing an Observatory row's position/timezone in the
  Django admin WITHOUT changing run.site and re-running the sweep produced
  ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0). Night boundaries stayed at
  2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00 while the corrected site's true sunset/sunrise were
  2026-07-09 07:20:39 / 20:57:12 (~15 h stale, silent, uncounted). Provenance token was v2|1|none|none before
  and after because it carried site_id, not the site's coordinates.

errors: None reported (the defect is silent — that is the defect).
reproduction: Test 4 in .planning/phases/35-allocation-layer-classical-cutover/35-UAT.md
started: Discovered during round-5 verification (35-VERIFICATION.md), escalated to UAT, owner decided 2026-09-16 to fix in round 6.

## Eliminated
<!-- APPEND only - prevents re-investigating after /clear -->

(none yet)

## Evidence
<!-- APPEND only - facts discovered during investigation -->

- timestamp: 2026-09-16T00:10:00Z
  checked: .planning/debug/knowledge-base.md (Phase 0 semantic/keyword recall)
  found: |
    Entry `start-time-idempotency-key` — `telescope_runs.sun_event()` is deterministic within a
    process but its output is a direct function of astropy's IERS Earth-orientation data, which
    astropy refreshes over time; independent ingests of the identical (site, night) days apart
    produced start_time values ~2 s apart.
  implication: |
    A sub-minute tolerance is MANDATORY, not merely a nice-to-have anti-churn measure — without
    one, plain astropy drift alone would re-mint every night on every sweep. This is the
    documented rationale for the 1-minute constant and independently confirms the owner's
    >1-minute constraint is the right threshold class.

- timestamp: 2026-09-16T00:15:00Z
  checked: .planning/phases/35-allocation-layer-classical-cutover/35-VERIFICATION.md (round-6 re-verification, HEAD 0bc1ccd)
  found: |
    `re_verification.gaps_closed[0]` states the round-5 escalation is CLOSED and that the
    verifier re-ran the round-5 probe himself: same fixture (null/null run at La Silla, night
    2026-07-09, stored boundaries 2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00), same
    in-place correction, now produces `ReconcileResult(created=1, ..., retired=1)`, a NEW primary
    key, and boundaries 2026-07-09 07:20:39+00:00 -> 2026-07-09 20:57:12+00:00 — equal to the
    corrected position's live `sun_event()` values. Plan 35-24 truth 4 also verified: "A site
    correction whose sun events do not move does NOT destroy and re-create the night."
  implication: |
    The round-5 reproduction in the UAT gap entry is STALE. The code fix landed in round 6, which
    is already committed (HEAD 5b0f431 is two commits past 0bc1ccd, both `.planning/`-only).
    Must confirm independently rather than trust the report.

- timestamp: 2026-09-16T00:20:00Z
  checked: .planning/phases/35-allocation-layer-classical-cutover/35-UAT.md Tests 4/5 and the G-35-4 entry
  found: |
    UAT Test 4 is explicitly labelled "DECISION, not a manual test" with options "fix in round 6 /
    file as a follow-up / defer". `result: issue`, `reported: "fix in round 6"`. The UAT file's
    `updated:` is 2026-09-16T20:20:00Z — AFTER the round-6 verification at 19:14:26Z which already
    reported the gap closed. Test 5 (the round-6 seam, declined re-mint writing an unproven token)
    was SKIPPED by the owner as a "tip of the icecube" non-problem.
  implication: |
    G-35-4's `status: failed` is a bookkeeping artifact of the DECISION being recorded as an
    `issue` result. The owner's answer ("fix in round 6") was a scheduling instruction that round 6
    already carried out. The only genuinely new content in the gap entry is the >1-minute
    threshold constraint in `notes`, which must be checked against HEAD.

- timestamp: 2026-09-16T00:30:00Z
  checked: solsys_code/allocation_projector.py:64-147, :454-520 (fingerprint + token)
  found: |
    - `_UNRECORDED_PROVENANCE_TOLERANCE = timedelta(minutes=1)` at :77, with a comment stating
      exactly the owner's reasoning (astropy drift = seconds; a stale operator value = minutes to
      hours; "one minute separates the two classes cleanly").
    - `_PROVENANCE_TOKEN_VERSION = 'v3'` at :99.
    - `_site_position_fingerprint()` at :102-147 — SHA-256 over
      `f'{site.lat!r},{site.lon!r},{site.altitude!r},{site.timezone!r}'`, truncated to 16 hex
      chars; returns the literal `'none'` when `run.site_id is None`.
    - `_sub_night_provenance_token()` at :454-520 returns the five-part
      `v3|{site_id}|{fingerprint}|{start}|{end}`.
  implication: |
    Question 5 answered: the fingerprint covers ALL FOUR fields the truth names —
    lat, lon, altitude AND timezone. Nothing the truth names is omitted. The >1-minute threshold
    the owner asked for already exists as a named module constant.

- timestamp: 2026-09-16T00:40:00Z
  checked: solsys_code/allocation_projector.py:684-754 (`_span_needs_remint()` body)
  found: |
    Four steps, in order:
    1. :687-694 — per-field exact comparison for each SET sub-night field; a mismatch returns True.
    2. :695-696 — if BOTH sub-night fields are set, `return False` BEFORE the token is ever read.
    3. :697-725 — version-AND-part-count test (`len==5` and `parts[0]=='v3'`); when trusted, a
       COMPONENT-WISE comparison: differing start/end/site_id returns True immediately;
       fingerprint equal returns False; fingerprint-ONLY difference FALLS THROUGH to step 4
       (:723-725) rather than returning True.
    4. :730-754 — one `sun_event(run.site, night, kind='sun')` call; compares only the NULL
       side(s) at `> _UNRECORDED_PROVENANCE_TOLERANCE`; if beyond tolerance logs a warning and
       returns True (re-mint); if within tolerance records the fresh v3 token and returns False.
  implication: |
    Questions 1 and 4 answered together, and the answer to Q4's "fingerprint hash only?" is NO:
    the fingerprint difference is deliberately routed to an actual BOUNDARY comparison at the
    1-minute tolerance (the :723-725 fall-through and the comment at :609-612 state this
    explicitly: "an INPUT moving is not the same fact as the BOUNDARY moving, and a trivial
    one-metre altitude correction must not destroy and re-create every night"). So the
    >1-minute threshold is already in the exact place the eventual fix would have had to put it.

- timestamp: 2026-09-16T00:45:00Z
  checked: solsys_code/allocation_projector.py:252-322 (`_night_span_utc()` / `_time_of_day_to_datetime()`)
  found: |
    `_night_span_utc()` reads ONLY `run.site.timezone` (`ZoneInfo(run.site.timezone)`, nominal
    local 18:00 + 12 h). It never calls `sun_event()` and never reads lat/lon/altitude.
  implication: |
    Question 2 answered: for a FULLY-SET sub-night run the boundaries are a pure function of the
    sub-night TimeFields and the site's TIMEZONE — lat/lon/altitude cannot move them at all, so
    there is nothing to go stale. A TIMEZONE correction DOES move them and is caught by step 1's
    exact comparison (no tolerance needed: the computation is astropy-free and therefore
    drift-free). Step 2's early `return False` is correct, not a hole. The only site-derived
    field left on that path is the stored dark-window description line, handled separately by
    `_site_provenance_differs()` (:757-801) + the guarded refresh on the update path.

- timestamp: 2026-09-16T00:50:00Z
  checked: solsys_code/allocation_projector.py:1250-1318 (re-mint / decline branch), :804-872 (`_remint_decline_reason()`)
  found: |
    On a needed re-mint the sweep calls `_remint_decline_reason()`. When it returns 'confirmed' or
    'staff_state' the branch logs a named `logger.warning` ("Allocation re-mint declined: event
    pk=... is human-confirmed to run pk=... -- an automated re-mint never clears it"), increments
    `totals['remint_declined']`, and falls through (CR-04) to the label-only update path. Otherwise
    it increments `retired` + `created`, computes `_mint_fields()` BEFORE deleting, and does
    delete/create/link/record inside `transaction.atomic()`.
  implication: |
    Question 3 answered: on a confirmed companion row the re-mint IS declined — deliberately, per
    the UAT-2026-09-09 "human outranks machine" decision the owner re-affirmed in UAT Test 6 — and
    the staleness is NOT silent: it is counted under the dedicated `remint_declined` counter and
    logged with a named warning on EVERY sweep. That satisfies the truth's "counted and logged,
    rather than leaving them silently stale".

- timestamp: 2026-09-16T00:55:00Z
  checked: solsys_code/solsys_code_observatory/utils.py:107-173, admin.py, campaign_utils.py:266
  found: |
    `MPCObscodeFetcher.to_observatory()` does `obs = Observatory()` ... `obs.save()` — it always
    INSERTS a new row and never updates an existing one in place. The only other programmatic
    writer is `campaign_utils.py:266`, which CREATES a tier-3 placeholder (obscode/name/short_name
    only). Site resolution reassigns `run.site` to a different row, which changes `site_id` and is
    therefore caught by step 3's immediate `return True`. `ObservatoryAdmin`
    (solsys_code_observatory/admin.py:6-12) is a plain `ModelAdmin` declaring no `readonly_fields`.
  implication: |
    Question 6 answered: the Django admin is genuinely the only route to an in-place
    lat/lon/altitude/timezone correction. No programmatic path needs covering.

- timestamp: 2026-09-16T01:00:00Z
  checked: solsys_code/tests/test_allocation_projector.py (existing round-6 coverage)
  found: |
    `TestObservatoryCorrectionRemints::test_in_place_observatory_correction_remints_to_the_corrected_positions_sun_event`
    (:2387) reproduces the round-5 probe transcript verbatim in its docstring and asserts
    retired=1/created=1/unchanged=0 with a new pk and boundaries equal to the corrected site's
    live `sun_event()`. `TestMintInputInvariant::test_changing_the_sites_position_in_place_remints`
    (:3021), `TestSiteChangeRemints` (:2307), `TestSetWindowSiteCorrection` (:2438),
    `TestProvenanceTokenFormat::test_a_boundary_exactly_at_the_tolerance_resolves_as_correct`
    (:2888) and `::test_a_boundary_one_microsecond_beyond_the_tolerance_remints` (:2905) pin the
    threshold from both sides.
  implication: |
    A regression test for the truth AND for the 1-minute threshold already exists and is committed.
    The recurrence guard the eventual fix would have had to add is already in the tree.

- timestamp: 2026-09-16T01:30:00Z
  checked: |
    MY OWN executed probe (throwaway `solsys_code/tests/test_zz_g354_probe.py`, run against a
    migrated Django test database at HEAD 5b0f431, deleted afterwards; `git status --porcelain --
    solsys_code/ src/ docs/` empty after deletion). Not the verifier's report — my own run.
  found: |
    PATH 1 — null/null run, unconfirmed companion row, in-place correction (La Silla row's
    lat/lon/altitude/timezone edited to the Siding Spring values, `run.site` never reassigned).
    Reproduces the round-5 fixture byte-for-byte:
      token_before      = v3|1|5884a60fe2946a56|none|none
      boundaries_before = 2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00   (round-5's values)
      result            = ReconcileResult(created=1, updated=0, unchanged=0, ..., remint_declined=0, retired=1)
      boundaries_after  = 2026-07-09 07:20:39+00:00 -> 2026-07-09 20:57:12+00:00
      live sun_event    = 2026-07-09 07:20:39.082 / 2026-07-09 20:57:12.363   (equal)
      same_pk?          = False        token_after = v3|1|e108c70af44d9462|none|none
      LOG               = WARNING 'Allocation unrecorded-provenance night pk=1 run pk=1
                          night=2026-07-09: stored boundary start=... end=... disagrees beyond
                          tolerance with the resolved sun event sunset=... sunrise=...'
  implication: |
    The round-5 defect is GONE. `unchanged=1` has become `retired=1 + created=1`, with boundaries
    equal to the corrected coordinates' live sun events and a warning on the way. G-35-4's truth is
    satisfied on the path the gap was raised against.

- timestamp: 2026-09-16T01:35:00Z
  checked: MY OWN probe, PATH 1b — the owner's anti-churn constraint (sub-minute coordinate tweak)
  found: |
    Tweak: lat + 0.00001 deg (~1.1 m) and altitude + 1 m; same timezone.
      fingerprint_changed?  = True
      resulting boundary drift = 0:00:01.035156 (start) / 0:00:00.699219 (end)
      tolerance             = 0:01:00
      result  = ReconcileResult(created=0, updated=0, unchanged=1, ..., retired=0)
      same_pk? = True   boundaries unchanged? = True
      token_before = v3|1|5884a60fe2946a56|none|none  ->  token_after = v3|1|f85a965824247faa|none|none
      second sweep = unchanged=1
    Separate mocked-cost probe (P5): sweep-1 after the tweak = 1 `sun_event()` call; sweep-2 = 0 calls.
  implication: |
    THE OWNER'S >1-MINUTE CONSTRAINT IS ALREADY IMPLEMENTED AND ALREADY HOLDS. A few-metre tweak
    produces ~1 s of boundary movement, is recognised as within tolerance, does NOT churn the night
    (same pk, same boundaries, `unchanged`), and costs exactly ONE astropy call ONCE — the refreshed
    token makes every later sweep astropy-free. Note the drift magnitude (~1 s) is the same order as
    the astropy/IERS drift the knowledge-base entry documents, which independently confirms 1 minute
    is the correct threshold class.

- timestamp: 2026-09-16T01:40:00Z
  checked: MY OWN probe, PATH 2 — fully-set sub-night window under an in-place correction
  found: |
    (a) lat/lon/altitude corrected (-30.5 / -71.5 / 1500), timezone unchanged:
      result = updated=1, same_pk = True, boundaries 2026-07-09 23:00:00+00:00 -> 2026-07-10
      05:00:00+00:00 UNCHANGED (correct — they are pinned by the sub-night TimeFields and the
      site's TIMEZONE only), dark-window description line REFRESHED
      23:09:10 -> 23:10:27 / 10:27:15 -> 10:32:07, token refreshed to the new fingerprint.
    (b) timezone then corrected to Australia/Sydney on the same fully-set run:
      `reconcile_run()` RAISES ValueError('Computed an inverted allocation-night span for run pk=1
      night=2026-07-09: start=2026-07-09T23:00:00+00:00 >= end=2026-07-09T05:00:00+00:00. Check
      night_start_utc/night_end_utc against the site timezone.')
      Command-level containment probe (P7): `call_command('reconcile_campaign_runs')` catches it
      per-run — stderr 'Run pk=1: reconcile failed (...) -- skipping', summary 'failed: 1', the
      other (null/null) run in the same sweep still re-minted normally (created: 1, retired: 1),
      and the fully-set run's stored boundaries survived untouched.
  implication: |
    No staleness exists on this path to fix. lat/lon/altitude CANNOT move a fully-set run's
    boundaries, so there is nothing to re-mint; the one site-derived field that can go stale (the
    dark-window line) IS refreshed. The cross-timezone sub-case fails LOUDLY and CONTAINED with the
    remedy in the message itself — the opposite of "silently stale". It is a decided, tested
    (`test_a_set_window_moved_across_timezones_has_a_pinned_outcome`) and runbook-documented outcome.

- timestamp: 2026-09-16T01:45:00Z
  checked: MY OWN probe, PATHS 3 and 3b — confirmed / staff-state companion row under a big correction
  found: |
    PATH 3 (`confirmed_by` + `confirmed_at` set):
      result = ReconcileResult(created=0, updated=0, unchanged=1, ..., remint_declined=1, retired=0)
      same_pk? = True; boundaries STAY 2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00
      while the TRUE corrected sun events are 07:20:39 / 20:57:12 (the ~15 h staleness)
      token NOT refreshed (still v3|1|5884a60fe2946a56|none|none)
      TWO warnings logged: the staleness warning (naming the stored boundary AND the resolved sun
      event) plus 'Allocation re-mint declined: event pk=1 night=2026-07-09 is human-confirmed to
      run pk=1 -- an automated re-mint never clears it.'
      second sweep = remint_declined=1 again, 2 warnings again (cost probe P6: 3 consecutive sweeps,
      each remint_declined=1 / unchanged=1 / exactly 1 `sun_event()` call)
    PATH 3b (`is_verified=False`): identical counters, with the 'carries staff-set state' variant of
      the decline warning.
  implication: |
    The night DOES stay stale here — but deliberately, and NOT silently: it is counted under the
    dedicated `remint_declined` counter and re-reported with two warnings on EVERY sweep, forever.
    This is the owner-accepted UAT-2026-09-09 "human outranks machine" rule, which the owner
    re-affirmed by PASSING UAT Test 6 in the same session that raised this gap. G-35-4's truth
    ("counted and logged, rather than leaving them silently stale") is therefore SATISFIED here too.
    The per-sweep cost is the documented WR-07 bound, not a regression.

- timestamp: 2026-09-16T01:50:00Z
  checked: MY OWN probe, PATH 4 — exact field coverage of `_site_position_fingerprint()`
  found: |
    Per-field, each edit applied and then reverted against a freshly DB-loaded run:
      lat        -29.2567 -> -29.0             fingerprint changed? True
      lon        -70.73 -> -70.0               fingerprint changed? True
      altitude   2347.0 -> 2000.0              fingerprint changed? True
      timezone   America/Santiago -> Sydney    fingerprint changed? True
      name       'ESO, La Silla' -> 'Renamed'  fingerprint changed? False
      short_name 'NTT' -> 'XXX'                fingerprint changed? False
      obscode    '809' -> 'Z99'                fingerprint changed? False
      observations_type 0 -> 4                 fingerprint changed? False
      uses_two_line_obs False -> True          fingerprint changed? False
    (A first probe run reported True for every field; that was MY probe's own artifact — it compared
    against a baseline computed from the cached in-memory `Observatory` created with `altitude=2347`
    as an int, whose `repr()` is '2347' rather than the DB FloatField's '2347.0'. Corrected by
    computing each baseline from a freshly DB-loaded run. No production path reaches the projector
    with a non-DB-loaded site — `project_allocation()` and `reproject_allocation_if_dispatched()`
    both receive a DB-loaded run — and even if one did, the consequence is bounded to one
    within-tolerance resolution, never a spurious re-mint.)
  implication: |
    Coverage is EXACT: all four fields the truth names are covered, and nothing else is, so an
    unrelated admin edit (fixing a site's name or obscode) does not trigger a single astropy call.
    Question 5's "any field the truth names but the fingerprint omits" answer is NONE.

- timestamp: 2026-09-16T01:55:00Z
  checked: docs/runbooks/telescope_runs_calendar.rst:1123-1138
  found: |
    A dedicated paragraph already exists: "**Correcting a site's own definition also re-mints,
    separately from correcting a run's** ``site``. Editing an ``Observatory`` row's latitude,
    longitude, altitude or timezone in the Django admin -- without touching any run's ``site``
    field at all -- now re-mints every allocation night already projected at that site, on the next
    sweep, counted under ``retired`` and ``created`` ..." and, crucially, the owner's own
    constraint: "One case deliberately does not re-mint: a position correction too small to move
    the computed sunset or sunrise by more than a minute is recognised as within tolerance and
    reported ``unchanged`` ... so a small position fix (rounding a coordinate, for instance) does
    not churn the calendar."
  implication: |
    The paired-docs obligation for this gap is ALREADY discharged, including the >1-minute
    constraint in operator-facing prose. No runbook update is owed for the truth itself.

- timestamp: 2026-09-16T02:00:00Z
  checked: |
    The ONE thing that is still wrong — the operator-facing warning text, observed verbatim in my
    own PATH 1 and PATH 3 runs (allocation_projector.py:739-750).
  found: |
    Emitted text: 'Allocation unrecorded-provenance night pk=1 run pk=1 night=2026-07-09: stored
    boundary ... disagrees beyond tolerance with the resolved sun event ...'
    But on this entry path the night's provenance WAS recorded — a well-formed current-version
    token, `v3|1|5884a60fe2946a56|none|none`. Only the POSITION FINGERPRINT component differed
    (the :723-725 fall-through). The step-4 branch is shared by two entry paths (genuinely
    unrecorded/legacy provenance, and trusted-but-repositioned) and its message only describes the
    first. The constant `_UNRECORDED_PROVENANCE_TOLERANCE` (:77) and the test class
    `TestUnrecordedProvenanceNight` carry the same now-too-narrow name.
  implication: |
    An operator who has just corrected a site position is told the night had NO recorded provenance,
    which is false, and is thereby pointed at the runbook's one-time legacy-audit reason (5) instead
    of at the site-definition-correction paragraph that actually describes what they just did. This
    degrades the "and logged" half of the truth (the log fires, but names the wrong cause and the
    wrong remedy). Already recorded independently as `advisory` #1 in 35-VERIFICATION.md and as an
    Anti-Pattern row for :739-750. This is the entire remaining delta.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: |
  G-35-4 WAS ALREADY CLOSED IN CODE BEFORE THIS INVESTIGATION STARTED. The gap entry's
  `status: failed` is a bookkeeping artifact, not a live defect: UAT Test 4 was explicitly a
  DECISION ("fix in round 6 / follow-up / defer"), the owner answered "fix in round 6", and round 6
  (plans 35-23/35-24/35-25, commits up to 0bc1ccd, two `.planning/`-only commits before HEAD
  5b0f431) already carried that decision out. The round-5 reproduction quoted in the gap entry is
  stale and no longer reproduces.

  Confirmed by MY OWN executed probe at HEAD, not by reading the round-6 verification report. With
  the round-5 fixture reproduced byte-for-byte (null/null run at La Silla, night 2026-07-09, stored
  boundaries 2026-07-09 22:06:35+00:00 -> 2026-07-10 11:29:46+00:00), an in-place correction of the
  same `Observatory` row's lat/lon/altitude/timezone with `run.site` never reassigned now yields
  `ReconcileResult(created=1, ..., retired=1)`, a new primary key, boundaries
  2026-07-09 07:20:39+00:00 -> 2026-07-09 20:57:12+00:00 equal to the corrected coordinates' live
  `sun_event()` values, and a `logger.warning`. The mechanism: `_site_position_fingerprint()`
  (:102-147, SHA-256 over `repr(lat), repr(lon), repr(altitude), repr(timezone)`, 16 hex chars)
  feeds a five-part `v3|{site_id}|{fingerprint}|{start}|{end}` token (:520), and
  `_span_needs_remint()` step 3 (:710-725) compares it COMPONENT-WISE, routing a
  fingerprint-only difference to step 4's single `sun_event()` boundary comparison.

  THE OWNER'S >1-MINUTE CONSTRAINT IS ALSO ALREADY SATISFIED, in exactly the place the fix would
  have had to put it. `_UNRECORDED_PROVENANCE_TOLERANCE = timedelta(minutes=1)`
  (allocation_projector.py:77) pre-dates this round (CR-01/plan 35-19) and is REUSED by the new
  fingerprint fall-through: step 3 deliberately does NOT re-mint on a fingerprint difference alone
  but falls through to a real BOUNDARY comparison at that tolerance (:723-725, rationale stated at
  :609-612 — "an INPUT moving is not the same fact as the BOUNDARY moving, and a trivial one-metre
  altitude correction must not destroy and re-create every night"). My probe: a ~1.1 m latitude +
  1 m altitude tweak moves the boundaries ~1.0 s / ~0.7 s, is recognised as within tolerance, and
  reports `unchanged=1` with the same primary key and identical boundaries — no churn — at a cost
  of exactly one `sun_event()` call once (sweep-2 measured 0 calls, because the resolution records
  the refreshed token). So the answer to the brief's question 4 is: `_span_needs_remint()` already
  compares recomputed boundaries against the stored ones, NOT the fingerprint hash alone.

  THE SINGLE REMAINING DELTA is operator-facing message accuracy, not staleness. On the
  site-correction entry path the shared step-4 branch emits (observed verbatim in my own run):
  'Allocation unrecorded-provenance night pk=1 run pk=1 night=2026-07-09: stored boundary ...
  disagrees beyond tolerance with the resolved sun event ...' (allocation_projector.py:739-750).
  The night's provenance WAS recorded — a well-formed `v3|1|5884a60fe2946a56|none|none` token; only
  the position-fingerprint component differed. The branch is reached by two entry paths (genuinely
  unrecorded/legacy provenance, and trusted-but-repositioned) and the message describes only the
  first, so an operator who has just corrected a site position is told the night had no recorded
  provenance and is pointed at the runbook's one-time legacy-audit reason (5) instead of at the
  site-definition-correction paragraph describing what they actually did. The constant
  `_UNRECORDED_PROVENANCE_TOLERANCE` and the test class `TestUnrecordedProvenanceNight` carry the
  same now-too-narrow name. This degrades the "and logged" half of the truth without falsifying it.

  Per-path answers to the brief's questions 1-3 (all with executed evidence):
  1. null/null, unconfirmed -> RE-MINTS (`retired=1 + created=1`), corrected boundaries, warning
     logged. Truth met.
  2. fully-set sub-night -> lat/lon/altitude CANNOT move the boundaries (they are pinned by the
     sub-night TimeFields and `_night_span_utc()`, which reads ONLY `site.timezone`, :282), so there
     is no staleness to fix; the one site-derived field that can go stale (the dark-window
     description line) IS refreshed via `_site_provenance_differs()` (:757-801). A TIMEZONE
     correction on such a run raises `ValueError` ('Computed an inverted allocation-night span ...
     Check night_start_utc/night_end_utc against the site timezone'), which the sweep command
     catches per-run ('Run pk=1: reconcile failed (...) -- skipping', `failed: 1`) while continuing
     to sweep every other run — loud and contained, the opposite of silently stale. Decided, tested
     and runbook-documented.
  3. confirmed companion row (and `is_verified=False`) -> re-mint DECLINED, night stays stale, but
     counted under the dedicated `remint_declined` counter and re-reported with TWO warnings on
     every sweep (measured over 3 consecutive sweeps). This is the owner-accepted "human outranks
     machine" rule, which the owner re-affirmed by PASSING UAT Test 6 in the same session. The
     truth's "counted and logged, rather than silently stale" is satisfied.
  Question 5: the fingerprint covers EXACTLY lat/lon/altitude/timezone — nothing the truth names is
  omitted, and nothing else triggers it (name/short_name/obscode/observations_type/uses_two_line_obs
  all measured as not changing it).
  Question 6: `MPCObscodeFetcher.to_observatory()` always does `obs = Observatory()` ... `obs.save()`
  — it INSERTS, never updates in place (utils.py:107-173). The only other writer creates a tier-3
  placeholder (campaign_utils.py:266), and site resolution reassigns `run.site`, changing `site_id`,
  which step 3 catches with an immediate `return True`. `ObservatoryAdmin` is a plain `ModelAdmin`
  with no `readonly_fields`, so the Django admin is genuinely the only in-place route.

fix: |
  (diagnose-only mode — no fix applied, nothing under solsys_code/, src/ or docs/ was modified;
  `git status --porcelain -- solsys_code/ src/ docs/` is empty). The throwaway probe module was
  deleted after the run.

  Recommendation for /gsd-plan-phase --gaps: there is NO staleness defect left to fix and NO
  threshold to add. Either close G-35-4 as already-satisfied, or scope a small message-accuracy
  task for the single remaining delta (see Suggested Fix Direction in the handback).

verification: |
  Not applicable (diagnose-only). Diagnosis evidence: my own throwaway Django `TestCase` probe at
  HEAD 5b0f431 covering seven scenarios (null/null big correction, null/null sub-minute correction,
  fully-set lat/lon/alt correction, fully-set timezone correction, confirmed decline, staff-state
  decline, per-field fingerprint coverage) plus two cost probes (mocked `sun_event()` call counts)
  and one command-level containment probe (`call_command('reconcile_campaign_runs')`).
oracle_type: derived
files_changed: []
