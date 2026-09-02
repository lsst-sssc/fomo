---
status: diagnosed
trigger: "This phase should not target 'LCO and Gemini sync commands' but rather 'LCO and SOAR' - we have no visibility into any of the Gemini queues through the existing GEMFacility class"
created: 2026-09-02T20:00:00Z
updated: 2026-09-02T20:42:00Z
---

## Current Focus

hypothesis: CONFIRMED (multi-cause). The user's report is factually correct.
  GEMFacility has no read path at all (get_observation_status is a hardcoded stub;
  submit_observation is its only network call), so `sync_gemini_observation_calendar` can only
  replay FOMO's own outbound ToO submissions — it has zero Gemini queue visibility. SOAR, by
  contrast, is a real live-API facility (SOARFacility subclasses LCOFacility, uses the LCO
  portal + LCO API key) and is ALREADY handled inside `sync_lco_observation_calendar`.
test: complete — source read of GEMFacility/SOARFacility/both sync commands, plus dev-DB counts
expecting: n/a (root cause confirmed)
next_action: return diagnosis to caller; plan-phase --gaps handles the fix

bug_class: Bohrbug (deterministic, fully reproducible by reading source — a scope/domain-accuracy
  defect in planning artifacts, not a runtime fault)

rca_branching:
  candidate_causes:
    - "code (third-party library): GEMFacility.get_observation_status() / get_observation_url() /
       data_products() / _archive_frames() are all hardcoded stubs in tomtoolkit's gemini.py —
       the class is submit-only"
    - "data: the dev DB holds 13 LCO ObservationRecords and ZERO GEM and ZERO SOAR records; zero
       CampaignRuns carry source='gemini_queue'. The Gemini adapter has never had real input."
    - "config/vocabulary: CampaignRun.Source declares GEMINI_QUEUE but has NO SOAR_QUEUE value,
       even though sync_lco_observation_calendar already treats SOAR as a distinct facility"
    - "documentation/process: v1.5-REQUIREMENTS.md's Out of Scope table recorded 'Live Gemini ODB
       status polling — GEMFacility.get_observation_status() is a stub returning empty state',
       but that limitation was never carried forward into PROJECT.md's active limitations, into
       v2.2's Source vocabulary, or into v2.3's REQUIREMENTS/ROADMAP"
  and_gate: "YES — this required at least three conditions simultaneously. (1) The library stub is
    real but its record lives only in an ARCHIVED v1.5 requirements doc. (2) The Source vocabulary
    minted GEMINI_QUEUE and omitted SOAR entirely, so the three-adapter framing was already baked
    into the model layer before v2.3 planning began. (3) Phase 31's own probe measured
    SOURCE_gemini_queue=0 and tagged the Gemini row 'Constructed-input code-path check', but that
    probe was designed to test constraint idempotency, not facility reachability — it could not
    distinguish 'adapter works, DB is empty' from 'adapter can never be fed'. Remove any one and
    the wrong scope would likely have been caught."

## Symptoms

expected: Every ingest path the SCHEMA-02 per-adapter table in 31-DECISION.md names
  (load_telescope_runs, sync_lco_observation_calendar, sync_gemini_observation_calendar) has
  real, current functional visibility into the facility it claims to sync. 31-DECISION.md and
  docs/design/run_identity_and_unattended_invocation_spike.rst treat
  sync_gemini_observation_calendar.py as a real, functioning third ingest adapter, citing exact
  source lines (e.g. sync_gemini_observation_calendar.py:150) and listing
  CampaignRun.Source.GEMINI_QUEUE as one of three live per-adapter identity keys.
actual: User reports no visibility into any Gemini queue through the existing GEMFacility class;
  the correct second/third facility pair is "LCO and SOAR", not "LCO and Gemini".
errors: none (scope/domain-accuracy report, not a runtime error)
reproduction: Test 3 in .planning/phases/31-foundation-spikes-run-identity-unattended-invocation/31-UAT.md
started: Discovered during UAT (phase 31 verification), 2026-09-02

## Eliminated

## Evidence

- timestamp: 2026-09-02T20:05:00Z
  checked: .planning/debug/knowledge-base.md for Gemini/SOAR entries
  found: No prior entry on Gemini queue visibility. Two incidental Gemini mentions only
    (calendar-event start_time tolerance; range-window GS-2026A-FT-115 projection).
  implication: No known-pattern shortcut; investigate from source.

- timestamp: 2026-09-02T20:08:00Z
  checked: tom_observations/facilities/gemini.py GEMFacility (site-packages, tomtoolkit)
  found: GEMFacility.get_observation_status() (gemini.py:506-507) is a hardcoded stub returning
    {'state': '', 'scheduled_start': None, 'scheduled_end': None} — no HTTP call, no argument use.
    get_observation_url() (gemini.py:490-492) returns '' with the real URL commented out.
    data_products() returns []. _archive_frames() returns []. The ONLY outbound network call in
    the entire class is submit_observation() (gemini.py:453-465), a POST to PORTAL_URL[site]+'/too'.
  implication: GEMFacility is WRITE-ONLY. It can push a ToO trigger to Gemini but has no read path
    of any kind — no queue listing, no status, no scheduled_start/scheduled_end, no archive.
    User's report is factually correct at the library level.

- timestamp: 2026-09-02T20:12:00Z
  checked: solsys_code/management/commands/sync_gemini_observation_calendar.py (full read, 193 lines)
  found: handle() queries ObservationRecord.objects.filter(facility='GEM') — the LOCAL FOMO
    database only. Zero outbound HTTP calls; GEMFacility is never even imported. add_arguments()
    is `pass`. Every field is derived from record.parameters (the payload FOMO itself submitted)
    plus settings.FACILITIES['GEM']['programs'] (a static local dict). Line 150's
    url = f'GEM:{prog}/{record.observation_id}' is a locally synthesized string, not a facility ID
    fetched from Gemini.
  implication: The command is live and functional as written, but its input is FOMO's own
    outbound ToO submissions replayed onto the calendar — NOT observations read back from a
    Gemini queue. It cannot learn about scheduling, execution, or any run FOMO did not submit.

- timestamp: 2026-09-02T20:15:00Z
  checked: tom_observations/facilities/soar.py
  found: `class SOARFacility(LCOFacility)` (soar.py:240) — inherits the entire LCO read path,
    including get_observation_status(). Docstring (soar.py:245): "SOAR is only available in
    AEON-mode. It also uses the LCO API key" with portal_url https://observe.lco.global.
  implication: SOAR has REAL queue visibility, via the LCO Observation Portal API. It is the
    genuine third live facility.

- timestamp: 2026-09-02T20:18:00Z
  checked: solsys_code/management/commands/sync_lco_observation_calendar.py
  found: Already handles BOTH facilities: `records = ObservationRecord.objects.filter(
    facility__in=['LCO', 'SOAR'])` (line 298), `facilities = {'LCO': LCOFacility(),
    'SOAR': SOARFacility()}` (line 289, separate instances per SELECT-05), per-facility counters
    (line 296), --proposal help text says "LCO/SOAR proposal code(s)", and it makes live API
    calls via resolve_placement_block / facility.get_observation_url.
  implication: SOAR is not a missing adapter needing to be built — it is already a live,
    API-backed second facility inside the existing LCO sync command.

- timestamp: 2026-09-02T20:20:00Z
  checked: src/fomo/settings.py TOM_FACILITY_CLASSES (lines 273-280)
  found: LCOFacility, LCORedirectFacility, GEMFacility, SOARFacility, ESOFacility all registered.
  implication: SOARFacility is already installed and available; nothing blocks its use.

- timestamp: 2026-09-02T20:24:00Z
  checked: CampaignRun.Source (solsys_code/models.py:108-138) + all usages
  found: Values are WEB, CLASSICAL_FILE, LCO_QUEUE, GEMINI_QUEUE, ESO_QUEUE, CSV_IMPORT, LEGACY.
    There is NO SOAR_QUEUE. The docstring (models.py:111-114) says CLASSICAL_FILE/LCO_QUEUE/
    GEMINI_QUEUE "are declared now but not produced by any code path until v2.3's ADAPT-01..03".
    GEMINI_QUEUE appears in exactly one place outside models.py: a single test fixture
    (test_campaign_reconciler.py:137). Also notable: models.py:126-128 explicitly reasons that
    mapping ESO rows onto LCO_QUEUE "would have been semantically wrong (they are not LCO-network
    runs)" — the identical argument was never applied to SOAR, which IS an LCO-network partner.
  implication: The three-adapter framing was already baked into the model vocabulary in v2.2,
    before v2.3 planning began. SOAR had no slot to be named in.

- timestamp: 2026-09-02T20:27:00Z
  checked: dev DB src/fomo_db.sqlite3 (read-only sqlite3 queries)
  found: ObservationRecord by facility: LCO=13, GEM=0, SOAR=0.
    CampaignRun by source: legacy=24, csv_import=11, eso_queue=7, lco_queue=5, web=1,
    classical_file=1, gemini_queue=0 (49 total, matching 31-DECISION.md's own count).
    CalendarEvent urls: 10 https://observe.lco.global/request..., 9 empty, rest RUN:* keys —
    zero GEM: keys.
  implication: No Gemini data has ever flowed through this system. 31-DECISION.md line 246/710
    noticed the absence ("no real GEM:-namespaced row exists in this dev DB to copy from") but
    read it as missing test data rather than as an unreachable facility.

- timestamp: 2026-09-02T20:30:00Z
  checked: tom_observations/facilities/ocs.py get_observation_status (inherited by LCOFacility,
    therefore by SOARFacility)
  found: ocs.py:1548-1568 makes live GET calls to {portal_url}/api/requests/{id} and
    /api/requests/{id}/observations/, returning state plus the scheduled COMPLETED/PENDING block.
  implication: SOAR has exactly the real queue read path Gemini lacks, over the same LCO portal.

- timestamp: 2026-09-02T20:33:00Z
  checked: prior planning record — .planning/milestones/v1.5-REQUIREMENTS.md and Phase 10 artifacts
  found: This is a PREVIOUSLY-RECORDED limitation, not a new discovery.
    v1.5-REQUIREMENTS.md:43 Out of Scope table: "Live Gemini ODB status polling |
    GEMFacility.get_observation_status() is a stub returning empty state".
    v1.5-REQUIREMENTS.md:36 GEM-GPP-02 (future): "Replace constructed GEM:{prog}/{obs_id} key with
    real portal URL from GEMFacility.get_observation_url() once un-stubbed".
    10-CONTEXT.md:188: "GEMFacility.get_observation_status() is a stub — this command does NOT
    call it."
    23-DISCUSSION-LOG.md:55: "this DB currently has 0 GEM ObservationRecords."
    PROJECT.md:410 carries only the softer framing ("stub get_observation_url()"), not the
    read-path-absent consequence.
  implication: The knowledge existed but lived in ARCHIVED v1.5/v2.1 milestone docs. It never
    propagated into v2.2's Source vocabulary or v2.3's REQUIREMENTS/ROADMAP/31-DECISION.md.

- timestamp: 2026-09-02T20:36:00Z
  checked: v2.3 scope docs — REQUIREMENTS.md ADAPT-01..05, ROADMAP.md Phase 32, STATE.md
  found: ADAPT-03 (REQUIREMENTS.md:20) names sync_gemini_observation_calendar. ROADMAP.md:180
    Phase 32 goal: "classical schedule file, LCO queue, Gemini queue". ROADMAP Phase 32 Success
    Criterion 3: sync_gemini "proving the pattern generalises to a second facility". The word
    SOAR appears NOWHERE in ROADMAP.md or REQUIREMENTS.md.
    BUT STATE.md:27 states the milestone core value as "Robotically scheduled LCO/SOAR
    observations and their outcomes appear and update on the calendar" — LCO/SOAR, no Gemini.
  implication: The milestone's own stated core value already says LCO/SOAR. The requirements and
    roadmap contradict it, inheriting the model-layer's three-value Source framing instead.

- timestamp: 2026-09-02T20:39:00Z
  checked: downstream impact on Phase 33 (OUTCOME-01..04)
  found: Outcome propagation requires reading a terminal observing state back from the facility.
    For LCO/SOAR that is ocs.py:1548's live API. For Gemini, GEMFacility.get_observation_status()
    returns {'state': ''} unconditionally — a value that is not in TERMINAL_OBSERVING_STATES and
    never changes.
  implication: Phase 33's outcome propagation is structurally impossible for Gemini and fully
    possible for SOAR. The mis-scoping is not confined to Phase 31/32.

## Resolution

root_cause: >
  Three simultaneous conditions (AND-gate), not one:

  (1) LIBRARY REALITY — tomtoolkit's `GEMFacility`
  (`tom_observations/facilities/gemini.py:434-523`) is submit-only. Its single outbound call is
  `submit_observation()` (gemini.py:453-465, POST to `PORTAL_URL[site] + '/too'`). Every read
  method is a hardcoded stub: `get_observation_status()` returns
  `{'state': '', 'scheduled_start': None, 'scheduled_end': None}` (gemini.py:506-507),
  `get_observation_url()` returns `''` (gemini.py:490-492), `data_products()` and
  `_archive_frames()` return `[]`. There is therefore no Gemini queue visibility of any kind.
  By contrast `SOARFacility` (`facilities/soar.py:240`) subclasses `LCOFacility`, uses the LCO
  API key and `observe.lco.global` portal, and inherits the real live `get_observation_status()`
  at `ocs.py:1548-1568`.

  (2) VOCABULARY — `CampaignRun.Source` (`solsys_code/models.py:132-138`) declares
  `GEMINI_QUEUE` but has no `SOAR_QUEUE`, so the "three ingest adapters" framing was fixed in
  the model layer during v2.2, before v2.3 planning started. This is inconsistent with
  `sync_lco_observation_calendar.py`, which has treated LCO and SOAR as two distinct facilities
  since v1.3 (`facility__in=['LCO', 'SOAR']` at line 298; `{'LCO': LCOFacility(),
  'SOAR': SOARFacility()}` at line 289; per-facility counters at line 296) — and with
  models.py:126-128's own reasoning that ESO deserved a distinct value rather than being folded
  into LCO_QUEUE.

  (3) PROVENANCE LOSS — the stub was a KNOWN, RECORDED limitation
  (`v1.5-REQUIREMENTS.md:43` Out of Scope: "Live Gemini ODB status polling |
  GEMFacility.get_observation_status() is a stub returning empty state"; GEM-GPP-02;
  `10-CONTEXT.md:188`), but it lived only in archived v1.5 milestone docs. It was never carried
  into PROJECT.md's active limitations, the v2.2 Source vocabulary, or v2.3's
  REQUIREMENTS/ROADMAP — so Phase 31 planned against a facility inventory that no longer
  reflected what was known.

  NET FACTUAL FINDING: `sync_gemini_observation_calendar.py` IS live, non-stub, working code —
  but it is not a queue sync. It reads `ObservationRecord.objects.filter(facility='GEM')` from
  the LOCAL FOMO database only (line 40), makes zero outbound calls, and never imports
  GEMFacility. Its input is FOMO's own previously-submitted ToO payloads replayed onto the
  calendar. Line 150's `url = f'GEM:{prog}/{record.observation_id}'` is a locally synthesized
  string, not an identifier obtained from Gemini. Its real-world corpus is currently empty
  (0 GEM ObservationRecords, 0 gemini_queue CampaignRuns). So the citation in 31-DECISION.md is
  accurate about the code line, but the premise the table rests on — that Gemini is one of three
  facilities FOMO can see into — is wrong. SOAR is the real third facility, and it is already
  half-built inside the LCO adapter.

fix: (not applied — diagnose-only mode; plan-phase --gaps will handle)
verification: (n/a)
files_changed: []
