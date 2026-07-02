---
title: Target-linked run submission form with approval queue — expanded to full campaign coordination (3I-spreadsheet replacement)
trigger_condition: Calendar feature is mature (v1.x complete — SATISFIED as of v1.7) and there is a coordination need for ad-hoc multi-PI follow-up campaigns (e.g. 4I, the next interstellar object; IAWN campaign)
planted_date: 2026-06-16
enriched_date: 2026-07-02
context: Explored during Phase 03 UAT session; enriched post-v1.7 against the real 3I/ATLAS campaign spreadsheet
---

# Seed: Target-linked run submission form with approval queue

## Idea

A web form that lets PIs and community members submit telescope runs directly into FOMO,
tied to a specific Target, with an admin approval queue before runs appear as CalendarEvents.

## Motivation

The `load_telescope_runs` file-based command works well for pre-planned, PI-coordinated programs
(e.g. a Didymos observing season loaded from a schedule file). But when a rare, urgent object
appears (e.g. 4I/Borisov-class interstellar object, IAWN campaign), coordination happens
fast and involves people outside the core team. A web form lowers the barrier to entry
without bypassing oversight.

## Shape of the feature

- **Mandatory:** Target (must exist in FOMO) — anchors the run to a specific object
- **Optional:** Proposal code — required only if the submitter wants to trigger observations through FOMO; otherwise the run is calendar-only
- **Submitter types:**
  - Approved-program PI (committed time) — may bypass approval
  - DDT PI (Director's Discretionary Time, urgent/new program) — requires admin approval
  - Community member / FOMO admin — requires admin approval
- **Approval flow:** Submission creates a pending CalendarEvent (or a pre-event record); admin reviews and approves before it becomes visible on the shared calendar
- **Spam/error control:** Approval step catches accidental duplicates, bad telescope/date combinations, and uncoordinated community submissions

## Enrichment (2026-07-02): the 3I/ATLAS campaign spreadsheet as the reference model

Operator (Tim) goal: for the next 4I-class object, FOMO should **replicate and improve on**
the community Google Sheet used to coordinate 3I/ATLAS observing runs
(https://docs.google.com/spreadsheets/d/1INhxLWlHoa-JkW-uKRzmSyms06wI80wEXTqBJSR3YAI/).
That sheet is, in effect, a field-requirements document written by real campaign practice.
The original seed below covers the **intake** side (form + approval queue); this enrichment
adds the **tracking schema** and the **display/coordination** side.

### Field inventory from the 3I sheet (per observing run)

- Contact person + email — **PII**: the sheet is link-shared; FOMO has `OPEN` targets and
  `AUTH_STRATEGY='READ_ONLY'`, so who can see submitter emails needs an explicit decision
  (auth-gated column, opt-in display, or store-but-never-render)
- Telescope/instrument (CalendarEvent already has both fields)
- **Site code** — joins directly onto FOMO's `Observatory` model (MPC obscode)
- Obs. date + UT time range (CalendarEvent start/end)
- Filter(s)/bandpass — new field
- Observation details (imaging/spectroscopy/IFU description) — new field
- Weather conditions — new field, low priority
- Observation status — a real lifecycle, richer than today's title prefixes:
  planned → observed → data reduced → published
- Observation outcome — new field
- Publication plans — new field
- Open to collaboration? flag — new field
- Other comments — new field

Example 3I rows spanned FTN/MuSCAT3 multi-band imaging, Palomar P200/NGPS
imaging+spectroscopy, and VLT/MUSE IFU monitoring — i.e. mostly facilities **outside**
FOMO's sync commands, reinforcing that community submission (not API sync) is the
intake path for campaign runs.

### Expanded scope beyond the original seed

1. **Target-anchored campaign grouping/display** — a per-object, spreadsheet-like table
   view of all campaign runs ("all runs for object X"), alongside the existing calendar.
   `CalendarEvent` has no Target link today; the campaign entity provides it.
2. **Storage shape** — the lifecycle/outcome/PII fields likely want a `CampaignRun`
   model (or a sidecar on `CalendarEvent`, following the Phase 8
   `CalendarEventTelescopeLabel` sidecar precedent, which post-dates the original seed)
   rather than widening `CalendarEvent` itself.
3. **Bootstrap import** — a one-off CSV ingest of the actual 3I sheet to validate the
   model against real campaign data before a 4I ever appears.
4. **Coverage-gap analysis (FOMO's differentiator over any spreadsheet)** — FOMO computes
   ephemerides and knows site geometry + dark windows (`solsys_code/telescope_runs.py`).
   A campaign view can show dates/wavelength ranges where the object is observable but
   no run is planned — active coordination, not just record-keeping. This is the reason
   to build it in FOMO rather than make another sheet.

## Open questions when this seed is revisited

- What model stores the pending submission (a separate `RunSubmission` model, or a `CalendarEvent` with a `status=pending` field)?
- How does the admin get notified (email, in-app notification, both)?
- Should approved-program PIs have a self-service approval path (no admin needed) or always go through admin?
- What proposal-code validation is needed — free text, or cross-checked against an LCO/ESO proposal database?
- (2026-07-02) How are submitter emails protected given FOMO's open/read-only permission model?
- (2026-07-02) Does the status lifecycle live on the campaign run record, and does any of it flow back into the calendar display (e.g. title prefixes like the existing [QUEUED]/[UNVERIFIED] vocabulary)?
- (2026-07-02) Scope estimate: a full milestone, likely multi-phase — data model + bootstrap import, submission form + approval queue, campaign table view, coverage-gap analysis.
