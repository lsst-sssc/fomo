# Feature Research

**Domain:** Astronomical follow-up scheduling — queue-network (OCS/LCO-family) telescope-time sync into a shared calendar
**Researched:** 2026-06-19
**Confidence:** MEDIUM (codebase/API facts HIGH from direct inspection of v1.2 code + PROJECT.md; general batch/UX best-practice claims LOW — generic websearch, no domain-specific authoritative source found for this niche)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist once a sync tool claims to handle "all LCO-family proposals/facilities." Missing these makes the v1.3 generalization feel half-done relative to what v1.2 already promised.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Multi-proposal selection (`--proposal a,b,c` and `--proposal ALL`) | A single hard-coded proposal code (v1.2) doesn't scale once more than one science program uses the calendar; "ALL" is the natural endpoint once you support a list | LOW | Pure queryset-filter change (`parameters__proposal__in=[...]`, or drop the filter entirely for `ALL`); no new external calls. Depends on existing `SELECT-01` filter logic. |
| Facility scope covers every facility that shares the OCS request/observation shape (LCO + SOAR) | Once the command is "generalized," users expect it to mean "every queue facility we have," not "still just LCO" | LOW–MEDIUM | `facility__in=['LCO', 'SOAR']` plus confirming SOAR's `ObservationRecord.parameters` shape matches LCO's (same OCS backend) — verify this assumption against real SOAR records before shipping, same caution v1.2 learned the hard way for `instrument_type`/`site`. |
| Correct instrument-type extraction from multi-configuration requests | v1.2 shipped with a `KeyError`/silent-skip bug against 100% of real records in this DB; this is a baseline correctness bar, not a stretch feature | MEDIUM | Scan `c_1_instrument_type`..`c_5_instrument_type`, pick the one whose matching `c_N_ic_*_exposure_time` is populated. Needs a defined behavior for the (rare/invalid) case where 0 or >1 configs look "populated" — see Pitfalls/Anti-Features below. |
| Run summary counts (created/updated/unchanged/skipped) printed at command end | Already shipped in v1.2 (`self.stdout.write(...)` summary line) — must be preserved and extended with the new failure-mode counts (e.g. API-call-failed-fell-back-to-coarse-label) | LOW | Direct extension of existing pattern; add a counter, don't replace it. |
| Per-record skip-and-continue on data errors, with the offending record identified in stderr | Already shipped in v1.2 (`except (KeyError, ValueError) as exc: self.stderr.write(...)`) — this is the existing convention for *data* problems and must extend cleanly to the new *API-call* failure mode | LOW (reuse), MEDIUM (extending to a new failure axis) | This existing pattern is the right model for partial failure (see dedicated section below) — don't introduce a second, inconsistent error-handling style for the new per-record API call. |
| No-churn idempotent create-or-update | Already shipped (v1.2 SYNC-04); must keep holding once telescope label resolution adds a new field that could vary between API-success and API-fallback runs | MEDIUM | Risk: if a record's telescope label flips between "FTS" (API succeeded) and "1m0" (API failed, fallback) across re-runs, that's churn that looks like a real schedule change. Needs an explicit decision (see Pitfalls). |
| Verified (not assumed) site/telescope mapping for every site actually in use | v1.2's `SITE_TELESCOPE_MAP` was `[ASSUMED]`/web-search-only and only covered 2 of 8 real sites — this was flagged as a known gap, not a nice-to-have | LOW (data entry) once the mapping table is verified | PROJECT.md already supplies the verified 8-site MPC-code table; this is transcription + lookup-by-fully-qualified-code, not new research. |

### Differentiators (Competitive Advantage)

Features that go beyond "sync works" into "sync is trustworthy and informative" — these align with the project's actual differentiator: a *unified* calendar across heterogeneous facility data without per-facility custom UI.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-record live API enrichment for the *real* placed site/enclosure/telescope (not just the submission-time guess) | Submission-time `parameters` don't carry final placement for multi-site-eligible proposals (OCS scheduler can place a request at any site offering the right instrument class); a live API call to the observation's own endpoint is the only way to know the *actual* assigned site once scheduled | MEDIUM–HIGH | This is the most valuable new capability in v1.3 — turns "probably FTS" into "confirmed FTS, telescope `coj-clma-2m0a`". Must be opportunistic only (skip the call/use fallback for QUEUED/banner records that have no placement yet — there's nothing to fetch). |
| Graceful instrument-class fallback label (`1m0`/`0m4`/`2m0`) when the API call fails | Keeps the calendar useful (a one-line "something will run on a 1m at some point") even when the network/API is flaky, instead of erroring the whole record out of the calendar | MEDIUM | This is the new, riskiest UX surface — see dedicated section below for how it should read. |
| `ALL` proposal mode as an operator/admin "what's on the whole network" view, distinct from per-PI proposal views | Lets an observer or TAC see the full picture across proposals without re-running the command per code | LOW once multi-proposal works | Natural extension, not separate engineering. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Abort the whole sync run on first per-record error (fail-fast) | Feels "safer" — surfaces problems loudly | One bad/malformed record (the next "v1.2-style" data surprise) would block every other valid record from syncing, recreating exactly the all-or-nothing failure the original `instrument_type` bug caused. v1.2's own design already rejected this (skip-and-continue, SELECT-01/SYNC-04 pattern) | Keep per-record isolation; reserve hard-abort for *systemic* failures only (DB unreachable, can't construct `LCOFacility()`, malformed `--proposal` argument) — i.e. failures that mean *no* record could possibly succeed. |
| Retry-with-backoff loop around the per-record LCO API call | Looks more "robust" against transient network blips | Adds real complexity (timeout tuning, retry budget, risk of a single slow/down facility blocking the whole batch for minutes) for a feature whose correctness need is "best-effort label," not "guaranteed eventual accuracy" — the next scheduled cron run will naturally retry | Single attempt with a short timeout; let the *next* sync run's API call succeed once the network/API recovers. Document this explicitly so it isn't mistaken for a bug. |
| Treating an API-call failure as a data error that skips the record entirely (same bucket as a `KeyError` on missing `site`) | Reuses the existing skip-and-continue exception handling with minimal new code | A transient API failure is recoverable info-loss (we *can* still show something useful — instrument class — and the record's schedule window/instrument/proposal are still valid), unlike a malformed-data `KeyError` where nothing trustworthy can be shown at all. Conflating the two buckets would make the calendar silently drop records that are actually fine, just temporarily under-labeled | Keep "API call failed" as its own non-fatal path that still produces a `CalendarEvent` (with the coarse fallback label), distinct from "record data unusable" which is still a skip. |
| Caching/memoizing per-record API results to "speed up" the sync | Seems like an obvious efficiency win once the command runs on every proposal/facility | Adds a cache-invalidation problem (what site a request *will be* observed at can change between scheduler runs) for a script whose main cost is small (cron-frequency batch, not a hot path) — premature optimization that risks staleness bugs worse than the latency it saves | Recompute per run; only optimize if real run-time becomes a measured problem. |
| Silently re-labeling a record from the verified site name back to the coarse fallback (or vice versa) without any visual distinction in the calendar | Minimal-code path — just write whatever the current run resolved | A user reading the calendar across days would see telescope labels flicker between "FTS" and "2m0" with no indication *why*, eroding trust in the data and looking like real schedule churn | Make the fallback state visually/textually distinguishable (see dedicated section) so a flip reads as "we lost confidence in placement," not "the schedule changed." |

## Feature Dependencies

```
Multi-proposal / ALL support
    └──independent of──> Facility scope generalization (LCO+SOAR)
                              └──requires (shared assumption)──> SOAR ObservationRecord.parameters shape verified == LCO's
                                                                      (same risk class as v1.2's unverified instrument_type/site assumptions)

Correct instrument-type extraction (c_1..c_5 scan)
    └──blocks──> Telescope label resolution
                    (need a real instrument_type string before it's worth resolving/displaying a telescope label at all)

Verified static site/telescope mapping (8 sites)
    └──required by──> Per-record LCO API call enrichment
                          (the API returns a fully-qualified siteid-enclid-telid code; without the verified
                           dict there is nothing correct to map it to)
    └──required by──> Coarse instrument-class fallback
                          (fallback needs its own mapping, e.g. instrument_type prefix -> '1m0'/'0m4'/'2m0',
                           independent of the per-site dict but same "verified, not assumed" requirement)

Per-record LCO API call enrichment ──enhances──> No-churn create-or-update
    (must not introduce spurious churn when a record flips between API-success and fallback across runs —
     see Pitfalls; this is a conflict to actively design around, not a clean enhancement)

Multi-proposal/ALL + facility scope ──amplifies──> partial-failure handling
    (more records per run = higher probability that *some* record in the run will hit a data error or an
     API-call failure; the existing skip-and-continue pattern must scale, not just "still technically work")
```

### Dependency Notes

- **Correct instrument-type extraction blocks telescope label resolution:** there's no point building the API-enrichment/fallback machinery before the instrument string itself is reliably populated — extraction should land first (or in the same phase, sequenced first) so it can be tested independently of the new network-call logic.
- **Verified static mapping is a hard prerequisite for both the API path and the fallback path:** these are two different lookup tables (fully-qualified site/enclosure/telescope code -> label; instrument_type -> coarse class) but both must be "verified against real data," repeating the exact mistake category from v1.2 (`[ASSUMED]` 2-site dict). Do not ship either lookup table without grounding it in PROJECT.md's confirmed 8-site MPC-code table or equivalent real-API output.
- **API enrichment enhances (and complicates) no-churn:** this is the one place where a new feature pulls against an existing, validated guarantee (SYNC-04). It needs an explicit design decision before being built, not an incidental side effect discovered in code review.
- **Multi-proposal/facility scope amplifies partial-failure risk:** this isn't a new dependency so much as a reason the partial-failure design (already adequate for v1.2's single-proposal scope) needs to be revisited at v1.3's larger blast radius — see below.

## Partial-Failure Handling — Detailed Recommendation

This is explicitly the riskiest new behavior in v1.3 (per the question's framing), so it gets its own section rather than a single table row.

**The existing v1.2 pattern is the right foundation and should be extended, not replaced.** v1.2 already established: try to build the record's fields; on `(KeyError, ValueError)`, write a one-line message to `stderr` naming the `observation_id`, increment a `skipped_count`, and `continue` to the next record. The run never aborts because one record's data is bad. This matches the general batch-processing best practice found in the broader engineering literature (AWS Lambda partial-batch-response guidance, Mulesoft/Spring Batch "continue on error with summary reporting", ETL "skip-and-quarantine, don't halt the whole load" pattern) — isolate failures at the smallest unit (one record), keep processing everything else, and report a clear summary at the end. Reserve a full-run abort for *systemic* failures (e.g., the DB is unreachable, `LCOFacility()` itself can't be constructed, the `--proposal` argument is malformed) — not for per-record problems, where failing the whole batch destroys the value of every other record that was otherwise fine.

**For v1.3, the per-record LCO API call introduces a failure mode that is categorically different from v1.2's existing `(KeyError, ValueError)` data-shape failures, and it should be handled differently:**

1. **A failed API call is not a skip — it's a graceful degrade.** Unlike a missing `site` key or a malformed timestamp (where nothing trustworthy can be shown), a failed per-record API call still leaves you with a perfectly good instrument_type, proposal, schedule window, and status. The *only* thing lost is the fine-grained telescope label. The record should still get a `CalendarEvent`, just with the coarse fallback label substituted for the verified one — not added to `skipped_count`.
2. **Catch network/API failures narrowly and locally**, around just the API call itself (e.g. `requests.RequestException`, timeout, non-200 response, or unexpected response shape) — not by widening the existing `except (KeyError, ValueError)` around the whole `_build_event_fields` call, which would conflate "API down" with "data malformed" into one bucket and one count. Use a short request timeout (a few seconds) and a single attempt — no retry/backoff loop inside the sync run (see Anti-Features); the next scheduled run is the natural retry.
3. **Track it as its own counter** (e.g. `telescope_fallback_count`) distinct from `skipped_count`, and report it in the final summary line alongside created/updated/unchanged/skipped — extending the existing summary-line pattern rather than introducing a new reporting mechanism.
4. **Log which records fell back**, the same way v1.2 already logs which records were skipped (`stderr`, one line per record, naming the `observation_id`) — an operator should be able to see "N records used the coarse fallback this run" and know which ones, without needing to dig through the calendar UI.
5. **Decide explicitly how fallback interacts with no-churn** (this is the one open design question, flagged for the planner): if a record's resolved telescope label can legitimately flip between a verified label (API succeeded) and a coarse label (API failed) across successive runs, is that a "real" field change that should trigger an update + `modified` timestamp bump, or should the no-churn comparison ignore a fallback-vs-verified flip to avoid spurious churn? Recommend treating it as a real, visible change (update fires) **because the fallback label itself should be visually distinguishable** (see below) — a user seeing the label change should be able to tell "we briefly lost the fine-grained placement," which is true and useful information, not noise to suppress.

## Fallback-State UX Representation

No astronomy-domain-specific authoritative convention was found for "unknown specific site, known instrument class" labeling (this sub-question returned only generic UX guidance, confidence LOW). Falling back to the project's own established conventions plus general "graceful degradation should be visible, not invisible" UX principle:

- **Use the same bracketed-prefix convention v1.2 already established** for `[QUEUED]`/`[EXPIRED]`/`[CANCELLED]`/`[FAILED]` title prefixes — this is a convention the calendar's users are already trained to read. A natural extension is a similar marker for the fallback state, e.g. a title built from the coarse class directly (`'1m0 <instrument>'` instead of `'FTS <instrument>'`) rather than inventing a new bracket tag — the coarse label *is* the visible signal that finer placement wasn't available, no extra decoration needed.
- **Do not silently present the coarse fallback as if it were a confirmed site/telescope.** The whole point of the verified per-site mapping work in v1.3 is to be trustworthy; a fallback that looks identical to a real resolved label would undermine that for every record it touches.
- **Put the "why" in the description field, not just the title** — v1.2 already populates `CalendarEvent.description` with proposal/status/window text; append a line such as `'Telescope: coarse class only (API lookup failed)'` when in fallback mode, mirroring the existing description-as-detail-surface pattern rather than overloading the title with explanation text.
- **This is a UX/data-trust decision, not a hard engineering blocker** — flag it to the user/planner as a place where a short product decision (exact wording, whether to add a calendar-event field vs. reuse title/description) is worth 5 minutes of discussion before implementation, rather than researching further; there is no deeper domain consensus to discover here.

## MVP Definition

### Launch With (v1.3)

Minimum viable product for this milestone — all five target features are interdependent enough that a partial v1.3 would leave the command broken against real data again (same failure mode as v1.2).

- [ ] Multi-proposal/`ALL` `--proposal` support — trivial relative to the rest, but required for the stated scope
- [ ] Facility scope LCO+SOAR (with the SOAR-parameters-shape assumption explicitly verified against at least one real SOAR record before shipping, not assumed)
- [ ] Correct `c_1..c_5_instrument_type` extraction — this fixes the bug that made v1.2 non-functional against real data; non-negotiable
- [ ] Verified static 8-site mapping dict (already supplied by PROJECT.md's MPC-code table — transcription, not research)
- [ ] Per-record API call + coarse-instrument-class fallback, with fallback handled as a distinct non-fatal degrade path (own counter, own log line, visually distinguishable label) — per the detailed recommendation above

### Add After Validation (v1.3.x or later)

- [ ] Status-aware `CalendarEvent` coloring (already explicitly deferred per PROJECT.md's pending todo) — natural pairing with the fallback-state visual distinction problem above; revisit together
- [ ] Distinguishing the fallback-label flip from a real schedule change more explicitly than title text (e.g. a dedicated boolean/field on `CalendarEvent` if the project's `tom_calendar` model allows extension) — only worth it if operators report confusion from the title-text-only approach

### Future Consideration (v2+ / Stage 4)

- [ ] Gemini facility support — explicitly out of scope per PROJECT.md (different base class, no usable portal URL to key idempotent sync on)
- [ ] Retry/backoff for the per-record API call — only worth revisiting if real operational experience shows the single-attempt-then-fallback approach causes a persistently high fallback rate that isn't actually due to genuine API/network failures

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Correct `c_1..c_5_instrument_type` extraction | HIGH | MEDIUM | P1 |
| Verified 8-site mapping dict | HIGH | LOW | P1 |
| Multi-proposal/`ALL` support | MEDIUM | LOW | P1 |
| Facility scope LCO+SOAR | MEDIUM | LOW–MEDIUM (verification risk) | P1 |
| Per-record API enrichment + fallback (with distinct non-fatal handling) | HIGH | MEDIUM–HIGH | P1 |
| Fallback-state visual distinction in title/description | MEDIUM | LOW | P1 (bundled with the above — shipping fallback without it is the anti-feature flagged above) |
| Status-aware CalendarEvent coloring | MEDIUM | MEDIUM | P3 (already deferred) |
| Retry/backoff on API calls | LOW | MEDIUM | P3 (anti-feature unless evidence emerges) |

**Priority key:**
- P1: Must have for v1.3 launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration / explicitly deferred

## Competitor Feature Analysis

No direct competitor product was identified for "unified follow-up calendar across heterogeneous OCS-family queue facilities" — this appears to be a bespoke internal tool rather than a category with established commercial competitors. The closest comparison points are general patterns from adjacent domains, not competing products:

| Feature | General batch-sync tooling (AWS/ETL/Spring Batch) | LCO's own Observation Portal UI | Our Approach |
|---------|----------------------------------------------------|----------------------------------|--------------|
| Partial failure handling | Per-record isolation, continue, summarize at end; full-batch abort reserved for systemic errors | N/A (LCO portal shows its own data directly, no sync/aggregation step) | Extend v1.2's existing skip-and-continue convention; new API-failure path treated as non-fatal degrade, not skip |
| Unknown/partial data display | Generic guidance: design explicitly for partial/stale/failed states, show awareness rather than hiding gaps | N/A | Coarse instrument-class label as the visible signal; description field carries the "why" |
| Multi-source aggregation | ETL "quarantine bad rows, load the rest" pattern | N/A (single source) | Multi-proposal/multi-facility queryset filter, same record-level isolation extended across both axes |

## Sources

- Direct inspection: `/home/tlister/git/fomo_devel/solsys_code/management/commands/sync_lco_observation_calendar.py` (v1.2 shipped code — HIGH confidence, primary basis for partial-failure recommendation)
- `/home/tlister/git/fomo_devel/.planning/PROJECT.md` (v1.3 milestone scope, verified MPC-code/site table, v1.2 real-data bug findings — HIGH confidence, project-internal source of record)
- [Best practices for implementing partial batch responses - AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/lambda-event-filtering-partial-batch-responses-for-sqs/best-practices-partial-batch-responses.html) — LOW confidence (generic web search), corroborates skip-and-report-don't-abort pattern
- [Handling Errors During Batch Jobs | MuleSoft Documentation](https://docs.mulesoft.com/mule-runtime/4.3/batch-error-handling-faq) — LOW confidence, corroborates continue-on-error with max-failure circuit breaker pattern
- [ETL Error Handling - Tim Mitchell](https://www.timmitchell.net/post/2016/12/28/etl-error-handling/) — LOW confidence, corroborates skip-and-quarantine vs. fail-entire-batch tradeoff framing
- [Observation Portal | Observatory Control System](https://observatorycontrolsystem.github.io/components/observation_portal/) — LOW confidence (web search summary, not direct doc read), describes `/api/observations/` as carrying final site/enclosure placement distinct from request-time parameters
- [Observatory Control System | Open source software for an API-driven observatory](https://observatorycontrolsystem.github.io/) — LOW confidence, general OCS architecture context (Configuration Database models site/enclosure/telescope hierarchy)
- General UX fallback-state guidance ("Error handling - UX design patterns" and related search results) — LOW confidence, no domain-specific authoritative source found for astronomy-scheduling fallback labeling; recommendation in this document leans primarily on the project's own existing `[QUEUED]`/`[EXPIRED]` bracket-prefix convention rather than this external search result

---
*Feature research for: astronomical follow-up scheduling — LCO-family OCS queue sync*
*Researched: 2026-06-19*
