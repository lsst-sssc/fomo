# Phase 18: Uncertain-Scheduling Investigation Spike - Decision

**Investigated:** 2026-07-09
**Status:** Findings recorded (SCHED-01 criteria 2-5) against the real 3I/ATLAS coordination
sheet (2026-07-09 snapshot) and the live local `Observatory` DB/MPC Obscodes API.
Recommendation and Durable summary to be completed in Plan 02.

This phase is investigation-only, mirroring Phase 13's ESO feasibility spike. No
`CampaignRun` schema migration, no `import_campaign_csv`/`parse_obs_window()` change, and no
fuzzy-match UI code ships from this plan — the sole committed deliverable is this findings
record, built from a throwaway, git-excluded probe script (`fuzzy_match_probe.py`, never
staged, per D-08) run by the executor against the real CSV (D-01 path) and the local
`Observatory` DB (read-only; every `resolve_site()` call was wrapped in a rolled-back
`transaction.atomic()` block — `Observatory.objects.count()` was 8 at both the start and the
end of the run, confirming nothing was persisted).

## Findings

### SCHED-01 criterion 3 — CSV range/TBD cell shapes

The real 2026-07-09 CSV snapshot (30 data rows, header located at row index 2 after a
2-row public-editing-notice preamble — the probe locates the header by content, not by a
fixed row offset) shows every shape D-03/D-04/D-05 anticipated, plus two additional
non-key-column shapes (literal `TBD` text; an approx-hour-with-seconds marker) already
covered by the existing `_APPROX_HOUR` regex family. One bullet per real shape, each with a
redacted verbatim example (Contact Person / Email columns are never read or printed by the
probe — redacted per D-01):

- **Blank `Obs. Date` entirely.** `telescope='ESO VLT FORS2' site='309' obs_date='' ut_range='TBD'`
- **`" to "`-separated full-date range in `Obs. Date`.** `telescope='LCO 1m' site='' obs_date='2025-07-05 to 2025-09-22' ut_range=''` (also `LCO 2m`: `obs_date='2026-01-15 to 2026-01-22'`)
- **Compact same-month range in `Obs. Date`.** `telescope='JUICE' site='' obs_date='2025-11-02 -25' ut_range=''`
- **`YYYY-MM-?` month-known-day-TBD marker in `Obs. Date`.** `telescope='JWST' site='500@-170' obs_date='2025-12-?' ut_range=''` (two distinct rows share this exact value — see criterion 2 below)
- **D-04: a multi-day range typed into `UT Time Range` instead of `Obs. Date`.** `telescope='HST STIS/COS' site='250' obs_date='2025-11-27' ut_range='2025-11-27 to 2025-12-10'` (also confirmed on `Swift/UVOT`: `obs_date='2025-07-11' ut_range='2025-07-11 to 2025-07-13'`, and `VLT/UVES`: `obs_date='2025-08-11' ut_range='2025-08-11 to 2025-08-19'`) — a parser that only inspects `Obs. Date` for range syntax would silently miss all three of these real rows.
- **D-05: a stray copy-paste-artifact `UT Time Range` cell (genuine unparseable garbage, not a date/time shape at all).** `telescope='Palomar P200/NGPS' site='675' obs_date='2025-07-03' ut_range="le is for the coordination of past and future observations of ISO 3I/ATLAS. It is fully publicly editable via GitHub pull requests or through this Google sheet. Please do not edit other people's entries or make any major changes to the overall sheet without conta"` (a fragment of the sheet's own preamble text, confirming the D-05 finding is still present in the live sheet).
- **Literal `TBD` text in `UT Time Range`** (several rows, e.g. `telescope='FTN/MuSCAT3' obs_date='2025-07-12' ut_range='TBD'`) and **an approx-hour marker with seconds baked in** (`telescope='Palomar P200/NGPS' obs_date='2025-07-03' ut_range='~7:00:00 AM'`) — both already handled gracefully today (see below).

`parse_obs_window()` spot-check (block E of the probe) confirms today's exact behavior
against these shapes: it **raises `ValueError` on every range/TBD `Obs. Date` shape**
(blank, `" to "` range, compact-range, and the `2025-12-?` marker all raised — e.g.
`parse_obs_window('2025-12-?', '')` -> `ValueError: time data '2025-12-?' does not match
format '%Y-%m-%d'`), confirming these are true natural-key failures the current parser
rejects (the exact gap Phase 20 must close). By contrast, when `Obs. Date` itself is a
valid exact date but `UT Time Range` is the messy/range/garbage column (the D-04 and D-05
cases), `parse_obs_window()` never raises — it falls back to midnight UTC with
`ut_needs_review=True` (e.g. `parse_obs_window('2025-11-27', '2025-11-27 to 2025-12-10')`
-> `(date(2025, 11, 27), datetime(2025, 11, 27, 0, 0, tzinfo=utc), None, True)`, and the
Palomar garbage-artifact case behaves identically). This confirms the existing "never raise
on the non-key time column, fall back and flag needs-review" discipline
(`_HHMM_RANGE`/`_APPROX_HOUR`/`_BARE_HOUR_UTC`) is still the right posture — Phase 20 should
extend this pattern-per-shape approach to `Obs. Date` itself rather than replacing it with a
general-purpose date parser.

### SCHED-01 criterion 2 — TBD natural-key collision (real evidence)

The real D-06 collision is confirmed present in the live 2026-07-09 snapshot: exactly two
rows share `Telescope / Instrument = 'JWST'` and `Obs. Date = '2025-12-?'`. Contact
Person / Email withheld (redacted per D-01); the `Filter(s)/Bandpass` column is the
load-bearing distinguishing evidence:

```
{'Telescope / Instrument': 'JWST', 'Site Code': '500@-170', 'Obs. Date': '2025-12-?',
 'UT Time Range': '', 'Filter(s)/Bandpass': 'MIRI MRS, 5-28 microns',
 'Observation Details': 'IFU Spectroscopy', 'Observation Status': 'Upcoming'}

{'Telescope / Instrument': 'JWST', 'Site Code': '500@-170', 'Obs. Date': '2025-12-?',
 'UT Time Range': '', 'Filter(s)/Bandpass': 'NIRSpec prism (0.6-5.3 um), NIRSPec grating
 (2.7-5.3 um), NIRCam, and MIRI LRS (5-14 um)', 'Observation Details': 'IFU Spectroscopy,
 Imaging, Spectroscopy', 'Observation Status': 'Upcoming'}
```

These are two genuinely distinct rows (different instrument config within JWST, distinct
contact persons per D-06 though not printed here) that collide on
`(campaign, telescope_instrument, window_start)` alone once `window_start` is `NULL` for
both (day-unknown). This is the concrete real-data evidence for the decision already locked
in CONTEXT.md D-06: extend the TBD-row natural key with `contact_person` (already a
`CampaignRun` field, populated on every real row) so these two rows don't merge. The exact
partial/conditional-constraint mechanism (e.g. a conditional `UniqueConstraint` including
`contact_person` only when `window_start IS NULL`) is Phase 19's job to design; this finding
just confirms the real evidence the design must satisfy.

### SCHED-01 criterion 4 — Fuzzy-match library live comparison

Live comparison of `rapidfuzz.process.extractOne` (three scorers: `WRatio`,
`token_sort_ratio`, `token_set_ratio`, all at `score_cutoff=60`) and
`difflib.get_close_matches` (`cutoff=0.6`, the normalized equivalent per RESEARCH Pitfall 3)
against the D-09 real messy `Site Code` corpus, run against the actual live candidate pool
built from `Observatory.objects.values_list('obscode', 'name', 'short_name', 'old_names')`
(Contact Person / Email are not part of this pool and were never read — redacted per D-01,
though moot since this table has no PII fields):

| Raw code | rapidfuzz WRatio | rapidfuzz token_sort | rapidfuzz token_set | difflib (n=3, cutoff=0.6) |
|----------|-------------------|----------------------|----------------------|---------------------------|
| `X09` (Sam Deen, Deep Random Survey / 43cm) | `('309', 66.67, 2)` | `('309', 66.67, 2)` | `('309', 66.67, 2)` | `['809', '309']` |
| `N50` (HCT) | `None` | `None` | `None` | `[]` |
| `X07` (Josep Trigo-Rodríguez, Deep Sky Chile) | `None` | `None` | `None` | `[]` |
| `C65` (Telescope Joan Oró, Montsec, Catalonia) | `('F65', 66.67, 11)` | `('F65', 66.67, 11)` | `('F65', 66.67, 11)` | `['F65']` |
| `` (D-07 blank/no-code case) | `None` | (not separately run) | (not separately run) | `[]` |

The candidate pool itself (25 distinct non-empty strings) was:
`['268', '269', '309', '705', '809', 'APO', 'Apache Point Observatory', 'DCT', 'E10',
'ESO, La Silla', 'European Southern Observatory, Paranal', 'F65', 'FTN', 'FTS',
'Faulkes Telescope North', 'Magellan Baade Telescope', 'Magellan Clay Telescope',
'Magellan-Baade', 'Magellan-Clay', 'NEEDS REVIEW: DCT', 'NTT', 'Siding Spring Observatory',
'VLT', "['New Horizons KBO Search-Magellan/Baade']", "['New Horizons KBO Search-Magellan/Clay']"]`.

Two important real findings from this live run:

1. **difflib's `[]` on `N50`/`X07`/blank is expected default behavior, not a defect** (per
   RESEARCH Pitfall 3) — both codes genuinely have no similar candidate in the pool at the
   `cutoff=0.6` threshold, and the blank case returning `[]` is exactly the documented
   "good enough or nothing" semantics.
2. **The two "hits" (`X09`->`'309'`, `C65`->`'F65'`) are almost certainly false positives,
   not meaningful matches.** Both are 3-character alphanumeric strings scoring ~67% purely
   from character-position overlap with an unrelated local `Observatory.obscode` — `X09` and
   `309` share two digits in the same position; `C65` and `F65` differ by one character.
   Neither `309` (Paranal VLT) nor `F65` (Faulkes Telescope North) has any real relationship
   to Sam Deen's Deep Random Survey or the Joan Oró telescope. This is a genuine live-test
   finding, not a documentation-based guess: **the current local `Observatory` table (8
   rows, all FOMO's own curated facilities) is far too narrow a candidate pool to
   meaningfully fuzzy-match arbitrary external site codes reported by 3I/ATLAS
   contributors** — none of the D-09 corpus's real external sites are present in it at all,
   so any "match" the pool produces is coincidental character overlap, not a real candidate
   the algorithm found. This is a scoping consideration Plan 02 needs to weigh (e.g. whether
   the future fuzzy-match UI's candidate pool should be the live MPC observatory list, or
   whether it should only run after resolve_site()'s own Tier 1/2 already missed) — not
   resolved here, no winning library is picked in this section per D-08 (that verdict is
   Plan 02's).

### SCHED-01 criterion 5 — resolve_site() MPC-code confirmation

`Observatory._meta.get_field('obscode').max_length` = **4**. `250`, `274`, and `289` are all
3 characters and fit comfortably within it — the default verdict of "no widening needed"
holds, confirmed directly against the live field definition, not assumed.

Live `resolve_site(..., create_placeholder=False)` results, each wrapped in a
`transaction.atomic()` block rolled back via `transaction.set_rollback(True)`
(`Observatory.objects.count()` was 8 before and after every call in this block — nothing
persisted):

| Code | Confidence level | Result | Mechanism |
|------|-------------------|--------|-----------|
| `500@-170` | Confirmed against real rows (the real JWST `Site Code` value, D-09) | `(None, True)` | Length guard (8 chars > `max_length=4`) — flagged before any tier is attempted, no network call made. |
| `250` | Confirmed against real rows (Jewitt's/Noonan's Hubble rows both use `250`) | `(None, True)` | **Not the length guard** — see finding below. |
| `274` | Constructed-input code-path check (no real row in the current snapshot types plain `274`) | `(None, True)` | Same mechanism as `250` — see finding below. |
| `289` | Constructed-input code-path check (no real row types plain `289`) | `(None, True)` | Same mechanism as `250`/`274` — see finding below. |

**Important unexpected real finding for `250`/`274`/`289`:** none of the three actually
reach the length guard or a real Tier-1/Tier-2 miss — all three genuinely exist in the live
MPC Obscodes API (independently confirmed with a direct `requests.get` call this session:
`250` -> `{"name": "Hubble Space Telescope", "observations_type": "satellite", "longitude":
null, ...}`, similarly for `274` James Webb Space Telescope and `289` Nancy Grace Roman
Space Telescope, all `HTTP 200`). `resolve_site()`'s Tier 2 (`MPCObscodeFetcher.query()`)
successfully finds all three, but `MPCObscodeFetcher.to_observatory()` then raises
`TypeError: float() argument must be a string or a real number, not 'NoneType'` when it
executes `elong = float(self.obs_data['longitude'])` — the MPC API returns `longitude: null`
for these space-observatory (`"observations_type": "satellite"`) records, since a space
telescope has no fixed geodetic position. This exception is caught by `resolve_site()`'s
existing `except (KeyError, ValueError, TypeError): pass` clause (documented there as a
WR-04 hardening fix for "a malformed-but-'ok' response"), which falls through to Tier 3 —
skipped since `create_placeholder=False` — producing the same safe `(None, True)` outcome as
the length-guard case, but via a genuinely different code path and root cause. This was
independently reproduced and confirmed directly (rolled back, no persistence):
`MPCObscodeFetcher().query('250', timeout=10)` returns `obs_data['longitude'] = None`,
`obs_data['observations_type'] = 'satellite'`; calling `.to_observatory()` on that data
raises the exact `TypeError` above.

This is a real, live-test-confirmed finding, not a hypothetical: **`resolve_site()` today
cannot actually resolve any of the three standard space-observatory MPC codes (250/274/289)
via its Tier 2 MPC-API path**, even though all three are legitimate, well-formed MPC
records that Tier 2's `query()` call successfully retrieves — the failure happens one step
later, in `to_observatory()`'s unconditional `float(longitude)` conversion, which has no
guard for the `null` longitude the MPC API returns for `"satellite"`-type entries. The
outcome is still safe (flagged for manual review, nothing fabricated, nothing persisted) —
consistent with the existing "never fabricate" discipline — but the *reason* is not "the
code doesn't exist" or "the code is over-length," it's an unhandled `null`-coordinate edge
case in `to_observatory()` specific to satellite-type MPC records. Phase 19/21 planners
should be aware of this distinct root cause if resolving these codes for real ever becomes
a requirement (out of scope for this phase — no code change is made here, per the
investigation-only phase boundary).

## Recommendation

<!-- completed in Plan 02 -->

## Durable summary

<!-- completed in Plan 02 -->
