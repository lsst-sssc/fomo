# Phase 18: Uncertain-Scheduling Investigation Spike - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-08
**Phase:** 18-uncertain-scheduling-investigation-spike
**Areas discussed:** Real-data access, TBD natural-key collision handling, validation depth for fuzzy-match/obscode confirmation

---

## Real-data access for CSV range/TBD parsing rules & fuzzy-match testing

| Option | Description | Selected |
|--------|-------------|----------|
| You supply real examples now | Tim pastes/describes actual (or redacted) cell text directly in discussion | |
| You point me to a local file | Tim gives a local file path to the real sheet CSV, read directly (not committed) | ✓ |
| Construct representative examples | No real data available; build plausible examples from known column structure | |

**User's choice:** "You point me to a local file"
**Notes:** Tim provided the path:
`/mnt/c/Users/liste/OneDrive/Documents/Asteroids/3I/3I_ATLAS Observations and Observing
Plans - Sheet1.csv`. Read successfully — real, messier-than-expected data (see CONTEXT.md
D-03..D-10 for the findings this produced).

---

## TBD row duplicate-prevention strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Always create new on re-import | TBD row identity is ambiguous until resolved; every import creates a fresh row | |
| Disambiguate like Phase 14 did | Deterministic per-batch offset/counter distinguishes TBD rows within one import | |
| Let the spike decide from the data | Don't lock now; weigh against real TBD examples once found | ✓ |

**User's choice:** "Let the spike decide from the data"
**Notes:** Follow-up analysis of the real CSV found a genuine collision: Matthew
Belyakov's JWST/MIRI row and Martin Cordiner's JWST/NIRSpec row share `campaign`,
`telescope_instrument = "JWST"`, and both have day-unknown `Obs. Date = "2025-12-?"`
(→ `window_start = NULL` in the new schema). This real evidence led to a targeted
follow-up question (below) rather than leaving the question fully open for Plan 18.

## Follow-up: how to resolve the real TBD collision found

| Option | Description | Selected |
|--------|-------------|----------|
| Add contact_person to the key | Extend the TBD natural key with the existing `contact_person` field to distinguish Belyakov from Cordiner | ✓ |
| No uniqueness enforcement for TBD rows | Partial constraint only applies when window_start IS NOT NULL; TBD rows never deduplicated by the DB | |

**User's choice:** "Add contact_person to the key (Recommended)"
**Notes:** Locked as CONTEXT.md D-06 — the real collision case is the recorded rationale
Phase 19's migration design should cite.

---

## Validation depth for fuzzy-match library choice & resolve_site() confirmation

| Option | Description | Selected |
|--------|-------------|----------|
| Live-test both | Actually run rapidfuzz vs. difflib and resolve_site() live, capture real output (mirrors Phase 13 precedent) | ✓ |
| Reason from docs only | Faster, no throwaway script/temp install; compare algorithmically and confirm via code-reading only | |

**User's choice:** "Live-test both (Recommended)"
**Notes:** `rapidfuzz` is not to be added to `pyproject.toml` this phase — install
temporarily/scratch-only for the comparison, same throwaway-script treatment as Phase
13's `eso_p2_probe.py`. Real messy `Site Code` test corpus identified from the sheet:
`X09`, `N50`, `X07`, `C65`, and the `"500@-170"` JWST-notation case (see CONTEXT.md D-09).

---

## Claude's Discretion

- Exact wording/structure of the decision doc(s).
- Single doc vs. full-detail + durable-summary pair (Phase 13's D-10 pattern).
- Exact redaction approach for quoted real-sheet examples.
- How far to take actual parsing-rule/regex design in this phase's decision doc vs.
  leaving it as a documented recommendation for Phase 19/20's planning.

## Deferred Ideas

None new. One adjacent finding (D-07 in CONTEXT.md — some real rows have no resolvable
`Observatory` at all, e.g. blank site code for "LCO 1m"/"LCO 2m"/Swift/JUICE rows, which
breaks ASSET-01's "derive ground-vs-space from resolved site" premise in the no-site
case) was logged as a note for Phase 20's research, not treated as new scope for this
phase.
