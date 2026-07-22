# Phase 2: Run Line Parsing - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 2-Run Line Parsing
**Areas discussed:** Telescope name resolution, Output shape & module location, Status vocabulary, Malformed-line & multi-line handling

---

## Telescope name resolution

| Option | Description | Selected |
|--------|-------------|----------|
| Raw token, unresolved | Parser emits the literal token ('Magellan', 'NTT') as-is | |
| Validate against SITES now | Parser checks token resolves to a known telescope, adding a bare 'Magellan' alias | |
| You decide | Claude picks based on testability | |
| (free text) | Resolve against SITES, raise + present matches for ambiguous entries | ✓ |

**User's choice:** Resolve against SITES by prefix match; exact match resolves directly, multiple prefix matches (e.g. 'Magellan' -> Clay/Baade) raise `ValueError` listing candidates, no match raises `ValueError`.
**Notes:** Follow-up confirmed this exact matching rule (D-01).

**Conflict discovered after this area:** ROADMAP.md's Phase 3 success criteria expected the two `Magellan ...` sample lines to successfully create 7 and 5 events. User chose to keep D-01 as-is and revise ROADMAP.md instead — the two Magellan sample lines are now documented as Phase 2 error-path fixtures (ValueError on ambiguous telescope), and Phase 3's numeric success criteria use only the `NTT EFOSC2 allocation 9-13 July` -> 5 events fixture. ROADMAP.md Phase 3 criterion 1 updated accordingly.

---

## Output shape & module location

| Option | Description | Selected |
|--------|-------------|----------|
| telescope_runs.py, dataclass | Add parse_run_line() to telescope_runs.py, returns ParsedRun dataclass | ✓ |
| telescope_runs.py, plain tuple | Same module, returns a 7-tuple per design doc literal wording | |
| New module | Separate solsys_code/telescope_run_parser.py | |

**User's choice:** telescope_runs.py + dataclass (D-02, D-03).
**Notes:** None.

---

## Status vocabulary

| Option | Description | Selected |
|--------|-------------|----------|
| Free-text string, default 'allocation' | status = whatever word appears, lowercased; default 'allocation' if absent | (partial) |
| Free-text string, default None/empty | Same extraction, default None/empty if absent | |
| Fixed enum | Restrict to known set, raise on unrecognized | (partial) |

**User's choice:** Hybrid — "leaning towards [fixed enum] but like the flexibility of [free-text default]; maybe multiple words in the enum or smart matching". Follow-up confirmed: small extendable known-status set (allocation/proposed/confirmed/cancelled), case-insensitive, multi-word phrases allowed, default 'allocation' if absent, **raise ValueError** on unrecognized status text.
**Notes:** Captured as D-04, D-05, D-06.

---

## Malformed-line & multi-line handling

| Option | Description | Selected |
|--------|-------------|----------|
| Single-line + ValueError | parse_run_line(line) raises ValueError on anything unparseable; Phase 3 handles file iteration | ✓ |
| Single-line + None on failure | Returns None instead of raising | |

**User's choice:** Single-line + ValueError (D-07).
**Notes:** None.

---

## Claude's Discretion

- Exact regex/parsing strategy for date-range orderings, month-name parsing, hyphenated-instrument handling
- Exact `ValueError` message wording
- `ParsedRun` implementation (dataclass / frozen dataclass / NamedTuple)

## Deferred Ideas

- Resolving Magellan-Clay vs Magellan-Baade ambiguity (e.g. richer input format or `Observatory.short_name`-based lookup) — out of scope per design doc "Open Items"; revisit if real-world lines hit the ambiguity `ValueError` often.
