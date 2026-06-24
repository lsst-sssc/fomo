# Phase 5: Multi-Proposal & Multi-Facility Selection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-19
**Phase:** 5-Multi-Proposal & Multi-Facility Selection
**Areas discussed:** Multi-proposal / ALL syntax, SOAR facility settings & credentials, Per-record facility dispatch & defensive behavior, Run summary reporting

---

## Multi-proposal / ALL syntax

**Q: How should whitespace and case be handled when parsing the comma-separated proposal list?**

| Option | Description | Selected |
|--------|-------------|----------|
| Strip whitespace, exact-case codes | Split on comma, strip each piece, keep proposal-code casing as-is for the filter | ✓ |
| Strip whitespace, lowercase-normalize codes | Same whitespace handling, but also lowercase both input codes and comparison | |

**Q: Should the special 'ALL' token be matched case-sensitively or case-insensitively?**

| Option | Description | Selected |
|--------|-------------|----------|
| Case-insensitive 'ALL' | Accept 'all'/'All'/'ALL' — friendlier for CLI operators | ✓ |
| Case-sensitive 'ALL' only | Only the exact uppercase literal triggers sync-everything | |

**Q: How should duplicate or empty entries in the list be handled?**

| Option | Description | Selected |
|--------|-------------|----------|
| Silently dedupe and drop empties | Split, strip, drop empty strings, dedupe | ✓ |
| Error out on empty/duplicate entries | Reject the whole command with a usage error | |

**Notes:** None — all three questions answered with the recommended option.

---

## SOAR facility settings & credentials

**Q: Should this phase add FACILITIES['SOAR'] to src/fomo/settings.py?**

| Option | Description | Selected |
|--------|-------------|----------|
| Add FACILITIES['SOAR'] in this phase | SOAR sync can't work end-to-end without it | ✓ |
| Leave it as a deploy-time/ops task | Write dispatch code assuming it'll exist in production, don't add it here | |

**Q: Should FACILITIES['SOAR'] reuse LCO's api_key/portal_url env vars, or use separate SOAR-specific ones?**

| Option | Description | Selected |
|--------|-------------|----------|
| Duplicate LCO's value via the same env var | Matches SOARFacility's documented behavior (same LCO Observation Portal API) | ✓ |
| Separate SOAR-specific env var | Introduce a distinct SOAR_API_KEY/SOAR_PORTAL_URL even though it'd hold the same value | |

**Notes:** None — both questions answered with the recommended option.

---

## Per-record facility dispatch & defensive behavior

**Q: If a record's facility value is neither 'LCO' nor 'SOAR', what should happen?**

| Option | Description | Selected |
|--------|-------------|----------|
| Skip + log, continue run | Same per-record error-handling convention as the rest of the command | ✓ |
| Hard-fail the whole run | Treats it as a hard invariant violation | |

**Q: Should the dispatch dict be built eagerly once, or lazily per facility seen?**

| Option | Description | Selected |
|--------|-------------|----------|
| Build both instances eagerly, once | Simplest, matches today's single-eager-instance pattern extended to two keys | ✓ |
| Lazily instantiate only facilities seen in this run | Avoids constructing an unused SOARFacility() instance on LCO-only runs | |

**Notes:** None — both questions answered with the recommended option.

---

## Run summary reporting

**Q: Should the summary line break counts down by facility, or stay aggregate-only?**

| Option | Description | Selected |
|--------|-------------|----------|
| Per-facility breakdown | e.g. 'LCO: 3 created... \| SOAR: 2 created...' — surfaces facility-specific problems at a glance | ✓ |
| Aggregate-only (current behavior) | Keep the existing single combined-count summary line | |

**Notes:** None — answered with the recommended option.

---

## Claude's Discretion

- Exact stdout formatting/line layout of the per-facility summary, as long as both facilities' counts are each individually visible.
- Exact log message wording for the unexpected-facility skip case and the settings-key addition's surrounding comments/docstring.
- Whether comma-list/ALL parsing is a small helper function or inlined in `add_arguments`/`handle()`.

## Deferred Ideas

None — discussion stayed within phase scope. No scope-creep items were raised.
