---
status: testing
phase: 31-foundation-spikes-run-identity-unattended-invocation
source: [31-VERIFICATION.md]
started: 2026-09-02T23:49:56Z
updated: 2026-09-02T23:49:56Z
---

## Current Test

number: 1
name: Accept (or reject) plan 31-06's prohibition against reopening any of the four settled verdicts
expected: |
  Operator confirms no settled decision was reopened — the correction is evidence framing only, as
  scoped. Non-authoritative verifier judgment: NOT violated. Evidence: `git diff --name-status
  ad6fd76..19146ee` touches 5 files, none under `solsys_code/` or `src/`; the complete set of
  deleted lines across both committed artifacts is 10, every one of which was re-added with its
  original text intact plus an appended dated qualification; all four verdict anchors are
  byte-present at HEAD (`null=True, blank=True`; `max_length=500` +
  `unique_campaign_run_source_identifier` + `source_identifier__isnull=False` at
  31-DECISION.md:663-680; '**not sufficient**' at :806; `/usr/bin/flock -n
  /var/lock/fomo/<command-name>.lock` at :927), and the four `### SCHEMA-0x` /
  `### Scheduling track (SCHED-07)` verdict headings are unchanged in count and order.
awaiting: user response

## Tests

### 1. Accept (or reject) plan 31-06's prohibition against reopening any of the four settled verdicts
expected: Operator confirms no settled decision was reopened — the correction is evidence framing
  only, as scoped. Declared `verification: judgment`; the verifier's own read is NOT violated, but
  judgment-tier prohibitions are a documented soft-gate never silently passed regardless of how
  strong the mechanical evidence is.
result: [pending]

### 2. Accept (or reject) plan 31-06's prohibition against silently rewriting an earlier finding
expected: Operator confirms the correction idiom was followed and no earlier finding was quietly
  erased. Declared `verification: judgment`; the verifier's own read is NOT violated — the three
  deleted 31-DECISION.md lines (Block (E) Gemini row, SCHEMA-02 per-ingest-path Gemini row, the
  LCO_QUEUE/GEMINI_QUEUE/CLASSICAL_FILE bullet) were each replaced by the identical original
  sentence plus a dated "**Corrected 2026-09-02:**" clause, matching the document's existing
  correction idiom. Whether a rewrite is "silent" is a reading judgment, not a grep.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
