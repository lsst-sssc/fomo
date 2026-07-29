---
status: complete
phase: 08-telescope-label-verification-sidecar
source: [08-VERIFICATION.md]
started: 2026-06-25T13:50:00Z
updated: 2026-06-25T14:30:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Visual perceptibility of the dashed border on a rendered calendar page
expected: Open `/calendar/` in a browser with at least one fallback-labeled event present (seed via the demo notebook's fallback fixture, or run `sync_lco_observation_calendar` against a record whose API call times out). The fallback event's block shows a visibly dashed border distinct from neighboring solid-bordered/unstyled events, perceivable at a normal glance without opening the event or reading its title.
result: pass

### 2. Native tooltip appears on hover in a real browser
expected: Hover over a fallback-labeled event block on the rendered calendar page. The browser's native `title=` tooltip appears, showing the plain-language "estimate... could not be verified against the LCO API... coarse fallback label" sentence.
result: pass

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
