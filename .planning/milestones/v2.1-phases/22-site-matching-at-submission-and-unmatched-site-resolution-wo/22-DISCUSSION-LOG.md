# Phase 22: Site Matching at Submission and Unmatched-Site Resolution Workflow - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-14
**Phase:** 22-site-matching-at-submission-and-unmatched-site-resolution-wo
**Areas discussed:** Endpoint access & abuse protection, What picking a suggestion means, Resolution surface & action, Widget behavior details

---

## Endpoint access & abuse protection

| Option | Description | Selected |
|--------|-------------|----------|
| Anonymous + throttled | Open to anonymous users with server-side rate limiting; pool already 24h-cached | ✓ |
| Anonymous, no throttle | Rely on client debounce + cached pool; accepts hostile-client CPU risk | |
| Login required | Blocks anonymous submitters — defeats the public form's purpose | |

**User's choice:** Anonymous + throttled — after first asking whether throttling
requires extra libraries/dependencies. Clarified: no — a ~10-line per-IP counter on the
existing Django cache framework (same cache `build_site_candidates()` uses) needs zero
new packages; DRF throttling exists but fits awkwardly with HTML fragments;
`django-ratelimit` would be a new dependency and is not needed.
**Notes:** Exact rate limit (e.g. 30–60 req/min) left to planning.

| Option | Description | Selected |
|--------|-------------|----------|
| HTML fragment | HTMX convention: server renders suggestion-list partial, hx-get swaps it in; zero custom JS | ✓ |
| JSON + small JS | JSON API + custom JS to render suggestions; more moving parts | |

**User's choice:** HTML fragment.

---

## What picking a suggestion means

The first framing of this question was rejected so the user could ask a clarifying
question: how feasible is pre-submit resolution — typing 'Faulkes' resolving to F65 or
E10, 'Lowell' to the various MPC Lowell sites, from the full cached list?

Investigation answer: highly feasible — the pool already maps every MPC record's
obscode/name_utf8/short_name/old_names to its obscode — but NOT with the existing
difflib-only `fuzzy_match_candidates()` (whole-string similarity scores 'Faulkes' vs
'Haleakala-Faulkes Telescope North' ~0.35, below the 0.6 cutoff). Requires a
substring-containment pass first. This reshaped the questions:

| Option | Description | Selected |
|--------|-------------|----------|
| Substring first, difflib fallback | Case-insensitive containment over cached pool; difflib only as typo fallback | ✓ |
| difflib only (status quo matcher) | Partial names like 'Faulkes' won't match long official MPC strings | |

**User's choice:** Substring first, difflib fallback.

| Option | Description | Selected |
|--------|-------------|----------|
| Store text; resolve at approval | Picked display string becomes site_raw; exact-matchable at approve time; zero model changes; no anonymous DB writes | ✓ |
| Resolve immediately at submit | run.site set at submission; anonymous traffic could trigger tier-2 Observatory creation | |

**User's choice:** Store text; resolve at approval.

---

## Resolution surface & action

| Option | Description | Selected |
|--------|-------------|----------|
| Section on approval-queue page | Third table (pending / decided / sites-needing-review) on the existing staff page | ✓ |
| Own page + navbar link | Dedicated page; second place staff must remember to check | |
| Filter on the decided table | Make unresolved rows in decided table actionable; mixes history with actions | |

**User's choice:** Section on approval-queue page.

| Option | Description | Selected |
|--------|-------------|----------|
| New 'resolve_site' action on decide view | Third action alongside approve/reject; reuses pool mapping + resolve_site(create_placeholder=False); fires factored-out calendar projection in the same request | ✓ |
| Dedicated CampaignRunSiteResolveView | Separate staff-only POST view; duplicates guard/mapping wiring | |

**User's choice:** New 'resolve_site' action on decide view.

---

## Widget behavior details

| Option | Description | Selected |
|--------|-------------|----------|
| No 'Create new Observatory' link on public form — staff-only | Site creation stays a vetted staff action; free text never blocks submission | ✓ |
| Yes — on the public form too | Exposes staff CRUD flow to the public; invites junk rows | |

**User's choice:** No — staff-only.

| Option | Description | Selected |
|--------|-------------|----------|
| Claude's discretion on fine-tuning | ~2-char minimum, ~300ms debounce, ~8-suggestion cap, no-match hint copy | ✓ |
| Discuss each now | Walk through each knob individually | |

**User's choice:** Claude's discretion.

---

## Claude's Discretion

- Exact throttle rate and cache-key scheme.
- Live-search fine-tuning: min characters, debounce delay, suggestion cap, no-match
  hint copy, input-population behavior.
- Endpoint URL naming; shared endpoint vs context flag for form/queue.
- Resolve-failure UX copy in the sites-needing-review table.
- Sites-needing-review table row cap / ordering.

## Deferred Ideas

None. One todo reviewed and not folded (extract site/telescope mapping from LCO sync
command — rejected as unrelated in Phases 13/18/21 as well).
