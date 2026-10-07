---
id: SEED-261007-j63
status: dormant
planted: 2026-10-07T02:59:35.000Z
planted_during: v2.4 intent review walkthrough (before /gsd-complete-milestone)
trigger_when: when attribution scoring, gap-analysis site resolution or allocation fetchability next changes
scope: small
audit_acknowledged:
  milestone: v2.4
  at: 2026-10-06
  status: dormant
---

# SEED-261007-j63: Per-site obscode sets for LCO site codes, with membership checks

## Why This Matters

`campaign_attribution.LCO_SITE_CODE_TO_OBSCODE` maps a site code to ONE obscode and today holds only `coj → E10`, because an LCO site has several (Cerro Tololo alone: W85/W86/W87/W89/I02/807). It feeds attribution scoring (equality with the run's `site.obscode`), gap-analysis site resolution (`campaign_gap.observation_site_obscode()`) and allocation fetchability (`proposal_allocation`). So `cpt`, `elp`, `lsc`, `tfn` resolve to nothing in all three. The F13 fix (quick task 261006-nga) worked around this for the tally only, with a tally-private site-code → timezone map.

## When to Surface

**Trigger:** when attribution scoring, gap-analysis site resolution or allocation fetchability next changes

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Small**

## Breadcrumbs

- `.planning/v2.4-INTENT-REVIEW.md` — Q8a, Q8b, F11, F13 (2026-10-06)

## Notes

- Change the table to per-site obscode SETS (`lsc → {W85, W86, W87, …}`) and turn the equality checks into membership: attribution matches any dome at the site; gap analysis resolves a site rather than asserting one dome; fetchability covers every dome.
- The tally could then take its timezone from any member Observatory and drop its private map.
- Developer's preferred general fix (2026-10-06), over seeding one representative 1m obscode per site, which would assert a dome the data does not support.
