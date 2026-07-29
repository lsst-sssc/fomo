---
status: resolved
trigger: "Typing 'Nordi' into the Observing Site field on the public 'Submit an Observing Run' form (for 'Nordic Optical Telescope, La Palma (Z23)', a genuine MPC obscode not yet a local Observatory row) shows the exact same suggestion 'Nordic Optical Telescope, La Palma (Z23)' twice in the dropdown list — a duplicate, not two distinct candidates."
created: 2026-07-16T14:00:00Z
updated: 2026-07-16T15:00:00Z
resolution: "Confirmed by user spot-check on a different affected record (obscode 434, 'Benedetto') resolving to a single entry with no duplicate, as suggested for systemic verification. Committed as commit 40f9bf5. 518/518 tests pass, ruff clean."
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: |
  'Nordi' is a substring match, so substring_or_fuzzy_match_candidates() returns
  hits[:limit] WITHOUT ever calling the fuzzy fallback (the `if hits: return` guard).
  So hypothesis-1 (fuzzy backfill dedup gap) is NOT the cause. The duplicate must be
  two DISTINCT keys in build_site_candidates()'s pool that both contain 'nordi' and both
  map to Z23, but render visually-identically as 'Nordic Optical Telescope, La Palma'.
  Candidate keys for Z23's MPC record: [code='Z23', name_utf8, short_name, *old_names].
  'Z23' doesn't contain 'nordi', so two of {name_utf8, short_name, old_names} must be
  distinct-but-visually-identical strings (whitespace/unicode diff, or one is a suffix/
  variant of the other that still renders the same visible glyphs).
test: |
  Reproduce in Django shell: inspect the live MPC record for Z23 (query_all()), then call
  substring_or_fuzzy_match_candidates('Nordi', build_site_candidates()) and repr() every
  result tuple to see the exact bytes of the two matching keys.
expecting: |
  Two result tuples whose display strings are byte-different but glyph-identical, both ->
  Z23. That reveals whether the dedup gap is in _flatten_mpc_candidates (two MPC fields)
  or a local/MPC merge.
next_action: |
  Apply fix: add _normalize_candidate() (collapse whitespace runs + strip, matching HTML
  render) and apply it to candidate strings in _flatten_mpc_candidates() and
  _local_observatory_candidates() so visual-duplicate keys collapse via the existing
  first-seen dedup. Add regression test with a Z23-shaped fixture.

status_note: fixing

reasoning_checkpoint:
  hypothesis: |
    Two byte-distinct candidate strings for the same obscode (name_utf8 vs an old_names
    whitespace-variant) both survive the exact-string dedup in _flatten_mpc_candidates(),
    both substring-match the query, and HTML collapses their whitespace so they render
    as identical duplicate suggestions.
  confirming_evidence:
    - "Z23 raw record: name_utf8 'Nordic Optical Telescope, La Palma' (1 space) vs old_names 'Nordic Optical Telescope,     La Palma' (5 spaces) -- direct repr from live query_all()."
    - "substring_or_fuzzy_match_candidates('Nordi', pool) returns exactly those 2 byte-distinct tuples, both -> Z23 -- reproduced live."
    - "Audit: 32/2,712 records have >=2 candidate strings collapsing to one visible form -- systemic, not Z23-only."
  falsification_test: |
    If, after normalizing candidate whitespace when building the pool, the pool still
    contains two distinct keys for Z23 that both contain 'nordi', the hypothesis is wrong.
    Conversely, if collapsing whitespace at pool-build reduces Z23 to a single 'Nordic
    Optical Telescope, La Palma' key and the search returns one result, it is confirmed.
  fix_rationale: |
    The redundant candidate originates in the pool build: the existing dedup is keyed on
    exact bytes. Normalizing each candidate string to its visible-render form (' '.join(
    s.split())) before the 'candidate not in mapping' check makes the EXISTING first-seen
    dedup collapse visual-duplicates at the source -- fixing every consumer (substring,
    fuzzy, selection_to_obscode) uniformly and generally (all 32 records), not just Z23.
    Normalization is a visual no-op (HTML already collapses whitespace) so displayed text
    is unchanged; only redundant byte-variant keys are removed.
  blind_spots: |
    (1) A local Observatory whose name is a whitespace-variant of its MPC name -- addressed
    by normalizing _local_observatory_candidates() too, so the merge collapses it.
    (2) Not tested: whether any downstream code relies on the exact un-normalized old_names
    string as a pool key (searched: only selection_to_obscode consumes pool keys, and it
    reads back the same normalized key the template rendered).

---

## Symptoms

expected: |
  Typing 'Nordi' (a substring of Z23's MPC name_utf8 'Nordic Optical Telescope, La Palma')
  into the public submission form's Observing Site live-search widget should show the site
  ONCE in the suggestion dropdown, mapping to obscode Z23 — one suggestion per genuinely
  distinct candidate, even if multiple underlying MPC fields (name_utf8/short_name/old_names)
  happen to match the same query.
actual: |
  The dropdown shows 'Nordic Optical Telescope, La Palma (Z23)' TWICE, as two separate rows
  with byte-for-byte identical text — a literal duplicate of the same candidate, not two
  different real candidates for the same site.
errors: |
  None visible to the user — no exception, no 500. Purely a rendering/duplication issue in
  the returned suggestion list.
timeline: |
  Found via manual browser testing shortly after the 260716-h8c quick task (Observatory
  timezone backfill) was committed. Z23 (Nordic Optical Telescope) is confirmed NOT to exist
  as a local Observatory row yet, so this suggestion must be coming entirely from the MPC
  candidate pool (build_site_candidates()'s MPC branch / _flatten_mpc_candidates()) rather
  than a local-vs-MPC pool overlap. This is on the PUBLIC submission form's widget
  (site_raw), not the approval-queue resolve widgets used in the recent debug sessions
  (site-search-mpc-no-match, site-resolve-list-old-names, site-search-degraded-pool-
  recurrence) -- though it may share the same underlying substring_or_fuzzy_match_candidates()
  matching code path.
reproduction: |
  1. Go to the public 'Submit an Observing Run' form (/campaigns/submit/).
  2. Type 'Nordi' into the 'Observing site' field.
  3. Wait for the debounced suggestion list to render.
  4. Observe: 'Nordic Optical Telescope, La Palma (Z23)' appears twice, identically, in the
     dropdown -- see screenshot at
     /home/tlister/.claude/image-cache/945378df-779f-43c5-8119-fb584930b1b2/1.png

## Evidence

- timestamp: 2026-07-16T14:20:00Z
  checked: Live MPC record for Z23 via MPCObscodeFetcher().query_all() in Django shell.
  found: |
    Z23's raw record: name_utf8 = 'Nordic Optical Telescope, La Palma' (single space
    after comma), short_name = identical, old_names = ['Nordic Optical Telescope,
    La Palma'] with FIVE spaces after the comma. _flatten_mpc_candidates() emits THREE
    keys for Z23: 'Z23', 'Nordic Optical Telescope, La Palma' (from name_utf8; short_name
    dedups away as an exact match), and 'Nordic Optical Telescope,     La Palma' (from
    old_names). The name_utf8 and old_names keys are byte-distinct (1 vs 5 spaces) so the
    'candidate not in mapping' exact-string dedup does NOT collapse them.
  implication: |
    Both keys contain 'nordi' so both substring-match. HTML collapses the 5-space run to
    one space on render, so the two suggestions display byte-for-byte identically. Root
    cause = visual-duplicate candidate keys, not a fuzzy-backfill dedup gap.

- timestamp: 2026-07-16T14:22:00Z
  checked: Direct call substring_or_fuzzy_match_candidates('Nordi', build_site_candidates()).
  found: |
    Returns 2 tuples: ('Nordic Optical Telescope, La Palma', 'Z23') and ('Nordic Optical
    Telescope,     La Palma', 'Z23'). Reproduces the duplicate exactly. The fuzzy fallback
    was NOT invoked (substring hits existed), confirming hypothesis 1 (fuzzy dedup gap) is
    NOT the cause.
  implication: |
    Confirmed hypothesis 2: two distinct MPC fields (name_utf8 and an old_names variant)
    that both substring-match but render identically.

- timestamp: 2026-07-16T14:26:00Z
  checked: Audit across all 2,712 live MPC records for >=2 distinct candidate strings that
    collapse to the same whitespace-normalized form.
  found: |
    32 records affected (obscodes 199, 434, 709, D50, D63, I45, K50, K52, K74, L07, L23,
    L98, M10, M24, M26, M30, M32, M45, N88, U63, V40, V94, W54, W62, W95, Y05, Y93, Z23,
    Z26, Z42, Z71, Z88). Nearly all are trailing-space old_names variants; Z23 is the only
    internal-multi-space one. Every one would produce the same duplicate-suggestion bug.
  implication: |
    Systemic dedup gap, NOT Z23-specific. Fix must be general: dedup candidate strings on
    their whitespace-normalized (visible-render) form, not their exact bytes.

## Eliminated

- hypothesis: |
    substring_or_fuzzy_match_candidates() appends fuzzy (difflib) results on top of
    substring results without deduping, producing the duplicate (grounding hypothesis 1).
  evidence: |
    'Nordi' is a substring hit, so the `if hits: return hits[:limit]` guard returns before
    the fuzzy fallback is ever reached. The two duplicate results both come from the
    substring pass over two distinct pool keys. Verified by direct call (Evidence entry 2).
  timestamp: 2026-07-16T14:22:00Z

- hypothesis: |
    A local Observatory row for Z23 overlaps the MPC pool and double-counts (grounding
    hypothesis 4).
  evidence: |
    Observatory.objects.filter(obscode='Z23').count() == 0 in the live DB. Both duplicate
    keys map to Z23 and both originate from the MPC record's fields (name_utf8 + old_names).
  timestamp: 2026-07-16T14:20:00Z

## Resolution

root_cause: |
  The candidate pool built by build_site_candidates() can hold two byte-distinct strings
  that are the same site name differing only in internal/trailing whitespace runs (MPC's
  old_names often carries a whitespace-variant of the current name -- 32/2,712 live
  records, e.g. Z23's name_utf8 'Nordic Optical Telescope, La Palma' vs its old_names
  'Nordic Optical Telescope,     La Palma'). The dedup in _flatten_mpc_candidates() (and
  _local_observatory_candidates()) is keyed on the EXACT string ('candidate not in
  mapping'), so these visual-duplicates both survive into the pool and both substring-match
  the same query. HTML collapses their whitespace on render, so the user sees two
  byte-for-byte identical suggestions.
fix: |
  Added _normalize_candidate() in campaign_utils.py (collapses whitespace runs + strips,
  matching HTML's visible-text rendering: ' '.join(value.split())). Applied it to each
  candidate string in _flatten_mpc_candidates() AND _local_observatory_candidates() BEFORE
  the existing first-seen 'candidate not in mapping' dedup, so two byte-distinct strings
  that render identically collapse to one candidate at the pool's source. General fix (all
  32 affected records), not Z23-specific. Displayed text is unchanged (browser already
  collapses whitespace); only redundant byte-variant keys are removed.
verification: |
  - Live Django shell (post-fix, cache busted): _flatten_mpc_candidates(query_all()) yields
    exactly one 'nordi' key for Z23 ('Nordic Optical Telescope, La Palma'); the 5-space
    old_names variant collapsed. substring_or_fuzzy_match_candidates('Nordi', pool) returns
    exactly 1 result. Audit of all 32 previously-affected obscodes: 0 still carry a visual
    duplicate.
  - New regression tests (WhitespaceVariantDedupTest, 4 tests) FAIL on pre-fix code (3
    failures + 1 error, showing the exact duplicate tuple) and PASS with the fix -- verified
    by git-stashing the campaign_utils.py change.
  - Full suite green: test_campaign_site_search + test_campaign_approval = 116 tests OK.
  - ruff check + ruff format --check clean on both modified files.
files_changed:
  - solsys_code/campaign_utils.py (added _normalize_candidate; applied in _flatten_mpc_candidates + _local_observatory_candidates)
  - solsys_code/tests/test_campaign_site_search.py (added WhitespaceVariantDedupTest regression class)
