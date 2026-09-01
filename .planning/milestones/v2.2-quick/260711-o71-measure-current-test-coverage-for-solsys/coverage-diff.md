# Coverage Diff: 260711-o71 (CR-01/CR-02 permanent regression tests)

Compares `coverage-before.txt` (baseline, 417 tests) against `coverage-after.txt`
(after adding `TestSiteSelectionNameCandidateResolution` and
`TestCreateObservatoryTemplateNextRoundTrip` to `test_campaign_approval.py`, 419 tests),
both from `coverage run --source=solsys_code manage.py test solsys_code` +
`coverage report`.

## Overall

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Statements (TOTAL) | 5550 | 5612 | +62 (new test file lines counted as statements under `--source=solsys_code`) |
| Missed (TOTAL) | 168 | 168 | 0 |
| Coverage % (TOTAL) | 97% | 97% | flat |

## Three touched files (CR-01/CR-02 code paths)

| File | Before (Stmts/Miss/Cover) | After (Stmts/Miss/Cover) | Delta |
|------|---------------------------|---------------------------|-------|
| `solsys_code/campaign_views.py` | 182 / 8 / 96% | 182 / 8 / 96% | flat |
| `solsys_code/campaign_utils.py` | 172 / 10 / 94% | 172 / 10 / 94% | flat |
| `solsys_code/solsys_code_observatory/views.py` | 59 / 7 / 88% | 59 / 7 / 88% | flat |

## Takeaway

The two new tests exercise `CampaignRunDecisionView.post()`'s display-string->obscode
mapping (`campaign_views.py:380`, the CR-01 fix line) and the `CreateObservatory`
template `next` round-trip (`solsys_code_observatory/views.py`'s `get_success_url()`/
`get_initial()`), but the per-file `Stmts`/`Miss`/`Cover` numbers are identical
before and after. This is a legitimate result, not a null result: those specific lines
were already reached by other existing tests in this module (e.g.
`TestSiteSelectionResolution.test_staff_typed_existing_obscode_resolves_via_site_selection_tier_1_hit`
already executes the `build_site_candidates().get(...)` line with a literal-obscode
input, and `TestCreateObservatoryRoundTrip.test_valid_create_with_safe_next_redirects_to_approval_queue`
already executes `get_success_url()`'s validated-redirect branch with a hand-built POST
body) — so line coverage was already at its ceiling for these branches. The point of
this task was never to move the coverage percentage; it was to close the
*regression*-coverage gap `21-REVIEW.md`/`21-VERIFICATION.md` flagged: those two
existing tests never proved the *display-string candidate lookup* (CR-01) or the *real
template's hidden field* (CR-02) specifically, only adjacent/similar code paths. A
future refactor that broke either fix while keeping the surrounding lines reachable via
the pre-existing tests would previously have gone undetected; it will now fail
`TestSiteSelectionNameCandidateResolution` or `TestCreateObservatoryTemplateNextRoundTrip`
respectively.
