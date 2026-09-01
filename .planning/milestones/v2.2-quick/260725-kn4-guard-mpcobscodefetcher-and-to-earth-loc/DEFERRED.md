# Deferred Items — quick task 260725-kn4

Two items this fix deliberately did not take on:

1. **Whether satellite sites should be excluded from campaign coverage-gap analysis
   entirely.** This is a separate design decision, out of scope for this quick task. After
   this fix, `campaign_gap.observable_dates()` will treat every date at a satellite site as
   "unknown" via its existing D-03 `ValueError` skip (since `to_earth_location()` now
   raises `ValueError` rather than crashing with `TypeError`), which is a safe holding
   pattern but probably not the desired end state — a satellite site arguably shouldn't be
   scanned for observable-but-unclaimed ground-based dates at all.

2. **`docs/design/telescope_runs_calendar.rst` line ~344's Open Questions bullet** says a
   guard against space-based observatories in `to_earth_location()`/`sun_event()` "would be
   needed". That guard now exists (Task 2 of this quick task), so the bullet is stale and
   should be closed out in a future docs pass.
