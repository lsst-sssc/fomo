"""Unit tests for campaign_reconciler.reconcile_run() (plan 29-01, Task 3).

Covers RECON-02 (queue half), RECON-03, RECON-05, RECON-06's dry-run, and RECON-01's
unit-level idempotency, isolated from the Django view/command layer. Fixture style mirrors
CampaignApprovalTestBase in test_campaign_approval.py.

Migrated onto the `ALLOC:` namespace for Phase 35 plan 35-02: 35-01 replaced the classical
per-night `RUN:{pk}:{date}` writer with a peer `allocation_projector` module, and D-10
inverted queue-sourced dispatch to the whole-window `RUN:{pk}` container regardless of a
resolved ground site. See 35-02-SUMMARY.md's classification table for the full per-class
kept/migrated/retired audit trail.
"""

from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.allocation_projector import allocation_events, writable_allocation_events
from solsys_code.calendar_utils import record_time_window
from solsys_code.campaign_reconciler import _may_write as _reconciler_may_write
from solsys_code.campaign_reconciler import (
    event_title,
    owned_events,
    reconcile_run,
    writable_events,
)
from solsys_code.campaign_reconciler import split_telescope_instrument as _split_telescope_instrument
from solsys_code.models import CalendarEventDismissal, CalendarEventMeta, CampaignRun, CampaignRunObservation
from solsys_code.solsys_code_observatory.models import Observatory
from solsys_code.telescope_runs import observing_night, sun_event


class CampaignReconcilerTestBase(TestCase):
    """Shared fixture: one campaign, one resolvable Australian ground Observatory (per-night
    default, allocation-dispatched under D-09/D-10), one Chilean ground Observatory
    (America/Santiago -- lets the migrated boundary-sensitive classes pick either hemisphere
    without adding a second base class, Task 1's own instruction) and one satellite one."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        cls.ground_site = Observatory.objects.create(
            obscode='F65',
            name='Faulkes Telescope South',
            short_name='FTS',
            lat=-31.2727,
            lon=149.0644,
            altitude=1149.0,
            timezone='Australia/Sydney',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        cls.chile_ground_site = Observatory.objects.create(
            obscode='W85',
            name='LCO Cerro Tololo 1m',
            short_name='CTIO-1m',
            lat=-30.1673,
            lon=-70.8046,
            altitude=2198.0,
            timezone='America/Santiago',
            observations_type=Observatory.OPTICAL_OBSTYPE,
        )
        cls.satellite_site = Observatory.objects.create(
            obscode='250',
            name='Test Space Telescope',
            short_name='TST',
            observations_type=Observatory.SATELLITE_OBSTYPE,
        )

    def _make_run(self, **overrides) -> CampaignRun:
        """Create a CampaignRun; kwargs override the default (approved, ground-sited,
        LEGACY-sourced -- and therefore, under D-09/D-10, allocation-dispatched) field set."""
        kwargs = {
            'campaign': self.campaign,
            'telescope_instrument': 'FTN/MuSCAT3',
            'site': self.ground_site,
            'site_raw': 'F65',
            'window_start': date(2026, 8, 1),
            'window_end': date(2026, 8, 1),
            'observation_details': 'Photometric monitoring',
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    def _link_record(
        self,
        run: CampaignRun,
        *,
        scheduled_start: datetime | None = None,
        scheduled_end: datetime | None = None,
        facility: str = 'LCO',
        status: str = 'COMPLETED',
    ) -> tuple[ObservationRecord, CampaignRunObservation]:
        """Create an ObservationRecord (NonSiderealTargetFactory target -- CLAUDE.md) and
        link it to `run` via a CampaignRunObservation. Returns (record, link)."""
        target = NonSiderealTargetFactory.create()
        owner = User.objects.create(username=f'obs-owner-{uuid4().hex[:8]}')
        record = ObservationRecord.objects.create(
            target=target,
            user=owner,
            facility=facility,
            observation_id=f'obs-{uuid4().hex[:8]}',
            status=status,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters={'proposal': 'TEST'},
        )
        link = CampaignRunObservation.objects.create(run=run, observation_record=record)
        return record, link


class TestSkipReasons(CampaignReconcilerTestBase):
    """One test per _skip_reason() branch (D-05's itemized skip vocabulary). Kept unchanged
    -- the stage-0 guard is untouched by Phase 35."""

    def test_pending_review_run_is_not_approved(self):
        run = self._make_run(approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'not approved')
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_blank_telescope_instrument_is_missing_telescope_instrument(self):
        run = self._make_run(telescope_instrument='')

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'missing telescope/instrument')
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_unset_window_start_is_tbd_window(self):
        run = self._make_run(window_start=None, window_end=None)

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'TBD window')
        self.assertEqual(CalendarEvent.objects.count(), 0)

    def test_no_site_and_no_telescope_class_is_unresolved_site(self):
        run = self._make_run(site=None, site_raw='', telescope_class='')

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'unresolved site')
        self.assertEqual(CalendarEvent.objects.count(), 0)


class TestQueueSourceDispatchesToContainer(CampaignReconcilerTestBase):
    """D-10 (Phase 35): a queue-scheduled (lco_queue/soar_queue/gemini_queue/eso_queue) run
    dispatches to the single whole-window `RUN:{pk}` container REGARDLESS of a resolved
    ground site -- inverting the premise quick task 260805-tad established (a queue-sourced
    run with a resolved site used to take the classical per-night branch there). Only a
    non-blank `telescope_class` selected the container branch before D-10; now `source`
    alone decides for these four queue values, read directly off the stored field, never
    inferred from a telescope name or a site."""

    def test_lco_queue_run_with_resolved_site_creates_one_bare_container(self):
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(
            source=CampaignRun.Source.LCO_QUEUE,
            window_start=window_start,
            window_end=window_end,
        )

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')
        self.assertEqual(allocation_events(run).count(), 0)

    def test_soar_queue_run_with_resolved_site_creates_one_bare_container(self):
        """No coverage existed for SOAR_QUEUE in this class before -- added per Task 1's
        instruction alongside the LCO/Gemini/ESO cases."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(
            source=CampaignRun.Source.SOAR_QUEUE,
            window_start=window_start,
            window_end=window_end,
        )

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')
        self.assertEqual(allocation_events(run).count(), 0)

    def test_gemini_queue_run_with_resolved_site_creates_one_bare_container(self):
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(
            source=CampaignRun.Source.GEMINI_QUEUE,
            window_start=window_start,
            window_end=window_end,
        )

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')
        self.assertEqual(allocation_events(run).count(), 0)

    def test_eso_queue_run_with_resolved_site_creates_one_bare_container(self):
        """ESO_QUEUE added in plan 29-06 (user-directed deviation, see 29-06-SUMMARY.md):
        real 3I/ATLAS ESO VLT rows needed a dedicated queue source. D-10 now routes it to
        the container branch alongside the other three queue sources, superseding quick task
        260805-tad's per-night fix for the ESO_QUEUE/RUN:3 case."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(
            source=CampaignRun.Source.ESO_QUEUE,
            window_start=window_start,
            window_end=window_end,
        )

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')
        self.assertEqual(allocation_events(run).count(), 0)

    def test_queue_sourced_run_with_telescope_class_still_gets_one_bare_container(self):
        """The inverse control, unchanged in outcome (a `telescope_class` run was already
        container-dispatched before D-10 too): a queue-scheduled run that is ALSO genuinely
        class-wide (no fixed site) still gets exactly one bare container."""
        run = self._make_run(
            source=CampaignRun.Source.LCO_QUEUE,
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 10),
        )

        result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')


class TestClassWideStage2(CampaignReconcilerTestBase):
    """RECON-03: a class-wide (or SPACE-classed) run projects a single bare container. Kept
    unchanged -- unaffected by D-09/D-10."""

    def test_class_wide_site_less_run_creates_one_container_and_is_not_skipped(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.TWO_M0,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 10),
        )

        result = reconcile_run(run)

        self.assertIsNone(result.skipped_reason)
        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')

    def test_space_classed_run_shares_the_same_container_branch(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.SPACE,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 10),
        )

        result = reconcile_run(run)

        self.assertIsNone(result.skipped_reason)
        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        self.assertEqual(events.get().url, f'RUN:{run.pk}')


class TestSatelliteContainer(CampaignReconcilerTestBase):
    """The ported satellite case: one bare RUN:{pk} whole-day-span event, no sun_event()
    call. Kept unchanged in intent -- unaffected by D-09/D-10 -- but the patch target moves:
    35-01 dropped `campaign_reconciler.py`'s own `sun_event` import entirely (the module no
    longer calls it at all, classical or otherwise), so patching it there now raises
    AttributeError. Patched at its source module (`telescope_runs.sun_event`) instead, which
    still guards against a call from anywhere -- `_reconcile_container()` (the branch this
    satellite run actually takes) never called it, before or after Phase 35."""

    def test_satellite_run_creates_one_container_event_without_calling_sun_event(self):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError('sun_event() must never be called for a satellite run')

        run = self._make_run(
            site=self.satellite_site,
            site_raw='250',
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 5),
        )

        with patch('solsys_code.telescope_runs.sun_event', side_effect=_fail_if_called):
            result = reconcile_run(run)

        self.assertEqual(result.created, 1)
        events = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(events.count(), 1)
        event = events.get()
        self.assertEqual(event.url, f'RUN:{run.pk}')
        self.assertEqual(event.start_time, datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(event.end_time, datetime(2026, 8, 5, 23, 59, tzinfo=dt_timezone.utc))


class TestOwnershipScoping(CampaignReconcilerTestBase):
    """RECON-05: the reconciler never creates, modifies or deletes an event it does not own.
    The two per-night cases are migrated onto the `ALLOC:` namespace (D-09): a resolved-site
    run with no queue source is allocation-dispatched now, not `RUN:`-per-night-dispatched."""

    def test_unowned_same_window_event_is_left_completely_untouched(self):
        """A hand-made event (blank url, no companion row) whose start_time falls inside the
        run's window is never adopted, modified or linked to a CalendarEventMeta row. Runs
        the allocation branch (LEGACY-sourced, resolved site -- D-09's per-night dispatch)
        over a 2-night window -- window length is not this test's point."""
        run = self._make_run(
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 2),
        )
        orphan = CalendarEvent.objects.create(
            title='Unrelated conference',
            url='',
            start_time=datetime(2026, 8, 2, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 2, 12, 0, tzinfo=dt_timezone.utc),
        )
        modified_before = orphan.modified

        reconcile_run(run)

        orphan.refresh_from_db()
        self.assertEqual(orphan.title, 'Unrelated conference')
        self.assertEqual(orphan.start_time, datetime(2026, 8, 2, 10, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(orphan.end_time, datetime(2026, 8, 2, 12, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(orphan.modified, modified_before)
        self.assertFalse(CalendarEventMeta.objects.filter(event=orphan).exists())

    def test_event_owned_by_a_different_run_is_blocked_and_untouched(self):
        """An event already keyed under this run's ALLOC:{pk}:{date} namespace, but whose
        companion row points at a DIFFERENT run, is blocked -- never written, never
        re-attributed. Single-night window: a resolved-site, non-queue run takes the
        allocation branch (D-09), so the clashing event is keyed at the allocation night
        url, not the bare container url."""
        night = date(2026, 8, 1)
        run = self._make_run(
            window_start=night,
            window_end=night,
        )
        other_run = self._make_run(
            telescope_instrument='Other Telescope/Instrument',
            window_start=night,
            window_end=night,
        )
        clashing_event = CalendarEvent.objects.create(
            title='Owned by a different run',
            url=f'ALLOC:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=clashing_event, run=other_run)
        modified_before = clashing_event.modified

        result = reconcile_run(run)

        self.assertEqual(result.blocked, 1)
        clashing_event.refresh_from_db()
        self.assertEqual(clashing_event.title, 'Owned by a different run')
        self.assertEqual(clashing_event.modified, modified_before)

    def test_owned_events_trailing_colon_guard_excludes_a_different_runs_night(self):
        """owned_events(run) for run pk=3 must not match an event keyed RUN:34:2026-08-01.
        Kept unchanged: this is `owned_events()`'s own `RUN:` prefix discipline, still
        applicable to the container namespace regardless of Phase 35."""
        run = self._make_run()
        # Force a low, predictable pk gap is unnecessary -- just create another run with a
        # numerically-later pk and assert its per-night event never matches run's query.
        other_run = self._make_run(telescope_instrument='Other Telescope/Instrument')
        other_event = CalendarEvent.objects.create(
            title='Other run night',
            url=f'RUN:{other_run.pk}:2026-08-01',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )

        self.assertNotIn(other_event, list(owned_events(run)))

    def test_may_write_agrees_with_both_queryset_twins_for_every_shape(self):
        """35-REVIEW.md NF-06: `_may_write()` (the row-level predicate) and its two
        namespace-specific queryset twins -- `writable_events()` for `RUN:`,
        `writable_allocation_events()` for `ALLOC:` -- must agree for every companion-row
        shape, at both namespaces. Before the fix, `_may_write()`'s fallback only ever
        matched the `RUN:` namespace, so it disagreed with `writable_allocation_events()`
        for shapes (a) and (b) at the `ALLOC:` namespace: the queryset admitted them, the
        predicate refused them. Concrete expected values are asserted alongside the
        agreement so the test cannot pass by both sides being wrong together."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        other_run = self._make_run(telescope_instrument='Other Telescope/Instrument')

        def _make_event(namespace_run, shape, url):
            event = CalendarEvent.objects.create(
                title=f'{namespace_run.pk}-{shape}',
                url=url,
                start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
                end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
            )
            if shape == 'a':
                pass  # no CalendarEventMeta companion row at all
            elif shape == 'b':
                CalendarEventMeta.objects.create(event=event, run=None)
            elif shape == 'c_other':
                CalendarEventMeta.objects.create(event=event, run=other_run)
            elif shape == 'c_this':
                CalendarEventMeta.objects.create(event=event, run=run)
            return event

        cases = []
        for shape, expected in (('a', True), ('b', True), ('c_other', False), ('c_this', True)):
            alloc_url = f'ALLOC:{run.pk}:{night.isoformat()}-{shape}'
            run_url = f'RUN:{run.pk}:{night.isoformat()}-{shape}'
            cases.append((_make_event(run, shape, alloc_url), 'ALLOC', shape, expected))
            cases.append((_make_event(run, shape, run_url), 'RUN', shape, expected))

        writable_alloc_ids = set(writable_allocation_events(run).values_list('pk', flat=True))
        writable_run_ids = set(writable_events(run).values_list('pk', flat=True))

        for event, namespace, shape, expected in cases:
            predicate_result = _reconciler_may_write(event, run)
            queryset_ids = writable_alloc_ids if namespace == 'ALLOC' else writable_run_ids
            queryset_result = event.pk in queryset_ids
            self.assertEqual(
                predicate_result,
                expected,
                f'_may_write() disagreed with the expected value for {namespace} shape {shape}',
            )
            self.assertEqual(
                queryset_result,
                expected,
                f'the {namespace} queryset twin disagreed with the expected value for shape {shape}',
            )
            self.assertEqual(
                predicate_result,
                queryset_result,
                f'_may_write() and the {namespace} queryset twin disagreed for shape {shape}',
            )


class TestContainerIdempotency(CampaignReconcilerTestBase):
    """RECON-01 (unit level) and RECON-06's dry-run. Kept unchanged -- unaffected by
    D-09/D-10 (this is the class-wide container branch)."""

    def test_second_reconcile_is_unchanged_and_dry_run_matches(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )

        first = reconcile_run(run)
        self.assertEqual(first.created, 1)
        event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        modified_after_first = event.modified

        second = reconcile_run(run)
        self.assertEqual(second.unchanged, 1)
        self.assertEqual(CalendarEvent.objects.count(), 1)
        event.refresh_from_db()
        self.assertEqual(event.modified, modified_after_first)

        third = reconcile_run(run, dry_run=True)
        self.assertEqual(third.unchanged, 1)
        self.assertEqual(CalendarEvent.objects.count(), 1)
        event.refresh_from_db()
        self.assertEqual(event.modified, modified_after_first)

    def test_dry_run_on_never_reconciled_run_reports_created_and_writes_nothing(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )

        result = reconcile_run(run, dry_run=True)

        self.assertEqual(result.created, 1)
        self.assertEqual(CalendarEvent.objects.count(), 0)
        self.assertEqual(CalendarEventMeta.objects.count(), 0)


# TestAttributedNightSkip -- RETIRED (named reason, no destination module needed).
#
# Covered the retired `_attributed_nights()`/skip-the-night rule: a night already
# attributed through a non-`RUN:` event was skipped entirely by `_reconcile_classical_nights()`.
# 35-01 deleted that function and its dispatch call, and `_attributed_nights()` itself is
# now dead code in campaign_reconciler.py (no caller). The allocation projector's handoff
# (D-05/D-07) supersedes this rule: a linked, placed/observed record's night is DELETED
# outright, never skipped-in-place -- see test_allocation_projector.TestObservationHandoff.

# TestObservingNightBoundary -- RETIRED, destination test_allocation_projector.py.
#
# `_observing_night()`'s (now `telescope_runs.observing_night()`, promoted 35-01) noon-anchor
# boundary coverage for both hemispheres is fully duplicated by
# test_allocation_projector.TestAllocationNightBoundary (8 tests: Sydney UTC-date-differs,
# exact-local-noon, one-second-before, post-local-midnight; Chile's mirror of all four) --
# confirmed present before retiring this class.


class TestReconcileThenAttributeOrdering(CampaignReconcilerTestBase):
    """CR-03 (33-REVIEW.md) migrated for Phase 35: the reconcile-then-attribute and
    attribute-then-reconcile orderings still converge, but the mechanism differs by branch.

    For a per-night ALLOCATION-dispatched run, the "attribution supersedes a minted night"
    case is now the observation handoff (D-05/D-07): a linked, placed/observed record's
    night is DELETED outright, never detached, and unlinking re-mints it (D-07's "no audit
    of its own" -- allocation nights carry no `confirmed_by` survival guarantee, unlike a
    `RUN:` container).

    The human-confirmation guard cases (`_stale_attributions()`/`_detach_stale_family_events()`)
    move to a fixture where a CLASS-WIDE container-dispatched run is re-classified to
    allocation dispatch, leaving its bare `RUN:{pk}` container stale -- the shape
    `_detach_stale_family_events()`'s bare-container branch still protects after Task 1
    (Phase 35, D-16): a date-bearing leftover under a container-dispatched run is now
    DELETED instead (see `TestLegacyPerNightFamilyDeletion`, below), so it can no longer
    exercise a detach-then-reconfirm-then-reclaim scenario at all -- there is no companion
    row left once the event is gone."""

    def test_second_reconcile_deletes_the_superseded_allocation_night_and_restore_on_third(self):
        run = self._make_run(window_start=date(2026, 8, 1), window_end=date(2026, 8, 2))

        first = reconcile_run(run)
        self.assertEqual(first.created, 2)
        self.assertEqual(allocation_events(run).count(), 2)

        scheduled_start = datetime(2026, 8, 2, 3, 0, tzinfo=dt_timezone.utc)
        scheduled_end = scheduled_start + timedelta(hours=2)
        site_zone = ZoneInfo(self.ground_site.timezone)
        retired_night = observing_night(scheduled_start, site_zone)
        retired_url = f'ALLOC:{run.pk}:{retired_night.isoformat()}'
        _record, link = self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_end)

        second = reconcile_run(run)

        self.assertEqual(second.retired, 1)
        self.assertFalse(CalendarEvent.objects.filter(url=retired_url).exists())
        self.assertFalse(CalendarEventMeta.objects.filter(event__url=retired_url).exists())
        self.assertEqual(allocation_events(run).count(), 1)

        link.delete()
        # 35-04 D-11: deleting the link already re-projects the run via the new post_delete
        # receiver on CampaignRunObservation (wired in SolsysCodeConfig.ready()), so the
        # night is restored (minted fresh) before this test's own explicit `reconcile_run()`
        # below ever runs. That call therefore converges on already-current state and
        # reports `unchanged` for both nights, not `created` -- mirrors the same fix applied
        # to `test_allocation_projector.TestObservationHandoff`'s analogous test.
        third = reconcile_run(run)

        self.assertEqual(third.unchanged, 2)
        self.assertTrue(CalendarEvent.objects.filter(url=retired_url).exists())
        self.assertEqual(allocation_events(run).count(), 2)

    def test_blocked_night_keeps_its_url_active_and_is_never_detached(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        other_run = self._make_run(telescope_instrument='Other Telescope/Instrument')
        clashing_event = CalendarEvent.objects.create(
            title='Owned by a different run',
            url=f'ALLOC:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=clashing_event, run=other_run)

        result = reconcile_run(run)

        self.assertEqual(result.blocked, 1)
        self.assertEqual(result.detached, 0)
        clashing_event.refresh_from_db()
        self.assertEqual(CalendarEventMeta.objects.get(event=clashing_event).run_id, other_run.pk)

    def _make_reclassified_run_with_stale_container(self, **overrides) -> tuple[CampaignRun, CalendarEvent]:
        """A class-wide container-dispatched run, already reconciled once (creating its
        `RUN:{pk}` container, self-attributed), then re-classified to per-night allocation
        dispatch by clearing `telescope_class` -- leaving the bare `RUN:{pk}` container
        stale. This is the shape `_detach_stale_family_events()`'s bare-container branch
        still protects after Task 1 (Phase 35): that form keeps detaching, never deleting."""
        night = date(2026, 8, 1)
        kwargs = {
            'telescope_class': CampaignRun.TelescopeClass.ONE_M0,
            'window_start': night,
            'window_end': night,
        }
        kwargs.update(overrides)
        run = self._make_run(**kwargs)
        reconcile_run(run)
        container_event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        run.telescope_class = ''
        run.save(update_fields=['telescope_class'])
        return run, container_event

    def test_staff_reconfirmation_of_the_detached_legacy_night_survives_every_later_sweep(self):
        """CR-04 (33-REVIEW.md) / ANNOT-01, moved to a reclassified-container fixture (see
        class docstring): a staff re-confirmation of a detached bare-container event is
        never erased again by an automated sweep, and the sweep reports the declined count
        instead of silently repeating the erasure."""
        run, container_event = self._make_reclassified_run_with_stale_container()
        container_pk = container_event.pk

        second = reconcile_run(run)
        self.assertEqual(second.detached, 1)
        self.assertEqual(second.detach_declined, 0)
        self.assertEqual(second.legacy_deleted, 0)
        container_meta = CalendarEventMeta.objects.get(event=container_event)
        self.assertIsNone(container_meta.run_id)

        staffer = User.objects.create(username='attribution-staffer')
        confirmed_at = datetime(2026, 8, 2, 9, 0, tzinfo=dt_timezone.utc)
        container_meta.run = run
        container_meta.confirmed_by = staffer
        container_meta.confirmed_at = confirmed_at
        container_meta.save(update_fields=['run', 'confirmed_by', 'confirmed_at'])

        third = reconcile_run(run)

        self.assertEqual(third.detached, 0)
        self.assertEqual(third.detach_declined, 1)
        container_event.refresh_from_db()
        self.assertEqual(container_event.pk, container_pk)
        container_meta.refresh_from_db()
        self.assertEqual(container_meta.run_id, run.pk)
        self.assertEqual(container_meta.confirmed_by_id, staffer.pk)
        self.assertEqual(container_meta.confirmed_at, confirmed_at)

        fourth = reconcile_run(run)

        self.assertEqual(fourth.detached, 0)
        self.assertEqual(fourth.detach_declined, 1)
        container_meta.refresh_from_db()
        self.assertEqual(container_meta.run_id, run.pk)
        self.assertEqual(container_meta.confirmed_by_id, staffer.pk)
        self.assertEqual(container_meta.confirmed_at, confirmed_at)

        self.assertEqual(CalendarEventDismissal.objects.count(), 0)

    def test_unconfirmed_reattribution_of_the_detached_legacy_night_is_still_reclaimable(self):
        """The same scenario as above, but with `confirmed_by` left null on the re-attached
        row: an automated (not human-confirmed) re-link is still reclaimable by a later
        sweep -- only a HUMAN confirmation outranks the automated detach."""
        run, container_event = self._make_reclassified_run_with_stale_container()

        reconcile_run(run)  # detaches the stale container
        container_meta = CalendarEventMeta.objects.get(event=container_event)
        container_meta.run = run
        container_meta.save(update_fields=['run'])

        result = reconcile_run(run)

        self.assertEqual(result.detached, 1)
        self.assertEqual(result.detach_declined, 0)
        container_meta.refresh_from_db()
        self.assertIsNone(container_meta.run_id)

    def test_dry_run_previews_the_detach_count_and_writes_nothing(self):
        """WR-11: `--dry-run` previews the one irreversible step (the detach) instead of
        refusing to -- the previewed number comes from the same predicate the real sweep
        detaches on, and the dry run still writes nothing at all."""
        run, container_event = self._make_reclassified_run_with_stale_container()
        title_before = container_event.title

        preview = reconcile_run(run, dry_run=True)

        self.assertEqual(preview.detached, 1)
        self.assertEqual(preview.detach_declined, 0)
        container_event.refresh_from_db()
        self.assertEqual(container_event.title, title_before)
        self.assertEqual(CalendarEventMeta.objects.get(event=container_event).run_id, run.pk)


class TestLegacyPerNightFamilyDeletion(CampaignReconcilerTestBase):
    """Task 1 (D-16, Phase 35): the second half of the retired ``RUN:{pk}:{date}`` per-night
    family's cutover -- the half the allocation projector cannot reach, because a
    container-dispatched run never enters the projector at all. A leftover date-bearing
    event belonging to a run that now dispatches to the whole-window container is DELETED,
    one-time, never detached -- the bare ``RUN:{pk}`` container keeps the existing
    detach-never-delete rule (see the reclassified-container tests in
    ``TestReconcileThenAttributeOrdering``, above, and
    ``test_stale_container_event_is_not_adopted_into_an_allocation_night`` in
    ``TestReclassificationConvergence``, below)."""

    def _make_container_run_with_legacy_nights(
        self, count: int = 1, **overrides
    ) -> tuple[CampaignRun, list[CalendarEvent]]:
        """A container-dispatched run (LCO_QUEUE, resolved site -- D-10) already reconciled
        once (creating its `RUN:{pk}` container), plus `count` hand-made legacy
        `RUN:{pk}:{date}` events attributed to it -- the pre-cutover artifact shape this
        task's delete branch targets, since a container's own `active_urls` is always just
        `{run_container_url(run)}`."""
        night = date(2026, 8, 1)
        kwargs = {'source': CampaignRun.Source.LCO_QUEUE, 'window_start': night, 'window_end': night}
        kwargs.update(overrides)
        run = self._make_run(**kwargs)
        reconcile_run(run)
        events = []
        for i in range(count):
            legacy_night = night + timedelta(days=i)
            legacy_event = CalendarEvent.objects.create(
                title='Legacy per-night artifact',
                url=f'RUN:{run.pk}:{legacy_night.isoformat()}',
                start_time=datetime.combine(legacy_night, datetime.min.time(), tzinfo=dt_timezone.utc),
                end_time=datetime.combine(legacy_night, datetime.max.time(), tzinfo=dt_timezone.utc),
            )
            CalendarEventMeta.objects.create(event=legacy_event, run=run)
            events.append(legacy_event)
        return run, events

    def test_three_leftover_nights_are_deleted_converging_to_one_bare_container(self):
        """Test 1: three pre-existing `RUN:{pk}:{date}` events reconcile to exactly one bare
        `RUN:{pk}` container event; the three date-bearing events are gone from the
        database, and `legacy_deleted == 3`."""
        run, legacy_events = self._make_container_run_with_legacy_nights(count=3)
        legacy_pks = [event.pk for event in legacy_events]

        result = reconcile_run(run)

        self.assertEqual(result.legacy_deleted, 3)
        self.assertEqual(result.detached, 0)
        for pk in legacy_pks:
            self.assertFalse(CalendarEvent.objects.filter(pk=pk).exists())
            self.assertFalse(CalendarEventMeta.objects.filter(event_id=pk).exists())
        remaining = CalendarEvent.objects.filter(url__startswith=f'RUN:{run.pk}')
        self.assertEqual(remaining.count(), 1)
        self.assertEqual(remaining.get().url, f'RUN:{run.pk}')

    def test_foreign_attribution_is_neither_deleted_nor_detached(self):
        """Test 3: a date-bearing event whose `CalendarEventMeta` attributes it to a
        DIFFERENT run is left completely alone by this run's reconcile.

        NF-15 (35-REVIEW.md): before the fix, this shape (a `RUN:{pk}:{date}` event in
        THIS run's own namespace, attributed to a DIFFERENT run) matched neither half of
        `_clearable_declined_and_unattributed()`'s partition -- not deleted, not declined,
        and (unlike the mirror `ALLOC:` case `project_allocation()` already handles) not
        even counted as `blocked` -- a silent, permanently-orphaned third outcome, in a key
        family this phase retires entirely, with no log line at all. It must now be
        reported: folded into `blocked`, with its own warning log line."""
        run, (legacy_event,) = self._make_container_run_with_legacy_nights(count=1)
        other_run = self._make_run(telescope_instrument='Other Telescope/Instrument')
        legacy_meta = CalendarEventMeta.objects.get(event=legacy_event)
        legacy_meta.run = other_run
        legacy_meta.save(update_fields=['run'])

        with self.assertLogs('solsys_code.campaign_reconciler', level='WARNING') as log_ctx:
            result = reconcile_run(run)

        self.assertEqual(result.legacy_deleted, 0)
        self.assertEqual(result.detached, 0)
        self.assertEqual(result.blocked, 1)  # NF-15: reported, not silently dropped
        self.assertTrue(CalendarEvent.objects.filter(pk=legacy_event.pk).exists())
        legacy_meta.refresh_from_db()
        self.assertEqual(legacy_meta.run_id, other_run.pk)
        self.assertTrue(
            any('attributed to a different run' in message for message in log_ctx.output),
            f'expected a warning naming the foreign attribution, got: {log_ctx.output}',
        )

        # A second sweep reports the SAME blocked count, not a growing one -- the event is
        # left alone forever, never double-counted across repeated sweeps.
        second_result = reconcile_run(run)
        self.assertEqual(second_result.blocked, 1)

    def test_unattributed_leftover_night_shape_b_is_deleted(self):
        """35-REVIEW.md NF-01 item 3, CR-02's call site: a `RUN:{pk}:{date}` event whose
        companion row exists but whose `run` is unset (shape (b)) must be deleted the same
        way as the meta-less shape-(a) case below, and reported under the same
        `legacy_deleted` counter."""
        run, (legacy_event,) = self._make_container_run_with_legacy_nights(count=1)
        meta = CalendarEventMeta.objects.get(event=legacy_event)
        meta.run = None
        meta.save(update_fields=['run'])

        result = reconcile_run(run)

        self.assertEqual(result.legacy_deleted, 1)
        self.assertEqual(result.detach_declined, 0)
        self.assertEqual(result.blocked, 0)
        self.assertEqual(result.detached, 0)
        self.assertFalse(CalendarEvent.objects.filter(pk=legacy_event.pk).exists())

    def test_human_confirmed_leftover_night_is_not_deleted_and_counts_as_declined(self):
        """Test 4: a companion row carrying `confirmed_by` is not deleted -- a human
        confirmation still outranks the automated sweep; it is reported under the existing
        `detach_declined` counter instead."""
        night = date(2026, 8, 1)
        run = self._make_run(source=CampaignRun.Source.LCO_QUEUE, window_start=night, window_end=night)
        reconcile_run(run)
        legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        staffer = User.objects.create(username='legacy-delete-staffer')
        CalendarEventMeta.objects.create(
            event=legacy_event,
            run=run,
            confirmed_by=staffer,
            confirmed_at=datetime(2026, 8, 1, 9, 0, tzinfo=dt_timezone.utc),
        )

        result = reconcile_run(run)

        self.assertEqual(result.legacy_deleted, 0)
        self.assertEqual(result.detach_declined, 1)
        self.assertTrue(CalendarEvent.objects.filter(pk=legacy_event.pk).exists())
        legacy_meta = CalendarEventMeta.objects.get(event=legacy_event)
        self.assertEqual(legacy_meta.run_id, run.pk)
        self.assertEqual(legacy_meta.confirmed_by_id, staffer.pk)

    def test_orphan_legacy_event_with_no_companion_row_is_still_deleted(self):
        """35-REVIEW.md WR-10: a `RUN:{pk}:{date}` event with NO `CalendarEventMeta`
        companion row at all (a pre-Phase-29 event, or one the admin FK picker created) is
        outside `_clearable_and_declined()`'s own scope -- it starts from a
        `CalendarEventMeta` queryset, so a meta-less event is in neither the clearable list
        nor the declined count. D-16's contract ("either re-keyed or removed -- no third
        outcome") means it must still be deleted, since there is no attribution to
        preserve."""
        night = date(2026, 8, 1)
        run = self._make_run(source=CampaignRun.Source.LCO_QUEUE, window_start=night, window_end=night)
        reconcile_run(run)
        orphan_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact, no companion row',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        self.assertFalse(CalendarEventMeta.objects.filter(event=orphan_event).exists())

        result = reconcile_run(run)

        self.assertEqual(result.legacy_deleted, 1)
        self.assertEqual(result.detach_declined, 0)
        self.assertEqual(result.blocked, 0)
        self.assertEqual(result.detached, 0)
        self.assertFalse(CalendarEvent.objects.filter(pk=orphan_event.pk).exists())

    def test_orphan_legacy_event_with_unset_run_companion_row_is_still_deleted(self):
        """35-REVIEW.md NF-01 item 3: the shape-(b) twin of the test above -- a
        `RUN:{pk}:{date}` event whose `CalendarEventMeta` companion row exists but whose
        `run` is unset. Before the fix, this shape fell between
        `_clearable_and_declined()`'s own scope (shape-(c)-this-run only) and the WR-10
        no-companion-row union, so it was neither deleted nor counted -- D-16's forbidden
        third outcome."""
        night = date(2026, 8, 1)
        run = self._make_run(source=CampaignRun.Source.LCO_QUEUE, window_start=night, window_end=night)
        reconcile_run(run)
        orphan_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact, unset-run companion row',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=orphan_event, run=None)

        result = reconcile_run(run)

        self.assertEqual(result.legacy_deleted, 1)
        self.assertEqual(result.detach_declined, 0)
        self.assertEqual(result.blocked, 0)
        self.assertEqual(result.detached, 0)
        self.assertFalse(CalendarEvent.objects.filter(pk=orphan_event.pk).exists())

    def test_deletion_is_one_time_not_per_sweep_churn(self):
        """Test 5: a second reconcile of the same run reports `legacy_deleted == 0` -- the
        deletion is one-time, not per-sweep churn."""
        run, _events = self._make_container_run_with_legacy_nights(count=2)

        first = reconcile_run(run)
        self.assertEqual(first.legacy_deleted, 2)

        second = reconcile_run(run)
        self.assertEqual(second.legacy_deleted, 0)

    def test_dry_run_previews_legacy_deleted_and_writes_nothing(self):
        """Module-level twin of Test 6 (the command-level naming lives in
        test_reconcile_campaign_runs.py, per Task 1's own file split): a dry run over
        `reconcile_run()` itself previews the count and deletes nothing."""
        run, legacy_events = self._make_container_run_with_legacy_nights(count=2)
        legacy_pks = [event.pk for event in legacy_events]

        preview = reconcile_run(run, dry_run=True)

        self.assertEqual(preview.legacy_deleted, 2)
        for pk in legacy_pks:
            self.assertTrue(CalendarEvent.objects.filter(pk=pk).exists())

    def test_dry_run_never_double_counts_a_legacy_night_the_projector_would_take_over(self):
        """Regression, found by 35-06 Task 3's real-database proof run: an allocation
        (per-night) dispatched run's OWN leftover `RUN:{pk}:{night}` event, still within its
        active window, is taken over (rekeyed) by `project_allocation()`'s own legacy-night
        takeover -- it must be counted ONCE, under `rekeyed`, never a second time under
        `legacy_deleted` just because a dry run never actually writes the url change."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 9, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 19, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)

        preview = reconcile_run(run, dry_run=True)

        self.assertEqual(preview.rekeyed, 1)
        self.assertEqual(preview.legacy_deleted, 0)

        real = reconcile_run(run)
        self.assertEqual(real.rekeyed, 1)
        self.assertEqual(real.legacy_deleted, 0)

    def test_dry_run_never_double_counts_a_retiring_nights_legacy_twin(self):
        """The same regression, for the retire branch: a linked placed record retires a
        night that also carries a leftover `RUN:{pk}:{night}` twin -- the projector's own
        retire step already accounts for deleting that twin (real mode) or would (dry-run
        preview), so it must not also surface under `legacy_deleted`."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        legacy_event = CalendarEvent.objects.create(
            title='NTT EFOSC2',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 9, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 19, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=run)
        scheduled_start = datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc)
        self._link_record(run, scheduled_start=scheduled_start, scheduled_end=scheduled_start + timedelta(hours=2))

        preview = reconcile_run(run, dry_run=True)

        self.assertEqual(preview.retired, 1)
        self.assertEqual(preview.legacy_deleted, 0)


class TestAttributedEventsSurviveReconcile(CampaignReconcilerTestBase):
    """D-04 proof (ROADMAP criterion 2), migrated for Phase 35: an event attributed to a
    run via `CalendarEventMeta` but never linked through a `CampaignRunObservation` -- the
    old "attributed via a stray companion row" fixture shape -- is byte-identical across a
    `reconcile_run()` call, now against an allocation-dispatched run. Unlike the retired
    `_attributed_nights()` skip rule, this attribution has no effect on which nights the
    allocation projector mints -- it neither retires a night (only a `CampaignRunObservation`
    link with a placed/observed block does that, D-05) nor prevents one being created
    alongside it; it is simply outside the projector's own namespace and therefore
    untouched, the same non-interference guarantee `TestRecordEventNonInterference` proves."""

    def _snapshot(self, event: CalendarEvent, meta: CalendarEventMeta) -> tuple:
        return (
            event.url,
            event.title,
            event.description,
            event.start_time,
            event.end_time,
            event.telescope,
            event.instrument,
            meta.run_id,
            meta.is_verified,
            meta.confirmed_by_id,
            meta.confirmed_at,
        )

    def test_blank_url_attributed_event_survives_reconcile(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        event = CalendarEvent.objects.create(
            title='FTN MuSCAT3',
            url='',
            description='Ingested by load_telescope_runs',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 18, 0, tzinfo=dt_timezone.utc),
        )
        meta = CalendarEventMeta.objects.create(event=event, run=run, is_verified=False)
        before = self._snapshot(event, meta)

        result = reconcile_run(run)

        event.refresh_from_db()
        meta.refresh_from_db()
        self.assertEqual(self._snapshot(event, meta), before)
        self.assertEqual(result.created, 1)
        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{night.isoformat()}').exists())

    def test_facility_url_keyed_attributed_event_survives_reconcile(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        event = CalendarEvent.objects.create(
            title='LCO record event',
            url='https://observe.lco.global/api/requestgroups/999999/',
            description='Synced by the retired LCO/SOAR sync command',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 18, 0, tzinfo=dt_timezone.utc),
        )
        meta = CalendarEventMeta.objects.create(event=event, run=run, is_verified=True)
        before = self._snapshot(event, meta)

        result = reconcile_run(run)

        event.refresh_from_db()
        meta.refresh_from_db()
        self.assertEqual(self._snapshot(event, meta), before)
        self.assertEqual(result.created, 1)
        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{night.isoformat()}').exists())

    def test_second_reconcile_over_attributed_events_is_idempotent(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        event = CalendarEvent.objects.create(
            title='FTN MuSCAT3',
            url='https://observe.lco.global/api/requestgroups/888888/',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=datetime(2026, 8, 1, 10, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 18, 0, tzinfo=dt_timezone.utc),
        )
        meta = CalendarEventMeta.objects.create(event=event, run=run)
        before = self._snapshot(event, meta)

        first = reconcile_run(run)
        second = reconcile_run(run)

        event.refresh_from_db()
        meta.refresh_from_db()
        self.assertEqual(self._snapshot(event, meta), before)
        self.assertEqual(first.created, 1)
        self.assertEqual(second.created, 0)
        self.assertEqual(second.updated, 0)
        self.assertEqual(second.unchanged, 1)
        self.assertEqual(CalendarEvent.objects.count(), 2)


class TestClassicalStage1(CampaignReconcilerTestBase):
    """RECON-02's classical half -- Phase 35 replaced the writer this class exercised
    (`_reconcile_classical_nights()`, `RUN:{pk}:{date}`) with the peer allocation projector.

    Five of the original seven tests are RETIRED with a named destination
    (test_allocation_projector.TestEndToEndAllocationNight /
    TestAllocationEventAttribution -- confirmed present before retiring): one-event-per-night
    creation, the single-night-never-a-bare-key case, dip-corrected sun-event bounds, the
    companion-row-per-night proof, and the cancelled-prefix flip-back.

    The remaining two tests have NO counterpart in test_allocation_projector.py (confirmed by
    reading that module in full) and are KEPT here, migrated to the `ALLOC:` key form per
    Task 1's own instruction not to silently drop uncovered behaviour."""

    def test_key_date_equals_site_local_night_of_its_own_start_time(self):
        """26-DECISION.md's 'site-local observing night, never the naive UTC date' rule,
        proved rather than assumed: converting each event's own start_time into the site's
        timezone and taking .date() must return the date embedded in its url. Migrated to
        `allocation_events()`/`ALLOC:` -- no direct counterpart in
        test_allocation_projector.py, which asserts the url's date matches `run.window_start
        + i` by construction but never round-trips back through the computed start_time."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 3)
        run = self._make_run(window_start=window_start, window_end=window_end)
        site_zone = ZoneInfo(self.ground_site.timezone)

        reconcile_run(run)

        for event in allocation_events(run):
            key_date = date.fromisoformat(event.url.rsplit(':', 1)[-1])
            self.assertEqual(event.start_time.astimezone(site_zone).date(), key_date)

    def test_mid_loop_sun_event_valueerror_propagates_and_leaves_earlier_nights_in_place(self):
        """D-06's accepted partial projection: a mid-window sun_event() ValueError is not
        caught -- it propagates uncaught out of reconcile_run(), and the earlier nights'
        already-written events are left in place (no transaction.atomic() wrap). No
        counterpart exists in test_allocation_projector.py -- kept here, patching
        `solsys_code.allocation_projector.sun_event` (the module that now calls it)."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 3)
        run = self._make_run(window_start=window_start, window_end=window_end)
        real_sun_event = sun_event

        def _side_effect(site, night, kind='sun'):
            if night == date(2026, 8, 2):
                raise ValueError('no crossings')
            return real_sun_event(site, night, kind=kind)

        with patch('solsys_code.allocation_projector.sun_event', side_effect=_side_effect):
            with self.assertRaises(ValueError):
                reconcile_run(run)

        self.assertTrue(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:2026-08-01').exists())
        self.assertFalse(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:2026-08-02').exists())
        self.assertFalse(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:2026-08-03').exists())


class TestRecordEventNonInterference(CampaignReconcilerTestBase):
    """RECON-04/RECON-05, migrated onto an allocation-dispatched run per Task 1's
    instruction ("point it at an allocation-dispatched run, since that is now the writer
    being constrained"): the reconciler never creates, modifies or deletes an
    ObservationRecord-derived event.

    Unlike the retired per-night RUN: branch, a CampaignRunObservation-linked record with a
    placed block now also retires the run's OWN allocation night it falls in (D-05) -- that
    is the intended new behaviour Phase 35 built, not a regression of this non-interference
    guarantee. This test keeps the record's own event un-linked from any CampaignRunObservation
    so it exercises pure coexistence (a record-derived event the reconciler has no
    relationship to at all), which remains the exact non-interference case RECON-04 names.
    `TestContainerRecordEventNonInterference` below covers the container branch as its twin
    (unaffected by D-09/D-10, kept unchanged)."""

    def test_reconciler_never_touches_the_record_derived_event(self):
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(
            window_start=window_start,
            window_end=window_end,
        )
        # NonSiderealTargetFactory (never SiderealTargetFactory) -- FOMO is exclusively for
        # Solar System targets (CLAUDE.md).
        target = NonSiderealTargetFactory.create()
        record_owner = User.objects.create(username='record-owner')
        scheduled_start = datetime(2026, 8, 2, 3, 0, tzinfo=dt_timezone.utc)
        scheduled_end = datetime(2026, 8, 2, 5, 0, tzinfo=dt_timezone.utc)
        record = ObservationRecord.objects.create(
            target=target,
            user=record_owner,
            facility='LCO',
            observation_id='555555',
            status='COMPLETED',
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters={'proposal': 'TEST'},
        )
        expected_start, expected_end = record_time_window(record)
        # Keyed the way the observation projector keys a record-derived event: an LCO
        # portal request url, NOT an ALLOC:-namespaced one. Deliberately NOT linked via a
        # CampaignRunObservation -- this is the pure coexistence case, not the handoff.
        record_event = CalendarEvent.objects.create(
            title='LCO record event',
            url='https://observe.lco.global/api/requestgroups/555555/',
            telescope='FTN',
            instrument='MuSCAT3',
            start_time=expected_start,
            end_time=expected_end,
        )
        modified_before = record_event.modified

        reconcile_run(run)

        record_event.refresh_from_db()
        self.assertEqual(record_event.url, 'https://observe.lco.global/api/requestgroups/555555/')
        self.assertEqual(record_event.title, 'LCO record event')
        self.assertEqual(record_event.start_time, expected_start)
        self.assertEqual(record_event.end_time, expected_end)
        self.assertEqual(record_event.modified, modified_before)

        # The run's own per-night allocation events coexist alongside it -- one per night in
        # the window, no bare RUN:{pk} container at all (D-09: a resolved-site, non-queue
        # run is allocation-dispatched).
        n_nights = (window_end - window_start).days + 1
        self.assertFalse(CalendarEvent.objects.filter(url=f'RUN:{run.pk}').exists())
        self.assertEqual(CalendarEvent.objects.count(), 1 + n_nights)

        # allocation_events(run) returns exactly the n date-bearing per-night rows.
        self.assertEqual(allocation_events(run).count(), n_nights)
        for i in range(n_nights):
            night = window_start + timedelta(days=i)
            self.assertTrue(allocation_events(run).filter(url=f'ALLOC:{run.pk}:{night.isoformat()}').exists())

        # The record-derived event's window still equals record_time_window(record) --
        # RECON-04's stage-3/stage-4 behaviour, expressed as non-interference.
        self.assertEqual(record_time_window(record), (record_event.start_time, record_event.end_time))

        # A second reconcile pass still leaves the record-derived event's modified alone.
        reconcile_run(run)
        record_event.refresh_from_db()
        self.assertEqual(record_event.modified, modified_before)


class TestContainerRecordEventNonInterference(CampaignReconcilerTestBase):
    """The container-branch twin of `TestRecordEventNonInterference` above: a class-wide
    run's own whole-window container write never creates, modifies or deletes an
    ObservationRecord-derived event either. Kept unchanged -- unaffected by D-09/D-10. Both
    classes together are the evidence 29-SECURITY.md's T-29-07 cites: the protection is
    `_may_write()`'s ownership check, the first condition checked in both
    `_reconcile_container()` and `allocation_projector.project_allocation()`, so it applies
    identically regardless of which branch a given run takes."""

    def test_reconciler_never_touches_the_record_derived_event(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 10),
        )
        # NonSiderealTargetFactory (never SiderealTargetFactory) -- FOMO is exclusively for
        # Solar System targets (CLAUDE.md).
        target = NonSiderealTargetFactory.create()
        record_owner = User.objects.create(username='container-record-owner')
        scheduled_start = datetime(2026, 8, 2, 3, 0, tzinfo=dt_timezone.utc)
        scheduled_end = datetime(2026, 8, 2, 5, 0, tzinfo=dt_timezone.utc)
        record = ObservationRecord.objects.create(
            target=target,
            user=record_owner,
            facility='LCO',
            observation_id='666666',
            status='COMPLETED',
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters={'proposal': 'TEST'},
        )
        expected_start, expected_end = record_time_window(record)
        # Keyed the way the observation projector keys a record-derived event: an LCO
        # portal request url, NOT a RUN:-namespaced one.
        record_event = CalendarEvent.objects.create(
            title='LCO record event (container branch)',
            url='https://observe.lco.global/api/requestgroups/666666/',
            telescope='LCO 1m0-SciCam-Sinistro',
            instrument='Sinistro',
            start_time=expected_start,
            end_time=expected_end,
        )
        CampaignRunObservation.objects.create(run=run, observation_record=record)
        modified_before = record_event.modified

        reconcile_run(run)

        record_event.refresh_from_db()
        self.assertEqual(record_event.url, 'https://observe.lco.global/api/requestgroups/666666/')
        self.assertEqual(record_event.title, 'LCO record event (container branch)')
        self.assertEqual(record_event.start_time, expected_start)
        self.assertEqual(record_event.end_time, expected_end)
        self.assertEqual(record_event.modified, modified_before)

        # The run's own whole-window container event coexists beside it.
        self.assertTrue(CalendarEvent.objects.filter(url=f'RUN:{run.pk}').exists())
        self.assertEqual(CalendarEvent.objects.count(), 2)

        self.assertEqual(record_time_window(record), (record_event.start_time, record_event.end_time))

        # A second reconcile pass still leaves the record-derived event's modified alone.
        reconcile_run(run)
        record_event.refresh_from_db()
        self.assertEqual(record_event.modified, modified_before)


class TestReclassificationConvergence(CampaignReconcilerTestBase):
    """CR-01 (29-REVIEW.md), migrated for Phase 35: reclassifying a run's family detaches
    -- never deletes, never leaves dangling -- the old family's `RUN:`-namespaced events,
    and a stale event from one family is never corrupted by being adopted into the other.

    `test_pre_fix_container_event_converges_to_per_night_on_next_reconcile` is RETIRED: it
    reproduced quick task 260805-tad's pre-fix bug, where a queue-sourced run with a
    resolved site wrongly stayed on the container branch. D-10 makes that the CORRECT,
    permanent behaviour for every queue source -- a queue-sourced, resolved-site run can no
    longer ever converge to the per-night/allocation branch, so the scenario this test
    reproduced can no longer occur. No destination module is needed: the scenario is dead,
    not the coverage.

    A new case is added proving D-14: a leftover `ALLOC:{pk}:*` event for a night no longer
    in the run's window is DELETED (not detached) by the allocation projector's own internal
    convergence -- unlike the `RUN:` family, which the shared detach step still protects."""

    def test_reclassifying_allocation_dispatch_to_class_wide_deletes_old_per_night_events(self):
        """A run reconciled once under the allocation (per-night) branch, then reclassified
        to the class-wide container branch (setting `telescope_class` on an
        already-resolved-site run), leaves the container's own reconcile to delete the old
        `ALLOC:`-keyed per-night events (35-REVIEW.md CR-02): once dispatch moves to the
        container branch, `project_allocation()` is never called again for this run, so its
        own D-14 convergence can never reach them again -- `_stale_allocation_events()`
        closes that gap from the `RUN:`-namespace convergence step instead, mirroring the
        date-bearing `RUN:{pk}:{date}` family's own one-time-churn delete exactly."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(window_start=window_start, window_end=window_end)

        first = reconcile_run(run)
        self.assertEqual(first.created, 2)
        alloc_urls = [f'ALLOC:{run.pk}:{window_start.isoformat()}', f'ALLOC:{run.pk}:{window_end.isoformat()}']
        for url in alloc_urls:
            self.assertTrue(CalendarEvent.objects.filter(url=url).exists())
            self.assertEqual(CalendarEventMeta.objects.get(event__url=url).run_id, run.pk)

        run.telescope_class = CampaignRun.TelescopeClass.ONE_M0
        run.save(update_fields=['telescope_class'])
        second = reconcile_run(run)

        self.assertEqual(second.created, 1)
        container_event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        self.assertEqual(CalendarEventMeta.objects.get(event=container_event).run_id, run.pk)

        # The old allocation nights are gone -- one-time churn, deleted (not detached) the
        # same way a re-classified run's leftover RUN:{pk}:{date} nights are.
        for url in alloc_urls:
            self.assertFalse(CalendarEvent.objects.filter(url=url).exists())
        self.assertEqual(second.legacy_deleted, 2)
        self.assertEqual(second.detached, 0)

    def test_reclassifying_allocation_dispatch_to_class_wide_deletes_unattributed_nights_shape_b(self):
        """35-REVIEW.md NF-01 item 1, CR-02's call site: the shape-(b) variant of the test
        above -- the old `ALLOC:` nights' companion rows have their `run` cleared (present
        but unset) BEFORE the reclassify-and-reconcile, rather than staying attributed to
        this run. Before the fix, `_stale_allocation_events()` routed through
        `_clearable_and_declined()` alone, which starts from
        `CalendarEventMeta.objects.filter(run_id=run.pk, ...)` and therefore never saw these
        rows at all -- unreachable forever."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(window_start=window_start, window_end=window_end)
        reconcile_run(run)
        alloc_urls = [f'ALLOC:{run.pk}:{window_start.isoformat()}', f'ALLOC:{run.pk}:{window_end.isoformat()}']
        for url in alloc_urls:
            meta = CalendarEventMeta.objects.get(event__url=url)
            meta.run = None
            meta.save(update_fields=['run'])

        run.telescope_class = CampaignRun.TelescopeClass.ONE_M0
        run.save(update_fields=['telescope_class'])
        result = reconcile_run(run)

        for url in alloc_urls:
            self.assertFalse(CalendarEvent.objects.filter(url=url).exists())
        self.assertEqual(result.legacy_deleted, len(alloc_urls))
        self.assertEqual(result.detach_declined, 0)
        self.assertEqual(result.blocked, 0)

    def test_second_reconcile_after_deleting_old_allocation_nights_reports_nothing_further(self):
        """RECON-01 idempotency: once the old `ALLOC:` family has been deleted by the first
        post-reclassification reconcile, a second reconcile of the same (still
        container-dispatched) run finds nothing left to delete."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(window_start=window_start, window_end=window_end)
        reconcile_run(run)
        run.telescope_class = CampaignRun.TelescopeClass.ONE_M0
        run.save(update_fields=['telescope_class'])
        reconcile_run(run)

        third = reconcile_run(run)

        self.assertEqual(third.legacy_deleted, 0)
        self.assertEqual(third.detached, 0)

    def test_dry_run_previews_the_stale_allocation_delete_and_writes_nothing(self):
        """The dry-run branch of `reconcile_run()` must agree with the real sweep: it
        previews the same `legacy_deleted` count without touching the database."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(window_start=window_start, window_end=window_end)
        reconcile_run(run)
        alloc_urls = [f'ALLOC:{run.pk}:{window_start.isoformat()}', f'ALLOC:{run.pk}:{window_end.isoformat()}']
        run.telescope_class = CampaignRun.TelescopeClass.ONE_M0
        run.save(update_fields=['telescope_class'])

        preview = reconcile_run(run, dry_run=True)

        self.assertEqual(preview.legacy_deleted, 2)
        for url in alloc_urls:
            self.assertTrue(CalendarEvent.objects.filter(url=url).exists())

    def test_stale_container_event_is_not_adopted_into_an_allocation_night(self):
        """Proves the legacy-takeover filter in `project_allocation()`'s per-night loop
        only ever matches a date-bearing `RUN:{pk}:{date}` url, never a bare `RUN:{pk}`
        container: a run's own stale container event (left over from a prior
        container-family reconcile) must never be re-keyed into a per-night slot -- which
        would leave it looking like one observing night while still timed as the entire
        original whole-window span -- even though its `CalendarEventMeta.run` already points
        at this run. The container is created via a non-blank `telescope_class`, then the
        run is reclassified into the allocation family by clearing it."""
        night = date(2026, 8, 1)
        run = self._make_run(
            window_start=night,
            window_end=night,
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )
        reconcile_run(run)
        container_event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        container_start, container_end = container_event.start_time, container_event.end_time

        run.telescope_class = ''
        run.save(update_fields=['telescope_class'])
        result = reconcile_run(run)

        # A brand-new allocation night was minted -- the stale container was NOT adopted.
        self.assertEqual(result.created, 1)
        night_event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        self.assertNotEqual(night_event.pk, container_event.pk)

        # The stale container event survives untouched: not re-keyed, not re-timed.
        container_event.refresh_from_db()
        self.assertEqual(container_event.url, f'RUN:{run.pk}')
        self.assertEqual(container_event.start_time, container_start)
        self.assertEqual(container_event.end_time, container_end)
        # ... and is now detached rather than left attributed to this run.
        self.assertIsNone(CalendarEventMeta.objects.get(event=container_event).run_id)
        # Task 1 (Phase 35) Test 2: the bare RUN:{pk} container form is still DETACHED,
        # never deleted -- the convergence rule for that key form is untouched.
        self.assertEqual(result.detached, 1)
        self.assertEqual(result.legacy_deleted, 0)

    def test_leftover_allocation_night_for_a_shrunk_window_is_deleted_not_detached(self):
        """New D-14 case (Task 1's own instruction): a window shrink drops a night from the
        active set. For an allocation-dispatched run this is handled entirely INSIDE
        `project_allocation()`'s own convergence -- the leftover `ALLOC:` night is DELETED,
        never detached, and `_detach_stale_family_events()`'s `RUN:`-namespace convergence
        (which runs unconditionally after every dispatch) has nothing to do here since no
        `RUN:`-keyed event exists for this run at all."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run = self._make_run(window_start=window_start, window_end=window_end)
        reconcile_run(run)
        self.assertEqual(allocation_events(run).count(), 2)

        run.window_end = window_start
        run.save(update_fields=['window_end'])
        result = reconcile_run(run)

        self.assertEqual(allocation_events(run).count(), 1)
        self.assertFalse(CalendarEvent.objects.filter(url=f'ALLOC:{run.pk}:{window_end.isoformat()}').exists())
        self.assertEqual(result.retired, 1)
        self.assertEqual(result.detached, 0)

    def test_detach_never_clears_a_foreign_attribution_in_the_same_namespace(self):
        """T-33-14: a stale-family `RUN:`-namespaced event already re-attributed to a
        DIFFERENT run keeps that attribution through a reconcile of the run whose namespace
        the url still carries -- the run=run filter term (T-29-19) the shared helper
        preserves. Uses a container-dispatched run with a hand-made legacy per-night event,
        since that is the shape `_detach_stale_family_events()` still protects."""
        night = date(2026, 8, 1)
        run = self._make_run(
            source=CampaignRun.Source.LCO_QUEUE,
            window_start=night,
            window_end=night,
        )
        other_run = self._make_run(window_start=date(2026, 9, 1), window_end=date(2026, 9, 1))
        reconcile_run(run)
        legacy_event = CalendarEvent.objects.create(
            title='Legacy per-night artifact, re-attributed to a different run',
            url=f'RUN:{run.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=legacy_event, run=other_run)

        result = reconcile_run(run)

        # Task 1 (Phase 35) Test 3: a date-bearing event attributed to a DIFFERENT run is
        # neither deleted nor detached -- the event and its foreign attribution survive.
        self.assertTrue(CalendarEvent.objects.filter(pk=legacy_event.pk).exists())
        self.assertEqual(result.legacy_deleted, 0)
        legacy_meta = CalendarEventMeta.objects.get(event=legacy_event)
        self.assertEqual(legacy_meta.run_id, other_run.pk)


class TestCampaignRunDeletionCascadesCalendarEvents(CampaignReconcilerTestBase):
    """WR-01 (29-REVIEW.md): deleting a CampaignRun must not permanently orphan the
    calendar events it owns -- `CalendarEventMeta.run`'s `on_delete=SET_NULL` alone leaves
    the `CalendarEvent` rows themselves on the shared calendar forever. Kept unchanged --
    this class only exercises the `RUN:{pk}` container, unaffected by D-09/D-10; the
    equivalent `ALLOC:` namespace cascade is covered by
    test_allocation_projector.TestAllocationDeletionCascade (35-01)."""

    def test_deleting_a_run_deletes_its_owned_calendar_events(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )
        reconcile_run(run)
        event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        event_pk = event.pk
        self.assertTrue(CalendarEventMeta.objects.filter(event_id=event_pk).exists())

        run.delete()

        self.assertFalse(CalendarEvent.objects.filter(pk=event_pk).exists())
        self.assertFalse(CalendarEventMeta.objects.filter(event_id=event_pk).exists())


class TestCrossRunOwnershipGuards(CampaignReconcilerTestBase):
    """Security finding T-29-19: two write paths -- `_delete_owned_calendar_events_on_campaign_run_delete`
    and `_detach_stale_family_events()` -- select calendar events by URL-namespace identity
    alone (`owned_events()`), never checking whether `CalendarEventMeta.run` still points at
    the run doing the writing. As a result `reconcile_run(run_a)` could silently clear a
    staff-confirmed Phase 28 attribution that belongs to run B, and deleting run A could
    hard-delete calendar events that currently belong to run B.

    The two "per-night" cases below are migrated to the `ALLOC:` namespace, since the fixture
    must collide with the writer it is meant to test: a `RUN:{pk}:{date}`-style fixture no
    longer collides with anything, because the live per-night writer is now
    `allocation_projector.project_allocation()`. The remaining three cases are kept unchanged
    -- they exercise `RUN:`-namespace-specific guards (the `writable_events()` cascade and
    the `_detach_stale_family_events()` convergence), unaffected by which branch a run
    dispatches to."""

    def test_deleting_a_run_never_deletes_an_event_attributed_to_a_different_run(self):
        """A `CalendarEvent` whose `url` sits in run A's `RUN:` namespace, but whose
        companion row has since been re-attributed to run B (a staff member confirming a
        stale event via Phase 28's queue while its url string still carries A's namespace),
        must survive `run_a.delete()` -- both the row and its B attribution. Kept unchanged
        -- exercises `writable_events()`'s `RUN:`-namespace-only cascade guard directly."""
        run_a = self._make_run()
        run_b = self._make_run(telescope_instrument='Other Telescope/Instrument')
        event = CalendarEvent.objects.create(
            title='Re-attributed to run B',
            url=f'RUN:{run_a.pk}:2026-08-01',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=event, run=run_b)
        event_pk = event.pk

        run_a.delete()

        self.assertTrue(CalendarEvent.objects.filter(pk=event_pk).exists())
        self.assertEqual(CalendarEventMeta.objects.get(event_id=event_pk).run_id, run_b.pk)

    def test_reconcile_never_detaches_an_event_attributed_to_a_different_run(self):
        """A stale-family `RUN:`-namespaced event left over in run A's namespace, but whose
        companion row has since been re-attributed to run B, must not have that attribution
        cleared by `reconcile_run(run_a)`'s `_detach_stale_family_events()` convergence step.
        Kept unchanged -- `_detach_stale_family_events()` runs unconditionally after every
        dispatch branch (including the allocation branch run_a now takes), and the `run=run_a`
        filter term this proves is untouched by Phase 35."""
        window_start = date(2026, 8, 1)
        window_end = date(2026, 8, 2)
        run_a = self._make_run(window_start=window_start, window_end=window_end)
        run_b = self._make_run(telescope_instrument='Other Telescope/Instrument')
        reconcile_run(run_a)

        stale_event = CalendarEvent.objects.create(
            title='Stale run-A-namespaced event, re-attributed to run B',
            url=f'RUN:{run_a.pk}:2026-09-15',
            start_time=datetime(2026, 9, 15, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 15, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=stale_event, run=run_b)

        reconcile_run(run_a)

        self.assertTrue(CalendarEvent.objects.filter(pk=stale_event.pk).exists())
        self.assertEqual(CalendarEventMeta.objects.get(event=stale_event).run_id, run_b.pk)

    def test_deleting_a_run_still_deletes_the_events_it_genuinely_owns(self):
        """Don't-regress-the-fix probe: the two cases
        `test_deleting_a_run_deletes_its_owned_calendar_events` does not cover -- a
        previously-detached (companion row with `run` unset) stale-family event, and a
        namespaced event with no companion row at all -- must still be deleted along with
        the run, so the fix does not re-introduce WR-01's permanently-orphaned events. Kept
        unchanged."""
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )
        reconcile_run(run)
        container_event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')

        detached_event = CalendarEvent.objects.create(
            title='Previously-detached stale-family event',
            url=f'RUN:{run.pk}:2026-08-03',
            start_time=datetime(2026, 8, 3, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 3, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=detached_event, run=None)

        no_meta_event = CalendarEvent.objects.create(
            title='Namespaced event with no companion row at all',
            url=f'RUN:{run.pk}:2026-08-04',
            start_time=datetime(2026, 8, 4, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 4, 23, 59, tzinfo=dt_timezone.utc),
        )

        pks = [container_event.pk, detached_event.pk, no_meta_event.pk]

        run.delete()

        for pk in pks:
            self.assertFalse(CalendarEvent.objects.filter(pk=pk).exists())
            self.assertFalse(CalendarEventMeta.objects.filter(event_id=pk).exists())

    def test_reconcile_reports_blocked_for_a_night_attributed_to_a_different_run(self):
        """D-02: an ALLOC:{pk}:{date} event whose companion row points at a DIFFERENT run is
        reported as blocked by reconcile_run(run_a) itself, and meta.run_id is never reset
        to run_a. Migrated to the `ALLOC:` namespace -- the fixture must collide with the
        live per-night writer to exercise the guard at all."""
        night = date(2026, 8, 1)
        run_a = self._make_run(window_start=night, window_end=night)
        run_b = self._make_run(telescope_instrument='Other Telescope/Instrument')
        event = CalendarEvent.objects.create(
            title='Foreign attribution',
            url=f'ALLOC:{run_a.pk}:{night.isoformat()}',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 1, 23, 59, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=event, run=run_b)

        result = reconcile_run(run_a)

        self.assertGreaterEqual(result.blocked, 1)
        event.refresh_from_db()
        self.assertEqual(CalendarEventMeta.objects.get(event=event).run_id, run_b.pk)

    def test_run_owned_night_event_is_refreshed_in_place_url_unchanged(self):
        """D-03: an existing ALLOC:{pk}:{date} event attributed to this same run is still
        refreshed in place (title/description) -- this phase un-keys nothing. Migrated to
        the `ALLOC:` namespace -- the live per-night writer's own key form."""
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)
        reconcile_run(run)
        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        event_pk = event.pk

        run.observation_details = 'Updated observation details'
        run.save(update_fields=['observation_details'])
        result = reconcile_run(run)

        event.refresh_from_db()
        self.assertEqual(event.pk, event_pk)
        self.assertEqual(event.url, f'ALLOC:{run.pk}:{night.isoformat()}')
        self.assertIn('Updated observation details', event.description)
        self.assertEqual(result.updated, 1)


class TestWindowEndBeforeWindowStart(CampaignReconcilerTestBase):
    """WR-02 (29-REVIEW.md): a run whose `window_end` precedes its `window_start` must be
    reported as skipped with an explicit reason, not silently contribute zero events with
    no reported reason (indistinguishable from an already-`unchanged` run). Kept unchanged
    -- the stage-0 guard fires before any dispatch branch is reached."""

    def test_window_end_before_window_start_is_skipped_with_explicit_reason(self):
        run = self._make_run(window_start=date(2026, 8, 5), window_end=date(2026, 8, 1))

        result = reconcile_run(run)

        self.assertEqual(result.skipped_reason, 'window_end before window_start')
        self.assertEqual(CalendarEvent.objects.count(), 0)


class TestTelescopeInstrumentSplitOnEvents(CampaignReconcilerTestBase):
    """Proves the split lands correctly on a real CalendarEvent through both write branches,
    plus the no-delimiter fallback and a title guard against a future regression. The
    classical-create half now runs through the allocation projector (`ALLOC:` namespace);
    the container-branch tests are unaffected and kept unchanged."""

    def test_container_branch_splits_the_base_fixtures_slash_delimited_value(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        self.assertEqual(event.telescope, 'FTN')
        self.assertEqual(event.instrument, 'MuSCAT3')

    def test_classical_create_path_splits_the_base_fixtures_slash_delimited_value(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night)

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        self.assertEqual(event.telescope, 'FTN')
        self.assertEqual(event.instrument, 'MuSCAT3')

    def test_plus_delimiter_splits_the_same_way(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
            telescope_instrument='Apache Point Observatory+ARCTIC',
        )

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        self.assertEqual(event.telescope, 'Apache Point Observatory')
        self.assertEqual(event.instrument, 'ARCTIC')

    def test_no_delimiter_fallback_is_preserved_on_the_classical_branch(self):
        night = date(2026, 8, 1)
        run = self._make_run(window_start=night, window_end=night, telescope_instrument='NTT EFOSC2')

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'ALLOC:{run.pk}:{night.isoformat()}')
        self.assertEqual(event.telescope, 'NTT EFOSC2')
        self.assertEqual(event.instrument, '')

    def test_title_still_carries_the_full_combined_string(self):
        run = self._make_run(
            site=None,
            site_raw='',
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )

        reconcile_run(run)

        event = CalendarEvent.objects.get(url=f'RUN:{run.pk}')
        self.assertIn('FTN/MuSCAT3', event.title)
        self.assertEqual(event.title, event_title(run))


class TestSplitTelescopeInstrumentHelper(TestCase):
    """Pure-function tests for split_telescope_instrument() -- no DB fixture needed. Kept
    unchanged; the import already points at the public split helper after 35-01."""

    def test_slash_separated_value_splits_and_strips_both_halves(self):
        self.assertEqual(_split_telescope_instrument(' FTN / MuSCAT3 '), ('FTN', 'MuSCAT3'))

    def test_plus_separated_value_splits_the_same_way(self):
        self.assertEqual(
            _split_telescope_instrument('Apache Point Observatory+ARCTIC'),
            ('Apache Point Observatory', 'ARCTIC'),
        )

    def test_no_delimiter_falls_back_to_the_whole_string_with_blank_instrument(self):
        self.assertEqual(_split_telescope_instrument('NTT EFOSC2'), ('NTT EFOSC2', ''))
        self.assertEqual(_split_telescope_instrument('SomeScope'), ('SomeScope', ''))

    def test_only_the_first_delimiter_splits(self):
        self.assertEqual(_split_telescope_instrument('A/B/C'), ('A', 'B/C'))
        self.assertEqual(_split_telescope_instrument('A/B+C'), ('A', 'B+C'))
