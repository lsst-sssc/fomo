"""Matcher unit tests for Phase 28's operator-assisted attribution (28-VALIDATION.md Wave 0).

Class names match 28-VALIDATION.md's requirement-to-test map exactly so it resolves without
editing. TestCriterion5RealCase is the acceptance test (ATTRIB-05): the phase has not shipped
if it does not pass.
"""

import difflib
from datetime import date, datetime
from datetime import timezone as dt_timezone

from django.contrib.auth.models import User
from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.campaign_attribution import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    HIGH_BAND_MIN,
    MEDIUM_BAND_MIN,
    TELESCOPE_MATCH_APERTURE_ONLY,
    TELESCOPE_MATCH_INDETERMINATE,
    TELESCOPE_MATCH_NONE,
    TELESCOPE_MATCH_SITE,
    WEIGHT_DATE_OVERLAP,
    WEIGHT_INSTRUMENT_SIMILARITY,
    WEIGHT_TELESCOPE_MATCH,
    band_for_score,
    candidates_for_event,
    candidates_for_record,
    date_overlap_score,
    event_attribution_backlog,
    instrument_similarity,
    orphan_calendar_events,
    orphan_observation_records,
    record_attribution_backlog,
    telescope_match_score,
    unattributable_orphan_count,
)
from solsys_code.models import (
    CalendarEventDismissal,
    CalendarEventMeta,
    CampaignRun,
    CampaignRunObservation,
    ObservationRecordDismissal,
)
from solsys_code.solsys_code_observatory.models import Observatory


class TestScoringAndBanding(TestCase):
    """ATTRIB-02: the weighted-sum scoring formula, band cut-points, and each of the three
    signals' behaviour in isolation, including every outcome of telescope_match_score()."""

    @classmethod
    def setUpTestData(cls):
        cls.campaign = TargetList.objects.create(name='Scoring Campaign')
        cls.observatory = Observatory.objects.create(obscode='E10', name='Scoring Site', short_name='SS')
        cls.other_observatory = Observatory.objects.create(obscode='K91', name='Scoring Other Site', short_name='OS')

    def test_instrument_similarity_real_measured_strings(self):
        """RESEARCH.md Pitfall 1's pinned fact, made executable: the naive whole-string ratio
        for the real 'FTS/MuSCAT4' vs '2M0-SCICAM-MUSCAT' pair is BELOW this codebase's own
        0.6 fuzzy cutoff, but the tokenised similarity this matcher actually computes clears
        0.85 -- this is what makes "instrument similarity must never disqualify" a pinned
        fact rather than only a comment."""
        left, right = 'FTS/MuSCAT4', '2M0-SCICAM-MUSCAT'

        whole_string_ratio = difflib.SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
        self.assertLess(whole_string_ratio, 0.6)

        self.assertGreater(instrument_similarity(left, right), 0.85)

    def test_instrument_similarity_blank_input_returns_zero_never_raises(self):
        self.assertEqual(instrument_similarity('', 'anything'), 0.0)
        self.assertEqual(instrument_similarity('anything', ''), 0.0)
        self.assertEqual(instrument_similarity('', ''), 0.0)

    def test_date_overlap_fully_inside(self):
        score = date_overlap_score(date(2026, 7, 10), date(2026, 7, 10), date(2026, 7, 7), date(2026, 7, 21))
        self.assertEqual(score, 1.0)

    def test_date_overlap_partial(self):
        score = date_overlap_score(date(2026, 7, 5), date(2026, 7, 9), date(2026, 7, 7), date(2026, 7, 21))
        # Orphan spans 5 inclusive days (Jul 5-9); 3 of those (Jul 7-9) fall inside the run.
        self.assertAlmostEqual(score, 3 / 5)

    def test_date_overlap_disjoint(self):
        score = date_overlap_score(date(2026, 1, 1), date(2026, 1, 2), date(2026, 7, 7), date(2026, 7, 21))
        self.assertEqual(score, 0.0)

    def test_date_overlap_unresolved_run_window_scores_zero_not_disqualifying(self):
        """D-11: a TBD run window is an ABSENCE of evidence, scored zero, never a
        disqualification -- the pair can still be offered on its other two signals."""
        score = date_overlap_score(date(2026, 7, 10), date(2026, 7, 10), None, None)
        self.assertEqual(score, 0.0)

    def test_date_overlap_orphan_with_no_derivable_window_scores_zero(self):
        score = date_overlap_score(None, None, date(2026, 7, 7), date(2026, 7, 21))
        self.assertEqual(score, 0.0)

    def test_telescope_match_site_tier_match(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='Scoring run A',
            window_start=None,
            window_end=None,
            site=self.observatory,
        )
        score, evidence = telescope_match_score(run, 'COJ-2m0', '2M0-SCICAM-MUSCAT')
        self.assertEqual(score, TELESCOPE_MATCH_SITE)
        self.assertIn('E10', evidence)

    def test_telescope_match_site_tier_mismatch(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='Scoring run B',
            window_start=None,
            window_end=None,
            site=self.other_observatory,
        )
        score, evidence = telescope_match_score(run, 'COJ-2m0', '2M0-SCICAM-MUSCAT')
        self.assertEqual(score, TELESCOPE_MATCH_NONE)
        self.assertIn('E10', evidence)

    def test_telescope_match_aperture_only_tier_match(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='LCO 2m0 network',
            window_start=None,
            window_end=None,
        )
        score, _evidence = telescope_match_score(run, '2m0', '2M0-SCICAM-MUSCAT')
        self.assertEqual(score, TELESCOPE_MATCH_APERTURE_ONLY)

    def test_telescope_match_aperture_only_tier_mismatch(self):
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='LCO 1m0 network',
            window_start=None,
            window_end=None,
        )
        score, _evidence = telescope_match_score(run, '2m0', '2M0-SCICAM-MUSCAT')
        self.assertEqual(score, TELESCOPE_MATCH_NONE)

    def test_telescope_match_indeterminate_tier(self):
        """The real CampaignRun pk=1 shape: a site-resolved run (blank telescope_class per
        the D-06 rule) whose telescope_instrument carries no aperture token. Must NOT score
        the same as a genuine mismatch."""
        run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=None,
            window_end=None,
            site=self.observatory,
        )
        score, _evidence = telescope_match_score(run, '2m0', '2M0-SCICAM-MUSCAT')
        self.assertEqual(score, TELESCOPE_MATCH_INDETERMINATE)
        self.assertNotEqual(score, TELESCOPE_MATCH_NONE)

    def test_weighted_sum_arithmetic(self):
        total = WEIGHT_DATE_OVERLAP * 1.0 + WEIGHT_INSTRUMENT_SIMILARITY * 0.0 + WEIGHT_TELESCOPE_MATCH * 1.0
        self.assertAlmostEqual(total, 0.65)

    def test_band_cut_point_boundaries(self):
        self.assertEqual(band_for_score(HIGH_BAND_MIN), BAND_HIGH)
        self.assertEqual(band_for_score(HIGH_BAND_MIN - 0.01), BAND_MEDIUM)
        self.assertEqual(band_for_score(MEDIUM_BAND_MIN), BAND_MEDIUM)
        self.assertEqual(band_for_score(MEDIUM_BAND_MIN - 0.01), BAND_LOW)


class TestCampaignBoundaryGate(TestCase):
    """ATTRIB-03: the campaign/target boundary is the single hard gate -- a cross-campaign
    run is never a candidate at any score, and a same-campaign run with a different target
    IS still a candidate (the failure mode that would break criterion 5 while looking like a
    correct tightening)."""

    @classmethod
    def setUpTestData(cls):
        cls.campaign_a = TargetList.objects.create(name='Boundary Campaign A')
        cls.campaign_b = TargetList.objects.create(name='Boundary Campaign B')
        cls.target_a = NonSiderealTargetFactory.create()
        cls.target_b = NonSiderealTargetFactory.create()
        cls.campaign_a.targets.add(cls.target_a)
        cls.campaign_b.targets.add(cls.target_b)
        cls.record_owner = User.objects.create(username='boundary-record-owner')
        cls.observatory = Observatory.objects.create(obscode='E10', name='Boundary Site', short_name='BS')

        # A run in campaign B, scored perfectly against campaign A's orphans on every signal
        # except the boundary itself.
        cls.run_b = CampaignRun.objects.create(
            campaign=cls.campaign_b,
            telescope_instrument='2m0 2M0-SCICAM-MUSCAT',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 7),
            site=cls.observatory,
        )
        # A run in campaign A whose OWN target differs from the record's target -- must still
        # be offered, since the gate compares campaigns only.
        cls.run_a = CampaignRun.objects.create(
            campaign=cls.campaign_a,
            telescope_instrument='2m0 2M0-SCICAM-MUSCAT',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 7),
            site=cls.observatory,
            target=cls.target_b,
        )

        cls.event_a = CalendarEvent.objects.create(
            title='Boundary event A',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
            telescope='COJ-2m0',
            instrument='2M0-SCICAM-MUSCAT',
            target_list=cls.campaign_a,
        )
        CalendarEventMeta.objects.create(event=cls.event_a, run=None)

        cls.event_no_campaign = CalendarEvent.objects.create(
            title='Conference (no campaign)',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=cls.event_no_campaign, run=None)

        cls.record_a = ObservationRecord.objects.create(
            target=cls.target_a,
            user=cls.record_owner,
            facility='LCO',
            observation_id='BOUND-A',
            status='PENDING',
            parameters={
                'instrument_type': '2M0-SCICAM-MUSCAT',
                'start': '2026-07-07T22:00:00',
                'end': '2026-07-08T06:00:00',
            },
        )

        cls.target_no_campaign = NonSiderealTargetFactory.create()
        cls.record_no_campaign = ObservationRecord.objects.create(
            target=cls.target_no_campaign,
            user=cls.record_owner,
            facility='LCO',
            observation_id='BOUND-NC',
            status='PENDING',
            parameters={
                'instrument_type': '2M0-SCICAM-MUSCAT',
                'start': '2026-07-07T22:00:00',
                'end': '2026-07-08T06:00:00',
            },
        )

    def test_cross_campaign_run_never_offered_for_event_even_at_perfect_score(self):
        candidate_run_pks = {c.run.pk for c in candidates_for_event(self.event_a)}
        self.assertNotIn(self.run_b.pk, candidate_run_pks)

    def test_cross_campaign_run_never_offered_for_record_even_at_perfect_score(self):
        candidate_run_pks = {c.run.pk for c in candidates_for_record(self.record_a)}
        self.assertNotIn(self.run_b.pk, candidate_run_pks)

    def test_event_with_no_target_list_has_zero_candidates(self):
        self.assertEqual(candidates_for_event(self.event_no_campaign), [])

    def test_record_whose_target_belongs_to_no_target_list_has_zero_candidates(self):
        self.assertEqual(candidates_for_record(self.record_no_campaign), [])

    def test_same_campaign_run_with_different_target_is_still_a_candidate(self):
        candidate_run_pks = {c.run.pk for c in candidates_for_record(self.record_a)}
        self.assertIn(self.run_a.pk, candidate_run_pks)


class TestCriterion5RealCase(TestCase):
    """ATTRIB-05, the acceptance test. An EQUIVALENT fixture, never live primary keys --
    RESEARCH.md's live-DB pass found one of the 11 real LCO queue events and one of the 11
    matching observation records already correctly/incorrectly linked, leaving 10 genuine
    orphans per side, not 11; this fixture does not assume 11 and does not "correct"
    CONTEXT.md's/ROADMAP's wording to match.

    A CampaignRun equivalent to the real pk=1 (FTS/MuSCAT4, 7-21 July, Siding Spring E10)
    must be offered -- in the High band, as the sole High candidate -- against ten LCO queue
    calendar events and ten LCO queue observation records, despite the mismatched instrument
    strings ('FTS/MuSCAT4' vs '2M0-SCICAM-MUSCAT') and a coarse aperture-only telescope
    label. If a candidate lands below High, the fix is to retune Task 1's weights/cut-points,
    never to loosen this test's assertions.
    """

    @classmethod
    def setUpTestData(cls):
        cls.campaign = TargetList.objects.create(name='Didymos 2026 (equivalent fixture)')
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.target)
        cls.record_owner = User.objects.create(username='criterion5-record-owner')

        cls.observatory = Observatory.objects.create(
            obscode='E10',
            name='Siding Spring-Faulkes Telescope South (equivalent fixture)',
            short_name='FTS (equivalent fixture)',
        )

        cls.campaign_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=cls.observatory,
            telescope_class='',  # D-06: a site-resolved run carries no class -- the real row's value
        )

        cls.events = []
        for i in range(10):
            start = datetime(2026, 7, 7 + i, 22, 0, tzinfo=dt_timezone.utc)
            end = datetime(2026, 7, 8 + i, 6, 0, tzinfo=dt_timezone.utc)
            event = CalendarEvent.objects.create(
                title=f'[QUEUED] 2m0 2M0-SCICAM-MUSCAT (night {i})',
                start_time=start,
                end_time=end,
                telescope='2m0',
                instrument='2M0-SCICAM-MUSCAT',
                target_list=cls.campaign,
            )
            CalendarEventMeta.objects.create(event=event, is_verified=False, run=None)
            cls.events.append(event)

        # The one site-resolved-label variant among the real 11 (COJ-2m0) -- still an orphan.
        cls.site_resolved_event = CalendarEvent.objects.create(
            title='2m0 2M0-SCICAM-MUSCAT (site-resolved label)',
            start_time=datetime(2026, 7, 17, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 18, 6, 0, tzinfo=dt_timezone.utc),
            telescope='COJ-2m0',
            instrument='2M0-SCICAM-MUSCAT',
            target_list=cls.campaign,
        )
        CalendarEventMeta.objects.create(event=cls.site_resolved_event, is_verified=True, run=None)

        cls.records = []
        for i in range(10):
            start = datetime(2026, 7, 7 + i, 22, 0)  # naive -- matches the real records' shape
            end = datetime(2026, 7, 8 + i, 6, 0)
            record = ObservationRecord.objects.create(
                target=cls.target,
                user=cls.record_owner,
                facility='LCO',
                observation_id=f'CRIT5-{i}',
                status='PENDING',
                parameters={
                    'instrument_type': '2M0-SCICAM-MUSCAT',
                    'start': start.isoformat(),
                    'end': end.isoformat(),
                },
                # scheduled_start left unset (None) -- the real records' common case.
            )
            cls.records.append(record)

    def test_every_orphan_calendar_event_offers_the_run_in_the_high_band_and_is_checkboxable(self):
        groups_by_pk = {g.orphan.pk: g for g in event_attribution_backlog()}
        for event in [*self.events, self.site_resolved_event]:
            group = groups_by_pk[event.pk]
            candidates_by_run = {c.run.pk: c for c in group.candidates}

            self.assertIn(self.campaign_run.pk, candidates_by_run)
            self.assertEqual(candidates_by_run[self.campaign_run.pk].band, BAND_HIGH)
            self.assertEqual(group.sole_high_candidate_pk, self.campaign_run.pk)

    def test_every_orphan_observation_record_offers_the_run_in_the_high_band_and_is_checkboxable(self):
        groups_by_pk = {g.orphan.pk: g for g in record_attribution_backlog()}
        for record in self.records:
            group = groups_by_pk[record.pk]
            candidates_by_run = {c.run.pk: c for c in group.candidates}

            self.assertIn(self.campaign_run.pk, candidates_by_run)
            self.assertEqual(candidates_by_run[self.campaign_run.pk].band, BAND_HIGH)
            self.assertEqual(group.sole_high_candidate_pk, self.campaign_run.pk)


class TestDismissalExclusion(TestCase):
    """A dismissed (orphan, run) pair disappears from that orphan's candidate list; the
    orphan's OTHER candidates survive; and an orphan whose every candidate is dismissed drops
    out of the backlog entirely while being counted by unattributable_orphan_count() (D-05,
    ATTRIB-06 -- this is what makes the queue drainable)."""

    @classmethod
    def setUpTestData(cls):
        cls.campaign = TargetList.objects.create(name='Dismissal Campaign')
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.target)
        cls.record_owner = User.objects.create(username='dismissal-record-owner')
        cls.observatory = Observatory.objects.create(obscode='E10', name='Dismissal Site', short_name='DS')

        cls.run_keep = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=cls.observatory,
        )
        cls.run_dismiss = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=cls.observatory,
        )

        cls.event = CalendarEvent.objects.create(
            title='Dismissal test event',
            start_time=datetime(2026, 7, 10, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 11, 6, 0, tzinfo=dt_timezone.utc),
            telescope='2m0',
            instrument='2M0-SCICAM-MUSCAT',
            target_list=cls.campaign,
        )
        CalendarEventMeta.objects.create(event=cls.event, run=None)

    def test_dismissed_pair_disappears_but_other_candidates_survive(self):
        CalendarEventDismissal.objects.create(event=self.event, run=self.run_dismiss)

        run_pks = {c.run.pk for c in candidates_for_event(self.event)}

        self.assertNotIn(self.run_dismiss.pk, run_pks)
        self.assertIn(self.run_keep.pk, run_pks)

    def test_orphan_with_every_candidate_dismissed_drops_from_backlog_and_is_counted_unattributable(self):
        CalendarEventDismissal.objects.create(event=self.event, run=self.run_keep)
        CalendarEventDismissal.objects.create(event=self.event, run=self.run_dismiss)

        self.assertEqual(candidates_for_event(self.event), [])

        backlog_pks = {g.orphan.pk for g in event_attribution_backlog()}
        self.assertNotIn(self.event.pk, backlog_pks)
        self.assertGreaterEqual(unattributable_orphan_count(), 1)

    def test_dismissed_pair_disappears_for_record_side_too(self):
        record = ObservationRecord.objects.create(
            target=self.target,
            user=self.record_owner,
            facility='LCO',
            observation_id='DISMISS-REC-1',
            status='PENDING',
            parameters={
                'instrument_type': '2M0-SCICAM-MUSCAT',
                'start': '2026-07-10T22:00:00',
                'end': '2026-07-11T06:00:00',
            },
        )
        ObservationRecordDismissal.objects.create(observation_record=record, run=self.run_dismiss)

        run_pks = {c.run.pk for c in candidates_for_record(record)}

        self.assertNotIn(self.run_dismiss.pk, run_pks)
        self.assertIn(self.run_keep.pk, run_pks)


class TestOrphanQuerysets(TestCase):
    """RESEARCH.md Pitfall 2: an event with no CalendarEventMeta companion row at all is just
    as much an orphan as one whose companion row exists with run unset."""

    def test_event_with_no_calendar_event_meta_row_at_all_is_included(self):
        event = CalendarEvent.objects.create(
            title='Classical event, no companion row',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
        )

        self.assertIn(event.pk, {e.pk for e in orphan_calendar_events()})

    def test_event_whose_companion_row_has_a_run_is_excluded(self):
        campaign = TargetList.objects.create(name='Orphan Queryset Meta Campaign')
        run = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 7),
        )
        event = CalendarEvent.objects.create(
            title='Already attributed event',
            start_time=datetime(2026, 7, 7, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 8, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=event, run=run)

        self.assertNotIn(event.pk, {e.pk for e in orphan_calendar_events()})

    def test_record_with_a_campaign_run_observation_row_is_excluded(self):
        campaign = TargetList.objects.create(name='Orphan Queryset Record Campaign')
        target = NonSiderealTargetFactory.create()
        user = User.objects.create(username='orphan-queryset-record-owner')
        run = CampaignRun.objects.create(
            campaign=campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 7),
        )
        record = ObservationRecord.objects.create(
            target=target,
            user=user,
            facility='LCO',
            observation_id='ORPHANQ-1',
            status='PENDING',
            parameters={'proposal': 'TEST'},
        )
        CampaignRunObservation.objects.create(run=run, observation_record=record)

        self.assertNotIn(record.pk, {r.pk for r in orphan_observation_records()})
