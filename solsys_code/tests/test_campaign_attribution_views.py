"""Phase 28's staff attribution write path -- POST action, race, tampering and staff-gating
tests (28-VALIDATION.md Wave 0), plus Plan 28-04's page-rendering tests (``TestEvidenceColumns``,
``TestBandFilterAndBanner``, ``TestQueueDrainsToEmpty``), which reuse the same shared fixture and
helper conventions below.

28-03's classes predate the template (``attribution_queue.html``, added by Plan 28-04) and so
never GET-render the attribution page expecting a 200 -- see their own class docstrings for the
reasoning that shaped their assertions (un-followed POST responses, ``RequestFactory``-only GET
context assembly). Plan 28-04's classes below are written against the real template and DO
GET-render the page directly through ``self.client.get()``, now that it exists.
"""

from datetime import date, datetime
from datetime import timezone as dt_timezone

from django.contrib.auth.models import User
from django.contrib.messages import get_messages
from django.test import RequestFactory, TestCase
from django.urls import reverse
from django.utils.html import escape
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import campaign_attribution
from solsys_code.campaign_attribution import candidates_for_event, event_attribution_backlog
from solsys_code.campaign_views import AttributionQueueView
from solsys_code.models import (
    CalendarEventDismissal,
    CalendarEventMeta,
    CampaignRun,
    CampaignRunObservation,
    ObservationRecordDismissal,
)
from solsys_code.solsys_code_observatory.models import Observatory


class AttributionViewTestBase(TestCase):
    """Shared fixture (both this plan's and Plan 28-04's test classes use it): a campaign
    ``TargetList``, a ``NonSiderealTargetFactory`` target, an ``Observatory`` with
    ``obscode='E10'``, one ``CampaignRun`` shaped like the real pk=1 row (criterion 5's
    reference case -- FTS/MuSCAT4, 7-21 July, Siding Spring E10), a staff user, a non-staff
    user, and a ``record_owner`` user distinct from both. ``_make_event()``/``_make_record()``
    build High-band orphans against ``cls.campaign_run`` by default, matching
    ``test_campaign_attribution.py``'s own ``TestCriterion5RealCase`` fixture shape. Named
    ``campaign_run``, never ``run`` -- 28-02-SUMMARY.md's documented pitfall: a bare ``cls.run``
    attribute overwrites ``unittest.TestCase.run()`` and crashes the test runner with no
    per-test traceback.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.campaign = TargetList.objects.create(name='Didymos 2026')
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign.targets.add(cls.target)
        cls.observatory = Observatory.objects.create(obscode='E10', name='Siding Spring', short_name='SSO')

        cls.staff_user = User.objects.create_user(username='staffcoordinator', password='pw', is_staff=True)
        cls.non_staff_user = User.objects.create_user(username='regularobserver', password='pw', is_staff=False)
        cls.record_owner = User.objects.create(username='record-owner')

        cls.campaign_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=cls.observatory,
            telescope_class='',  # D-06: a site-resolved run carries no class.
        )

        # A CampaignRun in a DIFFERENT campaign -- never eligible for anything fixtured
        # against cls.campaign/cls.target. Used by TestConcurrencyAndTampering to prove a
        # tampered POST naming a cross-boundary run writes nothing.
        cls.other_campaign = TargetList.objects.create(name='3I/ATLAS (other campaign)')
        cls.other_run = CampaignRun.objects.create(
            campaign=cls.other_campaign,
            telescope_instrument='JWST',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
        )

    def _make_event(self, night_offset: int = 0, **overrides) -> CalendarEvent:
        """One orphan CalendarEvent, High-band against cls.campaign_run by default, plus its
        unowned CalendarEventMeta companion (RESEARCH.md Pitfall 2 shape: the companion row
        exists, run is unset)."""
        start = datetime(2026, 7, 7 + night_offset, 22, 0, tzinfo=dt_timezone.utc)
        end = datetime(2026, 7, 8 + night_offset, 6, 0, tzinfo=dt_timezone.utc)
        fields = {
            'title': f'[QUEUED] 2m0 2M0-SCICAM-MUSCAT (night {night_offset})',
            'start_time': start,
            'end_time': end,
            'telescope': 'COJ-2m0',
            'instrument': '2M0-SCICAM-MUSCAT',
            'target_list': self.campaign,
        }
        fields.update(overrides)
        event = CalendarEvent.objects.create(**fields)
        CalendarEventMeta.objects.create(event=event, is_verified=False, run=None)
        return event

    def _make_record(self, night_offset: int = 0, observation_id: str | None = None, **overrides) -> ObservationRecord:
        """One orphan ObservationRecord, High-band against cls.campaign_run by default (no
        CampaignRunObservation row yet)."""
        start = datetime(2026, 7, 7 + night_offset, 22, 0)
        end = datetime(2026, 7, 8 + night_offset, 6, 0)
        fields = {
            'target': self.target,
            'user': self.record_owner,
            'facility': 'LCO',
            'observation_id': observation_id or f'ATTR-{night_offset}',
            'status': 'PENDING',
            'parameters': {
                'instrument_type': '2M0-SCICAM-MUSCAT',
                'start': start.isoformat(),
                'end': end.isoformat(),
            },
        }
        fields.update(overrides)
        return ObservationRecord.objects.create(**fields)

    @staticmethod
    def _message_strings(response) -> list[str]:
        """Read django.contrib.messages off a POST response WITHOUT following its redirect --
        the redirect target (campaigns:attribution) has no template until Plan 28-04, so
        following it would raise TemplateDoesNotExist."""
        return [str(m) for m in get_messages(response.wsgi_request)]


class TestAttributionStaffGating(AttributionViewTestBase):
    """T-28-10: anonymous/non-staff access must redirect, never write, and a GET to the
    decide endpoint (POST-only) returns 405."""

    def test_anonymous_get_attribution_redirects(self):
        url = reverse('campaigns:attribution')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)

    def test_non_staff_get_attribution_redirects(self):
        url = reverse('campaigns:attribution')
        self.client.login(username='regularobserver', password='pw')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)

    def test_anonymous_post_decide_redirects_and_writes_nothing(self):
        event = self._make_event()
        before_meta = CalendarEventMeta.objects.filter(run__isnull=False).count()
        before_obs = CampaignRunObservation.objects.count()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(CalendarEventMeta.objects.filter(run__isnull=False).count(), before_meta)
        self.assertEqual(CampaignRunObservation.objects.count(), before_obs)

    def test_non_staff_post_decide_redirects_and_writes_nothing(self):
        event = self._make_event()
        self.client.login(username='regularobserver', password='pw')
        before_meta = CalendarEventMeta.objects.filter(run__isnull=False).count()
        before_obs = CampaignRunObservation.objects.count()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(CalendarEventMeta.objects.filter(run__isnull=False).count(), before_meta)
        self.assertEqual(CampaignRunObservation.objects.count(), before_obs)

    def test_staff_get_decide_returns_405(self):
        self.client.login(username='staffcoordinator', password='pw')
        response = self.client.get(reverse('campaigns:attribution_decide'))
        self.assertEqual(response.status_code, 405)


class TestConfirmUndo(AttributionViewTestBase):
    """ATTRIB-04: for BOTH orphan kinds, confirm stamps who/when and creates the association;
    undo_confirmation clears/deletes the link and writes a dismissal row naming the same
    user with a timestamp; the orphan returns to the backlog only once that dismissal is
    itself undone."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_confirm_event_creates_link_and_stamps_audit_fields(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        meta = CalendarEventMeta.objects.get(event=event)
        self.assertEqual(meta.run_id, self.campaign_run.pk)
        self.assertEqual(meta.confirmed_by, self.staff_user)
        self.assertIsNotNone(meta.confirmed_at)
        self.assertIn('Attribution confirmed.', self._message_strings(response))

    def test_confirm_record_creates_link_and_stamps_audit_fields(self):
        record = self._make_record()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'record', 'orphan_pk': record.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        link = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link.run_id, self.campaign_run.pk)
        self.assertEqual(link.confirmed_by, self.staff_user)
        self.assertIsNotNone(link.confirmed_at)
        self.assertIn('Attribution confirmed.', self._message_strings(response))

    def test_undo_confirmation_event_clears_link_and_writes_dismissal(self):
        event = self._make_event()
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'undo_confirmation', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        meta = CalendarEventMeta.objects.get(event=event)
        self.assertIsNone(meta.run_id)
        self.assertIsNone(meta.confirmed_by)
        self.assertIsNone(meta.confirmed_at)
        dismissal = CalendarEventDismissal.objects.get(event=event, run=self.campaign_run)
        self.assertEqual(dismissal.dismissed_by, self.staff_user)
        self.assertIsNotNone(dismissal.dismissed_at)
        self.assertIn('Confirmation undone — back in the queue.', self._message_strings(response))
        # The orphan does NOT return to the backlog while the undo's dismissal stands.
        backlog_pks = {g.orphan.pk for g in event_attribution_backlog()}
        self.assertNotIn(event.pk, backlog_pks)

    def test_undo_confirmation_record_deletes_link_and_writes_dismissal(self):
        record = self._make_record()
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'record', 'orphan_pk': record.pk, 'run_pk': self.campaign_run.pk},
        )
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'undo_confirmation', 'kind': 'record', 'orphan_pk': record.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(CampaignRunObservation.objects.filter(observation_record=record).exists())
        dismissal = ObservationRecordDismissal.objects.get(observation_record=record, run=self.campaign_run)
        self.assertEqual(dismissal.dismissed_by, self.staff_user)
        self.assertIsNotNone(dismissal.dismissed_at)
        self.assertIn('Confirmation undone — back in the queue.', self._message_strings(response))

    def test_orphan_returns_to_backlog_only_after_dismissal_is_itself_undone(self):
        event = self._make_event()
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'undo_confirmation', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        backlog_pks = {g.orphan.pk for g in event_attribution_backlog()}
        self.assertNotIn(event.pk, backlog_pks)

        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'undo_dismissal', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(CalendarEventDismissal.objects.filter(event=event, run=self.campaign_run).exists())
        backlog_pks = {g.orphan.pk for g in event_attribution_backlog()}
        self.assertIn(event.pk, backlog_pks)
        self.assertIn('Dismissal undone — back in the queue.', self._message_strings(response))


class TestDismissAndUndoDismissal(AttributionViewTestBase):
    """ATTRIB-03: a blank reason writes nothing (server enforces D-06, not just the browser's
    ``required`` attribute); a real reason records who/when/why; a dismissed pair disappears
    while the orphan's other candidates survive; undo_dismissal offers the pair again."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_dismiss_blank_reason_writes_nothing_and_errors(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {
                'action': 'dismiss',
                'kind': 'event',
                'orphan_pk': event.pk,
                'run_pk': self.campaign_run.pk,
                'reason': '   ',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(CalendarEventDismissal.objects.filter(event=event, run=self.campaign_run).exists())
        messages_list = self._message_strings(response)
        self.assertTrue(any('reason' in m.lower() for m in messages_list))

    def test_dismiss_with_reason_creates_row_with_who_when_reason(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {
                'action': 'dismiss',
                'kind': 'event',
                'orphan_pk': event.pk,
                'run_pk': self.campaign_run.pk,
                'reason': 'Different campaign entirely.',
            },
        )
        self.assertEqual(response.status_code, 302)
        dismissal = CalendarEventDismissal.objects.get(event=event, run=self.campaign_run)
        self.assertEqual(dismissal.dismissed_by, self.staff_user)
        self.assertIsNotNone(dismissal.dismissed_at)
        self.assertEqual(dismissal.reason, 'Different campaign entirely.')
        self.assertIn('Candidate dismissed.', self._message_strings(response))

    def test_dismissed_pair_disappears_but_other_candidates_survive(self):
        event = self._make_event()
        other_run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=self.observatory,
        )
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {
                'action': 'dismiss',
                'kind': 'event',
                'orphan_pk': event.pk,
                'run_pk': self.campaign_run.pk,
                'reason': 'Wrong.',
            },
        )
        run_pks = {c.run.pk for c in candidates_for_event(event)}
        self.assertNotIn(self.campaign_run.pk, run_pks)
        self.assertIn(other_run.pk, run_pks)

    def test_undo_dismissal_deletes_row_and_pair_is_offered_again(self):
        event = self._make_event()
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {
                'action': 'dismiss',
                'kind': 'event',
                'orphan_pk': event.pk,
                'run_pk': self.campaign_run.pk,
                'reason': 'Test dismiss.',
            },
        )
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'undo_dismissal', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(CalendarEventDismissal.objects.filter(event=event, run=self.campaign_run).exists())
        run_pks = {c.run.pk for c in candidates_for_event(event)}
        self.assertIn(self.campaign_run.pk, run_pks)
        self.assertIn('Dismissal undone — back in the queue.', self._message_strings(response))

    def test_undo_dismissal_missing_row_errors(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'undo_dismissal', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn('This candidate no longer exists.', self._message_strings(response))


class TestBulkConfirmGate(AttributionViewTestBase):
    """ATTRIB-02/D-09: confirm_selected accepts only server-verified sole-High-band
    candidates, per pair, and loops rather than issuing one combined update."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_medium_band_candidate_confirms_nothing_but_single_confirm_accepts_it(self):
        # Partial date overlap (2 of 3 days) plus an unrecognised telescope alias
        # (TELESCOPE_MATCH_INDETERMINATE) keeps this pair's total score in the Medium band --
        # see test_campaign_attribution.py's own scoring tests for the same three-signal math.
        event = self._make_event(
            start_time=datetime(2026, 7, 20, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 22, 6, 0, tzinfo=dt_timezone.utc),
            telescope='Unresolved-1m0',
        )
        candidate = next(c for c in candidates_for_event(event) if c.run.pk == self.campaign_run.pk)
        self.assertEqual(candidate.band, campaign_attribution.BAND_MEDIUM)

        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm_selected', 'candidate_ids': [f'event:{event.pk}:{self.campaign_run.pk}']},
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(CalendarEventMeta.objects.filter(event=event, run__isnull=False).exists())
        self.assertIn(
            'This candidate was already confirmed or dismissed by someone else.', self._message_strings(response)
        )

        single_response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(single_response.status_code, 302)
        self.assertTrue(CalendarEventMeta.objects.filter(event=event, run_id=self.campaign_run.pk).exists())

    def test_high_band_candidate_that_is_not_sole_high_is_refused(self):
        event = self._make_event()
        second_run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=self.observatory,
        )
        candidates_by_run = {c.run.pk: c for c in candidates_for_event(event)}
        self.assertEqual(candidates_by_run[self.campaign_run.pk].band, campaign_attribution.BAND_HIGH)
        self.assertEqual(candidates_by_run[second_run.pk].band, campaign_attribution.BAND_HIGH)

        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm_selected', 'candidate_ids': [f'event:{event.pk}:{self.campaign_run.pk}']},
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(CalendarEventMeta.objects.filter(event=event, run__isnull=False).exists())
        self.assertIn(
            'This candidate was already confirmed or dismissed by someone else.', self._message_strings(response)
        )

    def test_several_sole_high_candidates_across_different_runs_all_confirm(self):
        """Proves the per-pair loop, not a single combined update: two orphans in two
        DIFFERENT campaigns (so their candidate pools never overlap), each with its own
        run as its sole High candidate, both confirm in one submit."""
        event = self._make_event()  # sole High candidate: self.campaign_run

        campaign_b = TargetList.objects.create(name='Second Campaign')
        target_b = NonSiderealTargetFactory.create()
        campaign_b.targets.add(target_b)
        run_b = CampaignRun.objects.create(
            campaign=campaign_b,
            telescope_instrument='FTS/MuSCAT4',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=self.observatory,
            telescope_class='',
        )
        record = ObservationRecord.objects.create(
            target=target_b,
            user=self.record_owner,
            facility='LCO',
            observation_id='BULK-DIFF-RUN',
            status='PENDING',
            parameters={
                'instrument_type': '2M0-SCICAM-MUSCAT',
                'start': datetime(2026, 7, 10, 22, 0).isoformat(),
                'end': datetime(2026, 7, 11, 6, 0).isoformat(),
            },
        )

        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {
                'action': 'confirm_selected',
                'candidate_ids': [f'event:{event.pk}:{self.campaign_run.pk}', f'record:{record.pk}:{run_b.pk}'],
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(CalendarEventMeta.objects.filter(event=event, run_id=self.campaign_run.pk).exists())
        self.assertTrue(CampaignRunObservation.objects.filter(observation_record=record, run_id=run_b.pk).exists())
        self.assertIn('2 candidates confirmed.', self._message_strings(response))


class TestConcurrencyAndTampering(AttributionViewTestBase):
    """Double-submit no-ops; the two-different-runs-race leaves exactly one link (Pitfall 3);
    a cross-campaign-boundary tampered POST writes nothing; malformed input returns 400."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_double_submit_confirm_is_noop_second_time(self):
        event = self._make_event()
        response1 = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response1.status_code, 302)
        response2 = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response2.status_code, 302)
        self.assertEqual(CalendarEventMeta.objects.filter(event=event, run_id=self.campaign_run.pk).count(), 1)
        self.assertIn(
            'This candidate was already confirmed or dismissed by someone else.',
            self._message_strings(response2),
        )

    def test_two_different_runs_confirmed_against_same_record_leaves_one_link(self):
        """RESEARCH.md Pitfall 3: the unique_campaign_run_observation_record constraint is on
        observation_record ALONE -- simulate two DIFFERENT runs racing for the SAME record,
        not the same pair twice."""
        record = self._make_record()
        second_run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=self.observatory,
        )
        response1 = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'record', 'orphan_pk': record.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response1.status_code, 302)
        response2 = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'record', 'orphan_pk': record.pk, 'run_pk': second_run.pk},
        )
        self.assertEqual(response2.status_code, 302)
        self.assertEqual(CampaignRunObservation.objects.filter(observation_record=record).count(), 1)
        self.assertEqual(CampaignRunObservation.objects.get(observation_record=record).run_id, self.campaign_run.pk)
        self.assertIn(
            'This candidate was already confirmed or dismissed by someone else.',
            self._message_strings(response2),
        )

    def test_cross_campaign_run_post_writes_nothing(self):
        """The executable form of criterion 3's absolute cross-boundary prohibition: a POST
        naming a run from a DIFFERENT campaign is never offered, so is_offered_candidate()
        rejects it regardless of what the (tampered) form claims."""
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.other_run.pk},
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(CalendarEventMeta.objects.filter(event=event, run__isnull=False).exists())
        self.assertIn(
            'This candidate was already confirmed or dismissed by someone else.', self._message_strings(response)
        )

    def test_malformed_action_returns_400_and_writes_nothing(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'bogus', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(CalendarEventMeta.objects.filter(event=event, run__isnull=False).exists())

    def test_malformed_kind_returns_400_and_writes_nothing(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'bogus', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(CalendarEventMeta.objects.filter(event=event, run__isnull=False).exists())

    def test_non_integer_pk_returns_400_and_writes_nothing(self):
        event = self._make_event()
        response = self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': 'abc', 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(CalendarEventMeta.objects.filter(event=event, run__isnull=False).exists())


class TestAttributionQueueViewContext(AttributionViewTestBase):
    """AttributionQueueView's GET context assembly, exercised directly against the view
    class (never through ``self.client.get()``) since the template does not exist until
    Plan 28-04 -- ``TemplateView.get()`` would otherwise raise ``TemplateDoesNotExist``."""

    def _get_context(self, get_params: dict | None = None) -> dict:
        factory = RequestFactory()
        request = factory.get(reverse('campaigns:attribution'), get_params or {})
        request.user = self.staff_user
        view = AttributionQueueView()
        view.request = request
        view.kwargs = {}
        return view.get_context_data()

    def test_context_includes_event_and_record_orphan_groups(self):
        event = self._make_event()
        record = self._make_record()
        context = self._get_context()
        event_orphan_pks = {g.orphan.pk for g in context['event_groups']}
        record_orphan_pks = {g.orphan.pk for g in context['record_groups']}
        self.assertIn(event.pk, event_orphan_pks)
        self.assertIn(record.pk, record_orphan_pks)

    def test_unrecognised_band_falls_back_to_none(self):
        context = self._get_context({'band': 'not-a-real-band'})
        self.assertIsNone(context['band'])

    def test_is_drained_true_when_no_candidates_exist(self):
        context = self._get_context()
        self.assertTrue(context['is_drained'])
        self.assertEqual(context['attribution_count'], 0)

    def test_is_drained_false_once_a_candidate_exists(self):
        self._make_event()
        context = self._get_context()
        self.assertFalse(context['is_drained'])
        self.assertGreaterEqual(context['attribution_count'], 1)

    def test_dismissed_and_confirmed_rows_reflect_recent_actions(self):
        event = self._make_event()
        CalendarEventDismissal.objects.create(event=event, run=self.campaign_run, dismissed_by=self.staff_user)
        record = self._make_record()
        CampaignRunObservation.objects.create(
            run=self.campaign_run, observation_record=record, confirmed_by=self.staff_user
        )

        context = self._get_context()

        self.assertTrue(
            any(
                isinstance(row, CalendarEventDismissal) and row.event_id == event.pk
                for row in context['dismissed_rows']
            )
        )
        self.assertTrue(
            any(
                isinstance(row, CampaignRunObservation) and row.observation_record_id == record.pk
                for row in context['confirmed_rows']
            )
        )


class TestEvidenceColumns(AttributionViewTestBase):
    """ATTRIB-01: a staff GET renders the page (200), and every candidate row shows the four
    evidence facts as DISTINCT content plus the numeric score IN ADDITION to them -- pins
    "score in addition to the evidence" rather than "score instead of it", ROADMAP criterion
    1's explicit "not a bare score". Also pins the free-text dismissal-reason stored-XSS
    control (RESEARCH.md Security Domain) -- the only community-influenced string this page
    renders."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def test_evidence_columns_score_and_band_badge_all_present(self):
        event = self._make_event()
        candidate = next(c for c in candidates_for_event(event) if c.run.pk == self.campaign_run.pk)
        self.assertEqual(candidate.band, campaign_attribution.BAND_HIGH)

        response = self.client.get(reverse('campaigns:attribution'))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()

        # The four evidence facts, each a distinct string built by the matcher (never the
        # template), so a rendering bug can never accidentally reduce a candidate to a score.
        self.assertIn(escape(candidate.telescope_evidence), content)
        self.assertIn(escape(candidate.date_evidence), content)
        self.assertIn(escape(candidate.campaign_evidence), content)
        self.assertIn(escape(candidate.instrument_evidence), content)
        # The score is ALSO present -- additional to the evidence, never a replacement.
        self.assertIn(str(candidate.score), content)
        # This candidate is the sole High-band candidate for its orphan, so per the
        # Checkbox Gate Contract (28-UI-SPEC.md) its leading cell renders a checkbox
        # instead of the confidence badge -- the row's border-left border-success is
        # the High-band signal here. TestBandFilterAndBanner's
        # test_checkbox_absent_when_orphan_has_two_high_band_candidates exercises the
        # non-checkboxable High case where badge-success itself renders.
        self.assertIn('border-success', content)

    def test_non_staff_get_redirects(self):
        url = reverse('campaigns:attribution')
        self.client.logout()
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)

    def test_dismissal_reason_xss_payload_is_escaped(self):
        event = self._make_event()
        CalendarEventDismissal.objects.create(
            event=event,
            run=self.campaign_run,
            dismissed_by=self.staff_user,
            reason='<script>alert(1)</script>',
        )
        response = self.client.get(reverse('campaigns:attribution'))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<script>alert(1)</script>')
        self.assertContains(response, '&lt;script&gt;alert(1)&lt;/script&gt;')


class TestBandFilterAndBanner(AttributionViewTestBase):
    """ATTRIB-02, D-02/D-10: ``?band=high``/``?band=medium`` narrow the rendered worklist, an
    unrecognised ``?band=`` value falls back to showing everything, the checkbox gate is
    exercised end-to-end through the rendered page (sole-High only), and the campaign-list
    banner is staff-only -- the executable form of ``campaign_list.html``'s nested-``{% if %}``
    warning."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def _medium_band_event(self):
        """A candidate that lands Medium: partial date overlap plus an unrecognised telescope
        alias (TELESCOPE_MATCH_INDETERMINATE), matching TestBulkConfirmGate's own fixture shape."""
        return self._make_event(
            start_time=datetime(2026, 7, 20, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 7, 22, 6, 0, tzinfo=dt_timezone.utc),
            telescope='Unresolved-1m0',
        )

    def test_band_high_filter_shows_only_high_band_candidates(self):
        high_event = self._make_event()
        medium_event = self._medium_band_event()
        high_candidate = next(c for c in candidates_for_event(high_event) if c.run.pk == self.campaign_run.pk)
        medium_candidate = next(c for c in candidates_for_event(medium_event) if c.run.pk == self.campaign_run.pk)
        self.assertEqual(medium_candidate.band, campaign_attribution.BAND_MEDIUM)

        response = self.client.get(reverse('campaigns:attribution'), {'band': 'high'})
        content = response.content.decode()
        self.assertIn(escape(high_candidate.telescope_evidence), content)
        self.assertNotIn(escape(medium_candidate.date_evidence), content)

    def test_band_medium_filter_shows_only_medium_band_candidates(self):
        high_event = self._make_event()
        medium_event = self._medium_band_event()
        high_candidate = next(c for c in candidates_for_event(high_event) if c.run.pk == self.campaign_run.pk)
        medium_candidate = next(c for c in candidates_for_event(medium_event) if c.run.pk == self.campaign_run.pk)

        response = self.client.get(reverse('campaigns:attribution'), {'band': 'medium'})
        content = response.content.decode()
        self.assertIn(escape(medium_candidate.date_evidence), content)
        self.assertNotIn(escape(high_candidate.telescope_evidence), content)

    def test_unrecognised_band_falls_back_to_showing_all(self):
        event = self._make_event()
        candidate = next(c for c in candidates_for_event(event) if c.run.pk == self.campaign_run.pk)
        response = self.client.get(reverse('campaigns:attribution'), {'band': 'not-a-real-band'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(escape(candidate.telescope_evidence), response.content.decode())

    def test_checkbox_renders_for_sole_high_band_candidate(self):
        event = self._make_event()
        response = self.client.get(reverse('campaigns:attribution'))
        self.assertIn(f'value="event:{event.pk}:{self.campaign_run.pk}"', response.content.decode())

    def test_checkbox_absent_for_medium_band_candidate(self):
        event = self._medium_band_event()
        response = self.client.get(reverse('campaigns:attribution'))
        self.assertNotIn(f'value="event:{event.pk}:{self.campaign_run.pk}"', response.content.decode())

    def test_checkbox_absent_when_orphan_has_two_high_band_candidates(self):
        event = self._make_event()
        second_run = CampaignRun.objects.create(
            campaign=self.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start=date(2026, 7, 7),
            window_end=date(2026, 7, 21),
            site=self.observatory,
        )
        candidates_by_run = {c.run.pk: c for c in candidates_for_event(event)}
        self.assertEqual(candidates_by_run[self.campaign_run.pk].band, campaign_attribution.BAND_HIGH)
        self.assertEqual(candidates_by_run[second_run.pk].band, campaign_attribution.BAND_HIGH)

        content = self.client.get(reverse('campaigns:attribution')).content.decode()
        self.assertNotIn(f'value="event:{event.pk}:{self.campaign_run.pk}"', content)
        self.assertNotIn(f'value="event:{event.pk}:{second_run.pk}"', content)
        # Neither candidate is checkboxable, so both render the confidence badge in the
        # leading cell instead -- the one case where badge-success itself appears.
        self.assertIn('badge-success', content)

    def test_staff_campaign_list_banner_shows_count_and_link(self):
        self._make_event()
        response = self.client.get(reverse('campaigns:list'))
        self.assertContains(response, 'awaiting attribution')
        self.assertContains(response, reverse('campaigns:attribution'))

    def test_anonymous_campaign_list_banner_shows_neither(self):
        self._make_event()
        self.client.logout()
        response = self.client.get(reverse('campaigns:list'))
        content = response.content.decode()
        self.assertNotIn('awaiting attribution', content)
        self.assertNotIn(reverse('campaigns:attribution'), content)


class TestQueueDrainsToEmpty(AttributionViewTestBase):
    """ATTRIB-06/D-15: an end-to-end pass -- confirm or dismiss every offered candidate, then
    GET the page and prove both worklists are empty, the "Attribution complete" heading is
    present, the stated remaining count matches ``unattributable_orphan_count()``, and
    ``orphans_needing_attribution_count()`` is zero -- so the campaign-list banner's
    attribution clause is gone too. This is the test that makes ATTRIB-06's "attribution can
    be completed before the first reconcile sweep" a checkable fact, which Phase 29 points at
    as its precondition."""

    def setUp(self):
        self.client.login(username='staffcoordinator', password='pw')

    def _make_unattributable_event(self):
        """An orphan with NO target_list at all (a conference/proposal-deadline event, D-03) --
        never offers a candidate, so it can never be confirmed or dismissed away; it stays in
        the "still have no matching run" count forever."""
        return CalendarEvent.objects.create(
            title='Conference (no campaign)',
            start_time=datetime(2026, 8, 1, 0, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 8, 2, 0, 0, tzinfo=dt_timezone.utc),
        )

    def test_confirming_and_dismissing_every_candidate_drains_the_queue(self):
        event = self._make_event()
        record = self._make_record()
        self._make_unattributable_event()

        self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'record', 'orphan_pk': record.pk, 'run_pk': self.campaign_run.pk},
        )

        self.assertEqual(campaign_attribution.orphans_needing_attribution_count(), 0)
        expected_remaining = campaign_attribution.unattributable_orphan_count()
        self.assertGreaterEqual(expected_remaining, 1)

        response = self.client.get(reverse('campaigns:attribution'))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('Attribution complete', content)
        self.assertIn(f'{expected_remaining} orphan', content)
        self.assertIn('No calendar events awaiting attribution', content)
        self.assertIn('No observation records awaiting attribution', content)

        list_response = self.client.get(reverse('campaigns:list'))
        self.assertNotContains(list_response, 'awaiting attribution')

    def test_dismissing_the_only_candidate_also_drains_that_orphan(self):
        """A dismissed pair leaves its orphan with zero surviving candidates (this fixture's
        only run is dismissed), which is D-03's noise filter working the same way for a
        dismissal outcome as for a never-had-a-candidate orphan -- both stop appearing in
        either worklist, and both count toward unattributable_orphan_count()."""
        event = self._make_event()
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {
                'action': 'dismiss',
                'kind': 'event',
                'orphan_pk': event.pk,
                'run_pk': self.campaign_run.pk,
                'reason': 'Wrong pair.',
            },
        )
        self.assertEqual(campaign_attribution.orphans_needing_attribution_count(), 0)
        self.assertGreaterEqual(campaign_attribution.unattributable_orphan_count(), 1)

        response = self.client.get(reverse('campaigns:attribution'))
        self.assertIn('Attribution complete', response.content.decode())

    def test_zero_remaining_orphans_renders_none_not_zero_orphans(self):
        event = self._make_event()
        self.client.post(
            reverse('campaigns:attribution_decide'),
            {'action': 'confirm', 'kind': 'event', 'orphan_pk': event.pk, 'run_pk': self.campaign_run.pk},
        )
        self.assertEqual(campaign_attribution.unattributable_orphan_count(), 0)

        response = self.client.get(reverse('campaigns:attribution'))
        content = response.content.decode()
        self.assertIn('Attribution complete', content)
        self.assertIn('None still have no matching run', content)
        self.assertNotIn('0 orphan', content)
