"""Unit tests for the exact-identity system-link matcher, writer and entry point (ALLOC-06).

Covers ``campaign_system_links.find_exact_run()`` / ``attempt_system_link()`` and
``campaign_utils.create_system_link()`` (37.1-01, CONTEXT D-08, D-11..D-15). Targets are always
``NonSiderealTargetFactory`` -- FOMO is exclusively for non-sidereal targets (CLAUDE.md), field
pointing targets included.
"""

import io
from datetime import date, datetime, timedelta
from datetime import timezone as dt_timezone
from itertools import count
from unittest.mock import patch

from django.contrib.auth.models import User
from django.db import IntegrityError
from django.test import TestCase
from django.utils import timezone
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code.campaign_system_links import (
    BASIS_CAMPAIGN,
    BASIS_TARGET,
    OUTCOME_LINKED,
    OUTCOME_NO_MATCH,
    OUTCOME_SKIPPED,
    OUTCOME_WOULD_LINK,
    attempt_system_link,
    find_exact_run,
)
from solsys_code.campaign_utils import create_system_link, write_and_reconcile_campaign_run
from solsys_code.models import CampaignRun, CampaignRunObservation, ObservationRecordDismissal

_PROPOSAL = 'KEY2026B-004'
_run_seq = count(1)
_record_seq = count(1)


class SystemLinkTestBase(TestCase):
    """Shared fixture: two campaigns (TargetLists) and a record owner."""

    @classmethod
    def setUpTestData(cls):
        cls.campaign_one = TargetList.objects.create(name='Campaign One')
        cls.campaign_two = TargetList.objects.create(name='Campaign Two')
        cls.owner = User.objects.create(username='system-link-owner')

    def _make_target(self, *campaigns):
        """A NonSiderealTargetFactory target, added to each given campaign."""
        target = NonSiderealTargetFactory.create()
        for campaign in campaigns:
            campaign.targets.add(target)
        return target

    def _make_run(self, **overrides):
        """An APPROVED, windowed, container-class run of the default proposal.

        The telescope class is what lets the container-run receiver resolve the run. Each
        run gets a distinct telescope_instrument so the natural-key unique constraints
        never collide when several runs share a campaign.
        """
        kwargs = {
            'campaign': None,
            'target': None,
            'approval_status': CampaignRun.ApprovalStatus.APPROVED,
            'proposal_code': _PROPOSAL,
            'telescope_instrument': f'1m0/Sinistro #{next(_run_seq)}',
            'telescope_class': '1m0',
            'source': CampaignRun.Source.LCO_QUEUE,
            'window_start': date(2026, 9, 28),
            'window_end': date(2026, 10, 4),
        }
        kwargs.update(overrides)
        return CampaignRun.objects.create(**kwargs)

    def _make_record(self, target, *, parameters=None, **overrides):
        """A saved LCO ObservationRecord whose parameters window is 2026-09-30 .. 2026-10-01."""
        record_parameters = {
            'proposal': _PROPOSAL,
            'instrument_type': '1M0-SCICAM-SINISTRO',
            'start': '2026-09-30T00:00:00',
            'end': '2026-10-01T00:00:00',
        }
        if parameters is not None:
            record_parameters = parameters
        kwargs = {
            'target': target,
            'user': self.owner,
            'facility': 'LCO',
            'observation_id': f'sys-link-{next(_record_seq)}',
            'status': 'COMPLETED',
            'parameters': record_parameters,
        }
        kwargs.update(overrides)
        return ObservationRecord.objects.create(**kwargs)


class TestTargetFirstMatch(SystemLinkTestBase):
    """D-11 step 1 / D-15: a unique containing run carrying the record's target wins."""

    def test_unique_target_run_links_even_with_other_campaign_runs(self):
        target = self._make_target(self.campaign_one)
        run = self._make_run(campaign=self.campaign_one, target=target)
        self._make_run(campaign=self.campaign_one, target=self._make_target())
        self._make_run(campaign=self.campaign_one, target=self._make_target())
        record = self._make_record(target)

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, run.pk)
        self.assertEqual(match.basis, BASIS_TARGET)

    def test_two_target_runs_are_ambiguous_with_no_fall_through(self):
        target = self._make_target(self.campaign_one)
        self._make_run(target=target, window_start=date(2026, 9, 28), window_end=date(2026, 10, 4))
        self._make_run(target=target, window_start=date(2026, 9, 29), window_end=date(2026, 10, 3))
        # A campaign-unique run that WOULD win step 2 -- it must never be reached.
        self._make_run(campaign=self.campaign_one, target=None)
        record = self._make_record(target)

        self.assertIsNone(find_exact_run(record))


class TestCampaignFallback(SystemLinkTestBase):
    """D-11 step 2 as amended at plan time: proposal unique within the record's campaigns."""

    def test_field_pointing_record_links_by_campaign(self):
        field_target = self._make_target(self.campaign_one)
        moving_target = self._make_target()
        run = self._make_run(campaign=self.campaign_one, target=moving_target)
        record = self._make_record(field_target)

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, run.pk)
        self.assertEqual(match.basis, BASIS_CAMPAIGN)

    def test_two_campaign_runs_with_only_one_containing_is_ambiguous(self):
        """RESEARCH Open Question 1: uniqueness is judged BEFORE the window filter."""
        field_target = self._make_target(self.campaign_one)
        self._make_run(campaign=self.campaign_one)
        self._make_run(campaign=self.campaign_one, window_start=date(2026, 10, 10), window_end=date(2026, 10, 12))
        record = self._make_record(field_target)

        self.assertIsNone(find_exact_run(record))

    def test_only_campaign_run_not_containing_the_window_is_not_linked(self):
        field_target = self._make_target(self.campaign_one)
        self._make_run(campaign=self.campaign_one, window_start=date(2026, 10, 10), window_end=date(2026, 10, 12))
        record = self._make_record(field_target)

        self.assertIsNone(find_exact_run(record))

    def test_own_target_run_outside_the_window_blocks_the_campaign_fallback(self):
        """SC2: the record's own target has a same-proposal run that does not contain it."""
        target = self._make_target(self.campaign_one)
        self._make_run(target=target, window_start=date(2026, 10, 10), window_end=date(2026, 10, 12))
        self._make_run(campaign=self.campaign_one, target=None)
        record = self._make_record(target)

        self.assertIsNone(find_exact_run(record))

    def test_two_campaigns_each_with_one_run_is_ambiguous(self):
        field_target = self._make_target(self.campaign_one, self.campaign_two)
        self._make_run(campaign=self.campaign_one)
        self._make_run(campaign=self.campaign_two)
        record = self._make_record(field_target)

        self.assertIsNone(find_exact_run(record))

    def test_a_target_in_no_campaign_with_no_run_is_not_linked(self):
        self._make_run(campaign=self.campaign_one)
        record = self._make_record(self._make_target())

        self.assertIsNone(find_exact_run(record))


class TestCandidateFiltering(SystemLinkTestBase):
    """D-13 / D-14: only APPROVED runs with a window are candidates, and run_status is ignored."""

    def test_tbd_run_beside_a_windowed_run_does_not_block_a_target_match(self):
        target = self._make_target()
        windowed = self._make_run(target=target)
        self._make_run(target=target, window_start=None, window_end=None)
        record = self._make_record(target)

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, windowed.pk)

    def test_tbd_run_in_the_campaign_does_not_make_the_campaign_set_non_unique(self):
        field_target = self._make_target(self.campaign_one)
        windowed = self._make_run(campaign=self.campaign_one)
        self._make_run(campaign=self.campaign_one, window_start=None, window_end=None)
        record = self._make_record(field_target)

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, windowed.pk)
        self.assertEqual(match.basis, BASIS_CAMPAIGN)

    def test_pending_and_rejected_runs_neither_link_nor_block(self):
        target = self._make_target()
        approved = self._make_run(target=target)
        self._make_run(target=target, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        self._make_run(target=target, approval_status=CampaignRun.ApprovalStatus.REJECTED)
        record = self._make_record(target)

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, approved.pk)

    def test_a_lone_pending_run_is_not_linked(self):
        target = self._make_target()
        self._make_run(target=target, approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW)
        record = self._make_record(target)

        self.assertIsNone(find_exact_run(record))

    def test_run_status_is_ignored(self):
        for run_status in (CampaignRun.RunStatus.CANCELLED, CampaignRun.RunStatus.WEATHER_TECH_FAILURE):
            with self.subTest(run_status=run_status):
                target = self._make_target()
                run = self._make_run(target=target, run_status=run_status)
                record = self._make_record(target)

                match = find_exact_run(record)

                self.assertIsNotNone(match)
                self.assertEqual(match.run.pk, run.pk)

    def test_record_status_is_ignored(self):
        """D-03: a WINDOW_EXPIRED record that matches exactly still links."""
        target = self._make_target()
        run = self._make_run(target=target)
        record = self._make_record(target, status='WINDOW_EXPIRED')

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, run.pk)


class TestWindowContainment(SystemLinkTestBase):
    """D-12: inclusive containment on UTC dates; RESEARCH Pitfall 7."""

    def test_record_straddling_window_start_is_not_linked(self):
        target = self._make_target()
        self._make_run(target=target)
        record = self._make_record(
            target, parameters={'proposal': _PROPOSAL, 'start': '2026-09-27T12:00:00', 'end': '2026-09-29T00:00:00'}
        )

        self.assertIsNone(find_exact_run(record))

    def test_record_straddling_window_end_is_not_linked(self):
        target = self._make_target()
        self._make_run(target=target)
        record = self._make_record(
            target, parameters={'proposal': _PROPOSAL, 'start': '2026-10-03T12:00:00', 'end': '2026-10-05T00:00:00'}
        )

        self.assertIsNone(find_exact_run(record))

    def test_block_on_window_end_date_links_and_window_end_plus_one_does_not(self):
        target = self._make_target()
        run = self._make_run(target=target, window_end=date(2026, 10, 4))
        on_end = datetime(2026, 10, 4, 2, 0, tzinfo=dt_timezone.utc)
        after_end = on_end + timedelta(days=1)

        linked = self._make_record(target, scheduled_start=on_end, scheduled_end=on_end + timedelta(hours=1))
        past = self._make_record(target, scheduled_start=after_end, scheduled_end=after_end + timedelta(hours=1))

        match = find_exact_run(linked)
        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, run.pk)
        self.assertIsNone(find_exact_run(past))

    def test_unreadable_window_is_not_linked(self):
        target = self._make_target()
        self._make_run(target=target)
        record = self._make_record(target, parameters={'proposal': _PROPOSAL})

        self.assertIsNone(find_exact_run(record))


class TestProposalComparison(SystemLinkTestBase):
    """RESEARCH Pitfall 1 and the proposal comparison rule."""

    def test_blank_or_missing_proposal_never_matches_a_blank_code_run(self):
        target = self._make_target()
        self._make_run(target=target, proposal_code='')
        window = {'start': '2026-09-30T00:00:00', 'end': '2026-10-01T00:00:00'}
        for label, parameters in (
            ('empty parameters', {}),
            ('proposal None', {'proposal': None, **window}),
            ('whitespace proposal', {'proposal': '   ', **window}),
            ('missing proposal', dict(window)),
            ('empty proposal', {'proposal': '', **window}),
        ):
            with self.subTest(label):
                record = self._make_record(target, parameters=parameters)
                self.assertIsNone(find_exact_run(record))

    def test_surrounding_whitespace_is_trimmed(self):
        target = self._make_target()
        run = self._make_run(target=target)
        record = self._make_record(
            target,
            parameters={'proposal': f' {_PROPOSAL} ', 'start': '2026-09-30T00:00:00', 'end': '2026-10-01T00:00:00'},
        )

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, run.pk)

    def test_comparison_is_case_sensitive(self):
        target = self._make_target()
        self._make_run(target=target)
        record = self._make_record(
            target,
            parameters={'proposal': _PROPOSAL.lower(), 'start': '2026-09-30T00:00:00', 'end': '2026-10-01T00:00:00'},
        )

        self.assertIsNone(find_exact_run(record))

    def test_a_run_of_another_proposal_is_not_linked(self):
        target = self._make_target()
        self._make_run(target=target, proposal_code='OTHER2026B-001')
        record = self._make_record(target)

        self.assertIsNone(find_exact_run(record))


class TestHumanOutranksMachine(SystemLinkTestBase):
    """SC3 and the never-overwrite rule."""

    def test_dismissed_unique_winner_is_not_linked(self):
        target = self._make_target()
        run = self._make_run(target=target)
        record = self._make_record(target)
        ObservationRecordDismissal.objects.create(observation_record=record, run=run, dismissed_by=self.owner)

        self.assertIsNone(find_exact_run(record))

    def test_dismissed_winner_is_never_re_routed_to_another_run(self):
        target = self._make_target(self.campaign_one)
        winner = self._make_run(target=target)
        # A second containing run that does not carry the record's target.
        self._make_run(campaign=self.campaign_one, target=None)
        record = self._make_record(target)
        ObservationRecordDismissal.objects.create(observation_record=record, run=winner, dismissed_by=self.owner)

        self.assertIsNone(find_exact_run(record))

    def test_dismissal_of_another_run_does_not_veto(self):
        target = self._make_target()
        winner = self._make_run(target=target)
        other = self._make_run(target=self._make_target(), window_start=date(2026, 1, 1), window_end=date(2026, 1, 2))
        record = self._make_record(target)
        ObservationRecordDismissal.objects.create(observation_record=record, run=other, dismissed_by=self.owner)

        match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, winner.pk)

    def test_record_with_a_staff_link_is_never_touched(self):
        target = self._make_target()
        self._make_run(target=target)
        staff_run = self._make_run(
            target=self._make_target(), window_start=date(2026, 1, 1), window_end=date(2026, 1, 2)
        )
        record = self._make_record(target)
        confirmed_at = timezone.now() - timedelta(days=3)
        staff_link = CampaignRunObservation.objects.create(
            run=staff_run, observation_record=record, confirmed_by=self.owner, confirmed_at=confirmed_at
        )

        self.assertIsNone(find_exact_run(record))
        self.assertFalse(create_system_link(record, self._make_run(target=target)))

        staff_link.refresh_from_db()
        self.assertEqual(staff_link.run_id, staff_run.pk)
        self.assertEqual(staff_link.confirmed_by_id, self.owner.pk)
        self.assertEqual(staff_link.confirmed_at, confirmed_at)
        self.assertEqual(CampaignRunObservation.objects.filter(observation_record=record).count(), 1)


class TestUnsavedRecord(SystemLinkTestBase):
    def test_unsaved_record_matches_without_consulting_links_or_dismissals(self):
        target = self._make_target()
        run = self._make_run(target=target)
        record = ObservationRecord(
            target=target,
            facility='LCO',
            observation_id='not-yet-saved',
            status='PENDING',
            parameters={'proposal': _PROPOSAL, 'start': '2026-09-30T00:00:00', 'end': '2026-10-01T00:00:00'},
        )

        # One query only: the candidate run lookup. No link or dismissal query for pk None.
        with self.assertNumQueries(1):
            match = find_exact_run(record)

        self.assertIsNotNone(match)
        self.assertEqual(match.run.pk, run.pk)
        self.assertEqual(match.basis, BASIS_TARGET)


class TestCreateSystemLink(SystemLinkTestBase):
    """The one system-link writer."""

    def test_creates_a_system_link_and_a_second_call_leaves_it_untouched(self):
        target = self._make_target()
        run = self._make_run(target=target)
        record = self._make_record(target)

        self.assertTrue(create_system_link(record, run))

        link = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link.run_id, run.pk)
        self.assertIsNone(link.confirmed_by)
        self.assertIsNotNone(link.confirmed_at)
        first_confirmed_at = link.confirmed_at

        self.assertFalse(create_system_link(record, run))

        link.refresh_from_db()
        self.assertEqual(link.confirmed_at, first_confirmed_at)
        self.assertEqual(CampaignRunObservation.objects.filter(observation_record=record).count(), 1)

    def test_write_and_reconcile_links_through_the_same_helper(self):
        target = self._make_target()
        run = self._make_run(target=target)
        record = self._make_record(target)

        with patch('solsys_code.campaign_utils.create_system_link', wraps=create_system_link) as spy:
            result = write_and_reconcile_campaign_run({'pk': run.pk}, {}, observation_record=record)

        spy.assert_called_once_with(record, run)
        self.assertEqual(result.action, 'unchanged')
        link = CampaignRunObservation.objects.get(observation_record=record)
        self.assertEqual(link.run_id, run.pk)
        self.assertIsNone(link.confirmed_by)


class TestAttemptSystemLink(SystemLinkTestBase):
    """The ingest entry point: outcomes, output lines and failure isolation (D-08)."""

    def setUp(self):
        self.target = self._make_target()
        self.run_obj = self._make_run(target=self.target)
        self.record = self._make_record(self.target)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def _attempt(self, *, dry_run):
        return attempt_system_link(self.record, dry_run=dry_run, stdout=self.stdout, stderr=self.stderr)

    def test_dry_run_reports_a_would_link_line_and_writes_nothing(self):
        self.assertEqual(self._attempt(dry_run=True), OUTCOME_WOULD_LINK)

        self.assertEqual(
            self.stdout.getvalue(),
            f'Would system-link ObservationRecord observation_id={self.record.observation_id!r} '
            f'to CampaignRun #{self.run_obj.pk} ({BASIS_TARGET}).\n',
        )
        self.assertEqual(CampaignRunObservation.objects.count(), 0)
        self.assertEqual(self.stderr.getvalue(), '')

    def test_real_run_links_and_names_the_basis(self):
        self.assertEqual(self._attempt(dry_run=False), OUTCOME_LINKED)

        self.assertEqual(
            self.stdout.getvalue(),
            f'System-linked ObservationRecord observation_id={self.record.observation_id!r} '
            f'to CampaignRun #{self.run_obj.pk} ({BASIS_TARGET}).\n',
        )
        self.assertEqual(CampaignRunObservation.objects.filter(observation_record=self.record).count(), 1)

    def test_a_second_attempt_finds_nothing_to_do(self):
        self._attempt(dry_run=False)
        self.stdout.truncate(0)
        self.stdout.seek(0)

        self.assertEqual(self._attempt(dry_run=False), OUTCOME_NO_MATCH)
        self.assertEqual(self.stdout.getvalue(), '')

    def test_a_failing_write_is_skipped_and_never_leaks_the_message(self):
        with patch('solsys_code.campaign_system_links.create_system_link', side_effect=IntegrityError('secret detail')):
            outcome = self._attempt(dry_run=False)

        self.assertEqual(outcome, OUTCOME_SKIPPED)
        self.assertIn('IntegrityError', self.stderr.getvalue())
        self.assertIn(f'CampaignRun #{self.run_obj.pk}', self.stderr.getvalue())
        self.assertNotIn('secret detail', self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), '')

    def test_a_failing_match_is_skipped_and_never_leaks_the_message(self):
        with patch('solsys_code.campaign_system_links.find_exact_run', side_effect=RuntimeError('secret detail')):
            outcome = self._attempt(dry_run=False)

        self.assertEqual(outcome, OUTCOME_SKIPPED)
        self.assertIn('RuntimeError', self.stderr.getvalue())
        self.assertIn(repr(self.record.observation_id), self.stderr.getvalue())
        self.assertNotIn('secret detail', self.stderr.getvalue())

    def test_a_record_with_no_parameters_never_raises(self):
        self.record.parameters = None

        outcome = self._attempt(dry_run=False)

        self.assertIn(outcome, (OUTCOME_SKIPPED, OUTCOME_NO_MATCH))
        self.assertEqual(CampaignRunObservation.objects.count(), 0)
