"""Tests for the ProposalTimeAllocation model and CampaignRun.proposal_code (Phase 37 D-07).

Task 2: the model's create-or-update behaviour on its unique key, the strip-on-save
discipline mirroring WatchedProposal, and CampaignRun.proposal_code's default-blank field.
Task 3: the solsys_code.proposal_allocation fetch/store/estimate module.
"""

from unittest.mock import MagicMock, patch

import requests
from django import forms
from django.db import IntegrityError, transaction
from django.test import TestCase
from django.utils import timezone
from tom_common.exceptions import ImproperCredentialsException

from solsys_code import proposal_allocation as pa
from solsys_code.models import CampaignRun, ProposalTimeAllocation, WatchedProposal


class ProposalTimeAllocationModelTests(TestCase):
    """Model-level behaviour: create-or-update on the unique key, and the strip-on-save rule."""

    def test_update_or_create_updates_existing_row_rather_than_duplicating(self):
        """A second call with the same (proposal_code, semester, instrument_type,
        allocation_type) key updates the matching row in place instead of creating a
        duplicate."""
        now = timezone.now()
        key = dict(
            proposal_code='UTX2026A-002',
            semester='2026A',
            instrument_type='1M0-SCICAM-SINISTRO',
            allocation_type='std',
        )
        ProposalTimeAllocation.objects.update_or_create(
            **key, defaults={'allocated_hours': 40.0, 'used_hours': 10.0, 'fetched_at': now}
        )
        ProposalTimeAllocation.objects.update_or_create(
            **key, defaults={'allocated_hours': 40.0, 'used_hours': 15.0, 'fetched_at': now}
        )

        self.assertEqual(ProposalTimeAllocation.objects.count(), 1)
        row = ProposalTimeAllocation.objects.get()
        self.assertEqual(row.used_hours, 15.0)

    def test_unique_constraint_rejects_a_duplicate_key(self):
        """The DB-level UniqueConstraint (not just app logic) enforces the key is unique --
        two concurrent creates for the same key cannot both succeed."""
        now = timezone.now()
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            semester='2026A',
            instrument_type='1M0-SCICAM-SINISTRO',
            allocation_type='std',
            fetched_at=now,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ProposalTimeAllocation.objects.create(
                    proposal_code='UTX2026A-002',
                    semester='2026A',
                    instrument_type='1M0-SCICAM-SINISTRO',
                    allocation_type='std',
                    fetched_at=now,
                )

    def test_strip_on_save(self):
        """A pasted code with leading/trailing whitespace is stripped before it is stored,
        mirroring WatchedProposal.save()."""
        row = ProposalTimeAllocation.objects.create(
            proposal_code='  UTX2026A-002  ',
            allocation_type='std',
            fetched_at=timezone.now(),
        )
        row.refresh_from_db()
        self.assertEqual(row.proposal_code, 'UTX2026A-002')

    def test_default_semester_and_instrument_type_are_blank_strings(self):
        row = ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            allocation_type='std',
            fetched_at=timezone.now(),
        )
        self.assertEqual(row.semester, '')
        self.assertEqual(row.instrument_type, '')


class CampaignRunProposalCodeFieldTests(TestCase):
    """CampaignRun.proposal_code defaults to the empty string and is blank-allowed."""

    def test_default_is_blank_string(self):
        run = CampaignRun.objects.create(telescope_instrument='LCO-1m-Sinistro')
        self.assertEqual(run.proposal_code, '')

    def test_explicit_value_persists_and_reloads(self):
        run = CampaignRun.objects.create(telescope_instrument='LCO-1m-Sinistro', proposal_code='0110.C-0234')
        reloaded = CampaignRun.objects.get(pk=run.pk)
        self.assertEqual(reloaded.proposal_code, '0110.C-0234')


def _mock_facility() -> MagicMock:
    facility = MagicMock()
    facility.facility_settings.get_setting.return_value = 'https://observe.lco.global'
    facility._portal_headers.return_value = {}
    return facility


class ProposalCodesToFetchTests(TestCase):
    """The union of active WatchedProposal codes and non-blank CampaignRun.proposal_code."""

    def test_union_of_watched_and_run_codes_sorted_and_deduped(self):
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        WatchedProposal.objects.create(proposal_code='CCC-2026-003', is_active=False)
        CampaignRun.objects.create(telescope_instrument='LCO-1m-Sinistro', proposal_code='AAA-2026-001')
        CampaignRun.objects.create(telescope_instrument='LCO-1m-Sinistro', proposal_code='BBB-2026-002')
        CampaignRun.objects.create(telescope_instrument='LCO-1m-Sinistro', proposal_code='')

        self.assertEqual(pa.proposal_codes_to_fetch(), ['AAA-2026-001', 'BBB-2026-002'])

    def test_no_codes_returns_empty_list(self):
        self.assertEqual(pa.proposal_codes_to_fetch(), [])


class FetchProposalAllocationsTests(TestCase):
    """fetch_proposal_allocations()'s success and failure-mode contract."""

    def test_success_returns_timeallocation_set(self):
        response = MagicMock()
        response.json.return_value = {'id': 'UTX2026A-002', 'timeallocation_set': [{'semester': '2026A'}]}
        with patch('solsys_code.proposal_allocation.make_request', return_value=response):
            result = pa.fetch_proposal_allocations('UTX2026A-002', _mock_facility())
        self.assertEqual(result, [{'semester': '2026A'}])

    def test_network_error_raises_portal_unavailable_with_class_name_only(self):
        with patch('solsys_code.proposal_allocation.make_request', side_effect=requests.exceptions.Timeout('boom')):
            with self.assertRaises(pa.PortalUnavailable) as ctx:
                pa.fetch_proposal_allocations('UTX2026A-002', _mock_facility())
        self.assertEqual(str(ctx.exception), 'Timeout')

    def test_credential_error_leaks_nothing(self):
        leak_marker = 'SECRET_API_KEY_LEAK_BODY'
        with patch(
            'solsys_code.proposal_allocation.make_request',
            side_effect=ImproperCredentialsException(f'OCS: {leak_marker}'),
        ):
            with self.assertRaises(pa.PortalUnavailable) as ctx:
                pa.fetch_proposal_allocations('UTX2026A-002', _mock_facility())
        self.assertNotIn(leak_marker, str(ctx.exception))
        self.assertEqual(str(ctx.exception), 'ImproperCredentialsException')

    def test_validation_error_leaks_nothing(self):
        leak_marker = 'SECRET_API_KEY_LEAK_BODY'
        with patch(
            'solsys_code.proposal_allocation.make_request',
            side_effect=forms.ValidationError(f'OCS: {leak_marker}'),
        ):
            with self.assertRaises(pa.PortalUnavailable) as ctx:
                pa.fetch_proposal_allocations('UTX2026A-002', _mock_facility())
        self.assertNotIn(leak_marker, str(ctx.exception))

    def test_non_json_body_raises_portal_unavailable(self):
        response = MagicMock()
        response.json.side_effect = ValueError('not json')
        with patch('solsys_code.proposal_allocation.make_request', return_value=response):
            with self.assertRaises(pa.PortalUnavailable):
                pa.fetch_proposal_allocations('UTX2026A-002', _mock_facility())

    def test_malformed_body_without_timeallocation_set_raises(self):
        response = MagicMock()
        response.json.return_value = {'id': 'UTX2026A-002'}
        with patch('solsys_code.proposal_allocation.make_request', return_value=response):
            with self.assertRaises(pa.PortalUnavailable):
                pa.fetch_proposal_allocations('UTX2026A-002', _mock_facility())


class StoreProposalAllocationsTests(TestCase):
    """store_proposal_allocations()'s per-allocation-type create-or-update behaviour."""

    def test_stores_one_row_per_present_allocation_type(self):
        rows = [
            {
                'semester': '2026A',
                'instrument_type': '1M0-SCICAM-SINISTRO',
                'std_allocation': 40.0,
                'std_time_used': 10.0,
                'rr_allocation': 5.0,
                'rr_time_used': 1.0,
                'tc_allocation': 10.0,
                'tc_time_used': 4.11,
                'realtime_allocation': 2.0,
                'realtime_time_used': 0.0,
            }
        ]
        written = pa.store_proposal_allocations('UTX2026A-002', rows)
        self.assertEqual(written, 4)
        self.assertEqual(ProposalTimeAllocation.objects.filter(proposal_code='UTX2026A-002').count(), 4)
        types_stored = set(
            ProposalTimeAllocation.objects.filter(proposal_code='UTX2026A-002').values_list(
                'allocation_type', flat=True
            )
        )
        self.assertEqual(types_stored, {'std', 'rr', 'tc', 'realtime'})

    def test_missing_pair_is_not_stored(self):
        rows = [{'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 40.0, 'std_time_used': 10.0}]
        written = pa.store_proposal_allocations('UTX2026A-002', rows)
        self.assertEqual(written, 1)
        self.assertEqual(ProposalTimeAllocation.objects.count(), 1)

    def test_second_call_with_same_key_updates_rather_than_duplicates(self):
        rows_v1 = [{'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 40.0, 'std_time_used': 10.0}]
        rows_v2 = [{'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 40.0, 'std_time_used': 15.0}]
        pa.store_proposal_allocations('UTX2026A-002', rows_v1)
        pa.store_proposal_allocations('UTX2026A-002', rows_v2)

        self.assertEqual(ProposalTimeAllocation.objects.count(), 1)
        self.assertEqual(ProposalTimeAllocation.objects.get().used_hours, 15.0)


class UnusedHoursForTests(TestCase):
    """unused_hours_for()'s summation-over-ESTIMATE_ALLOCATION_TYPES, floored-at-zero rule."""

    def test_none_when_no_stored_rows(self):
        self.assertIsNone(pa.unused_hours_for('NO-SUCH-PROPOSAL-CODE'))

    def test_sums_only_std_rows(self):
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            allocation_type='std',
            allocated_hours=40.0,
            used_hours=10.0,
            fetched_at=timezone.now(),
        )
        # A stored tc row must NOT be summed into the estimate (Task 1 checkpoint decision).
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            allocation_type='tc',
            allocated_hours=10.0,
            used_hours=4.11,
            fetched_at=timezone.now(),
        )
        self.assertEqual(pa.unused_hours_for('UTX2026A-002'), 30.0)

    def test_used_more_than_allocated_floors_at_zero(self):
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            allocation_type='std',
            allocated_hours=10.0,
            used_hours=25.0,
            fetched_at=timezone.now(),
        )
        self.assertEqual(pa.unused_hours_for('UTX2026A-002'), 0.0)


class EstimatedUnusedNightsTests(TestCase):
    """estimated_unused_nights()'s divide-and-round contract."""

    def test_none_when_never_fetched(self):
        self.assertIsNone(pa.estimated_unused_nights('NO-SUCH-PROPOSAL-CODE'))

    def test_25_hours_rounds_to_3_nights(self):
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            allocation_type='std',
            allocated_hours=25.0,
            used_hours=0.0,
            fetched_at=timezone.now(),
        )
        self.assertEqual(pa.estimated_unused_nights('UTX2026A-002'), 3)

    def test_zero_unused_hours_is_zero_nights_not_none(self):
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            allocation_type='std',
            allocated_hours=10.0,
            used_hours=25.0,
            fetched_at=timezone.now(),
        )
        self.assertEqual(pa.estimated_unused_nights('UTX2026A-002'), 0)


class RefreshAllTests(TestCase):
    """refresh_all()'s per-proposal isolation and counters."""

    def test_isolates_one_failing_proposal_from_the_rest(self):
        WatchedProposal.objects.create(proposal_code='AAA-2026-001')
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')

        ok_response = MagicMock()
        ok_response.json.return_value = {
            'timeallocation_set': [
                {'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 10.0, 'std_time_used': 2.0}
            ]
        }

        def _side_effect(*args, **kwargs):
            url = args[1] if len(args) > 1 else kwargs.get('url', '')
            if 'AAA' in url:
                raise requests.exceptions.Timeout('boom')
            return ok_response

        with patch('solsys_code.proposal_allocation.make_request', side_effect=_side_effect):
            attempted, rows_written, failed, first_exception = pa.refresh_all(_mock_facility())

        self.assertEqual(attempted, 2)
        self.assertEqual(failed, 1)
        self.assertEqual(rows_written, 1)
        self.assertEqual(first_exception, 'Timeout')

    def test_empty_code_list_is_a_no_op(self):
        attempted, rows_written, failed, first_exception = pa.refresh_all(_mock_facility())
        self.assertEqual((attempted, rows_written, failed, first_exception), (0, 0, 0, None))
