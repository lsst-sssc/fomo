"""Tests for the ProposalTimeAllocation model and CampaignRun.proposal_code (Phase 37 D-07).

Task 2: the model's create-or-update behaviour on its unique key, the strip-on-save
discipline mirroring WatchedProposal, and CampaignRun.proposal_code's default-blank field.
Task 3: the solsys_code.proposal_allocation fetch/store/estimate module.
"""

from unittest.mock import MagicMock, patch

import requests
from django import forms
from django.db import IntegrityError, connection, transaction
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
from tom_common.exceptions import ImproperCredentialsException

from solsys_code import proposal_allocation as pa
from solsys_code.models import CampaignRun, ProposalTimeAllocation, WatchedProposal
from solsys_code.solsys_code_observatory.models import Observatory


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
    """The union of active WatchedProposal codes and non-blank CampaignRun.proposal_code.

    The run half is limited to runs that can hold an LCO/SOAR portal proposal (F7, quick task 261002-gev).
    """

    def test_union_of_watched_and_run_codes_sorted_and_deduped(self):
        WatchedProposal.objects.create(proposal_code='BBB-2026-002')
        WatchedProposal.objects.create(proposal_code='CCC-2026-003', is_active=False)
        CampaignRun.objects.create(
            telescope_instrument='LCO-1m-Sinistro',
            proposal_code='AAA-2026-001',
            source=CampaignRun.Source.LCO_QUEUE,
        )
        CampaignRun.objects.create(
            telescope_instrument='LCO-1m-Sinistro',
            proposal_code='BBB-2026-002',
            source=CampaignRun.Source.LCO_QUEUE,
        )
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

    def test_path_traversal_like_code_is_rejected_before_any_request(self):
        """WR-07 (37-REVIEW.md): a proposal_code containing '..' would redirect this
        credentialed request to a different portal endpoint once urljoin()/the server
        normalises it -- must be rejected before make_request() is ever called."""
        with patch('solsys_code.proposal_allocation.make_request') as mock_request:
            with self.assertRaises(pa.PortalUnavailable):
                pa.fetch_proposal_allocations('../requestgroups', _mock_facility())
        mock_request.assert_not_called()

    def test_code_with_query_or_fragment_characters_is_rejected(self):
        with self.assertRaises(pa.PortalUnavailable):
            pa.fetch_proposal_allocations('ABC?x=1', _mock_facility())
        with self.assertRaises(pa.PortalUnavailable):
            pa.fetch_proposal_allocations('ABC#frag', _mock_facility())

    def test_over_long_code_is_rejected(self):
        """CampaignRun.proposal_code has max_length=100 -- a longer value would raise
        DataError on PostgreSQL if it ever reached a write; reject it here instead."""
        with self.assertRaises(pa.PortalUnavailable):
            pa.fetch_proposal_allocations('A' * 101, _mock_facility())

    def test_blank_code_is_rejected(self):
        with self.assertRaises(pa.PortalUnavailable):
            pa.fetch_proposal_allocations('', _mock_facility())

    def test_valid_code_reaches_a_properly_quoted_request_url(self):
        response = MagicMock()
        response.json.return_value = {'timeallocation_set': []}
        with patch('solsys_code.proposal_allocation.make_request', return_value=response) as mock_request:
            pa.fetch_proposal_allocations('UTX2026A-002', _mock_facility())
        called_url = mock_request.call_args[0][1]
        self.assertEqual(called_url, 'https://observe.lco.global/api/proposals/UTX2026A-002/')

    def test_an_eso_style_code_with_a_period_is_accepted_and_quoted(self):
        """Mirrors ProposalCodeDefaultTests.test_explicit_value_persists_and_reloads's own
        real-world fixture ('0110.C-0234') -- the charset must not be so strict it rejects
        a real, already-observed proposal code shape."""
        response = MagicMock()
        response.json.return_value = {'timeallocation_set': []}
        with patch('solsys_code.proposal_allocation.make_request', return_value=response) as mock_request:
            pa.fetch_proposal_allocations('0110.C-0234', _mock_facility())
        called_url = mock_request.call_args[0][1]
        self.assertEqual(called_url, 'https://observe.lco.global/api/proposals/0110.C-0234/')


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

    def test_a_key_absent_from_a_later_response_is_pruned(self):
        """WR-08 (37-REVIEW.md): a (semester, instrument_type, allocation_type) row that
        disappears from the portal's timeallocation_set must be deleted, not left to
        accumulate forever."""
        rows_v1 = [
            {'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 40.0, 'std_time_used': 10.0},
            {'semester': '2026B', 'instrument_type': 'X', 'std_allocation': 30.0, 'std_time_used': 5.0},
        ]
        pa.store_proposal_allocations('UTX2026A-002', rows_v1)
        self.assertEqual(ProposalTimeAllocation.objects.filter(proposal_code='UTX2026A-002').count(), 2)

        # 2026A retires from the response; only 2026B remains.
        rows_v2 = [{'semester': '2026B', 'instrument_type': 'X', 'std_allocation': 30.0, 'std_time_used': 5.0}]
        written = pa.store_proposal_allocations('UTX2026A-002', rows_v2)

        self.assertEqual(written, 1)
        remaining = ProposalTimeAllocation.objects.filter(proposal_code='UTX2026A-002')
        self.assertEqual(remaining.count(), 1)
        self.assertEqual(remaining.get().semester, '2026B')

    def test_prune_never_touches_a_different_proposal_codes_rows(self):
        pa.store_proposal_allocations(
            'UTX2026A-002',
            [{'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 40.0, 'std_time_used': 10.0}],
        )
        pa.store_proposal_allocations(
            'OTHER-2026A-099',
            [{'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 5.0, 'std_time_used': 1.0}],
        )
        # Re-fetch UTX2026A-002 with an EMPTY response -- prunes all of its own rows.
        pa.store_proposal_allocations('UTX2026A-002', [])
        self.assertEqual(ProposalTimeAllocation.objects.filter(proposal_code='UTX2026A-002').count(), 0)
        self.assertEqual(ProposalTimeAllocation.objects.filter(proposal_code='OTHER-2026A-099').count(), 1)


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

    def test_only_the_most_recent_semester_is_summed(self):
        """WR-08 (37-REVIEW.md) regression: a finished 2026A carrying 40 unused hours plus
        a fresh 2026B carrying 100 hours must report ONLY 2026B's unused hours, not both
        summed -- the finished semester's leftover time is no longer "currently relevant
        wasted time"."""
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            semester='2026A',
            allocation_type='std',
            allocated_hours=40.0,
            used_hours=0.0,
            fetched_at=timezone.now(),
        )
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            semester='2026B',
            allocation_type='std',
            allocated_hours=100.0,
            used_hours=30.0,
            fetched_at=timezone.now(),
        )
        self.assertEqual(pa.unused_hours_for('UTX2026A-002'), 70.0)

    def test_explicit_semester_overrides_the_most_recent_default(self):
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            semester='2026A',
            allocation_type='std',
            allocated_hours=40.0,
            used_hours=0.0,
            fetched_at=timezone.now(),
        )
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            semester='2026B',
            allocation_type='std',
            allocated_hours=100.0,
            used_hours=30.0,
            fetched_at=timezone.now(),
        )
        self.assertEqual(pa.unused_hours_for('UTX2026A-002', semester='2026A'), 40.0)

    def test_no_rows_in_an_explicitly_requested_semester_is_none(self):
        ProposalTimeAllocation.objects.create(
            proposal_code='UTX2026A-002',
            semester='2026A',
            allocation_type='std',
            allocated_hours=40.0,
            used_hours=0.0,
            fetched_at=timezone.now(),
        )
        self.assertIsNone(pa.unused_hours_for('UTX2026A-002', semester='2026C'))


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
            attempted, rows_written, failed, first_exception, not_fetchable = pa.refresh_all(_mock_facility())

        self.assertEqual(attempted, 2)
        self.assertEqual(failed, 1)
        self.assertEqual(rows_written, 1)
        self.assertEqual(first_exception, 'Timeout')
        self.assertEqual(not_fetchable, 0)

    def test_empty_code_list_is_a_no_op(self):
        attempted, rows_written, failed, first_exception, not_fetchable = pa.refresh_all(_mock_facility())
        self.assertEqual((attempted, rows_written, failed, first_exception, not_fetchable), (0, 0, 0, None, 0))


class ProposalCodeFetchabilityTests(TestCase):
    """Pins F7 / quick task 261002-gev: only codes that can be LCO/SOAR portal proposals are fetched.

    A proposal code on a run is sent to the LCO Observation Portal only when the run came from the
    LCO/SOAR queue or sits at one of the LCO/SOAR observatories ``campaign_attribution`` knows. Every other
    run code (e.g. an ESO code on a classical NTT line) is counted as not fetchable, never requested.
    """

    @classmethod
    def setUpTestData(cls):
        def _obs(obscode, name, short_name, lat, lon, altitude, timezone_name):
            return Observatory.objects.create(
                obscode=obscode,
                name=name,
                short_name=short_name,
                lat=lat,
                lon=lon,
                altitude=altitude,
                timezone=timezone_name,
                observations_type=Observatory.OPTICAL_OBSTYPE,
            )

        cls.la_silla = _obs('809', 'La Silla', 'La Silla', -29.2563, -70.7380, 2400.0, 'America/Santiago')
        cls.fts = _obs('E10', 'Faulkes Telescope South', 'FTS', -31.2733, 149.0706, 1149.0, 'Australia/Sydney')
        cls.ftn = _obs('F65', 'Faulkes Telescope North', 'FTN', 20.7073, -156.2575, 3055.0, 'Pacific/Honolulu')
        cls.soar = _obs('I33', 'SOAR', 'SOAR', -30.2379, -70.7337, 2738.0, 'America/Santiago')

    def _run(self, code, *, source, site=None, **extra):
        return CampaignRun.objects.create(
            telescope_instrument=f'F7 fixture {source} {code}',
            proposal_code=code,
            source=source,
            site=site,
            **extra,
        )

    def _refresh(self):
        ok_response = MagicMock()
        ok_response.json.return_value = {
            'timeallocation_set': [
                {'semester': '2026A', 'instrument_type': 'X', 'std_allocation': 10.0, 'std_time_used': 2.0}
            ]
        }
        with patch('solsys_code.proposal_allocation.make_request', return_value=ok_response) as mock_request:
            result = pa.refresh_all(_mock_facility())
        urls = [c.args[1] for c in mock_request.call_args_list]
        return result, urls

    def test_eso_code_on_a_classical_run_at_la_silla_is_never_sent_to_the_portal(self):
        self._run('117.2A2N.001', source=CampaignRun.Source.CLASSICAL_FILE, site=self.la_silla)
        self._run('LCO2026A-003', source=CampaignRun.Source.CLASSICAL_FILE, site=self.fts)

        (attempted, rows_written, failed, first_exception, not_fetchable), urls = self._refresh()

        self.assertEqual((attempted, rows_written, failed, first_exception, not_fetchable), (1, 1, 0, None, 1))
        self.assertFalse(any('117.2A2N.001' in url for url in urls))
        self.assertEqual(sum('LCO2026A-003' in url for url in urls), 1)
        self.assertEqual(pa.proposal_codes_to_fetch(), ['LCO2026A-003'])
        self.assertEqual(pa.proposal_codes_not_fetchable(), ['117.2A2N.001'])

    def test_lco_code_on_a_run_at_each_known_lco_or_soar_site_is_fetched(self):
        self._run('LCO2026A-003', source=CampaignRun.Source.CLASSICAL_FILE, site=self.fts)
        self._run('LCO2026A-004', source=CampaignRun.Source.LEGACY, site=self.ftn)
        self._run('SOAR2026A-005', source=CampaignRun.Source.WEB, site=self.soar)

        self.assertEqual(pa.proposal_codes_to_fetch(), ['LCO2026A-003', 'LCO2026A-004', 'SOAR2026A-005'])
        self.assertEqual(pa.proposal_codes_not_fetchable(), [])

        (attempted, _rows, _failed, _first, not_fetchable), _urls = self._refresh()
        self.assertEqual(attempted, 3)
        self.assertEqual(not_fetchable, 0)

    def test_queue_sourced_run_codes_are_fetched_with_no_site(self):
        self._run(
            'KEY2026B-004',
            source=CampaignRun.Source.LCO_QUEUE,
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )
        self._run('SOAR2026B-001', source=CampaignRun.Source.SOAR_QUEUE)

        fetched = pa.proposal_codes_to_fetch()
        self.assertIn('KEY2026B-004', fetched)
        self.assertIn('SOAR2026B-001', fetched)

    def test_active_watched_proposal_is_always_fetched(self):
        WatchedProposal.objects.create(proposal_code='LCO2026B-010')
        WatchedProposal.objects.create(proposal_code='SHARED2026A-001')
        WatchedProposal.objects.create(proposal_code='OFF2026A-002', is_active=False)
        self._run('SHARED2026A-001', source=CampaignRun.Source.CLASSICAL_FILE, site=self.la_silla)

        self.assertEqual(pa.proposal_codes_to_fetch(), ['LCO2026B-010', 'SHARED2026A-001'])
        self.assertEqual(pa.proposal_codes_not_fetchable(), [])

    def test_excluded_code_stays_not_yet_known_after_a_refresh(self):
        self._run('117.2A2N.001', source=CampaignRun.Source.CLASSICAL_FILE, site=self.la_silla)

        self._refresh()

        # D-07 / TALLY-01: an unfetched code reads "not yet known" (None), never zero.
        self.assertFalse(ProposalTimeAllocation.objects.filter(proposal_code='117.2A2N.001').exists())
        self.assertIsNone(pa.unused_hours_for('117.2A2N.001'))
        self.assertIsNone(pa.estimated_unused_nights('117.2A2N.001'))

    def test_eso_gemini_and_site_less_runs_are_not_fetchable(self):
        self._run('0110.C-0234', source=CampaignRun.Source.ESO_QUEUE)
        self._run('GS-2026A-FT-115', source=CampaignRun.Source.GEMINI_QUEUE)
        self._run('NOSITE2026A-001', source=CampaignRun.Source.CLASSICAL_FILE)
        self._run('LEG2026A-002', source=CampaignRun.Source.LEGACY, site=self.la_silla)

        codes = ['0110.C-0234', 'GS-2026A-FT-115', 'NOSITE2026A-001', 'LEG2026A-002']
        fetched = pa.proposal_codes_to_fetch()
        for code in codes:
            self.assertNotIn(code, fetched)
        self.assertEqual(pa.proposal_codes_not_fetchable(), sorted(codes))

    def test_code_selection_never_reads_a_contact_field(self):
        """T-37-06: choosing codes loads no contact column."""
        self._run(
            'KEY2026B-004',
            source=CampaignRun.Source.LCO_QUEUE,
            contact_person='Secret Person',
            contact_email='secret@example.org',
        )
        self._run(
            '117.2A2N.001',
            source=CampaignRun.Source.CLASSICAL_FILE,
            site=self.la_silla,
            contact_person='Secret Person',
            contact_email='secret@example.org',
        )

        with CaptureQueriesContext(connection) as ctx:
            pa.proposal_codes_to_fetch()
            pa.proposal_codes_not_fetchable()

        statements = [q['sql'].lower() for q in ctx.captured_queries]
        self.assertTrue(any('campaignrun' in sql for sql in statements))
        self.assertFalse(any('contact' in sql for sql in statements))
