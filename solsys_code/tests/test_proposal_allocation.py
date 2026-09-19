"""Tests for the ProposalTimeAllocation model and CampaignRun.proposal_code (Phase 37 D-07).

Task 2 scope only: the model's create-or-update behaviour on its unique key, the strip-on-save
discipline mirroring WatchedProposal, and CampaignRun.proposal_code's default-blank field.
Task 3 adds the fetch/estimate module tests to this same file.
"""

from django.db import IntegrityError, transaction
from django.test import TestCase
from django.utils import timezone

from solsys_code.models import CampaignRun, ProposalTimeAllocation


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
