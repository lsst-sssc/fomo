"""Tests for solsys_code.models.WatchedProposal (36-CONTEXT.md D-06)."""

from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.test import TestCase

from solsys_code.models import WatchedProposal


class TestWatchedProposalModel(TestCase):
    def test_defaults(self):
        row = WatchedProposal.objects.create(proposal_code='KEY2026B-004')
        row.refresh_from_db()
        self.assertTrue(row.is_active)
        self.assertEqual(row.target_list_name, '')
        self.assertIsNone(row.attributed_to)
        self.assertIsNone(row.last_run_at)
        self.assertEqual(row.last_run_summary, '')

    def test_duplicate_code_rejected(self):
        WatchedProposal.objects.create(proposal_code='KEY2026B-004')

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                WatchedProposal.objects.create(proposal_code='KEY2026B-004')

    def test_code_is_stripped_on_save(self):
        row = WatchedProposal.objects.create(proposal_code='  KEY2026B-004  ')
        row.refresh_from_db()
        self.assertEqual(row.proposal_code, 'KEY2026B-004')

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                WatchedProposal.objects.create(proposal_code=' KEY2026B-004')

    def test_code_comparison_is_case_sensitive(self):
        WatchedProposal.objects.create(proposal_code='KEY2026B-004')
        WatchedProposal.objects.create(proposal_code='key2026b-004')

        self.assertEqual(WatchedProposal.objects.count(), 2)

    def test_attributed_user_deletion_keeps_the_row(self):
        user = User.objects.create_user(username='attributee')
        row = WatchedProposal.objects.create(proposal_code='KEY2026B-004', attributed_to=user)

        user.delete()

        row.refresh_from_db()
        self.assertIsNone(row.attributed_to)

    def test_str(self):
        row = WatchedProposal.objects.create(proposal_code='KEY2026B-004')

        text = str(row)

        self.assertIn('KEY2026B-004', text)
        self.assertIn('active', text)
