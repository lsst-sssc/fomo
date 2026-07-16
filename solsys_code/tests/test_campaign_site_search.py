"""Tests for the shared anonymous live-search endpoint (Phase 22 Plan 01, D-01..D-05).

Covers the new ``substring_or_fuzzy_match_candidates()`` matcher (D-04/D-05) and the
per-IP throttle helper (D-02) in ``campaign_utils.py``, plus the ``SiteSearchView``
endpoint itself (D-01/D-03) -- anonymous access, HTML-fragment response shape, the
2-char minimum-length gate (22-REVIEWS.md finding 4), and the ``input_id`` server-side
allowlist + JS-string escaping (22-REVIEWS.md finding 2).

Reuses ``BULK_MPC_FIXTURE``/``campaign_utils._flatten_mpc_candidates()`` from
``test_campaign_approval.py`` (same pool-building convention already established there).
"""

import difflib
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from solsys_code import campaign_utils
from solsys_code.campaign_utils import (
    _check_and_increment_throttle,
    fuzzy_match_candidates,
    substring_or_fuzzy_match_candidates,
)
from solsys_code.tests.test_campaign_approval import BULK_MPC_FIXTURE, ISOLATED_TEST_CACHES


class SubstringOrFuzzyMatchCandidatesTest(TestCase):
    """D-04: substring-first containment matching, difflib fallback only on zero hits."""

    def setUp(self):
        self.pool = dict(campaign_utils._flatten_mpc_candidates(BULK_MPC_FIXTURE))
        # BULK_MPC_FIXTURE only has one Faulkes site (F65); add a second Faulkes-family
        # record here (operator's motivating "faulkes surfaces both" example) since
        # BULK_MPC_FIXTURE deliberately only fixtures one.
        self.pool['Haleakala-Faulkes Telescope North'] = 'F65N'

    def test_substring_hit_surfaces_all_faulkes_candidates(self):
        results = substring_or_fuzzy_match_candidates('faulkes', self.pool)
        displays = [display for display, _obscode in results]
        self.assertGreaterEqual(len(results), 2)
        self.assertIn('Faulkes Telescope South', displays)
        self.assertIn('Haleakala-Faulkes Telescope North', displays)
        for display in displays:
            self.assertIn('faulkes', display.lower())

    def test_case_insensitive_same_result_set(self):
        lower = substring_or_fuzzy_match_candidates('faulkes', self.pool)
        upper = substring_or_fuzzy_match_candidates('FAULKES', self.pool)
        self.assertEqual(set(lower), set(upper))

    def test_substring_beats_difflib_cutoff_for_lowell(self):
        results = substring_or_fuzzy_match_candidates('lowell', self.pool)
        displays = [display for display, _obscode in results]
        self.assertIn('Lowell Discovery Telescope', displays)
        # Prove difflib alone (at its 0.6 cutoff) would NOT bridge this short partial
        # query against the long official MPC string -- substring containment is what
        # actually finds it here, not the fallback.
        difflib_only = difflib.get_close_matches('lowell', self.pool.keys(), n=5, cutoff=0.6)
        self.assertNotIn('Lowell Discovery Telescope', difflib_only)

    def test_difflib_fallback_only_when_containment_finds_nothing(self):
        pool = {'Cassini Occultation Station': 'X01'}
        query = 'Cassini Ocultation Station'  # typo: missing one 'c', no substring hit
        self.assertNotIn(query.lower(), 'cassini occultation station')
        results = substring_or_fuzzy_match_candidates(query, pool)
        self.assertEqual(results, [('Cassini Occultation Station', 'X01')])

    def test_blank_or_whitespace_input_returns_empty_list(self):
        self.assertEqual(substring_or_fuzzy_match_candidates('', self.pool), [])
        self.assertEqual(substring_or_fuzzy_match_candidates('   ', self.pool), [])

    def test_limit_caps_result_length(self):
        pool = {f'Site Alpha {i}': f'A{i:02d}' for i in range(20)}
        default_capped = substring_or_fuzzy_match_candidates('site alpha', pool)
        self.assertEqual(len(default_capped), 8)
        explicit_capped = substring_or_fuzzy_match_candidates('site alpha', pool, limit=3)
        self.assertEqual(len(explicit_capped), 3)

    def test_fuzzy_match_candidates_unaffected_by_new_n_parameter_default(self):
        # fuzzy_match_candidates() itself must remain behaviorally unchanged for its
        # existing single call site (ApprovalQueueTable.render_site()) -- default n=5.
        pool = {f'Site Alpha {i}': f'A{i:02d}' for i in range(20)}
        results = fuzzy_match_candidates('Site Alpha', pool)
        self.assertLessEqual(len(results), 5)


@override_settings(CACHES=ISOLATED_TEST_CACHES)
class ThrottleTest(TestCase):
    """D-02: per-IP fixed-window throttle via django.core.cache.

    Cache-isolated (bug #3, debug/site-search-degraded-pool-recurrence): writes throttle keys
    and calls ``cache.clear()`` -- pinned to an in-memory LocMemCache so it never wipes the
    shared /tmp file cache the dev runserver serves site-search from.
    """

    def setUp(self):
        cache.clear()

    @patch.object(campaign_utils, 'SITE_SEARCH_THROTTLE_LIMIT', 3)
    def test_allows_up_to_limit_then_rejects(self):
        for _ in range(3):
            self.assertTrue(_check_and_increment_throttle('1.2.3.4'))
        self.assertFalse(_check_and_increment_throttle('1.2.3.4'))

    @patch.object(campaign_utils, 'SITE_SEARCH_THROTTLE_LIMIT', 3)
    def test_counts_different_ips_independently(self):
        for _ in range(3):
            self.assertTrue(_check_and_increment_throttle('1.1.1.1'))
        self.assertFalse(_check_and_increment_throttle('1.1.1.1'))
        self.assertTrue(_check_and_increment_throttle('2.2.2.2'))


@override_settings(CACHES=ISOLATED_TEST_CACHES)
class SiteSearchViewTest(TestCase):
    """D-01/D-03: anonymous, throttled, HTML-fragment live-search endpoint.

    Cache-isolated (bug #3, debug/site-search-degraded-pool-recurrence): calls ``cache.clear()``
    in setUp and exercises the throttle -- pinned to an in-memory LocMemCache so it never wipes
    the shared /tmp file cache the dev runserver serves site-search from.
    """

    @classmethod
    def setUpTestData(cls):
        cls.staff_user = User.objects.create_user(username='staffcoordinator', password='pw', is_staff=True)

    def setUp(self):
        cache.clear()
        patcher = patch(
            'solsys_code.campaign_views.build_site_candidates',
            return_value=campaign_utils._flatten_mpc_candidates(BULK_MPC_FIXTURE),
        )
        self.mock_build_site_candidates = patcher.start()
        self.addCleanup(patcher.stop)

    def test_anonymous_get_returns_html_fragment_with_suggestion(self):
        response = self.client.get(reverse('campaigns:site_search'), {'q': 'faulkes'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response['Content-Type'].startswith('text/html'))
        self.assertContains(response, '<li')
        self.assertContains(response, 'Faulkes Telescope South')

    @patch.object(campaign_utils, 'SITE_SEARCH_THROTTLE_LIMIT', 2)
    def test_over_limit_anonymous_request_returns_429(self):
        for _ in range(2):
            response = self.client.get(reverse('campaigns:site_search'), {'q': 'faulkes'})
            self.assertEqual(response.status_code, 200)
        response = self.client.get(reverse('campaigns:site_search'), {'q': 'faulkes'})
        self.assertEqual(response.status_code, 429)

    @patch.object(campaign_utils, 'SITE_SEARCH_THROTTLE_LIMIT', 2)
    def test_staff_session_not_throttled_at_anonymous_limit(self):
        self.client.login(username='staffcoordinator', password='pw')
        for _ in range(5):
            response = self.client.get(reverse('campaigns:site_search'), {'q': 'faulkes'})
            self.assertEqual(response.status_code, 200)

    def test_no_match_query_renders_correct_copy_per_input_id(self):
        response = self.client.get(reverse('campaigns:site_search'), {'q': 'zzzznomatch', 'input_id': 'id_site_raw'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'No matches — free text is fine, a staff member will resolve it.')

        response = self.client.get(
            reverse('campaigns:site_search'), {'q': 'zzzznomatch', 'input_id': 'site-selection-1'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'No matches for this search.')

    def test_blank_query_returns_empty_fragment_without_building_pool(self):
        self.mock_build_site_candidates.reset_mock()
        response = self.client.get(reverse('campaigns:site_search'), {'q': ''})
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<li')
        self.mock_build_site_candidates.assert_not_called()

    def test_one_char_query_returns_empty_fragment_without_building_pool(self):
        self.mock_build_site_candidates.reset_mock()
        response = self.client.get(reverse('campaigns:site_search'), {'q': 'x'})
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<li')
        self.mock_build_site_candidates.assert_not_called()

    def test_site_raw_param_without_q_returns_suggestions(self):
        # 22-04 gap closure (UAT test 1): the public submission form's widget is
        # `<input name="site_raw">` -- htmx's hx-get sends `?site_raw=<text>`, never `q`.
        response = self.client.get(reverse('campaigns:site_search'), {'site_raw': 'faulkes'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<li')
        self.assertContains(response, 'Faulkes Telescope South')

    def test_site_selection_param_without_q_returns_suggestions(self):
        # 22-04 gap closure (UAT test 3): the approval-queue / Sites Needing Review
        # widgets are `<input name="site_selection">` -- same missing-`q` defect.
        response = self.client.get(
            reverse('campaigns:site_search'), {'site_selection': 'faulkes', 'input_id': 'site-input-1'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<li')
        self.assertContains(response, 'Faulkes Telescope South')

    def test_q_takes_precedence_over_site_raw(self):
        # Guarantee no existing `?q=` caller regressed: when both are present, `q` wins.
        response = self.client.get(reverse('campaigns:site_search'), {'q': 'faulkes', 'site_raw': 'lowell'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Faulkes Telescope South')
        self.assertNotContains(response, 'Lowell Discovery Telescope')

    def test_one_char_site_raw_returns_empty_fragment_without_building_pool(self):
        self.mock_build_site_candidates.reset_mock()
        response = self.client.get(reverse('campaigns:site_search'), {'site_raw': 'x'})
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<li')
        self.mock_build_site_candidates.assert_not_called()

    def test_hostile_input_id_is_replaced_with_default_fallback(self):
        response = self.client.get(
            reverse('campaigns:site_search'),
            {'q': 'faulkes', 'input_id': "x');alert(1);//"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('alert(', response.content.decode())
        self.assertContains(response, "getElementById('id_site_raw')")

    def test_hostile_candidate_text_is_escaped_in_js_string_context(self):
        hostile_pool = {"Evil'); payload; ('Site": 'ZZZ'}
        with patch('solsys_code.campaign_views.build_site_candidates', return_value=hostile_pool):
            response = self.client.get(reverse('campaigns:site_search'), {'q': 'evil'})
        body = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn('\\u0027', body)
        self.assertNotIn("'); payload", body)
