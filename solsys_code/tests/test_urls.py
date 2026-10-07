"""Tests for the project-level URL configuration in ``src/fomo/urls.py``.

Main's commit ada2000 removed the project-level ``alerts/`` include when tomtoolkit 3.1.0 stopped
installing ``tom_alerts``. The Phase 38 merge resolution put it back (38-REVIEW.md CR-01), which left
the ``/alerts/`` pages failing. These tests stop a later merge from restoring it, and check that the
project-level routes FOMO and main add before ``tom_common.urls`` still win over tom_common's own.
"""

from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase
from django.urls import NoReverseMatch, Resolver404, resolve, reverse

from solsys_code.campaign_views import CampaignListView
from solsys_code.views import ProtectedUserDeleteView, fomo_render_calendar


class TestAlertsRouteRemoved(TestCase):
    """The alerts/ include main deleted (ada2000) must stay gone: tom_alerts is not an installed app."""

    def test_alerts_path_does_not_resolve(self) -> None:
        with self.assertRaises(Resolver404):
            resolve('/alerts/query/list/')

    def test_alerts_namespace_cannot_be_reversed(self) -> None:
        with self.assertRaises(NoReverseMatch):
            reverse('alerts:list')

    def test_logged_in_get_of_alerts_list_returns_404(self) -> None:
        user = User.objects.create_user(username='alerts-probe', password='pw')
        client = Client(raise_request_exception=False)
        client.force_login(user)

        response = client.get('/alerts/query/list/')

        self.assertEqual(response.status_code, 404)


class TestProjectRoutesStillResolve(SimpleTestCase):
    """Routes registered before tom_common.urls still win over tom_common's own."""

    def test_project_routes_resolve_to_fomo_and_main_views(self) -> None:
        self.assertEqual(resolve('/targets/').view_name, 'scout_target_list')
        self.assertEqual(resolve('/scout/rubin-too/').view_name, 'scout_rubin_too')
        self.assertEqual(resolve('/scout/rubin-too/stats/').view_name, 'scout_rubin_too_stats')

        calendar_match = resolve('/calendar/')
        self.assertEqual(calendar_match.view_name, 'calendar:calendar')
        self.assertIs(calendar_match.func, fomo_render_calendar)

        campaigns_match = resolve('/campaigns/')
        self.assertEqual(campaigns_match.view_name, 'campaigns:list')
        self.assertIs(campaigns_match.func.view_class, CampaignListView)

        delete_match = resolve('/users/1/delete/')
        self.assertEqual(delete_match.view_name, 'user-delete')
        self.assertIs(delete_match.func.view_class, ProtectedUserDeleteView)
