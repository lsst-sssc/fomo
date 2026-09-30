"""Tests for the target General Search override registered in GENERAL_SEARCH_FUNCTIONS."""

from django.contrib import admin
from django.contrib.auth.models import AnonymousUser, User
from django.test import RequestFactory, TestCase
from django.urls import reverse
from tom_targets.filters import TargetFilterSet
from tom_targets.models import Target

from solsys_code.search import target_general_search


class TestTargetGeneralSearch(TestCase):
    """The General Search box must match aliases, not just the current primary name."""

    def setUp(self):
        self.renamed = Target.objects.create(name='2026 RW1', type='NON_SIDEREAL', scheme='MPC_MINOR_PLANET')
        self.renamed.aliases.create(name='CERNQ52')
        self.other = Target.objects.create(name='2026 XY9', type='NON_SIDEREAL', scheme='MPC_MINOR_PLANET')

    def _pks(self, queryset):
        return sorted(target.pk for target in queryset)

    def _search(self, value):
        return self._pks(target_general_search(Target.objects.all(), 'query', value))

    def test_matches_alias(self):
        """A trksub that survives only as an alias still finds its renamed Target."""
        self.assertEqual(self._search('CERNQ52'), [self.renamed.pk])

    def test_matches_primary_name(self):
        self.assertEqual(self._search('2026 RW1'), [self.renamed.pk])

    def test_match_is_case_insensitive_and_partial(self):
        self.assertEqual(self._search('cernq'), [self.renamed.pk])

    def test_no_duplicate_rows_when_name_and_alias_both_match(self):
        """Joining across aliases must not return the same Target twice."""
        self.renamed.aliases.create(name='2026 RW1 (CERNQ52)')
        self.assertEqual(self._search('RW1'), [self.renamed.pk])

    def test_empty_value_returns_queryset_unchanged(self):
        self.assertEqual(self._search(''), self._pks(Target.objects.all()))

    def test_filterset_dispatches_to_the_configured_function(self):
        """GENERAL_SEARCH_FUNCTIONS must actually be picked up by the target list filter set."""
        request = RequestFactory().get('/targets/')
        request.user = AnonymousUser()
        filterset = TargetFilterSet(data={'query': 'CERNQ52'}, queryset=Target.objects.all(), request=request)
        self.assertEqual(self._pks(filterset.qs), [self.renamed.pk])


class TestTargetAdminOverride(TestCase):
    """solsys_code.admin must win over the bare ModelAdmin tom_targets registers."""

    def setUp(self):
        self.model_admin = admin.site._registry[Target]

    def _columns(self, query=''):
        return self.model_admin.get_list_display(RequestFactory().get(f'/admin/tom_targets/basetarget/{query}'))

    def test_admin_searches_aliases(self):
        self.assertIn('aliases__name', self.model_admin.search_fields)

    def test_unfiltered_changelist_shows_the_non_sidereal_columns(self):
        """ra/dec are null on every non-sidereal row, so they must not be the default."""
        columns = self._columns()
        self.assertIn('abs_mag', columns)
        self.assertIn('scheme', columns)
        self.assertNotIn('ra', columns)

    def test_filtering_to_sidereal_swaps_in_coordinates(self):
        columns = self._columns('?type__exact=SIDEREAL')
        self.assertIn('ra', columns)
        self.assertIn('dec', columns)
        self.assertNotIn('abs_mag', columns)

    def test_filtering_to_non_sidereal_keeps_the_default_columns(self):
        self.assertEqual(self._columns('?type__exact=NON_SIDEREAL'), self.model_admin.list_display)


class TestTargetSearchRequests(TestCase):
    """Alias search through real requests to the target list and the admin, not just their components."""

    def setUp(self):
        self.renamed = Target.objects.create(name='2026 RW1', type='NON_SIDEREAL', scheme='MPC_MINOR_PLANET')
        self.renamed.aliases.create(name='CERNQ52')
        self.other = Target.objects.create(name='2026 XY9', type='NON_SIDEREAL', scheme='MPC_MINOR_PLANET')

    def _pks(self, targets):
        return sorted(target.pk for target in targets)

    def _get_target_list(self, headers=None):
        response = self.client.get(reverse('targets:list'), {'query': 'CERNQ52'}, headers=headers or {})
        self.assertEqual(response.status_code, 200)
        return response

    def test_target_list_page_matches_alias(self):
        response = self._get_target_list()
        self.assertEqual(self._pks(response.context['object_list']), [self.renamed.pk])

    def test_target_list_htmx_partial_matches_alias(self):
        """Typing in the General Search box re-fetches only the table partial, via an HTMX request."""
        response = self._get_target_list(headers={'HX-Request': 'true'})
        self.assertTemplateNotUsed(response, 'tom_targets/target_list.html')
        self.assertEqual(self._pks(response.context['object_list']), [self.renamed.pk])

    def test_admin_changelist_search_matches_alias_once(self):
        """A Target with several matching aliases must still be listed once."""
        self.renamed.aliases.create(name='2026 RW1 (CERNQ52)')
        self.client.force_login(User.objects.create_superuser('admin', 'admin@example.com', 'password'))
        url = reverse(f'admin:{Target._meta.app_label}_{Target._meta.model_name}_changelist')
        response = self.client.get(url, {'q': 'CERNQ52'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._pks(response.context['cl'].result_list), [self.renamed.pk])
