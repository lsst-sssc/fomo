"""Tests for solsys_code/admin.py -- proves the load-bearing admin constraints via the
admin test client rather than by eyeballing the ModelAdmin class definitions:

- CampaignRun and CalendarEventMeta are both reachable under /admin/solsys_code/.
- approval_status is visible-but-non-editable in the CampaignRun change form (T-jpd-01: no
  admin path to APPROVED that bypasses CampaignRunDecisionView.post()'s calendar projection
  + D-06 clobber guard).
- contact_person/contact_email never appear in the CampaignRun change-list (T-jpd-02: PII is
  not scannable across rows) but remain editable in the detail/change view.
- CalendarEventMeta's event__title search path resolves without a FieldError.
- Phase 27 Plan 05 (CANON-05/D-06/D-07/D-19): the CalendarEventMetaInline and
  CampaignRunObservationInline are reachable on the CampaignRun change page,
  CampaignRunAdmin.save_formset stamps confirmed_by/confirmed_at on newly created
  CampaignRunObservation rows only, and the source/telescope_class list_filter entries work.
"""

import re
from datetime import datetime
from datetime import timezone as dt_timezone

from django.contrib import admin as django_admin
from django.contrib.auth.models import User
from django.test import RequestFactory, TestCase
from django.urls import reverse
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import Target, TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory, SiderealTargetFactory

from solsys_code.admin import CalendarEventMetaInline, CampaignRunAdmin, CampaignRunObservationInline
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation

PII_CONTACT_PERSON = 'Zztestcontact'
PII_CONTACT_EMAIL = 'pii-secret@example.test'


class AdminRegistrationAndGatingTests(TestCase):
    """T-jpd-01/T-jpd-02: approval_status read-only, PII gated from the change-list."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(username='adminuser', email='admin@example.test', password='pw')
        cls.campaign = TargetList.objects.create(name='3I/ATLAS')
        # NOTE: not named `cls.run` -- unittest.TestCase.run() is the method the test
        # framework itself invokes to execute each test; shadowing it with an attribute
        # breaks test execution with `TypeError: 'CampaignRun' object is not callable`.
        cls.campaign_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='LCO-1m-Sinistro',
            contact_person=PII_CONTACT_PERSON,
            contact_email=PII_CONTACT_EMAIL,
        )

    def setUp(self) -> None:
        self.client.force_login(self.superuser)

    def test_campaignrun_changelist_loads(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_changelist'))
        self.assertEqual(response.status_code, 200)

    def test_calendareventmeta_changelist_loads(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_calendareventmeta_changelist'))
        self.assertEqual(response.status_code, 200)

    def test_calendareventmeta_search_resolves(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_calendareventmeta_changelist'), {'q': 'anything'})
        self.assertEqual(response.status_code, 200)

    def test_approval_status_is_readonly_in_change_form(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.campaign_run.pk]))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('Pending Review', content)
        self.assertNotIn('name="approval_status"', content)

    def test_contact_fields_editable_in_change_form(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.campaign_run.pk]))
        content = response.content.decode()
        self.assertIn('name="contact_person"', content)
        self.assertIn('name="contact_email"', content)

    def test_pii_not_rendered_in_changelist(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_changelist'))
        content = response.content.decode()
        self.assertNotIn(PII_CONTACT_PERSON, content)
        self.assertNotIn(PII_CONTACT_EMAIL, content)
        self.assertIn('LCO-1m-Sinistro', content)


class TargetAdminChangelistAndTypeFilterTests(TestCase):
    """quick-260722-uhh: the Target change-list loads and the 'By type' filter separates
    SIDEREAL from NON_SIDEREAL rows (tom_targets' own bare ModelAdmin has neither)."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(
            username='targetadminuser', email='targetadmin@example.test', password='pw'
        )
        cls.sidereal_target = SiderealTargetFactory(name='Test Sidereal Star')
        cls.non_sidereal_target = NonSiderealTargetFactory(name='Test NonSidereal Comet')

    def setUp(self) -> None:
        self.client.force_login(self.superuser)

    def _target_changelist_url_name(self) -> str:
        # Target = get_target_model_class() (tom_targets/models.py), which resolves to
        # BaseTarget here (no TARGET_MODEL_CLASS override in settings.py) -- so the admin
        # URL name is keyed off app_label/model_name ('tom_targets'/'basetarget'), not the
        # literal string 'target'.
        return f'admin:{Target._meta.app_label}_{Target._meta.model_name}_changelist'

    def test_target_changelist_loads(self) -> None:
        response = self.client.get(reverse(self._target_changelist_url_name()))
        self.assertEqual(response.status_code, 200)

    def test_type_filter_shows_only_sidereal(self) -> None:
        response = self.client.get(
            reverse(self._target_changelist_url_name()), {'type__exact': self.sidereal_target.type}
        )
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn(self.sidereal_target.name, content)
        self.assertNotIn(self.non_sidereal_target.name, content)

    def test_type_filter_shows_only_non_sidereal(self) -> None:
        response = self.client.get(
            reverse(self._target_changelist_url_name()), {'type__exact': self.non_sidereal_target.type}
        )
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn(self.non_sidereal_target.name, content)
        self.assertNotIn(self.sidereal_target.name, content)


class CampaignRunAdminInlinesTests(TestCase):
    """Phase 27 Plan 05 (CANON-05/D-06/D-07/D-19): the two new inlines, save_formset's
    attribution stamping, and the two new list_filter entries.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(
            username='inlineadminuser', email='inlineadmin@example.test', password='pw'
        )
        # Kept separate from cls.staff_user/cls.other_staff_user (the confirming staffers) --
        # ObservationRecord.user is on_delete=DO_NOTHING, mirroring test_campaign_run_observation.py.
        cls.record_owner = User.objects.create(username='record-owner-inline')
        # Superusers, not merely is_staff=True: DeleteProtectedModelForm.has_changed()
        # (django.contrib.admin.options) short-circuits to False -- treating a genuinely
        # changed inline form as unchanged -- when request.user lacks the inline model's own
        # add/change permission, even though is_staff is True. A superuser bypasses that
        # per-model permission check, matching how a real staff admin with the inline visible
        # to them would be provisioned.
        cls.staff_user = User.objects.create_superuser(
            username='confirming-staffer', email='cs@example.test', password='pw'
        )
        cls.other_staff_user = User.objects.create_superuser(
            username='other-staffer', email='os@example.test', password='pw'
        )
        cls.campaign = TargetList.objects.create(name='3I/ATLAS Inline Test')
        cls.target = NonSiderealTargetFactory.create()
        cls.campaign_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN/MuSCAT3',
            window_start='2025-07-04',
            window_end='2025-07-04',
        )
        cls.record_1 = ObservationRecord.objects.create(
            target=cls.target,
            user=cls.record_owner,
            facility='LCO',
            observation_id='333333',
            status='PENDING',
            parameters={'proposal': 'TEST'},
        )

    def setUp(self) -> None:
        self.client.force_login(self.superuser)
        self.factory = RequestFactory()

    def _staff_request(self, user):
        request = self.factory.post(reverse('admin:solsys_code_campaignrun_change', args=[self.campaign_run.pk]))
        request.user = user
        return request

    def test_both_inline_formsets_are_reachable_on_the_change_page(self) -> None:
        """CANON-05: derive the two formset prefixes from the response HTML rather than
        assuming them, then assert both are present. related_name is what Django's default
        inline prefix derives from -- calendar_event_metas (CalendarEventMeta.run) and
        observation_links (CampaignRunObservation.run).
        """
        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.campaign_run.pk]))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        observed_prefixes = set(re.findall(r'name="([\w_]+)-TOTAL_FORMS"', content))
        self.assertIn('calendar_event_metas', observed_prefixes)
        self.assertIn('observation_links', observed_prefixes)

    def test_save_formset_stamps_confirmed_by_and_confirmed_at_on_create(self) -> None:
        """D-07: a newly created CampaignRunObservation row is stamped with the acting
        staff user and a non-null confirmed_at."""
        request = self._staff_request(self.staff_user)
        inline = CampaignRunObservationInline(CampaignRun, django_admin.site)
        formset_class = inline.get_formset(request)
        prefix = formset_class.get_default_prefix()
        data = {
            f'{prefix}-TOTAL_FORMS': '1',
            f'{prefix}-INITIAL_FORMS': '0',
            f'{prefix}-MIN_NUM_FORMS': '0',
            f'{prefix}-MAX_NUM_FORMS': '1000',
            f'{prefix}-0-observation_record': str(self.record_1.pk),
            f'{prefix}-0-id': '',
        }
        formset = formset_class(data=data, instance=self.campaign_run)
        self.assertTrue(formset.is_valid(), formset.errors)
        admin_instance = CampaignRunAdmin(CampaignRun, django_admin.site)
        admin_instance.save_formset(request, None, formset, change=False)

        link = CampaignRunObservation.objects.get(run=self.campaign_run, observation_record=self.record_1)
        self.assertEqual(link.confirmed_by, self.staff_user)
        self.assertIsNotNone(link.confirmed_at)

    def test_save_formset_does_not_restamp_on_edit(self) -> None:
        """D-07: editing an existing CampaignRunObservation row through the admin must not
        overwrite its original confirmed_by/confirmed_at."""
        original_confirmed_at = datetime(2026, 1, 1, 12, 0, tzinfo=dt_timezone.utc)
        link = CampaignRunObservation.objects.create(
            run=self.campaign_run,
            observation_record=self.record_1,
            confirmed_by=self.other_staff_user,
            confirmed_at=original_confirmed_at,
        )
        request = self._staff_request(self.staff_user)
        inline = CampaignRunObservationInline(CampaignRun, django_admin.site)
        formset_class = inline.get_formset(request)
        prefix = formset_class.get_default_prefix()
        data = {
            f'{prefix}-TOTAL_FORMS': '1',
            f'{prefix}-INITIAL_FORMS': '1',
            f'{prefix}-MIN_NUM_FORMS': '0',
            f'{prefix}-MAX_NUM_FORMS': '1000',
            f'{prefix}-0-id': str(link.pk),
            f'{prefix}-0-observation_record': str(self.record_1.pk),
        }
        formset = formset_class(data=data, instance=self.campaign_run)
        self.assertTrue(formset.is_valid(), formset.errors)
        admin_instance = CampaignRunAdmin(CampaignRun, django_admin.site)
        admin_instance.save_formset(request, None, formset, change=True)

        link.refresh_from_db()
        self.assertEqual(link.confirmed_by, self.other_staff_user)
        self.assertEqual(link.confirmed_at, original_confirmed_at)

    def test_save_formset_does_not_stamp_calendar_event_meta_formset(self) -> None:
        """The isinstance gate: save_formset is called once per inline formset, so the
        CalendarEventMeta formset flows through the same method and must complete without
        error and write the run link, proving it is not mistakenly treated as the
        CampaignRunObservation formset (CalendarEventMeta has no audit fields at all)."""
        event = CalendarEvent.objects.create(
            title='Inline test event',
            start_time=datetime(2025, 7, 4, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 7, 5, 6, 0, tzinfo=dt_timezone.utc),
        )
        request = self._staff_request(self.staff_user)
        inline = CalendarEventMetaInline(CampaignRun, django_admin.site)
        formset_class = inline.get_formset(request)
        prefix = formset_class.get_default_prefix()
        data = {
            f'{prefix}-TOTAL_FORMS': '1',
            f'{prefix}-INITIAL_FORMS': '0',
            f'{prefix}-MIN_NUM_FORMS': '0',
            f'{prefix}-MAX_NUM_FORMS': '1000',
            f'{prefix}-0-event': str(event.pk),
            f'{prefix}-0-is_verified': 'on',
        }
        formset = formset_class(data=data, instance=self.campaign_run)
        self.assertTrue(formset.is_valid(), formset.errors)
        admin_instance = CampaignRunAdmin(CampaignRun, django_admin.site)
        admin_instance.save_formset(request, None, formset, change=False)

        meta = CalendarEventMeta.objects.get(event=event)
        self.assertEqual(meta.run_id, self.campaign_run.pk)

    def test_source_and_telescope_class_filters_return_200_and_appear_in_sidebar(self) -> None:
        """D-19: both new list_filter entries are usable and show up in the filter sidebar."""
        response = self.client.get(
            reverse('admin:solsys_code_campaignrun_changelist'), {'source': CampaignRun.Source.CSV_IMPORT}
        )
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        # Django's admin filter sidebar renders "By {field.verbose_name}" -- match the
        # model's actual verbose_name text, not the bare field name.
        self.assertIn('By Ingest source', content)
        self.assertIn('By Telescope class allocation', content)

        response = self.client.get(
            reverse('admin:solsys_code_campaignrun_changelist'), {'telescope_class': CampaignRun.TelescopeClass.ONE_M0}
        )
        self.assertEqual(response.status_code, 200)
