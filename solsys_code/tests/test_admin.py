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
- Plan 27.1-02 (criteria 4/6): CampaignRun/CalendarEventMeta label distinguishability
  against the real 11-row companion-record shape, the CalendarEventMetaAdmin.run
  autocomplete endpoint, and the source provenance lock on an already-approved WEB run.
- Plan 27.1-05 (closing criterion 6, WR-03): the lock now covers every web-sourced row at
  any approval status, and SourceProvenanceTwoStepBypassTests pins the cross-path
  edit-then-approve sequence that used to bypass it.
"""

import re
from datetime import date, datetime
from datetime import timezone as dt_timezone
from unittest.mock import patch

from django.contrib import admin as django_admin
from django.contrib.auth.models import User
from django.forms.models import model_to_dict
from django.test import RequestFactory, TestCase
from django.urls import reverse
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord
from tom_targets.models import Target, TargetList
from tom_targets.tests.factories import NonSiderealTargetFactory, SiderealTargetFactory

from solsys_code.admin import CalendarEventMetaInline, CampaignRunAdmin, CampaignRunObservationInline
from solsys_code.models import CalendarEventMeta, CampaignRun, CampaignRunObservation
from solsys_code.solsys_code_observatory.models import Observatory

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


class CalendarEventMetaStandaloneAdminPkFreezeTests(TestCase):
    """CR-02: the WR-08 primary-key freeze must cover the STANDALONE
    ``CalendarEventMetaAdmin`` change form, not just ``CalendarEventMetaInline``.

    ``CalendarEventMeta.event`` is an explicitly declared
    ``OneToOneField(primary_key=True)``, which Django treats as editable, so it renders as a
    live ``<select>`` on the standalone change form. Re-pointing it makes ``instance.pk`` a
    value absent from the table, so ``instance.save()`` issues an UPDATE matching 0 rows and
    falls back to an INSERT -- leaving the original row behind as a duplicate with orphaned
    ``is_verified``/``run`` history (migration 0008's header comment), or silently clobbering
    the target event's own companion row if it already has one.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(
            username='pkfreezeadmin', email='pkfreeze@example.test', password='pw'
        )
        cls.original_event = CalendarEvent.objects.create(
            title='Original standalone event',
            start_time=datetime(2025, 9, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 9, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        cls.other_event = CalendarEvent.objects.create(
            title='Hijack target standalone event',
            start_time=datetime(2025, 9, 3, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 9, 4, 6, 0, tzinfo=dt_timezone.utc),
        )
        cls.meta = CalendarEventMeta.objects.create(event=cls.original_event, is_verified=True)

    def setUp(self) -> None:
        self.client.force_login(self.superuser)

    def test_change_form_does_not_expose_event_as_editable(self) -> None:
        """The pk must not render as an editable widget on the change form."""
        response = self.client.get(reverse('admin:solsys_code_calendareventmeta_change', args=[self.original_event.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('name="event"', response.content.decode())

    def test_add_form_still_exposes_event(self) -> None:
        """Freezing on change must not break linking a new event -- add stays editable, so
        the link remains re-pointable by delete + re-add."""
        response = self.client.get(reverse('admin:solsys_code_calendareventmeta_add'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="event"', response.content.decode())

    def test_posting_a_different_event_pk_does_not_duplicate_the_row(self) -> None:
        """The reproduction case: one row in, one row out, still pointing at its own event.

        WR-08: the response and a genuinely-changed editable field are both asserted, because
        every "nothing was duplicated" assertion below is equally satisfied by a POST that
        403'd, 500'd, or re-rendered the change form with errors and saved nothing. Without
        those two checks this test cannot tell "the pk freeze held" from "the write path was
        never reached", and a future change that breaks the change view outright would leave
        it green.
        """
        response = self.client.post(
            reverse('admin:solsys_code_calendareventmeta_change', args=[self.original_event.pk]),
            {'event': str(self.other_event.pk), 'is_verified': '', '_save': 'Save'},
        )

        # 302 = the admin processed the save and redirected, rather than re-rendering the
        # form (200) or refusing the request.
        self.assertEqual(response.status_code, 302)
        self.assertEqual(CalendarEventMeta.objects.count(), 1)
        self.meta.refresh_from_db()
        self.assertEqual(self.meta.event_id, self.original_event.pk)
        # is_verified started True (setUpTestData); it is False now only if the form really
        # saved -- so the pk freeze held on a POST that genuinely wrote.
        self.assertFalse(self.meta.is_verified)
        self.assertFalse(CalendarEventMeta.objects.filter(event=self.other_event).exists())


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

    def test_calendar_event_meta_inline_freezes_event_on_existing_rows(self) -> None:
        """WR-08: `event` is the model's primary key, so re-pointing it on an existing row
        would INSERT a second row and orphan the original instead of moving the link.
        The inline's formset disables the field on saved rows (and only on saved rows), so
        a submitted event pk for an existing row is ignored, no duplicate is written, and
        the blank "Add another" row keeps a usable widget."""
        original_event = CalendarEvent.objects.create(
            title='Original inline event',
            start_time=datetime(2025, 8, 1, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 8, 2, 6, 0, tzinfo=dt_timezone.utc),
        )
        other_event = CalendarEvent.objects.create(
            title='Hijack target event',
            start_time=datetime(2025, 8, 3, 22, 0, tzinfo=dt_timezone.utc),
            end_time=datetime(2025, 8, 4, 6, 0, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=original_event, run=self.campaign_run, is_verified=True)

        request = self._staff_request(self.staff_user)
        inline = CalendarEventMetaInline(CampaignRun, django_admin.site)
        formset_class = inline.get_formset(request, obj=self.campaign_run)
        prefix = formset_class.get_default_prefix()
        data = {
            f'{prefix}-TOTAL_FORMS': '1',
            f'{prefix}-INITIAL_FORMS': '1',
            f'{prefix}-MIN_NUM_FORMS': '0',
            f'{prefix}-MAX_NUM_FORMS': '1000',
            # The hijack attempt: point the existing row's pk at a different CalendarEvent.
            f'{prefix}-0-event': str(other_event.pk),
            f'{prefix}-0-is_verified': 'on',
        }
        formset = formset_class(data=data, instance=self.campaign_run)
        self.assertTrue(formset.is_valid(), formset.errors)
        self.assertTrue(formset.forms[0].fields['event'].disabled)
        # The blank add row must stay editable, or linking a new event from an existing
        # run's change page would be impossible.
        self.assertFalse(formset.empty_form.fields['event'].disabled)

        admin_instance = CampaignRunAdmin(CampaignRun, django_admin.site)
        admin_instance.save_formset(request, None, formset, change=True)

        self.assertEqual(CalendarEventMeta.objects.filter(run=self.campaign_run).count(), 1)
        self.assertTrue(CalendarEventMeta.objects.filter(event=original_event).exists())
        self.assertFalse(CalendarEventMeta.objects.filter(event=other_event).exists())

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


class CalendarEventMetaLabelLegibilityTests(TestCase):
    """27.1-02 criterion 4: CalendarEventMeta.__str__ against the real 11-row companion-record
    shape measured from the live dev DB. 7 of the 11 titles are byte-identical, which is what
    made the picker unusable before Task 1's fix -- the event start date is load-bearing.
    """

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(
            username='calmeta-label-admin', email='calmeta-label@example.test', password='pw'
        )
        # (title, start_date) pairs measured from the live dev DB (2026-07-30).
        fixture_rows = [
            ('[CANCELLED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 7)),
            ('COJ-2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 8)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 10)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 11)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 12)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 14)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 16)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 17)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 18)),
            ('[EXPIRED] 2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 20)),
            ('2m0 2M0-SCICAM-MUSCAT', date(2026, 7, 19)),
        ]
        assert len(fixture_rows) == 11
        for title, start_date in fixture_rows:
            event = CalendarEvent.objects.create(
                title=title,
                start_time=datetime(start_date.year, start_date.month, start_date.day, 22, 0, tzinfo=dt_timezone.utc),
                end_time=datetime(start_date.year, start_date.month, start_date.day, 23, 0, tzinfo=dt_timezone.utc),
            )
            CalendarEventMeta.objects.create(event=event, is_verified=True)

    def setUp(self) -> None:
        self.client.force_login(self.superuser)

    def test_11_companion_rows_have_11_distinct_labels(self) -> None:
        self.assertEqual(len({str(m) for m in CalendarEventMeta.objects.all()}), 11)

    def test_pre_fix_title_alone_would_have_collided(self) -> None:
        """Documents why the date is load-bearing: the event title alone is not distinct
        across all 11 rows (7 share the identical [EXPIRED] title)."""
        self.assertLess(len({m.event.title for m in CalendarEventMeta.objects.all()}), 11)

    def test_changelist_renders_a_date_staff_can_scan_by(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_calendareventmeta_changelist'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('2026-07-07', response.content.decode())


class CampaignRunLabelLegibilityTests(TestCase):
    """27.1-02 criterion 4: CampaignRun.__str__ covers every site_label branch and the real
    pk 27/28 TBD-collision case."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(
            username='camprun-label-admin', email='camprun-label@example.test', password='pw'
        )
        cls.campaign = TargetList.objects.create(name='Label Legibility Campaign')
        cls.observatory = Observatory.objects.create(
            obscode='TST1',
            name='Label Legibility Test Site',
            short_name='LLTS',
            lat=1.0,
            lon=2.0,
            altitude=100.0,
        )
        cls.site_resolved_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN-site-resolved',
            window_start=date(2026, 1, 1),
            window_end=date(2026, 1, 1),
            site=cls.observatory,
        )
        cls.classed_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN-classed',
            window_start=date(2026, 1, 2),
            window_end=date(2026, 1, 2),
            telescope_class=CampaignRun.TelescopeClass.ONE_M0,
        )
        cls.site_raw_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN-site-raw',
            window_start=date(2026, 1, 3),
            window_end=date(2026, 1, 3),
            site_raw='Some Unresolved Site Text',
        )
        cls.bare_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN-bare',
            window_start=date(2026, 1, 4),
            window_end=date(2026, 1, 4),
        )
        # The real pk 27/28 collision: same campaign, same telescope_instrument, both TBD --
        # only distinguishable via pk and (per the TBD natural-key constraint) contact_person.
        cls.tbd_run_1 = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='JWST',
            contact_person='Label Test Contact A',
        )
        cls.tbd_run_2 = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='JWST',
            contact_person='Label Test Contact B',
        )
        cls.pii_run = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='FTN-pii',
            window_start=date(2026, 1, 5),
            window_end=date(2026, 1, 5),
            contact_person=PII_CONTACT_PERSON,
            contact_email=PII_CONTACT_EMAIL,
        )

    def setUp(self) -> None:
        self.client.force_login(self.superuser)

    def _all_runs(self):
        return [
            self.site_resolved_run,
            self.classed_run,
            self.site_raw_run,
            self.bare_run,
            self.tbd_run_1,
            self.tbd_run_2,
            self.pii_run,
        ]

    def test_all_labels_distinct_and_none_contains_the_literal_none(self) -> None:
        labels = [str(r) for r in self._all_runs()]
        self.assertEqual(len(set(labels)), len(labels))
        for label in labels:
            self.assertNotIn('None', label)

    def test_tbd_runs_labelled_tbd_and_distinguishable(self) -> None:
        label_1 = str(self.tbd_run_1)
        label_2 = str(self.tbd_run_2)
        self.assertIn('TBD', label_1)
        self.assertIn('TBD', label_2)
        self.assertNotEqual(label_1, label_2)

    def test_site_resolved_label_contains_obscode(self) -> None:
        self.assertIn(self.observatory.obscode, str(self.site_resolved_run))

    def test_classed_run_label_contains_class_discriminator(self) -> None:
        self.assertIn('class 1m0', str(self.classed_run))

    def test_site_raw_run_label_contains_raw_discriminator(self) -> None:
        self.assertIn('raw ', str(self.site_raw_run))

    def test_bare_run_label_contains_no_site(self) -> None:
        self.assertIn('no site', str(self.bare_run))

    def test_pii_run_label_excludes_contact_fields(self) -> None:
        label = str(self.pii_run)
        self.assertNotIn(PII_CONTACT_PERSON, label)
        self.assertNotIn(PII_CONTACT_EMAIL, label)

    def test_autocomplete_endpoint_resolves_with_discriminating_text_and_no_pii(self) -> None:
        """The picker the UAT gap's second 'missing' item asks for: a search box, not a
        44-option flat <select>."""
        response = self.client.get(
            reverse('admin:autocomplete'),
            {
                'app_label': 'solsys_code',
                'model_name': 'calendareventmeta',
                'field_name': 'run',
                'term': 'FTN-pii',
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        results = payload['results']
        self.assertTrue(results)
        self.assertTrue(any('FTN-pii' in result['text'] for result in results))
        self.assertFalse(any(PII_CONTACT_PERSON in result['text'] for result in results))


class SourceProvenanceLockTests(TestCase):
    """27.1-05 (closing criterion 6, WR-03): `source` is non-overwritable on every
    `source == WEB` run, at any approval status, and stays editable on every non-WEB row
    of every approval status (D-19 preserved)."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(
            username='source-lock-admin', email='source-lock@example.test', password='pw'
        )
        cls.campaign = TargetList.objects.create(name='Source Lock Campaign')
        cls.approved_web = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Source-Lock-Approved-Web',
            source=CampaignRun.Source.WEB,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )
        cls.pending_web = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Source-Lock-Pending-Web',
            source=CampaignRun.Source.WEB,
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        cls.rejected_web = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Source-Lock-Rejected-Web',
            source=CampaignRun.Source.WEB,
            approval_status=CampaignRun.ApprovalStatus.REJECTED,
        )
        cls.approved_legacy = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Source-Lock-Approved-Legacy',
            source=CampaignRun.Source.LEGACY,
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
        )

    def setUp(self) -> None:
        self.client.force_login(self.superuser)
        self.factory = RequestFactory()
        self.admin_obj = CampaignRunAdmin(CampaignRun, django_admin.site)

    def _staff_request(self):
        request = self.factory.get(reverse('admin:solsys_code_campaignrun_changelist'))
        request.user = self.superuser
        return request

    def test_source_withheld_on_every_web_row(self) -> None:
        request = self._staff_request()
        self.assertIn('source', self.admin_obj.get_readonly_fields(request, self.approved_web))
        self.assertIn('source', self.admin_obj.get_readonly_fields(request, self.pending_web))
        self.assertIn('source', self.admin_obj.get_readonly_fields(request, self.rejected_web))
        self.assertNotIn('source', self.admin_obj.get_readonly_fields(request, self.approved_legacy))
        self.assertNotIn('source', self.admin_obj.get_readonly_fields(request, None))

    def test_approval_status_lock_survives_for_all_four(self) -> None:
        request = self._staff_request()
        for obj in (self.approved_web, self.pending_web, self.rejected_web, self.approved_legacy, None):
            self.assertIn('approval_status', self.admin_obj.get_readonly_fields(request, obj))

    def test_source_field_absent_from_generated_form_on_every_web_row(self) -> None:
        request = self._staff_request()
        approved_form = self.admin_obj.get_form(request, obj=self.approved_web, change=True)
        pending_form = self.admin_obj.get_form(request, obj=self.pending_web, change=True)
        legacy_form = self.admin_obj.get_form(request, obj=self.approved_legacy, change=True)
        self.assertNotIn('source', approved_form.base_fields)
        self.assertNotIn('source', pending_form.base_fields)
        self.assertIn('source', legacy_form.base_fields)

    def test_submitted_source_value_cannot_bind_on_approved_web(self) -> None:
        """End-to-end: a POSTed source value on an approved WEB run cannot bind, since
        Django excludes readonly fields from the generated ModelForm entirely."""
        request = self._staff_request()
        form_class = self.admin_obj.get_form(request, obj=self.approved_web, change=True)
        data = model_to_dict(self.approved_web, exclude=['id'])
        data['source'] = CampaignRun.Source.CSV_IMPORT
        form = form_class(data=data, instance=self.approved_web)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.approved_web.refresh_from_db()
        self.assertEqual(self.approved_web.source, CampaignRun.Source.WEB)

    def test_submitted_source_value_cannot_bind_on_pending_web(self) -> None:
        """27.1-05: this is the write half of the two-step sequence 27.1-05 closes -- a
        submitted value on a pending WEB row no longer binds, since Django excludes readonly
        fields from the generated ModelForm entirely."""
        request = self._staff_request()
        form_class = self.admin_obj.get_form(request, obj=self.pending_web, change=True)
        data = model_to_dict(self.pending_web, exclude=['id'])
        data['source'] = CampaignRun.Source.CSV_IMPORT
        form = form_class(data=data, instance=self.pending_web)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.pending_web.refresh_from_db()
        self.assertEqual(self.pending_web.source, CampaignRun.Source.WEB)

    def test_submitted_source_value_does_bind_on_non_web_row(self) -> None:
        """The lock is scoped to web rows, not blanket: the same write against a non-web
        row succeeds, proving D-19's correction use case survives with a passing
        assertion rather than only by prose."""
        request = self._staff_request()
        form_class = self.admin_obj.get_form(request, obj=self.approved_legacy, change=True)
        data = model_to_dict(self.approved_legacy, exclude=['id'])
        data['source'] = CampaignRun.Source.CSV_IMPORT
        form = form_class(data=data, instance=self.approved_legacy)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.approved_legacy.refresh_from_db()
        self.assertEqual(self.approved_legacy.source, CampaignRun.Source.CSV_IMPORT)

    def test_relabel_to_web_locks_the_row_and_cannot_be_undone(self) -> None:
        """WR-10: pins the accepted one-way ratchet, in the direction the lock deliberately
        leaves open.

        Relabelling a non-web row *to* web is still allowed (D-19 keeps `source` editable on
        every non-web row), but the moment it saves, `get_readonly_fields` keys off the new
        value and withholds `source` -- so the same staff user cannot undo their own
        mis-click through the admin at all. The CSV path cannot repair it either:
        `import_campaign_csv` pops `source`/`approval_status` for any row whose existing
        source is web (`test_reimport_preserves_web_source_and_approval_status` in
        `test_import_campaign_csv.py` pins that half). Both consequences are accepted, but
        they were previously unpinned in either direction -- a future refactor could have
        silently changed them.
        """
        request = self._staff_request()
        # The relabel itself must still be allowed -- this is D-19's surviving use case.
        self.assertNotIn('source', self.admin_obj.get_readonly_fields(request, self.approved_legacy))
        form_class = self.admin_obj.get_form(request, obj=self.approved_legacy, change=True)
        data = model_to_dict(self.approved_legacy, exclude=['id'])
        data['source'] = CampaignRun.Source.WEB
        form = form_class(data=data, instance=self.approved_legacy)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.approved_legacy.refresh_from_db()
        self.assertEqual(self.approved_legacy.source, CampaignRun.Source.WEB)

        # ...and now the trapdoor has shut: no admin write path back.
        self.assertIn('source', self.admin_obj.get_readonly_fields(request, self.approved_legacy))
        relocked_form_class = self.admin_obj.get_form(request, obj=self.approved_legacy, change=True)
        self.assertNotIn('source', relocked_form_class.base_fields)
        undo_data = model_to_dict(self.approved_legacy, exclude=['id'])
        undo_data['source'] = CampaignRun.Source.LEGACY
        undo_form = relocked_form_class(data=undo_data, instance=self.approved_legacy)
        self.assertTrue(undo_form.is_valid(), undo_form.errors)
        undo_form.save()
        self.approved_legacy.refresh_from_db()
        self.assertEqual(self.approved_legacy.source, CampaignRun.Source.WEB)

    def test_change_page_omits_source_field_on_every_web_row(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.approved_web.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('name="source"', response.content.decode())

        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.pending_web.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('name="source"', response.content.decode())

        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.approved_legacy.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="source"', response.content.decode())


class SourceProvenanceTwoStepBypassTests(TestCase):
    """27.1-05: pins the sequence `27.1-REVIEW.md` WR-03 demonstrated, end to end across two
    different write paths -- the admin change form and `CampaignRunDecisionView` -- because
    neither path's own tests can see the sequence in isolation."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.superuser = User.objects.create_superuser(
            username='two-step-admin', email='two-step@example.test', password='pw'
        )
        cls.campaign = TargetList.objects.create(name='Two-Step Bypass Campaign')
        # window_start/window_end intentionally left unset on both fixtures, so
        # _project_calendar_event() skips projection by design and the approve branch has
        # no calendar side effect to manage here.
        cls.pending_web = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Two-Step-Pending-Web',
            source=CampaignRun.Source.WEB,
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )
        cls.pending_csv_import = CampaignRun.objects.create(
            campaign=cls.campaign,
            telescope_instrument='Two-Step-Pending-Csv-Import',
            source=CampaignRun.Source.CSV_IMPORT,
            approval_status=CampaignRun.ApprovalStatus.PENDING_REVIEW,
        )

    def setUp(self) -> None:
        self.admin_obj = CampaignRunAdmin(CampaignRun, django_admin.site)
        self.factory = RequestFactory()

    def _staff_request(self):
        request = self.factory.get(reverse('admin:solsys_code_campaignrun_changelist'))
        request.user = self.superuser
        return request

    def test_pending_relabel_then_approve_keeps_web_source(self) -> None:
        """The non-negotiable regression test for WR-03's two-step bypass."""
        # Step 1: attempt the relabel exactly as SourceProvenanceLockTests models an admin
        # write -- build the form via get_form(), feed it a submitted `source`, and confirm
        # it does not bind.
        request = self._staff_request()
        form_class = self.admin_obj.get_form(request, obj=self.pending_web, change=True)
        data = model_to_dict(self.pending_web, exclude=['id'])
        data['source'] = CampaignRun.Source.CSV_IMPORT
        form = form_class(data=data, instance=self.pending_web)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.pending_web.refresh_from_db()
        self.assertEqual(self.pending_web.source, CampaignRun.Source.WEB)

        # Step 2: approve via CampaignRunDecisionView. The run has site=None, so the approve
        # branch calls resolve_site() -- patched here so the test never reaches the MPC
        # Obscodes API.
        self.client.force_login(self.superuser)
        with patch('solsys_code.campaign_views.resolve_site', return_value=(None, True)):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': self.pending_web.pk}), {'action': 'approve'}
            )
        self.assertEqual(response.status_code, 302)

        self.pending_web.refresh_from_db()
        # This is the assertion the whole test exists for: the run must land on APPROVED +
        # web, never on APPROVED + non-web -- the latter would, under CANON-01's derivation
        # rule, read as "no approval was required", which is the exact provenance loss
        # 27.1-05 closes.
        self.assertEqual(self.pending_web.approval_status, CampaignRun.ApprovalStatus.APPROVED)
        self.assertEqual(self.pending_web.source, CampaignRun.Source.WEB)

    def test_approve_still_lands_on_a_web_run(self) -> None:
        """Control: proves the approval itself really lands, so the regression test above
        cannot pass vacuously by the approval silently failing and reverting."""
        self.client.force_login(self.superuser)
        with patch('solsys_code.campaign_views.resolve_site', return_value=(None, True)):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': self.pending_web.pk}), {'action': 'approve'}
            )
        self.assertEqual(response.status_code, 302)
        self.pending_web.refresh_from_db()
        self.assertEqual(self.pending_web.approval_status, CampaignRun.ApprovalStatus.APPROVED)

    def test_non_web_relabel_then_approve_still_binds(self) -> None:
        """Same two-step sequence against a non-web row: the widened lock is scoped to web
        rows and did not turn into a blanket freeze."""
        request = self._staff_request()
        form_class = self.admin_obj.get_form(request, obj=self.pending_csv_import, change=True)
        data = model_to_dict(self.pending_csv_import, exclude=['id'])
        data['source'] = CampaignRun.Source.LEGACY
        form = form_class(data=data, instance=self.pending_csv_import)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.pending_csv_import.refresh_from_db()
        self.assertEqual(self.pending_csv_import.source, CampaignRun.Source.LEGACY)

        self.client.force_login(self.superuser)
        with patch('solsys_code.campaign_views.resolve_site', return_value=(None, True)):
            response = self.client.post(
                reverse('campaigns:decide', kwargs={'pk': self.pending_csv_import.pk}), {'action': 'approve'}
            )
        self.assertEqual(response.status_code, 302)
        self.pending_csv_import.refresh_from_db()
        self.assertEqual(self.pending_csv_import.approval_status, CampaignRun.ApprovalStatus.APPROVED)
