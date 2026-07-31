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
"""

import re
from datetime import date, datetime
from datetime import timezone as dt_timezone

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
    """27.1-02 criterion 6, option (a): `source` becomes non-overwritable on an
    already-approved WEB run, and stays editable everywhere else (D-19 preserved)."""

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

    def test_source_withheld_only_on_approved_web(self) -> None:
        request = self._staff_request()
        self.assertIn('source', self.admin_obj.get_readonly_fields(request, self.approved_web))
        self.assertNotIn('source', self.admin_obj.get_readonly_fields(request, self.pending_web))
        self.assertNotIn('source', self.admin_obj.get_readonly_fields(request, self.rejected_web))
        self.assertNotIn('source', self.admin_obj.get_readonly_fields(request, self.approved_legacy))
        self.assertNotIn('source', self.admin_obj.get_readonly_fields(request, None))

    def test_approval_status_lock_survives_for_all_four(self) -> None:
        request = self._staff_request()
        for obj in (self.approved_web, self.pending_web, self.rejected_web, self.approved_legacy, None):
            self.assertIn('approval_status', self.admin_obj.get_readonly_fields(request, obj))

    def test_source_field_absent_from_generated_form_only_on_approved_web(self) -> None:
        request = self._staff_request()
        approved_form = self.admin_obj.get_form(request, obj=self.approved_web, change=True)
        pending_form = self.admin_obj.get_form(request, obj=self.pending_web, change=True)
        self.assertNotIn('source', approved_form.base_fields)
        self.assertIn('source', pending_form.base_fields)

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

    def test_submitted_source_value_does_bind_on_pending_web(self) -> None:
        """The lock is scoped, not blanket: the same write against a pending WEB row
        succeeds, proving D-19's correction use case survives."""
        request = self._staff_request()
        form_class = self.admin_obj.get_form(request, obj=self.pending_web, change=True)
        data = model_to_dict(self.pending_web, exclude=['id'])
        data['source'] = CampaignRun.Source.CSV_IMPORT
        form = form_class(data=data, instance=self.pending_web)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.pending_web.refresh_from_db()
        self.assertEqual(self.pending_web.source, CampaignRun.Source.CSV_IMPORT)

    def test_change_page_omits_source_field_only_on_approved_web(self) -> None:
        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.approved_web.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('name="source"', response.content.decode())

        response = self.client.get(reverse('admin:solsys_code_campaignrun_change', args=[self.pending_web.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="source"', response.content.decode())
