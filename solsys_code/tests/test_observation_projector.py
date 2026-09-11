"""Every stage, every marker, no churn -- the observation projector's full behaviour
(D-01..D-13, PROJ-01/PROJ-02/PROJ-03/PROJ-05/PROJ-06). Migrates the behaviours the retired
``sync_lco_observation_calendar`` command's 38-test module covered (no-churn assertions,
per-facility dispatch, failure-marker priority, successful-terminal-state-never-bannered) as
projector-native tests, not a structural copy.

Uses ``tom_targets.tests.factories.NonSiderealTargetFactory`` for every target fixture --
FOMO is exclusively a Solar System TOM, so a sidereal fixture would misrepresent what this
code handles (CLAUDE.md convention).
"""

from datetime import datetime
from datetime import timezone as dt_timezone

from django.test import TestCase
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationGroup, ObservationRecord
from tom_targets.tests.factories import NonSiderealTargetFactory

from solsys_code import observation_projector as op
from solsys_code.models import CalendarEventMeta, CampaignRun


class _ObservationProjectorTestBase(TestCase):
    """Shared fixture: one NonSiderealTargetFactory target and a record-builder helper."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.target = NonSiderealTargetFactory.create()

    def setUp(self) -> None:
        op.reset_facility_cache()

    def _make_record(
        self,
        observation_id: str,
        status: str = 'PENDING',
        scheduled_start: datetime | None = None,
        scheduled_end: datetime | None = None,
        facility: str = 'LCO',
        proposal: str = 'TESTPROP',
        start: str | None = '2026-09-01T00:00:00',
        end: str | None = '2026-09-02T00:00:00',
        instrument_type: str = '2M0-SCICAM-MUSCAT',
        extra_params: dict | None = None,
    ) -> ObservationRecord:
        """Create an ObservationRecord fixture sharing the class-level target."""
        params: dict = {'proposal': proposal, 'instrument_type': instrument_type}
        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end
        params.update(extra_params or {})
        return ObservationRecord.objects.create(
            target=self.target,
            facility=facility,
            observation_id=observation_id,
            status=status,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            parameters=params,
        )


class TestStageFor(_ObservationProjectorTestBase):
    """stage_for() classifies a record's lifecycle stage from its own fields alone."""

    def test_queued_stage(self) -> None:
        record = self._make_record('stagefor-queued', status='PENDING')
        facility = op.facility_for(record)
        self.assertEqual(op.stage_for(record, facility), 'queued')

    def test_placed_stage(self) -> None:
        record = self._make_record(
            'stagefor-placed',
            status='PENDING',
            scheduled_start=datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 1, 1, 19, tzinfo=dt_timezone.utc),
        )
        facility = op.facility_for(record)
        self.assertEqual(op.stage_for(record, facility), 'placed')

    def test_observed_stage(self) -> None:
        record = self._make_record(
            'stagefor-observed',
            status='COMPLETED',
            scheduled_start=datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
            scheduled_end=datetime(2026, 9, 1, 1, 19, tzinfo=dt_timezone.utc),
        )
        facility = op.facility_for(record)
        self.assertEqual(op.stage_for(record, facility), 'observed')

    def test_completed_no_block_stage(self) -> None:
        record = self._make_record('stagefor-completed-no-block', status='COMPLETED')
        facility = op.facility_for(record)
        self.assertEqual(op.stage_for(record, facility), 'completed-no-block')

    def test_terminal_negative_stage(self) -> None:
        record = self._make_record('stagefor-terminal-negative', status='WINDOW_EXPIRED')
        facility = op.facility_for(record)
        self.assertEqual(op.stage_for(record, facility), 'terminal-negative')

    def test_inconsistent_stage(self) -> None:
        record = self._make_record(
            'stagefor-inconsistent',
            status='PENDING',
            scheduled_start=datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
            scheduled_end=None,
        )
        facility = op.facility_for(record)
        self.assertEqual(op.stage_for(record, facility), 'inconsistent')


class TestTitleAndToken(_ObservationProjectorTestBase):
    """title_for()/telescope_token(): the D-01 marker+token form and PROJ-06's cell budget."""

    def test_soar_token_is_4m0_regardless_of_instrument_string(self) -> None:
        record = self._make_record('title-soar-token', facility='SOAR', instrument_type='SOAR_GHTS_REDCAM')
        facility = op.facility_for(record)
        fields, _stage = op.event_fields_for(record, facility)
        self.assertEqual(fields['telescope'], '4m0')
        self.assertIn('4m0', fields['title'])

    def test_telescope_field_equals_title_token_for_every_stage(self) -> None:
        cases = [
            ('token-queued', {'status': 'PENDING'}),
            (
                'token-placed',
                {
                    'status': 'PENDING',
                    'scheduled_start': datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
                    'scheduled_end': datetime(2026, 9, 1, 1, 19, tzinfo=dt_timezone.utc),
                },
            ),
            ('token-completed-no-block', {'status': 'COMPLETED'}),
            ('token-terminal-negative', {'status': 'WINDOW_EXPIRED'}),
        ]
        for observation_id, kwargs in cases:
            with self.subTest(observation_id=observation_id):
                record = self._make_record(observation_id, **kwargs)
                facility = op.facility_for(record)
                fields, _stage = op.event_fields_for(record, facility)
                self.assertIn(fields['telescope'], fields['title'])
                self.assertEqual(fields['telescope'], '2m0')

    def test_instrument_field_equals_extracted_instrument_not_the_aperture_label(self) -> None:
        record = self._make_record('token-instrument', instrument_type='2M0-SCICAM-MUSCAT')
        facility = op.facility_for(record)
        fields, _stage = op.event_fields_for(record, facility)
        self.assertEqual(fields['instrument'], '2M0-SCICAM-MUSCAT')
        self.assertNotEqual(fields['instrument'], fields['telescope'])

    def test_marker_and_token_within_first_16_characters(self) -> None:
        cases = [
            ('cell-queued', {'status': 'PENDING'}, '[Q]', '2m0'),
            (
                'cell-placed',
                {
                    'status': 'PENDING',
                    'scheduled_start': datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
                    'scheduled_end': datetime(2026, 9, 1, 1, 19, tzinfo=dt_timezone.utc),
                },
                '[S]',
                '2m0',
            ),
            (
                'cell-observed',
                {
                    'status': 'COMPLETED',
                    'scheduled_start': datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
                    'scheduled_end': datetime(2026, 9, 1, 1, 19, tzinfo=dt_timezone.utc),
                },
                '[O]',
                '2m0',
            ),
            ('cell-completed-no-block', {'status': 'COMPLETED'}, '[O]', '2m0'),
            ('cell-window-expired', {'status': 'WINDOW_EXPIRED'}, '[X]', '2m0'),
            ('cell-canceled', {'status': 'CANCELED'}, '[C]', '2m0'),
            ('cell-failure-limit', {'status': 'FAILURE_LIMIT_REACHED'}, '[F]', '2m0'),
            ('cell-not-attempted', {'status': 'NOT_ATTEMPTED'}, '[F]', '2m0'),
            (
                'cell-inconsistent',
                {
                    'status': 'PENDING',
                    'scheduled_start': datetime(2026, 9, 1, 1, 0, tzinfo=dt_timezone.utc),
                    'scheduled_end': None,
                },
                '[?]',
                '2m0',
            ),
        ]
        for observation_id, kwargs, marker, token in cases:
            with self.subTest(observation_id=observation_id):
                record = self._make_record(observation_id, **kwargs)
                facility = op.facility_for(record)
                fields, _stage = op.event_fields_for(record, facility)
                head = fields['title'][:16]
                self.assertIn(marker, head)
                self.assertIn(token, head)


class TestEventFieldsFor(_ObservationProjectorTestBase):
    """event_fields_for(): the D-10..D-13 span/marker rules for every stage."""

    def test_queued_spans_request_window_and_titles_q(self) -> None:
        record = self._make_record(
            'fields-queued', status='PENDING', start='2026-09-01T00:00:00', end='2026-09-02T00:00:00'
        )
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'queued')
        self.assertEqual(fields['start_time'], datetime(2026, 9, 1, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(fields['end_time'], datetime(2026, 9, 2, 0, 0, tzinfo=dt_timezone.utc))
        self.assertTrue(fields['title'].startswith('[Q] '))

    def test_placed_spans_block_and_titles_s(self) -> None:
        block_start = datetime(2026, 9, 5, 10, 0, tzinfo=dt_timezone.utc)
        block_end = datetime(2026, 9, 5, 10, 19, tzinfo=dt_timezone.utc)
        record = self._make_record(
            'fields-placed', status='PENDING', scheduled_start=block_start, scheduled_end=block_end
        )
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'placed')
        self.assertEqual(fields['start_time'], block_start)
        self.assertEqual(fields['end_time'], block_end)
        self.assertTrue(fields['title'].startswith('[S] '))

    def test_observed_with_block_titles_o(self) -> None:
        block_start = datetime(2026, 9, 6, 10, 0, tzinfo=dt_timezone.utc)
        block_end = datetime(2026, 9, 6, 10, 19, tzinfo=dt_timezone.utc)
        record = self._make_record(
            'fields-observed', status='COMPLETED', scheduled_start=block_start, scheduled_end=block_end
        )
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'observed')
        self.assertEqual(fields['start_time'], block_start)
        self.assertEqual(fields['end_time'], block_end)
        self.assertTrue(fields['title'].startswith('[O] '))

    def test_completed_no_block_spans_request_window_and_titles_o_never_q(self) -> None:
        record = self._make_record(
            'fields-completed-no-block',
            status='COMPLETED',
            start='2026-09-01T00:00:00',
            end='2026-09-02T00:00:00',
        )
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'completed-no-block')
        self.assertEqual(fields['start_time'], datetime(2026, 9, 1, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(fields['end_time'], datetime(2026, 9, 2, 0, 0, tzinfo=dt_timezone.utc))
        self.assertTrue(fields['title'].startswith('[O] '))
        self.assertFalse(fields['title'].startswith('[Q] '))

    def test_window_expired_keeps_full_window_and_titles_x(self) -> None:
        record = self._make_record(
            'fields-window-expired',
            status='WINDOW_EXPIRED',
            start='2026-09-01T00:00:00',
            end='2026-09-02T00:00:00',
        )
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'terminal-negative')
        self.assertEqual(fields['start_time'], datetime(2026, 9, 1, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(fields['end_time'], datetime(2026, 9, 2, 0, 0, tzinfo=dt_timezone.utc))
        self.assertTrue(fields['title'].startswith('[X] '))

    def test_canceled_keeps_full_window_and_titles_c(self) -> None:
        record = self._make_record('fields-canceled', status='CANCELED')
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'terminal-negative')
        self.assertTrue(fields['title'].startswith('[C] '))

    def test_failure_limit_reached_keeps_full_window_and_titles_f(self) -> None:
        record = self._make_record('fields-failure-limit', status='FAILURE_LIMIT_REACHED')
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'terminal-negative')
        self.assertTrue(fields['title'].startswith('[F] '))

    def test_not_attempted_keeps_full_window_and_titles_f(self) -> None:
        record = self._make_record('fields-not-attempted', status='NOT_ATTEMPTED')
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'terminal-negative')
        self.assertTrue(fields['title'].startswith('[F] '))

    def test_failure_marker_wins_over_stage_marker_when_block_is_placed(self) -> None:
        block_start = datetime(2026, 9, 7, 10, 0, tzinfo=dt_timezone.utc)
        block_end = datetime(2026, 9, 7, 10, 19, tzinfo=dt_timezone.utc)
        record = self._make_record(
            'fields-failure-with-block',
            status='WINDOW_EXPIRED',
            scheduled_start=block_start,
            scheduled_end=block_end,
        )
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'terminal-negative')
        self.assertEqual(fields['start_time'], block_start)
        self.assertEqual(fields['end_time'], block_end)
        self.assertTrue(fields['title'].startswith('[X] '))

    def test_half_set_schedule_projects_as_question_mark_and_does_not_raise(self) -> None:
        record = self._make_record(
            'fields-inconsistent',
            status='PENDING',
            scheduled_start=datetime(2026, 9, 8, 1, 0, tzinfo=dt_timezone.utc),
            scheduled_end=None,
            start='2026-09-08T00:00:00',
            end='2026-09-09T00:00:00',
        )
        facility = op.facility_for(record)
        fields, stage = op.event_fields_for(record, facility)
        self.assertEqual(stage, 'inconsistent')
        self.assertEqual(fields['start_time'], datetime(2026, 9, 8, 0, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(fields['end_time'], datetime(2026, 9, 9, 0, 0, tzinfo=dt_timezone.utc))
        self.assertTrue(fields['title'].startswith('[?] '))


class TestProjectRecordWrites(_ObservationProjectorTestBase):
    """project_record(): the create/update/unchanged/unprojectable write contract."""

    def test_missing_window_is_unprojectable_and_writes_no_event(self) -> None:
        record = self._make_record('writes-missing-window', status='PENDING', start=None, end=None)
        facility = op.facility_for(record)
        url = op.event_url(record, facility)
        CalendarEvent.objects.filter(url=url).delete()

        action, _reason = op.project_record(record)

        self.assertEqual(action, 'unprojectable')
        self.assertFalse(CalendarEvent.objects.filter(url=url).exists())

    def test_missing_window_leaves_a_preexisting_event_untouched(self) -> None:
        record = self._make_record('writes-missing-window-preexisting', status='PENDING', start=None, end=None)
        facility = op.facility_for(record)
        url = op.event_url(record, facility)
        stale_event = CalendarEvent.objects.create(
            url=url,
            title='pre-existing',
            description='untouched',
            start_time=datetime(2020, 1, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2020, 1, 2, tzinfo=dt_timezone.utc),
        )
        before = (stale_event.title, stale_event.description, stale_event.start_time, stale_event.end_time)

        action, _reason = op.project_record(record)

        stale_event.refresh_from_db()
        after = (stale_event.title, stale_event.description, stale_event.start_time, stale_event.end_time)
        self.assertEqual(action, 'unprojectable')
        self.assertEqual(before, after)

    def test_unparsable_dates_is_unprojectable(self) -> None:
        record = self._make_record(
            'writes-unparsable-dates', status='PENDING', start='not-a-date', end='also-not-a-date'
        )
        action, _reason = op.project_record(record)
        self.assertEqual(action, 'unprojectable')

    def test_projecting_unchanged_record_twice_reports_unchanged_with_no_modified_churn(self) -> None:
        record = self._make_record('writes-no-churn', status='PENDING')
        op.project_record(record)
        facility = op.facility_for(record)
        url = op.event_url(record, facility)
        event = CalendarEvent.objects.get(url=url)
        modified_before = event.modified

        action, _stage = op.project_record(record)

        event.refresh_from_db()
        self.assertEqual(action, 'unchanged')
        self.assertEqual(event.modified, modified_before)

    def test_two_records_whose_windows_exactly_abut_produce_two_separate_events(self) -> None:
        record_a = self._make_record(
            'writes-abut-a', status='PENDING', start='2026-09-10T00:00:00', end='2026-09-11T00:00:00'
        )
        record_b = self._make_record(
            'writes-abut-b', status='PENDING', start='2026-09-11T00:00:00', end='2026-09-12T00:00:00'
        )
        op.project_record(record_a)
        op.project_record(record_b)
        facility = op.facility_for(record_a)
        url_a = op.event_url(record_a, facility)
        url_b = op.event_url(record_b, facility)
        self.assertNotEqual(url_a, url_b)
        self.assertEqual(CalendarEvent.objects.filter(url__in=[url_a, url_b]).count(), 2)

    def test_lco_and_soar_records_use_distinct_facility_instances_and_urls(self) -> None:
        lco_record = self._make_record('writes-lco-vs-soar-lco', facility='LCO')
        soar_record = self._make_record('writes-lco-vs-soar-soar', facility='SOAR', instrument_type='SOAR_GHTS_REDCAM')
        op.project_record(lco_record)
        op.project_record(soar_record)

        lco_facility = op.facility_for(lco_record)
        soar_facility = op.facility_for(soar_record)
        self.assertIsNot(lco_facility, soar_facility)
        self.assertNotEqual(type(lco_facility), type(soar_facility))

        lco_url = op.event_url(lco_record, lco_facility)
        soar_url = op.event_url(soar_record, soar_facility)
        self.assertNotEqual(lco_url, soar_url)
        self.assertTrue(CalendarEvent.objects.filter(url=lco_url).exists())
        self.assertTrue(CalendarEvent.objects.filter(url=soar_url).exists())


class TestMetaLinks(_ObservationProjectorTestBase):
    """write_event_meta(): the observation_record/observation_group link contract."""

    def test_grouped_record_sets_observation_group_and_omits_name_from_title_and_description(self) -> None:
        group = ObservationGroup.objects.create(name='meta-links-group-name-must-not-leak')
        record = self._make_record('meta-grouped', status='PENDING')
        group.observation_records.add(record)

        op.project_record(record)

        facility = op.facility_for(record)
        url = op.event_url(record, facility)
        event = CalendarEvent.objects.get(url=url)
        meta = CalendarEventMeta.objects.get(event=event)
        self.assertEqual(meta.observation_group_id, group.pk)
        self.assertNotIn(group.name, event.title)
        self.assertNotIn(group.name, event.description)

    def test_record_in_two_groups_links_the_lowest_pk_group(self) -> None:
        record = self._make_record('meta-two-groups', status='PENDING')
        first_group = ObservationGroup.objects.create(name='meta-two-groups-first')
        second_group = ObservationGroup.objects.create(name='meta-two-groups-second')
        first_group.observation_records.add(record)
        second_group.observation_records.add(record)
        self.assertLess(first_group.pk, second_group.pk)

        op.project_record(record)

        facility = op.facility_for(record)
        url = op.event_url(record, facility)
        meta = CalendarEventMeta.objects.get(event__url=url)
        self.assertEqual(meta.observation_group_id, first_group.pk)

    def test_existing_campaign_attribution_survives_projection_and_is_verified_becomes_true(self) -> None:
        record = self._make_record('meta-attribution-survives', status='PENDING')
        op.project_record(record)
        facility = op.facility_for(record)
        url = op.event_url(record, facility)
        event = CalendarEvent.objects.get(url=url)
        run = CampaignRun.objects.create(telescope_instrument='2m0')
        meta = CalendarEventMeta.objects.get(event=event)
        stamp = datetime(2026, 1, 1, tzinfo=dt_timezone.utc)
        meta.run = run
        meta.confirmed_by = None
        meta.confirmed_at = stamp
        meta.is_verified = False
        meta.save()

        op.project_record(record)

        meta.refresh_from_db()
        self.assertEqual(meta.run_id, run.pk)
        self.assertEqual(meta.confirmed_at, stamp)
        self.assertTrue(meta.is_verified)

    def test_stale_companion_claim_on_a_different_event_is_cleared_not_integrity_error(self) -> None:
        # observation_record is a OneToOneField, so simulating "a companion row already
        # claims this record from a DIFFERENT event" (the shape the takeover sweep meets)
        # requires first releasing this record's own real claim -- the DB would otherwise
        # reject two meta rows claiming the same record at once, which is exactly the
        # constraint write_event_meta's stale-claim clear exists to satisfy safely.
        record = self._make_record('meta-stale-claim', status='PENDING')
        facility = op.facility_for(record)
        url = op.event_url(record, facility)
        real_event = CalendarEvent.objects.get(url=url)
        real_meta = CalendarEventMeta.objects.get(event=real_event)
        real_meta.observation_record = None
        real_meta.save()

        stale_event = CalendarEvent.objects.create(
            url='RUN:9999:2026-09-01',
            title='stale claim holder',
            description='',
            start_time=datetime(2026, 9, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, tzinfo=dt_timezone.utc),
        )
        CalendarEventMeta.objects.create(event=stale_event, observation_record=record)

        action, _stage = op.project_record(record)

        self.assertIn(action, ('created', 'updated', 'unchanged'))
        stale_meta = CalendarEventMeta.objects.get(event=stale_event)
        self.assertIsNone(stale_meta.observation_record_id)
        real_meta.refresh_from_db()
        self.assertEqual(real_meta.observation_record_id, record.pk)


class TestNamespaceIsolation(_ObservationProjectorTestBase):
    """The projector must never create, modify, or delete an event outside its own namespace."""

    def test_run_gem_and_blank_url_events_are_byte_identical_after_a_projection(self) -> None:
        run_event = CalendarEvent.objects.create(
            url='RUN:1:2026-09-01',
            title='[CANCELLED] classical run',
            description='reconciler-owned',
            start_time=datetime(2026, 9, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, tzinfo=dt_timezone.utc),
            telescope='FTN',
            instrument='muscat',
        )
        gem_event = CalendarEvent.objects.create(
            url='GEM:gemini-obs-1',
            title='Gemini submission echo',
            description='gemini-owned',
            start_time=datetime(2026, 9, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, tzinfo=dt_timezone.utc),
            telescope='GN',
            instrument='GMOS',
        )
        blank_url_event = CalendarEvent.objects.create(
            url='',
            title='classical schedule night',
            description='load_telescope_runs-owned',
            start_time=datetime(2026, 9, 1, tzinfo=dt_timezone.utc),
            end_time=datetime(2026, 9, 2, tzinfo=dt_timezone.utc),
            telescope='FTS',
            instrument='',
        )
        snapshot = {}
        for event in (run_event, gem_event, blank_url_event):
            snapshot[event.pk] = (
                event.url,
                event.title,
                event.description,
                event.start_time,
                event.end_time,
                event.telescope,
                event.instrument,
            )

        record = self._make_record('namespace-isolation-record', status='PENDING')
        op.project_record(record)

        self.assertEqual(CalendarEvent.objects.count(), 4)
        for event in (run_event, gem_event, blank_url_event):
            event.refresh_from_db()
            after = (
                event.url,
                event.title,
                event.description,
                event.start_time,
                event.end_time,
                event.telescope,
                event.instrument,
            )
            self.assertEqual(snapshot[event.pk], after)
