"""Management command to sync Gemini queue ObservationRecords to CalendarEvents."""

import logging
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from tom_calendar.models import CalendarEvent
from tom_observations.models import ObservationRecord

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Sync Gemini queue ObservationRecords to the FOMO calendar as CalendarEvents."""

    help = 'Sync Gemini queue ObservationRecords to CalendarEvents'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        pass

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Sync all GEM ObservationRecords to CalendarEvents.

        For each ObservationRecord with facility='GEM', derives CalendarEvent fields
        (telescope, instrument, proposal, title, window) from the record's parameters
        JSON and settings.FACILITIES['GEM']['programs'], then creates or updates the
        event idempotently using a no-churn get_or_create + update_fields idiom.

        Password key is stripped from parameters immediately at record load time (D-04)
        and never reaches stdout, stderr, or any CalendarEvent field (GEM-SECURE-01).

        Returns:
            None
        """
        records = ObservationRecord.objects.filter(facility='GEM')
        counters: dict[str, dict[str, int]] = {
            'GS': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
            'GN': {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0},
        }

        for record in records:
            # D-04: strip password immediately, before any logging or field derivation.
            safe_params = {k: v for k, v in (record.parameters or {}).items() if k != 'password'}

            # GEM-TELE-01: derive site and telescope name from program prefix.
            prog = safe_params.get('prog', '')
            if prog.startswith('GS-'):
                site_key = 'GS'
                telescope = 'Gemini South'
            elif prog.startswith('GN-'):
                site_key = 'GN'
                telescope = 'Gemini North'
            else:
                self.stderr.write(f'Unknown Gemini program prefix in {prog!r}; skipping ObservationRecord {record.pk}')
                counters.setdefault('UNKNOWN', {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped': 0})
                counters['UNKNOWN']['skipped'] += 1
                continue

            try:
                # D-03: use first obsid entry; warn when multiple are present.
                obsid_list = safe_params['obsid']
                if not obsid_list:
                    logger.warning(
                        'ObservationRecord pk=%s has empty obsid list — skipping',
                        record.pk,
                    )
                    counters[site_key]['skipped'] += 1
                    continue
                if len(obsid_list) > 1:
                    logger.warning(
                        'ObservationRecord pk=%s has multiple obsid entries: %r — using first entry only',
                        record.pk,
                        obsid_list,
                    )
                obs_code = obsid_list[0]

                # GEM-INSTR-01 / D-02: look up instrument description and ToO-type prefix from settings.
                gem_programs = settings.FACILITIES.get('GEM', {}).get('programs', {})
                description_str: str | None = gem_programs.get(prog, {}).get(obs_code)

                # Determine whether an explicit window is present (GEM-WINDOW-01).
                window_date = safe_params.get('windowDate')
                window_time_str = safe_params.get('windowTime')
                window_duration = safe_params.get('windowDuration')
                has_explicit_window = bool(window_date and window_time_str and window_duration)

                if description_str is not None:
                    # Strip the 'Std: ' or 'Rap: ' prefix to get the instrument label.
                    instrument = description_str.split(': ', 1)[1] if ': ' in description_str else description_str
                elif has_explicit_window:
                    # GEM-INSTR-01 raw fallback: explicit window present but obs code absent from settings.
                    # Use the raw obs code as the instrument label rather than skipping the record.
                    instrument = obs_code
                else:
                    # D-01: no explicit window and obs code absent from settings — ToO-type is unknowable.
                    # An event with unknown time bounds would be misleading; skip and count.
                    logger.warning(
                        "%r obs code %r not found in FACILITIES['GEM']['programs'] — skipping ObservationRecord %s",
                        prog,
                        obs_code,
                        record.pk,
                    )
                    counters[site_key]['skipped'] += 1
                    continue

                # Derive the observing window.
                if has_explicit_window:
                    # GEM-WINDOW-01: parse explicit date + time + duration from parameters.
                    start_dt = datetime.strptime(window_date, '%Y-%m-%d').replace(tzinfo=dt_timezone.utc)
                    time_dt = datetime.strptime(window_time_str, '%H:%M')
                    start_time = start_dt.replace(hour=time_dt.hour, minute=time_dt.minute)
                    end_time = start_time + timedelta(hours=int(window_duration))
                else:
                    # GEM-WINDOW-02: fall back to ToO-type prefix when no explicit window is present.
                    # description_str is guaranteed non-None here (raw fallback took the explicit-window branch).
                    if description_str.startswith('Rap:'):
                        start_time = record.created
                        end_time = record.created + timedelta(hours=24)
                    elif description_str.startswith('Std:'):
                        start_time = record.created + timedelta(hours=24)
                        end_time = record.created + timedelta(days=7)
                    else:
                        logger.warning(
                            'Unrecognised ToO-type prefix in %r for ObservationRecord %s — skipping',
                            description_str,
                            record.pk,
                        )
                        counters[site_key]['skipped'] += 1
                        continue

                # GEM-STATUS-01: prefix title with [ON_HOLD] when ready == 'false'.
                # Use str().lower() to handle both boolean False and string 'false' from the JSONField.
                ready = safe_params.get('ready', 'true')
                title_prefix = '[ON_HOLD] ' if str(ready).lower() == 'false' else ''
                title = f'{title_prefix}{telescope} {instrument} ToO'

                # GEM-KEY-01: stable, human-readable URL key.
                url = f'GEM:{prog}/{record.observation_id}'

                fields: dict[str, Any] = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'title': title,
                    'telescope': telescope,
                    'instrument': instrument,
                    'proposal': prog,
                }

                # GEM-NOCHURN-01: get_or_create on the url key, then compare each field
                # to detect changes; only call save when something actually changed.
                event, created_flag = CalendarEvent.objects.get_or_create(url=url, defaults=fields)
                if created_flag:
                    counters[site_key]['created'] += 1
                else:
                    changed = [f for f, v in fields.items() if getattr(event, f) != v]
                    if changed:
                        for f, v in fields.items():
                            setattr(event, f, v)
                        event.save(update_fields=changed)
                        counters[site_key]['updated'] += 1
                    else:
                        counters[site_key]['unchanged'] += 1

            except (KeyError, ValueError) as exc:
                # Never interpolate safe_params or record.parameters into this message (GEM-SECURE-01).
                self.stderr.write(f'Skipping observation_id={record.observation_id!r}: {exc}')
                counters[site_key]['skipped'] += 1
                continue

        # D-08: two-line per-site summary mirroring the LCO sync format.
        self.stdout.write(
            f'Gemini South: created: {counters["GS"]["created"]}, '
            f'updated: {counters["GS"]["updated"]}, '
            f'unchanged: {counters["GS"]["unchanged"]}, '
            f'skipped: {counters["GS"]["skipped"]}'
        )
        self.stdout.write(
            f'Gemini North: created: {counters["GN"]["created"]}, '
            f'updated: {counters["GN"]["updated"]}, '
            f'unchanged: {counters["GN"]["unchanged"]}, '
            f'skipped: {counters["GN"]["skipped"]}'
        )
        self.stdout.write('Done.')
        return None
