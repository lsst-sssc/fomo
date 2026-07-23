from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError, CommandParser
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.ocs import make_request
from tom_observations.models import ObservationRecord
from tom_targets.models import Target, TargetList


def _matching_request_groups(facility: LCOFacility, proposal: str, name_prefix: str):
    """Page through GET /api/requestgroups/ for a proposal, yielding name-prefix matches.

    The 'proposal' filter is an exact match; 'name' is icontains server-side, so it's
    passed as a pre-filter to cut payload size, then re-checked client-side with a real
    str.startswith() since icontains would also match the prefix appearing mid-string.

    Args:
        facility: an LCOFacility instance (for portal_url/api_key settings and headers).
        proposal: LCO proposal code, exact match.
        name_prefix: only RequestGroups whose 'name' starts with this string are yielded.

    Yields:
        dict: each matching RequestGroup object (with its nested 'requests' list).
    """
    query = urlencode({'proposal': proposal, 'name': name_prefix, 'limit': 100})
    url = urljoin(facility.facility_settings.get_setting('portal_url'), f'/api/requestgroups/?{query}')
    while url:
        response = make_request('GET', url, headers=facility._portal_headers())
        payload = response.json()
        for request_group in payload.get('results', []):
            if request_group.get('name', '').startswith(name_prefix):
                yield request_group
        url = payload.get('next')


@dataclass
class RequestTargetInfo:
    """The target name, coordinates, epoch, proper motion, and parallax read from a
    request's first named-target configuration."""

    name: str
    ra: float | None
    dec: float | None
    epoch: float | None
    pm_ra: float | None
    pm_dec: float | None
    parallax: float | None


def _request_target_info(request: dict[str, Any]) -> RequestTargetInfo | None:
    """Return the name/coordinates/epoch/proper motion/parallax of a request's first
    configuration that has a named target.

    Args:
        request: a single request from request_group['requests'].

    Returns:
        RequestTargetInfo | None: name/ra/dec/epoch/pm_ra/pm_dec/parallax for the first
            configuration whose 'target' dict has a name, or None if no configuration has a
            named target. 'ra'/'dec' are float degrees as returned by the LCO API, or None if
            absent (e.g. non-sidereal targets that carry orbital elements instead of
            coordinates). 'epoch', 'pm_ra' (from the LCO wire key 'proper_motion_ra'),
            'pm_dec' (from 'proper_motion_dec'), and 'parallax' are likewise None if the
            request's target dict omits them.
    """
    for configuration in request.get('configurations', []):
        target = configuration.get('target') or {}
        if target.get('name'):
            return RequestTargetInfo(
                name=target['name'],
                ra=target.get('ra'),
                dec=target.get('dec'),
                epoch=target.get('epoch'),
                pm_ra=target.get('proper_motion_ra'),
                pm_dec=target.get('proper_motion_dec'),
                parallax=target.get('parallax'),
            )
    return None


def _request_target_name(request: dict[str, Any]) -> str | None:
    """Return the target name of a request's first configuration that has one."""
    info = _request_target_info(request)
    return info.name if info else None


def _build_parameters(request_group: dict[str, Any], request: dict[str, Any]) -> dict[str, Any] | None:
    """Build a minimal flat ObservationRecord.parameters dict for a backfilled request.

    Matches the legacy single-config flat shape ('proposal', 'start', 'end',
    'instrument_type') that solsys_code.calendar_utils._extract_instrument() falls back
    to when no c_N_*-prefixed multi-configuration keys are present -- deliberately not
    replicating the full submission-form c_N_* shape, since the RequestGroup API's
    'configurations' entries don't map 1:1 onto it.

    Args:
        request_group: the parent RequestGroup object (for 'proposal').
        request: a single request from request_group['requests'] (for 'windows' and the
            first configuration with an 'instrument_type').

    Returns:
        dict[str, Any] | None: the parameters dict, or None if the request has no
            configuration with a usable instrument_type.
    """
    for configuration in request.get('configurations', []):
        instrument_type = configuration.get('instrument_type')
        if not instrument_type:
            continue
        parameters = {
            'proposal': request_group.get('proposal'),
            'instrument_type': instrument_type,
        }
        windows = request.get('windows') or []
        if windows:
            if windows[0].get('start'):
                parameters['start'] = windows[0]['start']
            if windows[0].get('end'):
                parameters['end'] = windows[0]['end']
        return parameters
    return None


class Command(BaseCommand):
    """Backfill ObservationRecords for LCO RequestGroups submitted outside the TOM.

    Queries the LCO Observation Portal's 'Get All RequestGroups' API
    (GET /api/requestgroups/) for a proposal, keeps only RequestGroups whose name
    starts with --name-prefix, and creates one ObservationRecord per child request
    (mirroring LCORedirectFacility.request_id_to_group's per-request granularity) --
    skipping any request that already has an ObservationRecord (facility='LCO',
    observation_id=<request id>).

    Each request's target is matched by name against the Targets already belonging to
    a chosen campaign (a tom_targets.TargetList) -- a request whose target name isn't a
    member of that campaign is skipped and logged, never guessed at, unless
    --create-missing-targets is passed. With that flag set, an unmatched request instead
    gets a SIDEREAL field Target created from the request's own RA/Dec (create the Target
    if missing, otherwise reuse the existing Target of that name found anywhere in FOMO),
    added to the campaign, and is then processed normally. Combined with --dry-run, it
    reports what would be created/reused and added without writing anything.

    Immediately after a newly created ObservationRecord is saved (non-dry-run only), the
    command makes a live best-effort call to facility.update_observation_status(observation_id)
    -- the same TOM Toolkit method periodic polling uses -- so the new record's status,
    scheduled_start, and scheduled_end are populated from LCO right away instead of staying
    unset until the next poll. A failure of that call is skip-and-logged (never fatal, never
    rolls back the already-created record) and counted in the status_sync_failed summary count.
    """

    help = 'Backfill ObservationRecords from LCO RequestGroups submitted directly at the LCO portal'

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--proposal',
            required=True,
            help='LCO proposal code to filter RequestGroups by (exact match).',
        )
        parser.add_argument(
            '--name-prefix',
            required=True,
            help="Only backfill RequestGroups whose 'name' starts with this prefix.",
        )
        parser.add_argument(
            '--campaign',
            required=False,
            help=(
                'Name of the campaign (tom_targets.TargetList) to match request targets against. '
                'If omitted, you will be prompted to choose one interactively.'
            ),
        )
        parser.add_argument(
            '--username',
            required=False,
            help='Attribute created ObservationRecords to this username (default: unattributed).',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would be created without writing any ObservationRecord rows.',
        )
        parser.add_argument(
            '--create-missing-targets',
            action='store_true',
            help=(
                'For a request whose target name is not a campaign member, auto-create a SIDEREAL '
                "Target from the request's RA/Dec (reusing an existing Target of that name if one "
                'exists anywhere in FOMO), add it to the campaign, then process the request '
                'normally instead of skipping it. Default off. Combined with --dry-run, reports '
                'what would be created/reused and added without writing anything.'
            ),
        )

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Fetch matching RequestGroups and create ObservationRecords for their requests.

        Returns:
            str | None: a one-line summary of created/skipped counts.
        """
        proposal = options['proposal']
        name_prefix = options['name_prefix']
        dry_run = options['dry_run']
        create_missing_targets = options['create_missing_targets']

        user = None
        if options.get('username'):
            try:
                user = get_user_model().objects.get(username=options['username'])
            except get_user_model().DoesNotExist as exc:
                raise CommandError(f'Invalid username: {options["username"]!r}') from exc

        campaign = self._resolve_campaign(options.get('campaign'))
        targets_by_name = {target.name: target for target in campaign.targets.all()}
        if not targets_by_name:
            raise CommandError(f'Campaign {campaign.name!r} has no targets to match requests against.')

        facility = LCOFacility()
        facility.set_user(user)

        created = 0
        created_targets = 0
        skipped_existing = 0
        skipped_unmatched_target = 0
        skipped_no_config = 0
        status_sync_failed = 0

        for request_group in _matching_request_groups(facility, proposal, name_prefix):
            for request in request_group.get('requests', []):
                observation_id = str(request['id'])
                if ObservationRecord.objects.filter(facility=facility.name, observation_id=observation_id).exists():
                    skipped_existing += 1
                    continue

                target_info = _request_target_info(request)
                target_name = target_info.name if target_info else None
                target = targets_by_name.get(target_name)
                if target is None:
                    if not create_missing_targets:
                        self.stderr.write(
                            f'Skipping request {observation_id} (group {request_group.get("name")!r}): '
                            f'target {target_name!r} is not a member of campaign {campaign.name!r}.'
                        )
                        skipped_unmatched_target += 1
                        continue

                    target, is_new_target = self._resolve_or_build_field_target(target_info)
                    if dry_run:
                        self.stdout.write(
                            f'Would {"create" if is_new_target else "reuse"} field Target {target.name!r} '
                            f'and add it to campaign {campaign.name!r}.'
                        )
                    else:
                        if is_new_target:
                            target.save()
                            created_targets += 1
                        campaign.targets.add(target)
                        targets_by_name[target.name] = target

                parameters = _build_parameters(request_group, request)
                if parameters is None:
                    self.stderr.write(
                        f'Skipping request {observation_id}: no configuration with an instrument_type found.'
                    )
                    skipped_no_config += 1
                    continue

                if dry_run:
                    self.stdout.write(
                        f'Would create ObservationRecord: target={target.name!r}, observation_id={observation_id}, '
                        f'status={request.get("state", "")!r}'
                    )
                else:
                    ObservationRecord.objects.create(
                        target=target,
                        user=user,
                        facility=facility.name,
                        observation_id=observation_id,
                        status=request.get('state', ''),
                        parameters=parameters,
                    )
                    try:
                        facility.update_observation_status(observation_id)
                    except Exception as exc:
                        self.stderr.write(f'Failed to refresh status for observation_id={observation_id!r}: {exc}')
                        status_sync_failed += 1
                created += 1

        summary = (
            f'{"Would create" if dry_run else "Created"}: {created}, already existed: {skipped_existing}, '
            f'unmatched target: {skipped_unmatched_target}, no usable configuration: {skipped_no_config}, '
            f'created field targets: {created_targets}, status sync failed: {status_sync_failed}'
        )
        self.stdout.write(summary)
        return summary

    def _resolve_campaign(self, campaign_name: str | None) -> TargetList:
        """Resolve --campaign to a TargetList, prompting interactively if not given.

        Args:
            campaign_name: the --campaign value, or None to prompt.

        Returns:
            TargetList: the resolved campaign.

        Raises:
            CommandError: no TargetList by that name exists, more than one does, no
                TargetLists exist at all, or an interactive selection was invalid.
        """
        if campaign_name:
            matches = TargetList.objects.filter(name=campaign_name)
            if not matches.exists():
                raise CommandError(f'No campaign (TargetList) named {campaign_name!r} found.')
            if matches.count() > 1:
                raise CommandError(f'Multiple campaigns (TargetLists) named {campaign_name!r} found.')
            return matches.first()

        target_lists = list(TargetList.objects.order_by('name'))
        if not target_lists:
            raise CommandError('No campaigns (TargetLists) exist to select from.')
        self.stdout.write('Available campaigns:')
        for index, target_list in enumerate(target_lists, start=1):
            self.stdout.write(f'  {index}. {target_list.name} ({target_list.targets.count()} targets)')
        choice = input('Select a campaign by number: ').strip()
        try:
            selected_index = int(choice) - 1
            if selected_index < 0:
                raise ValueError
            return target_lists[selected_index]
        except (ValueError, IndexError) as exc:
            raise CommandError(f'Invalid selection: {choice!r}') from exc

    def _resolve_or_build_field_target(self, target_info: RequestTargetInfo | None) -> tuple[Target, bool]:
        """Find or build a SIDEREAL field Target for an unmatched --create-missing-targets request.

        Reuses an existing Target of that name found anywhere in FOMO (create the Target if
        missing, otherwise reuse it in place); the caller is responsible for saving a newly
        built Target and adding it to the campaign (or skipping both in --dry-run).

        Args:
            target_info: the name/ra/dec read from the request's target configuration.

        Returns:
            tuple[Target, bool]: the existing-or-new Target, and whether it is newly built
                (True) and not yet saved, versus reused and already persisted (False).
        """
        field_name = target_info.name
        existing = Target.objects.filter(name=field_name).first()
        if existing is not None:
            return existing, False
        return (
            Target(
                name=field_name,
                type=Target.SIDEREAL,
                ra=target_info.ra,
                dec=target_info.dec,
                epoch=target_info.epoch,
                pm_ra=target_info.pm_ra,
                pm_dec=target_info.pm_dec,
                parallax=target_info.parallax,
            ),
            True,
        )
