from typing import Any
from urllib.parse import urlencode, urljoin

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError, CommandParser
from tom_observations.facilities.lco import LCOFacility
from tom_observations.facilities.ocs import make_request
from tom_observations.models import ObservationRecord
from tom_targets.models import TargetList


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


def _request_target_name(request: dict[str, Any]) -> str | None:
    """Return the target name of a request's first configuration that has one."""
    for configuration in request.get('configurations', []):
        target = configuration.get('target') or {}
        if target.get('name'):
            return target['name']
    return None


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
    member of that campaign is skipped and logged, never guessed at.
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

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Fetch matching RequestGroups and create ObservationRecords for their requests.

        Returns:
            str | None: a one-line summary of created/skipped counts.
        """
        proposal = options['proposal']
        name_prefix = options['name_prefix']
        dry_run = options['dry_run']

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
        skipped_existing = 0
        skipped_unmatched_target = 0
        skipped_no_config = 0

        for request_group in _matching_request_groups(facility, proposal, name_prefix):
            for request in request_group.get('requests', []):
                observation_id = str(request['id'])
                if ObservationRecord.objects.filter(facility=facility.name, observation_id=observation_id).exists():
                    skipped_existing += 1
                    continue

                target_name = _request_target_name(request)
                target = targets_by_name.get(target_name)
                if target is None:
                    self.stderr.write(
                        f'Skipping request {observation_id} (group {request_group.get("name")!r}): '
                        f'target {target_name!r} is not a member of campaign {campaign.name!r}.'
                    )
                    skipped_unmatched_target += 1
                    continue

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
                created += 1

        summary = (
            f'{"Would create" if dry_run else "Created"}: {created}, already existed: {skipped_existing}, '
            f'unmatched target: {skipped_unmatched_target}, no usable configuration: {skipped_no_config}'
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
