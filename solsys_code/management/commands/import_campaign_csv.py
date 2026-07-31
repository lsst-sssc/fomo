import csv
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser
from tom_targets.models import TargetList

from solsys_code.calendar_utils import derive_telescope_class
from solsys_code.campaign_utils import (
    insert_or_create_campaign_run,
    map_observation_status,
    parse_obs_window,
    resolve_site,
)
from solsys_code.models import CampaignRun

# WR-09: the D-05 natural-key columns. If the CSV's header doesn't include these exactly
# (e.g. a renamed column in a future sheet export), every row would otherwise be silently
# skipped one-by-one with no single top-level diagnostic that the header shape is wrong.
_REQUIRED_HEADERS = ('Telescope / Instrument', 'Obs. Date', 'UT Time Range')

# The real 3I/ATLAS sheet export prepends a free-text attribution row and an entirely
# blank row before the real header, so the header isn't always row 1. Cap the leading-row
# scan so a genuinely malformed/wrong file (no header anywhere) fails fast instead of
# scanning the whole file.
_MAX_HEADER_SCAN = 10


class Command(BaseCommand):
    """Bootstrap-import a campaign coordination CSV (e.g. the 3I/ATLAS sheet) into CampaignRun rows."""

    help = (
        'Bootstrap-import a campaign coordination CSV into CampaignRun rows (CAMP-04). '
        "WARNING: re-running this command over the same campaign always resets each row's "
        '`target` to the auto-resolved value (D-07) -- any manual correction a staff user made '
        'to `target` after a previous import will be silently overwritten on re-import (WR-07). '
        'A row already carrying source=web (created by the public submission form) keeps its own '
        'source and approval_status on re-import (WR-01); all its other fields are still '
        'overwritten from the CSV. A row whose site is already resolved keeps its site, '
        "site_raw and site_needs_review when the CSV's Site Code cell does not resolve, so a "
        'repair made by repair_stale_campaign_run_sites cannot be silently reverted by a '
        're-import; a non-blank telescope_class is never blanked by a re-import either.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            'filepath',
            type=str,
            help='Path to the campaign coordination CSV file',
        )
        parser.add_argument(
            '--campaign',
            type=str,
            required=True,
            help='Campaign TargetList name (found-or-created, D-06)',
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Import campaign CSV rows into CampaignRun, row-by-row, skip-and-log on natural-key failure.

        Only a blank Telescope / Instrument is a true natural-key failure that skips a row
        (D-07); every other column defaults to a blank/None value rather than aborting the
        row. `Obs. Date` never skips a row either, per D-13's never-raise contract:
        `parse_obs_window()` always returns a usable window/TBD result, so every row
        creates or updates a `CampaignRun` -- a resolved single-night/range window, or a
        flagged TBD row (`window_needs_review=True`, counted in the summary, IMPORT-02).
        Site resolution (D-08/D-09) never skips a row either -- an unresolved site is
        flagged via `site_needs_review` and counted separately.

        The natural key branches on whether the row resolved to a window or TBD
        (Pitfall 2, matching `CampaignRun.Meta.constraints`'s two partial
        `UniqueConstraint`s exactly): a resolved window keys on `(campaign,
        telescope_instrument, window_start, window_end)`; a TBD row keys on `(campaign,
        telescope_instrument, contact_person)` instead, since `window_start`/`window_end`
        are always `NULL` for a TBD row. A genuine same-key collision within this batch is
        logged and skipped rather than silently merged into one `CampaignRun`.

        WR-07: `fields['target']` is unconditionally set to the campaign's auto-resolved
        Target (D-07) on every row, every run -- including on a re-import that updates an
        existing row. This is expected/acceptable for this bootstrap-import command (not
        a bug), but it does mean a staff user's manual `CampaignRun.target` correction
        made via the admin between imports will be reset back to the auto-resolved value
        the next time this command runs over the same campaign.

        WR-01: `source` and `approval_status` are the one exception to that
        "re-import overwrites everything" rule. A row already carrying
        `source=WEB` -- i.e. one created by the public submission form -- keeps its own
        `source` AND `approval_status` on re-import; every other field is still
        overwritten as above. Without that carve-out a CSV row colliding on the natural
        key would rewrite an unreviewed public submission to
        `source=CSV_IMPORT, approval_status=APPROVED`, which under CANON-01's derivation
        rule (`APPROVED + source != WEB` == "no approval was required") makes it
        indistinguishable from vetted backfill and immediately publicly visible.

        Returns:
            str | None: None on completion.
        """
        filepath = options['filepath']
        campaign, _ = TargetList.objects.get_or_create(name=options['campaign'])

        # D-07: single-target campaigns auto-assign that Target to every imported row.
        auto_target = campaign.targets.first() if campaign.targets.count() == 1 else None

        created_count = 0
        updated_count = 0
        unchanged_count = 0
        skipped_count = 0
        site_needs_review_count = 0
        window_needs_review_count = 0
        # Two distinct key shapes (Pitfall 2): a resolved window key
        # (campaign_pk, telescope_instrument, window_start, window_end), or a TBD key
        # (campaign_pk, telescope_instrument, contact_person). Track keys already seen in
        # this batch so a genuine duplicate is logged and skipped rather than silently
        # merged into one CampaignRun via insert_or_create_campaign_run's get_or_create.
        seen_window_keys: set[tuple[Any, ...]] = set()

        try:
            with open(filepath, encoding='utf-8', newline='') as f:
                lines = f.readlines()
        except OSError as exc:
            raise CommandError(f'Cannot open campaign CSV {filepath!r}: {exc}') from exc

        # Scan up to _MAX_HEADER_SCAN leading rows for the real header (see the constant's
        # comment) rather than assuming row 1 is the header -- fail fast (WR-09) if none of
        # the scanned rows contains every required column.
        header_idx = None
        for idx, parsed_row in enumerate(csv.reader(lines[:_MAX_HEADER_SCAN])):
            if all(h in parsed_row for h in _REQUIRED_HEADERS):
                header_idx = idx
                break

        if header_idx is None:
            raise CommandError(
                f'Campaign CSV {filepath!r}: no header row containing all required column(s) '
                f'{_REQUIRED_HEADERS!r} was found within the first {_MAX_HEADER_SCAN} rows.'
            )

        reader = csv.DictReader(lines[header_idx:])
        rows = list(reader)

        # Header is at file line header_idx + 1, first data row at header_idx + 2.
        for row_num, row in enumerate(rows, start=header_idx + 2):
            telescope_instrument = (row.get('Telescope / Instrument', '') or '').strip()
            if not telescope_instrument:
                # D-07: the one remaining true natural-key failure -- WR-06: log only the
                # natural-key fields needed to diagnose the skip, not the full row (which
                # also carries Contact Person/Email PII from the real 3I/ATLAS sheet).
                self.stderr.write(
                    f'Row {row_num}: Telescope / Instrument is required and was blank '
                    f'(Obs. Date={row.get("Obs. Date")!r})'
                )
                skipped_count += 1
                continue

            # D-13: parse_obs_window() never raises -- every Obs. Date shape resolves to
            # either a window (single-night or range) or the TBD tuple.
            (
                window_start,
                window_end,
                original_obs_date_raw,
                window_needs_review,
                _ut_start,
                _ut_end,
                ut_needs_review,
            ) = parse_obs_window(row.get('Obs. Date', ''), row.get('UT Time Range', ''))
            if window_needs_review:
                window_needs_review_count += 1

            contact_person = row.get('Contact Person', '') or ''

            # Pitfall 2: branch the natural key on whether this row resolved to a window
            # or fell through to TBD -- matches CampaignRun.Meta.constraints' two partial
            # UniqueConstraints exactly (resolved: campaign+telescope_instrument+
            # window_start+window_end; TBD: campaign+telescope_instrument+contact_person).
            if window_start is not None:
                collision_key = (campaign.pk, telescope_instrument, window_start, window_end)
            else:
                collision_key = (campaign.pk, telescope_instrument, contact_person)

            if collision_key in seen_window_keys:
                self.stderr.write(
                    f'Row {row_num}: WARNING duplicate natural key '
                    f'(Telescope/Instrument={telescope_instrument!r}, '
                    f'Obs. Date={row.get("Obs. Date")!r}); '
                    f'skipping row to avoid merging distinct observations into one CampaignRun'
                    + (' (unparseable/blank UT Time Range)' if ut_needs_review else '')
                )
                skipped_count += 1
                continue
            seen_window_keys.add(collision_key)

            site_raw = row.get('Site Code', '') or ''
            site, site_resolution_failed = resolve_site(site_raw)

            # Pitfall 2: branch the natural key on whether this row resolved to a window
            # or fell through to TBD -- matches CampaignRun.Meta.constraints' two partial
            # UniqueConstraints exactly (resolved: campaign+telescope_instrument+
            # window_start+window_end; TBD: campaign+telescope_instrument+contact_person).
            # CR-01: built HERE, before `fields`, rather than after it. The existing row this
            # finds is what decides whether this row will actually end up site-less, and that
            # in turn gates the telescope_class derivation immediately below.
            if window_start is not None:
                # Resolved-window branch: contact_person is a plain field, not part of
                # the key -- it goes into `fields` below instead.
                lookup = {
                    'campaign': campaign,
                    'telescope_instrument': telescope_instrument,
                    'window_start': window_start,
                    'window_end': window_end,
                }
            else:
                # TBD branch (Pitfall 2): contact_person is promoted into the lookup key
                # instead, so it's deliberately left out of `fields` to avoid
                # lookup/defaults key-overlap ambiguity.
                lookup = {
                    'campaign': campaign,
                    'telescope_instrument': telescope_instrument,
                    'contact_person': contact_person,
                    'window_start__isnull': True,
                }

            existing = CampaignRun.objects.filter(**lookup).first()

            # CR-01: the site-preservation guard (applied to `fields` further down, where its
            # full rationale lives) can decide this row keeps its ALREADY-resolved site, in
            # which case the CSV's own failed resolution is not this row's effective site at
            # all. The decision is computed here, up front, so telescope_class can gate on the
            # EFFECTIVE post-guard site rather than on the CSV's fresh result.
            preserve_site = (
                existing is not None and existing.site_id is not None and (site is None or site_resolution_failed)
            )

            # D-06 (26-CONTEXT.md:94): telescope_class records WHY there is no site -- it is
            # a permanent, correct campaign-level fact, not a placeholder cleared once a site
            # is known. D-20: the shared derivation helper's second required call site (the
            # 0011 backfill migration is the first). Derivation still gates on "no resolved
            # site", mirroring the backfill's site__isnull=True gate -- but a non-blank class,
            # once derived, is never cleared, and a row that has one is not a resolution
            # failure.
            #
            # CR-01: "no resolved site" means the site this row ENDS UP with, which is why
            # `preserve_site` is part of the gate. A preserved row keeps its existing resolved
            # site, so there is no "why is there no site" question for a class to answer, and
            # models.py's stated invariant -- "telescope_class is never inferred for a run
            # whose site DID resolve" -- would otherwise be violated permanently, since no
            # writer ever clears the field again.
            telescope_class = (
                derive_telescope_class(site_raw=site_raw, telescope_instrument=telescope_instrument)
                if site is None and not preserve_site
                else ''
            )
            # `site_resolution_failed` alone is not "needs review": resolve_site() here runs
            # with its default create_placeholder=True, so it can return a placeholder
            # Observatory (site is not None) with site_resolution_failed True -- that row
            # genuinely still needs staff review. Deliberately NOT the literal
            # `site is None and not telescope_class`: telescope_class is only ever non-blank
            # when site is None, so the two forms agree wherever the class can fire, but only
            # this form preserves the placeholder case, which the literal form would silently
            # unflag.
            needs_review = site_resolution_failed and not telescope_class

            fields = {
                # WR-07: unconditionally reset to auto_target on every run, including
                # re-imports -- see handle()'s docstring for why this is expected.
                'target': auto_target,
                'site': site,
                'site_raw': site_raw,
                'site_needs_review': needs_review,
                'original_obs_date_raw': original_obs_date_raw,  # D-04: TBD rows only, '' otherwise
                'window_needs_review': window_needs_review,
                'filters_bandpass': row.get('Filter(s)/Bandpass', '') or '',
                'observation_details': row.get('Observation Details', '') or '',
                'weather': row.get('Weather conditions or forecast', '') or '',
                'run_status': map_observation_status(row.get('Observation Status', '')),
                'approval_status': CampaignRun.ApprovalStatus.APPROVED,  # D-03: bootstrap rows are vetted backfill
                # CANON-01: this is the importer's real behaviour change -- approval_status
                # already wrote APPROVED before this phase and is deliberately unchanged
                # (26-DECISION Criterion 1: APPROVED + source != WEB means "no approval was
                # required", a different fact from "a human approved this").
                'source': CampaignRun.Source.CSV_IMPORT,
                'telescope_class': telescope_class,
                'observation_outcome': row.get('Observation Outcome', '') or '',
                'publication_plans': row.get('Publication Plans', '') or '',
                'open_to_collaboration': (row.get('Open to collaboration?', '') or '').strip().lower() == 'yes',
                'contact_email': row.get('Email', '') or '',
                'comments': row.get('Other comments', '') or '',
            }

            if window_start is not None:
                # Resolved-window branch: contact_person is a plain field, not part of the
                # lookup key (which was built above), so it belongs in `fields`. On the TBD
                # branch it is part of the key instead and is deliberately left out of
                # `fields` to avoid lookup/defaults key-overlap ambiguity.
                fields['contact_person'] = contact_person

            # WR-01/CANON-01: never relabel a run that came in through the public web form.
            # insert_or_create_campaign_run() setattr's every key in `fields` onto a matched
            # row, so without this guard a CSV row colliding on the natural key with a
            # WEB-sourced submission (entirely plausible -- the sheet and the form describe
            # the same runs) would rewrite it to source=CSV_IMPORT, approval_status=APPROVED.
            # That is not merely a lost label: per the derivation rule on CampaignRun.Source,
            # `APPROVED + source != WEB` reads as "no approval was required", so an unreviewed
            # public submission would become indistinguishable from vetted backfill AND
            # publicly visible (CampaignRunTableView only excludes PENDING_REVIEW). Both keys
            # must be preserved together -- keeping `source` while still forcing APPROVED
            # would publish the unreviewed row anyway.
            if existing is not None and existing.source == CampaignRun.Source.WEB:
                fields.pop('source', None)
                fields.pop('approval_status', None)

            # WR-01 (criterion 5): a re-import must not silently revert a site that
            # repair_stale_campaign_run_sites already fixed. When the existing row already
            # carries a resolved site (existing.site_id is not None) and the CSV's own Site
            # Code cell did NOT genuinely resolve this time -- either resolve_site() returned
            # no Observatory at all (site is None) or it returned a tier-3 placeholder
            # (site_resolution_failed True, which this call site can hit because it runs with
            # its default create_placeholder=True) -- drop site, site_raw and
            # site_needs_review from fields as a unit. They are preserved together, never
            # individually: keeping site while reverting site_raw would leave the row's
            # recorded provenance contradicting its resolved site, and
            # repair_stale_campaign_run_sites deliberately corrects site_raw too (D-16b), so
            # site_raw is exactly as repairable as site. A CSV cell that DOES genuinely
            # resolve still wins -- this guard only blocks the case that produced the stale
            # row in the first place: an unresolvable cell trying to overwrite something
            # better. The condition itself is computed further up as `preserve_site`, because
            # the telescope_class derivation has to gate on the same decision (CR-01).
            if preserve_site:
                fields.pop('site', None)
                fields.pop('site_raw', None)
                fields.pop('site_needs_review', None)

            # telescope_class is NEVER cleared by any writer once set
            # (solsys_code/models.py:207-219) -- Phase 27 code-review finding CR-01 proposed
            # clearing it here on site resolution and the user REJECTED CR-01
            # (27-REVIEW-FIX.md). Without this pop a re-import whose Site Code cell resolves
            # would write telescope_class='' over a non-blank value (see the
            # `telescope_class = ... if site is None else ''` computation above).
            if existing is not None and existing.telescope_class and not telescope_class:
                fields.pop('telescope_class', None)

            # WR-04 (criterion 5): the summary must report only flags this command actually
            # wrote. If the site-preservation guard above popped `site_needs_review` from
            # `fields`, the value it would have written is moot -- count it only when it is
            # still present, i.e. still going to be written (as a create, or as an update
            # where the guard did not fire).
            if 'site_needs_review' in fields and needs_review:
                site_needs_review_count += 1

            run, action = insert_or_create_campaign_run(lookup, fields)
            if action == 'created':
                created_count += 1
            elif action == 'updated':
                updated_count += 1
            else:
                unchanged_count += 1

        self.stdout.write(
            f'Done. created: {created_count}, '
            f'updated: {updated_count}, '
            f'unchanged: {unchanged_count}, '
            f'skipped: {skipped_count}, '
            f'site_needs_review: {site_needs_review_count}, '
            f'window_needs_review: {window_needs_review_count}'
        )
        return
