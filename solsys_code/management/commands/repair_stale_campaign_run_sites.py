"""One-time data repair (D-16): re-resolve stale, site-less approved CampaignRuns.

Several approved CampaignRun rows are site-less not because their site is genuinely
unresolvable, but because they were imported before the JPL Horizons observer-notation
alias table (``observer_codes.HORIZONS_OBSERVER_TO_OBSCODE``) landed on 2026-07-26. This
one-off command re-runs the real site-resolution path (``resolve_site()``) against every
approved, site-less row, so those rows get a genuine chance to resolve before Phase 27's
``telescope_class`` backfill migration operates on them.

This command deliberately does NOT touch ``approval_status``, ``run_status``, the
observing window fields, or ``target`` -- only ``site``, ``site_needs_review``, and
(conditionally) ``site_raw`` are ever written. It also never creates or updates a calendar
event: reconciling site-repaired runs onto the calendar is Phase 29's reconciler, out of
scope here (a repaired-but-unprojected run is the expected post-repair state, not a bug).

D-06 (26-CONTEXT.md:94): a candidate row that already carries a ``telescope_class`` is
permanently site-less by design -- the class IS the answer to "why is there no site", not a
resolution failure -- so there is nothing for this command to repair. Such rows are skipped
entirely (site, site_raw, and site_needs_review all left untouched) and reported under their
own ``skipped_class_wide`` counter, distinct from ``skipped_no_site_code``.

D-22: every ``resolve_site()`` call below passes ``create_placeholder=False`` (the
function's own default is ``True``) -- a live tier-2 MPC network failure must leave the row
site-less and flagged for review rather than fabricating a placeholder Observatory for a
site like HST. This is the fail-safe choice: a bad network day produces no row, not a fake
one.

D-16a: this command's real (non-``--dry-run``) run calls the FULL ``resolve_site()`` path,
including its live tier-2 MPC Obscodes API lookup. Its result therefore depends on a
reachable third-party network and is not reproducible offline or in CI -- the offline test
suite below mocks ``requests.get`` for every tier-2 scenario, and the live run is a separate,
manually-verified one-time operation (see the phase summary for its recorded outcome).

D-16b: for the one row known to need it, an owner-supplied ``site_raw`` correction is
applied before resolution (see ``_OWNER_SUPPLIED_SITE_RAW`` below) -- a domain-authority
value from the project owner, not an inference, and it does not generalise to another
database.

``--dry-run`` performs read-only checks and writes nothing: no ``CampaignRun.save()`` and no
``Observatory`` row is ever created under ``--dry-run``. Because a genuine tier-2 MPC lookup
creates an ``Observatory`` row as a side effect of resolution, ``--dry-run`` never calls
``resolve_site()`` itself -- see ``_probe_site_resolution()``'s docstring for the resulting
limitation.
"""

import logging
import re
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from solsys_code.campaign_utils import resolve_site
from solsys_code.models import CampaignRun
from solsys_code.observer_codes import HORIZONS_OBSERVER_TO_OBSCODE
from solsys_code.solsys_code_observatory.models import Observatory

logger = logging.getLogger(__name__)

# D-16b: this value is supplied by the project owner as domain authority for the one known
# stale Swift row (pk=13 on the dev DB) -- it is not inferred from any data in this database,
# and it does NOT generalise to another database. Applied only when a row's own site_raw is
# blank; a non-blank site_raw is never overwritten.
_OWNER_SUPPLIED_SITE_RAW: dict[str, str] = {'swift': 'C52'}

# Computed from the field itself (not hardcoded), mirroring campaign_utils._MAX_OBSCODE_LEN's
# own comment, so a future schema change can't silently desync this dry-run-only guard.
_MAX_OBSCODE_LEN = Observatory._meta.get_field('obscode').max_length


def _first_instrument_token(telescope_instrument: str) -> str:
    """Return the lowercased first whitespace-or-'/'-delimited token of telescope_instrument.

    Used only to decide whether the D-16b owner-supplied site_raw correction applies (e.g.
    ``'Swift/UVOT'`` -> ``'swift'``). Never raises for blank/None input.

    Args:
        telescope_instrument: the row's raw telescope/instrument text.

    Returns:
        str: the lowercased first token, or ``''`` for blank input.
    """
    text = (telescope_instrument or '').strip().lower()
    if not text:
        return ''
    return re.split(r'[\s/]+', text, maxsplit=1)[0]


def _probe_site_resolution(site_raw: str) -> str:
    """Read-only tier-1-only probe used under ``--dry-run``: never queries MPC, never writes.

    Calling ``resolve_site()`` itself under ``--dry-run`` would still reach a live tier-2 MPC
    lookup for any code with a genuine tier-1 miss, and a successful tier-2 hit creates an
    ``Observatory`` row as a side effect of resolution -- exactly what ``--dry-run`` must
    never do. This mirrors ``resolve_site()``'s alias-translation and length-guard steps,
    then stops at a tier-1 existence check. A row that would need a live tier-2 lookup is
    reported as "would query MPC" rather than resolved -- this is a known, documented
    limitation of ``--dry-run`` (see this command's ``--help`` text).

    Args:
        site_raw: the (possibly D-16b-corrected) site code text to probe.

    Returns:
        str: a human-readable description of the intended action for the operator.
    """
    code = (site_raw or '').strip()
    translated = HORIZONS_OBSERVER_TO_OBSCODE.get(code, code)
    if len(translated) > _MAX_OBSCODE_LEN:
        return f'site_raw={site_raw!r}: unresolvable (code too long) -- would be flagged for review, no MPC query'
    if Observatory.objects.filter(obscode=translated).exists():
        return f'site_raw={site_raw!r}: would resolve OFFLINE to existing Observatory {translated!r}'
    return f'site_raw={site_raw!r}: would query MPC (live tier-2 lookup needed)'


class Command(BaseCommand):
    """One-off repair: re-resolve every approved, site-less CampaignRun (D-16)."""

    help = (
        'One-time data repair (D-16): re-resolve every approved, site-less CampaignRun '
        'through the real resolve_site() path, including its live MPC tier-2 lookup, so '
        'rows imported before the Horizons alias table landed (2026-07-26) get a genuine '
        'chance to resolve. Never fabricates a placeholder Observatory on a tier-2 network '
        'failure (create_placeholder=False, D-22) -- the row is left site-less and flagged '
        'instead. Never touches approval_status, run_status, the observing window, or '
        'target, and never projects a calendar event (Phase 29 reconciler scope). '
        '--dry-run performs only a tier-1 existence check and cannot predict the outcome of '
        'a live tier-2 MPC lookup -- rows that would need one are reported as "would query '
        'MPC" rather than resolved, and no CampaignRun or Observatory row is written. '
        'A candidate row that already carries a telescope_class is skipped entirely (D-06: '
        'the class is a permanent answer to "why is there no site", not a resolution '
        'failure) and reported under its own skipped_class_wide counter.'
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Parse command line arguments."""
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help=(
                'Report the intended action for each row without writing any CampaignRun '
                'or Observatory changes. Limitation: this only performs a tier-1 existence '
                'check -- a row that would need a live tier-2 MPC lookup is reported as '
                '"would query MPC", not resolved.'
            ),
        )
        # No return statement — BaseCommand.add_arguments() returns None

    def handle(self, *args: Any, **options: Any) -> str | None:
        """Re-resolve every approved, site-less CampaignRun's site.

        Returns:
            str | None: None on completion.
        """
        dry_run = options['dry_run']

        candidates = CampaignRun.objects.filter(
            approval_status=CampaignRun.ApprovalStatus.APPROVED,
            site__isnull=True,
        ).order_by('pk')

        resolved_count = 0
        still_flagged_count = 0
        skipped_no_site_code = 0
        skipped_class_wide = 0

        for run in candidates:
            # D-06 (26-CONTEXT.md:94): telescope_class is a permanent "why is there no site"
            # fact, not a resolution failure -- a class-carrying row is site-less by design,
            # so there is no site to repair. Skip it entirely (site, site_raw, and
            # site_needs_review all untouched), mirroring the skipped_no_site_code pattern
            # below with its own counter.
            if run.telescope_class:
                skipped_class_wide += 1
                logger.info(
                    'pk=%s: carries telescope_class=%r -- permanently site-less by design, skipped',
                    run.pk,
                    run.telescope_class,
                )
                self.stdout.write(
                    f'pk={run.pk}: skipped (class-wide/space run; telescope_class={run.telescope_class!r})'
                )
                continue

            site_raw = (run.site_raw or '').strip()
            site_raw_changed = False

            if not site_raw:
                # D-16b: owner-supplied correction, applied only when site_raw is blank.
                owner_value = _OWNER_SUPPLIED_SITE_RAW.get(_first_instrument_token(run.telescope_instrument))
                if owner_value:
                    site_raw = owner_value
                    site_raw_changed = True

            if not site_raw:
                # resolve_site('') returns (None, True) immediately with no tier running --
                # calling it would be a pointless no-op. Count and log separately so the
                # operator can see which rows still need a site code supplied.
                skipped_no_site_code += 1
                logger.info(
                    'pk=%s: no site code available (telescope_instrument=%r) -- skipped',
                    run.pk,
                    run.telescope_instrument,
                )
                self.stdout.write(
                    f'pk={run.pk}: skipped (no site code; telescope_instrument={run.telescope_instrument!r})'
                )
                continue

            if dry_run:
                description = _probe_site_resolution(site_raw)
                suffix = ' [site_raw would be set via D-16b owner-supplied correction]' if site_raw_changed else ''
                self.stdout.write(f'pk={run.pk}: {description}{suffix}')
                continue

            # D-22: create_placeholder=False -- resolve_site()'s own default is True, which
            # on a tier-2 network failure would fabricate a placeholder Observatory for HST.
            # False leaves the row site-less and flagged instead (the fail-safe choice).
            site, needs_review = resolve_site(site_raw, create_placeholder=False)

            # D-17: no special-case flag clearing -- site_needs_review is set only from
            # resolve_site()'s own return value.
            run.site = site
            run.site_needs_review = needs_review
            update_fields = ['site', 'site_needs_review']
            if site_raw_changed:
                run.site_raw = site_raw
                update_fields.append('site_raw')
            run.save(update_fields=update_fields)

            # WR-04: `site is not None` alone is NOT "resolved". resolve_site() deliberately
            # returns needs_review=True when its tier-1 hit is itself a `NEEDS REVIEW: `
            # placeholder Observatory (campaign_utils.is_placeholder_observatory), so keying
            # the summary on the site object alone would log "resolved to Observatory 'XXX'"
            # for a row this command just wrote with site_needs_review=True -- over-reporting
            # success and under-reporting the operator's remaining work queue. Since the
            # candidate filter is site__isnull=True, such a row is excluded from every future
            # run of this command, so this summary line is its last word: it must be honest.
            if site is not None and not needs_review:
                resolved_count += 1
                logger.info('pk=%s: resolved to Observatory %r', run.pk, site.obscode)
            else:
                still_flagged_count += 1
                logger.info(
                    'pk=%s: still needs review (site=%r, site_raw=%r), flagged for review',
                    run.pk,
                    site.obscode if site else None,
                    site_raw,
                )

            self.stdout.write(f'pk={run.pk}: site={site.obscode if site else None}, site_needs_review={needs_review}')

        if dry_run:
            self.stdout.write(
                f'Done (dry run). candidates: {len(candidates)}, skipped_no_site_code: {skipped_no_site_code}, '
                f'skipped_class_wide: {skipped_class_wide}'
            )
        else:
            self.stdout.write(
                f'Done. candidates: {len(candidates)}, resolved: {resolved_count}, '
                f'still_flagged: {still_flagged_count}, skipped_no_site_code: {skipped_no_site_code}, '
                f'skipped_class_wide: {skipped_class_wide}'
            )
        return None
