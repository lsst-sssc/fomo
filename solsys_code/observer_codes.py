"""JPL Horizons observer-notation alias table -- deliberately free of Django model imports.

WR-07: this module exists so ``calendar_utils.derive_telescope_class()`` (called once per
site-less row by the ``0011_backfill_campaignrun_telescope_class`` data migration) can reach
the alias table without importing ``campaign_utils``, which imports the live ``CampaignRun``
model at module scope. The previous arrangement deferred that import to inside
``derive_telescope_class()`` and claimed in a comment that this kept the live model out of a
data migration's import graph -- but a function-local import still executes at *call* time,
and the migration's very first ``500@``-prefixed row (JWST/HST/JUICE rows are exactly what it
targets) pulled ``solsys_code.models`` in mid-migration anyway. Keeping the table here makes
that claim true instead of merely stated.

Nothing in this module may import ``solsys_code.models``, ``django.db.models``, or any module
that does. It is a data table and nothing else.
"""

# Quick task 260726-fqb: JPL Horizons/SPICE observer notation (`500@<NAIF SPK ID>` --
# "geocentric observer at body N") names a spacecraft, not an MPC obscode --
# `.planning/PROJECT.md:120` records the operator-caught correction that `500@-170` is
# Horizons notation, and that `Observatory.obscode`'s `max_length=4` deliberately does
# NOT need widening to fit it. The real 3I/ATLAS campaign sheet carries `500@-170` in
# three `CampaignRun`s. Each entry below was verified on BOTH sides on 2026-07-26 --
# NAIF ID -> spacecraft via the JPL Horizons API (ssd.jpl.nasa.gov/api/horizons.api),
# obscode -> the same spacecraft via the MPC obscodes API. Extension rule: verify BOTH
# sides before adding a row -- never infer a mapping from the NAIF ID alone.
HORIZONS_OBSERVER_TO_OBSCODE: dict[str, str] = {
    '500@-170': '274',  # James Webb Space Telescope
    '500@-48': '250',  # Hubble Space Telescope
    '500@-163': 'C51',  # WISE Spacecraft
    '500@-95': 'C57',  # TESS
}
