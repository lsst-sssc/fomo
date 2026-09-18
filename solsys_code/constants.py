"""Shared literal defaults for FOMO's unattended operation path (Phase 36, D-01/D-04).

IN-34 (36-REVIEW.md): deliberately a leaf module with no project-local imports. Both
``solsys_code/unattended.py`` (the tick runner -- importing it pulls in the full runner
graph: ``campaign_reconciler``, ``observation_projector``, ``backfill_lco_observations``,
``project_observation_calendar``, ``telescope_runs`` (astropy), and both facility classes)
and ``solsys_code/management/commands/check_unattended.py`` (a read-only preflight whose
whole contract is "reports, does not fix", D-05) need these same three literals without
either depending on the other's import graph. Before this module existed,
``check_unattended.py`` imported them from ``unattended.py`` itself (IN-22, 36-REVIEW.md
iteration 5), which worked (nothing in that graph currently reaches ``ephem_utils``, the
1.6 GB SPICE path) but left the coupling one accidental import away in any of the five
modules ``unattended.py`` pulls in. Owning the literals here removes that coupling
entirely -- neither command needs to import the other's module for these.
"""

# IN-14 (36-REVIEW.md): mirrors settings.py's own os.getenv(..., <default>) defaults for
# FOMO_LOCK_DIR/FOMO_LOG_FILE -- a hand-edited local_settings.py deriving one of these from
# an unset environment variable with no default of its own yields None (the same WR-07
# hazard FOMO_BASE_URL already guards against), and Path(None) raises TypeError from the
# very first thing run_tick() does. These are a last-resort fallback for that
# misconfiguration, not a substitute for settings.py's own defaults.
DEFAULT_LOCK_DIR = '/var/lock/fomo'
DEFAULT_LOG_FILE = '/var/log/fomo/unattended.log'

# D-04/WR-16/WR-36 (36-REVIEW.md): the single source for the cron schedule's own interval.
# unattended.py derives its _CRON_TICK_INTERVAL threshold from this, and
# check_unattended.py's cron_line() `*/{CRON_INTERVAL_MINUTES}` schedule and
# check_heartbeat()'s reminder text both read it directly, so the "15" the runbook and
# crontab template also document cannot drift between the two modules' own copies the way
# WR-36 found it already had.
CRON_INTERVAL_MINUTES = 15
