#!/usr/bin/env bash
# Refresh sorcha's cached MPC observatory codes (ObsCodes.json) and reload FOMO so its workers re-read them.
#
# sorcha downloads ObsCodes.json only when it is missing from its cache and never refreshes it, and
# solsys_code.ephem_utils reads it once at import, so new or corrected MPC sites only reach FOMO via this
# script. Meant for a monthly crontab entry, e.g.:
#   23 5 1 * * /path/to/fomo/deploy/refresh_obscodes.sh >> /var/log/fomo/refresh_obscodes.log 2>&1
set -o nounset -o errexit -o pipefail

URL=https://minorplanetcenter.net/Extended_Files/obscodes_extended.json.gz
CACHE_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/sorcha
JSON=$CACHE_DIR/ObsCodes.json
GZ=$CACHE_DIR/ObsCodes.json.gz
# A download missing any of these, or much smaller than the current file, is rejected as truncated or broken.
REQUIRED_CODES='500 X05 W84 I41'

log() { echo "$(date -Is) $*"; }

mkdir -p "$CACHE_DIR"
tmp=$(mktemp -d "$CACHE_DIR/.obscodes.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# --remote-time stores MPC's Last-Modified as the mtime, so --time-cond skips the download next time
# unless MPC has published a newer file.
time_cond=()
[ -f "$GZ" ] && time_cond=(--time-cond "$GZ")
curl --fail --silent --show-error --location --remote-time "${time_cond[@]}" \
    --connect-timeout 30 --max-time 300 --retry 3 --retry-delay 60 \
    --user-agent 'FOMO-TOM (https://github.com/lsst-sssc/fomo)' \
    --output "$tmp/ObsCodes.json.gz" "$URL"

if [ ! -s "$tmp/ObsCodes.json.gz" ]; then
    log "MPC's copy is not newer than $GZ; nothing to do."
    exit 0
fi

gunzip -c "$tmp/ObsCodes.json.gz" > "$tmp/ObsCodes.json"
# shellcheck disable=SC2086
python3 -I - "$tmp/ObsCodes.json" "$JSON" $REQUIRED_CODES <<'EOF'
import json
import os
import sys

new_path, old_path, *required = sys.argv[1:]
new = json.load(open(new_path))
old = json.load(open(old_path)) if os.path.exists(old_path) else {}
missing = [code for code in required if code not in new]
if not isinstance(new, dict) or missing or len(new) < 0.95 * len(old):
    sys.exit(f'Rejected download: {len(new)} codes (had {len(old)}), missing {missing}')
added = sorted(set(new) - set(old))
print(f'Validated: {len(new)} codes (had {len(old)}); new: {" ".join(added) or "none"}')
EOF

mv "$tmp/ObsCodes.json.gz" "$GZ"
if cmp -s "$tmp/ObsCodes.json" "$JSON"; then
    log "ObsCodes.json unchanged."
    exit 0
fi
[ -f "$JSON" ] && cp -p "$JSON" "$JSON.prev"
mv "$tmp/ObsCodes.json" "$JSON"
log "Installed new $JSON (previous copy kept as $JSON.prev)."

# cron provides no XDG_RUNTIME_DIR, which systemctl --user needs to reach the user manager (enabled at boot
# by lingering; see install_service.sh).
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
if systemctl --user --quiet is-active fomo; then
    systemctl --user reload fomo
    log "Reloaded fomo so its workers load the new codes."
fi
