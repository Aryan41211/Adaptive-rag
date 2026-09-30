#!/usr/bin/env bash
#
# Back up the Adaptive RAG data stores.
#
#   ./deploy/backup.sh [destination-directory]
#
# Writes a timestamped directory containing a MongoDB archive and one Qdrant
# snapshot per collection. Both stores support consistent online snapshots, so
# the stack keeps serving while this runs.
#
# Environment:
#   QDRANT_URL       Qdrant HTTP endpoint      (default http://localhost:6333)
#   QDRANT_API_KEY   Qdrant api key, if set    (default none)
#   COMPOSE          compose command           (default "docker compose")
#   MONGO_SERVICE    compose service name      (default mongo)
#   ENV_FILE         env file to read          (default ./.env)
#   MONGO_ROOT_USERNAME / MONGO_ROOT_PASSWORD
#                    MongoDB credentials. Read from ENV_FILE when not already
#                    set, so a scheduled run needs no extra configuration.
#   RETAIN           keep only the newest N backup directories under
#                    BACKUP_ROOT. Unset or 0 keeps every backup; pruning is
#                    opt-in because deleting backups by default is the wrong
#                    way round for a backup script to be wrong.
#
# Qdrant is reached over HTTP rather than through the container: its image
# ships no shell HTTP client. MongoDB is reached through the container,
# because mongodump has to run next to the server.
#
# Restore with deploy/restore.sh.

set -euo pipefail

BACKUP_ROOT="${1:-./backups}"
COMPOSE="${COMPOSE:-docker compose}"
MONGO_SERVICE="${MONGO_SERVICE:-mongo}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
ENV_FILE="${ENV_FILE:-./.env}"
RETAIN="${RETAIN:-0}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
destination="${BACKUP_ROOT}/${timestamp}"

log() { printf '  %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
size_of() { du -h "$1" 2>/dev/null | cut -f1; }

# Read one KEY=value out of the env file without sourcing it: sourcing would
# execute whatever happens to be in there.
env_file_value() {
    [ -f "${ENV_FILE}" ] || return 0
    sed -n "s/^[[:space:]]*$1=//p" "${ENV_FILE}" | tail -1 \
        | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

MONGO_ROOT_USERNAME="${MONGO_ROOT_USERNAME:-$(env_file_value MONGO_ROOT_USERNAME)}"
MONGO_ROOT_PASSWORD="${MONGO_ROOT_PASSWORD:-$(env_file_value MONGO_ROOT_PASSWORD)}"

# Built as an array so an empty credential set expands to nothing at all,
# rather than to empty string arguments that mongodump would reject.
mongo_auth_args=()
if [ -n "${MONGO_ROOT_USERNAME}" ] && [ -n "${MONGO_ROOT_PASSWORD}" ]; then
    mongo_auth_args=(
        -u "${MONGO_ROOT_USERNAME}"
        -p "${MONGO_ROOT_PASSWORD}"
        --authenticationDatabase admin
    )
fi

# Same reasoning for the Qdrant api key header.
qdrant_curl_args=(-fsS)
if [ -n "${QDRANT_API_KEY:-}" ]; then
    qdrant_curl_args+=(-H "api-key: ${QDRANT_API_KEY}")
fi

command -v curl >/dev/null || fail "curl is required"

mkdir -p "${destination}"
echo "Backing up to ${destination}"

# --- MongoDB ---------------------------------------------------------------
# --archive streams a single file to stdout, so nothing needs a writable path
# inside the container.
log "MongoDB: dumping"
${COMPOSE} exec -T "${MONGO_SERVICE}" \
    mongodump --archive --gzip --quiet "${mongo_auth_args[@]}" \
    > "${destination}/mongodb.archive.gz" \
    || fail "mongodump failed. Is the '${MONGO_SERVICE}' service running, and\
 are MONGO_ROOT_USERNAME / MONGO_ROOT_PASSWORD correct?"

# mongodump reports success on stdout it never wrote if the server rejected
# the connection mid-stream, so check that something actually landed.
[ -s "${destination}/mongodb.archive.gz" ] || fail "mongodump produced an empty archive"
log "MongoDB: $(size_of "${destination}/mongodb.archive.gz") written"

# --- Qdrant ----------------------------------------------------------------
# Per-collection snapshots, not a whole-storage one: this is the form Qdrant
# can restore through its API, without placing files on disk and restarting.
log "Qdrant: listing collections"
collections="$(curl "${qdrant_curl_args[@]}" "${QDRANT_URL}/collections" \
    | tr ',' '\n' | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || true)"

if [ -z "${collections}" ]; then
    log "Qdrant: no collections to back up"
else
    mkdir -p "${destination}/qdrant"
    for collection in ${collections}; do
        log "Qdrant: snapshotting ${collection}"

        snapshot="$(curl "${qdrant_curl_args[@]}" -X POST \
            "${QDRANT_URL}/collections/${collection}/snapshots" \
            | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)"
        [ -n "${snapshot}" ] || fail "could not snapshot ${collection}"

        curl "${qdrant_curl_args[@]}" \
            "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot}" \
            -o "${destination}/qdrant/${collection}.snapshot" \
            || fail "could not download the snapshot of ${collection}"

        # The snapshot also stays inside the container; drop it so repeated
        # backups do not fill the volume.
        curl "${qdrant_curl_args[@]}" -X DELETE \
            "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot}" \
            >/dev/null || true

        log "Qdrant: $(size_of "${destination}/qdrant/${collection}.snapshot") written"
    done
fi

# --- manifest --------------------------------------------------------------
{
    echo "Adaptive RAG backup"
    echo "created_utc=${timestamp}"
    echo "mongodb_archive=mongodb.archive.gz"
    echo "qdrant_collections=${collections:-none}"
} > "${destination}/manifest.txt"

# --- retention -------------------------------------------------------------
# Runs only after the new backup is complete, so a failed run can never be the
# reason an old one was deleted. Matches the timestamp layout this script
# writes, so nothing else living under BACKUP_ROOT is ever considered.
if [ "${RETAIN}" -gt 0 ] 2>/dev/null; then
    echo
    echo "Retention: keeping the newest ${RETAIN}"
    existing="$(find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d \
        -name '????????T??????Z' | sort)"
    total="$(printf '%s\n' "${existing}" | grep -c . || true)"
    if [ "${total}" -gt "${RETAIN}" ]; then
        printf '%s\n' "${existing}" | head -n "$((total - RETAIN))" \
        | while read -r old; do
            [ -n "${old}" ] || continue
            # Never prune the backup this run just wrote.
            [ "${old}" = "${destination}" ] && continue
            log "removing ${old}"
            rm -rf -- "${old}"
        done
    else
        log "${total} backup(s) present, nothing to remove"
    fi
fi

echo
echo "Backup complete: ${destination}"
echo "Restore with: ./deploy/restore.sh ${destination}"
