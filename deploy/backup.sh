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
#   QDRANT_URL      Qdrant HTTP endpoint      (default http://localhost:6333)
#   COMPOSE         compose command           (default "docker compose")
#   MONGO_SERVICE   compose service name      (default mongo)
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

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
destination="${BACKUP_ROOT}/${timestamp}"

log() { printf '  %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
size_of() { du -h "$1" 2>/dev/null | cut -f1; }

command -v curl >/dev/null || fail "curl is required"

mkdir -p "${destination}"
echo "Backing up to ${destination}"

# --- MongoDB ---------------------------------------------------------------
# --archive streams a single file to stdout, so nothing needs a writable path
# inside the container.
log "MongoDB: dumping"
${COMPOSE} exec -T "${MONGO_SERVICE}" \
    mongodump --archive --gzip --quiet > "${destination}/mongodb.archive.gz" \
    || fail "mongodump failed. Is the '${MONGO_SERVICE}' service running?"
log "MongoDB: $(size_of "${destination}/mongodb.archive.gz") written"

# --- Qdrant ----------------------------------------------------------------
# Per-collection snapshots, not a whole-storage one: this is the form Qdrant
# can restore through its API, without placing files on disk and restarting.
log "Qdrant: listing collections"
collections="$(curl -fsS "${QDRANT_URL}/collections" \
    | tr ',' '\n' | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || true)"

if [ -z "${collections}" ]; then
    log "Qdrant: no collections to back up"
else
    mkdir -p "${destination}/qdrant"
    for collection in ${collections}; do
        log "Qdrant: snapshotting ${collection}"

        snapshot="$(curl -fsS -X POST \
            "${QDRANT_URL}/collections/${collection}/snapshots" \
            | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)"
        [ -n "${snapshot}" ] || fail "could not snapshot ${collection}"

        curl -fsS \
            "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot}" \
            -o "${destination}/qdrant/${collection}.snapshot" \
            || fail "could not download the snapshot of ${collection}"

        # The snapshot also stays inside the container; drop it so repeated
        # backups do not fill the volume.
        curl -fsS -X DELETE \
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

echo
echo "Backup complete: ${destination}"
echo "Restore with: ./deploy/restore.sh ${destination}"
