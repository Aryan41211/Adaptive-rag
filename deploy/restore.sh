#!/usr/bin/env bash
#
# Restore the Adaptive RAG data stores from a backup.
#
#   ./deploy/restore.sh backups/20260101T120000Z
#
# This REPLACES the current contents of both stores. It prompts first; set
# FORCE=1 to skip the prompt during an automated recovery.
#
# Environment: as deploy/backup.sh.

set -euo pipefail

SOURCE="${1:-}"
COMPOSE="${COMPOSE:-docker compose}"
MONGO_SERVICE="${MONGO_SERVICE:-mongo}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

log() { printf '  %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || fail "curl is required"

[ -n "${SOURCE}" ] || fail "usage: $0 <backup-directory>"
[ -d "${SOURCE}" ] || fail "no such backup directory: ${SOURCE}"
[ -f "${SOURCE}/mongodb.archive.gz" ] || fail "backup is missing mongodb.archive.gz"

echo "Restoring from ${SOURCE}"
[ -f "${SOURCE}/manifest.txt" ] && sed 's/^/  /' "${SOURCE}/manifest.txt"
echo

if [ "${FORCE:-0}" != "1" ]; then
    printf 'This REPLACES the current documents and database. Continue? [y/N] '
    read -r reply
    case "${reply}" in
        [yY]*) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# --- MongoDB ---------------------------------------------------------------
# --drop replaces each collection in the archive rather than merging into
# whatever is currently there.
log "MongoDB: restoring"
${COMPOSE} exec -T "${MONGO_SERVICE}" \
    mongorestore --archive --gzip --drop --quiet < "${SOURCE}/mongodb.archive.gz" \
    || fail "mongorestore failed"
log "MongoDB: restored"

# --- Qdrant ----------------------------------------------------------------
if [ -d "${SOURCE}/qdrant" ] && ls "${SOURCE}"/qdrant/*.snapshot >/dev/null 2>&1; then
    for snapshot in "${SOURCE}"/qdrant/*.snapshot; do
        collection="$(basename "${snapshot}" .snapshot)"
        log "Qdrant: restoring ${collection}"

        # priority=snapshot makes the snapshot authoritative over whatever is
        # currently in the collection, so a restore is a replacement.
        curl -fsS -X POST \
            -H 'Content-Type: multipart/form-data' \
            -F "snapshot=@${snapshot}" \
            "${QDRANT_URL}/collections/${collection}/snapshots/upload?priority=snapshot" \
            >/dev/null \
            || fail "could not restore ${collection}"

        log "Qdrant: ${collection} restored"
    done
else
    log "Qdrant: nothing in this backup"
fi

echo
echo "Restore complete. Restart the API so it reconnects cleanly:"
echo "  ${COMPOSE} restart api"
