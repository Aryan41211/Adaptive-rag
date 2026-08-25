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
ENV_FILE="${ENV_FILE:-./.env}"

log() { printf '  %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Read one KEY=value out of the env file without sourcing it: sourcing would
# execute whatever happens to be in there.
env_file_value() {
    [ -f "${ENV_FILE}" ] || return 0
    sed -n "s/^[[:space:]]*$1=//p" "${ENV_FILE}" | tail -1 \
        | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

MONGO_ROOT_USERNAME="${MONGO_ROOT_USERNAME:-$(env_file_value MONGO_ROOT_USERNAME)}"
MONGO_ROOT_PASSWORD="${MONGO_ROOT_PASSWORD:-$(env_file_value MONGO_ROOT_PASSWORD)}"

mongo_auth_args=()
if [ -n "${MONGO_ROOT_USERNAME}" ] && [ -n "${MONGO_ROOT_PASSWORD}" ]; then
    mongo_auth_args=(
        -u "${MONGO_ROOT_USERNAME}"
        -p "${MONGO_ROOT_PASSWORD}"
        --authenticationDatabase admin
    )
fi

qdrant_curl_args=(-fsS)
if [ -n "${QDRANT_API_KEY:-}" ]; then
    qdrant_curl_args+=(-H "api-key: ${QDRANT_API_KEY}")
fi

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
    mongorestore --archive --gzip --drop --quiet "${mongo_auth_args[@]}" \
    < "${SOURCE}/mongodb.archive.gz" \
    || fail "mongorestore failed. Are MONGO_ROOT_USERNAME / MONGO_ROOT_PASSWORD correct?"
log "MongoDB: restored"

# --- Qdrant ----------------------------------------------------------------
if [ -d "${SOURCE}/qdrant" ] && ls "${SOURCE}"/qdrant/*.snapshot >/dev/null 2>&1; then
    for snapshot in "${SOURCE}"/qdrant/*.snapshot; do
        collection="$(basename "${snapshot}" .snapshot)"
        log "Qdrant: restoring ${collection}"

        # priority=snapshot makes the snapshot authoritative over whatever is
        # currently in the collection, so a restore is a replacement.
        curl "${qdrant_curl_args[@]}" -X POST \
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
