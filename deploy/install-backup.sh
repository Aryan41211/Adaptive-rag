#!/usr/bin/env bash
#
# Install the Adaptive RAG backup timer.
#
#   sudo ./deploy/install-backup.sh
#
# Copies the systemd unit files and enables the timer. The timer runs
# daily at 3 AM with up to 30 minutes of random delay.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing backup timer..."

sudo cp "${SCRIPT_DIR}/backup.service" /etc/systemd/system/
sudo cp "${SCRIPT_DIR}/backup.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now backup.timer

echo "Backup timer installed."
echo "Check status: systemctl list-timers backup.timer"
echo "View logs:    journalctl -u backup.service"
