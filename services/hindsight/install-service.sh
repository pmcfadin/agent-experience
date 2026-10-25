#!/usr/bin/env bash
# Install and start the Hindsight server as a systemd user service (lingering keeps it up
# without a login session). Logs: journalctl --user -u elh-hindsight
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/.config/systemd/user"
ln -sf "$here/elh-hindsight.service" "$HOME/.config/systemd/user/elh-hindsight.service"
systemctl --user daemon-reload
systemctl --user enable --now elh-hindsight.service
