#!/bin/bash
# Run once on any new machine to allow passwordless CAN interface setup.
# Usage: sudo bash setup_can_sudo.sh

RULE=''"$(whoami)"' ALL=(ALL) NOPASSWD: /sbin/ip link set can0 up type can bitrate 250000'
DEST=/etc/sudoers.d/rover_can

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash $0"
    exit 1
fi

echo "$RULE" > "$DEST"
chmod 0440 "$DEST"
visudo -c -f "$DEST" || { rm "$DEST"; echo "ERROR: sudoers rule is invalid, removed."; exit 1; }
echo "Done. Passwordless sudo for CAN setup enabled for $(whoami)."
