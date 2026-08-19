#!/bin/bash
# Superseded by aries/scripts/setup_system.sh.
#
# This script used to write the sudoers rule itself, and the rule it wrote could
# never match: run as `sudo bash setup_can_sudo.sh`, $(whoami) is root rather
# than the account that launches the robot, and it named /sbin/ip while sudo
# resolves `ip` through secure_path to /usr/sbin/ip. It also granted only
# `link set up`, not the `down` the drive stack issues first.
#
# The canonical setup covers the same rule (generated for the right account,
# every ip path, both commands, taken from devices.yaml) plus udev rules and
# group membership.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL="$(cd "$HERE/../../.." && pwd)/scripts/setup_system.sh"

echo "setup_can_sudo.sh is superseded — running $CANONICAL instead."
exec "$CANONICAL" "$@"
