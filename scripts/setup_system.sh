#!/usr/bin/env bash
#
# One-time system setup for the Aries workspace on a new computer.
#
#   ./scripts/setup_system.sh              install everything, then verify
#   ./scripts/setup_system.sh --check      verify only, change nothing
#   ./scripts/setup_system.sh --dry-run    print what would change
#
# Everything the stack needs from root lives here: the passwordless CAN
# bring-up rule, the udev rules, the DDS socket-buffer ceilings, and the group
# memberships. Re-running is safe —
# every step is idempotent and reports whether it changed anything.
#
# Run it as yourself, not with sudo: it calls sudo only for the steps that need
# it, and it has to know which account to grant the rules to. Running it under
# sudo works too (the account comes from $SUDO_USER).
#
# What it deliberately does NOT do: install apt packages, or bring can0 up.
# Missing packages are reported at the end with the exact command to run, and
# the CAN interface is brought up per boot by the launch files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICES_YAML="$REPO_ROOT/src/aries_common/config/devices.yaml"

# Rehearse the whole run against a throwaway tree rather than the real system:
#
#   ARIES_SETUP_PREFIX=/tmp/rehearsal ./scripts/setup_system.sh
#
# Files are written under the prefix, sudo is never called, and the steps that
# change the machine itself (usermod, udevadm) are reported but not run. This is
# how the script is tested — it is the only way to exercise the install path
# without root on a working robot.
PREFIX="${ARIES_SETUP_PREFIX:-}"

SUDOERS_FILE="$PREFIX/etc/sudoers.d/rover_can"
UDEV_DIR="$PREFIX/etc/udev/rules.d"
REALSENSE_RULES="$UDEV_DIR/99-aries-realsense.rules"
TEENSY_RULES="$UDEV_DIR/99-aries-teensy.rules"
SYSCTL_FILE="$PREFIX/etc/sysctl.d/99-aries-dds.conf"
REQUIRED_GROUPS=(dialout plugdev input video)

MODE=install
case "${1:-}" in
    --check)   MODE=check ;;
    --dry-run) MODE=dry-run ;;
    "")        ;;
    *) echo "usage: $0 [--check|--dry-run]" >&2; exit 2 ;;
esac

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RST=$'\033[0m'
CHANGED=0
FAILED=0
NEEDS_RELOGIN=0
NOTES=()

ok()      { printf '  %s✓%s %s\n' "$GREEN" "$RST" "$1"; }
changed() { printf '  %s+%s %s\n' "$GREEN" "$RST" "$1"; CHANGED=1; }
warn()    { printf '  %s~%s %s\n' "$YELLOW" "$RST" "$1"; }
fail()    { printf '  %s✗%s %s\n' "$RED" "$RST" "$1"; FAILED=1; }
head2()   { printf '\n%s%s%s\n' "$BOLD" "$1" "$RST"; }

# ── who the rules are for ────────────────────────────────────────────────────
TARGET_USER="${SUDO_USER:-${USER:-$(id -un)}}"
if [ "$TARGET_USER" = "root" ]; then
    echo "${RED}Refusing to grant the rules to root.${RST} Run this as the account that" >&2
    echo "launches the robot: ./scripts/setup_system.sh" >&2
    exit 1
fi

rehearsing() { [ -n "$PREFIX" ]; }

as_root() {
    if rehearsing; then "$@"
    elif [ "$(id -u)" -eq 0 ]; then "$@"
    else sudo "$@"
    fi
}

# Compare without needing root: 0 identical, 1 absent or different, 2 unknown
# because the file is there but only root may read it (a 0440 sudoers rule).
file_state() {
    local dest="$1" content="$2"
    if [ -r "$dest" ]; then
        [ "$(cat "$dest")" = "$content" ] && return 0 || return 1
    elif [ -e "$dest" ]; then
        return 2
    fi
    return 1
}

# Write $2 to file $1 with mode $3, only touching it when the content differs.
install_file() {
    local dest="$1" content="$2" mode="$3" state=0
    file_state "$dest" "$content" || state=$?

    case "$state" in
        0) ok "$dest already correct"; return 0 ;;
        2) if [ "$MODE" != install ]; then
               warn "$dest exists; needs sudo to compare"
               return 0
           fi ;;
    esac

    if [ "$MODE" != install ]; then
        warn "$dest would be written"
        CHANGED=1
        return 0
    fi
    as_root mkdir -p "$(dirname "$dest")"
    printf '%s\n' "$content" | as_root tee "$dest" >/dev/null
    as_root chmod "$mode" "$dest"
    changed "$dest written"
}

# ── device settings come from the same table the launch files read ───────────
read_device() {
    python3 - "$DEVICES_YAML" "$1" "$2" <<'PY' 2>/dev/null || echo "$3"
import sys
try:
    import yaml
    data = yaml.safe_load(open(sys.argv[1])) or {}
    print(data[sys.argv[2]][sys.argv[3]])
except Exception:
    raise SystemExit(1)
PY
}

CAN_IF="$(read_device rover can_interface can0)"
CAN_BITRATE="$(read_device rover can_bitrate 250000)"

printf '%s\n' "${BOLD}Aries system setup${RST}"
printf '  account : %s\n' "$TARGET_USER"
printf '  workspace: %s\n' "$REPO_ROOT"
printf '  CAN     : %s @ %s bit/s (from %s)\n' "$CAN_IF" "$CAN_BITRATE" \
    "${DEVICES_YAML#"$REPO_ROOT"/}"
[ "$MODE" = install ] || printf '  mode    : %s — nothing will be written\n' "$MODE"

# ── 1. passwordless CAN bring-up ─────────────────────────────────────────────
#
# The drive stack brings the CAN interface up at launch, and again whenever the
# adapter is unplugged and plugged back in (it returns administratively DOWN
# with a new interface index). Both go through `sudo -n`, which cannot answer a
# password prompt, so the two exact command lines need a NOPASSWD entry.
#
# sudoers matches the command by its resolved path, and which of /usr/sbin/ip,
# /sbin/ip or /bin/ip that is depends on the distro's secure_path. Grant every
# one that exists rather than guessing.
head2 "1. Passwordless CAN setup ($SUDOERS_FILE)"

IP_PATHS=()
for candidate in /usr/sbin/ip /sbin/ip /bin/ip /usr/bin/ip; do
    [ -x "$candidate" ] && IP_PATHS+=("$candidate")
done
if [ ${#IP_PATHS[@]} -eq 0 ]; then
    fail "no 'ip' binary found — install iproute2"
else
    commands=""
    for path in "${IP_PATHS[@]}"; do
        [ -n "$commands" ] && commands+=", "
        commands+="$path link set $CAN_IF down, $path link set $CAN_IF up type can bitrate $CAN_BITRATE"
    done
    RULE="# Installed by aries/scripts/setup_system.sh — do not edit by hand.
# Lets the drive stack bring $CAN_IF up without a password, and nothing else.
$TARGET_USER ALL=(root) NOPASSWD: $commands"

    tmp_rule="$(mktemp)"
    printf '%s\n' "$RULE" > "$tmp_rule"
    if visudo -cf "$tmp_rule" >/dev/null 2>&1; then
        install_file "$SUDOERS_FILE" "$RULE" 0440
    else
        fail "generated sudoers rule is invalid; not installing"
        visudo -cf "$tmp_rule" || true
    fi
    rm -f "$tmp_rule"
fi

# ── 2. udev rules ────────────────────────────────────────────────────────────
head2 "2. udev rules"

# RealSense: USB autosuspend drops the D435i mid-session, which surfaces as a
# camera that simply stops publishing.
REALSENSE_RULE='# Installed by aries/scripts/setup_system.sh — do not edit by hand.
# Intel RealSense D400-series: never let USB autosuspend power the camera down.
SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", ATTR{power/control}="on"'
install_file "$REALSENSE_RULES" "$REALSENSE_RULE" 0644

# Teensy: dialout membership covers /dev/ttyACM* for the gripper firmware, but
# flashing and the HID interfaces need the board itself writable. Distilled from
# pjrc.com/teensy/00-teensy.rules; installing the full upstream file as well is
# harmless.
TEENSY_RULE='# Installed by aries/scripts/setup_system.sh — do not edit by hand.
# Teensy (PJRC) — serial for the gripper firmware, raw access for flashing.
# Distilled from http://www.pjrc.com/teensy/00-teensy.rules
ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04[789ABCD]?", ENV{ID_MM_DEVICE_IGNORE}="1"
ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04[789B]?", ENV{MTP_NO_PROBE}="1"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04[789ABCD]?", MODE:="0666"
KERNEL=="ttyACM*", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04[789B]?", MODE:="0666"'
install_file "$TEENSY_RULES" "$TEENSY_RULE" 0644

if [ "$MODE" = install ] && [ "$CHANGED" -eq 1 ]; then
    if rehearsing; then
        warn "udev reload skipped (rehearsal)"
    else
        as_root udevadm control --reload-rules
        as_root udevadm trigger
        changed "udev rules reloaded"
    fi
fi

# ── 3. group membership ──────────────────────────────────────────────────────
#
#   dialout  /dev/ttyACM* — the Teensy gripper board
#   plugdev  USB devices claimed from userspace (RealSense)
#   input    /dev/input/event* — the joystick, read by game_controller_node
#   video    /dev/video* — the rear Brio, opened by usb_cam through V4L2
#
# 'video' is the one that looks unnecessary on a desktop and is not. /dev/video*
# is root:video, and a user logged in AT THE MACHINE also gets a per-device ACL
# from systemd-logind, so the camera opens fine from a local terminal whether or
# not the user is in the group. That ACL is granted to a seat session. Over SSH
# there is no seat, so on the rover — which is always driven over SSH — the same
# user hits "permission denied" opening the same camera that worked on the
# bench. Group membership is what makes it work in both places.
#
# The RealSenses are unaffected either way: their udev rules put them in plugdev
# and librealsense claims them from userspace rather than through V4L2.
# ── 3. DDS socket buffers ────────────────────────────────────────────────────
head2 "3. DDS socket buffers ($SYSCTL_FILE)"

# aries_common.comms asks every UDP socket for SOCKET_BUFFER_BYTES. The kernel
# clamps that request to net.core.rmem_max, so without this file the request is
# cut back to the stock 208 kB and NOTHING REPORTS IT — the profile still says
# 8 MB, the sockets just do not get it. Check with `ss -u -a -m | grep rb`.
#
# 208 kB is smaller than one downlink frame pair (98 kB colour + 91 kB depth),
# which is why the link felt slower after the move to Fast DDS: Cyclone had been
# asking for 1 MiB per socket and Fast DDS asks for nothing at all.
SYSCTL_RULE='# Installed by aries/scripts/setup_system.sh — do not edit by hand.
# DDS receive/send buffers. Fast DDS asks for 8 MB per socket; the kernel clamps
# the request to these ceilings, so they have to be raised or the ask is
# silently ignored. See SOCKET_BUFFER_BYTES in aries_common/comms.py.
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216'
install_file "$SYSCTL_FILE" "$SYSCTL_RULE" 0644

if [ "$MODE" = install ]; then
    if rehearsing; then
        warn "sysctl reload skipped (rehearsal)"
    else
        as_root sysctl -q --system >/dev/null 2>&1 || true
        for knob in net.core.rmem_max net.core.wmem_max; do
            have="$(sysctl -n "$knob" 2>/dev/null || echo 0)"
            if [ "$have" -ge 16777216 ]; then
                ok "$knob = $have"
            else
                fail "$knob = $have (wanted 16777216); DDS buffers will be clamped"
            fi
        done
    fi
fi

head2 "4. Group membership for $TARGET_USER"
for group in "${REQUIRED_GROUPS[@]}"; do
    if ! getent group "$group" >/dev/null; then
        warn "group '$group' does not exist on this system — skipping"
        continue
    fi
    if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx "$group"; then
        ok "already in $group"
    elif [ "$MODE" != install ]; then
        warn "would add $TARGET_USER to $group"
        CHANGED=1
    elif rehearsing; then
        warn "would add $TARGET_USER to $group (rehearsal)"
    else
        as_root usermod -aG "$group" "$TARGET_USER"
        changed "added to $group"
        NEEDS_RELOGIN=1
    fi
done

# ── 4. things this script will not install for you ───────────────────────────
head2 "5. Packages and kernel support"

if modinfo gs_usb >/dev/null 2>&1 || [ -d /sys/module/gs_usb ]; then
    ok "gs_usb CAN driver available (the USB CAN adapter binds to it)"
else
    warn "gs_usb module not found — the USB CAN adapter will not appear as $CAN_IF"
fi

if dpkg -s ros-jazzy-microstrain-inertial-driver >/dev/null 2>&1; then
    ok "microstrain driver installed (it ships the /dev/microstrain_main rule)"
else
    warn "microstrain driver missing — no /dev/microstrain_main, IMU falls back to wheel odometry"
    NOTES+=("sudo apt install ros-jazzy-microstrain-inertial-driver")
fi

if dpkg -s can-utils >/dev/null 2>&1; then
    ok "can-utils present (candump/cansend for debugging)"
else
    warn "can-utils missing — optional, but candump is how you watch the bus"
    NOTES+=("sudo apt install can-utils")
fi

# The rear camera is a Logitech Brio, a plain UVC webcam, so realsense2_camera
# cannot drive it. Without usb_cam the rest of the stack still comes up and the
# rear camera is simply absent — aries_hardware.launch.py's enable_rear_camera
# defaults to "auto" and skips a driver it cannot start.
#
# No udev rule is needed for it: the /dev/v4l/by-id/ symlinks come from the
# stock rules and devices.yaml pins that path. Access comes from the 'video'
# group added in section 3 — see the note there for why the ACL that makes this
# work on the bench does not exist over SSH.
if dpkg -s ros-jazzy-usb-cam >/dev/null 2>&1; then
    ok "usb_cam installed (drives the rear Brio watching the drill)"
else
    warn "usb_cam missing — the rear drill camera will not start; everything else is unaffected"
    NOTES+=("sudo apt install ros-jazzy-usb-cam")
fi

# ── 5. verify ────────────────────────────────────────────────────────────────
head2 "6. Verification"

if rehearsing; then
    [ -f "$SUDOERS_FILE" ] && ok "rehearsal wrote $SUDOERS_FILE" || fail "rehearsal did not write $SUDOERS_FILE"
    visudo -cf "$SUDOERS_FILE" >/dev/null 2>&1 \
        && ok "generated sudoers rule passes visudo" \
        || fail "generated sudoers rule fails visudo"
elif [ -f "$SUDOERS_FILE" ]; then
    if sudo -n -l "${IP_PATHS[0]}" link set "$CAN_IF" up type can bitrate "$CAN_BITRATE" >/dev/null 2>&1; then
        ok "sudo -n ip link set $CAN_IF up ... works without a password"
    elif [ "$NEEDS_RELOGIN" -eq 1 ] || [ "$MODE" != install ]; then
        warn "CAN rule not active yet — re-check after logging back in"
    else
        fail "CAN rule installed but still not passwordless; check that /etc/sudoers includes /etc/sudoers.d"
    fi
else
    [ "$MODE" = install ] && fail "$SUDOERS_FILE missing" || warn "$SUDOERS_FILE would be created"
fi

for f in "$REALSENSE_RULES" "$TEENSY_RULES" "$SYSCTL_FILE"; do
    if [ -f "$f" ]; then ok "$(basename "$f") installed"
    elif [ "$MODE" = install ]; then fail "$(basename "$f") missing"
    fi
done

for legacy in 99-realsense-usb.rules 99-ybimu.rules 99-imu-bno055.rules; do
    rehearsing && break
    if [ -f "$UDEV_DIR/$legacy" ]; then
        warn "$legacy is also present — superseded, safe to delete: sudo rm $UDEV_DIR/$legacy"
    fi
done

head2 "Summary"
if [ "$FAILED" -eq 1 ]; then
    printf '  %s✗ Some steps failed — see above.%s\n' "$RED" "$RST"
elif [ "$MODE" != install ]; then
    printf '  %s~ %s only: %s%s\n' "$YELLOW" "$MODE" \
        "$([ "$CHANGED" -eq 1 ] && echo 'changes are pending' || echo 'everything already in place')" "$RST"
elif [ "$CHANGED" -eq 1 ]; then
    printf '  %s✓ System setup applied.%s\n' "$GREEN" "$RST"
else
    printf '  %s✓ Everything was already in place.%s\n' "$GREEN" "$RST"
fi

if [ "$NEEDS_RELOGIN" -eq 1 ]; then
    printf '  %s→ Log out and back in%s — group changes only apply to new sessions.\n' "$YELLOW" "$RST"
fi
if [ ${#NOTES[@]} -gt 0 ]; then
    printf '  → Not installed by this script:\n'
    for note in "${NOTES[@]}"; do printf '      %s\n' "$note"; done
fi

[ "$FAILED" -eq 1 ] && exit 1 || exit 0
