#!/usr/bin/env bash
#
# Build and flash the drill / science firmware to the Teensy 4.1.
#
#   ./flash.sh                build and flash
#   ./flash.sh --build-only   compile, do not touch the board
#   ./flash.sh --clean        drop the cached micro-ROS library, then build+flash
#   ./flash.sh -v             show the full build output
#   ./flash.sh --help
#
# Prerequisite, once:   pipx install platformio
#
# ---------------------------------------------------------------------------
# WHY THIS IS NOT JUST `pio run -t upload`
#
# 1. ROS 2 POISONS THE BUILD.
#    ~/.bashrc sources /opt/ros/jazzy/setup.bash, so every terminal here has a
#    desktop x86 ROS 2 in its environment. micro_ros_platformio builds a
#    SEPARATE bare-metal ROS 2 from source, and the desktop one gets found
#    instead:
#        CMake Error ... No 'rosidl_typesupport_cpp' found
#        Call Stack: /opt/ros/jazzy/share/rosidl_typesupport_cpp/cmake/...
#    Unsetting AMENT_PREFIX_PATH and CMAKE_PREFIX_PATH is NOT enough. CMake's
#    find_package also searches every entry in PATH with bin/ stripped, and
#    /opt/ros/jazzy/bin is in PATH -- so it still finds /opt/ros/jazzy/share.
#    PATH itself has to be filtered. That is the part that is easy to miss.
#
# 2. micro_ros_platformio HARDCODES PlatformIO's virtualenv path.
#    extra_script.py builds "$PROJECT_CORE_DIR/penv/bin/activate", which only
#    exists if PlatformIO was installed by its own installer script. A pipx
#    install puts it elsewhere and the build dies with
#        .: cannot open /home/<user>/.platformio/penv/bin/activate
#
# 3. THE MICRO-ROS LIBRARY IS CACHED AND IGNORES CONFIG CHANGES.
#    Edit colcon.meta or build_flags in platformio.ini and the library is NOT
#    rebuilt -- the change silently does nothing. Use --clean after either.
#
# 4. THE FIRST WRITE TO THE BOOTLOADER ALWAYS FAILS.
#    Measured on this machine: after the board enters HalfKay, the first
#    teensy_loader_cli write reports
#        Programming...error writing to Teensy
#    and the second one, with no other change, succeeds every time. It is not a
#    settle-time problem -- waiting does not help, and the board is already
#    enumerated with 0666 permissions on its USB node. So the write is simply
#    retried below.
#
#    This is also why the build and the upload are separate steps here rather
#    than one `pio run -t upload`: retrying that would recompile and relink on
#    every attempt, and would print a full red FAILED block for something that
#    is expected and harmless.
#
# 5. teensy_loader_cli's OWN SOFT REBOOT IS UNRELIABLE HERE.
#    Its -s flag reports "Unable to soft reboot with USB error: Success" and
#    falls back to asking for the physical button. Opening the USB serial port
#    at 134 baud is the documented Teensy reboot-to-bootloader trigger and
#    works every time -- HalfKay appears in well under a second.
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$PWD"
DEVICES_YAML="$PROJECT_DIR/../../src/aries_common/config/devices.yaml"
HEX="$PROJECT_DIR/.pio/build/teensy41/firmware.hex"
MCU="TEENSY41"
WRITE_ATTEMPTS=5

DO_CLEAN=0
DO_UPLOAD=1
VERBOSE=0

for arg in "$@"; do
  case "$arg" in
    --clean)      DO_CLEAN=1 ;;
    --build-only) DO_UPLOAD=0 ;;
    -v|--verbose) VERBOSE=1 ;;
    -h|--help)    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    warning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. a PATH with no ROS on it -------------------------------------------
CLEAN_PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '^/opt/ros/' | paste -sd: -)"

PIO="$(PATH="$CLEAN_PATH" command -v pio || true)"
[ -n "$PIO" ] || die "pio not found. Install it with:  pipx install platformio"

# --- 2. make micro_ros_platformio's hardcoded penv path resolve -------------
PENV="$HOME/.platformio/penv"
if [ ! -e "$PENV/bin/activate" ]; then
  PIO_VENV="$(cd "$(dirname "$PIO")/.." && pwd)"
  if [ -f "$PIO_VENV/bin/activate" ]; then
    say "Linking $PENV -> $PIO_VENV"
    info "(micro_ros_platformio expects PlatformIO's virtualenv there)"
    mkdir -p "$HOME/.platformio"
    ln -sfn "$PIO_VENV" "$PENV"
  else
    warn "no virtualenv found alongside $PIO; the micro-ROS build may fail"
  fi
fi

# env -i so nothing from the calling shell leaks in.
run_pio() {
  env -i \
    HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-dumb}" \
    LANG="${LANG:-C.UTF-8}" PATH="$CLEAN_PATH" \
    "$PIO" "$@"
}

# Run pio, hiding the boilerplate unless it fails or -v was given. On failure
# the FULL log is printed -- never swallow an error to keep the output tidy.
run_pio_quiet() {
  local log rc=0
  log="$(mktemp)"
  if [ "$VERBOSE" = 1 ]; then
    run_pio "$@" 2>&1 | tee "$log" || rc=$?
  else
    run_pio "$@" >"$log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
      grep -E "Compiling|Linking|Building \.pio|micro-ROS|teensy_size|SUCCESS|warning:|error:" "$log" \
        | grep -vE "Requirement already satisfied|Installing .* with pip" \
        | sed 's/^/    /' || true
    fi
  fi
  if [ "$rc" -ne 0 ]; then
    echo "--- full output ---" >&2
    cat "$log" >&2
    rm -f "$log"
    return "$rc"
  fi
  rm -f "$log"
}

# Must always exit 0, even when no board is present: it is called as
# DEV="$(teensy_serial_dev)" inside a poll loop, and under `set -e` a non-zero
# command substitution aborts the whole script -- which it did, right after a
# successful write, while simply waiting for the board to come back.
teensy_serial_dev() {
  ls /dev/serial/by-id/ 2>/dev/null | grep -i teensy | head -1 || true
}
in_bootloader() { lsusb -d 16c0:0478 >/dev/null 2>&1; }

# --- the agent must not hold the port --------------------------------------
# `pgrep -x micro_ros_agent` and not -x micro_ros_agen: the process name is
# exactly 15 characters, which is the old comm-field width, so it is easy to
# write the truncated spelling -- and -x then matches NOTHING and the guard
# silently never fires. That is not theoretical: it let three agents pile up on
# one serial port here, and the symptom was ugly. The board established a
# session, sent its first CREATE, and got torn down over and over with zero
# entities ever created, which reads exactly like broken firmware.
#
# -f rather than -x, because the agent is usually started via `ros2 run`, which
# leaves a python wrapper process holding the same port -- and the wrapper is
# what `ros2 run` kills last. Matching the full command line catches both.
# flash.sh's own command line does not contain the string, so this cannot
# self-match.
if [ "$DO_UPLOAD" = 1 ] && pgrep -f micro_ros_agent >/dev/null 2>&1; then
  die "micro_ros_agent is running and holding the serial port.
       Stop the rover stack first, then:   pkill -x micro_ros_agent
       Flashing under a live agent either fails or leaves the board on a port
       the agent cannot reopen. More than one agent on one port also makes a
       healthy board look broken -- see the note above this check."
fi

# --- 3. clean the cached micro-ROS library ---------------------------------
if [ "$DO_CLEAN" = 1 ]; then
  say "Cleaning cached micro-ROS library"
  run_pio_quiet run -e teensy41 -t clean_microros
fi

# --- build ------------------------------------------------------------------
say "Building"
[ "$DO_CLEAN" = 1 ] && info "First build after a clean takes several minutes: micro-ROS is compiled from source."
run_pio_quiet run -e teensy41
[ -f "$HEX" ] || die "build reported success but $HEX is missing"
info "$(basename "$HEX") — $(stat -c%s "$HEX") bytes"

if [ "$DO_UPLOAD" = 0 ]; then
  say "Built (not flashed)"
  info "$HEX"
  exit 0
fi

# --- locate the loader ------------------------------------------------------
LOADER="$HOME/.platformio/packages/tool-teensy/teensy_loader_cli"
[ -x "$LOADER" ] || die "teensy_loader_cli not found at $LOADER"

# --- 5. put the board into HalfKay -----------------------------------------
say "Flashing"
if in_bootloader; then
  info "board is already in the HalfKay bootloader"
else
  DEV="$(teensy_serial_dev)"
  if [ -z "$DEV" ]; then
    die "no Teensy found — neither a serial device nor the HalfKay bootloader.
       Check the USB cable, then press the physical button on the board to
       force it into the bootloader and run this again."
  fi
  info "rebooting into the bootloader (134-baud trigger on $DEV)"
  stty -F "/dev/serial/by-id/$DEV" 134 >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do
    in_bootloader && break
    sleep 0.25
  done
  in_bootloader || die "board did not enter the bootloader.
       Press the physical button on the Teensy and run this again."
fi

# --- 4. write, retrying past the expected first failure --------------------
wrote=0
for n in $(seq 1 "$WRITE_ATTEMPTS"); do
  out="$("$LOADER" --mcu="$MCU" -w -v "$HEX" 2>&1 || true)"
  if printf '%s' "$out" | grep -q "error writing"; then
    # Expected on the first attempt; see note 4 at the top.
    [ "$n" -eq 1 ] && info "first write failed (expected on this machine) — retrying"
    [ "$n" -gt 1 ] && warn "write attempt $n failed"
    sleep 1
    continue
  fi
  if printf '%s' "$out" | grep -qE "Programming|Booting"; then
    info "written on attempt $n"
    wrote=1
    break
  fi
  warn "unexpected loader output on attempt $n:"
  printf '%s\n' "$out" | sed 's/^/      /' >&2
  sleep 1
done

if [ "$wrote" -ne 1 ]; then
  die "could not write to the Teensy after $WRITE_ATTEMPTS attempts.
       Press the physical button on the board and run this again. If it still
       fails, try a different USB cable -- a charge-only cable enumerates but
       cannot sustain a write."
fi

# --- verify the board came back --------------------------------------------
say "Waiting for the board to re-enumerate"
DEV=""
for _ in $(seq 1 24); do
  DEV="$(teensy_serial_dev)"
  [ -n "$DEV" ] && break
  sleep 0.5
done

if [ -z "$DEV" ]; then
  warn "no Teensy serial device appeared yet. The write succeeded, so the board
         is probably running -- check 'ls /dev/serial/by-id/' in a moment."
  exit 0
fi

DEV_PATH="/dev/serial/by-id/$DEV"
info "$DEV_PATH"

# --- does it match what the stack will look for? ---------------------------
if [ -f "$DEVICES_YAML" ]; then
  CONFIGURED="$(grep -oP '(?<=serial_port: ")[^"]+' "$DEVICES_YAML" | head -1 || true)"
  if [ -n "$CONFIGURED" ] && [ "$CONFIGURED" != "$DEV_PATH" ]; then
    warn "devices.yaml points at a DIFFERENT board:
           configured: $CONFIGURED
           connected:  $DEV_PATH
         A by-id path that does not exist is not a clean failure: the gripper
         silently resolves to mock_hardware, and the launch log, list_controllers
         and the checker all still read healthy while nothing reaches the servo.
         Fix gripper.serial_port in devices.yaml, or pass serial_port:= to the
         launch. 'ros2 control list_hardware_components' tells the truth."
  else
    info "matches devices.yaml"
  fi
fi

say "Done"
cat <<EOF
    The board is now waiting for the micro-ROS agent. The LED on pin 13:

      slow blink (500 ms)  flashed fine, waiting for the agent   <- expect this
      solid                agent connected, board driving
      fast blink (100 ms)  a pin is still PIN_UNASSIGNED

    Start the agent (or launch the rover stack, which starts its own):

      ros2 run micro_ros_agent micro_ros_agent serial \\
          --dev $DEV_PATH -b 115200

    Then check it came up:

      ros2 node list          # expect /teensy_drill_node
      ros2 topic hz /gripper/state

    Do NOT open a serial monitor on this board: Serial is the micro-ROS
    transport, not a console, and reading it corrupts the link.
EOF
