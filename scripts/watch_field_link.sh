#!/usr/bin/env bash
#
# Watch the field link across an event, and say WHICH LAYER dropped.
#
#   ./scripts/watch_field_link.sh                 watch until Ctrl-C
#   ./scripts/watch_field_link.sh --seconds 60    watch for 60 s
#
# WHY THIS EXISTS
#     "We lose comms when the arm engages" has at least four causes that look
#     identical from a terminal, and setup_field_link.sh --check cannot tell
#     them apart because it takes ONE sample: by the time you have run it the
#     event is over and everything is healthy again.
#
#     The arm and the antenna share a switch on the rover, so an arm that
#     browns out the switch takes the radio down with it -- and that is
#     indistinguishable, from ROS, from an RF link that faded. This samples all
#     four layers continuously through the event and then tells you which one
#     went first:
#
#       ETHERNET   the NIC's carrier dropped, or its flap counter moved.
#                  The switch reset, the cable moved, or something lost power.
#                  Nothing to do with radio. The arm goes with it.
#       OUR RADIO  the local airOS box stopped answering. It is on the same
#                  switch and usually the same PoE supply as everything else,
#                  so this is a power symptom, not an RF one.
#       RF LINK    our radio is fine, the far end is not. Alignment, distance,
#                  interference -- or the far end lost power.
#       DDS ONLY   every ping above stayed clean. The network held and the
#                  problem is above it: scripts/check_control_path.py.
#
# RUN IT ON THE ROVER, then engage the arm while it watches.
#
#     ./scripts/watch_field_link.sh --seconds 60
#     ...engage the arm...
#
# Read-only. It sends pings and reads /sys; it changes nothing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RST=$'\033[0m'
ok()    { printf '  %s✓%s %s\n' "$GREEN" "$RST" "$1"; }
warn()  { printf '  %s~%s %s\n' "$YELLOW" "$RST" "$1"; }
fail()  { printf '  %s✗%s %s\n' "$RED" "$RST" "$1"; }
head2() { printf '\n%s%s%s\n' "$BOLD" "$1" "$RST"; }

SECONDS_TO_WATCH=0
while [ $# -gt 0 ]; do
    case "$1" in
        --seconds) SECONDS_TO_WATCH="${2:-0}"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--seconds N]" >&2; exit 2 ;;
    esac
done

# Same source of truth as setup_field_link.sh and the launch files.
read_table() {
    PYTHONPATH="$REPO_ROOT/src/aries_common:${PYTHONPATH:-}" python3 - "$@" <<'PY'
import sys
from aries_common import comms
from aries_common.devices import device
what = sys.argv[1]
if what == "whoami":
    print(comms.local_address() or "")
elif what == "hosts":
    for name, addr in sorted(comms.hosts().items()):
        print(f"{name} {addr}")
elif what == "iface":
    print(comms.interfaces().get(sys.argv[2], ""))
elif what == "radio":
    print(comms.radios().get(sys.argv[2], ""))
elif what == "arm":
    print(device("arm.host"))
PY
}

HERE="$(read_table whoami)"
if [ -z "$HERE" ]; then
    fail "not on the field link — run scripts/setup_field_link.sh --check first"
    exit 1
fi
WHO="$(PYTHONPATH="$REPO_ROOT/src/aries_common:${PYTHONPATH:-}" \
       python3 -c "from aries_common import comms; print(comms.host_name_for('$HERE') or '')")"
[ -n "$WHO" ] || { fail "$HERE is not a configured host"; exit 1; }
FAR="$([ "$WHO" = rover ] && echo base || echo rover)"

IFACE="$(read_table iface "$WHO")"
FAR_ADDR="$(read_table hosts | awk -v n="$FAR" '$1==n{print $2}')"
OUR_RADIO="$(read_table radio "$WHO")"
ARM_ADDR="$(read_table arm)"

printf '%sWatching the field link from the %s (%s)%s\n' "$BOLD" "$WHO" "$HERE" "$RST"
printf '  ethernet   %s\n' "${IFACE:-<unknown>}"
printf '  our radio  %s\n' "${OUR_RADIO:-<none>}"
printf '  far end    %s (%s)\n' "${FAR_ADDR:-<none>}" "$FAR"
printf '  arm        %s (same switch, other subnet)\n' "${ARM_ADDR:-<none>}"
printf '\nEngage the arm now. Ctrl-C when done.\n'

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# One persistent ping per target rather than a one-shot per sample: -D stamps
# every reply, so a gap in the stamps is the outage with its start time and
# duration, not just a percentage after the fact.
PIDS=()
start_ping() {  # name address
    [ -n "$2" ] || return 0
    ping -n -i 0.2 -D "$2" > "$TMP/$1.txt" 2>&1 &
    PIDS+=($!)
}
start_ping far   "$FAR_ADDR"
start_ping radio "$OUR_RADIO"
start_ping arm   "$ARM_ADDR"

# The Ethernet layer. carrier_changes is a kernel counter: it moves even if the
# link bounces back between two of our samples, which a carrier read alone
# would miss entirely.
CARRIER_FILE="/sys/class/net/$IFACE/carrier_changes"
CARRIER_START=0
[ -r "$CARRIER_FILE" ] && CARRIER_START="$(cat "$CARRIER_FILE")"
DOWN_SAMPLES=0
SAMPLES=0

STOP=0
trap 'STOP=1' INT

BEGIN="$(date +%s)"
while :; do
    if [ -r "/sys/class/net/$IFACE/carrier" ]; then
        SAMPLES=$((SAMPLES + 1))
        [ "$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo 0)" = "1" ] \
            || DOWN_SAMPLES=$((DOWN_SAMPLES + 1))
    fi
    sleep 0.2 || true
    [ "$STOP" = 1 ] && break
    NOW="$(date +%s)"
    if [ "$SECONDS_TO_WATCH" -gt 0 ] && [ $((NOW - BEGIN)) -ge "$SECONDS_TO_WATCH" ]; then
        break
    fi
done

# SIGINT, not SIGTERM: ping answers INT by printing its own statistics line,
# which is the exact sent/received count. Killed any other way it prints
# nothing and the loss figure would have to be inferred from the elapsed time.
for p in "${PIDS[@]:-}"; do kill -INT "$p" 2>/dev/null || true; done
sleep 0.5
wait 2>/dev/null || true

ELAPSED=$(( $(date +%s) - BEGIN ))
CARRIER_END="$CARRIER_START"
[ -r "$CARRIER_FILE" ] && CARRIER_END="$(cat "$CARRIER_FILE")"
FLAPS=$((CARRIER_END - CARRIER_START))

# ---------------------------------------------------------------- results
loss_of() {  # name -> integer percent, or -1 if not measured
    local f="$TMP/$1.txt"
    [ -s "$f" ] || { echo -1; return; }
    # ping's own summary, printed on SIGINT: "N packets transmitted, M
    # received, X% packet loss, ...". Its count is authoritative; ours would
    # be an estimate from the interval.
    local pct
    pct="$(sed -n 's/.*, \([0-9]\+\)% packet loss.*/\1/p' "$f" | tail -1)"
    if [ -n "$pct" ]; then echo "$pct"; return; fi
    # No summary (killed too hard, or the host never resolved). Fall back to
    # "did anything ever reply".
    if grep -q 'bytes from' "$f"; then echo 0; else echo 100; fi
}
replies_of() {  # how many replies ever arrived
    local f="$TMP/$1.txt"
    [ -s "$f" ] || { echo 0; return; }
    # grep -c prints 0 AND exits 1 on no match, so `|| echo 0` would emit a
    # second line and every later [ ] test would see "0\n0".
    local n
    n="$(grep -c 'bytes from' "$f" 2>/dev/null || true)"
    echo "${n:-0}"
}

gap_of() {  # longest gap in seconds between replies, as a float string
    local f="$TMP/$1.txt"
    [ -s "$f" ] || { echo "-"; return; }
    awk -F'[][]' '/bytes from/ {t=$2+0; if (p && t-p > m) m=t-p; p=t}
                  END {if (m>0) printf "%.1f", m; else print "0.0"}' "$f"
}

head2 "Result after ${ELAPSED}s"

FAR_LOSS="$(loss_of far)"; RADIO_LOSS="$(loss_of radio)"; ARM_LOSS="$(loss_of arm)"

if [ "$FLAPS" -gt 0 ] || [ "$DOWN_SAMPLES" -gt 0 ]; then
    fail "ETHERNET: $IFACE bounced ($FLAPS carrier change(s), $DOWN_SAMPLES/$SAMPLES samples down)"
    echo "      The wire went down, not the radio. Everything on that switch"
    echo "      dropped together — which is why the arm and the link fail as one."
    echo "      Look at POWER first: the arm's motors pulling the switch (or the"
    echo "      PoE injector) below its brown-out threshold does exactly this."
    echo "      Put the switch and the radio on their own supply, or a separate"
    echo "      DC-DC off the battery, and repeat this test."
else
    ok "ETHERNET: $IFACE held carrier the whole time (0 flaps)"
fi

# A target that never answered ONCE did not drop out — it was not there to
# begin with. Reporting that as an outage would point the finger at the switch
# for a box that is simply powered off, which is the opposite of useful.
FAR_SEEN="$(replies_of far)"; RADIO_SEEN="$(replies_of radio)"; ARM_SEEN="$(replies_of arm)"
absent() { [ "$1" -eq 0 ]; }

report() {  # label loss gap replies
    if [ "$2" -lt 0 ]; then warn "$1: not measured"
    elif [ "$4" -eq 0 ]; then warn "$1: never answered — powered off, or not on this network"
    elif [ "$2" -eq 0 ]; then ok "$1: no loss (longest gap ${3}s)"
    elif [ "$2" -lt 5 ]; then warn "$1: ${2}% loss, longest gap ${3}s"
    else fail "$1: ${2}% loss, longest gap ${3}s — IT DROPPED"
    fi
}
report "OUR RADIO ${OUR_RADIO}" "$RADIO_LOSS" "$(gap_of radio)" "$RADIO_SEEN"
report "FAR END   ${FAR_ADDR}"  "$FAR_LOSS"   "$(gap_of far)"   "$FAR_SEEN"
report "ARM       ${ARM_ADDR}"  "$ARM_LOSS"   "$(gap_of arm)"   "$ARM_SEEN"

# Only a target that answered and then stopped counts as a drop.
dropped() {  # replies loss
    [ "$1" -gt 0 ] && [ "$2" -gt 0 ]
}

head2 "Verdict"
if [ "$FLAPS" -gt 0 ] || [ "$DOWN_SAMPLES" -gt 0 ]; then
    echo "  The Ethernet link itself dropped. This is a switch/cable/POWER"
    echo "  problem on the rover. Not RF, not DDS, not ROS."
elif absent "$RADIO_SEEN" && absent "$FAR_SEEN" && absent "$ARM_SEEN"; then
    echo "  Nothing answered at any point, so nothing DROPPED — there was no"
    echo "  event to catch. Power the rover, the radios and the arm up first,"
    echo "  confirm with scripts/setup_field_link.sh --check, then run this"
    echo "  again and engage the arm while it watches."
elif dropped "$RADIO_SEEN" "$RADIO_LOSS" && dropped "$ARM_SEEN" "$ARM_LOSS"; then
    echo "  Our own radio AND the arm both stopped answering while the wire"
    echo "  stayed up. Both hang off the same switch: suspect the switch or its"
    echo "  supply, not the radio link."
elif dropped "$RADIO_SEEN" "$RADIO_LOSS"; then
    echo "  Our own radio stopped answering over a wire that never dropped."
    echo "  The radio rebooted or browned out — check its PoE injector supply."
elif dropped "$FAR_SEEN" "$FAR_LOSS"; then
    echo "  Our radio stayed up but the FAR END did not: this is the RF hop."
    echo "  Motor drive EMI desensing the radio, or the far end losing power."
    echo "  Move the radio off the arm's cable run and re-test; airOS 'Signal'"
    echo "  on the local radio during the engage tells you which."
elif dropped "$ARM_SEEN" "$ARM_LOSS"; then
    echo "  Only the arm dropped. The link is fine; this is the arm's own"
    echo "  Ethernet or its control box, not comms."
else
    echo "  Every layer held for ${ELAPSED}s. If ROS still went quiet, the"
    echo "  network is not the problem — check the ROS side:"
    echo "      ./scripts/check_control_path.py --arm"
fi
