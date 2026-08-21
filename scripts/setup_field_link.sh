#!/usr/bin/env bash
#
# Pin this machine's field-link address. Run once per machine, per install.
#
#   ./scripts/setup_field_link.sh rover      this is the rover PC
#   ./scripts/setup_field_link.sh base       this is the base station
#   ./scripts/setup_field_link.sh --check    verify only, change nothing
#   ./scripts/setup_field_link.sh --dry-run  print what would change
#
# WHY THIS EXISTS
#     On a shared field you are not the only team with radios. DHCP means the
#     address is whatever a stranger's router felt like handing out, and
#     aries_common/comms.py then cannot tell which machine it is running on --
#     so the rover can announce itself as the base station, or neither end
#     finds the other. Static addressing is not a preference here, it is what
#     makes the machine identifiable.
#
#     Addresses come from the network: section of
#     src/aries_common/config/devices.yaml. Change them there, not here.
#
# THE ARM SHARES THIS WIRE
#     The igus ReBeL control box is on 192.168.3.0/24 (arm: in devices.yaml) --
#     a different subnet, but on the rover it may well be on the SAME Ethernet
#     port through a switch. This script therefore never removes an address on
#     the arm subnet and never disables a profile that carries one; it merges
#     the field-link address alongside. Losing the arm to a network script
#     would look like a dead arm, not a network problem, which is a bad hour.
#
# Run as yourself, not with sudo: it calls sudo only for the nmcli steps.
# Re-running is safe -- every step is idempotent and reports what it changed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONN_NAME="aries-field-link"

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RST=$'\033[0m'
CHANGED=0
FAILED=0

ok()      { printf '  %s✓%s %s\n' "$GREEN" "$RST" "$1"; }
changed() { printf '  %s+%s %s\n' "$GREEN" "$RST" "$1"; CHANGED=1; }
warn()    { printf '  %s~%s %s\n' "$YELLOW" "$RST" "$1"; }
fail()    { printf '  %s✗%s %s\n' "$RED" "$RST" "$1"; FAILED=1; }
head2()   { printf '\n%s%s%s\n' "$BOLD" "$1" "$RST"; }

ROLE=""
MODE=install
case "${1:-}" in
    rover|base) ROLE="$1" ;;
    --check)    MODE=check ;;
    --dry-run)  MODE=dry-run; ROLE="${2:-}" ;;
    *) echo "usage: $0 {rover|base} | --check | --dry-run {rover|base}" >&2; exit 2 ;;
esac

# The device table is the single source of truth. Read it with the same module
# the launch files use, so a typo here cannot disagree with what ROS believes.
read_table() {
    PYTHONPATH="$REPO_ROOT/src/aries_common:${PYTHONPATH:-}" python3 - "$@" <<'PY'
import sys
from aries_common import comms
what = sys.argv[1]
if what == "hosts":
    for name, addr in sorted(comms.hosts().items()):
        print(f"{name} {addr}")
elif what == "prefix":
    print(comms.subnet_prefix())
elif what == "iface":
    print(comms.interfaces().get(sys.argv[2], ""))
elif what == "addr":
    print(comms.hosts().get(sys.argv[2], ""))
elif what == "radio":
    print(comms.radios().get(sys.argv[2], ""))
elif what == "domain":
    print(comms.domain_id())
elif what == "whoami":
    print(comms.local_address() or "")
elif what == "arm":
    from aries_common.devices import device
    print(f"{device('arm.host')} {device('arm.port')}")
PY
}

PREFIX_LEN="$(read_table prefix)"
DOMAIN="$(read_table domain)"

# ---------------------------------------------------------------- check mode
if [ "$MODE" = check ]; then
    head2 "Field link"
    printf '  domain %s, /%s\n' "$DOMAIN" "$PREFIX_LEN"
    read_table hosts | while read -r name addr; do
        printf '  %-6s %s\n' "$name" "$addr"
    done

    head2 "This machine"
    HERE="$(read_table whoami)"
    if [ -z "$HERE" ]; then
        fail "not on the field link (no interface holds a configured address)"
        echo "      Plug the antenna in, or run: $0 {rover|base}"
    else
        WHO="$(PYTHONPATH="$REPO_ROOT/src/aries_common:${PYTHONPATH:-}" \
               python3 -c "from aries_common import comms; print(comms.host_name_for('$HERE') or '')")"
        if [ -n "$WHO" ]; then
            ok "$HERE — this is the $WHO"
        else
            fail "$HERE is not one of the configured hosts; this machine cannot be identified"
        fi
    fi

    head2 "Reachability"
    read_table hosts | while read -r name addr; do
        if [ "$addr" = "$HERE" ]; then continue; fi
        if ping -c1 -W2 "$addr" >/dev/null 2>&1; then ok "$name $addr"
        else warn "$name $addr unreachable (other end powered off? radios not linked?)"; fi
    done
    for who in rover base; do
        radio="$(read_table radio "$who")"
        [ -z "$radio" ] && continue
        if ping -c1 -W2 "$radio" >/dev/null 2>&1; then ok "$who radio $radio"
        else warn "$who radio $radio unreachable"; fi
    done

    head2 "Arm (separate subnet, may share this wire)"
    ARM_ENDPOINT="$(read_table arm)"
    ARM_HOST="${ARM_ENDPOINT%% *}"
    ARM_PORT="${ARM_ENDPOINT##* }"
    if ping -c1 -W2 "$ARM_HOST" >/dev/null 2>&1; then
        ok "arm control box $ARM_HOST responds to ping"
    else
        warn "arm control box $ARM_HOST unreachable (arm powered off, or this is the base station)"
    fi
    # Ping is not enough: the launch files decide real-vs-mock by opening this
    # TCP port, so an arm that pings but does not accept a connection still
    # silently falls back to mock hardware.
    if timeout 2 bash -c "</dev/tcp/$ARM_HOST/$ARM_PORT" 2>/dev/null; then
        ok "arm TCP $ARM_HOST:$ARM_PORT open — launches will pick the real arm"
    else
        warn "arm TCP $ARM_HOST:$ARM_PORT closed — arm_hardware_protocol:=auto will fall back to MOCK"
    fi
    ARM_LOCAL="$(ip -4 -o addr | awk -v n="${ARM_HOST%.*}." '$4 ~ "^"n {print $4; exit}')"
    if [ -n "$ARM_LOCAL" ]; then
        ok "this machine has $ARM_LOCAL on the arm subnet"
    else
        warn "no address on the arm subnet (${ARM_HOST%.*}.0/24) — expected on the rover"
    fi

    head2 "Address conflict"
    # arping finds a SECOND machine answering to our address. On a shared field
    # this is the failure that looks like a flaky link: ARP is last-writer-wins,
    # so traffic goes to whoever replied most recently.
    if [ -n "$HERE" ] && command -v arping >/dev/null 2>&1; then
        IFACE="$(ip -4 -o addr | awk -v a="$HERE" '$4 ~ "^"a"/" {print $2; exit}')"
        if [ -n "$IFACE" ] && sudo -n arping -D -I "$IFACE" -c 2 "$HERE" >/dev/null 2>&1; then
            ok "no other machine claims $HERE"
        else
            warn "could not test (needs sudo), or ANOTHER MACHINE CLAIMS $HERE"
            echo "      sudo arping -D -I $IFACE -c 2 $HERE"
        fi
    else
        warn "arping not installed; cannot check for a duplicate address"
        echo "      sudo apt install iputils-arping"
    fi

    [ "$FAILED" = 1 ] && exit 1
    exit 0
fi

# -------------------------------------------------------------- install mode
if [ -z "$ROLE" ]; then
    echo "usage: $0 {rover|base}" >&2; exit 2
fi

ADDRESS="$(read_table addr "$ROLE")"
IFACE="$(read_table iface "$ROLE")"

if [ -z "$ADDRESS" ]; then
    echo "No address for role '$ROLE' in devices.yaml (network.hosts)." >&2; exit 1
fi
if [ -z "$IFACE" ]; then
    echo "No interface for role '$ROLE' in devices.yaml (network.interface)." >&2; exit 1
fi

head2 "Configuring this machine as: $ROLE"
printf '  address    %s/%s\n' "$ADDRESS" "$PREFIX_LEN"
printf '  interface  %s\n'    "$IFACE"
printf '  connection %s\n'    "$CONN_NAME"

if ! ip link show "$IFACE" >/dev/null 2>&1; then
    fail "no interface named $IFACE on this machine"
    echo "      Available: $(ip -o link show | awk -F': ' '{print $2}' | grep -v lo | tr '\n' ' ')"
    echo "      Fix network.interface.$ROLE in src/aries_common/config/devices.yaml"
    exit 1
fi
ok "interface $IFACE exists"

# Anything already on this port that is NOT on the field-link subnet has to
# survive. On the rover that is the arm: the igus control box is on its own
# /24, and if it hangs off the same switch as the radio then this port carries
# both. Replacing the address list wholesale would take the arm off the
# network, and a dead arm does not look like a network problem when you are
# debugging it an hour later.
ARM_ENDPOINT="$(read_table arm)"
ARM_HOST="${ARM_ENDPOINT%% *}"
ARM_NET="${ARM_HOST%.*}."

KEEP=()
while read -r existing; do
    [ -z "$existing" ] && continue
    case "$existing" in
        "$ADDRESS"/*) continue ;;                       # ours, re-added below
    esac
    KEEP+=("$existing")
done < <(ip -4 -o addr show dev "$IFACE" 2>/dev/null | awk '{print $4}')

if [ ${#KEEP[@]} -gt 0 ]; then
    for k in "${KEEP[@]}"; do
        case "$k" in
            "$ARM_NET"*) ok "preserving arm-subnet address $k on $IFACE" ;;
            *)           warn "preserving unrecognised address $k on $IFACE" ;;
        esac
    done
fi

# Build the address list: ours first, then everything worth keeping.
ADDR_LIST="$ADDRESS/$PREFIX_LEN"
for k in "${KEEP[@]:-}"; do
    [ -n "$k" ] && ADDR_LIST="$ADDR_LIST,$k"
done

# No gateway and no DNS on purpose. This is a point-to-point link to one other
# machine, not a route to the internet: handing it a default route would send
# the rover's general traffic down a radio link, and on the base station it
# would break the operator's normal networking the moment the antenna is
# plugged in. never-default makes that explicit to NetworkManager.
NMCLI_ARGS=(
    con add type ethernet
    con-name "$CONN_NAME"
    ifname "$IFACE"
    ipv4.method manual
    ipv4.addresses "$ADDR_LIST"
    ipv4.never-default yes
    ipv6.method disabled
    connection.autoconnect yes
    # Beat any DHCP profile NetworkManager already has for this port; without
    # it a leftover "Wired connection 1" can win the race on plug-in and you
    # get a DHCP address on a field where nobody is running a DHCP server.
    connection.autoconnect-priority 100
)

if [ "$MODE" = dry-run ]; then
    head2 "Would run"
    echo "  sudo nmcli ${NMCLI_ARGS[*]}"
    echo "  sudo nmcli con up $CONN_NAME"
    exit 0
fi

if nmcli -t -f NAME con show | grep -qx "$CONN_NAME"; then
    CURRENT="$(nmcli -t -f ipv4.addresses con show "$CONN_NAME" | cut -d: -f2-)"
    # Whatever this profile already carries outside the field-link subnet stays:
    # re-running the script must not be the thing that drops the arm.
    for a in ${CURRENT//,/ }; do
        a="${a// /}"
        [ -z "$a" ] && continue
        case "$a" in
            "$ADDRESS"/*) continue ;;
            *) case ",$ADDR_LIST," in *",$a,"*) ;; *) ADDR_LIST="$ADDR_LIST,$a" ;; esac ;;
        esac
    done
    if [ "$CURRENT" = "$ADDR_LIST" ]; then
        ok "connection $CONN_NAME already set to $ADDR_LIST"
    else
        sudo nmcli con mod "$CONN_NAME" \
            ipv4.method manual \
            ipv4.addresses "$ADDR_LIST" \
            ipv4.gateway "" \
            ipv4.never-default yes \
            ipv6.method disabled \
            connection.interface-name "$IFACE"
        changed "connection $CONN_NAME: $CURRENT -> $ADDR_LIST"
    fi
else
    sudo nmcli "${NMCLI_ARGS[@]}"
    changed "created connection $CONN_NAME"
fi

# Any other profile bound to this port will fight ours for it on every plug-in.
# But a profile that carries the ARM's subnet is not a competitor, it is how the
# arm is reachable -- disabling it would silently unplug the arm. Those are
# reported for a human to merge, never touched.
OTHERS="$(nmcli -t -f NAME,DEVICE con show | awk -F: -v i="$IFACE" -v c="$CONN_NAME" \
          '$2 == i && $1 != c {print $1}')"
if [ -n "$OTHERS" ]; then
    while IFS= read -r other; do
        [ -z "$other" ] && continue
        other_addrs="$(nmcli -t -f ipv4.addresses con show "$other" 2>/dev/null | cut -d: -f2-)"
        case "$other_addrs" in
            *"$ARM_NET"*)
                warn "leaving profile '$other' alone — it carries the arm subnet ($other_addrs)"
                echo "      Two autoconnect profiles on $IFACE will fight on plug-in."
                echo "      Merge them by hand once you know which you want to keep:"
                echo "        sudo nmcli con mod '$CONN_NAME' ipv4.addresses '$ADDR_LIST,<arm addr>/24'"
                echo "        sudo nmcli con mod '$other' connection.autoconnect no"
                ;;
            *)
                sudo nmcli con mod "$other" connection.autoconnect no
                changed "disabled autoconnect on competing profile '$other'"
                ;;
        esac
    done <<< "$OTHERS"
fi

if sudo nmcli con up "$CONN_NAME" >/dev/null 2>&1; then
    ok "connection up"
else
    warn "could not bring the connection up — is the cable in?"
    echo "      It will come up on its own when the antenna is plugged in."
fi

head2 "Verify"
echo "  ./scripts/setup_field_link.sh --check"
echo
echo "  Then, on BOTH machines, in every shell you type ros2 commands in:"
echo "    source \"\$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh\""

[ "$FAILED" = 1 ] && exit 1
exit 0
