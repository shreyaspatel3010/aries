# ARIES DDS environment. Source this, do not execute it:
#
#     source "$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh"
#
# The same script on the rover and on the base station. The interface address
# is detected, not written down, so there is no per-machine copy to keep in
# sync -- see aries_common/comms.py for why a mirrored file is the failure this
# exists to prevent.
#
# Launch files set this environment for themselves. This script is for shells:
# rviz2, rqt_image_view and `ros2 topic list` are started by hand and a launch
# file cannot reach into your terminal. A process keeps whatever it was STARTED
# with for as long as it lives, so a terminal opened before these were exported
# stays on the old domain forever -- which looks exactly like a dead link: ping
# fine, empty topic list, no error anywhere.
#
# THE VARIABLES ARE NOT LISTED HERE ON PURPOSE. This script exports whatever
# comms.dds_environment() returns, which differs per middleware -- Fast DDS
# wants FASTRTPS_/FASTDDS_DEFAULT_PROFILES_FILE, Cyclone wants CYCLONEDDS_URI.
# Hard-coding either set in shell is how a launch file and a terminal end up on
# different transports while both look correct.

_aries_dds_env() {
    local out
    # require_link=False: a machine with no antenna gets a local-only config
    # rather than an error. This runs from ~/.bashrc on developer laptops that
    # only ever run simulation, and an interface pin they do not have is FATAL
    # -- the middleware refuses to create the domain and every node in a launch
    # dies at startup, which is a spectacular way to break a machine that was
    # fine.
    if ! out="$(python3 -c '
import aries_common.comms as c
env = c.dds_environment(require_link=False)
for k, v in env.items():
    print("%s=%s" % (k, v))
print("_ARIES_ADDRESS=%s" % (c.local_address() or ""))
' 2>&1)"; then
        echo "ARIES comms: could not configure DDS; leaving this shell alone." >&2
        echo "$out" >&2
        # Clear rather than leave a stale config behind: an unset variable is a
        # working default, a wrong one stops every node from starting. Both
        # vendors' variables, because the shell may be carrying the other one
        # from before a middleware change.
        unset CYCLONEDDS_URI
        unset FASTRTPS_DEFAULT_PROFILES_FILE
        unset FASTDDS_DEFAULT_PROFILES_FILE
        return 1
    fi

    # Drop the other vendor's leftovers BEFORE exporting, so switching
    # middleware in a live shell cannot leave a stale pointer behind.
    unset CYCLONEDDS_URI
    unset FASTRTPS_DEFAULT_PROFILES_FILE
    unset FASTDDS_DEFAULT_PROFILES_FILE

    local address="" line key value
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        key="${line%%=*}"
        value="${line#*=}"
        if [ "$key" = "_ARIES_ADDRESS" ]; then
            address="$value"
        else
            export "$key=$value"
        fi
    done <<EOF
$out
EOF

    if [ -n "$address" ]; then
        echo "ARIES comms: domain $ROS_DOMAIN_ID, $RMW_IMPLEMENTATION, interface $address"
    else
        echo "ARIES comms: domain $ROS_DOMAIN_ID, $RMW_IMPLEMENTATION, LOCAL ONLY"
        echo "             (not on the field link — fine for simulation; run"
        echo "              scripts/setup_field_link.sh before going to the field)"
    fi
    echo "             ${FASTDDS_DEFAULT_PROFILES_FILE:-$CYCLONEDDS_URI}"

    # The ros2 CLI daemon caches the node graph per (domain, rmw) and will
    # happily serve the old, empty one after a change. It is especially
    # important after a MIDDLEWARE change: the cached graph is from the other
    # vendor entirely and looks like a stack that half-vanished.
    ros2 daemon stop >/dev/null 2>&1
}

_aries_dds_env
unset -f _aries_dds_env
