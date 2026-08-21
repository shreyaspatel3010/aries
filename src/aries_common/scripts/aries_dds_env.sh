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

_aries_dds_env() {
    local out
    # require_link=False: a machine with no antenna gets a local-only config
    # rather than an error. This runs from ~/.bashrc on developer laptops that
    # only ever run simulation, and an interface pin they do not have is FATAL
    # -- Cyclone refuses to create the domain and every node in a launch dies
    # at startup, which is a spectacular way to break a machine that was fine.
    if ! out="$(python3 -c 'import aries_common.comms as c; print(c.domain_id()); p,a=c.write_cyclone_config(require_link=False); print(p); print(a or "")' 2>&1)"; then
        echo "ARIES comms: could not configure DDS; leaving this shell alone." >&2
        echo "$out" >&2
        # Clear rather than leave a stale URI behind: an unset CYCLONEDDS_URI
        # is a working default, a wrong one stops every node from starting.
        unset CYCLONEDDS_URI
        return 1
    fi

    local domain path address
    domain="$(echo "$out" | sed -n 1p)"
    path="$(echo "$out" | sed -n 2p)"
    address="$(echo "$out" | sed -n 3p)"

    export ROS_DOMAIN_ID="$domain"
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    # Deliberately overwritten rather than preserved: a stale value carried in
    # from this shell is the exact thing being corrected.
    export CYCLONEDDS_URI="file://${path}"

    if [ -n "$address" ]; then
        echo "ARIES comms: domain $ROS_DOMAIN_ID, $RMW_IMPLEMENTATION, interface $address"
    else
        echo "ARIES comms: domain $ROS_DOMAIN_ID, $RMW_IMPLEMENTATION, LOCAL ONLY"
        echo "             (not on the field link — fine for simulation; run"
        echo "              scripts/setup_field_link.sh before going to the field)"
    fi
    echo "             $CYCLONEDDS_URI"

    # The ros2 CLI daemon caches the node graph per (domain, rmw) and will
    # happily serve the old, empty one after a change.
    ros2 daemon stop >/dev/null 2>&1
}

_aries_dds_env
unset -f _aries_dds_env
