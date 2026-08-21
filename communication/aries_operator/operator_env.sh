# ARIES operator environment. Source this, do not execute it:
#
#     source ./operator_env.sh
#
# Needed by anything you start BY HAND -- rqt_image_view, rviz2, ros2 topic
# list. operator.launch.py sets its own environment for the nodes it spawns,
# but it cannot reach into your shell.

_OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Not ${ROS_DOMAIN_ID:-30}: that would preserve a stale domain from this shell,
# which is the exact failure this script exists to correct.
export ROS_DOMAIN_ID="${ARIES_DOMAIN_ID:-30}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Regenerated every time: the interface address is detected, so the same
# checkout works on any machine without editing a file.
if _OP_XML="$(python3 "$_OP_DIR/dds_config.py" 2>/dev/null | tail -1)"; then
    export CYCLONEDDS_URI="file://${_OP_XML}"
    echo "ARIES operator: domain $ROS_DOMAIN_ID, cyclonedds"
    echo "                $CYCLONEDDS_URI"
else
    echo "ARIES operator: WARNING - no interface on the rover subnet." >&2
    echo "                Check the cable, or set ARIES_ROVER_IP." >&2
fi

# The ros2 CLI daemon caches the node graph per (domain, rmw) and will serve
# the old, empty one after a change.
ros2 daemon stop >/dev/null 2>&1

unset _OP_DIR _OP_XML
