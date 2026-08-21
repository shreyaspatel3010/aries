# ARIES communication environment.  Source this, do not execute it:
#
#     source ~/aries/communication/comms_env.sh
#
# Every ROS process must agree on these three or it will see nothing at all --
# no error, just an empty `ros2 topic list`.  A process keeps whatever it was
# started with, so a terminal opened before these were set stays on the old
# domain forever; that is the single most common cause of "no topics".

_COMMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${_COMMS_DIR}/cyclonedds.xml"

# The ros2 CLI daemon caches the node graph per (domain, rmw).  After a change
# it will happily serve the old, empty graph until it is restarted.
ros2 daemon stop >/dev/null 2>&1

echo "ARIES comms: domain $ROS_DOMAIN_ID, $RMW_IMPLEMENTATION"
echo "             $CYCLONEDDS_URI"

unset _COMMS_DIR
