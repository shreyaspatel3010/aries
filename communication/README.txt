================================================================================
 ARIES — communication  (DDS domain 30, Cyclone DDS)
================================================================================

  communication/
      cyclonedds.xml            DDS transport config (interfaces, peers)
      comms_env.sh              source this in ANY terminal
      communication.launch.py   launches the comms layer on domain 30
      README.txt                this file
      aries_operator/           SELF-CONTAINED viewer for another machine:
                                copy the folder, run it, no build. See its
                                own README.txt.

  operator / this PC   192.168.1.11   enp130s0
  rover                192.168.1.10


--------------------------------------------------------------------------------
 0. "ros2 topic list is empty" / "no topics in rqt_image_view"
--------------------------------------------------------------------------------

Almost always this, and it is not a network fault:

    A process keeps the DDS environment it was STARTED with, forever.

~/.bashrc is read once, when a shell starts. A terminal you opened before the
exports existed still has the old domain, and every node launched from it
inherits that. The link is fine, ping is fine, nothing logs an error -- the
two sides are just on different domains and cannot see each other at all.

    Check what a running process actually has:
        tr '\0' '\n' < /proc/<pid>/environ | grep -E "ROS_DOMAIN_ID|RMW_"
        (no output = unset = domain 0 + fastrtps, NOT domain 30)

    Fix, in that terminal:
        source ~/aries/communication/comms_env.sh
    then restart whatever it launched. Sourcing does not move running nodes.

The launch file sets the domain itself, so nodes it starts are correct
regardless of the terminal. Your interactive shell is still not -- rqt, RViz
and ros2 topic list are started by hand and need comms_env.sh.


--------------------------------------------------------------------------------
 1. Normal use
--------------------------------------------------------------------------------

# every terminal, first thing
source ~/aries/communication/comms_env.sh

# ROVER — CAMERAS ONLY (no arm, no controllers, no MoveIt: ~6 nodes not ~30)
# This is the one to use when the job is to look through the rover rather than
# drive it. It brings up the drivers, the compression chain, and enough TF for
# a depth cloud to have somewhere to sit.
ros2 launch aries_bringup cameras.launch.py
ros2 launch aries_bringup cameras.launch.py cameras:=rover_camera
ros2 launch aries_bringup cameras.launch.py downlink_profile:=lean
ros2 launch aries_bringup cameras.launch.py use_rviz:=true       # check locally

# ROVER — compress and send, as part of the full stack
ros2 launch ~/aries/communication/communication.launch.py side:=rover
ros2 launch ~/aries/communication/communication.launch.py side:=rover downlink_profile:=lean

# OPERATOR — receive and decompress, then read /<camera>/view/*
ros2 launch ~/aries/communication/communication.launch.py side:=operator
ros2 launch ~/aries/communication/communication.launch.py side:=operator use_rviz:=true

# ONE MACHINE, everything on domain 30 in a single command
# (this is the one that fixes "no topics" -- it puts the robot stack on the
#  same domain instead of leaving it on whatever the terminal had)
# With stack:=true, full_hardware provides the camera chain itself, so this
# layer adds nothing on top and side is ignored. Do NOT also launch
# camera_view/camera_downlink by hand: two publishers on one topic run at
# double rate with interleaved frames, which reads as jitter.
ros2 launch ~/aries/communication/communication.launch.py stack:=true

# ANOTHER COMPUTER, view only, nothing to build
tar czf aries_operator.tar.gz -C ~/aries/communication aries_operator
scp aries_operator.tar.gz user@other:~ && ssh user@other
tar xzf aries_operator.tar.gz && ros2 launch ~/aries_operator/operator.launch.py

# viewers -- source comms_env.sh first, or they land on domain 0 and see nothing
rqt_image_view          # then pick /rover_camera/view/color
rviz2


--------------------------------------------------------------------------------
 2. Install (once per machine)
--------------------------------------------------------------------------------

sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp     # pulls cyclonedds + iceoryx

# add to ~/.bashrc, after sourcing ROS:
source ~/aries/communication/comms_env.sh

# THE ROVER NEEDS THE MIRROR OF cyclonedds.xml, NOT A COPY:
#     this PC : NetworkInterface 192.168.1.11 , Peer 192.168.1.10
#     rover   : NetworkInterface 192.168.1.10 , Peer 192.168.1.11
# Copied unchanged, the rover binds an address it does not have and Cyclone
# quietly falls back to picking its own interface.
#
# AllowMulticast=false with explicit <Peers> is deliberate -- the field link
# drops multicast. It also means a third machine must be added to <Peers> by
# hand; it will not discover the others on its own.


--------------------------------------------------------------------------------
 3. Verify, bottom up   (stop at the first failure)
--------------------------------------------------------------------------------

# 3a. link
ip -4 -br addr                       # enp130s0 UP, 192.168.1.11/24
ping -c3 192.168.1.10

# 3b. env — must match EXACTLY on both machines
env | grep -E "RMW_|ROS_DOMAIN|CYCLONEDDS"

# 3c. middleware actually loaded, not merely requested
ros2 doctor --report | grep -i middleware

# 3d. is the XML really being read?  --log-level debug will NOT tell you:
#     Cyclone's tracing is its own setting, not a ROS log level. Append a
#     fragment to the URI (it takes a comma-separated list, so this adds to
#     the file rather than replacing it):
CYCLONEDDS_URI="$CYCLONEDDS_URI,<Tracing><Verbosity>config</Verbosity><OutputFile>stderr</OutputFile></Tracing>" \
  ros2 run demo_nodes_cpp talker 2>&1 | grep -E "NetworkInterface\[@address\]|AllowMulticast"
#     Expect:  NetworkInterface[@address]: 192.168.1.11 {2}
#              AllowMulticast/#text: false {2}
#     The {2} means the value came from the file. A bare {} is the built-in
#     default -- your XML was never loaded.

# 3e. end to end
ros2 run demo_nodes_cpp talker       # rover
ros2 run demo_nodes_cpp listener     # operator

# 3f. the graph
ros2 node list
ros2 topic list
ros2 daemon stop                     # the CLI daemon caches a graph per
                                     # (domain, rmw) and serves it stale

# 3g. which domain is a thing actually on?  sweep them:
for d in 0 30; do for r in rmw_cyclonedds_cpp rmw_fastrtps_cpp; do
  n=$(ROS_DOMAIN_ID=$d RMW_IMPLEMENTATION=$r timeout 8 ros2 topic list 2>/dev/null | wc -l)
  echo "domain $d $r -> $n topics"; done; done


--------------------------------------------------------------------------------
 4. Topic layout
--------------------------------------------------------------------------------

Everything the antenna carries sits under ONE top-level prefix, so the
operator can list exactly what crosses the link instead of hunting through
the camera's own tree (where every raw topic also advertises four
image_transport codec names that nothing publishes):

    ros2 topic list | grep ^/downlink/

    /downlink/<camera>/color/compressed        JPEG     BEST_EFFORT, depth 1
    /downlink/<camera>/depth/compressedDepth   PNG16    BEST_EFFORT, depth 1
    /downlink/<camera>/camera_info             latched

    /<camera>/downlink_src/color               rover-internal, RAW.
    /<camera>/downlink_src/depth               Never subscribe across the link.

    /<camera>/view/color                       operator-local, decompressed
    /<camera>/view/depth                       point displays at these

BEST_EFFORT is not a detail. image_transport publishes RELIABLE by default,
which is the wrong contract for video on a lossy link: a lost fragment makes
the writer retransmit, frames queue to the history depth, and the operator
watches several hundred ms in the past with no way to catch up. Depth 1 keeps
only the newest frame. Both ends must agree -- a RELIABLE subscriber against
a BEST_EFFORT publisher is an incompatible pair and never matches at all, so
the topic lists fine and simply never delivers.


--------------------------------------------------------------------------------
 5. Bandwidth
--------------------------------------------------------------------------------

ros2 run aries_bringup downlink_report.py            # run on the OPERATOR side
ros2 run aries_bringup downlink_report.py --ros-args -p seconds:=30.0

ros2 topic hz /downlink/rover_camera/color/compressed
ros2 topic bw /downlink/rover_camera/color/compressed
ros2 topic info /downlink/rover_camera/color/compressed --verbose

# NEVER subscribe to a rover RAW image topic across the link. RViz's Image
# display has no transport selection -- it subscribes raw -- and one such
# display puts ~369 Mbit/s of uncompressed pixels on the wire. Compressed is
# ~28 Mbit/s for both cameras. Always read /<camera>/view/* locally.
# ros2 topic hz/bw subscribe like any other node: same rule applies to them.

iftop -i enp130s0
ethtool enp130s0 | grep -i speed


--------------------------------------------------------------------------------
 6. Symptoms
--------------------------------------------------------------------------------

  empty topic list, ping works
      -> domain / RMW mismatch. Section 0. Check /proc/<pid>/environ, not
         just your shell.

  topics listed, no data
      -> QoS mismatch (downlink is BEST_EFFORT/VOLATILE), or a Reliable
         writer retransmitting into a saturated link and stalling.
         ros2 topic info --verbose

  duplicate node names in ros2 node list
      -> two launches alive at once. pgrep -f "ros2 launch"

  worked, then died after a replug
      -> new ifindex, old sockets dead. Restart nodes both ends.

  one direction only / intermittent
      -> firewall; Cyclone needs UDP both ways.
         sudo ufw allow from 192.168.1.0/24

  nodes listed that no longer exist
      -> ros2 daemon stop
