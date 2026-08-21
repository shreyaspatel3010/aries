================================================================================
 ARIES operator view -- copy this folder, run it, no build
================================================================================

  aries_operator/
      operator.launch.py         the whole thing
      operator.rviz              camera layout
      dds_config.py              generates the DDS transport config per machine
      operator_env.sh            source in any shell you type ros2 commands in
      README.txt

--------------------------------------------------------------------------------
 Put it on another computer
--------------------------------------------------------------------------------

# from this machine
tar czf aries_operator.tar.gz -C ~/aries/communication aries_operator
scp aries_operator.tar.gz user@othermachine:~

# on that machine
tar xzf aries_operator.tar.gz
sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp \
                    ros-jazzy-image-transport-plugins ros-jazzy-rviz2
source /opt/ros/jazzy/setup.bash
ros2 launch ~/aries_operator/operator.launch.py

Nothing to build, nothing to edit. The interface address is detected from
whichever NIC holds an address on the rover's subnet, so the same folder works
unchanged on every machine.

    ros2 launch .../operator.launch.py cameras:=rover_camera
    ros2 launch .../operator.launch.py use_rviz:=false      # decompress only
    ARIES_ROVER_IP=10.0.0.5 ros2 launch .../operator.launch.py

--------------------------------------------------------------------------------
 What it shows
--------------------------------------------------------------------------------

    /downlink/<camera>/color/compressed       ->  /<camera>/view/color
    /downlink/<camera>/depth/compressedDepth  ->  /<camera>/view/depth

Two Image displays and two DepthClouds, reading the local decompressed copies.

It does NOT show the robot model. RViz loads meshes from the local filesystem
through package:// paths, and those live in the aries packages -- a machine
without them would only log resource errors. /robot_description and /tf do
cross the link, so if the target machine has the aries workspace installed you
can add a RobotModel display yourself and it will work.

--------------------------------------------------------------------------------
 Other shells on that machine
--------------------------------------------------------------------------------

source ~/aries_operator/operator_env.sh
rqt_image_view                  # pick /rover_camera/view/color
ros2 topic list | grep ^/downlink/

A process keeps the DDS environment it STARTED with. A terminal opened before
you sourced this keeps the old domain forever, and anything launched from it
inherits that -- which looks exactly like a dead link: ping fine, topic list
empty, no error anywhere.

--------------------------------------------------------------------------------
 Nothing appears
--------------------------------------------------------------------------------

  ping <rover ip>                            link up?
  env | grep -E "ROS_DOMAIN|RMW_|CYCLONE"    this shell set up?
  ros2 topic list | grep ^/downlink/         rover publishing?
  ros2 daemon stop                           stale cached graph

  Topic lists but never delivers a frame -> QoS. The rover publishes
  BEST_EFFORT; a RELIABLE subscriber will not match it at all.

  Everything empty but ping works -> domain or RMW mismatch. Both machines
  must agree on ROS_DOMAIN_ID and RMW_IMPLEMENTATION.

  Check the generated transport config actually names this machine:
      cat cyclonedds.generated.xml
