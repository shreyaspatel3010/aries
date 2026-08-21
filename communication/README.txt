================================================================================
 communication/
================================================================================

  LIVE:        stop_comms.sh    stop the whole stack, and repair this shell
  SUPERSEDED:  everything else -- it moved into the workspace, see below.


--------------------------------------------------------------------------------
 stop_comms.sh
--------------------------------------------------------------------------------

  ./communication/stop_comms.sh              stop everything
  ./communication/stop_comms.sh --status     list what is running, change nothing
  ./communication/stop_comms.sh --force      skip the graceful stage
  source ./communication/stop_comms.sh       stop everything AND fix this shell

Stops every node, launch process, Gazebo and the ros2 CLI daemon. SIGINT first
so `ros2 launch` shuts its children down in order -- controllers deactivate and
the ODrive bridge gets to publish a final zero; SIGKILL only for what refuses.

SOURCE IT if `ros2 topic list` errors with NOTHING RUNNING:

    can't open configuration file file:///.../cyclonedds.xml
    rmw_create_node: failed to create domain, error Error

That is your terminal, not the robot. A process keeps the DDS environment it
was STARTED with, so a shell opened before ~/.bashrc was fixed carries the old
value forever, and Cyclone treats a config it cannot open as fatal -- it
refuses to create the domain, so every command dies. Killing nodes cannot fix a
shell; sourcing this re-applies the workspace environment to it.

It only ever considers processes under this workspace, under /opt/ros, or
Gazebo, it never matches a shell, and it prints the list before signalling
anything.


================================================================================
 SUPERSEDED -- the rest of this folder moved into the workspace
================================================================================

Everything here now lives in packages both machines build, because both
machines run the same workspace. This folder was never committed to git, so the
rover -- deployed by pulling the repo -- did not have any of it.

  WAS                                  IS NOW
  ---                                  ------
  comms_env.sh                         aries_common/scripts/aries_dds_env.sh
                                         source "$(ros2 pkg prefix aries_common)\
/share/aries_common/aries_dds_env.sh"

  cyclonedds.xml                       generated per machine by
  (now .superseded)                    aries_common/aries_common/comms.py
                                       Addresses live in the network: section
                                       of aries_common/config/devices.yaml.

  communication.launch.py stack:=true  ros2 launch aries_bringup \
                                         rover_field.launch.py

  communication.launch.py side:=operator   ros2 launch aries_base_station \
  aries_operator/operator.launch.py          base_station.launch.py


--------------------------------------------------------------------------------
 Normal use now
--------------------------------------------------------------------------------

  FULL SETUP GUIDE: FIELD_SETUP.md at the repo root. It covers static
  addressing, the radio settings, and a competition-day checklist. The
  addresses also moved off 192.168.1.0/24 -- that is the airOS factory subnet
  and every other team using Ubiquiti is on it.

  ROVER                 ros2 launch aries_bringup rover_field.launch.py
  BASE STATION          ros2 launch aries_base_station base_station.launch.py

Plug the joystick into the BASE STATION. rover_field defaults to
use_joy_node:=false and base_station defaults to use_joy_node:=true, so exactly
one machine reads the pad; the teleop nodes keep running on the rover and take
/joy across the link.

Shells you start things in by hand (rviz2, rqt_image_view, ros2 topic list):

  source "$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh"


--------------------------------------------------------------------------------
 The one thing that changed behaviour
--------------------------------------------------------------------------------

cyclonedds.xml hardcoded NetworkInterface 192.168.1.11. Copied to the rover
unchanged it named an address that machine does not have, and Cyclone does not
treat that as fatal -- it warns, picks its own interface, and you get an empty
`ros2 topic list` on a link that pings fine.

The address is now DETECTED from whichever interface is on the field link, so
one file is correct on both machines and there is nothing to mirror by hand.

If ~/.bashrc still has this line, delete it. It is wrong on every machine but
one, and the launch files now overwrite it and log that they did:

    export CYCLONEDDS_URI=file:///home/shreyas/aries/communication/cyclonedds.xml


--------------------------------------------------------------------------------
 aries_operator/
--------------------------------------------------------------------------------

Superseded by the aries_base_station package. It existed for a machine with no
workspace; with the workspace on both ends, base_station.launch.py also renders
the robot model, which aries_operator could not.

The old troubleshooting notes in the git history of this file still apply --
domain mismatch, QoS mismatch, stale ros2 daemon, firewall. See
src/aries_base_station/README.md, which carries them forward.
