"""
aries_hardware.launch.py

Mixed real/fake hardware launch for Aries:
- arm: auto-detect igus ReBeL, otherwise mock_hardware
- gripper: auto-detect Teensy serial device, otherwise mock_hardware

Important detail:
ros2_control controller definitions are written to a temporary YAML file,
because controller_manager expects the multi-node YAML layout
(controller_manager + per-controller sections).
"""

import glob
import os
import socket
import tempfile
import time
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from aries_common.comms import write_agent_dds_config
from aries_common.devices import device, device_str
from aries_common.gripper_cal import cal_str

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["gripper_gear_left_joint"]

# Every gripper backend that has a controller to spawn. "st3215" is the
# secondary gripper's servo, reached over the rover PC's own USB bus-servo
# adapter; "rebel" is the primary gripper's servo behind the Teensy. They are
# separate protocols because they are separate WIRES, not because the URDFs
# differ - that part is gripper_type.
GRIPPER_PROTOCOLS = ("rebel", "st3215", "mock_hardware", "gazebo")

# The board ID is baked into the by-id path, so swapping the Teensy changes it
# (16739090 -> 20379650 on 2026-08-12) and an exact-path check then resolves to
# mock_hardware against a perfectly healthy board. Treat the configured path as
# a preference and accept any Teensy. full_hardware_checker globs the same way,
# which is why it kept reporting "Gripper serial connected" while this probe
# fell back to mock and no command reached the servo.
TEENSY_BY_ID_GLOB = "/dev/serial/by-id/*Teensy*-if00"


def resolve_gripper_serial(configured: str, detect_timeout: float):
    """Find the DRILL Teensy to talk to. Returns (port_or_None, note_for_log).

    Waits up to detect_timeout for the device: a Teensy reset re-enumerates over
    USB, which takes 1-2 s, so relaunching straight after a reset loses the race.
    Observed one probe at 17:10:58 with the by-id link appearing at 17:10:59.2 --
    a one-shot check ran the whole session on a simulated gripper.

    THE SINGLE-TEENSY FALLBACK IS NOW CONDITIONAL, and that changed when the
    science board arrived. This used to take found[0] whenever the configured
    path was missing, on the reasoning that if exactly one Teensy is plugged in
    it must be the one meant. That reasoning is sound and its premise stopped
    being true: there are two boards now, they enumerate with the same vendor
    and product ID, and they differ only by serial number -- so found[0] is
    whichever serial sorts first, which is arbitrary.

    Taking it anyway would point the gripper's hardware interface, the drill
    driver and the stack light at the SCIENCE board, which answers none of
    those topics. Every one of them would then report a connected-but-silent
    link, which is the hardest failure on this rover to read.

    So: fall back only when there is exactly one candidate AND it is not the
    board configured for something else. With two present and neither matching,
    refuse and say why.
    """
    other_ports = _other_configured_teensy_ports(exclude="gripper")

    deadline = time.monotonic() + detect_timeout
    while True:
        if Path(configured).exists():
            return configured, ""

        found = sorted(glob.glob(TEENSY_BY_ID_GLOB))
        # Never fall back onto a port another board explicitly claims.
        candidates = [p for p in found if p not in other_ports]

        if len(candidates) == 1:
            note = (f"  -- {configured} is absent, using the only unclaimed "
                    f"Teensy present: {candidates[0]}")
            return candidates[0], note

        if len(candidates) > 1:
            return None, (
                f"  -- {configured} is absent and {len(candidates)} unclaimed "
                f"Teensys are connected ({', '.join(candidates)}); REFUSING TO "
                f"GUESS. They differ only by serial number, so picking one "
                f"risks driving the gripper against the science board. Fix "
                f"gripper.serial_port in devices.yaml.")

        if time.monotonic() >= deadline:
            return None, ""
        time.sleep(0.1)


def _other_configured_teensy_ports(exclude: str):
    """by-id paths that devices.yaml assigns to a Teensy OTHER than `exclude`.

    Used so that a board looking for itself never falls back onto a port
    another board has claimed. Returns a set; an unset or empty entry
    contributes nothing.
    """
    ports = set()
    for key in ("gripper.serial_port", "science.serial_port"):
        if key.split(".")[0] == exclude:
            continue
        try:
            value = device_str(key)
        except Exception:
            continue
        if value:
            ports.add(value)
    return ports


def resolve_science_serial(configured: str):
    """Find the SCIENCE Teensy. Returns (port_or_None, note_for_the_log).

    NO FALLBACK AND NO WAIT, deliberately, and both differ from the drill board
    above. The drill board carries the gripper, the stack light and the drill
    itself, so it is worth waiting for and worth a careful guess; the science
    board carries sensors that are read on demand, so a missing one costs
    telemetry and nothing else.

    More to the point, a wrong guess here is worse than no board: pointing the
    science agent at the DRILL board gives it a session, an entity set, and no
    /science/telemetry ever -- while stealing nothing, because both agents can
    open different ports but not the same one. Silence is the honest outcome.
    """
    if not configured:
        return None, ("  -- science.serial_port is not set in devices.yaml. "
                      "Plug the science board in ALONE, run "
                      "`ls /dev/serial/by-id/`, and paste the path there.")
    if Path(configured).exists():
        return configured, ""
    return None, f"  -- {configured} is absent"


# Teensy bus-servo bridge, TEMPORARY -- see firmware/teensy_drill_sys/lib/servobus.
# A Teensy running that firmware puts its second USB serial device on the ST3215
# bus, so it can stand in for the adapter. The glob is deliberately narrow:
#
#   Dual_Serial  the bridge firmware is built -D USB_DUAL_SERIAL, which changes
#                the USB product string from "USB Serial". No OTHER Teensy here
#                carries servobus, and none of them says Dual_Serial -- so the
#                name IS the capability. A drill or science board cannot match.
#   -if02        the SECOND CDC interface, which is the bridge. -if00 is the
#                micro-ROS transport; matching it would push servo packets into
#                the agent's link.
SERVO_BUS_TEENSY_GLOB = "/dev/serial/by-id/usb-Teensyduino_Dual_Serial_*-if02"


def resolve_servo_bus(configured: str):
    """Find the ST3215 bus-servo adapter. Returns (port_or_None, note).

    Unlike the Teensy there is no by-id fallback worth taking FOR THE ADAPTER.
    /dev/aries_servo_bus comes from 99-aries-servo-bus.rules matching this
    adapter's serial, and the by-id name a CH340 would otherwise get
    (usb-1a86_USB_Serial-if00-port0) is shared by every CH340 on earth --
    stable but not unique. Guessing at one would mean commanding whatever
    generic USB-serial device happened to be plugged in.

    A TEENSY BRIDGE IS DIFFERENT AND IS TRIED SECOND. Its by-id name carries
    both the firmware's own signature (Dual_Serial) and a per-board serial
    number, so there is nothing to guess at: a match is a board someone
    deliberately flashed with the bridge firmware, not a coincidence. That is
    the whole reason this fallback is safe and the CH340 one is not.

    Order is adapter first, always. The bridge is a stopgap while the adapter
    is dead; the moment a real adapter is plugged back in it wins, with no
    config change and no argument.
    """
    if Path(configured).exists():
        return configured, ""

    bridges = sorted(glob.glob(SERVO_BUS_TEENSY_GLOB))
    if len(bridges) == 1:
        return bridges[0], (f"  -- {configured} is absent; using the TEENSY BRIDGE at "
                            f"{bridges[0]}. Temporary: it is a wire, not the adapter. "
                            "Plug a real adapter in and it takes over automatically.")
    if len(bridges) > 1:
        # Never pick one. Two bridges means two boards that could each be on a
        # different bus, and commanding the wrong one is a gripper that moves
        # when it should not.
        listed = ", ".join(bridges)
        return None, (f"  -- {configured} is absent and MORE THAN ONE Teensy bridge is "
                      f"attached ({listed}); refusing to guess. Unplug one, or pass "
                      "servo_bus_port:= explicitly")

    return None, (f"  -- {configured} is absent; run scripts/setup_system.sh to install "
                  "99-aries-servo-bus.rules, and check the adapter is plugged in. "
                  "(No Teensy bridge found either -- see "
                  "firmware/teensy_drill_sys/lib/servobus.)")


def build_ros2_control_yaml(arm_protocol: str, gripper_protocol: str) -> str:
    arm_command_interface = "velocity" if arm_protocol == "rebel" else "position"

    data = {
        "controller_manager": {
            "ros__parameters": {
                # 50 Hz for real rebel arm: each velocity command is held for 20 ms,
                # so the 100 Hz ALIVEJOG sends it twice before the next one arrives.
                # This naturally smooths the commanded velocity and reduces jitter.
                "update_rate": 80,
                "joint_state_broadcaster": {
                    "type": "joint_state_broadcaster/JointStateBroadcaster"
                },
                "rebel_arm_trajectory_controller": {
                    "type": "joint_trajectory_controller/JointTrajectoryController"
                },
            }
        },
        "joint_state_broadcaster": {
            "ros__parameters": {}
        },
        "rebel_arm_trajectory_controller": {
            "ros__parameters": {
                "joints": ARM_JOINTS,
                "command_interfaces": [arm_command_interface],
                "state_interfaces": ["position", "velocity"],
                "state_publish_rate": 80.0,
                "action_monitor_rate": 40.0,
                "allow_nonzero_velocity_at_trajectory_end": True,
            }
        },
    }

    if arm_protocol == "rebel":
        # JOINT_VELOCITY_SCALE=2.0 means effective max vel_cmd ≈ 0.87 rad/s,
        # so gain p saturates at a position error of 0.87/p rad.
        #   p=10 -> saturates at  5.0 deg   (the old jitter: measured release
        #                                    overshoot is 4-9 deg, i.e. right
        #                                    inside the saturating range)
        #   p= 3 -> saturates at 16.6 deg   (clear of it)
        #   p= 1 -> saturates at 49.8 deg   but brakes with only 0.07 rad/s at
        #                                   a 4 deg error, i.e. barely at all
        #
        # Measured with scripts/measure_teleop_tracking.py on the real arm
        # (27 clean releases): tracking lag 85 ms, release overshoot mean
        # 4.03 deg / worst 8.94 deg, against a designed lead of 1.38 deg. That
        # decomposes as v*lookahead (1.38) + v*lag (2.34) + the arm's own decel
        # (0.32) = 4.04 deg, which is the whole of the measured value.
        #
        # DO NOT RAISE p. Measured twice on hardware with
        # scripts/measure_teleop_tracking.py, the second time with the hold
        # target confirmed latched (holddrift 0.00 deg), so the JTC genuinely
        # had a growing position error to brake against:
        #
        #   p=1.0  stopping time 0.244 s, release overshoot 3.15 deg
        #   p=3.0  stopping time 0.339 s, release overshoot 5.78 deg   WORSE
        #
        # Raising the gain makes it worse because the loop has ~75 ms of
        # transport delay: the braking command is computed from an error that
        # is already 75 ms stale and lands 75 ms later still, so more gain
        # simply drives the loop under-damped and the peak displacement grows.
        # The d term compounds it, because igus_rebel/src/Rebel.cpp fills the
        # velocity state with an unfiltered first difference of a quantised
        # position (Rebel.cpp:683) and p=3 amplifies that noise.
        #
        # Saturation is NOT the limit here and was a red herring:
        # JOINT_VELOCITY_SCALE is 1.0 (Rebel.hpp:22 — the "2.0" that used to be
        # claimed in this comment was stale), so against the 1.5 rad/s rating
        # p=3 would not saturate until a 28 deg error, far past the ~6 deg
        # worst case observed. Dead time, not authority, is the constraint.
        #
        # The lever that does work is max_joint_velocity in
        # config/teleop_speeds.yaml: overshoot = release speed * ~0.25 s.
        data["rebel_arm_trajectory_controller"]["ros__parameters"]["gains"] = {
            # d damps the position-error correction so the arm does not
            # oscillate when the velocity stream stops (joystick centered).
            # ff_velocity_scale=1.0 keeps trajectory feed-forward as the primary
            # motion driver, so planned RViz trajectories are unaffected.
            "joint1": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint2": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint3": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint4": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint5": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint6": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
        }

    if gripper_protocol in GRIPPER_PROTOCOLS:
        data["controller_manager"]["ros__parameters"]["rebel_gripper_controller"] = {
            "type": "joint_trajectory_controller/JointTrajectoryController"
        }
        data["rebel_gripper_controller"] = {
            "ros__parameters": {
                "joints": GRIPPER_JOINTS,
                "command_interfaces": ["position"],
                # POSITION AND VELOCITY ONLY, on both grippers.
                #
                # The ST3215 hardware component exports an effort state
                # interface as well, and it is worth having - it is a real
                # measurement, not an echo. But it must NOT be listed here:
                # joint_trajectory_controller validates this list against
                # {position, velocity, acceleration} and REFUSES TO INITIALISE
                # on anything else. Measured against the servo emulator:
                #
                #   Invalid value set during initialization for parameter
                #   'state_interfaces': Entry 'effort' ... is not in the set
                #   '{position, velocity, acceleration}'
                #
                # which surfaces only as "Could not initialize the controller
                # named 'rebel_gripper_controller'" in the spawner - i.e. no
                # gripper at all, from adding a field that looked free.
                # joint_state_broadcaster publishes every available state
                # interface regardless, so the effort still reaches
                # /joint_states, which is where anything reading it wants it.
                "state_interfaces": ["position", "velocity"],
                "state_publish_rate": 80.0,
                "action_monitor_rate": 40.0,
                # The Teensy uses echo-mode feedback (state = cmd, no real sensor).
                # open_loop_control makes the JTC use its own last command as the
                # reference state rather than the echoed measurement.  Without this
                # the JTC can enter a correction loop that fights the servo: it sees
                # state ≈ cmd(t-1) and computes an extra correction that arrives at
                # the Teensy BEFORE the servo has physically moved, causing the servo
                # to overshoot, reverse, and overshoot again — the visible
                # close → open → close symptom.
                #
                # The ST3215 gripper is the exception and MUST run closed loop:
                # its servo reports a real encoder position, so the measurement
                # is the measurement and there is no echo to fight. Leaving
                # open_loop_control on there would throw away the one thing this
                # gripper has that the other does not — the ability to notice
                # that the jaws did not reach the commanded angle, which is what
                # tells an empty close from a grip.
                "open_loop_control": gripper_protocol != "st3215",
                # The vision grasp node owns a bounded feedback/contact watchdog
                # and cancels explicitly. Keep the JTC deadline outside that
                # window so rigid contact cannot abort first.
                "constraints": {
                    "stopped_velocity_tolerance": 0.01,
                    "goal_time": 30.0,
                    "gripper_gear_left_joint": {
                        # trajectory (path) tolerance MUST stay 0.0 = disabled on
                        # this backend, because this joint has no encoder and so
                        # the "error" it measures is never a physical quantity.
                        #
                        # Two ways it fires spuriously:
                        #  1. Teensy present. read() sets state_pos_ = servo_pos_
                        #     and write() sets servo_pos_ = cmd_pos_ afterwards in
                        #     the same cycle, so measured trails desired by
                        #     exactly one control cycle by construction. At
                        #     update_rate 80 Hz that is 12.5 ms, and
                        #     joint_limits.yaml lets MoveIt plan this joint at
                        #     10 rad/s, so the structural lag alone reads as up
                        #     to 10 * 0.0125 = 0.125 rad, well past 0.05.
                        #  2. Teensy absent. read() only assigns state_pos_ when
                        #     state_received_ is true, so with no agent session
                        #     the reported position stays pinned at the
                        #     on_activate() default (min_pos, -1.57) while the
                        #     command moves. The error is then just "how far the
                        #     command has left fully-open" and grows without
                        #     bound - observed 0.053 rad at the first abort and
                        #     0.846 rad later in the same run.
                        #
                        # Case 2 is a disconnected gripper, NOT a tolerance
                        # problem; it must be diagnosed from the TeensyGripperSystem
                        # "Teensy not connected" warning and micro_ros_agent, not
                        # from a JTC abort. Contact and stall belong to the grasp
                        # node's watchdog.
                        "trajectory": 0.0,
                        "goal": 0.01,
                    },
                },
            }
        }

    tmp = tempfile.NamedTemporaryFile(mode="w", prefix="aries_ros2_control_", suffix=".yaml", delete=False)
    yaml.safe_dump(data, tmp, sort_keys=False)
    tmp.flush()
    tmp.close()
    return tmp.name


def build_moveit_controller_config(include_gripper: bool):
    controllers = {
        # NOTE: trajectory_execution settings must be top-level move_group params,
        # NOT inside this dict.  See moveit_args in launch_setup() below.
        "controller_names": ["rebel_arm_trajectory_controller"],
        "rebel_arm_trajectory_controller": {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": True,
            "joints": ARM_JOINTS,
        },
    }

    if include_gripper:
        controllers["controller_names"].append("rebel_gripper_controller")
        controllers["rebel_gripper_controller"] = {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": False,
            "joints": GRIPPER_JOINTS,
        }

    return controllers


def build_ros_log_arguments(log_levels):
    if not log_levels:
        return []

    arguments = ["--ros-args"]
    for logger_name, level in log_levels:
        arguments.extend(["--log-level", f"{logger_name}:={level}"])
    return arguments


def launch_setup(context, *args, **kwargs):
    use_sim_time = False
    hardware_protocol = LaunchConfiguration("hardware_protocol").perform(context)
    arm_hardware_protocol = LaunchConfiguration("arm_hardware_protocol").perform(context)
    gripper_hardware_protocol = LaunchConfiguration("gripper_hardware_protocol").perform(context)
    gripper_type = LaunchConfiguration("gripper_type").perform(context)
    finger_type = LaunchConfiguration("finger_type").perform(context)
    serial_port = LaunchConfiguration("serial_port").perform(context)
    use_gui = LaunchConfiguration("use_gui").perform(context)
    suppress_rebel_logs = LaunchConfiguration("suppress_rebel_logs").perform(context).lower() in ("1", "true", "yes", "on")
    suppress_moveit_execution_logs = LaunchConfiguration("suppress_moveit_execution_logs").perform(context).lower() in ("1", "true", "yes", "on")

    if arm_hardware_protocol == "auto":
        try:
            arm_endpoint = (device("arm.host"), int(device("arm.port")))
            with socket.create_connection(arm_endpoint, timeout=0.25):
                arm_hardware_protocol = "rebel"
        except OSError:
            arm_hardware_protocol = "mock_hardware"

    # WHICH DEVICE TO LOOK FOR FOLLOWS gripper_type, NOT THE PROTOCOL NAME.
    # The two grippers reach their servos over different wires: v2 through the
    # Teensy on its by-id path, st3215 through the USB bus-servo adapter. Probing
    # for the wrong one resolves to mock_hardware while the fitted gripper is
    # sitting there working, which reads in the log as a dead gripper.
    servo_bus_port = LaunchConfiguration("servo_bus_port").perform(context)
    servo_bus_baud = LaunchConfiguration("servo_bus_baud").perform(context)
    servo_id = LaunchConfiguration("servo_id").perform(context)
    gripper_closed_steps = LaunchConfiguration("gripper_closed_steps").perform(context)
    gripper_servo_invert = LaunchConfiguration("gripper_servo_invert").perform(context)
    # Everything else about the stroke is DERIVED from those two plus the open
    # stop, in aries_common.gripper_cal, so the joint limit, the SRDF `open`
    # state, the teleop open position and the component's min_pos cannot drift
    # apart. Recomputed here rather than read, so a gripper_closed_steps passed
    # on the command line moves them all with it.
    from aries_common.gripper_cal import gripper_cal, PITCH_RADIUS_M, STEPS_PER_RAD, Q_CLOSED
    _cal = dict(gripper_cal())
    _cal["closed_steps"] = int(gripper_closed_steps)
    _stroke = abs(_cal["closed_steps"] - int(_cal["open_stop_steps"]))
    gripper_command_open = Q_CLOSED - (_stroke - int(_cal["margin_steps"])) / STEPS_PER_RAD
    gripper_open_travel = -(Q_CLOSED - (_stroke + int(_cal["limit_slack_steps"])) / STEPS_PER_RAD) * PITCH_RADIUS_M

    # THE TEENSY IS RESOLVED WHATEVER GRIPPER IS FITTED.
    #
    # That board is not "the gripper board" - it runs the DRILL, the STACK LIGHT
    # and the LOAD CELLS as well, over one micro-ROS link with one agent. The
    # agent used to be started only when the gripper itself was on the Teensy,
    # so selecting the ST3215 gripper silently took the drill, the stack light
    # and the load cells down with it: no agent, no session, and nothing in the
    # log connecting the two.
    detect_timeout = float(LaunchConfiguration("gripper_detect_timeout").perform(context))
    teensy_port, teensy_note = resolve_gripper_serial(serial_port, detect_timeout)
    if teensy_port:
        serial_port = teensy_port

    # THE SECOND BOARD. Sensors only; see firmware/teensy_science_sys. Resolved
    # here so its absence is reported in the same block as everything else's
    # rather than as a process that quietly never starts.
    use_science = LaunchConfiguration("use_science").perform(context).lower() == "true"
    science_port, science_note = (None, "")
    if use_science:
        science_port, science_note = resolve_science_serial(
            LaunchConfiguration("science_serial_port").perform(context))

    if gripper_type == "st3215":
        found_port, serial_note = resolve_servo_bus(servo_bus_port)
        if gripper_hardware_protocol in ("auto", "rebel"):
            gripper_hardware_protocol = "st3215" if found_port else "mock_hardware"
        live_protocol = "st3215"
        device_note = f"servo_bus={servo_bus_port} id={servo_id}"
    else:
        serial_note = teensy_note
        if gripper_hardware_protocol == "auto":
            gripper_hardware_protocol = "rebel" if teensy_port else "mock_hardware"
        live_protocol = "rebel"
        device_note = f"serial_port={serial_port}"

    # Always say which backend won. Silent fallback to mock is indistinguishable
    # from a dead gripper from the outside.
    gripper_detect_note = LogInfo(
        msg=f"[gripper auto] gripper_type={gripper_type} {device_note} "
            f"resolved={gripper_hardware_protocol}"
        + serial_note
        + ("" if gripper_hardware_protocol == live_protocol
           else "  -- SIMULATED gripper: no command will reach the servo")
    )
    teensy_note_log = LogInfo(
        msg=f"[teensy] {serial_port if teensy_port else 'NOT FOUND'}"
            + (f" -- micro-ROS agent starting; drill, stack light and load cells "
               f"go over this link{teensy_note}" if teensy_port
               else "  -- no agent: the DRILL, STACK LIGHT and LOAD CELLS will not "
                    "respond. This is independent of which gripper is fitted.")
    )

    # The science board gets its own line, and gets one whether or not it was
    # found. A second board that silently is not there looks identical to a
    # second board nobody has configured, and both look identical to a launch
    # that does not know about it at all.
    science_note_log = LogInfo(
        msg=f"[science] {science_port if science_port else 'NOT STARTED'}"
            + (f" -- second micro-ROS agent starting; /science/telemetry goes "
               f"over this link{science_note}" if science_port
               else f"  -- no science telemetry.{science_note}"
                    if use_science
                    else "  -- disabled with use_science:=false")
    )

    urdf_file = PathJoinSubstitution([FindPackageShare("aries"), "urdf", "my_robot.urdf.xacro"])
    robot_description_raw = Command(
        [
            FindExecutable(name="xacro"),
            " ", urdf_file,
            " hardware_protocol:=", hardware_protocol,
            " arm_hardware_protocol:=", arm_hardware_protocol,
            " gripper_hardware_protocol:=", gripper_hardware_protocol,
            " gripper_type:=", gripper_type,
            " finger_type:=", finger_type,
            " serial_port:=", serial_port,
            " servo_bus_port:=", servo_bus_port,
            " servo_bus_baud:=", servo_bus_baud,
            " servo_id:=", servo_id,
            " gripper_closed_steps:=", gripper_closed_steps,
            " gripper_servo_invert:=", gripper_servo_invert,
            " gripper_command_open:=", f"{gripper_command_open:.4f}",
            " gripper_open_travel:=", f"{gripper_open_travel:.6f}",
        ]
    ).perform(context)
    robot_description = ParameterValue(robot_description_raw, value_type=str)

    srdf_file = os.path.join(get_package_share_directory("aries_moveit"), "config", "aries.srdf")
    # aries.srdf is XACRO, not plain text: it carries both grippers' link sets
    # and gates them on gripper_type. `cat` here yields a semantic model with
    # the other gripper's links in it, which srdfdom reports as Errors, and
    # with the xacro tags left in as unknown elements.
    robot_description_semantic = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", srdf_file,
                 " gripper_type:=", gripper_type,
                 " gripper_command_open:=", f"{gripper_command_open:.4f}"]),
        value_type=str)

    kinematics_file = PathJoinSubstitution([FindPackageShare("aries_moveit"), "config", "kinematics.yaml"])
    joint_limits_file = PathJoinSubstitution([FindPackageShare("aries_moveit"), "config", "joint_limits.yaml"])
    ompl_file = PathJoinSubstitution([FindPackageShare("aries_moveit"), "config", "ompl_planning.yaml"])

    kinematics_config = load_yaml(Path(kinematics_file.perform(context)))
    joint_limits_config = load_yaml(Path(joint_limits_file.perform(context)))
    ompl_config = load_yaml(Path(ompl_file.perform(context)))

    if "move_group" in ompl_config:
        ompl_config.update(ompl_config.pop("move_group"))

    ompl_planning_yaml = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_config,
    }

    ros2_control_yaml = build_ros2_control_yaml(arm_hardware_protocol, gripper_hardware_protocol)
    controllers_dict = build_moveit_controller_config(
        gripper_hardware_protocol in GRIPPER_PROTOCOLS
    )
    ros2_control_log_levels = []
    if suppress_rebel_logs and arm_hardware_protocol == "rebel":
        ros2_control_log_levels.append(("igus_rebel", "fatal"))
    if suppress_moveit_execution_logs:
        ros2_control_log_levels.extend([
            ("rebel_arm_trajectory_controller", "warn"),
            ("rebel_gripper_controller", "warn"),
        ])
    ros2_control_arguments = build_ros_log_arguments(ros2_control_log_levels)

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        arguments=ros2_control_arguments,
        parameters=[
            {"robot_description": robot_description_raw},
            ros2_control_yaml,
            {"use_sim_time": use_sim_time},
        ],
        output="both",
    )

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
        ],
        output="both",
    )

    wheel_joint_publisher_node = Node(
        condition=IfCondition(
            LaunchConfiguration("use_wheel_joint_publisher")
        ),
        package="aries_moveit",
        executable="publish_wheel_joints.py",
        name="wheel_joint_publisher",
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--param-file", ros2_control_yaml,
            "--switch-timeout", "30",
        ],
        output="both",
    )

    arm_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "rebel_arm_trajectory_controller",
            "--controller-manager", "/controller_manager",
            "--param-file", ros2_control_yaml,
            "--switch-timeout", "30",
        ],
        output="both",
    )

    hand_guiding_node = None
    if arm_hardware_protocol == "rebel":
        hand_guiding_node = Node(
            package="igus_rebel",
            executable="rebel_hand_guiding.py",
            name="rebel_hand_guiding",
            output="screen",
        )

    # STARTED FOR THE BOARD, NOT FOR THE GRIPPER. See the note above: the drill,
    # the stack light and the load cells all live on this link too, so the agent
    # runs whenever the Teensy is present regardless of which gripper is fitted.
    micro_ros_agent = None
    if teensy_port:
        # GENERATED, not a checked-in file, and it carries this machine's
        # interface pin as well as the low-latency QoS. Fast DDS reads ONE
        # profiles file per process and the variables below outrank the
        # stack-wide ones the launch already exported, so this file is the
        # agent's entire DDS configuration -- if it does not pin the agent
        # where the rest of the stack is pinned, the agent cannot discover
        # ros2_control_node and no command ever reaches the servo. The
        # hand-written config/fastdds_low_latency.xml this replaced did
        # exactly that: it pinned the agent to 127.0.0.1 while everything
        # else ran on a transport whitelisted to the field-link address.
        # See write_agent_dds_config() for the full account.
        _fastdds_xml, _ = write_agent_dds_config(require_link=False)
        micro_ros_agent = ExecuteProcess(
            cmd=[
                "ros2", "run", "micro_ros_agent", "micro_ros_agent",
                # 115200, NOT 6000000. Linux speed_t values are encodings, not
                # literal bit rates: the largest valid one is B4000000 == 4111.
                # 6000000 is not in the agent's baud table
                # (xrceagent/.../baud_rate_table_linux.h), so it falls through to
                # a raw (speed_t)atoi() cast, and cfsetispeed/cfsetospeed then
                # reject it with EINVAL and leave c_ispeed/c_ospeed at 0. The
                # agent does not check either return value, so it proceeds to
                # tcsetattr with the line speed unset -- which is why the agent
                # came up most of the time and intermittently did not.
                # The Teensy is USB CDC (Tools > USB Type > Serial), where baud
                # is ignored by the device entirely, so nothing is lost here.
                "serial", "--dev", serial_port, "-b", "115200",
            ],
            additional_env={
                # Synchronous publish mode + zero latency budget, on the same
                # pinned UDPv4 transport as every other node. BOTH spellings:
                # FASTRTPS_ is the one this build actually reads, and setting
                # only it would leave the stack-wide FASTDDS_ value inherited
                # from the launch environment pointing at a different file --
                # which is confusing to debug even when it is not the one that
                # wins.
                "FASTRTPS_DEFAULT_PROFILES_FILE": str(_fastdds_xml),
                "FASTDDS_DEFAULT_PROFILES_FILE": str(_fastdds_xml),
            },
            output="screen",
            # The agent is the ONLY path between /gripper/cmd and the servo, and
            # it does die on its own: observed exit code 254 roughly 6 minutes
            # into a session, with /dev/ttyACM0 never re-enumerating (so not a
            # USB drop).
            #
            # BUT DO NOT READ 254 ALONE AS THAT FAULT. `ros2 run` exits 254 on
            # a normal Ctrl-C, where a native node reports -2, so every clean
            # shutdown of this launch also ends with 254 in the log. To tell
            # the two apart, check whether "user interrupted with ctrl-c" or
            # any other process death shares the timestamp. Chasing a 254 that
            # was only the operator stopping the stack costs an afternoon. Without respawn nothing restarts it, the Teensy falls
            # back to WAITING_AGENT and pings a closed port forever, and the
            # gripper stops responding mid-run with no error anywhere: the
            # hardware plugin keeps state_received_ latched true, so read()
            # still echoes servo_pos_ and the JTC still sees perfect tracking
            # while no command reaches the servo. The firmware already handles
            # its side of the reconnect, so restarting the agent is enough to
            # re-establish the session.
            respawn=True,
            respawn_delay=2.0,
        )

    # THE SECOND MICRO-ROS AGENT, for the science board.
    #
    # A SEPARATE PROCESS ON A SEPARATE PORT, not a second client on the drill's
    # agent -- an XRCE agent owns exactly one serial device, so two boards mean
    # two agents. They share the DDS profile below, which is correct: both need
    # pinning to the same interface as the rest of the stack, and the file is
    # per-machine rather than per-board.
    #
    # NO respawn=True, UNLIKE THE DRILL'S. That agent is the only path between
    # /gripper/cmd and a servo, so it is worth restarting aggressively; this one
    # carries sensors that are read on demand. A science agent that dies costs
    # telemetry until the next launch, which is a thing an operator can see on
    # /science/status -- whereas a respawn loop against a board that is not
    # there is noise in the log of a rover that has real problems to report.
    science_agent = None
    if science_port:
        _sci_fastdds_xml, _ = write_agent_dds_config(require_link=False)
        science_agent = ExecuteProcess(
            cmd=[
                "ros2", "run", "micro_ros_agent", "micro_ros_agent",
                # 115200 for the same reason as the drill board: the largest
                # valid speed_t is B4000000, anything above it is rejected by
                # cfsetospeed with EINVAL which the agent does not check, and
                # this is USB CDC where the device ignores baud anyway.
                "serial", "--dev", science_port, "-b", "115200",
            ],
            additional_env={
                "FASTRTPS_DEFAULT_PROFILES_FILE": str(_sci_fastdds_xml),
                "FASTDDS_DEFAULT_PROFILES_FILE": str(_sci_fastdds_xml),
            },
            output="screen",
        )

    gripper_controller_spawner = None
    if gripper_hardware_protocol in GRIPPER_PROTOCOLS:
        gripper_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "rebel_gripper_controller",
                "--controller-manager", "/controller_manager",
                "--param-file", ros2_control_yaml,
                "--switch-timeout", "30",
            ],
            output="both",
        )

    arm_spawner_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_trajectory_controller_spawner],
        )
    )

    gripper_spawner_event = None
    if gripper_controller_spawner is not None:
        gripper_spawner_event = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=arm_trajectory_controller_spawner,
                on_exit=[gripper_controller_spawner],
            )
        )

    moveit_args = {
        "robot_description": robot_description_raw,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics_config,
        "robot_description_planning": joint_limits_config,
        "moveit_simple_controller_manager": controllers_dict,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
        # trajectory_execution MUST be at move_group top-level (dot-notation keys).
        # Putting them inside moveit_simple_controller_manager (above) puts them in
        # the wrong namespace and MoveIt ignores them — causing execution_duration
        # monitoring to default ON (aborts mid-stroke) and start_tolerance to
        # default 0 (executes wrong-start trajectories that snap the servo open).
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.03,
        "trajectory_execution.execution_duration_monitoring": False,
        # Use full joint velocity/acceleration limits by default (MoveIt default is 0.1).
        # The gripper only travels 89 mm; at 10% speed it takes ~1 s to close.
        # At 100% it closes in ~0.2 s (limited by physical servo speed ~0.45 m/s).
        "default_velocity_scaling_factor": 1.0,
        "default_acceleration_scaling_factor": 1.0,
        **ompl_planning_yaml,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        arguments=build_ros_log_arguments([
            ("move_group", "warn"),
            ("move_group.moveit.moveit.ros.planning_pipeline", "error"),
            ("moveit.simple_controller_manager.follow_joint_trajectory_controller_handle", "warn"),
        ] if suppress_moveit_execution_logs else []),
        parameters=[{"use_sim_time": use_sim_time}, moveit_args],
        output="screen",
    )

    use_joystick = LaunchConfiguration("use_joystick")
    use_joy_node = LaunchConfiguration("use_joy_node")
    joy_driver = LaunchConfiguration("joy_driver")
    joy_layout = LaunchConfiguration("joy_layout")
    joy_dev = LaunchConfiguration("joy_dev")
    joystick_control_mode = LaunchConfiguration("joystick_control_mode")
    cartesian_frame = LaunchConfiguration("cartesian_frame")

    # Where the pad is READ, separately from whether teleop is enabled.
    #
    # These are two different questions and used to be one flag. With the
    # operator 150 m away the pad is plugged into the base station, so the
    # driver has to run there while every consumer below keeps running here --
    # use_joystick:=true use_joy_node:=false. Turning use_joystick off to move
    # the driver would have taken the arm teleop, the presets and the rover
    # drive with it.
    #
    # Exactly one machine may set this. Two joy_nodes put two publishers on
    # /joy, and the consumers would see the two pads' states interleaved at
    # double rate: buttons appear to chatter and nothing is reproducible.
    joy_driver_condition = IfCondition(PythonExpression([
        "'", use_joystick, "' == 'true' and '", use_joy_node, "' == 'true'"
    ]))
    servo_joystick_condition = IfCondition(PythonExpression([
        "'", use_joystick, "' == 'true' and '", joystick_control_mode, "' == 'servo'"
    ]))
    move_group_joystick_condition = IfCondition(PythonExpression([
        "'", use_joystick, "' == 'true' and '", joystick_control_mode, "' == 'move_group'"
    ]))
    servo_params_file = os.path.join(get_package_share_directory("aries_moveit"), "config", "servo.yaml")
    servo_context = load_yaml(Path(servo_params_file))
    servo_params = {"moveit_servo": servo_context}
    planning_group_name = {"planning_group_name": "igus_rebel_arm"}

    servo_node = Node(
        condition=servo_joystick_condition,
        package="moveit_servo",
        executable="servo_node",
        parameters=[
            {"use_sim_time": use_sim_time},
            servo_params,
            planning_group_name,
            moveit_args,
        ],
        output="screen",
    )

    servo_collision_guard_node = Node(
        condition=servo_joystick_condition,
        package="aries_moveit",
        executable="servo_collision_guard",
        name="servo_collision_guard",
        parameters=[
            {"use_sim_time": use_sim_time},
            moveit_args,
            {
                "input_topic": "servo_guard/input_joint_trajectory",
                "output_topic": "rebel_arm_trajectory_controller/joint_trajectory",
                "joint_state_topic": "joint_states",
                "status_topic": "/arm_joystick/status",
                "group_name": "arm_with_gripper",
                "min_self_distance": 0.015,
                "distance_tolerance": 0.001,
                "interpolation_steps": 1,
                "hold_time": 0.02,
            },
        ],
        output="screen",
    )

    joy_node = Node(
        condition=joy_driver_condition,
        package="joy",
        executable=joy_driver,
        name="joy_node",
        parameters=[{
            "dev": joy_dev,
            "autorepeat_rate": 80.0,
            "deadzone": 0.0,
            "coalesce_interval_ms": 1,
        }],
        remappings=[("joy", "joy/raw")],
        output="screen",
    )

    joy_layout_normalizer_node = Node(
        condition=joy_driver_condition,
        package="aries_moveit",
        executable="joy_layout_normalizer.py",
        name="joy_layout_normalizer",
        parameters=[{
            "input_topic": "joy/raw",
            "output_topic": "joy",
            "layout": joy_layout,
            "device": joy_dev,
        }],
        output="screen",
    )

    gamepad_file = os.path.join(get_package_share_directory("aries_moveit"), "config", "gamepad.yaml")
    # Speeds live in teleop_speeds.yaml and are loaded last so they win over the
    # copies still in gamepad.yaml. gamepad.yaml keeps the button/axis mapping.
    teleop_speeds_file = os.path.join(
        get_package_share_directory("aries_moveit"), "config", "teleop_speeds.yaml"
    )
    # Per-gripper overlay, loaded last so its keys win. Only the ST3215 gripper
    # has one, and it exists only because that mechanism's open position is
    # -4.065 where teleop_speeds.yaml's shared value is v2's -1.57.
    # One gripper now, so the overlay is unconditional. It stays a separate file
    # rather than being folded into teleop_speeds.yaml because that file is
    # shared with the arm and the rover, and these keys are the gripper's.
    teleop_gripper_files = []
    if True:
        teleop_gripper_files.append(os.path.join(
            get_package_share_directory("aries_moveit"), "config",
            "teleop_speeds_st3215.yaml"))
        # The overlay file carries the speeds; the open POSITION is derived, so
        # it is appended after the file and wins over it.
        teleop_gripper_files.append({"gripper_open_position": gripper_command_open})
    gamepad_node = Node(
        condition=servo_joystick_condition,
        package="aries_moveit",
        executable="rebel_servo_teleop_gamepad",
        name="rebel_servo_teleop_gamepad",
        parameters=[gamepad_file, teleop_speeds_file, *teleop_gripper_files,
                    # Last in the list so it beats the cartesian_frame
                    # gamepad.yaml sets, which is only the default.
                    {"cartesian_frame": cartesian_frame}],
        output="screen",
    )

    move_group_joystick_node = Node(
        condition=move_group_joystick_condition,
        package="aries_moveit",
        executable="rebel_movegroup_joystick.py",
        name="rebel_movegroup_joystick",
        parameters=[gamepad_file, teleop_speeds_file, *teleop_gripper_files,
                    {"cartesian_frame": cartesian_frame}],
        output="screen",
    )

    # LT + Y -> pick_home, LT + A -> probe_drop, LT + B -> soil_drop. MoveIt
    # collision-plans each preset; the idle arm controller executes it directly.
    # Refusing while RB/RT/LB are held prevents competition with manual teleop.
    arm_preset_pose_node = Node(
        condition=IfCondition(use_joystick),
        package="aries_moveit",
        executable="arm_preset_pose_joystick.py",
        name="arm_preset_pose_joystick",
        parameters=[gamepad_file, teleop_speeds_file, *teleop_gripper_files],
        output="screen",
    )

    # Gripper arc overlay for RViz (aries_moveit/scripts/gripper_arc_visualizer.py)
    # is not launched. Its geometry tables model the 85.563 mm four-bar of the
    # retired gripper_new only; on v2 (50 mm parallelogram, 83 mm stroke) the
    # sweep it draws is wrong by 100 mm. It used to be gated on
    # gripper_type == "new", which no longer exists. Re-fit the tables to v2
    # before adding the node back.
    #
    # That is a v2 problem only. On gripper_type:=st3215 the overlay needs no
    # tables at all - the jaws translate, so it draws the exact closed form -
    # and it IS launched on that gripper from move_group.launch.py. Adding it
    # back here would be safe for st3215 and still wrong for v2, so it stays
    # out until the v2 tables are re-fitted.

    # Live servo telemetry as an RViz text overlay above the gripper: jaw gap,
    # joint, TCP, current, voltage, temperature, and whether the servo is near
    # or past one of its own protection limits. Unlike the arc overlay above it
    # carries no geometry of its own - it renders whatever the hardware
    # component publishes on /diagnostics - so it is right on every gripper and
    # says NO TELEMETRY on the ones that publish nothing.
    #
    # Gated on use_gui because it exists only to be looked at; /diagnostics is
    # published by the hardware component either way.
    gripper_status_overlay_node = Node(
        condition=IfCondition(use_gui),
        package="aries_moveit",
        executable="gripper_status_overlay.py",
        name="gripper_status_overlay",
        output="screen",
    )

    rviz_config = os.path.join(get_package_share_directory("aries_moveit"), "launch", "moveit.rviz")
    rviz_node = Node(
        condition=IfCondition(use_gui),
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        parameters=[
            {
                "robot_description": robot_description_raw,
                "robot_description_semantic": robot_description_semantic,
                "robot_description_kinematics": kinematics_config,
                "robot_description_planning": joint_limits_config,
            }
        ],
        output={"both": "log"},
    )

    nodes = [
        gripper_detect_note,
        teensy_note_log,
        science_note_log,
        ros2_control_node,
        robot_state_pub,
        wheel_joint_publisher_node,
        joint_state_broadcaster_spawner,
        arm_spawner_event,
        move_group_node,
        servo_node,
        servo_collision_guard_node,
        joy_node,
        joy_layout_normalizer_node,
        gamepad_node,
        move_group_joystick_node,
        arm_preset_pose_node,
        gripper_status_overlay_node,
        rviz_node,
    ]
    if gripper_spawner_event:
        nodes.append(gripper_spawner_event)
    if hand_guiding_node:
        nodes.append(hand_guiding_node)
    if micro_ros_agent:
        nodes.append(micro_ros_agent)
    if science_agent:
        nodes.append(science_agent)
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_gui", default_value="true", description="Launch RViz with MoveIt interface"),
            DeclareLaunchArgument("gripper_type", default_value="st3215", choices=["st3215"], description="Which gripper is bolted to the flange. Only the ST3215 rack-and-pinion exists; v2 and the older four-bars are retired to aries/urdf/legacy/. Accepted-and-narrowed because a dozen launch files pass it down."),
            DeclareLaunchArgument("finger_type", default_value="bucket", choices=["bucket", "maintenance"], description="Swappable fingertip mesh; must match the pair bolted to the racks"),
            DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Hardware protocol for arm backend"),
            DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Global hardware protocol passed to xacro (arm+gripper)"),
            DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "st3215", "mock_hardware", "gazebo"], description="Hardware protocol for gripper backend. 'auto' resolves to st3215 when the bus-servo adapter is present, else mock_hardware. 'rebel' is gone with the Teensy gripper - the Teensy itself still runs the drill, stack light and load cells."),
            DeclareLaunchArgument("use_joystick", default_value="false", description="Start joystick arm teleop"),
            DeclareLaunchArgument(
                "use_joy_node",
                default_value="true",
                description=(
                    "Read the pad on THIS machine. false leaves the teleop "
                    "consumers running here and expects /joy from elsewhere -- "
                    "the base station, over the link. Exactly one machine may "
                    "set it true, or two publishers interleave on /joy."
                ),
            ),
            DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"], description="Joystick driver executable from the joy package"),
            DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"], description="Normalize joystick layout before teleop nodes consume /joy"),
            DeclareLaunchArgument("joy_dev", default_value=device_str("joystick.device"), description="Joystick device used by joy_node and the layout normalizer"),
            DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"], description="servo uses smooth Cartesian MoveIt Servo teleop with collision guard; move_group uses planned steps"),
            DeclareLaunchArgument("cartesian_frame", default_value="tool", choices=["tool", "base"], description="Frame the joystick Cartesian jog is read in: tool = the gripper's own axes (the stick pushes the gripper the way it is pointing), base = the rover's. Also picks the stick axis mapping, which differs between the two."),
            DeclareLaunchArgument("serial_port", default_value=device_str("gripper.serial_port"), description="USB-serial port for the Teensy gripper controller"),
            DeclareLaunchArgument("use_science", default_value="true", description="Start the micro-ROS agent for the SECOND Teensy, the science board. Harmless when no board is fitted: the agent simply is not started and a note says why."),
            DeclareLaunchArgument("science_serial_port", default_value=device_str("science.serial_port"), description="USB-serial port for the Teensy science board. Empty until somebody fills in science.serial_port in devices.yaml."),
            DeclareLaunchArgument("servo_bus_port", default_value=device_str("servo_bus.port"), description="Serial port of the ST3215 bus-servo adapter (gripper_type:=st3215 only)"),
            DeclareLaunchArgument("servo_bus_baud", default_value=device_str("servo_bus.baud"), description="Baud rate for the ST3215 bus-servo adapter"),
            DeclareLaunchArgument("servo_id", default_value=device_str("servo_bus.gripper_servo_id"), description="Bus ID of the ST3215 driving the secondary gripper"),
            DeclareLaunchArgument("gripper_closed_steps", default_value=cal_str("closed_steps"), description="BENCH CALIBRATION: the step the SERVO parks at with the jaws closed - NOT the hand-forced stop at 572, which the servo cannot reach and which leaves it holding a permanent preload. See st3215_gripper_hardware's header."),
            DeclareLaunchArgument("gripper_servo_invert", default_value=cal_str("invert"), choices=["true", "false"], description="BENCH CALIBRATION: true if a rising ST3215 step count OPENS the jaws. It does on this gripper (575 closed -> 1720 open), confirmed by labelling both poses by hand with torque off."),
            DeclareLaunchArgument("gripper_detect_timeout", default_value="8.0", description="Seconds to wait for the Teensy serial device before falling back to mock_hardware. Covers USB re-enumeration after a board reset."),
            DeclareLaunchArgument("suppress_rebel_logs", default_value="false", description="Suppress chatty igus_rebel logger output from ros2_control_node"),
            DeclareLaunchArgument("suppress_moveit_execution_logs", default_value="false", description="Suppress routine MoveIt execution chatter from move_group and ros2_control_node"),
            DeclareLaunchArgument(
                "use_wheel_joint_publisher",
                default_value="true",
                description=(
                    "Publish zero-valued rover wheel joints when no real "
                    "encoder-backed publisher is running"
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
