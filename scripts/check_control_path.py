#!/usr/bin/env python3
"""Prove the joystick actually reaches the wheels, and that it lets go.

    ./scripts/check_control_path.py              local, spawns its own node
    ./scripts/check_control_path.py --remote     across the link, rover running

This is the path that crosses the antenna: /joy is published on the base
station, rover_cmd_vel_joystick consumes it on the rover, and its /cmd_vel goes
to the ODrive bridge. Both modes exercise real DDS between real processes, so
they cover QoS matching and discovery -- not just Python function calls.

LOCAL (default)
    Spawns rover_cmd_vel_joystick here and commands motion at it. Nothing is
    connected to the ODrives, so this is safe on any machine, including a
    laptop with no rover attached. It checks both halves:

      * a held stick produces motion;
      * /joy going silent stops it inside joy_timeout_sec. That is the guard
        between a lost radio link and a rover that keeps driving on its last
        command, and it is the half that is easy to break and never notice.

REMOTE (--remote)
    Run on the BASE STATION with the rover up. It does NOT command motion:
    the enable button is left released and the sticks centred, so the wheels
    never turn. What it proves is that /joy crosses the link, the rover's node
    is alive and consuming it, and its /cmd_vel comes back -- with the observed
    round-trip rate.

    It cannot test the watchdog, because doing that means driving the rover.
    Test that by hand, once per event, with the rover on stands or with room
    ahead of it: hold LB, drive, then pull power on the base radio.

ARM (--arm)
    The wheels and the arm cross the link the same way but land in different
    places, so "the rover drives but the arm does nothing" is a real state and
    the two checks above say nothing about it. /joy is consumed on the rover by
    rebel_servo_teleop_gamepad (RB Cartesian, RT joint jog) and
    arm_preset_pose_joystick (LT + Y/A/B); the first feeds servo_node, which
    feeds servo_collision_guard, which feeds the arm controller.

    This mode PUBLISHES NOTHING. It only observes, so it is safe to run while
    the operator has the pad in hand -- and it must be, because a second
    publisher on /joy is exactly the fault the base-station contract exists to
    prevent. It reports which link of that chain is missing, and it prints what
    rebel_servo_teleop_gamepad says about itself on /arm_joystick/status, which
    is usually the answer on its own: that node reports its own refusals there
    ("MoveIt model unavailable: robot_description not loaded" and friends).

    Run it on the BASE STATION with the rover up. It works locally too, against
    a full_hardware stack on one machine.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String

ENABLE_BUTTON = 4          # LB
LINEAR_AXIS = 1            # left stick vertical
JOY_RATE_HZ = 80.0         # what joy_node autorepeats at
CMD_VEL_TOPIC = "/cmd_vel/teleop"

# The arm half of the path. Node names as launched by aries_hardware.launch.py.
ARM_STATUS_TOPIC = "/arm_joystick/status"
ARM_TRAJECTORY_TOPIC = "/rebel_arm_trajectory_controller/joint_trajectory"
ARM_NODES = (
    ("rebel_servo_teleop_gamepad",
     "RB/RT arm teleop. Without it the sticks reach nothing.",
     "use_joystick:=false on the rover, or joystick_control_mode:=move_group "
     "(which runs rebel_movegroup_joystick instead)"),
    ("servo_node",
     "MoveIt Servo. The gamepad node's commands go through it.",
     "joystick_control_mode is not 'servo', or move_group failed to start"),
    ("servo_collision_guard",
     "Self-collision gate between Servo and the controller.",
     "same cause as servo_node: it shares that condition"),
    ("arm_preset_pose_joystick",
     "LT + Y/A/B preset moves.",
     "use_joystick:=false on the rover"),
)

GREEN, RED, YELLOW, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _kill_group(proc):
    """Kill the spawned node and everything it spawned."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def _joy(drive):
    msg = Joy()
    msg.buttons = [0] * 8
    msg.axes = [0.0] * 8
    if drive:
        msg.buttons[ENABLE_BUTTON] = 1
        msg.axes[LINEAR_AXIS] = 1.0
    return msg


class Harness(Node):
    def __init__(self, observe_only=False):
        super().__init__("aries_control_path_check")
        # No /joy publisher at all in observe-only mode. Not merely unused: the
        # arm check COUNTS publishers on /joy to catch two joy drivers, and a
        # publisher of our own would make a healthy stack read as the fault. It
        # would also make this tool the second pad it is looking for.
        self.joy = None if observe_only else self.create_publisher(Joy, "/joy", 10)
        self.seen = []  # (receipt time, linear.x)
        self.create_subscription(
            Twist, CMD_VEL_TOPIC,
            lambda m: self.seen.append((time.time(), m.linear.x)), 10)
        self.arm_status = []
        self.create_subscription(
            String, ARM_STATUS_TOPIC,
            lambda m: self.arm_status.append(m.data), 10)

    def pump(self, seconds, drive=False, quiet=False):
        """Publish /joy for `seconds` at the driver's autorepeat rate."""
        end = time.time() + seconds
        while time.time() < end:
            if not quiet:
                self.joy.publish(_joy(drive))
            rclpy.spin_once(self, timeout_sec=1.0 / JOY_RATE_HZ)

    def wait_for_consumer(self, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.count_subscribers("/joy") > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


def run_arm(h, failures):
    """Observe the arm path. Publishes nothing -- see the module docstring."""
    # Discovery is not instant across a radio link, and a node that has not
    # been seen yet is indistinguishable from one that is not running. Spin
    # first, then judge.
    end = time.time() + 6.0
    while time.time() < end:
        rclpy.spin_once(h, timeout_sec=0.1)

    # Two joy drivers is the documented silent failure: the consumers see both
    # pads interleaved at double rate, so a held RB flickers and the arm gate
    # opens and shuts. The wheels survive it better than the arm does, because
    # a dropped enable frame there just costs one control period.
    joy_publishers = h.count_publishers("/joy")
    if joy_publishers > 1:
        print(f"  {RED}✗{RST} {joy_publishers} publishers on /joy")
        print("      Exactly one machine may run the joy driver. The rover "
              "must be launched with use_joy_node:=false")
        print("      (rover_field.launch.py already defaults to that); the "
              "base station runs it.")
        print("      Interleaved pads make a held RB flicker, which the arm "
              "gate reads as release.")
        failures.append(f"{joy_publishers} publishers on /joy, expected 1")
    elif joy_publishers == 1:
        print(f"  {GREEN}✓{RST} exactly one publisher on /joy")
    else:
        print(f"  {RED}✗{RST} nothing is PUBLISHING /joy")
        print("      The rover's nodes are subscribed and waiting, so the pad "
              "itself is not reaching them.")
        print("      On the base station: is the pad plugged in, is joy_dev "
              "right, did joy_node start?")
        print("      Every teleop node treats silent /joy as 'stop', so this "
              "looks exactly like a dead arm.")
        failures.append("no publisher on /joy: the pad is not reaching the rover")

    running = set(h.get_node_names())
    missing = []
    for name, what, why in ARM_NODES:
        if name in running:
            print(f"  {GREEN}✓{RST} {name} is running")
        else:
            missing.append((name, what, why))

    for name, what, why in missing:
        print(f"  {RED}✗{RST} {name} is NOT running")
        print(f"      {what}")
        print(f"      likely: {why}")
    if missing:
        failures.append(
            f"{len(missing)} arm node(s) are not running on the rover")

    # The controller has to be listening, or Servo's output goes nowhere. This
    # is downstream of the joystick entirely: it fails the same way whether the
    # pad is here or on the rover.
    if h.count_subscribers(ARM_TRAJECTORY_TOPIC) > 0:
        print(f"  {GREEN}✓{RST} the arm controller is subscribed to "
              f"{ARM_TRAJECTORY_TOPIC}")
    else:
        print(f"  {RED}✗{RST} nothing is subscribed to {ARM_TRAJECTORY_TOPIC}")
        print("      The arm controller is not loaded or not active. Check "
              "`ros2 control list_controllers` ON THE ROVER --")
        print("      a controller in 'inactive' looks identical to a broken "
              "joystick from here.")
        failures.append("the arm trajectory controller is not listening")

    # What the teleop node says about itself. This is the one that usually
    # names the real cause outright.
    if not h.arm_status:
        print(f"  {YELLOW}~{RST} nothing published on {ARM_STATUS_TOPIC} in 6 s")
        print("      rebel_servo_teleop_gamepad publishes there on every mode "
              "change and on every refusal.")
        print("      Silence is normal for an idle arm; combined with a "
              "missing node above it is the confirmation.")
    else:
        print(f"  {GREEN}✓{RST} {ARM_STATUS_TOPIC} says:")
        # It republishes the same line while nothing changes; show the distinct
        # ones so a real sequence is not buried under one repeated message.
        distinct = list(dict.fromkeys(h.arm_status))
        for line in distinct[-3:]:
            print(f"        {line}")
        bad = [t for t in h.arm_status
               if "unavailable" in t or "failed" in t.lower()]
        if bad:
            failures.append(f"the arm teleop is reporting: {bad[-1]}")


def run_local(h, failures):
    h.seen.clear()
    h.pump(2.0, drive=True)
    moving = [v for _, v in h.seen if v > 0.0]
    if moving:
        print(f"  {GREEN}✓{RST} drive: {len(moving)} non-zero commands, peak "
              f"{max(moving):.3f} m/s")
    else:
        failures.append("a held stick produced no motion command")

    h.seen.clear()
    last_joy = time.time()
    h.pump(2.0, quiet=True)
    stopped_at = next((t - last_joy for t, v in h.seen if v == 0.0), None)
    tail = h.seen[-10:]

    # joy_timeout_sec plus one control period: the watchdog can only act on the
    # next timer tick after it expires.
    budget = 0.35 + 1.0 / 30.0
    if stopped_at is None:
        failures.append("never commanded a stop after /joy went silent")
    elif not (tail and all(v == 0.0 for _, v in tail)):
        failures.append(
            f"resumed motion after stopping: {[round(v, 3) for _, v in tail]}")
    elif stopped_at > budget:
        failures.append(
            f"watchdog took {stopped_at * 1000:.0f} ms, budget "
            f"{budget * 1000:.0f} ms")
    else:
        print(f"  {GREEN}✓{RST} stop:  first zero {stopped_at * 1000:.0f} ms "
              f"after the last /joy (budget {budget * 1000:.0f} ms), then "
              f"stayed stopped")


def run_remote(h, failures):
    print(f"  {YELLOW}~{RST} neutral joystick only — the wheels will not turn")
    h.seen.clear()
    t0 = time.time()
    h.pump(3.0, drive=False)
    elapsed = time.time() - t0
    if not h.seen:
        failures.append(
            f"no {CMD_VEL_TOPIC} came back. Is the rover running "
            f"rover_field.launch.py? Is this shell on the right domain?")
        return
    if any(v != 0.0 for _, v in h.seen):
        failures.append("the rover commanded motion from a neutral stick")
        return
    print(f"  {GREEN}✓{RST} round trip: {len(h.seen)} replies in "
          f"{elapsed:.1f}s ({len(h.seen) / elapsed:.0f} Hz), all zero")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", action="store_true",
                    help="rover is running elsewhere; do not command motion")
    ap.add_argument("--arm", action="store_true",
                    help="check the ARM path instead of the wheels. Observes "
                         "only: publishes nothing, so it is safe to run while "
                         "the operator holds the pad")
    args = ap.parse_args()

    proc = None
    if not args.remote and not args.arm:
        # Its own session, so the whole tree can be killed. `ros2 run` execs a
        # launcher that spawns the node as a GRANDCHILD: terminating the child
        # alone leaves the node running, and an orphaned publisher on
        # /cmd_vel/teleop makes the next --remote run pass against itself
        # instead of against the rover.
        proc = subprocess.Popen(
            ["ros2", "run", "aries_teleop", "rover_cmd_vel_joystick.py",
             "--ros-args", "-p", f"cmd_vel_topic:={CMD_VEL_TOPIC}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)

    rclpy.init()
    h = Harness(observe_only=args.arm)
    failures = []
    try:
        if not h.wait_for_consumer():
            print(f"  {RED}✗{RST} nothing is subscribed to /joy")
            if args.remote or args.arm:
                print("      The rover is not running, or the two machines are "
                      "not on the same domain.")
                print("      source \"$(ros2 pkg prefix aries_common)"
                      "/share/aries_common/aries_dds_env.sh\"")
            return 1
        print(f"  {GREEN}✓{RST} a node is consuming /joy")
        if args.arm:
            # Deliberately not gated on --remote: the same observation is
            # useful against a full_hardware stack on one machine.
            run_arm(h, failures)
        else:
            (run_remote if args.remote else run_local)(h, failures)
    finally:
        h.destroy_node()
        rclpy.shutdown()
        if proc:
            _kill_group(proc)

    for f in failures:
        print(f"  {RED}✗{RST} {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
