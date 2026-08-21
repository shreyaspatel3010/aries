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

ENABLE_BUTTON = 4          # LB
LINEAR_AXIS = 1            # left stick vertical
JOY_RATE_HZ = 80.0         # what joy_node autorepeats at
CMD_VEL_TOPIC = "/cmd_vel/teleop"

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
    def __init__(self):
        super().__init__("aries_control_path_check")
        self.joy = self.create_publisher(Joy, "/joy", 10)
        self.seen = []  # (receipt time, linear.x)
        self.create_subscription(
            Twist, CMD_VEL_TOPIC,
            lambda m: self.seen.append((time.time(), m.linear.x)), 10)

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
    args = ap.parse_args()

    proc = None
    if not args.remote:
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
    h = Harness()
    failures = []
    try:
        if not h.wait_for_consumer():
            print(f"  {RED}✗{RST} nothing is subscribed to /joy")
            if args.remote:
                print("      The rover is not running, or the two machines are "
                      "not on the same domain.")
                print("      source \"$(ros2 pkg prefix aries_common)"
                      "/share/aries_common/aries_dds_env.sh\"")
            return 1
        print(f"  {GREEN}✓{RST} a node is consuming /joy")
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
