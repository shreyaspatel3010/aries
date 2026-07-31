#!/usr/bin/env python3
"""Measure what the arm actually does versus what the joystick node commands.

Two teleop complaints need numbers rather than guesses:

  overshoot   how far each joint keeps travelling after the stick is released
  direction   whether a pure Cartesian axis command produces motion along only
              that axis at the TCP

Both are settled by comparing the commanded joint trajectory against the
measured joint states, so this records:

  /joy                                              what was asked for
  /rebel_arm_trajectory_controller/joint_trajectory what the node commanded
  /joint_states                                     what the arm actually did

Run it, drive the arm the way that misbehaves, then Ctrl-C.

    ros2 run aries_moveit measure_teleop_tracking.py
    # or: python3 scripts/measure_teleop_tracking.py

Nothing is published, so this cannot move the arm.
"""

import math
import sys
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from trajectory_msgs.msg import JointTrajectory

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class TrackingProbe(Node):
    def __init__(self):
        super().__init__("measure_teleop_tracking")

        self.cmd = deque(maxlen=200000)    # (t, [target positions])
        self.meas = deque(maxlen=200000)   # (t, [measured positions])
        self.joy = deque(maxlen=200000)    # (t, active?) stick/dpad past deadzone

        self.deadzone = 0.04
        # Cartesian axes in gamepad.yaml: linear x/y/z = 1/0/7, angular = 3/4/6.
        self.motion_axes = [0, 1, 3, 4, 6, 7]

        self.create_subscription(JointState, "/joint_states", self._on_state, 50)
        self.create_subscription(
            JointTrajectory,
            "/rebel_arm_trajectory_controller/joint_trajectory",
            self._on_cmd,
            50,
        )
        self.create_subscription(Joy, "/joy", self._on_joy, 50)

        self.get_logger().info("recording — drive the arm, then Ctrl-C for the report")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in ARM_JOINTS):
            return
        vel = (
            [msg.velocity[idx[j]] for j in ARM_JOINTS]
            if len(msg.velocity) == len(msg.position)
            else [0.0] * len(ARM_JOINTS)
        )
        self.meas.append(
            (self._now(), [msg.position[idx[j]] for j in ARM_JOINTS], vel)
        )

    def _on_cmd(self, msg):
        if not msg.points:
            return
        idx = {n: i for i, n in enumerate(msg.joint_names)}
        if not all(j in idx for j in ARM_JOINTS):
            return
        last = msg.points[-1]
        self.cmd.append((self._now(), [last.positions[idx[j]] for j in ARM_JOINTS]))

    def _on_joy(self, msg):
        active = any(
            abs(msg.axes[a]) > self.deadzone
            for a in self.motion_axes
            if a < len(msg.axes)
        )
        # Also track the enable gates. A "quiet" window with the arm still
        # moving means something other than the sticks is driving it — RViz,
        # MoveIt, or hand guiding (Y) letting the arm sag under gravity — and
        # that must not be reported as teleop overshoot.
        rb = msg.buttons[5] if len(msg.buttons) > 5 else 0
        rt = msg.axes[5] > 0.5 if len(msg.axes) > 5 else False
        y = msg.buttons[3] if len(msg.buttons) > 3 else 0
        self.joy.append((self._now(), active, bool(rb), bool(rt), bool(y)))

    # ------------------------------------------------------------------ report

    def report(self):
        if len(self.meas) < 10 or len(self.cmd) < 10:
            print("\nnot enough data — was the launch running and the arm moved?")
            print(f"  joint_states={len(self.meas)} commands={len(self.cmd)} joy={len(self.joy)}")
            return

        print(f"\n{'='*66}\nsamples: {len(self.meas)} states, {len(self.cmd)} commands, "
              f"{len(self.joy)} joy\n{'='*66}")

        meas = list(self.meas)

        # --- true tracking lag ----------------------------------------------
        # Comparing a command against the measurement taken at the SAME instant
        # only recovers the lead the node designed in (command is built as
        # measured + velocity * lookahead), so it always reads back as exactly
        # max_joint_velocity * velocity_point_2_sec and says nothing about the
        # arm. Instead sweep a delay and find the one that best lines the
        # measured trace up with the commanded one — that delay is the real lag.
        def rms_at(delay):
            err, n = 0.0, 0
            mi = 0
            for t, c in self.cmd:
                tt = t + delay
                while mi + 1 < len(meas) and meas[mi + 1][0] <= tt:
                    mi += 1
                if not (meas[0][0] <= tt <= meas[-1][0]):
                    continue
                m = meas[mi][1]
                err += max(abs(a - b) for a, b in zip(c, m)) ** 2
                n += 1
            return (err / n) ** 0.5 if n else float("inf")

        best = min(((rms_at(d / 1000.0), d) for d in range(0, 305, 5)), key=lambda x: x[0])
        print("\nTRACKING LAG (delay that best aligns measured with commanded)")
        print(f"  lag {best[1]:3d} ms   residual error {math.degrees(best[0]):.2f} deg")
        print(f"  at 0 ms the error is {math.degrees(rms_at(0.0)):.2f} deg"
              f" — that is the designed lead, not a fault.")

        # --- overshoot after release ----------------------------------------
        # Each measurement window MUST end when the stick is pushed again.
        # Otherwise the window swallows the next commanded move and reports it
        # as overshoot — with releases 1-3 s apart and a fixed 4 s window that
        # inflated the figure roughly threefold.
        rel = [
            self.joy[i][0]
            for i in range(1, len(self.joy))
            if self.joy[i - 1][1] and not self.joy[i][1]
        ]
        act = [
            self.joy[i][0]
            for i in range(1, len(self.joy))
            if not self.joy[i - 1][1] and self.joy[i][1]
        ]

        def gates_during(a, b):
            """Was RB/RT/Y held at any point in [a, b]?"""
            g = [j for j in self.joy if a <= j[0] <= b]
            if not g:
                return "", False
            rb = any(j[2] for j in g)
            rt = any(j[3] for j in g)
            y = any(j[4] for j in g)
            tags = "".join([("R" if rb else ""), ("T" if rt else ""), ("Y" if y else "")])
            return (tags or "-"), y

        MAX_WINDOW = 4.0
        MIN_QUIET = 0.30   # shorter than this is a direction change, not a release

        print(f"\nRELEASES DETECTED: {len(rel)}")
        if rel:
            print("\nPOST-RELEASE TRAVEL (per release, worst joint)")
            print(f"  {'release':>9} {'travel':>10} {'settle':>9}  {'window':>8}"
                  f" {'cmds':>5} {'held':>5} {'v_rel':>6} {'stop_t':>8}"
                  f" {'holddrift':>9}")
            travels = []
            clean = []
            skipped = 0
            for r in rel:
                nxt = next((t for t in act if t > r), None)
                end = min(r + MAX_WINDOW, nxt - 0.05) if nxt else r + MAX_WINDOW
                quiet = end - r
                if quiet < MIN_QUIET:
                    skipped += 1
                    continue

                # Speed at the instant of release. Raw overshoot is meaningless
                # across sessions because it scales with how hard you were
                # driving; overshoot / release-speed is the invariant, and has
                # units of seconds — the arm's effective stopping time.
                #
                # Derive it from POSITIONS over a 100 ms baseline, not from the
                # velocity field. igus_rebel/src/Rebel.cpp fills the velocity
                # state with a raw unfiltered first difference,
                #   vel[i] = (pos[i] - last_pos[i]) / period
                # so differencing the CAN-quantised position produces large
                # spurious spikes. Taking a max over joints and samples then
                # selects the worst spike, which read 1.1 rad/s against a
                # commanded cap of 0.48. A longer baseline averages that out.
                BASE = 0.10
                pre = [m for m in self.meas if r - BASE <= m[0] <= r]
                if len(pre) >= 2:
                    dt = pre[-1][0] - pre[0][0]
                    v_rel = (
                        max(abs(a - b) for a, b in zip(pre[-1][1], pre[0][1])) / dt
                        if dt > 1e-3 else 0.0
                    )
                else:
                    v_rel = 0.0

                after = [(t, p) for t, p, _ in self.meas if r <= t <= end]
                if len(after) < 3:
                    continue
                p0 = after[0][1]
                far, tset = 0.0, 0.0
                for t, p in after:
                    d = max(abs(a - b) for a, b in zip(p, p0))
                    if d > far:
                        far, tset = d, t - r
                # How many trajectory commands did the node emit while "quiet"?
                # By design it publishes stop_zero_cycles hold frames and then
                # goes silent, so a big count means something else is driving.
                held_cmds = [c for t, c in self.cmd if r < t <= end]
                ncmd = len(held_cmds)
                tags, hand_guiding = gates_during(r, end)

                # Is the hold target actually frozen? The node is supposed to
                # latch it at the release point. If instead it tracks the live
                # measurement, the target drifts by roughly the overshoot and
                # the position error the JTC sees stays at ~0 — meaning no
                # braking, whatever the gains are.
                if len(held_cmds) >= 2:
                    drift = max(
                        max(c[j] for c in held_cmds) - min(c[j] for c in held_cmds)
                        for j in range(len(ARM_JOINTS))
                    )
                else:
                    drift = float("nan")

                travels.append(far)
                stop_t = far / v_rel if v_rel > 0.02 else float("nan")
                # Judge "is something else driving the arm?" by whether the
                # commanded target MOVES, not by how many messages arrive. A
                # raw count breaks whenever stop_zero_cycles changes: it is
                # legitimately 24 now, which a fixed <=12 threshold wrongly
                # flagged as every sample being untrustworthy. A frozen target
                # (drift ~0) means the node is holding, whatever the count.
                driven = drift == drift and drift > 0.05   # >2.9 deg of target motion
                if not driven and not hand_guiding:
                    clean.append((far, v_rel, stop_t))
                cap = " still moving" if tset > quiet - 0.05 else ""
                print(f"  {r % 1000:9.1f} {math.degrees(far):9.2f}d {tset:8.2f}s"
                      f"  {quiet:7.2f}s {ncmd:5d} {tags:>5} {v_rel:6.3f} "
                      f"{stop_t:7.3f}s {math.degrees(drift):7.2f}d{cap}")

            if skipped:
                print(f"\n  ({skipped} releases skipped: stick re-engaged within "
                      f"{MIN_QUIET}s, so they are direction changes, not stops)")
            print("\n  cmds = trajectory messages published during the quiet window."
                  "\n         Should equal stop_zero_cycles (check with:"
                  "\n         ros2 param get /rebel_servo_teleop_gamepad stop_zero_cycles)."
                  "\n  held = R(B) / (R)T / Y held during the window. Y is hand guiding:"
                  "\n         the arm goes limp and sags, which is not teleop overshoot."
                  "\n  holddrift = spread of the commanded hold target across the window."
                  "\n         ~0      -> target is latched, the JTC sees a real error"
                  "\n                    and CAN brake; any remaining overshoot is the"
                  "\n                    gain being too weak or plain transport delay."
                  "\n         ~travel -> target is still following the measurement, so"
                  "\n                    the error stays ~0 and NOTHING brakes the arm,"
                  "\n                    no matter what p is set to.")
            if clean:
                far_l = [c[0] for c in clean]
                v_l = [c[1] for c in clean]
                st_l = [c[2] for c in clean if c[2] == c[2]]
                print(f"\n  TRUSTWORTHY SAMPLES ({len(clean)} of {len(travels)}): "
                      f"mean {math.degrees(sum(far_l)/len(far_l)):.2f} deg, "
                      f"worst {math.degrees(max(far_l)):.2f} deg")
                print(f"  mean speed at release: {sum(v_l)/len(v_l):.3f} rad/s")
                if st_l:
                    print(f"\n  >>> STOPPING TIME  mean {sum(st_l)/len(st_l):.3f} s"
                          f"   (overshoot / release speed) <<<")
                    print("  THIS is the number to compare between tuning runs — it is"
                          "\n  independent of how hard you happened to be driving."
                          "\n  It should land near the measured tracking lag; if it does,"
                          "\n  the overshoot is transport delay and no gain can remove it.")
            else:
                print("\n  NO trustworthy samples — every window had the arm driven by"
                      "\n  something other than the sticks. Re-run: push a stick, stop,"
                      "\n  and wait a full second without touching anything, several times.")
            if travels:
                print(f"\n  (all samples incl. untrustworthy: mean "
                      f"{math.degrees(sum(travels)/len(travels)):.2f} deg, worst "
                      f"{math.degrees(max(travels)):.2f} deg — judge by the "
                      f"trustworthy figure above)")
                print("\n  Compare the TRUSTWORTHY mean against the designed lead"
                      "\n  (max_joint_velocity * velocity_point_2_sec, ~1.4 deg):"
                      "\n    similar        -> the arm is finishing commanded motion;"
                      "\n                      shrink the lead."
                      "\n    a few x larger -> the arm coasts on its own velocity loop"
                      "\n                      for the measured tracking lag; the fix is"
                      "\n                      lower max_joint_velocity or firmer JTC gains."
                      "\n    tens of deg    -> not overshoot at all. Something else is"
                      "\n                      moving the arm; check the cmds/held columns.")


def main():
    rclpy.init()
    n = TrackingProbe()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.report()
        n.destroy_node()
        # Ctrl-C makes rclpy tear the context down itself; shutting down twice
        # raises and would bury the report above.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main() or 0)
