#!/usr/bin/env python3

import math
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from odrive_can.msg import ControlMessage
from odrive_can.srv import AxisState
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty


class RoverJoystickController(Node):
    def __init__(self):
        super().__init__("rover_joystick_controller")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("max_velocity", 1.5)
        self.declare_parameter("turn_boost", 1.5)
        self.declare_parameter("turn_threshold", 0.2)
        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("accel_limit", 3.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("arm_retry_period", 3.0)
        self.declare_parameter("axis_state_request_timeout", 3.0)

        # Stop when /joy goes quiet. The publish loop below runs on a timer and
        # sends the LAST joystick state, so without this a dead /joy reads
        # exactly like a held stick and the wheels keep turning.
        #
        # This node writes to the ODrive axes directly -- there is no
        # cmd_vel_odrive_bridge in the path and therefore no second timeout
        # behind it. Whatever safety exists here is all of it.
        #
        # 0.35 s matches every other node on the pad. Unlike the cmd_vel path,
        # the stop RAMPS at accel_limit rather than jumping to zero: six axes
        # commanded to standstill in one control period is a mechanical shock
        # that can trip the drives, and a tripped drive needs LB+Y to recover,
        # which is a worse place to be than coasting to a halt.
        self.declare_parameter("joy_timeout_sec", 0.35)

        self.declare_parameter("vertical_axis", 1)
        self.declare_parameter("horizontal_axis", 0)
        self.declare_parameter("invert_vertical", True)
        self.declare_parameter("invert_horizontal", False)

        self.declare_parameter("trigger_button", 4)
        self.declare_parameter("sound_button", 3)

        # LB + Y re-initialises every ODrive axis. The modifier defaults to the
        # same LB that already gates drive output, so the combo cannot be hit
        # while the arm has the stick (RB) and cannot be hit by accident with a
        # bare button press.
        self.declare_parameter("reinit_button", 3)
        self.declare_parameter("reinit_modifier_button", 4)
        # A faulted axis rejects CLOSED_LOOP_CONTROL until its errors are
        # cleared, which is the usual reason a re-arm is needed at all.
        self.declare_parameter("clear_errors_on_reinit", True)
        # Arming closes the loop on all six motors at once. Hold output at zero
        # for this long afterwards so a deflected stick cannot lurch the rover
        # the instant the axes engage — LB is by definition already held.
        self.declare_parameter("reinit_hold_sec", 1.0)

        self.declare_parameter("num_axes", 6)
        # Defaults match the post-reassembly CAN node id -> wheel mapping in
        # aries_drive/config/cmd_vel_odrive_bridge.yaml.
        self.declare_parameter("right_wheels", [0, 4, 3])
        self.declare_parameter("left_wheels", [5, 1, 2])
        self.declare_parameter("sound_file", "")

        joy_topic = str(self.get_parameter("joy_topic").value)
        self.max_velocity = float(self.get_parameter("max_velocity").value)
        self.turn_boost = float(self.get_parameter("turn_boost").value)
        self.turn_threshold = float(self.get_parameter("turn_threshold").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.accel_limit = float(self.get_parameter("accel_limit").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.arm_retry_period = float(self.get_parameter("arm_retry_period").value)
        self.axis_state_request_timeout = float(self.get_parameter("axis_state_request_timeout").value)
        self.joy_timeout_sec = max(0.0, float(self.get_parameter("joy_timeout_sec").value))

        self.vertical_axis = int(self.get_parameter("vertical_axis").value)
        self.horizontal_axis = int(self.get_parameter("horizontal_axis").value)
        self.invert_vertical = bool(self.get_parameter("invert_vertical").value)
        self.invert_horizontal = bool(self.get_parameter("invert_horizontal").value)

        self.trigger_button = int(self.get_parameter("trigger_button").value)
        self.sound_button = int(self.get_parameter("sound_button").value)

        self.reinit_button = int(self.get_parameter("reinit_button").value)
        self.reinit_modifier_button = int(self.get_parameter("reinit_modifier_button").value)
        self.clear_errors_on_reinit = bool(self.get_parameter("clear_errors_on_reinit").value)
        self.reinit_hold_sec = float(self.get_parameter("reinit_hold_sec").value)

        self.num_axes = int(self.get_parameter("num_axes").value)
        self.right_wheels = list(self.get_parameter("right_wheels").value)
        self.left_wheels = list(self.get_parameter("left_wheels").value)
        self.sound_file = str(self.get_parameter("sound_file").value)

        self.period = 1.0 / max(self.publish_rate_hz, 1.0)

        self.trigger = 0
        self.target_right_vel = 0.0
        self.target_left_vel = 0.0
        self.current_right_vel = 0.0
        self.current_left_vel = 0.0

        # Zero, not "now": before the first Joy arrives there is no command to
        # be stale, and treating startup as fresh would open the gate.
        self._last_joy_at = 0.0
        self._joy_lost = False

        self.prev_sound_button = 0

        self.axis_publishers = []
        self.axis_clients = []
        self.clear_error_clients = []
        self.pending_axes = []

        for i in range(self.num_axes):
            self.axis_publishers.append(
                self.create_publisher(ControlMessage, f"/odrive_axis{i}/control_message", 10)
            )
            self.axis_clients.append(
                self.create_client(AxisState, f"/odrive_axis{i}/request_axis_state")
            )
            self.clear_error_clients.append(
                self.create_client(Empty, f"/odrive_axis{i}/clear_errors")
            )
            self.pending_axes.append(i)

        # Initialised before the callbacks that read them are registered; a timer
        # or subscription firing first would otherwise raise AttributeError.
        self._retry_in_progress = False
        self._last_arm_status_log = 0.0
        self._prev_reinit_combo = False
        self._reinit_hold_until = 0.0

        self.create_subscription(Joy, joy_topic, self._joy_callback, 10)
        self.create_timer(self.period, self._publish_loop)
        self.create_timer(self.arm_retry_period, self._retry_arm)

        self.get_logger().info(
            f"Rover hardware joystick ready. Hold LB/button {self.trigger_button}. "
            f"vertical_axis={self.vertical_axis}, horizontal_axis={self.horizontal_axis}. "
            f"Press button {self.reinit_modifier_button}+{self.reinit_button} (LB+Y) "
            f"to re-initialise all {self.num_axes} ODrive axes."
        )

    def _axis(self, msg, index):
        return float(msg.axes[index]) if 0 <= index < len(msg.axes) else 0.0

    def _button(self, msg, index):
        return int(msg.buttons[index]) if 0 <= index < len(msg.buttons) else 0

    def _deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * (abs(value) - self.deadzone) / max(1e-6, 1.0 - self.deadzone)

    def _ramp(self, current, target):
        step = abs(self.accel_limit) * self.period
        diff = target - current
        if abs(diff) <= step:
            return target
        return current + math.copysign(step, diff)

    def _joy_callback(self, msg):
        self._last_joy_at = time.monotonic()
        if self._joy_lost:
            self._joy_lost = False
            self.get_logger().info("joystick back: drive re-enabled (hold LB)")

        vertical = self._axis(msg, self.vertical_axis)
        horizontal = self._axis(msg, self.horizontal_axis)

        if self.invert_vertical:
            vertical = -vertical
        if self.invert_horizontal:
            horizontal = -horizontal

        vertical = self._deadzone(vertical)
        horizontal = self._deadzone(horizontal)

        self.trigger = self._button(msg, self.trigger_button)
        sound_button = self._button(msg, self.sound_button)

        if abs(vertical) < self.turn_threshold and abs(horizontal) > 0.1:
            turn_vel = horizontal * self.max_velocity * self.turn_boost
            self.target_right_vel = turn_vel
            self.target_left_vel = turn_vel
        else:
            self.target_right_vel = -(vertical - horizontal) * self.max_velocity
            self.target_left_vel = (vertical + horizontal) * self.max_velocity

        # LB + Y, edge-triggered so holding the pair re-arms exactly once.
        reinit_combo = bool(
            self._button(msg, self.reinit_modifier_button)
            and self._button(msg, self.reinit_button)
        )
        if reinit_combo and not self._prev_reinit_combo:
            self._request_reinit()
        self._prev_reinit_combo = reinit_combo

        # Y alone still plays the sound. Suppressed while the modifier is held,
        # otherwise LB+Y would re-arm the ODrives AND fire the clip, and the
        # sound is the louder of the two signals.
        if (
            sound_button == 1
            and self.prev_sound_button == 0
            and self.sound_file
            and not self._button(msg, self.reinit_modifier_button)
        ):
            subprocess.Popen(["aplay", self.sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.prev_sound_button = sound_button

    def _request_reinit(self):
        """LB + Y: clear errors on every ODrive axis, then re-arm CLOSED_LOOP."""
        if self._retry_in_progress:
            self.get_logger().warn("ODrive re-init already running — ignoring LB+Y")
            return

        # Drop the command and hold zero before anything engages.
        self.target_right_vel = 0.0
        self.target_left_vel = 0.0
        self.current_right_vel = 0.0
        self.current_left_vel = 0.0
        self._reinit_hold_until = time.monotonic() + self.reinit_hold_sec

        self.pending_axes = list(range(self.num_axes))
        self.get_logger().warn(
            f"LB+Y: re-initialising all {self.num_axes} ODrive axes "
            f"(clear_errors={self.clear_errors_on_reinit})"
        )
        self._retry_in_progress = True
        threading.Thread(target=self._reinit_thread, daemon=True).start()

    def _reinit_thread(self):
        if self.clear_errors_on_reinit:
            futures = {}
            for i in range(self.num_axes):
                client = self.clear_error_clients[i]
                if not client.wait_for_service(timeout_sec=0.1):
                    self.get_logger().warn(f"odrive_axis{i}: clear_errors unavailable")
                    continue
                futures[i] = client.call_async(Empty.Request())

            deadline = time.monotonic() + self.axis_state_request_timeout
            while futures and time.monotonic() < deadline:
                for i in [i for i, f in futures.items() if f.done()]:
                    futures.pop(i)
                if futures:
                    time.sleep(0.05)
            for i in sorted(futures):
                self.get_logger().warn(f"odrive_axis{i}: clear_errors timed out")

        # Re-arm through the same path the periodic retry uses; it also clears
        # _retry_in_progress on the way out.
        self._retry_arm_thread()

    def _retry_arm(self):
        if not self.pending_axes or self._retry_in_progress:
            return
        self._retry_in_progress = True
        threading.Thread(target=self._retry_arm_thread, daemon=True).start()

    def _retry_arm_thread(self):
        still_pending = []
        futures = {}
        failures = []

        for i in list(self.pending_axes):
            client = self.axis_clients[i]
            if not client.wait_for_service(timeout_sec=0.1):
                still_pending.append(i)
                failures.append(f"axis {i}: service unavailable")
                continue

            req = AxisState.Request()
            req.axis_requested_state = 8
            futures[i] = client.call_async(req)

        deadline = time.monotonic() + self.axis_state_request_timeout
        while futures and time.monotonic() < deadline:
            done_axes = [i for i, future in futures.items() if future.done()]
            for i in done_axes:
                future = futures.pop(i)
                result = future.result()
                if result is not None and result.axis_state == 8:
                    self.get_logger().info(f"odrive_axis{i} CLOSED_LOOP_CONTROL confirmed")
                else:
                    still_pending.append(i)
                    if result is None:
                        failures.append(f"axis {i}: empty service result")
                    else:
                        failures.append(
                            f"axis {i}: state={result.axis_state} "
                            f"active_errors=0x{result.active_errors:08X} "
                            f"procedure_result={result.procedure_result}"
                        )
            if futures:
                time.sleep(0.05)

        for i in sorted(futures):
            still_pending.append(i)
            failures.append(f"axis {i}: request timed out")

        if failures:
            now = time.monotonic()
            if now - self._last_arm_status_log > 5.0:
                self.get_logger().warn(
                    "ODrive axes still not CLOSED_LOOP: " + "; ".join(failures)
                )
                self._last_arm_status_log = now

        if not still_pending and self.pending_axes:
            self.get_logger().info("All ODrive axes are in CLOSED_LOOP_CONTROL")

        self.pending_axes = still_pending
        self._retry_in_progress = False

    def _publish_loop(self):
        now = time.monotonic()

        # A silent /joy is not a held stick. Treated as a release, so the ramp
        # below brings the wheels down at accel_limit instead of stopping them
        # dead -- see joy_timeout_sec.
        joy_stale = (
            self.joy_timeout_sec > 0.0
            and (now - self._last_joy_at) > self.joy_timeout_sec
        )
        if joy_stale:
            if not self._joy_lost and self._last_joy_at > 0.0:
                self._joy_lost = True
                self.get_logger().warn(
                    f"no /joy for {self.joy_timeout_sec:.2f} s — ramping the rover "
                    f"to a stop (link down, or the joy driver died)"
                )
            self.target_right_vel = 0.0
            self.target_left_vel = 0.0
            self.trigger = 0

        # LB is held throughout an LB+Y re-init, so without this the rover would
        # accelerate the moment the axes armed if the stick were off centre.
        holding_after_reinit = now < self._reinit_hold_until

        if self.trigger == 1 and not holding_after_reinit:
            target_right = self.target_right_vel
            target_left = self.target_left_vel
        else:
            target_right = 0.0
            target_left = 0.0

        self.current_right_vel = self._ramp(self.current_right_vel, target_right)
        self.current_left_vel = self._ramp(self.current_left_vel, target_left)

        right_msg = ControlMessage()
        right_msg.control_mode = 2
        right_msg.input_mode = 1
        right_msg.input_pos = 0.0
        right_msg.input_vel = self.current_right_vel
        right_msg.input_torque = 0.0

        left_msg = ControlMessage()
        left_msg.control_mode = 2
        left_msg.input_mode = 1
        left_msg.input_pos = 0.0
        left_msg.input_vel = self.current_left_vel
        left_msg.input_torque = 0.0

        for i in self.right_wheels:
            if 0 <= int(i) < len(self.axis_publishers):
                self.axis_publishers[int(i)].publish(right_msg)

        for i in self.left_wheels:
            if 0 <= int(i) < len(self.axis_publishers):
                self.axis_publishers[int(i)].publish(left_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoverJoystickController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
