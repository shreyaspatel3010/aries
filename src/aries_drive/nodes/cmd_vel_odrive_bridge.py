#!/usr/bin/env python3
"""Fail-safe skid-steer Twist adapter for the six Aries ODrive axes."""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Iterable

import rclpy
from geometry_msgs.msg import Twist
from odrive_can.msg import ControlMessage
from odrive_can.srv import AxisState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool


CLOSED_LOOP_CONTROL = 8
IDLE = 1


def select_drive_output_mode(
    enable_requested: bool,
    armed: bool,
    command_fresh: bool,
) -> str:
    """Choose whether the periodic loop should drive, stop, or stay silent.

    An IDLE/disarmed ODrive does not need a continuous stream of zero velocity
    commands.  Keeping that stream off the CAN bus leaves capacity for the six
    axes' high-rate encoder broadcasts and for status traffic.
    """
    if not enable_requested or not armed:
        return "silent"
    if not command_fresh:
        return "stop"
    return "drive"


def twist_to_wheel_rps(
    linear_mps: float,
    angular_rps: float,
    track_width_m: float,
    wheel_circumference_m: float,
    max_wheel_rps: float,
) -> tuple[float, float]:
    """Return right/left ODrive revolutions per second for an Aries Twist.

    The physical left wheels are mounted opposite to the right wheels, so a
    positive forward velocity requires positive right-axis velocity and
    negative left-axis velocity.
    """
    values = (
        float(linear_mps),
        float(angular_rps),
        float(track_width_m),
        float(wheel_circumference_m),
        float(max_wheel_rps),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Twist and drivetrain parameters must be finite")
    if track_width_m <= 0.0:
        raise ValueError("track_width_m must be positive")
    if wheel_circumference_m <= 0.0:
        raise ValueError("wheel_circumference_m must be positive")
    if max_wheel_rps <= 0.0:
        raise ValueError("max_wheel_rps must be positive")

    physical_right_mps = linear_mps + angular_rps * track_width_m * 0.5
    physical_left_mps = linear_mps - angular_rps * track_width_m * 0.5
    right_rps = physical_right_mps / wheel_circumference_m
    left_rps = -physical_left_mps / wheel_circumference_m

    peak = max(abs(right_rps), abs(left_rps))
    if peak > max_wheel_rps:
        scale = max_wheel_rps / peak
        right_rps *= scale
        left_rps *= scale
    return right_rps, left_rps


def ramp(current: float, target: float, max_delta: float) -> float:
    """Move current toward target by at most max_delta."""
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


class CmdVelOdriveBridge(Node):
    """Convert the single hardware-facing cmd_vel into six ODrive commands."""

    def __init__(self) -> None:
        super().__init__("cmd_vel_odrive_bridge")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("enable_service", "/aries_drive/enable")
        self.declare_parameter("enabled_topic", "/aries_drive/enabled")
        self.declare_parameter("status_topic", "/aries_drive/status")
        self.declare_parameter("num_axes", 6)
        # Overridden by cmd_vel_odrive_bridge.yaml in every launch path; these
        # defaults track it so running this node bare is not silently wrong.
        self.declare_parameter("right_wheels", [0, 4, 3])
        self.declare_parameter("left_wheels", [5, 1, 2])
        self.declare_parameter("wheel_circumference_m", 0.697)
        self.declare_parameter("track_width_m", 0.566)
        self.declare_parameter("max_linear_mps", 0.45)
        self.declare_parameter("max_angular_rps", 2.10)
        self.declare_parameter("max_wheel_rps", 1.50)
        self.declare_parameter("wheel_accel_rps2", 3.0)
        self.declare_parameter("command_timeout_s", 0.25)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("arm_retry_period_s", 2.0)
        self.declare_parameter("axis_state_request_timeout_s", 3.0)
        self.declare_parameter("auto_arm", False)

        self.cmd_vel_topic = self._topic("cmd_vel_topic")
        self.enable_service = self._topic("enable_service")
        self.enabled_topic = self._topic("enabled_topic")
        self.status_topic = self._topic("status_topic")
        self.num_axes = int(self.get_parameter("num_axes").value)
        self.right_wheels = self._wheel_indices("right_wheels")
        self.left_wheels = self._wheel_indices("left_wheels")
        self.wheel_circumference_m = float(
            self.get_parameter("wheel_circumference_m").value
        )
        self.track_width_m = float(self.get_parameter("track_width_m").value)
        self.max_linear_mps = max(
            0.0, float(self.get_parameter("max_linear_mps").value)
        )
        self.max_angular_rps = max(
            0.0, float(self.get_parameter("max_angular_rps").value)
        )
        self.max_wheel_rps = float(self.get_parameter("max_wheel_rps").value)
        self.wheel_accel_rps2 = max(
            0.0, float(self.get_parameter("wheel_accel_rps2").value)
        )
        self.command_timeout_s = max(
            0.05, float(self.get_parameter("command_timeout_s").value)
        )
        self.publish_rate_hz = max(
            10.0, float(self.get_parameter("publish_rate_hz").value)
        )
        self.arm_retry_period_s = max(
            0.25, float(self.get_parameter("arm_retry_period_s").value)
        )
        self.axis_state_request_timeout_s = max(
            0.25,
            float(self.get_parameter("axis_state_request_timeout_s").value),
        )
        self.auto_arm = bool(self.get_parameter("auto_arm").value)

        self._validate_drivetrain()
        self._period = 1.0 / self.publish_rate_hz
        self._target_linear = 0.0
        self._target_angular = 0.0
        self._current_right_rps = 0.0
        self._current_left_rps = 0.0
        self._last_cmd_at: float | None = None
        self._command_valid = False
        self._enable_requested = self.auto_arm
        self._armed = False
        self._pending_axes = set(range(self.num_axes))
        self._arm_in_progress = False
        self._state_lock = threading.Lock()

        self.axis_publishers = [
            self.create_publisher(
                ControlMessage, f"/odrive_axis{i}/control_message", 10
            )
            for i in range(self.num_axes)
        ]
        self.axis_clients = [
            self.create_client(
                AxisState, f"/odrive_axis{i}/request_axis_state"
            )
            for i in range(self.num_axes)
        ]

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.enabled_pub = self.create_publisher(
            Bool, self.enabled_topic, latched_qos
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_subscription(
            Twist, self.cmd_vel_topic, self._on_cmd_vel, 10
        )
        self.create_service(SetBool, self.enable_service, self._on_enable)
        self.create_timer(self._period, self._publish_loop)
        self.create_timer(self.arm_retry_period_s, self._retry_arm)
        self.create_timer(0.5, self._publish_status)

        self._publish_enabled(False)
        self._publish_stop()
        self.get_logger().info(
            "Fail-safe ODrive bridge ready: "
            f"{self.cmd_vel_topic} -> axes 0..{self.num_axes - 1}; "
            f"enable with {self.enable_service}; auto_arm={self.auto_arm}"
        )

        if self.auto_arm:
            self._retry_arm()

    def _topic(self, name: str) -> str:
        value = str(self.get_parameter(name).value or "").strip()
        if not value:
            raise ValueError(f"{name} must not be empty")
        return value if value.startswith("/") else "/" + value

    def _wheel_indices(self, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in self.get_parameter(name).value)

    def _validate_drivetrain(self) -> None:
        if self.num_axes <= 0:
            raise ValueError("num_axes must be positive")
        right = set(self.right_wheels)
        left = set(self.left_wheels)
        expected = set(range(self.num_axes))
        if not right or not left:
            raise ValueError("right_wheels and left_wheels must not be empty")
        if right & left:
            raise ValueError("right_wheels and left_wheels must be disjoint")
        if right | left != expected:
            raise ValueError(
                "right_wheels and left_wheels must cover every configured axis"
            )
        twist_to_wheel_rps(
            0.0,
            0.0,
            self.track_width_m,
            self.wheel_circumference_m,
            self.max_wheel_rps,
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        linear = float(msg.linear.x)
        angular = float(msg.angular.z)
        if not math.isfinite(linear) or not math.isfinite(angular):
            with self._state_lock:
                self._command_valid = False
                self._last_cmd_at = None
            self.get_logger().error("Rejected non-finite cmd_vel; stopping")
            return

        with self._state_lock:
            self._target_linear = max(
                -self.max_linear_mps, min(self.max_linear_mps, linear)
            )
            self._target_angular = max(
                -self.max_angular_rps, min(self.max_angular_rps, angular)
            )
            self._last_cmd_at = time.monotonic()
            self._command_valid = True

    def _on_enable(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        if request.data:
            with self._state_lock:
                self._enable_requested = True
                self._armed = False
                self._pending_axes = set(range(self.num_axes))
                self._last_cmd_at = None
                self._command_valid = False
            self._publish_enabled(False)
            self._retry_arm()
            response.success = True
            response.message = (
                "ODrive arming requested; motion remains blocked until all axes "
                "confirm CLOSED_LOOP_CONTROL"
            )
        else:
            with self._state_lock:
                self._enable_requested = False
                self._armed = False
                self._pending_axes = set(range(self.num_axes))
                self._last_cmd_at = None
                self._command_valid = False
                self._current_right_rps = 0.0
                self._current_left_rps = 0.0
            self._publish_stop()
            self._publish_enabled(False)
            self._request_axis_state(IDLE)
            response.success = True
            response.message = "Drive disabled, zero commanded, IDLE requested"
        return response

    def _retry_arm(self) -> None:
        with self._state_lock:
            should_arm = (
                self._enable_requested
                and not self._armed
                and bool(self._pending_axes)
                and not self._arm_in_progress
            )
            if should_arm:
                self._arm_in_progress = True
        if should_arm:
            threading.Thread(target=self._arm_thread, daemon=True).start()

    def _arm_thread(self) -> None:
        futures = {}
        failures = []
        with self._state_lock:
            pending = sorted(self._pending_axes)

        # Establish a zero setpoint before any axis enters CLOSED_LOOP_CONTROL.
        # The regular timer stays silent while disarmed to avoid flooding CAN.
        self._publish_stop()

        for axis in pending:
            client = self.axis_clients[axis]
            if not client.wait_for_service(timeout_sec=0.1):
                failures.append(f"axis {axis}: service unavailable")
                continue
            request = AxisState.Request()
            request.axis_requested_state = CLOSED_LOOP_CONTROL
            futures[axis] = client.call_async(request)

        still_pending = set(pending)
        deadline = time.monotonic() + self.axis_state_request_timeout_s
        while futures and time.monotonic() < deadline:
            for axis in [item for item, future in futures.items() if future.done()]:
                future = futures.pop(axis)
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(f"axis {axis}: service error: {exc}")
                    continue
                if result is not None and result.axis_state == CLOSED_LOOP_CONTROL:
                    still_pending.discard(axis)
                elif result is None:
                    failures.append(f"axis {axis}: empty service result")
                else:
                    failures.append(
                        f"axis {axis}: state={result.axis_state} "
                        f"errors=0x{result.active_errors:08X}"
                    )
            if futures:
                time.sleep(0.02)

        for axis in futures:
            failures.append(f"axis {axis}: request timed out")

        with self._state_lock:
            self._pending_axes = still_pending
            enabled = self._enable_requested and not still_pending
            self._armed = enabled
            self._arm_in_progress = False
            self._last_cmd_at = None
            self._command_valid = False

        self._publish_enabled(enabled)
        if enabled:
            self.get_logger().info(
                "All ODrive axes confirmed CLOSED_LOOP_CONTROL; drive enabled"
            )
        elif failures:
            self.get_logger().warn(
                "Drive remains disabled: " + "; ".join(failures)
            )

    def _request_axis_state(self, state: int) -> None:
        for client in self.axis_clients:
            if client.service_is_ready():
                request = AxisState.Request()
                request.axis_requested_state = state
                client.call_async(request)

    @staticmethod
    def _control_message(velocity: float) -> ControlMessage:
        msg = ControlMessage()
        msg.control_mode = 2
        msg.input_mode = 1
        msg.input_pos = 0.0
        msg.input_vel = float(velocity)
        msg.input_torque = 0.0
        return msg

    def _publish_wheels(self, right_rps: float, left_rps: float) -> None:
        right_msg = self._control_message(right_rps)
        left_msg = self._control_message(left_rps)
        for axis in self.right_wheels:
            self.axis_publishers[axis].publish(right_msg)
        for axis in self.left_wheels:
            self.axis_publishers[axis].publish(left_msg)

    def _publish_stop(self) -> None:
        self._publish_wheels(0.0, 0.0)

    def _publish_enabled(self, enabled: bool) -> None:
        self.enabled_pub.publish(Bool(data=bool(enabled)))

    def _publish_loop(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            fresh = (
                self._command_valid
                and self._last_cmd_at is not None
                and now - self._last_cmd_at <= self.command_timeout_s
            )
            enable_requested = self._enable_requested
            armed = self._armed
            target_linear = self._target_linear
            target_angular = self._target_angular

        output_mode = select_drive_output_mode(
            enable_requested,
            armed,
            fresh,
        )
        if output_mode == "silent":
            self._current_right_rps = 0.0
            self._current_left_rps = 0.0
            return
        if output_mode == "stop":
            self._current_right_rps = 0.0
            self._current_left_rps = 0.0
            self._publish_stop()
            return

        target_right, target_left = twist_to_wheel_rps(
            target_linear,
            target_angular,
            self.track_width_m,
            self.wheel_circumference_m,
            self.max_wheel_rps,
        )
        if abs(target_linear) < 1e-6 and abs(target_angular) < 1e-6:
            self._current_right_rps = 0.0
            self._current_left_rps = 0.0
        else:
            max_delta = self.wheel_accel_rps2 * self._period
            self._current_right_rps = ramp(
                self._current_right_rps, target_right, max_delta
            )
            self._current_left_rps = ramp(
                self._current_left_rps, target_left, max_delta
            )
        self._publish_wheels(
            self._current_right_rps, self._current_left_rps
        )

    def _publish_status(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            age = (
                None
                if self._last_cmd_at is None
                else max(0.0, now - self._last_cmd_at)
            )
            status = {
                "enable_requested": self._enable_requested,
                "armed": self._armed,
                "pending_axes": sorted(self._pending_axes),
                "command_valid": self._command_valid,
                "command_age_s": age,
                "command_timeout_s": self.command_timeout_s,
                "right_rps": self._current_right_rps,
                "left_rps": self._current_left_rps,
            }
        self.status_pub.publish(
            String(data=json.dumps(status, sort_keys=True))
        )

    def stop(self) -> None:
        with self._state_lock:
            self._enable_requested = False
            self._armed = False
            self._current_right_rps = 0.0
            self._current_left_rps = 0.0
        for _ in range(3):
            self._publish_stop()
            time.sleep(0.02)


def main(args: Iterable[str] | None = None) -> None:
    # Keep the context alive through Ctrl-C so final zero ControlMessages can
    # be delivered before DDS and the ODrive publishers are torn down.
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = CmdVelOdriveBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
