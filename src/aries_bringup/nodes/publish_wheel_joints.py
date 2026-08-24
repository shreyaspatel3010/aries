#!/usr/bin/env python3
"""Publish Aries wheel joint states from the six physical ODrive encoders."""

from __future__ import annotations

import math

import rclpy
from odrive_can.msg import ControllerStatus
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


PASSIVE_JOINTS = (
    "L_Rocker_Joint",
    "R_Rocker_Joint",
    "L_Boggie_Joint",
    "R_Boggie_Joint",
    "aux_L_Rocker_joint",
    "aux_R_Rocker_joint",
    # The drill joints exist in the URDF but nothing drives them: no
    # ros2_control interface, no hardware. Unpublished, MoveIt's planning scene
    # monitor never completes a robot state and warns "The complete state of the
    # robot is not yet known" forever, which blocks servo. Zero is the stowed
    # pose and is inside every limit (the container's upper bound is exactly 0).
    "drill_motor_joint",
    "drill_bit_joint",
    "drill_container_joint",
)

# CAN node id -> URDF wheel joint. TF needs the exact joint, unlike drive
# commands and odometry, which only care which side an axis is on.
#
# Re-established physically on 2026-08-24 by arming one axis at a time and
# seeing which wheel resisted turning:
#
#   axis 0 Left-Front    axis 1 Left-Mid     axis 2 Left-Rear
#   axis 3 Right-Rear    axis 4 Right-Mid    axis 5 Right-Front
#
# The node ids are contiguous per side (left 0..2, right 3..5) but the right
# block runs rear -> front, so it is not a plain 0..2 / 3..5 front-to-rear split.
#
# _3 = FRONT, _2 = mid, _1 = REAR on both sides. Do not try to read that order
# off the joint origins in right_link.xacro / left_link.xacro: _1 and _2 hang
# off the boggie link while _3 hangs off the rocker, so their origin x values
# are expressed in different parent frames and comparing them directly gives
# the front/rear order backwards. Measured instead from base_link through TF:
#
#   _3 x = +0.230   _2 x = -0.067   _1 x = -0.323     (+x forward, confirmed
#   by the forward-facing D435i at x = +0.276)
#
# Both this tuple and DEFAULT_AXIS_SIGNS below are indexed by axis, so an edit
# here that is not mirrored there silently spins one wheel backwards in TF;
# test_wheel_encoder_joint_states.py cross-checks the two against the side
# lists in aries_drive/config/cmd_vel_odrive_bridge.yaml.
AXIS_JOINTS = (
    "L_3_Wheel_Joint",
    "L_2_Wheel_Joint",
    "L_1_Wheel_Joint",
    "R_1_Wheel_Joint",
    "R_2_Wheel_Joint",
    "R_3_Wheel_Joint",
)

# The physical left motors/encoders are mounted opposite to the right side.
# Odom.py corrects this when calculating travel; apply the same convention to
# URDF wheel rotation so forward rover motion animates forward on both sides.
# Indexed by axis, so this follows the mapping above: right axes 5, 4, 3 keep
# +1 and left axes 0, 1, 2 take -1.
DEFAULT_AXIS_SIGNS = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)


def encoder_to_joint(
    position_turns: float,
    velocity_turns_per_second: float,
    zero_turns: float,
    sign: float = 1.0,
) -> tuple[float, float]:
    """Convert an ODrive encoder sample into URDF radians and radians/second."""
    values = (
        float(position_turns),
        float(velocity_turns_per_second),
        float(zero_turns),
        float(sign),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ODrive encoder values and sign must be finite")
    if abs(sign) < 1e-12:
        raise ValueError("encoder sign must be nonzero")
    return (
        sign * (position_turns - zero_turns) * math.tau,
        sign * velocity_turns_per_second * math.tau,
    )


class WheelJointPublisher(Node):
    """Translate each ODrive encoder directly into its physical wheel joint."""

    def __init__(self) -> None:
        super().__init__("wheel_joint_publisher")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("feedback_timeout_s", 0.5)
        self.declare_parameter("axis_signs", list(DEFAULT_AXIS_SIGNS))

        publish_rate = max(
            1.0, float(self.get_parameter("publish_rate_hz").value)
        )
        self.feedback_timeout_s = max(
            0.05, float(self.get_parameter("feedback_timeout_s").value)
        )
        self.axis_signs = tuple(
            float(value) for value in self.get_parameter("axis_signs").value
        )
        if len(self.axis_signs) != len(AXIS_JOINTS):
            raise ValueError(
                f"axis_signs must contain {len(AXIS_JOINTS)} values"
            )
        if not all(math.isfinite(value) and abs(value) > 1e-12
                   for value in self.axis_signs):
            raise ValueError("every axis_signs value must be finite and nonzero")

        self.publisher = self.create_publisher(
            JointState, "/joint_states", 10
        )
        self.encoder_zero = [None] * len(AXIS_JOINTS)
        self.wheel_positions = [0.0] * len(AXIS_JOINTS)
        self.wheel_velocities = [0.0] * len(AXIS_JOINTS)
        self.last_feedback_ns = [None] * len(AXIS_JOINTS)

        feedback_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        for axis in range(len(AXIS_JOINTS)):
            self.create_subscription(
                ControllerStatus,
                f"/odrive_axis{axis}/controller_status",
                lambda msg, axis=axis: self._feedback(msg, axis),
                feedback_qos,
            )

        self.create_timer(1.0 / publish_rate, self.publish_joint_states)
        mapping = ", ".join(
            f"{axis}:{joint}" for axis, joint in enumerate(AXIS_JOINTS)
        )
        self.get_logger().info(
            "ODrive encoder wheel joint publisher ready: " + mapping
        )

    def _feedback(self, msg: ControllerStatus, axis: int) -> None:
        position = float(msg.pos_estimate)
        velocity = float(msg.vel_estimate)
        if not math.isfinite(position) or not math.isfinite(velocity):
            self.get_logger().warn(
                f"Ignoring non-finite encoder sample from axis {axis}",
                throttle_duration_sec=2.0,
            )
            return
        if self.encoder_zero[axis] is None:
            self.encoder_zero[axis] = position
        angle, angular_velocity = encoder_to_joint(
            position,
            velocity,
            self.encoder_zero[axis],
            self.axis_signs[axis],
        )
        self.wheel_positions[axis] = angle
        self.wheel_velocities[axis] = angular_velocity
        self.last_feedback_ns[axis] = self.get_clock().now().nanoseconds

    def publish_joint_states(self) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        timeout_ns = int(self.feedback_timeout_s * 1e9)

        wheel_velocities = self.wheel_velocities.copy()
        for axis, received_ns in enumerate(self.last_feedback_ns):
            if received_ns is None or now_ns - received_ns > timeout_ns:
                wheel_velocities[axis] = 0.0

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(PASSIVE_JOINTS + AXIS_JOINTS)
        msg.position = (
            [0.0] * len(PASSIVE_JOINTS) + self.wheel_positions.copy()
        )
        msg.velocity = (
            [0.0] * len(PASSIVE_JOINTS) + wheel_velocities
        )
        self.publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
