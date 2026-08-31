#!/usr/bin/env python3

import math
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from sensor_msgs.msg import JointState, Joy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class RebelMoveGroupJoystick(Node):
    """Joystick arm control that uses the same MoveGroup action path as RViz."""

    def __init__(self):
        super().__init__("rebel_movegroup_joystick")

        self.declare_parameter("joy_topic", "joy")
        self.declare_parameter("joint_state_topic", "joint_states")
        self.declare_parameter("status_topic", "/arm_joystick/status")
        self.declare_parameter("move_action_name", "move_action")
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("planning_group", "igus_rebel_arm")
        self.declare_parameter("planning_link", "gripper_tcp")
        self.declare_parameter("move_group_control_mode", "cartesian")
        self.declare_parameter("cartesian_step_m", 0.025)
        self.declare_parameter("angular_step_rad", 0.10)
        self.declare_parameter("cartesian_directions", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("cartesian_frame", "tool")
        self.declare_parameter("position_goal_tolerance", 0.015)
        self.declare_parameter("orientation_goal_tolerance", 0.15)

        self.declare_parameter("joint_names", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
        self.declare_parameter(
            "joint_min_positions",
            [-3.1241, -1.4835, -1.39626, -3.12414, -1.65806, -3.12414],
        )
        self.declare_parameter(
            "joint_max_positions",
            [3.1241, 2.4435, 2.61799, 3.12414, 1.65806, 3.12414],
        )
        self.declare_parameter("joint_directions", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("joint_step_rad", 0.07)
        self.declare_parameter("joint_limit_margin", 0.03)
        self.declare_parameter("joint_goal_tolerance", 0.025)
        self.declare_parameter("allowed_planning_time", 2.0)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("velocity_scale", 0.35)
        self.declare_parameter("acceleration_scale", 0.35)
        self.declare_parameter("command_period_sec", 0.20)
        self.declare_parameter("failure_cooldown_sec", 0.80)
        self.declare_parameter("success_cooldown_sec", 0.05)
        self.declare_parameter("status_period_sec", 1.0)

        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("constant_speed_mode", True)

        self.declare_parameter("axis_joint1", 0)
        self.declare_parameter("axis_joint2", 1)
        self.declare_parameter("axis_joint3", 4)
        self.declare_parameter("axis_joint4", 3)
        self.declare_parameter("axis_joint5", 6)
        self.declare_parameter("axis_joint6", 7)
        # The Cartesian axis mapping follows cartesian_frame: a pairing that
        # reads correctly in one frame is wrong in the other, so switching the
        # frame alone would leave a layout nobody chose. "tool" transposes both
        # sticks (horizontal 0/3 -> first axis, vertical 1/4 -> second); "base"
        # is the pre-2026-08-31 layout. Setting any of the six keys in a params
        # file pins that axis in both frames.
        axis_defaults = (
            (0, 1, 7, 4, 3, 6)
            if self._resolve_cartesian_frame() == "tool"
            else (1, 0, 7, 3, 4, 6)
        )
        self.declare_parameter("axis_linear_x", axis_defaults[0])
        self.declare_parameter("axis_linear_y", axis_defaults[1])
        self.declare_parameter("axis_linear_z", axis_defaults[2])
        self.declare_parameter("axis_angular_x", axis_defaults[3])
        self.declare_parameter("axis_angular_y", axis_defaults[4])
        self.declare_parameter("axis_angular_z", axis_defaults[5])

        self.declare_parameter("button_enable", 5)
        self.declare_parameter("button_rover_enable", 4)
        self.declare_parameter("button_gripper_open", 2)
        self.declare_parameter("button_gripper_close", 1)
        self.declare_parameter("button_gripper_toggle", 0)

        self.declare_parameter("gripper_command_topic", "rebel_gripper_controller/joint_trajectory")
        self.declare_parameter("gripper_joint_name", "gripper_gear_left_joint")
        self.declare_parameter("gripper_speed", 2.0)
        self.declare_parameter("gripper_open_position", -1.57)
        self.declare_parameter("gripper_closed_position", 0.07)
        self.declare_parameter("gripper_trajectory_duration", 0.10)
        self.declare_parameter("max_gripper_command_step", 0.04)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.move_action_name = str(self.get_parameter("move_action_name").value)
        self.planning_frame = str(self.get_parameter("planning_frame").value)
        self.planning_group = str(self.get_parameter("planning_group").value)
        self.planning_link = str(self.get_parameter("planning_link").value)
        self.move_group_control_mode = str(self.get_parameter("move_group_control_mode").value).strip().lower()
        if self.move_group_control_mode not in ("cartesian", "joint"):
            self.get_logger().warn(
                f"Unknown move_group_control_mode={self.move_group_control_mode!r}; using cartesian."
            )
            self.move_group_control_mode = "cartesian"
        self.cartesian_step_m = max(0.001, float(self.get_parameter("cartesian_step_m").value))
        self.angular_step_rad = max(0.001, float(self.get_parameter("angular_step_rad").value))
        self.cartesian_directions = self._float_list("cartesian_directions", 6)
        # Frame the stick's Cartesian nudge is read in.
        #   "tool" - planning_link's own axes (gripper_tcp: +Z out of the jaws
        #            along the approach), so a nudge means the same thing to
        #            the operator whatever the wrist is doing.
        #   "base" - planning_frame (base_link), the old behaviour.
        # The goal sent to /move_action is still a planning_frame pose either
        # way; only the direction the step is taken in changes.
        raw_frame = str(self.get_parameter("cartesian_frame").value).strip().lower()
        self.cartesian_frame = self._resolve_cartesian_frame()
        if raw_frame != self.cartesian_frame:
            self.get_logger().warn(f"Unknown cartesian_frame={raw_frame!r}; using tool.")
        self.position_goal_tolerance = max(0.001, float(self.get_parameter("position_goal_tolerance").value))
        self.orientation_goal_tolerance = max(0.001, float(self.get_parameter("orientation_goal_tolerance").value))

        self.joint_names = self._string_list("joint_names")
        self.joint_min_positions = self._float_list("joint_min_positions", len(self.joint_names))
        self.joint_max_positions = self._float_list("joint_max_positions", len(self.joint_names))
        self.joint_directions = self._float_list("joint_directions", len(self.joint_names))
        self.joint_step_rad = max(0.001, float(self.get_parameter("joint_step_rad").value))
        self.joint_limit_margin = max(0.0, float(self.get_parameter("joint_limit_margin").value))
        self.joint_goal_tolerance = max(0.001, float(self.get_parameter("joint_goal_tolerance").value))
        self.allowed_planning_time = max(0.1, float(self.get_parameter("allowed_planning_time").value))
        self.num_planning_attempts = max(1, int(self.get_parameter("num_planning_attempts").value))
        self.velocity_scale = self._clamp01(float(self.get_parameter("velocity_scale").value))
        self.acceleration_scale = self._clamp01(float(self.get_parameter("acceleration_scale").value))
        self.command_period_sec = max(0.05, float(self.get_parameter("command_period_sec").value))
        self.failure_cooldown_sec = max(0.0, float(self.get_parameter("failure_cooldown_sec").value))
        self.success_cooldown_sec = max(0.0, float(self.get_parameter("success_cooldown_sec").value))
        self.status_period_sec = max(0.25, float(self.get_parameter("status_period_sec").value))

        self.deadzone = min(1.0, max(0.0, float(self.get_parameter("deadzone").value)))
        self.constant_speed_mode = bool(self.get_parameter("constant_speed_mode").value)

        self.axis_joint = [
            int(self.get_parameter("axis_joint1").value),
            int(self.get_parameter("axis_joint2").value),
            int(self.get_parameter("axis_joint3").value),
            int(self.get_parameter("axis_joint4").value),
            int(self.get_parameter("axis_joint5").value),
            int(self.get_parameter("axis_joint6").value),
        ]
        self.axis_linear = [
            int(self.get_parameter("axis_linear_x").value),
            int(self.get_parameter("axis_linear_y").value),
            int(self.get_parameter("axis_linear_z").value),
        ]
        self.axis_angular = [
            int(self.get_parameter("axis_angular_x").value),
            int(self.get_parameter("axis_angular_y").value),
            int(self.get_parameter("axis_angular_z").value),
        ]

        self.button_enable = int(self.get_parameter("button_enable").value)
        self.button_rover_enable = int(self.get_parameter("button_rover_enable").value)
        self.button_gripper_open = int(self.get_parameter("button_gripper_open").value)
        self.button_gripper_close = int(self.get_parameter("button_gripper_close").value)
        self.button_gripper_toggle = int(self.get_parameter("button_gripper_toggle").value)

        self.gripper_command_topic = str(self.get_parameter("gripper_command_topic").value)
        self.gripper_joint_name = str(self.get_parameter("gripper_joint_name").value)
        self.gripper_speed = max(0.0, float(self.get_parameter("gripper_speed").value))
        self.gripper_open_position = float(self.get_parameter("gripper_open_position").value)
        self.gripper_closed_position = float(self.get_parameter("gripper_closed_position").value)
        self.gripper_trajectory_duration = max(0.05, float(self.get_parameter("gripper_trajectory_duration").value))
        self.max_gripper_command_step = max(0.001, float(self.get_parameter("max_gripper_command_step").value))
        if self.gripper_open_position > self.gripper_closed_position:
            self.gripper_open_position, self.gripper_closed_position = (
                self.gripper_closed_position,
                self.gripper_open_position,
            )

        self.current_joint_positions: Dict[str, float] = {}
        self.last_joy: Optional[Joy] = None
        self.motion_active = False
        self.next_goal_time = 0.0
        self.status_text = self._idle_status()
        self.last_status_log = 0.0

        self.commanded_gripper_position = self.gripper_open_position
        self.current_gripper_position = self.gripper_open_position
        self.desired_gripper_position = self.gripper_open_position
        self.have_gripper_position = False
        self.previous_gripper_toggle_pressed = False
        self.last_gripper_update = time.monotonic()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.move_group_client = ActionClient(self, MoveGroup, self.move_action_name)
        self.gripper_pub = self.create_publisher(JointTrajectory, self.gripper_command_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_subscription(Joy, self.joy_topic, self._joy_callback, 10)
        self.create_subscription(JointState, self.joint_state_topic, self._joint_state_callback, 10)
        self.create_timer(self.command_period_sec, self._command_timer)
        self.create_timer(self.status_period_sec, self._status_timer)

        self._publish_status(self.status_text)
        self.get_logger().info(
            "MoveGroup joystick ready in %s mode. Hold RB/button %d to move arm; LB/button %d blocks arm output."
            % (self.move_group_control_mode, self.button_enable, self.button_rover_enable)
        )

    def _string_list(self, name: str) -> List[str]:
        value = self.get_parameter(name).value
        return [str(item) for item in value]

    def _float_list(self, name: str, expected_len: int) -> List[float]:
        value = [float(item) for item in self.get_parameter(name).value]
        if len(value) != expected_len:
            raise ValueError(f"{name} must have {expected_len} entries, got {len(value)}")
        return value

    def _resolve_cartesian_frame(self) -> str:
        """cartesian_frame normalized to "tool"/"base"; anything else is "tool".

        Read before the main parameter block because the Cartesian axis
        defaults depend on it.
        """
        frame = str(self.get_parameter("cartesian_frame").value).strip().lower()
        return frame if frame in ("tool", "base") else "tool"

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(1.0, max(0.0, value))

    def _idle_status(self) -> str:
        if self.move_group_control_mode == "cartesian":
            frame = "gripper axes" if self.cartesian_frame == "tool" else "rover axes"
            return f"ARM MODE: MoveGroup Cartesian ({frame}). Hold RB to move tool XYZ/rotation."
        return "ARM MODE: MoveGroup planned joints. Hold RB to move arm."

    def _button_pressed(self, msg: Joy, button: int) -> bool:
        if button < 0 or button >= 999:
            return False
        if button >= len(msg.buttons):
            self.get_logger().warn(
                f"Joystick button {button} is missing. Joy has {len(msg.buttons)} buttons.",
                throttle_duration_sec=5.0,
            )
            return False
        return msg.buttons[button] != 0

    def _axis_value(self, msg: Joy, axis: int) -> float:
        if axis < 0:
            return 0.0
        if axis >= len(msg.axes):
            self.get_logger().warn(
                f"Joystick axis {axis} is missing. Joy has {len(msg.axes)} axes.",
                throttle_duration_sec=5.0,
            )
            return 0.0
        raw = float(msg.axes[axis])
        if abs(raw) < self.deadzone:
            return 0.0
        sign = 1.0 if raw >= 0.0 else -1.0
        if self.constant_speed_mode:
            return sign
        return sign * (abs(raw) - self.deadzone) / max(1e-6, 1.0 - self.deadzone)

    def _publish_status(self, text: str):
        self.status_text = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _publish_status_throttled(self, text: str, period_sec: float = 1.0):
        now = time.monotonic()
        if text != self.status_text or now - self.last_status_log >= period_sec:
            self.last_status_log = now
            self._publish_status(text)

    def _status_timer(self):
        self._publish_status(self.status_text)

    def _joy_callback(self, msg: Joy):
        self.last_joy = msg

        rover_enabled = self._button_pressed(msg, self.button_rover_enable)
        arm_enabled = self._button_pressed(msg, self.button_enable) and not rover_enabled

        if rover_enabled:
            self.previous_gripper_toggle_pressed = False
            self._publish_status_throttled("LB rover drive active: arm/gripper output blocked")
            return

        if arm_enabled:
            self._update_gripper(msg)
        else:
            self.previous_gripper_toggle_pressed = False

    def _joint_state_callback(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            self.current_joint_positions[str(name)] = float(position)
            if name != self.gripper_joint_name:
                continue
            self.current_gripper_position = self._clamp_gripper(float(position))
            if not self.have_gripper_position:
                self.commanded_gripper_position = self.current_gripper_position
                self.desired_gripper_position = self.current_gripper_position
                self.have_gripper_position = True

    def _command_timer(self):
        if self.motion_active or time.monotonic() < self.next_goal_time:
            return

        msg = self.last_joy
        if msg is None:
            self._publish_status_throttled("MoveGroup joystick waiting for /joy")
            return

        rover_enabled = self._button_pressed(msg, self.button_rover_enable)
        arm_enabled = self._button_pressed(msg, self.button_enable) and not rover_enabled
        if not arm_enabled:
            self._publish_status_throttled(self._idle_status())
            return

        if not self.move_group_client.server_is_ready():
            self._publish_status_throttled("MoveGroup joystick waiting for /move_action")
            return

        missing = [name for name in self.joint_names if name not in self.current_joint_positions]
        if missing:
            self._publish_status_throttled("MoveGroup joystick waiting for joint_states: " + ", ".join(missing))
            return

        if self.move_group_control_mode == "cartesian":
            self._command_cartesian(msg)
        else:
            self._command_joint(msg)

    def _command_joint(self, msg: Joy):
        axis_values = [self._axis_value(msg, axis) for axis in self.axis_joint]
        if all(abs(value) <= 0.0 for value in axis_values):
            self._publish_status_throttled("ARM MODE: MoveGroup planned joints. Hold RB and move sticks.")
            return

        targets: List[float] = []
        moved = False
        clipped = False
        for i, name in enumerate(self.joint_names):
            current = self.current_joint_positions[name]
            delta = axis_values[i] * self.joint_directions[i] * self.joint_step_rad
            low = self.joint_min_positions[i] + self.joint_limit_margin
            high = self.joint_max_positions[i] - self.joint_limit_margin
            target = min(high, max(low, current + delta))
            if abs(axis_values[i]) > 0.0 and abs(target - (current + delta)) > 1e-6:
                clipped = True
            if abs(target - current) > 1e-4:
                moved = True
            targets.append(target)

        if not moved:
            self._publish_status_throttled("MoveGroup joystick at configured joint limit", period_sec=0.5)
            self.next_goal_time = time.monotonic() + self.failure_cooldown_sec
            return

        self._send_joint_goal(targets, clipped)

    def _command_cartesian(self, msg: Joy):
        linear_values = [self._axis_value(msg, axis) for axis in self.axis_linear]
        angular_values = [self._axis_value(msg, axis) for axis in self.axis_angular]
        if all(abs(value) <= 0.0 for value in linear_values + angular_values):
            self._publish_status_throttled("ARM MODE: MoveGroup Cartesian. Hold RB and move sticks.")
            return

        current_pose = self._current_tool_pose()
        if current_pose is None:
            self._publish_status_throttled(
                f"MoveGroup Cartesian waiting for TF {self.planning_frame}->{self.planning_link}"
            )
            return

        step = [
            linear_values[i] * self.cartesian_directions[i] * self.cartesian_step_m
            for i in range(3)
        ]
        roll = angular_values[0] * self.cartesian_directions[3] * self.angular_step_rad
        pitch = angular_values[1] * self.cartesian_directions[4] * self.angular_step_rad
        yaw = angular_values[2] * self.cartesian_directions[5] * self.angular_step_rad
        delta_q = self._rpy_to_quaternion(roll, pitch, yaw)

        if self.cartesian_frame == "tool":
            # Read the stick in the tool's own axes: rotate the translation
            # step by the current TCP orientation, and post-multiply the
            # rotation so it is applied about the tool axes rather than the
            # planning frame's. Pre-multiplying (the "base" branch) spins the
            # tool about base_link's X/Y/Z, which is what made a wrist roll
            # feel like it came from somewhere else once the arm was turned.
            step = self._rotate_vector(current_pose.orientation, step)
            new_orientation = self._multiply_quaternions(current_pose.orientation, delta_q)
        else:
            new_orientation = self._multiply_quaternions(delta_q, current_pose.orientation)

        target_pose = Pose()
        target_pose.position = Point(
            x=current_pose.position.x + step[0],
            y=current_pose.position.y + step[1],
            z=current_pose.position.z + step[2],
        )
        target_pose.orientation = self._normalize_quaternion(new_orientation)

        self._send_pose_goal(target_pose)

    def _current_tool_pose(self) -> Optional[Pose]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.planning_link,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"Tool pose TF unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        pose = Pose()
        pose.position = Point(
            x=float(transform.transform.translation.x),
            y=float(transform.transform.translation.y),
            z=float(transform.transform.translation.z),
        )
        pose.orientation = transform.transform.rotation
        return pose

    @staticmethod
    def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        )

    @staticmethod
    def _rotate_vector(q: Quaternion, v: List[float]) -> List[float]:
        """Rotate v by q (v' = q v q*), written out to avoid a numpy dependency."""
        u = (q.x, q.y, q.z)
        uv = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        uuv = (
            u[1] * uv[2] - u[2] * uv[1],
            u[2] * uv[0] - u[0] * uv[2],
            u[0] * uv[1] - u[1] * uv[0],
        )
        return [v[i] + 2.0 * (q.w * uv[i] + uuv[i]) for i in range(3)]

    @staticmethod
    def _multiply_quaternions(a: Quaternion, b: Quaternion) -> Quaternion:
        return Quaternion(
            x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        )

    @staticmethod
    def _normalize_quaternion(q: Quaternion) -> Quaternion:
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-9:
            return Quaternion(w=1.0)
        return Quaternion(x=q.x / norm, y=q.y / norm, z=q.z / norm, w=q.w / norm)

    def _send_pose_goal(self, target_pose: Pose):
        constraints = Constraints()

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.planning_frame
        position_constraint.header.stamp = self.get_clock().now().to_msg()
        position_constraint.link_name = self.planning_link
        region = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.position_goal_tolerance]
        region.primitives.append(sphere)
        region.primitive_poses.append(target_pose)
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0
        constraints.position_constraints.append(position_constraint)

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.planning_frame
        orientation_constraint.header.stamp = position_constraint.header.stamp
        orientation_constraint.link_name = self.planning_link
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = self.orientation_goal_tolerance
        orientation_constraint.absolute_y_axis_tolerance = self.orientation_goal_tolerance
        orientation_constraint.absolute_z_axis_tolerance = self.orientation_goal_tolerance
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)

        self._send_constraints_goal(constraints, "MoveGroup joystick planning Cartesian step")

    def _send_joint_goal(self, joint_targets: List[float], clipped: bool):
        constraints = Constraints()
        for name, target in zip(self.joint_names, joint_targets):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = float(target)
            joint_constraint.tolerance_above = self.joint_goal_tolerance
            joint_constraint.tolerance_below = self.joint_goal_tolerance
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)

        suffix = " (clipped at limit)" if clipped else ""
        self._send_constraints_goal(constraints, "MoveGroup joystick planning joint step" + suffix)

    def _send_constraints_goal(self, constraints: Constraints, status: str):
        goal = MoveGroup.Goal()
        goal.request.workspace_parameters.header.frame_id = self.planning_frame
        goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        goal.request.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-1.0)
        goal.request.workspace_parameters.max_corner = Vector3(x=2.0, y=2.0, z=2.0)

        seed_state = RobotState()
        seed_state.is_diff = True
        seed_joint_state = JointState()
        seed_joint_state.name = list(self.current_joint_positions.keys())
        seed_joint_state.position = [self.current_joint_positions[name] for name in seed_joint_state.name]
        seed_state.joint_state = seed_joint_state
        goal.request.start_state = seed_state

        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = self.num_planning_attempts
        goal.request.allowed_planning_time = self.allowed_planning_time
        goal.request.max_velocity_scaling_factor = self.velocity_scale
        goal.request.max_acceleration_scaling_factor = self.acceleration_scale
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.motion_active = True
        self._publish_status(status)
        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.motion_active = False
            self.next_goal_time = time.monotonic() + self.failure_cooldown_sec
            self._publish_status(f"MoveGroup joystick goal send failed: {exc}")
            return

        if not goal_handle.accepted:
            self.motion_active = False
            self.next_goal_time = time.monotonic() + self.failure_cooldown_sec
            self._publish_status("MoveGroup joystick goal rejected")
            return

        goal_handle.get_result_async().add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        self.motion_active = False
        try:
            wrapped_result = future.result()
        except Exception as exc:
            self.next_goal_time = time.monotonic() + self.failure_cooldown_sec
            self._publish_status(f"MoveGroup joystick result failed: {exc}")
            return

        error_code = wrapped_result.result.error_code.val
        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED and error_code == MoveItErrorCodes.SUCCESS:
            self.next_goal_time = time.monotonic() + self.success_cooldown_sec
            self._publish_status("MoveGroup joystick step complete")
            return

        self.next_goal_time = time.monotonic() + self.failure_cooldown_sec
        self._publish_status(
            f"MoveGroup joystick step blocked: status={wrapped_result.status} error={error_code}"
        )

    def _update_gripper(self, msg: Joy):
        open_pressed = self._button_pressed(msg, self.button_gripper_open)
        close_pressed = self._button_pressed(msg, self.button_gripper_close)
        toggle_pressed = self._button_pressed(msg, self.button_gripper_toggle)

        now = time.monotonic()
        dt = now - self.last_gripper_update
        if dt <= 0.0 or dt > 0.5:
            dt = 0.02
        self.last_gripper_update = now

        max_step = min(self.max_gripper_command_step, max(0.001, self.gripper_speed * dt))

        if open_pressed != close_pressed:
            direction = 1.0 if close_pressed else -1.0
            next_position = self._clamp_gripper(self.commanded_gripper_position + direction * max_step)
            if abs(next_position - self.commanded_gripper_position) >= 1e-4:
                self._publish_gripper(next_position)
            self.previous_gripper_toggle_pressed = toggle_pressed
            return

        if toggle_pressed and not self.previous_gripper_toggle_pressed:
            midpoint = 0.5 * (self.gripper_open_position + self.gripper_closed_position)
            reference = self.current_gripper_position if self.have_gripper_position else self.commanded_gripper_position
            target = self.gripper_closed_position if reference <= midpoint else self.gripper_open_position
            distance = abs(target - self.commanded_gripper_position)
            duration = max(self.gripper_trajectory_duration, distance / max(0.001, self.gripper_speed))
            self._publish_gripper(target, duration)

        self.previous_gripper_toggle_pressed = toggle_pressed

    def _publish_gripper(self, position: float, duration: Optional[float] = None):
        self.commanded_gripper_position = self._clamp_gripper(position)
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = [self.gripper_joint_name]
        point = JointTrajectoryPoint()
        point.positions = [self.commanded_gripper_position]
        seconds = self.gripper_trajectory_duration if duration is None else max(0.05, duration)
        point.time_from_start.sec = int(math.floor(seconds))
        point.time_from_start.nanosec = int(round((seconds - point.time_from_start.sec) * 1e9))
        if point.time_from_start.nanosec >= 1000000000:
            point.time_from_start.sec += 1
            point.time_from_start.nanosec -= 1000000000
        trajectory.points.append(point)
        self.gripper_pub.publish(trajectory)

    def _clamp_gripper(self, position: float) -> float:
        return min(self.gripper_closed_position, max(self.gripper_open_position, position))


def main(args=None):
    rclpy.init(args=args)
    node = RebelMoveGroupJoystick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
