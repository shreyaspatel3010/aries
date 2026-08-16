#!/usr/bin/env python3
"""Send the arm to a named joint preset from a gamepad combo.

Default map: LT + Y -> pick_home, LT + A -> probe_drop, LT + B -> soil_drop.

Runs alongside the teleop node rather than inside it. The arm already has a
second /joy consumer for a modifier combo -- rebel_hand_guiding.py owns RB + Y --
and keeping the preset move out of the teleop hot path means a planning stall
can never delay the velocity stream that is actually driving the arm.

MoveIt still performs planning, joint-limit and collision checking. Execution
goes directly to the arm FollowJointTrajectory action after planning. This
avoids MoveIt's global trajectory manager, which can remain permanently busy
when another client submits an overlapping plan-and-execute request.

Only arm joints are commanded. The gripper is deliberately left alone so the
preset can be used while holding something.
"""

import time
from typing import List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_action_status_default

from action_msgs.msg import GoalStatus, GoalStatusArray
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Vector3
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from sensor_msgs.msg import Joy
from std_msgs.msg import String


class ArmPresetPoseJoystick(Node):

    def __init__(self):
        super().__init__("arm_preset_pose_joystick")

        self.declare_parameter("joy_topic", "joy")
        self.declare_parameter("preset_status_topic", "/arm_preset_pose/status")
        self.declare_parameter("move_action_name", "move_action")
        self.declare_parameter(
            "arm_controller_action_name",
            "/rebel_arm_trajectory_controller/follow_joint_trajectory",
        )
        self.declare_parameter(
            "arm_controller_status_topic",
            "/rebel_arm_trajectory_controller/follow_joint_trajectory/_action/status",
        )
        self.declare_parameter("planning_group", "igus_rebel_arm")
        self.declare_parameter("planning_frame", "base_link")

        # LT is canonical axis 2 in every layout joy_layout_normalizer emits,
        # scaled 0.0 released -> 1.0 fully pressed, so this is a press fraction.
        # Release sits below press for hysteresis, matching the RT gate in
        # gamepad.yaml.
        self.declare_parameter("preset_modifier_axis", 2)
        self.declare_parameter("preset_modifier_threshold", 0.5)
        self.declare_parameter("preset_modifier_release", 0.35)

        # Combos that must NOT also fire the preset. RB and RT are the two arm
        # enables: if either is held the operator is driving the arm by hand and
        # a planned move would fight the velocity stream for the same controller.
        # LB hands the sticks to the rover. RB + Y is hand guiding, so requiring
        # RB to be released is also what keeps this combo distinct from that one.
        self.declare_parameter("button_enable", 5)
        self.declare_parameter("button_rover_enable", 4)
        self.declare_parameter("axis_joint_enable", 5)
        self.declare_parameter("axis_joint_enable_threshold", 0.5)

        # One entry per combo, index-aligned: preset_buttons[i] with the modifier
        # held sends the arm to preset_pose_names[i]. Positions are flattened
        # len(preset_pose_names) x len(preset_joint_names) because ROS 2
        # parameters cannot nest arrays -- the same shape vision_grasp_node uses
        # for pick_home_alternative_joint_positions_flat.
        self.declare_parameter("preset_buttons", [3, 0, 1])
        self.declare_parameter("preset_pose_names", ["pick_home", "probe_drop", "soil_drop"])
        self.declare_parameter(
            "preset_joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        )
        # Every row matches the SRDF group state of the same name, and pick_home
        # additionally matches vision_grasp_node's pick_home_joint_positions.
        # Keep all of them in step.
        self.declare_parameter(
            "preset_joint_positions_flat",
            [
                0.0, 0.366519, 1.18682, 0.0349066, 1.55334, 1.50098,
                1.81514, -1.18682, 1.97222, 0.0349066, 1.51844, 1.58825,
                1.29154, -0.331613, 1.8326, -0.0698132, 1.53589, 2.84489,
            ],
        )

        # Deliberately NOT the bare allowed_planning_time/num_planning_attempts/
        # joint_goal_tolerance names: gamepad.yaml sets those for the MoveGroup
        # joystick's 0.07 rad nudges, and its 2 s budget is not enough to plan a
        # traverse across the workspace. The /** wildcard would hand them over.
        self.declare_parameter("preset_velocity_scale", 0.25)
        self.declare_parameter("preset_acceleration_scale", 0.25)
        self.declare_parameter("preset_allowed_planning_time", 5.0)
        self.declare_parameter("preset_num_planning_attempts", 5)
        self.declare_parameter("preset_joint_goal_tolerance", 0.01)
        self.declare_parameter("joy_timeout_sec", 0.35)
        self.declare_parameter("action_wait_sec", 10.0)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.move_action_name = str(self.get_parameter("move_action_name").value)
        self.arm_controller_action_name = str(
            self.get_parameter("arm_controller_action_name").value
        )
        self.planning_group = str(self.get_parameter("planning_group").value)
        self.planning_frame = str(self.get_parameter("planning_frame").value)

        self.modifier_axis = int(self.get_parameter("preset_modifier_axis").value)
        self.modifier_threshold = float(self.get_parameter("preset_modifier_threshold").value)
        self.modifier_release = float(self.get_parameter("preset_modifier_release").value)

        self.button_enable = int(self.get_parameter("button_enable").value)
        self.button_rover_enable = int(self.get_parameter("button_rover_enable").value)
        self.axis_joint_enable = int(self.get_parameter("axis_joint_enable").value)
        self.joint_enable_threshold = float(
            self.get_parameter("axis_joint_enable_threshold").value
        )

        self.joint_names = [str(n) for n in self.get_parameter("preset_joint_names").value]
        buttons = [int(b) for b in self.get_parameter("preset_buttons").value]
        pose_names = [str(n) for n in self.get_parameter("preset_pose_names").value]
        flat = [float(v) for v in self.get_parameter("preset_joint_positions_flat").value]

        self.velocity_scale = self._clamp01(float(self.get_parameter("preset_velocity_scale").value))
        self.acceleration_scale = self._clamp01(
            float(self.get_parameter("preset_acceleration_scale").value)
        )
        self.allowed_planning_time = max(
            0.1, float(self.get_parameter("preset_allowed_planning_time").value)
        )
        self.num_planning_attempts = max(
            1, int(self.get_parameter("preset_num_planning_attempts").value)
        )
        self.joint_goal_tolerance = max(
            0.001, float(self.get_parameter("preset_joint_goal_tolerance").value)
        )
        self.joy_timeout_sec = max(0.0, float(self.get_parameter("joy_timeout_sec").value))
        self.action_wait_sec = max(0.0, float(self.get_parameter("action_wait_sec").value))

        # Fail loudly at startup rather than binding a combo to a short row of
        # joint values, which would plan a pose nobody asked for.
        width = len(self.joint_names)
        if len(buttons) != len(pose_names):
            raise ValueError(
                "preset_buttons has {} entries but preset_pose_names has {}".format(
                    len(buttons), len(pose_names)
                )
            )
        if len(flat) != len(pose_names) * width:
            raise ValueError(
                "preset_joint_positions_flat has {} values, expected {} "
                "({} poses x {} joints)".format(
                    len(flat), len(pose_names) * width, len(pose_names), width
                )
            )
        duplicates = {b for b in buttons if buttons.count(b) > 1}
        if duplicates:
            raise ValueError(
                "preset_buttons repeats button(s) {}: one button cannot select "
                "two poses".format(sorted(duplicates))
            )

        # button -> (pose name, joint positions)
        self.presets = {
            button: (name, flat[i * width:(i + 1) * width])
            for i, (button, name) in enumerate(zip(buttons, pose_names))
        }

        self.modifier_held = False
        self.buttons_were_pressed = {button: False for button in self.presets}
        self.motion_active = False
        self.controller_active = False
        self.active_pose_name = ""
        self.last_joy_time = 0.0

        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("preset_status_topic").value), 10
        )
        self.move_group_client = ActionClient(self, MoveGroup, self.move_action_name)
        self.arm_controller_client = ActionClient(
            self, FollowJointTrajectory, self.arm_controller_action_name
        )
        self.controller_status_sub = self.create_subscription(
            GoalStatusArray,
            str(self.get_parameter("arm_controller_status_topic").value),
            self._controller_status_callback,
            qos_profile_action_status_default,
        )
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self._joy_callback, 10)

        combos = ", ".join(
            "button %d -> '%s'" % (button, name)
            for button, (name, _) in sorted(self.presets.items())
        )
        self.get_logger().info(
            "Arm presets ready: hold LT/axis %d, then %s. "
            "Ignored while RB/button %d, RT/axis %d or LB/button %d is held."
            % (
                self.modifier_axis,
                combos,
                self.button_enable,
                self.axis_joint_enable,
                self.button_rover_enable,
            )
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(1.0, max(0.0, value))

    def _button_pressed(self, msg: Joy, button: int) -> bool:
        return 0 <= button < len(msg.buttons) and bool(msg.buttons[button])

    def _axis_value(self, msg: Joy, axis: int) -> float:
        if not 0 <= axis < len(msg.axes):
            return 0.0
        return float(msg.axes[axis])

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))

    def _controller_status_callback(self, msg: GoalStatusArray):
        active = {
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
            GoalStatus.STATUS_CANCELING,
        }
        self.controller_active = any(
            status.status in active for status in msg.status_list
        )

    def _joy_callback(self, msg: Joy):
        now = time.monotonic()
        # A stale link then a fresh message must not read as a press: drop the
        # edge state so a button held at disconnect cannot fire on reconnect.
        if self.joy_timeout_sec > 0.0 and self.last_joy_time > 0.0:
            if now - self.last_joy_time > self.joy_timeout_sec:
                self.modifier_held = False
                self.buttons_were_pressed = {b: False for b in self.presets}
        self.last_joy_time = now

        modifier_value = self._axis_value(msg, self.modifier_axis)
        if self.modifier_held:
            self.modifier_held = modifier_value > self.modifier_release
        else:
            self.modifier_held = modifier_value > self.modifier_threshold

        # Edge state is tracked for every mapped button on every message, not
        # just the one that fired, so releasing Y while LT is still held cannot
        # leave A looking like a fresh press.
        rising = []
        for button in self.presets:
            pressed = self._button_pressed(msg, button)
            if pressed and not self.buttons_were_pressed[button]:
                rising.append(button)
            self.buttons_were_pressed[button] = pressed

        if not (rising and self.modifier_held):
            return
        if len(rising) > 1:
            self._reject(
                "?",
                "two preset buttons went down together ({})".format(
                    ", ".join(str(b) for b in sorted(rising))
                ),
            )
            return

        button = rising[0]
        pose_name, joint_positions = self.presets[button]

        if self._button_pressed(msg, self.button_enable):
            self._reject(pose_name, "RB is held (that combo is hand guiding)")
            return
        if self._button_pressed(msg, self.button_rover_enable):
            self._reject(pose_name, "LB is held (rover has the sticks)")
            return
        if self._axis_value(msg, self.axis_joint_enable) > self.joint_enable_threshold:
            self._reject(pose_name, "RT is held (arm is under joint jog)")
            return
        if self.motion_active:
            self._reject(pose_name, "a preset move is already running")
            return
        if self.controller_active:
            self._reject(pose_name, "the arm controller is executing another goal")
            return

        self._send_preset_goal(pose_name, joint_positions)

    def _reject(self, pose_name: str, why: str):
        text = f"Preset '{pose_name}' ignored: {why}"
        self.get_logger().warn(text)
        self._publish_status(text)

    def _send_preset_goal(self, pose_name: str, joint_positions: List[float]):
        if not self.move_group_client.wait_for_server(timeout_sec=self.action_wait_sec):
            self._reject(pose_name, f"{self.move_action_name} is not available")
            return

        constraints = Constraints()
        for name, target in zip(self.joint_names, joint_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = float(target)
            joint_constraint.tolerance_above = self.joint_goal_tolerance
            joint_constraint.tolerance_below = self.joint_goal_tolerance
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)

        goal = MoveGroup.Goal()
        goal.request.workspace_parameters.header.frame_id = self.planning_frame
        goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        goal.request.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-1.0)
        goal.request.workspace_parameters.max_corner = Vector3(x=2.0, y=2.0, z=2.0)

        # Empty diff start state: plan from whatever the planning scene monitor
        # currently believes, rather than from a copy of /joint_states that may
        # be a cycle or two behind.
        start_state = RobotState()
        start_state.is_diff = True
        goal.request.start_state = start_state

        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = self.num_planning_attempts
        goal.request.allowed_planning_time = self.allowed_planning_time
        goal.request.max_velocity_scaling_factor = self.velocity_scale
        goal.request.max_acceleration_scaling_factor = self.acceleration_scale
        goal.request.goal_constraints = [constraints]
        # Preserve MoveIt's collision-aware plan but bypass its global
        # trajectory execution manager. Overlapping autonomous and preset
        # requests can leave that manager permanently marked busy.
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.motion_active = True
        # Which pose is in flight, for the async callbacks to report against.
        self.active_pose_name = pose_name
        text = f"Preset '{pose_name}': planning"
        self.get_logger().info(text)
        self._publish_status(text)

        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001 - report and re-arm, never wedge
            self.motion_active = False
            self._reject(self.active_pose_name, f"goal was not sent ({exc})")
            return

        if not goal_handle.accepted:
            self.motion_active = False
            self._reject(self.active_pose_name, "MoveGroup rejected the goal")
            return

        goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        try:
            wrapped = future.result()
        except Exception as exc:  # noqa: BLE001
            self.motion_active = False
            self._reject(self.active_pose_name, f"no result ({exc})")
            return

        if (wrapped.status != GoalStatus.STATUS_SUCCEEDED or
                wrapped.result.error_code.val != 1):
            self.motion_active = False
            self._reject(
                self.active_pose_name,
                "planning failed (status {}, MoveIt error {})".format(
                    wrapped.status, wrapped.result.error_code.val
                ),
            )
            return

        trajectory = wrapped.result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            self.motion_active = False
            self._reject(self.active_pose_name, "MoveIt returned an empty trajectory")
            return
        if self.controller_active:
            self.motion_active = False
            self._reject(
                self.active_pose_name,
                "the arm controller became busy while the preset was planning",
            )
            return
        if not self.arm_controller_client.wait_for_server(
                timeout_sec=self.action_wait_sec):
            self.motion_active = False
            self._reject(
                self.active_pose_name,
                f"{self.arm_controller_action_name} is not available",
            )
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        # Zero timestamp requests immediate execution; a planning timestamp can
        # be stale by the time the controller receives the trajectory.
        goal.trajectory.header.stamp.sec = 0
        goal.trajectory.header.stamp.nanosec = 0
        text = (
            f"Preset '{self.active_pose_name}': executing "
            f"{len(trajectory.points)} planned points"
        )
        self.get_logger().info(text)
        self._publish_status(text)
        self.arm_controller_client.send_goal_async(goal).add_done_callback(
            self._controller_goal_response_callback
        )

    def _controller_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.motion_active = False
            self._reject(self.active_pose_name, f"controller goal was not sent ({exc})")
            return
        if not goal_handle.accepted:
            self.motion_active = False
            self._reject(self.active_pose_name, "arm controller rejected the trajectory")
            return
        goal_handle.get_result_async().add_done_callback(
            self._controller_result_callback
        )

    def _controller_result_callback(self, future):
        self.motion_active = False
        try:
            wrapped = future.result()
        except Exception as exc:  # noqa: BLE001
            self._reject(self.active_pose_name, f"no controller result ({exc})")
            return

        result = wrapped.result
        if (wrapped.status == GoalStatus.STATUS_SUCCEEDED and
                result.error_code == FollowJointTrajectory.Result.SUCCESSFUL):
            text = f"Preset '{self.active_pose_name}': reached"
            self.get_logger().info(text)
            self._publish_status(text)
            return

        detail = result.error_string or "no controller detail"
        self._reject(
            self.active_pose_name,
            "controller failed (status {}, error {}: {})".format(
                wrapped.status, result.error_code, detail
            ),
        )


def main(args: Optional[List[str]] = None):
    rclpy.init(args=args)
    node = ArmPresetPoseJoystick()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
