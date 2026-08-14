#!/usr/bin/env python3
"""Keep the rover body level by commanding its virtual differential."""

import math
from typing import Dict, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, JointState

from std_msgs.msg import Float64


def quat_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return pitch from an orientation quaternion."""
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


class VirtualDifferential(Node):
    """
    Keep the body level by adjusting the average rocker angle.

    - diff_meas = (L - R)/2 is preserved (terrain decides)
    - avg_cmd   = avg_meas - Kp*(pitch - target) - Kd*(pitch_rate)
    - L_cmd = avg_cmd + diff_meas
      R_cmd = avg_cmd - diff_meas
    Publish to /aries/*_Rocker_Joint/cmd_pos (std_msgs/Float64).
    """

    def __init__(self):
        super().__init__('virtual_differential')

        # ---- parameters
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('left_joint_name', 'L_Rocker_Joint')
        self.declare_parameter('right_joint_name', 'R_Rocker_Joint')
        self.declare_parameter('left_cmd_topic', '/aries/L_Rocker_Joint/cmd_pos')
        self.declare_parameter('right_cmd_topic', '/aries/R_Rocker_Joint/cmd_pos')

        self.declare_parameter('target_pitch_deg', 0.0)
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('cmd_rate_hz', 50.0)
        self.declare_parameter('cmd_limit_deg', 45.0)
        self.declare_parameter('avg_rate_limit_deg_s', 60.0)
        self.declare_parameter('invert_pitch', False)

        # ---- resolve params
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        js_topic = self.get_parameter('joint_states_topic').get_parameter_value().string_value
        self.left_name = self.get_parameter('left_joint_name').get_parameter_value().string_value
        self.right_name = self.get_parameter('right_joint_name').get_parameter_value().string_value
        left_cmd_topic = self.get_parameter('left_cmd_topic').get_parameter_value().string_value
        right_cmd_topic = self.get_parameter('right_cmd_topic').get_parameter_value().string_value

        self.target_pitch = math.radians(self.get_parameter('target_pitch_deg').value)
        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.rate_hz = float(self.get_parameter('cmd_rate_hz').value)
        self.cmd_limit = math.radians(float(self.get_parameter('cmd_limit_deg').value))
        self.avg_rate_limit = math.radians(float(self.get_parameter('avg_rate_limit_deg_s').value))
        self.invert_pitch = bool(self.get_parameter('invert_pitch').value)

        # ---- state
        self._joint_pos: Dict[str, float] = {}
        self._have_left = False
        self._have_right = False
        self._pitch: Optional[float] = None
        self._pitch_rate: float = 0.0
        self._last_avg_cmd: Optional[float] = None

        # ---- I/O
        self.sub_imu = self.create_subscription(Imu, imu_topic, self.on_imu, 10)
        self.sub_js = self.create_subscription(JointState, js_topic, self.on_js, 50)
        self.pub_left = self.create_publisher(Float64, left_cmd_topic, 10)
        self.pub_right = self.create_publisher(Float64, right_cmd_topic, 10)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.update)
        self.get_logger().info('VirtualDifferential started (software leveling of body pitch).')

    def on_imu(self, msg: Imu):
        pitch = quat_to_pitch(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        self._pitch = -pitch if self.invert_pitch else pitch
        self._pitch_rate = float(msg.angular_velocity.y)

    def on_js(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name == self.left_name:
                self._joint_pos[self.left_name] = float(pos)
                self._have_left = True
            elif name == self.right_name:
                self._joint_pos[self.right_name] = float(pos)
                self._have_right = True

    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _rate_limit(self, target, last, limit_per_sec):
        if last is None:
            return target
        max_step = limit_per_sec / self.rate_hz
        return self._clamp(target, last - max_step, last + max_step)

    def update(self):
        if not (self._have_left and self._have_right and (self._pitch is not None)):
            return

        th_l = self._joint_pos[self.left_name]
        th_r = self._joint_pos[self.right_name]
        avg_meas = 0.5 * (th_l + th_r)
        diff_meas = 0.5 * (th_l - th_r)

        err = self._pitch - self.target_pitch
        avg_cmd = avg_meas - self.kp * err - self.kd * self._pitch_rate
        avg_cmd = self._rate_limit(avg_cmd, self._last_avg_cmd, self.avg_rate_limit)
        avg_cmd = self._clamp(avg_cmd, -self.cmd_limit, self.cmd_limit)
        self._last_avg_cmd = avg_cmd

        th_l_cmd = self._clamp(avg_cmd + diff_meas, -self.cmd_limit, self.cmd_limit)
        th_r_cmd = self._clamp(avg_cmd - diff_meas, -self.cmd_limit, self.cmd_limit)

        self.pub_left.publish(Float64(data=float(th_l_cmd)))
        self.pub_right.publish(Float64(data=float(th_r_cmd)))


def main():
    rclpy.init()
    node = VirtualDifferential()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
