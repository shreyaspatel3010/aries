#!/usr/bin/env python3
"""
Waypoint follower (ROS 2 rclpy, no Nav2).

Drives a differential robot through (x, y, yaw) waypoints in the **odom** frame,
using IMU yaw for heading control and /odom for position.

- IMU: heading source
- Odom: position source
- Publishes /cmd_vel

Parameters
----------
  - cmd_vel_topic (string)      default: /cmd_vel
  - odom_topic (string)         default: /odom
  - imu_topic (string)          default: /imu
  - waypoints (string, JSON)    default: '[[1.0,0.0,0.0],[1.0,1.0,1.57]]'
  - rate_hz (double)            default: 30.0
  - k_lin, k_ang, max_lin, max_ang, slow_down_radius
  - tol_dist, tol_yaw, face_first

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return yaw (ROS ENU)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class Gains:
    k_lin: float = 0.8
    k_ang: float = 2.0
    max_lin: float = 0.4  # m/s
    max_ang: float = 1.2  # rad/s
    slow_down_radius: float = 0.6  # m


@dataclass
class Tolerances:
    dist: float = 0.10  # m
    yaw: float = 0.07   # rad
    face_first: bool = True


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: float = 0.0


class WaypointFollower(Node):
    def __init__(self) -> None:
        super().__init__('waypoint_follower')

        # ---- parameters
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/imu')
        # IMPORTANT: string (JSON) for waypoints
        self.declare_parameter('waypoints', '[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]')
        self.declare_parameter('rate_hz', 30.0)

        self.declare_parameter('k_lin', 0.8)
        self.declare_parameter('k_ang', 2.0)
        self.declare_parameter('max_lin', 0.4)
        self.declare_parameter('max_ang', 1.2)
        self.declare_parameter('slow_down_radius', 0.6)

        self.declare_parameter('tol_dist', 0.10)
        self.declare_parameter('tol_yaw', 0.07)
        self.declare_parameter('face_first', True)

        cmd_vel_topic: str = str(self.get_parameter('cmd_vel_topic').value)
        odom_topic: str = str(self.get_parameter('odom_topic').value)
        imu_topic: str = str(self.get_parameter('imu_topic').value)
        rate_hz: float = float(self.get_parameter('rate_hz').value)

        self.gains = Gains(
            k_lin=float(self.get_parameter('k_lin').value),
            k_ang=float(self.get_parameter('k_ang').value),
            max_lin=float(self.get_parameter('max_lin').value),
            max_ang=float(self.get_parameter('max_ang').value),
            slow_down_radius=float(self.get_parameter('slow_down_radius').value),
        )
        self.tols = Tolerances(
            dist=float(self.get_parameter('tol_dist').value),
            yaw=float(self.get_parameter('tol_yaw').value),
            face_first=bool(self.get_parameter('face_first').value),
        )

        # ---- waypoints (string JSON -> list[Waypoint])
        raw_wps_str: str = str(self.get_parameter('waypoints').value)
        try:
            parsed = json.loads(raw_wps_str) if raw_wps_str.strip() else []
        except Exception as e:
            self.get_logger().warn(
                f"Failed to parse 'waypoints' JSON string: {e}. "
                'Using a single (0,0,0).'
            )
            parsed = []

        self.waypoints: List[Waypoint] = []
        for entry in parsed:
            if isinstance(entry, (list, tuple)) and (2 <= len(entry) <= 3):
                x, y = float(entry[0]), float(entry[1])
                yaw = float(entry[2]) if len(entry) == 3 else 0.0
                self.waypoints.append(Waypoint(x, y, yaw))
        if not self.waypoints:
            self.get_logger().warn('No waypoints provided; using a single (0,0,0).')
            self.waypoints = [Waypoint(0.0, 0.0, 0.0)]

        # ---- state
        self._odom_xy: Optional[Tuple[float, float]] = None
        self._imu_yaw: Optional[float] = None
        self._target_idx: int = 0
        self._phase: str = 'align'  # 'align' | 'drive' | 'final_yaw'
        self._last_log_t: float = 0.0
        self._done: bool = False

        # ---- I/O
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        self.sub_imu = self.create_subscription(Imu, imu_topic, self.on_imu, 50)
        self.pub_cmd = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.timer = self.create_timer(1.0 / rate_hz, self.update)
        self.get_logger().info(
            f'WaypointFollower up. cmd_vel={cmd_vel_topic}, odom={odom_topic}, '
            f'imu={imu_topic}, wps={len(self.waypoints)}'
        )

    # ---------- callbacks ----------
    def on_odom(self, msg: Odometry) -> None:
        self._odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        self._imu_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    # ---------- control loop ----------
    def update(self) -> None:
        if self._odom_xy is None or self._imu_yaw is None:
            return

        if self._target_idx >= len(self.waypoints):
            self.finish()
            return

        wp = self.waypoints[self._target_idx]
        x, y = self._odom_xy
        yaw = self._imu_yaw  # heading strictly from IMU

        dx = wp.x - x
        dy = wp.y - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        heading_err = wrap_pi(bearing - yaw)
        final_yaw_err = wrap_pi(wp.yaw - yaw)

        # ---- phase machine
        if self.tols.face_first and self._phase == 'align':
            if abs(heading_err) <= self.tols.yaw or dist < self.tols.dist:
                self._phase = 'drive'
        elif self._phase == 'drive':
            if dist <= self.tols.dist:
                if abs(final_yaw_err) > self.tols.yaw:
                    self._phase = 'final_yaw'
                else:
                    self._phase = 'align'
                    self._target_idx += 1
        elif self._phase == 'final_yaw':
            if abs(final_yaw_err) <= self.tols.yaw:
                self._phase = 'align'
                self._target_idx += 1

        # ---- compute command
        cmd = Twist()

        if self._phase == 'align':
            cmd.angular.z = self._sat(self.gains.k_ang * heading_err, self.gains.max_ang)
            cmd.linear.x = 0.0

        elif self._phase == 'drive':
            ang = self._sat(self.gains.k_ang * heading_err, self.gains.max_ang)
            v_nom = self.gains.k_lin * dist
            if dist < self.gains.slow_down_radius:
                v_nom *= max(0.2, dist / max(self.tols.dist, 1e-6))
            cmd.linear.x = self._sat(v_nom, self.gains.max_lin)
            if abs(ang) > 0.6:
                cmd.linear.x *= 0.5
            cmd.angular.z = ang

        elif self._phase == 'final_yaw':
            cmd.angular.z = self._sat(self.gains.k_ang * final_yaw_err, self.gains.max_ang)
            cmd.linear.x = 0.0

        self.pub_cmd.publish(cmd)

        # ---- status log (~1 Hz)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if (now_sec - self._last_log_t) > 1.0:
            self._last_log_t = now_sec
            self.get_logger().info(
                f'WP {self._target_idx+1}/{len(self.waypoints)} phase={self._phase} '
                f'dist={dist:.2f} head_err={math.degrees(heading_err):.1f}° '
                f'v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}'
            )

    def _sat(self, v: float, lim: float) -> float:
        return max(-lim, min(lim, v))

    def stop(self) -> None:
        self.pub_cmd.publish(Twist())

    def finish(self) -> None:
        """Stop the robot and print a green 'Finished' banner once."""
        if not self._done:
            self.stop()
            green = "\033[92m"  # bright green
            reset = "\033[0m"
            self.get_logger().info(green + "✓ Finished: all waypoints reached." + reset)
            self._done = True
        else:
            self.stop()


def main() -> None:
    rclpy.init()
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down (Ctrl-C).')
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
