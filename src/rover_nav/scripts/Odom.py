#!/usr/bin/env python3
import json
import math
import statistics

import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from odrive_can.msg import ControllerStatus
from rclpy.node import Node
from std_msgs.msg import String

# Constants
WHEEL_CIRCUMFERENCE = 0.697  # meters
WHEELBASE = 0.566  # meters
PUBLISH_RATE = 20  # Hz


def robust_side_displacement(
    wheel_distances,
    absolute_threshold_m=0.004,
    relative_threshold=0.35,
):
    """Return median side travel and indices of wheels inconsistent with it."""
    values = tuple(float(value) for value in wheel_distances)
    if len(values) != 3:
        raise ValueError("exactly three wheel distances are required per side")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wheel distances must be finite")
    if absolute_threshold_m < 0.0 or relative_threshold < 0.0:
        raise ValueError("slip thresholds must be non-negative")

    estimate = float(statistics.median(values))
    tolerance = (
        float(absolute_threshold_m)
        + float(relative_threshold) * abs(estimate)
    )
    outliers = tuple(
        index
        for index, value in enumerate(values)
        if abs(value - estimate) > tolerance
    )
    spread = max(values) - min(values)
    return estimate, outliers, spread


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odom_node')

        self.declare_parameter("slip_absolute_threshold_m", 0.004)
        self.declare_parameter("slip_relative_threshold", 0.35)
        self.declare_parameter("slip_covariance_multiplier", 10.0)
        self.slip_absolute_threshold_m = max(
            0.0,
            float(
                self.get_parameter("slip_absolute_threshold_m").value
            ),
        )
        self.slip_relative_threshold = max(
            0.0,
            float(self.get_parameter("slip_relative_threshold").value),
        )
        self.slip_covariance_multiplier = max(
            1.0,
            float(self.get_parameter("slip_covariance_multiplier").value),
        )
        
        # Which CAN node id drives which side, front -> rear. These were block
        # allocated ([0,1,2] right, [3,4,5] left) until the chassis was
        # reassembled on 2026-08-12 and the node ids stopped following the
        # sides. Keep them equal to the identically named parameters in
        # aries_drive/config/cmd_vel_odrive_bridge.yaml: if commanding and
        # odometry disagree about which side an axis is on, the rover still
        # drives correctly while reporting a mirrored twist, which the EKF then
        # fuses as real motion.
        self.declare_parameter("right_wheels", [0, 4, 3])
        self.declare_parameter("left_wheels", [5, 1, 2])
        self.right_wheels = [int(a) for a in self.get_parameter("right_wheels").value]
        self.left_wheels = [int(a) for a in self.get_parameter("left_wheels").value]

        # axis id -> (side, slot within that side's arrays)
        self.axis_side = {}
        for slot, axis in enumerate(self.right_wheels):
            self.axis_side[axis] = ("right", slot)
        for slot, axis in enumerate(self.left_wheels):
            self.axis_side[axis] = ("left", slot)

        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Velocity state
        self.vx = 0.0
        self.vth = 0.0
        
        # Previous encoder positions (revolutions)
        self.prev_left_pos = [None, None, None]
        self.prev_right_pos = [None, None, None]
        
        # Current encoder positions
        self.current_left_pos = [0.0, 0.0, 0.0]
        self.current_right_pos = [0.0, 0.0, 0.0]
        self.encoder_seen = [False] * 6
        self.encoder_generation = [0] * 6
        self.last_used_generation = [0] * 6
        self.slip_detected = False
        self.suspected_slip_axes = []
        self.last_left_distances = [0.0] * 3
        self.last_right_distances = [0.0] * 3
        
        # Timestamps for velocity calculation
        self.last_update_time = self.get_clock().now()
        
        # Subscribe to all 6 wheels
        for i in range(6):
            self.create_subscription(
                ControllerStatus,
                f'/odrive_axis{i}/controller_status',
                lambda msg, axis=i: self.feedback_callback(msg, axis),
                10
            )
        
        # Publisher - ONLY odometry topic, NO TF!
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.slip_pub = self.create_publisher(
            String, "/wheel_odometry/slip_status", 10
        )
        
        # Timer for periodic odometry calculation and publishing
        self.create_timer(1.0/PUBLISH_RATE, self.update_odometry)
        
        self.get_logger().info(
            "Odometry node started: six-wheel median slip rejection enabled "
            "(TF disabled; EKF publishes TF)"
        )
    
    def feedback_callback(self, msg, axis):
        """Store encoder positions from each wheel"""
        current_pos = float(msg.pos_estimate)  # revolutions
        if not math.isfinite(current_pos):
            self.get_logger().warn(
                f"Ignoring non-finite encoder position from axis {axis}",
                throttle_duration_sec=2.0,
            )
            return
        self.encoder_seen[axis] = True
        self.encoder_generation[axis] += 1
        
        side_slot = self.axis_side.get(axis)
        if side_slot is None:
            self.get_logger().warn(
                f"Encoder from axis {axis}, which is in neither right_wheels "
                f"{self.right_wheels} nor left_wheels {self.left_wheels}; ignoring",
                throttle_duration_sec=10.0,
            )
            return
        side, slot = side_slot
        if side == "right":
            self.current_right_pos[slot] = current_pos
        else:
            self.current_left_pos[slot] = current_pos
    
    def update_odometry(self):
        """Calculate and publish odometry at fixed rate"""
        current_time = self.get_clock().now()

        # Absolute encoder positions can be far from zero and the six streams
        # arrive asynchronously.  Do not establish a baseline until every
        # wheel has supplied at least one real sample.
        if not all(self.encoder_seen):
            self.vx = 0.0
            self.vth = 0.0
            return
        
        # Initialize previous positions on first run
        if self.prev_left_pos[0] is None:
            self.prev_left_pos = self.current_left_pos.copy()
            self.prev_right_pos = self.current_right_pos.copy()
            self.last_used_generation = self.encoder_generation.copy()
            self.vx = 0.0
            self.vth = 0.0
            self.last_update_time = current_time
            self.publish_odom(current_time)
            return

        # Do not compare asynchronous partial snapshots. Every axis must have a
        # newer encoder sample than the one used by the previous integration.
        if not all(
            generation > used
            for generation, used in zip(
                self.encoder_generation, self.last_used_generation
            )
        ):
            return

        # Calculate each physical wheel's distance independently. The left
        # motors are mounted opposite to the right motors.
        left_distances = [
            -(current - previous) * WHEEL_CIRCUMFERENCE
            for current, previous in zip(
                self.current_left_pos, self.prev_left_pos
            )
        ]
        right_distances = [
            (current - previous) * WHEEL_CIRCUMFERENCE
            for current, previous in zip(
                self.current_right_pos, self.prev_right_pos
            )
        ]

        d_left, left_outliers, left_spread = robust_side_displacement(
            left_distances,
            self.slip_absolute_threshold_m,
            self.slip_relative_threshold,
        )
        d_right, right_outliers, right_spread = robust_side_displacement(
            right_distances,
            self.slip_absolute_threshold_m,
            self.slip_relative_threshold,
        )
        # Map the per-side slot back to the CAN node id it came from, so the
        # reported axis numbers stay meaningful now that the sides are not
        # contiguous blocks.
        suspected_axes = [self.left_wheels[index] for index in left_outliers]
        suspected_axes.extend(self.right_wheels[index] for index in right_outliers)
        self.slip_detected = bool(suspected_axes)
        self.suspected_slip_axes = sorted(suspected_axes)
        self.last_left_distances = left_distances
        self.last_right_distances = right_distances
        
        # Update previous positions
        self.prev_left_pos = self.current_left_pos.copy()
        self.prev_right_pos = self.current_right_pos.copy()
        self.last_used_generation = self.encoder_generation.copy()

        self.slip_pub.publish(
            String(
                data=json.dumps(
                    {
                        "slip_detected": self.slip_detected,
                        "suspected_axes": self.suspected_slip_axes,
                        "left_wheel_delta_m": left_distances,
                        "right_wheel_delta_m": right_distances,
                        "left_median_delta_m": d_left,
                        "right_median_delta_m": d_right,
                        "left_spread_m": left_spread,
                        "right_spread_m": right_spread,
                    },
                    separators=(",", ":"),
                )
            )
        )
        if self.slip_detected:
            self.get_logger().warn(
                "Wheel slip disagreement; median odometry rejected axes "
                + ", ".join(str(axis) for axis in self.suspected_slip_axes),
                throttle_duration_sec=1.0,
            )
        
        # Skip if no significant movement
        if abs(d_left) < 0.0001 and abs(d_right) < 0.0001:
            # Never retain the velocity calculated from an earlier encoder
            # delta.  The EKF integrates twist, so stale nonzero velocity makes
            # a stationary rover drift indefinitely.
            self.vx = 0.0
            self.vth = 0.0
            self.last_update_time = current_time
            self.publish_odom(current_time)
            return
        
        # Calculate movement
        d_center = (d_left + d_right) / 2.0
        delta_theta = (d_right - d_left) / WHEELBASE
        
        # Calculate velocities
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        if dt > 0:
            self.vx = d_center / dt
            self.vth = delta_theta / dt
        
        # Update pose using midpoint method for better accuracy
        theta_mid = self.theta + delta_theta / 2.0
        delta_x = d_center * math.cos(theta_mid)
        delta_y = d_center * math.sin(theta_mid)
        
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Update timestamp
        self.last_update_time = current_time
        
        # Publish
        self.publish_odom(current_time)
    
    def publish_odom(self, current_time):
        """Publish odometry message (NO TF - EKF handles that!)"""
        # Create quaternion from yaw (for the odometry message)
        half_yaw = self.theta * 0.5
        q = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(half_yaw),
            w=math.cos(half_yaw),
        )
        
        # Publish odometry message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        
        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q
        
        # Velocity
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = self.vth
        
        # Pose covariance
        covariance_scale = (
            self.slip_covariance_multiplier if self.slip_detected else 1.0
        )
        odom.pose.covariance[0] = 0.01 * covariance_scale   # x
        odom.pose.covariance[7] = 0.01 * covariance_scale   # y
        odom.pose.covariance[14] = 1e6   # z (not measured)
        odom.pose.covariance[21] = 1e6   # roll (not measured)
        odom.pose.covariance[28] = 1e6   # pitch (not measured)
        odom.pose.covariance[35] = 1e6   # yaw - LARGE (don't trust encoder yaw!)

        # Twist covariance
        odom.twist.covariance[0] = 0.01 * covariance_scale   # vx
        odom.twist.covariance[7] = 0.1    # vy
        odom.twist.covariance[14] = 1e6   # vz (not measured)
        odom.twist.covariance[21] = 1e6   # roll rate (not measured)
        odom.twist.covariance[28] = 1e6   # pitch rate (not measured)
        odom.twist.covariance[35] = 0.1   # yaw rate
        
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
