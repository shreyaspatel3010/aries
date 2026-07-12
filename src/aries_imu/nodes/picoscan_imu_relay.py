#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


class PicoScanImuRelay(Node):
    def __init__(self):
        super().__init__("picoscan_imu_relay")

        self.declare_parameter("input_topic", "/picoscan/imu_raw")
        self.declare_parameter("output_topic", "/picoscan/imu")
        self.declare_parameter("target_frame", "")

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.publisher = self.create_publisher(Imu, self.output_topic, qos)
        self.subscription = self.create_subscription(
            Imu, self.input_topic, self._imu_cb, qos
        )
        self.seen_imu = False
        self.get_logger().info(
            f"Relaying picoScan IMU {self.input_topic} -> {self.output_topic}"
        )

    @staticmethod
    def _finite3(vector):
        return all(math.isfinite(value) for value in (vector.x, vector.y, vector.z))

    def _imu_cb(self, msg):
        output = Imu()
        output.header = msg.header
        if self.target_frame:
            output.header.frame_id = self.target_frame

        # picoScan supplies angular velocity and acceleration, not an absolute
        # heading estimate. Explicitly mark orientation as unavailable.
        output.orientation.w = 1.0
        output.orientation_covariance[0] = -1.0

        if self._finite3(msg.angular_velocity):
            output.angular_velocity = msg.angular_velocity
            output.angular_velocity_covariance = [
                0.02, 0.0, 0.0,
                0.0, 0.02, 0.0,
                0.0, 0.0, 0.02,
            ]
        else:
            output.angular_velocity_covariance[0] = -1.0

        if self._finite3(msg.linear_acceleration):
            output.linear_acceleration = msg.linear_acceleration
            output.linear_acceleration_covariance = [
                0.5, 0.0, 0.0,
                0.0, 0.5, 0.0,
                0.0, 0.0, 0.5,
            ]
        else:
            output.linear_acceleration_covariance[0] = -1.0

        self.publisher.publish(output)
        if not self.seen_imu:
            self.seen_imu = True
            self.get_logger().info(
                f"Received the first picoScan IMU message in frame {output.header.frame_id}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PicoScanImuRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
