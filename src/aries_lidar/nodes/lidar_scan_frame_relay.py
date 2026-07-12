#!/usr/bin/env python3
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class LidarScanFrameRelay(Node):
    def __init__(self):
        super().__init__("lidar_scan_frame_relay")

        self.declare_parameter("input_topic", "/picoscan/scan_raw")
        self.declare_parameter("output_topic", "/scan")
        self.declare_parameter("target_frame", "Lidar_Scan_Link")
        self.declare_parameter("restamp_to_ros_time", True)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.restamp = bool(self.get_parameter("restamp_to_ros_time").value)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(LaserScan, self.output_topic, qos)
        self.subscription = self.create_subscription(
            LaserScan, self.input_topic, self._scan_cb, qos
        )
        self.seen_scan = False
        self.get_logger().info(
            f"Relaying {self.input_topic} -> {self.output_topic}; "
            f"frame={self.target_frame}, restamp={self.restamp}"
        )

    def _scan_cb(self, msg):
        output = copy.deepcopy(msg)
        output.header.frame_id = self.target_frame
        if self.restamp:
            output.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(output)
        if not self.seen_scan:
            self.seen_scan = True
            self.get_logger().info("Received the first picoScan LaserScan")


def main(args=None):
    rclpy.init(args=args)
    node = LidarScanFrameRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
