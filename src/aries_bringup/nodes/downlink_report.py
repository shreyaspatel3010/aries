#!/usr/bin/env python3
"""What the operator camera downlink is actually costing, right now.

The sizing in camera_downlink.launch.py comes from a modelled scene. Real
terrain, real lighting and real D435i noise all move it, so measure before
trusting a profile. Run this on the OPERATOR side: subscribing here is what puts
the stream on the link, and the rate it reports is the rate that survived the
link rather than the rate the rover sent.

    ros2 run aries_bringup downlink_report.py
    ros2 run aries_bringup downlink_report.py --ros-args -p seconds:=30.0

Latency needs the rover and operator clocks to agree; without NTP/chrony between
them the age column is meaningless, so it is only shown when it is plausible.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST,
                        durability=DurabilityPolicy.VOLATILE, depth=10)


class DownlinkReport(Node):

    def __init__(self):
        super().__init__('downlink_report')
        self.declare_parameter(
            'cameras', ['gripper_camera', 'rover_camera', 'rear_camera'])
        # Cameras with no depth stream. Listing a depth topic for one of these
        # would report a permanent NO DATA row for a stream that was never
        # meant to exist, which reads as a fault -- and this tool is used
        # precisely to decide whether something is wrong with the link.
        self.declare_parameter('color_only', ['rear_camera'])
        self.declare_parameter('seconds', 15.0)
        cameras = self.get_parameter('cameras').value
        color_only = set(self.get_parameter('color_only').value)
        self.seconds = float(self.get_parameter('seconds').value)

        self.stats = {}
        for camera in cameras:
            topics = [f'/downlink/{camera}/color/compressed']
            if camera not in color_only:
                topics.append(f'/downlink/{camera}/depth/compressedDepth')
            for topic in topics:
                self.stats[topic] = {'n': 0, 'bytes': 0, 'age': 0.0, 'aged': 0}
                self.create_subscription(
                    CompressedImage, topic, self._make_cb(topic), SENSOR_QOS)

        self.t0 = self.get_clock().now()
        self.create_timer(self.seconds, self._report)
        self.get_logger().info(
            f'measuring {len(self.stats)} streams for {self.seconds:g} s '
            '(subscribing is what pulls them across the link)')

    def _make_cb(self, topic):
        def cb(msg):
            s = self.stats[topic]
            s['n'] += 1
            s['bytes'] += len(msg.data)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp > 0.0:
                age = self.get_clock().now().nanoseconds * 1e-9 - stamp
                if 0.0 <= age < 5.0:      # implausible => clocks are not synced
                    s['age'] += age
                    s['aged'] += 1
        return cb

    def _report(self):
        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        lines = ['', f'downlink over {elapsed:.1f} s',
                 f'  {"topic":<48}{"Hz":>7}{"kB/frame":>10}{"Mbit/s":>9}{"age ms":>9}',
                 '  ' + '-' * 83]
        total = 0.0
        for topic, s in self.stats.items():
            if not s['n']:
                lines.append(f'  {topic:<48}{"NO DATA":>7}')
                continue
            mbps = s['bytes'] * 8 / elapsed / 1e6
            total += mbps
            age = f"{s['age'] / s['aged'] * 1000:>9.0f}" if s['aged'] else f"{'-':>9}"
            lines.append(f"  {topic:<48}{s['n'] / elapsed:>7.1f}"
                         f"{s['bytes'] / s['n'] / 1024:>10.1f}{mbps:>9.2f}{age}")
        lines.append('  ' + '-' * 83)
        lines.append(f'  {"TOTAL":<48}{"":>7}{"":>10}{total:>9.2f}')
        lines.append('  age is blank unless the rover and operator clocks are synced.')
        self.get_logger().info('\n'.join(lines))
        for s in self.stats.values():
            s.update(n=0, bytes=0, age=0.0, aged=0)
        self.t0 = self.get_clock().now()


def main():
    rclpy.init()
    node = DownlinkReport()
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
