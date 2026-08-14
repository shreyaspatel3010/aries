#!/usr/bin/env python3

"""Relay /cmd_vel/teleop onto /cmd_vel while nothing else owns /cmd_vel.

The waypoint stack's cmd_vel_arbiter is the normal owner of /cmd_vel and fails
closed if it sees any other publisher on that topic, so this relay yields by
destroying its publisher -- staying advertised but silent would still trip the
arbiter's conflict check, which inspects endpoints rather than traffic.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelTeleopRelay(Node):
    def __init__(self):
        super().__init__("cmd_vel_teleop_relay")

        self.declare_parameter("input_topic", "/cmd_vel/teleop")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("check_rate_hz", 4.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        check_rate = max(1.0, float(self.get_parameter("check_rate_hz").value))

        self.pub = None
        self.last_cmd = None

        self.create_subscription(Twist, self.input_topic, self._on_teleop, 10)
        self.create_timer(1.0 / check_rate, self._check_ownership)

        self.get_logger().info(
            f"cmd_vel teleop relay ready: {self.input_topic} -> {self.output_topic} "
            f"(yields to any other {self.output_topic} publisher)"
        )

    def _foreign_publishers(self):
        own = f'/{self.get_namespace().strip("/")}/{self.get_name()}'.replace("//", "/")
        found = set()
        for info in self.get_publishers_info_by_topic(self.output_topic):
            name = str(getattr(info, "node_name", "") or "").strip("/")
            namespace = str(getattr(info, "node_namespace", "") or "").strip("/")
            source = f"/{namespace}/{name}" if namespace else f"/{name}"
            if source != own:
                found.add(source)
        return sorted(found)

    def _check_ownership(self):
        foreign = self._foreign_publishers()

        if foreign and self.pub is not None:
            self.destroy_publisher(self.pub)
            self.pub = None
            self.get_logger().info(
                f"yielding {self.output_topic} to: {', '.join(foreign)}"
            )
        elif not foreign and self.pub is None:
            self.pub = self.create_publisher(Twist, self.output_topic, 10)
            self.get_logger().info(f"owning {self.output_topic} (no other publisher)")

    def _on_teleop(self, msg):
        self.last_cmd = msg
        if self.pub is not None:
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelTeleopRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok() and node.pub is not None:
            node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
