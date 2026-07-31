#!/usr/bin/env python3
"""
ODrive CAN Status Poller

Periodically sends zero-length GET-request frames over the CAN bus to each
ODrive axis.  The ODrive hardware responds with encoder, IQ, torque, error,
temperature, and bus-voltage data frames.  Those responses are received by the
odrive_can_node (C++) which — once all required frame types have arrived —
publishes the controller_status and odrive_status ROS topics.

Without this poller the odrive_can_node only receives the heartbeat frame
(always sent automatically), so controller_status is never published unless the
ODrive firmware is configured for cyclic messages.

GET request format (ODrive CAN protocol):
  can_id = (node_id << 5) | cmd_id
  DLC    = 0  (no payload — ODrive interprets this as "please send data")
"""

import socket
import struct
from itertools import product

import rclpy
from rclpy.node import Node

# Linux socket CAN frame layout: id(4B)  dlc(1B)  pad(3B)  data(8B)
_CAN_FMT = "=IB3x8s"
_EMPTY   = bytes(8)

# ODrive CAN command IDs for the frames that make up controller_status.
# The C++ node publishes controller_status only when ALL three of these arrive
# in addition to the heartbeat (0x001) which is sent by the ODrive automatically.
_CTRL_CMDS = [
    0x009,  # kGetEncoderEstimates → pos_estimate, vel_estimate
    0x014,  # kGetIq               → iq_setpoint,  iq_measured
    0x01C,  # kGetTorques          → torque_target, torque_estimate
]

# ODrive CAN command IDs for the frames that make up odrive_status.
# The C++ node publishes odrive_status only when ALL three arrive.
_ODRV_CMDS = [
    0x003,  # kGetError             → active_errors, disarm_reason
    0x015,  # kGetTemp              → fet_temperature, motor_temperature
    0x017,  # kGetBusVoltageCurrent → bus_voltage, bus_current
]


class ODriveCANPoller(Node):
    def __init__(self) -> None:
        super().__init__("odrive_can_poller")

        self.declare_parameter("interface",    "can0")
        self.declare_parameter("num_axes",     6)
        self.declare_parameter("poll_rate_hz", 5.0)   # 5 Hz → status every ~200 ms

        interface     = self.get_parameter("interface").value
        self.num_axes = int(self.get_parameter("num_axes").value)
        poll_rate     = float(self.get_parameter("poll_rate_hz").value)
        self._requests = list(product(range(self.num_axes), _CTRL_CMDS + _ODRV_CMDS))
        self._request_index = 0

        self._sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            sock.bind((interface,))
            self._sock = sock
            self.get_logger().info(
                f"ODrive CAN poller ready on {interface}  "
                f"({self.num_axes} axes @ {poll_rate:.0f} Hz, "
                f"{len(self._requests)} requests spread over each cycle)"
            )
        except OSError as exc:
            self.get_logger().warn(
                f"Could not open CAN socket on '{interface}': {exc}  "
                f"— status polling disabled.  "
                f"Run: sudo ip link set {interface} up type can bitrate 250000"
            )

        if self._sock is not None and self._requests:
            request_rate = max(poll_rate * len(self._requests), 1.0)
            self.create_timer(1.0 / request_rate, self._poll_one)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _request(self, node_id: int, cmd_id: int) -> None:
        """Send a zero-length GET request to one ODrive axis."""
        can_id = (node_id << 5) | cmd_id
        frame  = struct.pack(_CAN_FMT, can_id, 0, _EMPTY)
        try:
            self._sock.send(frame)
        except OSError as exc:
            self.get_logger().warn(
                f"CAN send error (axis {node_id}, cmd 0x{cmd_id:03X}): {exc}",
                throttle_duration_sec=5.0,
            )

    # ── Poll loop ─────────────────────────────────────────────────────────────

    def _poll_one(self) -> None:
        if self._sock is None:
            return
        axis_id, cmd = self._requests[self._request_index]
        self._request(axis_id, cmd)
        self._request_index = (self._request_index + 1) % len(self._requests)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ODriveCANPoller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
