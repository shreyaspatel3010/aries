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

import errno
import socket
import struct
from itertools import product

import rclpy
from aries_common.detect import can_link_state, describe_can_link
from rclpy.node import Node

# Linux socket CAN frame layout: id(4B)  dlc(1B)  pad(3B)  data(8B)
_CAN_FMT = "=IB3x8s"
_EMPTY   = bytes(8)

# Send errors that mean the socket is bound to an interface that is no longer
# reachable: unplugged (ENXIO/ENODEV), brought down (ENETDOWN), or already
# closed here (EBADF). A replugged USB CAN adapter comes back with a new
# interface index, so only a freshly bound socket can reach it. Everything else
# — a full transmit queue above all — is transient and must not close anything.
_FATAL_SEND_ERRNOS = frozenset(
    {errno.ENXIO, errno.ENODEV, errno.ENETDOWN, errno.EBADF, errno.ENOTCONN}
)

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
        self.declare_parameter("reconnect_period_s", 1.0)

        self.interface = str(self.get_parameter("interface").value)
        self.num_axes = int(self.get_parameter("num_axes").value)
        poll_rate     = float(self.get_parameter("poll_rate_hz").value)
        reconnect_period = max(0.2, float(self.get_parameter("reconnect_period_s").value))
        self._requests = list(product(range(self.num_axes), _CTRL_CMDS + _ODRV_CMDS))
        self._request_index = 0

        self._sock: socket.socket | None = None
        self._open_logged = False
        self._closed_reason: str | None = None
        self._open()

        if self._requests:
            request_rate = max(poll_rate * len(self._requests), 1.0)
            self.create_timer(1.0 / request_rate, self._poll_one)
            # The socket is re-opened on this timer rather than only at startup:
            # the interface can be unplugged and plugged back in at any time,
            # and it then has a new interface index that the old socket — bound
            # to the index, not to the name — can never reach again.
            self.create_timer(reconnect_period, self._reopen_if_closed)

        self.get_logger().info(
            f"ODrive CAN poller on {self.interface}  "
            f"({self.num_axes} axes @ {poll_rate:.0f} Hz, "
            f"{len(self._requests)} requests spread over each cycle, "
            f"reconnecting every {reconnect_period:.1f} s while disconnected)"
        )

    # ── Socket lifecycle ──────────────────────────────────────────────────────

    def _open(self) -> bool:
        # Binding to a down interface succeeds and yields a socket that fails
        # on every send, so the link has to be checked first or this would
        # re-bind and break again once a second for as long as it stays down.
        link = can_link_state(self.interface)
        if not link["usable"]:
            if not self._open_logged:
                self._open_logged = True
                self.get_logger().warn(
                    f"CAN interface {self.interface} is {describe_can_link(link)}  "
                    f"— status polling paused until it is usable.  "
                    f"Run: sudo ip link set {self.interface} up type can bitrate 250000"
                )
            return False

        try:
            sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            sock.bind((self.interface,))
        except OSError as exc:
            if not self._open_logged:
                self._open_logged = True
                self.get_logger().warn(
                    f"Could not open CAN socket on '{self.interface}': {exc}  "
                    f"— status polling paused until it comes back.  "
                    f"Run: sudo ip link set {self.interface} up type can bitrate 250000"
                )
            return False

        self._sock = sock
        self._open_logged = False
        if self._closed_reason is not None:
            self.get_logger().info(
                f"CAN socket re-bound to {self.interface} "
                f"(ifindex {link['ifindex']}) after: {self._closed_reason}"
            )
            self._closed_reason = None
        return True

    def _close(self, reason: str) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._closed_reason = reason
        self.get_logger().warn(f"CAN socket closed: {reason}; will re-bind when possible")

    def _reopen_if_closed(self) -> None:
        if self._sock is None:
            self._open()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _request(self, node_id: int, cmd_id: int) -> None:
        """Send a zero-length GET request to one ODrive axis."""
        can_id = (node_id << 5) | cmd_id
        frame  = struct.pack(_CAN_FMT, can_id, 0, _EMPTY)
        try:
            self._sock.send(frame)
        except OSError as exc:
            if exc.errno in _FATAL_SEND_ERRNOS:
                self._close(f"axis {node_id} cmd 0x{cmd_id:03X}: {exc}")
                return
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
