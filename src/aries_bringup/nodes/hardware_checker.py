#!/usr/bin/env python3
"""
Rover Hardware Status Checker Node

Verifies all 6 ODrive axes and the joystick are online and in the
correct state before / during rover operation.

Checks per axis:
  • controller_status publishing  (recent message within timeout)
  • axis_state == 8 (CLOSED_LOOP_CONTROL)
  • active_errors == 0

Also checks:
  • bus_voltage from odrive_status (each axis)
  • /joy topic for joystick connectivity

Usage:
  ros2 service call /check_rover_hardware std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from odrive_can.msg import ControllerStatus, ODriveStatus
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger


# ── ODrive axis_state values ─────────────────────────────────────────────────
AXIS_STATE_NAMES = {
    0: "UNDEFINED",
    1: "IDLE",
    2: "STARTUP_SEQ",
    3: "FULL_CALIBRATION",
    4: "MOTOR_CALIBRATION",
    6: "ENCODER_OFFSET_CAL",
    7: "ENCODER_INDEX_SEARCH",
    8: "CLOSED_LOOP",
    9: "LOCKIN_SPIN",
    10: "ENCODER_DIR_FIND",
    11: "HOMING",
    12: "ENCODER_HALL_PHASE_CAL",
    13: "ENCODER_HALL_POLARITY_CAL",
}
CLOSED_LOOP = 8

# ── Wheel layout ─────────────────────────────────────────────────────────────
# CAN node id -> physical wheel, re-identified by arming one axis at a time on
# 2026-08-24. Keep in step with right_wheels/left_wheels in
# aries_drive/config/cmd_vel_odrive_bridge.yaml.
# right_wheels = [5, 4, 3]   left_wheels = [0, 1, 2]   (both listed front -> rear)
AXIS_LABELS = {
    0: "Left-Front ",
    1: "Left-Mid   ",
    2: "Left-Rear  ",
    3: "Right-Rear ",
    4: "Right-Mid  ",
    5: "Right-Front",
}
NUM_AXES = 6

# ─────────────────────────────────────────────────────────────────────────────


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class RoverHardwareChecker(Node):
    def __init__(self):
        super().__init__("rover_hardware_checker")

        # ── ANSI colours ──────────────────────────────────────────────────────
        self.GREEN  = "\033[92m"
        self.RED    = "\033[91m"
        self.YELLOW = "\033[93m"
        self.BLUE   = "\033[94m"
        self.CYAN   = "\033[96m"
        self.RESET  = "\033[0m"
        self.BOLD   = "\033[1m"

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("check_interval",    3.0)   # seconds
        self.declare_parameter("timeout",           5.0)   # seconds — stale if older
        self.declare_parameter("require_all_axes",  True)
        self.declare_parameter("check_joystick",    True)
        self.declare_parameter("require_closed_loop", True) # expect CLOSED_LOOP after init
        self.declare_parameter("check_odrive_status", True) # expects odrive_can_poller

        check_interval          = self.get_parameter("check_interval").value
        self.timeout            = self.get_parameter("timeout").value
        self.require_all_axes   = _as_bool(self.get_parameter("require_all_axes").value)
        self.check_joystick     = _as_bool(self.get_parameter("check_joystick").value)
        self.require_closed_loop = _as_bool(self.get_parameter("require_closed_loop").value)
        self.check_odrive_status = _as_bool(self.get_parameter("check_odrive_status").value)

        # ── Per-axis state ────────────────────────────────────────────────────
        self.ctrl_times  = [None] * NUM_AXES   # last controller_status time
        self.ctrl_states = [None] * NUM_AXES   # axis_state uint8
        self.ctrl_errors = [None] * NUM_AXES   # active_errors uint32
        self.ctrl_vel    = [None] * NUM_AXES   # vel_estimate float32

        self.odrv_times   = [None] * NUM_AXES  # last odrive_status time
        self.odrv_voltage = [None] * NUM_AXES  # bus_voltage float32
        self.odrv_errors  = [None] * NUM_AXES  # active_errors uint32

        self.joy_time = None

        # ── Change detection ──────────────────────────────────────────────────
        self.prev_axis_online   = [None] * NUM_AXES
        self.prev_axis_has_data = [None] * NUM_AXES
        self.prev_axis_has_odrv = [None] * NUM_AXES
        self.prev_axis_cl      = [None] * NUM_AXES
        self.prev_axis_err     = [None] * NUM_AXES
        self.prev_joystick_ok  = None
        self.initial_check_done = False
        self.manual_check_requested = False

        # ── QoS to match ODrive C++ publisher (RELIABLE / VOLATILE) ──────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── Subscriptions ─────────────────────────────────────────────────────
        for i in range(NUM_AXES):
            ns = f"odrive_axis{i}"
            self.create_subscription(
                ControllerStatus,
                f"/{ns}/controller_status",
                lambda msg, idx=i: self._ctrl_cb(msg, idx),
                reliable_qos,
            )
            self.create_subscription(
                ODriveStatus,
                f"/{ns}/odrive_status",
                lambda msg, idx=i: self._odrv_cb(msg, idx),
                reliable_qos,
            )

        self.create_subscription(Joy, "/joy", self._joy_cb, sensor_qos)

        # ── Manual check service ──────────────────────────────────────────────
        self.create_service(Trigger, "/check_rover_hardware", self._service_cb)

        # ── Timers ────────────────────────────────────────────────────────────
        # No immediate initial check — DDS discovery for 12 topics needs a few
        # seconds.  First real check fires after check_interval (default 3 s).
        self.create_timer(check_interval, self._check_status)

        self.get_logger().info(
            f"{self.BOLD}{self.BLUE}Rover Hardware Checker Started{self.RESET}\n"
            f"{self.YELLOW}Manual check: "
            f"ros2 service call /check_rover_hardware std_srvs/srv/Trigger{self.RESET}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _service_cb(self, request, response):
        self.manual_check_requested = True
        self._check_status()
        response.success = True
        response.message = "Hardware status check triggered"
        return response

    def _ctrl_cb(self, msg: ControllerStatus, idx: int):
        self.ctrl_times[idx]  = self.get_clock().now()
        self.ctrl_states[idx] = msg.axis_state
        self.ctrl_errors[idx] = msg.active_errors
        self.ctrl_vel[idx]    = msg.vel_estimate

    def _odrv_cb(self, msg: ODriveStatus, idx: int):
        self.odrv_times[idx]   = self.get_clock().now()
        self.odrv_voltage[idx] = msg.bus_voltage
        self.odrv_errors[idx]  = msg.active_errors

    def _joy_cb(self, msg: Joy):
        self.joy_time = self.get_clock().now()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_recent(self, ts) -> bool:
        if ts is None:
            return False
        age = (self.get_clock().now().nanoseconds - ts.nanoseconds) / 1e9
        return age < self.timeout

    # ── Main status check ─────────────────────────────────────────────────────

    def _check_status(self):
        # Software layer: is the odrive_can_node process running?
        axis_node_up = [
            self.count_publishers(f"/odrive_axis{i}/controller_status") > 0
            for i in range(NUM_AXES)
        ]
        # Hardware layer: are we receiving heartbeat frames from the ODrive?
        # With the C++ fix, controller_status is published on every heartbeat,
        # so no recent message = hardware not responding on the CAN bus.
        axis_has_data = [self._is_recent(self.ctrl_times[i]) for i in range(NUM_AXES)]
        axis_has_odrv = [self._is_recent(self.odrv_times[i]) for i in range(NUM_AXES)]

        axis_cl = [
            self.ctrl_states[i] == CLOSED_LOOP if axis_has_data[i] else False
            for i in range(NUM_AXES)
        ]
        axis_err = [
            ((self.ctrl_errors[i] or 0) != 0 or (self.odrv_errors[i] or 0) != 0)
            for i in range(NUM_AXES)
        ]
        joystick_ok = self._is_recent(self.joy_time)

        # ── Change detection ─────────────────────────────────────────────────
        status_changed = (
            axis_node_up   != self.prev_axis_online   or
            axis_has_data  != self.prev_axis_has_data or
            axis_has_odrv  != self.prev_axis_has_odrv or
            axis_cl        != self.prev_axis_cl       or
            axis_err       != self.prev_axis_err      or
            joystick_ok    != self.prev_joystick_ok
        )

        if not self.initial_check_done or status_changed or self.manual_check_requested:
            self.manual_check_requested = False
            self._print_status(
                axis_node_up,
                axis_has_data,
                axis_has_odrv,
                axis_cl,
                axis_err,
                joystick_ok,
            )

            self.prev_axis_online   = list(axis_node_up)
            self.prev_axis_has_data = list(axis_has_data)
            self.prev_axis_has_odrv = list(axis_has_odrv)
            self.prev_axis_cl       = list(axis_cl)
            self.prev_axis_err      = list(axis_err)
            self.prev_joystick_ok   = joystick_ok
            self.initial_check_done = True

    def _print_status(
        self,
        axis_node_up,
        axis_has_data,
        axis_has_odrv,
        axis_cl,
        axis_err,
        joystick_ok,
    ):
        W = 64
        G = self.GREEN
        R = self.RED
        Y = self.YELLOW
        B = self.BOLD
        RST = self.RESET

        responding_indices = [i for i in range(NUM_AXES) if axis_has_data[i]]
        all_node_up = all(axis_node_up)
        any_node_up = any(axis_node_up)
        all_responding = all(axis_has_data)
        any_responding = any(axis_has_data)
        all_odrv_status = all(axis_has_odrv)
        responding_odrv_ok = all(axis_has_odrv[i] for i in responding_indices) if responding_indices else False
        all_cl = all(axis_cl[i] for i in responding_indices) if responding_indices else False
        any_err        = any(axis_err)

        nodes_ok = all_node_up if self.require_all_axes else any_node_up
        responding_ok = all_responding if self.require_all_axes else any_responding
        odrv_ok = (
            all_odrv_status if self.require_all_axes else responding_odrv_ok
        ) if self.check_odrive_status else True
        closed_loop_ok = all_cl if self.require_closed_loop else True
        overall_ok = nodes_ok and responding_ok and not any_err and closed_loop_ok and odrv_ok

        print(f"\n{'═'*W}", flush=True)
        print(f"{B}{self.BLUE}  ROVER HARDWARE STATUS CHECK{RST}", flush=True)
        print(f"{'═'*W}", flush=True)

        # ── Per-axis rows ─────────────────────────────────────────────────────
        print(f"\n{B}  ODrive Axes  (right: 0-2 | left: 3-5):{RST}", flush=True)

        for i in range(NUM_AXES):
            label = AXIS_LABELS[i]

            if not axis_node_up[i]:
                print(f"  {R}✗{RST} Axis {i}  ({label})  — {R}CAN NODE NOT RUNNING{RST}", flush=True)
                continue

            if not axis_has_data[i]:
                # Node is up but no heartbeat received — ODrive hardware not responding
                print(f"  {Y}~{RST} Axis {i}  ({label})  — {Y}NO HEARTBEAT  (hardware not responding){RST}", flush=True)
                continue

            # Full status data available
            state_name = AXIS_STATE_NAMES.get(self.ctrl_states[i], f"STATE_{self.ctrl_states[i]}")
            vel_str    = f"vel:{self.ctrl_vel[i]:+.3f}" if self.ctrl_vel[i] is not None else "vel:---"
            volt_str   = (f"{self.odrv_voltage[i]:.1f}V"
                          if self.odrv_voltage[i] is not None else "---V")

            ctrl_e = self.ctrl_errors[i] or 0
            drv_e  = self.odrv_errors[i] or 0
            odrv_stale = self.check_odrive_status and not axis_has_odrv[i]

            if axis_err[i]:
                err_str = f"{R}ERR ctrl:0x{ctrl_e:08X}  drv:0x{drv_e:08X}{RST}"
                print(
                    f"  {R}✗{RST} Axis {i}  ({label})  — "
                    f"{Y}{state_name}{RST}  {volt_str}  {vel_str}  {err_str}",
                    flush=True,
                )
            elif axis_cl[i]:
                if odrv_stale:
                    print(
                        f"  {Y}~{RST} Axis {i}  ({label})  — "
                        f"{G}CLOSED_LOOP{RST}  {Y}ODRIVE STATUS STALE{RST}  {vel_str}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {G}✓{RST} Axis {i}  ({label})  — "
                        f"{G}CLOSED_LOOP{RST}  {volt_str}  {vel_str}",
                        flush=True,
                    )
            else:
                print(
                    f"  {Y}~{RST} Axis {i}  ({label})  — "
                    f"{Y}{state_name}{RST}  {volt_str}  {vel_str}",
                    flush=True,
                )

        # ── Joystick row ──────────────────────────────────────────────────────
        print(f"\n{B}  Joystick:{RST}", flush=True)
        if joystick_ok:
            print(f"  {G}✓{RST} /joy  — {G}Connected{RST}", flush=True)
        elif self.check_joystick:
            print(f"  {Y}○{RST} /joy  — Not detected  (optional)", flush=True)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'═'*W}", flush=True)

        if overall_ok:
            print(
                f"  {G}{B}✓  ALL SYSTEMS READY — Rover is armed and operational{RST}",
                flush=True,
            )
            if self.check_joystick and not joystick_ok:
                print(
                    f"  {Y}⚠   Joystick not detected — manual control unavailable{RST}",
                    flush=True,
                )
        else:
            print(f"  {R}{B}✗  SYSTEM NOT READY{RST}", flush=True)

            if self.require_all_axes and not all_node_up:
                missing = [i for i in range(NUM_AXES) if not axis_node_up[i]]
                print(
                    f"  {R}→  CAN nodes not running: {missing}{RST}",
                    flush=True,
                )
            elif not self.require_all_axes and not any_node_up:
                print(f"  {R}→  No CAN nodes are running{RST}", flush=True)

            if self.require_all_axes and not all_responding:
                no_hb = [i for i in range(NUM_AXES) if axis_node_up[i] and not axis_has_data[i]]
                if no_hb:
                    print(
                        f"  {Y}→  No heartbeat from ODrive hardware: {no_hb}{RST}  "
                        f"(check power + CAN wiring)",
                        flush=True,
                    )
            elif not self.require_all_axes and not any_responding:
                print(f"  {Y}→  No heartbeat from any ODrive hardware{RST}", flush=True)

            if self.check_odrive_status and not odrv_ok:
                stale_odrv = [
                    i for i in range(NUM_AXES)
                    if axis_has_data[i] and not axis_has_odrv[i]
                ]
                if stale_odrv:
                    print(
                        f"  {Y}→  Missing recent ODrive status frames: {stale_odrv}{RST}  "
                        f"(check odrive_can_poller)",
                        flush=True,
                    )

            if self.require_closed_loop and any(axis_has_data) and not all_cl:
                not_cl = [i for i in range(NUM_AXES) if axis_has_data[i] and not axis_cl[i]]
                if not_cl:
                    print(
                        f"  {Y}→  Axes not yet in CLOSED_LOOP: {not_cl}{RST}  "
                        f"(rover controller still initialising?)",
                        flush=True,
                    )

            if any_err:
                err_axes = [i for i in range(NUM_AXES) if axis_err[i]]
                print(
                    f"  {R}→  Axes reporting errors: {err_axes}{RST}",
                    flush=True,
                )

        print(f"{'═'*W}\n", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = RoverHardwareChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
