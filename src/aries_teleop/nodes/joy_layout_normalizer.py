#!/usr/bin/env python3

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


class JoyLayoutNormalizer(Node):
    """Republish joystick input in the canonical Xbox/dongle layout."""

    VALID_LAYOUTS = {"auto", "dongle", "bluetooth", "game_controller", "passthrough"}

    def __init__(self):
        super().__init__("joy_layout_normalizer")

        self.declare_parameter("input_topic", "joy/raw")
        self.declare_parameter("output_topic", "joy")
        self.declare_parameter("layout", "auto")
        self.declare_parameter("device", "/dev/input/js0")
        self.declare_parameter("device_name", "")
        self.declare_parameter("canonical_axis_count", 8)
        self.declare_parameter("canonical_button_count", 16)

        identity_buttons = list(range(16))

        self.declare_parameter("dongle_axis_map", [0, 1, 2, 3, 4, 5, 6, 7])
        self.declare_parameter("dongle_axis_signs", [1.0] * 8)
        self.declare_parameter("dongle_axis_button_positive", [-1] * 8)
        self.declare_parameter("dongle_axis_button_negative", [-1] * 8)
        self.declare_parameter("dongle_button_map", identity_buttons)

        # Common Linux Bluetooth HID order:
        #   axes 0/1 = left stick, 2/3 = right stick, 4/5 = triggers, 6/7 = d-pad.
        # Canonical/dongle order expected by gamepad.yaml:
        #   axes 0/1 = left stick, 2/5 = triggers, 3/4 = right stick, 6/7 = d-pad.
        self.declare_parameter("bluetooth_axis_map", [0, 1, 4, 2, 3, 5, 6, 7])
        self.declare_parameter("bluetooth_axis_signs", [1.0] * 8)
        self.declare_parameter("bluetooth_axis_button_positive", [-1] * 8)
        self.declare_parameter("bluetooth_axis_button_negative", [-1] * 8)
        self.declare_parameter("bluetooth_button_map", identity_buttons)

        # ROS 2 joy/game_controller_node SDL order:
        #   axes 0/1 = left stick, 2/3 = right stick, 4/5 = triggers.
        #   buttons 9/10 = LB/RB, buttons 11..14 = d-pad up/down/left/right.
        # Convert it back to the canonical layout already used by gamepad.yaml.
        self.declare_parameter("game_controller_axis_map", [0, 1, 4, 2, 3, 5, -1, -1])
        self.declare_parameter("game_controller_axis_signs", [1.0] * 8)
        self.declare_parameter("game_controller_axis_button_positive", [-1, -1, -1, -1, -1, -1, 13, 11])
        self.declare_parameter("game_controller_axis_button_negative", [-1, -1, -1, -1, -1, -1, 14, 12])
        self.declare_parameter("game_controller_button_map", [0, 1, 2, 3, 9, 10, 4, 6, 5, 7, 8, 11, 12, 13, 14, 15])

        # Canonical trigger axes are 2 (LT) and 5 (RT) in every layout above,
        # but the drivers disagree on what the numbers mean. Measured on a
        # "Microsoft X-Box 360 pad":
        #   joy_node (dongle/bluetooth): +1.0 released -> -1.0 fully pressed
        #   game_controller_node (SDL):   0.0 released -> -1.0 fully pressed
        # Both are negative-going because ros2 joy negates every axis; the
        # difference is only where "released" sits. Describe each layout by its
        # released/pressed endpoints and interpolate, so a driver with yet
        # another convention is a config change rather than a code change.
        # Output is always 0.0 released -> 1.0 fully pressed, letting a
        # consumer threshold a trigger without knowing which driver is running.
        self.declare_parameter("dongle_trigger_axes", [2, 5])
        self.declare_parameter("bluetooth_trigger_axes", [2, 5])
        self.declare_parameter("game_controller_trigger_axes", [2, 5])
        self.declare_parameter("dongle_trigger_released", 1.0)
        self.declare_parameter("dongle_trigger_pressed", -1.0)
        self.declare_parameter("bluetooth_trigger_released", 1.0)
        self.declare_parameter("bluetooth_trigger_pressed", -1.0)
        self.declare_parameter("game_controller_trigger_released", 0.0)
        self.declare_parameter("game_controller_trigger_pressed", -1.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.requested_layout = str(self.get_parameter("layout").value).strip().lower()
        self.device = str(self.get_parameter("device").value)
        self.device_name_override = str(self.get_parameter("device_name").value).strip()
        self.canonical_axis_count = max(0, int(self.get_parameter("canonical_axis_count").value))
        self.canonical_button_count = max(0, int(self.get_parameter("canonical_button_count").value))

        if self.requested_layout not in self.VALID_LAYOUTS:
            self.get_logger().warn(
                f"Unknown joy layout {self.requested_layout!r}; using auto."
            )
            self.requested_layout = "auto"

        self.axis_maps = {
            "dongle": self._int_list("dongle_axis_map", self.canonical_axis_count),
            "bluetooth": self._int_list("bluetooth_axis_map", self.canonical_axis_count),
            "game_controller": self._int_list("game_controller_axis_map", self.canonical_axis_count),
        }
        self.axis_signs = {
            "dongle": self._float_list("dongle_axis_signs", self.canonical_axis_count),
            "bluetooth": self._float_list("bluetooth_axis_signs", self.canonical_axis_count),
            "game_controller": self._float_list("game_controller_axis_signs", self.canonical_axis_count),
        }
        self.axis_button_positive = {
            "dongle": self._int_list("dongle_axis_button_positive", self.canonical_axis_count),
            "bluetooth": self._int_list("bluetooth_axis_button_positive", self.canonical_axis_count),
            "game_controller": self._int_list("game_controller_axis_button_positive", self.canonical_axis_count),
        }
        self.axis_button_negative = {
            "dongle": self._int_list("dongle_axis_button_negative", self.canonical_axis_count),
            "bluetooth": self._int_list("bluetooth_axis_button_negative", self.canonical_axis_count),
            "game_controller": self._int_list("game_controller_axis_button_negative", self.canonical_axis_count),
        }
        self.button_maps = {
            "dongle": self._int_list("dongle_button_map", self.canonical_button_count),
            "bluetooth": self._int_list("bluetooth_button_map", self.canonical_button_count),
            "game_controller": self._int_list("game_controller_button_map", self.canonical_button_count),
        }

        self.trigger_axes = {
            layout: self._int_list(f"{layout}_trigger_axes", 2)
            for layout in ("dongle", "bluetooth", "game_controller")
        }
        self.trigger_released = {
            layout: float(self.get_parameter(f"{layout}_trigger_released").value)
            for layout in ("dongle", "bluetooth", "game_controller")
        }
        self.trigger_pressed = {
            layout: float(self.get_parameter(f"{layout}_trigger_pressed").value)
            for layout in ("dongle", "bluetooth", "game_controller")
        }
        # Where "released" is not near zero, a trigger that has not moved since
        # the device was opened reads 0.0 and would decode as half pressed.
        # Hold such an axis at 0.0 until it has been seen genuinely released at
        # least once. Layouts whose released value is already 0.0 need no latch.
        self.trigger_armed = {}

        self.static_auto_layout, self.static_auto_reason = self._detect_layout_from_device()
        self.last_logged_layout: Optional[str] = None

        self.publisher = self.create_publisher(Joy, self.output_topic, 10)
        self.create_subscription(Joy, self.input_topic, self._joy_callback, 10)

        self.get_logger().info(
            "joy layout normalizer ready: "
            f"{self.input_topic} -> {self.output_topic}, requested_layout={self.requested_layout}"
        )

    def _int_list(self, parameter_name: str, size: int) -> List[int]:
        values = list(self.get_parameter(parameter_name).value)
        values = [int(value) for value in values]
        if len(values) < size:
            values.extend([-1] * (size - len(values)))
        return values[:size]

    def _float_list(self, parameter_name: str, size: int) -> List[float]:
        values = list(self.get_parameter(parameter_name).value)
        values = [float(value) for value in values]
        if len(values) < size:
            values.extend([1.0] * (size - len(values)))
        return values[:size]

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _detect_layout_from_device(self) -> Tuple[Optional[str], str]:
        identity_parts: List[str] = []
        name = ""
        phys = ""
        uniq = ""
        resolved_path = ""

        if self.device_name_override:
            identity_parts.append(self.device_name_override)
            name = self.device_name_override

        js_name = Path(self.device).name
        if js_name:
            sys_device = Path("/sys/class/input") / js_name / "device"
            name = name or self._read_text(sys_device / "name")
            phys = self._read_text(sys_device / "phys")
            uniq = self._read_text(sys_device / "uniq")
            identity_parts.extend([name, phys, uniq])
            try:
                resolved_path = str(sys_device.resolve())
                identity_parts.append(resolved_path)
            except OSError:
                pass

        identity = " ".join(part for part in identity_parts if part)
        lowered = identity.lower()

        if "bluetooth" in lowered or "/hci" in resolved_path.lower() or uniq.count(":") >= 5:
            return "bluetooth", identity or "device identity indicates Bluetooth"

        if "usb" in lowered or "receiver" in lowered or "dongle" in lowered or "x-box 360" in lowered:
            return "dongle", identity or "device identity indicates USB/dongle"

        return None, identity or "device identity unavailable"

    def _select_layout(self, msg: Joy) -> Tuple[str, str]:
        if self.requested_layout in ("dongle", "bluetooth", "game_controller", "passthrough"):
            return self.requested_layout, "forced by joy_layout parameter"

        if len(msg.axes) <= 6 and len(msg.buttons) >= 15:
            return "game_controller", "auto detected SDL game_controller_node shape"

        if self.static_auto_layout is not None:
            return self.static_auto_layout, self.static_auto_reason

        # Joy messages do not carry a reliable device name.  When auto cannot
        # prove Bluetooth, keep the existing configuration's dongle layout.
        if len(msg.axes) < self.canonical_axis_count:
            return "passthrough", (
                f"auto could not normalize {len(msg.axes)} axes into "
                f"{self.canonical_axis_count} canonical axes"
            )

        return "dongle", f"auto fallback: {self.static_auto_reason}"

    def _axis_value(self, msg: Joy, source_index: int, sign: float) -> float:
        if 0 <= source_index < len(msg.axes):
            return float(msg.axes[source_index]) * sign
        return 0.0

    def _button_value(self, msg: Joy, source_index: int) -> int:
        if 0 <= source_index < len(msg.buttons):
            return int(msg.buttons[source_index])
        return 0

    def _mapped_axes(self, msg: Joy, layout: str) -> List[float]:
        axes: List[float] = []
        for output_index, (source_index, sign) in enumerate(zip(self.axis_maps[layout], self.axis_signs[layout])):
            value = self._axis_value(msg, source_index, sign)
            positive_button = self.axis_button_positive[layout][output_index]
            negative_button = self.axis_button_negative[layout][output_index]
            value += float(self._button_value(msg, positive_button))
            value -= float(self._button_value(msg, negative_button))
            axes.append(max(-1.0, min(1.0, value)))
        return self._normalize_triggers(axes, layout)

    def _normalize_triggers(self, axes: List[float], layout: str) -> List[float]:
        released = self.trigger_released[layout]
        span = self.trigger_pressed[layout] - released

        if abs(span) < 1e-6:
            return axes

        needs_arming = abs(released) > 0.5

        for index in self.trigger_axes[layout]:
            if not 0 <= index < len(axes):
                continue

            raw = axes[index]

            if needs_arming:
                key = (layout, index)
                if abs(raw - released) < 0.1:
                    self.trigger_armed[key] = True
                if not self.trigger_armed.get(key, False):
                    axes[index] = 0.0
                    continue

            axes[index] = max(0.0, min(1.0, (raw - released) / span))

        return axes

    def _mapped_buttons(self, msg: Joy, layout: str) -> List[int]:
        return [
            self._button_value(msg, source_index)
            for source_index in self.button_maps[layout]
        ]

    def _joy_callback(self, msg: Joy) -> None:
        layout, reason = self._select_layout(msg)

        if layout != self.last_logged_layout:
            self.get_logger().info(
                f"joy layout active: {layout} ({reason}); "
                f"raw_axes={len(msg.axes)}, raw_buttons={len(msg.buttons)}"
            )
            self.last_logged_layout = layout

        if layout == "passthrough":
            self.publisher.publish(msg)
            return

        normalized = Joy()
        normalized.header = msg.header
        normalized.axes = self._mapped_axes(msg, layout)
        normalized.buttons = self._mapped_buttons(msg, layout)
        self.publisher.publish(normalized)


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node = JoyLayoutNormalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
