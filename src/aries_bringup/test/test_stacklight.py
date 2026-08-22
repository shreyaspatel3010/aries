"""The stack light must agree with the firmware, and never lie about safety.

The light is a bystander's only cue about a rover that can move six wheels and
spin an auger, so the two properties worth pinning are:

  * the colour CODES match teensy_gripper.ino's stacklight_color enum. They are
    bare integers on the wire with nothing to catch a mismatch -- swap two and
    the rover shows green while it drives.
  * green is only ever shown on EVIDENCE. Anything this node cannot see is
    `unknown`, which is red. A node that has lost the drive bridge must not
    keep claiming the rover is parked and safe.

Callbacks are driven by hand against a faked clock. No ROS graph, no hardware.
"""

import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

import pytest

# Bind DDS to loopback before rclpy touches the middleware -- see
# aries_teleop/test/test_joy_watchdog.py for why this is not optional.
_ISOLATED_DDS = Path(tempfile.gettempdir()) / "aries_test_cyclonedds.xml"
_ISOLATED_DDS.write_text(
    '<?xml version="1.0" encoding="UTF-8" ?>\n'
    '<CycloneDDS xmlns="https://cdds.io/config"><Domain id="any"><General>'
    '<Interfaces><NetworkInterface address="127.0.0.1"/></Interfaces>'
    "<AllowMulticast>false</AllowMulticast></General>"
    '<Discovery><Peers><Peer address="127.0.0.1"/></Peers></Discovery>'
    "</Domain></CycloneDDS>\n"
)
os.environ["CYCLONEDDS_URI"] = f"file://{_ISOLATED_DDS}"

rclpy = pytest.importorskip("rclpy")

from std_msgs.msg import Bool, Float64, String  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
NODES = Path(__file__).resolve().parents[1] / "nodes"
FIRMWARE = REPO / "firmware" / "teensy_gripper" / "teensy_gripper.ino"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, NODES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def light(monkeypatch):
    module = _load("stacklight")
    instance = module.Stacklight()
    clock = _Clock()
    monkeypatch.setattr(instance, "_now", clock)
    sent = []
    monkeypatch.setattr(instance.pub, "publish", lambda msg: sent.append(msg.data))
    yield module, instance, clock, sent
    instance.destroy_node()


def _status(**fields):
    payload = {"armed": True, "right_rps": 0.0, "left_rps": 0.0,
               "command_valid": True}
    payload.update(fields)
    return String(data=json.dumps(payload))


def _tick(instance, clock, ticks=1):
    for _ in range(ticks):
        clock.advance(0.1)
        instance._timer_cb()


class TestFirmwareContract:
    """The wire values are bare integers; nothing else checks them."""

    def test_colour_codes_match_the_teensy_enum(self):
        module = _load("stacklight")
        source = FIRMWARE.read_text()
        match = re.search(r"enum\s+stacklight_color\s*\{([^}]*)\}", source)
        assert match, "stacklight_color enum not found in the firmware"

        # `enum { red = 1, yellow, green, disable }` -- C++ implicit numbering.
        codes, value = {}, None
        for item in match.group(1).split(","):
            name, _, explicit = item.partition("=")
            value = int(explicit.strip()) if explicit.strip() else (value or 0) + 1
            codes[name.strip()] = value

        assert module.COLOR_CODES["red"] == codes["red"]
        assert module.COLOR_CODES["yellow"] == codes["yellow"]
        assert module.COLOR_CODES["green"] == codes["green"]
        assert module.COLOR_CODES["off"] == codes["disable"]

    def test_default_topic_matches_the_firmware_subscription(self, light):
        _, instance, _, _ = light
        source = FIRMWARE.read_text()
        topic = re.search(r'UInt8\),\s*"([^"]+)"\)', source).group(1)
        # The firmware declares it relative under an empty namespace, which
        # resolves to a leading slash.
        assert instance.pub.topic_name == "/" + topic.lstrip("/")


class TestNeverLiesAboutSafety:
    def test_starts_red_before_anything_reports(self, light):
        module, instance, clock, sent = light
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["red"], (
            "with no drive status the rover's state is unknown, not ready")

    def test_green_only_once_the_drive_reports(self, light):
        module, instance, clock, sent = light
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["red"]
        instance._drive_status_cb(_status())
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["green"]

    def test_falls_back_to_red_when_the_drive_goes_silent(self, light):
        module, instance, clock, sent = light
        instance._drive_status_cb(_status())
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["green"]

        clock.advance(instance.drive_status_timeout_s + 0.1)
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["red"], (
            "a stale status must not keep showing green")

    def test_malformed_status_is_not_treated_as_a_report(self, light):
        module, instance, clock, sent = light
        instance._drive_status_cb(String(data="{not json"))
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["red"]


class TestStates:
    def test_turning_wheels_are_yellow(self, light):
        module, instance, clock, sent = light
        instance._drive_status_cb(_status(right_rps=0.4, left_rps=0.4))
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["yellow"]

    def test_a_parked_rover_is_green(self, light):
        module, instance, clock, sent = light
        instance._drive_status_cb(_status(right_rps=0.0, left_rps=0.0))
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["green"]

    def test_a_running_drill_axis_is_yellow(self, light):
        module, instance, clock, sent = light
        instance._drive_status_cb(_status())
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["green"]

        auger = "/aries/drill_bit_joint/cmd_vel"
        assert auger in instance.motion, "the drill axes are watched by default"
        instance.motion[auger] = (-30.0, clock())
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["yellow"], (
            "a spinning auger is the rover operating")

    def test_a_quiet_rate_topic_is_not_a_moving_axis(self, light):
        """The drill teleop stops publishing after its zeros, so the last value
        seen must expire rather than latch the light yellow forever."""
        module, instance, clock, sent = light
        instance._drive_status_cb(_status())
        instance.motion["/aries/drill_bit_joint/cmd_vel"] = (-30.0, clock())
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["yellow"]

        clock.advance(instance.motion_timeout_s + 0.1)
        instance._drive_status_cb(_status())
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["green"]

    def test_emergency_and_halt_have_no_source_yet(self, light):
        """Both e-stops cut ODrive power, so nothing publishes them. The states
        exist and are wired; they are simply unreachable until a sense line
        does. If this ever fails, the hook was filled in -- update the test."""
        _, instance, _, _ = light
        assert instance.estop == {} and instance.halt == {}
        assert "emergency" in instance.priority and "halt" in instance.priority


class TestConfiguredSources:
    """What happens once estop_topics/halt_topics are filled in.

    The subscription callbacks do nothing but write these dicts, so setting
    them directly exercises the same path without a live ROS graph."""

    def test_a_pressed_estop_beats_everything_else(self, light):
        module, instance, clock, sent = light
        # Simulate the wired-up case directly: the subscription callbacks only
        # write these dicts.
        instance.estop["/aries/estop/wheels"] = False
        instance._drive_status_cb(_status(right_rps=0.4, left_rps=0.4))
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["yellow"]

        instance.estop["/aries/estop/wheels"] = True
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["red"], (
            "an e-stop pressed while moving must show red, not yellow")

    def test_halt_is_red(self, light):
        module, instance, clock, sent = light
        instance._drive_status_cb(_status())
        instance.halt["/aries/halt"] = True
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["red"]

    def test_active_low_estop_reads_the_other_way(self, light):
        module, instance, clock, sent = light
        instance.estop_active_high = False
        instance._drive_status_cb(_status())
        instance.estop["/aries/estop/arm"] = True     # healthy loop
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["green"]
        instance.estop["/aries/estop/arm"] = False    # loop broken = pressed
        _tick(instance, clock)
        assert sent[-1] == module.COLOR_CODES["red"]


class TestPublishing:
    def test_refreshes_so_a_reconnected_teensy_resyncs(self, light):
        """The firmware's subscription is not TRANSIENT_LOCAL and is recreated
        on every micro-ROS reconnect, so a publisher that only spoke on change
        would leave the light dark."""
        module, instance, clock, sent = light
        instance._drive_status_cb(_status())
        _tick(instance, clock)
        before = len(sent)
        for _ in range(30):                     # 3 s at 10 Hz, nothing changing
            clock.advance(0.1)
            instance._drive_status_cb(_status())
            instance._timer_cb()
        assert len(sent) > before, "the colour must be re-asserted periodically"
        assert set(sent[before:]) == {module.COLOR_CODES["green"]}

    def test_does_not_spam_every_tick(self, light):
        module, instance, clock, sent = light
        instance._drive_status_cb(_status())
        _tick(instance, clock)
        before = len(sent)
        for _ in range(5):                      # 0.5 s, under publish_period_s
            clock.advance(0.1)
            instance._drive_status_cb(_status())
            instance._timer_cb()
        assert len(sent) - before <= 1


class TestConfigValidation:
    def test_an_unknown_colour_is_refused(self, light):
        module, instance, _, _ = light
        instance.colors["ready"] = "purple"
        with pytest.raises(ValueError, match="unknown colour"):
            instance._validate_colors()

    def test_a_state_with_no_colour_is_refused(self, light):
        module, instance, _, _ = light
        instance.priority = instance.priority + ["autonomous"]
        with pytest.raises(ValueError, match="no entry in `colors`"):
            instance._validate_colors()
