"""The drill driver must agree with the firmware, in units and in names.

This node is the only thing between drill_joystick.py's physical rates and the
duty cycle the Teensy takes, and every disagreement it can have with the
firmware is silent - there is no handshake, no type that differs, nothing that
fails loudly. The three worth pinning:

  * THE ACTUATOR'S FREE SPEED. The firmware converts a requested distance into a
    run TIME using LinearActuator::m_oem_max_speed. This node computes that
    distance using container_actuator_mm_s. If the two ever drift apart, every
    bin move is wrong by the ratio between them - in the same direction, every
    time, with nothing anywhere reporting it.

  * THE OUTPUT TOPIC NAMES, against the strings in the firmware's
    create_entities(). A typo here is a topic that exists, lists fine, and is
    subscribed by nobody.

  * THE MAPPING ITSELF: a deadband that means stopped, a stiction floor that
    means a command to move is a duty cycle that moves, and a clip that does not
    change sign.

Callbacks are driven by hand against a faked clock. No ROS graph, no hardware.
"""

import importlib.util
import os
import re
import tempfile
from pathlib import Path

import pytest
import yaml

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

from std_msgs.msg import Float64  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
NODES = Path(__file__).resolve().parents[1] / "nodes"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "drill_driver.yaml"

FIRMWARE_DIR = REPO / "firmware" / "teensy_drill_sys"
FIRMWARE_MAIN = FIRMWARE_DIR / "src" / "main.cpp"
FIRMWARE_DRILL_H = FIRMWARE_DIR / "lib" / "drill" / "drill.h"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, NODES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    return yaml.safe_load(CONFIG.read_text())["drill_driver"]["ros__parameters"]


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
def driver(monkeypatch):
    module = _load("drill_driver")
    instance = module.DrillDriver()
    clock = _Clock()
    monkeypatch.setattr(instance, "_now", clock)
    # Re-stamp the inputs against the faked clock; the constructor stamped them
    # off the real one, which reads as stale immediately.
    instance.bit_stamp = clock()
    instance.motor_stamp = clock()
    instance.container_stamp = clock()

    sent = {"auger": [], "feed": [], "container": []}
    monkeypatch.setattr(instance.pub_auger, "publish",
                        lambda m: sent["auger"].append(m.data))
    monkeypatch.setattr(instance.pub_feed, "publish",
                        lambda m: sent["feed"].append(m.data))
    monkeypatch.setattr(instance.pub_container, "publish",
                        lambda m: sent["container"].append(m.data))
    yield module, instance, clock, sent
    instance.destroy_node()


class TestFirmwareContract:
    """Nothing at runtime checks any of this."""

    def test_actuator_speed_matches_the_firmware(self):
        """The one mismatch that would be wrong silently, every time."""
        source = FIRMWARE_DRILL_H.read_text()
        match = re.search(
            r"m_oem_max_speed\s*=\s*([0-9.]+)\s*;\s*//\s*\[mm/s\]", source)
        assert match, "m_oem_max_speed not found in %s" % FIRMWARE_DRILL_H
        firmware_mm_s = float(match.group(1))

        assert _config()["container_actuator_mm_s"] == pytest.approx(firmware_mm_s), (
            "drill_driver.yaml's container_actuator_mm_s (%s) must equal the "
            "firmware's LinearActuator::m_oem_max_speed (%s). The firmware turns "
            "a distance into a run time with its copy; this node turns a rate "
            "into that distance with the other. Disagree and every bin move is "
            "wrong by the ratio."
            % (_config()["container_actuator_mm_s"], firmware_mm_s)
        )

    # The handle names are the firmware's C++ identifiers, which say what each
    # axis MOVES. They deliberately differ from the topic names, which are the
    # frozen wire contract -- so `feed_cmd_sub` subscribes `motor2/cmd_speed`.
    # See the topic table at the top of main.cpp.
    @pytest.mark.parametrize("param,handle", [
        ("auger_pwm_topic", "auger_cmd_sub"),
        ("feed_pwm_topic", "feed_cmd_sub"),
        ("container_cext_topic", "bin_cext_cmd_sub"),
    ])
    def test_output_topics_match_the_firmware_subscriptions(self, param, handle):
        source = FIRMWARE_MAIN.read_text()
        # Anchored on the subscription HANDLE rather than the message type:
        # several subscriptions share a type, and a positional match would start
        # checking the wrong one if create_entities' order ever changed.
        match = re.search(r'&%s,.*?,\s*"([^"]+)"\)' % handle, source, re.DOTALL)
        assert match, "%s not found in %s" % (handle, FIRMWARE_MAIN)
        firmware_topic = "/" + match.group(1).lstrip("/")
        assert _config()[param] == firmware_topic


class TestRateMapping:
    def test_deadband_is_stopped_not_slow(self):
        module = _load("drill_driver")
        axis = module.RateToPwm(full_scale=30.0, deadband=0.05, min_pwm=40,
                                invert=False)
        assert axis(0.0)[0] == 0
        assert axis(0.04)[0] == 0, "inside the deadband is a stopped axis"

    def test_clearing_the_deadband_clears_the_stiction_floor(self):
        module = _load("drill_driver")
        axis = module.RateToPwm(full_scale=30.0, deadband=0.05, min_pwm=40,
                                invert=False)
        pwm, _ = axis(0.06)
        assert pwm == 40, (
            "a command that survived the deadband is a command to MOVE, so it "
            "must not round into a duty cycle the mechanism cannot break away at")

    def test_full_scale_is_full_duty_cycle(self):
        module = _load("drill_driver")
        axis = module.RateToPwm(30.0, 0.05, 40, False)
        assert axis(30.0)[0] == 255
        assert axis(-30.0)[0] == -255

    def test_over_range_clips_and_reports_without_changing_sign(self):
        module = _load("drill_driver")
        axis = module.RateToPwm(30.0, 0.05, 40, False)
        pwm, clipped = axis(45.0)
        assert (pwm, clipped) == (255, True)
        pwm, clipped = axis(-45.0)
        assert (pwm, clipped) == (-255, True)

    def test_invert_flips_the_axis_not_the_magnitude(self):
        module = _load("drill_driver")
        axis = module.RateToPwm(30.0, 0.05, 40, invert=True)
        assert axis(30.0)[0] == -255


class TestStopping:
    def test_a_stale_input_is_zero(self, driver):
        _, instance, clock, sent = driver
        instance._bit_cb(Float64(data=30.0))
        instance._tick()
        assert sent["auger"][-1] == 255

        # The joystick node dies mid-spin: no zeros, just silence.
        clock.advance(instance.input_timeout_s + 0.1)
        instance._tick()
        assert sent["auger"][-1] == 0, (
            "an input that was non-zero and went quiet is a publisher that "
            "died; drill_joystick.py always sends its zeros before going silent")

    def test_a_held_command_keeps_being_sent(self, driver):
        """The firmware's watchdog stops a motor that stops being asked."""
        _, instance, clock, sent = driver
        for _ in range(5):
            instance._bit_cb(Float64(data=30.0))
            instance._tick()
            clock.advance(1.0 / 30.0)
        assert len(sent["auger"]) == 5
        assert set(sent["auger"]) == {255}

    def test_zero_goes_quiet_once_it_has_been_said(self, driver):
        _, instance, clock, sent = driver
        for _ in range(5):
            instance._bit_cb(Float64(data=0.0))
            instance._tick()
            clock.advance(1.0 / 30.0)
        assert sent["auger"] == [0], (
            "the firmware is already stopped and its watchdog keeps it there; "
            "restating zero 30 times a second says nothing")

    def test_the_bin_stops_on_the_transition_not_on_silence(self, driver):
        _, instance, clock, sent = driver
        instance._container_cb(Float64(data=0.05))
        instance._tick()
        assert sent["container"][-1] > 0.0

        instance._container_cb(Float64(data=0.0))
        instance._tick()
        assert sent["container"][-1] == 0.0, (
            "the 0.0 is what cuts the remainder of the current timed slice short")

        before = len(sent["container"])
        instance._tick()
        instance._tick()
        assert len(sent["container"]) == before, "and it is said once"

    def test_the_bin_slice_is_a_full_speed_slice(self, driver):
        """The actuator has one speed; only the duration is ours to choose."""
        _, instance, _, sent = driver
        instance._container_cb(Float64(data=0.05))
        instance._tick()
        expected = instance.container_mm_s * instance.container_slice_s
        assert sent["container"][-1] == pytest.approx(expected)

    def test_the_bin_slice_overlaps_the_command_period(self, driver):
        """Otherwise a held command stutters on/off instead of running."""
        _, instance, _, _ = driver
        assert instance.container_slice_s > instance.period

    def test_stop_zeroes_every_axis(self, driver):
        _, instance, _, sent = driver
        instance._bit_cb(Float64(data=30.0))
        instance._motor_cb(Float64(data=0.05))
        instance._tick()
        instance.stop()
        assert sent["auger"][-1] == 0
        assert sent["feed"][-1] == 0
        assert sent["container"][-1] == 0.0
