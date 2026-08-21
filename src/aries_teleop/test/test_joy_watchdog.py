"""The rover must stop when /joy goes quiet.

Both rover drive nodes publish on a TIMER, carrying the last joystick state.
That is fine while the pad is bolted to the robot -- /joy stops only if the
driver dies -- and it is a runaway once the pad is at the base station and /joy
crosses a radio link, because a dropout is then indistinguishable from a held
stick.

It also defeats the guard downstream: cmd_vel_odrive_bridge times out on
/cmd_vel going SILENT, and rover_cmd_vel_joystick keeps it fed with fresh
messages carrying a stale command, so the bridge never sees a gap.

These drive the timer callbacks directly with a faked clock. No ROS graph, no
middleware, no wall-clock waiting.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

# Bind DDS to loopback before rclpy touches the middleware. Without this the
# test inherits whatever CYCLONEDDS_URI the developer's shell exports -- which
# names the field-link address, an interface no build machine has, and every
# node creation fails with "does not match an available interface". Nothing
# here needs a network: the timer callbacks are driven by hand.
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

from sensor_msgs.msg import Joy  # noqa: E402

NODES = Path(__file__).resolve().parents[1] / "nodes"


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


def _joy(buttons=(), axes=()):
    msg = Joy()
    msg.buttons = list(buttons)
    msg.axes = list(axes)
    return msg


class _Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# LB (button 4) held, left stick pushed fully forward (axis 1).
FULL_AHEAD_BUTTONS = [0, 0, 0, 0, 1, 0, 0, 0]
FULL_AHEAD_AXES = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _drive(instance, clock, joy_cb, timer_cb, period, ticks):
    """Hold the stick for `ticks` control periods, as a real pad would.

    joy_node autorepeats at 80 Hz, so a held stick is a continuous stream of
    identical messages -- not one message and then silence. Ticking the timer
    without feeding /joy is the LINK-DOWN case, which is a different test.
    """
    for _ in range(ticks):
        joy_cb(_joy(FULL_AHEAD_BUTTONS, FULL_AHEAD_AXES))
        clock.advance(period)
        timer_cb()


class TestCmdVelJoystick:
    """rover_cmd_vel_joystick: the node the real rover runs."""

    @pytest.fixture
    def node(self, monkeypatch):
        module = _load("rover_cmd_vel_joystick")
        clock = _Clock()
        monkeypatch.setattr(module.time, "monotonic", clock)
        # No enable service: this test is about the watchdog, not LB+Y.
        instance = module.RoverCmdVelJoystick()
        instance.enable_service = ""
        instance.enable_client = None
        published = []
        monkeypatch.setattr(instance.pub, "publish", published.append)
        yield instance, clock, published
        instance.destroy_node()

    def _hold(self, instance, clock, ticks=30):
        _drive(instance, clock, instance._joy_cb, instance._timer_cb,
               1.0 / instance.publish_rate_hz, ticks)

    def test_moves_while_joy_is_fresh(self, node):
        instance, clock, published = node
        self._hold(instance, clock)
        assert published[-1].linear.x > 0.0, "LB held and stick forward must drive"

    def test_stops_when_joy_goes_silent(self, node):
        instance, clock, published = node
        self._hold(instance, clock)
        assert published[-1].linear.x > 0.0

        # The link drops. No further Joy arrives; the timer keeps firing.
        clock.advance(instance.joy_timeout_sec + 0.01)
        instance._timer_cb()
        assert published[-1].linear.x == 0.0
        assert published[-1].angular.z == 0.0

        # And it stays stopped, rather than resuming on the stale command.
        for _ in range(60):
            clock.advance(1.0 / instance.publish_rate_hz)
            instance._timer_cb()
        assert all(m.linear.x == 0.0 for m in published[-60:])

    def test_keeps_publishing_zero_rather_than_falling_silent(self, node):
        """An explicit stop does not depend on a downstream timeout existing."""
        instance, clock, published = node
        self._hold(instance, clock, ticks=5)
        clock.advance(instance.joy_timeout_sec + 0.01)
        before = len(published)
        instance._timer_cb()
        assert len(published) == before + 1, "must publish a zero, not go quiet"

    def test_recovers_when_the_link_comes_back(self, node):
        instance, clock, published = node
        self._hold(instance, clock, ticks=5)
        clock.advance(instance.joy_timeout_sec + 0.01)
        instance._timer_cb()
        assert published[-1].linear.x == 0.0

        self._hold(instance, clock)
        assert published[-1].linear.x > 0.0

    def test_startup_before_the_first_joy_does_not_count_as_fresh(self, node):
        instance, clock, published = node
        clock.advance(5.0)
        instance._timer_cb()
        assert published[-1].linear.x == 0.0


class TestDirectOdriveController:
    """custom_joystick_controller: writes to the axes with nothing behind it."""

    @pytest.fixture
    def node(self, monkeypatch):
        module = _load("custom_joystick_controller")
        clock = _Clock()
        monkeypatch.setattr(module.time, "monotonic", clock)
        instance = module.RoverJoystickController()
        sent = []
        for pub in instance.axis_publishers:
            monkeypatch.setattr(pub, "publish", sent.append)
        yield instance, clock, sent
        instance.destroy_node()

    def _hold(self, instance, clock, ticks=30):
        _drive(instance, clock, instance._joy_callback, instance._publish_loop,
               instance.period, ticks)

    def test_moves_while_joy_is_fresh(self, node):
        instance, clock, sent = node
        self._hold(instance, clock)
        assert any(abs(m.input_vel) > 0.0 for m in sent[-6:])

    def test_ramps_to_a_stop_when_joy_goes_silent(self, node):
        """Ramped, not slammed: six axes stopped in one control period is a
        mechanical shock that can trip the drives, and a tripped drive needs
        LB+Y to recover."""
        instance, clock, sent = node
        self._hold(instance, clock)
        moving = max(abs(m.input_vel) for m in sent[-6:])
        assert moving > 0.0

        clock.advance(instance.joy_timeout_sec + 0.01)
        instance._publish_loop()
        # One period later it is slower but not yet zero -- that is the ramp.
        assert max(abs(m.input_vel) for m in sent[-6:]) < moving

        for _ in range(200):
            clock.advance(instance.period)
            instance._publish_loop()
        assert max(abs(m.input_vel) for m in sent[-6:]) == 0.0

    def test_startup_before_the_first_joy_does_not_count_as_fresh(self, node):
        instance, clock, sent = node
        clock.advance(5.0)
        instance._publish_loop()
        assert max(abs(m.input_vel) for m in sent[-6:]) == 0.0


def test_every_joy_consumer_agrees_on_the_timeout():
    """One release moment across the pad. A node that gives up later than the
    others keeps commanding a subsystem after the rest have stopped."""
    import re

    sources = {
        "rover_cmd_vel_joystick": NODES / "rover_cmd_vel_joystick.py",
        "custom_joystick_controller": NODES / "custom_joystick_controller.py",
        "drill_joystick": NODES / "drill_joystick.py",
    }
    timeouts = {}
    for name, path in sources.items():
        match = re.search(
            r'declare_parameter\(\s*["\']joy_timeout_sec["\']\s*,\s*([0-9.]+)',
            path.read_text(),
        )
        assert match, f"{name} has no joy_timeout_sec: a silent /joy is a runaway"
        timeouts[name] = float(match.group(1))
    assert len(set(timeouts.values())) == 1, timeouts
