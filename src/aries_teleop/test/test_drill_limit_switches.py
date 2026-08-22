"""The drill must not move unless a button is asking it to.

Two things this guards, both of them things the drill did:

MOTION ON THE TRIGGER ALONE. The axes used to be commanded as positions, and
the node re-seeded its target from the measured pose every time LT was pressed.
gz held that target with a force PID, which sagged 50 mm under the carriage's
4.1 kg, so every press re-seeded 50 mm lower and the carriage walked down its
own stroke on nothing but the trigger. The axes are rate-commanded now: LT with
an untouched d-pad has to publish exactly zero, however far the measurement has
drifted from anything.

LIMIT SWITCHES. drill_motor has one at the bottom of its travel and one at the
top, and the bin's actuator has one at each end of its stroke. A switch cuts
the motor in ONE direction: sitting on the bottom switch must still leave the
carriage free to come back up, or the drill parks itself at the bottom of the
mast for good.

Callbacks are driven by hand against a faked clock. No ROS graph, no
middleware, no simulator.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

# Bind DDS to loopback before rclpy touches the middleware -- see
# test_joy_watchdog.py, which needs this for the same reason: a developer's
# CYCLONEDDS_URI names an interface no build machine has, and node creation
# then fails outright.
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

from sensor_msgs.msg import Joy, JointState  # noqa: E402

NODES = Path(__file__).resolve().parents[1] / "nodes"

LT = 2          # canonical trigger axis, 0.0 released -> 1.0 pressed
LB = 4          # canonical button, blocks the drill outright
DPAD_V = 7      # +1 UP
DPAD_H = 6      # +1 LEFT
STICK_V = 1     # auger


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


def _joy(trigger=0.0, dpad_v=0.0, dpad_h=0.0, stick_v=0.0, block=False):
    msg = Joy()
    msg.buttons = [0] * 8
    if block:
        msg.buttons[LB] = 1
    axes = [0.0] * 8
    axes[LT] = trigger
    axes[DPAD_V] = dpad_v
    axes[DPAD_H] = dpad_h
    axes[STICK_V] = stick_v
    msg.axes = axes
    return msg


@pytest.fixture
def drill(monkeypatch):
    module = _load("drill_joystick")
    instance = module.DrillJoystick()
    clock = _Clock()
    monkeypatch.setattr(instance, "_now", clock)

    sent = {"motor": [], "container": [], "bit": []}
    for key, pub in (("motor", instance.pub_motor),
                     ("container", instance.pub_container),
                     ("bit", instance.pub_bit)):
        monkeypatch.setattr(
            pub, "publish",
            (lambda bucket: lambda msg: bucket.append(msg.data))(sent[key]))

    yield instance, clock, sent
    instance.destroy_node()


def _hold(instance, clock, ticks=10, **joy):
    """Hold a pad state for `ticks` control periods, as joy_node's autorepeat
    would deliver it."""
    for _ in range(ticks):
        instance._joy_cb(_joy(**joy))
        clock.advance(instance.dt)
        instance._timer_cb()


def _measure(instance, motor=None, container=None):
    msg = JointState()
    names, positions = [], []
    if motor is not None:
        names.append(instance.motor.joint)
        positions.append(motor)
    if container is not None:
        names.append(instance.container.joint)
        positions.append(container)
    msg.name = names
    msg.position = positions
    instance._joint_state_cb(msg)


class TestNothingMovesOnItsOwn:
    def test_silent_until_the_trigger_is_pressed(self, drill):
        instance, clock, sent = drill
        for _ in range(30):
            instance._joy_cb(_joy())
            clock.advance(instance.dt)
            instance._timer_cb()
        assert sent["motor"] == [] and sent["container"] == [] and sent["bit"] == [], (
            "an untouched pad must leave the command topics alone")

    def test_trigger_alone_commands_zero(self, drill):
        instance, clock, sent = drill
        _hold(instance, clock, trigger=1.0)
        assert set(sent["motor"]) == {0.0}
        assert set(sent["container"]) == {0.0}
        assert set(sent["bit"]) == {0.0}

    def test_trigger_alone_commands_zero_wherever_the_drill_is(self, drill):
        """The regression: a measurement away from home must not become a
        command. Positions used to be seeded from it on every press."""
        instance, clock, sent = drill
        for measured in (-0.05, -0.10, -0.20, 0.12):
            _measure(instance, motor=measured, container=-0.05)
            _hold(instance, clock, ticks=3, trigger=1.0)
            _hold(instance, clock, ticks=3, trigger=0.0)
        assert set(sent["motor"]) == {0.0}, "LT alone must never command a rate"
        assert set(sent["container"]) == {0.0}

    def test_release_stops_and_then_goes_quiet(self, drill):
        instance, clock, sent = drill
        _hold(instance, clock, trigger=1.0, dpad_v=-1.0)
        assert sent["motor"][-1] < 0.0

        _hold(instance, clock, ticks=2, trigger=0.0)
        assert sent["motor"][-1] == 0.0, "release must publish a stop"

        # Zeros are repeated for stop_hold_sec, then the topic is left free.
        clock.advance(instance.stop_hold_sec + 0.01)
        instance._timer_cb()
        before = len(sent["motor"])
        for _ in range(10):
            clock.advance(instance.dt)
            instance._timer_cb()
        assert len(sent["motor"]) == before, "an idle drill must not keep talking"

    def test_lb_blocks_and_stale_joy_stops(self, drill):
        instance, clock, sent = drill
        _hold(instance, clock, trigger=1.0, dpad_v=1.0)
        assert sent["motor"][-1] > 0.0

        _hold(instance, clock, ticks=2, trigger=1.0, dpad_v=1.0, block=True)
        assert sent["motor"][-1] == 0.0, "LB must win over LT"

        _hold(instance, clock, ticks=3, trigger=1.0, dpad_v=1.0)
        assert sent["motor"][-1] > 0.0
        clock.advance(instance.joy_timeout_sec + 0.01)
        instance._timer_cb()
        assert sent["motor"][-1] == 0.0, "a dead pad must stop the motor"


class TestFeedLimitSwitches:
    def test_runs_at_the_motor_speed_in_both_directions(self, drill):
        instance, clock, sent = drill
        _measure(instance, motor=0.0)
        _hold(instance, clock, trigger=1.0, dpad_v=-1.0)
        assert sent["motor"][-1] == pytest.approx(-instance.motor.speed)
        _hold(instance, clock, trigger=1.0, dpad_v=1.0)
        assert sent["motor"][-1] == pytest.approx(instance.motor.speed)

    def test_bottom_switch_cuts_down_but_not_up(self, drill):
        instance, clock, sent = drill
        _measure(instance, motor=instance.motor.lower)
        _hold(instance, clock, trigger=1.0, dpad_v=-1.0)
        assert set(sent["motor"]) == {0.0}, "bottom switch must cut the motor"

        _hold(instance, clock, trigger=1.0, dpad_v=1.0)
        assert sent["motor"][-1] > 0.0, "the switch must not trap the carriage"

    def test_top_switch_cuts_up_but_not_down(self, drill):
        instance, clock, sent = drill
        _measure(instance, motor=instance.motor.upper)
        _hold(instance, clock, trigger=1.0, dpad_v=1.0)
        assert set(sent["motor"]) == {0.0}, "top switch must cut the motor"

        _hold(instance, clock, trigger=1.0, dpad_v=-1.0)
        assert sent["motor"][-1] < 0.0

    def test_switch_trips_short_of_the_mechanical_stop(self, drill):
        """It is a switch, not the stop it protects: it fires limit_margin
        before the carriage lands on the plate."""
        instance, clock, sent = drill
        _measure(instance, motor=instance.motor.lower + instance.motor.margin / 2.0)
        _hold(instance, clock, trigger=1.0, dpad_v=-1.0)
        assert set(sent["motor"]) == {0.0}

    def test_container_stops_at_both_ends_of_the_stroke(self, drill):
        instance, clock, sent = drill
        # Parked (q = 0, the upper end): further +X is blocked, -X is free.
        _measure(instance, container=instance.container.upper)
        _hold(instance, clock, trigger=1.0, dpad_h=-1.0)   # RIGHT = park = +X
        assert set(sent["container"]) == {0.0}
        _hold(instance, clock, trigger=1.0, dpad_h=1.0)    # LEFT = under the bit
        assert sent["container"][-1] < 0.0

        # Under the bit (q = -0.1304, the lower end): the mirror image.
        _measure(instance, container=instance.container.lower)
        sent["container"].clear()
        _hold(instance, clock, trigger=1.0, dpad_h=1.0)
        assert set(sent["container"]) == {0.0}
        _hold(instance, clock, trigger=1.0, dpad_h=-1.0)
        assert sent["container"][-1] > 0.0

    def test_switches_work_with_no_joint_states_at_all(self, drill):
        """Nothing publishes the drill's state in a bare `ros2 run`, so the
        axis dead-reckons from what it commanded and still stops itself."""
        instance, clock, sent = drill
        assert not instance.motor.measured
        travel = instance.motor.upper - instance.motor.lower
        ticks = int(travel / instance.motor.speed / instance.dt) + 60
        _hold(instance, clock, ticks=ticks, trigger=1.0, dpad_v=-1.0)
        assert sent["motor"][-1] == 0.0
        assert instance.motor.position <= instance.motor.lower + instance.motor.margin


class TestAuger:
    def test_stick_spins_it_and_centring_stops_it(self, drill):
        instance, clock, sent = drill
        _hold(instance, clock, trigger=1.0, stick_v=1.0)
        # Stick UP is clockwise seen from above, a negative rate about +Z.
        assert sent["bit"][-1] == pytest.approx(
            -instance.bit_max_speed, rel=1e-6)
        _hold(instance, clock, trigger=1.0, stick_v=0.0)
        assert sent["bit"][-1] == 0.0

    def test_it_has_no_limits_to_hit(self, drill):
        instance, clock, sent = drill
        _hold(instance, clock, ticks=200, trigger=1.0, stick_v=-1.0)
        assert all(v == pytest.approx(instance.bit_max_speed) for v in sent["bit"])
