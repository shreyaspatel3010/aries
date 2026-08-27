"""The rover must ease into a move, not lurch into it.

accel_limit bounds how fast the commanded velocity may change. Nothing bounded
how fast that RATE may change, so the old trapezoidal ramp went from standstill
to the full acceleration limit inside one control tick -- unbounded jerk at the
start of every move. On a 90 kg six-wheeler the wheels sit in stiction while
the setpoint climbs and then break away into a command already well up the
ramp, which is what the operator feels as a sudden rise in speed.

These pin the shape of the ramp: bounded jerk on the way in, no overshoot on
the way out, and a stop that resets the acceleration so the NEXT start eases in
from zero as well.

Pure arithmetic driven through the node's own ramp and timer. No ROS graph, no
middleware, no wall-clock waiting.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

# Bind DDS to loopback before rclpy touches the middleware -- see
# test_joy_watchdog.py for why a developer's CYCLONEDDS_URI would break this.
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

# The real-rover values from config/joystick.yaml.
ACCEL_LIMIT = 1.25
JERK_LIMIT = 8.0
RATE_HZ = 60.0
DT = 1.0 / RATE_HZ

# LB (button 4) held, left stick fully forward (axis 1).
FULL_AHEAD_BUTTONS = [0, 0, 0, 0, 1, 0, 0, 0]
FULL_AHEAD_AXES = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CENTRED_AXES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, NODES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(monkeypatch):
    module = _load("rover_cmd_vel_joystick")
    clock = _Clock()
    monkeypatch.setattr(module.time, "monotonic", clock)
    instance = module.RoverCmdVelJoystick()
    # This is about the ramp, not LB+Y.
    instance.enable_service = ""
    instance.enable_client = None
    instance.accel_limit = ACCEL_LIMIT
    instance.jerk_limit = JERK_LIMIT
    instance.publish_rate_hz = RATE_HZ
    published = []
    monkeypatch.setattr(instance.pub, "publish", published.append)
    yield instance, clock, published
    instance.destroy_node()


def _joy(axes):
    msg = Joy()
    msg.buttons = list(FULL_AHEAD_BUTTONS)
    msg.axes = list(axes)
    return msg


def _hold(instance, clock, axes, ticks):
    """Hold the stick for `ticks` control periods, as an autorepeating pad does."""
    for _ in range(ticks):
        instance._joy_cb(_joy(axes))
        clock.advance(DT)
        instance._timer_cb()


def _profile(published, before=0.0):
    """Velocity, per-tick acceleration and per-tick jerk from what was sent.

    `before` is the speed the rover was already doing when the first of these
    messages went out, so the very first tick is measured against reality and
    not against itself -- the tick the whole complaint is about.
    """
    speeds = [msg.linear.x for msg in published]
    accels = [(b - a) / DT for a, b in zip([before] + speeds, speeds)]
    jerks = [(b - a) / DT for a, b in zip([0.0] + accels, accels)]
    return speeds, accels, jerks


class TestStartRamp:
    def test_start_does_not_jump_to_the_full_acceleration(self, node):
        """The complaint: the first tick of a move used to carry the whole limit."""
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=3)
        _, accels, _ = _profile(published)
        assert accels[0] < 0.25 * ACCEL_LIMIT, (
            f"first tick commanded {accels[0]:.2f} m/s^2 of {ACCEL_LIMIT} -- "
            "the ramp is stepping in, not easing in"
        )

    def test_jerk_stays_within_the_limit_while_accelerating(self, node):
        """Every tick that raises the acceleration must respect the limit."""
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=90)
        _, accels, jerks = _profile(published)
        rising = [
            j
            for j, a, b in zip(jerks, [0.0] + accels, accels)
            if abs(b) > abs(a)
        ]
        worst = max(abs(j) for j in rising)
        assert worst <= JERK_LIMIT * 1.01, (
            f"jerk reached {worst:.1f} m/s^3 against a limit of {JERK_LIMIT}"
        )

    def test_landing_on_the_target_is_not_a_lurch(self, node):
        """The one tick that can exceed the jerk limit is a DROP into cruise.

        Discretisation leaves a little acceleration to discard on the tick that
        reaches the commanded speed. It has to stay small: it is the last
        remnant of the step this ramp replaced, and it must not grow back into
        one.
        """
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=90)
        _, accels, jerks = _profile(published)
        falling = [
            j
            for j, a, b in zip(jerks, [0.0] + accels, accels)
            if abs(b) <= abs(a)
        ]
        worst = max(abs(j) for j in falling)
        assert worst <= 2.5 * JERK_LIMIT, (
            f"landing discarded {worst * DT:.2f} m/s^2 in one tick"
        )

    def test_acceleration_never_exceeds_its_limit(self, node):
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=60)
        _, accels, _ = _profile(published)
        worst = max(abs(a) for a in accels)
        assert worst <= ACCEL_LIMIT * 1.01, f"{worst:.2f} m/s^2 exceeds {ACCEL_LIMIT}"

    def test_reaches_full_speed_without_overshooting(self, node):
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=90)
        speeds, _, _ = _profile(published)
        assert max(speeds) <= instance.max_linear + 1e-9, (
            "the S-curve overshot the commanded speed"
        )
        assert speeds[-1] == pytest.approx(instance.max_linear, abs=1e-6)

    def test_ramp_time_stays_close_to_the_trapezoid_it_replaced(self, node):
        """Easing in must not turn the rover sluggish."""
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=90)
        speeds, _, _ = _profile(published)
        reached = next(
            i for i, v in enumerate(speeds) if v >= instance.max_linear - 1e-9
        )
        # Trapezoid: max_linear / accel_limit. S-curve adds about accel/jerk.
        budget = instance.max_linear / ACCEL_LIMIT + ACCEL_LIMIT / JERK_LIMIT
        assert (reached + 1) * DT <= budget * 1.1, (
            f"took {(reached + 1) * DT:.2f} s to reach top speed, budget {budget:.2f} s"
        )

    def test_stop_resets_the_acceleration_state(self, node):
        """Otherwise the next start inherits a wound-up acceleration and lurches."""
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=40)
        assert instance.current_linear_accel != 0.0 or instance.current_linear > 0.0

        # The link drops: the watchdog zeroes the output.
        clock.advance(instance.joy_timeout_sec + 0.01)
        instance._timer_cb()
        assert instance.current_linear == 0.0
        assert instance.current_linear_accel == 0.0
        assert instance.current_angular_accel == 0.0

        # And the move after it eases in from zero all over again.
        published.clear()
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=3)
        _, accels, _ = _profile(published)
        assert accels[0] < 0.25 * ACCEL_LIMIT

    def test_release_eases_back_to_zero(self, node):
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=90)
        published.clear()
        _hold(instance, clock, CENTRED_AXES, ticks=90)
        speeds, accels, jerks = _profile(published, before=instance.max_linear)
        assert min(speeds) >= -1e-9, "undershot past zero into reverse"
        assert speeds[-1] == pytest.approx(0.0, abs=1e-9)
        rising = [
            j for j, a, b in zip(jerks, [0.0] + accels, accels) if abs(b) > abs(a)
        ]
        assert max(abs(j) for j in rising) <= JERK_LIMIT * 1.01

    def test_reversal_stays_within_both_limits(self, node):
        """Slamming the stick fore to aft is the worst case for a ramp."""
        instance, clock, published = node
        _hold(instance, clock, FULL_AHEAD_AXES, ticks=90)
        published.clear()
        reverse = list(CENTRED_AXES)
        reverse[1] = -1.0
        _hold(instance, clock, reverse, ticks=120)
        speeds, accels, _ = _profile(published, before=instance.max_linear)
        assert max(abs(a) for a in accels) <= ACCEL_LIMIT * 1.01
        assert min(speeds) >= -instance.max_linear - 1e-9, "overshot into reverse"
        assert speeds[-1] == pytest.approx(-instance.max_linear, abs=1e-6)


class TestJerkLimitDisabled:
    """jerk_limit at 0 must be exactly the trapezoid it replaced."""

    def test_falls_back_to_the_plain_rate_limiter(self, node):
        instance, _, _ = node
        instance.jerk_limit = 0.0
        velocity, accel = 0.0, 0.0
        for _ in range(3):
            velocity, accel = instance._ramp(
                velocity, accel, 1.0, ACCEL_LIMIT, 0.0, DT
            )
        assert velocity == pytest.approx(3 * ACCEL_LIMIT * DT)
        assert accel == 0.0

    def test_lands_on_the_target_without_overshoot(self, node):
        instance, _, _ = node
        velocity, accel = 0.0, 0.0
        for _ in range(500):
            velocity, accel = instance._ramp(
                velocity, accel, 0.05, ACCEL_LIMIT, 0.0, DT
            )
        assert velocity == pytest.approx(0.05)
