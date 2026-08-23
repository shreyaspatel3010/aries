"""A load cell must never report a weight for something it is not holding.

The one that matters is the drill's sample bin. Its cell sits under the PARKED
end of the bin's stroke, so when the bin runs back under the auger to collect,
the cell is holding nothing -- and a cell holding nothing reads ZERO, which is
exactly what a parked-and-empty bin reads. The two are indistinguishable in the
number, so they have to be distinguished outside it. That is what
drill_container/valid is, and most of this file is about the ways it must not
say true:

  * bin off the parked end                      the cell is holding air
  * bin only just stopped, or the auger only
    just stopped                                still ringing
  * counts stale, or the cell at its rail       nothing to report

and the one way it must: parked, quiet, and reporting.

The rest pins the two conversions nothing else checks -- counts to kilograms
through the calibration, and a faulted cell to NaN rather than to 0.0.

Callbacks are driven by hand against a faked clock. No ROS graph, no hardware.
"""

import importlib.util
import json
import math
import os
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

from std_msgs.msg import Bool, Float64, Int32MultiArray  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
NODES = PKG / "nodes"
CONFIG = PKG / "config" / "load_cells.yaml"
REPO = PKG.parents[1]
DRILL_XACRO = REPO / "src" / "aries" / "urdf" / "drill.xacro"


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


@pytest.fixture
def module():
    return _load("load_cells")


# Counts per kilogram, and the empty reading. Round numbers so the arithmetic
# in the assertions is readable, not because a real cell looks like this.
SCALE = 100000.0
OFFSET = 20000.0


def _params(**overrides):
    """A calibrated three-cell node, with round numbers so the arithmetic in
    the assertions is readable. A real cell does not look like this."""
    params = {
        "cells": ["sand_box", "stone_box", "drill_container"],
        "source": "microros",
        "publish_rate_hz": 1000.0,
        "status_rate_hz": 1000.0,
    }
    for name in ("sand_box", "stone_box", "drill_container"):
        params[f"cell.{name}.scale"] = SCALE
        params[f"cell.{name}.offset"] = OFFSET
        params[f"cell.{name}.filter_samples"] = 1
    params.update(overrides)
    return params


def _construct(module, params):
    from rclpy.parameter import Parameter
    import rclpy.node

    overrides = [Parameter(k, value=v) for k, v in params.items()]
    node = module.LoadCells.__new__(module.LoadCells)
    rclpy.node.Node.__init__(
        node, "load_cells_test", parameter_overrides=overrides,
        allow_undeclared_parameters=False)
    # Re-run the node's own __init__ body against this pre-built Node. Calling
    # LoadCells.__init__ would call super().__init__() again, which is not
    # allowed, so the constructor is invoked with the base __init__ stubbed.
    real = rclpy.node.Node.__init__
    rclpy.node.Node.__init__ = lambda self, *a, **k: None
    try:
        module.LoadCells.__init__(node)
    finally:
        rclpy.node.Node.__init__ = real
    return node


class FakeClock:
    def __init__(self, node):
        self.t = 1000.0
        node._now = lambda: self.t

    def advance(self, dt):
        self.t += dt


def _feed(node, clock, sand=0, stone=0, container=0):
    node._raw_array_cb(Int32MultiArray(data=[int(sand), int(stone), int(container)]))


def _counts(kg):
    return OFFSET + kg * SCALE


# --------------------------------------------------------------------------
# counts -> kilograms
# --------------------------------------------------------------------------
def test_counts_convert_to_kilograms(module):
    node = _construct(module, _params())
    clock = FakeClock(node)
    _feed(node, clock, sand=_counts(0.35), stone=_counts(1.2), container=_counts(0.08))
    assert node.cells["sand_box"].weight() == pytest.approx(0.35)
    assert node.cells["stone_box"].weight() == pytest.approx(1.2)
    assert node.cells["drill_container"].weight() == pytest.approx(0.08)
    node.destroy_node()


def test_invert_flips_the_sign_without_a_negative_scale(module):
    node = _construct(module, _params(**{"cell.sand_box.invert": True}))
    FakeClock(node)
    _feed(node, None, sand=_counts(0.5))
    assert node.cells["sand_box"].weight() == pytest.approx(-0.5)
    node.destroy_node()


def test_a_cell_at_its_rail_reads_nan_not_zero(module):
    """An unplugged HX711 rails. Converted, that is a confident huge number;
    reported as 0.0 it is an empty box. Neither is true, so: NaN."""
    node = _construct(module, _params())
    FakeClock(node)
    _feed(node, None, sand=-(1 << 23), stone=_counts(1.0))
    assert math.isnan(node.cells["sand_box"].weight())
    assert node.cells["sand_box"].fault is not None
    # The rail must not enter the filter and poison its neighbours' average.
    assert node.cells["stone_box"].weight() == pytest.approx(1.0)
    node.destroy_node()


def test_an_uncalibrated_cell_refuses_to_start(module):
    """scale 0 divides every reading into infinity. Better a node that will not
    start, with the reason on the console."""
    with pytest.raises(ValueError, match="never been calibrated"):
        _construct(module, _params(**{"cell.sand_box.scale": 0.0}))


# --------------------------------------------------------------------------
# the drill bin's gate
# --------------------------------------------------------------------------
def test_parked_quiet_and_reporting_is_valid(module):
    node = _construct(module, _params())
    clock = FakeClock(node)
    _feed(node, clock, container=_counts(0.1))
    clock.advance(5.0)                       # nothing has moved, ever
    _feed(node, clock, container=_counts(0.1))
    valid, reason = node._container_state()
    assert valid, reason
    node.destroy_node()


def test_bin_under_the_auger_is_not_valid(module):
    """The failure this whole gate exists for: the cell reads a clean zero and
    it means "the bin is somewhere else", not "the bin is empty"."""
    node = _construct(module, _params())
    clock = FakeClock(node)
    # Run the bin back down its stroke for long enough to reach the far end.
    for _ in range(60):
        node._container_rate_cb(Float64(data=-0.05))   # held: a 30 Hz stream
        clock.advance(0.1)
        node._integrate_container()
    assert node.container_q == pytest.approx(-0.1304)

    # The cell now reads its empty value -- exactly what a parked empty bin
    # reads. The number alone cannot tell them apart.
    _feed(node, clock, container=OFFSET)
    assert node.cells["drill_container"].weight() == pytest.approx(0.0)

    node._container_rate_cb(Float64(data=0.0))
    clock.advance(10.0)
    valid, reason = node._container_state()
    assert not valid
    assert "off the cell" in reason
    node.destroy_node()


def test_a_bin_that_only_just_stopped_is_still_settling(module):
    node = _construct(module, _params())
    clock = FakeClock(node)
    _feed(node, clock, container=_counts(0.1))
    node._container_rate_cb(Float64(data=0.01))   # nudged, still parked
    node._container_rate_cb(Float64(data=0.0))
    clock.advance(0.2)
    _feed(node, clock, container=_counts(0.1))
    valid, reason = node._container_state()
    assert not valid and "settling" in reason

    clock.advance(5.0)
    _feed(node, clock, container=_counts(0.1))
    assert node._container_state()[0]
    node.destroy_node()


def test_a_running_auger_disqualifies_the_reading(module):
    """The bin can be parked while the drill cuts. The frame carries that."""
    node = _construct(module, _params())
    clock = FakeClock(node)
    _feed(node, clock, container=_counts(0.1))
    node._auger_rate_cb(Float64(data=20.0))
    clock.advance(0.5)
    _feed(node, clock, container=_counts(0.1))
    assert not node._container_state()[0]
    node.destroy_node()


def test_stale_counts_are_not_a_weight(module):
    node = _construct(module, _params())
    clock = FakeClock(node)
    _feed(node, clock, container=_counts(0.1))
    clock.advance(30.0)                      # the micro-ROS link dropped
    valid, reason = node._container_state()
    assert not valid and reason == "no counts"
    node.destroy_node()


def test_the_end_switch_overrides_dead_reckoning(module):
    """When the switch exists it is the only measurement this axis makes, so it
    wins outright and re-datums the estimate."""
    node = _construct(module, _params(parked_switch_topic="/drill/bin_parked"))
    clock = FakeClock(node)
    node.container_q = -0.09                 # dead reckoning has drifted
    node._parked_switch_cb(Bool(data=True))
    assert node.container_q == pytest.approx(0.0)
    _feed(node, clock, container=_counts(0.1))
    clock.advance(5.0)
    _feed(node, clock, container=_counts(0.1))
    assert node._container_state()[0]

    node._parked_switch_cb(Bool(data=False))
    assert not node._container_state()[0]
    node.destroy_node()


def test_held_weight_survives_the_trip_under_the_auger(module):
    """weight_held is the number the operator wants: what is in the bin, which
    does not stop being true while the bin is away from its cell."""
    node = _construct(module, _params())
    clock = FakeClock(node)
    _feed(node, clock, container=_counts(0.25))
    clock.advance(5.0)
    _feed(node, clock, container=_counts(0.25))
    node._publish_cb()
    assert node.weight_held["drill_container"] == pytest.approx(0.25)

    # Off to collect: the live reading collapses to zero, held does not move.
    for _ in range(60):
        node._container_rate_cb(Float64(data=-0.05))   # held: a 30 Hz stream
        clock.advance(0.1)
        node._integrate_container()
    _feed(node, clock, container=OFFSET)
    node._publish_cb()
    assert not node._container_state()[0]
    assert node.weight_held["drill_container"] == pytest.approx(0.25)
    node.destroy_node()


def test_running_the_bin_into_the_park_end_redatums_it(module):
    """Dead reckoning drifts; the software stop is where it stops drifting."""
    node = _construct(module, _params(container_initial_position=-0.05))
    clock = FakeClock(node)
    for _ in range(60):
        node._container_rate_cb(Float64(data=0.05))
        clock.advance(0.1)
        node._integrate_container()
    assert node.container_q == pytest.approx(0.0)
    node.destroy_node()


def test_an_idle_rate_topic_does_not_keep_integrating(module):
    """drill_joystick.py sends its zeros and then stops publishing entirely.
    Silence is not a moving axis -- treat it as one and the estimate walks to a
    stop on nothing at all."""
    node = _construct(module, _params())
    clock = FakeClock(node)
    node._integrate_container()          # prime: the first tick has no dt yet
    node._container_rate_cb(Float64(data=-0.05))
    clock.advance(0.1)
    node._integrate_container()
    moved = node.container_q
    assert moved < 0.0

    clock.advance(30.0)                      # the topic went quiet
    node._integrate_container()
    assert node.container_q == pytest.approx(moved)
    node.destroy_node()


def test_every_cell_reads_while_the_rover_is_moving(module):
    """All three weights are LIVE readings. None of them is gated on the rover
    holding still: a box being filled is worth watching while it fills, and the
    bin's cell keeps reading while the auger runs. The bin's `valid` labels its
    number; it never withholds it."""
    node = _construct(module, _params())
    clock = FakeClock(node)
    published = {n: [] for n in node.cell_names}
    for n in node.cell_names:
        node.weight_pubs[n].publish = (lambda k: lambda m: published[k].append(m.data))(n)

    # Everything moving at once: bin travelling, auger cutting.
    node._auger_rate_cb(Float64(data=25.0))
    for i in range(10):
        node._container_rate_cb(Float64(data=-0.05))
        clock.advance(0.05)
        _feed(node, clock, sand=_counts(0.1 * i), stone=_counts(1.0),
              container=_counts(0.03))
        node._publish_cb()

    assert not node._container_state()[0], "the bin is off its cell and moving"
    for n in node.cell_names:
        assert len(published[n]) == 10, f"{n} stopped publishing while moving"
    # ...and the sand box tracked the fill the whole way through.
    assert published["sand_box"][0] == pytest.approx(0.0)
    assert published["sand_box"][-1] == pytest.approx(0.9)
    # The bin's live weight is published too, valid or not.
    assert all(v == pytest.approx(0.03) for v in published["drill_container"])
    node.destroy_node()


def test_a_dead_link_publishes_nan_rather_than_going_silent(module):
    """A topic that stops is indistinguishable from a node that died, and an
    operator display would freeze on the last good number. NaN at the steady
    rate says "no reading" out loud."""
    node = _construct(module, _params())
    clock = FakeClock(node)
    published = []
    node.weight_pubs["sand_box"].publish = lambda m: published.append(m.data)

    _feed(node, clock, sand=_counts(0.4))
    node._publish_cb()
    assert published[-1] == pytest.approx(0.4)

    clock.advance(30.0)                      # the micro-ROS link dropped
    node._publish_cb()
    node._publish_cb()
    assert len(published) == 3, "the topic went quiet instead of saying nothing"
    assert math.isnan(published[-1])
    node.destroy_node()


# --------------------------------------------------------------------------
# config and status
# --------------------------------------------------------------------------
def test_status_names_why_the_bin_is_not_valid(module):
    node = _construct(module, _params())
    clock = FakeClock(node)
    _feed(node, clock, sand=_counts(0.4), stone=_counts(1.0), container=_counts(0.1))
    clock.advance(5.0)
    _feed(node, clock, sand=_counts(0.4), stone=_counts(1.0), container=_counts(0.1))

    published = []
    node.status_pub.publish = lambda msg: published.append(json.loads(msg.data))
    node._status_cb()
    s = published[-1]
    assert s["cells"]["sand_box"]["kg"] == pytest.approx(0.4)
    assert s["cells"]["sand_box"]["g"] == pytest.approx(400.0)
    assert s["drill_container"]["valid"] is True
    assert s["drill_container"]["bin_position"] == "dead reckoned"
    node.destroy_node()


def test_shipped_config_matches_the_drill_urdf():
    """The bin's stroke lives in two files. They have to agree, or the gate is
    checking the wrong end of it."""
    import re
    import yaml

    params = yaml.safe_load(CONFIG.read_text())["load_cells"]["ros__parameters"]
    urdf = DRILL_XACRO.read_text()
    joint = urdf.split('<joint name="drill_container_joint"')[1].split("</joint>")[0]
    lower = float(re.search(r'lower="(-?[\d.]+)"', joint).group(1))
    upper = float(re.search(r'upper="(-?[\d.]+)"', joint).group(1))

    assert params["container_lower"] == pytest.approx(lower)
    assert params["container_upper"] == pytest.approx(upper)
    # Parked is the end the cell is under, and it is the +X one.
    assert params["container_parked_position"] == pytest.approx(upper)


def test_shipped_config_lists_the_cells_the_boxes_are():
    params = yaml.safe_load(CONFIG.read_text())["load_cells"]["ros__parameters"]
    assert params["cells"] == ["sand_box", "stone_box", "drill_container"]
    assert params["container_cell"] == "drill_container"
    # Every cell named must have a calibration block, or the node starts with
    # a placeholder scale and reports counts as kilograms.
    for name in params["cells"]:
        assert name in params["cell"], f"{name} has no calibration block"
    # And nothing may quietly ship as calibrated when it is not.
    assert params["source"] == "auto", "the rover must not default to mock counts"

