"""Regression tests for physical wheel odometry startup and stationary hold."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.duration import Duration


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "Odom.py"
SPEC = importlib.util.spec_from_file_location("rover_physical_odom", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def odom_node(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    rclpy.init()
    node = MODULE.OdometryNode()
    try:
        yield node
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _sample(node, axis, position):
    node.feedback_callback(SimpleNamespace(pos_estimate=float(position)), axis)


def test_baseline_waits_for_every_encoder(odom_node):
    for axis in range(5):
        _sample(odom_node, axis, 100.0 + axis)
    odom_node.update_odometry()
    assert odom_node.prev_left_pos[0] is None
    assert odom_node.x == pytest.approx(0.0)

    _sample(odom_node, 5, 105.0)
    odom_node.update_odometry()
    # left_wheels = [0, 1, 2] and right_wheels = [5, 4, 3], both front -> rear.
    assert odom_node.prev_left_pos == pytest.approx([100.0, 101.0, 102.0])
    assert odom_node.prev_right_pos == pytest.approx([105.0, 104.0, 103.0])
    assert odom_node.x == pytest.approx(0.0)


def test_stationary_sample_clears_previous_velocity(odom_node):
    for axis in range(6):
        _sample(odom_node, axis, 0.0)
    odom_node.update_odometry()

    # Axes 0..2 are the left side, whose motors are mounted opposite to the
    # right side, so forward travel reads negative there and positive on 3..5.
    for axis in range(3):
        _sample(odom_node, axis, -0.1)
    for axis in range(3, 6):
        _sample(odom_node, axis, 0.1)
    odom_node.last_update_time = (
        odom_node.get_clock().now() - Duration(seconds=1.0)
    )
    odom_node.update_odometry()
    assert odom_node.vx > 0.0

    for axis in range(3):
        _sample(odom_node, axis, -0.1)
    for axis in range(3, 6):
        _sample(odom_node, axis, 0.1)
    odom_node.update_odometry()
    assert odom_node.vx == pytest.approx(0.0)
    assert odom_node.vth == pytest.approx(0.0)


def test_median_rejects_one_slipping_wheel():
    estimate, outliers, spread = MODULE.robust_side_displacement(
        [0.020, 0.021, 0.080],
        absolute_threshold_m=0.004,
        relative_threshold=0.35,
    )
    assert estimate == pytest.approx(0.021)
    assert outliers == (2,)
    assert spread == pytest.approx(0.060)


def test_healthy_wheel_variation_is_not_flagged():
    estimate, outliers, _ = MODULE.robust_side_displacement(
        [0.020, 0.021, 0.019],
        absolute_threshold_m=0.004,
        relative_threshold=0.35,
    )
    assert estimate == pytest.approx(0.020)
    assert outliers == ()


def test_physical_odom_rejects_single_axis_slip(odom_node):
    for axis in range(6):
        _sample(odom_node, axis, 0.0)
    odom_node.update_odometry()

    # Left axis 2 spins five times farther than its two side peers. The left
    # encoders use the opposite sign for the same forward travel.
    for axis, position in enumerate([-0.1, -0.1, -0.5, 0.1, 0.1, 0.1]):
        _sample(odom_node, axis, position)
    odom_node.last_update_time = (
        odom_node.get_clock().now() - Duration(seconds=1.0)
    )
    odom_node.update_odometry()

    assert odom_node.x == pytest.approx(0.1 * MODULE.WHEEL_CIRCUMFERENCE)
    assert odom_node.theta == pytest.approx(0.0)
    assert odom_node.slip_detected is True
    assert odom_node.suspected_slip_axes == [2]
