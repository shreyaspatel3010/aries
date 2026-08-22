"""Gazebo depth must survive the downlink.

The D435i publishes 16UC1 millimetres; Gazebo's depth camera publishes 32FC1
METRES, and the bridge passes that straight through. The reducer used to reject
anything that was not uint16 and return, which dropped the whole synchronised
pair - colour included - so in simulation the operator view was blank while the
bridged topics underneath were perfectly healthy.

The +Inf case is the one that bites quietly: gz writes it where the ray hit
nothing, and casting +Inf to uint16 is undefined behaviour that lands on 65535
in practice - a 65 m wall of phantom surface right where the sky should be.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

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

np = pytest.importorskip("numpy")
rclpy = pytest.importorskip("rclpy")

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


@pytest.fixture
def node():
    module = _load("camera_downlink")
    instance = module.CameraDownlink()
    yield instance
    instance.destroy_node()


def test_realsense_millimetres_pass_through_untouched(node):
    depth = np.array([[0, 150, 6000, 65535]], dtype=np.uint16)
    out = node._depth_to_mm(depth)
    assert out.dtype == np.uint16
    assert np.array_equal(out, depth)


def test_gazebo_metres_become_millimetres(node):
    depth = np.array([[0.0, 0.15, 1.234, 6.0]], dtype=np.float32)
    out = node._depth_to_mm(depth)
    assert out.dtype == np.uint16
    assert np.array_equal(out, np.array([[0, 150, 1234, 6000]], dtype=np.uint16))


def test_no_return_pixels_become_zero_not_a_phantom_wall(node):
    """0 is the "no reading" value _reduce and DepthCloud both skip."""
    depth = np.array([[np.inf, -np.inf, np.nan, 2.0]], dtype=np.float32)
    out = node._depth_to_mm(depth)
    assert np.array_equal(out, np.array([[0, 0, 0, 2000]], dtype=np.uint16))
    assert out.max() != 65535


def test_beyond_uint16_is_clamped_not_wrapped(node):
    """80 m is past the type, and wrapping it would put a surface at 14 m."""
    depth = np.array([[80.0]], dtype=np.float32)
    assert node._depth_to_mm(depth)[0, 0] == 65535


def test_the_incoming_message_buffer_is_not_mutated(node):
    """imgmsg_to_cv2 can hand back a view onto the message; writing through it
    would corrupt the frame for every other subscriber in the process."""
    depth = np.array([[1.0, np.inf]], dtype=np.float32)
    before = depth.copy()
    node._depth_to_mm(depth)
    assert np.array_equal(depth, before, equal_nan=True)


def test_an_unusable_encoding_is_reported_not_guessed(node):
    assert node._depth_to_mm(np.array([[1, 2]], dtype=np.int8)) is None


def test_the_conversion_lands_inside_the_configured_range(node):
    """_reduce clamps in millimetres, so the units have to agree with it or
    every pixel is zeroed as out of range."""
    metres = np.array([[node.depth_min_mm / 1000.0, node.depth_max_mm / 1000.0]],
                      dtype=np.float32)
    mm = node._depth_to_mm(metres)
    colour = np.zeros((1, 2, 3), dtype=np.uint8)
    _, reduced = node._reduce(colour, mm)
    assert (reduced > 0).all(), "in-range depth must survive the clamp"
