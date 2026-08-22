"""A camera with no depth sensor must still reach the operator.

The rear camera is a Logitech Brio under the tail aimed at the drill: a UVC
webcam, colour only. Every other camera on the robot is a D435i, and the
downlink was built around that assumption -- a two-input
ApproximateTimeSynchronizer that only fires when a colour frame and a depth
frame arrive together.

Handed a single stream, that synchroniser never fires at all. It does not
error: both topics list, the driver publishes, the node sits there, and nothing
reaches the link. That is the failure these tests exist to prevent, so the
colour-only mode is a different code path rather than the paired one with an
absent input, and the tests below check it is actually taken.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest
import yaml

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

PKG = Path(__file__).resolve().parents[1]
NODES = PKG / "nodes"
LAUNCH = PKG / "launch"


def _load(name, directory=NODES):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def color_only_node():
    """A CameraDownlink configured the way camera_downlink.launch.py configures
    the rear camera. The node reads its parameters in __init__, so they have to
    be in place before it runs -- rclpy takes them as parameter_overrides.
    """
    import rclpy.node
    from rclpy.parameter import Parameter

    module = _load("camera_downlink")
    original = rclpy.node.Node.__init__

    def patched(self, *args, **kwargs):
        kwargs.setdefault("parameter_overrides", [
            Parameter("camera", value="rear_camera"),
            Parameter("color_topic", value="/rear_camera/image_raw"),
            Parameter("camera_info_topic", value="/rear_camera/camera_info"),
            Parameter("depth_topic", value=""),
        ])
        return original(self, *args, **kwargs)

    rclpy.node.Node.__init__ = patched
    try:
        node = module.CameraDownlink()
    finally:
        rclpy.node.Node.__init__ = original
    yield node
    node.destroy_node()


@pytest.fixture
def paired_node():
    module = _load("camera_downlink")
    node = module.CameraDownlink()
    yield node
    node.destroy_node()


# ── the node ─────────────────────────────────────────────────────────────────

def test_an_empty_depth_topic_selects_colour_only(color_only_node):
    assert color_only_node.color_only is True


def test_the_default_camera_is_still_paired(paired_node):
    """The switch must be opt-in. A D435i camera that silently lost its depth
    half would look identical to one that never had any."""
    assert paired_node.color_only is False
    assert paired_node.sync is not None


def test_no_synchroniser_is_built_for_a_colour_only_camera(color_only_node):
    """This is the actual bug being prevented: a two-input synchroniser fed one
    stream never fires, and never fires quietly."""
    assert color_only_node.sync is None


def test_no_depth_topic_is_advertised(color_only_node):
    """An advertised topic nothing ever publishes reads as a broken stream."""
    assert color_only_node.pub_depth is None


def test_colour_still_flows(color_only_node):
    """The colour publisher is the entire point of the camera."""
    assert color_only_node.pub_color is not None


def test_decimation_applies_to_colour_without_depth(color_only_node):
    colour = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    color_only_node.decimation = 2
    out = color_only_node._reduce_color(colour)
    assert out.shape == (2, 2, 3)


def test_undecimated_colour_is_passed_through_unchanged(color_only_node):
    colour = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    color_only_node.decimation = 1
    assert np.array_equal(color_only_node._reduce_color(colour), colour)


def test_odd_sizes_are_cropped_to_the_decimation_factor(color_only_node):
    """A 5-pixel row cannot be halved; the remainder is dropped, not wrapped."""
    colour = np.zeros((5, 7, 3), dtype=np.uint8)
    color_only_node.decimation = 2
    assert color_only_node._reduce_color(colour).shape == (2, 3, 3)


def test_the_reducers_agree_on_colour(paired_node):
    """_reduce_color is the colour half of _reduce split out. If the two drift
    apart, the same camera would look different depending on which mode it ran
    in."""
    colour = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    depth = np.full((8, 8), 1000, dtype=np.uint16)
    paired_node.decimation = 2
    paired_colour, _ = paired_node._reduce(colour, depth)
    assert np.array_equal(paired_colour, paired_node._reduce_color(colour))


# ── the launch wiring ────────────────────────────────────────────────────────

CFG = dict(use_sim_time=False, rate_hz=15.0, depth_rate_hz=5.0, decimation=1,
           depth_min_m=0.15, depth_max_m=6.0, depth_quantization_mm=10,
           jpeg_quality=75, png_level=6)


def test_a_paired_camera_gets_three_nodes():
    launch = _load("camera_downlink.launch", LAUNCH)
    actions = launch._downlink_for("rover_camera", CFG, color_only=False)
    assert len(actions) == 3


def test_a_colour_only_camera_gets_two():
    """The third is the depth compressor, and there is no depth to compress."""
    launch = _load("camera_downlink.launch", LAUNCH)
    actions = launch._downlink_for("rear_camera", CFG, color_only=True)
    assert len(actions) == 2


def test_the_operator_side_skips_the_depth_decompressor():
    view = _load("camera_view.launch", LAUNCH)
    assert len(view._view_for("rover_camera", color_only=False)) == 2
    assert len(view._view_for("rear_camera", color_only=True)) == 1


def _text(value):
    """Flatten a launch substitution tuple back to the value it was built from.

    Launch wraps every string in a tuple of TextSubstitutions the moment it
    reaches a Node, and launch_ros serialises parameter VALUES through YAML on
    the way -- so '/rear_camera/image_raw' is stored as
    '/rear_camera/image_raw\n...\n' and the empty string as "''\n". Neither
    compares equal to what was written, so parse the YAML back rather than
    string-matching against its encoding. Non-string parameters (the bools and
    numbers) are not wrapped at all and come through untouched.
    """
    if not isinstance(value, (tuple, list)):
        return value
    text = "".join(getattr(s, "text", "") for s in value)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _remaps(action):
    """The remapping rules of a launch Node, as plain strings."""
    return {_text(src): _text(dst)
            for src, dst in getattr(action, "_Node__remappings", None) or []}


def _params(actions):
    """The first node's parameters, as plain keys and values."""
    merged = {}
    for entry in actions[0]._Node__parameters or []:
        merged.update(entry)
    return {_text(k): _text(v) for k, v in merged.items()}


def test_the_link_topic_names_do_not_depend_on_the_camera_kind():
    """The operator end is deliberately ignorant of which kind of camera is
    behind a stream; only the rover side knows. If the colour-only chain
    published somewhere else, camera_view and downlink_report would both need a
    special case for it."""
    launch = _load("camera_downlink.launch", LAUNCH)
    paired = launch._downlink_for("rover_camera", CFG, color_only=False)
    colour = launch._downlink_for("rear_camera", CFG, color_only=True)

    def color_topic(actions):
        for action in actions:
            out = _remaps(action).get("out/compressed")
            if out:
                return out
        return None

    assert color_topic(paired) == "/downlink/rover_camera/color/compressed"
    assert color_topic(colour) == "/downlink/rear_camera/color/compressed"


def test_the_source_topics_follow_the_driver_that_publishes_them():
    """usb_cam publishes <camera>/image_raw; realsense2_camera publishes
    <camera>/color/image_raw. Reading the wrong one is a node that subscribes
    successfully to a topic nobody publishes."""
    launch = _load("camera_downlink.launch", LAUNCH)
    paired = _params(launch._downlink_for("rover_camera", CFG, color_only=False))
    colour = _params(launch._downlink_for("rear_camera", CFG, color_only=True))

    assert paired["color_topic"] == "/rover_camera/color/image_raw"
    assert paired["depth_topic"] == "/rover_camera/aligned_depth_to_color/image_raw"
    assert colour["color_topic"] == "/rear_camera/image_raw"
    assert colour["camera_info_topic"] == "/rear_camera/camera_info"
    # The empty string is what puts the node in colour-only mode.
    assert colour["depth_topic"] == ""
