"""The three copies of the ODrive wheel map have to agree.

aries_bringup's hardware_checker and full_hardware_checker each carry their own
AXIS_LABELS/AXIS_STATE_NAMES, and aries_common.odrive_axes carries the copy the
base station prints against. Keeping the rover's copies local is deliberate --
nothing on the robot should depend on a shared module changing under it -- so
the cost is that they can drift.

Drift here is silent and expensive. The CAN node ids stopped following the
sides in blocks when the rover was reassembled, and were re-identified by
arming one axis at a time; a copy that missed that change still prints, still
looks right, and names the wrong wheel in the report somebody is about to act
on. Two operators reading two different names for axis 3 across a radio link is
worse again.

So the copies are compared here rather than by review. If this fails, the fix
is to bring all three into line with right_wheels/left_wheels in
aries_drive/config/cmd_vel_odrive_bridge.yaml, which is the file that actually
decides the mapping.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
SHARED = SRC / "aries_common" / "aries_common" / "odrive_axes.py"
ROVER_CHECKERS = (
    SRC / "aries_bringup" / "nodes" / "hardware_checker.py",
    SRC / "aries_bringup" / "nodes" / "full_hardware_checker.py",
)
BRIDGE_CONFIG = SRC / "aries_drive" / "config" / "cmd_vel_odrive_bridge.yaml"

TABLES = ("AXIS_LABELS", "AXIS_STATE_NAMES", "CLOSED_LOOP", "NUM_AXES")


def _module_constants(path):
    """Top-level literal assignments, read without importing.

    ast rather than import: these files are node scripts that pull in rclpy and
    the vendor odrive_can messages, neither of which a plain pytest run has.
    """
    tree = ast.parse(path.read_text())
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in TABLES:
                found[target.id] = ast.literal_eval(node.value)
    return found


@pytest.fixture(scope="module")
def shared():
    return _module_constants(SHARED)


@pytest.mark.parametrize("path", ROVER_CHECKERS, ids=lambda p: p.name)
@pytest.mark.parametrize("table", TABLES)
def test_rover_checkers_match_the_base_station(shared, path, table):
    rover = _module_constants(path)
    assert table in rover, f"{path.name} no longer defines {table}"
    assert table in shared, f"{SHARED.name} no longer defines {table}"
    assert rover[table] == shared[table], (
        f"{path.name} and {SHARED.name} disagree on {table}. The operator and "
        f"the robot would print different names for the same axis."
    )


def test_labels_cover_every_axis(shared):
    assert set(shared["AXIS_LABELS"]) == set(range(shared["NUM_AXES"]))


def test_sides_match_the_bridge_that_wires_them(shared):
    """AXIS_LABELS only names what cmd_vel_odrive_bridge.yaml wired up.

    That file is the one that decides which CAN ids are driven as the right
    side, so a label saying "Right-Rear" for an id listed under left_wheels is
    the mapping being wrong, not the label.
    """
    text = BRIDGE_CONFIG.read_text()
    sides = {}
    for key in ("right_wheels", "left_wheels"):
        line = next(
            (l for l in text.splitlines() if l.strip().startswith(f"{key}:")), None
        )
        assert line, f"{BRIDGE_CONFIG.name} no longer sets {key}"
        sides[key] = ast.literal_eval(line.split(":", 1)[1].strip())

    labels = shared["AXIS_LABELS"]
    for index in sides["right_wheels"]:
        assert labels[index].startswith("Right"), (
            f"axis {index} is driven as a RIGHT wheel but is labelled "
            f"{labels[index].strip()!r}"
        )
    for index in sides["left_wheels"]:
        assert labels[index].startswith("Left"), (
            f"axis {index} is driven as a LEFT wheel but is labelled "
            f"{labels[index].strip()!r}"
        )
