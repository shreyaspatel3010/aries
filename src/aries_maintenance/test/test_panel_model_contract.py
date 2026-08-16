"""Contract checks for the generated ERC maintenance-panel model."""

import json
import pathlib
import runpy
import xml.etree.ElementTree as ET

import numpy as np
import pytest


MODEL_DIR = (pathlib.Path(__file__).resolve().parents[2]
             / "aries" / "models" / "maintenance_panel")
BUILD_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
                / "aries" / "scripts" / "build_erc2026_props.py")


def _assets():
    table_path = MODEL_DIR / "panel_task.json"
    sdf_path = MODEL_DIR / "model.sdf"
    if not table_path.is_file() or not sdf_path.is_file():
        pytest.skip("maintenance panel assets have not been built")
    return json.loads(table_path.read_text()), ET.parse(sdf_path).getroot()


def test_marker_selection_is_seeded_unique_and_randomised():
    namespace = runpy.run_path(str(BUILD_SCRIPT))
    select = namespace["select_panel_marker_ids"]
    allowed = set(namespace["PANEL_MARKER_IDS"])
    layouts = {select(seed) for seed in range(32)}
    seeded_layout = select(7)

    assert len(layouts) > len(allowed)
    assert select(7) == seeded_layout
    assert {next(iter(allowed - set(layout))) for layout in layouts} == allowed
    for layout in layouts:
        assert len(layout) == 3
        assert len(set(layout)) == 3
        assert set(layout) < allowed


def test_report_marker_layout_has_three_slots_and_no_bottom_right():
    table, _ = _assets()
    markers = table["markers"]
    assert len(markers) == 3
    assert len({marker["id"] for marker in markers}) == 3
    assert {marker["id"] for marker in markers} <= {11, 13, 14, 15}

    across = np.array([marker["position"][1] for marker in markers])
    # The legacy table field is named `console_up_slope`, but its vector follows
    # increasing row order on this exported CAD (top controls toward the lower
    # handle). Keep this test tied to the rendered frame, not that old label.
    down_axis = np.asarray(table["console_up_slope"], dtype=float)
    down = np.array([np.asarray(marker["position"], dtype=float) @ down_axis
                     for marker in markers])
    top = np.argsort(down)[:2]
    bottom = int(np.argmax(down))

    assert down[top[0]] == pytest.approx(down[top[1]], abs=2e-4)
    assert down[bottom] - down[top].mean() == pytest.approx(0.380, abs=2e-4)
    assert abs(across[top[0]] - across[top[1]]) == pytest.approx(0.260, abs=2e-4)
    # Viewed from the panel front, +across is the left side. The sole lower
    # marker shares that side with the top-left marker; -across/bottom-right is
    # intentionally absent from the organiser's drawing.
    assert across[bottom] == pytest.approx(across[top].max(), abs=2e-4)

    buttons = [control for control in table["controls"]
               if control["kind"] == "button"]
    button_across = np.mean([control["pivot_position"][1]
                             for control in buttons])
    button_down = np.mean([
        np.asarray(control["pivot_position"], dtype=float) @ down_axis
        for control in buttons])
    assert across[top].mean() == pytest.approx(button_across, abs=2e-4)
    assert down[top].mean() == pytest.approx(button_down - 0.007, abs=2e-4)

    # The 50 mm black code has a white quiet zone, making the complete board
    # 69.2 mm wide. Even that full visual must stay within the 390 mm panel.
    board_half_width = 0.0692 / 2.0
    panel_half_width = 0.390 / 2.0
    assert all(abs(value) + board_half_width < panel_half_width
               for value in across)


def test_all_fourteen_mcbs_have_independent_links_and_joints():
    table, sdf = _assets()
    breakers = [control for control in table["controls"]
                if control["kind"] in {"breaker", "breaker_bank"}]
    assert len(breakers) == 14
    assert [control["name"] for control in breakers] == [
        f"mcb_{index}" for index in range(14)]

    model_names = [control["model_name"] for control in breakers]
    assert len(set(model_names)) == 14
    links = {link.attrib["name"] for link in sdf.findall(".//link")}
    joints = {joint.attrib["name"]: joint.findtext("child")
              for joint in sdf.findall(".//joint")}
    for control, model_name in zip(breakers, model_names):
        assert model_name in links
        assert joints[control["joint"]] == model_name
