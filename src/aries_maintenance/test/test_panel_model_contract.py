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
    # `console_up_slope` points back and up, away from the rover. Row 1 of the
    # report's front view -- the marker pair -- sits at the LOW, front edge of
    # the console, so the pair projects *low* on that axis and the lone marker
    # 380 mm above it. The organisers' CAD is what settles the direction: pair
    # at z = 0.7523, lone marker at 0.9599.
    up_axis = np.asarray(table["console_up_slope"], dtype=float)
    up = np.array([np.asarray(marker["position"], dtype=float) @ up_axis
                   for marker in markers])
    pair = np.argsort(up)[:2]
    lone = int(np.argmax(up))

    assert up[pair[0]] == pytest.approx(up[pair[1]], abs=2e-4)
    assert up[lone] - up[pair].mean() == pytest.approx(0.380, abs=2e-4)
    assert abs(across[pair[0]] - across[pair[1]]) == pytest.approx(0.260, abs=2e-4)
    # The lone marker shares a side with one of the pair; the opposite corner is
    # intentionally absent from the report's drawing.
    assert min(abs(across[lone] - across[pair])) == pytest.approx(0.0, abs=2e-4)

    buttons = [control for control in table["controls"]
               if control["kind"] == "button"]
    button_across = np.mean([control["pivot_position"][1]
                             for control in buttons])
    button_up = np.mean([
        np.asarray(control["pivot_position"], dtype=float) @ up_axis
        for control in buttons])
    # Buttons share row 1 with the marker pair: centred across it and 10 mm
    # up-slope of the marker centres, both measured off the front view.
    assert button_across == pytest.approx(np.mean(across[pair]), abs=2e-4)
    assert button_up - up[pair].mean() == pytest.approx(0.010, abs=1e-3)

    # The 50 mm black code must stay clear of the panel edges.
    marker_half = 0.050 / 2.0
    face_half_width = 0.330 / 2.0
    assert all(abs(value) + marker_half <= face_half_width + 1e-6
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
