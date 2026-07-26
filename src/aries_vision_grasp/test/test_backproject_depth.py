import numpy as np
import pytest

from aries_vision_grasp.grasp_verification import backproject_depth

# The gripper depth camera this runs against.
FX = FY = 465.74115647
CX, CY = 320.0, 240.0


def test_principal_point_pixel_maps_to_the_optical_axis():
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[int(CY), int(CX)] = 1.5
    pts = backproject_depth(depth, FX, FY, CX, CY)
    assert pts.shape == (1, 3)
    assert pts[0] == pytest.approx([0.0, 0.0, 1.5])


def test_offset_pixel_scales_with_depth_over_focal_length():
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[int(CY) + 100, int(CX) + 50] = 2.0
    pts = backproject_depth(depth, FX, FY, CX, CY)
    assert pts[0, 0] == pytest.approx(50.0 * 2.0 / FX)
    assert pts[0, 1] == pytest.approx(100.0 * 2.0 / FY)
    assert pts[0, 2] == pytest.approx(2.0)


def test_self_filtered_and_invalid_pixels_are_dropped():
    # MoveIt's self-filter blanks robot pixels; those must not become points at
    # the optical origin, which would sit inside the jaw volume and read as a
    # held probe.
    depth = np.full((8, 8), np.nan, dtype=np.float32)
    depth[2, 3] = 0.0
    depth[4, 5] = -1.0
    depth[6, 7] = np.inf
    depth[1, 1] = 0.8
    pts = backproject_depth(depth, FX, FY, CX, CY)
    assert pts.shape == (1, 3)
    assert pts[0, 2] == pytest.approx(0.8)


def test_fully_filtered_frame_returns_no_points_not_an_error():
    pts = backproject_depth(np.zeros((16, 16), dtype=np.float32), FX, FY, CX, CY)
    assert pts.shape == (0, 3)


def test_stride_keeps_true_pixel_geometry():
    # A strided call must return the SAME 3D point as full rate for a pixel both
    # sample, not a rescaled one — otherwise the jaw volume is tested against a
    # cloud that is subtly the wrong size.
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[100, 200] = 1.25  # even indices, so stride 2 samples it too
    full = backproject_depth(depth, FX, FY, CX, CY, stride=1)
    strided = backproject_depth(depth, FX, FY, CX, CY, stride=2)
    assert full.shape == (1, 3) and strided.shape == (1, 3)
    assert strided[0] == pytest.approx(full[0])


def test_stride_reduces_the_point_count():
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    assert len(backproject_depth(depth, FX, FY, CX, CY, stride=1)) == 10000
    assert len(backproject_depth(depth, FX, FY, CX, CY, stride=2)) == 2500


def test_rejects_bad_intrinsics_and_shapes():
    depth = np.full((8, 8), 1.0, dtype=np.float32)
    with pytest.raises(ValueError):
        backproject_depth(depth, 0.0, FY, CX, CY)
    with pytest.raises(ValueError):
        backproject_depth(np.zeros((4, 4, 3), dtype=np.float32), FX, FY, CX, CY)
