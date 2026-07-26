"""Pure geometry helpers for automatic base-box placement."""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# Fallback probe size, overridden by whatever probe.stl actually measures —
# pass probe_length_m/probe_width_m to derive_automatic_box_settings. These are
# only defaults for callers that have no mesh to measure.
PROBE_LENGTH_M = 0.200
PROBE_WIDTH_M = 0.030


def rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimum rotation taking one non-zero vector onto another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source.shape != (3,) or target.shape != (3,) or source_norm < 1e-12 or target_norm < 1e-12:
        raise ValueError('source and target must be non-zero 3D vectors')
    source = source / source_norm
    target = target / target_norm
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3)
    if dot < -1.0 + 1e-9:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(source, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - dot) / np.dot(cross, cross))


@dataclass(frozen=True)
class AutomaticBoxSettings:
    """Clearances and release-volume height samples derived automatically."""

    wall_thickness_m: float
    edge_clearance_m: float
    release_heights_m: Tuple[float, ...]
    probe_length_m: float = PROBE_LENGTH_M
    probe_width_m: float = PROBE_WIDTH_M


@dataclass(frozen=True)
class BoxDropLayout:
    """Derived box geometry used by planning and RViz."""

    center: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray
    top_center: np.ndarray
    candidate_points: Tuple[np.ndarray, ...]
    release_volume_center: np.ndarray
    release_volume_dimensions: np.ndarray
    probe_axis_yaw_rad: float
    probe_axis_name: str
    usable_xy: np.ndarray
    probe_length_fits: bool
    probe_width_fits: bool
    probe_fits: bool
    settings: AutomaticBoxSettings


@dataclass(frozen=True)
class ProbeInsertion:
    """A tilted pose that lands one probe end inside the box."""

    probe_centre: np.ndarray     # probe geometric centre, box parent frame
    leading_end: np.ndarray      # the end that goes into the box
    trailing_end: np.ndarray     # the end that stays above the rim
    axis: np.ndarray             # unit vector, leading end -> trailing end
    tilt_rad: float              # elevation of ``axis`` above the box top plane
    depth_m: float               # how far the leading end sits below the rim
    clears_opening: bool         # True when the probe crosses the rim plane
                                 # inside the usable opening rather than
                                 # through a wall


def probe_tip_from_attached_geometry(
    probe_centre_in_link: np.ndarray,
    stl_long_axis_in_link: np.ndarray,
    probe_length_m: float,
    stl_fat_end_sign: int,
) -> np.ndarray:
    """Return the tapered endpoint in the planning-link frame.

    ``stl_long_axis_in_link`` is the signed STL +Z axis.  The fat-end sign is
    measured from the STL profile, so the tapered tip is always on the
    opposite end.  Keeping this point explicit lets the box fallback constrain
    the probe tip, rather than constraining the probe centre and inadvertently
    pulling the gripper into the box.
    """
    centre = np.asarray(probe_centre_in_link, dtype=np.float64)
    axis = np.asarray(stl_long_axis_in_link, dtype=np.float64)
    length = float(probe_length_m)
    fat_sign = int(stl_fat_end_sign)
    axis_norm = float(np.linalg.norm(axis))
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError('probe centre must be a finite 3D point')
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or axis_norm < 1e-12:
        raise ValueError('STL long axis must be a finite non-zero 3D vector')
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError('probe length must be positive and finite')
    if fat_sign not in (-1, 1):
        raise ValueError('STL fat-end sign must be -1 or +1')
    tip_direction = -float(fat_sign) * axis / axis_norm
    return centre + 0.5 * length * tip_direction


def compute_probe_insertion(
    layout: 'BoxDropLayout',
    tilt_rad: float,
    depth_m: float,
    axis_sign: float = 1.0,
    entry_offset_m: float = 0.0,
) -> ProbeInsertion:
    """Pose the probe leaning into the box with one end below the rim.

    The probe is longer than the box's usable horizontal axis, so it cannot be
    released flat through the opening. This returns a one-end-leading pose
    that can settle diagonally inside when the interior diagonal is long
    enough, rather than the horizontal pose used for a free drop.

    The leading end sits ``depth_m`` below the top plane, displaced
    ``entry_offset_m`` along the box's long axis so the probe leans across the
    opening instead of straight up out of the middle. ``axis_sign`` selects
    which way it leans — the caller picks whichever needs less wrist travel,
    which also decides which probe end leads.
    """
    R = np.asarray(layout.rotation, dtype=np.float64)
    long_index = 0 if layout.probe_axis_name == 'X' else 1
    u = R[:, long_index] * (1.0 if axis_sign >= 0.0 else -1.0)
    n = R[:, 2]

    tilt = float(np.clip(tilt_rad, math.radians(5.0), math.radians(85.0)))
    depth = max(0.0, float(depth_m))
    # Never drive the leading end deeper than the box interior.
    interior_depth = max(0.0, float(layout.dimensions[2]) - layout.settings.wall_thickness_m)
    depth = min(depth, interior_depth)

    leading_end = (
        np.asarray(layout.top_center, dtype=np.float64)
        - n * depth
        - u * float(entry_offset_m)
    )
    axis = math.cos(tilt) * u + math.sin(tilt) * n
    axis = axis / float(np.linalg.norm(axis))

    probe_length = float(layout.settings.probe_length_m)
    probe_centre = leading_end + axis * (0.5 * probe_length)
    trailing_end = leading_end + axis * probe_length

    # Where the probe crosses the rim plane: inside the usable opening means it
    # leaves through the mouth, outside means it would clip a wall.
    usable_half = 0.5 * float(np.max(np.maximum(layout.usable_xy, 0.0)))
    cross_along = float(np.dot(leading_end - layout.top_center, u)) + depth / math.tan(tilt)
    clears = bool(abs(cross_along) <= usable_half)

    return ProbeInsertion(
        probe_centre=probe_centre,
        leading_end=leading_end,
        trailing_end=trailing_end,
        axis=axis,
        tilt_rad=tilt,
        depth_m=depth,
        clears_opening=clears,
    )


def derive_automatic_box_settings(
    dimensions_xyz: Sequence[float],
    probe_length_m: float = PROBE_LENGTH_M,
    probe_width_m: float = PROBE_WIDTH_M,
) -> AutomaticBoxSettings:
    """Derive a safe overhead release-height range from box size.

    Probe dimensions are arguments rather than module constants so the caller
    can pass what probe.stl actually measures; a stale constant here silently
    mis-sizes clearances and the fit checks.
    """
    dimensions = np.asarray(dimensions_xyz, dtype=np.float64)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise ValueError('base_box_dimensions_xyz must contain three positive finite values')
    probe_length = float(probe_length_m)
    probe_width = float(probe_width_m)
    if not math.isfinite(probe_length) or probe_length <= 0.0:
        raise ValueError('probe_length_m must be positive and finite')
    if not math.isfinite(probe_width) or probe_width <= 0.0:
        raise ValueError('probe_width_m must be positive and finite')

    # Approximate an ordinary box wall as 5% of its smaller horizontal side,
    # bounded to realistic 5–15 mm material thickness. Edge clearance also
    # includes a quarter probe width so the released probe cannot graze a rim.
    wall = float(np.clip(0.05 * min(dimensions[0], dimensions[1]), 0.005, 0.015))
    edge = max(0.012, 0.25 * probe_width)

    # Keep the release close to the rim so the probe cannot bounce out after a
    # long fall. The small continuous Z band still gives IK useful freedom.
    first_height = max(0.035, 0.50 * probe_width + 0.010)
    release_span = max(0.020, min(0.035, 0.20 * float(dimensions[2])))
    heights = (first_height, first_height + release_span)
    return AutomaticBoxSettings(
        wall_thickness_m=wall,
        edge_clearance_m=edge,
        release_heights_m=heights,
        probe_length_m=probe_length,
        probe_width_m=probe_width,
    )


def compute_box_drop_layout(
    center_xyz: Sequence[float],
    dimensions_xyz: Sequence[float],
    rotation: np.ndarray,
    settings: AutomaticBoxSettings,
) -> BoxDropLayout:
    """Calculate a safe overhead release volume from box pose and dimensions.

    Dimensions are the outside box dimensions along its local XYZ axes. The
    returned volume lies above the top centre along local +Z. The longest
    usable box axis is returned so the held probe can be placed lengthwise;
    fit fields remain diagnostic and do not reject an overhead drop.
    """
    center = np.asarray(center_xyz, dtype=np.float64)
    dimensions = np.asarray(dimensions_xyz, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    heights = np.asarray(settings.release_heights_m, dtype=np.float64)

    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('base_box_center_xyz must contain three finite values')
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise ValueError('base_box_dimensions_xyz must contain three positive finite values')
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError('base-box rotation must be a finite 3x3 matrix')
    if heights.ndim != 1 or heights.size == 0 or not np.all(np.isfinite(heights)) or np.any(heights < 0.0):
        raise ValueError('automatically derived box release heights are invalid')

    wall = max(0.0, float(settings.wall_thickness_m))
    edge = max(0.0, float(settings.edge_clearance_m))
    probe_length = max(0.0, float(settings.probe_length_m))
    probe_width = max(0.0, float(settings.probe_width_m))
    usable_xy = dimensions[:2] - 2.0 * (wall + edge)

    axis_index = int(np.argmax(usable_xy))
    cross_index = 1 - axis_index
    axis_world = R[:, axis_index]
    axis_xy_norm = float(np.linalg.norm(axis_world[:2]))
    probe_axis_yaw = (
        float(np.arctan2(axis_world[1], axis_world[0]))
        if axis_xy_norm >= 1e-6 else 0.0
    )
    probe_length_fits = bool(usable_xy[axis_index] >= probe_length)
    probe_width_fits = bool(usable_xy[cross_index] >= probe_width)
    probe_fits = bool(probe_length_fits and probe_width_fits)

    top_center = center + R[:, 2] * (0.5 * dimensions[2])
    # Preserve the derived order while removing duplicate height samples.
    unique_heights = []
    for value in heights:
        h = float(value)
        if not any(abs(h - old) <= 1e-9 for old in unique_heights):
            unique_heights.append(h)
    candidate_points = tuple(top_center + R[:, 2] * h for h in unique_heights)
    min_height = min(unique_heights)
    max_height = max(unique_heights)
    release_volume_center = top_center + R[:, 2] * (0.5 * (min_height + max_height))
    # Only offer MoveIt the central part of the opening. Giving it the whole
    # usable opening made IK consistently choose the nearest edge, after which
    # the released probe could bounce outside. Orientation stays unconstrained,
    # but the probe centre must be well over the middle of the box.
    central_release_xy = np.clip(0.30 * np.maximum(usable_xy, 0.0), 0.020, 0.045)
    release_volume_dimensions = np.array([
        float(central_release_xy[0]),
        float(central_release_xy[1]),
        max(0.010, max_height - min_height),
    ], dtype=np.float64)

    return BoxDropLayout(
        center=center,
        dimensions=dimensions,
        rotation=R,
        top_center=top_center,
        candidate_points=candidate_points,
        release_volume_center=release_volume_center,
        release_volume_dimensions=release_volume_dimensions,
        probe_axis_yaw_rad=probe_axis_yaw,
        probe_axis_name='X' if axis_index == 0 else 'Y',
        usable_xy=usable_xy,
        probe_length_fits=probe_length_fits,
        probe_width_fits=probe_width_fits,
        probe_fits=probe_fits,
        settings=settings,
    )
