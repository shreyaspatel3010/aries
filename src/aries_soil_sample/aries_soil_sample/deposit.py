"""Configurable deposit box, and the dump pose derived from its geometry.

Modelled on the probe task's box parameters (``base_box_center_xyz`` /
``base_box_dimensions_xyz`` / ``base_box_rpy`` in aries_vision_grasp): the box is
described once, in the planning frame, and every pose the node needs is computed
from that description rather than configured separately. Move the box and the
dump pose follows; there is no second number to keep in sync.

The box is an OPEN-TOPPED container, so "the rim" means its highest edge. The
bucket never enters it -- tipping only has to clear the rim -- which is why the
dump pose is expressed as a clearance above the rim rather than a depth inside.

Everything here is pure geometry with no ROS dependency, so the awkward parts
(a rotated box's rim height, whether the dump point is over the opening rather
than over a wall) are unit-testable.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation for an SDF/URDF-style roll-pitch-yaw triple (Rz @ Ry @ Rx)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class DepositBox:
    """An open-topped box in the planning frame.

    ``centre`` is the geometric centre of the box (as the probe task's
    ``base_box_center_xyz`` is), ``dimensions`` its outer X/Y/Z size, ``rpy`` its
    orientation. ``wall_thickness_m`` only narrows the usable opening; it does not
    change the outer size.
    """

    centre: np.ndarray
    dimensions: np.ndarray
    rpy: np.ndarray = None            # type: ignore[assignment]
    wall_thickness_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'centre',
                           np.asarray(self.centre, dtype=np.float64).reshape(3,))
        dims = np.abs(np.asarray(self.dimensions, dtype=np.float64).reshape(3,))
        object.__setattr__(self, 'dimensions', dims)
        rpy = np.zeros(3) if self.rpy is None else np.asarray(
            self.rpy, dtype=np.float64).reshape(3,)
        object.__setattr__(self, 'rpy', rpy)

    @property
    def rotation(self) -> np.ndarray:
        return rpy_matrix(*(float(v) for v in self.rpy))

    @property
    def corners(self) -> np.ndarray:
        """The 8 outer corners in the planning frame, ``(8, 3)``."""
        h = 0.5 * self.dimensions
        signs = np.array([[sx, sy, sz]
                          for sx in (-1.0, 1.0)
                          for sy in (-1.0, 1.0)
                          for sz in (-1.0, 1.0)])
        return self.centre + (signs * h) @ self.rotation.T

    @property
    def rim_z(self) -> float:
        """Height of the highest edge of the box.

        Taken as the maximum corner Z rather than ``centre_z + dz/2`` so a box
        with a non-zero roll or pitch reports the rim the bucket actually has to
        clear, not the rim it would have if it were level.
        """
        return float(self.corners[:, 2].max())

    @property
    def opening_half_extents(self) -> np.ndarray:
        """Usable half-width/half-depth of the opening, inside the walls."""
        h = 0.5 * self.dimensions[:2]
        return np.maximum(0.0, h - float(self.wall_thickness_m))

    def point_in_box_frame(self, point: Sequence[float]) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64).reshape(3,)
        return self.rotation.T @ (p - self.centre)

    def over_opening(self, point: Sequence[float], margin_m: float = 0.0) -> bool:
        """Is this point above the opening (not over a wall or outside)?

        Only the box's own X/Y matter -- height is the caller's business, since a
        dump happens above the rim by design.
        """
        local = self.point_in_box_frame(point)
        limits = self.opening_half_extents - max(0.0, float(margin_m))
        if np.any(limits <= 0.0):
            return False
        return bool(abs(local[0]) <= limits[0] and abs(local[1]) <= limits[1])

    def dump_contact(
        self,
        rim_clearance_m: float,
        offset_xy: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Bucket contact point for tipping the sample in.

        Sits ``rim_clearance_m`` above the rim, over the centre of the opening
        unless ``offset_xy`` shifts it (that offset is in the BOX frame, so it
        rotates with the box).
        """
        local_xy = (np.zeros(2) if offset_xy is None
                    else np.asarray(offset_xy, dtype=np.float64).reshape(2,))
        shift = self.rotation @ np.array([local_xy[0], local_xy[1], 0.0])
        return np.array([
            self.centre[0] + shift[0],
            self.centre[1] + shift[1],
            self.rim_z + float(rim_clearance_m),
        ], dtype=np.float64)

    def validate(self, rim_clearance_m: float,
                 offset_xy: Optional[Sequence[float]] = None,
                 margin_m: float = 0.0) -> Tuple[bool, str]:
        """Is this box + dump offset self-consistent? ``(ok, reason)``.

        Catches the configuration mistakes that would otherwise show up as a
        mystery IK failure or soil on the deck: a degenerate box, an opening the
        bucket cannot fit over, a dump point sitting over a wall, or a negative
        clearance that would drive the bucket into the box.
        """
        if np.any(self.dimensions <= 1e-6):
            return False, f'box has a zero dimension: {self.dimensions}'
        if np.any(self.opening_half_extents <= 0.0):
            return False, (f'wall_thickness {self.wall_thickness_m*1000:.0f}mm leaves no '
                           f'opening in a {self.dimensions[0]*1000:.0f}x'
                           f'{self.dimensions[1]*1000:.0f}mm box')
        if float(rim_clearance_m) < 0.0:
            return False, (f'rim clearance {float(rim_clearance_m)*1000:.0f}mm is negative; '
                           'the bucket would be inside the box, not above it')
        point = self.dump_contact(rim_clearance_m, offset_xy)
        if not self.over_opening(point, margin_m):
            local = self.point_in_box_frame(point)
            return False, (f'dump point is {local[0]*1000:+.0f},{local[1]*1000:+.0f}mm in the '
                           f'box frame, outside the usable opening '
                           f'{self.opening_half_extents[0]*1000:.0f}x'
                           f'{self.opening_half_extents[1]*1000:.0f}mm (margin '
                           f'{margin_m*1000:.0f}mm) -- the sample would miss the box')
        return True, (f'rim at z={self.rim_z:.3f}, dump contact at z={point[2]:.3f} '
                      f'({float(rim_clearance_m)*1000:.0f}mm clearance)')

    @property
    def summary(self) -> str:
        return (f'centre ({self.centre[0]:.3f},{self.centre[1]:.3f},{self.centre[2]:.3f}) '
                f'size {self.dimensions[0]*1000:.0f}x{self.dimensions[1]*1000:.0f}x'
                f'{self.dimensions[2]*1000:.0f}mm rim z={self.rim_z:.3f}')
