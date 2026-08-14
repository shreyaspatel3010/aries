# ERC 2026 Mars Yard assets

This directory contains the task textures and lightweight prop meshes used by
`worlds/marsyard2026.sdf`. The drivable terrain images live in `models/dem`
because they are shared with the terrain tooling.

## Authoritative data

- Terrain source: `Model3D_mesh1.ply` from `2026_MarsYard_3D.zip`, rasterised
  and levelled against the organiser's surveyed coordinates.
- Terrain visual and collision: the same 257 x 257 heightmap (17.2 cm per
  sample), so the robot cannot pass through a surface that only exists in the
  renderer. Gazebo maps the archive's 4096 x 4096 georeferenced
  `orthophoto.tif` onto it for the full-colour visual.
- A 1025 x 1025 heightmap (4.30 cm per sample) is retained as the high-resolution
  generated DEM. Gazebo's current Ogre2 / DART combination is most reliable
  when the shared 257 grid is used for both rendering and contact.
- S1-S9, L1-L15, W1-W9 and P1: exact X, Y and H values from
  `Coordinates_MarsYard2026.txt`. The source file lists Y before X.
- Landmark IDs 51-65 and task geometry: `[ERC 2026] MY Update Report Rev.1.pdf`.

Starts, waypoints and P1 are SDF frames, not synthetic visible markers. The
report defines their coordinates but does not show coloured posts or a painted
sampling square. Landmarks are physical, collidable ArUco boards.

## One documented practice assumption

The report identifies S8 as the maintenance-task start but gives no surveyed
pose for the panel. The world therefore puts the panel 2.5 m from S8 toward the
yard centre and points its face toward S8. Update that one pose if the organiser
publishes a placement drawing. The drone cage is intentionally not in this
world: the report explicitly places it outside the Mars Yard and supplies no
common coordinate transform.

## Launch

From the workspace root after building and sourcing:

```bash
ros2 launch aries my_robot.launch.py \
  world:=marsyard2026.sdf spawn_x:=0 spawn_y:=0 spawn_z:=0.1 \
  spawn_yaw:=1.5708
```

This starts at S1 and faces along the yard's positive Y axis. Other official
start poses can be copied directly from the named frames in the world file.

Regenerate terrain and landmark assets with
`scripts/build_marsyard2026_terrain.py`; its `--source` argument is the
extracted `2026_MarsYard_3D_Model` directory. A textured comparison GLB can be
exported with `--visual-faces`, but the world intentionally uses the shared
heightfield for exact visual-to-collision agreement.
