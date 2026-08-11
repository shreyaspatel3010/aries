# aries_soil_sample

Autonomous soil-sample collection from the ground with the Aries **bucket**
fingertip. Sibling to `aries_vision_grasp`, but it solves a different perception
problem and needs no trained model.

```bash
colcon build --symlink-install --packages-up-to aries_soil_sample
source install/setup.bash

# Simulation: the soil-sampling world (loose grains + a deposit box, no probe).
ros2 launch aries my_robot.launch.py world:=soil_world.sdf finger_type:=bucket

# Against Gazebo (use_sim_time must be explicit, as in aries_vision_grasp):
ros2 launch aries_soil_sample soil_sample.launch.py use_sim_time:=true

# Look at the ground without moving anything:
ros2 service call /soil_sample_node/survey std_srvs/srv/Trigger

# Run one full scoop cycle:
ros2 service call /soil_sample_node/scoop std_srvs/srv/Trigger
```

The node does **not** start scooping on launch. It drives a gripper into the
ground, so a cycle runs only when triggered (or with `auto_start:=true`).

## Why there is no YOLO model here

There is no trained detector for soil, and this task does not need one.

A probe is an **object**: you must recognise which pixels are the thing before
you can grasp it. Soil is **terrain**: any patch of ground that is flat enough,
level enough, reachable and unobstructed is a valid place to put the bucket.
That is a geometry question, and the depth image answers it directly:

1. back-project the depth image, keep points inside a work region in `base_link`;
2. reduce to a 2.5-D **height map**, one surface height per grid cell (soil has
   no overhangs, so nothing is lost);
3. slide the bucket footprint over the map and score each placement on
   **roughness** (rms residual to a fitted plane), **slope** and **coverage**.

Roughness is the primary safety gate. A buried rock or the probe standing in the
sand shows up as roughness long before it tilts the fitted plane, so a patch is
rejected for being *lumpy* before it is ever rejected for being *sloped*.

Nothing in this package imports `ultralytics` or `torch`, and it needs no GPU.

## How a scoop is verified

The two signals a normal pick would use are both unavailable on this robot:

- the gripper reports its **command**, not a measured position (the Teensy runs
  `USE_SERVO_FEEDBACK = false`), so "the jaws stopped early" cannot be observed;
- MoveIt's padded self-filter blanks the wrist camera's entire near field, so
  nothing can see inside the jaws.

So the ground is the sensor. A scoop that collected soil leaves a hole: survey
the site before, scoop, re-survey from the same posture, and difference the two
height maps (`terrain.divot_volume`). The ground at 0.4–0.5 m is comfortably
inside the camera's working range, which is exactly where the jaw volume is not.

Only downward changes count — soil pushed up at the rim of the hole is displaced,
not collected, and letting it cancel the hole would hide a good scoop.

The verdict is deliberately asymmetric, like the probe grasp's: `CAPTURED` needs
enough measured volume, `EMPTY` needs the ground to be *measurably untouched*,
and everything between is `UNKNOWN` — which the caller treats as a miss and
retries, never as success.

## The scoop itself

```
approach -> entry (at the surface) -> penetrate (below it) -> CLOSE -> extract
```

Everything runs along the surface normal, and extraction retraces the
penetration in reverse along the same axis rather than lifting world-vertical —
a vertical lift out of an angled channel levers the tool against the material.
That lesson is inherited from the probe grasp.

Penetration depth is clamped twice: by `scoop_max_depth_m`, and by the bucket's
own length less a margin (past that the four-bar linkage itself enters the soil).
Every waypoint is additionally checked against `absolute_min_contact_z`, a hard
floor that no perception result can talk the bucket below.

The whole path is IK pre-screened before the first motion: a scoop that fails
halfway leaves the bucket buried. Once the stroke is in the soil, extraction runs
even if the close command fails — the one thing worse than an empty bucket is a
buried one.

## Layout

- `scripts/soil_sample_node.py` — perception + scoop state machine, services.
- `aries_soil_sample/terrain.py` — height map, plane fitting, site selection,
  divot measurement. Pure NumPy, no ROS.
- `aries_soil_sample/scoop.py` — scoop waypoints, tool frame, contact→link
  conversion, capture verdict. Pure NumPy, no ROS.
- `config/soil_sample_params.yaml` — the full tuning set, commented.
- `test/` — 39 unit tests over both library modules.

## Reused from `aries_vision_grasp`

Not duplicated:

- `image_bridge.NumpyImageBridge` — cv_bridge segfaults under NumPy 2.x.
- `grasp_verification.backproject_depth` — depth image → XYZ.
- `fourbar` — the **bucket** rows are the field-calibrated originals, so the jaw
  gap and contact-point tables are already right for this task.

`finger_type` must be `bucket` and must match the URDF the robot was launched
with: the four-bar contact point differs by up to 23 mm between the three jaws,
which is a 23 mm depth error on every scoop.

## Calibrate before trusting

- **`min_sample_volume_m3`** (default 20 cm³) is derived from the fingertip mesh
  bounding box via `scoop.nominal_bucket_capacity_m3`, not from a measurement.
  Weigh or water-fill a real scoop and set it from that.
- **`work_region_*`** defaults come from measured rover geometry (floor grasp
  near x=0.55, ground near z=−0.07 in `base_link`). Everything downstream trusts
  this box to contain only ground — verify it against your terrain first.
- **`bucket_entry_q`** (−0.34 → 68.5 mm gap) is chosen so the shells straddle a
  50 mm soil column. Widen it for coarse material, but not to full open, where
  the bucket presents no cutting edge.

## Collision safety while digging

MoveIt does not build a live occupancy map from the cameras. That is deliberate
for digging: the measured ground is the material the bucket must enter, so
treating it as an obstacle would reject every scoop. Safety instead comes from
the surveyed work region, slope and roughness checks, short geometry-derived
strokes, the `absolute_min_contact_z` bound, IK pre-screening, robot
self-collision, and explicit collision objects.

## The test world: `aries/worlds/soil_world.sdf`

Derived from `sandbox_world.sdf` — same physics, plugins, lighting and ground
plane — with the probe and its socketed sand block removed and replaced by:

- **`soil_tray`** at world (0.42, −0.11): 150 mm square, **25 mm rim**, kept low
  so a ~30° bucket entry clears it.
- **36 × `soil_grain_NN`**: loose 12 mm cubes, 3 g each, µ=1.1. Granular rather
  than one solid block on purpose — a scoop has to displace and carry individual
  grains, and grains are what visibly ends up in the box. Every grain is a
  dynamic body, so RTF pays for the count.
- **`deposit_box`** at world (0.42, +0.16): open box, 120 mm square, **55 mm
  rim**, sitting **on the ground**, not on the rover deck. That is deliberate: a
  deck-mounted box is unreachable — measured on this arm, a tip-down release over
  the base-box column has no collision-free IK at *any* rim height, while a rim
  at or below ~0.22 m out in front does.

The world is still named `empty` internally **on purpose**: `gazebo_bridge.yaml`
and `rover_gazebo_bridge.yaml` hardcode `/world/empty/pose/info` and
`/world/empty/light_config`, and renaming it silently breaks both bridges.

`my_robot.launch.py` now takes `world:=` (choices: `sandbox_world.sdf`,
`soil_world.sdf`), defaulting to the sandbox so existing workflows are unchanged.

## Deposit into the rover box

After a `CAPTURED` verdict the sample is tipped into the box carried on the
rover — the same target the probe task uses (`base_box_center_xyz =
[0.003, 0.215, 0.287]`, rim at z=0.362). Measured over that column with the
bucket mouth pointing straight down, 4 wrist rolls:

| bucket contact z | vs rim | link z | reach | collision-free |
|---|---|---|---|---|
| 0.380 | +18 mm | 0.594 | 4/4 | 3/4 |
| **0.400** | **+38 mm** | **0.614** | **4/4** | **4/4** |
| 0.420 | +58 mm | 0.634 | 4/4 | 2/4 |
| 0.450 | +88 mm | 0.664 | 1/4 | 0/4 |

Two results worth keeping:

- **The bucket can use the rover box even though the probe cannot.** The probe
  needed the gripper link at z=0.787 (200 mm of probe hanging below a 270 mm
  contact); the bucket needs 0.614 — 173 mm less — because nothing protrudes
  past the jaws.
- **The dump must be vertical.** At a 30° tilt reach is still fine but collisions
  reject it at every height (0/4): leaning over the box re-enters the arm fold
  that blocked the probe. That is the *opposite* of the scoop, which cannot be
  vertical.

Dumping only has to clear the rim, so the bucket never enters the box. Opening
the jaws over it is the whole operation. If capture succeeded but the dump pose
is unreachable the cycle reports that split explicitly and leaves the bucket
closed, still holding the sample.

### Setting the box

The box is described once and every pose is derived from it, so moving the box
moves the dump automatically:

```yaml
deposit_box_center_xyz: [0.003, 0.215, 0.287]   # GEOMETRIC centre, not the floor
deposit_box_dimensions_xyz: [0.14, 0.20, 0.15]  # outer size -> rim at 0.362
deposit_box_rpy: [0.0, 0.0, 0.0]
deposit_box_wall_thickness_m: 0.006             # narrows the opening only
deposit_rim_clearance_m: 0.038                  # jaws open this far above the RIM
deposit_offset_xy: [0.0, 0.0]                   # shift within the opening, BOX frame
deposit_edge_margin_m: 0.015                    # keep the dump off the rim
```

`rim_z` is the highest **corner**, so a box with roll or pitch reports the edge
the bucket actually has to clear rather than the one it would have if level.
`deposit_offset_xy` rotates with `deposit_box_rpy`.

The configuration is validated at startup, not mid-cycle — a box you cannot dump
into is a config error, and finding it with a full bucket is the expensive way:

```
[deposit] box centre (0.003,0.215,0.287) size 140x200x150mm rim z=0.362;
          rim at z=0.362, dump contact at z=0.400 (38mm clearance)

[deposit] deposit box is misconfigured: dump point is +90,+0mm in the box frame,
          outside the usable opening 64x94mm (margin 15mm) -- the sample would
          miss the box. Deposit is DISABLED for this run.
```

`publish_deposit_box_marker` draws the box outline and the dump point on
`/soil_sample/markers`. Set the box by eye with it: if the orange dot is not
floating just above the opening, the configuration is wrong.

**Still missing: a physical container.** The probe task's base box is a
*configured region* only — there is no box model in the world and no box link on
the rover, so dumped grains currently land on the deck mesh. A world-file box
cannot fix this: the rover's world pose varies between runs (−0.32 vs −0.01 m
observed), so a fixed pose will not track the deck. The container has to be a
link fixed to `base_link` in the rover URDF, which also changes the probe task's
collision model — not done unilaterally.

## Not implemented yet
- **Hold verification during transport.** Soil can dribble out of a bucket; the
  divot measurement only proves capture at the moment of the scoop.
- **Multiple samples / caching.** One scoop per trigger, no sample bookkeeping.
