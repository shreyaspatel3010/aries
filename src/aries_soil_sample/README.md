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

## The octomap cannot be checked against soil you intend to dig

Measured on the live sim at the surveyed site, with the 30° tilted entry, per
waypoint:

| waypoint | contact z | reachable | with collision checking |
|---|---|---|---|
| approach (60 mm above) | −0.107 | 4/4 | **0/4** |
| entry (at surface) | −0.159 | 4/4 | **0/4** |
| penetrate (30 mm deep) | −0.185 | 4/4 | **0/4** |

Every pose is kinematically fine; all are rejected by collision checking. The
collider is **the ground** — the octomap models the terrain the scoop exists to
dig into, so a collision-free path into it cannot exist by definition.

So `octomap_disable_during_scoop` (default **true**) suppresses octomap collision
checking for the scoop, using the same mechanism the grasp package uses for the
held probe: the ACM **default** entry for `<octomap>`, one flag that makes it
allowed against everything rather than pairwise entries that need re-applying
whenever a link or object appears. The sensor pipeline keeps running and the
octomap keeps building — MoveIt just stops colliding against it.

It covers the whole scoop, not just the strokes below the surface: the approach
60 mm *above* the ground is 0/4 too, because the 100 mm bucket shells reach into
ground voxels well before the contact point does. It is restored in a `finally`,
so it comes back on every exit path — including the buried-bucket abort.

**The trade-off, stated plainly:** while off, the arm will not avoid obstacles
that exist only in the octomap. What replaces it for the scoop is narrower but
real: the site was surveyed for roughness and slope, the waypoints are a short
straight line whose geometry follows from that survey, every one is bounded by
`absolute_min_contact_z`, and all were IK pre-screened first. Set
`octomap_disable_during_scoop: false` to keep checking on and accept that scoops
will not plan.

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

## Not implemented yet

- **Sample deposit.** The cycle ends holding the sample at the transport
  posture. Depositing needs a reachable container: the existing base box sits in
  the arm's self-collision zone (measured — no rim height works in that column),
  so a deposit target has to be moved forward and lowered to a ≤0.22 m rim
  before any deposit sequence is worth writing.
- **Hold verification during transport.** Soil can dribble out of a bucket; the
  divot measurement only proves capture at the moment of the scoop.
- **Multiple samples / caching.** One scoop per trigger, no sample bookkeeping.
