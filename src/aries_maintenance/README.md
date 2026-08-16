# Aries maintenance-panel operator

This node uses both rover and gripper color cameras to detect the maintenance
panel's ArUco markers. Three unique IDs are randomly selected from 11, 13, 14
and 15 at build time and assigned to the fixed top-left, top-right and
bottom-left positions; the panel has no bottom-right marker. Recent agreeing
observations are transformed into `base_link` and fused into one panel pose. A
weaker one-tag frame cannot overwrite an accepted multi-tag pose. By default
both cameras must agree, with at least two unique marker IDs between them.
Camera extrinsics are looked up at each image's acquisition timestamp, which
matters for the moving gripper camera. The node then requires 15 fused samples
over at least 0.5 s to stay within 12 mm/1.5 degrees before averaging and
latching them. Arm occlusion cannot corrupt that accepted consensus pose.

The prop builder chooses a fresh assignment on each build. Pass
`--panel-marker-seed <integer>` to reproduce the same assignment in a practice
world.

Both registered depth streams are also used when available. Depth inside the
inner ArUco area back-projects each known marker centre into camera 3D and
refines PnP translation; RGB marker corners continue to determine rotation.
This works when the rover and gripper cameras see different marker IDs and
falls back to RGB-only consensus if a synchronized depth frame is unavailable.

MoveGroup plans from the current arm posture to a clear standoff. Only the short
standoff-to-contact motion is Cartesian, so the arm does not require a complete
Cartesian path from its arbitrary starting posture. That final stroke starts
from MoveIt's measured current state. Collision checking is disabled only for
this intentional contact phase because the panel has no Allowed Collision
Matrix entry; free-space standoff planning remains collision-aware.

MoveIt supplies the collision-checked and time-parameterized trajectories, but
the operator sends them directly to
`/rebel_arm_trajectory_controller/follow_joint_trajectory`. This prevents an
unrelated overlapping MoveGroup request (such as joystick `pick_home`) from
wedging MoveIt's global trajectory manager. If another controller action really
is active, the panel operator waits for it to finish rather than preempting it.

Waypoint geometry is calibrated from both models. Each target is placed on the
outermost surface of its movable panel-control mesh, rather than on the buried
joint pivot. This corrects the old depth by 8.6 mm for MCBs, 21.8 mm for the
disconnects, and about 36.8 mm for rotary selectors. It also accounts for the
maintenance fingertips extending 65 mm past `gripper_tcp`, so the TCP remains
outside the panel when the physical fingertip reaches the modeled surface.
For closed-jaw MCB and button pushes, the leading fingertip surface is farther
forward than the jaw meeting point, so those actions use the model-derived
84 mm offset instead. MCB ON motion is generated in the planning frame by
projecting `+Z` onto the console face; it therefore cannot become a downward
stroke because of marker or panel-frame orientation.

Gripper commands go through
`/rebel_gripper_controller/joint_trajectory`, the same active ros2_control
controller used by joystick teleoperation. The v2 maintenance fingertips close
at `-0.03 rad`; the generic bucket endpoint (`+0.07 rad`) must not be used for
this tool. Commands deliberately have a zero trajectory timestamp, meaning
"start immediately" even when the standalone panel node and Gazebo controller
are using different clocks.

Before planning, the operator checks both mechanically equivalent wrist poses
(0 and 180 degrees about tool +Z) with MoveIt's collision-aware IK service.
This preserves the approach direction and jaw line while allowing the bounded
Rebel wrist to use its reachable IK family. The successful joint solution is
sent directly to MoveGroup, avoiding pose-goal sampling failures.

Cartesian contact is limited to 30 mm/s. After every MoveGroup or Cartesian
execution, the node compares measured `base_link -> gripper_tcp` TF with the
commanded target; a controller result alone is not treated as proof of motion.

## Run controls selected in YAML

1. In `config/panel_tasks.yaml`, change only the controls you want to operate to
   `true`. The file is reloaded from disk for every
   `/panel/operate_enabled true` trigger, so the node does not need restarting.
2. Start the arm/MoveIt and both cameras, then launch the operator:

   ```bash
   ros2 launch aries_maintenance panel_operator.launch.py
   ```

3. Wait for `panel localised` in the log. The two cameras must together see at
   least two unique IDs from 11, 13, 14 and 15; they do not need to see the same
   marker.
4. Trigger the configured sequence:

   ```bash
   ros2 topic pub --once /panel/operate_enabled std_msgs/msg/Bool '{data: true}'
   ```

The two cameras may observe different marker IDs; the union must contain at
least two unique IDs. If the trigger arrives before localization, it is queued
and starts automatically as soon as a valid fused pose is latched.

Every `/panel/operate_enabled true` trigger first validates and snapshots the
latest YAML, then discards the previous pose latch and requires new agreeing
observations from both cameras. After recalibration, that new pose remains
latched for the complete sequence so marker occlusion by the arm does not
interrupt operation. A malformed or incomplete YAML starts no motion. A new
trigger is rejected while an existing panel sequence is moving the arm.

`false` is ignored. Progress and errors are published on `/panel/status`. The
sequence follows the order in `panel_task.json`. If one control fails, its error
is recorded and the operator continues with every remaining enabled control;
the final status reports how many succeeded and lists any failures.
MCBs are named `mcb_0` through `mcb_13` from the physical left edge of the
panel to the right edge. For an MCB, `true` always commands an upward flick
along `console_up_slope` into the ON position. The tool then retreats straight
out from the upper endpoint so it cannot brush the breaker back down.

You can still operate one named control without editing YAML:

```bash
ros2 topic pub --once /panel/operate std_msgs/msg/String '{data: mcb_3}'
```

The named groups `all_breakers`, `all_buttons`, and `all_switches` are also
accepted on `/panel/operate`.

At startup the log prints the resolved YAML path, its modification time, and
the enabled control names. A different file can be selected without rebuilding:

```bash
ros2 launch aries_maintenance panel_operator.launch.py \
  config_file:=/absolute/path/to/panel_tasks.yaml
```
