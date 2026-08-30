# aries_load_cells

The rover's three load cells, published as weights in kilograms.

| cell | where |
|---|---|
| `sand_box` | front-left deck box, the sand sample |
| `stone_box` | the box behind it, also on the left, the stone sample |
| `drill_container` | the drill's sample bin |

All three hang off the **drill/science Teensy** — the one board that also runs
the gripper servo, the stack light and the drill — so they arrive over the
micro-ROS link `aries_bringup`'s `aries_hardware.launch.py` already brings up.
There is no second board, no second agent, and nothing to add to
`devices.yaml`. This
package holds only the node that turns the firmware's raw converter counts into
kilograms, and the calibration it does it with.

## Topics

```
/load_cells/sand_box/weight              std_msgs/Float32   kilograms
/load_cells/sand_box/raw                 std_msgs/Int32     converter counts
/load_cells/stone_box/weight             "
/load_cells/stone_box/raw                "
/load_cells/drill_container/weight       what the bin's cell reads right now
/load_cells/drill_container/raw          "
/load_cells/drill_container/valid        std_msgs/Bool      is that a sample mass?
/load_cells/drill_container/weight_held  std_msgs/Float32   last reading taken while valid
/load_cells/status                       std_msgs/String    JSON, all of it at once
```

**All three read continuously**, at `publish_rate_hz` (10 Hz by default),
whatever the rover is doing — while a box is being filled, while the bin
travels, while the auger cuts. Nothing here waits for the rover to hold still.

A weight of `nan` means **no reading** — a stale link, or a cell sitting at its
converter's rail. It does not mean zero: zero is what an empty box reads, and
the two must not look alike. A stale or faulted cell publishes NaN at the same
steady rate rather than the topic going quiet, because a topic that stops looks
exactly like a node that died.

## Services

```
ros2 service call /load_cells/tare std_srvs/srv/Trigger              # all three
ros2 service call /load_cells/sand_box/tare std_srvs/srv/Trigger     # one
```

Tare zeroes the cell at whatever is on it now. The new offset is **logged, not
written back** — paste it into `config/load_cells.yaml` as `cell.<name>.offset`
to keep it across a restart.

## The drill bin's cell only carries the bin when it is parked

The bin rides its rails between `q = 0`, parked forward of the mast, and
`q = -0.1304`, back under the auger (`aries/urdf/drill.xacro`). Its load cell is
under the **parked** position. At the other end of that stroke the cell is
holding nothing at all.

That is a trap, because an unloaded cell does not read "no data". It reads
**zero** — which is exactly what a parked-and-empty bin reads. Published
unqualified, the operator would watch the bin empty itself the moment it slid
under the auger to collect, and fill back up on the way out.

So:

* `drill_container/weight` — what the cell reads, published continuously
  throughout the travel and the cut. Always honest, not always meaningful.
* `drill_container/valid` — whether that number is a sample mass.
* `drill_container/weight_held` — the last reading taken while it was. **This is
  the number you want** if you are asking "what is in the bin": it does not stop
  being true while the bin is away from its cell.

`valid` requires the bin parked, the bin stopped for `container_settle_s`, the
auger stopped for the same, fresh counts, and no cell fault. **None of that
gates the reading** — `drill_container/weight` keeps ticking at the full rate
through all of it. `valid` only says what the number means.

### Where the bin is, is dead-reckoned

From the rate commands on `/aries/drill_container_joint/cmd_vel`, the same model
`drill_joystick.py` runs its limit switches on. It has to be: the bin's actuator
is a DC motor with no encoder, and on the real rover `publish_wheel_joints.py`
publishes every drill joint at a constant `0.0` so MoveIt has a complete robot
state. Gate on *that* and the answer is "parked" forever — including while the
bin is under the auger, so the check would pass at exactly the moment it is
wrong.

Dead reckoning drifts. Two things bound it, and one retires it:

* Running the bin into the parked end **re-datums** the estimate, because the
  integrator clamps at the stop — homing against a hard stop, in software.
* `container_settle_s` disqualifies a bin still rocking on its rails.
* **The honest fix:** the bin's actuator has its own end switch at each end of
  its stroke. Put the forward one on a GPIO, publish a `std_msgs/Bool`, and name
  it in `parked_switch_topic`. It then overrides the dead reckoning completely
  and re-datums it on every trip to the park end. Nothing publishes one yet.

## The firmware contract

The firmware is `firmware/teensy_drill_sys`. `LoadCell` (`lib/drill/drill.h`) is
implemented, `main.cpp` constructs one per amplifier, and the HX711 pins are
assigned in `include/pins.h` — DT/SCK 17/16, 34/33, 32/31, **a private pair per
cell, not a shared clock**. See that project's `PINOUT.md`.

It publishes **one** topic:

```
load_cells/raw    std_msgs/Int32MultiArray, three elements,
                  in the order of `cells` in load_cells.yaml
```

declared with no leading slash under an empty namespace, exactly as
`stacklight_subscription` is, which resolves to `/load_cells/raw`.

**RELIABLE, on both ends.** The subscription in `load_cells.py` is created with
default rclpy QoS, which is reliable, so the firmware publisher is
`rclc_publisher_init_default` and not the best-effort one its 10 Hz rate would
otherwise argue for. A best-effort publisher against a reliable subscriber is an
incompatible pair: DDS makes no match, `ros2 topic info` shows one of each, and
nothing is ever delivered. Change one end only with the other.

**A cell that is not reporting sends the rail, not zero.** An amplifier that is
unplugged, unpowered or dead — and one whose last conversion is more than 500 ms
old — is published as `raw_min` (`-8388608`), which lands on the fault path
below and comes out as a NaN weight. It is deliberately not zero: zero is what
an *empty box* reads, so a dead cell would otherwise be indistinguishable from
one somebody had just emptied.

**The topic stays silent until at least one amplifier answers.** With no cells
fitted the board publishes nothing at all rather than three standing faults, and
`load_cells.py` says "no counts yet — is the Teensy's firmware publishing?".
Once any one cell is alive the full array goes out every cycle, rails included,
so one unplugged amplifier among three working ones is loud.

**Raw counts, not kilograms.** A cell's scale and zero are properties of the
cell, its amplifier and whatever it is bolted to; they are found by hanging
known masses off the rover, and they change whenever a box is unbolted or a cell
is swapped. Kept in `load_cells.yaml` they are a YAML edit and a relaunch — the
workspace is `--symlink-install`, so there is no rebuild either. Kept in the
sketch, every recalibration is a trip to the Arduino IDE with the rover open.

**One publisher, not three.** The entity budget is no longer a hard ceiling --
`micro_ros_platformio` builds the client library from source, so
`RMW_UXRCE_MAX_PUBLISHERS` is a number in `firmware/teensy_drill_sys/colcon.meta`
rather than something baked into a precompiled `libmicroros.a` the way
`micro_ros_arduino` shipped it. Raising it is an edit.

It is still one publisher, for the better reason: the three cells are polled
together on one pass of the firmware's loop and go out as one set, and splitting
them across three topics throws that away -- the subscriber then has to re-pair
samples that arrived together, and gets it wrong at exactly the moment the link
is slow.

Do note that the firmware spends seven subscriptions and four publishers
(`/gripper/state`, `drill/limits`, `drill/pin_scan`, and this one) against a
`colcon.meta` that allows eight and five. Adding to it means checking that file,
not just adding the entity — and after editing it, `./flash.sh --clean`, because
the micro-ROS library is cached and a config change to a cached library silently
does nothing.

If the firmware ends up with three separate `std_msgs/Int32` publishers anyway,
list them in `raw_topics` and this node reads those instead.

**The element order is the wire format.** The firmware sends no names to check
itself against. Swap two entries in `cells` and the sand box reports the stone.

## Seeing the topics without a Teensy

```bash
ros2 launch aries_load_cells load_cells.launch.py load_cell_source:=mock
```

`auto` (the rover default) **never** falls back to this. With no firmware
talking it advertises every topic, publishes nothing, and says so in the log. A
fabricated weight indistinguishable from a measured one is not something the
rover should ever emit by itself.

Mock counts are quantised to whole converter counts, as a real ADC is, so while
`scale` is still the `1.0` placeholder every mock weight lands on a whole
kilogram.

## Calibration

**Nothing is calibrated yet.** `scale` is counts per kilogram and ships as
`1.0`, a placeholder the node warns about at startup: until it is set, the
weight topics carry counts wearing a kilogram label.

Per cell:

1. Empty it, `ros2 service call /load_cells/<cell>/tare std_srvs/srv/Trigger`,
   and paste the logged offset into `offset`.
2. Put a known mass on it and read `ros2 topic echo /load_cells/<cell>/raw`.
3. `scale = (counts_with_mass - offset) / mass_in_kg`.
4. If the weight comes out negative, set `invert: true` rather than making
   `scale` negative — the sign is which way the cell is bolted in, and it
   belongs somewhere a reader will look for it.

An HX711 at gain 128 on a cell this size lands in the hundreds of thousands of
counts per kilogram, so expect a six-figure number.

## Where it starts

`full_hardware.launch.py` includes it by default (`use_load_cells:=false` to
skip, `load_cell_source:=mock` to fake it). Standalone:

```bash
ros2 launch aries_load_cells load_cells.launch.py
```
