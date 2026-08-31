# HSM Aries — drill / science firmware (Teensy 4.1)

Firmware for the Teensy 4.1 that runs the **drill, the gripper servo, the
front-left container lid and the mast stack light** — one board, one micro-ROS
serial link, one `micro_ros_agent`.

This replaces `firmware/legacy/teensy_gripper/teensy_gripper.ino`. If you are
used to the Arduino IDE and `.ino` files, see [Why it looks
different](#why-it-looks-different) below.

- **Pin map and wiring:** [`PINOUT.md`](PINOUT.md) — read this before powering
  the drill. Six pins are still proposals, and the pump's three are upstream's
  numbers rather than anything checked against this loom.
- **Host side:** `aries_bringup` (`drill_driver.py`, `stacklight.py`),
  `aries_moveit/teensy_gripper_hardware` (the gripper), `aries_load_cells`
  (scales the counts), `aries_teleop` (`drill_joystick.py`, which drives the
  feed, bin, auger, lid and pump from the pad).

---

## Topics

Everything is under an empty namespace, so a name without a leading slash
resolves to one.

| Topic | Type | Dir | QoS | Meaning |
|---|---|---|---|---|
| `/gripper/cmd` | `Float32` | in | best effort | Jaw target, 0–1 |
| `/gripper/state` | `Float32` | out | best effort | Jaw command echoed, 100 Hz |
| `drill/limits` | `UInt8` | out | reliable | Feed switches: bit0 bottom, bit1 top; bit2/bit3 = the PWM sign the gate believes drives INTO top/bottom (set = positive). On change + 2 Hz |
| `stacklight_subscription` | `UInt8` | in | reliable | 1 red, 2 yellow, 3 green, 4 off |
| `motor1/cmd_speed` | `Int32` | in | best effort | Auger, −255…255 |
| `motor2/cmd_speed` | `Int32` | in | best effort | Feed carriage, −255…255 |
| `linact/state` | `UInt8` | in | reliable | 1 extend, 2 retract, 3/4 home, 0 stop |
| `linact/cext` | `Float32` | in | reliable | Bin travel, signed mm |
| `sand_box/lid/cmd` | `Float32` | in | reliable | Lid **speed**, −1…1, 0 = stop |
| `pump/state` | `UInt8` | in | reliable | 1 release, 2 draw, 3/4 home, 5 home-then-draw, 0 stop |
| `pump/purge` | `Float32` | in | reliable | **Seconds** of reverse run to empty the tube. ≤ 0 and NaN stop; clamped to 120 s |
| `load_cells/raw` | `Int32MultiArray` | out | reliable | Three RAW converter counts, 10 Hz, in the order sand box / stone box / bin |
| `drill/pin_scan` | `UInt64` | out | reliable | Diagnostic: bit N set = pin N reads LOW |

Nine subscriptions and four publishers, against a `colcon.meta` that allows ten
and five. **There is one subscription slot left.** Check that file before adding
an eleventh — and remember that editing it does nothing until the cached
micro-ROS library is rebuilt, which `./flash.sh` does by default.

**The load cells send counts, not kilograms.** Scale, offset and tare live in
`aries_load_cells`' YAML on the host, so a recalibration is an edit and a
relaunch rather than a reflash with the rover open. A cell that is not answering
sends the converter's negative rail (`-8388608`), never zero — zero is what an
empty box reads.

### Three of these names are the workspace's, not this firmware's

`/gripper/cmd`, `/gripper/state` and `stacklight_subscription` were renamed from
the firmware's original `gservo/state` and `stalig/state`, and `/gripper/state`
was **added** — the delivered firmware had no publisher at all. That is not a
naming preference:

- `teensy_gripper_system.cpp` swallows *every* gripper command for the rest of
  the session if `/gripper/state` never carries a message. The gripper would
  silently never move, with `ros2 topic list` looking perfectly healthy.
- The stack light's codes arrived **transposed** — 1 = green, 2 = yellow,
  3 = red. `stacklight.py` shows red for an e-stop, a drive fault, a halt, and
  for the `unknown` it holds before it can see the rover at all. Under the
  original numbering every one of those lit **green**.

`test_stacklight.py` parses the enum out of `lib/emg/emg.h` and fails if the two
sides drift apart again.

### QoS is not decoration

`/gripper/state` publishes at 100 Hz over an XRCE serial stream. On a *reliable*
stream every sample must be acknowledged, and when the window fills before the
ACKs return the publisher stalls and retransmits the same frame instead of
sending new ones — which the host reports as `No /gripper/state for 2.0 s`.

A `BEST_EFFORT` publisher and a `RELIABLE` subscriber are an **incompatible
pair**: DDS makes no match at all, the topic lists fine, and nothing is ever
delivered. Both ends must agree. The host side is best-effort to match.

The two motor topics are best-effort for the same reason — a 30 Hz stream where
the newest value supersedes the last. The two `linact` topics are reliable
because they are one-shot events that nothing re-sends.

---

## Build and flash

```bash
pipx install platformio     # once
./flash.sh                  # wipe the cache, rebuild from scratch, flash
```

**`./flash.sh` wipes `.pio` entirely on every run** — build tree, resolved
`lib_deps`, and the compiled micro-ROS library with it — and builds from
nothing. That is the whole ~380 MB and it takes several minutes, because
micro-ROS is compiled from source.

That is the default on purpose. The two failure modes are not symmetric: a
needless rebuild costs minutes, while a build against a stale cached micro-ROS
library flashes a board that reports success and then fails in a way that looks
exactly like broken firmware (see `colcon.meta`, below).

- `./flash.sh --fast` keeps the cache — incremental, seconds. Use it when you
  have changed only `main.cpp` or the library sources and touched no build
  configuration.
- `./flash.sh --build-only` wipes and compiles without touching the board.
- `./flash.sh -v` shows the full build output.

**What a flash reaches on the board:** all of the program. HalfKay erases the
program flash as it writes, so nothing of the previous firmware survives —
there is no "leftover firmware" state to clear separately, and
`teensy_loader_cli` has no erase flag because it does not need one. It does not
clear the emulated EEPROM, which is irrelevant here: this firmware never
includes `<EEPROM.h>` and holds no persistent state at all. A true factory erase
is the 15-second press of the physical button, which is a bootloader feature and
cannot be scripted.

`~/.platformio` itself is **not** touched — the ARM toolchain and the Teensy
loader live there, and they are inputs, not build output.

### Use the script, not bare `pio`

`pio run -t upload` on its own does **not** work on this machine, for four
reasons `flash.sh` handles and documents at the top of itself:

1. **ROS 2 poisons the build.** `~/.bashrc` sources
   `/opt/ros/jazzy/setup.bash`, and micro_ros_platformio builds a *separate*
   bare-metal ROS 2 that finds the desktop x86 one instead:
   `CMake Error ... No 'rosidl_typesupport_cpp' found`. Unsetting
   `AMENT_PREFIX_PATH` / `CMAKE_PREFIX_PATH` is **not enough** — CMake also
   searches every `PATH` entry with `bin/` stripped, and `/opt/ros/jazzy/bin`
   is on `PATH`, so it still resolves `/opt/ros/jazzy/share`. `PATH` itself
   has to be filtered.
2. **micro_ros_platformio hardcodes PlatformIO's virtualenv** at
   `$PROJECT_CORE_DIR/penv/bin/activate`, which only exists for an
   installer-script install. A `pipx` install puts it elsewhere and the build
   dies with `.: cannot open ~/.platformio/penv/bin/activate`.
3. **The micro-ROS library is cached** and is not rebuilt when `colcon.meta` or
   `build_flags` change — the edit silently does nothing, and the board is
   flashed against the old entity ceilings and the old compiler flags while the
   build reports success. `pio run -t clean_microros` is the advertised fix and
   it is only a partial one: it drops the micro-ROS archive and leaves
   `.pio/libdeps`, the object files and scons's dependency database in place.
   `flash.sh` deletes the directory.
4. **The first upload usually loses a USB race.** The loader reboots the board
   into HalfKay, which re-enumerates as a different USB device; writing too
   early gives `Found device but unable to open` / `error writing to Teensy`
   even though permissions are fine. The script retries.

If the loader never finds the board at all, press the physical button on the
Teensy to force it into the bootloader and run the script again.

The VS Code **PlatformIO IDE** extension is fine for editing, but its toolbar
Build/Upload buttons inherit the terminal's ROS environment and hit problem 1.

### Running it

`aries_hardware.launch.py` starts the agent. By hand:

```bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
```

**Do not raise that baud.** Linux `speed_t` values are encodings, not bit rates,
and the largest valid one is `B4000000`. Anything above it is rejected by
`cfsetospeed` with `EINVAL`, which the agent does not check — leaving the port
speed unset. The link is USB CDC, so the device ignores baud anyway.

### Two more things that will bite

**`colcon.meta` is load-bearing.** `RMW_UXRCE_MAX_SUBSCRIPTIONS` defaults to
**5** in micro_ros_platformio, and this firmware creates **eight**;
`RMW_UXRCE_MAX_PUBLISHERS` defaults to **4** and it creates **four**, raised to
5 so the next one added does not fail at the ceiling. Without the raised limits
the over-the-line `rclc_*_init` fails, `create_entities()` bails, and the board
sits in `WAITING_AGENT` forever while USB stays enumerated — indistinguishable
from an unflashed Teensy. Editing it does nothing until the cached library is
rebuilt, which a default `./flash.sh` does.

**Nothing may write to `Serial`.** It is the micro-ROS transport. The only
status channel this board has is the LED on pin 13:

| Pattern | Meaning |
|---|---|
| Fast blink, 100 ms | A pin is still `PIN_UNASSIGNED` |
| Slow blink, 500 ms | Waiting for the agent |
| Solid | Connected and driving |

---

## Layout

```
flash.sh            build + flash, with the environment traps handled
platformio.ini      build config, distro, the colcon.meta pointer
colcon.meta         micro-ROS entity budget — see above
include/pins.h      EVERY pin number, and nothing else has one
src/main.cpp        micro-ROS entities, the reconnect state machine, callbacks
lib/drill/          motors, servos, limit switches, load-cell stub
lib/emg/            stack light
extra_packages/     rover_sensor_interfaces — a custom message, unused so far
```

`extra_packages/rover_sensor_interfaces` defines a `SoilTemperature` message for
a DS18B20. Nothing includes or publishes it yet; it is scaffolding for the
science module and is built into the client library by micro_ros_platformio if
it is left here.

---

## Why it looks different

**No `.ino`.** Arduino sketches hide a lot of standard C++ structure. This is
plain C++: `src/main.cpp` holds the entry points, and the hardware is split into
`lib/` with headers, so a class can be read without scrolling past everything
else on the board.

**No manual library installs.** `platformio.ini` pins the dependencies and
PlatformIO fetches them. Nobody has to hunt for the right `.zip`.

**Built from source, not linked against a blob.** `micro_ros_arduino` shipped
`libmicroros.a` **precompiled**, which meant a library built against an older
newlib failed to link against a current Teensy core with `undefined reference
to '__locale_ctype_ptr'` — and made the retired sketch unbuildable until it was
replaced. `micro_ros_platformio` compiles the client against the same toolchain
as the rest of the firmware, so that whole class of problem is gone.

---

## What is not done

- **Six pins are proposals** — the bin actuator's three, both limit switches,
  and the container lid servo. `PINOUT.md` says which and why. Confirm them
  before powering the drill.
- **The lid servo's neutral pulse is a guess.** `LID_SERVO_NEUTRAL_US` is 1500,
  which is where hobby servos cluster and not where any particular one stops.
  It is a continuous-rotation servo, so a neutral that is even slightly off is a
  lid that creeps with the stick released. Trim it on the bench — publish `0.0`
  to `sand_box/lid/cmd`, watch the horn, move the constant 10 µs at a time.
  Each attempt is a reflash. See `PINOUT.md`.
- **The bin's two limit switches are not wired.** The mechanism has four
  switches across two axes; `LIMIT_SWITCH_INSTANCES` is 2 and sizes the ISR
  router table. Until they exist the bin's position is dead-reckoned from the
  commanded rate, and drifts from the first slip.
- **The load cells are not calibrated.** `aries_load_cells/config/load_cells.yaml`
  now carries 20 / 21 / 30 counts per kg, taken from the embedded team's own
  firmware constants — but that is roughly five orders of magnitude below what
  an HX711 at gain 128 on a cell this size should give, so they were either
  derived against a different setup or never finished. The node says so at
  startup. One known mass in each box settles it, and fixing it is a YAML edit,
  not a reflash.
- **The pump's flow rate is the embedded team's, not ours.** 6.755 mL/s,
  averaged from three timed volumes. Every dose is that number times a duration
  — there is no flow sensor — so it drifts with head height, tube wear and
  battery state. Nothing here has been checked against a measuring cylinder.
- **The drill driver's calibration is not measured.** `drill_driver.yaml` maps
  rates to duty cycle with placeholders taken from `joystick.yaml`. The drill
  moves; the numbers on the topics are not yet the rates the mechanism is doing.
- **`/gripper/state` is an echo, not a measurement.** There is no position
  sensor on this gripper — no feedback line is wired, and the retired sketch ran
  with `USE_SERVO_FEEDBACK = false` for the same reason. If the jaws are
  obstructed this topic still reports that they closed.
- **`SoilTemperature` is unused.**
