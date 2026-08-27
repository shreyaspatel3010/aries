# Teensy 4.1 pinout — drill / science board

Every pin this firmware touches. `include/pins.h` is the code that must agree
with this table; nothing else in the firmware contains a pin number.

**This board replaces the gripper Teensy.** It drives the drill, the gripper
servo, the front-left container lid and the stack light from one Teensy 4.1,
over one micro-ROS serial link, with one `micro_ros_agent`. The retired sketch is at
`firmware/legacy/teensy_gripper/teensy_gripper.ino` — kept for reference only,
not flashed.

---

## The table

| Function | Code name | Pin | Direction | Confirmed? |
|---|---|---|---|---|
| **Feed carriage** — moves the whole drill **up/down** | `FEED_*` | 15 / 40 / 41 | out, PWM 10 kHz + 2 dir | ⚠️ |
| **Auger** — spins the cutting head | `AUGER_*` | 22 / 19 / 18 | out, PWM 10 kHz + 2 dir | ✅ |
| **Sample bin actuator** — slides bin fore/aft | `BIN_*` | 28 / 30 / 29 | out, PWM 10 kHz + 2 dir | ✅ |
| **Gripper servo** | `GRIPPER_SERVO` | 23 | out, servo | ✅ |
| **Container lid servo** (sand box) | `LID_SERVO_SAND_BOX` | 10 | out, servo | ⚠️ |
| **Stack light** green / yellow / red | `STALIG_*` | 37 / 36 / 35 | out, active **LOW** | ✅ |
| **Feed limit** bottom / top | `LIMIT_SWITCH1/2` | 7 / 6 | in, `INPUT_PULLUP` | ✅ |
| Status LED | `LED_BUILTIN` | 13 | out | ✅ |
| **Load cell** — sand box (front-left) | `HX711_SAND_BOX_*` | DT 17 / SCK 16 | in/out | ✅ |
| **Load cell** — stone box (back-left) | `HX711_STONE_BOX_*` | DT 34 / SCK 33 | in/out | ✅ |
| **Load cell** — drill container | `HX711_DRILL_CONTAINER_*` | DT 32 / SCK 31 | in/out | ✅ |

✅ confirmed from the bench  ⚠️ proposed — **check before powering the drill**

23 pins, all distinct, all within the Teensy 4.1 main header. The three motor
PWM pins (22, 15, 28) are all PWM-capable.

### Which motor moves what

The names in the URDF and on the ROS topics are actively misleading, so this
table is the one to trust:

| Moves | Firmware pins | C++ object | ROS topic | Host topic |
|---|---|---|---|---|
| **The whole drill, up and down** | `FEED_*` | `feed_motor` | `motor2/cmd_speed` | `/aries/drill_motor_joint/cmd_vel` |
| The auger spinning (no travel) | `AUGER_*` | `auger` | `motor1/cmd_speed` | `/aries/drill_bit_joint/cmd_vel` |
| The sample bin, fore and aft | `BIN_*` | `bin_actuator` | `linact/cext` | `/aries/drill_container_joint/cmd_vel` |

The feed's two limit switches go the other way — they are the only thing the
drill **reports**:

| Senses | Firmware pins | C++ object | ROS topic | Host consumer |
|---|---|---|---|---|
| Feed carriage at either end | `LIMIT_SWITCH1/2` | `switch_feed_bottom/top` | `drill/limits` | `drill_joystick.py` |

`drill/limits` is a `UInt8` bitfield — **bit0 bottom, bit1 top** — published on
change plus a 2 Hz heartbeat, so silence means the board is gone rather than the
switches being open. No drill axis has an encoder, so this is the entire
feedback path: before it existed, a switch on the wrong pin, an unwired switch
and a working one were indistinguishable from the host, because all three look
like a carriage that simply does not stop.

`drill_motor_joint` is the **vertical feed**, not the motor that drills.
`drill_bit_joint` is the auger's **rotation**. Confirmed from `drill.xacro`:
`drill_motor_joint` is prismatic on `0 0 1` (Z, −0.375 … +0.185),
`drill_container_joint` is prismatic on `1 0 0` (X), `drill_bit_joint` is
continuous on `0 0 1`.

Those URDF and topic names are frozen — they are the contract across the URDF,
the gz bridge, `drill_joystick.py`, `drill_driver.yaml` and
`test_drill_driver.py`. Only the **firmware** names were changed to say what
they move.

---

## What was confirmed, and from where

**`pin-def-ref.txt` was right.** It shipped inside `erc_embedded-drill-sys.zip`
and was never `#include`d by anything — `main.cpp` carried its own copy of the
same `#define` names with **every value set to `0`**, so the firmware as
delivered drove pin 0 for everything. Pin 0 is also `Serial1` RX. But the
*numbers* in it check out: the bench confirmed the auger on 22/19/18 and the
stack light on 37/36/35, exactly as that file says.

There was an interim scare where a bench report put the **sample bin's** PWM on
22, colliding with the auger. The auger's PWM was set `PIN_UNASSIGNED` while it
was resolved, and this document briefly called the file stale. That was wrong —
it resolved the other way: the bin is on **28**, and the file's auger line was
correct all along.

The one line still **not independently confirmed** is the feed carriage's
15/41/40. It is the only motor whose pins nothing has checked against the loom,
and it is the axis that drives the mast into the ground — so it is the one worth
a meter before the first power-up.

**The gripper servo is pin 23** as of 2026-08-26, given from the bench. It was
pin 9 in both `teensy_gripper.ino` and `legacy_controller.ino` and for the whole
life of this firmware before that — so **a board wired to the old loom drives
nothing on the gripper**, and instead pulses the bin actuator's H-bridge. Check
the servo lead before powering the drill.

23 had been the bin actuator's proposed PWM. The actuator has since been
confirmed on **22** from the bench, so pin 9 is now free.

The **lid** servo on pin 10 is a later addition and has no such provenance — see
below.

### The rewire that proves this is one board

The retired sketch drove the stack light on **18, 19, 22**. Those are now,
exactly and in order, the auger's three H-bridge pins:

```
teensy_gripper.ino:   RED_PIN 18      YELLOW_PIN 19   GREEN_PIN 22
now:                  AUGER_INB 18    AUGER_INA 19    AUGER_PWM 22
```

and the light moved to 37/36/35 to make room. A three-for-three reuse is not
coincidence: it only makes sense as a rewire of the one existing board, rather
than a second board being added alongside it.

**A board wired for the old sketch and flashed with this firmware drives the
stack light off the auger's H-bridge.** Check the loom before the first
power-up.

---

## What is still proposed

Everything is now assigned — no pin is `PIN_UNASSIGNED` and the status LED no
longer fast-blinks. Two things are still **guesses**, both mine:

* **The container lid servo, 10.** Free and PWM-capable, but unverified.
* **The feed carriage's 15 / 40 / 41**, from `pin-def-ref.txt` and never checked
  against the loom. The two direction pins were **swapped from 41 / 40 on
  2026-08-27**: positive PWM drove the carriage *down*, and because
  `apply_motor_commands()` chooses the top switch for a positive PWM and the
  bottom one for a negative, that reversal pointed the limit gate at the far end
  of the travel — the carriage ran into the top stop with the firmware watching
  the bottom switch. Fix a reversed feed HERE and never on the host, which sits
  above the sign the gating reads.

The **limit switches are on 7 (bottom) and 6 (top)**, measured 2026-08-27 and
the only pins in this file established by asking the hardware rather than
reading a document. They were `2 / 3` from `pin-def-ref.txt`, then `4 / 5` from
a bench report the same day; both were wrong and **nothing could tell**, because
an `INPUT_PULLUP` pin reads HIGH whether the switch is open or the pin is
connected to nothing. "Wrong pin" and "carriage mid-travel" are the same
reading, so three numbers were tried blind before the board was asked directly.

**Ask it, don't guess:** `scripts/check_drill_limits.py` holds every unused pin
`INPUT_PULLUP` and publishes them on `drill/pin_scan`; press a switch by hand and
the pin that moves is the pin it is on. Both switches are **normally open to
GND** — they rest HIGH and go LOW while held — so `is_at_stop()`'s `== LOW` and
the `FALLING` interrupt edge are both the correct sense. Note that pins 9, 14,
21, 24 and 27 *do* rest low on this harness; they belong to something else.

**Confirm all of these against the harness before powering the drill.** A wrong
direction pin runs an axis into its end stop at 100 % duty cycle.

To change one: edit `include/pins.h`, edit the table above, rebuild. Nothing
else needs touching.

### Collisions fail the build

`pins.h` ends in a `static_assert` over every pin in this table. Two functions
on one pin is otherwise **silent**: whichever `init()` runs last wins the pin
mode, and the other quietly drives — or reads — the wrong hardware.

That is exactly what the 2026-08-26 gripper move would have done. Moving the
servo onto 23 put it on top of the bin actuator's then-PWM, and the build would have been
perfectly happy; the bench symptom would have been a gripper twitching whenever
the sample bin moved.

**Add every new pin to `kMap` in `pins.h`.** A pin left out of that list is not
checked. `PIN_UNASSIGNED` entries are exempt, since several are legitimately
unassigned and all compare equal to each other.

### If a pin is genuinely not wired yet

Set it to `PIN_UNASSIGNED` (255) rather than to a plausible-looking number.
`Driver`, `LimitSwitch` and `SlewServo` all check, and refuse to configure or
drive an unassigned pin. The status LED then blinks fast (100 ms) so the board
says out loud that it is incomplete.

`PIN_UNASSIGNED` is 255 and not 0 deliberately: 0 is a real pin, so a forgotten
`0` configures hardware and looks like it worked. That is precisely how this
firmware arrived.

---

## Wiring notes

### Limit switches fail unsafe, and that is the harness's doing

Both switches are read `INPUT_PULLUP` and fire on `FALLING`, so **closed reads
LOW**. A disconnected switch, a cut wire or a pulled connector therefore reads
HIGH — *not at the stop* — and the carriage will happily run into its mechanical
stop under power.

That is a property of a normally-open switch to ground, not of the firmware. If
the drill is going to be run unattended, rewire the switches **normally closed**
so a broken wire reads as *at the stop*, and invert `is_at_stop()` to match.

### Both switches are on the feed carriage

One at each end of its travel. The firmware as delivered stopped the **auger**
on switch 1 and the feed on switch 2 — but the auger is a spindle and has
no end of travel to reach. `drill_joystick.py` has had the right model all
along:

> `drill_motor` has one at each end of its travel, bottom and top, and the bin's
> actuator has its own at each end of its stroke.

**The bin's two switches are not wired.** The mechanism has four switches across
two axes; the firmware has two instances, capped by `LIMIT_SWITCH_INSTANCES` in
`lib/drill/drill.h`, which sizes the ISR router table. Wiring the bin's pair
needs that constant raised to 4 and two more `isr_router_ls_*` functions added.
Until then the bin is dead-reckoned — which is what
`aries_load_cells/README.md` already describes, and it names the fix:

> The honest fix: the bin's actuator has its own end switch at each end of its
> stroke. Put the forward one on a GPIO, publish a `std_msgs/Bool`, and name it
> in `parked_switch_topic`.

### The container lid

The **front-left deck container** — the sand sample box, which the rest of the
workspace calls `sand_box` (`aries_load_cells`: *"front-left deck box, the sand
sample"*). The box behind it, also on the left, is `stone_box` and has no lid
servo.

That identification came from the load-cell table, not from the hardware. If it
is the wrong box, `LID_SERVO_SAND_BOX` in `pins.h` and the topic name in
`main.cpp` are the only two places to change.

It runs on the **same `SlewServo` class and the same constants as the gripper**
— 850–2200 µs, 550 °/s slew, normalised 0…1. Nothing about those numbers was
re-derived for the lid, so if the lid servo is a different model or has less
travel than the gripper's DSC55MG-270, **the range is too wide and will drive it
into its stop**. That is the one thing here worth a bench check before it is
bolted to the hinge.

`0.0` is **closed** and it is also where `SlewServo` starts, so the lid is
commanded shut within microseconds of reset. That first write is a **jump, not a
slew** — `init()` writes the position directly, before the slew clock starts —
so a lid left open snaps shut at power-up. Correct for a sample container, but
it is a real movement with fingers potentially near the hinge.

There is **no watchdog** on the lid, unlike the drill motors. A lid is meant to
stay where it was put; closing it on a lost link would tip out a sample the
operator was half way through collecting.

### The stack light is active LOW

The tier driver sinks current: **LOW lights a tier**, HIGH darkens it. The
retired sketch was active HIGH on different pins — both halves changed at once.

`emg.cpp` drives each pin to its OFF level *before* `pinMode(OUTPUT)`, because a
Teensy pin latches whatever is in the output register when it becomes an output,
and that is `0` — i.e. LOW, i.e. **on** for an active-low tier. Without that the
light flashes a colour on every boot.

### PWM timer groups

`Driver::init_driver()` calls `analogWriteFrequency(pin, 10000)` on each of the
three motor PWM pins. On a Teensy 4 that sets the frequency of the whole timer
submodule the pin belongs to, not the pin — so if two of these share a submodule
they share a frequency. Harmless here, because all three ask for the same
10 kHz. It stops being harmless the moment one of them wants a different one.

### Load cells

Three HX711 amplifiers — **sand box, stone box, drill bin, in that order**,
which is the element order of `cells` in
`aries_load_cells/config/load_cells.yaml` and therefore **the wire format**. The
firmware sends no names to check itself against, so swapping two entries makes
the sand box report the stone and nothing anywhere notices.

**Each cell has its own clock.** This is *not* the usual one-shared-SCK chain —
every amplifier gets a private DT/SCK pair, so they can be read independently
and one dead amplifier cannot stall the others. Six pins, not four. Anything
written against a shared clock (including an earlier draft of this file) is
wrong.

**Driven.** `main.cpp` constructs one `LoadCell` per amplifier and publishes
`load_cells/raw` (`std_msgs/Int32MultiArray`, three elements, **raw converter
counts**) at 10 Hz, RELIABLE. Scale, offset and tare live in `aries_load_cells`'
YAML, so a recalibration is an edit and a relaunch, not a reflash with the rover
open — and taring is `ros2 service call /load_cells/<cell>/tare`, not something
this board does.

**A cell that is not answering reports the converter's negative rail
(`-8388608`), never zero** — zero is what an empty box reads, so a dead
amplifier would otherwise look exactly like a box somebody had emptied. Reading
one is also strictly non-blocking: `HX711::read()` opens with `wait_ready()`,
which spins forever on an amplifier holding DOUT high, and this loop is also the
auger's watchdog. The topic stays silent until at least one cell has ever
answered, so a rover with no cells fitted is quiet rather than permanently
faulted.

---

## Status LED (pin 13)

`Serial` is the micro-ROS transport, so nothing may print. The LED is the only
channel this board has.

| Pattern | Meaning |
|---|---|
| Fast blink, 100 ms | A pin the firmware needs is still `PIN_UNASSIGNED` |
| Slow blink, 500 ms | Waiting for `micro_ros_agent` |
| Solid on | Connected, pins complete, driving |

The fast blink outranks the slow one: an incomplete pin map is worth knowing
about before the link comes up.
