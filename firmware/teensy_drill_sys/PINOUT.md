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
| **Auger** — spins the cutting head | `AUGER_*` | 25 / 8 / 9 | out, PWM 10 kHz + 2 dir | ⚠️ |
| **Sample bin actuator** — slides bin fore/aft | `BIN_*` | 22 / 19 / 18 | out, PWM 10 kHz + 2 dir | ⚠️ |
| **Gripper servo** | `GRIPPER_SERVO` | 23 | out, servo | ✅ |
| **Container lid servo** (sand box) | `LID_SERVO_SAND_BOX` | 38 | out, servo — **continuous rotation** | ⚠️ |
| **Stack light** green / yellow / red | `STALIG_*` | 37 / 36 / 35 | out, active **LOW** | ✅ |
| **Feed limit** bottom / top | `LIMIT_SWITCH1/2` | 7 / 6 | in, `INPUT_PULLUP` | ✅ |
| Status LED | `LED_BUILTIN` | 13 | out | ✅ |
| **Load cell** — sand box (front-left) | `HX711_SAND_BOX_*` | DT 17 / SCK 16 | in/out | ✅ |
| **Load cell** — stone box (back-left) | `HX711_STONE_BOX_*` | DT 34 / SCK 33 | in/out | ✅ |
| **Load cell** — drill container | `HX711_DRILL_CONTAINER_*` | DT 32 / SCK 31 | in/out | ✅ |

✅ confirmed from the bench  ⚠️ proposed — **check before powering the drill**

23 pins, all distinct, all within the Teensy 4.1 main header. The three motor
PWM pins (25, 15, 22) are all PWM-capable.

> **Changed 2026-08-29.** The auger moved to 25 / 8 / 9 and the sample bin took
> the auger's old 22 / 19 / 18; 28, 29 and 30 are now free. Both rows dropped
> from ✅ to ⚠️ because no provenance was recorded for the new numbers — see
> below, and see **the diagnostic scan list is not covered by that check**,
> which this change has already broken.

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

**The auger and the sample bin were renumbered on 2026-08-29**, and the two
swapped territory:

```
             before 2026-08-29    now
auger        22 / 19 / 18         25 /  8 /  9
sample bin   28 / 30 / 29         22 / 19 / 18
```

The bin has taken the auger's three old pins outright. **Where the new numbers
came from is not recorded here** — the change arrived in `pins.h` with the
surrounding comments left describing the old map, so neither row has provenance
any more and both are ⚠️ above. What follows is how the *previous* numbers were
established, kept because it is what those two rows now have to be re-confirmed
against.

**`pin-def-ref.txt` was right about the map as it stood.** It shipped inside
`erc_embedded-drill-sys.zip` and was never `#include`d by anything — `main.cpp`
carried its own copy of the same `#define` names with **every value set to
`0`**, so the firmware as delivered drove pin 0 for everything. Pin 0 is also
`Serial1` RX. The bench confirmed the numbers it gave for the auger (22/19/18)
and the stack light (37/36/35) on 2026-08-26. The 2026-08-29 change supersedes
its auger line; the stack light is untouched.

There was an interim scare where a bench report put the **sample bin's** PWM on
22, colliding with the then-auger. The auger's PWM was set `PIN_UNASSIGNED`
while it was resolved, and this document briefly called the file stale. That was
wrong *at the time* — it resolved the other way, onto 28. The bin **is** on 22
now, but by a later and separate decision; that does not retroactively make the
2026-08-26 report right, and it is worth being clear about which of the two
reasons put it there.

The feed carriage's 15/40/41 has **never** been independently confirmed. It is
the only motor whose pins nothing has checked against the loom, and it is the
axis that drives the mast into the ground — so it is the one worth a meter
before the first power-up.

**The gripper servo is pin 23** as of 2026-08-26, given from the bench. It was
pin 9 in both `teensy_gripper.ino` and `legacy_controller.ino` and for the whole
life of this firmware before that — so **a board wired to the old loom drives
nothing on the gripper**, and instead pulses whatever now sits on pin 9. Check
the servo lead before powering the drill.

Pin 9 was free for three days and is not free any more: **it is `AUGER_INB` as
of 2026-08-29**. An old-loom gripper lead now feeds a direction input on the
auger's H-bridge rather than a spare pin.

The **lid** servo is a later addition and has no such provenance — see below. It
moved from **10 to 38 on 2026-08-29**, and pin 10 is now free (and back in the
diagnostic scan list).

### The rewire that proves this is one board

The retired sketch drove the stack light on **18, 19, 22**. Those three pads
have belonged to an H-bridge ever since — the auger's until 2026-08-29, the
sample bin's now:

```
teensy_gripper.ino:   RED_PIN 18      YELLOW_PIN 19   GREEN_PIN 22
until 2026-08-29:     AUGER_INB 18    AUGER_INA 19    AUGER_PWM 22
now:                  BIN_INB 18      BIN_INA 19      BIN_PWM 22
```

and the light moved to 37/36/35 to make room. A three-for-three reuse is not
coincidence: it only makes sense as a rewire of the one existing board, rather
than a second board being added alongside it. The renumber does not weaken that
argument — the same three pads simply changed which motor they drive.

**A board wired for the old sketch and flashed with this firmware drives the
stack light off an H-bridge** — the sample bin's now, the auger's before.
Check the loom before the first power-up.

---

## What is still proposed

Everything is now assigned — no pin is `PIN_UNASSIGNED` and the status LED no
longer fast-blinks. What is **not** established:

* **The auger's 25 / 8 / 9 and the sample bin's 22 / 19 / 18**, changed
  2026-08-29 with no provenance recorded. The map is at least electrically
  valid — 25 and 22 are both PWM-capable, and the `static_assert` proves all six
  are distinct — but nothing here says the **loom** agrees, and these are the
  two axes whose direction pins were the reason this file exists. Meter them
  before power-up, and read the `kScanPins` warning below first.
* **The container lid servo, 38.** Unverified. Note it is **not** a PWM pin on
  a Teensy 4.1 — FlexPWM1_2 reaches pin 38 on a 4.0, but on the 4.1 that pad is
  46/47. This does not matter: the Teensy `Servo` library bit-bangs the pulses
  with `digitalWrite` from a timer ISR and works on any digital pin. It does
  mean `analogWrite()` on this pin would silently do nothing, so do not put a
  motor driver here later without re-checking.
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
the `FALLING` interrupt edge are both the correct sense. Note that pins 14, 21,
24 and 27 *do* rest low on this harness; they belong to something else. Pin 9
was on that list too and is now `AUGER_INB` — consistent with its having been an
H-bridge input on the loom all along, though nothing has confirmed that and it
should not be read as confirmation of the new auger map.

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

### The diagnostic scan list is *not* covered by that check

The `static_assert` guards `kMap`. It does **not** guard `kScanPins` in
`main.cpp` — the list of otherwise-unused pins the board holds `INPUT_PULLUP`
and reports on `drill/pin_scan`. Those two lists have to stay disjoint and
nothing anywhere checks that they are.

**They are disjoint as of 2026-08-30**, apart from the two limit switches, which
are in both on purpose (they are reported alongside the free pins for
comparison, and `LimitSwitch::init()` has already made them `INPUT_PULLUP`):

```
kMap      6 7 8 9 13 15 16 17 18 19 22 23 25 28 29 30 31 32 33 34 35 36 37 38 40 41
kScanPins 0 1 2 3 4 5 6 7 10 11 12 14 20 21 24 26 27 39
overlap   6 7   <- LIMIT_SWITCH1/2, deliberate
```

Getting there took two corrections in two days, in both directions:

* The 2026-08-29 renumber moved the auger onto 8, 9 and 25 and left all three in
  the scan list, while freeing 28, 29 and 30 from the bin and not adding them.
* The pump then took 28, 29 and 30 on 2026-08-30, so those three had to come
  straight back out again.

**Why it matters, since it never breaks the mechanism.** `setup()` runs the
scan's `pinMode(..., INPUT_PULLUP)` loop *before* the motors and servos
initialise, so the driver always wins the pin mode and the hardware works. What
breaks is the diagnostic: `pin_scan_state()` goes on reading those pins every
cycle and reports the driver's own switching as *something is pulling this pin
low* — flickering bits on `drill/pin_scan` whenever that axis moves. That is the
trap the lid servo hit on pin 38, and `check_drill_limits.py` reads exactly this
topic, so the tool this file tells you to trust for finding a switch is the tool
the collision blinds.

**ADD THE PIN YOU FREE, AND REMOVE THE PIN YOU TAKE.** Both halves, every time.

### If a pin is genuinely not wired yet

Set it to `PIN_UNASSIGNED` (255) rather than to a plausible-looking number.
`Driver`, `LimitSwitch`, `SlewServo` and `LidServo` all check, and refuse to
configure or drive an unassigned pin. The status LED then blinks fast (100 ms) so the board
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
Until then the bin's position is dead-reckoned from the commanded rate, which
drifts from the first slip onward. The honest fix: the bin's actuator has its
own end switch at each end of its stroke — put the forward one on a GPIO and
publish it.

### The peristaltic pump

New from the embedded team, 2026-08-30, on **28 / 29 / 30 — their numbers,
unchanged**. `pump/state` (`UInt8`) picks the action: `1` release, `2` draw,
`3`/`4` home in either direction, `5` home-then-draw, `0` stop.

**Commanded in millilitres, run as a timer.** There is no flow sensor and no
level sensor. `Pump` converts a volume into a run time at a flow rate the
embedded team measured — 6.755 mL/s, averaged over 100 mL/15 s, 150 mL/22 s and
200 mL/29.5 s — so every dose drifts with head height, tube wear and battery
state exactly as the drill's three axes do. A commanded volume is an estimate,
not a measurement.

**These three pins were the sample bin's until 2026-08-29.** Upstream's own pin
block still has the bin on them as well — `LINACT_PWM 28 / LINACT_INA 30 /
LINACT_INB 29`, the same three pins with the direction pair swapped — with both
objects constructed and both initialised in their `setup()`. On their board the
bin and the pump share one H-bridge and drive it in opposite directions. It is
only harmless here because the bin moved to 22 / 19 / 18 the day before, which
vacated exactly this group. The `static_assert` in `pins.h` is what makes that
safe to rely on rather than hope about: put the bin back on 28 and the build
fails instead of the two mechanisms fighting over a bridge at runtime.

They also came out of `kScanPins` in `main.cpp`, where they had landed when the
bin vacated them — a pin that is both scanned and driven reports the driver's
own switching as a phantom switch.

**Three defects were fixed from the delivered class**, each already familiar
from `LinearActuator`, which `Pump` was copy-pasted from:

1. **A zero volume ran the pump forever.** `req_vol` of 0 made a 0 µs
   `IntervalTimer` period, which never fires — the pump was switched on one line
   earlier and had nothing left to switch it off. Guarded by
   `move_duration_us()`/`start_move()`, exactly as the actuator is.
2. **State 5 never homed.** `home(false); draw();` re-armed the same timer on
   the second call, cancelling the first before a microsecond had elapsed, so
   only the draw ran. It is one combined move now — both halves drive the same
   direction at the same duty cycle, so summing the durations is exactly the
   intended sequence and avoids calling `IntervalTimer::begin()` from inside its
   own ISR.
3. **`home()` latched its duration.** It opened with `m_home_dur = 30;`, an
   assignment to a member, discarding the 14.8 s computed in the initialiser —
   which therefore never ran once. 30 s is kept, as a constant that cannot be
   overwritten.

**On the pad:** LT + X, one message per press, sending `draw`. Edge-triggered
rather than streamed — republishing at 30 Hz would re-arm the dose timer on
every message. A dose in progress is not cancelled by releasing LT; send
`data: 0` to stop one.

### The container lid

The **front-left deck container** — the sand sample box, which the rest of the
workspace calls `sand_box`. The box behind it, also on the left, is the stone
box (`rock_box` on the wire) and has no lid servo.

That identification came from the load-cell table, not from the hardware. If it
is the wrong box, `LID_SERVO_SAND_BOX` in `pins.h` and the topic name in
`main.cpp` are the only two places to change.

**On pin 38 since 2026-08-29**, having been on 10 before that. Whenever this pin
moves, check `kScanPins` in `main.cpp`: the diagnostic scan configures every pin
in that list `INPUT_PULLUP`, and a pin that is both scanned and driven reports
the servo's own pulse train as a switch chattering at 50 Hz. 38 was removed from
that list and 10 was added back to it.

**It is a 360° continuous-rotation servo, not a positional one** — corrected
2026-08-29 from the embedded team's own firmware, which models it that way and
has the hardware in hand. Until then it ran on `SlewServo`, the gripper's class,
commanded a normalised 0…1 **position**. That was an assumption nobody had
checked, and it is not a small one: writing an angle to a continuous-rotation
servo does not park it anywhere, it picks two fixed speeds and turns at one of
them until something else is written.

So `sand_box/lid/cmd` carries a **speed in −1…1**, where `0.0` is stop and the
sign is direction. `LidServo` in `lib/drill` owns it.

**`LID_SERVO_NEUTRAL_US` must be trimmed on the bench, and 1500 is a guess.**
The pulse width at which a continuous-rotation servo genuinely holds still is a
property of *this individual unit* — set by how its centring pot was trimmed on
the line — and it is usually not exactly 1500 µs. If the lid creeps with the
stick released, that number is why, and nothing else is. Publish `0.0` to the
topic, watch the horn, and move the constant 10 µs at a time until it is still.
Each attempt is a reflash.

**There is a watchdog, and here it is not optional.** `LidServo::update()` stops
the servo if no command has arrived for 500 ms. A *position* command has a
resting state, so the old positional lid could safely be left alone on a lost
link; a *speed* command has none, and a lid turning when the link drops turns
until somebody cuts the power. The board also stops it on a clean
`AGENT_DISCONNECTED` rather than waiting the timeout out.

Upstream declared this watchdog and never wrote it — `void update(uint32_t
command_timeout_ms = 500);` in the header, an `m_last_cmd_ms` member marked
`NEW`, no definition and no caller anywhere. Calling it would not have linked.
It is implemented here.

**It no longer snaps shut at power-up.** `init()` writes the neutral (stop)
pulse and nothing else, so a lid left open stays open until somebody drives it.
That is the safer boot for a servo with no idea where it is: the alternative is
turning at speed toward a closed position it has no way to detect reaching.

**On the pad:** LT + right stick up/down, via `drill_joystick`. `invert_lid` in
`joystick.yaml` chooses which way is open — there is nothing to derive it from,
it is how the horn is fitted.

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

After the 2026-08-29 renumber the three still land on three different
peripherals, from `pwm_pin_info[]` in the core's `cores/teensy4/pwm.c`:

| Pin | Timer | Used by |
|---|---|---|
| 25 | `FlexPWM1_3_X` | `AUGER_PWM` |
| 15 | `QuadTimer3_3` | `FEED_PWM` |
| 22 | `FlexPWM4_0_A` | `BIN_PWM` |

No two share a submodule, so the shared-frequency hazard is still only
theoretical. Two things about that table are worth keeping:

* **Pin 25 is an X channel**, not the usual A or B. `flexpwmWrite()` handles
  `case 0: // X` by writing `VAL0`, so `analogWrite()` on it is fully
  supported — this is not the pin-38 situation, where the pad has no PWM at all.
* **Pins 7, 8 and 25 are all on FlexPWM1 submodule 3.** Only 25 is driven as
  PWM; 8 is `AUGER_INA` (plain `digitalWrite`) and 7 is `LIMIT_SWITCH1`
  (`INPUT_PULLUP`), and neither cares what frequency that submodule runs at. So
  it is harmless — but the auger's own PWM and direction pins now sit on one
  submodule with a limit switch, which is one `analogWrite()` away from
  mattering. Do not put a second PWM function on 7 or 8.

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
open — and taring is host-side too, either

```
ros2 service call /load_cells/sand_box/tare std_srvs/srv/Trigger   # reports the new offset
ros2 topic pub --once /load_cells/sand_box/tare std_msgs/UInt8 "data: 1"
```

(`1` empty, `2` with the lid), not something this board does. The service is
the one to use while calibrating, because only it can hand the offset back for
pasting into `load_cells.yaml`.

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
