# Teensy 4.1 — science board firmware

The rover's **science module**: pH, soil moisture, TDS/EC, ORP, soil
temperature, an environmental sensor and CO₂, over micro-ROS on USB serial.

**This is the SECOND Teensy.** [`firmware/teensy_drill_sys`](../teensy_drill_sys)
is a different board on a different USB port with its own micro-ROS agent. This
one carries sensors only and **drives nothing** — no motors, no servos, no
outputs at all beyond the status LED.

- **Pin map and wiring:** [`PINOUT.md`](PINOUT.md) — read it before powering
  anything. Every pin is the embedded team's number and none are confirmed.
- **Calibration:** [`protocols.md`](protocols.md), the embedded team's own
  procedure, kept verbatim.
- **Host side:** `aries_science`, which splits the telemetry array into named
  topics.

---

## Topics

| Topic | Type | Dir | QoS | Meaning |
|---|---|---|---|---|
| `/science/telemetry` | `Float32MultiArray` | out | reliable | 10 values, 1 Hz |
| `/science/sensor_cmd` | `UInt8` | in | reliable | `sensor_id*10 + action` |

One publisher, one subscription and one timer — comfortably inside micro-ROS's
default entity ceilings, so `colcon.meta` here is headroom rather than a fix.

### The array carries no names

Ten floats and nothing else. The **order is the wire format**:

| Index | Field | Unit | Read cmd |
|---|---|---|---|
| 0 | pH | pH | `02` |
| 1 | Soil moisture | % | `12` |
| 2 | TDS / EC | ppm | `22` |
| 3 | ORP | mV | `32` |
| 4 | Soil temperature | °C | `42` |
| 5 | Air temperature | °C | `52` |
| 6 | Humidity | %RH | ← `52` |
| 7 | Pressure | hPa | ← `52` |
| 8 | Gas resistance | Ω | ← `52` |
| 9 | CO₂ | ppm | `92` |

Swap two entries and the rover reports the pH probe as soil moisture: both
numbers stay entirely plausible and nothing anywhere says a word. That order is
pinned against `aries_science/config/science.yaml` by
`test_science_telemetry.py`, which reads `TelemetryIndex` out of `main.cpp`
directly rather than keeping a copy.

Add `1` instead of `2` to initialise instead of read — `01` inits the pH probe,
`02` reads it. **One BME688 read fills four indices** (5–8), which is why
humidity, pressure and gas have no command of their own.

---

## Pull, not stream

**Nothing is sampled unless it is asked for.** The array is published every
second regardless, so a value that was read once keeps being republished until
it is read again — the topic is a *latest-known* board, not a live feed. That is
the embedded team's design and it is kept deliberately: several of these sensors
are slow, and two are consumed by being read.

So an index means one of two things, and they are distinguishable:

- **a number** — the last value that sensor was commanded to produce
- **`NaN`** — never read, or the read failed, or the sensor is not there

**Straight after a reset every index is `NaN`** and stays that way until the
operator sends the init and read commands. That is not a fault.

`NaN` rather than `0.0` is the one behavioural change from the delivered
firmware, and it matters because zero is a legitimate reading for nearly
everything here — 0 °C, 0 % moisture, 0 ppm TDS. The delivered version
zero-filled the array at boot, so an untouched pH index read `0.0`: a strong
acid.

### Reading one

```bash
ros2 topic pub --once /science/sensor_cmd std_msgs/UInt8 "{data: 1}"   # init pH
ros2 topic pub --once /science/sensor_cmd std_msgs/UInt8 "{data: 2}"   # read pH
ros2 topic echo /science/telemetry
```

or by name, through `aries_science`:

```bash
ros2 topic pub --once /science/read std_msgs/String "{data: ph:init}"
ros2 topic pub --once /science/read std_msgs/String "{data: ph}"
ros2 topic echo /science/ph
```

**Wait at least 5 s between DS18B20 reads** — see `protocols.md`.

---

## Build and flash

```bash
pipx install platformio     # once
./flash.sh                  # wipe the cache, rebuild from scratch, flash
```

Identical to the drill board's script and carrying the same traps in its header
— ROS 2 poisoning the build, PlatformIO's hardcoded venv path, the cached
micro-ROS library, and the first-write retry. `--fast` keeps the cache;
`--build-only` does not touch the board.

**It finds ITS OWN board.** Both scripts read `devices.yaml`, and each looks up
its own named block — `science.serial_port` here, `gripper.serial_port` for the
drill. They used to take whichever `serial_port` came first in the file, which
was correct while there was one board and silently wrong the moment there were
two.

---

## What was changed from the delivered firmware

- **The ROS domain is set.** The delivered `rclc_support_init(&support, 0, ...)`
  takes the default domain 0 while the rover runs on 30. That failure is
  completely silent: the agent connects, its log shows every entity created,
  and the board's node never appears and never delivers a sample. The drill
  board shipped with the same bug and it cost most of a day to find there.
- **`NaN`, not `0.0`, for an unread field.** See above.
- **`upload_protocol = teensy-cli`** added — without it the board is flashable
  only from the PlatformIO toolbar, and `flash.sh` cannot drive it.
- **`-DRMW_UXRCE_TRANSPORT=custom` removed** from `colcon.meta`. It contradicts
  `board_microros_transport = serial`, which is how micro_ros_platformio picks
  the transport and already writes the matching value; forcing `custom` asks
  the RMW for callbacks nothing here provides. Do not paste it back from an
  upstream zip.
- **The message buffer is wired in `setup()`**, not `create_entities()`. Both
  are globals that outlive an agent session and `create_entities()` reruns on
  every reconnect, so doing it there risks a reconnect leaving the message
  pointing at nothing — which does not fail loudly, it publishes an empty array.
- **Pins moved to `include/pins.h`** with a compile-time collision check,
  matching the drill board.
- **Dead code dropped** — `RCCHECK` was never used, `error_loop()` only by it,
  and `Wire2.begin()` configured a peripheral and claimed two pins for nothing.
- **A status LED**, slow-blinking until the agent connects. The board had no
  status channel at all, and `Serial` cannot be one.

---

## What is not done

- **No pin is confirmed against the loom.** Every number in `PINOUT.md` is the
  embedded team's.
- **No calibration constant is verified.** The pH curve's two buffer voltages,
  the moisture endpoints, the TDS factor and the ORP offset are all theirs, and
  correcting any of them is a reflash — see `PINOUT.md` for why they live here
  rather than in host YAML.
- **The DS18B20's 4.7 kΩ pull-up is not in the code and cannot be.** Fit it.
- **Nothing drives this board from the joystick.** It is `ros2 topic pub` and
  mission scripts, deliberately: taking a reading is not a thing to do by
  holding a button.
