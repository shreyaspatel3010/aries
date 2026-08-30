# Science board — pin map and wiring

The **second Teensy 4.1**. `firmware/teensy_drill_sys` is a different board on a
different USB port with its own micro-ROS agent; this one carries **sensors
only and drives nothing**.

Numbers live in [`include/pins.h`](include/pins.h) and nowhere else. Change one
there, change it here.

---

## The table

| Function | Code name | Pin | Direction | Confirmed? |
|---|---|---|---|---|
| **pH** (DFRobot analog board) | `PIN_PH` | 14 (A0) | in, analog | ⚠️ |
| **Soil moisture**, capacitive v1.2 | `PIN_MOISTURE` | 26 (A12) | in, analog | ⚠️ |
| **TDS / EC** (SEN0244) | `PIN_TDS` | 27 (A13) | in, analog | ⚠️ |
| **ORP** (SEN0165) | `PIN_ORP` | 15 (A1) | in, analog | ⚠️ |
| **DS18B20** soil temperature | `PIN_TEMP_SOIL` | 2 | OneWire | ⚠️ |
| **BME688** environmental | `Wire` | SDA 18 / SCL 19 | I2C | ⚠️ |
| **SCD41** CO₂ | `Wire1` | SDA 17 / SCL 16 | I2C | ⚠️ |
| Status LED | `LED_BUILTIN` | 13 | out | ✅ |

**Everything here is the embedded team's numbering, taken from the delivered
`main.cpp`, and none of it has been checked against a loom.** That is what the
⚠️ means throughout.

---

## The two boards cannot be merged

Four of these pins are something else entirely on the drill board:

| Pin | Science board | Drill board |
|---|---|---|
| 15 | ORP | `FEED_PWM` — the feed carriage |
| 18 / 19 | I2C `Wire` (BME688) | `BIN_INB` / `BIN_INA` — sample bin direction |
| 17 / 16 | I2C `Wire1` (SCD41) | `HX711_SAND_BOX_DT` / `_SCK` |
| 25 | (`Wire2`, unused — see below) | `AUGER_PWM` |

So this is two Teensys **by necessity**, not by preference, and the two
`pins.h` files are unrelated documents. Both carry their own compile-time
collision `static_assert`, and neither knows about the other.

---

## Why two I2C buses

The BME688 is driven by the Zanshin library, which is **hardcoded to the
default `Wire` object** and offers no way to point it at another bus. So the
SCD41 — whose SparkFun library does take a bus argument — goes on `Wire1` to
keep the two apart. It is a library constraint, not an electrical one.

The pairs are fixed by the silicon and cannot be reassigned: `Wire` is 18/19,
`Wire1` is 17/16, `Wire2` is 25/24.

**`Wire2` is deliberately not started.** The delivered firmware called
`Wire2.begin()` with the comment *"reserved / unused for now"*, which configures
a peripheral and claims two pins for nothing. Start it when something is
actually on it — and add its pins to `kMap` in `pins.h` at the same time.

---

## The DS18B20 needs a resistor you cannot see in the code

`init()` sets `INPUT_PULLUP`, but the Teensy's internal pull-up is weak (tens of
kΩ) and the DS18B20's datasheet asks for **4.7 kΩ to 3.3 V** on the data line.

This is the failure that wastes an afternoon: it usually works on a short bench
lead and stops working on a rover-length one, and the symptom is an
intermittent `-127` — the library's disconnect sentinel — rather than anything
that looks like a wiring fault. **Fit the resistor.**

The firmware maps that `-127` to `NaN` and retries once automatically before
reporting it, because a lone `-127` is far more often a transient OneWire glitch
than a probe that has genuinely come off.

---

## The ADC contract

`V_REF` is 3.3 V and `ADC_RES` is 1023 — the 10-bit default of `analogRead()`.

The Teensy 4 can do 12 bits, but only if `analogReadResolution(12)` is called,
and nothing here calls it. **If that is ever added, `ADC_RES` in `science.h`
must change with it.** Leaving 1023 against a 12-bit read scales every analog
sensor by 4, and nothing anywhere reports it, because the readings stay inside a
believable range.

All four analog sensors are on analog-capable pins on purpose: `analogRead()` on
a pin that is not analog-capable returns 0 without complaining, which reads as a
sensor at the bottom of its range rather than as a mistake.

---

## Calibration lives on this board

Unlike the load cells — whose scale and offset are host-side YAML in
`aries_load_cells`, so a recalibration is an edit and a relaunch — the science
sensors are calibrated **in the firmware**, and correcting one is a reflash.

That is not an oversight. Most of it is a probe-specific voltage curve rather
than a scale factor (the pH probe's two buffer voltages, the ORP board's op-amp
gain), and two of these are calibrated with a **physical potentiometer on the
sensor board** rather than in software at all.

[`protocols.md`](protocols.md) is the embedded team's procedure for every one of
them and is kept verbatim.

**None of the constants have been verified here.** The pH curve, the moisture
endpoints (759 air / 403 water), the TDS factor and the ORP offset are all
theirs.

---

## Status LED

| LED | Meaning |
|---|---|
| Slow blink (500 ms) | Flashed fine, waiting for the micro-ROS agent |
| Solid on | Agent connected |

There is no fast-blink pin-incomplete state like the drill board's: every pin
here is assigned, and this board drives nothing that a wrong pin could damage.

**Do not open a serial monitor on this board.** `Serial` is the micro-ROS
transport, not a console, and reading it corrupts the link.
