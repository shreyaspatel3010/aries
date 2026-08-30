// Teensy 4.1 pin map for the SCIENCE board.
//
// A SECOND BOARD, not the drill's. firmware/teensy_drill_sys is a different
// Teensy on a different USB port with its own micro-ROS agent, and the two pin
// maps are unrelated -- four of the pins below are used for something else
// entirely over there (15 is the feed carriage's PWM, 18/19 are the sample
// bin's direction pins, 17/16 are the sand box load cell, 25 is the auger's
// PWM). THE TWO FIRMWARES CANNOT BE MERGED ONTO ONE BOARD without renumbering
// one of them; that is why this directory exists rather than the sensors being
// added to the drill.
//
// THIS FILE IS THE ONLY PLACE A PIN NUMBER APPEARS. The delivered firmware kept
// its #defines at the top of main.cpp; they are here for the same reason the
// drill board's are, and the compile-time collision check at the bottom is the
// point of centralising them.
//
// See PINOUT.md for the wiring table and what is still unconfirmed.
//
// TEENSY 4.1 CONSTRAINTS THIS MAP RESPECTS
//   - Pin 13 is LED_BUILTIN and is the board's only status channel, because
//     Serial belongs to micro-ROS. Nothing else may claim it.
//   - Pins 0 and 1 are Serial1 RX/TX. Left free.
//   - The four analog sensors must be on ANALOG-CAPABLE pins: 14 and 15 are
//     A0/A1, and 26/27 are A12/A13. An analogRead() on a pin that is not
//     analog-capable returns 0 without complaining, which would read as a
//     sensor at the bottom of its range rather than as a wiring mistake.
//   - The I2C buses are FIXED PAIRS on this part and cannot be reassigned:
//     Wire is 18/19, Wire1 is 17/16, Wire2 is 25/24.

#ifndef PINS_H
#define PINS_H

// Same sentinel and the same contract as the drill board's pins.h: 255 is not
// a valid Teensy pin, and anything that takes a pin number checks this before
// configuring hardware. Deliberately NOT 0, which is a real pin (Serial1 RX)
// and so looks like it worked.
#define PIN_UNASSIGNED 255
#define PIN_IS_ASSIGNED(p) ((p) != PIN_UNASSIGNED)

// --- Analog sensors ----------------------------------------------------------
// All four are voltage dividers or op-amp boards read straight off the ADC.
// V_REF is 3.3 V and the ADC is left at its 10-bit default (0..1023) -- see
// science.h, where both are constants that have to agree with this comment.
//
// THESE ARE THE EMBEDDED TEAM'S NUMBERS, taken from the delivered main.cpp and
// not checked against a loom here.

// DFRobot analog pH board. A0.
#define PIN_PH 14

// Capacitive soil moisture v1.2. A12. Capacitive, NOT resistive -- the
// calibration constants in science.cpp (759 air, 403 water) are only valid for
// the capacitive part, and a resistive probe on this pin reads plausibly and
// wrongly.
#define PIN_MOISTURE 26

// DFRobot SEN0244 TDS / EC. A13.
#define PIN_TDS 27

// DFRobot SEN0165 ORP. A1.
#define PIN_ORP 15

// --- Digital sensors ---------------------------------------------------------
// DS18B20 soil temperature probe, on a OneWire bus of its own.
//
// NEEDS AN EXTERNAL PULL-UP. The library is driven with INPUT_PULLUP here, but
// the Teensy's internal pull-up is weak (tens of kOhms) and the DS18B20's
// datasheet asks for 4.7 kOhm to 3.3 V on the data line. It often works on a
// short bench lead and stops working on a rover-length one, which reads as an
// intermittent -127 rather than as a wiring fault. Fit the resistor.
#define PIN_TEMP_SOIL 2

// --- I2C ---------------------------------------------------------------------
// TWO BUSES, AND THAT IS NOT A CHOICE. The BME688 is driven by the Zanshin
// library, which is hardcoded to the default `Wire` object and offers no way to
// point it at another bus. So the SCD41, whose SparkFun library does take a
// bus, goes on Wire1 to keep the two apart.
//
// These pairs are fixed by the silicon and are here to be read, not changed.
#define I2C_WIRE_SDA 18   // BME688
#define I2C_WIRE_SCL 19
#define I2C_WIRE1_SDA 17  // SCD41
#define I2C_WIRE1_SCL 16

// Wire2 (25/24) is NOT started. The delivered firmware called Wire2.begin()
// with the comment "reserved / unused for now", which claims two pins and
// configures a peripheral for nothing. Start it in setup() when something is
// actually on it, and add its pins to kMap below at the same time.

// --- COMPILE-TIME COLLISION CHECK --------------------------------------------
//
// Two functions on one pin is silent at runtime: whichever init runs last wins
// the pin mode and the other quietly reads or drives the wrong hardware. The
// drill board carries the identical check, and it has already caught one real
// collision there.
//
// Add every new pin here. PIN_UNASSIGNED is exempt -- several entries may
// legitimately be unassigned and they all compare equal to each other.
#ifdef __cplusplus
namespace aries_science_pins
{
constexpr unsigned char kMap[] = {
    PIN_PH, PIN_MOISTURE, PIN_TDS, PIN_ORP,
    PIN_TEMP_SOIL,
    I2C_WIRE_SDA, I2C_WIRE_SCL,
    I2C_WIRE1_SDA, I2C_WIRE1_SCL,
    // LED_BUILTIN, spelled literally because pins.h does not include Arduino.h.
    // Here so that handing pin 13 to a sensor fails the build rather than
    // fighting the status LED.
    13,
};
constexpr int kCount = sizeof(kMap) / sizeof(kMap[0]);

constexpr bool no_pin_collision()
{
  for (int i = 0; i < kCount - 1; ++i)
    for (int j = i + 1; j < kCount; ++j)
      if (kMap[i] == kMap[j] && kMap[i] != PIN_UNASSIGNED)
        return false;
  return true;
}

static_assert(no_pin_collision(),
              "pins.h: two functions are mapped to the same Teensy pin. "
              "Compare the #defines above against PINOUT.md.");
} // namespace aries_science_pins
#endif // __cplusplus

#endif // PINS_H
