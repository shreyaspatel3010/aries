// Teensy 4.1 pin map for the drill / science board.
//
// THIS FILE IS THE ONLY PLACE A PIN NUMBER APPEARS. main.cpp used to carry its
// own #defines, all of them 0, with the real numbers sitting in an untracked
// src/pin-def-ref.txt that nothing included -- so the firmware as shipped drove
// pin 0 for every motor, every light and the servo. Pin 0 is also Serial1 RX.
// Keep every number here and that class of drift cannot come back.
//
// See PINOUT.md for the wiring table, the harness colours, and what is still
// unconfirmed. Change a number here, change it there.
//
// TEENSY 4.1 CONSTRAINTS THIS MAP RESPECTS
//   - Pin 13 is LED_BUILTIN. main.cpp blinks it as the agent-status and
//     unassigned-pin indicator, so nothing else may claim it.
//   - Pins 0 and 1 are Serial1 RX/TX. Left free.
//   - Every digital pin on a Teensy 4.x is interrupt-capable, so the limit
//     switches are not constrained to a special subset the way they would be
//     on an AVR.
//   - The PWM pins here (22, 15, 28) are all PWM-capable. analogWriteFrequency
//     sets the frequency of the whole timer submodule a pin belongs to, not the
//     pin -- but all three drivers ask for the same 10 kHz in
//     Driver::init_driver(), so a shared timer is harmless. It would NOT be if
//     one of them ever wanted a different frequency.

#ifndef PINS_H
#define PINS_H

// Sentinel for "no number has been established for this yet". 255 is not a
// valid Teensy pin, and Driver/LimitSwitch/SlewServo all refuse to
// initialise on it rather than quietly configuring pin 255 -- see
// PIN_IS_ASSIGNED below and its use in drill.cpp.
//
// Deliberately NOT 0: 0 is a real pin (Serial1 RX), so a forgotten 0 configures
// hardware and looks like it worked. Every unassigned pin in this file has bitten
// somebody already in exactly that way.
#define PIN_UNASSIGNED 255
#define PIN_IS_ASSIGNED(p) ((p) != PIN_UNASSIGNED)

// --- Drill motors -----------------------------------------------------------
// Both are DC motors behind an H-bridge: one PWM pin for speed, two direction
// pins. Numbers are the embedded team's own, from the pin-def-ref.txt that
// shipped alongside the firmware.
//
// THE AUGER -- the bit that spins and cuts. It does not travel.
//
// CONFIRMED from the bench, 2026-08-26: 22 / 19 / 18, which is exactly what
// pin-def-ref.txt said all along.
//
// This briefly looked stale. An interim bench report put the SAMPLE BIN's PWM
// on 22, which collided with the auger, and the auger's PWM was set
// PIN_UNASSIGNED while that was resolved. It resolved the other way: the bin is
// on 28 and pin-def-ref.txt was right about the auger. The file's credibility
// stands -- the collision was a reporting error, not a stale file.
#define AUGER_PWM 22
#define AUGER_INA 19
#define AUGER_INB 18

// THE FEED CARRIAGE -- the lead screw that moves the WHOLE DRILL up and down
// (drill_motor_joint, prismatic on Z, -0.375 .. +0.185). This is the vertical
// axis, despite the URDF calling it "motor" and the auger "bit".
//
// INA/INB SWAPPED 2026-08-27 (was 41 / 40). POSITIVE PWM MUST DRIVE THE
// CARRIAGE UP, and on the bench it drove it DOWN. That sign is not cosmetic:
// apply_motor_commands() picks WHICH SWITCH to consult from it --
//
//     feed_pwm > 0  ->  switch_feed_top
//     feed_pwm < 0  ->  switch_feed_bottom
//
// -- so a reversed bridge does not merely invert the pad, it points the gate at
// the far end of the travel. The carriage then climbs into the TOP switch while
// the firmware is watching the BOTTOM one, and neither switch ever stops
// anything. That is what "limit switches not working" was on 2026-08-27.
//
// FIX IT HERE, NOT ON THE HOST. Inverting in joystick.yaml or drill_driver.yaml
// makes the pad feel right and leaves the gate pointed the wrong way, because
// both of those sit ABOVE the sign the firmware gates on. Driver::drive() maps
// dir=true to INA HIGH / INB LOW, so exchanging these two numbers is exactly
// equivalent to swapping the motor's two leads at the bridge.
#define FEED_PWM 15
#define FEED_INA 40
#define FEED_INB 41

// --- Sample-bin linear actuator ---------------------------------------------
// The bin rides its rails between q = 0 (parked forward of the mast) and
// q = -0.1304 (back under the auger); see aries/urdf/drill.xacro and the
// aries_load_cells README, which depends on knowing which end it is at.
//
// CONFIRMED from the bench, 2026-08-26: 28 / 30 / 29. 28 is PWM-capable.
//
// These were proposals (23/21/20, then 22/21/20) until the bench supplied them.
// Nothing about the earlier numbers survives.
#define BIN_PWM 28
#define BIN_INA 30
#define BIN_INB 29

// --- Limit switches ---------------------------------------------------------
// Wired to GND through the switch and read INPUT_PULLUP, so CLOSED reads LOW
// and the ISR fires on FALLING. A disconnected switch therefore reads HIGH --
// "not at the stop" -- which is the unsafe-failing direction. That is a
// property of the harness, not of this file; it is called out in PINOUT.md so
// the person holding the loom knows to expect it.
//
// BOTH SWITCHES ARE ON THE FEED CARRIAGE, one at each end of its travel. The
// firmware as delivered stopped the auger on switch 1 and the feed on switch
// 2, which does not match the mechanism: the auger is a spindle and has
// no end of travel to reach. drill_joystick.py has had the correct model all
// along -- "drill_motor has one at each end of its travel, bottom and top".
//
// 7 = BOTTOM, 6 = TOP. MEASURED, 2026-08-27, and these are the first numbers
// here that were not somebody's guess: with drill/pin_scan running, each switch
// was pressed by hand and the board reported which pin moved. 2 / 3 came from
// pin-def-ref.txt, and 4 / 5 replaced them the same day on a bench report --
// both were wrong, and nothing could tell, because a limit switch on the wrong
// pin is SILENT. An INPUT_PULLUP pin reads HIGH whether the switch is open or
// the pin is connected to nothing at all, so "wrong pin" and "carriage
// mid-travel" are the same reading. Three pin numbers were tried blind before
// the board was simply asked. Ask it: scripts/check_drill_limits.py.
//
// NORMALLY OPEN, switching to GND, exactly as described above -- confirmed by
// the same test: both pins rest HIGH and go LOW while the switch is held. So
// is_at_stop()'s digitalRead(pin) == LOW is the right sense, and the FALLING
// interrupt edge is the closure. (Pins 9, 14, 21, 24 and 27 DO rest low on this
// harness; they are something else's, and are not these switches.)
//
// The BIN's two switches are not in this map. The mechanism has four switches
// across two axes; the firmware has two instances (LIMIT_SWITCH_INSTANCES in
// drill.h, which caps the ISR router table). Wiring the bin's pair needs that
// constant raised to 4 and two more routers added. Until then the bin is
// dead-reckoned, exactly as the aries_load_cells README describes.
#define LIMIT_SWITCH1 7  // feed carriage, BOTTOM of travel
#define LIMIT_SWITCH2 6  // feed carriage, TOP of travel

// --- Stack light ------------------------------------------------------------
// Three GPIOs, one per tier, on the mast. ACTIVE LOW: the driver sinks current,
// so LOW lights the tier and HIGH darkens it. The retired sketch was active
// HIGH on different pins -- both halves of that changed at once, which is why
// StackLight::state() is written against the host's colour codes rather than
// the firmware's original ones. See emg.cpp.
#define STALIG_G 37
#define STALIG_Y 36
#define STALIG_R 35

// --- Servos -----------------------------------------------------------------
// MOVED TO 23 on 2026-08-26. Was pin 9 in both retired sketches
// (teensy_gripper.ino and legacy_controller.ino) and for the whole life of this
// firmware before that, so a board wired to the old loom drives NOTHING on the
// gripper and instead pulses the bin actuator's H-bridge. Check the servo lead
// before powering the drill.
//
// 23 was BIN_PWM until this change; the actuator took the vacated pin 9.
#define GRIPPER_SERVO 23

// Lid of the FRONT-LEFT deck container -- the sand sample box, which the rest
// of the workspace calls `sand_box` (aries_load_cells/config/load_cells.yaml:
// "front-left deck box, the sand sample"). The box behind it, also on the
// left, is `stone_box` and has no lid servo.
//
// IF THIS IS THE WRONG BOX, this define and the topic name in main.cpp are the
// only two places to change. The identification came from the load-cell table,
// not from the hardware.
//
// PROPOSED, NOT CONFIRMED. Pin 10 is free and PWM-capable. It is also SPI0 CS
// -- harmless here, nothing on this board uses SPI, but worth knowing before
// anything is added that does.
//
// It used to sit next to the gripper servo on 9 so the two servo leads landed
// together; the gripper moved to 23 on 2026-08-26 and they are no longer
// adjacent.
#define LID_SERVO_SAND_BOX 10

// --- Load cells --------------------------------------------------------------
// Three HX711 amplifiers, CONFIRMED from the bench 2026-08-26.
//
// EACH CELL HAS ITS OWN CLOCK. This is not the usual one-shared-SCK chain --
// every amplifier gets a private DT/SCK pair, so they can be read
// independently and a dead amplifier cannot stall the others. Six pins, not
// four. Anything written against a shared clock (including an earlier draft of
// this file) is wrong.
//
// THE ORDER BELOW IS THE WIRE FORMAT. aries_load_cells expects one topic,
// `load_cells/raw` (std_msgs/Int32MultiArray), three elements, in the order of
// `cells` in aries_load_cells/config/load_cells.yaml:
//     ["sand_box", "stone_box", "drill_container"]
// The firmware sends no names to check itself against, so swapping two entries
// makes the sand box report the stone and nothing anywhere notices.
//
// RAW COUNTS, not kilograms. Scale and offset live in that package's YAML so a
// recalibration is an edit and a relaunch, not a reflash with the rover open.
//
// DRIVEN. main.cpp constructs one LoadCell per amplifier, in the order above,
// and publishes their counts on load_cells/raw at 10 Hz. A cell with no
// amplifier on its pins never blocks the board and never reports zero -- zero
// is what an empty box reads -- it reports the converter's negative rail, which
// the host already knows how to call a fault. See PINOUT.md.
#define HX711_SAND_BOX_DT 17          // front-left deck box, the sand sample
#define HX711_SAND_BOX_SCK 16
#define HX711_STONE_BOX_DT 34         // the box behind it, also on the left
#define HX711_STONE_BOX_SCK 33
#define HX711_DRILL_CONTAINER_DT 32   // the drill's sample bin
#define HX711_DRILL_CONTAINER_SCK 31

// --- COMPILE-TIME COLLISION CHECK -------------------------------------------
//
// Two functions on one pin is SILENT at runtime: whichever init() runs last
// wins the pin mode, and the other quietly drives, or reads, the wrong
// hardware. Nothing anywhere reports it.
//
// This is not hypothetical. On 2026-08-26 the gripper servo was moved from pin
// 9 to pin 23 -- which BIN_PWM already held. Caught by reading the file;
// the build would have been perfectly happy, and the symptom on the bench
// would have been a gripper that twitched whenever the sample bin moved.
//
// Add every new pin to kMap below. PIN_UNASSIGNED is exempt: several entries
// are legitimately unassigned and they all compare equal to each other.
#ifdef __cplusplus
namespace aries_pins
{
constexpr unsigned char kMap[] = {
    AUGER_PWM, AUGER_INA, AUGER_INB,
    FEED_PWM, FEED_INA, FEED_INB,
    BIN_PWM, BIN_INA, BIN_INB,
    LIMIT_SWITCH1, LIMIT_SWITCH2,
    STALIG_G, STALIG_Y, STALIG_R,
    GRIPPER_SERVO, LID_SERVO_SAND_BOX,
    HX711_SAND_BOX_DT, HX711_SAND_BOX_SCK,
    HX711_STONE_BOX_DT, HX711_STONE_BOX_SCK,
    HX711_DRILL_CONTAINER_DT, HX711_DRILL_CONTAINER_SCK,
    // LED_BUILTIN, spelled literally because pins.h does not include Arduino.h.
    // Here so that handing pin 13 to something else fails the build rather than
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
} // namespace aries_pins
#endif // __cplusplus

#endif  // PINS_H
