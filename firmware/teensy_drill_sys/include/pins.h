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
//   - The PWM pins here (25, 15, 22) are all PWM-capable. analogWriteFrequency
//     sets the frequency of the whole timer submodule a pin belongs to, not the
//     pin -- but all three drivers ask for the same 10 kHz in
//     Driver::init_driver(), so a shared timer is harmless. It would NOT be if
//     one of them ever wanted a different frequency. From pwm_pin_info[] in the
//     core's cores/teensy4/pwm.c the three land on three different peripherals
//     -- 25 = FlexPWM1_3_X, 15 = QuadTimer3_3, 22 = FlexPWM4_0_A -- so nothing
//     shares a submodule today.
//   - PIN 25 IS AN X CHANNEL, not the A or B the other PWM pins use. That is
//     fine: flexpwmWrite() handles `case 0: // X` by writing VAL0, so
//     analogWrite() drives it properly. Unlike pin 38, which has no PWM at all
//     on a 4.1.
//   - PINS 7, 8 AND 25 ARE ALL ON FLEXPWM1 SUBMODULE 3. Harmless as written --
//     only 25 is driven as PWM, 8 is a plain digitalWrite direction pin and 7
//     is an INPUT_PULLUP limit switch, and neither cares what frequency that
//     submodule runs at. It stops being harmless if anything ever calls
//     analogWrite() on 7 or 8, so do not put a second PWM function there.

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
// MOVED TO 25 / 8 / 9 ON 2026-08-29, from 22 / 19 / 18. THE PROVENANCE OF THE
// NEW NUMBERS IS NOT RECORDED -- they arrived as an edit to this file with no
// bench note attached, so unlike the numbers they replaced there is nothing
// here saying anybody metered them. PINOUT.md marks this row unconfirmed for
// that reason. If they did come off the bench, say so there and here.
//
// The sample bin took the three pins this vacated; see BIN_* below.
//
// WHAT THE OLD NUMBERS HAD BEHIND THEM, kept because it is what this row has to
// be re-confirmed against: 22 / 19 / 18 was confirmed from the bench on
// 2026-08-26 and was exactly what pin-def-ref.txt said all along. That briefly
// looked stale when an interim bench report put the SAMPLE BIN's PWM on 22,
// colliding with the auger, and the auger's PWM was set PIN_UNASSIGNED while it
// was resolved. It resolved the other way: the bin went to 28 and the file was
// right about the auger. (The bin IS on 22 now -- but by this later change, not
// by that report being right.)
#define AUGER_PWM 25
#define AUGER_INA 8
#define AUGER_INB 9

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
// q = -0.1304 (back under the auger); see aries/urdf/drill.xacro. Which end it
// is at matters for reading the bin's own load cell, and the bin has no
// position sensor -- see PINOUT.md.
//
// MOVED TO 22 / 19 / 18 ON 2026-08-29, from 28 / 30 / 29 -- the three pins the
// auger vacated in the same change. 22 is PWM-capable (FlexPWM4_0_A). AS WITH
// THE AUGER, THE PROVENANCE OF THESE NUMBERS IS NOT RECORDED here, and PINOUT.md
// marks the row unconfirmed.
//
// 28, 29 and 30 are now free, and have been added to kScanPins in main.cpp.
//
// The numbers this replaced, 28 / 30 / 29, were confirmed from the bench on
// 2026-08-26. Before that they were proposals (23/21/20, then 22/21/20);
// nothing about those earlier ones survives.
#define BIN_PWM 22
#define BIN_INA 19
#define BIN_INB 18

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
// interrupt edge is the closure. (Pins 14, 21, 24 and 27 DO rest low on this
// harness; they are something else's, and are not these switches. Pin 9 was on
// that list too and is now AUGER_INB -- consistent with its having been an
// H-bridge input on the loom all along, but that is an observation and NOT
// confirmation of the 2026-08-29 auger map.)
//
// The BIN's two switches are not in this map. The mechanism has four switches
// across two axes; the firmware has two instances (LIMIT_SWITCH_INSTANCES in
// drill.h, which caps the ISR router table). Wiring the bin's pair needs that
// constant raised to 4 and two more routers added. Until then the bin is
// dead-reckoned from the commanded rate, which drifts from the first slip.
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
// gripper and instead pulses whatever now sits on pin 9. Check the servo lead
// before powering the drill.
//
// 23 had been the bin actuator's proposed PWM until this change. PIN 9 IS NOT A
// SPARE ANY MORE: since 2026-08-29 it is AUGER_INB, so an old-loom gripper lead
// now feeds a direction input on the auger's H-bridge.
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
// PIN 38 SINCE 2026-08-29. It was 10 for the whole life of this firmware, and
// before that 9, next to the gripper servo.
//
// 38 IS NOT A PWM PIN ON A TEENSY 4.1, AND THAT DOES NOT MATTER. FlexPWM1_2
// reaches pin 38 on a Teensy 4.0; on the 4.1 that pad is pins 46/47 instead
// (see the table in the core's pwm.c). Nothing here needs it: the Teensy Servo
// library does not use PWM hardware at all -- Servo::attach() does
// pinMode(pin, OUTPUT) and the pulses are bit-banged with digitalWrite from a
// timer ISR, so any digital pin works. analogWrite() on this pin would NOT
// work, and nothing calls it here.
//
// 38 IS ALSO A PIN THE DIAGNOSTIC SCAN USED TO WATCH. It was in kScanPins in
// main.cpp -- the list of otherwise-unused pins the board reports on
// drill/pin_scan -- and it has been removed from it. Leaving it there would
// have had setup() configure it INPUT_PULLUP, Servo::attach() then take it back
// as an OUTPUT, and pin_scan_state() read the servo's own pulse train back as
// "something is pulling this pin low", flickering a diagnostic bit at 50 Hz
// forever. ADD THE PIN YOU FREE, AND REMOVE THE PIN YOU TAKE: the two lists
// have to stay disjoint and nothing checks that they are.
//
// A CONTINUOUS-ROTATION SERVO, NOT A POSITIONAL ONE, also since 2026-08-29. It
// is driven by LidServo rather than SlewServo and takes a SPEED in -1..1, not
// an angle. See the LidServo comment in drill.h, and LID_SERVO_NEUTRAL_US
// below.
#define LID_SERVO_SAND_BOX 38

// THE PULSE WIDTH AT WHICH THE LID SERVO ACTUALLY HOLDS STILL, in microseconds.
//
// NOT A CONSTANT OF THE PART -- it is a property of THIS INDIVIDUAL SERVO, set
// by how its centring potentiometer was trimmed on the line, and 1500 is only
// where they cluster. It is the one number here that cannot be looked up, and
// it is the first thing to check if the lid creeps with the stick released:
// creep at rest is this number being wrong and is nothing else.
//
// To trim it: `ros2 topic pub --once /sand_box/lid/cmd std_msgs/Float32
// "data: 0.0"`, watch the horn, and move this up or down 10 us at a time until
// it is genuinely still. It needs a reflash each time -- see the note in
// drill.h about what moving calibration into the firmware costs.
#define LID_SERVO_NEUTRAL_US 1500

// Offset from neutral that means full speed, in microseconds. 350-500 is the
// usual range for a hobby servo; 400 is upstream's value and is a starting
// point, not a measurement.
#define LID_SERVO_MAX_DEVIATION_US 400

// --- Load cells --------------------------------------------------------------
// Three HX711 amplifiers, CONFIRMED from the bench 2026-08-26.
//
// EACH CELL HAS ITS OWN CLOCK. This is not the usual one-shared-SCK chain --
// every amplifier gets a private DT/SCK pair, so they can be read
// independently and a dead amplifier cannot stall the others. Six pins, not
// four. Anything written against a shared clock (including an earlier draft of
// this file) is wrong.
//
// CALIBRATED WEIGHTS, NOT RAW COUNTS, SINCE 2026-08-29 -- and this reverses the
// decision that used to be recorded here. The board published raw converter
// counts on one `load_cells/raw` array and a host package (aries_load_cells)
// scaled them from YAML, so that a recalibration was an edit and a relaunch
// rather than a reflash with the rover open. That package has been REMOVED. The
// scale factors below are compiled in, the board publishes weights on three
// separate topics, and taring is a message to the board rather than a ROS
// service. The old argument still stands and is now simply a cost that has been
// accepted: changing a number here means a reflash.
//
// SCALE FACTOR IS COUNTS PER UNIT OF WEIGHT, the same sense as
// HX711::set_scale() -- weight = (counts - zero) / scale. These three values
// came with the embedded team's firmware; they have NOT been re-derived here,
// and there is no note anywhere of which unit they are in. Treat the numbers on
// the wire as provisional until one known mass has been put in each box.
//
// TO RECALIBRATE a cell:
//   1. Empty the box, then  ros2 topic pub --once /<cell>/tare std_msgs/UInt8 "data: 1"
//   2. Put a KNOWN mass in and read /<cell>/weight
//   3. new_scale = old_scale * (reading / true_mass)
//   4. Edit the value here and reflash.
//
// THE SIGN IS WHICH WAY THE CELL IS BOLTED IN, not an error. The drill
// container's factor is negative because that cell reads the opposite way for a
// positive load; leaving it positive makes a filling bin report a lightening
// one.
#define HX711_SAND_BOX_DT 17          // front-left deck box, the sand sample
#define HX711_SAND_BOX_SCK 16
#define HX711_SAND_BOX_SCALE 20.0f

// `stone_box` in this workspace, `rock_box` in the embedded team's firmware and
// on the topic name. Same box.
#define HX711_STONE_BOX_DT 34         // the box behind it, also on the left
#define HX711_STONE_BOX_SCK 33
#define HX711_STONE_BOX_SCALE 21.0f

#define HX711_DRILL_CONTAINER_DT 32   // the drill's sample bin
#define HX711_DRILL_CONTAINER_SCK 31
#define HX711_DRILL_CONTAINER_SCALE -30.0f

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
