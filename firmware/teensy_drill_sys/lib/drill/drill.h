// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge

#ifndef DRILL_H
#define DRILL_H

#include <Arduino.h>
#include <HX711.h>
#include <IntervalTimer.h>
#include <Servo.h>
#include <math.h>

#define LIMIT_SWITCH_INSTANCES 2

// Motor Classes ====================================================================================================
class Driver
{
public:
  Driver(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2);
  void init_driver();
  void drive(int pwm_speed, bool dir);
  void stop_driver();

  // False when any of the three pins is PIN_UNASSIGNED. Every entry point
  // checks it, so an un-numbered driver is inert rather than driving pin 255.
  bool usable() const { return m_usable; }

private:
  uint8_t m_pin_pwm;
  uint8_t m_pin_in1;
  uint8_t m_pin_in2;

  // Were uninitialised. drive() compares the incoming direction against
  // m_cache_dir to decide whether it has to cross zero first, so on the very
  // first command that comparison read an indeterminate value -- meaning the
  // dead time either happened or did not depending on what was in RAM at boot.
  int m_cache_speed = 0;
  bool m_cache_dir = true;
  bool m_usable = false;
};

class AugerMotor : private Driver
{
public:
  // Re-exported through PRIVATE inheritance, which otherwise hides every
  // Driver member from callers. main.cpp's led_update() asks each axis whether
  // its pins are assigned, and without this the board simply does not compile.
  using Driver::usable;

  AugerMotor(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2);
  void init_motor();
  void drive_motor(int pwm_speed, bool dir);
  void stop_motor();
};

class LeadScrewMotor : private Driver
{
public:
  // Re-exported through PRIVATE inheritance, which otherwise hides every
  // Driver member from callers. main.cpp's led_update() asks each axis whether
  // its pins are assigned, and without this the board simply does not compile.
  using Driver::usable;

  LeadScrewMotor(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2);
  void init_motor();
  void drive_motor(int pwm_speed, bool dir);
  void stop_motor();
};

class LinearActuator : private Driver
{
public:
  // Re-exported through PRIVATE inheritance, which otherwise hides every
  // Driver member from callers. main.cpp's led_update() asks each axis whether
  // its pins are assigned, and without this the board simply does not compile.
  using Driver::usable;

  LinearActuator(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2);
  void init_motor();
  void extend();
  void retract();
  void home(bool dir);
  void stop_motor();

  // functions in case we need to send modified values
  void extend(int pwm_speed, float req_ext);
  void retract(int pwm_speed, float req_ext);

private:
  // Shared by every timed move: converts a requested extension in mm into a
  // run time in microseconds, clamps it to the actuator's real stroke, and
  // returns 0 for anything that should not start the motor at all.
  //
  // The 0 case is not hypothetical. bin_actuator/cext with data exactly 0.0 took the
  // `else` branch in the host callback and asked for a 0 mm retraction, which
  // reached IntervalTimer::begin() as a 0 us period -- an interval the timer
  // cannot represent, so it never fires, so the motor that was just switched on
  // is never switched off. A zero-length move ran the actuator into its stop.
  uint32_t move_duration_us(float req_ext) const;

  void start_move(int pwm_speed, bool dir, uint32_t duration_us);

  const float m_oem_max_speed = 15.0; // [mm/s]
  const float m_oem_max_ext = 100.0;  // [mm]

  int m_pwm_speed = 255;  // 100% duty-cycle for max power
  float m_req_ext = 75.0; // [mm]

  // Homing runs the actuator at full stroke plus a margin, so it reaches the
  // end regardless of where it started and stalls there.
  //
  // This margin used to be applied with `m_home_dur += 1` inside home(), on a
  // member that persisted -- so the first home ran 7.67 s, the second 8.67 s,
  // the third 9.67 s, and every home for the rest of the session sat on the end
  // stop a second longer than the one before. Computed fresh each call now.
  const float m_home_margin_s = 1.0;

  IntervalTimer m_timer;
  static LinearActuator *m_instance_la;
  static void isr_timer_router();
  void handle_isr();
};

// PERISTALTIC PUMP -- new from the embedded team, 2026-08-30. A DC motor on an
// H-bridge like the other three, but commanded in MILLILITRES rather than
// millimetres: it runs open-loop for however long that volume takes at the
// measured flow rate. There is no flow sensor and no level sensor, so a
// commanded volume is a TIMER, not a measurement, and it drifts with head
// height, tube wear and battery state exactly as the drill's axes do.
//
// draw() pulls liquid in, release() pushes it out. home() runs it long enough
// to clear the whole tube in whichever direction it is given.
//
// THREE THINGS WERE FIXED FROM THE DELIVERED CLASS, all of them shapes that had
// already bitten LinearActuator, which this was copy-pasted from. Each is
// called out at its implementation in drill.cpp:
//
//   1. A zero volume armed an IntervalTimer with a 0 us period, which never
//      fires -- so the pump was switched on and never switched off.
//   2. State 5 ("home then draw") called home() and draw() back to back. The
//      second m_timer.begin() replaced the first immediately, so the home never
//      ran at all.
//   3. home() assigned m_home_dur = 30, permanently discarding the computed
//      value for the rest of the session.
class Pump : private Driver
{
public:
  // Re-exported through PRIVATE inheritance, as on the other three axes, so
  // main.cpp's led_update() can ask whether this axis has its pins.
  using Driver::usable;

  Pump(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2);
  void init_motor();

  void release();
  void draw();
  void release(int pwm_flowrate, float req_vol);
  void draw(int pwm_flowrate, float req_vol);

  // Clear the tube in the given direction. Fixed duration -- there is nothing
  // to sense arrival against.
  void home(bool dir);

  // home() in the DRAW direction and then draw(), as one continuous run.
  //
  // Upstream expressed this as two calls and it did not work; see note 2 above
  // and the implementation. It is one move here because both halves drive the
  // same direction at the same duty cycle, so the combined run is exactly
  // equivalent to the intended sequence and needs no timer chaining -- and
  // chaining would mean calling IntervalTimer::begin() from inside its own ISR.
  void home_then_draw();

  void stop_motor();

private:
  // Millilitres per second at full duty, averaged over three timed volumes
  // measured by the embedded team: 100 mL/15 s, 150 mL/22 s, 200 mL/29.5 s.
  // Works out at 6.755 mL/s. Their numbers, not re-measured here.
  static constexpr float m_oem_max_flowrate =
      (((100.0f / 15.0f) + (150.0f / 22.0f) + (200.0f / 29.5f)) / 3.0f); // [mL/s]

  // Ceiling on a single commanded volume. The tube does not hold more than
  // this, so a larger request is a mistake rather than a longer run -- and
  // without a clamp a wild number is an arbitrarily long unattended pump.
  static constexpr float m_oem_max_vol = 200.0f; // [mL]

  int m_pwm_flowrate = 255; // 100% duty-cycle for max power
  float m_req_vol = 75.0f;  // [mL] default dose

  // HOW LONG A HOME RUNS. Upstream computed 100 mL / 6.755 mL/s = 14.8 s in the
  // initialiser and then threw it away with `m_home_dur = 30;` on the first
  // line of home(), so 30 s is what the mechanism has actually been doing. Kept
  // at 30 s deliberately, as a constant that cannot be overwritten, rather than
  // silently changing the behaviour to the value that never ran.
  static constexpr float m_home_dur = 30.0f; // [s]

  uint32_t move_duration_us(float req_vol) const;
  void start_move(int pwm_flowrate, bool dir, uint32_t duration_us);

  IntervalTimer m_timer;
  static Pump *m_instance_pump;
  static void isr_timer_router();
  void handle_isr();
};

// Servo Classes ===================================================================================================
// A slew-rate-limited POSITIONAL hobby servo on a normalised 0..1 range. Only
// the gripper jaws use it. The sand box lid used to as well, and does not any
// more -- it is a continuous-rotation servo and has its own class below.
//
// The slew limiter is the point of the class. Commands arrive over USB serial
// with whatever jitter the link has, and writing each one straight to the servo
// turns that jitter into visible jerking; slewing at a fixed rate decouples the
// motion from the arrival timing.
class SlewServo
{
public:
  SlewServo(uint8_t servo_pin);
  void init();
  void set_target(float target_normalized);
  void update();

  // What the servo is actually being commanded to right now -- the slewed
  // value, not the target. This is what /gripper/state carries.
  //
  // AN ECHO, NOT A MEASUREMENT. Neither servo has a feedback line wired, so
  // this is the command on its way out. The gripper host side knows that
  // (teensy_gripper_system.cpp reads it purely as a liveness and delivery
  // signal, which is why USE_SERVO_FEEDBACK was false in the retired sketch
  // too), but it must not be mistaken for a reading of where anything is. If
  // the jaws are obstructed this still says they closed; if the lid is jammed
  // this still says it opened.
  float current() const { return m_current_normalized; }

  bool usable() const { return m_usable; }

private:
  int normalized_to_us(float t);

  uint8_t m_servo_pin;
  Servo m_servo;

  float m_current_normalized;
  float m_target_normalized;
  uint32_t m_last_update_ms;
  bool m_usable;

  // Use static constexpr to compile these directly into flash, saving RAM
  static constexpr int m_min_us = 850;
  static constexpr int m_max_us = 2200;
  static constexpr float m_slew_deg_per_sec = 550.0f;
};

// A 360-DEGREE CONTINUOUS-ROTATION SERVO, WHICH IS NOT A SERVO IN THE SENSE
// SlewServo ABOVE MEANS. Pulse width here is SPEED AND DIRECTION, not an angle:
// there is no position feedback and no position to command. It spins while it
// is being told to and does not stop on its own.
//
// This is the sand box lid, and it used to be driven as a SlewServo -- a
// normalised 0..1 POSITION. That was an assumption, never verified, and it was
// wrong: writing an angle to a continuous-rotation servo does not park it
// anywhere, it picks two fixed speeds and turns at one of them forever.
//
// neutral_us IS PER-UNIT AND MUST BE TRIMMED ON THE BENCH. It is the pulse
// width at which this individual servo genuinely holds still, set by how its
// internal centring pot was trimmed at the factory, and it is often not 1500.
// If the lid creeps at rest, that is this number and nothing else. Trim it
// until speed 0 is actually still.
//
// max_deviation_us is the offset from neutral that means full speed in each
// direction -- 350-500 us on typical hobby servos.
class LidServo
{
public:
  LidServo(uint8_t servo_pin, int neutral_us = 1500, int max_deviation_us = 400);
  void init();

  // -1.0 .. +1.0. 0.0 is stop. Sign is direction; which sign OPENS the lid is
  // a property of how the servo is mounted, and is chosen on the host side
  // (invert_lid in joystick.yaml) rather than here.
  void set_speed(float speed);
  void stop();
  float get_speed() const { return m_current_speed; }

  // Run at `speed` for `duration_s`, then stop from an IntervalTimer ISR.
  // Nothing on the host uses this yet; it is here for a mission script that
  // wants "open the lid for two seconds" without holding the loop.
  void rotate_for(float speed, float duration_s);

  // CALL EVERY LOOP. THIS IS THE WATCHDOG AND IT IS NOT OPTIONAL.
  //
  // SlewServo needs no equivalent: a position command has a resting state, so
  // a servo that stops hearing from the host simply holds where it is. A SPEED
  // command has no resting state at all. Drop the link -- or drop the single
  // message carrying the zero -- while this is turning and it turns until
  // somebody cuts the power.
  //
  // UPSTREAM DECLARED THIS AND NEVER DEFINED IT. The class as delivered has
  // `void update(uint32_t command_timeout_ms = 500);` in the header, an
  // m_last_cmd_ms member marked NEW, and no implementation and no caller
  // anywhere -- so the lid had no timeout, and calling it would not have
  // linked. It is implemented here.
  void update(uint32_t command_timeout_ms = 500);

  bool usable() const { return m_usable; }

private:
  uint8_t m_servo_pin;
  Servo m_servo;
  int m_neutral_us;
  int m_max_deviation_us;
  float m_current_speed;
  uint32_t m_last_cmd_ms;
  bool m_usable;

  IntervalTimer m_timer;
  static LidServo *m_instance_lid;
  static void isr_timer_router();
  void handle_isr();
};

// Switch Class ====================================================================================================
// Wired to GND through the switch and read INPUT_PULLUP: CLOSED reads LOW.
class LimitSwitch
{
public:
  LimitSwitch(uint8_t pin_switch, uint32_t debounce_ms);
  void init();

  // EDGE. True once per closure, then false until the switch opens and closes
  // again. Reading it CLEARS it. Use for "it just tripped" -- logging, an
  // event -- never for "may I move".
  bool is_triggered();

  // LEVEL. True for as long as the switch is held closed. This is the one to
  // gate motion on.
  //
  // The delivered firmware had only the edge, and stopped the motor from it.
  // That stops the motor exactly once: is_triggered() clears the flag, so the
  // very next command drove straight back into the stop with nothing to stop it
  // again -- the switch protected the mechanism for one loop iteration and then
  // let the operator lean on it. A limit switch has to be a condition.
  bool is_at_stop() const;

  bool usable() const { return m_usable; }

private:
  uint8_t m_pin_switch;
  uint32_t m_debounce_ms;
  volatile bool m_triggered;
  volatile uint32_t m_last_interrupt_time;
  bool m_usable;

  void handle_isr();

  static LimitSwitch *m_instances_ls[LIMIT_SWITCH_INSTANCES];
  static uint8_t m_instance_count_ls;

  static void isr_router_ls_0();
  static void isr_router_ls_1();
};

// Load Cell Class ====================================================================================================
// ONE HX711 AMPLIFIER. Three of these are constructed in main.cpp -- sand box,
// stone box, drill bin -- and their counts go out together on load_cells/raw.
//
// RAW CONVERTER COUNTS, AND NOTHING ELSE. No tare, no scale, no grams. The
// host half (aries_load_cells) owns the calibration, in that package's YAML, on
// purpose: a recalibration is then an edit and a relaunch rather than a reflash
// with the rover open. That is also why there is no tare_*() here -- taring is
// `ros2 service call /load_cells/<cell>/tare`, and a second tare living in the
// firmware would silently fight it.
//
// EACH CELL HAS ITS OWN CLOCK. Not the usual one-shared-SCK chain: every
// amplifier gets a private DT/SCK pair, so this class owns its HX711 outright
// and one dead amplifier cannot stall the others. See pins.h.
//
// NON-BLOCKING, WHICH IS THE WHOLE REASON THIS IS NOT JUST HX711::read().
// HX711::read() opens with wait_ready(), which spins until DOUT falls -- and an
// amplifier that is unplugged, unpowered or dead holds DOUT HIGH FOREVER. On a
// board whose main loop is also the auger's watchdog and the limit switches'
// motion gate, that is not a stalled sensor, it is a cutting tool that nothing
// can stop any more. update() polls is_ready() and returns without touching the
// bus when the answer is no, so a missing cell costs one digitalRead.
class LoadCell
{
public:
  LoadCell(uint8_t pin_dout, uint8_t pin_sck);

  void init();

  // Call every loop. True when a fresh conversion was collected on THIS call,
  // which is roughly 10 times a second -- the HX711's own rate at its default
  // 10 SPS strapping. Costs one digitalRead the rest of the time.
  //
  // Interrupts are off inside HX711::read() for about 60 us on this part: a
  // clock pulse stretched past 60 us puts the converter into power-down mid
  // word and every subsequent bit reads back as 1. The limit-switch ISR can
  // therefore be delayed by that much, which is harmless here -- it only sets
  // a flag nothing reads, and the motion gate is a raw level read.
  bool update();

  // False when either pin is PIN_UNASSIGNED, exactly as the motor drivers use
  // it. An unusable cell never touches a pin and always reports the rail.
  bool usable() const { return m_usable; }

  // True once this amplifier has ever produced a conversion -- i.e. there is
  // really an HX711 on those two pins. main.cpp does not publish the array at
  // all until at least one cell can say yes, so a rover with no cells fitted
  // stays quiet instead of reporting three permanent faults.
  bool has_reading() const { return m_has_reading; }

  // The last conversion, in raw 24-bit signed counts.
  long raw() const { return m_raw; }

  // WHAT TO PUT ON THE WIRE for this cell, which is not always raw().
  //
  // A cell that has stopped converting must not keep publishing its last
  // number, and it must not publish zero either: zero is what an EMPTY BOX
  // reads, so a silently dead amplifier would look exactly like a box somebody
  // had emptied. It reports the converter's negative rail instead, which
  // aries_load_cells already treats as "unplugged, wired backwards or crushed"
  // (raw_min in load_cells.yaml) and turns into a NaN weight plus a named
  // fault, rather than a confident wrong kilogram.
  int32_t reported(uint32_t now_ms) const;

  // -(1 << 23). The HX711 is a 24-bit signed converter and this is the bottom
  // of its range; it is `raw_min` in aries_load_cells/config/load_cells.yaml
  // and the two have to agree for the fault path above to fire.
  static constexpr int32_t kRail = -8388608L;

private:
  uint8_t m_pin_dout;
  uint8_t m_pin_sck;
  bool m_usable;

  HX711 m_hx711;

  long m_raw;
  bool m_has_reading;
  uint32_t m_last_read_ms;

  // How long a latched count stays believable. Five missed conversions at
  // 10 SPS -- long enough that ordinary jitter never trips it, short enough
  // that a cable pulled mid-task shows up within half a second. The host's own
  // timeout_s of 2.0 s is the coarser backstop for the whole board going away.
  static constexpr uint32_t kStaleMs = 500;
};

#endif
