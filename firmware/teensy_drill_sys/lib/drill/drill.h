// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge

#ifndef DRILL_H
#define DRILL_H

#include <Arduino.h>
#include <HX711.h>
#include <IntervalTimer.h>
#include <Servo.h>

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

// Servo Class ====================================================================================================
// A slew-rate-limited hobby servo on a normalised 0..1 range. Nothing about it
// is specific to any one mechanism -- TWO are instantiated: the gripper jaws
// and the front-left container lid. It was called GripperServo when the gripper
// was the only one.
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
