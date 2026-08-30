// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge

#include <drill.h>

#include "pins.h"

Driver::Driver(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2)
    : m_pin_pwm(pin_pwm),
      m_pin_in1(pin_in1),
      m_pin_in2(pin_in2),
      m_usable(PIN_IS_ASSIGNED(pin_pwm) && PIN_IS_ASSIGNED(pin_in1) &&
               PIN_IS_ASSIGNED(pin_in2)) {};

void Driver::init_driver()
{
  if (!m_usable)
    return;

  // Direction pins LOW before they become outputs, for the same reason the
  // stack light does it: pinMode(OUTPUT) latches the output register, which is
  // 0 out of reset. Both LOW is the H-bridge's coast state, so the motor cannot
  // twitch between here and the first command.
  digitalWrite(m_pin_in1, LOW);
  digitalWrite(m_pin_in2, LOW);
  pinMode(m_pin_in1, OUTPUT);
  pinMode(m_pin_in2, OUTPUT);

  pinMode(m_pin_pwm, OUTPUT);
  analogWriteFrequency(m_pin_pwm, 10000);
  stop_driver();
}

void Driver::drive(int pwm_speed, bool dir)
{
  if (!m_usable)
    return;

  pwm_speed = constrain(pwm_speed, 0, 255);

  // Reversing under load shoot-throughs the bridge if both halves are ever
  // briefly on together, so cross zero first and give the low side time to turn
  // off before the direction pins move.
  //
  // The delivered code had the intent but not the effect: it wrote analogWrite
  // 0 and then wrote the new duty cycle on the very next line, with nothing in
  // between. Two register writes microseconds apart is not a dead time. The
  // motor never coasted and the bridge saw the reversal as one continuous
  // command.
  //
  // 500 us is far longer than any gate-driver turn-off and is invisible next to
  // the mechanism's own inertia. This runs in a micro-ROS callback, never in an
  // ISR, so blocking here is safe.
  if (dir != m_cache_dir && m_cache_speed != 0)
  {
    analogWrite(m_pin_pwm, 0);
    digitalWrite(m_pin_in1, LOW);
    digitalWrite(m_pin_in2, LOW);
    delayMicroseconds(500);
  }

  if (dir)
  {
    digitalWrite(m_pin_in1, HIGH);
    digitalWrite(m_pin_in2, LOW);
  }
  else
  {
    digitalWrite(m_pin_in1, LOW);
    digitalWrite(m_pin_in2, HIGH);
  }
  analogWrite(m_pin_pwm, pwm_speed);

  // store the last state of driver for reference
  m_cache_speed = pwm_speed;
  m_cache_dir = dir;
}

void Driver::stop_driver()
{
  if (!m_usable)
    return;

  analogWrite(m_pin_pwm, 0);
  digitalWrite(m_pin_in1, LOW);
  digitalWrite(m_pin_in2, LOW);
  m_cache_speed = 0;
}

AugerMotor::AugerMotor(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2)
    : Driver(pin_pwm, pin_in1, pin_in2) {};

void AugerMotor::init_motor()
{
  init_driver();
}

void AugerMotor::drive_motor(int pwm_speed, bool dir)
{
  drive(pwm_speed, dir);
}

void AugerMotor::stop_motor()
{
  stop_driver();
}

LeadScrewMotor::LeadScrewMotor(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2)
    : Driver(pin_pwm, pin_in1, pin_in2) {};

void LeadScrewMotor::init_motor()
{
  init_driver();
}

void LeadScrewMotor::drive_motor(int pwm_speed, bool dir)
{
  drive(pwm_speed, dir);
}

void LeadScrewMotor::stop_motor()
{
  stop_driver();
}

LinearActuator::LinearActuator(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2)
    : Driver(pin_pwm, pin_in1, pin_in2)
{
  m_instance_la = this;
};

LinearActuator *LinearActuator::m_instance_la = nullptr;

void LinearActuator::init_motor()
{
  init_driver();
}

uint32_t LinearActuator::move_duration_us(float req_ext) const
{
  if (!(req_ext > 0.0f))
    return 0; // also catches NaN, which compares false against everything

  if (req_ext > m_oem_max_ext)
    req_ext = m_oem_max_ext;

  return (uint32_t)((req_ext / m_oem_max_speed) * 1000.0f * 1000.0f);
}

void LinearActuator::start_move(int pwm_speed, bool dir, uint32_t duration_us)
{
  if (duration_us == 0)
  {
    // Nothing to do, and critically: do NOT switch the motor on. Starting it
    // and arming a timer that can never fire is how a zero-length move became
    // an indefinite one.
    stop_motor();
    return;
  }

  drive(pwm_speed, dir);
  m_timer.begin(isr_timer_router, duration_us);
}

void LinearActuator::extend()
{
  start_move(m_pwm_speed, true, move_duration_us(m_req_ext));
}

void LinearActuator::retract()
{
  start_move(m_pwm_speed, false, move_duration_us(m_req_ext));
}

void LinearActuator::extend(int pwm_speed, float req_ext)
{
  start_move(pwm_speed, true, move_duration_us(req_ext));
}

void LinearActuator::retract(int pwm_speed, float req_ext)
{
  start_move(pwm_speed, false, move_duration_us(req_ext));
}

void LinearActuator::stop_motor()
{
  m_timer.end();
  stop_driver();
}

void LinearActuator::isr_timer_router()
{
  if (m_instance_la != nullptr)
  {
    m_instance_la->handle_isr();
  }
}

void LinearActuator::handle_isr()
{
  m_timer.end();
  stop_driver();
}

void LinearActuator::home(bool dir)
{
  // Full stroke plus a margin, computed fresh every call -- see m_home_margin_s
  // in the header for what this used to accumulate.
  const uint32_t duration_us =
      (uint32_t)(((m_oem_max_ext / m_oem_max_speed) + m_home_margin_s) * 1000.0f * 1000.0f);

  // `dir` here is the HOME END, not the drive direction, which is why the two
  // branches drive the opposite way round to what the name suggests: homing
  // "true" runs the actuator back to its retracted end.
  start_move(m_pwm_speed, !dir, duration_us);
}

SlewServo::SlewServo(uint8_t servo_pin)
    : m_servo_pin(servo_pin),
      m_current_normalized(0.0f),
      m_target_normalized(0.0f),
      m_last_update_ms(0),
      m_usable(PIN_IS_ASSIGNED(servo_pin))
{
}

void SlewServo::init()
{
  if (!m_usable)
    return;

  m_servo.attach(m_servo_pin, m_min_us, m_max_us);
  m_servo.writeMicroseconds(normalized_to_us(m_current_normalized));

  // Start the slew clock here rather than leaving it at 0. update() computes
  // dt as millis() - m_last_update_ms, so the first call after boot saw a dt of
  // however long setup() had taken -- which with the agent handshake in front
  // of it is seconds, enough for max_step to exceed the whole 0..1 range and
  // for the servo to slam to its first target at full speed instead of slewing.
  m_last_update_ms = millis();
}

void SlewServo::set_target(float target_normalized)
{
  if (target_normalized < 0.0f)
    target_normalized = 0.0f;
  if (target_normalized > 1.0f)
    target_normalized = 1.0f;

  m_target_normalized = target_normalized;
}

void SlewServo::update()
{
  if (!m_usable)
    return;

  uint32_t now_ms = millis();
  uint32_t dt_ms = now_ms - m_last_update_ms;

  if (dt_ms == 0)
    return;
  m_last_update_ms = now_ms;

  const float max_step = m_slew_deg_per_sec / 270.0f / 1000.0f * (float)dt_ms;
  float diff = m_target_normalized - m_current_normalized;

  if (diff > max_step)
  {
    m_current_normalized += max_step;
  }
  else if (diff < -max_step)
  {
    m_current_normalized -= max_step;
  }
  else
  {
    m_current_normalized = m_target_normalized;
  }

  m_servo.writeMicroseconds(normalized_to_us(m_current_normalized));
}

int SlewServo::normalized_to_us(float t)
{
  if (t < 0.0f)
    t = 0.0f;
  if (t > 1.0f)
    t = 1.0f;
  return (int)(m_min_us + t * (m_max_us - m_min_us));
}

// Pump ==============================================================================================================
Pump *Pump::m_instance_pump = nullptr;

Pump::Pump(uint8_t pin_pwm, uint8_t pin_in1, uint8_t pin_in2)
    : Driver(pin_pwm, pin_in1, pin_in2)
{
  m_instance_pump = this;
};

void Pump::init_motor()
{
  init_driver();
}

// FIX 1 of 3. Upstream computed `pwm_dur = req_vol / flowrate` and handed it
// straight to IntervalTimer::begin(). A req_vol of 0 -- which is what an
// unset field or a cleared command carries -- makes that 0 us, an interval the
// timer cannot represent, so it NEVER FIRES. The pump was switched on one line
// earlier and had nothing left to switch it off: a zero-volume dose ran the
// pump until somebody cut the power.
//
// This is the identical bug LinearActuator had with a 0 mm extension, and it is
// solved the same way -- return 0 for anything that must not start the motor,
// and let start_move() refuse to start on a 0.
uint32_t Pump::move_duration_us(float req_vol) const
{
  if (!(req_vol > 0.0f))
    return 0; // also catches NaN, which compares false against everything

  if (req_vol > m_oem_max_vol)
    req_vol = m_oem_max_vol;

  return (uint32_t)((req_vol / m_oem_max_flowrate) * 1000.0f * 1000.0f);
}

void Pump::start_move(int pwm_flowrate, bool dir, uint32_t duration_us)
{
  if (duration_us == 0)
  {
    // Do NOT switch the pump on. See move_duration_us().
    stop_motor();
    return;
  }

  drive(pwm_flowrate, dir);
  m_timer.begin(isr_timer_router, duration_us);
}

// dir=true pushes liquid out, dir=false pulls it in. Named rather than
// repeated, because `drive(x, false)` three lines apart in two functions is
// exactly how a pump ends up running the wrong way.
void Pump::release()
{
  start_move(m_pwm_flowrate, true, move_duration_us(m_req_vol));
}

void Pump::draw()
{
  start_move(m_pwm_flowrate, false, move_duration_us(m_req_vol));
}

void Pump::release(int pwm_flowrate, float req_vol)
{
  start_move(pwm_flowrate, true, move_duration_us(req_vol));
}

void Pump::draw(int pwm_flowrate, float req_vol)
{
  start_move(pwm_flowrate, false, move_duration_us(req_vol));
}

// FIX 3 of 3. Upstream opened this with `m_home_dur = 30;` -- an assignment to
// a member, not a local -- which discarded the 14.8 s computed in the
// initialiser and left 30 s in place for every later call. The computed value
// therefore never ran even once. m_home_dur is a constant now, so there is
// nothing left to overwrite; see the note on it in drill.h.
void Pump::home(bool dir)
{
  start_move(m_pwm_flowrate, dir,
             (uint32_t)(m_home_dur * 1000.0f * 1000.0f));
}

// FIX 2 of 3. Upstream's state 5 was:
//
//     pump.home(false);
//     pump.draw();
//
// Two calls, each ending in m_timer.begin(). The second begin() re-arms the
// SAME IntervalTimer, so it replaced the first before a microsecond of the home
// had elapsed: the home was dead code and only the draw ever ran. Nothing
// reported it, because both halves drive the pump in the same direction -- the
// mechanism just ran for the shorter time.
//
// One move, not two chained timers. home(false) and draw() both drive dir=false
// at the same duty cycle, so running for the sum of their durations is exactly
// the intended sequence. Chaining would mean calling IntervalTimer::begin()
// from inside that timer's own ISR, which is a great deal more fragile than
// adding two numbers.
void Pump::home_then_draw()
{
  const uint32_t home_us = (uint32_t)(m_home_dur * 1000.0f * 1000.0f);
  const uint32_t draw_us = move_duration_us(m_req_vol);

  if (draw_us == 0)
  {
    // A zero dose means there is no draw to append; run the home alone rather
    // than treating the whole command as a no-op.
    start_move(m_pwm_flowrate, false, home_us);
    return;
  }

  start_move(m_pwm_flowrate, false, home_us + draw_us);
}

void Pump::stop_motor()
{
  m_timer.end();
  stop_driver();
}

void Pump::isr_timer_router()
{
  if (m_instance_pump != nullptr)
    m_instance_pump->handle_isr();
}

void Pump::handle_isr()
{
  m_timer.end();
  stop_driver();
}

// LidServo ==========================================================================================================
LidServo *LidServo::m_instance_lid = nullptr;

LidServo::LidServo(uint8_t servo_pin, int neutral_us, int max_deviation_us)
    : m_servo_pin(servo_pin),
      m_neutral_us(neutral_us),
      m_max_deviation_us(max_deviation_us),
      m_current_speed(0.0f),
      m_last_cmd_ms(0),
      m_usable(PIN_IS_ASSIGNED(servo_pin))
{
  m_instance_lid = this;
}

void LidServo::init()
{
  if (!m_usable)
    return;

  m_servo.attach(m_servo_pin);

  // Command the neutral pulse IMMEDIATELY. An unattached pin is idle, but an
  // attached one with no pulse written yet is undefined, and on a
  // continuous-rotation servo "undefined" is a lid that turns at power-up with
  // nobody in the loop yet. stop() writes neutral before anything else can.
  stop();
  m_last_cmd_ms = millis();
}

void LidServo::set_speed(float speed)
{
  if (!m_usable)
    return;

  m_current_speed = constrain(speed, -1.0f, 1.0f);
  m_last_cmd_ms = millis();

  // A timed rotate_for() move is superseded by an explicit speed command --
  // otherwise its timer would fire partway through the new move and stop a lid
  // the operator is still driving.
  m_timer.end();

  const int us = m_neutral_us + (int)lroundf(m_current_speed * (float)m_max_deviation_us);
  m_servo.writeMicroseconds(us);
}

void LidServo::stop()
{
  if (!m_usable)
    return;

  m_current_speed = 0.0f;
  m_servo.writeMicroseconds(m_neutral_us);
}

void LidServo::rotate_for(float speed, float duration_s)
{
  if (!m_usable)
    return;

  // Guard the timer period for the same reason LinearActuator::move_duration_us
  // does: IntervalTimer::begin() with a 0 us period never fires, so the servo
  // that was just started would never be stopped.
  if (!(duration_s > 0.0f))
  {
    stop();
    return;
  }

  set_speed(speed);
  m_timer.begin(isr_timer_router, (unsigned int)(duration_s * 1000.0f * 1000.0f));
}

void LidServo::isr_timer_router()
{
  if (m_instance_lid != nullptr)
    m_instance_lid->handle_isr();
}

void LidServo::handle_isr()
{
  m_timer.end();
  stop();
}

void LidServo::update(uint32_t command_timeout_ms)
{
  if (!m_usable)
    return;

  // Already stopped: nothing to time out, and re-writing neutral every loop
  // would keep resetting nothing useful.
  if (m_current_speed == 0.0f)
    return;

  if (millis() - m_last_cmd_ms > command_timeout_ms)
  {
    // The host has gone quiet with the lid still turning. It cannot be left
    // turning: this servo has no stop of its own and no end of travel to reach.
    m_timer.end();
    stop();
  }
}

LimitSwitch *LimitSwitch::m_instances_ls[LIMIT_SWITCH_INSTANCES] = {nullptr};
uint8_t LimitSwitch::m_instance_count_ls = 0;

void LimitSwitch::isr_router_ls_0()
{
  if (m_instances_ls[0])
    m_instances_ls[0]->handle_isr();
}

void LimitSwitch::isr_router_ls_1()
{
  if (m_instances_ls[1])
    m_instances_ls[1]->handle_isr();
}

LimitSwitch::LimitSwitch(uint8_t pin_switch, uint32_t debounce_ms)
    : m_pin_switch(pin_switch),
      m_debounce_ms(debounce_ms),
      m_triggered(false),
      m_last_interrupt_time(0),
      m_usable(PIN_IS_ASSIGNED(pin_switch)) {};

void LimitSwitch::init()
{
  if (!m_usable)
    return;

  pinMode(m_pin_switch, INPUT_PULLUP);
  if (m_instance_count_ls < LIMIT_SWITCH_INSTANCES)
  {
    uint8_t id = m_instance_count_ls;
    m_instances_ls[id] = this;
    m_instance_count_ls++;

    void (*isr_func)() = nullptr;
    if (id == 0)
      isr_func = isr_router_ls_0;
    else if (id == 1)
      isr_func = isr_router_ls_1;

    if (isr_func != nullptr)
    {
      // FALLING, not RISING. The switch shorts the pin to GND and the pin is
      // INPUT_PULLUP, so CLOSING it is a HIGH->LOW edge; RISING fired on
      // RELEASE instead, marking the switch as tripped exactly as the carriage
      // left the stop. Latent until now only because is_triggered() has no
      // caller -- the motion gate uses the LEVEL, is_at_stop(). See pins.h,
      // which has described the wiring as FALLING all along.
      attachInterrupt(digitalPinToInterrupt(m_pin_switch), isr_func, FALLING);
    }
  }
}

void LimitSwitch::handle_isr()
{
  uint32_t current_time = millis();
  // debounce
  if (current_time - m_last_interrupt_time > m_debounce_ms)
  {
    m_triggered = true;
    m_last_interrupt_time = current_time;
  }
}

bool LimitSwitch::is_triggered()
{
  if (m_triggered)
  {
    m_triggered = false;
    return true;
  }
  return false;
}

bool LimitSwitch::is_at_stop() const
{
  // An unassigned switch reports NOT at the stop. That is the permissive
  // answer, and it is the honest one: there is no switch, so there is no
  // evidence of a stop. The motion it allows is the motion the mechanism had
  // before anybody wired a switch, and PINOUT.md says so in as many words.
  if (!m_usable)
    return false;

  return digitalRead(m_pin_switch) == LOW;
}

LoadCell::LoadCell(uint8_t pin_dout, uint8_t pin_sck)
    : m_pin_dout(pin_dout),
      m_pin_sck(pin_sck),
      m_usable(PIN_IS_ASSIGNED(pin_dout) && PIN_IS_ASSIGNED(pin_sck)),
      m_raw(0),
      m_has_reading(false),
      m_last_read_ms(0) {};

void LoadCell::init()
{
  if (!m_usable)
    return;

  // Gain 128, channel A -- the input the amplifier boards wire the bridge to,
  // and the part's power-up default, so this is stating the existing state
  // rather than changing it. Channel B exists at gain 32 and nothing here uses
  // it; selecting it costs an extra clock pulse per read and halves the
  // effective rate, because the channel only changes on the NEXT conversion.
  m_hx711.begin(m_pin_dout, m_pin_sck, 128);

  // NO PRIMING READ HERE, deliberately. begin() ends with a read() in some
  // versions of this library and the obvious next step is to take one more to
  // have a number in hand -- but setup() runs before the agent connects and
  // before the motors are known to be stopped, and read() blocks forever on an
  // amplifier that is not there. The first count arrives from update(), on the
  // loop, where waiting for it costs nothing.
}

bool LoadCell::update()
{
  if (!m_usable)
    return false;

  // is_ready() is a single digitalRead of DOUT: LOW means a conversion is
  // waiting. Everything about not blocking this board hangs off asking that
  // question first -- see the class comment in drill.h.
  if (!m_hx711.is_ready())
    return false;

  m_raw = m_hx711.read();
  m_has_reading = true;
  m_last_read_ms = millis();
  return true;
}

int32_t LoadCell::reported(uint32_t now_ms) const
{
  if (!m_usable || !m_has_reading)
    return kRail;
  if (now_ms - m_last_read_ms > kStaleMs)
    return kRail;

  // The HX711 cannot produce anything outside its 24-bit signed range, so this
  // clamp is not about the converter. It is about the sentinel: a genuine
  // reading that happened to land exactly on kRail would be read by the host as
  // a dead cell. Pushing it one count off is a lie of 1/8388608 of full scale,
  // and it keeps "reported the rail" meaning exactly one thing.
  if (m_raw <= kRail)
    return kRail + 1;
  return (int32_t)m_raw;
}
