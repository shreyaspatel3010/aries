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

LoadCell::LoadCell(uint8_t pin_dout, uint8_t pin_sck, float scale_factor)
    : m_pin_dout(pin_dout),
      m_pin_sck(pin_sck),
      m_usable(PIN_IS_ASSIGNED(pin_dout) && PIN_IS_ASSIGNED(pin_sck)),
      m_scale_factor(scale_factor),
      m_raw(0),
      m_has_reading(false),
      m_last_read_ms(0),
      m_zero_counts(0.0f),
      m_lid_counts(0.0f),
      m_zeroed(false),
      m_filtered_counts(0.0f),
      m_buffer_idx(0),
      m_buffer_fill(0),
      m_is_stable(false)
{
  for (uint8_t i = 0; i < kStabilityBufferSize; ++i)
    m_recent[i] = 0.0f;
}

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

  // BOUNDED BOOT ZERO. HX711::tare() would be the obvious call and it is the
  // one thing this must not do: it averages ten conversions through
  // wait_ready(), which has no timeout, so an amplifier that answers once and
  // then stops takes setup() with it -- and setup() runs before the agent is
  // up, so a board hung here is indistinguishable from one that was never
  // flashed. Poll instead, take whatever arrives inside the window, and give
  // up cleanly.
  const uint32_t deadline = millis() + kInitTimeoutMs;
  float sum = 0.0f;
  uint8_t n = 0;
  while (n < kInitSamples && (int32_t)(millis() - deadline) < 0)
  {
    if (m_hx711.is_ready())
    {
      const long v = m_hx711.read();
      sum += (float)v;
      ++n;
      m_raw = v;
      m_has_reading = true;
      m_last_read_ms = millis();
    }
  }

  if (n > 0)
  {
    m_filtered_counts = sum / (float)n;
    m_zero_counts = m_filtered_counts;
    m_zeroed = true;
    for (uint8_t i = 0; i < kStabilityBufferSize; ++i)
      m_recent[i] = m_filtered_counts;
    m_buffer_fill = kStabilityBufferSize;
  }
  // else: no amplifier answered. m_zeroed stays false and the first conversion
  // update() collects becomes the zero, so a cell plugged in after boot still
  // starts from a sensible reference instead of reading its whole standing
  // offset as sample mass.
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
  m_last_read_ms = millis();

  if (!m_has_reading)
  {
    // First conversion ever. Seed the filter with it rather than letting the
    // EMA crawl up from 0 over the next second, which would otherwise read as
    // a sample being poured in.
    m_filtered_counts = (float)m_raw;
    for (uint8_t i = 0; i < kStabilityBufferSize; ++i)
      m_recent[i] = m_filtered_counts;
    m_buffer_fill = kStabilityBufferSize;
    m_has_reading = true;
  }
  else
  {
    m_filtered_counts = kEmaAlpha * (float)m_raw + (1.0f - kEmaAlpha) * m_filtered_counts;
  }

  if (!m_zeroed)
  {
    m_zero_counts = m_filtered_counts;
    m_zeroed = true;
  }

  m_recent[m_buffer_idx] = m_filtered_counts;
  m_buffer_idx = (uint8_t)((m_buffer_idx + 1) % kStabilityBufferSize);
  if (m_buffer_fill < kStabilityBufferSize)
    ++m_buffer_fill;

  // Peak-to-peak over the window, in WEIGHT units rather than counts: the band
  // has to mean the same thing on all three cells, and their scale factors
  // differ by a factor of 1.5 and one of them is negative.
  float lo = m_recent[0];
  float hi = m_recent[0];
  for (uint8_t i = 1; i < m_buffer_fill; ++i)
  {
    if (m_recent[i] > hi)
      hi = m_recent[i];
    if (m_recent[i] < lo)
      lo = m_recent[i];
  }
  const float span = (m_scale_factor != 0.0f) ? fabsf((hi - lo) / m_scale_factor) : 0.0f;
  m_is_stable = (m_buffer_fill >= kStabilityBufferSize) && (span < kStableBand);

  return true;
}

void LoadCell::tare_empty()
{
  if (!m_usable || !m_has_reading)
    return;

  m_zero_counts = m_filtered_counts;
  m_zeroed = true;

  // The lid zero was measured as a difference from the OLD empty zero, so it
  // does not survive a new one. Upstream cleared it here for the same reason.
  m_lid_counts = 0.0f;
}

void LoadCell::tare_with_lid()
{
  if (!m_usable || !m_has_reading)
    return;

  m_lid_counts = m_filtered_counts - m_zero_counts;
}

float LoadCell::get_soil_weight() const
{
  // A zero scale factor would be a divide by zero and a NaN or an infinity on
  // the wire; report no reading instead, which is what it is.
  if (m_scale_factor == 0.0f)
    return NAN;

  return (m_filtered_counts - m_zero_counts - m_lid_counts) / m_scale_factor;
}

float LoadCell::reported(uint32_t now_ms) const
{
  if (!m_usable || !m_has_reading)
    return NAN;
  if (now_ms - m_last_read_ms > kStaleMs)
    return NAN;

  return get_soil_weight();
}
