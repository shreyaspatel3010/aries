// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge

#include "emg.h"

#include "pins.h"

StackLight::StackLight(uint8_t pin_green, uint8_t pin_yellow, uint8_t pin_red)
    : m_pin_green(pin_green),
      m_pin_yellow(pin_yellow),
      m_pin_red(pin_red),
      m_state(STACKLIGHT_OFF) {}

void StackLight::all_off()
{
  if (PIN_IS_ASSIGNED(m_pin_green))
    digitalWrite(m_pin_green, TIER_OFF);
  if (PIN_IS_ASSIGNED(m_pin_yellow))
    digitalWrite(m_pin_yellow, TIER_OFF);
  if (PIN_IS_ASSIGNED(m_pin_red))
    digitalWrite(m_pin_red, TIER_OFF);
}

void StackLight::init_light()
{
  // Drive the pins OFF before making them outputs. A Teensy pin comes out of
  // reset as a high-impedance input, and pinMode(OUTPUT) latches whatever was
  // last written to the output register -- which is 0, i.e. LOW, i.e. ON for
  // an active-low tier. Setting the level first means the light never flickers
  // through a colour on boot.
  if (PIN_IS_ASSIGNED(m_pin_green))
  {
    digitalWrite(m_pin_green, TIER_OFF);
    pinMode(m_pin_green, OUTPUT);
  }
  if (PIN_IS_ASSIGNED(m_pin_yellow))
  {
    digitalWrite(m_pin_yellow, TIER_OFF);
    pinMode(m_pin_yellow, OUTPUT);
  }
  if (PIN_IS_ASSIGNED(m_pin_red))
  {
    digitalWrite(m_pin_red, TIER_OFF);
    pinMode(m_pin_red, OUTPUT);
  }
  state(STACKLIGHT_OFF);
}

void StackLight::state(uint8_t state)
{
  m_state = state;
  all_off();

  switch (state)
  {
  case STACKLIGHT_RED:
    if (PIN_IS_ASSIGNED(m_pin_red))
      digitalWrite(m_pin_red, TIER_ON);
    break;
  case STACKLIGHT_YELLOW:
    if (PIN_IS_ASSIGNED(m_pin_yellow))
      digitalWrite(m_pin_yellow, TIER_ON);
    break;
  case STACKLIGHT_GREEN:
    if (PIN_IS_ASSIGNED(m_pin_green))
      digitalWrite(m_pin_green, TIER_ON);
    break;
  case STACKLIGHT_OFF:
    // Already dark.
    break;
  default:
    // An unrecognised code lights RED, and this is the one default worth being
    // deliberate about: the alternative -- ignoring it, or going dark -- means
    // a host that starts sending a code this firmware does not know leaves the
    // light showing whatever it showed before, indefinitely. Red is the state
    // that asks somebody to come and look.
    if (PIN_IS_ASSIGNED(m_pin_red))
      digitalWrite(m_pin_red, TIER_ON);
    m_state = STACKLIGHT_RED;
    break;
  }
}
