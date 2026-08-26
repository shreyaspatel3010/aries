// Developed as a part of HSM Aries team,
// by Shivansh Mehta (https://github.com/Shivansh-Mehta),
// for the European Rover Challenge

#ifndef EMG_H
#define EMG_H

#include <Arduino.h>

// The mast's three-tier light.
//
// THE COLOUR CODES ARE THE HOST'S, NOT THIS FIRMWARE'S ORIGINAL ONES.
// aries_bringup/nodes/stacklight.py has published 1=red, 2=yellow, 3=green,
// 4=off since the light was first wired, and its COLOR_CODES table, its
// config, and test_stacklight.py all agree on that. This firmware shipped with
// 1=green, 2=yellow, 3=red -- red and green transposed.
//
// That is not a cosmetic disagreement. stacklight.py shows RED for an e-stop, a
// drive fault, a halt, and for `unknown` (the state it holds before it can see
// the rover at all). Under the firmware's original numbering every one of those
// would have lit GREEN -- a rover that has just lost its drive telling a
// bystander it is safe to approach. The transposition is corrected HERE, in the
// firmware, because the host contract is the older and more widely-depended-on
// of the two: see stacklight.py, stacklight.yaml, stacklight_gz_visual.py and
// the RViz marker overlay, all of which would otherwise have to move together.
enum StackLightColor : uint8_t
{
  STACKLIGHT_RED = 1,
  STACKLIGHT_YELLOW = 2,
  STACKLIGHT_GREEN = 3,
  STACKLIGHT_OFF = 4,
};

class StackLight
{
public:
  StackLight(uint8_t pin_green, uint8_t pin_yellow, uint8_t pin_red);
  void init_light();
  void state(uint8_t state);

private:
  // ACTIVE LOW -- the tier driver sinks current, so LOW lights a tier. Named
  // rather than inlined so the one place this would have to change, if the
  // light is ever rewired to source, is a single pair of constants.
  static constexpr uint8_t TIER_ON = LOW;
  static constexpr uint8_t TIER_OFF = HIGH;

  void all_off();

  uint8_t m_pin_green;
  uint8_t m_pin_yellow;
  uint8_t m_pin_red;
  uint8_t m_state;
};

#endif  // EMG_H
