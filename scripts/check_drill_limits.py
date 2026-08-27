#!/usr/bin/env python3
"""Find out what the drill's limit switches are actually doing.

The drill has no encoder on any axis, so these two switches are its entire
feedback path -- and from the host a switch that is unwired, on the wrong pin,
or simply open all look identical: `drill/limits` reads 0 and the carriage does
not stop. This separates those cases.

RUN IT, THEN PRESS EACH SWITCH BY HAND. You do not need to move the carriage.

    ros2 run aries_bringup check_drill_limits.py     # or: python3 this file

WHAT THE ANSWERS MEAN

  "bottom closed" / "top closed" appears
        The switch works and the firmware sees it on the pin pins.h expects.
        Nothing more to fix.

  Nothing on drill/limits, but a pin lights up in the scan
        The switch works and is wired to a DIFFERENT pin than pins.h believes.
        The line names the pin; put that number in LIMIT_SWITCH1/2 and reflash.

  Nothing anywhere, on either topic
        The closure is not reaching the Teensy at all. That is a harness
        problem, not a firmware one: an open circuit, or a switch that is not
        switching its pin to GND. Renumbering pins will not help. Check
        continuity from the pin to GND with the switch held closed.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, UInt64


class Watch(Node):
    def __init__(self):
        super().__init__("check_drill_limits")
        self.limits = None
        self.scan = None
        self.limits_seen = False
        self.scan_seen = False
        self.hits = set()
        self.baseline = None
        self.create_subscription(UInt8, "/drill/limits", self._limits, 10)
        self.create_subscription(UInt64, "/drill/pin_scan", self._scan, 10)
        self.create_timer(5.0, self._nag)
        print(__doc__.split("WHAT THE ANSWERS MEAN")[0].rstrip())
        print("\nwatching -- press each limit switch by hand now\n")

    def _limits(self, msg):
        if not self.limits_seen:
            self.limits_seen = True
            print("  drill/limits is live (board is running and connected)")
        bits = int(msg.data)
        if bits != self.limits:
            names = []
            if bits & 0x01:
                names.append("BOTTOM")
            if bits & 0x02:
                names.append("TOP")
            state = " + ".join(names) if names else "both open"
            # bit2/bit3: which PWM sign the firmware believes drives INTO each
            # switch. It seeds these from the convention and corrects them from
            # what the mechanism does, so a disagreement with the seed is the
            # gate telling you the convention was wrong -- not a fault.
            into_top = "+" if bits & 0x04 else "-"
            into_bottom = "+" if bits & 0x08 else "-"
            note = ""
            if (bits & 0x04) == 0 or (bits & 0x08) != 0:
                note = "   (corrected from the seeded +up/-down convention)"
            print(f"  drill/limits = {bits}  ->  {state}"
                  f"   [into top: {into_top}PWM, into bottom: {into_bottom}PWM]{note}")
            self.limits = bits

    def _scan(self, msg):
        bits = int(msg.data)
        if not self.scan_seen:
            self.scan_seen = True
            self.scan = bits
            self.baseline = bits
            resting = [p for p in range(64) if bits & (1 << p)]
            print(f"  drill/pin_scan is live -- resting LOW: "
                  f"{resting if resting else 'none'}")
            print("     (a pin LOW at rest is a NORMALLY-CLOSED switch, or a "
                  "pin tied low by something else)")
            return

        if bits == self.scan:
            return

        # BOTH directions. A normally-closed switch OPENS when pressed, so its
        # pin goes HIGH -- reporting only pins going LOW misses it entirely,
        # which is how the first version of this script drew a blank on a
        # perfectly good switch.
        changed = bits ^ self.scan
        for p in range(64):
            if not (changed & (1 << p)):
                continue
            now_low = bool(bits & (1 << p))
            self.hits.add(p)
            if p in (4, 5):
                tag = "  <- the pin pins.h expects"
            else:
                tag = (f"  <- pins.h does not know pin {p}: put it in "
                       f"LIMIT_SWITCH1/2")
            # NORMALLY-CLOSED only if the pin RESTED low. For a normally-open
            # switch the LOW->HIGH edge is simply the release, and calling that
            # "normally closed" sends you looking for an inversion that is not
            # there.
            rests_low = bool(self.baseline is not None
                             and self.baseline & (1 << p))
            if now_low:
                kind = "closed to GND" if not rests_low else "returned to rest"
                print(f"  pin {p}: HIGH -> LOW  ({kind}){tag}")
            else:
                kind = ("opened -- NORMALLY CLOSED, sense must be inverted"
                        if rests_low else "released")
                print(f"  pin {p}: LOW -> HIGH  ({kind}){tag}")
        self.scan = bits

    def _nag(self):
        if not self.limits_seen and not self.scan_seen:
            print("  ...nothing from the board yet. Is the stack running, and "
                  "has this firmware been flashed?")
        elif not self.hits:
            print("  ...no pin has CHANGED yet, in either direction. Press and "
                  "HOLD a switch.")


def main():
    rclpy.init()
    node = Watch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nsummary")
        if node.hits:
            print(f"  pins that CHANGED while you pressed: {sorted(node.hits)}")
            stray = sorted(p for p in node.hits if p not in (4, 5))
            if stray:
                print(f"  the switches are on {stray}, NOT on 4/5.")
                nc = [p for p in stray
                      if node.baseline is not None and node.baseline & (1 << p)]
                if nc:
                    print(f"  pins {nc} rested LOW and went HIGH when pressed: "
                          f"NORMALLY-CLOSED. is_at_stop() tests for LOW, so the "
                          f"sense must be inverted for them as well as the pin "
                          f"number changed.")
            else:
                print("  the switches are on 4/5 as configured.")
        else:
            print("  NO pin changed in either direction. The switch closure is "
                  "not reaching the Teensy -- harness, not firmware.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
