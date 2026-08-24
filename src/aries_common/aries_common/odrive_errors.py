"""Human-readable names for the ODrive error bitfields carried over CAN.

Every field the vendor node exposes as a uint32 of errors decodes against this
one enum, ``ODriveError`` from ODrive firmware 0.6.x:

  * ``ControllerStatus.active_errors`` -- heartbeat (CAN cmd 0x001) bytes 0..3,
    which the ODrive CAN protocol calls ``Axis_Error``.
  * ``ODriveStatus.active_errors``     -- Get_Error (0x003) bytes 0..3.
  * ``ODriveStatus.disarm_reason``     -- Get_Error (0x003) bytes 4..7, the
    latched copy of whatever dropped the axis out of CLOSED_LOOP.

The name ``ControllerStatus`` is misleading and worth calling out, because the
mistake it invites is silent: firmware 0.5.x had a *separate* ``ControllerError``
enum in which 0x08 meant INVALID_MIRROR_AXIS. Under 0.6.x the same value is
MISSING_ESTIMATE -- a different fault with a different fix. Decode with this
table, not with a 0.5.x reference.
"""

# ODriveError, firmware 0.6.x. Gaps in the bit numbering are unused by the
# firmware and are reported by decode_odrive_error as raw hex.
ODRIVE_ERROR_NAMES = {
    0x00000001: "INITIALIZING",
    0x00000002: "SYSTEM_LEVEL",
    0x00000004: "TIMING_ERROR",
    # No valid position/velocity estimate, so the axis refuses CLOSED_LOOP and
    # falls back to IDLE. On this rover that is an encoder/hall cable before it
    # is anything else -- see the Right-Rear axis 3 investigation.
    0x00000008: "MISSING_ESTIMATE",
    0x00000010: "BAD_CONFIG",
    0x00000020: "DRV_FAULT",
    0x00000040: "MISSING_INPUT",
    0x00000100: "DC_BUS_OVER_VOLTAGE",
    0x00000200: "DC_BUS_UNDER_VOLTAGE",
    0x00000400: "DC_BUS_OVER_CURRENT",
    0x00000800: "DC_BUS_OVER_REGEN_CURRENT",
    0x00001000: "CURRENT_LIMIT_VIOLATION",
    0x00002000: "MOTOR_OVER_TEMP",
    0x00004000: "INVERTER_OVER_TEMP",
    0x00008000: "VELOCITY_LIMIT_VIOLATION",
    0x00010000: "POSITION_LIMIT_VIOLATION",
    0x01000000: "WATCHDOG_TIMER_EXPIRED",
    0x02000000: "ESTOP_REQUESTED",
    0x04000000: "SPINOUT_DETECTED",
    0x08000000: "BRAKE_RESISTOR_DISARMED",
    0x10000000: "THERMISTOR_DISCONNECTED",
    0x40000000: "CALIBRATION_ERROR",
}


def decode_odrive_error(value):
    """Return the set bits of ``value`` as ``NAME|NAME``, or "NONE" for zero.

    Bits the firmware does not define still show up, as ``UNKNOWN_BIT_17``, so
    a newer firmware revision reporting a fault this table has not caught up
    with is visible rather than silently dropped.
    """
    bits = int(value or 0)
    if bits == 0:
        return "NONE"

    names = []
    for bit in sorted(ODRIVE_ERROR_NAMES):
        if bits & bit:
            names.append(ODRIVE_ERROR_NAMES[bit])
            bits &= ~bit

    # Whatever is left is not in the table; report it a bit at a time so the
    # position is readable without counting hex digits.
    while bits:
        lowest = bits & -bits
        names.append(f"UNKNOWN_BIT_{lowest.bit_length() - 1}")
        bits &= ~lowest

    return "|".join(names)


def format_odrive_error(value):
    """Return ``0x00000008 MISSING_ESTIMATE`` -- hex first, then the names.

    The hex stays because it is what the ODrive documentation, the odrivetool
    output and every existing log line in this repo are keyed to; the names are
    what save a lookup.

    A clean field formats as bare ``0x00000000``. Status lines print both the
    controller and the drive error whenever either one is set, so spelling out
    "NONE" on the clean half would pad the line without adding anything.
    """
    bits = int(value or 0)
    if bits == 0:
        return "0x00000000"
    return f"0x{bits:08X} {decode_odrive_error(bits)}"
