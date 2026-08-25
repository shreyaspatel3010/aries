"""The ODrive axis tables the base station prints against.

aries_bringup's two rover-side checkers carry their own copies of these and
keep them: they run on the robot, where a shared module changing underneath a
checker is exactly the kind of surprise a checker must not have. This file is
the operator-side copy, so base_station_checker can name axis 3 the same wheel
that full_hardware_checker names it.

That is a deliberate duplicate, which means it can drift -- and the wheel map
is not a constant of the design. The CAN node ids stopped following the sides
in blocks when the rover was reassembled and had to be re-identified by arming
one axis at a time. A copy that missed such a change does not fail; it names
the wrong wheel in a report somebody is about to act on. So the drift is
pinned instead: aries_comms/test/test_axis_tables.py compares this file
against both rover checkers and fails if any of the three disagree.
"""

# ── ODrive axis_state values ─────────────────────────────────────────────────
# ODrive firmware 0.6.x AxisState. Gaps are unused by the firmware; a value not
# in here is printed raw by the checkers rather than guessed at.
AXIS_STATE_NAMES = {
    0: "UNDEFINED",
    1: "IDLE",
    2: "STARTUP_SEQ",
    3: "FULL_CALIBRATION",
    4: "MOTOR_CALIBRATION",
    6: "ENCODER_OFFSET_CAL",
    7: "ENCODER_INDEX_SEARCH",
    8: "CLOSED_LOOP",
    9: "LOCKIN_SPIN",
    10: "ENCODER_DIR_FIND",
    11: "HOMING",
    12: "ENCODER_HALL_PHASE_CAL",
    13: "ENCODER_HALL_POLARITY_CAL",
}

# The only state in which an axis will act on a velocity command. The drive
# bridge arms only when every axis reaches it, which is why one bad encoder
# stops all six wheels while every teleop topic still ticks.
CLOSED_LOOP = 8

# ── Wheel layout ─────────────────────────────────────────────────────────────
# CAN node id -> physical wheel, re-identified by arming one axis at a time on
# 2026-08-24. Keep in step with right_wheels/left_wheels in
# aries_drive/config/cmd_vel_odrive_bridge.yaml, which is the file that decides
# it -- this one only names what that file wired up.
# right_wheels = [5, 4, 3]   left_wheels = [0, 1, 2]   (both listed front -> rear)
AXIS_LABELS = {
    0: "Left-Front ",
    1: "Left-Mid   ",
    2: "Left-Rear  ",
    3: "Right-Rear ",
    4: "Right-Mid  ",
    5: "Right-Front",
}

NUM_AXES = 6

# The side each id is on, for the column header. Derived rather than written
# out, so it cannot disagree with AXIS_LABELS above.
LEFT_AXES = sorted(i for i, name in AXIS_LABELS.items() if name.startswith("Left"))
RIGHT_AXES = sorted(i for i, name in AXIS_LABELS.items() if name.startswith("Right"))


def _span(axes):
    if not axes:
        return "none"
    return f"{axes[0]}-{axes[-1]}" if axes == list(range(axes[0], axes[-1] + 1)) else \
        ",".join(str(i) for i in axes)


# "left: 0-2 | right: 3-5". The header used to be written by hand and had the
# two sides the wrong way round after the reassembly.
AXIS_HEADER = f"left: {_span(LEFT_AXES)} | right: {_span(RIGHT_AXES)}"


def axis_state_name(state):
    """A printable name for an axis_state, raw value if the firmware added one."""
    if state is None:
        return "---"
    return AXIS_STATE_NAMES.get(state, f"STATE_{state}")
