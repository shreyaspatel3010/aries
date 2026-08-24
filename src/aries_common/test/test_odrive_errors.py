"""Pins the ODriveError decode table used by the hardware checkers."""

import re
from pathlib import Path

import pytest

from aries_common.odrive_errors import (
    ODRIVE_ERROR_NAMES,
    decode_odrive_error,
    format_odrive_error,
)

# The vendored driver ships the same enum as C++. Walk up to the workspace src/
# so this resolves whether pytest runs from the package or the workspace root.
VENDOR_ENUMS = "vendor/ros_odrive/odrive_base/include/odrive_enums.h"


def _vendor_error_bits():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / VENDOR_ENUMS
        if candidate.is_file():
            break
    else:
        pytest.skip(f"vendored {VENDOR_ENUMS} not in this checkout")

    block = re.search(
        r"enum ODriveError \{(.*?)\};", candidate.read_text(), re.S
    )
    assert block, "ODriveError enum not found in the vendored header"
    bits = {
        int(value, 16): name.removeprefix("ODRIVE_ERROR_")
        for name, value in re.findall(
            r"(ODRIVE_ERROR_\w+)\s*=\s*(0x[0-9A-Fa-f]+)", block.group(1)
        )
    }
    bits.pop(0, None)  # NONE is the absence of bits, not a bit
    return bits


def test_table_matches_the_vendored_driver_enum():
    """The table is a Python copy of the driver's C++ enum; keep them equal.

    Bumping src/vendor/ros_odrive must not leave the checkers naming bits from
    an older firmware, which would mislabel a fault rather than fail loudly.
    """
    assert ODRIVE_ERROR_NAMES == _vendor_error_bits()


def test_reported_axis_fault_decodes_to_missing_estimate():
    """0x08 is MISSING_ESTIMATE under firmware 0.6.x.

    Firmware 0.5.x had a separate ControllerError enum in which the same value
    meant INVALID_MIRROR_AXIS. Decoding a 0.6.x heartbeat against the 0.5.x
    table is silent and sends you after the wrong fault, so pin the one the
    vendor node actually delivers.
    """
    assert decode_odrive_error(0x00000008) == "MISSING_ESTIMATE"


def test_zero_is_none_but_formats_bare():
    assert decode_odrive_error(0) == "NONE"
    # Status lines print both halves whenever either is set; the clean half
    # stays bare hex rather than padding the line with "NONE".
    assert format_odrive_error(0) == "0x00000000"
    assert format_odrive_error(None) == "0x00000000"


def test_format_leads_with_hex():
    assert format_odrive_error(0x00000008) == "0x00000008 MISSING_ESTIMATE"


def test_multiple_bits_are_listed_low_to_high():
    assert (
        decode_odrive_error(0x00000208)
        == "MISSING_ESTIMATE|DC_BUS_UNDER_VOLTAGE"
    )


def test_every_known_bit_round_trips_to_one_name():
    for bit, name in ODRIVE_ERROR_NAMES.items():
        assert decode_odrive_error(bit) == name


def test_all_bits_set_names_every_entry_exactly_once():
    every = 0
    for bit in ODRIVE_ERROR_NAMES:
        every |= bit
    names = decode_odrive_error(every).split("|")
    assert len(names) == len(ODRIVE_ERROR_NAMES)
    assert sorted(names) == sorted(ODRIVE_ERROR_NAMES.values())


def test_table_holds_only_single_bits():
    for bit in ODRIVE_ERROR_NAMES:
        assert bit and not (bit & (bit - 1)), f"0x{bit:08X} is not one bit"


def test_undefined_bits_survive_as_unknown():
    """A firmware newer than this table must not decode to silence."""
    assert decode_odrive_error(0x00020000) == "UNKNOWN_BIT_17"
    assert decode_odrive_error(0x00020008) == "MISSING_ESTIMATE|UNKNOWN_BIT_17"
