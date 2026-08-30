"""The array order in science.yaml must match the firmware's enum.

WHY THIS TEST EXISTS. /science/telemetry is ten anonymous floats. Nothing on
the wire says which is which, so the ONLY thing tying index 3 to ORP is that
firmware/teensy_science_sys/src/main.cpp and aries_science/config/science.yaml
were written in the same order. Nothing enforces that at build time, at launch
time, or at runtime -- a swap produces two plausible numbers under two wrong
names and no error anywhere.

So the test reads the actual firmware source and compares. It is deliberately
NOT a copy of the enum: a hardcoded expected list here would pass happily while
both this file and the YAML disagreed with the board.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
FIRMWARE_MAIN = REPO / "firmware" / "teensy_science_sys" / "src" / "main.cpp"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "science.yaml"

# IDX_<NAME> = <n>, taken from the TelemetryIndex enum. TELEMETRY_SIZE is the
# count and is excluded -- it is the enum's length, not a field.
ENUM_ENTRY = re.compile(r"^\s*IDX_([A-Z0-9_]+)\s*=\s*(\d+)\s*,", re.M)

# The firmware's enum names and the YAML's field names are deliberately not
# identical -- the firmware says what the SENSOR is, the topic says what the
# VALUE is -- so the mapping is stated here, once, and both sides are checked
# against it.
ENUM_TO_FIELD = {
    "PH": "ph",
    "SOIL_MOISTURE": "soil_moisture",
    "TDS": "tds",
    "ORP": "orp",
    "SOIL_TEMP": "soil_temp",
    "BME_TEMP": "air_temp",
    "BME_HUM": "humidity",
    "BME_PRESS": "pressure",
    "BME_GAS": "gas_resistance",
    "SCD_CO2": "co2",
}


@pytest.fixture(scope="module")
def firmware_order():
    assert FIRMWARE_MAIN.is_file(), f"firmware source not found at {FIRMWARE_MAIN}"
    text = FIRMWARE_MAIN.read_text()

    entries = [(int(n), name) for name, n in ENUM_ENTRY.findall(text)]
    assert entries, "no IDX_* entries found - has TelemetryIndex been renamed?"

    entries.sort()
    indices = [i for i, _ in entries]
    assert indices == list(range(len(indices))), (
        f"TelemetryIndex is not contiguous from 0: {indices}")
    return [name for _, name in entries]


@pytest.fixture(scope="module")
def config_fields():
    assert CONFIG.is_file(), f"config not found at {CONFIG}"
    params = yaml.safe_load(CONFIG.read_text())["science_telemetry"]["ros__parameters"]
    return params


def test_firmware_and_config_have_the_same_number_of_fields(firmware_order, config_fields):
    assert len(config_fields["fields"]) == len(firmware_order), (
        f"the firmware sends {len(firmware_order)} values "
        f"({', '.join(firmware_order)}) but science.yaml names "
        f"{len(config_fields['fields'])}. The node refuses to republish when "
        f"these disagree, so this is a total outage, not a partial one.")


def test_every_index_names_the_same_sensor(firmware_order, config_fields):
    """The check that actually matters: index-by-index, same sensor."""
    for index, enum_name in enumerate(firmware_order):
        expected = ENUM_TO_FIELD.get(enum_name)
        assert expected is not None, (
            f"firmware index {index} is IDX_{enum_name}, which this test does "
            f"not know about. A field was added to the firmware without being "
            f"added to science.yaml and to ENUM_TO_FIELD here.")
        actual = config_fields["fields"][index]
        assert actual == expected, (
            f"index {index}: the firmware sends IDX_{enum_name} but "
            f"science.yaml calls it '{actual}'. Every value from here on is "
            f"published under the wrong name.")


def test_every_field_has_a_unit_and_a_command(config_fields):
    for name in config_fields["fields"]:
        entry = config_fields["field"].get(name)
        assert entry is not None, f"field '{name}' has no field.{name} block"
        assert "unit" in entry, f"field '{name}' has no unit"
        assert "cmd" in entry, f"field '{name}' has no cmd"


def test_the_bme688_fields_share_one_command(config_fields):
    """One BME688 read fills four indices, so only the first carries a command.

    This is the assumption _owner_of() walks backwards on. If a later firmware
    gives humidity its own command, this test is where that shows up.
    """
    fields = config_fields["fields"]
    cmds = {n: config_fields["field"][n]["cmd"] for n in fields}

    assert cmds["air_temp"] >= 0, "air_temp is the BME688's commanded field"
    for name in ("humidity", "pressure", "gas_resistance"):
        assert cmds[name] == -1, (
            f"'{name}' carries cmd {cmds[name]}, but it is filled as a side "
            f"effect of reading air_temp and has no command of its own.")

    # ...and each of them must follow air_temp, because the node resolves an
    # owner by walking BACKWARDS to the nearest commanded field.
    owner_index = fields.index("air_temp")
    for name in ("humidity", "pressure", "gas_resistance"):
        assert fields.index(name) > owner_index, (
            f"'{name}' sits before air_temp, so the node would resolve its "
            f"owner to an earlier sensor entirely.")


def test_commands_match_their_index(firmware_order, config_fields):
    """`cmd` is the sensor id, and the firmware derives it as data / 10.

    The board's sensor_cmd_callback does `sensor_id = msg->data / 10` and then
    switches on it against the SAME enum, so a field's command id has to equal
    its array index. They are two spellings of one number.
    """
    for index, name in enumerate(config_fields["fields"]):
        cmd = config_fields["field"][name]["cmd"]
        if cmd < 0:
            continue
        assert cmd == index, (
            f"'{name}' is at index {index} but carries cmd {cmd}. The firmware "
            f"switches on data/10 against the telemetry index, so sending "
            f"{cmd * 10 + 2} would read a different sensor than the one this "
            f"name is published from.")
