"""devices.yaml claims to be the single source of truth. Where it isn't, say so.

The arm is the case that matters. Its address appears in two places:

  * ``devices.yaml`` -> ``arm.host``, which the launch files probe to decide
    real-arm-vs-mock, and which the hardware checker reports;
  * ``igus_rebel/include/igus_rebel/Rebel.hpp``, a compiled-in constant, which
    is what the driver actually connects to.

Nothing keeps them in step. Change the YAML alone and the probe tests one
address while the driver dials another: `arm_hardware_protocol:=auto` decides
the arm is present, the driver then cannot reach it, and you get an arm that
reports healthy and does not move. Change the header alone and auto-detect
decides the arm is absent and silently drops to mock.

Pinning them together here does not merge them -- the header is a vendor file
and the address is fixed in the control box -- but it makes the drift a failed
test instead of a confusing hour in the field.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from aries_common.devices import DEFAULTS  # noqa: E402

SRC = Path(__file__).resolve().parents[2]
REBEL_HPP = SRC / "aries_moveit" / "igus_rebel" / "include" / "igus_rebel" / "Rebel.hpp"
DEVICES_YAML = Path(__file__).resolve().parents[1] / "config" / "devices.yaml"


def device(path):
    """Read the SOURCE devices.yaml, not the installed copy.

    aries_common.devices resolves through the ament index, which needs the
    workspace sourced; unsourced it silently falls back to the DEFAULTS dict
    and every assertion here would pass against a value nobody edited. This
    test exists to catch an edit to the YAML, so it has to read the YAML.
    """
    section, _, key = path.partition(".")
    table = yaml.safe_load(DEVICES_YAML.read_text()) or {}
    assert section in table, f"no '{section}:' section in {DEVICES_YAML}"
    assert key in table[section], f"no '{path}' in {DEVICES_YAML}"
    return table[section][key]


def _compiled_endpoint():
    """(ip, port) as compiled into the arm driver."""
    text = REBEL_HPP.read_text()
    ip = re.search(r'const\s+std::string\s+ip\s*=\s*"([0-9.]+)"', text)
    port = re.search(r"const\s+int\s+port\s*=\s*(\d+)", text)
    assert ip and port, f"could not find the ip/port constants in {REBEL_HPP}"
    return ip.group(1), int(port.group(1))


def test_rebel_header_exists():
    assert REBEL_HPP.is_file(), (
        f"{REBEL_HPP} moved; this test is the only thing pinning the arm "
        f"address in devices.yaml to the one the driver compiles in"
    )


def test_arm_host_matches_the_driver():
    compiled_ip, _ = _compiled_endpoint()
    assert device("arm.host") == compiled_ip, (
        f"devices.yaml arm.host is {device('arm.host')} but the driver dials "
        f"{compiled_ip} (Rebel.hpp). The launch probe and the driver would "
        f"disagree: auto-detect can pick the real arm and then fail to reach it."
    )


def test_arm_port_matches_the_driver():
    _, compiled_port = _compiled_endpoint()
    assert int(device("arm.port")) == compiled_port, (
        f"devices.yaml arm.port is {device('arm.port')} but the driver dials "
        f"{compiled_port} (Rebel.hpp)."
    )


def test_arm_address_is_not_on_the_field_link():
    """The arm must stay reachable independently of the radio. If it shared the
    field-link subnet, losing the antenna would look like losing the arm."""
    import ipaddress

    from aries_common import comms

    arm = ipaddress.ip_address(device("arm.host"))
    prefix = comms.subnet_prefix()
    for name, addr in comms.hosts().items():
        net = ipaddress.ip_network(f"{addr}/{prefix}", strict=False)
        assert arm not in net, f"the arm at {arm} is inside {name}'s subnet {net}"


def test_builtin_defaults_match_the_yaml():
    """devices.py carries a fallback copy for a missing or unreadable file. A
    stale one turns a broken edit into a silent connection to the wrong box."""
    assert DEFAULTS["arm"]["host"] == device("arm.host")
    assert int(DEFAULTS["arm"]["port"]) == int(device("arm.port"))


def test_builtin_servo_bus_defaults_match_the_yaml():
    """Bench gear, but the same trap: st3215_test.py reads servo_bus.port to
    find the adapter, and an unreadable YAML drops it to this copy."""
    for key in ("port", "baud"):
        assert DEFAULTS["servo_bus"][key] == device(f"servo_bus.{key}"), (
            f"servo_bus.{key} differs between devices.py DEFAULTS and devices.yaml"
        )


def test_servo_bus_bridge_ids_match_the_udev_rule():
    """The chip IDs live in two places that cannot import each other.

    setup_system.sh generates the udev rule that makes /dev/aries_servo_bus;
    st3215_test.py falls back to scanning for the same chips when the symlink
    is not there (no rule installed yet, or a machine that is not the rover).
    Drift is silent in the worst direction: the scanner stops recognising an
    adapter the rule happily names, and the bench script says "no adapter"
    while /dev/aries_servo_bus is sitting right there.
    """
    setup_sh = (SRC.parent / "scripts" / "setup_system.sh").read_text()
    st3215 = (SRC.parent / "scripts" / "st3215_test.py").read_text()

    # The rule lines live inside a double-quoted bash string, so every
    # quote in the file is backslash-escaped. Drop the backslashes first
    # rather than trying to match through two layers of quoting.
    # Both branches of the if/else, so the chip-matched one is included:
    # split(maxsplit=1) or the second SERVO_BUS_RULE= assignment ends the slice.
    rule_block = setup_sh.split("SERVO_BUS_RULE=", 1)[1].split(
        'install_file "$SERVO_BUS_RULES"')[0]
    rule_vids = set(re.findall(r'idVendor}=="([0-9a-f]{4})"',
                               rule_block.replace("\\", "")))

    scanner = re.search(r"BRIDGE_VIDS\s*=\s*\(([^)]*)\)", st3215)
    assert scanner, "BRIDGE_VIDS not found in scripts/st3215_test.py"
    scanner_vids = set(re.findall(r'"([0-9a-f]{4})"', scanner.group(1)))

    assert rule_vids, "no idVendor matches found in the generated servo-bus rule"
    assert rule_vids == scanner_vids, (
        f"udev rule matches {sorted(rule_vids)} but st3215_test.py scans for "
        f"{sorted(scanner_vids)}"
    )


def test_builtin_network_defaults_match_the_yaml():
    yaml_net = yaml.safe_load(DEVICES_YAML.read_text())["network"]
    for key in ("domain_id", "hosts"):
        assert DEFAULTS["network"][key] == yaml_net[key], (
            f"network.{key} differs between devices.py DEFAULTS and devices.yaml"
        )
