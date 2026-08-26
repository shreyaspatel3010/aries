"""The DDS transport config has to be right on a machine nobody tested it on.

These guard the two ways it silently degrades rather than failing: naming an
interface the machine does not hold (Cyclone warns and picks its own, giving an
empty topic list on a link that pings fine), and a peer list that omits the far
end (nothing is ever discovered, because multicast is off).
"""

import sys
import re
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aries_common import comms  # noqa: E402
from aries_common import devices  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ARIES_DOMAIN_ID", "ARIES_EXTRA_PEERS", "CYCLONEDDS_URI",
                 "ARIES_KEEP_CYCLONEDDS_URI", "ARIES_LOCAL_ADDRESS",
                 "ARIES_RMW", "ARIES_KEEP_FASTDDS_PROFILES",
                 "FASTRTPS_DEFAULT_PROFILES_FILE",
                 "FASTDDS_DEFAULT_PROFILES_FILE"):
        monkeypatch.delenv(name, raising=False)
    devices.load_devices(refresh=True)
    yield
    devices.load_devices(refresh=True)


def test_hosts_table_has_both_ends():
    table = comms.hosts()
    assert "rover" in table, "the rover must be in the hosts table or it is never a peer"
    assert "base" in table
    assert table["rover"] != table["base"], "both ends cannot share one address"


def test_domain_id_is_a_string():
    # Launch substitutions and os.environ both reject a bare int.
    assert isinstance(comms.domain_id(), str)
    assert comms.domain_id().isdigit()


def test_domain_id_env_override(monkeypatch):
    monkeypatch.setenv("ARIES_DOMAIN_ID", "42")
    assert comms.domain_id() == "42"


def test_peers_exclude_self_and_include_the_far_end():
    table = comms.hosts()
    peers = comms.peers(table["rover"])
    assert table["rover"] not in peers, "a machine must not announce itself to itself"
    assert table["base"] in peers, "the far end must be an explicit peer without multicast"


def test_peers_are_symmetric():
    """Both ends list both ends, so convergence does not depend on boot order."""
    table = comms.hosts()
    from_rover = comms.peers(table["rover"])
    from_base = comms.peers(table["base"])
    assert table["base"] in from_rover
    assert table["rover"] in from_base


def test_extra_peers_env(monkeypatch):
    monkeypatch.setenv("ARIES_EXTRA_PEERS", "10.0.0.9, 10.0.0.8")
    peers = comms.peers(comms.hosts()["rover"])
    assert "10.0.0.9" in peers and "10.0.0.8" in peers


def test_xml_names_the_local_address_and_disables_multicast():
    rover, base = comms.hosts()["rover"], comms.hosts()["base"]
    xml = comms.cyclone_xml(rover, [base, "127.0.0.1"])
    assert f'<NetworkInterface address="{rover}"/>' in xml
    # The airMAX link sends multicast at its lowest data rate.
    assert "<AllowMulticast>false</AllowMulticast>" in xml
    assert f'<Peer address="{base}"/>' in xml
    # The stack is ~30 nodes; Cyclone's default cap of 9 kills bringup partway.
    assert comms.MAX_AUTO_PARTICIPANT_INDEX >= 40


def test_write_config_uses_the_detected_address(tmp_path, monkeypatch):
    rover = comms.hosts()["rover"]
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: rover)
    path, local = comms.write_cyclone_config(tmp_path / "cyclone.xml")
    assert local == rover
    assert f'address="{local}"' in Path(path).read_text()


def test_missing_link_raises_rather_than_falling_back(monkeypatch):
    """A missing cable must fail here, not later as an unexplained empty graph."""
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        comms.write_cyclone_config()


def test_environment_is_complete(tmp_path, monkeypatch):
    """Whichever middleware is pinned, the env must carry its config pointer."""
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: comms.hosts()["base"])
    env = comms.dds_environment(tmp_path / "dds.xml")
    assert env["RMW_IMPLEMENTATION"] == comms.RMW
    assert env["ROS_DOMAIN_ID"] == comms.domain_id()
    if comms.RMW == "rmw_fastrtps_cpp":
        # PLAIN PATH, no file:// -- Fast DDS silently ignores a prefixed value,
        # which would leave every participant on its own defaults.
        assert env["FASTRTPS_DEFAULT_PROFILES_FILE"] == env["FASTDDS_DEFAULT_PROFILES_FILE"]
        assert not env["FASTDDS_DEFAULT_PROFILES_FILE"].startswith("file://")
    else:
        assert env["CYCLONEDDS_URI"].startswith("file://")


def test_fastdds_profile_pins_the_interface_and_lists_peers():
    """The Fast DDS profile must give the field link what the Cyclone XML did."""
    rover, base = comms.hosts()["rover"], comms.hosts()["base"]
    xml = comms.fastdds_xml(rover, [base, "127.0.0.1"])
    assert f"<address>{rover}</address>" in xml, "interface not pinned"
    assert "interfaceWhiteList" in xml
    for peer in (base, "127.0.0.1"):
        assert f"<udpv4><address>{peer}</address></udpv4>" in xml, peer
    # Shared memory bypasses the interface pin entirely and is invisible on the
    # wire, so the profile must declare UDPv4 and nothing else.
    assert "<useBuiltinTransports>false</useBuiltinTransports>" in xml
    assert "<type>UDPv4</type>" in xml


def test_inherited_uri_is_replaced_not_honoured(tmp_path, monkeypatch):
    """The shell's value is almost always a stale export naming an address this
    machine does not have. A launch that inherited it would be back to the
    domain being a property of the terminal."""
    monkeypatch.setattr(comms, "RMW", "rmw_cyclonedds_cpp")
    monkeypatch.setenv("CYCLONEDDS_URI", "file:///tmp/stale-hardcoded.xml")
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: comms.hosts()["rover"])
    env = comms.dds_environment(tmp_path / "cyclone.xml")
    assert env["CYCLONEDDS_URI"] != "file:///tmp/stale-hardcoded.xml"
    assert env["CYCLONEDDS_URI"].endswith("cyclone.xml")


def test_inherited_fastdds_profile_is_replaced_not_honoured(tmp_path, monkeypatch):
    """Same rule for the other vendor: the shell does not get to decide."""
    monkeypatch.setattr(comms, "RMW", "rmw_fastrtps_cpp")
    monkeypatch.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", "/tmp/stale-profiles.xml")
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: comms.hosts()["rover"])
    env = comms.dds_environment(tmp_path / "fast.xml")
    assert env["FASTRTPS_DEFAULT_PROFILES_FILE"] != "/tmp/stale-profiles.xml"
    assert env["FASTDDS_DEFAULT_PROFILES_FILE"].endswith("fast.xml")


def test_deliberate_override_can_be_kept(monkeypatch):
    monkeypatch.setattr(comms, "RMW", "rmw_cyclonedds_cpp")
    monkeypatch.setenv("CYCLONEDDS_URI", "file:///tmp/hand-written.xml")
    monkeypatch.setenv("ARIES_KEEP_CYCLONEDDS_URI", "1")
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: None)
    env = comms.dds_environment()
    assert env["CYCLONEDDS_URI"] == "file:///tmp/hand-written.xml"


def test_deliberate_fastdds_override_can_be_kept(monkeypatch):
    monkeypatch.setattr(comms, "RMW", "rmw_fastrtps_cpp")
    monkeypatch.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", "/tmp/hand-written.xml")
    monkeypatch.setenv("ARIES_KEEP_FASTDDS_PROFILES", "1")
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: None)
    env = comms.dds_environment()
    assert env["FASTDDS_DEFAULT_PROFILES_FILE"] == "/tmp/hand-written.xml"


def test_env_dict_carries_only_what_the_middleware_needs(tmp_path, monkeypatch):
    """Anything else here becomes a real exported variable on every node."""
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: comms.hosts()["rover"])

    monkeypatch.setattr(comms, "RMW", "rmw_cyclonedds_cpp")
    env = comms.dds_environment(tmp_path / "cyclone.xml")
    assert set(env) == {"ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI"}

    # Two profile variables, not one: Fast DDS renamed it at 2.12 and still
    # honours the old name. Which one a given build reads is not worth finding
    # out in the field.
    monkeypatch.setattr(comms, "RMW", "rmw_fastrtps_cpp")
    env = comms.dds_environment(tmp_path / "fast.xml")
    assert set(env) == {"ROS_DOMAIN_ID", "RMW_IMPLEMENTATION",
                        "FASTRTPS_DEFAULT_PROFILES_FILE",
                        "FASTDDS_DEFAULT_PROFILES_FILE"}


def test_env_script_is_installed():
    script = Path(__file__).resolve().parents[1] / "scripts" / "aries_dds_env.sh"
    assert script.is_file()
    cmake = (Path(__file__).resolve().parents[1] / "CMakeLists.txt").read_text()
    assert "aries_dds_env.sh" in cmake, "the script is useless if it is not installed"


def test_local_address_override_is_the_bench_escape_hatch(monkeypatch):
    monkeypatch.setenv("ARIES_LOCAL_ADDRESS", "10.55.7.136")
    assert comms.local_address() == "10.55.7.136"


def test_loopback_only_peers_with_itself():
    """A single-host bench run must not announce to unreachable field addresses.

    Cyclone logs one ddsi_udp_conn_write failure per unreachable port per
    announcement -- sixty-odd lines per peer per participant, which buries
    every real message.
    """
    assert comms.peers("127.0.0.1") == ["127.0.0.1"]


def test_loopback_is_always_a_peer():
    assert "127.0.0.1" in comms.peers(comms.hosts()["rover"])


def test_a_routed_address_is_not_the_field_link(monkeypatch):
    """Every machine can route to the rover via its default gateway.

    Accepting that would name the Wi-Fi interface with the antenna unplugged.
    """
    monkeypatch.delenv("ARIES_LOCAL_ADDRESS", raising=False)
    assert comms._direct_route_source("192.0.2.1") is None


# --------------------------------------------------------------------------
# The address plan. On a shared field these are what stop two teams' machines
# from answering to the same address, which is not a clean failure: ARP is
# last-writer-wins, so traffic silently follows whoever replied most recently.
# --------------------------------------------------------------------------


def test_field_link_and_arm_are_separate_subnets():
    """The igus control box has its own /24 and its address is compiled into
    the driver. If the field link ever overlapped it, the rover would have two
    routes to one subnet and the arm would come and go with the radio."""
    import ipaddress

    from aries_common.devices import device

    prefix = comms.subnet_prefix()
    arm_net = ipaddress.ip_network(f"{device('arm.host')}/24", strict=False)
    for name, addr in comms.hosts().items():
        link_net = ipaddress.ip_network(f"{addr}/{prefix}", strict=False)
        assert not link_net.overlaps(arm_net), (
            f"field-link host {name} ({addr}/{prefix}) overlaps the arm subnet "
            f"{arm_net}; the arm address is hardcoded in Rebel.hpp and cannot move"
        )
    for name, addr in comms.radios().items():
        assert ipaddress.ip_address(addr) not in arm_net, (
            f"radio {name} at {addr} is inside the arm subnet"
        )


def test_duplicate_address_check_is_available():
    """192.168.1.0/24 is the airOS factory subnet and the commonest consumer
    router subnet, so another team on the field may share it. That risk is
    accepted deliberately (see devices.yaml) and mitigated operationally: the
    setup script arpings for a second machine claiming our address. If that
    check goes away, the accepted risk quietly becomes an unmitigated one."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "setup_field_link.sh"
    assert script.is_file(), "setup_field_link.sh is missing"
    text = script.read_text()
    assert "arping -D" in text, "the duplicate-address check is gone"


def test_hosts_are_all_on_one_subnet():
    """A point-to-point link is one L2 segment; split subnets need a router."""
    import ipaddress

    prefix = comms.subnet_prefix()
    nets = {
        ipaddress.ip_network(f"{a}/{prefix}", strict=False)
        for a in comms.hosts().values()
    }
    assert len(nets) == 1, f"hosts span {len(nets)} subnets: {nets}"


def test_every_host_has_an_interface_pinned():
    """setup_field_link.sh needs to know which port to configure."""
    assert set(comms.interfaces()) >= set(comms.hosts()), (
        "every host needs network.interface.<name> or the setup script cannot run"
    )


def test_radios_do_not_collide_with_the_machines():
    machines = set(comms.hosts().values())
    for name, addr in comms.radios().items():
        assert addr not in machines, f"radio {name} shares {addr} with a machine"


def test_subnet_prefix_is_sane():
    assert 8 <= comms.subnet_prefix() <= 30


def test_host_name_for_identifies_the_machine():
    for name, addr in comms.hosts().items():
        assert comms.host_name_for(addr) == name
    assert comms.host_name_for("203.0.113.1") is None


def test_a_guessed_address_is_never_silent(monkeypatch, capsys):
    """Attempts 2 and 3 can land on a stranger's DHCP lease. Silently deciding
    which machine you are is how the rover announces itself as the base."""
    monkeypatch.delenv("ARIES_LOCAL_ADDRESS", raising=False)
    prefix = comms.subnet_prefix()
    neighbour = comms.hosts()["rover"].rsplit(".", 1)[0] + ".99"
    monkeypatch.setattr(
        comms.subprocess, "run",
        lambda *a, **k: type("R", (), {"stdout": f"1: eth0 inet {neighbour}/{prefix} scope global eth0\n"})(),
    )
    assert comms.local_address() == neighbour
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------
# A machine with no antenna. This is the developer laptop running simulation,
# and it is also every machine before setup_field_link.sh has been run. An
# interface pin it does not hold is FATAL, not a warning: Cyclone refuses to
# create the domain and every node in the launch dies at startup.
# --------------------------------------------------------------------------


def test_local_only_config_pins_only_loopback(monkeypatch, tmp_path):
    """Off the link, the only address safe to pin is the one every machine has.

    The original rule was "pin nothing here", because Cyclone treats an address
    the machine does not hold as fatal. 127.0.0.1 is not such an address, and
    pinning nothing turned out to have its own failure: Cyclone then picks a
    physical NIC once, at participant creation, and never re-selects — so a
    bench run started with an Ethernet cable in dies into

        tev: ddsi_udp_conn_write to udp/239.255.0.1:14900 failed with retcode -1

    from every participant, forever, the moment the cable comes out.
    """
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: None)
    path, local = comms.write_cyclone_config(tmp_path / "c.xml", require_link=False)
    assert local is None
    xml = Path(path).read_text()

    pins = re.findall(r'<NetworkInterface\s+address="([^"]+)"', xml)
    assert pins == ["127.0.0.1"], (
        f"off the field link the only pinnable interface is loopback, got {pins}"
    )


def test_local_only_config_never_uses_multicast(monkeypatch, tmp_path):
    """`lo` carries no MULTICAST flag, so a multicast write over it cannot
    succeed; discovery has to go to an explicit localhost peer instead."""
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: None)
    path, _ = comms.write_cyclone_config(tmp_path / "c.xml", require_link=False)
    xml = Path(path).read_text()
    assert "<AllowMulticast>false</AllowMulticast>" in xml
    assert '<Peer address="127.0.0.1"/>' in xml, (
        "multicast off with no peer leaves a single-machine run with no "
        "discovery at all"
    )


def test_local_only_config_still_matches_the_robot_domain(monkeypatch, tmp_path):
    """Same domain and middleware, so a sim here is directly comparable and a
    shell configured this way can still talk to a rover if one appears."""
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: None)
    env = comms.dds_environment(tmp_path / "c.xml", require_link=False)
    assert env["ROS_DOMAIN_ID"] == comms.domain_id()
    assert env["RMW_IMPLEMENTATION"] == comms.RMW


def test_field_launches_still_demand_the_link(monkeypatch):
    """The fallback must not weaken rover_field/base_station: a missing antenna
    there has to fail loudly, not come up local-only and see nothing."""
    monkeypatch.setattr(comms, "local_address", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        comms.write_cyclone_config(require_link=True)
    with pytest.raises(RuntimeError):
        comms.dds_environment()


def test_env_script_never_requires_the_link():
    """It runs from ~/.bashrc on machines that have no antenna and never will."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "aries_dds_env.sh"
    text = script.read_text()
    assert "require_link=False" in text, (
        "the shell environment must degrade to local-only; requiring the link "
        "here breaks simulation on every machine off the field"
    )
    assert "unset CYCLONEDDS_URI" in text, (
        "on failure it must clear the variable — an unset URI is a working "
        "default, a wrong one stops every node from starting"
    )
