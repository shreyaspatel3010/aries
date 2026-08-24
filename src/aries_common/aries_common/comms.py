"""One DDS transport configuration, correct on every machine on the field link.

The rover and the base station run the same workspace, so this file is the same
file on both. The one setting that must differ between them --
``<NetworkInterface address>`` -- is *detected* here rather than written down,
because a hand-mirrored copy is the failure this module exists to prevent:

    Cyclone treats an address the machine does not hold, and a config file it
    cannot open, as FATAL. It refuses to create the domain, so every node in
    the launch dies at startup with

        can't open configuration file file:///.../cyclonedds.xml
        rmw_create_node: failed to create domain, error Error

    Measured on Cyclone under Jazzy, 2026-08-21. Older notes in this repo
    claimed it merely warns and falls back to defaults -- it does not, and a
    copied-and-not-edited config takes the whole stack down rather than
    degrading quietly.

That cuts both ways, and is why ``require_link`` exists below. A machine with
no antenna -- a developer laptop running simulation -- must NOT be pinned to a
field-link address, or the same fatality applies to it for no reason. It is
pinned to 127.0.0.1 instead, which every machine holds: see ``local_only_xml``
for why leaving the choice to Cyclone is worse than either.

Everything else -- domain, peers, multicast -- is identical on both ends and is
read from the ``network:`` section of ``aries_common/config/devices.yaml``.

USE FROM A LAUNCH FILE (the important one)

    from aries_common.comms import dds_launch_actions

    return LaunchDescription([
        *dds_launch_actions(),      # MUST come before any node action
        ...
    ])

Launch executes actions in order, so a Node placed above these would inherit
the calling shell's environment instead. That is not cosmetic: a stack launched
from a terminal opened before the exports existed lands on domain 0 with
rmw_fastrtps_cpp and cannot see the other machine at all.

USE FROM A SHELL

    source "$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh"

Needed by anything started by hand -- rviz2, rqt_image_view, ros2 topic list.
A launch file cannot reach into your shell, and a process keeps the DDS
environment it STARTED with for as long as it lives.

WHY UNICAST

    AllowMulticast is false and discovery goes to explicit peers, because the
    airMAX link sends multicast at its lowest data rate. The cost is that a
    machine which is not in the ``hosts`` table will never be discovered: add
    it to devices.yaml, or set ARIES_EXTRA_PEERS=<ip>,<ip> for a one-off.
"""

import ipaddress
import os
import socket
import subprocess
import tempfile

from aries_common.devices import device

# Cyclone needs a discovery port per participant when multicast is off, picked
# from a bounded range of participant indices. The default cap is 9 -- ten
# participants per machine -- while the rover stack is roughly thirty nodes.
# Past the cap, bringup dies partway through with "Failed to find a free
# participant index for domain 30" and a scatter of unrelated-looking
# rmw_create_node errors.
#
# Raising it costs discovery traffic, and not trivially: a participant unicasts
# SPDP to every index from 0 to this number, for every peer. 60 covers the whole
# stack with room to spare; lower it if `iftop` shows discovery crowding the
# link, but never below the node count.
MAX_AUTO_PARTICIPANT_INDEX = int(os.environ.get("ARIES_MAX_PARTICIPANT_INDEX", "60"))

RMW = "rmw_cyclonedds_cpp"

_CONFIG_BASENAME = "aries_cyclonedds.xml"


def domain_id():
    """The ROS domain both machines must agree on, as a string."""
    return str(os.environ.get("ARIES_DOMAIN_ID") or device("network.domain_id"))


def hosts():
    """Every machine on the field link: {name: address}."""
    table = device("network.hosts", {}) or {}
    return {str(name): str(addr) for name, addr in table.items()}


def subnet_prefix():
    """CIDR prefix length of the field link."""
    return int(device("network.subnet_prefix", 24))


def interfaces():
    """Wired port the antenna is on, per machine: {name: ifname}."""
    table = device("network.interface", {}) or {}
    return {str(name): str(iface) for name, iface in table.items()}


def radios():
    """airOS management addresses: {name: address}."""
    table = device("network.radios", {}) or {}
    return {str(name): str(addr) for name, addr in table.items()}


def host_name_for(address):
    """Which machine an address belongs to ('rover', 'base'), or None."""
    for name, addr in hosts().items():
        if addr == address:
            return name
    return None


def local_address(prefix=None):
    """This machine's address on the field link, or None if it is not on it.

    ``ARIES_LOCAL_ADDRESS`` short-circuits all of it. That is the bench escape
    hatch: on a developer machine that is not on the field link at all, set it
    to the LAN address of this machine (or 127.0.0.1 for a single-host run)
    rather than weakening the detection below, which has to stay strict.

    Otherwise, three attempts, narrowing from certain to merely likely:

      1. An interface holding one of the addresses in the hosts table. With
         static addressing -- which is the supported configuration, see
         FIELD_SETUP.md -- this always wins and identifies the machine exactly.
      2. An interface on the same subnet as one of them.
      3. The source address of a DIRECT route to one of them.

    Attempts 2 and 3 are fallbacks for a machine that did not get its static
    address, and both GUESS: on a shared field, "some interface on this subnet"
    can be a stranger's DHCP lease. They warn when used, because a silent guess
    about which machine you are is how the rover ends up announcing itself as
    the base station.

    Attempt 3 rejects anything reached "via" a gateway on purpose. Every
    machine can route to the rover's address through its default gateway, so
    accepting that would name the Wi-Fi interface whenever the antenna cable
    is out -- and DDS would come up on some other network, half-working, which
    is precisely the silent degradation this module exists to prevent. No
    interface on the field link has to stay an error.
    """
    override = os.environ.get("ARIES_LOCAL_ADDRESS", "").strip()
    if override:
        return override

    known = set(hosts().values())
    if not known:
        return None
    if prefix is None:
        prefix = subnet_prefix()

    nets = []
    for addr in known:
        try:
            nets.append(ipaddress.ip_network(f"{addr}/{prefix}", strict=False))
        except ValueError:
            continue

    on_subnet = []
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[2] != "inet":
                continue
            addr = parts[3].split("/")[0]
            if addr in known:
                return addr
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if any(ip in net for net in nets):
                on_subnet.append(addr)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    if on_subnet:
        _warn_guessed(on_subnet[0], "it is the only interface on the link subnet")
        return on_subnet[0]

    for target in sorted(known):
        addr = _direct_route_source(target)
        if addr:
            _warn_guessed(addr, f"it is the source of a direct route to {target}")
            return addr
    return None


def _warn_guessed(address, why):
    """Say so when the machine's identity was inferred rather than matched.

    Only reachable when this machine does not hold one of the addresses in the
    hosts table, i.e. it is not statically configured. On a field shared with
    other teams the inference can land on someone else's DHCP lease, so it must
    never be silent.
    """
    print(
        f"[aries_common] WARNING: using {address} as this machine's field-link "
        f"address because {why}. It is not one of the configured hosts "
        f"({', '.join(sorted(hosts().values()))}), so this is a GUESS -- run "
        f"scripts/setup_field_link.sh to set the static address."
    )


def _direct_route_source(target):
    """Our source address for a directly-attached route to ``target``, or None.

    ``ip route get`` prints "via <gateway>" when the destination is reached
    through a router. On the field link the two radios are one L2 segment, so
    the real answer never has one.
    """
    try:
        out = subprocess.run(
            ["ip", "-4", "route", "get", target],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    first = out.splitlines()[0] if out.splitlines() else ""
    fields = first.split()
    if "via" in fields or "src" not in fields:
        return None
    addr = fields[fields.index("src") + 1]
    return None if addr.startswith("127.") else addr


def peers(local=None):
    """Everyone this machine should announce itself to, minus itself.

    Both ends list both ends. A one-directional peer list happens to work --
    the far side learns our locators from the SPDP we send it and answers --
    but it only converges once the machine that does know reaches the one that
    does not, which turns "who booted first" into a variable.
    """
    if local is None:
        local = local_address()

    # Bound to loopback, nothing off this machine is reachable, and Cyclone
    # says so once per unreachable port per announcement -- sixty-odd lines per
    # peer per participant, which buries every real message in the log. A
    # single-host bench run peers with itself and nothing else.
    if str(local).startswith("127."):
        return ["127.0.0.1"]

    found = list(hosts().values()) + ["127.0.0.1"]
    extra = os.environ.get("ARIES_EXTRA_PEERS", "")
    found += [p.strip() for p in extra.split(",") if p.strip()]
    # Loopback stays in the list even when it IS the local address: it is how
    # the thirty-odd participants on one machine find each other, and dropping
    # it would leave a single-host bench run with no discovery at all.
    return [p for p in dict.fromkeys(found) if p != local or p == "127.0.0.1"]


def cyclone_xml(local, peer_list, domain=None):
    """The transport config text for one machine."""
    peer_xml = "\n".join(f'        <Peer address="{p}"/>' for p in peer_list)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- GENERATED by aries_common.comms on {socket.gethostname()}. Do not edit:
     it is rewritten every time a launch file or aries_dds_env.sh runs. The
     interface address below is detected, which is what lets the rover and the
     base station share one workspace without mirroring a file by hand. -->
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="{local}"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
{peer_xml}
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>{MAX_AUTO_PARTICIPANT_INDEX}</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
"""


def local_only_xml():
    """A config for a machine that is not on the field link at all.

    Loopback, unicast, no multicast -- deliberately identical to what
    ``ARIES_LOCAL_ADDRESS=127.0.0.1`` produces through the pinned path, because
    that is exactly the situation: one machine, talking to itself.

    IT USED TO LEAVE THE INTERFACE TO CYCLONE, AND THAT IS A TRAP.

        With no <Interfaces> pin Cyclone enumerates the machine's interfaces
        and picks one -- preferring a wired NIC -- **once, at participant
        creation**. It never re-selects. So a bench run started with an
        Ethernet cable plugged in binds every one of its ~30 participants to
        that NIC, and the moment the cable comes out (measured 2026-08-23:
        carrier lost 13:22:11, first error 13:23:05) every SPDP announcement
        fails forever:

            tev: ddsi_udp_conn_write to udp/239.255.0.1:14900 failed with
            retcode -1

        one line per participant per resend, drowning the launch. Nothing
        recovers it but a restart, and nothing about the message says
        "your cable came out".

        A simulation on one machine has no reason to be on a physical NIC at
        all, so it no longer is. Loopback cannot lose carrier.

    Multicast is off for the same reason it is off on the field link, plus one
    more here: ``lo`` does not carry the MULTICAST flag, so a multicast write
    over it could not succeed anyway. Discovery instead goes to an explicit
    localhost peer, which is what the thirty-odd participants of one launch use
    to find each other.

    TWO MACHINES ON A SWITCH is a different case and already has an answer:
    ``ARIES_LOCAL_ADDRESS=<this machine's LAN address>``. That takes the pinned
    path above -- interface pinned, unicast peers from the hosts table -- and
    never reaches this function. Do not restore multicast here to serve it.
    """
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- GENERATED by aries_common.comms on {socket.gethostname()}: this machine is
     NOT on the field link, so this is a loopback-only configuration. Nothing
     leaves the machine and no physical interface is touched, so an unplugged
     cable or a roaming Wi-Fi association cannot take the run down. Domain and
     middleware still match the robot. For two machines on a switch, set
     ARIES_LOCAL_ADDRESS to this machine's LAN address instead. -->
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="127.0.0.1"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <Peer address="127.0.0.1"/>
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>{MAX_AUTO_PARTICIPANT_INDEX}</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
"""


def write_cyclone_config(path=None, require_link=True):
    """Write the config for this machine and return (path, local_address).

    ``require_link=True`` (the field launches) raises when this machine is not
    on the link, so a missing cable fails loudly here instead of silently later
    as an empty topic list.

    ``require_link=False`` (the shell environment, simulation) falls back to a
    local-only config and returns ``(path, None)``. A developer laptop has no
    antenna and does not need one; making it an error there would mean the
    whole simulation stack refuses to start on any machine off the field.
    """
    local = local_address()
    if local is None and not require_link:
        if path is None:
            path = os.path.join(tempfile.gettempdir(), _CONFIG_BASENAME)
        with open(path, "w") as handle:
            handle.write(local_only_xml())
        return path, None
    if local is None:
        raise RuntimeError(
            "No interface on the field link "
            f"({', '.join(sorted(hosts().values())) or 'no hosts configured'}).\n"
            "  Check the antenna cable and `ip -4 -br addr`, or edit the network "
            "section of aries_common/config/devices.yaml if the addresses moved.\n"
            "  For a bench run with no link at all, set "
            "ARIES_LOCAL_ADDRESS=127.0.0.1 (one machine) or to this machine's "
            "LAN address (two machines on a switch)."
        )
    if path is None:
        path = os.path.join(tempfile.gettempdir(), _CONFIG_BASENAME)
    with open(path, "w") as handle:
        handle.write(cyclone_xml(local, peers(local)))
    return path, local


def dds_environment(path=None, require_link=True):
    """The three variables every ARIES process must agree on.

    A CYCLONEDDS_URI already in the environment is REPLACED, not honoured. That
    looks aggressive and is the entire point of this module: the value in your
    shell is almost always a stale export naming an address this machine does
    not have, and Cyclone answers that by warning once and choosing its own
    interface -- an empty topic list on a link that pings fine. A launch file
    that inherited it would be back to the domain being a property of the
    terminal, which is what it exists to stop.

    Set ARIES_KEEP_CYCLONEDDS_URI=1 to keep a deliberate hand-written config.
    """
    env = {
        "ROS_DOMAIN_ID": domain_id(),
        "RMW_IMPLEMENTATION": RMW,
    }
    existing = os.environ.get("CYCLONEDDS_URI", "").strip()
    if existing and os.environ.get("ARIES_KEEP_CYCLONEDDS_URI", "").strip():
        env["CYCLONEDDS_URI"] = existing
        return env
    config_path, _ = write_cyclone_config(path, require_link=require_link)
    env["CYCLONEDDS_URI"] = f"file://{config_path}"
    return env


def dds_launch_actions(path=None, require_link=True):
    """SetEnvironmentVariable actions to put at the TOP of a LaunchDescription.

    Place them above every node: launch runs actions in order and a node that
    starts first keeps the shell's environment, which is the whole failure
    being prevented here.
    """
    from launch.actions import LogInfo, SetEnvironmentVariable

    inherited = os.environ.get("CYCLONEDDS_URI", "").strip()
    env = dds_environment(path, require_link=require_link)
    actions = [SetEnvironmentVariable(name, value) for name, value in env.items()]
    actions.append(
        LogInfo(
            msg=f"[comms] domain {env['ROS_DOMAIN_ID']}, {env['RMW_IMPLEMENTATION']}"
        )
    )
    actions.append(LogInfo(msg=f"[comms] {env['CYCLONEDDS_URI']}"))
    if inherited and inherited != env["CYCLONEDDS_URI"]:
        # Said out loud rather than done quietly: someone put that there.
        actions.append(LogInfo(
            msg=f"[comms] replaced inherited CYCLONEDDS_URI={inherited} "
                f"(set ARIES_KEEP_CYCLONEDDS_URI=1 to keep yours). If that came "
                f"from ~/.bashrc, delete the line -- it names a fixed address "
                f"and is wrong on every machine but one."
        ))
    actions.append(
        LogInfo(
            msg="[comms] shells started by hand need: source \"$(ros2 pkg prefix "
            "aries_common)/share/aries_common/aries_dds_env.sh\""
        )
    )
    return actions


if __name__ == "__main__":
    written, address = write_cyclone_config(require_link=False)
    if address is None:
        print(f"local only (not on the field link), domain {domain_id()}")
    else:
        print(
            f"interface {address}, peers {', '.join(peers(address))}, "
            f"domain {domain_id()}"
        )
    print(written)
