"""One DDS transport configuration, correct on every machine on the field link.

The rover and the base station run the same workspace, so this file is the same
file on both. The one setting that must differ between them --
the pinned interface address -- is *detected* here rather than written down,
because a hand-mirrored copy is the failure this module exists to prevent:

    A DDS implementation treats an address the machine does not hold, and a
    config file it cannot open, as FATAL. It refuses to create the domain, so
    every node in the launch dies at startup with

        rmw_create_node: failed to create domain, error Error

    Measured under Jazzy, 2026-08-21. Older notes in this repo claimed it merely
    warns and falls back to defaults -- it does not, and a copied-and-not-edited
    config takes the whole stack down rather than degrading quietly.

That cuts both ways, and is why ``require_link`` exists below. A machine with
no antenna -- a developer laptop running simulation -- must NOT be pinned to a
field-link address, or the same fatality applies to it for no reason. It is
pinned to 127.0.0.1 instead, which every machine holds: see ``local_only_xml``
for why leaving the choice to the middleware is worse than either.

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

WHY EXPLICIT PEERS, AND WHY MULTICAST IS STILL ON

    Discovery goes to the explicit peers below so that a machine which the
    airMAX link will not carry multicast to is still found. A machine that is
    not in the ``hosts`` table will never be discovered: add it to devices.yaml,
    or set ARIES_EXTRA_PEERS=<ip>,<ip> for a one-off.

    MULTICAST IS NOT DISABLED HERE, whatever the Cyclone config this replaced
    did. An earlier version of this file claimed initialPeersList was the Fast
    DDS spelling of Cyclone's <AllowMulticast>false</AllowMulticast>. It is not:
    initial peers are announced to IN ADDITION TO the default multicast locator,
    and a participant started under this profile still joins 239.255.0.1 --
    check with ``ip maddr show dev <iface> | grep 239.255`` while the stack runs.

    DO NOT "FIX" THAT BY TURNING MULTICAST OFF without reading the next
    paragraph. Multicast is what currently discovers the thirty-odd participants
    on one machine to each other, and two things have to be dealt with first:

      * The peer list must gain this machine's own address. The 127.0.0.1 entry
        does NOT stand in for it -- see peers().
      * maxInitialPeersRange must be raised. Measured on one machine with
        multicast off and participant slots 0-5 already taken, a further
        listener heard 0 messages in 12 s at the default and 11 at 60.

    Measured both ways, same machine, talker and listener: multicast on, 11
    messages; multicast off with the peer list as it stands, 0.
"""

import ipaddress
import os
import socket
import subprocess
import tempfile

from aries_common.devices import device

# The middleware. Pinned, and there is only one -- a stack that comes up half on
# one vendor and half on another looks perfectly healthy per-node and cannot see
# itself.
#
# FAST DDS, AND THE REASON IS STRUCTURAL, NOT PREFERENCE. micro_ros_agent
# find_package(REQUIRED)s fastrtps, rmw_fastrtps_shared_cpp and
# rosidl_typesupport_fastrtps_cpp -- it CANNOT be built against CycloneDDS. The
# drill/gripper/load-cell board's topics are therefore always Fast DDS, and on
# 2026-08-26 a Cyclone stack could not discover them from any configuration
# tried: multicast either way, loopback-only, loopback plus the real NIC in
# <Interfaces>, the real NIC added to <Peers>, no config at all, and a Fast DDS
# UDP-only profile to rule out shared memory. Measured back to back with the
# board confirmed connected (agent log: 7 readers, 1 writer) both times:
#
#     rmw_fastrtps_cpp   board on the graph, gripper controller active,
#                        ZERO "Never received /gripper/state" warnings
#     rmw_cyclonedds_cpp 11 of those warnings in 60 s, gripper unusable
#
# Cyclone meant no gripper, no drill and no load cells, because all three live
# on that one board. The Cyclone path was REMOVED rather than left as an option:
# an escape hatch that silently disables three subsystems is not an escape
# hatch, it is a way to lose an afternoon.
#
# WHAT THE CYCLONE CONFIG GAVE THE FIELD LINK is reproduced in dds_xml():
# interfaceWhiteList for the interface pin, initialPeersList for the unicast
# peers. One Cyclone knob has no counterpart and needs none --
# MaxAutoParticipantIndex existed because Cyclone unicasts SPDP to a bounded
# range of participant indices and its default cap of 9 is below this stack's
# ~30 nodes. Fast DDS addresses peers by locator, so there is no cap to raise
# and no "Failed to find a free participant index" to hit.
RMW = "rmw_fastrtps_cpp"

# UDP socket buffer, in bytes, asked of every socket the transport opens.
#
# THIS IS THE MEASURED REGRESSION FROM THE 2026-08-26 SWITCH, and the reason the
# link felt slower afterwards. Fast DDS leaves sendBufferSize/receiveBufferSize
# at 0 (fastdds/rtps/transport/SocketTransportDescriptor.h), which means "take
# the OS default" -- it does NOT raise them. Cyclone asks for 1 MiB. Back to
# back on one machine, one `ros2 run demo_nodes_cpp talker` each, `ss -u -a -m`:
#
#     rmw_cyclonedds_cpp   rb2097152     (1 MiB asked, kernel doubles it)
#     rmw_fastrtps_cpp     rb212992      (net.core.rmem_default, untouched)
#
# A 10x smaller receive buffer, and 208 kB is less than ONE downlink frame pair
# -- camera_downlink measures 98 kB of colour plus 91 kB of depth. A burst that
# lands while the reader thread is descheduled overflows the socket, and on a
# best-effort keep_last(1) video topic the frame is simply gone. On the reliable
# topics it is worse: the loss is repaired, so it costs a NACK round trip on the
# link instead of a dropped frame.
#
# THE KERNEL CLAMPS THIS to net.core.rmem_max and then doubles what it grants,
# so what a socket actually gets is min(this, rmem_max) * 2.
# scripts/setup_system.sh raises rmem_max/wmem_max to match; on a machine that
# has not had that run, this number is silently cut down to the stock 208 kB and
# nothing anywhere says so. `ss -u -a -m | grep rb` is how you check.
SOCKET_BUFFER_BYTES = 8 * 1024 * 1024

# Largest UDP datagram the transport will build, in bytes.
#
# Fast DDS defaults this to 65500 (s_maximumMessageSize in
# fastdds/rtps/transport/TransportInterface.h), so a large sample is handed to
# the kernel as 64 kB datagrams and the IP layer chops each one into ~45
# fragments to fit the link's 1500-byte MTU. Losing ANY ONE of those 45 destroys
# the whole 64 kB, and the failed reassemblies sit in the kernel's fragment
# cache for ipfrag_time (30 s), where they crowd out unrelated flows -- which is
# how a burst of video ends up delaying a joystick packet. Cyclone never did
# this: it fragments at the RTPS layer instead, and its datagrams already fitted
# the MTU.
#
# 1400 leaves room for the IP and UDP headers with slack to spare, so nothing
# below RTPS ever has to fragment. Verified end to end: 640x480 rgb8 images
# (921600 bytes) publish and arrive intact at this setting, 99 of 100 sent.
MAX_DATAGRAM_BYTES = 1400

_CONFIG_BASENAME = "aries_fastdds.xml"
_AGENT_CONFIG_BASENAME = "aries_fastdds_agent.xml"


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

    # Bound to loopback, nothing off this machine is reachable, and the DDS
    # says so once per unreachable port per announcement -- sixty-odd lines per
    # peer per participant, which buries every real message in the log. A
    # single-host bench run peers with itself and nothing else.
    if str(local).startswith("127."):
        return ["127.0.0.1"]

    found = list(hosts().values()) + ["127.0.0.1"]
    extra = os.environ.get("ARIES_EXTRA_PEERS", "")
    found += [p.strip() for p in extra.split(",") if p.strip()]
    # Loopback stays in the list because a single-host bench run, where local
    # IS 127.0.0.1, has nothing else to peer with.
    #
    # IT DOES NOT DISCOVER THIS MACHINE'S OWN PARTICIPANTS TO EACH OTHER on a
    # machine that is on the link, whatever an earlier comment here said. The
    # transport is whitelisted to the field-link address, so participants listen
    # on that address and never on 127.0.0.1: announcements sent to the loopback
    # locator arrive nowhere. Multicast is what actually does that job today --
    # measured, with multicast off and this list unchanged, a listener heard 0
    # messages from a talker on the same machine in 12 s; adding `local` to the
    # list took it to 11. Adding `local` is therefore the first half of ever
    # turning multicast off; see the module docstring for the other half.
    return [p for p in dict.fromkeys(found) if p != local or p == "127.0.0.1"]


def dds_xml(local, peer_list, low_latency=False):
    """The transport config text for one machine.

    ``low_latency=True`` appends default DataWriter/DataReader profiles for the
    micro-ROS agent. See write_agent_dds_config() for why the agent needs the
    participant block below as well, and cannot simply be handed a file of its
    own that only carries the QoS.

    Two of the three guarantees, expressed in the other vendor's dialect:

      Cyclone <Interfaces><NetworkInterface address=/>  -> interfaceWhiteList
      Cyclone <Discovery><Peers><Peer address=/>        -> initialPeersList
      Cyclone <AllowMulticast>false</AllowMulticast>    -> NOT REPRODUCED

    The third one is not carried over and the module docstring says why: adding
    initial peers does not stop Fast DDS announcing to multicast as well, and
    multicast is currently the only thing discovering this machine's own
    participants to each other.

    useBuiltinTransports=false is not decoration: it stops Fast DDS adding
    transports of its own on top of the two declared here, so the only route off
    this machine is the whitelisted UDPv4 one and the interface pin holds.

    SHARED MEMORY IS DECLARED ON PURPOSE, which reverses an earlier decision in
    this file. The objection to it was that it bypasses the interface pin and is
    invisible to tooling watching the wire. The first half is not true -- SHM
    cannot leave the host, so it can never put a byte on the wrong NIC, which is
    what the pin exists to prevent -- and the second is a debugging cost that
    got much too expensive to keep paying once MAX_DATAGRAM_BYTES came down to
    1400. The rover's own raw camera streams are the heaviest traffic on this
    machine and never leave it (the RealSense drivers to camera_downlink, of the
    order of 400 Mbit/s), and pushing those through UDP loopback in 1400-byte
    datagrams is roughly 20k extra packets/s per camera of pure kernel work.
    SHM carries them instead, which is faster than the UDP loopback path they
    were on before this change, and UDPv4 is left carrying only the link.

    THE COST SHM DOES BRING is zombie segments. A process killed outright --
    e-stop, power cut, SIGKILL -- leaves its /dev/shm/fastrtps_* files behind,
    of the order of 12 MB per abandoned stack. Nothing here deletes them,
    because doing that at launch would risk taking out a stack that is already
    running. `fastdds shm clean` removes only the ones no process holds, and is
    the thing to run if /dev/shm ever fills.

    THERE IS a MaxAutoParticipantIndex equivalent, and an earlier version of
    this file said there was not. It is maxInitialPeersRange, it belongs to the
    transport descriptor, and it defaults to FOUR
    (s_maximumInitialPeersRange, fastdds/rtps/transport/TransportInterface.h) --
    lower than Cyclone's default of 9, against the same ~30-node stack. It is
    the number of participant slots probed at each address in initialPeersList,
    so unicast discovery reaches the first five participants on a peer machine
    and no more.

    It is left at the default here ON PURPOSE, because it is not currently load
    bearing: multicast has not been disabled (see above), so it is multicast and
    not the peer sweep that discovers the far machine's later participants. The
    two are a pair -- raise this the moment multicast goes away, and expect it
    to cost airtime when you do, since every participant then sweeps every slot
    at every peer on every announcement instead of sending one multicast frame.

    If the base station ever shows only the first handful of the rover's nodes,
    this is the knob: the link stopped carrying multicast and the sweep is all
    that is left.
    """
    peer_xml = "\n".join(
        f"            <locator><udpv4><address>{p}</address></udpv4></locator>"
        for p in peer_list
    )
    buffer_bytes = SOCKET_BUFFER_BYTES
    datagram_bytes = MAX_DATAGRAM_BYTES
    # SYNCHRONOUS + a zero latency budget are wanted for the agent's 100 Hz
    # gripper topics and NOT for the rest of the stack: as a default writer
    # profile they would also apply to the camera and point-cloud writers,
    # which send large samples and want the asynchronous thread.
    qos_xml = """
    <publisher profile_name="aries_low_latency" is_default_profile="true">
      <qos>
        <publishMode><kind>SYNCHRONOUS</kind></publishMode>
        <latencyBudget><duration><sec>0</sec><nanosec>0</nanosec></duration></latencyBudget>
      </qos>
      <historyMemoryPolicy>PREALLOCATED_WITH_REALLOC</historyMemoryPolicy>
    </publisher>
    <subscriber profile_name="aries_low_latency" is_default_profile="true">
      <qos>
        <latencyBudget><duration><sec>0</sec><nanosec>0</nanosec></duration></latencyBudget>
      </qos>
      <historyMemoryPolicy>PREALLOCATED_WITH_REALLOC</historyMemoryPolicy>
    </subscriber>""" if low_latency else ""
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- GENERATED by aries_common.comms on {socket.gethostname()}. Do not edit:
     it is rewritten every time a launch file or aries_dds_env.sh runs. The
     interface address below is detected, which is what lets the rover and the
     base station share one workspace without mirroring a file by hand. -->
<dds xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <profiles>
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>aries_udpv4</transport_id>
        <type>UDPv4</type>
        <!-- Order matters: fastRTPS_profiles.xsd declares these as an
             xs:sequence, so they go before interfaceWhiteList. -->
        <sendBufferSize>{buffer_bytes}</sendBufferSize>
        <receiveBufferSize>{buffer_bytes}</receiveBufferSize>
        <maxMessageSize>{datagram_bytes}</maxMessageSize>
        <interfaceWhiteList>
          <address>{local}</address>
        </interfaceWhiteList>
      </transport_descriptor>
      <!-- Same-host traffic only; SHM has no interfaceWhiteList because it can
           never reach another machine. Left at the stock 512 kB segment, which
           carries 640x480 rgb8 frames without a fallback to UDP. -->
      <transport_descriptor>
        <transport_id>aries_shm</transport_id>
        <type>SHM</type>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="aries" is_default_profile="true">
      <rtps>
        <userTransports>
          <transport_id>aries_shm</transport_id>
          <transport_id>aries_udpv4</transport_id>
        </userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
        <builtin>
          <initialPeersList>
{peer_xml}
          </initialPeersList>
        </builtin>
      </rtps>
    </participant>{qos_xml}
  </profiles>
</dds>
"""


def local_only_xml():
    """Fast DDS config for a machine that is not on the field link at all.

    Loopback only -- deliberately the same situation local_only_xml() covers,
    and pinned for the same reason: an unpinned participant binds to whatever
    interface it found at creation time and never re-selects, so pulling a cable
    mid-run kills every announcement it had already bound to.
    """
    return dds_xml("127.0.0.1", ["127.0.0.1"])


def write_dds_config(path=None, require_link=True):
    """Write the config for this machine and return (path, local_address).

    ``require_link=True`` (the field launches) raises when this machine is not
    on the link, so a missing cable fails loudly here instead of silently later
    as an empty topic list.

    ``require_link=False`` (the shell environment, simulation) falls back to a
    local-only config and returns ``(path, None)``. A developer laptop has no
    antenna and does not need one; making it an error there would mean the whole
    simulation stack refuses to start on any machine off the field.
    """
    local = local_address()
    if path is None:
        path = os.path.join(tempfile.gettempdir(), _CONFIG_BASENAME)
    if local is None and not require_link:
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
    with open(path, "w") as handle:
        handle.write(dds_xml(local, peers(local)))
    return path, local


def write_agent_dds_config(path=None, require_link=True):
    """The micro-ROS agent's profile: this machine's pin PLUS low-latency QoS.

    FAST DDS LOADS EXACTLY ONE PROFILES FILE PER PROCESS, and the variable that
    names it is read in preference to anything else the process inherited (of
    the two spellings, FASTRTPS_DEFAULT_PROFILES_FILE is the one that wins).
    So whatever this file says about the participant is the WHOLE story for the
    agent -- the stack-wide file is not merged in, it is simply not read.

    That is why this reuses dds_xml() rather than being a hand-written file of
    QoS overrides. config/fastdds_low_latency.xml used to be exactly that, and
    it pinned the agent's metatraffic to 127.0.0.1 -- correct when it was
    written, because the stack was on CycloneDDS then and the agent was the
    only Fast DDS participant on the machine. Once the stack moved to Fast DDS
    and started generating an interface-pinned profile of its own, the two no
    longer overlapped: the agent announced only on loopback, every other node
    ran on a UDPv4 transport whitelisted to the field-link address with
    useBuiltinTransports=false, and the two could not discover each other.

    Nothing about that looks like a fault. The agent starts, the board's
    session comes up, `ros2 topic list` shows /gripper/state because the host
    subscribes to it, and the controller goes active -- while the ONLY path
    from the Teensy to the servo is severed, and TeensyGripperSystem warns
    "Never received /gripper/state" every 5 s forever. Keep the agent on the
    same participant configuration as everything else it has to talk to.
    """
    local = local_address()
    if path is None:
        path = os.path.join(tempfile.gettempdir(), _AGENT_CONFIG_BASENAME)
    if local is None and not require_link:
        with open(path, "w") as handle:
            handle.write(dds_xml("127.0.0.1", ["127.0.0.1"], low_latency=True))
        return path, None
    if local is None:
        raise RuntimeError(
            "No interface on the field link "
            f"({', '.join(sorted(hosts().values())) or 'no hosts configured'}).\n"
            "  The micro-ROS agent needs the same pin as the rest of the stack."
        )
    with open(path, "w") as handle:
        handle.write(dds_xml(local, peers(local), low_latency=True))
    return path, local


def dds_environment(path=None, require_link=True):
    """The variables every ARIES process must agree on.

    A profile path already in the environment is REPLACED, not honoured. That
    looks aggressive and is the entire point of this module: the value in your
    shell is almost always a stale export naming an address this machine does
    not have, and the middleware answers that by falling back to its own
    defaults -- an empty topic list on a link that pings fine. A launch file
    that inherited it would be back to the domain being a property of the
    terminal, which is what this exists to stop.

    Set ARIES_KEEP_FASTDDS_PROFILES=1 to keep a deliberate hand-written profile.
    """
    env = {
        "ROS_DOMAIN_ID": domain_id(),
        "RMW_IMPLEMENTATION": RMW,
    }

    existing = (os.environ.get("FASTDDS_DEFAULT_PROFILES_FILE", "").strip()
                or os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip())
    if existing and os.environ.get("ARIES_KEEP_FASTDDS_PROFILES", "").strip():
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = existing
        env["FASTDDS_DEFAULT_PROFILES_FILE"] = existing
        return env

    config_path, _ = write_dds_config(path, require_link=require_link)

    # BOTH names, deliberately. Fast DDS renamed the variable at 2.12 and still
    # honours the old one; which of the two a given build reads is not worth
    # discovering in the field, and setting both costs nothing.
    #
    # PLAIN PATHS, no file:// prefix. Fast DDS silently IGNORES a prefixed value
    # -- every participant then quietly runs on its own defaults, which is the
    # same empty-topic-list symptom this function exists to prevent.
    #
    # str(), because callers pass pathlib paths and these become real
    # environment variables -- SetEnvironmentVariable and os.environ both want
    # text, and a PosixPath here surfaces much later as a launch that will not
    # start.
    env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(config_path)
    env["FASTDDS_DEFAULT_PROFILES_FILE"] = str(config_path)
    return env


def dds_launch_actions(path=None, require_link=True):
    """SetEnvironmentVariable actions to put at the TOP of a LaunchDescription.

    Place them above every node: launch runs actions in order and a node that
    starts first keeps the shell's environment, which is the whole failure
    being prevented here.
    """
    from launch.actions import LogInfo, SetEnvironmentVariable

    _CONFIG_VAR = "FASTDDS_DEFAULT_PROFILES_FILE"

    inherited = os.environ.get(_CONFIG_VAR, "").strip()
    env = dds_environment(path, require_link=require_link)
    actions = [SetEnvironmentVariable(name, value) for name, value in env.items()]
    actions.append(
        LogInfo(
            msg=f"[comms] domain {env['ROS_DOMAIN_ID']}, {env['RMW_IMPLEMENTATION']}"
        )
    )
    actions.append(LogInfo(msg=f"[comms] {_CONFIG_VAR}={env[_CONFIG_VAR]}"))
    if inherited and inherited != env[_CONFIG_VAR]:
        # Said out loud rather than done quietly: someone put that there.
        actions.append(LogInfo(
            msg=f"[comms] replaced inherited {_CONFIG_VAR}={inherited} "
                f"(set ARIES_KEEP_FASTDDS_PROFILES=1 to keep yours). If that came "
                f"from ~/.bashrc, delete the line -- it names a fixed address "
                f"and is wrong on every machine but one."
        ))
    # A CYCLONEDDS_URI left over from before 2026-08-26 is inert now, and is
    # exactly what someone will find later and be misled by. This stack no
    # longer supports Cyclone at all -- see the note on RMW above.
    stale_cyclone = os.environ.get("CYCLONEDDS_URI", "").strip()
    if stale_cyclone:
        actions.append(LogInfo(
            msg=f"[comms] NOTE CYCLONEDDS_URI={stale_cyclone} is set and is "
                f"IGNORED -- this stack is Fast DDS only. Delete the export."
        ))
    actions.append(
        LogInfo(
            msg="[comms] shells started by hand need: source \"$(ros2 pkg prefix "
            "aries_common)/share/aries_common/aries_dds_env.sh\""
        )
    )
    return actions


if __name__ == "__main__":
    written, address = write_dds_config(require_link=False)
    if address is None:
        print(f"local only (not on the field link), domain {domain_id()}")
    else:
        print(
            f"interface {address}, peers {', '.join(peers(address))}, "
            f"domain {domain_id()}"
        )
    print(written)
