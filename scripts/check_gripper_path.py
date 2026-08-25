#!/usr/bin/env python3
"""Find which link in the gripper chain is broken.

    ./scripts/check_gripper_path.py          # on the ROVER: every check
    ./scripts/check_gripper_path.py          # on the base station: graph checks

"The gripper does not move" has six possible causes and they all look the same
from RViz, because the controller reports success either way. The JTC tracks
`servo_pos_`, which teensy_gripper_system.cpp echoes from the command whenever
the board is silent -- so every goal SUCCEEDS, /joint_states moves, the model in
RViz closes its fingers, and nothing at all reaches the servo.

full_hardware_checker does not separate these either. Its "Serial/Teensy —
present" row globs for a device node, and a device node says only that a board
is plugged in: not that it is executing, not that the agent owns the port, not
that the micro-ROS session exists. It printed a tick through the whole session
this script was written for.

THE CHAIN, in the order this walks it

    1  Teensy enumerated              /dev/serial/by-id/*Teensy*-if00
    2  micro_ros_agent alive          and holding that port
    3  /teensy_gripper in the graph   the board created its entities
    4  /gripper/state ticking         the board is EXECUTING, not just enrolled
    5  /gripper/cmd wired both ways   plugin publishes, board subscribes
    6  controller + joint             rebel_gripper_controller, gripper_gear_left_joint

Steps 1 and 2 are rover-local and are skipped with a note when this is run at
the base station. Steps 3 to 6 cross the link and work from either end.

THE FAILURE THIS EXISTS FOR is step 4 passing step 3: the firmware registers its
entities, then stops executing. The agent keeps the stale entities alive, so
/teensy_gripper and /gripper/state both exist in the graph and neither ever
carries a message. destroy_entities() finalising a handle that was never
initialised hard-faults the MCU into exactly that state, and USB stays
enumerated through it -- indistinguishable from a healthy board unless you look
at the message rate, which is what step 4 does.

Nothing here commands the gripper. It is safe to run at any time, including
mid-operation.
"""

import argparse
import glob
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32

TEENSY_GLOB = "/dev/serial/by-id/*Teensy*-if00"
STATE_TOPIC = "/gripper/state"
CMD_TOPIC = "/gripper/cmd"
BOARD_NODE = "teensy_gripper"
CONTROLLER = "rebel_gripper_controller"
GRIPPER_JOINT = "gripper_gear_left_joint"

# The firmware publishes /gripper/state BEST_EFFORT on purpose -- a reliable
# 100 Hz stream over serial XRCE stalls and retransmits instead of sending new
# samples. A RELIABLE subscriber does not match a BEST_EFFORT publisher AT ALL,
# so asking for the wrong one here would report a healthy board as silent.
STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
    depth=10,
)

G, R, Y, C, GREY, B, RST = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[90m", "\033[1m", "\033[0m"
)


class Probe(Node):
    def __init__(self):
        super().__init__("gripper_path_probe")
        self.stamps = []
        self.values = []
        self.create_subscription(Float32, STATE_TOPIC, self._cb, STATE_QOS)

    def _cb(self, msg):
        self.stamps.append(time.monotonic())
        self.values.append(msg.data)

    def rate(self):
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else 0.0


def _agent_processes():
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.strip()
        for line in out.splitlines()
        if "micro_ros_agent" in line and "check_gripper_path" not in line
    ]


def _port_holder(port):
    """Which pids have the serial port open, if fuser is available."""
    try:
        done = subprocess.run(
            ["fuser", port], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.split() or None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample", type=float, default=3.0,
                    help="seconds to watch %s for (default 3)" % STATE_TOPIC)
    args = ap.parse_args()

    problems = []
    print(f"\n{'═'*74}\n{B}{C}  ARIES GRIPPER PATH CHECK{RST}\n{'═'*74}")

    # ── 1 & 2: rover-local ────────────────────────────────────────────────────
    ports = sorted(glob.glob(TEENSY_GLOB))
    agents = _agent_processes()
    local = bool(ports or agents)

    print(f"\n{B}  On this machine:{RST}")
    if not local:
        # Not a fault: at the base station the board and the agent are both a
        # radio link away. Say which checks were skipped rather than passing
        # them silently, or a clean report here would read as a clean rover.
        print(f"  {GREY}○ no Teensy and no agent here — base station, or the "
              f"rover is not this machine{RST}")
        print(f"  {GREY}  steps 1-2 skipped; run this ON THE ROVER to cover them{RST}")
    else:
        if ports:
            print(f"  {G}✓{RST} Teensy enumerated — {G}{ports[0]}{RST}")
            if len(ports) > 1:
                print(f"  {Y}~{RST} {len(ports)} Teensys present: {', '.join(ports)}")
        else:
            print(f"  {R}✗{RST} no Teensy at {TEENSY_GLOB}")
            problems.append(
                "no Teensy on USB: the gripper resolved to mock_hardware at "
                "launch, so no command can reach a servo. Replug and relaunch"
            )
        if agents:
            print(f"  {G}✓{RST} micro_ros_agent — {G}{len(agents)} running{RST}")
            if len(agents) > 1:
                print(f"  {R}✗{RST} more than one agent")
                problems.append(
                    f"{len(agents)} micro_ros_agent processes: only one may own "
                    "the port. Kill the stale one and let the launch respawn its own"
                )
            if ports:
                holders = _port_holder(ports[0])
                if holders:
                    print(f"  {G}✓{RST} port held by pid(s) {' '.join(holders)}")
                else:
                    print(f"  {R}✗{RST} nothing has {ports[0]} open")
                    problems.append(
                        "the agent is running but does not hold the serial port: "
                        "it failed to open it (permissions, or another process "
                        "took it) and is respawning against a port it cannot use"
                    )
        else:
            print(f"  {R}✗{RST} micro_ros_agent — {R}not running{RST}")
            problems.append(
                "no micro_ros_agent: it is the ONLY path between /gripper/cmd "
                "and the servo. aries_hardware.launch.py starts it only when the "
                "gripper resolved to the real board — check the '[gripper auto]' "
                "line in the launch output"
            )

    # ── 3-6: the graph, from either end ───────────────────────────────────────
    rclpy.init()
    probe = Probe()
    deadline = time.monotonic() + max(1.0, args.sample)
    while time.monotonic() < deadline:
        rclpy.spin_once(probe, timeout_sec=0.05)

    board_up = any(name == BOARD_NODE for name, _ in probe.get_node_names_and_namespaces())
    state_pubs = probe.count_publishers(STATE_TOPIC)
    cmd_pubs = probe.count_publishers(CMD_TOPIC)
    cmd_subs = probe.count_subscribers(CMD_TOPIC)
    hz = probe.rate()

    print(f"\n{B}  micro-ROS session:{RST}")
    if board_up:
        print(f"  {G}✓{RST} /{BOARD_NODE} — {G}node present, entities created{RST}")
    else:
        print(f"  {R}✗{RST} /{BOARD_NODE} — {R}not in the graph{RST}")
        problems.append(
            f"the board never created its entities: it is stuck in WAITING_AGENT "
            f"(pinging a port nothing answers) or create_entities() is failing. "
            f"Reset the Teensy with the agent already running"
        )

    if hz > 50.0:
        print(f"  {G}✓{RST} {STATE_TOPIC} — {G}{hz:.0f} Hz, "
              f"last={probe.values[-1]:.3f}{RST}")
    elif probe.stamps:
        print(f"  {Y}~{RST} {STATE_TOPIC} — {Y}{hz:.0f} Hz, expected ~100{RST}")
        problems.append(
            f"{STATE_TOPIC} at {hz:.0f} Hz of ~100: the session is up but "
            "starved. Over the field link that is the link; on the rover it is "
            "the serial stream"
        )
    elif state_pubs:
        # The one this script exists for.
        print(f"  {R}✗{RST} {STATE_TOPIC} — {R}{state_pubs} publisher, NO messages{RST}")
        problems.append(
            "THE BOARD IS ENROLLED BUT NOT EXECUTING. /gripper/state is "
            "advertised and silent, which is what a hard-faulted sketch looks "
            "like: the agent holds the stale entities open and USB stays "
            "enumerated. Only a PHYSICAL RESET of the Teensy clears it — press "
            "the button on the board, or unplug and replug it, with the agent "
            "left running. It will rejoin on its own"
        )
    else:
        print(f"  {R}✗{RST} {STATE_TOPIC} — {R}no publisher{RST}")
        problems.append(
            f"nothing publishes {STATE_TOPIC}: the board is not in a micro-ROS "
            "session at all. Reset the Teensy, then check the agent's output"
        )

    print(f"\n{B}  Command path:{RST}")
    if cmd_pubs:
        print(f"  {G}✓{RST} {CMD_TOPIC} — {G}{cmd_pubs} publisher "
              f"(the hardware plugin){RST}")
    else:
        print(f"  {R}✗{RST} {CMD_TOPIC} — {R}no publisher{RST}")
        problems.append(
            f"nothing publishes {CMD_TOPIC}: the gripper resolved to "
            "mock_hardware, so TeensyGripperSystem was never loaded. The launch "
            "log's '[gripper auto] resolved=' line says which"
        )
    if cmd_subs:
        print(f"  {G}✓{RST} {CMD_TOPIC} — {G}{cmd_subs} subscriber (the board){RST}")
    else:
        print(f"  {Y}~{RST} {CMD_TOPIC} — {Y}no subscriber{RST}")

    print(f"\n{B}  Controller:{RST}")
    ctrl_up = (
        probe.count_publishers(f"/{CONTROLLER}/state") > 0
        or probe.count_subscribers(f"/{CONTROLLER}/joint_trajectory") > 0
    )
    if ctrl_up:
        print(f"  {G}✓{RST} {CONTROLLER} — {G}active{RST}")
    else:
        print(f"  {R}✗{RST} {CONTROLLER} — {R}not detected{RST}")
        problems.append(f"{CONTROLLER} is not running: nothing accepts a goal")

    probe.destroy_node()
    rclpy.shutdown()

    print(f"\n{'═'*74}")
    if not problems:
        print(f"  {G}{B}✓  GRIPPER PATH INTACT{RST}")
        print(f"  {GREY}·  a goal now reaches the servo. If it still will not "
              f"move, the fault is mechanical or in the servo itself{RST}")
    else:
        print(f"  {R}{B}✗  GRIPPER PATH BROKEN{RST}")
        for problem in problems:
            print(f"  {R}→  {problem}{RST}")
        print(f"  {C}·  the stack light is on the SAME Teensy and the same "
              f"session — if it is dark, the board is not executing, which "
              f"confirms this from across the room{RST}")
    print(f"{'═'*74}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
