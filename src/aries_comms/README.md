# aries_comms

The field link, both ends. One command on each machine:

```bash
# ROVER
ros2 launch aries_comms rover_field.launch.py

# BASE STATION  — the joystick plugs in HERE
ros2 launch aries_comms base_station.launch.py
```

Both machines run the same workspace, so there is nothing to copy and nothing
to mirror by hand.

## Why one package

These two launch files are not two features, they are one decision written
twice. Which end reads the pad, which end runs RViz, which end decompresses the
downlink, which cameras are in the list — every one of those is a pair of
settings that has to agree, and neither half fails loudly when it does not: two
joy drivers make the buttons chatter, a `color_only` mismatch leaves a
decompressor waiting forever on a stream nobody sends.

They used to live in `aries_bringup` and `aries_base_station`, so a change to
one half could be reviewed, merged and deployed without the other ever being
opened. `test/test_field_link_contract.py` now compares them directly, which is
only possible because they are in the same package.

What stayed in `aries_bringup` is what is about the **robot** rather than about
the link: `full_hardware.launch.py`, the camera pipeline at both ends
(`camera_downlink.launch.py`, `camera_view.launch.py`), and the rover-side
checker. The dependency runs one way — `aries_comms` → `aries_bringup` — and
has to: an include pointing back the other way is a colcon dependency cycle.

The DDS transport itself is still `aries_common.comms`, not here. It has to be
importable on a machine that launches neither side, because
`aries_dds_env.sh` is what repairs a plain shell.

## Status checks: one per machine, and they are not interchangeable

`full_hardware_checker` stays on the rover. Everything it looks at is on the
robot — the gripper's serial port, the CAN link, ODrive heartbeats, RealSense
USB enumeration — and none of that can be sampled across DDS. Its output goes
to the rover's console, which in the field nobody is reading.

`base_station_checker` is the operator-side one, started by
`base_station.launch.py` and runnable alone:

```bash
ros2 launch aries_comms base_station_checker.launch.py
ros2 service call /check_base_station std_srvs/srv/Trigger   # force a print
```

It reports the four things that can be wrong at this end and are all silent:

- **the link** — this machine's address and which host it is, the domain, the
  RMW, and the interface the DDS config on disk actually pins, read from
  *this process's* environment rather than recomputed. That is the point: the
  classic failure is a launch started from a terminal older than the exports,
  sitting on domain 0 while everything else looks fine.
- **the pad** — the device node, and the **number of publishers on `/joy`**.
  Two is the expensive one, because it looks like a working pad.
- **the downlink** — per stream, whether the rover is publishing (from the
  graph) and whether frames are actually arriving. It measures arrivals on the
  machine-local `/<cam>/view/*`, *never* on `/downlink/*`: subscribing to a
  compressed stream is what pulls it over the antenna, and a second participant
  doing so pulls a second copy. Measuring the view output is also the better
  test — a frame there means the link *and* the decompressor both worked.
- **the rover** — `/tf`, `/joint_states`, `/robot_description`. Fresh `/tf`
  with no downlink is a camera problem; no `/tf` at all is a link or domain
  problem, and the two are worth telling apart before touching a radio.

It also counts RViz instances, and says so if there is more than one.

For actual bandwidth, `ros2 run aries_bringup downlink_report.py` — it
subscribes to the compressed topics on purpose, briefly, and says so.

**Setting up from scratch — addresses, radios, verification — is
[`FIELD_SETUP.md`](../../FIELD_SETUP.md) at the repo root.** This file is the
reference for running the link once that is done.

## What each side starts

| | rover (`rover_field.launch.py`) | base station (`base_station.launch.py`) |
|---|---|---|
| DDS | set by the launch | set by the launch |
| joystick driver | no (`use_joy_node:=false`) | **yes** (`use_joy_node:=true`) |
| teleop consumers | yes — arm, presets, drive | no |
| cameras | drivers + compressors | decompressors |
| RViz | no (`use_gui:=false`) | yes — **one** |
| `/tf`, `/joint_states` | published | consumed |
| status check | `full_hardware_checker` | `base_station_checker` |

## What crosses the antenna

```
rover                                    base station
  /downlink/<cam>/color/compressed   ->   /<cam>/view/color     JPEG,  BEST_EFFORT
  /downlink/<cam>/depth/compressedDepth-> /<cam>/view/depth     PNG16, BEST_EFFORT
  /downlink/<cam>/camera_info        ->                         latched
  /tf, /joint_states, /robot_description  ->
                                     <-   /joy                  ~130 kbit/s
```

`balanced` is 28.3 Mbit/s for both cameras. `downlink_profile:=lean` is 10.9,
`quality` is 42.3. All three are far inside what a Rocket 5AC pair carries at
150 m; pick on picture quality, not bandwidth.

**Never point anything here at a rover camera topic.** RViz's ROS 2 Image
display has no transport selection — it subscribes raw, always — and the two
cameras raw are about 737 Mbit/s (640×480, colour+depth, 30 fps). That does not
just make the images slow, it congestion-collapses the link and everything on
it goes laggy. `ros2 topic hz` and `ros2 topic bw` subscribe like any other
node: the same rule applies to them. Read `/<camera>/view/*`, which is local.

## Latency

The radio is not the problem at 150 m. Measured/derived budget, glass to glass:

| stage | ms |
|---|---|
| propagation, 150 m | 0.0005 |
| airMAX AC TDMA round trip | 1–4 |
| capture at 30 fps | ~33 |
| downlink rate gate at 15 Hz | 0–67 |
| JPEG encode + decode | 10–20 |
| RViz render | one frame |
| **total** | **~150–250** |

Control (`/joy` → ODrive) is 10–20 ms — effectively instant. If the video lag
matters for a delicate placement, `downlink_rate_hz:=30` halves the rate-gate
term and costs bandwidth you have.

## Safety: the pad is now across a radio link

Every teleop node stops on 0.35 s of `/joy` silence — the arm, the drill, the
presets, and (as of the same change that added this package) both rover drive
nodes. A dropout is indistinguishable from a held stick, so this is the only
thing standing between a lost link and a rover that keeps driving on its last
command.

`joy_node` autorepeats at 80 Hz, so a held stick is a continuous stream and
0.35 s is 28 missed messages. **Do not raise `joy_timeout_sec` to paper over a
marginal link.** Fix the link — see the radio notes below.

## Radio: Ubiquiti Rocket 5AC (R5AC-Lite) at 150 m

- **Turn TX power down.** At 150 m with dish antennas you will be around
  −30 dBm, which overdrives the receiver and *drops* the modulation rate.
  Target −50 to −60 dBm. The link getting worse the closer you park is the
  most confusing failure in this whole stack.
- **Use a non-DFS channel** (5.8 GHz, 5745–5825). On DFS, one radar detection
  blanks the link for 60 s in the middle of a run.
- **Fresnel clearance.** At 150 m / 5.8 GHz the first Fresnel zone radius at
  the midpoint is ~1.4 m; you want ~0.85 m clear. A rover antenna low to the
  ground looking over a berm clips it. Base antenna on a tripod.
- **40 MHz channel width**, not 80. 28 Mbit/s needs nothing wider, and narrow
  is more robust.
- **Fix the channel** — disable airSelect.
- One end AP-PTP, one Station-PTP. Both ends must be 5AC; no airMAX M interop.
- Multicast is off in the DDS config on purpose: airMAX sends multicast at its
  lowest data rate.

## Troubleshooting

Work bottom up and stop at the first failure.

```bash
# 1. link
./scripts/setup_field_link.sh --check    # address, identity, peers, conflicts

# 2. this shell's environment  — must match on both machines
source "$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh"
env | grep -E "RMW_|ROS_DOMAIN|CYCLONEDDS|FASTDDS|FASTRTPS"

# 3. what a RUNNING process actually has (this is the one that catches it)
tr '\0' '\n' < /proc/<pid>/environ | grep -E "ROS_DOMAIN_ID|RMW_"
#    no output = unset = domain 0, NOT domain 30 (the DOMAIN is what bites;
#    unset RMW happens to be Fast DDS, which is what the stack pins anyway)

# 4. end to end
ros2 run demo_nodes_cpp talker        # rover
ros2 run demo_nodes_cpp listener      # base

# 5. the graph
ros2 daemon stop                      # it caches per (domain, rmw) and serves stale
ros2 topic list | grep ^/downlink/
ros2 run aries_bringup downlink_report.py
```

| symptom | cause |
|---|---|
| empty topic list, ping works | domain / RMW mismatch. Check `/proc/<pid>/environ`, not just your shell — a process keeps what it STARTED with, so a terminal opened before the environment was set stays wrong forever. |
| topics listed, no data | QoS. The downlink is BEST_EFFORT; a RELIABLE subscriber never matches it at all, so the topic lists fine and never delivers. |
| `Failed to find a free participant index` | **Cyclone only** (`ARIES_RMW=rmw_cyclonedds_cpp`): more participants than `MAX_AUTO_PARTICIPANT_INDEX` in `aries_common/comms.py`. The stack is ~30 nodes; Cyclone's own default cap is 9. Fast DDS addresses peers by locator, not by a bounded index range, so it has no equivalent. |
| buttons chatter, teleop unreproducible | two joy drivers. Exactly one machine may set `use_joy_node:=true`; `base_station_checker` counts the publishers on `/joy`. |
| two RViz windows, one of them blank | fixed, and worth knowing why. `camera_view.launch.py` declared a `use_rviz` of its own, and an include inherits the parent's launch configurations — so this file's `use_rviz` (default true) switched that one on too, with this file's `rviz_config` of `""`. The node is gone from `camera_view` and the include is now `forwarding=False`. A second window today was started by hand. |
| link works, then drops when another team powers on | duplicate address. `./scripts/setup_field_link.sh --check` |
| `WARNING: ... this is a GUESS` at launch | this machine has no static address. `./scripts/setup_field_link.sh {rover\|base}` |
| duplicate node names | two launches alive. `pgrep -f "ros2 launch"` |
| worked, then died after a replug | new ifindex, old sockets dead. Restart nodes on both ends. |
| one direction only | firewall. DDS needs UDP both ways: `sudo ufw allow from 192.168.1.0/24` |
| `does not match an available interface` | a stale config naming a fixed address — `CYCLONEDDS_URI`, or `FASTDDS_DEFAULT_PROFILES_FILE` / `FASTRTPS_DEFAULT_PROFILES_FILE`. Delete the export from `~/.bashrc`; the launch files overwrite it and say so. |
| gripper, drill or load cells dead while everything else is fine | the stack is on Cyclone. `micro_ros_agent` cannot be built against Cyclone, so the Teensy's topics are always Fast DDS and a Cyclone stack cannot discover them. `ros2 topic info /gripper/state` shows **Publisher count: 0**. Fast DDS is the default; check nothing has set `ARIES_RMW`. |
| `FASTDDS_DEFAULT_PROFILES_FILE` set but ignored | it takes a **plain path**, not a `file://` URI. Fast DDS silently ignores a prefixed value and every participant quietly falls back to its own defaults. |

## Configuration

Addresses and the domain live in one place — `aries_common/config/devices.yaml`.
Note the **arm** is not on the field link: the igus control box is on
`192.168.3.11:3920`, a separate subnet, and that address is compiled into the
driver (`igus_rebel/include/igus_rebel/Rebel.hpp`) rather than read from here.
Losing the antenna does not touch the arm, and vice versa. See
[`FIELD_SETUP.md`](../../FIELD_SETUP.md) §1.

```yaml
network:
  domain_id: 30
  subnet_prefix: 24
  hosts:
    rover: "192.168.1.10"
    base: "192.168.1.11"
```

Adding a third machine (a second screen, a judge's laptop) is one line there.
Without it that machine is never discovered, because multicast is off.

For a one-off, `ARIES_EXTRA_PEERS=10.0.0.9`. For a bench run with no field link
at all, `ARIES_LOCAL_ADDRESS=127.0.0.1` (one machine) or this machine's LAN
address (two machines on a switch).

## Match the fingers to the rover

`base_station.launch.py` builds the robot description locally, so it is only as
correct as the arguments given to it:

```bash
ros2 launch aries_comms base_station.launch.py finger_type:=probe
```

A base station showing bucket fingers while the rover carries the probe tips is
a plausible-looking display that is quietly wrong, and MoveIt would plan against
the wrong collision model.
