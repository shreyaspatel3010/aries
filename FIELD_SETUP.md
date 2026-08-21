# Field setup — rover and base station

Setting up the two machines and the radio link between them, from a bare
install to a rover you can drive at 150 m.

Read this once end to end before the first setup. On competition day, use the
[checklist](#competition-day-checklist) at the bottom.

---

## 0. The shape of it

Both machines run **the same workspace**. Nothing is copied by hand; the
difference between them is which launch file you start and which static address
they hold.

```
        ROVER PC                    5 GHz link                 BASE STATION PC
   192.168.1.10                    150 m LOS                   192.168.1.11
   ┌──────────────┐          ┌────────┐   ┌────────┐         ┌──────────────┐
   │ arm + gripper│          │ Rocket │≈≈≈│ Rocket │         │ joystick     │
   │ MoveIt       │──enp130s0┤  5AC   │   │  5AC   ├enp130s0─│ decompressors│
   │ rover drive  │          │  .20   │   │  .21   │         │ RViz         │
   │ 2× RealSense │          └────────┘   └────────┘         └──────────────┘
   └──────┬───────┘
          │ 192.168.3.0/24  (separate subnet; may share the port via a switch)
   ┌──────┴───────┐
   │ igus ReBeL   │  192.168.3.11:3920 — compiled into the driver, do not change
   │ control box  │
   └──────────────┘

   rover_field.launch.py                             base_station.launch.py
```

| | rover | base station |
|---|---|---|
| launch | `ros2 launch aries_bringup rover_field.launch.py` | `ros2 launch aries_base_station base_station.launch.py` |
| static address | `192.168.1.10` | `192.168.1.11` |
| **joystick** | — | **plugged in here** |
| RViz | no | yes |
| cameras | drivers + compressors | decompressors |

Every address lives in one file — `src/aries_common/config/devices.yaml`,
`network:` section. Change it there and both machines follow.

---

## 1. The addresses

| | address | notes |
|---|---|---|
| rover PC | `192.168.1.10` | static |
| base station PC | `192.168.1.11` | static |
| rover radio | `192.168.1.20` | airOS factory address |
| base radio | `192.168.1.21` | moved off `.20` so the two radios do not collide |
| **arm control box** | **`192.168.3.11:3920`** | **separate subnet, do not change** |

### The arm is on its own subnet, and its address is not configurable

The igus ReBeL control box lives on `192.168.3.0/24`, and `192.168.3.11` is
**compiled into the driver** —
`src/aries_moveit/igus_rebel/include/igus_rebel/Rebel.hpp`. It is not read from
`devices.yaml`; the YAML copy exists so the launch files can probe the endpoint
to decide real-arm-vs-mock, and so the hardware checker can report it. A test
(`aries_common/test/test_device_table.py`) pins the two together so they cannot
drift apart silently.

Practically:

* the arm and the field link are **different `/24`s**, so they never collide;
* on the rover they may well share **one Ethernet port** through a switch, in
  which case the rover PC needs an address on **both** subnets. That is fine and
  supported — `setup_field_link.sh` preserves whatever else is already on the
  port and refuses to disable a NetworkManager profile carrying the arm subnet;
* losing the antenna does not touch the arm, and vice versa.

If `arm_hardware_protocol:=auto` cannot open `192.168.3.11:3920` it falls back
to **mock hardware** — the stack comes up looking healthy and the arm does not
move. `./scripts/setup_field_link.sh --check` tests that port explicitly for
exactly this reason.

### A known, accepted risk: this is a crowded subnet

`192.168.1.0/24` is the airOS factory subnet — Ubiquiti radios ship at
`192.168.1.20` — and the commonest consumer router subnet in the world. On a
shared field another team may be on it too.

A duplicate address is not a clean failure. ARP is last-writer-wins, so traffic
silently goes to whichever box replied most recently. The symptom is a link
that works until someone else powers on, then drops packets in a way that looks
like a radio problem and is not.

**The mitigation is operational, not structural.** Run

```bash
./scripts/setup_field_link.sh --check
```

before you drive, and again after anyone else sets up nearby. It arpings your
own address and tells you if a second machine is claiming it. If it ever fires,
change `network.hosts` in `devices.yaml` to a quieter subnet (`172.30.42.0/24`
is a good choice — the `172.16–31` range is almost untouched by consumer gear)
and re-run the setup script on both machines. Nothing else needs editing, and
the arm is unaffected either way.

### Static, not DHCP

On the field there is no DHCP server, and if there is one it belongs to someone
else. Beyond that, `aries_common/comms.py` works out *which machine it is
running on* by matching the interface address against the host table — on DHCP
it cannot, and the rover can end up announcing itself as the base station. The
fallback detection paths warn loudly rather than guessing silently.

## 2. One-time setup, per machine

### 2.1 System setup

```bash
cd ~/aries
./scripts/setup_system.sh          # udev rules, CAN sudoers, group membership
sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp \
                    ros-jazzy-image-transport-plugins \
                    iputils-arping
colcon build --symlink-install
```

`ros-jazzy-rmw-cyclonedds-cpp` and the image-transport plugins are not
optional. Without the codec plugins the compressed camera streams list as
topics and never decode — no error, just black displays.

### 2.2 Static address

```bash
./scripts/setup_field_link.sh rover     # on the rover PC
./scripts/setup_field_link.sh base      # on the base station PC
```

This creates a NetworkManager profile called `aries-field-link` on the wired
port named in `devices.yaml`, with:

* a fixed address and **no gateway** (`ipv4.never-default`) — this is a link to
  one other machine, not a route to the internet. Without it, plugging the
  antenna in would break the base station operator's normal networking;
* `autoconnect-priority 100`, so it beats any leftover DHCP profile that
  NetworkManager already has for that port;
* autoconnect disabled on any competing profile bound to the same port —
  **except** one carrying the arm's subnet, which is left alone and reported
  for you to merge by hand. Any address already on the port that is not on the
  field-link subnet is preserved, so running this cannot take the arm off the
  network.

Re-running is safe. `--dry-run rover` prints what it would do; `--check`
verifies without changing anything.

If your antenna is on a different port, fix `network.interface` in
`devices.yaml` — the script will tell you the available names.

### 2.3 Shell environment

Add to `~/.bashrc` on **every** machine — rover, base station, and any laptop
that only ever runs simulation:

```bash
if [ -f "$HOME/aries/install/setup.bash" ]; then
    source "$HOME/aries/install/setup.bash"
    _aries_dds="$(ros2 pkg prefix aries_common 2>/dev/null)/share/aries_common/aries_dds_env.sh"
    [ -f "$_aries_dds" ] && source "$_aries_dds"
    unset _aries_dds
fi
```

The same block works everywhere because the interface address is detected. On
a machine that is on the field link it pins that address; on one that is not,
it writes a **local-only** config and says so:

```
ARIES comms: domain 30, rmw_cyclonedds_cpp, LOCAL ONLY
             (not on the field link — fine for simulation; run
              scripts/setup_field_link.sh before going to the field)
```

**Delete any older hardcoded exports**, in particular:

```bash
export CYCLONEDDS_URI=file:///home/shreyas/aries/communication/cyclonedds.xml   # DELETE
```

That file named a fixed address and lived outside the workspace. Both halves of
that are fatal, and not gently: Cyclone treats a config file it cannot open,
**or an interface address the machine does not hold**, as a reason to refuse to
create the domain. Every node in the launch then dies at startup with

```
can't open configuration file file:///.../cyclonedds.xml
rmw_create_node: failed to create domain, error Error
```

which is an alarming way to discover that a path changed.

Why the shell matters at all: a process keeps the DDS environment it was
*started* with, forever. A terminal opened before these exports existed stays on
the old settings, and everything launched from it inherits them. The launch
files set their own environment, so what they start is always right; your
interactive shell is not.

---

## 3. Radio setup (Ubiquiti Rocket 5AC / R5AC-Lite)

The Rocket is a radio, not an all-in-one — it needs external antennas (a
RocketDish or a sector on each end).

Configure each radio by connecting a laptop directly to its PoE injector's LAN
port. Fresh out of the box a radio is at `192.168.1.20`; give your laptop
`192.168.1.50/24` temporarily to reach it.

| setting | value | why |
|---|---|---|
| Mode | one end **AP-PTP**, other end **Station-PTP** | point-to-point |
| airMAX | enabled, both ends | TDMA; both ends must be 5AC, no airMAX-M interop |
| Frequency | **non-DFS**, 5745–5825 MHz | on DFS, one radar detection blanks the link for 60 s mid-run |
| Channel width | **40 MHz** | 28 Mbit/s needs nothing wider; narrow is more robust |
| airSelect | **disabled** | channel hopping mid-run is not what you want |
| SSID | something unique to your team | stops a stranger's station associating |
| Security | WPA2, a real key | same |
| Management IP | `192.168.1.20` (rover) / `192.168.1.21` (base) | so the whole link is one subnet |
| **Output power** | **turn it DOWN** | see below |

### Output power is the one people get wrong

At 150 m with dish antennas you will land around **−30 dBm**, which *overdrives*
the receiver and drops your modulation rate. Target **−50 to −60 dBm** on both
ends. The link getting *worse* the closer you park is the most confusing
failure in this whole stack, and it is this.

Check it on airOS's main page (Signal Strength) or with the Align Antenna tool.

### Line of sight

At 150 m and 5.8 GHz the first Fresnel zone radius at the midpoint is **~1.4 m**;
you want about 60 % of it clear, so **~0.85 m**. A rover antenna low to the
ground looking over a berm clips it. Put the base antenna on a tripod, as high
as is practical.

### Reaching the radios

Both radios and both PCs are on `192.168.1.0/24`, so you can reach either radio's
web UI from either machine during a run — useful for watching signal strength
while someone drives.

The second radio must be moved off the factory `192.168.1.20`, or the two
collide with each other. `.21` is the convention here; it is in `devices.yaml`
under `network.radios` so the setup script can tell you whether it answers.

Note the arm sits on a **different** subnet (`192.168.3.0/24`). If the rover's
Ethernet port carries both through a switch, the rover PC needs an address on
each — see §1. `setup_field_link.sh` will not remove one to add the other.

## 4. Verify, bottom up

Stop at the first failure. Each step assumes the one above passed.

```bash
# 1. address, identity, reachability, and duplicate-address check
./scripts/setup_field_link.sh --check

# 2. this shell's environment  — must match on both machines
env | grep -E "ROS_DOMAIN_ID|RMW_IMPLEMENTATION|CYCLONEDDS_URI"

# 3. what a RUNNING process actually has (this is the one that catches it)
tr '\0' '\n' < /proc/<pid>/environ | grep -E "ROS_DOMAIN_ID|RMW_"
#    no output = unset = domain 0 + default RMW, NOT domain 30

# 4. end to end, one command on each machine
ros2 run demo_nodes_cpp talker           # rover
ros2 run demo_nodes_cpp listener         # base

# 5. the real thing
ros2 topic list | grep ^/downlink/
ros2 run aries_bringup downlink_report.py
```

`setup_field_link.sh --check` reports:

* which machine this is, by matching the interface address against the table —
  and fails if it cannot tell;
* whether the other machine and both radios answer;
* whether the **arm** answers on `192.168.3.11:3920` — ping is not enough, because
  the launch files decide real-vs-mock by opening that TCP port, so an arm that
  pings but does not accept a connection still silently falls back to mock;
* whether **anything else on the field is claiming your address** (`arping -D`).
  That last one is the check that exists because you are not alone.

Then prove the control path actually carries commands, over real DDS:

```bash
./scripts/check_control_path.py            # locally, on either machine
./scripts/check_control_path.py --remote   # on the base, with the rover up
```

Local mode spawns its own copy of the rover drive node and commands motion at
it — safe anywhere, nothing is connected to the ODrives — and checks both that
a held stick drives **and** that `/joy` going silent stops it inside
`joy_timeout_sec` (measured ~340–370 ms).

`--remote` runs on the base station against the live rover and deliberately
does **not** command motion: neutral stick, so the wheels never turn. It proves
`/joy` crosses the link, the rover's node is consuming it, and its `/cmd_vel`
comes back.

---

## 5. Running it

```bash
# ROVER
ros2 launch aries_bringup rover_field.launch.py

# BASE STATION  — joystick plugged in HERE
ros2 launch aries_base_station base_station.launch.py
```

Common variations:

```bash
# probe fingertips fitted — MUST match on both machines
ros2 launch aries_bringup rover_field.launch.py         finger_type:=probe
ros2 launch aries_base_station base_station.launch.py   finger_type:=probe

# weak link: half resolution, 10.9 Mbit/s instead of 28.3
ros2 launch aries_bringup rover_field.launch.py downlink_profile:=lean

# smoother video, if the link has room
ros2 launch aries_bringup rover_field.launch.py downlink_rate_hz:=30

# standing next to the robot with the pad in the rover's USB port
ros2 launch aries_bringup rover_field.launch.py use_joy_node:=true use_gui:=true
```

### What the joystick controls, from the base station

| hold | then | controls | on real hardware |
|---|---|---|---|
| **LB** | left stick | rover drive | ✅ |
| **LB + Y** | — | re-arm the ODrives (clear errors, CLOSED_LOOP) | ✅ |
| **RB** | sticks | arm, Cartesian | ✅ |
| **RT** | sticks | arm, joint jog | ✅ |
| **RB or RT** | **X** / **B** | gripper open / close | ✅ |
| **RB + Y** | — | hand guiding (ZeroTorque) — at the robot, not remotely | ✅ |
| **LT** | Y / A / B | planned move to a named arm preset | ✅ |
| **LT** | d-pad, left stick | drill feed, sample bin, auger | ⚠️ **simulation only** |
| Y alone | — | sound | ✅ |

**The drill is simulation-only.** `drill_joystick` runs and publishes
`/aries/drill_*/cmd_*`, but on the real rover nothing subscribes: those topics
are wired up only through the Gazebo bridge, and there is no drill firmware or
driver (`firmware/` has the gripper Teensy and the legacy controller, nothing
else). The drill mechanism is not built yet. Pressing LT + d-pad on hardware
does nothing and reports nothing — expected, not a fault. `publish_wheel_joints`
publishes the three drill joints at zero, so the drill also shows as stationary
in RViz.

Everything else in that table works from the base station unchanged.

### The two rules

1. **Exactly one machine reads the joystick.** `base_station.launch.py` defaults
   to `use_joy_node:=true`, `rover_field.launch.py` to `false`. Two joy drivers
   means two publishers on `/joy`, and the consumers see both pads interleaved
   at double rate — buttons appear to chatter and nothing is reproducible.

2. **Never point anything at a rover camera topic.** RViz's Image display has
   no transport selection — it subscribes raw, always — and the two cameras raw
   are about **737 Mbit/s**. That does not make the images slow, it collapses
   the link and everything on it goes laggy. Read `/<camera>/view/*`, which is
   local. `ros2 topic hz` and `ros2 topic bw` subscribe like any other node:
   the same rule applies to them.

### Safety: the pad is now across a radio link

Every teleop node stops after **0.35 s** of `/joy` silence — arm, drill,
presets, and both rover drive nodes. A dropout is indistinguishable from a held
stick, so this is the only thing between a lost link and a rover that keeps
driving on its last command.

Do not raise `joy_timeout_sec` to paper over a marginal link. Fix the link.

---

## 6. What crosses the antenna

```
rover  ──>  /downlink/<cam>/color/compressed        JPEG    BEST_EFFORT, depth 1
            /downlink/<cam>/depth/compressedDepth   PNG16   BEST_EFFORT, depth 1
            /downlink/<cam>/camera_info             latched
            /tf, /joint_states, /robot_description
      <──   /joy                                    ~130 kbit/s
```

| profile | resolution | both cameras |
|---|---|---|
| `quality` | 640×480 q90 | 42.3 Mbit/s |
| `balanced` *(default)* | 640×480 q75 | 28.3 Mbit/s |
| `lean` | 320×240 q90 | 10.9 Mbit/s |

All three fit comfortably in what a Rocket 5AC pair carries at 150 m. Pick on
picture quality, not bandwidth.

**Latency**, glass to glass: **~150–250 ms**, dominated by the 30 fps capture
floor (~33 ms) and the 15 Hz downlink rate gate (0–67 ms), not the radio, which
contributes 1–4 ms. Control (`/joy` → ODrive) is 10–20 ms — effectively
instant.

---

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| empty `ros2 topic list`, ping works | domain / RMW mismatch. Check `/proc/<pid>/environ`, not just your shell — a process keeps what it *started* with. |
| topics listed, no data ever arrives | QoS. The downlink is BEST_EFFORT; a RELIABLE subscriber never matches it at all, so the topic lists fine and never delivers. |
| link works, then drops when another team powers on | duplicate address. `./scripts/setup_field_link.sh --check`, and see §1. |
| arm reports healthy but does not move | `arm_hardware_protocol:=auto` fell back to mock because `192.168.3.11:3920` was closed. `./scripts/setup_field_link.sh --check` tests that port. Force it with `arm_hardware_protocol:=rebel` to make the failure loud. |
| arm stopped working after a network change | the rover PC lost its address on `192.168.3.0/24`. `ip -4 -br addr` — it needs one on the arm subnet *and* one on the field link. |
| `does not match an available interface`, or `can't open configuration file` — **every node dies at startup** | a stale `CYCLONEDDS_URI` in your shell. Cyclone treats an unopenable file or an address this machine does not hold as fatal. `unset CYCLONEDDS_URI` to unblock immediately, then fix `~/.bashrc` (§2.3). |
| `WARNING: ... this is a GUESS` at launch | this machine does not hold a configured address. Run `setup_field_link.sh {rover\|base}`. |
| `Failed to find a free participant index` | more participants than `MAX_AUTO_PARTICIPANT_INDEX` in `aries_common/comms.py`. The stack is ~30 nodes; Cyclone's own default cap is 9. |
| buttons chatter, teleop unreproducible | two joy drivers. Exactly one machine may set `use_joy_node:=true`. |
| link gets *worse* the closer you park | radio output power too high. Target −50 to −60 dBm (§3). |
| link drops for ~60 s at random | you are on a DFS channel and something looked like radar. Move to 5745–5825 (§3). |
| video fine, everything else laggy | something is subscribed to a rover raw topic. `ros2 topic bw` the suspects — from a machine on the rover, not over the link. |
| worked, then died after a cable replug | new ifindex, old sockets dead. Restart nodes on both ends. |
| one direction only | firewall. Cyclone needs UDP both ways: `sudo ufw allow from 192.168.1.0/24` |
| nodes listed that no longer exist | `ros2 daemon stop` — it caches a graph per (domain, rmw) and serves it stale. |
| `ros2` anything fails with **nothing running** | your terminal, not the robot: it holds a stale `CYCLONEDDS_URI` from before `~/.bashrc` was fixed. `source communication/stop_comms.sh` repairs the shell (or just open a new one). |
| need to stop everything | `./communication/stop_comms.sh` — SIGINT first so controllers deactivate and the drive bridge publishes a final zero. `--status` lists without killing. |

---

## Competition day checklist

**Before leaving the pit**

- [ ] `./scripts/setup_field_link.sh --check` on both machines — green, and each identifies itself correctly
- [ ] on the rover, that check shows **arm TCP 192.168.3.11:3920 open** — if it is closed the arm comes up as mock and looks healthy
- [ ] `ros2 topic list | grep ^/downlink/` shows four topics
- [ ] joystick enumerates on the **base station** (`ls /dev/input/js*`)
- [ ] `finger_type` matches the fingertips actually fitted, on both machines
- [ ] both radios at −50 to −60 dBm, non-DFS channel, fixed width

**Setting up on the field**

- [ ] base antenna on the tripod, as high as practical, aimed at the rover
- [ ] check signal on airOS before driving anywhere
- [ ] `ros2 launch aries_bringup rover_field.launch.py` on the rover
- [ ] `ros2 launch aries_base_station base_station.launch.py` on the base
- [ ] `ros2 run aries_bringup downlink_report.py` — confirm the rate and the age column
- [ ] `./scripts/check_control_path.py --remote` from the base — `/joy` reaches the rover and `/cmd_vel` comes back
- [ ] **drive-away test**: hold LB, drive 2 m, release. Then power the base radio off mid-drive and confirm the rover stops within a second.

That last one is worth doing once, every event. It is the difference between
knowing the watchdog works and assuming it does.

---

## Reference

| what | where |
|---|---|
| every address, the domain, the ports | `src/aries_common/config/devices.yaml` → `network:` |
| how the DDS config is generated | `src/aries_common/aries_common/comms.py` |
| static address setup | `scripts/setup_field_link.sh` |
| control path check | `scripts/check_control_path.py` |
| stop everything | `communication/stop_comms.sh` |
| base station package | `src/aries_base_station/README.md` |
| camera downlink internals | `src/aries_bringup/README.md` → *Camera downlink* |
| rover launch | `src/aries_bringup/launch/rover_field.launch.py` |
